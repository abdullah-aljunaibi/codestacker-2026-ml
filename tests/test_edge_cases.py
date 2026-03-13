from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import app
from src.anomaly import build_feature_vector, heuristic_score
from src.document_io import DocumentPage
from src.extractor import AMOUNT_REGEX, extract_date, extract_total, extract_vendor, extract_fields_from_ocr
from src.ocr import OCRResult
from src.pipeline import analyze_document
from src.summary import generate_anomaly_summary
from src.types import AnalysisResult, AnomalyResult, Box, ExtractionResult, FieldPrediction


class _ContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _field(name: str, value: str | None, confidence: float = 0.0, box: Box | None = None) -> FieldPrediction:
    return FieldPrediction(name=name, value=value, confidence=confidence, box=box)


def _analysis_result(
    *,
    vendor: str | None = "Store",
    date: str | None = "2024-01-24",
    total: str | None = "19.99",
    ocr_text: str = "Store\nDate: 2024-01-24\nTOTAL 19.99",
    page_count: int = 1,
) -> AnalysisResult:
    page_image = Image.new("RGB", (24, 24), color="white")
    return AnalysisResult(
        document_path="dummy.png",
        ocr_text=ocr_text,
        words=(),
        lines=(),
        extraction=ExtractionResult(
            vendor=_field("vendor", vendor, 0.8, Box(0, 0, 8, 8, 0) if vendor else None),
            date=_field("date", date, 0.7, Box(0, 8, 8, 16, 0) if date else None),
            total=_field("total", total, 0.9, Box(0, 16, 8, 24, 0) if total else None),
        ),
        anomaly=AnomalyResult(score=0.2, is_forged=0, feature_values={"field_presence": 1.0}),
        page_count=page_count,
        page_sizes=tuple((24, 24) for _ in range(page_count)),
        page_images=tuple(Image.new("RGB", (24, 24), color="white") for _ in range(page_count)),
    )


class EdgeCaseTests(unittest.TestCase):
    def test_analyze_document_returns_graceful_empty_result_for_invalid_looking_path(self) -> None:
        dummy_page = DocumentPage(image=Image.new("RGB", (16, 16), color="white"), page_index=0, source_path="ignored")
        empty_ocr = OCRResult(text="", words=(), image_size=(16, 16), lines=())

        with (
            patch("src.pipeline.load_document_pages", return_value=[dummy_page]),
            patch("src.pipeline.run_ocr", return_value=empty_ocr),
            patch("src.pipeline.build_feature_vector", return_value=({}, [])),
            patch("src.pipeline.heuristic_score", return_value=(0.0, ())),
            patch("src.pipeline.localize_suspicious_regions", return_value=()),
        ):
            analysis = analyze_document("corrupt://receipt.png", debug=False)

        self.assertEqual(analysis.document_path, "corrupt://receipt.png")
        self.assertEqual(analysis.ocr_text, "")
        self.assertEqual(analysis.page_count, 1)
        self.assertIsNone(analysis.extraction.vendor.value)
        self.assertIsNone(analysis.extraction.date.value)
        self.assertIsNone(analysis.extraction.total.value)
        self.assertEqual(analysis.anomaly.is_forged, 0)

    def test_extract_fields_from_empty_ocr_text_returns_empty_fields(self) -> None:
        result = extract_fields_from_ocr(OCRResult(text="", words=(), image_size=(10, 10), lines=()))

        self.assertIsNone(result.vendor.value)
        self.assertIsNone(result.date.value)
        self.assertIsNone(result.total.value)
        self.assertEqual(result.vendor.confidence, 0.0)
        self.assertEqual(result.date.confidence, 0.0)
        self.assertEqual(result.total.confidence, 0.0)

    def test_anomaly_features_handle_missing_extraction_fields(self) -> None:
        analysis = _analysis_result(vendor=None, date=None, total="19.99", ocr_text="TOTAL 19.99")

        feature_values, vector = build_feature_vector(analysis, stats={})
        score, reasons = heuristic_score(feature_values)

        self.assertAlmostEqual(feature_values["field_presence"], 1.0 / 3.0)
        self.assertGreater(len(vector), 0)
        self.assertIsInstance(score, float)
        self.assertIn("Missing key fields", reasons)

    def test_extract_total_handles_very_large_amounts(self) -> None:
        self.assertEqual(extract_total("TOTAL 999999.99"), "999999.99")

    def test_extract_date_returns_normalized_value_for_ambiguous_format(self) -> None:
        self.assertEqual(extract_date("Receipt Date: 01/02/2024"), "2024-02-01")

    def test_generate_anomaly_summary_falls_back_without_llm(self) -> None:
        analysis = _analysis_result(vendor=None, date=None, total="19.99", ocr_text="TOTAL 19.99")

        with patch("src.summary._generate_llm_summary", return_value=None):
            summary = generate_anomaly_summary(analysis)

        self.assertIsInstance(summary, str)
        self.assertTrue(summary.strip())

    def test_single_page_ui_does_not_render_page_selector(self) -> None:
        analysis = _analysis_result(page_count=1)
        pages = [DocumentPage(image=Image.new("RGB", (24, 24), color="white"), page_index=0, source_path="doc.png")]
        selectbox = MagicMock(return_value=0)

        with (
            patch("app.analyze_document", return_value=analysis),
            patch("app.load_document_pages", return_value=pages),
            patch("app.generate_anomaly_summary_with_method", return_value=("Rule-based summary", "nlg")),
            patch.object(app.st, "selectbox", selectbox),
            patch.object(app.st, "subheader"),
            patch.object(app.st, "columns", return_value=(_ContextManager(), _ContextManager())),
            patch.object(app.st, "tabs", return_value=[_ContextManager(), _ContextManager(), _ContextManager()]),
            patch.object(app.st, "image"),
            patch.object(app.st, "markdown"),
            patch.object(app.st, "json"),
            patch.object(app.st, "write"),
            patch.object(app.st, "caption"),
            patch.object(app.st, "expander", return_value=_ContextManager()),
            patch.object(app.st, "code"),
        ):
            app.render_analysis("receipt", "doc.png", None)

        selectbox.assert_not_called()

    def test_amount_detection_rejects_phone_numbers(self) -> None:
        self.assertIsNone(extract_total("Tel: +968 9999 1234"))

    def test_amount_regex_rejects_isolated_single_digits(self) -> None:
        self.assertFalse(AMOUNT_REGEX.search("Qty 7"))

    def test_vendor_extraction_skips_all_digit_lines(self) -> None:
        text = "123456789\nBright Market\nTOTAL 12.30"
        self.assertEqual(extract_vendor(text), "Bright Market")


if __name__ == "__main__":
    unittest.main()
