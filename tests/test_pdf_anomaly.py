from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.ocr import OCRResult
from src.pipeline import analyze_document
from src.types import Box, ExtractionResult, FieldPrediction, OCRLine, OCRWord


class PdfAnomalyTests(unittest.TestCase):
    def test_analyze_document_pdf_uses_rasterized_page_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "synthetic.pdf"
            page = Image.new("RGB", (180, 180), color="white")
            draw = ImageDraw.Draw(page)
            draw.rectangle((20, 20, 160, 80), fill="black")
            draw.text((30, 110), "TOTAL 19.99", fill="black")
            page.save(pdf_path, "PDF")

            fake_ocr = OCRResult(text="Store\nTOTAL 19.99", words=(), image_size=page.size, lines=())
            fake_extraction = ExtractionResult(
                vendor=FieldPrediction("vendor", "Store", 0.9),
                date=FieldPrediction("date", "2024-01-01", 0.8),
                total=FieldPrediction("total", "19.99", 0.95),
            )

            with patch("src.pipeline.run_ocr", return_value=fake_ocr), patch(
                "src.pipeline.extract_fields_from_ocr",
                return_value=fake_extraction,
            ):
                analysis = analyze_document(str(pdf_path), model_bundle=None, debug=True)

        self.assertGreater(analysis.anomaly.feature_values["img_mean"], 0.0)
        self.assertGreater(analysis.anomaly.feature_values["ela_mean"], 0.0)

    def test_pdf_suspicious_regions_preserve_page_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "multipage.pdf"
            first_page = Image.new("RGB", (180, 180), color="white")
            second_page = Image.new("RGB", (180, 180), color="white")
            draw = ImageDraw.Draw(second_page)
            draw.rectangle((30, 30, 150, 90), fill="black")
            first_page.save(pdf_path, "PDF", save_all=True, append_images=[second_page])

            page_zero_extraction = ExtractionResult(
                vendor=FieldPrediction("vendor", "Store", 0.9),
                date=FieldPrediction("date", "2024-01-01", 0.8),
                total=FieldPrediction("total", None, 0.0),
            )
            page_one_extraction = ExtractionResult(
                vendor=FieldPrediction("vendor", None, 0.0),
                date=FieldPrediction("date", None, 0.0),
                total=FieldPrediction("total", "19.99", 0.2),
            )

            with patch("src.pipeline.run_ocr") as mock_run_ocr, patch(
                "src.pipeline.extract_fields_from_ocr",
                side_effect=(page_zero_extraction, page_one_extraction),
            ):
                def run_ocr_side_effect(image, tesseract_config=None, page_index=0):
                    if page_index == 0:
                        return OCRResult(
                            text="Store",
                            words=(),
                            image_size=image.size,
                            lines=(),
                        )

                    words = (
                        OCRWord(
                            "19.99",
                            70,
                            120,
                            40,
                            10,
                            confidence=20.0,
                            line_num=1,
                            block_num=1,
                            paragraph_num=1,
                            page_index=1,
                        ),
                    )
                    lines = (
                        OCRLine(
                            "TOTAL 19.99",
                            words,
                            Box(20, 110, 140, 132, 1),
                            20.0,
                            1,
                        ),
                    )
                    return OCRResult(
                        text="TOTAL 19.99",
                        words=words,
                        image_size=image.size,
                        lines=lines,
                    )

                mock_run_ocr.side_effect = run_ocr_side_effect
                analysis = analyze_document(str(pdf_path), model_bundle=None, debug=False)

        self.assertTrue(analysis.anomaly.suspicious_regions)
        self.assertIn(1, {box.page_index for box in analysis.anomaly.suspicious_regions})
        self.assertTrue(all(box.page_index in {0, 1} for box in analysis.anomaly.suspicious_regions))


if __name__ == "__main__":
    unittest.main()
