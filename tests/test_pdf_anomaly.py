from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.ocr import OCRResult
from src.pipeline import analyze_document
from src.types import ExtractionResult, FieldPrediction


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


if __name__ == "__main__":
    unittest.main()
