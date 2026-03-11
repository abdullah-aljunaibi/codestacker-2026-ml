from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from solution import DocFusionSolution
from src.data.schema import PredictionRecord
from src.document_io import load_document_pages
from src.pipeline import analyze_document
from src.types import AnalysisResult, AnomalyResult, Box, ExtractionResult, FieldPrediction


class PipelineIntegrationTests(unittest.TestCase):
    def test_pdf_loading_rasterizes_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "doc.pdf"
            images = [Image.new("RGB", (32, 32), color="white"), Image.new("RGB", (32, 32), color="gray")]
            images[0].save(pdf_path, save_all=True, append_images=images[1:])
            pages = load_document_pages(pdf_path)
            self.assertEqual(len(pages), 2)

    def test_predict_does_not_require_fields_key(self) -> None:
        def fake_analysis(document_path: str, model_bundle=None, debug: bool = False) -> AnalysisResult:
            return AnalysisResult(
                document_path=document_path,
                ocr_text="Store\nTOTAL 9.99",
                words=(),
                lines=(),
                extraction=ExtractionResult(
                    vendor=FieldPrediction("vendor", "Store", 0.9, Box(0, 0, 10, 10, 0)),
                    date=FieldPrediction("date", "2024-01-01", 0.8, Box(0, 10, 10, 20, 0)),
                    total=FieldPrediction("total", "9.99", 0.95, Box(0, 20, 10, 30, 0)),
                ),
                anomaly=AnomalyResult(0.2, 0),
                page_count=1,
                page_sizes=((32, 32),),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "test"
            model_dir = Path(tmp_dir) / "model"
            data_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            image_dir = data_dir / "images"
            image_dir.mkdir()
            Image.new("RGB", (32, 32), color="white").save(image_dir / "a.png")
            (data_dir / "test.jsonl").write_text(json.dumps({"id": "a", "image_path": "images/a.png"}) + "\n")
            (model_dir / "stats.json").write_text(json.dumps({}))
            with (model_dir / "anomaly_model.pkl").open("wb") as handle:
                pickle.dump({"model": None, "threshold": 0.45}, handle)

            out_path = Path(tmp_dir) / "predictions.jsonl"
            with patch("solution.analyze_document", side_effect=fake_analysis):
                DocFusionSolution().predict(str(model_dir), str(data_dir), str(out_path))

            rows = [PredictionRecord.model_validate_json(line) for line in out_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].vendor, "Store")

    def test_shared_pipeline_surface(self) -> None:
        self.assertTrue(callable(analyze_document))


if __name__ == "__main__":
    unittest.main()
