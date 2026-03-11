from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.anomaly import extract_text_features, predict_anomaly, train_anomaly_model
from src.config import DEFAULT_CONFIG
from src.types import AnalysisResult, AnomalyResult, Box, ExtractionResult, FieldPrediction


class AnomalyTests(unittest.TestCase):
    def test_extract_text_features_handles_empty_text(self) -> None:
        features = extract_text_features("")
        self.assertEqual(features["text_length"], 0.0)
        self.assertEqual(features["line_count"], 0.0)
        self.assertEqual(features["empty_line_ratio"], 0.0)

    def test_predict_anomaly_heuristic_flags_risky_sample(self) -> None:
        model_data = {"model": None, "forged_ratio": 0.5}
        stats = {"amount_mean": 50.0, "amount_std": 5.0, "vendors": ["Acme Mart"]}

        # Patch heuristic_score directly to return a high-risk score (above threshold)
        with patch("src.anomaly.heuristic_score", return_value=(0.75, ("Elevated ELA artifact ratio", "High residual noise"))):
            prediction = predict_anomaly(
                model_data=model_data,
                img_path="unused.png",
                ocr_text="Unknown Vendor\nTOTAL 120.00",
                amount=120.0,
                stats=stats,
            )

        self.assertEqual(prediction, 1)

    def test_train_anomaly_model_writes_deterministic_artifact(self) -> None:
        records = [
            {
                "id": f"r{i}",
                "image_path": f"images/r{i}.png",
                "fields": {
                    "vendor": "ACME Corp" if i % 2 == 0 else "Budget Foods",
                    "total": f"{100 + i * 7:.2f}",
                },
                "label": {"is_forged": i % 2},
            }
            for i in range(1, 7)
        ]

        def fake_analysis(document_path: str, model_bundle=None, debug: bool = False) -> AnalysisResult:
            stem = Path(document_path).stem
            index = int(stem[1:])
            total = f"{100 + index * 7:.2f}"
            return AnalysisResult(
                document_path=document_path,
                ocr_text=f"Vendor {index}\nTOTAL {total}",
                words=(),
                lines=(),
                extraction=ExtractionResult(
                    vendor=FieldPrediction("vendor", f"Vendor {index}", 0.9, Box(0, 0, 10, 10, 0)),
                    date=FieldPrediction("date", "2024-01-01", 0.8, Box(0, 10, 10, 20, 0)),
                    total=FieldPrediction("total", total, 0.95, Box(0, 20, 10, 30, 0)),
                ),
                anomaly=AnomalyResult(0.0, 0),
                page_count=1,
                page_sizes=((48, 48),),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_dir = Path(tmp_dir) / "train"
            model_dir_a = Path(tmp_dir) / "model-a"
            model_dir_b = Path(tmp_dir) / "model-b"
            image_dir = train_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            for record in records:
                Image.new("L", (48, 48), color=100).save(train_dir / record["image_path"])

            with patch("src.pipeline.analyze_document", side_effect=fake_analysis):
                model_a = train_anomaly_model(records, str(train_dir), str(model_dir_a))
                model_b = train_anomaly_model(records, str(train_dir), str(model_dir_b))

            self.assertIsNotNone(model_a)
            self.assertIsNotNone(model_b)

            artifact_a = model_dir_a / DEFAULT_CONFIG.data.anomaly_model_file_name
            artifact_b = model_dir_b / DEFAULT_CONFIG.data.anomaly_model_file_name
            self.assertTrue(artifact_a.exists())
            self.assertTrue(artifact_b.exists())

            with artifact_a.open("rb") as handle:
                data_a = pickle.load(handle)
            with artifact_b.open("rb") as handle:
                data_b = pickle.load(handle)

            self.assertEqual(data_a["model_type"], data_b["model_type"])
            self.assertEqual(data_a["feature_keys"], data_b["feature_keys"])
            self.assertEqual(data_a["forged_ratio"], data_b["forged_ratio"])
            self.assertEqual(data_a["train_accuracy"], data_b["train_accuracy"])


if __name__ == "__main__":
    unittest.main()
