from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.anomaly import (
    extract_text_features,
    predict_anomaly,
    train_anomaly_model,
)
from src.config import DEFAULT_CONFIG


class AnomalyTests(unittest.TestCase):
    def test_extract_text_features_handles_empty_text(self) -> None:
        features = extract_text_features("")
        self.assertEqual(features["text_length"], 0.0)
        self.assertEqual(features["line_count"], 0.0)
        self.assertEqual(features["empty_line_ratio"], 0.0)

    def test_predict_anomaly_heuristic_flags_risky_sample(self) -> None:
        model_data = {"model": None, "forged_ratio": 0.5}
        stats = {"amount_mean": 50.0, "amount_std": 5.0, "vendors": ["Acme Mart"]}

        with patch(
            "src.anomaly.extract_image_features",
            return_value={
                "img_std": 10.0,
                "noise_std": 30.0,
                "ela_high_ratio": 0.2,
                "ela_block_std": 10.0,
                "block_var_mean": 20.0,
                "block_var_std": 1.0,
                "entropy": 2.0,
            },
        ):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_dir = Path(tmp_dir) / "train"
            model_dir_a = Path(tmp_dir) / "model-a"
            model_dir_b = Path(tmp_dir) / "model-b"
            image_dir = train_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            for record in records:
                Image.new("L", (48, 48), color=100).save(train_dir / record["image_path"])

            ocr_payloads = {
                record["image_path"]: f"{record['fields']['vendor']}\nTOTAL {record['fields']['total']}"
                for record in records
            }

            def fake_extract_text(image_path: str) -> str:
                return ocr_payloads[Path(image_path).relative_to(train_dir).as_posix()]

            with patch("src.extractor.extract_text", side_effect=fake_extract_text):
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

            self.assertEqual(data_a["model_type"], "GradientBoostingClassifier")
            self.assertEqual(data_a["feature_keys"], data_b["feature_keys"])
            self.assertEqual(data_a["forged_ratio"], data_b["forged_ratio"])
            self.assertEqual(data_a["train_accuracy"], data_b["train_accuracy"])


if __name__ == "__main__":
    unittest.main()
