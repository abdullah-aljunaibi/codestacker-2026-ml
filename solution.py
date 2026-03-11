"""Harness-facing DocFusion solution built on the shared analysis pipeline."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.anomaly import train_anomaly_model
from src.config import DEFAULT_CONFIG
from src.data.schema import PredictionRecord
from src.pipeline import analyze_document
from src.reproducibility import set_deterministic_seeds
from src.types import ModelBundle


class DocFusionSolution:
    """Submission entry point used by the challenge harness."""

    def train(self, train_dir: str, work_dir: str) -> str:
        set_deterministic_seeds(DEFAULT_CONFIG.training.random_state)

        train_path = Path(train_dir)
        model_dir = Path(work_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        records = list(self._iter_jsonl(train_path / DEFAULT_CONFIG.data.train_file_name))
        model_data = train_anomaly_model(records, str(train_path), str(model_dir))
        stats = dict(model_data.get("stats", {}))
        with (model_dir / DEFAULT_CONFIG.data.stats_file_name).open("w") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
        return str(model_dir)

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        set_deterministic_seeds(DEFAULT_CONFIG.training.random_state)

        model_path = Path(model_dir)
        data_path = Path(data_dir)
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with (model_path / DEFAULT_CONFIG.data.stats_file_name).open() as handle:
            stats = json.load(handle)
        with (model_path / DEFAULT_CONFIG.data.anomaly_model_file_name).open("rb") as handle:
            anomaly_model_data = pickle.load(handle)
        bundle = ModelBundle(stats=stats, anomaly_model_data=anomaly_model_data)

        with out_file.open("w") as handle:
            for record in self._iter_jsonl(data_path / DEFAULT_CONFIG.data.test_file_name):
                image_path = data_path / str(record.get("image_path", ""))
                analysis = analyze_document(str(image_path), model_bundle=bundle, debug=False)
                prediction = PredictionRecord(
                    id=str(record.get("id", "")),
                    vendor=analysis.extraction.vendor.value,
                    date=analysis.extraction.date.value,
                    total=analysis.extraction.total.value,
                    is_forged=analysis.anomaly.is_forged,
                )
                handle.write(prediction.model_dump_json())
                handle.write("\n")

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
