"""Dataset adapters for challenge JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import DEFAULT_CONFIG
from src.data.schema import (
    DatasetRecord,
    LabelOnlyRecord,
    PredictionRecord,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_dataset_records(
    data_dir: str | Path,
    split: str,
) -> list[DatasetRecord]:
    """Load and validate train/test records for a dataset split."""
    base_dir = Path(data_dir)
    file_name = (
        DEFAULT_CONFIG.data.train_file_name
        if split == "train"
        else DEFAULT_CONFIG.data.test_file_name
    )
    path = base_dir / file_name
    return [DatasetRecord.model_validate(record) for record in _load_jsonl(path)]


def load_label_records(data_dir: str | Path) -> list[LabelOnlyRecord]:
    """Load and validate labels.jsonl records."""
    path = Path(data_dir) / DEFAULT_CONFIG.data.labels_file_name
    return [LabelOnlyRecord.model_validate(record) for record in _load_jsonl(path)]


def save_predictions(
    predictions: list[PredictionRecord],
    out_path: str | Path,
) -> None:
    """Write validated predictions to JSONL."""
    path = Path(out_path)
    with path.open("w") as handle:
        for prediction in predictions:
            handle.write(prediction.model_dump_json())
            handle.write("\n")


def write_json_file(payload: dict[str, Any], out_path: str | Path) -> None:
    """Persist a JSON payload with stable formatting."""
    path = Path(out_path)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
