#!/usr/bin/env python3
"""Evaluate local extraction and anomaly predictions against labeled data."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import DEFAULT_CONFIG
from src.data.schema import DatasetRecord, LabelOnlyRecord
from src.pipeline import analyze_document
from src.types import ModelBundle

FIELD_NAMES = ("vendor", "date", "total")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_records(data_dir: Path) -> list[DatasetRecord]:
    record_path = data_dir / DEFAULT_CONFIG.data.train_file_name
    return [DatasetRecord.model_validate(row) for row in _iter_jsonl(record_path)]


def _load_label_overrides(data_dir: Path) -> dict[str, int]:
    labels_path = data_dir / DEFAULT_CONFIG.data.labels_file_name
    if not labels_path.exists():
        return {}

    labels: dict[str, int] = {}
    for row in _iter_jsonl(labels_path):
        record = LabelOnlyRecord.model_validate(row)
        labels[record.id] = record.label.is_forged
    return labels


def _load_model_bundle(model_dir: Path) -> ModelBundle:
    with (model_dir / DEFAULT_CONFIG.data.stats_file_name).open() as handle:
        stats = json.load(handle)
    with (model_dir / DEFAULT_CONFIG.data.anomaly_model_file_name).open("rb") as handle:
        anomaly_model_data = pickle.load(handle)
    return ModelBundle(stats=stats, anomaly_model_data=anomaly_model_data)


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    collapsed = " ".join(value.strip().casefold().split())
    if not collapsed:
        return None
    normalized = re.sub(r"[^0-9a-z]+", "", collapsed)
    return normalized or None


def _format_ratio(correct: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{correct / total:.4f} ({correct}/{total})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    bundle = _load_model_bundle(Path(args.model_dir).resolve()) if args.model_dir else None

    records = _load_records(data_dir)
    label_overrides = _load_label_overrides(data_dir)

    exact_correct = {field: 0 for field in FIELD_NAMES}
    normalized_correct = {field: 0 for field in FIELD_NAMES}
    total_records = len(records)
    anomaly_true: list[int] = []
    anomaly_pred: list[int] = []

    for record in records:
        image_path = data_dir / record.image_path
        analysis = analyze_document(str(image_path), model_bundle=bundle, debug=False)
        predicted_fields = analysis.extraction.as_fields()
        expected_fields = record.fields.model_dump()

        for field in FIELD_NAMES:
            expected_value = expected_fields.get(field)
            predicted_value = predicted_fields.get(field)
            if predicted_value == expected_value:
                exact_correct[field] += 1
            if _normalize(predicted_value) == _normalize(expected_value):
                normalized_correct[field] += 1

        if record.label is not None or record.id in label_overrides:
            anomaly_true.append(label_overrides.get(record.id, record.label.is_forged if record.label else 0))
            anomaly_pred.append(int(analysis.anomaly.is_forged))

    print(f"records={total_records}")
    print("extraction")
    for field in FIELD_NAMES:
        print(f"{field}_exact_match={_format_ratio(exact_correct[field], total_records)}")
        print(f"{field}_normalized_match={_format_ratio(normalized_correct[field], total_records)}")

    if anomaly_true:
        print("anomaly")
        print(f"accuracy={accuracy_score(anomaly_true, anomaly_pred):.4f}")
        print(f"precision={precision_score(anomaly_true, anomaly_pred, zero_division=0):.4f}")
        print(f"recall={recall_score(anomaly_true, anomaly_pred, zero_division=0):.4f}")
        print(f"f1={f1_score(anomaly_true, anomaly_pred, zero_division=0):.4f}")
        print(f"labeled_records={len(anomaly_true)}")
    else:
        print("anomaly")
        print("accuracy=n/a")
        print("precision=n/a")
        print("recall=n/a")
        print("f1=n/a")
        print("labeled_records=0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
