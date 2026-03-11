"""Harness-facing DocFusion solution with deterministic, batched inference."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.anomaly import (
    FEATURE_KEYS,
    TEXT_FEATURE_KEYS,
    extract_image_features,
    extract_text_features,
    features_to_vector,
    text_features_to_vector,
    train_anomaly_model,
)
from src.config import DEFAULT_CONFIG
from src.consistency import CONSISTENCY_FEATURE_KEYS, extract_consistency_features
from src.data.schema import PredictionRecord
from src.extractor import extract_fields


def _set_deterministic_seeds(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


class DocFusionSolution:
    """Submission entry point used by the challenge harness."""

    _BATCH_SIZE = 8

    def train(self, train_dir: str, work_dir: str) -> str:
        _set_deterministic_seeds(DEFAULT_CONFIG.training.random_state)

        train_path = Path(train_dir)
        model_dir = Path(work_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        raw_records: list[dict[str, Any]] = []
        vendors: set[str] = set()
        amounts: list[float] = []

        for record in self._iter_jsonl(train_path / DEFAULT_CONFIG.data.train_file_name):
            raw_records.append(record)

            vendor = self._clean_text(record.get("fields", {}).get("vendor"))
            if vendor is not None:
                vendors.add(vendor)

            amount = self._parse_amount(record.get("fields", {}).get("total"))
            if amount is not None:
                amounts.append(amount)

        stats = self._build_stats(vendors=vendors, amounts=amounts, total_records=len(raw_records))
        self._write_json(stats, model_dir / DEFAULT_CONFIG.data.stats_file_name)

        print(f"[train] Training anomaly model on {len(raw_records)} records...")
        train_anomaly_model(raw_records, str(train_path), str(model_dir))

        print(
            f"[train] {len(vendors)} unique vendors, amount mean=${stats['amount_mean']:.2f}"
        )
        print(f"[train] Model saved to: {model_dir}")
        return str(model_dir)

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        _set_deterministic_seeds(DEFAULT_CONFIG.training.random_state)

        model_path = Path(model_dir)
        data_path = Path(data_dir)
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with (model_path / DEFAULT_CONFIG.data.stats_file_name).open() as handle:
            stats = json.load(handle)

        with (model_path / DEFAULT_CONFIG.data.anomaly_model_file_name).open("rb") as handle:
            model_data = pickle.load(handle)

        total_predictions = 0
        record_iter = self._iter_jsonl(data_path / DEFAULT_CONFIG.data.test_file_name)

        with out_file.open("w") as handle:
            for batch in self._batched(record_iter, self._BATCH_SIZE):
                for prediction in self._predict_batch(
                    records=batch,
                    data_dir=data_path,
                    model_data=model_data,
                    stats=stats,
                ):
                    handle.write(prediction.model_dump_json())
                    handle.write("\n")
                    total_predictions += 1

        print(f"[predict] Wrote {total_predictions} predictions to {out_file}")

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _batched(
        items: Iterable[dict[str, Any]],
        batch_size: int,
    ) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for item in items:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _predict_batch(
        self,
        records: list[dict[str, Any]],
        data_dir: Path,
        model_data: dict[str, Any],
        stats: dict[str, Any],
    ) -> list[PredictionRecord]:
        prepared: list[dict[str, Any]] = []

        for record in records:
            fields = record.get("fields", {})
            image_path = data_dir / str(record.get("image_path", ""))
            extracted = self._safe_extract_fields(image_path)
            amount = self._parse_amount(fields.get("total")) or self._parse_amount(extracted.get("total"))
            prepared.append(
                {
                    "id": str(record.get("id", "")),
                    "vendor": self._clean_text(fields.get("vendor")) or extracted.get("vendor"),
                    "date": self._clean_text(fields.get("date")) or extracted.get("date"),
                    "total": self._clean_text(fields.get("total")) or extracted.get("total"),
                    "amount": amount or 0.0,
                    "ocr_text": extracted.get("_ocr_text", ""),
                    "image_path": str(image_path),
                }
            )

        forged_flags = self._predict_forgery_flags(prepared, model_data=model_data, stats=stats)

        return [
            PredictionRecord(
                id=item["id"],
                vendor=item["vendor"],
                date=item["date"],
                total=item["total"],
                is_forged=is_forged,
            )
            for item, is_forged in zip(prepared, forged_flags, strict=True)
        ]

    def _predict_forgery_flags(
        self,
        records: list[dict[str, Any]],
        model_data: dict[str, Any],
        stats: dict[str, Any],
    ) -> list[int]:
        model = model_data.get("model")
        feature_vectors: list[list[float]] = []
        fallbacks: list[tuple[dict[str, Any], dict[str, float], dict[str, float], dict[str, float]]] = []

        for record in records:
            image_features = extract_image_features(record["image_path"])
            text_features = extract_text_features(record["ocr_text"])
            consistency_features = extract_consistency_features(
                record["ocr_text"],
                float(record["amount"]),
                stats,
            )
            fallbacks.append((record, image_features, text_features, consistency_features))
            feature_vectors.append(
                features_to_vector(image_features)
                + text_features_to_vector(text_features)
                + [float(consistency_features.get(key, 0.0)) for key in CONSISTENCY_FEATURE_KEYS]
                + [float(record["amount"])]
            )

        if model is not None and feature_vectors:
            probabilities = model.predict_proba(np.asarray(feature_vectors, dtype=np.float64))
            threshold = DEFAULT_CONFIG.training.anomaly_threshold
            return [
                int((float(proba[1]) if len(proba) > 1 else 0.0) >= threshold)
                for proba in probabilities
            ]

        forged_ratio = float(model_data.get("forged_ratio", 0.5))
        return [
            self._heuristic_forgery(
                amount=float(record["amount"]),
                stats=stats,
                image_features=image_features,
                text_features=text_features,
                consistency_features=consistency_features,
                forged_ratio=forged_ratio,
            )
            for record, image_features, text_features, consistency_features in fallbacks
        ]

    @staticmethod
    def _heuristic_forgery(
        amount: float,
        stats: dict[str, Any],
        image_features: dict[str, float],
        text_features: dict[str, float],
        consistency_features: dict[str, float],
        forged_ratio: float,
    ) -> int:
        score = 0.0

        if image_features.get("img_std", 0.0) > 1.0:
            if image_features.get("noise_std", 0.0) > 20.0:
                score += 0.2
            if image_features.get("ela_high_ratio", 0.0) > 0.12:
                score += 0.25
            if image_features.get("ela_block_std", 0.0) > 8.0:
                score += 0.15
            block_var_mean = max(image_features.get("block_var_mean", 0.0), 1.0)
            if image_features.get("block_var_std", 0.0) / block_var_mean < 0.3:
                score += 0.15
            entropy = image_features.get("entropy", 0.0)
            if entropy and (entropy < 3.0 or entropy > 7.0):
                score += 0.1

        if text_features.get("text_length", 0.0) == 0:
            score += 0.1

        score += 0.35 * float(consistency_features.get("consistency_risk", 0.0))

        amount_std = float(stats.get("amount_std", 0.0) or 0.0)
        amount_mean = float(stats.get("amount_mean", 0.0) or 0.0)
        if amount > 0 and amount_std > 1e-6:
            z_score = abs(amount - amount_mean) / amount_std
            if z_score > 2.0:
                score += 0.2

        threshold = max(0.45, min(0.7, 0.4 + forged_ratio * 0.2))
        return int(score >= threshold)

    @staticmethod
    def _safe_extract_fields(image_path: Path) -> dict[str, str | None]:
        if not image_path.exists():
            return {"vendor": None, "date": None, "total": None, "_ocr_text": ""}
        try:
            result = extract_fields(str(image_path))
        except Exception:
            return {"vendor": None, "date": None, "total": None, "_ocr_text": ""}
        return {
            "vendor": DocFusionSolution._clean_text(result.get("vendor")),
            "date": DocFusionSolution._clean_text(result.get("date")),
            "total": DocFusionSolution._clean_text(result.get("total")),
            "_ocr_text": str(result.get("_ocr_text", "") or ""),
        }

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return str(value).strip() or None

    @staticmethod
    def _parse_amount(value: Any) -> float | None:
        text = DocFusionSolution._clean_text(value)
        if text is None:
            return None
        normalized = text.replace(",", "")
        try:
            return float(normalized)
        except ValueError:
            return None

    @staticmethod
    def _build_stats(
        vendors: set[str],
        amounts: list[float],
        total_records: int,
    ) -> dict[str, Any]:
        amount_values = np.asarray(amounts, dtype=np.float64) if amounts else np.asarray([], dtype=np.float64)
        return {
            "vendors": sorted(vendors),
            "amount_mean": float(amount_values.mean()) if amount_values.size else 0.0,
            "amount_std": float(amount_values.std()) if amount_values.size else 1.0,
            "amount_q1": float(np.percentile(amount_values, 25)) if amount_values.size else 0.0,
            "amount_q3": float(np.percentile(amount_values, 75)) if amount_values.size else 0.0,
            "total_records": total_records,
            "seed": DEFAULT_CONFIG.training.random_state,
            "batch_size": DocFusionSolution._BATCH_SIZE,
            "feature_keys": FEATURE_KEYS + TEXT_FEATURE_KEYS + CONSISTENCY_FEATURE_KEYS + ["amount"],
        }

    @staticmethod
    def _write_json(payload: dict[str, Any], out_path: Path) -> None:
        with out_path.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
