#!/usr/bin/env python3
"""Measure local latency and memory for the shared analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import pickle
import statistics
import time
import tracemalloc

from solution import DocFusionSolution
from src.config import DEFAULT_CONFIG
from src.data.schema import PredictionRecord
from src.pipeline import analyze_document
from src.reproducibility import set_deterministic_seeds
from src.types import ModelBundle


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("at least one value is required")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value

    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"model directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"model path is not a directory: {path}")

    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            total_bytes += child.stat().st_size
    return total_bytes


def _load_model_bundle(model_dir: Path) -> ModelBundle:
    with (model_dir / DEFAULT_CONFIG.data.stats_file_name).open() as handle:
        stats = json.load(handle)
    with (model_dir / DEFAULT_CONFIG.data.anomaly_model_file_name).open("rb") as handle:
        anomaly_model_data = pickle.load(handle)
    return ModelBundle(stats=stats, anomaly_model_data=anomaly_model_data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--model-dir")
    parser.add_argument("--harness", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--out-path")
    args = parser.parse_args()

    if args.harness:
        if not args.data_dir or not args.out_path:
            parser.error("--harness requires --data-dir and --out-path")
    elif not args.inputs:
        parser.error("--inputs is required unless --harness is set")

    if args.harness:
        test_path = Path(args.data_dir) / DEFAULT_CONFIG.data.test_file_name
        document_count = 0
        if test_path.exists():
            with test_path.open() as handle:
                document_count = sum(1 for line in handle if line.strip())
        if document_count == 0:
            print(
                f"warning: 0 documents found in {test_path}",
                file=sys.stderr,
            )

        set_deterministic_seeds(DEFAULT_CONFIG.training.random_state)
        model_dir = Path(args.model_dir) if args.model_dir else Path(".")
        data_path = Path(args.data_dir)
        out_file = Path(args.out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        bundle = _load_model_bundle(model_dir)

        durations = []
        peaks_mb = []
        started = time.perf_counter()
        try:
            with out_file.open("w") as handle:
                for record in DocFusionSolution._iter_jsonl(data_path / DEFAULT_CONFIG.data.test_file_name):
                    image_path = data_path / str(record.get("image_path", ""))
                    tracemalloc.start()
                    try:
                        document_started = time.perf_counter()
                        analysis = analyze_document(str(image_path), model_bundle=bundle, debug=False)
                        durations.append(time.perf_counter() - document_started)
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()

                    peaks_mb.append(peak / (1024 * 1024))
                    prediction = PredictionRecord(
                        id=str(record.get("id", "")),
                        vendor=analysis.extraction.vendor.value,
                        date=analysis.extraction.date.value,
                        total=analysis.extraction.total.value,
                        is_forged=analysis.anomaly.is_forged,
                    )
                    handle.write(prediction.model_dump_json())
                    handle.write("\n")
        finally:
            duration = time.perf_counter() - started

        print(f"documents={document_count}")
        print(f"latency_total_s={duration:.4f}")
        if durations:
            print(f"latency_mean_s={statistics.mean(durations):.4f}")
            print(f"latency_p95_s={_percentile(durations, 0.95):.4f}")
            print(f"peak_memory_mb_mean={statistics.mean(peaks_mb):.2f}")
        if args.model_dir:
            artifact_size_bytes = _directory_size_bytes(Path(args.model_dir))
            print(f"artifact_size_bytes={artifact_size_bytes}")
        return 0

    model_bundle = None
    if args.model_dir:
        model_bundle = _load_model_bundle(Path(args.model_dir))

    durations = []
    peaks_mb = []
    for input_path in args.inputs:
        tracemalloc.start()
        try:
            started = time.perf_counter()
            analyze_document(str(Path(input_path)), model_bundle=model_bundle, debug=False)
            durations.append(time.perf_counter() - started)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peaks_mb.append(peak / (1024 * 1024))

    print(f"documents={len(durations)}")
    print(f"latency_mean_s={statistics.mean(durations):.4f}")
    print(f"latency_p95_s={_percentile(durations, 0.95):.4f}")
    print(f"peak_memory_mb_mean={statistics.mean(peaks_mb):.2f}")
    if args.model_dir:
        artifact_size_bytes = _directory_size_bytes(Path(args.model_dir))
        print(f"artifact_size_bytes={artifact_size_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
