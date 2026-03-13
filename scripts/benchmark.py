#!/usr/bin/env python3
"""Measure local latency and memory for the shared analysis pipeline."""

from __future__ import annotations

import argparse
import math
import statistics
import time
import tracemalloc
from pathlib import Path

from src.pipeline import analyze_document


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--model-dir")
    args = parser.parse_args()

    durations = []
    peaks_mb = []
    for input_path in args.inputs:
        tracemalloc.start()
        started = time.perf_counter()
        analyze_document(str(Path(input_path)), model_bundle=None, debug=False)
        durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
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
