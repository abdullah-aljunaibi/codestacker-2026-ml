#!/usr/bin/env python3
"""Measure local latency and memory for the shared analysis pipeline."""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from pathlib import Path

from src.pipeline import analyze_document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
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
    print(f"latency_p95_s={max(durations):.4f}")
    print(f"peak_memory_mb_mean={statistics.mean(peaks_mb):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
