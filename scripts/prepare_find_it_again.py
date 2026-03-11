#!/usr/bin/env python3
"""Prepare a Find-It-Again style dataset into the challenge manifest schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for image_path in sorted(root.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
            continue
        records.append({"id": image_path.stem, "image_path": image_path.relative_to(root).as_posix()})
    with Path(args.output).open("w") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
