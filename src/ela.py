"""Error level analysis helpers for forgery detection."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageChops


def _empty_ela_features() -> dict[str, float]:
    return {
        "ela_mean": 0.0,
        "ela_std": 0.0,
        "ela_max": 0.0,
        "ela_high_ratio": 0.0,
        "ela_block_std": 0.0,
    }


ELA_FEATURE_KEYS = list(_empty_ela_features().keys())


def extract_ela_features(image_path: str, quality: int = 90) -> dict[str, float]:
    """Compute compact ELA statistics from a document image."""
    try:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
    except Exception:
        return _empty_ela_features()

    width, height = rgb.size
    if width == 0 or height == 0:
        return _empty_ela_features()

    buffer = BytesIO()
    try:
        rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            diff = ImageChops.difference(rgb, recompressed.convert("RGB"))
    except Exception:
        return _empty_ela_features()

    ela = np.asarray(diff.convert("L"), dtype=np.float64)
    if ela.size == 0:
        return _empty_ela_features()

    block_size = max(16, min(width, height) // 10)
    block_means: list[float] = []
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            block = ela[top : top + block_size, left : left + block_size]
            if block.size:
                block_means.append(float(block.mean()))

    return {
        "ela_mean": float(ela.mean()),
        "ela_std": float(ela.std()),
        "ela_max": float(ela.max()),
        "ela_high_ratio": float((ela >= 25.0).mean()),
        "ela_block_std": float(np.std(block_means)) if block_means else 0.0,
    }
