"""Error level analysis helpers for forgery detection."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

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


def _load_rgb_image(image_or_path: Image.Image | str | Path) -> Image.Image | None:
    if isinstance(image_or_path, Image.Image):
        try:
            return image_or_path.convert("RGB")
        except Exception:
            return None

    try:
        with Image.open(image_or_path) as source:
            return source.convert("RGB")
    except Exception:
        return None


def compute_ela_array(image_or_path: Image.Image | str | Path, quality: int = 90) -> np.ndarray:
    """Return the grayscale ELA map for localization."""
    rgb = _load_rgb_image(image_or_path)
    if rgb is None:
        return np.zeros((1, 1), dtype=np.float64)

    buffer = BytesIO()
    try:
        rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            diff = ImageChops.difference(rgb, recompressed.convert("RGB"))
    except Exception:
        return np.zeros((max(rgb.height, 1), max(rgb.width, 1)), dtype=np.float64)

    return np.asarray(diff.convert("L"), dtype=np.float64)


def extract_ela_features(image_or_path: Image.Image | str | Path, quality: int = 90) -> dict[str, float]:
    """Compute compact ELA statistics from a document image."""
    ela = compute_ela_array(image_or_path, quality=quality)
    if ela.size == 0:
        return _empty_ela_features()

    height, width = ela.shape[:2]
    if width == 0 or height == 0:
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
