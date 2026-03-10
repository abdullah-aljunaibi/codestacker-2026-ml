"""Reusable image preprocessing helpers."""

from __future__ import annotations

from PIL import Image

from src.config import DEFAULT_CONFIG


def preprocess_for_ocr(
    img: Image.Image,
    min_width: int = DEFAULT_CONFIG.preprocessing.min_width,
) -> Image.Image:
    """Apply the OCR preprocessing policy defined in config."""
    processed = img.convert("L") if DEFAULT_CONFIG.preprocessing.grayscale else img.copy()
    width, height = processed.size

    if width > 0 and width < min_width:
        scale = min_width / width
        processed = processed.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS,
        )

    return processed
