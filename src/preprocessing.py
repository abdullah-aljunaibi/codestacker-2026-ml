"""Reusable image preprocessing helpers."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.config import DEFAULT_CONFIG


def preprocess_for_ocr(
    img: Image.Image,
    min_width: int = DEFAULT_CONFIG.preprocessing.min_width,
) -> Image.Image:
    """Apply deterministic OCR preprocessing, including lightweight deskew."""
    processed = img.convert("L") if DEFAULT_CONFIG.preprocessing.grayscale else img.copy()
    width, height = processed.size

    if width > 0 and width < min_width:
        scale = min_width / width
        processed = processed.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS,
        )

    processed = ImageOps.autocontrast(processed)
    processed = ImageEnhance.Contrast(processed).enhance(1.5)
    processed = processed.filter(ImageFilter.MedianFilter(size=3))
    angle = _estimate_skew_angle(processed)
    if abs(angle) > 0.1:
        processed = processed.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=255,
        )
    return ImageOps.autocontrast(processed)


def _estimate_skew_angle(image: Image.Image) -> float:
    resized = image.copy()
    if max(resized.size) > 900:
        resized.thumbnail((900, 900), Image.LANCZOS)

    binary = resized.point(lambda value: 255 if value > 180 else 0)
    best_angle = 0.0
    best_score = -math.inf

    for angle in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
        rotated = binary.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        pixels = 255.0 - np.asarray(rotated, dtype=np.float32)
        if pixels.size == 0:
            continue
        score = float(np.var(pixels.sum(axis=1)))
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle
