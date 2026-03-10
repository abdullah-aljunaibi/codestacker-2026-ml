"""Tesseract OCR helpers with structured word-level output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytesseract
from PIL import Image

from src.config import DEFAULT_CONFIG
from src.preprocessing import preprocess_for_ocr


@dataclass(frozen=True)
class OCRWord:
    """A single OCR token with its bounding box."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float | None = None
    line_num: int | None = None
    block_num: int | None = None
    paragraph_num: int | None = None

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class OCRResult:
    """Structured OCR response for downstream extractors."""

    text: str
    words: tuple[OCRWord, ...]
    image_size: tuple[int, int]


def preprocess_image(image: Image.Image) -> Image.Image:
    """Apply the project's OCR preprocessing policy."""
    return preprocess_for_ocr(image)


def load_image(image_path: str | Path) -> Image.Image:
    """Open an image from disk and detach it from the file handle."""
    with Image.open(image_path) as image:
        return image.copy()


def run_ocr(image: Image.Image, tesseract_config: str | None = None) -> OCRResult:
    """Run Tesseract on an already loaded image."""
    processed = preprocess_image(image)
    config = tesseract_config or DEFAULT_CONFIG.ocr.tesseract_config
    text = pytesseract.image_to_string(processed, config=config)
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    words = tuple(_iter_words(data))
    return OCRResult(text=text, words=words, image_size=processed.size)


def extract_ocr(image_path: str | Path, tesseract_config: str | None = None) -> OCRResult:
    """Load an image from disk and run OCR on it."""
    return run_ocr(load_image(image_path), tesseract_config=tesseract_config)


def _iter_words(data: dict[str, list[object]]) -> Iterable[OCRWord]:
    total_entries = len(data.get("text", []))
    for index in range(total_entries):
        raw_text = str(data["text"][index]).strip()
        if not raw_text:
            continue
        yield OCRWord(
            text=raw_text,
            left=_safe_int(data["left"][index]),
            top=_safe_int(data["top"][index]),
            width=_safe_int(data["width"][index]),
            height=_safe_int(data["height"][index]),
            confidence=_safe_float(data["conf"][index]),
            line_num=_safe_int(data.get("line_num", [None] * total_entries)[index]),
            block_num=_safe_int(data.get("block_num", [None] * total_entries)[index]),
            paragraph_num=_safe_int(data.get("par_num", [None] * total_entries)[index]),
        )


def _safe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if numeric < 0 else numeric
