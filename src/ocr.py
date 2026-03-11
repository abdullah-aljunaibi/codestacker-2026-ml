"""Tesseract OCR helpers with structured word and line-level output."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytesseract
from PIL import Image

from src.config import DEFAULT_CONFIG
from src.preprocessing import preprocess_for_ocr
from src.types import Box, OCRLine, OCRWord


class OCRResult(tuple):
    """Backward-compatible OCR result container."""

    __slots__ = ()

    def __new__(
        cls,
        text: str,
        words: tuple[OCRWord, ...],
        image_size: tuple[int, int],
        lines: tuple[OCRLine, ...] = (),
    ) -> "OCRResult":
        return super().__new__(cls, (text, words, image_size, lines))

    @property
    def text(self) -> str:
        return self[0]

    @property
    def words(self) -> tuple[OCRWord, ...]:
        return self[1]

    @property
    def image_size(self) -> tuple[int, int]:
        return self[2]

    @property
    def lines(self) -> tuple[OCRLine, ...]:
        return self[3]


def preprocess_image(image: Image.Image) -> Image.Image:
    """Apply the project's OCR preprocessing policy."""
    return preprocess_for_ocr(image)


def load_image(image_path: str | Path) -> Image.Image:
    """Open an image from disk and detach it from the file handle."""
    with Image.open(image_path) as image:
        return image.copy()


def run_ocr(
    image: Image.Image,
    tesseract_config: str | None = None,
    page_index: int = 0,
) -> OCRResult:
    """Run Tesseract on an already loaded image."""
    processed = preprocess_image(image)
    config = tesseract_config or DEFAULT_CONFIG.ocr.tesseract_config
    text = pytesseract.image_to_string(processed, config=config)
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    words = tuple(_iter_words(data, page_index=page_index))
    lines = tuple(_group_lines(words))
    return OCRResult(text=text, words=words, image_size=processed.size, lines=lines)


def extract_ocr(image_path: str | Path, tesseract_config: str | None = None) -> OCRResult:
    """Load an image from disk and run OCR on it."""
    return run_ocr(load_image(image_path), tesseract_config=tesseract_config)


def _iter_words(data: dict[str, list[object]], page_index: int = 0) -> Iterable[OCRWord]:
    total_entries = len(data.get("text", []))
    for index in range(total_entries):
        raw_text = str(data["text"][index]).strip()
        if not raw_text:
            continue
        left = _safe_int(data["left"][index])
        top = _safe_int(data["top"][index])
        width = _safe_int(data["width"][index])
        height = _safe_int(data["height"][index])
        if None in (left, top, width, height):
            continue
        yield OCRWord(
            text=raw_text,
            left=left,
            top=top,
            width=width,
            height=height,
            confidence=_safe_float(data["conf"][index]),
            line_num=_safe_int(data.get("line_num", [None] * total_entries)[index]),
            block_num=_safe_int(data.get("block_num", [None] * total_entries)[index]),
            paragraph_num=_safe_int(data.get("par_num", [None] * total_entries)[index]),
            page_index=page_index,
        )


def _group_lines(words: tuple[OCRWord, ...]) -> Iterable[OCRLine]:
    grouped: dict[tuple[int, int | None, int | None, int | None], list[OCRWord]] = {}
    for word in words:
        key = (
            word.page_index,
            word.block_num,
            word.paragraph_num,
            word.line_num,
        )
        grouped.setdefault(key, []).append(word)

    ordered_groups = sorted(
        grouped.values(),
        key=lambda line_words: (
            line_words[0].page_index,
            min(word.top for word in line_words),
            min(word.left for word in line_words),
        ),
    )
    for line_words in ordered_groups:
        ordered_words = tuple(sorted(line_words, key=lambda word: (word.left, word.top)))
        confidence_values = [
            word.confidence
            for word in ordered_words
            if word.confidence is not None and word.confidence >= 0
        ]
        yield OCRLine(
            text=" ".join(word.text for word in ordered_words).strip(),
            words=ordered_words,
            box=Box(
                left=min(word.left for word in ordered_words),
                top=min(word.top for word in ordered_words),
                right=max(word.right for word in ordered_words),
                bottom=max(word.bottom for word in ordered_words),
                page_index=ordered_words[0].page_index,
            ),
            confidence=sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
            page_index=ordered_words[0].page_index,
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
