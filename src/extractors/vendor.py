"""Vendor field extraction heuristics."""

from __future__ import annotations

import re
from typing import Iterable

from src.ocr import OCRWord


VENDOR_STOP_WORDS = {
    "receipt",
    "invoice",
    "tax",
    "date",
    "time",
    "cashier",
    "total",
    "subtotal",
    "change",
    "cash",
    "card",
    "visa",
    "master",
    "thank",
    "tel",
    "phone",
    "fax",
    "address",
    "no.",
    "gst",
    "sst",
}


def extract_vendor(text: str, words: Iterable[OCRWord] | None = None) -> str | None:
    """Extract vendor name from the OCR text, using layout hints when available."""
    if words:
        candidate = _extract_vendor_from_words(words)
        if candidate:
            return candidate
    return _extract_vendor_from_text(text)


def _extract_vendor_from_words(words: Iterable[OCRWord]) -> str | None:
    grouped_lines: dict[tuple[int | None, int | None, int | None], list[OCRWord]] = {}
    for word in words:
        key = (word.block_num, word.paragraph_num, word.line_num)
        grouped_lines.setdefault(key, []).append(word)

    ordered_lines = sorted(
        (
            sorted(line_words, key=lambda word: (word.left or 0, word.top or 0))
            for line_words in grouped_lines.values()
        ),
        key=lambda line_words: min((word.top or 0) for word in line_words),
    )

    for line_words in ordered_lines[:5]:
        line = " ".join(word.text for word in line_words).strip()
        if _is_vendor_candidate(line):
            return line
    return None


def _extract_vendor_from_text(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:5]:
        if _is_vendor_candidate(line):
            return line
    return None


def _is_vendor_candidate(line: str) -> bool:
    normalized = line.strip()
    if len(normalized) < 3:
        return False

    lower = normalized.lower()
    if re.match(r"^[\d\s\-/.:,$%]+$", normalized):
        return False
    if any(word in lower for word in VENDOR_STOP_WORDS):
        return False
    return True
