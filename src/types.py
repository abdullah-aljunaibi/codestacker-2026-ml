"""Typed analysis structures shared across OCR, extraction, anomaly, and UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int
    page_index: int = 0

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float | None = None
    line_num: int | None = None
    block_num: int | None = None
    paragraph_num: int | None = None
    page_index: int = 0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def box(self) -> Box:
        return Box(
            left=self.left,
            top=self.top,
            right=self.right,
            bottom=self.bottom,
            page_index=self.page_index,
        )


@dataclass(frozen=True)
class OCRLine:
    text: str
    words: tuple[OCRWord, ...]
    box: Box
    confidence: float
    page_index: int = 0


@dataclass(frozen=True)
class FieldPrediction:
    name: str
    value: str | None
    confidence: float
    box: Box | None = None
    page_index: int = 0
    raw_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "page_index": self.page_index,
            "raw_value": self.raw_value,
        }
        payload["box"] = self.box.to_dict() if self.box is not None else None
        return payload


@dataclass(frozen=True)
class ExtractionResult:
    vendor: FieldPrediction
    date: FieldPrediction
    total: FieldPrediction

    def as_fields(self) -> dict[str, str | None]:
        return {
            "vendor": self.vendor.value,
            "date": self.date.value,
            "total": self.total.value,
        }


@dataclass(frozen=True)
class AnomalyResult:
    score: float
    is_forged: int
    reasons: tuple[str, ...] = ()
    suspicious_regions: tuple[Box, ...] = ()
    feature_values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    document_path: str
    ocr_text: str
    words: tuple[OCRWord, ...]
    lines: tuple[OCRLine, ...]
    extraction: ExtractionResult
    anomaly: AnomalyResult
    page_count: int
    page_sizes: tuple[tuple[int, int], ...]
    page_images: tuple[Image.Image, ...] = field(default_factory=tuple, hash=False, compare=False)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelBundle:
    stats: dict[str, Any]
    anomaly_model_data: dict[str, Any] | None = None
