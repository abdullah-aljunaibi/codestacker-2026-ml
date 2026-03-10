"""Compact summary helpers for extraction and anomaly outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExtractionSummary:
    vendor: str
    date: str
    total: str
    ocr_text_present: bool


@dataclass(frozen=True)
class AnomalySummary:
    status: str
    score: float
    threshold: float
    reason_count: int
    strongest_signal: str


@dataclass(frozen=True)
class DocumentSummary:
    extraction: ExtractionSummary
    anomaly: AnomalySummary

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_extraction(fields: dict[str, object]) -> ExtractionSummary:
    """Create a stable extraction summary for template and UI consumption."""
    ocr_text = str(fields.get("_ocr_text") or "").strip()
    return ExtractionSummary(
        vendor=str(fields.get("vendor") or ""),
        date=str(fields.get("date") or ""),
        total=str(fields.get("total") or ""),
        ocr_text_present=bool(ocr_text),
    )


def summarize_anomaly(
    score: float,
    threshold: float,
    reasons: list[str] | tuple[str, ...] | None = None,
) -> AnomalySummary:
    """Create a compact anomaly summary from heuristic or model output."""
    safe_reasons = [reason.strip() for reason in (reasons or []) if reason and reason.strip()]
    return AnomalySummary(
        status="Suspicious" if score >= threshold else "Genuine",
        score=round(float(score), 4),
        threshold=round(float(threshold), 4),
        reason_count=len(safe_reasons),
        strongest_signal=safe_reasons[0] if safe_reasons else "",
    )


def summarize_document(
    fields: dict[str, object],
    anomaly_score: float,
    threshold: float,
    reasons: list[str] | tuple[str, ...] | None = None,
) -> DocumentSummary:
    """Bundle extraction and anomaly summaries into one serializable object."""
    return DocumentSummary(
        extraction=summarize_extraction(fields),
        anomaly=summarize_anomaly(anomaly_score, threshold, reasons),
    )
