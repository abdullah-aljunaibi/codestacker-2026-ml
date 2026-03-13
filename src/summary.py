"""Compact summary helpers for extraction, anomaly outputs, and NLG summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os

from openai import OpenAI

from src.config import DEFAULT_CONFIG
from src.types import AnalysisResult

LOGGER = logging.getLogger(__name__)


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


def generate_anomaly_summary(analysis: AnalysisResult) -> str:
    """Generate a human-readable forensic summary with deterministic fallback."""
    summary, _method = generate_anomaly_summary_with_method(analysis)
    return summary


def generate_anomaly_summary_with_method(analysis: AnalysisResult) -> tuple[str, str]:
    """Return the anomaly summary and generation method for UI/debug use."""
    llm_summary = _generate_llm_summary(analysis)
    if llm_summary:
        return llm_summary, "llm"

    LOGGER.info("Falling back to rule-based anomaly summary generation")
    return _generate_nlg_summary(analysis), "nlg"


def _generate_nlg_summary(analysis: AnalysisResult) -> str:
    """Generate a deterministic, human-readable forensic summary."""
    features = analysis.anomaly.feature_values or {}
    extraction = analysis.extraction
    score = float(analysis.anomaly.score)
    threshold = float(DEFAULT_CONFIG.training.anomaly_threshold)

    missing_fields = [
        field_name
        for field_name in ("vendor", "date", "total")
        if not getattr(extraction, field_name).value
    ]
    field_anomalies = _field_local_anomalies(features)
    evidence = _collect_evidence(features, missing_fields, field_anomalies)
    qualifier = _confidence_qualifier(score, threshold, len(evidence))
    status_text = _status_sentence(analysis, qualifier, evidence, score, threshold)

    if not evidence:
        if analysis.anomaly.is_forged:
            follow_up = "The current indicators are limited, but the combined anomaly score warrants manual review."
        else:
            follow_up = "No material irregularities were identified in the extracted text, field structure, or image characteristics."
        return f"{status_text} {follow_up}"

    lead = _lead_sentence(evidence, score, threshold)
    evidence_text = _join_phrases(evidence)
    closing = _closing_sentence(analysis, missing_fields, field_anomalies)
    return f"{status_text} {lead} {evidence_text}. {closing}"


def _generate_llm_summary(analysis: AnalysisResult) -> str | None:
    """Generate a short forensic summary via an OpenAI-compatible endpoint."""
    base_url = os.getenv("DOCFUSION_LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("DOCFUSION_LLM_MODEL", "mistral")
    api_key = os.getenv("DOCFUSION_LLM_API_KEY", "ollama")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=5.0)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=140,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a forensic document analysis assistant. "
                        "Write a concise 2-3 sentence summary of receipt tampering risk. "
                        "Ground the summary only in the provided analysis and avoid bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_llm_prompt(analysis),
                },
            ],
        )
    except Exception:
        return None

    content = response.choices[0].message.content if response.choices else None
    if isinstance(content, str):
        cleaned = " ".join(content.split())
        return cleaned or None
    return None


def _build_llm_prompt(analysis: AnalysisResult) -> str:
    extraction = analysis.extraction
    features = analysis.anomaly.feature_values or {}
    feature_items = _select_prompt_features(features)
    reasons = list(analysis.anomaly.reasons)

    return "\n".join(
        [
            "Prepare a forensic document analysis summary for this receipt.",
            "Requirements:",
            "- Write exactly 2 or 3 sentences.",
            "- Mention whether the receipt appears genuine or suspicious.",
            "- Reference the anomaly score and the strongest evidence.",
            "- If fields are missing or low-confidence, mention that briefly.",
            "",
            f"Anomaly score: {float(analysis.anomaly.score):.4f}",
            f"Decision threshold: {float(DEFAULT_CONFIG.training.anomaly_threshold):.4f}",
            f"Is forged: {bool(analysis.anomaly.is_forged)}",
            f"Reasons: {', '.join(reasons) if reasons else 'None'}",
            "Extraction results:",
            f"- Vendor: value={extraction.vendor.value or 'missing'}, confidence={extraction.vendor.confidence:.3f}",
            f"- Date: value={extraction.date.value or 'missing'}, confidence={extraction.date.confidence:.3f}",
            f"- Total: value={extraction.total.value or 'missing'}, confidence={extraction.total.confidence:.3f}",
            f"OCR text present: {bool(analysis.ocr_text.strip())}",
            f"Page count: {analysis.page_count}",
            "Key feature values:",
            *[f"- {name}: {value}" for name, value in feature_items],
        ]
    )


def _select_prompt_features(features: dict[str, float]) -> list[tuple[str, str]]:
    preferred_keys = (
        "ela_high_ratio",
        "ela_max",
        "ocr_low_conf_ratio",
        "ocr_mean_confidence",
        "field_conflict_count",
        "consistency_risk",
        "amount_zscore",
        "amount_iqr_gap",
        "field_ela_vendor",
        "field_conf_vendor",
        "field_ela_date",
        "field_conf_date",
        "field_ela_total",
        "field_conf_total",
        "field_present_vendor",
        "field_present_date",
        "field_present_total",
    )
    items: list[tuple[str, str]] = []
    for key in preferred_keys:
        if key in features:
            items.append((key, f"{float(features[key]):.4f}"))
    if not items:
        for key in sorted(features):
            items.append((key, f"{float(features[key]):.4f}"))
            if len(items) >= 12:
                break
    return items


def _collect_evidence(
    features: dict[str, float],
    missing_fields: list[str],
    field_anomalies: list[str],
) -> list[str]:
    evidence: list[str] = []

    if features.get("ela_high_ratio", 0.0) >= 0.12 or features.get("ela_max", 0.0) >= 40.0:
        evidence.append("Signs of digital manipulation were detected in the image, with elevated high-ELA regions")

    if features.get("ocr_low_conf_ratio", 0.0) >= 0.35 or features.get("ocr_mean_confidence", 1.0) <= 0.55:
        evidence.append("Text quality appears degraded, suggesting possible tampering or post-processing")

    if missing_fields:
        evidence.append(f"Key fields are missing ({'/'.join(missing_fields)})")

    if features.get("field_conflict_count", 0.0) >= 1.0 or features.get("consistency_risk", 0.0) >= 0.58:
        evidence.append("The reported total does not align cleanly with other amount evidence on the document")

    amount_zscore = features.get("amount_zscore", 0.0)
    amount_iqr_gap = features.get("amount_iqr_gap", 0.0)
    if amount_zscore >= 2.5 or amount_iqr_gap >= 1.0:
        evidence.append("The total amount is unusually high or low compared with the learned document distribution")

    for field_name in field_anomalies:
        evidence.append(f"Suspicious patterns were detected near the {field_name} field")

    return evidence


def _field_local_anomalies(features: dict[str, float]) -> list[str]:
    flagged: list[str] = []
    for field_name in ("vendor", "date", "total"):
        local_ela = features.get(f"field_ela_{field_name}", 0.0)
        local_conf = features.get(f"field_conf_{field_name}", 1.0)
        present = features.get(f"field_present_{field_name}", 1.0)
        if present <= 0.0:
            continue
        if local_ela >= 18.0 or local_conf <= 0.4:
            flagged.append(field_name)
    return flagged


def _confidence_qualifier(score: float, threshold: float, evidence_count: int) -> str:
    if score >= threshold + 0.2 or evidence_count >= 4:
        return "strongly suggests"
    if score >= threshold or evidence_count >= 2:
        return "likely"
    return "possibly"


def _status_sentence(
    analysis: AnalysisResult,
    qualifier: str,
    evidence: list[str],
    score: float,
    threshold: float,
) -> str:
    suspicious_openers = (
        "The document appears suspicious",
        "The receipt shows indicators of possible manipulation",
        "This document warrants concern for potential tampering",
    )
    genuine_openers = (
        "The document appears genuine",
        "The receipt is broadly consistent with an untampered document",
        "No strong signs of forgery are apparent in this document",
    )
    variant_index = _variant_index(analysis)

    if analysis.anomaly.is_forged:
        opener = suspicious_openers[variant_index % len(suspicious_openers)]
        return f"{opener} and {qualifier} alteration."

    if evidence:
        opener = genuine_openers[variant_index % len(genuine_openers)]
        return f"{opener}, although the available signals {qualifier} localized review."

    opener = genuine_openers[variant_index % len(genuine_openers)]
    margin = max(threshold - score, 0.0)
    if margin >= 0.15:
        return f"{opener} with no meaningful anomaly pressure."
    return f"{opener}, with only limited weak indicators noted."


def _lead_sentence(evidence: list[str], score: float, threshold: float) -> str:
    lead_options = (
        "This assessment is based on several observable indicators:",
        "The conclusion is supported by the following findings:",
        "Key factors informing this assessment include:",
    )
    index = (len(evidence) + int(round(score * 10)) + int(round(threshold * 10))) % len(lead_options)
    return lead_options[index]


def _closing_sentence(
    analysis: AnalysisResult,
    missing_fields: list[str],
    field_anomalies: list[str],
) -> str:
    if analysis.anomaly.is_forged and field_anomalies:
        return "Targeted verification of the highlighted fields would be appropriate before accepting the document"
    if missing_fields:
        return "The missing field coverage reduces confidence in the document record and should be checked manually"
    if analysis.anomaly.is_forged:
        return "Taken together, these indicators support a cautious forensic interpretation"
    return "Overall, the document structure remains largely coherent despite the noted weak signals"


def _join_phrases(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}, and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _variant_index(analysis: AnalysisResult) -> int:
    return (
        int(round(analysis.anomaly.score * 100))
        + analysis.page_count
        + len(analysis.anomaly.reasons)
        + len(analysis.anomaly.suspicious_regions)
    ) % 7
