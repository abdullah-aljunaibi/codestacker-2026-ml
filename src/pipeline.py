"""Shared document analysis engine used by the harness and UI."""

from __future__ import annotations

import numpy as np

from src.anomaly import build_feature_vector, heuristic_score, localize_suspicious_regions
from src.config import DEFAULT_CONFIG
from src.document_io import load_document_pages
from src.extractor import extract_fields_from_ocr
from src.ocr import run_ocr
from src.types import AnalysisResult, AnomalyResult, ExtractionResult, ModelBundle, OCRLine, OCRWord


def analyze_document(
    document_path: str,
    model_bundle: ModelBundle | None = None,
    debug: bool = False,
) -> AnalysisResult:
    pages = load_document_pages(document_path)
    page_ocr_results = [run_ocr(page.image, page_index=page.page_index) for page in pages]

    words: list[OCRWord] = []
    lines: list[OCRLine] = []
    text_parts: list[str] = []
    extractions: list[ExtractionResult] = []

    for ocr_result in page_ocr_results:
        words.extend(ocr_result.words)
        lines.extend(ocr_result.lines)
        if ocr_result.text.strip():
            text_parts.append(ocr_result.text.strip())
        extractions.append(extract_fields_from_ocr(ocr_result))

    ocr_text = "\n".join(text_parts)
    extraction = _merge_extraction_results(extractions)
    provisional = AnalysisResult(
        document_path=document_path,
        ocr_text=ocr_text,
        words=tuple(words),
        lines=tuple(lines),
        extraction=extraction,
        anomaly=AnomalyResult(score=0.0, is_forged=0),
        page_count=len(pages),
        page_sizes=tuple(page.image.size for page in pages),
        page_images=tuple(page.image.copy() for page in pages),
        debug={},
    )

    stats = model_bundle.stats if model_bundle is not None else None
    feature_values, vector = build_feature_vector(provisional, stats, page_images=provisional.page_images)
    if model_bundle and model_bundle.anomaly_model_data and model_bundle.anomaly_model_data.get("model") is not None:
        model = model_bundle.anomaly_model_data["model"]
        score = float(model.predict_proba(np.asarray([vector], dtype=np.float64))[0][1])
        reasons = ("Model-based anomaly estimate",)
        threshold = float(
            model_bundle.anomaly_model_data.get(
                "threshold",
                DEFAULT_CONFIG.training.anomaly_threshold,
            )
        )
    else:
        score, reasons = heuristic_score(feature_values)
        threshold = DEFAULT_CONFIG.training.anomaly_threshold

    suspicious_regions = localize_suspicious_regions(
        provisional.page_images,
        tuple(words),
        extraction=extraction,
    )
    return AnalysisResult(
        document_path=document_path,
        ocr_text=ocr_text,
        words=tuple(words),
        lines=tuple(lines),
        extraction=extraction,
        anomaly=AnomalyResult(
            score=score,
            is_forged=int(score >= threshold),
            reasons=reasons,
            suspicious_regions=suspicious_regions,
            feature_values=feature_values if debug else {},
        ),
        page_count=len(pages),
        page_sizes=tuple(page.image.size for page in pages),
        page_images=provisional.page_images,
        debug={"vector_length": len(vector)} if debug else {},
    )


def _merge_extraction_results(results: list[ExtractionResult]) -> ExtractionResult:
    if not results:
        raise ValueError("at least one extraction result is required")

    def choose(field_name: str):
        best = None
        for result in results:
            candidate = getattr(result, field_name)
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best

    return ExtractionResult(
        vendor=choose("vendor"),
        date=choose("date"),
        total=choose("total"),
    )
