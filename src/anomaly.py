"""Anomaly detection based on visual, OCR, extraction, and consistency features."""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter

from src.config import DEFAULT_CONFIG
from src.consistency import CONSISTENCY_FEATURE_KEYS, extract_consistency_features
from src.ela import ELA_FEATURE_KEYS, compute_ela_array, extract_ela_features
from src.extractor import (
    AMOUNT_REGEX,
    NEGATIVE_TOTAL_TERMS,
    POSITIVE_TOTAL_TERMS,
    parse_amount,
    _looks_like_date_amount,
    _looks_like_invoice_id,
    _looks_like_phone_amount,
    _normalize_amount_token,
)
from src.reproducibility import set_deterministic_seeds
from src.types import AnalysisResult, Box, ExtractionResult, OCRWord


def _empty_base_image_features() -> dict[str, float]:
    return {
        "img_mean": 0.0,
        "img_std": 0.0,
        "img_median": 0.0,
        "edge_mean": 0.0,
        "edge_std": 0.0,
        "noise_std": 0.0,
        "noise_mean": 0.0,
        "dynamic_range": 0.0,
        "entropy": 0.0,
        "block_var_std": 0.0,
        "block_var_mean": 0.0,
        "aspect_ratio": 1.0,
        "total_pixels": 0.0,
        "p5": 0.0,
        "p95": 0.0,
    }


BASE_IMAGE_FEATURE_KEYS = list(_empty_base_image_features().keys())
EXTRA_FEATURE_KEYS = [
    "ocr_mean_confidence",
    "ocr_low_conf_ratio",
    "line_count",
    "word_count",
    "vendor_confidence",
    "date_confidence",
    "total_confidence",
    "field_presence",
]
FEATURE_KEYS = BASE_IMAGE_FEATURE_KEYS + ELA_FEATURE_KEYS
FIELD_LOCAL_FEATURE_KEYS = [
    "field_ela_vendor",
    "field_ela_date",
    "field_ela_total",
    "field_conf_vendor",
    "field_conf_date",
    "field_conf_total",
    "field_present_vendor",
    "field_present_date",
    "field_present_total",
    "field_conflict_count",
]
TEXT_FEATURE_KEYS = [
    "text_length",
    "line_count",
    "digit_ratio",
    "alpha_ratio",
    "special_ratio",
    "avg_line_length",
    "empty_line_ratio",
]
MODEL_FEATURE_KEYS = FEATURE_KEYS + TEXT_FEATURE_KEYS + CONSISTENCY_FEATURE_KEYS + EXTRA_FEATURE_KEYS + FIELD_LOCAL_FEATURE_KEYS + ["amount"]


def _load_gray_image(image_or_path: Image.Image | str | Path) -> Image.Image | None:
    if isinstance(image_or_path, Image.Image):
        try:
            return image_or_path.convert("L")
        except Exception:
            return None

    try:
        with Image.open(image_or_path) as source:
            return source.convert("L")
    except Exception:
        return None


def _extract_base_image_features(image_or_path: Image.Image | str | Path) -> dict[str, float]:
    gray = _load_gray_image(image_or_path)
    if gray is None:
        return _empty_base_image_features()

    width, height = gray.size
    if width == 0 or height == 0:
        return _empty_base_image_features()

    pixels = np.asarray(gray, dtype=np.float64)
    if pixels.size == 0:
        return _empty_base_image_features()

    edge_pixels = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    blur_pixels = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float64)
    noise = pixels - blur_pixels

    hist, _ = np.histogram(pixels.flatten(), bins=64, range=(0, 256))
    if hist.sum() == 0:
        return _empty_base_image_features()
    hist_norm = hist / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]

    block_size = max(16, min(width, height) // 8)
    block_vars: list[float] = []
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            block = pixels[top : top + block_size, left : left + block_size]
            if block.size:
                block_vars.append(float(block.var()))

    p5 = float(np.percentile(pixels, 5))
    p95 = float(np.percentile(pixels, 95))
    return {
        "img_mean": float(pixels.mean()),
        "img_std": float(pixels.std()),
        "img_median": float(np.median(pixels)),
        "edge_mean": float(edge_pixels.mean()),
        "edge_std": float(edge_pixels.std()),
        "noise_std": float(noise.std()),
        "noise_mean": float(np.abs(noise).mean()),
        "dynamic_range": p95 - p5,
        "entropy": float(-np.sum(hist_norm * np.log2(hist_norm))),
        "block_var_std": float(np.std(block_vars)) if block_vars else 0.0,
        "block_var_mean": float(np.mean(block_vars)) if block_vars else 0.0,
        "aspect_ratio": width / height if height else 1.0,
        "total_pixels": float(width * height),
        "p5": p5,
        "p95": p95,
    }


def extract_image_features(image_or_path: Image.Image | str | Path) -> dict[str, float]:
    features = _extract_base_image_features(image_or_path)
    features.update(extract_ela_features(image_or_path))
    return features


def _aggregate_page_feature_sets(page_feature_sets: Iterable[dict[str, float]]) -> dict[str, float]:
    feature_sets = list(page_feature_sets)
    if not feature_sets:
        return dict.fromkeys(FEATURE_KEYS, 0.0)

    aggregated: dict[str, float] = {}
    for key in FEATURE_KEYS:
        values = [float(features.get(key, 0.0)) for features in feature_sets]
        if key in {"ela_max", "ela_high_ratio"}:
            aggregated[key] = float(np.max(values))
        else:
            aggregated[key] = float(np.mean(values))
    return aggregated


def extract_text_features(ocr_text: str) -> dict[str, float]:
    if not ocr_text or not ocr_text.strip():
        return {
            "text_length": 0.0,
            "line_count": 0.0,
            "digit_ratio": 0.0,
            "alpha_ratio": 0.0,
            "special_ratio": 0.0,
            "avg_line_length": 0.0,
            "empty_line_ratio": 0.0,
        }

    lines = ocr_text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    total_chars = len(ocr_text)
    digits = sum(char.isdigit() for char in ocr_text)
    alpha = sum(char.isalpha() for char in ocr_text)
    special = total_chars - digits - alpha - ocr_text.count(" ") - ocr_text.count("\n")
    return {
        "text_length": float(total_chars),
        "line_count": float(len(non_empty)),
        "digit_ratio": digits / total_chars if total_chars else 0.0,
        "alpha_ratio": alpha / total_chars if total_chars else 0.0,
        "special_ratio": special / total_chars if total_chars else 0.0,
        "avg_line_length": float(np.mean([len(line) for line in non_empty])) if non_empty else 0.0,
        "empty_line_ratio": (len(lines) - len(non_empty)) / len(lines) if lines else 0.0,
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in FEATURE_KEYS]


def text_features_to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in TEXT_FEATURE_KEYS]


def extra_features_to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in EXTRA_FEATURE_KEYS]


def field_local_features_to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in FIELD_LOCAL_FEATURE_KEYS]


def _box_overlap_ratio(first: Box, second: Box) -> float:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    first_area = float(max(first.width * first.height, 1))
    second_area = float(max(second.width * second.height, 1))
    return intersection / min(first_area, second_area)


def _expand_box(box: Box, width: int, height: int) -> Box:
    padding_x = max(6, int(round(box.width * 0.15)))
    padding_y = max(6, int(round(box.height * 0.2)))
    return Box(
        left=max(0, box.left - padding_x),
        top=max(0, box.top - padding_y),
        right=min(width, box.right + padding_x),
        bottom=min(height, box.bottom + padding_y),
        page_index=box.page_index,
    )


def _group_amount_candidate_lines(words: tuple[OCRWord, ...]) -> tuple[tuple[str, Box, float], ...]:
    grouped: dict[tuple[int, int | None, int | None, int | None], list[OCRWord]] = {}
    for word in words:
        grouped.setdefault(
            (word.page_index, word.block_num, word.paragraph_num, word.line_num),
            [],
        ).append(word)

    line_entries: list[tuple[str, Box, float]] = []
    for line_words in grouped.values():
        ordered = sorted(line_words, key=lambda word: (word.left, word.top))
        text = " ".join(word.text for word in ordered).strip()
        if not text:
            continue
        confidences = [
            word.confidence
            for word in ordered
            if word.confidence is not None and word.confidence >= 0
        ]
        line_entries.append(
            (
                text,
                Box(
                    left=min(word.left for word in ordered),
                    top=min(word.top for word in ordered),
                    right=max(word.right for word in ordered),
                    bottom=max(word.bottom for word in ordered),
                    page_index=ordered[0].page_index,
                ),
                float(np.mean(confidences)) if confidences else 0.0,
            )
        )
    return tuple(line_entries)


def _score_amount_candidate_lines(
    words: tuple[OCRWord, ...],
) -> tuple[tuple[str, str, float, Box], ...]:
    lines = _group_amount_candidate_lines(words)
    if not lines:
        return ()

    max_bottom = max((line_box.bottom for _, line_box, _ in lines), default=1)
    candidates: list[tuple[str, str, float, Box]] = []
    for index, (line_text, line_box, line_confidence) in enumerate(lines):
        lower = re.sub(r"\s+", " ", line_text.lower()).strip()
        amount_tokens = [match.group(0) for match in AMOUNT_REGEX.finditer(line_text)]
        if not amount_tokens:
            continue

        line_score = 0.0
        for term in POSITIVE_TOTAL_TERMS:
            if term in lower:
                line_score += 2.0 if term != "total" else 1.2
        for term in NEGATIVE_TOTAL_TERMS:
            if term in lower:
                line_score -= 1.4
        line_midpoint = (line_box.top + line_box.bottom) / 2.0
        vertical_ratio = line_midpoint / max(max_bottom, 1)
        if vertical_ratio >= 0.60:
            line_score += 0.9
        elif vertical_ratio >= 0.50:
            line_score += 0.4
        if index >= max(0, len(lines) - 4):
            line_score += 0.4

        for token in amount_tokens:
            normalized = _normalize_amount_token(token)
            if normalized is None:
                continue
            amount = float(normalized)
            score = line_score
            if amount > 0:
                score += 0.8
            if abs(amount - round(amount)) > 1e-6:
                score += 0.2
            if line_confidence >= 65:
                score += 0.2
            if "total" in lower and any(term in lower for term in ("subtotal", "sub total")):
                score -= 0.8
            if _looks_like_date_amount(token):
                score -= 2.2
            if _looks_like_phone_amount(token):
                score -= 2.0
            if _looks_like_invoice_id(token):
                score -= 1.8
            if lower.startswith(("qty", "quantity", "item", "no.", "number")):
                score -= 1.2
            candidates.append((line_text, normalized, score, line_box))

    candidates.sort(key=lambda item: item[2], reverse=True)
    return tuple(candidates)


def _count_conflicting_amount_candidates(words: tuple[OCRWord, ...], predicted_total: float) -> float:
    if predicted_total <= 0:
        return 0.0

    lines = _group_amount_candidate_lines(words)
    if not lines:
        return 0.0

    max_bottom = max((line_box.bottom for _, line_box, _ in lines), default=1)
    contradictory_amounts: set[str] = set()
    for index, (line_text, line_box, line_confidence) in enumerate(lines):
        lower = re.sub(r"\s+", " ", line_text.lower()).strip()
        amount_tokens = [match.group(0) for match in AMOUNT_REGEX.finditer(line_text)]
        if not amount_tokens:
            continue

        line_score = 0.0
        for term in POSITIVE_TOTAL_TERMS:
            if term in lower:
                line_score += 2.0 if term != "total" else 1.2
        for term in NEGATIVE_TOTAL_TERMS:
            if term in lower:
                line_score -= 1.4
        line_midpoint = (line_box.top + line_box.bottom) / 2.0
        vertical_ratio = line_midpoint / max(max_bottom, 1)
        if vertical_ratio >= 0.60:
            line_score += 0.9
        elif vertical_ratio >= 0.50:
            line_score += 0.4
        if index >= max(0, len(lines) - 4):
            line_score += 0.4

        for token in amount_tokens:
            normalized = _normalize_amount_token(token)
            if normalized is None:
                continue
            amount = float(normalized)
            score = line_score
            if amount > 0:
                score += 0.8
            if abs(amount - round(amount)) > 1e-6:
                score += 0.2
            if line_confidence >= 65:
                score += 0.2
            if "total" in lower and any(term in lower for term in ("subtotal", "sub total")):
                score -= 0.8
            if _looks_like_date_amount(token):
                score -= 2.2
            if _looks_like_phone_amount(token):
                score -= 2.0
            if _looks_like_invoice_id(token):
                score -= 1.8
            if lower.startswith(("qty", "quantity", "item", "no.", "number")):
                score -= 1.2
            if score < 0.9 or abs(amount - predicted_total) <= 0.06:
                continue
            contradictory_amounts.add(normalized)

    return float(len(contradictory_amounts))


def extract_field_local_features(
    page_images: tuple[Image.Image, ...],
    extraction: ExtractionResult,
    words: tuple[OCRWord, ...],
) -> dict[str, float]:
    field_features = dict.fromkeys(FIELD_LOCAL_FEATURE_KEYS, 0.0)
    ela_pages = tuple(compute_ela_array(page) for page in page_images)

    for field_name in ("vendor", "date", "total"):
        field = getattr(extraction, field_name)
        box = field.box
        key_suffix = field_name
        if box is None:
            continue
        field_features[f"field_present_{key_suffix}"] = 1.0
        page_index = min(max(box.page_index, 0), max(len(page_images) - 1, 0))
        if not page_images or page_index >= len(page_images):
            continue

        page_width, page_height = page_images[page_index].size
        expanded_box = _expand_box(box, page_width, page_height)

        ela_page = ela_pages[page_index]
        if ela_page.size:
            patch = ela_page[expanded_box.top : expanded_box.bottom, expanded_box.left : expanded_box.right]
            if patch.size:
                field_features[f"field_ela_{key_suffix}"] = float(np.mean(patch))

        overlapping_confidences = [
            word.confidence / 100.0
            for word in words
            if word.page_index == box.page_index
            and word.confidence is not None
            and word.confidence >= 0
            and _box_overlap_ratio(word.box, expanded_box) > 0.0
        ]
        if overlapping_confidences:
            field_features[f"field_conf_{key_suffix}"] = float(np.mean(overlapping_confidences))

    predicted_total = parse_amount(extraction.total.value) or 0.0
    field_features["field_conflict_count"] = _count_conflicting_amount_candidates(words, predicted_total)
    return field_features


def _compose_vector(
    image_features: dict[str, float],
    text_features: dict[str, float],
    consistency_features: dict[str, float],
    extra_features: dict[str, float],
    field_local_features: dict[str, float],
    amount: float,
) -> list[float]:
    return (
        features_to_vector(image_features)
        + text_features_to_vector(text_features)
        + [float(consistency_features.get(key, 0.0)) for key in CONSISTENCY_FEATURE_KEYS]
        + extra_features_to_vector(extra_features)
        + field_local_features_to_vector(field_local_features)
        + [float(amount)]
    )


def build_feature_vector(
    analysis: AnalysisResult,
    stats: dict[str, Any] | None,
    page_images: Iterable[Image.Image] | None = None,
) -> tuple[dict[str, float], list[float]]:
    amount = parse_amount(analysis.extraction.total.value) or 0.0
    pages = tuple(page_images) if page_images is not None else analysis.page_images
    if pages:
        image_features = _aggregate_page_feature_sets(extract_image_features(page) for page in pages)
    else:
        image_features = extract_image_features(analysis.document_path)
    text_features = extract_text_features(analysis.ocr_text)
    consistency_features = extract_consistency_features(
        analysis.ocr_text,
        amount,
        stats,
        vendor=analysis.extraction.vendor.value,
    )
    field_local_features = extract_field_local_features(tuple(pages), analysis.extraction, analysis.words)
    confidences = [
        word.confidence / 100.0
        for word in analysis.words
        if word.confidence is not None and word.confidence >= 0
    ]
    extra_features = {
        "ocr_mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "ocr_low_conf_ratio": float(np.mean([value < 0.5 for value in confidences])) if confidences else 0.0,
        "line_count": float(len(analysis.lines)),
        "word_count": float(len(analysis.words)),
        "vendor_confidence": analysis.extraction.vendor.confidence,
        "date_confidence": analysis.extraction.date.confidence,
        "total_confidence": analysis.extraction.total.confidence,
        "field_presence": float(
            sum(
                field.value is not None
                for field in (
                    analysis.extraction.vendor,
                    analysis.extraction.date,
                    analysis.extraction.total,
                )
            )
            / 3.0
        ),
    }

    all_features = {}
    all_features.update(image_features)
    all_features.update(text_features)
    all_features.update(consistency_features)
    all_features.update(extra_features)
    all_features.update(field_local_features)
    vector = _compose_vector(
        image_features,
        text_features,
        consistency_features,
        extra_features,
        field_local_features,
        amount,
    )
    return all_features, vector


def localize_suspicious_regions(
    page_images: Iterable[Image.Image] | Image.Image | str | Path,
    words: tuple[OCRWord, ...],
    max_regions: int = 6,
    extraction: ExtractionResult | None = None,
) -> tuple[Box, ...]:
    regions: list[Box] = []
    if isinstance(page_images, (str, Path, Image.Image)):
        pages = (page_images,)
    else:
        pages = tuple(page_images)

    ela_pages = tuple(compute_ela_array(page) for page in pages)

    block_scores: list[tuple[float, Box]] = []
    for page_index, ela in enumerate(ela_pages):
        if not ela.size:
            continue
        height, width = ela.shape[:2]
        block = max(24, min(height, width) // 10)
        for top in range(0, height, block):
            for left in range(0, width, block):
                patch = ela[top : top + block, left : left + block]
                if patch.size == 0:
                    continue
                score = float(np.mean(patch))
                if score >= 18.0:
                    block_scores.append(
                        (
                            score,
                            Box(
                                left=left,
                                top=top,
                                right=min(left + block, width),
                                bottom=min(top + block, height),
                                page_index=page_index,
                            ),
                        )
                    )

    block_scores.sort(key=lambda item: item[0], reverse=True)
    regions.extend(box for _, box in block_scores[: max_regions // 2])

    if extraction is not None and pages:
        field_boxes: list[tuple[float, Box]] = []
        for field_name in ("vendor", "date", "total"):
            field = getattr(extraction, field_name)
            if field.box is None:
                continue
            page_index = min(max(field.box.page_index, 0), max(len(pages) - 1, 0))
            page_image = _load_gray_image(pages[page_index])
            if page_image is None:
                continue
            page_width, page_height = page_image.size
            expanded_box = _expand_box(field.box, page_width, page_height)
            local_ela = 0.0
            ela_page = ela_pages[page_index]
            if ela_page.size:
                patch = ela_page[expanded_box.top : expanded_box.bottom, expanded_box.left : expanded_box.right]
                if patch.size:
                    local_ela = float(np.mean(patch))
            if field.confidence < 0.5 or local_ela >= 18.0:
                field_boxes.append((max(1.0 - field.confidence, local_ela / 18.0), expanded_box))

        field_boxes.sort(key=lambda item: item[0], reverse=True)
        for _, box in field_boxes:
            if len(regions) >= max_regions:
                break
            regions.append(box)

        amount_candidates = _score_amount_candidate_lines(words)
        if amount_candidates:
            best_score = amount_candidates[0][2]
            score_threshold = max(1.4, best_score - 0.6)
            predicted_total = _normalize_amount_token(extraction.total.value or "")
            high_scoring: list[tuple[str, float, Box]] = []
            seen_amounts: set[str] = set()
            for _, normalized, score, box in amount_candidates:
                if score < score_threshold or normalized in seen_amounts:
                    continue
                seen_amounts.add(normalized)
                high_scoring.append((normalized, score, box))

            if len(high_scoring) > 1:
                if predicted_total is None:
                    ambiguous = high_scoring[1:]
                else:
                    ambiguous = [entry for entry in high_scoring if entry[0] != predicted_total]
                    if not ambiguous:
                        ambiguous = high_scoring[1:]
                for _, _, box in ambiguous:
                    if len(regions) >= max_regions:
                        break
                    regions.append(box)

    low_conf_words = [
        word
        for word in words
        if word.confidence is not None and 0 <= word.confidence < 45
    ]
    low_conf_words.sort(key=lambda word: word.confidence or 0.0)
    for word in low_conf_words[: max_regions - len(regions)]:
        regions.append(word.box)

    deduped: list[Box] = []
    seen = set()
    for region in regions:
        key = (region.left, region.top, region.right, region.bottom, region.page_index)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    return tuple(deduped[:max_regions])


def heuristic_score(feature_values: dict[str, float]) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    score = 0.0
    if feature_values.get("ela_high_ratio", 0.0) > 0.12:
        score += 0.25
        reasons.append("Elevated ELA artifact ratio")
    if feature_values.get("noise_std", 0.0) > 20.0:
        score += 0.15
        reasons.append("High residual noise")
    if feature_values.get("consistency_risk", 0.0) > 0.5:
        score += 0.25
        reasons.append("Field consistency risk is high")
    if feature_values.get("ocr_low_conf_ratio", 0.0) > 0.35:
        score += 0.15
        reasons.append("Large share of low-confidence OCR tokens")
    if feature_values.get("total_confidence", 0.0) < 0.35:
        score += 0.1
        reasons.append("Weak total extraction confidence")
    if feature_values.get("field_presence", 0.0) < 0.67:
        score += 0.1
        reasons.append("Missing key fields")
    return float(min(score, 1.0)), tuple(reasons)


def _calibrate_threshold_from_oof(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    from sklearn.metrics import f1_score

    valid_mask = np.isfinite(probabilities)
    if valid_mask.sum() < 4:
        return None

    y_true = labels[valid_mask]
    y_score = probabilities[valid_mask]
    if len(np.unique(y_true)) < 2 or np.sum(y_true == 1) < 2:
        return None

    candidate_thresholds = np.unique(np.round(y_score, 4))
    if candidate_thresholds.size == 0:
        return None

    best_threshold = None
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        predictions = (y_score >= threshold).astype(int)
        score = float(f1_score(y_true, predictions, pos_label=1, zero_division=0))
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_threshold = float(threshold)
            continue
        if abs(score - best_f1) <= 1e-12 and best_threshold is not None:
            if abs(threshold - DEFAULT_CONFIG.training.anomaly_threshold) < abs(
                best_threshold - DEFAULT_CONFIG.training.anomaly_threshold
            ):
                best_threshold = float(threshold)

    return best_threshold


def predict_anomaly(
    model_data: dict[str, Any] | None,
    img_path: str,
    ocr_text: str,
    amount: float,
    stats: dict[str, Any] | None,
) -> int:
    image_features = extract_image_features(img_path)
    text_features = extract_text_features(ocr_text)
    consistency_features = extract_consistency_features(ocr_text, amount, stats)
    feature_values = {}
    feature_values.update(image_features)
    feature_values.update(text_features)
    feature_values.update(consistency_features)
    feature_values.update(
        {
            "ocr_mean_confidence": 0.0,
            "ocr_low_conf_ratio": 0.0,
            "line_count": text_features["line_count"],
            "word_count": 0.0,
            "vendor_confidence": 0.0,
            "date_confidence": 0.0,
            "total_confidence": 1.0 if amount > 0 else 0.0,
            "field_presence": 1.0 if amount > 0 else 0.0,
        }
    )
    field_local_features = dict.fromkeys(FIELD_LOCAL_FEATURE_KEYS, 0.0)
    extra_features = {key: float(feature_values.get(key, 0.0)) for key in EXTRA_FEATURE_KEYS}
    feature_values.update(field_local_features)
    vector = _compose_vector(
        image_features,
        text_features,
        consistency_features,
        extra_features,
        field_local_features,
        amount,
    )
    if model_data and model_data.get("model") is not None:
        probability = float(model_data["model"].predict_proba(np.asarray([vector], dtype=np.float64))[0][1])
        return int(probability >= float(model_data.get("threshold", DEFAULT_CONFIG.training.anomaly_threshold)))
    score, _ = heuristic_score(feature_values)
    return int(score >= DEFAULT_CONFIG.training.anomaly_threshold)


def train_anomaly_model(records: list[dict[str, Any]], train_dir: str, model_dir: str) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold

    from src.pipeline import analyze_document

    set_deterministic_seeds(DEFAULT_CONFIG.training.random_state)
    os.makedirs(model_dir, exist_ok=True)

    # Pass 1: analyze documents, collect stats, store lightweight metadata (no page images)
    analysis_meta: list[tuple[AnalysisResult, int]] = []
    predicted_vendors: set[str] = set()
    predicted_amounts: list[float] = []

    for record in records:
        image_path = Path(train_dir) / str(record.get("image_path", ""))
        analysis = analyze_document(str(image_path), model_bundle=None, debug=False)
        label = int(record.get("label", {}).get("is_forged", 0))
        if analysis.extraction.vendor.value:
            predicted_vendors.add(analysis.extraction.vendor.value)
        amount = parse_amount(analysis.extraction.total.value)
        if amount is not None:
            predicted_amounts.append(amount)
        # Drop page_images to free RAM — features were already computed during analyze_document
        analysis_no_images = AnalysisResult(
            document_path=analysis.document_path,
            ocr_text=analysis.ocr_text,
            words=analysis.words,
            lines=analysis.lines,
            extraction=analysis.extraction,
            anomaly=analysis.anomaly,
            page_count=analysis.page_count,
            page_sizes=analysis.page_sizes,
            page_images=(),
            debug=analysis.debug,
        )
        analysis_meta.append((analysis_no_images, label))

    stats = {
        "vendors": sorted(predicted_vendors),
        "amount_mean": float(np.mean(predicted_amounts)) if predicted_amounts else 0.0,
        "amount_std": float(np.std(predicted_amounts)) if predicted_amounts else 0.0,
        "amount_q1": float(np.percentile(predicted_amounts, 25)) if predicted_amounts else 0.0,
        "amount_q3": float(np.percentile(predicted_amounts, 75)) if predicted_amounts else 0.0,
    }

    # Pass 2: build feature vectors — re-load pages on demand to avoid holding all in RAM
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    for analysis, label in analysis_meta:
        # Re-load page images for this single document to compute features
        from src.document_io import load_document_pages
        pages = load_document_pages(analysis.document_path)
        page_images = tuple(page.image for page in pages)
        _, vector = build_feature_vector(analysis, stats, page_images=page_images)
        X_rows.append(vector)
        y_rows.append(label)
        del pages, page_images  # Free immediately

    X = np.asarray(X_rows, dtype=np.float64) if X_rows else np.zeros((0, 0), dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)

    model = None
    validation_accuracy = None
    train_accuracy = None
    calibrated_threshold = DEFAULT_CONFIG.training.anomaly_threshold
    if len(X_rows) >= 6 and len(set(y_rows)) >= 2:
        model = GradientBoostingClassifier(
            n_estimators=DEFAULT_CONFIG.training.gradient_boosting_estimators,
            learning_rate=DEFAULT_CONFIG.training.gradient_boosting_learning_rate,
            max_depth=DEFAULT_CONFIG.training.gradient_boosting_max_depth,
            min_samples_leaf=DEFAULT_CONFIG.training.gradient_boosting_min_samples_leaf,
            random_state=DEFAULT_CONFIG.training.random_state,
        )
        n_splits = min(5, len(y_rows) // 2)
        if n_splits >= 2:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=DEFAULT_CONFIG.training.random_state,
            )
            fold_scores = []
            oof_probabilities = np.full(len(y_rows), np.nan, dtype=np.float64)
            for train_index, test_index in cv.split(X, y):
                fold_model = GradientBoostingClassifier(
                    n_estimators=DEFAULT_CONFIG.training.gradient_boosting_estimators,
                    learning_rate=DEFAULT_CONFIG.training.gradient_boosting_learning_rate,
                    max_depth=DEFAULT_CONFIG.training.gradient_boosting_max_depth,
                    min_samples_leaf=DEFAULT_CONFIG.training.gradient_boosting_min_samples_leaf,
                    random_state=DEFAULT_CONFIG.training.random_state,
                )
                fold_model.fit(X[train_index], y[train_index])
                oof_probabilities[test_index] = fold_model.predict_proba(X[test_index])[:, 1]
                predictions = fold_model.predict(X[test_index])
                fold_scores.append(float(accuracy_score(y[test_index], predictions)))
            validation_accuracy = float(np.mean(fold_scores)) if fold_scores else None
            calibrated = _calibrate_threshold_from_oof(y, oof_probabilities)
            if calibrated is not None:
                calibrated_threshold = calibrated
        model.fit(X, y)
        train_accuracy = float(accuracy_score(y, model.predict(X)))

    artifact = {
        "model": model,
        "model_type": type(model).__name__ if model is not None else "heuristic",
        "feature_keys": MODEL_FEATURE_KEYS,
        "forged_ratio": float(np.mean(y)) if len(y) else 0.0,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "threshold": calibrated_threshold,
        "stats": stats,
    }

    with (Path(model_dir) / DEFAULT_CONFIG.data.anomaly_model_file_name).open("wb") as handle:
        pickle.dump(artifact, handle)
    return artifact
