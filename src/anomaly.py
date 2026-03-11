"""Anomaly detection based on visual, OCR, extraction, and consistency features."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from src.config import DEFAULT_CONFIG
from src.consistency import CONSISTENCY_FEATURE_KEYS, extract_consistency_features
from src.ela import ELA_FEATURE_KEYS, compute_ela_array, extract_ela_features
from src.extractor import parse_amount
from src.reproducibility import set_deterministic_seeds
from src.types import AnalysisResult, Box, OCRWord


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
TEXT_FEATURE_KEYS = [
    "text_length",
    "line_count",
    "digit_ratio",
    "alpha_ratio",
    "special_ratio",
    "avg_line_length",
    "empty_line_ratio",
]


def _extract_base_image_features(image_path: str) -> dict[str, float]:
    try:
        with Image.open(image_path) as source:
            gray = source.convert("L")
    except Exception:
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


def extract_image_features(image_path: str) -> dict[str, float]:
    features = _extract_base_image_features(image_path)
    features.update(extract_ela_features(image_path))
    return features


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


def build_feature_vector(
    analysis: AnalysisResult,
    stats: dict[str, Any] | None,
) -> tuple[dict[str, float], list[float]]:
    amount = parse_amount(analysis.extraction.total.value) or 0.0
    image_features = extract_image_features(analysis.document_path)
    text_features = extract_text_features(analysis.ocr_text)
    consistency_features = extract_consistency_features(
        analysis.ocr_text,
        amount,
        stats,
        vendor=analysis.extraction.vendor.value,
    )
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
    vector = (
        features_to_vector(image_features)
        + text_features_to_vector(text_features)
        + [float(consistency_features.get(key, 0.0)) for key in CONSISTENCY_FEATURE_KEYS]
        + extra_features_to_vector(extra_features)
        + [float(amount)]
    )
    return all_features, vector


def localize_suspicious_regions(
    image_path: str,
    words: tuple[OCRWord, ...],
    max_regions: int = 6,
) -> tuple[Box, ...]:
    regions: list[Box] = []
    ela = compute_ela_array(image_path)
    if ela.size:
        height, width = ela.shape[:2]
        block = max(24, min(height, width) // 10)
        block_scores: list[tuple[float, Box]] = []
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
                                page_index=0,
                            ),
                        )
                    )
        block_scores.sort(key=lambda item: item[0], reverse=True)
        regions.extend(box for _, box in block_scores[: max_regions // 2])

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
            "ocr_low_conf_ratio": 0.0,
            "total_confidence": 1.0 if amount > 0 else 0.0,
            "field_presence": 1.0 if amount > 0 else 0.0,
        }
    )
    vector = (
        features_to_vector(image_features)
        + text_features_to_vector(text_features)
        + [float(consistency_features.get(key, 0.0)) for key in CONSISTENCY_FEATURE_KEYS]
        + [0.0, 0.0, text_features["line_count"], 0.0, 0.0, 0.0, 1.0 if amount > 0 else 0.0, 1.0 if amount > 0 else 0.0]
        + [amount]
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

    analyses: list[tuple[AnalysisResult, int]] = []
    predicted_vendors: set[str] = set()
    predicted_amounts: list[float] = []

    for record in records:
        image_path = Path(train_dir) / str(record.get("image_path", ""))
        analysis = analyze_document(str(image_path), model_bundle=None, debug=False)
        label = int(record.get("label", {}).get("is_forged", 0))
        analyses.append((analysis, label))
        if analysis.extraction.vendor.value:
            predicted_vendors.add(analysis.extraction.vendor.value)
        amount = parse_amount(analysis.extraction.total.value)
        if amount is not None:
            predicted_amounts.append(amount)

    stats = {
        "vendors": sorted(predicted_vendors),
        "amount_mean": float(np.mean(predicted_amounts)) if predicted_amounts else 0.0,
        "amount_std": float(np.std(predicted_amounts)) if predicted_amounts else 0.0,
        "amount_q1": float(np.percentile(predicted_amounts, 25)) if predicted_amounts else 0.0,
        "amount_q3": float(np.percentile(predicted_amounts, 75)) if predicted_amounts else 0.0,
    }

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    for analysis, label in analyses:
        _, vector = build_feature_vector(analysis, stats)
        X_rows.append(vector)
        y_rows.append(label)

    X = np.asarray(X_rows, dtype=np.float64) if X_rows else np.zeros((0, 0), dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)

    model = None
    validation_accuracy = None
    train_accuracy = None
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
            for train_index, test_index in cv.split(X, y):
                fold_model = GradientBoostingClassifier(
                    n_estimators=DEFAULT_CONFIG.training.gradient_boosting_estimators,
                    learning_rate=DEFAULT_CONFIG.training.gradient_boosting_learning_rate,
                    max_depth=DEFAULT_CONFIG.training.gradient_boosting_max_depth,
                    min_samples_leaf=DEFAULT_CONFIG.training.gradient_boosting_min_samples_leaf,
                    random_state=DEFAULT_CONFIG.training.random_state,
                )
                fold_model.fit(X[train_index], y[train_index])
                predictions = fold_model.predict(X[test_index])
                fold_scores.append(float(accuracy_score(y[test_index], predictions)))
            validation_accuracy = float(np.mean(fold_scores)) if fold_scores else None
        model.fit(X, y)
        train_accuracy = float(accuracy_score(y, model.predict(X)))

    artifact = {
        "model": model,
        "model_type": type(model).__name__ if model is not None else "heuristic",
        "feature_keys": FEATURE_KEYS + TEXT_FEATURE_KEYS + CONSISTENCY_FEATURE_KEYS + EXTRA_FEATURE_KEYS + ["amount"],
        "forged_ratio": float(np.mean(y)) if len(y) else 0.0,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "threshold": DEFAULT_CONFIG.training.anomaly_threshold,
        "stats": stats,
    }

    with (Path(model_dir) / DEFAULT_CONFIG.data.anomaly_model_file_name).open("wb") as handle:
        pickle.dump(artifact, handle)
    return artifact
