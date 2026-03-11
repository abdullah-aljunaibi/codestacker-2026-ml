"""Anomaly detection for receipt forgery using visual, ELA, and consistency signals."""

from __future__ import annotations

import os
import pickle

import numpy as np
from PIL import Image, ImageFilter

from src.config import DEFAULT_CONFIG
from src.consistency import (
    CONSISTENCY_FEATURE_KEYS,
    extract_consistency_features,
)
from src.ela import ELA_FEATURE_KEYS, extract_ela_features
from src.reproducibility import set_deterministic_seeds


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


def _extract_base_image_features(image_path: str) -> dict[str, float]:
    """Extract visual statistics from a receipt image."""
    try:
        with Image.open(image_path) as source:
            gray = source.convert("L")
    except Exception:
        return _empty_base_image_features()

    width, height = gray.size
    if width == 0 or height == 0:
        return _empty_base_image_features()

    pixels = np.asarray(gray, dtype=np.float64)
    if pixels.size == 0 or pixels.std() < 1.0:
        return _empty_base_image_features()

    edge_pixels = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    blur_pixels = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float64)
    noise = pixels - blur_pixels

    hist, _ = np.histogram(pixels.flatten(), bins=64, range=(0, 256))
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
    """Return the full visual feature set, including ELA signals."""
    features = _extract_base_image_features(image_path)
    features.update(extract_ela_features(image_path))
    return features


FEATURE_KEYS = BASE_IMAGE_FEATURE_KEYS + ELA_FEATURE_KEYS


def extract_text_features(ocr_text: str) -> dict[str, float]:
    """Extract text-based features for anomaly detection."""
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


TEXT_FEATURE_KEYS = list(extract_text_features("").keys())


def _vectorize(features: dict[str, float], keys: list[str]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in keys]


def features_to_vector(features: dict[str, float]) -> list[float]:
    return _vectorize(features, FEATURE_KEYS)


def text_features_to_vector(features: dict[str, float]) -> list[float]:
    return _vectorize(features, TEXT_FEATURE_KEYS)


def _compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    class_counts = np.bincount(labels.astype(int), minlength=2)
    weights = np.ones_like(labels, dtype=np.float64)
    for label, count in enumerate(class_counts):
        if count:
            weights[labels == label] = len(labels) / (len(class_counts) * count)
    return weights


def _build_feature_vector(
    img_path: str,
    ocr_text: str,
    amount: float,
    stats: dict | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], list[float]]:
    img_features = extract_image_features(img_path)
    text_features = extract_text_features(ocr_text)
    consistency_features = extract_consistency_features(ocr_text, amount, stats)
    vector = (
        features_to_vector(img_features)
        + text_features_to_vector(text_features)
        + _vectorize(consistency_features, CONSISTENCY_FEATURE_KEYS)
        + [float(amount)]
    )
    return img_features, text_features, consistency_features, vector


def train_anomaly_model(records: list, train_dir: str, model_dir: str):
    """Train a gradient boosting classifier on visual, text, and consistency features."""
    from sklearn.ensemble import GradientBoostingClassifier

    from src.extractor import extract_text

    set_deterministic_seeds(DEFAULT_CONFIG.training.random_state)
    os.makedirs(model_dir, exist_ok=True)

    stats = {
        "vendors": sorted(
            {
                record.get("fields", {}).get("vendor", "").strip()
                for record in records
                if record.get("fields", {}).get("vendor")
            }
        )
    }

    amounts = []
    for record in records:
        total = record.get("fields", {}).get("total")
        if total is None:
            continue
        try:
            amounts.append(float(total))
        except (TypeError, ValueError):
            continue
    if amounts:
        stats.update(
            {
                "amount_mean": float(np.mean(amounts)),
                "amount_std": float(np.std(amounts)),
                "amount_q1": float(np.percentile(amounts, 25)),
                "amount_q3": float(np.percentile(amounts, 75)),
            }
        )

    features: list[list[float]] = []
    labels: list[int] = []

    for record in records:
        img_path = os.path.join(train_dir, record["image_path"])
        try:
            ocr_text = extract_text(img_path) if os.path.exists(img_path) else ""
        except Exception:
            ocr_text = ""

        try:
            amount = float(record.get("fields", {}).get("total") or 0.0)
        except (TypeError, ValueError):
            amount = 0.0

        _, _, _, vector = _build_feature_vector(img_path, ocr_text, amount, stats)
        features.append(vector)
        labels.append(int(record.get("label", {}).get("is_forged", 0)))

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    model = None
    train_accuracy = None
    if len(X) >= 6 and len(set(y.tolist())) >= 2:
        feature_variance = X.std(axis=0)
        useful_features = feature_variance > 1e-6
        if useful_features.sum() >= 3:
            model = GradientBoostingClassifier(
                n_estimators=DEFAULT_CONFIG.training.gradient_boosting_estimators,
                learning_rate=DEFAULT_CONFIG.training.gradient_boosting_learning_rate,
                max_depth=DEFAULT_CONFIG.training.gradient_boosting_max_depth,
                min_samples_leaf=DEFAULT_CONFIG.training.gradient_boosting_min_samples_leaf,
                random_state=DEFAULT_CONFIG.training.random_state,
            )
            sample_weight = _compute_sample_weights(y)
            model.fit(X, y, sample_weight=sample_weight)
            train_accuracy = float(model.score(X, y))
            print(
                f"[anomaly] GradientBoosting trained: {len(X)} samples, "
                f"train acc={train_accuracy:.2f}"
            )
        else:
            print("[anomaly] Features have insufficient variance; using heuristic fallback")
    else:
        print("[anomaly] Not enough labeled data; using heuristic fallback")

    forged_ratio = float(y.mean()) if len(y) else 0.5
    model_path = os.path.join(model_dir, DEFAULT_CONFIG.data.anomaly_model_file_name)
    with open(model_path, "wb") as handle:
        pickle.dump(
            {
                "model": model,
                "model_type": "GradientBoostingClassifier" if model is not None else "heuristic",
                "feature_keys": FEATURE_KEYS
                + TEXT_FEATURE_KEYS
                + CONSISTENCY_FEATURE_KEYS
                + ["amount"],
                "forged_ratio": forged_ratio,
                "train_accuracy": train_accuracy,
            },
            handle,
        )

    return model


def predict_anomaly(
    model_data: dict,
    img_path: str,
    ocr_text: str,
    amount: float,
    stats: dict,
) -> int:
    """Predict whether a document is forged."""
    img_features, text_features, consistency_features, vector = _build_feature_vector(
        img_path,
        ocr_text,
        amount,
        stats,
    )
    model = model_data.get("model")

    if model is not None:
        proba = model.predict_proba(np.asarray([vector], dtype=np.float64))[0]
        forged_prob = float(proba[1]) if len(proba) > 1 else 0.0
        return int(forged_prob >= DEFAULT_CONFIG.training.anomaly_threshold)

    forged_ratio = float(model_data.get("forged_ratio", 0.5))
    return _heuristic_forgery(
        amount=amount,
        stats=stats,
        img_features=img_features,
        text_features=text_features,
        consistency_features=consistency_features,
        forged_ratio=forged_ratio,
    )


def _heuristic_forgery(
    amount: float,
    stats: dict,
    img_features: dict[str, float],
    text_features: dict[str, float],
    consistency_features: dict[str, float],
    forged_ratio: float = 0.5,
) -> int:
    """Rule-based forgery detection when the ML model is unavailable."""
    score = 0.0

    if img_features.get("img_std", 0.0) > 1.0:
        if img_features.get("noise_std", 0.0) > 20.0:
            score += 0.2
        if img_features.get("ela_high_ratio", 0.0) > 0.12:
            score += 0.25
        if img_features.get("ela_block_std", 0.0) > 8.0:
            score += 0.15
        block_var_mean = max(img_features.get("block_var_mean", 0.0), 1.0)
        if img_features.get("block_var_std", 0.0) / block_var_mean < 0.3:
            score += 0.15
        entropy = img_features.get("entropy", 0.0)
        if entropy and (entropy < 3.0 or entropy > 7.0):
            score += 0.1

    if text_features.get("text_length", 0.0) == 0:
        score += 0.1

    score += 0.35 * consistency_features.get("consistency_risk", 0.0)

    amount_std = float(stats.get("amount_std", 0.0) or 0.0)
    amount_mean = float(stats.get("amount_mean", 0.0) or 0.0)
    if amount > 0 and amount_std > 1e-6:
        z_score = abs(amount - amount_mean) / amount_std
        if z_score > 2.0:
            score += 0.2

    threshold = max(0.45, min(0.7, 0.4 + forged_ratio * 0.2))
    return int(score >= threshold)
