"""Consistency signals between OCR text, extracted totals, and training stats."""

from __future__ import annotations

import re

import numpy as np


_AMOUNT_PATTERN = re.compile(r"\d+(?:[.,]\d{2})?")


def _empty_consistency_features() -> dict[str, float]:
    return {
        "amount_in_text": 0.0,
        "amount_zscore": 0.0,
        "amount_iqr_gap": 0.0,
        "known_vendor_hit": 0.0,
        "numeric_line_ratio": 0.0,
        "amount_token_gap": 0.0,
        "consistency_risk": 0.0,
    }


CONSISTENCY_FEATURE_KEYS = list(_empty_consistency_features().keys())


def _normalize_amount_token(token: str) -> str:
    normalized = token.replace(",", ".").strip()
    return normalized.lstrip("0") or "0"


def _format_amount_variants(amount: float) -> set[str]:
    if amount <= 0:
        return set()
    return {
        _normalize_amount_token(f"{amount:.2f}"),
        _normalize_amount_token(f"{amount:.1f}"),
        _normalize_amount_token(f"{int(round(amount))}"),
    }


def extract_consistency_features(
    ocr_text: str,
    amount: float,
    stats: dict | None,
) -> dict[str, float]:
    """Build lightweight consistency features for anomaly detection."""
    features = _empty_consistency_features()
    if stats is None:
        stats = {}

    text = (ocr_text or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lower_text = text.lower()

    amount_tokens = {
        _normalize_amount_token(match.group(0))
        for match in _AMOUNT_PATTERN.finditer(text)
    }
    amount_variants = _format_amount_variants(amount)
    if amount_variants:
        features["amount_in_text"] = float(bool(amount_tokens & amount_variants))
        features["amount_token_gap"] = float(
            min(
                (
                    abs(float(token) - amount)
                    for token in amount_tokens
                    if token.replace(".", "", 1).isdigit()
                ),
                default=amount,
            )
        )

    mean = float(stats.get("amount_mean", 0.0) or 0.0)
    std = float(stats.get("amount_std", 0.0) or 0.0)
    if amount > 0 and std > 1e-6:
        features["amount_zscore"] = abs(amount - mean) / std

    q1 = float(stats.get("amount_q1", 0.0) or 0.0)
    q3 = float(stats.get("amount_q3", 0.0) or 0.0)
    iqr = q3 - q1
    if amount > 0 and iqr > 1e-6:
        if amount < q1:
            features["amount_iqr_gap"] = (q1 - amount) / iqr
        elif amount > q3:
            features["amount_iqr_gap"] = (amount - q3) / iqr

    vendors = [str(v).strip().lower() for v in stats.get("vendors", []) if str(v).strip()]
    if lower_text and any(vendor in lower_text for vendor in vendors):
        features["known_vendor_hit"] = 1.0

    if lines:
        numeric_lines = sum(any(char.isdigit() for char in line) for line in lines)
        features["numeric_line_ratio"] = numeric_lines / len(lines)

    features["consistency_risk"] = float(
        np.clip(
            0.35 * (1.0 - features["amount_in_text"])
            + 0.25 * min(features["amount_zscore"] / 3.0, 1.0)
            + 0.2 * min(features["amount_iqr_gap"], 1.0)
            + 0.1 * (1.0 - features["known_vendor_hit"])
            + 0.1 * min(features["amount_token_gap"] / max(amount, 1.0), 1.0),
            0.0,
            1.0,
        )
    )
    return features
