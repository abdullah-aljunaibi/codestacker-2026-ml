"""
Anomaly detection for receipt forgery.

Features:
  - Image-level: edge density, noise variance, contrast, JPEG artifact patterns
  - OCR-level: confidence scores, text density
  - Statistical: amount z-score, vendor frequency

Uses sklearn Random Forest trained on extracted features.
"""
import json
import os
import pickle
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageStat


def extract_image_features(image_path: str) -> dict:
    """Extract visual features from a receipt image for forgery detection."""
    try:
        img = Image.open(image_path)
    except Exception:
        return _empty_features()

    # Convert to grayscale for analysis
    gray = img.convert('L')
    w, h = gray.size

    if w == 0 or h == 0:
        return _empty_features()

    pixels = np.array(gray, dtype=np.float64)

    # Skip blank/empty images
    if pixels.std() < 1.0:
        return _empty_features()

    # 1. Basic statistics
    mean_val = float(pixels.mean())
    std_val = float(pixels.std())
    median_val = float(np.median(pixels))

    # 2. Edge density (Laplacian-like)
    edge_img = gray.filter(ImageFilter.FIND_EDGES)
    edge_pixels = np.array(edge_img, dtype=np.float64)
    edge_mean = float(edge_pixels.mean())
    edge_std = float(edge_pixels.std())

    # 3. Noise estimation (high-frequency content)
    blur = gray.filter(ImageFilter.GaussianBlur(radius=2))
    blur_pixels = np.array(blur, dtype=np.float64)
    noise = pixels - blur_pixels
    noise_std = float(noise.std())
    noise_mean = float(np.abs(noise).mean())

    # 4. Contrast and dynamic range
    p5 = float(np.percentile(pixels, 5))
    p95 = float(np.percentile(pixels, 95))
    dynamic_range = p95 - p5

    # 5. Histogram entropy
    hist, _ = np.histogram(pixels.flatten(), bins=64, range=(0, 256))
    hist_norm = hist / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))

    # 6. Block-level variance (detect copy-paste)
    block_size = max(16, min(w, h) // 8)
    block_vars = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = pixels[y:y+block_size, x:x+block_size]
            block_vars.append(float(block.var()))

    block_var_std = float(np.std(block_vars)) if block_vars else 0
    block_var_mean = float(np.mean(block_vars)) if block_vars else 0

    # 7. Aspect ratio and size
    aspect = w / h if h > 0 else 1
    total_pixels = w * h

    return {
        'img_mean': mean_val,
        'img_std': std_val,
        'img_median': median_val,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'noise_std': noise_std,
        'noise_mean': noise_mean,
        'dynamic_range': dynamic_range,
        'entropy': entropy,
        'block_var_std': block_var_std,
        'block_var_mean': block_var_mean,
        'aspect_ratio': aspect,
        'total_pixels': total_pixels,
        'p5': p5,
        'p95': p95,
    }


def _empty_features() -> dict:
    """Return zero features for blank/missing images."""
    return {
        'img_mean': 0, 'img_std': 0, 'img_median': 0,
        'edge_mean': 0, 'edge_std': 0,
        'noise_std': 0, 'noise_mean': 0,
        'dynamic_range': 0, 'entropy': 0,
        'block_var_std': 0, 'block_var_mean': 0,
        'aspect_ratio': 1, 'total_pixels': 0,
        'p5': 0, 'p95': 0,
    }


FEATURE_KEYS = list(_empty_features().keys())


def features_to_vector(features: dict) -> list:
    """Convert feature dict to ordered vector."""
    return [features.get(k, 0) for k in FEATURE_KEYS]


def extract_text_features(ocr_text: str) -> dict:
    """Extract text-based features for anomaly detection."""
    if not ocr_text or not ocr_text.strip():
        return {
            'text_length': 0,
            'line_count': 0,
            'digit_ratio': 0,
            'alpha_ratio': 0,
            'special_ratio': 0,
            'avg_line_length': 0,
            'empty_line_ratio': 0,
        }

    lines = ocr_text.split('\n')
    non_empty = [l for l in lines if l.strip()]
    total_chars = len(ocr_text)

    digits = sum(c.isdigit() for c in ocr_text)
    alpha = sum(c.isalpha() for c in ocr_text)
    special = total_chars - digits - alpha - ocr_text.count(' ') - ocr_text.count('\n')

    return {
        'text_length': total_chars,
        'line_count': len(non_empty),
        'digit_ratio': digits / total_chars if total_chars > 0 else 0,
        'alpha_ratio': alpha / total_chars if total_chars > 0 else 0,
        'special_ratio': special / total_chars if total_chars > 0 else 0,
        'avg_line_length': np.mean([len(l) for l in non_empty]) if non_empty else 0,
        'empty_line_ratio': (len(lines) - len(non_empty)) / len(lines) if lines else 0,
    }


TEXT_FEATURE_KEYS = list(extract_text_features("").keys())


def text_features_to_vector(features: dict) -> list:
    return [features.get(k, 0) for k in TEXT_FEATURE_KEYS]


def train_anomaly_model(records: list, train_dir: str, model_dir: str):
    """
    Train a Random Forest classifier on image + text features.
    Falls back to a heuristic model if not enough data or sklearn unavailable.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = []
    y = []

    for r in records:
        img_path = os.path.join(train_dir, r["image_path"])

        # Image features
        img_feats = extract_image_features(img_path)
        img_vec = features_to_vector(img_feats)

        # Text features (from OCR)
        from src.extractor import extract_text
        try:
            ocr_text = extract_text(img_path) if os.path.exists(img_path) else ""
        except:
            ocr_text = ""
        txt_feats = extract_text_features(ocr_text)
        txt_vec = text_features_to_vector(txt_feats)

        # Amount features
        total = r.get("fields", {}).get("total")
        amount_val = float(total) if total else 0

        feature_vec = img_vec + txt_vec + [amount_val]
        X.append(feature_vec)
        y.append(r.get("label", {}).get("is_forged", 0))

    X = np.array(X)
    y = np.array(y)

    # Check if we have enough data and variance
    if len(set(y)) < 2 or len(X) < 4:
        print("[anomaly] Not enough labeled data — using heuristic fallback")
        model = None
    else:
        # Check if features have any variance (blank images = all zeros)
        feature_variance = X.std(axis=0)
        useful_features = feature_variance > 1e-6

        if useful_features.sum() < 2:
            print("[anomaly] Features have no variance (blank images?) — supplementing with synthetic data")
            model = None  # default
            try:
                from src.synthetic import generate_records
                syn_records = generate_records(n=300, seed=42)
                X_syn, y_syn = [], []
                feat_len = len(FEATURE_KEYS) + len(TEXT_FEATURE_KEYS) + 1
                for sr in syn_records:
                    amt = float(sr.get("total") or 0)
                    has_vendor = 1.0 if sr.get("vendor") else 0.0
                    has_date = 1.0 if sr.get("date") else 0.0
                    is_round = 1.0 if amt > 0 and amt % 50 == 0 and amt >= 100 else 0.0
                    high_amt = 1.0 if amt > 500 else 0.0
                    low_amt = 1.0 if 0 < amt < 2 else 0.0
                    fv = [amt, has_vendor, has_date, is_round, high_amt, low_amt]
                    # Pad or truncate to feat_len
                    fv = fv + [0.0] * max(0, feat_len - len(fv))
                    fv = fv[:feat_len]
                    X_syn.append(fv)
                    y_syn.append(sr["label"]["is_forged"])
                X_new = np.array(X_syn)
                y_new = np.array(y_syn)
                if len(set(y_new)) >= 2:
                    print(f"[anomaly] Synthetic data loaded: {len(X_new)} records ({sum(y_new)} forged)")
                    model = RandomForestClassifier(
                        n_estimators=100, max_depth=5,
                        min_samples_leaf=2, random_state=42,
                        class_weight='balanced',
                    )
                    model.fit(X_new, y_new)
                    train_acc = model.score(X_new, y_new)
                    print(f"[anomaly] RF trained on synthetic: {len(X_new)} samples, train acc={train_acc:.2f}")
            except Exception as e:
                print(f"[anomaly] Synthetic training failed: {e} — using heuristic fallback")
                model = None
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
            )
            model.fit(X, y)
            train_acc = model.score(X, y)
            print(f"[anomaly] RF trained: {len(X)} samples, train acc={train_acc:.2f}")

    # Compute forged ratio from training data for prior
    forged_count = sum(1 for r in records if r.get("label", {}).get("is_forged", 0) == 1)
    forged_ratio = forged_count / len(records) if records else 0.5

    # Compute vendor frequency for heuristic
    from collections import Counter
    vendor_counts = Counter(
        r.get("vendor") or r.get("fields", {}).get("vendor")
        for r in records
        if (r.get("vendor") or r.get("fields", {}).get("vendor"))
    )

    # Compute amount stats for heuristic
    amounts = [float(r.get("total") or r.get("fields", {}).get("total") or 0) for r in records]
    amounts = [a for a in amounts if a > 0]
    amount_stats = {
        "amount_mean": float(np.mean(amounts)) if amounts else 0,
        "amount_std": float(np.std(amounts)) if amounts else 1,
    }

    # Save model
    model_path = os.path.join(model_dir, "anomaly_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_keys': FEATURE_KEYS + TEXT_FEATURE_KEYS + ['amount'],
            'forged_ratio': forged_ratio,
            'vendor_counts': dict(vendor_counts),
            'amount_stats': amount_stats,
        }, f)

    return model


def predict_anomaly(model_data: dict, img_path: str, ocr_text: str,
                    amount: float, stats: dict,
                    vendor: str = None, date_str: str = None) -> int:
    """Predict whether a document is forged."""
    model = model_data.get('model')

    # Image features
    img_feats = extract_image_features(img_path)
    img_vec = features_to_vector(img_feats)

    # Text features
    txt_feats = extract_text_features(ocr_text)
    txt_vec = text_features_to_vector(txt_feats)

    feature_vec = img_vec + txt_vec + [amount]

    if model is not None:
        X = np.array([feature_vec])
        # Use probability threshold adjusted by base rate
        proba = model.predict_proba(X)[0]
        forged_prob = proba[1] if len(proba) > 1 else 0
        threshold = 0.45  # Slightly below 0.5 to catch more forgeries
        return 1 if forged_prob >= threshold else 0

    # Heuristic fallback — use rich statistical rules
    forged_ratio = model_data.get('forged_ratio', 0.5)
    vendor_counts = model_data.get('vendor_counts', {})
    # Use stored amount stats if available (more reliable than per-predict stats)
    stored_stats = model_data.get('amount_stats', stats)
    merged_stats = {**stats, **stored_stats}
    return _heuristic_forgery(
        amount, merged_stats, img_feats, txt_feats,
        forged_ratio, vendor=vendor, date_str=date_str, vendor_counts=vendor_counts
    )


def _heuristic_forgery(amount: float, stats: dict,
                       img_feats: dict, txt_feats: dict,
                       forged_ratio: float = 0.5,
                       vendor: str = None,
                       date_str: str = None,
                       vendor_counts: dict = None) -> int:
    """
    Rule-based forgery detection when ML model isn't available.
    Uses statistical signals: amount outliers, round numbers, vendor rarity,
    missing fields, and date anomalies.
    """
    score = 0.0
    mean = stats.get("amount_mean", 0)
    std = stats.get("amount_std", 1)

    has_image = img_feats.get('img_std', 0) > 1.0

    # === IMAGE SIGNALS (when real images exist) ===
    if has_image:
        if img_feats.get('noise_std', 0) > 20:
            score += 1.5
        bv_std = img_feats.get('block_var_std', 0)
        bv_mean = img_feats.get('block_var_mean', 1)
        if bv_mean > 0 and bv_std / bv_mean < 0.3:
            score += 1.0
        entropy = img_feats.get('entropy', 0)
        if entropy > 0 and (entropy < 3.0 or entropy > 7.0):
            score += 0.5
        if img_feats.get('edge_mean', 0) > 50:
            score += 0.5
        return 1 if score >= 1.5 else 0

    # === STATISTICAL RULES (text/metadata only) ===

    # 1. Amount z-score outlier (strong signal)
    if std > 0 and amount > 0:
        z = abs(amount - mean) / std
        if z > 2.5:
            score += 2.0   # strong outlier
        elif z > 1.5:
            score += 0.8

    # 2. Suspiciously round number (e.g. exactly $100, $500, $1000)
    if amount > 0 and amount % 50 == 0 and amount >= 100:
        score += 1.2

    # 3. Extreme high amount (possible inflated receipt)
    if mean > 0 and amount > mean * 5:
        score += 1.5

    # 4. Extremely low amount (under-reporting)
    if 0 < amount < 1.0:
        score += 1.0

    # 5. Missing vendor (suspicious)
    if not vendor or vendor.strip() == "":
        score += 1.0

    # 6. Missing date (suspicious)
    if not date_str or date_str.strip() == "":
        score += 0.8

    # 7. Rare/unknown vendor (seen < 2 times in training)
    if vendor and vendor_counts:
        count = vendor_counts.get(vendor, 0)
        if count == 0:
            score += 1.2   # never seen before
        elif count == 1:
            score += 0.5   # only once

    # 8. Weekend transaction date (mild signal)
    if date_str:
        try:
            from datetime import datetime
            d = datetime.strptime(date_str[:10], "%Y-%m-%d")
            if d.weekday() >= 5:  # Saturday=5, Sunday=6
                score += 0.3
        except Exception:
            pass

    # Threshold: score >= 2.0 → forged
    # This gives high precision while still catching obvious forgeries
    return 1 if score >= 2.0 else 0
