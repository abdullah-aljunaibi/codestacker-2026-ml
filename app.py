"""
DocFusion Web UI — Receipt Analysis Dashboard

Upload a receipt image to:
1. Extract structured fields (vendor, date, total)
2. Detect potential forgery
3. View OCR output and anomaly indicators
"""
import json
import os
import sys
import tempfile

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extractor import extract_fields, extract_text, preprocess_image
from src.anomaly import (
    extract_image_features, extract_text_features,
    FEATURE_KEYS, TEXT_FEATURE_KEYS,
)

st.set_page_config(
    page_title="DocFusion — Receipt Analyzer",
    page_icon="🔍",
    layout="wide",
)

# --- Header ---
st.title("🔍 DocFusion — Intelligent Document Processor")
st.markdown(
    "Upload a scanned receipt to extract structured fields and detect potential forgery."
)

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Forgery Sensitivity", 0.0, 1.0, 0.5, 0.05,
    help="Lower = more aggressive forgery detection"
)
show_ocr = st.sidebar.checkbox("Show Raw OCR Text", value=False)
show_features = st.sidebar.checkbox("Show Image Features", value=False)

# --- Main ---
uploaded = st.file_uploader(
    "Upload Receipt Image", type=["png", "jpg", "jpeg", "tiff", "bmp"],
    help="Supported: PNG, JPG, JPEG, TIFF, BMP"
)

# Also allow selecting from dummy data
st.sidebar.header("📂 Sample Data")
dummy_dir = os.path.join(os.path.dirname(__file__), "dummy_data", "train", "images")
if os.path.exists(dummy_dir):
    samples = sorted(os.listdir(dummy_dir))
    selected_sample = st.sidebar.selectbox("Or pick a sample:", ["None"] + samples)
else:
    selected_sample = "None"


def analyze_receipt(img_path: str, img: Image.Image):
    """Run full analysis on a receipt image."""

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Receipt Image")
        st.image(img, use_container_width=True)

    # Extract fields
    result = extract_fields(img_path)
    vendor = result.get("vendor")
    date = result.get("date")
    total = result.get("total")
    ocr_text = result.get("_ocr_text", "")

    # Image features for anomaly
    img_feats = extract_image_features(img_path)
    txt_feats = extract_text_features(ocr_text)

    # Simple anomaly scoring
    anomaly_score = 0.0
    anomaly_reasons = []

    if img_feats.get('img_std', 0) > 1.0:
        # Has actual image content
        if img_feats.get('noise_std', 0) > 20:
            anomaly_score += 0.3
            anomaly_reasons.append("🔴 High noise level (possible manipulation)")
        if img_feats.get('block_var_std', 0) > 0:
            bv_ratio = img_feats['block_var_std'] / max(img_feats['block_var_mean'], 1)
            if bv_ratio < 0.3:
                anomaly_score += 0.2
                anomaly_reasons.append("🟡 Uniform block variance (copy-paste indicator)")
        entropy = img_feats.get('entropy', 0)
        if entropy > 0 and (entropy < 3.0 or entropy > 7.0):
            anomaly_score += 0.15
            anomaly_reasons.append("🟡 Unusual histogram entropy")
        if img_feats.get('edge_mean', 0) > 50:
            anomaly_score += 0.1
            anomaly_reasons.append("🟡 High edge density")
    else:
        anomaly_reasons.append("⚪ Blank/empty image — cannot analyze visually")

    is_forged = anomaly_score >= confidence_threshold

    with col2:
        st.subheader("📊 Extracted Fields")

        # Vendor
        if vendor:
            st.metric("🏪 Vendor", vendor)
        else:
            st.metric("🏪 Vendor", "Not detected", delta="⚠️", delta_color="off")

        # Date
        if date:
            st.metric("📅 Date", date)
        else:
            st.metric("📅 Date", "Not detected", delta="⚠️", delta_color="off")

        # Total
        if total:
            st.metric("💰 Total", f"${total}")
        else:
            st.metric("💰 Total", "Not detected", delta="⚠️", delta_color="off")

        # Anomaly status
        st.markdown("---")
        st.subheader("🔎 Anomaly Status")

        if is_forged:
            st.error(f"⚠️ SUSPICIOUS — Anomaly Score: {anomaly_score:.2f}")
        else:
            st.success(f"✅ GENUINE — Anomaly Score: {anomaly_score:.2f}")

        # Progress bar for anomaly score
        st.progress(min(anomaly_score, 1.0))

    # Anomaly reasons
    if anomaly_reasons:
        st.subheader("🔬 Anomaly Indicators")
        for reason in anomaly_reasons:
            st.markdown(f"- {reason}")

    # OCR text
    if show_ocr:
        st.subheader("📝 Raw OCR Output")
        if ocr_text.strip():
            st.code(ocr_text, language=None)
        else:
            st.info("No text detected (blank image or OCR failed)")

    # Image features
    if show_features:
        st.subheader("📐 Image Features")
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.markdown("**Visual Features**")
            for k in FEATURE_KEYS:
                v = img_feats.get(k, 0)
                st.text(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        with feat_col2:
            st.markdown("**Text Features**")
            for k in TEXT_FEATURE_KEYS:
                v = txt_feats.get(k, 0)
                st.text(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# Process uploaded file
if uploaded is not None:
    img = Image.open(uploaded)
    # Save to temp file for analysis
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        analyze_receipt(tmp.name, img)
        os.unlink(tmp.name)

elif selected_sample != "None":
    img_path = os.path.join(dummy_dir, selected_sample)
    img = Image.open(img_path)

    # Show ground truth if available
    train_jsonl = os.path.join(os.path.dirname(__file__), "dummy_data", "train", "train.jsonl")
    if os.path.exists(train_jsonl):
        with open(train_jsonl) as f:
            records = {json.loads(l)["id"]: json.loads(l) for l in f}
        rid = selected_sample.replace(".png", "")
        if rid in records:
            r = records[rid]
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Ground Truth:**")
            st.sidebar.json(r["fields"])
            st.sidebar.json(r["label"])

    analyze_receipt(img_path, img)

else:
    st.info("👆 Upload a receipt image or select a sample from the sidebar to get started.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "DocFusion — CodeStacker 2026 ML Challenge | Abdullah Al Junaibi"
    "</div>",
    unsafe_allow_html=True,
)
