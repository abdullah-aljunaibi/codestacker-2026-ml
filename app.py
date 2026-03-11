"""
DocFusion Web UI — Receipt Analysis Dashboard

Upload one or more receipt images to:
1. Extract structured fields (vendor, date, total)
2. Detect potential forgery
3. Inspect OCR confidence, bounding boxes, and ELA overlays
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from io import BytesIO

import streamlit as st
from PIL import Image, ImageChops, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.anomaly import FEATURE_KEYS, TEXT_FEATURE_KEYS, extract_image_features, extract_text_features
from src.extractor import extract_date, extract_total, extract_vendor
from src.ocr import OCRResult, extract_ocr, load_image, preprocess_image

st.set_page_config(
    page_title="DocFusion — Receipt Analyzer",
    page_icon="🔍",
    layout="wide",
)


def compute_anomaly_assessment(img_feats: dict[str, float], threshold: float) -> tuple[float, bool, list[str]]:
    """Score suspicious visual patterns using the project's heuristic fallback."""
    anomaly_score = 0.0
    anomaly_reasons: list[str] = []

    if img_feats.get("img_std", 0) > 1.0:
        if img_feats.get("noise_std", 0) > 20:
            anomaly_score += 0.3
            anomaly_reasons.append("High noise level suggests possible manipulation.")
        if img_feats.get("block_var_std", 0) > 0:
            bv_ratio = img_feats["block_var_std"] / max(img_feats["block_var_mean"], 1)
            if bv_ratio < 0.3:
                anomaly_score += 0.2
                anomaly_reasons.append("Uniform block variance can indicate copy-paste edits.")
        entropy = img_feats.get("entropy", 0)
        if entropy > 0 and (entropy < 3.0 or entropy > 7.0):
            anomaly_score += 0.15
            anomaly_reasons.append("Histogram entropy is outside the expected document range.")
        if img_feats.get("edge_mean", 0) > 50:
            anomaly_score += 0.1
            anomaly_reasons.append("High edge density may reflect tampered regions or heavy resampling.")
        if img_feats.get("ela_high_ratio", 0) > 0.2:
            anomaly_score += 0.25
            anomaly_reasons.append("ELA highlights unusually strong recompression artifacts.")
    else:
        anomaly_reasons.append("Image appears blank or too flat for reliable visual analysis.")

    return anomaly_score, anomaly_score >= threshold, anomaly_reasons


def build_ocr_overlay(image_path: str, ocr_result: OCRResult) -> Image.Image:
    """Render OCR word boxes on the preprocessed image used for OCR."""
    base = preprocess_image(load_image(image_path)).convert("RGB")
    draw = ImageDraw.Draw(base)

    for word in ocr_result.words:
        if None in (word.left, word.top, word.width, word.height):
            continue
        confidence = word.confidence if word.confidence is not None else 0.0
        if confidence >= 85:
            color = "#16a34a"
        elif confidence >= 60:
            color = "#f59e0b"
        else:
            color = "#dc2626"
        draw.rectangle(
            [(word.left, word.top), (word.right, word.bottom)],
            outline=color,
            width=2,
        )
    return base


def build_ela_overlay(image_path: str, quality: int = 90) -> tuple[Image.Image | None, float]:
    """Blend ELA intensity into the original image for visual inspection."""
    try:
        with Image.open(image_path) as source:
            original = source.convert("RGB")
    except Exception:
        return None, 0.0

    buffer = BytesIO()
    try:
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            diff = ImageChops.difference(original, recompressed.convert("RGB"))
    except Exception:
        return None, 0.0

    ela_gray = diff.convert("L")
    max_diff = max(1, ela_gray.getextrema()[1])
    amplified = ela_gray.point(lambda value: min(255, int(value * 255 / max_diff)))
    heat = Image.merge("RGBA", (amplified, Image.new("L", amplified.size, 32), Image.new("L", amplified.size, 32), amplified))
    overlay = Image.alpha_composite(original.convert("RGBA"), heat)
    return overlay.convert("RGB"), min(1.0, max_diff / 255.0)


def mean_ocr_confidence(ocr_result: OCRResult) -> float:
    confidences = [word.confidence for word in ocr_result.words if word.confidence is not None]
    if not confidences:
        return 0.0
    return sum(confidences) / (100.0 * len(confidences))


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def render_analysis(name: str, image_path: str, sample_record: dict | None = None) -> dict[str, object]:
    image = load_image(image_path)
    ocr_result = extract_ocr(image_path)
    ocr_text = ocr_result.text

    vendor = extract_vendor(ocr_text)
    date = extract_date(ocr_text)
    total = extract_total(ocr_text)

    img_feats = extract_image_features(image_path)
    txt_feats = extract_text_features(ocr_text)
    anomaly_score, is_forged, anomaly_reasons = compute_anomaly_assessment(img_feats, confidence_threshold)
    ocr_confidence = mean_ocr_confidence(ocr_result)
    ela_overlay, ela_strength = build_ela_overlay(image_path)
    ocr_overlay = build_ocr_overlay(image_path, ocr_result)

    st.markdown(f"## {name}")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        preview_tabs = st.tabs(["Original", "OCR Boxes", "ELA Overlay"])
        with preview_tabs[0]:
            st.image(image, use_container_width=True)
        with preview_tabs[1]:
            st.image(ocr_overlay, use_container_width=True)
        with preview_tabs[2]:
            if ela_overlay is not None:
                st.image(ela_overlay, use_container_width=True)
            else:
                st.info("ELA overlay is unavailable for this image.")

    with col2:
        st.subheader("Extracted Fields")
        st.metric("Vendor", vendor or "Not detected")
        st.metric("Date", date or "Not detected")
        st.metric("Total", f"${total}" if total else "Not detected")

        st.markdown("---")
        st.subheader("Confidence Meters")
        st.caption(f"Forgery confidence: {anomaly_score:.0%}")
        st.progress(min(anomaly_score, 1.0))
        st.caption(f"OCR confidence: {ocr_confidence:.0%}")
        st.progress(ocr_confidence)
        st.caption(f"ELA intensity: {ela_strength:.0%}")
        st.progress(ela_strength)

        st.markdown("---")
        st.subheader("Forgery Status")
        if is_forged:
            st.error(f"Suspicious at threshold {confidence_threshold:.2f}")
        else:
            st.success(f"Within threshold {confidence_threshold:.2f}")

        if sample_record is not None:
            st.markdown("---")
            st.subheader("Sample Ground Truth")
            st.json(sample_record.get("fields", {}))
            st.json(sample_record.get("label", {}))

    st.subheader("Anomaly Indicators")
    if anomaly_reasons:
        for reason in anomaly_reasons:
            st.markdown(f"- {reason}")
    else:
        st.markdown("- No strong anomaly indicators triggered.")

    if show_ocr:
        st.subheader("Raw OCR Output")
        if ocr_text.strip():
            st.code(ocr_text, language=None)
        else:
            st.info("No text detected.")

    if show_features:
        st.subheader("Feature Debug Panel")
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.markdown("**Visual Features**")
            for key in FEATURE_KEYS:
                value = img_feats.get(key, 0)
                st.text(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        with feat_col2:
            st.markdown("**Text Features**")
            for key in TEXT_FEATURE_KEYS:
                value = txt_feats.get(key, 0)
                st.text(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    st.markdown("---")

    return {
        "file": name,
        "vendor": vendor or "",
        "date": date or "",
        "total": total or "",
        "ocr_confidence": round(ocr_confidence, 3),
        "forgery_score": round(anomaly_score, 3),
        "ela_intensity": round(ela_strength, 3),
        "status": "Suspicious" if is_forged else "Genuine",
    }


st.title("🔍 DocFusion — Intelligent Document Processor")
st.markdown(
    "Upload one or more scanned receipts to extract structured fields, inspect OCR boxes, and review forgery signals."
)

st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Forgery Sensitivity",
    0.0,
    1.0,
    0.5,
    0.05,
    help="Lower values mark more receipts as suspicious.",
)
show_ocr = st.sidebar.checkbox("Show Raw OCR Text", value=False)
show_features = st.sidebar.checkbox("Show Image Features", value=False)

st.sidebar.header("📂 Sample Data")
dummy_dir = os.path.join(os.path.dirname(__file__), "dummy_data", "train", "images")
if os.path.exists(dummy_dir):
    samples = sorted(os.listdir(dummy_dir))
    selected_sample = st.sidebar.selectbox("Or pick a sample:", ["None"] + samples)
else:
    selected_sample = "None"

uploaded_files = st.file_uploader(
    "Upload Receipt Images",
    type=["png", "jpg", "jpeg", "tiff", "bmp"],
    accept_multiple_files=True,
    help="Supported: PNG, JPG, JPEG, TIFF, BMP",
)

sample_record = None
sample_path = None
if selected_sample != "None":
    sample_path = os.path.join(dummy_dir, selected_sample)
    train_jsonl = os.path.join(os.path.dirname(__file__), "dummy_data", "train", "train.jsonl")
    if os.path.exists(train_jsonl):
        with open(train_jsonl, encoding="utf-8") as handle:
            records = {}
            for line in handle:
                record = json.loads(line)
                records[record["id"]] = record
        receipt_id = os.path.splitext(selected_sample)[0]
        sample_record = records.get(receipt_id)

analysis_targets: list[tuple[str, str, dict | None, bool]] = []
temp_paths: list[str] = []

for uploaded_file in uploaded_files or []:
    temp_path = save_uploaded_file(uploaded_file)
    temp_paths.append(temp_path)
    analysis_targets.append((uploaded_file.name, temp_path, None, False))

if sample_path is not None:
    analysis_targets.append((selected_sample, sample_path, sample_record, True))

try:
    if analysis_targets:
        batch_results: list[dict[str, object]] = []
        st.subheader("Batch Queue")
        st.caption(f"Processing {len(analysis_targets)} receipt(s).")
        for index, (name, image_path, record, is_sample) in enumerate(analysis_targets, start=1):
            label = f"{index}. {name}"
            with st.expander(label, expanded=index == 1):
                batch_results.append(render_analysis(name, image_path, sample_record=record if is_sample else None))
        st.subheader("Batch Summary")
        st.dataframe(batch_results, use_container_width=True)
    else:
        st.info("Upload one or more receipt images or select a sample from the sidebar to get started.")
finally:
    for temp_path in temp_paths:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "DocFusion — CodeStacker 2026 ML Challenge | Abdullah Al Junaibi"
    "</div>",
    unsafe_allow_html=True,
)
