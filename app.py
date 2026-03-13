"""Streamlit inspection UI backed by the shared document analysis pipeline."""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DEFAULT_CONFIG
from src.document_io import load_document_pages
from src.pipeline import analyze_document
from src.types import Box, ModelBundle


st.set_page_config(page_title="DocFusion Reviewer", layout="wide")


def load_model_bundle(model_dir: str | None) -> ModelBundle | None:
    if not model_dir:
        return None
    path = Path(model_dir)
    stats_path = path / DEFAULT_CONFIG.data.stats_file_name
    model_path = path / DEFAULT_CONFIG.data.anomaly_model_file_name
    if not stats_path.exists() or not model_path.exists():
        return None

    with stats_path.open() as handle:
        stats = json.load(handle)
    with model_path.open("rb") as handle:
        anomaly_model_data = pickle.load(handle)
    return ModelBundle(stats=stats, anomaly_model_data=anomaly_model_data)


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def draw_boxes(image: Image.Image, boxes: list[tuple[Box, str, str]]) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box, color, label in boxes:
        draw.rectangle([(box.left, box.top), (box.right, box.bottom)], outline=color, width=3)
        draw.text((box.left + 2, max(0, box.top - 14)), label, fill=color)
    return canvas


def render_analysis(name: str, document_path: str, bundle: ModelBundle | None) -> None:
    analysis = analyze_document(document_path, model_bundle=bundle, debug=True)
    pages = load_document_pages(document_path)
    selected_page_index = 0
    if len(pages) > 1:
        selected_page_index = st.selectbox(
            "Page",
            options=[page.page_index for page in pages],
            format_func=lambda index: f"Page {index + 1}",
            key=f"page-selector-{name}-{document_path}",
        )
    page = next(
        (p for p in pages if p.page_index == selected_page_index),
        pages[0],
    )

    field_boxes = [
        (field.box, "#0f766e", f"{field.name}: {field.value or 'n/a'}")
        for field in (
            analysis.extraction.vendor,
            analysis.extraction.date,
            analysis.extraction.total,
        )
        if field.box is not None and field.page_index == selected_page_index
    ]
    suspicious_boxes = [
        (box, "#b91c1c", "suspicious")
        for box in analysis.anomaly.suspicious_regions
        if box.page_index == selected_page_index
    ]

    st.subheader(name)
    left, right = st.columns([1.4, 1])
    with left:
        tabs = st.tabs(["Original", "Fields", "Suspicious"])
        with tabs[0]:
            st.image(page.image, use_container_width=True)
        with tabs[1]:
            st.image(draw_boxes(page.image, field_boxes), use_container_width=True)
        with tabs[2]:
            st.image(draw_boxes(page.image, suspicious_boxes), use_container_width=True)
    with right:
        st.markdown("**Fields**")
        st.json(
            {
                "vendor": analysis.extraction.vendor.to_dict(),
                "date": analysis.extraction.date.to_dict(),
                "total": analysis.extraction.total.to_dict(),
            }
        )
        st.markdown("**Forgery**")
        st.write(
            {
                "score": round(analysis.anomaly.score, 4),
                "is_forged": analysis.anomaly.is_forged,
                "reasons": list(analysis.anomaly.reasons),
            }
        )
        st.markdown("**Document**")
        st.write(
            {
                "pages": analysis.page_count,
                "words": len(analysis.words),
                "lines": len(analysis.lines),
            }
        )
    with st.expander("OCR Text"):
        st.code(analysis.ocr_text or "", language=None)
    with st.expander("Feature Debug"):
        st.json(analysis.anomaly.feature_values)


def main() -> None:
    st.title("DocFusion Document Inspector")
    st.caption("Shared analysis engine for extraction, forgery scoring, PDFs, and overlays.")

    default_model_dir = str((Path.cwd() / "tmp_work" / "model").resolve())
    model_dir = st.sidebar.text_input("Model directory", value=default_model_dir)
    bundle = load_model_bundle(model_dir)
    if bundle is None:
        st.sidebar.info("No trained model bundle found. Using deterministic heuristic anomaly scoring.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        path = save_uploaded_file(uploaded_file)
        render_analysis(uploaded_file.name, path, bundle)


if __name__ == "__main__":
    main()
