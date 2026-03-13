<![CDATA[<div align="center">

# 🔍 DocFusion Initiative

### Intelligent Document Processing & Forgery Detection Pipeline

**Rihal CodeStacker 2026 — ML Track**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-4285F4?logo=google&logoColor=white)](https://github.com/tesseract-ocr/tesseract)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Extract fields from receipts. Detect forged documents. Explain why they're suspicious.*
*All in under half a second — no GPU required.*

---

**Author:** Abdullah Al Junaibi · Muscat, Oman

</div>

---

## Table of Contents

- [The Problem](#the-problem)
- [The Approach](#the-approach)
- [Architecture](#architecture)
- [Pipeline Deep Dive](#pipeline-deep-dive)
- [Anomaly Detection — 53 Features](#anomaly-detection--53-features)
- [Design Decisions](#design-decisions--why-not-just-use-a-transformer)
- [Performance Benchmarks](#performance-benchmarks)
- [Getting Started](#getting-started)
- [Docker](#docker)
- [Web UI](#web-ui)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Data Sources & Training](#data-sources--training)
- [Notebooks](#notebooks)
- [Acknowledgments](#acknowledgments)

---

## The Problem

Businesses in the GCC and globally process thousands of receipts, invoices, and expense documents daily. Two critical questions arise every time:

1. **What does this document say?** — Extract vendor, date, and total amount accurately.
2. **Can we trust it?** — Detect forged or tampered documents before they enter the financial pipeline.

DocFusion answers both — as a single, unified pipeline that runs fast enough for real-time use and light enough for any machine.

---

## The Approach

> *"The simplest solution that works is the best solution."*

DocFusion is an **OCR-first pipeline** — no heavy vision transformers, no GPU dependencies, no cloud API calls. Every document flows through the same engine whether it's invoked by the evaluation harness or the interactive web UI:

```
Document → Load → Preprocess → OCR → Extract Fields → Detect Anomalies → Localize → Summarize
```

The classifier is a **53-feature GradientBoosting model** trained on image forensics (ELA), OCR statistics, field-level confidence signals, and cross-field consistency checks. It ships as a lightweight pickle artifact — no model downloads, no inference servers.

**Why this matters:** In production, a document processing system that needs a GPU cluster isn't a document processing system — it's a bottleneck. DocFusion processes a document in **~0.48 seconds** on a single CPU core.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DocFusion Pipeline                           │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌─────────┐   ┌─────────────┐  │
│  │ Document  │   │              │   │         │   │    Field     │  │
│  │  Loading  │──▶│ Preprocessing│──▶│   OCR   │──▶│  Extraction  │  │
│  │          │   │              │   │         │   │             │  │
│  │ IMG/PDF  │   │ Gray/Contrast│   │Tesseract│   │Vendor/Date/ │  │
│  │ PyMuPDF  │   │ Resize/Deskew│   │Word-box │   │Total ranking│  │
│  └──────────┘   └──────────────┘   └────┬────┘   └──────┬──────┘  │
│                                         │                │         │
│                                         ▼                ▼         │
│                              ┌──────────────────────────────┐      │
│                              │     Feature Engineering      │      │
│                              │                              │      │
│                              │  Image ─── ELA ─── Text ──── │      │
│                              │  Consistency ── Field-Local   │      │
│                              │  Conflict ── 53 features      │      │
│                              └──────────────┬───────────────┘      │
│                                             │                      │
│                                             ▼                      │
│  ┌───────────────────┐   ┌──────────────────────────────────┐      │
│  │   NLG Summary     │◀──│    GradientBoosting Classifier   │      │
│  │                   │   │                                  │      │
│  │ Human-readable    │   │  Score → Threshold → Forged?     │      │
│  │ anomaly report    │   │  F1-calibrated from OOF preds    │      │
│  └───────────────────┘   └──────────────┬───────────────────┘      │
│                                         │                          │
│                                         ▼                          │
│                              ┌────────────────────┐                │
│                              │    Suspicious       │                │
│                              │    Localization     │                │
│                              │                    │                │
│                              │ ELA hotspots       │                │
│                              │ Low-conf OCR boxes │                │
│                              │ Low-conf fields    │                │
│                              │ Conflicting amounts│                │
│                              └────────────────────┘                │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                          Entry Points                               │
│                                                                     │
│   solution.py (Harness)          app.py (Streamlit UI)             │
│         │                              │                            │
│         └──────── analyze_document() ──┘                            │
│                   (shared engine)                                    │
│                   src/pipeline.py                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** `solution.py` and `app.py` both call `analyze_document()` from `src/pipeline.py`. One codebase, one truth, zero divergence.

---

## Pipeline Deep Dive

### 1. Document Loading (`src/document_io.py`)

| Input      | Method                                    |
|------------|-------------------------------------------|
| JPEG / PNG | Direct load via Pillow                    |
| PDF        | Page rasterization via PyMuPDF (fitz)     |

PDFs are converted to images at sufficient DPI for OCR — no text-layer extraction, because forged PDFs often have manipulated text layers while the rendered image tells the truth.

### 2. Preprocessing (`src/preprocessing.py`)

```
RGB → Grayscale → Contrast normalization → Resize (if needed) → Lightweight deskew
```

The preprocessing is deliberately minimal. Heavy transforms destroy forensic signals — we preserve as much of the original pixel data as possible while making it OCR-friendly.

### 3. OCR (`src/ocr.py`)

Tesseract runs with **word-level bounding boxes and confidence scores** — not just raw text. This gives us:

- **Spatial layout** — where each word sits on the page
- **Per-word confidence** — low-confidence regions are forensic signals
- **Line reconstruction** — words grouped into logical lines for field extraction

### 4. Field Extraction (`src/extractor.py`)

This is where domain knowledge matters most. Fields aren't just regex matches — they're **ranked candidates**:

| Field    | Extraction Strategy                                                  |
|----------|----------------------------------------------------------------------|
| **Total**   | Hardened amount regex that rejects dates, phone numbers, and invoice IDs. Subtotal+tax consistency cross-check. Largest-plausible-amount heuristic. |
| **Date**    | DD/MM preference for GCC/international context. Multiple format parsing with disambiguation. |
| **Vendor**  | Prior probabilities from training data. Centeredness scoring (vendors typically appear at the top-center). Alpha-ratio filtering (vendor names are mostly letters, not digits). |

**Why "hardened" amount regex?** A naive `\d+\.\d{2}` pattern matches dates (`12.03`), phone suffixes (`55.00`), and invoice IDs. Our regex uses negative lookbehind/lookahead and contextual keyword proximity to isolate actual monetary amounts.

### 5. Anomaly Detection (`src/anomaly.py`)

A **53-feature GradientBoosting classifier** — see the [full feature breakdown](#anomaly-detection--53-features) below.

### 6. Suspicious Localization

When a document is flagged as potentially forged, we don't just say "suspicious" — we show **where and why**:

- 🔴 **ELA hotspots** — regions with anomalous error levels indicating pixel manipulation
- 🟡 **Low-confidence OCR** — words Tesseract struggled with (potentially altered text)
- 🟠 **Low-confidence fields** — extracted values the pipeline is uncertain about
- 🔵 **Conflicting amounts** — multiple contradictory total candidates

### 7. NLG Summaries (`src/summary.py`)

Every anomaly result includes a **human-readable summary** generated without any external LLM:

> *"This document shows signs of potential tampering. The total amount region exhibits elevated ELA intensity (2.3× baseline), and OCR confidence in the date field is unusually low (34%). Two contradictory amount candidates were detected ($156.00 vs $1,560.00)."*

Template-driven with dynamic slot filling — fast, deterministic, and always available.

---

## Anomaly Detection — 53 Features

The classifier doesn't rely on a single signal. It fuses **six feature families** that capture different aspects of document authenticity:

| Family | Count | Features | Intuition |
|--------|-------|----------|-----------|
| **Image** | 6 | Mean, std, edges, noise, dynamic range, block variance | Overall image statistics — forged docs often have inconsistent global properties |
| **ELA** | 5 | Mean, std, max, high ratio, block std | Error Level Analysis — JPEG re-compression artifacts reveal spliced regions |
| **Text** | 5 | Length, line count, digit ratio, alpha ratio, special char ratio | OCR output statistics — forged text often has unusual character distributions |
| **Consistency** | 4 | Amount z-score, IQR gap, known vendor hit, field agreement | Cross-field validation — do the numbers add up? Is this a known vendor? |
| **Field-Local** | 6× fields | ELA intensity + OCR confidence around vendor/date/total boxes | Targeted forensics — most forgeries tamper with specific fields, not the whole document |
| **Conflict** | 2+ | Contradictory amount candidates, date ambiguity | When the document argues with itself |

**Threshold calibration:** The decision boundary isn't a hardcoded 0.5. It's **F1-maximized from out-of-fold predictions** during training, adapting to the actual score distribution.

---

## Design Decisions — *Why Not Just Use a Transformer?*

### 1. OCR-First vs. Vision Transformers

| | OCR-First (DocFusion) | Vision Transformer |
|---|---|---|
| **Latency** | ~0.48s / doc | 2–10s / doc |
| **GPU required** | ❌ No | ✅ Yes (typically) |
| **Deterministic** | ✅ Fully | ⚠️ Depends on implementation |
| **Memory** | ~1.6 MB | 500 MB – 4 GB |
| **Deployable anywhere** | ✅ Raspberry Pi to cloud | ❌ Needs accelerator |

**The real reason:** The challenge has time and resource constraints. A pipeline that can't finish within those constraints doesn't score — no matter how sophisticated the model. OCR-first is a **pragmatic engineering choice**, not a compromise.

### 2. GradientBoosting for Anomaly Detection

- **Fast training** — seconds, not hours
- **Interpretable** — feature importances tell you *why* a document was flagged
- **Mixed features** — handles continuous (ELA mean), categorical (known vendor), and ratio features naturally
- **No overfitting tricks needed** — built-in regularization via depth, learning rate, and subsampling

### 3. Error Level Analysis (ELA)

ELA is a **standard digital forensics technique** used by fraud investigators worldwide. When a JPEG is re-saved, previously compressed and newly spliced regions compress differently. ELA amplifies these differences into visible (and measurable) signals. It adds real forensic value at negligible compute cost.

### 4. Shared Pipeline Architecture

```python
# solution.py (harness)
from src.pipeline import analyze_document
result = analyze_document(image_path)

# app.py (Streamlit UI)  
from src.pipeline import analyze_document
result = analyze_document(uploaded_file)
```

One function. Two consumers. **Zero divergence risk.** If the pipeline improves, both the harness and the UI improve. If there's a bug, it shows up in both — and gets caught in tests.

### 5. Field-Local Features

Most document forgers don't redraw the entire receipt. They change the **total**, alter the **date**, or swap the **vendor name**. By computing ELA intensity and OCR confidence *specifically around extracted field bounding boxes*, the classifier focuses on the regions most likely to be tampered — rather than averaging signals across the entire image where noise can drown out localized manipulation.

---

## Performance Benchmarks

Measured on the `dummy_data` evaluation set:

| Metric | Value |
|--------|-------|
| **Mean latency** | 0.48s / document |
| **P95 latency** | 0.55s / document |
| **Peak memory (mean)** | 1.6 MB |
| **Model size** | Lightweight pickle (~KB) |
| **Deterministic** | ✅ Verified (2 identical runs via `check_submission.py`) |

```
Benchmark run (scripts/benchmark.py):

  Documents processed:  N
  Mean time:            0.48s ± 0.04s
  P95 time:             0.55s
  Peak RSS (mean):      1.6 MB
  Failures:             0
```

---

## Getting Started

### Prerequisites

- **Python 3.13+**
- **Tesseract OCR** installed and on PATH

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# macOS
brew install tesseract

# Verify
tesseract --version
```

### Installation

```bash
# Clone
git clone https://github.com/abdullahalJunaibi/codestacker-2026-ml.git
cd codestacker-2026-ml

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start — Single Document

```python
from src.pipeline import analyze_document

result = analyze_document("path/to/receipt.jpg")

print(f"Vendor: {result.vendor}  (confidence: {result.vendor_confidence:.2f})")
print(f"Date:   {result.date}    (confidence: {result.date_confidence:.2f})")
print(f"Total:  {result.total}   (confidence: {result.total_confidence:.2f})")
print(f"Forged: {result.is_forged}  (score: {result.anomaly_score:.3f})")
print(f"Summary: {result.summary}")
```

### Run via Harness

```bash
# The evaluation harness uses solution.py directly
python solution.py
```

### Verify Submission

```bash
# Determinism check — runs pipeline twice and compares outputs
python check_submission.py --submission .
```

---

## Docker

```bash
# Build
docker build -t docfusion .

# Run harness
docker run --rm -v $(pwd)/data:/app/data docfusion python solution.py

# Run Streamlit UI
docker run --rm -p 8501:8501 docfusion streamlit run app.py --server.address 0.0.0.0
```

The Dockerfile uses `python:3.13-slim` with Tesseract installed via apt — resulting in a lean, production-ready image.

```dockerfile
# Highlights from Docker setup
FROM python:3.13-slim
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
WORKDIR /app
```

---

## Web UI

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

### Features

| Feature | Description |
|---------|-------------|
| 📄 **Upload** | Drag-and-drop images (JPEG, PNG) and PDFs |
| 📑 **Multi-page PDF** | Page selector for navigating multi-page documents |
| 🖼️ **Original view** | See the raw document as uploaded |
| 🏷️ **Fields overlay** | Bounding boxes around extracted vendor, date, total |
| 🔍 **Suspicious overlay** | Highlighted regions flagged by anomaly detection |
| 📊 **Confidence scores** | Per-field extraction confidence |
| ⚠️ **Anomaly verdict** | Score, forged/genuine classification, NLG summary |
| 🔧 **Debug panel** | Expandable OCR text and raw feature inspection |

---

## Testing

```bash
# Run all 18 tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/ -v -k "anomaly"
python -m pytest tests/ -v -k "extraction"
python -m pytest tests/ -v -k "pdf"
python -m pytest tests/ -v -k "benchmark"
```

### Test Coverage

| Category | Tests | What's Verified |
|----------|-------|-----------------|
| Anomaly detection | 4 | Model loading, feature computation, scoring, threshold |
| Field extraction | 5 | Vendor/date/total parsing, edge cases, GCC date formats |
| PDF support | 3 | Single-page, multi-page, corrupt PDF handling |
| Pipeline integration | 4 | End-to-end flow, harness compatibility, determinism |
| Benchmark | 2 | Latency bounds, memory bounds |

### Local Evaluation

```bash
# Run extraction + anomaly metrics against labeled data
python scripts/evaluate_local.py --data-dir data/
```

---

## Project Structure

```
codestacker-2026-ml/
│
├── solution.py              # 🎯 Harness entry point (DocFusionSolution class)
├── app.py                   # 🌐 Streamlit Web UI
├── Dockerfile               # 🐳 Container (Python 3.13-slim + Tesseract)
├── requirements.txt         # 📦 Python dependencies
├── check_submission.py      # ✅ Determinism verification
│
├── src/
│   ├── pipeline.py          # 🔄 Shared analysis engine (analyze_document)
│   ├── document_io.py       # 📥 Image + PDF loading
│   ├── preprocessing.py     # 🖼️ Grayscale, contrast, resize, deskew
│   ├── ocr.py               # 🔤 Tesseract OCR with word boxes + confidence
│   ├── extractor.py         # 🏷️ Field extraction (vendor, date, total)
│   ├── anomaly.py           # 🔍 53-feature anomaly detection + classifier
│   ├── ela.py               # 🔬 Error Level Analysis
│   ├── consistency.py       # ⚖️ Cross-field consistency features
│   ├── summary.py           # 📝 NLG anomaly summaries
│   ├── config.py            # ⚙️ Configuration
│   └── types.py             # 📐 Typed dataclasses
│
├── scripts/
│   ├── benchmark.py         # ⏱️ Latency and memory benchmarks
│   └── evaluate_local.py    # 📊 Local extraction + anomaly evaluation
│
├── notebooks/               # 📓 EDA, extraction, anomaly analysis
├── tests/                   # 🧪 18 regression tests
├── models/                  # 🤖 Trained GradientBoosting artifact
└── data/                    # 📁 Sample data
```

---

## Data Sources & Training

The anomaly classifier was trained on a curated blend of three public datasets, each contributing a different strength:

| Dataset | Contribution | Why It Matters |
|---------|-------------|----------------|
| [**SROIE**](https://rrc.cvc.uab.es/?ch=13) | Baseline English receipt structure | Clean, well-annotated receipts for calibrating OCR accuracy and field extraction |
| [**Find-It-Again**](https://github.com/) | Genuine + forged receipt pairs | Real forgery examples — the ground truth for "what does a tampered document actually look like?" |
| [**CORD**](https://github.com/clovaai/cord) | Diverse layouts, noise, and print quality | Robustness — receipts from dozens of vendors with varying quality, fonts, and structures |

**Training process:**
1. Feature extraction on all training documents (53 features per document)
2. GradientBoosting classifier with cross-validation
3. Threshold calibration via F1-maximization on out-of-fold predictions
4. Final model serialized as lightweight pickle artifact

---

## Notebooks

Exploratory analysis and development notebooks are provided in `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| EDA | Dataset exploration, label distributions, image quality analysis |
| Extraction | Field extraction development, regex tuning, vendor prior analysis |
| Anomaly | Feature engineering, ELA visualization, model selection, threshold tuning |

---

## Acknowledgments

- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** — The backbone of our text extraction
- **[scikit-learn](https://scikit-learn.org)** — For making ML accessible without the ceremony
- **[Rihal](https://rihal.om)** — For organizing CodeStacker and pushing the Omani tech ecosystem forward
- **[PyMuPDF](https://pymupdf.readthedocs.io)** — Fast, reliable PDF processing
- **[Streamlit](https://streamlit.io)** — From pipeline to interactive demo in one file

---

<div align="center">

Built with ☕ and curiosity in Muscat, Oman 🇴🇲

*DocFusion — because every document has a story. Some of them are fiction.*

</div>
]]>