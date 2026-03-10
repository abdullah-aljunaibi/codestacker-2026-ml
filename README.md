# DocFusion — Intelligent Document Processing Pipeline

**CodeStacker 2026 ML Challenge**  
**Author:** Abdullah Al Junaibi  
**Date:** March 10, 2026

---

## Overview

An end-to-end document processing pipeline that:
1. **Extracts** structured fields (vendor, date, total) from scanned receipts via OCR
2. **Detects** forged/suspicious documents using image analysis + machine learning
3. **Displays** results in an interactive web dashboard

## Quick Start

### Prerequisites
- Python 3.13+
- Tesseract OCR (`apt install tesseract-ocr`)

### Setup

```bash
# Clone
git clone https://github.com/abdullah-aljunaibi/codestacker-2026-ml.git
cd codestacker-2026-ml

# Install dependencies
pip install -r requirements.txt

# Run smoke test
python check_submission.py --submission .
```

### Run Web UI

```bash
streamlit run app.py
# Open http://localhost:8501
```

### Docker

```bash
docker build -t docfusion .
docker run -p 8501:8501 docfusion
```

---

## Architecture

```
Receipt Image
     │
     ▼
┌─────────────┐    ┌──────────────┐
│ Tesseract   │───▶│ Regex        │──▶ vendor, date, total
│ OCR         │    │ Extraction   │
└─────────────┘    └──────────────┘
     │
     ▼
┌─────────────┐    ┌──────────────┐
│ Image       │───▶│ Random       │──▶ is_forged (0/1)
│ Features    │    │ Forest       │
│ (15 visual  │    │ Classifier   │
│  + 7 text)  │    └──────────────┘
└─────────────┘
```

### Extraction (Level 2)
- **Tesseract OCR** with preprocessing (grayscale, upscale)
- **Regex heuristics** for date patterns (ISO, US, European, written)
- **Total extraction** via keyword matching ("TOTAL", "AMOUNT DUE") + fallback to largest number
- **Vendor detection** from first meaningful line of OCR text

### Anomaly Detection (Level 3)
- **15 image features**: pixel stats, edge density, noise variance, block variance (copy-paste detection), histogram entropy, dynamic range
- **7 text features**: text length, line count, digit/alpha/special ratios
- **Random Forest classifier** trained on labeled data (when real images available)
- **Heuristic fallback** for blank/synthetic images using amount statistics + base rate

### Web UI (Level 3B)
- **Streamlit dashboard** with image upload, field extraction, anomaly scoring
- Adjustable sensitivity slider, raw OCR view, feature debug panel
- Sample data browser with ground truth comparison

---

## Project Structure

```
.
├── solution.py              # Harness interface (DocFusionSolution class)
├── app.py                   # Streamlit Web UI
├── Dockerfile               # Container for model + UI
├── requirements.txt         # Python dependencies
├── check_submission.py      # Local smoke test (from challenge)
├── src/
│   ├── extractor.py         # OCR + regex field extraction
│   └── anomaly.py           # Image features + RF anomaly detection
├── notebooks/
│   ├── eda.py               # EDA script
│   ├── eda.ipynb             # Jupyter notebook
│   ├── eda_dummy_data.png   # Visualization
│   └── eda_amount_comparison.png
├── dummy_data/              # Challenge-provided synthetic data
│   ├── train/
│   └── test/
└── data/                    # Real datasets (gitignored)
```

---

## Harness Interface

```python
from solution import DocFusionSolution

sol = DocFusionSolution()
model_dir = sol.train("path/to/train_dir", "path/to/work_dir")
sol.predict(model_dir, "path/to/test_dir", "predictions.jsonl")
```

### Output Format (`predictions.jsonl`)
```json
{"id": "t001", "vendor": "ACME Corp", "date": "2024-01-01", "total": "10.00", "is_forged": 0}
{"id": "t002", "vendor": null, "date": null, "total": null, "is_forged": 1}
```

---

## EDA Key Findings

| Metric | Value |
|--------|-------|
| Dummy train | 20 receipts (50% forged) |
| Fraud types | price_change, text_edit, layout_edit |
| CORD dataset | 800 train, 100 test (diverse layouts) |
| Forged vs genuine amounts | Statistically similar (mean ~$252) |

**Implication:** Amount-only detection is insufficient — visual features (noise, edges, block variance) are essential for forgery detection.

---

## Design Decisions

1. **OCR + regex over vision models**: Faster inference, lower memory — important for harness benchmarks
2. **Random Forest over deep learning**: Trains quickly on small datasets, interpretable features
3. **Two-tier anomaly detection**: ML model when image features are available, heuristic fallback for edge cases
4. **Atomic feature extraction**: Image and text features computed independently, combined for classification

---

## Technical Stack

- Python 3.13+
- Tesseract OCR 5.x
- scikit-learn (Random Forest)
- Pillow (image processing)
- Streamlit (web UI)
- NumPy, Matplotlib (analysis)
