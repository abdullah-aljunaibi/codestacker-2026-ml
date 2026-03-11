# DocFusion

DocFusion is a receipt-processing submission for the CodeStacker 2026 ML challenge. It extracts structured fields from scanned receipts, scores likely tampering, and exposes both the harness entrypoint and a Streamlit inspection UI from the same codebase.

## What It Does

- OCRs receipt images with Tesseract.
- Extracts `vendor`, `date`, and `total` using layout-aware and regex heuristics.
- Detects suspicious receipts with visual, ELA, text, and consistency features.
- Packages the workflow behind `DocFusionSolution` for local challenge validation.

## Repository Layout

```text
.
├── app.py
├── check_submission.py
├── Dockerfile
├── requirements.txt
├── solution.py
├── dummy_data/
├── notebooks/
├── scripts/
├── tests/
└── src/
    ├── anomaly.py
    ├── config.py
    ├── consistency.py
    ├── ela.py
    ├── extractor.py
    ├── ocr.py
    ├── preprocessing.py
    ├── reproducibility.py
    ├── summary.py
    └── data/
```

## Architecture

```text
Receipt image
    |
    +--> src/preprocessing.py
    |       image cleanup + minimum width normalization
    |
    +--> src/ocr.py
    |       Tesseract OCR -> text + word boxes + confidences
    |
    +--> src/extractor.py
    |       vendor/date/total extraction
    |
    +--> src/anomaly.py
            base image statistics
            + ELA features from src/ela.py
            + text features
            + amount/vendor consistency signals from src/consistency.py
            -> GradientBoostingClassifier or heuristic fallback

solution.py
    train()  -> builds stats.json + anomaly_model.pkl
    predict() -> emits predictions.jsonl

app.py
    Streamlit UI for OCR review, feature inspection, and anomaly overlays
```

## Core Components

### Extraction pipeline

- `src/ocr.py` returns structured OCR output, including word-level bounding boxes.
- `src/extractor.py` normalizes dates, amounts, and vendor names from OCR text.
- `src/extractors/vendor.py` uses OCR layout context to improve vendor selection.

### Anomaly pipeline

- `src/anomaly.py` computes grayscale image statistics, text ratios, and model-ready vectors.
- `src/ela.py` adds JPEG recompression signals for edit detection.
- `src/consistency.py` measures amount plausibility and vendor/text consistency against training-set stats.
- `solution.py` uses the trained classifier when available and falls back to deterministic heuristics otherwise.

### Summary helpers

- `src/summary.py` formats extraction and anomaly outputs into compact summaries for templates, logs, or UI surfaces.

## Local Setup

### Prerequisites

- Python 3.13+
- Tesseract OCR installed on the host

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### Run the challenge smoke check

```bash
python check_submission.py --submission .
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

### Use the harness entrypoint directly

```python
from solution import DocFusionSolution

solution = DocFusionSolution()
model_dir = solution.train("dummy_data/train", "tmp_work")
solution.predict(model_dir, "dummy_data/test", "tmp_work/predictions.jsonl")
```

## Docker

Build:

```bash
docker build -t docfusion .
```

Run:

```bash
docker run --rm -p 8501:8501 docfusion
```

The container starts Streamlit on `0.0.0.0:8501`.

## Outputs

`predict()` writes JSONL rows like:

```json
{"id":"t001","vendor":"ACME Corp","date":"2024-01-01","total":"10.00","is_forged":0}
```

Training artifacts are written under the chosen work directory:

- `stats.json`
- `anomaly_model.pkl`

## Validation

Common checks during development:

```bash
python -m unittest discover -s tests
python check_submission.py --submission .
python -m compileall src solution.py app.py
```
