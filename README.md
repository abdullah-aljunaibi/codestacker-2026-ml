# DocFusion

DocFusion is a deterministic OCR-first submission for the Rihal CodeStacker 2026 ML challenge. It extracts `vendor`, `date`, and `total`, scores likely forgery, supports image and PDF inputs, and exposes the same shared analysis engine to both the harness and the Streamlit reviewer UI.

## Architecture

`src/pipeline.py` is the single analysis path used everywhere:

1. `src/document_io.py` loads images directly and rasterizes PDFs through PyMuPDF.
2. `src/preprocessing.py` applies grayscale conversion, contrast cleanup, resizing, and lightweight deskew.
3. `src/ocr.py` runs Tesseract with `image_to_data()` to produce word boxes, confidences, and reconstructed lines.
4. `src/extractor.py` ranks field candidates using OCR line position, text cues, box context, normalization, and confidence scoring.
5. `src/anomaly.py` combines image statistics, ELA signals, OCR quality, extraction confidence, and consistency features, then uses either the trained classifier or a deterministic heuristic fallback.

The harness entrypoint stays in `solution.py` as `DocFusionSolution`. The UI in `app.py` calls the same `analyze_document()` function and renders extracted-field overlays plus suspicious-region overlays.

## Install

Prerequisites:

- Python 3.13+
- Tesseract OCR installed on the host

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Project setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Train And Predict

Harness smoke flow:

```bash
python check_submission.py --submission .
```

Direct usage:

```python
from solution import DocFusionSolution

solution = DocFusionSolution()
model_dir = solution.train("dummy_data/train", "tmp_work")
solution.predict(model_dir, "dummy_data/test", "tmp_work/predictions.jsonl")
```

## UI

Run:

```bash
streamlit run app.py
```

The UI supports images and PDFs, shows:

- extracted fields with confidences and boxes
- suspicious regions from ELA hotspots and low-confidence OCR
- shared anomaly score and label
- raw OCR text and feature debug output

## Validation

Commands used during development:

```bash
python -m compileall src solution.py app.py
python check_submission.py --submission .
pytest
python scripts/benchmark.py --inputs dummy_data/test/images/t001.png
```

## Data Preparation Scripts

Helper manifests are included for external dataset curation:

- `scripts/prepare_sroie.py`
- `scripts/prepare_find_it_again.py`
- `scripts/prepare_cord.py`

They create JSONL manifests only. They do not download datasets and should not be used to commit external data into this repository.

## Limitations

- The default harness path intentionally avoids large transformer inference.
- PDF rasterization depends on `pymupdf`.
- The anomaly model is lightweight and should be recalibrated on real challenge data for best ranking performance.
