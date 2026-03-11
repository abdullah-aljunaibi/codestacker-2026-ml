# AGENTS.md — DocFusion ML (CodeStacker 2026 ML Track)

## Project Summary
Receipt processing submission for CodeStacker 2026 ML challenge.
Extracts structured fields from scanned receipts + detects tampering.

## Tech Stack
- **Language**: Python 3
- **ML**: PyTorch, Transformers, Tesseract OCR
- **UI**: Streamlit (inspection UI)
- **Packaging**: Docker

## Key File Locations
```
solution.py            → Main harness entrypoint (DocFusionSolution class)
app.py                 → Streamlit inspection UI
check_submission.py    → Local challenge validator
src/
  anomaly.py           → Tampering detection
  config.py            → Config constants
  consistency.py       → Field consistency checks
  ela.py               → Error Level Analysis (image tamper detection)
  extractor.py         → OCR field extraction (vendor, date, total)
tests/                 → Test suite
dummy_data/            → Sample receipts for testing
sample_submission/     → Reference output format
```

## Pipeline
1. Tesseract OCR → raw text from receipt image
2. Layout-aware + regex heuristics → extract vendor, date, total
3. Visual + ELA + text + consistency features → tampering score
4. Output: structured JSON per receipt

## Verify Commands
```bash
python check_submission.py    # validate submission locally
python -m pytest tests/       # run tests
docker build -t docfusion .   # build container
```

## Critical Rules
1. `solution.py` DocFusionSolution class is the harness entrypoint — never rename it
2. All dependencies must be in requirements.txt
3. Dockerfile must produce a working container
4. Output format must match sample_submission/ exactly
5. No internet access assumed at inference time — all models must be bundled or downloaded at build time
