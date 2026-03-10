"""
DocFusion Solution — Intelligent Document Processing Pipeline.

Architecture:
  - Training: OCR all images, extract features, train RF classifier for forgery
  - Extraction: Tesseract OCR + regex heuristics for vendor, date, total
  - Anomaly Detection: Image features + text features + Random Forest
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extractor import extract_fields, extract_text
from src.anomaly import (
    extract_image_features, features_to_vector,
    extract_text_features, text_features_to_vector,
    train_anomaly_model, predict_anomaly,
    FEATURE_KEYS, TEXT_FEATURE_KEYS,
)


class DocFusionSolution:
    def train(self, train_dir: str, work_dir: str) -> str:
        """
        Train on labeled data:
        1. Learn vendor patterns and amount distributions
        2. OCR all training images
        3. Train anomaly detection model
        """
        model_dir = os.path.join(work_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        train_jsonl = os.path.join(train_dir, "train.jsonl")
        with open(train_jsonl) as f:
            records = [json.loads(line) for line in f]

        # Learn vendor vocabulary and amount statistics
        vendors = set()
        amounts = []

        for r in records:
            fields = r.get("fields", {})
            if fields.get("vendor"):
                vendors.add(fields["vendor"])
            total = fields.get("total")
            if total:
                try:
                    amounts.append(float(total))
                except:
                    pass

        stats = {
            "vendors": sorted(vendors),
            "amount_mean": float(np.mean(amounts)) if amounts else 0,
            "amount_std": float(np.std(amounts)) if amounts else 1,
            "amount_q1": float(np.percentile(amounts, 25)) if amounts else 0,
            "amount_q3": float(np.percentile(amounts, 75)) if amounts else 0,
            "total_records": len(records),
        }

        with open(os.path.join(model_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        # Train anomaly detection model
        print(f"[train] Training anomaly model on {len(records)} records...")
        train_anomaly_model(records, train_dir, model_dir)

        print(f"[train] {len(vendors)} unique vendors, amount mean=${stats['amount_mean']:.2f}")
        print(f"[train] Model saved to: {model_dir}")
        return model_dir

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """
        Run inference: extract fields + detect anomalies.
        """
        # Load model artifacts
        with open(os.path.join(model_dir, "stats.json")) as f:
            stats = json.load(f)

        # Load anomaly model
        model_path = os.path.join(model_dir, "anomaly_model.pkl")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        test_jsonl = os.path.join(data_dir, "test.jsonl")
        with open(test_jsonl) as f:
            records = [json.loads(line) for line in f]

        predictions = []
        for r in records:
            img_path = os.path.join(data_dir, r["image_path"])

            # Extract fields via OCR
            extracted = {"vendor": None, "date": None, "total": None}
            ocr_text = ""
            if os.path.exists(img_path):
                try:
                    result = extract_fields(img_path)
                    extracted["vendor"] = result.get("vendor")
                    extracted["date"] = result.get("date")
                    extracted["total"] = result.get("total")
                    ocr_text = result.get("_ocr_text", "")
                except:
                    pass

            # Use provided fields if available, fall back to OCR
            fields = r.get("fields", {})
            vendor = fields.get("vendor") or extracted.get("vendor")
            date = fields.get("date") or extracted.get("date")
            total = fields.get("total") or extracted.get("total")

            # Parse amount for anomaly detection
            amount_val = 0
            if total:
                try:
                    amount_val = float(total)
                except:
                    pass

            # Anomaly detection
            is_forged = predict_anomaly(
                model_data, img_path, ocr_text, amount_val, stats
            )

            predictions.append({
                "id": r["id"],
                "vendor": vendor,
                "date": date,
                "total": total,
                "is_forged": is_forged,
            })

        with open(out_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        print(f"[predict] Wrote {len(predictions)} predictions to {out_path}")
