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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DEFAULT_CONFIG
from src.data.adapters import load_dataset_records, save_predictions, write_json_file
from src.data.schema import PredictionRecord
from src.extractor import extract_fields
from src.anomaly import (
    train_anomaly_model, predict_anomaly,
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

        records = load_dataset_records(train_dir, "train")

        # Learn vendor vocabulary and amount statistics
        vendors = set()
        amounts = []

        for r in records:
            fields = r.fields
            if fields.vendor:
                vendors.add(fields.vendor)
            total = fields.total
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

        write_json_file(
            stats,
            os.path.join(model_dir, DEFAULT_CONFIG.data.stats_file_name),
        )

        # Train anomaly detection model
        print(f"[train] Training anomaly model on {len(records)} records...")
        train_anomaly_model(
            [record.model_dump(mode="json") for record in records],
            train_dir,
            model_dir,
        )

        print(f"[train] {len(vendors)} unique vendors, amount mean=${stats['amount_mean']:.2f}")
        print(f"[train] Model saved to: {model_dir}")
        return model_dir

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """
        Run inference: extract fields + detect anomalies.
        """
        # Load model artifacts
        with open(os.path.join(model_dir, DEFAULT_CONFIG.data.stats_file_name)) as f:
            stats = json.load(f)

        # Load anomaly model
        model_path = os.path.join(model_dir, DEFAULT_CONFIG.data.anomaly_model_file_name)
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        records = load_dataset_records(data_dir, "test")

        predictions = []
        for r in records:
            img_path = os.path.join(data_dir, r.image_path)

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
            fields = r.fields
            vendor = fields.vendor or extracted.get("vendor")
            date = fields.date or extracted.get("date")
            total = fields.total or extracted.get("total")

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

            predictions.append(
                PredictionRecord(
                    id=r.id,
                    vendor=vendor,
                    date=date,
                    total=total,
                    is_forged=is_forged,
                )
            )

        save_predictions(predictions, out_path)

        print(f"[predict] Wrote {len(predictions)} predictions to {out_path}")
