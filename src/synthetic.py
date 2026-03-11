"""
Synthetic receipt data generator for DocFusion training.
Generates realistic labeled training data with forged/legitimate patterns.
"""
import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

VENDORS_COMMON = [
    "7-Eleven", "McDonald's", "Walmart", "Starbucks", "CVS Pharmacy",
    "Target", "Subway", "Shell Gas", "BP Station", "Carrefour",
    "Lulu Hypermarket", "Al Meera", "KFC", "Pizza Hut", "Dominos",
    "Noon.com", "Talabat", "Nandos", "Costa Coffee", "Tim Hortons",
    "IKEA", "H&M", "Zara", "Noon Express", "Careem",
]

VENDORS_SUSPICIOUS = [
    "ACME Corp", "XYZ Store", "Generic Shop", "Unknown Vendor",
    "Test Store", "Temp Business", "Cash Only Store",
]


def random_date(start_year=2023, end_year=2024, suspicious=False):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days))
    if suspicious:
        # Push to weekend
        while d.weekday() < 5:
            d += timedelta(days=random.randint(1, 3))
    return d.strftime("%Y-%m-%d")


def random_amount(forged=False):
    if forged:
        r = random.random()
        if r < 0.35:
            # Suspiciously round
            return float(random.choice([100, 200, 500, 1000, 50, 250, 750]))
        elif r < 0.65:
            # Extreme outlier (high)
            return round(random.uniform(800, 2000), 2)
        else:
            # Very low (under-reporting)
            return round(random.uniform(0.01, 2.0), 2)
    else:
        # Normal receipt amount: $3 - $150, realistic distribution
        base = random.expovariate(1 / 35)
        return round(max(1.5, min(base, 150.0)), 2)


def generate_records(n=300, forged_rate=0.25, seed=42):
    random.seed(seed)
    records = []
    n_forged = int(n * forged_rate)
    labels = [1] * n_forged + [0] * (n - n_forged)
    random.shuffle(labels)

    for i, is_forged in enumerate(labels):
        rec_id = f"syn{i:04d}"

        if is_forged:
            r = random.random()
            if r < 0.3:
                vendor = random.choice(VENDORS_SUSPICIOUS)
            elif r < 0.6:
                vendor = random.choice(VENDORS_COMMON)
            else:
                vendor = None  # missing field
            amount = random_amount(forged=True)
            date = random_date(suspicious=random.random() < 0.5)
            if random.random() < 0.2:
                date = None  # missing date
        else:
            vendor = random.choice(VENDORS_COMMON)
            amount = random_amount(forged=False)
            date = random_date(suspicious=False)

        record = {
            "id": rec_id,
            "image_path": f"dummy_data/train/{rec_id}.png",
            "vendor": vendor,
            "date": date,
            "total": str(amount) if amount else None,
            "label": {"is_forged": is_forged},
        }
        records.append(record)

    return records


def save_synthetic(output_path="data/synthetic_train.jsonl", n=300):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    records = generate_records(n=n)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    forged = sum(1 for r in records if r["label"]["is_forged"])
    print(f"[synthetic] Generated {len(records)} records ({forged} forged, {len(records)-forged} legitimate)")
    print(f"[synthetic] Saved to {output_path}")
    return records


if __name__ == "__main__":
    save_synthetic()
