# %% [markdown]
# # DocFusion — Exploratory Data Analysis
# 
# This notebook explores the three datasets used in the DocFusion challenge:
# 1. **Dummy Data** (provided by the challenge — synthetic receipts)
# 2. **CORD** (Consolidated Receipt Dataset — real receipts, diverse layouts)
# 3. **SROIE** (Scanned Receipts OCR — English receipts)
# 
# Goal: Understand data structure, field distributions, and anomaly patterns.

# %%
import json
import os
import glob
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## 1. Dummy Data Analysis
# 
# The challenge provides 20 training + 10 test synthetic receipts.

# %%
# Load dummy train data
DUMMY_DIR = "../dummy_data"

with open(f"{DUMMY_DIR}/train/train.jsonl") as f:
    train_records = [json.loads(line) for line in f]

with open(f"{DUMMY_DIR}/test/test.jsonl") as f:
    test_records = [json.loads(line) for line in f]

with open(f"{DUMMY_DIR}/test/labels.jsonl") as f:
    test_labels = {json.loads(line)["id"]: json.loads(line)["label"] for line in f}

print(f"Train records: {len(train_records)}")
print(f"Test records: {len(test_records)}")
print(f"\nSample train record:")
print(json.dumps(train_records[0], indent=2))

# %%
# Field analysis
vendors = [r["fields"]["vendor"] for r in train_records]
totals = [float(r["fields"]["total"]) for r in train_records]
dates = [r["fields"]["date"] for r in train_records]
forged = [r["label"]["is_forged"] for r in train_records]
fraud_types = [r["label"]["fraud_type"] for r in train_records]

print("=== Vendor Distribution ===")
for vendor, count in Counter(vendors).most_common():
    print(f"  {vendor}: {count}")

print(f"\n=== Total Amount Stats ===")
print(f"  Min: ${min(totals):.2f}")
print(f"  Max: ${max(totals):.2f}")
print(f"  Mean: ${np.mean(totals):.2f}")
print(f"  Median: ${np.median(totals):.2f}")
print(f"  Std: ${np.std(totals):.2f}")

print(f"\n=== Forgery Distribution ===")
print(f"  Genuine: {forged.count(0)} ({forged.count(0)/len(forged)*100:.0f}%)")
print(f"  Forged: {forged.count(1)} ({forged.count(1)/len(forged)*100:.0f}%)")

print(f"\n=== Fraud Types ===")
for ft, count in Counter(fraud_types).most_common():
    print(f"  {ft}: {count}")

# %%
# Visualizations

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DocFusion Dummy Data — EDA", fontsize=16)

# 1. Vendor distribution
vendor_counts = Counter(vendors)
axes[0,0].bar(vendor_counts.keys(), vendor_counts.values(), color='steelblue')
axes[0,0].set_title("Vendor Distribution")
axes[0,0].set_ylabel("Count")
axes[0,0].tick_params(axis='x', rotation=30)

# 2. Total amount distribution
axes[0,1].hist(totals, bins=10, color='teal', edgecolor='black', alpha=0.7)
axes[0,1].set_title("Total Amount Distribution")
axes[0,1].set_xlabel("Amount ($)")
axes[0,1].set_ylabel("Frequency")
axes[0,1].axvline(np.mean(totals), color='red', linestyle='--', label=f'Mean: ${np.mean(totals):.0f}')
axes[0,1].legend()

# 3. Forgery by vendor
vendors_forged = Counter()
vendors_genuine = Counter()
for r in train_records:
    v = r["fields"]["vendor"]
    if r["label"]["is_forged"] == 1:
        vendors_forged[v] += 1
    else:
        vendors_genuine[v] += 1

all_vendors = sorted(set(vendors))
x = np.arange(len(all_vendors))
width = 0.35
axes[1,0].bar(x - width/2, [vendors_genuine.get(v, 0) for v in all_vendors], width, label='Genuine', color='green', alpha=0.7)
axes[1,0].bar(x + width/2, [vendors_forged.get(v, 0) for v in all_vendors], width, label='Forged', color='red', alpha=0.7)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(all_vendors, rotation=30)
axes[1,0].set_title("Forgery by Vendor")
axes[1,0].legend()

# 4. Fraud type distribution
ft_counts = Counter(fraud_types)
labels = [k for k in ft_counts.keys() if k != 'none']
sizes = [ft_counts[k] for k in labels]
colors = ['#ff6b6b', '#ffa94d', '#ffd43b']
axes[1,1].pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.0f%%', startangle=90)
axes[1,1].set_title("Fraud Type Breakdown (Forged Only)")

plt.tight_layout()
plt.savefig("../notebooks/eda_dummy_data.png", dpi=150, bbox_inches='tight')
print("Saved: notebooks/eda_dummy_data.png")

# %%
# Amount comparison: forged vs genuine
forged_amounts = [float(r["fields"]["total"]) for r in train_records if r["label"]["is_forged"] == 1]
genuine_amounts = [float(r["fields"]["total"]) for r in train_records if r["label"]["is_forged"] == 0]

print(f"\n=== Amount by Forgery Status ===")
print(f"  Genuine — Mean: ${np.mean(genuine_amounts):.2f}, Std: ${np.std(genuine_amounts):.2f}")
print(f"  Forged  — Mean: ${np.mean(forged_amounts):.2f}, Std: ${np.std(forged_amounts):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([genuine_amounts, forged_amounts], labels=['Genuine', 'Forged'])
ax.set_title("Total Amount: Genuine vs Forged")
ax.set_ylabel("Amount ($)")
plt.savefig("../notebooks/eda_amount_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: notebooks/eda_amount_comparison.png")

# %% [markdown]
# ## 2. CORD Dataset Analysis
# 
# CORD provides 800 train + 100 test receipts with structured ground truth.

# %%
try:
    from datasets import load_dataset
    cord = load_dataset('naver-clova-ix/cord-v2')
    
    print(f"CORD dataset loaded:")
    print(f"  Train: {len(cord['train'])} receipts")
    print(f"  Validation: {len(cord['validation'])} receipts") 
    print(f"  Test: {len(cord['test'])} receipts")
    
    # Parse ground truth structure
    cord_totals = []
    cord_has_total = 0
    cord_menu_counts = []
    
    for item in cord['train']:
        gt = json.loads(item['ground_truth'])
        parse = gt.get('gt_parse', {})
        
        # Count menu items
        menu = parse.get('menu', [])
        cord_menu_counts.append(len(menu))
        
        # Extract total
        total_info = parse.get('total', {})
        if isinstance(total_info, dict):
            total_price = total_info.get('total_price') or total_info.get('total_etc')
            if total_price:
                cord_has_total += 1
                # Try to parse numeric value
                try:
                    cleaned = total_price.replace(',', '').replace('.', '').strip()
                    if cleaned.isdigit():
                        cord_totals.append(float(cleaned))
                except:
                    pass
        elif isinstance(total_info, list) and len(total_info) > 0:
            cord_has_total += 1
    
    print(f"\n=== CORD Field Coverage ===")
    print(f"  Receipts with total: {cord_has_total}/{len(cord['train'])} ({cord_has_total/len(cord['train'])*100:.0f}%)")
    print(f"  Avg menu items: {np.mean(cord_menu_counts):.1f}")
    print(f"  Max menu items: {max(cord_menu_counts)}")
    
    # Image sizes
    widths = []
    heights = []
    for item in cord['train'][:50]:  # Sample 50
        w, h = item['image'].size
        widths.append(w)
        heights.append(h)
    
    print(f"\n=== CORD Image Sizes (sample 50) ===")
    print(f"  Width range: {min(widths)}–{max(widths)} px")
    print(f"  Height range: {min(heights)}–{max(heights)} px")
    print(f"  Common aspect: {np.mean([h/w for w,h in zip(widths, heights)]):.2f}:1")

except Exception as e:
    print(f"CORD analysis skipped: {e}")

# %% [markdown]
# ## 3. Key Observations
# 
# ### Dummy Data Patterns
# - 50/50 split between genuine and forged documents
# - Three fraud types: `price_change`, `text_edit`, `layout_edit`
# - Forged receipts tend to have slightly higher totals (but small sample)
# - 5 vendors with roughly equal distribution
# 
# ### CORD Dataset
# - 800 diverse receipts with varying layouts (Indonesian, English, etc.)
# - Structured ground truth with menu items, totals, subtotals
# - Variable image sizes and quality
# - Rich vocabulary of fields for training extraction models
#
# ### Implications for Pipeline Design
# 1. **Extraction**: Must handle diverse layouts — rule-based OCR won't scale
# 2. **Anomaly Detection**: Fraud types suggest we need both:
#    - Statistical anomaly detection (price_change → unusual amounts)
#    - Visual analysis (layout_edit, text_edit → image-level features)
# 3. **The autograder uses private data** — solution must generalize beyond dummy data

# %%
print("\n=== EDA COMPLETE ===")
print("Outputs saved to notebooks/")
