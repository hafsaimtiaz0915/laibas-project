#!/usr/bin/env python3
"""
Comprehensive TFT V2 Model Audit - Developer Focus

Tests the model across key developers and identifies:
1. Developer mapping issues
2. Prediction anomalies
3. Coverage gaps
4. Potential inaccuracies
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime

print("=" * 80)
print("TFT V2 MODEL AUDIT - DEVELOPER FOCUS")
print("=" * 80)
print(f"Run time: {datetime.now().isoformat()}")

# Paths
MODEL_PATH = REPO_ROOT / "backend" / "models" / "tft_final_v3__patched.ckpt"
DATA_PATH = REPO_ROOT / "Data" / "tft" / "runs" / "20251218T163716Z" / "tft_training_data_v2.csv"

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH, low_memory=False)
print(f"   Rows: {len(data):,}, Groups: {data['group_id'].nunique():,}")

# Load model
print("\n2. Loading model...")
from pytorch_forecasting import TemporalFusionTransformer
import torchmetrics

# Patch for Mac
_orig = torchmetrics.Metric._apply
def _safe(self, fn, *args, **kwargs):
    if hasattr(self, '_device') and self._device and 'cuda' in str(self._device) and not torch.cuda.is_available():
        self._device = torch.device('cpu')
    return _orig(self, fn, *args, **kwargs)
torchmetrics.Metric._apply = _safe

model = TemporalFusionTransformer.load_from_checkpoint(str(MODEL_PATH), map_location='cpu')
model.eval()
print("   ✅ Model loaded")

# =============================================================================
# SECTION A: Developer Mapping Analysis
# =============================================================================
print("\n" + "=" * 80)
print("SECTION A: DEVELOPER MAPPING ANALYSIS")
print("=" * 80)

# Get unique developer labels
dev_cols = ["developer_brand_label", "developer_umbrella", "developer_id"]
dev_cols = [c for c in dev_cols if c in data.columns]

if "developer_brand_label" in data.columns:
    brand_counts = data.groupby("developer_brand_label").agg({
        "group_id": "nunique",
        "transaction_count": "sum" if "transaction_count" in data.columns else "count"
    }).reset_index()
    brand_counts.columns = ["developer_brand_label", "groups", "transactions"]
    brand_counts = brand_counts.sort_values("transactions", ascending=False)
    
    print("\nTop 20 Developer Brands by Transaction Count:")
    print(brand_counts.head(20).to_string(index=False))
    
    # Check for suspicious patterns
    print("\n--- Developer Mapping Issues ---")
    
    # Issue 1: UNKNOWN or empty developers
    unknown_devs = brand_counts[brand_counts["developer_brand_label"].isin(["UNKNOWN", "Unknown", "", "nan", "NaN"])]
    if not unknown_devs.empty:
        print(f"\n⚠️  UNKNOWN developers: {unknown_devs['groups'].sum()} groups, {unknown_devs['transactions'].sum():,.0f} transactions")
    
    # Issue 2: Numeric-looking developer IDs
    numeric_pattern = brand_counts[brand_counts["developer_brand_label"].str.match(r"^DEVELOPER_ID_\d+$", na=False)]
    if not numeric_pattern.empty:
        print(f"\n⚠️  Unmapped DEVELOPER_ID_* labels: {len(numeric_pattern)} brands")
        print(numeric_pattern.head(10).to_string(index=False))

# Developer umbrella analysis
if "developer_umbrella" in data.columns:
    umbrella_counts = data.groupby("developer_umbrella")["group_id"].nunique().reset_index()
    umbrella_counts.columns = ["developer_umbrella", "groups"]
    umbrella_counts = umbrella_counts.sort_values("groups", ascending=False)
    
    print("\n\nTop 15 Developer Umbrellas:")
    print(umbrella_counts.head(15).to_string(index=False))

# =============================================================================
# SECTION B: Test Predictions for Top Developers
# =============================================================================
print("\n" + "=" * 80)
print("SECTION B: PREDICTIONS BY TOP DEVELOPERS")
print("=" * 80)

def make_prediction(group_id, group_data, model):
    """Make a 12-month prediction for a group."""
    try:
        group_data = group_data.sort_values("time_idx").copy()
        if len(group_data) < 24:
            return None, "Insufficient history"
        
        last_time_idx = int(group_data["time_idx"].max())
        
        # Create future rows
        future_rows = []
        for i in range(1, 13):
            future_row = group_data.iloc[-1].copy()
            future_row["time_idx"] = last_time_idx + i
            future_month = (int(future_row.get("month", 1)) + i - 1) % 12 + 1
            future_row["month"] = str(future_month)
            future_row["quarter"] = str((future_month - 1) // 3 + 1)
            future_row["month_sin"] = np.sin(2 * np.pi * future_month / 12)
            future_row["month_cos"] = np.cos(2 * np.pi * future_month / 12)
            uc = pd.to_numeric(group_data["units_completing"], errors="coerce").dropna()
            future_row["units_completing"] = float(uc.tail(6).mean()) if len(uc) else 0.0
            future_rows.append(future_row)
        
        pred_data = pd.concat([group_data, pd.DataFrame(future_rows)], ignore_index=True)
        
        # Prepare data
        for col in ["month", "quarter"]:
            if col in pred_data.columns:
                pred_data[col] = pred_data[col].astype(str)
        
        if "has_actual_rent" not in pred_data.columns:
            has_rent = (pd.to_numeric(pred_data["median_rent"], errors="coerce").fillna(0) > 0).any()
            pred_data["has_actual_rent"] = int(has_rent)
        
        for col in ["area_name", "property_type", "bedroom", "reg_type", 
                    "developer_brand_label", "developer_umbrella"]:
            if col in pred_data.columns:
                pred_data[col] = pred_data[col].fillna("UNKNOWN").astype(str)
        
        for col in pred_data.columns:
            if col in ["group_id", "year_month"]:
                continue
            if pred_data[col].dtype == object:
                continue
            pred_data[col] = pd.to_numeric(pred_data[col], errors="coerce").fillna(0)
        
        with torch.no_grad():
            raw_output = model.predict(pred_data, mode="raw", return_x=False)
        
        output = getattr(raw_output, "output", None) or raw_output
        
        if isinstance(output, list) and len(output) >= 2:
            price_q50 = output[0][0, -1, 3].item()
            rent_q50 = output[1][0, -1, 3].item()
            return {
                "price_q50": price_q50,
                "rent_q50": rent_q50,
            }, None
        return None, "Unexpected output format"
    except Exception as e:
        return None, str(e)

# Get top developers to test
if "developer_brand_label" in data.columns:
    top_devs = brand_counts.head(25)["developer_brand_label"].tolist()
else:
    top_devs = []

results = []
issues = []

print("\nTesting predictions for top developers...")

for dev in top_devs:
    dev_data = data[data["developer_brand_label"] == dev]
    dev_groups = dev_data["group_id"].unique()
    
    # Test up to 3 groups per developer
    for gid in dev_groups[:3]:
        group_data = data[data["group_id"] == gid]
        latest = group_data.sort_values("time_idx").iloc[-1]
        current_price = latest.get("median_price", 0)
        current_rent = latest.get("median_rent", 0)
        
        pred, error = make_prediction(gid, group_data, model)
        
        if pred:
            appreciation = ((pred["price_q50"] / current_price) - 1) * 100 if current_price > 0 else None
            rent_change = ((pred["rent_q50"] / current_rent) - 1) * 100 if current_rent > 0 else None
            
            result = {
                "developer": dev,
                "group_id": gid,
                "area": latest.get("area_name", ""),
                "bedroom": latest.get("bedroom", ""),
                "reg_type": latest.get("reg_type", ""),
                "current_price": current_price,
                "forecast_price": pred["price_q50"],
                "appreciation_%": appreciation,
                "current_rent": current_rent,
                "forecast_rent": pred["rent_q50"],
                "rent_change_%": rent_change,
                "history_months": len(group_data),
            }
            results.append(result)
            
            # Flag anomalies
            if appreciation is not None:
                if appreciation < -20:
                    issues.append({"type": "CRASH_FORECAST", "group": gid, "developer": dev, "value": f"{appreciation:.1f}%"})
                elif appreciation > 50:
                    issues.append({"type": "EXTREME_GROWTH", "group": gid, "developer": dev, "value": f"{appreciation:.1f}%"})
            
            if pred["price_q50"] < 100:
                issues.append({"type": "IMPLAUSIBLE_PRICE", "group": gid, "developer": dev, "value": f"{pred['price_q50']:.0f} AED/sqm"})
            
            if pred["rent_q50"] < 1000 and current_rent > 10000:
                issues.append({"type": "RENT_COLLAPSE", "group": gid, "developer": dev, "value": f"{pred['rent_q50']:.0f} AED/yr"})
        else:
            issues.append({"type": "PREDICTION_FAILED", "group": gid, "developer": dev, "value": error})

# Print results
results_df = pd.DataFrame(results)
if not results_df.empty:
    print(f"\n✅ Successfully tested {len(results_df)} groups across {len(top_devs)} developers")
    
    # Summary stats
    print("\n--- Prediction Summary by Developer ---")
    dev_summary = results_df.groupby("developer").agg({
        "appreciation_%": ["mean", "min", "max"],
        "rent_change_%": "mean",
        "group_id": "count"
    }).round(1)
    dev_summary.columns = ["avg_apprec_%", "min_apprec_%", "max_apprec_%", "avg_rent_chg_%", "groups_tested"]
    print(dev_summary.head(20).to_string())

# =============================================================================
# SECTION C: Coverage Gaps
# =============================================================================
print("\n" + "=" * 80)
print("SECTION C: COVERAGE GAPS")
print("=" * 80)

# Groups with very short history
group_lengths = data.groupby("group_id")["time_idx"].count()
short_groups = group_lengths[group_lengths < 24]
print(f"\n⚠️  Groups with <24 months history: {len(short_groups)} ({len(short_groups)/len(group_lengths)*100:.1f}%)")

# Areas with no Ready transactions
if "reg_type" in data.columns:
    area_reg = data.groupby(["area_name", "reg_type"])["group_id"].nunique().unstack(fill_value=0)
    if "Ready" in area_reg.columns and "OffPlan" in area_reg.columns:
        offplan_only = area_reg[area_reg["Ready"] == 0]
        if not offplan_only.empty:
            print(f"\n⚠️  Areas with ONLY OffPlan data (no Ready): {len(offplan_only)}")
            print("   ", list(offplan_only.index[:10]))

# Developers with only 1 group
if "developer_brand_label" in data.columns:
    dev_group_counts = data.groupby("developer_brand_label")["group_id"].nunique()
    single_group_devs = dev_group_counts[dev_group_counts == 1]
    print(f"\n⚠️  Developers with only 1 group: {len(single_group_devs)}")

# =============================================================================
# SECTION D: Data Quality Issues
# =============================================================================
print("\n" + "=" * 80)
print("SECTION D: DATA QUALITY FLAGS")
print("=" * 80)

# Check for zero or negative prices
zero_prices = data[data["median_price"] <= 0]
if len(zero_prices) > 0:
    print(f"\n⚠️  Rows with zero/negative median_price: {len(zero_prices)}")

# Check units_completing distribution
if "units_completing" in data.columns:
    uc = data["units_completing"]
    uc_nonzero = (uc > 0).sum()
    print(f"\n✅ units_completing: {uc_nonzero:,} non-zero rows ({uc_nonzero/len(data)*100:.1f}%)")
    print(f"   Max: {uc.max():,}, Mean (when >0): {uc[uc>0].mean():.0f}")

# =============================================================================
# SECTION E: Issue Summary
# =============================================================================
print("\n" + "=" * 80)
print("SECTION E: ISSUE SUMMARY")
print("=" * 80)

if issues:
    issues_df = pd.DataFrame(issues)
    issue_counts = issues_df["type"].value_counts()
    
    print("\nIssues found:")
    for issue_type, count in issue_counts.items():
        print(f"   {issue_type}: {count}")
    
    print("\n--- Sample Issues (first 15) ---")
    print(issues_df.head(15).to_string(index=False))
else:
    print("\n✅ No critical issues found!")

# =============================================================================
# SAVE REPORT
# =============================================================================
report_path = REPO_ROOT / "data_profiles" / "tft_v3_audit_report.json"
report = {
    "run_time": datetime.now().isoformat(),
    "model": str(MODEL_PATH),
    "data": str(DATA_PATH),
    "summary": {
        "total_groups": int(data["group_id"].nunique()),
        "total_rows": len(data),
        "developers_tested": len(top_devs),
        "predictions_made": len(results),
        "issues_found": len(issues),
    },
    "issues": issues,
    "results_sample": results[:20] if results else [],
}

report_path.parent.mkdir(exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n✅ Report saved to: {report_path}")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

