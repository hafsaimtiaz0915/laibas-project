#!/usr/bin/env python3
"""
Test script for the new TFT V2 model.

Prerequisites:
1. Download tft_final_v2.ckpt from Google Drive to backend/models/tft_final_v2.ckpt
2. Copy the V2 training data:
   cp Data/tft/runs/20251218T163716Z/tft_training_data_v2.csv backend/models/tft_training_data_v2.csv
"""

import sys
from pathlib import Path

# Add backend to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd
import numpy as np
import torch

print("=" * 70)
print("TFT V2 MODEL TEST")
print("=" * 70)

# Paths
MODEL_PATH = REPO_ROOT / "backend" / "models" / "tft_final_v3.ckpt"
DATA_PATH = REPO_ROOT / "Data" / "tft" / "runs" / "20251218T163716Z" / "tft_training_data_v2.csv"

# Check files exist
print("\n1. Checking files...")
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    print("   Download from Google Drive: /MyDrive/Properly/tft_final_v2.ckpt")
    sys.exit(1)
print(f"✅ Model: {MODEL_PATH}")

if not DATA_PATH.exists():
    print(f"❌ Data not found: {DATA_PATH}")
    sys.exit(1)
print(f"✅ Data: {DATA_PATH}")

# Load data
print("\n2. Loading training data...")
data = pd.read_csv(DATA_PATH, low_memory=False)
print(f"   Rows: {len(data):,}")
print(f"   Groups: {data['group_id'].nunique():,}")
print(f"   Columns: {len(data.columns)}")

# Check key columns
key_cols = ["group_id", "time_idx", "median_price", "median_rent", "units_completing"]
missing = [c for c in key_cols if c not in data.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    sys.exit(1)
print(f"✅ All key columns present")

# Load model
print("\n3. Loading model...")
try:
    from pytorch_forecasting import TemporalFusionTransformer
    import torchmetrics
    
    # Patch torchmetrics to handle CUDA device mismatch on Mac
    _orig_metric_apply = torchmetrics.Metric._apply
    def _safe_metric_apply(self, fn, *args, **kwargs):
        if hasattr(self, '_device') and self._device is not None:
            if 'cuda' in str(self._device) and not torch.cuda.is_available():
                self._device = torch.device('cpu')
        return _orig_metric_apply(self, fn, *args, **kwargs)
    torchmetrics.Metric._apply = _safe_metric_apply
    
    def remove_bad_keys(d):
        """Recursively remove 'monotone_constaints' from nested dicts."""
        if not isinstance(d, dict):
            return d
        d.pop("monotone_constaints", None)
        for k, v in d.items():
            if isinstance(v, dict):
                remove_bad_keys(v)
        return d
    
    # Patch checkpoint
    print("   Patching checkpoint...")
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    
    # Remove stale hyperparameters recursively
    if "hyper_parameters" in ckpt:
        remove_bad_keys(ckpt["hyper_parameters"])
    
    # Force CPU device
    if "state_dict" in ckpt:
        for key in list(ckpt["state_dict"].keys()):
            if isinstance(ckpt["state_dict"][key], torch.Tensor):
                ckpt["state_dict"][key] = ckpt["state_dict"][key].cpu()
    
    # Save patched checkpoint
    patched_path = MODEL_PATH.with_name("tft_final_v3__patched.ckpt")
    torch.save(ckpt, str(patched_path))
    print(f"   Saved patched checkpoint: {patched_path}")
    
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(patched_path),
        map_location='cpu'
    )
    
    model.eval()
    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get model config
print("\n4. Model configuration...")
try:
    dp = model.dataset_parameters or {}
    print(f"   max_encoder_length: {dp.get('max_encoder_length')}")
    print(f"   max_prediction_length: {dp.get('max_prediction_length')}")
    print(f"   static_categoricals: {dp.get('static_categoricals')}")
    print(f"   time_varying_known_reals: {dp.get('time_varying_known_reals')}")
except Exception as e:
    print(f"   Could not read config: {e}")

# Test prediction on a sample group
print("\n5. Testing prediction...")
try:
    # Get a group with good history
    group_counts = data.groupby("group_id")["time_idx"].count()
    good_groups = group_counts[group_counts >= 36].index.tolist()
    
    if not good_groups:
        print("   No groups with 36+ months of history")
        good_groups = group_counts.nlargest(5).index.tolist()
    
    test_group = good_groups[0]
    print(f"   Test group: {test_group}")
    
    # Get history for this group
    group_data = data[data["group_id"] == test_group].copy()
    group_data = group_data.sort_values("time_idx")
    print(f"   History rows: {len(group_data)}")
    
    # Get latest values
    latest = group_data.iloc[-1]
    print(f"   Latest time_idx: {latest['time_idx']}")
    print(f"   Latest median_price: {latest['median_price']:.2f}")
    print(f"   Latest median_rent: {latest['median_rent']:.2f}")
    print(f"   Latest units_completing: {latest['units_completing']}")
    
    # Prepare prediction data (history + future rows)
    prediction_length = 12
    last_time_idx = int(group_data["time_idx"].max())
    
    future_rows = []
    for i in range(1, prediction_length + 1):
        future_row = group_data.iloc[-1].copy()
        future_row["time_idx"] = last_time_idx + i
        
        # Update known features
        future_month = (int(future_row.get("month", 1)) + i - 1) % 12 + 1
        future_row["month"] = str(future_month)
        future_row["quarter"] = str((future_month - 1) // 3 + 1)
        future_row["month_sin"] = np.sin(2 * np.pi * future_month / 12)
        future_row["month_cos"] = np.cos(2 * np.pi * future_month / 12)
        
        # units_completing: use recent average (this should come from actual project schedule!)
        uc = pd.to_numeric(group_data["units_completing"], errors="coerce").dropna()
        future_row["units_completing"] = float(uc.tail(6).mean()) if len(uc) else 0.0
        
        future_rows.append(future_row)
    
    pred_data = pd.concat([group_data, pd.DataFrame(future_rows)], ignore_index=True)
    
    # Ensure categoricals are strings
    for col in ["month", "quarter"]:
        if col in pred_data.columns:
            pred_data[col] = pred_data[col].astype(str)
    
    # Add has_actual_rent if missing (required by model)
    if "has_actual_rent" not in pred_data.columns:
        has_rent = (pd.to_numeric(pred_data["median_rent"], errors="coerce").fillna(0) > 0).any()
        pred_data["has_actual_rent"] = int(has_rent)
    
    # Ensure all categoricals used in training are strings
    for col in ["area_name", "property_type", "bedroom", "reg_type", 
                "developer_brand_label", "developer_umbrella"]:
        if col in pred_data.columns:
            pred_data[col] = pred_data[col].fillna("UNKNOWN").astype(str)
    
    # Fill ALL numeric columns with 0 (model doesn't accept NaN)
    for col in pred_data.columns:
        if col in ["group_id", "year_month"]:
            continue
        if pred_data[col].dtype == object:
            continue
        pred_data[col] = pd.to_numeric(pred_data[col], errors="coerce").fillna(0)
    
    # Make prediction
    print("\n   Running model.predict()...")
    with torch.no_grad():
        raw_output = model.predict(pred_data, mode="raw", return_x=False)
    
    # Extract predictions
    output = getattr(raw_output, "output", None) or getattr(raw_output, "prediction", None) or raw_output
    
    if isinstance(output, list) and len(output) >= 2:
        price_preds = output[0]  # [batch, horizon, quantiles]
        rent_preds = output[1]
        
        # Quantile indices: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        Q10, Q50, Q90 = 1, 3, 5
        
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS (12-month horizon)")
        print("=" * 70)
        print(f"\nPrice forecast (AED/sqm):")
        print(f"   10th percentile: {price_preds[0, -1, Q10].item():.2f}")
        print(f"   50th percentile: {price_preds[0, -1, Q50].item():.2f}")
        print(f"   90th percentile: {price_preds[0, -1, Q90].item():.2f}")
        
        print(f"\nRent forecast (AED/year):")
        print(f"   10th percentile: {rent_preds[0, -1, Q10].item():.2f}")
        print(f"   50th percentile: {rent_preds[0, -1, Q50].item():.2f}")
        print(f"   90th percentile: {rent_preds[0, -1, Q90].item():.2f}")
        
        # Calculate appreciation
        current_price = latest["median_price"]
        forecast_price = price_preds[0, -1, Q50].item()
        appreciation = ((forecast_price / current_price) - 1) * 100
        print(f"\nImplied appreciation: {appreciation:.1f}%")
        
        print("\n✅ Model prediction successful!")
    else:
        print(f"   Unexpected output format: {type(output)}")
        if hasattr(output, "shape"):
            print(f"   Shape: {output.shape}")

except Exception as e:
    import traceback
    print(f"❌ Prediction failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

