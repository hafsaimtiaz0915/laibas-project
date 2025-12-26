"""
TFT V2 Training Script for Google Colab
========================================
Upload Data/tft/runs/<BUILD_ID>/tft_training_data_v2.csv to Drive as tft_training_data_v2.csv
Then run this script cell-by-cell in Colab.
"""

# =============================================================================
# CELL 1 — Setup
# =============================================================================
# !pip install -q pytorch-forecasting==1.5.0 lightning==2.4.0

from google.colab import drive
drive.mount("/content/drive")

import subprocess, torch
subprocess.run(["bash", "-lc", "nvidia-smi -L && nvidia-smi"], check=False)

assert torch.cuda.is_available(), "FAIL: CUDA GPU not detected"
gpu_name = torch.cuda.get_device_name(0)
print("GPU:", gpu_name)
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# =============================================================================
# CELL 2 — Load Data
# =============================================================================
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

DATA_PATH = "/content/drive/MyDrive/Properly/tft_training_data_v2.csv"
data = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 60)
print("LOADED DATA")
print("=" * 60)
print("Rows:", len(data))
print("Columns:", len(data.columns))
print("Column names:", list(data.columns))

# =============================================================================
# CELL 3 — Config + Data Quality Gates
# =============================================================================
PRED_LEN_MONTHS = 12        # Forecast horizon (12 or 24)
ENCODER_LEN_MONTHS = 96     # Lookback window: 8 years (96 months) to capture full market cycles
MIN_GROUP_HISTORY = 24      # Cold-start filter: groups must have >= 2 years history

# Gate 1: Required columns
required_cols = ["time_idx", "group_id", "median_price", "median_rent", "units_completing"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise ValueError(f"GATE FAIL: Missing required columns: {missing}")

# Gate 2: units_completing must be non-trivial
uc = pd.to_numeric(data["units_completing"], errors="coerce").fillna(0)
pct_nonzero = (uc > 0).sum() / len(data) * 100
print(f"units_completing: {pct_nonzero:.1f}% rows > 0")
if pct_nonzero < 5:
    raise ValueError(f"GATE FAIL: units_completing is mostly zeros ({pct_nonzero:.1f}%). Check your data build.")

# Gate 3: Both OffPlan and Ready should exist
if "reg_type" in data.columns:
    reg_types = set(data["reg_type"].astype(str).unique())
    if "OffPlan" not in reg_types or "Ready" not in reg_types:
        print(f"WARNING: reg_type values are {reg_types}. Expected both OffPlan and Ready.")

print("✅ All gates passed")

# =============================================================================
# CELL 4 — Cold Start Filter + Feature Engineering
# =============================================================================
# Cold start filter: remove groups with insufficient history
group_lengths = data.groupby("group_id")["time_idx"].count()
short_groups = group_lengths[group_lengths < MIN_GROUP_HISTORY].index.tolist()
print(f"Groups with < {MIN_GROUP_HISTORY} months history: {len(short_groups)}")

if short_groups:
    before = len(data)
    data = data[~data["group_id"].isin(short_groups)]
    after = len(data)
    print(f"Filtered: {before:,} → {after:,} rows ({before - after:,} removed)")

# Ensure month/quarter exist and are categorical strings
data["month"] = data["month"].astype(int).astype(str)
data["quarter"] = data["quarter"].astype(int).astype(str)

# Ensure sin/cos exist
if "month_sin" not in data.columns:
    m = data["month"].astype(int)
    data["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * m / 12.0)

# Static categoricals
static_categoricals = []
for c in ["area_name", "property_type", "bedroom", "reg_type", "developer_brand_label", "developer_umbrella"]:
    if c in data.columns:
        data[c] = data[c].fillna("UNKNOWN").astype(str)
        static_categoricals.append(c)

print("Static categoricals:", static_categoricals)

# has_actual_rent flag
data["has_actual_rent"] = data.groupby("group_id")["median_rent"].transform(
    lambda x: (x.notna() & (x > 0)).any()
).astype(int)
print(f"Groups WITH actual rent: {data.loc[data['has_actual_rent']==1, 'group_id'].nunique():,}")
print(f"Groups WITHOUT rent: {data.loc[data['has_actual_rent']==0, 'group_id'].nunique():,}")

# Sort by group + time
data = data.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

# Fill targets
data["median_price"] = data.groupby("group_id")["median_price"].transform(lambda x: x.ffill().bfill())
data["median_rent"] = data.groupby("group_id")["median_rent"].transform(lambda x: x.ffill().bfill()).fillna(0)

# Identify numeric columns
protected = set(["group_id", "time_idx", "median_price", "median_rent", "month", "quarter"] + static_categoricals)
numeric_cols = [c for c in data.columns if c not in protected and pd.api.types.is_numeric_dtype(data[c])]

# Clean numeric
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

assert data["median_price"].notna().all(), "median_price still has NaNs"
print(f"\n✅ Ready: {data['group_id'].nunique():,} groups, {len(data):,} rows")

# =============================================================================
# CELL 5 — Train/Val Split
# =============================================================================
max_time = int(data["time_idx"].max())
cutoff = max_time - PRED_LEN_MONTHS

print("=" * 60)
print("TRAIN/VAL SPLIT")
print("=" * 60)
print(f"Forecast horizon: {PRED_LEN_MONTHS} months")
print(f"Encoder length: {ENCODER_LEN_MONTHS} months")
print(f"Cutoff time_idx: {cutoff}")

# =============================================================================
# CELL 6 — Create TimeSeriesDataSet
# =============================================================================
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

# Known reals: features we can generate into the future
known_reals = ["time_idx", "month_sin", "month_cos", "units_completing"]

# Unknown reals: features only observed historically
unknown_reals = [c for c in numeric_cols if c not in known_reals]
if "has_actual_rent" not in unknown_reals:
    unknown_reals.append("has_actual_rent")

print("Known reals (can generate forward):", known_reals)
print("Unknown reals count:", len(unknown_reals))

training = TimeSeriesDataSet(
    data[data["time_idx"] <= cutoff],
    time_idx="time_idx",
    target=["median_price", "median_rent"],
    group_ids=["group_id"],
    min_encoder_length=12,  # Minimum 1 year of history required
    max_encoder_length=ENCODER_LEN_MONTHS,  # Up to 8 years
    min_prediction_length=1,
    max_prediction_length=PRED_LEN_MONTHS,
    static_categoricals=static_categoricals,
    time_varying_known_categoricals=["month", "quarter"],
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=unknown_reals,
    target_normalizer=MultiNormalizer([
        GroupNormalizer(groups=["group_id"], transformation="softplus"),
        GroupNormalizer(groups=["group_id"], transformation="softplus"),
    ]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

print(f"\n✅ Training samples: {len(training):,}")
print(f"✅ Validation samples: {len(validation):,}")

# =============================================================================
# CELL 7 — Create Model
# =============================================================================
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

os.makedirs("/content/checkpoints", exist_ok=True)

batch_size = 512
train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    lstm_layers=2,
    output_size=[7, 7],
    loss=MultiLoss([QuantileLoss(), QuantileLoss()]),
    learning_rate=1e-3,
    reduce_on_plateau_patience=4,
)

print(f"Model parameters: {tft.size()/1e6:.2f}M")

# =============================================================================
# CELL 8 — Train
# =============================================================================
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision=32,  # Use float32 - bf16-mixed causes gradient issues with MultiLoss
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            dirpath="/content/checkpoints/",
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            every_n_epochs=1,
        ),
    ],
    enable_progress_bar=True,
)

print("=" * 60)
print(f"TRAINING: horizon={PRED_LEN_MONTHS}m, encoder={ENCODER_LEN_MONTHS}m")
print("=" * 60)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

# =============================================================================
# CELL 9 — Evaluate + Save
# =============================================================================
import shutil, glob

best_ckpt = trainer.checkpoint_callback.best_model_path
print("Best checkpoint:", best_ckpt)

# Compute validation metrics
best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
predictions = best_model.predict(val_dl, mode="prediction", return_y=True)

# Multi-target returns lists: predictions.output is [price_preds, rent_preds]
# Each is shape (batch, horizon, quantiles) - take median quantile (index 3)
y_pred_price = predictions.output[0]  # Price predictions
y_pred_rent = predictions.output[1]   # Rent predictions
y_true_price = predictions.y[0]       # True price
y_true_rent = predictions.y[1]        # True rent

# Take median quantile (index 3 of 7 quantiles: 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
price_mae = (y_pred_price[..., 3] - y_true_price).abs().mean().item()
rent_mae = (y_pred_rent[..., 3] - y_true_rent).abs().mean().item()

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"Price MAE: {price_mae:.2f} AED/sqft")
print(f"Rent MAE: {rent_mae:.2f} AED/year")

# Save to Drive
out_dir = "/content/drive/MyDrive/Properly/checkpoints_v2/"
os.makedirs(out_dir, exist_ok=True)

shutil.copy(best_ckpt, os.path.join(out_dir, "tft_best_v2.ckpt"))
for p in glob.glob("/content/checkpoints/*.ckpt"):
    shutil.copy(p, out_dir)

print(f"\n✅ Checkpoint saved to: {out_dir}")

# =============================================================================
# CELL 10 — Write Training Manifest
# =============================================================================
manifest = {
    "trained_at": datetime.utcnow().isoformat() + "Z",
    "data_source": DATA_PATH,
    "rows_after_filter": len(data),
    "groups_after_filter": int(data["group_id"].nunique()),
    
    "horizon": {
        "pred_len_months": PRED_LEN_MONTHS,
        "encoder_len_months": ENCODER_LEN_MONTHS,
        "min_group_history": MIN_GROUP_HISTORY,
    },
    
    "group_id_definition": {
        "source": "V2 builder: area_id_property_type_bedroom_reg_type",
        "example": data["group_id"].iloc[0] if len(data) > 0 else None,
    },
    
    "features": {
        "static_categoricals": static_categoricals,
        "time_varying_known_categoricals": ["month", "quarter"],
        "time_varying_known_reals": known_reals,
        "time_varying_unknown_reals": unknown_reals,
    },
    
    "targets": ["median_price", "median_rent"],
    
    "validation_metrics": {
        "price_mae": round(price_mae, 2),
        "rent_mae": round(rent_mae, 2),
    },
    
    "inference_contract": {
        "units_completing": {
            "type": "known_future_real",
            "generation": "MUST compute from Projects.csv schedule for future months",
            "fallback": "DO NOT use rolling average - it will degrade forecasts",
        }
    },
    
    "checkpoint": "tft_best_v2.ckpt",
}

manifest_path = os.path.join(out_dir, "training_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Manifest saved to: {manifest_path}")
print("\nManifest contents:")
print(json.dumps(manifest, indent=2))

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Checkpoint: {out_dir}tft_best_v2.ckpt")
print(f"Manifest: {out_dir}training_manifest.json")

