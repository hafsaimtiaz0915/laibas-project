Cell 1

!pip install -q pytorch-forecasting==1.5.0 lightning==2.4.0

from google.colab import drive
drive.mount("/content/drive")

import subprocess, torch
subprocess.run(["bash", "-lc", "nvidia-smi -L && nvidia-smi"], check=False)

assert torch.cuda.is_available(), "FAIL: CUDA GPU not detected"
gpu_name = torch.cuda.get_device_name(0)
print("GPU:", gpu_name)
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


Cell 2

import pandas as pd
import numpy as np

DATA_PATH = "/content/drive/MyDrive/Properly/tft_training_data_v2.csv"
data = pd.read_csv(DATA_PATH)

print("Loaded rows:", len(data))
print("Columns:", len(data.columns))
print("Head columns:", list(data.columns)[:30])

Cell 3

PRED_LEN_MONTHS = 12        # 12-month forecast horizon (set 24 if you truly want direct 24m)
ENCODER_LEN_MONTHS = 36     # lookback window (trend context)

# --- Required targets ---
required_targets = ["median_price", "median_rent"]
for c in required_targets:
    if c not in data.columns:
        raise ValueError(f"Missing target column: {c}")

# --- time_idx ---
if "time_idx" not in data.columns:
    raise ValueError("Missing time_idx. v2 must include time_idx for TFT.")

# --- future supply schedule (CRITICAL for investor lens) ---
# We want a scheduled completions series per month (area-level), i.e. "units completing in this month".
# This should be produced by the V2 builder from Projects' handover schedule.
#
# NOTE: If this is missing, you are training WITHOUT the most important supply-pressure driver.
if "units_completing" not in data.columns:
    raise ValueError(
        "Missing units_completing in v2 training data. "
        "Rebuild using scripts/build_tft_data_v2.py that emits a monthly scheduled completions series."
    )

# --- group_id ---
# v1 expected group_id; if v2 doesn't have it, build one from common identifiers.
if "group_id" not in data.columns:
    candidate_keys = [c for c in ["project_id", "building_id", "property_id", "unit_type_id"] if c in data.columns]
    if not candidate_keys:
        # fallback: build a stable group from area + beds + type + developer label if available
        fallback_parts = [c for c in ["area_name", "bedroom", "property_type", "developer_brand_label"] if c in data.columns]
        if not fallback_parts:
            raise ValueError("No group_id and no suitable columns to construct it. Need group_id or project/building keys.")
        data["group_id"] = data[fallback_parts].astype(str).agg("|".join, axis=1)
        print("Constructed group_id from:", fallback_parts)
    else:
        data["group_id"] = data[candidate_keys].astype(str).agg("|".join, axis=1)
        print("Constructed group_id from:", candidate_keys)

# --- seasonality features: month_sin / month_cos ---
# If month is available use it; else infer month from time_idx modulo 12 (works if time_idx increments monthly).
if "month" not in data.columns:
    data["month"] = ((data["time_idx"].astype(int) - data["time_idx"].astype(int).min()) % 12 + 1).astype(int)

data["month"] = data["month"].astype(int)
if "quarter" not in data.columns:
    data["quarter"] = ((data["month"] - 1) // 3 + 1).astype(int)

# Create sin/cos if absent
if "month_sin" not in data.columns or "month_cos" not in data.columns:
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

# --- categoricals (use v2 columns; do NOT require developer_name) ---
static_categoricals = []
for c in ["area_name", "property_type", "bedroom", "reg_type", "developer_brand_label", "developer_umbrella"]:
    if c in data.columns:
        static_categoricals.append(c)

# Ensure month/quarter are strings for known categoricals
data["month"] = data["month"].astype(str)
data["quarter"] = data["quarter"].astype(str)

for c in static_categoricals:
    data[c] = data[c].astype(str)

# --- numeric cleanup ---
# We'll treat all non-categorical, non-id columns as candidates, but keep it safe.
protected = set(["group_id", "time_idx"] + required_targets + ["month", "quarter"] + static_categoricals)
numeric_candidates = [c for c in data.columns if c not in protected]

# Keep only numeric-like columns
numeric_cols = []
for c in numeric_candidates:
    if pd.api.types.is_numeric_dtype(data[c]):
        numeric_cols.append(c)

data = data.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

# Forward/back fill targets per group (keeps trend continuity), then rent->0 if still missing
data["median_price"] = data.groupby("group_id")["median_price"].transform(lambda x: x.ffill().bfill())
data["median_rent"]  = data.groupby("group_id")["median_rent"].transform(lambda x: x.ffill().bfill()).fillna(0)

# Replace inf/nan in numeric covariates
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Sanity
assert data["median_price"].notna().all(), "median_price still has NaNs after fill"
print("OK. groups:", data["group_id"].nunique(), "rows:", len(data))
print("Static categoricals:", static_categoricals)
print("Numeric covariates used:", len(numeric_cols))

Cell 4 

max_time = int(data["time_idx"].max())
cutoff = max_time - int(PRED_LEN_MONTHS)

print("Forecast horizon months:", PRED_LEN_MONTHS)
print("Cutoff time_idx:", cutoff)
print("Train max time_idx:", int(data.loc[data["time_idx"] <= cutoff, "time_idx"].max()))
print("Val min time_idx:", int(data.loc[data["time_idx"] > cutoff, "time_idx"].min()))

Cell 5 

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

# Known reals = things we know into the future (seasonality and planned supply, etc.)
known_reals = ["time_idx", "month_sin", "month_cos"]
for c in ["units_completing", "supply_units", "active_projects"]:
    if c in data.columns and c in numeric_cols:
        known_reals.append(c)

# Unknown reals = observed signals (transactions, macro series, valuations...) the model can use historically
unknown_reals = [c for c in numeric_cols if c not in known_reals]

training = TimeSeriesDataSet(
    data[data["time_idx"] <= cutoff],
    time_idx="time_idx",
    target=["median_price", "median_rent"],
    group_ids=["group_id"],
    min_encoder_length=1,
    max_encoder_length=int(ENCODER_LEN_MONTHS),
    min_prediction_length=1,
    max_prediction_length=int(PRED_LEN_MONTHS),
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

print("Training samples:", len(training))
print("Validation samples:", len(validation))
print("Known reals:", known_reals)
print("Unknown reals count:", len(unknown_reals))

Cell 6 

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import os

os.makedirs("/content/checkpoints", exist_ok=True)

batch_size = 512
train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dl   = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    lstm_layers=2,
    output_size=[7, 7],  # quantiles
    loss=MultiLoss([QuantileLoss(), QuantileLoss()]),
    learning_rate=1e-3,
    reduce_on_plateau_patience=4,
)

print(f"Parameters: {tft.size()/1e6:.2f}M")

Cell 7

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="bf16-mixed" if torch.cuda.is_available() else 32,
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

print(f"Training TFT (price + rent), horizon={PRED_LEN_MONTHS} months, encoder={ENCODER_LEN_MONTHS} months...")
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

Cell 8 

import shutil, glob, os

best_ckpt = trainer.checkpoint_callback.best_model_path
print("Best checkpoint:", best_ckpt)

out_dir = "/content/drive/MyDrive/Properly/checkpoints_v2/"
os.makedirs(out_dir, exist_ok=True)

# Copy best + top-k
shutil.copy(best_ckpt, os.path.join(out_dir, "tft_best_v2.ckpt"))
for p in glob.glob("/content/checkpoints/*.ckpt"):
    shutil.copy(p, out_dir)

print("✅ Saved to:", out_dir)

Cell 9 

from pytorch_forecasting.metrics import MAE

pred = tft.predict(val_dl, mode="prediction", return_y=True, trainer=trainer)
y_true = pred.y[0]  # (batch, horizon, targets)
y_pred = pred.output # same shape

# target index: 0=price, 1=rent
price_mae = (y_pred[...,0] - y_true[...,0]).abs().mean().item()
rent_mae  = (y_pred[...,1] - y_true[...,1]).abs().mean().item()

print("Val MAE (price):", price_mae)
print("Val MAE (rent): ", rent_mae)