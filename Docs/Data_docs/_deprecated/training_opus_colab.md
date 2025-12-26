"""
TFT training (multi-target: price + rent) — Colab pasteable

IMPORTANT:
- Do NOT paste markdown fences like ```python into a Colab CODE cell.
- This file contains ONLY executable Python so you can paste any “CELL” section into a code cell.

Key settings:
- PRED_LEN_MONTHS controls the maximum forecast horizon (default 12; set 24 if you truly want a direct 24m model).
- ENCODER_LEN_MONTHS controls the lookback window the TFT can use as context (default 36).
"""

"""CELL 1 — Setup"""

!pip install pytorch-forecasting==1.5.0 lightning==2.4.0 -q

from google.colab import drive
drive.mount("/content/drive")

import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: GPU not detected")

"""CELL 2 — Load data"""

import pandas as pd
import numpy as np

DATA_PATH = "/content/drive/MyDrive/Properly/tft_training_data_1.csv"
data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(data):,} rows, {data['group_id'].nunique():,} groups")
print("Columns:", len(data.columns))

"""CELL 3 — Data prep (dtype fixes + rent flag + NaN cleanup)"""

PRED_LEN_MONTHS = 12   # set to 24 if you truly want a 24-month direct forecast
ENCODER_LEN_MONTHS = 36

required_cols = [
    "group_id", "time_idx",
    "median_price", "median_rent",
    "area_name", "property_type", "bedroom", "reg_type", "developer_name",
    "month", "quarter",
    "month_sin", "month_cos",
    "units_completing",
]
missing_required = [c for c in required_cols if c not in data.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

static_cols = ["area_name", "property_type", "bedroom", "reg_type", "developer_name", "group_id"]
for opt in ["developer_registered_name", "developer_brand", "reg_type_dld"]:
    if opt in data.columns and opt not in static_cols:
        static_cols.append(opt)
for col in static_cols:
    data[col] = data[col].astype(str)

data["month"] = data["month"].astype(str)
data["quarter"] = data["quarter"].astype(str)

data = data.sort_values(["group_id", "time_idx"])

data["has_actual_rent"] = data.groupby("group_id")["median_rent"].transform(
    lambda x: (x.notna() & (x > 0)).any()
).astype(int)

print(f"Groups WITH actual rent: {data.loc[data['has_actual_rent']==1, 'group_id'].nunique():,}")
print(f"Groups WITHOUT rent:     {data.loc[data['has_actual_rent']==0, 'group_id'].nunique():,}")

data["median_price"] = data.groupby("group_id")["median_price"].transform(lambda x: x.ffill().bfill())
data["median_rent"] = data.groupby("group_id")["median_rent"].transform(lambda x: x.ffill().bfill())
data["median_rent"] = data["median_rent"].fillna(0)

numeric_cols = [
    "median_rent_sqft",
    "months_since_launch", "months_to_handover",
    "months_to_handover_signed", "months_since_handover", "handover_window_6m",
    "dld_offplan_after_handover",
    "project_percent_complete", "project_duration_months", "phase_ratio",
    "avg_building_floors", "avg_building_flats",
    "avg_building_floors_missing", "avg_building_flats_missing",
    "dev_total_projects", "dev_completed_projects", "dev_total_units", "dev_avg_completion",
    "dev_overall_median_price", "dev_overall_transactions",
    "market_median_price", "market_transactions",
    "govt_valuation_median", "valuation_count", "govt_valuation_median_missing",
    "supply_units", "supply_buildings", "supply_villas", "active_projects",
    "units_registered", "buildings_registered", "units_completing",
    "transaction_count", "rent_count",
    "eibor_overnight", "eibor_1w", "eibor_1m", "eibor_3m", "eibor_6m", "eibor_12m",
    "eibor_missing",
    "visitors_total", "hotel_rooms", "hotel_apartments",
    "tourism_missing",
    "month_sin", "month_cos",
]
present_numeric_cols = [c for c in numeric_cols if c in data.columns]
data[present_numeric_cols] = (
    data[present_numeric_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)
print(f"Filled NaN/inf in {len(present_numeric_cols)} numeric features")

valid_price = data.groupby("group_id")["median_price"].apply(lambda x: x.notna().any())
data = data[data["group_id"].isin(valid_price[valid_price].index)]

print(f"After cleaning: {len(data):,} rows, {data['group_id'].nunique():,} groups")

nan_sum = data[present_numeric_cols].isna().sum().sum()
print("✓ No NaN in numeric features" if nan_sum == 0 else f"WARNING: NaNs remain: {nan_sum}")

"""CELL 4 — Train/Val split (aligned to the forecast horizon)"""

max_time = int(data["time_idx"].max())
cutoff = max_time - int(PRED_LEN_MONTHS)

print(f"Forecast horizon (months): {PRED_LEN_MONTHS}")
print(f"Cutoff time_idx: {cutoff}  (validation = last {PRED_LEN_MONTHS} months)")

"""CELL 5 — Create TFT dataset"""

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

static_categoricals = ["area_name", "property_type", "bedroom", "reg_type", "developer_name"]
for opt in ["developer_registered_name", "developer_brand", "reg_type_dld"]:
    if opt in data.columns and opt not in static_categoricals:
        static_categoricals.append(opt)

known_reals = ["time_idx", "month_sin", "month_cos", "units_completing"]
missing_known = [c for c in known_reals if c not in data.columns]
if missing_known:
    raise ValueError(f"Missing required known real features: {missing_known}")

unknown_reals = [
    "transaction_count", "rent_count", "median_rent_sqft",
    "has_actual_rent",
    "months_since_launch", "months_to_handover",
    "months_to_handover_signed", "months_since_handover", "handover_window_6m",
    "dld_offplan_after_handover",
    "project_percent_complete", "project_duration_months", "phase_ratio",
    "supply_units", "supply_buildings", "supply_villas", "active_projects",
    "units_registered", "buildings_registered",
    "avg_building_floors", "avg_building_flats",
    "avg_building_floors_missing", "avg_building_flats_missing",
    "dev_total_projects", "dev_completed_projects", "dev_total_units", "dev_avg_completion",
    "dev_overall_median_price", "dev_overall_transactions",
    "market_median_price", "market_transactions",
    "govt_valuation_median", "valuation_count", "govt_valuation_median_missing",
    "eibor_overnight", "eibor_1w", "eibor_1m", "eibor_3m", "eibor_6m", "eibor_12m",
    "eibor_missing",
    "visitors_total", "hotel_rooms", "hotel_apartments",
    "tourism_missing",
]
unknown_reals = [c for c in unknown_reals if c in data.columns]

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
print(f"Training samples:   {len(training):,}")
print(f"Validation samples: {len(validation):,}")
print(f"Static categoricals: {static_categoricals}")
print(f"Known reals: {known_reals}")
print(f"Unknown reals used: {len(unknown_reals)}")

"""CELL 6 — Create model + dataloaders"""

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import os

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
    learning_rate=0.001,
    reduce_on_plateau_patience=4,
)

if torch.cuda.is_available():
    tft = tft.cuda()
    print(f"Model device: {next(tft.parameters()).device}")
print(f"Parameters: {tft.size()/1e6:.2f}M")

"""CELL 7 — Train"""

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

print(f"Training multi-target TFT (price + rent) with horizon={PRED_LEN_MONTHS} months...")
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

print("\nCopying checkpoints to Drive...")
!mkdir -p /content/drive/MyDrive/Properly/checkpoints/
!cp /content/checkpoints/*.ckpt /content/drive/MyDrive/Properly/checkpoints/
print("✅ Checkpoints copied to Drive!")

"""CELL 8 — Save best model"""

import shutil

best_ckpt = trainer.checkpoint_callback.best_model_path
print(f"Best checkpoint: {best_ckpt}")

out_path = "/content/drive/MyDrive/Properly/tft_final.ckpt"
shutil.copy(best_ckpt, out_path)
print(f"Saved best model to: {out_path}")

print("\nModel predicts BOTH:")
print(" - median_price (capital appreciation proxy, per-sqft series)")
print(" - median_rent  (rental yield proxy)")