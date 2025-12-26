# Rent Imputation Solution for Multi-Target TFT Training

**Date**: December 19, 2025  
**Author**: Automated via Cursor AI  
**Status**: Implemented and tested  

---

## Problem Statement

Training the TFT model was blocked because `median_rent` and `median_rent_sqft` had ~14% NaN values in the training dataset (`Data/tft/latest/tft_training_data_v2.csv`).

### Key constraints
- The TFT model predicts **two targets simultaneously**: `median_price` (sale price/sqft) and `median_rent` (annual rent)
- **Dropping rows** with missing rent would also remove those time steps from **price training** — unacceptable data loss
- **Filling with 0** (what the training script was doing) is wrong — 0 rent is not a real economic state and creates artificial shocks

---

## Root Cause Analysis

### Why rent is missing

Rent data is joined to the training dataset at the key `(year_month, area_id, bedroom)`:

```python
df = transactions.merge(rents, on=["year_month", "area_id", "bedroom"], how="left")
```

When there are **no rent contracts** for a specific `(year_month, area_id, bedroom)` combination, the LEFT join produces:
- `rent_count = 0`
- `median_rent = NaN`
- `median_rent_sqft = NaN`

### Quantified breakdown (pre-imputation)

| Metric | Value |
|--------|-------|
| Total rows | 272,257 |
| Rows with missing `median_rent` | 37,454 (13.76%) |
| Rows with missing `median_rent_sqft` | 37,836 (13.90%) |
| Groups with 100% missing rent (zero observations ever) | 110 |

### Key finding: missingness correlates perfectly with `rent_count == 0`

Cross-tabulation confirmed:
```
                       rent_count==0
median_rent_missing    False    True
False                 234,803      0
True                        0  37,454
```

Every row with missing rent had `rent_count == 0`. No parsing issues, no Inf values, no string artifacts — these are **true missing joins**.

### Time distribution of missingness

- Worst months: 2008–2009 had **100% missing rent** (no rent contract data in that period)
- Best months: 2022–2023 had only **~2% missing rent**

### Property type distribution

| Bedroom | Missing rate |
|---------|--------------|
| Room | 100% |
| Penthouse | 66.3% |
| 6BR+ | 46.2% |
| 5BR | 40.0% |
| 1BR–3BR | 12–16% |

---

## Solution: Tiered Rent Imputation

We implemented a **3-tier imputation strategy** that fills missing rent values without dropping any rows:

### Tier 1: Forward-fill + Backward-fill within `group_id`

**Method**: For each time series (`group_id`), propagate the last observed rent forward, and if leading months are missing, propagate the first observed rent backward.

**Why it works**: If a specific segment (e.g., "Marina, 2BR, Ready") had rent data in some months but not others, the most reasonable estimate for a gap is the last observed rent for that exact segment.

**Result**: Fixed **34,428 rows** (91.0% of missing rent)

```python
df["median_rent"] = df.groupby("group_id")["median_rent"].transform(lambda x: x.ffill().bfill())
```

### Tier 2: Area-level median fallback

**Method**: For groups where the **entire series** has zero rent observations (e.g., area 432 has no rent contracts at all), use the median rent from **other groups in the same area** (different bedroom/property_type combinations).

**Why it works**: If there's no 1BR rent data for an area but there IS 2BR/3BR rent data, the area-level median provides a reasonable local estimate.

**Result**: Fixed **2,020 rows** (5.3% of missing rent)

```python
area_medians = df.loc[~orig_rent_missing].groupby("area_id").agg(
    area_median_rent=("median_rent", "median"),
    area_median_rent_sqft=("median_rent_sqft", "median"),
)
df.loc[miss, "median_rent"] = df.loc[miss, "area_id"].map(area_medians["area_median_rent"])
```

### Tier 3: Global median fallback

**Method**: For areas with **no rent data at all** (rare), use the global median rent across the entire dataset.

**Why it works**: Last resort — better than 0 or dropping the row, and the imputation flag preserves transparency.

**Result**: Fixed **1,388 rows** (3.7% of missing rent)

```python
global_median_rent = df.loc[~orig_rent_missing, "median_rent"].median()
df.loc[df["median_rent"].isna(), "median_rent"] = global_median_rent
```

---

## Summary of imputation results

| Tier | Method | Rows filled | % of missing |
|------|--------|-------------|--------------|
| 1 | `group_id` ffill+bfill | 34,428 | 91.0% |
| 2 | Area-level median | 2,020 | 5.3% |
| 3 | Global median | 1,388 | 3.7% |
| **Total** | — | **37,836** | **100%** |

### Post-imputation state

| Metric | Value |
|--------|-------|
| Total rows | 272,257 (unchanged — no rows dropped) |
| `median_rent` NaN | **0** |
| `median_rent_sqft` NaN | **0** |
| Rows with imputed rent | 37,836 (13.90%) |
| Rows with observed rent | 234,421 (86.10%) |

---

## Imputation tracking columns

Two new columns were added to the dataset for transparency:

| Column | Type | Description |
|--------|------|-------------|
| `rent_imputed` | int (0/1) | Flag: 1 if rent was imputed, 0 if observed |
| `rent_imputation_source` | string | Source of imputation: `group_ffill_bfill`, `area_median`, `global_median`, or null (observed) |

### Breakdown by source

```
rent_imputation_source
<NA>                  234,421  (observed)
group_ffill_bfill      34,428  (tier 1)
area_median             2,020  (tier 2)
global_median           1,388  (tier 3)
```

---

## Files produced

### 1. Training-ready dataset (use for training)

**Path**: `Data/tft/latest/tft_training_data_v2_imputed.csv`

- 272,257 rows
- 0 NaN in rent targets
- Includes `rent_imputed` and `rent_imputation_source` columns

### 2. Original dataset (pre-imputation backup)

**Path**: `Data/tft/latest/tft_training_data_v2.csv`

- Unchanged original with NaN rent values

---

## Code changes

### 1. Pipeline update: `scripts/build_tft_data_v2.py`

Added config option:

```python
@dataclass(frozen=True)
class V2Config:
    # ...
    # Strategy for handling missing rent values:
    #   "tiered" = ffill+bfill within group → area-level median → global median
    #   "drop"   = drop rows where rent is missing (loses price data too)
    #   "none"   = leave NaNs as-is (training script must handle)
    rent_imputation_strategy: str = "tiered"
```

The pipeline now automatically applies tiered imputation when `rent_imputation_strategy = "tiered"` (default).

### 2. One-off imputation script (for immediate use)

The imputation was also applied directly to the existing dataset via an inline Python script. The logic is:

```python
import pandas as pd
import numpy as np

# Load
df = pd.read_csv("Data/tft/latest/tft_training_data_v2.csv")
df = df.sort_values(["group_id", "time_idx"])

# Track original missing
orig_rent_missing = df["median_rent"].isna() | df["median_rent_sqft"].isna()
df["rent_imputed"] = 0
df["rent_imputation_source"] = pd.NA

# TIER 1: ffill+bfill within group
for rent_col in ["median_rent", "median_rent_sqft"]:
    df[rent_col] = df.groupby("group_id")[rent_col].transform(lambda x: x.ffill().bfill())

tier1_imputed = orig_rent_missing & df["median_rent"].notna()
df.loc[tier1_imputed, "rent_imputed"] = 1
df.loc[tier1_imputed, "rent_imputation_source"] = "group_ffill_bfill"

# TIER 2: area-level median
still_missing = df["median_rent"].isna() | df["median_rent_sqft"].isna()
if still_missing.any():
    area_medians = df.loc[~orig_rent_missing].groupby("area_id").agg(
        area_median_rent=("median_rent", "median"),
        area_median_rent_sqft=("median_rent_sqft", "median"),
    )
    for rent_col, area_col in [("median_rent", "area_median_rent"), ("median_rent_sqft", "area_median_rent_sqft")]:
        miss = df[rent_col].isna()
        df.loc[miss, rent_col] = df.loc[miss, "area_id"].map(area_medians[area_col])
    
    tier2_imputed = orig_rent_missing & ~tier1_imputed & df["median_rent"].notna()
    df.loc[tier2_imputed, "rent_imputed"] = 1
    df.loc[tier2_imputed, "rent_imputation_source"] = "area_median"

# TIER 3: global median
still_missing = df["median_rent"].isna() | df["median_rent_sqft"].isna()
if still_missing.any():
    global_median_rent = df.loc[~orig_rent_missing, "median_rent"].median()
    global_median_sqft = df.loc[~orig_rent_missing, "median_rent_sqft"].median()
    df.loc[df["median_rent"].isna(), "median_rent"] = global_median_rent
    df.loc[df["median_rent_sqft"].isna(), "median_rent_sqft"] = global_median_sqft
    
    tier3_imputed = orig_rent_missing & df["rent_imputation_source"].isna()
    df.loc[tier3_imputed, "rent_imputed"] = 1
    df.loc[tier3_imputed, "rent_imputation_source"] = "global_median"

# Save
df.to_csv("Data/tft/latest/tft_training_data_v2_imputed.csv", index=False)
```

---

## Training script implications

The training script (`Docs/Data_docs/training_colab_v2.py`) previously had:

```python
data["median_rent"] = data.groupby("group_id")["median_rent"].transform(lambda x: x.ffill().bfill()).fillna(0)
```

The `.fillna(0)` was the problem — it filled unfillable gaps (entire groups with no rent) with **0**, which is not a real rent value.

### Recommended update for training script

Since the pipeline now handles imputation, the training script can simply:

```python
# Rent is already imputed by the pipeline — just validate
assert data["median_rent"].notna().all(), "median_rent has NaNs — use the imputed dataset"
assert data["median_rent_sqft"].notna().all(), "median_rent_sqft has NaNs — use the imputed dataset"
```

Or, if you want to add the imputation flag as a feature:

```python
# Add rent_imputed as an unknown real (model can learn "this rent was estimated")
if "rent_imputed" not in unknown_reals:
    unknown_reals.append("rent_imputed")
```

---

## Why this approach is better than alternatives

| Alternative | Problem |
|-------------|---------|
| Drop rows with missing rent | Loses price data for those time steps (14% data loss) |
| Fill with 0 | 0 rent is not real; creates artificial shocks; model learns wrong patterns |
| Fill with global mean everywhere | Ignores local variation; area-level is more accurate |
| Leave NaN | PyTorch Forecasting's `TimeSeriesDataSet` doesn't handle NaN targets well |

### Our approach
- **Preserves all rows** (no price data loss)
- **Uses the best local estimate** (same group → same area → global, in order of preference)
- **Tracks imputation** so the model can learn to weight imputed vs observed differently if needed

---

## Diagnostic artifacts

The following diagnostic files were also produced during the analysis:

| File | Description |
|------|-------------|
| `Data/tft/latest/diagnostics_top50_missing_rent_groups.csv` | Top 50 group_ids contributing most missing rent rows |
| `Data/tft/latest/diagnostics_rent_missingness_v2/rent_missingness_report.json` | Full JSON diagnostic report |
| `Data/tft/latest/tft_training_data_v2__rent_observed_only.csv` | Alternative: rent-observed-only dataset (for rent-only model) |

---

## Future considerations

1. **If rent contracts data improves**: Re-run the pipeline (`python scripts/build_tft_data_v2.py`) — imputation will be applied automatically, and the percentage of imputed rows should decrease.

2. **If you want to train a rent-only model**: Use `rent_imputation_strategy = "drop"` to drop rows with no observed rent (this is acceptable when rent is the only target).

3. **Weighted loss**: Consider reducing the rent loss weight for imputed rows during training if you want the model to focus on observed rent.

---

## Verification

After imputation:
```
median_rent NaN: 0
median_rent_sqft NaN: 0
Total rows: 272,257 (unchanged)
```

The dataset is now ready for TFT training with both price and rent targets.

