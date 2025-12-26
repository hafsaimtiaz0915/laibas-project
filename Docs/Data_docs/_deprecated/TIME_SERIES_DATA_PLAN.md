# TFT Data Preparation Plan

> **Document Version**: 2.0  
> **Created**: 2025-12-10  
> **Last Updated**: 2025-12-10  
> **Purpose**: Define the data structure and preparation pipeline for training a Temporal Fusion Transformer (TFT) on Dubai real estate data.

---

## 1. Architecture Overview

### 1.1 Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT QUERY                                      │
│            "Binghatti JVC 2BR 2.2M - what's the outlook?"               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TFT MODEL (Runs on Mac)                               │
│                                                                          │
│  • Trained on Google Colab (GPU, 1-2 hours)                             │
│  • Downloaded checkpoint runs locally on CPU                            │
│  • Learns patterns from raw aggregated data                             │
│                                                                          │
│  Outputs:                                                                │
│  • Numerical predictions (price at handover, appreciation %)            │
│  • Attention weights (what influenced the prediction)                   │
│  • Confidence intervals                                                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM AGENT (Claude API)                                │
│                                                                          │
│  Receives:                                                               │
│  • TFT predictions + attention weights                                  │
│  • Context from lookup tables                                           │
│  • System prompt (controls output format)                               │
│                                                                          │
│  Outputs:                                                                │
│  • Natural language market briefing for agents                          │
│  • Factual, NOT investment advice                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Principles

| Principle | Description |
|-----------|-------------|
| **No Feature Engineering** | TFT learns patterns from raw data - we don't compute lag, volatility, momentum |
| **Raw Aggregations Only** | Only median, count - no ratios, slopes, or derived values |
| **TFT Handles Complexity** | Model discovers correlations between EIBOR, supply, developer history |
| **LLM for Explanation** | TFT outputs numbers, LLM converts to natural language |
| **Train on Colab, Run on Mac** | $10/month Colab Pro for training, free inference locally |

---

## 2. Source Data Inventory

### 2.1 Cleaned Data Available

| File | Records | Date Range | Status |
|------|---------|------------|--------|
| `Transactions_Cleaned.csv` | 1,606,520 | 2000-2025 | ✅ Ready |
| `Rent_Contracts_Cleaned.csv` | 5,743,849 | 2007-2025 | ✅ Ready |
| `Projects_Cleaned.csv` | 3,039 | 2003-2028 | ✅ Ready |
| `Units_Cleaned.csv` | 2,335,623 | - | ✅ Ready |
| `Buildings_Cleaned.csv` | 239,277 | - | ✅ Ready |
| `Valuation_Cleaned.csv` | 87,093 | - | ✅ Ready |
| `eibor_monthly.csv` | 182 months | 2009-2025 | ✅ Ready |
| `tourism_visitors.csv` | 70 records | 2024-2025 | ✅ Ready |
| `tourism_inventory.csv` | 10 records | 2023-2025 | ✅ Ready |

---

## 3. TFT Data Requirements

### 3.1 PyTorch Forecasting Format

TFT requires a specific data structure:

```python
from pytorch_forecasting import TimeSeriesDataSet

training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",                    # Integer 0, 1, 2...
    target=["median_price", "median_rent"], # MULTI-TARGET: price + rent
    group_ids=["group_id"],                 # Unique per series
    
    # Don't change over time for a group
    static_categoricals=[
        "area_name", 
        "property_type", 
        "bedroom",
        "reg_type",                         # OffPlan vs Ready
        "developer_name"                    # 96 unique developers
    ],
    
    # Known in advance (calendar + supply schedule)
    time_varying_known_categoricals=["month", "quarter"],
    time_varying_known_reals=[
        "month_sin", "month_cos",
        "units_completing"                  # Bedroom-level supply schedule (known from off-plan contracts)
    ],
    
    # Only known historically
    time_varying_unknown_reals=[
        # Group-specific
        "median_price",                     # TARGET
        "transaction_count", 
        "median_rent",
        "rent_count",
        "median_rent_sqft",
        # Project phase (off-plan appreciation tracking)
        "months_since_launch", "months_to_handover",
        "project_percent_complete", "project_duration_months", "phase_ratio",
        # Supply
        "supply_units", "supply_buildings", "supply_villas", "active_projects",
        # Registrations
        "units_registered", "buildings_registered",
        "avg_building_floors", "avg_building_flats",
        # Developer stats (from projects)
        "dev_total_projects", "dev_completed_projects", 
        "dev_total_units", "dev_avg_completion",
        # HIERARCHICAL CONTEXT - Developer Overall
        "dev_overall_median_price",         # This developer's overall performance
        "dev_overall_transactions",
        # HIERARCHICAL CONTEXT - Market Overall
        "market_median_price",              # Entire market benchmark
        "market_transactions",
        # Valuations
        "govt_valuation_median", "valuation_count",
        # EIBOR (all raw rates)
        "eibor_overnight", "eibor_1w", "eibor_1m", 
        "eibor_3m", "eibor_6m", "eibor_12m",
        # Tourism
        "visitors_total", "hotel_rooms", "hotel_apartments"
    ],
    
    min_encoder_length=12,                  # Minimum 1 year history
    max_encoder_length=96,                  # Up to 8 years (full market cycle)
    min_prediction_length=1,                # Can predict just next month
    max_prediction_length=60                # Up to 5 years (full off-plan horizon)
)
```

### 3.2 Column Specifications (49 Total)

**TFT Required Columns:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `time_idx` | int | Required index | Computed | 0, 1, 2... per group |
| `year_month` | str | Reference | All | YYYY-MM format |
| `group_id` | str | group_ids | Computed | `{area}_{property_type}_{bedroom}_{reg_type}_{developer}` |

**Static Categoricals (don't change within a group):**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `area_name` | str | static_categorical | All sources | 78 unique areas |
| `property_type` | str | static_categorical | Transactions | Unit, Villa |
| `bedroom` | str | static_categorical | Transactions/Rents | Studio, 1BR-6BR+, Penthouse, Room |
| `reg_type` | str | static_categorical | Transactions | OffPlan, Ready |
| `developer_name` | str | static_categorical | Projects | 96 unique developers |

**Time-Varying Known (calendar + supply schedule - known in advance):**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `month` | int | time_varying_known_categorical | Computed | 1-12 |
| `quarter` | int | time_varying_known_categorical | Computed | 1-4 |
| `month_sin` | float | time_varying_known_real | Computed | sin(2π × month/12) |
| `month_cos` | float | time_varying_known_real | Computed | cos(2π × month/12) |
| `units_completing` | int | **time_varying_known_real** | Off-plan transactions | Bedroom-level supply (see below) |

**⭐ Supply Schedule (units_completing) - TIME-VARYING KNOWN:**

This is bedroom-level future supply computed from off-plan sales:
- For each area × bedroom × month: count of units scheduled to complete
- Known in advance because we have the off-plan contracts + project end dates
- TFT can "see" future supply when making predictions

| Coverage | Value |
|----------|-------|
| Off-plan transactions with handover dates | 449,920 |
| Area-bedroom-months with data | 4,332 |
| Supply schedule range | 2006-04 to 2031-12 |

**Example**: In 2025, Business Bay 1BR has 8,185 units completing. TFT knows this and factors it into price/rent predictions.

**Time-Varying Unknown - Group-Specific:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `median_price` | float | **TARGET** | Transactions | median(price_sqft) |
| `transaction_count` | int | time_varying_unknown_real | Transactions | count per group |
| `median_rent` | float | time_varying_unknown_real | Rent Contracts | median(annual_amount) |
| `rent_count` | int | time_varying_unknown_real | Rent Contracts | count |
| `median_rent_sqft` | float | time_varying_unknown_real | Rent Contracts | median(rent/area) |

**⭐ Time-Varying Unknown - Project Phase (Off-Plan Appreciation Tracking):**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `months_since_launch` | float | time_varying_unknown_real | Projects | Months since project started (97% coverage for OffPlan) |
| `months_to_handover` | float | time_varying_unknown_real | Projects | Months until expected completion (94% coverage for OffPlan) |
| `project_percent_complete` | float | time_varying_unknown_real | Projects | Build phase 0-100% (97% coverage for OffPlan) |
| `project_duration_months` | float | time_varying_unknown_real | Projects | Total project timeline (94% coverage for OffPlan) |
| `phase_ratio` | float | time_varying_unknown_real | Computed | Position in timeline 0.0→1.0 (94% coverage for OffPlan) |

**Why Phase Columns Matter:**
TFT can learn appreciation curves like:
- Early phase (0-25%): +8-12% (early buyer premium)
- Mid phase (25-50%): +5-8% (steady growth)
- Late phase (50-75%): +3-5% (risk reduces)
- Near handover (75-100%): +2-3% (approaching ready price)

**Time-Varying Unknown - Supply:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `supply_units` | int | time_varying_unknown_real | Projects | Active pipeline units |
| `supply_buildings` | int | time_varying_unknown_real | Projects | Active pipeline buildings |
| `supply_villas` | int | time_varying_unknown_real | Projects | Active pipeline villas |
| `active_projects` | int | time_varying_unknown_real | Projects | Count of active projects |

**Time-Varying Unknown - Registrations:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `units_registered` | int | time_varying_unknown_real | Units | Monthly registrations |
| `buildings_registered` | int | time_varying_unknown_real | Buildings | Monthly registrations |
| `avg_building_floors` | float | time_varying_unknown_real | Buildings | Average floors |
| `avg_building_flats` | float | time_varying_unknown_real | Buildings | Average flats |

**Time-Varying Unknown - Developer Stats (from Projects):**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `dev_total_projects` | int | time_varying_unknown_real | Projects | Cumulative projects |
| `dev_completed_projects` | int | time_varying_unknown_real | Projects | Cumulative completed |
| `dev_total_units` | int | time_varying_unknown_real | Projects | Cumulative units |
| `dev_avg_completion` | float | time_varying_unknown_real | Projects | Avg % complete |

**⭐ HIERARCHICAL CONTEXT - Developer Overall:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `dev_overall_median_price` | float | time_varying_unknown_real | Transactions | Developer's median across ALL projects |
| `dev_overall_transactions` | int | time_varying_unknown_real | Transactions | Developer's total transactions |

**⭐ HIERARCHICAL CONTEXT - Market Overall:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `market_median_price` | float | time_varying_unknown_real | Transactions | Entire market median |
| `market_transactions` | int | time_varying_unknown_real | Transactions | Entire market transaction count |

**Time-Varying Unknown - Valuations:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `govt_valuation_median` | float | time_varying_unknown_real | Valuation | median(value_sqft) |
| `valuation_count` | int | time_varying_unknown_real | Valuation | count |

**Time-Varying Unknown - EIBOR (Raw Rates Only):**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `eibor_overnight` | float | time_varying_unknown_real | EIBOR | Raw overnight rate |
| `eibor_1w` | float | time_varying_unknown_real | EIBOR | Raw 1-week rate |
| `eibor_1m` | float | time_varying_unknown_real | EIBOR | Raw 1-month rate |
| `eibor_3m` | float | time_varying_unknown_real | EIBOR | Raw 3-month rate |
| `eibor_6m` | float | time_varying_unknown_real | EIBOR | Raw 6-month rate |
| `eibor_12m` | float | time_varying_unknown_real | EIBOR | Raw 12-month rate |

**Time-Varying Unknown - Tourism:**

| Column | Type | TFT Role | Source | Notes |
|--------|------|----------|--------|-------|
| `visitors_total` | float | time_varying_unknown_real | Tourism | Visitors (thousands) |
| `hotel_rooms` | int | time_varying_unknown_real | Tourism | Hotel rooms |
| `hotel_apartments` | int | time_varying_unknown_real | Tourism | Hotel apartments |

### 3.3 What We Do NOT Include

| Excluded | Reason |
|----------|--------|
| `yield_curve_slope` | COMPUTED (12m - 3m) - TFT learns this |
| `new_lease_ratio` | COMPUTED (new / total) - TFT learns this |
| `rent_per_sqft` | COMPUTED (rent / area) - TFT learns this |
| `price_momentum` | COMPUTED - TFT learns this |
| `volatility` | COMPUTED - TFT learns this |
| `lag_1m`, `lag_3m` | COMPUTED - TFT handles lags internally |

**TFT learns all patterns from raw data. We don't pre-compute anything.**

---

## 4. Data Aggregation

### 4.1 Transactions → Monthly (Includes Developer + Reg Type)

```python
# Group by month, area, property type, bedroom, reg_type, AND developer
transactions.groupby([
    'year_month', 'area_name_en', 'property_type_en', 
    'rooms_en', 'reg_type', 'developer_name'
]).agg(
    median_price=('price_sqft', 'median'),
    transaction_count=('transaction_id', 'count')
)
```

### 4.2 Developer-Overall Context (Hierarchical)

```python
# Aggregate across ALL areas/bedrooms for each developer per month
transactions.groupby(['year_month', 'developer_name']).agg(
    dev_overall_median_price=('median_price', 'median'),
    dev_overall_transactions=('transaction_count', 'sum')
)
```

### 4.3 Market-Overall Context (Hierarchical)

```python
# Aggregate across ALL groups for each month
transactions.groupby(['year_month']).agg(
    market_median_price=('median_price', 'median'),
    market_transactions=('transaction_count', 'sum')
)
```

### 4.2 Rent Contracts → Monthly

```python
# Group by month, area, bedroom
rents.groupby(['year_month', 'area_name_en', 'bedrooms']).agg(
    median_rent=('annual_amount', 'median'),
    rent_count=('contract_id', 'count')
)
```

### 4.3 EIBOR → Monthly (Already Done)

```python
# Just extract raw rate, drop computed columns
eibor_clean = eibor[['year_month', '3_month']].rename(columns={'3_month': 'eibor_3m'})
```

### 4.4 Supply → Monthly by Area

```python
# Count units in active projects per area/month
for each year_month, area:
    supply_units = sum(no_of_units) WHERE project_status != 'FINISHED' 
                   AND project_end_date > year_month
```

---

## 5. Output Format

### 5.1 Single Training File

**File**: `Data/tft/tft_training_data.csv`

Key columns (showing hierarchical context):

```csv
time_idx,year_month,group_id,area_name,property_type,bedroom,reg_type,developer_name,median_price,...,dev_overall_median_price,dev_overall_transactions,market_median_price,market_transactions,...
0,2015-01,Business_Bay_Unit_2BR_OffPlan_Emaar,Business Bay,Unit,2BR,OffPlan,Emaar,1450,...,1380,2500,1200,45000,...
1,2015-02,Business_Bay_Unit_2BR_OffPlan_Emaar,Business Bay,Unit,2BR,OffPlan,Emaar,1465,...,1395,2650,1210,46500,...
```

**What each row represents**:
- `median_price`: Price for this specific group (Business Bay, 2BR, OffPlan, Emaar)
- `dev_overall_median_price`: Emaar's overall median across ALL their projects in this month
- `market_median_price`: Entire Dubai market median in this month

**Why this matters**:
- TFT can compare this developer's specific project to their overall portfolio
- TFT can compare this developer to the market
- Even new area launches have rich developer history context

### 5.2 Actual Output (Built 2025-12-10)

| Metric | Value |
|--------|-------|
| **Total Rows** | 77,197 |
| **Unique Groups** | 1,787 |
| **Unique Developers** | 96 |
| **Unique Areas** | 78 |
| **Columns** | 50 |
| **Date Range** | 2003-06 to 2025-12 (240 months) |
| **OffPlan Groups** | 928 |
| **Ready Groups** | 859 |
| **File Size** | 36.2 MB |

**Multi-Target Outputs:**
- `median_price` - Capital appreciation prediction
- `median_rent` - Rental yield prediction

**group_id format**: `{area}_{property_type}_{bedroom}_{reg_type}_{developer}`

Example: `Business_Bay_Unit_2BR_OffPlan_اعمار_العقارية_ش__م__ع`

**Hierarchical Context Coverage**:
- `dev_overall_median_price`: 100% coverage
- `dev_overall_transactions`: 100% coverage
- `market_median_price`: 100% coverage
- `market_transactions`: 100% coverage

**Project Phase Coverage (OffPlan)**:
- `months_since_launch`: 97.3% coverage
- `months_to_handover`: 94.0% coverage
- `project_percent_complete`: 97.3% coverage
- `project_duration_months`: 94.0% coverage
- `phase_ratio`: 94.0% coverage

---

## 6. Directory Structure

```
Data/
├── cleaned/                          # ✅ KEEP (source data)
│   ├── Transactions_Cleaned.csv
│   ├── Rent_Contracts_Cleaned.csv
│   ├── Projects_Cleaned.csv
│   ├── eibor_monthly.csv
│   └── ...
│
├── tft/                              # 🆕 CREATE
│   ├── tft_training_data.csv         # Main training file
│   └── data_config.json              # TFT configuration
│
└── training/                         # ❌ DELETE (old XGBoost format)
```

---

## 7. Training Workflow

### 7.1 Colab → Mac Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STEP 1: BUILD DATA (Mac)                      │
│                                                                      │
│  $ python scripts/build_tft_data.py                                 │
│  Output: Data/tft/tft_training_data.csv (~50-100MB)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STEP 2: TRAIN (Colab)                         │
│                                                                      │
│  1. Upload tft_training_data.csv to Google Drive                    │
│  2. Run training notebook (1-2 hours on GPU)                        │
│  3. Download checkpoint: tft_model.ckpt (~200-500MB)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STEP 3: INFERENCE (Mac)                       │
│                                                                      │
│  • Load checkpoint once at startup                                  │
│  • Run predictions on CPU (1-5 seconds per query)                   │
│  • LLM interprets outputs                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Cost

| Item | Cost |
|------|------|
| Google Colab Pro | $10/month |
| Mac inference | Free |
| Claude API | ~$10-50/month (usage-based) |
| **Total** | **~$20-60/month** |

---

## 8. Script to Create

### `scripts/build_tft_data.py`

```python
"""
Build TFT-compatible training data from cleaned sources.

Usage:
    python scripts/build_tft_data.py

Output:
    Data/tft/tft_training_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 1. Load cleaned data
    transactions = pd.read_csv('Data/cleaned/Transactions_Cleaned.csv')
    rents = pd.read_csv('Data/cleaned/Rent_Contracts_Cleaned.csv')
    eibor = pd.read_csv('Data/cleaned/eibor_monthly.csv')
    projects = pd.read_csv('Data/cleaned/Projects_Cleaned.csv')
    
    # 2. Aggregate transactions
    # 3. Aggregate rents
    # 4. Clean EIBOR (raw rates only)
    # 5. Compute supply
    # 6. Merge all
    # 7. Create TFT columns (time_idx, group_id, cyclical encoding)
    # 8. Save
    
    output_path = Path('Data/tft/tft_training_data.csv')
    output_path.parent.mkdir(exist_ok=True)
    merged.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
```

---

## 9. Execution Checklist

| Step | Task | Time |
|------|------|------|
| 1 | Create `Data/tft/` directory | 1 min |
| 2 | Create `scripts/build_tft_data.py` | 1 hour |
| 3 | Run data build | 15 min |
| 4 | Validate output | 10 min |
| 5 | Upload to Colab | 5 min |
| 6 | Train TFT | 1-2 hours |
| 7 | Download checkpoint | 5 min |
| **Total** | | **~3-4 hours** |

---

## 10. Success Criteria

| Metric | Target |
|--------|--------|
| All cleaned data included | 100% |
| No computed features | Zero (raw aggregations only) |
| TFT columns present | time_idx, group_id, cyclical encoding |
| Multi-target prediction | `median_price` + `median_rent` |
| Supply schedule included | `units_completing` (time-varying known) |
| Encoder length | 12-96 months (flexible) |
| Prediction length | 1-60 months (up to 5 years) |
| File loads in PyTorch Forecasting | ✅ No errors |
| Model trains on Colab | ✅ Converges |

## 11. What the Model Predicts

| Output | For | Agent Question |
|--------|-----|----------------|
| `median_price` forecast | Capital appreciation | "What will this be worth at handover?" |
| `median_rent` forecast | Rental yield | "What will this rent for?" |
| Confidence intervals | Risk assessment | "How confident is this prediction?" |
| Attention weights | Explainability | "What factors drove this prediction?" |

**Example Agent Interaction:**
```
Agent: "Binghatti JVC 2BR 2.2M off-plan, 3 years to handover"

TFT Predicts:
├── Price at handover: 1,850 AED/sqft (±150)
├── Rent at handover: 95,000 AED/year (±8,000)
├── Key factors: EIBOR trend, JVC supply (8,185 1BR completing), Binghatti track record

LLM Interprets:
├── Appreciation: ~26% over 3 years
├── Yield at handover: ~3.4%
├── Risk: High supply in area - 8,185 units completing
└── Developer: Binghatti historically +18% appreciation
```

---

## References

- [PyTorch Forecasting - TFT](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)
- [Google Colab Pro](https://colab.research.google.com/signup)
- [Temporal Fusion Transformers Paper](https://arxiv.org/abs/1912.09363)
