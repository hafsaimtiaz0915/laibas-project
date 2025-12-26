# Data Documentation Index

> **Last Updated**: 2025-12-10  
> **Purpose**: Index of all data-related documentation for the TFT-based real estate AI platform.

---

## Architecture: TFT + LLM

This project uses a **Temporal Fusion Transformer (TFT)** trained on aggregated real estate data, with an **LLM layer** for natural language interpretation.

```
Agent Query → TFT Model (predictions + attention) → LLM (interpretation) → Response
```

**Key Principle**: Raw data only. TFT learns patterns - we don't compute features.

---

## Document Reading Order

| Order | Document | Purpose |
|-------|----------|---------|
| **1** | [TIME_SERIES_DATA_PLAN.md](./TIME_SERIES_DATA_PLAN.md) | 📋 TFT data format specification |
| **2** | [DATA_PIPELINE_ERRORS_AND_LESSONS.md](./DATA_PIPELINE_ERRORS_AND_LESSONS.md) | 📚 Historical errors and lessons |
| **3** | [DATA_QUALITY_REPORT.md](./DATA_QUALITY_REPORT.md) | 📊 Data profiling results |
| **4** | [DATA_PROFILING_PLAN.md](./DATA_PROFILING_PLAN.md) | 📚 Original profiling methodology |

---

## Current Data Status

### Cleaned Data (`Data/cleaned/`)

| File | Records | Status |
|------|---------|--------|
| `Transactions_Cleaned.csv` | 1,606,520 | ✅ Ready |
| `Rent_Contracts_Cleaned.csv` | 5,743,849 | ✅ Ready |
| `Projects_Cleaned.csv` | 3,039 | ✅ Ready |
| `Units_Cleaned.csv` | 2,335,623 | ✅ Ready |
| `Buildings_Cleaned.csv` | 239,277 | ✅ Ready |
| `Valuation_Cleaned.csv` | 87,093 | ✅ Ready |
| `eibor_monthly.csv` | 182 months | ✅ Ready |
| `tourism_visitors.csv` | 70 records | ✅ Ready |
| `tourism_inventory.csv` | 10 records | ✅ Ready |

### TFT Training Data (`Data/tft/`)

| File | Status |
|------|--------|
| `tft_training_data.csv` | 🔄 To be created |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/clean_all_data.py` | Clean raw DLD data |
| `scripts/clean_rent_contracts.py` | Clean rent contract data |
| `scripts/clean_tourism_data.py` | Process tourism data |
| `scripts/build_tft_data.py` | Build TFT training data (to be created) |

---

## Model Documentation

See `Docs/models/` for:
- `00_OVERVIEW_ARCHITECTURE.md` - System architecture
- `02_TIME_SERIES_FORECASTING.md` - TFT model specification
- `04_ROI_CALCULATOR_OFFPLAN.md` - ROI calculation formulas
- `06_DEPLOYMENT_ARCHITECTURE.md` - Deployment details

---

## Data Flow

```
Raw DLD Data
    │
    ▼
Cleaning Scripts (scripts/clean_*.py)
    │
    ▼
Data/cleaned/*.csv (individual cleaned files)
    │
    ▼
scripts/build_tft_data.py
    │
    ▼
Data/tft/tft_training_data.csv (TFT format)
    │
    ▼
Train on Google Colab
    │
    ▼
TFT Model Checkpoint
```

---

## Next Steps

1. [ ] Create `scripts/build_tft_data.py`
2. [ ] Create `Data/tft/` directory
3. [ ] Build TFT training data
4. [ ] Train model on Colab
5. [ ] Integrate with LLM

---

## For AI Assistants

If you are an AI assistant working on this project:

1. **Use TFT, not XGBoost** - All old XGBoost docs have been deleted
2. **No feature engineering** - TFT learns from raw aggregated data
3. **Check TIME_SERIES_DATA_PLAN.md** for the data format
4. **Training data format**: Single CSV with time_idx, group_id, raw values
