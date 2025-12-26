# Project State — Data Scientist Onboarding

**Last Updated:** 2025-12-21  
**Purpose:** Single source of truth for understanding the current state of the ML pipeline  
**IMPORTANT:** This entire pipeline was built with AI assistance and requires human expert verification

---

## 🚨 START HERE

**Primary briefing document:** `Docs/Data_docs/DATA_SCIENCE_AUDIT_BRIEF.md`

This document explains what files to use. The briefing document explains the problems.

---

## 🎯 CANONICAL FILES (Use These Only)

### Training Data

| File | Description | Rows | Status |
|------|-------------|------|--------|
| `Data/tft/latest/tft_training_data_v2_imputed.csv` | Current V2 training data with rent imputation | 272,257 | ✅ CANONICAL |
| `Data/tft/tft_training_data_v2.csv` | Same data (symlinked from latest/) | 272,257 | ✅ CANONICAL |
| `backend/models/Console_model/console_tft_training_data_v2 (2).csv` | Copy used for Console training | 272,257 | ✅ BACKUP |
| `backend/models/colab_model/colab_tft_training_data_v2 (3).csv` | Copy used for Colab training | 272,257 | ✅ BACKUP |

**Note:** Console and Colab training data are **identical** except Console has 2 extra columns (`rent_imputed`, `rent_imputation_source`) for tracking rent imputation.

### Trained Models

| Model | File | Training Steps | Features | Status |
|-------|------|----------------|----------|--------|
| **Console V2** | `backend/models/Console_model/output_tft_best_v2.ckpt` | 250,074 | 11 time-varying | ✅ RECOMMENDED |
| **Colab V3** | `backend/models/colab_model/tft_final_v3.ckpt` | 60,000 | 34 time-varying | ⚠️ More features, less training |
| Colab V3 Patched | `backend/models/colab_model/tft_final_v3__patched.ckpt` | 60,000 | 34 time-varying | ⚠️ Patched for CPU loading |

**Recommendation:** Use Console V2 for now — simpler (less overfitting risk) and trained 4x longer.

### Key Lookup Files

| File | Description | Must Understand |
|------|-------------|-----------------|
| `Data/lookups/developer_brand_consolidation.json` | SPV → Parent brand mapping | ✅ YES |
| `Data/lookups/umbrella_map.json` | Developer ID → Corporate umbrella | ✅ YES (has errors) |
| `Data/lookups/top50_developers_2025.json` | Major developers list | ✅ YES |
| `Data/lookups/area_mapping.json` | area_id → area_name | ✅ YES |
| `Data/lookups/public_brands.json` | Canonical brand names | Helpful |
| `Data/lookups/blocked_brand_labels.json` | Labels to exclude | Helpful |

### Essential Documentation

| File | Description |
|------|-------------|
| `Docs/Data_docs/DATA_SCIENCE_AUDIT_BRIEF.md` | **START HERE** — Full problem list |
| `Docs/ARCHITECTURE_FILE_REFERENCE.md` | Complete file reference (what each file does) |
| `Docs/Data_audits/TFT_V3_MODEL_AUDIT_REPORT.md` | Model issues found during testing |
| `Docs/PROJECT_STATE_FOR_DATA_SCIENTIST.md` | This file |

---

## 🗂️ DATA PIPELINE OVERVIEW

### Raw Data Sources

```
Data/raw_data/
├── Transactions.csv        (~500K sales transactions)
├── Projects.csv            (~3K development projects)  
├── Buildings.csv           (~240K property records)
├── Rent_Contracts.csv      (lease agreements)
├── Units.csv               (individual units with FLOOR data)
├── Valuation.csv           (valuations)
└── new_raw_data/
    ├── Developers.csv      (~2.1K registered developers)
    ├── Lkp_Areas.csv       (~300 geographic zones)
    ├── Residential_Sale_Index.csv (price indices)
    └── Community.kml       (area polygons)
```

### Processing Pipeline

```
Raw Data
    ↓
scripts/clean_all_data.py
    ↓
Data/cleaned/*.csv
    ↓
scripts/build_tft_data_v2.py  ← CORE AGGREGATION SCRIPT
    ↓
Data/tft/runs/<BUILD_ID>/tft_training_data_v2.csv
    ↓
Data/tft/latest/  (symlinks to current build)
```

### Current Build

```
BUILD_ID: 20251218T163716Z
Location: Data/tft/runs/20251218T163716Z/
```

---

## 🔑 DEVELOPER MAPPING (Critical to Understand)

### How It Works

1. **Raw Transaction** has `project_number`
2. **Join:** `project_number` → `Projects.csv` → `developer_id`
3. **Resolve:** `developer_id` → `umbrella_map.json` → `developer_umbrella`
4. **Brand:** `developer_id` → `developer_brand_consolidation.json` → `developer_brand_label`

### Known Problems

| Issue | Description | Impact |
|-------|-------------|--------|
| **Duplicate brands** | "Emaar" vs "EMAAR PROPERTIES" vs "EMAAR DEVELOPMENT" | 3 embeddings for 1 developer |
| **Unmapped IDs** | 363 labels like `DEVELOPER_ID_14868751` | 15% of transactions unreadable |
| **Wrong umbrellas** | Some mappings are incorrect | Noise in umbrella feature |
| **Case sensitivity** | "Nakheel" vs "NAKHEEL" | Signal fragmentation |

### Top 50 Developers

See: `Data/lookups/top50_developers_2025.json`

Key players: Emaar, DAMAC, Nakheel, Sobha, Meraas, Dubai Properties, Binghatti, Danube, Azizi, Nshama

---

## 🗺️ GEOGRAPHY MAPPING

### How It Works

- `area_id` is the primary key (official land registry zones)
- ~300 unique areas, ~77 have significant transaction volume
- Mapping: `Data/lookups/area_mapping.json`
- Centroids: Extracted from `Data/raw_data/new_raw_data/Community.kml`

### Key Areas

| area_id | Name | Transaction Volume |
|---------|------|-------------------|
| 330 | Marsa Dubai (Marina) | Very High |
| 338 | Burj Khalifa (Downtown) | Very High |
| 277 | Jumeirah Village Circle | Very High |
| 213 | Business Bay | Very High |
| 523 | Dubai Hills | High |

---

## 📊 GROUP_ID DEFINITION (Critical Design Decision)

### Current Formula

```
group_id = area_id + "_" + property_type + "_" + bedroom + "_" + reg_type
```

**Example:** `330_Unit_2BR_Ready` = Marina, Apartment, 2 Bedroom, Resale

### The Problem

**Developer is NOT in group_id.**

This means:
- Each time series mixes transactions from MULTIPLE developers
- Model cannot learn "Emaar apartments appreciate faster than DAMAC"
- Developer categorical is noise within each group

### Trade-off

| Option | Groups | Data per Group |
|--------|--------|----------------|
| Current (no developer) | ~800 | Dense |
| With developer | ~5,000+ | Sparse (cold start issues) |

---

## 🔧 MODEL CONFIGURATION

### Common Settings (Both Models)

| Parameter | Value |
|-----------|-------|
| Encoder length | 96 months (8 years) |
| Prediction horizon | 12 months |
| Targets | `median_price`, `median_rent` |
| Output | 7 quantiles each |
| Static categoricals | `area_name`, `property_type`, `bedroom`, `reg_type`, `developer_brand_label`, `developer_umbrella` |
| Time-varying known | `time_idx`, `month_sin`, `month_cos`, `units_completing` |

### Console Model (Recommended)

**11 time-varying unknown features:**
- `transaction_count`, `rent_count`, `median_rent_sqft`
- `months_to_handover_signed`, `months_since_handover`, `handover_window_6m`
- `supply_units`, `active_projects`, `sale_index`
- `median_price`, `median_rent`

### Colab V3 Model

**34 time-varying unknown features:**
- All Console features PLUS geographic, index, and developer ID features
- Higher risk of overfitting

---

## ❌ DEPRECATED FILES (Ignore These)

### Archived Locations

| Archive | Contents |
|---------|----------|
| `Data/tft/_archive_v1/` | V1 training data and stats |
| `Data/tft/_archive_intermediate/` | Debug CSVs, suggestions, tripwires |
| `Data/tft/runs/_archive/` | Old build runs |
| `backend/models/_deprecated/` | Old model checkpoints |
| `Docs/Data_docs/_deprecated/` | Outdated training docs |

### Specifically Ignore

- `tft_training_data.csv` (V1)
- `build_stats.json` (V1)
- `tft_final.ckpt` (V1 model)
- `training_opus_colab.md` (V1 training instructions)
- Any file with `suggested_`, `learned_`, `spv_` prefix (debug artifacts)

---

## 🧪 TRAINING SCRIPTS

### Current (V2)

| File | Description |
|------|-------------|
| `Docs/Data_docs/training_colab_v2.py` | Current training script |
| `Docs/Data_docs/training_colab_v2.ipynb` | Colab notebook version |
| `Docs/Data_docs/training_colab_v2.md` | Documentation |

### Data Build

| File | Description |
|------|-------------|
| `scripts/build_tft_data_v2.py` | Builds training data from raw |
| `scripts/brand_resolver.py` | Developer entity resolution |
| `scripts/clean_all_data.py` | Data cleaning |

---

## 🚨 CRITICAL ISSUES (Priority Order)

### Must Fix Before Production

1. **Developer not in group_id** — Model can't learn developer-specific trends
2. **Duplicate brand names** — "Emaar" split across 3+ labels
3. **Incorrect umbrella mapping** — Wrong corporate groupings found
4. **Training-inference mismatch** — `units_completing` generated differently at inference
5. **Unmapped developer IDs** — 15% of transactions have meaningless labels

### Should Fix

6. **Cold start** — 30% of groups have <24 months history
7. **No floor data** — Floor height not included (data exists but not joined)

---

## 📁 CLEAN FILE STRUCTURE (After Cleanup)

```
Data/
├── raw_data/              # Original data (don't modify)
├── cleaned/               # Cleaned data
├── lookups/               # Mapping files ← UNDERSTAND THESE
├── tft/
│   ├── latest/            # ← CURRENT TRAINING DATA
│   ├── runs/
│   │   └── 20251218T163716Z/  # ← CURRENT BUILD
│   ├── tft_training_data_v2.csv
│   ├── build_stats_v2.json
│   ├── baseline_snapshot_v2.json
│   ├── delta_report_v2.json
│   └── umbrella_seed_audit_v2.json
│   
backend/models/
├── Console_model/
│   ├── output_tft_best_v2.ckpt     # ← RECOMMENDED MODEL
│   └── console_tft_training_data_v2 (2).csv
├── colab_model/
│   ├── tft_final_v3.ckpt           # ← EXPERIMENTAL MODEL
│   ├── tft_final_v3__patched.ckpt
│   └── colab_tft_training_data_v2 (3).csv
└── group_ids.json

Docs/
├── Data_docs/
│   ├── DATA_SCIENCE_AUDIT_BRIEF.md  # ← PRIMARY BRIEFING
│   ├── training_colab_v2.py
│   └── Colab_Training_v2/
└── Data_audits/
    └── TFT_V3_MODEL_AUDIT_REPORT.md
```

---

## ✅ VERIFICATION CHECKLIST

Before trusting this data, verify:

- [ ] Transaction filtering is correct (residential sales only)
- [ ] Developer joins are accurate (spot-check major developers)
- [ ] Aggregation logic produces sensible medians
- [ ] `units_completing` matches project schedules
- [ ] No data leakage (features don't "know" the target)
- [ ] Join Transaction → Unit for floor data is feasible

---

*Document version: 1.0*  
*Created: 2025-12-21*

