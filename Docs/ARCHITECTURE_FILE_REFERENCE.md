# Project Architecture — Complete File Reference

**Last Updated:** 2025-12-21  
**Purpose:** High-level overview of every active file in the project  
**For:** Data scientists and developers onboarding to the codebase

---

## Quick Navigation

| Section | Description |
|---------|-------------|
| [1. Scripts](#1-scripts) | Python scripts for data processing and analysis |
| [2. Raw Data](#2-raw-data) | Original data from land registry |
| [3. Cleaned Data](#3-cleaned-data) | Processed data ready for use |
| [4. Lookup Tables](#4-lookup-tables) | Mapping and reference files |
| [5. TFT Training Data](#5-tft-training-data) | Model training datasets |
| [6. Trained Models](#6-trained-models) | Model checkpoints |
| [7. Documentation](#7-documentation) | Markdown docs and guides |
| [8. Data Profiles](#8-data-profiles) | Data quality reports |

---

## 1. Scripts

Location: `scripts/`

### Core Data Pipeline

| File | Description | When to Use |
|------|-------------|-------------|
| `build_tft_data_v2.py` | **MAIN SCRIPT** — Builds training data from raw → aggregated | Run when rebuilding training data |
| `build_tft_data.py` | V1 version (deprecated but kept for reference) | Don't use |
| `clean_all_data.py` | Cleans all raw CSVs (transactions, projects, buildings) | Run first before build_tft_data |
| `brand_resolver.py` | Developer entity resolution (SPV → brand mapping) | Called by build_tft_data_v2.py |
| `generate_lookup_tables.py` | Creates area/developer lookup JSONs | Run to regenerate lookups |

### Data Quality & Analysis

| File | Description | When to Use |
|------|-------------|-------------|
| `audit_developer_segmentation.py` | Checks developer brand coverage in training data | Run to audit developer mapping |
| `pre_retrain_gate.py` | Pre-training validation gate (checks data quality) | Run before training |
| `validate_area_mapping.py` | Validates area name mappings | Run to check geography |
| `generate_area_mapping_coverage.py` | Reports area mapping coverage | Run to check area resolution |
| `diagnose_rent_missingness_v2.py` | Analyzes missing rent data patterns | Run to understand rent gaps |
| `generate_quality_report.py` | Generates data quality report | Run for QA |

### Profiling & Analysis

| File | Description | When to Use |
|------|-------------|-------------|
| `profile_large_csv.py` | Profiles large CSVs (Transactions, Buildings) | Run for data exploration |
| `profile_small_csv.py` | Profiles smaller CSVs | Run for data exploration |
| `run_all_profiling.py` | Runs all profiling scripts | Run for full data audit |
| `analyze_entity_resolution.py` | Analyzes entity (developer/area) resolution | Run to understand mappings |
| `apply_entity_resolution.py` | Applies entity resolution fixes | Run to fix entity issues |

### Model Testing

| File | Description | When to Use |
|------|-------------|-------------|
| `test_tft_v2.py` | Tests TFT model locally | Run to validate model predictions |
| `test_tft_v2_developers.py` | Tests developer-specific predictions | Run to audit developer handling |
| `smoke_test_tft_checkpoint.py` | Quick model load test | Run to verify checkpoint works |

### Utilities

| File | Description | When to Use |
|------|-------------|-------------|
| `bedroom_mapping.py` | Bedroom code standardization | Called by build_tft_data |
| `clean_rent_contracts.py` | Cleans rent contract data | Run for rent data prep |
| `clean_tourism_data.py` | Cleans tourism data | Optional - tourism features |
| `process_eibor_data.py` | Processes EIBOR interest rate data | Optional - macro features |
| `profile_tourism_data.py` | Profiles tourism data | Optional - tourism analysis |

---

## 2. Raw Data

Location: `Data/raw_data/`

### Primary Files

| File | Rows | Description | Key Columns |
|------|------|-------------|-------------|
| `Transactions.csv` | ~500K | Property sale transactions | `transaction_id`, `area_id`, `actual_worth`, `meter_sale_price`, `project_number` |
| `Projects.csv` | ~3K | Development projects | `project_id`, `developer_id`, `completion_date`, `no_of_units` |
| `Buildings.csv` | ~240K | Building/property records | `property_id`, `area_id`, `floors`, `project_id` |
| `Units.csv` | ~2.3M | Individual units | `property_id`, `floor`, `rooms_en`, `actual_area` |
| `Rent_Contracts.csv` | ~100K | Lease agreements | `contract_id`, `annual_amount`, `area_id` |
| `Valuation.csv` | ~87K | Property valuations | Reference data |

### New Raw Data (Updated Versions)

Location: `Data/raw_data/new_raw_data/`

| File | Description |
|------|-------------|
| `Developers.csv` | Developer registry (~2.1K developers) |
| `Lkp_Areas.csv` | Area lookup table (~300 areas) |
| `Lkp_Transaction_Groups.csv` | Transaction type lookup |
| `Residential_Sale_Index.csv` | Monthly price index (~160 months) |
| `Community.kml` | Area polygon boundaries (KML format) |
| `Map_Requests.csv` | Map request data (large, rarely used) |

---

## 3. Cleaned Data

Location: `Data/cleaned/`

| File | Description | Created By |
|------|-------------|------------|
| `Transactions_Cleaned.csv` | Cleaned transactions (residential sales only) | `clean_all_data.py` |
| `Projects_Cleaned.csv` | Cleaned projects | `clean_all_data.py` |
| `Buildings_Cleaned.csv` | Cleaned buildings | `clean_all_data.py` |
| `Units_Cleaned.csv` | Cleaned units | `clean_all_data.py` |
| `Rent_Contracts_Cleaned.csv` | Cleaned rent contracts | `clean_rent_contracts.py` |
| `Valuation_Cleaned.csv` | Cleaned valuations | `clean_all_data.py` |
| `cleaning_stats.json` | Cleaning statistics | `clean_all_data.py` |
| `eibor_daily.csv` | Daily EIBOR rates | `process_eibor_data.py` |
| `eibor_monthly.csv` | Monthly EIBOR rates | `process_eibor_data.py` |

---

## 4. Lookup Tables

Location: `Data/lookups/`

### Developer Mapping (Critical)

| File | Description | Must Understand |
|------|-------------|-----------------|
| `developer_brand_consolidation.json` | SPV → Parent brand mapping (Emaar SPVs → "Emaar") | ✅ YES |
| `umbrella_map.json` | Developer ID → Corporate umbrella | ✅ YES (has known errors) |
| `top50_developers_2025.json` | Major developers list with IDs | ✅ YES |
| `developer_mapping.json` | Developer ID → name mapping | Helpful |
| `developer_reference.csv` | Developer reference table | Helpful |
| `developer_stats.csv` | Developer transaction statistics | Helpful |
| `public_brands.json` | Canonical brand names | Helpful |
| `public_brand_aliases.json` | Brand name aliases | Helpful |
| `blocked_brand_labels.json` | Labels to exclude from training | Helpful |
| `building_developers.json` | Building → developer mapping | Reference |

### Developer Overrides

| File | Description |
|------|-------------|
| `brand_overrides_developer_id.csv` | Manual developer ID → brand corrections |
| `brand_overrides_master_project.csv` | Master project → brand corrections |
| `brand_overrides_project_number.csv` | Project number → brand corrections |
| `entity_owner_overrides_developer_id.csv` | Entity ownership corrections |

### Area Mapping

| File | Description | Must Understand |
|------|-------------|-----------------|
| `area_mapping.json` | area_id → area_name mapping | ✅ YES |
| `area_reference.csv` | Full area reference table | Helpful |
| `area_stats.csv` | Transaction counts per area | Helpful |
| `area_mapping_coverage.csv` | Area name resolution coverage | Reference |

### Other Lookups

| File | Description |
|------|-------------|
| `rent_benchmarks.csv` | Rent benchmarks by segment |

---

## 5. TFT Training Data

### Current Build

Location: `Data/tft/latest/` (symlinks to current build)

| File | Description | Status |
|------|-------------|--------|
| `tft_training_data_v2.csv` | **MAIN TRAINING DATA** (272K rows) | ✅ USE THIS |
| `tft_training_data_v2_imputed.csv` | Training data with rent imputation | ✅ USE THIS |
| `tft_training_data_v2__rent_observed_only.csv` | Only rows with actual rent data | Reference |
| `build_stats_v2.json` | Build statistics | Reference |
| `latest_build_id.txt` | Current build ID | Reference |

### Build Artifacts

Location: `Data/tft/latest/` and `Data/tft/runs/20251218T163716Z/`

| File | Description |
|------|-------------|
| `baseline_snapshot_v2.json` | Baseline metrics snapshot |
| `delta_report_v2.json` | Changes from baseline |
| `umbrella_seed_audit_v2.json` | Umbrella mapping audit |
| `holding_policy_audit_v2.json` | Holding company policy audit |
| `owner_assertions_report.json` | Owner resolution report |
| `run_manifest.json` | Build manifest (in runs folder) |

### Quality Check Artifacts

| File | Description |
|------|-------------|
| `spv_candidates_ranked.csv` | SPV candidates for brand consolidation |
| `suspicious_label_gate_offenders.csv` | Suspicious labels flagged |
| `label_dispersion_report_v2.csv` | Label dispersion analysis |
| `high_impact_non_top50_suspicious_labels.csv` | High-impact unmapped labels |

---

## 6. Trained Models

Location: `backend/models/`

### Active Models

| File | Description | Training Steps | Recommended |
|------|-------------|----------------|-------------|
| `Console_model/output_tft_best_v2.ckpt` | Console-trained model (simpler) | 250,074 | ✅ YES |
| `colab_model/tft_final_v3.ckpt` | Colab-trained model (more features) | 60,000 | ⚠️ Experimental |
| `colab_model/tft_final_v3__patched.ckpt` | Patched for CPU loading | 60,000 | ⚠️ Experimental |

### Training Data Copies

| File | Description |
|------|-------------|
| `Console_model/console_tft_training_data.csv` | Copy used for Console training |
| `colab_model/colab_tft_training_data.csv` | Copy used for Colab training |

### Model Metadata

| File | Description |
|------|-------------|
| `group_ids.json` | Group ID list |
| `group_index.json` | Group index mapping |

---

## 7. Documentation

### Primary Docs (Start Here)

Location: `Docs/`

| File | Description | Priority |
|------|-------------|----------|
| `PROJECT_STATE_FOR_DATA_SCIENTIST.md` | **START HERE** — Current state and file locations | 🔴 FIRST |
| `Data_docs/DATA_SCIENCE_AUDIT_BRIEF.md` | Full problem list and context | 🔴 SECOND |
| `ARCHITECTURE_FILE_REFERENCE.md` | This file — complete file reference | 🔴 THIRD |

### Data Documentation

Location: `Docs/Data_docs/`

| File | Description |
|------|-------------|
| `training_colab_v2.py` | Current training script |
| `training_colab_v2.ipynb` | Colab notebook version |
| `training_colab_v2.md` | Training documentation |
| `RETRAINING_INVESTOR_LENS.md` | Investor-focused training context |
| `RENT_IMPUTATION_SOLUTION.md` | How rent imputation works |
| `DATA_QUALITY_REPORT.md` | Data quality findings |
| `DATA_PIPELINE_ERRORS_AND_LESSONS.md` | Pipeline error history |
| `phase2_add_unit_sqft_feature.md` | Feature addition guide |
| `README.md` | Data docs overview |

### Audit Reports

Location: `Docs/Data_audits/`

| File | Description |
|------|-------------|
| `TFT_V3_MODEL_AUDIT_REPORT.md` | V3 model audit findings |
| `DEVELOPER_SEGMENTATION_AUDIT.md` | Developer mapping audit |
| `DEVELOPER_SEGMENTATION_AUDIT.csv` | Developer audit data |
| `FINAL_PRE_RETRAIN_INVESTOR_AUDIT.md` | Pre-training audit |
| `PRE_RETRAIN_GATE_REPORT.md` | Pre-training gate results |

### Product Docs

| File | Description |
|------|-------------|
| `PRD.md` | Product requirements document |
| `PROPRLY_PRODUCT_OVERVIEW.md` | Product overview |
| `requirements.md` | Technical requirements |
| `tft_retraining_checklist.md` | Retraining checklist (partially outdated) |
| `Backend_hosting.md` | Backend deployment |

### Frontend Docs

Location: `Docs/frontend/`

| File | Description |
|------|-------------|
| `00_FRONTEND_ARCHITECTURE.md` | Frontend architecture |
| `01_COMPONENT_MAPPING.md` | Component mapping |
| `02_MODEL_HOSTING.md` | Model hosting |
| `03_COMPONENT_CODE.md` | Component code |
| `04_landingpage_components.md` | Landing page components |
| `Known_issues.md` | Known frontend issues |

### LLM Integration Docs

Location: `Docs/frontend/LLM/`

| File | Description |
|------|-------------|
| `00_LLM_INTEGRATION_PLAN.md` | LLM integration plan |
| `01_LLM_SERVICE_CODE.md` | LLM service code |
| `AREA_MAPPING_COVERAGE.md` | Area mapping for LLM |
| `OUTPUT_CONTRACT.md` | LLM output contract |
| `PIPELINE_AUDIT_AND_GATING.md` | Pipeline audit |

### Model Architecture Docs

Location: `Docs/models/`

| File | Description |
|------|-------------|
| `00_OVERVIEW_ARCHITECTURE.md` | Model architecture overview |
| `02_TIME_SERIES_FORECASTING.md` | TFT model documentation |
| `04_ROI_CALCULATOR_OFFPLAN.md` | ROI calculator logic |
| `06_DEPLOYMENT_ARCHITECTURE.md` | Deployment architecture |

---

## 8. Data Profiles

Location: `data_profiles/`

### Main Profiles

| File | Description |
|------|-------------|
| `Transactions_profile.json` | Transactions data profile |
| `Units_profile.json` | Units data profile |
| `Rent_Contracts_profile.json` | Rent contracts profile |
| `temporal_analysis.json` | Temporal analysis results |
| `tft_v3_audit_report.json` | V3 model audit data |

### Detailed Profiles

Location: `data_profiles/detailed_reports/`

| File | Description |
|------|-------------|
| `Buildings_profile.json` | Buildings detailed profile |
| `Projects_profile.json` | Projects detailed profile |
| `Valuation_profile.json` | Valuation detailed profile |

### Entity Resolution

Location: `data_profiles/entity_resolution/`

| File | Description |
|------|-------------|
| `entity_resolution_master.json` | Master entity resolution |
| `canonical_areas.json` | Canonical area names |
| `canonical_projects.json` | Canonical project names |
| `area_name_mapping.json` | Area name mappings |
| `master_project_mapping.json` | Project mappings |
| `conflicts.json` | Resolution conflicts |
| `similar_entities.json` | Similar entities found |

---

## 9. Policies

Location: `Data/policies/`

| File | Description |
|------|-------------|
| `label_policy_v2.md` | Developer label policies |

---

## File Count Summary

| Category | Count |
|----------|-------|
| Python scripts | 24 |
| Raw data files | 14 |
| Cleaned data files | 14 |
| Lookup tables | 19 |
| TFT build artifacts | ~30 |
| Model checkpoints | 3 |
| Documentation (MD) | ~35 |
| Data profiles | ~20 |
| **Total Active Files** | ~160 |

---

## Archived/Deprecated (Do Not Use)

| Location | Contents |
|----------|----------|
| `Data/tft/_archive_v1/` | V1 training data |
| `Data/tft/_archive_intermediate/` | Debug artifacts |
| `Data/tft/runs/_archive/` | Old build runs |
| `backend/models/_deprecated/` | Old checkpoints |
| `Docs/Data_docs/_deprecated/` | Old docs |

---

*Document version: 1.0*  
*Created: 2025-12-21*

