# Data Quality Report

> ⚠️ **NOTE: This document covers PROFILING results, not CLEANING status.**  
> Profiling results are accurate. However, cleaning was incomplete.  
> **See [`DATA_PIPELINE_ERRORS_AND_LESSONS.md`](./DATA_PIPELINE_ERRORS_AND_LESSONS.md)** for actual cleaning status.

**Generated:** 2025-12-09 18:13  
**Status:** Profiling Complete (Cleaning Incomplete - see errors doc)

---

## 1. Executive Summary

### Dataset Overview

| Dataset | Rows | Columns | Null Rate | Status |
|---------|------|---------|-----------|--------|
| Transactions | 1,612,276 | 46 | - | ✅ Profiled |
| Rent_Contracts | 9,524,298 | 40 | - | ✅ Profiled |
| Units | 2,335,623 | 46 | - | ✅ Profiled |
| Buildings | 239,277 | 45 | - | ✅ Profiled |
| Projects | 3,039 | 37 | - | ✅ Profiled |
| Valuation | 87,093 | 20 | - | ✅ Profiled |

## 2. Critical Findings

### 2.1 Transactions Analysis

**Date Parsing Success Rate:** 100.0%

**Transaction Types:**

| Type | Count |
|------|-------|
| Sell - Pre registration | 537,071 |
| Sell | 481,758 |
| Mortgage Registration | 218,826 |
| Delayed Sell | 145,759 |
| Lease to Own Registration | 58,677 |
| Grant | 53,592 |
| Modify Mortgage | 21,125 |
| Delayed Mortgage | 16,994 |
| Development Registration | 12,501 |
| Sell Development | 12,037 |

**Price Distribution:**

| Range | Count |
|-------|-------|
| 0-100K | 4,050 |
| 100K-500K | 179,547 |
| 500K-1M | 400,145 |
| 1M-2M | 488,516 |
| 2M-5M | 397,558 |
| 5M-10M | 76,304 |
| 10M-50M | 54,532 |
| 50M+ | 11,624 |

### 2.2 Rent Contracts Analysis

**Bedroom Parsing Success Rate:** 15.3%

Parsed bedroom distribution:

| Bedroom Type | Count |
|--------------|-------|
| 1BR | 126 |
| 2BR | 147 |
| 3BR | 96 |
| 4BR | 96 |
| 5BR+ | 312 |
| Studio | 96 |

⚠️ **Unparseable samples (need manual mapping):**
```
Office
Shop
Hotel
GYM
Commercial villa
8 bed rooms+hall
Warehouse
Kiosk
Clinic
Shop with a mezzanine
```

**Date Coverage:** 2001-02-15 to 2205-07-16

**Rent Distribution (Annual):**

| Range | Count |
|-------|-------|
| 0-20K AED | 466,650 |
| 20K-40K AED | 1,789,287 |
| 40K-60K AED | 1,947,583 |
| 60K-80K AED | 1,329,622 |
| 80K-100K AED | 754,037 |
| 100K-150K AED | 972,591 |
| 150K-200K AED | 482,604 |
| 200K-300K AED | 419,410 |
| 300K-500K AED | 335,485 |
| 500K+ AED | 1,027,029 |

### 2.3 Entity Resolution

**Unique Entities Found:**

| Entity Type | Count |
|-------------|-------|
| area | 262 |
| project | 6,479 |
| developer | 112 |

**Spelling Conflicts (same entity, different spellings):**

| Entity Type | Conflicts |
|-------------|----------|
| area | 1 |
| project | 12 |

### 2.4 Tourism Data

- **Files processed:** 58
- **Year coverage:** [2021, 2022, 2023, 2024, 2025]
- **Translation coverage:** 0.0%
- **Untranslated columns:** 6

### 2.5 Small File Analysis

**Buildings.csv - High Null Columns (>50%):**
- `pre_registration_number`: 100.0%
- `swimming_pools`: 98.2%
- `elevators`: 98.2%
- `flats`: 97.7%
- `bld_levels`: 97.2%

**Projects.csv - High Null Columns (>50%):**
- `cancellation_date`: 98.9%

## 3. Recommended Actions

### Priority 1 (Blocking)

1. **Fix date parsing** - Ensure DD-MM-YYYY format is consistently parsed
2. **Create bedroom mapping** - Map all unique bedroom strings to standard labels (Studio, 1BR, 2BR, 3BR, 4BR+)
3. **Resolve entity conflicts** - Review and approve canonical entity names for areas and projects

### Priority 2 (Important)

4. **Complete tourism translations** - Review untranslated Arabic columns
5. **Handle outliers** - Define filtering rules for extreme price/rent values
6. **Document data gaps** - Identify temporal/geographic coverage gaps

### Priority 3 (Enhancement)

7. **Create data dictionary** - Document all column meanings and valid values
8. **Build validation tests** - Automated checks for data quality

---

## 4. Profiling Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Transaction Profile | `data_profiles/transactions_profile.json` | Chunked analysis of transactions |
| Rent Contracts Profile | `data_profiles/rent_contracts_profile.json` | RERA-focused analysis |
| Entity Master List | `data_profiles/entity_resolution/master_entity_list.json` | Canonical entity names |
| Entity Conflicts | `data_profiles/entity_resolution/conflicts.json` | Spelling variations |
| Tourism Catalog | `data_profiles/tourism/tourism_catalog.json` | All tourism files profiled |
| Column Translations | `data_profiles/tourism/column_translations.json` | Arabic→English mapping |

---

## 5. Next Steps

- [ ] Review this report
- [ ] Address Priority 1 items
- [ ] Create manual mappings for unparseable values
- [ ] Re-run profiling to verify fixes
- [ ] Update PRD timeline if significant issues found
- [ ] Proceed to Phase 1 (Database Setup)

