# TFT V3 Model Audit Report

**Generated:** 2025-12-20  
**Model:** `tft_final_v3.ckpt`  
**Data:** `tft_training_data_v2.csv` (272,257 rows, 821 groups)

---

## Executive Summary

The V3 model loaded and initialized correctly with:
- ✅ 96-month encoder (8 years of history)
- ✅ 12-month prediction horizon
- ✅ `units_completing` as a known future feature
- ✅ Developer brand labels as static categoricals

**However, the audit found critical data quality issues that need attention:**

| Issue | Severity | Impact |
|-------|----------|--------|
| Duplicate developer brand names | 🔴 HIGH | Emaar split across 3 labels (212K+ transactions) |
| 363 unmapped DEVELOPER_ID_* labels | 🔴 HIGH | 5,675 transactions under just the top unmapped ID |
| 30.6% groups with <24 months history | 🟡 MEDIUM | Cold start predictions less reliable |
| 4 areas with OffPlan-only data | 🟡 MEDIUM | No Ready comparables for these areas |

---

## 🔴 CRITICAL: Developer Mapping Issues

### Issue 1: Duplicate Developer Brand Names

The same developer appears under multiple brand labels, splitting their transaction history:

| Developer | Labels | Total Transactions |
|-----------|--------|-------------------|
| **Emaar** | "Emaar" (162K) + "EMAAR PROPERTIES" (50K) + "EMAAR DEVELOPMENT" (26K) | **238,749** |
| **Nakheel** | "Nakheel" (40K) + "NAKHEEL" (12K) | **51,473** |
| **DAMAC** | "DAMAC" (50K) + "DAMAC PROPERTIES" (10K) | **60,131** |
| **Deyaar** | "Deyaar" (14K) + "DEYAAR DEVELOPMENT" (4K) | **~18,000** |

**Impact:** The model learns separate embeddings for each variant, fragmenting the developer's true market signal.

**Recommendation:** Consolidate these in the V2 data builder under a single canonical brand name.

---

### Issue 2: Unmapped DEVELOPER_ID_* Labels

363 developer labels remain as raw IDs (e.g., `DEVELOPER_ID_14868751`) instead of human-readable names:

| Label | Groups | Transactions |
|-------|--------|--------------|
| DEVELOPER_ID_14868751 | 12 | 5,675 |
| DEVELOPER_ID_153 | 13 | 5,664 |
| DEVELOPER_ID_547 | 6 | 5,108 |
| DEVELOPER_ID_52800220 | 17 | 2,576 |
| DEVELOPER_ID_920 | 4 | 2,512 |
| *...361 more...* | | |

**Impact:** 
- Users see meaningless IDs instead of developer names
- Model can't leverage brand reputation signal for these developers
- Represents ~15% of total transaction volume

**Recommendation:** 
1. Identify these developer IDs in the raw DLD data
2. Map them to canonical brand names
3. Update `scripts/brand_resolver.py` with new mappings

---

## 🟡 Coverage Gaps

### Groups with Insufficient History

| History Length | Groups | % of Total |
|----------------|--------|------------|
| < 12 months | 89 | 10.8% |
| 12-23 months | 162 | 19.7% |
| 24+ months | 570 | 69.4% |

**30.6% of groups have <24 months history** — the model's cold start filter should exclude these during training (which it does), but inference on these groups will be less reliable.

---

### Areas with OffPlan-Only Data

These 4 areas have no Ready transaction comparables:

1. **Bukadra**
2. **Hessyan Second**
3. **Palm Deira**
4. **Ras Al Khor Industrial First**

**Impact:** Model can't learn Ready vs OffPlan price differentials for these areas.

---

## ✅ Model Configuration (Verified)

| Parameter | Value |
|-----------|-------|
| `max_encoder_length` | 96 months (8 years) |
| `max_prediction_length` | 12 months |
| `static_categoricals` | area_name, property_type, bedroom, reg_type, developer_brand_label, developer_umbrella |
| `time_varying_known_reals` | time_idx, month_sin, month_cos, **units_completing** |

The model correctly includes `units_completing` as a known future feature, which was the primary goal of V3 training.

---

## ✅ Data Quality (Verified)

| Metric | Value |
|--------|-------|
| `units_completing` non-zero rows | 53,697 (19.7%) |
| `units_completing` max | 10,853 units |
| `units_completing` mean (when >0) | 466 units |
| Rows with zero/negative price | 0 ✅ |

---

## Top 20 Developers by Transaction Volume

| Rank | Developer | Groups | Transactions |
|------|-----------|--------|--------------|
| 1 | Emaar | 720 | 162,053 |
| 2 | EMAAR PROPERTIES | 61 | 50,506 |
| 3 | DAMAC | 620 | 50,131 |
| 4 | Nakheel | 496 | 39,673 |
| 5 | EMAAR DEVELOPMENT | 54 | 26,190 |
| 6 | Sobha | 468 | 23,085 |
| 7 | Meraas | 560 | 20,794 |
| 8 | Binghatti | 53 | 19,017 |
| 9 | Nshama | 574 | 17,930 |
| 10 | Dubai Properties | 507 | 17,592 |
| 11 | Danube | 442 | 16,754 |
| 12 | Tiger Group | 532 | 16,531 |
| 13 | Dubai South Properties | 431 | 15,893 |
| 14 | Dubai Holding | 559 | 14,909 |
| 15 | Deyaar | 545 | 13,564 |
| 16 | Select Group | 517 | 12,978 |
| 17 | Tameer | 448 | 12,059 |
| 18 | Meydan | 261 | 11,879 |
| 19 | NAKHEEL | 14 | 11,800 |
| 20 | Aldar | 467 | 11,722 |

---

## Top 15 Developer Umbrellas

| Umbrella | Groups |
|----------|--------|
| Emaar | 145 |
| Nakheel | 66 |
| Dubai Holding | 60 |
| Azizi | 58 |
| DAMAC | 55 |
| Dubai Properties | 44 |
| Meraas | 41 |
| Sobha | 37 |
| TECOM | 35 |
| Samana Developers | 33 |
| Dubai Investments | 30 |
| London Gate | 27 |
| Palma Holding | 24 |
| Imtiaz Developments | 22 |
| ETA Star | 20 |

---

## Recommendations

### Priority 1 (Before Production)
1. **Consolidate duplicate developer names** in `build_tft_data_v2.py`:
   - Map "EMAAR PROPERTIES" → "Emaar"
   - Map "EMAAR DEVELOPMENT" → "Emaar"
   - Map "NAKHEEL" → "Nakheel"
   - Map "DAMAC PROPERTIES" → "DAMAC"
   - Map "DEYAAR DEVELOPMENT" → "Deyaar"

### Priority 2 (Next Data Refresh)
2. **Map remaining DEVELOPER_ID_* labels** by:
   - Cross-referencing with DLD developer registry
   - Manual lookup for top 50 unmapped IDs

### Priority 3 (Inference Code)
3. **Update `tft_inference.py`** to:
   - Use the patched checkpoint (`tft_final_v3__patched.ckpt`)
   - Generate `units_completing` from actual project schedules (not rolling average)
   - Handle the V2 group_id format (area_id based, not developer-in-group)

---

## Files

- **Patched model:** `backend/models/tft_final_v3__patched.ckpt`
- **Training data:** `Data/tft/runs/20251218T163716Z/tft_training_data_v2.csv`
- **Full audit JSON:** `data_profiles/tft_v3_audit_report.json`

