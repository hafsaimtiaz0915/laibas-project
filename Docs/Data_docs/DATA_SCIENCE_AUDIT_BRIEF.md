# Complete Data Science Audit Brief — Real Estate Price Forecasting

---

## ⚠️ Important Context

**This entire data pipeline has been built using AI assistance (Claude/Cursor).** While functional, there are significant concerns about:

1. **Data cleaning accuracy** — The cleaning scripts were AI-generated and may contain logic errors, edge cases not handled, or incorrect assumptions about the raw data.

2. **Entity resolution correctness** — The developer/brand mapping logic was designed by AI and has already shown errors (e.g., incorrect umbrella groupings, duplicate brand names not consolidated).

3. **Aggregation logic** — The decision to exclude developer from `group_id`, the feature engineering choices, and the merge logic were all AI-driven without domain expert validation.

4. **Silent failures** — There may be data quality issues that haven't surfaced yet — incorrect joins, dropped rows, type coercions, or business logic violations.

**What we need:** A human data scientist to independently verify:
- That the raw → cleaned → aggregated pipeline is correct
- That the entity resolution (especially developers) makes business sense
- That the model architecture and feature choices are appropriate
- That there are no systematic data quality issues

**This is not a "refine the model" request — this is a "please verify the foundations are sound" request.**

**Related Documents:**
- `Docs/PROJECT_STATE_FOR_DATA_SCIENTIST.md` — What files to use
- `Docs/ARCHITECTURE_FILE_REFERENCE.md` — Complete file reference with descriptions

---

## 1. Project Goal

Build a time series forecasting model to predict:
- **Property sale prices** (price per square foot) — 12-month horizon
- **Rental yields** (annual rent) — 12-month horizon

The model should understand patterns from geography, property characteristics, developer/builder reputation, market conditions, and upcoming supply.

---

## 2. Raw Data Sources

All raw data comes from an official land registry (government property database).

### 2.1 Transactions (Primary Source)

| File | Records | Description |
|------|---------|-------------|
| `Data/raw_data/Transactions.csv` | ~500,000 | Every property sale transaction |

**Key columns:**
| Column | Description |
|--------|-------------|
| `transaction_id` | Unique transaction identifier |
| `transaction_date` | Date of sale |
| `property_id` | Links to physical unit |
| `area_id` | Geographic zone (official boundary) |
| `trans_group_id` | Transaction type (sale, mortgage, gift, etc.) |
| `actual_worth` | Transaction price |
| `meter_sale_price` | Price per square meter |
| `procedure_area` | Size in sqm |
| `rooms_en` | Bedroom count ("1 B/R", "2 B/R", "Studio", etc.) |
| `property_type_en` | "Unit" (apartment) or "Villa" |
| `reg_type_en` | "Off-Plan Properties" (pre-construction) or "Existing Properties" (resale/ready) |
| `project_number` | Links to development project (for off-plan) |
| `building_number` | Physical building identifier |

---

### 2.2 Projects (Development Pipeline)

| File | Records | Description |
|------|---------|-------------|
| `Data/raw_data/Projects.csv` | ~3,000 | Development projects (new builds) |

**Key columns:**
| Column | Description |
|--------|-------------|
| `project_id`, `project_number`, `project_name` | Identifiers |
| `developer_id`, `developer_name` | Who built it (Arabic + English names) |
| `master_developer_id`, `master_developer_name` | Parent company |
| `area_id`, `area_name_en` | Location |
| `project_start_date`, `project_end_date`, `completion_date` | Timeline |
| `percent_completed` | 0-100 progress |
| `project_status` | "FINISHED", "ONGOING", "CANCELLED" |
| `no_of_units`, `no_of_villas`, `no_of_buildings` | Unit counts |

**Critical Use:** Calculates future supply (`units_completing`) by handover date.

---

### 2.3 Buildings (Property Registry)

| File | Records | Description |
|------|---------|-------------|
| `Data/raw_data/Buildings.csv` | ~240,000 | Physical building/property records |

**Key columns:**
| Column | Description |
|--------|-------------|
| `property_id` | Unique identifier (joins to transactions) |
| `area_id`, `area_name_en` | Location |
| `floors`, `rooms_en` | Physical characteristics |
| `built_up_area`, `actual_area` | Sizes |
| `property_type_en`, `property_sub_type_en` | "Unit", "Villa" |
| `project_id`, `project_name_en` | Links to project |
| `is_free_hold`, `is_registered` | Ownership status |

---

### 2.4 Developers (Entity Registry)

| File | Records | Description |
|------|---------|-------------|
| `Data/raw_data/new_raw_data/Developers.csv` | ~2,100 | Registered developers/builders |

**Key columns:**
| Column | Description |
|--------|-------------|
| `developer_id`, `developer_number` | Unique identifiers |
| `developer_name_ar`, `developer_name_en` | Arabic and English names |
| `registration_date` | When they registered |
| `license_type_en`, `license_number` | Regulatory info |
| `legal_status_en` | Corporate structure |

---

### 2.5 Areas (Geography Lookup)

| File | Records | Description |
|------|---------|-------------|
| `Data/raw_data/new_raw_data/Lkp_Areas.csv` | ~300 | Geographic zone lookup |

**Key columns:**
| Column | Description |
|--------|-------------|
| `area_id` | Unique identifier (THE primary join key) |
| `name_en`, `name_ar` | Area names in English and Arabic |
| `municipality_number` | Links to KML polygon boundaries |

**Supplementary:** `Community.kml` contains polygon boundaries with lat/lon centroids.

---

### 2.6 Rent Contracts

| File | Records | Description |
|------|---------|-------------|
| `Data/raw_data/Rent_Contracts.csv` | ~100,000 | Lease agreements |

**Key columns:**
| Column | Description |
|--------|-------------|
| `contract_id`, `contract_date` | Identifiers |
| `area_id`, `property_id` | Location |
| `annual_amount` | Yearly rent |
| `property_type`, `bedroom` | Characteristics |

---

### 2.7 Price Index (Market Benchmark)

| File | Records | Description |
|------|---------|-------------|
| `Data/raw_data/new_raw_data/Residential_Sale_Index.csv` | ~160 | Monthly market-wide price index |

**Key columns:**
| Column | Description |
|--------|-------------|
| `year_month` | Time period |
| `all_monthly_index` | Overall market index |
| `flat_monthly_index` | Apartment index |
| `villa_monthly_index` | Villa index |

---

## 3. Data Processing Pipeline

### Stage 1: Data Cleaning

**Script:** `scripts/clean_all_data.py`

**Operations:**
- Remove duplicate records
- Standardize column names and date formats
- Filter to residential transactions only (exclude commercial, land)
- Handle missing values
- Validate data types

**Output:** `Data/cleaned/*.csv`

---

### Stage 2: Lookup Table Generation

**Script:** `scripts/generate_lookup_tables.py`

**Operations:**
- Build area reference tables (id → name mapping)
- Build developer entity resolution tables
- Create brand consolidation mappings (SPVs → parent brands)

**Output:** `Data/lookups/*.json` and `*.csv`

---

### Stage 3: TFT Data Build (Core Aggregation)

**Script:** `scripts/build_tft_data_v2.py`

This is the **core aggregation script** that creates training data. It performs:

1. **Load cleaned transactions** — filter to sales only
2. **Map bedroom codes** — "1 B/R" → "1BR", "Studio" → "Studio"
3. **Map property types** — filter to "Unit" or "Villa"
4. **Map registration types** — "Off-Plan Properties" → "OffPlan", "Existing Properties" → "Ready"
5. **Aggregate by month-segment** — calculate medians per group
6. **Resolve developer brands** — via lookup files, handle SPVs
7. **Merge rent data** — median rent per segment
8. **Merge supply data** — active projects, upcoming completions (`units_completing`)
9. **Merge geography** — lat/lon centroids, area names
10. **Merge price index** — macro regime indicator
11. **Create time features** — month, quarter, sin/cos encoding
12. **Create time_idx** — monotonic counter per group (required for TFT)

**Output:** `Data/tft/runs/<BUILD_ID>/tft_training_data_v2.csv`

---

### Stage 4: Group Definition (Current Implementation)

**Formula:**
```
group_id = area_id + "_" + property_type + "_" + bedroom + "_" + reg_type
```

**Example:** `330_Unit_2BR_Ready`
- Area 330 (Dubai Marina)
- Unit (apartment)
- 2 Bedroom
- Ready (resale, not off-plan)

**Result:** ~800 unique time series groups

**Critical Design Choice:** Developer is NOT included in the group_id.

---

## 4. Lookup Tables Reference

### Developer Resolution Files

| File | Description |
|------|-------------|
| `Data/lookups/developer_brand_consolidation.json` | Maps multiple SPVs/legal entities to parent brand |
| `Data/lookups/umbrella_map.json` | Maps developer_id to corporate umbrella |
| `Data/lookups/public_brands.json` | Canonical brand names |
| `Data/lookups/blocked_brand_labels.json` | Brands to exclude |
| `Data/lookups/top50_developers_2025.json` | Major developers list |

### Geography Files

| File | Description |
|------|-------------|
| `Data/lookups/area_mapping.json` | area_id → area_name |
| `Data/lookups/area_reference.csv` | Full area reference table |
| `Data/lookups/area_stats.csv` | Transaction counts per area |

### Override Files

| File | Description |
|------|-------------|
| `Data/lookups/brand_overrides_developer_id.csv` | Manual developer corrections |
| `Data/lookups/entity_owner_overrides_developer_id.csv` | Owner entity overrides |

---

## 5. Final Training Data Schema

**File:** `Data/tft/runs/<BUILD_ID>/tft_training_data_v2.csv`  
**Rows:** ~272,000  
**Groups:** ~800  
**Time span:** 8 years (96 months)

### Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `year_month` | string | "YYYY-MM" format |
| `area_id` | int | Geographic zone ID |
| `property_type` | string | "Unit" or "Villa" |
| `bedroom` | string | "Studio", "1BR", "2BR", ... "6BR+" |
| `reg_type` | string | "OffPlan" or "Ready" |
| `developer_brand_label` | string | Resolved brand name or "DEVELOPER_ID_xxx" |
| `developer_umbrella` | string | Corporate parent group |
| `developer_id` | float | Raw developer ID |
| `median_price` | float | **TARGET 1:** Median price per sqft |
| `median_rent` | float | **TARGET 2:** Median annual rent |
| `transaction_count` | int | Number of sales in this month/segment |
| `rent_count` | int | Number of rent contracts |
| `months_since_launch` | float | Months since project started |
| `months_to_handover` | float | Months until project completes |
| `project_percent_complete` | float | 0-100 completion progress |
| `units_completing` | int | **CRITICAL:** Units scheduled to complete this month in this area |
| `active_projects` | int | Ongoing projects in area |
| `supply_units` | int | Total pipeline units |
| `area_name` | string | Human-readable area name |
| `centroid_lon`, `centroid_lat` | float | Geographic center |
| `all_monthly_index` | float | Market-wide price index |
| `flat_monthly_index` | float | Apartment price index |
| `villa_monthly_index` | float | Villa price index |
| `month` | string | Month number (1-12) |
| `quarter` | string | Quarter (1-4) |
| `month_sin`, `month_cos` | float | Cyclical encoding of month |
| `time_idx` | int | Monotonic time index per group (required for TFT) |
| `group_id` | string | Unique series identifier |

---

## 6. Complete Problem List

### 6.0 AI-Generated Pipeline Concerns (🔴 FOUNDATIONAL)

The entire pipeline from raw data to training data was built with AI assistance. **We do not have high confidence that it is correct.**

| Concern | Description | Risk |
|---------|-------------|------|
| **Cleaning logic** | Scripts may silently drop valid records or include invalid ones | Data loss / contamination |
| **Join correctness** | Merges on area_id, developer_id, project_number may have issues | Misaligned features |
| **Type coercion** | String/int/float handling may cause silent data corruption | Wrong aggregations |
| **Business rules** | AI may have encoded incorrect assumptions about real estate | Invalid features |
| **Edge cases** | Null handling, duplicates, date parsing may be wrong | Systematic bias |

**Specific areas that need human verification:**

1. **Transaction filtering** — Are we correctly filtering to residential sales only? Are we excluding mortgages, gifts, family transfers correctly?

2. **Developer resolution** — The brand_resolver.py logic is complex and AI-generated. We've already found errors (duplicate brands, wrong umbrellas).

3. **Project-to-transaction linking** — Off-plan transactions link to projects via `project_number`. Is this join correct? Are we losing records?

4. **Rent imputation** — Missing rent values are forward-filled and imputed. Is this appropriate? Does it introduce bias?

5. **Supply calculation** — `units_completing` is calculated from project handover dates. Is the logic correct? Are cancelled projects excluded?

6. **Time series construction** — The `time_idx` and `group_id` creation logic was AI-designed. Is it correct for TFT requirements?

---

### 6.1 Developer Entity Resolution (🔴 CRITICAL)

| Problem | Description | Evidence |
|---------|-------------|----------|
| **A. Duplicate brand names** | Same developer appears under multiple labels | "Emaar" (162K tx) + "EMAAR PROPERTIES" (50K) + "EMAAR DEVELOPMENT" (26K) = 238K total, but model sees 3 separate entities |
| **B. Case sensitivity** | "Nakheel" vs "NAKHEEL" treated as different | 40K + 12K = 52K total split across 2 labels |
| **C. Unmapped IDs** | 363 developers show as `DEVELOPER_ID_12345` | ~15% of transaction volume has meaningless labels |
| **D. Umbrella inaccuracy** | Corporate grouping is incorrect in places | Found DAMAC/Nakheel incorrectly grouped under "Emaar" umbrella |
| **E. SPV complexity** | Developers use multiple legal entities per project | Emaar alone has 18+ registered SPVs (Dubai Hills Estate LLC, Creek Heights LLC, etc.) |

**Root cause:** The brand resolution logic has gaps and inconsistent case handling.

**Impact:** Model learns separate embeddings for each variant, fragmenting the developer's true market signal.

---

### 6.2 Group ID Design Problem (🔴 CRITICAL)

| Problem | Description | Impact |
|---------|-------------|--------|
| **Developer not in group_id** | Each time series mixes multiple developers | Model cannot learn developer-specific trends |
| **Developer varies within series** | Row 1 = Emaar, Row 2 = DAMAC, Row 3 = Nakheel | The `developer_brand_label` feature becomes noise |

**Example of the problem:**
```
group_id: 330_Unit_2BR_Ready (Dubai Marina, 2BR apartments, Ready)

Month 1: Emaar transaction
Month 2: DAMAC transaction  
Month 3: Nakheel transaction
Month 4: Emaar transaction
```

The model sees this as ONE time series, but the developer changes every month!

**Trade-off options:**
| Option | Pros | Cons |
|--------|------|------|
| A. Add developer to group_id | Learn developer-specific trends | ~5,000+ groups, many too thin |
| B. Keep as-is (developer as feature) | Fewer groups, more data per group | Developer signal is noise |
| C. Aggregate developer features | Capture developer "premium" | Loses individual developer learning |
| D. Hierarchical model | Best of both worlds | More complex architecture |

---

### 6.3 Cold Start Problem (🟡 HIGH)

**Current data distribution:**
| History Length | Groups | % of Total |
|----------------|--------|------------|
| < 12 months | 89 | 10.8% |
| 12-23 months | 162 | 19.7% |
| 24+ months | 570 | 69.4% |

**30.6% of groups have less than 24 months of history.**

**Causes:**
- New areas being developed (greenfield projects)
- New property types in existing areas
- Market shifts (villas becoming available in apartment-only areas)
- New developers entering market

**Current handling:** Filter groups with insufficient history during training. But inference still asked to forecast these groups.

**Questions:**
- What's the minimum viable history length?
- How to handle forecasts for thin groups?
- Should we use hierarchical fallbacks?
- Should we show explicit uncertainty for thin groups?

---

### 6.4 Geography/Area Mapping (🟡 MEDIUM)

| Problem | Description |
|---------|-------------|
| **Granularity variation** | Some areas have 50K transactions, others have 50 |
| **Missing boundaries** | 95% of areas have KML polygon matches, 5% don't |
| **No sub-area granularity** | Large areas (Business Bay, Marina) have heterogeneous neighborhoods |
| **No proximity features** | Distance to landmarks, metro, beach not captured |

**Available but unused data:**
- Lat/lon centroids for each area
- Polygon boundaries in KML
- Building-level coordinates

---

### 6.5 Feature Engineering Gaps (🟡 MEDIUM)

**Currently included (58 columns):**
- Time: `month`, `quarter`, `month_sin`, `month_cos`, `time_idx`
- Price/rent: `median_price`, `median_rent`, `transaction_count`, `rent_count`
- Project: `months_since_launch`, `months_to_handover`, `project_percent_complete`
- Supply: `supply_units`, `active_projects`, `units_completing`
- Index: `all_monthly_index`, `flat_monthly_index`, `villa_monthly_index`
- Developer: `developer_brand_label`, `developer_umbrella`, `developer_id`

**Missing/questionable features:**
| Feature | Description | Status |
|---------|-------------|--------|
| Developer premium | Does Emaar command higher prices than market? | Not calculated |
| Developer track record | Completion rate, delay history | Not included |
| Area momentum | Is this area appreciating vs market? | Not calculated |
| Inventory pressure | Supply/demand ratio | Not included |
| Macro factors | Interest rates, GDP, population | Have EIBOR, not using |

**Potential data leakage:**
- `transaction_count` in same month as target `median_price`
- `percent_completed` for exact project being sold

---

### 6.6 Target Variable Definition (🟡 MEDIUM)

**Current definition:**
- `median_price` = median of all transaction prices (per sqft) in that month/segment
- `median_rent` = median annual rent

**Issues:**
| Issue | Description |
|-------|-------------|
| Small sample noise | Median with <5 transactions is unreliable |
| Price type mixing | Off-plan = "list prices", Ready = "market prices" |
| Primary vs secondary | No distinction between developer sales vs resales |
| Sparse rent data | Many segments have 0 rent observations |

**Current handling:** Forward-fill + back-fill rent, impute missing with area-level benchmark.

---

### 6.7 Training-Inference Mismatch (🔴 CRITICAL)

**The `units_completing` feature:**

| Stage | Current Implementation |
|-------|------------------------|
| **Training** | Calculated from actual project handover dates |
| **Inference** | Uses rolling average (NOT actual schedule) |

**Impact:** The model learned that `units_completing = 500` means "500 units will complete this month" because that's what training data showed. But inference generates fake numbers, breaking the entire future supply signal.

**Required fix:** Inference must query the same project schedule data used in training.

---

## 7. Expected Model Output

### 7.1 Forecast Format

| Field | Description |
|-------|-------------|
| `group_id` | Which segment (area + property + bedroom + reg_type) |
| `forecast_date` | Which month being predicted |
| `price_p10` | 10th percentile price forecast (lower bound) |
| `price_p50` | 50th percentile price forecast (point estimate) |
| `price_p90` | 90th percentile price forecast (upper bound) |
| `rent_p10`, `rent_p50`, `rent_p90` | Same for rent |
| `confidence_score` | Reliability indicator |

### 7.2 Use Cases

| Use Case | Query | Required Output |
|----------|-------|-----------------|
| Property valuation | "What's a 2BR in Area X worth in 12 months?" | Point estimate + confidence interval |
| Developer comparison | "Is Developer A better than Developer B?" | Developer-specific forecasts for same segment |
| Market trends | "Is this area appreciating?" | Direction + magnitude |
| Investment analysis | "What's the rental yield outlook?" | Rent forecast ÷ price forecast |

### 7.3 Interpretability Requirements

The model should explain:
- Which features drove the forecast up/down
- How this segment compares to similar segments
- What's the developer effect (premium/discount)
- Confidence level based on data quality

---

## 8. Trained Models

We have **two trained TFT models** in the codebase. Both have the same underlying data quality issues, but differ in training approach.

### 8.1 Model Comparison

| Metric | Console Model | Colab V3 Model |
|--------|---------------|----------------|
| **File** | `backend/models/Console_model/output_tft_best_v2.ckpt` | `backend/models/tft_final_v3.ckpt` |
| **File Size** | 5.5 MB | 9.8 MB |
| **Training Steps** | 250,074 ✅ | 60,000 |
| **Training Epochs** | 98 | 99 |
| **Encoder Length** | 96 months (8 years) | 96 months (8 years) |
| **Prediction Length** | 12 months | 12 months |
| **PyTorch Lightning** | 2.6.0 | 2.4.0 |
| **Time-Varying Features** | 11 (simpler) | 34 (complex) |

### 8.2 Feature Differences

**Both models share:**
- Static categoricals: `area_name`, `property_type`, `bedroom`, `reg_type`, `developer_brand_label`, `developer_umbrella`
- Time-varying known: `time_idx`, `month_sin`, `month_cos`, `units_completing`
- Targets: `median_price`, `median_rent` (7 quantiles each)

**Console Model (11 time-varying unknown features):**
```
transaction_count, rent_count, median_rent_sqft,
months_to_handover_signed, months_since_handover, handover_window_6m,
supply_units, active_projects, sale_index,
median_price, median_rent
```

**Colab V3 Model (34 time-varying unknown features):**
```
All Console features PLUS:
- area_id, developer_id, developer_brand_evidence_score, developer_fallback_used
- months_since_launch, months_to_handover, project_percent_complete, project_duration_months, phase_ratio
- supply_buildings, supply_villas, community_num
- centroid_lon, centroid_lat, polygon_area_deg2
- all_monthly_index, flat_monthly_index, villa_monthly_index
- all_monthly_price_index, flat_monthly_price_index, villa_monthly_price_index
- sale_index_missing, is_top50_2025, has_actual_rent, dld_offplan_after_handover
```

### 8.3 What's Missing from BOTH Models (🔴 CRITICAL)

| Missing Element | Why It's a Problem |
|-----------------|-------------------|
| **Developer in group_id** | Both models treat developer as a categorical feature that varies within each time series. This means the model sees "Area X, 2BR, Ready" as ONE series with Emaar transactions in month 1, DAMAC in month 2, Nakheel in month 3. **The model cannot learn developer-specific pricing patterns.** |
| **Correct developer entity resolution** | Both models were trained on data with duplicate brand names ("Emaar" vs "EMAAR PROPERTIES" vs "EMAAR DEVELOPMENT"). The model learns 3 separate embeddings for what is actually 1 developer. **Developer signal is fragmented.** |
| **Correct umbrella mapping** | Both models include `developer_umbrella` as a static categorical, but we've confirmed the umbrella mapping is incorrect (e.g., DAMAC incorrectly grouped under "Emaar" umbrella). **The umbrella feature may be adding noise, not signal.** |
| **Unmapped developer IDs** | ~15% of transactions have meaningless labels like `DEVELOPER_ID_14868751` instead of brand names. The model learns embeddings for these IDs, but **they provide no semantic meaning.** |
| **Developer track record features** | Neither model includes features like "developer's historical completion rate", "average delay", or "price premium vs market". **There's no way to capture developer reputation.** |

### 8.4 Console Model Specific Issues

| Issue | Impact |
|-------|--------|
| **Fewer features** | May miss important signals (price indices, geographic coordinates, project details) |
| **No developer_id numeric** | Cannot learn from developer ID patterns (though this may be noise anyway) |
| **No geographic features** | No lat/lon, no polygon area — cannot learn spatial patterns |

**However:** Simpler models are less prone to overfitting. The Console model trained 4x longer and may generalize better.

### 8.5 Colab V3 Model Specific Issues

| Issue | Impact |
|-------|--------|
| **34 time-varying features** | High risk of overfitting — many features may be noise |
| **Includes `area_id` as numeric** | Area ID is arbitrary (e.g., 330 vs 232) — model may learn spurious patterns |
| **Includes `developer_id` as numeric** | Developer IDs are arbitrary — no semantic meaning |
| **Shorter training** | Only 60K steps vs 250K for Console — may not have converged |
| **Geographic coordinates as features** | `centroid_lon`, `centroid_lat` may cause issues if model tries to extrapolate spatially |

### 8.6 Fundamental Problem Shared by Both

**Neither model can answer: "How does Emaar compare to DAMAC in the same area?"**

Because:
1. Developer is not in group_id, so there's no Emaar-specific time series
2. Developer varies within each time series, so the model sees mixed signals
3. Developer features are categorical embeddings learned from noisy data
4. The model has no explicit "developer premium" feature

**To properly learn developer effects, we would need:**
- Option A: Include developer in group_id (creates ~5,000 thin groups)
- Option B: Engineer explicit developer features (premium, track record, completion rate)
- Option C: Use a hierarchical model (area-level + developer adjustment)
- Option D: Post-hoc developer adjustment layer

### 8.7 Technical Context

| Parameter | Value |
|-----------|-------|
| **Framework** | PyTorch Forecasting (Temporal Fusion Transformer) |
| **Training environment** | Google Colab / Google Cloud Console |
| **Encoder length** | 96 months (8 years of history) |
| **Prediction horizon** | 12 months |
| **Number of groups** | ~800 |
| **Training rows** | ~272,000 |
| **Targets** | `median_price`, `median_rent` |
| **Loss function** | MultiLoss (QuantileLoss for each target) |
| **Quantiles** | [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] |

### 8.8 Recommendation

**For immediate use:** The Console model is likely more reliable due to:
- Simpler feature set (less overfitting risk)
- Longer training (better convergence)
- Same core architecture

**For production:** Neither model should be deployed until:
1. Developer entity resolution is fixed
2. A decision is made on how to model developer effects
3. The training-inference mismatch for `units_completing` is resolved
4. Independent validation confirms the pipeline is correct

### 8.9 Recommended Additional Features

#### Floor Height (🟢 SHOULD ADD)

**Data availability:** ✅ `Units.csv` has floor data for 99% of 2.3M units

| Data Point | Status |
|------------|--------|
| Unit floor number | ✅ In Units.csv (99% populated) |
| Building total floors | ✅ In Buildings.csv |
| Floor range | Ground to 132 floors |

**Implementation complexity:** MEDIUM
- Requires joining Transactions → Units (fuzzy match on building_name + area + size)
- Expected match rate: 60-80%
- Ambiguity in multi-unit matches

**Features to engineer:**

| Feature | Description | Expected Value |
|---------|-------------|----------------|
| `floor_number` | Unit's floor (from Units.csv) | Direct signal |
| `floor_percentile` | floor / building_total_floors | Normalized position |
| `is_high_floor` | floor > 20 | Binary premium indicator |
| `building_total_floors` | Total floors in building | Building quality proxy |

**Expected value:** Floor premiums are well-documented in real estate:
- Ground floor: typically -10-15% discount
- Mid-floors (10-20): baseline
- High floors (30+): +5-15% premium
- Penthouse/top floor: +20-30% premium

**Blocker:** Need to verify Transaction→Unit join quality before implementing.

#### View Analysis (❌ NOT FEASIBLE WITH CURRENT DATA)

**Data availability:** ❌ No orientation/facing data exists

| Required Data | Status |
|---------------|--------|
| Unit facing/orientation | ❌ Not in any file |
| Unit-level coordinates | ❌ Not available |
| Building orientation | ❌ Not available |

**Why it cannot be done properly:**
- `Units.csv` has no "facing" or "orientation" column
- Same building has units facing all directions (sea vs city)
- Without knowing which direction a unit faces, we cannot determine its view
- Proximity-based scoring would average across all orientations (noisy signal)

**What we'd need for true view analysis:**
- Manual labeling of unit orientations, OR
- Listing data with view descriptions scraped, OR
- Building floorplans with unit positions

**Recommendation:** Park view analysis until orientation data is available. Floor height captures some of the view signal (higher = better views regardless of direction).

---

## 9. Deliverables Requested

### Priority 0: Pipeline Verification (MOST IMPORTANT)
**Before anything else, we need independent verification that the AI-generated pipeline is correct:**
- Spot-check raw data → cleaned data transformations
- Verify join logic (are records being lost or duplicated?)
- Validate entity resolution (especially developer mapping)
- Confirm aggregation produces sensible outputs
- Check for data leakage or target contamination

### Priority 1: Data Structure Audit
- Is current aggregation (group_id) correct?
- Should developer be in group_id or as a feature?
- What's the right granularity level?

### Priority 2: Feature Engineering Review
- Which features are actually predictive?
- Are there data leakage issues?
- What features are missing?

### Priority 3: Developer Modeling Strategy
- How to handle entity resolution (duplicates, SPVs)?
- How to capture developer premium/reputation?
- How to handle unmapped developers?

### Priority 4: Cold Start Strategy
- What to do with thin segments (<24 months)?
- Hierarchical fallback approach?
- Explicit uncertainty quantification?

### Priority 5: Evaluation Framework
- What metrics are appropriate for real estate forecasting?
- How to evaluate across different market conditions?
- How to detect when forecasts are unreliable?

### Priority 6: Model Architecture Assessment
- Is TFT the right choice for this problem?
- Alternative approaches to consider?
- Ensemble strategies?

### Priority 7: Trained Model Evaluation
- Compare Console Model vs Colab V3 Model on holdout data
- Which feature set performs better?
- Is the simpler model (Console) or richer model (Colab V3) preferable?
- Are the 34 features in Colab V3 adding signal or noise?

---

## 10. Sample Data

### Sample Training Row

```
year_month: 2023-06
area_id: 330
area_name: Marsa Dubai
property_type: Unit
bedroom: 2BR
reg_type: Ready
developer_brand_label: Emaar
developer_umbrella: Emaar
developer_id: 555
median_price: 1850.00
median_rent: 125000.00
transaction_count: 47
rent_count: 23
units_completing: 0
active_projects: 3
supply_units: 245
months_since_launch: 180.5
months_to_handover: 0.0
project_percent_complete: 100.0
centroid_lon: 55.1234
centroid_lat: 25.0789
all_monthly_index: 1.542
month: 6
quarter: 2
month_sin: 0.866
month_cos: -0.5
time_idx: 89
group_id: 330_Unit_2BR_Ready
```

---

## 11. File Locations Summary

### Raw Data
```
Data/raw_data/
├── Transactions.csv        (~500K rows)
├── Projects.csv            (~3K rows)
├── Buildings.csv           (~240K rows)
├── Rent_Contracts.csv      (~100K rows)
├── Valuation.csv           (~87K rows)
├── Units.csv
└── new_raw_data/
    ├── Developers.csv      (~2.1K rows)
    ├── Lkp_Areas.csv       (~300 rows)
    ├── Residential_Sale_Index.csv
    └── Community.kml
```

### Cleaned Data
```
Data/cleaned/
├── Transactions_Cleaned.csv
├── Projects_Cleaned.csv
├── Buildings_Cleaned.csv
├── Rent_Contracts_Cleaned.csv
└── cleaning_stats.json
```

### Lookup Tables
```
Data/lookups/
├── developer_brand_consolidation.json
├── umbrella_map.json
├── public_brands.json
├── area_mapping.json
├── top50_developers_2025.json
└── [various override files]
```

### Training Data
```
Data/tft/
├── latest/
│   └── tft_training_data_v2_imputed.csv
└── runs/
    └── <BUILD_ID>/
        ├── tft_training_data_v2.csv
        ├── build_stats_v2.json
        └── run_manifest.json
```

### Scripts
```
scripts/
├── clean_all_data.py
├── generate_lookup_tables.py
├── build_tft_data_v2.py
├── brand_resolver.py
└── [various analysis scripts]
```

### Trained Models
```
backend/models/
├── Console_model/
│   └── output_tft_best_v2.ckpt    (5.5 MB, 250K steps, 11 features)
└── tft_final_v3.ckpt              (9.8 MB, 60K steps, 34 features)
```

---

## 12. Contact & Next Steps

### What We're Asking For

This is **not** a request to "tune hyperparameters" or "improve model performance."

This is a request to **verify that the foundations are sound** before we invest more time building on potentially broken logic.

**Specifically:**

1. **Verify the data pipeline** — Independently confirm that the AI-generated cleaning, joining, and aggregation logic is correct. We need someone to look at the actual code and spot-check the outputs.

2. **Validate business logic** — Confirm that the choices made (how to aggregate, what features to include, how to handle missing data) make sense for real estate forecasting.

3. **Identify blind spots** — What are we missing? What assumptions have we made that a domain expert would question?

4. **Recommend architecture** — Given the data we have, is TFT the right approach? Should we be doing something simpler or different?

### Access Provided

We will provide:
- Full codebase access (scripts, lookups, configs)
- Raw data files
- Cleaned data files
- Final training data
- Trained model checkpoint
- Any documentation needed

### Outcome Expected

A report covering:
- ✅ / ❌ for each pipeline stage (is it correct or not)
- List of bugs or logic errors found
- Recommendations for fixes
- Assessment of whether the current approach is viable

---

*Document version: 1.1*  
*Last updated: 2025-12-21*  
*Note: This pipeline was built with AI assistance and requires human expert verification.*  
*Two trained models exist in the codebase — neither should be deployed without addressing the issues documented above.*

