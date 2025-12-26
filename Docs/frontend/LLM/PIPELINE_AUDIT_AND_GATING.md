## Pipeline audit + gating reference (runtime truth)

### Purpose
This document is the reference for:
- **What data sources are used at runtime** (vs documentation-only reference CSVs)
- **How area/developer/rent lookups are matched** and what confidence thresholds gate them
- **How model forecasts are gated** so we don’t show wrong-series predictions
- **What remains intentionally “Not available”** until we compute it correctly (e.g., developer delays, future supply to handover)

---

## 1) Runtime data sources (what the app actually uses)

### 1.1 Entity resolution inputs (canonicalization)
Used by `backend/app/services/entity_validator.py`:
- `Data/lookups/area_mapping.json`
  - `abbreviations`: user inputs (e.g., "Creek Harbour") → **DLD area** (e.g., `Al Khairan First`)
  - `dld_to_common`: DLD area → display name (e.g., `Al Khairan First` → "Dubai Creek Harbour")
  - `all_areas`: allowed DLD areas
- `Data/lookups/developer_mapping.json`
  - English developer names + aliases → Arabic developer name used in training/group_id

### 1.2 Lookup tables (market/context)
Used by `backend/app/services/trend_lookup.py`:
- `Data/lookups/area_stats.csv` (area-wide price + supply pipeline)
- `Data/lookups/developer_stats.csv` (developer aggregates; delay currently not computed)
- `Data/lookups/rent_benchmarks.csv` (area + bedroom rent benchmark)

### 1.3 Model time series (TFT)
Used by `backend/app/services/tft_inference.py`:
- `Data/tft/tft_training_data.csv` (group histories + supply schedule + market context)
- TFT checkpoint (if present) for inference

### 1.4 Output contract (LLM formatting)
Used by `backend/app/services/response_generator.py`:
- `Docs/frontend/LLM/OUTPUT_CONTRACT.md` is loaded into the system prompt to enforce standardized structure.

---

## 2) Reference CSVs (documentation-only)
These files exist to help validate mappings but are **not used at runtime**:
- `Data/lookups/area_reference.csv`
- `Data/lookups/developer_reference.csv`

If you want these to become the source of truth, we should generate `area_mapping.json` / `developer_mapping.json` from them (future enhancement).

---

## 3) Confidence thresholds (fail-closed gating)

Configured in backend code:
- **Area threshold**: `area_confidence >= 80`
- **Developer threshold**: `developer_confidence >= 70`
- **Model forecast threshold**: `prediction.confidence >= 70` AND `match_type ∈ {exact, exact_unknown_dev, partial_area_bedroom, model}`

Where applied:
- `backend/app/services/chat_service.py`: gates which lookups are fetched.
- `backend/app/routes/chat.py` (`/api/chat/test`): same gating for debug route.
- `backend/app/services/response_generator.py`: gates whether model price/rent forecasts are shown.

Why this exists:
- Prevents “wrong area/dev” stats from being shown as factual.
- Prevents the model forecast from being shown when the matched series is not a strong segment match.

---

## 4) Matching + audit (how we prove correctness)

### 4.1 Entity resolution audit fields
`ValidatedEntities` includes:
- `area_resolution_method`: `abbrev|exact|fuzzy|fallback`
- `developer_resolution_method`: `exact|alias|fuzzy|unknown|building_dev`

### 4.2 Lookup audit fields
`trend_lookup.get_all_trends()` returns `lookup_audit`:
- `lookup_audit.area`: `{ input, matched, score, method }`
- `lookup_audit.developer`: `{ input, mapped_to_arabic, matched, score, method }`
- `lookup_audit.rent`: `{ area_input, matched_area, area_score, area_method, bedroom_input, matched_bedroom }`

These are attached to:
- `report_data.lookup_audit`
- `/api/chat/test` under `trends.lookup_audit`

This allows immediate diagnosis of issues like:
- "Creek Harbour" resolving to the wrong DLD area
- English developer not mapping to Arabic developer name
- rent benchmark missing due to bedroom normalization mismatch

---

## 5) Critical correctness fixes already applied

### 5.1 Area mapping correctness
Marketing names must map to **DLD area names** used in `all_areas`.
Example:
- "Dubai Creek Harbour" / "Creek Harbour" / "Creek" → `Al Khairan First`

### 5.2 Supply pipeline correctness
`FINISHED` status casing must match cleaned data. Incorrect casing inflates pipeline counts.

### 5.3 Model matching correctness (group_id parsing)
A core bug was fixed:
- `parse_group_id()` originally used `rsplit('_', 4)`, which breaks when the developer name contains underscores (common in Arabic).
- Fixed by splitting on `_OffPlan_` / `_Ready_` first.

Impact:
- Correct series matching for segment queries (e.g., Creek Harbour 1BR OffPlan) improves from `area_only` → `partial_area_bedroom`.

---

## 6) Off-plan investor relevance: what matters and time horizons

### 6.1 Primary horizon
For off-plan, the primary horizon is:
- **Now → handover** (months), using `handover_months` if given.

Heuristic added:
- If user mentions a year/month (e.g., "handover in 2026"), infer `handover_months` if it wasn’t extracted.

### 6.2 What to show (factual)
- **Price per sqft today** (segment if match is strong; area-wide as secondary)
- **Model forecast to handover** (median + P10/P90) only when match quality passes gating
- **Implied appreciation to handover** (only from the same matched series)
- **Rent benchmark** (area+bedroom) for current rent context
- **Yield** (rent/price) if price is provided
- **Supply pipeline (active projects)** + absorption-based years of inventory (area-wide currently)

### 6.3 What is NOT yet correctly supported
These should remain "Not available" until implemented properly:
- **Future estimated supply to handover** (next 6/12/24 months) computed from schedule
- **Developer delay metrics** (planned vs actual completion dates)

---

## 7) Advice prohibition
Even if a user asks "Should I buy?", the system must:
- Provide factual metrics only
- Include: **"This report is factual and is not investment advice."**

---

## 8) Known operational risks

### 8.1 Lookup generation can overwrite manual mapping improvements
`scripts/generate_lookup_tables.py` regenerates `area_mapping.json` / `developer_mapping.json` templates.
If run without preserving manual overrides, mapping quality can regress.

Recommended hardening:
- Make reference CSVs the source of truth OR
- Maintain a separate `*_overrides.json` merged after generation.

---

## 9) Quick verification checklist
Run `/api/chat/test` and check:
- `entities.area_name` and `entities.area_confidence` (>=80)
- `entities.developer_arabic` and `entities.developer_confidence` (>=70)
- `report_data.lookup_audit` shows the exact matched keys
- `prediction.match_type` and `prediction.confidence` (>=70 to show model forecast)
- Output contains the disclaimer and no buy/sell/hold recommendations
