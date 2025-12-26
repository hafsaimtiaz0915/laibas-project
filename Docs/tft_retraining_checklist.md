# TFT Model Retraining Checklist

> ⚠️ **PARTIALLY OUTDATED** (2025-12-21)
> 
> This checklist was written for V1 training. Some sections are still relevant, but:
> - File paths reference V1 (`tft_training_data.csv`) — use V2 (`tft_training_data_v2.csv`)
> - Training script is now `Docs/Data_docs/training_colab_v2.py`
> - Data build script is now `scripts/build_tft_data_v2.py`
> 
> **For current state, see:**
> - `Docs/PROJECT_STATE_FOR_DATA_SCIENTIST.md` — File locations and what to use
> - `Docs/Data_docs/DATA_SCIENCE_AUDIT_BRIEF.md` — Full problem list and context

---

Use this checklist when retraining the TFT model after data updates or feature changes.

## Pre-Training Checklist

### 1. Data Preparation

- [ ] Run `build_tft_data.py` to generate fresh training data
  ```bash
  cd /path/to/Properly
  python scripts/build_tft_data.py
  ```

- [ ] Verify output file exists: `Data/tft/tft_training_data.csv`

- [ ] Check `Data/tft/build_stats.json` for:
  - [ ] Total rows > 100,000 (minimum for reliable training)
  - [ ] Unique groups > 1,000
  - [ ] Unique areas = 78 (or expected number)
  - [ ] Date range covers at least 36 months
  - [ ] **Reg-type coverage**: both `OffPlan` and `Ready` are present with meaningful history (not just a few groups)
  - [ ] **Group definition sanity**: `group_id` is consistent and includes the intended segmentation keys (area × type × bedroom × reg_type × developer where applicable)
  - [ ] **Developer linkage coverage**: project-phase / developer features are present for a large share of rows (not mostly zeros)
  - [ ] **Supply schedule coverage**: `units_completing` is present and non-trivial for major areas (not mostly zeros)

### 1.4 Developer identity + project linkage (must be correct or the model will learn noise)

Investor-facing reports depend on developer + project features being **reliably linked** to the transaction series.

**Goal:** ensure that “developer execution” and “project phase” features used in training are not random artifacts of bad joins.

Required mapping principles:
- [ ] Maintain a single canonical developer identity:
  - **brand name** (English, user-facing)
  - **registered entity** (Arabic, appears in DLD / training)
  - optional **master developer** relationship (building developer cases)
- [ ] **Hard rule: no Unknown developers in training data**
  - [ ] `developer_name` must never be `"Unknown"` in `Data/tft/tft_training_data.csv`
  - [ ] If the raw data does not provide a developer, the pipeline must map to a deterministic fallback:
    - [ ] building developer → brand/master developer mapping (documented in lookups)
    - [ ] otherwise → explicit canonical placeholder like `"UNMAPPED_DEVELOPER__REQUIRES_FIX"` (so it is visible and fails QA)
  - [ ] Retraining must **fail/stop** if any Unknown/unmapped developer rows exist (fix ETL/mapping first)
  - [ ] Document the concrete root cause and fix:
    - [ ] In DLD Transactions, `project_number` is frequently missing for **Ready** transactions (≈37% of Ready rows in the current dataset).
    - [ ] Fix implemented in `scripts/build_tft_data.py`: when `project_number` is missing, resolve developer via `master_project_en → master_developer_name` (from Projects), else use `ALL_DEVELOPERS` bucket (never `Unknown`).

Required linkage checks (run on the built training data and/or build logs):
- [ ] **Unknown/unmapped developer check (hard gate)**:
  - [ ] `developer_name == "Unknown"` rows: **0**
  - [ ] `developer_name` starts with `"UNMAPPED_DEVELOPER__"` rows: **0**
  - [ ] Top 50 developers by transaction volume have stable `developer_name` (no aliases/duplication)
- [ ] **Brand segmentation audit (hard gate for major brands)**:
  - [ ] Run:
    ```bash
    python scripts/audit_developer_segmentation.py
    ```
  - [ ] Review outputs:
    - [ ] `Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.md`
    - [ ] `Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.csv`
  - [ ] Confirm key brands appear as first-class series (non-zero `training_rows`), especially:
    - [ ] Binghatti, Danube, Ellington
    - [ ] Damac is consolidated via `Data/lookups/developer_brand_consolidation.json` (no leakage)
- [ ] **Project-phase feature plausibility** (spot-check distributions are non-zero and sane):
  - [ ] `months_since_launch` not 100% zeros
  - [ ] `months_to_handover` not 100% zeros (for OffPlan regimes)
  - [ ] `project_percent_complete` within [0, 100] or [0, 1] consistently (document scale)
  - [ ] `project_duration_months` sane (e.g. 6–120 months, depending on your definitions)
- [ ] **Join validation**:
  - [ ] Pick 3 known developers + areas (e.g., Emaar/Dubai Hills, Ellington/Dubai Hills, Binghatti/JVC):
    - confirm that project-phase features change over time in ways that match known project timelines (not flat zeros)
    - confirm that developer aggregates (`dev_*` columns) are stable and not missing
  - [ ] Validate fallback behavior on Ready markets:
    - [ ] `ALL_DEVELOPERS` exists as an explicit bucket for rows where developer cannot be deterministically attributed
    - [ ] `ALL_DEVELOPERS` share stays small (recommend: < 5% of rows) and is audited if it grows

If any of these fail: stop retraining and fix the ETL/join logic before training.

### 1.5 Future supply schedule (area-level truth for investors)

Agents/investors need credible “future supply” to understand pipeline pressure and handover risk.

Minimum requirements for the dataset/report:
- [ ] **Active pipeline supply** (units in non-finished projects) per area
- [ ] **Scheduled completions** per area for next 6/12/24 months (or to handover)

Data build requirements:
- [ ] `units_completing` should be derived from Projects/Buildings completion schedules (not hard-coded to 0)
- [ ] Validate `FINISHED` status casing and filtering so pipeline counts are not inflated/deflated
- [ ] For each major area, sanity-check:
  - [ ] next-6m completions are not all zero
  - [ ] pipeline units roughly aligns with known market reality (spot check)

Reporting requirements (so it’s not misleading):
- [ ] Always label supply as **area-wide** unless you can compute segment-specific supply correctly
- [ ] If future supply is incomplete/unknown, report “Not available” rather than inventing a number

### 2. Data Quality Verification

- [ ] Run data quality checks:
  ```python
  import pandas as pd
  df = pd.read_csv('Data/tft/tft_training_data.csv')
  
  # Check for nulls in critical columns
  critical_cols = ['median_price', 'area_name', 'bedroom', 'time_idx', 'group_id']
  for col in critical_cols:
      null_pct = df[col].isna().mean() * 100
      print(f"{col}: {null_pct:.2f}% null")
  
  # Check target column distributions
  print(df['median_price'].describe())
  print(df['median_rent'].describe())
  
  # Check group lengths (need min 12 months per group for TFT)
  group_lengths = df.groupby('group_id')['time_idx'].count()
  print(f"Groups with < 12 months: {(group_lengths < 12).sum()}")
  ```

- [ ] Verify no extreme outliers in prices (< 100 or > 50,000 AED/sqft)

- [ ] Check developer names match between Projects and Transactions
  - [ ] Confirm that developer mapping does not collapse to `"Unknown"` for common brands (e.g., Ellington, Emaar)
  - [ ] Confirm `developer_name` in `tft_training_data.csv` aligns with `Data/lookups/developer_mapping.json` (English→Arabic used at inference)

- [ ] **Area mapping sanity**
  - [ ] Confirm `area_name` values are canonical DLD area names that exist in `Data/lookups/area_mapping.json:all_areas`
  - [ ] Spot-check common marketing names resolve to the correct DLD areas (see `Docs/frontend/LLM/AREA_MAPPING_COVERAGE.md`)

- [ ] **Reg-type + investor relevance sanity**
  - [ ] Ensure both **OffPlan** and **Ready** groups exist for major areas (Dubai Hills, Marina, Downtown, Business Bay, JVC)
  - [ ] Ensure rent targets are present for **Ready** groups (or explicitly use `has_actual_rent` / benchmarks when missing)

- [ ] **Base-effect / launch spike risk checks (transaction volume)**
  - [ ] Identify groups where `transaction_count` has extreme spikes (likely launch months) that can distort YoY volume comparisons.
  - [ ] Add/verify a flag in the training data build (recommended): `launch_spike_flag` and/or rolling baselines (12m mean, 24m median) to contextualize volume.
  - [ ] If this is not implemented yet, document that “YoY 3m volume” can be misleading without base-effect detection.

- [ ] **Feature completeness + inference parity**
  - [ ] Ensure every numeric feature used in training has no NaNs (fill or impute)
  - [ ] Ensure inference code can supply required features (even if via safe defaults)
  - [ ] If adding new features (e.g., `median_unit_sqft`), ensure both training and inference are updated
    - Reference: `Docs/phase2_add_unit_sqft_feature.md`

### 3. Upload to Colab

- [ ] Upload `tft_training_data.csv` to Google Drive

- [ ] If using new features, update the notebook:
  - [ ] Add to `time_varying_unknown_reals` or appropriate list
  - [ ] Update feature documentation
  - [ ] Ensure `has_actual_rent` is created consistently (see `Docs/Data_docs/training_opus_colab.md`)
  - [ ] Re-check categorical dtypes (`month`, `quarter`, `area_name`, `developer_name`, `group_id`) are `str` before dataset creation

## Training Configuration

### 4. TimeSeriesDataSet Configuration

Verify these settings match your data:

```python
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=["median_price", "median_rent"],  # Multi-target
    group_ids=["group_id"],
    min_encoder_length=12,
    max_encoder_length=96,
    min_prediction_length=1,
    max_prediction_length=6,
    
    static_categoricals=["area_name", "property_type", "bedroom", "reg_type", "developer_name"],
    
    time_varying_known_categoricals=[],
    time_varying_known_reals=["month_sin", "month_cos", "units_completing"],
    
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "median_price",
        "median_rent",
        "transaction_count",
        "rent_count",
        # Add new features here
    ],
    
    target_normalizer=MultiNormalizer([
        GroupNormalizer(groups=["group_id"]),
        GroupNormalizer(groups=["group_id"]),
    ]),
    
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
```

**IMPORTANT (Investor product requirements):**
- If you need **12/24-month** forecasts, you must train with `max_prediction_length >= 12/24`.
  - Otherwise 12/24 month outputs are **extrapolations**, not a learned forecast.
- Off-plan investors care about:
  - **Now → handover (OffPlan dynamics)**
  - **Post-handover behaviour (Ready resale + Ready rent/yield)**

Recommended training approaches (choose one and document which you used):
- [ ] **Approach A (recommended): Train two TFT checkpoints**
  - OffPlan model: train on `reg_type == "OffPlan"`
  - Ready model: train on `reg_type == "Ready"`
  - Use both in the report: OffPlan for construction journey; Ready for post-handover resale + yield context.
- [ ] **Approach B: One model but explicit handover-transition features**
  - Requires engineered features like `months_from_handover`, `is_handover_month`
  - Only feasible if completion timelines are reliably available per group.

Feature classification note (avoid misleading long-horizon learning):
- [ ] Decide which variables are truly **known in the future** vs **unknown**.
  - If using macro/supply inputs (EIBOR, scheduled completions, pipeline), either:
    - Provide **scenario paths** for future months (known reals), OR
    - Keep them unknown and do not over-interpret them as “forward drivers”.
  - Reference implementation patterns: `Docs/Data_docs/training_opus_colab.md`

---

## 4.5 Handover transition (OffPlan → Ready) — make the model “aware” of the journey

**Investor reality:** Off-plan buyers care about the *journey*:
- price path during construction (OffPlan market)
- the **handover transition** (regime shift / potential “bump”)
- post-handover resale + yield (Ready market)

To model this properly, you must choose and document one of these approaches:

### Approach A (recommended): Two-model + explicit transition calibration
Train **two TFT checkpoints**:
- **OffPlan model** (reg_type=OffPlan): learns the construction-period dynamics (launch absorption, supply schedule effects, etc.)
- **Ready model** (reg_type=Ready): learns post-handover resale + rent/yield context

Then add a **handover transition calibration** computed from historical data:
- For each (area × type × bedroom) and optionally developer:
  - estimate the historical relationship between OffPlan series and Ready series around handover (e.g., median ratio or spread).
- Report output should show:
  - “Now → handover” path (OffPlan model)
  - “At handover” expected level (calibrated bridge)
  - “Post-handover” path (Ready model)

Checklist requirements:
- [ ] Both checkpoints trained & validated independently (OffPlan vs Ready)
- [ ] “Bridge” calibration documented and versioned (method + windows + sample sizes)
- [ ] Reports must clearly label which parts are OffPlan vs Ready vs calibrated transition

### Approach B: Single-model with explicit lifecycle state (harder, but fully learned)
Build training data so the model can learn the **regime switch** inside one sequence:
- Add lifecycle features such as:
  - `months_from_handover` (negative before, 0 at handover month, positive after)
  - `is_handover_month` (binary)
  - `post_handover_flag` (binary)
- Keep `reg_type` as a categorical indicator (OffPlan/Ready), but ensure the series can see a transition signal.

This requires reliable linking of handover dates into the time series. If you cannot build `months_from_handover` reliably, do **not** use this approach.

Checklist requirements:
- [ ] `months_from_handover` computed and spot-checked for major areas/projects
- [ ] `is_handover_month` distribution looks sensible (not all zeros)
- [ ] Validation includes sequences that span a handover transition (where data exists)

### Minimum standard (do not skip)
Regardless of approach:
- [ ] Report must show **both**:
  - OffPlan market dynamics (construction period)
  - Ready market dynamics (post-handover performance + rent/yield)
- [ ] Never present OffPlan-only volume trends as “area liquidity” — always provide both OffPlan and Ready volume context

### 5. Model Configuration

Recommended hyperparameters:

```python
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,  # 7 quantiles
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4,
)
```

## Training Process

### 6. Train Model

- [ ] Start training with early stopping:
  ```python
  trainer = pl.Trainer(
      max_epochs=100,
      gpus=1,
      gradient_clip_val=0.1,
      callbacks=[
          EarlyStopping(monitor="val_loss", patience=10),
          LearningRateMonitor(),
      ],
  )
  trainer.fit(tft, train_dataloader, val_dataloader)
  ```

- [ ] Monitor training metrics:
  - [ ] Training loss decreasing
  - [ ] Validation loss decreasing
  - [ ] No signs of overfitting (val loss diverging from train)
  - [ ] For longer horizons (12/24), ensure validation window matches the horizon (do not validate 24-month forecasts on a 6-month window)

### 7. Validate Model

- [ ] Check validation metrics:
  ```python
  best_model = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
  predictions, x = best_model.predict(val_dataloader, return_x=True)
  
  # Calculate MAE, RMSE, etc.
  ```

- [ ] Spot-check predictions for key areas:
  - [ ] Dubai Marina (Marsa Dubai)
  - [ ] Downtown Dubai (Burj Khalifa)
  - [ ] JVC (Al Barsha South Fourth)
  - [ ] Business Bay

- [ ] Verify predictions are reasonable:
  - [ ] Prices within 500-10,000 AED/sqft range
  - [ ] Rents within 20,000-500,000 AED/year range
  - [ ] Confidence intervals not too wide
  - [ ] Model does not produce pathological “flatline” forecasts across all groups (indicates feature issues)

- [ ] **OffPlan vs Ready validation (investor lens)**
  - [ ] Pick one area+bedroom (e.g., Dubai Hills 2BR) and compare:
    - OffPlan series forecast (construction journey)
    - Ready series forecast (post-handover behaviour)
  - [ ] Confirm yields shown to investors are sourced from **Ready rent benchmarks** (not OffPlan pseudo-rent)

## Post-Training Checklist

### 8. Export Model

- [ ] Save best checkpoint:
  ```python
  best_model.save("tft_model_v{version}.ckpt")
  ```

- [ ] Download checkpoint from Colab

### 9. Deploy to Backend

- [ ] Copy checkpoint to `backend/models/tft_model.ckpt`
  - [ ] If using two-model approach: store both (example)
    - `backend/models/tft_offplan.ckpt`
    - `backend/models/tft_ready.ckpt`
  - [ ] Document which file(s) are active and how the report selects them.

- [ ] Update `backend/app/core/config.py` if path changed:
  ```python
  tft_model_path: str = "models/tft_model.ckpt"
  ```

- [ ] Test locally:
  ```bash
  cd backend
  python -c "from app.services.tft_inference import get_tft_service; s = get_tft_service(); print(s.is_loaded)"
  ```

### 10. Verify Predictions

- [ ] Run test predictions for key areas:
  ```bash
  # Start server
  uvicorn app.main:app --reload
  
  # Test endpoints
  curl -X POST "http://localhost:8000/api/analyze" \
    -H "Content-Type: application/json" \
    -d '{"area": "Dubai Marina", "bedroom": "2BR"}'
  ```

- [ ] Compare predictions before/after retraining

- [ ] Document any significant changes
  - [ ] Include: mapping changes, group_id definition changes, horizon changes (6→24), and any feature reclassification (known vs unknown)
  - [ ] Link to relevant docs updated during the change:
    - `Docs/Data_docs/training_opus_colab.md`
    - `Docs/phase2_add_unit_sqft_feature.md`
    - `Docs/frontend/LLM/PIPELINE_AUDIT_AND_GATING.md`

## Rollback Plan

If model performance is worse:

1. Keep previous checkpoint as `tft_model_backup.ckpt`
2. Revert to previous checkpoint:
   ```bash
   cp models/tft_model_backup.ckpt models/tft_model.ckpt
   ```
3. Restart server

## Version History

| Version | Date | Changes | Validation MAE |
|---------|------|---------|----------------|
| v1.0 | YYYY-MM-DD | Initial model | X.XX |
| v1.1 | YYYY-MM-DD | Added unit_sqft feature | X.XX |

---

## Quick Reference: Key File Paths

| File | Path |
|------|------|
| Training data script | `scripts/build_tft_data.py` |
| Training data output | `Data/tft/tft_training_data.csv` |
| Build stats | `Data/tft/build_stats.json` |
| Area mapping | `Data/lookups/area_mapping.json` |
| Developer mapping | `Data/lookups/developer_mapping.json` |
| Model checkpoint | `backend/models/tft_model.ckpt` |
| Inference service | `backend/app/services/tft_inference.py` |
| Entity validator | `backend/app/services/entity_validator.py` |

---

## Notes for agent/investor report correctness (must stay true as data/model changes)
- Segment trend metrics must always state **what they are for**:
  - area (DLD) + property_type + bedroom + reg_type (+ developer when applicable)
- Volume metrics must avoid base-effect traps:
  - include rolling baselines and/or launch spike flags before presenting YoY comparisons as “market health”
- Off-plan investor view requires both:
  - **OffPlan dynamics (now→handover)** and
  - **Ready market + yield context (post-handover)**

