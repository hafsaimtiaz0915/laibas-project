## Retraining (Investor Lens) — What changed, why, and what the model is being retrained to learn

This document is the **single source of truth** for:
- **What we changed** in the training data + ETL.
- **Why we changed it** (in plain “investor/agent value” terms).
- The **lens we are retraining under**: helping agents sell off-plan to investors by forecasting value and yield **at/after handover**, not just “off-plan momentum”.

---

## 0) Current state (what has ACTUALLY been trained + what is deployed)

This repo currently contains:

### 0.1 Reality check: production is still on legacy artifacts (for now)

- **Currently deployed backend checkpoint (legacy)**: `backend/models/tft_final__patched.ckpt` (built 2025-12-15)
- **Currently deployed backend training CSV (legacy, used for group matching at inference time)**: `backend/models/tft_training_data.csv` (built 2025-12-16)

### 0.2 Canonical direction: V2 is the training truth going forward

V2 is the dataset we are investing in and the only one we should continue cleaning/auditing.

Training must move to V2. Anything V1-related should be treated as **temporary compatibility glue** until inference is migrated.

Important context for another ML agent:

- The **currently deployed model** (the `.ckpt` in `backend/models/`) was trained using the **V1-style training CSV schema**.
- The repo now also has a **V2 training-data builder** (`scripts/build_tft_data_v2.py`) which outputs a **run bundle** in `Data/tft/runs/<BUILD_ID>/`.
  - V2 now includes a **future supply schedule** feature: `units_completing` (monthly scheduled completions from Projects’ handover schedule).
  - Training can and should be done on V2 using `Docs/Data_docs/training_colab_v2.md`.

This doc explains both:
- what has been trained (so you can reproduce/continue correctly), and
- what the new V2 ETL/audit contract is (so you can migrate safely).

---

## 1) What investors (and agents) actually need from this report

The report is sold to agents selling **off-plan** units to investors. The investor is asking:
- **What will this unit likely be worth once it’s built?**
- **What yield will it likely generate once it’s rentable?**
- **What happens to pricing across the lifecycle**:
  - early off-plan
  - later off-plan (as completion approaches)
  - **handover**
  - post-handover (ready market dynamics)
- **Developer execution risk**: “Do they deliver on time?” and “How long do they usually take?”
- **Area risk/return context**:
  - pricing level + trend
  - liquidity/transactions (off-plan + ready, not just one slice)
  - future supply and scheduled completions (pipeline pressure)

So the retraining goal is: **learn price/rent dynamics as a function of lifecycle stage, supply, and market context**, with clean developer/area identity.

---

## 2) Core retraining lens (how we’re structuring the learning problem)

### 2.1 Lifecycle-aware learning (off-plan → handover → ready)
We explicitly model the journey by adding lifecycle/timeline features and ensuring transaction stage labels are consistent with the timeline.

### 2.2 No “Unknown developer” buckets
Unknown developers create noise and destroy investor trust. Developer identity must be deterministic in the training data (or be placed into an explicit auditable bucket, not “Unknown”).

### 2.3 Supply is not a footnote — it’s a driver
Future supply and scheduled completions are treated as **core time-varying inputs** so the model can learn supply pressure patterns.

### 2.4 Explainability is a product requirement
We need credible “why” outputs (feature-driven, non-causal) so agents can sell the story and investors can sanity-check it.

---

## 3) What changed in ETL / training data (and why)

We now have **two ETL “generations”** in this repo:

- **V1 (what the current trained model uses)**:
  - Builder: `scripts/build_tft_data.py`
  - Typical export used in Colab: `tft_training_data_1.csv` (Google Drive)
  - Backend mirror used for inference group matching: `backend/models/tft_training_data.csv`
  - Rich feature set including lifecycle + supply + macro series (this matches the assumptions in `Docs/Data_docs/training_opus_colab.md`).

- **V2 (new run-bundled identity + audit contract)**:
  - Builder: `scripts/build_tft_data_v2.py`
  - Output is **not a single flat file**; it is a **versioned run bundle**:
    - `Data/tft/latest/latest_build_id.txt` contains the latest successful `BUILD_ID`
    - `RUN_DIR = Data/tft/runs/<BUILD_ID>/` contains auditable artifacts (training CSV + stats + delta + gates + owner-aware reports)

If you are retraining right now and want maximum compatibility with current inference code, you must train on the **V1 schema** until the V2 migration work is completed.

### 3.1 Developer identity cleanup (registered entity vs brand)
**What changed**
- Introduced/standardized:
  - `developer_registered_name`: DLD-registered/master developer entity (Arabic) derived from Projects linkage.
  - `developer_brand`: optional user-facing brand override (e.g., Binghatti/Danube/Ellington) inferred deterministically from transaction text fields (project/building names) and gated rules.
  - `developer_name`: the **series key** used in `group_id` (brand if available, else registered name).
- Added explicit brand consolidation via `Data/lookups/developer_brand_consolidation.json` (e.g., Damac, Emaar) so fragmented legal entities don’t split the brand signal across multiple series.

**Why it matters for investors**
- Investors think in brands (“Emaar”, “Damac”), not fragmented DLD entities.
- Brand fragmentation produces “random” trend outputs because the model sees multiple weak series instead of one strong brand signal.

### 3.2 “No Unknown developers” (fixed at the source)
**What changed**
- Removed naive `fillna("Unknown")` behavior.
- Implemented deterministic developer resolution:
  1. `project_number` → developer via `Projects` linkage (highest quality).
  2. For Ready rows missing `project_number`: `master_project_en` → `master_developer_name` fallback.
  3. Final fallback bucket: `ALL_DEVELOPERS` (explicit market bucket; auditable; not “Unknown”).

**Why it matters for investors**
- “Unknown developer” invalidates trust and breaks developer performance sections.
- The fallback strategy preserves the ability to compute developer/area context while keeping unmapped cases explicit and measurable.

### 3.3 Fixing handover/timeline + stage labeling
**Problem observed**
- Many rows were labeled “OffPlan” even when their transaction date was after the project’s handover date (a DLD labeling/data artifact).

**What changed**
- Defined `handover_date = coalesce(completion_date, project_end_date)`.
- Added lifecycle features:
  - `months_to_handover_signed` (positive before handover, negative after)
  - `months_since_handover`
  - `handover_window_6m` (within ±6 months window)
- Introduced:
  - `reg_type_dld`: raw DLD label preserved.
  - `reg_type`: lifecycle-corrected stage derived from `transaction_date` vs `handover_date` (Ready if post-handover).
  - `dld_offplan_after_handover`: explicit flag for the anomaly.

**Why it matters for investors**
- The investor journey depends on handover. If the model can’t see the stage transition cleanly, it can’t learn the “handover bump” dynamics and will produce incoherent trends.

### 3.4 Supply schedule corrections (avoid inflating completions)
**What changed**
- `units_completing`/supply schedule logic excludes “DLD OffPlan after handover” rows so supply isn’t inflated by mislabeled late-stage transactions.

**Why it matters for investors**
- Overstated “units completing” makes the report claim supply pressure that isn’t real, which misleads investment decisions.

### 3.5 Outlier + unit consistency fixes (price and rent)
**What changed**
- Enforced sanity bounds for price per sqft used to compute medians: `100 <= AED/sqft <= 50,000`.
- Corrected a rent scaling issue (100x) by applying a deterministic fix when `rent_per_sqft` was implausibly high.

**Why it matters for investors**
- Outliers create “random” forecasts and fake volatility.
- Rent scaling errors destroy yield credibility.

### 3.6 Missingness handling (Option A — missingness-aware features)
**What changed**
- Stopped treating missing numeric inputs as literal zeros at the data-source level.
- Implemented missingness-aware strategy:
  - Macro/tourism series: fill internal gaps via ffill/bfill, avoid extrapolating beyond coverage; add flags like `eibor_missing`, `tourism_missing`.
  - Sparse structural features: area median → global median → 0, with flags like `govt_valuation_median_missing`, `avg_building_floors_missing`, `avg_building_flats_missing`.

**Why it matters for investors**
- “0 tourism” or “0 EIBOR” is not a real economic state; it’s “unknown”. The model must learn the difference.

---

## 4) Training configuration changes (Colab notebook)

Training is documented in `Docs/Data_docs/training_opus_colab.md`.

### 4.0 V2 training is the default path

Use **`Docs/Data_docs/training_colab_v2.md`** to train on **`tft_training_data_v2.csv`**.

V2 training has one non-negotiable product requirement:
- **`units_completing` must be present** (future scheduled completions). If it is missing, the V2 Colab script must fail fast.

### 4.0 The training artifact you must produce (for backend + reproducibility)

Another ML agent should treat “the training output” as a small **artifact set**, not just “a model file”:

- **Model checkpoint**: a PyTorch Lightning checkpoint (`.ckpt`)
  - Current convention in this repo: `tft_final.ckpt`
  - Deployed backend default: `backend/models/tft_final__patched.ckpt` (a patched variant to avoid Mac/CUDA device issues)
- **Training CSV used for inference matching**: the exact CSV the backend uses to find/validate `group_id`s
  - Current backend path: `backend/models/tft_training_data.csv`
  - This must match the schema the inference service expects.
- **Training manifest (RECOMMENDED)**: a small JSON that captures the run inputs needed to reproduce the checkpoint:
  - dataset build id / date range
  - horizon (`PRED_LEN_MONTHS`) and encoder length
  - feature lists used (static categoricals, known reals, unknown reals)
  - code version/hash (if available)

This repo does not currently have a `.train` file format. If your pipeline requires a “`.train`” artifact, define it explicitly as the **training manifest JSON** (e.g., `tft_training_manifest.train.json`) and store it alongside the checkpoint in the same bucket/folder.

### 4.1 Horizon is now configurable (investor-relevant)
**What changed**
- Removed hardcoded 6-month forecast settings.
- Added a single knob:
  - `PRED_LEN_MONTHS = 12` (default; aligns better with investor decision cycles)
  - optional `24` if we truly want direct 24-month forecasts

**Why it matters**
- Investors rarely care only about 6 months. A 12–24 month view is materially more useful.
- If we train on 12/24, inference must also generate future known-features to the same horizon (supply schedule + seasonality at minimum).

### 4.2 What has actually been trained (current checkpoint training recipe)

The current training approach (see `Docs/Data_docs/training_opus_colab.md`) is:

- **Model family**: Temporal Fusion Transformer (PyTorch Forecasting)
- **Targets**: multi-target regression
  - `median_price` (AED/sqft)
  - `median_rent` (annual rent)
- **Loss**: quantile loss for each target (default quantiles from PyTorch Forecasting)
- **Horizon**: `PRED_LEN_MONTHS = 12` (default)
- **Encoder length**: `ENCODER_LEN_MONTHS = 36` (default)
- **Validation**: last `PRED_LEN_MONTHS` months held out by `time_idx`
- **Output**: best checkpoint copied to Drive as `tft_final.ckpt`, then deployed to backend (often as `tft_final__patched.ckpt`)

Critical note for product correctness:
- If the model is trained with `PRED_LEN_MONTHS=12`, then “24 month” outputs in the backend are **derived** (not a direct learned 24-step forecast) unless you train a 24‑month model and update inference to use it.

### 4.3 “Paste into Colab” reliability
**What changed**
- Notebook instructions were cleaned up to avoid basic “paste error” issues and to make the workflow deterministic.

---

## 5) Quality gates and audits (pre-retrain signoff)

Before retraining, we run deterministic checks to prevent garbage-in:
- `scripts/pre_retrain_gate.py`: one-command PASS/FAIL style report.
- `scripts/audit_developer_segmentation.py`: confirms brand capture and leakage (e.g., Emaar/Damac consolidation, building developer capture rules).

Hard expectations for signoff:
- **0 “Unknown developer” rows** in training data.
- Lifecycle labels consistent with timeline (OffPlan vs Ready based on handover_date).
- Supply schedule not inflated by mislabeled post-handover rows.

### 5.1 V2-specific auditable run bundle (identity + owner-aware project/area handling)

V2 introduces a stricter, auditable contract so identity mistakes do not silently leak into training labels.

To find the latest successful bundle:

- `BUILD_ID` is stored in: `Data/tft/latest/latest_build_id.txt`
- `RUN_DIR` is: `Data/tft/runs/<BUILD_ID>/`

Inside `RUN_DIR`, the auditor contract includes (non-exhaustive):
- `tft_training_data_v2.csv`
- `build_stats_v2.json`
- `delta_report_v2.json`
- `baseline_snapshot_v2.json`
- `holding_policy_audit_v2.json`
- `owner_assertions_report.json` (hard assertions; must PASS)
- `owner_override_self_check.csv` (must have 0 CONFLICT)
- `noncanonical_brand_overrides_skipped.csv` (legacy override soft-validation evidence)
- Gates:
  - `project_area_labels_surviving_gate.csv` (must be header-only)
  - `suspicious_label_gate_offenders.csv` (must be header-only)

Owner-aware intent (high-level):
- suspicious project/area-like legal entities should be remapped to the safe owner brand when possible (Tiered),
- only nuked to `DEVELOPER_ID_<id>` when owner is unknown/unsafe,
- authorities/freezones must remain `DEVELOPER_ID_<id>`.

If you want to train from V2, you must also ensure the **training recipe and inference code** are updated for the V2 schema (see section 10).

---

## 6) What this retrain should improve in the investor-facing report

After retraining on this cleaned, lifecycle-aware dataset, the report should become:
- **Less “random”**: fewer contradictions between trend snippets and forecasts caused by bad joins/outliers/fragmented developer identity.
- **More lifecycle-aware**: the model can learn differences between off-plan dynamics and ready-market dynamics and the handover transition.
- **More credible for agents**:
  - developer execution signals (time to complete / delays) are meaningful inputs and supporting context
  - supply context is grounded and not inflated
- **More explainable**: feature importance / driver summaries reflect real inputs investors care about (supply, lifecycle timing, market context).

---

## 7) Known limitations (transparent, for roadmap)

These are not solved purely by ETL and may require new data:
- **Repeat-sales at unit level**: if we don’t have a stable unit identifier, we can’t cleanly compute true unit repeat sales trajectories.
- **Perfect developer mapping for every ready transaction**: best-effort deterministic linking exists, but “brand inference from names” must remain rules-based and audited.

---

## 8) Practical “what to do next”

This section is written for an ML agent who has no prior context and must produce the correct training outputs.

### 8.1 If you are reproducing / extending the CURRENT deployed model (legacy V1-compatible path)

This is legacy-only. Do not continue investing in V1 unless you are unblocking a hotfix deployment.

1. **Build the V1 training CSV** (or confirm it already exists) using `scripts/build_tft_data.py`.
2. **Run gates**:
   - `python scripts/pre_retrain_gate.py` (writes `Docs/Data_audits/PRE_RETRAIN_GATE_REPORT.md` + `.json`)
   - `python scripts/audit_developer_segmentation.py` (writes `Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.md`)
3. **Upload the training CSV to Drive** as `tft_training_data_1.csv` (this is what `training_opus_colab.md` loads by default).
4. **Train in Colab** using `Docs/Data_docs/training_opus_colab.md`:
   - Keep `PRED_LEN_MONTHS=12` unless you explicitly want/need a 24‑month learned model.
   - Save best checkpoint to Drive as `tft_final.ckpt`.
5. **Deploy artifacts back into repo** (or your model bucket):
   - Copy the checkpoint to `backend/models/tft_final.ckpt` (or `tft_final__patched.ckpt` if applying the Mac/CUDA patching convention).
   - Copy the exact training CSV used for group matching to `backend/models/tft_training_data.csv`.
6. **Smoke test** using the repo’s loader/inference path:
   - Run: `backend/venv/bin/python scripts/smoke_test_tft_checkpoint.py`

### 8.2 V2 training + migration (canonical path)

1. **Build a new V2 run bundle**:
   - `python scripts/build_tft_data_v2.py`
   - This writes to `Data/tft/runs/<BUILD_ID>/` and updates `Data/tft/latest/latest_build_id.txt` on success.
2. **Audit the latest successful run bundle** using the artifacts in `RUN_DIR` (owner assertions, gate CSVs, holding policy audit, legacy override skip report).
3. **Train on V2** using `Docs/Data_docs/training_colab_v2.md`:
   - Upload `RUN_DIR/tft_training_data_v2.csv` to Drive as `tft_training_data_v2.csv`.
   - Train with `PRED_LEN_MONTHS=12` (or 24 if you truly want a learned 24‑month model).
4. **Deploy + align inference**:
   - The backend must load the V2-trained checkpoint and a V2-compatible group lookup CSV.
   - If the backend inference still expects legacy group_id format or legacy feature lists, update it before switching the model.
5. Once schema alignment is complete:
   - upload `RUN_DIR/tft_training_data_v2.csv` (or a derived training export) to Drive,
   - train, then deploy the checkpoint + matching CSV + manifest back into backend/bucket,
   - run the same smoke test.

---

## 9) “No context” glossary for an ML agent (what is what)

- **`group_id`**: the time-series identity key the TFT trains on. It represents the “segment” the investor report forecasts for (area × property type × bedroom × lifecycle stage, optionally developer depending on dataset generation).
- **`time_idx`**: integer month index (monotonic) used by the TimeSeriesDataSet.
- **`median_price`**: target price series (AED/sqft).
- **`median_rent`**: target rent series (annual AED); used for yield context.
- **`PRED_LEN_MONTHS`**: forecast horizon the model is trained to predict (must match inference behavior for “direct” forecasts).
- **Run bundle (`Data/tft/runs/<BUILD_ID>/`)**: an auditable set of build outputs; the training CSV is only one artifact inside it.

---

## 10) Known mismatch to resolve (so you don’t build the wrong code)

Right now:
- **Training recipe (`training_opus_colab.md`) expects V1-style features** (notably `units_completing` and several macro/structural features).
- **V2 training CSV (`tft_training_data_v2.csv`) currently has a different, smaller schema** (57 columns, includes identity/audit fields like `developer_id`, `developer_brand_label`, `developer_umbrella`, top‑50 reporting fields, and sale index fields, but not the full V1 macro/supply feature set).

So: if you are “the ML agent in charge of building the training `.train` file”, your first decision is:
- **Do we retrain with the existing V1 feature set (fastest path, matches deployed inference),** or
- **Do we migrate the model to V2 schema (requires coordinated training + inference changes)?**

Do not proceed with training until this decision is explicit, because it changes:
- the dataset schema,
- the model’s feature lists (known vs unknown),
- and the inference-time feature generation contract.

---

## 11) V2 inference requirements (so a V2-trained model is actually usable)

If you train a new checkpoint on `tft_training_data_v2.csv`, the backend must be migrated to the **V2 group + feature contract**.

Minimum requirements:

- **Group identity alignment**
  - V2 `group_id` is currently built from `area_id` (numeric) + `property_type` + `bedroom` + `reg_type` (and optionally developer if enabled in the V2 builder).
  - The backend currently expects a legacy `group_id` based on **area name** and a trailing developer token.
  - Therefore the backend must either:
    - map user-facing `area_name` → `area_id` deterministically, and match groups by V2 `group_id`, or
    - change the V2 builder’s group_id format (not recommended without a clear plan).

- **Future covariates for the forecast horizon**
  - If `units_completing` is used as a known future real, inference must provide future values for the requested horizon.
  - Using “recent average” is a temporary fallback; investor-grade behavior should use the Projects schedule to generate the future path.

- **Artifact deployment**
  - Deploy the checkpoint **together with** a V2-compatible training CSV used for group matching.
  - Do not mix a V2 checkpoint with a legacy training CSV (or vice versa): the model’s learned embedding space depends on the training categoricals.


