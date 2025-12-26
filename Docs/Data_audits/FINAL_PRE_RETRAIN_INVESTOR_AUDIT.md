# Final Pre-Retrain Investor-Lens Audit (Training Data)

This is the final “investor lens” quality review for retraining the TFT model, focused on:
- **Developer identity correctness** (no brands lost, no duplicates that break segmentation)
- **Area identity correctness** (canonical DLD areas)
- **Lifecycle correctness** (OffPlan journey vs post-handover Ready behavior)
- **Supply + handover correctness** (handover date source is explicit, and supply schedule is not inflated by post-handover events)
- **Outlier sanity** (prices/rents within plausible investor ranges)

## Snapshot (current `Data/tft/tft_training_data.csv`)
- **Rows**: 72,205
- **Unique groups**: 1,745
- **Unique areas**: 78
- **Date range**: 2003-06 → 2025-12

## 1) Hard gates (must pass)
- **Unknown developers**: ✅ 0 rows (`developer_name == "Unknown"`)
- **Unmapped developer bucket**: ✅ 0 rows (`developer_name` starts with `UNMAPPED_DEVELOPER__`)
- **ALL_DEVELOPERS bucket**: ✅ 1,791 rows (~2.48%) — small and auditable

## 2) Developer identity (brands + consolidation)
Major brands are first-class series in training data and are not silently swallowed by master developers:
- Binghatti / Danube / Ellington (brand series via `developer_brand`)
- Damac / Emaar (brand consolidation via `Data/lookups/developer_brand_consolidation.json`)

Reference audit:
- `Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.md`
- `Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.csv`

## 3) OffPlan vs Ready (investor lifecycle)
**Investor reality**: after handover, the market behaves like Ready (resale + rent/yield), even if the source label sometimes remains OffPlan.

We now store BOTH:
- `reg_type_dld`: the raw DLD label (OffPlan/Ready)
- `reg_type`: lifecycle stage label derived from transaction_date vs handover_date

**DLD OffPlan after handover** is flagged (audit only):
- `dld_offplan_after_handover`: 7,370 series-rows (~10.2%)

Impact:
- OffPlan-stage trends are not polluted by post-handover transfers.
- Post-handover behavior moves into Ready-stage, which is what investors care about for “worth once built + yield”.

## 4) Handover date correctness
Handover is explicit in Projects, not Transactions.

Canonical handover date used:
- `handover_date = completion_date (actual) else project_end_date (planned)`

Master-project timeline fallback (when project_number is missing):
- start = min(start_date)
- handover = max(handover_date)

This prevents the common failure mode: assigning an **early** handover to later-phase transactions.

## 5) Supply schedule sanity (scheduled completions)
We avoid inflating completions by excluding post-handover events:
- In `compute_supply_schedule()`, DLD OffPlan transactions are only counted if `transaction_date < handover_date`.

Note: `units_completing` is still derived from off-plan transaction counts (proxy). For true investor-grade “units completing”, we should migrate to Projects `no_of_units` by handover month (area-level).

## 6) Outlier sanity (price/rent)
Investor-sanity guards are now enforced:
- **Price per sqft** (AED/sqft): 100 → 50,000
- **Rent scaling fix**: detect 100× unit issues using rent-per-sqft plausibility and rescale (÷100)

Post-fix checks:
- `median_price < 100`: ✅ 0
- `median_price > 50,000`: ✅ 0
- `median_rent > 2,000,000`: ✅ 0

## 7) Remaining “known risks” (not blockers, but must be understood)
- **Short-history groups**: Many groups have < 12 or < 24 months history (new projects / new segmentation). TFT can train with `min_encoder_length=1`, but forecast stability must be validated.
- **Supply schedule proxy**: `units_completing` is currently a proxy based on off-plan contract volume; it can over/under-state true completions.

## 8) Final recommendation
✅ Dataset is now **investor-credible** for retraining with respect to:
- brand developer identity
- lifecycle OffPlan→handover→Ready separation
- handover date sourcing
- major outlier correction

Before retraining, re-run:
- `python scripts/audit_developer_segmentation.py`
- (optional) training data QA notebook checks in `Docs/Data_docs/training_opus_colab.md`






