# Pre-Retrain Gate Report

## Dataset snapshot

- **rows**: 72205
- **groups**: 1745
- **areas**: 78
- **date_range**: 2003-06 -> 2025-12

## Gate results

| Gate | Severity | Result | Details |
| --- | --- | --- | --- |
| Developer segmentation audit script runs | HARD | PASS | ok |
| Developer segmentation audit output exists | HARD | PASS | /Users/imraan/Desktop/Properly/Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.md |
| Training data exists | HARD | PASS | /Users/imraan/Desktop/Properly/Data/tft/tft_training_data.csv |
| Required columns present | HARD | PASS | ok |
| No Unknown developers | HARD | PASS | Unknown rows=0 |
| No UNMAPPED_DEVELOPER__ rows | HARD | PASS | unmapped rows=0 |
| ALL_DEVELOPERS bucket small (<5%) | SOFT | PASS | 1791 rows (2.48%) |
| Lifecycle reg_type has both OffPlan and Ready | HARD | PASS | {'Ready': 46108, 'OffPlan': 26097} |
| DLD OffPlan after handover is tracked | INFO | PASS | flagged series-rows=7370 (10.21%) |
| median_price within [100, 50k] AED/sqft | HARD | PASS | low=0, high=0 |
| median_rent non-negative | HARD | PASS | neg=0 |
| median_rent not absurd (>2M) | SOFT | PASS | >2M rows=0 |
| units_completing has signal (non-zero rows) | SOFT | PASS | nonzero=8810 (12.20%) |
| Group history <12 months (monitor) | INFO | PASS | 599/1745 groups (34.33%) |
| Group history <24 months (monitor) | INFO | PASS | 888/1745 groups (50.89%) |


## Notes
- HARD failures must be fixed before retraining.
- SOFT items are acceptable but should be monitored.
