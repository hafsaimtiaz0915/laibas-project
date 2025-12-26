# Stage 1 & 2 Verification Report

## Cleaning stats summary

- Path: C:\Users\hafsa\Downloads\proprly-main\proprly-main\Data\cleaned\cleaning_stats.json

- Keys present: buildings, projects, rent_contracts, transactions, units, valuation


### Transactions checks

- Total rows: 1612276

- Valid rows: 1359647

- Valid/Total ratio: 84.33%


### Rent contracts checks

- Total rent rows: 9524298

- Residential rows: 5743849

- Residential fraction: 60.31%


### Data loss checks

- Duplicates removed (transactions): 0

- Missing price/area removed: 0


## Lookup tables checks

- developer_mapping.json: total=114, unmapped=0

- area_stats.csv rows (excluding header): 166

- rent_benchmarks.csv rows (excluding header): 973

- area_mapping.json ambiguous aliases: 4 (<=10 expected)

- developer_stats.csv rows: 112


## Tests summary

- Total tests: 9

- Passed: 9

- Failed: 0


## Conclusion: PASS (pass ratio: 100.00%)
