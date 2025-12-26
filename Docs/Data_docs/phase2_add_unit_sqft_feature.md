# Adding unit_sqft as a TFT Model Feature

This document explains how to add `unit_sqft` (unit size in square feet) as a feature in the TFT model.

## Overview

The `procedure_area` field in the Transactions data contains the unit size (sqft). Currently it's only used to calculate `price_sqft` but is not included as a model feature.

Adding unit size can help the model understand:
- Price variations by unit size within the same bedroom count
- Premium pricing for larger units
- Market preferences for different unit sizes

## Step 1: Modify build_tft_data.py

### 1.1 Add to Aggregation

In the `aggregate_transactions()` function (~line 230-241), add `median_unit_sqft` to the aggregation:

```python
# Aggregate by group INCLUDING developer_name
agg = df.groupby(['year_month', 'area_name_en', 'property_type_en', 'bedroom', 'reg_type', 'developer_name']).agg(
    median_price=('price_sqft', 'median'),
    transaction_count=('transaction_id', 'count'),
    # ADD THIS LINE:
    median_unit_sqft=('procedure_area', 'median'),
    # Project phase columns (median for the group)
    months_since_launch=('months_since_launch', 'median'),
    months_to_handover=('months_to_handover', 'median'),
    project_percent_complete=('project_percent_complete', 'median'),
    project_duration_months=('project_duration_months', 'median'),
    phase_ratio=('phase_ratio', 'median')
).reset_index()
```

### 1.2 Add to Final Columns

In the `main()` function (~line 1024-1057), add the column to the final columns list:

```python
columns = [
    # TFT required
    'time_idx', 'year_month', 'group_id',
    # Static categoricals
    'area_name', 'property_type', 'bedroom', 'reg_type', 'developer_name',
    # Time-varying known (calendar)
    'month', 'quarter', 'month_sin', 'month_cos',
    # Time-varying known - supply schedule
    'units_completing',
    # Time-varying unknown - prices (TARGET)
    'median_price', 'transaction_count',
    # ADD THIS LINE (time-varying unknown - unit characteristics)
    'median_unit_sqft',
    # ... rest of columns
]
```

## Step 2: Modify TFT Model Configuration

In your Colab training notebook, update the TimeSeriesDataSet configuration:

### Option A: As Time-Varying Unknown Real

If unit sizes can change over time for a group (more likely):

```python
time_varying_unknown_reals=[
    "median_price",          # TARGET
    "median_rent",           # TARGET
    "transaction_count",
    "median_unit_sqft",      # ADD THIS
    # ... other existing columns
],
```

### Option B: As Static Real

If you want to use the overall median unit size for the group (less common):

```python
# First compute static version in preprocessing:
# group_static_sqft = df.groupby('group_id')['median_unit_sqft'].median()

static_reals=[
    "static_unit_sqft",      # ADD THIS
],
```

### Recommended: Option A (Time-Varying Unknown)

Unit sizes can vary over time as:
- Larger/smaller units sell at different times
- Market mix changes within a group
- Developer inventory changes

## Step 3: Data Quality Checks

Before retraining, verify the data:

```python
# Check distribution
print(df['median_unit_sqft'].describe())

# Check for outliers (sanity filter)
# Typical Dubai residential: 250-10,000 sqft
outliers = df[(df['median_unit_sqft'] < 100) | (df['median_unit_sqft'] > 20000)]
print(f"Outliers: {len(outliers)} rows")

# Check null rate
print(f"Null rate: {df['median_unit_sqft'].isna().mean()*100:.1f}%")
```

## Step 4: Normalization

PyTorch Forecasting will automatically normalize continuous features, but you may want to consider:

1. **Log transformation** - if distribution is highly skewed:
   ```python
   df['median_unit_sqft_log'] = np.log1p(df['median_unit_sqft'])
   ```

2. **Binning** - for categorical treatment:
   ```python
   df['unit_size_bin'] = pd.cut(
       df['median_unit_sqft'], 
       bins=[0, 500, 800, 1200, 2000, 5000, float('inf')],
       labels=['Compact', 'Small', 'Medium', 'Large', 'XLarge', 'Mega']
   )
   ```

## Step 5: Update Inference

After retraining, update `tft_inference.py` to handle the new feature in predictions:

1. Ensure the feature is included in encoder data preparation
2. Handle null values appropriately (median imputation recommended)

## Expected Impact

Adding unit_sqft should:
- Improve price prediction accuracy for groups with varying unit sizes
- Help explain within-group price variations
- Provide better predictions for premium/economy segments

## Rollback

If the feature doesn't improve performance:
1. Simply remove from `time_varying_unknown_reals` in model config
2. No need to rebuild training data (column will be ignored)
