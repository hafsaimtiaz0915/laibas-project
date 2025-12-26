# Model Specification: TFT Time Series Forecasting

> **Document Version**: 2.0  
> **Last Updated**: 2025-12-10  
> **Purpose**: Specification for the Temporal Fusion Transformer (TFT) model that powers off-plan investment predictions.

---

## 1. Overview

**Model Type**: Temporal Fusion Transformer (TFT)  
**Training Environment**: Google Colab Pro (GPU)  
**Inference Environment**: Local Mac (CPU)  
**Primary Use Case**: Predict outcomes for off-plan real estate investments

**Example Query**:
> "Binghatti development in JVC, 2BR for 2.2M - what's the outlook?"

**Example Output**:
- Predicted appreciation: 25-32%
- Estimated handover value: 2.75M - 2.90M
- Key factors: Developer history (35%), Area trends (28%), Supply (18%)

---

## 2. Why TFT?

### 2.1 Comparison with Alternatives

| Model | Interpretability | Covariates | Multi-horizon | Complexity |
|-------|------------------|------------|---------------|------------|
| **TFT** | ✅ Attention weights | ✅ Full support | ✅ Yes | Medium |
| Chronos | ❌ Black box | ⚠️ Limited | ✅ Yes | Low |
| Prophet | ⚠️ Components | ⚠️ Regressors only | ✅ Yes | Low |
| ARIMA | ⚠️ Coefficients | ⚠️ ARIMAX | ✅ Yes | Low |
| XGBoost | ✅ SHAP | ✅ Full | ❌ Point only | Medium |

### 2.2 TFT Advantages for Off-Plan

| Feature | Benefit |
|---------|---------|
| **Attention Weights** | Shows what influenced prediction ("Binghatti history: 35%") |
| **Multi-horizon** | Predicts 12-24 months ahead (construction timeline) |
| **Static + Time-varying** | Handles developer (static) + EIBOR (time-varying) |
| **Interpretable** | LLM can explain predictions using attention outputs |
| **Proven** | Google research, used in production forecasting |

---

## 3. Model Architecture

### 3.1 TFT Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TFT ARCHITECTURE                             │
│                                                                      │
│  ┌─────────────────┐                                                │
│  │ Static Inputs   │  area_name, property_type, bedroom            │
│  │ (Embeddings)    │                                                │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Variable Selection Network                      │    │
│  │  Learns which inputs matter for this prediction             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LSTM Encoder-Decoder                            │    │
│  │  Processes historical sequence, generates future states     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Multi-Head Attention                            │    │
│  │  Learns temporal patterns, outputs attention weights        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Quantile Outputs                                │    │
│  │  Predictions with confidence intervals (p10, p50, p90)      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Input Categories

| Category | TFT Term | Examples | Updates Over Time? |
|----------|----------|----------|-------------------|
| **Static** | static_categoricals | area_name, developer, bedroom | No |
| **Known Future** | time_varying_known | month, quarter, holidays | Yes (known ahead) |
| **Observed** | time_varying_unknown | price, rent, EIBOR, supply | Yes (only historical) |
| **Target** | target | median_price | What we predict |

---

## 4. Data Requirements

### 4.1 Training Data Format

Single CSV with columns:

```csv
time_idx,year_month,group_id,area_name,property_type,bedroom,month,quarter,month_sin,month_cos,median_price,transaction_count,median_rent,rent_count,eibor_3m,supply_units
```

### 4.2 Key Constraints

| Constraint | Requirement |
|------------|-------------|
| `time_idx` | Integer starting from 0, sequential per group |
| `group_id` | Unique identifier: `{area}_{property_type}_{bedroom}` |
| No gaps | Missing months should be interpolated or excluded |
| No future data | Only include data known at each time step |

### 4.3 Minimum Data

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| History per group | 24 months | 48+ months |
| Groups | 100 | 500+ |
| Total rows | 50,000 | 500,000+ |

---

## 5. Training Configuration

### 5.1 PyTorch Forecasting Setup

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Dataset configuration
training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="median_price",
    group_ids=["group_id"],
    
    static_categoricals=["area_name", "property_type", "bedroom"],
    
    time_varying_known_categoricals=["month", "quarter"],
    time_varying_known_reals=["month_sin", "month_cos"],
    
    time_varying_unknown_reals=[
        "median_price", "transaction_count", 
        "median_rent", "rent_count",
        "eibor_3m", "supply_units"
    ],
    
    max_encoder_length=24,      # 24 months history
    max_prediction_length=12,   # 12 months forecast
    
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Model configuration
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4,
)
```

### 5.2 Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_encoder_length` | 24 | 2 years of history |
| `max_prediction_length` | 12 | 1 year forecast |
| `hidden_size` | 64 | Adjust based on data size |
| `attention_head_size` | 4 | Standard for this task |
| `dropout` | 0.1 | Regularization |
| `learning_rate` | 0.001 | With scheduler |
| `batch_size` | 64 | Adjust for GPU memory |
| `max_epochs` | 50 | With early stopping |

---

## 6. Inference

### 6.1 Prediction Output

```python
# Load trained model
tft = TemporalFusionTransformer.load_from_checkpoint("tft_model.ckpt")

# Make prediction
predictions = tft.predict(dataloader, return_index=True, return_x=True)

# Output structure
{
    "prediction": {
        "p10": [2650000, 2680000, ...],  # 10th percentile
        "p50": [2780000, 2820000, ...],  # Median (main prediction)
        "p90": [2910000, 2960000, ...]   # 90th percentile
    },
    "attention_weights": {
        "area_name": 0.28,
        "developer": 0.35,
        "eibor_3m": 0.12,
        "supply_units": 0.18,
        "bedroom": 0.07
    }
}
```

### 6.2 Interpretation

The attention weights tell us **why** the model made this prediction:

| Weight | Meaning |
|--------|---------|
| `developer: 0.35` | 35% of prediction based on developer history |
| `area_name: 0.28` | 28% based on area price trends |
| `supply_units: 0.18` | 18% based on supply pipeline |
| `eibor_3m: 0.12` | 12% based on interest rates |

This enables the LLM to generate explanations:

> "The prediction is primarily influenced by Binghatti's historical delivery and pricing patterns (35%), followed by JVC area trends (28%). The high supply pipeline in JVC (18% weight) suggests some caution."

---

## 7. Integration with LLM

### 7.1 Workflow

```python
def process_query(query: str) -> str:
    # 1. Parse query
    parsed = llm.parse_entities(query)
    # {"developer": "Binghatti", "area": "JVC", "bedroom": "2BR", "price": 2200000}
    
    # 2. Run TFT prediction
    prediction = tft.predict(parsed)
    # {"p50": 2780000, "attention": {...}}
    
    # 3. Get context
    context = lookup_tables.get_context(parsed)
    # {"developer_projects": 12, "area_median": 2100000, ...}
    
    # 4. LLM synthesizes response
    response = llm.synthesize(
        query=query,
        prediction=prediction,
        context=context,
        system_prompt=AGENT_PROMPT
    )
    
    return response
```

### 7.2 LLM System Prompt

```
You are a real estate market analyst. You will receive:
1. TFT model predictions (numbers + confidence intervals)
2. Attention weights (what influenced the prediction)
3. Context (developer history, area stats)

Your job: Convert this into a clear, factual briefing.

RULES:
- State facts, never give investment advice
- Explain what influenced the prediction using attention weights
- Include confidence ranges
- Mention relevant risks
- Be concise but comprehensive
```

---

## 8. Training Workflow

### 8.1 Steps

| Step | Where | Command/Action | Time |
|------|-------|----------------|------|
| 1 | Mac | `python scripts/build_tft_data.py` | 15 min |
| 2 | Mac | Upload `tft_training_data.csv` to Google Drive | 5 min |
| 3 | Colab | Run training notebook | 1-2 hours |
| 4 | Colab | Evaluate on validation set | 10 min |
| 5 | Mac | Download `tft_model.ckpt` | 5 min |
| 6 | Mac | Test inference | 10 min |

### 8.2 Retraining Schedule

| Trigger | Action |
|---------|--------|
| Monthly | Retrain with new DLD data |
| Performance degradation | Investigate and retrain |
| New areas/developers | Retrain to include new entities |

---

## 9. Evaluation Metrics

### 9.1 Forecasting Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **MAPE** | mean(\|actual - predicted\| / actual) | < 15% |
| **MAE** | mean(\|actual - predicted\|) | Context-dependent |
| **Coverage** | % of actuals within p10-p90 | > 80% |

### 9.2 Backtesting

```python
# Walk-forward validation
for t in range(12):
    train_end = len(data) - 12 + t
    train = data[:train_end]
    test = data[train_end:train_end + 12]
    
    model = train_tft(train)
    predictions = model.predict(test)
    
    mape = calculate_mape(predictions, test.actual)
    coverage = calculate_coverage(predictions, test.actual)
```

---

## 10. Known Limitations

| Limitation | Mitigation |
|------------|------------|
| Requires sufficient history per group | Filter groups with < 24 months |
| Cannot predict black swan events | Clearly communicate uncertainty |
| Attention weights are approximate | Use for explanation, not exact causation |
| New developers have no history | Use area-level predictions as fallback |
| Forecast accuracy degrades > 12 months | Focus on 12-month horizon |

---

## 11. Files & Artifacts

| File | Location | Purpose |
|------|----------|---------|
| Training data | `Data/tft/tft_training_data.csv` | Input to model |
| Model checkpoint | `models/tft_model.ckpt` | Trained model |
| Config | `Data/tft/data_config.json` | Dataset configuration |
| Training notebook | `notebooks/train_tft.ipynb` | Colab training |

---

## References

- [Temporal Fusion Transformers Paper](https://arxiv.org/abs/1912.09363)
- [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/)
- [TFT Tutorial](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)
