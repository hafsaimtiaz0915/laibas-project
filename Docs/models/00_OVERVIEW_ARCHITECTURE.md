# System Architecture Overview: Dubai Real Estate AI Platform

> **Document Version**: 2.0  
> **Last Updated**: 2025-12-10  
> **Purpose**: Define the complete system architecture for AI-driven off-plan investment analysis.

---

## 1. Executive Vision

This platform helps real estate agents qualify off-plan investments by providing **factual, data-driven market intelligence**.

**What agents ask:**
> "Binghatti development in JVC, 2BR for 2.2M - what's the outlook?"

**What we provide:**
- Predicted appreciation and handover value
- Developer track record analysis
- Area trend comparison
- Supply pipeline impact
- Clear explanation of what influenced the prediction
- **White-labeled PDF reports** for clients with agent branding

**What we DON'T provide:**
- Investment recommendations
- "Buy" or "Don't buy" advice
- Guaranteed returns

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE LAYER                                │
│                                                                                  │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌────────────────────┐  │
│  │   Chat Interface      │  │   Agent Dashboard     │  │  PDF Report Gen    │  │
│  │   "Binghatti JVC..."  │  │   Market Overview     │  │  White-label       │  │
│  │                       │  │   Portfolio           │  │  Agent branding    │  │
│  └───────────────────────┘  └───────────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATION LAYER                                 │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         LLM AGENT (Claude API)                           │   │
│  │                                                                          │   │
│  │  1. Parse query → Extract: developer, area, bedroom, price              │   │
│  │  2. Call TFT model → Get predictions + attention weights                │   │
│  │  3. Retrieve context → Lookup tables, developer stats                   │   │
│  │  4. Synthesize response → Natural language briefing                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                        ┌───────────────┼───────────────┐
                        │               │               │
                        ▼               ▼               ▼
┌───────────────────────────┐ ┌─────────────────┐ ┌─────────────────────────────┐
│    TFT MODEL (Local)      │ │ LOOKUP TABLES   │ │    ROI CALCULATOR           │
│                           │ │                 │ │                             │
│ • Trained on Colab        │ │ • Developer     │ │ • RERA formulas             │
│ • Runs on Mac CPU         │ │   stats         │ │ • Payment plans             │
│ • Predictions + attention │ │ • Area medians  │ │ • DLD fees                  │
│                           │ │ • Rent benchmarks│ │ • Service charges           │
└───────────────────────────┘ └─────────────────┘ └─────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               DATA LAYER                                         │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    Cleaned Data (Data/cleaned/)                          │    │
│  │                                                                          │    │
│  │  • Transactions_Cleaned.csv (1.6M records)                              │    │
│  │  • Rent_Contracts_Cleaned.csv (5.7M records)                            │    │
│  │  • Projects_Cleaned.csv (3K records)                                    │    │
│  │  • eibor_monthly.csv                                                    │    │
│  │  • tourism_visitors.csv                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    TFT Training Data (Data/tft/)                         │    │
│  │                                                                          │    │
│  │  • tft_training_data.csv (~750K rows, ~50-100MB)                        │    │
│  │  • Monthly aggregated: price, rent, EIBOR, supply by area/bedroom       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 TFT Model (Core Prediction Engine)

| Aspect | Details |
|--------|---------|
| **Model** | Temporal Fusion Transformer |
| **Training** | Google Colab Pro ($10/month) |
| **Inference** | Local Mac (CPU) |
| **Input** | Developer, area, bedroom, price, current market conditions |
| **Output** | Predictions + attention weights + confidence intervals |

**What TFT Learns:**
- Area price trends over time
- Developer delivery patterns
- Impact of EIBOR on prices
- Supply pipeline effects
- Bedroom dynamics (2BR vs 1BR performance)

**What TFT Outputs:**
```python
{
    "prediction": {
        "p10": 2650000,   # Pessimistic
        "p50": 2780000,   # Most likely
        "p90": 2910000    # Optimistic
    },
    "attention_weights": {
        "developer_history": 0.35,
        "area_trends": 0.28,
        "supply_pipeline": 0.18,
        "eibor_rates": 0.12,
        "bedroom_dynamics": 0.07
    }
}
```

### 3.2 LLM Agent (Interpretation Layer)

| Aspect | Details |
|--------|---------|
| **Model** | Claude API (claude-sonnet-4-20250514) |
| **Role** | Parse queries, interpret TFT outputs, generate responses |
| **Constraints** | Factual only, no investment advice |

**LLM Responsibilities:**
1. Parse natural language queries
2. Extract entities (developer, area, bedroom, price)
3. Format input for TFT model
4. Interpret attention weights for explanation
5. Generate clear, factual response

### 3.3 Lookup Tables (Context)

| Table | Contents | Use |
|-------|----------|-----|
| `developer_stats.csv` | Projects completed, avg delay, total units | Developer credibility |
| `area_medians.csv` | Current median prices by area/bedroom | Price comparison |
| `rent_benchmarks.csv` | RERA median rents by area/bedroom | Yield calculation |
| `supply_pipeline.csv` | Units expected by area/year | Risk assessment |

### 3.4 ROI Calculator (Deterministic)

| Calculation | Formula |
|-------------|---------|
| DLD Fee | 4% of purchase price |
| Service Charges | Based on area/building |
| Gross Yield | Annual Rent / Purchase Price |
| Net Yield | (Annual Rent - Service Charges) / (Purchase Price + DLD Fee) |

---

## 4. Data Flow

### 4.1 Query Processing

```
1. Agent: "Binghatti JVC 2BR 2.2M outlook?"
           │
           ▼
2. LLM parses: {developer: "Binghatti", area: "JVC", bedroom: "2BR", price: 2200000}
           │
           ▼
3. TFT predicts: {p50: 2780000, attention: {developer: 0.35, area: 0.28, ...}}
           │
           ▼
4. Lookups retrieve: {developer_projects: 12, area_median: 2100000, rent: 85000}
           │
           ▼
5. LLM synthesizes response
           │
           ▼
6. Agent receives: Structured market briefing
```

### 4.2 Response Structure

```
**Binghatti JVC 2BR at 2.2M - Market Analysis**

**Predicted Outcome:**
Estimated handover value: 2.65M - 2.91M (most likely: 2.78M)
Implied appreciation: 20-32%

**Key Factors (What Influenced This Prediction):**
• Binghatti Track Record (35%): 12 completed projects, avg 4mo delay
• JVC Area Trends (28%): 42% appreciation over past 3 years
• Supply Pipeline (18%): 12,000 units expected in JVC next 24mo
• EIBOR Impact (12%): 5.3% rate affecting buyer demand

**Context:**
• Purchase price (2.2M) is 5% above current JVC 2BR off-plan median
• Expected rental yield at handover: 6.1% (assuming 170K/yr rent)

**Considerations:**
• High supply pipeline may limit post-handover appreciation
• Binghatti has historically delivered within +/- 4 months of target
```

---

## 5. Infrastructure

### 5.1 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MONTHLY RETRAINING                            │
│                                                                      │
│  1. New DLD data arrives                                            │
│  2. Run: python scripts/build_tft_data.py                           │
│  3. Upload to Google Drive                                          │
│  4. Train on Colab (1-2 hours)                                      │
│  5. Download new checkpoint                                          │
│  6. Deploy to production                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Cost Structure

| Component | Cost | Notes |
|-----------|------|-------|
| Google Colab Pro | $10/month | Training |
| Mac | Existing | Inference |
| Claude API | $10-50/month | Usage-based |
| **Total** | **$20-60/month** | |

### 5.3 Deployment

```
Mac Server/Laptop
├── TFT Model (loaded at startup)
├── Lookup Tables (CSV files)
├── FastAPI Server
│   ├── POST /predict
│   └── POST /chat
└── LLM Client (Claude API)
```

---

## 6. Key Design Decisions

### 6.1 Why TFT (Not XGBoost/Chronos)?

| Decision | Rationale |
|----------|-----------|
| **TFT over XGBoost** | TFT handles time series natively, no manual feature engineering |
| **TFT over Chronos** | TFT provides interpretable attention weights |
| **TFT over Prophet** | TFT handles covariates (EIBOR, supply) better |

### 6.2 Why LLM Wrapper (Not Direct TFT)?

| Decision | Rationale |
|----------|-----------|
| Natural language interface | Agents speak naturally, not in API parameters |
| Explanation generation | TFT outputs numbers, LLM explains them |
| Context integration | LLM can incorporate lookup data in response |
| Output control | System prompt ensures consistent, safe responses |

### 6.3 Why Raw Data (No Feature Engineering)?

| Decision | Rationale |
|----------|-----------|
| TFT learns patterns | Model discovers lags, volatility, momentum itself |
| Simpler pipeline | Less code, fewer bugs |
| More robust | Model adapts to changing patterns |
| Less bias | No human assumptions baked in |

---

## 7. Files & Scripts

### 7.1 Data Files

| File | Location | Purpose |
|------|----------|---------|
| Cleaned data | `Data/cleaned/` | Source of truth |
| TFT training data | `Data/tft/tft_training_data.csv` | Model input |
| Model checkpoint | `models/tft_model.ckpt` | Trained model |

### 7.2 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_tft_data.py` | Build TFT training data from cleaned sources |
| `scripts/clean_all_data.py` | Clean raw DLD data |

### 7.3 Documentation

| Document | Purpose |
|----------|---------|
| `00_OVERVIEW_ARCHITECTURE.md` | This file - system overview |
| `02_TIME_SERIES_FORECASTING.md` | TFT model specification |
| `04_ROI_CALCULATOR_OFFPLAN.md` | ROI calculation formulas |
| `06_DEPLOYMENT_ARCHITECTURE.md` | Deployment details |
| `frontend/00_FRONTEND_ARCHITECTURE.md` | Frontend, chat UI, PDF reports |

---

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Prediction MAPE | < 15% |
| Response latency | < 5 seconds |
| Agent satisfaction | Qualitative feedback |
| Query volume | Track adoption |

---

## 9. Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Data cleaning, TFT data preparation | ✅ Data cleaned |
| **Phase 2** | TFT model training | 🔄 In progress |
| **Phase 3** | LLM integration | Pending |
| **Phase 4** | API deployment | Pending |
| **Phase 5** | Agent testing | Pending |

---

## References

- [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/)
- [Claude API Documentation](https://docs.anthropic.com/)
