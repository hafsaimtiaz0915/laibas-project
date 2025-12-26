# Proprly — Product Overview & Requirements Document

> **Version**: 1.0  
> **Last Updated**: December 2025  
> **Document Type**: Product Requirements Document (PRD)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Target Users](#3-target-users)
4. [Product Overview](#4-product-overview)
5. [Key Features](#5-key-features)
6. [Output Format](#6-output-format)
7. [Technical Architecture](#7-technical-architecture)
8. [Model & Training](#8-model--training)
9. [Data Sources](#9-data-sources)
10. [Pricing](#10-pricing)
11. [Infrastructure & Deployment](#11-infrastructure--deployment)
12. [Security & Compliance](#12-security--compliance)
13. [Success Metrics](#13-success-metrics)
14. [Roadmap](#14-roadmap)

---

## 1. Executive Summary

**Proprly** is an AI-powered investment analysis platform for Dubai's off-plan real estate market. It helps real estate agents and brokers qualify off-plan investments by providing **data-driven market intelligence** that they can share with investor clients.

### What Proprly Does

Agents ask natural language questions about properties:
> "Binghatti development in JVC, 2BR for AED 2.2M — what's the outlook?"

Proprly provides:
- **Predicted appreciation** and handover value (with confidence ranges)
- **Developer track record** analysis
- **Area trend comparison** and market context
- **Supply pipeline** impact assessment
- **Clear explanation** of what drives the forecast
- **White-labeled PDF reports** for investor clients

### What Proprly Does NOT Do

- ❌ Provide investment recommendations
- ❌ Offer "buy" or "don't buy" advice
- ❌ Guarantee returns
- ❌ Provide financial advice

Proprly delivers **factual, data-backed market intelligence** — the investment decision remains with the investor.

---

## 2. Problem Statement

### The Agent's Challenge

Real estate agents selling off-plan properties in Dubai face a credibility gap:

1. **Investors want data** — "How much will this appreciate?" "What's the rental yield?" "Is the developer reliable?"
2. **Agents lack tools** — Most rely on developer marketing materials or anecdotal evidence
3. **Trust is fragile** — Unsubstantiated claims damage relationships and sales conversions

### The Investor's Perspective

Investors evaluating off-plan purchases need answers to:
- What will this unit likely be worth once it's built?
- What yield will it generate once it's rentable?
- How does this price compare to similar properties?
- Does this developer deliver on time?
- Is there oversupply coming in this area?

### The Solution

Proprly bridges this gap by:
1. **Aggregating official DLD transaction data** (1.6M+ transactions)
2. **Training a forecasting model** on historical patterns
3. **Generating investor-grade reports** agents can share with confidence

---

## 3. Target Users

### Primary User: Real Estate Agents & Brokers

| Segment | Description | Use Case |
|---------|-------------|----------|
| **Individual Brokers** | Solo agents with investor clients | Quick property lookups, PDF reports for pitches |
| **Active Brokerages** | Teams of 3-10 agents | Shared reports, branded templates |
| **Investment Desks** | Dedicated investment advisory teams | High-volume analysis, internal workflows |
| **Developer Sales Teams** | Agents aligned with specific developers | Credible third-party validation |

### User Jobs-to-be-Done

1. **Answer investor objections quickly** with data-backed responses
2. **Generate professional reports** to send to prospects
3. **Compare areas and projects** for portfolio recommendations
4. **Validate developer claims** with independent track record data

---

## 4. Product Overview

### Core Interface: AI Chat

Proprly uses a ChatGPT-style conversational interface:

```
┌────────────────────────────────────────────────────────────────────┐
│  ┌────────────┐  ┌──────────────────────────────────────────────┐ │
│  │ SIDEBAR    │  │              CHAT AREA                        │ │
│  │            │  │                                               │ │
│  │ + New Chat │  │  [User Message]                               │ │
│  │            │  │  "Binghatti JVC 2BR at 2.2M?"                 │ │
│  │ Today      │  │                                               │ │
│  │ • Chat 1   │  │  [Assistant Response]                         │ │
│  │ • Chat 2   │  │  📊 Off-Plan Investment Snapshot              │ │
│  │            │  │  Based on market data...                      │ │
│  │ Yesterday  │  │                                               │ │
│  │ • Chat 3   │  │  ┌────────────────────────────────────────┐  │ │
│  │            │  │  │ Analysis Card + [Generate PDF Report]  │  │ │
│  │            │  │  └────────────────────────────────────────┘  │ │
│  │ ────────   │  │                                               │ │
│  │ ⚙ Settings │  │  [Message Input]                     [Send]  │ │
│  └────────────┘  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### Workflow

1. **Ask** — Agent asks about a specific property, project, or area
2. **Analyze** — AI extracts entities, runs model, retrieves context
3. **Review** — Agent sees text-based investment snapshot
4. **Export** — Agent generates branded PDF for client

---

## 5. Key Features

### 5.1 Natural Language Queries

Agents speak naturally:
- "What's the outlook for a 2BR in Emaar Beachfront?"
- "Compare JVC to Business Bay for 1BR apartments"
- "Is Damac a reliable developer for off-plan?"

The LLM parses intent and extracts:
- Developer name
- Area/location
- Bedroom configuration
- Price (if provided)
- Unit size (if provided)

### 5.2 Investment Forecast

For each query, Proprly generates:

| Metric | Description |
|--------|-------------|
| **Estimated Value at Handover** | Median + range (low/high) |
| **Estimated Value +12 Months** | Post-handover appreciation |
| **Projected Uplift** | AED and % vs. purchase price |
| **Gross Yield** | Expected rental yield after handover |

### 5.3 Market Context

Supporting data includes:
- Current area median price (AED/sqft)
- 12-month and 36-month price change
- Transaction volume (liquidity)
- Supply pipeline (upcoming units)

### 5.4 Developer Track Record

For each developer:
- Projects completed vs. total
- Average time to complete
- Average delay (months)
- Total units delivered

### 5.5 Forecast Explanation

"What drives the forecast" section explains:
- Lifecycle timing to handover
- Supply pipeline pressure
- Area momentum
- Market liquidity
- Interest rate environment (EIBOR)

### 5.6 Branded PDF Reports

Agents can generate white-labeled reports with:
- Custom logo
- Primary/secondary colors
- Contact information
- Legal disclaimer
- Professional formatting with charts

Reports are saved to the chat for future downloads.

### 5.7 Agent Settings

Customization options:
- Logo upload
- Color scheme
- Contact details
- Report header/footer text

---

## 6. Output Format

### 6.1 Text Output (Chat)

The assistant generates a structured **Off-Plan Investment Snapshot**:

```
📊 OFF-PLAN INVESTMENT SNAPSHOT
Generated: [Date]

DEAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Area: JVC (Jumeirah Village Circle)
Developer: Binghatti
Unit: Apartment • 2 Bedrooms • Off-Plan
Handover: in 18 months

UNIT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit size: 850 sqft
Purchase price: AED 2,200,000
Purchase price/sqft: AED 2,588/sqft

ESTIMATED VALUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
At Handover:
  Estimated: AED 2,780,000
  Range: AED 2,650,000 – AED 2,910,000

12 Months After Handover:
  Estimated: AED 2,950,000
  Range: AED 2,800,000 – AED 3,100,000

PROJECTED UPLIFT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
By Handover: +AED 580,000 (+26%)
By +12 Months: +AED 750,000 (+34%)

RENT & YIELD (POST-HANDOVER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estimated Annual Rent: AED 130,000/yr
Range: AED 120,000 – AED 140,000/yr
Gross Yield: 5.9% (range: 5.4% – 6.4%)

AREA MARKET CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Median: AED 1,180/sqft
12-Month Change: +18%
36-Month Change: +42%
Transactions (12m): 2,847
Supply Pipeline: 8,500 units

DEVELOPER EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Projects: 12 completed
Avg Time to Complete: 32 months
Avg Delay: 4 months
Units Delivered: 5,200+

WHAT DRIVES THE FORECAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Lifecycle timing (18 months to handover)
• Moderate supply in pipeline
• Strong area momentum (+18% YoY)
• High liquidity (2,800+ transactions)
• Current EIBOR: 5.3%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Press Generate Report to export the investor PDF.
```

### 6.2 PDF Report

The PDF is a 2-page investor-grade document:

**Page 1: Investment Summary**
- Header with agent logo and branding
- Deal summary box
- Value forecast (handover + 12m) with visual chart
- Uplift projections
- Rent & yield estimates

**Page 2: Market Context & Model Explanation**
- Area market statistics
- Developer track record
- "What drives the forecast" explanation
- Training data coverage
- Legal disclaimer

---

## 7. Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Vercel)                      │
│                                                                  │
│   Next.js 14 • Tailwind CSS • shadcn/ui • Zustand               │
│   Chat Interface • Settings • PDF Generation                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS / REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND API (Railway)                        │
│                                                                  │
│   FastAPI • Python 3.11 • Uvicorn                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│   │ Query Parser│  │ TFT Model   │  │ Response Generator  │    │
│   │ (GPT-4)     │  │ (PyTorch)   │  │ (GPT-4)             │    │
│   └─────────────┘  └─────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌───────────────────┐ ┌─────────────┐ ┌─────────────────────────┐
│  Supabase Auth    │ │ Supabase DB │ │  Supabase Storage       │
│  (JWT)            │ │ (Postgres)  │ │  • agent-logos (public) │
│                   │ │             │ │  • reports (private)    │
│                   │ │             │ │  • model (private)      │
└───────────────────┘ └─────────────┘ └─────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Next.js 14 (App Router), Tailwind CSS, shadcn/ui |
| **State Management** | Zustand |
| **PDF Generation** | @react-pdf/renderer |
| **Backend** | FastAPI, Python 3.11, Uvicorn |
| **Database** | Supabase (PostgreSQL) |
| **Authentication** | Supabase Auth (JWT) |
| **Storage** | Supabase Storage |
| **Forecasting Model** | Temporal Fusion Transformer (PyTorch Forecasting) |
| **LLM** | OpenAI GPT-4 |
| **Frontend Hosting** | Vercel |
| **Backend Hosting** | Railway |

---

## 8. Model & Training

### 8.1 Model Architecture: Temporal Fusion Transformer (TFT)

The TFT is a state-of-the-art deep learning architecture for time series forecasting that provides:

1. **Multi-horizon forecasting** — Predict 6-12+ months ahead
2. **Interpretable attention** — Understand what drives each prediction
3. **Covariate handling** — Incorporate static and time-varying features
4. **Quantile outputs** — Provide confidence ranges (P10, P50, P90)

### 8.2 Model Inputs

| Feature Type | Examples |
|--------------|----------|
| **Static (per group)** | Developer, area, bedroom, property type |
| **Time-varying known** | Handover schedule, seasonality, EIBOR |
| **Time-varying unknown** | Price per sqft, rent per sqft, transaction volume |

### 8.3 Training Data

The model is trained on ~750K monthly observations segmented by:
- Area (DLD area names)
- Developer (Arabic registered names)
- Bedroom configuration (Studio, 1BR, 2BR, 3BR+)
- Registration type (Off-Plan, Ready)

### 8.4 Lifecycle-Aware Learning

The model is trained with explicit lifecycle features:
- `months_to_handover` — Positive before handover, negative after
- `months_since_handover` — Time in ready market
- `handover_window_6m` — Within ±6 months of handover

This allows the model to learn:
- Off-plan pricing dynamics
- Handover transition effects
- Ready market stabilization

### 8.5 Training Infrastructure

| Component | Details |
|-----------|---------|
| **Training Environment** | Google Colab Pro |
| **Training Time** | ~1-2 hours per run |
| **Model Size** | ~12 MB checkpoint |
| **Retraining Frequency** | Monthly (when new DLD data arrives) |

### 8.6 Model Outputs

```python
{
    "prediction": {
        "p10": 2650000,   # 10th percentile (pessimistic)
        "p50": 2780000,   # Median (most likely)
        "p90": 2910000    # 90th percentile (optimistic)
    },
    "attention_weights": {
        "area_trend": 0.28,
        "supply_pipeline": 0.18,
        "lifecycle_timing": 0.22,
        "eibor_rates": 0.12,
        "liquidity": 0.15,
        "other": 0.05
    }
}
```

---

## 9. Data Sources

### 9.1 Primary Data: Dubai Land Department (DLD)

All core data comes from official DLD open data:

| Dataset | Records | Purpose |
|---------|---------|---------|
| **Transactions** | 1.6M+ | Historical prices, trends |
| **Rent Contracts** | 5.7M+ | Yield calculations, rent benchmarks |
| **Projects** | 3K+ | Supply pipeline, completion schedules |
| **Buildings** | 10K+ | Physical attributes |

### 9.2 Supplementary Data

| Dataset | Source | Purpose |
|---------|--------|---------|
| **EIBOR Rates** | CBUAE | Interest rate impact |
| **Tourism Stats** | Dubai Tourism | Demand drivers |

### 9.3 Derived Lookup Tables

Generated during ETL:

| Table | Contents |
|-------|----------|
| `developer_stats` | Track record, delays, units delivered |
| `area_medians` | Current median prices by segment |
| `rent_benchmarks` | RERA-compliant rent medians |
| `supply_pipeline` | Units expected by area/year |

---

## 10. Pricing

### Subscription Tiers

| Tier | Price | Reports/Month | Users | Key Features |
|------|-------|---------------|-------|--------------|
| **Starter** | AED 2,500/mo | 5 | 1 | Standard PDF, basic forecasts |
| **Professional** | AED 7,500/mo | 40 | 4 | Branded templates, deeper drivers |
| **Advanced** | AED 20,000/mo | 100+ | 10 | Fully branded, team workflows |
| **Enterprise** | Custom | Unlimited | Custom | Custom templates, dedicated support |

### What's Included

**Starter**
- AI-validated buyer-facing PDF report
- Standard report format (non-customizable)
- Basic area & unit forecasts
- Email support

**Professional**
- Editable buyer-facing templates
- Broker branding (logo + footer)
- Internal broker view with deeper drivers
- Priority support

**Advanced**
- Fully branded reports
- High-volume usage
- Team workflows (multi-user)
- Phone support

**Enterprise**
- Custom limits & rollout
- Custom templates & branding
- Dedicated onboarding
- API access

### Operational Costs

| Component | Monthly Cost |
|-----------|--------------|
| Google Colab Pro (training) | $10 |
| Railway (backend hosting) | ~$20-50 |
| OpenAI API | ~$10-50 (usage-based) |
| Supabase | Free tier / $25 |
| Vercel | Free tier / $20 |
| **Total Infrastructure** | **~$60-150/mo** |

---

## 11. Infrastructure & Deployment

### 11.1 Architecture

| Service | Provider | Purpose |
|---------|----------|---------|
| Frontend | Vercel | Next.js hosting, auto-deploy from GitHub |
| Backend | Railway | FastAPI, always-on, model inference |
| Database | Supabase | PostgreSQL, auth, storage |

### 11.2 Model Deployment Strategy

1. **Model stored in Supabase Storage** (private bucket)
2. **Downloaded at backend startup** (~5-10 seconds)
3. **Loaded once, kept in memory** (singleton pattern)
4. **Railway always-on** — no cold starts

### 11.3 Storage Buckets

| Bucket | Visibility | Contents |
|--------|------------|----------|
| `agent-logos` | Public | Agent logos for PDF reports |
| `reports` | Private | Generated PDF reports (RLS per user) |
| `model` | Private | TFT checkpoint + training data |

### 11.4 Environment Variables

**Backend (Railway):**
```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...
OPENAI_API_KEY=sk-...
CORS_ORIGINS=https://proprly.ae,https://www.proprly.ae
```

**Frontend (Vercel):**
```
NEXT_PUBLIC_API_URL=https://api.proprly.ae
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
```

---

## 12. Security & Compliance

### 12.1 Authentication

- **Supabase Auth** with JWT tokens
- Frontend passes JWT to backend on every request
- Backend validates token before processing

### 12.2 Data Privacy

- Reports stored per-user with Row Level Security (RLS)
- Signed URLs for private file downloads (expire after 1 hour)
- No sharing of client data between agents

### 12.3 Model Protection

- Model checkpoint stored in private Supabase bucket
- Only accessible via service role key (backend-only)
- Not exposed to end users

### 12.4 Compliance Considerations

- **RERA**: Rent calculations follow Dubai RERA guidelines
- **Disclaimers**: All reports include legal disclaimers
- **No financial advice**: Explicitly avoided in outputs

---

## 13. Success Metrics

### Product Metrics

| Metric | Target |
|--------|--------|
| **Prediction MAPE** | < 15% |
| **Response latency** | < 5 seconds |
| **PDF generation time** | < 3 seconds |
| **Uptime** | 99.5% |

### Business Metrics

| Metric | Description |
|--------|-------------|
| **MAU** | Monthly active users |
| **Reports generated** | PDF exports per month |
| **Conversion rate** | Free trial → paid |
| **Churn rate** | Monthly subscription cancellations |

### User Satisfaction

- Qualitative feedback from agents
- Feature requests tracking
- NPS score

---

## 14. Roadmap

### ✅ Phase 1: MVP (Complete)

- [x] Data cleaning & ETL pipeline
- [x] TFT model training
- [x] Chat interface
- [x] PDF report generation
- [x] Agent branding settings
- [x] Backend deployment (Railway)
- [x] Frontend deployment (Vercel)

### 🔄 Phase 2: Polish & Launch (In Progress)

- [ ] Landing page
- [ ] Payment integration
- [ ] User onboarding flow
- [ ] Help documentation
- [ ] Beta testing with agents

### 📋 Phase 3: Growth Features

- [ ] Multi-user team support
- [ ] Report history & analytics
- [ ] Area comparison tool
- [ ] Portfolio tracking
- [ ] API access (Enterprise)

### 🔮 Phase 4: Advanced

- [ ] Ready market analysis
- [ ] Rental-focused reports
- [ ] Developer leaderboards
- [ ] Market alerts & notifications
- [ ] Mobile app

---

## Appendix: Key Documents

| Document | Path | Description |
|----------|------|-------------|
| PRD (Original) | `Docs/PRD.md` | Technical requirements |
| Architecture | `Docs/models/00_OVERVIEW_ARCHITECTURE.md` | System design |
| Frontend | `Docs/frontend/00_FRONTEND_ARCHITECTURE.md` | UI/UX specs |
| LLM Integration | `Docs/frontend/LLM/00_LLM_INTEGRATION_PLAN.md` | AI pipeline |
| Output Contract | `Docs/frontend/LLM/OUTPUT_CONTRACT.md` | Response format |
| Hosting | `Docs/Backend_hosting.md` | Deployment guide |
| Model Retraining | `Docs/Data_docs/RETRAINING_INVESTOR_LENS.md` | Training details |

---

*This document is the single source of truth for what Proprly is, who it's for, and how it works.*

