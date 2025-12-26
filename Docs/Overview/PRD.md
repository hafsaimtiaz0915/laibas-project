# Product Requirements Document (PRD): AI-Driven Predictive Modeling for Dubai Real Estate

## 1. Executive Summary
This document outlines the product requirements for building a **Hybrid Neuro-Symbolic AI System** tailored for the Dubai real estate market. The system aims to empower real estate agents and investment advisors with high-fidelity, actionable predictions regarding Return on Investment (ROI), capital appreciation, and rental yields.

By fusing **Large Language Models (LLMs)** with **quantitative time-series forecasting engines** (e.g., Amazon Chronos, XGBoost), the platform will act as an "AI Co-pilot," capable of answering complex natural language queries with mathematical precision while adhering to local regulatory frameworks (RERA).

## 2. Product Objectives
1.  **Predictive Accuracy**: Deliver micro-level valuation and ROI forecasts (e.g., specific unit in a specific building) superior to simple market averages.
2.  **Regulatory Compliance**: Embed Dubai’s RERA rental index rules directly into the forecasting logic to prevent unrealistic yield promises.
3.  **Natural Language Interface**: Enable non-technical users to query complex data using conversational AI.
4.  **Secure Personalization**: Allow agents to integrate private client data via a multi-tenant RAG architecture without data leakage.

## 3. Data Architecture & Gap Analysis

### 3.1 Existing Data (Available in `Data/` Directory)
The core training data is sourced from the Dubai Land Department (DLD) open data, currently available as CSV files. These files form the backbone of the quantitative engine.

#### **A. Transactions (`Data/Transactions.csv`)**
*   **Purpose**: Primary source for historical price movements and determining market value.
*   **Key Features**:
    *   `instance_date`: Timeline for time-series forecasting.
    *   `actual_worth`: Target variable ($y$) for price prediction.
    *   `procedure_name_en`: Filter for "Sales" vs. "Mortgages".
    *   `reg_type_en`: Distinguish "Off-Plan" vs. "Existing Properties".
    *   `meter_sale_price`: Normalization metric (Price/sqft).
    *   `area_name_en`, `project_name_en`: Location clustering.
    *   `rooms_en`: Unit configuration.

#### **B. Units (`Data/Units.csv`)**
*   **Purpose**: "Micro-prediction" features for Hedonic Pricing Models.
*   **Key Features**:
    *   `unit_balcony_area`: Premium feature for post-pandemic valuation.
    *   `parking_allocation_type` / `unit_parking_number`: Value add for high-density zones.
    *   `floor`: Proxy for view quality and noise levels.
    *   `actual_area`: Denominator for price-per-sqft.

#### **C. Projects (`Data/Projects.csv`)**
*   **Purpose**: Supply-side analysis and risk assessment.
*   **Key Features**:
    *   `percent_completed`: Construction risk metric.
    *   `completion_date`: Forecasting supply shocks (handover spikes).
    *   `escrow_agent_name`: Proxy for developer financial stability.

#### **D. Rent Contracts (`Data/Rent_Contracts.csv`)**
*   **Purpose**: Yield calculation and rental trend analysis.
*   **Size**: ~9.5M rows, 4.2GB (too large for direct database storage)
*   **Processing**: Aggregated during ETL into three tables:
    *   `rent_benchmarks`: Market rent by area/type/bedrooms/month
    *   `rent_trends`: Area-level rent trends with YoY/MoM changes
    *   `vacancy_stats`: Pre-computed void period analysis
*   **Key Source Features**:
    *   `annual_amount`: Actual contracted rent (truth data vs. listing price).
    *   `contract_start_date` / `contract_end_date`: Void period calculation.
    *   `property_usage_en`: Commercial vs. Residential filtering.
    *   `ejari_property_sub_type_en`: Bedroom count parsing

#### **E. Buildings (`Data/Buildings.csv`)**
*   **Purpose**: Linking units to master projects and getting physical building attributes.
*   **Key Features**: `floors`, `master_project_en`, `bld_levels`.

#### **F. Valuation (`Data/Valuation.csv`)**
*   **Purpose**: Supplementary validation data for property evaluations.

#### **G. Tourism & Hospitality (`Data/Toursim Data/`)**
*   **Purpose**: Demand drivers for short-term rentals and holiday homes.
*   **Key Datasets**:
    *   **Visitors by Geographic Region** (`الزوار حسب المنطقة الجغرافية`): Corresponds to "Visitor Arrivals by Source Market". Available quarterly (2023-2025).
    *   **Hotel & Apartment Occupancy** (`الفنادق ومتوسط إشغال الغرف`): Occupancy rates by classification (Star rating), essential for yield prediction in short-term rentals.
    *   **Inventory Stats** (`الغرف الفندقية والشقق الفندقية`): Total supply of hotel rooms and apartments.
    *   **FDI Reports** (PDFs): Foreign Direct Investment trends (Annual reports).
*   **Processing Requirement**: These files are in `.xlsx` format and may require Arabic-to-English translation of headers during the ETL process.

---

### 3.2 Missing Data & Integration Requirements (Critical Gaps)
To achieve the "Strategic Architecture" defined in the requirements, the following datasets must be acquired or engineered.

#### **A. Macroeconomic Indicators (External APIs Required)**
*   **Oil Prices**:
    *   *Requirement*: Historical and real-time Brent Crude / Dubai Crude prices.
    *   *Source*: FRED API or OilPrice API.
    *   *Impact*: Correlates with regional liquidity and luxury demand.
*   **Interest Rates**:
    *   *Requirement*: EIBOR (UAE) and Fed Funds Rate.
    *   *Source*: CBUAE or financial data providers (e.g., Bloomberg/Reuters API).
    *   *Impact*: Inverse correlation with mortgage affordability.


## 4. Functional Requirements

### 4.1 Data Ingestion & ETL Pipeline
*   **Automated Extraction**: Scripts to poll DLD sources for new CSV dumps or API updates.
*   **Tourism Data Processing**: Dedicated pipeline to ingest `.xlsx` files from `Data/Toursim Data/`, handling Arabic headers and converting to a standardized schema for the Data Warehouse.
*   **Entity Resolution**: Fuzzy matching algorithms to link `Projects.csv` (Developer Name) with `Transactions.csv` (Master Project) and clean inconsistent Arabic/English naming.
*   **Data Warehousing**: Load cleaned data into a structured SQL database (PostgreSQL) for querying and a Vector Database (Pinecone) for semantic search.

### 4.2 Quantitative Core ("The Symbolic Engine")
*   **Macro Forecasting**:
    *   Implement **Amazon Chronos** (or equivalent foundation model) for zero-shot time-series forecasting of community-level price trends.
*   **Micro Valuation**:
    *   Train **XGBoost/LightGBM** models on `Transactions.csv` + `Units.csv`.
    *   **Feature Engineering**: Create features for `Price/SqFt`, `Floor_Premium`, `Supply_Pressure_Index`.
*   **Sparse Data Handling**:
    *   Implement **LLMTime** for extrapolating trends in new projects with <12 months of data.

### 4.3 RERA Logic Engine
*   **Rent Cap Algorithm**: Hard-code Decree No. 43/2013 logic.
    *   Input: Current Rent vs. Market Average (from `Rent_Contracts.csv`).
    *   Output: Legally permissible rent increase (0%, 5%, 10%, 15%, 20%).
*   **ROI Calculator**:
    *   Implement Cash-on-Cash (CoC) return logic for off-plan properties considering payment plans.

### 4.4 Agentic Workflow ("The Neural Interface")
*   **Intent Recognition**: LLM (GPT-4/Claude 3.5) to parse user queries (e.g., separate "Budget" from "Location").
*   **Tool Usage**:
    *   **Text-to-SQL**: Convert natural language questions into SQL queries for the DLD database.
    *   **RAG System**: Retrieve relevant context from Docs and Private Client Data.
*   **Response Synthesis**: Combine SQL results, Model Forecasts, and RAG context into a coherent advisory response.

### 4.5 Security & Multi-Tenancy
*   **Vector Isolation**: Enforce strict namespace isolation in the Vector DB using `tenant_id`.
*   **Private Data Upload**: Secure endpoints for agents to upload client portfolios (CSV/PDF) which are then chunked and embedded into their specific namespace.

## 5. Technical Stack Recommendations
*   **Language**: Python (Data Science/Backend), TypeScript (Frontend).
*   **Database**: PostgreSQL with pgvector (Supabase).
*   **Storage**: Supabase Storage / AWS S3.
*   **ML Frameworks**: XGBoost, Amazon Chronos.
*   **LLM Orchestration**: LangChain or LangGraph.
*   **LLM Provider**: OpenAI (GPT-4) or Anthropic (Claude).
*   **Frontend**: Next.js / React.
*   **Hosting**: Vercel (Frontend), Railway (Backend).

## 6. Implementation Roadmap

See `Docs/models/00_OVERVIEW_ARCHITECTURE.md` for detailed phasing.

### Phase 0: Data Preprocessing & Aggregation (Weeks 1-2)

**Critical First Step**: The 9.5M row Rent_Contracts.csv (~4.2GB) must be aggregated to ~10MB of benchmark tables before main development begins. This enables Supabase Pro ($25/mo) instead of expensive enterprise databases.

**Aggregation Strategy:**
*   **Raw Data Kept**: Transactions, Units, Buildings, Projects (needed for ML training)
*   **Aggregated**: Rent_Contracts → `rent_benchmarks`, `rent_trends`, `vacancy_stats`

**ETL Validation Checklist:**

| Step | Validation | Action if Failed |
|------|------------|------------------|
| Parse bedrooms | "2 bed rooms+hall" → "2BR" | Log & skip row |
| Parse dates | "07-04-2019" → 2019-04-07 | Log & skip row |
| Validate rent | annual_amount > 0 and < 50,000,000 | Flag as outlier |
| Normalize area names | "Business Bay" = "BUSINESS BAY" | Entity resolution |
| Check actual_area | > 0 and < 100,000 sqft | Flag as outlier |
| Property type mapping | Standardize to enum | Log unknowns |

**Deliverables:**
*   `rent_benchmarks` table: RERA lookups, yield calculations
*   `rent_trends` table: Time series analysis, YoY/MoM changes
*   `vacancy_stats` table: Pre-computed void period analysis
*   Validation queries passing against sample manual calculations

### Phase 1: Foundation (Weeks 3-4)
*   Set up PostgreSQL schema and load cleaned data
*   Implement entity resolution algorithms
*   Build RERA calculator using aggregated `rent_benchmarks`
*   Process Tourism data (XLSX parser)

### Phase 2: Predictive Core (Weeks 5-10)
*   Train Property Valuation Model (XGBoost) - `01_PROPERTY_VALUATION_MODEL.md`
*   Integrate Time Series Forecaster (Chronos) - `02_TIME_SERIES_FORECASTING.md`
*   Implement ROI Calculator - `04_ROI_CALCULATOR_OFFPLAN.md`
*   Backtest against Expo 2020 period

### Phase 3: Intelligence Layer (Weeks 11-16)
*   Build Agentic Orchestrator (LangChain/LangGraph)
*   Implement Text-to-SQL interface
*   Integrate RAG pipeline with Vector DB
*   Multi-tenant namespace isolation

### Phase 4: Advanced Features (Weeks 17-20)
*   Supply Pressure Index - `05_SUPPLY_PRESSURE_INDEX.md`
*   Macro indicators integration (Oil, EIBOR)
*   Tourism data integration for STR yields
*   Production deployment

## 7. Model Documentation Index

| Model ID | Name | Document |
|---|---|---|
| 00 | System Architecture | `Docs/models/00_OVERVIEW_ARCHITECTURE.md` |
| 01 | Property Valuation | `Docs/models/01_PROPERTY_VALUATION_MODEL.md` |
| 02 | Time Series Forecasting | `Docs/models/02_TIME_SERIES_FORECASTING.md` |
| 03 | Rental Yield (RERA) | `Docs/models/03_RENTAL_YIELD_MODEL.md` |
| 04 | ROI Calculator (Off-Plan) | `Docs/models/04_ROI_CALCULATOR_OFFPLAN.md` |
| 05 | Supply Pressure Index | `Docs/models/05_SUPPLY_PRESSURE_INDEX.md` |
| 06 | Deployment Architecture | `Docs/models/06_DEPLOYMENT_ARCHITECTURE.md` |

