# Deployment Architecture: Full System Integration & Infrastructure

## 1. Overview

This document specifies the complete deployment architecture for the Dubai Real Estate AI Platform, including:
- How all models are combined and orchestrated
- Infrastructure and hosting decisions
- Frontend specification
- LLM integration layer
- API design and data flow

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CLIENT LAYER                                          │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         FRONTEND (Next.js / React)                               │    │
│  │                         Hosted: Vercel / AWS Amplify                             │    │
│  │                                                                                  │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐ │    │
│  │  │ Chat Interface  │  │ Dashboard       │  │ Private Data Upload             │ │    │
│  │  │ (Conversational │  │ (Visualizations │  │ (Client Portfolio Management)   │ │    │
│  │  │  AI Assistant)  │  │  & Reports)     │  │                                 │ │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │ HTTPS (REST + WebSocket)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    API GATEWAY                                           │
│                              AWS API Gateway / Kong                                      │
│                                                                                          │
│  • Rate Limiting          • Authentication (JWT)          • Request Routing             │
│  • CORS Handling          • API Key Management            • Load Balancing              │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                                           │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                     BACKEND API (FastAPI / Python)                               │    │
│  │                     Hosted: AWS ECS (Fargate) or Lambda                          │    │
│  │                                                                                  │    │
│  │  Endpoints:                                                                      │    │
│  │  • POST /api/chat              → Conversational AI queries                       │    │
│  │  • POST /api/valuation         → Direct valuation requests                       │    │
│  │  • POST /api/forecast          → Time series forecasts                           │    │
│  │  • POST /api/roi               → ROI calculations                                │    │
│  │  • POST /api/upload            → Private data upload                             │    │
│  │  • GET  /api/market/{area}     → Market overview                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                                │
│                                         ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                   AGENTIC ORCHESTRATOR (LangChain / LangGraph)                   │    │
│  │                                                                                  │    │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────────┐   │    │
│  │  │ Intent Parser │───▶│ Query Router  │───▶│ Response Synthesizer          │   │    │
│  │  │ (LLM Call)    │    │ (Tool Select) │    │ (LLM Call + SHAP)             │   │    │
│  │  └───────────────┘    └───────────────┘    └───────────────────────────────┘   │    │
│  │          │                    │                         ▲                       │    │
│  │          ▼                    ▼                         │                       │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐   │    │
│  │  │                         TOOL REGISTRY                                    │   │    │
│  │  │                                                                          │   │    │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │   │    │
│  │  │  │ Text-to-SQL │ │ RAG Search  │ │ Valuation   │ │ Time Series     │   │   │    │
│  │  │  │             │ │             │ │ Model       │ │ Forecast        │   │   │    │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘   │   │    │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────────────┐   │   │    │
│  │  │  │ Yield Calc  │ │ ROI Calc    │ │ Supply Index                    │   │   │    │
│  │  │  │             │ │             │ │                                 │   │   │    │
│  │  │  └─────────────┘ └─────────────┘ └─────────────────────────────────┘   │   │    │
│  │  └─────────────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   LLM PROVIDER          │ │   ML MODEL SERVING      │ │   EXTERNAL APIS         │
│                         │ │                         │ │                         │
│ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │
│ │ OpenAI API          │ │ │ │ Railway / Render    │ │ │ │ FRED API            │ │
│ │ (GPT-4 Turbo)       │ │ │ │ or SageMaker        │ │ │ │ (Oil Prices)        │ │
│ │                     │ │ │ │                     │ │ │ │                     │ │
│ │ OR                  │ │ │ │ • XGBoost Model     │ │ │ │ CBUAE               │ │
│ │                     │ │ │ │ • Chronos Model     │ │ │ │ (EIBOR Rates)       │ │
│ │ Anthropic API       │ │ │ │                     │ │ │ │                     │ │
│ │ (Claude 3.5 Sonnet) │ │ │ │                     │ │ │ │                     │ │
│ └─────────────────────┘ │ │ └─────────────────────┘ │ │ └─────────────────────┘ │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
              │                          │                          │
              └──────────────────────────┴──────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA LAYER                                            │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                      PRIMARY DATABASE (PostgreSQL)                               │    │
│  │                      Hosted: AWS RDS / Supabase                                  │    │
│  │                                                                                  │    │
│  │  Tables:                                                                         │    │
│  │  • transactions    • units         • buildings      • projects                   │    │
│  │  • rent_contracts  • valuations    • tourism_stats  • macro_indicators           │    │
│  │  • users           • tenants       • audit_logs                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ VECTOR DATABASE (pgvector extension in PostgreSQL)                              │    │
│  │                                                                                  │    │
│  │ • RAG Documents     • Private Client Data (namespaced)    • Market Reports      │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         OBJECT STORAGE (AWS S3 / Supabase Storage)               │    │
│  │                                                                                  │    │
│  │  • Raw CSV uploads          • Trained model artifacts (.pkl, .pt)               │    │
│  │  • Tourism XLSX files       • FDI PDF reports          • Backup data            │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              ETL & BACKGROUND JOBS                                       │
│                              (Apache Airflow on AWS MWAA)                                │
│                                                                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐     │
│  │ DAG: DLD Ingestion  │  │ DAG: Macro Update   │  │ DAG: Model Retrain          │     │
│  │ Schedule: Daily     │  │ Schedule: Weekly    │  │ Schedule: Monthly           │     │
│  │                     │  │                     │  │                             │     │
│  │ • Fetch new CSVs    │  │ • FRED API (Oil)    │  │ • Retrain XGBoost           │     │
│  │ • Entity resolution │  │ • CBUAE (EIBOR)     │  │ • Validate metrics          │     │
│  │ • Load to Postgres  │  │ • Tourism XLSX      │  │ • Deploy new model          │     │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Frontend (Client Layer)

| Attribute | Specification |
|---|---|
| **Framework** | Next.js 14+ (App Router) with React 18 |
| **UI Library** | shadcn/ui + Tailwind CSS |
| **State Management** | Zustand or React Query |
| **Charts** | Recharts or Tremor |
| **Hosting** | Vercel (recommended) or AWS Amplify |
| **Authentication** | NextAuth.js with JWT |

**Key Pages:**

```
/                       → Landing page
/login                  → Authentication
/dashboard              → Main dashboard with market overview
/chat                   → Conversational AI interface
/properties/[id]        → Individual property analysis
/portfolio              → Private client portfolio management
/settings               → User settings, API keys
```

**Chat Interface Component:**

```typescript
// components/ChatInterface.tsx
'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const chatMutation = useMutation({
    mutationFn: async (query: string) => {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history: messages }),
      });
      return response.json();
    },
    onSuccess: (data) => {
      setMessages(prev => [...prev, 
        { role: 'user', content: input },
        { role: 'assistant', content: data.response, metadata: data.metadata }
      ]);
    },
  });

  return (
    <div className="flex flex-col h-full">
      <MessageList messages={messages} />
      <ChatInput 
        value={input}
        onChange={setInput}
        onSubmit={() => chatMutation.mutate(input)}
        isLoading={chatMutation.isPending}
      />
    </div>
  );
}
```

---

### 3.2 Backend API (Application Layer)

| Attribute | Specification |
|---|---|
| **Framework** | FastAPI (Python 3.11+) |
| **Hosting** | AWS ECS Fargate (containerized) |
| **Container** | Docker with Python slim base |
| **Scaling** | Auto-scaling based on CPU/memory |
| **Concurrency** | Uvicorn with multiple workers |

**API Structure:**

```
backend/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py         # POST /api/chat
│   │   │   ├── valuation.py    # POST /api/valuation
│   │   │   ├── forecast.py     # POST /api/forecast
│   │   │   ├── roi.py          # POST /api/roi
│   │   │   └── upload.py       # POST /api/upload
│   │   └── deps.py             # Dependencies (auth, db)
│   ├── core/
│   │   ├── config.py           # Settings from environment
│   │   ├── security.py         # JWT handling
│   │   └── database.py         # DB connections
│   ├── services/
│   │   ├── orchestrator.py     # LangChain agent
│   │   ├── tools/              # Tool implementations
│   │   │   ├── valuation.py
│   │   │   ├── time_series.py
│   │   │   ├── yield_calc.py
│   │   │   ├── roi_calc.py
│   │   │   ├── supply_index.py
│   │   │   ├── text_to_sql.py
│   │   │   └── rag_search.py
│   │   └── ml_clients/         # Model inference clients
│   └── models/                 # Pydantic schemas
├── tests/
├── Dockerfile
└── requirements.txt
```

**Main API Entry Point:**

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, valuation, forecast, roi, upload
from app.core.config import settings

app = FastAPI(
    title="Dubai Real Estate AI API",
    version="1.0.0",
    description="AI-powered property investment analysis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(valuation.router, prefix="/api", tags=["Valuation"])
app.include_router(forecast.router, prefix="/api", tags=["Forecast"])
app.include_router(roi.router, prefix="/api", tags=["ROI"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

---

### 3.3 LLM Orchestration Layer

| Attribute | Specification |
|---|---|
| **Framework** | LangChain 0.2+ or LangGraph |
| **Primary LLM** | GPT-4 Turbo (OpenAI) or Claude 3.5 Sonnet (Anthropic) |
| **Fallback LLM** | GPT-3.5 Turbo (cost optimization) |
| **Tool Calling** | OpenAI Function Calling / Anthropic Tool Use |

**Orchestrator Implementation:**

```python
# app/services/orchestrator.py
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from app.services.tools import (
    valuation_tool,
    time_series_tool,
    yield_calculator_tool,
    roi_calculator_tool,
    supply_index_tool,
    text_to_sql_tool,
    rag_search_tool,
)
from app.core.config import settings

class PropertyAIOrchestrator:
    """
    Agentic orchestrator that routes user queries to appropriate tools.
    """
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Register all tools
        self.tools = [
            Tool(
                name="property_valuation",
                func=valuation_tool.run,
                description="""
                Get current market value for a property unit.
                Input should be JSON with: area_name, building_name, bedrooms, floor, area_sqft
                Returns: estimated value in AED with confidence interval
                """
            ),
            Tool(
                name="price_forecast",
                func=time_series_tool.run,
                description="""
                Forecast future price trends for an area.
                Input should be JSON with: area_name, horizon_months (default 12)
                Returns: price forecast with confidence intervals
                """
            ),
            Tool(
                name="rental_yield",
                func=yield_calculator_tool.run,
                description="""
                Calculate rental yield with RERA compliance.
                Input should be JSON with: property_value, current_rent (optional), area_name, bedrooms
                Returns: gross yield, net yield, RERA-capped rent projections
                """
            ),
            Tool(
                name="offplan_roi",
                func=roi_calculator_tool.run,
                description="""
                Calculate ROI for off-plan property investment.
                Input should be JSON with: purchase_price, payment_plan_type, construction_months, appreciation_scenario
                Returns: Cash-on-Cash ROI, IRR, breakeven analysis
                """
            ),
            Tool(
                name="supply_pressure",
                func=supply_index_tool.run,
                description="""
                Get supply pressure index for an area.
                Input should be: area_name
                Returns: SPI value, risk level, upcoming supply details
                """
            ),
            Tool(
                name="database_query",
                func=text_to_sql_tool.run,
                description="""
                Query the property database for specific data.
                Input should be a natural language question about transactions, projects, or market data.
                Returns: query results as structured data
                """
            ),
            Tool(
                name="document_search",
                func=rag_search_tool.run,
                description="""
                Search private client documents and market reports.
                Input should be a search query.
                Returns: relevant document excerpts
                """
            ),
        ]
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Dubai real estate investment advisor AI.
            
Your role is to help real estate agents analyze properties and provide investment advice.

IMPORTANT RULES:
1. NEVER make up numbers. Always use tools to get accurate data.
2. When asked about valuations, forecasts, or yields, ALWAYS use the appropriate tool.
3. Explain your reasoning and cite the tool outputs.
4. Be specific about risks and uncertainties.
5. Always mention RERA compliance when discussing rental yields.
6. Format currency values in AED with thousands separators.

Available tools: property_valuation, price_forecast, rental_yield, offplan_roi, 
supply_pressure, database_query, document_search
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=10
        )
    
    async def process_query(
        self, 
        query: str, 
        chat_history: list = None,
        tenant_id: str = None
    ) -> dict:
        """
        Process a user query through the agentic workflow.
        
        Args:
            query: User's natural language question
            chat_history: Previous conversation messages
            tenant_id: For multi-tenant RAG isolation
        
        Returns:
            dict with response, metadata, and tool outputs
        """
        # Set tenant context for RAG
        if tenant_id:
            rag_search_tool.set_namespace(tenant_id)
        
        # Run agent
        result = await self.agent_executor.ainvoke({
            "input": query,
            "chat_history": chat_history or []
        })
        
        # Extract tool outputs for transparency
        tool_outputs = []
        for step in result.get("intermediate_steps", []):
            action, output = step
            tool_outputs.append({
                "tool": action.tool,
                "input": action.tool_input,
                "output": output
            })
        
        return {
            "response": result["output"],
            "tool_outputs": tool_outputs,
            "tokens_used": self._count_tokens(result)
        }
```

---

### 3.4 ML Model Serving

| Model | Hosting | Instance Type | Scaling |
|---|---|---|---|
| XGBoost (Valuation) | Railway/Render or SageMaker | CPU (containerized) | Auto (1-2 instances) |
| Chronos (Time Series) | Railway/Render or SageMaker | CPU or GPU optional | Auto (1-2 instances) |

**SageMaker Client Example:**

```python
# app/services/ml_clients/valuation_client.py
import boto3
import json
from app.core.config import settings

class ValuationModelClient:
    """Client for XGBoost valuation model on SageMaker."""
    
    def __init__(self):
        self.runtime = boto3.client(
            'sagemaker-runtime',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY
        )
        self.endpoint_name = settings.VALUATION_ENDPOINT_NAME
    
    def predict(self, features: dict) -> dict:
        """
        Get valuation prediction from SageMaker endpoint.
        
        Args:
            features: Dict with property features
        
        Returns:
            Predicted value and confidence interval
        """
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(features)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        return {
            "predicted_value_aed": result['prediction'],
            "confidence_interval": result['confidence_interval'],
            "feature_importance": result.get('shap_values', {})
        }
```

---

### 3.5 Database Schema

**PostgreSQL Tables:**

```sql
-- Core DLD Data
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE,
    instance_date DATE,
    actual_worth DECIMAL(15,2),
    meter_sale_price DECIMAL(10,2),
    area_name_en VARCHAR(100),
    master_project_en VARCHAR(200),
    rooms_en VARCHAR(20),
    property_type_en VARCHAR(50),
    reg_type_en VARCHAR(50),
    trans_group_en VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_transactions_area ON transactions(area_name_en);
CREATE INDEX idx_transactions_date ON transactions(instance_date);

CREATE TABLE units (
    id SERIAL PRIMARY KEY,
    property_id VARCHAR(50) UNIQUE,
    area_name_en VARCHAR(100),
    floor INTEGER,
    actual_area DECIMAL(10,2),
    unit_balcony_area DECIMAL(10,2),
    unit_parking_number VARCHAR(20),
    rooms_en VARCHAR(20),
    project_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(50) UNIQUE,
    project_name VARCHAR(200),
    developer_name VARCHAR(200),
    percent_completed INTEGER,
    completion_date DATE,
    no_of_units INTEGER,
    area_name_en VARCHAR(100),
    escrow_agent_name VARCHAR(200),
    project_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Feature Store Tables
CREATE TABLE supply_pressure_index (
    id SERIAL PRIMARY KEY,
    area_name VARCHAR(100),
    computed_date DATE,
    spi_value DECIMAL(5,2),
    projected_supply INTEGER,
    avg_transactions INTEGER,
    risk_level VARCHAR(20),
    UNIQUE(area_name, computed_date)
);

-- Aggregated Rent Tables (from Rent_Contracts.csv aggregation)
-- These replace the 9.5M row raw data with ~65K rows of benchmarks

-- 1. rent_benchmarks: For RERA lookups & yield calculations (~50K rows)
CREATE TABLE rent_benchmarks (
    id SERIAL PRIMARY KEY,
    
    -- Dimensions
    area_name VARCHAR(100) NOT NULL,
    property_type VARCHAR(50) NOT NULL,      -- Residential/Commercial
    property_subtype VARCHAR(100),           -- Villa/Apartment/Office
    bedrooms VARCHAR(20) NOT NULL,           -- Studio, 1BR, 2BR, 3BR, 4BR+
    year_month DATE NOT NULL,
    
    -- Rent Metrics
    median_annual_rent DECIMAL(12,2),
    avg_annual_rent DECIMAL(12,2),
    p25_annual_rent DECIMAL(12,2),           -- 25th percentile
    p75_annual_rent DECIMAL(12,2),           -- 75th percentile
    min_annual_rent DECIMAL(12,2),
    max_annual_rent DECIMAL(12,2),
    
    -- Per SqFt Metrics
    median_rent_per_sqft DECIMAL(8,2),
    avg_rent_per_sqft DECIMAL(8,2),
    
    -- Volume
    contract_count INT,
    new_contract_count INT,                   -- New leases
    renewal_count INT,                        -- Renewals
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE (area_name, property_type, bedrooms, year_month)
);

CREATE INDEX idx_rent_benchmarks_lookup ON rent_benchmarks(area_name, bedrooms, year_month);

-- 2. rent_trends: For time series analysis (~5K rows)
CREATE TABLE rent_trends (
    id SERIAL PRIMARY KEY,
    
    area_name VARCHAR(100) NOT NULL,
    year_month DATE NOT NULL,
    
    -- Overall metrics (all property types combined)
    median_rent_per_sqft DECIMAL(8,2),
    total_contracts INT,
    total_new_leases INT,
    total_renewals INT,
    
    -- Calculated change metrics
    mom_change_pct DECIMAL(5,2),             -- Month-over-month
    yoy_change_pct DECIMAL(5,2),             -- Year-over-year
    
    UNIQUE (area_name, year_month)
);

CREATE INDEX idx_rent_trends_area ON rent_trends(area_name, year_month);

-- 3. vacancy_stats: For void period analysis (~10K rows)
CREATE TABLE vacancy_stats (
    id SERIAL PRIMARY KEY,
    
    area_name VARCHAR(100) NOT NULL,
    building_name VARCHAR(200),               -- NULL for area-level aggregates
    property_type VARCHAR(50),
    bedrooms VARCHAR(20),
    year INT NOT NULL,
    
    -- Void metrics (calculated from gaps between contracts)
    avg_void_days DECIMAL(6,1),
    median_void_days DECIMAL(6,1),
    max_void_days INT,
    
    -- Derived vacancy rate
    vacancy_rate_pct DECIMAL(5,2),            -- avg_void_days / 365 * 100
    
    -- Sample size
    turnover_events_analyzed INT,
    
    UNIQUE (area_name, COALESCE(building_name, ''), property_type, bedrooms, year)
);

CREATE INDEX idx_vacancy_stats_area ON vacancy_stats(area_name, year);
```

### 3.5.1 ETL Validation Rules (Phase 0)

Before aggregating Rent_Contracts into the benchmark tables, apply these validation rules:

```python
# etl/validators.py

import re
import pandas as pd
from datetime import datetime

class RentContractValidator:
    """Validation rules for Rent_Contracts ETL."""
    
    BEDROOM_PATTERNS = {
        r'studio': 'Studio',
        r'1\s*bed|1\s*br|1br': '1BR',
        r'2\s*bed|2\s*br|2br': '2BR',
        r'3\s*bed|3\s*br|3br': '3BR',
        r'4\s*bed|4\s*br|4br|4\+': '4BR+',
        r'5\s*bed|5\s*br|5br': '4BR+',  # Group 5+ as 4BR+
    }
    
    def parse_bedrooms(self, property_subtype: str) -> str:
        """Parse bedroom count from ejari_property_sub_type_en."""
        if pd.isna(property_subtype):
            return 'Unknown'
        
        text = str(property_subtype).lower()
        for pattern, label in self.BEDROOM_PATTERNS.items():
            if re.search(pattern, text):
                return label
        return 'Unknown'
    
    def parse_date(self, date_str: str) -> datetime | None:
        """Parse dates in DD-MM-YYYY format."""
        if pd.isna(date_str):
            return None
        try:
            return datetime.strptime(str(date_str), '%d-%m-%Y')
        except ValueError:
            return None
    
    def validate_rent(self, annual_amount: float) -> tuple[bool, str]:
        """Validate rent amount is reasonable."""
        if pd.isna(annual_amount):
            return False, "null_value"
        if annual_amount <= 0:
            return False, "negative_or_zero"
        if annual_amount > 50_000_000:  # 50M AED max
            return False, "extreme_outlier"
        if annual_amount < 1_000:  # Less than 1K annual
            return False, "too_low"
        return True, "valid"
    
    def validate_area(self, actual_area: float) -> tuple[bool, str]:
        """Validate property area is reasonable."""
        if pd.isna(actual_area):
            return True, "null_ok"  # Optional field
        if actual_area <= 0:
            return False, "negative_or_zero"
        if actual_area > 100_000:  # 100K sqft max
            return False, "extreme_outlier"
        return True, "valid"
```

**ETL Validation Checklist:**

| Step | Validation | Action if Failed |
|------|------------|------------------|
| 1. Parse bedrooms | "2 bed rooms+hall" → "2BR" | Log & categorize as "Unknown" |
| 2. Parse dates | "07-04-2019" → 2019-04-07 | Log & skip row |
| 3. Validate rent | 0 < annual_amount < 50M | Flag as outlier, exclude from aggregates |
| 4. Normalize area names | "Business Bay" = "BUSINESS BAY" | Apply entity resolution mapping |
| 5. Check actual_area | 0 < area < 100,000 sqft | Flag as outlier |
| 6. Property type mapping | Standardize to enum | Log unknowns |

**Validation Queries (Post-Aggregation):**

```sql
-- Verify RERA lookups work
SELECT median_annual_rent, contract_count FROM rent_benchmarks
WHERE area_name = 'Dubai Marina' AND bedrooms = '2BR'
ORDER BY year_month DESC LIMIT 1;

-- Verify trend data
SELECT year_month, yoy_change_pct FROM rent_trends
WHERE area_name = 'Business Bay'
ORDER BY year_month DESC LIMIT 12;

-- Verify vacancy stats
SELECT vacancy_rate_pct FROM vacancy_stats
WHERE area_name = 'Jumeirah Beach Residence' AND year = 2023;
```

```sql
-- Multi-tenant User Data
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200),
    email VARCHAR(200) UNIQUE,
    api_key_hash VARCHAR(256),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    action VARCHAR(50),
    query TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### 3.6 Multi-Tenant RAG Architecture

```python
# app/services/tools/rag_search.py
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class RAGSearchTool:
    """
    Multi-tenant RAG search with namespace isolation.
    """
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings()
        self.current_namespace = "public"  # Default to public docs
    
    def set_namespace(self, tenant_id: str):
        """Set tenant namespace for isolation."""
        self.current_namespace = f"tenant_{tenant_id}"
    
    def run(self, query: str) -> str:
        """
        Search documents in tenant's namespace.
        """
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in tenant namespace + public namespace
        results = self.index.query(
            vector=query_embedding,
            top_k=5,
            namespace=self.current_namespace,
            include_metadata=True
        )
        
        # Also search public namespace
        public_results = self.index.query(
            vector=query_embedding,
            top_k=3,
            namespace="public",
            include_metadata=True
        )
        
        # Combine and format results
        all_results = results['matches'] + public_results['matches']
        
        formatted = []
        for match in sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]:
            formatted.append({
                "content": match['metadata'].get('text', ''),
                "source": match['metadata'].get('source', 'Unknown'),
                "score": match['score']
            })
        
        return json.dumps(formatted)
    
    def upload_document(self, tenant_id: str, document: str, metadata: dict):
        """
        Upload document to tenant's isolated namespace.
        """
        namespace = f"tenant_{tenant_id}"
        
        # Chunk document
        chunks = self._chunk_document(document)
        
        # Embed and upsert
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk)
            self.index.upsert(
                vectors=[{
                    "id": f"{tenant_id}_{metadata['doc_id']}_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "source": metadata.get('filename', 'Unknown'),
                        "tenant_id": tenant_id
                    }
                }],
                namespace=namespace
            )
```

---

## 4. Infrastructure Costs Estimate

| Component | Service | Monthly Cost (Est.) |
|---|---|---|
| Frontend Hosting | Vercel Pro | $20 |
| Backend + ML Models | Railway | $50 |
| Database + Auth + Storage | Supabase Pro | $25 |
| OpenAI API | GPT-4 (~50k tokens/day) | $200-400 |
| **Total** | | **~$300-500/month** |

*Note: This simplified stack avoids expensive GPU inference and managed ML services. Models run in containerized Python on Railway.*

---

## 5. Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SECURITY LAYERS                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 1: Network Security                                │    │
│  │ • AWS VPC with private subnets                          │    │
│  │ • WAF rules on API Gateway                               │    │
│  │ • DDoS protection (AWS Shield)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 2: Authentication                                  │    │
│  │ • JWT tokens (RS256 signed)                              │    │
│  │ • API key authentication for programmatic access         │    │
│  │ • OAuth 2.0 for SSO integration                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 3: Data Isolation                                  │    │
│  │ • Tenant namespace isolation in Pinecone                 │    │
│  │ • Row-level security in PostgreSQL                       │    │
│  │ • Encrypted at rest (AES-256)                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 4: Audit & Compliance                              │    │
│  │ • All queries logged with tenant_id                      │    │
│  │ • Token usage tracking                                   │    │
│  │ • GDPR-compliant data handling                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Deployment Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD PIPELINE (GitHub Actions)               │
│                                                                  │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐ │
│  │ Push to   │───▶│ Run Tests │───▶│ Build     │───▶│ Deploy │ │
│  │ main      │    │ (pytest)  │    │ Docker    │    │ to ECS │ │
│  └───────────┘    └───────────┘    └───────────┘    └────────┘ │
│                                                                  │
│  Environments:                                                   │
│  • dev.properly.ai    → Development (auto-deploy on PR merge)   │
│  • staging.properly.ai → Staging (manual promotion)             │
│  • app.properly.ai    → Production (manual promotion + approval)│
└─────────────────────────────────────────────────────────────────┘
```

**GitHub Actions Workflow:**

```yaml
# .github/workflows/deploy.yml
name: Deploy Backend

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: me-south-1  # Bahrain (closest to Dubai)
      - name: Build and push Docker image
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker build -t properly-backend .
          docker tag properly-backend:latest $ECR_REGISTRY/properly-backend:latest
          docker push $ECR_REGISTRY/properly-backend:latest
      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster properly-cluster --service properly-api --force-new-deployment
```

---

## 7. Monitoring & Observability

| Component | Tool | Purpose |
|---|---|---|
| Application Logs | AWS CloudWatch | Error tracking, query logs |
| Metrics | Prometheus + Grafana | Latency, throughput, model performance |
| Tracing | AWS X-Ray | Request tracing across services |
| Alerts | PagerDuty / Slack | On-call notifications |
| LLM Monitoring | LangSmith | Token usage, chain debugging |

**Key Metrics Dashboard:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING DASHBOARD                          │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Request Latency     │  │ Model Accuracy (Weekly Backtest) │   │
│  │ P50: 1.2s          │  │ Valuation MAPE: 8.3%             │   │
│  │ P95: 3.8s          │  │ Forecast MAPE: 12.1%             │   │
│  │ P99: 7.2s          │  │                                   │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Token Usage (Daily) │  │ Error Rate                       │   │
│  │ GPT-4: 85,000      │  │ 4xx: 0.3%                        │   │
│  │ Cost: $2.55        │  │ 5xx: 0.1%                        │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Active Users: 45  │ Queries Today: 1,234  │ Avg/User: 27│    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Summary

| Layer | Technology | Key Decisions |
|---|---|---|
| **Frontend** | Next.js + React | Modern, fast, SSR for SEO |
| **Backend** | FastAPI (Python) on Railway | Async, fast, ML-friendly, affordable |
| **Orchestration** | LangChain/LangGraph | Agentic workflow, tool calling |
| **LLM** | GPT-4 / Claude 3.5 | Best-in-class reasoning |
| **ML Serving** | Containerized on Railway | Simple deployment, co-located with API |
| **Primary DB** | PostgreSQL (Supabase) | Relational data, SQL queries, Auth |
| **Vector DB** | pgvector (Supabase) | RAG, multi-tenant isolation |
| **Storage** | Supabase Storage / S3 | Raw files, model artifacts |
| **CI/CD** | GitHub Actions | Automated deployments |
| **Monitoring** | Vercel Analytics + Railway Logs | Basic observability |

