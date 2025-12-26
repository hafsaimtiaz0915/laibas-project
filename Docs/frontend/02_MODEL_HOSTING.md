# Model Hosting & Backend Deployment

> **Document Version**: 1.0  
> **Last Updated**: 2025-12-11  
> **Purpose**: How to host the TFT model after training

---

## Overview

After training the TFT model in Google Colab, the model checkpoint (~200MB) needs to be hosted for inference. This document covers three deployment options.

---

## Option A: Local Mac Development (Free)

Best for: Development, testing, personal use

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR MAC                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FastAPI Backend (localhost:8000)             │   │
│  │                                                           │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │   │
│  │  │ TFT Model  │  │ Lookup     │  │ Claude API         │  │   │
│  │  │ .ckpt      │  │ CSVs       │  │ (external)         │  │   │
│  │  │ ~200MB     │  │ ~50MB      │  │                    │  │   │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │   │
│  │                                                           │   │
│  │  POST /api/predict  → TFT inference                       │   │
│  │  POST /api/chat     → Full chat pipeline                  │   │
│  │  POST /api/report   → PDF generation                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▲                                   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Next.js Frontend (localhost:3000)               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Setup Steps

```bash
# 1. Create backend directory
mkdir -p backend/app backend/models backend/data

# 2. Download model from Colab
# In Colab, run:
#   from google.colab import files
#   files.download('/content/drive/MyDrive/Properly/tft_final.ckpt')

# 3. Move to backend
mv ~/Downloads/tft_final.ckpt backend/models/

# 4. Install dependencies
cd backend
pip install fastapi uvicorn pytorch-forecasting anthropic pandas

# 5. Run server
uvicorn app.main:app --reload --port 8000
```

### Cost

| Item | Cost |
|------|------|
| Hosting | Free |
| Claude API | ~$10-20/month (usage-based) |
| **Total** | **~$10-20/month** |

### Pros/Cons

| Pros | Cons |
|------|------|
| ✅ Free hosting | ❌ Mac must be running |
| ✅ Fast iteration | ❌ Only accessible locally |
| ✅ No deployment complexity | ❌ Not suitable for real users |

---

## Option B: Railway Deployment (Recommended for Production)

Best for: Production, real users, team access

### Architecture

```
┌────────────────────┐         ┌─────────────────────────────────┐
│     Vercel         │         │     Railway ($5-20/month)       │
│     (Frontend)     │         │     (Backend)                   │
│                    │  HTTPS  │                                 │
│   Next.js App      │────────▶│   Docker Container:             │
│   21st.dev UI      │         │   ├── FastAPI                   │
│                    │         │   ├── TFT Model (.ckpt)         │
│   Free tier        │         │   ├── Lookup CSVs               │
│                    │         │   └── Python 3.11               │
└────────────────────┘         └─────────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────────┐
                               │    Supabase ($25/month)         │
                               │                                 │
                               │    ├── PostgreSQL               │
                               │    │   └── Agent accounts       │
                               │    │   └── Query history        │
                               │    │   └── Report metadata      │
                               │    │                            │
                               │    ├── Auth                     │
                               │    │   └── Agent login          │
                               │    │                            │
                               │    └── Storage                  │
                               │        └── Agent logos          │
                               └─────────────────────────────────┘
```

### Dockerfile

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```txt
fastapi==0.104.1
uvicorn==0.24.0
pytorch-forecasting==1.5.0
torch==2.1.0
pandas==2.1.3
numpy==1.26.2
anthropic==0.7.7
python-multipart==0.0.6
```

### Deployment Steps

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
cd backend
railway init

# 4. Deploy
railway up

# 5. Get deployment URL
railway domain
# Example: properly-backend.up.railway.app
```

### Environment Variables (Railway Dashboard)

```
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
```

### Cost

| Service | Cost/Month |
|---------|------------|
| Railway (Backend) | $5-20 |
| Vercel (Frontend) | Free |
| Supabase (DB + Auth) | $25 |
| Claude API | $20-50 |
| **Total** | **$50-95/month** |

### Pros/Cons

| Pros | Cons |
|------|------|
| ✅ Always available | ❌ Monthly cost |
| ✅ Real user access | ❌ Deployment complexity |
| ✅ Scalable | ❌ Cold starts possible |
| ✅ HTTPS included | |

---

## Option C: Hybrid Approach (Best Practice)

Use both: Local for development, Railway for production.

### Workflow

```
Development (Local Mac):
├── Fast iteration
├── No deployment wait
├── Free
└── Use: npm run dev

Production (Railway):
├── Real users
├── Always available
├── Monitored
└── Use: railway up
```

### Environment Switching

```typescript
// frontend/lib/api.ts
const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://properly-backend.up.railway.app'
  : 'http://localhost:8000';
```

---

## Backend API Specification

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/predict` | POST | TFT model inference |
| `/api/chat` | POST | Full chat pipeline |
| `/api/report/pdf` | POST | Generate PDF report |
| `/api/agent/settings` | GET/PUT | Agent settings |
| `/api/agent/logo` | POST | Upload logo |
| `/health` | GET | Health check |

### Request/Response Examples

#### POST /api/predict

```json
// Request
{
  "developer": "Binghatti",
  "area": "JVC",
  "bedroom": "2BR",
  "price": 2200000,
  "property_type": "Unit"
}

// Response
{
  "handover_value": {
    "low": 2650000,
    "median": 2780000,
    "high": 2910000
  },
  "appreciation": {
    "percent_low": 20.5,
    "percent_median": 26.4,
    "percent_high": 32.3
  },
  "rental_yield": {
    "gross": 6.1,
    "net": 5.2,
    "annual_rent": 170000
  },
  "time_horizon_months": 24
}
```

#### POST /api/chat

```json
// Request
{
  "query": "Binghatti JVC 2BR at 2.2M - what's the outlook?",
  "agent_id": "agent_123"
}

// Response
{
  "response": "Based on current market data, here's the analysis...",
  "report": {
    "query": {
      "developer": "Binghatti",
      "area": "JVC",
      "bedroom": "2BR",
      "purchasePrice": 2200000
    },
    "predictions": { ... },
    "trends": {
      "developer": {
        "projectsCompleted": 12,
        "avgDelayMonths": 4,
        "totalUnitsDelivered": 8500
      },
      "area": {
        "priceChange12Months": 18.5,
        "priceChange36Months": 42.3,
        "supplyPipeline": 12000
      }
    }
  }
}
```

---

## Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py          # /api/predict
│   │   ├── chat.py             # /api/chat
│   │   ├── report.py           # /api/report
│   │   └── agent.py            # /api/agent
│   └── services/
│       ├── __init__.py
│       ├── tft_inference.py    # Model loading & prediction
│       ├── llm_orchestrator.py # Claude integration
│       ├── trend_lookup.py     # CSV lookups
│       └── pdf_generator.py    # Report generation
│
├── models/
│   └── tft_final.ckpt          # Trained TFT model (~200MB)
│
├── data/
│   ├── developer_stats.csv     # Developer lookup
│   ├── area_medians.csv        # Area price medians
│   └── rent_benchmarks.csv     # Rent data
│
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Model Loading (Startup)

```python
# app/services/tft_inference.py

import torch
from pytorch_forecasting import TemporalFusionTransformer
from functools import lru_cache

MODEL_PATH = "models/tft_final.ckpt"

@lru_cache(maxsize=1)
def load_model():
    """
    Load TFT model once at startup.
    Uses CPU inference (no GPU required).
    """
    print("Loading TFT model...")
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    model.eval()
    model.to("cpu")
    print("Model loaded successfully")
    return model

def predict(developer: str, area: str, bedroom: str, price: float):
    """
    Run TFT inference.
    """
    model = load_model()
    
    # Prepare input (format depends on training data structure)
    input_data = prepare_input_data(developer, area, bedroom, price)
    
    with torch.no_grad():
        predictions = model.predict(input_data, mode="prediction")
    
    return {
        "handover_value": {
            "low": float(predictions.quantile(0.1)),
            "median": float(predictions.quantile(0.5)),
            "high": float(predictions.quantile(0.9)),
        }
    }
```

---

## Performance Considerations

### CPU Inference Speed

| Batch Size | Inference Time |
|------------|----------------|
| 1 (single query) | ~100-500ms |
| 10 | ~500-1000ms |
| 100 | ~2-5s |

**Note:** CPU inference is fast enough for real-time queries. No GPU needed.

### Memory Usage

| Component | Memory |
|-----------|--------|
| TFT Model | ~500MB |
| Lookup Data | ~100MB |
| Python Runtime | ~200MB |
| **Total** | **~800MB-1GB** |

Railway's smallest plan (512MB) may be tight. Recommend 1GB+ plan.

---

## Security Checklist

- [ ] API keys in environment variables (not code)
- [ ] CORS configured for frontend domain only
- [ ] Rate limiting on /api/chat endpoint
- [ ] Input validation on all endpoints
- [ ] HTTPS only (Railway provides this)
- [ ] Agent authentication before API access

---

## Monitoring

### Railway Dashboard

- CPU usage
- Memory usage
- Request count
- Error rate

### Application Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    logger.info(f"Chat query: {request.query[:50]}...")
    # ...
    logger.info(f"Response generated in {elapsed_ms}ms")
```

---

## Next Steps

1. **Train TFT model** (Colab)
2. **Download checkpoint** to `backend/models/`
3. **Create lookup CSVs** from cleaned data
4. **Build FastAPI backend** locally
5. **Test endpoints** with curl/Postman
6. **Deploy to Railway** when ready for users

