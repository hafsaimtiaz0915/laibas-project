# LLM Integration Documentation

> **Last Updated**: 2025-12-11  
> **Version**: 2.0

---

## Overview

This folder contains the plan and code for integrating OpenAI API with TFT model and Supabase for a natural language property analysis chat interface.

---

## Documents

| Document | Description |
|----------|-------------|
| [00_LLM_INTEGRATION_PLAN.md](./00_LLM_INTEGRATION_PLAN.md) | Full integration plan, architecture, 9 phases |
| [01_LLM_SERVICE_CODE.md](./01_LLM_SERVICE_CODE.md) | Complete Python code for all services |

---

## Architecture

```
User Query → Parse (OpenAI) → Validate (Arabic) → TFT Predict → Generate Response
                    ↓                                    ↓
              Supabase                            Supabase
           (Chat History)                      (Lookup Tables)
```

---

## Key Components

| Service | Purpose |
|---------|---------|
| `QueryParser` | Extract entities using OpenAI tools API (includes `reg_type`) |
| `EntityValidator` | Resolve English → Arabic developer names, validate areas |
| `TFTInference` | Load model, construct input, run prediction, extract quantiles |
| `TrendLookup` | Fetch developer/area stats from Supabase |
| `ChatService` | Manage sessions and messages in Supabase |
| `ResponseGenerator` | Create natural language from predictions |

---

## Critical Requirements

### 1. Arabic Developer Names

TFT data uses Arabic names. Must map:

| English | Arabic |
|---------|--------|
| Binghatti | `بن غاتي للتطوير العقاري` |
| Emaar | `اعمار العقارية ش. م. ع` |

**File**: `Data/lookups/developer_mapping.json` (MANUAL creation required)

### 2. reg_type Field

Must extract "OffPlan" or "Ready" from queries. Default to OffPlan.

### 3. Group ID Format

```
{area}_{property_type}_{bedroom}_{reg_type}_{developer_arabic}
```

---

## Supabase Tables

| Table | Purpose |
|-------|---------|
| `users` | Agent accounts |
| `chat_sessions` | Chat threads |
| `chat_messages` | Messages with predictions |
| `reports` | Generated PDFs |
| `developer_mapping` | Arabic ↔ English (optional) |
| `developer_stats` | Developer statistics |
| `area_stats` | Area price trends |

---

## Implementation Phases

| Phase | Description |
|-------|-------------|
| 0 | Data Preparation (lookup CSVs, developer mapping) |
| 1 | Supabase Setup (schema, RLS) |
| 2 | Query Parser (OpenAI tools API) |
| 3 | Entity Validator (Arabic support) |
| 4 | TFT Inference (full implementation) |
| 5 | Trend Lookup (Supabase) |
| 6 | Response Generator |
| 7 | Chat Endpoint (with persistence) |
| 8 | PDF Reports |
| 9 | Testing |

---

## Cost Estimates

| Item | Cost |
|------|------|
| OpenAI (per query) | ~$0.03 |
| Supabase Free Tier | $0 |
| **Monthly (100 queries/day)** | **~$90** |

---

## Prerequisites

Before implementing:

1. ✅ TFT model trained
2. ⬜ `Data/lookups/developer_mapping.json` created (MANUAL)
3. ⬜ `scripts/generate_lookup_tables.py` run
4. ⬜ Supabase project created
5. ⬜ Supabase tables migrated
