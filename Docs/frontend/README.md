# Frontend Documentation

> **Last Updated**: 2025-12-11

This folder contains documentation for the agent-facing frontend application.

---

## Documents

| Document | Description |
|----------|-------------|
| [00_FRONTEND_ARCHITECTURE.md](./00_FRONTEND_ARCHITECTURE.md) | Complete frontend architecture, PDF reports, agent branding |
| [01_COMPONENT_MAPPING.md](./01_COMPONENT_MAPPING.md) | Page layouts and 21st.dev component mapping |
| [02_MODEL_HOSTING.md](./02_MODEL_HOSTING.md) | TFT model hosting options (local, Railway, hybrid) |
| [03_COMPONENT_CODE.md](./03_COMPONENT_CODE.md) | Actual component code (sidebar, chat input, loading states) |

### LLM Integration

| Document | Description |
|----------|-------------|
| [LLM/00_LLM_INTEGRATION_PLAN.md](./LLM/00_LLM_INTEGRATION_PLAN.md) | Full LLM integration plan, architecture, phases |
| [LLM/01_LLM_SERVICE_CODE.md](./LLM/01_LLM_SERVICE_CODE.md) | Complete backend service code (Python) |

---

## Key Features

### 1. AI Chat Interface (Main Feature)
- ChatGPT/Claude-style interface
- Chat history in left sidebar
- Natural language property analysis queries
- Real-time responses from TFT model
- Inline analysis cards with PDF export

### 2. PDF Report Generation
- White-labeled client reports
- Agent logo and branding
- Customizable disclaimers
- Download directly from chat

### 3. Agent Settings
- Logo upload
- Contact information
- Report customization

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| Components | shadcn/ui |
| PDF | @react-pdf/renderer |
| State | Zustand |
| Auth | NextAuth.js |

---

## Quick Start

```bash
cd frontend
npm install
npm run dev
```

See [00_FRONTEND_ARCHITECTURE.md](./00_FRONTEND_ARCHITECTURE.md) for full implementation details.

