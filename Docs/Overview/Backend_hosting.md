## Backend Hosting Plan (Railway) + Model Runtime

### Goal
- Vercel hosts the web app (frontend).
- Railway hosts the FastAPI backend.
- Model is **already loaded** when users hit the app (no noticeable delays after login).

---

## Architecture
- **Frontend**: Vercel (Next.js)
- **Backend**: Railway (FastAPI/Uvicorn)
- **Auth**: Supabase Auth (frontend gets JWT; backend validates JWT)
- **Storage**:
  - `agent-logos` bucket: public (logos used in PDF)
  - `reports` bucket: private (PDFs saved per broker; downloaded via signed URLs)
  - `models` bucket: private (stores model checkpoint)

---

## Model Strategy (11.8MB checkpoint)
### Requirements
- Do not commit raw data to Git.
- Prefer not committing the `.ckpt` either.
- Keep inference fast by:
  - Loading model **once per process**
  - Keeping Railway service **warm** (always-on)

### Plan
- Store model file in Supabase Storage (private), e.g.:
  - bucket: `models`
  - path: `tft_final.ckpt`
- Backend startup:
  - if model file not present locally, download it to disk
  - load into memory once (singleton)
- Railway:
  - run at least **1 instance always-on**
  - (optional) add a health check ping to avoid cold starts

---

## Deployment Options (pick one)

### Option A — No Docker (Railway Nixpacks) [OK]
- Railway reads `backend/requirements.txt`
- Start command:
  - `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Pros: simpler, fewer files
- Cons: less reproducible than Docker

### Option B — Docker (recommended)
- Add `backend/Dockerfile`
- Railway builds exactly the same environment every time
- Pros: most reliable
- Cons: slightly more setup

Recommendation: start with Option A if you want minimum steps; move to Docker if you hit build/runtime inconsistencies.

---

## Railway Setup (both options)
1. Create Railway project → “Deploy from GitHub repo”
2. Set service **Root Directory** to: `backend`
3. Set environment variables (see below)
4. Ensure service is **always-on** (no cold start)
5. Add a `/health` endpoint and configure Railway health checks (recommended)

---

## Required Environment Variables (Railway)
### Supabase / Auth
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY` (if needed)
- `SUPABASE_SERVICE_ROLE_KEY` (server-side only)
- Any existing backend settings you already use (CORS, OpenAI key, etc.)

### Model
- `MODEL_BUCKET=models`
- `MODEL_PATH=tft_final.ckpt`
- `MODEL_LOCAL_PATH=/app/models/tft_final.ckpt` (or similar)
- (Optional) `MODEL_DOWNLOAD_ON_STARTUP=true`

### CORS
- `CORS_ORIGINS=https://<your-vercel-domain>,http://localhost:3000`

---

## Storage Security Summary
### `reports` bucket (private, paid SaaS requirement)
- Stored as `reports/{user_id}/{message_id}.pdf`
- RLS policies ensure each broker only accesses their own folder
- Downloads happen via backend endpoint returning **signed URLs**

### `models` bucket (private)
- Accessible only server-side (service role) OR via secure server endpoint
- Not publicly downloadable

---

## Backend Work Items (implementation)
- Add startup hook to:
  - download model if missing
  - load model into memory once
- Ensure inference endpoints reuse the loaded model (no per-request load)
- Add `/health` endpoint:
  - returns OK only when model is loaded (so “warm” truly means ready)
- Add logging for:
  - model load time
  - inference time
  - errors

---

## Operational Checklist
- First deploy:
  - confirm model downloads and loads once
  - confirm first inference request is fast
- Confirm no cold starts:
  - keep 1 instance running
  - optional uptime ping every 5 minutes
- Confirm secrets:
  - never commit `.env` or raw data
  - keep service role key only in Railway env vars