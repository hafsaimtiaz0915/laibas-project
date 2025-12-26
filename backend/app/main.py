"""
Property Analysis API

FastAPI backend for the Dubai property analysis chat application.
"""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .core.config import get_settings
from .core.supabase import get_supabase_admin
from .routes import chat, sessions, report, agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Property Analysis API...")
    settings = get_settings()
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")
    
    # Verify critical settings
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured!")
    if not settings.supabase_url:
        logger.warning("Supabase URL not configured!")

    # Ensure model artifacts exist locally (download from Supabase Storage if missing)
    try:
        client = get_supabase_admin()
        bucket = settings.tft_storage_bucket

        model_path = Path(settings.tft_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            logger.info(f"Downloading TFT model from storage: {bucket}/{settings.tft_model_object} -> {model_path}")
            content = client.storage.from_(bucket).download(settings.tft_model_object)
            model_path.write_bytes(content)
            logger.info(f"Downloaded model ({len(content):,} bytes)")

        data_path = Path(settings.tft_data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            logger.info(f"Downloading TFT training data from storage: {bucket}/{settings.tft_data_object} -> {data_path}")
            content = client.storage.from_(bucket).download(settings.tft_data_object)
            data_path.write_bytes(content)
            logger.info(f"Downloaded training data ({len(content):,} bytes)")

        # Eager-load the TFT service so users don't pay cold-start/model-load latency on first request
        try:
            from .services.tft_inference import get_tft_service
            _svc = get_tft_service()
            logger.info(f"TFT service initialized (model_loaded={_svc.is_loaded}, groups={len(_svc.groups)})")
        except Exception as e:
            logger.error(f"Failed to initialize TFT service on startup: {e}")
    except Exception as e:
        logger.error(f"Failed to ensure model artifacts on startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Property Analysis API...")


# Create FastAPI app
app = FastAPI(
    title="Property Analysis API",
    description="Dubai property analysis and forecasting service",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(report.router, prefix="/api", tags=["Reports"])
app.include_router(agent.router, prefix="/api", tags=["Agent"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Property Analysis API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    settings = get_settings()
    return {
        "status": "healthy",
        "openai_configured": bool(settings.openai_api_key),
        "supabase_configured": bool(settings.supabase_url and settings.supabase_service_key),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

