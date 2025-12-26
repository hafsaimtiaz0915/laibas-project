"""
Application Configuration

Loads settings from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache
from pathlib import Path

# Resolve paths relative to the repo root so running the backend from different working
# directories (repo root vs backend/) doesn't break file lookups.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = ""
    
    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    
    # TFT Model Paths
    # Local paths (in the container/runtime filesystem)
    # Using Console Model V2 (250K training steps, 29 features, better convergence)
    tft_model_path: str = str(_REPO_ROOT / "backend" / "models" / "Console_model" / "output_tft_best_v2.ckpt")
    # Full training data CSV - used for group matching (same as local dev)
    tft_data_path: str = str(_REPO_ROOT / "backend" / "models" / "Console_model" / "console_tft_training_data.csv")

    # Supabase Storage: model artifacts (private bucket)
    # NOTE: Upload Console model to bucket before deploying
    tft_storage_bucket: str = "model"
    tft_model_object: str = "output_tft_best_v2.ckpt"
    tft_data_object: str = "console_tft_training_data.csv"

    
    # Lookups
    lookups_path: str = str(_REPO_ROOT / "Data" / "lookups")
    
    # App Configuration
    debug: bool = False
    log_level: str = "INFO"
    
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:3001,https://www.proprly.ae,https://proprly.ae,https://proprly.vercel.app"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

