"""Application configuration."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    app_name: str = "Label Verification API"
    debug: bool = False
    
    # CORS - Allowed frontend origins
    cors_origins: list[str] = [
        "https://kind-meadow-060b0b60f-preview.eastus2.1.azurestaticapps.net",
        "http://localhost:5173",  # Local development
    ]
    
    # Image processing limits
    max_image_size_mb: int = 3
    max_image_width: int = 1500
    allowed_extensions: set = {"png", "jpg", "jpeg", "webp"}
    
    # OCR settings
    ocr_lang: str = "en"
    
    # Vision Assist (optional cloud fallback)
    vision_assist_enabled: bool = False
    openai_api_key: str | None = None
    
    # Verification thresholds
    brand_match_threshold: float = 0.95
    brand_review_threshold: float = 0.85
    ocr_confidence_threshold: float = 0.75
    abv_tolerance: float = 0.5
    
    # Batch processing
    max_batch_size: int = 300
    max_workers: int = 4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
