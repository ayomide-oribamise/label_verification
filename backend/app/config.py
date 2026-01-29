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
    max_image_width: int = 1600  # Optimal for EasyOCR
    max_image_dimension: int = 1600  # Clamp both dimensions
    min_image_dimension: int = 300  # Below this, upscale for OCR
    allowed_extensions: set = {"png", "jpg", "jpeg", "webp"}
    
    # OCR settings
    ocr_lang: str = "en"
    
    # EasyOCR enhancement settings
    ocr_confidence_threshold: float = 0.45  # Below this triggers fallback
    ocr_min_token_count: int = 3  # Fewer tokens triggers fallback
    ocr_fallback_scale: float = 1.5  # Scale factor for fallback pass
    ocr_max_concurrent: int = 2  # Max concurrent OCR operations
    
    # Image quality thresholds
    blur_threshold: float = 100.0  # Laplacian variance below this = blurry
    contrast_threshold: float = 30.0  # Std dev below this = low contrast
    
    # Rotation detection
    try_rotation_candidates: bool = True  # Try 0, 90, 180, 270 if confidence low
    rotation_confidence_threshold: float = 0.35  # Below this, try rotations
    
    # Vision Assist (optional cloud fallback)
    vision_assist_enabled: bool = False
    openai_api_key: str | None = None
    
    # Verification thresholds
    brand_match_threshold: float = 0.95
    brand_review_threshold: float = 0.85
    abv_tolerance: float = 0.5
    
    # Batch processing
    max_batch_size: int = 300
    max_workers: int = 4
    max_total_pixels_per_batch: int = 50_000_000  # ~50MP total per batch
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
