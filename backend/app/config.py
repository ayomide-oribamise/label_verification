"""Application configuration."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    app_name: str = "Label Verification API"
    debug: bool = False
    
    # CORS - Allow all origins for prototype (restrict in production)
    cors_origins: list[str] = ["*"]
    
    # Image processing limits
    # Allow large uploads (PNG can be 7MB+), but convert to JPEG before processing
    max_upload_size_mb: int = 15  # Allow large PNG uploads
    max_converted_size_mb: float = 3.0  # After JPEG conversion, must be under this
    max_image_width: int = 2000  # Max dimension before conversion
    max_image_dimension: int = 1024  # Max dimension for OCR processing
    min_image_dimension: int = 300  # Below this, upscale for OCR
    allowed_extensions: set = {"png", "jpg", "jpeg", "webp"}
    jpeg_quality: int = 82  # JPEG quality for conversion (80-85 optimal for OCR)
    
    # OCR settings
    ocr_lang: str = "en"
    
    # EasyOCR enhancement settings (tuned for speed)
    ocr_confidence_threshold: float = 0.30  # Lower threshold = fewer fallbacks
    ocr_min_token_count: int = 2  # Fewer tokens triggers fallback
    ocr_fallback_scale: float = 1.5  # Scale factor for fallback pass
    ocr_max_concurrent: int = 1  # Single OCR at a time (CPU-bound, no benefit from concurrency)
    
    # Image quality thresholds
    blur_threshold: float = 50.0  # Lowered - only trigger on very blurry
    contrast_threshold: float = 20.0  # Lowered - only trigger on very low contrast
    
    # Rotation detection - DISABLED for speed (add back if needed)
    try_rotation_candidates: bool = False  # Skip rotation testing for speed
    rotation_confidence_threshold: float = 0.20  # Very low threshold
    
    # Vision Assist (optional cloud fallback)
    vision_assist_enabled: bool = False
    openai_api_key: str | None = None
    
    # Verification thresholds (lowered for token-set matching)
    brand_match_threshold: float = 0.80  # Token-set is more forgiving
    brand_review_threshold: float = 0.65  # Send to review if partial match
    abv_tolerance: float = 0.5
    
    # Batch processing (sequential recommended on 2 vCPU)
    max_batch_size: int = 50  # Reasonable limit for sequential processing
    max_workers: int = 1  # Sequential: avoids 2x model load + OOM on limited memory
    max_total_pixels_per_batch: int = 50_000_000  # ~50MP total per batch
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
