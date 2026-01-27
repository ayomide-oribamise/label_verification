"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .services import OCRService
from .config import get_settings
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    # Startup
    logger.info("Starting Label Verification API...")
    settings = get_settings()
    
    # Initialize OCR engine on startup (keep warm)
    ocr_service = OCRService()
    if ocr_service.initialize():
        logger.info("OCR engine initialized and ready")
    else:
        logger.warning("OCR engine failed to initialize - will retry on first request")
    
    logger.info(f"API ready - Version {__version__}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Label Verification API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="""
## AI-Powered Alcohol Label Verification API

This API provides automated verification of alcohol label images against application data.

### Features
- **Image Upload**: Upload label images (PNG, JPG, JPEG, WEBP)
- **OCR Extraction**: Extract text using PaddleOCR
- **Field Verification**: Compare extracted fields against expected values
- **Batch Processing**: Process multiple labels at once

### Quick Start
1. Use `/health` to check API status
2. Use `/extract` to test OCR on an image
3. Use `/verify` to verify a label against application data
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Configure CORS - restrict to allowed frontend origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "Label Verification API",
            "version": __version__,
            "docs": "/docs"
        }
    
    return app


# Create app instance
app = create_app()
