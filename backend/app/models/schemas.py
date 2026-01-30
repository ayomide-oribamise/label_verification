"""Pydantic schemas for API requests and responses."""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class VerificationStatus(str, Enum):
    """Status of a field verification."""
    MATCH = "match"
    REVIEW = "review"
    MISMATCH = "mismatch"
    NOT_FOUND = "not_found"


class FieldResult(BaseModel):
    """Result for a single field verification."""
    field_name: str
    status: VerificationStatus
    extracted_value: Optional[str] = None
    expected_value: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "field_name": "brand_name",
                "status": "match",
                "extracted_value": "OLD TOM DISTILLERY",
                "expected_value": "Old Tom Distillery",
                "confidence": 0.97,
                "message": "Brand name matches (case-insensitive)"
            }
        }


class VerificationResult(BaseModel):
    """Overall verification result for a label."""
    overall_status: VerificationStatus
    fields: list[FieldResult]
    summary: str
    processing_time_ms: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_status": "match",
                "fields": [],
                "summary": "All fields match. Label verified successfully.",
                "processing_time_ms": 1250
            }
        }


class ExtractedFields(BaseModel):
    """Fields extracted from label image via OCR."""
    brand_name: Optional[str] = None
    class_type: Optional[str] = None
    abv_percent: Optional[float] = None
    net_contents_ml: Optional[float] = None
    government_warning: Optional[str] = None
    raw_text: str
    ocr_confidence: float = Field(ge=0.0, le=1.0)


class ApplicationData(BaseModel):
    """Application data to verify against."""
    brand_name: str = Field(..., min_length=1, description="Expected brand name")
    class_type: Optional[str] = Field(None, description="Expected class/type (e.g., Kentucky Straight Bourbon)")
    abv_percent: Optional[float] = Field(None, ge=0, le=100, description="Expected ABV percentage")
    net_contents_ml: Optional[float] = Field(None, gt=0, description="Expected net contents in mL")
    has_warning: bool = Field(True, description="Whether label should have government warning")
    
    class Config:
        json_schema_extra = {
            "example": {
                "brand_name": "OLD TOM DISTILLERY",
                "class_type": "Kentucky Straight Bourbon Whiskey",
                "abv_percent": 45.0,
                "net_contents_ml": 750,
                "has_warning": True
            }
        }


class VerificationRequest(BaseModel):
    """Request body for single label verification (when using JSON instead of form)."""
    application_data: ApplicationData


class VerificationResponse(BaseModel):
    """Response for single label verification."""
    success: bool
    result: Optional[VerificationResult] = None
    extracted: Optional[ExtractedFields] = None
    error: Optional[str] = None


class BatchRowResult(BaseModel):
    """Result for a single row in batch verification."""
    filename: str
    success: bool
    result: Optional[VerificationResult] = None
    error: Optional[str] = None


class BatchVerificationResponse(BaseModel):
    """Response for batch verification."""
    success: bool
    total: int
    processed: int
    passed: int
    needs_review: int
    failed: int
    results: list[BatchRowResult]
    processing_time_ms: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format",
                "detail": "Allowed formats: PNG, JPG, JPEG, WEBP"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    ocr_ready: bool
