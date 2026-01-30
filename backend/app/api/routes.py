"""API route definitions."""

import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import logging

from ..models import (
    VerificationResponse,
    VerificationResult,
    FieldResult,
    VerificationStatus,
    ExtractedFields,
    ErrorResponse,
    HealthResponse,
    ApplicationData,
    BatchVerificationResponse,
    BatchRowResult,
)
from ..services import (
    ImagePreprocessor, 
    OCRService, 
    FieldExtractor, 
    VerificationService,
    CSVParser,
    SequentialBatchProcessor,
)
from ..config import get_settings
from .. import __version__

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
preprocessor = ImagePreprocessor()
ocr_service = OCRService()
field_extractor = FieldExtractor()
verification_service = VerificationService()
csv_parser = CSVParser()
batch_processor = SequentialBatchProcessor()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and OCR readiness."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        ocr_ready=ocr_service.is_ready
    )


@router.post(
    "/verify",
    response_model=VerificationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    tags=["Verification"]
)
async def verify_label(
    image: UploadFile = File(..., description="Label image file"),
    brand_name: str = Form(..., description="Expected brand name"),
    class_type: Optional[str] = Form(None, description="Expected class/type"),
    abv_percent: Optional[float] = Form(None, description="Expected ABV percentage"),
    net_contents_ml: Optional[float] = Form(None, description="Expected net contents in mL"),
    has_warning: bool = Form(True, description="Whether label should have government warning"),
):
    """
    Verify a single label image against application data.
    
    Upload a label image and provide the expected field values.
    Returns verification results for each field.
    """
    start_time = time.time()
    settings = get_settings()
    
    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded image: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded image")
    
    # Validate image
    is_valid, error_msg = preprocessor.validate_image(image_bytes, image.filename or "unknown")
    if not is_valid:
        return VerificationResponse(
            success=False,
            error=error_msg
        )
    
    # Check OCR readiness
    if not ocr_service.is_ready:
        return VerificationResponse(
            success=False,
            error="OCR service not ready. Please try again in a moment."
        )
    
    try:
        # Stage 1: Preprocess image
        preprocess_start = time.time()
        processed_image, preprocessing_meta = preprocessor.preprocess(image_bytes)
        preprocess_ms = int((time.time() - preprocess_start) * 1000)
        logger.info(f"Preprocessing steps: {preprocessing_meta['preprocessing_steps']} ({preprocess_ms}ms)")
        
        # Check for quality warnings and log
        if preprocessing_meta.get("quality", {}).get("is_blurry"):
            logger.warning("Image detected as blurry")
        if preprocessing_meta.get("quality", {}).get("is_low_contrast"):
            logger.warning("Image detected as low contrast")
        
        # Stage 2: Run OCR with fallback strategies for better accuracy
        ocr_start = time.time()
        ocr_result, ocr_metrics = ocr_service.process_with_fallback(
            processed_image,
            preprocessor=preprocessor,
            image_bytes=image_bytes
        )
        ocr_ms = int((time.time() - ocr_start) * 1000)
        logger.info(f"OCR completed ({ocr_ms}ms)")
        
        if not ocr_result.boxes:
            # Include quality recommendation if available
            quality_rec = preprocessing_meta.get("quality_recommendation", "")
            error_msg = "Unable to extract text from image. Please upload a clearer label."
            if quality_rec:
                error_msg = quality_rec
            return VerificationResponse(
                success=False,
                error=error_msg
            )
        
        # Stage 3: Extract fields from OCR result
        extract_start = time.time()
        extraction_result = field_extractor.extract_all(ocr_result)
        extract_ms = int((time.time() - extract_start) * 1000)
        
        # Build extracted fields response
        extracted = ExtractedFields(
            brand_name=extraction_result.brand_name.value,
            class_type=extraction_result.class_type.value,
            abv_percent=float(extraction_result.abv_percent.value) if extraction_result.abv_percent.value else None,
            net_contents_ml=float(extraction_result.net_contents_ml.value) if extraction_result.net_contents_ml.value else None,
            government_warning=extraction_result.government_warning.value,
            raw_text=ocr_result.raw_text,
            ocr_confidence=extraction_result.overall_confidence
        )
        
        # Stage 4: Perform verification
        verify_start = time.time()
        verification_result = verification_service.verify(
            extraction=extraction_result,
            expected_brand=brand_name,
            expected_class_type=class_type,
            expected_abv=abv_percent,
            expected_net_contents=net_contents_ml,
            expected_has_warning=has_warning,
        )
        verify_ms = int((time.time() - verify_start) * 1000)
        
        total_time = int((time.time() - start_time) * 1000)
        
        # Log timing breakdown
        logger.info(f"Timing breakdown: preprocess={preprocess_ms}ms, ocr={ocr_ms}ms, extract={extract_ms}ms, verify={verify_ms}ms, total={total_time}ms")
        
        # Convert verification result to response model
        field_results = [
            FieldResult(
                field_name=f.field_name,
                status=VerificationStatus(f.status.value),
                extracted_value=f.extracted_value,
                expected_value=f.expected_value,
                confidence=f.confidence,
                message=f.message + (f" {f.details}" if f.details else "")
            )
            for f in verification_result.fields
        ]
        
        result = VerificationResult(
            overall_status=VerificationStatus(verification_result.overall_status.value),
            fields=field_results,
            summary=verification_result.summary,
            processing_time_ms=total_time
        )
        
        # Add timing breakdown to result for debugging
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        result_dict['timing'] = {
            'preprocess_ms': preprocess_ms,
            'ocr_ms': ocr_ms,
            'extract_ms': extract_ms,
            'verify_ms': verify_ms
        }
        
        return VerificationResponse(
            success=True,
            result=result,
            extracted=extracted,
            error=None
        )
        
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        return VerificationResponse(
            success=False,
            error=f"Error processing image: {str(e)}"
        )


@router.post(
    "/extract",
    response_model=VerificationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
    },
    tags=["Extraction"]
)
async def extract_only(
    image: UploadFile = File(..., description="Label image file"),
):
    """
    Extract text from label image without verification.
    
    Useful for testing OCR and preprocessing.
    """
    start_time = time.time()
    
    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded image: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded image")
    
    # Validate image
    is_valid, error_msg = preprocessor.validate_image(image_bytes, image.filename or "unknown")
    if not is_valid:
        return VerificationResponse(
            success=False,
            error=error_msg
        )
    
    # Check OCR readiness
    if not ocr_service.is_ready:
        return VerificationResponse(
            success=False,
            error="OCR service not ready. Please try again in a moment."
        )
    
    try:
        # Preprocess image
        processed_image, preprocessing_meta = preprocessor.preprocess(image_bytes)
        
        # Run OCR with fallback - catch OCR-specific errors
        try:
            ocr_result, ocr_metrics = ocr_service.process_with_fallback(
                processed_image,
                preprocessor=preprocessor,
                image_bytes=image_bytes
            )
        except Exception as e:
            logger.exception(f"OCR processing failed: {e}")
            quality_rec = preprocessing_meta.get("quality_recommendation", "")
            error_msg = "Unable to extract text from image. Please upload a clearer label."
            if quality_rec:
                error_msg = quality_rec
            raise HTTPException(
                status_code=422,
                detail=error_msg
            )
        
        if not ocr_result.boxes:
            quality_rec = preprocessing_meta.get("quality_recommendation", "")
            error_msg = "Unable to extract text from image. Please upload a clearer label."
            if quality_rec:
                error_msg = quality_rec
            raise HTTPException(
                status_code=422,
                detail=error_msg
            )
        
        # Extract fields from OCR result
        extraction_result = field_extractor.extract_all(ocr_result)
        
        # Build response with extracted fields
        extracted = ExtractedFields(
            brand_name=extraction_result.brand_name.value,
            class_type=extraction_result.class_type.value,
            abv_percent=float(extraction_result.abv_percent.value) if extraction_result.abv_percent.value else None,
            net_contents_ml=float(extraction_result.net_contents_ml.value) if extraction_result.net_contents_ml.value else None,
            government_warning=extraction_result.government_warning.value,
            raw_text=ocr_result.raw_text,
            ocr_confidence=extraction_result.overall_confidence
        )
        
        return VerificationResponse(
            success=True,
            result=None,
            extracted=extracted,
            error=None
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (already have proper status codes)
        raise
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post(
    "/verify/batch",
    response_model=BatchVerificationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    tags=["Verification"]
)
async def verify_batch(
    images: List[UploadFile] = File(..., description="Label image files"),
    csv_file: UploadFile = File(..., description="CSV file with application data"),
):
    """
    Verify multiple label images against application data from CSV.
    
    Upload multiple label images and a CSV file containing the expected data.
    
    CSV format:
    - Required columns: filename, brand_name
    - Optional columns: class_type, abv_percent, net_contents_ml, has_warning
    
    Example CSV:
    ```
    filename,brand_name,class_type,abv_percent,net_contents_ml,has_warning
    label1.png,OLD TOM DISTILLERY,Kentucky Bourbon,45,750,true
    label2.png,JACK DANIELS,Tennessee Whiskey,40,1000,true
    ```
    
    Returns verification results for each label.
    """
    start_time = time.time()
    settings = get_settings()
    
    # Check batch size limit
    if len(images) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size is {settings.max_batch_size} files."
        )
    
    # Check OCR readiness
    if not ocr_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="OCR service not ready. Please try again in a moment."
        )
    
    # Read CSV file
    try:
        csv_content = (await csv_file.read()).decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="CSV file must be UTF-8 encoded"
        )
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read CSV file")
    
    # Parse and validate CSV
    csv_rows, csv_errors = csv_parser.parse(csv_content)
    
    if csv_errors and not csv_rows:
        # Critical errors - no valid rows
        error_messages = [f"Row {e.row_number}: {e.field} - {e.message}" for e in csv_errors[:5]]
        raise HTTPException(
            status_code=400,
            detail=f"CSV validation failed: {'; '.join(error_messages)}"
        )
    
    # Read all image files
    image_data = {}
    for upload_file in images:
        try:
            image_bytes = await upload_file.read()
            filename = upload_file.filename or "unknown"
            image_data[filename] = image_bytes
        except Exception as e:
            logger.error(f"Failed to read image {upload_file.filename}: {e}")
            # Continue with other files
    
    # Validate filenames match
    matched_rows, match_errors = csv_parser.validate_filenames_match(
        csv_rows, 
        list(image_data.keys())
    )
    
    # Combine errors for reporting but continue processing matched rows
    all_errors = csv_errors + match_errors
    
    if not matched_rows:
        error_messages = [f"Row {e.row_number}: {e.field} - {e.message}" for e in all_errors[:5]]
        raise HTTPException(
            status_code=400,
            detail=f"No valid image/CSV matches: {'; '.join(error_messages)}"
        )
    
    # Process batch
    try:
        results = batch_processor.process_batch(
            images=image_data,
            csv_rows=matched_rows,
            ocr_service=ocr_service,
            preprocessor=preprocessor,
            field_extractor=field_extractor,
            verification_service=verification_service
        )
    except Exception as e:
        logger.exception(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )
    
    # Convert results to response models
    batch_results = []
    passed_count = 0
    review_count = 0
    failed_count = 0
    
    for r in results:
        if r["success"] and r["result"]:
            result_data = r["result"]
            
            # Count by status
            status = result_data["overall_status"]
            if status == "match":
                passed_count += 1
            elif status == "review":
                review_count += 1
            else:
                failed_count += 1
            
            # Convert field results
            field_results = [
                FieldResult(
                    field_name=f["field_name"],
                    status=VerificationStatus(f["status"]),
                    extracted_value=f["extracted_value"],
                    expected_value=f["expected_value"],
                    confidence=f["confidence"],
                    message=f["message"]
                )
                for f in result_data["fields"]
            ]
            
            verification_result = VerificationResult(
                overall_status=VerificationStatus(result_data["overall_status"]),
                fields=field_results,
                summary=result_data["summary"],
                processing_time_ms=result_data["processing_time_ms"]
            )
            
            batch_results.append(BatchRowResult(
                filename=r["filename"],
                success=True,
                result=verification_result,
                error=None
            ))
        else:
            failed_count += 1
            batch_results.append(BatchRowResult(
                filename=r["filename"],
                success=False,
                result=None,
                error=r.get("error", "Unknown error")
            ))
    
    # Add errors for unmatched files
    for error in match_errors:
        if "not found" in error.message.lower():
            # CSV row without matching image
            batch_results.append(BatchRowResult(
                filename=error.message.split("'")[1] if "'" in error.message else "unknown",
                success=False,
                result=None,
                error=error.message
            ))
            failed_count += 1
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return BatchVerificationResponse(
        success=True,
        total=len(csv_rows),
        processed=len(results),
        passed=passed_count,
        needs_review=review_count,
        failed=failed_count,
        results=batch_results,
        processing_time_ms=processing_time
    )


@router.get("/ocr/boxes", tags=["Debug"])
async def get_ocr_boxes(
    image: UploadFile = File(..., description="Label image file"),
):
    """
    Debug endpoint: Get raw OCR boxes with positions.
    
    Returns detailed bounding box information for debugging.
    """
    # Read and validate
    try:
        image_bytes = await image.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read image")
    
    is_valid, error_msg = preprocessor.validate_image(image_bytes, image.filename or "unknown")
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    if not ocr_service.is_ready:
        raise HTTPException(status_code=503, detail="OCR service not ready")
    
    # Process with enhanced OCR
    processed_image, meta = preprocessor.preprocess(image_bytes)
    ocr_result, ocr_metrics = ocr_service.process_with_fallback(
        processed_image,
        preprocessor=preprocessor,
        image_bytes=image_bytes
    )
    
    # Return detailed box info
    boxes_data = []
    for box in ocr_result.boxes:
        boxes_data.append({
            "text": box.text,
            "confidence": box.confidence,
            "bbox": box.bbox,
            "top": box.top,
            "left": box.left,
            "height": box.height,
            "width": box.width,
            "area": box.area
        })
    
    return {
        "preprocessing": meta,
        "ocr_metrics": {
            "average_confidence": ocr_metrics.average_confidence,
            "token_count": ocr_metrics.token_count,
            "processing_time_ms": ocr_metrics.processing_time_ms,
            "image_dimensions": ocr_metrics.image_dimensions,
            "fallback_used": ocr_metrics.fallback_used,
            "rotation_used": ocr_metrics.rotation_used,
            "quality_warning": ocr_metrics.quality_warning
        },
        "box_count": len(boxes_data),
        "average_confidence": ocr_result.average_confidence,
        "raw_text": ocr_result.raw_text,
        "boxes": boxes_data
    }
