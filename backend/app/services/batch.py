"""Batch processing service for multiple label verification."""

import csv
import io
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CSVRow:
    """Parsed and validated CSV row."""
    filename: str
    brand_name: str
    class_type: Optional[str] = None
    abv_percent: Optional[float] = None
    net_contents_ml: Optional[float] = None
    has_warning: bool = True
    row_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing."""
        return {
            "filename": self.filename,
            "brand_name": self.brand_name,
            "class_type": self.class_type,
            "abv_percent": self.abv_percent,
            "net_contents_ml": self.net_contents_ml,
            "has_warning": self.has_warning,
        }


@dataclass
class CSVValidationError:
    """Error from CSV validation."""
    row_number: int
    field: str
    message: str


class CSVParser:
    """Parse and validate batch CSV files."""
    
    # Required columns
    REQUIRED_COLUMNS = {"filename", "brand_name"}
    
    # Optional columns with their types
    OPTIONAL_COLUMNS = {
        "class_type": str,
        "abv_percent": float,
        "net_contents_ml": float,
        "has_warning": bool,
    }
    
    # All valid columns
    VALID_COLUMNS = REQUIRED_COLUMNS | set(OPTIONAL_COLUMNS.keys())
    
    def parse(self, csv_content: str) -> Tuple[List[CSVRow], List[CSVValidationError]]:
        """
        Parse CSV content and return validated rows.
        
        Args:
            csv_content: CSV file content as string
            
        Returns:
            Tuple of (valid_rows, errors)
        """
        rows: List[CSVRow] = []
        errors: List[CSVValidationError] = []
        
        try:
            # Detect dialect and parse
            reader = csv.DictReader(io.StringIO(csv_content))
            
            if reader.fieldnames is None:
                errors.append(CSVValidationError(
                    row_number=0,
                    field="header",
                    message="CSV file is empty or has no header"
                ))
                return rows, errors
            
            # Normalize column names (lowercase, strip whitespace)
            fieldnames = [f.lower().strip() for f in reader.fieldnames]
            
            # Validate required columns
            missing_required = self.REQUIRED_COLUMNS - set(fieldnames)
            if missing_required:
                errors.append(CSVValidationError(
                    row_number=0,
                    field="header",
                    message=f"Missing required columns: {', '.join(missing_required)}"
                ))
                return rows, errors
            
            # Warn about unknown columns (but don't fail)
            unknown_columns = set(fieldnames) - self.VALID_COLUMNS
            if unknown_columns:
                logger.warning(f"Unknown CSV columns will be ignored: {unknown_columns}")
            
            # Parse each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (1-indexed + header)
                # Normalize keys
                normalized_row = {k.lower().strip(): v for k, v in row.items()}
                
                row_errors = []
                
                # Validate required fields
                filename = (normalized_row.get("filename") or "").strip()
                if not filename:
                    row_errors.append(CSVValidationError(
                        row_number=row_num,
                        field="filename",
                        message="Filename is required"
                    ))
                
                brand_name = (normalized_row.get("brand_name") or "").strip()
                if not brand_name:
                    row_errors.append(CSVValidationError(
                        row_number=row_num,
                        field="brand_name",
                        message="Brand name is required"
                    ))
                
                # Parse optional fields
                class_type = (normalized_row.get("class_type") or "").strip() or None
                
                abv_percent = None
                abv_str = (normalized_row.get("abv_percent") or "").strip()
                if abv_str:
                    try:
                        abv_percent = float(abv_str)
                        if abv_percent < 0 or abv_percent > 100:
                            row_errors.append(CSVValidationError(
                                row_number=row_num,
                                field="abv_percent",
                                message=f"ABV must be between 0 and 100, got {abv_percent}"
                            ))
                            abv_percent = None
                    except ValueError:
                        row_errors.append(CSVValidationError(
                            row_number=row_num,
                            field="abv_percent",
                            message=f"Invalid ABV value: '{abv_str}'"
                        ))
                
                net_contents_ml = None
                net_str = (normalized_row.get("net_contents_ml") or "").strip()
                if net_str:
                    try:
                        net_contents_ml = float(net_str)
                        if net_contents_ml <= 0:
                            row_errors.append(CSVValidationError(
                                row_number=row_num,
                                field="net_contents_ml",
                                message=f"Net contents must be positive, got {net_contents_ml}"
                            ))
                            net_contents_ml = None
                    except ValueError:
                        row_errors.append(CSVValidationError(
                            row_number=row_num,
                            field="net_contents_ml",
                            message=f"Invalid net contents value: '{net_str}'"
                        ))
                
                has_warning = True  # Default to True
                warning_str = (normalized_row.get("has_warning") or "").strip().lower()
                if warning_str:
                    if warning_str in ("true", "yes", "1"):
                        has_warning = True
                    elif warning_str in ("false", "no", "0"):
                        has_warning = False
                    else:
                        row_errors.append(CSVValidationError(
                            row_number=row_num,
                            field="has_warning",
                            message=f"Invalid has_warning value: '{warning_str}'. Use true/false."
                        ))
                
                # Add errors or valid row
                errors.extend(row_errors)
                
                # Only add row if no critical errors (filename and brand_name present)
                if filename and brand_name:
                    rows.append(CSVRow(
                        filename=filename,
                        brand_name=brand_name,
                        class_type=class_type,
                        abv_percent=abv_percent,
                        net_contents_ml=net_contents_ml,
                        has_warning=has_warning,
                        row_number=row_num
                    ))
            
        except csv.Error as e:
            errors.append(CSVValidationError(
                row_number=0,
                field="csv",
                message=f"CSV parsing error: {str(e)}"
            ))
        
        return rows, errors
    
    def validate_filenames_match(
        self, 
        csv_rows: List[CSVRow], 
        uploaded_filenames: List[str]
    ) -> Tuple[List[CSVRow], List[CSVValidationError]]:
        """
        Validate that CSV filenames match uploaded files.
        
        Returns:
            Tuple of (matched_rows, errors for unmatched)
        """
        uploaded_set = set(uploaded_filenames)
        matched_rows = []
        errors = []
        
        for row in csv_rows:
            if row.filename in uploaded_set:
                matched_rows.append(row)
            else:
                errors.append(CSVValidationError(
                    row_number=row.row_number,
                    field="filename",
                    message=f"Image file not found: '{row.filename}'"
                ))
        
        # Check for uploaded files without CSV rows
        csv_filenames = {row.filename for row in csv_rows}
        unmatched_uploads = uploaded_set - csv_filenames
        if unmatched_uploads:
            for filename in unmatched_uploads:
                errors.append(CSVValidationError(
                    row_number=0,
                    field="filename",
                    message=f"Uploaded image has no CSV entry: '{filename}'"
                ))
        
        return matched_rows, errors


# Worker function for multiprocessing (must be at module level)
def _process_single_label(args: Tuple) -> Dict[str, Any]:
    """
    Process a single label in a worker process.
    
    This function runs in a separate process, so it needs to
    initialize its own OCR engine.
    """
    from .preprocessing import ImagePreprocessor
    from .ocr import OCRService
    from .extraction import FieldExtractor
    from .verification import VerificationService
    
    image_bytes, row_data, filename = args
    start_time = time.time()
    
    try:
        # Initialize services (each process needs its own)
        preprocessor = ImagePreprocessor()
        ocr_service = OCRService()
        field_extractor = FieldExtractor()
        verification_service = VerificationService()
        
        # Validate image
        is_valid, error_msg = preprocessor.validate_image(image_bytes, filename)
        if not is_valid:
            return {
                "filename": filename,
                "success": False,
                "error": error_msg,
                "result": None
            }
        
        # Wait for OCR to be ready
        if not ocr_service.is_ready:
            ocr_service.initialize()
        
        # Preprocess
        processed_image, preprocessing_meta = preprocessor.preprocess(image_bytes)
        
        # OCR with fallback for better accuracy
        ocr_result, ocr_metrics = ocr_service.process_with_fallback(
            processed_image,
            preprocessor=preprocessor,
            image_bytes=image_bytes
        )
        
        if not ocr_result.boxes:
            quality_rec = preprocessing_meta.get("quality_recommendation", "")
            error_msg = "Unable to extract text from image"
            if quality_rec:
                error_msg = quality_rec
            return {
                "filename": filename,
                "success": False,
                "error": error_msg,
                "result": None
            }
        
        # Extract fields
        extraction_result = field_extractor.extract_all(ocr_result)
        
        # Verify
        verification_result = verification_service.verify(
            extraction=extraction_result,
            expected_brand=row_data["brand_name"],
            expected_class_type=row_data.get("class_type"),
            expected_abv=row_data.get("abv_percent"),
            expected_net_contents=row_data.get("net_contents_ml"),
            expected_has_warning=row_data.get("has_warning", True),
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Convert to serializable format
        field_results = [
            {
                "field_name": f.field_name,
                "status": f.status.value,
                "extracted_value": f.extracted_value,
                "expected_value": f.expected_value,
                "confidence": f.confidence,
                "message": f.message + (f" {f.details}" if f.details else "")
            }
            for f in verification_result.fields
        ]
        
        return {
            "filename": filename,
            "success": True,
            "error": None,
            "result": {
                "overall_status": verification_result.overall_status.value,
                "fields": field_results,
                "summary": verification_result.summary,
                "processing_time_ms": processing_time
            }
        }
        
    except Exception as e:
        logger.exception(f"Error processing {filename}: {e}")
        return {
            "filename": filename,
            "success": False,
            "error": f"Processing error: {str(e)}",
            "result": None
        }


class BatchProcessor:
    """Process multiple labels in parallel."""
    
    def __init__(self):
        self.settings = get_settings()
        self.csv_parser = CSVParser()
    
    def process_batch(
        self,
        images: Dict[str, bytes],
        csv_rows: List[CSVRow],
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images with their application data.
        
        Args:
            images: Dict mapping filename to image bytes
            csv_rows: List of CSVRow with expected data
            max_workers: Max parallel workers (defaults to config)
            
        Returns:
            List of result dicts
        """
        if max_workers is None:
            max_workers = min(
                self.settings.max_workers,  # Fixed: was batch_max_workers
                multiprocessing.cpu_count(),
                len(csv_rows)  # No point having more workers than items
            )
        
        # Prepare arguments for each worker
        work_items = []
        for row in csv_rows:
            if row.filename in images:
                work_items.append((
                    images[row.filename],
                    row.to_dict(),
                    row.filename
                ))
        
        results = []
        
        if len(work_items) <= 1 or max_workers <= 1:
            # Process sequentially for small batches
            for args in work_items:
                result = _process_single_label(args)
                results.append(result)
        else:
            # Process in parallel
            # Note: Using ProcessPoolExecutor to bypass GIL for CPU-bound OCR
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_single_label, args): args[2]  # filename
                    for args in work_items
                }
                
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.exception(f"Worker error for {filename}: {e}")
                        results.append({
                            "filename": filename,
                            "success": False,
                            "error": f"Worker error: {str(e)}",
                            "result": None
                        })
        
        # Sort results by original order
        filename_order = {row.filename: i for i, row in enumerate(csv_rows)}
        results.sort(key=lambda r: filename_order.get(r["filename"], 999))
        
        return results


# Singleton instance for sequential processing (uses pre-initialized OCR)
class SequentialBatchProcessor:
    """
    Process batch sequentially using shared OCR engine.
    
    More memory efficient than ProcessPoolExecutor for small batches.
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def process_batch(
        self,
        images: Dict[str, bytes],
        csv_rows: List[CSVRow],
        ocr_service,
        preprocessor,
        field_extractor,
        verification_service
    ) -> List[Dict[str, Any]]:
        """
        Process batch sequentially with shared services.
        
        Args:
            images: Dict mapping filename to image bytes
            csv_rows: List of CSVRow with expected data
            ocr_service: Shared OCR service instance
            preprocessor: Shared preprocessor instance
            field_extractor: Shared field extractor instance
            verification_service: Shared verification service instance
            
        Returns:
            List of result dicts
        """
        results = []
        
        for row in csv_rows:
            start_time = time.time()
            filename = row.filename
            
            if filename not in images:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": f"Image file not found: {filename}",
                    "result": None
                })
                continue
            
            image_bytes = images[filename]
            
            try:
                # Validate image
                is_valid, error_msg = preprocessor.validate_image(image_bytes, filename)
                if not is_valid:
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": error_msg,
                        "result": None
                    })
                    continue
                
                # Preprocess
                processed_image, preprocessing_meta = preprocessor.preprocess(image_bytes)
                
                # OCR with fallback for better accuracy
                ocr_result, ocr_metrics = ocr_service.process_with_fallback(
                    processed_image,
                    preprocessor=preprocessor,
                    image_bytes=image_bytes
                )
                
                if not ocr_result.boxes:
                    # Include quality recommendation if available
                    quality_rec = preprocessing_meta.get("quality_recommendation", "")
                    error_msg = "Unable to extract text from image"
                    if quality_rec:
                        error_msg = quality_rec
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": error_msg,
                        "result": None
                    })
                    continue
                
                # Extract fields
                extraction_result = field_extractor.extract_all(ocr_result)
                
                # Verify
                verification_result = verification_service.verify(
                    extraction=extraction_result,
                    expected_brand=row.brand_name,
                    expected_class_type=row.class_type,
                    expected_abv=row.abv_percent,
                    expected_net_contents=row.net_contents_ml,
                    expected_has_warning=row.has_warning,
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Convert to serializable format
                field_results = [
                    {
                        "field_name": f.field_name,
                        "status": f.status.value,
                        "extracted_value": f.extracted_value,
                        "expected_value": f.expected_value,
                        "confidence": f.confidence,
                        "message": f.message + (f" {f.details}" if f.details else "")
                    }
                    for f in verification_result.fields
                ]
                
                results.append({
                    "filename": filename,
                    "success": True,
                    "error": None,
                    "result": {
                        "overall_status": verification_result.overall_status.value,
                        "fields": field_results,
                        "summary": verification_result.summary,
                        "processing_time_ms": processing_time
                    }
                })
                
            except Exception as e:
                logger.exception(f"Error processing {filename}: {e}")
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": f"Processing error: {str(e)}",
                    "result": None
                })
        
        return results
