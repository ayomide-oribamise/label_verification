"""OCR service using EasyOCR (PyTorch-based, no PaddlePaddle/PyMuPDF issues).

Enhanced with:
- Confidence gating + fallback passes
- Rotation candidate testing (0, 90, 180, 270)
- Allowlists for specific fields (ABV, SKU, dates)
- Text normalization (Unicode NFKC, whitespace collapse)
- Health metrics logging
- Concurrency control via semaphore
- Line merging for adjacent boxes
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging
import os
import time
import threading
import unicodedata
import re

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OCRBox:
    """Represents a detected text box with position and confidence."""
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    @property
    def top(self) -> int:
        """Top Y coordinate (minimum Y)."""
        return min(p[1] for p in self.bbox)
    
    @property
    def bottom(self) -> int:
        """Bottom Y coordinate (maximum Y)."""
        return max(p[1] for p in self.bbox)
    
    @property
    def left(self) -> int:
        """Left X coordinate (minimum X)."""
        return min(p[0] for p in self.bbox)
    
    @property
    def right(self) -> int:
        """Right X coordinate (maximum X)."""
        return max(p[0] for p in self.bbox)
    
    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.bottom - self.top
    
    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.right - self.left
    
    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.height * self.width
    
    @property
    def center_y(self) -> int:
        """Center Y coordinate."""
        return (self.top + self.bottom) // 2
    
    @property
    def center_x(self) -> int:
        """Center X coordinate."""
        return (self.left + self.right) // 2


@dataclass 
class OCRResult:
    """Result from OCR processing."""
    boxes: List[OCRBox]
    raw_text: str
    average_confidence: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def empty(cls) -> "OCRResult":
        """Create empty result for failed OCR."""
        return cls(boxes=[], raw_text="", average_confidence=0.0, metrics={})


@dataclass
class OCRHealthMetrics:
    """Health metrics for OCR processing."""
    average_confidence: float
    token_count: int
    processing_time_ms: float
    image_dimensions: Tuple[int, int]
    fallback_used: bool
    rotation_used: int
    quality_warning: Optional[str] = None


@dataclass
class FieldTargetedResult:
    """Result from field-targeted OCR processing."""
    brand_text: str
    brand_confidence: float
    class_type_text: str
    class_type_confidence: float
    abv_text: str
    abv_confidence: float
    net_contents_text: str
    net_contents_confidence: float
    warning_text: str
    warning_confidence: float
    warning_detected: bool  # Keyword-based detection flag
    combined_raw_text: str
    overall_confidence: float
    processing_time_ms: float
    zones_processed: int
    zone_timings: Dict[str, float]  # Per-zone timing for debugging


# =============================================================================
# TWO-ZONE FAST PASS ARCHITECTURE
# =============================================================================
# Most labels can be processed with just 2 zones:
#   Pass A: Brand (top) + Footer (bottom)
#   → Often gets Brand, ABV, Net Contents, Warning from these alone
#   → Only run mid-zones if something is missing
# =============================================================================

# Fast pass zones (2 zones only)
FAST_PASS_ZONES = {
    "brand": {
        "y_start": 0.0,
        "y_end": 0.25,  # Top 25% - brand + often class/type
        "x_start": 0.0,
        "x_end": 1.0,
        "allowlist": None,  # General text
    },
    "footer": {
        "y_start": 0.70,
        "y_end": 1.0,  # Bottom 30% - ABV, net, warning all here
        "x_start": 0.0,
        "x_end": 1.0,
        "allowlist": None,  # Mixed content
    },
}

# Corner crops for ABV/Net (MUCH fewer pixels than full-width bands)
# Beer labels: ABV is usually bottom-left, Net is usually bottom-right
CORNER_CROPS = {
    "abv_corner": {
        "y_start": 0.55,
        "y_end": 0.88,
        "x_start": 0.0,
        "x_end": 0.45,  # Left 45%
        "allowlist": "0123456789.%ABVabvPROOFproofALCVOLalcvol/() ",
    },
    "net_corner": {
        "y_start": 0.55,
        "y_end": 0.92,
        "x_start": 0.55,
        "x_end": 1.0,  # Right 45%
        "allowlist": "0123456789.()mMlLcCfFoOzZ/ ",
    },
}

# Mid-zone fallbacks (only if fast pass misses)
MID_ZONES = {
    "class_type": {
        "y_start": 0.18,
        "y_end": 0.45,
        "x_start": 0.0,
        "x_end": 1.0,
        "allowlist": None,
    },
}

# Warning-specific zone with preprocessing
WARNING_ZONE = {
    "y_start": 0.82,
    "y_end": 1.0,
    "x_start": 0.0,
    "x_end": 1.0,
    "allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz():.,'%-/ ",
    "preprocess": "threshold",
}

# Warning detection keywords (any one = warning present)
WARNING_KEYWORDS = [
    "government warning",
    "surgeon general",
    "pregnancy",
    "birth defects",
    "alcoholic beverages",
    "impairs",
    "machinery",
    "health problems",
]

# Beverage keywords to detect class/type in brand zone
BEVERAGE_KEYWORDS_FOR_TYPE = {
    "whiskey", "whisky", "bourbon", "scotch", "vodka", "gin", "rum", "tequila",
    "wine", "cabernet", "chardonnay", "merlot", "pinot", "sauvignon",
    "beer", "ale", "lager", "stout", "porter", "ipa", "pilsner",
    "brandy", "cognac", "liqueur", "cider",
}

# ABV patterns for quick detection
ABV_PATTERN = re.compile(r'\d+\.?\d*\s*(%|proof)', re.IGNORECASE)
# Net contents patterns for quick detection  
NET_PATTERN = re.compile(r'\d+\.?\d*\s*(ml|cl|l|oz|fl)', re.IGNORECASE)


class OCRService:
    """EasyOCR wrapper service with enhanced accuracy features."""
    
    _instance: Optional["OCRService"] = None
    _reader = None
    _initialized = False
    _lock = threading.Lock()
    _semaphore: Optional[threading.Semaphore] = None
    
    def __new__(cls):
        """Singleton pattern to reuse OCR engine."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.settings = get_settings()
        # Initialize semaphore for concurrency control
        if self._semaphore is None:
            self._semaphore = threading.Semaphore(self.settings.ocr_max_concurrent)
    
    def initialize(self) -> bool:
        """
        Initialize OCR engine. Call on app startup.
        Thread-safe initialization.
        
        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._initialized:
                return True
            
            try:
                import easyocr
                import torch
                import os
                
                # Set thread limits for CPU inference
                # Use available CPUs (from env or cpu_count), but cap at reasonable limit
                num_threads = int(os.environ.get('TORCH_NUM_THREADS', min(4, os.cpu_count() or 2)))
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(1)  # Keep interop threads low
                
                logger.info(f"Initializing EasyOCR engine with {num_threads} threads...")
                
                # EasyOCR with English language, CPU mode
                model_dir = os.environ.get('EASYOCR_MODULE_PATH', '/home/app/.EasyOCR/model')
                
                self._reader = easyocr.Reader(
                    ['en'],
                    gpu=False,
                    model_storage_directory=model_dir,
                    verbose=False
                )
                
                self._initialized = True
                logger.info("EasyOCR initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    @property
    def is_ready(self) -> bool:
        """Check if OCR engine is ready."""
        return self._initialized and self._reader is not None
    
    def process_with_fallback(
        self, 
        image: np.ndarray,
        preprocessor=None,
        image_bytes: Optional[bytes] = None
    ) -> Tuple[OCRResult, OCRHealthMetrics]:
        """
        Run OCR with confidence gating and fallback strategies.
        
        Single readtext() call (region OCR was slower due to 3x detection passes).
        
        Args:
            image: Preprocessed image as numpy array
            preprocessor: ImagePreprocessor instance for fallback processing
            image_bytes: Original image bytes for fallback reprocessing
            
        Returns:
            Tuple of (OCRResult, OCRHealthMetrics)
        """
        start_time = time.time()
        fallback_used = False
        rotation_used = 0
        
        # Single OCR pass on full image (fastest approach)
        result = self._process_single(image)
        
        # Check if we need fallback
        needs_fallback = (
            result.average_confidence < self.settings.ocr_confidence_threshold or
            len(result.boxes) < self.settings.ocr_min_token_count
        )
        
        # Try rotation candidates if confidence is very low
        if (self.settings.try_rotation_candidates and 
            result.average_confidence < self.settings.rotation_confidence_threshold and
            preprocessor is not None):
            
            logger.info(f"Low confidence ({result.average_confidence:.2f}), trying rotation candidates...")
            best_result = result
            best_rotation = 0
            
            for angle in [90, 180, 270]:  # Already tried 0
                rotated = preprocessor.rotate_image(image, angle)
                rot_result = self._process_single(rotated)
                
                if rot_result.average_confidence > best_result.average_confidence:
                    best_result = rot_result
                    best_rotation = angle
            
            if best_rotation != 0:
                result = best_result
                rotation_used = best_rotation
                logger.info(f"Rotation {best_rotation}° improved confidence to {result.average_confidence:.2f}")
        
        # Fallback pass with different preprocessing
        if needs_fallback and preprocessor is not None and image_bytes is not None:
            logger.info(f"Confidence ({result.average_confidence:.2f}) below threshold, trying fallback pass...")
            fallback_used = True
            
            try:
                # Reprocess with fallback settings (upscaling, different contrast)
                fallback_image, _ = preprocessor.preprocess(image_bytes, for_fallback=True)
                fallback_result = self._process_single(fallback_image)
                
                # Merge results, keeping higher confidence boxes
                result = self._merge_results(result, fallback_result)
                
            except Exception as e:
                logger.warning(f"Fallback pass failed: {e}")
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000  # ms
        metrics = OCRHealthMetrics(
            average_confidence=result.average_confidence,
            token_count=len(result.boxes),
            processing_time_ms=processing_time,
            image_dimensions=(image.shape[1], image.shape[0]),
            fallback_used=fallback_used,
            rotation_used=rotation_used
        )
        
        # Log health metrics
        logger.info(
            f"OCR metrics: confidence={metrics.average_confidence:.2f}, "
            f"tokens={metrics.token_count}, time={metrics.processing_time_ms:.0f}ms, "
            f"fallback={fallback_used}, rotation={rotation_used}"
        )
        
        # Add metrics to result
        result.metrics = {
            "confidence": metrics.average_confidence,
            "token_count": metrics.token_count,
            "processing_time_ms": metrics.processing_time_ms,
            "fallback_used": metrics.fallback_used,
            "rotation_used": metrics.rotation_used
        }
        
        return result, metrics
    
    def process_detect_once(
        self,
        image: np.ndarray,
        max_dimension: int = 1600
    ) -> OCRResult:
        """
        DETECT ONCE architecture: Run EasyOCR detection + recognition ONCE.
        
        This is the KEY optimization:
        - Detection is expensive (~80% of OCR time)
        - Running readtext() per zone = running detection per zone
        - Instead: detect once, get all boxes, then slice by position
        
        Returns:
            OCRResult with all boxes in full-image coordinates
        """
        if not self.is_ready:
            logger.error("OCR engine not initialized")
            return OCRResult.empty()
        
        # Downscale once
        h, w = image.shape[:2]
        scale = min(1.0, max_dimension / max(h, w))
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Downscaled from {w}x{h} to {new_w}x{new_h}")
        
        with self._semaphore:
            try:
                # Single readtext call - detection + recognition in one pass
                # This is the ONLY OCR call we make for the entire image
                results = self._reader.readtext(
                    image,
                    decoder='greedy',
                    batch_size=1,
                    paragraph=False,
                    detail=1  # Return bounding boxes
                )
            except Exception as e:
                logger.error(f"OCR detect_once failed: {e}")
                return OCRResult.empty()
        
        if not results:
            logger.warning("OCR returned no results")
            return OCRResult.empty()
        
        # Parse results into OCRBox objects
        boxes = []
        for det in results:
            bbox_points = det[0]
            text = det[1]
            conf = det[2]
            
            # Normalize text
            text = self._normalize_text(text)
            if not text:
                continue
            
            # Convert bbox to int
            bbox_int = [[int(p[0]), int(p[1])] for p in bbox_points]
            boxes.append(OCRBox(
                text=text,
                confidence=float(conf),
                bbox=bbox_int
            ))
        
        # Merge adjacent boxes (same line)
        boxes = self._merge_adjacent_boxes(boxes)
        
        # Calculate average confidence
        avg_conf = (sum(b.confidence for b in boxes) / len(boxes)) if boxes else 0.0
        
        # Build raw text (sorted by position)
        if boxes:
            line_h = int(np.median([b.height for b in boxes]))
            line_h = max(12, min(line_h, 60))
            sorted_boxes = sorted(boxes, key=lambda b: (b.top // line_h, b.left))
            raw_text = " ".join(b.text for b in sorted_boxes)
        else:
            raw_text = ""
        
        return OCRResult(
            boxes=boxes,
            raw_text=raw_text,
            average_confidence=avg_conf,
            metrics={"detect_once": True}
        )
    
    def process_field_targeted(
        self,
        image: np.ndarray,
        max_dimension: int = 1600
    ) -> Tuple[FieldTargetedResult, OCRResult]:
        """
        DETECT ONCE + SLICE BY POSITION architecture.
        
        Strategy (per engineer):
        1. Run OCR detection ONCE on full image
        2. Slice detected boxes by Y-position to get zone texts
        3. NO additional OCR calls for zones
        4. Only if field STILL missing: ONE focused fallback crop
        
        This cuts detection from 5x to 1x = ~5x speedup.
        
        Returns:
            Tuple of (FieldTargetedResult, OCRResult for compatibility)
        """
        start_time = time.time()
        zone_timings = {}
        
        # =================================================================
        # STEP 1: DETECT ONCE (single OCR call for entire image)
        # =================================================================
        t0 = time.time()
        ocr_result = self.process_detect_once(image, max_dimension=max_dimension)
        detect_ms = (time.time() - t0) * 1000
        zone_timings["detect_once_ms"] = round(detect_ms, 1)
        zone_timings["total_boxes"] = len(ocr_result.boxes)
        
        # Get image bounds from detected boxes
        h, w = image.shape[:2]
        if ocr_result.boxes:
            # Use box positions to determine effective image height
            min_y = min(b.top for b in ocr_result.boxes)
            max_y = max(b.bottom for b in ocr_result.boxes)
            used_h = max(1, max_y - min_y)
            base_y = min_y
        else:
            used_h = h
            base_y = 0
        
        # =================================================================
        # STEP 2: SLICE BOXES BY Y-POSITION (no OCR, just filtering)
        # =================================================================
        t1 = time.time()
        
        def get_zone_text(y0_pct: float, y1_pct: float) -> Tuple[str, float]:
            """Get text from boxes within Y-range (as percentage of used_h)."""
            if not ocr_result.boxes:
                return "", 0.0
            
            y_start = base_y + int(used_h * y0_pct)
            y_end = base_y + int(used_h * y1_pct)
            
            # Filter boxes by center_y position
            zone_boxes = [b for b in ocr_result.boxes 
                         if y_start <= b.center_y <= y_end]
            
            if not zone_boxes:
                return "", 0.0
            
            # Sort left-to-right within lines
            line_h = int(np.median([b.height for b in zone_boxes]))
            line_h = max(12, min(line_h, 60))
            sorted_boxes = sorted(zone_boxes, key=lambda b: (b.top // line_h, b.left))
            
            text = " ".join(b.text for b in sorted_boxes)
            conf = sum(b.confidence for b in sorted_boxes) / len(sorted_boxes)
            return text, conf
        
        # Slice zones from detected boxes (NO OCR calls here)
        brand_text, brand_conf = get_zone_text(0.00, 0.30)      # Top 30%
        class_text, class_conf = get_zone_text(0.15, 0.55)      # Upper-mid
        abv_text, abv_conf = get_zone_text(0.40, 0.85)          # Mid-lower
        net_text, net_conf = get_zone_text(0.55, 0.95)          # Lower
        warn_text, warn_conf = get_zone_text(0.75, 1.00)        # Bottom
        
        slice_ms = (time.time() - t1) * 1000
        zone_timings["slice_ms"] = round(slice_ms, 1)
        
        # =================================================================
        # STEP 3: CHECK WHAT'S MISSING AND DO FOCUSED FALLBACKS
        # =================================================================
        combined_raw_text = ocr_result.raw_text
        fallbacks_run = []
        
        # Check if ABV is found
        has_abv = bool(ABV_PATTERN.search(abv_text)) or bool(ABV_PATTERN.search(combined_raw_text))
        if not has_abv and not ABV_PATTERN.search(combined_raw_text):
            # Fallback: OCR bottom-left corner with allowlist
            t_abv = time.time()
            h_img, w_img = image.shape[:2]
            abv_crop = image[int(h_img*0.55):int(h_img*0.95), 0:int(w_img*0.50)]
            abv_ocr = self._process_single(abv_crop, allowlist="0123456789.%ABVabvPROOFproofALCVOL/() ")
            abv_text = abv_ocr.raw_text
            abv_conf = abv_ocr.average_confidence
            zone_timings["abv_fallback_ms"] = round((time.time() - t_abv) * 1000, 1)
            fallbacks_run.append("abv")
        
        # Check if net contents is found
        has_net = bool(NET_PATTERN.search(net_text)) or bool(NET_PATTERN.search(combined_raw_text))
        if not has_net and not NET_PATTERN.search(combined_raw_text):
            # Fallback: OCR bottom-right corner with allowlist
            t_net = time.time()
            h_img, w_img = image.shape[:2]
            net_crop = image[int(h_img*0.55):int(h_img*0.95), int(w_img*0.50):w_img]
            net_ocr = self._process_single(net_crop, allowlist="0123456789.()mMlLcCfFoOzZ/ ")
            net_text = net_ocr.raw_text
            net_conf = net_ocr.average_confidence
            zone_timings["net_fallback_ms"] = round((time.time() - t_net) * 1000, 1)
            fallbacks_run.append("net")
        
        zone_timings["fallbacks_run"] = fallbacks_run
        
        # =================================================================
        # STEP 4: WARNING DETECTION (keywords + density heuristic)
        # =================================================================
        warning_detected = self._detect_warning_keywords(warn_text, combined_raw_text)
        
        # Density heuristic: many small boxes in bottom = likely warning
        if not warning_detected and ocr_result.boxes:
            bottom_boxes = [b for b in ocr_result.boxes if b.center_y >= base_y + used_h * 0.80]
            if len(bottom_boxes) >= 5:
                avg_height = sum(b.height for b in bottom_boxes) / len(bottom_boxes)
                if avg_height < used_h * 0.04:  # Small text
                    warning_detected = True
                    zone_timings["warning_by_density"] = True
        
        # =================================================================
        # BUILD RESULTS
        # =================================================================
        overall_confidence = ocr_result.average_confidence
        processing_time = (time.time() - start_time) * 1000
        zone_timings["total_ms"] = round(processing_time, 1)
        
        field_result = FieldTargetedResult(
            brand_text=brand_text,
            brand_confidence=brand_conf if brand_conf > 0 else overall_confidence,
            class_type_text=class_text,
            class_type_confidence=class_conf if class_conf > 0 else overall_confidence,
            abv_text=abv_text,
            abv_confidence=abv_conf if abv_conf > 0 else overall_confidence,
            net_contents_text=net_text,
            net_contents_confidence=net_conf if net_conf > 0 else overall_confidence,
            warning_text=warn_text,
            warning_confidence=warn_conf if warn_conf > 0 else overall_confidence,
            warning_detected=warning_detected,
            combined_raw_text=combined_raw_text,
            overall_confidence=overall_confidence,
            processing_time_ms=processing_time,
            zones_processed=0,  # No zone OCR, just slicing
            zone_timings=zone_timings
        )
        
        # Update OCRResult metrics
        ocr_result.metrics = {
            "field_targeted": True,
            "strategy": "detect_once_then_slice",
            "processing_time_ms": processing_time,
            "zone_timings": zone_timings,
        }
        
        # Log timing breakdown
        logger.info(
            f"DETECT-ONCE OCR: {len(ocr_result.boxes)} boxes, conf={overall_confidence:.2f}, "
            f"total={processing_time:.0f}ms | "
            f"detect={detect_ms:.0f}ms, slice={slice_ms:.0f}ms"
        )
        if fallbacks_run:
            logger.info(f"  + fallbacks: {fallbacks_run}")
        
        return field_result, ocr_result
    
    def _crop_zone(
        self, 
        image: np.ndarray, 
        h: int, 
        w: int, 
        zone: Dict[str, Any]
    ) -> np.ndarray:
        """Crop a zone from the image."""
        y_start = int(h * zone["y_start"])
        y_end = int(h * zone["y_end"])
        x_start = int(w * zone.get("x_start", 0.0))
        x_end = int(w * zone.get("x_end", 1.0))
        return image[y_start:y_end, x_start:x_end]
    
    def _adjust_boxes_to_image_xy(
        self,
        boxes: List[OCRBox],
        y_offset: int,
        x_offset: int
    ) -> List[OCRBox]:
        """Adjust box coordinates from zone space to full image space (with X offset)."""
        adjusted = []
        for box in boxes:
            adjusted_bbox = [
                [p[0] + x_offset, p[1] + y_offset] for p in box.bbox
            ]
            adjusted.append(OCRBox(
                text=box.text,
                confidence=box.confidence,
                bbox=adjusted_bbox
            ))
        return adjusted
    
    def _check_warning_density(
        self,
        footer_boxes: List[OCRBox],
        image_height: int
    ) -> bool:
        """
        Heuristic: If bottom strip has high text density, likely contains warning.
        
        Warning blocks are characterized by:
        - Many small boxes
        - Dense paragraph-like structure
        - Located in bottom 20% of image
        
        Returns True if warning likely present even if OCR didn't read it clearly.
        """
        if not footer_boxes:
            return False
        
        # Filter to boxes in bottom 20%
        bottom_threshold = image_height * 0.80
        bottom_boxes = [b for b in footer_boxes if b.top >= bottom_threshold]
        
        if len(bottom_boxes) < 5:
            return False
        
        # Check for small, dense text (typical warning characteristics)
        avg_height = sum(b.height for b in bottom_boxes) / len(bottom_boxes)
        
        # Warning text is typically small (< 3% of image height)
        if avg_height < image_height * 0.03 and len(bottom_boxes) >= 8:
            logger.debug(f"Warning detected by density: {len(bottom_boxes)} small boxes in bottom strip")
            return True
        
        return False
    
    def _preprocess_warning_zone(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess warning zone for better OCR on dense small text.
        
        Warning text is typically:
        - Very small font
        - Dense paragraph
        - Low contrast on some labels
        
        Apply grayscale + adaptive threshold for cleaner text.
        """
        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # Apply adaptive threshold for better text/background separation
        # ADAPTIVE_THRESH_GAUSSIAN_C works well for text
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,  # Neighborhood size
            C=8  # Constant subtracted from mean
        )
        
        return binary
    
    def _adjust_boxes_to_image(
        self, 
        boxes: List[OCRBox], 
        y_offset: int
    ) -> List[OCRBox]:
        """Adjust box coordinates from zone space to full image space."""
        adjusted = []
        for box in boxes:
            adjusted_bbox = [
                [p[0], p[1] + y_offset] for p in box.bbox
            ]
            adjusted.append(OCRBox(
                text=box.text,
                confidence=box.confidence,
                bbox=adjusted_bbox
            ))
        return adjusted
    
    def _identify_missing_fields(
        self, 
        zone_results: Dict[str, Any]
    ) -> List[str]:
        """
        Identify fields that need expanded zone search.
        
        A field is "missing" if:
        - No text detected
        - Very low confidence
        - Text doesn't contain expected patterns
        """
        missing = []
        
        # Check ABV - should have digits and % or proof
        abv_text = zone_results.get("abv", {}).get("text", "")
        if not re.search(r'\d+\.?\d*\s*(%|proof)', abv_text, re.IGNORECASE):
            missing.append("abv")
        
        # Check net contents - should have digits and units
        net_text = zone_results.get("net_contents", {}).get("text", "")
        if not re.search(r'\d+\.?\d*\s*(ml|cl|l|oz|fl)', net_text, re.IGNORECASE):
            missing.append("net_contents")
        
        # Check brand - should have some text
        brand_text = zone_results.get("brand", {}).get("text", "")
        if len(brand_text.strip()) < 3:
            missing.append("brand")
        
        # Check class_type - should have some text
        type_text = zone_results.get("class_type", {}).get("text", "")
        if len(type_text.strip()) < 3:
            missing.append("class_type")
        
        if missing:
            logger.debug(f"Missing fields for expansion: {missing}")
        
        return missing
    
    def _detect_warning_keywords(
        self, 
        warning_text: str, 
        full_text: str
    ) -> bool:
        """
        Keyword-based warning detection.
        
        Per engineer: "Any one keyword is enough for 'warning present'."
        Don't need to parse the entire warning text perfectly.
        """
        combined = (warning_text + " " + full_text).lower()
        
        for keyword in WARNING_KEYWORDS:
            if keyword in combined:
                logger.debug(f"Warning detected via keyword: '{keyword}'")
                return True
        
        return False
    
    def _process_single(
        self, 
        image: np.ndarray,
        allowlist: Optional[str] = None,
        blocklist: Optional[str] = None
    ) -> OCRResult:
        """
        Run single OCR pass on image.
        
        Args:
            image: Image as numpy array
            allowlist: Characters to allow (e.g., "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            blocklist: Characters to block
            
        Returns:
            OCRResult with detected text boxes
        """
        if not self.is_ready:
            logger.error("OCR engine not initialized")
            return OCRResult.empty()
        
        # Use semaphore for concurrency control
        with self._semaphore:
            try:
                # Build kwargs for readtext
                kwargs = {
                    'decoder': 'greedy',  # Faster than beamsearch
                    'batch_size': 1,      # Predictable CPU usage
                    'paragraph': False,   # Don't merge paragraphs (we do our own)
                }
                if allowlist:
                    kwargs['allowlist'] = allowlist
                if blocklist:
                    kwargs['blocklist'] = blocklist
                
                # EasyOCR accepts BGR or grayscale images directly
                results = self._reader.readtext(image, **kwargs)
                
                if not results:
                    logger.warning("OCR returned no results")
                    return OCRResult.empty()
                
                # Parse results into OCRBox objects
                boxes = []
                for detection in results:
                    bbox_points = detection[0]
                    text = detection[1]
                    confidence = detection[2]
                    
                    if not text.strip():
                        continue
                    
                    # Normalize text
                    normalized_text = self._normalize_text(text)
                    if not normalized_text:
                        continue
                    
                    # Convert bbox points to int
                    bbox_int = [[int(p[0]), int(p[1])] for p in bbox_points]
                    
                    boxes.append(OCRBox(
                        text=normalized_text,
                        confidence=float(confidence),
                        bbox=bbox_int
                    ))
                
                # Merge adjacent boxes on same line
                boxes = self._merge_adjacent_boxes(boxes)
                
                # Calculate average confidence
                if boxes:
                    avg_confidence = sum(b.confidence for b in boxes) / len(boxes)
                else:
                    avg_confidence = 0.0
                
                # Build raw text (sorted by line, then left-to-right)
                # Use dynamic line height based on median box height for better word ordering
                if boxes:
                    line_h = int(np.median([b.height for b in boxes]))
                    line_h = max(12, min(line_h, 60))  # Clamp to reasonable range
                else:
                    line_h = 20
                sorted_boxes = sorted(boxes, key=lambda b: (b.top // line_h, b.left))
                raw_text = " ".join(b.text for b in sorted_boxes)
                
                return OCRResult(
                    boxes=boxes,
                    raw_text=raw_text,
                    average_confidence=avg_confidence
                )
                
            except Exception as e:
                logger.error(f"OCR processing failed: {e}")
                import traceback
                traceback.print_exc()
                return OCRResult.empty()
    
    
    def process(self, image: np.ndarray) -> OCRResult:
        """
        Run OCR on preprocessed image (simple interface).
        For full features, use process_with_fallback().
        
        Args:
            image: Preprocessed image as numpy array (BGR or grayscale)
            
        Returns:
            OCRResult with detected text boxes
        """
        return self._process_single(image)
    
    def process_for_field(
        self, 
        image: np.ndarray, 
        field_type: str
    ) -> OCRResult:
        """
        Run OCR optimized for specific field type with allowlists.
        
        Args:
            image: Image as numpy array
            field_type: One of "abv", "date", "sku", "general"
            
        Returns:
            OCRResult with detected text
        """
        allowlists = {
            "abv": "0123456789.%ABVabv ",
            "date": "0123456789/-",
            "sku": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            "net_contents": "0123456789.mMlLfFoOzZ ",
        }
        
        allowlist = allowlists.get(field_type)
        return self._process_single(image, allowlist=allowlist)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize OCR text output.
        - Unicode NFKC normalization
        - Collapse whitespace
        - Strip leading/trailing whitespace
        """
        # Unicode normalize (NFKC converts ligatures, etc.)
        normalized = unicodedata.normalize('NFKC', text)
        
        # Collapse multiple whitespace to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Strip
        normalized = normalized.strip()
        
        return normalized
    
    def _merge_adjacent_boxes(
        self, 
        boxes: List[OCRBox],
        y_threshold: int = 15,
        x_gap_threshold: int = 30
    ) -> List[OCRBox]:
        """
        Merge adjacent boxes on same line into single boxes.
        Improves text coherence for multi-word phrases.
        
        Args:
            boxes: List of OCRBox
            y_threshold: Max Y difference to consider same line
            x_gap_threshold: Max X gap to merge
            
        Returns:
            Merged list of OCRBox
        """
        if len(boxes) <= 1:
            return boxes
        
        # Sort by Y position, then X
        sorted_boxes = sorted(boxes, key=lambda b: (b.center_y, b.left))
        
        merged = []
        current_line = [sorted_boxes[0]]
        
        for box in sorted_boxes[1:]:
            last_box = current_line[-1]
            
            # Check if on same line
            if abs(box.center_y - last_box.center_y) <= y_threshold:
                # Check if close enough to merge
                gap = box.left - last_box.right
                if gap <= x_gap_threshold:
                    current_line.append(box)
                else:
                    # Same line but gap too large, finalize current merge
                    merged.append(self._merge_box_group(current_line))
                    current_line = [box]
            else:
                # New line
                merged.append(self._merge_box_group(current_line))
                current_line = [box]
        
        # Don't forget last group
        if current_line:
            merged.append(self._merge_box_group(current_line))
        
        return merged
    
    def _merge_box_group(self, boxes: List[OCRBox]) -> OCRBox:
        """Merge a group of boxes into single box."""
        if len(boxes) == 1:
            return boxes[0]
        
        # Combine text
        combined_text = " ".join(b.text for b in boxes)
        
        # Average confidence
        avg_confidence = sum(b.confidence for b in boxes) / len(boxes)
        
        # Compute bounding box that contains all boxes
        min_x = min(b.left for b in boxes)
        min_y = min(b.top for b in boxes)
        max_x = max(b.right for b in boxes)
        max_y = max(b.bottom for b in boxes)
        
        # Create merged bbox in standard 4-point format
        merged_bbox = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]
        
        return OCRBox(
            text=combined_text,
            confidence=avg_confidence,
            bbox=merged_bbox
        )
    
    def _merge_results(self, result1: OCRResult, result2: OCRResult) -> OCRResult:
        """
        Merge two OCR results, keeping higher confidence boxes.
        Uses IoU overlap to detect duplicates.
        """
        all_boxes = list(result1.boxes)
        
        for box2 in result2.boxes:
            is_duplicate = False
            for i, box1 in enumerate(all_boxes):
                iou = self._calculate_iou(box1, box2)
                if iou > 0.5:  # Significant overlap
                    is_duplicate = True
                    # Keep higher confidence
                    if box2.confidence > box1.confidence:
                        all_boxes[i] = box2
                    break
            
            if not is_duplicate:
                all_boxes.append(box2)
        
        # Recalculate average confidence
        if all_boxes:
            avg_confidence = sum(b.confidence for b in all_boxes) / len(all_boxes)
        else:
            avg_confidence = 0.0
        
        # Rebuild raw text
        sorted_boxes = sorted(all_boxes, key=lambda b: (b.top // 15, b.left))
        raw_text = " ".join(b.text for b in sorted_boxes)
        
        return OCRResult(
            boxes=all_boxes,
            raw_text=raw_text,
            average_confidence=avg_confidence
        )
    
    def _calculate_iou(self, box1: OCRBox, box2: OCRBox) -> float:
        """Calculate Intersection over Union for two boxes."""
        # Calculate intersection
        x_left = max(box1.left, box2.left)
        y_top = max(box1.top, box2.top)
        x_right = min(box1.right, box2.right)
        y_bottom = min(box1.bottom, box2.bottom)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def get_text_by_position(
        self, 
        result: OCRResult, 
        region: str = "all"
    ) -> List[OCRBox]:
        """
        Get text boxes filtered by position.
        
        Args:
            result: OCR result
            region: "top", "middle", "bottom", or "all"
            
        Returns:
            List of OCRBox in the specified region
        """
        if not result.boxes:
            return []
        
        if region == "all":
            return result.boxes
        
        # Calculate image bounds from boxes
        min_y = min(b.top for b in result.boxes)
        max_y = max(b.bottom for b in result.boxes)
        height = max_y - min_y
        
        third = height // 3
        
        if region == "top":
            threshold = min_y + third
            return [b for b in result.boxes if b.center_y < threshold]
        elif region == "middle":
            lower = min_y + third
            upper = min_y + 2 * third
            return [b for b in result.boxes if lower <= b.center_y < upper]
        elif region == "bottom":
            threshold = min_y + 2 * third
            return [b for b in result.boxes if b.center_y >= threshold]
        
        return result.boxes
    
    def get_largest_text(self, result: OCRResult, top_n: int = 3) -> List[OCRBox]:
        """
        Get the largest text boxes (by height, likely headlines/brand names).
        
        Args:
            result: OCR result
            top_n: Number of largest boxes to return
            
        Returns:
            List of largest OCRBox sorted by height descending
        """
        if not result.boxes:
            return []
        
        sorted_by_height = sorted(result.boxes, key=lambda b: b.height, reverse=True)
        return sorted_by_height[:top_n]
