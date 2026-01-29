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
                
                # Set thread limits for CPU inference (prevents thread explosion)
                torch.set_num_threads(1)
                
                logger.info("Initializing EasyOCR engine...")
                
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
        
        # First pass: standard OCR
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
                kwargs = {}
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
                
                # Build raw text (sorted by vertical position, then horizontal)
                sorted_boxes = sorted(boxes, key=lambda b: (b.top // 15, b.left))
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
