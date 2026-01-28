"""OCR service using PaddleOCR 2.7.0.3 (stable classic API, avoids PaddleX pipeline)."""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
import logging
import os

from ..config import get_settings

# Disable OneDNN and PIR to avoid compatibility issues with Azure Container Apps
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_gpu"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_pir_apply_inplace_pass"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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


@dataclass 
class OCRResult:
    """Result from OCR processing."""
    boxes: List[OCRBox]
    raw_text: str
    average_confidence: float
    
    @classmethod
    def empty(cls) -> "OCRResult":
        """Create empty result for failed OCR."""
        return cls(boxes=[], raw_text="", average_confidence=0.0)


class OCRService:
    """PaddleOCR 2.7.0.3 wrapper service (stable classic API, avoids PaddleX pipeline)."""
    
    _instance: Optional["OCRService"] = None
    _ocr = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to reuse OCR engine."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.settings = get_settings()
    
    def initialize(self) -> bool:
        """
        Initialize OCR engine. Call on app startup.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        try:
            from paddleocr import PaddleOCR
            
            logger.info("Initializing PaddleOCR 2.7.0.3 engine (classic API)...")
            
            # PaddleOCR 2.7.0.3 uses classic API (not PaddleX pipeline) - stable and well-tested
            self._ocr = PaddleOCR(
                use_angle_cls=True,  # Enable angle classification
                lang=self.settings.ocr_lang,
                use_gpu=False,  # CPU-only for container compatibility
                show_log=False,  # Reduce logging noise
            )
            
            self._initialized = True
            logger.info("PaddleOCR 2.7.0.3 initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    @property
    def is_ready(self) -> bool:
        """Check if OCR engine is ready."""
        return self._initialized and self._ocr is not None
    
    def process(self, image: np.ndarray) -> OCRResult:
        """
        Run OCR on preprocessed image.
        
        Args:
            image: Preprocessed image as numpy array (BGR format)
            
        Returns:
            OCRResult with detected text boxes
        """
        if not self.is_ready:
            logger.error("OCR engine not initialized")
            return OCRResult.empty()
        
        try:
            # Run PaddleOCR 2.7.0.3 - classic API uses ocr() method with cls=True
            results = self._ocr.ocr(image, cls=True)
            
            if not results or len(results) == 0:
                logger.warning("OCR returned no results")
                return OCRResult.empty()
            
            # PaddleOCR 2.7.0.3 classic API returns: [[[bbox], (text, confidence)], ...]
            # The first level is for each image (we only process one)
            detections = results[0]
            
            if not detections:
                logger.warning("OCR found no text")
                return OCRResult.empty()
            
            # Parse results into OCRBox objects
            boxes = []
            for detection in detections:
                # Each detection is [bbox_points, (text, confidence)]
                bbox_points = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = detection[1]    # (text, confidence)
                
                text = text_info[0]
                confidence = text_info[1]
                
                if not text.strip():
                    continue
                
                # Convert bbox points to int
                bbox_int = [[int(p[0]), int(p[1])] for p in bbox_points]
                
                boxes.append(OCRBox(
                    text=text,
                    confidence=float(confidence),
                    bbox=bbox_int
                ))
            
            # Calculate average confidence
            if boxes:
                avg_confidence = sum(b.confidence for b in boxes) / len(boxes)
            else:
                avg_confidence = 0.0
            
            # Build raw text (sorted by vertical position, then horizontal)
            sorted_boxes = sorted(boxes, key=lambda b: (b.top // 20, b.left))
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
