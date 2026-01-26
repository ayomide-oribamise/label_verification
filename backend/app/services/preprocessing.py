"""Image preprocessing utilities for improving OCR accuracy."""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple
from ..config import get_settings


class ImagePreprocessor:
    """Handles image preprocessing to improve OCR accuracy."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def preprocess(self, image_bytes: bytes) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for OCR.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (preprocessed image as numpy array, metadata dict)
        """
        # Load image
        image = self._load_image(image_bytes)
        
        metadata = {
            "original_size": image.shape[:2],
            "preprocessing_steps": []
        }
        
        # Resize if too large
        image, resized = self._resize_if_needed(image)
        if resized:
            metadata["preprocessing_steps"].append("resize")
            metadata["resized_to"] = image.shape[:2]
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            metadata["preprocessing_steps"].append("grayscale")
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        metadata["preprocessing_steps"].append("denoise")
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        metadata["preprocessing_steps"].append("contrast_enhancement")
        
        # Deskew if needed
        enhanced, angle = self._deskew(enhanced)
        if abs(angle) > 0.5:
            metadata["preprocessing_steps"].append(f"deskew_{angle:.1f}deg")
            metadata["deskew_angle"] = angle
        
        # Convert back to BGR for PaddleOCR (it expects color images)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result, metadata
    
    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes."""
        # Use PIL to handle various formats, then convert to OpenCV
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array (BGR for OpenCV)
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def _resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Resize image if width exceeds maximum."""
        height, width = image.shape[:2]
        max_width = self.settings.max_image_width
        
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized, True
        
        return image, False
    
    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew in image.
        
        Returns:
            Tuple of (deskewed image, detected angle in degrees)
        """
        # Use Hough transform to detect lines
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi / 180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return image, 0.0
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (within 45 degrees)
                if abs(angle) < 45:
                    angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # Use median angle to be robust to outliers
        median_angle = np.median(angles)
        
        # Only correct if angle is significant but not too extreme
        if abs(median_angle) < 0.5 or abs(median_angle) > 15:
            return image, 0.0
        
        # Rotate image to correct skew
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated, median_angle
    
    def get_image_info(self, image_bytes: bytes) -> dict:
        """Get basic image information without full preprocessing."""
        pil_image = Image.open(io.BytesIO(image_bytes))
        return {
            "format": pil_image.format,
            "mode": pil_image.mode,
            "width": pil_image.width,
            "height": pil_image.height,
            "size_bytes": len(image_bytes),
            "size_mb": len(image_bytes) / (1024 * 1024)
        }
    
    def validate_image(self, image_bytes: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate image meets requirements.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in self.settings.allowed_extensions:
            allowed = ", ".join(self.settings.allowed_extensions).upper()
            return False, f"Invalid file type. Allowed formats: {allowed}"
        
        # Check file size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > self.settings.max_image_size_mb:
            return False, f"Image exceeds {self.settings.max_image_size_mb}MB limit. Please resize or compress."
        
        # Try to load image
        try:
            info = self.get_image_info(image_bytes)
            if info["width"] < 100 or info["height"] < 100:
                return False, "Image too small. Minimum dimensions: 100x100 pixels."
        except Exception as e:
            return False, f"Unable to read image: {str(e)}"
        
        return True, ""
