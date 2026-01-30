"""Image preprocessing utilities for improving OCR accuracy.

Enhanced for EasyOCR with:
- PNG to JPEG conversion (critical for OCR speed)
- ROI detection (find label region)
- Blur/quality detection
- Adaptive thresholding for glossy labels
- Sharpening after denoise
- Rotation candidate support (0, 90, 180, 270)
- Intelligent resizing
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

from ..config import get_settings

logger = logging.getLogger(__name__)


def convert_to_ocr_friendly(
    image_bytes: bytes, 
    max_dim: int = 2000, 
    jpeg_quality: int = 82,
    max_output_mb: float = 3.0
) -> Tuple[bytes, dict]:
    """
    Convert ANY image format to OCR-friendly JPEG.
    
    Handles: PNG, JPEG, WEBP, GIF, BMP, TIFF, etc.
    
    This is CRITICAL for performance:
    - Large images (7MB PNG, 5MB WEBP) → Optimized JPEG (800KB-1.2MB)
    - Normalizes all formats to consistent JPEG
    - Reduces pixel entropy for faster CNN inference
    - Smooths edges for more stable OCR
    - Can cut OCR time by 30-50%
    
    Args:
        image_bytes: Raw uploaded image bytes (any supported format)
        max_dim: Maximum dimension (width or height)
        jpeg_quality: JPEG quality (80-85 is optimal for OCR)
        max_output_mb: Maximum output size in MB
        
    Returns:
        Tuple of (converted_bytes, metadata_dict)
        
    Raises:
        ValueError: If converted image still exceeds max_output_mb
    """
    img = Image.open(io.BytesIO(image_bytes))
    original_format = img.format or "UNKNOWN"
    original_size = len(image_bytes)
    original_dims = img.size
    
    # Convert to RGB (handles RGBA PNGs, palette images, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize if needed
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    resized = False
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        resized = True
        logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    # Convert to JPEG
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    converted_bytes = out.getvalue()
    converted_size = len(converted_bytes)
    
    # Check output size
    converted_mb = converted_size / (1024 * 1024)
    if converted_mb > max_output_mb:
        # Try with lower quality
        for q in [70, 60, 50]:
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=q, optimize=True)
            converted_bytes = out.getvalue()
            converted_size = len(converted_bytes)
            converted_mb = converted_size / (1024 * 1024)
            if converted_mb <= max_output_mb:
                jpeg_quality = q
                break
        
        if converted_mb > max_output_mb:
            raise ValueError(
                f"Image too complex. After conversion: {converted_mb:.1f}MB "
                f"(max {max_output_mb}MB). Please use a simpler image."
            )
    
    metadata = {
        "original_format": original_format,
        "original_size_bytes": original_size,
        "original_dimensions": original_dims,
        "converted_size_bytes": converted_size,
        "converted_dimensions": img.size,
        "compression_ratio": original_size / converted_size if converted_size > 0 else 0,
        "resized": resized,
        "jpeg_quality": jpeg_quality,
    }
    
    logger.info(
        f"Image converted: {original_format} ({original_size/1024:.0f}KB) → "
        f"JPEG ({converted_size/1024:.0f}KB), "
        f"ratio={metadata['compression_ratio']:.1f}x"
    )
    
    return converted_bytes, metadata


@dataclass
class ImageQuality:
    """Image quality assessment results."""
    blur_score: float  # Laplacian variance - higher = sharper
    contrast_score: float  # Std deviation - higher = more contrast
    is_blurry: bool
    is_low_contrast: bool
    recommendation: Optional[str] = None


@dataclass
class PreprocessingResult:
    """Result of preprocessing with metadata."""
    image: np.ndarray
    metadata: dict
    quality: ImageQuality
    roi_applied: bool = False
    upscaled: bool = False


class ImagePreprocessor:
    """Handles image preprocessing to improve OCR accuracy."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def preprocess(self, image_bytes: bytes, for_fallback: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for OCR.
        
        Optimized pipeline order:
        1. Load & resize
        2. ROI detection (optional)
        3. Grayscale
        4. Deskew (on grayscale - better for Hough detection)
        5. Denoise (ONLY when needed - biggest performance saver)
        6. Sharpen
        7. CLAHE/adaptive threshold
        
        Args:
            image_bytes: Raw image bytes
            for_fallback: If True, apply more aggressive preprocessing for fallback pass
            
        Returns:
            Tuple of (preprocessed image as numpy array, metadata dict)
        """
        # Load image
        image = self._load_image(image_bytes)
        
        metadata = {
            "original_size": image.shape[:2],
            "preprocessing_steps": [],
            "quality": {}
        }
        
        # Assess image quality first
        quality = self._assess_quality(image)
        metadata["quality"] = {
            "blur_score": quality.blur_score,
            "contrast_score": quality.contrast_score,
            "is_blurry": quality.is_blurry,
            "is_low_contrast": quality.is_low_contrast
        }
        if quality.recommendation:
            metadata["quality_recommendation"] = quality.recommendation
        
        # Resize intelligently (clamp max dimension, upscale tiny images)
        image, resize_action = self._resize_intelligent(image, for_fallback)
        if resize_action:
            metadata["preprocessing_steps"].append(resize_action)
            metadata["resized_to"] = image.shape[:2]
        
        # Try ROI detection to focus on label region
        roi_image, roi_applied = self._detect_and_crop_roi(image)
        if roi_applied:
            image = roi_image
            metadata["preprocessing_steps"].append("roi_crop")
            metadata["roi_size"] = image.shape[:2]
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            metadata["preprocessing_steps"].append("grayscale")
        else:
            gray = image
        
        # Deskew ONLY in fallback mode (HoughLinesP is expensive)
        # Most clean images don't need deskew
        if for_fallback:
            gray, angle = self._deskew(gray)
            if abs(angle) > 0.5:
                metadata["preprocessing_steps"].append(f"deskew_{angle:.1f}deg")
                metadata["deskew_angle"] = angle
        
        # Denoise ONLY when needed (expensive operation - biggest perf win)
        # Only apply for: fallback pass, blurry images, or low contrast images
        do_denoise = for_fallback or quality.is_blurry or quality.is_low_contrast
        if do_denoise:
            h_param = 8 if for_fallback else 10
            gray = cv2.fastNlMeansDenoising(gray, None, h=h_param, templateWindowSize=7, searchWindowSize=21)
            metadata["preprocessing_steps"].append("denoise")
        
        # Sharpen lightly (helps EasyOCR)
        sharpened = self._sharpen(gray)
        metadata["preprocessing_steps"].append("sharpen")
        
        # Contrast enhancement
        if quality.is_low_contrast or for_fallback:
            # Adaptive threshold for glossy/reflective packaging
            enhanced = self._adaptive_threshold(sharpened)
            metadata["preprocessing_steps"].append("adaptive_threshold")
        else:
            # Standard CLAHE contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            metadata["preprocessing_steps"].append("clahe")
        
        # Convert back to BGR for EasyOCR (it accepts both but BGR is standard)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result, metadata
    
    def _assess_quality(self, image: np.ndarray) -> ImageQuality:
        """Assess image quality (blur, contrast) to guide preprocessing."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Blur detection using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        # Contrast detection using standard deviation
        contrast_score = gray.std()
        
        is_blurry = blur_score < self.settings.blur_threshold
        is_low_contrast = contrast_score < self.settings.contrast_threshold
        
        # Generate recommendation
        recommendation = None
        if is_blurry and is_low_contrast:
            recommendation = "Image is blurry and has low contrast. Please retake with better focus and lighting."
        elif is_blurry:
            recommendation = "Image appears blurry. Please retake with better focus or hold camera steady."
        elif is_low_contrast:
            recommendation = "Image has low contrast. Please ensure good lighting on the label."
        
        return ImageQuality(
            blur_score=blur_score,
            contrast_score=contrast_score,
            is_blurry=is_blurry,
            is_low_contrast=is_low_contrast,
            recommendation=recommendation
        )
    
    def _resize_intelligent(self, image: np.ndarray, upscale_if_small: bool = False) -> Tuple[np.ndarray, Optional[str]]:
        """
        Resize image intelligently:
        - Clamp max dimension to prevent slow OCR
        - Optionally upscale tiny images for fallback pass
        """
        height, width = image.shape[:2]
        max_dim = self.settings.max_image_dimension
        min_dim = self.settings.min_image_dimension
        
        # Downscale if too large
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized, "downscale"
        
        # Upscale if too small (only for fallback or explicitly requested)
        if upscale_if_small and max(width, height) < min_dim:
            scale = self.settings.ocr_fallback_scale
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return resized, f"upscale_{scale}x"
        
        return image, None
    
    def _detect_and_crop_roi(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detect the label/text-dense region and crop to it.
        Improves OCR by focusing on relevant area.
        
        Optimized:
        - Runs detection on downscaled preview (faster)
        - Maps bbox back to original resolution
        - Conservative settings to avoid cropping too much
        
        Returns:
            Tuple of (cropped image, whether ROI was applied)
        """
        try:
            height, width = image.shape[:2]
            
            # Downscale for faster detection (target ~600px width)
            PREVIEW_WIDTH = 600
            if width > PREVIEW_WIDTH:
                scale = PREVIEW_WIDTH / width
                preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                preview = image
                scale = 1.0
            
            # Convert to grayscale
            if len(preview.shape) == 3:
                gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            else:
                gray = preview.copy()
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect text regions - use smaller kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image, False
            
            # Find the largest contour (likely the label)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle (on preview)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Check if the ROI is significant (at least 30% of preview)
            preview_area = preview.shape[0] * preview.shape[1]
            roi_area = w * h
            
            if roi_area < 0.3 * preview_area or roi_area > 0.95 * preview_area:
                return image, False
            
            # Map back to original resolution
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            
            # Add padding (5-10%)
            pad_x = int(w * 0.08)
            pad_y = int(h * 0.08)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)
            
            cropped = image[y1:y2, x1:x2]
            
            # Ensure cropped region is substantial
            if cropped.shape[0] < 100 or cropped.shape[1] < 100:
                return image, False
            
            return cropped, True
            
        except Exception as e:
            logger.warning(f"ROI detection failed: {e}")
            return image, False
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply light sharpening to enhance text edges."""
        # Unsharp masking - gentle sharpening
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.3, gaussian, -0.3, 0)
        return sharpened
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for glossy/reflective labels.
        Better than global threshold for uneven lighting.
        """
        # Adaptive threshold with Gaussian weighting
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,
            C=10
        )
        return binary
    
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
    
    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew in image (±10° range).
        Uses minAreaRect for more robust angle detection.
        
        Returns:
            Tuple of (deskewed image, detected angle in degrees)
        """
        try:
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
                # Try minAreaRect as fallback
                return self._deskew_minarea(image)
            
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
                return self._deskew_minarea(image)
            
            # Use median angle to be robust to outliers
            median_angle = np.median(angles)
            
            # Only correct if angle is significant but within ±10°
            if abs(median_angle) < 0.5 or abs(median_angle) > 10:
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
            
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image, 0.0
    
    def _deskew_minarea(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fallback deskew using minAreaRect on contours."""
        try:
            # Find contours
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image, 0.0
            
            # Combine all contours
            all_points = np.vstack(contours)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(all_points)
            angle = rect[2]
            
            # Normalize angle to [-45, 45] range
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Only correct if within ±10°
            if abs(angle) < 0.5 or abs(angle) > 10:
                return image, 0.0
            
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, 
                rotation_matrix, 
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return rotated, angle
            
        except Exception:
            return image, 0.0
    
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate image by fixed angle (0, 90, 180, 270).
        Used for rotation candidate testing.
        """
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Arbitrary angle rotation
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    def get_rotation_candidates(self) -> List[int]:
        """Get list of rotation angles to try."""
        return [0, 90, 180, 270]
    
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
        Validate image meets requirements for upload (before conversion).
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in self.settings.allowed_extensions:
            allowed = ", ".join(self.settings.allowed_extensions).upper()
            return False, f"Invalid file type. Allowed formats: {allowed}"
        
        # Check file size against upload limit (not converted limit)
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > self.settings.max_upload_size_mb:
            return False, f"Image exceeds {self.settings.max_upload_size_mb}MB upload limit. Please resize or compress."
        
        # Try to load image
        try:
            info = self.get_image_info(image_bytes)
            if info["width"] < 100 or info["height"] < 100:
                return False, "Image too small. Minimum dimensions: 100x100 pixels."
        except Exception as e:
            return False, f"Unable to read image: {str(e)}"
        
        return True, ""
