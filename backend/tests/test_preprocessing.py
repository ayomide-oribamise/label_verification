"""Tests for image preprocessing service."""

import pytest
import numpy as np
from PIL import Image
import io

from app.services.preprocessing import ImagePreprocessor
from app.config import get_settings


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return ImagePreprocessor()


@pytest.fixture
def sample_image_bytes():
    """Create a simple test image."""
    # Create a 200x100 white image with some text-like pattern
    img = Image.new("RGB", (200, 100), color="white")
    
    # Add some variation
    pixels = img.load()
    for i in range(50, 150):
        for j in range(30, 70):
            pixels[i, j] = (0, 0, 0)  # Black rectangle
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def large_image_bytes():
    """Create a large test image (2000px wide)."""
    img = Image.new("RGB", (2000, 1000), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestImagePreprocessor:
    """Test ImagePreprocessor class."""
    
    def test_validate_valid_image(self, preprocessor, sample_image_bytes):
        """Test validation passes for valid image."""
        is_valid, error = preprocessor.validate_image(sample_image_bytes, "test.png")
        assert is_valid is True
        assert error == ""
    
    def test_validate_invalid_extension(self, preprocessor, sample_image_bytes):
        """Test validation fails for invalid extension."""
        is_valid, error = preprocessor.validate_image(sample_image_bytes, "test.gif")
        # Note: gif might be invalid depending on settings
        # This test checks the extension validation logic
        settings = get_settings()
        if "gif" not in settings.allowed_extensions:
            assert is_valid is False
            assert "Allowed formats" in error
    
    def test_validate_image_too_small(self, preprocessor):
        """Test validation fails for too small image."""
        # Create 50x50 image (below minimum)
        img = Image.new("RGB", (50, 50), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        is_valid, error = preprocessor.validate_image(buffer.getvalue(), "small.png")
        assert is_valid is False
        assert "too small" in error.lower()
    
    def test_get_image_info(self, preprocessor, sample_image_bytes):
        """Test image info extraction."""
        info = preprocessor.get_image_info(sample_image_bytes)
        
        assert info["format"] == "PNG"
        assert info["width"] == 200
        assert info["height"] == 100
        assert info["size_bytes"] > 0
    
    def test_preprocess_returns_numpy_array(self, preprocessor, sample_image_bytes):
        """Test preprocessing returns numpy array."""
        result, metadata = preprocessor.preprocess(sample_image_bytes)
        
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3  # H, W, C
        assert result.shape[2] == 3  # BGR channels
    
    def test_preprocess_resizes_large_image(self, preprocessor, large_image_bytes):
        """Test that large images are resized."""
        result, metadata = preprocessor.preprocess(large_image_bytes)
        
        settings = get_settings()
        assert result.shape[1] <= settings.max_image_width
        assert "resize" in metadata["preprocessing_steps"]
    
    def test_preprocess_does_not_resize_small_image(self, preprocessor, sample_image_bytes):
        """Test that small images are not resized."""
        result, metadata = preprocessor.preprocess(sample_image_bytes)
        
        # Original image is 200px wide, well under limit
        assert "resize" not in metadata["preprocessing_steps"]
    
    def test_preprocess_applies_grayscale_and_enhancement(self, preprocessor, sample_image_bytes):
        """Test preprocessing applies expected steps."""
        result, metadata = preprocessor.preprocess(sample_image_bytes)
        
        assert "grayscale" in metadata["preprocessing_steps"]
        assert "denoise" in metadata["preprocessing_steps"]
        assert "contrast_enhancement" in metadata["preprocessing_steps"]


class TestImageValidation:
    """Test image validation edge cases."""
    
    def test_invalid_image_data(self, preprocessor):
        """Test handling of invalid image data."""
        is_valid, error = preprocessor.validate_image(b"not an image", "test.png")
        assert is_valid is False
        assert "Unable to read" in error
    
    def test_empty_image_data(self, preprocessor):
        """Test handling of empty image data."""
        is_valid, error = preprocessor.validate_image(b"", "test.png")
        assert is_valid is False
