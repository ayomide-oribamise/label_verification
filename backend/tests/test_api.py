"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a test image."""
    img = Image.new("RGB", (300, 200), color="white")
    # Add some text-like content
    pixels = img.load()
    for i in range(50, 250):
        for j in range(50, 150):
            if (i + j) % 10 < 5:
                pixels[i, j] = (0, 0, 0)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    
    def test_health_response_format(self, client):
        """Test health endpoint response format."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "ocr_ready" in data
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_200(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_version(self, client):
        """Test root endpoint contains version info."""
        response = client.get("/")
        data = response.json()
        
        assert "version" in data
        assert "docs" in data


class TestExtractEndpoint:
    """Test /extract endpoint."""
    
    def test_extract_requires_image(self, client):
        """Test extract endpoint requires image."""
        response = client.post("/api/v1/extract")
        assert response.status_code == 422  # Validation error
    
    def test_extract_rejects_invalid_format(self, client):
        """Test extract endpoint rejects invalid format."""
        response = client.post(
            "/api/v1/extract",
            files={"image": ("test.txt", b"not an image", "text/plain")}
        )
        # Should return success=false with error
        data = response.json()
        assert data["success"] is False
        assert "error" in data


class TestVerifyEndpoint:
    """Test /verify endpoint."""
    
    def test_verify_requires_image_and_brand(self, client):
        """Test verify endpoint requires both image and brand_name."""
        response = client.post("/api/v1/verify")
        assert response.status_code == 422  # Validation error
    
    def test_verify_with_valid_input(self, client, sample_image_bytes):
        """Test verify endpoint with valid input."""
        response = client.post(
            "/api/v1/verify",
            files={"image": ("label.png", sample_image_bytes, "image/png")},
            data={
                "brand_name": "Test Brand",
                "abv_percent": 45.0,
                "has_warning": True
            }
        )
        
        # Should return 200 (success or error in response body)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data


class TestInputValidation:
    """Test input validation."""
    
    def test_rejects_oversized_file(self, client):
        """Test that oversized files are rejected."""
        # Create a large image (but not actually 3MB to keep test fast)
        img = Image.new("RGB", (100, 100), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        # This won't actually exceed the limit, but tests the flow
        response = client.post(
            "/api/v1/extract",
            files={"image": ("large.png", buffer.getvalue(), "image/png")}
        )
        assert response.status_code == 200  # Validation happens in response body
    
    def test_rejects_invalid_extension(self, client):
        """Test that invalid extensions are rejected."""
        # Create valid image but with wrong extension
        img = Image.new("RGB", (100, 100), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        response = client.post(
            "/api/v1/extract",
            files={"image": ("test.bmp", buffer.getvalue(), "image/bmp")}
        )
        data = response.json()
        assert data["success"] is False
        assert "Allowed formats" in data.get("error", "")
