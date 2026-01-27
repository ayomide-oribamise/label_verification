# Label Verification Backend Dockerfile
# Optimized for PaddleOCR 3.x with CPU inference

FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLEOCR_HOME=/home/app/.paddleocr

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pre-download PaddleOCR models during build (not at runtime)
# This makes container startup faster and more predictable
# Using PaddleOCR 3.x API
RUN python -c "import os; os.environ['DISABLE_MODEL_SOURCE_CHECK']='True'; from paddleocr import PaddleOCR; ocr = PaddleOCR(lang='en')" && \
    chown -R app:app /home/app/.paddlex 2>/dev/null || true && \
    chown -R app:app /root/.paddlex 2>/dev/null || true

# Copy application code
COPY --chown=app:app app/ ./app/

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run with uvicorn (production settings)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
