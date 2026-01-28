# Label Verification Backend Dockerfile
# Optimized for PaddleOCR 2.7.0.3 with CPU inference (stable classic API, avoids PaddleX)

FROM python:3.11-slim AS builder

# Set environment variables
# Disable OneDNN and PIR to avoid compatibility issues
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLAGS_use_mkldnn=0 \
    FLAGS_use_gpu=0 \
    FLAGS_enable_pir_api=0 \
    FLAGS_pir_apply_inplace_pass=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PADDLEOCR_HOME=/home/app/.paddleocr

# Install build dependencies (including PyMuPDF build requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Upgrade pip first to ensure best wheel support
WORKDIR /build
COPY requirements.txt .
# Debug: Check pip config to see if anything is forcing source builds
RUN python -m pip config list -v || echo "No pip config found (this is fine)" && \
    python -m pip debug --verbose | grep -i "no-binary\|only-binary" || echo "No binary restrictions found"
# Install PyMuPDF first with --only-binary to force wheel (avoid source build)
# Pin to <1.23.0 to get manylinux_2_17 wheels compatible with Debian 11
# Version 1.23.x uses manylinux_2_28 which requires Debian 12+
RUN pip install --upgrade pip setuptools wheel && \
    pip install --only-binary=:all: "PyMuPDF<1.23.0" && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables for PaddleOCR 2.7.x
# Disable OneDNN and PIR to avoid compatibility issues
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLAGS_use_mkldnn=0 \
    FLAGS_use_gpu=0 \
    FLAGS_enable_pir_api=0 \
    FLAGS_pir_apply_inplace_pass=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
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
# Using PaddleOCR 2.7.0.3 classic API (not PaddleX pipeline)
RUN python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)" && \
    chown -R app:app /home/app/.paddleocr 2>/dev/null || true && \
    chown -R app:app /root/.paddleocr 2>/dev/null || true

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
