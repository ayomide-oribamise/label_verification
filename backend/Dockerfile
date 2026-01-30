# Label Verification Backend Dockerfile
# Using EasyOCR (PyTorch-based) with enhanced accuracy features
# - Thread control for CPU inference
# - Pre-downloaded models
# - Optimized for Azure Container Apps

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Thread control for CPU inference
# These are defaults - override via Container App env vars for different CPU configs
# For 4 vCPU: set OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, TORCH_NUM_THREADS=4
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EASYOCR_MODULE_PATH=/home/app/.EasyOCR \
    # Thread settings - defaults for 2 vCPU, override for 4 vCPU
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    BLIS_NUM_THREADS=2 \
    TORCH_NUM_THREADS=2 \
    # Disable GPU for PyTorch
    CUDA_VISIBLE_DEVICES="" \
    # Reduce memory fragmentation
    MALLOC_TRIM_THRESHOLD_=100000

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

# Pre-download EasyOCR models during build (not at runtime)
# This makes container startup faster and more predictable
# Also set thread limits during model download
RUN mkdir -p /home/app/.EasyOCR/model && \
    python -c "import torch; torch.set_num_threads(1); import easyocr; reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/home/app/.EasyOCR/model')" && \
    chown -R app:app /home/app/.EasyOCR

# Copy application code
COPY --chown=app:app app/ ./app/

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check with longer start period for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run with uvicorn (single worker - model is memory heavy)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
