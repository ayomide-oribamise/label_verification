# AI-Powered Alcohol Label Verification

**Author:** Ayomide Oribamise

An automated system for verifying alcohol label compliance by comparing label images against application data using OCR and intelligent field extraction.

## Quick Demo

**Try it in under 10 seconds:**
1. Visit the [deployed application](https://kind-meadow-060b0b60f.1.azurestaticapps.net)
2. Click "Load Sample Application" → Select "Bourbon"
3. Click "Verify Label"

## Project Overview

This application extracts text from alcohol label images and verifies it against expected application data, checking:
- **Brand Name** - Fuzzy matching with token reordering support
- **Class/Type** - Semantic beverage classification (e.g., "IPA" = "India Pale Ale")
- **Alcohol Content (ABV)** - Numeric extraction with proof conversion
- **Net Contents** - Volume extraction with unit conversion (oz → mL)
- **Government Warning** - Presence detection of required health warning

## Repository Structure

This project uses a **multi-branch architecture** for clear separation of concerns:

```
main              ← Documentation and project overview (this README)
├── backend       ← FastAPI backend service
├── frontend      ← React frontend application  
└── infra         ← Terraform infrastructure as code
```

### Why This Structure?

1. **Independent Deployment Cycles**: Backend and frontend can be deployed separately without affecting each other

2. **Clear Ownership**: Each branch has a single responsibility, making code review and maintenance easier

3. **Infrastructure as Code Isolation**: Terraform state and configurations are separate from application code, preventing accidental infrastructure changes during app updates

4. **Simplified CI/CD**: Each branch can have its own deployment pipeline triggered only when relevant code changes

## Architecture

```
┌─────────────────┐         ┌─────────────────────────────────────┐
│                 │         │     Azure Container Apps            │
│  React Frontend │  HTTPS  │                                     │
│  (Static Web    │────────▶│  FastAPI Backend                    │
│   App)          │         │  ├── Image Preprocessing            │
│                 │         │  ├── EasyOCR (Text Detection)       │
└─────────────────┘         │  ├── Field Extraction               │
                            │  └── Verification Engine            │
                            │                                     │
                            │  Azure Container Registry           │
                            └─────────────────────────────────────┘
```

## Technology Choices & Constraints

### Assessment Constraints

The assessment specified:
- Build a **standalone prototype** (no direct TTB COLA database integration)
- Target processing time of **~5 seconds** per label
- **"Grandmother-friendly" UI** - simple, accessible interface
- Support both **single label** and **batch verification**

### Backend: FastAPI + EasyOCR

| Choice | Reasoning |
|--------|-----------|
| **FastAPI** | Async support, automatic OpenAPI docs, Pydantic validation, excellent for ML workloads |
| **EasyOCR** | Offline-capable, no API keys required, supports CPU-only deployment |
| **OpenCV** | Industry-standard image preprocessing (contrast, ROI detection) |
| **RapidFuzz** | Fast fuzzy string matching for brand name verification |

#### Why EasyOCR over Cloud OCR?

The original plan was to use **PaddleOCR**, but I encountered critical compatibility issues:

```
NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support 
[pir::ArrayAttribute<pir::DoubleAttribute>]
```

This error occurred in Azure's Linux/AMD64 environment due to PaddleOCR's dependency on PaddlePaddle, which has architecture-specific issues.

**EasyOCR** was chosen as the alternative because:
1. **PyTorch-based** - More stable cross-platform support
2. **No cloud dependency** - Works offline, no API rate limits or costs
3. **CPU-compatible** - Runs without GPU (important for cost-effective cloud deployment)

**Tradeoff**: EasyOCR on CPU is slower than cloud OCR APIs (~1-2 seconds), but with optimizations (2 vCPU, "Detect Once" architecture, zone slicing), processing time is typically **3-6 seconds** per label. This eliminates external dependencies and ongoing API costs.

### Frontend: React + Vite

| Choice | Reasoning |
|--------|-----------|
| **React** | Component-based architecture, large ecosystem |
| **Vite** | Fast development builds, optimized production bundles |
| **react-dropzone** | Accessible drag-and-drop file uploads |
| **Papa Parse** | Robust CSV parsing for batch uploads |

### Infrastructure: Azure + Terraform

| Choice | Reasoning |
|--------|-----------|
| **Azure Container Apps** | Serverless containers with auto-scaling, managed HTTPS |
| **Azure Static Web Apps** | Free tier for SPAs, global CDN, automatic HTTPS |
| **Azure Container Registry** | Private Docker registry, integrated with Container Apps |
| **Terraform** | Infrastructure as code, reproducible deployments |

## Challenges & Solutions

### Challenge 1: OCR Performance on CPU

**Problem**: EasyOCR processing took 15-20 seconds per image initially.

**Solution**: Implemented "Detect Once" architecture:
- Single OCR pass on the full image (detection is the expensive part)
- Slice detected text boxes by Y-position for field extraction
- Combined with 2 vCPU / 4Gi resources, reduced processing time to **3-6 seconds**

```python
# Before: 5 separate OCR calls
brand_text = ocr.process(brand_crop)      # ~3s
type_text = ocr.process(type_crop)        # ~3s
abv_text = ocr.process(abv_crop)          # ~3s
# Total: ~15s

# After: 1 OCR call, slice by position
all_boxes = ocr.detect_once(full_image)   # ~2-4s
brand_text = slice_boxes(all_boxes, y_range=(0, 0.25))
type_text = slice_boxes(all_boxes, y_range=(0.15, 0.50))
# Total: ~2.5-6s depending on image complexity
```

### Challenge 2: Azure Container Apps Resource Limits

**Problem**: Attempted to use 4 vCPU / 8Gi memory, but Consumption tier limits are 2 vCPU / 4Gi.

```
Error: ContainerAppInvalidResourceTotal
Total CPU and memory must be one of: [cpu: 2.0, memory: 4.0Gi]
```

**Solution**: 
- Accepted the 2 vCPU / 4Gi limit
- Optimized for sequential processing (`MAX_WORKERS=1`)
- Tuned thread settings (`OMP_NUM_THREADS=2`, `TORCH_NUM_THREADS=2`)
- Documented that Dedicated workload profile would enable faster processing

### Challenge 3: Brand Name Extraction Inconsistency

**Problem**: OCR sometimes returned "TOM OLD" instead of "OLD TOM DISTILLERY", causing verification failures.

**Root Cause**: Word order varied based on OCR detection order.

**Solution**: Multi-strategy extraction with candidate scoring:
1. Search for brand suffixes (DISTILLERY, BREWING, WINERY)
2. Score candidates by similarity to expected value
3. Use token-set matching (order-insensitive)
4. "Rescue" short matches by searching raw text for extended versions

### Challenge 4: Class/Type Detection for Wine/Beer

**Problem**: Zone-based slicing missed "KENTUCKY STRAIGHT BOURBON WHISKEY" when it appeared outside the expected zone.

**Solution**: Priority phrase search on full raw text:
```python
PRIORITY_PHRASES = [
    (r"KENTUCKY\s+STRAIGHT\s+BOURBON\s+WHISKEY", "Kentucky Straight Bourbon Whiskey"),
    (r"INDIA\s+PALE\s+ALE", "India Pale Ale"),
    (r"CABERNET\s+SAUVIGNON", "Cabernet Sauvignon"),
    # ... 30+ patterns
]

# Search FULL raw text first, before zone filtering
for pattern, canonical in PRIORITY_PHRASES:
    if re.search(pattern, full_raw_text):
        return canonical
```

### Challenge 5: Stylized Font OCR Accuracy

**Problem**: Decorative fonts on craft beer labels produced garbled OCR output (e.g., "MOUNTAII [BREW 091" instead of "MOUNTAIN BREW CO").

**Reality Check**: This is a fundamental limitation of offline OCR models trained on standard fonts.

**Mitigations**:
1. Hard rejection of bracket/noise artifacts in candidate scoring
2. Fuzzy matching against expected values to find best candidate
3. Clear documentation that production deployment would use Azure Computer Vision for improved accuracy

**Tradeoff Acknowledged**: Prototype demonstrates the architecture; production would use cloud OCR APIs for stylized labels.

### Challenge 6: Net Contents Parsing Garbage

**Problem**: OCR artifacts like "(055" were being parsed as "55 mL" instead of finding "12 FL OZ (355 mL)".

**Solution**:
1. Tightened regex patterns with word boundaries
2. Added plausibility checks (reject values outside 100-2000 mL)
3. Snap to common sizes (354.9 → 355 mL)
4. Unit conversion fallback (oz → mL)

### Challenge 7: Docker Platform Mismatch

**Problem**: Images built on M1/M2 Mac (ARM64) failed on Azure (AMD64).

```
Error: image OS/Arc must be linux/amd64 but found linux/arm64
```

**Solution**: Always build with explicit platform flag:
```bash
docker build --platform linux/amd64 -t <image> .
```

## Tradeoffs Summary

| Decision | Benefit | Cost |
|----------|---------|------|
| Offline OCR (EasyOCR) | No API costs, no rate limits, works offline | Lower accuracy on stylized fonts |
| Detect Once + Zone Slicing | ~3-6s processing (down from 15s) | Slightly less accurate field localization |
| 2 vCPU / 4Gi (Consumption tier max) | Lower cost, serverless scaling, meets ~5s target | Limited headroom for complex labels |
| Sequential batch processing | Stable memory usage, no OOM | Slower batch throughput |
| Fuzzy matching | Handles OCR errors gracefully | May accept incorrect matches (mitigated by review status) |

## Performance Metrics

Tested on Azure Container Apps (2 vCPU / 4Gi) with optimizations:

| Metric | Value |
|--------|-------|
| **Clean labels (bourbon, wine)** | **2.4-3.8 seconds** |
| **Complex labels (beer, stylized)** | **5.9-6.3 seconds** |
| Preprocessing (resize, contrast) | ~70-100ms |
| OCR detection (Detect Once) | ~2000-4500ms |
| Field extraction (zone slicing) | ~0-1ms |
| Verification | ~0-1ms |
| Image compression (PNG→JPEG) | 17-19x ratio |
| Batch (10 labels, sequential) | ~30-50 seconds |

**Actual log examples:**
```
# Bourbon label (clean fonts)
total=2414ms | detect=2067ms, slice=0ms

# Wine label (clean fonts)  
total=2510ms | detect=2137ms, slice=0ms

# Beer label (stylized fonts, needs fallback)
total=6229ms | detect=4427ms + fallbacks
```

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- Docker
- Azure CLI (for deployment)
- Terraform (for infrastructure)

### Local Development

**Backend:**
```bash
git checkout backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
git checkout frontend
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev
```

### Deployment

See the `infra` branch for Terraform configurations:
```bash
git checkout infra
cd infra/backend
terraform init && terraform apply

cd ../frontend
terraform init && terraform apply
```

## API Documentation

When running locally: http://localhost:8000/docs

Key endpoints:
- `POST /api/v1/verify` - Single label verification
- `POST /api/v1/verify/batch` - Batch verification
- `GET /api/v1/health` - Health check

## Future Improvements

1. **Cloud OCR Integration**: Add Azure Computer Vision as optional backend for improved accuracy on stylized labels

2. **Confidence Thresholds**: Allow users to configure match/review/mismatch thresholds

3. **Label Region Detection**: Use object detection to isolate label from bottle before OCR

4. **Caching**: Cache OCR results by image hash to speed up repeated verifications

5. **Dedicated Workload Profile**: Upgrade to 4+ vCPU for faster processing

## License

This project was created as a technical assessment demonstrating AI-powered document verification capabilities.
