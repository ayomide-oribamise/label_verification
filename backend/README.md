# Label Verification Backend

FastAPI backend for AI-powered alcohol label verification using EasyOCR.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload --port 8000
```

API Documentation: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/verify` | POST | Verify single label |
| `/api/v1/verify/batch` | POST | Verify multiple labels |

### Single Label Verification

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -F "image=@label.png" \
  -F "brand_name=OLD TOM DISTILLERY" \
  -F "class_type=Kentucky Straight Bourbon Whiskey" \
  -F "abv_percent=45" \
  -F "net_contents_ml=750" \
  -F "has_warning=true"
```

### Batch Verification

```bash
curl -X POST "http://localhost:8000/api/v1/verify/batch" \
  -F "images=@label1.png" \
  -F "images=@label2.png" \
  -F "csv_file=@application_data.csv"
```

**CSV Format:**
```csv
image_filename,brand_name,class_type,abv_percent,net_contents_ml,has_warning
sample_bourbon.png,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45,750,true
sample_wine.png,SILVER OAK,Cabernet Sauvignon,14.5,750,true
```

## Docker

```bash
# Build for deployment (AMD64 required for Azure)
docker build --platform linux/amd64 -t label-verification-backend .

# Run locally
docker run -p 8000:8000 label-verification-backend
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Environment mode |
| `LOG_LEVEL` | INFO | Logging level |
| `MAX_UPLOAD_SIZE_MB` | 15 | Max image upload size |
| `OCR_MAX_CONCURRENT` | 1 | OCR concurrency limit |
| `MAX_WORKERS` | 1 | Batch processing workers |
| `CORS_ORIGINS` | * | Allowed CORS origins |

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── api/
│   │   └── routes.py        # API endpoint definitions
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── services/
│       ├── preprocessing.py # Image preprocessing pipeline
│       ├── ocr.py           # EasyOCR integration
│       ├── extraction.py    # Field extraction from OCR results
│       ├── verification.py  # Field matching and verification
│       └── batch.py         # Batch processing logic
├── tests/                   # Unit tests
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
└── .env.example             # Environment template
```

## Architecture

### Processing Pipeline

```
Image Upload
    ↓
Preprocessing (resize, contrast, ROI detection)
    ↓
OCR (EasyOCR - detect once, slice by position)
    ↓
Field Extraction (brand, class/type, ABV, net contents, warning)
    ↓
Verification (fuzzy matching against application data)
    ↓
Results (match/review/mismatch per field)
```

### Key Design Decisions

1. **Offline-First OCR**: Uses EasyOCR (PyTorch-based) for offline processing without cloud API dependencies

2. **Detect-Once Architecture**: Single OCR pass on full image, then slice detected text boxes by position for field extraction

3. **Priority Phrase Search**: Searches full raw text for known beverage phrases before zone-based extraction

4. **Fuzzy Matching**: Token-set similarity for brand matching, semantic classification for beverage types

5. **Candidate Rescoring**: During verification, rescores all extraction candidates against expected values

## Testing

```bash
pytest -v
```
