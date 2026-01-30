# Label Verification Backend

FastAPI backend for AI-powered alcohol label verification using PaddleOCR.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload --port 8000

# Test
pytest -v
```

API docs: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/verify` | POST | Verify single label |
| `/api/v1/verify/batch` | POST | Verify multiple labels |

### Single Verification

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -F "image=@label.png" \
  -F "brand_name=OLD TOM DISTILLERY" \
  -F "class_type=Kentucky Bourbon" \
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

## Docker

```bash
docker build -t label-verification-backend .
docker run -p 8000:8000 label-verification-backend

# Or with docker-compose
docker-compose up --build
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings
│   ├── api/routes.py        # API endpoints
│   ├── models/schemas.py    # Pydantic models
│   └── services/
│       ├── preprocessing.py # Image preprocessing
│       ├── ocr.py           # PaddleOCR wrapper
│       ├── extraction.py    # Field extraction
│       ├── verification.py  # Matching logic
│       └── batch.py         # Batch processing
├── tests/                   # Unit tests
├── Dockerfile
├── requirements.txt
└── .env.example
```
