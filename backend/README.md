# Label Verification Backend

FastAPI backend service for AI-powered alcohol label verification using PaddleOCR.

## Quick Start

### Local Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy environment file:
```bash
cp .env.example .env
```

4. Run the server:
```bash
uvicorn app.main:app --reload --port 8000
```

5. Open API docs:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Running Tests

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest tests/test_api.py  # Run specific test file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/extract` | POST | Extract text from image (OCR only) |
| `/api/v1/verify` | POST | Verify single label against application data |
| `/api/v1/verify/batch` | POST | Verify multiple labels with CSV data |
| `/api/v1/ocr/boxes` | POST | Debug: Get raw OCR bounding boxes |

### Single Label Verification

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

CSV format:
```csv
filename,brand_name,class_type,abv_percent,net_contents_ml,has_warning
label1.png,OLD TOM DISTILLERY,Kentucky Bourbon,45,750,true
label2.png,JACK DANIELS,Tennessee Whiskey,40,1000,true
```

## Docker

### Build and Run Locally

```bash
# Build
docker build -t label-verification-backend .

# Run
docker run -p 8000:8000 label-verification-backend

# Or use docker-compose
docker-compose up --build
```

### Resource Requirements

- **Memory**: Minimum 1GB, recommended 2GB
- **CPU**: 1 core minimum
- **Startup time**: ~60-90 seconds (OCR model loading)

## Deployment

### Azure Container Apps (Recommended)

#### Option 1: Using deployment script

```bash
cd deploy
chmod +x deploy.sh
./deploy.sh <resource-group> <location>
```

#### Option 2: Using Azure CLI manually

```bash
# Login to Azure
az login

# Create resource group
az group create --name label-verification-rg --location eastus

# Create Container Registry
az acr create --resource-group label-verification-rg --name labelverificationacr --sku Basic --admin-enabled true

# Build and push image
az acr build --registry labelverificationacr --image label-verification-backend:latest .

# Deploy using Bicep
az deployment group create \
  --resource-group label-verification-rg \
  --template-file deploy/azure-container-app.bicep \
  --parameters appName=label-verification-api \
               containerImage=labelverificationacr.azurecr.io/label-verification-backend:latest \
               containerRegistry=labelverificationacr.azurecr.io \
               minReplicas=1
```

#### Option 3: GitHub Actions CI/CD

1. Copy `deploy/github-actions-deploy.yml` to `.github/workflows/deploy.yml`
2. Add the following secrets to your repository:
   - `AZURE_CREDENTIALS`: Azure service principal JSON
   - `ACR_LOGIN_SERVER`: Your ACR login server
   - `ACR_USERNAME`: ACR username
   - `ACR_PASSWORD`: ACR password

3. Push to `backend` branch to trigger deployment

### Important: Cold Start Prevention

Set `minReplicas=1` to keep at least one instance always running:
- Avoids cold start delays (OCR model loading takes ~60s)
- Ensures consistent response times

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration settings
│   ├── api/
│   │   └── routes.py        # API route definitions
│   ├── models/
│   │   └── schemas.py       # Pydantic schemas
│   └── services/
│       ├── preprocessing.py # Image preprocessing (OpenCV)
│       ├── ocr.py           # PaddleOCR integration
│       ├── extraction.py    # Field extraction logic
│       ├── verification.py  # Verification/matching logic
│       └── batch.py         # Batch processing
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   ├── test_extraction.py
│   ├── test_verification.py
│   └── test_batch.py
├── deploy/
│   ├── azure-container-app.bicep  # Azure Bicep template
│   ├── deploy.sh                  # Deployment script
│   └── github-actions-deploy.yml  # CI/CD workflow
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Configuration

See `.env.example` for all configuration options.

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_MAX_SIZE_MB` | 3 | Maximum upload size in MB |
| `IMAGE_MAX_DIMENSION` | 1500 | Max width/height (larger images resized) |
| `ABV_TOLERANCE` | 0.5 | ABV matching tolerance (±%) |
| `BRAND_MATCH_THRESHOLD` | 0.95 | Fuzzy match threshold for brand (pass) |
| `BRAND_REVIEW_THRESHOLD` | 0.85 | Fuzzy match threshold for brand (review) |
| `BATCH_MAX_FILES` | 50 | Maximum files per batch |
| `BATCH_MAX_WORKERS` | 4 | Max parallel workers for batch |

## Test Coverage

```
101 tests across 5 test files:
- test_api.py          - 10 tests (API endpoints)
- test_preprocessing.py - 10 tests (image processing)
- test_extraction.py   - 28 tests (field extraction)
- test_verification.py - 28 tests (matching logic)
- test_batch.py        - 25 tests (CSV parsing, batch)
```
