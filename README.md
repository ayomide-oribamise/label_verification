# AI-Powered Alcohol Label Verification

**Author:** Ayomide Oribamise

Automated prototype for verifying alcohol label images against submitted application data. The app extracts label text with OCR, normalizes key fields, and returns a clear verification result for single-label and batch workflows.

## Live Demo

[Open the deployed application](https://red-mushroom-08407180f-preview.eastus2.7.azurestaticapps.net)

To try the demo:

1. Open the deployed application.
2. Load one of the sample labels.
3. Run verification and review the field-level result.

The UI is designed around the assessment target of verifying labels in under 5 seconds after the backend OCR model is warm.

## What It Checks

- **Brand name** with fuzzy, order-insensitive matching.
- **Class/type** with semantic beverage matching.
- **Alcohol content (ABV)** with numeric normalization.
- **Net contents** with mL extraction and common-size handling.
- **Bottler/producer** as a flexible-location compliance field.
- **Country of origin** for imported products.
- **Government warning** as a flexible-location compliance field.

Flexible-location fields can legally appear on another panel of the package. When those fields are not visible in the uploaded image, the system returns **Review Required** instead of incorrectly marking the label as failed.

## Repository Layout

This repository is organized by branch:

| Branch | Purpose |
| --- | --- |
| `main` | Project overview and submission README |
| `frontend` | React/Vite frontend |
| `backend` | FastAPI OCR and verification API |
| `infra` | Terraform for Azure resources |

## Architecture

```text
Azure Static Web Apps
  React/Vite frontend
        |
        | HTTPS
        v
Azure Container Apps
  FastAPI backend
  PaddleOCR extraction
  verification engine
        |
        v
Azure Container Registry
  backend Docker image
```

Infrastructure is managed with Terraform. The backend is deployed as an Azure Container App, and the frontend is deployed as an Azure Static Web App.

## Backend Summary

The backend uses FastAPI with a PaddleOCR adapter and a detect-once OCR pipeline:

1. Preprocess uploaded image.
2. Run OCR once on the full image.
3. Slice detected text by position and apply full-text extraction fallbacks.
4. Compare extracted fields against application data.
5. Return `match`, `review`, `incomplete`, or `mismatch`.

The verification policy separates hard contradictions from fields that are merely absent from the uploaded image. This avoids false failures for required information that may appear on back or neck labels.

## Frontend Summary

The frontend supports:

- Single-label verification.
- Batch verification from multiple images.
- CSV template generation and upload.
- Built-in sample labels for quick testing.
- Field-level result tables with extracted, expected, and guidance columns.

The production build uses `VITE_API_URL` to point the frontend at the deployed backend API.

## Infrastructure Summary

Terraform creates:

- Azure Resource Groups.
- Azure Container Registry.
- Azure Log Analytics Workspace.
- Azure Container Apps Environment.
- Azure Container App for the API.
- Azure Static Web App for the frontend.

For first deployment, the backend infrastructure is bootstrapped first so the container registry exists before pushing the Docker image. After the image is pushed, the final Terraform apply creates or updates the API container app.

## Deployment Notes

Backend image build and push:

```bash
docker build --platform linux/amd64 -t <acr_login_server>/label-verification-backend:latest .
docker push <acr_login_server>/label-verification-backend:latest
```

Frontend build and deploy:

```bash
VITE_API_URL=<backend_api_url> npm run build
npx @azure/static-web-apps-cli deploy ./dist \
  --deployment-token <token> \
  --env production
```

## Validation

Recent submission checks:

- Backend Docker image builds and runs in Azure Container Apps.
- Backend OCR initializes successfully with PaddleOCR.
- Frontend lint passes.
- Frontend production build passes.
- Terraform format check passes.
- Terraform validate passes for backend and frontend infrastructure.

## Known Constraints

- First backend startup may take longer because PaddleOCR downloads and initializes models at runtime.
- Stylized labels are still OCR-sensitive, especially metallic, curved, or decorative text.
- Batch verification is processed conservatively to avoid memory pressure on the Azure Container Apps consumption tier.
- Static Web Apps require `VITE_API_URL` at build time; changing it in the Azure portal after build does not rewrite the generated JavaScript bundle.
