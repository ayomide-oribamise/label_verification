# Infrastructure - Azure Deployment

Terraform configurations for deploying the Label Verification app to Azure.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Azure Cloud                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐     ┌────────────────────────────────┐ │
│  │  Static Web App     │     │  Container Apps Environment    │ │
│  │  (Frontend)         │     │                                │ │
│  │  React/Vite SPA     │────▶│  Container App (API)           │ │
│  │                     │     │  - FastAPI + PaddleOCR         │ │
│  └─────────────────────┘     │  - Auto-scaling 1-3 replicas   │ │
│                              │                                │ │
│                              │  Container Registry (ACR)      │ │
│                              └────────────────────────────────┘ │
│                                                                  │
│  Log Analytics Workspace (Monitoring)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Structure

```
infra/
├── backend/       # Container App, ACR, Log Analytics
│   ├── main.tf
│   └── variables.tf
├── frontend/      # Azure Static Web App
│   ├── main.tf
│   └── variables.tf
└── README.md
```

## Quick Start

### Prerequisites
- Azure CLI (`az login`)
- Terraform 1.0+
- Docker

### Deploy Backend

```bash
cd infra/backend
terraform init
terraform apply

# Note the outputs: api_url, container_registry_login_server
```

### Push Docker Image

```bash
# Login to ACR
az acr login --name <acr-name>

# Build and push
cd backend
docker build -t <acr-server>/label-verification-backend:latest .
docker push <acr-server>/label-verification-backend:latest
```

### Deploy Frontend

```bash
cd infra/frontend
terraform init
terraform apply

# Note the output: static_web_app_url, deployment_token
```

### Deploy Frontend Code

```bash
cd frontend
npm run build

# Deploy using SWA CLI
npx @azure/static-web-apps-cli deploy ./dist --deployment-token <token>
```

## Viewing Logs

```bash
az containerapp logs show \
  --name ca-labelverify-dev-api \
  --resource-group rg-labelverify-dev \
  --follow
```

## Cleanup

```bash
cd infra/frontend && terraform destroy
cd ../backend && terraform destroy
```
