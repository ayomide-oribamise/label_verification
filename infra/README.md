# Infrastructure - Azure Deployment

Terraform configurations for deploying the Label Verification application to Azure.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            Azure Cloud                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐       ┌──────────────────────────────────┐ │
│  │  Static Web App     │       │  Container Apps Environment      │ │
│  │  (Frontend)         │       │                                  │ │
│  │  React/Vite SPA     │──────▶│  Container App (API)             │ │
│  │                     │ HTTPS │  - FastAPI + EasyOCR             │ │
│  └─────────────────────┘       │  - 2 vCPU / 4Gi Memory           │ │
│                                │  - Auto-scaling 1-3 replicas     │ │
│                                │                                  │ │
│                                │  Container Registry (ACR)        │ │
│                                │  - Backend Docker images         │ │
│                                └──────────────────────────────────┘ │
│                                                                      │
│  Log Analytics Workspace (Monitoring & Logs)                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Structure

```
infra/
├── backend/           # Container App, ACR, Log Analytics
│   ├── main.tf
│   └── variables.tf
├── frontend/          # Azure Static Web App
│   ├── main.tf
│   └── variables.tf
└── README.md
```

## Prerequisites

- Azure CLI installed and authenticated (`az login`)
- Terraform >= 1.0.0
- Azure subscription with required resource providers registered

## Deployment

### Backend

```bash
cd infra/backend

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy
terraform apply
```

After deployment, note the outputs:
- `container_registry_login_server` - ACR URL for pushing images
- `api_url` - Backend API endpoint

### Push Docker Image

```bash
# Login to ACR
az acr login --name <acr_name>

# Build and push (from backend directory)
docker build --platform linux/amd64 -t <acr_login_server>/label-verification-backend:latest .
docker push <acr_login_server>/label-verification-backend:latest

# Update container app with new image
az containerapp update \
  --name ca-labelverify-dev-api \
  --resource-group rg-labelverify-dev \
  --image <acr_login_server>/label-verification-backend:latest
```

### Frontend

```bash
cd infra/frontend

# Initialize and deploy
terraform init
terraform apply
```

Deploy the built frontend:
```bash
# Get deployment token
terraform output -raw deployment_token

# Deploy using SWA CLI
cd ../../frontend
npm run build
npx @azure/static-web-apps-cli deploy ./dist \
  --deployment-token <token> \
  --env production
```

## Configuration

### Backend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `project_name` | labelverify | Project name for resource naming |
| `environment` | dev | Environment (dev/staging/prod) |
| `location` | eastus | Azure region |
| `container_cpu` | 2.0 | CPU cores (max 2.0 for Consumption tier) |
| `container_memory` | 4Gi | Memory (max 4Gi for Consumption tier) |
| `min_replicas` | 1 | Minimum replicas |
| `max_replicas` | 3 | Maximum replicas |

### Frontend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `project_name` | labelverify | Project name for resource naming |
| `environment` | dev | Environment (dev/staging/prod) |
| `location` | eastus2 | Azure region (SWA limited regions) |
| `sku_tier` | Free | SKU tier (Free/Standard) |

## Resource Limits

Azure Container Apps Consumption tier limits:
- **Max CPU**: 2.0 vCPU per container
- **Max Memory**: 4Gi per container
- **OCR Processing**: ~8-12 seconds per image at 2 vCPU

For faster processing, use a Dedicated workload profile (4+ vCPU).

## Monitoring

View logs in Azure Portal:
1. Navigate to Container App → Monitoring → Log stream
2. Or use Log Analytics → Logs with KQL queries

```kusto
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "ca-labelverify-dev-api"
| order by TimeGenerated desc
| take 100
```

## Cleanup

```bash
# Destroy frontend
cd infra/frontend
terraform destroy

# Destroy backend
cd ../backend
terraform destroy
```
