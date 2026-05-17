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
- Docker installed for the backend image build/push
- Node.js/npm installed for the frontend build/deploy
- New Azure account/subscription ID and tenant ID
- A short lowercase `name_suffix` for fresh redeploys, because Azure Container Registry names are globally unique

## Deployment

For a new Azure account, copy both example variable files first:

```bash
cp infra/backend/terraform.tfvars.example infra/backend/terraform.tfvars
cp infra/frontend/terraform.tfvars.example infra/frontend/terraform.tfvars
```

Edit both files and set:

- `subscription_id` and `tenant_id` for the new Azure account
- the same unique `name_suffix`, for example your initials plus a run number
- optional `resource_group_name` values if you want exact new resource group names instead of the generated names

Use a fresh Terraform state for the new account. Do not reuse local state from an old subscription when switching accounts, because Terraform state is tied to the resources it already created.

If you prefer Azure CLI context instead of explicit provider IDs, leave `subscription_id` and `tenant_id` as `null`, then run:

```bash
az login
az account set --subscription <new-subscription-id>
```

This repository uses branch-per-concern layout. The `infra` branch contains only Terraform. Build and deploy commands for app code must be run from a separate checkout or worktree of the matching `backend` or `frontend` branch.

### Backend

The first backend deployment is a two-step flow: create the registry and platform resources, push the image, then create/update the Container App. This avoids a fresh Azure Container App trying to pull an image before the new registry contains one.

```bash
cd infra/backend

# Initialize Terraform
terraform init

# Bootstrap Azure resources that do not require the backend image yet
terraform apply \
  -target=azurerm_resource_group.main \
  -target=azurerm_log_analytics_workspace.main \
  -target=azurerm_container_registry.main \
  -target=azurerm_container_app_environment.main
```

After bootstrap, note the output:

- `container_registry_login_server` - ACR URL for pushing images
- `container_registry_name` - ACR name for `az acr login`

### Push Docker Image

```bash
# Login to ACR.
az acr login --name <container_registry_name>

# Build and push from a separate checkout/worktree of the backend branch
cd backend
docker build --platform linux/amd64 -t <acr_login_server>/label-verification-backend:latest .
docker push <acr_login_server>/label-verification-backend:latest
```

Then finish backend Terraform:

```bash
# Return to the infra checkout/worktree
cd infra/backend
terraform plan
terraform apply
```

After final backend apply, note the output:

- `api_url` - Backend API endpoint for the frontend build

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

# Build from a separate checkout/worktree of the frontend branch
cd frontend
VITE_API_URL=<backend_api_url_from_backend_output> npm run build
npx @azure/static-web-apps-cli deploy ./dist \
  --deployment-token <token> \
  --env production
```

Important: `VITE_API_URL` is a Vite build-time variable. Set it when running `npm run build`; setting it only in the Azure portal after build will not update the already-built JavaScript bundle.

## Configuration

### Backend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `project_name` | labelverify | Project name for resource naming |
| `environment` | dev | Environment (dev/staging/prod) |
| `location` | eastus | Azure region |
| `subscription_id` | null | New Azure subscription ID; falls back to Azure CLI if null |
| `tenant_id` | null | Azure tenant ID; falls back to Azure CLI if null |
| `name_suffix` | blank | Optional suffix for globally unique resource names |
| `resource_group_name` | blank | Optional exact backend resource group name |
| `container_cpu` | 2.0 | CPU cores (max 2.0 for Consumption tier) |
| `container_memory` | 4Gi | Memory (max 4Gi for Consumption tier) |
| `min_replicas` | 1 | Minimum replicas |
| `max_replicas` | 3 | Maximum replicas |
| `skip_provider_registration` | false | Set true only when Azure provider registration is managed externally |

### Frontend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `project_name` | labelverify | Project name for resource naming |
| `environment` | dev | Environment (dev/staging/prod) |
| `location` | eastus2 | Azure region (SWA limited regions) |
| `subscription_id` | null | New Azure subscription ID; falls back to Azure CLI if null |
| `tenant_id` | null | Azure tenant ID; falls back to Azure CLI if null |
| `name_suffix` | blank | Optional suffix for globally unique resource names |
| `resource_group_name` | blank | Optional exact frontend resource group name |
| `sku_tier` | Free | SKU tier (Free/Standard) |
| `skip_provider_registration` | false | Set true only when Azure provider registration is managed externally |

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
