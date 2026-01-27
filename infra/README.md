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

## Deployed URLs

- **Backend API**: https://ca-labelverify-dev-api.blackcoast-a65d3b7e.eastus.azurecontainerapps.io
- **API Docs**: https://ca-labelverify-dev-api.blackcoast-a65d3b7e.eastus.azurecontainerapps.io/docs
