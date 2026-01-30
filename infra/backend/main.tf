# Azure Backend Infrastructure for Label Verification API
# Deploys: Resource Group, Container Registry, Container Apps Environment, Container App

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }
}

provider "azurerm" {
  features {}
  skip_provider_registration = true
}

# Locals
locals {
  resource_prefix = "${var.project_name}-${var.environment}"
  tags = {
    Project     = "Label Verification"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-${local.resource_prefix}"
  location = var.location
  tags     = local.tags
}

# Log Analytics Workspace (required for Container Apps)
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.resource_prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

# Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = replace("acr${local.resource_prefix}", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

# Container Apps Environment
resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${local.resource_prefix}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags
}

# Container App for Backend API
resource "azurerm_container_app" "api" {
  name                         = "ca-${local.resource_prefix}-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = local.tags

  # Registry credentials
  registry {
    server               = azurerm_container_registry.main.login_server
    username             = azurerm_container_registry.main.admin_username
    password_secret_name = "registry-password"
  }

  secret {
    name  = "registry-password"
    value = azurerm_container_registry.main.admin_password
  }

  # Ingress configuration
  # Note: Azure Container Apps provides HTTPS by default with managed TLS
  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "http"    # Internal: container serves HTTP, Azure handles TLS termination

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
    # Note: CORS is handled at application level (FastAPI middleware)
  }

  template {
    # Container configuration
    # Note: EasyOCR/PyTorch on CPU requires significant resources
    # - 2 vCPU / 4Gi is max for Consumption tier (~8-10s OCR time)
    # - For faster OCR (~5s), use Dedicated workload profile with 4 vCPU
    # - Sequential batch processing (max_workers=1) to avoid OOM
    container {
      name   = "api"
      image  = "${azurerm_container_registry.main.login_server}/label-verification-backend:${var.container_image_tag}"
      cpu    = var.container_cpu
      memory = var.container_memory

      # Environment variables (matching backend config.py settings)
      env {
        name  = "MAX_IMAGE_SIZE_MB"
        value = "3"
      }

      env {
        name  = "MAX_IMAGE_DIMENSION"
        value = "1024"  # Optimized for speed
      }

      env {
        name  = "MAX_BATCH_SIZE"
        value = "50"
      }

      env {
        name  = "MAX_WORKERS"
        value = "1"  # Sequential processing - safer on limited CPU
      }

      env {
        name  = "OCR_MAX_CONCURRENT"
        value = "1"  # Prevent concurrent OCR - CPU bound
      }

      # Thread settings for 2 vCPU (Consumption tier max)
      env {
        name  = "OMP_NUM_THREADS"
        value = "2"
      }

      env {
        name  = "MKL_NUM_THREADS"
        value = "2"
      }

      env {
        name  = "OPENBLAS_NUM_THREADS"
        value = "2"
      }

      env {
        name  = "TORCH_NUM_THREADS"
        value = "2"
      }

      # Startup probe - give OCR model time to load
      # Max 10 failures × 30s interval = 5 minutes for model loading
      startup_probe {
        transport               = "HTTP"
        path                    = "/api/v1/health"
        port                    = 8000
        interval_seconds        = 30
        failure_count_threshold = 10
      }

      # Liveness probe
      liveness_probe {
        transport               = "HTTP"
        path                    = "/api/v1/health"
        port                    = 8000
        interval_seconds        = 30
        failure_count_threshold = 3
      }

      # Readiness probe
      readiness_probe {
        transport               = "HTTP"
        path                    = "/api/v1/health"
        port                    = 8000
        interval_seconds        = 10
        failure_count_threshold = 3
      }
    }

    # Scale configuration - min 1 to avoid cold starts
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    http_scale_rule {
      name                = "http-scaling"
      concurrent_requests = 10
    }
  }
}

# Outputs
output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "container_registry_login_server" {
  description = "ACR login server"
  value       = azurerm_container_registry.main.login_server
}

output "container_registry_admin_username" {
  description = "ACR admin username"
  value       = azurerm_container_registry.main.admin_username
  sensitive   = true
}

output "container_registry_admin_password" {
  description = "ACR admin password"
  value       = azurerm_container_registry.main.admin_password
  sensitive   = true
}

output "api_url" {
  description = "Backend API URL"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "api_fqdn" {
  description = "Backend API FQDN"
  value       = azurerm_container_app.api.ingress[0].fqdn
}