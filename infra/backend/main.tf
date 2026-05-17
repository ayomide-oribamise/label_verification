terraform {
  required_version = ">= 1.0.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id            = var.subscription_id
  tenant_id                  = var.tenant_id
  skip_provider_registration = var.skip_provider_registration
}

locals {
  name_suffix         = var.name_suffix == "" ? "" : "-${var.name_suffix}"
  resource_prefix     = "${var.project_name}-${var.environment}${local.name_suffix}"
  resource_group_name = var.resource_group_name == "" ? "rg-${local.resource_prefix}" : var.resource_group_name
  tags = {
    Project     = "Label Verification"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "azurerm_resource_group" "main" {
  name     = local.resource_group_name
  location = var.location
  tags     = local.tags
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.resource_prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

resource "azurerm_container_registry" "main" {
  name                = replace("acr${local.resource_prefix}", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${local.resource_prefix}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags
}

resource "azurerm_container_app" "api" {
  name                         = "ca-${local.resource_prefix}-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = local.tags

  registry {
    server               = azurerm_container_registry.main.login_server
    username             = azurerm_container_registry.main.admin_username
    password_secret_name = "registry-password"
  }

  secret {
    name  = "registry-password"
    value = azurerm_container_registry.main.admin_password
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    container {
      name   = "api"
      image  = "${azurerm_container_registry.main.login_server}/label-verification-backend:${var.container_image_tag}"
      cpu    = var.container_cpu
      memory = var.container_memory

      env {
        name  = "MAX_UPLOAD_SIZE_MB"
        value = "15"
      }

      env {
        name  = "MAX_CONVERTED_SIZE_MB"
        value = "3"
      }

      env {
        name  = "MAX_IMAGE_DIMENSION"
        value = "1024"
      }

      env {
        name  = "MAX_BATCH_SIZE"
        value = "50"
      }

      env {
        name  = "MAX_WORKERS"
        value = "1"
      }

      env {
        name  = "OCR_MAX_CONCURRENT"
        value = "1"
      }

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
        name  = "BLIS_NUM_THREADS"
        value = "2"
      }

      env {
        name  = "FLAGS_enable_pir_api"
        value = "0"
      }

      env {
        name  = "FLAGS_use_mkldnn"
        value = "1"
      }

      env {
        name  = "FLAGS_allocator_strategy"
        value = "auto_growth"
      }

      startup_probe {
        transport               = "HTTP"
        path                    = "/api/v1/health"
        port                    = 8000
        interval_seconds        = 30
        failure_count_threshold = 10
      }

      liveness_probe {
        transport               = "HTTP"
        path                    = "/api/v1/health"
        port                    = 8000
        interval_seconds        = 30
        failure_count_threshold = 3
      }

      readiness_probe {
        transport               = "HTTP"
        path                    = "/api/v1/health"
        port                    = 8000
        interval_seconds        = 10
        failure_count_threshold = 3
      }
    }

    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    http_scale_rule {
      name                = "http-scaling"
      concurrent_requests = 10
    }
  }
}

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "container_registry_login_server" {
  description = "ACR login server"
  value       = azurerm_container_registry.main.login_server
}

output "container_registry_name" {
  description = "ACR resource name for az acr login"
  value       = azurerm_container_registry.main.name
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
