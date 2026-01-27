# Azure Frontend Infrastructure for Label Verification UI
# Deploys: Azure Static Web App (perfect for React/Vite SPAs)

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

# Resource Group (can be shared with backend or separate)
resource "azurerm_resource_group" "main" {
  name     = "rg-${local.resource_prefix}-frontend"
  location = var.location
  tags     = local.tags
}

# Azure Static Web App
resource "azurerm_static_web_app" "main" {
  name                = "swa-${local.resource_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku_tier            = var.sku_tier
  sku_size            = var.sku_tier
  tags                = local.tags

  # Note: App settings (environment variables) are configured via:
  # - Azure Portal
  # - GitHub Actions during deployment
  # - Or using azurerm_static_web_app_custom_domain for custom domains
}

# Static Web App Environment Variable (API URL)
# Note: For Static Web Apps, environment variables are set during build time
# via GitHub Actions or Azure DevOps. This is documented in the outputs.

# Outputs
output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "static_web_app_name" {
  description = "Static Web App name"
  value       = azurerm_static_web_app.main.name
}

output "static_web_app_id" {
  description = "Static Web App resource ID"
  value       = azurerm_static_web_app.main.id
}

output "static_web_app_default_hostname" {
  description = "Default hostname for the Static Web App"
  value       = azurerm_static_web_app.main.default_host_name
}

output "static_web_app_url" {
  description = "URL of the deployed Static Web App"
  value       = "https://${azurerm_static_web_app.main.default_host_name}"
}

output "deployment_token" {
  description = "Deployment token for GitHub Actions (keep secret!)"
  value       = azurerm_static_web_app.main.api_key
  sensitive   = true
}

output "setup_instructions" {
  description = "Next steps for deployment"
  value       = <<-EOT
    
    ========================================
    FRONTEND DEPLOYMENT SETUP
    ========================================
    
    1. Get the deployment token:
       terraform output -raw deployment_token
    
    2. Add to GitHub repository secrets:
       - Name: AZURE_STATIC_WEB_APPS_API_TOKEN
       - Value: [the deployment token]
    
    3. Add backend API URL as build-time env var in GitHub Actions:
       env:
         VITE_API_URL: ${var.backend_api_url != "" ? var.backend_api_url : "<your-backend-url>"}
    
    4. Your frontend will be available at:
       https://${azurerm_static_web_app.main.default_host_name}
    
    ========================================
  EOT
}
