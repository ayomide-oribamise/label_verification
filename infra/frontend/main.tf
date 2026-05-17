terraform {
  required_version = ">= 1.0.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
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
  resource_group_name = var.resource_group_name == "" ? "rg-${local.resource_prefix}-frontend" : var.resource_group_name
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

resource "azurerm_static_web_app" "main" {
  name                = "swa-${local.resource_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku_tier            = var.sku_tier
  sku_size            = var.sku_tier
  tags                = local.tags
}

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
  description = "Deployment token for SWA CLI"
  value       = azurerm_static_web_app.main.api_key
  sensitive   = true
}
