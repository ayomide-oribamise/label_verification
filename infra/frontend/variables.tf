# Variables for Frontend Infrastructure

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "labelverify"

  validation {
    condition     = can(regex("^[a-z0-9]+$", var.project_name))
    error_message = "Project name must be lowercase alphanumeric only."
  }
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region for resources (Static Web Apps: westus2, centralus, eastus2, westeurope, eastasia)"
  type        = string
  default     = "eastus2"
}

variable "backend_api_url" {
  description = "Backend API URL (from backend terraform output)"
  type        = string
  default     = ""
}

variable "sku_tier" {
  description = "Static Web App SKU tier"
  type        = string
  default     = "Free"

  validation {
    condition     = contains(["Free", "Standard"], var.sku_tier)
    error_message = "SKU tier must be Free or Standard."
  }
}
