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
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "subscription_id" {
  description = "Azure subscription ID to deploy into. Leave null to use the active Azure CLI subscription."
  type        = string
  default     = null
  sensitive   = true
}

variable "tenant_id" {
  description = "Azure tenant ID for the subscription. Leave null to use the active Azure CLI tenant."
  type        = string
  default     = null
  sensitive   = true
}

variable "name_suffix" {
  description = "Optional short suffix to make globally-scoped resource names unique for a new Azure account (example: your initials or assessment run id)"
  type        = string
  default     = ""

  validation {
    condition     = var.name_suffix == "" || can(regex("^[a-z0-9]{2,8}$", var.name_suffix))
    error_message = "Name suffix must be blank or 2-8 lowercase alphanumeric characters."
  }
}

variable "resource_group_name" {
  description = "Optional explicit resource group name. Leave blank to use rg-<project>-<environment>-<suffix>."
  type        = string
  default     = ""

  validation {
    condition     = var.resource_group_name == "" || can(regex("^[A-Za-z0-9._()\\-]{1,90}$", var.resource_group_name))
    error_message = "Resource group name must be blank or a valid Azure resource group name."
  }
}

variable "skip_provider_registration" {
  description = "Set true only if the Azure account cannot register resource providers automatically"
  type        = bool
  default     = true
}

variable "container_image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "container_cpu" {
  description = "CPU cores for container (Consumption tier max: 2.0)"
  type        = number
  default     = 2.0
}

variable "container_memory" {
  description = "Memory for container (Consumption tier max: 4Gi)"
  type        = string
  default     = "4Gi"
}

variable "min_replicas" {
  description = "Minimum number of container replicas (1+ to avoid cold starts)"
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Maximum number of container replicas"
  type        = number
  default     = 3
}
