# Variables for Backend Infrastructure

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

variable "container_image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "container_cpu" {
  description = "CPU cores for container (Consumption tier max: 2.0)"
  type        = number
  default     = 2.0  # Max for Consumption tier (4 vCPU requires Dedicated profile)
}

variable "container_memory" {
  description = "Memory for container (Consumption tier max: 4Gi)"
  type        = string
  default     = "4Gi"  # Max for Consumption tier (8Gi requires Dedicated profile)
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
