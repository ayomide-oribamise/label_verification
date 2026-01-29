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
  description = "CPU cores for container (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)"
  type        = number
  default     = 2.0  # Increased for EasyOCR performance (was 1.0)
}

variable "container_memory" {
  description = "Memory for container (e.g., 0.5Gi, 1Gi, 2Gi, 4Gi)"
  type        = string
  default     = "4Gi"  # Increased for EasyOCR/PyTorch (was 2Gi)
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
