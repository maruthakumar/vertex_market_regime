# Terraform Variables for Market Regime Master Framework
# Version: 1.0
# Date: 2025-08-10

variable "project_id" {
  description = "GCP Project ID for Market Regime Framework"
  type        = string
  validation {
    condition     = length(var.project_id) > 6 && length(var.project_id) <= 30
    error_message = "Project ID must be between 6 and 30 characters."
  }
}

variable "billing_account_id" {
  description = "Billing Account ID for cost management"
  type        = string
  validation {
    condition     = can(regex("^[0-9A-F]{6}-[0-9A-F]{6}-[0-9A-F]{6}$", var.billing_account_id))
    error_message = "Billing account ID must be in the format XXXXXX-XXXXXX-XXXXXX."
  }
}

variable "region" {
  description = "GCP Region for resource deployment"
  type        = string
  default     = "us-central1"
  validation {
    condition     = contains(["us-central1", "us-east1", "us-west1", "europe-west1"], var.region)
    error_message = "Region must be one of: us-central1, us-east1, us-west1, europe-west1."
  }
}

variable "zone" {
  description = "GCP Zone for compute resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cost_budget_amount" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 500
  validation {
    condition     = var.cost_budget_amount >= 100 && var.cost_budget_amount <= 10000
    error_message = "Budget amount must be between $100 and $10,000."
  }
}

variable "vertex_ai_config" {
  description = "Vertex AI configuration settings"
  type = object({
    machine_type = string
    gpu_type     = string
    gpu_count    = number
    disk_size_gb = number
  })
  default = {
    machine_type = "n1-standard-4"
    gpu_type     = "NVIDIA_TESLA_T4"
    gpu_count    = 1
    disk_size_gb = 100
  }
}

variable "bigquery_config" {
  description = "BigQuery configuration settings"
  type = object({
    location                    = string
    default_table_expiration_ms = number
  })
  default = {
    location                    = "US"
    default_table_expiration_ms = 7776000000 # 90 days
  }
}

variable "storage_config" {
  description = "Cloud Storage configuration settings"
  type = object({
    location      = string
    storage_class = string
    versioning    = bool
  })
  default = {
    location      = "US-CENTRAL1"
    storage_class = "STANDARD"
    versioning    = true
  }
}

variable "network_config" {
  description = "Network configuration settings"
  type = object({
    subnet_cidr          = string
    services_cidr        = string
    enable_private_access = bool
  })
  default = {
    subnet_cidr           = "10.1.0.0/24"
    services_cidr         = "10.1.1.0/24"
    enable_private_access = true
  }
}

variable "monitoring_config" {
  description = "Monitoring and alerting configuration"
  type = object({
    email_address           = string
    cost_alert_threshold    = number
    performance_alert_threshold = number
  })
  default = {
    email_address              = "admin@marketregimeframework.com"
    cost_alert_threshold       = 100
    performance_alert_threshold = 1000 # 1 second latency threshold
  }
}

variable "security_config" {
  description = "Security configuration settings"
  type = object({
    allowed_ip_ranges = list(string)
    enable_audit_logs = bool
  })
  default = {
    allowed_ip_ranges = ["0.0.0.0/0"] # Restrict in production
    enable_audit_logs = true
  }
}

variable "data_retention_config" {
  description = "Data retention policies"
  type = object({
    raw_data_retention_days     = number
    feature_data_retention_days = number
    model_retention_days        = number
  })
  default = {
    raw_data_retention_days     = 180 # 6 months
    feature_data_retention_days = 365 # 1 year
    model_retention_days        = 90  # 3 months
  }
}

variable "parquet_uris" {
  description = "List of GCS URIs for Parquet files to register as a BigQuery external table (e.g., gs://bucket/path/*.parquet)"
  type        = list(string)
  default     = []
}

# Local values for computed configurations
locals {
  project_prefix = "mr-${var.environment}"
  
  common_labels = {
    project     = "market-regime-framework"
    environment = var.environment
    managed_by  = "terraform"
    team        = "quant-research"
    cost_center = "trading-systems"
  }
  
  # Vertex AI Endpoint configurations
  vertex_endpoints = {
    component_1 = {
      name         = "triple-straddle-analyzer"
      machine_type = "n1-standard-2"
      min_replicas = 1
      max_replicas = 3
    }
    component_2 = {
      name         = "greeks-sentiment-analyzer"
      machine_type = "n1-standard-2"
      min_replicas = 1
      max_replicas = 3
    }
    master_integration = {
      name         = "master-regime-classifier"
      machine_type = "n1-standard-4"
      min_replicas = 1
      max_replicas = 5
    }
  }
  
  # BigQuery table configurations
  bigquery_tables = {
    component_analysis_results = {
      partition_field = "timestamp"
      clustering_fields = ["symbol", "component_id"]
      expiration_ms = var.bigquery_config.default_table_expiration_ms
    }
    adaptive_learning_weights = {
      partition_field = "last_updated"
      clustering_fields = ["component_id", "dte_bucket"]
      expiration_ms = 31536000000 # 1 year
    }
    master_regime_analysis = {
      partition_field = "timestamp"
      clustering_fields = ["symbol", "master_regime"]
      expiration_ms = 31536000000 # 1 year
    }
  }
  
  # Storage bucket configurations
  storage_buckets = {
    model_artifacts = {
      name = "${local.project_prefix}-model-artifacts"
      lifecycle_age = 90
    }
    data_lake = {
      name = "${local.project_prefix}-data-lake"
      lifecycle_age = var.data_retention_config.raw_data_retention_days
    }
    feature_store = {
      name = "${local.project_prefix}-feature-store"
      lifecycle_age = var.data_retention_config.feature_data_retention_days
    }
    model_registry = {
      name = "${local.project_prefix}-model-registry"
      lifecycle_age = var.data_retention_config.model_retention_days
    }
  }
}