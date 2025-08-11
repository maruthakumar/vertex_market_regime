# Market Regime Master Framework - GCP Infrastructure
# Infrastructure as Code (IaC) for Vertex AI Integration
# Version: 1.0
# Date: 2025-08-10

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.84.0"
    }
  }
}

# Provider Configuration
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Variables
variable "project_id" {
  description = "GCP Project ID for Market Regime Framework"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "cost_budget_amount" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 500
}

# Local Variables
locals {
  project_name = "market-regime-framework"
  common_tags = {
    Project     = local.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Component   = "MarketRegimeAI"
  }
}

# Enable Required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",     # Vertex AI
    "bigquery.googleapis.com",       # BigQuery
    "storage.googleapis.com",        # Cloud Storage
    "compute.googleapis.com",        # Compute Engine (for VPN)
    "iam.googleapis.com",           # Identity and Access Management
    "cloudresourcemanager.googleapis.com",
    "servicenetworking.googleapis.com",
    "monitoring.googleapis.com",     # Cloud Monitoring
    "logging.googleapis.com"         # Cloud Logging
  ])

  service                    = each.value
  disable_dependent_services = true
}

# Service Accounts
resource "google_service_account" "vertex_ai_service_account" {
  account_id   = "${local.project_name}-vertex-ai"
  display_name = "Market Regime Vertex AI Service Account"
  description  = "Service account for Vertex AI operations"
  
  depends_on = [google_project_service.apis]
}

resource "google_service_account" "data_pipeline_service_account" {
  account_id   = "${local.project_name}-data-pipeline"
  display_name = "Market Regime Data Pipeline Service Account"
  description  = "Service account for data pipeline operations"
  
  depends_on = [google_project_service.apis]
}

# IAM Roles - Least Privilege Access
resource "google_project_iam_member" "vertex_ai_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/aiplatform.serviceAgent",
    "roles/storage.objectViewer",
    "roles/bigquery.dataViewer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.vertex_ai_service_account.email}"
}

resource "google_project_iam_member" "data_pipeline_roles" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.data_pipeline_service_account.email}"
}

# Cloud Storage Buckets
resource "google_storage_bucket" "model_artifacts" {
  name          = "${local.project_name}-model-artifacts-${var.environment}"
  location      = "US-CENTRAL1"
  storage_class = "STANDARD"
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.common_tags
  
  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "data_lake" {
  name          = "${local.project_name}-data-lake-${var.environment}"
  location      = "US-CENTRAL1"
  storage_class = "STANDARD"
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 180  # 6 months retention for raw data
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.common_tags
  
  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "feature_store" {
  name          = "${local.project_name}-feature-store-${var.environment}"
  location      = "US-CENTRAL1"
  storage_class = "STANDARD"
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365  # 1 year retention for features
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.common_tags
  
  depends_on = [google_project_service.apis]
}

# BigQuery Dataset
resource "google_bigquery_dataset" "market_regime_dataset" {
  dataset_id    = "market_regime_${var.environment}"
  friendly_name = "Market Regime Master Framework Dataset"
  description   = "Dataset for market regime analysis and ML training data"
  location      = "US"
  
  default_table_expiration_ms = 7776000000  # 90 days
  
  labels = local.common_tags
  
  depends_on = [google_project_service.apis]
}

# BigQuery Tables
resource "google_bigquery_table" "component_analysis_results" {
  dataset_id = google_bigquery_dataset.market_regime_dataset.dataset_id
  table_id   = "component_analysis_results"
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["symbol", "component_id"]
  
  schema = <<EOF
[
  {
    "name": "analysis_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Unique identifier for each analysis run"
  },
  {
    "name": "component_id",
    "type": "INTEGER",
    "mode": "REQUIRED",
    "description": "Component identifier (1-8)"
  },
  {
    "name": "timestamp",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Analysis execution time"
  },
  {
    "name": "symbol",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Asset symbol (NIFTY, BANKNIFTY, etc.)"
  },
  {
    "name": "regime_prediction",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Individual component regime prediction"
  },
  {
    "name": "confidence_score",
    "type": "FLOAT64",
    "mode": "REQUIRED",
    "description": "Component confidence (0.0-1.0)"
  },
  {
    "name": "processing_time_ms",
    "type": "INTEGER",
    "mode": "REQUIRED",
    "description": "Component processing time"
  },
  {
    "name": "weight_factor",
    "type": "FLOAT64",
    "mode": "REQUIRED",
    "description": "Current adaptive weight for this component"
  },
  {
    "name": "features",
    "type": "JSON",
    "mode": "NULLABLE",
    "description": "JSON object containing all component features"
  }
]
EOF

  labels = local.common_tags
}

resource "google_bigquery_table" "adaptive_learning_weights" {
  dataset_id = google_bigquery_dataset.market_regime_dataset.dataset_id
  table_id   = "adaptive_learning_weights"
  
  time_partitioning {
    type  = "DAY"
    field = "last_updated"
  }
  
  clustering = ["component_id", "dte_bucket"]
  
  schema = <<EOF
[
  {"name": "weight_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique weight record identifier"},
  {"name": "component_id", "type": "INTEGER", "mode": "REQUIRED", "description": "Component being weighted (1-8)"},
  {"name": "dte_bucket", "type": "STRING", "mode": "REQUIRED", "description": "DTE range (0-7, 8-30, 31+)"},
  {"name": "regime_context", "type": "STRING", "mode": "REQUIRED", "description": "Market regime context"},
  {"name": "weight_value", "type": "FLOAT64", "mode": "REQUIRED", "description": "Adaptive weight value"},
  {"name": "performance_metric", "type": "FLOAT64", "mode": "REQUIRED", "description": "Historical performance score"},
  {"name": "last_updated", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Weight update timestamp"}
]
EOF

  labels = local.common_tags
}

resource "google_bigquery_table" "master_regime_analysis" {
  dataset_id = google_bigquery_dataset.market_regime_dataset.dataset_id
  table_id   = "master_regime_analysis"
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["symbol", "master_regime"]
  
  schema = <<EOF
[
  {
    "name": "analysis_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Links to individual component analyses"
  },
  {
    "name": "symbol",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Asset symbol"
  },
  {
    "name": "timestamp",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Analysis timestamp"
  },
  {
    "name": "master_regime",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Final regime classification"
  },
  {
    "name": "master_confidence",
    "type": "FLOAT64",
    "mode": "REQUIRED",
    "description": "Overall system confidence"
  },
  {
    "name": "component_agreement",
    "type": "FLOAT64",
    "mode": "REQUIRED",
    "description": "Inter-component correlation score"
  },
  {
    "name": "processing_time_total_ms",
    "type": "INTEGER",
    "mode": "REQUIRED",
    "description": "Total 8-component processing time"
  },
  {
    "name": "vertex_ai_model_version",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "ML model version used"
  }
]
EOF

  labels = local.common_tags
}

resource "google_bigquery_table" "market_data_external" {
  count      = length(var.parquet_uris) > 0 ? 1 : 0
  dataset_id = google_bigquery_dataset.market_regime_dataset.dataset_id
  table_id   = "market_data_external"

  external_data_configuration {
    autodetect    = true
    source_format = "PARQUET"
    source_uris   = var.parquet_uris
  }

  labels = local.common_tags
}

# Vertex AI Workbench Instance
resource "google_notebooks_instance" "ml_workbench" {
  name         = "${local.project_name}-ml-workbench"
  location     = var.zone
  machine_type = "n1-standard-4"
  
  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf2-ent-2-11-cu113-notebooks"
  }
  
  accelerator_config {
    type       = "NVIDIA_TESLA_T4"
    core_count = 1
  }
  
  disk_size_gb = 100
  disk_type    = "PD_STANDARD"
  
  service_account = google_service_account.vertex_ai_service_account.email
  
  labels = local.common_tags
  
  depends_on = [google_project_service.apis]
}

# VPC Network for Secure Connectivity
resource "google_compute_network" "market_regime_network" {
  name                    = "${local.project_name}-network"
  auto_create_subnetworks = false
  
  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "market_regime_subnet" {
  name          = "${local.project_name}-subnet"
  ip_cidr_range = "10.1.0.0/24"
  region        = var.region
  network       = google_compute_network.market_regime_network.id
  
  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "10.1.1.0/24"
  }
  
  private_ip_google_access = true
}

# VPN Gateway for Secure Local Connectivity
resource "google_compute_vpn_gateway" "target_gateway" {
  name    = "${local.project_name}-vpn-gateway"
  network = google_compute_network.market_regime_network.id
  region  = var.region
  
  depends_on = [google_project_service.apis]
}

# Firewall Rules
resource "google_compute_firewall" "allow_vertex_ai" {
  name    = "${local.project_name}-allow-vertex-ai"
  network = google_compute_network.market_regime_network.name
  
  allow {
    protocol = "tcp"
    ports    = ["443", "8080"]
  }
  
  source_ranges = ["10.1.0.0/24"]
  target_tags   = ["vertex-ai"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${local.project_name}-allow-ssh"
  network = google_compute_network.market_regime_network.name
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  
  source_ranges = ["0.0.0.0/0"]  # Restrict this to your IP range in production
  target_tags   = ["ssh-allowed"]
}

# Budget and Cost Monitoring
resource "google_billing_budget" "project_budget" {
  billing_account = var.billing_account_id
  display_name    = "${local.project_name} Monthly Budget"
  
  budget_filter {
    projects = ["projects/${var.project_id}"]
  }
  
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.cost_budget_amount)
    }
  }
  
  threshold_rules {
    threshold_percent = 0.8
  }
  
  threshold_rules {
    threshold_percent = 0.9
    spend_basis      = "FORECASTED_SPEND"
  }
  
  all_updates_rule {
    monitoring_notification_channels = [
      google_monitoring_notification_channel.email_alerts.name
    ]
  }
}

# Monitoring and Alerting
resource "google_monitoring_notification_channel" "email_alerts" {
  display_name = "Email Alerts for Market Regime Framework"
  type         = "email"
  
  labels = {
    email_address = "admin@marketregimeframework.com"  # Update with actual email
  }
  
  depends_on = [google_project_service.apis]
}

resource "google_monitoring_alert_policy" "high_cost_alert" {
  display_name = "High Cost Alert - Market Regime Framework"
  combiner     = "OR"
  
  conditions {
    display_name = "High Vertex AI Cost"
    condition_threshold {
      filter          = "resource.type=\"aiplatform.googleapis.com/Endpoint\""
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 100
      duration        = "300s"
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = [
    google_monitoring_notification_channel.email_alerts.name
  ]
  
  alert_strategy {
    auto_close = "1800s"
  }
  
  depends_on = [google_project_service.apis]
}

# Cloud Functions for Data Pipeline Triggers (Optional)
resource "google_cloudfunctions_function" "data_pipeline_trigger" {
  name        = "${local.project_name}-data-pipeline-trigger"
  description = "Trigger for data pipeline processing"
  runtime     = "python39"
  
  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.model_artifacts.name
  source_archive_object = "functions/data-pipeline-trigger.zip"
  trigger {
    event_type = "google.storage.object.finalize"
    resource   = google_storage_bucket.data_lake.name
  }
  timeout               = 540
  entry_point          = "process_data"
  service_account_email = google_service_account.data_pipeline_service_account.email
  
  environment_variables = {
    BIGQUERY_DATASET = google_bigquery_dataset.market_regime_dataset.dataset_id
    PROJECT_ID       = var.project_id
  }
  
  depends_on = [google_project_service.apis]
}

# Outputs
output "vertex_ai_service_account_email" {
  description = "Email of the Vertex AI service account"
  value       = google_service_account.vertex_ai_service_account.email
}

output "data_pipeline_service_account_email" {
  description = "Email of the data pipeline service account"
  value       = google_service_account.data_pipeline_service_account.email
}

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "model_artifacts_bucket" {
  description = "Name of the model artifacts storage bucket"
  value       = google_storage_bucket.model_artifacts.name
}

output "data_lake_bucket" {
  description = "Name of the data lake storage bucket"
  value       = google_storage_bucket.data_lake.name
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.market_regime_dataset.dataset_id
}

output "ml_workbench_url" {
  description = "URL of the ML workbench instance"
  value       = "https://console.cloud.google.com/ai-platform/notebooks/instances/${var.zone}/${google_notebooks_instance.ml_workbench.name}"
}

output "vpn_gateway_ip" {
  description = "External IP of VPN gateway"
  value       = google_compute_vpn_gateway.target_gateway.id
}

# TODO: Add BigQuery table for adaptive_learning_weights and Vertex AI endpoint/feature store resources.

# Additional Variables File
variable "billing_account_id" {
  description = "Billing Account ID for cost management"
  type        = string
}