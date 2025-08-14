# Terraform Configuration for IAM Service Accounts
# Story 2.5: IAM, Artifact Registry, Budgets/Monitoring
# Minimal-privilege service accounts for Vertex AI operations

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "arched-bot-269016"
}

variable "region" {
  description = "Primary region for resources"
  type        = string
  default     = "us-central1"
}

# Service Account: Vertex AI Pipeline
resource "google_service_account" "vertex_ai_pipeline" {
  account_id   = "vertex-ai-pipeline-sa"
  display_name = "Vertex AI Pipeline Service Account"
  description  = "Service account for Vertex AI training pipelines and model operations"
  project      = var.project_id
}

# Service Account: Vertex AI Serving
resource "google_service_account" "vertex_ai_serving" {
  account_id   = "vertex-ai-serving-sa"
  display_name = "Vertex AI Model Serving Service Account"
  description  = "Service account for Vertex AI model endpoints and serving"
  project      = var.project_id
}

# Service Account: Monitoring and Alerts
resource "google_service_account" "monitoring_alerts" {
  account_id   = "monitoring-alerts-sa"
  display_name = "Cloud Monitoring Service Account"
  description  = "Service account for monitoring, alerting, and budget management"
  project      = var.project_id
}

# IAM Role Bindings - Vertex AI Pipeline Service Account
resource "google_project_iam_member" "vertex_pipeline_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_project_iam_member" "vertex_pipeline_aiplatform_custom" {
  project = var.project_id
  role    = "roles/aiplatform.customCodeServiceAgent"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_project_iam_member" "vertex_pipeline_bigquery_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_project_iam_member" "vertex_pipeline_bigquery_data" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_project_iam_member" "vertex_pipeline_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_project_iam_member" "vertex_pipeline_storage_legacy" {
  project = var.project_id
  role    = "roles/storage.legacyBucketReader"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_project_iam_member" "vertex_pipeline_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_project_iam_member" "vertex_pipeline_token_creator" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

# IAM Role Bindings - Vertex AI Serving Service Account
resource "google_project_iam_member" "vertex_serving_predictor" {
  project = var.project_id
  role    = "roles/aiplatform.predictor"
  member  = "serviceAccount:${google_service_account.vertex_ai_serving.email}"
}

resource "google_project_iam_member" "vertex_serving_featurestore" {
  project = var.project_id
  role    = "roles/aiplatform.featurestoreUser"
  member  = "serviceAccount:${google_service_account.vertex_ai_serving.email}"
}

resource "google_project_iam_member" "vertex_serving_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.vertex_ai_serving.email}"
}

resource "google_project_iam_member" "vertex_serving_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.vertex_ai_serving.email}"
}

# IAM Role Bindings - Monitoring Service Account
resource "google_project_iam_member" "monitoring_editor" {
  project = var.project_id
  role    = "roles/monitoring.editor"
  member  = "serviceAccount:${google_service_account.monitoring_alerts.email}"
}

resource "google_project_iam_member" "monitoring_alert_policy" {
  project = var.project_id
  role    = "roles/monitoring.alertPolicyEditor"
  member  = "serviceAccount:${google_service_account.monitoring_alerts.email}"
}

resource "google_project_iam_member" "monitoring_billing_viewer" {
  project = var.project_id
  role    = "roles/billing.viewer"
  member  = "serviceAccount:${google_service_account.monitoring_alerts.email}"
}

resource "google_project_iam_member" "monitoring_budget_viewer" {
  project = var.project_id
  role    = "roles/billing.budgetsViewer"
  member  = "serviceAccount:${google_service_account.monitoring_alerts.email}"
}

resource "google_project_iam_member" "monitoring_logging_viewer" {
  project = var.project_id
  role    = "roles/logging.viewer"
  member  = "serviceAccount:${google_service_account.monitoring_alerts.email}"
}

resource "google_project_iam_member" "monitoring_logging_config" {
  project = var.project_id
  role    = "roles/logging.configWriter"
  member  = "serviceAccount:${google_service_account.monitoring_alerts.email}"
}

# Outputs
output "vertex_ai_pipeline_service_account_email" {
  description = "Email of the Vertex AI Pipeline service account"
  value       = google_service_account.vertex_ai_pipeline.email
}

output "vertex_ai_serving_service_account_email" {
  description = "Email of the Vertex AI Serving service account"
  value       = google_service_account.vertex_ai_serving.email
}

output "monitoring_alerts_service_account_email" {
  description = "Email of the Monitoring service account"
  value       = google_service_account.monitoring_alerts.email
}

# Security validation resource
resource "google_project_iam_audit_config" "vertex_ai_audit" {
  project = var.project_id
  service = "aiplatform.googleapis.com"
  
  audit_log_config {
    log_type = "ADMIN_READ"
  }
  
  audit_log_config {
    log_type = "DATA_READ"
  }
  
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}