# Terraform Configuration for Audit Logging and Security
# Story 2.5: IAM, Artifact Registry, Budgets/Monitoring
# Comprehensive audit logging and security monitoring setup

# Enable required APIs
resource "google_project_service" "logging_api" {
  service = "logging.googleapis.com"
  project = var.project_id
  
  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "monitoring_api" {
  service = "monitoring.googleapis.com"
  project = var.project_id
  
  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "security_center_api" {
  service = "securitycenter.googleapis.com"
  project = var.project_id
  
  disable_dependent_services = false
  disable_on_destroy         = false
}

# Audit Logging Configuration
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

resource "google_project_iam_audit_config" "artifact_registry_audit" {
  project = var.project_id
  service = "artifactregistry.googleapis.com"
  
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

resource "google_project_iam_audit_config" "storage_audit" {
  project = var.project_id
  service = "storage.googleapis.com"
  
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

resource "google_project_iam_audit_config" "bigquery_audit" {
  project = var.project_id
  service = "bigquery.googleapis.com"
  
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

resource "google_project_iam_audit_config" "iam_audit" {
  project = var.project_id
  service = "iam.googleapis.com"
  
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

# BigQuery dataset for security audit logs
resource "google_bigquery_dataset" "security_audit_logs" {
  dataset_id  = "security_audit_logs"
  project     = var.project_id
  location    = var.region
  
  description = "Security audit logs for ML infrastructure"
  
  default_table_expiration_ms = 220752000000  # 7 years in milliseconds
  
  labels = {
    environment = "production"
    purpose     = "security-audit"
    team        = "security"
  }
  
  access {
    role          = "OWNER"
    user_by_email = "security-team@company.com"  # Replace with actual email
  }
  
  access {
    role          = "READER"
    user_by_email = "compliance-team@company.com"  # Replace with actual email
  }
  
  access {
    role           = "WRITER"
    special_group  = "projectWriters"
  }
}

# Log sink for security audit logs
resource "google_logging_project_sink" "security_audit_sink" {
  name        = "security-audit-logs"
  project     = var.project_id
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.security_audit_logs.dataset_id}"
  
  filter = <<-EOT
    (protoPayload.serviceName="aiplatform.googleapis.com" OR
     protoPayload.serviceName="artifactregistry.googleapis.com" OR
     protoPayload.serviceName="iam.googleapis.com" OR
     protoPayload.serviceName="storage.googleapis.com" OR
     protoPayload.serviceName="bigquery.googleapis.com") AND
    (protoPayload.methodName!="google.logging.v2.ConfigServiceV2.ListSinks")
  EOT
  
  unique_writer_identity = true
}

# Grant BigQuery Data Editor role to the log sink's writer identity
resource "google_project_iam_member" "log_sink_bigquery_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = google_logging_project_sink.security_audit_sink.writer_identity
}

# Pub/Sub topics for security alerts
resource "google_pubsub_topic" "security_alerts" {
  name    = "security-alerts"
  project = var.project_id
  
  labels = {
    purpose = "security-alerting"
    team    = "security"
  }
}

resource "google_pubsub_subscription" "security_alerts_email" {
  name    = "security-alerts-email"
  topic   = google_pubsub_topic.security_alerts.name
  project = var.project_id
  
  ack_deadline_seconds = 60
  
  push_config {
    push_endpoint = "https://your-alert-handler.com/security-alerts"  # Replace with actual endpoint
  }
}

# Security monitoring alert policies
resource "google_monitoring_alert_policy" "privileged_role_assignment" {
  display_name = "Privileged Role Assignment Alert"
  project      = var.project_id
  combiner     = "OR"
  
  conditions {
    display_name = "Privileged role assigned"
    
    condition_threshold {
      filter          = "resource.type=\"gce_project\" AND protoPayload.serviceName=\"cloudresourcemanager.googleapis.com\" AND protoPayload.methodName=\"SetIamPolicy\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email_security.id]
  
  alert_strategy {
    auto_close = "86400s"  # 24 hours
  }
  
  documentation {
    content   = "Alert when privileged roles (Owner, Editor, Security Admin) are assigned in the project."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "service_account_key_creation" {
  display_name = "Service Account Key Creation Alert"
  project      = var.project_id
  combiner     = "OR"
  
  conditions {
    display_name = "Service account key created"
    
    condition_threshold {
      filter          = "protoPayload.serviceName=\"iam.googleapis.com\" AND protoPayload.methodName=\"google.iam.admin.v1.IAM.CreateServiceAccountKey\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email_security.id]
  
  alert_strategy {
    auto_close = "86400s"
  }
  
  documentation {
    content   = "Alert when service account keys are created. Service account keys should be avoided in favor of workload identity."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "vertex_ai_failures" {
  display_name = "Vertex AI Operation Failures"
  project      = var.project_id
  combiner     = "OR"
  
  conditions {
    display_name = "Vertex AI operation failures"
    
    condition_threshold {
      filter          = "protoPayload.serviceName=\"aiplatform.googleapis.com\" AND protoPayload.response.error.code!=0"
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_COUNT"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email_ml_team.id]
  
  alert_strategy {
    auto_close = "3600s"  # 1 hour
  }
  
  documentation {
    content   = "Alert when there are multiple Vertex AI operation failures, which may indicate unauthorized access attempts or service issues."
    mime_type = "text/markdown"
  }
}

# Notification channels
resource "google_monitoring_notification_channel" "email_security" {
  display_name = "Security Team Email"
  type         = "email"
  project      = var.project_id
  
  labels = {
    email_address = "security-team@company.com"  # Replace with actual email
  }
  
  user_labels = {
    team = "security"
  }
}

resource "google_monitoring_notification_channel" "email_ml_team" {
  display_name = "ML Team Email"
  type         = "email"
  project      = var.project_id
  
  labels = {
    email_address = "ml-team@company.com"  # Replace with actual email
  }
  
  user_labels = {
    team = "ml"
  }
}

# Security dashboard
resource "google_monitoring_dashboard" "security_overview" {
  dashboard_json = jsonencode({
    displayName = "Security Overview Dashboard"
    mosaicLayout = {
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Failed Authentication Attempts"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "protoPayload.authenticationInfo.principalEmail!=\"\" AND protoPayload.response.error.code!=0"
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_COUNT"
                    }
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "IAM Policy Changes"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "protoPayload.serviceName=\"iam.googleapis.com\" AND protoPayload.methodName=\"SetIamPolicy\""
                    aggregation = {
                      alignmentPeriod  = "3600s"
                      perSeriesAligner = "ALIGN_COUNT"
                    }
                  }
                }
              }]
            }
          }
        }
      ]
    }
  })
  
  project = var.project_id
}

# Outputs
output "security_audit_dataset_id" {
  description = "BigQuery dataset ID for security audit logs"
  value       = google_bigquery_dataset.security_audit_logs.dataset_id
}

output "security_alerts_topic" {
  description = "Pub/Sub topic for security alerts"
  value       = google_pubsub_topic.security_alerts.name
}

output "security_dashboard_url" {
  description = "URL of the security monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.security_overview.id}?project=${var.project_id}"
}