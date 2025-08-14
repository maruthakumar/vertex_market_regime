# Terraform Configuration for Monitoring and Alerting Infrastructure
# Story 2.5: IAM, Artifact Registry, Budgets/Monitoring
# Comprehensive monitoring setup for Vertex AI ML operations

# Notification Channels
resource "google_monitoring_notification_channel" "email_ml_team" {
  display_name = "ML Team Email"
  type         = "email"
  project      = var.project_id
  
  labels = {
    email_address = "ml-team@company.com"  # Replace with actual email
  }
  
  user_labels = {
    team        = "ml"
    environment = "production"
  }
}

resource "google_monitoring_notification_channel" "email_oncall" {
  display_name = "On-Call Email"
  type         = "email"
  project      = var.project_id
  
  labels = {
    email_address = "oncall@company.com"  # Replace with actual email
  }
  
  user_labels = {
    team        = "sre"
    priority    = "critical"
  }
}

# Alert Policies
resource "google_monitoring_alert_policy" "training_job_failure_rate" {
  display_name = "Training Job Failure Rate High"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Training job failure rate > 20%"
    
    condition_threshold {
      filter          = "resource.type=\"aiplatform_training_job\" AND resource.labels.project_id=\"${var.project_id}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.2
      
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.job_id"]
      }
    }
  }
  
  notification_channels = [
    google_monitoring_notification_channel.email_ml_team.id
  ]
  
  alert_strategy {
    auto_close = "1800s"  # 30 minutes
  }
  
  documentation {
    content   = "High failure rate in ML training jobs. Check logs and resource availability."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "vertex_ai_api_error_rate" {
  display_name = "Vertex AI API Error Rate High"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Vertex AI API error rate > 5%"
    
    condition_threshold {
      filter          = "resource.type=\"consumed_api\" AND resource.labels.service=\"aiplatform.googleapis.com\" AND metric.labels.response_code_class!=\"2xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
      
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }
  
  notification_channels = [
    google_monitoring_notification_channel.email_ml_team.id
  ]
  
  alert_strategy {
    auto_close = "3600s"  # 1 hour
  }
  
  documentation {
    content   = "High error rate in Vertex AI API calls. Check service status and authentication."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "artifact_registry_storage_quota" {
  display_name = "Artifact Registry Storage Quota High"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Artifact Registry storage > 80% of quota"
    
    condition_threshold {
      filter          = "resource.type=\"artifactregistry_repository\" AND metric.type=\"artifactregistry.googleapis.com/repository/storage_utilization_by_repository\""
      duration        = "3600s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8
      
      aggregations {
        alignment_period   = "3600s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = [
    google_monitoring_notification_channel.email_ml_team.id
  ]
  
  alert_strategy {
    auto_close = "86400s"  # 24 hours
  }
  
  documentation {
    content   = "Artifact Registry storage approaching quota. Clean up old images or increase quota."
    mime_type = "text/markdown"
  }
}

# Uptime Checks
resource "google_monitoring_uptime_check_config" "vertex_ai_api_check" {
  display_name = "Vertex AI API Health Check"
  project      = var.project_id
  timeout      = "10s"
  period       = "60s"
  
  http_check {
    path           = "/v1/projects/${var.project_id}/locations"
    port           = 443
    use_ssl        = true
    validate_ssl   = true
    request_method = "GET"
    
    headers = {
      "User-Agent" = "Google-Cloud-Monitoring/1.0"
    }
    
    accepted_response_status_codes {
      status_class = "STATUS_CLASS_2XX"
    }
  }
  
  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = "aiplatform.googleapis.com"
    }
  }
  
  content_matchers {
    content = "locations"
    matcher = "MATCHES_JSON_PATH"
    json_path_matcher {
      json_path = "$.locations[0].name"
    }
  }
}

# Custom Dashboards
resource "google_monitoring_dashboard" "ml_operations_overview" {
  dashboard_json = jsonencode({
    displayName = "ML Operations Overview"
    mosaicLayout = {
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Vertex AI Training Jobs"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"aiplatform_training_job\""
                    aggregation = {
                      alignmentPeriod     = "300s"
                      perSeriesAligner   = "ALIGN_COUNT"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["resource.label.job_state"]
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Job Count"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Vertex AI API Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"consumed_api\" AND resource.labels.service=\"aiplatform.googleapis.com\""
                    aggregation = {
                      alignmentPeriod     = "300s"
                      perSeriesAligner   = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Requests/second"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Artifact Registry Storage Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"artifactregistry_repository\" AND metric.type=\"artifactregistry.googleapis.com/repository/storage_utilization_by_repository\""
                    aggregation = {
                      alignmentPeriod   = "3600s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Storage Utilization"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "BigQuery Slot Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"bigquery_project\" AND metric.type=\"bigquery.googleapis.com/slots/allocated\""
                    aggregation = {
                      alignmentPeriod   = "300s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Allocated Slots"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
  
  project = var.project_id
}

resource "google_monitoring_dashboard" "security_monitoring" {
  dashboard_json = jsonencode({
    displayName = "Security Monitoring Dashboard"
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
                      alignmentPeriod   = "300s"
                      perSeriesAligner = "ALIGN_COUNT"
                    }
                  }
                }
                plotType = "LINE"
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
                      alignmentPeriod   = "3600s"
                      perSeriesAligner = "ALIGN_COUNT"
                    }
                  }
                }
                plotType = "LINE"
              }]
            }
          }
        },
        {
          width  = 12
          height = 4
          widget = {
            title = "Service Account Activity"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "protoPayload.authenticationInfo.principalEmail=~\".*@arched-bot-269016.iam.gserviceaccount.com\""
                    aggregation = {
                      alignmentPeriod     = "300s"
                      perSeriesAligner   = "ALIGN_COUNT"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["protoPayload.authenticationInfo.principalEmail"]
                    }
                  }
                }
                plotType = "STACKED_BAR"
              }]
            }
          }
        }
      ]
    }
  })
  
  project = var.project_id
}

# Log-based Metrics
resource "google_logging_metric" "training_job_errors" {
  name   = "training_job_errors"
  project = var.project_id
  
  filter = "resource.type=\"aiplatform_training_job\" AND severity=\"ERROR\""
  
  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "INT64"
    display_name = "Training Job Errors"
  }
  
  label_extractors = {
    "component"   = "EXTRACT(jsonPayload.component)"
    "error_type"  = "EXTRACT(jsonPayload.error_type)"
  }
}

resource "google_logging_metric" "security_violations" {
  name   = "security_violations"
  project = var.project_id
  
  filter = "protoPayload.serviceName=\"iam.googleapis.com\" AND protoPayload.response.error.code!=0"
  
  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "INT64"
    display_name = "Security Policy Violations"
  }
  
  label_extractors = {
    "violation_type" = "EXTRACT(protoPayload.methodName)"
  }
}

# SLO Configuration (using Google Cloud Monitoring API)
resource "google_monitoring_slo" "training_job_success_rate" {
  service      = google_monitoring_service.ml_training_service.service_id
  slo_id       = "training-job-success-rate"
  display_name = "Training Job Success Rate SLO"
  
  goal                = 0.95
  rolling_period_days = 30
  
  request_based_sli {
    good_total_ratio {
      total_service_filter = "resource.type=\"aiplatform_training_job\""
      good_service_filter  = "resource.type=\"aiplatform_training_job\" AND resource.labels.job_state=\"JOB_STATE_SUCCEEDED\""
    }
  }
}

resource "google_monitoring_service" "ml_training_service" {
  service_id   = "ml-training-service"
  display_name = "ML Training Service"
  project      = var.project_id
}

# Outputs
output "ml_operations_dashboard_url" {
  description = "URL of the ML Operations Overview dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.ml_operations_overview.id}?project=${var.project_id}"
}

output "security_monitoring_dashboard_url" {
  description = "URL of the Security Monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.security_monitoring.id}?project=${var.project_id}"
}

output "monitoring_notification_channels" {
  description = "Monitoring notification channel IDs"
  value = {
    ml_team = google_monitoring_notification_channel.email_ml_team.id
    oncall  = google_monitoring_notification_channel.email_oncall.id
  }
}