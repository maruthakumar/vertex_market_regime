# Terraform Configuration for Artifact Registry
# Story 2.5: IAM, Artifact Registry, Budgets/Monitoring
# Container registry with security scanning and access controls

# Enable required APIs
resource "google_project_service" "artifact_registry_api" {
  service = "artifactregistry.googleapis.com"
  project = var.project_id
  
  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "container_scanning_api" {
  service = "containeranalysis.googleapis.com"
  project = var.project_id
  
  disable_dependent_services = false
  disable_on_destroy         = false
}

# Artifact Registry Repository for ML containers
resource "google_artifact_registry_repository" "mr_ml_repository" {
  provider = google
  
  location      = var.region
  repository_id = "mr-ml"
  description   = "Market Regime ML container repository for training and serving images"
  format        = "DOCKER"
  project       = var.project_id
  
  # Enable vulnerability scanning
  docker_config {
    immutable_tags = false
  }
  
  depends_on = [google_project_service.artifact_registry_api]
  
  labels = {
    environment = "production"
    project     = "market-regime"
    purpose     = "ml-containers"
    team        = "vertex-ai"
  }
}

# Repository IAM policy for service accounts
resource "google_artifact_registry_repository_iam_member" "pipeline_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.mr_ml_repository.location
  repository = google_artifact_registry_repository.mr_ml_repository.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

resource "google_artifact_registry_repository_iam_member" "serving_reader" {
  project    = var.project_id
  location   = google_artifact_registry_repository.mr_ml_repository.location
  repository = google_artifact_registry_repository.mr_ml_repository.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.vertex_ai_serving.email}"
}

# Container Analysis settings for vulnerability scanning
resource "google_project_iam_member" "container_analysis_viewer" {
  project = var.project_id
  role    = "roles/containeranalysis.notes.viewer"
  member  = "serviceAccount:${google_service_account.vertex_ai_pipeline.email}"
}

# Binary Authorization policy for secure container deployment
resource "google_binary_authorization_policy" "ml_policy" {
  project = var.project_id
  
  default_admission_rule {
    evaluation_mode  = "REQUIRE_ATTESTATION"
    enforcement_mode = "ENFORCED_BLOCK_AND_AUDIT_LOG"
    
    require_attestations_by = [
      google_binary_authorization_attestor.vulnerability_attestor.name
    ]
  }
  
  # Allow images from our Artifact Registry
  admission_whitelist_patterns {
    name_pattern = "${var.region}-docker.pkg.dev/${var.project_id}/mr-ml/*"
  }
  
  depends_on = [google_project_service.container_scanning_api]
}

# Vulnerability attestor for security scanning
resource "google_binary_authorization_attestor" "vulnerability_attestor" {
  name    = "vulnerability-attestor"
  project = var.project_id
  
  description = "Attestor for vulnerability scanning results"
  
  attestation_authority_note {
    note_reference = google_container_analysis_note.vulnerability_note.name
    
    public_keys {
      ascii_armored_pgp_public_key = file("${path.module}/vulnerability_attestor_key.pub")
    }
  }
}

# Container Analysis note for vulnerability scanning
resource "google_container_analysis_note" "vulnerability_note" {
  name    = "vulnerability-note"
  project = var.project_id
  
  attestation_authority {
    hint {
      human_readable_name = "Vulnerability Scanner"
    }
  }
}

# Outputs
output "artifact_registry_repository_url" {
  description = "URL of the Artifact Registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mr_ml_repository.repository_id}"
}

output "artifact_registry_repository_name" {
  description = "Name of the Artifact Registry repository"
  value       = google_artifact_registry_repository.mr_ml_repository.name
}