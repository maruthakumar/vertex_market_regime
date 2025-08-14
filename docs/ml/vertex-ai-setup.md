# Vertex AI Setup Guide (Market Regime)

Version: 1.0  
Date: 2025-08-12

## Services
- Vertex AI, BigQuery, Artifact Registry, Cloud Storage, Cloud Monitoring/Billing

## Roles (Service Account Matrix)
- vertex-ai-sa: Vertex AI Admin, BQ JobUser/DataEditor, Storage Object Viewer, Artifact Registry Writer
- data-pipeline-sa: BQ JobUser/DataEditor, Storage Object Viewer

## Artifact Registry
- Region: us-central1
- Repo: `mr-ml`

## Environment
- Project: `arched-bot-269016`
- Region: `us-central1`

## Commands (indicative)
- Enable APIs: `gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com bigquery.googleapis.com`
- Create AR Repo: `gcloud artifacts repositories create mr-ml --repository-format=docker --location=us-central1`
- Build/Push Image (Cloud Build or local): `gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT/mr-ml/fe-job:v0`
- Submit CustomJob: `gcloud ai custom-jobs create --region=us-central1 --display-name=fe-smoke --config=customjob.yaml`

## Notes
- Keep endpoints for Epic 3; Epic 2 focuses on data/feature infra and model registration only.



