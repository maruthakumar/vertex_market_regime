# External API Integration

## **Google Vertex AI API**
- **Purpose:** ML model training, serving, and hyperparameter optimization for adaptive learning system
- **Documentation:** https://cloud.google.com/vertex-ai/docs
- **Base URL:** https://us-central1-aiplatform.googleapis.com
- **Authentication:** Service Account with Vertex AI permissions
- **Integration Method:** REST API calls from Python backend with connection pooling

**Key Endpoints Used:**
- `POST /v1/projects/{project}/locations/{location}/endpoints/{endpoint}:predict` - Real-time ML predictions
- `POST /v1/projects/{project}/locations/{location}/trainingPipelines` - Automated model retraining

**Error Handling:** Graceful fallback to rule-based regime classification if Vertex AI unavailable, with automatic retry logic and circuit breaker pattern

## **Google BigQuery API**
- **Purpose:** ML data warehouse for historical pattern analysis and feature engineering
- **Documentation:** https://cloud.google.com/bigquery/docs/reference/rest
- **Base URL:** https://bigquery.googleapis.com
- **Authentication:** Service Account with BigQuery read/write permissions  
- **Integration Method:** Python BigQuery client with connection pooling and query optimization

**Key Endpoints Used:**
- `POST /bigquery/v2/projects/{project}/jobs` - Execute feature engineering queries
- `GET /bigquery/v2/projects/{project}/datasets/{dataset}/tables/{table}/data` - Historical data retrieval

**Error Handling:** Automatic query retry with exponential backoff, fallback to HeavyDB for critical data needs
