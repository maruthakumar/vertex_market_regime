# Tech Stack Alignment

## Existing Technology Stack

| Category | Current Technology | Version | Usage in Enhancement | Notes |
|----------|-------------------|---------|---------------------|-------|
| Database | GCS Parquet + Arrow | Latest | Primary storage + in-memory processing | HeavyDB deprecated (migration-only) |
| Language | Python | 3.8+ | All component implementations | Leverage existing codebase |
| Data Processing | RAPIDS cuDF + Apache Arrow | Latest | Feature engineering + analysis | GPU acceleration primary |
| Web Framework | FastAPI/Flask | Latest | API endpoints | Extend existing REST framework |
| Configuration | Excel + YAML | N/A | Parameter management | Excel sheets mapped to ML hyperparameters |
| Monitoring | Custom logging | N/A | Performance tracking | Enhanced with ML metrics |
| Deployment | SSH + tmux | N/A | Local orchestration | Augmented with BMAD automation |

## New Technology Additions

| Technology | Version | Purpose | Rationale | Integration Method |
|------------|---------|---------|-----------|-------------------|
| Google Vertex AI | Latest | ML model training/serving | Scalable ML infrastructure for adaptive learning | API integration with existing system |
| Google BigQuery | Latest | ML data warehouse | Structured data for model training | ETL pipeline from HeavyDB |
| Google Cloud Storage | Latest | Model artifacts + data pipeline | Durable storage for ML assets | Batch data transfer and streaming |
| scikit-learn | 1.3+ | Feature engineering + validation | Proven ML library for ensemble models | Python integration |
| TensorFlow/PyTorch | Latest | Deep learning components | Advanced pattern recognition in market regimes | Optional for neural network components |
