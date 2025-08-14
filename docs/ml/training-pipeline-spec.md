# Training Pipeline Specification
## Market Regime Prediction Model Training Pipeline

Version: 2.0  
Date: 2025-08-13

### Overview
This document provides the complete specification for the Market Regime Training Pipeline built on Vertex AI Pipelines (KFP v2). The pipeline handles end-to-end training of market regime prediction models from BigQuery offline feature tables to registered models in Vertex AI Model Registry.

### Pipeline Architecture

#### High-Level Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline Architecture                │
│                         (KFP v2 - Vertex AI)                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
         ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
         │  Data Pipeline  │ │   Training  │ │   Evaluation    │
         │                 │ │   Pipeline  │ │   & Registry    │
         │ • BigQuery Load │ │ • TabNet    │ │ • Metrics       │
         │ • Feature Prep  │ │ • XGBoost   │ │ • Validation    │
         │ • Time Splits   │ │ • LSTM      │ │ • Registration  │
         │ • Quality Check │ │ • TFT       │ │ • Experiments   │
         └─────────────────┘ └─────────────┘ └─────────────────┘
```

#### Component Architecture
```
Pipeline Components:
├── 1. Data Preparation Component
│   ├── Input: BigQuery offline feature tables
│   ├── Processing: Load → Validate → Split → Preprocess
│   └── Output: Train/Val/Test datasets (Parquet)
│
├── 2. Model Training Components (Parallel)
│   ├── TabNet Training Component
│   ├── XGBoost Training Component  
│   ├── LSTM Training Component
│   └── TFT Training Component (Optional)
│
├── 3. Model Evaluation Components (Parallel)
│   ├── Input: Trained models + Test data
│   ├── Processing: Evaluate → Metrics → Validation
│   └── Output: Evaluation metrics + Reports
│
├── 4. Model Registration Components (Parallel)
│   ├── Input: Trained models + Evaluation results
│   ├── Processing: Register → Metadata → Versioning
│   └── Output: Model Registry IDs + URIs
│
└── 5. Experiment Tracking Component
    ├── Input: Training + Evaluation metrics
    ├── Processing: Log → Track → Compare
    └── Output: Experiment run metadata
```

### Pipeline IO Contracts

#### Pipeline Inputs
```yaml
PipelineInputs:
  project_id: str                    # GCP project ID
  location: str                      # GCP region (default: us-central1)
  staging_bucket: str                # GCS staging bucket
  training_dataset_table: str        # BigQuery source table
  experiment_name: str               # Vertex AI experiment name
  model_display_name: str            # Model registry display name
  model_types: List[str]             # Models to train (tabnet, xgboost, lstm, tft)
  validation_split: float            # Validation data ratio (default: 0.2)
  test_split: float                  # Test data ratio (default: 0.1)
  hyperparameters: Dict[str, Any]    # Model-specific hyperparameters
```

#### Pipeline Outputs
```yaml
PipelineOutputs:
  trained_models: Dict[str, str]     # Model registry IDs by model type
  evaluation_metrics: Dict[str, float] # Performance metrics by model type
  model_artifacts: Dict[str, str]    # GCS paths to model artifacts
  experiment_run_id: str             # Vertex AI experiment run ID
  data_processing_stats: Dict        # Data preparation statistics
  pipeline_metadata: Dict            # Complete pipeline execution metadata
```

### Component Specifications

#### 1. Data Preparation Component
**Purpose**: Load and prepare training data from BigQuery offline feature tables

**Inputs**:
- `project_id`: GCP project ID
- `dataset_table`: BigQuery table path (project.dataset.table)
- `staging_bucket`: GCS bucket for staging data
- `validation_split`: Validation data ratio
- `test_split`: Test data ratio
- `output_format`: parquet or tfrecords

**Processing**:
1. Load data from BigQuery offline tables (`c1_features` through `c8_features` + `training_dataset`)
2. Data quality validation (missing values, feature counts, distribution checks)
3. Time-based train/validation/test splitting (maintaining temporal order)
4. Feature preprocessing (normalization, missing value handling)
5. Export to Parquet/TFRecords format in GCS

**Outputs**:
- `train_data`: Training dataset (70% of data)
- `validation_data`: Validation dataset (20% of data)
- `test_data`: Test dataset (10% of data)
- `data_schema`: Dataset schema and metadata
- `data_stats`: Data quality and preprocessing statistics

#### 2. Model Training Components
**Purpose**: Train baseline models for market regime prediction

**Models Supported**:

**TabNet Model**:
- **Architecture**: Attention-based tabular model
- **Hyperparameters**: n_d=64, n_a=64, n_steps=5, gamma=1.3
- **Target Use**: Primary regime classification model
- **Expected Accuracy**: >85% on validation set

**XGBoost Model**:
- **Architecture**: Gradient boosting decision trees
- **Hyperparameters**: max_depth=6, learning_rate=0.1, n_estimators=1000
- **Target Use**: Baseline comparison model
- **Expected Accuracy**: >80% on validation set

**LSTM Model**:
- **Architecture**: Recurrent neural network for sequence modeling
- **Hyperparameters**: hidden_size=128, num_layers=2, sequence_length=60
- **Target Use**: Transition forecasting model
- **Expected Performance**: Directional accuracy >75%

**Temporal Fusion Transformer (Optional)**:
- **Architecture**: Advanced attention-based sequence model
- **Hyperparameters**: hidden_size=128, attention_heads=4
- **Target Use**: Advanced transition forecasting
- **Expected Performance**: Directional accuracy >80%

#### 3. Model Evaluation Components
**Purpose**: Comprehensive evaluation of trained models

**Evaluation Metrics**:

**Classification Metrics** (Regime Prediction):
- Accuracy: Overall prediction accuracy
- Precision: Per-class precision scores
- Recall: Per-class recall scores
- F1-Score: Harmonic mean of precision and recall
- AUROC: Area under ROC curve
- AUPRC: Area under Precision-Recall curve
- Confusion Matrix: Detailed classification results

**Forecasting Metrics** (Transition Prediction):
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- Directional Accuracy: Correct direction prediction rate

**Market-Specific Metrics**:
- Regime Accuracy: Accuracy within each market regime
- Transition Detection F1: F1-score for regime transitions
- Regime Stability Score: Model consistency within regimes

#### 4. Model Registration Components
**Purpose**: Register validated models in Vertex AI Model Registry

**Registration Process**:
1. Validate model meets performance thresholds
2. Prepare model metadata and documentation
3. Register model in Vertex AI Model Registry
4. Create model version with complete lineage
5. Tag model with evaluation metrics and metadata
6. Set up model monitoring and governance

#### 5. Experiment Tracking Component
**Purpose**: Track and compare model training experiments

**Tracking Features**:
- Experiment organization and versioning
- Hyperparameter logging and comparison
- Metric tracking across training runs
- Artifact and model versioning
- Performance comparison dashboards
- Experiment reproducibility

### Pipeline Orchestration

#### Execution Flow
```
Start Pipeline
     │
     ▼
┌─────────────────┐
│ Data Preparation│ ← Load config and validate inputs
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Parallel Training│ ← Train TabNet, XGBoost, LSTM (parallel)
│   Components    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Parallel Eval   │ ← Evaluate all models (parallel)
│   Components    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Parallel Regist │ ← Register validated models (parallel)
│   Components    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Experiment      │ ← Track results and generate reports
│ Tracking        │
└─────────────────┘
     │
     ▼
Pipeline Complete
```

### Resource Requirements
**Compute Resources**:
- Data Preparation: n1-standard-4 (4 vCPU, 15 GB RAM)
- TabNet Training: n1-highmem-8 + Tesla T4 GPU
- XGBoost Training: n1-highmem-4 (4 vCPU, 26 GB RAM)
- LSTM Training: n1-highmem-8 + Tesla T4 GPU
- Model Evaluation: n1-standard-4 (4 vCPU, 15 GB RAM)

**Storage Requirements**:
- Training Data: ~5 GB (7 years of data)
- Model Artifacts: ~2 GB per model
- Staging Storage: ~20 GB total
- Artifact Retention: 90 days

### Performance Requirements

#### Training Performance Targets
- **Pipeline Execution Time**: < 4 hours end-to-end
- **Data Processing**: < 30 minutes for full dataset
- **Model Training**: < 2 hours per model (parallel execution)
- **Model Evaluation**: < 15 minutes per model
- **Model Registration**: < 5 minutes per model

#### Model Performance Targets
- **TabNet Accuracy**: > 85% on validation set
- **XGBoost Accuracy**: > 80% on validation set
- **LSTM Directional Accuracy**: > 75% for transitions
- **Overall System Accuracy**: > 87% (weighted ensemble)

### Security & Cost
- Service accounts with least privilege
- Region-scoped resources; budget alerts enabled
- TLS 1.3 encryption in transit, AES-256 at rest
- VPC with private Google access

### Parity Requirements
- Use the same feature transforms as Epic 1 implementation
- Validate against parity harness samples
- 774 total features from 8 components
- Maintain compatibility with existing backtester_v2 system



