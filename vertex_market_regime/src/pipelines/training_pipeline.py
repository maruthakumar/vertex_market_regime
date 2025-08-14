"""
Training Pipeline for Market Regime Model
Kubeflow Pipelines v2 implementation for Vertex AI Pipelines

This module defines the complete training pipeline including:
- Data loading from BigQuery offline feature tables
- Feature preprocessing and validation
- Model training (TabNet, XGBoost, LSTM, TFT)
- Model evaluation and validation
- Model registration in Vertex AI Model Registry
- Experiment tracking with Vertex AI Experiments
"""

from typing import Dict, Any, Optional, NamedTuple, List
import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics, Artifact
from google.cloud import aiplatform


# Pipeline IO Contracts
class PipelineInputs(NamedTuple):
    """Input specifications for the training pipeline"""
    project_id: str
    location: str = "us-central1"
    staging_bucket: str
    training_dataset_table: str  # BigQuery table path
    experiment_name: str
    model_display_name: str
    pipeline_config: Dict[str, Any]


class PipelineOutputs(NamedTuple):
    """Output specifications for the training pipeline"""
    trained_models: Dict[str, str]  # Model registry IDs
    evaluation_metrics: Dict[str, float]
    model_artifacts: Dict[str, str]  # GCS paths
    experiment_run_id: str


class ComponentInputs:
    """Input contracts for individual pipeline components"""
    
    class DataPreparation(NamedTuple):
        project_id: str
        dataset_table: str
        validation_split: float = 0.2
        test_split: float = 0.1
        output_format: str = "parquet"
        
    class ModelTraining(NamedTuple):
        training_data_path: str
        model_type: str  # "tabnet", "xgboost", "lstm", "tft"
        hyperparameters: Dict[str, Any]
        experiment_name: str
        
    class ModelEvaluation(NamedTuple):
        model_path: str
        test_data_path: str
        evaluation_metrics: List[str]
        
    class ModelRegistration(NamedTuple):
        model_path: str
        model_display_name: str
        model_metadata: Dict[str, Any]


class ComponentOutputs:
    """Output contracts for individual pipeline components"""
    
    class DataPreparation(NamedTuple):
        train_data_path: str
        validation_data_path: str
        test_data_path: str
        data_schema: Dict[str, str]
        data_stats: Dict[str, Any]
        
    class ModelTraining(NamedTuple):
        model_path: str
        training_metrics: Dict[str, float]
        training_logs: str
        
    class ModelEvaluation(NamedTuple):
        evaluation_metrics: Dict[str, float]
        evaluation_report: str
        model_performance: Dict[str, Any]
        
    class ModelRegistration(NamedTuple):
        model_id: str
        model_version: str
        model_uri: str


# Component Specifications
@component(
    base_image="python:3.8-slim",
    packages_to_install=[
        "google-cloud-bigquery==3.4.0",
        "google-cloud-storage==2.7.0",
        "pandas==1.5.3",
        "pyarrow==11.0.0",
        "scikit-learn==1.3.0",
    ]
)
def data_preparation_component(
    project_id: str,
    dataset_table: str,
    staging_bucket: str,
    validation_split: float,
    test_split: float,
    output_format: str,
    train_data: Output[Dataset],
    validation_data: Output[Dataset],
    test_data: Output[Dataset],
    data_schema: Output[Artifact],
    data_stats: Output[Artifact]
) -> NamedTuple('Outputs', [('train_samples', int), ('val_samples', int), ('test_samples', int)]):
    """
    Data preparation component for training pipeline
    
    Loads data from BigQuery offline feature tables, performs train/validation/test split,
    and outputs datasets in specified format (Parquet/TFRecords)
    
    Args:
        project_id: GCP project ID
        dataset_table: BigQuery table path (project.dataset.table)
        staging_bucket: GCS bucket for staging data
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        output_format: Output format (parquet/tfrecords)
        
    Outputs:
        train_data: Training dataset
        validation_data: Validation dataset
        test_data: Test dataset
        data_schema: Dataset schema information
        data_stats: Dataset statistics
    """
    import pandas as pd
    from google.cloud import bigquery
    from google.cloud import storage
    import json
    from pathlib import Path
    
    # Implementation will be added in subsequent tasks
    # This is the IO contract definition
    
    # Placeholder return values
    Outputs = NamedTuple('Outputs', [('train_samples', int), ('val_samples', int), ('test_samples', int)])
    return Outputs(train_samples=10000, val_samples=2000, test_samples=1000)


@component(
    base_image="python:3.8-slim",
    packages_to_install=[
        "torch==2.0.0",
        "pytorch-tabnet==4.0",
        "xgboost==1.7.3",
        "scikit-learn==1.3.0",
        "google-cloud-aiplatform==1.25.0",
    ]
)
def model_training_component(
    training_data: Input[Dataset],
    validation_data: Input[Dataset],
    model_type: str,
    hyperparameters: dict,
    experiment_name: str,
    trained_model: Output[Model],
    training_metrics: Output[Metrics],
    training_logs: Output[Artifact]
) -> NamedTuple('Outputs', [('training_time_minutes', float), ('final_loss', float)]):
    """
    Model training component supporting multiple architectures
    
    Trains baseline models (TabNet, XGBoost, LSTM, TFT) based on model_type parameter
    
    Args:
        training_data: Training dataset from data preparation
        validation_data: Validation dataset
        model_type: Type of model to train (tabnet/xgboost/lstm/tft)
        hyperparameters: Model-specific hyperparameters
        experiment_name: Vertex AI experiment name for tracking
        
    Outputs:
        trained_model: Trained model artifact
        training_metrics: Training performance metrics
        training_logs: Detailed training logs
    """
    # Implementation will be added in subsequent tasks
    # This is the IO contract definition
    
    Outputs = NamedTuple('Outputs', [('training_time_minutes', float), ('final_loss', float)])
    return Outputs(training_time_minutes=45.0, final_loss=0.25)


@component(
    base_image="python:3.8-slim",
    packages_to_install=[
        "scikit-learn==1.3.0",
        "matplotlib==3.6.3",
        "seaborn==0.12.2",
        "google-cloud-aiplatform==1.25.0",
    ]
)
def model_evaluation_component(
    trained_model: Input[Model],
    test_data: Input[Dataset],
    evaluation_metrics_config: list,
    evaluation_metrics: Output[Metrics],
    evaluation_report: Output[Artifact],
    confusion_matrix: Output[Artifact]
) -> NamedTuple('Outputs', [('accuracy', float), ('f1_score', float), ('auroc', float)]):
    """
    Model evaluation component with comprehensive metrics
    
    Evaluates trained models on test dataset with regime classification and
    transition forecasting specific metrics
    
    Args:
        trained_model: Trained model from training component
        test_data: Test dataset
        evaluation_metrics_config: List of metrics to compute
        
    Outputs:
        evaluation_metrics: Computed evaluation metrics
        evaluation_report: Detailed evaluation report
        confusion_matrix: Confusion matrix visualization
    """
    # Implementation will be added in subsequent tasks
    # This is the IO contract definition
    
    Outputs = NamedTuple('Outputs', [('accuracy', float), ('f1_score', float), ('auroc', float)])
    return Outputs(accuracy=0.87, f1_score=0.85, auroc=0.92)


@component(
    base_image="python:3.8-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.25.0",
    ]
)
def model_registration_component(
    trained_model: Input[Model],
    evaluation_metrics: Input[Metrics],
    model_display_name: str,
    model_description: str,
    registered_model: Output[Artifact]
) -> NamedTuple('Outputs', [('model_id', str), ('model_version', str)]):
    """
    Model registration component for Vertex AI Model Registry
    
    Registers trained and validated models in Vertex AI Model Registry
    with comprehensive metadata
    
    Args:
        trained_model: Trained model artifact
        evaluation_metrics: Model evaluation results
        model_display_name: Display name for registered model
        model_description: Model description and metadata
        
    Outputs:
        registered_model: Model registry information
    """
    # Implementation will be added in subsequent tasks
    # This is the IO contract definition
    
    Outputs = NamedTuple('Outputs', [('model_id', str), ('model_version', str)])
    return Outputs(model_id="model_123", model_version="v1.0")


@component(
    base_image="python:3.8-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.25.0",
    ]
)
def experiment_tracking_component(
    training_metrics: Input[Metrics],
    evaluation_metrics: Input[Metrics],
    experiment_name: str,
    run_name: str,
    experiment_tracking: Output[Artifact]
) -> NamedTuple('Outputs', [('experiment_run_id', str)]):
    """
    Experiment tracking component for Vertex AI Experiments
    
    Logs training and evaluation metrics to Vertex AI Experiments
    for comprehensive experiment tracking
    
    Args:
        training_metrics: Training performance metrics
        evaluation_metrics: Model evaluation metrics
        experiment_name: Experiment name in Vertex AI
        run_name: Specific run name
        
    Outputs:
        experiment_tracking: Experiment tracking information
    """
    # Implementation will be added in subsequent tasks
    # This is the IO contract definition
    
    Outputs = NamedTuple('Outputs', [('experiment_run_id', str)])
    return Outputs(experiment_run_id="run_456")


# Pipeline Definition
@pipeline(
    name="market-regime-training-pipeline",
    description="Training pipeline for market regime prediction models",
    pipeline_root="gs://vertex-mr-pipelines/"
)
def market_regime_training_pipeline(
    project_id: str = "arched-bot-269016",
    location: str = "us-central1",
    staging_bucket: str = "vertex-mr-data",
    training_dataset_table: str = "market_regime_dev.training_dataset",
    experiment_name: str = "market-regime-experiment",
    model_display_name: str = "market-regime-model",
    model_types: list = ["tabnet", "xgboost", "lstm"],
    validation_split: float = 0.2,
    test_split: float = 0.1,
    hyperparameters: dict = None
) -> Dict[str, Any]:
    """
    Complete training pipeline for market regime prediction
    
    This pipeline orchestrates the complete training workflow:
    1. Data preparation from BigQuery offline tables
    2. Training multiple baseline models
    3. Comprehensive model evaluation
    4. Model registration in Vertex AI
    5. Experiment tracking
    
    Args:
        project_id: GCP project ID
        location: GCP region
        staging_bucket: GCS bucket for staging
        training_dataset_table: BigQuery source table
        experiment_name: Vertex AI experiment name
        model_display_name: Display name for models
        model_types: List of model types to train
        validation_split: Validation data split ratio
        test_split: Test data split ratio
        hyperparameters: Model hyperparameters
        
    Returns:
        Dictionary with pipeline outputs and metadata
    """
    
    # Default hyperparameters if not provided
    if hyperparameters is None:
        hyperparameters = {
            "tabnet": {
                "n_d": 64,
                "n_a": 64,
                "n_steps": 5,
                "gamma": 1.3,
                "lambda_sparse": 1e-3,
                "optimizer_fn": "Adam",
                "optimizer_params": {"lr": 2e-2},
                "scheduler_params": {"step_size": 50, "gamma": 0.9},
                "mask_type": "entmax"
            },
            "xgboost": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 1000,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss"
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "sequence_length": 60,
                "epochs": 100
            }
        }
    
    # Data Preparation Step
    data_prep_task = data_preparation_component(
        project_id=project_id,
        dataset_table=training_dataset_table,
        staging_bucket=staging_bucket,
        validation_split=validation_split,
        test_split=test_split,
        output_format="parquet"
    )
    
    # Model Training Steps (parallel execution)
    training_tasks = {}
    evaluation_tasks = {}
    registration_tasks = {}
    
    for model_type in model_types:
        # Train model
        training_task = model_training_component(
            training_data=data_prep_task.outputs["train_data"],
            validation_data=data_prep_task.outputs["validation_data"],
            model_type=model_type,
            hyperparameters=hyperparameters.get(model_type, {}),
            experiment_name=f"{experiment_name}-{model_type}"
        )
        training_tasks[model_type] = training_task
        
        # Evaluate model
        evaluation_task = model_evaluation_component(
            trained_model=training_task.outputs["trained_model"],
            test_data=data_prep_task.outputs["test_data"],
            evaluation_metrics_config=["accuracy", "f1_score", "auroc", "precision", "recall"]
        )
        evaluation_tasks[model_type] = evaluation_task
        
        # Register model
        registration_task = model_registration_component(
            trained_model=training_task.outputs["trained_model"],
            evaluation_metrics=evaluation_task.outputs["evaluation_metrics"],
            model_display_name=f"{model_display_name}-{model_type}",
            model_description=f"Market regime {model_type} model trained on {training_dataset_table}"
        )
        registration_tasks[model_type] = registration_task
        
        # Track experiment
        experiment_task = experiment_tracking_component(
            training_metrics=training_task.outputs["training_metrics"],
            evaluation_metrics=evaluation_task.outputs["evaluation_metrics"],
            experiment_name=experiment_name,
            run_name=f"{experiment_name}-{model_type}-run"
        )
    
    return {
        "pipeline_name": "market-regime-training-pipeline",
        "trained_models": len(model_types),
        "data_preparation_outputs": data_prep_task.outputs,
        "training_outputs": {k: v.outputs for k, v in training_tasks.items()},
        "evaluation_outputs": {k: v.outputs for k, v in evaluation_tasks.items()},
        "registration_outputs": {k: v.outputs for k, v in registration_tasks.items()}
    }


# Pipeline Configuration Schema
PIPELINE_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "project_id": {"type": "string", "description": "GCP project ID"},
        "location": {"type": "string", "default": "us-central1"},
        "staging_bucket": {"type": "string", "description": "GCS bucket for staging"},
        "training_dataset_table": {"type": "string", "description": "BigQuery source table"},
        "experiment_name": {"type": "string", "description": "Vertex AI experiment name"},
        "model_display_name": {"type": "string", "description": "Display name for models"},
        "model_types": {
            "type": "array",
            "items": {"type": "string", "enum": ["tabnet", "xgboost", "lstm", "tft"]},
            "default": ["tabnet", "xgboost", "lstm"]
        },
        "validation_split": {"type": "number", "minimum": 0.1, "maximum": 0.3, "default": 0.2},
        "test_split": {"type": "number", "minimum": 0.1, "maximum": 0.3, "default": 0.1},
        "hyperparameters": {
            "type": "object",
            "properties": {
                "tabnet": {"type": "object"},
                "xgboost": {"type": "object"},
                "lstm": {"type": "object"},
                "tft": {"type": "object"}
            }
        }
    },
    "required": ["project_id", "staging_bucket", "training_dataset_table", "experiment_name"]
}


# Pipeline Utilities
def compile_pipeline(output_path: str = "market_regime_training_pipeline.yaml") -> str:
    """
    Compile the training pipeline to YAML format
    
    Args:
        output_path: Path to save compiled pipeline
        
    Returns:
        Path to compiled pipeline file
    """
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=market_regime_training_pipeline,
        package_path=output_path
    )
    
    return output_path


def validate_pipeline_config(config: Dict[str, Any]) -> bool:
    """
    Validate pipeline configuration against schema
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        True if valid, raises exception if invalid
    """
    import jsonschema
    
    try:
        jsonschema.validate(config, PIPELINE_CONFIG_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        raise ValueError(f"Invalid pipeline configuration: {e.message}")


if __name__ == "__main__":
    # Compile pipeline for testing
    compiled_path = compile_pipeline()
    print(f"Pipeline compiled to: {compiled_path}")