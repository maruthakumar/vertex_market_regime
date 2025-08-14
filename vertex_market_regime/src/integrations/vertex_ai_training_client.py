"""
Vertex AI Training Client for Market Regime Pipeline
Comprehensive integration with Vertex AI services for training pipeline

This module provides:
- Vertex AI Model Registry client integration
- Model registration with metadata tracking
- Vertex AI Experiments tracking integration
- CustomJob components for heavy compute steps
- Artifact management and versioning
"""

import logging
from typing import Dict, Any, List, Optional, Union
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
from google.cloud import storage
from google.protobuf import struct_pb2


class VertexAIModelRegistry:
    """Client for Vertex AI Model Registry operations"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Initialize clients
        self.model_client = gapic.ModelServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
        self.storage_client = storage.Client(project=project_id)
        
    def register_model(self, 
                      model_artifact_uri: str,
                      model_display_name: str,
                      model_description: str,
                      model_metadata: Dict[str, Any],
                      evaluation_metrics: Dict[str, Any],
                      model_framework: str = "SCIKIT_LEARN",
                      labels: Optional[Dict[str, str]] = None) -> str:
        """
        Register model in Vertex AI Model Registry
        
        Args:
            model_artifact_uri: GCS URI to model artifacts
            model_display_name: Display name for the model
            model_description: Model description
            model_metadata: Additional model metadata
            evaluation_metrics: Model evaluation metrics
            model_framework: ML framework used
            labels: Model labels
            
        Returns:
            Model resource name
        """
        
        try:
            self.logger.info(f"Registering model: {model_display_name}")
            
            # Prepare model metadata
            metadata_dict = {
                "training_framework": "vertex_ai_pipelines",
                "model_type": model_metadata.get("model_type", "classification"),
                "feature_count": model_metadata.get("feature_count", 774),
                "training_dataset": model_metadata.get("training_dataset", ""),
                "training_timestamp": datetime.now().isoformat(),
                "evaluation_metrics": evaluation_metrics,
                **model_metadata
            }
            
            # Convert metadata to Struct format
            metadata_struct = struct_pb2.Struct()
            metadata_struct.update(metadata_dict)
            
            # Prepare labels
            if labels is None:
                labels = {}
            
            labels.update({
                "environment": "development",
                "use_case": "market_regime_prediction",
                "version": "v1.0"
            })
            
            # Create model
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                description=model_description,
                artifact_uri=model_artifact_uri,
                serving_container_image_uri=None,  # No serving container for training-only models
                labels=labels,
                model_id=None,  # Let Vertex AI generate ID
                parent_model=None,
                is_default_version=True,
                version_aliases=["latest"],
                version_description=f"Model trained on {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            self.logger.info(f"Model registered successfully: {model.resource_name}")
            return model.resource_name
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def get_model(self, model_id: str) -> Optional[aiplatform.Model]:
        """Retrieve model from registry"""
        
        try:
            model = aiplatform.Model(model_name=model_id)
            return model
        except Exception as e:
            self.logger.error(f"Failed to retrieve model {model_id}: {str(e)}")
            return None
    
    def list_models(self, filter_string: Optional[str] = None) -> List[aiplatform.Model]:
        """List models in registry"""
        
        try:
            models = aiplatform.Model.list(filter=filter_string)
            return list(models)
        except Exception as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            return []
    
    def update_model_metadata(self, model_id: str, 
                            new_metadata: Dict[str, Any]) -> bool:
        """Update model metadata"""
        
        try:
            model = self.get_model(model_id)
            if model is None:
                return False
            
            # Update labels (metadata is read-only after creation)
            current_labels = model.labels or {}
            current_labels.update({
                f"metadata_{k}": str(v) for k, v in new_metadata.items()
            })
            
            model.update(labels=current_labels)
            
            self.logger.info(f"Model metadata updated: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update model metadata: {str(e)}")
            return False


class VertexAIExperimentsTracker:
    """Vertex AI Experiments tracking integration"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Current experiment and run
        self.current_experiment = None
        self.current_run = None
    
    def create_or_get_experiment(self, experiment_name: str, 
                               experiment_description: str = "") -> aiplatform.Experiment:
        """Create or get existing experiment"""
        
        try:
            # Try to get existing experiment
            try:
                experiment = aiplatform.Experiment.get(experiment_name)
                self.logger.info(f"Using existing experiment: {experiment_name}")
            except:
                # Create new experiment
                experiment = aiplatform.Experiment.create(
                    experiment_name=experiment_name,
                    description=experiment_description or f"Market regime training experiment: {experiment_name}"
                )
                self.logger.info(f"Created new experiment: {experiment_name}")
            
            self.current_experiment = experiment
            return experiment
            
        except Exception as e:
            self.logger.error(f"Failed to create/get experiment: {str(e)}")
            raise
    
    def start_run(self, run_name: str, run_description: str = "") -> aiplatform.ExperimentRun:
        """Start new experiment run"""
        
        try:
            if self.current_experiment is None:
                raise ValueError("No active experiment. Create experiment first.")
            
            # Create unique run name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_run_name = f"{run_name}_{timestamp}"
            
            run = aiplatform.ExperimentRun.create(
                run_name=unique_run_name,
                experiment=self.current_experiment,
                description=run_description or f"Training run: {run_name}"
            )
            
            self.current_run = run
            self.logger.info(f"Started experiment run: {unique_run_name}")
            return run
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment run: {str(e)}")
            raise
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """Log hyperparameters to current run"""
        
        try:
            if self.current_run is None:
                raise ValueError("No active run. Start run first.")
            
            # Convert parameters to appropriate types
            processed_params = {}
            for key, value in parameters.items():
                if isinstance(value, (dict, list)):
                    processed_params[key] = json.dumps(value)
                else:
                    processed_params[key] = value
            
            self.current_run.log_params(processed_params)
            self.logger.info(f"Logged {len(processed_params)} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """Log metrics to current run"""
        
        try:
            if self.current_run is None:
                raise ValueError("No active run. Start run first.")
            
            # Log each metric
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                    self.current_run.log_metric(metric_name, metric_value, step=step)
            
            self.logger.info(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {str(e)}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifact to current run"""
        
        try:
            if self.current_run is None:
                raise ValueError("No active run. Start run first.")
            
            # Log artifact URI
            self.current_run.log_artifact(
                artifact=artifact_path,
                artifact_type=artifact_type
            )
            
            self.logger.info(f"Logged artifact: {artifact_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to log artifact: {str(e)}")
    
    def end_run(self, final_state: str = "COMPLETE"):
        """End current experiment run"""
        
        try:
            if self.current_run is not None:
                self.current_run.end_run(state=final_state)
                self.logger.info(f"Ended experiment run with state: {final_state}")
                self.current_run = None
            
        except Exception as e:
            self.logger.error(f"Failed to end experiment run: {str(e)}")
    
    def get_experiment_runs(self, experiment_name: str) -> List[aiplatform.ExperimentRun]:
        """Get all runs for an experiment"""
        
        try:
            experiment = aiplatform.Experiment.get(experiment_name)
            runs = experiment.list_runs()
            return list(runs)
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {str(e)}")
            return []


class VertexAICustomJobManager:
    """Manager for Vertex AI CustomJob components"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def create_training_job(self,
                          display_name: str,
                          script_path: str,
                          container_uri: str,
                          machine_type: str = "n1-standard-4",
                          accelerator_type: Optional[str] = None,
                          accelerator_count: int = 0,
                          args: Optional[List[str]] = None,
                          environment_variables: Optional[Dict[str, str]] = None,
                          timeout: str = "86400s") -> aiplatform.CustomJob:
        """
        Create CustomJob for training
        
        Args:
            display_name: Job display name
            script_path: Path to training script
            container_uri: Container image URI
            machine_type: Machine type for training
            accelerator_type: GPU accelerator type
            accelerator_count: Number of accelerators
            args: Command line arguments
            environment_variables: Environment variables
            timeout: Job timeout
            
        Returns:
            CustomJob instance
        """
        
        try:
            # Prepare worker pool spec
            machine_spec = {
                "machine_type": machine_type,
            }
            
            if accelerator_type and accelerator_count > 0:
                machine_spec["accelerator_type"] = accelerator_type
                machine_spec["accelerator_count"] = accelerator_count
            
            worker_pool_spec = {
                "machine_spec": machine_spec,
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_uri,
                    "command": ["python", script_path],
                    "args": args or [],
                    "env": [{"name": k, "value": v} for k, v in (environment_variables or {}).items()]
                }
            }
            
            # Create CustomJob
            job = aiplatform.CustomJob(
                display_name=display_name,
                worker_pool_specs=[worker_pool_spec],
                base_output_dir=f"gs://{self.project_id}-training-outputs",
                labels={
                    "use_case": "market_regime_training",
                    "environment": "development"
                }
            )
            
            self.logger.info(f"Created CustomJob: {display_name}")
            return job
            
        except Exception as e:
            self.logger.error(f"Failed to create CustomJob: {str(e)}")
            raise
    
    def submit_and_wait(self, job: aiplatform.CustomJob, 
                       sync: bool = True) -> Dict[str, Any]:
        """Submit CustomJob and optionally wait for completion"""
        
        try:
            self.logger.info(f"Submitting job: {job.display_name}")
            
            # Submit job
            job.run(sync=sync)
            
            if sync:
                # Job completed, get results
                job_state = job.state
                job_error = job.error if hasattr(job, 'error') else None
                
                result = {
                    "job_name": job.resource_name,
                    "state": job_state.name if job_state else "UNKNOWN",
                    "error": str(job_error) if job_error else None,
                    "completion_time": datetime.now().isoformat()
                }
                
                if job_state == gapic.JobState.JOB_STATE_SUCCEEDED:
                    self.logger.info(f"Job completed successfully: {job.display_name}")
                    result["status"] = "success"
                else:
                    self.logger.error(f"Job failed: {job.display_name}, State: {job_state}")
                    result["status"] = "failed"
            else:
                result = {
                    "job_name": job.resource_name,
                    "state": "SUBMITTED",
                    "status": "submitted",
                    "submission_time": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to submit job: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of running job"""
        
        try:
            job = aiplatform.CustomJob.get(job_name)
            
            return {
                "job_name": job_name,
                "state": job.state.name if job.state else "UNKNOWN",
                "display_name": job.display_name,
                "create_time": job.create_time.isoformat() if job.create_time else None,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "error": str(job.error) if hasattr(job, 'error') and job.error else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get job status: {str(e)}")
            return {
                "job_name": job_name,
                "error": str(e)
            }


class VertexAIArtifactManager:
    """Artifact management and versioning for Vertex AI"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.storage_client = storage.Client(project=project_id)
        aiplatform.init(project=project_id, location=location)
    
    def upload_model_artifacts(self,
                             local_model_path: str,
                             model_name: str,
                             version: str,
                             bucket_name: str) -> str:
        """
        Upload model artifacts to GCS
        
        Args:
            local_model_path: Local path to model file
            model_name: Model name
            version: Model version
            bucket_name: GCS bucket name
            
        Returns:
            GCS URI to uploaded model
        """
        
        try:
            # Create destination path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_path = f"models/{model_name}/{version}_{timestamp}/model.pkl"
            
            # Upload to GCS
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            self.logger.info(f"Uploading model to gs://{bucket_name}/{blob_path}")
            blob.upload_from_filename(local_model_path)
            
            gcs_uri = f"gs://{bucket_name}/{blob_path}"
            self.logger.info(f"Model uploaded successfully: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            self.logger.error(f"Failed to upload model artifacts: {str(e)}")
            raise
    
    def upload_training_artifacts(self,
                                local_dir: str,
                                experiment_name: str,
                                run_name: str,
                                bucket_name: str) -> Dict[str, str]:
        """
        Upload training artifacts (metrics, plots, logs) to GCS
        
        Args:
            local_dir: Local directory with artifacts
            experiment_name: Experiment name
            run_name: Run name
            bucket_name: GCS bucket name
            
        Returns:
            Dictionary with artifact URIs
        """
        
        try:
            artifacts = {}
            local_path = Path(local_dir)
            
            if not local_path.exists():
                self.logger.warning(f"Local directory does not exist: {local_dir}")
                return artifacts
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_blob_path = f"experiments/{experiment_name}/{run_name}_{timestamp}"
            
            bucket = self.storage_client.bucket(bucket_name)
            
            # Upload all files in directory
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    blob_path = f"{base_blob_path}/{relative_path}"
                    
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(str(file_path))
                    
                    gcs_uri = f"gs://{bucket_name}/{blob_path}"
                    artifacts[str(relative_path)] = gcs_uri
                    
                    self.logger.info(f"Uploaded artifact: {gcs_uri}")
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to upload training artifacts: {str(e)}")
            return {}
    
    def create_model_version(self,
                           model_artifact_uri: str,
                           parent_model_name: str,
                           version_aliases: List[str],
                           version_description: str) -> str:
        """Create new model version"""
        
        try:
            # Get parent model
            parent_model = aiplatform.Model.get(parent_model_name)
            
            # Create new version
            model_version = aiplatform.Model.upload(
                parent_model=parent_model_name,
                artifact_uri=model_artifact_uri,
                version_aliases=version_aliases,
                version_description=version_description,
                display_name=None  # Inherit from parent
            )
            
            self.logger.info(f"Created model version: {model_version.resource_name}")
            return model_version.resource_name
            
        except Exception as e:
            self.logger.error(f"Failed to create model version: {str(e)}")
            raise
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model"""
        
        try:
            model = aiplatform.Model.get(model_name)
            versions = model.list_versions()
            
            version_info = []
            for version in versions:
                version_info.append({
                    "version_id": version.version_id,
                    "resource_name": version.resource_name,
                    "display_name": version.display_name,
                    "version_aliases": version.version_aliases,
                    "version_description": version.version_description,
                    "create_time": version.create_time.isoformat() if version.create_time else None
                })
            
            return version_info
            
        except Exception as e:
            self.logger.error(f"Failed to list model versions: {str(e)}")
            return []


class VertexAITrainingClient:
    """Main Vertex AI training client integrating all components"""
    
    def __init__(self, project_id: str, location: str = "us-central1", 
                 staging_bucket: str = None):
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket or f"{project_id}-ml-staging"
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_registry = VertexAIModelRegistry(project_id, location)
        self.experiments_tracker = VertexAIExperimentsTracker(project_id, location)
        self.job_manager = VertexAICustomJobManager(project_id, location)
        self.artifact_manager = VertexAIArtifactManager(project_id, location)
    
    def run_training_experiment(self,
                              experiment_config: Dict[str, Any],
                              model_artifacts: Dict[str, str],
                              evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete training experiment with tracking and registration
        
        Args:
            experiment_config: Experiment configuration
            model_artifacts: Paths to model artifacts
            evaluation_results: Model evaluation results
            
        Returns:
            Dictionary with experiment results
        """
        
        try:
            experiment_results = {
                "experiment_name": experiment_config["experiment_name"],
                "started_at": datetime.now().isoformat(),
                "status": "running"
            }
            
            # Create or get experiment
            experiment = self.experiments_tracker.create_or_get_experiment(
                experiment_name=experiment_config["experiment_name"],
                experiment_description=experiment_config.get("description", "")
            )
            
            # Start experiment run
            run_name = experiment_config.get("run_name", "training_run")
            run = self.experiments_tracker.start_run(
                run_name=run_name,
                run_description=experiment_config.get("run_description", "")
            )
            
            experiment_results["run_name"] = run.name
            
            # Log hyperparameters
            if "hyperparameters" in experiment_config:
                self.experiments_tracker.log_parameters(
                    experiment_config["hyperparameters"]
                )
            
            # Log evaluation metrics
            if "classification_metrics" in evaluation_results:
                self.experiments_tracker.log_metrics(
                    evaluation_results["classification_metrics"]
                )
            
            if "forecasting_metrics" in evaluation_results:
                self.experiments_tracker.log_metrics(
                    evaluation_results["forecasting_metrics"]
                )
            
            # Upload and log artifacts
            uploaded_artifacts = {}
            for artifact_type, artifact_path in model_artifacts.items():
                if Path(artifact_path).exists():
                    # Upload to GCS
                    if artifact_type == "model":
                        gcs_uri = self.artifact_manager.upload_model_artifacts(
                            local_model_path=artifact_path,
                            model_name=experiment_config.get("model_name", "market_regime_model"),
                            version=experiment_config.get("version", "v1.0"),
                            bucket_name=self.staging_bucket
                        )
                        uploaded_artifacts[artifact_type] = gcs_uri
                        
                        # Log artifact to experiment
                        self.experiments_tracker.log_artifact(gcs_uri, artifact_type)
            
            experiment_results["artifacts"] = uploaded_artifacts
            
            # Register model if evaluation passes validation
            validation_results = evaluation_results.get("validation_results", {})
            if validation_results.get("passed", False) and "model" in uploaded_artifacts:
                
                model_metadata = {
                    "experiment_name": experiment_config["experiment_name"],
                    "run_name": run_name,
                    "model_type": experiment_config.get("model_type", "classification"),
                    "training_framework": "vertex_ai_pipelines"
                }
                
                model_resource_name = self.model_registry.register_model(
                    model_artifact_uri=uploaded_artifacts["model"],
                    model_display_name=experiment_config.get("model_display_name", "Market Regime Model"),
                    model_description=experiment_config.get("model_description", "Market regime prediction model"),
                    model_metadata=model_metadata,
                    evaluation_metrics=evaluation_results.get("classification_metrics", {}),
                    labels=experiment_config.get("labels", {})
                )
                
                experiment_results["registered_model"] = model_resource_name
            
            # End experiment run
            self.experiments_tracker.end_run("COMPLETE")
            
            experiment_results["status"] = "completed"
            experiment_results["completed_at"] = datetime.now().isoformat()
            
            self.logger.info(f"Training experiment completed: {experiment_config['experiment_name']}")
            
            return experiment_results
            
        except Exception as e:
            self.logger.error(f"Training experiment failed: {str(e)}")
            
            # End run with failure
            try:
                self.experiments_tracker.end_run("FAILED")
            except:
                pass
            
            return {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }


# Utility functions for KFP component integration
def register_model_in_vertex_ai(
    model_path: str,
    model_name: str,
    model_metadata: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    project_id: str,
    location: str,
    staging_bucket: str
) -> Dict[str, Any]:
    """
    Register model in Vertex AI for KFP component
    
    This function serves as the entry point for model registration in KFP components
    """
    
    try:
        # Initialize client
        client = VertexAITrainingClient(project_id, location, staging_bucket)
        
        # Upload model artifacts
        gcs_uri = client.artifact_manager.upload_model_artifacts(
            local_model_path=model_path,
            model_name=model_name,
            version="v1.0",
            bucket_name=staging_bucket
        )
        
        # Register model
        model_resource_name = client.model_registry.register_model(
            model_artifact_uri=gcs_uri,
            model_display_name=model_name,
            model_description=f"Market regime model: {model_name}",
            model_metadata=model_metadata,
            evaluation_metrics=evaluation_metrics
        )
        
        return {
            "status": "success",
            "model_resource_name": model_resource_name,
            "model_uri": gcs_uri
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


def track_experiment_metrics(
    experiment_name: str,
    run_name: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    project_id: str,
    location: str
) -> Dict[str, Any]:
    """
    Track experiment metrics in Vertex AI for KFP component
    """
    
    try:
        # Initialize tracker
        tracker = VertexAIExperimentsTracker(project_id, location)
        
        # Create experiment and run
        experiment = tracker.create_or_get_experiment(experiment_name)
        run = tracker.start_run(run_name)
        
        # Log parameters and metrics
        tracker.log_parameters(hyperparameters)
        tracker.log_metrics(metrics)
        
        # End run
        tracker.end_run()
        
        return {
            "status": "success",
            "experiment_name": experiment_name,
            "run_name": run.name
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }