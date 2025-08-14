"""
Pipeline Orchestration Implementation for Market Regime Training
KFP v2 Pipeline Management, Scheduling, and Monitoring

This module provides:
- KFP v2 pipeline definition and compilation
- Pipeline scheduling and trigger mechanisms
- Pipeline monitoring and status tracking
- Pipeline configuration management system
- Error handling and retry logic
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
import time

import kfp
from kfp import dsl
from kfp.dsl import pipeline, component
from kfp import compiler
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google.cloud import scheduler_v1
from google.cloud import monitoring_v3
from google.cloud.exceptions import NotFound


class PipelineConfigManager:
    """Pipeline configuration management system"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self._config_cache = {}
        self._last_modified = {}
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load pipeline configuration from YAML file"""
        
        try:
            current_modified = self.config_path.stat().st_mtime
            
            # Check if reload is needed
            if (not force_reload and 
                str(self.config_path) in self._config_cache and
                self._last_modified.get(str(self.config_path)) == current_modified):
                return self._config_cache[str(self.config_path)]
            
            # Load configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Cache configuration
            self._config_cache[str(self.config_path)] = config
            self._last_modified[str(self.config_path)] = current_modified
            
            self.logger.info(f"Loaded pipeline configuration from {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline configuration"""
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Required sections
            required_sections = ["project", "data", "models", "training", "pipeline"]
            for section in required_sections:
                if section not in config:
                    validation_results["errors"].append(f"Missing required section: {section}")
                    validation_results["valid"] = False
            
            # Project configuration
            if "project" in config:
                project_config = config["project"]
                required_project_fields = ["project_id", "location", "staging_bucket"]
                for field in required_project_fields:
                    if field not in project_config:
                        validation_results["errors"].append(f"Missing project field: {field}")
                        validation_results["valid"] = False
            
            # Model configuration
            if "models" in config:
                models_config = config["models"]
                if "baseline_models" in models_config:
                    for model in models_config["baseline_models"]:
                        if "name" not in model:
                            validation_results["errors"].append("Model missing name field")
                            validation_results["valid"] = False
                        if "enabled" not in model:
                            validation_results["warnings"].append(f"Model {model.get('name', 'unknown')} missing enabled field")
            
            # Training configuration
            if "training" in config:
                training_config = config["training"]
                if "experiment_name" not in training_config:
                    validation_results["warnings"].append("Missing experiment_name in training config")
            
            self.logger.info(f"Configuration validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def get_environment_config(self, environment: str = "development") -> Dict[str, Any]:
        """Get environment-specific configuration"""
        
        config = self.load_config()
        
        # Apply environment overrides
        if "environments" in config and environment in config["environments"]:
            env_config = config["environments"][environment]
            
            # Deep merge environment config
            merged_config = self._deep_merge(config.copy(), env_config)
            return merged_config
        
        return config
    
    def _deep_merge(self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
        
        return base_dict


class PipelineCompiler:
    """KFP v2 pipeline definition and compilation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compile_training_pipeline(self, output_path: str) -> str:
        """Compile the training pipeline to YAML"""
        
        try:
            # Import the pipeline function
            from .training_pipeline import market_regime_training_pipeline
            
            # Create compiler
            kfp_compiler = compiler.Compiler()
            
            # Compile pipeline
            kfp_compiler.compile(
                pipeline_func=market_regime_training_pipeline,
                package_path=output_path
            )
            
            self.logger.info(f"Pipeline compiled successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Pipeline compilation failed: {str(e)}")
            raise
    
    def create_pipeline_spec(self) -> Dict[str, Any]:
        """Create pipeline specification from config"""
        
        pipeline_spec = {
            "pipelineInfo": {
                "name": self.config.get("pipeline", {}).get("name", "market-regime-training-pipeline"),
                "description": self.config.get("pipeline", {}).get("description", "Market regime training pipeline")
            },
            "deploymentSpec": {
                "executors": self._create_executors_spec()
            },
            "components": self._create_components_spec(),
            "root": self._create_root_spec()
        }
        
        return pipeline_spec
    
    def _create_executors_spec(self) -> Dict[str, Any]:
        """Create executors specification"""
        
        return {
            "exec-data-preparation": {
                "container": {
                    "image": "python:3.8-slim",
                    "command": ["python", "/app/data_preparation.py"],
                    "args": ["--config", "{{$.inputs.parameters['config']}}"]
                }
            },
            "exec-model-training": {
                "container": {
                    "image": "python:3.8-slim",
                    "command": ["python", "/app/model_training.py"],
                    "args": ["--model-type", "{{$.inputs.parameters['model_type']}}"]
                }
            },
            "exec-model-evaluation": {
                "container": {
                    "image": "python:3.8-slim", 
                    "command": ["python", "/app/model_evaluation.py"]
                }
            }
        }
    
    def _create_components_spec(self) -> Dict[str, Any]:
        """Create components specification"""
        
        return {
            "comp-data-preparation": {
                "executorLabel": "exec-data-preparation",
                "inputDefinitions": {
                    "parameters": {
                        "project_id": {"type": "STRING"},
                        "dataset_table": {"type": "STRING"},
                        "staging_bucket": {"type": "STRING"}
                    }
                },
                "outputDefinitions": {
                    "artifacts": {
                        "train_data": {"artifactType": {"schemaTitle": "system.Dataset"}},
                        "validation_data": {"artifactType": {"schemaTitle": "system.Dataset"}},
                        "test_data": {"artifactType": {"schemaTitle": "system.Dataset"}}
                    }
                }
            }
        }
    
    def _create_root_spec(self) -> Dict[str, Any]:
        """Create root DAG specification"""
        
        return {
            "dag": {
                "tasks": {
                    "data-preparation": {
                        "componentRef": {"name": "comp-data-preparation"},
                        "inputs": {
                            "parameters": {
                                "project_id": {"runtimeValue": {"constantValue": {"stringValue": self.config["project"]["project_id"]}}},
                                "dataset_table": {"runtimeValue": {"constantValue": {"stringValue": self.config["data"]["source"]["full_table_id"]}}},
                                "staging_bucket": {"runtimeValue": {"constantValue": {"stringValue": self.config["project"]["staging_bucket"]}}}
                            }
                        }
                    }
                }
            }
        }


class PipelineScheduler:
    """Pipeline scheduling and trigger mechanisms"""
    
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.scheduler_client = scheduler_v1.CloudSchedulerClient()
        self.aiplatform_client = aiplatform.gapic.PipelineServiceClient()
    
    def create_scheduled_job(self,
                           job_name: str,
                           schedule: str,
                           pipeline_spec_uri: str,
                           pipeline_parameters: Dict[str, Any],
                           timezone: str = "UTC") -> str:
        """
        Create scheduled pipeline job
        
        Args:
            job_name: Name for the scheduled job
            schedule: Cron schedule expression
            pipeline_spec_uri: GCS URI to compiled pipeline
            pipeline_parameters: Pipeline parameters
            timezone: Timezone for schedule
            
        Returns:
            Job resource name
        """
        
        try:
            # Create Cloud Scheduler job
            parent = f"projects/{self.project_id}/locations/{self.location}"
            
            # Prepare HTTP target for pipeline execution
            http_target = {
                "uri": f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/pipelineJobs",
                "http_method": scheduler_v1.HttpMethod.POST,
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer $(gcloud auth print-access-token)"
                },
                "body": json.dumps({
                    "displayName": f"{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "templateUri": pipeline_spec_uri,
                    "parameterValues": pipeline_parameters
                }).encode()
            }
            
            job = {
                "name": f"{parent}/jobs/{job_name}",
                "schedule": schedule,
                "time_zone": timezone,
                "http_target": http_target,
                "description": f"Scheduled training pipeline: {job_name}"
            }
            
            # Create the job
            response = self.scheduler_client.create_job(
                parent=parent,
                job=job
            )
            
            self.logger.info(f"Created scheduled job: {response.name}")
            return response.name
            
        except Exception as e:
            self.logger.error(f"Failed to create scheduled job: {str(e)}")
            raise
    
    def update_schedule(self, job_name: str, new_schedule: str) -> bool:
        """Update existing schedule"""
        
        try:
            job_path = f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}"
            
            # Get existing job
            job = self.scheduler_client.get_job(name=job_path)
            
            # Update schedule
            job.schedule = new_schedule
            
            # Update the job
            self.scheduler_client.update_job(job=job)
            
            self.logger.info(f"Updated schedule for job: {job_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update schedule: {str(e)}")
            return False
    
    def pause_schedule(self, job_name: str) -> bool:
        """Pause scheduled job"""
        
        try:
            job_path = f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}"
            
            self.scheduler_client.pause_job(name=job_path)
            
            self.logger.info(f"Paused scheduled job: {job_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pause schedule: {str(e)}")
            return False
    
    def resume_schedule(self, job_name: str) -> bool:
        """Resume scheduled job"""
        
        try:
            job_path = f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}"
            
            self.scheduler_client.resume_job(name=job_path)
            
            self.logger.info(f"Resumed scheduled job: {job_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume schedule: {str(e)}")
            return False
    
    def trigger_pipeline_run(self,
                           pipeline_spec_uri: str,
                           pipeline_parameters: Dict[str, Any],
                           run_name: Optional[str] = None) -> str:
        """Manually trigger pipeline run"""
        
        try:
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Create pipeline job
            if run_name is None:
                run_name = f"manual_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            job = aiplatform.PipelineJob(
                display_name=run_name,
                template_path=pipeline_spec_uri,
                parameter_values=pipeline_parameters,
                enable_caching=True
            )
            
            # Submit job
            job.submit()
            
            self.logger.info(f"Triggered pipeline run: {job.resource_name}")
            return job.resource_name
            
        except Exception as e:
            self.logger.error(f"Failed to trigger pipeline run: {str(e)}")
            raise


class PipelineMonitor:
    """Pipeline monitoring and status tracking"""
    
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def get_pipeline_status(self, pipeline_job_name: str) -> Dict[str, Any]:
        """Get current pipeline status"""
        
        try:
            # Get pipeline job
            job = aiplatform.PipelineJob.get(pipeline_job_name)
            
            status = {
                "job_name": pipeline_job_name,
                "display_name": job.display_name,
                "state": job.state.name if job.state else "UNKNOWN",
                "create_time": job.create_time.isoformat() if job.create_time else None,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "update_time": job.update_time.isoformat() if job.update_time else None
            }
            
            # Add error information if available
            if hasattr(job, 'error') and job.error:
                status["error"] = {
                    "code": job.error.code,
                    "message": job.error.message
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline status: {str(e)}")
            return {
                "job_name": pipeline_job_name,
                "error": str(e)
            }
    
    def list_recent_runs(self, hours: int = 24) -> List[Dict[str, Any]]:
        """List recent pipeline runs"""
        
        try:
            # Calculate time filter
            start_time = datetime.now() - timedelta(hours=hours)
            
            # List pipeline jobs
            jobs = aiplatform.PipelineJob.list(
                filter=f'create_time>="{start_time.isoformat()}"',
                order_by="create_time desc"
            )
            
            runs = []
            for job in jobs:
                run_info = {
                    "job_name": job.resource_name,
                    "display_name": job.display_name,
                    "state": job.state.name if job.state else "UNKNOWN",
                    "create_time": job.create_time.isoformat() if job.create_time else None,
                    "duration_minutes": None
                }
                
                # Calculate duration if completed
                if job.start_time and job.end_time:
                    duration = job.end_time - job.start_time
                    run_info["duration_minutes"] = duration.total_seconds() / 60
                
                runs.append(run_info)
            
            return runs
            
        except Exception as e:
            self.logger.error(f"Failed to list recent runs: {str(e)}")
            return []
    
    def get_pipeline_metrics(self, pipeline_job_name: str) -> Dict[str, Any]:
        """Get pipeline execution metrics"""
        
        try:
            job = aiplatform.PipelineJob.get(pipeline_job_name)
            
            metrics = {
                "execution_metrics": {},
                "performance_metrics": {}
            }
            
            # Basic execution metrics
            if job.start_time and job.end_time:
                duration = job.end_time - job.start_time
                metrics["execution_metrics"]["total_duration_minutes"] = duration.total_seconds() / 60
            
            if job.create_time and job.start_time:
                queue_time = job.start_time - job.create_time
                metrics["execution_metrics"]["queue_time_minutes"] = queue_time.total_seconds() / 60
            
            # Component-level metrics (if available)
            try:
                task_details = job.task_details
                if task_details:
                    component_metrics = {}
                    for task in task_details:
                        task_name = task.task_name
                        if task.start_time and task.end_time:
                            duration = task.end_time - task.start_time
                            component_metrics[task_name] = {
                                "duration_minutes": duration.total_seconds() / 60,
                                "state": task.state.name if task.state else "UNKNOWN"
                            }
                    
                    metrics["component_metrics"] = component_metrics
            except:
                pass
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline metrics: {str(e)}")
            return {"error": str(e)}
    
    def create_monitoring_dashboard(self, dashboard_name: str) -> str:
        """Create monitoring dashboard for pipelines"""
        
        try:
            # This would create a custom monitoring dashboard
            # For now, return a placeholder
            dashboard_config = {
                "displayName": dashboard_name,
                "mosaicLayout": {
                    "tiles": [
                        {
                            "width": 6,
                            "height": 4,
                            "widget": {
                                "title": "Pipeline Success Rate",
                                "xyChart": {
                                    "dataSets": [
                                        {
                                            "timeSeriesQuery": {
                                                "unitOverride": "1",
                                                "outputFullResourceTypes": False
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    ]
                }
            }
            
            # In a real implementation, this would create the dashboard
            # using the monitoring API
            
            self.logger.info(f"Dashboard configuration created: {dashboard_name}")
            return json.dumps(dashboard_config, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to create monitoring dashboard: {str(e)}")
            raise


class PipelineErrorHandler:
    """Error handling and retry logic for pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Retry configuration
        self.max_retries = config.get("retry_policy", {}).get("max_retries", 2)
        self.retry_delay = config.get("retry_policy", {}).get("retry_delay_seconds", 300)
        
        # Notification configuration
        self.notification_config = config.get("notifications", {})
    
    def handle_pipeline_failure(self, pipeline_job_name: str, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline failure with appropriate actions"""
        
        try:
            self.logger.error(f"Pipeline failure detected: {pipeline_job_name}")
            
            # Analyze error
            error_analysis = self._analyze_error(error_details)
            
            # Determine if retry is appropriate
            should_retry = self._should_retry_error(error_analysis)
            
            response = {
                "pipeline_job": pipeline_job_name,
                "error_analysis": error_analysis,
                "should_retry": should_retry,
                "actions_taken": []
            }
            
            # Send notifications
            if self.notification_config.get("email_on_failure", False):
                notification_sent = self._send_failure_notification(pipeline_job_name, error_analysis)
                response["actions_taken"].append(f"Notification sent: {notification_sent}")
            
            # Log failure details
            self._log_failure_details(pipeline_job_name, error_analysis)
            response["actions_taken"].append("Failure details logged")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in failure handler: {str(e)}")
            return {
                "pipeline_job": pipeline_job_name,
                "handler_error": str(e)
            }
    
    def implement_retry_logic(self, pipeline_job_name: str, 
                            retry_count: int,
                            original_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement retry logic for failed pipelines"""
        
        try:
            if retry_count >= self.max_retries:
                self.logger.error(f"Maximum retries exceeded for {pipeline_job_name}")
                return {
                    "status": "max_retries_exceeded",
                    "retry_count": retry_count
                }
            
            # Wait before retry
            time.sleep(self.retry_delay)
            
            # Modify parameters for retry if needed
            retry_parameters = self._modify_parameters_for_retry(original_parameters, retry_count)
            
            # Trigger retry
            scheduler = PipelineScheduler(self.config["project"]["project_id"], 
                                        self.config["project"]["location"])
            
            new_job_name = scheduler.trigger_pipeline_run(
                pipeline_spec_uri=original_parameters.get("pipeline_spec_uri"),
                pipeline_parameters=retry_parameters,
                run_name=f"retry_{retry_count}_{pipeline_job_name.split('/')[-1]}"
            )
            
            self.logger.info(f"Retry {retry_count} triggered: {new_job_name}")
            
            return {
                "status": "retry_triggered",
                "new_job_name": new_job_name,
                "retry_count": retry_count + 1
            }
            
        except Exception as e:
            self.logger.error(f"Retry logic failed: {str(e)}")
            return {
                "status": "retry_failed",
                "error": str(e)
            }
    
    def _analyze_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error to determine cause and appropriate response"""
        
        analysis = {
            "error_type": "unknown",
            "error_category": "unknown",
            "is_transient": False,
            "recommended_action": "manual_investigation"
        }
        
        error_message = error_details.get("message", "").lower()
        
        # Categorize errors
        if "timeout" in error_message or "deadline" in error_message:
            analysis.update({
                "error_type": "timeout",
                "error_category": "resource",
                "is_transient": True,
                "recommended_action": "retry_with_more_resources"
            })
        elif "out of memory" in error_message or "oom" in error_message:
            analysis.update({
                "error_type": "out_of_memory",
                "error_category": "resource",
                "is_transient": False,
                "recommended_action": "increase_memory"
            })
        elif "permission" in error_message or "unauthorized" in error_message:
            analysis.update({
                "error_type": "permission_error",
                "error_category": "authentication",
                "is_transient": False,
                "recommended_action": "check_iam_permissions"
            })
        elif "quota" in error_message or "limit" in error_message:
            analysis.update({
                "error_type": "quota_exceeded",
                "error_category": "resource",
                "is_transient": True,
                "recommended_action": "wait_and_retry"
            })
        
        return analysis
    
    def _should_retry_error(self, error_analysis: Dict[str, Any]) -> bool:
        """Determine if error should trigger automatic retry"""
        
        # Retry transient errors
        if error_analysis.get("is_transient", False):
            return True
        
        # Don't retry authentication/permission errors
        if error_analysis.get("error_category") == "authentication":
            return False
        
        # Don't retry out of memory errors without modification
        if error_analysis.get("error_type") == "out_of_memory":
            return False
        
        return False
    
    def _send_failure_notification(self, pipeline_job_name: str, 
                                 error_analysis: Dict[str, Any]) -> bool:
        """Send failure notification"""
        
        try:
            # In a real implementation, this would send email/Slack notification
            notification_message = f"""
            Pipeline Failure Alert
            
            Pipeline: {pipeline_job_name}
            Error Type: {error_analysis.get('error_type', 'unknown')}
            Error Category: {error_analysis.get('error_category', 'unknown')}
            Recommended Action: {error_analysis.get('recommended_action', 'manual_investigation')}
            
            Please investigate and take appropriate action.
            """
            
            self.logger.warning(f"Notification would be sent: {notification_message}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")
            return False
    
    def _log_failure_details(self, pipeline_job_name: str, error_analysis: Dict[str, Any]):
        """Log detailed failure information"""
        
        failure_log = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_job": pipeline_job_name,
            "error_analysis": error_analysis,
            "system_info": {
                "project_id": self.config["project"]["project_id"],
                "location": self.config["project"]["location"]
            }
        }
        
        # Log to structured logger
        self.logger.error(f"Pipeline failure details: {json.dumps(failure_log, indent=2)}")
    
    def _modify_parameters_for_retry(self, original_parameters: Dict[str, Any], 
                                   retry_count: int) -> Dict[str, Any]:
        """Modify parameters for retry attempt"""
        
        retry_parameters = original_parameters.copy()
        
        # Add retry metadata
        retry_parameters["retry_count"] = retry_count
        retry_parameters["retry_timestamp"] = datetime.now().isoformat()
        
        # Potentially modify resource allocation
        if "machine_type" in retry_parameters:
            # Scale up resources for retry
            machine_type = retry_parameters["machine_type"]
            if "standard-4" in machine_type and retry_count == 1:
                retry_parameters["machine_type"] = machine_type.replace("standard-4", "standard-8")
        
        return retry_parameters


class PipelineOrchestrator:
    """Main pipeline orchestration coordinator"""
    
    def __init__(self, config_path: str, environment: str = "development"):
        self.config_manager = PipelineConfigManager(config_path)
        self.config = self.config_manager.get_environment_config(environment)
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.compiler = PipelineCompiler(self.config)
        self.scheduler = PipelineScheduler(
            self.config["project"]["project_id"],
            self.config["project"]["location"]
        )
        self.monitor = PipelineMonitor(
            self.config["project"]["project_id"],
            self.config["project"]["location"]
        )
        self.error_handler = PipelineErrorHandler(self.config)
    
    def deploy_pipeline(self, pipeline_name: str, enable_schedule: bool = True) -> Dict[str, Any]:
        """Deploy complete pipeline with scheduling and monitoring"""
        
        try:
            deployment_results = {
                "pipeline_name": pipeline_name,
                "environment": self.environment,
                "deployment_timestamp": datetime.now().isoformat(),
                "status": "in_progress"
            }
            
            # Validate configuration
            validation_results = self.config_manager.validate_config(self.config)
            if not validation_results["valid"]:
                raise ValueError(f"Configuration validation failed: {validation_results['errors']}")
            
            # Compile pipeline
            compiled_path = f"/tmp/{pipeline_name}_compiled.yaml"
            self.compiler.compile_training_pipeline(compiled_path)
            deployment_results["compiled_pipeline_path"] = compiled_path
            
            # Upload compiled pipeline to GCS (in real implementation)
            pipeline_spec_uri = f"gs://{self.config['project']['staging_bucket']}/pipelines/{pipeline_name}.yaml"
            deployment_results["pipeline_spec_uri"] = pipeline_spec_uri
            
            # Create scheduled job if enabled
            if enable_schedule and self.config.get("pipeline", {}).get("scheduling", {}).get("enabled", False):
                schedule_config = self.config["pipeline"]["scheduling"]
                
                job_name = self.scheduler.create_scheduled_job(
                    job_name=f"{pipeline_name}_scheduled",
                    schedule=schedule_config.get("cron_schedule", "0 2 * * 0"),
                    pipeline_spec_uri=pipeline_spec_uri,
                    pipeline_parameters=self._extract_pipeline_parameters(),
                    timezone=schedule_config.get("timezone", "UTC")
                )
                
                deployment_results["scheduled_job"] = job_name
            
            # Set up monitoring
            monitoring_dashboard = self.monitor.create_monitoring_dashboard(
                f"{pipeline_name}_dashboard"
            )
            deployment_results["monitoring_dashboard"] = monitoring_dashboard
            
            deployment_results["status"] = "success"
            self.logger.info(f"Pipeline deployed successfully: {pipeline_name}")
            
            return deployment_results
            
        except Exception as e:
            self.logger.error(f"Pipeline deployment failed: {str(e)}")
            return {
                "pipeline_name": pipeline_name,
                "status": "failed",
                "error": str(e)
            }
    
    def trigger_training_run(self, run_parameters: Optional[Dict[str, Any]] = None) -> str:
        """Trigger manual training run"""
        
        try:
            # Merge default parameters with provided ones
            default_parameters = self._extract_pipeline_parameters()
            if run_parameters:
                default_parameters.update(run_parameters)
            
            # Get pipeline spec URI
            pipeline_spec_uri = f"gs://{self.config['project']['staging_bucket']}/pipelines/market_regime_training_pipeline.yaml"
            
            # Trigger run
            job_name = self.scheduler.trigger_pipeline_run(
                pipeline_spec_uri=pipeline_spec_uri,
                pipeline_parameters=default_parameters
            )
            
            self.logger.info(f"Training run triggered: {job_name}")
            return job_name
            
        except Exception as e:
            self.logger.error(f"Failed to trigger training run: {str(e)}")
            raise
    
    def monitor_pipeline_health(self) -> Dict[str, Any]:
        """Monitor overall pipeline health"""
        
        try:
            # Get recent runs
            recent_runs = self.monitor.list_recent_runs(hours=24)
            
            # Calculate health metrics
            total_runs = len(recent_runs)
            successful_runs = len([r for r in recent_runs if r["state"] == "PIPELINE_STATE_SUCCEEDED"])
            failed_runs = len([r for r in recent_runs if r["state"] == "PIPELINE_STATE_FAILED"])
            
            success_rate = successful_runs / total_runs if total_runs > 0 else 0
            
            health_status = {
                "overall_health": "healthy" if success_rate >= 0.8 else "degraded" if success_rate >= 0.5 else "unhealthy",
                "metrics": {
                    "total_runs_24h": total_runs,
                    "successful_runs": successful_runs,
                    "failed_runs": failed_runs,
                    "success_rate": success_rate
                },
                "recent_runs": recent_runs[:5],  # Last 5 runs
                "timestamp": datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Failed to monitor pipeline health: {str(e)}")
            return {
                "overall_health": "unknown",
                "error": str(e)
            }
    
    def _extract_pipeline_parameters(self) -> Dict[str, Any]:
        """Extract pipeline parameters from configuration"""
        
        return {
            "project_id": self.config["project"]["project_id"],
            "location": self.config["project"]["location"],
            "staging_bucket": self.config["project"]["staging_bucket"],
            "training_dataset_table": self.config["data"]["source"]["full_table_id"],
            "experiment_name": self.config["training"]["experiment_name"],
            "model_display_name": f"market-regime-model-{self.environment}",
            "model_types": [model["name"] for model in self.config["models"]["baseline_models"] if model.get("enabled", False)],
            "validation_split": self.config["data"]["splits"]["validation_ratio"],
            "test_split": self.config["data"]["splits"]["test_ratio"]
        }


# Utility functions for KFP component integration
def orchestrate_training_pipeline(
    config_path: str,
    environment: str,
    enable_scheduling: bool = True
) -> Dict[str, Any]:
    """
    Orchestrate training pipeline deployment
    
    This function serves as the entry point for pipeline orchestration
    """
    
    try:
        orchestrator = PipelineOrchestrator(config_path, environment)
        
        results = orchestrator.deploy_pipeline(
            pipeline_name="market_regime_training",
            enable_schedule=enable_scheduling
        )
        
        return results
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }