"""
Data Versioning and Lineage Tracking for Training Pipeline
Comprehensive data versioning, lineage tracking, and metadata management

This module provides:
- Data versioning with semantic versioning
- Complete data lineage tracking from source to training
- Metadata management and artifact tracking
- Data quality monitoring and drift detection
- Integration with Vertex AI ML Metadata
"""

import logging
from typing import Dict, Any, List, Optional, NamedTuple
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
import pandas as pd
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform import metadata
import uuid


class DataVersion:
    """Data version information with semantic versioning"""
    
    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0, 
                 timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
    @property
    def version_string(self) -> str:
        """Get semantic version string"""
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    @property
    def version_tag(self) -> str:
        """Get version tag with timestamp"""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{self.version_string}_{timestamp_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "version_string": self.version_string,
            "version_tag": self.version_tag,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        """Create from dictionary"""
        return cls(
            major=data["major"],
            minor=data["minor"],
            patch=data["patch"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class DataLineage:
    """Data lineage tracking for training datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.lineage_data = {
            "lineage_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "source_data": {},
            "transformations": [],
            "outputs": {},
            "metadata": {},
            "quality_metrics": {}
        }
        
    def add_source_data(self, source_type: str, source_path: str, 
                       schema_info: Dict[str, Any], row_count: int,
                       date_range: Dict[str, str], checksum: str):
        """Add source data information to lineage"""
        source_info = {
            "source_type": source_type,
            "source_path": source_path,
            "schema_info": schema_info,
            "row_count": row_count,
            "date_range": date_range,
            "checksum": checksum,
            "accessed_at": datetime.now().isoformat()
        }
        
        if source_type not in self.lineage_data["source_data"]:
            self.lineage_data["source_data"][source_type] = []
        
        self.lineage_data["source_data"][source_type].append(source_info)
        self.logger.info(f"Added source data: {source_type} - {source_path}")
    
    def add_transformation(self, transformation_type: str, transformation_name: str,
                          input_info: Dict[str, Any], output_info: Dict[str, Any],
                          parameters: Dict[str, Any]):
        """Add transformation step to lineage"""
        transformation_info = {
            "transformation_id": str(uuid.uuid4()),
            "transformation_type": transformation_type,
            "transformation_name": transformation_name,
            "input_info": input_info,
            "output_info": output_info,
            "parameters": parameters,
            "executed_at": datetime.now().isoformat()
        }
        
        self.lineage_data["transformations"].append(transformation_info)
        self.logger.info(f"Added transformation: {transformation_name}")
    
    def add_output_data(self, output_type: str, output_path: str,
                       schema_info: Dict[str, Any], row_count: int,
                       checksum: str, data_split: str):
        """Add output data information to lineage"""
        output_info = {
            "output_type": output_type,
            "output_path": output_path,
            "schema_info": schema_info,
            "row_count": row_count,
            "checksum": checksum,
            "data_split": data_split,
            "created_at": datetime.now().isoformat()
        }
        
        if output_type not in self.lineage_data["outputs"]:
            self.lineage_data["outputs"][output_type] = []
        
        self.lineage_data["outputs"][output_type].append(output_info)
        self.logger.info(f"Added output data: {output_type} - {output_path}")
    
    def add_quality_metrics(self, metrics: Dict[str, Any]):
        """Add data quality metrics to lineage"""
        self.lineage_data["quality_metrics"].update(metrics)
        self.logger.info("Added quality metrics to lineage")
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to lineage"""
        self.lineage_data["metadata"][key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete lineage data"""
        return self.lineage_data.copy()
    
    def save_lineage(self, output_path: str):
        """Save lineage data to file"""
        with open(output_path, 'w') as f:
            json.dump(self.lineage_data, f, indent=2)
        self.logger.info(f"Lineage data saved to: {output_path}")


class DataVersionManager:
    """Manage data versions and artifacts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.storage_client = storage.Client(project=config['project']['project_id'])
        self.bigquery_client = bigquery.Client(project=config['project']['project_id'])
        
        # Version storage configuration
        self.version_bucket = config['project']['staging_bucket']
        self.version_prefix = "data_versions"
        
    def create_data_version(self, source_data_info: Dict[str, Any], 
                           processing_config: Dict[str, Any]) -> DataVersion:
        """Create a new data version based on source data and processing config"""
        
        # Generate version based on data changes
        data_hash = self._generate_data_hash(source_data_info, processing_config)
        
        # Check for existing versions
        existing_versions = self._get_existing_versions()
        
        if existing_versions:
            latest_version = max(existing_versions, key=lambda v: (v.major, v.minor, v.patch))
            
            # Determine version increment based on changes
            if self._is_major_change(source_data_info, processing_config):
                new_version = DataVersion(
                    major=latest_version.major + 1,
                    minor=0,
                    patch=0,
                    metadata={"change_type": "major", "data_hash": data_hash}
                )
            elif self._is_minor_change(source_data_info, processing_config):
                new_version = DataVersion(
                    major=latest_version.major,
                    minor=latest_version.minor + 1,
                    patch=0,
                    metadata={"change_type": "minor", "data_hash": data_hash}
                )
            else:
                new_version = DataVersion(
                    major=latest_version.major,
                    minor=latest_version.minor,
                    patch=latest_version.patch + 1,
                    metadata={"change_type": "patch", "data_hash": data_hash}
                )
        else:
            # First version
            new_version = DataVersion(
                major=1,
                minor=0,
                patch=0,
                metadata={"change_type": "initial", "data_hash": data_hash}
            )
        
        # Add additional metadata
        new_version.metadata.update({
            "source_tables": source_data_info.get("tables", []),
            "feature_count": source_data_info.get("feature_count", 0),
            "row_count": source_data_info.get("row_count", 0),
            "date_range": source_data_info.get("date_range", {}),
            "processing_config": processing_config
        })
        
        self.logger.info(f"Created data version: {new_version.version_tag}")
        return new_version
    
    def save_version_metadata(self, version: DataVersion, lineage: DataLineage) -> str:
        """Save version metadata and lineage to GCS"""
        
        # Create version metadata
        version_metadata = {
            "version": version.to_dict(),
            "lineage": lineage.to_dict(),
            "created_at": datetime.now().isoformat()
        }
        
        # Save to GCS
        blob_path = f"{self.version_prefix}/{version.version_tag}/metadata.json"
        bucket = self.storage_client.bucket(self.version_bucket)
        blob = bucket.blob(blob_path)
        
        blob.upload_from_string(json.dumps(version_metadata, indent=2))
        
        gcs_path = f"gs://{self.version_bucket}/{blob_path}"
        self.logger.info(f"Version metadata saved to: {gcs_path}")
        
        return gcs_path
    
    def get_version_metadata(self, version_tag: str) -> Optional[Dict[str, Any]]:
        """Retrieve version metadata from GCS"""
        
        blob_path = f"{self.version_prefix}/{version_tag}/metadata.json"
        bucket = self.storage_client.bucket(self.version_bucket)
        blob = bucket.blob(blob_path)
        
        if blob.exists():
            metadata_json = blob.download_as_text()
            return json.loads(metadata_json)
        else:
            self.logger.warning(f"Version metadata not found: {version_tag}")
            return None
    
    def list_versions(self) -> List[DataVersion]:
        """List all available data versions"""
        
        bucket = self.storage_client.bucket(self.version_bucket)
        blobs = bucket.list_blobs(prefix=f"{self.version_prefix}/")
        
        versions = []
        for blob in blobs:
            if blob.name.endswith("/metadata.json"):
                try:
                    metadata_json = blob.download_as_text()
                    metadata = json.loads(metadata_json)
                    version = DataVersion.from_dict(metadata["version"])
                    versions.append(version)
                except Exception as e:
                    self.logger.warning(f"Failed to parse version metadata: {blob.name} - {e}")
        
        return sorted(versions, key=lambda v: (v.major, v.minor, v.patch))
    
    def _get_existing_versions(self) -> List[DataVersion]:
        """Get all existing versions"""
        return self.list_versions()
    
    def _generate_data_hash(self, source_data_info: Dict[str, Any], 
                           processing_config: Dict[str, Any]) -> str:
        """Generate hash for data and processing configuration"""
        
        # Combine source data info and processing config
        combined_data = {
            "source_data": source_data_info,
            "processing_config": processing_config
        }
        
        # Generate hash
        data_string = json.dumps(combined_data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()[:16]
    
    def _is_major_change(self, source_data_info: Dict[str, Any], 
                        processing_config: Dict[str, Any]) -> bool:
        """Determine if changes constitute a major version change"""
        
        # Major changes: schema changes, new data sources, major config changes
        major_change_indicators = [
            "new_feature_columns",
            "removed_feature_columns",
            "new_data_sources",
            "major_preprocessing_changes"
        ]
        
        # Check for major changes (simplified logic)
        return any(key in source_data_info for key in major_change_indicators)
    
    def _is_minor_change(self, source_data_info: Dict[str, Any], 
                        processing_config: Dict[str, Any]) -> bool:
        """Determine if changes constitute a minor version change"""
        
        # Minor changes: parameter tuning, minor config changes
        minor_change_indicators = [
            "hyperparameter_changes",
            "preprocessing_parameter_changes",
            "date_range_extension"
        ]
        
        # Check for minor changes (simplified logic)
        return any(key in source_data_info for key in minor_change_indicators)


class DataQualityMonitor:
    """Monitor data quality and detect drift"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data statistics"""
        
        stats = {
            "basic_stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "missing_values_total": df.isnull().sum().sum(),
                "missing_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns))
            },
            "numeric_stats": {},
            "categorical_stats": {},
            "feature_distribution": {}
        }
        
        # Numeric column statistics
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            stats["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "missing_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique())
            }
        
        # Categorical column statistics
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            stats["categorical_stats"][col] = {
                "unique_count": int(df[col].nunique()),
                "missing_count": int(df[col].isnull().sum()),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        
        # Feature distribution (simplified)
        component_prefixes = ['c1_', 'c2_', 'c3_', 'c4_', 'c5_', 'c6_', 'c7_', 'c8_']
        for prefix in component_prefixes:
            component_cols = [col for col in df.columns if col.startswith(prefix)]
            if component_cols:
                stats["feature_distribution"][prefix] = {
                    "feature_count": len(component_cols),
                    "mean_value": float(df[component_cols].mean().mean()),
                    "std_value": float(df[component_cols].std().mean())
                }
        
        return stats
    
    def detect_data_drift(self, current_stats: Dict[str, Any], 
                         reference_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect data drift between current and reference statistics"""
        
        drift_results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "drift_details": {},
            "alerts": []
        }
        
        # Check basic statistics drift
        current_basic = current_stats.get("basic_stats", {})
        reference_basic = reference_stats.get("basic_stats", {})
        
        # Missing ratio drift
        current_missing = current_basic.get("missing_ratio", 0)
        reference_missing = reference_basic.get("missing_ratio", 0)
        missing_drift = abs(current_missing - reference_missing)
        
        if missing_drift > 0.05:  # 5% threshold
            drift_results["drift_detected"] = True
            drift_results["alerts"].append(f"Missing ratio drift: {missing_drift:.3f}")
        
        # Feature distribution drift (simplified)
        current_dist = current_stats.get("feature_distribution", {})
        reference_dist = reference_stats.get("feature_distribution", {})
        
        for component in current_dist:
            if component in reference_dist:
                current_mean = current_dist[component]["mean_value"]
                reference_mean = reference_dist[component]["mean_value"]
                
                if reference_mean != 0:
                    drift_ratio = abs(current_mean - reference_mean) / abs(reference_mean)
                    if drift_ratio > 0.2:  # 20% threshold
                        drift_results["drift_detected"] = True
                        drift_results["alerts"].append(f"Feature drift in {component}: {drift_ratio:.3f}")
        
        # Calculate overall drift score
        drift_results["drift_score"] = len(drift_results["alerts"]) / 10.0  # Normalized score
        
        return drift_results


class MLMetadataTracker:
    """Track ML metadata using Vertex AI ML Metadata"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config['project']['project_id'],
            location=config['project']['location']
        )
        
    def create_dataset_artifact(self, version: DataVersion, lineage: DataLineage,
                               dataset_paths: Dict[str, str]) -> str:
        """Create dataset artifact in ML Metadata"""
        
        try:
            # Create dataset artifact
            dataset_artifact = metadata.Artifact.create(
                schema_title="system.Dataset",
                display_name=f"training_dataset_{version.version_tag}",
                description=f"Training dataset version {version.version_string}",
                metadata={
                    "version": version.to_dict(),
                    "lineage_id": lineage.lineage_data["lineage_id"],
                    "dataset_paths": dataset_paths,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Created dataset artifact: {dataset_artifact.resource_name}")
            return dataset_artifact.resource_name
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset artifact: {str(e)}")
            return ""
    
    def create_execution_context(self, pipeline_name: str, run_id: str) -> str:
        """Create execution context for pipeline run"""
        
        try:
            execution_context = metadata.Context.create(
                schema_title="system.PipelineRun",
                display_name=f"{pipeline_name}_run_{run_id}",
                description=f"Training pipeline execution {run_id}",
                metadata={
                    "pipeline_name": pipeline_name,
                    "run_id": run_id,
                    "started_at": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Created execution context: {execution_context.resource_name}")
            return execution_context.resource_name
            
        except Exception as e:
            self.logger.error(f"Failed to create execution context: {str(e)}")
            return ""


class DataVersioningPipeline:
    """Main data versioning and lineage tracking pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.version_manager = DataVersionManager(config)
        self.quality_monitor = DataQualityMonitor(config)
        self.ml_metadata_tracker = MLMetadataTracker(config)
        
    def create_versioned_dataset(self, source_data_info: Dict[str, Any],
                                processing_config: Dict[str, Any],
                                dataset_paths: Dict[str, str],
                                data_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Create versioned dataset with complete lineage tracking"""
        
        try:
            # Create data version
            version = self.version_manager.create_data_version(source_data_info, processing_config)
            
            # Create lineage tracker
            lineage = DataLineage(self.config)
            
            # Add source data to lineage
            for table_info in source_data_info.get("tables", []):
                lineage.add_source_data(
                    source_type="bigquery",
                    source_path=table_info["table_id"],
                    schema_info=table_info.get("schema", {}),
                    row_count=table_info.get("row_count", 0),
                    date_range=table_info.get("date_range", {}),
                    checksum=table_info.get("checksum", "")
                )
            
            # Add transformations to lineage
            transformations = processing_config.get("transformations", [])
            for transform in transformations:
                lineage.add_transformation(
                    transformation_type=transform["type"],
                    transformation_name=transform["name"],
                    input_info=transform.get("input_info", {}),
                    output_info=transform.get("output_info", {}),
                    parameters=transform.get("parameters", {})
                )
            
            # Add output data to lineage
            for split_name, path in dataset_paths.items():
                lineage.add_output_data(
                    output_type="training_data",
                    output_path=path,
                    schema_info=data_stats.get("schema_info", {}),
                    row_count=data_stats.get(f"{split_name}_samples", 0),
                    checksum="",  # Would calculate actual checksum in production
                    data_split=split_name
                )
            
            # Add quality metrics to lineage
            lineage.add_quality_metrics(data_stats.get("quality_metrics", {}))
            
            # Save version metadata
            metadata_path = self.version_manager.save_version_metadata(version, lineage)
            
            # Create ML metadata artifacts
            dataset_artifact_name = self.ml_metadata_tracker.create_dataset_artifact(
                version, lineage, dataset_paths
            )
            
            # Create execution context
            run_id = f"data_prep_{version.version_tag}"
            execution_context = self.ml_metadata_tracker.create_execution_context(
                "data_preparation", run_id
            )
            
            result = {
                "version": version.to_dict(),
                "lineage": lineage.to_dict(),
                "metadata_path": metadata_path,
                "dataset_artifact": dataset_artifact_name,
                "execution_context": execution_context,
                "status": "success"
            }
            
            self.logger.info(f"Successfully created versioned dataset: {version.version_tag}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create versioned dataset: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }


# Utility functions for integration
def create_data_version_for_pipeline(
    source_data_info: Dict[str, Any],
    processing_config: Dict[str, Any],
    dataset_paths: Dict[str, str],
    data_stats: Dict[str, Any],
    pipeline_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create data version for training pipeline
    
    This function serves as the entry point for data versioning in KFP components
    """
    
    versioning_pipeline = DataVersioningPipeline(pipeline_config)
    
    return versioning_pipeline.create_versioned_dataset(
        source_data_info=source_data_info,
        processing_config=processing_config,
        dataset_paths=dataset_paths,
        data_stats=data_stats
    )