"""
Data Preparation Pipeline for Market Regime Training
BigQuery to Training Data Conversion and Processing

This module handles the complete data preparation workflow:
- BigQuery offline feature table loading
- Time-based train/validation/test splitting
- Feature preprocessing and validation
- Data quality checks and monitoring
- Output to Parquet/TFRecords format for training
"""

import logging
from typing import Dict, Any, Tuple, List, Optional, NamedTuple
from pathlib import Path
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import json


class DataPreparationConfig:
    """Configuration class for data preparation pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data preparation configuration
        
        Args:
            config: Configuration dictionary from YAML
        """
        self.project_id = config['project']['project_id']
        self.location = config['project']['location']
        self.staging_bucket = config['project']['staging_bucket']
        
        # Data source configuration
        self.source_table = config['data']['source']['full_table_id']
        self.train_ratio = config['data']['splits']['train_ratio']
        self.validation_ratio = config['data']['splits']['validation_ratio']
        self.test_ratio = config['data']['splits']['test_ratio']
        
        # Preprocessing configuration
        self.output_format = config['data']['preprocessing']['output_format']
        self.feature_engineering = config['data']['preprocessing']['feature_engineering']
        self.normalization = config['data']['preprocessing']['normalization']
        self.handle_missing = config['data']['preprocessing']['handle_missing']
        
        # Quality checks
        self.min_samples = config['data']['quality_checks']['min_samples']
        self.max_missing_ratio = config['data']['quality_checks']['max_missing_ratio']
        self.feature_count_validation = config['data']['quality_checks']['feature_count_validation']
        
        # Feature engineering
        self.total_features = config['feature_engineering']['total_features']
        self.component_features = config['feature_engineering']['components']


class DataQualityChecker:
    """Data quality validation and monitoring"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            "total_samples": len(df),
            "total_features": len(df.columns),
            "missing_data_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns)),
            "quality_checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Check minimum sample size
        if len(df) < self.config.min_samples:
            validation_results["errors"].append(
                f"Insufficient samples: {len(df)} < {self.config.min_samples}"
            )
        
        # Check missing data ratio
        if validation_results["missing_data_ratio"] > self.config.max_missing_ratio:
            validation_results["errors"].append(
                f"Too much missing data: {validation_results['missing_data_ratio']:.3f} > {self.config.max_missing_ratio}"
            )
        
        # Feature count validation
        if self.config.feature_count_validation:
            expected_features = self.config.total_features + 10  # Account for metadata columns
            if len(df.columns) < expected_features * 0.9:  # Allow 10% tolerance
                validation_results["warnings"].append(
                    f"Feature count may be low: {len(df.columns)} vs expected ~{expected_features}"
                )
        
        # Target distribution check
        if 'target' in df.columns:
            target_dist = df['target'].value_counts(normalize=True)
            validation_results["target_distribution"] = target_dist.to_dict()
            
            # Check for imbalanced classes
            min_class_ratio = target_dist.min()
            if min_class_ratio < 0.05:  # Less than 5%
                validation_results["warnings"].append(
                    f"Imbalanced target classes: minimum class ratio {min_class_ratio:.3f}"
                )
        
        # Feature-specific validations
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        validation_results["feature_stats"] = {
            "numeric_features": len(numeric_columns),
            "categorical_features": len(categorical_columns),
            "datetime_features": len(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Check for infinite values
        inf_counts = np.isinf(df[numeric_columns]).sum().sum()
        if inf_counts > 0:
            validation_results["warnings"].append(f"Found {inf_counts} infinite values")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["warnings"].append(f"Found {duplicate_count} duplicate rows")
        
        self.logger.info(f"Data validation completed: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings")
        
        return validation_results


class TimeBasedSplitter:
    """Time-based data splitting for financial time series"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def split_data(self, df: pd.DataFrame, time_column: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets based on time
        
        Args:
            df: Input dataframe with time column
            time_column: Name of the timestamp column
            
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        # Ensure time column is datetime
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.sort_values(time_column)
        else:
            self.logger.warning(f"Time column '{time_column}' not found, using row-based split")
            return self._row_based_split(df)
        
        total_samples = len(df)
        
        # Calculate split indices
        train_end_idx = int(total_samples * self.config.train_ratio)
        val_end_idx = int(total_samples * (self.config.train_ratio + self.config.validation_ratio))
        
        # Split the data
        train_df = df.iloc[:train_end_idx].copy()
        validation_df = df.iloc[train_end_idx:val_end_idx].copy()
        test_df = df.iloc[val_end_idx:].copy()
        
        # Log split information
        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/total_samples:.1%})")
        self.logger.info(f"  Validation: {len(validation_df)} samples ({len(validation_df)/total_samples:.1%})")
        self.logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/total_samples:.1%})")
        
        if time_column in df.columns:
            self.logger.info(f"  Train period: {train_df[time_column].min()} to {train_df[time_column].max()}")
            self.logger.info(f"  Validation period: {validation_df[time_column].min()} to {validation_df[time_column].max()}")
            self.logger.info(f"  Test period: {test_df[time_column].min()} to {test_df[time_column].max()}")
        
        return train_df, validation_df, test_df
    
    def _row_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fallback to row-based splitting if no time column"""
        total_samples = len(df)
        
        train_end_idx = int(total_samples * self.config.train_ratio)
        val_end_idx = int(total_samples * (self.config.train_ratio + self.config.validation_ratio))
        
        train_df = df.iloc[:train_end_idx].copy()
        validation_df = df.iloc[train_end_idx:val_end_idx].copy()
        test_df = df.iloc[val_end_idx:].copy()
        
        return train_df, validation_df, test_df


class FeatureProcessor:
    """Feature preprocessing and engineering"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.preprocessing_stats = {}
        
    def process_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process features for all datasets using training data statistics
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
            
        Returns:
            Tuple of processed (train_df, val_df, test_df)
        """
        self.logger.info("Starting feature processing...")
        
        # Handle missing values
        train_df, val_df, test_df = self._handle_missing_values(train_df, val_df, test_df)
        
        # Feature engineering
        if self.config.feature_engineering:
            train_df, val_df, test_df = self._engineer_features(train_df, val_df, test_df)
        
        # Normalization
        if self.config.normalization:
            train_df, val_df, test_df = self._normalize_features(train_df, val_df, test_df)
        
        self.logger.info("Feature processing completed")
        return train_df, val_df, test_df
    
    def _handle_missing_values(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Handle missing values based on configuration"""
        
        if self.config.handle_missing == "median_fill":
            # Calculate medians from training data only
            numeric_columns = train_df.select_dtypes(include=[np.number]).columns
            medians = train_df[numeric_columns].median()
            self.preprocessing_stats['medians'] = medians.to_dict()
            
            # Apply to all datasets
            for df in [train_df, val_df, test_df]:
                df[numeric_columns] = df[numeric_columns].fillna(medians)
        
        elif self.config.handle_missing == "forward_fill":
            # Forward fill for time series data
            for df in [train_df, val_df, test_df]:
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)  # Backward fill remaining
        
        return train_df, val_df, test_df
    
    def _engineer_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Additional feature engineering if needed"""
        
        # Time-based features
        if 'timestamp' in train_df.columns:
            for df in [train_df, val_df, test_df]:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Component interaction features (if not already present)
        component_prefixes = ['c1_', 'c2_', 'c3_', 'c4_', 'c5_', 'c6_', 'c7_', 'c8_']
        
        for prefix in component_prefixes:
            component_cols = [col for col in train_df.columns if col.startswith(prefix)]
            if len(component_cols) > 1:
                # Add component mean and std as features
                for df in [train_df, val_df, test_df]:
                    df[f'{prefix}mean'] = df[component_cols].mean(axis=1)
                    df[f'{prefix}std'] = df[component_cols].std(axis=1)
        
        return train_df, val_df, test_df
    
    def _normalize_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize features using training data statistics"""
        
        numeric_columns = train_df.select_dtypes(include=[np.number]).columns
        
        # Exclude target and identifier columns from normalization
        exclude_columns = ['target', 'timestamp', 'symbol', 'id']
        normalize_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Calculate statistics from training data only
        means = train_df[normalize_columns].mean()
        stds = train_df[normalize_columns].std()
        
        self.preprocessing_stats['normalization'] = {
            'means': means.to_dict(),
            'stds': stds.to_dict()
        }
        
        # Apply normalization to all datasets
        for df in [train_df, val_df, test_df]:
            df[normalize_columns] = (df[normalize_columns] - means) / stds
        
        return train_df, val_df, test_df


class DataExporter:
    """Export processed data to various formats"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.storage_client = storage.Client(project=self.config.project_id)
        
    def export_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                       base_path: str) -> Dict[str, str]:
        """
        Export datasets to specified format and location
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset  
            test_df: Test dataset
            base_path: Base path for output files
            
        Returns:
            Dictionary with output file paths
        """
        output_paths = {}
        
        if self.config.output_format == "parquet":
            output_paths = self._export_parquet(train_df, val_df, test_df, base_path)
        elif self.config.output_format == "tfrecords":
            output_paths = self._export_tfrecords(train_df, val_df, test_df, base_path)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
        
        self.logger.info(f"Datasets exported to {self.config.output_format} format")
        return output_paths
    
    def _export_parquet(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                       base_path: str) -> Dict[str, str]:
        """Export to Parquet format"""
        
        output_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        datasets = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        for split_name, df in datasets.items():
            # Create filename
            filename = f"{split_name}_data_{timestamp}.parquet"
            
            if base_path.startswith('gs://'):
                # Upload to GCS
                gcs_path = f"{base_path}/{filename}"
                self._upload_parquet_to_gcs(df, gcs_path)
                output_paths[f"{split_name}_data"] = gcs_path
            else:
                # Save locally
                local_path = Path(base_path) / filename
                local_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(local_path, index=False)
                output_paths[f"{split_name}_data"] = str(local_path)
        
        return output_paths
    
    def _upload_parquet_to_gcs(self, df: pd.DataFrame, gcs_path: str):
        """Upload dataframe as Parquet to GCS"""
        
        # Parse GCS path
        bucket_name = gcs_path.replace('gs://', '').split('/')[0]
        blob_name = '/'.join(gcs_path.replace('gs://', '').split('/')[1:])
        
        # Convert to Parquet bytes
        table = pa.Table.from_pandas(df)
        
        # Upload to GCS
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
            pq.write_table(table, tmp_file.name)
            blob.upload_from_filename(tmp_file.name)
    
    def _export_tfrecords(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                         base_path: str) -> Dict[str, str]:
        """Export to TFRecords format (placeholder for future implementation)"""
        # TFRecords export would require TensorFlow dependencies
        # For now, fall back to Parquet
        self.logger.warning("TFRecords export not yet implemented, using Parquet")
        return self._export_parquet(train_df, val_df, test_df, base_path)


class BigQueryDataLoader:
    """Load data from BigQuery offline feature tables"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = bigquery.Client(project=self.config.project_id)
        
    def load_training_data(self) -> pd.DataFrame:
        """
        Load training data from BigQuery offline feature tables
        
        Returns:
            DataFrame with complete training dataset
        """
        
        # Construct query for training dataset
        query = f"""
        SELECT *
        FROM `{self.config.source_table}`
        WHERE timestamp IS NOT NULL
        ORDER BY timestamp ASC
        """
        
        self.logger.info(f"Loading data from {self.config.source_table}")
        
        # Execute query and load to DataFrame
        df = self.client.query(query).to_dataframe()
        
        self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Basic data info
        if len(df) > 0:
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Log component feature counts
            component_prefixes = ['c1_', 'c2_', 'c3_', 'c4_', 'c5_', 'c6_', 'c7_', 'c8_']
            for prefix in component_prefixes:
                component_cols = [col for col in df.columns if col.startswith(prefix)]
                if component_cols:
                    self.logger.info(f"Component {prefix}: {len(component_cols)} features")
        
        return df


class DataPreparationPipeline:
    """Main data preparation pipeline orchestrator"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = BigQueryDataLoader(config)
        self.quality_checker = DataQualityChecker(config)
        self.splitter = TimeBasedSplitter(config)
        self.feature_processor = FeatureProcessor(config)
        self.exporter = DataExporter(config)
        
    def run_pipeline(self, output_base_path: str) -> Dict[str, Any]:
        """
        Run the complete data preparation pipeline
        
        Args:
            output_base_path: Base path for output files
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        pipeline_start_time = datetime.now()
        
        try:
            # Step 1: Load data from BigQuery
            self.logger.info("Step 1: Loading data from BigQuery...")
            raw_df = self.data_loader.load_training_data()
            
            # Step 2: Data quality validation
            self.logger.info("Step 2: Validating data quality...")
            quality_results = self.quality_checker.validate_dataset(raw_df)
            
            if quality_results["errors"]:
                raise ValueError(f"Data quality validation failed: {quality_results['errors']}")
            
            # Step 3: Time-based data splitting
            self.logger.info("Step 3: Splitting data...")
            train_df, val_df, test_df = self.splitter.split_data(raw_df)
            
            # Step 4: Feature processing
            self.logger.info("Step 4: Processing features...")
            train_df, val_df, test_df = self.feature_processor.process_features(train_df, val_df, test_df)
            
            # Step 5: Export datasets
            self.logger.info("Step 5: Exporting datasets...")
            output_paths = self.exporter.export_datasets(train_df, val_df, test_df, output_base_path)
            
            # Prepare pipeline results
            pipeline_end_time = datetime.now()
            processing_time = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            results = {
                "status": "success",
                "processing_time_seconds": processing_time,
                "data_splits": {
                    "train_samples": len(train_df),
                    "validation_samples": len(val_df),
                    "test_samples": len(test_df)
                },
                "output_paths": output_paths,
                "data_quality": quality_results,
                "preprocessing_stats": self.feature_processor.preprocessing_stats,
                "pipeline_metadata": {
                    "start_time": pipeline_start_time.isoformat(),
                    "end_time": pipeline_end_time.isoformat(),
                    "config": {
                        "source_table": self.config.source_table,
                        "output_format": self.config.output_format,
                        "train_ratio": self.config.train_ratio,
                        "validation_ratio": self.config.validation_ratio,
                        "test_ratio": self.config.test_ratio
                    }
                }
            }
            
            self.logger.info(f"Data preparation pipeline completed successfully in {processing_time:.1f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Data preparation pipeline failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "processing_time_seconds": (datetime.now() - pipeline_start_time).total_seconds()
            }


# Utility functions for KFP component integration
def create_data_preparation_config(pipeline_config: Dict[str, Any]) -> DataPreparationConfig:
    """Create data preparation configuration from pipeline config"""
    return DataPreparationConfig(pipeline_config)


def run_data_preparation_component(
    project_id: str,
    dataset_table: str,
    staging_bucket: str,
    validation_split: float,
    test_split: float,
    output_format: str
) -> Dict[str, Any]:
    """
    Run data preparation as a KFP component
    
    This function serves as the entry point for the KFP component
    """
    
    # Create minimal config for component execution
    config_dict = {
        'project': {
            'project_id': project_id,
            'location': 'us-central1',
            'staging_bucket': staging_bucket
        },
        'data': {
            'source': {'full_table_id': dataset_table},
            'splits': {
                'train_ratio': 1.0 - validation_split - 0.1,  # Assuming 10% test split
                'validation_ratio': validation_split,
                'test_ratio': 0.1
            },
            'preprocessing': {
                'output_format': output_format,
                'feature_engineering': True,
                'normalization': True,
                'handle_missing': 'median_fill'
            },
            'quality_checks': {
                'min_samples': 1000,
                'max_missing_ratio': 0.1,
                'feature_count_validation': True
            }
        },
        'feature_engineering': {
            'total_features': 774,
            'components': {}
        }
    }
    
    config = DataPreparationConfig(config_dict)
    pipeline = DataPreparationPipeline(config)
    
    # Run pipeline
    output_base_path = f"gs://{staging_bucket}/training_data"
    results = pipeline.run_pipeline(output_base_path)
    
    return results