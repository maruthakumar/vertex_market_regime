"""
Feature Ingestion Pipeline
Handles batch and streaming ingestion from BigQuery to Vertex AI Feature Store
"""

import asyncio
import logging
import time
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud.aiplatform import featurestore
from google.api_core import retry
from google.api_core import exceptions as gcp_exceptions
import pandas as pd
import numpy as np


@dataclass
class IngestionJobConfig:
    """Configuration for feature ingestion job"""
    job_name: str
    source_dataset: str
    source_table: str
    batch_size: int = 10000
    validation_enabled: bool = True
    retry_attempts: int = 3


@dataclass
class IngestionResult:
    """Result of feature ingestion operation"""
    job_name: str
    status: str
    records_processed: int
    batches_processed: int
    execution_time_seconds: float
    error_message: Optional[str] = None
    validation_results: Optional[Dict] = None


class FeatureIngestionPipeline:
    """
    Feature Ingestion Pipeline for Market Regime Feature Store
    Handles batch and streaming ingestion with data validation
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "dev"):
        """Initialize ingestion pipeline"""
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "feature_store_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._initialize_clients()
        
        # Ingestion metrics
        self.metrics = {
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "records_processed": 0,
            "avg_processing_rate": 0.0,
            "validation_failures": 0
        }
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients"""
        project_id = self.config["project_config"]["project_id"]
        location = self.config["project_config"]["location"]
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Initialize clients
        self.bigquery_client = bigquery.Client(project=project_id)
        
        # Get Feature Store references
        self.featurestore_id = self.config["feature_store"]["featurestore_id"]
        self.entity_type_id = self.config["feature_store"]["entity_types"]["instrument_minute"]["entity_type_id"]
        
        try:
            self.featurestore = aiplatform.Featurestore(featurestore_name=self.featurestore_id)
            self.entity_type = self.featurestore.get_entity_type(entity_type_id=self.entity_type_id)
        except Exception as e:
            self.logger.error(f"Failed to connect to Feature Store: {e}")
            raise
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality before ingestion
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results
        """
        results = {
            "validation_passed": True,
            "total_records": len(df),
            "issues": []
        }
        
        # Get expected features
        expected_features = list(self.config["feature_store"]["entity_types"]["instrument_minute"]["online_features"].keys())
        
        # Check for required columns
        required_columns = ["entity_id", "ts_minute"] + expected_features
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results["validation_passed"] = False
            results["issues"].append({
                "type": "missing_columns",
                "columns": missing_columns,
                "severity": "critical"
            })
        
        # Check for null values in critical features
        for feature in expected_features:
            if feature in df.columns:
                null_count = df[feature].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                
                if null_percentage > 50:  # More than 50% null values
                    results["validation_passed"] = False
                    results["issues"].append({
                        "type": "excessive_nulls",
                        "feature": feature,
                        "null_percentage": round(null_percentage, 2),
                        "severity": "high"
                    })
                elif null_percentage > 20:  # More than 20% null values
                    results["issues"].append({
                        "type": "moderate_nulls", 
                        "feature": feature,
                        "null_percentage": round(null_percentage, 2),
                        "severity": "medium"
                    })
        
        # Check for duplicated entity_ids
        if "entity_id" in df.columns:
            duplicates = df["entity_id"].duplicated().sum()
            if duplicates > 0:
                results["validation_passed"] = False
                results["issues"].append({
                    "type": "duplicate_entities",
                    "count": duplicates,
                    "severity": "high"
                })
        
        # Check timestamp validity
        if "ts_minute" in df.columns:
            # Check for future timestamps
            now = datetime.utcnow()
            future_timestamps = df[df["ts_minute"] > now]
            if len(future_timestamps) > 0:
                results["issues"].append({
                    "type": "future_timestamps",
                    "count": len(future_timestamps),
                    "severity": "medium"
                })
            
            # Check for very old timestamps (older than 7 days)
            cutoff = now - timedelta(days=7)
            old_timestamps = df[df["ts_minute"] < cutoff]
            if len(old_timestamps) > len(df) * 0.1:  # More than 10% old data
                results["issues"].append({
                    "type": "old_timestamps",
                    "count": len(old_timestamps),
                    "oldest": old_timestamps["ts_minute"].min().isoformat() if not old_timestamps.empty else None,
                    "severity": "low"
                })
        
        # Check numerical features for outliers
        numerical_features = []
        for feature, config in self.config["feature_store"]["entity_types"]["instrument_minute"]["online_features"].items():
            if config["value_type"] == "DOUBLE" and feature in df.columns:
                numerical_features.append(feature)
        
        for feature in numerical_features:
            if feature in df.columns and not df[feature].empty:
                # Check for extreme outliers using IQR method
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
                outlier_percentage = (len(outliers) / len(df)) * 100
                
                if outlier_percentage > 5:  # More than 5% outliers
                    results["issues"].append({
                        "type": "outliers_detected",
                        "feature": feature,
                        "outlier_percentage": round(outlier_percentage, 2),
                        "severity": "medium"
                    })
        
        # Log validation results
        if results["validation_passed"]:
            self.logger.info(f"Data validation passed for {results['total_records']} records")
        else:
            critical_issues = [issue for issue in results["issues"] if issue["severity"] == "critical"]
            self.logger.error(f"Data validation failed with {len(critical_issues)} critical issues")
        
        return results
    
    async def run_batch_ingestion(self, job_config: IngestionJobConfig) -> IngestionResult:
        """
        Run batch ingestion from BigQuery to Feature Store
        
        Args:
            job_config: Ingestion job configuration
            
        Returns:
            Ingestion result
        """
        start_time = time.time()
        self.metrics["total_jobs"] += 1
        
        result = IngestionResult(
            job_name=job_config.job_name,
            status="started",
            records_processed=0,
            batches_processed=0,
            execution_time_seconds=0
        )
        
        try:
            self.logger.info(f"Starting batch ingestion job: {job_config.job_name}")
            
            # Get online features configuration
            online_features = list(self.config["feature_store"]["entity_types"]["instrument_minute"]["online_features"].keys())
            
            # Build extraction query
            query = self._build_ingestion_query(
                job_config.source_dataset,
                job_config.source_table,
                online_features
            )
            
            self.logger.info(f"Extracting data from {job_config.source_dataset}.{job_config.source_table}")
            
            # Execute BigQuery extraction
            query_job = self.bigquery_client.query(query)
            df = query_job.to_dataframe()
            
            if df.empty:
                result.status = "no_data"
                result.execution_time_seconds = time.time() - start_time
                self.logger.warning("No data found for ingestion")
                return result
            
            self.logger.info(f"Extracted {len(df)} records from BigQuery")
            
            # Data validation
            if job_config.validation_enabled:
                validation_results = self.validate_data_quality(df)
                result.validation_results = validation_results
                
                if not validation_results["validation_passed"]:
                    result.status = "validation_failed"
                    result.error_message = f"Data validation failed with {len(validation_results['issues'])} issues"
                    result.execution_time_seconds = time.time() - start_time
                    self.metrics["failed_jobs"] += 1
                    self.metrics["validation_failures"] += 1
                    return result
            
            # Process data in batches
            total_records = len(df)
            batches_processed = 0
            records_processed = 0
            
            for i in range(0, total_records, job_config.batch_size):
                batch_df = df.iloc[i:i + job_config.batch_size]
                
                batch_success = await self._process_batch(
                    batch_df,
                    online_features,
                    job_config.retry_attempts
                )
                
                if batch_success:
                    batches_processed += 1
                    records_processed += len(batch_df)
                    
                    self.logger.info(f"Processed batch {batches_processed}: {len(batch_df)} records")
                else:
                    self.logger.error(f"Failed to process batch {batches_processed + 1}")
            
            # Update result
            execution_time = time.time() - start_time
            
            result.status = "completed" if records_processed > 0 else "failed"
            result.records_processed = records_processed
            result.batches_processed = batches_processed
            result.execution_time_seconds = round(execution_time, 2)
            
            # Update metrics
            if result.status == "completed":
                self.metrics["successful_jobs"] += 1
                self.metrics["records_processed"] += records_processed
                
                # Calculate processing rate
                if execution_time > 0:
                    rate = records_processed / execution_time
                    self.metrics["avg_processing_rate"] = (
                        (self.metrics["avg_processing_rate"] * (self.metrics["successful_jobs"] - 1) + rate) /
                        self.metrics["successful_jobs"]
                    )
            else:
                self.metrics["failed_jobs"] += 1
            
            self.logger.info(f"Batch ingestion completed: {records_processed} records in {execution_time:.2f}s")
            
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            result.execution_time_seconds = time.time() - start_time
            self.metrics["failed_jobs"] += 1
            self.logger.error(f"Batch ingestion failed: {e}")
            
        return result
    
    def _build_ingestion_query(
        self, 
        dataset_id: str, 
        table_name: str, 
        features: List[str],
        lookback_hours: int = 2
    ) -> str:
        """Build BigQuery extraction query for ingestion"""
        
        # Feature columns
        feature_columns = ', '.join(features)
        
        query = f"""
        WITH latest_features AS (
            SELECT 
                CONCAT(symbol, '_', FORMAT_TIMESTAMP('%Y%m%d%H%M', ts_minute), '_', CAST(dte AS STRING)) as entity_id,
                ts_minute as feature_timestamp,
                {feature_columns},
                symbol,
                dte,
                zone_name,
                -- Add row number for deduplication
                ROW_NUMBER() OVER (
                    PARTITION BY symbol, ts_minute, dte 
                    ORDER BY ts_minute DESC
                ) as rn
            FROM `{self.config["project_config"]["project_id"]}.{dataset_id}.{table_name}`
            WHERE ts_minute >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_hours} HOUR)
                AND ts_minute <= CURRENT_TIMESTAMP()
        )
        SELECT *
        FROM latest_features
        WHERE rn = 1  -- Remove duplicates
        ORDER BY feature_timestamp DESC
        """
        
        return query
    
    async def _process_batch(
        self, 
        batch_df: pd.DataFrame, 
        features: List[str], 
        retry_attempts: int
    ) -> bool:
        """Process a single batch with retry logic"""
        
        for attempt in range(retry_attempts):
            try:
                # Prepare batch data for Feature Store
                ingestion_data = self._prepare_feature_data(batch_df, features)
                
                if not ingestion_data:
                    self.logger.warning("No valid data in batch after preparation")
                    return False
                
                # Create temporary BigQuery table for batch ingestion
                temp_table_id = f"temp_ingestion_{int(time.time())}"
                temp_table_ref = self.bigquery_client.dataset(f"market_regime_{self.environment}").table(temp_table_id)
                
                # Upload data to temporary table
                job = self.bigquery_client.load_table_from_dataframe(batch_df, temp_table_ref)
                job.result()  # Wait for job completion
                
                # Use Feature Store batch ingestion API
                feature_configs = {}
                for feature_name in features:
                    feature_config = self.config["feature_store"]["entity_types"]["instrument_minute"]["online_features"][feature_name]
                    feature_configs[feature_name] = {"value_type": feature_config["value_type"]}
                
                # Start batch ingestion job
                ingestion_job = self.entity_type.batch_create_features(
                    feature_configs=feature_configs,
                    bigquery_source_uri=f"bq://{self.config['project_config']['project_id']}.market_regime_{self.environment}.{temp_table_id}",
                    entity_id_field="entity_id",
                    feature_time_field="feature_timestamp",
                    disable_online_serving=False
                )
                
                # Wait for ingestion to complete
                ingestion_job.wait()
                
                # Clean up temporary table
                self.bigquery_client.delete_table(temp_table_ref)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Batch processing attempt {attempt + 1} failed: {e}")
                if attempt == retry_attempts - 1:
                    return False
                
                # Exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        return False
    
    def _prepare_feature_data(self, df: pd.DataFrame, features: List[str]) -> List[Dict]:
        """Prepare feature data for ingestion"""
        ingestion_data = []
        
        for _, row in df.iterrows():
            # Skip rows with invalid entity_id or timestamp
            if pd.isna(row.get("entity_id")) or pd.isna(row.get("feature_timestamp")):
                continue
            
            entity_data = {
                "entity_id": str(row["entity_id"]),
                "feature_timestamp": row["feature_timestamp"]
            }
            
            # Add feature values (only non-null values)
            for feature_name in features:
                if feature_name in row and pd.notna(row[feature_name]):
                    value = row[feature_name]
                    
                    # Convert numpy types to Python types
                    if isinstance(value, np.integer):
                        value = int(value)
                    elif isinstance(value, np.floating):
                        value = float(value)
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    
                    entity_data[feature_name] = value
            
            ingestion_data.append(entity_data)
        
        return ingestion_data
    
    async def run_streaming_ingestion(
        self, 
        dataset_id: str,
        monitoring_duration_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Run streaming ingestion for real-time feature updates
        
        Args:
            dataset_id: Source BigQuery dataset
            monitoring_duration_seconds: How long to run streaming
            
        Returns:
            Streaming ingestion results
        """
        start_time = time.time()
        results = {
            "status": "started",
            "streaming_duration_seconds": monitoring_duration_seconds,
            "updates_processed": 0,
            "last_update_timestamp": None
        }
        
        try:
            self.logger.info(f"Starting streaming ingestion for {monitoring_duration_seconds} seconds")
            
            last_check_timestamp = datetime.utcnow() - timedelta(minutes=5)
            updates_processed = 0
            
            # Monitor for new data every 30 seconds
            while (time.time() - start_time) < monitoring_duration_seconds:
                # Check for new data since last check
                check_timestamp = datetime.utcnow()
                
                new_data_query = f"""
                SELECT COUNT(*) as new_records
                FROM `{self.config["project_config"]["project_id"]}.{dataset_id}.training_dataset`
                WHERE ts_minute > TIMESTAMP('{last_check_timestamp.isoformat()}')
                    AND ts_minute <= TIMESTAMP('{check_timestamp.isoformat()}')
                """
                
                query_job = self.bigquery_client.query(new_data_query)
                result = query_job.result()
                
                for row in result:
                    new_records = row.new_records
                    
                    if new_records > 0:
                        self.logger.info(f"Found {new_records} new records, starting ingestion")
                        
                        # Create ingestion job for new data
                        job_config = IngestionJobConfig(
                            job_name=f"streaming_update_{int(time.time())}",
                            source_dataset=dataset_id,
                            source_table="training_dataset",
                            batch_size=min(new_records, 1000)  # Smaller batches for streaming
                        )
                        
                        # Run batch ingestion for new data
                        ingestion_result = await self.run_batch_ingestion(job_config)
                        
                        if ingestion_result.status == "completed":
                            updates_processed += ingestion_result.records_processed
                            results["last_update_timestamp"] = check_timestamp.isoformat()
                
                last_check_timestamp = check_timestamp
                
                # Wait 30 seconds before next check
                await asyncio.sleep(30)
            
            # Final results
            execution_time = time.time() - start_time
            results.update({
                "status": "completed",
                "updates_processed": updates_processed,
                "execution_time_seconds": round(execution_time, 2),
                "average_update_rate": round(updates_processed / execution_time, 2) if execution_time > 0 else 0
            })
            
            self.logger.info(f"Streaming ingestion completed: {updates_processed} updates in {execution_time:.2f}s")
            
        except Exception as e:
            results.update({
                "status": "error", 
                "error": str(e),
                "execution_time_seconds": time.time() - start_time
            })
            self.logger.error(f"Streaming ingestion failed: {e}")
        
        return results
    
    def get_ingestion_metrics(self) -> Dict[str, Any]:
        """Get ingestion pipeline metrics"""
        return {
            "total_jobs": self.metrics["total_jobs"],
            "successful_jobs": self.metrics["successful_jobs"], 
            "failed_jobs": self.metrics["failed_jobs"],
            "success_rate": round(self.metrics["successful_jobs"] / max(self.metrics["total_jobs"], 1), 3),
            "records_processed": self.metrics["records_processed"],
            "avg_processing_rate_records_per_second": round(self.metrics["avg_processing_rate"], 2),
            "validation_failures": self.metrics["validation_failures"]
        }


# Example usage
async def main():
    """Example usage of Feature Ingestion Pipeline"""
    
    # Initialize pipeline
    pipeline = FeatureIngestionPipeline(environment="dev")
    
    # Create batch ingestion job
    job_config = IngestionJobConfig(
        job_name="test_batch_ingestion",
        source_dataset="market_regime_dev",
        source_table="training_dataset",
        batch_size=5000,
        validation_enabled=True
    )
    
    # Run batch ingestion
    print("Running batch ingestion...")
    result = await pipeline.run_batch_ingestion(job_config)
    print(f"Batch Ingestion: {result.status}")
    print(f"Records Processed: {result.records_processed}")
    
    # Run streaming ingestion test
    print("Running streaming ingestion test...")
    streaming_result = await pipeline.run_streaming_ingestion("market_regime_dev", 60)
    print(f"Streaming: {streaming_result['status']}")
    print(f"Updates: {streaming_result['updates_processed']}")
    
    # Get metrics
    metrics = pipeline.get_ingestion_metrics()
    print(f"Pipeline Metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())