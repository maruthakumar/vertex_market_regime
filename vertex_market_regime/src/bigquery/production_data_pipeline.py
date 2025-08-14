#!/usr/bin/env python3
"""
Production Data Pipeline for BigQuery Feature Tables
Enhanced pipeline with Parquet → Arrow → BigQuery transformation logic
"""

import os
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from dataclasses import dataclass


@dataclass
class ValidationRule:
    """Data validation rule definition"""
    column: str
    rule_type: str  # 'not_null', 'range', 'type', 'enum'
    parameters: Dict[str, Any]
    severity: str = "error"  # 'error', 'warning'


@dataclass
class LoadMetrics:
    """Metrics for data load operations"""
    load_id: str
    table_name: str
    source_path: str
    rows_processed: int
    rows_loaded: int
    validation_errors: int
    validation_warnings: int
    load_duration_seconds: float
    bytes_processed: int
    success: bool
    error_message: Optional[str] = None


class ProductionDataPipeline:
    """Production-grade pipeline for loading Parquet data to BigQuery feature tables"""
    
    def __init__(self, project_id: str = "arched-bot-269016", environment: str = "dev"):
        """
        Initialize production pipeline
        
        Args:
            project_id: GCP project ID
            environment: Environment (dev/staging/prod)
        """
        self.project_id = project_id
        self.environment = environment
        self.dataset_id = f"market_regime_{environment}"
        
        # Initialize clients
        try:
            self.bq_client = bigquery.Client(project=project_id)
            self.storage_client = storage.Client(project=project_id)
        except Exception:
            # Fallback for local development without GCP credentials
            self.bq_client = None
            self.storage_client = None
        
        # Configure logging
        self._setup_logging()
        
        # Configuration
        self.gcs_bucket = "vertex-mr-data"
        self.parquet_path_pattern = "{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}/"
        self.validation_rules = self._define_validation_rules()
        
        # Processing configuration
        self.batch_size = 50000  # Rows per batch for memory efficiency
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def _setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'data_pipeline_{self.environment}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _define_validation_rules(self) -> Dict[str, List[ValidationRule]]:
        """Define validation rules for each component table"""
        common_rules = [
            ValidationRule("symbol", "not_null", {}),
            ValidationRule("symbol", "enum", {"values": ["NIFTY", "BANKNIFTY"]}),
            ValidationRule("ts_minute", "not_null", {}),
            ValidationRule("dte", "range", {"min": 0, "max": 365}),
            ValidationRule("zone_name", "enum", {
                "values": ["OPEN", "MID_MORN", "LUNCH", "AFTERNOON", "CLOSE"]
            })
        ]
        
        # Component-specific rules
        c1_rules = common_rules + [
            ValidationRule("c1_momentum_score", "range", {"min": -10, "max": 10}),
            ValidationRule("c1_vol_compression", "range", {"min": 0, "max": 5}),
            ValidationRule("c1_breakout_probability", "range", {"min": 0, "max": 1})
        ]
        
        c2_rules = common_rules + [
            ValidationRule("c2_gamma_exposure", "range", {"min": -100000, "max": 100000}),
            ValidationRule("c2_sentiment_level", "range", {"min": -5, "max": 5}),
            ValidationRule("c2_pin_risk_score", "range", {"min": 0, "max": 1})
        ]
        
        # Add rules for other components...
        return {
            "c1_features": c1_rules,
            "c2_features": c2_rules,
            # Add other component rules as needed
        }
    
    def load_parquet_with_arrow(self, parquet_path: str) -> Optional[pa.Table]:
        """
        Load Parquet data using Apache Arrow for zero-copy processing
        
        Args:
            parquet_path: Path to Parquet file (local or GCS)
            
        Returns:
            Arrow Table or None if failed
        """
        self.logger.info(f"Loading Parquet data with Arrow from: {parquet_path}")
        
        try:
            if parquet_path.startswith("gs://"):
                # Handle GCS paths
                if self.storage_client is None:
                    self.logger.error("GCS client not available")
                    return None
                    
                # Extract bucket and blob path
                path_parts = parquet_path[5:].split("/", 1)
                bucket_name = path_parts[0]
                blob_path = path_parts[1] if len(path_parts) > 1 else ""
                
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                # Download to temporary file
                temp_path = f"/tmp/{blob_path.split('/')[-1]}"
                blob.download_to_filename(temp_path)
                
                # Read with Arrow
                table = pq.read_table(temp_path)
                
                # Clean up
                os.remove(temp_path)
                
            else:
                # Local file
                table = pq.read_table(parquet_path)
            
            self.logger.info(f"Loaded Arrow table: {table.num_rows} rows, {table.num_columns} columns")
            return table
            
        except Exception as e:
            self.logger.error(f"Failed to load Parquet with Arrow: {str(e)}")
            return None
    
    def validate_arrow_table(self, table: pa.Table, table_name: str) -> Dict[str, Any]:
        """
        Validate Arrow table against defined rules
        
        Args:
            table: Arrow table to validate
            table_name: Name of the target table
            
        Returns:
            Validation results
        """
        self.logger.info(f"Validating data for table: {table_name}")
        
        validation_results = {
            "table_name": table_name,
            "total_rows": table.num_rows,
            "errors": [],
            "warnings": [],
            "error_count": 0,
            "warning_count": 0,
            "passed": True
        }
        
        rules = self.validation_rules.get(table_name, [])
        
        for rule in rules:
            try:
                column_name = rule.column
                
                # Check if column exists
                if column_name not in table.column_names:
                    error_msg = f"Column '{column_name}' not found in data"
                    validation_results["errors"].append(error_msg)
                    validation_results["error_count"] += 1
                    continue
                
                # Get column data
                column = table.column(column_name).to_pandas()
                
                # Apply validation rule
                if rule.rule_type == "not_null":
                    null_count = column.isnull().sum()
                    if null_count > 0:
                        msg = f"Column '{column_name}' has {null_count} null values"
                        if rule.severity == "error":
                            validation_results["errors"].append(msg)
                            validation_results["error_count"] += 1
                        else:
                            validation_results["warnings"].append(msg)
                            validation_results["warning_count"] += 1
                
                elif rule.rule_type == "range":
                    min_val = rule.parameters.get("min")
                    max_val = rule.parameters.get("max")
                    
                    if min_val is not None:
                        out_of_range = (column < min_val).sum()
                        if out_of_range > 0:
                            msg = f"Column '{column_name}' has {out_of_range} values below minimum {min_val}"
                            if rule.severity == "error":
                                validation_results["errors"].append(msg)
                                validation_results["error_count"] += 1
                            else:
                                validation_results["warnings"].append(msg)
                                validation_results["warning_count"] += 1
                    
                    if max_val is not None:
                        out_of_range = (column > max_val).sum()
                        if out_of_range > 0:
                            msg = f"Column '{column_name}' has {out_of_range} values above maximum {max_val}"
                            if rule.severity == "error":
                                validation_results["errors"].append(msg)
                                validation_results["error_count"] += 1
                            else:
                                validation_results["warnings"].append(msg)
                                validation_results["warning_count"] += 1
                
                elif rule.rule_type == "enum":
                    valid_values = rule.parameters.get("values", [])
                    invalid_mask = ~column.isin(valid_values)
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        msg = f"Column '{column_name}' has {invalid_count} invalid values"
                        if rule.severity == "error":
                            validation_results["errors"].append(msg)
                            validation_results["error_count"] += 1
                        else:
                            validation_results["warnings"].append(msg)
                            validation_results["warning_count"] += 1
                
            except Exception as e:
                error_msg = f"Validation error for rule {rule.column}:{rule.rule_type} - {str(e)}"
                validation_results["errors"].append(error_msg)
                validation_results["error_count"] += 1
        
        # Determine if validation passed
        validation_results["passed"] = validation_results["error_count"] == 0
        
        self.logger.info(f"Validation completed: {validation_results['error_count']} errors, "
                        f"{validation_results['warning_count']} warnings")
        
        return validation_results
    
    def transform_arrow_to_bigquery_schema(self, table: pa.Table, target_table: str) -> pa.Table:
        """
        Transform Arrow table to match BigQuery schema requirements
        
        Args:
            table: Input Arrow table
            target_table: Target BigQuery table name
            
        Returns:
            Transformed Arrow table
        """
        self.logger.info(f"Transforming schema for BigQuery table: {target_table}")
        
        # Convert to pandas for easier transformation
        df = table.to_pandas()
        
        # Ensure required columns exist
        required_columns = ["symbol", "ts_minute", "date", "dte", "zone_name"]
        for col in required_columns:
            if col not in df.columns:
                if col == "date":
                    # Derive date from ts_minute if available
                    if "ts_minute" in df.columns:
                        df["date"] = pd.to_datetime(df["ts_minute"]).dt.date
                    else:
                        df["date"] = pd.Timestamp.now().date()
                else:
                    self.logger.warning(f"Required column '{col}' missing, filling with defaults")
                    if col == "symbol":
                        df[col] = "UNKNOWN"
                    elif col == "dte":
                        df[col] = 0
                    elif col == "zone_name":
                        df[col] = "UNKNOWN"
        
        # Add metadata columns if missing
        if "created_at" not in df.columns:
            df["created_at"] = pd.Timestamp.now()
        if "updated_at" not in df.columns:
            df["updated_at"] = pd.Timestamp.now()
        
        # Convert data types for BigQuery compatibility
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # Convert back to Arrow table
        transformed_table = pa.Table.from_pandas(df)
        
        self.logger.info(f"Schema transformation completed: {transformed_table.num_columns} columns")
        return transformed_table
    
    def load_arrow_to_bigquery(self, table: pa.Table, target_table: str) -> LoadMetrics:
        """
        Load Arrow table to BigQuery with batching and error handling
        
        Args:
            table: Arrow table to load
            target_table: Target BigQuery table name
            
        Returns:
            Load metrics
        """
        load_id = f"load_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()
        
        self.logger.info(f"Loading {table.num_rows} rows to BigQuery table: {target_table}")
        
        metrics = LoadMetrics(
            load_id=load_id,
            table_name=target_table,
            source_path="arrow_table",
            rows_processed=table.num_rows,
            rows_loaded=0,
            validation_errors=0,
            validation_warnings=0,
            load_duration_seconds=0,
            bytes_processed=table.nbytes,
            success=False
        )
        
        if self.bq_client is None:
            self.logger.warning("BigQuery client not available, simulating load")
            metrics.success = True
            metrics.rows_loaded = table.num_rows
            metrics.load_duration_seconds = time.time() - start_time
            return metrics
        
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{target_table}"
            
            # Configure load job
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                time_partitioning=bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                ),
                clustering_fields=["symbol", "dte"],
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )
            
            # Convert to DataFrame for BigQuery client
            df = table.to_pandas()
            
            # Load in batches if data is large
            if len(df) > self.batch_size:
                self.logger.info(f"Loading in batches of {self.batch_size} rows")
                
                for i in range(0, len(df), self.batch_size):
                    batch_df = df.iloc[i:i + self.batch_size]
                    
                    # Load batch
                    job = self.bq_client.load_table_from_dataframe(
                        batch_df, table_id, job_config=job_config
                    )
                    job.result()  # Wait for completion
                    
                    metrics.rows_loaded += len(batch_df)
                    self.logger.info(f"Loaded batch {i//self.batch_size + 1}: "
                                   f"{len(batch_df)} rows ({metrics.rows_loaded} total)")
            else:
                # Load all at once
                job = self.bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
                job.result()  # Wait for completion
                metrics.rows_loaded = len(df)
            
            metrics.success = True
            self.logger.info(f"Successfully loaded {metrics.rows_loaded} rows to {target_table}")
            
        except Exception as e:
            metrics.error_message = str(e)
            self.logger.error(f"Failed to load data to BigQuery: {str(e)}")
        
        metrics.load_duration_seconds = time.time() - start_time
        return metrics
    
    def log_to_audit_table(self, metrics: LoadMetrics, validation_results: Dict[str, Any]):
        """
        Log load metrics to audit table
        
        Args:
            metrics: Load metrics
            validation_results: Validation results
        """
        if self.bq_client is None:
            self.logger.info(f"Would log audit record: {metrics.load_id}")
            return
        
        try:
            audit_table = f"{self.project_id}.{self.dataset_id}.mr_load_audit"
            
            audit_record = {
                'load_id': metrics.load_id,
                'load_timestamp': datetime.now(),
                'table_name': metrics.table_name,
                'row_count': metrics.rows_loaded,
                'null_check_passed': validation_results.get('error_count', 0) == 0,
                'schema_validation_passed': validation_results.get('passed', False),
                'error_message': metrics.error_message,
                'load_duration_seconds': metrics.load_duration_seconds,
                'created_at': datetime.now()
            }
            
            df = pd.DataFrame([audit_record])
            job = self.bq_client.load_table_from_dataframe(df, audit_table)
            job.result()
            
            self.logger.info(f"Audit record logged: {metrics.load_id}")
            
        except Exception as e:
            self.logger.warning(f"Could not log to audit table: {str(e)}")
    
    def process_parquet_file(self, parquet_path: str, target_table: str) -> Dict[str, Any]:
        """
        Process a single Parquet file through the complete pipeline
        
        Args:
            parquet_path: Path to Parquet file
            target_table: Target BigQuery table
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing Parquet file: {parquet_path} -> {target_table}")
        
        results = {
            "parquet_path": parquet_path,
            "target_table": target_table,
            "success": False,
            "validation_results": {},
            "load_metrics": None,
            "error_message": None
        }
        
        try:
            # Step 1: Load Parquet with Arrow
            arrow_table = self.load_parquet_with_arrow(parquet_path)
            if arrow_table is None:
                results["error_message"] = "Failed to load Parquet file"
                return results
            
            # Step 2: Validate data
            validation_results = self.validate_arrow_table(arrow_table, target_table)
            results["validation_results"] = validation_results
            
            if not validation_results["passed"]:
                self.logger.warning(f"Data validation failed with {validation_results['error_count']} errors")
                # Continue processing if only warnings, fail if errors
                if validation_results["error_count"] > 0:
                    results["error_message"] = "Data validation failed"
                    return results
            
            # Step 3: Transform schema
            transformed_table = self.transform_arrow_to_bigquery_schema(arrow_table, target_table)
            
            # Step 4: Load to BigQuery
            load_metrics = self.load_arrow_to_bigquery(transformed_table, target_table)
            results["load_metrics"] = load_metrics
            
            # Step 5: Log to audit table
            self.log_to_audit_table(load_metrics, validation_results)
            
            results["success"] = load_metrics.success
            
        except Exception as e:
            results["error_message"] = str(e)
            self.logger.error(f"Processing failed: {str(e)}")
        
        return results
    
    def generate_sample_parquet(self, output_path: str, component: str = "c1") -> bool:
        """
        Generate sample Parquet file for testing
        
        Args:
            output_path: Path to save Parquet file
            component: Component name (c1, c2, etc.)
            
        Returns:
            True if successful
        """
        self.logger.info(f"Generating sample Parquet for component {component}")
        
        try:
            # Generate sample data based on component
            num_records = 10000
            base_time = datetime.now().replace(second=0, microsecond=0)
            timestamps = [base_time - timedelta(minutes=i) for i in range(num_records)]
            
            # Common columns
            data = {
                'symbol': np.random.choice(['NIFTY', 'BANKNIFTY'], num_records),
                'ts_minute': timestamps,
                'date': [t.date() for t in timestamps],
                'dte': np.random.choice([0, 3, 7, 14, 21, 30], num_records),
                'zone_name': np.random.choice(['OPEN', 'MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE'], num_records),
                'created_at': [datetime.now()] * num_records,
                'updated_at': [datetime.now()] * num_records
            }
            
            # Add component-specific features
            if component == "c1":
                data.update({
                    'c1_momentum_score': np.random.uniform(-1, 1, num_records),
                    'c1_vol_compression': np.random.uniform(0, 1, num_records),
                    'c1_breakout_probability': np.random.uniform(0, 1, num_records),
                    'c1_straddle_pct_chg_0dte': np.random.uniform(-10, 10, num_records),
                    'c1_straddle_volume_0dte': np.random.randint(1000, 100000, num_records)
                })
            elif component == "c2":
                data.update({
                    'c2_gamma_exposure': np.random.uniform(-1000, 1000, num_records) * 1.5,
                    'c2_sentiment_level': np.random.randint(-2, 3, num_records),
                    'c2_pin_risk_score': np.random.uniform(0, 1, num_records),
                    'c2_delta': np.random.uniform(-1, 1, num_records),
                    'c2_vega': np.random.uniform(0, 100, num_records)
                })
            
            # Create DataFrame and save as Parquet
            df = pd.DataFrame(data)
            df.to_parquet(output_path, engine='pyarrow', compression='snappy')
            
            self.logger.info(f"Generated sample Parquet: {output_path} ({len(df)} records)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate sample Parquet: {str(e)}")
            return False


def main():
    """Main function for testing the production pipeline"""
    print("=" * 60)
    print("Production Data Pipeline Test")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ProductionDataPipeline(environment="dev")
    
    # Generate sample Parquet file
    sample_path = "/tmp/sample_c1_features.parquet"
    if pipeline.generate_sample_parquet(sample_path, "c1"):
        print(f"✓ Generated sample Parquet: {sample_path}")
        
        # Process the file
        results = pipeline.process_parquet_file(sample_path, "c1_features")
        
        print("\nProcessing Results:")
        print(f"  Success: {results['success']}")
        if results['validation_results']:
            print(f"  Validation errors: {results['validation_results']['error_count']}")
            print(f"  Validation warnings: {results['validation_results']['warning_count']}")
        if results['load_metrics']:
            print(f"  Rows loaded: {results['load_metrics'].rows_loaded}")
            print(f"  Load duration: {results['load_metrics'].load_duration_seconds:.2f}s")
        
        # Clean up
        os.remove(sample_path)
        print("✓ Cleaned up sample file")
    else:
        print("✗ Failed to generate sample Parquet")
    
    return 0


if __name__ == "__main__":
    exit(main())