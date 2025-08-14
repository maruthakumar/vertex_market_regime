#!/usr/bin/env python3
"""
Sample Data Pipeline for BigQuery Feature Tables
Loads sample Parquet data from GCS to BigQuery for smoke testing
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError


class SampleDataPipeline:
    """Pipeline to load sample data from Parquet to BigQuery"""
    
    def __init__(self, project_id: str = "arched-bot-269016", environment: str = "dev"):
        """
        Initialize pipeline
        
        Args:
            project_id: GCP project ID
            environment: Environment (dev/staging/prod)
        """
        self.project_id = project_id
        self.environment = environment
        self.dataset_id = f"market_regime_{environment}"
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # GCS paths (from tech stack doc)
        self.gcs_bucket = "vertex-mr-data"
        self.parquet_path_pattern = "{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}/"
    
    def generate_sample_data(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate sample feature data for testing
        
        Args:
            num_records: Number of sample records to generate
            
        Returns:
            DataFrame with sample features
        """
        self.logger.info(f"Generating {num_records} sample records...")
        
        # Generate timestamps (every minute)
        base_time = datetime.now().replace(second=0, microsecond=0)
        timestamps = [base_time - timedelta(minutes=i) for i in range(num_records)]
        
        # Generate sample data
        data = {
            # Common columns
            'symbol': np.random.choice(['NIFTY', 'BANKNIFTY'], num_records),
            'ts_minute': timestamps,
            'date': [t.date() for t in timestamps],
            'dte': np.random.choice([0, 3, 7, 14, 21, 30], num_records),
            'zone_name': np.random.choice(['OPEN', 'MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE'], num_records),
            
            # Component 1 features (sample)
            'c1_momentum_score': np.random.uniform(-1, 1, num_records),
            'c1_vol_compression': np.random.uniform(0, 1, num_records),
            'c1_breakout_probability': np.random.uniform(0, 1, num_records),
            'c1_straddle_pct_chg_0dte': np.random.uniform(-10, 10, num_records),
            'c1_straddle_volume_0dte': np.random.randint(1000, 100000, num_records),
            
            # Component 2 features (sample)
            'c2_gamma_exposure': np.random.uniform(-1000, 1000, num_records) * 1.5,  # gamma weight = 1.5
            'c2_sentiment_level': np.random.randint(-2, 3, num_records),
            'c2_pin_risk_score': np.random.uniform(0, 1, num_records),
            'c2_delta': np.random.uniform(-1, 1, num_records),
            'c2_vega': np.random.uniform(0, 100, num_records),
            
            # Component 3 features (sample)
            'c3_institutional_flow_score': np.random.uniform(-1, 1, num_records),
            'c3_divergence_type': np.random.randint(0, 4, num_records),
            'c3_range_expansion_score': np.random.uniform(0, 1, num_records),
            'c3_oi_total': np.random.randint(10000, 1000000, num_records),
            'c3_oi_put_call_ratio': np.random.uniform(0.5, 2.0, num_records),
            
            # Component 4 features (sample)
            'c4_skew_bias_score': np.random.uniform(-1, 1, num_records),
            'c4_term_structure_signal': np.random.randint(-1, 2, num_records),
            'c4_iv_regime_level': np.random.randint(0, 4, num_records),
            'c4_iv_atm': np.random.uniform(10, 30, num_records),
            'c4_iv_percentile_20d': np.random.uniform(0, 100, num_records),
            
            # Component 5 features (sample)
            'c5_momentum_score': np.random.uniform(-1, 1, num_records),
            'c5_volatility_regime_score': np.random.randint(0, 3, num_records),
            'c5_confluence_score': np.random.uniform(0, 1, num_records),
            'c5_atr_15min': np.random.uniform(10, 100, num_records),
            'c5_ema_21': np.random.uniform(15000, 20000, num_records),
            
            # Component 6 features (sample - including correlation matrix)
            'c6_correlation_agreement_score': np.random.uniform(0, 1, num_records),
            'c6_breakdown_alert': np.random.randint(0, 2, num_records),
            'c6_system_stability_score': np.random.uniform(0, 1, num_records),
            'c6_corr_matrix_1': np.random.uniform(-1, 1, num_records),
            'c6_corr_nifty_banknifty': np.random.uniform(0, 1, num_records),
            
            # Component 7 features (sample)
            'c7_level_strength_score': np.random.uniform(0, 1, num_records),
            'c7_breakout_probability': np.random.uniform(0, 1, num_records),
            'c7_support_1': np.random.uniform(15000, 18000, num_records),
            'c7_resistance_1': np.random.uniform(18000, 20000, num_records),
            
            # Component 8 features (sample)
            'c8_component_agreement_score': np.random.uniform(0, 1, num_records),
            'c8_integration_confidence': np.random.uniform(0, 1, num_records),
            'c8_transition_probability_hint': np.random.uniform(0, 1, num_records),
            'c8_regime_classification': np.random.randint(1, 5, num_records),
            'c8_weight_c1': np.random.uniform(0, 0.2, num_records),
            
            # Metadata
            'created_at': [datetime.now()] * num_records,
            'updated_at': [datetime.now()] * num_records
        }
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} sample records")
        return df
    
    def load_parquet_from_gcs(self, symbol: str = "NIFTY", date: datetime = None) -> Optional[pd.DataFrame]:
        """
        Load sample Parquet data from GCS
        
        Args:
            symbol: Trading symbol
            date: Date to load data for
            
        Returns:
            DataFrame with Parquet data or None if not found
        """
        if date is None:
            date = datetime.now()
        
        # Construct GCS path
        gcs_path = self.parquet_path_pattern.format(
            symbol=symbol,
            year=date.year,
            month=date.month,
            day=date.day,
            hour=date.hour
        )
        
        full_path = f"gs://{self.gcs_bucket}/{gcs_path}"
        self.logger.info(f"Attempting to load from: {full_path}")
        
        try:
            # List blobs in the path
            bucket = self.storage_client.bucket(self.gcs_bucket)
            blobs = list(bucket.list_blobs(prefix=gcs_path))
            
            if not blobs:
                self.logger.warning(f"No Parquet files found in {full_path}")
                return None
            
            # Read first Parquet file found
            parquet_blob = blobs[0]
            self.logger.info(f"Reading: {parquet_blob.name}")
            
            # Download to temp file and read
            temp_path = f"/tmp/{parquet_blob.name.split('/')[-1]}"
            parquet_blob.download_to_filename(temp_path)
            
            df = pd.read_parquet(temp_path)
            self.logger.info(f"Loaded {len(df)} records from Parquet")
            
            # Clean up temp file
            os.remove(temp_path)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Parquet from GCS: {str(e)}")
            return None
    
    def create_bigquery_tables(self) -> bool:
        """
        Create BigQuery tables from DDL files
        
        Returns:
            True if successful
        """
        self.logger.info("Creating BigQuery tables...")
        
        # Create dataset if not exists
        dataset_id = f"{self.project_id}.{self.dataset_id}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        dataset.description = f"Market Regime feature tables ({self.environment})"
        
        try:
            dataset = self.bq_client.create_dataset(dataset, exists_ok=True)
            self.logger.info(f"Dataset {dataset_id} ready")
        except Exception as e:
            self.logger.error(f"Error creating dataset: {str(e)}")
            return False
        
        # Read and execute DDL files
        ddl_dir = Path(__file__).parent / "ddl"
        ddl_files = list(ddl_dir.glob("*.sql"))
        
        for ddl_file in ddl_files:
            self.logger.info(f"Creating table from: {ddl_file.name}")
            
            with open(ddl_file, 'r') as f:
                ddl_content = f.read()
            
            # Replace environment placeholder
            ddl_content = ddl_content.replace("{env}", self.environment)
            
            try:
                # Execute DDL
                query_job = self.bq_client.query(ddl_content)
                query_job.result()  # Wait for completion
                self.logger.info(f"  ✓ Created table from {ddl_file.name}")
            except Exception as e:
                self.logger.error(f"  ✗ Error creating table: {str(e)}")
                # Continue with other tables
        
        return True
    
    def load_sample_to_bigquery(self, df: pd.DataFrame, table_suffix: str = "c1_features") -> bool:
        """
        Load sample data to BigQuery table
        
        Args:
            df: DataFrame with sample data
            table_suffix: Table name suffix (e.g., 'c1_features')
            
        Returns:
            True if successful
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_suffix}"
        self.logger.info(f"Loading {len(df)} records to {table_id}...")
        
        # Configure load job
        job_config = bigquery.LoadJobConfig(
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            time_partitioning=bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="date"
            ),
            clustering_fields=["symbol", "dte"]
        )
        
        try:
            # Load data
            job = self.bq_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            job.result()  # Wait for completion
            
            self.logger.info(f"  ✓ Loaded {job.output_rows} rows to {table_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"  ✗ Error loading data: {str(e)}")
            return False
    
    def validate_loaded_data(self, table_suffix: str = "c1_features") -> Dict[str, Any]:
        """
        Validate data loaded to BigQuery
        
        Args:
            table_suffix: Table name suffix
            
        Returns:
            Validation results
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_suffix}"
        self.logger.info(f"Validating data in {table_id}...")
        
        validation_results = {}
        
        # Check row count
        query = f"SELECT COUNT(*) as row_count FROM `{table_id}` WHERE date >= CURRENT_DATE() - 1"
        result = self.bq_client.query(query).result()
        row_count = list(result)[0].row_count
        validation_results['row_count'] = row_count
        
        # Check null values for key columns
        query = f"""
        SELECT 
            COUNTIF(symbol IS NULL) as null_symbol,
            COUNTIF(ts_minute IS NULL) as null_timestamp,
            COUNTIF(dte IS NULL) as null_dte
        FROM `{table_id}`
        WHERE date >= CURRENT_DATE() - 1
        """
        result = self.bq_client.query(query).result()
        null_counts = list(result)[0]
        validation_results['null_counts'] = {
            'symbol': null_counts.null_symbol,
            'timestamp': null_counts.null_timestamp,
            'dte': null_counts.null_dte
        }
        
        # Check data freshness
        query = f"""
        SELECT 
            MIN(ts_minute) as earliest,
            MAX(ts_minute) as latest
        FROM `{table_id}`
        WHERE date >= CURRENT_DATE() - 1
        """
        result = self.bq_client.query(query).result()
        timestamps = list(result)[0]
        validation_results['data_freshness'] = {
            'earliest': timestamps.earliest,
            'latest': timestamps.latest
        }
        
        # Check partitioning effectiveness
        query = f"""
        SELECT 
            COUNT(DISTINCT date) as partition_count,
            COUNT(DISTINCT symbol) as symbol_count,
            COUNT(DISTINCT dte) as dte_count
        FROM `{table_id}`
        WHERE date >= CURRENT_DATE() - 7
        """
        result = self.bq_client.query(query).result()
        stats = list(result)[0]
        validation_results['partitioning'] = {
            'partition_count': stats.partition_count,
            'symbol_count': stats.symbol_count,
            'dte_count': stats.dte_count
        }
        
        return validation_results
    
    def generate_performance_metrics(self) -> Dict[str, Any]:
        """
        Generate performance metrics for sample queries
        
        Returns:
            Performance metrics
        """
        self.logger.info("Generating performance metrics...")
        
        metrics = {}
        
        # Test point-in-time query
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.c1_features`
        WHERE date = CURRENT_DATE()
          AND symbol = 'NIFTY'
          AND dte = 7
        LIMIT 1
        """
        
        start_time = time.time()
        job = self.bq_client.query(query)
        result = job.result()
        query_time = time.time() - start_time
        
        metrics['point_in_time_query'] = {
            'execution_time_seconds': query_time,
            'bytes_processed': job.total_bytes_processed,
            'bytes_billed': job.total_bytes_billed,
            'cache_hit': job.cache_hit
        }
        
        # Test aggregation query
        query = f"""
        SELECT 
            zone_name,
            AVG(c1_momentum_score) as avg_momentum,
            COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.c1_features`
        WHERE date >= CURRENT_DATE() - 7
        GROUP BY zone_name
        """
        
        start_time = time.time()
        job = self.bq_client.query(query)
        result = job.result()
        query_time = time.time() - start_time
        
        metrics['aggregation_query'] = {
            'execution_time_seconds': query_time,
            'bytes_processed': job.total_bytes_processed,
            'bytes_billed': job.total_bytes_billed
        }
        
        return metrics
    
    def run_pipeline(self) -> bool:
        """
        Run the complete sample data pipeline
        
        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Sample Data Pipeline")
        self.logger.info("=" * 60)
        
        success = True
        
        # Step 1: Create BigQuery tables
        if not self.create_bigquery_tables():
            self.logger.error("Failed to create BigQuery tables")
            success = False
        
        # Step 2: Try to load from GCS first
        df = self.load_parquet_from_gcs("NIFTY")
        
        # Step 3: If no GCS data, generate sample data
        if df is None:
            self.logger.info("No GCS data found, generating sample data...")
            df = self.generate_sample_data(1000)
        
        # Step 4: Load sample data to each component table
        # For simplicity, we'll load a subset to c1_features as example
        if df is not None:
            # Select relevant columns for c1_features
            c1_columns = ['symbol', 'ts_minute', 'date', 'dte', 'zone_name',
                         'c1_momentum_score', 'c1_vol_compression', 'c1_breakout_probability',
                         'created_at', 'updated_at']
            
            c1_df = df[[col for col in c1_columns if col in df.columns]]
            
            if not self.load_sample_to_bigquery(c1_df, "c1_features"):
                self.logger.error("Failed to load sample data")
                success = False
        
        # Step 5: Validate loaded data
        validation = self.validate_loaded_data("c1_features")
        self.logger.info("Validation Results:")
        self.logger.info(f"  - Row count: {validation['row_count']}")
        self.logger.info(f"  - Null counts: {validation['null_counts']}")
        self.logger.info(f"  - Data freshness: {validation['data_freshness']}")
        
        # Step 6: Generate performance metrics
        metrics = self.generate_performance_metrics()
        self.logger.info("Performance Metrics:")
        for query_type, stats in metrics.items():
            self.logger.info(f"  {query_type}:")
            self.logger.info(f"    - Execution time: {stats['execution_time_seconds']:.2f}s")
            self.logger.info(f"    - Bytes processed: {stats['bytes_processed']:,}")
        
        # Log to audit table
        self._log_to_audit(success, validation, metrics)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline {'completed successfully' if success else 'failed'}")
        self.logger.info("=" * 60)
        
        return success
    
    def _log_to_audit(self, success: bool, validation: Dict, metrics: Dict):
        """Log pipeline run to audit table"""
        try:
            audit_table = f"{self.project_id}.{self.dataset_id}.mr_load_audit"
            
            audit_record = {
                'load_id': f"sample_load_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'load_timestamp': datetime.now(),
                'table_name': 'c1_features',
                'row_count': validation.get('row_count', 0),
                'null_check_passed': all(v == 0 for v in validation.get('null_counts', {}).values()),
                'schema_validation_passed': success,
                'error_message': None if success else 'Check logs for details',
                'load_duration_seconds': sum(m['execution_time_seconds'] for m in metrics.values()),
                'created_at': datetime.now()
            }
            
            df = pd.DataFrame([audit_record])
            job = self.bq_client.load_table_from_dataframe(df, audit_table)
            job.result()
            
            self.logger.info(f"Audit record logged to {audit_table}")
            
        except Exception as e:
            self.logger.warning(f"Could not log to audit table: {str(e)}")


def main():
    """Main function"""
    pipeline = SampleDataPipeline(environment="dev")
    success = pipeline.run_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())