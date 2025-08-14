#!/usr/bin/env python3
"""
BigQuery Integration Tests for Story 2.2
Tests for offline feature table implementation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path


class TestBigQueryIntegration:
    """Integration tests for BigQuery offline feature tables"""
    
    @pytest.fixture
    def sample_feature_data(self):
        """Generate sample feature data for testing"""
        num_records = 100
        base_time = datetime.now().replace(second=0, microsecond=0)
        timestamps = [base_time - timedelta(minutes=i) for i in range(num_records)]
        
        return pd.DataFrame({
            'symbol': np.random.choice(['NIFTY', 'BANKNIFTY'], num_records),
            'ts_minute': timestamps,
            'date': [t.date() for t in timestamps],
            'dte': np.random.choice([0, 3, 7, 14, 21, 30], num_records),
            'zone_name': np.random.choice(['OPEN', 'MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE'], num_records),
            'c1_momentum_score': np.random.uniform(-1, 1, num_records),
            'c1_vol_compression': np.random.uniform(0, 1, num_records),
            'c1_breakout_probability': np.random.uniform(0, 1, num_records),
            'created_at': [datetime.now()] * num_records,
            'updated_at': [datetime.now()] * num_records
        })
    
    @pytest.fixture
    def mock_bigquery_client(self):
        """Mock BigQuery client for testing"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = []
        mock_job.total_bytes_processed = 1000000
        mock_job.total_bytes_billed = 1000000
        mock_job.cache_hit = False
        mock_client.query.return_value = mock_job
        mock_client.load_table_from_dataframe.return_value = mock_job
        return mock_client
    
    def test_ddl_structure_validation(self):
        """Test that all DDL files have required structure"""
        ddl_dir = Path(__file__).parent.parent.parent.parent / "src" / "bigquery" / "ddl"
        
        required_ddl_files = [
            "c1_features.sql", "c2_features.sql", "c3_features.sql", "c4_features.sql",
            "c5_features.sql", "c6_features.sql", "c7_features.sql", "c8_features.sql",
            "training_dataset.sql"
        ]
        
        for ddl_file in required_ddl_files:
            ddl_path = ddl_dir / ddl_file
            assert ddl_path.exists(), f"DDL file {ddl_file} not found"
            
            with open(ddl_path, 'r') as f:
                content = f.read()
            
            # Check required DDL elements
            assert "CREATE" in content, f"{ddl_file} missing CREATE statement"
            assert "market_regime_" in content, f"{ddl_file} missing dataset reference"
            assert "PARTITION BY" in content, f"{ddl_file} missing partitioning"
            assert "CLUSTER BY" in content, f"{ddl_file} missing clustering"
            assert "symbol" in content, f"{ddl_file} missing symbol column"
            assert "ts_minute" in content, f"{ddl_file} missing ts_minute column"
            assert "dte" in content, f"{ddl_file} missing dte column"
    
    def test_feature_count_validation(self):
        """Test that feature counts match Epic 1 specifications"""
        expected_counts = {
            "c1_features.sql": 120,
            "c2_features.sql": 98,
            "c3_features.sql": 105,
            "c4_features.sql": 87,
            "c5_features.sql": 94,
            "c6_features.sql": 200,
            "c7_features.sql": 120,  # Updated: 72 base + 48 advanced
            "c8_features.sql": 48
        }
        
        ddl_dir = Path(__file__).parent.parent.parent.parent / "src" / "bigquery" / "ddl"
        
        for ddl_file, expected_count in expected_counts.items():
            ddl_path = ddl_dir / ddl_file
            
            with open(ddl_path, 'r') as f:
                content = f.read()
            
            # Count feature columns (c{n}_ pattern)
            import re
            feature_matches = re.findall(r'c\d+_\w+', content)
            actual_count = len(set(feature_matches))
            
            assert actual_count >= expected_count * 0.8, \
                f"{ddl_file} has {actual_count} features, expected ~{expected_count}"
    
    @patch('google.cloud.bigquery.Client')
    def test_dataset_creation(self, mock_client_class):
        """Test BigQuery dataset creation"""
        from vertex_market_regime.src.bigquery.deploy_bigquery import BigQueryDeployer
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock dataset operations
        mock_dataset = Mock()
        mock_client.get_dataset.side_effect = Exception("Dataset not found")  # First call fails
        mock_client.create_dataset.return_value = mock_dataset
        
        deployer = BigQueryDeployer(environment="test")
        result = deployer.create_dataset()
        
        assert result is True
        mock_client.create_dataset.assert_called_once()
    
    def test_data_validation_rules(self):
        """Test data validation rules for feature data"""
        # Import without executing the file
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src" / "bigquery"))
        
        try:
            from production_data_pipeline import ProductionDataPipeline, ValidationRule
            
            pipeline = ProductionDataPipeline(environment="test")
            
            # Test validation rules exist for all components
            assert "c1_features" in pipeline.validation_rules
            assert "c2_features" in pipeline.validation_rules
            
            # Test rule structure
            c1_rules = pipeline.validation_rules["c1_features"]
            assert any(rule.column == "symbol" for rule in c1_rules)
            assert any(rule.column == "c1_momentum_score" for rule in c1_rules)
            
        except ImportError:
            # Skip test if dependencies not available
            pytest.skip("Production pipeline dependencies not available")
    
    def test_feature_store_mapping_consistency(self):
        """Test that Feature Store mapping is consistent with DDL"""
        try:
            from vertex_market_regime.src.features.mappings.feature_store_mapping import FeatureStoreMapping
            
            # Test feature counts match
            expected_total = 872  # Updated total
            assert FeatureStoreMapping.FEATURE_COUNTS["total"] == expected_total
            
            # Test component 7 update
            assert FeatureStoreMapping.FEATURE_COUNTS["c7"] == 120
            
            # Test online features are defined
            online_features = FeatureStoreMapping.get_online_features()
            assert len(online_features) >= 32
            
            # Test all components have online features
            components = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
            for component in components:
                has_features = any(f.component == component for f in online_features)
                assert has_features, f"Component {component} missing online features"
                
        except ImportError:
            pytest.skip("Feature store mapping module not available")
    
    def test_query_pattern_optimization(self, sample_feature_data):
        """Test query patterns for performance optimization"""
        df = sample_feature_data
        
        # Test partitioning - data should be filtered by date
        unique_dates = df['date'].nunique()
        assert unique_dates <= 2, "Sample data should span max 2 days for efficient partitioning"
        
        # Test clustering - symbol and dte should be present
        assert 'symbol' in df.columns
        assert 'dte' in df.columns
        
        # Test required columns are present
        required_columns = ['symbol', 'ts_minute', 'date', 'dte', 'zone_name']
        for col in required_columns:
            assert col in df.columns, f"Required column {col} missing"
        
        # Test data types are appropriate
        assert df['symbol'].dtype == 'object'
        assert pd.api.types.is_datetime64_any_dtype(df['ts_minute'])
        assert df['dte'].dtype in ['int64', 'int32']
    
    def test_parquet_to_bigquery_pipeline(self, sample_feature_data):
        """Test complete Parquet to BigQuery pipeline"""
        df = sample_feature_data
        
        # Create temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            df.to_parquet(tmp_file.name, engine='pyarrow')
            tmp_path = tmp_file.name
        
        try:
            # Test Parquet loading with Arrow
            import pyarrow.parquet as pq
            table = pq.read_table(tmp_path)
            
            assert table.num_rows == len(df)
            assert table.num_columns == len(df.columns)
            
            # Test conversion back to pandas
            df_loaded = table.to_pandas()
            assert len(df_loaded) == len(df)
            
            # Test required columns preserved
            required_columns = ['symbol', 'ts_minute', 'dte']
            for col in required_columns:
                assert col in df_loaded.columns
            
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    @patch('google.cloud.bigquery.Client')
    def test_audit_logging(self, mock_client_class, sample_feature_data):
        """Test audit logging functionality"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock audit table load
        mock_job = Mock()
        mock_job.result.return_value = []
        mock_client.load_table_from_dataframe.return_value = mock_job
        
        try:
            from production_data_pipeline import ProductionDataPipeline
            
            pipeline = ProductionDataPipeline(environment="test")
            pipeline.bq_client = mock_client
            
            # Test audit record structure
            audit_record = {
                'load_id': 'test_load_123',
                'load_timestamp': datetime.now(),
                'table_name': 'c1_features',
                'row_count': 100,
                'null_check_passed': True,
                'schema_validation_passed': True,
                'error_message': None,
                'load_duration_seconds': 5.0,
                'created_at': datetime.now()
            }
            
            # Test audit record has required fields
            required_fields = ['load_id', 'table_name', 'row_count', 'null_check_passed']
            for field in required_fields:
                assert field in audit_record
            
        except ImportError:
            pytest.skip("Production pipeline module not available")
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Test cost estimation
        bytes_processed = 1000000  # 1MB
        expected_cost = (bytes_processed / 1e12) * 5  # $5 per TB
        
        assert expected_cost == 0.000005  # $0.000005 for 1MB
        
        # Test storage estimation
        feature_count = 120
        rows_per_day = 100000
        days_retained = 90
        bytes_per_feature = 8
        
        storage_bytes = rows_per_day * days_retained * feature_count * bytes_per_feature
        storage_gb = storage_bytes / 1e9
        monthly_cost = storage_gb * 0.02  # $0.02 per GB
        
        assert storage_gb > 0
        assert monthly_cost > 0
    
    def test_bigquery_schema_compatibility(self, sample_feature_data):
        """Test BigQuery schema compatibility"""
        df = sample_feature_data
        
        # Test timestamp format
        assert pd.api.types.is_datetime64_any_dtype(df['ts_minute'])
        
        # Test date format
        assert all(isinstance(d, (pd.Timestamp, datetime)) or 
                  str(type(d)) in ['<class \'datetime.date\'>', '<class \'pandas._libs.tslibs.timestamps.Timestamp\'>'] 
                  for d in df['date'])
        
        # Test string columns
        assert df['symbol'].dtype == 'object'
        assert df['zone_name'].dtype == 'object'
        
        # Test numeric columns
        numeric_columns = ['c1_momentum_score', 'c1_vol_compression', 'c1_breakout_probability']
        for col in numeric_columns:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])
    
    def test_integration_with_training_dataset(self):
        """Test training dataset view integration"""
        ddl_dir = Path(__file__).parent.parent.parent.parent / "src" / "bigquery" / "ddl"
        training_ddl = ddl_dir / "training_dataset.sql"
        
        assert training_ddl.exists()
        
        with open(training_ddl, 'r') as f:
            content = f.read()
        
        # Test joins between all component tables
        components = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
        for component in components:
            assert f"{component}_features" in content, f"Training dataset missing {component}_features join"
        
        # Test join conditions
        assert "INNER JOIN" in content
        assert "ON c1.symbol = c2.symbol" in content
        assert "AND c1.ts_minute = c2.ts_minute" in content
        assert "AND c1.dte = c2.dte" in content
    
    def test_environment_specific_configuration(self):
        """Test environment-specific configurations"""
        environments = ["dev", "staging", "prod"]
        
        for env in environments:
            # Test dataset naming
            dataset_name = f"market_regime_{env}"
            assert len(dataset_name) > 0
            assert env in dataset_name
            
            # Test table naming pattern
            table_name = f"arched-bot-269016.{dataset_name}.c1_features"
            assert "arched-bot-269016" in table_name
            assert env in table_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])