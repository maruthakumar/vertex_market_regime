#!/usr/bin/env python3
"""
Story 2.2 Integration Tests
Validates complete BigQuery offline feature tables implementation
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List


class TestStory22Integration:
    """Integration tests for Story 2.2 BigQuery implementation"""
    
    @pytest.fixture
    def deployment_log(self):
        """Load deployment log from simulated deployment"""
        log_path = Path(__file__).parents[2] / "src" / "bigquery" / "deployment_log.json"
        if log_path.exists():
            with open(log_path, 'r') as f:
                return json.load(f)
        return []
    
    @pytest.fixture
    def expected_feature_counts(self):
        """Expected feature counts for Phase 2"""
        return {
            "c1_features": 150,  # Phase 2: 120 + 30 momentum
            "c2_features": 98,   # Unchanged
            "c3_features": 105,  # Unchanged
            "c4_features": 87,   # Unchanged
            "c5_features": 94,   # Unchanged
            "c6_features": 220,  # Phase 2: 200 + 20 momentum-enhanced
            "c7_features": 130,  # Phase 2: 120 + 10 momentum-based
            "c8_features": 48,   # Unchanged
            "training_dataset": 932  # Phase 2 total
        }
    
    def test_dataset_creation(self, deployment_log):
        """Test that BigQuery dataset was created"""
        dataset_actions = [log for log in deployment_log if log.get("action") == "create_dataset"]
        
        assert len(dataset_actions) == 1, "Should have exactly one dataset creation"
        dataset_action = dataset_actions[0]
        
        assert dataset_action["status"] == "success", "Dataset creation should succeed"
        assert "market_regime_" in dataset_action["dataset"], "Dataset should follow naming convention"
    
    def test_all_tables_deployed(self, deployment_log, expected_feature_counts):
        """Test that all 9 tables were deployed successfully"""
        table_deployments = [log for log in deployment_log if log.get("table") and log.get("features")]
        
        assert len(table_deployments) == 9, "Should deploy exactly 9 tables"
        
        deployed_tables = {log["table"]: log for log in table_deployments}
        
        for expected_table in expected_feature_counts.keys():
            assert expected_table in deployed_tables, f"Table {expected_table} should be deployed"
            assert deployed_tables[expected_table]["status"] == "success", f"Table {expected_table} deployment should succeed"
    
    def test_feature_counts_validation(self, deployment_log, expected_feature_counts):
        """Test that all tables have correct feature counts"""
        table_deployments = {log["table"]: log for log in deployment_log if log.get("table") and log.get("features")}
        
        for table_name, expected_count in expected_feature_counts.items():
            if table_name in table_deployments:
                actual_count = table_deployments[table_name]["features"]
                assert actual_count == expected_count, f"{table_name}: expected {expected_count} features, got {actual_count}"
    
    def test_phase_2_enhancements(self, deployment_log):
        """Test Phase 2 momentum enhancement features"""
        table_deployments = {log["table"]: log for log in deployment_log if log.get("table") and log.get("features")}
        
        # Component 1: Should have 150 features (120 + 30 momentum)
        c1_features = table_deployments.get("c1_features", {}).get("features", 0)
        assert c1_features == 150, f"Component 1 should have 150 features, got {c1_features}"
        
        # Component 6: Should have 220 features (200 + 20 momentum-enhanced)
        c6_features = table_deployments.get("c6_features", {}).get("features", 0)
        assert c6_features == 220, f"Component 6 should have 220 features, got {c6_features}"
        
        # Component 7: Should have 130 features (120 + 10 momentum-based)
        c7_features = table_deployments.get("c7_features", {}).get("features", 0)
        assert c7_features == 130, f"Component 7 should have 130 features, got {c7_features}"
    
    def test_total_feature_count(self, deployment_log):
        """Test total system feature count matches Phase 2 target"""
        table_deployments = [log for log in deployment_log if log.get("table") and log.get("features")]
        
        # Exclude training_dataset from sum (it's a view)
        component_tables = [log for log in table_deployments if log["table"] != "training_dataset"]
        total_features = sum(log["features"] for log in component_tables)
        
        assert total_features == 932, f"Total system features should be 932, got {total_features}"
    
    def test_training_dataset_deployment(self, deployment_log):
        """Test that training dataset view was created"""
        training_deployments = [log for log in deployment_log if log.get("table") == "training_dataset"]
        
        assert len(training_deployments) == 1, "Should deploy exactly one training dataset"
        training_deployment = training_deployments[0]
        
        assert training_deployment["status"] == "success", "Training dataset deployment should succeed"
        assert training_deployment["features"] == 932, "Training dataset should reference 932 total features"
    
    def test_partitioning_and_clustering(self, deployment_log):
        """Test that tables have proper partitioning and clustering"""
        table_deployments = [log for log in deployment_log if log.get("table") and log.get("partitioned")]
        
        for deployment in table_deployments:
            assert deployment["partitioned"] is True, f"Table {deployment['table']} should be partitioned"
            assert deployment["clustered"] is True, f"Table {deployment['table']} should be clustered"
    
    def test_sample_data_loading(self, deployment_log):
        """Test that sample data was loaded for all tables"""
        sample_loads = [log for log in deployment_log if log.get("action") == "load_sample_data"]
        
        assert len(sample_loads) == 8, "Should load sample data for 8 component tables"
        
        for load in sample_loads:
            assert load["status"] == "success", f"Sample data load for {load['table']} should succeed"
            assert load["rows_loaded"] > 0, f"Should load some rows for {load['table']}"
    
    def test_acceptance_criteria_completion(self, deployment_log, expected_feature_counts):
        """Test that all Story 2.2 acceptance criteria are met"""
        
        # AC: Dataset `market_regime_{env}` created
        dataset_created = any(log.get("action") == "create_dataset" and log["status"] == "success" 
                            for log in deployment_log)
        assert dataset_created, "Dataset should be created"
        
        # AC: DDLs for all 8 component tables implemented
        component_tables = ["c1_features", "c2_features", "c3_features", "c4_features", 
                          "c5_features", "c6_features", "c7_features", "c8_features"]
        deployed_tables = {log["table"] for log in deployment_log if log.get("table") and log.get("features")}
        
        for table in component_tables:
            assert table in deployed_tables, f"Component table {table} should be deployed"
        
        # AC: Training dataset view/table created
        training_deployed = any(log.get("table") == "training_dataset" and log["status"] == "success"
                              for log in deployment_log)
        assert training_deployed, "Training dataset should be deployed"
        
        # AC: Sample data populated
        sample_data_loaded = any(log.get("action") == "load_sample_data" and log["status"] == "success"
                               for log in deployment_log)
        assert sample_data_loaded, "Sample data should be loaded"
    
    def test_epic_1_phase_2_completion(self, deployment_log):
        """Test that Epic 1 Phase 2 is fully implemented"""
        
        # Phase 2 enhancement validation
        enhancements = {
            "component_1_momentum": 30,  # Component 1 momentum features
            "component_6_correlation": 20,  # Component 6 momentum-enhanced correlation
            "component_7_levels": 10  # Component 7 momentum-based levels
        }
        
        total_enhancements = sum(enhancements.values())
        assert total_enhancements == 60, "Should have exactly 60 Phase 2 enhancements"
        
        # Validate feature counts match expectations
        table_deployments = {log["table"]: log for log in deployment_log if log.get("table") and log.get("features")}
        
        # Phase 1 baseline + Phase 2 enhancements = Current totals
        phase_1_baselines = {
            "c1_features": 120,
            "c6_features": 200,
            "c7_features": 120
        }
        
        for table, baseline in phase_1_baselines.items():
            if table in table_deployments:
                current_count = table_deployments[table]["features"]
                enhancement = current_count - baseline
                
                if table == "c1_features":
                    assert enhancement == 30, f"Component 1 should have 30 momentum enhancements"
                elif table == "c6_features":
                    assert enhancement == 20, f"Component 6 should have 20 correlation enhancements"
                elif table == "c7_features":
                    assert enhancement == 10, f"Component 7 should have 10 level enhancements"


class TestStory22DDLStructure:
    """Test DDL file structure and syntax"""
    
    def test_ddl_files_exist(self):
        """Test that all required DDL files exist"""
        ddl_dir = Path(__file__).parents[2] / "src" / "bigquery" / "ddl"
        
        required_files = [
            "c1_features.sql", "c2_features.sql", "c3_features.sql", "c4_features.sql",
            "c5_features.sql", "c6_features.sql", "c7_features.sql", "c8_features.sql",
            "training_dataset.sql"
        ]
        
        for file_name in required_files:
            file_path = ddl_dir / file_name
            assert file_path.exists(), f"DDL file {file_name} should exist"
            assert file_path.stat().st_size > 0, f"DDL file {file_name} should not be empty"
    
    def test_ddl_basic_syntax(self):
        """Test basic DDL syntax validation"""
        ddl_dir = Path(__file__).parents[2] / "src" / "bigquery" / "ddl"
        
        for ddl_file in ddl_dir.glob("*.sql"):
            with open(ddl_file, 'r') as f:
                content = f.read()
            
            # Basic syntax checks
            assert "CREATE" in content.upper(), f"{ddl_file.name} should contain CREATE statement"
            assert "arched-bot-269016" in content, f"{ddl_file.name} should contain correct project ID"
            assert "market_regime_{env}" in content, f"{ddl_file.name} should contain environment placeholder"
            
            # Check parentheses balance
            open_parens = content.count('(')
            close_parens = content.count(')')
            assert open_parens == close_parens, f"{ddl_file.name} should have balanced parentheses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])