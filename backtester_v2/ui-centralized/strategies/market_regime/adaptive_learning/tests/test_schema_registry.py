"""
Test Schema Registry Implementation

Tests for version-controlled feature schema definitions, validation utilities,
and Vertex AI Feature Store integration.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from ..schema_registry import (
    FeatureDefinition,
    ComponentSchema, 
    SchemaRegistry,
    get_schema_registry
)
from .. import COMPONENT_FEATURE_COUNT, SchemaValidationError


class TestFeatureDefinition:
    """Test FeatureDefinition class."""
    
    def test_feature_definition_creation(self):
        """Test basic feature definition creation."""
        feature = FeatureDefinition(
            name="test_feature",
            data_type="float",
            description="Test feature for validation",
            valid_range={"min": 0.0, "max": 1.0},
            calibration_notes="Test calibration",
            thresholds={"warning": 0.8, "critical": 0.9}
        )
        
        assert feature.name == "test_feature"
        assert feature.data_type == "float"
        assert feature.description == "Test feature for validation"
        assert feature.valid_range == {"min": 0.0, "max": 1.0}
        assert feature.vertex_ai_compatible is True
        assert feature.version_introduced == "1.0.0"
    
    def test_feature_definition_defaults(self):
        """Test feature definition with defaults."""
        feature = FeatureDefinition(
            name="simple_feature",
            data_type="int",
            description="Simple test feature"
        )
        
        assert feature.valid_range is None
        assert feature.calibration_notes is None
        assert feature.thresholds is None
        assert feature.vertex_ai_compatible is True
        assert feature.version_introduced == "1.0.0"


class TestSchemaRegistry:
    """Test SchemaRegistry functionality."""
    
    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = SchemaRegistry(registry_path=Path(temp_dir))
            yield registry
    
    def test_registry_initialization(self, temp_registry):
        """Test registry initialization and schema loading."""
        # Should have schemas for all components
        schemas = temp_registry.get_all_schemas()
        assert len(schemas) == len(COMPONENT_FEATURE_COUNT)
        
        for component_id in COMPONENT_FEATURE_COUNT:
            assert component_id in schemas
            schema = schemas[component_id]
            assert schema.component_id == component_id
            assert schema.feature_count == COMPONENT_FEATURE_COUNT[component_id]
    
    def test_component_schema_structure(self, temp_registry):
        """Test component schema structure and content."""
        # Test component 01 (triple straddle)
        schema = temp_registry.get_schema("component_01_triple_straddle")
        assert schema is not None
        assert schema.feature_count == 120
        assert len(schema.features) == 120
        assert schema.version == "1.0.0"
        
        # Check for expected features
        feature_names = [f.name for f in schema.features]
        assert "atm_straddle_price" in feature_names
        assert "correlation_score" in feature_names
        
        # Validate feature definitions
        for feature in schema.features:
            assert isinstance(feature, FeatureDefinition)
            assert feature.name.strip() != ""
            assert feature.data_type in ["float", "int", "str", "bool"]
            assert feature.description.strip() != ""
    
    def test_component_02_gamma_weight_fix(self, temp_registry):
        """Test Component 02 has critical gamma weight fix."""
        schema = temp_registry.get_schema("component_02_greeks_sentiment")
        assert schema is not None
        assert schema.feature_count == 98
        
        # Find gamma feature
        gamma_features = [f for f in schema.features if "gamma" in f.name.lower()]
        assert len(gamma_features) > 0
        
        gamma_feature = gamma_features[0]  # volume_weighted_gamma
        assert gamma_feature.calibration_notes is not None
        assert "1.5" in gamma_feature.calibration_notes
        assert "pin risk" in gamma_feature.calibration_notes.lower()
    
    def test_feature_count_validation(self, temp_registry):
        """Test feature count validation across components."""
        validation_results = temp_registry.validate_all_schemas()
        
        # All schemas should validate successfully
        for component_id, is_valid in validation_results.items():
            assert is_valid, f"Schema validation failed for {component_id}"
        
        # Check total feature count
        total_features = sum(
            schema.feature_count 
            for schema in temp_registry.get_all_schemas().values()
        )
        assert total_features == 774
    
    def test_schema_validation_valid_features(self, temp_registry):
        """Test schema validation with valid feature data."""
        component_id = "component_01_triple_straddle"
        schema = temp_registry.get_schema(component_id)
        
        # Create valid test features
        test_features = {}
        for feature in schema.features:
            if feature.data_type == "float":
                if feature.valid_range:
                    min_val = feature.valid_range.get("min", 0.0)
                    max_val = feature.valid_range.get("max", 1.0)
                    test_features[feature.name] = (min_val + max_val) / 2
                else:
                    test_features[feature.name] = 0.5
            elif feature.data_type == "int":
                test_features[feature.name] = 1
            elif feature.data_type == "str":
                test_features[feature.name] = "test_value"
            else:  # bool
                test_features[feature.name] = True
        
        # Should validate successfully
        result = temp_registry.validate_features(component_id, test_features)
        assert result is True
    
    def test_schema_validation_invalid_features(self, temp_registry):
        """Test schema validation with invalid feature data."""
        component_id = "component_01_triple_straddle"
        
        # Test missing features
        with pytest.raises(SchemaValidationError, match="Feature count mismatch"):
            temp_registry.validate_features(component_id, {"incomplete": 1.0})
        
        # Test invalid feature name
        schema = temp_registry.get_schema(component_id)
        test_features = {f"invalid_feature_{i}": 0.0 for i in range(schema.feature_count)}
        
        with pytest.raises(SchemaValidationError, match="Missing feature"):
            temp_registry.validate_features(component_id, test_features)
        
        # Test invalid data type
        valid_features = {feature.name: 0.0 for feature in schema.features}
        valid_features[schema.features[0].name] = "invalid_type"  # Should be float
        
        with pytest.raises(SchemaValidationError, match="Type mismatch"):
            temp_registry.validate_features(component_id, valid_features)
    
    def test_schema_validation_out_of_range(self, temp_registry):
        """Test schema validation with out-of-range values."""
        component_id = "component_01_triple_straddle"
        schema = temp_registry.get_schema(component_id)
        
        # Find a feature with range constraints
        range_feature = None
        for feature in schema.features:
            if feature.valid_range and feature.data_type in ["float", "int"]:
                range_feature = feature
                break
        
        if range_feature:
            # Create features with one out-of-range value
            test_features = {feature.name: 0.0 for feature in schema.features}
            
            # Set value outside valid range
            max_val = range_feature.valid_range.get("max", 1.0)
            test_features[range_feature.name] = max_val + 1000  # Way out of range
            
            with pytest.raises(SchemaValidationError, match="Value out of range"):
                temp_registry.validate_features(component_id, test_features)
    
    def test_vertex_ai_export(self, temp_registry):
        """Test Vertex AI Feature Store export functionality."""
        component_id = "component_01_triple_straddle"
        vertex_export = temp_registry.export_to_vertex_ai(component_id)
        
        assert "entity_type_id" in vertex_export
        assert vertex_export["entity_type_id"] == component_id
        
        assert "features" in vertex_export
        features = vertex_export["features"]
        assert len(features) > 0
        
        # Check feature format
        for feature in features:
            assert "name" in feature
            assert "value_type" in feature
            assert "description" in feature
            assert feature["value_type"] in ["DOUBLE", "INT64", "STRING", "BOOL"]
        
        assert "metadata" in vertex_export
        metadata = vertex_export["metadata"]
        assert "version" in metadata
        assert "feature_count" in metadata
        assert "created_timestamp" in metadata
        assert "schema_hash" in metadata
    
    def test_schema_persistence(self, temp_registry):
        """Test schema persistence and loading."""
        component_id = "component_01_triple_straddle"
        
        # Modify a schema
        schema = temp_registry.get_schema(component_id)
        original_feature_count = len(schema.features)
        
        # Add a test feature
        test_feature = FeatureDefinition(
            name="test_persistence_feature",
            data_type="float",
            description="Test feature for persistence"
        )
        schema.features.append(test_feature)
        schema.feature_count = len(schema.features)
        
        # Save schema
        temp_registry.save_schema(component_id)
        
        # Create new registry from same path
        new_registry = SchemaRegistry(temp_registry.registry_path)
        
        # Check that modification was persisted
        loaded_schema = new_registry.get_schema(component_id)
        assert loaded_schema is not None
        assert len(loaded_schema.features) == original_feature_count + 1
        
        # Find the test feature
        test_features = [f for f in loaded_schema.features if f.name == "test_persistence_feature"]
        assert len(test_features) == 1
        assert test_features[0].description == "Test feature for persistence"
    
    def test_data_type_mapping(self, temp_registry):
        """Test data type mapping for Vertex AI."""
        # Test internal type mapping
        registry = temp_registry
        
        assert registry._map_to_vertex_type("float") == "DOUBLE"
        assert registry._map_to_vertex_type("int") == "INT64"
        assert registry._map_to_vertex_type("str") == "STRING"
        assert registry._map_to_vertex_type("bool") == "BOOL"
        assert registry._map_to_vertex_type("unknown") == "DOUBLE"  # Default
    
    def test_schema_hash_consistency(self, temp_registry):
        """Test schema hash consistency."""
        component_id = "component_01_triple_straddle"
        schema = temp_registry.get_schema(component_id)
        
        # Hash should be consistent
        hash1 = temp_registry._compute_schema_hash(schema.features)
        hash2 = temp_registry._compute_schema_hash(schema.features)
        assert hash1 == hash2
        
        # Hash should change if features change
        modified_features = schema.features.copy()
        modified_features.append(FeatureDefinition("new_feature", "float", "New feature"))
        hash3 = temp_registry._compute_schema_hash(modified_features)
        assert hash3 != hash1


class TestSchemaRegistryIntegration:
    """Integration tests for schema registry."""
    
    def test_global_registry_singleton(self):
        """Test global registry singleton behavior."""
        registry1 = get_schema_registry()
        registry2 = get_schema_registry()
        
        # Should be same instance
        assert registry1 is registry2
        
        # Should have all component schemas
        schemas = registry1.get_all_schemas()
        assert len(schemas) == len(COMPONENT_FEATURE_COUNT)
    
    def test_performance_budgets_in_metadata(self):
        """Test that performance budgets are included in schema metadata."""
        registry = get_schema_registry()
        
        for component_id in COMPONENT_FEATURE_COUNT:
            schema = registry.get_schema(component_id)
            assert schema is not None
            
            metadata = schema.metadata
            assert "performance_budget_ms" in metadata
            assert "memory_budget_mb" in metadata
            
            # Validate budget values are reasonable
            perf_budget = metadata["performance_budget_ms"]
            memory_budget = metadata["memory_budget_mb"]
            
            assert 50 <= perf_budget <= 200  # Reasonable processing budget
            assert 100 <= memory_budget <= 500  # Reasonable memory budget
    
    def test_all_774_features_accounted(self):
        """Test that all 774 features are properly accounted for."""
        registry = get_schema_registry()
        
        total_features = 0
        component_breakdown = {}
        
        for component_id, expected_count in COMPONENT_FEATURE_COUNT.items():
            schema = registry.get_schema(component_id)
            assert schema is not None
            
            actual_count = len(schema.features)
            assert actual_count == expected_count, f"Feature count mismatch for {component_id}: {actual_count} != {expected_count}"
            
            total_features += actual_count
            component_breakdown[component_id] = actual_count
        
        assert total_features == 774, f"Total feature count: {total_features} != 774"
        
        # Log breakdown for verification
        print(f"Feature breakdown: {component_breakdown}")
        print(f"Total features: {total_features}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])