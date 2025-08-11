"""
Schema Registry for Adaptive Learning Framework

Provides version-controlled feature schema definitions for all 8 components,
schema validation utilities, and integration with Vertex AI Feature Store.

This module ensures consistent, versioned feature definitions across all components
and provides export/import functionality for Epic 2 handoff to cloud services.
"""

from typing import Dict, Any, List, Optional, Union
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from .. import SchemaValidationError, COMPONENT_FEATURE_COUNT


logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    data_type: str  # 'float', 'int', 'str', 'bool'
    description: str
    valid_range: Optional[Dict[str, Union[float, int]]] = None  # {'min': x, 'max': y}
    calibration_notes: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None
    vertex_ai_compatible: bool = True
    version_introduced: str = "1.0.0"


@dataclass
class ComponentSchema:
    """Schema definition for a single component."""
    component_id: str
    version: str
    feature_count: int
    features: List[FeatureDefinition]
    metadata: Dict[str, Any]
    created_timestamp: str
    schema_hash: str


class SchemaRegistry:
    """
    Registry for managing feature schemas across all 8 adaptive components.
    
    Provides:
    - Version-controlled schema management
    - Schema validation utilities
    - Vertex AI Feature Store integration
    - Migration path management
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize schema registry.
        
        Args:
            registry_path: Path to schema registry directory
        """
        if registry_path is None:
            registry_path = Path(__file__).parent / "schemas"
        
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self._schemas: Dict[str, ComponentSchema] = {}
        self._load_schemas()
        
        logger.info(f"Schema registry initialized with {len(self._schemas)} component schemas")
    
    def _load_schemas(self) -> None:
        """Load all component schemas from registry."""
        for component_id in COMPONENT_FEATURE_COUNT.keys():
            schema_file = self.registry_path / f"{component_id}_schema.json"
            if schema_file.exists():
                try:
                    with open(schema_file, 'r') as f:
                        schema_data = json.load(f)
                    schema = self._dict_to_schema(schema_data)
                    self._schemas[component_id] = schema
                    logger.debug(f"Loaded schema for {component_id}")
                except Exception as e:
                    logger.error(f"Failed to load schema for {component_id}: {str(e)}")
            else:
                # Create default schema if not exists
                logger.info(f"Creating default schema for {component_id}")
                self._create_default_schema(component_id)
    
    def _create_default_schema(self, component_id: str) -> None:
        """Create default schema for a component."""
        feature_count = COMPONENT_FEATURE_COUNT[component_id]
        
        # Create default features based on component type
        features = self._generate_default_features(component_id, feature_count)
        
        schema = ComponentSchema(
            component_id=component_id,
            version="1.0.0",
            feature_count=feature_count,
            features=features,
            metadata={
                "description": f"Default schema for {component_id}",
                "author": "Adaptive Learning Framework",
                "performance_budget_ms": self._get_component_budget(component_id),
                "memory_budget_mb": self._get_component_memory_budget(component_id)
            },
            created_timestamp=datetime.now().isoformat(),
            schema_hash=self._compute_schema_hash(features)
        )
        
        self._schemas[component_id] = schema
        self.save_schema(component_id)
    
    def _generate_default_features(self, component_id: str, feature_count: int) -> List[FeatureDefinition]:
        """Generate default feature definitions for a component."""
        features = []
        
        if component_id == "component_01_triple_straddle":
            features.extend([
                FeatureDefinition("atm_straddle_price", "float", "ATM straddle price", 
                                {"min": 0.0, "max": 10000.0}, "Current market price of ATM straddle"),
                FeatureDefinition("itm1_straddle_price", "float", "ITM1 straddle price",
                                {"min": 0.0, "max": 15000.0}, "Current market price of ITM1 straddle"),
                FeatureDefinition("otm1_straddle_price", "float", "OTM1 straddle price",
                                {"min": 0.0, "max": 8000.0}, "Current market price of OTM1 straddle"),
                FeatureDefinition("straddle_ema_5", "float", "5-period EMA of straddle prices",
                                {"min": 0.0, "max": 12000.0}, "Short-term trend indicator"),
                FeatureDefinition("straddle_ema_20", "float", "20-period EMA of straddle prices",
                                {"min": 0.0, "max": 12000.0}, "Medium-term trend indicator"),
                FeatureDefinition("straddle_vwap", "float", "Volume-weighted average price of straddles",
                                {"min": 0.0, "max": 12000.0}, "Volume-based fair value"),
                FeatureDefinition("correlation_score", "float", "10x10 correlation matrix score",
                                {"min": -1.0, "max": 1.0}, "Cross-component correlation strength")
            ])
        
        elif component_id == "component_02_greeks_sentiment":
            features.extend([
                FeatureDefinition("volume_weighted_delta", "float", "Volume-weighted delta exposure",
                                {"min": -10.0, "max": 10.0}, "Delta exposure adjusted by volume"),
                FeatureDefinition("volume_weighted_gamma", "float", "Volume-weighted gamma exposure",
                                {"min": 0.0, "max": 5.0}, "CRITICAL: Fixed weight 1.5 for pin risk"),
                FeatureDefinition("volume_weighted_theta", "float", "Volume-weighted theta decay",
                                {"min": -1000.0, "max": 0.0}, "Time decay exposure"),
                FeatureDefinition("volume_weighted_vega", "float", "Volume-weighted vega exposure",
                                {"min": -100.0, "max": 100.0}, "Volatility sensitivity"),
                FeatureDefinition("sentiment_score", "float", "7-level sentiment classification",
                                {"min": -3.0, "max": 3.0}, "Extreme bearish (-3) to extreme bullish (+3)"),
                FeatureDefinition("institutional_flow_strength", "float", "Large player detection score",
                                {"min": 0.0, "max": 1.0}, "Institutional activity indicator")
            ])
        
        # Fill remaining features with generic pattern
        remaining_features = feature_count - len(features)
        for i in range(remaining_features):
            features.append(
                FeatureDefinition(
                    f"{component_id}_feature_{i+len(features)+1:03d}",
                    "float",
                    f"Component feature {i+len(features)+1}",
                    {"min": -1.0, "max": 1.0},
                    f"Generic feature for {component_id}"
                )
            )
        
        return features
    
    def _get_component_budget(self, component_id: str) -> int:
        """Get processing budget for component."""
        budgets = {
            "component_01_triple_straddle": 100,
            "component_02_greeks_sentiment": 80,
            "component_03_oi_pa_trending": 120,
            "component_04_iv_skew": 90,
            "component_05_atr_ema_cpr": 110,
            "component_06_correlation": 150,
            "component_07_support_resistance": 85,
            "component_08_master_integration": 50
        }
        return budgets.get(component_id, 100)
    
    def _get_component_memory_budget(self, component_id: str) -> int:
        """Get memory budget for component."""
        budgets = {
            "component_01_triple_straddle": 320,
            "component_02_greeks_sentiment": 280,
            "component_03_oi_pa_trending": 300,
            "component_04_iv_skew": 250,
            "component_05_atr_ema_cpr": 270,
            "component_06_correlation": 450,
            "component_07_support_resistance": 220,
            "component_08_master_integration": 180
        }
        return budgets.get(component_id, 256)
    
    def _compute_schema_hash(self, features: List[FeatureDefinition]) -> str:
        """Compute hash of schema for version tracking."""
        schema_str = json.dumps([asdict(f) for f in features], sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    def _dict_to_schema(self, data: Dict[str, Any]) -> ComponentSchema:
        """Convert dictionary to ComponentSchema object."""
        features = [FeatureDefinition(**f) for f in data['features']]
        return ComponentSchema(
            component_id=data['component_id'],
            version=data['version'],
            feature_count=data['feature_count'],
            features=features,
            metadata=data['metadata'],
            created_timestamp=data['created_timestamp'],
            schema_hash=data['schema_hash']
        )
    
    def get_schema(self, component_id: str) -> Optional[ComponentSchema]:
        """Get schema for component."""
        return self._schemas.get(component_id)
    
    def save_schema(self, component_id: str) -> None:
        """Save schema to registry."""
        if component_id not in self._schemas:
            raise ValueError(f"Schema not found for component: {component_id}")
        
        schema = self._schemas[component_id]
        schema_file = self.registry_path / f"{component_id}_schema.json"
        
        schema_dict = asdict(schema)
        
        with open(schema_file, 'w') as f:
            json.dump(schema_dict, f, indent=2)
        
        logger.info(f"Saved schema for {component_id}")
    
    def validate_features(self, component_id: str, features: Dict[str, Any]) -> bool:
        """
        Validate feature data against component schema.
        
        Args:
            component_id: Component identifier
            features: Feature data to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            SchemaValidationError: If validation fails
        """
        schema = self.get_schema(component_id)
        if not schema:
            raise SchemaValidationError(f"No schema found for component: {component_id}")
        
        # Check feature count
        expected_count = schema.feature_count
        actual_count = len(features)
        if actual_count != expected_count:
            raise SchemaValidationError(
                f"Feature count mismatch for {component_id}: "
                f"expected {expected_count}, got {actual_count}"
            )
        
        # Validate individual features
        for feature_def in schema.features:
            feature_name = feature_def.name
            
            if feature_name not in features:
                raise SchemaValidationError(
                    f"Missing feature '{feature_name}' for component {component_id}"
                )
            
            value = features[feature_name]
            
            # Type validation
            expected_type = feature_def.data_type
            if not self._validate_feature_type(value, expected_type):
                raise SchemaValidationError(
                    f"Type mismatch for feature '{feature_name}': "
                    f"expected {expected_type}, got {type(value).__name__}"
                )
            
            # Range validation
            if feature_def.valid_range and expected_type in ['float', 'int']:
                if not self._validate_feature_range(value, feature_def.valid_range):
                    raise SchemaValidationError(
                        f"Value out of range for feature '{feature_name}': "
                        f"{value} not in {feature_def.valid_range}"
                    )
        
        return True
    
    def _validate_feature_type(self, value: Any, expected_type: str) -> bool:
        """Validate feature value type."""
        type_mapping = {
            'float': (float, int),  # Allow int as float
            'int': int,
            'str': str,
            'bool': bool
        }
        
        expected_types = type_mapping.get(expected_type, str)
        if isinstance(expected_types, tuple):
            return isinstance(value, expected_types)
        else:
            return isinstance(value, expected_types)
    
    def _validate_feature_range(self, value: Union[float, int], valid_range: Dict[str, Union[float, int]]) -> bool:
        """Validate feature value is within valid range."""
        min_val = valid_range.get('min')
        max_val = valid_range.get('max')
        
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        
        return True
    
    def export_to_vertex_ai(self, component_id: str) -> Dict[str, Any]:
        """
        Export schema to Vertex AI Feature Store format.
        
        Args:
            component_id: Component to export
            
        Returns:
            Dictionary in Vertex AI Feature Store format
        """
        schema = self.get_schema(component_id)
        if not schema:
            raise ValueError(f"No schema found for component: {component_id}")
        
        # Convert to Vertex AI format
        vertex_features = []
        for feature in schema.features:
            if feature.vertex_ai_compatible:
                vertex_feature = {
                    "name": feature.name,
                    "value_type": self._map_to_vertex_type(feature.data_type),
                    "description": feature.description
                }
                vertex_features.append(vertex_feature)
        
        return {
            "entity_type_id": component_id,
            "features": vertex_features,
            "metadata": {
                "version": schema.version,
                "feature_count": len(vertex_features),
                "created_timestamp": schema.created_timestamp,
                "schema_hash": schema.schema_hash
            }
        }
    
    def _map_to_vertex_type(self, data_type: str) -> str:
        """Map internal data type to Vertex AI type."""
        mapping = {
            'float': 'DOUBLE',
            'int': 'INT64',
            'str': 'STRING',
            'bool': 'BOOL'
        }
        return mapping.get(data_type, 'DOUBLE')
    
    def get_all_schemas(self) -> Dict[str, ComponentSchema]:
        """Get all registered schemas."""
        return self._schemas.copy()
    
    def validate_all_schemas(self) -> Dict[str, bool]:
        """Validate all schemas and return status."""
        results = {}
        total_features = 0
        
        for component_id, schema in self._schemas.items():
            try:
                # Validate feature count matches expectation
                expected_count = COMPONENT_FEATURE_COUNT[component_id]
                if schema.feature_count != expected_count:
                    results[component_id] = False
                    logger.error(
                        f"Feature count mismatch for {component_id}: "
                        f"expected {expected_count}, schema has {schema.feature_count}"
                    )
                else:
                    results[component_id] = True
                    total_features += schema.feature_count
                    logger.info(f"Schema validation passed for {component_id}")
            except Exception as e:
                results[component_id] = False
                logger.error(f"Schema validation failed for {component_id}: {str(e)}")
        
        # Validate total feature count
        if total_features != 774:
            logger.error(f"Total feature count mismatch: {total_features} != 774")
        else:
            logger.info("Total feature count validation passed: 774 features")
        
        return results


# Global schema registry instance
_registry_instance = None


def get_schema_registry() -> SchemaRegistry:
    """Get global schema registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SchemaRegistry()
    return _registry_instance


# Export key classes
__all__ = [
    "FeatureDefinition",
    "ComponentSchema", 
    "SchemaRegistry",
    "get_schema_registry"
]