"""
Feature Store Mapping and Schema Definitions
Maps BigQuery offline features to Vertex AI Feature Store online features
32 core features for <50ms serving latency
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path


class FeatureType(Enum):
    """Feature value types supported by Vertex AI Feature Store"""
    DOUBLE = "DOUBLE"
    INT64 = "INT64"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    BYTES = "BYTES"
    DOUBLE_ARRAY = "DOUBLE_ARRAY"
    INT64_ARRAY = "INT64_ARRAY"
    STRING_ARRAY = "STRING_ARRAY"
    BOOLEAN_ARRAY = "BOOLEAN_ARRAY"


@dataclass
class FeatureDefinition:
    """Definition of a single feature"""
    feature_id: str
    display_name: str
    description: str
    value_type: FeatureType
    component_source: str
    bigquery_column: str
    ttl_hours: int = 48
    monitoring_enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_vertex_ai_config(self) -> Dict[str, Any]:
        """Convert to Vertex AI Feature Store configuration"""
        return {
            "feature_id": self.feature_id,
            "value_type": self.value_type.value,
            "description": self.description,
            "labels": {
                **self.labels,
                "component": self.component_source,
                "ttl_hours": str(self.ttl_hours)
            }
        }


@dataclass
class ComponentFeatureGroup:
    """Group of features from a single component"""
    component_id: str
    component_name: str
    features: List[FeatureDefinition]
    priority: int = 1  # 1=highest, 5=lowest
    
    def get_feature_count(self) -> int:
        """Get number of features in this group"""
        return len(self.features)
    
    def get_feature_ids(self) -> List[str]:
        """Get list of feature IDs"""
        return [f.feature_id for f in self.features]


class FeatureStoreMapping:
    """
    Feature Store Mapping System
    Manages mapping between BigQuery offline features and Feature Store online features
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize feature mapping system"""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "feature_store_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize feature definitions
        self.feature_groups: Dict[str, ComponentFeatureGroup] = {}
        self.all_features: Dict[str, FeatureDefinition] = {}
        
        self._initialize_feature_definitions()
        
    def _initialize_feature_definitions(self):
        """Initialize the 32 core online feature definitions"""
        
        # Component 1: Triple Rolling Straddle (4 features)
        c1_features = [
            FeatureDefinition(
                feature_id="c1_momentum_score",
                display_name="Momentum Score",
                description="Momentum score from straddle analysis indicating directional strength",
                value_type=FeatureType.DOUBLE,
                component_source="component_01_triple_straddle",
                bigquery_column="c1_momentum_score",
                labels={"category": "momentum", "priority": "high"}
            ),
        FeatureDefinition(
            name="c1_vol_compression",
            feature_type=FeatureType.FLOAT32,
            description="Volatility compression indicator",
            component="c1",
            online=True,
            bigquery_column="c1_vol_compression"
        ),
        FeatureDefinition(
            name="c1_breakout_probability",
            feature_type=FeatureType.FLOAT32,
            description="Probability of price breakout",
            component="c1",
            online=True,
            bigquery_column="c1_breakout_probability"
        ),
        
        # Component 2 - Greeks Sentiment
        FeatureDefinition(
            name="c2_gamma_exposure",
            feature_type=FeatureType.FLOAT32,
            description="Gamma exposure with weight=1.5",
            component="c2",
            online=True,
            bigquery_column="c2_gamma_exposure"
        ),
        FeatureDefinition(
            name="c2_sentiment_level",
            feature_type=FeatureType.INT32,
            description="Market sentiment level",
            component="c2",
            online=True,
            bigquery_column="c2_sentiment_level"
        ),
        FeatureDefinition(
            name="c2_pin_risk_score",
            feature_type=FeatureType.FLOAT32,
            description="Pin risk at max pain level",
            component="c2",
            online=True,
            bigquery_column="c2_pin_risk_score"
        ),
        
        # Component 3 - OI-PA Trending
        FeatureDefinition(
            name="c3_institutional_flow_score",
            feature_type=FeatureType.FLOAT32,
            description="Institutional flow indicator",
            component="c3",
            online=True,
            bigquery_column="c3_institutional_flow_score"
        ),
        FeatureDefinition(
            name="c3_divergence_type",
            feature_type=FeatureType.INT32,
            description="Type of OI-Price divergence",
            component="c3",
            online=True,
            bigquery_column="c3_divergence_type"
        ),
        FeatureDefinition(
            name="c3_range_expansion_score",
            feature_type=FeatureType.FLOAT32,
            description="Range expansion probability",
            component="c3",
            online=True,
            bigquery_column="c3_range_expansion_score"
        ),
        
        # Component 4 - IV Skew/Percentiles
        FeatureDefinition(
            name="c4_skew_bias_score",
            feature_type=FeatureType.FLOAT32,
            description="IV skew bias indicator",
            component="c4",
            online=True,
            bigquery_column="c4_skew_bias_score"
        ),
        FeatureDefinition(
            name="c4_term_structure_signal",
            feature_type=FeatureType.INT32,
            description="Term structure signal",
            component="c4",
            online=True,
            bigquery_column="c4_term_structure_signal"
        ),
        FeatureDefinition(
            name="c4_iv_regime_level",
            feature_type=FeatureType.INT32,
            description="IV regime classification",
            component="c4",
            online=True,
            bigquery_column="c4_iv_regime_level"
        ),
        
        # Component 5 - ATR-EMA-CPR
        FeatureDefinition(
            name="c5_momentum_score",
            feature_type=FeatureType.FLOAT32,
            description="Technical momentum score",
            component="c5",
            online=True,
            bigquery_column="c5_momentum_score"
        ),
        FeatureDefinition(
            name="c5_volatility_regime_score",
            feature_type=FeatureType.INT32,
            description="Volatility regime classification",
            component="c5",
            online=True,
            bigquery_column="c5_volatility_regime_score"
        ),
        FeatureDefinition(
            name="c5_confluence_score",
            feature_type=FeatureType.FLOAT32,
            description="Technical confluence score",
            component="c5",
            online=True,
            bigquery_column="c5_confluence_score"
        ),
        
        # Component 6 - Correlation/Prediction
        FeatureDefinition(
            name="c6_correlation_agreement_score",
            feature_type=FeatureType.FLOAT32,
            description="Correlation matrix agreement",
            component="c6",
            online=True,
            bigquery_column="c6_correlation_agreement_score"
        ),
        FeatureDefinition(
            name="c6_breakdown_alert",
            feature_type=FeatureType.INT32,
            description="Correlation breakdown alert",
            component="c6",
            online=True,
            bigquery_column="c6_breakdown_alert"
        ),
        FeatureDefinition(
            name="c6_system_stability_score",
            feature_type=FeatureType.FLOAT32,
            description="System stability indicator",
            component="c6",
            online=True,
            bigquery_column="c6_system_stability_score"
        ),
        
        # Component 7 - Support/Resistance
        FeatureDefinition(
            name="c7_level_strength_score",
            feature_type=FeatureType.FLOAT32,
            description="S/R level strength",
            component="c7",
            online=True,
            bigquery_column="c7_level_strength_score"
        ),
        FeatureDefinition(
            name="c7_breakout_probability",
            feature_type=FeatureType.FLOAT32,
            description="Breakout probability at levels",
            component="c7",
            online=True,
            bigquery_column="c7_breakout_probability"
        ),
        
        # Component 8 - Integration
        FeatureDefinition(
            name="c8_component_agreement_score",
            feature_type=FeatureType.FLOAT32,
            description="Component agreement score",
            component="c8",
            online=True,
            bigquery_column="c8_component_agreement_score"
        ),
        FeatureDefinition(
            name="c8_integration_confidence",
            feature_type=FeatureType.FLOAT32,
            description="Integration confidence level",
            component="c8",
            online=True,
            bigquery_column="c8_integration_confidence"
        ),
        FeatureDefinition(
            name="c8_transition_probability_hint",
            feature_type=FeatureType.FLOAT32,
            description="Regime transition probability",
            component="c8",
            online=True,
            bigquery_column="c8_transition_probability_hint"
        ),
        
        # Context features
        FeatureDefinition(
            name="zone_name",
            feature_type=FeatureType.STRING,
            description="Trading zone (OPEN/MID_MORN/LUNCH/AFTERNOON/CLOSE)",
            component="common",
            online=True,
            bigquery_column="zone_name"
        ),
        FeatureDefinition(
            name="ts_minute",
            feature_type=FeatureType.TIMESTAMP,
            description="Minute timestamp",
            component="common",
            online=True,
            bigquery_column="ts_minute"
        ),
        FeatureDefinition(
            name="symbol",
            feature_type=FeatureType.STRING,
            description="Trading symbol",
            component="common",
            online=True,
            bigquery_column="symbol"
        ),
        FeatureDefinition(
            name="dte",
            feature_type=FeatureType.INT64,
            description="Days to expiry",
            component="common",
            online=True,
            bigquery_column="dte"
        ),
    ]
    
    # BigQuery table mapping
    BIGQUERY_TABLES = {
        "c1": "c1_features",
        "c2": "c2_features",
        "c3": "c3_features",
        "c4": "c4_features",
        "c5": "c5_features",
        "c6": "c6_features",
        "c7": "c7_features",
        "c8": "c8_features",
        "training": "training_dataset"
    }
    
    # Feature counts per component (Phase 2 Enhanced)
    FEATURE_COUNTS = {
        "c1": 150,  # Triple Straddle + 30 momentum features (Phase 2)
        "c2": 98,   # Greeks Sentiment (γ=1.5)
        "c3": 105,  # OI-PA Trending
        "c4": 87,   # IV Skew Percentile
        "c5": 94,   # ATR-EMA-CPR
        "c6": 220,  # Correlation & Predictive + 20 momentum-enhanced features (Phase 2)
        "c7": 130,  # Support/Resistance + 10 momentum-based features (Phase 2)
        "c8": 48,   # Master Integration
        "total": 932  # Phase 2 total: 872 + 60 momentum enhancements
    }
    
    @classmethod
    def get_online_features(cls) -> List[FeatureDefinition]:
        """Get list of online features for Feature Store"""
        return [f for f in cls.ONLINE_FEATURES if f.online]
    
    @classmethod
    def get_offline_features_by_component(cls, component: str) -> Dict[str, str]:
        """Get BigQuery column names for offline features by component"""
        # This would be expanded with full offline feature list
        # For now, returning table name reference
        return {
            "table": f"market_regime_{{env}}.{cls.BIGQUERY_TABLES.get(component)}",
            "feature_count": cls.FEATURE_COUNTS.get(component, 0)
        }
    
    @classmethod
    def get_entity_id(cls, symbol: str, timestamp: str, dte: int) -> str:
        """Generate entity ID for Feature Store"""
        # Format: NIFTY_202508121430_7
        return f"{symbol}_{timestamp}_{dte}"
    
    @classmethod
    def get_feature_store_config(cls) -> Dict:
        """Get Feature Store configuration"""
        return {
            "entity_type": cls.ENTITY_TYPE,
            "entity_id_format": cls.ENTITY_ID_FORMAT,
            "online_feature_count": len([f for f in cls.ONLINE_FEATURES if f.online]),
            "offline_feature_count": cls.FEATURE_COUNTS["total"] - len([f for f in cls.ONLINE_FEATURES if f.online]),
            "ttl_hours": 48,
            "ttl_days_daily_aggregates": 30
        }
    
    @classmethod
    def validate_feature_completeness(cls) -> Dict[str, bool]:
        """Validate that all components have features defined"""
        validation = {}
        for component in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]:
            online_features = [f for f in cls.ONLINE_FEATURES if f.component == component]
            validation[f"{component}_has_online_features"] = len(online_features) > 0
            validation[f"{component}_has_table"] = component in cls.BIGQUERY_TABLES
        
        validation["total_online_features_valid"] = len([f for f in cls.ONLINE_FEATURES if f.online]) >= 32
        validation["all_components_covered"] = all(
            f"{c}_has_online_features" in validation and validation[f"{c}_has_online_features"]
            for c in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
        )
        
        return validation


if __name__ == "__main__":
    # Validate mapping
    mapping = FeatureStoreMapping()
    
    print("Feature Store Configuration:")
    print("-" * 50)
    config = mapping.get_feature_store_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    
    print("\nFeature Validation:")
    print("-" * 50)
    validation = mapping.validate_feature_completeness()
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}: {result}")
    
    print("\nOnline Features Summary:")
    print("-" * 50)
    online_features = mapping.get_online_features()
    component_counts = {}
    for feature in online_features:
        component_counts[feature.component] = component_counts.get(feature.component, 0) + 1
    
    for component, count in sorted(component_counts.items()):
        print(f"Component {component}: {count} online features")
    
    print(f"\nTotal online features: {len(online_features)}")
    print(f"Total offline features: {mapping.FEATURE_COUNTS['total'] - len(online_features)}")