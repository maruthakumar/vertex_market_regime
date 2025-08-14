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


class ComprehensiveFeatureStoreMapping:
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
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning("Feature store config not found, using defaults")
            self.config = self._get_default_config()
        
        # Initialize feature definitions
        self.feature_groups: Dict[str, ComponentFeatureGroup] = {}
        self.all_features: Dict[str, FeatureDefinition] = {}
        
        self._initialize_feature_definitions()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file not found"""
        return {
            "project_config": {
                "project_id": "arched-bot-269016",
                "location": "us-central1"
            },
            "feature_store": {
                "featurestore_id": "market_regime_featurestore"
            }
        }
        
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
                feature_id="c1_vol_compression",
                display_name="Volume Compression",
                description="Volume compression indicator showing market squeeze conditions",
                value_type=FeatureType.DOUBLE,
                component_source="component_01_triple_straddle", 
                bigquery_column="c1_vol_compression",
                labels={"category": "volatility", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c1_breakout_probability",
                display_name="Breakout Probability",
                description="Probability of breakout from current trading range",
                value_type=FeatureType.DOUBLE,
                component_source="component_01_triple_straddle",
                bigquery_column="c1_breakout_probability", 
                labels={"category": "probability", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c1_transition_probability",
                display_name="Transition Probability", 
                description="Market regime transition probability",
                value_type=FeatureType.DOUBLE,
                component_source="component_01_triple_straddle",
                bigquery_column="c1_transition_probability",
                labels={"category": "regime", "priority": "critical"}
            )
        ]
        
        # Component 2: Greeks & Sentiment (4 features)
        c2_features = [
            FeatureDefinition(
                feature_id="c2_gamma_exposure",
                display_name="Gamma Exposure",
                description="Market gamma exposure level affecting price sensitivity",
                value_type=FeatureType.DOUBLE,
                component_source="component_02_greeks_sentiment",
                bigquery_column="c2_gamma_exposure",
                labels={"category": "greeks", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c2_sentiment_level",
                display_name="Sentiment Level",
                description="Options sentiment classification from flow analysis",
                value_type=FeatureType.DOUBLE,
                component_source="component_02_greeks_sentiment",
                bigquery_column="c2_sentiment_level",
                labels={"category": "sentiment", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c2_pin_risk_score",
                display_name="Pin Risk Score", 
                description="Pin risk assessment score for expiry scenarios",
                value_type=FeatureType.DOUBLE,
                component_source="component_02_greeks_sentiment",
                bigquery_column="c2_pin_risk_score",
                labels={"category": "risk", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c2_max_pain_level",
                display_name="Max Pain Level",
                description="Maximum pain level for options positioning",
                value_type=FeatureType.DOUBLE,
                component_source="component_02_greeks_sentiment",
                bigquery_column="c2_max_pain_level",
                labels={"category": "positioning", "priority": "medium"}
            )
        ]
        
        # Component 3: OI & PA Trending (4 features)
        c3_features = [
            FeatureDefinition(
                feature_id="c3_institutional_flow_score",
                display_name="Institutional Flow Score",
                description="Institutional flow direction and strength score",
                value_type=FeatureType.DOUBLE,
                component_source="component_03_oi_pa_trending",
                bigquery_column="c3_institutional_flow_score",
                labels={"category": "flow", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c3_divergence_type",
                display_name="Divergence Type",
                description="Volume/OI divergence classification (bullish/bearish/neutral)",
                value_type=FeatureType.STRING,
                component_source="component_03_oi_pa_trending",
                bigquery_column="c3_divergence_type",
                labels={"category": "divergence", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c3_range_expansion_score",
                display_name="Range Expansion Score",
                description="Probability and strength of range expansion",
                value_type=FeatureType.DOUBLE,
                component_source="component_03_oi_pa_trending", 
                bigquery_column="c3_range_expansion_score",
                labels={"category": "range", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c3_volume_profile", 
                display_name="Volume Profile",
                description="Volume profile classification (accumulation/distribution/balanced)",
                value_type=FeatureType.STRING,
                component_source="component_03_oi_pa_trending",
                bigquery_column="c3_volume_profile",
                labels={"category": "volume", "priority": "medium"}
            )
        ]
        
        # Component 4: IV Skew Analysis (4 features)
        c4_features = [
            FeatureDefinition(
                feature_id="c4_skew_bias_score",
                display_name="Skew Bias Score", 
                description="IV skew bias directional score (put/call weighted)",
                value_type=FeatureType.DOUBLE,
                component_source="component_04_iv_skew",
                bigquery_column="c4_skew_bias_score",
                labels={"category": "skew", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c4_term_structure_signal",
                display_name="Term Structure Signal",
                description="Volatility term structure signal strength",
                value_type=FeatureType.DOUBLE,
                component_source="component_04_iv_skew",
                bigquery_column="c4_term_structure_signal",
                labels={"category": "term_structure", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c4_iv_regime_level",
                display_name="IV Regime Level", 
                description="Implied volatility regime classification level",
                value_type=FeatureType.DOUBLE,
                component_source="component_04_iv_skew",
                bigquery_column="c4_iv_regime_level",
                labels={"category": "regime", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c4_volatility_rank",
                display_name="Volatility Rank",
                description="Historical volatility percentile ranking",
                value_type=FeatureType.DOUBLE,
                component_source="component_04_iv_skew",
                bigquery_column="c4_volatility_rank",
                labels={"category": "volatility", "priority": "high"}
            )
        ]
        
        # Component 5: ATR-EMA-CPR (4 features)
        c5_features = [
            FeatureDefinition(
                feature_id="c5_momentum_score",
                display_name="Technical Momentum Score",
                description="Technical momentum score from ATR-EMA analysis",
                value_type=FeatureType.DOUBLE,
                component_source="component_05_atr_ema_cpr",
                bigquery_column="c5_momentum_score", 
                labels={"category": "technical", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c5_volatility_regime_score",
                display_name="Volatility Regime Score",
                description="Volatility regime assessment from technical indicators",
                value_type=FeatureType.DOUBLE,
                component_source="component_05_atr_ema_cpr",
                bigquery_column="c5_volatility_regime_score",
                labels={"category": "regime", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c5_confluence_score",
                display_name="Confluence Score",
                description="Technical confluence score across ATR/EMA/CPR signals",
                value_type=FeatureType.DOUBLE,
                component_source="component_05_atr_ema_cpr",
                bigquery_column="c5_confluence_score",
                labels={"category": "confluence", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c5_trend_strength",
                display_name="Trend Strength",
                description="Trend strength indicator from technical analysis", 
                value_type=FeatureType.DOUBLE,
                component_source="component_05_atr_ema_cpr",
                bigquery_column="c5_trend_strength",
                labels={"category": "trend", "priority": "medium"}
            )
        ]
        
        # Component 6: Correlation Analysis (4 features) 
        c6_features = [
            FeatureDefinition(
                feature_id="c6_correlation_agreement_score",
                display_name="Correlation Agreement Score",
                description="Cross-component correlation agreement score",
                value_type=FeatureType.DOUBLE,
                component_source="component_06_correlation",
                bigquery_column="c6_correlation_agreement_score",
                labels={"category": "correlation", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c6_breakdown_alert",
                display_name="Breakdown Alert",
                description="Correlation breakdown alert flag (boolean)",
                value_type=FeatureType.BOOLEAN,
                component_source="component_06_correlation",
                bigquery_column="c6_breakdown_alert",
                labels={"category": "alert", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c6_system_stability_score",
                display_name="System Stability Score",
                description="Overall system stability and reliability score",
                value_type=FeatureType.DOUBLE,
                component_source="component_06_correlation",
                bigquery_column="c6_system_stability_score",
                labels={"category": "stability", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c6_prediction_confidence", 
                display_name="Prediction Confidence",
                description="Model prediction confidence level",
                value_type=FeatureType.DOUBLE,
                component_source="component_06_correlation",
                bigquery_column="c6_prediction_confidence",
                labels={"category": "confidence", "priority": "critical"}
            )
        ]
        
        # Component 7: Support/Resistance (4 features)
        c7_features = [
            FeatureDefinition(
                feature_id="c7_level_strength_score",
                display_name="Level Strength Score",
                description="Support/resistance level strength assessment",
                value_type=FeatureType.DOUBLE,
                component_source="component_07_support_resistance",
                bigquery_column="c7_level_strength_score",
                labels={"category": "levels", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c7_breakout_probability",
                display_name="Level Breakout Probability",
                description="Probability of support/resistance level breakout",
                value_type=FeatureType.DOUBLE,
                component_source="component_07_support_resistance",
                bigquery_column="c7_breakout_probability",
                labels={"category": "probability", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c7_support_confluence",
                display_name="Support Confluence",
                description="Support level confluence score from multiple indicators",
                value_type=FeatureType.DOUBLE,
                component_source="component_07_support_resistance",
                bigquery_column="c7_support_confluence",
                labels={"category": "support", "priority": "high"}
            ),
            FeatureDefinition(
                feature_id="c7_resistance_confluence",
                display_name="Resistance Confluence", 
                description="Resistance level confluence score from multiple indicators",
                value_type=FeatureType.DOUBLE,
                component_source="component_07_support_resistance",
                bigquery_column="c7_resistance_confluence",
                labels={"category": "resistance", "priority": "high"}
            )
        ]
        
        # Component 8: Master Integration (4 features)
        c8_features = [
            FeatureDefinition(
                feature_id="c8_component_agreement_score",
                display_name="Component Agreement Score",
                description="Overall agreement score across all 8 components",
                value_type=FeatureType.DOUBLE,
                component_source="component_08_master_integration",
                bigquery_column="c8_component_agreement_score",
                labels={"category": "agreement", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c8_integration_confidence",
                display_name="Integration Confidence",
                description="Master integration confidence level",
                value_type=FeatureType.DOUBLE,
                component_source="component_08_master_integration",
                bigquery_column="c8_integration_confidence",
                labels={"category": "confidence", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c8_transition_probability_hint",
                display_name="Transition Probability Hint",
                description="Market regime transition probability hint from integration",
                value_type=FeatureType.DOUBLE,
                component_source="component_08_master_integration",
                bigquery_column="c8_transition_probability_hint",
                labels={"category": "transition", "priority": "critical"}
            ),
            FeatureDefinition(
                feature_id="c8_regime_classification",
                display_name="Regime Classification",
                description="Final market regime classification (trending_up/trending_down/ranging/volatile)",
                value_type=FeatureType.STRING,
                component_source="component_08_master_integration",
                bigquery_column="c8_regime_classification",
                labels={"category": "classification", "priority": "critical"}
            )
        ]
        
        # Create feature groups
        self.feature_groups = {
            "c1": ComponentFeatureGroup("c1", "Triple Rolling Straddle", c1_features, priority=1),
            "c2": ComponentFeatureGroup("c2", "Greeks & Sentiment", c2_features, priority=1), 
            "c3": ComponentFeatureGroup("c3", "OI & PA Trending", c3_features, priority=2),
            "c4": ComponentFeatureGroup("c4", "IV Skew Analysis", c4_features, priority=1),
            "c5": ComponentFeatureGroup("c5", "ATR-EMA-CPR", c5_features, priority=2),
            "c6": ComponentFeatureGroup("c6", "Correlation Analysis", c6_features, priority=1),
            "c7": ComponentFeatureGroup("c7", "Support/Resistance", c7_features, priority=2),
            "c8": ComponentFeatureGroup("c8", "Master Integration", c8_features, priority=1)
        }
        
        # Build unified feature dictionary
        for group in self.feature_groups.values():
            for feature in group.features:
                self.all_features[feature.feature_id] = feature
                
        self.logger.info(f"Initialized {len(self.all_features)} online features across {len(self.feature_groups)} components")
    
    def get_all_online_features(self) -> List[FeatureDefinition]:
        """Get all 32 online features"""
        return list(self.all_features.values())
    
    def get_feature_by_id(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Get feature definition by ID"""
        return self.all_features.get(feature_id)
    
    def get_features_by_component(self, component_id: str) -> List[FeatureDefinition]:
        """Get all features for a specific component"""
        group = self.feature_groups.get(component_id)
        return group.features if group else []
    
    def get_critical_features(self) -> List[FeatureDefinition]:
        """Get features marked as critical priority"""
        return [f for f in self.all_features.values() if f.labels.get("priority") == "critical"]
    
    def get_high_priority_features(self) -> List[FeatureDefinition]:
        """Get features marked as high or critical priority"""
        return [f for f in self.all_features.values() if f.labels.get("priority") in ["high", "critical"]]
    
    def generate_feature_store_yaml(self) -> str:
        """Generate YAML configuration for Feature Store deployment"""
        
        config = {
            "feature_store_configuration": {
                "name": "market-regime-feature-store",
                "description": "32 core online features for market regime classification",
                "online_serving_config": {
                    "fixed_node_count": 2
                },
                "labels": {
                    "environment": "production",
                    "system": "market-regime",
                    "features": "32_core_online"
                }
            },
            "entity_type": {
                "entity_type_id": "instrument_minute", 
                "description": "Market instruments with minute-level granularity",
                "entity_id_columns": ["entity_id"]
            },
            "features": {}
        }
        
        # Add all feature definitions
        for feature_id, feature_def in self.all_features.items():
            config["features"][feature_id] = {
                "value_type": feature_def.value_type.value,
                "description": feature_def.description,
                "monitoring_config": {
                    "snapshot_analysis": {
                        "disabled": not feature_def.monitoring_enabled
                    }
                },
                "labels": feature_def.labels
            }
        
        return yaml.dump(config, default_flow_style=False, sort_keys=True)
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about the feature mapping"""
        
        stats = {
            "total_features": len(self.all_features),
            "target_features": 32,
            "feature_coverage": len(self.all_features) >= 32,
            "components": len(self.feature_groups),
            "features_per_component": {},
            "priority_distribution": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "type_distribution": {},
            "category_distribution": {}
        }
        
        # Features per component
        for group_id, group in self.feature_groups.items():
            stats["features_per_component"][group_id] = group.get_feature_count()
        
        # Priority and type distribution
        for feature in self.all_features.values():
            # Priority distribution
            priority = feature.labels.get("priority", "medium")
            if priority in stats["priority_distribution"]:
                stats["priority_distribution"][priority] += 1
            
            # Type distribution
            type_name = feature.value_type.value
            stats["type_distribution"][type_name] = stats["type_distribution"].get(type_name, 0) + 1
            
            # Category distribution
            category = feature.labels.get("category", "general")
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
        
        return stats


# Example usage and validation
def main():
    """Example usage of Feature Store Mapping"""
    
    # Initialize mapping system
    mapping = ComprehensiveFeatureStoreMapping()
    
    # Get statistics
    stats = mapping.get_feature_statistics()
    print("Feature Mapping Statistics:")
    print(f"Total Features: {stats['total_features']}")
    print(f"Components: {stats['components']}")
    print(f"Features per Component: {stats['features_per_component']}")
    print(f"Priority Distribution: {stats['priority_distribution']}")
    
    # Get critical features
    critical_features = mapping.get_critical_features()
    print(f"\nCritical Features ({len(critical_features)}):")
    for feature in critical_features:
        print(f"  - {feature.feature_id}: {feature.description}")
    
    # Generate Feature Store YAML
    yaml_config = mapping.generate_feature_store_yaml()
    print(f"\nGenerated Feature Store YAML ({len(yaml_config)} characters)")
    
    # Validate that we have exactly 32 features
    assert len(mapping.get_all_online_features()) == 32, f"Expected 32 features, got {len(mapping.get_all_online_features())}"
    print("✓ Validation passed: 32 core online features defined")


if __name__ == "__main__":
    main()