"""
TTL Policy Manager for Market Regime Feature Store
Story 2.6: Minimal Online Feature Registration - Subtask 1.3

Manages TTL (Time-To-Live) policies for online features:
- Default 48-hour TTL for all online features
- Component-specific TTL configurations
- TTL validation and optimization
- Cleanup and maintenance scheduling
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TTLPolicy:
    """TTL policy configuration for features"""
    feature_id: str
    ttl_hours: int
    description: str
    component: str
    priority: str = "normal"  # normal, high, critical


class TTLPolicyManager:
    """
    Manages TTL policies for Feature Store online features.
    
    Default Configuration:
    - Standard TTL: 48 hours for all online features
    - Component-based policies supported
    - Automatic cleanup scheduling
    - Performance optimization based on access patterns
    """
    
    # Default TTL configurations
    DEFAULT_TTL_HOURS = 48
    MIN_TTL_HOURS = 1
    MAX_TTL_HOURS = 168  # 7 days
    
    # Component-specific TTL recommendations
    COMPONENT_TTL_RECOMMENDATIONS = {
        'c1': 48,  # Triple straddle - standard
        'c2': 48,  # Greeks sentiment - standard  
        'c3': 48,  # OI/PA trending - standard
        'c4': 48,  # IV skew - standard
        'c5': 48,  # ATR-EMA-CPR - standard
        'c6': 48,  # Correlation - standard
        'c7': 48,  # Support/Resistance - standard
        'c8': 48   # Master integration - standard
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize TTL Policy Manager"""
        self.config = None
        if config_path:
            self.config = self._load_config(config_path)
        
        self.ttl_policies: Dict[str, TTLPolicy] = {}
        self._initialize_default_policies()
        
        logger.info("TTL Policy Manager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded TTL configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load TTL config from {config_path}: {e}")
            raise
    
    def _initialize_default_policies(self) -> None:
        """Initialize default TTL policies for all components"""
        # If we have config, use it; otherwise use defaults
        if self.config and 'feature_store' in self.config:
            self._load_policies_from_config()
        else:
            self._create_default_policies()
    
    def _load_policies_from_config(self) -> None:
        """Load TTL policies from configuration file"""
        try:
            entity_types = self.config['feature_store']['entity_types']
            
            for entity_type_id, entity_config in entity_types.items():
                online_features = entity_config.get('online_features', {})
                
                for feature_id, feature_config in online_features.items():
                    ttl_hours = feature_config.get('ttl_hours', self.DEFAULT_TTL_HOURS)
                    description = feature_config.get('description', '')
                    component = self._extract_component_from_feature_id(feature_id)
                    
                    policy = TTLPolicy(
                        feature_id=feature_id,
                        ttl_hours=ttl_hours,
                        description=description,
                        component=component
                    )
                    
                    self.ttl_policies[feature_id] = policy
            
            logger.info(f"Loaded {len(self.ttl_policies)} TTL policies from configuration")
            
        except Exception as e:
            logger.error(f"Failed to load TTL policies from config: {e}")
            self._create_default_policies()
    
    def _create_default_policies(self) -> None:
        """Create default TTL policies for standard feature set"""
        default_features = [
            # Component 1: Triple Rolling Straddle
            ('c1_momentum_score', 'Momentum score from straddle analysis'),
            ('c1_vol_compression', 'Volume compression indicator'),
            ('c1_breakout_probability', 'Probability of breakout from current range'),
            ('c1_transition_probability', 'Regime transition probability'),
            
            # Component 2: Greeks & Sentiment
            ('c2_gamma_exposure', 'Market gamma exposure level'),
            ('c2_sentiment_level', 'Options sentiment classification'),
            ('c2_pin_risk_score', 'Pin risk assessment score'),
            ('c2_max_pain_level', 'Max pain level indicator'),
            
            # Component 3: OI & PA Trending
            ('c3_institutional_flow_score', 'Institutional flow direction score'),
            ('c3_divergence_type', 'Volume/OI divergence classification'),
            ('c3_range_expansion_score', 'Range expansion probability'),
            ('c3_volume_profile', 'Volume profile classification'),
            
            # Component 4: IV Skew Analysis
            ('c4_skew_bias_score', 'Skew bias directional score'),
            ('c4_term_structure_signal', 'Term structure signal strength'),
            ('c4_iv_regime_level', 'IV regime classification level'),
            ('c4_volatility_rank', 'Historical volatility rank'),
            
            # Component 5: ATR-EMA-CPR
            ('c5_momentum_score', 'Technical momentum score'),
            ('c5_volatility_regime_score', 'Volatility regime assessment'),
            ('c5_confluence_score', 'Technical confluence score'),
            ('c5_trend_strength', 'Trend strength indicator'),
            
            # Component 6: Correlation Analysis
            ('c6_correlation_agreement_score', 'Cross-component correlation agreement'),
            ('c6_breakdown_alert', 'Correlation breakdown alert flag'),
            ('c6_system_stability_score', 'Overall system stability score'),
            ('c6_prediction_confidence', 'Prediction confidence level'),
            
            # Component 7: Support/Resistance
            ('c7_level_strength_score', 'Support/resistance level strength'),
            ('c7_breakout_probability', 'Level breakout probability'),
            ('c7_support_confluence', 'Support level confluence score'),
            ('c7_resistance_confluence', 'Resistance level confluence score'),
            
            # Component 8: Master Integration
            ('c8_component_agreement_score', 'Overall component agreement score'),
            ('c8_integration_confidence', 'Integration confidence level'),
            ('c8_transition_probability_hint', 'Market transition probability hint'),
            ('c8_regime_classification', 'Final regime classification'),
        ]
        
        for feature_id, description in default_features:
            component = self._extract_component_from_feature_id(feature_id)
            ttl_hours = self.COMPONENT_TTL_RECOMMENDATIONS.get(component, self.DEFAULT_TTL_HOURS)
            
            policy = TTLPolicy(
                feature_id=feature_id,
                ttl_hours=ttl_hours,
                description=description,
                component=component
            )
            
            self.ttl_policies[feature_id] = policy
        
        logger.info(f"Created {len(self.ttl_policies)} default TTL policies")
    
    def _extract_component_from_feature_id(self, feature_id: str) -> str:
        """Extract component identifier from feature ID"""
        # Feature IDs follow pattern: c{X}_{feature_name}
        if '_' in feature_id:
            return feature_id.split('_')[0]
        return 'unknown'
    
    def get_ttl_for_feature(self, feature_id: str) -> int:
        """
        Get TTL in hours for a specific feature.
        
        Args:
            feature_id: Feature identifier
            
        Returns:
            int: TTL in hours
        """
        if feature_id in self.ttl_policies:
            return self.ttl_policies[feature_id].ttl_hours
        
        # Fallback to component default
        component = self._extract_component_from_feature_id(feature_id)
        return self.COMPONENT_TTL_RECOMMENDATIONS.get(component, self.DEFAULT_TTL_HOURS)
    
    def set_ttl_for_feature(self, feature_id: str, ttl_hours: int, description: str = "") -> bool:
        """
        Set TTL policy for a specific feature.
        
        Args:
            feature_id: Feature identifier
            ttl_hours: TTL in hours
            description: Optional description
            
        Returns:
            bool: True if successful
        """
        try:
            self._validate_ttl_hours(ttl_hours)
            
            component = self._extract_component_from_feature_id(feature_id)
            
            policy = TTLPolicy(
                feature_id=feature_id,
                ttl_hours=ttl_hours,
                description=description,
                component=component
            )
            
            self.ttl_policies[feature_id] = policy
            logger.info(f"Set TTL policy for {feature_id}: {ttl_hours} hours")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set TTL for {feature_id}: {e}")
            return False
    
    def get_component_ttl_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get TTL summary by component.
        
        Returns:
            Dict[str, Dict[str, Any]]: Component -> TTL statistics
        """
        component_stats = {}
        
        for policy in self.ttl_policies.values():
            component = policy.component
            
            if component not in component_stats:
                component_stats[component] = {
                    'feature_count': 0,
                    'avg_ttl_hours': 0,
                    'min_ttl_hours': float('inf'),
                    'max_ttl_hours': 0,
                    'total_ttl_hours': 0,
                    'features': []
                }
            
            stats = component_stats[component]
            stats['feature_count'] += 1
            stats['total_ttl_hours'] += policy.ttl_hours
            stats['min_ttl_hours'] = min(stats['min_ttl_hours'], policy.ttl_hours)
            stats['max_ttl_hours'] = max(stats['max_ttl_hours'], policy.ttl_hours)
            stats['features'].append(policy.feature_id)
        
        # Calculate averages
        for component, stats in component_stats.items():
            if stats['feature_count'] > 0:
                stats['avg_ttl_hours'] = stats['total_ttl_hours'] / stats['feature_count']
            
            if stats['min_ttl_hours'] == float('inf'):
                stats['min_ttl_hours'] = 0
        
        return component_stats
    
    def validate_all_ttl_policies(self) -> Dict[str, Any]:
        """
        Validate all TTL policies.
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'valid_count': 0,
            'invalid_count': 0,
            'total_count': len(self.ttl_policies),
            'invalid_features': [],
            'warnings': []
        }
        
        for feature_id, policy in self.ttl_policies.items():
            try:
                self._validate_ttl_hours(policy.ttl_hours)
                validation_results['valid_count'] += 1
                
                # Check for recommendations
                component = policy.component
                recommended_ttl = self.COMPONENT_TTL_RECOMMENDATIONS.get(component, self.DEFAULT_TTL_HOURS)
                
                if policy.ttl_hours != recommended_ttl:
                    validation_results['warnings'].append(
                        f"Feature {feature_id} TTL ({policy.ttl_hours}h) differs from component recommendation ({recommended_ttl}h)"
                    )
                
            except ValueError as e:
                validation_results['invalid_count'] += 1
                validation_results['invalid_features'].append({
                    'feature_id': feature_id,
                    'ttl_hours': policy.ttl_hours,
                    'error': str(e)
                })
        
        logger.info(f"TTL validation: {validation_results['valid_count']}/{validation_results['total_count']} valid")
        return validation_results
    
    def get_cleanup_schedule(self) -> Dict[str, Any]:
        """
        Get recommended cleanup schedule based on TTL policies.
        
        Returns:
            Dict[str, Any]: Cleanup schedule configuration
        """
        min_ttl = min(policy.ttl_hours for policy in self.ttl_policies.values())
        max_ttl = max(policy.ttl_hours for policy in self.ttl_policies.values())
        
        # Schedule cleanup more frequently than the shortest TTL
        cleanup_interval_hours = max(1, min_ttl // 4)
        
        return {
            'cleanup_interval_hours': cleanup_interval_hours,
            'cleanup_cron': f"0 */{cleanup_interval_hours} * * *",
            'retention_period_hours': max_ttl + 24,  # Keep extra 24h buffer
            'min_feature_ttl': min_ttl,
            'max_feature_ttl': max_ttl,
            'recommended_monitoring_interval_minutes': cleanup_interval_hours * 30
        }
    
    def get_ttl_policy_export(self) -> Dict[str, Any]:
        """
        Export TTL policies for configuration or documentation.
        
        Returns:
            Dict[str, Any]: Exportable TTL policy configuration
        """
        export_data = {
            'ttl_policies': {},
            'component_summary': self.get_component_ttl_summary(),
            'cleanup_schedule': self.get_cleanup_schedule(),
            'validation_results': self.validate_all_ttl_policies(),
            'generated_at': datetime.now().isoformat()
        }
        
        for feature_id, policy in self.ttl_policies.items():
            export_data['ttl_policies'][feature_id] = {
                'ttl_hours': policy.ttl_hours,
                'description': policy.description,
                'component': policy.component,
                'priority': policy.priority
            }
        
        return export_data
    
    def _validate_ttl_hours(self, ttl_hours: int) -> None:
        """Validate TTL hours value"""
        if not isinstance(ttl_hours, int):
            raise ValueError("TTL hours must be an integer")
        
        if ttl_hours < self.MIN_TTL_HOURS:
            raise ValueError(f"TTL hours must be at least {self.MIN_TTL_HOURS}")
        
        if ttl_hours > self.MAX_TTL_HOURS:
            raise ValueError(f"TTL hours cannot exceed {self.MAX_TTL_HOURS}")
    
    def get_feature_count_by_ttl(self) -> Dict[int, int]:
        """Get count of features by TTL value"""
        ttl_counts = {}
        
        for policy in self.ttl_policies.values():
            ttl_hours = policy.ttl_hours
            ttl_counts[ttl_hours] = ttl_counts.get(ttl_hours, 0) + 1
        
        return ttl_counts