"""
Feature Registration for Market Regime Feature Store
Story 2.6: Minimal Online Feature Registration - Task 2

Handles registration of 32 core online features across components C1-C8:
- 4 critical features per component
- Automated registration with Vertex AI Feature Store
- Feature validation and verification
- Registration status tracking
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
import yaml
import time

logger = logging.getLogger(__name__)


@dataclass
class FeatureRegistrationResult:
    """Result of feature registration operation"""
    feature_id: str
    component: str
    success: bool
    feature_path: Optional[str] = None
    error: Optional[str] = None
    registration_time: Optional[float] = None


class FeatureRegistration:
    """
    Manages registration of core online features for Market Regime system.
    
    Handles registration of 32 critical features (4 per component):
    - C1: Triple Straddle features
    - C2: Greeks & Sentiment features  
    - C3: OI & PA Trending features
    - C4: IV Skew features
    - C5: ATR-EMA-CPR features
    - C6: Correlation features
    - C7: Support/Resistance features
    - C8: Master Integration features
    """
    
    def __init__(self, config_path: str):
        """Initialize Feature Registration with configuration"""
        self.config = self._load_config(config_path)
        self.project_id = self.config['project_config']['project_id']
        self.location = self.config['project_config']['location']
        self.featurestore_id = self.config['feature_store']['featurestore_id']
        
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project_id,
            location=self.location
        )
        
        self.featurestore_client = gapic.FeaturestoreServiceClient()
        self.featurestore_path = self.featurestore_client.featurestore_path(
            project=self.project_id,
            location=self.location,
            featurestore=self.featurestore_id
        )
        
        # Core features mapping (32 total - 4 per component)
        self.core_features = self._get_core_feature_mapping()
        
        logger.info(f"Feature Registration initialized for {len(self.core_features)} core features")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _get_core_feature_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Get mapping of 32 core features (4 per component).
        
        Based on Epic 1 component specifications and feature importance.
        
        Returns:
            Dict[str, Dict[str, Any]]: Feature ID -> feature configuration
        """
        entity_config = self.config['feature_store']['entity_types']['instrument_minute']
        online_features = entity_config.get('online_features', {})
        
        return online_features
    
    def get_features_by_component(self) -> Dict[str, List[str]]:
        """
        Get features organized by component.
        
        Returns:
            Dict[str, List[str]]: Component -> list of feature IDs
        """
        component_features = {}
        
        for feature_id in self.core_features.keys():
            component = feature_id.split('_')[0]  # e.g., 'c1' from 'c1_momentum_score'
            
            if component not in component_features:
                component_features[component] = []
            
            component_features[component].append(feature_id)
        
        return component_features
    
    def validate_feature_selection(self) -> Dict[str, Any]:
        """
        Validate that feature selection meets requirements.
        
        Requirements:
        - Exactly 32 core features
        - 4 features per component (C1-C8)
        - Critical features for real-time market regime classification
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'total_features': len(self.core_features),
            'expected_features': 32,
            'components_validated': {},
            'validation_passed': True,
            'issues': []
        }
        
        # Check total feature count
        if validation_results['total_features'] != validation_results['expected_features']:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(
                f"Expected 32 features, found {validation_results['total_features']}"
            )
        
        # Check features per component
        component_features = self.get_features_by_component()
        expected_components = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
        expected_features_per_component = 4
        
        for component in expected_components:
            component_feature_count = len(component_features.get(component, []))
            
            validation_results['components_validated'][component] = {
                'feature_count': component_feature_count,
                'expected_count': expected_features_per_component,
                'features': component_features.get(component, []),
                'valid': component_feature_count == expected_features_per_component
            }
            
            if component_feature_count != expected_features_per_component:
                validation_results['validation_passed'] = False
                validation_results['issues'].append(
                    f"Component {component}: expected {expected_features_per_component} features, "
                    f"found {component_feature_count}"
                )
        
        logger.info(f"Feature selection validation: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        return validation_results
    
    def register_all_features(self) -> Dict[str, Any]:
        """
        Register all 32 core features with the Feature Store.
        
        Returns:
            Dict[str, Any]: Registration results for all features
        """
        registration_results = {
            'total_features': len(self.core_features),
            'successful_registrations': 0,
            'failed_registrations': 0,
            'results': [],
            'summary_by_component': {},
            'registration_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Get entity type path
            entity_type_path = self.featurestore_client.entity_type_path(
                project=self.project_id,
                location=self.location,
                featurestore=self.featurestore_id,
                entity_type='instrument_minute'
            )
            
            # Register features by component
            component_features = self.get_features_by_component()
            
            for component, feature_ids in component_features.items():
                component_results = self._register_component_features(
                    entity_type_path, component, feature_ids
                )
                
                registration_results['results'].extend(component_results['results'])
                registration_results['successful_registrations'] += component_results['successful_count']
                registration_results['failed_registrations'] += component_results['failed_count']
                registration_results['summary_by_component'][component] = component_results
            
            registration_results['registration_time'] = time.time() - start_time
            
            logger.info(
                f"Feature registration complete: {registration_results['successful_registrations']}/"
                f"{registration_results['total_features']} successful"
            )
            
            return registration_results
            
        except Exception as e:
            logger.error(f"Failed to register features: {e}")
            registration_results['error'] = str(e)
            return registration_results
    
    def _register_component_features(
        self, 
        entity_type_path: str, 
        component: str, 
        feature_ids: List[str]
    ) -> Dict[str, Any]:
        """Register features for a specific component"""
        component_results = {
            'component': component,
            'total_features': len(feature_ids),
            'successful_count': 0,
            'failed_count': 0,
            'results': []
        }
        
        logger.info(f"Registering {len(feature_ids)} features for component {component}")
        
        for feature_id in feature_ids:
            result = self._register_single_feature(entity_type_path, feature_id)
            component_results['results'].append(result)
            
            if result.success:
                component_results['successful_count'] += 1
            else:
                component_results['failed_count'] += 1
        
        logger.info(
            f"Component {component}: {component_results['successful_count']}/"
            f"{component_results['total_features']} features registered"
        )
        
        return component_results
    
    def _register_single_feature(self, entity_type_path: str, feature_id: str) -> FeatureRegistrationResult:
        """Register a single feature with the Feature Store"""
        start_time = time.time()
        
        try:
            feature_config = self.core_features[feature_id]
            component = feature_id.split('_')[0]
            
            # Check if feature already exists
            feature_path = f"{entity_type_path}/features/{feature_id}"
            try:
                existing_feature = self.featurestore_client.get_feature(name=feature_path)
                logger.info(f"Feature {feature_id} already exists: {existing_feature.name}")
                return FeatureRegistrationResult(
                    feature_id=feature_id,
                    component=component,
                    success=True,
                    feature_path=existing_feature.name,
                    registration_time=time.time() - start_time
                )
            except Exception:
                # Feature doesn't exist, proceed with creation
                pass
            
            # Map value types
            value_type_mapping = {
                'DOUBLE': gapic.Feature.ValueType.DOUBLE,
                'STRING': gapic.Feature.ValueType.STRING,
                'BOOLEAN': gapic.Feature.ValueType.BOOL,
                'INT64': gapic.Feature.ValueType.INT64
            }
            
            value_type = value_type_mapping.get(
                feature_config['value_type'], 
                gapic.Feature.ValueType.DOUBLE
            )
            
            # Create feature
            feature = gapic.Feature(
                value_type=value_type,
                description=feature_config['description'],
                labels={
                    'component': component,
                    'ttl-hours': str(feature_config['ttl_hours']),
                    'story': 'story-2-6',
                    'critical': 'true'
                }
            )
            
            operation = self.featurestore_client.create_feature(
                parent=entity_type_path,
                feature=feature,
                feature_id=feature_id
            )
            
            result = operation.result(timeout=300)  # 5 minute timeout
            
            logger.info(f"Feature {feature_id} registered successfully")
            return FeatureRegistrationResult(
                feature_id=feature_id,
                component=component,
                success=True,
                feature_path=result.name,
                registration_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to register feature {feature_id}: {e}")
            return FeatureRegistrationResult(
                feature_id=feature_id,
                component=component,
                success=False,
                error=str(e),
                registration_time=time.time() - start_time
            )
    
    def verify_feature_registration(self) -> Dict[str, Any]:
        """
        Verify that all core features are properly registered.
        
        Returns:
            Dict[str, Any]: Verification results
        """
        verification_results = {
            'total_expected': len(self.core_features),
            'total_found': 0,
            'verified_features': [],
            'missing_features': [],
            'verification_passed': True,
            'component_summary': {}
        }
        
        try:
            entity_type_path = self.featurestore_client.entity_type_path(
                project=self.project_id,
                location=self.location,
                featurestore=self.featurestore_id,
                entity_type='instrument_minute'
            )
            
            # List all features in the entity type
            features = self.featurestore_client.list_features(parent=entity_type_path)
            existing_feature_ids = set()
            
            for feature in features:
                feature_id = feature.name.split('/')[-1]
                existing_feature_ids.add(feature_id)
            
            # Check each core feature
            component_features = self.get_features_by_component()
            
            for component, feature_ids in component_features.items():
                component_verified = 0
                component_missing = []
                
                for feature_id in feature_ids:
                    if feature_id in existing_feature_ids:
                        verification_results['verified_features'].append(feature_id)
                        verification_results['total_found'] += 1
                        component_verified += 1
                    else:
                        verification_results['missing_features'].append(feature_id)
                        component_missing.append(feature_id)
                        verification_results['verification_passed'] = False
                
                verification_results['component_summary'][component] = {
                    'expected': len(feature_ids),
                    'verified': component_verified,
                    'missing': component_missing
                }
            
            logger.info(
                f"Feature verification: {verification_results['total_found']}/"
                f"{verification_results['total_expected']} features verified"
            )
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Failed to verify feature registration: {e}")
            verification_results['verification_passed'] = False
            verification_results['error'] = str(e)
            return verification_results
    
    def get_registration_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report of feature registration.
        
        Returns:
            Dict[str, Any]: Complete registration status report
        """
        try:
            # Validate feature selection
            validation_results = self.validate_feature_selection()
            
            # Verify current registration status
            verification_results = self.verify_feature_registration()
            
            # Get component breakdown
            component_features = self.get_features_by_component()
            
            report = {
                'summary': {
                    'total_core_features': len(self.core_features),
                    'components': len(component_features),
                    'features_per_component': 4,
                    'selection_valid': validation_results['validation_passed'],
                    'registration_complete': verification_results['verification_passed'],
                    'features_registered': verification_results['total_found'],
                    'features_missing': len(verification_results['missing_features'])
                },
                'feature_selection_validation': validation_results,
                'registration_verification': verification_results,
                'component_breakdown': component_features,
                'core_feature_details': self.core_features,
                'generated_at': time.time()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate registration status report: {e}")
            return {'error': str(e)}
    
    def get_feature_importance_ranking(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get feature importance ranking for each component.
        
        Based on Epic 1 component specifications and real-time serving requirements.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Component -> ranked features
        """
        feature_importance = {
            'c1': [
                {'feature': 'c1_momentum_score', 'importance': 1, 'reason': 'Primary momentum indicator'},
                {'feature': 'c1_breakout_probability', 'importance': 2, 'reason': 'Regime transition signal'},
                {'feature': 'c1_transition_probability', 'importance': 3, 'reason': 'Regime change indicator'},
                {'feature': 'c1_vol_compression', 'importance': 4, 'reason': 'Volume compression signal'}
            ],
            'c2': [
                {'feature': 'c2_gamma_exposure', 'importance': 1, 'reason': 'Market gamma risk indicator'},
                {'feature': 'c2_sentiment_level', 'importance': 2, 'reason': 'Options sentiment classification'},
                {'feature': 'c2_pin_risk_score', 'importance': 3, 'reason': 'Pin risk assessment'},
                {'feature': 'c2_max_pain_level', 'importance': 4, 'reason': 'Max pain analysis'}
            ],
            'c3': [
                {'feature': 'c3_institutional_flow_score', 'importance': 1, 'reason': 'Smart money flow indicator'},
                {'feature': 'c3_range_expansion_score', 'importance': 2, 'reason': 'Range breakout predictor'},
                {'feature': 'c3_divergence_type', 'importance': 3, 'reason': 'Volume/OI divergence signal'},
                {'feature': 'c3_volume_profile', 'importance': 4, 'reason': 'Volume profile classification'}
            ],
            'c4': [
                {'feature': 'c4_iv_regime_level', 'importance': 1, 'reason': 'IV regime classification'},
                {'feature': 'c4_skew_bias_score', 'importance': 2, 'reason': 'Skew directional bias'},
                {'feature': 'c4_volatility_rank', 'importance': 3, 'reason': 'Historical volatility context'},
                {'feature': 'c4_term_structure_signal', 'importance': 4, 'reason': 'Term structure signal'}
            ],
            'c5': [
                {'feature': 'c5_momentum_score', 'importance': 1, 'reason': 'Technical momentum indicator'},
                {'feature': 'c5_volatility_regime_score', 'importance': 2, 'reason': 'Volatility regime assessment'},
                {'feature': 'c5_confluence_score', 'importance': 3, 'reason': 'Technical confluence indicator'},
                {'feature': 'c5_trend_strength', 'importance': 4, 'reason': 'Trend strength measurement'}
            ],
            'c6': [
                {'feature': 'c6_system_stability_score', 'importance': 1, 'reason': 'Overall system stability'},
                {'feature': 'c6_correlation_agreement_score', 'importance': 2, 'reason': 'Component agreement level'},
                {'feature': 'c6_prediction_confidence', 'importance': 3, 'reason': 'Prediction confidence metric'},
                {'feature': 'c6_breakdown_alert', 'importance': 4, 'reason': 'Correlation breakdown warning'}
            ],
            'c7': [
                {'feature': 'c7_level_strength_score', 'importance': 1, 'reason': 'Support/resistance strength'},
                {'feature': 'c7_breakout_probability', 'importance': 2, 'reason': 'Level breakout probability'},
                {'feature': 'c7_support_confluence', 'importance': 3, 'reason': 'Support level confluence'},
                {'feature': 'c7_resistance_confluence', 'importance': 4, 'reason': 'Resistance level confluence'}
            ],
            'c8': [
                {'feature': 'c8_regime_classification', 'importance': 1, 'reason': 'Final regime classification'},
                {'feature': 'c8_integration_confidence', 'importance': 2, 'reason': 'Integration confidence level'},
                {'feature': 'c8_component_agreement_score', 'importance': 3, 'reason': 'Component agreement score'},
                {'feature': 'c8_transition_probability_hint', 'importance': 4, 'reason': 'Transition probability hint'}
            ]
        }
        
        return feature_importance