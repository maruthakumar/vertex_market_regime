"""
Component 08: Master Integration Analyzer

Main analyzer that orchestrates all sub-modules to generate 48 integration features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from dataclasses import asdict

from .feature_engine import MasterIntegrationFeatureEngine, IntegrationFeatures
from .component_aggregator import ComponentAggregator
from .dte_pattern_extractor import DTEPatternExtractor
from .correlation_analyzer import CorrelationAnalyzer
from .confidence_metrics import ConfidenceMetrics
from .synergy_detector import SynergyDetector

logger = logging.getLogger(__name__)


class Component08Analyzer:
    """
    Master Integration Analyzer
    
    Generates 48 cross-component integration features from Components 1-7 outputs.
    Pure feature engineering with no classification logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Component 08 analyzer
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Performance budgets
        self.processing_budget_ms = self.config.get('processing_budget_ms', 100)
        self.memory_budget_mb = self.config.get('memory_budget_mb', 150)
        
        # Initialize sub-modules
        self.feature_engine = MasterIntegrationFeatureEngine(config)
        self.aggregator = ComponentAggregator(config)
        self.dte_extractor = DTEPatternExtractor(config)
        self.correlation_analyzer = CorrelationAnalyzer(config)
        self.confidence_metrics = ConfidenceMetrics(config)
        self.synergy_detector = SynergyDetector(config)
        
        # Feature validation
        self.expected_feature_count = 48
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Initialized Component08Analyzer with {self.processing_budget_ms}ms budget")
    
    def analyze_master_integration(
        self,
        component_outputs: Dict[str, Any],
        dte: Optional[int] = None,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze master integration and generate 48 features
        
        Args:
            component_outputs: Outputs from Components 1-7
            dte: Days to expiry (optional)
            historical_data: Optional historical performance data
            
        Returns:
            Dictionary containing:
                - features: Dict of 48 integration features
                - metadata: Processing metadata
                - performance_metrics: Performance statistics
        """
        start_time = time.time()
        
        try:
            # Step 1: Aggregate and normalize component signals
            aggregated_components = self.aggregator.aggregate_components(component_outputs)
            aggregation_summary = self.aggregator.get_aggregation_summary(aggregated_components)
            
            # Step 2: Analyze correlations
            correlation_analysis = self.correlation_analyzer.analyze_correlations(
                aggregated_components
            )
            
            # Step 3: Extract DTE patterns
            dte_patterns = self.dte_extractor.extract_dte_patterns(
                aggregated_components,
                dte,
                historical_data
            )
            
            # Step 4: Calculate confidence metrics
            confidence_analysis = self.confidence_metrics.calculate_confidence_metrics(
                aggregated_components,
                correlation_analysis
            )
            
            # Step 5: Detect synergies
            synergy_analysis = self.synergy_detector.detect_synergies(
                aggregated_components
            )
            
            # Step 6: Build final 48 features
            features = self._build_48_features(
                aggregated_components,
                correlation_analysis,
                dte_patterns,
                confidence_analysis,
                synergy_analysis
            )
            
            # Calculate performance metrics
            processing_time = (time.time() - start_time) * 1000
            
            # Validate features
            is_valid = self._validate_features(features)
            
            # Build response
            result = {
                'features': features,
                'metadata': {
                    'dte': dte if dte is not None else 30,
                    'component_count': aggregation_summary['total_components'],
                    'healthy_components': aggregation_summary['healthy_components'],
                    'average_health': aggregation_summary['average_health'],
                    'confidence_level': confidence_analysis['overall_confidence']['confidence_level'],
                    'processing_time_ms': processing_time,
                    'is_valid': is_valid,
                    'feature_count': len(features)
                },
                'performance_metrics': {
                    'processing_time_ms': processing_time,
                    'within_budget': processing_time <= self.processing_budget_ms,
                    'feature_generation_rate': len(features) / (processing_time / 1000) if processing_time > 0 else 0,
                    'average_confidence': confidence_analysis['overall_confidence']['overall_confidence']
                },
                'component_health': {
                    comp_name: comp_data['health_score']
                    for comp_name, comp_data in aggregated_components.items()
                }
            }
            
            # Track performance
            self._track_performance(processing_time, is_valid)
            
            # Log if over budget
            if processing_time > self.processing_budget_ms:
                logger.warning(f"Processing time {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in master integration analysis: {str(e)}")
            # Return default features on error
            return self._get_default_result(str(e))
    
    def _build_48_features(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        correlation_analysis: Dict[str, Any],
        dte_patterns: Any,  # DTEPatterns object
        confidence_analysis: Dict[str, Any],
        synergy_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Build the final 48 integration features
        
        Args:
            aggregated_components: Aggregated component data
            correlation_analysis: Correlation analysis results
            dte_patterns: DTE pattern extraction results
            confidence_analysis: Confidence metrics results
            synergy_analysis: Synergy detection results
            
        Returns:
            Dictionary of 48 features
        """
        features = {}
        
        # 1. Component Agreement Features (12)
        agreement_features = self._extract_agreement_features(aggregated_components)
        features.update({
            'agreement_mean': agreement_features['pairwise_mean'],
            'agreement_std': agreement_features['pairwise_std'],
            'consensus_score': agreement_features['consensus'],
            'divergence_score': agreement_features['divergence'],
            'weighted_agreement': agreement_features['weighted'],
            'alignment_ratio': agreement_features['alignment'],
            'agreement_stability': agreement_features['stability'],
            'agreement_momentum': agreement_features['momentum'],
            'agreement_acceleration': agreement_features['acceleration'],
            'agreement_entropy': agreement_features['entropy'],
            'agreement_concentration': agreement_features['concentration'],
            'agreement_dispersion': agreement_features['dispersion']
        })
        
        # 2. DTE-Adaptive Features (12)
        features.update({
            'dte_specific': dte_patterns.dte_specific_patterns.get(
                dte_patterns.dte_specific_patterns.keys().__iter__().__next__() if dte_patterns.dte_specific_patterns else 30, 
                0.5
            ),
            'dte_weekly': dte_patterns.weekly_pattern,
            'dte_monthly': dte_patterns.monthly_pattern,
            'dte_far': dte_patterns.far_pattern,
            'dte_transition': dte_patterns.transition_score,
            'dte_evolution': dte_patterns.evolution_rate,
            'dte_perf_mean': dte_patterns.performance_mean,
            'dte_perf_std': dte_patterns.performance_std,
            'dte_reliability': dte_patterns.reliability_score,
            'dte_consistency': dte_patterns.consistency_score,
            'dte_adaptation': dte_patterns.adaptation_rate,
            'dte_efficiency': dte_patterns.efficiency_ratio
        })
        
        # 3. System Coherence Metrics (12)
        coherence_features = self.correlation_analyzer.get_correlation_features(correlation_analysis)
        confidence_features = self.confidence_metrics.get_confidence_features(confidence_analysis)
        
        features.update({
            'stability_score': confidence_features['system_stability'],
            'transition_prob': confidence_features['transition_probability'],
            'integration_quality': confidence_features['integration_quality'],
            'coherence_index': coherence_features['coherence_index'],
            'system_entropy': -coherence_features['correlation_mean'] * np.log(abs(coherence_features['correlation_mean']) + 1e-10),
            'signal_noise_ratio': abs(coherence_features['correlation_mean']) / (coherence_features['correlation_std'] + 1e-10),
            'confidence_mean': confidence_features['confidence_score'],
            'confidence_std': confidence_features.get('confidence_trend', 0.0),
            'health_score': confidence_features['health_mean'],
            'robustness_metric': confidence_features['robustness'],
            'consistency_score': confidence_features['quality_consistency'],
            'reliability_index': confidence_features['regime_stability']
        })
        
        # 4. Cross-Component Synergies (12)
        synergy_features = self.synergy_detector.get_synergy_features(synergy_analysis)
        
        features.update({
            'synergy_mean': synergy_features['synergy_mean'],
            'synergy_std': synergy_features['synergy_std'],
            'interaction_strength': synergy_features['interaction_strength'],
            'triad_synergy': synergy_features['triad_synergy_mean'],
            'complementary_ratio': synergy_features['complementary_ratio'],
            'antagonistic_ratio': synergy_features['antagonistic_ratio'],
            'synergy_concentration': synergy_features['synergy_concentration'],
            'synergy_dispersion': synergy_features['synergy_dispersion'],
            'synergy_momentum': synergy_features.get('synergy_mean', 0.0),  # Using mean as proxy
            'synergy_acceleration': synergy_features['interaction_variance'],
            'synergy_stability': synergy_features['synergy_stability'],
            'synergy_efficiency': synergy_features['synergy_efficiency']
        })
        
        # Ensure all values are floats and handle any NaN/Inf
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
            else:
                features[key] = float(value)
        
        return features
    
    def _extract_agreement_features(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Extract component agreement features
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary of agreement features
        """
        # Get primary signals from each component
        component_signals = []
        component_weights = []
        
        for comp_name, comp_data in aggregated_components.items():
            signals = comp_data.get('signals', {})
            if signals:
                # Get first signal as primary
                primary_signal = list(signals.values())[0]
                component_signals.append(primary_signal)
                
                # Use health score as weight
                weight = comp_data.get('health_score', 0.5)
                component_weights.append(weight)
        
        if not component_signals:
            return self._get_default_agreement_features()
        
        component_signals = np.array(component_signals)
        component_weights = np.array(component_weights)
        
        # Calculate pairwise agreements
        n = len(component_signals)
        agreements = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Agreement as correlation-like measure
                agreement = 1.0 - abs(component_signals[i] - component_signals[j]) / 2.0
                agreements.append(agreement)
        
        agreements = np.array(agreements) if agreements else np.array([0.5])
        
        return {
            'pairwise_mean': float(np.mean(agreements)),
            'pairwise_std': float(np.std(agreements)),
            'consensus': float(np.mean(component_signals > 0)),
            'divergence': float(np.std(component_signals)),
            'weighted': float(np.average(component_signals, weights=component_weights)),
            'alignment': float(np.sum(np.sign(component_signals) == np.sign(np.mean(component_signals))) / n),
            'stability': float(1.0 - np.std(agreements)),
            'momentum': float(np.mean(np.diff(agreements)) if len(agreements) > 1 else 0.0),
            'acceleration': float(np.mean(np.diff(np.diff(agreements))) if len(agreements) > 2 else 0.0),
            'entropy': float(-np.sum(agreements * np.log(agreements + 1e-10)) / len(agreements)),
            'concentration': float(np.max(agreements)),
            'dispersion': float(np.ptp(agreements))
        }
    
    def _get_default_agreement_features(self) -> Dict[str, float]:
        """Get default agreement features"""
        return {
            'pairwise_mean': 0.5,
            'pairwise_std': 0.0,
            'consensus': 0.5,
            'divergence': 0.0,
            'weighted': 0.0,
            'alignment': 0.5,
            'stability': 0.5,
            'momentum': 0.0,
            'acceleration': 0.0,
            'entropy': 0.0,
            'concentration': 0.5,
            'dispersion': 0.0
        }
    
    def _validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate the generated features
        
        Args:
            features: Dictionary of features
            
        Returns:
            True if valid, False otherwise
        """
        # Check feature count
        if len(features) != self.expected_feature_count:
            logger.error(f"Feature count mismatch: {len(features)} != {self.expected_feature_count}")
            return False
        
        # Check for invalid values
        for name, value in features.items():
            if np.isnan(value) or np.isinf(value):
                logger.error(f"Invalid feature value for {name}: {value}")
                return False
        
        # Check expected feature names
        expected_features = [
            'agreement_mean', 'agreement_std', 'consensus_score', 'divergence_score',
            'weighted_agreement', 'alignment_ratio', 'agreement_stability', 'agreement_momentum',
            'agreement_acceleration', 'agreement_entropy', 'agreement_concentration', 'agreement_dispersion',
            'dte_specific', 'dte_weekly', 'dte_monthly', 'dte_far',
            'dte_transition', 'dte_evolution', 'dte_perf_mean', 'dte_perf_std',
            'dte_reliability', 'dte_consistency', 'dte_adaptation', 'dte_efficiency',
            'stability_score', 'transition_prob', 'integration_quality', 'coherence_index',
            'system_entropy', 'signal_noise_ratio', 'confidence_mean', 'confidence_std',
            'health_score', 'robustness_metric', 'consistency_score', 'reliability_index',
            'synergy_mean', 'synergy_std', 'interaction_strength', 'triad_synergy',
            'complementary_ratio', 'antagonistic_ratio', 'synergy_concentration', 'synergy_dispersion',
            'synergy_momentum', 'synergy_acceleration', 'synergy_stability', 'synergy_efficiency'
        ]
        
        missing_features = set(expected_features) - set(features.keys())
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
        
        return True
    
    def _track_performance(self, processing_time: float, is_valid: bool):
        """
        Track performance metrics
        
        Args:
            processing_time: Processing time in milliseconds
            is_valid: Whether features are valid
        """
        self.performance_history.append({
            'processing_time_ms': processing_time,
            'is_valid': is_valid,
            'within_budget': processing_time <= self.processing_budget_ms
        })
        
        # Keep limited history
        max_history = 1000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
    
    def _get_default_result(self, error_message: str) -> Dict[str, Any]:
        """
        Get default result on error
        
        Args:
            error_message: Error message to include
            
        Returns:
            Default result dictionary
        """
        # Create default features (all 0.5)
        default_features = {
            'agreement_mean': 0.5, 'agreement_std': 0.0, 'consensus_score': 0.5, 'divergence_score': 0.0,
            'weighted_agreement': 0.0, 'alignment_ratio': 0.5, 'agreement_stability': 0.5, 'agreement_momentum': 0.0,
            'agreement_acceleration': 0.0, 'agreement_entropy': 0.0, 'agreement_concentration': 0.5, 'agreement_dispersion': 0.0,
            'dte_specific': 0.5, 'dte_weekly': 0.0, 'dte_monthly': 1.0, 'dte_far': 0.0,
            'dte_transition': 0.5, 'dte_evolution': 0.5, 'dte_perf_mean': 0.5, 'dte_perf_std': 0.0,
            'dte_reliability': 0.5, 'dte_consistency': 0.5, 'dte_adaptation': 0.5, 'dte_efficiency': 1.0,
            'stability_score': 0.5, 'transition_prob': 0.5, 'integration_quality': 0.5, 'coherence_index': 0.5,
            'system_entropy': 0.0, 'signal_noise_ratio': 1.0, 'confidence_mean': 0.5, 'confidence_std': 0.0,
            'health_score': 0.5, 'robustness_metric': 0.5, 'consistency_score': 0.5, 'reliability_index': 0.5,
            'synergy_mean': 0.0, 'synergy_std': 0.0, 'interaction_strength': 0.0, 'triad_synergy': 0.0,
            'complementary_ratio': 0.5, 'antagonistic_ratio': 0.0, 'synergy_concentration': 0.0, 'synergy_dispersion': 0.0,
            'synergy_momentum': 0.0, 'synergy_acceleration': 0.0, 'synergy_stability': 0.5, 'synergy_efficiency': 1.0
        }
        
        return {
            'features': default_features,
            'metadata': {
                'error': error_message,
                'is_default': True,
                'feature_count': len(default_features)
            },
            'performance_metrics': {
                'processing_time_ms': 0,
                'within_budget': True,
                'feature_generation_rate': 0,
                'average_confidence': 0.5
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of performance history
        
        Returns:
            Performance summary statistics
        """
        if not self.performance_history:
            return {
                'total_runs': 0,
                'average_time_ms': 0,
                'success_rate': 0,
                'budget_compliance_rate': 0
            }
        
        times = [p['processing_time_ms'] for p in self.performance_history]
        valid_runs = sum(1 for p in self.performance_history if p['is_valid'])
        within_budget = sum(1 for p in self.performance_history if p['within_budget'])
        
        return {
            'total_runs': len(self.performance_history),
            'average_time_ms': float(np.mean(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'success_rate': float(valid_runs / len(self.performance_history)),
            'budget_compliance_rate': float(within_budget / len(self.performance_history))
        }