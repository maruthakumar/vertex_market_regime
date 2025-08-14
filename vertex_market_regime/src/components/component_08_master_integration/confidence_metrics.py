"""
Integration Confidence Metrics

Calculates confidence and quality metrics for the integrated system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ConfidenceMetrics:
    """
    Calculates integration confidence and quality metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence metrics calculator
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Confidence thresholds
        self.confidence_thresholds = self.config.get('confidence_thresholds', {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        })
        
        # Health assessment parameters
        self.health_params = self.config.get('health_params', {
            'min_component_health': 0.3,
            'optimal_component_health': 0.7,
            'signal_quality_weight': 0.4,
            'consistency_weight': 0.3,
            'reliability_weight': 0.3
        })
        
        # Transition tracking
        self.signal_history = []
        self.confidence_history = []
        
        logger.info("Initialized ConfidenceMetrics")
    
    def calculate_confidence_metrics(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        correlation_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence metrics
        
        Args:
            aggregated_components: Aggregated component data
            correlation_analysis: Optional correlation analysis results
            
        Returns:
            Dictionary of confidence metrics
        """
        # Component health assessment
        health_metrics = self._assess_component_health(aggregated_components)
        
        # Integration quality metrics
        quality_metrics = self._calculate_integration_quality(
            aggregated_components,
            correlation_analysis
        )
        
        # Transition probability features
        transition_features = self._calculate_transition_probabilities(
            aggregated_components
        )
        
        # System stability indicators
        stability_indicators = self._calculate_stability_indicators(
            aggregated_components,
            health_metrics
        )
        
        # Overall confidence score
        overall_confidence = self._calculate_overall_confidence(
            health_metrics,
            quality_metrics,
            stability_indicators
        )
        
        return {
            'health_metrics': health_metrics,
            'quality_metrics': quality_metrics,
            'transition_features': transition_features,
            'stability_indicators': stability_indicators,
            'overall_confidence': overall_confidence
        }
    
    def _assess_component_health(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Assess health of individual components
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary of health metrics
        """
        health_scores = []
        component_statuses = {}
        
        for comp_name, comp_data in aggregated_components.items():
            health_score = comp_data.get('health_score', 0.5)
            health_scores.append(health_score)
            
            # Classify component health status
            if health_score >= self.health_params['optimal_component_health']:
                status = 'healthy'
            elif health_score >= self.health_params['min_component_health']:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            component_statuses[comp_name] = {
                'health': health_score,
                'status': status
            }
        
        # Calculate aggregate health metrics
        if health_scores:
            mean_health = np.mean(health_scores)
            min_health = np.min(health_scores)
            max_health = np.max(health_scores)
            health_std = np.std(health_scores)
            
            # Count healthy components
            healthy_count = sum(1 for score in health_scores 
                              if score >= self.health_params['optimal_component_health'])
            health_ratio = healthy_count / len(health_scores)
        else:
            mean_health = 0.5
            min_health = 0.0
            max_health = 1.0
            health_std = 0.0
            health_ratio = 0.0
        
        return {
            'mean_health': float(mean_health),
            'min_health': float(min_health),
            'max_health': float(max_health),
            'health_std': float(health_std),
            'health_ratio': float(health_ratio),
            'component_statuses': component_statuses
        }
    
    def _calculate_integration_quality(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        correlation_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate integration quality metrics
        
        Args:
            aggregated_components: Aggregated component data
            correlation_analysis: Optional correlation analysis results
            
        Returns:
            Dictionary of quality metrics
        """
        # Signal quality assessment
        signal_qualities = []
        
        for comp_data in aggregated_components.values():
            metadata = comp_data.get('metadata', {})
            data_quality = metadata.get('data_quality', 0.5)
            confidence = metadata.get('confidence', 0.5)
            
            # Combined signal quality
            signal_quality = data_quality * confidence
            signal_qualities.append(signal_quality)
        
        # Calculate quality metrics
        if signal_qualities:
            mean_quality = np.mean(signal_qualities)
            quality_consistency = 1.0 - np.std(signal_qualities)
        else:
            mean_quality = 0.5
            quality_consistency = 0.5
        
        # Integration coherence from correlation analysis
        if correlation_analysis:
            coherence_metrics = correlation_analysis.get('coherence_metrics', {})
            integration_score = coherence_metrics.get('integration_score', 0.5)
            coherence_index = coherence_metrics.get('coherence_index', 0.5)
        else:
            integration_score = 0.5
            coherence_index = 0.5
        
        # Overall integration quality
        quality_score = (
            self.health_params['signal_quality_weight'] * mean_quality +
            self.health_params['consistency_weight'] * quality_consistency +
            self.health_params['reliability_weight'] * coherence_index
        )
        
        return {
            'signal_quality_mean': float(mean_quality),
            'quality_consistency': float(quality_consistency),
            'integration_score': float(integration_score),
            'coherence_index': float(coherence_index),
            'overall_quality': float(quality_score)
        }
    
    def _calculate_transition_probabilities(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate transition probability features
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary of transition features
        """
        # Extract current signals
        current_signals = []
        
        for comp_data in aggregated_components.values():
            signals = comp_data.get('signals', {})
            if signals:
                # Get primary signal
                primary_signal = list(signals.values())[0] if signals else 0.0
                current_signals.append(primary_signal)
        
        # Update signal history
        self.signal_history.append(current_signals)
        
        # Keep limited history
        max_history = 50
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
        
        # Calculate transition features
        if len(self.signal_history) > 1:
            # Signal changes
            signal_changes = []
            for i in range(1, len(self.signal_history)):
                prev_signals = self.signal_history[i-1]
                curr_signals = self.signal_history[i]
                
                if len(prev_signals) == len(curr_signals):
                    changes = [abs(c - p) for c, p in zip(curr_signals, prev_signals)]
                    signal_changes.append(np.mean(changes))
            
            if signal_changes:
                transition_rate = np.mean(signal_changes)
                transition_volatility = np.std(signal_changes)
                transition_trend = np.polyfit(range(len(signal_changes)), 
                                            signal_changes, 1)[0] if len(signal_changes) > 1 else 0.0
            else:
                transition_rate = 0.0
                transition_volatility = 0.0
                transition_trend = 0.0
        else:
            transition_rate = 0.0
            transition_volatility = 0.0
            transition_trend = 0.0
        
        # Calculate transition probability based on recent changes
        if transition_volatility > 0:
            transition_probability = min(1.0, transition_rate / transition_volatility)
        else:
            transition_probability = transition_rate
        
        # Regime stability (inverse of transition probability)
        regime_stability = 1.0 - transition_probability
        
        return {
            'transition_rate': float(transition_rate),
            'transition_volatility': float(transition_volatility),
            'transition_trend': float(transition_trend),
            'transition_probability': float(transition_probability),
            'regime_stability': float(regime_stability)
        }
    
    def _calculate_stability_indicators(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        health_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate system stability indicators
        
        Args:
            aggregated_components: Aggregated component data
            health_metrics: Component health metrics
            
        Returns:
            Dictionary of stability indicators
        """
        # Component stability based on health variance
        health_stability = 1.0 / (1.0 + health_metrics['health_std'])
        
        # Signal consistency across components
        all_signals = []
        for comp_data in aggregated_components.values():
            signals = comp_data.get('signals', {})
            all_signals.extend(list(signals.values()))
        
        if all_signals:
            signal_consistency = 1.0 / (1.0 + np.std(all_signals))
            signal_range = np.ptp(all_signals)  # Peak-to-peak range
        else:
            signal_consistency = 0.5
            signal_range = 0.0
        
        # Processing time stability
        processing_times = []
        for comp_data in aggregated_components.values():
            proc_time = comp_data.get('processing_time', 0)
            if proc_time > 0:
                processing_times.append(proc_time)
        
        if processing_times:
            time_consistency = 1.0 / (1.0 + np.std(processing_times) / np.mean(processing_times))
        else:
            time_consistency = 0.5
        
        # Overall system stability
        system_stability = np.mean([
            health_stability,
            signal_consistency,
            time_consistency
        ])
        
        # Robustness metric (ability to handle component failures)
        min_health = health_metrics['min_health']
        robustness = min_health * health_metrics['health_ratio']
        
        return {
            'health_stability': float(health_stability),
            'signal_consistency': float(signal_consistency),
            'signal_range': float(signal_range),
            'time_consistency': float(time_consistency),
            'system_stability': float(system_stability),
            'robustness_metric': float(robustness)
        }
    
    def _calculate_overall_confidence(
        self,
        health_metrics: Dict[str, float],
        quality_metrics: Dict[str, float],
        stability_indicators: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate overall confidence score
        
        Args:
            health_metrics: Component health metrics
            quality_metrics: Integration quality metrics
            stability_indicators: System stability indicators
            
        Returns:
            Dictionary with overall confidence metrics
        """
        # Component confidence factors
        health_confidence = health_metrics['mean_health']
        quality_confidence = quality_metrics['overall_quality']
        stability_confidence = stability_indicators['system_stability']
        
        # Weighted overall confidence
        overall_confidence = np.mean([
            health_confidence,
            quality_confidence,
            stability_confidence
        ])
        
        # Update confidence history
        self.confidence_history.append(overall_confidence)
        
        # Keep limited history
        max_history = 100
        if len(self.confidence_history) > max_history:
            self.confidence_history = self.confidence_history[-max_history:]
        
        # Calculate confidence statistics
        if self.confidence_history:
            confidence_mean = np.mean(self.confidence_history)
            confidence_std = np.std(self.confidence_history)
            confidence_trend = np.polyfit(range(len(self.confidence_history)), 
                                         self.confidence_history, 1)[0] if len(self.confidence_history) > 1 else 0.0
        else:
            confidence_mean = overall_confidence
            confidence_std = 0.0
            confidence_trend = 0.0
        
        # Confidence level classification
        if overall_confidence >= self.confidence_thresholds['high']:
            confidence_level = 'high'
        elif overall_confidence >= self.confidence_thresholds['medium']:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        return {
            'overall_confidence': float(overall_confidence),
            'confidence_mean': float(confidence_mean),
            'confidence_std': float(confidence_std),
            'confidence_trend': float(confidence_trend),
            'confidence_level': confidence_level,
            'health_component': float(health_confidence),
            'quality_component': float(quality_confidence),
            'stability_component': float(stability_confidence)
        }
    
    def get_confidence_features(
        self,
        confidence_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract features from confidence metrics for ML consumption
        
        Args:
            confidence_metrics: Full confidence metrics results
            
        Returns:
            Dictionary of confidence features
        """
        features = {}
        
        # Health features
        health = confidence_metrics['health_metrics']
        features['health_mean'] = health['mean_health']
        features['health_min'] = health['min_health']
        features['health_ratio'] = health['health_ratio']
        
        # Quality features
        quality = confidence_metrics['quality_metrics']
        features['signal_quality'] = quality['signal_quality_mean']
        features['quality_consistency'] = quality['quality_consistency']
        features['integration_quality'] = quality['overall_quality']
        
        # Transition features
        transition = confidence_metrics['transition_features']
        features['transition_probability'] = transition['transition_probability']
        features['regime_stability'] = transition['regime_stability']
        
        # Stability features
        stability = confidence_metrics['stability_indicators']
        features['system_stability'] = stability['system_stability']
        features['robustness'] = stability['robustness_metric']
        
        # Overall confidence
        overall = confidence_metrics['overall_confidence']
        features['confidence_score'] = overall['overall_confidence']
        features['confidence_trend'] = overall['confidence_trend']
        
        return features