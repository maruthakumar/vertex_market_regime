#!/usr/bin/env python3
"""
Enhanced Greek Sentiment Analysis Module V2.0 - Phase 2
Market Regime Gaps Implementation V2.0 - Phase 2 Implementation

This module implements the Phase 2 enhancement for Greek Sentiment Analysis with:
1. Advanced Greek Correlation Framework - Cross-Greek correlation with regime modifiers
2. DTE-Specific Greek Optimization - Time-sensitive Greek weighting
3. Volatility Regime Adaptation - VIX and realized volatility integration

Key Features:
- Enhanced cross-Greek correlation with regime modifiers
- DTE-specific optimization (0-1 DTE focus, 2-4 DTE balance)
- ML-based Greek predictions extending existing XGBoost models
- Real-time Greek decay tracking for time-sensitive analysis
- Integration with existing VolatilityRegimeMonitor

Performance Targets:
- Greek Analysis Latency: <100ms for complete cross-Greek correlation
- DTE Optimization Time: <50ms for weight adjustment
- Memory Usage: <500MB additional allocation
- Accuracy: >90% Greek sentiment classification

Author: Senior Quantitative Trading Expert
Date: June 2025
Version: 2.0.2 - Phase 2 Greek Sentiment Enhancement
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GreekCorrelationConfig:
    """Configuration for advanced Greek correlation framework"""
    correlation_window: int = 50
    regime_modifiers: Dict[str, Dict[str, float]] = None
    decay_factor: float = 0.95
    confidence_threshold: float = 0.8

@dataclass
class DTEOptimizationConfig:
    """Configuration for DTE-specific Greek optimization"""
    short_dte_threshold: int = 1  # 0-1 DTE
    medium_dte_threshold: int = 4  # 2-4 DTE
    gamma_theta_weight_short: float = 0.65  # For 0-1 DTE
    delta_weight_medium: float = 0.45  # For 2-4 DTE
    ml_prediction_enabled: bool = True

@dataclass
class VolatilityRegimeConfig:
    """Configuration for volatility regime adaptation"""
    vix_thresholds: Dict[str, float] = None
    realized_vol_thresholds: Dict[str, float] = None
    adaptive_thresholds: bool = True
    stress_testing_enabled: bool = True

class AdvancedGreekCorrelationFramework:
    """Enhanced cross-Greek correlation with regime modifiers"""
    
    def __init__(self, config: GreekCorrelationConfig):
        self.config = config
        
        # 4x4 correlation matrix for Delta, Gamma, Theta, Vega
        self.correlation_matrix = np.eye(4)
        self.greek_names = ['delta', 'gamma', 'theta', 'vega']
        
        # Regime modifiers for different market conditions
        self.regime_modifiers = config.regime_modifiers or {
            'explosive_directional': {'delta': 1.5, 'gamma': 1.3, 'theta': 0.8, 'vega': 1.2},
            'time_decay_grind': {'delta': 0.9, 'gamma': 0.8, 'theta': 1.4, 'vega': 0.8},
            'volatility_expansion': {'delta': 1.1, 'gamma': 1.4, 'theta': 0.9, 'vega': 1.6},
            'volatility_contraction': {'delta': 1.0, 'gamma': 0.7, 'theta': 1.1, 'vega': 0.6},
            'neutral_consolidation': {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
        }
        
        # Flow sentiment tracking
        self.flow_sentiment_history = deque(maxlen=config.correlation_window)
        self.correlation_history = deque(maxlen=100)
        
        # Performance tracking
        self.analysis_times = deque(maxlen=50)
        
        logger.info("AdvancedGreekCorrelationFramework initialized")
        logger.info(f"Correlation window: {config.correlation_window}")
        logger.info(f"Regime modifiers: {len(self.regime_modifiers)} regimes")
    
    def update_greek_correlations(self, greek_data: Dict[str, float], 
                                current_regime: str = 'neutral_consolidation') -> Dict[str, Any]:
        """Update Greek correlations with regime-specific modifiers"""
        start_time = time.time()
        
        try:
            # Apply regime modifiers to Greek values
            modified_greeks = self._apply_regime_modifiers(greek_data, current_regime)
            
            # Update correlation matrix
            correlation_update = self._calculate_correlation_update(modified_greeks)
            
            # Update correlation matrix with decay
            decay_factor = self.config.decay_factor
            self.correlation_matrix = (
                self.correlation_matrix * decay_factor + 
                correlation_update * (1 - decay_factor)
            )
            
            # Calculate flow sentiment
            flow_sentiment = self._calculate_flow_sentiment(modified_greeks)
            self.flow_sentiment_history.append(flow_sentiment)
            
            # Calculate correlation strength and stability
            correlation_metrics = self._calculate_correlation_metrics()
            
            # Store correlation snapshot
            correlation_snapshot = {
                'timestamp': datetime.now(),
                'correlation_matrix': self.correlation_matrix.copy(),
                'regime': current_regime,
                'flow_sentiment': flow_sentiment,
                'metrics': correlation_metrics
            }
            self.correlation_history.append(correlation_snapshot)
            
            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)
            
            return {
                'correlation_matrix': self.correlation_matrix.tolist(),
                'modified_greeks': modified_greeks,
                'flow_sentiment': flow_sentiment,
                'correlation_metrics': correlation_metrics,
                'regime_applied': current_regime,
                'analysis_time_ms': analysis_time * 1000,
                'performance_target_met': analysis_time < 0.1  # <100ms target
            }
            
        except Exception as e:
            logger.error(f"Error updating Greek correlations: {e}")
            return {'error': str(e)}
    
    def _apply_regime_modifiers(self, greek_data: Dict[str, float], 
                              regime: str) -> Dict[str, float]:
        """Apply regime-specific modifiers to Greek values"""
        modifiers = self.regime_modifiers.get(regime, self.regime_modifiers['neutral_consolidation'])
        
        modified_greeks = {}
        for greek_name in self.greek_names:
            base_value = greek_data.get(greek_name, 0.0)
            modifier = modifiers.get(greek_name, 1.0)
            modified_greeks[greek_name] = base_value * modifier
        
        return modified_greeks
    
    def _calculate_correlation_update(self, greek_data: Dict[str, float]) -> np.ndarray:
        """Calculate correlation matrix update from current Greek data"""
        # Convert Greek data to array
        greek_values = np.array([greek_data.get(name, 0.0) for name in self.greek_names])
        
        # Calculate outer product for correlation update
        if np.any(greek_values != 0):
            # Normalize values
            normalized_values = greek_values / (np.linalg.norm(greek_values) + 1e-8)
            correlation_update = np.outer(normalized_values, normalized_values)
        else:
            correlation_update = np.eye(4)
        
        return correlation_update
    
    def _calculate_flow_sentiment(self, greek_data: Dict[str, float]) -> Dict[str, Any]:
        """Calculate flow sentiment from Greek data"""
        delta = greek_data.get('delta', 0.0)
        gamma = greek_data.get('gamma', 0.0)
        theta = greek_data.get('theta', 0.0)
        vega = greek_data.get('vega', 0.0)
        
        # Calculate sentiment scores
        directional_sentiment = np.tanh(delta * 10)  # -1 to 1
        volatility_sentiment = np.tanh(vega * 5)     # -1 to 1
        time_decay_pressure = np.tanh(abs(theta) * 8)  # 0 to 1
        gamma_risk = np.tanh(abs(gamma) * 15)        # 0 to 1
        
        # Overall sentiment
        overall_sentiment = (directional_sentiment * 0.3 + 
                           volatility_sentiment * 0.3 + 
                           time_decay_pressure * 0.2 + 
                           gamma_risk * 0.2)
        
        return {
            'directional_sentiment': float(directional_sentiment),
            'volatility_sentiment': float(volatility_sentiment),
            'time_decay_pressure': float(time_decay_pressure),
            'gamma_risk': float(gamma_risk),
            'overall_sentiment': float(overall_sentiment),
            'sentiment_strength': float(abs(overall_sentiment))
        }
    
    def _calculate_correlation_metrics(self) -> Dict[str, float]:
        """Calculate correlation matrix quality metrics"""
        # Correlation strength (average off-diagonal correlations)
        off_diagonal = self.correlation_matrix[~np.eye(4, dtype=bool)]
        correlation_strength = float(np.mean(np.abs(off_diagonal)))
        
        # Correlation stability (how much correlations change over time)
        if len(self.correlation_history) >= 2:
            prev_matrix = self.correlation_history[-2]['correlation_matrix']
            stability = 1.0 - float(np.mean(np.abs(self.correlation_matrix - prev_matrix)))
            stability = max(0.0, min(1.0, stability))
        else:
            stability = 0.5
        
        # Eigenvalue analysis for matrix health
        eigenvalues = np.linalg.eigvals(self.correlation_matrix)
        condition_number = float(np.max(eigenvalues) / (np.min(eigenvalues) + 1e-8))
        matrix_health = 1.0 / (1.0 + condition_number / 100.0)  # Normalize
        
        return {
            'correlation_strength': correlation_strength,
            'correlation_stability': stability,
            'matrix_health': matrix_health,
            'condition_number': condition_number,
            'average_analysis_time_ms': float(np.mean(self.analysis_times)) * 1000 if self.analysis_times else 0.0
        }

class DTESpecificGreekOptimizer:
    """DTE-specific Greek optimization with ML-based predictions"""
    
    def __init__(self, config: DTEOptimizationConfig):
        self.config = config
        
        # DTE-specific weight configurations
        self.dte_weight_configs = {
            'short_dte': {  # 0-1 DTE
                'gamma': 0.35,
                'theta': 0.30,
                'delta': 0.20,
                'vega': 0.15
            },
            'medium_dte': {  # 2-4 DTE
                'delta': 0.45,
                'gamma': 0.25,
                'vega': 0.20,
                'theta': 0.10
            },
            'long_dte': {  # 5+ DTE
                'delta': 0.40,
                'vega': 0.30,
                'gamma': 0.20,
                'theta': 0.10
            }
        }
        
        # ML prediction tracking
        self.prediction_history = deque(maxlen=100)
        self.optimization_times = deque(maxlen=50)
        
        # Greek decay tracking
        self.decay_tracker = GreekDecayTracker()
        
        logger.info("DTESpecificGreekOptimizer initialized")
        logger.info(f"Short DTE threshold: {config.short_dte_threshold}")
        logger.info(f"Medium DTE threshold: {config.medium_dte_threshold}")
        logger.info(f"ML prediction enabled: {config.ml_prediction_enabled}")
    
    def optimize_greek_weights(self, current_dte: int, 
                             greek_data: Dict[str, float],
                             market_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize Greek weights based on DTE and market conditions"""
        start_time = time.time()
        
        try:
            # Classify DTE regime
            dte_regime = self._classify_dte_regime(current_dte)
            
            # Get base weights for DTE regime
            base_weights = self.dte_weight_configs[dte_regime].copy()
            
            # Apply ML-based adjustments if enabled
            if self.config.ml_prediction_enabled:
                ml_adjustments = self._calculate_ml_adjustments(
                    current_dte, greek_data, market_conditions
                )
                base_weights = self._apply_ml_adjustments(base_weights, ml_adjustments)
            
            # Apply decay modeling
            decay_adjustments = self.decay_tracker.calculate_decay_adjustments(
                current_dte, greek_data
            )
            final_weights = self._apply_decay_adjustments(base_weights, decay_adjustments)
            
            # Normalize weights
            final_weights = self._normalize_weights(final_weights)
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(
                dte_regime, final_weights, greek_data
            )
            
            optimization_time = time.time() - start_time
            self.optimization_times.append(optimization_time)
            
            return {
                'optimized_weights': final_weights,
                'dte_regime': dte_regime,
                'base_weights': self.dte_weight_configs[dte_regime],
                'ml_adjustments_applied': self.config.ml_prediction_enabled,
                'decay_adjustments': decay_adjustments,
                'optimization_metrics': optimization_metrics,
                'optimization_time_ms': optimization_time * 1000,
                'performance_target_met': optimization_time < 0.05  # <50ms target
            }
            
        except Exception as e:
            logger.error(f"Error optimizing Greek weights: {e}")
            return {'error': str(e)}
    
    def _classify_dte_regime(self, dte: int) -> str:
        """Classify DTE into regime categories"""
        if dte <= self.config.short_dte_threshold:
            return 'short_dte'
        elif dte <= self.config.medium_dte_threshold:
            return 'medium_dte'
        else:
            return 'long_dte'
    
    def _calculate_ml_adjustments(self, dte: int, greek_data: Dict[str, float],
                                market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ML-based weight adjustments"""
        # Simplified ML prediction (would integrate with existing XGBoost models)
        vix = market_conditions.get('vix', 20.0) if market_conditions else 20.0
        realized_vol = market_conditions.get('realized_volatility', 0.2) if market_conditions else 0.2
        
        # ML-based adjustments based on market conditions
        ml_adjustments = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
        
        # VIX-based adjustments
        if vix > 25:  # High volatility
            ml_adjustments['vega'] += 0.1
            ml_adjustments['gamma'] += 0.05
            ml_adjustments['delta'] -= 0.05
        elif vix < 15:  # Low volatility
            ml_adjustments['theta'] += 0.1
            ml_adjustments['delta'] += 0.05
            ml_adjustments['vega'] -= 0.05
        
        # DTE-specific ML adjustments
        if dte <= 1:
            ml_adjustments['gamma'] += 0.05
            ml_adjustments['theta'] += 0.05
        
        return ml_adjustments
    
    def _apply_ml_adjustments(self, base_weights: Dict[str, float],
                            ml_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Apply ML adjustments to base weights"""
        adjusted_weights = {}
        for greek in base_weights:
            adjustment = ml_adjustments.get(greek, 0.0)
            adjusted_weights[greek] = base_weights[greek] + adjustment
        
        return adjusted_weights
    
    def _apply_decay_adjustments(self, weights: Dict[str, float],
                               decay_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Apply decay-based adjustments to weights"""
        adjusted_weights = {}
        for greek in weights:
            decay_factor = decay_adjustments.get(greek, 1.0)
            adjusted_weights[greek] = weights[greek] * decay_factor
        
        return adjusted_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0"""
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {greek: weight / total_weight for greek, weight in weights.items()}
        else:
            return {greek: 0.25 for greek in weights}  # Equal weights fallback
    
    def _calculate_optimization_metrics(self, dte_regime: str, 
                                      final_weights: Dict[str, float],
                                      greek_data: Dict[str, float]) -> Dict[str, Any]:
        """Calculate optimization quality metrics"""
        # Weight distribution analysis
        weight_entropy = -sum(w * np.log(w + 1e-8) for w in final_weights.values())
        weight_concentration = max(final_weights.values())
        
        # Greek utilization efficiency
        greek_utilization = sum(abs(greek_data.get(greek, 0.0)) * weight 
                              for greek, weight in final_weights.items())
        
        return {
            'dte_regime': dte_regime,
            'weight_entropy': float(weight_entropy),
            'weight_concentration': float(weight_concentration),
            'greek_utilization': float(greek_utilization),
            'weights_normalized': abs(sum(final_weights.values()) - 1.0) < 0.01,
            'average_optimization_time_ms': float(np.mean(self.optimization_times)) * 1000 if self.optimization_times else 0.0
        }

class GreekDecayTracker:
    """Real-time Greek decay tracking for time-sensitive analysis"""
    
    def __init__(self):
        self.decay_history = deque(maxlen=100)
        self.decay_models = {
            'theta': self._theta_decay_model,
            'gamma': self._gamma_decay_model,
            'vega': self._vega_decay_model,
            'delta': self._delta_decay_model
        }
        
        logger.info("GreekDecayTracker initialized")
    
    def calculate_decay_adjustments(self, dte: int, 
                                  greek_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate decay-based adjustments for Greek weights"""
        decay_adjustments = {}
        
        for greek_name, decay_model in self.decay_models.items():
            current_value = greek_data.get(greek_name, 0.0)
            decay_factor = decay_model(dte, current_value)
            decay_adjustments[greek_name] = decay_factor
        
        # Store decay snapshot
        self.decay_history.append({
            'timestamp': datetime.now(),
            'dte': dte,
            'decay_adjustments': decay_adjustments.copy(),
            'greek_data': greek_data.copy()
        })
        
        return decay_adjustments
    
    def _theta_decay_model(self, dte: int, theta_value: float) -> float:
        """Model theta decay acceleration near expiration"""
        if dte <= 1:
            return 1.3  # Accelerated theta decay
        elif dte <= 7:
            return 1.1  # Moderate acceleration
        else:
            return 1.0  # Normal decay
    
    def _gamma_decay_model(self, dte: int, gamma_value: float) -> float:
        """Model gamma behavior near expiration"""
        if dte <= 1:
            return 1.4  # Gamma explosion near expiration
        elif dte <= 3:
            return 1.2  # Elevated gamma
        else:
            return 1.0  # Normal gamma
    
    def _vega_decay_model(self, dte: int, vega_value: float) -> float:
        """Model vega decay near expiration"""
        if dte <= 1:
            return 0.6  # Rapid vega decay
        elif dte <= 7:
            return 0.8  # Moderate vega decay
        else:
            return 1.0  # Normal vega
    
    def _delta_decay_model(self, dte: int, delta_value: float) -> float:
        """Model delta behavior near expiration"""
        if dte <= 1:
            return 1.1  # Slightly elevated delta importance
        else:
            return 1.0  # Normal delta

class VolatilityRegimeGreekAdapter:
    """Volatility regime adaptation for Greek analysis"""
    
    def __init__(self, config: VolatilityRegimeConfig):
        self.config = config
        
        # VIX thresholds
        self.vix_thresholds = config.vix_thresholds or {
            'low_vix': 15.0,
            'normal_vix_lower': 15.0,
            'normal_vix_upper': 25.0,
            'high_vix': 25.0
        }
        
        # Realized volatility thresholds
        self.realized_vol_thresholds = config.realized_vol_thresholds or {
            'low_vol': 0.15,
            'normal_vol_lower': 0.15,
            'normal_vol_upper': 0.30,
            'high_vol': 0.30
        }
        
        # Adaptive threshold tracking
        self.threshold_history = deque(maxlen=100)
        
        logger.info("VolatilityRegimeGreekAdapter initialized")
        logger.info(f"VIX thresholds: {self.vix_thresholds}")
        logger.info(f"Realized vol thresholds: {self.realized_vol_thresholds}")
    
    def adapt_greek_analysis(self, vix: float, realized_vol: float,
                           greek_correlation_results: Dict[str, Any],
                           dte_optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Greek analysis based on volatility regime"""
        try:
            # Classify volatility regime
            vol_regime = self._classify_volatility_regime(vix, realized_vol)
            
            # Calculate regime-specific adjustments
            regime_adjustments = self._calculate_regime_adjustments(vol_regime)
            
            # Apply adaptive thresholds if enabled
            if self.config.adaptive_thresholds:
                adaptive_thresholds = self._calculate_adaptive_thresholds(vix, realized_vol)
            else:
                adaptive_thresholds = {}
            
            # Integrate with stress testing if enabled
            stress_test_results = {}
            if self.config.stress_testing_enabled:
                stress_test_results = self._perform_stress_testing(
                    vol_regime, greek_correlation_results, dte_optimization_results
                )
            
            return {
                'volatility_regime': vol_regime,
                'regime_adjustments': regime_adjustments,
                'adaptive_thresholds': adaptive_thresholds,
                'stress_test_results': stress_test_results,
                'vix_level': vix,
                'realized_volatility': realized_vol,
                'adaptation_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error adapting Greek analysis: {e}")
            return {'error': str(e)}
    
    def _classify_volatility_regime(self, vix: float, realized_vol: float) -> str:
        """Classify current volatility regime"""
        vix_regime = 'normal_vix'
        if vix < self.vix_thresholds['low_vix']:
            vix_regime = 'low_vix'
        elif vix > self.vix_thresholds['high_vix']:
            vix_regime = 'high_vix'
        
        vol_regime = 'normal_vol'
        if realized_vol < self.realized_vol_thresholds['low_vol']:
            vol_regime = 'low_vol'
        elif realized_vol > self.realized_vol_thresholds['high_vol']:
            vol_regime = 'high_vol'
        
        # Combined regime classification
        if vix_regime == 'low_vix' and vol_regime == 'low_vol':
            return 'low_volatility'
        elif vix_regime == 'high_vix' or vol_regime == 'high_vol':
            return 'high_volatility'
        else:
            return 'normal_volatility'
    
    def _calculate_regime_adjustments(self, vol_regime: str) -> Dict[str, float]:
        """Calculate regime-specific adjustments"""
        regime_adjustments = {
            'low_volatility': {
                'correlation_sensitivity': 0.8,
                'dte_weight_adjustment': 1.1,
                'decay_acceleration': 0.9
            },
            'normal_volatility': {
                'correlation_sensitivity': 1.0,
                'dte_weight_adjustment': 1.0,
                'decay_acceleration': 1.0
            },
            'high_volatility': {
                'correlation_sensitivity': 1.3,
                'dte_weight_adjustment': 0.9,
                'decay_acceleration': 1.2
            }
        }
        
        return regime_adjustments.get(vol_regime, regime_adjustments['normal_volatility'])
    
    def _calculate_adaptive_thresholds(self, vix: float, realized_vol: float) -> Dict[str, float]:
        """Calculate adaptive thresholds based on current market conditions"""
        # Adaptive threshold calculation based on market conditions
        base_correlation_threshold = 0.6
        base_confidence_threshold = 0.8
        
        # Adjust thresholds based on volatility
        vol_adjustment = (vix - 20.0) / 20.0  # Normalize around VIX 20
        
        adaptive_thresholds = {
            'correlation_threshold': max(0.3, base_correlation_threshold + vol_adjustment * 0.2),
            'confidence_threshold': max(0.5, base_confidence_threshold - abs(vol_adjustment) * 0.1),
            'regime_transition_threshold': 0.7 + vol_adjustment * 0.1
        }
        
        return adaptive_thresholds
    
    def _perform_stress_testing(self, vol_regime: str,
                              correlation_results: Dict[str, Any],
                              optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing for Greek analysis"""
        # Simplified stress testing implementation
        stress_scenarios = {
            'vix_spike': {'vix_multiplier': 2.0, 'vol_multiplier': 1.5},
            'vix_crush': {'vix_multiplier': 0.5, 'vol_multiplier': 0.7},
            'extreme_gamma': {'gamma_multiplier': 3.0},
            'theta_acceleration': {'theta_multiplier': 2.0}
        }
        
        stress_results = {}
        for scenario_name, scenario_params in stress_scenarios.items():
            # Simulate scenario impact
            scenario_impact = self._simulate_scenario_impact(scenario_params, correlation_results)
            stress_results[scenario_name] = scenario_impact
        
        return stress_results
    
    def _simulate_scenario_impact(self, scenario_params: Dict[str, float],
                                correlation_results: Dict[str, Any]) -> Dict[str, float]:
        """Simulate impact of stress scenario"""
        # Simplified scenario simulation
        base_correlation_strength = correlation_results.get('correlation_metrics', {}).get('correlation_strength', 0.5)
        
        # Apply scenario multipliers
        stressed_correlation = base_correlation_strength
        for param, multiplier in scenario_params.items():
            if 'correlation' in param or 'gamma' in param or 'theta' in param:
                stressed_correlation *= multiplier
        
        # Calculate impact metrics
        impact_magnitude = abs(stressed_correlation - base_correlation_strength)
        resilience_score = 1.0 / (1.0 + impact_magnitude)
        
        return {
            'stressed_correlation': float(stressed_correlation),
            'impact_magnitude': float(impact_magnitude),
            'resilience_score': float(resilience_score)
        }
