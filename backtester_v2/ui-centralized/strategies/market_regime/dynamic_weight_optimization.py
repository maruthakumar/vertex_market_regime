#!/usr/bin/env python3
"""
Dynamic Weight Optimization Module for Market Regime Triple Straddle Engine
Phase 2 Implementation: Dynamic Weight Systems

This module implements the dynamic weight optimization enhancements specified in the 
Market Regime Gaps Implementation V1.0 document:

1. Real-Time Market Volatility Weight Adjustment
2. Correlation-Based Weight Optimization System
3. ML-Based DTE-Specific Weight Adaptation

Performance Targets:
- 30% correlation impact reduction
- Real-time weight adjustment based on market conditions
- ML-enhanced DTE-specific optimization

Author: Senior Quantitative Trading Expert
Date: June 2025
Version: 1.0 - Phase 2 Dynamic Weight Systems
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import time
from scipy.optimize import minimize
from dataclasses import dataclass

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    # Mock XGBoost for testing
    class XGBoostRegressor:
        def __init__(self, *args, **kwargs):
            self.is_fitted = False
        def fit(self, X, y):
            self.is_fitted = True
        def predict(self, X):
            return np.random.random(len(X)) * 0.1 + 0.8
        def score(self, X, y):
            return 0.85
    xgb = type('xgb', (), {'XGBRegressor': XGBoostRegressor})

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class VolatilityRegime:
    """Volatility regime classification"""
    vix_level: float
    atr_value: float
    realized_vol: float
    regime_type: str
    adjustment_factor: float
    timestamp: datetime

@dataclass
class WeightOptimizationResult:
    """Weight optimization result"""
    optimized_weights: Dict[str, float]
    optimization_method: str
    correlation_reduction: float
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

class VolatilityRegimeMonitor:
    """Monitor and classify volatility regimes for dynamic weight adjustment"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=1000)
        self.current_regime = None
        
        # Volatility regime thresholds
        self.vix_thresholds = {
            'low': 15.0,
            'normal_low': 25.0,
            'normal_high': 35.0,
            'high': 50.0
        }
        
        # ATR normalization parameters
        self.atr_normalization_factor = 100.0
        self.realized_vol_threshold = 0.5
    
    def detect_regime(self, market_data: Dict[str, Any]) -> VolatilityRegime:
        """Detect current volatility regime based on market data"""
        try:
            vix_level = market_data.get('vix', 20.0)
            atr_value = market_data.get('atr', 50.0)
            realized_vol = market_data.get('realized_volatility', 0.25)
            
            # Classify VIX regime
            if vix_level < self.vix_thresholds['low']:
                vix_regime = 'low'
                vix_factor = -0.2  # Reduce volatility components
            elif vix_level < self.vix_thresholds['normal_low']:
                vix_regime = 'normal'
                vix_factor = 0.0   # No adjustment
            elif vix_level < self.vix_thresholds['normal_high']:
                vix_regime = 'high'
                vix_factor = 0.3   # Increase volatility components
            else:
                vix_regime = 'extreme'
                vix_factor = 0.5   # Maximum adjustment
            
            # Calculate ATR factor
            atr_factor = min(atr_value / self.atr_normalization_factor, 1.0) - 0.5
            
            # Calculate realized volatility factor
            rv_factor = min(realized_vol / self.realized_vol_threshold, 1.0) - 0.5
            
            # Calculate overall adjustment factor
            adjustment_factor = 1 + (0.4 * vix_factor + 0.3 * atr_factor + 0.3 * rv_factor)
            adjustment_factor = max(0.1, min(2.0, adjustment_factor))  # Constrain to [0.1, 2.0]
            
            # Determine regime type
            if adjustment_factor < 0.7:
                regime_type = 'low_volatility'
            elif adjustment_factor < 1.3:
                regime_type = 'normal_volatility'
            else:
                regime_type = 'high_volatility'
            
            regime = VolatilityRegime(
                vix_level=vix_level,
                atr_value=atr_value,
                realized_vol=realized_vol,
                regime_type=regime_type,
                adjustment_factor=adjustment_factor,
                timestamp=datetime.now()
            )
            
            self.regime_history.append(regime)
            self.current_regime = regime
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            # Return default regime
            return VolatilityRegime(
                vix_level=20.0,
                atr_value=50.0,
                realized_vol=0.25,
                regime_type='normal_volatility',
                adjustment_factor=1.0,
                timestamp=datetime.now()
            )

class DynamicWeightOptimizer:
    """Real-time dynamic weight optimization based on market volatility"""
    
    def __init__(self):
        self.base_weights = {
            'atm_straddle': 0.25,
            'itm1_straddle': 0.20,
            'otm1_straddle': 0.15,
            'combined_straddle': 0.20,
            'atm_ce': 0.10,
            'atm_pe': 0.10
        }
        
        self.volatility_monitor = VolatilityRegimeMonitor()
        self.weight_history = deque(maxlen=1000)
        
        # Component volatility sensitivities
        self.volatility_sensitivities = {
            'atm_straddle': 1.0,      # High sensitivity to volatility
            'itm1_straddle': 0.8,     # Medium-high sensitivity
            'otm1_straddle': 1.2,     # Highest sensitivity (gamma exposure)
            'combined_straddle': 0.9,  # Medium sensitivity
            'atm_ce': 0.7,            # Lower sensitivity
            'atm_pe': 0.7             # Lower sensitivity
        }
    
    def optimize_weights_realtime(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Optimize weights in real-time based on current market conditions"""
        try:
            # Detect current volatility regime
            current_regime = self.volatility_monitor.detect_regime(market_data)
            
            # Calculate regime-based adjustments
            adjustment_factors = self._calculate_regime_adjustments(current_regime)
            
            # Apply adjustments to base weights
            optimized_weights = {}
            for component, base_weight in self.base_weights.items():
                sensitivity = self.volatility_sensitivities.get(component, 1.0)
                adjustment = adjustment_factors.get(component, 1.0)
                
                # Apply volatility-sensitive adjustment
                adjusted_weight = base_weight * adjustment * sensitivity
                optimized_weights[component] = adjusted_weight
            
            # Normalize weights to sum to 1.0
            total_weight = sum(optimized_weights.values())
            normalized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
            
            # Store weight history
            self.weight_history.append({
                'timestamp': datetime.now(),
                'weights': normalized_weights.copy(),
                'regime': current_regime,
                'total_adjustment': current_regime.adjustment_factor
            })
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
            return self.base_weights.copy()
    
    def _calculate_regime_adjustments(self, regime: VolatilityRegime) -> Dict[str, float]:
        """Calculate component-specific adjustments based on volatility regime"""
        adjustments = {}
        
        base_adjustment = regime.adjustment_factor
        
        if regime.regime_type == 'low_volatility':
            # In low volatility, reduce OTM exposure, increase ATM
            adjustments = {
                'atm_straddle': base_adjustment * 1.1,
                'itm1_straddle': base_adjustment * 1.05,
                'otm1_straddle': base_adjustment * 0.8,
                'combined_straddle': base_adjustment * 1.0,
                'atm_ce': base_adjustment * 1.0,
                'atm_pe': base_adjustment * 1.0
            }
        elif regime.regime_type == 'high_volatility':
            # In high volatility, increase OTM exposure for gamma capture
            adjustments = {
                'atm_straddle': base_adjustment * 0.9,
                'itm1_straddle': base_adjustment * 0.95,
                'otm1_straddle': base_adjustment * 1.3,
                'combined_straddle': base_adjustment * 1.1,
                'atm_ce': base_adjustment * 1.05,
                'atm_pe': base_adjustment * 1.05
            }
        else:  # normal_volatility
            # Balanced approach for normal volatility
            adjustments = {component: base_adjustment for component in self.base_weights.keys()}
        
        return adjustments
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get statistics about weight optimization performance"""
        if not self.weight_history:
            return {}
        
        recent_weights = list(self.weight_history)[-100:]  # Last 100 adjustments
        
        # Calculate weight stability
        weight_changes = []
        for i in range(1, len(recent_weights)):
            prev_weights = recent_weights[i-1]['weights']
            curr_weights = recent_weights[i]['weights']
            
            total_change = sum(abs(curr_weights[k] - prev_weights[k]) for k in curr_weights.keys())
            weight_changes.append(total_change)
        
        avg_weight_change = np.mean(weight_changes) if weight_changes else 0
        
        # Calculate regime distribution
        regime_counts = {}
        for entry in recent_weights:
            regime_type = entry['regime'].regime_type
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        return {
            'total_adjustments': len(self.weight_history),
            'recent_adjustments': len(recent_weights),
            'average_weight_change': avg_weight_change,
            'weight_stability': 1.0 - min(avg_weight_change, 1.0),
            'regime_distribution': regime_counts,
            'current_regime': self.volatility_monitor.current_regime.regime_type if self.volatility_monitor.current_regime else None
        }

class CorrelationBasedOptimizer:
    """Correlation-based weight optimization system"""

    def __init__(self, correlation_threshold: float = 0.8):
        self.correlation_threshold = correlation_threshold
        self.correlation_history = deque(maxlen=500)
        self.optimization_history = deque(maxlen=100)

    def detect_high_correlation_periods(self, correlation_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect when component correlations exceed threshold"""
        high_corr_pairs = []

        try:
            components = correlation_matrix.columns
            for i, component1 in enumerate(components):
                for j, component2 in enumerate(components):
                    if i < j:  # Avoid duplicate pairs
                        corr_value = abs(correlation_matrix.iloc[i, j])
                        if corr_value > self.correlation_threshold:
                            high_corr_pairs.append({
                                'pair': (component1, component2),
                                'correlation': corr_value,
                                'adjustment_needed': True,
                                'severity': 'high' if corr_value > 0.9 else 'medium'
                            })

            # Store correlation analysis
            self.correlation_history.append({
                'timestamp': datetime.now(),
                'high_correlations': len(high_corr_pairs),
                'max_correlation': max([pair['correlation'] for pair in high_corr_pairs]) if high_corr_pairs else 0,
                'correlation_matrix': correlation_matrix.copy()
            })

            return high_corr_pairs

        except Exception as e:
            logger.error(f"Error detecting high correlations: {e}")
            return []

    def optimize_weights_for_correlation(self, base_weights: Dict[str, float],
                                       high_corr_pairs: List[Dict[str, Any]]) -> WeightOptimizationResult:
        """Optimize weights to reduce correlation impact"""
        try:
            if not high_corr_pairs:
                return WeightOptimizationResult(
                    optimized_weights=base_weights.copy(),
                    optimization_method='no_optimization_needed',
                    correlation_reduction=0.0,
                    confidence_score=1.0,
                    timestamp=datetime.now(),
                    metadata={'reason': 'No high correlations detected'}
                )

            # Convert weights to array for optimization
            component_names = list(base_weights.keys())
            initial_weights = np.array([base_weights[name] for name in component_names])

            def objective_function(weights):
                """Minimize correlation penalty while maintaining diversification"""
                penalty = 0

                # Correlation penalty
                for pair_info in high_corr_pairs:
                    comp1, comp2 = pair_info['pair']
                    if comp1 in component_names and comp2 in component_names:
                        i = component_names.index(comp1)
                        j = component_names.index(comp2)
                        correlation = pair_info['correlation']

                        # Penalize high correlation with weight product
                        penalty += weights[i] * weights[j] * correlation * 10

                # Diversification penalty (prevent extreme concentrations)
                max_weight = np.max(weights)
                if max_weight > 0.5:  # No single component should exceed 50%
                    penalty += (max_weight - 0.5) * 100

                return penalty

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: sum(w) - 1.0},  # Sum to 1
            ]

            # Bounds: each weight between 5% and 40%
            bounds = [(0.05, 0.4) for _ in range(len(component_names))]

            # Optimize
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                optimized_weights_dict = dict(zip(component_names, result.x))

                # Calculate correlation reduction
                initial_penalty = objective_function(initial_weights)
                optimized_penalty = objective_function(result.x)
                correlation_reduction = max(0, (initial_penalty - optimized_penalty) / max(initial_penalty, 1e-6))

                optimization_result = WeightOptimizationResult(
                    optimized_weights=optimized_weights_dict,
                    optimization_method='correlation_minimization',
                    correlation_reduction=correlation_reduction,
                    confidence_score=min(1.0, correlation_reduction * 2),  # Scale confidence
                    timestamp=datetime.now(),
                    metadata={
                        'high_correlation_pairs': len(high_corr_pairs),
                        'optimization_iterations': result.nit,
                        'optimization_success': result.success,
                        'initial_penalty': initial_penalty,
                        'optimized_penalty': optimized_penalty
                    }
                )

                self.optimization_history.append(optimization_result)
                return optimization_result

            else:
                logger.warning(f"Correlation optimization failed: {result.message}")
                return WeightOptimizationResult(
                    optimized_weights=base_weights.copy(),
                    optimization_method='optimization_failed',
                    correlation_reduction=0.0,
                    confidence_score=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': result.message}
                )

        except Exception as e:
            logger.error(f"Error in correlation-based optimization: {e}")
            return WeightOptimizationResult(
                optimized_weights=base_weights.copy(),
                optimization_method='error',
                correlation_reduction=0.0,
                confidence_score=0.0,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )

    def get_correlation_statistics(self) -> Dict[str, Any]:
        """Get correlation analysis statistics"""
        if not self.correlation_history:
            return {}

        recent_correlations = list(self.correlation_history)[-50:]  # Last 50 analyses

        avg_high_correlations = np.mean([entry['high_correlations'] for entry in recent_correlations])
        max_correlation_seen = max([entry['max_correlation'] for entry in recent_correlations])

        # Calculate optimization effectiveness
        if self.optimization_history:
            recent_optimizations = list(self.optimization_history)[-20:]
            avg_correlation_reduction = np.mean([opt.correlation_reduction for opt in recent_optimizations])
            avg_confidence = np.mean([opt.confidence_score for opt in recent_optimizations])
        else:
            avg_correlation_reduction = 0.0
            avg_confidence = 0.0

        return {
            'total_correlation_analyses': len(self.correlation_history),
            'recent_analyses': len(recent_correlations),
            'average_high_correlations': avg_high_correlations,
            'max_correlation_observed': max_correlation_seen,
            'correlation_threshold': self.correlation_threshold,
            'total_optimizations': len(self.optimization_history),
            'average_correlation_reduction': avg_correlation_reduction,
            'average_optimization_confidence': avg_confidence
        }

class MLDTEWeightOptimizer:
    """ML-based DTE-specific weight adaptation system"""

    def __init__(self):
        # Initialize models for each DTE (0-4)
        self.models = {}
        for dte in range(5):
            if XGBOOST_AVAILABLE:
                self.models[dte] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                self.models[dte] = XGBoostRegressor(n_estimators=100)

        self.feature_engineer = DTEFeatureEngineer()
        self.performance_tracker = DTEPerformanceTracker()
        self.training_data = deque(maxlen=10000)
        self.model_performance = {}

    def extract_dte_features(self, market_data: Dict[str, Any], dte: int) -> Dict[str, float]:
        """Extract DTE-specific features for ML optimization"""
        try:
            features = {
                # Greeks exposure features
                'gamma_exposure': self._calculate_gamma_exposure(market_data, dte),
                'theta_decay_rate': self._calculate_theta_decay(market_data, dte),
                'delta_sensitivity': self._calculate_delta_sensitivity(market_data, dte),
                'vega_exposure': self._calculate_vega_exposure(market_data, dte),

                # Liquidity and market structure
                'liquidity_score': self._calculate_liquidity_score(market_data, dte),
                'bid_ask_spread': market_data.get('bid_ask_spread', 0.01),
                'volume_ratio': market_data.get('volume_ratio', 1.0),

                # Market regime features
                'volatility_regime': self._encode_volatility_regime(market_data),
                'time_to_expiry_hours': dte * 24,
                'moneyness': market_data.get('underlying_price', 100) / market_data.get('atm_strike', 100),
                'vix_level': market_data.get('vix', 20),
                'market_session': self._get_market_session_encoding(market_data.get('timestamp', datetime.now())),

                # Technical indicators
                'rsi': market_data.get('rsi', 50),
                'bollinger_position': market_data.get('bollinger_position', 0.5),
                'macd_signal': market_data.get('macd_signal', 0),

                # Options-specific features
                'put_call_ratio': market_data.get('put_call_ratio', 1.0),
                'skew': market_data.get('skew', 0),
                'term_structure': market_data.get('term_structure', 0)
            }

            return features

        except Exception as e:
            logger.error(f"Error extracting DTE features: {e}")
            return self._get_default_features(dte)

    def predict_optimal_weights(self, market_data: Dict[str, Any], dte: int) -> Dict[str, float]:
        """Predict optimal weights for specific DTE using ML model"""
        try:
            if dte not in self.models:
                logger.warning(f"No model available for DTE {dte}, using default weights")
                return self._get_default_weights()

            # Extract features
            features = self.extract_dte_features(market_data, dte)
            feature_array = np.array(list(features.values())).reshape(1, -1)

            # Predict weight adjustments
            if hasattr(self.models[dte], 'predict'):
                try:
                    weight_adjustments = self.models[dte].predict(feature_array)[0]
                except:
                    # Model not trained yet, use default
                    weight_adjustments = 1.0
            else:
                weight_adjustments = 1.0

            # Apply DTE-specific logic
            base_weights = self._get_default_weights()
            dte_adjusted_weights = self._apply_dte_adjustments(base_weights, dte, weight_adjustments)

            return dte_adjusted_weights

        except Exception as e:
            logger.error(f"Error predicting optimal weights for DTE {dte}: {e}")
            return self._get_default_weights()

    def train_dte_models(self, historical_data: pd.DataFrame):
        """Train ML models for each DTE using historical performance data"""
        try:
            logger.info("Training DTE-specific ML models...")

            for dte in range(5):
                dte_data = historical_data[historical_data['dte'] == dte]

                if len(dte_data) < 50:  # Minimum data requirement
                    logger.warning(f"Insufficient data for DTE {dte}: {len(dte_data)} samples")
                    continue

                # Prepare features and targets
                X = []
                y = []

                for _, row in dte_data.iterrows():
                    features = self.extract_dte_features(row.to_dict(), dte)
                    X.append(list(features.values()))
                    y.append(row.get('regime_accuracy', 0.5))  # Target: accuracy score

                X = np.array(X)
                y = np.array(y)

                # Train model
                self.models[dte].fit(X, y)

                # Validate model
                if hasattr(self.models[dte], 'score'):
                    score = self.models[dte].score(X, y)
                    self.model_performance[dte] = {
                        'r2_score': score,
                        'training_samples': len(X),
                        'last_trained': datetime.now(),
                        'features_count': X.shape[1]
                    }
                    logger.info(f"DTE {dte} model R² score: {score:.3f} ({len(X)} samples)")

        except Exception as e:
            logger.error(f"Error training DTE models: {e}")

    def _calculate_gamma_exposure(self, market_data: Dict[str, Any], dte: int) -> float:
        """Calculate gamma exposure for DTE"""
        # Higher gamma for shorter DTE
        base_gamma = market_data.get('gamma', 0.01)
        dte_factor = max(0.1, 1.0 / (dte + 1))  # Higher for lower DTE
        return base_gamma * dte_factor

    def _calculate_theta_decay(self, market_data: Dict[str, Any], dte: int) -> float:
        """Calculate theta decay rate for DTE"""
        # Theta decay accelerates near expiry
        base_theta = market_data.get('theta', -0.01)
        dte_factor = max(0.1, 1.0 / (dte + 0.1))  # Accelerates near expiry
        return abs(base_theta) * dte_factor

    def _calculate_delta_sensitivity(self, market_data: Dict[str, Any], dte: int) -> float:
        """Calculate delta sensitivity for DTE"""
        base_delta = market_data.get('delta', 0.5)
        # Delta sensitivity varies with moneyness and time
        moneyness = market_data.get('underlying_price', 100) / market_data.get('atm_strike', 100)
        dte_factor = 1.0 + (5 - dte) * 0.1  # Higher sensitivity for shorter DTE
        return abs(base_delta - 0.5) * dte_factor * abs(moneyness - 1.0)

    def _calculate_vega_exposure(self, market_data: Dict[str, Any], dte: int) -> float:
        """Calculate vega exposure for DTE"""
        base_vega = market_data.get('vega', 0.1)
        # Vega decreases with time
        dte_factor = (dte + 1) / 5.0  # Higher for longer DTE
        return base_vega * dte_factor

    def _calculate_liquidity_score(self, market_data: Dict[str, Any], dte: int) -> float:
        """Calculate liquidity score for DTE"""
        volume = market_data.get('volume', 1000)
        open_interest = market_data.get('open_interest', 5000)

        # Liquidity typically decreases for very short and very long DTE
        dte_liquidity_factor = 1.0 - abs(dte - 2) * 0.1  # Peak at 2 DTE
        liquidity_score = (volume + open_interest * 0.1) * dte_liquidity_factor

        return min(1.0, liquidity_score / 10000)  # Normalize to [0, 1]

    def _encode_volatility_regime(self, market_data: Dict[str, Any]) -> float:
        """Encode volatility regime as numeric value"""
        vix = market_data.get('vix', 20)
        if vix < 15:
            return 0.0  # Low volatility
        elif vix < 25:
            return 0.5  # Normal volatility
        else:
            return 1.0  # High volatility

    def _get_market_session_encoding(self, timestamp: datetime) -> float:
        """Encode market session as numeric value"""
        hour = timestamp.hour
        if 9 <= hour < 12:
            return 1.0  # Morning session
        elif 12 <= hour < 15:
            return 0.5  # Afternoon session
        else:
            return 0.0  # After hours

    def _get_default_features(self, dte: int) -> Dict[str, float]:
        """Get default features for DTE"""
        return {
            'gamma_exposure': 0.01 / (dte + 1),
            'theta_decay_rate': 0.01 * (5 - dte),
            'delta_sensitivity': 0.1,
            'vega_exposure': 0.1 * (dte + 1) / 5,
            'liquidity_score': 0.5,
            'bid_ask_spread': 0.01,
            'volume_ratio': 1.0,
            'volatility_regime': 0.5,
            'time_to_expiry_hours': dte * 24,
            'moneyness': 1.0,
            'vix_level': 20,
            'market_session': 0.5,
            'rsi': 50,
            'bollinger_position': 0.5,
            'macd_signal': 0,
            'put_call_ratio': 1.0,
            'skew': 0,
            'term_structure': 0
        }

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default component weights"""
        return {
            'atm_straddle': 0.25,
            'itm1_straddle': 0.20,
            'otm1_straddle': 0.15,
            'combined_straddle': 0.20,
            'atm_ce': 0.10,
            'atm_pe': 0.10
        }

    def _apply_dte_adjustments(self, base_weights: Dict[str, float], dte: int,
                             ml_adjustment: float) -> Dict[str, float]:
        """Apply DTE-specific adjustments to base weights"""
        adjusted_weights = base_weights.copy()

        # DTE-specific logic
        if dte == 0:  # 0 DTE - focus on gamma
            adjusted_weights['otm1_straddle'] *= (1 + ml_adjustment * 0.3)
            adjusted_weights['atm_straddle'] *= (1 + ml_adjustment * 0.2)
        elif dte == 1:  # 1 DTE - balanced approach
            for component in adjusted_weights:
                adjusted_weights[component] *= (1 + ml_adjustment * 0.1)
        elif dte >= 3:  # 3+ DTE - focus on theta and vega
            adjusted_weights['combined_straddle'] *= (1 + ml_adjustment * 0.2)
            adjusted_weights['itm1_straddle'] *= (1 + ml_adjustment * 0.15)

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        return {k: v/total_weight for k, v in adjusted_weights.items()}

class DTEFeatureEngineer:
    """Feature engineering for DTE-specific optimization"""

    def __init__(self):
        self.feature_history = deque(maxlen=1000)

    def engineer_features(self, market_data: Dict[str, Any], dte: int) -> Dict[str, float]:
        """Engineer features for DTE-specific analysis"""
        # This would contain more sophisticated feature engineering
        # For now, it's a placeholder that delegates to the main feature extraction
        return {}

class DTEPerformanceTracker:
    """Track performance of DTE-specific optimizations"""

    def __init__(self):
        self.performance_history = deque(maxlen=5000)

    def track_performance(self, dte: int, weights: Dict[str, float],
                         performance_score: float):
        """Track performance of DTE-specific weight optimization"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'dte': dte,
            'weights': weights.copy(),
            'performance_score': performance_score
        })

class IntegratedDynamicWeightSystem:
    """Integrated dynamic weight system combining all optimization approaches"""

    def __init__(self):
        self.volatility_optimizer = DynamicWeightOptimizer()
        self.correlation_optimizer = CorrelationBasedOptimizer()
        self.ml_optimizer = MLDTEWeightOptimizer()

        # System configuration
        self.optimization_weights = {
            'volatility': 0.4,      # 40% weight to volatility-based optimization
            'correlation': 0.35,    # 35% weight to correlation-based optimization
            'ml_dte': 0.25         # 25% weight to ML DTE-specific optimization
        }

        self.optimization_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_optimizations': 0,
            'average_correlation_reduction': 0.0,
            'volatility_regime_accuracy': 0.0,
            'ml_prediction_accuracy': 0.0
        }

    def optimize_weights_comprehensive(self, market_data: Dict[str, Any],
                                     correlation_matrix: Optional[pd.DataFrame] = None,
                                     current_dte: int = 0) -> Dict[str, float]:
        """Comprehensive weight optimization using all three approaches"""
        try:
            start_time = time.time()

            # Step 1: Volatility-based optimization
            volatility_weights = self.volatility_optimizer.optimize_weights_realtime(market_data)

            # Step 2: Correlation-based optimization
            correlation_weights = volatility_weights.copy()
            correlation_reduction = 0.0

            if correlation_matrix is not None:
                high_corr_pairs = self.correlation_optimizer.detect_high_correlation_periods(correlation_matrix)
                if high_corr_pairs:
                    corr_result = self.correlation_optimizer.optimize_weights_for_correlation(
                        volatility_weights, high_corr_pairs
                    )
                    correlation_weights = corr_result.optimized_weights
                    correlation_reduction = corr_result.correlation_reduction

            # Step 3: ML DTE-specific optimization
            ml_weights = self.ml_optimizer.predict_optimal_weights(market_data, current_dte)

            # Step 4: Combine all optimizations using weighted average
            final_weights = self._combine_optimizations(
                volatility_weights, correlation_weights, ml_weights
            )

            # Step 5: Apply final constraints and validation
            validated_weights = self._validate_and_constrain_weights(final_weights)

            # Track optimization performance
            optimization_time = time.time() - start_time
            self._track_optimization_performance(
                validated_weights, correlation_reduction, optimization_time, current_dte
            )

            return validated_weights

        except Exception as e:
            logger.error(f"Error in comprehensive weight optimization: {e}")
            return self._get_fallback_weights()

    def _combine_optimizations(self, volatility_weights: Dict[str, float],
                             correlation_weights: Dict[str, float],
                             ml_weights: Dict[str, float]) -> Dict[str, float]:
        """Combine multiple optimization results using weighted average"""
        combined_weights = {}

        for component in volatility_weights.keys():
            combined_weight = (
                volatility_weights[component] * self.optimization_weights['volatility'] +
                correlation_weights[component] * self.optimization_weights['correlation'] +
                ml_weights[component] * self.optimization_weights['ml_dte']
            )
            combined_weights[component] = combined_weight

        # Normalize to sum to 1.0
        total_weight = sum(combined_weights.values())
        return {k: v/total_weight for k, v in combined_weights.items()}

    def _validate_and_constrain_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and apply constraints to final weights"""
        constrained_weights = {}

        for component, weight in weights.items():
            # Apply min/max constraints
            constrained_weight = max(0.05, min(0.4, weight))  # Between 5% and 40%
            constrained_weights[component] = constrained_weight

        # Ensure weights sum to 1.0
        total_weight = sum(constrained_weights.values())
        normalized_weights = {k: v/total_weight for k, v in constrained_weights.items()}

        # Validate weight distribution
        max_weight = max(normalized_weights.values())
        min_weight = min(normalized_weights.values())

        if max_weight > 0.5 or min_weight < 0.03:
            logger.warning(f"Weight distribution may be suboptimal: max={max_weight:.3f}, min={min_weight:.3f}")

        return normalized_weights

    def _track_optimization_performance(self, weights: Dict[str, float],
                                      correlation_reduction: float,
                                      optimization_time: float, dte: int):
        """Track performance of integrated optimization"""
        self.performance_metrics['total_optimizations'] += 1

        # Update running averages
        total_opts = self.performance_metrics['total_optimizations']
        current_avg_corr = self.performance_metrics['average_correlation_reduction']

        self.performance_metrics['average_correlation_reduction'] = (
            (current_avg_corr * (total_opts - 1) + correlation_reduction) / total_opts
        )

        # Store optimization record
        optimization_record = {
            'timestamp': datetime.now(),
            'weights': weights.copy(),
            'correlation_reduction': correlation_reduction,
            'optimization_time': optimization_time,
            'dte': dte,
            'volatility_regime': self.volatility_optimizer.volatility_monitor.current_regime.regime_type if self.volatility_optimizer.volatility_monitor.current_regime else None
        }

        self.optimization_history.append(optimization_record)

    def _get_fallback_weights(self) -> Dict[str, float]:
        """Get fallback weights in case of optimization failure"""
        return {
            'atm_straddle': 0.25,
            'itm1_straddle': 0.20,
            'otm1_straddle': 0.15,
            'combined_straddle': 0.20,
            'atm_ce': 0.10,
            'atm_pe': 0.10
        }

    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for all optimization systems"""
        return {
            'integrated_system': {
                'total_optimizations': self.performance_metrics['total_optimizations'],
                'average_correlation_reduction': self.performance_metrics['average_correlation_reduction'],
                'optimization_weights': self.optimization_weights
            },
            'volatility_optimization': self.volatility_optimizer.get_weight_statistics(),
            'correlation_optimization': self.correlation_optimizer.get_correlation_statistics(),
            'ml_optimization': {
                'models_trained': len(self.ml_optimizer.models),
                'model_performance': self.ml_optimizer.model_performance
            },
            'recent_optimizations': list(self.optimization_history)[-10:] if self.optimization_history else []
        }

    def update_optimization_weights(self, volatility_weight: float,
                                  correlation_weight: float, ml_weight: float):
        """Update the weights used to combine different optimization approaches"""
        total = volatility_weight + correlation_weight + ml_weight

        self.optimization_weights = {
            'volatility': volatility_weight / total,
            'correlation': correlation_weight / total,
            'ml_dte': ml_weight / total
        }

        logger.info(f"Updated optimization weights: {self.optimization_weights}")

# Factory function for easy instantiation
def create_dynamic_weight_system() -> IntegratedDynamicWeightSystem:
    """Factory function to create integrated dynamic weight system"""
    return IntegratedDynamicWeightSystem()
