"""Dynamic Weight Engine for Enhanced OI System."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque

from .enhanced_models import DynamicWeightConfig, FactorConfig, PerformanceMetrics

logger = logging.getLogger(__name__)

class DynamicWeightEngine:
    """Core engine for dynamic weight calculation and adjustment."""
    
    def __init__(self, weight_config: DynamicWeightConfig, factor_configs: List[FactorConfig]):
        """Initialize the dynamic weight engine."""
        self.config = weight_config
        self.factor_configs = {fc.factor_name: fc for fc in factor_configs}
        
        # Initialize weights with base configuration
        self.current_weights = self._initialize_weights()
        self.weight_history = deque(maxlen=1000)  # Store weight history
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))  # Fixed maxlen to integer
        self.factor_performance = defaultdict(float)
        self.correlation_matrix = pd.DataFrame()
        
        # State tracking
        self.last_adjustment = None
        self.adjustment_count = 0
        self.total_weight_change = 0.0
        
        logger.info("Dynamic Weight Engine initialized")
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize weights with base configuration."""
        # Start with factor-specific weights from configuration
        all_weights = {}

        # Add factor-specific weights first (these take priority)
        for factor_name, factor_config in self.factor_configs.items():
            all_weights[factor_name] = factor_config.base_weight

        # Add base weights only if not already specified in factor configs
        base_weights = {
            'oi_factor': self.config.oi_factor_weight,
            'coi_factor': self.config.coi_factor_weight,
            'greek_factor': self.config.greek_factor_weight,
            'market_factor': self.config.market_factor_weight,
            'performance_factor': self.config.performance_factor_weight
        }

        for factor_name, weight in base_weights.items():
            if factor_name not in all_weights:
                all_weights[factor_name] = weight

        # Add sub-factor weights only if no main factors are specified
        if not any(factor in all_weights for factor in base_weights.keys()):
            oi_sub_weights = {
                'current_oi': self.config.current_oi_weight,
                'oi_concentration': self.config.oi_concentration_weight,
                'oi_distribution': self.config.oi_distribution_weight,
                'oi_momentum': self.config.oi_momentum_weight,
                'oi_trend': self.config.oi_trend_weight,
                'oi_seasonal': self.config.oi_seasonal_weight,
                'oi_liquidity': self.config.oi_liquidity_weight,
                'oi_anomaly': self.config.oi_anomaly_weight
            }
            all_weights.update(oi_sub_weights)

        return all_weights
    
    def calculate_factor_performance(self, factor_name: str, factor_data: pd.Series, 
                                   returns: pd.Series) -> float:
        """Calculate performance metric for a specific factor."""
        try:
            if len(factor_data) < 2 or len(returns) < 2:
                return 0.0
            
            # Align data
            aligned_data = pd.concat([factor_data, returns], axis=1, join='inner')
            if aligned_data.empty:
                return 0.0
            
            factor_values = aligned_data.iloc[:, 0]
            return_values = aligned_data.iloc[:, 1]
            
            # Calculate correlation as performance metric
            correlation = factor_values.corr(return_values)
            
            # Handle NaN correlation
            if pd.isna(correlation):
                return 0.0
            
            # Convert correlation to performance score (0 to 1)
            performance = (correlation + 1) / 2  # Scale from [-1,1] to [0,1]
            
            return performance
            
        except Exception as e:
            logger.warning(f"Error calculating performance for {factor_name}: {e}")
            return 0.0
    
    def update_weights(self, performance_data: Dict[str, float], 
                      market_conditions: Dict[str, float],
                      factor_data: Dict[str, pd.Series] = None) -> Dict[str, float]:
        """Update weights based on performance and market conditions."""
        try:
            # Calculate performance-based adjustments
            performance_adjustments = self._calculate_performance_adjustments(performance_data)
            
            # Calculate market condition adjustments
            market_adjustments = self._calculate_market_adjustments(market_conditions)
            
            # Calculate correlation-based adjustments
            correlation_adjustments = self._calculate_correlation_adjustments(factor_data)
            
            # Apply adjustments with learning rate
            new_weights = {}
            total_adjustment = 0.0
            
            for factor, current_weight in self.current_weights.items():
                # Combine all adjustments
                perf_adj = performance_adjustments.get(factor, 1.0)
                market_adj = market_adjustments.get(factor, 1.0)
                corr_adj = correlation_adjustments.get(factor, 1.0)
                
                # Calculate combined adjustment
                combined_adjustment = perf_adj * market_adj * corr_adj
                
                # Apply learning rate
                weight_change = self.config.weight_learning_rate * (combined_adjustment - 1.0) * current_weight
                new_weight = current_weight + weight_change
                
                # Apply constraints
                factor_config = self.factor_configs.get(factor)
                if factor_config:
                    min_weight = factor_config.min_weight
                    max_weight = factor_config.max_weight
                else:
                    min_weight = self.config.min_factor_weight
                    max_weight = self.config.max_factor_weight
                
                new_weight = max(min_weight, min(max_weight, new_weight))
                new_weights[factor] = new_weight
                
                total_adjustment += abs(weight_change)
            
            # Normalize weights to maintain total weight constraint
            new_weights = self._normalize_weights(new_weights)
            
            # Apply smoothing
            smoothed_weights = self._apply_smoothing(new_weights)
            
            # Update state
            self.current_weights = smoothed_weights
            self.last_adjustment = datetime.now()
            self.adjustment_count += 1
            self.total_weight_change += total_adjustment
            
            # Store weight history
            self.weight_history.append({
                'timestamp': datetime.now(),
                'weights': smoothed_weights.copy(),
                'adjustment': total_adjustment
            })
            
            logger.info(f"Weights updated. Total adjustment: {total_adjustment:.4f}")
            
            return self.current_weights.copy()
            
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
            return self.current_weights.copy()
    
    def _calculate_performance_adjustments(self, performance_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate weight adjustments based on performance."""
        adjustments = {}
        
        for factor, performance in performance_data.items():
            if factor in self.current_weights:
                # Store performance history
                self.performance_history[factor].append(performance)
                
                # Calculate adjustment based on performance
                if performance > 0.6:  # Good performance
                    adjustment = 1.0 + min(0.2, (performance - 0.6) * 0.5)
                elif performance < 0.4:  # Poor performance
                    adjustment = 1.0 - min(0.2, (0.4 - performance) * 0.5)
                else:  # Neutral performance
                    adjustment = 1.0
                
                adjustments[factor] = adjustment
        
        return adjustments
    
    def _calculate_market_adjustments(self, market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate weight adjustments based on market conditions."""
        adjustments = {}
        
        volatility = market_conditions.get('volatility', 0.5)
        trend_strength = market_conditions.get('trend_strength', 0.0)
        liquidity = market_conditions.get('liquidity', 0.5)
        regime = market_conditions.get('regime', 'normal')
        
        # Base adjustments for all factors
        base_adjustments = {
            'oi_factor': 1.0 + (liquidity - 0.5) * 0.2,
            'coi_factor': 1.0 + (volatility - 0.5) * 0.3,
            'greek_factor': 1.0 + (volatility - 0.5) * 0.4,
            'market_factor': 1.0 + abs(trend_strength) * 0.2,
            'performance_factor': 1.0
        }
        
        # Apply regime-specific adjustments
        if regime == 'high_volatility':
            base_adjustments['greek_factor'] *= self.config.volatility_adjustment
            base_adjustments['coi_factor'] *= 1.1
        elif regime == 'trending':
            base_adjustments['market_factor'] *= self.config.trend_adjustment
            base_adjustments['oi_factor'] *= 0.9
        elif regime == 'sideways':
            base_adjustments['oi_factor'] *= 1.1
            base_adjustments['market_factor'] *= 0.9
        
        # Apply to all weights
        for factor in self.current_weights:
            if factor in base_adjustments:
                adjustments[factor] = base_adjustments[factor]
            else:
                adjustments[factor] = 1.0
        
        return adjustments
    
    def _calculate_correlation_adjustments(self, factor_data: Dict[str, pd.Series] = None) -> Dict[str, float]:
        """Calculate adjustments based on factor correlations."""
        adjustments = {}
        
        if not factor_data or len(factor_data) < 2:
            return {factor: 1.0 for factor in self.current_weights}
        
        try:
            # Create correlation matrix
            df = pd.DataFrame(factor_data)
            self.correlation_matrix = df.corr()
            
            # Calculate diversification adjustments
            for factor in self.current_weights:
                if factor in self.correlation_matrix.columns:
                    # Calculate average correlation with other factors
                    correlations = self.correlation_matrix[factor].drop(factor)
                    avg_correlation = abs(correlations).mean()
                    
                    # Adjust based on correlation
                    if avg_correlation > self.config.correlation_threshold:
                        # High correlation - reduce weight
                        adjustment = 1.0 - (avg_correlation - self.config.correlation_threshold) * 0.5
                    else:
                        # Low correlation - increase weight (diversification bonus)
                        adjustment = self.config.diversification_bonus
                    
                    adjustments[factor] = adjustment
                else:
                    adjustments[factor] = 1.0
        
        except Exception as e:
            logger.warning(f"Error calculating correlation adjustments: {e}")
            adjustments = {factor: 1.0 for factor in self.current_weights}
        
        return adjustments
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to maintain total weight constraint."""
        # Separate main factors and sub-factors
        main_factors = ['oi_factor', 'coi_factor', 'greek_factor', 'market_factor', 'performance_factor']
        oi_sub_factors = ['current_oi', 'oi_concentration', 'oi_distribution', 'oi_momentum',
                         'oi_trend', 'oi_seasonal', 'oi_liquidity', 'oi_anomaly']

        # Create a copy to avoid modifying the original
        normalized_weights = weights.copy()

        # Normalize main factors to sum to 1.0
        main_weights = {factor: normalized_weights.get(factor, 0) for factor in main_factors if factor in normalized_weights}
        main_total = sum(main_weights.values())

        if main_total > 0:
            for factor in main_factors:
                if factor in normalized_weights:
                    normalized_weights[factor] = normalized_weights[factor] / main_total

        # Normalize sub-factors within their groups to sum to 1.0
        oi_sub_weights = {factor: normalized_weights.get(factor, 0) for factor in oi_sub_factors if factor in normalized_weights}
        oi_sub_total = sum(oi_sub_weights.values())

        if oi_sub_total > 0:
            for factor in oi_sub_factors:
                if factor in normalized_weights:
                    normalized_weights[factor] = normalized_weights[factor] / oi_sub_total

        return normalized_weights
    
    def _apply_smoothing(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply smoothing to weight changes."""
        smoothed_weights = {}
        
        for factor, new_weight in new_weights.items():
            current_weight = self.current_weights.get(factor, new_weight)
            
            # Apply exponential smoothing
            smoothed_weight = (
                (1 - self.config.weight_decay_factor) * new_weight +
                self.config.weight_decay_factor * current_weight
            )
            
            smoothed_weights[factor] = smoothed_weight
        
        return smoothed_weights
    
    def should_rebalance(self) -> bool:
        """Check if weights should be rebalanced."""
        if self.last_adjustment is None:
            return True
        
        time_since_last = datetime.now() - self.last_adjustment
        return time_since_last.total_seconds() >= self.config.weight_rebalance_freq
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current factor weights."""
        return self.current_weights.copy()
    
    def get_weight_history(self) -> List[Dict]:
        """Get weight adjustment history."""
        return list(self.weight_history)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the weight engine."""
        return {
            'adjustment_count': self.adjustment_count,
            'total_weight_change': self.total_weight_change,
            'avg_weight_change': self.total_weight_change / max(1, self.adjustment_count),
            'last_adjustment': self.last_adjustment,
            'current_weights': self.current_weights.copy(),
            'factor_performance': dict(self.factor_performance),
            'correlation_matrix': self.correlation_matrix.to_dict() if not self.correlation_matrix.empty else {}
        }
