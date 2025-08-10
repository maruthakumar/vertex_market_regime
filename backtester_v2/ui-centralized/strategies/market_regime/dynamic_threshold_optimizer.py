"""
Dynamic Threshold Optimizer for Enhanced Market Regime Formation

This module implements dynamic threshold optimization to replace static thresholds
with adaptive market-condition-based thresholds for improved signal quality.

Features:
1. Volatility-based threshold adjustment
2. Time-of-day threshold optimization
3. Volume-based threshold scaling
4. Market regime adaptive thresholds
5. Performance-based threshold learning
6. Real-time threshold calibration
7. Production-ready optimization

Author: The Augster
Date: 2025-01-16
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ThresholdOptimizationResult:
    """Result structure for threshold optimization"""
    oi_threshold: float
    price_threshold: float
    volatility_multiplier: float
    time_factor: float
    volume_factor: float
    regime_factor: float
    confidence: float
    optimization_reason: str
    supporting_metrics: Dict[str, Any]
    calculation_timestamp: datetime

class DynamicThresholdOptimizer:
    """
    Dynamic Threshold Optimizer
    
    Optimizes OI and price thresholds based on current market conditions including
    volatility, time of day, volume, and market regime for improved signal quality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Dynamic Threshold Optimizer"""
        self.config = config or {}
        
        # Base threshold configuration
        self.base_oi_threshold = float(self.config.get('base_oi_threshold', 0.02))  # 2%
        self.base_price_threshold = float(self.config.get('base_price_threshold', 0.01))  # 1%
        
        # Volatility adjustment configuration
        self.volatility_adjustment_enabled = self.config.get('volatility_adjustment_enabled', True)
        self.low_volatility_threshold = float(self.config.get('low_volatility_threshold', 0.10))
        self.high_volatility_threshold = float(self.config.get('high_volatility_threshold', 0.25))
        self.max_volatility_multiplier = float(self.config.get('max_volatility_multiplier', 2.0))
        self.min_volatility_multiplier = float(self.config.get('min_volatility_multiplier', 0.5))
        
        # Time-based adjustment configuration
        self.time_adjustment_enabled = self.config.get('time_adjustment_enabled', True)
        self.market_open_start = time(9, 15)   # Market open
        self.market_open_end = time(10, 30)    # End of opening session
        self.market_close_start = time(14, 30) # Start of closing session
        self.market_close_end = time(15, 30)   # Market close
        
        # Time-based multipliers
        self.opening_multiplier = float(self.config.get('opening_multiplier', 1.2))
        self.closing_multiplier = float(self.config.get('closing_multiplier', 1.3))
        self.mid_session_multiplier = float(self.config.get('mid_session_multiplier', 1.0))
        
        # Volume adjustment configuration
        self.volume_adjustment_enabled = self.config.get('volume_adjustment_enabled', True)
        self.high_volume_threshold = float(self.config.get('high_volume_threshold', 1.5))  # 1.5x average
        self.low_volume_threshold = float(self.config.get('low_volume_threshold', 0.5))   # 0.5x average
        self.max_volume_multiplier = float(self.config.get('max_volume_multiplier', 1.3))
        self.min_volume_multiplier = float(self.config.get('min_volume_multiplier', 0.8))
        
        # Market regime adjustment configuration
        self.regime_adjustment_enabled = self.config.get('regime_adjustment_enabled', True)
        self.trending_regime_multiplier = float(self.config.get('trending_regime_multiplier', 1.1))
        self.volatile_regime_multiplier = float(self.config.get('volatile_regime_multiplier', 1.4))
        self.consolidation_regime_multiplier = float(self.config.get('consolidation_regime_multiplier', 0.8))
        
        # Performance-based learning configuration
        self.performance_learning_enabled = self.config.get('performance_learning_enabled', True)
        self.learning_rate = float(self.config.get('learning_rate', 0.01))
        self.performance_window = int(self.config.get('performance_window', 50))
        
        # Historical performance tracking
        self.performance_history = deque(maxlen=self.performance_window)
        self.threshold_performance_map = {}
        
        # Optimization bounds
        self.min_oi_threshold = float(self.config.get('min_oi_threshold', 0.005))    # 0.5%
        self.max_oi_threshold = float(self.config.get('max_oi_threshold', 0.10))     # 10%
        self.min_price_threshold = float(self.config.get('min_price_threshold', 0.002)) # 0.2%
        self.max_price_threshold = float(self.config.get('max_price_threshold', 0.05))  # 5%
        
        logger.info(f"Dynamic Threshold Optimizer initialized with base thresholds: OI={self.base_oi_threshold:.3f}, Price={self.base_price_threshold:.3f}")
    
    def calculate_adaptive_thresholds(self, market_data: Dict[str, Any]) -> ThresholdOptimizationResult:
        """
        Calculate adaptive thresholds based on current market conditions
        
        Args:
            market_data: Market data including volatility, volume, time, and regime info
            
        Returns:
            ThresholdOptimizationResult with optimized thresholds and factors
        """
        try:
            # Initialize factors
            volatility_multiplier = 1.0
            time_factor = 1.0
            volume_factor = 1.0
            regime_factor = 1.0
            optimization_reasons = []
            
            # 1. Volatility-based adjustment
            if self.volatility_adjustment_enabled:
                volatility_multiplier = self._calculate_volatility_multiplier(market_data)
                if volatility_multiplier != 1.0:
                    optimization_reasons.append(f"volatility_adj_{volatility_multiplier:.2f}")
            
            # 2. Time-based adjustment
            if self.time_adjustment_enabled:
                time_factor = self._calculate_time_factor(market_data)
                if time_factor != 1.0:
                    optimization_reasons.append(f"time_adj_{time_factor:.2f}")
            
            # 3. Volume-based adjustment
            if self.volume_adjustment_enabled:
                volume_factor = self._calculate_volume_factor(market_data)
                if volume_factor != 1.0:
                    optimization_reasons.append(f"volume_adj_{volume_factor:.2f}")
            
            # 4. Market regime adjustment
            if self.regime_adjustment_enabled:
                regime_factor = self._calculate_regime_factor(market_data)
                if regime_factor != 1.0:
                    optimization_reasons.append(f"regime_adj_{regime_factor:.2f}")
            
            # 5. Performance-based learning adjustment
            performance_factor = 1.0
            if self.performance_learning_enabled:
                performance_factor = self._calculate_performance_factor(market_data)
                if performance_factor != 1.0:
                    optimization_reasons.append(f"performance_adj_{performance_factor:.2f}")
            
            # Calculate combined multiplier
            combined_multiplier = (
                volatility_multiplier * 
                time_factor * 
                volume_factor * 
                regime_factor * 
                performance_factor
            )
            
            # Calculate final thresholds
            oi_threshold = self.base_oi_threshold * combined_multiplier
            price_threshold = self.base_price_threshold * combined_multiplier
            
            # Apply bounds
            oi_threshold = np.clip(oi_threshold, self.min_oi_threshold, self.max_oi_threshold)
            price_threshold = np.clip(price_threshold, self.min_price_threshold, self.max_price_threshold)
            
            # Calculate confidence based on data quality and factor consistency
            confidence = self._calculate_optimization_confidence(
                market_data, volatility_multiplier, time_factor, volume_factor, regime_factor
            )
            
            # Prepare optimization reason
            optimization_reason = "; ".join(optimization_reasons) if optimization_reasons else "no_adjustment"
            
            # Prepare supporting metrics
            supporting_metrics = {
                'base_oi_threshold': self.base_oi_threshold,
                'base_price_threshold': self.base_price_threshold,
                'combined_multiplier': combined_multiplier,
                'performance_factor': performance_factor,
                'bounds_applied': {
                    'oi_bounded': oi_threshold != self.base_oi_threshold * combined_multiplier,
                    'price_bounded': price_threshold != self.base_price_threshold * combined_multiplier
                },
                'market_conditions': {
                    'volatility': market_data.get('volatility', 0),
                    'volume_ratio': market_data.get('volume', 0) / max(market_data.get('avg_volume', 1), 1),
                    'current_time': market_data.get('timestamp', datetime.now()).time(),
                    'market_regime': market_data.get('market_regime', 'Unknown')
                }
            }
            
            # Create result
            result = ThresholdOptimizationResult(
                oi_threshold=oi_threshold,
                price_threshold=price_threshold,
                volatility_multiplier=volatility_multiplier,
                time_factor=time_factor,
                volume_factor=volume_factor,
                regime_factor=regime_factor,
                confidence=confidence,
                optimization_reason=optimization_reason,
                supporting_metrics=supporting_metrics,
                calculation_timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {e}")
            return self._get_default_threshold_result()
    
    def _calculate_volatility_multiplier(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based threshold multiplier"""
        try:
            volatility = market_data.get('volatility', 0.15)
            
            # Volatility-based adjustment logic
            if volatility <= self.low_volatility_threshold:
                # Low volatility: reduce thresholds for higher sensitivity
                multiplier = self.min_volatility_multiplier + (
                    (volatility / self.low_volatility_threshold) * 
                    (1.0 - self.min_volatility_multiplier)
                )
            elif volatility >= self.high_volatility_threshold:
                # High volatility: increase thresholds to reduce noise
                excess_volatility = min(volatility - self.high_volatility_threshold, 0.25)
                multiplier = 1.0 + (excess_volatility / 0.25) * (self.max_volatility_multiplier - 1.0)
            else:
                # Normal volatility: no adjustment
                multiplier = 1.0
            
            return np.clip(multiplier, self.min_volatility_multiplier, self.max_volatility_multiplier)
            
        except Exception as e:
            logger.error(f"Error calculating volatility multiplier: {e}")
            return 1.0
    
    def _calculate_time_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate time-based adjustment factor"""
        try:
            timestamp = market_data.get('timestamp', datetime.now())
            current_time = timestamp.time()
            
            # Market opening session (9:15-10:30): Higher sensitivity
            if self.market_open_start <= current_time <= self.market_open_end:
                return self.opening_multiplier
            
            # Market closing session (14:30-15:30): Highest sensitivity
            elif self.market_close_start <= current_time <= self.market_close_end:
                return self.closing_multiplier
            
            # Mid-session: Normal sensitivity
            else:
                return self.mid_session_multiplier
                
        except Exception as e:
            logger.error(f"Error calculating time factor: {e}")
            return 1.0
    
    def _calculate_volume_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume-based adjustment factor"""
        try:
            current_volume = market_data.get('volume', 0)
            avg_volume = market_data.get('avg_volume', current_volume)
            
            if avg_volume <= 0:
                return 1.0
            
            volume_ratio = current_volume / avg_volume
            
            # High volume: increase sensitivity (higher thresholds to avoid noise)
            if volume_ratio >= self.high_volume_threshold:
                excess_ratio = min(volume_ratio - self.high_volume_threshold, 2.0)
                factor = 1.0 + (excess_ratio / 2.0) * (self.max_volume_multiplier - 1.0)
            
            # Low volume: decrease sensitivity (lower thresholds)
            elif volume_ratio <= self.low_volume_threshold:
                deficit_ratio = self.low_volume_threshold - volume_ratio
                factor = 1.0 - (deficit_ratio / self.low_volume_threshold) * (1.0 - self.min_volume_multiplier)
            
            # Normal volume: no adjustment
            else:
                factor = 1.0
            
            return np.clip(factor, self.min_volume_multiplier, self.max_volume_multiplier)
            
        except Exception as e:
            logger.error(f"Error calculating volume factor: {e}")
            return 1.0
    
    def _calculate_regime_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate market regime-based adjustment factor"""
        try:
            market_regime = market_data.get('market_regime', '').upper()
            
            # Trending regimes: slightly higher thresholds for confirmation
            if any(trend in market_regime for trend in ['BULLISH', 'BEARISH', 'TRENDING']):
                return self.trending_regime_multiplier
            
            # Volatile regimes: higher thresholds to reduce noise
            elif any(vol in market_regime for vol in ['VOLATILE', 'VOLATILITY', 'EXPANSION']):
                return self.volatile_regime_multiplier
            
            # Consolidation regimes: lower thresholds for sensitivity
            elif any(cons in market_regime for cons in ['CONSOLIDATION', 'NEUTRAL', 'SIDEWAYS']):
                return self.consolidation_regime_multiplier
            
            # Unknown or other regimes: no adjustment
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating regime factor: {e}")
            return 1.0
    
    def _calculate_performance_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate performance-based learning factor"""
        try:
            if len(self.performance_history) < 10:
                return 1.0  # Not enough data for learning
            
            # Calculate recent performance
            recent_performance = np.mean(list(self.performance_history)[-10:])
            
            # If performance is poor, adjust thresholds
            if recent_performance < 0.5:
                # Poor performance: try different thresholds
                adjustment = (0.5 - recent_performance) * self.learning_rate * 2
                return 1.0 + adjustment
            elif recent_performance > 0.7:
                # Good performance: maintain current approach
                return 1.0
            else:
                # Average performance: slight adjustment
                adjustment = (recent_performance - 0.6) * self.learning_rate
                return 1.0 + adjustment
                
        except Exception as e:
            logger.error(f"Error calculating performance factor: {e}")
            return 1.0
    
    def _calculate_optimization_confidence(self, market_data: Dict[str, Any], 
                                         vol_mult: float, time_fact: float, 
                                         vol_fact: float, regime_fact: float) -> float:
        """Calculate confidence in threshold optimization"""
        try:
            confidence_factors = []
            
            # Data quality confidence
            volatility = market_data.get('volatility', 0)
            if volatility > 0:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.5)
            
            # Volume data confidence
            volume = market_data.get('volume', 0)
            avg_volume = market_data.get('avg_volume', 0)
            if volume > 0 and avg_volume > 0:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.7)
            
            # Time data confidence
            timestamp = market_data.get('timestamp')
            if timestamp:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.8)
            
            # Regime data confidence
            regime = market_data.get('market_regime', '')
            if regime and regime != 'Unknown':
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.6)
            
            # Factor consistency (factors shouldn't be too extreme)
            factors = [vol_mult, time_fact, vol_fact, regime_fact]
            factor_std = np.std(factors)
            consistency_confidence = max(0.3, 1.0 - factor_std)
            
            # Combined confidence
            data_confidence = np.mean(confidence_factors)
            combined_confidence = (data_confidence * 0.7 + consistency_confidence * 0.3)
            
            return np.clip(combined_confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating optimization confidence: {e}")
            return 0.5
    
    def update_performance(self, performance_score: float):
        """Update performance history for learning"""
        try:
            if 0.0 <= performance_score <= 1.0:
                self.performance_history.append(performance_score)
                logger.debug(f"Updated performance history: {performance_score:.3f}")
            else:
                logger.warning(f"Invalid performance score: {performance_score}")
                
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _get_default_threshold_result(self) -> ThresholdOptimizationResult:
        """Get default threshold result for error cases"""
        return ThresholdOptimizationResult(
            oi_threshold=self.base_oi_threshold,
            price_threshold=self.base_price_threshold,
            volatility_multiplier=1.0,
            time_factor=1.0,
            volume_factor=1.0,
            regime_factor=1.0,
            confidence=0.5,
            optimization_reason="default_fallback",
            supporting_metrics={},
            calculation_timestamp=datetime.now()
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get threshold optimization statistics"""
        try:
            return {
                'base_thresholds': {
                    'oi_threshold': self.base_oi_threshold,
                    'price_threshold': self.base_price_threshold
                },
                'adjustment_settings': {
                    'volatility_enabled': self.volatility_adjustment_enabled,
                    'time_enabled': self.time_adjustment_enabled,
                    'volume_enabled': self.volume_adjustment_enabled,
                    'regime_enabled': self.regime_adjustment_enabled,
                    'performance_learning_enabled': self.performance_learning_enabled
                },
                'performance_history_size': len(self.performance_history),
                'recent_performance': np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else None,
                'bounds': {
                    'min_oi_threshold': self.min_oi_threshold,
                    'max_oi_threshold': self.max_oi_threshold,
                    'min_price_threshold': self.min_price_threshold,
                    'max_price_threshold': self.max_price_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization statistics: {e}")
            return {}
    
    def reset_performance_history(self):
        """Reset performance history"""
        try:
            self.performance_history.clear()
            self.threshold_performance_map.clear()
            logger.info("Performance history reset")
            
        except Exception as e:
            logger.error(f"Error resetting performance history: {e}")
