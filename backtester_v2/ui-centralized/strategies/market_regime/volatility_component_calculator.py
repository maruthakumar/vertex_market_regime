"""
Comprehensive Volatility Component Calculator for Enhanced Market Regime Formation

This module implements comprehensive volatility component calculation using multiple
volatility sources with weighted combination for accurate regime classification.

Features:
1. ATR-based volatility calculation (40% weight)
2. OI volatility calculation (30% weight)
3. Price volatility calculation (30% weight)
4. Combined volatility scoring (0 to 1)
5. Volatility confidence assessment
6. Real-time volatility tracking
7. Performance optimization for production use

Author: The Augster
Date: 2025-01-16
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class VolatilityComponentResult:
    """Result structure for volatility component calculation"""
    volatility_component: float
    atr_volatility: float
    oi_volatility: float
    price_volatility: float
    confidence: float
    volatility_regime: str
    supporting_metrics: Dict[str, Any]
    calculation_timestamp: datetime

class VolatilityComponentCalculator:
    """
    Comprehensive Volatility Component Calculator
    
    Calculates unified volatility component for regime classification using
    multiple volatility sources with weighted combination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Volatility Component Calculator"""
        self.config = config or {}
        
        # Volatility weights configuration
        self.atr_weight = float(self.config.get('atr_weight', 0.4))
        self.oi_volatility_weight = float(self.config.get('oi_volatility_weight', 0.3))
        self.price_volatility_weight = float(self.config.get('price_volatility_weight', 0.3))
        
        # Validate weights sum to 1.0
        total_weight = self.atr_weight + self.oi_volatility_weight + self.price_volatility_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Volatility weights sum to {total_weight}, normalizing to 1.0")
            self.atr_weight /= total_weight
            self.oi_volatility_weight /= total_weight
            self.price_volatility_weight /= total_weight
        
        # ATR configuration
        self.atr_period = int(self.config.get('atr_period', 14))
        self.atr_ema_period = int(self.config.get('atr_ema_period', 20))
        
        # OI volatility configuration
        self.oi_volatility_window = int(self.config.get('oi_volatility_window', 20))
        self.oi_change_threshold = float(self.config.get('oi_change_threshold', 0.05))
        
        # Price volatility configuration
        self.price_volatility_window = int(self.config.get('price_volatility_window', 20))
        self.price_change_threshold = float(self.config.get('price_change_threshold', 0.02))
        
        # CALIBRATED: Volatility regime thresholds for Indian market
        self.low_volatility_threshold = float(self.config.get('low_volatility_threshold', 0.25))  # Reduced from 0.3 to 0.25
        self.high_volatility_threshold = float(self.config.get('high_volatility_threshold', 0.65))  # Reduced from 0.7 to 0.65
        
        # Historical data storage for calculations
        self.max_history_size = int(self.config.get('max_history_size', 100))
        self.atr_history = deque(maxlen=self.max_history_size)
        self.oi_change_history = deque(maxlen=self.max_history_size)
        self.price_change_history = deque(maxlen=self.max_history_size)
        
        # Performance optimization
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_duration_seconds = int(self.config.get('cache_duration_seconds', 60))
        self.volatility_cache = {}
        
        logger.info(f"Volatility Component Calculator initialized with weights: ATR={self.atr_weight:.2f}, OI={self.oi_volatility_weight:.2f}, Price={self.price_volatility_weight:.2f}")
    
    def calculate_volatility_component(self, market_data: Dict[str, Any]) -> VolatilityComponentResult:
        """
        Calculate comprehensive volatility component
        
        Args:
            market_data: Market data including price, OI, volume, and historical data
            
        Returns:
            VolatilityComponentResult with complete volatility analysis
        """
        try:
            # Check cache if enabled
            cache_key = self._generate_cache_key(market_data)
            if self.enable_caching and self._is_cache_valid(cache_key):
                return self.volatility_cache[cache_key]['result']
            
            # Calculate individual volatility components
            atr_volatility = self._calculate_atr_volatility(market_data)
            oi_volatility = self._calculate_oi_volatility(market_data)
            price_volatility = self._calculate_price_volatility(market_data)
            
            # Calculate combined volatility score
            combined_volatility = (
                atr_volatility * self.atr_weight +
                oi_volatility * self.oi_volatility_weight +
                price_volatility * self.price_volatility_weight
            )
            
            # Ensure volatility component is in [0, 1] range
            volatility_component = np.clip(combined_volatility, 0.0, 1.0)
            
            # Determine volatility regime
            volatility_regime = self._classify_volatility_regime(volatility_component)
            
            # Calculate confidence based on data quality and consistency
            confidence = self._calculate_volatility_confidence(
                market_data, atr_volatility, oi_volatility, price_volatility
            )
            
            # Prepare supporting metrics
            supporting_metrics = {
                'atr_raw': market_data.get('atr_data', {}).get('current_atr', 0),
                'atr_ema': market_data.get('atr_data', {}).get('atr_ema_20', 0),
                'oi_std': np.std(list(self.oi_change_history)) if len(self.oi_change_history) > 5 else 0,
                'price_std': np.std(list(self.price_change_history)) if len(self.price_change_history) > 5 else 0,
                'data_points_atr': len(self.atr_history),
                'data_points_oi': len(self.oi_change_history),
                'data_points_price': len(self.price_change_history),
                'weights_used': {
                    'atr_weight': self.atr_weight,
                    'oi_weight': self.oi_volatility_weight,
                    'price_weight': self.price_volatility_weight
                }
            }
            
            # Create result
            result = VolatilityComponentResult(
                volatility_component=volatility_component,
                atr_volatility=atr_volatility,
                oi_volatility=oi_volatility,
                price_volatility=price_volatility,
                confidence=confidence,
                volatility_regime=volatility_regime,
                supporting_metrics=supporting_metrics,
                calculation_timestamp=datetime.now()
            )
            
            # Update cache if enabled
            if self.enable_caching:
                self._update_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating volatility component: {e}")
            return self._get_default_volatility_result()
    
    def _calculate_atr_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate ATR-based volatility component"""
        try:
            atr_data = market_data.get('atr_data', {})
            current_atr = atr_data.get('current_atr', 0)
            atr_ema_20 = atr_data.get('atr_ema_20', current_atr)
            
            # Store ATR in history
            if current_atr > 0:
                self.atr_history.append(current_atr)
            
            # Calculate ATR volatility ratio
            if atr_ema_20 > 0:
                atr_ratio = current_atr / atr_ema_20
                
                # Normalize to [0, 1] range
                # ATR ratio of 0.5 = low volatility (0.0)
                # ATR ratio of 1.0 = normal volatility (0.5)
                # ATR ratio of 2.0+ = high volatility (1.0)
                if atr_ratio <= 0.5:
                    atr_volatility = 0.0
                elif atr_ratio <= 1.0:
                    atr_volatility = (atr_ratio - 0.5) * 1.0  # 0.5 to 1.0 -> 0.0 to 0.5
                else:
                    atr_volatility = 0.5 + min((atr_ratio - 1.0) * 0.5, 0.5)  # 1.0+ -> 0.5 to 1.0
                
                return np.clip(atr_volatility, 0.0, 1.0)
            else:
                return 0.5  # Default to medium volatility if no ATR data
                
        except Exception as e:
            logger.error(f"Error calculating ATR volatility: {e}")
            return 0.5
    
    def _calculate_oi_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate OI-based volatility component"""
        try:
            oi_changes = market_data.get('oi_changes', [])
            
            # Add current OI change to history
            if oi_changes:
                current_oi_change = oi_changes[-1] if isinstance(oi_changes, list) else oi_changes
                self.oi_change_history.append(current_oi_change)
            
            # Need sufficient data points for volatility calculation
            if len(self.oi_change_history) < 5:
                return 0.5  # Default to medium volatility
            
            # Calculate OI volatility using standard deviation
            oi_std = np.std(list(self.oi_change_history))
            
            # Normalize OI volatility to [0, 1] range
            # Typical OI changes are in the range of 0-20%
            # 0% std = low volatility (0.0)
            # 5% std = medium volatility (0.5)
            # 10%+ std = high volatility (1.0)
            if oi_std <= 0.02:  # 2% or less
                oi_volatility = oi_std * 12.5  # 0.02 * 12.5 = 0.25
            elif oi_std <= 0.05:  # 2% to 5%
                oi_volatility = 0.25 + (oi_std - 0.02) * 8.33  # Scale to 0.25-0.5
            else:  # 5%+
                oi_volatility = 0.5 + min((oi_std - 0.05) * 10, 0.5)  # Scale to 0.5-1.0
            
            return np.clip(oi_volatility, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating OI volatility: {e}")
            return 0.5
    
    def _calculate_price_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate price-based volatility component"""
        try:
            price_changes = market_data.get('price_changes', [])
            
            # Add current price change to history
            if price_changes:
                current_price_change = price_changes[-1] if isinstance(price_changes, list) else price_changes
                self.price_change_history.append(current_price_change)
            
            # Need sufficient data points for volatility calculation
            if len(self.price_change_history) < 5:
                return 0.5  # Default to medium volatility
            
            # Calculate price volatility using standard deviation
            price_std = np.std(list(self.price_change_history))
            
            # Normalize price volatility to [0, 1] range
            # Typical price changes are in the range of 0-10%
            # 0% std = low volatility (0.0)
            # 2% std = medium volatility (0.5)
            # 4%+ std = high volatility (1.0)
            if price_std <= 0.01:  # 1% or less
                price_volatility = price_std * 25  # 0.01 * 25 = 0.25
            elif price_std <= 0.02:  # 1% to 2%
                price_volatility = 0.25 + (price_std - 0.01) * 25  # Scale to 0.25-0.5
            else:  # 2%+
                price_volatility = 0.5 + min((price_std - 0.02) * 25, 0.5)  # Scale to 0.5-1.0
            
            return np.clip(price_volatility, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating price volatility: {e}")
            return 0.5
    
    def _classify_volatility_regime(self, volatility_component: float) -> str:
        """Classify volatility regime based on volatility component"""
        try:
            if volatility_component <= self.low_volatility_threshold:
                return "LOW_VOLATILITY"
            elif volatility_component >= self.high_volatility_threshold:
                return "HIGH_VOLATILITY"
            else:
                return "MEDIUM_VOLATILITY"
                
        except Exception as e:
            logger.error(f"Error classifying volatility regime: {e}")
            return "MEDIUM_VOLATILITY"
    
    def _calculate_volatility_confidence(self, market_data: Dict[str, Any], 
                                       atr_vol: float, oi_vol: float, 
                                       price_vol: float) -> float:
        """Calculate confidence in volatility calculation"""
        try:
            # Data quality factors
            data_quality_factors = []
            
            # ATR data quality
            atr_data = market_data.get('atr_data', {})
            if atr_data.get('current_atr', 0) > 0 and atr_data.get('atr_ema_20', 0) > 0:
                data_quality_factors.append(1.0)
            else:
                data_quality_factors.append(0.3)
            
            # OI data quality
            if len(self.oi_change_history) >= 10:
                data_quality_factors.append(1.0)
            elif len(self.oi_change_history) >= 5:
                data_quality_factors.append(0.7)
            else:
                data_quality_factors.append(0.3)
            
            # Price data quality
            if len(self.price_change_history) >= 10:
                data_quality_factors.append(1.0)
            elif len(self.price_change_history) >= 5:
                data_quality_factors.append(0.7)
            else:
                data_quality_factors.append(0.3)
            
            # Consistency check (volatility components should be somewhat aligned)
            volatility_values = [atr_vol, oi_vol, price_vol]
            volatility_std = np.std(volatility_values)
            
            # Lower standard deviation = higher consistency = higher confidence
            consistency_factor = max(0.3, 1.0 - volatility_std)
            
            # Combined confidence
            data_quality_confidence = np.mean(data_quality_factors)
            combined_confidence = (data_quality_confidence * 0.7 + consistency_factor * 0.3)
            
            return np.clip(combined_confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating volatility confidence: {e}")
            return 0.5
    
    def _generate_cache_key(self, market_data: Dict[str, Any]) -> str:
        """Generate cache key for volatility calculation"""
        try:
            timestamp = market_data.get('timestamp', datetime.now())
            underlying_price = market_data.get('underlying_price', 0)
            
            # Create cache key based on timestamp and price (rounded for caching efficiency)
            cache_key = f"{timestamp.strftime('%Y%m%d_%H%M')}_{underlying_price:.2f}"
            return cache_key
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        try:
            if cache_key not in self.volatility_cache:
                return False
            
            cache_entry = self.volatility_cache[cache_key]
            cache_time = cache_entry['timestamp']
            
            # Check if cache is still valid
            time_diff = (datetime.now() - cache_time).total_seconds()
            return time_diff < self.cache_duration_seconds
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def _update_cache(self, cache_key: str, result: VolatilityComponentResult):
        """Update volatility cache"""
        try:
            self.volatility_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            # Limit cache size (simple cleanup)
            if len(self.volatility_cache) > 100:
                # Remove oldest entries
                oldest_keys = sorted(self.volatility_cache.keys())[:50]
                for key in oldest_keys:
                    del self.volatility_cache[key]
                    
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
    
    def _get_default_volatility_result(self) -> VolatilityComponentResult:
        """Get default volatility result for error cases"""
        return VolatilityComponentResult(
            volatility_component=0.5,
            atr_volatility=0.5,
            oi_volatility=0.5,
            price_volatility=0.5,
            confidence=0.3,
            volatility_regime="MEDIUM_VOLATILITY",
            supporting_metrics={},
            calculation_timestamp=datetime.now()
        )
    
    def get_volatility_statistics(self) -> Dict[str, Any]:
        """Get volatility calculation statistics"""
        try:
            return {
                'atr_history_size': len(self.atr_history),
                'oi_history_size': len(self.oi_change_history),
                'price_history_size': len(self.price_change_history),
                'cache_size': len(self.volatility_cache),
                'weights': {
                    'atr_weight': self.atr_weight,
                    'oi_weight': self.oi_volatility_weight,
                    'price_weight': self.price_volatility_weight
                },
                'thresholds': {
                    'low_volatility': self.low_volatility_threshold,
                    'high_volatility': self.high_volatility_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting volatility statistics: {e}")
            return {}
    
    def reset_history(self):
        """Reset volatility history and cache"""
        try:
            self.atr_history.clear()
            self.oi_change_history.clear()
            self.price_change_history.clear()
            self.volatility_cache.clear()
            logger.info("Volatility history and cache reset")
            
        except Exception as e:
            logger.error(f"Error resetting volatility history: {e}")
