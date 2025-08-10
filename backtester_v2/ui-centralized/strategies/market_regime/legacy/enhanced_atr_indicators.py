"""
Enhanced ATR Indicators for Market Regime Formation

This module implements comprehensive ATR (Average True Range) indicators with volatility bands
and breakout detection to bridge the 70% feature gap identified in the market regime system.

Features:
1. Multi-period ATR calculation (14, 21, 50 periods)
2. ATR volatility bands (Bollinger-style)
3. ATR breakout detection
4. Volatility expansion/contraction analysis
5. ATR regime classification
6. Confidence scoring based on data quality
7. Real-time ATR tracking
8. Integration with market regime formation

Author: The Augster
Date: 2025-01-16
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ATRRegime(Enum):
    """ATR regime classifications"""
    EXTREMELY_LOW_VOLATILITY = "Extremely_Low_Volatility"    # Very low ATR
    LOW_VOLATILITY = "Low_Volatility"                        # Low ATR
    NORMAL_VOLATILITY = "Normal_Volatility"                  # Normal ATR
    HIGH_VOLATILITY = "High_Volatility"                      # High ATR
    EXTREMELY_HIGH_VOLATILITY = "Extremely_High_Volatility"  # Very high ATR
    VOLATILITY_EXPANSION = "Volatility_Expansion"            # ATR increasing
    VOLATILITY_CONTRACTION = "Volatility_Contraction"        # ATR decreasing

class ATRBreakoutType(Enum):
    """ATR breakout types"""
    NO_BREAKOUT = "No_Breakout"
    UPSIDE_BREAKOUT = "Upside_Breakout"
    DOWNSIDE_BREAKOUT = "Downside_Breakout"
    VOLATILITY_BREAKOUT = "Volatility_Breakout"

@dataclass
class ATRIndicatorResult:
    """Result structure for ATR indicator analysis"""
    atr_14: float
    atr_21: float
    atr_50: float
    atr_percentile: float
    atr_regime: ATRRegime
    volatility_bands: Dict[str, float]
    breakout_signals: Dict[str, Any]
    confidence: float
    regime_strength: float
    supporting_metrics: Dict[str, Any]

class EnhancedATRIndicators:
    """
    Enhanced ATR Indicators for Market Regime Formation
    
    Implements comprehensive ATR analysis with volatility bands, breakout detection,
    and regime classification for enhanced market regime detection accuracy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Enhanced ATR Indicators"""
        self.config = config or {}
        
        # ATR calculation periods
        self.atr_periods = {
            'short': int(self.config.get('atr_short_period', 14)),
            'medium': int(self.config.get('atr_medium_period', 21)),
            'long': int(self.config.get('atr_long_period', 50))
        }
        
        # Historical price data storage
        self.price_history = deque(maxlen=252)  # 1 year of data
        self.atr_history = {
            'atr_14': deque(maxlen=252),
            'atr_21': deque(maxlen=252),
            'atr_50': deque(maxlen=252)
        }
        
        # CALIBRATED: ATR regime thresholds for Indian market (percentile-based)
        self.atr_regime_thresholds = {
            'extremely_low': 10,     # 0-10th percentile
            'low': 25,               # 10-25th percentile
            'normal_low': 40,        # 25-40th percentile
            'normal_high': 60,       # 40-60th percentile
            'high': 75,              # 60-75th percentile
            'extremely_high': 90     # 75-90th percentile
        }
        
        # Volatility band parameters
        self.band_periods = int(self.config.get('band_periods', 20))
        self.band_std_multiplier = float(self.config.get('band_std_multiplier', 2.0))
        
        # Breakout detection parameters
        self.breakout_threshold = float(self.config.get('breakout_threshold', 1.5))  # ATR multiplier
        self.min_breakout_volume = int(self.config.get('min_breakout_volume', 1000))
        
        # Minimum data requirements
        self.min_data_points = int(self.config.get('min_data_points', 50))
        
        logger.info("Enhanced ATR Indicators initialized")
    
    def analyze_atr_indicators(self, market_data: Dict[str, Any]) -> ATRIndicatorResult:
        """
        Main analysis function for ATR indicators
        
        Args:
            market_data: Market data including price history and current price data
            
        Returns:
            ATRIndicatorResult with complete ATR analysis
        """
        try:
            # Extract and prepare price data
            price_data = self._prepare_price_data(market_data)
            
            if len(price_data) < self.atr_periods['short']:
                logger.warning("Insufficient price data for ATR calculation")
                return self._get_default_result()
            
            # Calculate multi-period ATR values
            atr_values = self._calculate_multi_period_atr(price_data)
            
            # Update ATR history
            self._update_atr_history(atr_values)
            
            # Calculate ATR percentile
            atr_percentile = self._calculate_atr_percentile(atr_values['atr_14'])
            
            # Classify ATR regime
            atr_regime = self._classify_atr_regime(atr_percentile, atr_values)
            
            # Calculate volatility bands
            volatility_bands = self._calculate_volatility_bands(price_data, atr_values)
            
            # Detect breakout signals
            breakout_signals = self._detect_breakout_signals(price_data, atr_values, volatility_bands)
            
            # Calculate regime strength
            regime_strength = self._calculate_regime_strength(atr_percentile, atr_regime)
            
            # Calculate confidence
            confidence = self._calculate_confidence(price_data, atr_values)
            
            # Prepare supporting metrics
            supporting_metrics = self._prepare_supporting_metrics(
                price_data, atr_values, market_data
            )
            
            return ATRIndicatorResult(
                atr_14=atr_values['atr_14'],
                atr_21=atr_values['atr_21'],
                atr_50=atr_values['atr_50'],
                atr_percentile=atr_percentile,
                atr_regime=atr_regime,
                volatility_bands=volatility_bands,
                breakout_signals=breakout_signals,
                confidence=confidence,
                regime_strength=regime_strength,
                supporting_metrics=supporting_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in ATR indicator analysis: {e}")
            return self._get_default_result()
    
    def _prepare_price_data(self, market_data: Dict[str, Any]) -> List[Dict[str, float]]:
        """Prepare and validate price data for ATR calculation"""
        try:
            # Try to get price history from market data
            price_history = market_data.get('price_history', [])
            
            # Add current price data if available
            current_price_data = {
                'high': market_data.get('high', market_data.get('underlying_price', 0)),
                'low': market_data.get('low', market_data.get('underlying_price', 0)),
                'close': market_data.get('close', market_data.get('underlying_price', 0)),
                'timestamp': market_data.get('timestamp', datetime.now())
            }
            
            # Combine historical and current data
            combined_data = list(price_history)
            if current_price_data['close'] > 0:
                combined_data.append(current_price_data)
            
            # Update internal price history
            for price_point in combined_data[-10:]:  # Keep last 10 points
                self.price_history.append(price_point)
            
            # Return recent data for calculation
            return list(self.price_history)
            
        except Exception as e:
            logger.error(f"Error preparing price data: {e}")
            return []
    
    def _calculate_multi_period_atr(self, price_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate ATR for multiple periods"""
        try:
            if len(price_data) < 2:
                return {'atr_14': 0.0, 'atr_21': 0.0, 'atr_50': 0.0}
            
            # Calculate True Range for each period
            true_ranges = []
            for i in range(1, len(price_data)):
                current = price_data[i]
                previous = price_data[i-1]
                
                high_low = current['high'] - current['low']
                high_close_prev = abs(current['high'] - previous['close'])
                low_close_prev = abs(current['low'] - previous['close'])
                
                true_range = max(high_low, high_close_prev, low_close_prev)
                true_ranges.append(true_range)
            
            if not true_ranges:
                return {'atr_14': 0.0, 'atr_21': 0.0, 'atr_50': 0.0}
            
            # Calculate ATR for different periods
            atr_values = {}
            for period_name, period_length in self.atr_periods.items():
                if len(true_ranges) >= period_length:
                    # Use exponential moving average for ATR
                    atr = self._calculate_ema(true_ranges, period_length)
                    atr_values[f'atr_{period_length}'] = atr
                else:
                    # Use simple average if insufficient data
                    atr_values[f'atr_{period_length}'] = np.mean(true_ranges)
            
            return atr_values
            
        except Exception as e:
            logger.error(f"Error calculating multi-period ATR: {e}")
            return {'atr_14': 0.0, 'atr_21': 0.0, 'atr_50': 0.0}
    
    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(data) < period:
                return np.mean(data) if data else 0.0
            
            # Use last 'period' data points
            recent_data = data[-period:]
            
            # Calculate EMA
            multiplier = 2.0 / (period + 1)
            ema = recent_data[0]
            
            for value in recent_data[1:]:
                ema = (value * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return 0.0

    def _update_atr_history(self, atr_values: Dict[str, float]):
        """Update ATR history for percentile calculations"""
        try:
            for atr_type, value in atr_values.items():
                if atr_type in self.atr_history:
                    self.atr_history[atr_type].append({
                        'value': value,
                        'timestamp': datetime.now()
                    })

        except Exception as e:
            logger.error(f"Error updating ATR history: {e}")

    def _calculate_atr_percentile(self, current_atr: float) -> float:
        """Calculate ATR percentile based on historical data"""
        try:
            atr_14_history = self.atr_history['atr_14']

            if len(atr_14_history) < 30:
                return 50.0  # Neutral percentile for insufficient data

            # Extract ATR values from historical data
            historical_atrs = [data['value'] for data in atr_14_history]

            # Calculate percentile rank
            atr_array = np.array(historical_atrs)
            percentile = (np.sum(atr_array <= current_atr) / len(atr_array)) * 100

            return np.clip(percentile, 0.0, 100.0)

        except Exception as e:
            logger.error(f"Error calculating ATR percentile: {e}")
            return 50.0

    def _classify_atr_regime(self, atr_percentile: float, atr_values: Dict[str, float]) -> ATRRegime:
        """Classify ATR regime based on percentile and trend"""
        try:
            # Check for volatility expansion/contraction
            if len(self.atr_history['atr_14']) >= 5:
                recent_atrs = [data['value'] for data in list(self.atr_history['atr_14'])[-5:]]
                atr_trend = (recent_atrs[-1] - recent_atrs[0]) / recent_atrs[0] if recent_atrs[0] > 0 else 0

                if atr_trend > 0.1:  # 10% increase
                    return ATRRegime.VOLATILITY_EXPANSION
                elif atr_trend < -0.1:  # 10% decrease
                    return ATRRegime.VOLATILITY_CONTRACTION

            # Classify based on percentile
            if atr_percentile <= self.atr_regime_thresholds['extremely_low']:
                return ATRRegime.EXTREMELY_LOW_VOLATILITY
            elif atr_percentile <= self.atr_regime_thresholds['low']:
                return ATRRegime.LOW_VOLATILITY
            elif atr_percentile <= self.atr_regime_thresholds['normal_high']:
                return ATRRegime.NORMAL_VOLATILITY
            elif atr_percentile <= self.atr_regime_thresholds['high']:
                return ATRRegime.HIGH_VOLATILITY
            else:
                return ATRRegime.EXTREMELY_HIGH_VOLATILITY

        except Exception as e:
            logger.error(f"Error classifying ATR regime: {e}")
            return ATRRegime.NORMAL_VOLATILITY

    def _calculate_volatility_bands(self, price_data: List[Dict[str, float]],
                                  atr_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate volatility bands based on ATR"""
        try:
            if len(price_data) < self.band_periods:
                return {'upper_band': 0.0, 'lower_band': 0.0, 'middle_band': 0.0}

            # Get recent closing prices
            recent_closes = [data['close'] for data in price_data[-self.band_periods:]]

            # Calculate middle band (moving average)
            middle_band = np.mean(recent_closes)

            # Calculate bands using ATR
            atr_14 = atr_values.get('atr_14', 0)
            band_width = atr_14 * self.band_std_multiplier

            upper_band = middle_band + band_width
            lower_band = middle_band - band_width

            return {
                'upper_band': upper_band,
                'lower_band': lower_band,
                'middle_band': middle_band,
                'band_width': band_width
            }

        except Exception as e:
            logger.error(f"Error calculating volatility bands: {e}")
            return {'upper_band': 0.0, 'lower_band': 0.0, 'middle_band': 0.0}

    def _detect_breakout_signals(self, price_data: List[Dict[str, float]],
                               atr_values: Dict[str, float],
                               volatility_bands: Dict[str, float]) -> Dict[str, Any]:
        """Detect breakout signals based on ATR and volatility bands"""
        try:
            if len(price_data) < 2:
                return {'breakout_type': ATRBreakoutType.NO_BREAKOUT, 'strength': 0.0}

            current_price = price_data[-1]
            previous_price = price_data[-2]

            # Price movement relative to ATR
            price_change = current_price['close'] - previous_price['close']
            atr_14 = atr_values.get('atr_14', 0)

            if atr_14 == 0:
                return {'breakout_type': ATRBreakoutType.NO_BREAKOUT, 'strength': 0.0}

            # Calculate breakout strength
            breakout_strength = abs(price_change) / atr_14

            # Determine breakout type
            breakout_type = ATRBreakoutType.NO_BREAKOUT

            if breakout_strength > self.breakout_threshold:
                if price_change > 0:
                    breakout_type = ATRBreakoutType.UPSIDE_BREAKOUT
                else:
                    breakout_type = ATRBreakoutType.DOWNSIDE_BREAKOUT

            # Check for volatility breakout (price outside bands)
            upper_band = volatility_bands.get('upper_band', 0)
            lower_band = volatility_bands.get('lower_band', 0)

            if current_price['close'] > upper_band or current_price['close'] < lower_band:
                if breakout_type == ATRBreakoutType.NO_BREAKOUT:
                    breakout_type = ATRBreakoutType.VOLATILITY_BREAKOUT

            return {
                'breakout_type': breakout_type,
                'strength': breakout_strength,
                'price_change': price_change,
                'atr_multiple': breakout_strength,
                'band_breach': current_price['close'] > upper_band or current_price['close'] < lower_band
            }

        except Exception as e:
            logger.error(f"Error detecting breakout signals: {e}")
            return {'breakout_type': ATRBreakoutType.NO_BREAKOUT, 'strength': 0.0}

    def _calculate_regime_strength(self, atr_percentile: float, atr_regime: ATRRegime) -> float:
        """Calculate strength of the ATR regime classification"""
        try:
            # Calculate distance from regime boundaries
            if atr_regime in [ATRRegime.VOLATILITY_EXPANSION, ATRRegime.VOLATILITY_CONTRACTION]:
                # For trend-based regimes, use trend strength
                if len(self.atr_history['atr_14']) >= 5:
                    recent_atrs = [data['value'] for data in list(self.atr_history['atr_14'])[-5:]]
                    trend_strength = abs(recent_atrs[-1] - recent_atrs[0]) / recent_atrs[0] if recent_atrs[0] > 0 else 0
                    return min(trend_strength * 10, 1.0)  # Scale to [0, 1]
                else:
                    return 0.5

            # For percentile-based regimes
            if atr_regime == ATRRegime.EXTREMELY_LOW_VOLATILITY:
                strength = (self.atr_regime_thresholds['extremely_low'] - atr_percentile) / self.atr_regime_thresholds['extremely_low']
            elif atr_regime == ATRRegime.LOW_VOLATILITY:
                range_size = self.atr_regime_thresholds['low'] - self.atr_regime_thresholds['extremely_low']
                mid_point = self.atr_regime_thresholds['extremely_low'] + range_size / 2
                strength = 1.0 - abs(atr_percentile - mid_point) / (range_size / 2)
            elif atr_regime == ATRRegime.NORMAL_VOLATILITY:
                range_size = self.atr_regime_thresholds['normal_high'] - self.atr_regime_thresholds['low']
                mid_point = self.atr_regime_thresholds['low'] + range_size / 2
                strength = 1.0 - abs(atr_percentile - mid_point) / (range_size / 2)
            elif atr_regime == ATRRegime.HIGH_VOLATILITY:
                range_size = self.atr_regime_thresholds['high'] - self.atr_regime_thresholds['normal_high']
                mid_point = self.atr_regime_thresholds['normal_high'] + range_size / 2
                strength = 1.0 - abs(atr_percentile - mid_point) / (range_size / 2)
            else:  # EXTREMELY_HIGH_VOLATILITY
                strength = (atr_percentile - self.atr_regime_thresholds['high']) / (100 - self.atr_regime_thresholds['high'])

            return np.clip(strength, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating regime strength: {e}")
            return 0.5

    def _calculate_confidence(self, price_data: List[Dict[str, float]],
                            atr_values: Dict[str, float]) -> float:
        """Calculate confidence in ATR analysis"""
        try:
            # Data quality confidence (amount of price data)
            data_quality = min(len(price_data) / self.min_data_points, 1.0)

            # ATR consistency confidence (stability of recent ATR values)
            if len(self.atr_history['atr_14']) >= 5:
                recent_atrs = [data['value'] for data in list(self.atr_history['atr_14'])[-5:]]
                atr_std = np.std(recent_atrs)
                atr_mean = np.mean(recent_atrs)
                atr_consistency = max(0.1, 1.0 - (atr_std / atr_mean)) if atr_mean > 0 else 0.5
            else:
                atr_consistency = 0.5

            # Price data quality (completeness of OHLC data)
            complete_data_count = 0
            for data_point in price_data[-10:]:  # Check last 10 points
                if all(key in data_point and data_point[key] > 0 for key in ['high', 'low', 'close']):
                    complete_data_count += 1

            data_completeness = complete_data_count / min(len(price_data), 10)

            # Combined confidence
            combined_confidence = (
                data_quality * 0.4 +
                atr_consistency * 0.4 +
                data_completeness * 0.2
            )

            return np.clip(combined_confidence, 0.1, 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _prepare_supporting_metrics(self, price_data: List[Dict[str, float]],
                                  atr_values: Dict[str, float],
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare supporting metrics for the analysis"""
        try:
            metrics = {
                'price_data_points': len(price_data),
                'atr_14': atr_values.get('atr_14', 0),
                'atr_21': atr_values.get('atr_21', 0),
                'atr_50': atr_values.get('atr_50', 0),
                'analysis_timestamp': datetime.now(),
                'historical_atr_points': len(self.atr_history['atr_14'])
            }

            # Add price statistics
            if price_data:
                recent_closes = [data['close'] for data in price_data[-20:]]
                metrics.update({
                    'current_price': price_data[-1]['close'],
                    'price_volatility': np.std(recent_closes) if len(recent_closes) > 1 else 0,
                    'price_trend': (recent_closes[-1] - recent_closes[0]) / recent_closes[0] if len(recent_closes) > 1 and recent_closes[0] > 0 else 0
                })

            # Add ATR statistics if sufficient history
            if len(self.atr_history['atr_14']) >= 10:
                atr_values_hist = [data['value'] for data in self.atr_history['atr_14']]
                metrics.update({
                    'atr_mean': np.mean(atr_values_hist),
                    'atr_std': np.std(atr_values_hist),
                    'atr_min': np.min(atr_values_hist),
                    'atr_max': np.max(atr_values_hist)
                })

            return metrics

        except Exception as e:
            logger.error(f"Error preparing supporting metrics: {e}")
            return {'error': str(e)}

    def _get_default_result(self) -> ATRIndicatorResult:
        """Get default result for error cases"""
        return ATRIndicatorResult(
            atr_14=0.0,
            atr_21=0.0,
            atr_50=0.0,
            atr_percentile=50.0,
            atr_regime=ATRRegime.NORMAL_VOLATILITY,
            volatility_bands={'upper_band': 0.0, 'lower_band': 0.0, 'middle_band': 0.0},
            breakout_signals={'breakout_type': ATRBreakoutType.NO_BREAKOUT, 'strength': 0.0},
            confidence=0.3,
            regime_strength=0.5,
            supporting_metrics={'error': 'Insufficient data'}
        )

    def get_regime_component(self, market_data: Dict[str, Any]) -> float:
        """Get ATR regime component for market regime formation (0-1 scale)"""
        try:
            result = self.analyze_atr_indicators(market_data)

            # Convert ATR percentile to regime component
            regime_component = result.atr_percentile / 100.0

            # Apply confidence weighting
            weighted_component = regime_component * result.confidence + 0.5 * (1 - result.confidence)

            return np.clip(weighted_component, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error getting regime component: {e}")
            return 0.5

    def reset_history(self):
        """Reset ATR and price history"""
        try:
            self.price_history.clear()
            for atr_type in self.atr_history:
                self.atr_history[atr_type].clear()
            logger.info("ATR history reset")

        except Exception as e:
            logger.error(f"Error resetting ATR history: {e}")

    def get_current_statistics(self) -> Dict[str, Any]:
        """Get current ATR statistics"""
        try:
            stats = {
                'price_data_points': len(self.price_history),
                'atr_history_points': {
                    atr_type: len(history) for atr_type, history in self.atr_history.items()
                }
            }

            # Add latest ATR values if available
            for atr_type, history in self.atr_history.items():
                if history:
                    values = [data['value'] for data in history]
                    stats[f'{atr_type}_latest'] = values[-1]
                    stats[f'{atr_type}_mean'] = np.mean(values)
                    stats[f'{atr_type}_std'] = np.std(values)

            return stats

        except Exception as e:
            logger.error(f"Error getting current statistics: {e}")
            return {'error': str(e)}
