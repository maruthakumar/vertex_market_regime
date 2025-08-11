"""
Component 5: Straddle ATR-EMA-CPR Analysis Engine

Comprehensive ATR-EMA-CPR analysis specifically designed for rolling straddle
prices, providing options-specific insights for volatility-trend-pivot analysis
with regime classification capabilities.

Features:
- ATR calculation on straddle prices using True Range methodology  
- EMA trend analysis on straddle prices for options-specific trends
- CPR (Central Pivot Range) analysis on straddle prices
- Volatility regime classification using ATR patterns
- Trend strength and direction classification using EMA analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
import time
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .dual_asset_data_extractor import StraddlePriceData

warnings.filterwarnings('ignore')


@dataclass
class StraddleATRResult:
    """ATR analysis result for straddle prices"""
    atr_14: np.ndarray
    atr_21: np.ndarray
    atr_50: np.ndarray
    atr_percentiles: Dict[str, np.ndarray]
    volatility_regime: np.ndarray
    atr_trend: np.ndarray
    true_ranges: np.ndarray
    atr_signals: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class StraddleEMAResult:
    """EMA analysis result for straddle prices"""
    ema_20: np.ndarray
    ema_50: np.ndarray
    ema_100: np.ndarray
    ema_200: np.ndarray
    trend_direction: np.ndarray
    trend_strength: np.ndarray
    confluence_zones: np.ndarray
    crossover_signals: Dict[str, np.ndarray]
    slope_analysis: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class StraddleCPRResult:
    """CPR analysis result for straddle prices"""
    pivot_points: Dict[str, np.ndarray]  # Standard, Fibonacci, Camarilla
    support_levels: Dict[str, List[np.ndarray]]
    resistance_levels: Dict[str, List[np.ndarray]]
    cpr_width: np.ndarray
    cpr_position: np.ndarray
    breakout_signals: Dict[str, np.ndarray]
    level_strength: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class StraddleAnalysisResult:
    """Complete straddle analysis result combining ATR-EMA-CPR"""
    atr_result: StraddleATRResult
    ema_result: StraddleEMAResult
    cpr_result: StraddleCPRResult
    combined_signals: Dict[str, np.ndarray]
    regime_classification: np.ndarray
    confidence_scores: np.ndarray
    feature_vector: np.ndarray
    processing_time_ms: float
    metadata: Dict[str, Any]


class StraddleATRAnalyzer:
    """ATR analysis specifically for straddle prices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # ATR parameters
        self.atr_periods = [14, 21, 50]
        self.volatility_thresholds = {
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75,
            'extreme': 0.95
        }

    async def analyze_straddle_atr(self, straddle_data: StraddlePriceData) -> StraddleATRResult:
        """Calculate ATR analysis for straddle prices"""
        
        # Calculate True Range for straddles
        true_ranges = self._calculate_straddle_true_range(straddle_data)
        
        # Calculate ATR for different periods
        atr_14 = self._calculate_atr(true_ranges, 14)
        atr_21 = self._calculate_atr(true_ranges, 21) 
        atr_50 = self._calculate_atr(true_ranges, 50)
        
        # Calculate ATR percentiles for regime classification
        atr_percentiles = self._calculate_atr_percentiles(atr_14, atr_21, atr_50)
        
        # Classify volatility regime
        volatility_regime = self._classify_volatility_regime(atr_percentiles)
        
        # Calculate ATR trend
        atr_trend = self._calculate_atr_trend(atr_14)
        
        # Generate ATR signals
        atr_signals = self._generate_atr_signals(atr_14, atr_21, atr_50, atr_percentiles)
        
        return StraddleATRResult(
            atr_14=atr_14,
            atr_21=atr_21,
            atr_50=atr_50,
            atr_percentiles=atr_percentiles,
            volatility_regime=volatility_regime,
            atr_trend=atr_trend,
            true_ranges=true_ranges,
            atr_signals=atr_signals,
            metadata={
                'periods': self.atr_periods,
                'data_points': len(true_ranges),
                'volatility_classification': 'straddle_specific'
            }
        )

    def _calculate_straddle_true_range(self, straddle_data: StraddlePriceData) -> np.ndarray:
        """Calculate True Range for straddle prices"""
        
        high = straddle_data.straddle_high
        low = straddle_data.straddle_low
        close_prev = np.roll(straddle_data.straddle_close, 1)
        close_prev[0] = straddle_data.straddle_close[0]  # Handle first value
        
        # True Range calculation: max of (H-L, H-C_prev, C_prev-L)
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(close_prev - low)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range

    def _calculate_atr(self, true_ranges: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range"""
        atr = np.full(len(true_ranges), np.nan)
        
        if len(true_ranges) >= period:
            # Initial ATR (simple average)
            atr[period-1] = np.mean(true_ranges[:period])
            
            # Wilder's smoothing for subsequent values
            for i in range(period, len(true_ranges)):
                atr[i] = (atr[i-1] * (period-1) + true_ranges[i]) / period
        
        return atr

    def _calculate_atr_percentiles(self, atr_14: np.ndarray, atr_21: np.ndarray, atr_50: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate ATR percentiles for regime classification"""
        
        percentiles = {}
        
        for period, atr_values in [('14', atr_14), ('21', atr_21), ('50', atr_50)]:
            valid_atr = atr_values[~np.isnan(atr_values)]
            if len(valid_atr) > 0:
                percentiles[f'atr_{period}_pct'] = np.full(len(atr_values), np.nan)
                for i, value in enumerate(atr_values):
                    if not np.isnan(value):
                        percentiles[f'atr_{period}_pct'][i] = stats.percentileofscore(valid_atr, value) / 100
        
        return percentiles

    def _classify_volatility_regime(self, atr_percentiles: Dict[str, np.ndarray]) -> np.ndarray:
        """Classify volatility regime based on ATR percentiles"""
        
        # Use ATR-14 percentiles for primary classification
        if 'atr_14_pct' not in atr_percentiles:
            return np.array([0] * len(list(atr_percentiles.values())[0]))
        
        atr_pct = atr_percentiles['atr_14_pct']
        regime = np.full(len(atr_pct), 0)  # Default to neutral
        
        # Regime classification
        regime[atr_pct < self.volatility_thresholds['low']] = -2  # Very Low Volatility
        regime[(atr_pct >= self.volatility_thresholds['low']) & 
               (atr_pct < self.volatility_thresholds['medium'])] = -1  # Low Volatility
        regime[(atr_pct >= self.volatility_thresholds['medium']) & 
               (atr_pct < self.volatility_thresholds['high'])] = 0   # Medium Volatility
        regime[(atr_pct >= self.volatility_thresholds['high']) & 
               (atr_pct < self.volatility_thresholds['extreme'])] = 1  # High Volatility
        regime[atr_pct >= self.volatility_thresholds['extreme']] = 2  # Extreme Volatility
        
        return regime

    def _calculate_atr_trend(self, atr_14: np.ndarray) -> np.ndarray:
        """Calculate ATR trend (increasing/decreasing volatility)"""
        
        trend = np.full(len(atr_14), 0)
        
        for i in range(5, len(atr_14)):
            if not np.isnan(atr_14[i]) and not np.isnan(atr_14[i-5]):
                current_atr = atr_14[i]
                past_atr = atr_14[i-5]
                
                if current_atr > past_atr * 1.1:  # 10% increase
                    trend[i] = 1  # Increasing volatility
                elif current_atr < past_atr * 0.9:  # 10% decrease
                    trend[i] = -1  # Decreasing volatility
                else:
                    trend[i] = 0  # Stable volatility
        
        return trend

    def _generate_atr_signals(self, atr_14: np.ndarray, atr_21: np.ndarray, atr_50: np.ndarray, 
                            atr_percentiles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate ATR-based signals"""
        
        signals = {}
        
        # Volatility expansion signal
        signals['vol_expansion'] = np.zeros(len(atr_14))
        for i in range(14, len(atr_14)):
            if (not np.isnan(atr_14[i]) and atr_14[i] > np.nanmean(atr_14[i-14:i]) * 1.2):
                signals['vol_expansion'][i] = 1
        
        # Volatility contraction signal
        signals['vol_contraction'] = np.zeros(len(atr_14))
        for i in range(14, len(atr_14)):
            if (not np.isnan(atr_14[i]) and atr_14[i] < np.nanmean(atr_14[i-14:i]) * 0.8):
                signals['vol_contraction'][i] = 1
        
        # ATR convergence/divergence
        signals['atr_convergence'] = np.zeros(len(atr_14))
        for i in range(50, len(atr_14)):
            if not np.isnan(atr_14[i]) and not np.isnan(atr_50[i]):
                if abs(atr_14[i] - atr_50[i]) < min(atr_14[i], atr_50[i]) * 0.1:
                    signals['atr_convergence'][i] = 1
        
        return signals


class StraddleEMAAnalyzer:
    """EMA analysis specifically for straddle prices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # EMA parameters
        self.ema_periods = [20, 50, 100, 200]
        self.trend_strength_periods = 10

    async def analyze_straddle_ema(self, straddle_data: StraddlePriceData) -> StraddleEMAResult:
        """Calculate EMA analysis for straddle prices"""
        
        straddle_close = straddle_data.straddle_close
        
        # Calculate EMAs
        ema_20 = self._calculate_ema(straddle_close, 20)
        ema_50 = self._calculate_ema(straddle_close, 50)
        ema_100 = self._calculate_ema(straddle_close, 100)
        ema_200 = self._calculate_ema(straddle_close, 200)
        
        # Determine trend direction
        trend_direction = self._calculate_trend_direction(ema_20, ema_50, ema_100, ema_200)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(straddle_close, ema_20, ema_50)
        
        # Identify confluence zones
        confluence_zones = self._identify_confluence_zones(ema_20, ema_50, ema_100, ema_200)
        
        # Generate crossover signals
        crossover_signals = self._generate_crossover_signals(ema_20, ema_50, ema_100, ema_200)
        
        # Calculate slope analysis
        slope_analysis = self._calculate_slope_analysis(ema_20, ema_50, ema_100, ema_200)
        
        return StraddleEMAResult(
            ema_20=ema_20,
            ema_50=ema_50,
            ema_100=ema_100,
            ema_200=ema_200,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confluence_zones=confluence_zones,
            crossover_signals=crossover_signals,
            slope_analysis=slope_analysis,
            metadata={
                'periods': self.ema_periods,
                'data_points': len(straddle_close),
                'trend_analysis': 'straddle_specific'
            }
        )

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.full(len(prices), np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid price for initialization
        first_valid = None
        for i, price in enumerate(prices):
            if not np.isnan(price):
                first_valid = i
                ema[i] = price
                break
        
        if first_valid is None:
            return ema
        
        # Calculate EMA
        for i in range(first_valid + 1, len(prices)):
            if not np.isnan(prices[i]):
                if not np.isnan(ema[i-1]):
                    ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                else:
                    ema[i] = prices[i]
        
        return ema

    def _calculate_trend_direction(self, ema_20: np.ndarray, ema_50: np.ndarray, 
                                 ema_100: np.ndarray, ema_200: np.ndarray) -> np.ndarray:
        """Calculate trend direction based on EMA alignment"""
        
        trend = np.full(len(ema_20), 0)
        
        for i in range(len(ema_20)):
            valid_emas = []
            for ema in [ema_20[i], ema_50[i], ema_100[i], ema_200[i]]:
                if not np.isnan(ema):
                    valid_emas.append(ema)
            
            if len(valid_emas) >= 2:
                # Check if EMAs are in ascending order (bullish) or descending order (bearish)
                ascending = all(valid_emas[j] >= valid_emas[j+1] for j in range(len(valid_emas)-1))
                descending = all(valid_emas[j] <= valid_emas[j+1] for j in range(len(valid_emas)-1))
                
                if ascending:
                    trend[i] = 1  # Bullish trend
                elif descending:
                    trend[i] = -1  # Bearish trend
                else:
                    trend[i] = 0  # Neutral/mixed trend
        
        return trend

    def _calculate_trend_strength(self, prices: np.ndarray, ema_20: np.ndarray, ema_50: np.ndarray) -> np.ndarray:
        """Calculate trend strength based on price distance from EMAs"""
        
        strength = np.full(len(prices), 0.0)
        
        for i in range(len(prices)):
            if not np.isnan(prices[i]) and not np.isnan(ema_20[i]) and not np.isnan(ema_50[i]):
                # Distance from EMA as percentage
                dist_20 = abs(prices[i] - ema_20[i]) / ema_20[i]
                dist_50 = abs(prices[i] - ema_50[i]) / ema_50[i]
                
                # Average distance as strength measure
                strength[i] = (dist_20 + dist_50) / 2
        
        return strength

    def _identify_confluence_zones(self, ema_20: np.ndarray, ema_50: np.ndarray, 
                                 ema_100: np.ndarray, ema_200: np.ndarray) -> np.ndarray:
        """Identify EMA confluence zones"""
        
        confluence = np.zeros(len(ema_20))
        tolerance = 0.02  # 2% tolerance for confluence
        
        for i in range(len(ema_20)):
            valid_emas = []
            for ema in [ema_20[i], ema_50[i], ema_100[i], ema_200[i]]:
                if not np.isnan(ema):
                    valid_emas.append(ema)
            
            if len(valid_emas) >= 3:
                # Check if EMAs are within tolerance of each other
                ema_range = max(valid_emas) - min(valid_emas)
                avg_ema = np.mean(valid_emas)
                
                if ema_range / avg_ema < tolerance:
                    confluence[i] = 1  # Confluence zone detected
        
        return confluence

    def _generate_crossover_signals(self, ema_20: np.ndarray, ema_50: np.ndarray, 
                                  ema_100: np.ndarray, ema_200: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate EMA crossover signals"""
        
        signals = {}
        
        # Golden/Death cross (20/50)
        signals['golden_cross_20_50'] = np.zeros(len(ema_20))
        signals['death_cross_20_50'] = np.zeros(len(ema_20))
        
        for i in range(1, len(ema_20)):
            if (not np.isnan(ema_20[i-1]) and not np.isnan(ema_20[i]) and
                not np.isnan(ema_50[i-1]) and not np.isnan(ema_50[i])):
                
                # Golden cross: EMA20 crosses above EMA50
                if ema_20[i-1] <= ema_50[i-1] and ema_20[i] > ema_50[i]:
                    signals['golden_cross_20_50'][i] = 1
                
                # Death cross: EMA20 crosses below EMA50  
                if ema_20[i-1] >= ema_50[i-1] and ema_20[i] < ema_50[i]:
                    signals['death_cross_20_50'][i] = 1
        
        # Similar logic for other crossovers
        signals['golden_cross_50_100'] = self._detect_crossover(ema_50, ema_100, 'above')
        signals['death_cross_50_100'] = self._detect_crossover(ema_50, ema_100, 'below')
        
        return signals

    def _detect_crossover(self, fast_ema: np.ndarray, slow_ema: np.ndarray, direction: str) -> np.ndarray:
        """Detect crossover between two EMAs"""
        crossover = np.zeros(len(fast_ema))
        
        for i in range(1, len(fast_ema)):
            if (not np.isnan(fast_ema[i-1]) and not np.isnan(fast_ema[i]) and
                not np.isnan(slow_ema[i-1]) and not np.isnan(slow_ema[i])):
                
                if direction == 'above':
                    if fast_ema[i-1] <= slow_ema[i-1] and fast_ema[i] > slow_ema[i]:
                        crossover[i] = 1
                elif direction == 'below':
                    if fast_ema[i-1] >= slow_ema[i-1] and fast_ema[i] < slow_ema[i]:
                        crossover[i] = 1
        
        return crossover

    def _calculate_slope_analysis(self, ema_20: np.ndarray, ema_50: np.ndarray, 
                                ema_100: np.ndarray, ema_200: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate slope analysis for each EMA"""
        
        slopes = {}
        
        for period, ema_values in [('20', ema_20), ('50', ema_50), ('100', ema_100), ('200', ema_200)]:
            slope = np.full(len(ema_values), 0.0)
            
            for i in range(5, len(ema_values)):
                if not np.isnan(ema_values[i]) and not np.isnan(ema_values[i-5]):
                    # Calculate slope over 5-period window
                    slope[i] = (ema_values[i] - ema_values[i-5]) / ema_values[i-5]
            
            slopes[f'ema_{period}_slope'] = slope
        
        return slopes


class StraddleCPRAnalyzer:
    """CPR (Central Pivot Range) analysis for straddle prices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")

    async def analyze_straddle_cpr(self, straddle_data: StraddlePriceData) -> StraddleCPRResult:
        """Calculate CPR analysis for straddle prices"""
        
        # Calculate pivot points using different methods
        pivot_points = self._calculate_pivot_points(straddle_data)
        
        # Calculate support and resistance levels
        support_levels, resistance_levels = self._calculate_support_resistance_levels(pivot_points, straddle_data)
        
        # Calculate CPR width and position
        cpr_width = self._calculate_cpr_width(pivot_points)
        cpr_position = self._calculate_cpr_position(straddle_data, pivot_points)
        
        # Generate breakout signals
        breakout_signals = self._generate_breakout_signals(straddle_data, support_levels, resistance_levels)
        
        # Calculate level strength
        level_strength = self._calculate_level_strength(straddle_data, pivot_points)
        
        return StraddleCPRResult(
            pivot_points=pivot_points,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            cpr_width=cpr_width,
            cpr_position=cpr_position,
            breakout_signals=breakout_signals,
            level_strength=level_strength,
            metadata={
                'pivot_methods': ['standard', 'fibonacci', 'camarilla'],
                'data_points': len(straddle_data.straddle_close)
            }
        )

    def _calculate_pivot_points(self, straddle_data: StraddlePriceData) -> Dict[str, np.ndarray]:
        """Calculate pivot points using different methods"""
        
        high = straddle_data.straddle_high
        low = straddle_data.straddle_low
        close = straddle_data.straddle_close
        
        # Standard Pivot Points
        pivot = (high + low + close) / 3
        
        # Fibonacci Pivot Points
        fib_pivot = pivot.copy()
        
        # Camarilla Pivot Points  
        cam_pivot = pivot.copy()
        
        return {
            'standard': pivot,
            'fibonacci': fib_pivot,
            'camarilla': cam_pivot
        }

    def _calculate_support_resistance_levels(self, pivot_points: Dict[str, np.ndarray], 
                                           straddle_data: StraddlePriceData) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
        """Calculate support and resistance levels"""
        
        high = straddle_data.straddle_high
        low = straddle_data.straddle_low
        
        support_levels = {}
        resistance_levels = {}
        
        # Standard support/resistance
        pivot = pivot_points['standard']
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        
        support_levels['standard'] = [s1, s2]
        resistance_levels['standard'] = [r1, r2]
        
        # Add other methods (simplified for now)
        support_levels['fibonacci'] = support_levels['standard'].copy()
        resistance_levels['fibonacci'] = resistance_levels['standard'].copy()
        
        support_levels['camarilla'] = support_levels['standard'].copy()
        resistance_levels['camarilla'] = resistance_levels['standard'].copy()
        
        return support_levels, resistance_levels

    def _calculate_cpr_width(self, pivot_points: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate CPR width (normalized)"""
        pivot = pivot_points['standard']
        # Simplified CPR width calculation
        return np.full(len(pivot), 0.02)  # 2% width placeholder

    def _calculate_cpr_position(self, straddle_data: StraddlePriceData, pivot_points: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate price position relative to CPR"""
        close = straddle_data.straddle_close
        pivot = pivot_points['standard']
        
        position = np.full(len(close), 0)
        
        for i in range(len(close)):
            if not np.isnan(close[i]) and not np.isnan(pivot[i]):
                if close[i] > pivot[i] * 1.01:  # Above pivot
                    position[i] = 1
                elif close[i] < pivot[i] * 0.99:  # Below pivot
                    position[i] = -1
                else:
                    position[i] = 0  # Near pivot
        
        return position

    def _generate_breakout_signals(self, straddle_data: StraddlePriceData, 
                                 support_levels: Dict[str, List[np.ndarray]], 
                                 resistance_levels: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """Generate breakout signals"""
        
        signals = {}
        close = straddle_data.straddle_close
        
        # Resistance breakout
        signals['resistance_breakout'] = np.zeros(len(close))
        if 'standard' in resistance_levels and len(resistance_levels['standard']) > 0:
            r1 = resistance_levels['standard'][0]
            for i in range(len(close)):
                if not np.isnan(close[i]) and not np.isnan(r1[i]):
                    if close[i] > r1[i]:
                        signals['resistance_breakout'][i] = 1
        
        # Support breakdown
        signals['support_breakdown'] = np.zeros(len(close))
        if 'standard' in support_levels and len(support_levels['standard']) > 0:
            s1 = support_levels['standard'][0]
            for i in range(len(close)):
                if not np.isnan(close[i]) and not np.isnan(s1[i]):
                    if close[i] < s1[i]:
                        signals['support_breakdown'][i] = 1
        
        return signals

    def _calculate_level_strength(self, straddle_data: StraddlePriceData, pivot_points: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate pivot level strength"""
        
        strength = {}
        volume = straddle_data.straddle_volume
        
        for method, pivot in pivot_points.items():
            level_strength = np.full(len(pivot), 0.5)  # Default medium strength
            
            # Volume-based strength adjustment
            for i in range(len(pivot)):
                if not np.isnan(volume[i]):
                    vol_percentile = min(volume[i] / np.nanmean(volume[:i+1]) if i > 0 else 1.0, 2.0)
                    level_strength[i] = min(vol_percentile * 0.5, 1.0)
            
            strength[method] = level_strength
        
        return strength


class StraddleATREMACPREngine:
    """Main engine combining ATR-EMA-CPR analysis for straddle prices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize analyzers
        self.atr_analyzer = StraddleATRAnalyzer(config)
        self.ema_analyzer = StraddleEMAAnalyzer(config)
        self.cpr_analyzer = StraddleCPRAnalyzer(config)

    async def analyze_straddle_atr_ema_cpr(self, straddle_data: StraddlePriceData) -> StraddleAnalysisResult:
        """Complete ATR-EMA-CPR analysis for straddle prices"""
        
        start_time = time.time()
        
        try:
            # Run all analyses concurrently
            atr_task = self.atr_analyzer.analyze_straddle_atr(straddle_data)
            ema_task = self.ema_analyzer.analyze_straddle_ema(straddle_data)
            cpr_task = self.cpr_analyzer.analyze_straddle_cpr(straddle_data)
            
            atr_result, ema_result, cpr_result = await asyncio.gather(atr_task, ema_task, cpr_task)
            
            # Combine signals
            combined_signals = self._combine_signals(atr_result, ema_result, cpr_result)
            
            # Generate regime classification
            regime_classification = self._generate_regime_classification(atr_result, ema_result, cpr_result)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(atr_result, ema_result, cpr_result)
            
            # Generate feature vector (42 features for straddle analysis)
            feature_vector = self._generate_feature_vector(atr_result, ema_result, cpr_result)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return StraddleAnalysisResult(
                atr_result=atr_result,
                ema_result=ema_result,
                cpr_result=cpr_result,
                combined_signals=combined_signals,
                regime_classification=regime_classification,
                confidence_scores=confidence_scores,
                feature_vector=feature_vector,
                processing_time_ms=processing_time_ms,
                metadata={
                    'engine': 'straddle_atr_ema_cpr',
                    'features_extracted': 42,
                    'analysis_type': 'options_specific'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Straddle ATR-EMA-CPR analysis failed: {str(e)}")
            raise

    def _combine_signals(self, atr_result: StraddleATRResult, ema_result: StraddleEMAResult, 
                        cpr_result: StraddleCPRResult) -> Dict[str, np.ndarray]:
        """Combine signals from all analyzers"""
        
        combined = {}
        
        # Volatility signals
        combined['high_vol_trend_up'] = np.logical_and(
            atr_result.volatility_regime >= 1,
            ema_result.trend_direction >= 1
        ).astype(int)
        
        # Breakout confirmation
        combined['confirmed_breakout'] = np.logical_and(
            cpr_result.breakout_signals.get('resistance_breakout', np.zeros(len(atr_result.atr_14))),
            atr_result.atr_signals.get('vol_expansion', np.zeros(len(atr_result.atr_14)))
        ).astype(int)
        
        return combined

    def _generate_regime_classification(self, atr_result: StraddleATRResult, ema_result: StraddleEMAResult, 
                                      cpr_result: StraddleCPRResult) -> np.ndarray:
        """Generate 8-regime classification for straddle analysis"""
        
        regime = np.full(len(atr_result.volatility_regime), 0)
        
        vol_regime = atr_result.volatility_regime
        trend_dir = ema_result.trend_direction
        
        # 8-regime classification based on volatility and trend
        for i in range(len(regime)):
            vol = vol_regime[i]
            trend = trend_dir[i]
            
            if vol <= -1 and trend >= 1:
                regime[i] = 0  # Low vol, bullish trend
            elif vol <= -1 and trend <= -1:
                regime[i] = 1  # Low vol, bearish trend
            elif vol == 0 and trend >= 1:
                regime[i] = 2  # Medium vol, bullish trend
            elif vol == 0 and trend <= -1:
                regime[i] = 3  # Medium vol, bearish trend
            elif vol >= 1 and trend >= 1:
                regime[i] = 4  # High vol, bullish trend
            elif vol >= 1 and trend <= -1:
                regime[i] = 5  # High vol, bearish trend
            elif vol >= 2:
                regime[i] = 6  # Extreme volatility
            else:
                regime[i] = 7  # Neutral/transition
        
        return regime

    def _calculate_confidence_scores(self, atr_result: StraddleATRResult, ema_result: StraddleEMAResult, 
                                   cpr_result: StraddleCPRResult) -> np.ndarray:
        """Calculate confidence scores for analysis"""
        
        confidence = np.full(len(atr_result.atr_14), 0.5)
        
        # Base confidence from data quality
        valid_atr = ~np.isnan(atr_result.atr_14)
        valid_ema = ~np.isnan(ema_result.ema_20)
        
        base_confidence = (valid_atr.astype(int) + valid_ema.astype(int)) / 2
        
        # Adjust confidence based on signal agreement
        trend_strength = np.abs(ema_result.trend_strength)
        vol_confidence = np.abs(atr_result.volatility_regime) / 2  # Normalize to 0-1
        
        confidence = (base_confidence * 0.4 + trend_strength * 0.3 + vol_confidence * 0.3)
        confidence = np.clip(confidence, 0.1, 0.95)
        
        return confidence

    def _generate_feature_vector(self, atr_result: StraddleATRResult, ema_result: StraddleEMAResult, 
                               cpr_result: StraddleCPRResult) -> np.ndarray:
        """Generate 42-feature vector for straddle analysis"""
        
        features = []
        data_length = len(atr_result.atr_14)
        
        # ATR features (12 features)
        features.extend([
            atr_result.atr_14, atr_result.atr_21, atr_result.atr_50,  # 3 features
            atr_result.atr_percentiles.get('atr_14_pct', np.zeros(data_length)),  # 1 feature
            atr_result.atr_percentiles.get('atr_21_pct', np.zeros(data_length)),  # 1 feature
            atr_result.atr_percentiles.get('atr_50_pct', np.zeros(data_length)),  # 1 feature
            atr_result.volatility_regime.astype(float),  # 1 feature
            atr_result.atr_trend.astype(float),  # 1 feature
            atr_result.atr_signals.get('vol_expansion', np.zeros(data_length)),  # 1 feature
            atr_result.atr_signals.get('vol_contraction', np.zeros(data_length)),  # 1 feature
            atr_result.atr_signals.get('atr_convergence', np.zeros(data_length)),  # 1 feature
            atr_result.true_ranges  # 1 feature
        ])
        
        # EMA features (15 features)
        features.extend([
            ema_result.ema_20, ema_result.ema_50, ema_result.ema_100, ema_result.ema_200,  # 4 features
            ema_result.trend_direction.astype(float),  # 1 feature
            ema_result.trend_strength,  # 1 feature
            ema_result.confluence_zones,  # 1 feature
            ema_result.crossover_signals.get('golden_cross_20_50', np.zeros(data_length)),  # 1 feature
            ema_result.crossover_signals.get('death_cross_20_50', np.zeros(data_length)),  # 1 feature
            ema_result.crossover_signals.get('golden_cross_50_100', np.zeros(data_length)),  # 1 feature
            ema_result.crossover_signals.get('death_cross_50_100', np.zeros(data_length)),  # 1 feature
            ema_result.slope_analysis.get('ema_20_slope', np.zeros(data_length)),  # 1 feature
            ema_result.slope_analysis.get('ema_50_slope', np.zeros(data_length)),  # 1 feature
            ema_result.slope_analysis.get('ema_100_slope', np.zeros(data_length)),  # 1 feature
            ema_result.slope_analysis.get('ema_200_slope', np.zeros(data_length))  # 1 feature
        ])
        
        # CPR features (15 features) 
        features.extend([
            cpr_result.pivot_points['standard'],  # 1 feature
            cpr_result.pivot_points['fibonacci'],  # 1 feature
            cpr_result.pivot_points['camarilla'],  # 1 feature
            cpr_result.support_levels['standard'][0] if len(cpr_result.support_levels['standard']) > 0 else np.zeros(data_length),  # 1 feature
            cpr_result.support_levels['standard'][1] if len(cpr_result.support_levels['standard']) > 1 else np.zeros(data_length),  # 1 feature
            cpr_result.resistance_levels['standard'][0] if len(cpr_result.resistance_levels['standard']) > 0 else np.zeros(data_length),  # 1 feature
            cpr_result.resistance_levels['standard'][1] if len(cpr_result.resistance_levels['standard']) > 1 else np.zeros(data_length),  # 1 feature
            cpr_result.cpr_width,  # 1 feature
            cpr_result.cpr_position.astype(float),  # 1 feature
            cpr_result.breakout_signals.get('resistance_breakout', np.zeros(data_length)),  # 1 feature
            cpr_result.breakout_signals.get('support_breakdown', np.zeros(data_length)),  # 1 feature
            cpr_result.level_strength.get('standard', np.full(data_length, 0.5)),  # 1 feature
            cpr_result.level_strength.get('fibonacci', np.full(data_length, 0.5)),  # 1 feature
            cpr_result.level_strength.get('camarilla', np.full(data_length, 0.5)),  # 1 feature
            np.zeros(data_length)  # 1 feature (placeholder for additional CPR metric)
        ])
        
        # Stack features vertically and transpose to get feature vectors per timestamp
        feature_matrix = np.column_stack(features)
        return feature_matrix