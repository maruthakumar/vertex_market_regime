"""
Component 5: Underlying ATR-EMA-CPR Multi-Timeframe Analysis Engine

Comprehensive multi-timeframe ATR-EMA-CPR analysis for underlying prices providing
traditional trend analysis across daily/weekly/monthly timeframes with cross-timeframe
validation and comprehensive regime classification.

Features:
- Multi-timeframe ATR analysis (daily/weekly/monthly) with different periods
- Multi-timeframe EMA trend analysis with comprehensive trend context  
- Multi-timeframe CPR analysis (standard/fibonacci/camarilla pivots)
- Cross-timeframe validation and trend agreement analysis
- Historical percentile tracking for each timeframe
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

from .dual_asset_data_extractor import UnderlyingPriceData

warnings.filterwarnings('ignore')


@dataclass
class UnderlyingATRResult:
    """Multi-timeframe ATR analysis result for underlying prices"""
    daily_atr: Dict[str, np.ndarray]  # ATR-14/21/50 for daily
    weekly_atr: Dict[str, np.ndarray]  # ATR-14/21/50 for weekly
    monthly_atr: Dict[str, np.ndarray]  # ATR-14/21 for monthly
    atr_percentiles: Dict[str, Dict[str, np.ndarray]]  # Percentiles per timeframe
    cross_timeframe_consistency: Dict[str, np.ndarray]
    volatility_regimes: Dict[str, np.ndarray]
    atr_trends: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class UnderlyingEMAResult:
    """Multi-timeframe EMA analysis result for underlying prices"""
    daily_ema: Dict[str, np.ndarray]  # EMA-20/50/100/200 for daily
    weekly_ema: Dict[str, np.ndarray]  # EMA-10/20/50 for weekly  
    monthly_ema: Dict[str, np.ndarray]  # EMA-6/12/24 for monthly
    trend_directions: Dict[str, np.ndarray]
    trend_strengths: Dict[str, np.ndarray]
    cross_timeframe_agreement: Dict[str, np.ndarray]
    confluence_zones: Dict[str, np.ndarray]
    crossover_signals: Dict[str, Dict[str, np.ndarray]]
    metadata: Dict[str, Any]


@dataclass
class UnderlyingCPRResult:
    """Multi-timeframe CPR analysis result for underlying prices"""
    daily_cpr: Dict[str, Dict[str, np.ndarray]]
    weekly_cpr: Dict[str, Dict[str, np.ndarray]]
    monthly_cpr: Dict[str, Dict[str, np.ndarray]]
    support_resistance_validation: Dict[str, np.ndarray]
    cross_timeframe_levels: Dict[str, np.ndarray]
    level_confluence: Dict[str, np.ndarray]
    breakout_signals: Dict[str, Dict[str, np.ndarray]]
    metadata: Dict[str, Any]


@dataclass
class UnderlyingAnalysisResult:
    """Complete underlying multi-timeframe analysis result"""
    atr_result: UnderlyingATRResult
    ema_result: UnderlyingEMAResult
    cpr_result: UnderlyingCPRResult
    combined_regime_classification: np.ndarray
    cross_timeframe_confidence: np.ndarray
    feature_vector: np.ndarray
    processing_time_ms: float
    metadata: Dict[str, Any]


class UnderlyingATRAnalyzer:
    """Multi-timeframe ATR analysis for underlying prices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Multi-timeframe ATR parameters
        self.timeframe_periods = {
            'daily': [14, 21, 50],
            'weekly': [14, 21, 50], 
            'monthly': [14, 21]
        }
        
        # Volatility regime thresholds
        self.volatility_thresholds = {
            'low': 0.25,
            'medium': 0.5, 
            'high': 0.75,
            'extreme': 0.95
        }

    async def analyze_underlying_atr(self, underlying_data: UnderlyingPriceData) -> UnderlyingATRResult:
        """Multi-timeframe ATR analysis for underlying prices"""
        
        # Analyze each timeframe
        daily_atr = await self._analyze_timeframe_atr(underlying_data.timeframes['daily'], 'daily')
        weekly_atr = await self._analyze_timeframe_atr(underlying_data.timeframes['weekly'], 'weekly')
        monthly_atr = await self._analyze_timeframe_atr(underlying_data.timeframes['monthly'], 'monthly')
        
        # Calculate percentiles for each timeframe
        atr_percentiles = {
            'daily': self._calculate_timeframe_atr_percentiles(daily_atr),
            'weekly': self._calculate_timeframe_atr_percentiles(weekly_atr),
            'monthly': self._calculate_timeframe_atr_percentiles(monthly_atr)
        }
        
        # Cross-timeframe consistency analysis
        cross_timeframe_consistency = self._calculate_cross_timeframe_consistency(
            daily_atr, weekly_atr, monthly_atr
        )
        
        # Volatility regime classification per timeframe
        volatility_regimes = {
            'daily': self._classify_volatility_regime(atr_percentiles['daily']),
            'weekly': self._classify_volatility_regime(atr_percentiles['weekly']),
            'monthly': self._classify_volatility_regime(atr_percentiles['monthly'])
        }
        
        # ATR trend analysis per timeframe
        atr_trends = {
            'daily': self._calculate_atr_trend(daily_atr),
            'weekly': self._calculate_atr_trend(weekly_atr),
            'monthly': self._calculate_atr_trend(monthly_atr)
        }
        
        return UnderlyingATRResult(
            daily_atr=daily_atr,
            weekly_atr=weekly_atr,
            monthly_atr=monthly_atr,
            atr_percentiles=atr_percentiles,
            cross_timeframe_consistency=cross_timeframe_consistency,
            volatility_regimes=volatility_regimes,
            atr_trends=atr_trends,
            metadata={
                'timeframes': ['daily', 'weekly', 'monthly'],
                'periods': self.timeframe_periods,
                'analysis_type': 'multi_timeframe_underlying'
            }
        )

    async def _analyze_timeframe_atr(self, timeframe_data: Dict[str, np.ndarray], timeframe: str) -> Dict[str, np.ndarray]:
        """Analyze ATR for specific timeframe"""
        
        high = timeframe_data['high']
        low = timeframe_data['low']
        close = timeframe_data['close']
        
        # Calculate True Range
        true_ranges = self._calculate_underlying_true_range(high, low, close)
        
        atr_results = {}
        periods = self.timeframe_periods[timeframe]
        
        for period in periods:
            atr_results[f'atr_{period}'] = self._calculate_atr(true_ranges, period)
        
        atr_results['true_ranges'] = true_ranges
        
        return atr_results

    def _calculate_underlying_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range for underlying prices"""
        
        close_prev = np.roll(close, 1)
        close_prev[0] = close[0]  # Handle first value
        
        # True Range: max of (H-L, H-C_prev, C_prev-L)
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(close_prev - low)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range

    def _calculate_atr(self, true_ranges: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range using Wilder's smoothing"""
        
        atr = np.full(len(true_ranges), np.nan)
        
        if len(true_ranges) >= period:
            # Initial ATR (simple average)
            atr[period-1] = np.mean(true_ranges[:period])
            
            # Wilder's smoothing
            for i in range(period, len(true_ranges)):
                atr[i] = (atr[i-1] * (period-1) + true_ranges[i]) / period
        
        return atr

    def _calculate_timeframe_atr_percentiles(self, atr_results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate ATR percentiles for timeframe"""
        
        percentiles = {}
        
        for atr_name, atr_values in atr_results.items():
            if atr_name != 'true_ranges':
                valid_atr = atr_values[~np.isnan(atr_values)]
                if len(valid_atr) > 0:
                    percentiles[f'{atr_name}_pct'] = np.full(len(atr_values), np.nan)
                    for i, value in enumerate(atr_values):
                        if not np.isnan(value):
                            percentiles[f'{atr_name}_pct'][i] = stats.percentileofscore(valid_atr, value) / 100
        
        return percentiles

    def _calculate_cross_timeframe_consistency(self, daily_atr: Dict[str, np.ndarray], 
                                             weekly_atr: Dict[str, np.ndarray], 
                                             monthly_atr: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate cross-timeframe ATR consistency"""
        
        consistency = {}
        
        # Compare ATR-14 across timeframes (all have this period)
        daily_atr_14 = daily_atr.get('atr_14', np.array([]))
        weekly_atr_14 = weekly_atr.get('atr_14', np.array([]))
        monthly_atr_14 = monthly_atr.get('atr_14', np.array([]))
        
        if len(daily_atr_14) > 0:
            # Weekly-daily consistency
            consistency['weekly_daily'] = self._calculate_timeframe_agreement(daily_atr_14, weekly_atr_14)
            
            # Monthly-daily consistency  
            consistency['monthly_daily'] = self._calculate_timeframe_agreement(daily_atr_14, monthly_atr_14)
            
            # Three-way consistency
            consistency['all_timeframes'] = self._calculate_three_way_consistency(
                daily_atr_14, weekly_atr_14, monthly_atr_14
            )
        
        return consistency

    def _calculate_timeframe_agreement(self, tf1_values: np.ndarray, tf2_values: np.ndarray) -> np.ndarray:
        """Calculate agreement between two timeframe ATR values"""
        
        min_length = min(len(tf1_values), len(tf2_values))
        agreement = np.full(min_length, 0.0)
        
        for i in range(min_length):
            if not np.isnan(tf1_values[i]) and not np.isnan(tf2_values[i]):
                # Calculate relative difference
                rel_diff = abs(tf1_values[i] - tf2_values[i]) / max(tf1_values[i], tf2_values[i], 1e-10)
                # Agreement score (1 = perfect agreement, 0 = maximum disagreement)
                agreement[i] = max(0, 1 - rel_diff * 2)
        
        return agreement

    def _calculate_three_way_consistency(self, daily: np.ndarray, weekly: np.ndarray, monthly: np.ndarray) -> np.ndarray:
        """Calculate three-way consistency across all timeframes"""
        
        min_length = min(len(daily), len(weekly), len(monthly))
        consistency = np.full(min_length, 0.0)
        
        for i in range(min_length):
            values = []
            for val in [daily[i], weekly[i], monthly[i]]:
                if not np.isnan(val):
                    values.append(val)
            
            if len(values) >= 2:
                # Calculate coefficient of variation
                if np.mean(values) > 0:
                    cv = np.std(values) / np.mean(values)
                    consistency[i] = max(0, 1 - cv)  # Higher consistency = lower CV
        
        return consistency

    def _classify_volatility_regime(self, atr_percentiles: Dict[str, np.ndarray]) -> np.ndarray:
        """Classify volatility regime for timeframe"""
        
        # Use first available ATR percentiles for classification
        atr_pct_key = next(iter(atr_percentiles.keys()))
        atr_pct = atr_percentiles[atr_pct_key]
        
        regime = np.full(len(atr_pct), 0)
        
        regime[atr_pct < self.volatility_thresholds['low']] = -2    # Very Low
        regime[(atr_pct >= self.volatility_thresholds['low']) & 
               (atr_pct < self.volatility_thresholds['medium'])] = -1  # Low
        regime[(atr_pct >= self.volatility_thresholds['medium']) & 
               (atr_pct < self.volatility_thresholds['high'])] = 0     # Medium
        regime[(atr_pct >= self.volatility_thresholds['high']) & 
               (atr_pct < self.volatility_thresholds['extreme'])] = 1   # High
        regime[atr_pct >= self.volatility_thresholds['extreme']] = 2    # Extreme
        
        return regime

    def _calculate_atr_trend(self, atr_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate ATR trend for timeframe"""
        
        # Use ATR-14 for trend analysis
        atr_14 = atr_results.get('atr_14', np.array([]))
        
        if len(atr_14) == 0:
            return np.array([])
        
        trend = np.full(len(atr_14), 0)
        
        # 5-period trend analysis
        for i in range(5, len(atr_14)):
            if not np.isnan(atr_14[i]) and not np.isnan(atr_14[i-5]):
                current = atr_14[i]
                past = atr_14[i-5]
                
                if current > past * 1.1:  # 10% increase
                    trend[i] = 1  # Increasing volatility
                elif current < past * 0.9:  # 10% decrease
                    trend[i] = -1  # Decreasing volatility
                else:
                    trend[i] = 0  # Stable volatility
        
        return trend


class UnderlyingEMAAnalyzer:
    """Multi-timeframe EMA analysis for underlying prices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Multi-timeframe EMA parameters
        self.timeframe_ema_periods = {
            'daily': [20, 50, 100, 200],
            'weekly': [10, 20, 50],
            'monthly': [6, 12, 24]
        }

    async def analyze_underlying_ema(self, underlying_data: UnderlyingPriceData) -> UnderlyingEMAResult:
        """Multi-timeframe EMA analysis for underlying prices"""
        
        # Analyze each timeframe
        daily_ema = await self._analyze_timeframe_ema(underlying_data.timeframes['daily'], 'daily')
        weekly_ema = await self._analyze_timeframe_ema(underlying_data.timeframes['weekly'], 'weekly') 
        monthly_ema = await self._analyze_timeframe_ema(underlying_data.timeframes['monthly'], 'monthly')
        
        # Calculate trend directions per timeframe
        trend_directions = {
            'daily': self._calculate_trend_direction(daily_ema),
            'weekly': self._calculate_trend_direction(weekly_ema),
            'monthly': self._calculate_trend_direction(monthly_ema)
        }
        
        # Calculate trend strengths per timeframe
        trend_strengths = {
            'daily': self._calculate_trend_strength(underlying_data.timeframes['daily'], daily_ema),
            'weekly': self._calculate_trend_strength(underlying_data.timeframes['weekly'], weekly_ema),
            'monthly': self._calculate_trend_strength(underlying_data.timeframes['monthly'], monthly_ema)
        }
        
        # Cross-timeframe agreement analysis
        cross_timeframe_agreement = self._calculate_cross_timeframe_trend_agreement(trend_directions)
        
        # Identify confluence zones per timeframe
        confluence_zones = {
            'daily': self._identify_confluence_zones(daily_ema),
            'weekly': self._identify_confluence_zones(weekly_ema),
            'monthly': self._identify_confluence_zones(monthly_ema)
        }
        
        # Generate crossover signals per timeframe
        crossover_signals = {
            'daily': self._generate_crossover_signals(daily_ema, 'daily'),
            'weekly': self._generate_crossover_signals(weekly_ema, 'weekly'),
            'monthly': self._generate_crossover_signals(monthly_ema, 'monthly')
        }
        
        return UnderlyingEMAResult(
            daily_ema=daily_ema,
            weekly_ema=weekly_ema,
            monthly_ema=monthly_ema,
            trend_directions=trend_directions,
            trend_strengths=trend_strengths,
            cross_timeframe_agreement=cross_timeframe_agreement,
            confluence_zones=confluence_zones,
            crossover_signals=crossover_signals,
            metadata={
                'timeframes': ['daily', 'weekly', 'monthly'],
                'ema_periods': self.timeframe_ema_periods,
                'analysis_type': 'multi_timeframe_ema'
            }
        )

    async def _analyze_timeframe_ema(self, timeframe_data: Dict[str, np.ndarray], timeframe: str) -> Dict[str, np.ndarray]:
        """Analyze EMA for specific timeframe"""
        
        close = timeframe_data['close']
        ema_results = {}
        
        periods = self.timeframe_ema_periods[timeframe]
        
        for period in periods:
            ema_results[f'ema_{period}'] = self._calculate_ema(close, period)
        
        return ema_results

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        
        ema = np.full(len(prices), np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid price
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

    def _calculate_trend_direction(self, ema_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate trend direction based on EMA alignment"""
        
        ema_values = list(ema_results.values())
        if not ema_values:
            return np.array([])
        
        trend = np.full(len(ema_values[0]), 0)
        
        for i in range(len(trend)):
            valid_emas = []
            for ema_array in ema_values:
                if not np.isnan(ema_array[i]):
                    valid_emas.append(ema_array[i])
            
            if len(valid_emas) >= 2:
                # Check EMA alignment
                ascending = all(valid_emas[j] >= valid_emas[j+1] for j in range(len(valid_emas)-1))
                descending = all(valid_emas[j] <= valid_emas[j+1] for j in range(len(valid_emas)-1))
                
                if ascending:
                    trend[i] = 1    # Bullish
                elif descending:
                    trend[i] = -1   # Bearish
                else:
                    trend[i] = 0    # Neutral
        
        return trend

    def _calculate_trend_strength(self, timeframe_data: Dict[str, np.ndarray], ema_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate trend strength for timeframe"""
        
        close = timeframe_data['close']
        
        # Use shortest EMA for strength calculation
        ema_keys = list(ema_results.keys())
        if not ema_keys:
            return np.full(len(close), 0.0)
        
        shortest_ema = ema_results[ema_keys[0]]  # First EMA (typically shortest period)
        
        strength = np.full(len(close), 0.0)
        
        for i in range(len(close)):
            if not np.isnan(close[i]) and not np.isnan(shortest_ema[i]):
                # Distance from EMA as strength measure
                strength[i] = abs(close[i] - shortest_ema[i]) / shortest_ema[i]
        
        return strength

    def _calculate_cross_timeframe_trend_agreement(self, trend_directions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate agreement between timeframe trends"""
        
        agreement = {}
        
        daily_trend = trend_directions.get('daily', np.array([]))
        weekly_trend = trend_directions.get('weekly', np.array([]))
        monthly_trend = trend_directions.get('monthly', np.array([]))
        
        if len(daily_trend) > 0:
            # Daily-Weekly agreement
            agreement['daily_weekly'] = self._calculate_trend_agreement(daily_trend, weekly_trend)
            
            # Daily-Monthly agreement
            agreement['daily_monthly'] = self._calculate_trend_agreement(daily_trend, monthly_trend)
            
            # All timeframes agreement
            agreement['all_timeframes'] = self._calculate_three_way_trend_agreement(
                daily_trend, weekly_trend, monthly_trend
            )
        
        return agreement

    def _calculate_trend_agreement(self, trend1: np.ndarray, trend2: np.ndarray) -> np.ndarray:
        """Calculate agreement between two trend arrays"""
        
        min_length = min(len(trend1), len(trend2))
        agreement = np.full(min_length, 0.0)
        
        for i in range(min_length):
            if trend1[i] == trend2[i]:
                agreement[i] = 1.0  # Perfect agreement
            elif (trend1[i] == 0 and trend2[i] != 0) or (trend1[i] != 0 and trend2[i] == 0):
                agreement[i] = 0.5  # Partial agreement (neutral vs directional)
            else:
                agreement[i] = 0.0  # Disagreement
        
        return agreement

    def _calculate_three_way_trend_agreement(self, daily: np.ndarray, weekly: np.ndarray, monthly: np.ndarray) -> np.ndarray:
        """Calculate three-way trend agreement"""
        
        min_length = min(len(daily), len(weekly), len(monthly))
        agreement = np.full(min_length, 0.0)
        
        for i in range(min_length):
            trends = [daily[i], weekly[i], monthly[i]]
            
            # All same
            if len(set(trends)) == 1:
                agreement[i] = 1.0
            # Two same, one different
            elif len(set(trends)) == 2:
                agreement[i] = 0.6
            # All different
            else:
                agreement[i] = 0.0
        
        return agreement

    def _identify_confluence_zones(self, ema_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Identify EMA confluence zones for timeframe"""
        
        ema_values = list(ema_results.values())
        if not ema_values:
            return np.array([])
        
        confluence = np.zeros(len(ema_values[0]))
        tolerance = 0.02  # 2% tolerance
        
        for i in range(len(confluence)):
            valid_emas = []
            for ema_array in ema_values:
                if not np.isnan(ema_array[i]):
                    valid_emas.append(ema_array[i])
            
            if len(valid_emas) >= 3:
                # Check if EMAs are close together
                ema_range = max(valid_emas) - min(valid_emas)
                avg_ema = np.mean(valid_emas)
                
                if ema_range / avg_ema < tolerance:
                    confluence[i] = 1
        
        return confluence

    def _generate_crossover_signals(self, ema_results: Dict[str, np.ndarray], timeframe: str) -> Dict[str, np.ndarray]:
        """Generate EMA crossover signals for timeframe"""
        
        signals = {}
        
        ema_keys = list(ema_results.keys())
        if len(ema_keys) < 2:
            return signals
        
        # Use first two EMAs for crossover signals (typically fastest)
        fast_ema = ema_results[ema_keys[0]]
        slow_ema = ema_results[ema_keys[1]]
        
        signals['golden_cross'] = self._detect_crossover(fast_ema, slow_ema, 'above')
        signals['death_cross'] = self._detect_crossover(fast_ema, slow_ema, 'below')
        
        return signals

    def _detect_crossover(self, fast_ema: np.ndarray, slow_ema: np.ndarray, direction: str) -> np.ndarray:
        """Detect crossover between EMAs"""
        
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


class UnderlyingCPRAnalyzer:
    """Multi-timeframe CPR analysis for underlying prices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")

    async def analyze_underlying_cpr(self, underlying_data: UnderlyingPriceData) -> UnderlyingCPRResult:
        """Multi-timeframe CPR analysis for underlying prices"""
        
        # Analyze each timeframe
        daily_cpr = await self._analyze_timeframe_cpr(underlying_data.timeframes['daily'])
        weekly_cpr = await self._analyze_timeframe_cpr(underlying_data.timeframes['weekly'])
        monthly_cpr = await self._analyze_timeframe_cpr(underlying_data.timeframes['monthly'])
        
        # Cross-timeframe support/resistance validation
        support_resistance_validation = self._validate_cross_timeframe_levels(daily_cpr, weekly_cpr, monthly_cpr)
        
        # Cross-timeframe level analysis
        cross_timeframe_levels = self._analyze_cross_timeframe_levels(daily_cpr, weekly_cpr, monthly_cpr)
        
        # Level confluence analysis
        level_confluence = self._analyze_level_confluence(daily_cpr, weekly_cpr, monthly_cpr)
        
        # Breakout signals per timeframe
        breakout_signals = {
            'daily': self._generate_timeframe_breakout_signals(underlying_data.timeframes['daily'], daily_cpr),
            'weekly': self._generate_timeframe_breakout_signals(underlying_data.timeframes['weekly'], weekly_cpr),
            'monthly': self._generate_timeframe_breakout_signals(underlying_data.timeframes['monthly'], monthly_cpr)
        }
        
        return UnderlyingCPRResult(
            daily_cpr=daily_cpr,
            weekly_cpr=weekly_cpr,
            monthly_cpr=monthly_cpr,
            support_resistance_validation=support_resistance_validation,
            cross_timeframe_levels=cross_timeframe_levels,
            level_confluence=level_confluence,
            breakout_signals=breakout_signals,
            metadata={
                'timeframes': ['daily', 'weekly', 'monthly'],
                'pivot_methods': ['standard', 'fibonacci', 'camarilla'],
                'analysis_type': 'multi_timeframe_cpr'
            }
        )

    async def _analyze_timeframe_cpr(self, timeframe_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze CPR for specific timeframe"""
        
        high = timeframe_data['high']
        low = timeframe_data['low']
        close = timeframe_data['close']
        
        cpr_results = {}
        
        # Standard Pivot Points
        cpr_results['standard'] = self._calculate_standard_pivots(high, low, close)
        
        # Fibonacci Pivot Points
        cpr_results['fibonacci'] = self._calculate_fibonacci_pivots(high, low, close)
        
        # Camarilla Pivot Points
        cpr_results['camarilla'] = self._calculate_camarilla_pivots(high, low, close)
        
        return cpr_results

    def _calculate_standard_pivots(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate standard pivot points"""
        
        pivot = (high + low + close) / 3
        
        # Support levels
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = s1 - (high - low)
        
        # Resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = r1 + (high - low)
        
        return {
            'pivot': pivot,
            's1': s1, 's2': s2, 's3': s3,
            'r1': r1, 'r2': r2, 'r3': r3
        }

    def _calculate_fibonacci_pivots(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Fibonacci pivot points"""
        
        pivot = (high + low + close) / 3
        hl_diff = high - low
        
        # Fibonacci ratios
        fib_618 = 0.618
        fib_382 = 0.382
        
        # Support levels
        s1 = pivot - fib_382 * hl_diff
        s2 = pivot - fib_618 * hl_diff
        s3 = pivot - hl_diff
        
        # Resistance levels  
        r1 = pivot + fib_382 * hl_diff
        r2 = pivot + fib_618 * hl_diff
        r3 = pivot + hl_diff
        
        return {
            'pivot': pivot,
            's1': s1, 's2': s2, 's3': s3,
            'r1': r1, 'r2': r2, 'r3': r3
        }

    def _calculate_camarilla_pivots(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Camarilla pivot points"""
        
        hl_diff = high - low
        
        # Camarilla multipliers
        c1_mult = 0.0916  # 1.1/12
        c2_mult = 0.183   # 2.2/12
        c3_mult = 0.275   # 3.3/12
        c4_mult = 0.55    # 6.6/12
        
        # Support levels
        s1 = close - c1_mult * hl_diff
        s2 = close - c2_mult * hl_diff
        s3 = close - c3_mult * hl_diff
        s4 = close - c4_mult * hl_diff
        
        # Resistance levels
        r1 = close + c1_mult * hl_diff
        r2 = close + c2_mult * hl_diff
        r3 = close + c3_mult * hl_diff
        r4 = close + c4_mult * hl_diff
        
        return {
            'pivot': close,  # Camarilla uses close as pivot
            's1': s1, 's2': s2, 's3': s3, 's4': s4,
            'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4
        }

    def _validate_cross_timeframe_levels(self, daily_cpr: Dict[str, Dict[str, np.ndarray]], 
                                       weekly_cpr: Dict[str, Dict[str, np.ndarray]], 
                                       monthly_cpr: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Validate support/resistance levels across timeframes"""
        
        validation = {}
        
        # Get pivot levels from each timeframe
        daily_pivot = daily_cpr.get('standard', {}).get('pivot', np.array([]))
        weekly_pivot = weekly_cpr.get('standard', {}).get('pivot', np.array([]))
        monthly_pivot = monthly_cpr.get('standard', {}).get('pivot', np.array([]))
        
        if len(daily_pivot) > 0:
            # Level agreement across timeframes
            validation['pivot_agreement'] = self._calculate_level_agreement(daily_pivot, weekly_pivot, monthly_pivot)
            
            # Support level strength
            daily_s1 = daily_cpr.get('standard', {}).get('s1', np.array([]))
            weekly_s1 = weekly_cpr.get('standard', {}).get('s1', np.array([]))
            validation['support_strength'] = self._calculate_level_strength(daily_s1, weekly_s1)
            
            # Resistance level strength  
            daily_r1 = daily_cpr.get('standard', {}).get('r1', np.array([]))
            weekly_r1 = weekly_cpr.get('standard', {}).get('r1', np.array([]))
            validation['resistance_strength'] = self._calculate_level_strength(daily_r1, weekly_r1)
        
        return validation

    def _calculate_level_agreement(self, daily: np.ndarray, weekly: np.ndarray, monthly: np.ndarray) -> np.ndarray:
        """Calculate agreement between pivot levels across timeframes"""
        
        min_length = min(len(daily), len(weekly), len(monthly))
        agreement = np.full(min_length, 0.0)
        
        tolerance = 0.02  # 2% tolerance for level agreement
        
        for i in range(min_length):
            if not np.isnan(daily[i]) and not np.isnan(weekly[i]) and not np.isnan(monthly[i]):
                levels = [daily[i], weekly[i], monthly[i]]
                level_range = max(levels) - min(levels)
                avg_level = np.mean(levels)
                
                if level_range / avg_level < tolerance:
                    agreement[i] = 1.0  # Strong agreement
                elif level_range / avg_level < tolerance * 2:
                    agreement[i] = 0.5  # Moderate agreement
                else:
                    agreement[i] = 0.0  # Poor agreement
        
        return agreement

    def _calculate_level_strength(self, tf1_levels: np.ndarray, tf2_levels: np.ndarray) -> np.ndarray:
        """Calculate strength of support/resistance levels"""
        
        min_length = min(len(tf1_levels), len(tf2_levels))
        strength = np.full(min_length, 0.5)  # Default medium strength
        
        tolerance = 0.015  # 1.5% tolerance for level confluence
        
        for i in range(min_length):
            if not np.isnan(tf1_levels[i]) and not np.isnan(tf2_levels[i]):
                rel_diff = abs(tf1_levels[i] - tf2_levels[i]) / max(tf1_levels[i], tf2_levels[i], 1e-10)
                
                if rel_diff < tolerance:
                    strength[i] = 0.9  # Strong level
                elif rel_diff < tolerance * 2:
                    strength[i] = 0.7  # Medium-strong level
                else:
                    strength[i] = 0.3  # Weak level
        
        return strength

    def _analyze_cross_timeframe_levels(self, daily_cpr: Dict[str, Dict[str, np.ndarray]], 
                                      weekly_cpr: Dict[str, Dict[str, np.ndarray]], 
                                      monthly_cpr: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Analyze cross-timeframe level interactions"""
        
        cross_levels = {}
        
        # Weekly levels acting as major support/resistance for daily
        daily_close = daily_cpr.get('standard', {}).get('pivot', np.array([]))
        weekly_r1 = weekly_cpr.get('standard', {}).get('r1', np.array([]))
        weekly_s1 = weekly_cpr.get('standard', {}).get('s1', np.array([]))
        
        if len(daily_close) > 0 and len(weekly_r1) > 0:
            cross_levels['weekly_resistance_test'] = self._detect_level_test(daily_close, weekly_r1)
            
        if len(daily_close) > 0 and len(weekly_s1) > 0:
            cross_levels['weekly_support_test'] = self._detect_level_test(daily_close, weekly_s1)
        
        return cross_levels

    def _detect_level_test(self, prices: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Detect when prices test support/resistance levels"""
        
        min_length = min(len(prices), len(levels))
        tests = np.zeros(min_length)
        
        tolerance = 0.01  # 1% tolerance for level test
        
        for i in range(min_length):
            if not np.isnan(prices[i]) and not np.isnan(levels[i]):
                rel_diff = abs(prices[i] - levels[i]) / levels[i]
                
                if rel_diff < tolerance:
                    tests[i] = 1  # Level test detected
        
        return tests

    def _analyze_level_confluence(self, daily_cpr: Dict[str, Dict[str, np.ndarray]], 
                                weekly_cpr: Dict[str, Dict[str, np.ndarray]], 
                                monthly_cpr: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Analyze confluence between different pivot methods and timeframes"""
        
        confluence = {}
        
        # Method confluence within daily timeframe
        if 'standard' in daily_cpr and 'fibonacci' in daily_cpr:
            standard_pivot = daily_cpr['standard'].get('pivot', np.array([]))
            fib_pivot = daily_cpr['fibonacci'].get('pivot', np.array([]))
            
            confluence['daily_method_confluence'] = self._calculate_method_confluence(standard_pivot, fib_pivot)
        
        return confluence

    def _calculate_method_confluence(self, method1_levels: np.ndarray, method2_levels: np.ndarray) -> np.ndarray:
        """Calculate confluence between different pivot calculation methods"""
        
        min_length = min(len(method1_levels), len(method2_levels))
        confluence = np.zeros(min_length)
        
        tolerance = 0.015  # 1.5% tolerance
        
        for i in range(min_length):
            if not np.isnan(method1_levels[i]) and not np.isnan(method2_levels[i]):
                rel_diff = abs(method1_levels[i] - method2_levels[i]) / max(method1_levels[i], method2_levels[i], 1e-10)
                
                if rel_diff < tolerance:
                    confluence[i] = 1  # Strong confluence
        
        return confluence

    def _generate_timeframe_breakout_signals(self, timeframe_data: Dict[str, np.ndarray], 
                                           cpr_result: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Generate breakout signals for specific timeframe"""
        
        signals = {}
        
        close = timeframe_data['close']
        standard_cpr = cpr_result.get('standard', {})
        
        if 'r1' in standard_cpr and 's1' in standard_cpr:
            r1 = standard_cpr['r1']
            s1 = standard_cpr['s1']
            
            # Resistance breakout
            signals['resistance_breakout'] = np.zeros(len(close))
            for i in range(len(close)):
                if not np.isnan(close[i]) and not np.isnan(r1[i]):
                    if close[i] > r1[i]:
                        signals['resistance_breakout'][i] = 1
            
            # Support breakdown
            signals['support_breakdown'] = np.zeros(len(close))
            for i in range(len(close)):
                if not np.isnan(close[i]) and not np.isnan(s1[i]):
                    if close[i] < s1[i]:
                        signals['support_breakdown'][i] = 1
        
        return signals


class UnderlyingATREMACPREngine:
    """Main engine for multi-timeframe underlying ATR-EMA-CPR analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize analyzers
        self.atr_analyzer = UnderlyingATRAnalyzer(config)
        self.ema_analyzer = UnderlyingEMAAnalyzer(config)
        self.cpr_analyzer = UnderlyingCPRAnalyzer(config)

    async def analyze_underlying_atr_ema_cpr(self, underlying_data: UnderlyingPriceData) -> UnderlyingAnalysisResult:
        """Complete multi-timeframe ATR-EMA-CPR analysis for underlying prices"""
        
        start_time = time.time()
        
        try:
            # Run all analyses concurrently
            atr_task = self.atr_analyzer.analyze_underlying_atr(underlying_data)
            ema_task = self.ema_analyzer.analyze_underlying_ema(underlying_data)
            cpr_task = self.cpr_analyzer.analyze_underlying_cpr(underlying_data)
            
            atr_result, ema_result, cpr_result = await asyncio.gather(atr_task, ema_task, cpr_task)
            
            # Generate combined regime classification
            combined_regime = self._generate_combined_regime_classification(atr_result, ema_result, cpr_result)
            
            # Calculate cross-timeframe confidence
            cross_timeframe_confidence = self._calculate_cross_timeframe_confidence(atr_result, ema_result, cpr_result)
            
            # Generate feature vector (36 features for underlying analysis)
            feature_vector = self._generate_underlying_feature_vector(atr_result, ema_result, cpr_result)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return UnderlyingAnalysisResult(
                atr_result=atr_result,
                ema_result=ema_result,
                cpr_result=cpr_result,
                combined_regime_classification=combined_regime,
                cross_timeframe_confidence=cross_timeframe_confidence,
                feature_vector=feature_vector,
                processing_time_ms=processing_time_ms,
                metadata={
                    'engine': 'underlying_multi_timeframe_atr_ema_cpr',
                    'features_extracted': 36,
                    'timeframes': ['daily', 'weekly', 'monthly'],
                    'analysis_type': 'traditional_technical_analysis'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Underlying ATR-EMA-CPR analysis failed: {str(e)}")
            raise

    def _generate_combined_regime_classification(self, atr_result: UnderlyingATRResult, 
                                               ema_result: UnderlyingEMAResult, 
                                               cpr_result: UnderlyingCPRResult) -> np.ndarray:
        """Generate combined regime classification across all timeframes"""
        
        # Use daily timeframe as primary, with weekly/monthly confirmation
        daily_vol_regime = atr_result.volatility_regimes.get('daily', np.array([]))
        daily_trend = ema_result.trend_directions.get('daily', np.array([]))
        
        if len(daily_vol_regime) == 0 or len(daily_trend) == 0:
            return np.array([])
        
        regime = np.full(len(daily_vol_regime), 0)
        
        # Get cross-timeframe agreements for confirmation
        trend_agreement = ema_result.cross_timeframe_agreement.get('all_timeframes', np.ones(len(daily_trend)) * 0.5)
        atr_consistency = atr_result.cross_timeframe_consistency.get('all_timeframes', np.ones(len(daily_vol_regime)) * 0.5)
        
        for i in range(len(regime)):
            vol = daily_vol_regime[i]
            trend = daily_trend[i]
            agreement = trend_agreement[i]
            consistency = atr_consistency[i]
            
            # Base regime from volatility and trend
            base_regime = self._calculate_base_regime(vol, trend)
            
            # Adjust based on cross-timeframe agreement
            if agreement > 0.8 and consistency > 0.8:
                # Strong agreement across timeframes
                regime[i] = base_regime
            elif agreement < 0.3 or consistency < 0.3:
                # Poor agreement - transition regime
                regime[i] = 7  # Transition/uncertain
            else:
                # Moderate agreement
                regime[i] = base_regime
        
        return regime

    def _calculate_base_regime(self, vol_regime: int, trend_direction: int) -> int:
        """Calculate base regime from volatility and trend"""
        
        if vol_regime <= -1 and trend_direction >= 1:
            return 0  # Low vol, bullish
        elif vol_regime <= -1 and trend_direction <= -1:
            return 1  # Low vol, bearish  
        elif vol_regime == 0 and trend_direction >= 1:
            return 2  # Medium vol, bullish
        elif vol_regime == 0 and trend_direction <= -1:
            return 3  # Medium vol, bearish
        elif vol_regime >= 1 and trend_direction >= 1:
            return 4  # High vol, bullish
        elif vol_regime >= 1 and trend_direction <= -1:
            return 5  # High vol, bearish
        elif vol_regime >= 2:
            return 6  # Extreme volatility
        else:
            return 7  # Neutral/uncertain

    def _calculate_cross_timeframe_confidence(self, atr_result: UnderlyingATRResult, 
                                            ema_result: UnderlyingEMAResult, 
                                            cpr_result: UnderlyingCPRResult) -> np.ndarray:
        """Calculate confidence based on cross-timeframe analysis"""
        
        # Get agreement metrics
        trend_agreement = ema_result.cross_timeframe_agreement.get('all_timeframes', np.array([]))
        atr_consistency = atr_result.cross_timeframe_consistency.get('all_timeframes', np.array([]))
        
        if len(trend_agreement) == 0:
            return np.array([0.5])  # Default medium confidence
        
        # Combine different confidence factors
        base_confidence = (trend_agreement + atr_consistency) / 2
        
        # Adjust for level confluence if available
        level_confluence = cpr_result.level_confluence.get('daily_method_confluence', np.ones(len(base_confidence)) * 0.5)
        
        final_confidence = (base_confidence * 0.7 + level_confluence * 0.3)
        final_confidence = np.clip(final_confidence, 0.1, 0.95)
        
        return final_confidence

    def _generate_underlying_feature_vector(self, atr_result: UnderlyingATRResult, 
                                          ema_result: UnderlyingEMAResult, 
                                          cpr_result: UnderlyingCPRResult) -> np.ndarray:
        """Generate 36-feature vector for underlying analysis"""
        
        features = []
        
        # Determine data length from first available result
        data_length = 0
        if atr_result.daily_atr:
            first_atr_key = next(iter(atr_result.daily_atr.keys()))
            data_length = len(atr_result.daily_atr[first_atr_key])
        
        if data_length == 0:
            return np.array([]).reshape(0, 36)
        
        # Multi-timeframe ATR features (12 features)
        daily_atr_14 = atr_result.daily_atr.get('atr_14', np.zeros(data_length))
        weekly_atr_14 = atr_result.weekly_atr.get('atr_14', np.zeros(data_length))
        monthly_atr_14 = atr_result.monthly_atr.get('atr_14', np.zeros(data_length))
        
        features.extend([
            daily_atr_14, weekly_atr_14, monthly_atr_14,  # 3 features
            atr_result.volatility_regimes.get('daily', np.zeros(data_length)).astype(float),  # 1 feature  
            atr_result.volatility_regimes.get('weekly', np.zeros(data_length)).astype(float),  # 1 feature
            atr_result.volatility_regimes.get('monthly', np.zeros(data_length)).astype(float),  # 1 feature
            atr_result.cross_timeframe_consistency.get('weekly_daily', np.zeros(data_length)),  # 1 feature
            atr_result.cross_timeframe_consistency.get('monthly_daily', np.zeros(data_length)),  # 1 feature
            atr_result.cross_timeframe_consistency.get('all_timeframes', np.zeros(data_length)),  # 1 feature
            atr_result.atr_trends.get('daily', np.zeros(data_length)).astype(float),  # 1 feature
            atr_result.atr_trends.get('weekly', np.zeros(data_length)).astype(float),  # 1 feature
            atr_result.atr_trends.get('monthly', np.zeros(data_length)).astype(float)  # 1 feature
        ])
        
        # Multi-timeframe EMA features (12 features)
        daily_ema_20 = ema_result.daily_ema.get('ema_20', np.zeros(data_length))
        weekly_ema_10 = ema_result.weekly_ema.get('ema_10', np.zeros(data_length))
        monthly_ema_6 = ema_result.monthly_ema.get('ema_6', np.zeros(data_length))
        
        features.extend([
            daily_ema_20, weekly_ema_10, monthly_ema_6,  # 3 features
            ema_result.trend_directions.get('daily', np.zeros(data_length)).astype(float),  # 1 feature
            ema_result.trend_directions.get('weekly', np.zeros(data_length)).astype(float),  # 1 feature
            ema_result.trend_directions.get('monthly', np.zeros(data_length)).astype(float),  # 1 feature
            ema_result.trend_strengths.get('daily', np.zeros(data_length)),  # 1 feature
            ema_result.trend_strengths.get('weekly', np.zeros(data_length)),  # 1 feature
            ema_result.trend_strengths.get('monthly', np.zeros(data_length)),  # 1 feature
            ema_result.cross_timeframe_agreement.get('daily_weekly', np.zeros(data_length)),  # 1 feature
            ema_result.cross_timeframe_agreement.get('daily_monthly', np.zeros(data_length)),  # 1 feature
            ema_result.cross_timeframe_agreement.get('all_timeframes', np.zeros(data_length))  # 1 feature
        ])
        
        # Multi-timeframe CPR features (12 features)
        daily_pivot = cpr_result.daily_cpr.get('standard', {}).get('pivot', np.zeros(data_length))
        weekly_pivot = cpr_result.weekly_cpr.get('standard', {}).get('pivot', np.zeros(data_length))
        monthly_pivot = cpr_result.monthly_cpr.get('standard', {}).get('pivot', np.zeros(data_length))
        
        features.extend([
            daily_pivot, weekly_pivot, monthly_pivot,  # 3 features
            cpr_result.daily_cpr.get('standard', {}).get('r1', np.zeros(data_length)),  # 1 feature
            cpr_result.daily_cpr.get('standard', {}).get('s1', np.zeros(data_length)),  # 1 feature
            cpr_result.weekly_cpr.get('standard', {}).get('r1', np.zeros(data_length)),  # 1 feature
            cpr_result.weekly_cpr.get('standard', {}).get('s1', np.zeros(data_length)),  # 1 feature
            cpr_result.support_resistance_validation.get('pivot_agreement', np.zeros(data_length)),  # 1 feature
            cpr_result.support_resistance_validation.get('support_strength', np.zeros(data_length)),  # 1 feature
            cpr_result.support_resistance_validation.get('resistance_strength', np.zeros(data_length)),  # 1 feature
            cpr_result.breakout_signals.get('daily', {}).get('resistance_breakout', np.zeros(data_length)),  # 1 feature
            cpr_result.breakout_signals.get('daily', {}).get('support_breakdown', np.zeros(data_length))  # 1 feature
        ])
        
        # Stack features and transpose
        feature_matrix = np.column_stack(features)
        return feature_matrix