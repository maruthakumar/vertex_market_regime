"""
Rolling Straddle EMA Analysis for Component 1

Revolutionary approach applying Exponential Moving Average (EMA) analysis to 
ROLLING STRADDLE PRICES instead of underlying prices. Implements 4 EMA periods 
(20, 50, 100, 200) with EMA alignment scoring, confluence zone detection, 
and EMA continuity across missing data points.

This is the key innovation: traditional technical indicators applied to the
time-series evolution of rolling straddle prices as strikes dynamically 
adjust minute-by-minute.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

# GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class EMAConfiguration:
    """EMA analysis configuration"""
    short_period: int = 20      # Short EMA (30% weight)
    medium_period: int = 50     # Medium EMA (30% weight)  
    long_period: int = 100      # Long EMA (25% weight)
    trend_filter_period: int = 200  # Trend filter EMA (15% weight)
    
    # Weights for different EMA periods
    short_weight: float = 0.30
    medium_weight: float = 0.30
    long_weight: float = 0.25
    trend_filter_weight: float = 0.15


@dataclass
class EMAResult:
    """Result from EMA calculation for single straddle type"""
    straddle_type: str
    ema_short: np.ndarray
    ema_medium: np.ndarray
    ema_long: np.ndarray
    ema_trend_filter: np.ndarray
    alignment_score: float
    confluence_zones: List[Tuple[int, int]]  # Start/end indices
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0.0 to 1.0


@dataclass
class RollingStraddleEMAAnalysis:
    """Complete EMA analysis result for all rolling straddle types"""
    atm_ema: EMAResult
    itm1_ema: EMAResult
    otm1_ema: EMAResult
    
    # Cross-straddle analysis
    overall_alignment_score: float
    dominant_trend: str
    trend_consistency: float
    confluence_strength: float
    
    # Processing metadata
    processing_time_ms: float
    data_points_processed: int
    missing_data_handled: int
    metadata: Dict[str, Any]


class RollingStraddleEMAEngine:
    """
    Revolutionary EMA Engine for Rolling Straddle Prices
    
    Applies traditional EMA technical analysis to the evolution of rolling
    straddle prices over time. As strikes "roll" minute-by-minute with spot
    movement, the straddle prices create a unique time series perfect for
    EMA analysis.
    
    Key Features:
    - 4 EMA periods with weighted scoring
    - EMA alignment scoring (-1.0 to +1.0)
    - Confluence zone detection
    - Missing data continuity handling
    - Cross-straddle trend analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Rolling Straddle EMA Engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance configuration
        self.processing_budget_ms = config.get('processing_budget_ms', 40)
        self.use_gpu = config.get('use_gpu', GPU_AVAILABLE)
        
        # EMA configuration
        self.ema_config = EMAConfiguration(
            short_period=config.get('ema_short_period', 20),
            medium_period=config.get('ema_medium_period', 50),
            long_period=config.get('ema_long_period', 100),
            trend_filter_period=config.get('ema_trend_filter_period', 200),
            short_weight=config.get('ema_short_weight', 0.30),
            medium_weight=config.get('ema_medium_weight', 0.30),
            long_weight=config.get('ema_long_weight', 0.25),
            trend_filter_weight=config.get('ema_trend_filter_weight', 0.15)
        )
        
        # Analysis configuration
        self.alignment_threshold = config.get('alignment_threshold', 0.02)  # 2% threshold
        self.confluence_min_period = config.get('confluence_min_period', 10)  # Min 10 minutes
        self.trend_change_threshold = config.get('trend_change_threshold', 0.05)  # 5%
        
        # Missing data handling
        self.handle_missing_data = config.get('handle_missing_data', True)
        self.max_interpolate_gaps = config.get('max_interpolate_gaps', 5)
        
        self.logger.info("RollingStraddleEMAEngine initialized with 4 EMA periods")
    
    async def analyze_rolling_straddle_emas(self, 
                                          straddle_time_series: Dict[str, np.ndarray]) -> RollingStraddleEMAAnalysis:
        """
        Analyze EMA for all rolling straddle types
        
        Args:
            straddle_time_series: Dictionary with straddle time series
                - 'atm_straddle': ATM straddle price series
                - 'itm1_straddle': ITM1 straddle price series  
                - 'otm1_straddle': OTM1 straddle price series
                
        Returns:
            RollingStraddleEMAAnalysis with complete EMA analysis
        """
        start_time = time.time()
        
        try:
            # Validate input data
            required_series = ['atm_straddle', 'itm1_straddle', 'otm1_straddle']
            for series_name in required_series:
                if series_name not in straddle_time_series:
                    raise ValueError(f"Missing required straddle series: {series_name}")
            
            # Handle missing data preprocessing
            cleaned_series = {}
            missing_data_count = 0
            
            for series_name in required_series:
                original_series = straddle_time_series[series_name]
                cleaned_series[series_name], missing_count = await self._handle_missing_data(original_series)
                missing_data_count += missing_count
            
            # Calculate EMA for each straddle type
            atm_ema = await self._calculate_ema_for_straddle(cleaned_series['atm_straddle'], 'ATM')
            itm1_ema = await self._calculate_ema_for_straddle(cleaned_series['itm1_straddle'], 'ITM1')
            otm1_ema = await self._calculate_ema_for_straddle(cleaned_series['otm1_straddle'], 'OTM1')
            
            # Cross-straddle analysis
            cross_analysis = await self._analyze_cross_straddle_trends([atm_ema, itm1_ema, otm1_ema])
            
            processing_time = (time.time() - start_time) * 1000
            
            # Performance validation
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"EMA processing {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            return RollingStraddleEMAAnalysis(
                atm_ema=atm_ema,
                itm1_ema=itm1_ema,
                otm1_ema=otm1_ema,
                overall_alignment_score=cross_analysis['overall_alignment'],
                dominant_trend=cross_analysis['dominant_trend'],
                trend_consistency=cross_analysis['trend_consistency'],
                confluence_strength=cross_analysis['confluence_strength'],
                processing_time_ms=processing_time,
                data_points_processed=len(cleaned_series['atm_straddle']),
                missing_data_handled=missing_data_count,
                metadata={
                    'ema_config': self.ema_config.__dict__,
                    'alignment_threshold': self.alignment_threshold,
                    'confluence_min_period': self.confluence_min_period,
                    'gpu_acceleration': self.use_gpu
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze rolling straddle EMAs: {e}")
            raise
    
    async def _handle_missing_data(self, series: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Handle missing data in straddle price series
        
        Args:
            series: Raw straddle price series
            
        Returns:
            Tuple of (cleaned_series, missing_data_count)
        """
        if not self.handle_missing_data:
            return series, 0
        
        # Identify missing/invalid data
        valid_mask = ~(np.isnan(series) | np.isinf(series) | (series <= 0))
        missing_count = len(series) - np.sum(valid_mask)
        
        if missing_count == 0:
            return series, 0
        
        # Create cleaned series
        cleaned_series = series.copy()
        
        # Forward fill short gaps
        if missing_count > 0:
            # Convert to pandas for gap filling
            temp_series = pd.Series(cleaned_series)
            
            # Forward fill with limit
            temp_series = temp_series.fillna(method='ffill', limit=self.max_interpolate_gaps)
            
            # Backward fill remaining
            temp_series = temp_series.fillna(method='bfill', limit=self.max_interpolate_gaps)
            
            # Linear interpolation for any remaining gaps
            temp_series = temp_series.interpolate(method='linear', limit=self.max_interpolate_gaps)
            
            # Final forward fill for edges
            temp_series = temp_series.fillna(method='ffill')
            temp_series = temp_series.fillna(method='bfill')
            
            cleaned_series = temp_series.values
        
        return cleaned_series, missing_count
    
    async def _calculate_ema_for_straddle(self, straddle_series: np.ndarray, straddle_type: str) -> EMAResult:
        """
        Calculate EMA analysis for single straddle type
        
        Args:
            straddle_series: Straddle price time series
            straddle_type: Type of straddle ('ATM', 'ITM1', 'OTM1')
            
        Returns:
            EMAResult with complete EMA analysis
        """
        try:
            # Calculate EMAs using optimized approach
            ema_short = self._calculate_ema(straddle_series, self.ema_config.short_period)
            ema_medium = self._calculate_ema(straddle_series, self.ema_config.medium_period)
            ema_long = self._calculate_ema(straddle_series, self.ema_config.long_period)
            ema_trend_filter = self._calculate_ema(straddle_series, self.ema_config.trend_filter_period)
            
            # Calculate alignment score
            alignment_score = self._calculate_ema_alignment_score(
                ema_short, ema_medium, ema_long, ema_trend_filter
            )
            
            # Detect confluence zones
            confluence_zones = self._detect_confluence_zones(
                ema_short, ema_medium, ema_long, ema_trend_filter
            )
            
            # Determine trend direction and strength
            trend_direction, trend_strength = self._analyze_trend(
                straddle_series, ema_short, ema_medium, ema_long
            )
            
            return EMAResult(
                straddle_type=straddle_type,
                ema_short=ema_short,
                ema_medium=ema_medium,
                ema_long=ema_long,
                ema_trend_filter=ema_trend_filter,
                alignment_score=alignment_score,
                confluence_zones=confluence_zones,
                trend_direction=trend_direction,
                trend_strength=trend_strength
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate EMA for {straddle_type}: {e}")
            # Return empty result
            zeros = np.zeros(len(straddle_series))
            return EMAResult(
                straddle_type=straddle_type,
                ema_short=zeros,
                ema_medium=zeros,
                ema_long=zeros,
                ema_trend_filter=zeros,
                alignment_score=0.0,
                confluence_zones=[],
                trend_direction='neutral',
                trend_strength=0.0
            )
    
    def _calculate_ema(self, series: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average
        
        Args:
            series: Price series
            period: EMA period
            
        Returns:
            EMA array
        """
        try:
            if len(series) < period:
                # Not enough data, return simple moving average
                return np.full_like(series, np.mean(series))
            
            # Calculate EMA using pandas for efficiency
            df = pd.Series(series)
            ema = df.ewm(span=period, adjust=False).mean()
            return ema.values
            
        except Exception as e:
            self.logger.debug(f"EMA calculation failed for period {period}: {e}")
            return np.full_like(series, np.mean(series))
    
    def _calculate_ema_alignment_score(self, 
                                     ema_short: np.ndarray, 
                                     ema_medium: np.ndarray, 
                                     ema_long: np.ndarray, 
                                     ema_trend_filter: np.ndarray) -> float:
        """
        Calculate EMA alignment score (-1.0 to +1.0)
        
        Perfect bullish alignment: short > medium > long > trend_filter
        Perfect bearish alignment: short < medium < long < trend_filter
        
        Args:
            ema_short: Short EMA series
            ema_medium: Medium EMA series
            ema_long: Long EMA series
            ema_trend_filter: Trend filter EMA series
            
        Returns:
            Alignment score (-1.0 to +1.0)
        """
        try:
            # Use recent data for alignment calculation
            recent_points = min(50, len(ema_short))
            
            if recent_points < 10:
                return 0.0
            
            # Get recent values
            short_recent = ema_short[-recent_points:]
            medium_recent = ema_medium[-recent_points:]
            long_recent = ema_long[-recent_points:]
            trend_recent = ema_trend_filter[-recent_points:]
            
            # Calculate alignment percentages
            bullish_alignment = np.mean(
                (short_recent > medium_recent) & 
                (medium_recent > long_recent) & 
                (long_recent > trend_recent)
            )
            
            bearish_alignment = np.mean(
                (short_recent < medium_recent) & 
                (medium_recent < long_recent) & 
                (long_recent < trend_recent)
            )
            
            # Calculate net alignment score
            net_alignment = bullish_alignment - bearish_alignment
            
            # Apply weighting
            weighted_score = (
                net_alignment * self.ema_config.short_weight +
                net_alignment * self.ema_config.medium_weight +
                net_alignment * self.ema_config.long_weight +
                net_alignment * self.ema_config.trend_filter_weight
            )
            
            return float(np.clip(weighted_score, -1.0, 1.0))
            
        except Exception as e:
            self.logger.debug(f"Alignment score calculation failed: {e}")
            return 0.0
    
    def _detect_confluence_zones(self, 
                               ema_short: np.ndarray, 
                               ema_medium: np.ndarray, 
                               ema_long: np.ndarray, 
                               ema_trend_filter: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect EMA confluence zones where EMAs converge
        
        Args:
            ema_short: Short EMA series
            ema_medium: Medium EMA series
            ema_long: Long EMA series
            ema_trend_filter: Trend filter EMA series
            
        Returns:
            List of confluence zone (start_index, end_index) tuples
        """
        try:
            confluence_zones = []
            
            if len(ema_short) < self.confluence_min_period:
                return confluence_zones
            
            # Calculate EMA spreads (normalized by price level)
            price_level = np.maximum(ema_medium, 1.0)  # Avoid division by zero
            
            spread_sm = np.abs(ema_short - ema_medium) / price_level
            spread_ml = np.abs(ema_medium - ema_long) / price_level
            spread_lt = np.abs(ema_long - ema_trend_filter) / price_level
            
            # Confluence occurs when all spreads are below threshold
            confluence_mask = (
                (spread_sm < self.alignment_threshold) &
                (spread_ml < self.alignment_threshold) &
                (spread_lt < self.alignment_threshold)
            )
            
            # Find continuous confluence periods
            confluence_changes = np.diff(confluence_mask.astype(int))
            starts = np.where(confluence_changes == 1)[0] + 1
            ends = np.where(confluence_changes == -1)[0] + 1
            
            # Handle edge cases
            if len(starts) > 0 and len(ends) > 0:
                # Adjust for unmatched starts/ends
                if len(starts) > len(ends):
                    ends = np.append(ends, len(confluence_mask) - 1)
                elif len(ends) > len(starts):
                    starts = np.insert(starts, 0, 0)
            
            # Filter minimum duration zones
            for start, end in zip(starts, ends):
                if end - start >= self.confluence_min_period:
                    confluence_zones.append((int(start), int(end)))
            
            return confluence_zones
            
        except Exception as e:
            self.logger.debug(f"Confluence zone detection failed: {e}")
            return []
    
    def _analyze_trend(self, 
                      original_series: np.ndarray, 
                      ema_short: np.ndarray, 
                      ema_medium: np.ndarray, 
                      ema_long: np.ndarray) -> Tuple[str, float]:
        """
        Analyze trend direction and strength
        
        Args:
            original_series: Original straddle price series
            ema_short: Short EMA
            ema_medium: Medium EMA
            ema_long: Long EMA
            
        Returns:
            Tuple of (trend_direction, trend_strength)
        """
        try:
            recent_points = min(20, len(original_series))
            if recent_points < 5:
                return 'neutral', 0.0
            
            # Current levels
            current_price = original_series[-1]
            current_short = ema_short[-1]
            current_medium = ema_medium[-1]
            current_long = ema_long[-1]
            
            # Trend analysis
            price_above_short = current_price > current_short
            short_above_medium = current_short > current_medium
            medium_above_long = current_medium > current_long
            
            price_below_short = current_price < current_short
            short_below_medium = current_short < current_medium
            medium_below_long = current_medium < current_long
            
            # Score trend strength
            bullish_signals = sum([price_above_short, short_above_medium, medium_above_long])
            bearish_signals = sum([price_below_short, short_below_medium, medium_below_long])
            
            # Calculate trend changes
            short_slope = self._calculate_slope(ema_short[-recent_points:])
            medium_slope = self._calculate_slope(ema_medium[-recent_points:])
            
            # Determine trend
            if bullish_signals >= 2 and short_slope > 0:
                trend_direction = 'bullish'
                trend_strength = (bullish_signals / 3.0) * min(abs(short_slope), 1.0)
            elif bearish_signals >= 2 and short_slope < 0:
                trend_direction = 'bearish'
                trend_strength = (bearish_signals / 3.0) * min(abs(short_slope), 1.0)
            else:
                trend_direction = 'neutral'
                trend_strength = 0.5 - abs(bullish_signals - bearish_signals) / 6.0
            
            return trend_direction, float(np.clip(trend_strength, 0.0, 1.0))
            
        except Exception as e:
            self.logger.debug(f"Trend analysis failed: {e}")
            return 'neutral', 0.0
    
    def _calculate_slope(self, series: np.ndarray) -> float:
        """
        Calculate slope of series using linear regression
        
        Args:
            series: Time series data
            
        Returns:
            Slope value
        """
        try:
            if len(series) < 2:
                return 0.0
            
            x = np.arange(len(series))
            slope, _, _, _, _ = stats.linregress(x, series)
            return float(slope)
            
        except:
            return 0.0
    
    async def _analyze_cross_straddle_trends(self, ema_results: List[EMAResult]) -> Dict[str, Any]:
        """
        Analyze trends across all straddle types
        
        Args:
            ema_results: List of EMA results for all straddle types
            
        Returns:
            Cross-straddle analysis dictionary
        """
        try:
            # Calculate overall alignment
            alignment_scores = [result.alignment_score for result in ema_results]
            overall_alignment = float(np.mean(alignment_scores))
            
            # Determine dominant trend
            trend_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            trend_strengths = []
            
            for result in ema_results:
                trend_votes[result.trend_direction] += 1
                trend_strengths.append(result.trend_strength)
            
            dominant_trend = max(trend_votes, key=trend_votes.get)
            
            # Calculate trend consistency
            max_votes = max(trend_votes.values())
            trend_consistency = float(max_votes / len(ema_results))
            
            # Calculate confluence strength
            total_confluence_zones = sum(len(result.confluence_zones) for result in ema_results)
            confluence_strength = float(min(total_confluence_zones / 10.0, 1.0))  # Normalize to [0,1]
            
            return {
                'overall_alignment': overall_alignment,
                'dominant_trend': dominant_trend,
                'trend_consistency': trend_consistency,
                'confluence_strength': confluence_strength,
                'individual_strengths': trend_strengths
            }
            
        except Exception as e:
            self.logger.error(f"Cross-straddle analysis failed: {e}")
            return {
                'overall_alignment': 0.0,
                'dominant_trend': 'neutral',
                'trend_consistency': 0.0,
                'confluence_strength': 0.0,
                'individual_strengths': [0.0, 0.0, 0.0]
            }
    
    async def get_ema_feature_vector(self, analysis: RollingStraddleEMAAnalysis) -> Dict[str, float]:
        """
        Extract feature vector from EMA analysis
        
        Args:
            analysis: Complete EMA analysis result
            
        Returns:
            Feature vector dictionary
        """
        return {
            # Individual EMA alignment scores
            'atm_ema_alignment': analysis.atm_ema.alignment_score,
            'itm1_ema_alignment': analysis.itm1_ema.alignment_score,
            'otm1_ema_alignment': analysis.otm1_ema.alignment_score,
            
            # Individual trend strengths
            'atm_trend_strength': analysis.atm_ema.trend_strength,
            'itm1_trend_strength': analysis.itm1_ema.trend_strength,
            'otm1_trend_strength': analysis.otm1_ema.trend_strength,
            
            # Cross-straddle features
            'overall_ema_alignment': analysis.overall_alignment_score,
            'trend_consistency': analysis.trend_consistency,
            'confluence_strength': analysis.confluence_strength,
            
            # Trend direction encoding
            'dominant_trend_bullish': 1.0 if analysis.dominant_trend == 'bullish' else 0.0,
            'dominant_trend_bearish': 1.0 if analysis.dominant_trend == 'bearish' else 0.0,
            'dominant_trend_neutral': 1.0 if analysis.dominant_trend == 'neutral' else 0.0
        }


# Factory function
def create_ema_engine(config: Dict[str, Any]) -> RollingStraddleEMAEngine:
    """Create and configure RollingStraddleEMAEngine instance"""
    return RollingStraddleEMAEngine(config)