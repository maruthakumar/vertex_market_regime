"""
Rolling Straddle VWAP Analysis for Component 1

Revolutionary VWAP (Volume Weighted Average Price) analysis applied to rolling
straddle prices using combined volume (ce_volume + pe_volume). Implements current
day and previous day rolling straddle VWAP, underlying futures VWAP for regime
context, VWAP deviation scoring, and 5-level standard deviation bands.

Key Innovation: VWAP calculated on rolling straddle prices, not underlying prices,
using combined options volume for true options flow analysis.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
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
class VWAPConfiguration:
    """VWAP analysis configuration"""
    # Rolling straddle VWAP weights
    current_day_weight: float = 0.50    # Current day weight
    previous_day_weight: float = 0.50   # Previous day weight
    
    # Underlying VWAP weights (for regime context)
    underlying_today_weight: float = 0.40
    underlying_previous_weight: float = 0.40
    underlying_weekly_weight: float = 0.20
    
    # Standard deviation bands
    std_bands: List[float] = None  # Will be set to [0.5, 1.0, 1.5, 2.0, 2.5]
    
    def __post_init__(self):
        if self.std_bands is None:
            self.std_bands = [0.5, 1.0, 1.5, 2.0, 2.5]


@dataclass
class VWAPResult:
    """VWAP result for single straddle type"""
    straddle_type: str
    vwap_current: np.ndarray
    vwap_previous: np.ndarray
    vwap_combined: np.ndarray
    deviation_scores: np.ndarray
    std_bands: Dict[str, np.ndarray]  # Band level -> band values
    volume_profile: np.ndarray
    price_position: str  # 'above_vwap', 'below_vwap', 'at_vwap'
    deviation_strength: float  # How far from VWAP (normalized)


@dataclass
class UnderlyingVWAPContext:
    """Underlying futures VWAP for regime context"""
    futures_vwap_today: np.ndarray
    futures_vwap_previous: np.ndarray
    futures_vwap_weekly: np.ndarray
    futures_vwap_combined: np.ndarray
    regime_context: str  # 'bullish_regime', 'bearish_regime', 'neutral_regime'
    regime_strength: float


@dataclass
class RollingStraddleVWAPAnalysis:
    """Complete VWAP analysis for all rolling straddle types"""
    atm_vwap: VWAPResult
    itm1_vwap: VWAPResult
    otm1_vwap: VWAPResult
    underlying_context: UnderlyingVWAPContext
    
    # Cross-straddle VWAP analysis
    overall_deviation_score: float
    volume_concentration: str  # 'atm_heavy', 'itm_heavy', 'otm_heavy', 'balanced'
    vwap_trend_alignment: float  # -1.0 to +1.0
    zero_volume_periods: int
    
    # Processing metadata
    processing_time_ms: float
    data_points_processed: int
    volume_outliers_handled: int
    metadata: Dict[str, Any]


class RollingStraddleVWAPEngine:
    """
    Revolutionary VWAP Engine for Rolling Straddle Prices
    
    Calculates VWAP on rolling straddle prices using combined options volume
    (ce_volume + pe_volume), providing unique insights into options flow 
    dynamics and volume-price relationships in the options market.
    
    Key Features:
    - Rolling straddle VWAP with current/previous day weighting
    - Combined volume calculation (CE + PE volumes)
    - 5-level standard deviation bands
    - Underlying futures VWAP for regime context
    - Zero-volume period handling
    - Volume outlier detection and normalization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Rolling Straddle VWAP Engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance configuration
        self.processing_budget_ms = config.get('processing_budget_ms', 35)
        self.use_gpu = config.get('use_gpu', GPU_AVAILABLE)
        
        # VWAP configuration
        self.vwap_config = VWAPConfiguration(
            current_day_weight=config.get('vwap_current_day_weight', 0.50),
            previous_day_weight=config.get('vwap_previous_day_weight', 0.50),
            underlying_today_weight=config.get('underlying_today_weight', 0.40),
            underlying_previous_weight=config.get('underlying_previous_weight', 0.40),
            underlying_weekly_weight=config.get('underlying_weekly_weight', 0.20),
            std_bands=config.get('vwap_std_bands', [0.5, 1.0, 1.5, 2.0, 2.5])
        )
        
        # Volume handling configuration
        self.min_volume_threshold = config.get('min_volume_threshold', 1)
        self.volume_outlier_threshold = config.get('volume_outlier_threshold', 3.0)  # 3 std devs
        self.zero_volume_fill_method = config.get('zero_volume_fill_method', 'forward_fill')
        
        # Session configuration (for day separation)
        self.session_start_hour = config.get('session_start_hour', 9)
        self.session_end_hour = config.get('session_end_hour', 15)
        
        self.logger.info("RollingStraddleVWAPEngine initialized")
    
    async def analyze_rolling_straddle_vwap(self, 
                                          straddle_data: Dict[str, np.ndarray],
                                          volume_data: Dict[str, np.ndarray],
                                          futures_data: Optional[Dict[str, np.ndarray]] = None,
                                          timestamps: Optional[List[str]] = None) -> RollingStraddleVWAPAnalysis:
        """
        Analyze VWAP for all rolling straddle types
        
        Args:
            straddle_data: Dictionary with straddle price series
            volume_data: Dictionary with volume series
            futures_data: Optional futures price/volume data for context
            timestamps: Optional timestamp series for session separation
            
        Returns:
            RollingStraddleVWAPAnalysis with complete VWAP analysis
        """
        start_time = time.time()
        
        try:
            # Validate input data
            required_straddles = ['atm_straddle', 'itm1_straddle', 'otm1_straddle']
            for straddle in required_straddles:
                if straddle not in straddle_data:
                    raise ValueError(f"Missing straddle data: {straddle}")
            
            # Prepare volume data with outlier handling
            cleaned_volume_data, outlier_count = await self._handle_volume_outliers(volume_data)
            
            # Separate sessions if timestamps provided
            session_data = await self._separate_trading_sessions(
                straddle_data, cleaned_volume_data, timestamps
            )
            
            # Calculate VWAP for each straddle type
            atm_vwap = await self._calculate_vwap_for_straddle(
                session_data['straddle_data']['atm_straddle'],
                session_data['volume_data'].get('atm_volume', cleaned_volume_data.get('combined_volume')),
                'ATM'
            )
            
            itm1_vwap = await self._calculate_vwap_for_straddle(
                session_data['straddle_data']['itm1_straddle'],
                session_data['volume_data'].get('itm1_volume', cleaned_volume_data.get('combined_volume')),
                'ITM1'
            )
            
            otm1_vwap = await self._calculate_vwap_for_straddle(
                session_data['straddle_data']['otm1_straddle'],
                session_data['volume_data'].get('otm1_volume', cleaned_volume_data.get('combined_volume')),
                'OTM1'
            )
            
            # Calculate underlying futures VWAP context
            underlying_context = await self._calculate_underlying_vwap_context(futures_data, timestamps)
            
            # Cross-straddle analysis
            cross_analysis = await self._analyze_cross_straddle_vwap(
                [atm_vwap, itm1_vwap, otm1_vwap], cleaned_volume_data
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Performance validation
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"VWAP processing {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            return RollingStraddleVWAPAnalysis(
                atm_vwap=atm_vwap,
                itm1_vwap=itm1_vwap,
                otm1_vwap=otm1_vwap,
                underlying_context=underlying_context,
                overall_deviation_score=cross_analysis['overall_deviation'],
                volume_concentration=cross_analysis['volume_concentration'],
                vwap_trend_alignment=cross_analysis['trend_alignment'],
                zero_volume_periods=cross_analysis['zero_volume_periods'],
                processing_time_ms=processing_time,
                data_points_processed=len(straddle_data['atm_straddle']),
                volume_outliers_handled=outlier_count,
                metadata={
                    'vwap_config': self.vwap_config.__dict__,
                    'session_separation_enabled': timestamps is not None,
                    'futures_context_available': futures_data is not None,
                    'min_volume_threshold': self.min_volume_threshold
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze rolling straddle VWAP: {e}")
            raise
    
    async def _handle_volume_outliers(self, volume_data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Handle volume outliers and zero-volume periods
        
        Args:
            volume_data: Raw volume data dictionary
            
        Returns:
            Tuple of (cleaned_volume_data, outlier_count)
        """
        cleaned_data = {}
        total_outliers = 0
        
        for volume_key, volume_series in volume_data.items():
            # Handle zero volumes
            cleaned_series = np.maximum(volume_series, self.min_volume_threshold)
            
            # Detect outliers using z-score
            if len(volume_series) > 10:
                z_scores = np.abs(stats.zscore(volume_series))
                outlier_mask = z_scores > self.volume_outlier_threshold
                outlier_count = np.sum(outlier_mask)
                
                if outlier_count > 0:
                    # Replace outliers with median
                    median_volume = np.median(volume_series[~outlier_mask])
                    cleaned_series[outlier_mask] = median_volume
                    total_outliers += outlier_count
            
            cleaned_data[volume_key] = cleaned_series
        
        return cleaned_data, total_outliers
    
    async def _separate_trading_sessions(self, 
                                       straddle_data: Dict[str, np.ndarray],
                                       volume_data: Dict[str, np.ndarray],
                                       timestamps: Optional[List[str]]) -> Dict[str, Any]:
        """
        Separate data into trading sessions for current/previous day analysis
        
        Args:
            straddle_data: Straddle price data
            volume_data: Volume data
            timestamps: Optional timestamp series
            
        Returns:
            Dictionary with session-separated data
        """
        if timestamps is None:
            # No timestamp separation, treat all as current session
            return {
                'straddle_data': straddle_data,
                'volume_data': volume_data,
                'current_session_mask': np.ones(len(next(iter(straddle_data.values()))), dtype=bool),
                'previous_session_mask': np.zeros(len(next(iter(straddle_data.values()))), dtype=bool)
            }
        
        try:
            # Convert timestamps to datetime
            dt_timestamps = pd.to_datetime(timestamps)
            
            # Identify current and previous trading days
            unique_dates = dt_timestamps.date
            current_date = max(unique_dates)
            
            # Find previous trading day (skip weekends)
            previous_date = current_date - timedelta(days=1)
            while previous_date.weekday() > 4:  # Skip weekends
                previous_date -= timedelta(days=1)
            
            # Create session masks
            current_mask = dt_timestamps.date == current_date
            previous_mask = dt_timestamps.date == previous_date
            
            return {
                'straddle_data': straddle_data,
                'volume_data': volume_data,
                'current_session_mask': current_mask.values,
                'previous_session_mask': previous_mask.values,
                'current_date': current_date,
                'previous_date': previous_date
            }
            
        except Exception as e:
            self.logger.warning(f"Session separation failed: {e}")
            # Fallback to no separation
            return {
                'straddle_data': straddle_data,
                'volume_data': volume_data,
                'current_session_mask': np.ones(len(next(iter(straddle_data.values()))), dtype=bool),
                'previous_session_mask': np.zeros(len(next(iter(straddle_data.values()))), dtype=bool)
            }
    
    async def _calculate_vwap_for_straddle(self, 
                                         straddle_prices: np.ndarray,
                                         volumes: np.ndarray,
                                         straddle_type: str) -> VWAPResult:
        """
        Calculate VWAP for single straddle type
        
        Args:
            straddle_prices: Straddle price series
            volumes: Combined volume series (ce_volume + pe_volume)
            straddle_type: Type of straddle ('ATM', 'ITM1', 'OTM1')
            
        Returns:
            VWAPResult with complete VWAP analysis
        """
        try:
            # Calculate running VWAP
            vwap_current = self._calculate_vwap(straddle_prices, volumes)
            
            # Calculate previous day VWAP (simplified - use earlier period)
            half_point = len(straddle_prices) // 2
            if half_point > 10:
                vwap_previous = self._calculate_vwap(
                    straddle_prices[:half_point], 
                    volumes[:half_point]
                )
                # Extend to full length
                vwap_previous = np.full_like(straddle_prices, vwap_previous[-1])
            else:
                vwap_previous = vwap_current.copy()
            
            # Combined VWAP
            vwap_combined = (
                vwap_current * self.vwap_config.current_day_weight +
                vwap_previous * self.vwap_config.previous_day_weight
            )
            
            # Calculate deviation scores
            deviation_scores = self._calculate_deviation_scores(straddle_prices, vwap_combined)
            
            # Calculate standard deviation bands
            std_bands = self._calculate_std_bands(straddle_prices, vwap_combined)
            
            # Analyze price position
            price_position = self._analyze_price_position(straddle_prices, vwap_combined)
            
            # Calculate deviation strength
            deviation_strength = self._calculate_deviation_strength(deviation_scores)
            
            return VWAPResult(
                straddle_type=straddle_type,
                vwap_current=vwap_current,
                vwap_previous=vwap_previous,
                vwap_combined=vwap_combined,
                deviation_scores=deviation_scores,
                std_bands=std_bands,
                volume_profile=volumes,
                price_position=price_position,
                deviation_strength=deviation_strength
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate VWAP for {straddle_type}: {e}")
            # Return empty result
            zeros = np.zeros(len(straddle_prices))
            return VWAPResult(
                straddle_type=straddle_type,
                vwap_current=zeros,
                vwap_previous=zeros,
                vwap_combined=zeros,
                deviation_scores=zeros,
                std_bands={},
                volume_profile=volumes,
                price_position='at_vwap',
                deviation_strength=0.0
            )
    
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            VWAP series
        """
        try:
            # Ensure volumes are positive
            volumes = np.maximum(volumes, self.min_volume_threshold)
            
            # Calculate cumulative price*volume and cumulative volume
            pv_cumsum = np.cumsum(prices * volumes)
            volume_cumsum = np.cumsum(volumes)
            
            # Calculate VWAP
            vwap = pv_cumsum / volume_cumsum
            
            # Handle edge cases
            vwap = np.nan_to_num(vwap, nan=prices[0] if len(prices) > 0 else 0.0)
            
            return vwap
            
        except Exception as e:
            self.logger.debug(f"VWAP calculation failed: {e}")
            return np.full_like(prices, np.mean(prices) if len(prices) > 0 else 0.0)
    
    def _calculate_deviation_scores(self, prices: np.ndarray, vwap: np.ndarray) -> np.ndarray:
        """
        Calculate VWAP deviation scores
        
        Args:
            prices: Current prices
            vwap: VWAP series
            
        Returns:
            Deviation scores (normalized)
        """
        try:
            # Calculate percentage deviation from VWAP
            deviation = (prices - vwap) / np.maximum(vwap, 1e-6)
            
            # Normalize to [-1, 1] range
            max_deviation = np.percentile(np.abs(deviation), 95)  # 95th percentile
            if max_deviation > 0:
                normalized_deviation = deviation / max_deviation
                normalized_deviation = np.clip(normalized_deviation, -1.0, 1.0)
            else:
                normalized_deviation = np.zeros_like(deviation)
            
            return normalized_deviation
            
        except:
            return np.zeros_like(prices)
    
    def _calculate_std_bands(self, prices: np.ndarray, vwap: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate standard deviation bands around VWAP
        
        Args:
            prices: Price series
            vwap: VWAP series
            
        Returns:
            Dictionary of standard deviation bands
        """
        try:
            # Calculate rolling standard deviation of price deviations
            deviations = prices - vwap
            
            # Use rolling window for std calculation
            window_size = min(50, len(prices) // 4)  # Dynamic window size
            if window_size < 10:
                window_size = len(prices)
            
            # Calculate standard deviation
            if len(deviations) >= window_size:
                rolling_std = pd.Series(deviations).rolling(window_size).std().fillna(method='bfill')
                std_values = rolling_std.values
            else:
                std_values = np.full_like(deviations, np.std(deviations))
            
            # Create bands
            bands = {}
            for multiplier in self.vwap_config.std_bands:
                bands[f'std_{multiplier}'] = vwap + (std_values * multiplier)
                bands[f'std_neg_{multiplier}'] = vwap - (std_values * multiplier)
            
            return bands
            
        except Exception as e:
            self.logger.debug(f"Standard deviation bands calculation failed: {e}")
            return {}
    
    def _analyze_price_position(self, prices: np.ndarray, vwap: np.ndarray) -> str:
        """
        Analyze current price position relative to VWAP
        
        Args:
            prices: Current prices
            vwap: VWAP series
            
        Returns:
            Position string
        """
        try:
            if len(prices) == 0:
                return 'at_vwap'
            
            current_price = prices[-1]
            current_vwap = vwap[-1]
            
            deviation_pct = (current_price - current_vwap) / current_vwap
            
            if deviation_pct > 0.01:  # Above 1%
                return 'above_vwap'
            elif deviation_pct < -0.01:  # Below 1%
                return 'below_vwap'
            else:
                return 'at_vwap'
                
        except:
            return 'at_vwap'
    
    def _calculate_deviation_strength(self, deviation_scores: np.ndarray) -> float:
        """
        Calculate deviation strength from VWAP
        
        Args:
            deviation_scores: Deviation score series
            
        Returns:
            Deviation strength (0.0 to 1.0)
        """
        try:
            recent_scores = deviation_scores[-10:] if len(deviation_scores) >= 10 else deviation_scores
            avg_abs_deviation = np.mean(np.abs(recent_scores))
            return float(np.clip(avg_abs_deviation, 0.0, 1.0))
        except:
            return 0.0
    
    async def _calculate_underlying_vwap_context(self, 
                                               futures_data: Optional[Dict[str, np.ndarray]],
                                               timestamps: Optional[List[str]]) -> UnderlyingVWAPContext:
        """
        Calculate underlying futures VWAP for regime context
        
        Args:
            futures_data: Optional futures price/volume data
            timestamps: Optional timestamp series
            
        Returns:
            UnderlyingVWAPContext with regime analysis
        """
        if futures_data is None:
            # Return neutral context
            zeros = np.zeros(100)  # Placeholder
            return UnderlyingVWAPContext(
                futures_vwap_today=zeros,
                futures_vwap_previous=zeros,
                futures_vwap_weekly=zeros,
                futures_vwap_combined=zeros,
                regime_context='neutral_regime',
                regime_strength=0.0
            )
        
        try:
            futures_prices = futures_data.get('future_close', np.array([]))
            futures_volumes = futures_data.get('future_volume', np.array([]))
            
            if len(futures_prices) == 0:
                zeros = np.zeros(100)
                return UnderlyingVWAPContext(
                    futures_vwap_today=zeros,
                    futures_vwap_previous=zeros,
                    futures_vwap_weekly=zeros,
                    futures_vwap_combined=zeros,
                    regime_context='neutral_regime',
                    regime_strength=0.0
                )
            
            # Calculate futures VWAPs
            futures_vwap_today = self._calculate_vwap(futures_prices, futures_volumes)
            
            # Simplified previous day (use first half of data)
            half_point = len(futures_prices) // 2
            if half_point > 5:
                futures_vwap_prev = self._calculate_vwap(
                    futures_prices[:half_point], 
                    futures_volumes[:half_point]
                )
                futures_vwap_previous = np.full_like(futures_prices, futures_vwap_prev[-1])
            else:
                futures_vwap_previous = futures_vwap_today.copy()
            
            # Weekly VWAP (simplified - same as today for now)
            futures_vwap_weekly = futures_vwap_today.copy()
            
            # Combined VWAP
            futures_vwap_combined = (
                futures_vwap_today * self.vwap_config.underlying_today_weight +
                futures_vwap_previous * self.vwap_config.underlying_previous_weight +
                futures_vwap_weekly * self.vwap_config.underlying_weekly_weight
            )
            
            # Analyze regime
            regime_context, regime_strength = self._analyze_futures_regime(
                futures_prices, futures_vwap_combined
            )
            
            return UnderlyingVWAPContext(
                futures_vwap_today=futures_vwap_today,
                futures_vwap_previous=futures_vwap_previous,
                futures_vwap_weekly=futures_vwap_weekly,
                futures_vwap_combined=futures_vwap_combined,
                regime_context=regime_context,
                regime_strength=regime_strength
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate underlying VWAP context: {e}")
            zeros = np.zeros(len(futures_data.get('future_close', [100])))
            return UnderlyingVWAPContext(
                futures_vwap_today=zeros,
                futures_vwap_previous=zeros,
                futures_vwap_weekly=zeros,
                futures_vwap_combined=zeros,
                regime_context='neutral_regime',
                regime_strength=0.0
            )
    
    def _analyze_futures_regime(self, futures_prices: np.ndarray, futures_vwap: np.ndarray) -> Tuple[str, float]:
        """
        Analyze futures regime based on VWAP relationship
        
        Args:
            futures_prices: Futures price series
            futures_vwap: Futures VWAP series
            
        Returns:
            Tuple of (regime_context, regime_strength)
        """
        try:
            # Calculate how much time price spends above/below VWAP
            recent_points = min(50, len(futures_prices))
            recent_prices = futures_prices[-recent_points:]
            recent_vwap = futures_vwap[-recent_points:]
            
            above_vwap = np.mean(recent_prices > recent_vwap)
            below_vwap = np.mean(recent_prices < recent_vwap)
            
            if above_vwap > 0.6:
                regime_context = 'bullish_regime'
                regime_strength = above_vwap
            elif below_vwap > 0.6:
                regime_context = 'bearish_regime'
                regime_strength = below_vwap
            else:
                regime_context = 'neutral_regime'
                regime_strength = 1.0 - abs(above_vwap - 0.5) * 2  # Distance from neutral
            
            return regime_context, float(regime_strength)
            
        except:
            return 'neutral_regime', 0.5
    
    async def _analyze_cross_straddle_vwap(self, 
                                         vwap_results: List[VWAPResult],
                                         volume_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze VWAP patterns across all straddle types
        
        Args:
            vwap_results: List of VWAP results for all straddle types
            volume_data: Volume data dictionary
            
        Returns:
            Cross-straddle VWAP analysis
        """
        try:
            # Calculate overall deviation score
            deviation_scores = [result.deviation_strength for result in vwap_results]
            overall_deviation = float(np.mean(deviation_scores))
            
            # Analyze volume concentration
            total_volumes = {}
            for result in vwap_results:
                total_volumes[result.straddle_type.lower()] = np.sum(result.volume_profile)
            
            max_volume_type = max(total_volumes, key=total_volumes.get)
            
            if max_volume_type == 'atm':
                volume_concentration = 'atm_heavy'
            elif max_volume_type == 'itm1':
                volume_concentration = 'itm_heavy'
            elif max_volume_type == 'otm1':
                volume_concentration = 'otm_heavy'
            else:
                volume_concentration = 'balanced'
            
            # Calculate VWAP trend alignment
            price_positions = [result.price_position for result in vwap_results]
            above_count = sum(1 for pos in price_positions if pos == 'above_vwap')
            below_count = sum(1 for pos in price_positions if pos == 'below_vwap')
            
            trend_alignment = (above_count - below_count) / len(price_positions)
            
            # Count zero volume periods
            all_volumes = np.concatenate([result.volume_profile for result in vwap_results])
            zero_volume_periods = int(np.sum(all_volumes <= self.min_volume_threshold))
            
            return {
                'overall_deviation': overall_deviation,
                'volume_concentration': volume_concentration,
                'trend_alignment': float(trend_alignment),
                'zero_volume_periods': zero_volume_periods
            }
            
        except Exception as e:
            self.logger.error(f"Cross-straddle VWAP analysis failed: {e}")
            return {
                'overall_deviation': 0.0,
                'volume_concentration': 'balanced',
                'trend_alignment': 0.0,
                'zero_volume_periods': 0
            }
    
    async def get_vwap_feature_vector(self, analysis: RollingStraddleVWAPAnalysis) -> Dict[str, float]:
        """
        Extract feature vector from VWAP analysis
        
        Args:
            analysis: Complete VWAP analysis result
            
        Returns:
            Feature vector dictionary
        """
        return {
            # Individual VWAP deviation strengths
            'atm_vwap_deviation': analysis.atm_vwap.deviation_strength,
            'itm1_vwap_deviation': analysis.itm1_vwap.deviation_strength,
            'otm1_vwap_deviation': analysis.otm1_vwap.deviation_strength,
            
            # Price positions (encoded)
            'atm_above_vwap': 1.0 if analysis.atm_vwap.price_position == 'above_vwap' else 0.0,
            'itm1_above_vwap': 1.0 if analysis.itm1_vwap.price_position == 'above_vwap' else 0.0,
            'otm1_above_vwap': 1.0 if analysis.otm1_vwap.price_position == 'above_vwap' else 0.0,
            
            # Cross-straddle features
            'overall_vwap_deviation': analysis.overall_deviation_score,
            'vwap_trend_alignment': analysis.vwap_trend_alignment,
            
            # Volume concentration
            'volume_atm_heavy': 1.0 if analysis.volume_concentration == 'atm_heavy' else 0.0,
            'volume_itm_heavy': 1.0 if analysis.volume_concentration == 'itm_heavy' else 0.0,
            'volume_otm_heavy': 1.0 if analysis.volume_concentration == 'otm_heavy' else 0.0,
            
            # Underlying regime context
            'regime_bullish': 1.0 if analysis.underlying_context.regime_context == 'bullish_regime' else 0.0,
            'regime_bearish': 1.0 if analysis.underlying_context.regime_context == 'bearish_regime' else 0.0,
            'regime_strength': analysis.underlying_context.regime_strength
        }


# Factory function
def create_vwap_engine(config: Dict[str, Any]) -> RollingStraddleVWAPEngine:
    """Create and configure RollingStraddleVWAPEngine instance"""
    return RollingStraddleVWAPEngine(config)