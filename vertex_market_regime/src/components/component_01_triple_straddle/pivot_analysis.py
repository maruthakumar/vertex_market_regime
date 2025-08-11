"""
Rolling Straddle Pivot Analysis for Component 1

Revolutionary pivot analysis applied to rolling straddle prices with standard
pivot calculations (PP, R1-R3, S1-S3), CPR analysis for underlying futures 
prices (regime classification), pivot level scoring system for straddle position
relative to pivots, and DTE-specific pivot weighting where short DTE favors pivots.

Key Innovation: Pivot levels calculated on rolling straddle prices create unique
support/resistance levels for options market dynamics.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class PivotLevels:
    """Pivot levels for single time period"""
    pivot_point: float
    resistance_1: float
    resistance_2: float
    resistance_3: float
    support_1: float
    support_2: float
    support_3: float
    
    # CPR levels (for futures)
    tc: Optional[float] = None  # Top Central Pivot
    pivot: Optional[float] = None  # Central Pivot (same as pivot_point)
    bc: Optional[float] = None  # Bottom Central Pivot
    cpr_width: Optional[float] = None  # CPR width


@dataclass
class PivotAnalysisResult:
    """Pivot analysis result for single straddle type"""
    straddle_type: str
    current_pivots: PivotLevels
    previous_pivots: PivotLevels
    
    # Position analysis
    price_position_score: float  # -3 to +3 (S3 to R3)
    nearest_pivot_distance: float
    pivot_breakout_strength: float  # 0.0 to 1.0
    
    # DTE-specific weighting
    dte_weight: float
    pivot_dominance: float  # How much pivots dominate for this DTE
    
    # Pivot interactions
    pivot_bounces: int
    pivot_breaks: int
    consolidation_zones: List[Tuple[float, float]]  # (support, resistance) pairs


@dataclass
class CPRAnalysisResult:
    """Central Pivot Range analysis for underlying futures"""
    cpr_levels: PivotLevels
    cpr_width_normalized: float  # CPR width relative to price
    price_position_in_cpr: str  # 'above_cpr', 'in_cpr', 'below_cpr'
    regime_classification: str  # 'trending_up', 'trending_down', 'sideways'
    breakout_probability: float  # 0.0 to 1.0
    volume_confirmation: float  # Volume support for moves


@dataclass
class RollingStraddlePivotAnalysis:
    """Complete pivot analysis for all rolling straddle types"""
    atm_pivots: PivotAnalysisResult
    itm1_pivots: PivotAnalysisResult
    otm1_pivots: PivotAnalysisResult
    cpr_analysis: CPRAnalysisResult
    
    # Cross-straddle pivot analysis
    overall_pivot_alignment: float  # -1.0 to +1.0
    dominant_pivot_zone: str  # 'resistance', 'support', 'pivot'
    market_structure: str  # 'trending', 'ranging', 'breakout'
    pivot_confluence_strength: float
    
    # Processing metadata
    processing_time_ms: float
    data_points_processed: int
    dte_average: float
    metadata: Dict[str, Any]


class RollingStraddlePivotEngine:
    """
    Revolutionary Pivot Engine for Rolling Straddle Prices
    
    Calculates pivot levels on rolling straddle prices and underlying futures,
    providing unique support/resistance analysis for options market structure.
    
    Key Features:
    - Standard pivots (PP, R1-R3, S1-S3) on rolling straddle prices
    - CPR analysis on underlying futures for regime classification
    - DTE-specific pivot weighting (short DTE favors pivots)
    - Pivot level scoring system
    - Breakout and consolidation detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Rolling Straddle Pivot Engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance configuration
        self.processing_budget_ms = config.get('processing_budget_ms', 30)
        self.use_gpu = config.get('use_gpu', GPU_AVAILABLE)
        
        # DTE-specific weighting configuration
        self.dte_weight_matrix = config.get('dte_weight_matrix', {
            'dte_0_3': 0.80,    # 0-3 DTE: Pivot analysis dominates (80-50%)
            'dte_4_7': 0.65,    # 4-7 DTE: High pivot weight
            'dte_8_15': 0.50,   # 8-15 DTE: Balanced approach
            'dte_16_plus': 0.30 # 16+ DTE: EMA analysis increases (up to 60%)
        })
        
        # Pivot calculation configuration
        self.use_traditional_pivots = config.get('use_traditional_pivots', True)
        self.pivot_sensitivity = config.get('pivot_sensitivity', 0.005)  # 0.5% sensitivity
        self.breakout_threshold = config.get('breakout_threshold', 0.01)  # 1% breakout
        self.consolidation_range = config.get('consolidation_range', 0.02)  # 2% consolidation
        
        # Session configuration for OHLC calculation
        self.session_start_hour = config.get('session_start_hour', 9)
        self.session_end_hour = config.get('session_end_hour', 15)
        self.min_session_data = config.get('min_session_data', 10)
        
        self.logger.info("RollingStraddlePivotEngine initialized with DTE-specific weighting")
    
    async def analyze_rolling_straddle_pivots(self, 
                                            straddle_data: Dict[str, np.ndarray],
                                            futures_data: Dict[str, np.ndarray],
                                            dte_data: Optional[np.ndarray] = None,
                                            timestamps: Optional[List[str]] = None) -> RollingStraddlePivotAnalysis:
        """
        Analyze pivot levels for all rolling straddle types
        
        Args:
            straddle_data: Dictionary with straddle price series
            futures_data: Underlying futures OHLCV data
            dte_data: Optional DTE series for weighting
            timestamps: Optional timestamp series for session analysis
            
        Returns:
            RollingStraddlePivotAnalysis with complete pivot analysis
        """
        start_time = time.time()
        
        try:
            # Validate input data
            required_straddles = ['atm_straddle', 'itm1_straddle', 'otm1_straddle']
            for straddle in required_straddles:
                if straddle not in straddle_data:
                    raise ValueError(f"Missing straddle data: {straddle}")
            
            # Calculate DTE-specific weights
            avg_dte = float(np.mean(dte_data)) if dte_data is not None else 10.0
            dte_weights = self._calculate_dte_weights(dte_data if dte_data is not None else [avg_dte])
            
            # Extract session OHLC data for pivot calculation
            session_data = await self._extract_session_ohlc(straddle_data, timestamps)
            futures_ohlc = await self._extract_futures_ohlc(futures_data, timestamps)
            
            # Calculate pivot analysis for each straddle type
            atm_pivots = await self._calculate_pivots_for_straddle(
                session_data['atm_straddle'], 'ATM', dte_weights, avg_dte
            )
            
            itm1_pivots = await self._calculate_pivots_for_straddle(
                session_data['itm1_straddle'], 'ITM1', dte_weights, avg_dte
            )
            
            otm1_pivots = await self._calculate_pivots_for_straddle(
                session_data['otm1_straddle'], 'OTM1', dte_weights, avg_dte
            )
            
            # Calculate CPR analysis for underlying futures
            cpr_analysis = await self._calculate_cpr_analysis(futures_ohlc, futures_data)
            
            # Cross-straddle pivot analysis
            cross_analysis = await self._analyze_cross_pivot_structure(
                [atm_pivots, itm1_pivots, otm1_pivots], cpr_analysis
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Performance validation
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"Pivot processing {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            return RollingStraddlePivotAnalysis(
                atm_pivots=atm_pivots,
                itm1_pivots=itm1_pivots,
                otm1_pivots=otm1_pivots,
                cpr_analysis=cpr_analysis,
                overall_pivot_alignment=cross_analysis['pivot_alignment'],
                dominant_pivot_zone=cross_analysis['dominant_zone'],
                market_structure=cross_analysis['market_structure'],
                pivot_confluence_strength=cross_analysis['confluence_strength'],
                processing_time_ms=processing_time,
                data_points_processed=len(straddle_data['atm_straddle']),
                dte_average=avg_dte,
                metadata={
                    'dte_weight_matrix': self.dte_weight_matrix,
                    'pivot_sensitivity': self.pivot_sensitivity,
                    'breakout_threshold': self.breakout_threshold,
                    'session_separation': timestamps is not None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze rolling straddle pivots: {e}")
            raise
    
    def _calculate_dte_weights(self, dte_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate DTE-specific weights for pivot analysis
        
        Args:
            dte_data: DTE series
            
        Returns:
            Dictionary with DTE-specific weights
        """
        try:
            avg_dte = float(np.mean(dte_data))
            
            # Determine DTE bucket and weight
            if avg_dte <= 3:
                base_weight = self.dte_weight_matrix['dte_0_3']
            elif avg_dte <= 7:
                base_weight = self.dte_weight_matrix['dte_4_7']
            elif avg_dte <= 15:
                base_weight = self.dte_weight_matrix['dte_8_15']
            else:
                base_weight = self.dte_weight_matrix['dte_16_plus']
            
            return {
                'pivot_weight': base_weight,
                'ema_weight': 1.0 - base_weight,
                'avg_dte': avg_dte
            }
            
        except Exception as e:
            self.logger.warning(f"DTE weight calculation failed: {e}")
            return {'pivot_weight': 0.5, 'ema_weight': 0.5, 'avg_dte': 10.0}
    
    async def _extract_session_ohlc(self, 
                                   straddle_data: Dict[str, np.ndarray],
                                   timestamps: Optional[List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Extract session OHLC data for pivot calculation
        
        Args:
            straddle_data: Straddle price data
            timestamps: Optional timestamp series
            
        Returns:
            Dictionary with OHLC data for each straddle type
        """
        try:
            session_data = {}
            
            for straddle_type, price_series in straddle_data.items():
                if len(price_series) == 0:
                    session_data[straddle_type] = {
                        'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0
                    }
                    continue
                
                # Calculate OHLC for the session
                if timestamps is not None:
                    # Try to separate by trading session
                    try:
                        dt_timestamps = pd.to_datetime(timestamps)
                        current_date = dt_timestamps.date.iloc[-1]
                        current_session_mask = dt_timestamps.date == current_date
                        session_prices = price_series[current_session_mask]
                        
                        if len(session_prices) > 0:
                            ohlc = {
                                'open': float(session_prices[0]),
                                'high': float(np.max(session_prices)),
                                'low': float(np.min(session_prices)),
                                'close': float(session_prices[-1])
                            }
                        else:
                            # Fallback to full series
                            ohlc = {
                                'open': float(price_series[0]),
                                'high': float(np.max(price_series)),
                                'low': float(np.min(price_series)),
                                'close': float(price_series[-1])
                            }
                    except:
                        # Fallback to full series
                        ohlc = {
                            'open': float(price_series[0]),
                            'high': float(np.max(price_series)),
                            'low': float(np.min(price_series)),
                            'close': float(price_series[-1])
                        }
                else:
                    # Use full series for OHLC
                    ohlc = {
                        'open': float(price_series[0]),
                        'high': float(np.max(price_series)),
                        'low': float(np.min(price_series)),
                        'close': float(price_series[-1])
                    }
                
                session_data[straddle_type] = ohlc
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"Session OHLC extraction failed: {e}")
            # Return placeholder data
            return {
                straddle_type: {'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0}
                for straddle_type in straddle_data.keys()
            }
    
    async def _extract_futures_ohlc(self, 
                                   futures_data: Dict[str, np.ndarray],
                                   timestamps: Optional[List[str]]) -> Dict[str, float]:
        """
        Extract futures OHLC data for CPR analysis
        
        Args:
            futures_data: Futures price data
            timestamps: Optional timestamp series
            
        Returns:
            Dictionary with futures OHLC
        """
        try:
            # Use existing OHLC if available
            if all(key in futures_data for key in ['future_open', 'future_high', 'future_low', 'future_close']):
                return {
                    'open': float(futures_data['future_open'][-1]) if len(futures_data['future_open']) > 0 else 0.0,
                    'high': float(np.max(futures_data['future_high'])),
                    'low': float(np.min(futures_data['future_low'])),
                    'close': float(futures_data['future_close'][-1]) if len(futures_data['future_close']) > 0 else 0.0
                }
            
            # Calculate from close prices if OHLC not available
            close_prices = futures_data.get('future_close', np.array([]))
            if len(close_prices) == 0:
                return {'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0}
            
            return {
                'open': float(close_prices[0]),
                'high': float(np.max(close_prices)),
                'low': float(np.min(close_prices)),
                'close': float(close_prices[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Futures OHLC extraction failed: {e}")
            return {'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0}
    
    async def _calculate_pivots_for_straddle(self, 
                                           ohlc: Dict[str, float],
                                           straddle_type: str,
                                           dte_weights: Dict[str, float],
                                           avg_dte: float) -> PivotAnalysisResult:
        """
        Calculate pivot analysis for single straddle type
        
        Args:
            ohlc: OHLC data dictionary
            straddle_type: Type of straddle ('ATM', 'ITM1', 'OTM1')
            dte_weights: DTE-specific weights
            avg_dte: Average DTE
            
        Returns:
            PivotAnalysisResult with complete pivot analysis
        """
        try:
            # Calculate current pivot levels
            current_pivots = self._calculate_pivot_levels(ohlc)
            
            # Calculate previous pivot levels (simplified - use adjusted OHLC)
            previous_ohlc = {
                'open': ohlc['close'] * 0.99,  # Simulated previous
                'high': ohlc['high'] * 0.995,
                'low': ohlc['low'] * 1.005,
                'close': ohlc['open']
            }
            previous_pivots = self._calculate_pivot_levels(previous_ohlc)
            
            # Analyze current price position relative to pivots
            current_price = ohlc['close']
            price_position_score = self._calculate_price_position_score(current_price, current_pivots)
            
            # Calculate distance to nearest pivot level
            nearest_distance = self._calculate_nearest_pivot_distance(current_price, current_pivots)
            
            # Calculate breakout strength
            breakout_strength = self._calculate_breakout_strength(ohlc, current_pivots)
            
            # Apply DTE-specific weighting
            dte_weight = dte_weights['pivot_weight']
            pivot_dominance = self._calculate_pivot_dominance(avg_dte)
            
            # Analyze pivot interactions (simplified)
            pivot_bounces = self._count_pivot_bounces(ohlc, current_pivots)
            pivot_breaks = self._count_pivot_breaks(ohlc, current_pivots)
            consolidation_zones = self._identify_consolidation_zones(current_pivots)
            
            return PivotAnalysisResult(
                straddle_type=straddle_type,
                current_pivots=current_pivots,
                previous_pivots=previous_pivots,
                price_position_score=price_position_score,
                nearest_pivot_distance=nearest_distance,
                pivot_breakout_strength=breakout_strength,
                dte_weight=dte_weight,
                pivot_dominance=pivot_dominance,
                pivot_bounces=pivot_bounces,
                pivot_breaks=pivot_breaks,
                consolidation_zones=consolidation_zones
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate pivots for {straddle_type}: {e}")
            # Return empty result
            empty_pivots = PivotLevels(0, 0, 0, 0, 0, 0, 0)
            return PivotAnalysisResult(
                straddle_type=straddle_type,
                current_pivots=empty_pivots,
                previous_pivots=empty_pivots,
                price_position_score=0.0,
                nearest_pivot_distance=0.0,
                pivot_breakout_strength=0.0,
                dte_weight=0.5,
                pivot_dominance=0.5,
                pivot_bounces=0,
                pivot_breaks=0,
                consolidation_zones=[]
            )
    
    def _calculate_pivot_levels(self, ohlc: Dict[str, float]) -> PivotLevels:
        """
        Calculate standard pivot levels from OHLC
        
        Args:
            ohlc: OHLC data dictionary
            
        Returns:
            PivotLevels with calculated pivot points
        """
        try:
            high = ohlc['high']
            low = ohlc['low']
            close = ohlc['close']
            
            # Standard pivot point calculation
            pivot_point = (high + low + close) / 3.0
            
            # Resistance levels
            r1 = 2 * pivot_point - low
            r2 = pivot_point + (high - low)
            r3 = high + 2 * (pivot_point - low)
            
            # Support levels
            s1 = 2 * pivot_point - high
            s2 = pivot_point - (high - low)
            s3 = low - 2 * (high - pivot_point)
            
            return PivotLevels(
                pivot_point=pivot_point,
                resistance_1=r1,
                resistance_2=r2,
                resistance_3=r3,
                support_1=s1,
                support_2=s2,
                support_3=s3
            )
            
        except Exception as e:
            self.logger.debug(f"Pivot calculation failed: {e}")
            return PivotLevels(0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_price_position_score(self, price: float, pivots: PivotLevels) -> float:
        """
        Calculate price position score relative to pivot levels
        
        Score ranges from -3 (at S3) to +3 (at R3)
        
        Args:
            price: Current price
            pivots: Pivot levels
            
        Returns:
            Position score (-3.0 to +3.0)
        """
        try:
            # Define pivot levels in order
            levels = [
                (pivots.support_3, -3.0),
                (pivots.support_2, -2.0),
                (pivots.support_1, -1.0),
                (pivots.pivot_point, 0.0),
                (pivots.resistance_1, 1.0),
                (pivots.resistance_2, 2.0),
                (pivots.resistance_3, 3.0)
            ]
            
            # Find position between levels
            for i, (level, score) in enumerate(levels):
                if price <= level:
                    if i == 0:
                        return -3.0  # Below S3
                    else:
                        # Interpolate between levels
                        prev_level, prev_score = levels[i-1]
                        ratio = (price - prev_level) / (level - prev_level) if level != prev_level else 0
                        return prev_score + ratio * (score - prev_score)
            
            return 3.0  # Above R3
            
        except:
            return 0.0
    
    def _calculate_nearest_pivot_distance(self, price: float, pivots: PivotLevels) -> float:
        """
        Calculate distance to nearest pivot level (normalized)
        
        Args:
            price: Current price
            pivots: Pivot levels
            
        Returns:
            Normalized distance (0.0 to 1.0)
        """
        try:
            pivot_levels = [
                pivots.support_3, pivots.support_2, pivots.support_1,
                pivots.pivot_point,
                pivots.resistance_1, pivots.resistance_2, pivots.resistance_3
            ]
            
            distances = [abs(price - level) for level in pivot_levels if level > 0]
            
            if not distances:
                return 0.0
            
            min_distance = min(distances)
            
            # Normalize by price level
            normalized_distance = min_distance / max(price, 1.0)
            return float(min(normalized_distance, 1.0))
            
        except:
            return 0.0
    
    def _calculate_breakout_strength(self, ohlc: Dict[str, float], pivots: PivotLevels) -> float:
        """
        Calculate breakout strength beyond pivot levels
        
        Args:
            ohlc: OHLC data
            pivots: Pivot levels
            
        Returns:
            Breakout strength (0.0 to 1.0)
        """
        try:
            high = ohlc['high']
            low = ohlc['low']
            close = ohlc['close']
            
            # Check for breakouts
            resistance_breakout = 0.0
            support_breakout = 0.0
            
            # Calculate breakout strength above resistance levels
            if close > pivots.resistance_1:
                if close > pivots.resistance_3:
                    resistance_breakout = 1.0
                elif close > pivots.resistance_2:
                    resistance_breakout = 0.75
                else:
                    resistance_breakout = 0.5
            
            # Calculate breakout strength below support levels
            if close < pivots.support_1:
                if close < pivots.support_3:
                    support_breakout = 1.0
                elif close < pivots.support_2:
                    support_breakout = 0.75
                else:
                    support_breakout = 0.5
            
            # Combine breakout strengths
            total_breakout = max(resistance_breakout, support_breakout)
            
            # Factor in volume/range confirmation (simplified)
            range_confirmation = (high - low) / max(close, 1.0)
            if range_confirmation > self.breakout_threshold:
                total_breakout *= 1.2  # Boost for wide range
            
            return float(min(total_breakout, 1.0))
            
        except:
            return 0.0
    
    def _calculate_pivot_dominance(self, avg_dte: float) -> float:
        """
        Calculate how much pivots dominate based on DTE
        
        Args:
            avg_dte: Average days to expiry
            
        Returns:
            Pivot dominance factor (0.0 to 1.0)
        """
        # Short DTE (0-3) = high pivot dominance
        # Long DTE (16+) = low pivot dominance
        
        if avg_dte <= 3:
            return 0.9
        elif avg_dte <= 7:
            return 0.7
        elif avg_dte <= 15:
            return 0.5
        else:
            return 0.3
    
    def _count_pivot_bounces(self, ohlc: Dict[str, float], pivots: PivotLevels) -> int:
        """Count approximate pivot bounces (simplified)"""
        # Simplified bounce detection
        bounces = 0
        price = ohlc['close']
        
        # Check if price bounced off major levels
        major_levels = [pivots.support_1, pivots.pivot_point, pivots.resistance_1]
        
        for level in major_levels:
            if level > 0 and abs(price - level) / max(price, 1.0) < self.pivot_sensitivity:
                bounces += 1
        
        return bounces
    
    def _count_pivot_breaks(self, ohlc: Dict[str, float], pivots: PivotLevels) -> int:
        """Count pivot level breaks (simplified)"""
        breaks = 0
        high = ohlc['high']
        low = ohlc['low']
        
        # Count breaks above resistance
        if high > pivots.resistance_1:
            breaks += 1
        if high > pivots.resistance_2:
            breaks += 1
        if high > pivots.resistance_3:
            breaks += 1
        
        # Count breaks below support
        if low < pivots.support_1:
            breaks += 1
        if low < pivots.support_2:
            breaks += 1
        if low < pivots.support_3:
            breaks += 1
        
        return breaks
    
    def _identify_consolidation_zones(self, pivots: PivotLevels) -> List[Tuple[float, float]]:
        """Identify consolidation zones between pivot levels"""
        zones = []
        
        # Major consolidation zones
        zones.append((pivots.support_1, pivots.resistance_1))  # S1-R1 zone
        zones.append((pivots.support_2, pivots.pivot_point))    # S2-PP zone
        zones.append((pivots.pivot_point, pivots.resistance_2)) # PP-R2 zone
        
        return zones
    
    async def _calculate_cpr_analysis(self, 
                                    futures_ohlc: Dict[str, float],
                                    futures_data: Dict[str, np.ndarray]) -> CPRAnalysisResult:
        """
        Calculate Central Pivot Range (CPR) analysis for underlying futures
        
        Args:
            futures_ohlc: Futures OHLC data
            futures_data: Complete futures data
            
        Returns:
            CPRAnalysisResult with regime classification
        """
        try:
            # Calculate CPR levels
            high = futures_ohlc['high']
            low = futures_ohlc['low']
            close = futures_ohlc['close']
            
            # CPR calculation
            pivot = (high + low + close) / 3.0
            tc = (pivot - low) + pivot  # Top Central
            bc = pivot - (pivot - low)  # Bottom Central
            
            cpr_width = tc - bc
            cpr_width_normalized = cpr_width / max(close, 1.0)
            
            cpr_levels = PivotLevels(
                pivot_point=pivot,
                resistance_1=(2 * pivot) - low,
                resistance_2=pivot + (high - low),
                resistance_3=high + 2 * (pivot - low),
                support_1=(2 * pivot) - high,
                support_2=pivot - (high - low),
                support_3=low - 2 * (high - pivot),
                tc=tc,
                pivot=pivot,
                bc=bc,
                cpr_width=cpr_width
            )
            
            # Determine price position in CPR
            if close > tc:
                price_position = 'above_cpr'
            elif close < bc:
                price_position = 'below_cpr'
            else:
                price_position = 'in_cpr'
            
            # Regime classification based on CPR
            if cpr_width_normalized < 0.01:  # Narrow CPR
                if price_position == 'above_cpr':
                    regime = 'trending_up'
                elif price_position == 'below_cpr':
                    regime = 'trending_down'
                else:
                    regime = 'sideways'
            else:  # Wide CPR
                regime = 'sideways'
            
            # Calculate breakout probability
            breakout_prob = min(1.0 - cpr_width_normalized * 10, 1.0)
            
            # Volume confirmation (simplified)
            volume_confirmation = 0.5  # Placeholder
            if 'future_volume' in futures_data:
                volumes = futures_data['future_volume']
                if len(volumes) > 1:
                    recent_vol = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
                    avg_vol = np.mean(volumes)
                    volume_confirmation = min(recent_vol / max(avg_vol, 1), 2.0) / 2.0
            
            return CPRAnalysisResult(
                cpr_levels=cpr_levels,
                cpr_width_normalized=cpr_width_normalized,
                price_position_in_cpr=price_position,
                regime_classification=regime,
                breakout_probability=breakout_prob,
                volume_confirmation=volume_confirmation
            )
            
        except Exception as e:
            self.logger.error(f"CPR analysis failed: {e}")
            empty_pivots = PivotLevels(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            return CPRAnalysisResult(
                cpr_levels=empty_pivots,
                cpr_width_normalized=0.0,
                price_position_in_cpr='in_cpr',
                regime_classification='sideways',
                breakout_probability=0.5,
                volume_confirmation=0.5
            )
    
    async def _analyze_cross_pivot_structure(self, 
                                           pivot_results: List[PivotAnalysisResult],
                                           cpr_analysis: CPRAnalysisResult) -> Dict[str, Any]:
        """
        Analyze pivot structure across all straddle types
        
        Args:
            pivot_results: List of pivot analysis results
            cpr_analysis: CPR analysis result
            
        Returns:
            Cross-pivot structure analysis
        """
        try:
            # Calculate overall pivot alignment
            position_scores = [result.price_position_score for result in pivot_results]
            pivot_alignment = float(np.mean(position_scores)) / 3.0  # Normalize to [-1, 1]
            
            # Determine dominant pivot zone
            avg_score = np.mean(position_scores)
            if avg_score > 0.5:
                dominant_zone = 'resistance'
            elif avg_score < -0.5:
                dominant_zone = 'support'
            else:
                dominant_zone = 'pivot'
            
            # Analyze market structure
            breakout_strengths = [result.pivot_breakout_strength for result in pivot_results]
            avg_breakout = np.mean(breakout_strengths)
            
            if avg_breakout > 0.7:
                market_structure = 'breakout'
            elif cpr_analysis.regime_classification == 'sideways':
                market_structure = 'ranging'
            else:
                market_structure = 'trending'
            
            # Calculate confluence strength
            distances = [result.nearest_pivot_distance for result in pivot_results]
            confluence_strength = float(1.0 - np.mean(distances))  # Closer = higher confluence
            
            return {
                'pivot_alignment': pivot_alignment,
                'dominant_zone': dominant_zone,
                'market_structure': market_structure,
                'confluence_strength': confluence_strength
            }
            
        except Exception as e:
            self.logger.error(f"Cross-pivot structure analysis failed: {e}")
            return {
                'pivot_alignment': 0.0,
                'dominant_zone': 'pivot',
                'market_structure': 'ranging',
                'confluence_strength': 0.5
            }
    
    async def get_pivot_feature_vector(self, analysis: RollingStraddlePivotAnalysis) -> Dict[str, float]:
        """
        Extract feature vector from pivot analysis
        
        Args:
            analysis: Complete pivot analysis result
            
        Returns:
            Feature vector dictionary
        """
        return {
            # Individual pivot position scores
            'atm_pivot_position': analysis.atm_pivots.price_position_score / 3.0,  # Normalize
            'itm1_pivot_position': analysis.itm1_pivots.price_position_score / 3.0,
            'otm1_pivot_position': analysis.otm1_pivots.price_position_score / 3.0,
            
            # Breakout strengths
            'atm_breakout_strength': analysis.atm_pivots.pivot_breakout_strength,
            'itm1_breakout_strength': analysis.itm1_pivots.pivot_breakout_strength,
            'otm1_breakout_strength': analysis.otm1_pivots.pivot_breakout_strength,
            
            # DTE-adjusted pivot weights
            'atm_pivot_dominance': analysis.atm_pivots.pivot_dominance,
            'itm1_pivot_dominance': analysis.itm1_pivots.pivot_dominance,
            'otm1_pivot_dominance': analysis.otm1_pivots.pivot_dominance,
            
            # Cross-straddle features
            'overall_pivot_alignment': analysis.overall_pivot_alignment,
            'pivot_confluence_strength': analysis.pivot_confluence_strength,
            
            # Market structure encoding
            'structure_trending': 1.0 if analysis.market_structure == 'trending' else 0.0,
            'structure_ranging': 1.0 if analysis.market_structure == 'ranging' else 0.0,
            'structure_breakout': 1.0 if analysis.market_structure == 'breakout' else 0.0,
            
            # CPR analysis
            'cpr_width_normalized': analysis.cpr_analysis.cpr_width_normalized,
            'cpr_breakout_probability': analysis.cpr_analysis.breakout_probability,
            'cpr_above': 1.0 if analysis.cpr_analysis.price_position_in_cpr == 'above_cpr' else 0.0,
            'cpr_below': 1.0 if analysis.cpr_analysis.price_position_in_cpr == 'below_cpr' else 0.0,
            
            # Regime classification
            'regime_trending_up': 1.0 if analysis.cpr_analysis.regime_classification == 'trending_up' else 0.0,
            'regime_trending_down': 1.0 if analysis.cpr_analysis.regime_classification == 'trending_down' else 0.0,
            'regime_sideways': 1.0 if analysis.cpr_analysis.regime_classification == 'sideways' else 0.0
        }


# Factory function
def create_pivot_engine(config: Dict[str, Any]) -> RollingStraddlePivotEngine:
    """Create and configure RollingStraddlePivotEngine instance"""
    return RollingStraddlePivotEngine(config)