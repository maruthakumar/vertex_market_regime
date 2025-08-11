"""
OI-PA Trending Classification Engine for Component 3

This module implements the comprehensive OI-PA trending classification system with:
- CE side option seller analysis (4 patterns)
- PE side option seller analysis (4 patterns) 
- Future (underlying) OI seller analysis with correlation
- 3-way correlation matrix (CE+PE+Future)
- Comprehensive market regime formation (8-regime classification)

As per story requirements:
- CE SIDE: Price DOWN + CE_OI UP = SHORT BUILDUP, Price UP + CE_OI DOWN = SHORT COVERING, etc.
- PE SIDE: Price UP + PE_OI UP = SHORT BUILDUP, Price DOWN + PE_OI DOWN = SHORT COVERING, etc.  
- FUTURE: Price UP + FUTURE_OI UP = LONG BUILDUP, Price DOWN + FUTURE_OI DOWN = LONG UNWINDING, etc.
- 3-WAY CORRELATION: Strong bullish/bearish correlation, institutional positioning, ranging market, etc.
- COMPREHENSIVE MARKET REGIME: 8-regime classification using complete correlation matrix
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CEOptionSellerPattern(Enum):
    """CE side option seller patterns (as per story requirements)."""
    CE_SHORT_BUILDUP = "ce_short_buildup"      # Price DOWN + CE_OI UP (bearish sentiment, call writers selling calls)
    CE_SHORT_COVERING = "ce_short_covering"    # Price UP + CE_OI DOWN (call writers buying back calls)  
    CE_LONG_BUILDUP = "ce_long_buildup"        # Price UP + CE_OI UP (bullish sentiment, call buyers buying calls)
    CE_LONG_UNWINDING = "ce_long_unwinding"    # Price DOWN + CE_OI DOWN (call buyers selling calls)


class PEOptionSellerPattern(Enum):
    """PE side option seller patterns (as per story requirements).""" 
    PE_SHORT_BUILDUP = "pe_short_buildup"      # Price UP + PE_OI UP (bullish underlying, put writers selling puts)
    PE_SHORT_COVERING = "pe_short_covering"    # Price DOWN + PE_OI DOWN (put writers buying back puts)
    PE_LONG_BUILDUP = "pe_long_buildup"        # Price DOWN + PE_OI UP (bearish sentiment, put buyers buying puts)
    PE_LONG_UNWINDING = "pe_long_unwinding"    # Price UP + PE_OI DOWN (put buyers selling puts)


class FutureSellerPattern(Enum):
    """Future (underlying) OI seller patterns (as per story requirements)."""
    FUTURE_LONG_BUILDUP = "future_long_buildup"    # Price UP + FUTURE_OI UP (bullish sentiment, future buyers)
    FUTURE_LONG_UNWINDING = "future_long_unwinding"  # Price DOWN + FUTURE_OI DOWN (future buyers closing positions)
    FUTURE_SHORT_BUILDUP = "future_short_buildup"   # Price DOWN + FUTURE_OI UP (bearish sentiment, future sellers)  
    FUTURE_SHORT_COVERING = "future_short_covering"  # Price UP + FUTURE_OI DOWN (future sellers covering positions)


class ThreeWayCorrelationPattern(Enum):
    """3-way correlation matrix patterns (as per story requirements)."""
    STRONG_BULLISH_CORRELATION = "strong_bullish_correlation"      # CE Long Buildup + PE Short Buildup + Future Long Buildup
    STRONG_BEARISH_CORRELATION = "strong_bearish_correlation"      # CE Short Buildup + PE Long Buildup + Future Short Buildup  
    INSTITUTIONAL_POSITIONING = "institutional_positioning"        # Mixed patterns across CE/PE/Future (hedging/arbitrage)
    RANGING_SIDEWAYS_MARKET = "ranging_sideways_market"           # Non-aligned patterns across instruments
    TRANSITION_REVERSAL_SETUP = "transition_reversal_setup"       # Correlation breakdown between instruments
    ARBITRAGE_COMPLEX_STRATEGY = "arbitrage_complex_strategy"     # Opposite positioning patterns across instruments


class ComprehensiveMarketRegime(Enum):
    """Comprehensive 8-regime classification system (as per story requirements)."""
    TRENDING_BULLISH = "trending_bullish"                    # CE Long Buildup + PE Short Buildup + Future Long Buildup
    TRENDING_BEARISH = "trending_bearish"                    # CE Short Buildup + PE Long Buildup + Future Short Buildup
    BULLISH_REVERSAL_SETUP = "bullish_reversal_setup"       # CE Short Covering + PE Long Unwinding + Future Short Covering
    BEARISH_REVERSAL_SETUP = "bearish_reversal_setup"       # CE Long Unwinding + PE Short Covering + Future Long Unwinding
    INSTITUTIONAL_ACCUMULATION = "institutional_accumulation"  # Mixed CE/PE patterns + Future Long Buildup
    INSTITUTIONAL_DISTRIBUTION = "institutional_distribution" # Mixed CE/PE patterns + Future Short Buildup
    RANGING_MARKET = "ranging_market"                        # Non-correlated patterns across all instruments
    VOLATILE_MARKET = "volatile_market"                      # Rapid pattern changes + high OI velocity


@dataclass
class OIPATrendingMetrics:
    """Data class for OI-PA trending analysis metrics."""
    
    # CE Side Analysis
    ce_pattern: CEOptionSellerPattern = CEOptionSellerPattern.CE_LONG_BUILDUP
    ce_pattern_strength: float = 0.0
    ce_pattern_confidence: float = 0.0
    
    # PE Side Analysis  
    pe_pattern: PEOptionSellerPattern = PEOptionSellerPattern.PE_LONG_BUILDUP
    pe_pattern_strength: float = 0.0
    pe_pattern_confidence: float = 0.0
    
    # Future Analysis
    future_pattern: FutureSellerPattern = FutureSellerPattern.FUTURE_LONG_BUILDUP
    future_pattern_strength: float = 0.0
    future_pattern_confidence: float = 0.0
    
    # 3-Way Correlation Matrix
    three_way_correlation: ThreeWayCorrelationPattern = ThreeWayCorrelationPattern.RANGING_SIDEWAYS_MARKET
    correlation_strength: float = 0.0
    correlation_consistency: float = 0.0
    
    # Cumulative OI-Price Correlations
    cumulative_ce_oi_price_correlation: float = 0.0
    cumulative_pe_oi_price_correlation: float = 0.0
    underlying_movement_correlation: float = 0.0
    
    # Comprehensive Market Regime
    market_regime: ComprehensiveMarketRegime = ComprehensiveMarketRegime.RANGING_MARKET
    regime_confidence: float = 0.0
    regime_transition_probability: float = 0.0
    
    # Supporting Metrics
    oi_momentum_score: float = 0.0
    volume_correlation_score: float = 0.0
    divergence_type: str = "no_divergence"
    range_expansion_score: float = 0.0
    institutional_flow_score: float = 0.0


class OIPATrendingEngine:
    """
    OI-PA Trending Classification Engine for comprehensive option seller analysis.
    
    Implements the complete option seller correlation framework with:
    - 4 CE patterns, 4 PE patterns, 4 Future patterns
    - 3-way correlation matrix analysis
    - 8-regime comprehensive market classification
    """
    
    def __init__(self, correlation_windows: List[int] = [10, 20, 50]):
        """
        Initialize the OI-PA Trending Engine.
        
        Args:
            correlation_windows: Rolling correlation windows for analysis [10, 20, 50] days
        """
        self.correlation_windows = correlation_windows
        
        # Pattern detection thresholds
        self.price_change_threshold = 0.5  # Minimum price change % for pattern detection
        self.oi_change_threshold = 5.0     # Minimum OI change % for pattern detection
        self.correlation_threshold = 0.3    # Minimum correlation for pattern strength
        
        # Historical data for lag/lead analysis
        self.historical_data = []
        self.correlation_history = {}
        
        logger.info(f"Initialized OIPATrendingEngine with correlation windows: {correlation_windows}")
    
    def analyze_oi_pa_trending(self, current_data: pd.DataFrame, 
                              previous_data: Optional[pd.DataFrame] = None,
                              underlying_price: Optional[float] = None,
                              future_oi: Optional[float] = None) -> OIPATrendingMetrics:
        """
        Analyze comprehensive OI-PA trending using option seller correlation framework.
        
        Args:
            current_data: Current period DataFrame with OI, volume, price data
            previous_data: Previous period DataFrame for change analysis (optional)
            underlying_price: Current underlying price for correlation analysis (optional)
            future_oi: Current future OI for 3-way correlation (optional)
            
        Returns:
            OIPATrendingMetrics with comprehensive analysis
        """
        logger.info("Analyzing comprehensive OI-PA trending patterns")
        
        try:
            # Initialize metrics
            metrics = OIPATrendingMetrics()
            
            # Calculate price and OI changes
            price_changes, oi_changes = self._calculate_price_oi_changes(current_data, previous_data)
            
            # 1. CE Side Option Seller Analysis (4 patterns)
            logger.info("Analyzing CE side option seller patterns")
            ce_analysis = self._analyze_ce_option_seller_patterns(
                price_changes, oi_changes, current_data
            )
            metrics.ce_pattern = ce_analysis['pattern']
            metrics.ce_pattern_strength = ce_analysis['strength']
            metrics.ce_pattern_confidence = ce_analysis['confidence']
            
            # 2. PE Side Option Seller Analysis (4 patterns)
            logger.info("Analyzing PE side option seller patterns") 
            pe_analysis = self._analyze_pe_option_seller_patterns(
                price_changes, oi_changes, current_data
            )
            metrics.pe_pattern = pe_analysis['pattern']
            metrics.pe_pattern_strength = pe_analysis['strength']
            metrics.pe_pattern_confidence = pe_analysis['confidence']
            
            # 3. Future OI Seller Analysis (if data available)
            logger.info("Analyzing future OI seller patterns")
            future_analysis = self._analyze_future_seller_patterns(
                underlying_price, future_oi, previous_data
            )
            metrics.future_pattern = future_analysis['pattern']
            metrics.future_pattern_strength = future_analysis['strength']
            metrics.future_pattern_confidence = future_analysis['confidence']
            
            # 4. 3-Way Correlation Matrix (CE+PE+Future)
            logger.info("Analyzing 3-way correlation matrix")
            correlation_analysis = self._analyze_three_way_correlation(
                ce_analysis, pe_analysis, future_analysis
            )
            metrics.three_way_correlation = correlation_analysis['pattern']
            metrics.correlation_strength = correlation_analysis['strength']
            metrics.correlation_consistency = correlation_analysis['consistency']
            
            # 5. Cumulative OI-Price Correlation Analysis
            logger.info("Analyzing cumulative OI-price correlations")
            cumulative_correlations = self._calculate_cumulative_oi_price_correlations(
                current_data, underlying_price
            )
            metrics.cumulative_ce_oi_price_correlation = cumulative_correlations['ce_correlation']
            metrics.cumulative_pe_oi_price_correlation = cumulative_correlations['pe_correlation']
            metrics.underlying_movement_correlation = cumulative_correlations['underlying_correlation']
            
            # 6. Comprehensive Market Regime Formation (8-regime classification)
            logger.info("Determining comprehensive market regime")
            regime_analysis = self._determine_comprehensive_market_regime(
                ce_analysis, pe_analysis, future_analysis, correlation_analysis
            )
            metrics.market_regime = regime_analysis['regime']
            metrics.regime_confidence = regime_analysis['confidence']
            metrics.regime_transition_probability = regime_analysis['transition_probability']
            
            # 7. Supporting Metrics
            supporting_metrics = self._calculate_supporting_trending_metrics(
                current_data, previous_data, metrics
            )
            metrics.oi_momentum_score = supporting_metrics['oi_momentum_score']
            metrics.volume_correlation_score = supporting_metrics['volume_correlation_score']
            metrics.divergence_type = supporting_metrics['divergence_type']
            metrics.range_expansion_score = supporting_metrics['range_expansion_score']
            metrics.institutional_flow_score = supporting_metrics['institutional_flow_score']
            
            # Store for historical analysis
            self._update_historical_data(metrics, current_data)
            
            logger.info(f"OI-PA trending analysis complete: "
                       f"Regime={metrics.market_regime.value}, "
                       f"CE={metrics.ce_pattern.value}, "
                       f"PE={metrics.pe_pattern.value}, "
                       f"Future={metrics.future_pattern.value}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"OI-PA trending analysis failed: {str(e)}")
            raise
    
    def analyze_correlation_windows(self, historical_data: List[pd.DataFrame],
                                  correlation_windows: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Analyze OI-price correlations across multiple time windows (10, 20, 50 days).
        
        Args:
            historical_data: List of historical DataFrames for rolling correlation analysis
            correlation_windows: Custom correlation windows (optional)
            
        Returns:
            Dictionary with correlation analysis across windows
        """
        if correlation_windows is None:
            correlation_windows = self.correlation_windows
            
        logger.info(f"Analyzing correlations across windows: {correlation_windows}")
        
        try:
            analysis = {
                'correlation_windows': correlation_windows,
                'window_analysis': {},
                'lag_lead_analysis': {},
                'correlation_trends': {},
                'institutional_implications': {}
            }
            
            # Prepare time series data
            if not historical_data:
                logger.warning("No historical data provided for correlation analysis")
                return analysis
            
            # Combine historical data
            combined_data = pd.concat(historical_data, ignore_index=True)
            
            if len(combined_data) < max(correlation_windows):
                logger.warning("Insufficient data for largest correlation window")
                return analysis
            
            # Calculate rolling correlations for each window
            for window in correlation_windows:
                if len(combined_data) >= window:
                    window_correlations = self._calculate_rolling_correlations(combined_data, window)
                    analysis['window_analysis'][f'{window}d'] = window_correlations
            
            # Lag/Lead Analysis: OI leading price vs price leading OI
            lag_lead_analysis = self._analyze_oi_price_lag_lead(combined_data)
            analysis['lag_lead_analysis'] = lag_lead_analysis
            
            # Correlation trends over time
            trends = self._analyze_correlation_trends(analysis['window_analysis'])
            analysis['correlation_trends'] = trends
            
            # Institutional implications based on correlation patterns
            implications = self._interpret_institutional_implications(analysis)
            analysis['institutional_implications'] = implications
            
            logger.info(f"Correlation window analysis complete: {len(correlation_windows)} windows analyzed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Correlation window analysis failed: {str(e)}")
            raise
    
    def detect_regime_transitions(self, current_metrics: OIPATrendingMetrics,
                                historical_metrics: List[OIPATrendingMetrics]) -> Dict[str, any]:
        """
        Detect market regime transitions using correlation breakdown patterns.
        
        Args:
            current_metrics: Current OI-PA trending metrics
            historical_metrics: Historical metrics for transition analysis
            
        Returns:
            Dictionary with regime transition analysis
        """
        logger.info("Detecting market regime transitions")
        
        try:
            transition_analysis = {
                'transition_detected': False,
                'transition_type': 'none',
                'transition_strength': 0.0,
                'previous_regime': None,
                'current_regime': current_metrics.market_regime,
                'transition_drivers': [],
                'stability_score': 0.0,
                'reversal_probability': 0.0
            }
            
            if len(historical_metrics) < 3:
                logger.warning("Insufficient historical data for transition detection")
                return transition_analysis
            
            # Analyze regime stability
            recent_regimes = [m.market_regime for m in historical_metrics[-5:]]
            regime_consistency = len(set(recent_regimes)) == 1
            
            # Check for regime change
            if len(recent_regimes) > 0:
                previous_regime = recent_regimes[-1]
                transition_analysis['previous_regime'] = previous_regime
                
                if current_metrics.market_regime != previous_regime:
                    transition_analysis['transition_detected'] = True
                    
                    # Analyze transition characteristics
                    transition_chars = self._analyze_transition_characteristics(
                        current_metrics, historical_metrics
                    )
                    
                    transition_analysis['transition_type'] = transition_chars['type']
                    transition_analysis['transition_strength'] = transition_chars['strength']
                    transition_analysis['transition_drivers'] = transition_chars['drivers']
            
            # Calculate stability score
            if len(historical_metrics) >= 5:
                recent_regimes = [m.market_regime for m in historical_metrics[-5:]]
                stability = recent_regimes.count(current_metrics.market_regime) / 5.0
                transition_analysis['stability_score'] = stability
            
            # Calculate reversal probability based on pattern analysis
            reversal_prob = self._calculate_reversal_probability(current_metrics, historical_metrics)
            transition_analysis['reversal_probability'] = reversal_prob
            
            logger.info(f"Regime transition analysis complete: "
                       f"Transition={transition_analysis['transition_detected']}, "
                       f"Type={transition_analysis['transition_type']}")
            
            return transition_analysis
            
        except Exception as e:
            logger.error(f"Regime transition detection failed: {str(e)}")
            raise
    
    # Private helper methods
    
    def _calculate_price_oi_changes(self, current_data: pd.DataFrame, 
                                  previous_data: Optional[pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate price and OI changes for pattern analysis."""
        
        # Current period aggregates
        current_ce_price = current_data['ce_close'].sum()
        current_pe_price = current_data['pe_close'].sum()
        current_ce_oi = current_data['ce_oi'].sum()
        current_pe_oi = current_data['pe_oi'].sum()
        
        if previous_data is not None:
            # Previous period aggregates
            previous_ce_price = previous_data['ce_close'].sum()
            previous_pe_price = previous_data['pe_close'].sum()
            previous_ce_oi = previous_data['ce_oi'].sum()
            previous_pe_oi = previous_data['pe_oi'].sum()
            
            # Calculate percentage changes
            price_changes = {
                'ce_price_change': ((current_ce_price - previous_ce_price) / previous_ce_price) * 100 if previous_ce_price > 0 else 0,
                'pe_price_change': ((current_pe_price - previous_pe_price) / previous_pe_price) * 100 if previous_pe_price > 0 else 0,
                'straddle_price_change': (((current_ce_price + current_pe_price) - (previous_ce_price + previous_pe_price)) / (previous_ce_price + previous_pe_price)) * 100 if (previous_ce_price + previous_pe_price) > 0 else 0
            }
            
            oi_changes = {
                'ce_oi_change': ((current_ce_oi - previous_ce_oi) / previous_ce_oi) * 100 if previous_ce_oi > 0 else 0,
                'pe_oi_change': ((current_pe_oi - previous_pe_oi) / previous_pe_oi) * 100 if previous_pe_oi > 0 else 0,
                'total_oi_change': (((current_ce_oi + current_pe_oi) - (previous_ce_oi + previous_pe_oi)) / (previous_ce_oi + previous_pe_oi)) * 100 if (previous_ce_oi + previous_pe_oi) > 0 else 0
            }
        else:
            # No previous data - use intra-period changes
            ce_price_change = current_data['ce_close'].pct_change().fillna(0).sum() * 100
            pe_price_change = current_data['pe_close'].pct_change().fillna(0).sum() * 100
            
            price_changes = {
                'ce_price_change': ce_price_change,
                'pe_price_change': pe_price_change,
                'straddle_price_change': (ce_price_change + pe_price_change) / 2
            }
            
            oi_changes = {
                'ce_oi_change': 0.0,  # Can't calculate without previous data
                'pe_oi_change': 0.0,
                'total_oi_change': 0.0
            }
        
        return price_changes, oi_changes
    
    def _analyze_ce_option_seller_patterns(self, price_changes: Dict[str, float], 
                                         oi_changes: Dict[str, float], 
                                         current_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze CE side option seller patterns (4 patterns as per story)."""
        
        ce_price_change = price_changes['ce_price_change']
        ce_oi_change = oi_changes['ce_oi_change']
        
        # Determine pattern based on price and OI directions
        if ce_price_change <= -self.price_change_threshold and ce_oi_change >= self.oi_change_threshold:
            # Price DOWN + CE_OI UP = SHORT BUILDUP (bearish sentiment, call writers selling calls)
            pattern = CEOptionSellerPattern.CE_SHORT_BUILDUP
            strength = min(abs(ce_price_change), abs(ce_oi_change)) / 10.0
            confidence = 0.8
            
        elif ce_price_change >= self.price_change_threshold and ce_oi_change <= -self.oi_change_threshold:
            # Price UP + CE_OI DOWN = SHORT COVERING (call writers buying back calls)
            pattern = CEOptionSellerPattern.CE_SHORT_COVERING
            strength = min(abs(ce_price_change), abs(ce_oi_change)) / 10.0
            confidence = 0.8
            
        elif ce_price_change >= self.price_change_threshold and ce_oi_change >= self.oi_change_threshold:
            # Price UP + CE_OI UP = LONG BUILDUP (bullish sentiment, call buyers buying calls)
            pattern = CEOptionSellerPattern.CE_LONG_BUILDUP
            strength = min(abs(ce_price_change), abs(ce_oi_change)) / 10.0
            confidence = 0.9
            
        elif ce_price_change <= -self.price_change_threshold and ce_oi_change <= -self.oi_change_threshold:
            # Price DOWN + CE_OI DOWN = LONG UNWINDING (call buyers selling calls)
            pattern = CEOptionSellerPattern.CE_LONG_UNWINDING
            strength = min(abs(ce_price_change), abs(ce_oi_change)) / 10.0
            confidence = 0.9
            
        else:
            # Default to most likely pattern based on available data
            if ce_oi_change > 0:
                pattern = CEOptionSellerPattern.CE_LONG_BUILDUP
            else:
                pattern = CEOptionSellerPattern.CE_LONG_UNWINDING
            strength = 0.1
            confidence = 0.3
        
        # Validate pattern strength with volume data
        if 'ce_volume' in current_data.columns:
            total_volume = current_data['ce_volume'].sum()
            volume_support = min(1.0, total_volume / 10000)  # Normalize volume support
            confidence *= (0.7 + 0.3 * volume_support)  # Adjust confidence based on volume
        
        return {
            'pattern': pattern,
            'strength': min(1.0, strength),
            'confidence': min(1.0, confidence),
            'ce_price_change': ce_price_change,
            'ce_oi_change': ce_oi_change
        }
    
    def _analyze_pe_option_seller_patterns(self, price_changes: Dict[str, float], 
                                         oi_changes: Dict[str, float], 
                                         current_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze PE side option seller patterns (4 patterns as per story)."""
        
        pe_price_change = price_changes['pe_price_change']
        pe_oi_change = oi_changes['pe_oi_change']
        straddle_price_change = price_changes['straddle_price_change']  # Use underlying movement proxy
        
        # Determine pattern based on underlying price movement and PE OI changes
        if straddle_price_change >= self.price_change_threshold and pe_oi_change >= self.oi_change_threshold:
            # Price UP + PE_OI UP = SHORT BUILDUP (bullish underlying, put writers selling puts)
            pattern = PEOptionSellerPattern.PE_SHORT_BUILDUP
            strength = min(abs(straddle_price_change), abs(pe_oi_change)) / 10.0
            confidence = 0.8
            
        elif straddle_price_change <= -self.price_change_threshold and pe_oi_change <= -self.oi_change_threshold:
            # Price DOWN + PE_OI DOWN = SHORT COVERING (put writers buying back puts)
            pattern = PEOptionSellerPattern.PE_SHORT_COVERING
            strength = min(abs(straddle_price_change), abs(pe_oi_change)) / 10.0
            confidence = 0.8
            
        elif straddle_price_change <= -self.price_change_threshold and pe_oi_change >= self.oi_change_threshold:
            # Price DOWN + PE_OI UP = LONG BUILDUP (bearish sentiment, put buyers buying puts)
            pattern = PEOptionSellerPattern.PE_LONG_BUILDUP
            strength = min(abs(straddle_price_change), abs(pe_oi_change)) / 10.0
            confidence = 0.9
            
        elif straddle_price_change >= self.price_change_threshold and pe_oi_change <= -self.oi_change_threshold:
            # Price UP + PE_OI DOWN = LONG UNWINDING (put buyers selling puts)
            pattern = PEOptionSellerPattern.PE_LONG_UNWINDING
            strength = min(abs(straddle_price_change), abs(pe_oi_change)) / 10.0
            confidence = 0.9
            
        else:
            # Default pattern based on available data
            if pe_oi_change > 0:
                pattern = PEOptionSellerPattern.PE_LONG_BUILDUP
            else:
                pattern = PEOptionSellerPattern.PE_LONG_UNWINDING
            strength = 0.1
            confidence = 0.3
        
        # Validate pattern strength with volume data
        if 'pe_volume' in current_data.columns:
            total_volume = current_data['pe_volume'].sum()
            volume_support = min(1.0, total_volume / 10000)  # Normalize volume support
            confidence *= (0.7 + 0.3 * volume_support)  # Adjust confidence based on volume
        
        return {
            'pattern': pattern,
            'strength': min(1.0, strength),
            'confidence': min(1.0, confidence),
            'pe_price_change': pe_price_change,
            'pe_oi_change': pe_oi_change,
            'underlying_price_change': straddle_price_change
        }
    
    def _analyze_future_seller_patterns(self, underlying_price: Optional[float], 
                                       future_oi: Optional[float],
                                       previous_data: Optional[pd.DataFrame]) -> Dict[str, any]:
        """Analyze Future OI seller patterns with underlying correlation (4 patterns as per story)."""
        
        # Default result if insufficient data
        default_result = {
            'pattern': FutureSellerPattern.FUTURE_LONG_BUILDUP,
            'strength': 0.0,
            'confidence': 0.0,
            'underlying_price_change': 0.0,
            'future_oi_change': 0.0
        }
        
        if underlying_price is None or future_oi is None:
            logger.warning("Future analysis requires underlying_price and future_oi data")
            return default_result
        
        # Calculate changes if previous data available
        if previous_data is not None:
            if 'spot' in previous_data.columns:
                previous_underlying = previous_data['spot'].iloc[-1]
                underlying_price_change = ((underlying_price - previous_underlying) / previous_underlying) * 100
            else:
                underlying_price_change = 0.0
                
            # Future OI change calculation (would need previous future OI)
            future_oi_change = 0.0  # Placeholder - needs historical future OI data
        else:
            underlying_price_change = 0.0
            future_oi_change = 0.0
        
        # Determine future seller pattern based on price and OI movements
        if underlying_price_change >= self.price_change_threshold and future_oi_change >= self.oi_change_threshold:
            # Price UP + FUTURE_OI UP = LONG BUILDUP (bullish sentiment, future buyers)
            pattern = FutureSellerPattern.FUTURE_LONG_BUILDUP
            strength = min(abs(underlying_price_change), abs(future_oi_change)) / 10.0
            confidence = 0.8
            
        elif underlying_price_change <= -self.price_change_threshold and future_oi_change <= -self.oi_change_threshold:
            # Price DOWN + FUTURE_OI DOWN = LONG UNWINDING (future buyers closing positions)
            pattern = FutureSellerPattern.FUTURE_LONG_UNWINDING
            strength = min(abs(underlying_price_change), abs(future_oi_change)) / 10.0
            confidence = 0.8
            
        elif underlying_price_change <= -self.price_change_threshold and future_oi_change >= self.oi_change_threshold:
            # Price DOWN + FUTURE_OI UP = SHORT BUILDUP (bearish sentiment, future sellers)
            pattern = FutureSellerPattern.FUTURE_SHORT_BUILDUP
            strength = min(abs(underlying_price_change), abs(future_oi_change)) / 10.0
            confidence = 0.9
            
        elif underlying_price_change >= self.price_change_threshold and future_oi_change <= -self.oi_change_threshold:
            # Price UP + FUTURE_OI DOWN = SHORT COVERING (future sellers covering positions)
            pattern = FutureSellerPattern.FUTURE_SHORT_COVERING
            strength = min(abs(underlying_price_change), abs(future_oi_change)) / 10.0
            confidence = 0.9
            
        else:
            # Default based on underlying price movement
            if underlying_price_change > 0:
                pattern = FutureSellerPattern.FUTURE_LONG_BUILDUP
            else:
                pattern = FutureSellerPattern.FUTURE_SHORT_BUILDUP
            strength = abs(underlying_price_change) / 10.0
            confidence = 0.5
        
        return {
            'pattern': pattern,
            'strength': min(1.0, strength),
            'confidence': min(1.0, confidence),
            'underlying_price_change': underlying_price_change,
            'future_oi_change': future_oi_change
        }
    
    def _analyze_three_way_correlation(self, ce_analysis: Dict[str, any], 
                                     pe_analysis: Dict[str, any],
                                     future_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze 3-way correlation matrix (CE+PE+Future) as per story requirements."""
        
        ce_pattern = ce_analysis['pattern']
        pe_pattern = pe_analysis['pattern']  
        future_pattern = future_analysis['pattern']
        
        # Determine 3-way correlation pattern based on pattern combinations
        
        # STRONG BULLISH CORRELATION = CE Long Buildup + PE Short Buildup + Future Long Buildup
        if (ce_pattern == CEOptionSellerPattern.CE_LONG_BUILDUP and
            pe_pattern == PEOptionSellerPattern.PE_SHORT_BUILDUP and
            future_pattern == FutureSellerPattern.FUTURE_LONG_BUILDUP):
            correlation_pattern = ThreeWayCorrelationPattern.STRONG_BULLISH_CORRELATION
            strength = (ce_analysis['strength'] + pe_analysis['strength'] + future_analysis['strength']) / 3.0
            consistency = min(ce_analysis['confidence'], pe_analysis['confidence'], future_analysis['confidence'])
        
        # STRONG BEARISH CORRELATION = CE Short Buildup + PE Long Buildup + Future Short Buildup  
        elif (ce_pattern == CEOptionSellerPattern.CE_SHORT_BUILDUP and
              pe_pattern == PEOptionSellerPattern.PE_LONG_BUILDUP and
              future_pattern == FutureSellerPattern.FUTURE_SHORT_BUILDUP):
            correlation_pattern = ThreeWayCorrelationPattern.STRONG_BEARISH_CORRELATION
            strength = (ce_analysis['strength'] + pe_analysis['strength'] + future_analysis['strength']) / 3.0
            consistency = min(ce_analysis['confidence'], pe_analysis['confidence'], future_analysis['confidence'])
        
        # INSTITUTIONAL POSITIONING = Mixed patterns across CE/PE/Future (hedging/arbitrage)
        elif self._is_mixed_institutional_pattern(ce_pattern, pe_pattern, future_pattern):
            correlation_pattern = ThreeWayCorrelationPattern.INSTITUTIONAL_POSITIONING
            strength = np.std([ce_analysis['strength'], pe_analysis['strength'], future_analysis['strength']])
            consistency = (ce_analysis['confidence'] + pe_analysis['confidence'] + future_analysis['confidence']) / 3.0
        
        # RANGING/SIDEWAYS MARKET = Non-aligned patterns across instruments
        elif self._is_non_aligned_pattern(ce_pattern, pe_pattern, future_pattern):
            correlation_pattern = ThreeWayCorrelationPattern.RANGING_SIDEWAYS_MARKET
            strength = 0.3  # Low strength for ranging market
            consistency = 0.5  # Moderate consistency
        
        # TRANSITION/REVERSAL SETUP = Correlation breakdown between instruments
        elif self._is_reversal_setup_pattern(ce_pattern, pe_pattern, future_pattern):
            correlation_pattern = ThreeWayCorrelationPattern.TRANSITION_REVERSAL_SETUP
            strength = 0.7  # High strength for reversal setup
            consistency = 0.6  # Moderate to high consistency
        
        # ARBITRAGE/COMPLEX STRATEGY = Opposite positioning patterns
        elif self._is_arbitrage_pattern(ce_pattern, pe_pattern, future_pattern):
            correlation_pattern = ThreeWayCorrelationPattern.ARBITRAGE_COMPLEX_STRATEGY
            strength = (ce_analysis['strength'] + pe_analysis['strength'] + future_analysis['strength']) / 3.0
            consistency = 0.8  # High consistency for arbitrage
        
        else:
            # Default to ranging market
            correlation_pattern = ThreeWayCorrelationPattern.RANGING_SIDEWAYS_MARKET
            strength = 0.2
            consistency = 0.3
        
        return {
            'pattern': correlation_pattern,
            'strength': min(1.0, strength),
            'consistency': min(1.0, consistency),
            'ce_pattern': ce_pattern,
            'pe_pattern': pe_pattern,
            'future_pattern': future_pattern
        }
    
    def _calculate_cumulative_oi_price_correlations(self, current_data: pd.DataFrame,
                                                   underlying_price: Optional[float]) -> Dict[str, float]:
        """Calculate cumulative OI-price correlations across ATM ±7 strikes."""
        
        if len(current_data) < 2:
            return {
                'ce_correlation': 0.0,
                'pe_correlation': 0.0,
                'underlying_correlation': 0.0
            }
        
        # Calculate correlations between cumulative OI and cumulative prices
        ce_oi_price_corr = current_data['ce_oi'].corr(current_data['ce_close'])
        pe_oi_price_corr = current_data['pe_oi'].corr(current_data['pe_close'])
        
        # Calculate underlying movement correlation if available
        underlying_correlation = 0.0
        if underlying_price is not None and 'spot' in current_data.columns:
            total_oi = current_data['ce_oi'] + current_data['pe_oi']
            # Use correlation with spot price as proxy for underlying movement
            underlying_corr = total_oi.corr(current_data['spot'])
            underlying_correlation = underlying_corr if not pd.isna(underlying_corr) else 0.0
        
        return {
            'ce_correlation': ce_oi_price_corr if not pd.isna(ce_oi_price_corr) else 0.0,
            'pe_correlation': pe_oi_price_corr if not pd.isna(pe_oi_price_corr) else 0.0,
            'underlying_correlation': underlying_correlation
        }
    
    def _determine_comprehensive_market_regime(self, ce_analysis: Dict[str, any], 
                                             pe_analysis: Dict[str, any],
                                             future_analysis: Dict[str, any],
                                             correlation_analysis: Dict[str, any]) -> Dict[str, any]:
        """Determine comprehensive market regime (8-regime classification)."""
        
        correlation_pattern = correlation_analysis['pattern']
        correlation_strength = correlation_analysis['strength']
        
        # Map 3-way correlations to market regimes
        if correlation_pattern == ThreeWayCorrelationPattern.STRONG_BULLISH_CORRELATION:
            market_regime = ComprehensiveMarketRegime.TRENDING_BULLISH
            confidence = correlation_strength * 0.9
            
        elif correlation_pattern == ThreeWayCorrelationPattern.STRONG_BEARISH_CORRELATION:
            market_regime = ComprehensiveMarketRegime.TRENDING_BEARISH
            confidence = correlation_strength * 0.9
            
        elif correlation_pattern == ThreeWayCorrelationPattern.TRANSITION_REVERSAL_SETUP:
            # Determine bullish vs bearish reversal based on patterns
            if (ce_analysis['pattern'] == CEOptionSellerPattern.CE_SHORT_COVERING or
                pe_analysis['pattern'] == PEOptionSellerPattern.PE_LONG_UNWINDING):
                market_regime = ComprehensiveMarketRegime.BULLISH_REVERSAL_SETUP
            else:
                market_regime = ComprehensiveMarketRegime.BEARISH_REVERSAL_SETUP
            confidence = correlation_strength * 0.8
            
        elif correlation_pattern == ThreeWayCorrelationPattern.INSTITUTIONAL_POSITIONING:
            # Determine accumulation vs distribution based on future pattern
            if future_analysis['pattern'] in [FutureSellerPattern.FUTURE_LONG_BUILDUP]:
                market_regime = ComprehensiveMarketRegime.INSTITUTIONAL_ACCUMULATION
            elif future_analysis['pattern'] in [FutureSellerPattern.FUTURE_SHORT_BUILDUP]:
                market_regime = ComprehensiveMarketRegime.INSTITUTIONAL_DISTRIBUTION
            else:
                market_regime = ComprehensiveMarketRegime.RANGING_MARKET
            confidence = correlation_strength * 0.7
            
        elif correlation_pattern == ThreeWayCorrelationPattern.RANGING_SIDEWAYS_MARKET:
            market_regime = ComprehensiveMarketRegime.RANGING_MARKET
            confidence = 0.6
            
        elif correlation_pattern == ThreeWayCorrelationPattern.ARBITRAGE_COMPLEX_STRATEGY:
            # High velocity patterns suggest volatile market
            if (ce_analysis['strength'] > 0.7 and pe_analysis['strength'] > 0.7):
                market_regime = ComprehensiveMarketRegime.VOLATILE_MARKET
            else:
                market_regime = ComprehensiveMarketRegime.RANGING_MARKET
            confidence = correlation_strength * 0.6
            
        else:
            market_regime = ComprehensiveMarketRegime.RANGING_MARKET
            confidence = 0.5
        
        # Calculate transition probability based on pattern stability
        transition_probability = 1.0 - confidence
        
        return {
            'regime': market_regime,
            'confidence': min(1.0, confidence),
            'transition_probability': min(1.0, transition_probability)
        }
    
    def _calculate_supporting_trending_metrics(self, current_data: pd.DataFrame,
                                             previous_data: Optional[pd.DataFrame],
                                             metrics: OIPATrendingMetrics) -> Dict[str, any]:
        """Calculate supporting metrics for trending analysis."""
        
        # OI momentum score based on velocity
        total_oi = current_data['ce_oi'] + current_data['pe_oi']
        if previous_data is not None:
            prev_total_oi = previous_data['ce_oi'] + previous_data['pe_oi']
            oi_momentum = (total_oi.sum() - prev_total_oi.sum()) / prev_total_oi.sum()
        else:
            oi_momentum = 0.0
        oi_momentum_score = min(1.0, abs(oi_momentum) * 10)
        
        # Volume correlation score
        total_volume = current_data['ce_volume'] + current_data['pe_volume']
        if len(current_data) > 1:
            volume_oi_corr = total_volume.corr(total_oi)
            volume_correlation_score = abs(volume_oi_corr) if not pd.isna(volume_oi_corr) else 0.0
        else:
            volume_correlation_score = 0.0
        
        # Divergence type based on correlation patterns
        if metrics.cumulative_ce_oi_price_correlation < -0.3:
            divergence_type = "bearish_divergence"
        elif metrics.cumulative_pe_oi_price_correlation < -0.3:
            divergence_type = "bullish_divergence"
        elif abs(metrics.cumulative_ce_oi_price_correlation) < 0.1 and abs(metrics.cumulative_pe_oi_price_correlation) < 0.1:
            divergence_type = "institutional_hedging"
        else:
            divergence_type = "no_divergence"
        
        # Range expansion score (placeholder - would need historical volatility data)
        range_expansion_score = min(1.0, abs(metrics.cumulative_ce_oi_price_correlation + metrics.cumulative_pe_oi_price_correlation))
        
        # Institutional flow score based on pattern analysis
        pattern_scores = [metrics.ce_pattern_confidence, metrics.pe_pattern_confidence, metrics.future_pattern_confidence]
        institutional_flow_score = sum(pattern_scores) / len(pattern_scores)
        
        return {
            'oi_momentum_score': min(1.0, oi_momentum_score),
            'volume_correlation_score': min(1.0, volume_correlation_score),
            'divergence_type': divergence_type,
            'range_expansion_score': min(1.0, range_expansion_score),
            'institutional_flow_score': min(1.0, institutional_flow_score)
        }
    
    def _update_historical_data(self, metrics: OIPATrendingMetrics, current_data: pd.DataFrame):
        """Update historical data for pattern learning and analysis."""
        
        historical_entry = {
            'timestamp': datetime.now(),
            'market_regime': metrics.market_regime,
            'ce_pattern': metrics.ce_pattern,
            'pe_pattern': metrics.pe_pattern,
            'future_pattern': metrics.future_pattern,
            'three_way_correlation': metrics.three_way_correlation,
            'regime_confidence': metrics.regime_confidence,
            'data_size': len(current_data)
        }
        
        self.historical_data.append(historical_entry)
        
        # Keep only recent history
        if len(self.historical_data) > 200:
            self.historical_data = self.historical_data[-200:]
    
    # Pattern classification helper methods
    
    def _is_mixed_institutional_pattern(self, ce_pattern: CEOptionSellerPattern, 
                                       pe_pattern: PEOptionSellerPattern,
                                       future_pattern: FutureSellerPattern) -> bool:
        """Check if patterns indicate mixed institutional positioning (hedging/arbitrage)."""
        
        # Hedging patterns: opposite directional exposure
        hedging_patterns = [
            (ce_pattern == CEOptionSellerPattern.CE_LONG_BUILDUP and pe_pattern == PEOptionSellerPattern.PE_LONG_BUILDUP),
            (ce_pattern == CEOptionSellerPattern.CE_SHORT_BUILDUP and pe_pattern == PEOptionSellerPattern.PE_SHORT_BUILDUP)
        ]
        
        return any(hedging_patterns)
    
    def _is_non_aligned_pattern(self, ce_pattern: CEOptionSellerPattern,
                               pe_pattern: PEOptionSellerPattern, 
                               future_pattern: FutureSellerPattern) -> bool:
        """Check if patterns are non-aligned (ranging/sideways market)."""
        
        # Patterns that don't align directionally
        non_aligned = [
            (ce_pattern == CEOptionSellerPattern.CE_LONG_BUILDUP and 
             pe_pattern == PEOptionSellerPattern.PE_LONG_BUILDUP and
             future_pattern == FutureSellerPattern.FUTURE_SHORT_BUILDUP),
            (ce_pattern == CEOptionSellerPattern.CE_SHORT_BUILDUP and
             pe_pattern == PEOptionSellerPattern.PE_SHORT_BUILDUP and
             future_pattern == FutureSellerPattern.FUTURE_LONG_BUILDUP)
        ]
        
        return any(non_aligned)
    
    def _is_reversal_setup_pattern(self, ce_pattern: CEOptionSellerPattern,
                                  pe_pattern: PEOptionSellerPattern,
                                  future_pattern: FutureSellerPattern) -> bool:
        """Check if patterns indicate reversal setup."""
        
        # Covering/unwinding patterns suggest reversal
        reversal_patterns = [
            ce_pattern in [CEOptionSellerPattern.CE_SHORT_COVERING, CEOptionSellerPattern.CE_LONG_UNWINDING],
            pe_pattern in [PEOptionSellerPattern.PE_SHORT_COVERING, PEOptionSellerPattern.PE_LONG_UNWINDING],
            future_pattern in [FutureSellerPattern.FUTURE_SHORT_COVERING, FutureSellerPattern.FUTURE_LONG_UNWINDING]
        ]
        
        return sum(reversal_patterns) >= 2  # At least 2 out of 3 showing reversal
    
    def _is_arbitrage_pattern(self, ce_pattern: CEOptionSellerPattern,
                             pe_pattern: PEOptionSellerPattern,
                             future_pattern: FutureSellerPattern) -> bool:
        """Check if patterns indicate arbitrage/complex strategy."""
        
        # Opposite positioning patterns across instruments
        arbitrage_patterns = [
            (ce_pattern == CEOptionSellerPattern.CE_LONG_BUILDUP and 
             pe_pattern == PEOptionSellerPattern.PE_SHORT_BUILDUP and
             future_pattern == FutureSellerPattern.FUTURE_SHORT_BUILDUP),
            (ce_pattern == CEOptionSellerPattern.CE_SHORT_BUILDUP and
             pe_pattern == PEOptionSellerPattern.PE_LONG_BUILDUP and  
             future_pattern == FutureSellerPattern.FUTURE_LONG_BUILDUP)
        ]
        
        return any(arbitrage_patterns)
    
    def _calculate_rolling_correlations(self, data: pd.DataFrame, window: int) -> Dict[str, float]:
        """Calculate rolling correlations for given window."""
        
        if len(data) < window:
            return {'ce_oi_price': 0.0, 'pe_oi_price': 0.0, 'underlying': 0.0}
        
        # Calculate rolling correlations
        ce_corr = data['ce_oi'].rolling(window=window).corr(data['ce_close']).iloc[-1]
        pe_corr = data['pe_oi'].rolling(window=window).corr(data['pe_close']).iloc[-1]
        
        underlying_corr = 0.0
        if 'spot' in data.columns:
            total_oi = data['ce_oi'] + data['pe_oi']
            underlying_corr = total_oi.rolling(window=window).corr(data['spot']).iloc[-1]
        
        return {
            'ce_oi_price': ce_corr if not pd.isna(ce_corr) else 0.0,
            'pe_oi_price': pe_corr if not pd.isna(pe_corr) else 0.0,
            'underlying': underlying_corr if not pd.isna(underlying_corr) else 0.0
        }
    
    def _analyze_oi_price_lag_lead(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze OI leading price vs price leading OI correlation patterns."""
        
        if len(data) < 10:
            return {'oi_leads_price': False, 'price_leads_oi': False, 'lag_periods': 0}
        
        # Calculate price and OI changes
        price_changes = (data['ce_close'] + data['pe_close']).pct_change().fillna(0)
        oi_changes = (data['ce_oi'] + data['pe_oi']).pct_change().fillna(0)
        
        # Test different lag periods
        max_lag = min(5, len(data) // 3)
        best_correlation = 0.0
        best_lag = 0
        oi_leads = False
        
        for lag in range(1, max_lag + 1):
            # Test OI leading price
            if len(oi_changes) > lag:
                oi_lead_corr = oi_changes[:-lag].corr(price_changes[lag:])
                price_lead_corr = price_changes[:-lag].corr(oi_changes[lag:])
                
                if abs(oi_lead_corr) > abs(best_correlation):
                    best_correlation = oi_lead_corr
                    best_lag = lag
                    oi_leads = True
                
                if abs(price_lead_corr) > abs(best_correlation):
                    best_correlation = price_lead_corr
                    best_lag = lag
                    oi_leads = False
        
        return {
            'oi_leads_price': oi_leads and abs(best_correlation) > 0.3,
            'price_leads_oi': not oi_leads and abs(best_correlation) > 0.3,
            'lag_periods': best_lag,
            'correlation': best_correlation
        }
    
    def _analyze_correlation_trends(self, window_analysis: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Analyze correlation trends across different time windows."""
        
        trends = {}
        
        for metric in ['ce_oi_price', 'pe_oi_price', 'underlying']:
            values = []
            windows = []
            
            for window_name, correlations in window_analysis.items():
                if metric in correlations:
                    values.append(correlations[metric])
                    windows.append(int(window_name.replace('d', '')))
            
            if len(values) >= 2:
                # Determine trend (strengthening/weakening over longer windows)
                short_term = values[0] if windows[0] < windows[-1] else values[-1]
                long_term = values[-1] if windows[0] < windows[-1] else values[0]
                
                if long_term - short_term > 0.1:
                    trends[metric] = 'strengthening'
                elif long_term - short_term < -0.1:
                    trends[metric] = 'weakening'
                else:
                    trends[metric] = 'stable'
            else:
                trends[metric] = 'insufficient_data'
        
        return trends
    
    def _interpret_institutional_implications(self, analysis: Dict[str, any]) -> Dict[str, str]:
        """Interpret institutional implications based on correlation patterns."""
        
        implications = {}
        
        # Analyze window patterns
        if 'window_analysis' in analysis:
            window_data = analysis['window_analysis']
            
            # Look for institutional patterns
            if len(window_data) >= 2:
                short_correlations = list(window_data.values())[0]
                long_correlations = list(window_data.values())[-1]
                
                # Institutional accumulation: stronger correlations in longer windows
                if (long_correlations.get('ce_oi_price', 0) > short_correlations.get('ce_oi_price', 0) and
                    long_correlations.get('pe_oi_price', 0) > short_correlations.get('pe_oi_price', 0)):
                    implications['pattern_type'] = 'institutional_accumulation'
                    implications['strategy'] = 'systematic_positioning'
                elif (long_correlations.get('ce_oi_price', 0) < short_correlations.get('ce_oi_price', 0) and
                      long_correlations.get('pe_oi_price', 0) < short_correlations.get('pe_oi_price', 0)):
                    implications['pattern_type'] = 'institutional_distribution'
                    implications['strategy'] = 'systematic_unwinding'
                else:
                    implications['pattern_type'] = 'mixed_activity'
                    implications['strategy'] = 'complex_positioning'
        
        # Analyze lag/lead patterns
        if 'lag_lead_analysis' in analysis:
            lag_data = analysis['lag_lead_analysis']
            
            if lag_data.get('oi_leads_price', False):
                implications['flow_direction'] = 'oi_driven'
                implications['market_efficiency'] = 'institutional_information'
            elif lag_data.get('price_leads_oi', False):
                implications['flow_direction'] = 'price_driven'
                implications['market_efficiency'] = 'retail_following'
            else:
                implications['flow_direction'] = 'synchronized'
                implications['market_efficiency'] = 'efficient_market'
        
        return implications
    
    def _analyze_transition_characteristics(self, current_metrics: OIPATrendingMetrics,
                                          historical_metrics: List[OIPATrendingMetrics]) -> Dict[str, any]:
        """Analyze regime transition characteristics."""
        
        # Identify transition drivers
        drivers = []
        
        if len(historical_metrics) > 0:
            prev_metrics = historical_metrics[-1]
            
            # Check what changed
            if current_metrics.ce_pattern != prev_metrics.ce_pattern:
                drivers.append('ce_pattern_shift')
            if current_metrics.pe_pattern != prev_metrics.pe_pattern:
                drivers.append('pe_pattern_shift')
            if current_metrics.future_pattern != prev_metrics.future_pattern:
                drivers.append('future_pattern_shift')
            if current_metrics.three_way_correlation != prev_metrics.three_way_correlation:
                drivers.append('correlation_breakdown')
        
        # Determine transition type
        if 'correlation_breakdown' in drivers:
            transition_type = 'correlation_driven'
        elif len(drivers) >= 2:
            transition_type = 'multi_pattern_shift'
        elif len(drivers) == 1:
            transition_type = 'single_pattern_shift'
        else:
            transition_type = 'confidence_driven'
        
        # Calculate transition strength
        strength = min(1.0, len(drivers) / 3.0) * current_metrics.regime_confidence
        
        return {
            'type': transition_type,
            'strength': strength,
            'drivers': drivers
        }
    
    def _calculate_reversal_probability(self, current_metrics: OIPATrendingMetrics,
                                       historical_metrics: List[OIPATrendingMetrics]) -> float:
        """Calculate probability of regime reversal based on pattern analysis."""
        
        if len(historical_metrics) < 3:
            return 0.5  # Default probability
        
        # Analyze recent regime stability
        recent_regimes = [m.market_regime for m in historical_metrics[-3:]]
        regime_changes = len(set(recent_regimes))
        
        # High regime instability suggests higher reversal probability
        instability_score = regime_changes / 3.0
        
        # Check for reversal setup patterns
        reversal_indicators = 0
        if current_metrics.three_way_correlation == ThreeWayCorrelationPattern.TRANSITION_REVERSAL_SETUP:
            reversal_indicators += 1
        
        if current_metrics.ce_pattern in [CEOptionSellerPattern.CE_SHORT_COVERING, CEOptionSellerPattern.CE_LONG_UNWINDING]:
            reversal_indicators += 1
            
        if current_metrics.pe_pattern in [PEOptionSellerPattern.PE_SHORT_COVERING, PEOptionSellerPattern.PE_LONG_UNWINDING]:
            reversal_indicators += 1
        
        # Calculate reversal probability
        reversal_score = reversal_indicators / 3.0
        
        # Combine instability and reversal indicators
        reversal_probability = (instability_score * 0.4 + reversal_score * 0.6)
        
        return min(1.0, max(0.0, reversal_probability))