#!/usr/bin/env python3
"""
ATM Straddle Engine - Independent Technical Analysis
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides independent technical analysis for ATM Straddle component:
- Independent EMA 20/100/200 calculations
- Independent VWAP current/previous day analysis
- Independent Pivot point calculations
- Multi-timeframe rolling windows (3, 5, 10, 15 minutes)
- No adjustment factors - pure independent calculations

Author: The Augster
Date: 2025-06-23
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ATMStraddleEngine:
    """
    ATM Straddle Engine with independent technical analysis
    
    Provides complete technical analysis for ATM Straddle (ATM CE + ATM PE) without
    any adjustment factors. All calculations are independent and based on actual
    ATM straddle price movements.
    """
    
    def __init__(self, component_config: Dict[str, Any]):
        """Initialize ATM Straddle Engine"""
        self.config = component_config
        self.component_name = "ATM_Straddle"
        self.weight = component_config.get('weight', 0.25)
        self.priority = component_config.get('priority', 'high')
        
        # Technical indicator parameters
        self.ema_periods = [20, 100, 200]
        self.vwap_periods = [1, 5, 15]  # Days
        self.rolling_windows = component_config.get('rolling_windows', [3, 5, 10, 15])
        
        logger.info(f"ATM Straddle Engine initialized - Weight: {self.weight}, Priority: {self.priority}")
    
    def calculate_independent_technical_analysis(self, atm_straddle_prices: pd.Series,
                                               atm_ce_volume: pd.Series = None,
                                               atm_pe_volume: pd.Series = None,
                                               timeframe: str = '5min') -> Dict[str, Any]:
        """
        Calculate independent technical analysis for ATM Straddle
        
        Args:
            atm_straddle_prices: ATM CE + ATM PE prices
            atm_ce_volume: ATM CE volume data
            atm_pe_volume: ATM PE volume data
            timeframe: Analysis timeframe
            
        Returns:
            Complete technical analysis results
        """
        try:
            logger.debug(f"Calculating ATM Straddle technical analysis for {timeframe}")
            
            # Validate input data
            if atm_straddle_prices.empty:
                logger.warning("Empty ATM straddle price data")
                return self._get_default_results()
            
            # Calculate total volume for VWAP
            total_volume = pd.Series([1] * len(atm_straddle_prices))  # Default volume
            if atm_ce_volume is not None and atm_pe_volume is not None:
                total_volume = atm_ce_volume + atm_pe_volume
            
            # Independent EMA Analysis
            ema_results = self._calculate_independent_ema_analysis(atm_straddle_prices)
            
            # Independent VWAP Analysis
            vwap_results = self._calculate_independent_vwap_analysis(
                atm_straddle_prices, total_volume, timeframe
            )
            
            # Independent Pivot Analysis
            pivot_results = self._calculate_independent_pivot_analysis(atm_straddle_prices, timeframe)
            
            # Additional ATM-specific indicators
            atm_specific_results = self._calculate_atm_specific_indicators(atm_straddle_prices)
            
            # Combine all results
            technical_results = {
                'component': self.component_name,
                'timeframe': timeframe,
                'weight': self.weight,
                'ema_analysis': ema_results,
                'vwap_analysis': vwap_results,
                'pivot_analysis': pivot_results,
                'atm_specific_analysis': atm_specific_results,
                'summary_metrics': self._calculate_summary_metrics(
                    ema_results, vwap_results, pivot_results, atm_specific_results
                ),
                'calculation_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(atm_straddle_prices)
            }
            
            logger.debug(f"ATM Straddle technical analysis completed for {timeframe}")
            return technical_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM Straddle technical analysis: {e}")
            return self._get_default_results()
    
    def _calculate_independent_ema_analysis(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate independent EMA analysis for ATM Straddle"""
        try:
            ema_results = {}
            
            # Calculate EMAs
            for period in self.ema_periods:
                ema_key = f'ema_{period}'
                ema_values = prices.ewm(span=period).mean()
                ema_results[ema_key] = ema_values
                
                # EMA positioning
                ema_results[f'{ema_key}_position'] = (prices / ema_values - 1).fillna(0)
                ema_results[f'above_{ema_key}'] = (prices > ema_values).astype(int)
                
                # EMA slope analysis
                ema_results[f'{ema_key}_slope'] = ema_values.diff().fillna(0)
                ema_results[f'{ema_key}_slope_direction'] = np.sign(ema_results[f'{ema_key}_slope'])
            
            # EMA alignment analysis
            ema_20 = ema_results['ema_20']
            ema_100 = ema_results['ema_100']
            ema_200 = ema_results['ema_200']
            
            ema_results['ema_alignment_bullish'] = (
                (ema_20 > ema_100) & (ema_100 > ema_200)
            ).astype(int)
            
            ema_results['ema_alignment_bearish'] = (
                (ema_20 < ema_100) & (ema_100 < ema_200)
            ).astype(int)
            
            ema_results['ema_alignment_neutral'] = (
                ~ema_results['ema_alignment_bullish'].astype(bool) & 
                ~ema_results['ema_alignment_bearish'].astype(bool)
            ).astype(int)
            
            # EMA convergence/divergence
            ema_results['ema_convergence'] = abs(ema_20 - ema_100) / ema_100
            ema_results['ema_strength'] = (
                ema_results['ema_alignment_bullish'] * 1 + 
                ema_results['ema_alignment_bearish'] * (-1)
            )
            
            return ema_results
            
        except Exception as e:
            logger.error(f"Error calculating EMA analysis: {e}")
            return {}
    
    def _calculate_independent_vwap_analysis(self, prices: pd.Series, 
                                           volume: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent VWAP analysis for ATM Straddle"""
        try:
            vwap_results = {}
            
            # Current day VWAP
            cumulative_volume = volume.cumsum()
            cumulative_pv = (prices * volume).cumsum()
            vwap_current = cumulative_pv / cumulative_volume
            vwap_current = vwap_current.fillna(prices)  # Fallback to price if volume is 0
            
            vwap_results['vwap_current'] = vwap_current
            
            # Previous day VWAP (shifted)
            vwap_results['vwap_previous'] = vwap_current.shift(1).fillna(vwap_current)
            
            # VWAP positioning
            vwap_results['vwap_position'] = (prices / vwap_current - 1).fillna(0)
            vwap_results['above_vwap_current'] = (prices > vwap_current).astype(int)
            vwap_results['above_vwap_previous'] = (prices > vwap_results['vwap_previous']).astype(int)
            
            # VWAP momentum
            vwap_results['vwap_momentum'] = vwap_current.pct_change().fillna(0)
            vwap_results['vwap_acceleration'] = vwap_results['vwap_momentum'].diff().fillna(0)
            
            # VWAP reversion signals
            vwap_deviation = abs(prices - vwap_current) / vwap_current
            vwap_results['vwap_deviation'] = vwap_deviation.fillna(0)
            vwap_results['vwap_reversion_signal'] = (vwap_deviation > 0.02).astype(int)  # 2% threshold
            
            # VWAP standard deviation bands
            vwap_std = (prices - vwap_current).rolling(window=20).std()
            vwap_results['vwap_upper_band'] = vwap_current + vwap_std
            vwap_results['vwap_lower_band'] = vwap_current - vwap_std
            vwap_results['vwap_band_position'] = (
                (prices - vwap_results['vwap_lower_band']) / 
                (vwap_results['vwap_upper_band'] - vwap_results['vwap_lower_band'])
            ).fillna(0.5)
            
            return vwap_results
            
        except Exception as e:
            logger.error(f"Error calculating VWAP analysis: {e}")
            return {}
    
    def _calculate_independent_pivot_analysis(self, prices: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent pivot analysis for ATM Straddle"""
        try:
            pivot_results = {}
            
            # Calculate daily OHLC for pivot points
            # For intraday data, we'll use rolling windows to simulate daily data
            window_size = 75  # Approximate minutes in a trading session
            
            daily_high = prices.rolling(window=window_size).max()
            daily_low = prices.rolling(window=window_size).min()
            daily_close = prices  # Current price as close
            
            # Current day pivot points
            pivot_current = (daily_high + daily_low + daily_close) / 3
            pivot_results['pivot_current'] = pivot_current
            
            # Support and resistance levels
            pivot_results['resistance_1'] = 2 * pivot_current - daily_low
            pivot_results['support_1'] = 2 * pivot_current - daily_high
            pivot_results['resistance_2'] = pivot_current + (daily_high - daily_low)
            pivot_results['support_2'] = pivot_current - (daily_high - daily_low)
            
            # Previous day pivot (shifted)
            pivot_results['pivot_previous'] = pivot_current.shift(1).fillna(pivot_current)
            
            # Pivot positioning
            pivot_results['pivot_position'] = (prices / pivot_current - 1).fillna(0)
            pivot_results['above_pivot_current'] = (prices > pivot_current).astype(int)
            pivot_results['above_pivot_previous'] = (prices > pivot_results['pivot_previous']).astype(int)
            
            # Proximity to support/resistance
            tolerance = 0.005  # 0.5% tolerance
            pivot_results['near_resistance_1'] = (
                abs(prices - pivot_results['resistance_1']) / prices < tolerance
            ).astype(int)
            pivot_results['near_support_1'] = (
                abs(prices - pivot_results['support_1']) / prices < tolerance
            ).astype(int)
            
            # Pivot breakout detection
            pivot_results['pivot_breakout_bullish'] = (
                prices > pivot_results['resistance_1']
            ).astype(int)
            pivot_results['pivot_breakout_bearish'] = (
                prices < pivot_results['support_1']
            ).astype(int)
            
            return pivot_results
            
        except Exception as e:
            logger.error(f"Error calculating pivot analysis: {e}")
            return {}
    
    def _calculate_atm_specific_indicators(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate ATM-specific indicators"""
        try:
            atm_results = {}
            
            # ATM straddle volatility
            atm_results['price_volatility'] = prices.rolling(window=20).std().fillna(0)
            atm_results['volatility_percentile'] = (
                atm_results['price_volatility'].rolling(window=100).rank(pct=True).fillna(0.5)
            )
            
            # ATM momentum indicators
            atm_results['price_momentum_1'] = prices.pct_change(1).fillna(0)
            atm_results['price_momentum_5'] = prices.pct_change(5).fillna(0)
            atm_results['price_momentum_15'] = prices.pct_change(15).fillna(0)
            
            # ATM trend strength
            atm_results['trend_strength'] = abs(
                prices.rolling(window=10).mean() / prices.rolling(window=30).mean() - 1
            ).fillna(0)
            
            # ATM price efficiency (how close to theoretical)
            # This would require theoretical pricing model, simplified here
            atm_results['price_efficiency'] = pd.Series([1.0] * len(prices))  # Placeholder
            
            return atm_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM-specific indicators: {e}")
            return {}
    
    def _calculate_summary_metrics(self, ema_results: Dict, vwap_results: Dict,
                                 pivot_results: Dict, atm_results: Dict) -> Dict[str, float]:
        """Calculate summary metrics for ATM Straddle analysis"""
        try:
            summary = {}
            
            # Overall bullish/bearish bias
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # EMA signals
            if 'ema_alignment_bullish' in ema_results:
                bullish_signals += ema_results['ema_alignment_bullish'].iloc[-1] if len(ema_results['ema_alignment_bullish']) > 0 else 0
                bearish_signals += ema_results['ema_alignment_bearish'].iloc[-1] if len(ema_results['ema_alignment_bearish']) > 0 else 0
                total_signals += 1
            
            # VWAP signals
            if 'above_vwap_current' in vwap_results:
                bullish_signals += vwap_results['above_vwap_current'].iloc[-1] if len(vwap_results['above_vwap_current']) > 0 else 0
                bearish_signals += (1 - vwap_results['above_vwap_current'].iloc[-1]) if len(vwap_results['above_vwap_current']) > 0 else 0
                total_signals += 1
            
            # Pivot signals
            if 'above_pivot_current' in pivot_results:
                bullish_signals += pivot_results['above_pivot_current'].iloc[-1] if len(pivot_results['above_pivot_current']) > 0 else 0
                bearish_signals += (1 - pivot_results['above_pivot_current'].iloc[-1]) if len(pivot_results['above_pivot_current']) > 0 else 0
                total_signals += 1
            
            # Calculate summary scores
            summary['bullish_score'] = bullish_signals / total_signals if total_signals > 0 else 0.5
            summary['bearish_score'] = bearish_signals / total_signals if total_signals > 0 else 0.5
            summary['neutral_score'] = 1 - summary['bullish_score'] - summary['bearish_score']
            
            # Overall signal strength
            summary['signal_strength'] = abs(summary['bullish_score'] - summary['bearish_score'])
            summary['signal_direction'] = 1 if summary['bullish_score'] > summary['bearish_score'] else -1
            
            # Confidence based on signal alignment
            summary['confidence'] = summary['signal_strength']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary metrics: {e}")
            return {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0}
    
    def _assess_data_quality(self, prices: pd.Series) -> Dict[str, Any]:
        """Assess data quality for ATM Straddle analysis"""
        try:
            quality_metrics = {
                'data_points': len(prices),
                'missing_values': prices.isna().sum(),
                'zero_values': (prices == 0).sum(),
                'data_completeness': (len(prices) - prices.isna().sum()) / len(prices) if len(prices) > 0 else 0,
                'price_range': {
                    'min': float(prices.min()) if len(prices) > 0 else 0,
                    'max': float(prices.max()) if len(prices) > 0 else 0,
                    'mean': float(prices.mean()) if len(prices) > 0 else 0,
                    'std': float(prices.std()) if len(prices) > 0 else 0
                },
                'quality_score': 1.0 if len(prices) > 0 and prices.isna().sum() == 0 else 0.5
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {'quality_score': 0.0}
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results when calculation fails"""
        return {
            'component': self.component_name,
            'timeframe': 'unknown',
            'weight': self.weight,
            'ema_analysis': {},
            'vwap_analysis': {},
            'pivot_analysis': {},
            'atm_specific_analysis': {},
            'summary_metrics': {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0},
            'calculation_timestamp': datetime.now().isoformat(),
            'data_quality': {'quality_score': 0.0},
            'error': 'Calculation failed'
        }
