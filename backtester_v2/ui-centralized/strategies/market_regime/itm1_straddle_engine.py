#!/usr/bin/env python3
"""
ITM1 Straddle Engine - Independent Technical Analysis
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides independent technical analysis for ITM1 Straddle component:
- Independent EMA 20/100/200 calculations (NO adjustment factors)
- Independent VWAP current/previous day analysis
- Independent Pivot point calculations
- Multi-timeframe rolling windows (3, 5, 10, 15 minutes)
- ITM1-specific delta sensitivity and intrinsic value analysis

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

class ITM1StraddleEngine:
    """
    ITM1 Straddle Engine with independent technical analysis
    
    Provides complete technical analysis for ITM1 Straddle (ITM1 CE + ITM1 PE) without
    any adjustment factors. All calculations are independent and based on actual
    ITM1 straddle price movements with ITM1-specific characteristics.
    """
    
    def __init__(self, component_config: Dict[str, Any]):
        """Initialize ITM1 Straddle Engine"""
        self.config = component_config
        self.component_name = "ITM1_Straddle"
        self.weight = component_config.get('weight', 0.20)
        self.priority = component_config.get('priority', 'high')
        
        # Technical indicator parameters
        self.ema_periods = [20, 100, 200]
        self.vwap_periods = [1, 5, 15]  # Days
        self.rolling_windows = component_config.get('rolling_windows', [3, 5, 10, 15])
        
        # ITM1-specific parameters
        self.delta_sensitivity_threshold = 0.1
        self.intrinsic_value_weight = 0.6  # ITM1 has higher intrinsic value component
        
        logger.info(f"ITM1 Straddle Engine initialized - Weight: {self.weight}, Priority: {self.priority}")
    
    def calculate_independent_technical_analysis(self, itm1_straddle_prices: pd.Series,
                                               itm1_ce_volume: pd.Series = None,
                                               itm1_pe_volume: pd.Series = None,
                                               spot_price: pd.Series = None,
                                               itm1_strike: float = None,
                                               timeframe: str = '5min') -> Dict[str, Any]:
        """
        Calculate independent technical analysis for ITM1 Straddle
        
        Args:
            itm1_straddle_prices: ITM1 CE + ITM1 PE prices (NO adjustments)
            itm1_ce_volume: ITM1 CE volume data
            itm1_pe_volume: ITM1 PE volume data
            spot_price: Underlying spot price for intrinsic value calculations
            itm1_strike: ITM1 strike price
            timeframe: Analysis timeframe
            
        Returns:
            Complete technical analysis results
        """
        try:
            logger.debug(f"Calculating ITM1 Straddle technical analysis for {timeframe}")
            
            # Validate input data
            if itm1_straddle_prices.empty:
                logger.warning("Empty ITM1 straddle price data")
                return self._get_default_results()
            
            # Calculate total volume for VWAP
            total_volume = pd.Series([1] * len(itm1_straddle_prices))  # Default volume
            if itm1_ce_volume is not None and itm1_pe_volume is not None:
                total_volume = itm1_ce_volume + itm1_pe_volume
            
            # Independent EMA Analysis (NO adjustment factors)
            ema_results = self._calculate_independent_ema_analysis(itm1_straddle_prices)
            
            # Independent VWAP Analysis
            vwap_results = self._calculate_independent_vwap_analysis(
                itm1_straddle_prices, total_volume, timeframe
            )
            
            # Independent Pivot Analysis
            pivot_results = self._calculate_independent_pivot_analysis(itm1_straddle_prices, timeframe)
            
            # ITM1-specific indicators
            itm1_specific_results = self._calculate_itm1_specific_indicators(
                itm1_straddle_prices, spot_price, itm1_strike
            )
            
            # Combine all results
            technical_results = {
                'component': self.component_name,
                'timeframe': timeframe,
                'weight': self.weight,
                'ema_analysis': ema_results,
                'vwap_analysis': vwap_results,
                'pivot_analysis': pivot_results,
                'itm1_specific_analysis': itm1_specific_results,
                'summary_metrics': self._calculate_summary_metrics(
                    ema_results, vwap_results, pivot_results, itm1_specific_results
                ),
                'calculation_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(itm1_straddle_prices)
            }
            
            logger.debug(f"ITM1 Straddle technical analysis completed for {timeframe}")
            return technical_results
            
        except Exception as e:
            logger.error(f"Error calculating ITM1 Straddle technical analysis: {e}")
            return self._get_default_results()
    
    def _calculate_independent_ema_analysis(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate independent EMA analysis for ITM1 Straddle (NO adjustments)"""
        try:
            ema_results = {}
            
            # Calculate EMAs independently
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
            
            # ITM1-specific EMA characteristics
            ema_results['ema_volatility'] = ema_20.rolling(window=10).std().fillna(0)
            ema_results['ema_momentum'] = (ema_20 / ema_20.shift(5) - 1).fillna(0)
            
            # ITM1 EMA sensitivity (higher for ITM options)
            ema_results['ema_sensitivity'] = abs(ema_results['ema_20_position']) * 1.2  # ITM1 sensitivity
            
            return ema_results
            
        except Exception as e:
            logger.error(f"Error calculating ITM1 EMA analysis: {e}")
            return {}
    
    def _calculate_independent_vwap_analysis(self, prices: pd.Series, 
                                           volume: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent VWAP analysis for ITM1 Straddle"""
        try:
            vwap_results = {}
            
            # Current day VWAP
            cumulative_volume = volume.cumsum()
            cumulative_pv = (prices * volume).cumsum()
            vwap_current = cumulative_pv / cumulative_volume
            vwap_current = vwap_current.fillna(prices)  # Fallback to price if volume is 0
            
            vwap_results['vwap_current'] = vwap_current
            vwap_results['vwap_previous'] = vwap_current.shift(1).fillna(vwap_current)
            
            # VWAP positioning
            vwap_results['vwap_position'] = (prices / vwap_current - 1).fillna(0)
            vwap_results['above_vwap_current'] = (prices > vwap_current).astype(int)
            
            # ITM1-specific VWAP analysis
            vwap_results['vwap_momentum'] = vwap_current.pct_change().fillna(0)
            vwap_results['vwap_acceleration'] = vwap_results['vwap_momentum'].diff().fillna(0)
            
            # ITM1 VWAP reversion (ITM options have different reversion characteristics)
            vwap_deviation = abs(prices - vwap_current) / vwap_current
            vwap_results['vwap_deviation'] = vwap_deviation.fillna(0)
            vwap_results['vwap_reversion_signal'] = (vwap_deviation > 0.025).astype(int)  # 2.5% threshold for ITM1
            
            # ITM1 VWAP efficiency
            vwap_results['vwap_efficiency'] = 1 / (1 + vwap_deviation)  # Higher efficiency = lower deviation
            
            return vwap_results
            
        except Exception as e:
            logger.error(f"Error calculating ITM1 VWAP analysis: {e}")
            return {}
    
    def _calculate_independent_pivot_analysis(self, prices: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent pivot analysis for ITM1 Straddle"""
        try:
            pivot_results = {}
            
            # Calculate daily OHLC for pivot points
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
            
            # Previous day pivot
            pivot_results['pivot_previous'] = pivot_current.shift(1).fillna(pivot_current)
            
            # Pivot positioning
            pivot_results['pivot_position'] = (prices / pivot_current - 1).fillna(0)
            pivot_results['above_pivot_current'] = (prices > pivot_current).astype(int)
            
            # ITM1-specific pivot analysis
            tolerance = 0.007  # 0.7% tolerance for ITM1 (slightly higher than ATM)
            pivot_results['near_resistance_1'] = (
                abs(prices - pivot_results['resistance_1']) / prices < tolerance
            ).astype(int)
            pivot_results['near_support_1'] = (
                abs(prices - pivot_results['support_1']) / prices < tolerance
            ).astype(int)
            
            # ITM1 pivot breakout strength
            pivot_results['pivot_breakout_strength'] = np.maximum(
                (prices - pivot_results['resistance_1']) / pivot_results['resistance_1'],
                (pivot_results['support_1'] - prices) / pivot_results['support_1']
            ).fillna(0)
            
            return pivot_results
            
        except Exception as e:
            logger.error(f"Error calculating ITM1 pivot analysis: {e}")
            return {}
    
    def _calculate_itm1_specific_indicators(self, prices: pd.Series, 
                                          spot_price: pd.Series = None,
                                          itm1_strike: float = None) -> Dict[str, Any]:
        """Calculate ITM1-specific indicators"""
        try:
            itm1_results = {}
            
            # ITM1 volatility characteristics
            itm1_results['price_volatility'] = prices.rolling(window=20).std().fillna(0)
            itm1_results['volatility_rank'] = (
                itm1_results['price_volatility'].rolling(window=100).rank(pct=True).fillna(0.5)
            )
            
            # ITM1 momentum (typically lower than ATM due to higher intrinsic value)
            itm1_results['price_momentum_1'] = prices.pct_change(1).fillna(0)
            itm1_results['price_momentum_5'] = prices.pct_change(5).fillna(0)
            itm1_results['price_momentum_15'] = prices.pct_change(15).fillna(0)
            
            # ITM1 delta sensitivity analysis
            if spot_price is not None and itm1_strike is not None:
                # Approximate delta for ITM1 options
                moneyness = spot_price / itm1_strike
                itm1_results['moneyness'] = moneyness
                itm1_results['delta_sensitivity'] = np.where(
                    moneyness > 1, 
                    0.7 + 0.2 * (moneyness - 1),  # ITM CE delta approximation
                    0.3 - 0.2 * (1 - moneyness)   # ITM PE delta approximation
                ).clip(0.1, 0.9)
                
                # Intrinsic value component
                itm1_results['intrinsic_value_ratio'] = np.maximum(
                    0, (spot_price - itm1_strike) / spot_price
                ).fillna(0)
                
                # Time value component
                itm1_results['time_value_ratio'] = 1 - itm1_results['intrinsic_value_ratio']
                
                # ITM1 efficiency (how close actual price is to theoretical)
                theoretical_min = np.maximum(0, spot_price - itm1_strike)
                itm1_results['price_efficiency'] = (
                    prices / (theoretical_min + prices.mean() * 0.1)
                ).fillna(1.0).clip(0.5, 2.0)
            else:
                # Default values when spot/strike not available
                itm1_results['delta_sensitivity'] = pd.Series([0.6] * len(prices))
                itm1_results['intrinsic_value_ratio'] = pd.Series([0.6] * len(prices))
                itm1_results['time_value_ratio'] = pd.Series([0.4] * len(prices))
                itm1_results['price_efficiency'] = pd.Series([1.0] * len(prices))
            
            # ITM1 trend strength (adjusted for intrinsic value component)
            itm1_results['trend_strength'] = abs(
                prices.rolling(window=10).mean() / prices.rolling(window=30).mean() - 1
            ).fillna(0) * 0.8  # Reduced impact due to intrinsic value
            
            # ITM1 gamma exposure (lower than ATM)
            itm1_results['gamma_exposure'] = (
                itm1_results['delta_sensitivity'] * (1 - itm1_results['delta_sensitivity']) * 0.8
            )
            
            return itm1_results
            
        except Exception as e:
            logger.error(f"Error calculating ITM1-specific indicators: {e}")
            return {}
    
    def _calculate_summary_metrics(self, ema_results: Dict, vwap_results: Dict,
                                 pivot_results: Dict, itm1_results: Dict) -> Dict[str, float]:
        """Calculate summary metrics for ITM1 Straddle analysis"""
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
            
            # ITM1-specific signals
            if 'delta_sensitivity' in itm1_results:
                delta_signal = itm1_results['delta_sensitivity'].iloc[-1] if len(itm1_results['delta_sensitivity']) > 0 else 0.5
                if delta_signal > 0.6:
                    bullish_signals += 1
                elif delta_signal < 0.4:
                    bearish_signals += 1
                total_signals += 1
            
            # Calculate summary scores
            summary['bullish_score'] = bullish_signals / total_signals if total_signals > 0 else 0.5
            summary['bearish_score'] = bearish_signals / total_signals if total_signals > 0 else 0.5
            summary['neutral_score'] = 1 - summary['bullish_score'] - summary['bearish_score']
            
            # Overall signal strength (adjusted for ITM1 characteristics)
            summary['signal_strength'] = abs(summary['bullish_score'] - summary['bearish_score']) * 0.9  # ITM1 adjustment
            summary['signal_direction'] = 1 if summary['bullish_score'] > summary['bearish_score'] else -1
            
            # Confidence based on signal alignment and ITM1 characteristics
            intrinsic_weight = itm1_results.get('intrinsic_value_ratio', pd.Series([0.6])).iloc[-1] if len(itm1_results.get('intrinsic_value_ratio', [])) > 0 else 0.6
            summary['confidence'] = summary['signal_strength'] * (0.7 + 0.3 * intrinsic_weight)  # Higher intrinsic = higher confidence
            
            # ITM1-specific metrics
            summary['delta_sensitivity'] = itm1_results.get('delta_sensitivity', pd.Series([0.6])).iloc[-1] if len(itm1_results.get('delta_sensitivity', [])) > 0 else 0.6
            summary['intrinsic_value_ratio'] = intrinsic_weight
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating ITM1 summary metrics: {e}")
            return {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0}
    
    def _assess_data_quality(self, prices: pd.Series) -> Dict[str, Any]:
        """Assess data quality for ITM1 Straddle analysis"""
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
                'quality_score': 1.0 if len(prices) > 0 and prices.isna().sum() == 0 else 0.5,
                'itm1_specific_quality': {
                    'price_stability': 1 - (prices.std() / prices.mean()) if len(prices) > 0 and prices.mean() > 0 else 0,
                    'data_consistency': 1.0 if (prices > 0).all() else 0.5  # ITM1 should always be positive
                }
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing ITM1 data quality: {e}")
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
            'itm1_specific_analysis': {},
            'summary_metrics': {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0},
            'calculation_timestamp': datetime.now().isoformat(),
            'data_quality': {'quality_score': 0.0},
            'error': 'ITM1 calculation failed'
        }
