#!/usr/bin/env python3
"""
OTM1 Straddle Engine - Independent Technical Analysis
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides independent technical analysis for OTM1 Straddle component:
- Independent EMA 20/100/200 calculations (NO adjustment factors)
- Independent VWAP current/previous day analysis
- Independent Pivot point calculations
- Multi-timeframe rolling windows (3, 5, 10, 15 minutes)
- OTM1-specific gamma sensitivity and time value analysis

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

class OTM1StraddleEngine:
    """
    OTM1 Straddle Engine with independent technical analysis
    
    Provides complete technical analysis for OTM1 Straddle (OTM1 CE + OTM1 PE) without
    any adjustment factors. All calculations are independent and based on actual
    OTM1 straddle price movements with OTM1-specific characteristics.
    """
    
    def __init__(self, component_config: Dict[str, Any]):
        """Initialize OTM1 Straddle Engine"""
        self.config = component_config
        self.component_name = "OTM1_Straddle"
        self.weight = component_config.get('weight', 0.15)
        self.priority = component_config.get('priority', 'high')
        
        # Technical indicator parameters
        self.ema_periods = [20, 100, 200]
        self.vwap_periods = [1, 5, 15]  # Days
        self.rolling_windows = component_config.get('rolling_windows', [3, 5, 10, 15])
        
        # OTM1-specific parameters
        self.gamma_sensitivity_threshold = 0.15
        self.time_value_weight = 0.8  # OTM1 has higher time value component
        self.volatility_sensitivity = 1.25  # OTM1 more sensitive to volatility
        
        logger.info(f"OTM1 Straddle Engine initialized - Weight: {self.weight}, Priority: {self.priority}")
    
    def calculate_independent_technical_analysis(self, otm1_straddle_prices: pd.Series,
                                               otm1_ce_volume: pd.Series = None,
                                               otm1_pe_volume: pd.Series = None,
                                               spot_price: pd.Series = None,
                                               otm1_strike: float = None,
                                               implied_volatility: pd.Series = None,
                                               timeframe: str = '5min') -> Dict[str, Any]:
        """
        Calculate independent technical analysis for OTM1 Straddle
        
        Args:
            otm1_straddle_prices: OTM1 CE + OTM1 PE prices (NO adjustments)
            otm1_ce_volume: OTM1 CE volume data
            otm1_pe_volume: OTM1 PE volume data
            spot_price: Underlying spot price for gamma calculations
            otm1_strike: OTM1 strike price
            implied_volatility: IV data for vega analysis
            timeframe: Analysis timeframe
            
        Returns:
            Complete technical analysis results
        """
        try:
            logger.debug(f"Calculating OTM1 Straddle technical analysis for {timeframe}")
            
            # Validate input data
            if otm1_straddle_prices.empty:
                logger.warning("Empty OTM1 straddle price data")
                return self._get_default_results()
            
            # Calculate total volume for VWAP
            total_volume = pd.Series([1] * len(otm1_straddle_prices))  # Default volume
            if otm1_ce_volume is not None and otm1_pe_volume is not None:
                total_volume = otm1_ce_volume + otm1_pe_volume
            
            # Independent EMA Analysis (NO adjustment factors)
            ema_results = self._calculate_independent_ema_analysis(otm1_straddle_prices)
            
            # Independent VWAP Analysis
            vwap_results = self._calculate_independent_vwap_analysis(
                otm1_straddle_prices, total_volume, timeframe
            )
            
            # Independent Pivot Analysis
            pivot_results = self._calculate_independent_pivot_analysis(otm1_straddle_prices, timeframe)
            
            # OTM1-specific indicators
            otm1_specific_results = self._calculate_otm1_specific_indicators(
                otm1_straddle_prices, spot_price, otm1_strike, implied_volatility
            )
            
            # Combine all results
            technical_results = {
                'component': self.component_name,
                'timeframe': timeframe,
                'weight': self.weight,
                'ema_analysis': ema_results,
                'vwap_analysis': vwap_results,
                'pivot_analysis': pivot_results,
                'otm1_specific_analysis': otm1_specific_results,
                'summary_metrics': self._calculate_summary_metrics(
                    ema_results, vwap_results, pivot_results, otm1_specific_results
                ),
                'calculation_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(otm1_straddle_prices)
            }
            
            logger.debug(f"OTM1 Straddle technical analysis completed for {timeframe}")
            return technical_results
            
        except Exception as e:
            logger.error(f"Error calculating OTM1 Straddle technical analysis: {e}")
            return self._get_default_results()
    
    def _calculate_independent_ema_analysis(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate independent EMA analysis for OTM1 Straddle (NO adjustments)"""
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
            
            # OTM1-specific EMA characteristics
            ema_results['ema_volatility'] = ema_20.rolling(window=10).std().fillna(0)
            ema_results['ema_momentum'] = (ema_20 / ema_20.shift(5) - 1).fillna(0)
            
            # OTM1 EMA sensitivity (higher volatility sensitivity)
            ema_results['ema_sensitivity'] = abs(ema_results['ema_20_position']) * self.volatility_sensitivity
            
            # OTM1 EMA acceleration (important for time decay analysis)
            ema_results['ema_acceleration'] = ema_results['ema_momentum'].diff().fillna(0)
            
            return ema_results
            
        except Exception as e:
            logger.error(f"Error calculating OTM1 EMA analysis: {e}")
            return {}
    
    def _calculate_independent_vwap_analysis(self, prices: pd.Series, 
                                           volume: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent VWAP analysis for OTM1 Straddle"""
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
            
            # OTM1-specific VWAP analysis
            vwap_results['vwap_momentum'] = vwap_current.pct_change().fillna(0)
            vwap_results['vwap_acceleration'] = vwap_results['vwap_momentum'].diff().fillna(0)
            
            # OTM1 VWAP reversion (OTM options have higher reversion tendency)
            vwap_deviation = abs(prices - vwap_current) / vwap_current
            vwap_results['vwap_deviation'] = vwap_deviation.fillna(0)
            vwap_results['vwap_reversion_signal'] = (vwap_deviation > 0.03).astype(int)  # 3% threshold for OTM1
            
            # OTM1 VWAP volatility impact
            vwap_results['vwap_volatility_impact'] = vwap_deviation * self.volatility_sensitivity
            
            # OTM1 VWAP efficiency (lower efficiency expected due to wider spreads)
            vwap_results['vwap_efficiency'] = 1 / (1 + vwap_deviation * 1.2)  # Adjusted for OTM characteristics
            
            return vwap_results
            
        except Exception as e:
            logger.error(f"Error calculating OTM1 VWAP analysis: {e}")
            return {}
    
    def _calculate_independent_pivot_analysis(self, prices: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent pivot analysis for OTM1 Straddle"""
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
            
            # OTM1-specific pivot analysis
            tolerance = 0.01  # 1% tolerance for OTM1 (higher than ITM1 due to wider spreads)
            pivot_results['near_resistance_1'] = (
                abs(prices - pivot_results['resistance_1']) / prices < tolerance
            ).astype(int)
            pivot_results['near_support_1'] = (
                abs(prices - pivot_results['support_1']) / prices < tolerance
            ).astype(int)
            
            # OTM1 pivot breakout strength (more volatile)
            pivot_results['pivot_breakout_strength'] = np.maximum(
                (prices - pivot_results['resistance_1']) / pivot_results['resistance_1'],
                (pivot_results['support_1'] - prices) / pivot_results['support_1']
            ).fillna(0) * 1.2  # Amplified for OTM1
            
            # OTM1 pivot volatility
            pivot_results['pivot_volatility'] = (
                (daily_high - daily_low) / pivot_current
            ).fillna(0)
            
            return pivot_results
            
        except Exception as e:
            logger.error(f"Error calculating OTM1 pivot analysis: {e}")
            return {}
    
    def _calculate_otm1_specific_indicators(self, prices: pd.Series, 
                                          spot_price: pd.Series = None,
                                          otm1_strike: float = None,
                                          implied_volatility: pd.Series = None) -> Dict[str, Any]:
        """Calculate OTM1-specific indicators"""
        try:
            otm1_results = {}
            
            # OTM1 volatility characteristics (higher than ITM1)
            otm1_results['price_volatility'] = prices.rolling(window=20).std().fillna(0)
            otm1_results['volatility_rank'] = (
                otm1_results['price_volatility'].rolling(window=100).rank(pct=True).fillna(0.5)
            )
            
            # OTM1 momentum (typically higher than ITM1 due to leverage)
            otm1_results['price_momentum_1'] = prices.pct_change(1).fillna(0)
            otm1_results['price_momentum_5'] = prices.pct_change(5).fillna(0)
            otm1_results['price_momentum_15'] = prices.pct_change(15).fillna(0)
            
            # OTM1 gamma sensitivity analysis
            if spot_price is not None and otm1_strike is not None:
                # Approximate gamma for OTM1 options
                moneyness = spot_price / otm1_strike
                otm1_results['moneyness'] = moneyness
                
                # OTM1 gamma is highest near ATM, lower for deep OTM
                distance_from_atm = abs(moneyness - 1)
                otm1_results['gamma_sensitivity'] = np.exp(-distance_from_atm * 2) * 0.4  # Peak gamma at ATM
                
                # Time value component (very high for OTM1)
                otm1_results['time_value_ratio'] = pd.Series([self.time_value_weight] * len(prices))
                otm1_results['intrinsic_value_ratio'] = 1 - otm1_results['time_value_ratio']
                
                # OTM1 efficiency (how close actual price is to time value)
                time_value_estimate = prices.mean() * 0.8  # Simplified time value estimate
                otm1_results['price_efficiency'] = (
                    prices / time_value_estimate
                ).fillna(1.0).clip(0.3, 3.0)  # Wider range for OTM1
            else:
                # Default values when spot/strike not available
                otm1_results['gamma_sensitivity'] = pd.Series([0.3] * len(prices))
                otm1_results['time_value_ratio'] = pd.Series([self.time_value_weight] * len(prices))
                otm1_results['intrinsic_value_ratio'] = pd.Series([1 - self.time_value_weight] * len(prices))
                otm1_results['price_efficiency'] = pd.Series([1.0] * len(prices))
            
            # OTM1 vega impact (high sensitivity to IV changes)
            if implied_volatility is not None:
                otm1_results['vega_impact'] = implied_volatility * otm1_results['gamma_sensitivity'] * 5
                otm1_results['iv_momentum'] = implied_volatility.pct_change().fillna(0)
            else:
                otm1_results['vega_impact'] = pd.Series([0.2] * len(prices))
                otm1_results['iv_momentum'] = pd.Series([0.0] * len(prices))
            
            # OTM1 trend strength (amplified due to leverage)
            otm1_results['trend_strength'] = abs(
                prices.rolling(window=10).mean() / prices.rolling(window=30).mean() - 1
            ).fillna(0) * 1.3  # Amplified for OTM1
            
            # OTM1 time decay impact (theta)
            otm1_results['time_decay_impact'] = (
                otm1_results['time_value_ratio'] * 0.1  # Simplified theta approximation
            )
            
            # OTM1 leverage factor
            otm1_results['leverage_factor'] = (
                otm1_results['gamma_sensitivity'] * otm1_results['time_value_ratio'] * 2
            )
            
            return otm1_results
            
        except Exception as e:
            logger.error(f"Error calculating OTM1-specific indicators: {e}")
            return {}
    
    def _calculate_summary_metrics(self, ema_results: Dict, vwap_results: Dict,
                                 pivot_results: Dict, otm1_results: Dict) -> Dict[str, float]:
        """Calculate summary metrics for OTM1 Straddle analysis"""
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
            
            # OTM1-specific signals
            if 'gamma_sensitivity' in otm1_results:
                gamma_signal = otm1_results['gamma_sensitivity'].iloc[-1] if len(otm1_results['gamma_sensitivity']) > 0 else 0.3
                if gamma_signal > 0.35:
                    bullish_signals += 1  # High gamma suggests potential for movement
                total_signals += 1
            
            # Calculate summary scores
            summary['bullish_score'] = bullish_signals / total_signals if total_signals > 0 else 0.5
            summary['bearish_score'] = bearish_signals / total_signals if total_signals > 0 else 0.5
            summary['neutral_score'] = 1 - summary['bullish_score'] - summary['bearish_score']
            
            # Overall signal strength (amplified for OTM1 characteristics)
            summary['signal_strength'] = abs(summary['bullish_score'] - summary['bearish_score']) * 1.1  # OTM1 amplification
            summary['signal_direction'] = 1 if summary['bullish_score'] > summary['bearish_score'] else -1
            
            # Confidence based on signal alignment and OTM1 characteristics
            time_value_weight = otm1_results.get('time_value_ratio', pd.Series([self.time_value_weight])).iloc[-1] if len(otm1_results.get('time_value_ratio', [])) > 0 else self.time_value_weight
            summary['confidence'] = summary['signal_strength'] * (0.5 + 0.5 * time_value_weight)  # Higher time value = higher volatility = lower confidence
            
            # OTM1-specific metrics
            summary['gamma_sensitivity'] = otm1_results.get('gamma_sensitivity', pd.Series([0.3])).iloc[-1] if len(otm1_results.get('gamma_sensitivity', [])) > 0 else 0.3
            summary['time_value_ratio'] = time_value_weight
            summary['leverage_factor'] = otm1_results.get('leverage_factor', pd.Series([1.0])).iloc[-1] if len(otm1_results.get('leverage_factor', [])) > 0 else 1.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating OTM1 summary metrics: {e}")
            return {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0}
    
    def _assess_data_quality(self, prices: pd.Series) -> Dict[str, Any]:
        """Assess data quality for OTM1 Straddle analysis"""
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
                'otm1_specific_quality': {
                    'price_volatility': (prices.std() / prices.mean()) if len(prices) > 0 and prices.mean() > 0 else 0,
                    'data_consistency': 1.0 if (prices >= 0).all() else 0.5,  # OTM1 should be non-negative
                    'volatility_reasonableness': 1.0 if len(prices) > 0 and (prices.std() / prices.mean()) < 2.0 else 0.5  # Check for reasonable volatility
                }
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing OTM1 data quality: {e}")
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
            'otm1_specific_analysis': {},
            'summary_metrics': {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0},
            'calculation_timestamp': datetime.now().isoformat(),
            'data_quality': {'quality_score': 0.0},
            'error': 'OTM1 calculation failed'
        }
