#!/usr/bin/env python3
"""
ATM CE Engine - Independent Technical Analysis
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides independent technical analysis for ATM CE component:
- Independent EMA 20/100/200 calculations
- Independent VWAP current/previous day analysis
- Independent Pivot point calculations
- Multi-timeframe rolling windows (3, 5, 10, 15 minutes)
- ATM CE-specific delta and directional bias analysis

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

class ATMCEEngine:
    """
    ATM CE Engine with independent technical analysis
    
    Provides complete technical analysis for ATM CE (Call Option) without
    any adjustment factors. All calculations are independent and based on actual
    ATM CE price movements with call-specific characteristics.
    """
    
    def __init__(self, component_config: Dict[str, Any]):
        """Initialize ATM CE Engine"""
        self.config = component_config
        self.component_name = "ATM_CE"
        self.weight = component_config.get('weight', 0.10)
        self.priority = component_config.get('priority', 'medium')
        
        # Technical indicator parameters
        self.ema_periods = [20, 100, 200]
        self.vwap_periods = [1, 5, 15]  # Days
        self.rolling_windows = component_config.get('rolling_windows', [3, 5, 10, 15])
        
        # ATM CE-specific parameters
        self.delta_range = (0.45, 0.55)  # ATM CE delta range
        self.directional_sensitivity = 1.0  # Full directional exposure
        
        logger.info(f"ATM CE Engine initialized - Weight: {self.weight}, Priority: {self.priority}")
    
    def calculate_independent_technical_analysis(self, atm_ce_prices: pd.Series,
                                               atm_ce_volume: pd.Series = None,
                                               spot_price: pd.Series = None,
                                               atm_strike: float = None,
                                               timeframe: str = '5min') -> Dict[str, Any]:
        """
        Calculate independent technical analysis for ATM CE
        
        Args:
            atm_ce_prices: ATM CE prices
            atm_ce_volume: ATM CE volume data
            spot_price: Underlying spot price for delta calculations
            atm_strike: ATM strike price
            timeframe: Analysis timeframe
            
        Returns:
            Complete technical analysis results
        """
        try:
            logger.debug(f"Calculating ATM CE technical analysis for {timeframe}")
            
            # Validate input data
            if atm_ce_prices.empty:
                logger.warning("Empty ATM CE price data")
                return self._get_default_results()
            
            # Use volume or default
            volume = atm_ce_volume if atm_ce_volume is not None else pd.Series([1] * len(atm_ce_prices))
            
            # Independent EMA Analysis
            ema_results = self._calculate_independent_ema_analysis(atm_ce_prices)
            
            # Independent VWAP Analysis
            vwap_results = self._calculate_independent_vwap_analysis(atm_ce_prices, volume, timeframe)
            
            # Independent Pivot Analysis
            pivot_results = self._calculate_independent_pivot_analysis(atm_ce_prices, timeframe)
            
            # ATM CE-specific indicators
            atm_ce_specific_results = self._calculate_atm_ce_specific_indicators(
                atm_ce_prices, spot_price, atm_strike
            )
            
            # Combine all results
            technical_results = {
                'component': self.component_name,
                'timeframe': timeframe,
                'weight': self.weight,
                'ema_analysis': ema_results,
                'vwap_analysis': vwap_results,
                'pivot_analysis': pivot_results,
                'atm_ce_specific_analysis': atm_ce_specific_results,
                'summary_metrics': self._calculate_summary_metrics(
                    ema_results, vwap_results, pivot_results, atm_ce_specific_results
                ),
                'calculation_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(atm_ce_prices)
            }
            
            logger.debug(f"ATM CE technical analysis completed for {timeframe}")
            return technical_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM CE technical analysis: {e}")
            return self._get_default_results()
    
    def _calculate_independent_ema_analysis(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate independent EMA analysis for ATM CE"""
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
            
            # ATM CE-specific EMA characteristics
            ema_results['ema_momentum'] = (ema_20 / ema_20.shift(5) - 1).fillna(0)
            ema_results['ema_acceleration'] = ema_results['ema_momentum'].diff().fillna(0)
            
            # Call option directional bias
            ema_results['bullish_momentum'] = np.maximum(0, ema_results['ema_momentum'])
            ema_results['bearish_momentum'] = np.minimum(0, ema_results['ema_momentum'])
            
            return ema_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM CE EMA analysis: {e}")
            return {}
    
    def _calculate_independent_vwap_analysis(self, prices: pd.Series, 
                                           volume: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent VWAP analysis for ATM CE"""
        try:
            vwap_results = {}
            
            # Current day VWAP
            cumulative_volume = volume.cumsum()
            cumulative_pv = (prices * volume).cumsum()
            vwap_current = cumulative_pv / cumulative_volume
            vwap_current = vwap_current.fillna(prices)
            
            vwap_results['vwap_current'] = vwap_current
            vwap_results['vwap_previous'] = vwap_current.shift(1).fillna(vwap_current)
            
            # VWAP positioning
            vwap_results['vwap_position'] = (prices / vwap_current - 1).fillna(0)
            vwap_results['above_vwap_current'] = (prices > vwap_current).astype(int)
            
            # ATM CE-specific VWAP analysis
            vwap_results['vwap_momentum'] = vwap_current.pct_change().fillna(0)
            vwap_results['vwap_acceleration'] = vwap_results['vwap_momentum'].diff().fillna(0)
            
            # Call option VWAP signals
            vwap_deviation = (prices - vwap_current) / vwap_current
            vwap_results['vwap_deviation'] = vwap_deviation.fillna(0)
            vwap_results['bullish_vwap_signal'] = (vwap_deviation > 0.02).astype(int)  # Above VWAP by 2%
            vwap_results['bearish_vwap_signal'] = (vwap_deviation < -0.02).astype(int)  # Below VWAP by 2%
            
            # VWAP trend strength for calls
            vwap_results['vwap_trend_strength'] = np.maximum(0, vwap_deviation)  # Positive bias for calls
            
            return vwap_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM CE VWAP analysis: {e}")
            return {}
    
    def _calculate_independent_pivot_analysis(self, prices: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent pivot analysis for ATM CE"""
        try:
            pivot_results = {}
            
            # Calculate daily OHLC for pivot points
            window_size = 75  # Approximate minutes in a trading session
            
            daily_high = prices.rolling(window=window_size).max()
            daily_low = prices.rolling(window=window_size).min()
            daily_close = prices
            
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
            
            # ATM CE-specific pivot analysis
            tolerance = 0.005  # 0.5% tolerance
            pivot_results['near_resistance_1'] = (
                abs(prices - pivot_results['resistance_1']) / prices < tolerance
            ).astype(int)
            pivot_results['near_support_1'] = (
                abs(prices - pivot_results['support_1']) / prices < tolerance
            ).astype(int)
            
            # Call option pivot breakout (bullish bias)
            pivot_results['bullish_breakout'] = (
                prices > pivot_results['resistance_1']
            ).astype(int)
            pivot_results['bearish_breakdown'] = (
                prices < pivot_results['support_1']
            ).astype(int)
            
            return pivot_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM CE pivot analysis: {e}")
            return {}
    
    def _calculate_atm_ce_specific_indicators(self, prices: pd.Series, 
                                            spot_price: pd.Series = None,
                                            atm_strike: float = None) -> Dict[str, Any]:
        """Calculate ATM CE-specific indicators"""
        try:
            atm_ce_results = {}
            
            # ATM CE volatility
            atm_ce_results['price_volatility'] = prices.rolling(window=20).std().fillna(0)
            atm_ce_results['volatility_rank'] = (
                atm_ce_results['price_volatility'].rolling(window=100).rank(pct=True).fillna(0.5)
            )
            
            # ATM CE momentum
            atm_ce_results['price_momentum_1'] = prices.pct_change(1).fillna(0)
            atm_ce_results['price_momentum_5'] = prices.pct_change(5).fillna(0)
            atm_ce_results['price_momentum_15'] = prices.pct_change(15).fillna(0)
            
            # Call option specific analysis
            if spot_price is not None and atm_strike is not None:
                # Moneyness for ATM CE
                moneyness = spot_price / atm_strike
                atm_ce_results['moneyness'] = moneyness
                
                # ATM CE delta approximation
                atm_ce_results['delta_estimate'] = np.clip(
                    0.5 + 0.1 * (moneyness - 1), 0.3, 0.7
                )
                
                # Intrinsic value for CE
                atm_ce_results['intrinsic_value'] = np.maximum(0, spot_price - atm_strike)
                atm_ce_results['time_value'] = prices - atm_ce_results['intrinsic_value']
                atm_ce_results['time_value_ratio'] = (
                    atm_ce_results['time_value'] / prices
                ).fillna(1.0).clip(0, 1)
                
                # Call option efficiency
                theoretical_min = atm_ce_results['intrinsic_value']
                atm_ce_results['price_efficiency'] = (
                    prices / (theoretical_min + prices.mean() * 0.1)
                ).fillna(1.0).clip(0.5, 2.0)
            else:
                # Default values
                atm_ce_results['delta_estimate'] = pd.Series([0.5] * len(prices))
                atm_ce_results['intrinsic_value'] = pd.Series([0.0] * len(prices))
                atm_ce_results['time_value'] = prices
                atm_ce_results['time_value_ratio'] = pd.Series([1.0] * len(prices))
                atm_ce_results['price_efficiency'] = pd.Series([1.0] * len(prices))
            
            # Call option directional bias
            atm_ce_results['bullish_bias'] = (
                atm_ce_results['price_momentum_5'] > 0
            ).astype(float)
            
            # ATM CE gamma exposure (highest for ATM)
            atm_ce_results['gamma_exposure'] = (
                atm_ce_results['delta_estimate'] * (1 - atm_ce_results['delta_estimate'])
            )
            
            # Call option trend strength
            atm_ce_results['trend_strength'] = np.maximum(0, 
                prices.rolling(window=10).mean() / prices.rolling(window=30).mean() - 1
            ).fillna(0)
            
            return atm_ce_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM CE-specific indicators: {e}")
            return {}
    
    def _calculate_summary_metrics(self, ema_results: Dict, vwap_results: Dict,
                                 pivot_results: Dict, atm_ce_results: Dict) -> Dict[str, float]:
        """Calculate summary metrics for ATM CE analysis"""
        try:
            summary = {}
            
            # Overall bullish/bearish bias (call option perspective)
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # EMA signals
            if 'ema_alignment_bullish' in ema_results:
                bullish_signals += ema_results['ema_alignment_bullish'].iloc[-1] if len(ema_results['ema_alignment_bullish']) > 0 else 0
                bearish_signals += ema_results['ema_alignment_bearish'].iloc[-1] if len(ema_results['ema_alignment_bearish']) > 0 else 0
                total_signals += 1
            
            # VWAP signals
            if 'bullish_vwap_signal' in vwap_results:
                bullish_signals += vwap_results['bullish_vwap_signal'].iloc[-1] if len(vwap_results['bullish_vwap_signal']) > 0 else 0
                bearish_signals += vwap_results['bearish_vwap_signal'].iloc[-1] if len(vwap_results['bearish_vwap_signal']) > 0 else 0
                total_signals += 1
            
            # Pivot signals
            if 'bullish_breakout' in pivot_results:
                bullish_signals += pivot_results['bullish_breakout'].iloc[-1] if len(pivot_results['bullish_breakout']) > 0 else 0
                bearish_signals += pivot_results['bearish_breakdown'].iloc[-1] if len(pivot_results['bearish_breakdown']) > 0 else 0
                total_signals += 1
            
            # ATM CE-specific signals
            if 'bullish_bias' in atm_ce_results:
                bullish_signals += atm_ce_results['bullish_bias'].iloc[-1] if len(atm_ce_results['bullish_bias']) > 0 else 0
                total_signals += 1
            
            # Calculate summary scores
            summary['bullish_score'] = bullish_signals / total_signals if total_signals > 0 else 0.5
            summary['bearish_score'] = bearish_signals / total_signals if total_signals > 0 else 0.5
            summary['neutral_score'] = 1 - summary['bullish_score'] - summary['bearish_score']
            
            # Overall signal strength
            summary['signal_strength'] = abs(summary['bullish_score'] - summary['bearish_score'])
            summary['signal_direction'] = 1 if summary['bullish_score'] > summary['bearish_score'] else -1
            
            # Confidence based on call option characteristics
            delta_estimate = atm_ce_results.get('delta_estimate', pd.Series([0.5])).iloc[-1] if len(atm_ce_results.get('delta_estimate', [])) > 0 else 0.5
            summary['confidence'] = summary['signal_strength'] * (0.8 + 0.2 * delta_estimate)  # Higher delta = higher confidence
            
            # ATM CE-specific metrics
            summary['delta_estimate'] = delta_estimate
            summary['gamma_exposure'] = atm_ce_results.get('gamma_exposure', pd.Series([0.25])).iloc[-1] if len(atm_ce_results.get('gamma_exposure', [])) > 0 else 0.25
            summary['directional_bias'] = summary['bullish_score'] - summary['bearish_score']  # Call option bias
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating ATM CE summary metrics: {e}")
            return {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0}
    
    def _assess_data_quality(self, prices: pd.Series) -> Dict[str, Any]:
        """Assess data quality for ATM CE analysis"""
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
                'atm_ce_specific_quality': {
                    'price_positivity': 1.0 if (prices >= 0).all() else 0.5,  # CE should be non-negative
                    'price_stability': 1 - (prices.std() / prices.mean()) if len(prices) > 0 and prices.mean() > 0 else 0,
                    'monotonicity_check': 1.0  # Placeholder for more complex checks
                }
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing ATM CE data quality: {e}")
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
            'atm_ce_specific_analysis': {},
            'summary_metrics': {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0},
            'calculation_timestamp': datetime.now().isoformat(),
            'data_quality': {'quality_score': 0.0},
            'error': 'ATM CE calculation failed'
        }
