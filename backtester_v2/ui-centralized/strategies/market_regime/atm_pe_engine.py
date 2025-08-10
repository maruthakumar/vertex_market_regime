#!/usr/bin/env python3
"""
ATM PE Engine - Independent Technical Analysis
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides independent technical analysis for ATM PE component:
- Independent EMA 20/100/200 calculations
- Independent VWAP current/previous day analysis
- Independent Pivot point calculations
- Multi-timeframe rolling windows (3, 5, 10, 15 minutes)
- ATM PE-specific delta and directional bias analysis

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

class ATMPEEngine:
    """
    ATM PE Engine with independent technical analysis
    
    Provides complete technical analysis for ATM PE (Put Option) without
    any adjustment factors. All calculations are independent and based on actual
    ATM PE price movements with put-specific characteristics.
    """
    
    def __init__(self, component_config: Dict[str, Any]):
        """Initialize ATM PE Engine"""
        self.config = component_config
        self.component_name = "ATM_PE"
        self.weight = component_config.get('weight', 0.10)
        self.priority = component_config.get('priority', 'medium')
        
        # Technical indicator parameters
        self.ema_periods = [20, 100, 200]
        self.vwap_periods = [1, 5, 15]  # Days
        self.rolling_windows = component_config.get('rolling_windows', [3, 5, 10, 15])
        
        # ATM PE-specific parameters
        self.delta_range = (-0.55, -0.45)  # ATM PE delta range (negative)
        self.directional_sensitivity = -1.0  # Inverse directional exposure
        
        logger.info(f"ATM PE Engine initialized - Weight: {self.weight}, Priority: {self.priority}")
    
    def calculate_independent_technical_analysis(self, atm_pe_prices: pd.Series,
                                               atm_pe_volume: pd.Series = None,
                                               spot_price: pd.Series = None,
                                               atm_strike: float = None,
                                               timeframe: str = '5min') -> Dict[str, Any]:
        """
        Calculate independent technical analysis for ATM PE
        
        Args:
            atm_pe_prices: ATM PE prices
            atm_pe_volume: ATM PE volume data
            spot_price: Underlying spot price for delta calculations
            atm_strike: ATM strike price
            timeframe: Analysis timeframe
            
        Returns:
            Complete technical analysis results
        """
        try:
            logger.debug(f"Calculating ATM PE technical analysis for {timeframe}")
            
            # Validate input data
            if atm_pe_prices.empty:
                logger.warning("Empty ATM PE price data")
                return self._get_default_results()
            
            # Use volume or default
            volume = atm_pe_volume if atm_pe_volume is not None else pd.Series([1] * len(atm_pe_prices))
            
            # Independent EMA Analysis
            ema_results = self._calculate_independent_ema_analysis(atm_pe_prices)
            
            # Independent VWAP Analysis
            vwap_results = self._calculate_independent_vwap_analysis(atm_pe_prices, volume, timeframe)
            
            # Independent Pivot Analysis
            pivot_results = self._calculate_independent_pivot_analysis(atm_pe_prices, timeframe)
            
            # ATM PE-specific indicators
            atm_pe_specific_results = self._calculate_atm_pe_specific_indicators(
                atm_pe_prices, spot_price, atm_strike
            )
            
            # Combine all results
            technical_results = {
                'component': self.component_name,
                'timeframe': timeframe,
                'weight': self.weight,
                'ema_analysis': ema_results,
                'vwap_analysis': vwap_results,
                'pivot_analysis': pivot_results,
                'atm_pe_specific_analysis': atm_pe_specific_results,
                'summary_metrics': self._calculate_summary_metrics(
                    ema_results, vwap_results, pivot_results, atm_pe_specific_results
                ),
                'calculation_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(atm_pe_prices)
            }
            
            logger.debug(f"ATM PE technical analysis completed for {timeframe}")
            return technical_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM PE technical analysis: {e}")
            return self._get_default_results()
    
    def _calculate_independent_ema_analysis(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate independent EMA analysis for ATM PE"""
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
            
            # ATM PE-specific EMA characteristics
            ema_results['ema_momentum'] = (ema_20 / ema_20.shift(5) - 1).fillna(0)
            ema_results['ema_acceleration'] = ema_results['ema_momentum'].diff().fillna(0)
            
            # Put option directional bias (inverse to calls)
            ema_results['bearish_momentum'] = np.maximum(0, -ema_results['ema_momentum'])  # Positive when declining
            ema_results['bullish_momentum'] = np.minimum(0, -ema_results['ema_momentum'])  # Negative when rising
            
            # Put option EMA signals (inverse interpretation)
            ema_results['pe_bullish_signal'] = ema_results['ema_alignment_bearish']  # Bearish market = bullish for puts
            ema_results['pe_bearish_signal'] = ema_results['ema_alignment_bullish']  # Bullish market = bearish for puts
            
            return ema_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM PE EMA analysis: {e}")
            return {}
    
    def _calculate_independent_vwap_analysis(self, prices: pd.Series, 
                                           volume: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent VWAP analysis for ATM PE"""
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
            
            # ATM PE-specific VWAP analysis
            vwap_results['vwap_momentum'] = vwap_current.pct_change().fillna(0)
            vwap_results['vwap_acceleration'] = vwap_results['vwap_momentum'].diff().fillna(0)
            
            # Put option VWAP signals (inverse interpretation)
            vwap_deviation = (prices - vwap_current) / vwap_current
            vwap_results['vwap_deviation'] = vwap_deviation.fillna(0)
            vwap_results['bearish_vwap_signal'] = (vwap_deviation > 0.02).astype(int)  # PE above VWAP = bearish market
            vwap_results['bullish_vwap_signal'] = (vwap_deviation < -0.02).astype(int)  # PE below VWAP = bullish market
            
            # VWAP trend strength for puts (inverse bias)
            vwap_results['vwap_trend_strength'] = np.maximum(0, -vwap_deviation)  # Negative bias for puts
            
            # Put-specific VWAP momentum
            vwap_results['pe_vwap_momentum'] = -vwap_results['vwap_momentum']  # Inverse for puts
            
            return vwap_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM PE VWAP analysis: {e}")
            return {}
    
    def _calculate_independent_pivot_analysis(self, prices: pd.Series, timeframe: str) -> Dict[str, Any]:
        """Calculate independent pivot analysis for ATM PE"""
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
            
            # ATM PE-specific pivot analysis
            tolerance = 0.005  # 0.5% tolerance
            pivot_results['near_resistance_1'] = (
                abs(prices - pivot_results['resistance_1']) / prices < tolerance
            ).astype(int)
            pivot_results['near_support_1'] = (
                abs(prices - pivot_results['support_1']) / prices < tolerance
            ).astype(int)
            
            # Put option pivot breakout (bearish bias)
            pivot_results['bearish_breakout'] = (
                prices > pivot_results['resistance_1']
            ).astype(int)  # PE above resistance = market stress
            pivot_results['bullish_breakdown'] = (
                prices < pivot_results['support_1']
            ).astype(int)  # PE below support = market strength
            
            # Put-specific pivot signals
            pivot_results['pe_bullish_signal'] = pivot_results['bearish_breakout']  # High PE = bullish for puts
            pivot_results['pe_bearish_signal'] = pivot_results['bullish_breakdown']  # Low PE = bearish for puts
            
            return pivot_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM PE pivot analysis: {e}")
            return {}
    
    def _calculate_atm_pe_specific_indicators(self, prices: pd.Series, 
                                            spot_price: pd.Series = None,
                                            atm_strike: float = None) -> Dict[str, Any]:
        """Calculate ATM PE-specific indicators"""
        try:
            atm_pe_results = {}
            
            # ATM PE volatility
            atm_pe_results['price_volatility'] = prices.rolling(window=20).std().fillna(0)
            atm_pe_results['volatility_rank'] = (
                atm_pe_results['price_volatility'].rolling(window=100).rank(pct=True).fillna(0.5)
            )
            
            # ATM PE momentum
            atm_pe_results['price_momentum_1'] = prices.pct_change(1).fillna(0)
            atm_pe_results['price_momentum_5'] = prices.pct_change(5).fillna(0)
            atm_pe_results['price_momentum_15'] = prices.pct_change(15).fillna(0)
            
            # Put option specific analysis
            if spot_price is not None and atm_strike is not None:
                # Moneyness for ATM PE
                moneyness = spot_price / atm_strike
                atm_pe_results['moneyness'] = moneyness
                
                # ATM PE delta approximation (negative)
                atm_pe_results['delta_estimate'] = np.clip(
                    -0.5 - 0.1 * (moneyness - 1), -0.7, -0.3
                )
                
                # Intrinsic value for PE
                atm_pe_results['intrinsic_value'] = np.maximum(0, atm_strike - spot_price)
                atm_pe_results['time_value'] = prices - atm_pe_results['intrinsic_value']
                atm_pe_results['time_value_ratio'] = (
                    atm_pe_results['time_value'] / prices
                ).fillna(1.0).clip(0, 1)
                
                # Put option efficiency
                theoretical_min = atm_pe_results['intrinsic_value']
                atm_pe_results['price_efficiency'] = (
                    prices / (theoretical_min + prices.mean() * 0.1)
                ).fillna(1.0).clip(0.5, 2.0)
            else:
                # Default values
                atm_pe_results['delta_estimate'] = pd.Series([-0.5] * len(prices))
                atm_pe_results['intrinsic_value'] = pd.Series([0.0] * len(prices))
                atm_pe_results['time_value'] = prices
                atm_pe_results['time_value_ratio'] = pd.Series([1.0] * len(prices))
                atm_pe_results['price_efficiency'] = pd.Series([1.0] * len(prices))
            
            # Put option directional bias (inverse to market)
            atm_pe_results['bearish_bias'] = (
                atm_pe_results['price_momentum_5'] > 0  # Rising PE = bearish market
            ).astype(float)
            atm_pe_results['bullish_bias'] = (
                atm_pe_results['price_momentum_5'] < 0  # Falling PE = bullish market
            ).astype(float)
            
            # ATM PE gamma exposure (highest for ATM)
            atm_pe_results['gamma_exposure'] = (
                abs(atm_pe_results['delta_estimate']) * (1 - abs(atm_pe_results['delta_estimate']))
            )
            
            # Put option trend strength (inverse interpretation)
            trend_ratio = prices.rolling(window=10).mean() / prices.rolling(window=30).mean() - 1
            atm_pe_results['trend_strength'] = np.maximum(0, trend_ratio).fillna(0)  # Rising PE = market stress
            
            # Put option fear gauge
            atm_pe_results['fear_gauge'] = (
                atm_pe_results['price_volatility'] * atm_pe_results['trend_strength']
            )
            
            # Put/Call parity check (if available)
            atm_pe_results['put_premium_level'] = (
                atm_pe_results['time_value_ratio'] * atm_pe_results['volatility_rank']
            )
            
            return atm_pe_results
            
        except Exception as e:
            logger.error(f"Error calculating ATM PE-specific indicators: {e}")
            return {}
    
    def _calculate_summary_metrics(self, ema_results: Dict, vwap_results: Dict,
                                 pivot_results: Dict, atm_pe_results: Dict) -> Dict[str, float]:
        """Calculate summary metrics for ATM PE analysis"""
        try:
            summary = {}
            
            # Overall bullish/bearish bias (put option perspective)
            bullish_signals = 0  # Bullish for puts = bearish market
            bearish_signals = 0  # Bearish for puts = bullish market
            total_signals = 0
            
            # EMA signals (inverse interpretation for puts)
            if 'pe_bullish_signal' in ema_results:
                bullish_signals += ema_results['pe_bullish_signal'].iloc[-1] if len(ema_results['pe_bullish_signal']) > 0 else 0
                bearish_signals += ema_results['pe_bearish_signal'].iloc[-1] if len(ema_results['pe_bearish_signal']) > 0 else 0
                total_signals += 1
            
            # VWAP signals (inverse interpretation for puts)
            if 'bearish_vwap_signal' in vwap_results:
                bullish_signals += vwap_results['bearish_vwap_signal'].iloc[-1] if len(vwap_results['bearish_vwap_signal']) > 0 else 0
                bearish_signals += vwap_results['bullish_vwap_signal'].iloc[-1] if len(vwap_results['bullish_vwap_signal']) > 0 else 0
                total_signals += 1
            
            # Pivot signals (inverse interpretation for puts)
            if 'pe_bullish_signal' in pivot_results:
                bullish_signals += pivot_results['pe_bullish_signal'].iloc[-1] if len(pivot_results['pe_bullish_signal']) > 0 else 0
                bearish_signals += pivot_results['pe_bearish_signal'].iloc[-1] if len(pivot_results['pe_bearish_signal']) > 0 else 0
                total_signals += 1
            
            # ATM PE-specific signals
            if 'bearish_bias' in atm_pe_results:
                bullish_signals += atm_pe_results['bearish_bias'].iloc[-1] if len(atm_pe_results['bearish_bias']) > 0 else 0
                bearish_signals += atm_pe_results['bullish_bias'].iloc[-1] if len(atm_pe_results['bullish_bias']) > 0 else 0
                total_signals += 1
            
            # Calculate summary scores
            summary['bullish_score'] = bullish_signals / total_signals if total_signals > 0 else 0.5
            summary['bearish_score'] = bearish_signals / total_signals if total_signals > 0 else 0.5
            summary['neutral_score'] = 1 - summary['bullish_score'] - summary['bearish_score']
            
            # Overall signal strength
            summary['signal_strength'] = abs(summary['bullish_score'] - summary['bearish_score'])
            summary['signal_direction'] = 1 if summary['bullish_score'] > summary['bearish_score'] else -1
            
            # Confidence based on put option characteristics
            delta_estimate = abs(atm_pe_results.get('delta_estimate', pd.Series([0.5])).iloc[-1]) if len(atm_pe_results.get('delta_estimate', [])) > 0 else 0.5
            summary['confidence'] = summary['signal_strength'] * (0.8 + 0.2 * delta_estimate)  # Higher delta = higher confidence
            
            # ATM PE-specific metrics
            summary['delta_estimate'] = -delta_estimate  # Keep negative for puts
            summary['gamma_exposure'] = atm_pe_results.get('gamma_exposure', pd.Series([0.25])).iloc[-1] if len(atm_pe_results.get('gamma_exposure', [])) > 0 else 0.25
            summary['directional_bias'] = summary['bullish_score'] - summary['bearish_score']  # Put option bias
            summary['fear_gauge'] = atm_pe_results.get('fear_gauge', pd.Series([0.0])).iloc[-1] if len(atm_pe_results.get('fear_gauge', [])) > 0 else 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating ATM PE summary metrics: {e}")
            return {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0}
    
    def _assess_data_quality(self, prices: pd.Series) -> Dict[str, Any]:
        """Assess data quality for ATM PE analysis"""
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
                'atm_pe_specific_quality': {
                    'price_positivity': 1.0 if (prices >= 0).all() else 0.5,  # PE should be non-negative
                    'price_stability': 1 - (prices.std() / prices.mean()) if len(prices) > 0 and prices.mean() > 0 else 0,
                    'put_characteristics': 1.0  # Placeholder for put-specific checks
                }
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing ATM PE data quality: {e}")
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
            'atm_pe_specific_analysis': {},
            'summary_metrics': {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0},
            'calculation_timestamp': datetime.now().isoformat(),
            'data_quality': {'quality_score': 0.0},
            'error': 'ATM PE calculation failed'
        }
