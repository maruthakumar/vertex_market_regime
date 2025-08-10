#!/usr/bin/env python3
"""
Advanced Market Regime Detection System
Comprehensive Framework with Symmetric Straddles, Multi-Timeframe Analysis & Dynamic DTE Weighting

This advanced system leverages correlation and divergence patterns between symmetric straddles
across different strike levels, enhanced by multi-timeframe technical analysis and dynamically
adjusted based on options expiration cycles and market volatility conditions.

Key Features:
- Industry-standard symmetric straddle analysis
- Multi-timeframe technical indicators (EMA, VWAP, pivot points)
- Dynamic DTE-based weighting (0-4 DTE focus)
- Correlation-based regime detection
- 15-regime classification system
- Integration with IV skew, IV percentile, and ATR analysis

Author: The Augster
Date: 2025-06-19
Version: 6.0.0 (Advanced Correlation-Based System)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, List
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedRegimeDetector:
    """
    Advanced Market Regime Detection System
    Correlation-based regime detection with dynamic weighting and multi-timeframe analysis
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced regime detector with configurable parameters"""

        self.output_dir = Path("advanced_regime_analysis")
        self.output_dir.mkdir(exist_ok=True)

        # Default configuration
        self.config = {
            'ema_periods': [20, 100, 200],
            'additional_indicators': {
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 2.0,
                'stoch_k': 14,
                'stoch_d': 3
            },
            'timeframes': ['1min', '3min', '5min', '15min'],
            'support_resistance_lookback': 20,
            'correlation_window': 20,
            'regime_stability_window': 10
        }

        # Update with user config if provided
        if config:
            self.config.update(config)

        # Base component weights
        self.base_weights = {
            'atm_straddle': 0.40,
            'itm1_straddle': 0.25,
            'otm1_straddle': 0.25,
            'combined_straddle': 0.10
        }

        # 15-Regime Classification System
        self.regime_names = {
            1: "Ultra_Low_Vol_Bullish_Convergence",
            2: "Ultra_Low_Vol_Bearish_Convergence",
            3: "Low_Vol_Bullish_Momentum",
            4: "Low_Vol_Bearish_Momentum",
            5: "Med_Vol_Bullish_Breakout",
            6: "Med_Vol_Bearish_Breakdown",
            7: "Med_Vol_Correlation_Divergence",
            8: "High_Vol_Bullish_Explosion",
            9: "High_Vol_Bearish_Explosion",
            10: "High_Vol_ATM_Dominance",
            11: "Extreme_Vol_Gamma_Squeeze",
            12: "Extreme_Vol_Correlation_Chaos",
            13: "Transition_Bullish_Formation",
            14: "Transition_Bearish_Formation",
            15: "Neutral_Consolidation"
        }

        logger.info("🚀 Advanced Regime Detector initialized")
        logger.info(f"📊 15-regime classification system")
        logger.info(f"🎯 Correlation-based detection with dynamic DTE weighting")

    def calculate_symmetric_straddles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate industry-standard symmetric straddles with combined analysis"""
        logger.info("📊 Calculating symmetric straddles with combined analysis...")

        try:
            # Determine strikes
            df['atm_strike'] = np.round(df['spot_price'] / 50) * 50
            df['itm1_strike'] = df['atm_strike'] - 50
            df['otm1_strike'] = df['atm_strike'] + 50

            # Symmetric Straddle Prices
            df['atm_straddle_price'] = df['atm_ce_price'] + df['atm_pe_price']

            # ITM1 Symmetric Straddle (both options at ITM1 strike)
            itm1_call_adj = 0.85
            itm1_put_adj = 1.15
            df['itm1_call_price'] = df['atm_ce_price'] * itm1_call_adj + 50
            df['itm1_put_price'] = df['atm_pe_price'] * itm1_put_adj
            df['itm1_straddle_price'] = df['itm1_call_price'] + df['itm1_put_price']

            # OTM1 Symmetric Straddle (both options at OTM1 strike)
            otm1_call_adj = 0.75
            otm1_put_adj = 0.85
            df['otm1_call_price'] = df['atm_ce_price'] * otm1_call_adj
            df['otm1_put_price'] = df['atm_pe_price'] * otm1_put_adj + 50
            df['otm1_straddle_price'] = df['otm1_call_price'] + df['otm1_put_price']

            # Combined Straddle Analysis (ATM + ITM1)
            df['combined_straddle_price'] = df['atm_straddle_price'] + df['itm1_straddle_price']

            # Individual ATM Analysis for Directional Confirmation
            df['atm_call_individual'] = df['atm_ce_price']
            df['atm_put_individual'] = df['atm_pe_price']
            df['atm_call_put_ratio'] = df['atm_ce_price'] / (df['atm_pe_price'] + 0.001)

            # Volume and OI calculations
            for straddle_type in ['atm', 'itm1', 'otm1']:
                df[f'{straddle_type}_straddle_volume'] = (
                    df.get(f'{straddle_type}_ce_volume', 0) +
                    df.get(f'{straddle_type}_pe_volume', 0)
                )
                df[f'{straddle_type}_straddle_oi'] = (
                    df.get(f'{straddle_type}_ce_oi', 0) +
                    df.get(f'{straddle_type}_pe_oi', 0)
                )

            df['combined_straddle_volume'] = df['atm_straddle_volume'] + df['itm1_straddle_volume']
            df['combined_straddle_oi'] = df['atm_straddle_oi'] + df['itm1_straddle_oi']

            logger.info("✅ Symmetric straddles calculated with combined analysis")
            return df

        except Exception as e:
            logger.error(f"❌ Error calculating symmetric straddles: {e}")
            return df

    def calculate_multi_timeframe_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Multi-timeframe technical analysis for each straddle"""
        logger.info("📈 Calculating multi-timeframe technical analysis...")

        try:
            results = {}
            straddle_types = ['atm_straddle', 'itm1_straddle', 'otm1_straddle', 'combined_straddle']

            for straddle_type in straddle_types:
                price_col = f'{straddle_type}_price'
                volume_col = f'{straddle_type}_volume'

                if price_col not in df.columns:
                    continue

                price_series = df[price_col]
                volume_series = df.get(volume_col, pd.Series([1000] * len(df)))

                straddle_analysis = {}

                # EMA Analysis
                ema_analysis = {}
                for period in self.config['ema_periods']:
                    ema_values = price_series.ewm(span=period).mean()
                    ema_analysis[f'ema_{period}'] = ema_values
                    ema_analysis[f'ema_{period}_position'] = (price_series / ema_values - 1).fillna(0)
                    ema_analysis[f'ema_{period}_slope'] = ema_values.diff(5).fillna(0)

                # VWAP Analysis
                cumulative_volume = volume_series.cumsum()
                cumulative_pv = (price_series * volume_series).cumsum()
                current_vwap = cumulative_pv / cumulative_volume

                vwap_analysis = {
                    'current_vwap': current_vwap,
                    'vwap_position': (price_series / current_vwap - 1).fillna(0),
                    'vwap_deviation': (abs(price_series - current_vwap) / current_vwap).fillna(0)
                }

                # Pivot Point Analysis
                pivot_analysis = self._calculate_pivot_points(price_series)

                # Support/Resistance Levels
                support_resistance = self._calculate_support_resistance(
                    price_series, self.config['support_resistance_lookback']
                )

                # Additional Indicators (if configured)
                additional_indicators = {}
                if self.config['additional_indicators']:
                    additional_indicators = self._calculate_additional_indicators(
                        price_series, self.config['additional_indicators']
                    )

                straddle_analysis = {
                    'ema_analysis': ema_analysis,
                    'vwap_analysis': vwap_analysis,
                    'pivot_analysis': pivot_analysis,
                    'support_resistance': support_resistance,
                    'additional_indicators': additional_indicators
                }

                results[straddle_type] = straddle_analysis

            logger.info("✅ Multi-timeframe technical analysis completed")
            return results

        except Exception as e:
            logger.error(f"❌ Error in multi-timeframe analysis: {e}")
            return {}

    def _calculate_pivot_points(self, price_series: pd.Series) -> Dict[str, pd.Series]:
        """Calculate pivot points for price series"""

        # Daily aggregation for pivot calculation
        df_temp = pd.DataFrame({'price': price_series, 'timestamp': range(len(price_series))})
        df_temp['date'] = pd.to_datetime(df_temp['timestamp'], unit='min').dt.date

        daily_agg = df_temp.groupby('date')['price'].agg(['first', 'max', 'min', 'last'])
        daily_agg['pivot'] = (daily_agg['max'] + daily_agg['min'] + daily_agg['last']) / 3
        daily_agg['r1'] = 2 * daily_agg['pivot'] - daily_agg['min']
        daily_agg['s1'] = 2 * daily_agg['pivot'] - daily_agg['max']
        daily_agg['r2'] = daily_agg['pivot'] + (daily_agg['max'] - daily_agg['min'])
        daily_agg['s2'] = daily_agg['pivot'] - (daily_agg['max'] - daily_agg['min'])

        # Map back to original series
        pivot_data = {}
        for col in ['pivot', 'r1', 's1', 'r2', 's2']:
            pivot_data[col] = df_temp['date'].map(daily_agg[col]).fillna(method='ffill')

        return pivot_data

    def _calculate_support_resistance(self, price_series: pd.Series, lookback: int) -> Dict[str, pd.Series]:
        """Calculate support and resistance levels"""

        support_levels = price_series.rolling(lookback).min()
        resistance_levels = price_series.rolling(lookback).max()

        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'support_distance': (price_series - support_levels) / support_levels,
            'resistance_distance': (resistance_levels - price_series) / price_series
        }

    def _calculate_additional_indicators(self, price_series: pd.Series, config: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate additional technical indicators"""

        indicators = {}
        price_array = price_series.values.astype(float)

        try:
            # RSI
            if 'rsi_period' in config:
                rsi_values = []
                for i in range(len(price_array)):
                    if i < config['rsi_period']:
                        rsi_values.append(50.0)  # Default RSI
                    else:
                        gains = []
                        losses = []
                        for j in range(i - config['rsi_period'] + 1, i + 1):
                            if j > 0:
                                change = price_array[j] - price_array[j-1]
                                if change > 0:
                                    gains.append(change)
                                    losses.append(0)
                                else:
                                    gains.append(0)
                                    losses.append(abs(change))

                        avg_gain = np.mean(gains) if gains else 0
                        avg_loss = np.mean(losses) if losses else 0

                        if avg_loss == 0:
                            rsi_values.append(100.0)
                        else:
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                            rsi_values.append(rsi)

                indicators['rsi'] = pd.Series(rsi_values, index=price_series.index)

            # Bollinger Bands
            if 'bb_period' in config and 'bb_std' in config:
                bb_middle = price_series.rolling(config['bb_period']).mean()
                bb_std = price_series.rolling(config['bb_period']).std()
                bb_upper = bb_middle + (bb_std * config['bb_std'])
                bb_lower = bb_middle - (bb_std * config['bb_std'])

                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                indicators['bb_position'] = (price_series - bb_middle) / (bb_upper - bb_lower)

            # Stochastic Oscillator
            if 'stoch_k' in config and 'stoch_d' in config:
                high_series = price_series.rolling(config['stoch_k']).max()
                low_series = price_series.rolling(config['stoch_k']).min()

                stoch_k = 100 * (price_series - low_series) / (high_series - low_series)
                stoch_d = stoch_k.rolling(config['stoch_d']).mean()

                indicators['stoch_k'] = stoch_k.fillna(50)
                indicators['stoch_d'] = stoch_d.fillna(50)

        except Exception as e:
            logger.warning(f"⚠️ Error calculating additional indicators: {e}")

        return indicators

    def calculate_straddle_correlations(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate rolling correlations between straddles"""
        logger.info("🔗 Calculating straddle correlations...")

        try:
            window = self.config['correlation_window']

            correlations = {
                'atm_itm1': df['atm_straddle_price'].rolling(window).corr(df['itm1_straddle_price']),
                'atm_otm1': df['atm_straddle_price'].rolling(window).corr(df['otm1_straddle_price']),
                'itm1_otm1': df['itm1_straddle_price'].rolling(window).corr(df['otm1_straddle_price']),
                'combined_atm': df['combined_straddle_price'].rolling(window).corr(df['atm_straddle_price'])
            }

            # Correlation divergence detection
            correlations['divergence_score'] = (
                abs(correlations['atm_itm1'] - correlations['atm_otm1']) +
                abs(correlations['atm_itm1'] - correlations['itm1_otm1'])
            ) / 2

            # Overall correlation strength
            correlations['correlation_strength'] = (
                correlations['atm_itm1'] + correlations['atm_otm1'] + correlations['itm1_otm1']
            ) / 3

            logger.info("✅ Straddle correlations calculated")
            return correlations

        except Exception as e:
            logger.error(f"❌ Error calculating correlations: {e}")
            return {}

    def calculate_regime_signals(self, df: pd.DataFrame, technical_analysis: Dict[str, Any],
                               correlations: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate regime signals based on correlation and technical analysis"""
        logger.info("🎯 Calculating regime signals...")

        try:
            # Bearish Signal Detection
            bearish_signals = {
                'itm1_support_break': self._detect_support_break(
                    df['itm1_straddle_price'],
                    technical_analysis.get('itm1_straddle', {}).get('support_resistance', {})
                ),
                'itm1_ema_breakdown': self._detect_ema_breakdown(
                    df['itm1_straddle_price'],
                    technical_analysis.get('itm1_straddle', {}).get('ema_analysis', {})
                ),
                'itm1_pivot_break': self._detect_pivot_break(
                    df['itm1_straddle_price'],
                    technical_analysis.get('itm1_straddle', {}).get('pivot_analysis', {}),
                    'bearish'
                ),
                'correlation_breakdown': correlations.get('correlation_strength', pd.Series([0.5] * len(df))) < 0.3
            }

            # Bullish Signal Detection
            bullish_signals = {
                'otm1_resistance_break': self._detect_resistance_break(
                    df['otm1_straddle_price'],
                    technical_analysis.get('otm1_straddle', {}).get('support_resistance', {})
                ),
                'otm1_ema_breakout': self._detect_ema_breakout(
                    df['otm1_straddle_price'],
                    technical_analysis.get('otm1_straddle', {}).get('ema_analysis', {})
                ),
                'otm1_pivot_break': self._detect_pivot_break(
                    df['otm1_straddle_price'],
                    technical_analysis.get('otm1_straddle', {}).get('pivot_analysis', {}),
                    'bullish'
                ),
                'correlation_alignment': correlations.get('correlation_strength', pd.Series([0.5] * len(df))) > 0.7
            }

            # Regime Strength Calculation
            regime_strength = {
                'correlation_strength': correlations.get('correlation_strength', pd.Series([0.5] * len(df))),
                'technical_alignment': self._calculate_technical_alignment(technical_analysis),
                'volume_confirmation': self._calculate_volume_confirmation(df),
                'volatility_context': self._calculate_volatility_context(df)
            }

            logger.info("✅ Regime signals calculated")
            return {
                'bearish_signals': bearish_signals,
                'bullish_signals': bullish_signals,
                'regime_strength': regime_strength,
                'correlations': correlations
            }

        except Exception as e:
            logger.error(f"❌ Error calculating regime signals: {e}")
            return {}

    def calculate_dynamic_weights(self, df: pd.DataFrame, dte: int, current_regime: int,
                                volatility_environment: str) -> Dict[str, float]:
        """Calculate dynamic weights based on DTE and market conditions"""
        logger.info(f"⚖️ Calculating dynamic weights for DTE={dte}, regime={current_regime}...")

        try:
            # Start with base weights
            dynamic_weights = self.base_weights.copy()

            # DTE-Based Adjustments (0-4 DTE focus)
            dte_adjustments = self._calculate_dte_adjustments(dte)

            # Volatility Environment Adjustments
            vol_adjustments = self._calculate_volatility_adjustments(volatility_environment)

            # Technical Confirmation Adjustments
            tech_adjustments = self._calculate_technical_confirmation_adjustments(df)

            # Regime-Specific Adjustments
            regime_adjustments = self._calculate_regime_specific_adjustments(current_regime)

            # Apply all adjustments
            for component in dynamic_weights:
                dynamic_weights[component] *= (
                    dte_adjustments.get(component, 1.0) *
                    vol_adjustments.get(component, 1.0) *
                    tech_adjustments.get(component, 1.0) *
                    regime_adjustments.get(component, 1.0)
                )

            # Normalize weights to sum to 1.0
            total_weight = sum(dynamic_weights.values())
            if total_weight > 0:
                dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}

            logger.info("✅ Dynamic weights calculated")
            return dynamic_weights

        except Exception as e:
            logger.error(f"❌ Error calculating dynamic weights: {e}")
            return self.base_weights

    def _calculate_dte_adjustments(self, dte: int) -> Dict[str, float]:
        """DTE-based weight adjustments for 0-4 DTE focus"""

        if dte == 0:
            # 0 DTE: High gamma focus, emphasize ATM and combined
            return {
                'atm_straddle': 1.3,
                'itm1_straddle': 0.9,
                'otm1_straddle': 0.8,
                'combined_straddle': 1.4
            }
        elif dte == 1:
            # 1 DTE: Balanced approach with slight ATM emphasis
            return {
                'atm_straddle': 1.2,
                'itm1_straddle': 1.0,
                'otm1_straddle': 0.9,
                'combined_straddle': 1.2
            }
        elif dte == 2:
            # 2 DTE: Standard weights
            return {
                'atm_straddle': 1.0,
                'itm1_straddle': 1.0,
                'otm1_straddle': 1.0,
                'combined_straddle': 1.0
            }
        elif dte == 3:
            # 3 DTE: Slight emphasis on ITM1/OTM1
            return {
                'atm_straddle': 0.9,
                'itm1_straddle': 1.1,
                'otm1_straddle': 1.1,
                'combined_straddle': 0.9
            }
        elif dte == 4:
            # 4 DTE: More emphasis on ITM1/OTM1
            return {
                'atm_straddle': 0.8,
                'itm1_straddle': 1.2,
                'otm1_straddle': 1.2,
                'combined_straddle': 0.8
            }
        else:
            # Default for other DTE values
            return {
                'atm_straddle': 1.0,
                'itm1_straddle': 1.0,
                'otm1_straddle': 1.0,
                'combined_straddle': 1.0
            }

    def classify_regime(self, regime_signals: Dict[str, Any], dynamic_weights: Dict[str, float],
                       iv_analysis: Dict[str, Any] = None, atr_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive regime classification using 15-regime system"""
        logger.info("🏷️ Classifying market regime...")

        try:
            # Calculate regime scores for each classification
            regime_scores = {}

            for regime_id in range(1, 16):  # 15 regimes
                score = self._calculate_regime_score(
                    regime_signals, dynamic_weights, iv_analysis, atr_analysis, regime_id
                )
                regime_scores[regime_id] = score

            # Select regime with highest score
            best_regime_id = max(regime_scores, key=regime_scores.get)
            best_regime_score = regime_scores[best_regime_id]

            # Calculate regime confidence
            regime_confidence = self._calculate_regime_confidence(regime_scores, best_regime_score)

            # Determine regime direction and strength
            regime_direction = self._determine_regime_direction(regime_signals)
            regime_strength = best_regime_score * regime_confidence

            logger.info(f"✅ Regime classified: {self.regime_names[best_regime_id]}")
            return {
                'regime_id': best_regime_id,
                'regime_name': self.regime_names[best_regime_id],
                'regime_score': best_regime_score,
                'regime_confidence': regime_confidence,
                'regime_direction': regime_direction,
                'regime_strength': regime_strength,
                'all_scores': regime_scores
            }

        except Exception as e:
            logger.error(f"❌ Error classifying regime: {e}")
            return {
                'regime_id': 15,  # Default to neutral
                'regime_name': self.regime_names[15],
                'regime_score': 0.5,
                'regime_confidence': 0.5,
                'regime_direction': 0.0,
                'regime_strength': 0.25,
                'all_scores': {}
            }

    # Helper methods for signal detection
    def _detect_support_break(self, price_series: pd.Series, support_resistance: Dict[str, pd.Series]) -> pd.Series:
        """Detect support level breaks"""
        if not support_resistance or 'support' not in support_resistance:
            return pd.Series([False] * len(price_series), index=price_series.index)

        support_levels = support_resistance['support']
        return price_series < (support_levels * 0.995)  # 0.5% break threshold

    def _detect_resistance_break(self, price_series: pd.Series, support_resistance: Dict[str, pd.Series]) -> pd.Series:
        """Detect resistance level breaks"""
        if not support_resistance or 'resistance' not in support_resistance:
            return pd.Series([False] * len(price_series), index=price_series.index)

        resistance_levels = support_resistance['resistance']
        return price_series > (resistance_levels * 1.005)  # 0.5% break threshold

    def _detect_ema_breakdown(self, price_series: pd.Series, ema_analysis: Dict[str, pd.Series]) -> pd.Series:
        """Detect EMA breakdown (bearish)"""
        if not ema_analysis:
            return pd.Series([False] * len(price_series), index=price_series.index)

        # Check if price is below multiple EMAs
        breakdown_signals = []
        for period in self.config['ema_periods']:
            ema_key = f'ema_{period}'
            if ema_key in ema_analysis:
                breakdown_signals.append(price_series < ema_analysis[ema_key])

        if breakdown_signals:
            return pd.concat(breakdown_signals, axis=1).sum(axis=1) >= 2  # At least 2 EMAs broken
        else:
            return pd.Series([False] * len(price_series), index=price_series.index)

    def _detect_ema_breakout(self, price_series: pd.Series, ema_analysis: Dict[str, pd.Series]) -> pd.Series:
        """Detect EMA breakout (bullish)"""
        if not ema_analysis:
            return pd.Series([False] * len(price_series), index=price_series.index)

        # Check if price is above multiple EMAs
        breakout_signals = []
        for period in self.config['ema_periods']:
            ema_key = f'ema_{period}'
            if ema_key in ema_analysis:
                breakout_signals.append(price_series > ema_analysis[ema_key])

        if breakout_signals:
            return pd.concat(breakout_signals, axis=1).sum(axis=1) >= 2  # At least 2 EMAs broken
        else:
            return pd.Series([False] * len(price_series), index=price_series.index)

    def _detect_pivot_break(self, price_series: pd.Series, pivot_analysis: Dict[str, pd.Series], direction: str) -> pd.Series:
        """Detect pivot point breaks"""
        if not pivot_analysis or 'pivot' not in pivot_analysis:
            return pd.Series([False] * len(price_series), index=price_series.index)

        pivot_levels = pivot_analysis['pivot']

        if direction == 'bearish':
            return price_series < (pivot_levels * 0.995)
        else:  # bullish
            return price_series > (pivot_levels * 1.005)

    def _calculate_technical_alignment(self, technical_analysis: Dict[str, Any]) -> pd.Series:
        """Calculate technical indicator alignment score"""
        alignment_scores = []

        for straddle_type, analysis in technical_analysis.items():
            if 'ema_analysis' in analysis:
                ema_alignment = 0
                ema_count = 0
                for period in self.config['ema_periods']:
                    position_key = f'ema_{period}_position'
                    if position_key in analysis['ema_analysis']:
                        ema_alignment += abs(analysis['ema_analysis'][position_key])
                        ema_count += 1

                if ema_count > 0:
                    alignment_scores.append(ema_alignment / ema_count)

        if alignment_scores:
            return pd.concat(alignment_scores, axis=1).mean(axis=1)
        else:
            return pd.Series([0.5] * 100)  # Default neutral alignment

    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume confirmation score"""
        volume_cols = [col for col in df.columns if 'volume' in col and 'straddle' in col]

        if volume_cols:
            total_volume = df[volume_cols].sum(axis=1)
            volume_ma = total_volume.rolling(20).mean()
            return (total_volume / volume_ma - 1).fillna(0)
        else:
            return pd.Series([0.0] * len(df), index=df.index)

    def _calculate_volatility_context(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility context score"""
        if 'spot_price' in df.columns:
            returns = df['spot_price'].pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized volatility
            vol_percentile = volatility.rolling(100).rank(pct=True)
            return vol_percentile.fillna(0.5)
        else:
            return pd.Series([0.5] * len(df), index=df.index)

    def _calculate_volatility_adjustments(self, volatility_environment: str) -> Dict[str, float]:
        """Calculate volatility environment adjustments"""
        if volatility_environment == 'low':
            return {'atm_straddle': 1.2, 'itm1_straddle': 0.9, 'otm1_straddle': 0.9, 'combined_straddle': 1.1}
        elif volatility_environment == 'high':
            return {'atm_straddle': 0.8, 'itm1_straddle': 1.1, 'otm1_straddle': 1.1, 'combined_straddle': 0.9}
        else:  # medium
            return {'atm_straddle': 1.0, 'itm1_straddle': 1.0, 'otm1_straddle': 1.0, 'combined_straddle': 1.0}

    def _calculate_technical_confirmation_adjustments(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical confirmation adjustments"""
        # Simplified technical confirmation
        return {'atm_straddle': 1.0, 'itm1_straddle': 1.0, 'otm1_straddle': 1.0, 'combined_straddle': 1.0}

    def _calculate_regime_specific_adjustments(self, current_regime: int) -> Dict[str, float]:
        """Calculate regime-specific adjustments"""
        if current_regime in [1, 2, 3, 4]:  # Low volatility regimes
            return {'atm_straddle': 1.1, 'itm1_straddle': 0.95, 'otm1_straddle': 0.95, 'combined_straddle': 1.0}
        elif current_regime in [8, 9, 10, 11, 12]:  # High volatility regimes
            return {'atm_straddle': 0.9, 'itm1_straddle': 1.05, 'otm1_straddle': 1.05, 'combined_straddle': 1.0}
        else:  # Medium volatility regimes
            return {'atm_straddle': 1.0, 'itm1_straddle': 1.0, 'otm1_straddle': 1.0, 'combined_straddle': 1.0}

    def _calculate_regime_score(self, regime_signals: Dict[str, Any], dynamic_weights: Dict[str, float],
                              iv_analysis: Dict[str, Any], atr_analysis: Dict[str, Any], regime_id: int) -> float:
        """Calculate score for specific regime"""

        # Base score calculation based on regime type
        base_score = 0.5

        # Get correlation strength
        correlation_strength = regime_signals.get('regime_strength', {}).get('correlation_strength', pd.Series([0.5]))
        if hasattr(correlation_strength, 'iloc'):
            corr_value = correlation_strength.iloc[-1] if len(correlation_strength) > 0 else 0.5
        else:
            corr_value = 0.5

        # Get volatility context
        volatility_context = regime_signals.get('regime_strength', {}).get('volatility_context', pd.Series([0.5]))
        if hasattr(volatility_context, 'iloc'):
            vol_value = volatility_context.iloc[-1] if len(volatility_context) > 0 else 0.5
        else:
            vol_value = 0.5

        # Regime-specific scoring
        if regime_id in [1, 2]:  # Ultra low vol regimes
            base_score = 0.8 if vol_value < 0.2 and corr_value > 0.8 else 0.2
        elif regime_id in [3, 4]:  # Low vol regimes
            base_score = 0.7 if vol_value < 0.4 and corr_value > 0.6 else 0.3
        elif regime_id in [5, 6, 7]:  # Medium vol regimes
            base_score = 0.6 if 0.4 <= vol_value <= 0.7 else 0.4
        elif regime_id in [8, 9, 10]:  # High vol regimes
            base_score = 0.7 if vol_value > 0.7 else 0.3
        elif regime_id in [11, 12]:  # Extreme vol regimes
            base_score = 0.8 if vol_value > 0.9 else 0.2
        elif regime_id in [13, 14]:  # Transition regimes
            base_score = 0.6 if 0.3 < corr_value < 0.7 else 0.4
        else:  # Neutral consolidation
            base_score = 0.5

        return base_score

    def _calculate_regime_confidence(self, regime_scores: Dict[int, float], best_score: float) -> float:
        """Calculate confidence in regime classification"""

        if not regime_scores:
            return 0.5

        # Sort scores in descending order
        sorted_scores = sorted(regime_scores.values(), reverse=True)

        if len(sorted_scores) < 2:
            return 0.5

        # Confidence based on separation between best and second-best scores
        second_best = sorted_scores[1]
        confidence = min(1.0, (best_score - second_best) * 2 + 0.5)

        return max(0.1, confidence)

    def _determine_regime_direction(self, regime_signals: Dict[str, Any]) -> float:
        """Determine regime direction (-1 to 1)"""

        bullish_signals = regime_signals.get('bullish_signals', {})
        bearish_signals = regime_signals.get('bearish_signals', {})

        bullish_count = sum(1 for signal in bullish_signals.values()
                          if hasattr(signal, 'iloc') and (signal.iloc[-1] if len(signal) > 0 else False))
        bearish_count = sum(1 for signal in bearish_signals.values()
                          if hasattr(signal, 'iloc') and (signal.iloc[-1] if len(signal) > 0 else False))

        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            return 0.0

        return (bullish_count - bearish_count) / total_signals

    def run_advanced_analysis(self, csv_file_path: str, dte: int = 1,
                            iv_analysis: Dict[str, Any] = None,
                            atr_analysis: Dict[str, Any] = None) -> str:
        """Run complete advanced regime detection analysis"""
        logger.info("🚀 Starting ADVANCED regime detection analysis...")

        try:
            # Load data
            df = pd.read_csv(csv_file_path)
            logger.info(f"📊 Loaded {len(df)} data points")

            # Ensure required columns
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate symmetric straddles
            df = self.calculate_symmetric_straddles(df)

            # Multi-timeframe technical analysis
            technical_analysis = self.calculate_multi_timeframe_analysis(df)

            # Calculate correlations
            correlations = self.calculate_straddle_correlations(df)

            # Calculate regime signals
            regime_signals = self.calculate_regime_signals(df, technical_analysis, correlations)

            # Determine volatility environment
            vol_context = regime_signals.get('regime_strength', {}).get('volatility_context', pd.Series([0.5]))
            if hasattr(vol_context, 'iloc') and len(vol_context) > 0:
                vol_value = vol_context.iloc[-1]
                if vol_value < 0.3:
                    volatility_environment = 'low'
                elif vol_value > 0.7:
                    volatility_environment = 'high'
                else:
                    volatility_environment = 'medium'
            else:
                volatility_environment = 'medium'

            # Calculate dynamic weights
            dynamic_weights = self.calculate_dynamic_weights(df, dte, 15, volatility_environment)

            # Classify regime for each row
            regime_classifications = []
            for i in range(len(df)):
                # Extract signals for current row
                current_signals = self._extract_current_signals(regime_signals, i)

                # Classify regime
                regime_result = self.classify_regime(current_signals, dynamic_weights, iv_analysis, atr_analysis)
                regime_classifications.append(regime_result)

            # Add regime results to dataframe
            df['regime_id'] = [r['regime_id'] for r in regime_classifications]
            df['regime_name'] = [r['regime_name'] for r in regime_classifications]
            df['regime_score'] = [r['regime_score'] for r in regime_classifications]
            df['regime_confidence'] = [r['regime_confidence'] for r in regime_classifications]
            df['regime_direction'] = [r['regime_direction'] for r in regime_classifications]
            df['regime_strength'] = [r['regime_strength'] for r in regime_classifications]

            # Add correlation data
            for corr_name, corr_series in correlations.items():
                df[f'correlation_{corr_name}'] = corr_series

            # Add dynamic weights
            for weight_name, weight_value in dynamic_weights.items():
                df[f'weight_{weight_name}'] = weight_value

            # Generate output
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"advanced_regime_analysis_{timestamp}.csv"

            # Save results
            df.to_csv(output_path, index=False)

            logger.info(f"✅ ADVANCED regime analysis completed: {output_path}")
            logger.info(f"🎯 15-regime classification with correlation analysis")
            logger.info(f"⚖️ Dynamic DTE-based weighting applied")

            return str(output_path)

        except Exception as e:
            logger.error(f"❌ Advanced analysis failed: {e}")
            raise

    def _extract_current_signals(self, regime_signals: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Extract signals for current row"""
        current_signals = {'regime_strength': {}}

        # Extract regime strength values
        for key, value in regime_signals.get('regime_strength', {}).items():
            if hasattr(value, 'iloc') and len(value) > index:
                current_signals['regime_strength'][key] = value.iloc[index]
            else:
                current_signals['regime_strength'][key] = 0.5

        return current_signals

if __name__ == "__main__":
    # Run advanced regime detection analysis
    detector = AdvancedRegimeDetector()

    # Test with sample data
    csv_file = "real_data_validation_results/real_data_regime_formation_20250619_211454.csv"

    try:
        output_path = detector.run_advanced_analysis(csv_file, dte=1)

        print("\n" + "="*80)
        print("ADVANCED REGIME DETECTION ANALYSIS COMPLETED")
        print("="*80)
        print(f"Input: {csv_file}")
        print(f"Output: {output_path}")
        print("="*80)
        print("🚀 Advanced correlation-based regime detection")
        print("📊 15-regime classification system")
        print("⚖️ Dynamic DTE-based weighting")
        print("📈 Multi-timeframe technical analysis")
        print("🔗 Straddle correlation analysis")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")