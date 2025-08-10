"""
Price RSI - Relative Strength Index for Underlying Asset
========================================================

Calculates traditional RSI on underlying asset price to identify
overbought/oversold conditions and momentum shifts.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PriceRSI:
    """
    Traditional RSI implementation for underlying asset
    
    Features:
    - Standard RSI calculation
    - Multi-timeframe RSI analysis
    - RSI divergence detection
    - Dynamic overbought/oversold levels
    - RSI trend analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Price RSI calculator"""
        # RSI periods
        self.primary_period = config.get('primary_period', 14)
        self.fast_period = config.get('fast_period', 7)
        self.slow_period = config.get('slow_period', 21)
        
        # Thresholds
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.extreme_overbought = config.get('extreme_overbought', 80)
        self.extreme_oversold = config.get('extreme_oversold', 20)
        
        # Dynamic threshold settings
        self.enable_dynamic_thresholds = config.get('enable_dynamic_thresholds', True)
        self.volatility_adjustment = config.get('volatility_adjustment', 0.2)
        
        # Advanced features
        self.enable_divergence_detection = config.get('enable_divergence_detection', True)
        self.enable_multi_timeframe = config.get('enable_multi_timeframe', True)
        self.smoothing_factor = config.get('smoothing_factor', 3)
        
        # History tracking
        self.rsi_history = {
            'primary': [],
            'fast': [],
            'slow': [],
            'divergences': []
        }
        
        logger.info(f"PriceRSI initialized: primary_period={self.primary_period}")
    
    def calculate_price_rsi(self,
                          price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive price RSI
        
        Args:
            price_data: DataFrame with underlying price data
            
        Returns:
            Dict with RSI values and analysis
        """
        try:
            results = {
                'primary_rsi': {},
                'multi_timeframe': {},
                'divergences': [],
                'signals': {},
                'trend': {},
                'regime': None
            }
            
            # Ensure we have price column
            if 'close' not in price_data.columns and 'price' not in price_data.columns:
                logger.error("No price column found in data")
                return self._get_default_results()
            
            price_col = 'close' if 'close' in price_data.columns else 'price'
            
            # Calculate primary RSI
            results['primary_rsi'] = self._calculate_rsi(
                price_data[price_col], 
                self.primary_period
            )
            
            # Multi-timeframe analysis
            if self.enable_multi_timeframe:
                results['multi_timeframe'] = {
                    'fast': self._calculate_rsi(price_data[price_col], self.fast_period),
                    'primary': results['primary_rsi'],
                    'slow': self._calculate_rsi(price_data[price_col], self.slow_period)
                }
            
            # Detect divergences
            if self.enable_divergence_detection:
                results['divergences'] = self._detect_divergences(
                    price_data[price_col], 
                    results
                )
            
            # Analyze trend
            results['trend'] = self._analyze_rsi_trend(results)
            
            # Generate signals
            results['signals'] = self._generate_rsi_signals(results)
            
            # Classify regime
            results['regime'] = self._classify_rsi_regime(results)
            
            # Update history
            self._update_rsi_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating price RSI: {e}")
            return self._get_default_results()
    
    def _calculate_rsi(self, 
                      price_series: pd.Series, 
                      period: int) -> Dict[str, Any]:
        """Calculate RSI for given period"""
        try:
            if len(price_series) < period:
                return {'value': 50.0, 'status': 'insufficient_data'}
            
            # Calculate price changes
            delta = price_series.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses (Wilder's smoothing)
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Handle initial period with simple average
            avg_gain.iloc[period-1] = gains.iloc[:period].mean()
            avg_loss.iloc[period-1] = losses.iloc[:period].mean()
            
            # Apply Wilder's smoothing for subsequent values
            for i in range(period, len(gains)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Handle division by zero
            rsi = rsi.fillna(100)  # If avg_loss is 0, RSI is 100
            
            # Get latest RSI value
            latest_rsi = rsi.iloc[-1]
            
            # Apply smoothing if enabled
            if self.smoothing_factor > 1:
                smoothed_rsi = rsi.rolling(window=self.smoothing_factor).mean()
                latest_rsi = smoothed_rsi.iloc[-1] if not pd.isna(smoothed_rsi.iloc[-1]) else latest_rsi
            
            # Calculate RSI statistics
            rsi_clean = rsi.dropna()
            
            return {
                'value': float(latest_rsi),
                'average': float(rsi_clean.mean()),
                'std': float(rsi_clean.std()),
                'trend': self._calculate_trend(rsi),
                'strength': self._calculate_strength(latest_rsi),
                'momentum': self._calculate_momentum(rsi),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return {'value': 50.0, 'status': 'error'}
    
    def _detect_divergences(self,
                          price_series: pd.Series,
                          rsi_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect price-RSI divergences"""
        try:
            divergences = []
            
            if 'value' not in rsi_results['primary_rsi']:
                return divergences
            
            # Need sufficient history for divergence detection
            if len(price_series) < 50:
                return divergences
            
            # Get RSI series
            rsi_series = self._reconstruct_rsi_series(price_series, self.primary_period)
            
            # Detect bullish divergence (price lower low, RSI higher low)
            bullish_div = self._detect_bullish_divergence(price_series, rsi_series)
            if bullish_div:
                divergences.append(bullish_div)
            
            # Detect bearish divergence (price higher high, RSI lower high)
            bearish_div = self._detect_bearish_divergence(price_series, rsi_series)
            if bearish_div:
                divergences.append(bearish_div)
            
            # Detect hidden divergences
            hidden_divs = self._detect_hidden_divergences(price_series, rsi_series)
            divergences.extend(hidden_divs)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {e}")
            return []
    
    def _analyze_rsi_trend(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze RSI trend across timeframes"""
        try:
            trend_analysis = {
                'primary_trend': 'neutral',
                'multi_tf_alignment': 'mixed',
                'trend_strength': 0.0,
                'reversal_probability': 0.0
            }
            
            # Analyze primary RSI trend
            if 'trend' in results['primary_rsi']:
                trend_analysis['primary_trend'] = results['primary_rsi']['trend']
            
            # Multi-timeframe alignment
            if self.enable_multi_timeframe and results['multi_timeframe']:
                trends = []
                for tf, rsi_data in results['multi_timeframe'].items():
                    if 'trend' in rsi_data:
                        trends.append(rsi_data['trend'])
                
                if all(t == 'rising' for t in trends):
                    trend_analysis['multi_tf_alignment'] = 'bullish_aligned'
                    trend_analysis['trend_strength'] = 1.0
                elif all(t == 'falling' for t in trends):
                    trend_analysis['multi_tf_alignment'] = 'bearish_aligned'
                    trend_analysis['trend_strength'] = -1.0
                else:
                    trend_analysis['multi_tf_alignment'] = 'mixed'
                    trend_analysis['trend_strength'] = 0.0
            
            # Calculate reversal probability
            current_rsi = results['primary_rsi'].get('value', 50)
            if current_rsi >= self.extreme_overbought:
                trend_analysis['reversal_probability'] = (current_rsi - self.extreme_overbought) / 20
            elif current_rsi <= self.extreme_oversold:
                trend_analysis['reversal_probability'] = (self.extreme_oversold - current_rsi) / 20
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing RSI trend: {e}")
            return {}
    
    def _generate_rsi_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from RSI analysis"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'signal_confidence': 0.0,
                'supporting_factors': [],
                'warning_signals': []
            }
            
            # Get primary RSI value
            rsi_value = results['primary_rsi'].get('value', 50)
            
            # Adjust thresholds if dynamic thresholds enabled
            overbought = self.overbought_threshold
            oversold = self.oversold_threshold
            
            if self.enable_dynamic_thresholds and 'std' in results['primary_rsi']:
                volatility = results['primary_rsi']['std']
                adjustment = volatility * self.volatility_adjustment
                overbought = min(self.overbought_threshold + adjustment, 85)
                oversold = max(self.oversold_threshold - adjustment, 15)
            
            # Generate primary signal
            if rsi_value >= self.extreme_overbought:
                signals['primary_signal'] = 'extreme_overbought'
                signals['signal_strength'] = -1.0
                signals['supporting_factors'].append('extreme_rsi_level')
            elif rsi_value >= overbought:
                signals['primary_signal'] = 'overbought'
                signals['signal_strength'] = -(rsi_value - 50) / 50
                signals['supporting_factors'].append('overbought_condition')
            elif rsi_value <= self.extreme_oversold:
                signals['primary_signal'] = 'extreme_oversold'
                signals['signal_strength'] = 1.0
                signals['supporting_factors'].append('extreme_rsi_level')
            elif rsi_value <= oversold:
                signals['primary_signal'] = 'oversold'
                signals['signal_strength'] = (50 - rsi_value) / 50
                signals['supporting_factors'].append('oversold_condition')
            else:
                signals['primary_signal'] = 'neutral'
                signals['signal_strength'] = (rsi_value - 50) / 50
            
            # Add multi-timeframe confirmation
            if self.enable_multi_timeframe and results['trend']['multi_tf_alignment'] != 'mixed':
                signals['supporting_factors'].append(f"multi_tf_{results['trend']['multi_tf_alignment']}")
                signals['signal_confidence'] += 0.3
            
            # Add divergence signals
            if results['divergences']:
                for div in results['divergences']:
                    if div['type'] == 'bullish_divergence':
                        signals['supporting_factors'].append('bullish_divergence')
                        if signals['primary_signal'] in ['oversold', 'extreme_oversold']:
                            signals['signal_confidence'] += 0.4
                    elif div['type'] == 'bearish_divergence':
                        signals['supporting_factors'].append('bearish_divergence')
                        if signals['primary_signal'] in ['overbought', 'extreme_overbought']:
                            signals['signal_confidence'] += 0.4
            
            # Add warning signals
            if results['trend'].get('reversal_probability', 0) > 0.7:
                signals['warning_signals'].append('high_reversal_probability')
            
            # Calculate final confidence
            base_confidence = abs(signals['signal_strength']) * 0.5
            signals['signal_confidence'] = min(base_confidence + signals.get('signal_confidence', 0), 1.0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            return {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            }
    
    def _classify_rsi_regime(self, results: Dict[str, Any]) -> str:
        """Classify market regime based on RSI analysis"""
        try:
            rsi_value = results['primary_rsi'].get('value', 50)
            rsi_std = results['primary_rsi'].get('std', 10)
            
            # Check extreme conditions
            if rsi_value >= self.extreme_overbought:
                return 'extreme_bullish_exhaustion'
            elif rsi_value <= self.extreme_oversold:
                return 'extreme_bearish_exhaustion'
            
            # Check standard conditions
            if rsi_value >= self.overbought_threshold:
                if rsi_std < 5:
                    return 'stable_overbought'
                else:
                    return 'volatile_overbought'
            elif rsi_value <= self.oversold_threshold:
                if rsi_std < 5:
                    return 'stable_oversold'
                else:
                    return 'volatile_oversold'
            
            # Check trend alignment
            if self.enable_multi_timeframe and results['trend']['multi_tf_alignment'] == 'bullish_aligned':
                return 'trending_bullish'
            elif self.enable_multi_timeframe and results['trend']['multi_tf_alignment'] == 'bearish_aligned':
                return 'trending_bearish'
            
            # Neutral conditions
            if 45 <= rsi_value <= 55:
                return 'neutral_equilibrium'
            elif rsi_value > 55:
                return 'mild_bullish'
            else:
                return 'mild_bearish'
                
        except Exception as e:
            logger.error(f"Error classifying RSI regime: {e}")
            return 'undefined'
    
    def _calculate_trend(self, rsi_series: pd.Series) -> str:
        """Calculate RSI trend direction"""
        try:
            if len(rsi_series) < 5:
                return 'neutral'
            
            recent_rsi = rsi_series.tail(5).dropna()
            if len(recent_rsi) < 3:
                return 'neutral'
            
            # Calculate slope
            x = range(len(recent_rsi))
            slope = np.polyfit(x, recent_rsi.values, 1)[0]
            
            if slope > 1:
                return 'rising'
            elif slope < -1:
                return 'falling'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _calculate_strength(self, rsi_value: float) -> str:
        """Calculate RSI signal strength"""
        if rsi_value >= self.extreme_overbought or rsi_value <= self.extreme_oversold:
            return 'extreme'
        elif rsi_value >= self.overbought_threshold or rsi_value <= self.oversold_threshold:
            return 'strong'
        elif 45 <= rsi_value <= 55:
            return 'neutral'
        else:
            return 'moderate'
    
    def _calculate_momentum(self, rsi_series: pd.Series) -> float:
        """Calculate RSI momentum (rate of change)"""
        try:
            if len(rsi_series) < 10:
                return 0.0
            
            recent_rsi = rsi_series.tail(10).dropna()
            if len(recent_rsi) < 5:
                return 0.0
            
            # Calculate rate of change
            current = recent_rsi.iloc[-1]
            previous = recent_rsi.iloc[-5]
            
            momentum = (current - previous) / 5  # Change per period
            return float(np.clip(momentum, -10, 10))
            
        except:
            return 0.0
    
    def _reconstruct_rsi_series(self, price_series: pd.Series, period: int) -> pd.Series:
        """Reconstruct full RSI series for divergence detection"""
        try:
            # Calculate price changes
            delta = price_series.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
            
        except:
            return pd.Series([50] * len(price_series))
    
    def _detect_bullish_divergence(self, 
                                 price_series: pd.Series, 
                                 rsi_series: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect bullish divergence pattern"""
        try:
            # Find recent price lows
            price_lows = self._find_local_extremes(price_series, 'min', window=10)
            
            if len(price_lows) < 2:
                return None
            
            # Check last two lows
            if price_lows[-1]['value'] < price_lows[-2]['value']:  # Lower low in price
                # Check corresponding RSI values
                rsi_at_low1 = rsi_series.iloc[price_lows[-2]['index']]
                rsi_at_low2 = rsi_series.iloc[price_lows[-1]['index']]
                
                if rsi_at_low2 > rsi_at_low1:  # Higher low in RSI
                    return {
                        'type': 'bullish_divergence',
                        'strength': abs(rsi_at_low2 - rsi_at_low1) / 20,
                        'price_points': [price_lows[-2]['value'], price_lows[-1]['value']],
                        'rsi_points': [float(rsi_at_low1), float(rsi_at_low2)],
                        'confirmation': 'pending'
                    }
            
            return None
            
        except:
            return None
    
    def _detect_bearish_divergence(self,
                                 price_series: pd.Series,
                                 rsi_series: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect bearish divergence pattern"""
        try:
            # Find recent price highs
            price_highs = self._find_local_extremes(price_series, 'max', window=10)
            
            if len(price_highs) < 2:
                return None
            
            # Check last two highs
            if price_highs[-1]['value'] > price_highs[-2]['value']:  # Higher high in price
                # Check corresponding RSI values
                rsi_at_high1 = rsi_series.iloc[price_highs[-2]['index']]
                rsi_at_high2 = rsi_series.iloc[price_highs[-1]['index']]
                
                if rsi_at_high2 < rsi_at_high1:  # Lower high in RSI
                    return {
                        'type': 'bearish_divergence',
                        'strength': abs(rsi_at_high1 - rsi_at_high2) / 20,
                        'price_points': [price_highs[-2]['value'], price_highs[-1]['value']],
                        'rsi_points': [float(rsi_at_high1), float(rsi_at_high2)],
                        'confirmation': 'pending'
                    }
            
            return None
            
        except:
            return None
    
    def _detect_hidden_divergences(self,
                                 price_series: pd.Series,
                                 rsi_series: pd.Series) -> List[Dict[str, Any]]:
        """Detect hidden divergence patterns"""
        # Simplified implementation
        # In production, would implement full hidden divergence detection
        return []
    
    def _find_local_extremes(self, 
                           series: pd.Series, 
                           extreme_type: str = 'max',
                           window: int = 10) -> List[Dict[str, Any]]:
        """Find local maxima or minima in series"""
        try:
            extremes = []
            
            if extreme_type == 'max':
                # Find local maxima
                for i in range(window, len(series) - window):
                    if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                        extremes.append({
                            'index': i,
                            'value': float(series.iloc[i])
                        })
            else:
                # Find local minima
                for i in range(window, len(series) - window):
                    if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                        extremes.append({
                            'index': i,
                            'value': float(series.iloc[i])
                        })
            
            return extremes
            
        except:
            return []
    
    def _update_rsi_history(self, results: Dict[str, Any]):
        """Update RSI history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update primary RSI history
            if 'value' in results['primary_rsi']:
                self.rsi_history['primary'].append({
                    'timestamp': timestamp,
                    'value': results['primary_rsi']['value'],
                    'regime': results['regime']
                })
            
            # Update multi-timeframe history
            if self.enable_multi_timeframe:
                for tf in ['fast', 'slow']:
                    if tf in results['multi_timeframe'] and 'value' in results['multi_timeframe'][tf]:
                        self.rsi_history[tf].append({
                            'timestamp': timestamp,
                            'value': results['multi_timeframe'][tf]['value']
                        })
            
            # Track divergences
            if results['divergences']:
                self.rsi_history['divergences'].extend([
                    {**div, 'timestamp': timestamp} for div in results['divergences']
                ])
            
            # Keep only recent history
            max_history = 100
            for key in self.rsi_history:
                if len(self.rsi_history[key]) > max_history:
                    self.rsi_history[key] = self.rsi_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating RSI history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'primary_rsi': {'value': 50.0, 'status': 'error'},
            'multi_timeframe': {},
            'divergences': [],
            'signals': {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            },
            'trend': {},
            'regime': 'undefined'
        }
    
    def get_rsi_analysis(self) -> Dict[str, Any]:
        """Get comprehensive RSI analysis summary"""
        try:
            if not self.rsi_history['primary']:
                return {'status': 'no_history'}
            
            recent_values = [h['value'] for h in self.rsi_history['primary'][-20:]]
            
            summary = {
                'current_rsi': recent_values[-1] if recent_values else 50.0,
                'average_rsi': np.mean(recent_values),
                'rsi_volatility': np.std(recent_values),
                'trend_consistency': self._calculate_trend_consistency(),
                'divergence_count': len(self.rsi_history['divergences']),
                'regime_distribution': self._calculate_regime_distribution()
            }
            
            # Add multi-timeframe summary
            if self.enable_multi_timeframe:
                summary['multi_tf_correlation'] = self._calculate_tf_correlation()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting RSI analysis: {e}")
            return {'status': 'error'}
    
    def _calculate_trend_consistency(self) -> float:
        """Calculate how consistent the RSI trend has been"""
        try:
            if len(self.rsi_history['primary']) < 10:
                return 0.0
            
            recent_values = [h['value'] for h in self.rsi_history['primary'][-10:]]
            
            # Count trend changes
            trend_changes = 0
            for i in range(1, len(recent_values)):
                if (recent_values[i] > 50) != (recent_values[i-1] > 50):
                    trend_changes += 1
            
            # More changes = less consistency
            consistency = 1.0 - (trend_changes / (len(recent_values) - 1))
            return float(consistency)
            
        except:
            return 0.0
    
    def _calculate_regime_distribution(self) -> Dict[str, float]:
        """Calculate distribution of RSI regimes"""
        try:
            if not self.rsi_history['primary']:
                return {}
            
            regimes = [h['regime'] for h in self.rsi_history['primary'] if 'regime' in h]
            total = len(regimes)
            
            if total == 0:
                return {}
            
            distribution = {}
            for regime in set(regimes):
                count = regimes.count(regime)
                distribution[regime] = count / total
            
            return distribution
            
        except:
            return {}
    
    def _calculate_tf_correlation(self) -> float:
        """Calculate correlation between timeframes"""
        try:
            if len(self.rsi_history['primary']) < 20:
                return 0.0
            
            # Get recent values for each timeframe
            primary_values = [h['value'] for h in self.rsi_history['primary'][-20:]]
            fast_values = [h['value'] for h in self.rsi_history['fast'][-20:]]
            
            if len(primary_values) != len(fast_values):
                return 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(primary_values, fast_values)[0, 1]
            return float(correlation)
            
        except:
            return 0.0