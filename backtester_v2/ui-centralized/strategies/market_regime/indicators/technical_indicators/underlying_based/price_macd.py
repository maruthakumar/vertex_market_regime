"""
Price MACD - Moving Average Convergence Divergence for Underlying Asset
======================================================================

Calculates traditional MACD on underlying asset price to identify
momentum changes and trend reversals.

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


class PriceMACD:
    """
    Traditional MACD implementation for underlying asset
    
    Features:
    - Standard MACD calculation (12, 26, 9)
    - Signal line crossovers
    - Histogram analysis
    - Zero-line crossovers
    - MACD divergence detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Price MACD calculator"""
        # MACD parameters
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
        
        # Signal thresholds
        self.histogram_threshold = config.get('histogram_threshold', 0.0)
        self.signal_strength_threshold = config.get('signal_strength_threshold', 0.5)
        
        # Advanced features
        self.enable_divergence_detection = config.get('enable_divergence_detection', True)
        self.enable_crossover_confirmation = config.get('enable_crossover_confirmation', True)
        
        # History tracking
        self.macd_history = {
            'macd': [],
            'signal': [],
            'histogram': [],
            'crossovers': []
        }
        
        logger.info(f"PriceMACD initialized: fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")
    
    def calculate_price_macd(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive price MACD
        
        Args:
            price_data: DataFrame with underlying price data
            
        Returns:
            Dict with MACD values and analysis
        """
        try:
            results = {
                'macd_line': {},
                'signal_line': {},
                'histogram': {},
                'crossovers': [],
                'divergences': [],
                'signals': {},
                'momentum': None
            }
            
            # Ensure we have price column
            if 'close' not in price_data.columns and 'price' not in price_data.columns:
                logger.error("No price column found in data")
                return self._get_default_results()
            
            price_col = 'close' if 'close' in price_data.columns else 'price'
            price_series = price_data[price_col]
            
            if len(price_series) < self.slow_period:
                return self._get_default_results()
            
            # Calculate EMAs
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Get latest values
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            latest_histogram = histogram.iloc[-1]
            
            results['macd_line'] = {
                'value': float(latest_macd),
                'trend': self._calculate_trend(macd_line)
            }
            
            results['signal_line'] = {
                'value': float(latest_signal),
                'trend': self._calculate_trend(signal_line)
            }
            
            results['histogram'] = {
                'value': float(latest_histogram),
                'trend': self._calculate_histogram_trend(histogram),
                'strength': self._calculate_histogram_strength(latest_histogram)
            }
            
            # Detect crossovers
            results['crossovers'] = self._detect_crossovers(macd_line, signal_line, histogram)
            
            # Detect divergences
            if self.enable_divergence_detection:
                results['divergences'] = self._detect_macd_divergences(price_series, macd_line, histogram)
            
            # Generate signals
            results['signals'] = self._generate_macd_signals(results)
            
            # Classify momentum
            results['momentum'] = self._classify_momentum(results)
            
            # Update history
            self._update_macd_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating price MACD: {e}")
            return self._get_default_results()
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        try:
            if len(series) < 5:
                return 'neutral'
            
            recent_values = series.tail(5).values
            slope = np.polyfit(range(5), recent_values, 1)[0]
            
            if slope > 0.01:
                return 'rising'
            elif slope < -0.01:
                return 'falling'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _calculate_histogram_trend(self, histogram: pd.Series) -> str:
        """Calculate histogram trend (momentum of momentum)"""
        try:
            if len(histogram) < 5:
                return 'neutral'
            
            recent_hist = histogram.tail(5).values
            
            # Check if histogram is increasing or decreasing
            if all(recent_hist[i] > recent_hist[i-1] for i in range(1, len(recent_hist))):
                return 'accelerating'
            elif all(recent_hist[i] < recent_hist[i-1] for i in range(1, len(recent_hist))):
                return 'decelerating'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _calculate_histogram_strength(self, histogram_value: float) -> str:
        """Calculate histogram signal strength"""
        abs_hist = abs(histogram_value)
        
        if abs_hist > self.signal_strength_threshold:
            return 'strong'
        elif abs_hist > self.histogram_threshold:
            return 'moderate'
        else:
            return 'weak'
    
    def _detect_crossovers(self, 
                         macd_line: pd.Series, 
                         signal_line: pd.Series,
                         histogram: pd.Series) -> List[Dict[str, Any]]:
        """Detect MACD crossovers"""
        try:
            crossovers = []
            
            # Need at least 2 values
            if len(macd_line) < 2:
                return crossovers
            
            # MACD-Signal crossover
            if self._check_crossover(macd_line, signal_line):
                crossover_type = 'bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'bearish'
                crossovers.append({
                    'type': 'macd_signal_crossover',
                    'direction': crossover_type,
                    'strength': abs(histogram.iloc[-1]),
                    'location': 'above_zero' if macd_line.iloc[-1] > 0 else 'below_zero'
                })
            
            # Zero-line crossover
            if self._check_zero_crossover(macd_line):
                crossover_type = 'bullish' if macd_line.iloc[-1] > 0 else 'bearish'
                crossovers.append({
                    'type': 'zero_line_crossover',
                    'direction': crossover_type,
                    'momentum': 'positive' if macd_line.iloc[-1] > 0 else 'negative'
                })
            
            return crossovers
            
        except Exception as e:
            logger.error(f"Error detecting crossovers: {e}")
            return []
    
    def _check_crossover(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Check if two series have crossed"""
        try:
            if len(series1) < 2 or len(series2) < 2:
                return False
            
            # Check if sign of difference changed
            prev_diff = series1.iloc[-2] - series2.iloc[-2]
            curr_diff = series1.iloc[-1] - series2.iloc[-1]
            
            return (prev_diff * curr_diff) < 0
            
        except:
            return False
    
    def _check_zero_crossover(self, series: pd.Series) -> bool:
        """Check if series crossed zero"""
        try:
            if len(series) < 2:
                return False
            
            return (series.iloc[-2] * series.iloc[-1]) < 0
            
        except:
            return False
    
    def _detect_macd_divergences(self,
                               price_series: pd.Series,
                               macd_line: pd.Series,
                               histogram: pd.Series) -> List[Dict[str, Any]]:
        """Detect MACD divergences"""
        try:
            divergences = []
            
            # Need sufficient history
            if len(price_series) < 50:
                return divergences
            
            # Find price extremes
            price_highs = self._find_extremes(price_series, 'max')
            price_lows = self._find_extremes(price_series, 'min')
            
            # Check for bullish divergence
            if len(price_lows) >= 2:
                if price_lows[-1]['value'] < price_lows[-2]['value']:  # Lower low in price
                    macd_at_low1 = macd_line.iloc[price_lows[-2]['index']]
                    macd_at_low2 = macd_line.iloc[price_lows[-1]['index']]
                    
                    if macd_at_low2 > macd_at_low1:  # Higher low in MACD
                        divergences.append({
                            'type': 'bullish_divergence',
                            'strength': abs(macd_at_low2 - macd_at_low1),
                            'confirmed': histogram.iloc[-1] > 0
                        })
            
            # Check for bearish divergence
            if len(price_highs) >= 2:
                if price_highs[-1]['value'] > price_highs[-2]['value']:  # Higher high in price
                    macd_at_high1 = macd_line.iloc[price_highs[-2]['index']]
                    macd_at_high2 = macd_line.iloc[price_highs[-1]['index']]
                    
                    if macd_at_high2 < macd_at_high1:  # Lower high in MACD
                        divergences.append({
                            'type': 'bearish_divergence',
                            'strength': abs(macd_at_high1 - macd_at_high2),
                            'confirmed': histogram.iloc[-1] < 0
                        })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting MACD divergences: {e}")
            return []
    
    def _find_extremes(self, series: pd.Series, extreme_type: str = 'max') -> List[Dict[str, Any]]:
        """Find local extremes in series"""
        try:
            extremes = []
            window = 10
            
            for i in range(window, len(series) - window):
                if extreme_type == 'max':
                    if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                        extremes.append({'index': i, 'value': float(series.iloc[i])})
                else:
                    if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                        extremes.append({'index': i, 'value': float(series.iloc[i])})
            
            return extremes
            
        except:
            return []
    
    def _generate_macd_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from MACD analysis"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'confirmation_factors': [],
                'trade_bias': 'neutral'
            }
            
            # Analyze histogram
            histogram_value = results['histogram']['value']
            histogram_trend = results['histogram']['trend']
            
            # Basic signal from histogram
            if histogram_value > self.signal_strength_threshold:
                signals['primary_signal'] = 'bullish'
                signals['signal_strength'] = min(histogram_value, 1.0)
            elif histogram_value < -self.signal_strength_threshold:
                signals['primary_signal'] = 'bearish'
                signals['signal_strength'] = max(histogram_value, -1.0)
            
            # Add crossover signals
            for crossover in results['crossovers']:
                if crossover['type'] == 'macd_signal_crossover':
                    if crossover['direction'] == 'bullish':
                        signals['confirmation_factors'].append('bullish_crossover')
                        if crossover['location'] == 'above_zero':
                            signals['trade_bias'] = 'strong_bullish'
                    else:
                        signals['confirmation_factors'].append('bearish_crossover')
                        if crossover['location'] == 'below_zero':
                            signals['trade_bias'] = 'strong_bearish'
                
                elif crossover['type'] == 'zero_line_crossover':
                    signals['confirmation_factors'].append(f"{crossover['direction']}_zero_cross")
            
            # Add divergence signals
            for divergence in results['divergences']:
                if divergence['type'] == 'bullish_divergence' and divergence.get('confirmed', False):
                    signals['confirmation_factors'].append('confirmed_bullish_divergence')
                    signals['trade_bias'] = 'bullish' if signals['trade_bias'] == 'neutral' else signals['trade_bias']
                elif divergence['type'] == 'bearish_divergence' and divergence.get('confirmed', False):
                    signals['confirmation_factors'].append('confirmed_bearish_divergence')
                    signals['trade_bias'] = 'bearish' if signals['trade_bias'] == 'neutral' else signals['trade_bias']
            
            # Histogram trend confirmation
            if histogram_trend == 'accelerating' and histogram_value > 0:
                signals['confirmation_factors'].append('bullish_acceleration')
            elif histogram_trend == 'accelerating' and histogram_value < 0:
                signals['confirmation_factors'].append('bearish_acceleration')
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating MACD signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _classify_momentum(self, results: Dict[str, Any]) -> str:
        """Classify momentum state based on MACD"""
        try:
            macd_value = results['macd_line']['value']
            histogram_value = results['histogram']['value']
            histogram_trend = results['histogram']['trend']
            
            # Strong momentum states
            if macd_value > 0 and histogram_value > self.signal_strength_threshold:
                if histogram_trend == 'accelerating':
                    return 'strong_bullish_accelerating'
                else:
                    return 'strong_bullish_momentum'
            elif macd_value < 0 and histogram_value < -self.signal_strength_threshold:
                if histogram_trend == 'accelerating':
                    return 'strong_bearish_accelerating'
                else:
                    return 'strong_bearish_momentum'
            
            # Moderate momentum states
            elif macd_value > 0 and histogram_value > 0:
                return 'bullish_momentum'
            elif macd_value < 0 and histogram_value < 0:
                return 'bearish_momentum'
            
            # Weakening momentum
            elif macd_value > 0 and histogram_value < 0:
                return 'bullish_weakening'
            elif macd_value < 0 and histogram_value > 0:
                return 'bearish_weakening'
            
            else:
                return 'neutral_momentum'
                
        except Exception as e:
            logger.error(f"Error classifying momentum: {e}")
            return 'undefined'
    
    def _update_macd_history(self, results: Dict[str, Any]):
        """Update MACD history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update MACD values
            self.macd_history['macd'].append({
                'timestamp': timestamp,
                'value': results['macd_line']['value']
            })
            
            self.macd_history['signal'].append({
                'timestamp': timestamp,
                'value': results['signal_line']['value']
            })
            
            self.macd_history['histogram'].append({
                'timestamp': timestamp,
                'value': results['histogram']['value'],
                'momentum': results['momentum']
            })
            
            # Track crossovers
            if results['crossovers']:
                self.macd_history['crossovers'].extend([
                    {**crossover, 'timestamp': timestamp} for crossover in results['crossovers']
                ])
            
            # Keep only recent history
            max_history = 100
            for key in self.macd_history:
                if len(self.macd_history[key]) > max_history:
                    self.macd_history[key] = self.macd_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating MACD history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'macd_line': {'value': 0.0, 'trend': 'neutral'},
            'signal_line': {'value': 0.0, 'trend': 'neutral'},
            'histogram': {'value': 0.0, 'trend': 'neutral', 'strength': 'weak'},
            'crossovers': [],
            'divergences': [],
            'signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'momentum': 'undefined'
        }
    
    def get_macd_analysis(self) -> Dict[str, Any]:
        """Get comprehensive MACD analysis summary"""
        try:
            if not self.macd_history['histogram']:
                return {'status': 'no_history'}
            
            recent_histograms = [h['value'] for h in self.macd_history['histogram'][-20:]]
            
            return {
                'current_histogram': recent_histograms[-1] if recent_histograms else 0.0,
                'average_histogram': np.mean(recent_histograms),
                'histogram_volatility': np.std(recent_histograms),
                'recent_crossovers': len([c for c in self.macd_history['crossovers'] if c.get('timestamp', datetime.min) > datetime.now() - pd.Timedelta(hours=1)]),
                'momentum_distribution': self._calculate_momentum_distribution()
            }
            
        except Exception as e:
            logger.error(f"Error getting MACD analysis: {e}")
            return {'status': 'error'}
    
    def _calculate_momentum_distribution(self) -> Dict[str, float]:
        """Calculate distribution of momentum states"""
        try:
            if not self.macd_history['histogram']:
                return {}
            
            momentum_states = [h['momentum'] for h in self.macd_history['histogram'] if 'momentum' in h]
            total = len(momentum_states)
            
            if total == 0:
                return {}
            
            distribution = {}
            for state in set(momentum_states):
                count = momentum_states.count(state)
                distribution[state] = count / total
            
            return distribution
            
        except:
            return {}