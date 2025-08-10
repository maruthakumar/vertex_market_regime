"""
Price Bollinger Bands - Bollinger Bands for Underlying Asset
============================================================

Calculates traditional Bollinger Bands on underlying asset price to identify
volatility expansions, contractions, and mean reversion opportunities.

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


class PriceBollinger:
    """
    Traditional Bollinger Bands implementation for underlying asset
    
    Features:
    - Standard Bollinger Bands calculation
    - Band squeeze detection
    - Band breakout identification
    - %B and Band Width indicators
    - Mean reversion signals
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Price Bollinger Bands calculator"""
        # Bollinger parameters
        self.period = config.get('period', 20)
        self.num_std = config.get('num_std', 2.0)
        
        # Band width thresholds
        self.squeeze_threshold = config.get('squeeze_threshold', 0.05)
        self.expansion_threshold = config.get('expansion_threshold', 0.15)
        
        # Breakout settings
        self.breakout_confirmation_bars = config.get('breakout_confirmation_bars', 2)
        self.mean_reversion_threshold = config.get('mean_reversion_threshold', 0.95)
        
        # Advanced features
        self.enable_squeeze_detection = config.get('enable_squeeze_detection', True)
        self.enable_multi_timeframe = config.get('enable_multi_timeframe', True)
        
        # History tracking
        self.band_history = {
            'upper': [],
            'middle': [],
            'lower': [],
            'width': [],
            'percent_b': [],
            'squeezes': [],
            'breakouts': []
        }
        
        logger.info(f"PriceBollinger initialized: period={self.period}, std={self.num_std}")
    
    def calculate_price_bollinger(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive Bollinger Bands
        
        Args:
            price_data: DataFrame with underlying price data
            
        Returns:
            Dict with band values and analysis
        """
        try:
            results = {
                'bands': {},
                'indicators': {},
                'squeezes': [],
                'breakouts': [],
                'signals': {},
                'volatility_state': None
            }
            
            # Ensure we have price column
            if 'close' not in price_data.columns and 'price' not in price_data.columns:
                logger.error("No price column found in data")
                return self._get_default_results()
            
            price_col = 'close' if 'close' in price_data.columns else 'price'
            price_series = price_data[price_col]
            
            if len(price_series) < self.period:
                return self._get_default_results()
            
            # Calculate moving average (middle band)
            sma = price_series.rolling(window=self.period).mean()
            
            # Calculate standard deviation
            std = price_series.rolling(window=self.period).std()
            
            # Calculate bands
            upper_band = sma + (self.num_std * std)
            lower_band = sma - (self.num_std * std)
            
            # Get latest values
            latest_price = price_series.iloc[-1]
            latest_sma = sma.iloc[-1]
            latest_std = std.iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            
            # Calculate band width
            band_width = (latest_upper - latest_lower) / latest_sma if latest_sma > 0 else 0
            
            # Calculate %B (position within bands)
            percent_b = (latest_price - latest_lower) / (latest_upper - latest_lower) if latest_upper != latest_lower else 0.5
            
            results['bands'] = {
                'upper': float(latest_upper),
                'middle': float(latest_sma),
                'lower': float(latest_lower),
                'current_price': float(latest_price),
                'std': float(latest_std)
            }
            
            results['indicators'] = {
                'band_width': float(band_width),
                'percent_b': float(percent_b),
                'width_percentile': self._calculate_width_percentile(band_width),
                'price_position': self._classify_price_position(percent_b)
            }
            
            # Detect squeezes
            if self.enable_squeeze_detection:
                results['squeezes'] = self._detect_squeezes(band_width, std)
            
            # Detect breakouts
            results['breakouts'] = self._detect_breakouts(price_series, upper_band, lower_band, percent_b)
            
            # Generate signals
            results['signals'] = self._generate_bollinger_signals(results)
            
            # Classify volatility state
            results['volatility_state'] = self._classify_volatility_state(band_width, results)
            
            # Update history
            self._update_band_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return self._get_default_results()
    
    def _calculate_width_percentile(self, current_width: float) -> float:
        """Calculate percentile rank of current band width"""
        try:
            if len(self.band_history['width']) < 20:
                return 50.0
            
            recent_widths = [h['value'] for h in self.band_history['width'][-100:]]
            percentile = (sum(1 for w in recent_widths if w < current_width) / len(recent_widths)) * 100
            
            return float(percentile)
            
        except:
            return 50.0
    
    def _classify_price_position(self, percent_b: float) -> str:
        """Classify price position relative to bands"""
        if percent_b > 1.0:
            return 'above_upper_band'
        elif percent_b > self.mean_reversion_threshold:
            return 'near_upper_band'
        elif percent_b < 0.0:
            return 'below_lower_band'
        elif percent_b < (1 - self.mean_reversion_threshold):
            return 'near_lower_band'
        elif 0.4 <= percent_b <= 0.6:
            return 'middle_band'
        elif percent_b > 0.6:
            return 'upper_half'
        else:
            return 'lower_half'
    
    def _detect_squeezes(self, band_width: float, std: pd.Series) -> List[Dict[str, Any]]:
        """Detect Bollinger Band squeezes"""
        try:
            squeezes = []
            
            # Check for band width squeeze
            if band_width < self.squeeze_threshold:
                # Calculate squeeze intensity
                if len(self.band_history['width']) > 20:
                    recent_widths = [h['value'] for h in self.band_history['width'][-20:]]
                    avg_width = np.mean(recent_widths)
                    squeeze_intensity = 1 - (band_width / avg_width) if avg_width > 0 else 0
                else:
                    squeeze_intensity = 0.5
                
                squeezes.append({
                    'type': 'band_squeeze',
                    'band_width': band_width,
                    'intensity': float(squeeze_intensity),
                    'duration': self._calculate_squeeze_duration()
                })
            
            # Check for volatility contraction
            if len(std) >= 20:
                recent_std = std.tail(20)
                if recent_std.iloc[-1] < recent_std.mean() * 0.7:
                    squeezes.append({
                        'type': 'volatility_contraction',
                        'current_vol': float(recent_std.iloc[-1]),
                        'avg_vol': float(recent_std.mean())
                    })
            
            return squeezes
            
        except Exception as e:
            logger.error(f"Error detecting squeezes: {e}")
            return []
    
    def _detect_breakouts(self,
                        price_series: pd.Series,
                        upper_band: pd.Series,
                        lower_band: pd.Series,
                        percent_b: float) -> List[Dict[str, Any]]:
        """Detect band breakouts"""
        try:
            breakouts = []
            
            # Check for upper band breakout
            if percent_b > 1.0:
                breakout_strength = percent_b - 1.0
                confirmed = self._confirm_breakout(price_series, upper_band, 'upper')
                
                breakouts.append({
                    'type': 'upper_band_breakout',
                    'strength': float(breakout_strength),
                    'confirmed': confirmed,
                    'price_above_band': float(price_series.iloc[-1] - upper_band.iloc[-1])
                })
            
            # Check for lower band breakout
            elif percent_b < 0.0:
                breakout_strength = abs(percent_b)
                confirmed = self._confirm_breakout(price_series, lower_band, 'lower')
                
                breakouts.append({
                    'type': 'lower_band_breakout',
                    'strength': float(breakout_strength),
                    'confirmed': confirmed,
                    'price_below_band': float(lower_band.iloc[-1] - price_series.iloc[-1])
                })
            
            # Check for band walk
            if len(price_series) >= 5:
                if self._detect_band_walk(price_series, upper_band, lower_band):
                    breakouts.append({
                        'type': 'band_walk',
                        'direction': 'upper' if percent_b > 0.8 else 'lower'
                    })
            
            return breakouts
            
        except Exception as e:
            logger.error(f"Error detecting breakouts: {e}")
            return []
    
    def _confirm_breakout(self,
                        price_series: pd.Series,
                        band_series: pd.Series,
                        band_type: str) -> bool:
        """Confirm if breakout is valid"""
        try:
            if len(price_series) < self.breakout_confirmation_bars:
                return False
            
            # Check if price has been outside band for confirmation period
            recent_prices = price_series.tail(self.breakout_confirmation_bars)
            recent_bands = band_series.tail(self.breakout_confirmation_bars)
            
            if band_type == 'upper':
                return all(price > band for price, band in zip(recent_prices, recent_bands))
            else:
                return all(price < band for price, band in zip(recent_prices, recent_bands))
                
        except:
            return False
    
    def _detect_band_walk(self,
                        price_series: pd.Series,
                        upper_band: pd.Series,
                        lower_band: pd.Series) -> bool:
        """Detect if price is walking along a band"""
        try:
            recent_prices = price_series.tail(5)
            recent_upper = upper_band.tail(5)
            recent_lower = lower_band.tail(5)
            
            # Check upper band walk
            upper_touches = sum(1 for p, u in zip(recent_prices, recent_upper) if p >= u * 0.98)
            if upper_touches >= 4:
                return True
            
            # Check lower band walk
            lower_touches = sum(1 for p, l in zip(recent_prices, recent_lower) if p <= l * 1.02)
            if lower_touches >= 4:
                return True
            
            return False
            
        except:
            return False
    
    def _generate_bollinger_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from Bollinger Bands"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'mean_reversion': False,
                'trend_following': False,
                'volatility_signal': 'normal'
            }
            
            percent_b = results['indicators']['percent_b']
            band_width = results['indicators']['band_width']
            price_position = results['indicators']['price_position']
            
            # Mean reversion signals
            if price_position == 'above_upper_band':
                signals['primary_signal'] = 'overbought'
                signals['signal_strength'] = -min(percent_b - 1.0, 1.0)
                signals['mean_reversion'] = True
            elif price_position == 'below_lower_band':
                signals['primary_signal'] = 'oversold'
                signals['signal_strength'] = min(abs(percent_b), 1.0)
                signals['mean_reversion'] = True
            
            # Trend following signals (breakouts)
            for breakout in results['breakouts']:
                if breakout.get('confirmed', False):
                    if breakout['type'] == 'upper_band_breakout':
                        signals['primary_signal'] = 'bullish_breakout'
                        signals['signal_strength'] = 0.8
                        signals['trend_following'] = True
                    elif breakout['type'] == 'lower_band_breakout':
                        signals['primary_signal'] = 'bearish_breakout'
                        signals['signal_strength'] = -0.8
                        signals['trend_following'] = True
            
            # Squeeze signals
            if results['squeezes']:
                signals['volatility_signal'] = 'squeeze'
                if signals['primary_signal'] == 'neutral':
                    signals['primary_signal'] = 'consolidation'
            
            # Band expansion signals
            elif band_width > self.expansion_threshold:
                signals['volatility_signal'] = 'expansion'
            
            # Neutral zone signals
            if price_position == 'middle_band' and signals['primary_signal'] == 'neutral':
                signals['primary_signal'] = 'neutral_equilibrium'
                signals['signal_strength'] = 0.0
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Bollinger signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _classify_volatility_state(self, band_width: float, results: Dict[str, Any]) -> str:
        """Classify volatility state based on bands"""
        try:
            width_percentile = results['indicators']['width_percentile']
            
            # Check for squeezes
            if results['squeezes']:
                if any(s['type'] == 'band_squeeze' for s in results['squeezes']):
                    return 'extreme_compression'
                else:
                    return 'volatility_contraction'
            
            # Check for expansion
            elif band_width > self.expansion_threshold:
                if width_percentile > 90:
                    return 'extreme_expansion'
                else:
                    return 'volatility_expansion'
            
            # Normal states
            elif width_percentile > 70:
                return 'high_volatility'
            elif width_percentile < 30:
                return 'low_volatility'
            else:
                return 'normal_volatility'
                
        except Exception as e:
            logger.error(f"Error classifying volatility state: {e}")
            return 'undefined'
    
    def _calculate_squeeze_duration(self) -> int:
        """Calculate how long the current squeeze has lasted"""
        try:
            if not self.band_history['width']:
                return 0
            
            # Count consecutive periods below squeeze threshold
            duration = 0
            for i in range(len(self.band_history['width']) - 1, -1, -1):
                if self.band_history['width'][i]['value'] < self.squeeze_threshold:
                    duration += 1
                else:
                    break
            
            return duration
            
        except:
            return 0
    
    def _update_band_history(self, results: Dict[str, Any]):
        """Update band history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update band values
            self.band_history['upper'].append({
                'timestamp': timestamp,
                'value': results['bands']['upper']
            })
            
            self.band_history['middle'].append({
                'timestamp': timestamp,
                'value': results['bands']['middle']
            })
            
            self.band_history['lower'].append({
                'timestamp': timestamp,
                'value': results['bands']['lower']
            })
            
            self.band_history['width'].append({
                'timestamp': timestamp,
                'value': results['indicators']['band_width']
            })
            
            self.band_history['percent_b'].append({
                'timestamp': timestamp,
                'value': results['indicators']['percent_b']
            })
            
            # Track squeezes and breakouts
            if results['squeezes']:
                self.band_history['squeezes'].extend([
                    {**squeeze, 'timestamp': timestamp} for squeeze in results['squeezes']
                ])
            
            if results['breakouts']:
                self.band_history['breakouts'].extend([
                    {**breakout, 'timestamp': timestamp} for breakout in results['breakouts']
                ])
            
            # Keep only recent history
            max_history = 200
            for key in self.band_history:
                if len(self.band_history[key]) > max_history:
                    self.band_history[key] = self.band_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating band history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'bands': {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0,
                'current_price': 0.0,
                'std': 0.0
            },
            'indicators': {
                'band_width': 0.0,
                'percent_b': 0.5,
                'width_percentile': 50.0,
                'price_position': 'unknown'
            },
            'squeezes': [],
            'breakouts': [],
            'signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'volatility_state': 'undefined'
        }
    
    def get_bollinger_analysis(self) -> Dict[str, Any]:
        """Get comprehensive Bollinger Bands analysis summary"""
        try:
            if not self.band_history['width']:
                return {'status': 'no_history'}
            
            recent_widths = [h['value'] for h in self.band_history['width'][-20:]]
            recent_percent_b = [h['value'] for h in self.band_history['percent_b'][-20:]]
            
            return {
                'current_width': recent_widths[-1] if recent_widths else 0.0,
                'average_width': np.mean(recent_widths),
                'width_trend': self._calculate_width_trend(),
                'average_percent_b': np.mean(recent_percent_b),
                'squeeze_count': len(self.band_history['squeezes']),
                'breakout_count': len(self.band_history['breakouts']),
                'volatility_cycles': self._count_volatility_cycles()
            }
            
        except Exception as e:
            logger.error(f"Error getting Bollinger analysis: {e}")
            return {'status': 'error'}
    
    def _calculate_width_trend(self) -> str:
        """Calculate band width trend"""
        try:
            if len(self.band_history['width']) < 10:
                return 'neutral'
            
            recent_widths = [h['value'] for h in self.band_history['width'][-10:]]
            
            # Calculate trend
            x = range(len(recent_widths))
            slope = np.polyfit(x, recent_widths, 1)[0]
            
            if slope > 0.001:
                return 'expanding'
            elif slope < -0.001:
                return 'contracting'
            else:
                return 'stable'
                
        except:
            return 'neutral'
    
    def _count_volatility_cycles(self) -> int:
        """Count number of volatility expansion/contraction cycles"""
        try:
            if len(self.band_history['width']) < 20:
                return 0
            
            widths = [h['value'] for h in self.band_history['width']]
            
            # Count transitions from low to high volatility
            cycles = 0
            in_low_vol = widths[0] < self.squeeze_threshold
            
            for width in widths[1:]:
                if in_low_vol and width > self.expansion_threshold:
                    cycles += 1
                    in_low_vol = False
                elif not in_low_vol and width < self.squeeze_threshold:
                    in_low_vol = True
            
            return cycles
            
        except:
            return 0