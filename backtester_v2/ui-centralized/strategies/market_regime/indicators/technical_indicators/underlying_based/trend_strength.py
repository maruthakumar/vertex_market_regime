"""
Trend Strength - Comprehensive Trend Analysis for Underlying Asset
=================================================================

Analyzes trend strength using multiple indicators including ADX, 
moving averages, and price action patterns.

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


class TrendStrength:
    """
    Comprehensive trend strength analyzer
    
    Features:
    - ADX (Average Directional Index) calculation
    - Multiple moving average analysis
    - Trend consistency measurement
    - Support/Resistance level detection
    - Price action pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Trend Strength analyzer"""
        # ADX parameters
        self.adx_period = config.get('adx_period', 14)
        self.adx_strong_trend = config.get('adx_strong_trend', 25)
        self.adx_weak_trend = config.get('adx_weak_trend', 20)
        
        # Moving average parameters
        self.ma_fast = config.get('ma_fast', 10)
        self.ma_medium = config.get('ma_medium', 20)
        self.ma_slow = config.get('ma_slow', 50)
        
        # Trend detection parameters
        self.min_trend_bars = config.get('min_trend_bars', 5)
        self.trend_angle_threshold = config.get('trend_angle_threshold', 15)  # degrees
        
        # Support/Resistance parameters
        self.sr_lookback = config.get('sr_lookback', 50)
        self.sr_touch_threshold = config.get('sr_touch_threshold', 0.02)  # 2%
        
        # Advanced features
        self.enable_pattern_detection = config.get('enable_pattern_detection', True)
        self.enable_multi_timeframe = config.get('enable_multi_timeframe', True)
        
        # History tracking
        self.trend_history = {
            'adx': [],
            'trend_direction': [],
            'trend_strength': [],
            'patterns': []
        }
        
        logger.info(f"TrendStrength initialized: adx_period={self.adx_period}")
    
    def analyze_trend_strength(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze comprehensive trend strength
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Dict with trend analysis results
        """
        try:
            results = {
                'adx_analysis': {},
                'ma_analysis': {},
                'trend_metrics': {},
                'support_resistance': {},
                'price_patterns': [],
                'signals': {},
                'trend_regime': None
            }
            
            # Calculate ADX
            results['adx_analysis'] = self._calculate_adx(price_data)
            
            # Moving average analysis
            results['ma_analysis'] = self._analyze_moving_averages(price_data)
            
            # Calculate trend metrics
            results['trend_metrics'] = self._calculate_trend_metrics(price_data)
            
            # Detect support/resistance
            results['support_resistance'] = self._detect_support_resistance(price_data)
            
            # Detect price patterns
            if self.enable_pattern_detection:
                results['price_patterns'] = self._detect_price_patterns(price_data)
            
            # Generate signals
            results['signals'] = self._generate_trend_signals(results)
            
            # Classify trend regime
            results['trend_regime'] = self._classify_trend_regime(results)
            
            # Update history
            self._update_trend_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing trend strength: {e}")
            return self._get_default_results()
    
    def _calculate_adx(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Average Directional Index"""
        try:
            if len(data) < self.adx_period + 1:
                return {'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0, 'status': 'insufficient_data'}
            
            # Ensure we have required columns
            required_cols = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                return {'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0, 'status': 'missing_columns'}
            
            # Calculate True Range
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate directional movements
            up_move = data['high'] - data['high'].shift()
            down_move = data['low'].shift() - data['low']
            
            plus_dm = pd.Series(0, index=data.index)
            minus_dm = pd.Series(0, index=data.index)
            
            plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
            minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
            
            # Calculate smoothed values
            atr = self._wilder_smoothing(tr, self.adx_period)
            plus_di = 100 * self._wilder_smoothing(plus_dm, self.adx_period) / atr
            minus_di = 100 * self._wilder_smoothing(minus_dm, self.adx_period) / atr
            
            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = self._wilder_smoothing(dx, self.adx_period)
            
            # Get latest values
            latest_adx = adx.iloc[-1]
            latest_plus_di = plus_di.iloc[-1]
            latest_minus_di = minus_di.iloc[-1]
            
            return {
                'adx': float(latest_adx),
                'plus_di': float(latest_plus_di),
                'minus_di': float(latest_minus_di),
                'trend_strength': self._classify_adx_strength(latest_adx),
                'trend_direction': 'bullish' if latest_plus_di > latest_minus_di else 'bearish',
                'di_spread': float(abs(latest_plus_di - latest_minus_di)),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return {'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0, 'status': 'error'}
    
    def _wilder_smoothing(self, series: pd.Series, period: int) -> pd.Series:
        """Apply Wilder's smoothing (used in ADX calculation)"""
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    def _analyze_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze multiple moving averages"""
        try:
            price_col = 'close' if 'close' in data.columns else 'price'
            
            if price_col not in data.columns:
                return {'status': 'no_price_data'}
            
            price_series = data[price_col]
            
            # Calculate MAs
            ma_fast = price_series.rolling(window=self.ma_fast).mean()
            ma_medium = price_series.rolling(window=self.ma_medium).mean()
            ma_slow = price_series.rolling(window=self.ma_slow).mean()
            
            # Get latest values
            current_price = price_series.iloc[-1]
            latest_fast = ma_fast.iloc[-1] if len(ma_fast) >= self.ma_fast else current_price
            latest_medium = ma_medium.iloc[-1] if len(ma_medium) >= self.ma_medium else current_price
            latest_slow = ma_slow.iloc[-1] if len(ma_slow) >= self.ma_slow else current_price
            
            # Analyze MA alignment
            ma_alignment = self._analyze_ma_alignment(latest_fast, latest_medium, latest_slow)
            
            # Calculate MA slopes
            ma_slopes = {
                'fast': self._calculate_ma_slope(ma_fast),
                'medium': self._calculate_ma_slope(ma_medium),
                'slow': self._calculate_ma_slope(ma_slow)
            }
            
            return {
                'values': {
                    'fast': float(latest_fast),
                    'medium': float(latest_medium),
                    'slow': float(latest_slow),
                    'current_price': float(current_price)
                },
                'alignment': ma_alignment,
                'slopes': ma_slopes,
                'price_position': self._analyze_price_position(current_price, latest_fast, latest_medium, latest_slow),
                'golden_cross': self._check_golden_cross(ma_fast, ma_slow),
                'death_cross': self._check_death_cross(ma_fast, ma_slow),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing moving averages: {e}")
            return {'status': 'error'}
    
    def _calculate_trend_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various trend metrics"""
        try:
            price_col = 'close' if 'close' in data.columns else 'price'
            price_series = data[price_col]
            
            if len(price_series) < self.min_trend_bars:
                return {'status': 'insufficient_data'}
            
            # Calculate trend angle
            trend_angle = self._calculate_trend_angle(price_series)
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(price_series)
            
            # Calculate trend duration
            trend_duration = self._calculate_trend_duration(price_series)
            
            # Calculate volatility-adjusted trend strength
            volatility = price_series.pct_change().std()
            trend_return = (price_series.iloc[-1] - price_series.iloc[-20]) / price_series.iloc[-20] if len(price_series) >= 20 else 0
            sharpe_like_ratio = trend_return / volatility if volatility > 0 else 0
            
            return {
                'trend_angle': float(trend_angle),
                'trend_consistency': float(trend_consistency),
                'trend_duration': trend_duration,
                'volatility_adjusted_strength': float(sharpe_like_ratio),
                'current_trend': self._identify_current_trend(price_series),
                'trend_quality': self._assess_trend_quality(trend_angle, trend_consistency, sharpe_like_ratio),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend metrics: {e}")
            return {'status': 'error'}
    
    def _detect_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels"""
        try:
            price_col = 'close' if 'close' in data.columns else 'price'
            
            if len(data) < self.sr_lookback:
                return {'status': 'insufficient_data'}
            
            recent_data = data.tail(self.sr_lookback)
            highs = recent_data['high'] if 'high' in recent_data.columns else recent_data[price_col]
            lows = recent_data['low'] if 'low' in recent_data.columns else recent_data[price_col]
            
            # Find local extremes
            resistance_levels = self._find_resistance_levels(highs)
            support_levels = self._find_support_levels(lows)
            
            # Current price position
            current_price = data[price_col].iloc[-1]
            
            # Find nearest levels
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else None
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else None
            
            return {
                'resistance_levels': resistance_levels[:3],  # Top 3
                'support_levels': support_levels[:3],  # Top 3
                'nearest_resistance': float(nearest_resistance) if nearest_resistance else None,
                'nearest_support': float(nearest_support) if nearest_support else None,
                'price_to_resistance': float((nearest_resistance - current_price) / current_price) if nearest_resistance else None,
                'price_to_support': float((current_price - nearest_support) / current_price) if nearest_support else None,
                'level_strength': self._calculate_level_strength(data, support_levels, resistance_levels),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return {'status': 'error'}
    
    def _detect_price_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect common price patterns"""
        try:
            patterns = []
            
            # Higher highs and higher lows (uptrend)
            hh_hl = self._detect_hh_hl(data)
            if hh_hl:
                patterns.append(hh_hl)
            
            # Lower highs and lower lows (downtrend)
            lh_ll = self._detect_lh_ll(data)
            if lh_ll:
                patterns.append(lh_ll)
            
            # Double top/bottom
            double_patterns = self._detect_double_patterns(data)
            patterns.extend(double_patterns)
            
            # Flag/Pennant patterns
            consolidation = self._detect_consolidation_patterns(data)
            if consolidation:
                patterns.append(consolidation)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting price patterns: {e}")
            return []
    
    def _generate_trend_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from trend analysis"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'trend_quality': 'poor',
                'confirmation_factors': [],
                'warnings': []
            }
            
            # ADX-based signals
            if results['adx_analysis'].get('status') == 'calculated':
                adx = results['adx_analysis']['adx']
                direction = results['adx_analysis']['trend_direction']
                
                if adx > self.adx_strong_trend:
                    signals['trend_quality'] = 'strong'
                    if direction == 'bullish':
                        signals['primary_signal'] = 'strong_uptrend'
                        signals['signal_strength'] = 0.8
                    else:
                        signals['primary_signal'] = 'strong_downtrend'
                        signals['signal_strength'] = -0.8
                elif adx > self.adx_weak_trend:
                    signals['trend_quality'] = 'moderate'
                    if direction == 'bullish':
                        signals['primary_signal'] = 'uptrend'
                        signals['signal_strength'] = 0.5
                    else:
                        signals['primary_signal'] = 'downtrend'
                        signals['signal_strength'] = -0.5
                else:
                    signals['trend_quality'] = 'weak'
                    signals['primary_signal'] = 'no_trend'
            
            # MA confirmation
            if results['ma_analysis'].get('status') == 'calculated':
                alignment = results['ma_analysis']['alignment']
                if alignment == 'bullish_aligned':
                    signals['confirmation_factors'].append('ma_bullish_alignment')
                elif alignment == 'bearish_aligned':
                    signals['confirmation_factors'].append('ma_bearish_alignment')
                
                # Golden/Death cross
                if results['ma_analysis'].get('golden_cross'):
                    signals['confirmation_factors'].append('golden_cross')
                if results['ma_analysis'].get('death_cross'):
                    signals['confirmation_factors'].append('death_cross')
            
            # Support/Resistance considerations
            if results['support_resistance'].get('status') == 'calculated':
                if results['support_resistance'].get('price_to_resistance'):
                    if results['support_resistance']['price_to_resistance'] < 0.02:
                        signals['warnings'].append('near_resistance')
                if results['support_resistance'].get('price_to_support'):
                    if results['support_resistance']['price_to_support'] < 0.02:
                        signals['warnings'].append('near_support')
            
            # Pattern-based adjustments
            for pattern in results['price_patterns']:
                if pattern['type'] == 'higher_highs_higher_lows':
                    signals['confirmation_factors'].append('hh_hl_pattern')
                elif pattern['type'] == 'lower_highs_lower_lows':
                    signals['confirmation_factors'].append('lh_ll_pattern')
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trend signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _classify_trend_regime(self, results: Dict[str, Any]) -> str:
        """Classify overall trend regime"""
        try:
            # Get key metrics
            adx = results['adx_analysis'].get('adx', 0)
            trend_quality = results['signals'].get('trend_quality', 'poor')
            primary_signal = results['signals'].get('primary_signal', 'neutral')
            
            # Strong trending regimes
            if adx > self.adx_strong_trend:
                if 'uptrend' in primary_signal:
                    return 'strong_bullish_trend'
                elif 'downtrend' in primary_signal:
                    return 'strong_bearish_trend'
            
            # Moderate trending regimes
            elif adx > self.adx_weak_trend:
                if 'uptrend' in primary_signal:
                    return 'moderate_uptrend'
                elif 'downtrend' in primary_signal:
                    return 'moderate_downtrend'
            
            # Weak/No trend regimes
            else:
                # Check for consolidation patterns
                if any(p['type'] == 'consolidation' for p in results['price_patterns']):
                    return 'consolidation_regime'
                
                # Check MA alignment
                if results['ma_analysis'].get('alignment') == 'choppy':
                    return 'choppy_market'
                
                return 'trendless_regime'
                
        except Exception as e:
            logger.error(f"Error classifying trend regime: {e}")
            return 'undefined'
    
    def _classify_adx_strength(self, adx: float) -> str:
        """Classify ADX strength"""
        if adx < 20:
            return 'no_trend'
        elif adx < 25:
            return 'weak_trend'
        elif adx < 50:
            return 'strong_trend'
        elif adx < 75:
            return 'very_strong_trend'
        else:
            return 'extremely_strong_trend'
    
    def _analyze_ma_alignment(self, fast: float, medium: float, slow: float) -> str:
        """Analyze moving average alignment"""
        if fast > medium > slow:
            return 'bullish_aligned'
        elif fast < medium < slow:
            return 'bearish_aligned'
        else:
            return 'mixed'
    
    def _calculate_ma_slope(self, ma_series: pd.Series) -> float:
        """Calculate moving average slope"""
        try:
            if len(ma_series) < 5:
                return 0.0
            
            recent_ma = ma_series.tail(5).dropna()
            if len(recent_ma) < 3:
                return 0.0
            
            # Calculate angle in degrees
            x = range(len(recent_ma))
            slope = np.polyfit(x, recent_ma.values, 1)[0]
            angle = np.degrees(np.arctan(slope))
            
            return float(angle)
            
        except:
            return 0.0
    
    def _analyze_price_position(self, price: float, fast: float, medium: float, slow: float) -> str:
        """Analyze price position relative to MAs"""
        if price > fast > medium > slow:
            return 'above_all_mas'
        elif price < fast < medium < slow:
            return 'below_all_mas'
        elif fast < price < slow:
            return 'between_mas'
        else:
            return 'mixed_position'
    
    def _check_golden_cross(self, fast_ma: pd.Series, slow_ma: pd.Series) -> bool:
        """Check for golden cross (fast MA crossing above slow MA)"""
        try:
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return False
            
            # Check if fast crossed above slow
            prev_diff = fast_ma.iloc[-2] - slow_ma.iloc[-2]
            curr_diff = fast_ma.iloc[-1] - slow_ma.iloc[-1]
            
            return prev_diff < 0 and curr_diff > 0
            
        except:
            return False
    
    def _check_death_cross(self, fast_ma: pd.Series, slow_ma: pd.Series) -> bool:
        """Check for death cross (fast MA crossing below slow MA)"""
        try:
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return False
            
            # Check if fast crossed below slow
            prev_diff = fast_ma.iloc[-2] - slow_ma.iloc[-2]
            curr_diff = fast_ma.iloc[-1] - slow_ma.iloc[-1]
            
            return prev_diff > 0 and curr_diff < 0
            
        except:
            return False
    
    def _calculate_trend_angle(self, price_series: pd.Series) -> float:
        """Calculate trend angle in degrees"""
        try:
            if len(price_series) < 20:
                return 0.0
            
            # Use linear regression on recent prices
            recent_prices = price_series.tail(20)
            x = range(len(recent_prices))
            
            # Normalize prices to percentage change
            normalized_prices = (recent_prices.values / recent_prices.iloc[0] - 1) * 100
            
            # Calculate slope
            slope = np.polyfit(x, normalized_prices, 1)[0]
            
            # Convert to angle
            angle = np.degrees(np.arctan(slope))
            
            return float(angle)
            
        except:
            return 0.0
    
    def _calculate_trend_consistency(self, price_series: pd.Series) -> float:
        """Calculate how consistent the trend is (0-1)"""
        try:
            if len(price_series) < 10:
                return 0.0
            
            # Calculate daily returns
            returns = price_series.pct_change().dropna()
            
            # Count positive vs negative days
            positive_days = (returns > 0).sum()
            total_days = len(returns)
            
            # Calculate consistency (deviation from 50/50)
            consistency = abs(positive_days / total_days - 0.5) * 2
            
            return float(consistency)
            
        except:
            return 0.0
    
    def _calculate_trend_duration(self, price_series: pd.Series) -> int:
        """Calculate current trend duration in bars"""
        try:
            if len(price_series) < 2:
                return 0
            
            # Determine current trend direction
            current_trend = 1 if price_series.iloc[-1] > price_series.iloc[-2] else -1
            
            # Count consecutive bars in same direction
            duration = 1
            for i in range(len(price_series) - 2, 0, -1):
                bar_trend = 1 if price_series.iloc[i] > price_series.iloc[i-1] else -1
                if bar_trend == current_trend:
                    duration += 1
                else:
                    break
            
            return duration
            
        except:
            return 0
    
    def _identify_current_trend(self, price_series: pd.Series) -> str:
        """Identify current trend direction"""
        try:
            if len(price_series) < 20:
                return 'insufficient_data'
            
            # Compare current price to various lookbacks
            current = price_series.iloc[-1]
            short_term = price_series.iloc[-5]
            medium_term = price_series.iloc[-20]
            
            if current > short_term and current > medium_term:
                return 'uptrend'
            elif current < short_term and current < medium_term:
                return 'downtrend'
            else:
                return 'sideways'
                
        except:
            return 'undefined'
    
    def _assess_trend_quality(self, angle: float, consistency: float, sharpe: float) -> str:
        """Assess overall trend quality"""
        score = 0
        
        # Angle contribution
        if abs(angle) > 30:
            score += 3
        elif abs(angle) > 15:
            score += 2
        elif abs(angle) > 5:
            score += 1
        
        # Consistency contribution
        if consistency > 0.7:
            score += 3
        elif consistency > 0.5:
            score += 2
        elif consistency > 0.3:
            score += 1
        
        # Sharpe-like ratio contribution
        if abs(sharpe) > 2:
            score += 3
        elif abs(sharpe) > 1:
            score += 2
        elif abs(sharpe) > 0.5:
            score += 1
        
        # Classify quality
        if score >= 7:
            return 'excellent'
        elif score >= 5:
            return 'good'
        elif score >= 3:
            return 'moderate'
        else:
            return 'poor'
    
    def _find_resistance_levels(self, highs: pd.Series) -> List[float]:
        """Find resistance levels from price highs"""
        try:
            levels = []
            
            # Find local maxima
            for i in range(5, len(highs) - 5):
                if highs.iloc[i] == highs.iloc[i-5:i+6].max():
                    level = highs.iloc[i]
                    
                    # Check if this level was tested multiple times
                    touches = sum(1 for h in highs if abs(h - level) / level < self.sr_touch_threshold)
                    
                    if touches >= 2:
                        levels.append(float(level))
            
            # Remove duplicates and sort
            levels = sorted(list(set(levels)), reverse=True)
            
            return levels
            
        except:
            return []
    
    def _find_support_levels(self, lows: pd.Series) -> List[float]:
        """Find support levels from price lows"""
        try:
            levels = []
            
            # Find local minima
            for i in range(5, len(lows) - 5):
                if lows.iloc[i] == lows.iloc[i-5:i+6].min():
                    level = lows.iloc[i]
                    
                    # Check if this level was tested multiple times
                    touches = sum(1 for l in lows if abs(l - level) / level < self.sr_touch_threshold)
                    
                    if touches >= 2:
                        levels.append(float(level))
            
            # Remove duplicates and sort
            levels = sorted(list(set(levels)))
            
            return levels
            
        except:
            return []
    
    def _calculate_level_strength(self, data: pd.DataFrame, support: List[float], resistance: List[float]) -> Dict[str, Any]:
        """Calculate strength of support/resistance levels"""
        try:
            price_col = 'close' if 'close' in data.columns else 'price'
            prices = data[price_col]
            
            level_strength = {
                'strongest_support': None,
                'strongest_resistance': None,
                'support_touches': {},
                'resistance_touches': {}
            }
            
            # Count touches for each level
            for s_level in support:
                touches = sum(1 for p in prices if abs(p - s_level) / s_level < self.sr_touch_threshold)
                level_strength['support_touches'][s_level] = touches
            
            for r_level in resistance:
                touches = sum(1 for p in prices if abs(p - r_level) / r_level < self.sr_touch_threshold)
                level_strength['resistance_touches'][r_level] = touches
            
            # Find strongest levels
            if level_strength['support_touches']:
                level_strength['strongest_support'] = max(level_strength['support_touches'], key=level_strength['support_touches'].get)
            
            if level_strength['resistance_touches']:
                level_strength['strongest_resistance'] = max(level_strength['resistance_touches'], key=level_strength['resistance_touches'].get)
            
            return level_strength
            
        except:
            return {}
    
    def _detect_hh_hl(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect higher highs and higher lows pattern"""
        try:
            if 'high' in data.columns and 'low' in data.columns:
                highs = data['high'].tail(20)
                lows = data['low'].tail(20)
            else:
                return None
            
            # Find recent peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(2, len(highs) - 2):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    peaks.append((i, highs.iloc[i]))
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    troughs.append((i, lows.iloc[i]))
            
            # Check for HH/HL pattern
            if len(peaks) >= 2 and len(troughs) >= 2:
                if peaks[-1][1] > peaks[-2][1] and troughs[-1][1] > troughs[-2][1]:
                    return {
                        'type': 'higher_highs_higher_lows',
                        'strength': 'confirmed',
                        'last_high': float(peaks[-1][1]),
                        'last_low': float(troughs[-1][1])
                    }
            
            return None
            
        except:
            return None
    
    def _detect_lh_ll(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect lower highs and lower lows pattern"""
        try:
            if 'high' in data.columns and 'low' in data.columns:
                highs = data['high'].tail(20)
                lows = data['low'].tail(20)
            else:
                return None
            
            # Find recent peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(2, len(highs) - 2):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    peaks.append((i, highs.iloc[i]))
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    troughs.append((i, lows.iloc[i]))
            
            # Check for LH/LL pattern
            if len(peaks) >= 2 and len(troughs) >= 2:
                if peaks[-1][1] < peaks[-2][1] and troughs[-1][1] < troughs[-2][1]:
                    return {
                        'type': 'lower_highs_lower_lows',
                        'strength': 'confirmed',
                        'last_high': float(peaks[-1][1]),
                        'last_low': float(troughs[-1][1])
                    }
            
            return None
            
        except:
            return None
    
    def _detect_double_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect double top/bottom patterns"""
        # Simplified implementation
        return []
    
    def _detect_consolidation_patterns(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect consolidation patterns (flags, triangles, etc.)"""
        try:
            price_col = 'close' if 'close' in data.columns else 'price'
            prices = data[price_col].tail(20)
            
            if len(prices) < 10:
                return None
            
            # Calculate price range
            price_range = prices.max() - prices.min()
            avg_price = prices.mean()
            
            # Check if range is contracting (triangle pattern)
            first_half_range = prices.iloc[:10].max() - prices.iloc[:10].min()
            second_half_range = prices.iloc[10:].max() - prices.iloc[10:].min()
            
            if second_half_range < first_half_range * 0.7:
                return {
                    'type': 'consolidation',
                    'pattern': 'triangle',
                    'range_contraction': float(1 - second_half_range / first_half_range)
                }
            
            # Check for flag pattern (tight range)
            if price_range / avg_price < 0.03:  # Less than 3% range
                return {
                    'type': 'consolidation',
                    'pattern': 'flag',
                    'tightness': float(1 - price_range / avg_price)
                }
            
            return None
            
        except:
            return None
    
    def _update_trend_history(self, results: Dict[str, Any]):
        """Update trend history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update ADX history
            if results['adx_analysis'].get('status') == 'calculated':
                self.trend_history['adx'].append({
                    'timestamp': timestamp,
                    'value': results['adx_analysis']['adx'],
                    'direction': results['adx_analysis']['trend_direction']
                })
            
            # Update trend metrics
            self.trend_history['trend_direction'].append({
                'timestamp': timestamp,
                'direction': results['signals'].get('primary_signal', 'neutral')
            })
            
            self.trend_history['trend_strength'].append({
                'timestamp': timestamp,
                'strength': results['signals'].get('signal_strength', 0.0),
                'quality': results['signals'].get('trend_quality', 'poor')
            })
            
            # Track patterns
            if results['price_patterns']:
                self.trend_history['patterns'].extend([
                    {**pattern, 'timestamp': timestamp} for pattern in results['price_patterns']
                ])
            
            # Keep only recent history
            max_history = 100
            for key in self.trend_history:
                if len(self.trend_history[key]) > max_history:
                    self.trend_history[key] = self.trend_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating trend history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'adx_analysis': {'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0, 'status': 'error'},
            'ma_analysis': {'status': 'error'},
            'trend_metrics': {'status': 'error'},
            'support_resistance': {'status': 'error'},
            'price_patterns': [],
            'signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'trend_regime': 'undefined'
        }
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Get comprehensive trend analysis summary"""
        try:
            if not self.trend_history['adx']:
                return {'status': 'no_history'}
            
            recent_adx = [h['value'] for h in self.trend_history['adx'][-20:]]
            recent_strength = [h['strength'] for h in self.trend_history['trend_strength'][-20:]]
            
            return {
                'current_adx': recent_adx[-1] if recent_adx else 0.0,
                'average_adx': np.mean(recent_adx),
                'trend_persistence': self._calculate_trend_persistence(),
                'average_strength': np.mean(recent_strength) if recent_strength else 0.0,
                'pattern_frequency': len(self.trend_history['patterns']),
                'dominant_direction': self._get_dominant_direction()
            }
            
        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            return {'status': 'error'}
    
    def _calculate_trend_persistence(self) -> float:
        """Calculate how persistent trends have been"""
        try:
            if len(self.trend_history['trend_direction']) < 10:
                return 0.0
            
            recent_directions = [h['direction'] for h in self.trend_history['trend_direction'][-20:]]
            
            # Count direction changes
            changes = 0
            for i in range(1, len(recent_directions)):
                if recent_directions[i] != recent_directions[i-1]:
                    changes += 1
            
            # Fewer changes = more persistent
            persistence = 1.0 - (changes / (len(recent_directions) - 1))
            
            return float(persistence)
            
        except:
            return 0.0
    
    def _get_dominant_direction(self) -> str:
        """Get dominant trend direction from history"""
        try:
            if not self.trend_history['trend_direction']:
                return 'neutral'
            
            recent_directions = [h['direction'] for h in self.trend_history['trend_direction'][-20:]]
            
            # Count occurrences
            uptrend_count = sum(1 for d in recent_directions if 'uptrend' in d)
            downtrend_count = sum(1 for d in recent_directions if 'downtrend' in d)
            
            if uptrend_count > downtrend_count * 1.5:
                return 'bullish_dominant'
            elif downtrend_count > uptrend_count * 1.5:
                return 'bearish_dominant'
            else:
                return 'mixed'
                
        except:
            return 'neutral'