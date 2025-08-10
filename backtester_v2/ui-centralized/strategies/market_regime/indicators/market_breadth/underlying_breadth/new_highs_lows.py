"""
New Highs Lows - New Highs/Lows Analysis for Market Breadth
==========================================================

Analyzes new highs and lows patterns for market breadth assessment.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NewHighsLows:
    """New highs/lows analyzer for market breadth assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize New Highs Lows analyzer"""
        self.hl_window = config.get('hl_window', 20)
        self.lookback_periods = config.get('lookback_periods', [20, 52, 252])  # 20d, 52d, 252d
        self.expansion_threshold = config.get('expansion_threshold', 0.1)
        
        # Historical tracking
        self.hl_history = {
            'new_highs': [],
            'new_lows': [],
            'hl_ratio': [],
            'hl_index': [],
            'expanding_issues': [],
            'contracting_issues': [],
            'timestamps': []
        }
        
        # Trend tracking
        self.trend_metrics = {
            'hl_momentum': [],
            'expansion_days': 0,
            'contraction_days': 0,
            'trend_direction': 'neutral'
        }
        
        logger.info("NewHighsLows initialized")
    
    def analyze_new_highs_lows(self, underlying_data: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze new highs and lows for market breadth
        
        Args:
            underlying_data: Current underlying price data
            historical_data: Historical price data for highs/lows calculation
            
        Returns:
            Dict with new highs/lows analysis results
        """
        try:
            if underlying_data.empty:
                return self._get_default_hl_analysis()
            
            # Calculate new highs and lows
            hl_metrics = self._calculate_hl_metrics(underlying_data, historical_data)
            
            # Analyze HL trends
            hl_trends = self._analyze_hl_trends(hl_metrics)
            
            # Detect breadth expansion/contraction
            breadth_expansion = self._detect_breadth_expansion(hl_metrics, underlying_data)
            
            # Calculate HL divergences
            hl_divergences = self._calculate_hl_divergences(hl_metrics, underlying_data)
            
            # Analyze participation patterns
            participation_patterns = self._analyze_participation_patterns(hl_metrics)
            
            # Generate HL signals
            hl_signals = self._generate_hl_signals(hl_metrics, hl_trends, breadth_expansion)
            
            # Update historical tracking
            self._update_hl_history(hl_metrics)
            
            return {
                'hl_metrics': hl_metrics,
                'hl_trends': hl_trends,
                'breadth_expansion': breadth_expansion,
                'hl_divergences': hl_divergences,
                'participation_patterns': participation_patterns,
                'hl_signals': hl_signals,
                'breadth_score': self._calculate_hl_breadth_score(hl_metrics, hl_trends)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing new highs/lows: {e}")
            return self._get_default_hl_analysis()
    
    def _calculate_hl_metrics(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate new highs and lows metrics"""
        try:
            metrics = {}
            
            # Initialize counters for different periods
            for period in self.lookback_periods:
                metrics[f'new_highs_{period}d'] = 0
                metrics[f'new_lows_{period}d'] = 0
                metrics[f'hl_ratio_{period}d'] = 0.0
            
            if historical_data is not None and not historical_data.empty:
                # Calculate new highs/lows for each period
                for period in self.lookback_periods:
                    highs, lows = self._calculate_period_hl(current_data, historical_data, period)
                    
                    metrics[f'new_highs_{period}d'] = highs
                    metrics[f'new_lows_{period}d'] = lows
                    
                    # HL Ratio
                    total_hl = highs + lows
                    if total_hl > 0:
                        metrics[f'hl_ratio_{period}d'] = float(highs / total_hl)
                    else:
                        metrics[f'hl_ratio_{period}d'] = 0.5
            else:
                # Fallback: use mock data based on current price movements
                logger.warning("No historical data available, using fallback HL calculation")
                if 'price_change' in current_data.columns:
                    strong_moves = current_data[abs(current_data['price_change']) > 0.05]
                    positive_moves = strong_moves[strong_moves['price_change'] > 0]
                    negative_moves = strong_moves[strong_moves['price_change'] < 0]
                    
                    for period in self.lookback_periods:
                        scaling_factor = period / 20.0  # Scale based on period
                        metrics[f'new_highs_{period}d'] = int(len(positive_moves) * scaling_factor)
                        metrics[f'new_lows_{period}d'] = int(len(negative_moves) * scaling_factor)
                        
                        total = metrics[f'new_highs_{period}d'] + metrics[f'new_lows_{period}d']
                        if total > 0:
                            metrics[f'hl_ratio_{period}d'] = float(metrics[f'new_highs_{period}d'] / total)
                        else:
                            metrics[f'hl_ratio_{period}d'] = 0.5
            
            # Calculate composite metrics
            total_issues = len(current_data)
            
            # Overall new highs/lows (using 20-day as primary)
            metrics['total_new_highs'] = metrics.get('new_highs_20d', 0)
            metrics['total_new_lows'] = metrics.get('new_lows_20d', 0)
            metrics['net_new_highs'] = metrics['total_new_highs'] - metrics['total_new_lows']
            
            # Percentage metrics
            if total_issues > 0:
                metrics['new_highs_percentage'] = float(metrics['total_new_highs'] / total_issues)
                metrics['new_lows_percentage'] = float(metrics['total_new_lows'] / total_issues)
            else:
                metrics['new_highs_percentage'] = 0.0
                metrics['new_lows_percentage'] = 0.0
            
            # HL Index (cumulative new highs - new lows)
            current_hl_index = self.hl_history['hl_index'][-1] if self.hl_history['hl_index'] else 0
            metrics['hl_index'] = current_hl_index + metrics['net_new_highs']
            
            # Breadth thrust indicators
            if total_issues > 0:
                metrics['highs_thrust'] = metrics['new_highs_percentage'] > 0.4
                metrics['lows_thrust'] = metrics['new_lows_percentage'] > 0.4
            else:
                metrics['highs_thrust'] = False
                metrics['lows_thrust'] = False
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating HL metrics: {e}")
            return {'total_new_highs': 0, 'total_new_lows': 0, 'net_new_highs': 0}
    
    def _calculate_period_hl(self, current_data: pd.DataFrame, historical_data: pd.DataFrame, period: int) -> Tuple[int, int]:
        """Calculate new highs and lows for a specific period"""
        try:
            new_highs = 0
            new_lows = 0
            
            if 'symbol' in current_data.columns and 'symbol' in historical_data.columns:
                # Match symbols between current and historical data
                for symbol in current_data['symbol'].unique():
                    current_price = current_data[current_data['symbol'] == symbol]['close'].iloc[0] if len(current_data[current_data['symbol'] == symbol]) > 0 else None
                    
                    if current_price is not None:
                        symbol_history = historical_data[historical_data['symbol'] == symbol]
                        
                        if len(symbol_history) >= period:
                            # Get last 'period' days of data
                            recent_history = symbol_history.tail(period)
                            
                            highest = recent_history['high'].max()
                            lowest = recent_history['low'].min()
                            
                            # Check for new high/low
                            if current_price > highest:
                                new_highs += 1
                            elif current_price < lowest:
                                new_lows += 1
            else:
                # Fallback: estimate based on price changes
                if 'price_change' in current_data.columns:
                    # Assume stronger moves are more likely to be new highs/lows
                    threshold = 0.02 * (period / 20.0)  # Scale threshold by period
                    
                    potential_highs = current_data[current_data['price_change'] > threshold]
                    potential_lows = current_data[current_data['price_change'] < -threshold]
                    
                    new_highs = len(potential_highs)
                    new_lows = len(potential_lows)
            
            return new_highs, new_lows
            
        except Exception as e:
            logger.error(f"Error calculating period HL for {period} days: {e}")
            return 0, 0
    
    def _analyze_hl_trends(self, hl_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in new highs/lows data"""
        try:
            trends = {}
            
            # HL Index trend
            if len(self.hl_history['hl_index']) >= 5:
                recent_hl_index = self.hl_history['hl_index'][-5:] + [hl_metrics['hl_index']]
                hl_slope = self._calculate_slope(recent_hl_index)
                
                trends['hl_index_slope'] = float(hl_slope)
                trends['hl_index_trend'] = 'rising' if hl_slope > 1.0 else 'falling' if hl_slope < -1.0 else 'flat'
            
            # HL Ratio trend (20-day)
            if len(self.hl_history['hl_ratio']) >= 5:
                recent_ratios = self.hl_history['hl_ratio'][-5:] + [hl_metrics.get('hl_ratio_20d', 0.5)]
                ratio_slope = self._calculate_slope(recent_ratios)
                
                trends['hl_ratio_slope'] = float(ratio_slope)
                trends['hl_ratio_trend'] = 'improving' if ratio_slope > 0.02 else 'deteriorating' if ratio_slope < -0.02 else 'stable'
            
            # Net new highs momentum
            if len(self.hl_history['new_highs']) >= 3 and len(self.hl_history['new_lows']) >= 3:
                recent_net = []
                for i in range(-3, 0):
                    if i < -len(self.hl_history['new_highs']):
                        continue
                    net = self.hl_history['new_highs'][i] - self.hl_history['new_lows'][i]
                    recent_net.append(net)
                
                current_net = hl_metrics['net_new_highs']
                recent_net.append(current_net)
                
                net_momentum = self._calculate_slope(recent_net)
                trends['net_hl_momentum'] = float(net_momentum)
                
                # Momentum classification
                if net_momentum > 2.0:
                    trends['momentum_class'] = 'strong_bullish'
                elif net_momentum > 0.5:
                    trends['momentum_class'] = 'moderate_bullish'
                elif net_momentum < -2.0:
                    trends['momentum_class'] = 'strong_bearish'
                elif net_momentum < -0.5:
                    trends['momentum_class'] = 'moderate_bearish'
                else:
                    trends['momentum_class'] = 'neutral'
            
            # Cross-period analysis
            hl_20d = hl_metrics.get('hl_ratio_20d', 0.5)
            hl_52d = hl_metrics.get('hl_ratio_52d', 0.5)
            hl_252d = hl_metrics.get('hl_ratio_252d', 0.5)
            
            trends['short_term_bias'] = 'bullish' if hl_20d > 0.6 else 'bearish' if hl_20d < 0.4 else 'neutral'
            trends['medium_term_bias'] = 'bullish' if hl_52d > 0.6 else 'bearish' if hl_52d < 0.4 else 'neutral'
            trends['long_term_bias'] = 'bullish' if hl_252d > 0.6 else 'bearish' if hl_252d < 0.4 else 'neutral'
            
            # Consistency check
            biases = [trends['short_term_bias'], trends['medium_term_bias'], trends['long_term_bias']]
            if all(bias == 'bullish' for bias in biases):
                trends['overall_consistency'] = 'strongly_bullish'
            elif all(bias == 'bearish' for bias in biases):
                trends['overall_consistency'] = 'strongly_bearish'
            elif biases.count('bullish') >= 2:
                trends['overall_consistency'] = 'moderately_bullish'
            elif biases.count('bearish') >= 2:
                trends['overall_consistency'] = 'moderately_bearish'
            else:
                trends['overall_consistency'] = 'mixed'
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing HL trends: {e}")
            return {}
    
    def _detect_breadth_expansion(self, hl_metrics: Dict[str, Any], current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect breadth expansion and contraction patterns"""
        try:
            expansion = {
                'expansion_signals': [],
                'contraction_signals': [],
                'expansion_strength': 0.0,
                'current_phase': 'neutral'
            }
            
            total_issues = len(current_data)
            new_highs_pct = hl_metrics.get('new_highs_percentage', 0.0)
            new_lows_pct = hl_metrics.get('new_lows_percentage', 0.0)
            
            # Expansion detection
            expansion_score = 0.0
            
            # High percentage of new highs
            if new_highs_pct > self.expansion_threshold:
                expansion_score += (new_highs_pct - self.expansion_threshold) / (0.5 - self.expansion_threshold) * 0.4
                expansion['expansion_signals'].append({
                    'type': 'high_new_highs_percentage',
                    'percentage': float(new_highs_pct),
                    'strength': float(new_highs_pct / 0.5)
                })
            
            # Thrust in new highs
            if hl_metrics.get('highs_thrust', False):
                expansion_score += 0.3
                expansion['expansion_signals'].append({
                    'type': 'new_highs_thrust',
                    'percentage': float(new_highs_pct),
                    'strength': 1.0
                })
            
            # Positive HL Index momentum
            hl_index_trend = self._get_hl_index_momentum()
            if hl_index_trend > 0:
                expansion_score += min(hl_index_trend / 10.0, 0.3)
                expansion['expansion_signals'].append({
                    'type': 'positive_hl_index_momentum',
                    'momentum': float(hl_index_trend),
                    'strength': min(hl_index_trend / 10.0, 1.0)
                })
            
            expansion['expansion_strength'] = min(expansion_score, 1.0)
            
            # Contraction detection
            contraction_score = 0.0
            
            # High percentage of new lows
            if new_lows_pct > self.expansion_threshold:
                contraction_score += (new_lows_pct - self.expansion_threshold) / (0.5 - self.expansion_threshold) * 0.4
                expansion['contraction_signals'].append({
                    'type': 'high_new_lows_percentage',
                    'percentage': float(new_lows_pct),
                    'strength': float(new_lows_pct / 0.5)
                })
            
            # Thrust in new lows
            if hl_metrics.get('lows_thrust', False):
                contraction_score += 0.3
                expansion['contraction_signals'].append({
                    'type': 'new_lows_thrust',
                    'percentage': float(new_lows_pct),
                    'strength': 1.0
                })
            
            # Negative HL Index momentum
            if hl_index_trend < 0:
                contraction_score += min(abs(hl_index_trend) / 10.0, 0.3)
                expansion['contraction_signals'].append({
                    'type': 'negative_hl_index_momentum',
                    'momentum': float(hl_index_trend),
                    'strength': min(abs(hl_index_trend) / 10.0, 1.0)
                })
            
            # Determine current phase
            if expansion_score > contraction_score and expansion_score > 0.5:
                expansion['current_phase'] = 'expansion'
                self.trend_metrics['expansion_days'] += 1
                self.trend_metrics['contraction_days'] = 0
            elif contraction_score > expansion_score and contraction_score > 0.5:
                expansion['current_phase'] = 'contraction'
                self.trend_metrics['contraction_days'] += 1
                self.trend_metrics['expansion_days'] = 0
            else:
                expansion['current_phase'] = 'neutral'
                self.trend_metrics['expansion_days'] = max(0, self.trend_metrics['expansion_days'] - 1)
                self.trend_metrics['contraction_days'] = max(0, self.trend_metrics['contraction_days'] - 1)
            
            # Update trend direction
            if self.trend_metrics['expansion_days'] >= 3:
                self.trend_metrics['trend_direction'] = 'expanding'
            elif self.trend_metrics['contraction_days'] >= 3:
                self.trend_metrics['trend_direction'] = 'contracting'
            else:
                self.trend_metrics['trend_direction'] = 'neutral'
            
            expansion['trend_duration'] = {
                'expansion_days': self.trend_metrics['expansion_days'],
                'contraction_days': self.trend_metrics['contraction_days'],
                'trend_direction': self.trend_metrics['trend_direction']
            }
            
            return expansion
            
        except Exception as e:
            logger.error(f"Error detecting breadth expansion: {e}")
            return {'expansion_signals': [], 'contraction_signals': [], 'current_phase': 'neutral'}
    
    def _calculate_hl_divergences(self, hl_metrics: Dict[str, Any], current_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate divergences in HL patterns"""
        try:
            divergences = {
                'divergence_signals': [],
                'divergence_count': 0,
                'divergence_severity': 0.0
            }
            
            # Price-HL divergence (if index price available)
            if 'index_price' in current_data.columns and len(self.hl_history['hl_index']) >= 5:
                recent_prices = current_data['index_price'].tail(5)
                if len(recent_prices) >= 2:
                    price_trend = 1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1
                    
                    # HL Index trend
                    recent_hl_index = self.hl_history['hl_index'][-4:] + [hl_metrics['hl_index']]
                    hl_trend = 1 if recent_hl_index[-1] > recent_hl_index[0] else -1
                    
                    # Detect divergence
                    if price_trend != hl_trend:
                        divergence_type = 'bullish_hl_divergence' if price_trend < 0 and hl_trend > 0 else 'bearish_hl_divergence'
                        
                        divergences['divergence_signals'].append({
                            'type': divergence_type,
                            'price_trend': price_trend,
                            'hl_trend': hl_trend,
                            'severity': 0.5
                        })
                        divergences['divergence_severity'] += 0.5
            
            # Cross-period HL divergence
            hl_20d = hl_metrics.get('hl_ratio_20d', 0.5)
            hl_52d = hl_metrics.get('hl_ratio_52d', 0.5)
            
            if abs(hl_20d - hl_52d) > 0.3:
                divergence_type = 'short_long_term_hl_divergence'
                severity = min(abs(hl_20d - hl_52d), 0.5)
                
                divergences['divergence_signals'].append({
                    'type': divergence_type,
                    'short_term_ratio': float(hl_20d),
                    'long_term_ratio': float(hl_52d),
                    'severity': float(severity)
                })
                divergences['divergence_severity'] += severity
            
            # HL momentum divergence
            net_hl = hl_metrics.get('net_new_highs', 0)
            if len(self.hl_history['new_highs']) >= 3:
                recent_net = []
                for i in range(-3, 0):
                    if i >= -len(self.hl_history['new_highs']):
                        net = self.hl_history['new_highs'][i] - self.hl_history['new_lows'][i]
                        recent_net.append(net)
                
                if recent_net:
                    avg_net = np.mean(recent_net)
                    
                    # Significant deviation from recent pattern
                    if abs(net_hl - avg_net) > 5:
                        divergences['divergence_signals'].append({
                            'type': 'hl_momentum_divergence',
                            'current_net': int(net_hl),
                            'average_net': float(avg_net),
                            'deviation': float(abs(net_hl - avg_net))
                        })
                        divergences['divergence_severity'] += 0.3
            
            divergences['divergence_count'] = len(divergences['divergence_signals'])
            divergences['divergence_severity'] = min(divergences['divergence_severity'], 1.0)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error calculating HL divergences: {e}")
            return {'divergence_signals': [], 'divergence_count': 0, 'divergence_severity': 0.0}
    
    def _analyze_participation_patterns(self, hl_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze participation patterns in new highs/lows"""
        try:
            participation = {}
            
            # Current participation levels
            new_highs_pct = hl_metrics.get('new_highs_percentage', 0.0)
            new_lows_pct = hl_metrics.get('new_lows_percentage', 0.0)
            total_hl_pct = new_highs_pct + new_lows_pct
            
            participation['active_participation'] = float(total_hl_pct)
            participation['directional_bias'] = 'bullish' if new_highs_pct > new_lows_pct else 'bearish' if new_lows_pct > new_highs_pct else 'neutral'
            
            # Participation quality
            if total_hl_pct > 0.3:
                participation['participation_quality'] = 'high'
            elif total_hl_pct > 0.15:
                participation['participation_quality'] = 'moderate'
            else:
                participation['participation_quality'] = 'low'
            
            # Cross-period participation analysis
            periods_data = {}
            for period in self.lookback_periods:
                highs = hl_metrics.get(f'new_highs_{period}d', 0)
                lows = hl_metrics.get(f'new_lows_{period}d', 0)
                total = highs + lows
                
                periods_data[f'{period}d'] = {
                    'total_hl': total,
                    'hl_ratio': float(highs / total) if total > 0 else 0.5
                }
            
            participation['period_analysis'] = periods_data
            
            # Participation consistency across periods
            ratios = [data['hl_ratio'] for data in periods_data.values()]
            if ratios:
                ratio_std = np.std(ratios)
                participation['consistency'] = 'high' if ratio_std < 0.1 else 'moderate' if ratio_std < 0.2 else 'low'
                participation['ratio_consistency'] = float(1.0 - min(ratio_std, 1.0))
            
            return participation
            
        except Exception as e:
            logger.error(f"Error analyzing participation patterns: {e}")
            return {}
    
    def _generate_hl_signals(self, hl_metrics: Dict[str, Any], hl_trends: Dict[str, Any], breadth_expansion: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable HL signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'hl_signals': [],
                'breadth_implications': []
            }
            
            # Expansion/Contraction signals
            current_phase = breadth_expansion.get('current_phase', 'neutral')
            expansion_strength = breadth_expansion.get('expansion_strength', 0.0)
            
            if current_phase == 'expansion' and expansion_strength > 0.6:
                signals['hl_signals'].append('strong_breadth_expansion')
                signals['breadth_implications'].append('broad_market_leadership')
                signals['signal_strength'] += expansion_strength * 0.4
                
            elif current_phase == 'contraction' and len(breadth_expansion.get('contraction_signals', [])) > 0:
                signals['hl_signals'].append('breadth_contraction_detected')
                signals['breadth_implications'].append('narrowing_market_leadership')
                signals['signal_strength'] += 0.4
            
            # Thrust signals
            if hl_metrics.get('highs_thrust', False):
                signals['hl_signals'].append('new_highs_thrust')
                signals['breadth_implications'].append('broad_bullish_momentum')
                signals['signal_strength'] += 0.3
                
            if hl_metrics.get('lows_thrust', False):
                signals['hl_signals'].append('new_lows_thrust')
                signals['breadth_implications'].append('broad_bearish_momentum')
                signals['signal_strength'] += 0.3
            
            # Trend signals
            hl_index_trend = hl_trends.get('hl_index_trend', 'flat')
            if hl_index_trend == 'rising':
                signals['hl_signals'].append('improving_hl_trend')
                signals['breadth_implications'].append('increasing_breadth_strength')
                signals['signal_strength'] += 0.2
            elif hl_index_trend == 'falling':
                signals['hl_signals'].append('deteriorating_hl_trend')
                signals['breadth_implications'].append('decreasing_breadth_strength')
                signals['signal_strength'] += 0.2
            
            # Consistency signals
            overall_consistency = hl_trends.get('overall_consistency', 'mixed')
            if overall_consistency == 'strongly_bullish':
                signals['hl_signals'].append('consistent_bullish_hl_pattern')
                signals['breadth_implications'].append('broad_time_frame_alignment')
                signals['signal_strength'] += 0.3
            elif overall_consistency == 'strongly_bearish':
                signals['hl_signals'].append('consistent_bearish_hl_pattern')
                signals['breadth_implications'].append('broad_time_frame_deterioration')
                signals['signal_strength'] += 0.3
            
            # Momentum class signals
            momentum_class = hl_trends.get('momentum_class', 'neutral')
            if momentum_class in ['strong_bullish', 'strong_bearish']:
                signals['hl_signals'].append(f'{momentum_class}_hl_momentum')
                signals['breadth_implications'].append(f'{momentum_class.split("_")[1]}_acceleration')
                signals['signal_strength'] += 0.2
            
            # Determine primary signal
            if signals['signal_strength'] > 0.7:
                if any('expansion' in sig for sig in signals['hl_signals']):
                    signals['primary_signal'] = 'strong_breadth_expansion'
                elif any('contraction' in sig for sig in signals['hl_signals']):
                    signals['primary_signal'] = 'strong_breadth_contraction'
                elif any('bullish' in sig for sig in signals['hl_signals']):
                    signals['primary_signal'] = 'strong_bullish_breadth'
                elif any('bearish' in sig for sig in signals['hl_signals']):
                    signals['primary_signal'] = 'strong_bearish_breadth'
                else:
                    signals['primary_signal'] = 'strong_hl_signal'
            elif signals['signal_strength'] > 0.4:
                if any('improving' in sig or 'bullish' in sig for sig in signals['hl_signals']):
                    signals['primary_signal'] = 'positive_hl_bias'
                elif any('deteriorating' in sig or 'bearish' in sig for sig in signals['hl_signals']):
                    signals['primary_signal'] = 'negative_hl_bias'
                else:
                    signals['primary_signal'] = 'moderate_hl_change'
            elif signals['signal_strength'] > 0.2:
                signals['primary_signal'] = 'weak_hl_signal'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating HL signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_hl_breadth_score(self, hl_metrics: Dict[str, Any], hl_trends: Dict[str, Any]) -> float:
        """Calculate overall breadth score from HL analysis"""
        try:
            score = 0.5  # Base neutral score
            
            # HL ratio contribution (30%) - using 20-day as primary
            hl_ratio = hl_metrics.get('hl_ratio_20d', 0.5)
            score += (hl_ratio - 0.5) * 0.6  # Scale to ±0.3
            
            # HL Index trend contribution (25%)
            hl_slope = hl_trends.get('hl_index_slope', 0.0)
            slope_contribution = np.tanh(hl_slope / 5.0) * 0.25  # Normalize slope
            score += slope_contribution
            
            # Net HL momentum contribution (20%)
            net_momentum = hl_trends.get('net_hl_momentum', 0.0)
            momentum_contribution = np.tanh(net_momentum / 3.0) * 0.2
            score += momentum_contribution
            
            # Consistency contribution (15%)
            consistency = hl_trends.get('overall_consistency', 'mixed')
            if consistency == 'strongly_bullish':
                score += 0.15
            elif consistency == 'strongly_bearish':
                score -= 0.15
            elif consistency == 'moderately_bullish':
                score += 0.075
            elif consistency == 'moderately_bearish':
                score -= 0.075
            
            # Participation contribution (10%)
            new_highs_pct = hl_metrics.get('new_highs_percentage', 0.0)
            new_lows_pct = hl_metrics.get('new_lows_percentage', 0.0)
            participation = new_highs_pct + new_lows_pct
            
            if participation > 0.2:  # High participation gets bonus
                score += 0.05
            elif participation < 0.05:  # Low participation gets penalty
                score -= 0.05
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating HL breadth score: {e}")
            return 0.5
    
    def _calculate_slope(self, data: List[float]) -> float:
        """Calculate slope of data series"""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            y = np.array(data)
            
            # Simple linear regression slope
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x)) if np.std(x) > 0 else 0.0
            
            return float(slope)
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            return 0.0
    
    def _get_hl_index_momentum(self) -> float:
        """Get recent HL Index momentum"""
        try:
            if len(self.hl_history['hl_index']) >= 3:
                recent_values = self.hl_history['hl_index'][-3:]
                return self._calculate_slope(recent_values)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting HL index momentum: {e}")
            return 0.0
    
    def _update_hl_history(self, hl_metrics: Dict[str, Any]):
        """Update historical HL tracking"""
        try:
            # Update basic metrics
            self.hl_history['new_highs'].append(hl_metrics.get('total_new_highs', 0))
            self.hl_history['new_lows'].append(hl_metrics.get('total_new_lows', 0))
            self.hl_history['hl_ratio'].append(hl_metrics.get('hl_ratio_20d', 0.5))
            self.hl_history['hl_index'].append(hl_metrics.get('hl_index', 0))
            
            # Trim history to window size
            for key in ['new_highs', 'new_lows', 'hl_ratio', 'hl_index']:
                if len(self.hl_history[key]) > self.hl_window * 2:
                    self.hl_history[key].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating HL history: {e}")
    
    def _get_default_hl_analysis(self) -> Dict[str, Any]:
        """Get default HL analysis when data is insufficient"""
        return {
            'hl_metrics': {'total_new_highs': 0, 'total_new_lows': 0, 'net_new_highs': 0},
            'hl_trends': {},
            'breadth_expansion': {'expansion_signals': [], 'contraction_signals': [], 'current_phase': 'neutral'},
            'hl_divergences': {'divergence_signals': [], 'divergence_count': 0},
            'participation_patterns': {},
            'hl_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_hl_summary(self) -> Dict[str, Any]:
        """Get summary of new highs/lows analysis system"""
        try:
            return {
                'history_length': len(self.hl_history['hl_index']),
                'current_hl_index': self.hl_history['hl_index'][-1] if self.hl_history['hl_index'] else 0,
                'average_hl_ratio': np.mean(self.hl_history['hl_ratio']) if self.hl_history['hl_ratio'] else 0.5,
                'trend_metrics': self.trend_metrics.copy(),
                'lookback_periods': self.lookback_periods,
                'analysis_config': {
                    'hl_window': self.hl_window,
                    'expansion_threshold': self.expansion_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting HL summary: {e}")
            return {'status': 'error', 'error': str(e)}