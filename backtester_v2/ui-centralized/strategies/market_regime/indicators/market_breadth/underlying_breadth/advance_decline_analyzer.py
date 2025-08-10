"""
Advance Decline Analyzer - Advance/Decline Analysis for Market Breadth
=====================================================================

Analyzes advance/decline patterns in underlying assets for breadth assessment.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvanceDeclineAnalyzer:
    """Advance/Decline analyzer for market breadth assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Advance Decline Analyzer"""
        self.ad_window = config.get('ad_window', 20)
        self.thrust_threshold = config.get('thrust_threshold', 0.8)
        self.decline_threshold = config.get('decline_threshold', 0.2)
        
        # Historical tracking
        self.ad_history = {
            'ad_line': [],
            'ad_ratio': [],
            'advancing': [],
            'declining': [],
            'unchanged': [],
            'timestamps': []
        }
        
        # Momentum tracking
        self.momentum_metrics = {
            'thrust_signals': [],
            'decline_signals': [],
            'divergence_signals': []
        }
        
        logger.info("AdvanceDeclineAnalyzer initialized")
    
    def analyze_advance_decline(self, underlying_data: pd.DataFrame, price_changes: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze advance/decline patterns for market breadth
        
        Args:
            underlying_data: DataFrame with underlying asset data
            price_changes: Optional DataFrame with price change data
            
        Returns:
            Dict with advance/decline analysis results
        """
        try:
            if underlying_data.empty:
                return self._get_default_ad_analysis()
            
            # Calculate advance/decline metrics
            ad_metrics = self._calculate_ad_metrics(underlying_data, price_changes)
            
            # Analyze AD line trends
            ad_trends = self._analyze_ad_trends(ad_metrics)
            
            # Detect thrust and decline signals
            thrust_decline_signals = self._detect_thrust_decline_signals(ad_metrics)
            
            # Calculate breadth divergences
            breadth_divergences = self._calculate_breadth_divergences(ad_metrics, underlying_data)
            
            # Analyze participation patterns
            participation_analysis = self._analyze_participation_patterns(ad_metrics)
            
            # Generate breadth signals
            breadth_signals = self._generate_breadth_signals(ad_metrics, ad_trends, thrust_decline_signals)
            
            # Update historical tracking
            self._update_ad_history(ad_metrics)
            
            return {
                'ad_metrics': ad_metrics,
                'ad_trends': ad_trends,
                'thrust_decline_signals': thrust_decline_signals,
                'breadth_divergences': breadth_divergences,
                'participation_analysis': participation_analysis,
                'breadth_signals': breadth_signals,
                'breadth_score': self._calculate_ad_breadth_score(ad_metrics, ad_trends)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing advance/decline: {e}")
            return self._get_default_ad_analysis()
    
    def _calculate_ad_metrics(self, underlying_data: pd.DataFrame, price_changes: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate basic advance/decline metrics"""
        try:
            metrics = {}
            
            # Use price changes if provided, otherwise calculate from current data
            if price_changes is not None and 'price_change' in price_changes.columns:
                changes = price_changes['price_change']
            elif 'price_change' in underlying_data.columns:
                changes = underlying_data['price_change']
            elif 'close' in underlying_data.columns and 'prev_close' in underlying_data.columns:
                changes = (underlying_data['close'] - underlying_data['prev_close']) / underlying_data['prev_close']
            else:
                # Fallback: use mock data
                logger.warning("No price change data available, using mock data")
                changes = pd.Series(np.random.normal(0, 0.02, len(underlying_data)))
            
            # Count advances, declines, unchanged
            advancing = len(changes[changes > 0])
            declining = len(changes[changes < 0])
            unchanged = len(changes[changes == 0])
            total_issues = len(changes)
            
            metrics['advancing'] = advancing
            metrics['declining'] = declining
            metrics['unchanged'] = unchanged
            metrics['total_issues'] = total_issues
            
            # Calculate ratios
            if total_issues > 0:
                metrics['advance_ratio'] = float(advancing / total_issues)
                metrics['decline_ratio'] = float(declining / total_issues)
                metrics['advance_decline_ratio'] = float(advancing / declining) if declining > 0 else float('inf')
            else:
                metrics['advance_ratio'] = 0.0
                metrics['decline_ratio'] = 0.0
                metrics['advance_decline_ratio'] = 1.0
            
            # Net advances
            metrics['net_advances'] = advancing - declining
            
            # Advance/Decline line (cumulative)
            current_ad_line = self.ad_history['ad_line'][-1] if self.ad_history['ad_line'] else 0
            metrics['ad_line'] = current_ad_line + metrics['net_advances']
            
            # Breadth thrust calculation
            if total_issues > 0:
                metrics['breadth_thrust'] = float(advancing / total_issues)
            else:
                metrics['breadth_thrust'] = 0.5
            
            # McClellan Oscillator components
            if len(self.ad_history['ad_ratio']) >= 19:
                # 19-day EMA of advance ratio
                recent_ratios = self.ad_history['ad_ratio'][-19:] + [metrics['advance_ratio']]
                ema_19 = self._calculate_ema(recent_ratios, 19)
                metrics['ema_19'] = float(ema_19)
                
                if len(self.ad_history['ad_ratio']) >= 39:
                    # 39-day EMA of advance ratio
                    longer_ratios = self.ad_history['ad_ratio'][-39:] + [metrics['advance_ratio']]
                    ema_39 = self._calculate_ema(longer_ratios, 39)
                    metrics['ema_39'] = float(ema_39)
                    metrics['mcclellan_oscillator'] = float(ema_19 - ema_39)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating AD metrics: {e}")
            return {'advancing': 0, 'declining': 0, 'advance_ratio': 0.5}
    
    def _analyze_ad_trends(self, ad_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in advance/decline data"""
        try:
            trends = {}
            
            # AD Line trend
            if len(self.ad_history['ad_line']) >= 5:
                recent_ad = self.ad_history['ad_line'][-5:] + [ad_metrics['ad_line']]
                ad_slope = self._calculate_slope(recent_ad)
                
                trends['ad_line_slope'] = float(ad_slope)
                trends['ad_line_trend'] = 'rising' if ad_slope > 0.1 else 'falling' if ad_slope < -0.1 else 'flat'
            
            # Advance ratio trend
            if len(self.ad_history['ad_ratio']) >= 5:
                recent_ratios = self.ad_history['ad_ratio'][-5:] + [ad_metrics['advance_ratio']]
                ratio_slope = self._calculate_slope(recent_ratios)
                
                trends['advance_ratio_slope'] = float(ratio_slope)
                trends['advance_ratio_trend'] = 'improving' if ratio_slope > 0.01 else 'deteriorating' if ratio_slope < -0.01 else 'stable'
            
            # McClellan Oscillator trend
            if 'mcclellan_oscillator' in ad_metrics and len(self.ad_history['ad_ratio']) >= 10:
                # Simple trend based on current value
                mcc_osc = ad_metrics['mcclellan_oscillator']
                
                if mcc_osc > 0.05:
                    trends['mcclellan_trend'] = 'bullish'
                elif mcc_osc < -0.05:
                    trends['mcclellan_trend'] = 'bearish'
                else:
                    trends['mcclellan_trend'] = 'neutral'
                
                trends['mcclellan_value'] = float(mcc_osc)
            
            # Net advances momentum
            if len(self.ad_history['advancing']) >= 3 and len(self.ad_history['declining']) >= 3:
                recent_net = []
                for i in range(-3, 0):
                    net = self.ad_history['advancing'][i] - self.ad_history['declining'][i]
                    recent_net.append(net)
                
                current_net = ad_metrics['net_advances']
                recent_net.append(current_net)
                
                net_momentum = self._calculate_slope(recent_net)
                trends['net_advances_momentum'] = float(net_momentum)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing AD trends: {e}")
            return {}
    
    def _detect_thrust_decline_signals(self, ad_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect breadth thrust and decline signals"""
        try:
            signals = {
                'thrust_signals': [],
                'decline_signals': [],
                'signal_strength': 0.0
            }
            
            advance_ratio = ad_metrics.get('advance_ratio', 0.5)
            
            # Breadth thrust detection
            if advance_ratio >= self.thrust_threshold:
                thrust_strength = (advance_ratio - self.thrust_threshold) / (1.0 - self.thrust_threshold)
                signals['thrust_signals'].append({
                    'type': 'breadth_thrust',
                    'strength': float(thrust_strength),
                    'advance_ratio': float(advance_ratio)
                })
                signals['signal_strength'] += thrust_strength * 0.5
                
                if advance_ratio >= 0.9:
                    signals['thrust_signals'].append({
                        'type': 'extreme_breadth_thrust',
                        'strength': 1.0,
                        'advance_ratio': float(advance_ratio)
                    })
                    signals['signal_strength'] += 0.3
            
            # Breadth decline detection
            if advance_ratio <= self.decline_threshold:
                decline_strength = (self.decline_threshold - advance_ratio) / self.decline_threshold
                signals['decline_signals'].append({
                    'type': 'breadth_decline',
                    'strength': float(decline_strength),
                    'advance_ratio': float(advance_ratio)
                })
                signals['signal_strength'] += decline_strength * 0.5
                
                if advance_ratio <= 0.1:
                    signals['decline_signals'].append({
                        'type': 'extreme_breadth_decline',
                        'strength': 1.0,
                        'advance_ratio': float(advance_ratio)
                    })
                    signals['signal_strength'] += 0.3
            
            # McClellan extreme signals
            if 'mcclellan_oscillator' in ad_metrics:
                mcc_osc = ad_metrics['mcclellan_oscillator']
                
                if mcc_osc > 0.1:
                    signals['thrust_signals'].append({
                        'type': 'mcclellan_thrust',
                        'strength': min(mcc_osc / 0.2, 1.0),
                        'oscillator_value': float(mcc_osc)
                    })
                elif mcc_osc < -0.1:
                    signals['decline_signals'].append({
                        'type': 'mcclellan_decline',
                        'strength': min(abs(mcc_osc) / 0.2, 1.0),
                        'oscillator_value': float(mcc_osc)
                    })
            
            signals['signal_strength'] = min(signals['signal_strength'], 1.0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting thrust/decline signals: {e}")
            return {'thrust_signals': [], 'decline_signals': [], 'signal_strength': 0.0}
    
    def _calculate_breadth_divergences(self, ad_metrics: Dict[str, Any], underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate divergences between breadth and price"""
        try:
            divergences = {
                'divergence_signals': [],
                'divergence_count': 0,
                'divergence_severity': 0.0
            }
            
            # Price-breadth divergence (if price index available)
            if 'index_price' in underlying_data.columns and len(self.ad_history['ad_line']) >= 5:
                # Calculate price trend
                recent_prices = underlying_data['index_price'].tail(5)
                if len(recent_prices) >= 2:
                    price_trend = 1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1
                    
                    # Calculate AD line trend
                    recent_ad_line = self.ad_history['ad_line'][-4:] + [ad_metrics['ad_line']]
                    ad_trend = 1 if recent_ad_line[-1] > recent_ad_line[0] else -1
                    
                    # Detect divergence
                    if price_trend != ad_trend:
                        divergence_type = 'bullish_divergence' if price_trend < 0 and ad_trend > 0 else 'bearish_divergence'
                        
                        divergences['divergence_signals'].append({
                            'type': divergence_type,
                            'price_trend': price_trend,
                            'breadth_trend': ad_trend,
                            'severity': 0.5
                        })
                        divergences['divergence_severity'] += 0.5
            
            # Volume-breadth divergence
            if 'volume' in underlying_data.columns:
                current_volume = underlying_data['volume'].sum()
                advance_ratio = ad_metrics.get('advance_ratio', 0.5)
                
                # High volume with low breadth (distribution)
                if current_volume > underlying_data['volume'].mean() * 1.5 and advance_ratio < 0.4:
                    divergences['divergence_signals'].append({
                        'type': 'volume_breadth_divergence',
                        'description': 'high_volume_low_breadth',
                        'severity': 0.4
                    })
                    divergences['divergence_severity'] += 0.4
            
            # McClellan divergence with advance ratio
            if 'mcclellan_oscillator' in ad_metrics:
                mcc_osc = ad_metrics['mcclellan_oscillator']
                advance_ratio = ad_metrics.get('advance_ratio', 0.5)
                
                # McClellan positive but low advance ratio (or vice versa)
                if (mcc_osc > 0.05 and advance_ratio < 0.4) or (mcc_osc < -0.05 and advance_ratio > 0.6):
                    divergences['divergence_signals'].append({
                        'type': 'mcclellan_breadth_divergence',
                        'mcclellan_value': float(mcc_osc),
                        'advance_ratio': float(advance_ratio),
                        'severity': 0.3
                    })
                    divergences['divergence_severity'] += 0.3
            
            divergences['divergence_count'] = len(divergences['divergence_signals'])
            divergences['divergence_severity'] = min(divergences['divergence_severity'], 1.0)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error calculating breadth divergences: {e}")
            return {'divergence_signals': [], 'divergence_count': 0, 'divergence_severity': 0.0}
    
    def _analyze_participation_patterns(self, ad_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market participation patterns"""
        try:
            participation = {}
            
            # Current participation
            total_issues = ad_metrics.get('total_issues', 0)
            advancing = ad_metrics.get('advancing', 0)
            declining = ad_metrics.get('declining', 0)
            unchanged = ad_metrics.get('unchanged', 0)
            
            if total_issues > 0:
                participation['active_participation'] = float((advancing + declining) / total_issues)
                participation['passive_participation'] = float(unchanged / total_issues)
            else:
                participation['active_participation'] = 0.0
                participation['passive_participation'] = 0.0
            
            # Historical participation trend
            if len(self.ad_history['advancing']) >= 5:
                recent_active = []
                for i in range(-5, 0):
                    if i < -len(self.ad_history['advancing']):
                        continue
                    total = (self.ad_history['advancing'][i] + 
                            self.ad_history['declining'][i] + 
                            self.ad_history['unchanged'][i])
                    if total > 0:
                        active = (self.ad_history['advancing'][i] + self.ad_history['declining'][i]) / total
                        recent_active.append(active)
                
                if recent_active:
                    current_active = participation['active_participation']
                    avg_active = np.mean(recent_active)
                    
                    participation['participation_trend'] = 'increasing' if current_active > avg_active * 1.1 else 'decreasing' if current_active < avg_active * 0.9 else 'stable'
                    participation['participation_momentum'] = float((current_active - avg_active) / avg_active) if avg_active > 0 else 0.0
            
            # Participation quality
            advance_ratio = ad_metrics.get('advance_ratio', 0.5)
            if advance_ratio > 0.6:
                participation['participation_quality'] = 'positive'
            elif advance_ratio < 0.4:
                participation['participation_quality'] = 'negative'
            else:
                participation['participation_quality'] = 'mixed'
            
            return participation
            
        except Exception as e:
            logger.error(f"Error analyzing participation patterns: {e}")
            return {}
    
    def _generate_breadth_signals(self, ad_metrics: Dict[str, Any], ad_trends: Dict[str, Any], thrust_decline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable breadth signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'breadth_signals': [],
                'market_implications': []
            }
            
            # Thrust/Decline signals
            if thrust_decline['thrust_signals']:
                signals['breadth_signals'].append('breadth_thrust_detected')
                signals['market_implications'].append('broad_market_strength')
                signals['signal_strength'] += thrust_decline['signal_strength'] * 0.4
                
                if any(sig['type'] == 'extreme_breadth_thrust' for sig in thrust_decline['thrust_signals']):
                    signals['breadth_signals'].append('extreme_breadth_thrust')
                    signals['market_implications'].append('overwhelming_bullish_participation')
                    signals['signal_strength'] += 0.3
            
            if thrust_decline['decline_signals']:
                signals['breadth_signals'].append('breadth_decline_detected')
                signals['market_implications'].append('broad_market_weakness')
                signals['signal_strength'] += thrust_decline['signal_strength'] * 0.4
                
                if any(sig['type'] == 'extreme_breadth_decline' for sig in thrust_decline['decline_signals']):
                    signals['breadth_signals'].append('extreme_breadth_decline')
                    signals['market_implications'].append('overwhelming_bearish_participation')
                    signals['signal_strength'] += 0.3
            
            # Trend signals
            ad_line_trend = ad_trends.get('ad_line_trend', 'flat')
            if ad_line_trend == 'rising':
                signals['breadth_signals'].append('improving_breadth_trend')
                signals['market_implications'].append('increasing_market_participation')
                signals['signal_strength'] += 0.2
            elif ad_line_trend == 'falling':
                signals['breadth_signals'].append('deteriorating_breadth_trend')
                signals['market_implications'].append('decreasing_market_participation')
                signals['signal_strength'] += 0.2
            
            # McClellan signals
            mcclellan_trend = ad_trends.get('mcclellan_trend', 'neutral')
            if mcclellan_trend == 'bullish':
                signals['breadth_signals'].append('positive_mcclellan_momentum')
                signals['market_implications'].append('bullish_breadth_momentum')
                signals['signal_strength'] += 0.2
            elif mcclellan_trend == 'bearish':
                signals['breadth_signals'].append('negative_mcclellan_momentum')
                signals['market_implications'].append('bearish_breadth_momentum')
                signals['signal_strength'] += 0.2
            
            # Determine primary signal
            if signals['signal_strength'] > 0.6:
                if any('thrust' in sig for sig in signals['breadth_signals']):
                    signals['primary_signal'] = 'strong_bullish_breadth'
                elif any('decline' in sig for sig in signals['breadth_signals']):
                    signals['primary_signal'] = 'strong_bearish_breadth'
                elif any('improving' in sig for sig in signals['breadth_signals']):
                    signals['primary_signal'] = 'improving_breadth'
                elif any('deteriorating' in sig for sig in signals['breadth_signals']):
                    signals['primary_signal'] = 'deteriorating_breadth'
                else:
                    signals['primary_signal'] = 'mixed_breadth_signals'
            elif signals['signal_strength'] > 0.3:
                signals['primary_signal'] = 'moderate_breadth_change'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating breadth signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_ad_breadth_score(self, ad_metrics: Dict[str, Any], ad_trends: Dict[str, Any]) -> float:
        """Calculate overall breadth score from advance/decline analysis"""
        try:
            score = 0.5  # Base neutral score
            
            # Advance ratio contribution (40%)
            advance_ratio = ad_metrics.get('advance_ratio', 0.5)
            score += (advance_ratio - 0.5) * 0.8  # Scale to ±0.4
            
            # AD line trend contribution (25%)
            ad_slope = ad_trends.get('ad_line_slope', 0.0)
            slope_contribution = np.tanh(ad_slope / 10.0) * 0.25  # Normalize slope
            score += slope_contribution
            
            # McClellan oscillator contribution (20%)
            if 'mcclellan_oscillator' in ad_metrics:
                mcc_osc = ad_metrics['mcclellan_oscillator']
                mcc_contribution = np.tanh(mcc_osc / 0.2) * 0.2  # Normalize oscillator
                score += mcc_contribution
            
            # Net advances momentum contribution (15%)
            net_momentum = ad_trends.get('net_advances_momentum', 0.0)
            momentum_contribution = np.tanh(net_momentum / 5.0) * 0.15
            score += momentum_contribution
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating AD breadth score: {e}")
            return 0.5
    
    def _calculate_slope(self, data: List[float]) -> float:
        """Calculate slope of data series"""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            y = np.array(data)
            
            # Simple linear regression
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x)) if np.std(x) > 0 else 0.0
            
            return float(slope)
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            return 0.0
    
    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate exponential moving average"""
        try:
            if len(data) < period:
                return np.mean(data) if data else 0.0
            
            alpha = 2.0 / (period + 1)
            ema = data[0]
            
            for value in data[1:]:
                ema = alpha * value + (1 - alpha) * ema
            
            return float(ema)
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return 0.0
    
    def _update_ad_history(self, ad_metrics: Dict[str, Any]):
        """Update historical advance/decline tracking"""
        try:
            # Update basic metrics
            self.ad_history['ad_line'].append(ad_metrics.get('ad_line', 0))
            self.ad_history['ad_ratio'].append(ad_metrics.get('advance_ratio', 0.5))
            self.ad_history['advancing'].append(ad_metrics.get('advancing', 0))
            self.ad_history['declining'].append(ad_metrics.get('declining', 0))
            self.ad_history['unchanged'].append(ad_metrics.get('unchanged', 0))
            
            # Trim history to window size
            for key in ['ad_line', 'ad_ratio', 'advancing', 'declining', 'unchanged']:
                if len(self.ad_history[key]) > self.ad_window * 2:
                    self.ad_history[key].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating AD history: {e}")
    
    def _get_default_ad_analysis(self) -> Dict[str, Any]:
        """Get default advance/decline analysis when data is insufficient"""
        return {
            'ad_metrics': {'advancing': 0, 'declining': 0, 'advance_ratio': 0.5},
            'ad_trends': {},
            'thrust_decline_signals': {'thrust_signals': [], 'decline_signals': []},
            'breadth_divergences': {'divergence_signals': [], 'divergence_count': 0},
            'participation_analysis': {},
            'breadth_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_ad_summary(self) -> Dict[str, Any]:
        """Get summary of advance/decline analysis system"""
        try:
            return {
                'history_length': len(self.ad_history['ad_line']),
                'current_ad_line': self.ad_history['ad_line'][-1] if self.ad_history['ad_line'] else 0,
                'average_advance_ratio': np.mean(self.ad_history['ad_ratio']) if self.ad_history['ad_ratio'] else 0.5,
                'thrust_signals_count': len(self.momentum_metrics['thrust_signals']),
                'decline_signals_count': len(self.momentum_metrics['decline_signals']),
                'analysis_config': {
                    'ad_window': self.ad_window,
                    'thrust_threshold': self.thrust_threshold,
                    'decline_threshold': self.decline_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting AD summary: {e}")
            return {'status': 'error', 'error': str(e)}