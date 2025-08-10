"""
Volume Flow Indicator - Volume Flow Analysis for Market Breadth
==============================================================

Analyzes volume flow patterns in underlying assets for breadth assessment.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VolumeFlowIndicator:
    """Volume flow indicator for market breadth assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Volume Flow Indicator"""
        self.flow_window = config.get('flow_window', 20)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.accumulation_threshold = config.get('accumulation_threshold', 0.6)
        
        # Volume flow tracking
        self.volume_history = {
            'total_volume': [],
            'up_volume': [],
            'down_volume': [],
            'neutral_volume': [],
            'volume_ratio': [],
            'timestamps': []
        }
        
        # Flow momentum tracking
        self.flow_momentum = {
            'accumulation_days': 0,
            'distribution_days': 0,
            'flow_direction': 'neutral',
            'momentum_strength': 0.0
        }
        
        logger.info("VolumeFlowIndicator initialized")
    
    def analyze_volume_flow(self, underlying_data: pd.DataFrame, price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze volume flow patterns for market breadth
        
        Args:
            underlying_data: DataFrame with underlying volume data
            price_data: Optional price data for flow direction calculation
            
        Returns:
            Dict with volume flow analysis results
        """
        try:
            if underlying_data.empty:
                return self._get_default_flow_analysis()
            
            # Calculate volume flow metrics
            flow_metrics = self._calculate_flow_metrics(underlying_data, price_data)
            
            # Analyze flow patterns
            flow_patterns = self._analyze_flow_patterns(flow_metrics)
            
            # Detect accumulation/distribution
            accumulation_distribution = self._detect_accumulation_distribution(flow_metrics)
            
            # Calculate flow divergences
            flow_divergences = self._calculate_flow_divergences(flow_metrics, price_data)
            
            # Analyze volume momentum
            volume_momentum = self._analyze_volume_momentum(flow_metrics)
            
            # Generate flow signals
            flow_signals = self._generate_flow_signals(flow_metrics, flow_patterns, accumulation_distribution)
            
            # Update historical tracking
            self._update_volume_history(flow_metrics)
            
            return {
                'flow_metrics': flow_metrics,
                'flow_patterns': flow_patterns,
                'accumulation_distribution': accumulation_distribution,
                'flow_divergences': flow_divergences,
                'volume_momentum': volume_momentum,
                'flow_signals': flow_signals,
                'breadth_score': self._calculate_volume_breadth_score(flow_metrics, flow_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume flow: {e}")
            return self._get_default_flow_analysis()
    
    def _calculate_flow_metrics(self, underlying_data: pd.DataFrame, price_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate basic volume flow metrics"""
        try:
            metrics = {}
            
            # Total volume
            total_volume = underlying_data['volume'].sum()
            metrics['total_volume'] = float(total_volume)
            
            # Calculate up/down volume based on price changes
            if price_data is not None and 'price_change' in price_data.columns:
                # Merge volume with price changes
                if 'symbol' in underlying_data.columns and 'symbol' in price_data.columns:
                    merged = underlying_data.merge(price_data[['symbol', 'price_change']], on='symbol', how='left')
                else:
                    # Assume same order
                    merged = underlying_data.copy()
                    merged['price_change'] = price_data['price_change'].iloc[:len(underlying_data)]
                
                up_volume = merged[merged['price_change'] > 0]['volume'].sum()
                down_volume = merged[merged['price_change'] < 0]['volume'].sum()
                neutral_volume = merged[merged['price_change'] == 0]['volume'].sum()
                
            elif 'price_change' in underlying_data.columns:
                up_volume = underlying_data[underlying_data['price_change'] > 0]['volume'].sum()
                down_volume = underlying_data[underlying_data['price_change'] < 0]['volume'].sum()
                neutral_volume = underlying_data[underlying_data['price_change'] == 0]['volume'].sum()
                
            else:
                # Fallback: assume random distribution
                up_volume = total_volume * 0.4
                down_volume = total_volume * 0.4
                neutral_volume = total_volume * 0.2
            
            metrics['up_volume'] = float(up_volume)
            metrics['down_volume'] = float(down_volume)
            metrics['neutral_volume'] = float(neutral_volume)
            
            # Volume ratios
            if total_volume > 0:
                metrics['up_volume_ratio'] = float(up_volume / total_volume)
                metrics['down_volume_ratio'] = float(down_volume / total_volume)
                metrics['neutral_volume_ratio'] = float(neutral_volume / total_volume)
            else:
                metrics['up_volume_ratio'] = 0.0
                metrics['down_volume_ratio'] = 0.0
                metrics['neutral_volume_ratio'] = 0.0
            
            # Up/Down volume ratio
            metrics['up_down_volume_ratio'] = float(up_volume / down_volume) if down_volume > 0 else float('inf')
            
            # Net volume flow
            metrics['net_volume_flow'] = float(up_volume - down_volume)
            
            # Volume flow percentage
            if total_volume > 0:
                metrics['volume_flow_percentage'] = float(metrics['net_volume_flow'] / total_volume)
            else:
                metrics['volume_flow_percentage'] = 0.0
            
            # Participation metrics
            if 'symbol' in underlying_data.columns:
                total_symbols = len(underlying_data)
                active_symbols = len(underlying_data[underlying_data['volume'] > 0])
                
                metrics['participation_rate'] = float(active_symbols / total_symbols) if total_symbols > 0 else 0.0
                
                # High volume symbols
                avg_volume = underlying_data['volume'].mean()
                high_volume_symbols = len(underlying_data[underlying_data['volume'] > avg_volume * 2])
                metrics['high_volume_participation'] = float(high_volume_symbols / total_symbols) if total_symbols > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating flow metrics: {e}")
            return {'total_volume': 0.0, 'up_volume_ratio': 0.5}
    
    def _analyze_flow_patterns(self, flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume flow patterns"""
        try:
            patterns = {}
            
            # Current flow direction
            up_ratio = flow_metrics.get('up_volume_ratio', 0.5)
            down_ratio = flow_metrics.get('down_volume_ratio', 0.5)
            
            if up_ratio > 0.6:
                patterns['current_flow'] = 'bullish'
                patterns['flow_strength'] = float(up_ratio)
            elif down_ratio > 0.6:
                patterns['current_flow'] = 'bearish'
                patterns['flow_strength'] = float(down_ratio)
            else:
                patterns['current_flow'] = 'neutral'
                patterns['flow_strength'] = float(max(up_ratio, down_ratio))
            
            # Flow consistency
            if len(self.volume_history['up_volume']) >= 5:
                recent_up_ratios = []
                recent_down_ratios = []
                
                for i in range(-5, 0):
                    if i < -len(self.volume_history['up_volume']):
                        continue
                    
                    total = (self.volume_history['up_volume'][i] + 
                            self.volume_history['down_volume'][i] + 
                            self.volume_history['neutral_volume'][i])
                    
                    if total > 0:
                        recent_up_ratios.append(self.volume_history['up_volume'][i] / total)
                        recent_down_ratios.append(self.volume_history['down_volume'][i] / total)
                
                if recent_up_ratios and recent_down_ratios:
                    up_consistency = 1.0 - np.std(recent_up_ratios)
                    down_consistency = 1.0 - np.std(recent_down_ratios)
                    
                    patterns['flow_consistency'] = float(max(up_consistency, down_consistency))
                    
                    # Trend strength
                    avg_up_ratio = np.mean(recent_up_ratios)
                    avg_down_ratio = np.mean(recent_down_ratios)
                    
                    if avg_up_ratio > 0.55:
                        patterns['trend_strength'] = 'strong_bullish'
                    elif avg_down_ratio > 0.55:
                        patterns['trend_strength'] = 'strong_bearish'
                    elif avg_up_ratio > 0.52 or avg_down_ratio > 0.52:
                        patterns['trend_strength'] = 'moderate'
                    else:
                        patterns['trend_strength'] = 'weak'
            
            # Volume surge detection
            if len(self.volume_history['total_volume']) >= 3:
                recent_volumes = self.volume_history['total_volume'][-3:]
                current_volume = flow_metrics.get('total_volume', 0)
                avg_recent = np.mean(recent_volumes) if recent_volumes else 0
                
                if avg_recent > 0:
                    volume_ratio = current_volume / avg_recent
                    patterns['volume_surge'] = volume_ratio > self.volume_threshold
                    patterns['volume_ratio'] = float(volume_ratio)
                else:
                    patterns['volume_surge'] = False
                    patterns['volume_ratio'] = 1.0
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing flow patterns: {e}")
            return {}
    
    def _detect_accumulation_distribution(self, flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect accumulation and distribution patterns"""
        try:
            acc_dist = {
                'current_phase': 'neutral',
                'phase_strength': 0.0,
                'accumulation_score': 0.0,
                'distribution_score': 0.0
            }
            
            up_ratio = flow_metrics.get('up_volume_ratio', 0.5)
            volume_flow_pct = flow_metrics.get('volume_flow_percentage', 0.0)
            participation = flow_metrics.get('participation_rate', 0.5)
            
            # Accumulation detection
            accumulation_score = 0.0
            
            # High up volume ratio
            if up_ratio > self.accumulation_threshold:
                accumulation_score += (up_ratio - self.accumulation_threshold) / (1.0 - self.accumulation_threshold) * 0.4
            
            # Positive volume flow
            if volume_flow_pct > 0:
                accumulation_score += min(volume_flow_pct * 5, 0.3)  # Cap at 0.3
            
            # High participation
            if participation > 0.7:
                accumulation_score += (participation - 0.7) / 0.3 * 0.3
            
            acc_dist['accumulation_score'] = float(min(accumulation_score, 1.0))
            
            # Distribution detection
            down_ratio = flow_metrics.get('down_volume_ratio', 0.5)
            distribution_score = 0.0
            
            # High down volume ratio
            if down_ratio > self.accumulation_threshold:
                distribution_score += (down_ratio - self.accumulation_threshold) / (1.0 - self.accumulation_threshold) * 0.4
            
            # Negative volume flow
            if volume_flow_pct < 0:
                distribution_score += min(abs(volume_flow_pct) * 5, 0.3)  # Cap at 0.3
            
            # High participation in decline
            if participation > 0.7 and down_ratio > 0.5:
                distribution_score += (participation - 0.7) / 0.3 * 0.3
            
            acc_dist['distribution_score'] = float(min(distribution_score, 1.0))
            
            # Determine current phase
            if accumulation_score > distribution_score and accumulation_score > 0.5:
                acc_dist['current_phase'] = 'accumulation'
                acc_dist['phase_strength'] = accumulation_score
                
                # Update momentum tracking
                self.flow_momentum['accumulation_days'] += 1
                self.flow_momentum['distribution_days'] = 0
                
            elif distribution_score > accumulation_score and distribution_score > 0.5:
                acc_dist['current_phase'] = 'distribution'
                acc_dist['phase_strength'] = distribution_score
                
                # Update momentum tracking
                self.flow_momentum['distribution_days'] += 1
                self.flow_momentum['accumulation_days'] = 0
                
            else:
                # Reset counters for neutral phase
                if acc_dist['current_phase'] == 'neutral':
                    self.flow_momentum['accumulation_days'] = max(0, self.flow_momentum['accumulation_days'] - 1)
                    self.flow_momentum['distribution_days'] = max(0, self.flow_momentum['distribution_days'] - 1)
            
            # Update flow direction
            if self.flow_momentum['accumulation_days'] >= 3:
                self.flow_momentum['flow_direction'] = 'accumulation'
                self.flow_momentum['momentum_strength'] = min(self.flow_momentum['accumulation_days'] / 10.0, 1.0)
            elif self.flow_momentum['distribution_days'] >= 3:
                self.flow_momentum['flow_direction'] = 'distribution'
                self.flow_momentum['momentum_strength'] = min(self.flow_momentum['distribution_days'] / 10.0, 1.0)
            else:
                self.flow_momentum['flow_direction'] = 'neutral'
                self.flow_momentum['momentum_strength'] = 0.0
            
            return acc_dist
            
        except Exception as e:
            logger.error(f"Error detecting accumulation/distribution: {e}")
            return {'current_phase': 'neutral', 'phase_strength': 0.0}
    
    def _calculate_flow_divergences(self, flow_metrics: Dict[str, Any], price_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate divergences in volume flow"""
        try:
            divergences = {
                'divergence_signals': [],
                'divergence_count': 0,
                'divergence_severity': 0.0
            }
            
            # Price-volume divergence
            if price_data is not None and 'index_return' in price_data.columns:
                index_return = price_data['index_return'].iloc[0] if len(price_data) > 0 else 0
                volume_flow_pct = flow_metrics.get('volume_flow_percentage', 0.0)
                
                # Positive price, negative volume flow (bearish divergence)
                if index_return > 0.02 and volume_flow_pct < -0.1:
                    divergences['divergence_signals'].append({
                        'type': 'bearish_price_volume_divergence',
                        'price_return': float(index_return),
                        'volume_flow': float(volume_flow_pct),
                        'severity': min(abs(volume_flow_pct), 0.5)
                    })
                    divergences['divergence_severity'] += 0.4
                
                # Negative price, positive volume flow (bullish divergence)
                elif index_return < -0.02 and volume_flow_pct > 0.1:
                    divergences['divergence_signals'].append({
                        'type': 'bullish_price_volume_divergence',
                        'price_return': float(index_return),
                        'volume_flow': float(volume_flow_pct),
                        'severity': min(volume_flow_pct, 0.5)
                    })
                    divergences['divergence_severity'] += 0.4
            
            # Participation-volume divergence
            participation = flow_metrics.get('participation_rate', 0.5)
            total_volume = flow_metrics.get('total_volume', 0)
            
            if len(self.volume_history['total_volume']) >= 3:
                avg_volume = np.mean(self.volume_history['total_volume'][-3:])
                volume_ratio = total_volume / avg_volume if avg_volume > 0 else 1.0
                
                # High volume but low participation (institutional activity)
                if volume_ratio > 1.5 and participation < 0.4:
                    divergences['divergence_signals'].append({
                        'type': 'volume_participation_divergence',
                        'volume_ratio': float(volume_ratio),
                        'participation': float(participation),
                        'description': 'high_volume_low_participation'
                    })
                    divergences['divergence_severity'] += 0.3
            
            # Flow direction divergence with historical pattern
            if len(self.volume_history['volume_ratio']) >= 5:
                recent_ratios = self.volume_history['volume_ratio'][-5:]
                current_up_ratio = flow_metrics.get('up_volume_ratio', 0.5)
                avg_up_ratio = np.mean(recent_ratios) if recent_ratios else 0.5
                
                # Significant divergence from recent pattern
                if abs(current_up_ratio - avg_up_ratio) > 0.3:
                    divergence_type = 'sudden_bullish_shift' if current_up_ratio > avg_up_ratio else 'sudden_bearish_shift'
                    
                    divergences['divergence_signals'].append({
                        'type': divergence_type,
                        'current_ratio': float(current_up_ratio),
                        'average_ratio': float(avg_up_ratio),
                        'deviation': float(abs(current_up_ratio - avg_up_ratio))
                    })
                    divergences['divergence_severity'] += 0.2
            
            divergences['divergence_count'] = len(divergences['divergence_signals'])
            divergences['divergence_severity'] = min(divergences['divergence_severity'], 1.0)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error calculating flow divergences: {e}")
            return {'divergence_signals': [], 'divergence_count': 0, 'divergence_severity': 0.0}
    
    def _analyze_volume_momentum(self, flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume momentum patterns"""
        try:
            momentum = {
                'momentum_direction': self.flow_momentum['flow_direction'],
                'momentum_strength': self.flow_momentum['momentum_strength'],
                'momentum_duration': {
                    'accumulation_days': self.flow_momentum['accumulation_days'],
                    'distribution_days': self.flow_momentum['distribution_days']
                },
                'volume_acceleration': 0.0
            }
            
            # Calculate volume acceleration
            if len(self.volume_history['total_volume']) >= 3:
                recent_volumes = self.volume_history['total_volume'][-3:]
                current_volume = flow_metrics.get('total_volume', 0)
                
                if len(recent_volumes) >= 2:
                    recent_avg = np.mean(recent_volumes)
                    previous_avg = np.mean(recent_volumes[:-1])
                    
                    if previous_avg > 0:
                        acceleration = (current_volume - recent_avg) / previous_avg
                        momentum['volume_acceleration'] = float(acceleration)
            
            # Momentum quality assessment
            up_ratio = flow_metrics.get('up_volume_ratio', 0.5)
            participation = flow_metrics.get('participation_rate', 0.5)
            
            if momentum['momentum_direction'] == 'accumulation':
                # Quality factors for accumulation
                quality_score = 0.0
                
                if up_ratio > 0.6:
                    quality_score += 0.3
                if participation > 0.7:
                    quality_score += 0.3
                if momentum['volume_acceleration'] > 0:
                    quality_score += 0.2
                if momentum['momentum_duration']['accumulation_days'] >= 5:
                    quality_score += 0.2
                
                momentum['momentum_quality'] = 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.4 else 'low'
                
            elif momentum['momentum_direction'] == 'distribution':
                # Quality factors for distribution
                down_ratio = flow_metrics.get('down_volume_ratio', 0.5)
                quality_score = 0.0
                
                if down_ratio > 0.6:
                    quality_score += 0.3
                if participation > 0.7:
                    quality_score += 0.3
                if momentum['volume_acceleration'] > 0:  # High volume in distribution
                    quality_score += 0.2
                if momentum['momentum_duration']['distribution_days'] >= 5:
                    quality_score += 0.2
                
                momentum['momentum_quality'] = 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.4 else 'low'
            else:
                momentum['momentum_quality'] = 'neutral'
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error analyzing volume momentum: {e}")
            return {'momentum_direction': 'neutral', 'momentum_strength': 0.0}
    
    def _generate_flow_signals(self, flow_metrics: Dict[str, Any], flow_patterns: Dict[str, Any], accumulation_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable volume flow signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'flow_signals': [],
                'breadth_implications': []
            }
            
            # Accumulation/Distribution signals
            current_phase = accumulation_distribution.get('current_phase', 'neutral')
            phase_strength = accumulation_distribution.get('phase_strength', 0.0)
            
            if current_phase == 'accumulation' and phase_strength > 0.6:
                signals['flow_signals'].append('strong_accumulation_detected')
                signals['breadth_implications'].append('broad_institutional_buying')
                signals['signal_strength'] += phase_strength * 0.4
                
            elif current_phase == 'distribution' and phase_strength > 0.6:
                signals['flow_signals'].append('strong_distribution_detected')
                signals['breadth_implications'].append('broad_institutional_selling')
                signals['signal_strength'] += phase_strength * 0.4
            
            # Flow pattern signals
            current_flow = flow_patterns.get('current_flow', 'neutral')
            flow_strength = flow_patterns.get('flow_strength', 0.0)
            
            if current_flow == 'bullish' and flow_strength > 0.7:
                signals['flow_signals'].append('strong_bullish_flow')
                signals['breadth_implications'].append('broad_upside_participation')
                signals['signal_strength'] += 0.3
                
            elif current_flow == 'bearish' and flow_strength > 0.7:
                signals['flow_signals'].append('strong_bearish_flow')
                signals['breadth_implications'].append('broad_downside_participation')
                signals['signal_strength'] += 0.3
            
            # Volume surge signals
            if flow_patterns.get('volume_surge', False):
                volume_ratio = flow_patterns.get('volume_ratio', 1.0)
                
                if volume_ratio > 2.0:
                    signals['flow_signals'].append('extreme_volume_surge')
                    signals['breadth_implications'].append('exceptional_market_activity')
                    signals['signal_strength'] += 0.3
                else:
                    signals['flow_signals'].append('volume_surge_detected')
                    signals['breadth_implications'].append('increased_market_activity')
                    signals['signal_strength'] += 0.2
            
            # Momentum signals
            momentum_direction = self.flow_momentum['flow_direction']
            momentum_strength = self.flow_momentum['momentum_strength']
            
            if momentum_direction != 'neutral' and momentum_strength > 0.5:
                signals['flow_signals'].append(f'sustained_{momentum_direction}_momentum')
                signals['breadth_implications'].append(f'persistent_{momentum_direction}_pattern')
                signals['signal_strength'] += momentum_strength * 0.2
            
            # Participation signals
            participation = flow_metrics.get('participation_rate', 0.5)
            if participation > 0.8:
                signals['flow_signals'].append('broad_market_participation')
                signals['breadth_implications'].append('wide_market_engagement')
                signals['signal_strength'] += 0.2
            elif participation < 0.3:
                signals['flow_signals'].append('narrow_market_participation')
                signals['breadth_implications'].append('limited_market_engagement')
                signals['signal_strength'] += 0.2
            
            # Determine primary signal
            if signals['signal_strength'] > 0.7:
                if any('accumulation' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'strong_accumulation_phase'
                elif any('distribution' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'strong_distribution_phase'
                elif any('bullish' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'strong_bullish_breadth'
                elif any('bearish' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'strong_bearish_breadth'
                else:
                    signals['primary_signal'] = 'strong_flow_signal'
            elif signals['signal_strength'] > 0.4:
                if any('accumulation' in sig or 'bullish' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'positive_flow_bias'
                elif any('distribution' in sig or 'bearish' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'negative_flow_bias'
                else:
                    signals['primary_signal'] = 'moderate_flow_change'
            elif signals['signal_strength'] > 0.2:
                signals['primary_signal'] = 'weak_flow_signal'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating flow signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_volume_breadth_score(self, flow_metrics: Dict[str, Any], flow_patterns: Dict[str, Any]) -> float:
        """Calculate overall breadth score from volume flow analysis"""
        try:
            score = 0.5  # Base neutral score
            
            # Up/Down volume ratio contribution (30%)
            up_ratio = flow_metrics.get('up_volume_ratio', 0.5)
            score += (up_ratio - 0.5) * 0.6  # Scale to ±0.3
            
            # Participation rate contribution (25%)
            participation = flow_metrics.get('participation_rate', 0.5)
            score += (participation - 0.5) * 0.5  # Scale to ±0.25
            
            # Flow consistency contribution (20%)
            flow_consistency = flow_patterns.get('flow_consistency', 0.5)
            if flow_consistency is not None:
                score += (flow_consistency - 0.5) * 0.4  # Scale to ±0.2
            
            # Volume momentum contribution (15%)
            momentum_strength = self.flow_momentum['momentum_strength']
            momentum_direction = self.flow_momentum['flow_direction']
            
            if momentum_direction == 'accumulation':
                score += momentum_strength * 0.15
            elif momentum_direction == 'distribution':
                score -= momentum_strength * 0.15
            
            # Volume surge contribution (10%)
            if flow_patterns.get('volume_surge', False):
                volume_ratio = flow_patterns.get('volume_ratio', 1.0)
                surge_contribution = min((volume_ratio - 1.0) / 2.0, 0.1)  # Cap at 0.1
                score += surge_contribution
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating volume breadth score: {e}")
            return 0.5
    
    def _update_volume_history(self, flow_metrics: Dict[str, Any]):
        """Update historical volume tracking"""
        try:
            # Update basic volume metrics
            self.volume_history['total_volume'].append(flow_metrics.get('total_volume', 0.0))
            self.volume_history['up_volume'].append(flow_metrics.get('up_volume', 0.0))
            self.volume_history['down_volume'].append(flow_metrics.get('down_volume', 0.0))
            self.volume_history['neutral_volume'].append(flow_metrics.get('neutral_volume', 0.0))
            self.volume_history['volume_ratio'].append(flow_metrics.get('up_volume_ratio', 0.5))
            
            # Trim history to window size
            for key in ['total_volume', 'up_volume', 'down_volume', 'neutral_volume', 'volume_ratio']:
                if len(self.volume_history[key]) > self.flow_window * 2:
                    self.volume_history[key].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating volume history: {e}")
    
    def _get_default_flow_analysis(self) -> Dict[str, Any]:
        """Get default volume flow analysis when data is insufficient"""
        return {
            'flow_metrics': {'total_volume': 0.0, 'up_volume_ratio': 0.5},
            'flow_patterns': {},
            'accumulation_distribution': {'current_phase': 'neutral', 'phase_strength': 0.0},
            'flow_divergences': {'divergence_signals': [], 'divergence_count': 0},
            'volume_momentum': {'momentum_direction': 'neutral', 'momentum_strength': 0.0},
            'flow_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_volume_flow_summary(self) -> Dict[str, Any]:
        """Get summary of volume flow analysis system"""
        try:
            return {
                'history_length': len(self.volume_history['total_volume']),
                'average_up_volume_ratio': np.mean(self.volume_history['volume_ratio']) if self.volume_history['volume_ratio'] else 0.5,
                'current_momentum': {
                    'direction': self.flow_momentum['flow_direction'],
                    'strength': self.flow_momentum['momentum_strength'],
                    'accumulation_days': self.flow_momentum['accumulation_days'],
                    'distribution_days': self.flow_momentum['distribution_days']
                },
                'analysis_config': {
                    'flow_window': self.flow_window,
                    'volume_threshold': self.volume_threshold,
                    'accumulation_threshold': self.accumulation_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting volume flow summary: {e}")
            return {'status': 'error', 'error': str(e)}