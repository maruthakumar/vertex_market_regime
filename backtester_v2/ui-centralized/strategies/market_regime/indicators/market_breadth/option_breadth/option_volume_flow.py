"""
Option Volume Flow - Option Volume Flow Analysis
===============================================

Analyzes volume flow patterns in option markets for breadth analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class OptionVolumeFlow:
    """Option volume flow analyzer for market breadth"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Option Volume Flow analyzer"""
        self.flow_threshold = config.get('flow_threshold', 1.5)
        self.volume_percentiles = config.get('volume_percentiles', [25, 50, 75, 90])
        self.flow_window = config.get('flow_window', 20)
        
        # Flow tracking
        self.volume_history = []
        self.flow_signals = []
        
        logger.info("OptionVolumeFlow initialized")
    
    def analyze_volume_flow(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze option volume flow patterns
        
        Args:
            option_data: DataFrame with option volume data
            
        Returns:
            Dict with volume flow analysis
        """
        try:
            if option_data.empty:
                return self._get_default_flow_analysis()
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(option_data)
            
            # Analyze flow patterns
            flow_patterns = self._analyze_flow_patterns(option_data)
            
            # Calculate flow divergences
            flow_divergences = self._detect_flow_divergences(option_data)
            
            # Generate flow signals
            flow_signals = self._generate_flow_signals(volume_metrics, flow_patterns)
            
            return {
                'volume_metrics': volume_metrics,
                'flow_patterns': flow_patterns,
                'flow_divergences': flow_divergences,
                'flow_signals': flow_signals,
                'breadth_score': self._calculate_breadth_score(volume_metrics, flow_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option volume flow: {e}")
            return self._get_default_flow_analysis()
    
    def _calculate_volume_metrics(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive volume metrics"""
        try:
            metrics = {}
            
            # Basic volume statistics
            total_volume = option_data['volume'].sum()
            avg_volume = option_data['volume'].mean()
            
            metrics['total_volume'] = float(total_volume)
            metrics['average_volume'] = float(avg_volume)
            
            # Call/Put volume breakdown
            if 'option_type' in option_data.columns:
                call_volume = option_data[option_data['option_type'] == 'CE']['volume'].sum()
                put_volume = option_data[option_data['option_type'] == 'PE']['volume'].sum()
                
                metrics['call_volume'] = float(call_volume)
                metrics['put_volume'] = float(put_volume)
                metrics['call_put_volume_ratio'] = float(call_volume / put_volume) if put_volume > 0 else 0.0
            
            # Volume percentiles
            volume_percentiles = np.percentile(option_data['volume'], self.volume_percentiles)
            metrics['volume_percentiles'] = {
                f'p{p}': float(vol) for p, vol in zip(self.volume_percentiles, volume_percentiles)
            }
            
            # High volume concentration
            high_volume_threshold = metrics['volume_percentiles']['p90']
            high_volume_count = len(option_data[option_data['volume'] > high_volume_threshold])
            metrics['high_volume_concentration'] = float(high_volume_count / len(option_data))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volume metrics: {e}")
            return {'total_volume': 0.0, 'average_volume': 0.0}
    
    def _analyze_flow_patterns(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume flow patterns"""
        try:
            patterns = {}
            
            # ITM/OTM flow analysis
            if 'moneyness' in option_data.columns:
                itm_flow = option_data[option_data['moneyness'] < 1.0]['volume'].sum()
                otm_flow = option_data[option_data['moneyness'] > 1.0]['volume'].sum()
                
                patterns['itm_flow'] = float(itm_flow)
                patterns['otm_flow'] = float(otm_flow)
                patterns['itm_otm_flow_ratio'] = float(itm_flow / otm_flow) if otm_flow > 0 else 0.0
            
            # Strike level flow distribution
            if 'strike' in option_data.columns:
                strike_flows = option_data.groupby('strike')['volume'].sum().sort_values(ascending=False)
                patterns['dominant_strikes'] = strike_flows.head(5).to_dict()
                patterns['flow_concentration'] = float(strike_flows.iloc[:5].sum() / strike_flows.sum())
            
            # Time-based flow patterns
            if 'dte' in option_data.columns:
                dte_flows = option_data.groupby('dte')['volume'].sum()
                patterns['near_term_flow'] = float(dte_flows[dte_flows.index <= 30].sum())
                patterns['far_term_flow'] = float(dte_flows[dte_flows.index > 30].sum())
                patterns['term_flow_ratio'] = float(patterns['near_term_flow'] / patterns['far_term_flow']) if patterns['far_term_flow'] > 0 else 0.0
            
            # Flow velocity
            self.volume_history.append(option_data['volume'].sum())
            if len(self.volume_history) > self.flow_window:
                self.volume_history.pop(0)
            
            if len(self.volume_history) >= 2:
                recent_avg = np.mean(self.volume_history[-5:]) if len(self.volume_history) >= 5 else self.volume_history[-1]
                historical_avg = np.mean(self.volume_history[:-5]) if len(self.volume_history) > 5 else np.mean(self.volume_history[:-1])
                patterns['flow_velocity'] = float(recent_avg / historical_avg) if historical_avg > 0 else 1.0
            else:
                patterns['flow_velocity'] = 1.0
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing flow patterns: {e}")
            return {'flow_velocity': 1.0}
    
    def _detect_flow_divergences(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect flow divergences that may indicate breadth issues"""
        try:
            divergences = {
                'divergence_signals': [],
                'divergence_count': 0,
                'divergence_severity': 0.0
            }
            
            # Volume vs Price divergence (if price data available)
            if 'price' in option_data.columns and len(self.volume_history) >= 5:
                price_trend = 1 if option_data['price'].iloc[-1] > option_data['price'].iloc[0] else -1
                volume_trend = 1 if self.volume_history[-1] > np.mean(self.volume_history[:-1]) else -1
                
                if price_trend != volume_trend:
                    divergences['divergence_signals'].append('volume_price_divergence')
                    divergences['divergence_severity'] += 0.3
            
            # Call/Put flow divergence
            if 'option_type' in option_data.columns:
                call_vol = option_data[option_data['option_type'] == 'CE']['volume'].sum()
                put_vol = option_data[option_data['option_type'] == 'PE']['volume'].sum()
                cp_ratio = call_vol / put_vol if put_vol > 0 else float('inf')
                
                if cp_ratio > 3.0 or cp_ratio < 0.33:
                    divergences['divergence_signals'].append('extreme_call_put_divergence')
                    divergences['divergence_severity'] += 0.4
            
            # Strike concentration divergence
            if 'strike' in option_data.columns:
                strike_volumes = option_data.groupby('strike')['volume'].sum()
                top_5_concentration = strike_volumes.nlargest(5).sum() / strike_volumes.sum()
                
                if top_5_concentration > 0.8:
                    divergences['divergence_signals'].append('extreme_strike_concentration')
                    divergences['divergence_severity'] += 0.3
            
            divergences['divergence_count'] = len(divergences['divergence_signals'])
            divergences['divergence_severity'] = min(divergences['divergence_severity'], 1.0)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting flow divergences: {e}")
            return {'divergence_signals': [], 'divergence_count': 0, 'divergence_severity': 0.0}
    
    def _generate_flow_signals(self, volume_metrics: Dict[str, Any], flow_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable flow signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'flow_signals': [],
                'breadth_implications': []
            }
            
            # Flow velocity signal
            flow_velocity = flow_patterns.get('flow_velocity', 1.0)
            if flow_velocity > self.flow_threshold:
                signals['flow_signals'].append('accelerating_flow')
                signals['breadth_implications'].append('increasing_participation')
                signals['signal_strength'] += 0.3
            elif flow_velocity < (1.0 / self.flow_threshold):
                signals['flow_signals'].append('decelerating_flow')
                signals['breadth_implications'].append('decreasing_participation')
                signals['signal_strength'] += 0.2
            
            # Call/Put ratio signal
            cp_ratio = volume_metrics.get('call_put_volume_ratio', 1.0)
            if cp_ratio > 2.0:
                signals['flow_signals'].append('bullish_flow_bias')
                signals['breadth_implications'].append('call_dominated_breadth')
                signals['signal_strength'] += 0.2
            elif cp_ratio < 0.5:
                signals['flow_signals'].append('bearish_flow_bias')
                signals['breadth_implications'].append('put_dominated_breadth')
                signals['signal_strength'] += 0.2
            
            # Flow concentration signal
            concentration = volume_metrics.get('high_volume_concentration', 0.0)
            if concentration > 0.3:
                signals['flow_signals'].append('concentrated_flow')
                signals['breadth_implications'].append('narrow_breadth')
                signals['signal_strength'] += 0.2
            elif concentration < 0.1:
                signals['flow_signals'].append('distributed_flow')
                signals['breadth_implications'].append('broad_breadth')
                signals['signal_strength'] += 0.1
            
            # Determine primary signal
            if signals['signal_strength'] > 0.5:
                if any('accelerating' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'expanding_breadth'
                elif any('concentrated' in sig for sig in signals['flow_signals']):
                    signals['primary_signal'] = 'narrowing_breadth'
                else:
                    signals['primary_signal'] = 'shifting_breadth'
            elif signals['signal_strength'] > 0.2:
                signals['primary_signal'] = 'moderate_breadth_change'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating flow signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_breadth_score(self, volume_metrics: Dict[str, Any], flow_patterns: Dict[str, Any]) -> float:
        """Calculate overall breadth score from volume flow"""
        try:
            score = 0.5  # Base neutral score
            
            # Flow velocity contribution (30%)
            flow_velocity = flow_patterns.get('flow_velocity', 1.0)
            velocity_score = min(flow_velocity / 2.0, 1.0) if flow_velocity > 1.0 else max(flow_velocity, 0.0)
            score += (velocity_score - 0.5) * 0.3
            
            # Volume distribution contribution (25%)
            concentration = volume_metrics.get('high_volume_concentration', 0.0)
            distribution_score = max(1.0 - concentration * 2, 0.0)  # Lower concentration = higher breadth
            score += (distribution_score - 0.5) * 0.25
            
            # Call/Put balance contribution (25%)
            cp_ratio = volume_metrics.get('call_put_volume_ratio', 1.0)
            balance_score = 1.0 - abs(np.log(cp_ratio)) / 2.0 if cp_ratio > 0 else 0.0
            balance_score = max(min(balance_score, 1.0), 0.0)
            score += (balance_score - 0.5) * 0.25
            
            # Term structure contribution (20%)
            term_ratio = flow_patterns.get('term_flow_ratio', 1.0)
            term_score = 1.0 - abs(np.log(term_ratio)) / 2.0 if term_ratio > 0 else 0.5
            term_score = max(min(term_score, 1.0), 0.0)
            score += (term_score - 0.5) * 0.2
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating breadth score: {e}")
            return 0.5
    
    def _get_default_flow_analysis(self) -> Dict[str, Any]:
        """Get default flow analysis when data is insufficient"""
        return {
            'volume_metrics': {'total_volume': 0.0, 'average_volume': 0.0},
            'flow_patterns': {'flow_velocity': 1.0},
            'flow_divergences': {'divergence_signals': [], 'divergence_count': 0},
            'flow_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get summary of option volume flow analysis"""
        try:
            return {
                'flow_history_length': len(self.volume_history),
                'signal_history_length': len(self.flow_signals),
                'current_flow_velocity': self.volume_history[-1] / np.mean(self.volume_history[:-1]) if len(self.volume_history) > 1 else 1.0,
                'analysis_config': {
                    'flow_threshold': self.flow_threshold,
                    'flow_window': self.flow_window
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting flow summary: {e}")
            return {'status': 'error', 'error': str(e)}