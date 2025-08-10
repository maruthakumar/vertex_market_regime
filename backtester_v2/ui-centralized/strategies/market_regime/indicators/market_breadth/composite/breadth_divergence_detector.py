"""
Breadth Divergence Detector - Cross-Asset Breadth Divergence Detection
=====================================================================

Detects divergences between option and underlying breadth metrics.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BreadthDivergenceDetector:
    """Comprehensive breadth divergence detector"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Breadth Divergence Detector"""
        self.divergence_threshold = config.get('divergence_threshold', 0.3)
        self.lookback_window = config.get('lookback_window', 10)
        self.correlation_threshold = config.get('correlation_threshold', 0.5)
        
        # Historical tracking
        self.divergence_history = {
            'option_underlying_divergence': [],
            'volume_price_divergence': [],
            'breadth_momentum_divergence': [],
            'timestamps': []
        }
        
        # Severity tracking
        self.severity_metrics = {
            'extreme_divergences': 0,
            'moderate_divergences': 0,
            'mild_divergences': 0
        }
        
        logger.info("BreadthDivergenceDetector initialized")
    
    def detect_breadth_divergences(self, 
                                 option_breadth: Dict[str, Any], 
                                 underlying_breadth: Dict[str, Any],
                                 market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect comprehensive breadth divergences
        
        Args:
            option_breadth: Option breadth analysis results
            underlying_breadth: Underlying breadth analysis results
            market_context: Optional market context data
            
        Returns:
            Dict with divergence analysis results
        """
        try:
            if not option_breadth or not underlying_breadth:
                return self._get_default_divergence_analysis()
            
            # Core divergence detection
            core_divergences = self._detect_core_divergences(option_breadth, underlying_breadth)
            
            # Volume-price divergences
            volume_price_divergences = self._detect_volume_price_divergences(option_breadth, underlying_breadth, market_context)
            
            # Momentum divergences
            momentum_divergences = self._detect_momentum_divergences(option_breadth, underlying_breadth)
            
            # Cross-timeframe divergences
            timeframe_divergences = self._detect_timeframe_divergences(option_breadth, underlying_breadth)
            
            # Participation divergences
            participation_divergences = self._detect_participation_divergences(option_breadth, underlying_breadth)
            
            # Calculate divergence severity
            divergence_severity = self._calculate_divergence_severity(
                core_divergences, volume_price_divergences, momentum_divergences
            )
            
            # Generate divergence signals
            divergence_signals = self._generate_divergence_signals(
                core_divergences, volume_price_divergences, momentum_divergences, divergence_severity
            )
            
            # Update historical tracking
            self._update_divergence_history(core_divergences, volume_price_divergences, momentum_divergences)
            
            return {
                'core_divergences': core_divergences,
                'volume_price_divergences': volume_price_divergences,
                'momentum_divergences': momentum_divergences,
                'timeframe_divergences': timeframe_divergences,
                'participation_divergences': participation_divergences,
                'divergence_severity': divergence_severity,
                'divergence_signals': divergence_signals,
                'breadth_alignment_score': self._calculate_breadth_alignment_score(core_divergences, divergence_severity)
            }
            
        except Exception as e:
            logger.error(f"Error detecting breadth divergences: {e}")
            return self._get_default_divergence_analysis()
    
    def _detect_core_divergences(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Detect core breadth divergences between option and underlying metrics"""
        try:
            divergences = {
                'divergence_signals': [],
                'divergence_count': 0,
                'option_breadth_score': 0.5,
                'underlying_breadth_score': 0.5,
                'divergence_magnitude': 0.0
            }
            
            # Extract breadth scores
            option_score = self._extract_breadth_score(option_breadth)
            underlying_score = self._extract_breadth_score(underlying_breadth)
            
            divergences['option_breadth_score'] = float(option_score)
            divergences['underlying_breadth_score'] = float(underlying_score)
            
            # Calculate divergence magnitude
            divergence_magnitude = abs(option_score - underlying_score)
            divergences['divergence_magnitude'] = float(divergence_magnitude)
            
            # Detect significant divergences
            if divergence_magnitude > self.divergence_threshold:
                divergence_type = 'option_bullish_divergence' if option_score > underlying_score else 'option_bearish_divergence'
                
                divergences['divergence_signals'].append({
                    'type': divergence_type,
                    'magnitude': float(divergence_magnitude),
                    'option_score': float(option_score),
                    'underlying_score': float(underlying_score),
                    'severity': 'high' if divergence_magnitude > 0.5 else 'moderate'
                })
                
                divergences['divergence_count'] += 1
            
            # Check for specific metric divergences
            option_signals = option_breadth.get('flow_signals', {}).get('primary_signal', 'neutral')
            underlying_signals = underlying_breadth.get('breadth_signals', {}).get('primary_signal', 'neutral')
            
            if self._are_signals_divergent(option_signals, underlying_signals):
                divergences['divergence_signals'].append({
                    'type': 'signal_divergence',
                    'option_signal': option_signals,
                    'underlying_signal': underlying_signals,
                    'severity': 'moderate'
                })
                divergences['divergence_count'] += 1
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting core divergences: {e}")
            return {'divergence_signals': [], 'divergence_count': 0}
    
    def _detect_volume_price_divergences(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any], market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect volume-price divergences"""
        try:
            divergences = {
                'volume_divergences': [],
                'price_divergences': [],
                'combined_divergences': []
            }
            
            # Option volume vs underlying volume divergence
            option_volume_signals = option_breadth.get('volume_metrics', {})
            underlying_volume_signals = underlying_breadth.get('flow_metrics', {})
            
            if option_volume_signals and underlying_volume_signals:
                option_volume_flow = option_volume_signals.get('volume_flow_percentage', 0.0)
                underlying_volume_flow = underlying_volume_signals.get('volume_flow_percentage', 0.0)
                
                volume_divergence = abs(option_volume_flow - underlying_volume_flow)
                
                if volume_divergence > 0.2:  # 20% divergence threshold
                    divergences['volume_divergences'].append({
                        'type': 'option_underlying_volume_divergence',
                        'option_flow': float(option_volume_flow),
                        'underlying_flow': float(underlying_volume_flow),
                        'divergence': float(volume_divergence)
                    })
            
            # Price momentum vs breadth divergence
            if market_context and 'price_momentum' in market_context:
                price_momentum = market_context['price_momentum']
                option_momentum = option_breadth.get('flow_patterns', {}).get('flow_velocity', 1.0)
                
                momentum_divergence = self._calculate_momentum_divergence(price_momentum, option_momentum)
                
                if abs(momentum_divergence) > 0.3:
                    divergences['price_divergences'].append({
                        'type': 'price_breadth_momentum_divergence',
                        'price_momentum': float(price_momentum),
                        'breadth_momentum': float(option_momentum),
                        'divergence': float(momentum_divergence)
                    })
            
            # Participation vs price action divergence
            option_participation = option_breadth.get('flow_metrics', {}).get('participation_rate', 0.5)
            underlying_participation = underlying_breadth.get('participation_metrics', {}).get('overall_participation', 0.5)
            
            participation_divergence = abs(option_participation - underlying_participation)
            
            if participation_divergence > 0.25:
                divergences['combined_divergences'].append({
                    'type': 'participation_divergence',
                    'option_participation': float(option_participation),
                    'underlying_participation': float(underlying_participation),
                    'divergence': float(participation_divergence)
                })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting volume-price divergences: {e}")
            return {'volume_divergences': [], 'price_divergences': [], 'combined_divergences': []}
    
    def _detect_momentum_divergences(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Detect momentum divergences between option and underlying breadth"""
        try:
            divergences = {
                'momentum_divergences': [],
                'acceleration_divergences': [],
                'trend_divergences': []
            }
            
            # Flow momentum divergence
            option_momentum = option_breadth.get('flow_patterns', {}).get('flow_velocity', 1.0)
            underlying_momentum = underlying_breadth.get('volume_momentum', {}).get('momentum_strength', 0.0)
            
            momentum_diff = abs(option_momentum - 1.0) - abs(underlying_momentum)
            
            if abs(momentum_diff) > 0.3:
                divergences['momentum_divergences'].append({
                    'type': 'flow_momentum_divergence',
                    'option_momentum': float(option_momentum),
                    'underlying_momentum': float(underlying_momentum),
                    'difference': float(momentum_diff)
                })
            
            # Breadth acceleration divergence
            option_signals = option_breadth.get('flow_signals', {}).get('flow_signals', [])
            underlying_signals = underlying_breadth.get('breadth_signals', {}).get('breadth_signals', [])
            
            option_acceleration = any('accelerating' in sig for sig in option_signals)
            underlying_acceleration = any('thrust' in sig or 'acceleration' in sig for sig in underlying_signals)
            
            if option_acceleration != underlying_acceleration:
                divergences['acceleration_divergences'].append({
                    'type': 'acceleration_divergence',
                    'option_accelerating': option_acceleration,
                    'underlying_accelerating': underlying_acceleration
                })
            
            # Trend strength divergence
            option_trend_strength = option_breadth.get('flow_signals', {}).get('signal_strength', 0.0)
            underlying_trend_strength = underlying_breadth.get('breadth_signals', {}).get('signal_strength', 0.0)
            
            trend_divergence = abs(option_trend_strength - underlying_trend_strength)
            
            if trend_divergence > 0.4:
                divergences['trend_divergences'].append({
                    'type': 'trend_strength_divergence',
                    'option_strength': float(option_trend_strength),
                    'underlying_strength': float(underlying_trend_strength),
                    'divergence': float(trend_divergence)
                })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting momentum divergences: {e}")
            return {'momentum_divergences': [], 'acceleration_divergences': [], 'trend_divergences': []}
    
    def _detect_timeframe_divergences(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Detect cross-timeframe divergences"""
        try:
            divergences = {
                'short_term_divergences': [],
                'medium_term_divergences': [],
                'long_term_divergences': []
            }
            
            # Check for timeframe-specific patterns in option breadth
            if 'flow_patterns' in option_breadth:
                flow_patterns = option_breadth['flow_patterns']
                
                # Near-term vs far-term option flow
                near_flow = flow_patterns.get('near_term_flow', 0)
                far_flow = flow_patterns.get('far_term_flow', 0)
                
                if near_flow > 0 and far_flow > 0:
                    term_ratio = near_flow / far_flow
                    
                    if term_ratio > 2.0:  # Heavy near-term bias
                        divergences['short_term_divergences'].append({
                            'type': 'near_term_bias',
                            'term_ratio': float(term_ratio),
                            'implication': 'short_term_focused_activity'
                        })
                    elif term_ratio < 0.5:  # Heavy far-term bias
                        divergences['long_term_divergences'].append({
                            'type': 'far_term_bias',
                            'term_ratio': float(term_ratio),
                            'implication': 'long_term_positioning'
                        })
            
            # Check underlying breadth timeframes
            if 'hl_trends' in underlying_breadth:
                hl_trends = underlying_breadth['hl_trends']
                
                short_bias = hl_trends.get('short_term_bias', 'neutral')
                long_bias = hl_trends.get('long_term_bias', 'neutral')
                
                if short_bias != long_bias and short_bias != 'neutral' and long_bias != 'neutral':
                    divergences['medium_term_divergences'].append({
                        'type': 'timeframe_bias_divergence',
                        'short_term_bias': short_bias,
                        'long_term_bias': long_bias
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting timeframe divergences: {e}")
            return {'short_term_divergences': [], 'medium_term_divergences': [], 'long_term_divergences': []}
    
    def _detect_participation_divergences(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Detect participation pattern divergences"""
        try:
            divergences = {
                'participation_divergences': [],
                'quality_divergences': [],
                'breadth_divergences': []
            }
            
            # Option vs underlying participation rates
            option_participation = option_breadth.get('flow_metrics', {}).get('participation_rate', 0.5)
            underlying_participation = underlying_breadth.get('participation_metrics', {}).get('overall_participation', 0.5)
            
            participation_diff = abs(option_participation - underlying_participation)
            
            if participation_diff > 0.2:
                divergences['participation_divergences'].append({
                    'type': 'participation_rate_divergence',
                    'option_participation': float(option_participation),
                    'underlying_participation': float(underlying_participation),
                    'difference': float(participation_diff)
                })
            
            # Quality of participation divergence
            option_quality = option_breadth.get('breadth_score', 0.5)
            underlying_quality = underlying_breadth.get('breadth_score', 0.5)
            
            quality_diff = abs(option_quality - underlying_quality)
            
            if quality_diff > 0.3:
                divergences['quality_divergences'].append({
                    'type': 'participation_quality_divergence',
                    'option_quality': float(option_quality),
                    'underlying_quality': float(underlying_quality),
                    'difference': float(quality_diff)
                })
            
            # Sector vs strike breadth divergence
            if 'sector_participation' in option_breadth:
                sector_breadth = option_breadth['sector_participation'].get('participation_ratio', 0.5)
                strike_breadth = option_breadth.get('flow_metrics', {}).get('participation_rate', 0.5)
                
                breadth_diff = abs(sector_breadth - strike_breadth)
                
                if breadth_diff > 0.25:
                    divergences['breadth_divergences'].append({
                        'type': 'sector_strike_breadth_divergence',
                        'sector_breadth': float(sector_breadth),
                        'strike_breadth': float(strike_breadth),
                        'difference': float(breadth_diff)
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting participation divergences: {e}")
            return {'participation_divergences': [], 'quality_divergences': [], 'breadth_divergences': []}
    
    def _calculate_divergence_severity(self, core_div: Dict[str, Any], volume_div: Dict[str, Any], momentum_div: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall divergence severity"""
        try:
            severity = {
                'overall_severity': 0.0,
                'severity_classification': 'minimal',
                'component_severities': {},
                'risk_level': 'low'
            }
            
            total_severity = 0.0
            component_count = 0
            
            # Core divergence severity
            core_severity = core_div.get('divergence_magnitude', 0.0)
            severity['component_severities']['core'] = float(core_severity)
            total_severity += core_severity * 0.4  # 40% weight
            component_count += 1
            
            # Volume divergence severity
            volume_divergences = (len(volume_div.get('volume_divergences', [])) +
                                len(volume_div.get('price_divergences', [])) +
                                len(volume_div.get('combined_divergences', [])))
            
            volume_severity = min(volume_divergences * 0.2, 1.0)  # Cap at 1.0
            severity['component_severities']['volume'] = float(volume_severity)
            total_severity += volume_severity * 0.3  # 30% weight
            component_count += 1
            
            # Momentum divergence severity
            momentum_divergences = (len(momentum_div.get('momentum_divergences', [])) +
                                  len(momentum_div.get('acceleration_divergences', [])) +
                                  len(momentum_div.get('trend_divergences', [])))
            
            momentum_severity = min(momentum_divergences * 0.25, 1.0)  # Cap at 1.0
            severity['component_severities']['momentum'] = float(momentum_severity)
            total_severity += momentum_severity * 0.3  # 30% weight
            component_count += 1
            
            # Calculate overall severity
            severity['overall_severity'] = float(total_severity)
            
            # Classify severity
            if total_severity > 0.7:
                severity['severity_classification'] = 'extreme'
                severity['risk_level'] = 'high'
                self.severity_metrics['extreme_divergences'] += 1
            elif total_severity > 0.4:
                severity['severity_classification'] = 'moderate'
                severity['risk_level'] = 'medium'
                self.severity_metrics['moderate_divergences'] += 1
            elif total_severity > 0.2:
                severity['severity_classification'] = 'mild'
                severity['risk_level'] = 'low'
                self.severity_metrics['mild_divergences'] += 1
            else:
                severity['severity_classification'] = 'minimal'
                severity['risk_level'] = 'very_low'
            
            return severity
            
        except Exception as e:
            logger.error(f"Error calculating divergence severity: {e}")
            return {'overall_severity': 0.0, 'severity_classification': 'minimal', 'risk_level': 'low'}
    
    def _generate_divergence_signals(self, core_div: Dict[str, Any], volume_div: Dict[str, Any], momentum_div: Dict[str, Any], severity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable divergence signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'divergence_signals': [],
                'trading_implications': []
            }
            
            overall_severity = severity.get('overall_severity', 0.0)
            
            # Core divergence signals
            core_signals = core_div.get('divergence_signals', [])
            for signal in core_signals:
                signal_type = signal.get('type', '')
                if 'bullish' in signal_type:
                    signals['divergence_signals'].append('option_bullish_divergence')
                    signals['trading_implications'].append('option_market_leading_upside')
                elif 'bearish' in signal_type:
                    signals['divergence_signals'].append('option_bearish_divergence')
                    signals['trading_implications'].append('option_market_leading_downside')
            
            # Volume divergence signals
            if volume_div.get('volume_divergences') or volume_div.get('combined_divergences'):
                signals['divergence_signals'].append('volume_breadth_divergence')
                signals['trading_implications'].append('institutional_retail_divergence')
            
            # Momentum divergence signals
            if momentum_div.get('momentum_divergences') or momentum_div.get('acceleration_divergences'):
                signals['divergence_signals'].append('momentum_breadth_divergence')
                signals['trading_implications'].append('breadth_momentum_misalignment')
            
            # Calculate signal strength
            signals['signal_strength'] = float(overall_severity)
            
            # Determine primary signal
            if overall_severity > 0.6:
                if any('bullish' in sig for sig in signals['divergence_signals']):
                    signals['primary_signal'] = 'strong_bullish_divergence'
                elif any('bearish' in sig for sig in signals['divergence_signals']):
                    signals['primary_signal'] = 'strong_bearish_divergence'
                else:
                    signals['primary_signal'] = 'strong_breadth_divergence'
            elif overall_severity > 0.3:
                signals['primary_signal'] = 'moderate_breadth_divergence'
            elif overall_severity > 0.1:
                signals['primary_signal'] = 'mild_breadth_divergence'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating divergence signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_breadth_alignment_score(self, core_div: Dict[str, Any], severity: Dict[str, Any]) -> float:
        """Calculate breadth alignment score (1.0 = perfect alignment, 0.0 = maximum divergence)"""
        try:
            overall_severity = severity.get('overall_severity', 0.0)
            
            # Invert severity to get alignment score
            alignment_score = 1.0 - overall_severity
            
            # Bonus for consistent signals
            option_score = core_div.get('option_breadth_score', 0.5)
            underlying_score = core_div.get('underlying_breadth_score', 0.5)
            
            # If both scores are in same direction (both > 0.5 or both < 0.5), give bonus
            if (option_score > 0.5 and underlying_score > 0.5) or (option_score < 0.5 and underlying_score < 0.5):
                alignment_score += 0.1
            
            return max(min(float(alignment_score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating breadth alignment score: {e}")
            return 0.5
    
    def _extract_breadth_score(self, breadth_data: Dict[str, Any]) -> float:
        """Extract breadth score from breadth analysis data"""
        try:
            # Try multiple possible keys for breadth score
            possible_keys = ['breadth_score', 'overall_score', 'composite_score', 'score']
            
            for key in possible_keys:
                if key in breadth_data:
                    return float(breadth_data[key])
            
            # If no direct score, try to calculate from components
            if 'flow_signals' in breadth_data:
                signal_strength = breadth_data['flow_signals'].get('signal_strength', 0.0)
                return 0.5 + (signal_strength - 0.5)  # Center around 0.5
            
            if 'breadth_signals' in breadth_data:
                signal_strength = breadth_data['breadth_signals'].get('signal_strength', 0.0)
                return 0.5 + (signal_strength - 0.5)  # Center around 0.5
            
            return 0.5  # Default neutral score
            
        except Exception as e:
            logger.error(f"Error extracting breadth score: {e}")
            return 0.5
    
    def _are_signals_divergent(self, option_signal: str, underlying_signal: str) -> bool:
        """Check if two signals are divergent"""
        try:
            bullish_signals = ['bullish', 'positive', 'expanding', 'thrust']
            bearish_signals = ['bearish', 'negative', 'contracting', 'decline']
            
            option_bullish = any(bull in option_signal.lower() for bull in bullish_signals)
            option_bearish = any(bear in option_signal.lower() for bear in bearish_signals)
            
            underlying_bullish = any(bull in underlying_signal.lower() for bull in bullish_signals)
            underlying_bearish = any(bear in underlying_signal.lower() for bear in bearish_signals)
            
            # Divergent if one is bullish and other is bearish
            return (option_bullish and underlying_bearish) or (option_bearish and underlying_bullish)
            
        except Exception as e:
            logger.error(f"Error checking signal divergence: {e}")
            return False
    
    def _calculate_momentum_divergence(self, price_momentum: float, breadth_momentum: float) -> float:
        """Calculate momentum divergence between price and breadth"""
        try:
            # Normalize both to same scale
            price_norm = np.tanh(price_momentum)  # Normalize to [-1, 1]
            breadth_norm = (breadth_momentum - 1.0) * 2  # Convert [0.5, 1.5] to [-1, 1]
            
            return price_norm - breadth_norm
            
        except Exception as e:
            logger.error(f"Error calculating momentum divergence: {e}")
            return 0.0
    
    def _update_divergence_history(self, core_div: Dict[str, Any], volume_div: Dict[str, Any], momentum_div: Dict[str, Any]):
        """Update historical divergence tracking"""
        try:
            # Update core divergence history
            divergence_magnitude = core_div.get('divergence_magnitude', 0.0)
            self.divergence_history['option_underlying_divergence'].append(divergence_magnitude)
            
            # Update volume divergence history
            volume_count = len(volume_div.get('volume_divergences', [])) + len(volume_div.get('combined_divergences', []))
            self.divergence_history['volume_price_divergence'].append(volume_count)
            
            # Update momentum divergence history
            momentum_count = len(momentum_div.get('momentum_divergences', [])) + len(momentum_div.get('trend_divergences', []))
            self.divergence_history['breadth_momentum_divergence'].append(momentum_count)
            
            # Trim history to window size
            for key in ['option_underlying_divergence', 'volume_price_divergence', 'breadth_momentum_divergence']:
                if len(self.divergence_history[key]) > self.lookback_window * 2:
                    self.divergence_history[key].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating divergence history: {e}")
    
    def _get_default_divergence_analysis(self) -> Dict[str, Any]:
        """Get default divergence analysis when data is insufficient"""
        return {
            'core_divergences': {'divergence_signals': [], 'divergence_count': 0},
            'volume_price_divergences': {'volume_divergences': [], 'price_divergences': [], 'combined_divergences': []},
            'momentum_divergences': {'momentum_divergences': [], 'acceleration_divergences': [], 'trend_divergences': []},
            'timeframe_divergences': {'short_term_divergences': [], 'medium_term_divergences': [], 'long_term_divergences': []},
            'participation_divergences': {'participation_divergences': [], 'quality_divergences': [], 'breadth_divergences': []},
            'divergence_severity': {'overall_severity': 0.0, 'severity_classification': 'minimal', 'risk_level': 'low'},
            'divergence_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_alignment_score': 0.5
        }
    
    def get_divergence_summary(self) -> Dict[str, Any]:
        """Get summary of divergence detection system"""
        try:
            return {
                'history_lengths': {key: len(values) for key, values in self.divergence_history.items()},
                'severity_metrics': self.severity_metrics.copy(),
                'recent_divergence_trend': np.mean(self.divergence_history['option_underlying_divergence'][-5:]) if len(self.divergence_history['option_underlying_divergence']) >= 5 else 0.0,
                'analysis_config': {
                    'divergence_threshold': self.divergence_threshold,
                    'lookback_window': self.lookback_window,
                    'correlation_threshold': self.correlation_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting divergence summary: {e}")
            return {'status': 'error', 'error': str(e)}