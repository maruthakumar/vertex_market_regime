"""
Indicator Fusion - Combining Option and Underlying Technical Indicators
======================================================================

Fuses signals from option-based and underlying-based technical indicators
to generate comprehensive market insights.

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


class IndicatorFusion:
    """
    Fuses option and underlying technical indicators
    
    Features:
    - Signal correlation analysis
    - Weighted signal combination
    - Divergence detection between option and underlying
    - Confidence scoring
    - Multi-indicator consensus
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Indicator Fusion"""
        # Weight configuration
        self.option_weight = config.get('option_weight', 0.6)
        self.underlying_weight = config.get('underlying_weight', 0.4)
        
        # Indicator weights
        self.indicator_weights = config.get('indicator_weights', {
            'rsi': 0.25,
            'macd': 0.25,
            'bollinger': 0.25,
            'volume_flow': 0.15,
            'trend_strength': 0.10
        })
        
        # Fusion parameters
        self.min_agreement_threshold = config.get('min_agreement_threshold', 0.6)
        self.divergence_threshold = config.get('divergence_threshold', 0.3)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Advanced features
        self.enable_ml_fusion = config.get('enable_ml_fusion', False)
        self.enable_adaptive_weights = config.get('enable_adaptive_weights', True)
        
        # History tracking
        self.fusion_history = {
            'signals': [],
            'divergences': [],
            'consensus': [],
            'performance': []
        }
        
        logger.info(f"IndicatorFusion initialized: option_weight={self.option_weight}")
    
    def fuse_indicators(self,
                       option_indicators: Dict[str, Any],
                       underlying_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse option and underlying technical indicators
        
        Args:
            option_indicators: Results from option-based indicators
            underlying_indicators: Results from underlying-based indicators
            
        Returns:
            Dict with fused analysis and signals
        """
        try:
            results = {
                'fused_signals': {},
                'indicator_agreement': {},
                'divergences': [],
                'consensus': {},
                'confidence_scores': {},
                'recommendations': {},
                'fusion_regime': None
            }
            
            # Extract signals from each indicator type
            option_signals = self._extract_option_signals(option_indicators)
            underlying_signals = self._extract_underlying_signals(underlying_indicators)
            
            # Calculate indicator agreement
            results['indicator_agreement'] = self._calculate_agreement(
                option_signals, underlying_signals
            )
            
            # Detect divergences
            results['divergences'] = self._detect_indicator_divergences(
                option_signals, underlying_signals
            )
            
            # Fuse signals with weights
            results['fused_signals'] = self._fuse_weighted_signals(
                option_signals, underlying_signals
            )
            
            # Calculate consensus
            results['consensus'] = self._calculate_consensus(
                option_signals, underlying_signals
            )
            
            # Generate confidence scores
            results['confidence_scores'] = self._calculate_confidence_scores(
                results
            )
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # Classify fusion regime
            results['fusion_regime'] = self._classify_fusion_regime(results)
            
            # Update history
            self._update_fusion_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fusing indicators: {e}")
            return self._get_default_results()
    
    def _extract_option_signals(self, option_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signals from option indicators"""
        try:
            signals = {
                'rsi': {},
                'macd': {},
                'bollinger': {},
                'volume_flow': {}
            }
            
            # Extract RSI signals
            if 'rsi' in option_indicators and 'signals' in option_indicators['rsi']:
                signals['rsi'] = {
                    'signal': option_indicators['rsi']['signals'].get('primary_signal', 'neutral'),
                    'strength': option_indicators['rsi']['signals'].get('signal_strength', 0.0),
                    'regime': option_indicators['rsi'].get('regime', 'undefined')
                }
            
            # Extract MACD signals
            if 'macd' in option_indicators and 'signals' in option_indicators['macd']:
                signals['macd'] = {
                    'signal': option_indicators['macd']['signals'].get('primary_signal', 'neutral'),
                    'strength': option_indicators['macd']['signals'].get('signal_strength', 0.0),
                    'momentum': option_indicators['macd'].get('momentum', 'undefined')
                }
            
            # Extract Bollinger signals
            if 'bollinger' in option_indicators and 'signals' in option_indicators['bollinger']:
                signals['bollinger'] = {
                    'signal': option_indicators['bollinger']['signals'].get('primary_signal', 'neutral'),
                    'strength': option_indicators['bollinger']['signals'].get('signal_strength', 0.0),
                    'volatility': option_indicators['bollinger'].get('volatility_state', 'undefined')
                }
            
            # Extract Volume Flow signals
            if 'volume_flow' in option_indicators and 'flow_signals' in option_indicators['volume_flow']:
                signals['volume_flow'] = {
                    'signal': option_indicators['volume_flow']['flow_signals'].get('primary_signal', 'neutral'),
                    'strength': option_indicators['volume_flow']['flow_signals'].get('signal_strength', 0.0),
                    'smart_money': option_indicators['volume_flow']['flow_signals'].get('smart_money_signal', 'none')
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error extracting option signals: {e}")
            return {}
    
    def _extract_underlying_signals(self, underlying_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signals from underlying indicators"""
        try:
            signals = {
                'rsi': {},
                'macd': {},
                'bollinger': {},
                'trend_strength': {}
            }
            
            # Extract RSI signals
            if 'rsi' in underlying_indicators and 'signals' in underlying_indicators['rsi']:
                signals['rsi'] = {
                    'signal': underlying_indicators['rsi']['signals'].get('primary_signal', 'neutral'),
                    'strength': underlying_indicators['rsi']['signals'].get('signal_strength', 0.0),
                    'confidence': underlying_indicators['rsi']['signals'].get('signal_confidence', 0.0)
                }
            
            # Extract MACD signals
            if 'macd' in underlying_indicators and 'signals' in underlying_indicators['macd']:
                signals['macd'] = {
                    'signal': underlying_indicators['macd']['signals'].get('primary_signal', 'neutral'),
                    'strength': underlying_indicators['macd']['signals'].get('signal_strength', 0.0),
                    'trade_bias': underlying_indicators['macd']['signals'].get('trade_bias', 'neutral')
                }
            
            # Extract Bollinger signals
            if 'bollinger' in underlying_indicators and 'signals' in underlying_indicators['bollinger']:
                signals['bollinger'] = {
                    'signal': underlying_indicators['bollinger']['signals'].get('primary_signal', 'neutral'),
                    'strength': underlying_indicators['bollinger']['signals'].get('signal_strength', 0.0),
                    'mean_reversion': underlying_indicators['bollinger']['signals'].get('mean_reversion', False)
                }
            
            # Extract Trend Strength signals
            if 'trend_strength' in underlying_indicators and 'signals' in underlying_indicators['trend_strength']:
                signals['trend_strength'] = {
                    'signal': underlying_indicators['trend_strength']['signals'].get('primary_signal', 'neutral'),
                    'strength': underlying_indicators['trend_strength']['signals'].get('signal_strength', 0.0),
                    'quality': underlying_indicators['trend_strength']['signals'].get('trend_quality', 'poor')
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error extracting underlying signals: {e}")
            return {}
    
    def _calculate_agreement(self,
                           option_signals: Dict[str, Any],
                           underlying_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate agreement between option and underlying signals"""
        try:
            agreement = {
                'overall': 0.0,
                'by_indicator': {},
                'directional_agreement': 'neutral',
                'strength_correlation': 0.0
            }
            
            agreements = []
            option_strengths = []
            underlying_strengths = []
            
            # Check agreement for each indicator type
            for indicator in ['rsi', 'macd', 'bollinger']:
                if indicator in option_signals and indicator in underlying_signals:
                    opt_sig = option_signals[indicator]
                    und_sig = underlying_signals[indicator]
                    
                    if opt_sig and und_sig:
                        # Check signal agreement
                        signal_match = self._signals_agree(
                            opt_sig.get('signal', 'neutral'),
                            und_sig.get('signal', 'neutral')
                        )
                        
                        agreement['by_indicator'][indicator] = {
                            'agrees': signal_match,
                            'option_strength': opt_sig.get('strength', 0.0),
                            'underlying_strength': und_sig.get('strength', 0.0)
                        }
                        
                        agreements.append(1.0 if signal_match else 0.0)
                        option_strengths.append(opt_sig.get('strength', 0.0))
                        underlying_strengths.append(und_sig.get('strength', 0.0))
            
            # Calculate overall agreement
            if agreements:
                agreement['overall'] = np.mean(agreements)
            
            # Calculate strength correlation
            if option_strengths and underlying_strengths:
                if len(option_strengths) > 1:
                    correlation = np.corrcoef(option_strengths, underlying_strengths)[0, 1]
                    agreement['strength_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            
            # Determine directional agreement
            if agreement['overall'] > 0.7:
                agreement['directional_agreement'] = 'strong_agreement'
            elif agreement['overall'] > 0.5:
                agreement['directional_agreement'] = 'moderate_agreement'
            elif agreement['overall'] < 0.3:
                agreement['directional_agreement'] = 'strong_disagreement'
            else:
                agreement['directional_agreement'] = 'mixed'
            
            return agreement
            
        except Exception as e:
            logger.error(f"Error calculating agreement: {e}")
            return {'overall': 0.0, 'by_indicator': {}}
    
    def _detect_indicator_divergences(self,
                                    option_signals: Dict[str, Any],
                                    underlying_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect divergences between option and underlying indicators"""
        try:
            divergences = []
            
            # Check each indicator type
            for indicator in ['rsi', 'macd', 'bollinger']:
                if indicator in option_signals and indicator in underlying_signals:
                    opt_sig = option_signals[indicator]
                    und_sig = underlying_signals[indicator]
                    
                    if opt_sig and und_sig:
                        opt_strength = opt_sig.get('strength', 0.0)
                        und_strength = und_sig.get('strength', 0.0)
                        
                        # Check for opposing signals
                        if (opt_strength > 0.3 and und_strength < -0.3) or \
                           (opt_strength < -0.3 and und_strength > 0.3):
                            divergences.append({
                                'type': f'{indicator}_divergence',
                                'option_signal': opt_sig.get('signal', 'neutral'),
                                'underlying_signal': und_sig.get('signal', 'neutral'),
                                'severity': abs(opt_strength - und_strength),
                                'favored_side': 'option' if abs(opt_strength) > abs(und_strength) else 'underlying'
                            })
            
            # Check for volume flow vs price divergence
            if 'volume_flow' in option_signals and 'trend_strength' in underlying_signals:
                vol_signal = option_signals['volume_flow'].get('signal', 'neutral')
                trend_signal = underlying_signals['trend_strength'].get('signal', 'neutral')
                
                if self._check_volume_price_divergence(vol_signal, trend_signal):
                    divergences.append({
                        'type': 'volume_price_divergence',
                        'volume_signal': vol_signal,
                        'price_signal': trend_signal,
                        'severity': 'high',
                        'implication': 'potential_reversal'
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {e}")
            return []
    
    def _fuse_weighted_signals(self,
                             option_signals: Dict[str, Any],
                             underlying_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse signals using weighted combination"""
        try:
            fused = {
                'composite_signal': 'neutral',
                'composite_strength': 0.0,
                'dominant_indicator': None,
                'signal_components': {}
            }
            
            weighted_strengths = []
            signal_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            # Process each indicator
            for indicator in self.indicator_weights:
                weight = self.indicator_weights[indicator]
                
                # Get option signal
                opt_strength = 0.0
                if indicator in option_signals and option_signals[indicator]:
                    opt_strength = option_signals[indicator].get('strength', 0.0)
                
                # Get underlying signal
                und_strength = 0.0
                if indicator in underlying_signals and underlying_signals[indicator]:
                    und_strength = underlying_signals[indicator].get('strength', 0.0)
                
                # Combine with source weights
                if indicator == 'volume_flow':
                    # Volume flow is option-only
                    combined_strength = opt_strength
                elif indicator == 'trend_strength':
                    # Trend strength is underlying-only
                    combined_strength = und_strength
                else:
                    # Combine both sources
                    combined_strength = (
                        self.option_weight * opt_strength +
                        self.underlying_weight * und_strength
                    )
                
                # Apply indicator weight
                weighted_strength = combined_strength * weight
                weighted_strengths.append(weighted_strength)
                
                # Track components
                fused['signal_components'][indicator] = {
                    'strength': combined_strength,
                    'weighted_strength': weighted_strength,
                    'weight': weight
                }
                
                # Count signal types
                if combined_strength > 0.3:
                    signal_counts['bullish'] += 1
                elif combined_strength < -0.3:
                    signal_counts['bearish'] += 1
                else:
                    signal_counts['neutral'] += 1
            
            # Calculate composite strength
            fused['composite_strength'] = sum(weighted_strengths)
            
            # Determine composite signal
            if fused['composite_strength'] > 0.3:
                fused['composite_signal'] = 'bullish'
            elif fused['composite_strength'] < -0.3:
                fused['composite_signal'] = 'bearish'
            else:
                fused['composite_signal'] = 'neutral'
            
            # Find dominant indicator
            if fused['signal_components']:
                fused['dominant_indicator'] = max(
                    fused['signal_components'].items(),
                    key=lambda x: abs(x[1]['weighted_strength'])
                )[0]
            
            return fused
            
        except Exception as e:
            logger.error(f"Error fusing signals: {e}")
            return {'composite_signal': 'neutral', 'composite_strength': 0.0}
    
    def _calculate_consensus(self,
                           option_signals: Dict[str, Any],
                           underlying_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus across all indicators"""
        try:
            consensus = {
                'level': 'no_consensus',
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'confidence': 0.0,
                'key_supporters': [],
                'key_dissenters': []
            }
            
            # Collect all signals
            all_signals = []
            
            for indicator, signals in option_signals.items():
                if signals and 'strength' in signals:
                    all_signals.append({
                        'source': f'option_{indicator}',
                        'strength': signals['strength']
                    })
            
            for indicator, signals in underlying_signals.items():
                if signals and 'strength' in signals:
                    all_signals.append({
                        'source': f'underlying_{indicator}',
                        'strength': signals['strength']
                    })
            
            # Count signal directions
            for sig in all_signals:
                if sig['strength'] > 0.3:
                    consensus['bullish_count'] += 1
                    consensus['key_supporters'].append(sig['source'])
                elif sig['strength'] < -0.3:
                    consensus['bearish_count'] += 1
                    consensus['key_supporters'].append(sig['source'])
                else:
                    consensus['neutral_count'] += 1
                    consensus['key_dissenters'].append(sig['source'])
            
            total_signals = len(all_signals)
            
            # Determine consensus level
            if total_signals > 0:
                max_count = max(consensus['bullish_count'], consensus['bearish_count'])
                consensus_ratio = max_count / total_signals
                
                if consensus_ratio > 0.8:
                    consensus['level'] = 'strong_consensus'
                    consensus['confidence'] = consensus_ratio
                elif consensus_ratio > 0.6:
                    consensus['level'] = 'moderate_consensus'
                    consensus['confidence'] = consensus_ratio
                elif consensus_ratio > 0.4:
                    consensus['level'] = 'weak_consensus'
                    consensus['confidence'] = consensus_ratio
                else:
                    consensus['level'] = 'no_consensus'
                    consensus['confidence'] = 0.0
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return {'level': 'no_consensus', 'confidence': 0.0}
    
    def _calculate_confidence_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence scores for the fused analysis"""
        try:
            confidence = {
                'overall': 0.0,
                'signal_confidence': 0.0,
                'agreement_confidence': 0.0,
                'consensus_confidence': 0.0,
                'factors': []
            }
            
            # Signal strength confidence
            composite_strength = abs(results['fused_signals'].get('composite_strength', 0.0))
            confidence['signal_confidence'] = min(composite_strength, 1.0)
            
            # Agreement confidence
            agreement_level = results['indicator_agreement'].get('overall', 0.0)
            confidence['agreement_confidence'] = agreement_level
            
            # Consensus confidence
            confidence['consensus_confidence'] = results['consensus'].get('confidence', 0.0)
            
            # Calculate overall confidence
            confidence_components = [
                confidence['signal_confidence'] * 0.4,
                confidence['agreement_confidence'] * 0.3,
                confidence['consensus_confidence'] * 0.3
            ]
            
            confidence['overall'] = sum(confidence_components)
            
            # Add confidence factors
            if confidence['signal_confidence'] > 0.7:
                confidence['factors'].append('strong_signal')
            
            if confidence['agreement_confidence'] > 0.7:
                confidence['factors'].append('high_agreement')
            
            if confidence['consensus_confidence'] > 0.7:
                confidence['factors'].append('strong_consensus')
            
            # Penalize for divergences
            if results['divergences']:
                confidence['overall'] *= (1 - 0.1 * len(results['divergences']))
                confidence['factors'].append('divergence_penalty')
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return {'overall': 0.0}
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendations based on fusion analysis"""
        try:
            recommendations = {
                'action': 'hold',
                'confidence': 0.0,
                'rationale': [],
                'risk_level': 'medium',
                'suggested_position_size': 0.5
            }
            
            confidence = results['confidence_scores'].get('overall', 0.0)
            signal = results['fused_signals'].get('composite_signal', 'neutral')
            strength = results['fused_signals'].get('composite_strength', 0.0)
            
            # Determine action
            if confidence > self.confidence_threshold:
                if signal == 'bullish' and strength > 0.5:
                    recommendations['action'] = 'strong_buy'
                    recommendations['rationale'].append('strong_bullish_consensus')
                elif signal == 'bullish':
                    recommendations['action'] = 'buy'
                    recommendations['rationale'].append('bullish_consensus')
                elif signal == 'bearish' and strength < -0.5:
                    recommendations['action'] = 'strong_sell'
                    recommendations['rationale'].append('strong_bearish_consensus')
                elif signal == 'bearish':
                    recommendations['action'] = 'sell'
                    recommendations['rationale'].append('bearish_consensus')
            
            # Set confidence
            recommendations['confidence'] = confidence
            
            # Add specific rationale
            if results['consensus'].get('level') == 'strong_consensus':
                recommendations['rationale'].append('strong_indicator_consensus')
            
            if results['indicator_agreement'].get('directional_agreement') == 'strong_agreement':
                recommendations['rationale'].append('option_underlying_agreement')
            
            # Check for volume confirmation
            volume_component = results['fused_signals']['signal_components'].get('volume_flow', {})
            if volume_component.get('strength', 0) * strength > 0:  # Same direction
                recommendations['rationale'].append('volume_confirmation')
            
            # Set risk level
            if results['divergences']:
                recommendations['risk_level'] = 'high'
                recommendations['rationale'].append('divergence_warning')
            elif confidence > 0.8:
                recommendations['risk_level'] = 'low'
            
            # Suggest position size based on confidence
            if confidence > 0.8:
                recommendations['suggested_position_size'] = 1.0
            elif confidence > 0.6:
                recommendations['suggested_position_size'] = 0.7
            elif confidence > 0.4:
                recommendations['suggested_position_size'] = 0.5
            else:
                recommendations['suggested_position_size'] = 0.3
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    def _classify_fusion_regime(self, results: Dict[str, Any]) -> str:
        """Classify the market regime based on fusion analysis"""
        try:
            signal = results['fused_signals'].get('composite_signal', 'neutral')
            strength = abs(results['fused_signals'].get('composite_strength', 0.0))
            consensus = results['consensus'].get('level', 'no_consensus')
            divergences = len(results['divergences'])
            
            # High confidence regimes
            if consensus == 'strong_consensus':
                if signal == 'bullish' and strength > 0.7:
                    return 'strong_bullish_regime'
                elif signal == 'bearish' and strength > 0.7:
                    return 'strong_bearish_regime'
                elif signal == 'bullish':
                    return 'bullish_regime'
                elif signal == 'bearish':
                    return 'bearish_regime'
            
            # Divergent regimes
            elif divergences >= 2:
                return 'high_divergence_regime'
            elif divergences == 1:
                return 'divergent_regime'
            
            # Mixed regimes
            elif consensus == 'no_consensus':
                return 'choppy_regime'
            
            # Moderate regimes
            elif consensus == 'moderate_consensus':
                if signal == 'bullish':
                    return 'mild_bullish_regime'
                elif signal == 'bearish':
                    return 'mild_bearish_regime'
            
            return 'neutral_regime'
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return 'undefined'
    
    def _signals_agree(self, signal1: str, signal2: str) -> bool:
        """Check if two signals agree in direction"""
        bullish_signals = ['bullish', 'strong_bullish', 'overbought', 'strong_buy', 
                          'uptrend', 'strong_uptrend', 'bullish_breakout']
        bearish_signals = ['bearish', 'strong_bearish', 'oversold', 'strong_sell',
                          'downtrend', 'strong_downtrend', 'bearish_breakout']
        
        signal1_bullish = any(b in signal1.lower() for b in bullish_signals)
        signal1_bearish = any(b in signal1.lower() for b in bearish_signals)
        
        signal2_bullish = any(b in signal2.lower() for b in bullish_signals)
        signal2_bearish = any(b in signal2.lower() for b in bearish_signals)
        
        return (signal1_bullish and signal2_bullish) or (signal1_bearish and signal2_bearish)
    
    def _check_volume_price_divergence(self, volume_signal: str, price_signal: str) -> bool:
        """Check for volume-price divergence"""
        volume_bullish = 'bullish' in volume_signal.lower()
        volume_bearish = 'bearish' in volume_signal.lower()
        
        price_bullish = 'uptrend' in price_signal.lower() or 'bullish' in price_signal.lower()
        price_bearish = 'downtrend' in price_signal.lower() or 'bearish' in price_signal.lower()
        
        return (volume_bullish and price_bearish) or (volume_bearish and price_bullish)
    
    def _update_fusion_history(self, results: Dict[str, Any]):
        """Update fusion history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Track signals
            self.fusion_history['signals'].append({
                'timestamp': timestamp,
                'signal': results['fused_signals'].get('composite_signal', 'neutral'),
                'strength': results['fused_signals'].get('composite_strength', 0.0),
                'confidence': results['confidence_scores'].get('overall', 0.0)
            })
            
            # Track divergences
            if results['divergences']:
                self.fusion_history['divergences'].extend([
                    {**div, 'timestamp': timestamp} for div in results['divergences']
                ])
            
            # Track consensus
            self.fusion_history['consensus'].append({
                'timestamp': timestamp,
                'level': results['consensus'].get('level', 'no_consensus'),
                'confidence': results['consensus'].get('confidence', 0.0)
            })
            
            # Keep only recent history
            max_history = 100
            for key in self.fusion_history:
                if len(self.fusion_history[key]) > max_history:
                    self.fusion_history[key] = self.fusion_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating fusion history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'fused_signals': {'composite_signal': 'neutral', 'composite_strength': 0.0},
            'indicator_agreement': {'overall': 0.0},
            'divergences': [],
            'consensus': {'level': 'no_consensus', 'confidence': 0.0},
            'confidence_scores': {'overall': 0.0},
            'recommendations': {'action': 'hold', 'confidence': 0.0},
            'fusion_regime': 'undefined'
        }
    
    def get_fusion_analysis(self) -> Dict[str, Any]:
        """Get comprehensive fusion analysis summary"""
        try:
            if not self.fusion_history['signals']:
                return {'status': 'no_history'}
            
            recent_signals = self.fusion_history['signals'][-20:]
            
            return {
                'current_signal': recent_signals[-1]['signal'] if recent_signals else 'neutral',
                'average_strength': np.mean([s['strength'] for s in recent_signals]),
                'average_confidence': np.mean([s['confidence'] for s in recent_signals]),
                'signal_consistency': self._calculate_signal_consistency(),
                'divergence_frequency': len(self.fusion_history['divergences']),
                'consensus_quality': self._assess_consensus_quality()
            }
            
        except Exception as e:
            logger.error(f"Error getting fusion analysis: {e}")
            return {'status': 'error'}
    
    def _calculate_signal_consistency(self) -> float:
        """Calculate how consistent signals have been"""
        try:
            if len(self.fusion_history['signals']) < 10:
                return 0.0
            
            recent_signals = [s['signal'] for s in self.fusion_history['signals'][-10:]]
            
            # Count signal changes
            changes = 0
            for i in range(1, len(recent_signals)):
                if recent_signals[i] != recent_signals[i-1]:
                    changes += 1
            
            consistency = 1.0 - (changes / (len(recent_signals) - 1))
            return float(consistency)
            
        except:
            return 0.0
    
    def _assess_consensus_quality(self) -> str:
        """Assess overall consensus quality"""
        try:
            if not self.fusion_history['consensus']:
                return 'no_data'
            
            recent_consensus = self.fusion_history['consensus'][-10:]
            
            strong_count = sum(1 for c in recent_consensus if c['level'] == 'strong_consensus')
            moderate_count = sum(1 for c in recent_consensus if c['level'] == 'moderate_consensus')
            
            if strong_count >= 7:
                return 'excellent'
            elif strong_count + moderate_count >= 7:
                return 'good'
            elif moderate_count >= 5:
                return 'fair'
            else:
                return 'poor'
                
        except:
            return 'unknown'