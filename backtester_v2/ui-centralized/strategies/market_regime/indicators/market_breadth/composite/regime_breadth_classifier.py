"""
Regime Breadth Classifier - Market Regime Classification using Breadth Analysis
==============================================================================

Classifies market regimes based on comprehensive breadth analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RegimeBreadthClassifier:
    """Market regime classifier using comprehensive breadth analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Regime Breadth Classifier"""
        self.regime_thresholds = config.get('regime_thresholds', {
            'strong_bullish': {'breadth_score': 0.8, 'participation': 0.7, 'momentum': 0.6},
            'moderate_bullish': {'breadth_score': 0.65, 'participation': 0.6, 'momentum': 0.4},
            'weak_bullish': {'breadth_score': 0.55, 'participation': 0.5, 'momentum': 0.2},
            'neutral': {'breadth_score': 0.5, 'participation': 0.4, 'momentum': 0.0},
            'weak_bearish': {'breadth_score': 0.45, 'participation': 0.5, 'momentum': -0.2},
            'moderate_bearish': {'breadth_score': 0.35, 'participation': 0.6, 'momentum': -0.4},
            'strong_bearish': {'breadth_score': 0.2, 'participation': 0.7, 'momentum': -0.6}
        })
        
        self.regime_confidence_threshold = config.get('regime_confidence_threshold', 0.6)
        self.regime_persistence_window = config.get('regime_persistence_window', 5)
        
        # Historical regime tracking
        self.regime_history = {
            'regimes': [],
            'confidence_scores': [],
            'breadth_scores': [],
            'regime_durations': [],
            'timestamps': []
        }
        
        # Regime transition tracking
        self.transition_metrics = {
            'regime_changes': 0,
            'false_signals': 0,
            'regime_persistence': 0,
            'current_regime_duration': 0
        }
        
        logger.info("RegimeBreadthClassifier initialized")
    
    def classify_market_regime(self, 
                             option_breadth: Dict[str, Any],
                             underlying_breadth: Dict[str, Any],
                             divergence_analysis: Optional[Dict[str, Any]] = None,
                             market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify market regime based on comprehensive breadth analysis
        
        Args:
            option_breadth: Option breadth analysis results
            underlying_breadth: Underlying breadth analysis results
            divergence_analysis: Optional divergence analysis results
            market_context: Optional market context data
            
        Returns:
            Dict with regime classification results
        """
        try:
            if not option_breadth or not underlying_breadth:
                return self._get_default_regime_classification()
            
            # Calculate composite breadth metrics
            composite_metrics = self._calculate_composite_metrics(option_breadth, underlying_breadth)
            
            # Assess breadth quality and participation
            breadth_quality = self._assess_breadth_quality(option_breadth, underlying_breadth)
            
            # Calculate momentum indicators
            momentum_indicators = self._calculate_momentum_indicators(option_breadth, underlying_breadth)
            
            # Incorporate divergence analysis
            divergence_impact = self._assess_divergence_impact(divergence_analysis) if divergence_analysis else {}
            
            # Primary regime classification
            primary_regime = self._classify_primary_regime(composite_metrics, breadth_quality, momentum_indicators)
            
            # Secondary regime characteristics
            secondary_characteristics = self._identify_secondary_characteristics(option_breadth, underlying_breadth, divergence_analysis)
            
            # Calculate regime confidence
            regime_confidence = self._calculate_regime_confidence(primary_regime, composite_metrics, breadth_quality)
            
            # Check regime persistence
            regime_persistence = self._check_regime_persistence(primary_regime)
            
            # Generate regime signals
            regime_signals = self._generate_regime_signals(primary_regime, regime_confidence, secondary_characteristics)
            
            # Update historical tracking
            self._update_regime_history(primary_regime, regime_confidence, composite_metrics)
            
            return {
                'primary_regime': primary_regime,
                'secondary_characteristics': secondary_characteristics,
                'regime_confidence': regime_confidence,
                'regime_persistence': regime_persistence,
                'composite_metrics': composite_metrics,
                'breadth_quality': breadth_quality,
                'momentum_indicators': momentum_indicators,
                'divergence_impact': divergence_impact,
                'regime_signals': regime_signals,
                'regime_stability_score': self._calculate_regime_stability_score(primary_regime, regime_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return self._get_default_regime_classification()
    
    def _calculate_composite_metrics(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite breadth metrics"""
        try:
            metrics = {}
            
            # Composite breadth score (weighted average)
            option_score = self._extract_breadth_score(option_breadth)
            underlying_score = self._extract_breadth_score(underlying_breadth)
            
            # Weight option breadth slightly higher (60/40) as it's more forward-looking
            composite_score = (option_score * 0.6) + (underlying_score * 0.4)
            metrics['composite_breadth_score'] = float(composite_score)
            
            # Participation metrics
            option_participation = self._extract_participation_metric(option_breadth)
            underlying_participation = self._extract_participation_metric(underlying_breadth)
            
            metrics['composite_participation'] = float((option_participation + underlying_participation) / 2)
            
            # Volume and flow metrics
            option_volume_strength = self._extract_volume_strength(option_breadth)
            underlying_volume_strength = self._extract_volume_strength(underlying_breadth)
            
            metrics['composite_volume_strength'] = float((option_volume_strength + underlying_volume_strength) / 2)
            
            # Signal alignment
            option_signals = option_breadth.get('flow_signals', {}).get('primary_signal', 'neutral')
            underlying_signals = underlying_breadth.get('breadth_signals', {}).get('primary_signal', 'neutral')
            
            metrics['signal_alignment'] = self._calculate_signal_alignment(option_signals, underlying_signals)
            
            # Breadth momentum
            option_momentum = self._extract_momentum_metric(option_breadth)
            underlying_momentum = self._extract_momentum_metric(underlying_breadth)
            
            metrics['composite_momentum'] = float((option_momentum + underlying_momentum) / 2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating composite metrics: {e}")
            return {'composite_breadth_score': 0.5, 'composite_participation': 0.5}
    
    def _assess_breadth_quality(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall breadth quality"""
        try:
            quality = {
                'overall_quality': 'moderate',
                'quality_score': 0.5,
                'quality_factors': {}
            }
            
            quality_score = 0.0
            factor_count = 0
            
            # Option breadth quality
            option_signals = option_breadth.get('flow_signals', {}).get('flow_signals', [])
            option_quality_indicators = [
                any('broad' in sig for sig in option_signals),
                any('high' in sig for sig in option_signals),
                option_breadth.get('breadth_score', 0.5) > 0.6
            ]
            
            option_quality = sum(option_quality_indicators) / len(option_quality_indicators)
            quality['quality_factors']['option_quality'] = float(option_quality)
            quality_score += option_quality * 0.4
            factor_count += 1
            
            # Underlying breadth quality
            underlying_signals = underlying_breadth.get('breadth_signals', {}).get('breadth_signals', [])
            underlying_quality_indicators = [
                any('broad' in sig for sig in underlying_signals),
                any('thrust' in sig for sig in underlying_signals),
                underlying_breadth.get('breadth_score', 0.5) > 0.6
            ]
            
            underlying_quality = sum(underlying_quality_indicators) / len(underlying_quality_indicators)
            quality['quality_factors']['underlying_quality'] = float(underlying_quality)
            quality_score += underlying_quality * 0.4
            factor_count += 1
            
            # Participation quality
            option_participation = self._extract_participation_metric(option_breadth)
            underlying_participation = self._extract_participation_metric(underlying_breadth)
            
            participation_quality = (option_participation + underlying_participation) / 2
            quality['quality_factors']['participation_quality'] = float(participation_quality)
            quality_score += participation_quality * 0.2
            factor_count += 1
            
            # Overall quality assessment
            quality['quality_score'] = float(quality_score)
            
            if quality_score > 0.8:
                quality['overall_quality'] = 'excellent'
            elif quality_score > 0.6:
                quality['overall_quality'] = 'good'
            elif quality_score > 0.4:
                quality['overall_quality'] = 'moderate'
            elif quality_score > 0.2:
                quality['overall_quality'] = 'poor'
            else:
                quality['overall_quality'] = 'very_poor'
            
            return quality
            
        except Exception as e:
            logger.error(f"Error assessing breadth quality: {e}")
            return {'overall_quality': 'moderate', 'quality_score': 0.5}
    
    def _calculate_momentum_indicators(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate momentum indicators for regime classification"""
        try:
            momentum = {
                'momentum_direction': 'neutral',
                'momentum_strength': 0.0,
                'acceleration': 0.0,
                'momentum_quality': 'moderate'
            }
            
            # Option momentum
            option_momentum = self._extract_momentum_metric(option_breadth)
            option_signals = option_breadth.get('flow_signals', {}).get('flow_signals', [])
            
            # Underlying momentum
            underlying_momentum = self._extract_momentum_metric(underlying_breadth)
            underlying_signals = underlying_breadth.get('breadth_signals', {}).get('breadth_signals', [])
            
            # Composite momentum
            composite_momentum = (option_momentum + underlying_momentum) / 2
            momentum['momentum_strength'] = float(abs(composite_momentum))
            
            # Momentum direction
            if composite_momentum > 0.1:
                momentum['momentum_direction'] = 'bullish'
            elif composite_momentum < -0.1:
                momentum['momentum_direction'] = 'bearish'
            else:
                momentum['momentum_direction'] = 'neutral'
            
            # Acceleration detection
            acceleration_signals = []
            acceleration_signals.extend([sig for sig in option_signals if 'accelerating' in sig or 'thrust' in sig])
            acceleration_signals.extend([sig for sig in underlying_signals if 'accelerating' in sig or 'thrust' in sig])
            
            if acceleration_signals:
                momentum['acceleration'] = 0.5 + len(acceleration_signals) * 0.1
            
            # Momentum quality assessment
            momentum_indicators = [
                momentum['momentum_strength'] > 0.3,
                len(acceleration_signals) > 0,
                abs(option_momentum - underlying_momentum) < 0.3  # Momentum alignment
            ]
            
            quality_score = sum(momentum_indicators) / len(momentum_indicators)
            
            if quality_score > 0.7:
                momentum['momentum_quality'] = 'high'
            elif quality_score > 0.4:
                momentum['momentum_quality'] = 'moderate'
            else:
                momentum['momentum_quality'] = 'low'
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {'momentum_direction': 'neutral', 'momentum_strength': 0.0}
    
    def _assess_divergence_impact(self, divergence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of divergences on regime classification"""
        try:
            impact = {
                'divergence_severity': 'minimal',
                'regime_risk': 'low',
                'stability_impact': 0.0,
                'classification_adjustment': 0.0
            }
            
            if not divergence_analysis:
                return impact
            
            # Extract divergence severity
            severity = divergence_analysis.get('divergence_severity', {})
            overall_severity = severity.get('overall_severity', 0.0)
            
            impact['stability_impact'] = float(overall_severity)
            impact['divergence_severity'] = severity.get('severity_classification', 'minimal')
            impact['regime_risk'] = severity.get('risk_level', 'low')
            
            # Calculate classification adjustment
            # Strong divergences reduce regime confidence
            if overall_severity > 0.6:
                impact['classification_adjustment'] = -0.2  # Reduce confidence
            elif overall_severity > 0.3:
                impact['classification_adjustment'] = -0.1
            else:
                impact['classification_adjustment'] = 0.0
            
            return impact
            
        except Exception as e:
            logger.error(f"Error assessing divergence impact: {e}")
            return {'divergence_severity': 'minimal', 'regime_risk': 'low'}
    
    def _classify_primary_regime(self, composite_metrics: Dict[str, Any], breadth_quality: Dict[str, Any], momentum_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Classify primary market regime"""
        try:
            regime = {
                'regime_type': 'neutral',
                'regime_strength': 'moderate',
                'regime_characteristics': [],
                'classification_confidence': 0.0
            }
            
            # Extract key metrics
            breadth_score = composite_metrics.get('composite_breadth_score', 0.5)
            participation = composite_metrics.get('composite_participation', 0.5)
            momentum_strength = momentum_indicators.get('momentum_strength', 0.0)
            momentum_direction = momentum_indicators.get('momentum_direction', 'neutral')
            quality_score = breadth_quality.get('quality_score', 0.5)
            
            # Classification logic
            classification_scores = {}
            
            # Strong bullish regime
            if (breadth_score >= self.regime_thresholds['strong_bullish']['breadth_score'] and
                participation >= self.regime_thresholds['strong_bullish']['participation'] and
                momentum_direction == 'bullish' and
                momentum_strength >= self.regime_thresholds['strong_bullish']['momentum']):
                
                classification_scores['strong_bullish'] = 0.9
                regime['regime_characteristics'].extend(['high_breadth', 'broad_participation', 'strong_momentum'])
            
            # Moderate bullish regime
            elif (breadth_score >= self.regime_thresholds['moderate_bullish']['breadth_score'] and
                  participation >= self.regime_thresholds['moderate_bullish']['participation'] and
                  momentum_direction in ['bullish', 'neutral'] and
                  momentum_strength >= self.regime_thresholds['moderate_bullish']['momentum']):
                
                classification_scores['moderate_bullish'] = 0.7
                regime['regime_characteristics'].extend(['moderate_breadth', 'decent_participation'])
            
            # Weak bullish regime
            elif (breadth_score >= self.regime_thresholds['weak_bullish']['breadth_score'] and
                  momentum_direction != 'bearish'):
                
                classification_scores['weak_bullish'] = 0.5
                regime['regime_characteristics'].append('limited_breadth')
            
            # Strong bearish regime
            elif (breadth_score <= self.regime_thresholds['strong_bearish']['breadth_score'] and
                  participation >= self.regime_thresholds['strong_bearish']['participation'] and
                  momentum_direction == 'bearish' and
                  momentum_strength >= abs(self.regime_thresholds['strong_bearish']['momentum'])):
                
                classification_scores['strong_bearish'] = 0.9
                regime['regime_characteristics'].extend(['weak_breadth', 'broad_decline', 'strong_selling'])
            
            # Moderate bearish regime
            elif (breadth_score <= self.regime_thresholds['moderate_bearish']['breadth_score'] and
                  participation >= self.regime_thresholds['moderate_bearish']['participation'] and
                  momentum_direction in ['bearish', 'neutral']):
                
                classification_scores['moderate_bearish'] = 0.7
                regime['regime_characteristics'].extend(['declining_breadth', 'broad_weakness'])
            
            # Weak bearish regime
            elif (breadth_score <= self.regime_thresholds['weak_bearish']['breadth_score'] and
                  momentum_direction != 'bullish'):
                
                classification_scores['weak_bearish'] = 0.5
                regime['regime_characteristics'].append('deteriorating_breadth')
            
            # Default to neutral
            else:
                classification_scores['neutral'] = 0.4
                regime['regime_characteristics'].append('mixed_signals')
            
            # Select regime with highest score
            if classification_scores:
                best_regime = max(classification_scores, key=classification_scores.get)
                regime['regime_type'] = best_regime
                regime['classification_confidence'] = float(classification_scores[best_regime])
                
                # Adjust for quality
                if quality_score > 0.7:
                    regime['regime_strength'] = 'strong'
                elif quality_score > 0.4:
                    regime['regime_strength'] = 'moderate'
                else:
                    regime['regime_strength'] = 'weak'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error classifying primary regime: {e}")
            return {'regime_type': 'neutral', 'regime_strength': 'moderate', 'classification_confidence': 0.0}
    
    def _identify_secondary_characteristics(self, option_breadth: Dict[str, Any], underlying_breadth: Dict[str, Any], divergence_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify secondary regime characteristics"""
        try:
            characteristics = {
                'market_structure': 'normal',
                'leadership': 'balanced',
                'rotation_activity': 'low',
                'volatility_regime': 'normal',
                'special_conditions': []
            }
            
            # Market structure analysis
            option_signals = option_breadth.get('flow_signals', {}).get('flow_signals', [])
            underlying_signals = underlying_breadth.get('breadth_signals', {}).get('breadth_signals', [])
            
            # Leadership analysis
            option_strength = option_breadth.get('breadth_score', 0.5)
            underlying_strength = underlying_breadth.get('breadth_score', 0.5)
            
            if option_strength > underlying_strength + 0.2:
                characteristics['leadership'] = 'option_led'
            elif underlying_strength > option_strength + 0.2:
                characteristics['leadership'] = 'underlying_led'
            else:
                characteristics['leadership'] = 'balanced'
            
            # Rotation activity
            if 'sector_rotation' in option_breadth:
                rotation_strength = option_breadth['sector_rotation'].get('rotation_strength', 0.0)
                if rotation_strength > 0.5:
                    characteristics['rotation_activity'] = 'high'
                elif rotation_strength > 0.25:
                    characteristics['rotation_activity'] = 'moderate'
                else:
                    characteristics['rotation_activity'] = 'low'
            
            # Market structure conditions
            if any('thrust' in sig for sig in underlying_signals):
                characteristics['special_conditions'].append('breadth_thrust')
            
            if any('extreme' in sig for sig in option_signals):
                characteristics['special_conditions'].append('extreme_option_activity')
            
            if any('narrow' in sig for sig in option_signals + underlying_signals):
                characteristics['market_structure'] = 'narrow'
            elif any('broad' in sig for sig in option_signals + underlying_signals):
                characteristics['market_structure'] = 'broad'
            
            # Divergence-based characteristics
            if divergence_analysis:
                divergence_signals = divergence_analysis.get('divergence_signals', {}).get('divergence_signals', [])
                if divergence_signals:
                    characteristics['special_conditions'].append('breadth_divergence')
                    
                    if any('bullish' in sig for sig in divergence_signals):
                        characteristics['special_conditions'].append('bullish_divergence')
                    elif any('bearish' in sig for sig in divergence_signals):
                        characteristics['special_conditions'].append('bearish_divergence')
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error identifying secondary characteristics: {e}")
            return {'market_structure': 'normal', 'leadership': 'balanced'}
    
    def _calculate_regime_confidence(self, primary_regime: Dict[str, Any], composite_metrics: Dict[str, Any], breadth_quality: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence in regime classification"""
        try:
            confidence = {
                'overall_confidence': 0.0,
                'confidence_factors': {},
                'confidence_level': 'low'
            }
            
            base_confidence = primary_regime.get('classification_confidence', 0.0)
            
            # Factors affecting confidence
            confidence_factors = {}
            
            # Signal alignment factor
            signal_alignment = composite_metrics.get('signal_alignment', 0.5)
            confidence_factors['signal_alignment'] = float(signal_alignment)
            
            # Quality factor
            quality_score = breadth_quality.get('quality_score', 0.5)
            confidence_factors['breadth_quality'] = float(quality_score)
            
            # Participation factor
            participation = composite_metrics.get('composite_participation', 0.5)
            confidence_factors['participation_level'] = float(participation)
            
            # Historical persistence factor
            if len(self.regime_history['regimes']) >= 3:
                recent_regimes = self.regime_history['regimes'][-3:]
                current_regime = primary_regime.get('regime_type', 'neutral')
                
                persistence = sum(1 for r in recent_regimes if r == current_regime) / len(recent_regimes)
                confidence_factors['historical_persistence'] = float(persistence)
            else:
                confidence_factors['historical_persistence'] = 0.5
            
            # Calculate weighted confidence
            weights = {
                'base_confidence': 0.4,
                'signal_alignment': 0.2,
                'breadth_quality': 0.2,
                'participation_level': 0.1,
                'historical_persistence': 0.1
            }
            
            overall_confidence = (
                base_confidence * weights['base_confidence'] +
                signal_alignment * weights['signal_alignment'] +
                quality_score * weights['breadth_quality'] +
                participation * weights['participation_level'] +
                confidence_factors['historical_persistence'] * weights['historical_persistence']
            )
            
            confidence['overall_confidence'] = float(overall_confidence)
            confidence['confidence_factors'] = confidence_factors
            
            # Classify confidence level
            if overall_confidence > 0.8:
                confidence['confidence_level'] = 'very_high'
            elif overall_confidence > 0.6:
                confidence['confidence_level'] = 'high'
            elif overall_confidence > 0.4:
                confidence['confidence_level'] = 'medium'
            elif overall_confidence > 0.2:
                confidence['confidence_level'] = 'low'
            else:
                confidence['confidence_level'] = 'very_low'
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return {'overall_confidence': 0.0, 'confidence_level': 'low'}
    
    def _check_regime_persistence(self, primary_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Check regime persistence and stability"""
        try:
            persistence = {
                'current_duration': 0,
                'stability_score': 0.0,
                'regime_maturity': 'new',
                'transition_risk': 'medium'
            }
            
            current_regime = primary_regime.get('regime_type', 'neutral')
            
            # Calculate current regime duration
            if self.regime_history['regimes']:
                duration = 1
                for regime in reversed(self.regime_history['regimes']):
                    if regime == current_regime:
                        duration += 1
                    else:
                        break
                
                persistence['current_duration'] = duration
                self.transition_metrics['current_regime_duration'] = duration
                
                # Regime maturity
                if duration >= 10:
                    persistence['regime_maturity'] = 'mature'
                elif duration >= 5:
                    persistence['regime_maturity'] = 'established'
                elif duration >= 2:
                    persistence['regime_maturity'] = 'developing'
                else:
                    persistence['regime_maturity'] = 'new'
            
            # Stability score based on recent consistency
            if len(self.regime_history['regimes']) >= 5:
                recent_regimes = self.regime_history['regimes'][-5:]
                stability = sum(1 for r in recent_regimes if r == current_regime) / len(recent_regimes)
                persistence['stability_score'] = float(stability)
                
                # Transition risk
                if stability > 0.8:
                    persistence['transition_risk'] = 'low'
                elif stability > 0.6:
                    persistence['transition_risk'] = 'medium'
                else:
                    persistence['transition_risk'] = 'high'
            
            return persistence
            
        except Exception as e:
            logger.error(f"Error checking regime persistence: {e}")
            return {'current_duration': 0, 'stability_score': 0.0}
    
    def _generate_regime_signals(self, primary_regime: Dict[str, Any], regime_confidence: Dict[str, Any], secondary_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable regime signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'regime_signals': [],
                'trading_implications': []
            }
            
            regime_type = primary_regime.get('regime_type', 'neutral')
            confidence_level = regime_confidence.get('overall_confidence', 0.0)
            regime_strength = primary_regime.get('regime_strength', 'moderate')
            
            # Primary signal based on regime
            if regime_type in ['strong_bullish', 'moderate_bullish']:
                signals['primary_signal'] = 'bullish_regime'
                signals['trading_implications'].extend(['broad_market_strength', 'positive_breadth'])
            elif regime_type in ['strong_bearish', 'moderate_bearish']:
                signals['primary_signal'] = 'bearish_regime'
                signals['trading_implications'].extend(['broad_market_weakness', 'negative_breadth'])
            elif regime_type in ['weak_bullish', 'weak_bearish']:
                signals['primary_signal'] = 'transitional_regime'
                signals['trading_implications'].append('mixed_signals')
            else:
                signals['primary_signal'] = 'neutral_regime'
                signals['trading_implications'].append('range_bound_market')
            
            # Signal strength
            signals['signal_strength'] = float(confidence_level)
            
            # Specific regime signals
            if regime_strength == 'strong' and confidence_level > 0.7:
                signals['regime_signals'].append('high_conviction_regime')
            
            if 'breadth_thrust' in secondary_characteristics.get('special_conditions', []):
                signals['regime_signals'].append('breadth_thrust_signal')
                signals['trading_implications'].append('momentum_acceleration')
            
            if 'breadth_divergence' in secondary_characteristics.get('special_conditions', []):
                signals['regime_signals'].append('breadth_divergence_warning')
                signals['trading_implications'].append('regime_instability_risk')
            
            # Leadership signals
            leadership = secondary_characteristics.get('leadership', 'balanced')
            if leadership == 'option_led':
                signals['regime_signals'].append('option_market_leadership')
                signals['trading_implications'].append('derivatives_driven_regime')
            elif leadership == 'underlying_led':
                signals['regime_signals'].append('cash_market_leadership')
                signals['trading_implications'].append('fundamental_driven_regime')
            
            # Rotation signals
            rotation_activity = secondary_characteristics.get('rotation_activity', 'low')
            if rotation_activity == 'high':
                signals['regime_signals'].append('active_sector_rotation')
                signals['trading_implications'].append('selective_stock_picking')
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating regime signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_regime_stability_score(self, primary_regime: Dict[str, Any], regime_confidence: Dict[str, Any]) -> float:
        """Calculate overall regime stability score"""
        try:
            regime_type = primary_regime.get('regime_type', 'neutral')
            confidence = regime_confidence.get('overall_confidence', 0.0)
            classification_confidence = primary_regime.get('classification_confidence', 0.0)
            
            # Base stability from confidence
            stability = (confidence + classification_confidence) / 2
            
            # Adjust for regime type (extreme regimes are less stable)
            if regime_type in ['strong_bullish', 'strong_bearish']:
                stability *= 0.9  # Slightly less stable
            elif regime_type in ['weak_bullish', 'weak_bearish']:
                stability *= 0.8  # More unstable
            
            # Historical stability factor
            if len(self.regime_history['regimes']) >= 5:
                recent_regimes = self.regime_history['regimes'][-5:]
                regime_changes = len(set(recent_regimes))
                stability_factor = max(0.5, 1.0 - (regime_changes - 1) * 0.2)
                stability *= stability_factor
            
            return max(min(float(stability), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating regime stability score: {e}")
            return 0.5
    
    def _extract_breadth_score(self, breadth_data: Dict[str, Any]) -> float:
        """Extract breadth score from breadth analysis data"""
        try:
            possible_keys = ['breadth_score', 'overall_score', 'composite_score', 'score']
            
            for key in possible_keys:
                if key in breadth_data:
                    return float(breadth_data[key])
            
            # Fallback to signal strength
            if 'flow_signals' in breadth_data:
                return float(breadth_data['flow_signals'].get('signal_strength', 0.5))
            
            if 'breadth_signals' in breadth_data:
                return float(breadth_data['breadth_signals'].get('signal_strength', 0.5))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error extracting breadth score: {e}")
            return 0.5
    
    def _extract_participation_metric(self, breadth_data: Dict[str, Any]) -> float:
        """Extract participation metric from breadth data"""
        try:
            # Try multiple possible keys
            if 'flow_metrics' in breadth_data:
                return float(breadth_data['flow_metrics'].get('participation_rate', 0.5))
            
            if 'participation_metrics' in breadth_data:
                return float(breadth_data['participation_metrics'].get('overall_participation', 0.5))
            
            if 'sector_participation' in breadth_data:
                return float(breadth_data['sector_participation'].get('participation_ratio', 0.5))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error extracting participation metric: {e}")
            return 0.5
    
    def _extract_volume_strength(self, breadth_data: Dict[str, Any]) -> float:
        """Extract volume strength metric"""
        try:
            if 'volume_metrics' in breadth_data:
                return float(breadth_data['volume_metrics'].get('up_volume_ratio', 0.5))
            
            if 'flow_metrics' in breadth_data:
                return float(breadth_data['flow_metrics'].get('up_volume_ratio', 0.5))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error extracting volume strength: {e}")
            return 0.5
    
    def _extract_momentum_metric(self, breadth_data: Dict[str, Any]) -> float:
        """Extract momentum metric from breadth data"""
        try:
            if 'flow_patterns' in breadth_data:
                flow_velocity = breadth_data['flow_patterns'].get('flow_velocity', 1.0)
                return float(flow_velocity - 1.0)  # Convert to momentum scale
            
            if 'volume_momentum' in breadth_data:
                return float(breadth_data['volume_momentum'].get('momentum_strength', 0.0))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting momentum metric: {e}")
            return 0.0
    
    def _calculate_signal_alignment(self, option_signal: str, underlying_signal: str) -> float:
        """Calculate alignment between option and underlying signals"""
        try:
            # Define signal categories
            bullish_keywords = ['bullish', 'positive', 'expanding', 'thrust', 'strong']
            bearish_keywords = ['bearish', 'negative', 'contracting', 'decline', 'weak']
            
            option_bullish = any(keyword in option_signal.lower() for keyword in bullish_keywords)
            option_bearish = any(keyword in option_signal.lower() for keyword in bearish_keywords)
            
            underlying_bullish = any(keyword in underlying_signal.lower() for keyword in bullish_keywords)
            underlying_bearish = any(keyword in underlying_signal.lower() for keyword in bearish_keywords)
            
            # Perfect alignment
            if (option_bullish and underlying_bullish) or (option_bearish and underlying_bearish):
                return 1.0
            
            # Complete divergence
            elif (option_bullish and underlying_bearish) or (option_bearish and underlying_bullish):
                return 0.0
            
            # Partial alignment (one neutral)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating signal alignment: {e}")
            return 0.5
    
    def _update_regime_history(self, primary_regime: Dict[str, Any], regime_confidence: Dict[str, Any], composite_metrics: Dict[str, Any]):
        """Update historical regime tracking"""
        try:
            regime_type = primary_regime.get('regime_type', 'neutral')
            confidence = regime_confidence.get('overall_confidence', 0.0)
            breadth_score = composite_metrics.get('composite_breadth_score', 0.5)
            
            # Check for regime change
            if self.regime_history['regimes']:
                last_regime = self.regime_history['regimes'][-1]
                if regime_type != last_regime:
                    self.transition_metrics['regime_changes'] += 1
                    self.transition_metrics['current_regime_duration'] = 1
                else:
                    self.transition_metrics['current_regime_duration'] += 1
            
            # Update history
            self.regime_history['regimes'].append(regime_type)
            self.regime_history['confidence_scores'].append(confidence)
            self.regime_history['breadth_scores'].append(breadth_score)
            
            # Trim history
            max_history = self.regime_persistence_window * 10
            for key in ['regimes', 'confidence_scores', 'breadth_scores']:
                if len(self.regime_history[key]) > max_history:
                    self.regime_history[key].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating regime history: {e}")
    
    def _get_default_regime_classification(self) -> Dict[str, Any]:
        """Get default regime classification when data is insufficient"""
        return {
            'primary_regime': {'regime_type': 'neutral', 'regime_strength': 'moderate', 'classification_confidence': 0.0},
            'secondary_characteristics': {'market_structure': 'normal', 'leadership': 'balanced'},
            'regime_confidence': {'overall_confidence': 0.0, 'confidence_level': 'low'},
            'regime_persistence': {'current_duration': 0, 'stability_score': 0.0},
            'composite_metrics': {'composite_breadth_score': 0.5, 'composite_participation': 0.5},
            'breadth_quality': {'overall_quality': 'moderate', 'quality_score': 0.5},
            'momentum_indicators': {'momentum_direction': 'neutral', 'momentum_strength': 0.0},
            'divergence_impact': {'divergence_severity': 'minimal', 'regime_risk': 'low'},
            'regime_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'regime_stability_score': 0.5
        }
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of regime classification system"""
        try:
            return {
                'history_length': len(self.regime_history['regimes']),
                'current_regime': self.regime_history['regimes'][-1] if self.regime_history['regimes'] else 'neutral',
                'regime_distribution': self._calculate_regime_distribution(),
                'transition_metrics': self.transition_metrics.copy(),
                'average_confidence': np.mean(self.regime_history['confidence_scores']) if self.regime_history['confidence_scores'] else 0.0,
                'analysis_config': {
                    'regime_confidence_threshold': self.regime_confidence_threshold,
                    'regime_persistence_window': self.regime_persistence_window
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting regime summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_regime_distribution(self) -> Dict[str, float]:
        """Calculate distribution of regimes in history"""
        try:
            if not self.regime_history['regimes']:
                return {}
            
            regime_counts = {}
            for regime in self.regime_history['regimes']:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            total = len(self.regime_history['regimes'])
            return {regime: count / total for regime, count in regime_counts.items()}
            
        except Exception as e:
            logger.error(f"Error calculating regime distribution: {e}")
            return {}