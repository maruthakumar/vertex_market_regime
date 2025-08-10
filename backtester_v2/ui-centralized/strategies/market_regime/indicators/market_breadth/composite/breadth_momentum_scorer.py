"""
Breadth Momentum Scorer - Comprehensive Breadth Momentum Scoring System
======================================================================

Scores and tracks momentum patterns across all breadth components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BreadthMomentumScorer:
    """Comprehensive breadth momentum scoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Breadth Momentum Scorer"""
        self.momentum_window = config.get('momentum_window', 14)
        self.acceleration_window = config.get('acceleration_window', 5)
        self.momentum_thresholds = config.get('momentum_thresholds', {
            'strong': 0.7,
            'moderate': 0.4,
            'weak': 0.2
        })
        
        # Historical momentum tracking
        self.momentum_history = {
            'option_momentum_scores': [],
            'underlying_momentum_scores': [],
            'composite_momentum_scores': [],
            'momentum_accelerations': [],
            'momentum_directions': [],
            'timestamps': []
        }
        
        # Momentum pattern recognition
        self.momentum_patterns = {
            'momentum_divergences': [],
            'acceleration_events': [],
            'deceleration_events': [],
            'momentum_reversals': []
        }
        
        # Performance tracking
        self.scoring_metrics = {
            'total_scores_calculated': 0,
            'momentum_events_detected': 0,
            'pattern_accuracy': 0.0
        }
        
        logger.info("BreadthMomentumScorer initialized")
    
    def calculate_momentum_scores(self, 
                                option_breadth: Dict[str, Any],
                                underlying_breadth: Dict[str, Any],
                                historical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive momentum scores for breadth analysis
        
        Args:
            option_breadth: Option breadth analysis results
            underlying_breadth: Underlying breadth analysis results
            historical_context: Optional historical data for momentum calculation
            
        Returns:
            Dict with momentum scoring results
        """
        try:
            if not option_breadth or not underlying_breadth:
                return self._get_default_momentum_scores()
            
            # Calculate individual momentum scores
            option_momentum = self._calculate_option_momentum_score(option_breadth)
            underlying_momentum = self._calculate_underlying_momentum_score(underlying_breadth)
            
            # Calculate composite momentum
            composite_momentum = self._calculate_composite_momentum_score(option_momentum, underlying_momentum)
            
            # Analyze momentum acceleration
            momentum_acceleration = self._analyze_momentum_acceleration(composite_momentum)
            
            # Detect momentum patterns
            momentum_patterns = self._detect_momentum_patterns(option_momentum, underlying_momentum, composite_momentum)
            
            # Calculate momentum quality
            momentum_quality = self._assess_momentum_quality(option_momentum, underlying_momentum, momentum_patterns)
            
            # Generate momentum signals
            momentum_signals = self._generate_momentum_signals(composite_momentum, momentum_acceleration, momentum_patterns)
            
            # Calculate momentum persistence
            momentum_persistence = self._calculate_momentum_persistence(composite_momentum)
            
            # Update historical tracking
            self._update_momentum_history(option_momentum, underlying_momentum, composite_momentum, momentum_acceleration)
            
            return {
                'option_momentum': option_momentum,
                'underlying_momentum': underlying_momentum,
                'composite_momentum': composite_momentum,
                'momentum_acceleration': momentum_acceleration,
                'momentum_patterns': momentum_patterns,
                'momentum_quality': momentum_quality,
                'momentum_signals': momentum_signals,
                'momentum_persistence': momentum_persistence,
                'momentum_regime': self._classify_momentum_regime(composite_momentum, momentum_quality)
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum scores: {e}")
            return self._get_default_momentum_scores()
    
    def _calculate_option_momentum_score(self, option_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate momentum score for option breadth"""
        try:
            momentum = {
                'score': 0.0,
                'direction': 'neutral',
                'strength': 'weak',
                'components': {}
            }
            
            score_components = []
            
            # Volume flow momentum
            if 'flow_patterns' in option_breadth:
                flow_velocity = option_breadth['flow_patterns'].get('flow_velocity', 1.0)
                flow_momentum = (flow_velocity - 1.0) * 2  # Convert to [-2, 2] scale
                score_components.append(np.tanh(flow_momentum))  # Normalize to [-1, 1]
                momentum['components']['flow_momentum'] = float(flow_momentum)
            
            # Signal strength momentum
            if 'flow_signals' in option_breadth:
                signal_strength = option_breadth['flow_signals'].get('signal_strength', 0.0)
                signal_momentum = (signal_strength - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(signal_momentum)
                momentum['components']['signal_momentum'] = float(signal_momentum)
            
            # Participation momentum
            if 'flow_metrics' in option_breadth:
                participation = option_breadth['flow_metrics'].get('participation_rate', 0.5)
                participation_momentum = (participation - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(participation_momentum)
                momentum['components']['participation_momentum'] = float(participation_momentum)
            
            # Volume ratio momentum
            if 'volume_metrics' in option_breadth:
                up_volume_ratio = option_breadth['volume_metrics'].get('up_volume_ratio', 0.5)
                volume_momentum = (up_volume_ratio - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(volume_momentum)
                momentum['components']['volume_momentum'] = float(volume_momentum)
            
            # Sector rotation momentum
            if 'sector_rotation' in option_breadth:
                rotation_strength = option_breadth['sector_rotation'].get('rotation_strength', 0.0)
                rotation_momentum = rotation_strength * 2 - 1  # Convert to [-1, 1]
                score_components.append(rotation_momentum)
                momentum['components']['rotation_momentum'] = float(rotation_momentum)
            
            # Calculate composite score
            if score_components:
                momentum['score'] = float(np.mean(score_components))
            
            # Determine direction and strength
            if momentum['score'] > 0.1:
                momentum['direction'] = 'bullish'
            elif momentum['score'] < -0.1:
                momentum['direction'] = 'bearish'
            else:
                momentum['direction'] = 'neutral'
            
            abs_score = abs(momentum['score'])
            if abs_score > self.momentum_thresholds['strong']:
                momentum['strength'] = 'strong'
            elif abs_score > self.momentum_thresholds['moderate']:
                momentum['strength'] = 'moderate'
            elif abs_score > self.momentum_thresholds['weak']:
                momentum['strength'] = 'weak'
            else:
                momentum['strength'] = 'minimal'
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating option momentum score: {e}")
            return {'score': 0.0, 'direction': 'neutral', 'strength': 'weak'}
    
    def _calculate_underlying_momentum_score(self, underlying_breadth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate momentum score for underlying breadth"""
        try:
            momentum = {
                'score': 0.0,
                'direction': 'neutral',
                'strength': 'weak',
                'components': {}
            }
            
            score_components = []
            
            # Advance/decline momentum
            if 'ad_metrics' in underlying_breadth:
                advance_ratio = underlying_breadth['ad_metrics'].get('advance_ratio', 0.5)
                ad_momentum = (advance_ratio - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(ad_momentum)
                momentum['components']['ad_momentum'] = float(ad_momentum)
            
            # Volume flow momentum
            if 'flow_metrics' in underlying_breadth:
                up_volume_ratio = underlying_breadth['flow_metrics'].get('up_volume_ratio', 0.5)
                volume_momentum = (up_volume_ratio - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(volume_momentum)
                momentum['components']['volume_momentum'] = float(volume_momentum)
            
            # New highs/lows momentum
            if 'hl_metrics' in underlying_breadth:
                hl_ratio = underlying_breadth['hl_metrics'].get('hl_ratio_20d', 0.5)
                hl_momentum = (hl_ratio - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(hl_momentum)
                momentum['components']['hl_momentum'] = float(hl_momentum)
            
            # Participation momentum
            if 'participation_metrics' in underlying_breadth:
                participation = underlying_breadth['participation_metrics'].get('overall_participation', 0.5)
                participation_momentum = (participation - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(participation_momentum)
                momentum['components']['participation_momentum'] = float(participation_momentum)
            
            # Breadth signal momentum
            if 'breadth_signals' in underlying_breadth:
                signal_strength = underlying_breadth['breadth_signals'].get('signal_strength', 0.0)
                signal_momentum = (signal_strength - 0.5) * 2  # Convert to [-1, 1]
                score_components.append(signal_momentum)
                momentum['components']['signal_momentum'] = float(signal_momentum)
            
            # Calculate composite score
            if score_components:
                momentum['score'] = float(np.mean(score_components))
            
            # Determine direction and strength
            if momentum['score'] > 0.1:
                momentum['direction'] = 'bullish'
            elif momentum['score'] < -0.1:
                momentum['direction'] = 'bearish'
            else:
                momentum['direction'] = 'neutral'
            
            abs_score = abs(momentum['score'])
            if abs_score > self.momentum_thresholds['strong']:
                momentum['strength'] = 'strong'
            elif abs_score > self.momentum_thresholds['moderate']:
                momentum['strength'] = 'moderate'
            elif abs_score > self.momentum_thresholds['weak']:
                momentum['strength'] = 'weak'
            else:
                momentum['strength'] = 'minimal'
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating underlying momentum score: {e}")
            return {'score': 0.0, 'direction': 'neutral', 'strength': 'weak'}
    
    def _calculate_composite_momentum_score(self, option_momentum: Dict[str, Any], underlying_momentum: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite momentum score"""
        try:
            composite = {
                'score': 0.0,
                'direction': 'neutral',
                'strength': 'weak',
                'alignment': 'neutral',
                'weighted_score': 0.0
            }
            
            option_score = option_momentum.get('score', 0.0)
            underlying_score = underlying_momentum.get('score', 0.0)
            
            # Simple average (can be weighted based on market conditions)
            composite['score'] = float((option_score + underlying_score) / 2)
            
            # Weighted score (option breadth weighted higher for forward-looking nature)
            composite['weighted_score'] = float((option_score * 0.6) + (underlying_score * 0.4))
            
            # Determine alignment
            option_direction = option_momentum.get('direction', 'neutral')
            underlying_direction = underlying_momentum.get('direction', 'neutral')
            
            if option_direction == underlying_direction:
                composite['alignment'] = 'aligned'
            elif (option_direction == 'neutral' or underlying_direction == 'neutral'):
                composite['alignment'] = 'partial'
            else:
                composite['alignment'] = 'divergent'
            
            # Overall direction (use weighted score)
            if composite['weighted_score'] > 0.1:
                composite['direction'] = 'bullish'
            elif composite['weighted_score'] < -0.1:
                composite['direction'] = 'bearish'
            else:
                composite['direction'] = 'neutral'
            
            # Overall strength
            abs_weighted_score = abs(composite['weighted_score'])
            if abs_weighted_score > self.momentum_thresholds['strong']:
                composite['strength'] = 'strong'
            elif abs_weighted_score > self.momentum_thresholds['moderate']:
                composite['strength'] = 'moderate'
            elif abs_weighted_score > self.momentum_thresholds['weak']:
                composite['strength'] = 'weak'
            else:
                composite['strength'] = 'minimal'
            
            return composite
            
        except Exception as e:
            logger.error(f"Error calculating composite momentum score: {e}")
            return {'score': 0.0, 'direction': 'neutral', 'strength': 'weak'}
    
    def _analyze_momentum_acceleration(self, composite_momentum: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum acceleration patterns"""
        try:
            acceleration = {
                'acceleration_score': 0.0,
                'acceleration_direction': 'neutral',
                'velocity_change': 0.0,
                'acceleration_strength': 'minimal'
            }
            
            current_score = composite_momentum.get('weighted_score', 0.0)
            
            # Calculate acceleration if we have historical data
            if len(self.momentum_history['composite_momentum_scores']) >= 2:
                recent_scores = self.momentum_history['composite_momentum_scores'][-self.acceleration_window:]
                recent_scores.append(current_score)
                
                if len(recent_scores) >= 3:
                    # Calculate velocity (rate of change)
                    velocities = []
                    for i in range(1, len(recent_scores)):
                        velocity = recent_scores[i] - recent_scores[i-1]
                        velocities.append(velocity)
                    
                    # Calculate acceleration (rate of velocity change)
                    if len(velocities) >= 2:
                        recent_velocity = np.mean(velocities[-2:])
                        previous_velocity = np.mean(velocities[:-2]) if len(velocities) > 2 else velocities[0]
                        
                        acceleration_score = recent_velocity - previous_velocity
                        acceleration['acceleration_score'] = float(acceleration_score)
                        acceleration['velocity_change'] = float(recent_velocity)
                        
                        # Determine acceleration direction
                        if acceleration_score > 0.05:
                            acceleration['acceleration_direction'] = 'accelerating'
                        elif acceleration_score < -0.05:
                            acceleration['acceleration_direction'] = 'decelerating'
                        else:
                            acceleration['acceleration_direction'] = 'stable'
                        
                        # Determine acceleration strength
                        abs_acceleration = abs(acceleration_score)
                        if abs_acceleration > 0.2:
                            acceleration['acceleration_strength'] = 'strong'
                        elif abs_acceleration > 0.1:
                            acceleration['acceleration_strength'] = 'moderate'
                        elif abs_acceleration > 0.05:
                            acceleration['acceleration_strength'] = 'weak'
                        else:
                            acceleration['acceleration_strength'] = 'minimal'
            
            return acceleration
            
        except Exception as e:
            logger.error(f"Error analyzing momentum acceleration: {e}")
            return {'acceleration_score': 0.0, 'acceleration_direction': 'neutral'}
    
    def _detect_momentum_patterns(self, option_momentum: Dict[str, Any], underlying_momentum: Dict[str, Any], composite_momentum: Dict[str, Any]) -> Dict[str, Any]:
        """Detect momentum patterns and anomalies"""
        try:
            patterns = {
                'divergence_patterns': [],
                'convergence_patterns': [],
                'momentum_reversals': [],
                'pattern_strength': 'weak'
            }
            
            option_score = option_momentum.get('score', 0.0)
            underlying_score = underlying_momentum.get('score', 0.0)
            composite_score = composite_momentum.get('weighted_score', 0.0)
            
            # Divergence detection
            momentum_divergence = abs(option_score - underlying_score)
            if momentum_divergence > 0.3:
                divergence_type = 'option_leading' if option_score > underlying_score else 'underlying_leading'
                patterns['divergence_patterns'].append({
                    'type': divergence_type,
                    'magnitude': float(momentum_divergence),
                    'option_score': float(option_score),
                    'underlying_score': float(underlying_score)
                })
            
            # Convergence detection (both moving in same direction)
            if abs(momentum_divergence) < 0.15 and abs(composite_score) > 0.2:
                patterns['convergence_patterns'].append({
                    'type': 'synchronized_momentum',
                    'direction': composite_momentum.get('direction', 'neutral'),
                    'strength': composite_momentum.get('strength', 'weak')
                })
            
            # Momentum reversal detection
            if len(self.momentum_history['composite_momentum_scores']) >= 5:
                recent_scores = self.momentum_history['composite_momentum_scores'][-5:] + [composite_score]
                
                # Check for momentum reversal patterns
                if len(recent_scores) >= 6:
                    # Look for sign changes
                    sign_changes = 0
                    for i in range(1, len(recent_scores)):
                        if np.sign(recent_scores[i]) != np.sign(recent_scores[i-1]) and abs(recent_scores[i]) > 0.1:
                            sign_changes += 1
                    
                    if sign_changes >= 2:
                        patterns['momentum_reversals'].append({
                            'type': 'multiple_reversals',
                            'reversal_count': sign_changes,
                            'current_direction': composite_momentum.get('direction', 'neutral')
                        })
                    
                    # Check for momentum exhaustion
                    recent_momentum = recent_scores[-3:]
                    if all(abs(score) > 0.5 for score in recent_momentum) and abs(composite_score) < 0.3:
                        patterns['momentum_reversals'].append({
                            'type': 'momentum_exhaustion',
                            'previous_strength': 'strong',
                            'current_strength': composite_momentum.get('strength', 'weak')
                        })
            
            # Determine overall pattern strength
            total_patterns = (len(patterns['divergence_patterns']) + 
                            len(patterns['convergence_patterns']) + 
                            len(patterns['momentum_reversals']))
            
            if total_patterns >= 3:
                patterns['pattern_strength'] = 'strong'
            elif total_patterns >= 2:
                patterns['pattern_strength'] = 'moderate'
            elif total_patterns >= 1:
                patterns['pattern_strength'] = 'weak'
            else:
                patterns['pattern_strength'] = 'minimal'
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting momentum patterns: {e}")
            return {'divergence_patterns': [], 'convergence_patterns': [], 'momentum_reversals': []}
    
    def _assess_momentum_quality(self, option_momentum: Dict[str, Any], underlying_momentum: Dict[str, Any], momentum_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of momentum signals"""
        try:
            quality = {
                'overall_quality': 'moderate',
                'quality_score': 0.5,
                'quality_factors': {}
            }
            
            quality_factors = []
            
            # Consistency factor (alignment between option and underlying)
            option_direction = option_momentum.get('direction', 'neutral')
            underlying_direction = underlying_momentum.get('direction', 'neutral')
            
            if option_direction == underlying_direction and option_direction != 'neutral':
                consistency_score = 1.0
            elif option_direction == 'neutral' or underlying_direction == 'neutral':
                consistency_score = 0.6
            else:
                consistency_score = 0.2
            
            quality_factors.append(consistency_score)
            quality['quality_factors']['consistency'] = float(consistency_score)
            
            # Strength factor
            option_strength = option_momentum.get('strength', 'weak')
            underlying_strength = underlying_momentum.get('strength', 'weak')
            
            strength_mapping = {'strong': 1.0, 'moderate': 0.7, 'weak': 0.4, 'minimal': 0.2}
            avg_strength = (strength_mapping.get(option_strength, 0.4) + 
                          strength_mapping.get(underlying_strength, 0.4)) / 2
            
            quality_factors.append(avg_strength)
            quality['quality_factors']['strength'] = float(avg_strength)
            
            # Pattern stability (fewer reversals = higher quality)
            reversal_count = len(momentum_patterns.get('momentum_reversals', []))
            stability_score = max(0.2, 1.0 - reversal_count * 0.2)
            
            quality_factors.append(stability_score)
            quality['quality_factors']['stability'] = float(stability_score)
            
            # Convergence factor (synchronized momentum = higher quality)
            convergence_count = len(momentum_patterns.get('convergence_patterns', []))
            divergence_count = len(momentum_patterns.get('divergence_patterns', []))
            
            if convergence_count > divergence_count:
                synchronization_score = 0.8
            elif convergence_count == divergence_count:
                synchronization_score = 0.5
            else:
                synchronization_score = 0.3
            
            quality_factors.append(synchronization_score)
            quality['quality_factors']['synchronization'] = float(synchronization_score)
            
            # Calculate overall quality score
            quality['quality_score'] = float(np.mean(quality_factors))
            
            # Classify quality
            if quality['quality_score'] > 0.8:
                quality['overall_quality'] = 'excellent'
            elif quality['quality_score'] > 0.6:
                quality['overall_quality'] = 'good'
            elif quality['quality_score'] > 0.4:
                quality['overall_quality'] = 'moderate'
            elif quality['quality_score'] > 0.2:
                quality['overall_quality'] = 'poor'
            else:
                quality['overall_quality'] = 'very_poor'
            
            return quality
            
        except Exception as e:
            logger.error(f"Error assessing momentum quality: {e}")
            return {'overall_quality': 'moderate', 'quality_score': 0.5}
    
    def _generate_momentum_signals(self, composite_momentum: Dict[str, Any], momentum_acceleration: Dict[str, Any], momentum_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable momentum signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'momentum_signals': [],
                'trading_implications': []
            }
            
            momentum_direction = composite_momentum.get('direction', 'neutral')
            momentum_strength = composite_momentum.get('strength', 'weak')
            acceleration_direction = momentum_acceleration.get('acceleration_direction', 'neutral')
            
            # Primary signal determination
            if momentum_direction == 'bullish' and momentum_strength in ['strong', 'moderate']:
                signals['primary_signal'] = 'bullish_momentum'
                signals['trading_implications'].append('broad_market_strength')
            elif momentum_direction == 'bearish' and momentum_strength in ['strong', 'moderate']:
                signals['primary_signal'] = 'bearish_momentum'
                signals['trading_implications'].append('broad_market_weakness')
            elif momentum_direction != 'neutral' and momentum_strength == 'weak':
                signals['primary_signal'] = 'weak_momentum'
                signals['trading_implications'].append('developing_trend')
            else:
                signals['primary_signal'] = 'neutral_momentum'
                signals['trading_implications'].append('range_bound_market')
            
            # Signal strength
            strength_mapping = {'strong': 0.8, 'moderate': 0.6, 'weak': 0.4, 'minimal': 0.2}
            signals['signal_strength'] = float(strength_mapping.get(momentum_strength, 0.2))
            
            # Acceleration signals
            if acceleration_direction == 'accelerating':
                signals['momentum_signals'].append('momentum_acceleration')
                signals['trading_implications'].append('strengthening_trend')
                signals['signal_strength'] += 0.1
            elif acceleration_direction == 'decelerating':
                signals['momentum_signals'].append('momentum_deceleration')
                signals['trading_implications'].append('weakening_trend')
            
            # Pattern-based signals
            if momentum_patterns.get('divergence_patterns'):
                signals['momentum_signals'].append('momentum_divergence')
                signals['trading_implications'].append('leadership_rotation')
            
            if momentum_patterns.get('convergence_patterns'):
                signals['momentum_signals'].append('momentum_convergence')
                signals['trading_implications'].append('broad_market_agreement')
            
            if momentum_patterns.get('momentum_reversals'):
                signals['momentum_signals'].append('momentum_reversal')
                signals['trading_implications'].append('trend_change_risk')
            
            # Alignment signals
            alignment = composite_momentum.get('alignment', 'neutral')
            if alignment == 'aligned':
                signals['momentum_signals'].append('momentum_alignment')
                signals['trading_implications'].append('consistent_market_direction')
            elif alignment == 'divergent':
                signals['momentum_signals'].append('momentum_misalignment')
                signals['trading_implications'].append('conflicting_market_signals')
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_momentum_persistence(self, composite_momentum: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate momentum persistence metrics"""
        try:
            persistence = {
                'persistence_score': 0.0,
                'trend_duration': 0,
                'momentum_stability': 'unstable',
                'reversal_risk': 'medium'
            }
            
            current_direction = composite_momentum.get('direction', 'neutral')
            
            if len(self.momentum_history['momentum_directions']) >= 5:
                recent_directions = self.momentum_history['momentum_directions'][-5:] + [current_direction]
                
                # Calculate persistence
                direction_changes = 0
                for i in range(1, len(recent_directions)):
                    if recent_directions[i] != recent_directions[i-1] and recent_directions[i] != 'neutral':
                        direction_changes += 1
                
                persistence_score = max(0.0, 1.0 - direction_changes * 0.2)
                persistence['persistence_score'] = float(persistence_score)
                
                # Calculate trend duration
                trend_duration = 1
                for direction in reversed(recent_directions[:-1]):
                    if direction == current_direction:
                        trend_duration += 1
                    else:
                        break
                
                persistence['trend_duration'] = trend_duration
                
                # Momentum stability
                if persistence_score > 0.8:
                    persistence['momentum_stability'] = 'very_stable'
                elif persistence_score > 0.6:
                    persistence['momentum_stability'] = 'stable'
                elif persistence_score > 0.4:
                    persistence['momentum_stability'] = 'moderately_stable'
                else:
                    persistence['momentum_stability'] = 'unstable'
                
                # Reversal risk
                if trend_duration >= 8:
                    persistence['reversal_risk'] = 'high'  # Long trends may reverse
                elif trend_duration >= 5:
                    persistence['reversal_risk'] = 'medium'
                elif trend_duration >= 2:
                    persistence['reversal_risk'] = 'low'
                else:
                    persistence['reversal_risk'] = 'very_low'
            
            return persistence
            
        except Exception as e:
            logger.error(f"Error calculating momentum persistence: {e}")
            return {'persistence_score': 0.0, 'trend_duration': 0}
    
    def _classify_momentum_regime(self, composite_momentum: Dict[str, Any], momentum_quality: Dict[str, Any]) -> Dict[str, Any]:
        """Classify momentum regime"""
        try:
            regime = {
                'momentum_regime': 'neutral',
                'regime_confidence': 0.0,
                'regime_characteristics': []
            }
            
            momentum_score = composite_momentum.get('weighted_score', 0.0)
            momentum_strength = composite_momentum.get('strength', 'weak')
            quality_score = momentum_quality.get('quality_score', 0.5)
            alignment = composite_momentum.get('alignment', 'neutral')
            
            # Regime classification
            if abs(momentum_score) > 0.6 and momentum_strength == 'strong' and quality_score > 0.7:
                if momentum_score > 0:
                    regime['momentum_regime'] = 'strong_bullish_momentum'
                else:
                    regime['momentum_regime'] = 'strong_bearish_momentum'
                regime['regime_confidence'] = 0.9
                regime['regime_characteristics'].extend(['high_momentum', 'high_quality', 'strong_signals'])
                
            elif abs(momentum_score) > 0.4 and momentum_strength in ['strong', 'moderate']:
                if momentum_score > 0:
                    regime['momentum_regime'] = 'moderate_bullish_momentum'
                else:
                    regime['momentum_regime'] = 'moderate_bearish_momentum'
                regime['regime_confidence'] = 0.7
                regime['regime_characteristics'].extend(['moderate_momentum', 'directional_bias'])
                
            elif abs(momentum_score) > 0.2:
                if momentum_score > 0:
                    regime['momentum_regime'] = 'weak_bullish_momentum'
                else:
                    regime['momentum_regime'] = 'weak_bearish_momentum'
                regime['regime_confidence'] = 0.5
                regime['regime_characteristics'].append('weak_momentum')
                
            else:
                regime['momentum_regime'] = 'neutral_momentum'
                regime['regime_confidence'] = 0.3
                regime['regime_characteristics'].append('range_bound')
            
            # Add quality characteristics
            if quality_score > 0.8:
                regime['regime_characteristics'].append('high_quality_momentum')
            elif quality_score < 0.3:
                regime['regime_characteristics'].append('low_quality_momentum')
            
            # Add alignment characteristics
            if alignment == 'aligned':
                regime['regime_characteristics'].append('synchronized_breadth')
            elif alignment == 'divergent':
                regime['regime_characteristics'].append('divergent_breadth')
            
            return regime
            
        except Exception as e:
            logger.error(f"Error classifying momentum regime: {e}")
            return {'momentum_regime': 'neutral', 'regime_confidence': 0.0}
    
    def _update_momentum_history(self, option_momentum: Dict[str, Any], underlying_momentum: Dict[str, Any], composite_momentum: Dict[str, Any], momentum_acceleration: Dict[str, Any]):
        """Update historical momentum tracking"""
        try:
            # Update scores
            self.momentum_history['option_momentum_scores'].append(option_momentum.get('score', 0.0))
            self.momentum_history['underlying_momentum_scores'].append(underlying_momentum.get('score', 0.0))
            self.momentum_history['composite_momentum_scores'].append(composite_momentum.get('weighted_score', 0.0))
            self.momentum_history['momentum_accelerations'].append(momentum_acceleration.get('acceleration_score', 0.0))
            self.momentum_history['momentum_directions'].append(composite_momentum.get('direction', 'neutral'))
            
            # Trim history
            max_history = self.momentum_window * 3
            for key in self.momentum_history.keys():
                if key != 'timestamps' and len(self.momentum_history[key]) > max_history:
                    self.momentum_history[key].pop(0)
            
            # Update performance metrics
            self.scoring_metrics['total_scores_calculated'] += 1
            
        except Exception as e:
            logger.error(f"Error updating momentum history: {e}")
    
    def _get_default_momentum_scores(self) -> Dict[str, Any]:
        """Get default momentum scores when data is insufficient"""
        return {
            'option_momentum': {'score': 0.0, 'direction': 'neutral', 'strength': 'weak'},
            'underlying_momentum': {'score': 0.0, 'direction': 'neutral', 'strength': 'weak'},
            'composite_momentum': {'score': 0.0, 'direction': 'neutral', 'strength': 'weak', 'alignment': 'neutral'},
            'momentum_acceleration': {'acceleration_score': 0.0, 'acceleration_direction': 'neutral'},
            'momentum_patterns': {'divergence_patterns': [], 'convergence_patterns': [], 'momentum_reversals': []},
            'momentum_quality': {'overall_quality': 'moderate', 'quality_score': 0.5},
            'momentum_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'momentum_persistence': {'persistence_score': 0.0, 'trend_duration': 0},
            'momentum_regime': {'momentum_regime': 'neutral', 'regime_confidence': 0.0}
        }
    
    def get_momentum_summary(self) -> Dict[str, Any]:
        """Get summary of momentum scoring system"""
        try:
            return {
                'history_length': len(self.momentum_history['composite_momentum_scores']),
                'current_momentum_direction': self.momentum_history['momentum_directions'][-1] if self.momentum_history['momentum_directions'] else 'neutral',
                'average_momentum_score': np.mean(self.momentum_history['composite_momentum_scores']) if self.momentum_history['composite_momentum_scores'] else 0.0,
                'momentum_volatility': np.std(self.momentum_history['composite_momentum_scores']) if len(self.momentum_history['composite_momentum_scores']) > 1 else 0.0,
                'scoring_metrics': self.scoring_metrics.copy(),
                'pattern_summary': {
                    'total_divergences': len(self.momentum_patterns['momentum_divergences']),
                    'total_reversals': len(self.momentum_patterns['momentum_reversals']),
                    'acceleration_events': len(self.momentum_patterns['acceleration_events'])
                },
                'analysis_config': {
                    'momentum_window': self.momentum_window,
                    'acceleration_window': self.acceleration_window,
                    'momentum_thresholds': self.momentum_thresholds
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting momentum summary: {e}")
            return {'status': 'error', 'error': str(e)}