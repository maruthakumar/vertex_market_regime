"""
Regime Classifier - Market Regime Classification using Technical Indicators
=========================================================================

Classifies market regimes based on comprehensive technical analysis
from both option and underlying indicators.

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


class RegimeClassifier:
    """
    Classifies market regimes using technical indicators
    
    Features:
    - Multi-indicator regime classification
    - Regime transition detection
    - Regime persistence analysis
    - Confidence scoring for regime classification
    - Historical regime tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Regime Classifier"""
        # Regime thresholds
        self.trend_threshold = config.get('trend_threshold', 0.6)
        self.volatility_threshold = config.get('volatility_threshold', 0.7)
        self.momentum_threshold = config.get('momentum_threshold', 0.5)
        
        # Regime persistence
        self.min_regime_duration = config.get('min_regime_duration', 5)
        self.transition_smoothing = config.get('transition_smoothing', 3)
        
        # Regime definitions
        self.regime_definitions = config.get('regime_definitions', {
            'strong_trending': {'trend': 0.8, 'momentum': 0.7, 'volatility': 'any'},
            'trending': {'trend': 0.6, 'momentum': 0.5, 'volatility': 'any'},
            'volatile_trending': {'trend': 0.5, 'momentum': 0.4, 'volatility': 0.7},
            'ranging': {'trend': 0.3, 'momentum': 0.3, 'volatility': 0.5},
            'volatile_ranging': {'trend': 0.3, 'momentum': 0.3, 'volatility': 0.7},
            'quiet': {'trend': 0.2, 'momentum': 0.2, 'volatility': 0.3}
        })
        
        # Advanced features
        self.enable_ml_classification = config.get('enable_ml_classification', False)
        self.enable_regime_forecasting = config.get('enable_regime_forecasting', True)
        
        # History tracking
        self.regime_history = {
            'regimes': [],
            'transitions': [],
            'durations': [],
            'accuracy': []
        }
        
        logger.info("RegimeClassifier initialized")
    
    def classify_regime(self,
                       technical_indicators: Dict[str, Any],
                       fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify market regime based on technical indicators
        
        Args:
            technical_indicators: All technical indicator results
            fusion_results: Results from indicator fusion
            
        Returns:
            Dict with regime classification and analysis
        """
        try:
            results = {
                'current_regime': None,
                'regime_confidence': 0.0,
                'regime_characteristics': {},
                'sub_regimes': [],
                'transition_probability': {},
                'regime_forecast': {},
                'supporting_indicators': []
            }
            
            # Extract regime components
            components = self._extract_regime_components(
                technical_indicators, fusion_results
            )
            
            # Calculate regime scores
            regime_scores = self._calculate_regime_scores(components)
            
            # Determine current regime
            results['current_regime'] = self._determine_regime(regime_scores)
            
            # Calculate confidence
            results['regime_confidence'] = self._calculate_regime_confidence(
                regime_scores, components
            )
            
            # Analyze regime characteristics
            results['regime_characteristics'] = self._analyze_regime_characteristics(
                components
            )
            
            # Identify sub-regimes
            results['sub_regimes'] = self._identify_sub_regimes(
                components, results['current_regime']
            )
            
            # Calculate transition probabilities
            results['transition_probability'] = self._calculate_transition_probabilities(
                results['current_regime'], components
            )
            
            # Forecast regime changes
            if self.enable_regime_forecasting:
                results['regime_forecast'] = self._forecast_regime_change(
                    results, components
                )
            
            # Identify supporting indicators
            results['supporting_indicators'] = self._identify_supporting_indicators(
                technical_indicators, results['current_regime']
            )
            
            # Update history
            self._update_regime_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return self._get_default_results()
    
    def _extract_regime_components(self,
                                 technical_indicators: Dict[str, Any],
                                 fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regime classification components"""
        try:
            components = {
                'trend_strength': 0.0,
                'momentum': 0.0,
                'volatility': 0.0,
                'volume_flow': 0.0,
                'option_sentiment': 0.0,
                'divergences': 0
            }
            
            # Trend strength from underlying indicators
            if 'underlying' in technical_indicators:
                if 'trend_strength' in technical_indicators['underlying']:
                    adx = technical_indicators['underlying']['trend_strength']['adx_analysis'].get('adx', 0)
                    components['trend_strength'] = min(adx / 50, 1.0)  # Normalize ADX
            
            # Momentum from MACD
            if 'option' in technical_indicators and 'macd' in technical_indicators['option']:
                opt_momentum = technical_indicators['option']['macd'].get('momentum', 'neutral')
                components['momentum'] += self._momentum_to_score(opt_momentum) * 0.5
            
            if 'underlying' in technical_indicators and 'macd' in technical_indicators['underlying']:
                und_momentum = technical_indicators['underlying']['macd'].get('momentum', 'neutral')
                components['momentum'] += self._momentum_to_score(und_momentum) * 0.5
            
            # Volatility from Bollinger Bands
            volatility_scores = []
            if 'option' in technical_indicators and 'bollinger' in technical_indicators['option']:
                vol_state = technical_indicators['option']['bollinger'].get('volatility_state', 'normal')
                volatility_scores.append(self._volatility_to_score(vol_state))
            
            if 'underlying' in technical_indicators and 'bollinger' in technical_indicators['underlying']:
                vol_state = technical_indicators['underlying']['bollinger'].get('volatility_state', 'normal')
                volatility_scores.append(self._volatility_to_score(vol_state))
            
            if volatility_scores:
                components['volatility'] = np.mean(volatility_scores)
            
            # Volume flow
            if 'option' in technical_indicators and 'volume_flow' in technical_indicators['option']:
                flow_regime = technical_indicators['option']['volume_flow'].get('regime', 'balanced')
                components['volume_flow'] = self._flow_to_score(flow_regime)
            
            # Option sentiment from RSI
            if 'option' in technical_indicators and 'rsi' in technical_indicators['option']:
                rsi_regime = technical_indicators['option']['rsi'].get('regime', 'neutral')
                components['option_sentiment'] = self._sentiment_to_score(rsi_regime)
            
            # Divergences from fusion
            if fusion_results and 'divergences' in fusion_results:
                components['divergences'] = len(fusion_results['divergences'])
            
            return components
            
        except Exception as e:
            logger.error(f"Error extracting components: {e}")
            return {}
    
    def _calculate_regime_scores(self, components: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scores for each regime type"""
        try:
            scores = {}
            
            for regime_name, criteria in self.regime_definitions.items():
                score = 0.0
                matches = 0
                total_criteria = 0
                
                # Check trend criterion
                if 'trend' in criteria and criteria['trend'] != 'any':
                    total_criteria += 1
                    if components['trend_strength'] >= criteria['trend']:
                        score += 1
                        matches += 1
                
                # Check momentum criterion
                if 'momentum' in criteria and criteria['momentum'] != 'any':
                    total_criteria += 1
                    if abs(components['momentum']) >= criteria['momentum']:
                        score += 1
                        matches += 1
                
                # Check volatility criterion
                if 'volatility' in criteria and criteria['volatility'] != 'any':
                    total_criteria += 1
                    if components['volatility'] >= criteria['volatility']:
                        score += 1
                        matches += 1
                
                # Calculate final score
                if total_criteria > 0:
                    scores[regime_name] = score / total_criteria
                else:
                    scores[regime_name] = 0.0
                
                # Bonus for specific regime conditions
                if regime_name == 'strong_trending' and components['divergences'] == 0:
                    scores[regime_name] *= 1.2
                elif regime_name == 'volatile_ranging' and components['divergences'] > 1:
                    scores[regime_name] *= 1.1
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating regime scores: {e}")
            return {}
    
    def _determine_regime(self, regime_scores: Dict[str, float]) -> str:
        """Determine the most likely regime"""
        try:
            if not regime_scores:
                return 'undefined'
            
            # Find regime with highest score
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            
            # Check if score is sufficient
            if best_regime[1] < 0.5:
                return 'transitional'
            
            return best_regime[0]
            
        except Exception as e:
            logger.error(f"Error determining regime: {e}")
            return 'undefined'
    
    def _calculate_regime_confidence(self,
                                   regime_scores: Dict[str, float],
                                   components: Dict[str, Any]) -> float:
        """Calculate confidence in regime classification"""
        try:
            if not regime_scores:
                return 0.0
            
            # Get top two scores
            sorted_scores = sorted(regime_scores.values(), reverse=True)
            
            if len(sorted_scores) < 2:
                return sorted_scores[0] if sorted_scores else 0.0
            
            # Base confidence on score separation
            best_score = sorted_scores[0]
            second_best = sorted_scores[1]
            
            separation = best_score - second_best
            confidence = best_score * (1 + separation)
            
            # Adjust for divergences
            if components.get('divergences', 0) > 0:
                confidence *= 0.9
            
            # Adjust for clear signals
            if components.get('trend_strength', 0) > 0.8:
                confidence *= 1.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _analyze_regime_characteristics(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed regime characteristics"""
        try:
            characteristics = {
                'primary_driver': None,
                'strength_profile': {},
                'stability': 'stable',
                'complexity': 'simple'
            }
            
            # Identify primary driver
            drivers = {
                'trend': components.get('trend_strength', 0),
                'momentum': abs(components.get('momentum', 0)),
                'volatility': components.get('volatility', 0),
                'volume': components.get('volume_flow', 0)
            }
            
            characteristics['primary_driver'] = max(drivers.items(), key=lambda x: x[1])[0]
            
            # Create strength profile
            characteristics['strength_profile'] = {
                'trend': 'strong' if components.get('trend_strength', 0) > 0.7 else 'weak',
                'momentum': 'strong' if abs(components.get('momentum', 0)) > 0.7 else 'weak',
                'volatility': 'high' if components.get('volatility', 0) > 0.7 else 'low'
            }
            
            # Assess stability
            if components.get('divergences', 0) > 1:
                characteristics['stability'] = 'unstable'
            elif components.get('volatility', 0) > 0.8:
                characteristics['stability'] = 'volatile'
            
            # Assess complexity
            active_components = sum(1 for v in drivers.values() if v > 0.5)
            if active_components >= 3:
                characteristics['complexity'] = 'complex'
            elif active_components == 2:
                characteristics['complexity'] = 'moderate'
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing characteristics: {e}")
            return {}
    
    def _identify_sub_regimes(self, components: Dict[str, Any], main_regime: str) -> List[str]:
        """Identify sub-regimes within the main regime"""
        try:
            sub_regimes = []
            
            # Trending sub-regimes
            if 'trending' in main_regime:
                if components.get('momentum', 0) > 0.7:
                    sub_regimes.append('accelerating_trend')
                elif components.get('momentum', 0) < 0.3:
                    sub_regimes.append('weakening_trend')
                
                if components.get('volume_flow', 0) > 0.7:
                    sub_regimes.append('volume_confirmed_trend')
            
            # Ranging sub-regimes
            elif 'ranging' in main_regime:
                if components.get('volatility', 0) < 0.3:
                    sub_regimes.append('tight_range')
                elif components.get('volatility', 0) > 0.7:
                    sub_regimes.append('wide_range')
                
                if abs(components.get('momentum', 0)) < 0.2:
                    sub_regimes.append('equilibrium')
            
            # Volatility sub-regimes
            if components.get('volatility', 0) > 0.8:
                if components.get('divergences', 0) > 0:
                    sub_regimes.append('divergent_volatility')
                else:
                    sub_regimes.append('directional_volatility')
            
            return sub_regimes
            
        except Exception as e:
            logger.error(f"Error identifying sub-regimes: {e}")
            return []
    
    def _calculate_transition_probabilities(self,
                                          current_regime: str,
                                          components: Dict[str, Any]) -> Dict[str, float]:
        """Calculate probabilities of transitioning to other regimes"""
        try:
            probabilities = {}
            
            # Base transition matrix (simplified)
            if 'trending' in current_regime:
                probabilities = {
                    'trending': 0.6,
                    'volatile_trending': 0.2,
                    'ranging': 0.15,
                    'volatile_ranging': 0.05
                }
            elif 'ranging' in current_regime:
                probabilities = {
                    'ranging': 0.5,
                    'trending': 0.25,
                    'volatile_ranging': 0.15,
                    'volatile_trending': 0.1
                }
            else:
                # Equal probabilities for undefined regimes
                probabilities = {
                    'trending': 0.25,
                    'ranging': 0.25,
                    'volatile_trending': 0.25,
                    'volatile_ranging': 0.25
                }
            
            # Adjust based on current components
            if components.get('momentum', 0) > 0.7:
                probabilities['trending'] *= 1.3
            elif components.get('momentum', 0) < 0.3:
                probabilities['ranging'] *= 1.3
            
            if components.get('volatility', 0) > 0.7:
                for regime in probabilities:
                    if 'volatile' in regime:
                        probabilities[regime] *= 1.2
            
            # Normalize probabilities
            total = sum(probabilities.values())
            if total > 0:
                probabilities = {k: v/total for k, v in probabilities.items()}
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating transition probabilities: {e}")
            return {}
    
    def _forecast_regime_change(self,
                              current_results: Dict[str, Any],
                              components: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast potential regime changes"""
        try:
            forecast = {
                'change_probability': 0.0,
                'likely_next_regime': None,
                'change_timeframe': 'unknown',
                'warning_signals': []
            }
            
            # Check regime duration
            current_duration = self._get_current_regime_duration()
            
            # Base change probability on duration
            if current_duration > 20:
                forecast['change_probability'] = 0.3
            elif current_duration > 10:
                forecast['change_probability'] = 0.2
            else:
                forecast['change_probability'] = 0.1
            
            # Adjust for instability signals
            if components.get('divergences', 0) > 1:
                forecast['change_probability'] += 0.2
                forecast['warning_signals'].append('multiple_divergences')
            
            # Check momentum exhaustion
            if abs(components.get('momentum', 0)) > 0.8:
                forecast['change_probability'] += 0.1
                forecast['warning_signals'].append('momentum_exhaustion')
            
            # Extreme volatility
            if components.get('volatility', 0) > 0.9:
                forecast['change_probability'] += 0.15
                forecast['warning_signals'].append('extreme_volatility')
            
            # Determine likely next regime
            transition_probs = current_results.get('transition_probability', {})
            if transition_probs:
                likely_regime = max(transition_probs.items(), key=lambda x: x[1])
                if likely_regime[0] != current_results['current_regime']:
                    forecast['likely_next_regime'] = likely_regime[0]
            
            # Estimate timeframe
            if forecast['change_probability'] > 0.6:
                forecast['change_timeframe'] = 'imminent'
            elif forecast['change_probability'] > 0.4:
                forecast['change_timeframe'] = 'near_term'
            elif forecast['change_probability'] > 0.2:
                forecast['change_timeframe'] = 'medium_term'
            else:
                forecast['change_timeframe'] = 'stable'
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting regime change: {e}")
            return {}
    
    def _identify_supporting_indicators(self,
                                      technical_indicators: Dict[str, Any],
                                      current_regime: str) -> List[str]:
        """Identify indicators supporting the regime classification"""
        try:
            supporting = []
            
            # Check option indicators
            if 'option' in technical_indicators:
                # RSI
                if 'rsi' in technical_indicators['option']:
                    rsi_regime = technical_indicators['option']['rsi'].get('regime', '')
                    if self._regime_matches(rsi_regime, current_regime):
                        supporting.append('option_rsi')
                
                # MACD
                if 'macd' in technical_indicators['option']:
                    momentum = technical_indicators['option']['macd'].get('momentum', '')
                    if self._momentum_supports_regime(momentum, current_regime):
                        supporting.append('option_macd')
                
                # Volume Flow
                if 'volume_flow' in technical_indicators['option']:
                    flow_regime = technical_indicators['option']['volume_flow'].get('regime', '')
                    if self._flow_supports_regime(flow_regime, current_regime):
                        supporting.append('option_volume_flow')
            
            # Check underlying indicators
            if 'underlying' in technical_indicators:
                # Trend Strength
                if 'trend_strength' in technical_indicators['underlying']:
                    trend_regime = technical_indicators['underlying']['trend_strength'].get('trend_regime', '')
                    if self._regime_matches(trend_regime, current_regime):
                        supporting.append('underlying_trend_strength')
                
                # Bollinger
                if 'bollinger' in technical_indicators['underlying']:
                    vol_state = technical_indicators['underlying']['bollinger'].get('volatility_state', '')
                    if self._volatility_supports_regime(vol_state, current_regime):
                        supporting.append('underlying_bollinger')
            
            return supporting
            
        except Exception as e:
            logger.error(f"Error identifying supporting indicators: {e}")
            return []
    
    def _momentum_to_score(self, momentum: str) -> float:
        """Convert momentum string to numerical score"""
        momentum_scores = {
            'strong_bullish_accelerating': 1.0,
            'strong_bullish_momentum': 0.9,
            'bullish_momentum': 0.7,
            'bullish_weakening': 0.4,
            'neutral_momentum': 0.0,
            'bearish_weakening': -0.4,
            'bearish_momentum': -0.7,
            'strong_bearish_momentum': -0.9,
            'strong_bearish_accelerating': -1.0
        }
        return momentum_scores.get(momentum, 0.0)
    
    def _volatility_to_score(self, volatility: str) -> float:
        """Convert volatility string to numerical score"""
        volatility_scores = {
            'extreme_expansion': 1.0,
            'volatility_expansion': 0.8,
            'high_volatility': 0.6,
            'normal_volatility': 0.4,
            'low_volatility': 0.2,
            'volatility_contraction': 0.1,
            'extreme_compression': 0.0
        }
        return volatility_scores.get(volatility, 0.5)
    
    def _flow_to_score(self, flow_regime: str) -> float:
        """Convert flow regime to numerical score"""
        flow_scores = {
            'smart_money_accumulation_regime': 0.9,
            'institutional_dominated_regime': 0.8,
            'selective_accumulation_regime': 0.6,
            'expanding_activity_regime': 0.5,
            'balanced_flow_regime': 0.0,
            'contracting_activity_regime': -0.5,
            'retail_dominated_regime': -0.6
        }
        return flow_scores.get(flow_regime, 0.0)
    
    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment to numerical score"""
        if 'extreme' in sentiment and 'bullish' in sentiment:
            return 1.0
        elif 'bullish' in sentiment:
            return 0.6
        elif 'extreme' in sentiment and 'bearish' in sentiment:
            return -1.0
        elif 'bearish' in sentiment:
            return -0.6
        else:
            return 0.0
    
    def _regime_matches(self, indicator_regime: str, classified_regime: str) -> bool:
        """Check if indicator regime matches classified regime"""
        # Simplified matching logic
        if 'trend' in classified_regime and 'trend' in indicator_regime:
            return True
        if 'rang' in classified_regime and ('rang' in indicator_regime or 'neutral' in indicator_regime):
            return True
        if 'volatile' in classified_regime and 'volatile' in indicator_regime:
            return True
        return False
    
    def _momentum_supports_regime(self, momentum: str, regime: str) -> bool:
        """Check if momentum supports regime"""
        if 'trending' in regime:
            return 'momentum' in momentum and momentum != 'neutral_momentum'
        elif 'ranging' in regime:
            return momentum in ['neutral_momentum', 'bullish_weakening', 'bearish_weakening']
        return False
    
    def _flow_supports_regime(self, flow_regime: str, classified_regime: str) -> bool:
        """Check if flow regime supports classified regime"""
        if 'trending' in classified_regime:
            return 'accumulation' in flow_regime or 'institutional' in flow_regime
        elif 'volatile' in classified_regime:
            return 'expanding' in flow_regime
        return True
    
    def _volatility_supports_regime(self, vol_state: str, regime: str) -> bool:
        """Check if volatility state supports regime"""
        if 'volatile' in regime:
            return 'high' in vol_state or 'expansion' in vol_state
        elif 'quiet' in regime:
            return 'low' in vol_state or 'compression' in vol_state
        return True
    
    def _get_current_regime_duration(self) -> int:
        """Get duration of current regime"""
        try:
            if not self.regime_history['regimes']:
                return 0
            
            current_regime = self.regime_history['regimes'][-1]['regime']
            duration = 1
            
            # Count consecutive same regime
            for i in range(len(self.regime_history['regimes']) - 2, -1, -1):
                if self.regime_history['regimes'][i]['regime'] == current_regime:
                    duration += 1
                else:
                    break
            
            return duration
            
        except:
            return 0
    
    def _update_regime_history(self, results: Dict[str, Any]):
        """Update regime history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Track regime
            self.regime_history['regimes'].append({
                'timestamp': timestamp,
                'regime': results['current_regime'],
                'confidence': results['regime_confidence']
            })
            
            # Track transitions
            if len(self.regime_history['regimes']) > 1:
                prev_regime = self.regime_history['regimes'][-2]['regime']
                if prev_regime != results['current_regime']:
                    self.regime_history['transitions'].append({
                        'timestamp': timestamp,
                        'from': prev_regime,
                        'to': results['current_regime']
                    })
            
            # Keep only recent history
            max_history = 200
            for key in self.regime_history:
                if len(self.regime_history[key]) > max_history:
                    self.regime_history[key] = self.regime_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating regime history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'current_regime': 'undefined',
            'regime_confidence': 0.0,
            'regime_characteristics': {},
            'sub_regimes': [],
            'transition_probability': {},
            'regime_forecast': {},
            'supporting_indicators': []
        }
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regime analysis summary"""
        try:
            if not self.regime_history['regimes']:
                return {'status': 'no_history'}
            
            recent_regimes = self.regime_history['regimes'][-50:]
            
            # Count regime occurrences
            regime_counts = {}
            for r in recent_regimes:
                regime = r['regime']
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Calculate regime persistence
            regime_durations = []
            current_regime = recent_regimes[0]['regime']
            duration = 1
            
            for i in range(1, len(recent_regimes)):
                if recent_regimes[i]['regime'] == current_regime:
                    duration += 1
                else:
                    regime_durations.append(duration)
                    current_regime = recent_regimes[i]['regime']
                    duration = 1
            regime_durations.append(duration)
            
            return {
                'current_regime': recent_regimes[-1]['regime'],
                'current_confidence': recent_regimes[-1]['confidence'],
                'regime_distribution': {k: v/len(recent_regimes) for k, v in regime_counts.items()},
                'average_duration': np.mean(regime_durations) if regime_durations else 0,
                'transition_count': len(self.regime_history['transitions']),
                'stability_score': self._calculate_stability_score()
            }
            
        except Exception as e:
            logger.error(f"Error getting regime analysis: {e}")
            return {'status': 'error'}
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall regime stability score"""
        try:
            if len(self.regime_history['regimes']) < 20:
                return 0.5
            
            recent_regimes = self.regime_history['regimes'][-20:]
            
            # Count transitions
            transitions = 0
            for i in range(1, len(recent_regimes)):
                if recent_regimes[i]['regime'] != recent_regimes[i-1]['regime']:
                    transitions += 1
            
            # Fewer transitions = more stable
            stability = 1.0 - (transitions / (len(recent_regimes) - 1))
            
            # Adjust for confidence
            avg_confidence = np.mean([r['confidence'] for r in recent_regimes])
            
            return float(stability * avg_confidence)
            
        except:
            return 0.5