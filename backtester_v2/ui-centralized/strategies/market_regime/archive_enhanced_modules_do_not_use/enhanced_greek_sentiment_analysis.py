"""
Enhanced Greek Sentiment Analysis for Backtester V2

This module implements comprehensive Greek sentiment analysis with the following features:

1. Multi-Greek Analysis (Delta, Gamma, Theta, Vega)
2. Baseline Tracking System (Opening values as reference)
3. DTE-Specific Weight Adjustments
4. Dynamic Weight Optimization based on performance
5. Cross-Strike Greek Correlation Analysis
6. Volatility Regime Adaptation
7. Integration with Triple Straddle Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class GreekSentimentType(Enum):
    """Greek sentiment classifications"""
    STRONG_BULLISH = "Strong_Bullish"
    MILD_BULLISH = "Mild_Bullish"
    SIDEWAYS_TO_BULLISH = "Sideways_To_Bullish"
    NEUTRAL = "Neutral"
    SIDEWAYS_TO_BEARISH = "Sideways_To_Bearish"
    MILD_BEARISH = "Mild_Bearish"
    STRONG_BEARISH = "Strong_Bearish"

@dataclass
class GreekAnalysisResult:
    """Result structure for Greek analysis"""
    sentiment_score: float
    sentiment_type: GreekSentimentType
    confidence: float
    delta_contribution: float
    gamma_contribution: float
    theta_contribution: float
    vega_contribution: float
    baseline_changes: Dict[str, float]
    dte_adjustments: Dict[str, float]
    cross_strike_correlation: float

class EnhancedGreekSentimentAnalysis:
    """
    Enhanced Greek Sentiment Analysis Engine
    
    Implements comprehensive Greek analysis with baseline tracking, DTE adjustments,
    dynamic weighting, and integration with market regime detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Enhanced Greek Sentiment Analysis"""
        self.config = config or {}
        
        # Base Greek weights (dynamic, starting values)
        self.base_greek_weights = {
            'delta': 1.2,   # 40% influence (normalized)
            'vega': 1.5,    # 50% influence (normalized)
            'theta': 0.3,   # 10% influence (normalized)
            'gamma': 0.0    # Initially 0%, can be enabled
        }
        
        # DTE-specific weight adjustments
        self.dte_weight_adjustments = {
            'near_expiry': {    # 0-7 DTE
                'delta': 1.0,
                'vega': 0.8,
                'theta': 1.5,   # Higher theta impact near expiry
                'gamma': 1.2
            },
            'medium_expiry': {  # 8-30 DTE
                'delta': 1.2,
                'vega': 1.5,    # Balanced vega impact
                'theta': 0.8,
                'gamma': 1.0
            },
            'far_expiry': {     # 30+ DTE
                'delta': 1.0,
                'vega': 2.0,    # Higher vega impact far from expiry
                'theta': 0.3,
                'gamma': 0.8
            }
        }
        
        # CALIBRATED: Sentiment classification thresholds (7-level system) for Indian market
        self.sentiment_thresholds = {
            'strong_bullish': 0.45,      # Reduced from 0.6 to 0.45 for better sensitivity
            'mild_bullish': 0.15,        # Reduced from 0.2 to 0.15
            'sideways_to_bullish': 0.08, # Reduced from 0.1 to 0.08
            'neutral_upper': 0.05,       # Reduced from 0.1 to 0.05
            'neutral_lower': -0.05,      # Adjusted from -0.1 to -0.05
            'sideways_to_bearish': -0.08, # Adjusted from -0.2 to -0.08
            'mild_bearish': -0.15,       # Adjusted from -0.2 to -0.15
            'strong_bearish': -0.45      # Adjusted from -0.6 to -0.45
        }
        
        # Performance tracking for dynamic weights
        self.greek_performance_history = {
            'delta': {'accuracy': 0.5, 'weight_adjustment': 0.0},
            'vega': {'accuracy': 0.5, 'weight_adjustment': 0.0},
            'theta': {'accuracy': 0.5, 'weight_adjustment': 0.0},
            'gamma': {'accuracy': 0.5, 'weight_adjustment': 0.0}
        }
        
        # Baseline tracking
        self.session_baselines = {}
        self.baseline_update_frequency = int(self.config.get('baseline_update_frequency', 30))  # minutes
        
        # Volatility regime adaptation
        self.volatility_thresholds = {
            'low': 0.10,
            'normal': 0.25,
            'high': 0.40
        }
        
        # Learning rate for dynamic adjustments
        self.learning_rate = float(self.config.get('learning_rate', 0.05))
        
        logger.info("Enhanced Greek Sentiment Analysis initialized")
    
    def analyze_greek_sentiment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis function for Greek sentiment
        
        Args:
            market_data: Comprehensive market data including Greeks, options data, etc.
            
        Returns:
            Dictionary with complete Greek sentiment analysis results
        """
        try:
            # Step 1: Extract and prepare Greek data
            greek_data = self._prepare_greek_data(market_data)
            
            if not greek_data:
                logger.warning("Insufficient Greek data for sentiment analysis")
                return self._get_default_result()
            
            # Step 2: Update/establish baselines
            baselines = self._update_session_baselines(greek_data)
            
            # Step 3: Calculate baseline changes
            baseline_changes = self._calculate_baseline_changes(greek_data, baselines)
            
            # Step 4: Apply DTE-specific adjustments
            dte_adjusted_greeks = self._apply_dte_adjustments(baseline_changes, market_data)
            
            # Step 5: Calculate individual Greek contributions
            greek_contributions = self._calculate_greek_contributions(dte_adjusted_greeks)
            
            # Step 6: Apply dynamic weight optimization
            optimized_weights = self._get_optimized_weights(market_data)
            
            # Step 7: Calculate weighted sentiment score
            sentiment_score = self._calculate_weighted_sentiment(greek_contributions, optimized_weights)
            
            # Step 8: Apply volatility regime adaptation
            regime_adjusted_score = self._apply_volatility_regime_adaptation(sentiment_score, market_data)
            
            # Step 9: Calculate cross-strike correlation
            cross_strike_correlation = self._calculate_cross_strike_correlation(market_data)
            
            # Step 10: Classify sentiment and calculate confidence
            sentiment_type = self._classify_sentiment(regime_adjusted_score)
            confidence = self._calculate_sentiment_confidence(
                greek_contributions, cross_strike_correlation, market_data
            )
            
            # Step 11: Update performance tracking
            self._update_performance_tracking(sentiment_score, sentiment_type, market_data)
            
            return {
                'sentiment_score': regime_adjusted_score,
                'sentiment_type': sentiment_type.value,
                'confidence': confidence,
                'greek_contributions': greek_contributions,
                'baseline_changes': baseline_changes,
                'dte_adjustments': dte_adjusted_greeks,
                'optimized_weights': optimized_weights,
                'cross_strike_correlation': cross_strike_correlation,
                'volatility_regime': self._get_volatility_regime(market_data),
                'timestamp': datetime.now(),
                'analysis_type': 'enhanced_greek_sentiment_v2'
            }
            
        except Exception as e:
            logger.error(f"Error in Enhanced Greek Sentiment analysis: {e}")
            return self._get_default_result()
    
    def _prepare_greek_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and validate Greek data for analysis"""
        try:
            # Extract Greek data from market data
            greek_data = market_data.get('greek_data', {})
            options_data = market_data.get('options_data', {})
            
            if not greek_data and not options_data:
                return {}
            
            # If Greek data is provided directly
            if greek_data:
                return {
                    'aggregate_greeks': greek_data,
                    'strike_level_greeks': {},
                    'data_source': 'direct'
                }
            
            # Calculate Greeks from options data
            strike_level_greeks = {}
            aggregate_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            total_positions = 0
            
            for strike, option_data in options_data.items():
                strike_greeks = {}
                
                # Extract Greeks for calls
                if 'CE' in option_data:
                    ce_data = option_data['CE']
                    strike_greeks['call_delta'] = ce_data.get('delta', 0)
                    strike_greeks['call_gamma'] = ce_data.get('gamma', 0)
                    strike_greeks['call_theta'] = ce_data.get('theta', 0)
                    strike_greeks['call_vega'] = ce_data.get('vega', 0)
                    
                    # Weight by position size (OI)
                    call_oi = ce_data.get('oi', 0)
                    if call_oi > 0:
                        aggregate_greeks['delta'] += strike_greeks['call_delta'] * call_oi
                        aggregate_greeks['gamma'] += strike_greeks['call_gamma'] * call_oi
                        aggregate_greeks['theta'] += strike_greeks['call_theta'] * call_oi
                        aggregate_greeks['vega'] += strike_greeks['call_vega'] * call_oi
                        total_positions += call_oi
                
                # Extract Greeks for puts
                if 'PE' in option_data:
                    pe_data = option_data['PE']
                    strike_greeks['put_delta'] = pe_data.get('delta', 0)
                    strike_greeks['put_gamma'] = pe_data.get('gamma', 0)
                    strike_greeks['put_theta'] = pe_data.get('theta', 0)
                    strike_greeks['put_vega'] = pe_data.get('vega', 0)
                    
                    # Weight by position size (OI)
                    put_oi = pe_data.get('oi', 0)
                    if put_oi > 0:
                        aggregate_greeks['delta'] += strike_greeks['put_delta'] * put_oi
                        aggregate_greeks['gamma'] += strike_greeks['put_gamma'] * put_oi
                        aggregate_greeks['theta'] += strike_greeks['put_theta'] * put_oi
                        aggregate_greeks['vega'] += strike_greeks['put_vega'] * put_oi
                        total_positions += put_oi
                
                strike_level_greeks[strike] = strike_greeks
            
            # Normalize aggregate Greeks by total positions
            if total_positions > 0:
                for greek in aggregate_greeks:
                    aggregate_greeks[greek] /= total_positions
            
            return {
                'aggregate_greeks': aggregate_greeks,
                'strike_level_greeks': strike_level_greeks,
                'total_positions': total_positions,
                'data_source': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error preparing Greek data: {e}")
            return {}
    
    def _update_session_baselines(self, greek_data: Dict[str, Any]) -> Dict[str, float]:
        """Update or establish session baselines for Greek values"""
        try:
            current_time = datetime.now()
            session_date = current_time.date()
            
            # Check if we need to establish new baselines (new session or first time)
            if session_date not in self.session_baselines:
                # Establish new baselines
                aggregate_greeks = greek_data.get('aggregate_greeks', {})
                self.session_baselines[session_date] = {
                    'delta': aggregate_greeks.get('delta', 0),
                    'gamma': aggregate_greeks.get('gamma', 0),
                    'theta': aggregate_greeks.get('theta', 0),
                    'vega': aggregate_greeks.get('vega', 0),
                    'established_time': current_time,
                    'last_update': current_time
                }
                logger.info(f"Established new Greek baselines for session {session_date}")
            
            # Check if baselines need updating (every 30 minutes)
            last_update = self.session_baselines[session_date]['last_update']
            if (current_time - last_update).total_seconds() > (self.baseline_update_frequency * 60):
                # Update baselines with exponential smoothing
                aggregate_greeks = greek_data.get('aggregate_greeks', {})
                alpha = 0.1  # Smoothing factor
                
                for greek in ['delta', 'gamma', 'theta', 'vega']:
                    current_value = aggregate_greeks.get(greek, 0)
                    baseline_value = self.session_baselines[session_date][greek]
                    
                    # Exponential smoothing update
                    new_baseline = alpha * current_value + (1 - alpha) * baseline_value
                    self.session_baselines[session_date][greek] = new_baseline
                
                self.session_baselines[session_date]['last_update'] = current_time
                logger.debug(f"Updated Greek baselines for session {session_date}")
            
            return self.session_baselines[session_date]
            
        except Exception as e:
            logger.error(f"Error updating session baselines: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _calculate_baseline_changes(self, greek_data: Dict[str, Any], 
                                  baselines: Dict[str, float]) -> Dict[str, float]:
        """Calculate changes from session baselines"""
        try:
            aggregate_greeks = greek_data.get('aggregate_greeks', {})
            baseline_changes = {}
            
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                current_value = aggregate_greeks.get(greek, 0)
                baseline_value = baselines.get(greek, 0)
                
                # Calculate percentage change from baseline
                if abs(baseline_value) > 1e-6:  # Avoid division by zero
                    change = (current_value - baseline_value) / abs(baseline_value)
                else:
                    change = current_value  # If baseline is zero, use absolute value
                
                baseline_changes[greek] = change
            
            return baseline_changes
            
        except Exception as e:
            logger.error(f"Error calculating baseline changes: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    def _apply_dte_adjustments(self, baseline_changes: Dict[str, float],
                             market_data: Dict[str, Any]) -> Dict[str, float]:
        """Apply DTE-specific weight adjustments to Greek changes"""
        try:
            # Get DTE information
            dte = market_data.get('dte', 30)  # Default to 30 DTE if not provided

            # Classify DTE
            if dte <= 7:
                dte_category = 'near_expiry'
            elif dte <= 30:
                dte_category = 'medium_expiry'
            else:
                dte_category = 'far_expiry'

            # Get DTE-specific weights
            dte_weights = self.dte_weight_adjustments[dte_category]

            # Apply DTE adjustments
            dte_adjusted_greeks = {}
            for greek, change in baseline_changes.items():
                dte_weight = dte_weights.get(greek, 1.0)
                dte_adjusted_greeks[greek] = change * dte_weight

            return dte_adjusted_greeks

        except Exception as e:
            logger.error(f"Error applying DTE adjustments: {e}")
            return baseline_changes

    def _calculate_greek_contributions(self, dte_adjusted_greeks: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate individual Greek contributions to sentiment

        CRITICAL FIX: Replaced arbitrary normalization factors with market-calibrated values
        based on Indian options market standards and enhanced-market-regime-optimizer analysis
        """
        try:
            contributions = {}

            # Market-calibrated normalization factors for Indian options market
            # Based on analysis of enhanced-market-regime-optimizer configurations
            # and typical Greek value ranges in NIFTY options
            normalization_factors = {
                'delta': {
                    'method': 'direct',  # Delta is already in [-1, 1] range
                    'factor': 1.0,
                    'description': 'Delta naturally bounded, no scaling needed'
                },
                'gamma': {
                    'method': 'scale',
                    'factor': 50.0,  # Calibrated: was 100 (too aggressive), now 50 for NIFTY
                    'description': 'Gamma scaling for NIFTY options (typical range 0.001-0.02)'
                },
                'theta': {
                    'method': 'scale',
                    'factor': 5.0,   # Calibrated: was 10 (too aggressive), now 5 for daily theta
                    'description': 'Theta scaling for daily decay (typical range -0.1 to -0.4)'
                },
                'vega': {
                    'method': 'divide',
                    'factor': 20.0,  # Calibrated: was 10, now 20 for NIFTY vega ranges
                    'description': 'Vega normalization for NIFTY options (typical range 5-40)'
                }
            }

            # Apply market-calibrated normalization
            for greek, value in dte_adjusted_greeks.items():
                if greek in normalization_factors:
                    norm_config = normalization_factors[greek]

                    if norm_config['method'] == 'direct':
                        # Delta: already in proper range
                        normalized_value = np.clip(value, -1.0, 1.0)

                    elif norm_config['method'] == 'scale':
                        # Gamma, Theta: multiply by calibrated factor
                        normalized_value = np.clip(value * norm_config['factor'], -1.0, 1.0)

                    elif norm_config['method'] == 'divide':
                        # Vega: divide by calibrated factor
                        normalized_value = np.clip(value / norm_config['factor'], -1.0, 1.0)

                    else:
                        # Fallback: direct clipping
                        normalized_value = np.clip(value, -1.0, 1.0)

                    logger.debug(f"Greek {greek}: raw={value:.6f}, normalized={normalized_value:.6f} "
                               f"(factor={norm_config['factor']}, method={norm_config['method']})")
                else:
                    # Unknown Greek: direct clipping
                    normalized_value = np.clip(value, -1.0, 1.0)
                    logger.debug(f"Unknown Greek {greek}: raw={value:.6f}, normalized={normalized_value:.6f}")

                contributions[greek] = normalized_value

            # Log normalization summary for validation
            logger.info(f"Greek normalization applied: "
                       f"delta={contributions.get('delta', 0):.4f}, "
                       f"gamma={contributions.get('gamma', 0):.4f}, "
                       f"theta={contributions.get('theta', 0):.4f}, "
                       f"vega={contributions.get('vega', 0):.4f}")

            return contributions

        except Exception as e:
            logger.error(f"Error calculating Greek contributions: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    def _get_optimized_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Get dynamically optimized Greek weights"""
        try:
            optimized_weights = self.base_greek_weights.copy()

            # Apply performance-based adjustments
            for greek, performance in self.greek_performance_history.items():
                accuracy = performance['accuracy']

                # Adjust weight based on recent performance
                if accuracy > 0.7:  # Good performance
                    weight_adjustment = (accuracy - 0.5) * 2 * self.learning_rate
                elif accuracy < 0.4:  # Poor performance
                    weight_adjustment = (accuracy - 0.5) * 2 * self.learning_rate
                else:
                    weight_adjustment = 0.0

                # Apply bounded adjustment
                current_weight = optimized_weights.get(greek, 0.0)
                new_weight = current_weight + weight_adjustment
                optimized_weights[greek] = np.clip(new_weight, 0.1, 3.0)

            # Normalize weights
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                for greek in optimized_weights:
                    optimized_weights[greek] /= total_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"Error getting optimized weights: {e}")
            return self.base_greek_weights

    def _calculate_weighted_sentiment(self, greek_contributions: Dict[str, float],
                                    optimized_weights: Dict[str, float]) -> float:
        """Calculate weighted sentiment score from Greek contributions"""
        try:
            weighted_sentiment = 0.0
            total_weight = 0.0

            for greek, contribution in greek_contributions.items():
                weight = optimized_weights.get(greek, 0.0)
                weighted_sentiment += contribution * weight
                total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                weighted_sentiment /= total_weight

            return np.clip(weighted_sentiment, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating weighted sentiment: {e}")
            return 0.0

    def _apply_volatility_regime_adaptation(self, sentiment_score: float,
                                          market_data: Dict[str, Any]) -> float:
        """Apply volatility regime-based adaptations to sentiment score"""
        try:
            volatility = market_data.get('volatility', 0.15)

            # Determine volatility regime
            if volatility < self.volatility_thresholds['low']:
                regime = 'low'
                # In low volatility, reduce sentiment extremes
                adaptation_factor = 0.8
            elif volatility > self.volatility_thresholds['high']:
                regime = 'high'
                # In high volatility, amplify sentiment signals
                adaptation_factor = 1.2
            else:
                regime = 'normal'
                adaptation_factor = 1.0

            # Apply adaptation
            adapted_score = sentiment_score * adaptation_factor

            return np.clip(adapted_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Error applying volatility regime adaptation: {e}")
            return sentiment_score

    def _calculate_cross_strike_correlation(self, market_data: Dict[str, Any]) -> float:
        """Calculate correlation of Greeks across different strikes"""
        try:
            options_data = market_data.get('options_data', {})
            if len(options_data) < 3:  # Need at least 3 strikes for correlation
                return 0.5

            # Extract delta values across strikes
            call_deltas = []
            put_deltas = []
            strikes = []

            for strike, option_data in options_data.items():
                if 'CE' in option_data and 'PE' in option_data:
                    call_delta = option_data['CE'].get('delta', 0)
                    put_delta = option_data['PE'].get('delta', 0)

                    call_deltas.append(call_delta)
                    put_deltas.append(put_delta)
                    strikes.append(strike)

            if len(call_deltas) < 3:
                return 0.5

            # Calculate correlation between call and put deltas
            correlation = np.corrcoef(call_deltas, put_deltas)[0, 1]

            # Convert correlation to confidence measure
            # High absolute correlation = high confidence
            confidence = abs(correlation) if not np.isnan(correlation) else 0.5

            return np.clip(confidence, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating cross-strike correlation: {e}")
            return 0.5

    def _classify_sentiment(self, sentiment_score: float) -> GreekSentimentType:
        """Classify sentiment score into sentiment type"""
        try:
            if sentiment_score > self.sentiment_thresholds['strong_bullish']:
                return GreekSentimentType.STRONG_BULLISH
            elif sentiment_score > self.sentiment_thresholds['mild_bullish']:
                return GreekSentimentType.MILD_BULLISH
            elif sentiment_score > self.sentiment_thresholds['sideways_to_bullish']:
                return GreekSentimentType.SIDEWAYS_TO_BULLISH
            elif sentiment_score > self.sentiment_thresholds['neutral_lower']:
                return GreekSentimentType.NEUTRAL
            elif sentiment_score > self.sentiment_thresholds['sideways_to_bearish']:
                return GreekSentimentType.SIDEWAYS_TO_BEARISH
            elif sentiment_score > self.sentiment_thresholds['strong_bearish']:
                return GreekSentimentType.MILD_BEARISH
            else:
                return GreekSentimentType.STRONG_BEARISH

        except Exception as e:
            logger.error(f"Error classifying sentiment: {e}")
            return GreekSentimentType.NEUTRAL

    def _calculate_sentiment_confidence(self, greek_contributions: Dict[str, float],
                                      cross_strike_correlation: float,
                                      market_data: Dict[str, Any]) -> float:
        """Calculate confidence in sentiment analysis"""
        try:
            # Base confidence from Greek consistency
            greek_values = list(greek_contributions.values())
            greek_consistency = 1.0 - np.std(greek_values) if len(greek_values) > 1 else 0.5

            # Cross-strike correlation confidence
            correlation_confidence = cross_strike_correlation

            # Data quality confidence
            options_data = market_data.get('options_data', {})
            data_quality = min(len(options_data) / 10, 1.0)  # More strikes = higher confidence

            # Volume confidence
            total_volume = 0
            for strike, option_data in options_data.items():
                if 'CE' in option_data:
                    total_volume += option_data['CE'].get('volume', 0)
                if 'PE' in option_data:
                    total_volume += option_data['PE'].get('volume', 0)

            volume_confidence = min(total_volume / 10000, 1.0)  # Normalize volume

            # Combined confidence
            combined_confidence = (
                greek_consistency * 0.3 +
                correlation_confidence * 0.3 +
                data_quality * 0.2 +
                volume_confidence * 0.2
            )

            return np.clip(combined_confidence, 0.1, 1.0)

        except Exception as e:
            logger.error(f"Error calculating sentiment confidence: {e}")
            return 0.5

    def _get_volatility_regime(self, market_data: Dict[str, Any]) -> str:
        """Get current volatility regime"""
        try:
            volatility = market_data.get('volatility', 0.15)

            if volatility < self.volatility_thresholds['low']:
                return 'low_volatility'
            elif volatility > self.volatility_thresholds['high']:
                return 'high_volatility'
            else:
                return 'normal_volatility'

        except Exception as e:
            logger.error(f"Error getting volatility regime: {e}")
            return 'normal_volatility'

    def _update_performance_tracking(self, sentiment_score: float,
                                   sentiment_type: GreekSentimentType,
                                   market_data: Dict[str, Any]):
        """Update performance tracking for dynamic weight optimization"""
        try:
            # This would be implemented with actual market outcome validation
            # For now, we'll use a simplified approach

            # Simulate performance based on sentiment strength and market conditions
            sentiment_strength = abs(sentiment_score)

            # Higher sentiment strength should correlate with better performance
            simulated_accuracy = 0.5 + (sentiment_strength * 0.3)

            # Update performance history for each Greek
            for greek in self.greek_performance_history:
                current_accuracy = self.greek_performance_history[greek]['accuracy']

                # Exponential moving average update
                alpha = 0.1
                new_accuracy = alpha * simulated_accuracy + (1 - alpha) * current_accuracy
                self.greek_performance_history[greek]['accuracy'] = new_accuracy

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    def _get_default_result(self) -> Dict[str, Any]:
        """Get default result when analysis fails"""
        return {
            'sentiment_score': 0.0,
            'sentiment_type': GreekSentimentType.NEUTRAL.value,
            'confidence': 0.5,
            'greek_contributions': {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0},
            'baseline_changes': {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0},
            'dte_adjustments': {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0},
            'optimized_weights': self.base_greek_weights,
            'cross_strike_correlation': 0.5,
            'volatility_regime': 'normal_volatility',
            'timestamp': datetime.now(),
            'analysis_type': 'enhanced_greek_sentiment_v2_default'
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for optimization"""
        try:
            summary = {}
            for greek, performance in self.greek_performance_history.items():
                summary[greek] = {
                    'accuracy': performance['accuracy'],
                    'reliability': 'High' if performance['accuracy'] > 0.7 else
                                  'Medium' if performance['accuracy'] > 0.5 else 'Low',
                    'weight_adjustment': performance.get('weight_adjustment', 0.0)
                }
            return summary
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

    def reset_session_baselines(self):
        """Reset session baselines (useful for new trading session)"""
        try:
            self.session_baselines.clear()
            logger.info("Session baselines reset")
        except Exception as e:
            logger.error(f"Error resetting session baselines: {e}")

    def update_sentiment_thresholds(self, new_thresholds: Dict[str, float]):
        """Update sentiment classification thresholds"""
        try:
            self.sentiment_thresholds.update(new_thresholds)
            logger.info("Sentiment thresholds updated")
        except Exception as e:
            logger.error(f"Error updating sentiment thresholds: {e}")

    def get_current_baselines(self) -> Dict[str, Any]:
        """Get current session baselines"""
        try:
            current_date = datetime.now().date()
            return self.session_baselines.get(current_date, {})
        except Exception as e:
            logger.error(f"Error getting current baselines: {e}")
            return {}
