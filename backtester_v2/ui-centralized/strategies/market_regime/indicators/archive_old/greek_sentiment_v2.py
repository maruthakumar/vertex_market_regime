"""
Greek Sentiment Analysis V2 - Refactored Implementation
======================================================

Refactored Greek sentiment analysis with volume+OI weighting, ITM analysis,
and modular architecture using the new base classes.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from base.base_indicator import BaseIndicator, IndicatorConfig, IndicatorOutput, IndicatorState
from base.strike_selector_base import BaseStrikeSelector, StrikeSelectionStrategy, create_strike_selector
from base.option_data_manager import OptionDataManager

logger = logging.getLogger(__name__)

class GreekSentimentV2(BaseIndicator):
    """
    Enhanced Greek Sentiment Analysis V2
    
    Features:
    - Dual weighting system (α × OI + β × Volume)
    - ITM analysis for institutional sentiment
    - Adaptive weight optimization
    - Modular strike selection
    - Preserved 9:15 AM baseline logic
    - 7-level sentiment classification
    - DTE-specific adjustments
    """
    
    def __init__(self, config: IndicatorConfig):
        """Initialize Greek Sentiment V2"""
        super().__init__(config)
        
        # Configuration parameters
        params = config.parameters
        
        # Dual weighting parameters (α × OI + β × Volume)
        self.oi_weight_alpha = params.get('oi_weight_alpha', 0.6)
        self.volume_weight_beta = params.get('volume_weight_beta', 0.4)
        
        # Greek weights (preserved from original)
        self.base_greek_weights = {
            'delta': params.get('delta_weight', 1.2),
            'vega': params.get('vega_weight', 1.5),
            'theta': params.get('theta_weight', 0.3),
            'gamma': params.get('gamma_weight', 0.0)
        }
        
        # DTE-specific adjustments (preserved)
        self.dte_adjustments = {
            'near_expiry': {'delta': 1.0, 'vega': 0.8, 'theta': 1.5, 'gamma': 1.2},
            'medium_expiry': {'delta': 1.2, 'vega': 1.5, 'theta': 0.8, 'gamma': 1.0},
            'far_expiry': {'delta': 1.0, 'vega': 2.0, 'theta': 0.3, 'gamma': 0.8}
        }
        
        # Sentiment thresholds (7-level system, preserved)
        self.sentiment_thresholds = {
            'strong_bullish': params.get('strong_bullish_threshold', 0.45),
            'mild_bullish': params.get('mild_bullish_threshold', 0.15),
            'sideways_to_bullish': 0.08,
            'neutral_upper': 0.05,
            'neutral_lower': -0.05,
            'sideways_to_bearish': -0.08,
            'mild_bearish': -0.15,
            'strong_bearish': -0.45
        }
        
        # ITM analysis configuration
        self.enable_itm_analysis = params.get('enable_itm_analysis', True)
        self.itm_threshold = params.get('itm_threshold', 0.02)  # 2% ITM threshold
        
        # Strike selection
        self.strike_selector = create_strike_selector(
            config.strike_selection_strategy,
            params.get('strike_selector_config', {})
        )
        
        # Option data manager for ATM tracking
        self.option_data_manager = OptionDataManager(
            params.get('option_data_config', {})
        )
        
        # Session baseline tracking (preserved logic)
        self.session_baselines = {}
        self.baseline_update_frequency = params.get('baseline_update_frequency', 30)  # minutes
        
        # Performance tracking for adaptive weights
        self.greek_performance_history = {
            greek: {'accuracy': 0.5, 'recent_performance': []}
            for greek in self.base_greek_weights.keys()
        }
        
        self.state = IndicatorState.READY
        logger.info(f"GreekSentimentV2 initialized with dual weighting (α={self.oi_weight_alpha}, β={self.volume_weight_beta})")
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return [
            'strike', 'option_type', 'expiry_date', 'timestamp',
            'ce_delta', 'ce_gamma', 'ce_theta', 'ce_vega',
            'pe_delta', 'pe_gamma', 'pe_theta', 'pe_vega',
            'ce_oi', 'pe_oi', 'ce_volume', 'pe_volume',
            'underlying_price', 'dte'
        ]
    
    def validate_data(self, market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data"""
        errors = []
        
        # Check required columns
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in market_data.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check data volume
        if len(market_data) < 3:
            errors.append("Insufficient data: need at least 3 option strikes")
        
        # Check for both CE and PE data
        if 'option_type' in market_data.columns:
            option_types = set(market_data['option_type'].unique())
            if not ({'CE', 'PE'}.issubset(option_types)):
                errors.append("Missing CE or PE option data")
        
        # Check for valid Greeks (non-zero)
        greek_cols = [col for col in market_data.columns if any(greek in col for greek in ['delta', 'gamma', 'theta', 'vega'])]
        if greek_cols:
            if market_data[greek_cols].abs().sum().sum() == 0:
                errors.append("All Greeks are zero - invalid data")
        
        return len(errors) == 0, errors
    
    def analyze(self, market_data: pd.DataFrame, **kwargs) -> IndicatorOutput:
        """
        Analyze Greek sentiment with dual weighting and ITM analysis
        
        Args:
            market_data: Option market data
            **kwargs: Additional parameters
            
        Returns:
            IndicatorOutput: Sentiment analysis result
        """
        try:
            start_time = datetime.now()
            spot_price = kwargs.get('spot_price') or market_data['underlying_price'].iloc[0]
            current_dte = kwargs.get('dte') or market_data['dte'].iloc[0]
            
            # Step 1: Select strikes using configured strategy
            selected_strikes = self.strike_selector.select_strikes(
                market_data, spot_price, dte=current_dte, **kwargs
            )
            
            if not selected_strikes:
                return IndicatorOutput(
                    value=0.0,
                    confidence=0.0,
                    metadata={'error': 'No strikes selected', 'method': 'greek_sentiment_v2'}
                )
            
            # Step 2: Calculate weighted Greeks with dual weighting (α × OI + β × Volume)
            weighted_greeks = self._calculate_dual_weighted_greeks(
                market_data, selected_strikes, spot_price
            )
            
            # Step 3: Update session baselines (preserved 9:15 AM logic)
            baselines = self._update_session_baselines(weighted_greeks)
            
            # Step 4: Calculate baseline changes
            baseline_changes = self._calculate_baseline_changes(weighted_greeks, baselines)
            
            # Step 5: Apply DTE-specific adjustments
            dte_adjusted_greeks = self._apply_dte_adjustments(baseline_changes, current_dte)
            
            # Step 6: ITM analysis for institutional sentiment
            itm_sentiment = 0.0
            if self.enable_itm_analysis:
                itm_sentiment = self._analyze_itm_sentiment(market_data, spot_price)
            
            # Step 7: Calculate individual Greek contributions
            greek_contributions = self._calculate_greek_contributions(dte_adjusted_greeks)
            
            # Step 8: Apply adaptive weights
            optimized_weights = self._get_adaptive_weights()
            
            # Step 9: Calculate final sentiment score
            sentiment_score = self._calculate_weighted_sentiment(
                greek_contributions, optimized_weights, itm_sentiment
            )
            
            # Step 10: Classify sentiment (7-level system)
            sentiment_classification = self._classify_sentiment(sentiment_score)
            
            # Step 11: Calculate confidence
            confidence = self._calculate_confidence(
                greek_contributions, selected_strikes, market_data
            )
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return IndicatorOutput(
                value=sentiment_score,
                confidence=confidence,
                metadata={
                    'sentiment_classification': sentiment_classification,
                    'greek_contributions': greek_contributions,
                    'dual_weighting': {
                        'oi_alpha': self.oi_weight_alpha,
                        'volume_beta': self.volume_weight_beta
                    },
                    'itm_sentiment': itm_sentiment,
                    'baseline_changes': baseline_changes,
                    'dte_adjustments': dte_adjusted_greeks,
                    'selected_strikes_count': len(selected_strikes),
                    'adaptive_weights': optimized_weights,
                    'method': 'greek_sentiment_v2',
                    'preserved_logic': 'baseline_tracking_dte_adjustments_7_level_classification'
                },
                computation_time=computation_time
            )
            
        except Exception as e:
            logger.error(f"Error in GreekSentimentV2 analysis: {e}")
            return IndicatorOutput(
                value=0.0,
                confidence=0.0,
                metadata={'error': str(e), 'method': 'greek_sentiment_v2'}
            )
    
    def _calculate_dual_weighted_greeks(self, 
                                      market_data: pd.DataFrame,
                                      selected_strikes: List,
                                      spot_price: float) -> Dict[str, float]:
        """Calculate Greeks with dual weighting (α × OI + β × Volume)"""
        weighted_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        total_weight = 0
        
        for strike_info in selected_strikes:
            strike = strike_info.strike
            option_type = strike_info.option_type
            strike_weight = strike_info.weight
            
            # Get data for this strike and option type
            strike_data = market_data[
                (market_data['strike'] == strike) & 
                (market_data['option_type'] == option_type)
            ]
            
            if strike_data.empty:
                continue
            
            row = strike_data.iloc[0]
            
            # Extract Greeks
            if option_type == 'CE':
                delta = row.get('ce_delta', 0)
                gamma = row.get('ce_gamma', 0)
                theta = row.get('ce_theta', 0)
                vega = row.get('ce_vega', 0)
                oi = row.get('ce_oi', 0)
                volume = row.get('ce_volume', 0)
            else:  # PE
                delta = row.get('pe_delta', 0)
                gamma = row.get('pe_gamma', 0)
                theta = row.get('pe_theta', 0)
                vega = row.get('pe_vega', 0)
                oi = row.get('pe_oi', 0)
                volume = row.get('pe_volume', 0)
            
            # Calculate dual weight: α × OI + β × Volume
            dual_weight = (
                self.oi_weight_alpha * oi + 
                self.volume_weight_beta * volume
            ) * strike_weight
            
            if dual_weight > 0:
                weighted_greeks['delta'] += delta * dual_weight
                weighted_greeks['gamma'] += gamma * dual_weight
                weighted_greeks['theta'] += theta * dual_weight
                weighted_greeks['vega'] += vega * dual_weight
                total_weight += dual_weight
        
        # Normalize by total weight
        if total_weight > 0:
            for greek in weighted_greeks:
                weighted_greeks[greek] /= total_weight
        
        return weighted_greeks
    
    def _analyze_itm_sentiment(self, market_data: pd.DataFrame, spot_price: float) -> float:
        """Analyze ITM options for institutional sentiment"""
        try:
            itm_sentiment = 0.0
            
            # Analyze ITM calls (strikes below spot)
            itm_calls = market_data[
                (market_data['option_type'] == 'CE') & 
                (market_data['strike'] < spot_price * (1 - self.itm_threshold))
            ]
            
            # Analyze ITM puts (strikes above spot)
            itm_puts = market_data[
                (market_data['option_type'] == 'PE') & 
                (market_data['strike'] > spot_price * (1 + self.itm_threshold))
            ]
            
            if not itm_calls.empty:
                call_oi_flow = itm_calls['ce_oi'].sum()
                call_volume_flow = itm_calls['ce_volume'].sum()
                call_sentiment = self.oi_weight_alpha * call_oi_flow + self.volume_weight_beta * call_volume_flow
            else:
                call_sentiment = 0
            
            if not itm_puts.empty:
                put_oi_flow = itm_puts['pe_oi'].sum()
                put_volume_flow = itm_puts['pe_volume'].sum()
                put_sentiment = self.oi_weight_alpha * put_oi_flow + self.volume_weight_beta * put_volume_flow
            else:
                put_sentiment = 0
            
            # Net ITM sentiment (calls positive, puts negative)
            if call_sentiment + put_sentiment > 0:
                itm_sentiment = (call_sentiment - put_sentiment) / (call_sentiment + put_sentiment)
            
            # Scale and bound
            itm_sentiment = np.clip(itm_sentiment * 0.3, -0.3, 0.3)  # 30% max contribution
            
            return itm_sentiment
            
        except Exception as e:
            logger.error(f"Error in ITM analysis: {e}")
            return 0.0
    
    def _update_session_baselines(self, weighted_greeks: Dict[str, float]) -> Dict[str, float]:
        """Update session baselines (preserved 9:15 AM logic)"""
        try:
            current_time = datetime.now()
            session_date = current_time.date()
            
            # Check if we need new baselines (new session)
            if session_date not in self.session_baselines:
                self.session_baselines[session_date] = {
                    'delta': weighted_greeks['delta'],
                    'gamma': weighted_greeks['gamma'],
                    'theta': weighted_greeks['theta'],
                    'vega': weighted_greeks['vega'],
                    'established_time': current_time,
                    'last_update': current_time
                }
                logger.info(f"Established new Greek baselines for session {session_date}")
            
            # Update baselines every 30 minutes (preserved logic)
            last_update = self.session_baselines[session_date]['last_update']
            if (current_time - last_update).total_seconds() > (self.baseline_update_frequency * 60):
                alpha = 0.1  # Smoothing factor
                
                for greek in ['delta', 'gamma', 'theta', 'vega']:
                    current_baseline = self.session_baselines[session_date][greek]
                    new_value = weighted_greeks[greek]
                    
                    # Exponential smoothing
                    self.session_baselines[session_date][greek] = (
                        alpha * new_value + (1 - alpha) * current_baseline
                    )
                
                self.session_baselines[session_date]['last_update'] = current_time
            
            return self.session_baselines[session_date]
            
        except Exception as e:
            logger.error(f"Error updating baselines: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _calculate_baseline_changes(self, 
                                  weighted_greeks: Dict[str, float],
                                  baselines: Dict[str, float]) -> Dict[str, float]:
        """Calculate changes from session baselines (preserved logic)"""
        baseline_changes = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            current_value = weighted_greeks[greek]
            baseline_value = baselines.get(greek, 0)
            
            if abs(baseline_value) > 1e-6:
                change = (current_value - baseline_value) / abs(baseline_value)
            else:
                change = current_value
            
            baseline_changes[greek] = change
        
        return baseline_changes
    
    def _apply_dte_adjustments(self, 
                             baseline_changes: Dict[str, float],
                             dte: int) -> Dict[str, float]:
        """Apply DTE-specific adjustments (preserved logic)"""
        # Classify DTE
        if dte <= 7:
            dte_category = 'near_expiry'
        elif dte <= 30:
            dte_category = 'medium_expiry'
        else:
            dte_category = 'far_expiry'
        
        # Get adjustments
        adjustments = self.dte_adjustments[dte_category]
        
        # Apply adjustments
        adjusted_greeks = {}
        for greek, change in baseline_changes.items():
            adjustment = adjustments.get(greek, 1.0)
            adjusted_greeks[greek] = change * adjustment
        
        return adjusted_greeks
    
    def _calculate_greek_contributions(self, dte_adjusted_greeks: Dict[str, float]) -> Dict[str, float]:
        """Calculate individual Greek contributions (preserved normalization)"""
        contributions = {}
        
        # Market-calibrated normalization (preserved from original)
        normalization_factors = {
            'delta': {'method': 'direct', 'factor': 1.0},
            'gamma': {'method': 'scale', 'factor': 50.0},
            'theta': {'method': 'scale', 'factor': 5.0},
            'vega': {'method': 'divide', 'factor': 20.0}
        }
        
        for greek, value in dte_adjusted_greeks.items():
            if greek in normalization_factors:
                norm_config = normalization_factors[greek]
                
                if norm_config['method'] == 'direct':
                    normalized_value = np.clip(value, -1.0, 1.0)
                elif norm_config['method'] == 'scale':
                    normalized_value = np.clip(value * norm_config['factor'], -1.0, 1.0)
                elif norm_config['method'] == 'divide':
                    normalized_value = np.clip(value / norm_config['factor'], -1.0, 1.0)
                else:
                    normalized_value = np.clip(value, -1.0, 1.0)
            else:
                normalized_value = np.clip(value, -1.0, 1.0)
            
            contributions[greek] = normalized_value
        
        return contributions
    
    def _get_adaptive_weights(self) -> Dict[str, float]:
        """Get adaptive Greek weights based on performance"""
        adaptive_weights = self.base_greek_weights.copy()
        
        # Apply performance-based adjustments
        for greek, base_weight in adaptive_weights.items():
            performance_data = self.greek_performance_history[greek]
            accuracy = performance_data['accuracy']
            
            # Adjust based on recent performance
            if accuracy > 0.7:
                adjustment = 1.2  # Boost good performers
            elif accuracy < 0.4:
                adjustment = 0.8  # Reduce poor performers
            else:
                adjustment = 1.0
            
            adaptive_weights[greek] = base_weight * adjustment
        
        # Normalize
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            for greek in adaptive_weights:
                adaptive_weights[greek] /= total_weight
        
        return adaptive_weights
    
    def _calculate_weighted_sentiment(self, 
                                    greek_contributions: Dict[str, float],
                                    weights: Dict[str, float],
                                    itm_sentiment: float) -> float:
        """Calculate final weighted sentiment score"""
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for greek, contribution in greek_contributions.items():
            weight = weights.get(greek, 0.0)
            weighted_sentiment += contribution * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_sentiment /= total_weight
        
        # Add ITM sentiment contribution
        final_sentiment = weighted_sentiment + itm_sentiment
        
        return np.clip(final_sentiment, -1.0, 1.0)
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment using 7-level system (preserved)"""
        if sentiment_score > self.sentiment_thresholds['strong_bullish']:
            return 'Strong_Bullish'
        elif sentiment_score > self.sentiment_thresholds['mild_bullish']:
            return 'Mild_Bullish'
        elif sentiment_score > self.sentiment_thresholds['sideways_to_bullish']:
            return 'Sideways_To_Bullish'
        elif sentiment_score > self.sentiment_thresholds['neutral_lower']:
            return 'Neutral'
        elif sentiment_score > self.sentiment_thresholds['sideways_to_bearish']:
            return 'Sideways_To_Bearish'
        elif sentiment_score > self.sentiment_thresholds['mild_bearish']:
            return 'Mild_Bearish'
        else:
            return 'Strong_Bearish'
    
    def _calculate_confidence(self, 
                            greek_contributions: Dict[str, float],
                            selected_strikes: List,
                            market_data: pd.DataFrame) -> float:
        """Calculate confidence in analysis"""
        try:
            # Greek consistency
            greek_values = list(greek_contributions.values())
            greek_consistency = 1.0 - np.std(greek_values) if len(greek_values) > 1 else 0.5
            
            # Strike coverage
            strike_coverage = min(len(selected_strikes) / 10, 1.0)
            
            # Volume quality
            total_volume = 0
            for _, row in market_data.iterrows():
                total_volume += row.get('ce_volume', 0) + row.get('pe_volume', 0)
            
            volume_quality = min(total_volume / 10000, 1.0)
            
            # Combined confidence
            confidence = (
                greek_consistency * 0.4 +
                strike_coverage * 0.3 +
                volume_quality * 0.3
            )
            
            return np.clip(confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5