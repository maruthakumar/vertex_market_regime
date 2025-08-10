"""
Greek Sentiment Analyzer - Main Orchestrator
===========================================

Main analyzer that orchestrates all Greek sentiment components:
- Baseline tracking with 9:15 AM logic
- Dual weighting system (α×OI + β×Volume)
- ITM/OTM analysis for institutional sentiment
- DTE-specific adjustments
- Market-calibrated Greek calculations

Author: Market Regime Refactoring Team  
Date: 2025-07-06
Version: 2.0.0 - Modular Orchestrator
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from base.base_indicator import BaseIndicator, IndicatorConfig, IndicatorOutput, IndicatorState
from base.strike_selector_base import create_strike_selector

from .baseline_tracker import BaselineTracker
from .volume_oi_weighter import VolumeOIWeighter
from .itm_otm_analyzer import ITMOTMAnalyzer
from .dte_adjuster import DTEAdjuster
from .greek_calculator import GreekCalculator

logger = logging.getLogger(__name__)

class GreekSentimentAnalyzer(BaseIndicator):
    """
    Main Greek Sentiment Analyzer using modular architecture
    
    This is the orchestrator that coordinates all the specialized components
    to produce comprehensive Greek sentiment analysis with all enhanced features.
    
    Enhanced Features:
    - Modular component architecture
    - Dual weighting system (α×OI + β×Volume) 
    - ITM/OTM institutional sentiment analysis
    - Preserved 9:15 AM baseline logic
    - Market-calibrated Greek normalization
    - DTE-specific adjustments
    - 7-level sentiment classification
    - Adaptive weight optimization
    """
    
    def __init__(self, config: IndicatorConfig):
        """Initialize Greek Sentiment Analyzer"""
        super().__init__(config)
        
        # Extract component configurations
        params = config.parameters
        
        # Initialize specialized components
        self.baseline_tracker = BaselineTracker(
            params.get('baseline_config', {})
        )
        
        self.volume_oi_weighter = VolumeOIWeighter({
            'oi_weight_alpha': params.get('oi_weight_alpha', 0.6),
            'volume_weight_beta': params.get('volume_weight_beta', 0.4),
            **params.get('weighting_config', {})
        })
        
        self.itm_otm_analyzer = ITMOTMAnalyzer({
            'itm_threshold': params.get('itm_threshold', 0.02),
            'enable_itm_analysis': params.get('enable_itm_analysis', True),
            **params.get('itm_otm_config', {})
        })
        
        self.dte_adjuster = DTEAdjuster({
            'near_expiry_threshold': params.get('near_expiry_threshold', 7),
            'medium_expiry_threshold': params.get('medium_expiry_threshold', 30),
            **params.get('dte_config', {})
        })
        
        self.greek_calculator = GreekCalculator({
            'gamma_factor': params.get('gamma_factor', 50.0),
            'theta_factor': params.get('theta_factor', 5.0),
            'vega_factor': params.get('vega_factor', 20.0),
            **params.get('calculator_config', {})
        })
        
        # Sentiment classification thresholds (preserved from original)
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
        
        # Strike selection strategy
        self.strike_selector = create_strike_selector(
            config.strike_selection_strategy,
            params.get('strike_selector_config', {})
        )
        
        # Component orchestration settings
        self.enable_all_components = params.get('enable_all_components', True)
        self.fallback_on_component_failure = params.get('fallback_on_component_failure', True)
        
        # Analysis metadata tracking
        self.component_health = {}
        self.analysis_history = []
        
        self.state = IndicatorState.READY
        logger.info("GreekSentimentAnalyzer initialized with modular architecture")
    
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
        """Validate input data for all components"""
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
        
        # Validate Greeks data
        greek_cols = [col for col in market_data.columns if any(greek in col for greek in ['delta', 'gamma', 'theta', 'vega'])]
        if greek_cols:
            if market_data[greek_cols].abs().sum().sum() == 0:
                errors.append("All Greeks are zero - invalid data")
        
        # Component-specific validations
        if self.enable_all_components:
            # Additional validation can be added here for specific components
            pass
        
        return len(errors) == 0, errors
    
    def analyze(self, market_data: pd.DataFrame, **kwargs) -> IndicatorOutput:
        """
        Main analysis orchestrating all components
        
        Args:
            market_data: Option market data
            **kwargs: Additional parameters
            
        Returns:
            IndicatorOutput: Comprehensive Greek sentiment analysis
        """
        try:
            start_time = datetime.now()
            
            # Extract parameters
            spot_price = kwargs.get('spot_price') or market_data['underlying_price'].iloc[0]
            current_dte = kwargs.get('dte') or market_data['dte'].iloc[0]
            volatility = kwargs.get('volatility', 0.2)
            current_time = kwargs.get('timestamp', datetime.now())
            
            # Market conditions for enhanced analysis
            market_conditions = {
                'spot_price': spot_price,
                'dte': current_dte,
                'volatility': volatility,
                'timestamp': current_time,
                'hour': current_time.hour,
                'weekday': current_time.weekday()
            }
            
            # Initialize analysis results
            analysis_results = {
                'component_results': {},
                'component_health': {},
                'orchestration_metadata': {
                    'components_enabled': {},
                    'components_successful': {},
                    'fallbacks_used': []
                }
            }
            
            # Step 1: Strike Selection
            try:
                selected_strikes = self.strike_selector.select_strikes(
                    market_data, spot_price, dte=current_dte, volatility=volatility
                )
                
                if not selected_strikes:
                    return self._get_error_output("No strikes selected")
                
                analysis_results['orchestration_metadata']['strikes_selected'] = len(selected_strikes)
                self.component_health['strike_selector'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Strike selection failed: {e}")
                if not self.fallback_on_component_failure:
                    return self._get_error_output(f"Strike selection failed: {e}")
                else:
                    # Use simple ATM fallback
                    selected_strikes = self._get_fallback_strikes(market_data, spot_price)
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('strike_selector')
                    self.component_health['strike_selector'] = 'degraded'
            
            # Step 2: Dual Weighted Greeks Calculation
            try:
                weighted_greeks = self.volume_oi_weighter.calculate_dual_weighted_greeks(
                    market_data, selected_strikes, spot_price, market_conditions
                )
                
                analysis_results['component_results']['dual_weighting'] = {
                    'weighted_greeks': weighted_greeks,
                    'alpha': self.volume_oi_weighter.oi_weight_alpha,
                    'beta': self.volume_oi_weighter.volume_weight_beta
                }
                self.component_health['volume_oi_weighter'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Dual weighting failed: {e}")
                if not self.fallback_on_component_failure:
                    return self._get_error_output(f"Dual weighting failed: {e}")
                else:
                    weighted_greeks = self._get_fallback_greeks(market_data, selected_strikes)
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('volume_oi_weighter')
                    self.component_health['volume_oi_weighter'] = 'degraded'
            
            # Step 3: Baseline Tracking (9:15 AM Logic)
            try:
                baselines = self.baseline_tracker.update_baselines(weighted_greeks)
                baseline_changes = self.baseline_tracker.calculate_baseline_changes(
                    weighted_greeks, baselines
                )
                
                analysis_results['component_results']['baseline_tracking'] = {
                    'baselines': baselines,
                    'baseline_changes': baseline_changes,
                    'baseline_summary': self.baseline_tracker.get_baseline_summary()
                }
                self.component_health['baseline_tracker'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Baseline tracking failed: {e}")
                if not self.fallback_on_component_failure:
                    return self._get_error_output(f"Baseline tracking failed: {e}")
                else:
                    baseline_changes = weighted_greeks  # Use raw Greeks as fallback
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('baseline_tracker')
                    self.component_health['baseline_tracker'] = 'degraded'
            
            # Step 4: DTE Adjustments
            try:
                dte_adjusted_greeks = self.dte_adjuster.apply_dte_adjustments(
                    baseline_changes, current_dte, market_conditions
                )
                
                analysis_results['component_results']['dte_adjustment'] = {
                    'dte_adjusted_greeks': dte_adjusted_greeks,
                    'dte_category': self.dte_adjuster._classify_dte_category(current_dte),
                    'dte_summary': self.dte_adjuster.get_dte_analysis_summary()
                }
                self.component_health['dte_adjuster'] = 'healthy'
                
            except Exception as e:
                logger.error(f"DTE adjustment failed: {e}")
                if not self.fallback_on_component_failure:
                    return self._get_error_output(f"DTE adjustment failed: {e}")
                else:
                    dte_adjusted_greeks = baseline_changes  # Use baseline changes as fallback
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('dte_adjuster')
                    self.component_health['dte_adjuster'] = 'degraded'
            
            # Step 5: ITM/OTM Analysis (Enhanced Feature)
            itm_sentiment = 0.0
            try:
                if self.itm_otm_analyzer.enable_itm_analysis:
                    itm_otm_analysis = self.itm_otm_analyzer.analyze_itm_otm_sentiment(
                        market_data, spot_price, current_time
                    )
                    
                    itm_sentiment = itm_otm_analysis.get('sentiment_scores', {}).get('combined_sentiment', 0.0)
                    
                    analysis_results['component_results']['itm_otm_analysis'] = itm_otm_analysis
                    self.component_health['itm_otm_analyzer'] = 'healthy'
                
            except Exception as e:
                logger.error(f"ITM/OTM analysis failed: {e}")
                analysis_results['orchestration_metadata']['fallbacks_used'].append('itm_otm_analyzer')
                self.component_health['itm_otm_analyzer'] = 'degraded'
            
            # Step 6: Greek Calculations with Market-Calibrated Normalization
            try:
                greek_contributions = self.greek_calculator.calculate_greek_contributions(
                    dte_adjusted_greeks
                )
                
                analysis_results['component_results']['greek_calculation'] = {
                    'greek_contributions': greek_contributions,
                    'calculation_summary': self.greek_calculator.get_calculation_summary()
                }
                self.component_health['greek_calculator'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Greek calculation failed: {e}")
                if not self.fallback_on_component_failure:
                    return self._get_error_output(f"Greek calculation failed: {e}")
                else:
                    greek_contributions = self._get_fallback_contributions(dte_adjusted_greeks)
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('greek_calculator')
                    self.component_health['greek_calculator'] = 'degraded'
            
            # Step 7: Final Sentiment Calculation
            final_sentiment = self._calculate_final_sentiment(
                greek_contributions, itm_sentiment, market_conditions
            )
            
            # Step 8: Sentiment Classification (7-level system)
            sentiment_classification = self._classify_sentiment(final_sentiment)
            
            # Step 9: Confidence Calculation
            confidence = self._calculate_analysis_confidence(
                analysis_results, selected_strikes, market_data
            )
            
            # Step 10: Record Analysis
            computation_time = (datetime.now() - start_time).total_seconds()
            self._record_analysis(analysis_results, final_sentiment, confidence)
            
            # Compile comprehensive output
            return IndicatorOutput(
                value=final_sentiment,
                confidence=confidence,
                metadata={
                    'sentiment_classification': sentiment_classification,
                    'component_results': analysis_results['component_results'],
                    'component_health': self.component_health.copy(),
                    'orchestration_metadata': analysis_results['orchestration_metadata'],
                    'market_conditions': market_conditions,
                    'selected_strikes_count': len(selected_strikes),
                    'itm_sentiment_contribution': itm_sentiment,
                    'method': 'modular_greek_sentiment_v2',
                    'architecture': 'component_orchestrated',
                    'enhanced_features': [
                        'dual_weighting_system',
                        'itm_otm_analysis', 
                        'baseline_tracking_9_15am',
                        'dte_adjustments',
                        'market_calibrated_normalization',
                        '7_level_classification'
                    ]
                },
                computation_time=computation_time
            )
            
        except Exception as e:
            logger.error(f"Error in GreekSentimentAnalyzer: {e}")
            return self._get_error_output(f"Analysis failed: {str(e)}")
    
    def _calculate_final_sentiment(self, 
                                 greek_contributions: Dict[str, float],
                                 itm_sentiment: float,
                                 market_conditions: Dict[str, Any]) -> float:
        """Calculate final sentiment score combining all components"""
        try:
            # Get adaptive weights (this could be enhanced with ML optimization)
            weights = self._get_adaptive_greek_weights(market_conditions)
            
            # Calculate weighted Greek sentiment
            greek_sentiment = 0.0
            total_weight = 0.0
            
            for greek, contribution in greek_contributions.items():
                if greek in weights:
                    weight = weights[greek]
                    greek_sentiment += contribution * weight
                    total_weight += weight
            
            # Normalize Greek sentiment
            if total_weight > 0:
                greek_sentiment /= total_weight
            
            # Combine with ITM sentiment (limited contribution)
            max_itm_contribution = 0.3  # 30% max from ITM analysis
            itm_contribution = np.clip(itm_sentiment * max_itm_contribution, 
                                     -max_itm_contribution, max_itm_contribution)
            
            final_sentiment = greek_sentiment + itm_contribution
            
            return np.clip(final_sentiment, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating final sentiment: {e}")
            return 0.0
    
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
    
    def _calculate_analysis_confidence(self, 
                                     analysis_results: Dict[str, Any],
                                     selected_strikes: List,
                                     market_data: pd.DataFrame) -> float:
        """Calculate confidence in the overall analysis"""
        try:
            confidence_factors = []
            
            # Component health confidence
            healthy_components = sum(1 for health in self.component_health.values() if health == 'healthy')
            total_components = len(self.component_health)
            component_confidence = healthy_components / total_components if total_components > 0 else 0.5
            confidence_factors.append(component_confidence * 0.3)
            
            # Strike coverage confidence
            strike_confidence = min(len(selected_strikes) / 10, 1.0)
            confidence_factors.append(strike_confidence * 0.2)
            
            # Data quality confidence
            total_volume = sum(
                row.get('ce_volume', 0) + row.get('pe_volume', 0)
                for _, row in market_data.iterrows()
            )
            volume_confidence = min(total_volume / 10000, 1.0)
            confidence_factors.append(volume_confidence * 0.2)
            
            # Greek consistency confidence
            greek_results = analysis_results.get('component_results', {}).get('greek_calculation', {})
            if greek_results:
                greek_contributions = greek_results.get('greek_contributions', {})
                if greek_contributions:
                    greek_values = list(greek_contributions.values())
                    greek_consistency = 1.0 - np.std(greek_values) if len(greek_values) > 1 else 0.5
                    confidence_factors.append(greek_consistency * 0.2)
            
            # Fallback penalty
            fallbacks_used = len(analysis_results.get('orchestration_metadata', {}).get('fallbacks_used', []))
            fallback_penalty = fallbacks_used * 0.1
            
            # Calculate final confidence
            base_confidence = sum(confidence_factors) if confidence_factors else 0.5
            final_confidence = max(0.1, base_confidence - fallback_penalty)
            
            return np.clip(final_confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    def _get_adaptive_greek_weights(self, market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Get adaptive Greek weights (placeholder for ML optimization)"""
        # Base weights (preserved from original)
        base_weights = {
            'delta': 1.2,   # 40% influence (normalized)
            'vega': 1.5,    # 50% influence (normalized) 
            'theta': 0.3,   # 10% influence (normalized)
            'gamma': 0.0    # Initially 0%, can be enabled
        }
        
        # Simple adaptation based on DTE (can be enhanced with ML)
        dte = market_conditions.get('dte', 30)
        if dte <= 7:  # Near expiry
            base_weights['theta'] *= 1.5
            base_weights['gamma'] = 0.5
        elif dte >= 30:  # Far expiry
            base_weights['vega'] *= 1.3
            
        # Normalize weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            return {k: v/total_weight for k, v in base_weights.items()}
        
        return base_weights
    
    def _record_analysis(self, 
                       analysis_results: Dict[str, Any],
                       sentiment_score: float,
                       confidence: float):
        """Record analysis for performance tracking"""
        try:
            record = {
                'timestamp': datetime.now(),
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'component_health': self.component_health.copy(),
                'fallbacks_used': analysis_results.get('orchestration_metadata', {}).get('fallbacks_used', []),
                'components_successful': len([h for h in self.component_health.values() if h == 'healthy'])
            }
            
            self.analysis_history.append(record)
            
            # Keep only last 100 analyses
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
                
        except Exception as e:
            logger.error(f"Error recording analysis: {e}")
    
    # Fallback methods for component failures
    def _get_fallback_strikes(self, market_data: pd.DataFrame, spot_price: float) -> List:
        """Get fallback strikes when strike selector fails"""
        # Simple ATM-focused fallback
        strikes = market_data['strike'].unique()
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        # Create minimal strike info
        from base.strike_selector_base import StrikeInfo
        
        fallback_strikes = []
        for option_type in ['CE', 'PE']:
            fallback_strikes.append(StrikeInfo(
                strike=atm_strike,
                distance_from_atm=abs(atm_strike - spot_price) / spot_price,
                weight=1.0,
                option_type=option_type,
                dte=30,
                selection_reason='fallback_atm',
                confidence=0.5
            ))
        
        return fallback_strikes
    
    def _get_fallback_greeks(self, market_data: pd.DataFrame, selected_strikes: List) -> Dict[str, float]:
        """Get fallback Greeks when dual weighting fails"""
        # Simple equal-weighted average
        greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        count = 0
        
        for strike_info in selected_strikes:
            strike_data = market_data[
                (market_data['strike'] == strike_info.strike) & 
                (market_data['option_type'] == strike_info.option_type)
            ]
            
            if not strike_data.empty:
                row = strike_data.iloc[0]
                option_type = strike_info.option_type
                
                if option_type == 'CE':
                    greeks['delta'] += row.get('ce_delta', 0)
                    greeks['gamma'] += row.get('ce_gamma', 0)
                    greeks['theta'] += row.get('ce_theta', 0)
                    greeks['vega'] += row.get('ce_vega', 0)
                else:
                    greeks['delta'] += row.get('pe_delta', 0)
                    greeks['gamma'] += row.get('pe_gamma', 0)
                    greeks['theta'] += row.get('pe_theta', 0)
                    greeks['vega'] += row.get('pe_vega', 0)
                
                count += 1
        
        if count > 0:
            for greek in greeks:
                greeks[greek] /= count
        
        return greeks
    
    def _get_fallback_contributions(self, dte_adjusted_greeks: Dict[str, float]) -> Dict[str, float]:
        """Get fallback Greek contributions when calculator fails"""
        # Simple clipping fallback
        contributions = {}
        for greek, value in dte_adjusted_greeks.items():
            contributions[greek] = np.clip(value, -1.0, 1.0)
        
        return contributions
    
    def _get_error_output(self, error_message: str) -> IndicatorOutput:
        """Get standardized error output"""
        return IndicatorOutput(
            value=0.0,
            confidence=0.0,
            metadata={
                'error': True,
                'error_message': error_message,
                'method': 'modular_greek_sentiment_v2',
                'component_health': self.component_health.copy()
            }
        )
    
    def get_component_health_status(self) -> Dict[str, Any]:
        """Get health status of all components"""
        return {
            'component_health': self.component_health.copy(),
            'overall_health': 'healthy' if all(h == 'healthy' for h in self.component_health.values()) else 'degraded',
            'healthy_components': len([h for h in self.component_health.values() if h == 'healthy']),
            'total_components': len(self.component_health),
            'recent_analyses': len(self.analysis_history),
            'fallback_enabled': self.fallback_on_component_failure
        }
    
    def reset_component_health(self):
        """Reset component health tracking"""
        self.component_health.clear()
        self.analysis_history.clear()
        logger.info("Component health tracking reset")