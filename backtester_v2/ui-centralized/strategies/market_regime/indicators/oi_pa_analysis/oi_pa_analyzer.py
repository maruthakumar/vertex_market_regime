"""
OI/PA Analyzer - Main Orchestrator for Open Interest and Price Action Analysis
=============================================================================

Main analyzer that orchestrates all OI/PA analysis components:
- OI Pattern Detection and Classification
- Divergence Detection (5 types)
- Volume Flow Analysis for institutional detection
- Correlation Analysis with mathematical precision
- Session Weight Management for time-based analysis

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Modular OI/PA Orchestrator
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from base.base_indicator import BaseIndicator, IndicatorConfig, IndicatorOutput, IndicatorState
from base.strike_selector_base import create_strike_selector

from .oi_pattern_detector import OIPatternDetector, OIPattern
from .divergence_detector import DivergenceDetector, DivergenceType
from .volume_flow_analyzer import VolumeFlowAnalyzer
from .correlation_analyzer import CorrelationAnalyzer
from .session_weight_manager import SessionWeightManager

logger = logging.getLogger(__name__)

class OIPAAnalyzer(BaseIndicator):
    """
    Main OI/PA Analyzer using modular architecture
    
    This orchestrator coordinates all specialized OI/PA components to produce
    comprehensive open interest and price action analysis with enhanced features.
    
    Enhanced Features:
    - Modular component architecture
    - Corrected OI-Price relationship detection
    - 5-type divergence detection system
    - Institutional vs retail flow analysis
    - Mathematical correlation analysis with precision
    - Session-based time weighting
    - Multi-timeframe analysis coordination
    """
    
    def __init__(self, config: IndicatorConfig):
        """Initialize OI/PA Analyzer"""
        super().__init__(config)
        
        # Extract component configurations
        params = config.parameters
        
        # Initialize specialized components
        self.oi_pattern_detector = OIPatternDetector(
            params.get('pattern_detector_config', {})
        )
        
        self.divergence_detector = DivergenceDetector(
            params.get('divergence_detector_config', {})
        )
        
        self.volume_flow_analyzer = VolumeFlowAnalyzer(
            params.get('volume_flow_config', {})
        )
        
        self.correlation_analyzer = CorrelationAnalyzer(
            params.get('correlation_config', {})
        )
        
        self.session_weight_manager = SessionWeightManager(
            params.get('session_weight_config', {})
        )
        
        # OI/PA analysis parameters
        self.enable_multi_timeframe = params.get('enable_multi_timeframe', True)
        self.primary_timeframe_weight = params.get('primary_timeframe_weight', 0.7)
        self.confirmation_timeframe_weight = params.get('confirmation_timeframe_weight', 0.3)
        
        # Signal combination weights
        self.pattern_weight = params.get('pattern_weight', 0.3)
        self.divergence_weight = params.get('divergence_weight', 0.25)
        self.flow_weight = params.get('flow_weight', 0.25)
        self.correlation_weight = params.get('correlation_weight', 0.2)
        
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
        logger.info("OIPAAnalyzer initialized with modular architecture")
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return [
            'strike', 'option_type', 'expiry_date', 'timestamp',
            'ce_oi', 'pe_oi', 'ce_volume', 'pe_volume',
            'ce_ltp', 'pe_ltp', 'ce_price', 'pe_price',
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
        
        # Validate OI and Volume data
        oi_cols = ['ce_oi', 'pe_oi']
        volume_cols = ['ce_volume', 'pe_volume']
        
        for col in oi_cols + volume_cols:
            if col in market_data.columns:
                if market_data[col].sum() == 0:
                    errors.append(f"All {col} values are zero - invalid data")
        
        return len(errors) == 0, errors
    
    def analyze(self, market_data: pd.DataFrame, **kwargs) -> IndicatorOutput:
        """
        Main analysis orchestrating all OI/PA components
        
        Args:
            market_data: Option market data
            **kwargs: Additional parameters including historical_data
            
        Returns:
            IndicatorOutput: Comprehensive OI/PA analysis
        """
        try:
            start_time = datetime.now()
            
            # Extract parameters
            spot_price = kwargs.get('spot_price') or market_data['underlying_price'].iloc[0]
            current_dte = kwargs.get('dte') or market_data['dte'].iloc[0]
            current_time = kwargs.get('timestamp', datetime.now())
            historical_data = kwargs.get('historical_data')
            
            # Market conditions for enhanced analysis
            market_conditions = {
                'spot_price': spot_price,
                'dte': current_dte,
                'timestamp': current_time,
                'hour': current_time.hour,
                'weekday': current_time.weekday(),
                'volatility': kwargs.get('volatility', 0.2),
                'volume_ratio': kwargs.get('volume_ratio', 1.0)
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
                    market_data, spot_price, dte=current_dte
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
                    selected_strikes = self._get_fallback_strikes(market_data, spot_price)
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('strike_selector')
                    self.component_health['strike_selector'] = 'degraded'
            
            # Step 2: OI Pattern Detection
            try:
                pattern_results = self.oi_pattern_detector.detect_oi_patterns(
                    market_data, selected_strikes, spot_price, historical_data
                )
                
                analysis_results['component_results']['pattern_detection'] = pattern_results
                self.component_health['oi_pattern_detector'] = 'healthy'
                
            except Exception as e:
                logger.error(f"OI pattern detection failed: {e}")
                if not self.fallback_on_component_failure:
                    return self._get_error_output(f"OI pattern detection failed: {e}")
                else:
                    pattern_results = self._get_fallback_pattern_results()
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('oi_pattern_detector')
                    self.component_health['oi_pattern_detector'] = 'degraded'
            
            # Step 3: Volume Flow Analysis
            try:
                flow_results = self.volume_flow_analyzer.analyze_volume_flows(
                    market_data, selected_strikes, current_time
                )
                
                analysis_results['component_results']['volume_flow_analysis'] = {
                    'institutional_flow': flow_results.institutional_flow,
                    'retail_flow': flow_results.retail_flow,
                    'flow_sentiment': flow_results.flow_sentiment,
                    'institutional_ratio': flow_results.institutional_ratio,
                    'flow_divergence': flow_results.flow_divergence,
                    'flow_quality': flow_results.flow_quality
                }
                self.component_health['volume_flow_analyzer'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Volume flow analysis failed: {e}")
                if not self.fallback_on_component_failure:
                    return self._get_error_output(f"Volume flow analysis failed: {e}")
                else:
                    flow_results = self._get_fallback_flow_results()
                    analysis_results['orchestration_metadata']['fallbacks_used'].append('volume_flow_analyzer')
                    self.component_health['volume_flow_analyzer'] = 'degraded'
            
            # Step 4: Divergence Detection
            try:
                divergence_results = self.divergence_detector.detect_all_divergences(
                    market_data, pattern_results, 
                    analysis_results['component_results'].get('volume_flow_analysis', {}),
                    spot_price, historical_data
                )
                
                analysis_results['component_results']['divergence_detection'] = divergence_results
                self.component_health['divergence_detector'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Divergence detection failed: {e}")
                analysis_results['orchestration_metadata']['fallbacks_used'].append('divergence_detector')
                self.component_health['divergence_detector'] = 'degraded'
                divergence_results = self._get_fallback_divergence_results()
            
            # Step 5: Correlation Analysis
            try:
                correlation_results = self.correlation_analyzer.analyze_correlations(
                    market_data, historical_data
                )
                
                analysis_results['component_results']['correlation_analysis'] = {
                    'pearson_correlation': correlation_results.pearson_correlation,
                    'correlation_confidence': correlation_results.correlation_confidence,
                    'pattern_similarity_score': correlation_results.pattern_similarity_score,
                    'time_decay_weight': correlation_results.time_decay_weight,
                    'mathematical_accuracy': correlation_results.mathematical_accuracy,
                    'correlation_threshold_met': correlation_results.correlation_threshold_met
                }
                self.component_health['correlation_analyzer'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}")
                analysis_results['orchestration_metadata']['fallbacks_used'].append('correlation_analyzer')
                self.component_health['correlation_analyzer'] = 'degraded'
                correlation_results = self._get_fallback_correlation_results()
            
            # Step 6: Session Weight Calculation
            try:
                session_weight = self.session_weight_manager.get_session_weight(
                    current_time, market_conditions
                )
                
                analysis_results['component_results']['session_weighting'] = {
                    'session_weight': session_weight,
                    'session_analysis': self.session_weight_manager.get_session_analysis(current_time)
                }
                self.component_health['session_weight_manager'] = 'healthy'
                
            except Exception as e:
                logger.error(f"Session weight calculation failed: {e}")
                session_weight = 1.0
                analysis_results['orchestration_metadata']['fallbacks_used'].append('session_weight_manager')
                self.component_health['session_weight_manager'] = 'degraded'
            
            # Step 7: Signal Combination and Final Scoring
            final_signal = self._calculate_final_oi_pa_signal(
                pattern_results, flow_results, divergence_results, 
                correlation_results, session_weight, market_conditions
            )
            
            # Step 8: Confidence Calculation
            confidence = self._calculate_analysis_confidence(
                analysis_results, selected_strikes, market_data
            )
            
            # Step 9: Multi-timeframe Analysis (if enabled)
            multi_timeframe_results = None
            if self.enable_multi_timeframe and historical_data is not None:
                multi_timeframe_results = self._perform_multi_timeframe_analysis(
                    market_data, historical_data, final_signal
                )
            
            # Step 10: Record Analysis
            computation_time = (datetime.now() - start_time).total_seconds()
            self._record_analysis(analysis_results, final_signal, confidence)
            
            # Compile comprehensive output
            return IndicatorOutput(
                value=final_signal,
                confidence=confidence,
                metadata={
                    'component_results': analysis_results['component_results'],
                    'component_health': self.component_health.copy(),
                    'orchestration_metadata': analysis_results['orchestration_metadata'],
                    'market_conditions': market_conditions,
                    'selected_strikes_count': len(selected_strikes),
                    'session_weight': session_weight,
                    'multi_timeframe_results': multi_timeframe_results,
                    'method': 'modular_oi_pa_analysis_v2',
                    'architecture': 'component_orchestrated',
                    'enhanced_features': [
                        'corrected_oi_price_patterns',
                        '5_type_divergence_detection',
                        'institutional_retail_flow_analysis',
                        'mathematical_correlation_analysis',
                        'session_based_weighting',
                        'multi_timeframe_coordination'
                    ]
                },
                computation_time=computation_time
            )
            
        except Exception as e:
            logger.error(f"Error in OIPAAnalyzer: {e}")
            return self._get_error_output(f"Analysis failed: {str(e)}")
    
    def _calculate_final_oi_pa_signal(self,
                                    pattern_results: Dict[str, Any],
                                    flow_results: Any,
                                    divergence_results: Dict[str, Any],
                                    correlation_results: Any,
                                    session_weight: float,
                                    market_conditions: Dict[str, Any]) -> float:
        """Calculate final OI/PA signal combining all components"""
        try:
            # Extract component signals
            pattern_signal = self._extract_pattern_signal(pattern_results)
            flow_signal = getattr(flow_results, 'flow_sentiment', 0.0)
            divergence_signal = self._extract_divergence_signal(divergence_results)
            correlation_signal = getattr(correlation_results, 'pearson_correlation', 0.0)
            
            # Apply component weights
            weighted_signal = (
                self.pattern_weight * pattern_signal +
                self.flow_weight * flow_signal +
                self.divergence_weight * divergence_signal +
                self.correlation_weight * correlation_signal
            )
            
            # Apply session weighting
            session_adjusted_signal = weighted_signal * session_weight
            
            # Apply market condition adjustments
            final_signal = self._apply_market_condition_adjustments(
                session_adjusted_signal, market_conditions
            )
            
            return np.clip(final_signal, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating final OI/PA signal: {e}")
            return 0.0
    
    def _extract_pattern_signal(self, pattern_results: Dict[str, Any]) -> float:
        """Extract signal from pattern detection results"""
        try:
            aggregated = pattern_results.get('aggregated_pattern', {})
            dominant_pattern = aggregated.get('dominant_pattern')
            confidence = aggregated.get('aggregate_confidence', 0)
            
            if dominant_pattern == OIPattern.LONG_BUILD_UP:
                return confidence
            elif dominant_pattern == OIPattern.SHORT_COVERING:
                return confidence * 0.8  # Slightly weaker bullish signal
            elif dominant_pattern == OIPattern.SHORT_BUILD_UP:
                return -confidence
            elif dominant_pattern == OIPattern.LONG_UNWINDING:
                return -confidence * 0.8  # Slightly weaker bearish signal
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error extracting pattern signal: {e}")
            return 0.0
    
    def _extract_divergence_signal(self, divergence_results: Dict[str, Any]) -> float:
        """Extract signal from divergence detection results"""
        try:
            overall_metrics = divergence_results.get('overall_metrics', {})
            divergence_strength = overall_metrics.get('overall_divergence_strength', 0)
            
            # Divergence typically indicates potential reversal
            # Higher divergence = higher reversal probability
            return -divergence_strength * 0.5  # Scale down divergence impact
            
        except Exception as e:
            logger.error(f"Error extracting divergence signal: {e}")
            return 0.0
    
    def _apply_market_condition_adjustments(self, 
                                          signal: float,
                                          market_conditions: Dict[str, Any]) -> float:
        """Apply market condition adjustments to signal"""
        try:
            adjusted_signal = signal
            
            # DTE adjustments
            dte = market_conditions.get('dte', 30)
            if dte <= 7:  # Near expiry - reduce signal strength
                adjusted_signal *= 0.8
            elif dte >= 45:  # Far expiry - reduce signal strength
                adjusted_signal *= 0.9
            
            # Volatility adjustments
            volatility = market_conditions.get('volatility', 0.2)
            if volatility > 0.4:  # High volatility - reduce confidence
                adjusted_signal *= 0.85
            
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying market condition adjustments: {e}")
            return signal
    
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
            total_oi = market_data['ce_oi'].sum() + market_data['pe_oi'].sum()
            total_volume = market_data['ce_volume'].sum() + market_data['pe_volume'].sum()
            data_quality = min((total_oi + total_volume) / 50000, 1.0)
            confidence_factors.append(data_quality * 0.2)
            
            # Component agreement confidence
            component_results = analysis_results.get('component_results', {})
            agreement_score = self._calculate_component_agreement(component_results)
            confidence_factors.append(agreement_score * 0.2)
            
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
    
    def _calculate_component_agreement(self, component_results: Dict[str, Any]) -> float:
        """Calculate agreement between component signals"""
        try:
            signals = []
            
            # Extract signals from each component
            pattern_results = component_results.get('pattern_detection', {})
            if pattern_results:
                signals.append(self._extract_pattern_signal(pattern_results))
            
            flow_analysis = component_results.get('volume_flow_analysis', {})
            if flow_analysis:
                signals.append(flow_analysis.get('flow_sentiment', 0))
            
            correlation_analysis = component_results.get('correlation_analysis', {})
            if correlation_analysis:
                signals.append(correlation_analysis.get('pearson_correlation', 0))
            
            if len(signals) < 2:
                return 0.5
            
            # Calculate agreement as 1 - standard deviation of signals
            agreement = 1.0 - np.std(signals)
            return max(0.0, agreement)
            
        except Exception as e:
            logger.error(f"Error calculating component agreement: {e}")
            return 0.5
    
    def _perform_multi_timeframe_analysis(self,
                                        current_data: pd.DataFrame,
                                        historical_data: pd.DataFrame,
                                        primary_signal: float) -> Dict[str, Any]:
        """Perform multi-timeframe analysis for signal confirmation"""
        try:
            # This would implement multi-timeframe analysis
            # For now, return a placeholder structure
            return {
                'primary_signal': primary_signal,
                'confirmation_signal': primary_signal * 0.8,  # Simplified
                'timeframe_agreement': 0.8,
                'combined_signal': primary_signal * self.primary_timeframe_weight + 
                                 (primary_signal * 0.8) * self.confirmation_timeframe_weight
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}
    
    def _record_analysis(self,
                       analysis_results: Dict[str, Any],
                       signal: float,
                       confidence: float):
        """Record analysis for performance tracking"""
        try:
            record = {
                'timestamp': datetime.now(),
                'signal': signal,
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
        from base.strike_selector_base import StrikeInfo
        
        strikes = market_data['strike'].unique()
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
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
    
    def _get_fallback_pattern_results(self) -> Dict[str, Any]:
        """Get fallback pattern results"""
        return {
            'aggregated_pattern': {
                'dominant_pattern': OIPattern.NEUTRAL,
                'aggregate_confidence': 0.0,
                'aggregate_signal_strength': 0.0,
                'pattern_consensus': 0.0
            },
            'strike_patterns': [],
            'pattern_distribution': {}
        }
    
    def _get_fallback_flow_results(self):
        """Get fallback flow results"""
        from .volume_flow_analyzer import FlowAnalysisResult
        return FlowAnalysisResult(
            institutional_flow={'calls': 0, 'puts': 0, 'total': 0},
            retail_flow={'calls': 0, 'puts': 0, 'total': 0},
            flow_sentiment=0.0,
            institutional_ratio=0.0,
            flow_divergence=0.0,
            flow_quality=0.0
        )
    
    def _get_fallback_divergence_results(self) -> Dict[str, Any]:
        """Get fallback divergence results"""
        return {
            'divergences_detected': [],
            'divergence_scores': {},
            'overall_metrics': {
                'overall_divergence_strength': 0.0,
                'overall_confidence': 0.0,
                'divergence_concentration': 0.0,
                'risk_level': 'low'
            }
        }
    
    def _get_fallback_correlation_results(self):
        """Get fallback correlation results"""
        from .correlation_analyzer import CorrelationAnalysisResult
        return CorrelationAnalysisResult(
            pearson_correlation=0.0,
            correlation_confidence=0.0,
            pattern_similarity_score=0.0,
            time_decay_weight=1.0,
            mathematical_accuracy=False,
            correlation_threshold_met=False,
            historical_pattern_match=None
        )
    
    def _get_error_output(self, error_message: str) -> IndicatorOutput:
        """Get standardized error output"""
        return IndicatorOutput(
            value=0.0,
            confidence=0.0,
            metadata={
                'error': True,
                'error_message': error_message,
                'method': 'modular_oi_pa_analysis_v2',
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