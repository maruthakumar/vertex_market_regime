"""
OI/PA Analysis V2 - Enhanced Open Interest and Price Action Analysis
===================================================================

Enhanced OI/PA analysis system with modular architecture providing:
- Corrected OI-Price relationship detection
- 5-type divergence detection system  
- Institutional vs retail flow analysis
- Mathematical correlation analysis with precision
- Session-based time weighting
- Multi-timeframe analysis coordination

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Modular OI/PA Architecture
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from base.base_indicator import BaseIndicator, IndicatorConfig, IndicatorOutput
from .oi_pa_analysis.oi_pa_analyzer import OIPAAnalyzer

logger = logging.getLogger(__name__)

class OIPAAnalysisV2(BaseIndicator):
    """
    Enhanced OI/PA Analysis V2 with modular architecture
    
    This is the main interface for the OI/PA analysis system, providing
    a clean API while delegating to the modular analyzer components.
    
    Key Features:
    - Corrected OI-Price relationship logic (both calls and puts follow same pattern)
    - 5-type divergence detection (Pattern, OI-Price, Call-Put, Institutional-Retail, Cross-Strike)
    - Institutional vs retail flow detection using size and pattern analysis
    - Mathematical correlation analysis with Pearson correlation >0.80 threshold
    - Time-decay weighting using exponential decay formula exp(-λ × (T-t))
    - Session-based weighting for time-sensitive analysis
    - Multi-timeframe coordination and confirmation
    
    Enhanced Accuracy Features:
    - Mathematical precision validation (±0.001 tolerance)
    - Pattern similarity scoring using cosine similarity
    - Historical pattern matching and validation
    - Component health monitoring and fallback mechanisms
    """
    
    def __init__(self, config: IndicatorConfig):
        """Initialize OI/PA Analysis V2"""
        super().__init__(config)
        
        # Initialize the modular analyzer
        self.analyzer = OIPAAnalyzer(config)
        
        logger.info("OIPAAnalysisV2 initialized with enhanced modular architecture")
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return self.analyzer.get_required_columns()
    
    def analyze(self, market_data: pd.DataFrame, **kwargs) -> IndicatorOutput:
        """
        Perform comprehensive OI/PA analysis
        
        Args:
            market_data: Option market data DataFrame
            **kwargs: Additional parameters:
                - spot_price: Current underlying price
                - dte: Days to expiry
                - timestamp: Current timestamp
                - historical_data: Historical data for comparison
                - volatility: Current volatility estimate
                - volume_ratio: Volume relative to average
                
        Returns:
            IndicatorOutput: Comprehensive OI/PA analysis results
        """
        try:
            # Validate input data
            is_valid, errors = self.analyzer.validate_data(market_data)
            if not is_valid:
                return IndicatorOutput(
                    value=0.0,
                    confidence=0.0,
                    metadata={
                        'error': True,
                        'validation_errors': errors,
                        'method': 'oi_pa_analysis_v2'
                    }
                )
            
            # Delegate to modular analyzer
            result = self.analyzer.analyze(market_data, **kwargs)
            
            # Enhance metadata with V2 specific information
            if result.metadata:
                result.metadata.update({
                    'version': '2.0.0',
                    'architecture': 'modular_enhanced',
                    'analysis_type': 'oi_pa_comprehensive',
                    'features_enabled': self._get_enabled_features()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in OI/PA Analysis V2: {e}")
            return IndicatorOutput(
                value=0.0,
                confidence=0.0,
                metadata={
                    'error': True,
                    'error_message': str(e),
                    'method': 'oi_pa_analysis_v2'
                }
            )
    
    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled features"""
        return [
            'corrected_oi_price_relationships',
            'five_type_divergence_detection',
            'institutional_retail_flow_analysis', 
            'mathematical_correlation_analysis',
            'session_based_time_weighting',
            'multi_timeframe_coordination',
            'pattern_similarity_scoring',
            'component_health_monitoring',
            'fallback_mechanisms',
            'mathematical_precision_validation'
        ]
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all analyzer components"""
        return self.analyzer.get_component_health_status()
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        try:
            component_status = self.get_component_status()
            
            # Get component-specific summaries
            summaries = {}
            
            # Pattern detector summary
            if hasattr(self.analyzer.oi_pattern_detector, 'get_pattern_detection_summary'):
                summaries['pattern_detection'] = self.analyzer.oi_pattern_detector.get_pattern_detection_summary()
            
            # Divergence detector summary  
            if hasattr(self.analyzer.divergence_detector, 'get_divergence_detection_summary'):
                summaries['divergence_detection'] = self.analyzer.divergence_detector.get_divergence_detection_summary()
            
            # Volume flow analyzer summary
            if hasattr(self.analyzer.volume_flow_analyzer, 'get_flow_analysis_summary'):
                summaries['volume_flow_analysis'] = self.analyzer.volume_flow_analyzer.get_flow_analysis_summary()
            
            # Correlation analyzer summary
            if hasattr(self.analyzer.correlation_analyzer, 'get_correlation_analysis_summary'):
                summaries['correlation_analysis'] = self.analyzer.correlation_analyzer.get_correlation_analysis_summary()
            
            # Session weight manager summary
            if hasattr(self.analyzer.session_weight_manager, 'get_session_weight_summary'):
                summaries['session_weighting'] = self.analyzer.session_weight_manager.get_session_weight_summary()
            
            return {
                'overall_status': component_status,
                'component_summaries': summaries,
                'features_enabled': self._get_enabled_features(),
                'version': '2.0.0',
                'architecture': 'modular_enhanced'
            }
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return {'error': str(e)}
    
    def reset_analysis_state(self):
        """Reset all component states and histories"""
        try:
            self.analyzer.reset_component_health()
            logger.info("OI/PA Analysis V2 state reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting analysis state: {e}")
    
    def get_optimal_session_timing(self) -> Dict[str, Any]:
        """Get optimal session timing recommendations"""
        try:
            return self.analyzer.session_weight_manager.get_optimal_session_timing()
            
        except Exception as e:
            logger.error(f"Error getting optimal session timing: {e}")
            return {'error': str(e)}
    
    def analyze_institutional_patterns(self) -> Dict[str, Any]:
        """Analyze institutional trading patterns"""
        try:
            return self.analyzer.volume_flow_analyzer.analyze_institutional_patterns()
            
        except Exception as e:
            logger.error(f"Error analyzing institutional patterns: {e}")
            return {'error': str(e)}
    
    def analyze_correlation_trends(self) -> Dict[str, Any]:
        """Analyze correlation trends over time"""
        try:
            return self.analyzer.correlation_analyzer.analyze_correlation_trends()
            
        except Exception as e:
            logger.error(f"Error analyzing correlation trends: {e}")
            return {'error': str(e)}

# Convenience function for quick analysis
def analyze_oi_pa_v2(market_data: pd.DataFrame, 
                     config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> IndicatorOutput:
    """
    Convenience function for quick OI/PA analysis
    
    Args:
        market_data: Option market data
        config: Optional configuration parameters
        **kwargs: Additional analysis parameters
        
    Returns:
        IndicatorOutput: Analysis results
    """
    try:
        # Create default config if not provided
        if config is None:
            config = {
                'strike_selection_strategy': 'adaptive_range',
                'parameters': {}
            }
        
        # Convert dict config to IndicatorConfig
        from base.base_indicator import IndicatorConfig
        indicator_config = IndicatorConfig(
            name='oi_pa_analysis_v2',
            strike_selection_strategy=config.get('strike_selection_strategy', 'adaptive_range'),
            parameters=config.get('parameters', {})
        )
        
        # Create and run analyzer
        analyzer = OIPAAnalysisV2(indicator_config)
        return analyzer.analyze(market_data, **kwargs)
        
    except Exception as e:
        logger.error(f"Error in convenience function analyze_oi_pa_v2: {e}")
        return IndicatorOutput(
            value=0.0,
            confidence=0.0,
            metadata={
                'error': True,
                'error_message': str(e),
                'method': 'oi_pa_analysis_v2_convenience'
            }
        )