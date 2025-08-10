"""
DEPRECATED: Resistance Analyzer for Triple Straddle Components

⚠️  WARNING: This module is DEPRECATED and has been replaced by the enhanced 10-component version.

The legacy resistance analyzer has been upgraded to handle all 10 components:
- 6 Individual Components: ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
- 3 Individual Straddles: ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE
- 1 Combined Triple: COMBINED_TRIPLE_STRADDLE

Migration Guide:
OLD: from ..core.resistance_analyzer import ResistanceAnalyzer
NEW: from ..enhanced.enhanced_resistance_analyzer import Enhanced10ComponentResistanceAnalyzer

The enhanced version provides:
- 10-component resistance analysis (vs 6-component)
- Multi-timeframe level detection (3,5,10,15 min)
- Advanced confluence zone detection
- Pattern-relevant level identification
- Real-time breakout/reversal signals

Legacy backup: resistance_analyzer_legacy_backup.py
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

# Import enhanced version
try:
    from ..enhanced.enhanced_resistance_analyzer import (
        Enhanced10ComponentResistanceAnalyzer,
        EnhancedResistanceAnalysisResult,
        ResistanceLevel,
        ComponentResistanceResult
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

logger = logging.getLogger(__name__)

# Legacy compatibility classes
@dataclass
class ResistanceAnalysisResult:
    """DEPRECATED: Legacy resistance result - use EnhancedResistanceAnalysisResult instead"""
    timestamp: datetime
    support_levels: List[float]
    resistance_levels: List[float]
    current_price: float
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    
    def __post_init__(self):
        warnings.warn(
            "ResistanceAnalysisResult is deprecated. Use EnhancedResistanceAnalysisResult from enhanced.enhanced_resistance_analyzer",
            DeprecationWarning,
            stacklevel=2
        )


class ResistanceAnalyzer:
    """
    DEPRECATED: Resistance Analyzer for Triple Straddle Components
    
    ⚠️  This class is DEPRECATED and has been replaced by Enhanced10ComponentResistanceAnalyzer
    
    Please migrate to the enhanced version:
    from ..enhanced.enhanced_resistance_analyzer import Enhanced10ComponentResistanceAnalyzer
    
    The enhanced version provides:
    - 10-component resistance analysis (vs 6-component)
    - Multi-timeframe level detection
    - Advanced confluence zone detection
    - Pattern-relevant level identification
    - Real-time breakout/reversal signals
    """
    
    def __init__(self, window_sizes: List[int] = None, window_manager=None, config: Optional[Dict] = None):
        """Initialize deprecated resistance analyzer with migration warning"""
        
        # Issue deprecation warning
        warnings.warn(
            "ResistanceAnalyzer is deprecated. "
            "Use Enhanced10ComponentResistanceAnalyzer from enhanced.enhanced_resistance_analyzer. "
            "The enhanced version supports all 10 components with advanced pattern detection.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning("Using deprecated ResistanceAnalyzer - please migrate to Enhanced10ComponentResistanceAnalyzer")
        
        self.window_sizes = window_sizes or [3, 5, 10, 15]
        self.window_manager = window_manager
        self.config = config or {}
        
        # Legacy setup for 6 components only
        self.legacy_components = ['ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE']
        
        # Try to use enhanced version if available
        if ENHANCED_AVAILABLE:
            logger.info("Enhanced10ComponentResistanceAnalyzer available - consider migrating")
            self._enhanced_analyzer = Enhanced10ComponentResistanceAnalyzer(window_manager, config)
            self._use_enhanced = True
        else:
            logger.warning("Enhanced10ComponentResistanceAnalyzer not available - using legacy mode")
            self._use_enhanced = False
            self._init_legacy_mode()
    
    def _init_legacy_mode(self):
        """Initialize in legacy mode"""
        self.support_levels = {}
        self.resistance_levels = {}
        
        for component in self.legacy_components:
            self.support_levels[component] = []
            self.resistance_levels[component] = []
    
    def analyze(self, market_data: Dict[str, Any], timestamp: Optional[datetime] = None) -> Optional[ResistanceAnalysisResult]:
        """
        DEPRECATED: Analyze resistance levels
        
        Returns legacy result for first component only. For full 10-component analysis,
        use Enhanced10ComponentResistanceAnalyzer.analyze_all_components()
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if self._use_enhanced:
            # Use enhanced version but return legacy-compatible result
            enhanced_result = self._enhanced_analyzer.analyze_all_components(market_data, timestamp)
            
            if enhanced_result and enhanced_result.component_results:
                # Get first component for legacy compatibility
                first_component = list(enhanced_result.component_results.keys())[0]
                component_result = enhanced_result.component_results[first_component]
                
                # Convert to legacy format
                support_levels = [level.level for level in component_result.support_levels]
                resistance_levels = [level.level for level in component_result.resistance_levels]
                
                nearest_support = component_result.nearest_support.level if component_result.nearest_support else None
                nearest_resistance = component_result.nearest_resistance.level if component_result.nearest_resistance else None
                
                return ResistanceAnalysisResult(
                    timestamp=timestamp,
                    support_levels=support_levels,
                    resistance_levels=resistance_levels,
                    current_price=component_result.current_price,
                    nearest_support=nearest_support,
                    nearest_resistance=nearest_resistance
                )
        
        # Legacy mode fallback
        logger.warning("Running in legacy resistance analysis mode - limited functionality")
        
        # Simple legacy analysis for first component
        first_component = self.legacy_components[0]
        current_price = market_data.get(first_component, 0.0)
        
        return ResistanceAnalysisResult(
            timestamp=timestamp,
            support_levels=[],
            resistance_levels=[],
            current_price=current_price,
            nearest_support=None,
            nearest_resistance=None
        )
    
    def get_straddle_adjustments(self, analysis_result) -> Dict[str, Any]:
        """DEPRECATED: Get straddle adjustments"""
        if self._use_enhanced and hasattr(analysis_result, 'breakout_signals'):
            # Return enhanced signals as legacy adjustments
            return {
                'enhanced_available': True,
                'breakout_signals': len(analysis_result.breakout_signals),
                'reversal_signals': len(analysis_result.reversal_signals),
                'migration_recommended': True
            }
        
        return {
            'legacy_mode': True,
            'migration_required': True,
            'message': 'Please migrate to Enhanced10ComponentResistanceAnalyzer'
        }
    
    def get_regime_contribution(self, analysis_result) -> Dict[str, float]:
        """DEPRECATED: Get regime contribution"""
        if self._use_enhanced and hasattr(analysis_result, 'regime_indicators'):
            # Return subset of regime indicators for legacy compatibility
            regime_indicators = analysis_result.regime_indicators
            return {
                'resistance_regime': regime_indicators.get('overall_regime_strength', 0.5),
                'enhanced_available': True
            }
        
        return {
            'resistance_regime': 0.5,
            'legacy_mode': True
        }
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """DEPRECATED: Get analyzer status"""
        if self._use_enhanced:
            enhanced_summary = self._enhanced_analyzer.get_analyzer_summary()
            return {
                'legacy_mode': False,
                'enhanced_available': True,
                'migration_recommended': True,
                'enhanced_summary': enhanced_summary,
                'components_analyzed': 10  # Enhanced version
            }
        
        return {
            'legacy_mode': True,
            'enhanced_available': False,
            'migration_required': True,
            'components_analyzed': 6,  # Legacy version
            'message': 'Please migrate to Enhanced10ComponentResistanceAnalyzer'
        }


# Compatibility aliases for migration
ResistanceAnalysisEngine = ResistanceAnalyzer  # Common alias

# Migration helper function
def migrate_to_enhanced(window_manager=None, config: Optional[Dict] = None) -> 'Enhanced10ComponentResistanceAnalyzer':
    """
    Migration helper to create Enhanced10ComponentResistanceAnalyzer
    
    Args:
        window_manager: Window manager instance
        config: Analyzer configuration
        
    Returns:
        Enhanced10ComponentResistanceAnalyzer instance
    """
    if not ENHANCED_AVAILABLE:
        raise ImportError("Enhanced10ComponentResistanceAnalyzer not available")
    
    logger.info("Migrating from legacy to enhanced 10-component resistance analyzer")
    return Enhanced10ComponentResistanceAnalyzer(window_manager, config)


# Module deprecation notice
def __deprecated_module_warning():
    """Issue module-level deprecation warning"""
    warnings.warn(
        "Module 'resistance_analyzer' is deprecated. "
        "Use 'enhanced.enhanced_resistance_analyzer' instead. "
        "Enhanced version supports 10-component analysis with advanced pattern detection.",
        DeprecationWarning,
        stacklevel=3
    )

# Issue warning when module is imported
__deprecated_module_warning()