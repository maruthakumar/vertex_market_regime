"""
DEPRECATED: 6×6 Correlation Matrix for Triple Straddle Components

⚠️  WARNING: This module is DEPRECATED and has been replaced by the enhanced 10×10 version.

The legacy 6×6 correlation matrix has been upgraded to handle all 10 components:
- 6 Individual Components: ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
- 3 Individual Straddles: ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE
- 1 Combined Triple: COMBINED_TRIPLE_STRADDLE

Migration Guide:
OLD: from ..rolling.correlation_matrix import CorrelationMatrix
NEW: from ..enhanced.enhanced_correlation_matrix import Enhanced10x10CorrelationMatrix

The enhanced version provides:
- 10×10 correlation matrix (45 unique pairs vs 15)
- Multi-timeframe pattern detection
- Advanced confluence detection
- Pattern-based correlation analysis
- Real-time correlation pattern alerts

Legacy backup: correlation_matrix_legacy_backup.py
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

# Import enhanced version
try:
    from ..enhanced.enhanced_correlation_matrix import (
        Enhanced10x10CorrelationMatrix, 
        EnhancedCorrelationResult,
        CorrelationPattern
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

logger = logging.getLogger(__name__)

# Legacy compatibility classes
@dataclass
class CorrelationResult:
    """DEPRECATED: Legacy correlation result - use EnhancedCorrelationResult instead"""
    matrix: np.ndarray
    component_names: List[str]
    timestamp: pd.Timestamp
    window_size: int
    avg_correlation: float
    max_correlation: float
    min_correlation: float
    
    def __post_init__(self):
        warnings.warn(
            "CorrelationResult is deprecated. Use EnhancedCorrelationResult from enhanced.enhanced_correlation_matrix",
            DeprecationWarning,
            stacklevel=2
        )


class CorrelationMatrix:
    """
    DEPRECATED: 6×6 Rolling Correlation Matrix Manager
    
    ⚠️  This class is DEPRECATED and has been replaced by Enhanced10x10CorrelationMatrix
    
    Please migrate to the enhanced version:
    from ..enhanced.enhanced_correlation_matrix import Enhanced10x10CorrelationMatrix
    
    The enhanced version provides:
    - 10×10 correlation matrix (vs 6×6)
    - Multi-timeframe pattern detection
    - Advanced confluence detection
    - Pattern-based correlation analysis
    """
    
    def __init__(self, config: Dict[str, Any], window_manager=None):
        """Initialize deprecated correlation matrix with migration warning"""
        
        # Issue deprecation warning
        warnings.warn(
            "CorrelationMatrix (6×6) is deprecated. "
            "Use Enhanced10x10CorrelationMatrix from enhanced.enhanced_correlation_matrix. "
            "The enhanced version supports all 10 components with advanced pattern detection.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning("Using deprecated 6×6 CorrelationMatrix - please migrate to Enhanced10x10CorrelationMatrix")
        
        self.config = config
        self.window_manager = window_manager
        
        # Legacy 6-component setup
        self.components = ['atm_ce', 'atm_pe', 'itm1_ce', 'itm1_pe', 'otm1_ce', 'otm1_pe']
        self.n_components = len(self.components)
        
        # Try to use enhanced version if available
        if ENHANCED_AVAILABLE:
            logger.info("Enhanced10x10CorrelationMatrix available - consider migrating")
            self._enhanced_matrix = Enhanced10x10CorrelationMatrix(config, window_manager)
            self._use_enhanced = True
        else:
            logger.warning("Enhanced10x10CorrelationMatrix not available - using legacy mode")
            self._use_enhanced = False
            self._init_legacy_mode()
    
    def _init_legacy_mode(self):
        """Initialize in legacy 6×6 mode"""
        self.window_sizes = self.config.get('rolling_windows', [3, 5, 10, 15])
        self.correlation_matrices = {}
        self.correlation_history = {}
        
        for window_size in self.window_sizes:
            self.correlation_matrices[window_size] = np.eye(self.n_components)
            self.correlation_history[window_size] = []
    
    def analyze(self, timestamp: pd.Timestamp) -> Optional[CorrelationResult]:
        """
        DEPRECATED: Analyze correlations
        
        Returns legacy 6×6 result or None. For full 10×10 analysis,
        use Enhanced10x10CorrelationMatrix.analyze_all_timeframes()
        """
        if self._use_enhanced:
            # Use enhanced version but return only 6×6 subset for compatibility
            enhanced_results = self._enhanced_matrix.analyze_all_timeframes(timestamp)
            
            if enhanced_results:
                # Get first timeframe result
                first_timeframe = list(enhanced_results.keys())[0]
                enhanced_result = enhanced_results[first_timeframe]
                
                # Extract 6×6 subset (first 6 components)
                legacy_matrix = enhanced_result.matrix[:6, :6]
                legacy_components = enhanced_result.component_names[:6]
                
                return CorrelationResult(
                    matrix=legacy_matrix,
                    component_names=legacy_components,
                    timestamp=timestamp,
                    window_size=first_timeframe,
                    avg_correlation=enhanced_result.avg_correlation,
                    max_correlation=enhanced_result.max_correlation,
                    min_correlation=enhanced_result.min_correlation
                )
        
        # Legacy mode fallback
        logger.warning("Running in legacy 6×6 mode - limited functionality")
        return None
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """DEPRECATED: Get correlation summary"""
        if self._use_enhanced:
            enhanced_summary = self._enhanced_matrix.get_correlation_summary()
            # Return subset for legacy compatibility
            return {
                'legacy_mode': False,
                'enhanced_available': True,
                'migration_recommended': True,
                'enhanced_summary': enhanced_summary
            }
        
        return {
            'legacy_mode': True,
            'enhanced_available': False,
            'migration_required': True,
            'components': self.components,
            'message': 'Please migrate to Enhanced10x10CorrelationMatrix'
        }


# Compatibility aliases for migration
CorrelationMatrixAnalyzer = CorrelationMatrix  # Common alias

# Migration helper function
def migrate_to_enhanced(config: Dict[str, Any], window_manager=None) -> 'Enhanced10x10CorrelationMatrix':
    """
    Migration helper to create Enhanced10x10CorrelationMatrix
    
    Args:
        config: Configuration dictionary
        window_manager: Window manager instance
        
    Returns:
        Enhanced10x10CorrelationMatrix instance
    """
    if not ENHANCED_AVAILABLE:
        raise ImportError("Enhanced10x10CorrelationMatrix not available")
    
    logger.info("Migrating from legacy 6×6 to enhanced 10×10 correlation matrix")
    return Enhanced10x10CorrelationMatrix(config, window_manager)


# Module deprecation notice
def __deprecated_module_warning():
    """Issue module-level deprecation warning"""
    warnings.warn(
        "Module 'correlation_matrix' is deprecated. "
        "Use 'enhanced.enhanced_correlation_matrix' instead. "
        "Enhanced version supports 10×10 correlation matrix with pattern detection.",
        DeprecationWarning,
        stacklevel=3
    )

# Issue warning when module is imported
__deprecated_module_warning()