"""
Enhanced Components for 10-Component Multi-Timeframe Straddle Analysis

This module provides enhanced versions of the core straddle analysis components
upgraded to handle all 10 components across multiple timeframes:

- 6 Individual Components: ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
- 3 Individual Straddles: ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE
- 1 Combined Triple: COMBINED_TRIPLE_STRADDLE

Enhanced Features:
- 10x10 correlation matrix (upgraded from 6x6)
- Multi-timeframe resistance analysis (3, 5, 10, 15 minutes)
- Pattern recognition integration
- Advanced confluence detection
- Real-time signal generation
"""

from .enhanced_correlation_matrix import Enhanced10x10CorrelationMatrix, EnhancedCorrelationResult, CorrelationPattern
from .enhanced_resistance_analyzer import Enhanced10ComponentResistanceAnalyzer, EnhancedResistanceAnalysisResult, ResistanceLevel

__all__ = [
    'Enhanced10x10CorrelationMatrix',
    'EnhancedCorrelationResult', 
    'CorrelationPattern',
    'Enhanced10ComponentResistanceAnalyzer',
    'EnhancedResistanceAnalysisResult',
    'ResistanceLevel'
]

__version__ = "2.0.0"

# Migration note: These enhanced components replace the legacy 6x6 versions
# Legacy modules: rolling/correlation_matrix.py, core/resistance_analyzer.py