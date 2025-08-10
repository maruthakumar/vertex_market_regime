"""
Triple Straddle Analysis Module

A comprehensive, unified system for triple straddle analysis with:
- 6 individual component analyzers (ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE)
- 3 straddle combination analyzers (ATM, ITM1, OTM1)
- 1 combined weighted analysis system
- Rolling analysis across [3,5,10,15] minute windows
- 6×6 correlation matrix tracking
- Support/resistance level integration
- Excel-driven configuration
"""

# Main engine
from .core.straddle_engine import TripleStraddleEngine, TripleStraddleAnalysisResult

# Core components
from .core.calculation_engine import CalculationEngine
from .core.resistance_analyzer import ResistanceAnalyzer, ResistanceAnalysisResult

# Configuration
from .config.excel_reader import StraddleConfigReader, StraddleConfig

# Rolling window analysis
from .rolling.window_manager import RollingWindowManager
from .rolling.correlation_matrix import CorrelationMatrix

# Component analyzers
from .components.combined_straddle_analyzer import CombinedStraddleAnalyzer, CombinedStraddleResult
from .components.atm_straddle_analyzer import ATMStraddleAnalyzer
from .components.itm1_straddle_analyzer import ITM1StraddleAnalyzer
from .components.otm1_straddle_analyzer import OTM1StraddleAnalyzer

# Individual component analyzers
from .components import (
    ATMCallAnalyzer, ATMPutAnalyzer,
    ITM1CallAnalyzer, ITM1PutAnalyzer,
    OTM1CallAnalyzer, OTM1PutAnalyzer
)

__all__ = [
    # Main engine
    'TripleStraddleEngine',
    'TripleStraddleAnalysisResult',
    
    # Core components
    'CalculationEngine',
    'ResistanceAnalyzer',
    'ResistanceAnalysisResult',
    
    # Configuration
    'StraddleConfigReader',
    'StraddleConfig',
    
    # Rolling analysis
    'RollingWindowManager',
    'CorrelationMatrix',
    
    # Combined analyzer
    'CombinedStraddleAnalyzer',
    'CombinedStraddleResult',
    
    # Straddle analyzers
    'ATMStraddleAnalyzer',
    'ITM1StraddleAnalyzer', 
    'OTM1StraddleAnalyzer',
    
    # Individual components
    'ATMCallAnalyzer', 'ATMPutAnalyzer',
    'ITM1CallAnalyzer', 'ITM1PutAnalyzer',
    'OTM1CallAnalyzer', 'OTM1PutAnalyzer'
]

__version__ = "3.0.0"
__author__ = "Market Regime Team"