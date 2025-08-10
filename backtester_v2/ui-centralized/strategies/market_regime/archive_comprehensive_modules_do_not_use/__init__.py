"""
Comprehensive Modules for Market Regime Detection System

This package contains comprehensive implementations and test suites
for the market regime detection system.
"""

# Core comprehensive modules
from .comprehensive_triple_straddle_engine import ComprehensiveTripleStraddleEngine
from .comprehensive_market_regime_analyzer import ComprehensiveMarketRegimeAnalyzer
from .comprehensive_csv_validator import ComprehensiveCSVValidator
from .comprehensive_regime_validation_analysis import ComprehensiveRegimeValidationAnalysis
from .comprehensive_api_fix import ComprehensiveAPIFix

# Test phase modules
from .comprehensive_test_phase0_data_validation import ComprehensiveTestPhase0DataValidation
from .comprehensive_test_phase1_environment import ComprehensiveTestPhase1Environment
from .comprehensive_test_phase2_ui_upload import ComprehensiveTestPhase2UIUpload
from .comprehensive_test_phase3_backend_excel_yaml import ComprehensiveTestPhase3BackendExcelYAML
from .comprehensive_test_phase4_indicator_logic import ComprehensiveTestPhase4IndicatorLogic
from .comprehensive_test_phase5_output_generation import ComprehensiveTestPhase5OutputGeneration

# Test suite
from .comprehensive_test_suite import ComprehensiveTestSuite

__all__ = [
    # Core modules
    'ComprehensiveTripleStraddleEngine',
    'ComprehensiveMarketRegimeAnalyzer',
    'ComprehensiveCSVValidator',
    'ComprehensiveRegimeValidationAnalysis',
    'ComprehensiveAPIFix',
    
    # Test phases
    'ComprehensiveTestPhase0DataValidation',
    'ComprehensiveTestPhase1Environment',
    'ComprehensiveTestPhase2UIUpload',
    'ComprehensiveTestPhase3BackendExcelYAML',
    'ComprehensiveTestPhase4IndicatorLogic',
    'ComprehensiveTestPhase5OutputGeneration',
    
    # Test suite
    'ComprehensiveTestSuite'
]