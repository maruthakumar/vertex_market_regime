"""
Enhanced Modules for Market Regime Detection System

This package contains enhanced implementations of various components
for the market regime detection system.
"""

# Core engine imports
from .enhanced_market_regime_engine import EnhancedMarketRegimeEngine
from .enhanced_regime_engine import EnhancedMarketRegimeEngine as EnhancedRegimeEngine
from .enhanced_regime_formation_engine import EnhancedRegimeFormationEngine

# Detector and classifier imports
from .enhanced_regime_detector import EnhancedRegimeDetector
from .enhanced_regime_detector_v2 import EnhancedRegimeDetectorV2
from .enhanced_12_regime_detector import Enhanced12RegimeDetector
from .enhanced_18_regime_classifier import Enhanced18RegimeClassifier

# Configuration and parsing imports
from .enhanced_configurable_excel_manager import EnhancedConfigurableExcelManager
from .enhanced_excel_parser import EnhancedExcelParser
from .enhanced_excel_config_generator import EnhancedExcelConfigGenerator
from .enhanced_excel_config_template import EnhancedExcelConfigTemplate

# Analysis imports
from .enhanced_greek_sentiment_analysis import EnhancedGreekSentimentAnalysis
from .enhanced_greek_sentiment_analysis_v2 import EnhancedGreekSentimentAnalysisV2
from .enhanced_trending_oi_pa_analysis import EnhancedTrendingOIPAAnalysis
from .enhanced_oi_pattern_mathematical_correlation import EnhancedOIPatternMathematicalCorrelation

# Indicator and calculation imports
from .enhanced_atr_indicators import EnhancedATRIndicators
from .enhanced_indicator_parameters import EnhancedIndicatorParameters
from .enhanced_volume_weighted_greeks import EnhancedVolumeWeightedGreeks
from .enhanced_triple_rolling_straddle_engine_v2 import EnhancedTripleRollingStraddleEngineV2
from .enhanced_triple_straddle_analyzer import EnhancedTripleStraddleAnalyzer
from .enhanced_multi_indicator_engine import EnhancedMultiIndicatorEngine

# Integration and optimization imports
from .enhanced_integration_manager import EnhancedIntegrationManager
from .enhanced_adaptive_integration_framework import EnhancedAdaptiveIntegrationFramework
from .enhanced_historical_weightage_optimizer import EnhancedHistoricalWeightageOptimizer

# Validation and monitoring imports
from .enhanced_market_regime_validator import EnhancedMarketRegimeValidator
from .enhanced_mathematical_accuracy_validation import EnhancedMathematicalAccuracyValidation
from .enhanced_performance_monitor import EnhancedPerformanceMonitor
from .enhanced_logging_system import EnhancedLoggingSystem

__all__ = [
    # Core engines
    'EnhancedMarketRegimeEngine',
    'EnhancedRegimeEngine',
    'EnhancedRegimeFormationEngine',
    
    # Detectors and classifiers
    'EnhancedRegimeDetector',
    'EnhancedRegimeDetectorV2',
    'Enhanced12RegimeDetector',
    'Enhanced18RegimeClassifier',
    
    # Configuration
    'EnhancedConfigurableExcelManager',
    'EnhancedExcelParser',
    'EnhancedExcelConfigGenerator',
    'EnhancedExcelConfigTemplate',
    
    # Analysis
    'EnhancedGreekSentimentAnalysis',
    'EnhancedGreekSentimentAnalysisV2',
    'EnhancedTrendingOIPAAnalysis',
    'EnhancedOIPatternMathematicalCorrelation',
    
    # Indicators
    'EnhancedATRIndicators',
    'EnhancedIndicatorParameters',
    'EnhancedVolumeWeightedGreeks',
    'EnhancedTripleRollingStraddleEngineV2',
    'EnhancedTripleStraddleAnalyzer',
    'EnhancedMultiIndicatorEngine',
    
    # Integration
    'EnhancedIntegrationManager',
    'EnhancedAdaptiveIntegrationFramework',
    'EnhancedHistoricalWeightageOptimizer',
    
    # Validation
    'EnhancedMarketRegimeValidator',
    'EnhancedMathematicalAccuracyValidation',
    'EnhancedPerformanceMonitor',
    'EnhancedLoggingSystem'
]