"""
Market Regime Detection and Classification Module

This module provides comprehensive market regime analysis capabilities
integrated with the backtester_v2 architecture.

Key Features:
- Multi-timeframe regime detection
- Dynamic indicator weighting
- Performance-based adaptation
- GPU-accelerated calculations
- Real-time regime updates
- Excel configuration support
- Refactored architecture with base classes
- 12-regime and 18-regime classification systems
- Enhanced 10×10 correlation and resistance matrices

Version 2.0.0 - Refactored Architecture
"""

# Import existing models and strategies
try:
    from .models import (
        RegimeConfig,
        IndicatorConfig,
        RegimeClassification as LegacyRegimeClassification,
        RegimeType as LegacyRegimeType,
        PerformanceMetrics
    )
    from .strategy import MarketRegimeStrategy
    from .query_builder import MarketRegimeQueryBuilder, RegimeQueryConfig
    from .parser import RegimeConfigParser
    from .processor import RegimeProcessor
    from .calculator import RegimeCalculator
    from .classifier import RegimeClassifier
    _legacy_imports_available = True
except ImportError:
    _legacy_imports_available = False

# Import refactored components
from .base import (
    RegimeDetectorBase,
    RegimeClassification,
    RegimeType,
    PerformanceMonitor,
    CacheManager,
    BaseIndicator,
    IndicatorState,
    IndicatorConfig as BaseIndicatorConfig,
    PerformanceTracker,
    AdaptiveWeightManager
)

# Import refactored detectors
# NOTE: These are now in archive_enhanced_modules_do_not_use - DO NOT USE
# Refactored system uses indicators/ directory structure
_refactored_detectors_available = False

# Import component registry for all 9 components
try:
    from .base.component_registry import ComponentRegistry, get_component_registry
    _component_registry_available = True
except ImportError:
    _component_registry_available = False

# Import configuration management
try:
    from .config_manager import get_config_manager, ConfigManager as RegimeConfigManager
    from .advanced_config_validator import ConfigurationValidator, ValidationIssue, ValidationSeverity
    _config_management_available = True
except ImportError:
    _config_management_available = False

# Import enhanced analysis components
try:
    from .indicators.straddle_analysis.enhanced.enhanced_correlation_matrix import Enhanced10x10CorrelationMatrix
    from .indicators.straddle_analysis.enhanced.enhanced_resistance_analyzer import Enhanced10ComponentResistanceAnalyzer
    _enhanced_analysis_available = True
except ImportError:
    _enhanced_analysis_available = False

# Import engines
try:
    from .correlation_matrix_engine import CorrelationMatrixEngine
    from .unified_enhanced_market_regime_engine import UnifiedMarketRegimeEngine
    _engines_available = True
except ImportError:
    try:
        from .unified_market_regime_engine import UnifiedMarketRegimeEngine
        _engines_available = True
    except ImportError:
        _engines_available = False

__version__ = "2.0.0"
__author__ = "Market Regime System Team"

# Build __all__ dynamically based on available imports
__all__ = []

# Add legacy imports if available
if _legacy_imports_available:
    __all__.extend([
        'RegimeConfig',
        'IndicatorConfig', 
        'LegacyRegimeClassification',
        'LegacyRegimeType',
        'PerformanceMetrics',
        'MarketRegimeStrategy',
        'RegimeConfigParser',
        'RegimeProcessor',
        'RegimeCalculator',
        'RegimeClassifier'
    ])

# Add refactored base components
__all__.extend([
    'RegimeDetectorBase',
    'RegimeClassification',
    'RegimeType',
    'PerformanceMonitor',
    'CacheManager',
    'BaseIndicator',
    'IndicatorState',
    'BaseIndicatorConfig',
    'PerformanceTracker',
    'AdaptiveWeightManager'
])

# Add refactored detectors
if _refactored_detectors_available:
    __all__.extend([
        'Refactored12RegimeDetector',
        'Refactored18RegimeClassifier'
    ])

# Add configuration management
if _config_management_available:
    __all__.extend([
        'get_config_manager',
        'RegimeConfigManager',
        'ConfigurationValidator',
        'ValidationIssue',
        'ValidationSeverity'
    ])

# Add enhanced analysis
if _enhanced_analysis_available:
    __all__.extend([
        'Enhanced10x10CorrelationMatrix',
        'Enhanced10ComponentResistanceAnalyzer'
    ])

# Add engines
if _engines_available:
    __all__.extend([
        'CorrelationMatrixEngine',
        'UnifiedMarketRegimeEngine'
    ])

# Add component registry
if _component_registry_available:
    __all__.extend([
        'ComponentRegistry',
        'get_component_registry'
    ])

# Add version info
__all__.extend(['__version__', '__author__'])

# Convenience functions
def create_regime_detector(regime_type='12', config=None):
    """
    Create a regime detector instance
    
    Args:
        regime_type: '12' or '18' for regime count
        config: Optional configuration dictionary
        
    Returns:
        Configured regime detector instance
    """
    if not _refactored_detectors_available:
        raise ImportError("Refactored detectors not available")
        
    if regime_type == '12':
        return Refactored12RegimeDetector(config)
    elif regime_type == '18':
        return Refactored18RegimeClassifier(config)
    else:
        raise ValueError(f"Unknown regime type: {regime_type}")

def get_available_components():
    """Get list of available components"""
    return {
        'legacy_imports': _legacy_imports_available,
        'refactored_detectors': _refactored_detectors_available,
        'config_management': _config_management_available,
        'enhanced_analysis': _enhanced_analysis_available,
        'engines': _engines_available,
        'version': __version__
    }
