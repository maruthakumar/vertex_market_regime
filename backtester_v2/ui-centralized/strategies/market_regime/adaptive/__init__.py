"""
Adaptive Market Regime Formation System

A sophisticated, data-driven market regime formation system that supports
configurable regime counts (8, 12, or 18) with intelligent transition
management and continuous performance optimization for intraday trading.

Key Features:
- Configurable regime count via Excel configuration
- Historical data-driven parameter tuning
- Adaptive weight optimization using ASL-inspired mechanisms
- Intelligent transition management with noise filtering
- Continuous performance feedback and optimization
"""

__version__ = "1.0.0"
__author__ = "Backtester V2 Team"

# Import core components
from .config.adaptive_regime_config_manager import (
    AdaptiveRegimeConfigManager,
    AdaptiveRegimeConfig,
    RegimeCount
)

from .analysis.historical_regime_analyzer import (
    HistoricalRegimeAnalyzer,
    RegimePattern,
    TransitionDynamics
)

from .core.regime_definition_builder import (
    RegimeDefinitionBuilder,
    RegimeDefinition,
    RegimeBoundary
)

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'phase': 'Phase 1 - Core Infrastructure'
}

# Module status
MODULE_STATUS = {
    'config': 'implemented',
    'analysis': 'implemented',
    'core': 'implemented',
    'intelligence': 'pending',
    'optimization': 'pending',
    'validation': 'pending',
    'integration': 'pending',
    'dashboard': 'pending'
}

__all__ = [
    # Configuration
    'AdaptiveRegimeConfigManager',
    'AdaptiveRegimeConfig',
    'RegimeCount',
    
    # Analysis
    'HistoricalRegimeAnalyzer',
    'RegimePattern',
    'TransitionDynamics',
    
    # Core
    'RegimeDefinitionBuilder',
    'RegimeDefinition',
    'RegimeBoundary',
    
    # Version
    'VERSION_INFO',
    'MODULE_STATUS'
]