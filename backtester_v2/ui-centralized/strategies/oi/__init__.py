#!/usr/bin/env python3
"""
OI Strategy Module - Open Interest based trading strategies

This module provides functionality for Open Interest (OI) based strategies
that select strikes dynamically based on OI and COI (Change in Open Interest) patterns.
"""

from .models import (
    OIMethod,
    COIBasedOn,
    OILegModel,
    OISettingModel,
    OIRanking,
    OISignal,
    ProcessedOISignal
)

from .parser import OIParser
from .archive_parser import OIArchiveParser
from .processor import OIProcessor
from .oi_analyzer import OIAnalyzer
from .query_builder import OIQueryBuilder
from .executor import OIExecutor
from .websocket_integration import OIWebSocketIntegration, OIStreamConfig, OIEventType, OIWebSocketEvent, get_oi_websocket_integration

# Import enhanced modules
try:
    from .enhanced_models import (
        EnhancedOIConfig, EnhancedLegConfig, DynamicWeightConfig,
        FactorConfig, PortfolioConfig, StrategyConfig, LegacyConfig,
        OiThresholdType, StrikeRangeType, NormalizationMethod, OutlierHandling,
        PerformanceMetrics
    )
    from .enhanced_parser import EnhancedOIParser
    from .dynamic_weight_engine import DynamicWeightEngine
    from .enhanced_processor import EnhancedOIProcessor
    from .unified_oi_interface import UnifiedOIInterface

    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced OI modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

__all__ = [
    # Enums
    'OIMethod',
    'COIBasedOn',
    
    # Models
    'OILegModel',
    'OISettingModel',
    'OIRanking',
    'OISignal',
    'ProcessedOISignal',
    
    # Core Classes
    'OIParser',
    'OIArchiveParser',
    'OIProcessor',
    'OIAnalyzer',
    'OIQueryBuilder',
    'OIExecutor',

    # Enhanced modules availability flag
    'ENHANCED_MODULES_AVAILABLE'
]

# Add enhanced modules to __all__ if available
if ENHANCED_MODULES_AVAILABLE:
    __all__.extend([
        'EnhancedOIConfig', 'EnhancedLegConfig', 'DynamicWeightConfig',
        'FactorConfig', 'PortfolioConfig', 'StrategyConfig', 'LegacyConfig',
        'OiThresholdType', 'StrikeRangeType', 'NormalizationMethod', 'OutlierHandling',
        'PerformanceMetrics', 'EnhancedOIParser', 'DynamicWeightEngine',
        'EnhancedOIProcessor', 'UnifiedOIInterface'
    ])

# Version info
__version__ = '1.0.0'
__author__ = 'Enterprise GPU Backtester Team'