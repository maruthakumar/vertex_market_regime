"""
Base Infrastructure for Market Regime Indicator Refactoring
===========================================================

This package provides the foundational infrastructure for the refactored
market regime indicator system, including base classes, strike selectors,
performance tracking, and adaptive weight management.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from .base_indicator import BaseIndicator, IndicatorState, IndicatorConfig
from .strike_selector_base import BaseStrikeSelector, StrikeInfo, StrikeSelectionStrategy
from .option_data_manager import OptionDataManager, RollingATMTracker, ATMData
from .performance_tracker import PerformanceTracker, PerformanceMetrics
from .adaptive_weight_manager import AdaptiveWeightManager, WeightOptimizationConfig
from .regime_detector_base import RegimeDetectorBase, RegimeClassification, RegimeType, PerformanceMonitor, CacheManager

__all__ = [
    'BaseIndicator',
    'IndicatorState', 
    'IndicatorConfig',
    'BaseStrikeSelector',
    'StrikeInfo',
    'StrikeSelectionStrategy',
    'OptionDataManager',
    'RollingATMTracker',
    'ATMData',
    'PerformanceTracker',
    'PerformanceMetrics',
    'AdaptiveWeightManager',
    'WeightOptimizationConfig',
    'RegimeDetectorBase',
    'RegimeClassification',
    'RegimeType',
    'PerformanceMonitor',
    'CacheManager'
]