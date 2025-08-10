"""
Adaptive Optimization System - Market Regime Adaptive Optimization
================================================================

Main module for adaptive optimization system including historical analysis,
ML models, and performance optimization components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

from .core.historical_optimizer import HistoricalOptimizer
from .core.performance_evaluator import PerformanceEvaluator
from .core.weight_validator import WeightValidator
from .ml_models.random_forest_optimizer import RandomForestOptimizer
from .ml_models.linear_regression_optimizer import LinearRegressionOptimizer
from .ml_models.ensemble_optimizer import EnsembleOptimizer
from .historical_analysis.data_processor import HistoricalDataProcessor
from .historical_analysis.pattern_analyzer import PatternAnalyzer
from .historical_analysis.performance_tracker import AdaptivePerformanceTracker

__version__ = "2.0.0"
__author__ = "Market Regime Refactoring Team"

__all__ = [
    'HistoricalOptimizer',
    'PerformanceEvaluator', 
    'WeightValidator',
    'RandomForestOptimizer',
    'LinearRegressionOptimizer',
    'EnsembleOptimizer',
    'HistoricalDataProcessor',
    'PatternAnalyzer',
    'AdaptivePerformanceTracker'
]