"""
Technical Indicator Strategy Module

This module implements technical indicator-based strategies:
- 200+ TA-Lib indicators
- Smart Money Concepts (SMC)
- Candlestick patterns
- Volume profile analysis
- Rule-based signal generation
- Multi-timeframe analysis
"""

# Strategy import now fixed
from .strategy import IndicatorStrategy
from .parser import IndicatorParser
# from .processor import IndicatorProcessor
# from .query_builder import IndicatorQueryBuilder
from .models import (
    IndicatorStrategyModel,
    IndicatorPortfolioModel,
    IndicatorSignal,
    IndicatorTrade,
    IndicatorConfig,
    SMCConfig,
    SignalCondition,
    RiskManagementConfig,
    ExecutionConfig,
    IndicatorPresets,
    # Backward compatibility aliases
    MLSignal,
    MLTrade
)

__version__ = "1.0.0"

__all__ = [
    # New names
    "IndicatorStrategy",  # Now enabled
    "IndicatorParser",
    # "IndicatorProcessor",  # To be implemented
    # "IndicatorQueryBuilder",  # To be implemented
    "IndicatorStrategyModel",
    "IndicatorPortfolioModel",
    "IndicatorSignal",
    "IndicatorTrade",
    "IndicatorConfig",
    "SMCConfig",
    "SignalCondition",
    "RiskManagementConfig",
    "ExecutionConfig",
    "IndicatorPresets",
    # Backward compatibility
    "MLIndicatorStrategy",  # Alias for IndicatorStrategy
    # "MLIndicatorStrategyModel",  # To be added
    # "MLIndicatorPortfolioModel",  # To be added
    "MLSignal",
    "MLTrade"
]

# Backward compatibility aliases
MLIndicatorStrategy = IndicatorStrategy