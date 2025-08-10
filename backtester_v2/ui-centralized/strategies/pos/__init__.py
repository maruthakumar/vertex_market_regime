"""
POS (Positional) Strategy Module

This module implements multi-leg positional options strategies with support for:
- Calendar Spreads
- Iron Condors
- Iron Flies
- Custom multi-leg strategies
- Dynamic adjustments
- Greek-based risk management
"""

from .strategy import POSStrategy
from .parser import POSParser
from .processor import POSProcessor
from .query_builder import POSQueryBuilder
from .models import (
    POSLegModel,
    POSPortfolioModel,
    POSStrategyModel,
    GreekLimits,
    AdjustmentRule,
    MarketRegimeFilter,
    POSTemplates
)

__version__ = "1.0.0"

__all__ = [
    "POSStrategy",
    "POSParser",
    "POSProcessor",
    "POSQueryBuilder",
    "POSLegModel",
    "POSPortfolioModel", 
    "POSStrategyModel",
    "GreekLimits",
    "AdjustmentRule",
    "MarketRegimeFilter",
    "POSTemplates"
]