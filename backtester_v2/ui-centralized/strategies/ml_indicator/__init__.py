"""
ML Indicator Strategy Module

This module implements technical indicator-based strategies with ML integration:
- 200+ TA-Lib indicators
- Smart Money Concepts (SMC)
- Candlestick patterns
- Volume profile analysis
- ML-based signal generation
- Multi-timeframe analysis
"""

from .strategy import MLIndicatorStrategy
from .parser import MLIndicatorParser
from .processor import MLIndicatorProcessor
from .query_builder import MLIndicatorQueryBuilder
from .heavydb_integration import MLHeavyDBIntegration, get_ml_heavydb_integration
from .websocket_integration import MLWebSocketIntegration, MLStreamConfig, MLEventType, MLWebSocketEvent, get_ml_websocket_integration
from .models import (
    MLIndicatorStrategyModel,
    MLIndicatorPortfolioModel,
    MLLegModel,
    IndicatorConfig,
    SMCConfig,
    MLModelConfig,
    SignalCondition,
    RiskManagementConfig,
    ExecutionConfig,
    IndicatorPresets
)

__version__ = "1.0.0"

__all__ = [
    "MLIndicatorStrategy",
    "MLIndicatorParser",
    "MLIndicatorProcessor",
    "MLIndicatorQueryBuilder",
    "MLIndicatorStrategyModel",
    "MLIndicatorPortfolioModel",
    "MLLegModel",
    "IndicatorConfig",
    "SMCConfig",
    "MLModelConfig",
    "SignalCondition",
    "RiskManagementConfig",
    "ExecutionConfig",
    "IndicatorPresets"
]