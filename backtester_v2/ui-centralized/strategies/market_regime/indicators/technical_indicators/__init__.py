"""
Technical Indicators V2 - Modular Technical Analysis System
==========================================================

This package provides comprehensive technical analysis with separate
implementations for option-based and underlying-based indicators.

Components:
- Option-Based Indicators: RSI, MACD, Bollinger Bands on option data
- Underlying-Based Indicators: Traditional technical analysis
- Composite Analysis: Fusion of option and underlying signals
- Technical Indicators Analyzer: Main orchestrator

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0 - Modular Technical Indicators
"""

from .technical_indicators_analyzer import TechnicalIndicatorsAnalyzer

# Option-based indicators
from .option_based.option_rsi import OptionRSI
from .option_based.option_macd import OptionMACD
from .option_based.option_bollinger import OptionBollinger
from .option_based.option_volume_flow import OptionVolumeFlow

# Underlying-based indicators
from .underlying_based.price_rsi import PriceRSI
from .underlying_based.price_macd import PriceMACD
from .underlying_based.price_bollinger import PriceBollinger
from .underlying_based.trend_strength import TrendStrength

# Composite analysis
from .composite.indicator_fusion import IndicatorFusion
from .composite.regime_classifier import RegimeClassifier

__all__ = [
    'TechnicalIndicatorsAnalyzer',
    # Option-based
    'OptionRSI',
    'OptionMACD',
    'OptionBollinger',
    'OptionVolumeFlow',
    # Underlying-based
    'PriceRSI',
    'PriceMACD',
    'PriceBollinger',
    'TrendStrength',
    # Composite
    'IndicatorFusion',
    'RegimeClassifier'
]