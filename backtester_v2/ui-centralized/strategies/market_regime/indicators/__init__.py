"""
Refactored Market Regime Indicators
===================================

This package contains the refactored market regime indicators with modular
architecture, adaptive weight optimization, and enhanced performance tracking.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from .greek_sentiment_v2 import GreekSentimentV2
from .oi_pa_analysis_v2 import OIPAAnalysisV2

# TODO: Import other indicators as they are implemented
# from .technical_indicators_v2 import (
#     OptionRSI,
#     OptionMACD,
#     OptionBollingerBands,
#     TechnicalIndicatorsV2
# )

__all__ = [
    'GreekSentimentV2',
    'OIPAAnalysisV2', 
    # 'OptionRSI',
    # 'OptionMACD',
    # 'OptionBollingerBands',
    # 'TechnicalIndicatorsV2'
]