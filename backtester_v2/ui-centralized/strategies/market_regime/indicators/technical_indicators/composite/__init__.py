"""
Composite Technical Analysis
============================

Combines option-based and underlying-based technical indicators
for comprehensive market analysis.
"""

from .indicator_fusion import IndicatorFusion
from .regime_classifier import RegimeClassifier

__all__ = [
    'IndicatorFusion',
    'RegimeClassifier'
]