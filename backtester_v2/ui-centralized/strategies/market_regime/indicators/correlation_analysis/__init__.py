"""
Correlation Analysis Module
===========================

Analyzes cross-market and inter-asset correlations to identify:
- Market synchronization levels
- Divergence patterns
- Sector rotation signals
- Risk-on/risk-off transitions

This module is part of the 9 active components in the market regime system.
"""

from .correlation_analyzer import CorrelationAnalyzer
from .dynamic_correlation_matrix import DynamicCorrelationMatrix

__all__ = ['CorrelationAnalyzer', 'DynamicCorrelationMatrix']