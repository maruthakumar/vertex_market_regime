"""
Component 8: Master Integration Feature Engineering Module

This module provides the master integration feature engineering framework that generates
48 cross-component integration features from Components 1-7 outputs with DTE-adaptive
patterns and system coherence measurements.

No classification logic - only feature engineering for ML consumption.
"""

from .component_08_analyzer import Component08Analyzer
from .feature_engine import MasterIntegrationFeatureEngine

__all__ = [
    'Component08Analyzer',
    'MasterIntegrationFeatureEngine'
]