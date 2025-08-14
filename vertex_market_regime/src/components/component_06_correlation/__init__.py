"""
Component 6: Enhanced Correlation & Predictive Feature Engineering

This module implements the Enhanced Correlation & Predictive Feature Engineering system
with 200+ systematic features for ML consumption without hard-coded classification logic.

Key Features:
- Raw Correlation Feature Engineering (120 features)
- Predictive Straddle Intelligence (50 features) 
- Meta-Correlation Intelligence (30 features)
- Cross-component correlation measurements
- Gap-adjusted correlation analysis
- Performance target: <200ms processing time

Author: Market Regime System
Version: 1.0.0
"""

from .component_06_analyzer import (
    Component06CorrelationAnalyzer,
    Component06AnalysisResult,
    RawCorrelationFeatures,
    PredictiveStraddleFeatures,
    MetaCorrelationFeatures,
    RawCorrelationMeasurementSystem,
    PredictiveStraddleIntelligence,
    MetaCorrelationIntelligence
)

__all__ = [
    'Component06CorrelationAnalyzer',
    'Component06AnalysisResult', 
    'RawCorrelationFeatures',
    'PredictiveStraddleFeatures',
    'MetaCorrelationFeatures',
    'RawCorrelationMeasurementSystem',
    'PredictiveStraddleIntelligence',
    'MetaCorrelationIntelligence'
]