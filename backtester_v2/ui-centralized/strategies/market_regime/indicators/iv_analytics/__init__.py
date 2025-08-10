"""
IV Analytics V2 - Comprehensive Implied Volatility Analysis System
================================================================

Advanced implied volatility analysis with surface modeling, term structure,
skew analysis, volatility forecasting, and arbitrage detection.

Components:
- Surface Analysis: IV surface modeling and interpolation
- Term Structure: Volatility term structure analysis and curves
- Skew Analysis: Volatility skew detection and trading signals
- Volatility Forecasting: Predictive volatility models
- Arbitrage Detection: Cross-instrument volatility arbitrage opportunities
- IV Analytics Analyzer: Main orchestrator

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0 - Comprehensive IV Analytics
"""

from .iv_analytics_analyzer import IVAnalyticsAnalyzer

# Surface Analysis
from .surface_analysis.iv_surface_modeler import IVSurfaceModeler
from .surface_analysis.surface_interpolator import SurfaceInterpolator
from .surface_analysis.smile_analyzer import SmileAnalyzer

# Term Structure
from .term_structure.term_structure_analyzer import TermStructureAnalyzer
from .term_structure.curve_fitter import CurveFitter
from .term_structure.forward_vol_calculator import ForwardVolCalculator

# Skew Analysis
from .skew_analysis.skew_detector import SkewDetector
from .skew_analysis.skew_momentum import SkewMomentum
from .skew_analysis.risk_reversal_analyzer import RiskReversalAnalyzer

# Volatility Forecasting
from .volatility_forecasting.vol_predictor import VolPredictor
from .volatility_forecasting.garch_model import GARCHModel
from .volatility_forecasting.regime_vol_model import RegimeVolModel

# Arbitrage Detection
from .arbitrage_detection.calendar_arbitrage import CalendarArbitrage
from .arbitrage_detection.strike_arbitrage import StrikeArbitrage
from .arbitrage_detection.vol_arbitrage_scanner import VolArbitrageScanner

__all__ = [
    'IVAnalyticsAnalyzer',
    # Surface Analysis
    'IVSurfaceModeler',
    'SurfaceInterpolator', 
    'SmileAnalyzer',
    # Term Structure
    'TermStructureAnalyzer',
    'CurveFitter',
    'ForwardVolCalculator',
    # Skew Analysis
    'SkewDetector',
    'SkewMomentum',
    'RiskReversalAnalyzer',
    # Volatility Forecasting
    'VolPredictor',
    'GARCHModel',
    'RegimeVolModel',
    # Arbitrage Detection
    'CalendarArbitrage',
    'StrikeArbitrage',
    'VolArbitrageScanner'
]