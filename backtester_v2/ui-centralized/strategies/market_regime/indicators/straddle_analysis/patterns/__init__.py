"""
Multi-Timeframe Pattern Recognition System for Triple Straddle Analysis

This module provides comprehensive pattern recognition across all 10 components:
- 6 individual components (ATM/ITM1/OTM1 × CE/PE)
- 3 individual straddles (ATM, ITM1, OTM1)
- 1 combined triple straddle

Key Features:
- Multi-timeframe analysis (3, 5, 10, 15 minutes)
- 7-layer validation system for >90% success rate
- Statistical significance testing
- 5-model ML ensemble for ultra-sophisticated pattern scoring
- Risk-adjusted pattern selection
- Real-time pattern adaptation
"""

from .pattern_repository import PatternRepository, PatternSchema
from .pattern_detector import MultiTimeframePatternDetector
from .pattern_validator import SevenLayerPatternValidator
from .statistical_validator import StatisticalPatternValidator
from .ml_ensemble import AdvancedMLEnsemble, EnsemblePrediction, ModelPrediction

__all__ = [
    'PatternRepository',
    'PatternSchema',
    'MultiTimeframePatternDetector', 
    'SevenLayerPatternValidator',
    'StatisticalPatternValidator',
    'AdvancedMLEnsemble',
    'EnsemblePrediction',
    'ModelPrediction'
]

__version__ = "1.0.0"