"""
Component 07: Support/Resistance Feature Engineering
120+ feature extraction for ML-based S&R analysis
Includes base 72 features + 48 advanced pattern features
"""

from .component_07_analyzer import Component07Analyzer
from .feature_engine import SupportResistanceFeatureEngine, SupportResistanceFeatures
from .straddle_level_detector import StraddleLevelDetector
from .underlying_level_detector import UnderlyingLevelDetector
from .confluence_analyzer import ConfluenceAnalyzer
from .weight_learning_engine import SupportResistanceWeightLearner
from .advanced_pattern_detector import AdvancedPatternDetector

__all__ = [
    "Component07Analyzer",
    "SupportResistanceFeatureEngine",
    "SupportResistanceFeatures",
    "StraddleLevelDetector",
    "UnderlyingLevelDetector",
    "ConfluenceAnalyzer",
    "SupportResistanceWeightLearner",
    "AdvancedPatternDetector"
]