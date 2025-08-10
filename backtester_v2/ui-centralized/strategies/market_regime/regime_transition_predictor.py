#!/usr/bin/env python3
"""
Regime Transition Predictor for Enhanced Market Regime Framework V2.0

This module implements sophisticated algorithms for predicting regime transitions
using advanced statistical and machine learning techniques.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Import Sentry configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.sentry_config import capture_exception, add_breadcrumb, set_tag, track_errors, capture_message
except ImportError:
    # Fallback if sentry not available
    def capture_exception(*args, **kwargs): pass
    def add_breadcrumb(*args, **kwargs): pass
    def set_tag(*args, **kwargs): pass
    def track_errors(func): return func
    def capture_message(*args, **kwargs): pass

logger = logging.getLogger(__name__)

class TransitionType(Enum):
    """Types of regime transitions"""
    GRADUAL_TRANSITION = "gradual_transition"
    SUDDEN_TRANSITION = "sudden_transition"
    CYCLICAL_TRANSITION = "cyclical_transition"
    VOLATILITY_DRIVEN = "volatility_driven"
    TREND_DRIVEN = "trend_driven"
    STRUCTURE_DRIVEN = "structure_driven"

@dataclass
class TransitionSignal:
    """Transition signal information"""
    transition_type: TransitionType
    probability: float
    confidence: float
    time_horizon: str
    target_regime: Any
    signal_strength: float
    contributing_factors: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class TransitionPredictorConfig:
    """Configuration for transition predictor"""
    sensitivity: float = 0.3
    lookback_window: int = 60
    prediction_horizon: int = 30
    min_transition_probability: float = 0.2
    enable_volatility_signals: bool = True
    enable_trend_signals: bool = True
    enable_structure_signals: bool = True
    enable_cyclical_analysis: bool = True
    enable_momentum_analysis: bool = True

class RegimeTransitionPredictor:
    """
    Regime Transition Predictor for Market Regime Formation
    
    Implements sophisticated transition prediction algorithms including:
    - Statistical transition analysis
    - Momentum-based transition detection
    - Volatility regime transition signals
    - Cyclical pattern recognition
    - Multi-factor transition scoring
    """
    
    def __init__(self, sensitivity: float = 0.3,
                 config: Optional[TransitionPredictorConfig] = None):
        """
        Initialize Regime Transition Predictor
        
        Args:
            sensitivity: Transition detection sensitivity
            config: Transition predictor configuration
        """
        set_tag("component", "regime_transition_predictor")
        
        self.sensitivity = sensitivity
        self.config = config or TransitionPredictorConfig()
        
        # Transition history and patterns
        self.transition_history = []
        self.transition_patterns = {}
        self.regime_duration_stats = {}
        
        # Prediction models
        self.volatility_model = None
        self.trend_model = None
        self.structure_model = None
        
        logger.info("Regime Transition Predictor initialized")
        add_breadcrumb(
            message="Regime Transition Predictor initialized",
            category="initialization",
            data={"sensitivity": sensitivity}
        )
    
    @track_errors
    async def predict_transitions(self, transition_features: Dict[str, Any],
                                current_regime: Any) -> Dict[str, float]:
        """
        Predict regime transitions based on current features
        
        Args:
            transition_features: Features for transition analysis
            current_regime: Current regime type
            
        Returns:
            Dict: Transition probabilities for different regimes
        """
        set_tag("operation", "transition_prediction")
        
        try:
            transition_probabilities = {}
            
            # Volatility-driven transitions
            if self.config.enable_volatility_signals:
                vol_transitions = await self._predict_volatility_transitions(
                    transition_features, current_regime
                )
                self._merge_transition_probabilities(transition_probabilities, vol_transitions)
            
            # Trend-driven transitions
            if self.config.enable_trend_signals:
                trend_transitions = await self._predict_trend_transitions(
                    transition_features, current_regime
                )
                self._merge_transition_probabilities(transition_probabilities, trend_transitions)
            
            # Structure-driven transitions
            if self.config.enable_structure_signals:
                structure_transitions = await self._predict_structure_transitions(
                    transition_features, current_regime
                )
                self._merge_transition_probabilities(transition_probabilities, structure_transitions)
            
            # Cyclical transitions
            if self.config.enable_cyclical_analysis:
                cyclical_transitions = await self._predict_cyclical_transitions(
                    transition_features, current_regime
                )
                self._merge_transition_probabilities(transition_probabilities, cyclical_transitions)
            
            # Momentum-based transitions
            if self.config.enable_momentum_analysis:
                momentum_transitions = await self._predict_momentum_transitions(
                    transition_features, current_regime
                )
                self._merge_transition_probabilities(transition_probabilities, momentum_transitions)
            
            # Normalize probabilities
            transition_probabilities = self._normalize_probabilities(transition_probabilities)
            
            add_breadcrumb(
                message="Transition prediction completed",
                category="transition_prediction",
                data={
                    "current_regime": str(current_regime),
                    "transitions_predicted": len(transition_probabilities)
                }
            )
            
            return transition_probabilities
            
        except Exception as e:
            logger.error(f"Error predicting transitions: {e}")
            capture_exception(e, component="transition_prediction")
            return {}
    
    @track_errors
    async def _predict_volatility_transitions(self, features: Dict[str, Any],
                                            current_regime: Any) -> Dict[str, float]:
        """Predict transitions based on volatility signals"""
        try:
            transitions = {}
            
            # Extract volatility features
            volatility_data = features.get('volatility_data', {})
            if not volatility_data:
                return transitions
            
            current_vol = volatility_data.get('current_volatility', 0.2)
            vol_trend = volatility_data.get('volatility_trend', 0.0)
            vol_percentile = volatility_data.get('volatility_percentile', 0.5)
            
            # Volatility expansion signals
            if vol_trend > 0.1 and vol_percentile > 0.8:
                # High probability of transition to high volatility regime
                transitions['HIGH_VOLATILITY'] = 0.7
                transitions['VOLATILE_TRENDING'] = 0.6
                transitions['VOLATILE_RANGING'] = 0.5
            
            # Volatility contraction signals
            elif vol_trend < -0.1 and vol_percentile < 0.2:
                # High probability of transition to low volatility regime
                transitions['LOW_VOLATILITY'] = 0.7
                transitions['STABLE_TRENDING'] = 0.6
                transitions['STABLE_RANGING'] = 0.5
            
            # Volatility regime persistence
            else:
                # Lower probability of volatility-driven transitions
                current_regime_str = str(current_regime)
                if 'VOLATILE' in current_regime_str:
                    transitions[current_regime_str] = 0.8  # Stay in volatile regime
                elif 'STABLE' in current_regime_str:
                    transitions[current_regime_str] = 0.8  # Stay in stable regime
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error predicting volatility transitions: {e}")
            capture_exception(e, component="volatility_transitions")
            return {}
    
    @track_errors
    async def _predict_trend_transitions(self, features: Dict[str, Any],
                                       current_regime: Any) -> Dict[str, float]:
        """Predict transitions based on trend signals"""
        try:
            transitions = {}
            
            # Extract trend features
            trend_data = features.get('trend_data', {})
            if not trend_data:
                return transitions
            
            trend_strength = trend_data.get('trend_strength', 0.0)
            trend_direction = trend_data.get('trend_direction', 0.0)
            trend_momentum = trend_data.get('trend_momentum', 0.0)
            
            # Strong uptrend signals
            if trend_direction > 0.3 and trend_strength > 0.6:
                transitions['BULLISH_TRENDING'] = 0.8
                transitions['BULLISH_VOLATILE'] = 0.6
                transitions['BULLISH_STABLE'] = 0.5
            
            # Strong downtrend signals
            elif trend_direction < -0.3 and trend_strength > 0.6:
                transitions['BEARISH_TRENDING'] = 0.8
                transitions['BEARISH_VOLATILE'] = 0.6
                transitions['BEARISH_STABLE'] = 0.5
            
            # Weak trend / ranging signals
            elif abs(trend_direction) < 0.1 or trend_strength < 0.3:
                transitions['NEUTRAL_RANGING'] = 0.7
                transitions['NEUTRAL_BALANCED'] = 0.6
            
            # Trend reversal signals
            if trend_momentum < -0.2 and trend_strength > 0.4:
                # Potential trend reversal
                if trend_direction > 0:
                    transitions['BEARISH_TRENDING'] = 0.5
                else:
                    transitions['BULLISH_TRENDING'] = 0.5
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error predicting trend transitions: {e}")
            capture_exception(e, component="trend_transitions")
            return {}
    
    @track_errors
    async def _predict_structure_transitions(self, features: Dict[str, Any],
                                           current_regime: Any) -> Dict[str, float]:
        """Predict transitions based on market structure signals"""
        try:
            transitions = {}
            
            # Extract structure features
            structure_data = features.get('structure_data', {})
            if not structure_data:
                return transitions
            
            support_strength = structure_data.get('support_strength', 0.5)
            resistance_strength = structure_data.get('resistance_strength', 0.5)
            breakout_probability = structure_data.get('breakout_probability', 0.0)
            
            # Strong structure signals
            if support_strength > 0.7 and resistance_strength > 0.7:
                transitions['RANGING_STABLE'] = 0.8
                transitions['NEUTRAL_RANGING'] = 0.6
            
            # Weak structure / breakout signals
            elif breakout_probability > 0.6:
                transitions['TRENDING_VOLATILE'] = 0.7
                transitions['BREAKOUT_MOMENTUM'] = 0.6
            
            # Structure breakdown signals
            elif support_strength < 0.3 or resistance_strength < 0.3:
                transitions['VOLATILE_RANGING'] = 0.6
                transitions['UNCERTAIN_VOLATILE'] = 0.5
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error predicting structure transitions: {e}")
            capture_exception(e, component="structure_transitions")
            return {}
    
    @track_errors
    async def _predict_cyclical_transitions(self, features: Dict[str, Any],
                                          current_regime: Any) -> Dict[str, float]:
        """Predict transitions based on cyclical patterns"""
        try:
            transitions = {}
            
            # Extract cyclical features
            cyclical_data = features.get('cyclical_data', {})
            if not cyclical_data:
                return transitions
            
            cycle_position = cyclical_data.get('cycle_position', 0.5)
            cycle_strength = cyclical_data.get('cycle_strength', 0.0)
            
            # Market cycle analysis
            if cycle_position < 0.25:  # Early cycle
                transitions['BULLISH_STABLE'] = 0.6
                transitions['ACCUMULATION'] = 0.5
            elif cycle_position < 0.5:  # Mid cycle
                transitions['BULLISH_TRENDING'] = 0.7
                transitions['MOMENTUM_BUILDING'] = 0.6
            elif cycle_position < 0.75:  # Late cycle
                transitions['VOLATILE_TRENDING'] = 0.6
                transitions['DISTRIBUTION'] = 0.5
            else:  # Cycle end
                transitions['BEARISH_VOLATILE'] = 0.7
                transitions['CORRECTION'] = 0.6
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error predicting cyclical transitions: {e}")
            capture_exception(e, component="cyclical_transitions")
            return {}
    
    @track_errors
    async def _predict_momentum_transitions(self, features: Dict[str, Any],
                                          current_regime: Any) -> Dict[str, float]:
        """Predict transitions based on momentum signals"""
        try:
            transitions = {}
            
            # Extract momentum features
            momentum_data = features.get('momentum_data', {})
            if not momentum_data:
                return transitions
            
            price_momentum = momentum_data.get('price_momentum', 0.0)
            volume_momentum = momentum_data.get('volume_momentum', 0.0)
            volatility_momentum = momentum_data.get('volatility_momentum', 0.0)
            
            # Strong positive momentum
            if price_momentum > 0.3 and volume_momentum > 0.2:
                transitions['BULLISH_MOMENTUM'] = 0.8
                transitions['TRENDING_STRONG'] = 0.6
            
            # Strong negative momentum
            elif price_momentum < -0.3 and volume_momentum > 0.2:
                transitions['BEARISH_MOMENTUM'] = 0.8
                transitions['DECLINING_STRONG'] = 0.6
            
            # Momentum divergence
            elif abs(price_momentum - volume_momentum) > 0.3:
                transitions['MOMENTUM_DIVERGENCE'] = 0.6
                transitions['POTENTIAL_REVERSAL'] = 0.5
            
            # Low momentum
            elif abs(price_momentum) < 0.1 and volume_momentum < 0.1:
                transitions['LOW_MOMENTUM'] = 0.7
                transitions['CONSOLIDATION'] = 0.6
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error predicting momentum transitions: {e}")
            capture_exception(e, component="momentum_transitions")
            return {}
