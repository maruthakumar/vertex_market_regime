#!/usr/bin/env python3
"""
🔄 Phase 5: Sophisticated Regime Formation Enhancement (FINAL PHASE)
Enhanced Market Regime Framework V2.0

This module implements the final phase of the Enhanced Market Regime Framework V2.0
with advanced regime formation algorithms, sophisticated pattern recognition, and
comprehensive integration with the existing 18-regime classification system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
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

# Import existing components
try:
    from enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType
    from enhanced_configurable_excel_manager import EnhancedConfigurableExcelManager
    from time_series_regime_storage import TimeSeriesRegimeStorage
    from comprehensive_triple_straddle_engine import StraddleAnalysisEngine
except ImportError:
    # Fallback for testing
    class Enhanced18RegimeType:
        NEUTRAL_BALANCED = "neutral_balanced"
        BULLISH_TRENDING = "bullish_trending"
        BEARISH_TRENDING = "bearish_trending"

    class Enhanced18RegimeDetector:
        def detect_regime(self, data):
            return type('RegimeResult', (), {
                'regime_type': Enhanced18RegimeType.NEUTRAL_BALANCED,
                'confidence': 0.5,
                'volatility_component': 0.5,
                'trend_component': 0.5,
                'structure_component': 0.5
            })()

    class EnhancedConfigurableExcelManager:
        def __init__(self, path=None): pass

    class TimeSeriesRegimeStorage:
        def __init__(self, path=None): pass

    class StraddleAnalysisEngine:
        def __init__(self): pass

logger = logging.getLogger(__name__)

class RegimeFormationComplexity(Enum):
    """Regime formation complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    ADVANCED = "advanced"
    SOPHISTICATED = "sophisticated"
    EXPERT = "expert"

class PatternRecognitionType(Enum):
    """Pattern recognition algorithm types"""
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"

@dataclass
class SophisticatedRegimeFormationConfig:
    """Configuration for sophisticated regime formation"""
    complexity_level: RegimeFormationComplexity = RegimeFormationComplexity.SOPHISTICATED
    pattern_recognition_type: PatternRecognitionType = PatternRecognitionType.HYBRID
    enable_adaptive_learning: bool = True
    enable_cross_timeframe_analysis: bool = True
    enable_regime_transition_prediction: bool = True
    enable_confidence_calibration: bool = True
    enable_ensemble_voting: bool = True
    lookback_periods: List[int] = field(default_factory=lambda: [5, 15, 30, 60, 120])
    confidence_threshold: float = 0.75
    regime_stability_threshold: float = 0.85
    transition_sensitivity: float = 0.3
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'statistical': 0.3,
        'ml_enhanced': 0.4,
        'pattern_recognition': 0.3
    })

@dataclass
class RegimeFormationResult:
    """Comprehensive regime formation result"""
    regime_type: Enhanced18RegimeType
    confidence_score: float
    stability_score: float
    transition_probability: float
    pattern_strength: float
    ensemble_agreement: float
    formation_timestamp: datetime
    
    # Component scores
    volatility_component: float
    trend_component: float
    structure_component: float
    
    # Advanced metrics
    regime_persistence: float
    cross_timeframe_consistency: float
    adaptive_learning_score: float
    
    # Supporting data
    indicator_contributions: Dict[str, float]
    pattern_matches: List[Dict[str, Any]]
    transition_signals: Dict[str, float]
    metadata: Dict[str, Any]

class SophisticatedRegimeFormationEngine:
    """
    🔄 Phase 5: Sophisticated Regime Formation Enhancement Engine
    
    This engine implements the final phase of the Enhanced Market Regime Framework V2.0
    with advanced algorithms for sophisticated regime formation and pattern recognition.
    """
    
    def __init__(self, config: Optional[SophisticatedRegimeFormationConfig] = None,
                 excel_config_path: Optional[str] = None):
        """
        Initialize Sophisticated Regime Formation Engine
        
        Args:
            config: Sophisticated regime formation configuration
            excel_config_path: Path to Excel configuration file
        """
        set_tag("component", "sophisticated_regime_formation")
        set_tag("phase", "phase_5_final")
        
        self.config = config or SophisticatedRegimeFormationConfig()
        self.excel_config_path = excel_config_path
        
        # Initialize core components
        self.regime_detector = Enhanced18RegimeDetector()
        self.excel_manager = EnhancedConfigurableExcelManager(excel_config_path)
        self.storage = TimeSeriesRegimeStorage("sophisticated_regime_formation.db")
        self.straddle_engine = StraddleAnalysisEngine()
        
        # Advanced components
        self.pattern_recognizer = None
        self.adaptive_learner = None
        self.ensemble_voter = None
        self.transition_predictor = None
        
        # Performance tracking
        self.formation_history = []
        self.performance_metrics = {}
        self.calibration_data = {}
        
        # Initialize advanced components
        self._initialize_advanced_components()
        
        logger.info("🔄 Sophisticated Regime Formation Engine initialized for Phase 5")
        add_breadcrumb(
            message="Sophisticated Regime Formation Engine initialized",
            category="initialization",
            data={"complexity_level": self.config.complexity_level.value}
        )
    
    @track_errors
    def _initialize_advanced_components(self):
        """Initialize advanced regime formation components"""
        try:
            # Initialize pattern recognizer
            self.pattern_recognizer = SophisticatedPatternRecognizer(
                recognition_type=self.config.pattern_recognition_type,
                lookback_periods=self.config.lookback_periods
            )
            
            # Initialize adaptive learner if enabled
            if self.config.enable_adaptive_learning:
                self.adaptive_learner = AdaptiveLearningEngine(
                    learning_rate=0.01,
                    memory_decay=0.95
                )
            
            # Initialize ensemble voter if enabled
            if self.config.enable_ensemble_voting:
                self.ensemble_voter = EnsembleVotingSystem(
                    weights=self.config.ensemble_weights
                )
            
            # Initialize transition predictor if enabled
            if self.config.enable_regime_transition_prediction:
                self.transition_predictor = RegimeTransitionPredictor(
                    sensitivity=self.config.transition_sensitivity
                )
            
            logger.info("✅ Advanced components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing advanced components: {e}")
            capture_exception(e, component="advanced_components_init")
            raise
    
    @track_errors
    async def form_sophisticated_regime(self, market_data: Dict[str, Any],
                                      user_config: Optional[Dict[str, Any]] = None) -> RegimeFormationResult:
        """
        Form sophisticated market regime using advanced algorithms
        
        Args:
            market_data: Comprehensive market data
            user_config: User-specific configuration
            
        Returns:
            RegimeFormationResult: Sophisticated regime formation result
        """
        # Set tags for regime formation context
        set_tag("complexity_level", self.config.complexity_level.value)
        set_tag("pattern_recognition", self.config.pattern_recognition_type.value)
        set_tag("adaptive_learning", str(self.config.enable_adaptive_learning))
        set_tag("ensemble_voting", str(self.config.enable_ensemble_voting))
        
        add_breadcrumb(
            message="Starting sophisticated regime formation",
            category="regime_formation",
            data={"data_keys": list(market_data.keys())}
        )
        
        try:
            # Step 1: Multi-timeframe analysis
            timeframe_results = await self._analyze_multiple_timeframes(market_data)
            
            # Step 2: Pattern recognition
            pattern_results = await self._recognize_regime_patterns(market_data, timeframe_results)
            
            # Step 3: Ensemble voting
            ensemble_result = await self._perform_ensemble_voting(
                timeframe_results, pattern_results, market_data
            )
            
            # Step 4: Adaptive learning adjustment
            if self.adaptive_learner:
                ensemble_result = await self._apply_adaptive_learning(ensemble_result, market_data)
            
            # Step 5: Transition prediction
            transition_signals = await self._predict_regime_transitions(ensemble_result, market_data)
            
            # Step 6: Confidence calibration
            calibrated_result = await self._calibrate_confidence(ensemble_result, transition_signals)
            
            # Step 7: Create comprehensive result
            formation_result = self._create_formation_result(
                calibrated_result, timeframe_results, pattern_results, 
                transition_signals, market_data
            )
            
            # Store result and update learning
            await self._store_and_learn(formation_result, market_data)
            
            logger.info(f"✅ Sophisticated regime formed: {formation_result.regime_type.value} "
                       f"(confidence: {formation_result.confidence_score:.3f})")
            
            return formation_result
            
        except Exception as e:
            error_context = {
                "complexity_level": self.config.complexity_level.value,
                "market_data_keys": list(market_data.keys()),
                "user_config": user_config
            }
            logger.error(f"Error in sophisticated regime formation: {e}")
            capture_exception(e, **error_context)
            raise
    
    @track_errors
    async def _analyze_multiple_timeframes(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime formation across multiple timeframes"""
        try:
            timeframe_results = {}
            
            # Define timeframes for analysis
            timeframes = ['1min', '5min', '15min', '30min', '1hour']
            
            for timeframe in timeframes:
                # Extract timeframe-specific data
                tf_data = self._extract_timeframe_data(market_data, timeframe)
                
                # Perform regime detection for this timeframe
                regime_result = await self._detect_regime_for_timeframe(tf_data, timeframe)
                
                timeframe_results[timeframe] = regime_result
            
            # Calculate cross-timeframe consistency
            consistency_score = self._calculate_timeframe_consistency(timeframe_results)
            timeframe_results['consistency_score'] = consistency_score
            
            add_breadcrumb(
                message="Multi-timeframe analysis completed",
                category="timeframe_analysis",
                data={"timeframes": len(timeframes), "consistency": consistency_score}
            )
            
            return timeframe_results
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            capture_exception(e, component="timeframe_analysis")
            return {}

    @track_errors
    async def _recognize_regime_patterns(self, market_data: Dict[str, Any],
                                       timeframe_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize sophisticated regime patterns"""
        try:
            if not self.pattern_recognizer:
                return {}

            # Extract pattern features from market data
            pattern_features = self._extract_pattern_features(market_data, timeframe_results)

            # Recognize patterns using sophisticated algorithms
            pattern_matches = await self.pattern_recognizer.recognize_patterns(pattern_features)

            # Calculate pattern strength and confidence
            pattern_strength = self._calculate_pattern_strength(pattern_matches)

            result = {
                'pattern_matches': pattern_matches,
                'pattern_strength': pattern_strength,
                'feature_importance': self._calculate_feature_importance(pattern_features),
                'pattern_confidence': self._calculate_pattern_confidence(pattern_matches)
            }

            add_breadcrumb(
                message="Pattern recognition completed",
                category="pattern_recognition",
                data={"patterns_found": len(pattern_matches), "strength": pattern_strength}
            )

            return result

        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            capture_exception(e, component="pattern_recognition")
            return {}

    @track_errors
    async def _perform_ensemble_voting(self, timeframe_results: Dict[str, Any],
                                     pattern_results: Dict[str, Any],
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ensemble voting across different methods"""
        try:
            if not self.ensemble_voter:
                # Fallback to simple averaging
                return self._simple_ensemble_average(timeframe_results, pattern_results)

            # Prepare voting inputs
            voting_inputs = {
                'timeframe_votes': self._extract_timeframe_votes(timeframe_results),
                'pattern_votes': self._extract_pattern_votes(pattern_results),
                'statistical_votes': self._calculate_statistical_votes(market_data),
                'ml_votes': self._calculate_ml_votes(market_data) if self.config.enable_adaptive_learning else {}
            }

            # Perform ensemble voting
            ensemble_result = await self.ensemble_voter.vote(voting_inputs)

            # Calculate ensemble agreement
            agreement_score = self._calculate_ensemble_agreement(voting_inputs, ensemble_result)
            ensemble_result['agreement_score'] = agreement_score

            add_breadcrumb(
                message="Ensemble voting completed",
                category="ensemble_voting",
                data={"agreement": agreement_score, "voters": len(voting_inputs)}
            )

            return ensemble_result

        except Exception as e:
            logger.error(f"Error in ensemble voting: {e}")
            capture_exception(e, component="ensemble_voting")
            return {}

    @track_errors
    async def _apply_adaptive_learning(self, ensemble_result: Dict[str, Any],
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive learning adjustments"""
        try:
            if not self.adaptive_learner:
                return ensemble_result

            # Get historical performance for this regime type
            regime_type = ensemble_result.get('regime_type')
            historical_performance = self._get_historical_performance(regime_type)

            # Apply adaptive adjustments
            adjusted_result = await self.adaptive_learner.adjust_prediction(
                ensemble_result, historical_performance, market_data
            )

            # Update learning score
            learning_score = self._calculate_learning_score(ensemble_result, adjusted_result)
            adjusted_result['adaptive_learning_score'] = learning_score

            add_breadcrumb(
                message="Adaptive learning applied",
                category="adaptive_learning",
                data={"learning_score": learning_score}
            )

            return adjusted_result

        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
            capture_exception(e, component="adaptive_learning")
            return ensemble_result

    @track_errors
    async def _predict_regime_transitions(self, regime_result: Dict[str, Any],
                                        market_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict potential regime transitions"""
        try:
            if not self.transition_predictor:
                return {}

            # Extract transition features
            transition_features = self._extract_transition_features(regime_result, market_data)

            # Predict transitions
            transition_probabilities = await self.transition_predictor.predict_transitions(
                transition_features, regime_result.get('regime_type')
            )

            # Calculate transition signals
            transition_signals = self._calculate_transition_signals(transition_probabilities)

            add_breadcrumb(
                message="Regime transition prediction completed",
                category="transition_prediction",
                data={"transitions_predicted": len(transition_probabilities)}
            )

            return transition_signals

        except Exception as e:
            logger.error(f"Error in transition prediction: {e}")
            capture_exception(e, component="transition_prediction")
            return {}

    @track_errors
    async def _calibrate_confidence(self, regime_result: Dict[str, Any],
                                  transition_signals: Dict[str, float]) -> Dict[str, Any]:
        """Calibrate confidence scores based on historical accuracy"""
        try:
            if not self.config.enable_confidence_calibration:
                return regime_result

            # Get calibration data for this regime type
            regime_type = regime_result.get('regime_type')
            calibration_data = self.calibration_data.get(regime_type, {})

            # Calculate base confidence
            base_confidence = regime_result.get('confidence_score', 0.5)

            # Apply calibration adjustments
            calibrated_confidence = self._apply_confidence_calibration(
                base_confidence, calibration_data, transition_signals
            )

            # Update result
            calibrated_result = regime_result.copy()
            calibrated_result['confidence_score'] = calibrated_confidence
            calibrated_result['calibration_adjustment'] = calibrated_confidence - base_confidence

            add_breadcrumb(
                message="Confidence calibration completed",
                category="confidence_calibration",
                data={
                    "base_confidence": base_confidence,
                    "calibrated_confidence": calibrated_confidence
                }
            )

            return calibrated_result

        except Exception as e:
            logger.error(f"Error in confidence calibration: {e}")
            capture_exception(e, component="confidence_calibration")
            return regime_result

    def _create_formation_result(self, calibrated_result: Dict[str, Any],
                               timeframe_results: Dict[str, Any],
                               pattern_results: Dict[str, Any],
                               transition_signals: Dict[str, float],
                               market_data: Dict[str, Any]) -> RegimeFormationResult:
        """Create comprehensive regime formation result"""
        try:
            # Extract regime type
            regime_type = calibrated_result.get('regime_type', Enhanced18RegimeType.NEUTRAL_BALANCED)
            if isinstance(regime_type, str):
                regime_type = Enhanced18RegimeType(regime_type)

            # Calculate component scores
            volatility_component = self._calculate_volatility_component(market_data)
            trend_component = self._calculate_trend_component(market_data)
            structure_component = self._calculate_structure_component(market_data)

            # Calculate advanced metrics
            regime_persistence = self._calculate_regime_persistence(calibrated_result, timeframe_results)
            cross_timeframe_consistency = timeframe_results.get('consistency_score', 0.5)
            adaptive_learning_score = calibrated_result.get('adaptive_learning_score', 0.5)

            # Create result
            result = RegimeFormationResult(
                regime_type=regime_type,
                confidence_score=calibrated_result.get('confidence_score', 0.5),
                stability_score=calibrated_result.get('stability_score', 0.5),
                transition_probability=max(transition_signals.values()) if transition_signals else 0.0,
                pattern_strength=pattern_results.get('pattern_strength', 0.5),
                ensemble_agreement=calibrated_result.get('agreement_score', 0.5),
                formation_timestamp=datetime.now(),

                # Component scores
                volatility_component=volatility_component,
                trend_component=trend_component,
                structure_component=structure_component,

                # Advanced metrics
                regime_persistence=regime_persistence,
                cross_timeframe_consistency=cross_timeframe_consistency,
                adaptive_learning_score=adaptive_learning_score,

                # Supporting data
                indicator_contributions=calibrated_result.get('indicator_contributions', {}),
                pattern_matches=pattern_results.get('pattern_matches', []),
                transition_signals=transition_signals,
                metadata={
                    'complexity_level': self.config.complexity_level.value,
                    'pattern_recognition_type': self.config.pattern_recognition_type.value,
                    'timeframe_results': timeframe_results,
                    'calibration_applied': self.config.enable_confidence_calibration,
                    'adaptive_learning_enabled': self.config.enable_adaptive_learning
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error creating formation result: {e}")
            capture_exception(e, component="result_creation")
            # Return default result
            return RegimeFormationResult(
                regime_type=Enhanced18RegimeType.NEUTRAL_BALANCED,
                confidence_score=0.5,
                stability_score=0.5,
                transition_probability=0.0,
                pattern_strength=0.5,
                ensemble_agreement=0.5,
                formation_timestamp=datetime.now(),
                volatility_component=0.5,
                trend_component=0.5,
                structure_component=0.5,
                regime_persistence=0.5,
                cross_timeframe_consistency=0.5,
                adaptive_learning_score=0.5,
                indicator_contributions={},
                pattern_matches=[],
                transition_signals={},
                metadata={}
            )

    @track_errors
    async def _store_and_learn(self, formation_result: RegimeFormationResult,
                             market_data: Dict[str, Any]):
        """Store formation result and update learning systems"""
        try:
            # Store in time-series database
            await self._store_formation_result(formation_result)

            # Update adaptive learning if enabled
            if self.adaptive_learner:
                await self.adaptive_learner.update_learning(formation_result, market_data)

            # Update calibration data
            if self.config.enable_confidence_calibration:
                self._update_calibration_data(formation_result)

            # Update performance metrics
            self._update_performance_metrics(formation_result)

            # Add to formation history
            self.formation_history.append(formation_result)

            # Maintain history size
            if len(self.formation_history) > 1000:
                self.formation_history = self.formation_history[-1000:]

            add_breadcrumb(
                message="Formation result stored and learning updated",
                category="storage_learning",
                data={"regime_type": formation_result.regime_type.value}
            )

        except Exception as e:
            logger.error(f"Error storing and learning: {e}")
            capture_exception(e, component="storage_learning")

    # Helper methods for calculations
    def _extract_timeframe_data(self, market_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Extract data for specific timeframe"""
        # Implementation for timeframe data extraction
        return market_data.get(f'{timeframe}_data', market_data)

    async def _detect_regime_for_timeframe(self, tf_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Detect regime for specific timeframe"""
        try:
            # Use existing regime detector
            regime_result = self.regime_detector.detect_regime(tf_data)
            return {
                'timeframe': timeframe,
                'regime_type': regime_result.regime_type,
                'confidence': regime_result.confidence,
                'components': {
                    'volatility': regime_result.volatility_component,
                    'trend': regime_result.trend_component,
                    'structure': regime_result.structure_component
                }
            }
        except Exception as e:
            logger.warning(f"Error detecting regime for {timeframe}: {e}")
            return {'timeframe': timeframe, 'regime_type': Enhanced18RegimeType.NEUTRAL_BALANCED, 'confidence': 0.5}

    def _calculate_timeframe_consistency(self, timeframe_results: Dict[str, Any]) -> float:
        """Calculate consistency across timeframes"""
        try:
            regimes = [result.get('regime_type') for result in timeframe_results.values()
                      if isinstance(result, dict) and 'regime_type' in result]

            if not regimes:
                return 0.5

            # Calculate regime agreement
            unique_regimes = set(regimes)
            if len(unique_regimes) == 1:
                return 1.0
            elif len(unique_regimes) == len(regimes):
                return 0.0
            else:
                # Partial agreement
                most_common = max(set(regimes), key=regimes.count)
                agreement_ratio = regimes.count(most_common) / len(regimes)
                return agreement_ratio

        except Exception as e:
            logger.error(f"Error calculating timeframe consistency: {e}")
            return 0.5
