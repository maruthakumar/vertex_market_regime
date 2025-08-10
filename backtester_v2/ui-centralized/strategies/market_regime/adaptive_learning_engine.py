#!/usr/bin/env python3
"""
Adaptive Learning Engine for Enhanced Market Regime Framework V2.0

This module implements adaptive learning algorithms that continuously improve
regime formation accuracy based on historical performance and market feedback.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from collections import deque
import asyncio
import json
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

@dataclass
class LearningMetrics:
    """Metrics for adaptive learning performance"""
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    confidence_calibration: float
    regime_stability: float
    transition_accuracy: float
    learning_rate_adjustment: float

@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning"""
    learning_rate: float = 0.01
    memory_decay: float = 0.95
    min_samples_for_learning: int = 50
    max_memory_size: int = 1000
    confidence_threshold: float = 0.7
    stability_threshold: float = 0.8
    adaptation_frequency: int = 10  # Learn every N predictions
    enable_online_learning: bool = True
    enable_batch_learning: bool = True
    enable_meta_learning: bool = True

class AdaptiveLearningEngine:
    """
    Adaptive Learning Engine for Market Regime Formation
    
    Implements sophisticated adaptive learning algorithms including:
    - Online learning with immediate feedback
    - Batch learning with historical data
    - Meta-learning for learning rate optimization
    - Confidence calibration
    - Performance-based weight adjustment
    """
    
    def __init__(self, config: Optional[AdaptiveLearningConfig] = None):
        """
        Initialize Adaptive Learning Engine
        
        Args:
            config: Adaptive learning configuration
        """
        set_tag("component", "adaptive_learning_engine")
        
        self.config = config or AdaptiveLearningConfig()
        
        # Learning memory
        self.prediction_memory = deque(maxlen=self.config.max_memory_size)
        self.performance_memory = deque(maxlen=self.config.max_memory_size)
        self.feedback_memory = deque(maxlen=self.config.max_memory_size)
        
        # Learning state
        self.learning_weights = {}
        self.confidence_calibration_params = {}
        self.regime_performance_history = {}
        self.adaptation_counter = 0
        
        # Performance tracking
        self.learning_metrics = LearningMetrics(
            accuracy_score=0.5,
            precision_score=0.5,
            recall_score=0.5,
            f1_score=0.5,
            confidence_calibration=0.5,
            regime_stability=0.5,
            transition_accuracy=0.5,
            learning_rate_adjustment=1.0
        )
        
        logger.info("Adaptive Learning Engine initialized")
        add_breadcrumb(
            message="Adaptive Learning Engine initialized",
            category="initialization",
            data={"learning_rate": self.config.learning_rate}
        )
    
    @track_errors
    async def adjust_prediction(self, prediction_result: Dict[str, Any],
                              historical_performance: Dict[str, Any],
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust prediction based on adaptive learning
        
        Args:
            prediction_result: Original prediction result
            historical_performance: Historical performance data
            market_data: Current market data
            
        Returns:
            Dict: Adjusted prediction result
        """
        set_tag("operation", "prediction_adjustment")
        
        try:
            # Extract prediction components
            regime_type = prediction_result.get('regime_type')
            confidence = prediction_result.get('confidence_score', 0.5)
            
            # Apply learning-based adjustments
            adjusted_result = prediction_result.copy()
            
            # 1. Confidence calibration
            calibrated_confidence = self._calibrate_confidence(confidence, regime_type)
            adjusted_result['confidence_score'] = calibrated_confidence
            
            # 2. Weight adjustment based on historical performance
            weight_adjustments = self._calculate_weight_adjustments(
                regime_type, historical_performance
            )
            adjusted_result['weight_adjustments'] = weight_adjustments
            
            # 3. Stability adjustment
            stability_adjustment = self._calculate_stability_adjustment(
                regime_type, market_data
            )
            adjusted_result['stability_score'] = stability_adjustment
            
            # 4. Meta-learning adjustment
            if self.config.enable_meta_learning:
                meta_adjustment = self._apply_meta_learning(adjusted_result)
                adjusted_result.update(meta_adjustment)
            
            # Store prediction for future learning
            self._store_prediction(adjusted_result, market_data)
            
            add_breadcrumb(
                message="Prediction adjusted using adaptive learning",
                category="prediction_adjustment",
                data={
                    "regime_type": str(regime_type),
                    "original_confidence": confidence,
                    "adjusted_confidence": calibrated_confidence
                }
            )
            
            return adjusted_result
            
        except Exception as e:
            logger.error(f"Error in prediction adjustment: {e}")
            capture_exception(e, component="prediction_adjustment")
            return prediction_result
    
    @track_errors
    async def update_learning(self, formation_result: Any, market_data: Dict[str, Any]):
        """
        Update learning based on formation result feedback
        
        Args:
            formation_result: Regime formation result
            market_data: Market data used for formation
        """
        try:
            # Extract feedback information
            regime_type = formation_result.regime_type
            confidence = formation_result.confidence_score
            actual_performance = self._calculate_actual_performance(formation_result, market_data)
            
            # Store feedback
            feedback_data = {
                'regime_type': regime_type,
                'predicted_confidence': confidence,
                'actual_performance': actual_performance,
                'timestamp': formation_result.formation_timestamp,
                'market_conditions': self._extract_market_conditions(market_data)
            }
            
            self.feedback_memory.append(feedback_data)
            
            # Update learning if enough samples
            self.adaptation_counter += 1
            if self.adaptation_counter >= self.config.adaptation_frequency:
                await self._perform_learning_update()
                self.adaptation_counter = 0
            
            # Update performance metrics
            self._update_performance_metrics()
            
            add_breadcrumb(
                message="Learning updated with feedback",
                category="learning_update",
                data={
                    "regime_type": str(regime_type),
                    "performance": actual_performance
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating learning: {e}")
            capture_exception(e, component="learning_update")
    
    @track_errors
    async def _perform_learning_update(self):
        """Perform batch learning update"""
        try:
            if len(self.feedback_memory) < self.config.min_samples_for_learning:
                return
            
            # Online learning update
            if self.config.enable_online_learning:
                await self._perform_online_learning()
            
            # Batch learning update
            if self.config.enable_batch_learning:
                await self._perform_batch_learning()
            
            # Meta-learning update
            if self.config.enable_meta_learning:
                await self._perform_meta_learning()
            
            logger.info("Learning update completed")
            
        except Exception as e:
            logger.error(f"Error in learning update: {e}")
            capture_exception(e, component="learning_update")
    
    async def _perform_online_learning(self):
        """Perform online learning with recent feedback"""
        try:
            # Get recent feedback
            recent_feedback = list(self.feedback_memory)[-10:]  # Last 10 samples
            
            for feedback in recent_feedback:
                regime_type = feedback['regime_type']
                predicted_confidence = feedback['predicted_confidence']
                actual_performance = feedback['actual_performance']
                
                # Update confidence calibration
                self._update_confidence_calibration(
                    regime_type, predicted_confidence, actual_performance
                )
                
                # Update regime performance history
                self._update_regime_performance(regime_type, actual_performance)
            
        except Exception as e:
            logger.error(f"Error in online learning: {e}")
            capture_exception(e, component="online_learning")
    
    async def _perform_batch_learning(self):
        """Perform batch learning with historical data"""
        try:
            # Convert feedback to DataFrame for analysis
            feedback_df = pd.DataFrame(list(self.feedback_memory))
            
            if feedback_df.empty:
                return
            
            # Group by regime type
            for regime_type in feedback_df['regime_type'].unique():
                regime_data = feedback_df[feedback_df['regime_type'] == regime_type]
                
                # Calculate performance statistics
                mean_performance = regime_data['actual_performance'].mean()
                std_performance = regime_data['actual_performance'].std()
                
                # Update learning weights
                self.learning_weights[str(regime_type)] = {
                    'mean_performance': mean_performance,
                    'std_performance': std_performance,
                    'sample_count': len(regime_data),
                    'last_updated': datetime.now()
                }
            
        except Exception as e:
            logger.error(f"Error in batch learning: {e}")
            capture_exception(e, component="batch_learning")
    
    async def _perform_meta_learning(self):
        """Perform meta-learning to optimize learning parameters"""
        try:
            # Analyze learning rate effectiveness
            if len(self.performance_memory) >= 20:
                recent_performance = list(self.performance_memory)[-20:]
                performance_trend = self._calculate_performance_trend(recent_performance)
                
                # Adjust learning rate based on performance trend
                if performance_trend > 0.1:  # Improving
                    self.config.learning_rate *= 1.05  # Increase slightly
                elif performance_trend < -0.1:  # Degrading
                    self.config.learning_rate *= 0.95  # Decrease slightly
                
                # Keep learning rate within bounds
                self.config.learning_rate = np.clip(self.config.learning_rate, 0.001, 0.1)
                
                self.learning_metrics.learning_rate_adjustment = self.config.learning_rate
            
        except Exception as e:
            logger.error(f"Error in meta-learning: {e}")
            capture_exception(e, component="meta_learning")
    
    def _calibrate_confidence(self, confidence: float, regime_type: Any) -> float:
        """Calibrate confidence based on historical accuracy"""
        try:
            regime_key = str(regime_type)
            
            if regime_key in self.confidence_calibration_params:
                params = self.confidence_calibration_params[regime_key]
                # Apply calibration adjustment
                calibrated = confidence * params.get('adjustment_factor', 1.0)
                calibrated += params.get('bias_correction', 0.0)
                return np.clip(calibrated, 0.0, 1.0)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calibrating confidence: {e}")
            return confidence
    
    def _calculate_weight_adjustments(self, regime_type: Any,
                                    historical_performance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weight adjustments based on historical performance"""
        try:
            regime_key = str(regime_type)
            adjustments = {}
            
            if regime_key in self.learning_weights:
                weights = self.learning_weights[regime_key]
                mean_perf = weights.get('mean_performance', 0.5)
                
                # Adjust weights based on performance
                if mean_perf > 0.7:  # Good performance
                    adjustments['confidence_boost'] = 0.1
                    adjustments['stability_boost'] = 0.05
                elif mean_perf < 0.3:  # Poor performance
                    adjustments['confidence_penalty'] = -0.1
                    adjustments['stability_penalty'] = -0.05
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating weight adjustments: {e}")
            return {}
    
    def _calculate_stability_adjustment(self, regime_type: Any,
                                      market_data: Dict[str, Any]) -> float:
        """Calculate stability adjustment based on market conditions"""
        try:
            # Base stability
            base_stability = 0.5
            
            # Adjust based on market volatility
            volatility = market_data.get('volatility', {}).get('realized_volatility', 0.2)
            if volatility < 0.1:  # Low volatility
                stability_adjustment = 0.2
            elif volatility > 0.3:  # High volatility
                stability_adjustment = -0.2
            else:
                stability_adjustment = 0.0
            
            return np.clip(base_stability + stability_adjustment, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating stability adjustment: {e}")
            return 0.5
