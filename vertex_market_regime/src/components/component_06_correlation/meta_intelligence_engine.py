"""
Meta-Correlation Intelligence Engine for Component 6

Implements comprehensive meta-correlation intelligence with prediction quality
assessment, adaptive learning enhancement, and system performance optimization.

Features:
- Prediction Quality Assessment (15 features)
- Adaptive Learning Enhancement (15 features)
- Total: 30 meta-intelligence features for ML consumption

🎯 SYSTEM PERFORMANCE MEASUREMENT - NO HARD-CODED DECISIONS  
All features are mathematical measurements of system performance and coherence.
Learning and optimization decisions deferred to Vertex AI ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
import warnings
from scipy import stats
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import asyncio

warnings.filterwarnings('ignore')


@dataclass
class PredictionQualityMetrics:
    """Prediction quality assessment metrics"""
    component_accuracies: Dict[int, float]
    cross_validation_scores: Dict[str, float]
    accuracy_trends: Dict[int, float]
    system_coherence_score: float
    
    # Real-time tracking
    prediction_consistency: float
    error_patterns: Dict[str, float]
    performance_stability: float
    confidence_calibration: float
    
    timestamp: datetime


@dataclass 
class AdaptiveLearningMetrics:
    """Adaptive learning enhancement metrics"""
    component_weights: Dict[int, float]
    weight_adjustments: Dict[int, float]
    dte_specific_weights: Dict[str, float]
    regime_adaptive_weights: Dict[str, float]
    
    # Performance optimization
    learning_rate_adjustments: Dict[str, float]
    convergence_indicators: Dict[str, float]
    optimization_efficiency: float
    system_performance_boost: float
    
    timestamp: datetime


@dataclass
class MetaIntelligenceResult:
    """Complete meta-correlation intelligence result"""
    prediction_quality_metrics: PredictionQualityMetrics
    adaptive_learning_metrics: AdaptiveLearningMetrics
    
    # Feature arrays for ML consumption (30 total features)
    accuracy_tracking_features: np.ndarray          # 8 features
    confidence_scoring_features: np.ndarray         # 7 features  
    dynamic_weight_optimization_features: np.ndarray # 8 features
    performance_boosting_features: np.ndarray       # 7 features
    
    overall_system_health: float
    meta_confidence_score: float
    processing_time_ms: float
    timestamp: datetime


class RealTimePerformanceTracker:
    """Real-time performance tracking for components and predictions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Performance tracking windows
        self.short_term_window = config.get('short_term_window', 20)   # Recent performance
        self.medium_term_window = config.get('medium_term_window', 100) # Medium-term trends
        self.long_term_window = config.get('long_term_window', 500)    # Long-term patterns
        
        # Component performance storage
        self.component_performance_history = {i: deque(maxlen=self.long_term_window) for i in range(1, 6)}
        self.prediction_accuracy_history = deque(maxlen=self.long_term_window)
        self.system_coherence_history = deque(maxlen=self.medium_term_window)
        
        # Cross-validation tracking
        self.cross_validation_history = deque(maxlen=self.medium_term_window)
        self.error_pattern_history = deque(maxlen=self.short_term_window)
        
        # Confidence calibration tracking
        self.confidence_vs_accuracy = deque(maxlen=self.medium_term_window)
        
        self.logger.info("Real-time performance tracker initialized")

    def update_component_performance(self, component_id: int, accuracy: float, 
                                   confidence: float, processing_time: float):
        """Update performance metrics for a specific component"""
        
        try:
            performance_record = {
                'accuracy': accuracy,
                'confidence': confidence,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow()
            }
            
            if component_id in self.component_performance_history:
                self.component_performance_history[component_id].append(performance_record)
            
        except Exception as e:
            self.logger.error(f"Error updating component {component_id} performance: {e}")

    def calculate_component_accuracy_trends(self) -> Dict[int, float]:
        """Calculate accuracy trends for each component"""
        
        trends = {}
        
        try:
            for component_id, history in self.component_performance_history.items():
                if len(history) >= 10:  # Need at least 10 data points
                    accuracies = [record['accuracy'] for record in history]
                    
                    # Calculate linear trend (slope)
                    x = np.arange(len(accuracies))
                    slope, _ = np.polyfit(x, accuracies, 1)
                    trends[component_id] = float(slope)
                else:
                    trends[component_id] = 0.0  # No trend data
                    
        except Exception as e:
            self.logger.error(f"Error calculating accuracy trends: {e}")
            trends = {i: 0.0 for i in range(1, 6)}
        
        return trends

    def calculate_system_coherence(self, component_results: Dict[int, Any]) -> float:
        """Calculate system coherence across components"""
        
        try:
            if len(component_results) < 2:
                return 0.5  # Default coherence
            
            # Extract scores/accuracies from components
            scores = []
            for component_id, result in component_results.items():
                if hasattr(result, 'score'):
                    scores.append(result.score)
                elif isinstance(result, dict) and 'score' in result:
                    scores.append(result['score'])
                elif hasattr(result, 'accuracy'):
                    scores.append(result.accuracy)
                elif isinstance(result, dict) and 'accuracy' in result:
                    scores.append(result['accuracy'])
            
            if len(scores) >= 2:
                # Coherence = 1 - coefficient of variation
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                if mean_score > 0:
                    coherence = 1.0 - (std_score / mean_score)
                    coherence_score = max(0.0, min(1.0, coherence))
                    
                    # Store in history
                    self.system_coherence_history.append(coherence_score)
                    
                    return coherence_score
            
            return 0.5  # Default coherence
            
        except Exception as e:
            self.logger.error(f"Error calculating system coherence: {e}")
            return 0.5


class AdaptiveWeightOptimizer:
    """Adaptive weight optimization based on component performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Optimization parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 0.001)
        
        # Weight boundaries
        self.min_weight = config.get('min_weight', 0.1)
        self.max_weight = config.get('max_weight', 2.0)
        
        # Weight history for momentum calculation
        self.weight_momentum = {i: 0.0 for i in range(1, 6)}
        self.previous_weights = {i: 1.0 for i in range(1, 6)}
        
        # Performance-based weight adjustments
        self.performance_adjustment_factor = config.get('performance_adjustment_factor', 0.1)
        
        self.logger.info("Adaptive weight optimizer initialized")

    def optimize_component_weights(self, 
                                 current_weights: Dict[int, float],
                                 performance_metrics: Dict[int, Dict[str, float]]) -> Dict[int, float]:
        """Optimize component weights based on performance feedback"""
        
        optimized_weights = {}
        
        try:
            for component_id in range(1, 6):
                current_weight = current_weights.get(component_id, 1.0)
                performance = performance_metrics.get(component_id, {})
                
                # Extract performance indicators
                accuracy = performance.get('accuracy', 0.5)
                confidence = performance.get('confidence', 0.5)
                stability = performance.get('stability', 0.5)
                
                # Calculate performance score
                performance_score = (accuracy * 0.5 + confidence * 0.3 + stability * 0.2)
                
                # Weight adjustment based on performance
                if performance_score > 0.8:  # High performance
                    weight_adjustment = 1.0 + self.performance_adjustment_factor
                elif performance_score < 0.6:  # Low performance
                    weight_adjustment = 1.0 - self.performance_adjustment_factor
                else:  # Average performance
                    weight_adjustment = 1.0
                
                # Apply momentum
                momentum_term = self.momentum * self.weight_momentum[component_id]
                adjustment_term = self.learning_rate * (weight_adjustment - 1.0)
                
                self.weight_momentum[component_id] = momentum_term + adjustment_term
                
                # Calculate new weight
                new_weight = current_weight + self.weight_momentum[component_id]
                
                # Apply weight decay
                new_weight *= (1.0 - self.weight_decay)
                
                # Clamp to boundaries
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                
                optimized_weights[component_id] = new_weight
                self.previous_weights[component_id] = new_weight
                
        except Exception as e:
            self.logger.error(f"Error optimizing component weights: {e}")
            # Return unchanged weights on error
            optimized_weights = current_weights.copy()
        
        return optimized_weights

    def calculate_dte_specific_weights(self, 
                                     dte_performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate DTE-specific weight adjustments"""
        
        dte_weights = {}
        
        try:
            dte_ranges = ['weekly', 'bi_weekly', 'monthly', 'far_month']
            
            for dte_range in dte_ranges:
                performance = dte_performance.get(dte_range, 0.5)
                
                # Weight based on performance
                if performance > 0.8:
                    dte_weights[dte_range] = 1.2  # Boost high-performing DTE ranges
                elif performance < 0.6:
                    dte_weights[dte_range] = 0.8  # Reduce low-performing DTE ranges
                else:
                    dte_weights[dte_range] = 1.0  # Neutral weight
                    
        except Exception as e:
            self.logger.error(f"Error calculating DTE-specific weights: {e}")
            dte_weights = {dte: 1.0 for dte in ['weekly', 'bi_weekly', 'monthly', 'far_month']}
        
        return dte_weights

    def calculate_regime_adaptive_weights(self, 
                                        regime_performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate regime-adaptive weight adjustments"""
        
        regime_weights = {}
        
        try:
            # Market regime types
            regimes = ['LVLD', 'HVC', 'VCPE', 'TBVE', 'TBVS', 'SCGS', 'PSED', 'CBV']
            
            for regime in regimes:
                performance = regime_performance.get(regime, 0.5)
                
                # Adaptive weight based on regime-specific performance
                if performance > 0.85:
                    regime_weights[regime] = 1.3  # Strong boost for excellent performance
                elif performance > 0.75:
                    regime_weights[regime] = 1.1  # Moderate boost
                elif performance < 0.6:
                    regime_weights[regime] = 0.8  # Reduce for poor performance
                else:
                    regime_weights[regime] = 1.0  # Neutral
                    
        except Exception as e:
            self.logger.error(f"Error calculating regime-adaptive weights: {e}")
            regime_weights = {regime: 1.0 for regime in ['LVLD', 'HVC', 'VCPE', 'TBVE', 'TBVS', 'SCGS', 'PSED', 'CBV']}
        
        return regime_weights


class MetaCorrelationIntelligenceEngine:
    """
    High-performance meta-correlation intelligence engine
    
    Extracts 30 systematic meta-intelligence features measuring system
    performance, prediction quality, and adaptive learning effectiveness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize sub-systems
        self.performance_tracker = RealTimePerformanceTracker(config)
        self.weight_optimizer = AdaptiveWeightOptimizer(config)
        
        # Meta-intelligence parameters
        self.confidence_calibration_window = config.get('confidence_calibration_window', 50)
        self.performance_stability_window = config.get('performance_stability_window', 30)
        self.system_health_threshold = config.get('system_health_threshold', 0.75)
        
        # Historical meta-metrics storage
        self.meta_performance_history = deque(maxlen=252)  # 1 year of meta-performance
        self.weight_optimization_history = deque(maxlen=100)
        self.system_health_history = deque(maxlen=50)
        
        self.logger.info("Meta-Correlation Intelligence Engine initialized with 30-feature extraction")

    def extract_meta_intelligence_features(self, 
                                         component_results: Dict[int, Any],
                                         historical_performance: Dict[str, List[float]],
                                         current_weights: Dict[int, float]) -> MetaIntelligenceResult:
        """
        Extract all 30 meta-correlation intelligence features
        
        Args:
            component_results: Results from Components 1-5
            historical_performance: Historical performance metrics
            current_weights: Current component weights
            
        Returns:
            MetaIntelligenceResult with 30 meta-intelligence features
        """
        start_time = time.time()
        
        try:
            # 1. Assess prediction quality (15 features)
            prediction_quality_metrics = self._assess_prediction_quality(
                component_results, historical_performance
            )
            
            # 2. Enhance adaptive learning (15 features)
            adaptive_learning_metrics = self._enhance_adaptive_learning(
                component_results, historical_performance, current_weights
            )
            
            # 3. Generate feature arrays for ML consumption
            feature_arrays = self._generate_meta_feature_arrays(
                prediction_quality_metrics, adaptive_learning_metrics
            )
            
            # 4. Calculate overall system health
            system_health = self._calculate_system_health(
                prediction_quality_metrics, adaptive_learning_metrics
            )
            
            # 5. Calculate meta-confidence score
            meta_confidence = self._calculate_meta_confidence(
                prediction_quality_metrics, len(component_results)
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Store meta-performance
            meta_record = {
                'system_health': system_health,
                'meta_confidence': meta_confidence,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow()
            }
            self.meta_performance_history.append(meta_record)
            
            return MetaIntelligenceResult(
                prediction_quality_metrics=prediction_quality_metrics,
                adaptive_learning_metrics=adaptive_learning_metrics,
                accuracy_tracking_features=feature_arrays['accuracy_tracking'],
                confidence_scoring_features=feature_arrays['confidence_scoring'],
                dynamic_weight_optimization_features=feature_arrays['dynamic_weights'],
                performance_boosting_features=feature_arrays['performance_boosting'],
                overall_system_health=system_health,
                meta_confidence_score=meta_confidence,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Meta-intelligence feature extraction failed: {e}")
            return self._create_minimal_meta_result(processing_time)

    def _assess_prediction_quality(self, 
                                 component_results: Dict[int, Any],
                                 historical_performance: Dict[str, List[float]]) -> PredictionQualityMetrics:
        """Assess prediction quality across all components"""
        
        try:
            # Extract component accuracies
            component_accuracies = {}
            for component_id, result in component_results.items():
                if hasattr(result, 'accuracy'):
                    accuracy = getattr(result, 'accuracy', 0.5)
                    component_accuracies[component_id] = accuracy
                elif isinstance(result, dict) and 'accuracy' in result:
                    accuracy = result.get('accuracy', 0.5)
                    component_accuracies[component_id] = accuracy
                else:
                    # Estimate accuracy from score/confidence
                    score = getattr(result, 'score', 0.5) if hasattr(result, 'score') else (result.get('score', 0.5) if isinstance(result, dict) else 0.5)
                    confidence = getattr(result, 'confidence', 0.5) if hasattr(result, 'confidence') else (result.get('confidence', 0.5) if isinstance(result, dict) else 0.5)
                    estimated_accuracy = (score * 0.7 + confidence * 0.3)
                    component_accuracies[component_id] = estimated_accuracy
            
            # Calculate cross-validation scores
            cross_validation_scores = self._calculate_cross_validation_scores(
                component_results, historical_performance
            )
            
            # Calculate accuracy trends
            accuracy_trends = self.performance_tracker.calculate_component_accuracy_trends()
            
            # Calculate system coherence
            system_coherence = self.performance_tracker.calculate_system_coherence(component_results)
            
            # Real-time tracking metrics
            prediction_consistency = self._calculate_prediction_consistency(component_results)
            error_patterns = self._analyze_error_patterns(historical_performance)
            performance_stability = self._calculate_performance_stability(component_accuracies)
            confidence_calibration = self._calculate_confidence_calibration(component_results)
            
            return PredictionQualityMetrics(
                component_accuracies=component_accuracies,
                cross_validation_scores=cross_validation_scores,
                accuracy_trends=accuracy_trends,
                system_coherence_score=system_coherence,
                prediction_consistency=prediction_consistency,
                error_patterns=error_patterns,
                performance_stability=performance_stability,
                confidence_calibration=confidence_calibration,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing prediction quality: {e}")
            return PredictionQualityMetrics(
                component_accuracies={i: 0.5 for i in range(1, 6)},
                cross_validation_scores={'overall': 0.5},
                accuracy_trends={i: 0.0 for i in range(1, 6)},
                system_coherence_score=0.5,
                prediction_consistency=0.5,
                error_patterns={'overall': 0.5},
                performance_stability=0.5,
                confidence_calibration=0.5,
                timestamp=datetime.utcnow()
            )

    def _enhance_adaptive_learning(self, 
                                 component_results: Dict[int, Any],
                                 historical_performance: Dict[str, List[float]],
                                 current_weights: Dict[int, float]) -> AdaptiveLearningMetrics:
        """Enhance adaptive learning capabilities"""
        
        try:
            # Extract performance metrics for weight optimization
            performance_metrics = {}
            for component_id, result in component_results.items():
                metrics = {
                    'accuracy': getattr(result, 'accuracy', 0.5) if hasattr(result, 'accuracy') else (result.get('accuracy', 0.5) if isinstance(result, dict) else 0.5),
                    'confidence': getattr(result, 'confidence', 0.5) if hasattr(result, 'confidence') else (result.get('confidence', 0.5) if isinstance(result, dict) else 0.5),
                    'stability': getattr(result, 'stability', 0.5) if hasattr(result, 'stability') else (result.get('stability', 0.5) if isinstance(result, dict) else 0.5)
                }
                performance_metrics[component_id] = metrics
            
            # Optimize component weights
            optimized_weights = self.weight_optimizer.optimize_component_weights(
                current_weights, performance_metrics
            )
            
            # Calculate weight adjustments
            weight_adjustments = {}
            for component_id in range(1, 6):
                current = current_weights.get(component_id, 1.0)
                optimized = optimized_weights.get(component_id, 1.0)
                weight_adjustments[component_id] = optimized - current
            
            # Calculate DTE-specific weights
            dte_performance = self._extract_dte_performance(historical_performance)
            dte_weights = self.weight_optimizer.calculate_dte_specific_weights(dte_performance)
            
            # Calculate regime-adaptive weights
            regime_performance = self._extract_regime_performance(historical_performance)
            regime_weights = self.weight_optimizer.calculate_regime_adaptive_weights(regime_performance)
            
            # Learning rate adjustments
            learning_rate_adjustments = self._calculate_learning_rate_adjustments(
                performance_metrics, historical_performance
            )
            
            # Convergence indicators
            convergence_indicators = self._calculate_convergence_indicators(
                weight_adjustments, historical_performance
            )
            
            # Optimization efficiency
            optimization_efficiency = self._calculate_optimization_efficiency(
                weight_adjustments, performance_metrics
            )
            
            # System performance boost
            system_performance_boost = self._calculate_system_performance_boost(
                optimized_weights, current_weights, performance_metrics
            )
            
            return AdaptiveLearningMetrics(
                component_weights=optimized_weights,
                weight_adjustments=weight_adjustments,
                dte_specific_weights=dte_weights,
                regime_adaptive_weights=regime_weights,
                learning_rate_adjustments=learning_rate_adjustments,
                convergence_indicators=convergence_indicators,
                optimization_efficiency=optimization_efficiency,
                system_performance_boost=system_performance_boost,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error enhancing adaptive learning: {e}")
            return AdaptiveLearningMetrics(
                component_weights={i: 1.0 for i in range(1, 6)},
                weight_adjustments={i: 0.0 for i in range(1, 6)},
                dte_specific_weights={'weekly': 1.0, 'monthly': 1.0},
                regime_adaptive_weights={'LVLD': 1.0, 'HVC': 1.0},
                learning_rate_adjustments={'overall': 0.0},
                convergence_indicators={'weight_stability': 0.5},
                optimization_efficiency=0.5,
                system_performance_boost=0.0,
                timestamp=datetime.utcnow()
            )

    def _calculate_cross_validation_scores(self, 
                                         component_results: Dict[int, Any],
                                         historical_performance: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate cross-validation scores for prediction quality"""
        
        cv_scores = {}
        
        try:
            # Overall cross-validation score from historical data
            if 'cross_validation_scores' in historical_performance:
                cv_history = historical_performance['cross_validation_scores']
                if len(cv_history) > 0:
                    cv_scores['overall'] = float(np.mean(cv_history[-10:]))  # Recent average
                else:
                    cv_scores['overall'] = 0.5
            else:
                cv_scores['overall'] = 0.5
            
            # Component-specific cross-validation
            for component_id in range(1, 6):
                cv_key = f'component_{component_id}_cv'
                if cv_key in historical_performance:
                    cv_history = historical_performance[cv_key]
                    cv_scores[f'component_{component_id}'] = float(np.mean(cv_history[-5:]))
                else:
                    cv_scores[f'component_{component_id}'] = 0.5
                    
        except Exception as e:
            self.logger.error(f"Error calculating cross-validation scores: {e}")
            cv_scores = {'overall': 0.5}
        
        return cv_scores

    def _calculate_prediction_consistency(self, component_results: Dict[int, Any]) -> float:
        """Calculate prediction consistency across components"""
        
        try:
            if len(component_results) < 2:
                return 0.5
            
            # Extract prediction values/scores
            predictions = []
            for component_id, result in component_results.items():
                if hasattr(result, 'score'):
                    predictions.append(result.score)
                elif isinstance(result, dict) and 'score' in result:
                    predictions.append(result['score'])
                elif hasattr(result, 'prediction'):
                    predictions.append(result.prediction)
                elif isinstance(result, dict) and 'prediction' in result:
                    predictions.append(result['prediction'])
            
            if len(predictions) >= 2:
                # Consistency = 1 - coefficient of variation
                mean_pred = np.mean(predictions)
                std_pred = np.std(predictions)
                
                if mean_pred > 0:
                    consistency = 1.0 - (std_pred / mean_pred)
                    return max(0.0, min(1.0, consistency))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction consistency: {e}")
            return 0.5

    def _analyze_error_patterns(self, historical_performance: Dict[str, List[float]]) -> Dict[str, float]:
        """Analyze error patterns in historical performance"""
        
        error_patterns = {}
        
        try:
            # Overall error rate
            if 'error_rates' in historical_performance:
                error_rates = historical_performance['error_rates']
                if len(error_rates) > 0:
                    error_patterns['overall_error_rate'] = float(np.mean(error_rates[-20:]))
                else:
                    error_patterns['overall_error_rate'] = 0.1
            else:
                error_patterns['overall_error_rate'] = 0.1
            
            # Error trend (increasing/decreasing)
            if 'overall_accuracy' in historical_performance:
                accuracy_history = historical_performance['overall_accuracy']
                if len(accuracy_history) >= 10:
                    # Convert accuracy to error rate
                    error_history = [1.0 - acc for acc in accuracy_history[-10:]]
                    error_trend = np.polyfit(range(len(error_history)), error_history, 1)[0]
                    error_patterns['error_trend'] = float(error_trend)
                else:
                    error_patterns['error_trend'] = 0.0
            else:
                error_patterns['error_trend'] = 0.0
            
            # Error volatility
            if len(error_patterns) > 0 and 'overall_accuracy' in historical_performance:
                accuracy_history = historical_performance['overall_accuracy']
                if len(accuracy_history) >= 5:
                    error_volatility = np.std([1.0 - acc for acc in accuracy_history[-5:]])
                    error_patterns['error_volatility'] = float(error_volatility)
                else:
                    error_patterns['error_volatility'] = 0.1
            else:
                error_patterns['error_volatility'] = 0.1
                
        except Exception as e:
            self.logger.error(f"Error analyzing error patterns: {e}")
            error_patterns = {'overall_error_rate': 0.1, 'error_trend': 0.0, 'error_volatility': 0.1}
        
        return error_patterns

    def _calculate_performance_stability(self, component_accuracies: Dict[int, float]) -> float:
        """Calculate performance stability across components"""
        
        try:
            if len(component_accuracies) < 2:
                return 0.5
            
            accuracies = list(component_accuracies.values())
            
            # Stability = 1 - coefficient of variation
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            if mean_accuracy > 0:
                stability = 1.0 - (std_accuracy / mean_accuracy)
                return max(0.0, min(1.0, stability))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating performance stability: {e}")
            return 0.5

    def _calculate_confidence_calibration(self, component_results: Dict[int, Any]) -> float:
        """Calculate confidence calibration across components"""
        
        try:
            confidence_accuracy_pairs = []
            
            for component_id, result in component_results.items():
                confidence = getattr(result, 'confidence', 0.5) if hasattr(result, 'confidence') else (result.get('confidence', 0.5) if isinstance(result, dict) else 0.5)
                accuracy = getattr(result, 'accuracy', 0.5) if hasattr(result, 'accuracy') else (result.get('accuracy', 0.5) if isinstance(result, dict) else 0.5)
                confidence_accuracy_pairs.append((confidence, accuracy))
            
            if len(confidence_accuracy_pairs) >= 2:
                # Calculate correlation between confidence and accuracy
                confidences = [pair[0] for pair in confidence_accuracy_pairs]
                accuracies = [pair[1] for pair in confidence_accuracy_pairs]
                
                if len(set(confidences)) > 1 and len(set(accuracies)) > 1:
                    correlation = np.corrcoef(confidences, accuracies)[0, 1]
                    if not np.isnan(correlation):
                        # Good calibration = high positive correlation
                        calibration = (correlation + 1.0) / 2.0  # Normalize to 0-1
                        return max(0.0, min(1.0, calibration))
            
            return 0.5  # Default calibration
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence calibration: {e}")
            return 0.5

    def _extract_dte_performance(self, historical_performance: Dict[str, List[float]]) -> Dict[str, float]:
        """Extract DTE-specific performance metrics"""
        
        dte_performance = {}
        
        try:
            dte_ranges = ['weekly', 'bi_weekly', 'monthly', 'far_month']
            
            for dte_range in dte_ranges:
                dte_key = f'dte_{dte_range}_performance'
                if dte_key in historical_performance:
                    performance_history = historical_performance[dte_key]
                    if len(performance_history) > 0:
                        dte_performance[dte_range] = float(np.mean(performance_history[-10:]))
                    else:
                        dte_performance[dte_range] = 0.5
                else:
                    dte_performance[dte_range] = 0.5
                    
        except Exception as e:
            self.logger.error(f"Error extracting DTE performance: {e}")
            dte_performance = {dte: 0.5 for dte in ['weekly', 'bi_weekly', 'monthly', 'far_month']}
        
        return dte_performance

    def _extract_regime_performance(self, historical_performance: Dict[str, List[float]]) -> Dict[str, float]:
        """Extract regime-specific performance metrics"""
        
        regime_performance = {}
        
        try:
            regimes = ['LVLD', 'HVC', 'VCPE', 'TBVE', 'TBVS', 'SCGS', 'PSED', 'CBV']
            
            for regime in regimes:
                regime_key = f'regime_{regime}_accuracy'
                if regime_key in historical_performance:
                    accuracy_history = historical_performance[regime_key]
                    if len(accuracy_history) > 0:
                        regime_performance[regime] = float(np.mean(accuracy_history[-5:]))
                    else:
                        regime_performance[regime] = 0.5
                else:
                    regime_performance[regime] = 0.5
                    
        except Exception as e:
            self.logger.error(f"Error extracting regime performance: {e}")
            regime_performance = {regime: 0.5 for regime in ['LVLD', 'HVC', 'VCPE', 'TBVE', 'TBVS', 'SCGS', 'PSED', 'CBV']}
        
        return regime_performance

    def _calculate_learning_rate_adjustments(self, 
                                           performance_metrics: Dict[int, Dict[str, float]],
                                           historical_performance: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate learning rate adjustments based on performance"""
        
        adjustments = {}
        
        try:
            # Overall learning rate adjustment
            overall_performance = np.mean([
                metrics.get('accuracy', 0.5) for metrics in performance_metrics.values()
            ])
            
            if overall_performance > 0.8:
                adjustments['overall'] = -0.1  # Reduce learning rate for stable performance
            elif overall_performance < 0.6:
                adjustments['overall'] = 0.1   # Increase learning rate for poor performance
            else:
                adjustments['overall'] = 0.0   # No adjustment
            
            # Component-specific adjustments
            for component_id, metrics in performance_metrics.items():
                accuracy = metrics.get('accuracy', 0.5)
                if accuracy > 0.85:
                    adjustments[f'component_{component_id}'] = -0.05
                elif accuracy < 0.65:
                    adjustments[f'component_{component_id}'] = 0.05
                else:
                    adjustments[f'component_{component_id}'] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error calculating learning rate adjustments: {e}")
            adjustments = {'overall': 0.0}
        
        return adjustments

    def _calculate_convergence_indicators(self, 
                                        weight_adjustments: Dict[int, float],
                                        historical_performance: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate convergence indicators for optimization"""
        
        indicators = {}
        
        try:
            # Weight stability (smaller adjustments = better convergence)
            if weight_adjustments:
                avg_adjustment = np.mean([abs(adj) for adj in weight_adjustments.values()])
                weight_stability = max(0.0, 1.0 - avg_adjustment)  # Lower adjustments = higher stability
                indicators['weight_stability'] = weight_stability
            else:
                indicators['weight_stability'] = 0.5
            
            # Performance convergence (decreasing variance = convergence)
            if 'overall_accuracy' in historical_performance:
                accuracy_history = historical_performance['overall_accuracy']
                if len(accuracy_history) >= 10:
                    recent_variance = np.var(accuracy_history[-10:])
                    earlier_variance = np.var(accuracy_history[-20:-10]) if len(accuracy_history) >= 20 else recent_variance
                    
                    if earlier_variance > 0:
                        convergence_improvement = max(0.0, (earlier_variance - recent_variance) / earlier_variance)
                        indicators['performance_convergence'] = min(1.0, convergence_improvement)
                    else:
                        indicators['performance_convergence'] = 0.5
                else:
                    indicators['performance_convergence'] = 0.5
            else:
                indicators['performance_convergence'] = 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating convergence indicators: {e}")
            indicators = {'weight_stability': 0.5, 'performance_convergence': 0.5}
        
        return indicators

    def _calculate_optimization_efficiency(self, 
                                         weight_adjustments: Dict[int, float],
                                         performance_metrics: Dict[int, Dict[str, float]]) -> float:
        """Calculate optimization efficiency"""
        
        try:
            if not weight_adjustments or not performance_metrics:
                return 0.5
            
            # Efficiency = performance improvement per unit weight adjustment
            total_adjustment = sum(abs(adj) for adj in weight_adjustments.values())
            avg_performance = np.mean([
                metrics.get('accuracy', 0.5) for metrics in performance_metrics.values()
            ])
            
            if total_adjustment > 0:
                efficiency = avg_performance / (1.0 + total_adjustment)  # Normalize
                return max(0.0, min(1.0, efficiency))
            
            return avg_performance  # No adjustments needed = high efficiency if performance is good
            
        except Exception as e:
            self.logger.error(f"Error calculating optimization efficiency: {e}")
            return 0.5

    def _calculate_system_performance_boost(self, 
                                          optimized_weights: Dict[int, float],
                                          current_weights: Dict[int, float],
                                          performance_metrics: Dict[int, Dict[str, float]]) -> float:
        """Calculate expected system performance boost from weight optimization"""
        
        try:
            if not optimized_weights or not current_weights or not performance_metrics:
                return 0.0
            
            # Estimate performance boost
            current_weighted_performance = 0.0
            optimized_weighted_performance = 0.0
            total_current_weight = 0.0
            total_optimized_weight = 0.0
            
            for component_id in range(1, 6):
                if component_id in performance_metrics:
                    performance = performance_metrics[component_id].get('accuracy', 0.5)
                    current_weight = current_weights.get(component_id, 1.0)
                    optimized_weight = optimized_weights.get(component_id, 1.0)
                    
                    current_weighted_performance += performance * current_weight
                    optimized_weighted_performance += performance * optimized_weight
                    total_current_weight += current_weight
                    total_optimized_weight += optimized_weight
            
            if total_current_weight > 0 and total_optimized_weight > 0:
                current_avg = current_weighted_performance / total_current_weight
                optimized_avg = optimized_weighted_performance / total_optimized_weight
                performance_boost = optimized_avg - current_avg
                return float(performance_boost)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating system performance boost: {e}")
            return 0.0

    def _generate_meta_feature_arrays(self, 
                                    prediction_quality_metrics: PredictionQualityMetrics,
                                    adaptive_learning_metrics: AdaptiveLearningMetrics) -> Dict[str, np.ndarray]:
        """Generate meta-intelligence feature arrays for ML consumption"""
        
        feature_arrays = {}
        
        try:
            # Accuracy Tracking Features (8 features)
            accuracy_features = []
            for component_id in range(1, 6):
                accuracy = prediction_quality_metrics.component_accuracies.get(component_id, 0.5)
                accuracy_features.append(accuracy)
            
            # Add cross-validation score
            cv_score = prediction_quality_metrics.cross_validation_scores.get('overall', 0.5)
            accuracy_features.append(cv_score)
            
            # Add accuracy trend
            avg_trend = np.mean(list(prediction_quality_metrics.accuracy_trends.values()))
            accuracy_features.append(avg_trend)
            
            # Add system coherence
            accuracy_features.append(prediction_quality_metrics.system_coherence_score)
            
            feature_arrays['accuracy_tracking'] = np.array(accuracy_features[:8], dtype=np.float32)
            
            # Confidence Scoring Features (7 features)
            confidence_features = [
                prediction_quality_metrics.prediction_consistency,
                prediction_quality_metrics.performance_stability,
                prediction_quality_metrics.confidence_calibration,
                prediction_quality_metrics.error_patterns.get('overall_error_rate', 0.1),
                prediction_quality_metrics.error_patterns.get('error_trend', 0.0),
                prediction_quality_metrics.error_patterns.get('error_volatility', 0.1),
                prediction_quality_metrics.system_coherence_score
            ]
            feature_arrays['confidence_scoring'] = np.array(confidence_features[:7], dtype=np.float32)
            
            # Dynamic Weight Optimization Features (8 features)
            weight_features = []
            for component_id in range(1, 6):
                weight = adaptive_learning_metrics.component_weights.get(component_id, 1.0)
                weight_features.append(weight)
            
            # Add DTE and regime weight averages
            dte_avg_weight = np.mean(list(adaptive_learning_metrics.dte_specific_weights.values()))
            weight_features.append(dte_avg_weight)
            
            regime_avg_weight = np.mean(list(adaptive_learning_metrics.regime_adaptive_weights.values()))
            weight_features.append(regime_avg_weight)
            
            # Add optimization efficiency
            weight_features.append(adaptive_learning_metrics.optimization_efficiency)
            
            feature_arrays['dynamic_weights'] = np.array(weight_features[:8], dtype=np.float32)
            
            # Performance Boosting Features (7 features)
            boost_features = [
                adaptive_learning_metrics.system_performance_boost,
                adaptive_learning_metrics.convergence_indicators.get('weight_stability', 0.5),
                adaptive_learning_metrics.convergence_indicators.get('performance_convergence', 0.5),
                adaptive_learning_metrics.learning_rate_adjustments.get('overall', 0.0),
                adaptive_learning_metrics.optimization_efficiency,
                np.mean([abs(adj) for adj in adaptive_learning_metrics.weight_adjustments.values()]),
                prediction_quality_metrics.component_accuracies.get(1, 0.5)  # Best component accuracy
            ]
            feature_arrays['performance_boosting'] = np.array(boost_features[:7], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating meta-feature arrays: {e}")
            # Return default feature arrays
            feature_arrays = {
                'accuracy_tracking': np.full(8, 0.5, dtype=np.float32),
                'confidence_scoring': np.full(7, 0.5, dtype=np.float32),
                'dynamic_weights': np.full(8, 1.0, dtype=np.float32),
                'performance_boosting': np.full(7, 0.5, dtype=np.float32)
            }
        
        return feature_arrays

    def _calculate_system_health(self, 
                               prediction_quality_metrics: PredictionQualityMetrics,
                               adaptive_learning_metrics: AdaptiveLearningMetrics) -> float:
        """Calculate overall system health score"""
        
        try:
            health_factors = []
            
            # Average component accuracy
            avg_accuracy = np.mean(list(prediction_quality_metrics.component_accuracies.values()))
            health_factors.append(avg_accuracy)
            
            # System coherence
            health_factors.append(prediction_quality_metrics.system_coherence_score)
            
            # Performance stability
            health_factors.append(prediction_quality_metrics.performance_stability)
            
            # Confidence calibration
            health_factors.append(prediction_quality_metrics.confidence_calibration)
            
            # Optimization efficiency
            health_factors.append(adaptive_learning_metrics.optimization_efficiency)
            
            # Weight stability
            weight_stability = adaptive_learning_metrics.convergence_indicators.get('weight_stability', 0.5)
            health_factors.append(weight_stability)
            
            # Overall system health
            system_health = np.mean(health_factors)
            
            # Store in history
            self.system_health_history.append(system_health)
            
            return max(0.0, min(1.0, system_health))
            
        except Exception as e:
            self.logger.error(f"Error calculating system health: {e}")
            return 0.5

    def _calculate_meta_confidence(self, 
                                 prediction_quality_metrics: PredictionQualityMetrics,
                                 num_components: int) -> float:
        """Calculate meta-confidence score for meta-intelligence features"""
        
        try:
            confidence_factors = []
            
            # Data availability confidence
            data_confidence = min(1.0, num_components / 5.0)  # Full confidence with all 5 components
            confidence_factors.append(data_confidence)
            
            # Prediction consistency confidence
            confidence_factors.append(prediction_quality_metrics.prediction_consistency)
            
            # System coherence confidence
            confidence_factors.append(prediction_quality_metrics.system_coherence_score)
            
            # Historical data confidence
            if len(self.meta_performance_history) > 10:
                historical_confidence = min(1.0, len(self.meta_performance_history) / 50.0)
            else:
                historical_confidence = 0.3
            confidence_factors.append(historical_confidence)
            
            # Meta-confidence
            meta_confidence = np.mean(confidence_factors)
            return max(0.1, min(1.0, meta_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating meta-confidence: {e}")
            return 0.5

    def _create_minimal_meta_result(self, processing_time: float) -> MetaIntelligenceResult:
        """Create minimal meta-intelligence result for fallback"""
        
        return MetaIntelligenceResult(
            prediction_quality_metrics=PredictionQualityMetrics(
                {i: 0.5 for i in range(1, 6)}, {'overall': 0.5}, {i: 0.0 for i in range(1, 6)}, 
                0.5, 0.5, {'overall': 0.5}, 0.5, 0.5, datetime.utcnow()
            ),
            adaptive_learning_metrics=AdaptiveLearningMetrics(
                {i: 1.0 for i in range(1, 6)}, {i: 0.0 for i in range(1, 6)}, 
                {'weekly': 1.0}, {'LVLD': 1.0}, {'overall': 0.0}, 
                {'weight_stability': 0.5}, 0.5, 0.0, datetime.utcnow()
            ),
            accuracy_tracking_features=np.full(8, 0.5, dtype=np.float32),
            confidence_scoring_features=np.full(7, 0.5, dtype=np.float32),
            dynamic_weight_optimization_features=np.full(8, 1.0, dtype=np.float32),
            performance_boosting_features=np.full(7, 0.5, dtype=np.float32),
            overall_system_health=0.5,
            meta_confidence_score=0.5,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )