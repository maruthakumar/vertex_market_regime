"""
Performance Feedback System

This module implements comprehensive performance tracking and feedback loops
for the adaptive market regime formation system.

Key Features:
- Real-time performance monitoring across all components
- Adaptive feedback mechanisms for continuous improvement
- Performance-based parameter adjustment
- Multi-dimensional performance analysis
- Historical performance tracking and trends
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import json

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    REGIME_STABILITY = "regime_stability"
    TRANSITION_QUALITY = "transition_quality"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    ADAPTATION_SPEED = "adaptation_speed"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    LATENCY = "latency"


class ComponentType(Enum):
    """System components being monitored"""
    ASL = "adaptive_scoring_layer"
    TRANSITION_ANALYZER = "transition_matrix_analyzer"
    BOUNDARY_OPTIMIZER = "dynamic_boundary_optimizer"
    TRANSITION_MANAGER = "intelligent_transition_manager"
    STABILITY_MONITOR = "regime_stability_monitor"
    NOISE_FILTER = "adaptive_noise_filter"
    INTEGRATED_SYSTEM = "integrated_system"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_type: PerformanceMetricType
    component: ComponentType
    value: float
    timestamp: datetime
    confidence: float
    context: Dict[str, Any]
    baseline_value: Optional[float] = None
    improvement: Optional[float] = None


@dataclass
class ComponentPerformance:
    """Performance summary for a component"""
    component: ComponentType
    current_metrics: Dict[PerformanceMetricType, float]
    historical_trends: Dict[PerformanceMetricType, List[float]]
    performance_score: float
    last_updated: datetime
    total_evaluations: int
    improvement_trend: float
    stability_score: float


@dataclass
class FeedbackAction:
    """Feedback action to improve performance"""
    component: ComponentType
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    priority: int
    timestamp: datetime
    applied: bool = False
    result: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    system_performance_score: float
    component_performances: Dict[ComponentType, ComponentPerformance]
    recent_improvements: List[FeedbackAction]
    performance_trends: Dict[str, Any]
    recommendations: List[str]
    report_timestamp: datetime
    evaluation_period: timedelta


class PerformanceFeedbackSystem:
    """
    Comprehensive performance monitoring and feedback system
    """
    
    def __init__(self, evaluation_window: int = 1000,
                 feedback_threshold: float = 0.05,
                 update_frequency: int = 50):
        """
        Initialize performance feedback system
        
        Args:
            evaluation_window: Window size for performance calculations
            feedback_threshold: Minimum improvement threshold for feedback
            update_frequency: How often to generate feedback (evaluations)
        """
        self.evaluation_window = evaluation_window
        self.feedback_threshold = feedback_threshold
        self.update_frequency = update_frequency
        
        # Performance tracking
        self.metrics_history: Dict[ComponentType, deque] = {
            component: deque(maxlen=evaluation_window) 
            for component in ComponentType
        }
        
        self.component_performances: Dict[ComponentType, ComponentPerformance] = {}
        
        # Baseline performance
        self.baseline_metrics: Dict[ComponentType, Dict[PerformanceMetricType, float]] = {}
        self.performance_baselines_set = False
        
        # Feedback system
        self.feedback_actions: deque = deque(maxlen=500)
        self.pending_actions: List[FeedbackAction] = []
        self.applied_actions: List[FeedbackAction] = []
        
        # Evaluation state
        self.total_evaluations = 0
        self.last_feedback_generation = datetime.now()
        self.system_start_time = datetime.now()
        
        # Performance improvement tracking
        self.improvement_history: Dict[ComponentType, deque] = {
            component: deque(maxlen=100) 
            for component in ComponentType
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds: Dict[ComponentType, Dict[str, float]] = {}
        self._initialize_adaptive_thresholds()
        
        logger.info("PerformanceFeedbackSystem initialized")
    
    def _initialize_adaptive_thresholds(self):
        """Initialize adaptive performance thresholds"""
        
        for component in ComponentType:
            self.adaptive_thresholds[component] = {
                'accuracy_threshold': 0.6,
                'stability_threshold': 0.7,
                'improvement_rate': 0.02,
                'degradation_threshold': -0.05
            }
    
    def record_performance_metric(self, metric: PerformanceMetric):
        """
        Record a performance metric for analysis
        
        Args:
            metric: Performance metric to record
        """
        # Store metric
        self.metrics_history[metric.component].append(metric)
        
        # Update component performance
        self._update_component_performance(metric.component)
        
        # Check if feedback is needed
        self.total_evaluations += 1
        if self.total_evaluations % self.update_frequency == 0:
            self._generate_feedback()
    
    def evaluate_component_performance(self, component: ComponentType,
                                     predictions: List[Any],
                                     actual_values: List[Any],
                                     context: Dict[str, Any] = None) -> Dict[PerformanceMetricType, float]:
        """
        Evaluate comprehensive performance for a component
        
        Args:
            component: Component being evaluated
            predictions: Predicted values
            actual_values: Actual values
            context: Additional context information
            
        Returns:
            Dictionary of performance metrics
        """
        context = context or {}
        timestamp = datetime.now()
        
        # Calculate basic accuracy metrics
        accuracy = accuracy_score(actual_values, predictions)
        
        # Calculate additional metrics for multi-class problems
        try:
            precision = precision_score(actual_values, predictions, average='weighted', zero_division=0)
            recall = recall_score(actual_values, predictions, average='weighted', zero_division=0)
            f1 = f1_score(actual_values, predictions, average='weighted', zero_division=0)
        except:
            precision = recall = f1 = accuracy
        
        # Calculate regime-specific metrics
        regime_stability = self._calculate_regime_stability(predictions, context)
        transition_quality = self._calculate_transition_quality(predictions, actual_values, context)
        
        # Calculate confidence metrics
        confidence_scores = context.get('confidence_scores', [0.5] * len(predictions))
        prediction_confidence = np.mean(confidence_scores)
        
        # Calculate latency if available
        latency = context.get('processing_time', 0.0)
        
        # Create metrics
        metrics = {
            PerformanceMetricType.ACCURACY: accuracy,
            PerformanceMetricType.PRECISION: precision,
            PerformanceMetricType.RECALL: recall,
            PerformanceMetricType.F1_SCORE: f1,
            PerformanceMetricType.REGIME_STABILITY: regime_stability,
            PerformanceMetricType.TRANSITION_QUALITY: transition_quality,
            PerformanceMetricType.PREDICTION_CONFIDENCE: prediction_confidence,
            PerformanceMetricType.LATENCY: latency
        }
        
        # Record each metric
        for metric_type, value in metrics.items():
            metric_obj = PerformanceMetric(
                metric_type=metric_type,
                component=component,
                value=value,
                timestamp=timestamp,
                confidence=prediction_confidence,
                context=context.copy()
            )
            self.record_performance_metric(metric_obj)
        
        return metrics
    
    def _calculate_regime_stability(self, predictions: List[Any], 
                                  context: Dict[str, Any]) -> float:
        """Calculate regime stability score"""
        
        if len(predictions) < 2:
            return 1.0
        
        # Count regime changes
        changes = sum(1 for i in range(1, len(predictions)) 
                     if predictions[i] != predictions[i-1])
        
        # Stability = 1 - (changes / possible_changes)
        max_changes = len(predictions) - 1
        stability = 1.0 - (changes / max_changes) if max_changes > 0 else 1.0
        
        return stability
    
    def _calculate_transition_quality(self, predictions: List[Any],
                                    actual_values: List[Any],
                                    context: Dict[str, Any]) -> float:
        """Calculate transition quality score"""
        
        if len(predictions) < 2:
            return 1.0
        
        # Find transition points
        pred_transitions = []
        actual_transitions = []
        
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                pred_transitions.append(i)
            if actual_values[i] != actual_values[i-1]:
                actual_transitions.append(i)
        
        if not actual_transitions:
            return 1.0
        
        # Calculate transition timing accuracy
        transition_scores = []
        
        for actual_trans in actual_transitions:
            # Find closest predicted transition
            if pred_transitions:
                closest_pred = min(pred_transitions, key=lambda x: abs(x - actual_trans))
                timing_error = abs(closest_pred - actual_trans)
                # Score based on timing accuracy (closer = better)
                timing_score = max(0, 1.0 - timing_error / 10.0)
                transition_scores.append(timing_score)
            else:
                transition_scores.append(0.0)
        
        return np.mean(transition_scores) if transition_scores else 0.0
    
    def _update_component_performance(self, component: ComponentType):
        """Update performance summary for a component"""
        
        metrics = list(self.metrics_history[component])
        if not metrics:
            return
        
        # Calculate current metrics
        current_metrics = {}
        historical_trends = defaultdict(list)
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric.value)
            historical_trends[metric.metric_type].append(metric.value)
        
        # Calculate current averages
        for metric_type, values in metrics_by_type.items():
            current_metrics[metric_type] = np.mean(values[-10:])  # Recent average
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(current_metrics)
        
        # Calculate improvement trend
        improvement_trend = self._calculate_improvement_trend(component, current_metrics)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(historical_trends)
        
        # Update component performance
        self.component_performances[component] = ComponentPerformance(
            component=component,
            current_metrics=current_metrics,
            historical_trends=dict(historical_trends),
            performance_score=performance_score,
            last_updated=datetime.now(),
            total_evaluations=len(metrics),
            improvement_trend=improvement_trend,
            stability_score=stability_score
        )
    
    def _calculate_performance_score(self, metrics: Dict[PerformanceMetricType, float]) -> float:
        """Calculate overall performance score for a component"""
        
        # Weight different metrics
        weights = {
            PerformanceMetricType.ACCURACY: 0.25,
            PerformanceMetricType.F1_SCORE: 0.20,
            PerformanceMetricType.REGIME_STABILITY: 0.20,
            PerformanceMetricType.TRANSITION_QUALITY: 0.15,
            PerformanceMetricType.PREDICTION_CONFIDENCE: 0.10,
            PerformanceMetricType.PRECISION: 0.05,
            PerformanceMetricType.RECALL: 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_type, weight in weights.items():
            if metric_type in metrics:
                weighted_score += metrics[metric_type] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_improvement_trend(self, component: ComponentType,
                                   current_metrics: Dict[PerformanceMetricType, float]) -> float:
        """Calculate improvement trend for a component"""
        
        # Compare current performance to baseline
        if component not in self.baseline_metrics:
            return 0.0
        
        baseline = self.baseline_metrics[component]
        improvements = []
        
        for metric_type, current_value in current_metrics.items():
            if metric_type in baseline:
                baseline_value = baseline[metric_type]
                if baseline_value > 0:
                    improvement = (current_value - baseline_value) / baseline_value
                    improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _calculate_stability_score(self, historical_trends: Dict[PerformanceMetricType, List[float]]) -> float:
        """Calculate stability score based on variance in performance"""
        
        stability_scores = []
        
        for metric_type, values in historical_trends.items():
            if len(values) >= 10:
                # Calculate coefficient of variation (lower = more stable)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val > 0:
                    cv = std_val / mean_val
                    stability = 1.0 / (1.0 + cv)  # Higher = more stable
                    stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _generate_feedback(self):
        """Generate feedback actions based on current performance"""
        
        self.last_feedback_generation = datetime.now()
        new_actions = []
        
        # Analyze each component
        for component, performance in self.component_performances.items():
            actions = self._analyze_component_for_feedback(component, performance)
            new_actions.extend(actions)
        
        # System-level feedback
        system_actions = self._generate_system_level_feedback()
        new_actions.extend(system_actions)
        
        # Prioritize and store actions
        new_actions.sort(key=lambda x: x.priority, reverse=True)
        self.pending_actions.extend(new_actions)
        
        # Store in history
        for action in new_actions:
            self.feedback_actions.append(action)
        
        logger.info(f"Generated {len(new_actions)} feedback actions")
    
    def _analyze_component_for_feedback(self, component: ComponentType,
                                      performance: ComponentPerformance) -> List[FeedbackAction]:
        """Analyze a component and generate feedback actions"""
        
        actions = []
        thresholds = self.adaptive_thresholds[component]
        
        # Check accuracy
        accuracy = performance.current_metrics.get(PerformanceMetricType.ACCURACY, 0.0)
        if accuracy < thresholds['accuracy_threshold']:
            action = FeedbackAction(
                component=component,
                action_type="improve_accuracy",
                parameters={
                    'current_accuracy': accuracy,
                    'target_accuracy': thresholds['accuracy_threshold'] + 0.05,
                    'suggested_changes': ['increase_learning_rate', 'adjust_feature_weights']
                },
                expected_improvement=0.1,
                priority=8,
                timestamp=datetime.now()
            )
            actions.append(action)
        
        # Check stability
        stability = performance.stability_score
        if stability < thresholds['stability_threshold']:
            action = FeedbackAction(
                component=component,
                action_type="improve_stability",
                parameters={
                    'current_stability': stability,
                    'target_stability': thresholds['stability_threshold'] + 0.05,
                    'suggested_changes': ['reduce_sensitivity', 'increase_smoothing']
                },
                expected_improvement=0.15,
                priority=7,
                timestamp=datetime.now()
            )
            actions.append(action)
        
        # Check for degradation
        if performance.improvement_trend < thresholds['degradation_threshold']:
            action = FeedbackAction(
                component=component,
                action_type="prevent_degradation",
                parameters={
                    'degradation_rate': performance.improvement_trend,
                    'suggested_changes': ['reset_parameters', 'increase_regularization']
                },
                expected_improvement=0.2,
                priority=9,
                timestamp=datetime.now()
            )
            actions.append(action)
        
        # Component-specific feedback
        component_actions = self._get_component_specific_feedback(component, performance)
        actions.extend(component_actions)
        
        return actions
    
    def _get_component_specific_feedback(self, component: ComponentType,
                                       performance: ComponentPerformance) -> List[FeedbackAction]:
        """Get component-specific feedback actions"""
        
        actions = []
        
        if component == ComponentType.ASL:
            # ASL-specific feedback
            if performance.current_metrics.get(PerformanceMetricType.PREDICTION_CONFIDENCE, 0) < 0.6:
                actions.append(FeedbackAction(
                    component=component,
                    action_type="adjust_asl_weights",
                    parameters={'target': 'increase_confidence'},
                    expected_improvement=0.1,
                    priority=6,
                    timestamp=datetime.now()
                ))
        
        elif component == ComponentType.TRANSITION_ANALYZER:
            # Transition analyzer feedback
            transition_quality = performance.current_metrics.get(PerformanceMetricType.TRANSITION_QUALITY, 0)
            if transition_quality < 0.7:
                actions.append(FeedbackAction(
                    component=component,
                    action_type="improve_transition_detection",
                    parameters={'current_quality': transition_quality},
                    expected_improvement=0.15,
                    priority=7,
                    timestamp=datetime.now()
                ))
        
        elif component == ComponentType.BOUNDARY_OPTIMIZER:
            # Boundary optimizer feedback
            if performance.current_metrics.get(PerformanceMetricType.REGIME_STABILITY, 0) < 0.6:
                actions.append(FeedbackAction(
                    component=component,
                    action_type="optimize_boundaries",
                    parameters={'target': 'regime_stability'},
                    expected_improvement=0.2,
                    priority=8,
                    timestamp=datetime.now()
                ))
        
        return actions
    
    def _generate_system_level_feedback(self) -> List[FeedbackAction]:
        """Generate system-level feedback actions"""
        
        actions = []
        
        # Calculate system performance
        system_score = self.get_system_performance_score()
        
        if system_score < 0.6:
            actions.append(FeedbackAction(
                component=ComponentType.INTEGRATED_SYSTEM,
                action_type="system_recalibration",
                parameters={
                    'current_score': system_score,
                    'target_score': 0.7,
                    'suggested_changes': ['full_retraining', 'parameter_reset']
                },
                expected_improvement=0.3,
                priority=10,
                timestamp=datetime.now()
            ))
        
        # Check for component imbalance
        component_scores = {
            comp: perf.performance_score 
            for comp, perf in self.component_performances.items()
        }
        
        if component_scores:
            score_variance = np.var(list(component_scores.values()))
            if score_variance > 0.1:  # High variance indicates imbalance
                actions.append(FeedbackAction(
                    component=ComponentType.INTEGRATED_SYSTEM,
                    action_type="balance_components",
                    parameters={
                        'component_scores': component_scores,
                        'variance': score_variance
                    },
                    expected_improvement=0.1,
                    priority=5,
                    timestamp=datetime.now()
                ))
        
        return actions
    
    def get_system_performance_score(self) -> float:
        """Calculate overall system performance score"""
        
        if not self.component_performances:
            return 0.0
        
        scores = [perf.performance_score for perf in self.component_performances.values()]
        return np.mean(scores)
    
    def get_pending_feedback_actions(self, max_actions: int = 10) -> List[FeedbackAction]:
        """Get pending feedback actions sorted by priority"""
        
        return sorted(self.pending_actions, key=lambda x: x.priority, reverse=True)[:max_actions]
    
    def apply_feedback_action(self, action: FeedbackAction, result: Dict[str, Any]):
        """
        Mark a feedback action as applied and record results
        
        Args:
            action: The feedback action that was applied
            result: Results of applying the action
        """
        action.applied = True
        action.result = result
        
        # Move from pending to applied
        if action in self.pending_actions:
            self.pending_actions.remove(action)
        
        self.applied_actions.append(action)
        
        logger.info(f"Applied feedback action: {action.action_type} for {action.component.value}")
    
    def set_performance_baselines(self):
        """Set current performance as baselines for future comparison"""
        
        for component, performance in self.component_performances.items():
            self.baseline_metrics[component] = performance.current_metrics.copy()
        
        self.performance_baselines_set = True
        logger.info("Performance baselines set for all components")
    
    def generate_performance_report(self, period_hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        # Calculate report period
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)
        
        # Get recent improvements
        recent_improvements = [
            action for action in self.applied_actions
            if action.timestamp >= start_time
        ]
        
        # Calculate trends
        performance_trends = self._calculate_performance_trends(start_time, end_time)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return PerformanceReport(
            system_performance_score=self.get_system_performance_score(),
            component_performances=self.component_performances.copy(),
            recent_improvements=recent_improvements,
            performance_trends=performance_trends,
            recommendations=recommendations,
            report_timestamp=end_time,
            evaluation_period=timedelta(hours=period_hours)
        )
    
    def _calculate_performance_trends(self, start_time: datetime, 
                                    end_time: datetime) -> Dict[str, Any]:
        """Calculate performance trends over a period"""
        
        trends = {}
        
        for component, metrics in self.metrics_history.items():
            component_trends = {}
            
            # Filter metrics by time period
            period_metrics = [
                m for m in metrics 
                if start_time <= m.timestamp <= end_time
            ]
            
            if len(period_metrics) >= 2:
                # Group by metric type
                metrics_by_type = defaultdict(list)
                for metric in period_metrics:
                    metrics_by_type[metric.metric_type].append(metric.value)
                
                # Calculate trends
                for metric_type, values in metrics_by_type.items():
                    if len(values) >= 3:
                        # Linear regression for trend
                        x = np.arange(len(values))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                        
                        component_trends[metric_type.value] = {
                            'slope': slope,
                            'r_squared': r_value ** 2,
                            'trend_direction': 'improving' if slope > 0 else 'declining',
                            'significance': p_value < 0.05
                        }
            
            trends[component.value] = component_trends
        
        return trends
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current performance"""
        
        recommendations = []
        
        # Analyze pending actions
        high_priority_actions = [a for a in self.pending_actions if a.priority >= 8]
        if high_priority_actions:
            recommendations.append(
                f"Address {len(high_priority_actions)} high-priority performance issues"
            )
        
        # System performance
        system_score = self.get_system_performance_score()
        if system_score < 0.6:
            recommendations.append("System performance below threshold - consider full recalibration")
        elif system_score > 0.8:
            recommendations.append("System performing well - consider optimization for efficiency")
        
        # Component balance
        if self.component_performances:
            scores = [p.performance_score for p in self.component_performances.values()]
            if max(scores) - min(scores) > 0.3:
                recommendations.append("Large performance gap between components - balance needed")
        
        # Stability issues
        unstable_components = [
            comp.value for comp, perf in self.component_performances.items()
            if perf.stability_score < 0.6
        ]
        if unstable_components:
            recommendations.append(f"Stability issues in: {', '.join(unstable_components)}")
        
        return recommendations
    
    def export_performance_data(self, filepath: str):
        """Export comprehensive performance data"""
        
        # Prepare export data
        export_data = {
            'system_overview': {
                'total_evaluations': self.total_evaluations,
                'system_performance_score': self.get_system_performance_score(),
                'system_uptime_hours': (datetime.now() - self.system_start_time).total_seconds() / 3600,
                'baselines_set': self.performance_baselines_set
            },
            'component_performances': {
                comp.value: {
                    'performance_score': perf.performance_score,
                    'improvement_trend': perf.improvement_trend,
                    'stability_score': perf.stability_score,
                    'total_evaluations': perf.total_evaluations,
                    'current_metrics': {mt.value: val for mt, val in perf.current_metrics.items()}
                }
                for comp, perf in self.component_performances.items()
            },
            'feedback_summary': {
                'total_actions_generated': len(self.feedback_actions),
                'pending_actions': len(self.pending_actions),
                'applied_actions': len(self.applied_actions),
                'recent_actions': [
                    {
                        'component': a.component.value,
                        'action_type': a.action_type,
                        'priority': a.priority,
                        'applied': a.applied,
                        'timestamp': a.timestamp.isoformat()
                    }
                    for a in list(self.feedback_actions)[-20:]
                ]
            },
            'adaptive_thresholds': {
                comp.value: thresholds 
                for comp, thresholds in self.adaptive_thresholds.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance data exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create performance feedback system
    feedback_system = PerformanceFeedbackSystem(
        evaluation_window=500,
        feedback_threshold=0.05,
        update_frequency=25
    )
    
    # Simulate performance evaluation
    for i in range(100):
        # Simulate predictions and actual values
        predictions = np.random.randint(0, 8, 20)
        actual_values = predictions.copy()
        
        # Add some noise to simulate real performance
        noise_indices = np.random.choice(len(predictions), size=int(len(predictions) * 0.2), replace=False)
        for idx in noise_indices:
            actual_values[idx] = np.random.randint(0, 8)
        
        # Simulate different components
        component = list(ComponentType)[i % len(ComponentType)]
        
        # Add context
        context = {
            'confidence_scores': np.random.beta(2, 2, len(predictions)),
            'processing_time': np.random.exponential(0.01),
            'market_conditions': 'normal'
        }
        
        # Evaluate performance
        metrics = feedback_system.evaluate_component_performance(
            component=component,
            predictions=predictions.tolist(),
            actual_values=actual_values.tolist(),
            context=context
        )
    
    # Set baselines after initial period
    if not feedback_system.performance_baselines_set:
        feedback_system.set_performance_baselines()
    
    # Get performance report
    report = feedback_system.generate_performance_report(period_hours=1)
    
    print("Performance Feedback System Report:")
    print(f"System Performance Score: {report.system_performance_score:.3f}")
    print(f"Components Monitored: {len(report.component_performances)}")
    print(f"Recent Improvements: {len(report.recent_improvements)}")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")
    
    # Get pending actions
    pending_actions = feedback_system.get_pending_feedback_actions(max_actions=5)
    print(f"\nPending Feedback Actions: {len(pending_actions)}")
    for action in pending_actions[:3]:
        print(f"- {action.component.value}: {action.action_type} (priority: {action.priority})")