"""
Regime Stability Monitor

This module monitors regime stability, detects anomalies, and tracks
persistence metrics for intelligent regime management.

Key Features:
- Real-time stability scoring
- Anomaly detection in regime behavior
- Persistence tracking and analysis
- Stability trend monitoring
- Early warning system for regime degradation
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class StabilityLevel(Enum):
    """Regime stability levels"""
    VERY_STABLE = "very_stable"
    STABLE = "stable"
    MODERATE = "moderate"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of detected anomalies"""
    VOLATILITY_SPIKE = "volatility_spike"
    UNEXPECTED_TRANSITION = "unexpected_transition"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SIGNAL_INCONSISTENCY = "signal_inconsistency"
    EXTERNAL_SHOCK = "external_shock"


@dataclass
class StabilityMetrics:
    """Comprehensive stability metrics for a regime"""
    regime_id: int
    current_score: float
    trend_score: float
    persistence_score: float
    consistency_score: float
    predictability_score: float
    overall_stability: StabilityLevel
    confidence_interval: Tuple[float, float]
    last_updated: datetime
    sample_count: int


@dataclass
class AnomalyEvent:
    """Detected anomaly event"""
    anomaly_type: AnomalyType
    severity: float
    timestamp: datetime
    regime_id: int
    description: str
    affected_metrics: List[str]
    confidence: float
    context_data: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class StabilityAlert:
    """Stability alert for regime degradation"""
    regime_id: int
    alert_level: str
    message: str
    timestamp: datetime
    metrics_snapshot: Dict[str, float]
    recommended_actions: List[str]


class RegimeStabilityMonitor:
    """
    Monitors and analyzes regime stability in real-time
    """
    
    def __init__(self, regime_count: int = 12,
                 stability_window: int = 100,
                 anomaly_sensitivity: float = 0.05):
        """
        Initialize regime stability monitor
        
        Args:
            regime_count: Number of regimes to monitor
            stability_window: Window size for stability calculations
            anomaly_sensitivity: Sensitivity threshold for anomaly detection
        """
        self.regime_count = regime_count
        self.stability_window = stability_window
        self.anomaly_sensitivity = anomaly_sensitivity
        
        # Stability tracking for each regime
        self.regime_metrics: Dict[int, StabilityMetrics] = {}
        self.regime_histories: Dict[int, deque] = {
            i: deque(maxlen=stability_window) for i in range(regime_count)
        }
        
        # Performance tracking
        self.prediction_accuracy: Dict[int, deque] = {
            i: deque(maxlen=stability_window) for i in range(regime_count)
        }
        self.transition_success: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=stability_window)
        )
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=anomaly_sensitivity,
            random_state=42
        )
        self.anomaly_history = deque(maxlen=1000)
        self.active_anomalies: List[AnomalyEvent] = []
        
        # Alert system
        self.stability_alerts: List[StabilityAlert] = []
        self.alert_thresholds = {
            'critical': 0.3,
            'warning': 0.5,
            'info': 0.7
        }
        
        # Monitoring state
        self.current_regime = 0
        self.monitoring_start_time = datetime.now()
        self.last_update_time = datetime.now()
        
        # Feature scaling for anomaly detection
        self.scaler = StandardScaler()
        self.feature_buffer = deque(maxlen=1000)
        
        logger.info(f"RegimeStabilityMonitor initialized for {regime_count} regimes")
    
    def update_regime_data(self, regime_id: int, 
                          regime_scores: Dict[int, float],
                          market_data: Dict[str, Any],
                          prediction_accuracy: Optional[float] = None):
        """
        Update regime data and calculate stability metrics
        
        Args:
            regime_id: Current active regime
            regime_scores: All regime scores
            market_data: Current market data
            prediction_accuracy: Optional accuracy score
        """
        self.current_regime = regime_id
        self.last_update_time = datetime.now()
        
        # Store regime data
        regime_data = {
            'timestamp': self.last_update_time,
            'regime_scores': regime_scores.copy(),
            'market_data': market_data.copy(),
            'active_regime': regime_id
        }
        
        self.regime_histories[regime_id].append(regime_data)
        
        # Update prediction accuracy if provided
        if prediction_accuracy is not None:
            self.prediction_accuracy[regime_id].append(prediction_accuracy)
        
        # Calculate stability metrics for current regime
        self._calculate_stability_metrics(regime_id)
        
        # Update feature buffer for anomaly detection
        self._update_feature_buffer(regime_scores, market_data)
        
        # Check for anomalies
        self._detect_anomalies(regime_id, regime_scores, market_data)
        
        # Generate alerts if needed
        self._check_stability_alerts(regime_id)
    
    def _calculate_stability_metrics(self, regime_id: int):
        """Calculate comprehensive stability metrics for a regime"""
        
        regime_history = list(self.regime_histories[regime_id])
        if len(regime_history) < 10:
            return  # Insufficient data
        
        # Extract time series data
        timestamps = [data['timestamp'] for data in regime_history]
        regime_scores = [data['regime_scores'].get(regime_id, 0.0) for data in regime_history]
        
        # 1. Current Score (recent average)
        current_score = np.mean(regime_scores[-10:])
        
        # 2. Trend Score (stability of trend)
        trend_score = self._calculate_trend_stability(regime_scores)
        
        # 3. Persistence Score (how long regime stays active)
        persistence_score = self._calculate_persistence_score(regime_id)
        
        # 4. Consistency Score (variance in regime scores)
        consistency_score = self._calculate_consistency_score(regime_scores)
        
        # 5. Predictability Score (based on prediction accuracy)
        predictability_score = self._calculate_predictability_score(regime_id)
        
        # Overall stability assessment
        overall_score = (
            current_score * 0.25 +
            trend_score * 0.20 +
            persistence_score * 0.20 +
            consistency_score * 0.20 +
            predictability_score * 0.15
        )
        
        # Determine stability level
        stability_level = self._determine_stability_level(overall_score)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(regime_scores)
        
        # Update metrics
        self.regime_metrics[regime_id] = StabilityMetrics(
            regime_id=regime_id,
            current_score=current_score,
            trend_score=trend_score,
            persistence_score=persistence_score,
            consistency_score=consistency_score,
            predictability_score=predictability_score,
            overall_stability=stability_level,
            confidence_interval=confidence_interval,
            last_updated=self.last_update_time,
            sample_count=len(regime_history)
        )
    
    def _calculate_trend_stability(self, regime_scores: List[float]) -> float:
        """Calculate trend stability score"""
        
        if len(regime_scores) < 5:
            return 0.5
        
        # Calculate rolling trends
        window_size = min(5, len(regime_scores) // 2)
        trends = []
        
        for i in range(window_size, len(regime_scores)):
            window = regime_scores[i-window_size:i]
            if len(window) >= 3:
                # Linear regression slope
                x = np.arange(len(window))
                slope, _, r_value, _, _ = stats.linregress(x, window)
                trends.append(abs(slope))
        
        if not trends:
            return 0.5
        
        # Stability = low variance in trends
        trend_variance = np.var(trends)
        trend_stability = 1.0 / (1.0 + trend_variance * 10)
        
        return np.clip(trend_stability, 0.0, 1.0)
    
    def _calculate_persistence_score(self, regime_id: int) -> float:
        """Calculate regime persistence score"""
        
        # Count consecutive occurrences of this regime
        all_regimes = []
        for regime_hist in self.regime_histories.values():
            for data in regime_hist:
                all_regimes.append(data['active_regime'])
        
        if len(all_regimes) < 10:
            return 0.5
        
        # Find persistence periods for this regime
        persistence_periods = []
        current_period = 0
        
        for regime in all_regimes:
            if regime == regime_id:
                current_period += 1
            else:
                if current_period > 0:
                    persistence_periods.append(current_period)
                current_period = 0
        
        # Add final period if ongoing
        if current_period > 0:
            persistence_periods.append(current_period)
        
        if not persistence_periods:
            return 0.0
        
        # Score based on average persistence
        avg_persistence = np.mean(persistence_periods)
        max_possible = len(all_regimes) / self.regime_count
        
        persistence_score = min(avg_persistence / max_possible, 1.0)
        
        return persistence_score
    
    def _calculate_consistency_score(self, regime_scores: List[float]) -> float:
        """Calculate consistency score based on score variance"""
        
        if len(regime_scores) < 3:
            return 0.5
        
        # Calculate coefficient of variation
        mean_score = np.mean(regime_scores)
        std_score = np.std(regime_scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / mean_score
        
        # Lower coefficient of variation = higher consistency
        consistency_score = 1.0 / (1.0 + cv)
        
        return np.clip(consistency_score, 0.0, 1.0)
    
    def _calculate_predictability_score(self, regime_id: int) -> float:
        """Calculate predictability score based on prediction accuracy"""
        
        accuracy_history = list(self.prediction_accuracy[regime_id])
        
        if len(accuracy_history) < 5:
            return 0.5
        
        # Recent accuracy
        recent_accuracy = np.mean(accuracy_history[-10:])
        
        # Accuracy trend
        if len(accuracy_history) >= 10:
            early_accuracy = np.mean(accuracy_history[:5])
            late_accuracy = np.mean(accuracy_history[-5:])
            
            # Bonus for improving accuracy
            improvement_bonus = max(0, late_accuracy - early_accuracy) * 0.5
            predictability_score = recent_accuracy + improvement_bonus
        else:
            predictability_score = recent_accuracy
        
        return np.clip(predictability_score, 0.0, 1.0)
    
    def _determine_stability_level(self, overall_score: float) -> StabilityLevel:
        """Determine stability level from overall score"""
        
        if overall_score >= 0.8:
            return StabilityLevel.VERY_STABLE
        elif overall_score >= 0.65:
            return StabilityLevel.STABLE
        elif overall_score >= 0.5:
            return StabilityLevel.MODERATE
        elif overall_score >= 0.3:
            return StabilityLevel.UNSTABLE
        else:
            return StabilityLevel.CRITICAL
    
    def _calculate_confidence_interval(self, regime_scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for regime scores"""
        
        if len(regime_scores) < 5:
            mean_score = np.mean(regime_scores) if regime_scores else 0.5
            return (mean_score - 0.1, mean_score + 0.1)
        
        mean_score = np.mean(regime_scores)
        std_score = np.std(regime_scores)
        n = len(regime_scores)
        
        # 95% confidence interval
        margin = 1.96 * std_score / np.sqrt(n)
        
        return (
            max(0.0, mean_score - margin),
            min(1.0, mean_score + margin)
        )
    
    def _update_feature_buffer(self, regime_scores: Dict[int, float],
                             market_data: Dict[str, Any]):
        """Update feature buffer for anomaly detection"""
        
        # Extract relevant features
        features = []
        
        # Regime score features
        features.extend(list(regime_scores.values()))
        
        # Market data features
        market_features = [
            'volatility', 'trend', 'volume_ratio', 'spot_price',
            'total_delta', 'total_gamma', 'total_vega'
        ]
        
        for feature in market_features:
            value = market_data.get(feature, 0.0)
            if isinstance(value, (int, float)) and not np.isnan(value):
                features.append(float(value))
            else:
                features.append(0.0)
        
        self.feature_buffer.append(features)
        
        # Retrain anomaly detector periodically
        if len(self.feature_buffer) >= 100 and len(self.feature_buffer) % 50 == 0:
            self._retrain_anomaly_detector()
    
    def _retrain_anomaly_detector(self):
        """Retrain anomaly detector with recent data"""
        
        try:
            # Get feature matrix
            feature_matrix = np.array(list(self.feature_buffer))
            
            # Scale features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Retrain detector
            self.anomaly_detector.fit(feature_matrix_scaled)
            
            logger.debug("Anomaly detector retrained with new data")
            
        except Exception as e:
            logger.warning(f"Failed to retrain anomaly detector: {e}")
    
    def _detect_anomalies(self, regime_id: int,
                         regime_scores: Dict[int, float],
                         market_data: Dict[str, Any]):
        """Detect anomalies in current regime behavior"""
        
        if len(self.feature_buffer) < 50:
            return  # Insufficient data for anomaly detection
        
        # Prepare current features
        current_features = []
        current_features.extend(list(regime_scores.values()))
        
        market_features = [
            'volatility', 'trend', 'volume_ratio', 'spot_price',
            'total_delta', 'total_gamma', 'total_vega'
        ]
        
        for feature in market_features:
            value = market_data.get(feature, 0.0)
            if isinstance(value, (int, float)) and not np.isnan(value):
                current_features.append(float(value))
            else:
                current_features.append(0.0)
        
        try:
            # Scale current features
            current_features_scaled = self.scaler.transform([current_features])
            
            # Detect anomaly
            anomaly_score = self.anomaly_detector.decision_function(current_features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(current_features_scaled)[0] == -1
            
            if is_anomaly:
                # Classify anomaly type
                anomaly_type = self._classify_anomaly_type(regime_scores, market_data)
                
                # Create anomaly event
                anomaly = AnomalyEvent(
                    anomaly_type=anomaly_type,
                    severity=abs(anomaly_score),
                    timestamp=self.last_update_time,
                    regime_id=regime_id,
                    description=self._generate_anomaly_description(anomaly_type, market_data),
                    affected_metrics=self._identify_affected_metrics(anomaly_type),
                    confidence=min(abs(anomaly_score) * 2, 1.0),
                    context_data={
                        'regime_scores': regime_scores.copy(),
                        'market_data': market_data.copy(),
                        'anomaly_score': anomaly_score
                    }
                )
                
                self.anomaly_history.append(anomaly)
                self.active_anomalies.append(anomaly)
                
                logger.warning(f"Anomaly detected: {anomaly_type.value} "
                             f"(severity: {anomaly.severity:.3f})")
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    def _classify_anomaly_type(self, regime_scores: Dict[int, float],
                             market_data: Dict[str, Any]) -> AnomalyType:
        """Classify the type of detected anomaly"""
        
        # Check for volatility spike
        volatility = market_data.get('volatility', 0.0)
        if volatility > 0.4:
            return AnomalyType.VOLATILITY_SPIKE
        
        # Check for unexpected transitions
        current_score = regime_scores.get(self.current_regime, 0.0)
        if current_score < 0.3:  # Current regime has low score
            return AnomalyType.UNEXPECTED_TRANSITION
        
        # Check for signal inconsistency
        score_variance = np.var(list(regime_scores.values()))
        if score_variance < 0.01:  # All scores very similar
            return AnomalyType.SIGNAL_INCONSISTENCY
        
        # Check for performance degradation
        if self.current_regime in self.prediction_accuracy:
            recent_accuracy = list(self.prediction_accuracy[self.current_regime])[-5:]
            if recent_accuracy and np.mean(recent_accuracy) < 0.4:
                return AnomalyType.PERFORMANCE_DEGRADATION
        
        # Default to external shock
        return AnomalyType.EXTERNAL_SHOCK
    
    def _generate_anomaly_description(self, anomaly_type: AnomalyType,
                                    market_data: Dict[str, Any]) -> str:
        """Generate human-readable anomaly description"""
        
        descriptions = {
            AnomalyType.VOLATILITY_SPIKE: f"Volatility spike detected: {market_data.get('volatility', 0):.3f}",
            AnomalyType.UNEXPECTED_TRANSITION: "Unexpected regime transition with low confidence",
            AnomalyType.PERFORMANCE_DEGRADATION: "Regime prediction performance has degraded",
            AnomalyType.SIGNAL_INCONSISTENCY: "Inconsistent signals across regime indicators",
            AnomalyType.EXTERNAL_SHOCK: "External market shock detected"
        }
        
        return descriptions.get(anomaly_type, "Unknown anomaly detected")
    
    def _identify_affected_metrics(self, anomaly_type: AnomalyType) -> List[str]:
        """Identify which metrics are affected by the anomaly"""
        
        affected_metrics = {
            AnomalyType.VOLATILITY_SPIKE: ['trend_score', 'consistency_score'],
            AnomalyType.UNEXPECTED_TRANSITION: ['persistence_score', 'current_score'],
            AnomalyType.PERFORMANCE_DEGRADATION: ['predictability_score'],
            AnomalyType.SIGNAL_INCONSISTENCY: ['consistency_score', 'current_score'],
            AnomalyType.EXTERNAL_SHOCK: ['trend_score', 'consistency_score', 'current_score']
        }
        
        return affected_metrics.get(anomaly_type, ['overall_stability'])
    
    def _check_stability_alerts(self, regime_id: int):
        """Check if stability alerts should be generated"""
        
        if regime_id not in self.regime_metrics:
            return
        
        metrics = self.regime_metrics[regime_id]
        stability_score = metrics.current_score
        
        # Generate alerts based on thresholds
        alert_level = None
        if stability_score < self.alert_thresholds['critical']:
            alert_level = 'critical'
        elif stability_score < self.alert_thresholds['warning']:
            alert_level = 'warning'
        elif stability_score < self.alert_thresholds['info']:
            alert_level = 'info'
        
        if alert_level:
            # Check if we already have a recent alert for this regime
            recent_alerts = [
                alert for alert in self.stability_alerts[-10:]
                if alert.regime_id == regime_id and
                (self.last_update_time - alert.timestamp).total_seconds() < 300
            ]
            
            if not recent_alerts:
                alert = StabilityAlert(
                    regime_id=regime_id,
                    alert_level=alert_level,
                    message=self._generate_alert_message(regime_id, alert_level, metrics),
                    timestamp=self.last_update_time,
                    metrics_snapshot={
                        'current_score': metrics.current_score,
                        'trend_score': metrics.trend_score,
                        'persistence_score': metrics.persistence_score,
                        'consistency_score': metrics.consistency_score,
                        'predictability_score': metrics.predictability_score
                    },
                    recommended_actions=self._generate_recommendations(alert_level, metrics)
                )
                
                self.stability_alerts.append(alert)
                logger.warning(f"Stability alert ({alert_level}): {alert.message}")
    
    def _generate_alert_message(self, regime_id: int, alert_level: str,
                              metrics: StabilityMetrics) -> str:
        """Generate alert message"""
        
        stability_desc = metrics.overall_stability.value.replace('_', ' ').title()
        
        message = (f"Regime {regime_id} stability is {alert_level.upper()}: "
                  f"{stability_desc} (score: {metrics.current_score:.3f})")
        
        # Add specific concerns
        concerns = []
        if metrics.trend_score < 0.4:
            concerns.append("unstable trend")
        if metrics.persistence_score < 0.3:
            concerns.append("low persistence")
        if metrics.consistency_score < 0.5:
            concerns.append("inconsistent signals")
        if metrics.predictability_score < 0.4:
            concerns.append("poor predictability")
        
        if concerns:
            message += f". Issues: {', '.join(concerns)}"
        
        return message
    
    def _generate_recommendations(self, alert_level: str,
                                metrics: StabilityMetrics) -> List[str]:
        """Generate recommended actions for stability issues"""
        
        recommendations = []
        
        if alert_level == 'critical':
            recommendations.append("Consider immediate regime recalibration")
            recommendations.append("Increase transition sensitivity")
            recommendations.append("Review market conditions for external factors")
        
        if metrics.trend_score < 0.4:
            recommendations.append("Check for trend-based filtering")
        
        if metrics.persistence_score < 0.3:
            recommendations.append("Increase minimum regime duration")
        
        if metrics.consistency_score < 0.5:
            recommendations.append("Review component weight balance")
            recommendations.append("Check for data quality issues")
        
        if metrics.predictability_score < 0.4:
            recommendations.append("Retrain prediction models")
            recommendations.append("Validate feature engineering")
        
        return recommendations
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get comprehensive stability report"""
        
        # Overall system stability
        all_scores = [m.current_score for m in self.regime_metrics.values()]
        system_stability = np.mean(all_scores) if all_scores else 0.0
        
        # Regime stability distribution
        stability_distribution = defaultdict(int)
        for metrics in self.regime_metrics.values():
            stability_distribution[metrics.overall_stability.value] += 1
        
        # Recent anomalies
        recent_anomalies = [
            a for a in self.anomaly_history
            if (datetime.now() - a.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Active alerts
        active_alerts = [
            a for a in self.stability_alerts
            if (datetime.now() - a.timestamp).total_seconds() < 1800  # Last 30 minutes
        ]
        
        return {
            'system_stability_score': system_stability,
            'monitoring_duration': (datetime.now() - self.monitoring_start_time).total_seconds(),
            'regime_metrics': {
                regime_id: {
                    'current_score': metrics.current_score,
                    'stability_level': metrics.overall_stability.value,
                    'trend_score': metrics.trend_score,
                    'persistence_score': metrics.persistence_score,
                    'consistency_score': metrics.consistency_score,
                    'predictability_score': metrics.predictability_score,
                    'confidence_interval': metrics.confidence_interval,
                    'sample_count': metrics.sample_count
                }
                for regime_id, metrics in self.regime_metrics.items()
            },
            'stability_distribution': dict(stability_distribution),
            'anomaly_summary': {
                'total_anomalies': len(self.anomaly_history),
                'recent_anomalies': len(recent_anomalies),
                'active_anomalies': len(self.active_anomalies),
                'anomaly_types': dict(defaultdict(int, {
                    a.anomaly_type.value: 1 for a in recent_anomalies
                }))
            },
            'alert_summary': {
                'total_alerts': len(self.stability_alerts),
                'active_alerts': len(active_alerts),
                'alert_levels': dict(defaultdict(int, {
                    a.alert_level: 1 for a in active_alerts
                }))
            }
        }
    
    def resolve_anomaly(self, anomaly_id: int):
        """Mark an anomaly as resolved"""
        
        if 0 <= anomaly_id < len(self.active_anomalies):
            anomaly = self.active_anomalies[anomaly_id]
            anomaly.resolved = True
            anomaly.resolution_time = datetime.now()
            
            # Remove from active list
            self.active_anomalies.pop(anomaly_id)
            
            logger.info(f"Anomaly resolved: {anomaly.anomaly_type.value}")
    
    def export_stability_report(self, filepath: str):
        """Export comprehensive stability report"""
        
        report = self.get_stability_report()
        
        # Add detailed anomaly information
        report['detailed_anomalies'] = [
            {
                'type': a.anomaly_type.value,
                'severity': a.severity,
                'timestamp': a.timestamp.isoformat(),
                'regime_id': a.regime_id,
                'description': a.description,
                'confidence': a.confidence,
                'resolved': a.resolved
            }
            for a in list(self.anomaly_history)[-50:]  # Last 50 anomalies
        ]
        
        # Add alert details
        report['detailed_alerts'] = [
            {
                'regime_id': a.regime_id,
                'level': a.alert_level,
                'message': a.message,
                'timestamp': a.timestamp.isoformat(),
                'metrics': a.metrics_snapshot,
                'recommendations': a.recommended_actions
            }
            for a in self.stability_alerts[-20:]  # Last 20 alerts
        ]
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Stability report exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create stability monitor
    monitor = RegimeStabilityMonitor(
        regime_count=8,
        stability_window=100,
        anomaly_sensitivity=0.05
    )
    
    # Simulate monitoring data
    for i in range(200):
        # Generate synthetic regime scores
        regime_scores = {j: np.random.beta(2, 2) for j in range(8)}
        regime_scores[i % 8] += 0.3  # Boost current regime
        
        # Normalize scores
        total = sum(regime_scores.values())
        regime_scores = {k: v/total for k, v in regime_scores.items()}
        
        # Generate market data
        market_data = {
            'volatility': 0.1 + 0.2 * np.random.random(),
            'trend': 0.02 * np.random.randn(),
            'volume_ratio': 0.8 + 0.4 * np.random.random(),
            'spot_price': 25000 + 1000 * np.random.randn(),
            'total_delta': 1000 * np.random.randn(),
            'total_gamma': 500 * np.random.randn(),
            'total_vega': 1000 * np.random.randn()
        }
        
        # Add some anomalies
        if i % 50 == 0:
            market_data['volatility'] = 0.6  # Volatility spike
        
        # Update monitor
        current_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
        accuracy = 0.6 + 0.3 * np.random.random()
        
        monitor.update_regime_data(
            regime_id=current_regime,
            regime_scores=regime_scores,
            market_data=market_data,
            prediction_accuracy=accuracy
        )
    
    # Get stability report
    report = monitor.get_stability_report()
    
    print("Stability Report:")
    print(f"System stability score: {report['system_stability_score']:.3f}")
    print(f"Total anomalies detected: {report['anomaly_summary']['total_anomalies']}")
    print(f"Recent anomalies: {report['anomaly_summary']['recent_anomalies']}")
    print(f"Active alerts: {report['alert_summary']['active_alerts']}")
    
    print("\nRegime Stability:")
    for regime_id, metrics in report['regime_metrics'].items():
        print(f"Regime {regime_id}: {metrics['stability_level']} "
              f"(score: {metrics['current_score']:.3f})")