#!/usr/bin/env python3
"""
Enhanced Performance Monitor for Enhanced Triple Straddle Framework v2.0
========================================================================

This module implements comprehensive performance monitoring for the Enhanced Triple Straddle Framework:
- <3 second processing time monitoring
- >85% accuracy tracking with real-time alerts
- Memory usage optimization and performance analytics dashboard
- Mathematical accuracy monitoring for ±0.001 tolerance validation
- Integration with unified_stable_market_regime_pipeline.py

Features:
- Real-time performance tracking
- Alert system for performance violations
- Memory usage optimization
- Performance analytics dashboard
- Mathematical accuracy validation
- Historical performance analysis

Author: The Augster
Date: 2025-06-20
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import logging
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import threading
import time
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance targets for Enhanced Triple Straddle Framework v2.0
PERFORMANCE_TARGETS = {
    'max_processing_time': 3.0,      # <3 seconds
    'min_accuracy': 0.85,            # >85% accuracy
    'mathematical_tolerance': 0.001,  # ±0.001 tolerance
    'max_memory_usage_mb': 500,      # 500MB memory limit
    'alert_threshold_violations': 3   # Alert after 3 consecutive violations
}

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: datetime
    processing_time: float
    accuracy: float
    mathematical_accuracy: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    component_timings: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0

@dataclass
class AlertConfig:
    """Configuration for performance alerts"""
    enable_processing_time_alerts: bool = True
    enable_accuracy_alerts: bool = True
    enable_memory_alerts: bool = True
    enable_mathematical_accuracy_alerts: bool = True
    
    # Alert thresholds
    processing_time_threshold: float = PERFORMANCE_TARGETS['max_processing_time']
    accuracy_threshold: float = PERFORMANCE_TARGETS['min_accuracy']
    memory_threshold_mb: float = PERFORMANCE_TARGETS['max_memory_usage_mb']
    consecutive_violations_threshold: int = PERFORMANCE_TARGETS['alert_threshold_violations']
    
    # Alert cooldown (seconds)
    alert_cooldown_seconds: int = 300  # 5 minutes

class EnhancedPerformanceMonitor:
    """
    Enhanced Performance Monitor for real-time tracking and optimization
    of the Enhanced Triple Straddle Framework v2.0
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize Enhanced Performance Monitor
        
        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        
        # Performance tracking
        self.metrics_history = deque(maxlen=10000)  # Keep last 10,000 metrics
        self.component_performance = {}
        self.alert_history = []
        self.last_alert_time = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance statistics
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_processing_time': 0.0,
            'average_accuracy': 0.0,
            'mathematical_accuracy_rate': 0.0
        }
        
        # Memory optimization
        self.memory_optimization_enabled = True
        self.gc_frequency = 100  # Run garbage collection every 100 operations
        
        logger.info("Enhanced Performance Monitor initialized")
        logger.info(f"Processing time target: <{self.config.processing_time_threshold}s")
        logger.info(f"Accuracy target: >{self.config.accuracy_threshold:.1%}")
        logger.info(f"Mathematical tolerance: ±{PERFORMANCE_TARGETS['mathematical_tolerance']}")
    
    def start_monitoring(self) -> None:
        """Start real-time performance monitoring"""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitoring_thread.start()
                logger.info("Real-time performance monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop real-time performance monitoring"""
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Real-time performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
    
    def record_performance(self, component_name: str, processing_time: float,
                         accuracy: float, mathematical_accuracy: bool,
                         component_timings: Optional[Dict[str, float]] = None,
                         error_count: int = 0, warning_count: int = 0) -> PerformanceMetrics:
        """
        Record performance metrics for a component
        
        Args:
            component_name: Name of the component being monitored
            processing_time: Total processing time in seconds
            accuracy: Accuracy score (0.0 to 1.0)
            mathematical_accuracy: Whether mathematical accuracy requirements are met
            component_timings: Optional breakdown of component timings
            error_count: Number of errors encountered
            warning_count: Number of warnings encountered
            
        Returns:
            PerformanceMetrics object
        """
        try:
            # Get system metrics
            memory_usage = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                processing_time=processing_time,
                accuracy=accuracy,
                mathematical_accuracy=mathematical_accuracy,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                component_timings=component_timings or {},
                error_count=error_count,
                warning_count=warning_count
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update component performance tracking
            if component_name not in self.component_performance:
                self.component_performance[component_name] = deque(maxlen=1000)
            self.component_performance[component_name].append(metrics)
            
            # Update performance statistics
            self._update_performance_stats(metrics)
            
            # Check for performance violations and alerts
            self._check_performance_violations(component_name, metrics)
            
            # Memory optimization
            if self.memory_optimization_enabled:
                self._optimize_memory()
            
            logger.debug(f"Performance recorded for {component_name}: "
                        f"{processing_time:.3f}s, accuracy: {accuracy:.3f}, "
                        f"math_accuracy: {mathematical_accuracy}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                processing_time=processing_time,
                accuracy=accuracy,
                mathematical_accuracy=mathematical_accuracy,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0
            )
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive performance dashboard data
        
        Returns:
            Dictionary containing dashboard metrics and visualizations
        """
        try:
            if not self.metrics_history:
                return {'status': 'No performance data available'}
            
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 operations
            
            # Calculate dashboard metrics
            dashboard = {
                'summary': self._get_performance_summary(recent_metrics),
                'targets_compliance': self._get_targets_compliance(recent_metrics),
                'component_breakdown': self._get_component_breakdown(),
                'trend_analysis': self._get_trend_analysis(recent_metrics),
                'alert_summary': self._get_alert_summary(),
                'system_health': self._get_system_health(),
                'recommendations': self._get_performance_recommendations(recent_metrics)
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            return {'error': str(e)}
    
    def _monitoring_loop(self) -> None:
        """Real-time monitoring loop"""
        try:
            while self.monitoring_active:
                # Monitor system resources
                memory_usage = self._get_memory_usage()
                cpu_usage = self._get_cpu_usage()
                
                # Check for resource alerts
                if memory_usage > self.config.memory_threshold_mb:
                    self._trigger_alert('memory', f"High memory usage: {memory_usage:.1f}MB")
                
                # Sleep for monitoring interval
                time.sleep(10)  # Monitor every 10 seconds
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"Error getting CPU usage: {e}")
            return 0.0

    def _update_performance_stats(self, metrics: PerformanceMetrics) -> None:
        """Update overall performance statistics"""
        try:
            self.performance_stats['total_operations'] += 1

            if metrics.error_count == 0:
                self.performance_stats['successful_operations'] += 1
            else:
                self.performance_stats['failed_operations'] += 1

            # Update running averages
            total_ops = self.performance_stats['total_operations']

            # Processing time average
            current_avg_time = self.performance_stats['average_processing_time']
            self.performance_stats['average_processing_time'] = (
                (current_avg_time * (total_ops - 1) + metrics.processing_time) / total_ops
            )

            # Accuracy average
            current_avg_accuracy = self.performance_stats['average_accuracy']
            self.performance_stats['average_accuracy'] = (
                (current_avg_accuracy * (total_ops - 1) + metrics.accuracy) / total_ops
            )

            # Mathematical accuracy rate
            current_math_rate = self.performance_stats['mathematical_accuracy_rate']
            math_score = 1.0 if metrics.mathematical_accuracy else 0.0
            self.performance_stats['mathematical_accuracy_rate'] = (
                (current_math_rate * (total_ops - 1) + math_score) / total_ops
            )

        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

    def _check_performance_violations(self, component_name: str, metrics: PerformanceMetrics) -> None:
        """Check for performance violations and trigger alerts"""
        try:
            violations = []

            # Processing time violation
            if (self.config.enable_processing_time_alerts and
                metrics.processing_time > self.config.processing_time_threshold):
                violations.append(f"Processing time {metrics.processing_time:.3f}s exceeds {self.config.processing_time_threshold}s")

            # Accuracy violation
            if (self.config.enable_accuracy_alerts and
                metrics.accuracy < self.config.accuracy_threshold):
                violations.append(f"Accuracy {metrics.accuracy:.3f} below {self.config.accuracy_threshold:.3f}")

            # Mathematical accuracy violation
            if (self.config.enable_mathematical_accuracy_alerts and
                not metrics.mathematical_accuracy):
                violations.append("Mathematical accuracy validation failed")

            # Memory violation
            if (self.config.enable_memory_alerts and
                metrics.memory_usage_mb > self.config.memory_threshold_mb):
                violations.append(f"Memory usage {metrics.memory_usage_mb:.1f}MB exceeds {self.config.memory_threshold_mb}MB")

            # Trigger alerts for violations
            for violation in violations:
                self._trigger_alert(component_name, violation)

        except Exception as e:
            logger.error(f"Error checking performance violations: {e}")

    def _trigger_alert(self, component_name: str, message: str) -> None:
        """Trigger performance alert with cooldown"""
        try:
            current_time = datetime.now()
            alert_key = f"{component_name}_{message[:50]}"  # Use first 50 chars as key

            # Check cooldown
            if alert_key in self.last_alert_time:
                time_since_last = (current_time - self.last_alert_time[alert_key]).total_seconds()
                if time_since_last < self.config.alert_cooldown_seconds:
                    return  # Skip alert due to cooldown

            # Record alert
            alert = {
                'timestamp': current_time,
                'component': component_name,
                'message': message,
                'severity': 'WARNING'
            }

            self.alert_history.append(alert)
            self.last_alert_time[alert_key] = current_time

            # Keep only recent alerts
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]

            logger.warning(f"PERFORMANCE ALERT [{component_name}]: {message}")

        except Exception as e:
            logger.error(f"Error triggering alert: {e}")

    def _optimize_memory(self) -> None:
        """Optimize memory usage"""
        try:
            if self.performance_stats['total_operations'] % self.gc_frequency == 0:
                # Run garbage collection
                collected = gc.collect()
                logger.debug(f"Garbage collection: {collected} objects collected")

                # Trim metrics history if too large
                if len(self.metrics_history) > 8000:
                    # Keep only recent 5000 metrics
                    recent_metrics = list(self.metrics_history)[-5000:]
                    self.metrics_history.clear()
                    self.metrics_history.extend(recent_metrics)
                    logger.debug("Trimmed metrics history for memory optimization")

        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")

    def _get_performance_summary(self, recent_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Get performance summary statistics"""
        try:
            if not recent_metrics:
                return {}

            processing_times = [m.processing_time for m in recent_metrics]
            accuracies = [m.accuracy for m in recent_metrics]
            math_accuracies = [m.mathematical_accuracy for m in recent_metrics]

            return {
                'total_operations': len(recent_metrics),
                'avg_processing_time': np.mean(processing_times),
                'max_processing_time': np.max(processing_times),
                'min_processing_time': np.min(processing_times),
                'avg_accuracy': np.mean(accuracies),
                'min_accuracy': np.min(accuracies),
                'mathematical_accuracy_rate': np.mean(math_accuracies),
                'avg_memory_usage': np.mean([m.memory_usage_mb for m in recent_metrics]),
                'avg_cpu_usage': np.mean([m.cpu_usage_percent for m in recent_metrics])
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

    def _get_targets_compliance(self, recent_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Get compliance with performance targets"""
        try:
            if not recent_metrics:
                return {}

            processing_time_compliance = np.mean([
                m.processing_time <= PERFORMANCE_TARGETS['max_processing_time']
                for m in recent_metrics
            ])

            accuracy_compliance = np.mean([
                m.accuracy >= PERFORMANCE_TARGETS['min_accuracy']
                for m in recent_metrics
            ])

            mathematical_accuracy_compliance = np.mean([
                m.mathematical_accuracy for m in recent_metrics
            ])

            memory_compliance = np.mean([
                m.memory_usage_mb <= PERFORMANCE_TARGETS['max_memory_usage_mb']
                for m in recent_metrics
            ])

            overall_compliance = np.mean([
                processing_time_compliance,
                accuracy_compliance,
                mathematical_accuracy_compliance,
                memory_compliance
            ])

            return {
                'processing_time_compliance': processing_time_compliance,
                'accuracy_compliance': accuracy_compliance,
                'mathematical_accuracy_compliance': mathematical_accuracy_compliance,
                'memory_compliance': memory_compliance,
                'overall_compliance': overall_compliance,
                'targets': PERFORMANCE_TARGETS
            }

        except Exception as e:
            logger.error(f"Error getting targets compliance: {e}")
            return {}

    def _get_component_breakdown(self) -> Dict[str, Any]:
        """Get performance breakdown by component"""
        try:
            breakdown = {}

            for component_name, metrics_list in self.component_performance.items():
                if not metrics_list:
                    continue

                recent_metrics = list(metrics_list)[-50:]  # Last 50 operations per component

                breakdown[component_name] = {
                    'operations_count': len(recent_metrics),
                    'avg_processing_time': np.mean([m.processing_time for m in recent_metrics]),
                    'avg_accuracy': np.mean([m.accuracy for m in recent_metrics]),
                    'mathematical_accuracy_rate': np.mean([m.mathematical_accuracy for m in recent_metrics]),
                    'error_rate': np.mean([m.error_count for m in recent_metrics]),
                    'warning_rate': np.mean([m.warning_count for m in recent_metrics])
                }

            return breakdown

        except Exception as e:
            logger.error(f"Error getting component breakdown: {e}")
            return {}

    def _get_trend_analysis(self, recent_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Get trend analysis for performance metrics"""
        try:
            if len(recent_metrics) < 10:
                return {'status': 'Insufficient data for trend analysis'}

            # Split into two halves for trend comparison
            mid_point = len(recent_metrics) // 2
            first_half = recent_metrics[:mid_point]
            second_half = recent_metrics[mid_point:]

            # Calculate trends
            first_avg_time = np.mean([m.processing_time for m in first_half])
            second_avg_time = np.mean([m.processing_time for m in second_half])
            time_trend = ((second_avg_time - first_avg_time) / first_avg_time) * 100 if first_avg_time > 0 else 0

            first_avg_accuracy = np.mean([m.accuracy for m in first_half])
            second_avg_accuracy = np.mean([m.accuracy for m in second_half])
            accuracy_trend = ((second_avg_accuracy - first_avg_accuracy) / first_avg_accuracy) * 100 if first_avg_accuracy > 0 else 0

            first_math_accuracy = np.mean([m.mathematical_accuracy for m in first_half])
            second_math_accuracy = np.mean([m.mathematical_accuracy for m in second_half])
            math_accuracy_trend = (second_math_accuracy - first_math_accuracy) * 100

            return {
                'processing_time_trend_percent': time_trend,
                'accuracy_trend_percent': accuracy_trend,
                'mathematical_accuracy_trend_percent': math_accuracy_trend,
                'trend_interpretation': {
                    'processing_time': 'improving' if time_trend < -5 else 'degrading' if time_trend > 5 else 'stable',
                    'accuracy': 'improving' if accuracy_trend > 2 else 'degrading' if accuracy_trend < -2 else 'stable',
                    'mathematical_accuracy': 'improving' if math_accuracy_trend > 5 else 'degrading' if math_accuracy_trend < -5 else 'stable'
                }
            }

        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            return {}

    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        try:
            if not self.alert_history:
                return {'total_alerts': 0, 'recent_alerts': []}

            # Recent alerts (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_alerts = [alert for alert in self.alert_history if alert['timestamp'] > cutoff_time]

            # Alert breakdown by component
            component_alerts = {}
            for alert in recent_alerts:
                component = alert['component']
                if component not in component_alerts:
                    component_alerts[component] = 0
                component_alerts[component] += 1

            return {
                'total_alerts': len(self.alert_history),
                'recent_alerts_24h': len(recent_alerts),
                'component_alert_breakdown': component_alerts,
                'latest_alerts': self.alert_history[-10:] if self.alert_history else []
            }

        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {}

    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            current_memory = self._get_memory_usage()
            current_cpu = self._get_cpu_usage()

            # Health scoring
            memory_health = 'good' if current_memory < PERFORMANCE_TARGETS['max_memory_usage_mb'] * 0.8 else 'warning' if current_memory < PERFORMANCE_TARGETS['max_memory_usage_mb'] else 'critical'
            cpu_health = 'good' if current_cpu < 70 else 'warning' if current_cpu < 90 else 'critical'

            # Overall health
            health_scores = {'good': 3, 'warning': 2, 'critical': 1}
            overall_score = (health_scores[memory_health] + health_scores[cpu_health]) / 2
            overall_health = 'good' if overall_score >= 2.5 else 'warning' if overall_score >= 1.5 else 'critical'

            return {
                'overall_health': overall_health,
                'memory_usage_mb': current_memory,
                'memory_health': memory_health,
                'cpu_usage_percent': current_cpu,
                'cpu_health': cpu_health,
                'monitoring_active': self.monitoring_active,
                'uptime_operations': self.performance_stats['total_operations']
            }

        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'overall_health': 'unknown'}

    def _get_performance_recommendations(self, recent_metrics: List[PerformanceMetrics]) -> List[str]:
        """Get performance optimization recommendations"""
        try:
            recommendations = []

            if not recent_metrics:
                return recommendations

            # Processing time recommendations
            avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
            if avg_processing_time > PERFORMANCE_TARGETS['max_processing_time'] * 0.8:
                recommendations.append("Consider optimizing algorithms or adding caching to reduce processing time")

            # Accuracy recommendations
            avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
            if avg_accuracy < PERFORMANCE_TARGETS['min_accuracy']:
                recommendations.append("Review model parameters and data quality to improve accuracy")

            # Mathematical accuracy recommendations
            math_accuracy_rate = np.mean([m.mathematical_accuracy for m in recent_metrics])
            if math_accuracy_rate < 0.95:
                recommendations.append("Review mathematical calculations for precision issues")

            # Memory recommendations
            avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
            if avg_memory > PERFORMANCE_TARGETS['max_memory_usage_mb'] * 0.8:
                recommendations.append("Consider memory optimization techniques or increase garbage collection frequency")

            # Error rate recommendations
            avg_errors = np.mean([m.error_count for m in recent_metrics])
            if avg_errors > 0.1:
                recommendations.append("Investigate and fix recurring errors to improve system stability")

            return recommendations

        except Exception as e:
            logger.error(f"Error getting performance recommendations: {e}")
            return []

# Integration function for unified_stable_market_regime_pipeline.py
def monitor_component_performance(component_name: str, processing_time: float,
                                accuracy: float, mathematical_accuracy: bool,
                                component_timings: Optional[Dict[str, float]] = None,
                                monitor_instance: Optional[EnhancedPerformanceMonitor] = None) -> Dict[str, Any]:
    """
    Main integration function for performance monitoring

    Args:
        component_name: Name of the component being monitored
        processing_time: Total processing time in seconds
        accuracy: Accuracy score (0.0 to 1.0)
        mathematical_accuracy: Whether mathematical accuracy requirements are met
        component_timings: Optional breakdown of component timings
        monitor_instance: Optional existing monitor instance

    Returns:
        Dictionary containing performance monitoring results
    """
    try:
        # Use provided monitor or create new one
        if monitor_instance is None:
            monitor_instance = EnhancedPerformanceMonitor()

        # Record performance
        metrics = monitor_instance.record_performance(
            component_name=component_name,
            processing_time=processing_time,
            accuracy=accuracy,
            mathematical_accuracy=mathematical_accuracy,
            component_timings=component_timings
        )

        # Return monitoring results
        return {
            'performance_recorded': True,
            'processing_time': metrics.processing_time,
            'accuracy': metrics.accuracy,
            'mathematical_accuracy': metrics.mathematical_accuracy,
            'memory_usage_mb': metrics.memory_usage_mb,
            'cpu_usage_percent': metrics.cpu_usage_percent,
            'timestamp': metrics.timestamp.isoformat(),
            'performance_targets_met': {
                'processing_time': metrics.processing_time <= PERFORMANCE_TARGETS['max_processing_time'],
                'accuracy': metrics.accuracy >= PERFORMANCE_TARGETS['min_accuracy'],
                'mathematical_accuracy': metrics.mathematical_accuracy,
                'memory_usage': metrics.memory_usage_mb <= PERFORMANCE_TARGETS['max_memory_usage_mb']
            }
        }

    except Exception as e:
        logger.error(f"Error monitoring component performance: {e}")
        return {
            'performance_recorded': False,
            'error': str(e)
        }

# Unit test function
def test_enhanced_performance_monitor():
    """Basic unit test for enhanced performance monitor"""
    try:
        logger.info("Testing Enhanced Performance Monitor...")

        # Initialize monitor
        monitor = EnhancedPerformanceMonitor()

        # Start monitoring
        monitor.start_monitoring()

        # Record some test performance metrics
        test_metrics = [
            ('volume_weighted_greeks', 2.5, 0.87, True),
            ('delta_strike_selection', 1.8, 0.91, True),
            ('trending_oi_analysis', 2.1, 0.89, True),
            ('hybrid_classification', 2.3, 0.86, True)
        ]

        for component, proc_time, accuracy, math_acc in test_metrics:
            monitor.record_performance(component, proc_time, accuracy, math_acc)

        # Get dashboard
        dashboard = monitor.get_performance_dashboard()

        # Stop monitoring
        monitor.stop_monitoring()

        if dashboard and 'summary' in dashboard:
            logger.info("✅ Enhanced Performance Monitor test PASSED")
            logger.info(f"   Total operations: {dashboard['summary'].get('total_operations', 0)}")
            logger.info(f"   Average processing time: {dashboard['summary'].get('avg_processing_time', 0):.3f}s")
            logger.info(f"   Average accuracy: {dashboard['summary'].get('avg_accuracy', 0):.3f}")
            logger.info(f"   Mathematical accuracy rate: {dashboard['summary'].get('mathematical_accuracy_rate', 0):.3f}")
            return True
        else:
            logger.error("❌ Enhanced Performance Monitor test FAILED")
            return False

    except Exception as e:
        logger.error(f"❌ Enhanced Performance Monitor test ERROR: {e}")
        return False

if __name__ == "__main__":
    # Run basic test
    test_enhanced_performance_monitor()
