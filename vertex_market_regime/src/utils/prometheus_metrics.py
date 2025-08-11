"""
Prometheus Metrics Integration for Market Regime Components

Comprehensive metrics collection and monitoring for all 8 components
with performance tracking, error monitoring, and business metrics.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
from contextlib import contextmanager

# Prometheus client imports
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info, Enum as PrometheusEnum,
        CollectorRegistry, MetricsHandler, start_http_server, push_to_gateway,
        REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus_client")


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class ComponentMetrics:
    """Component-specific metrics structure"""
    component_id: int
    component_name: str
    processing_time_histogram: Optional[Any] = None
    memory_usage_gauge: Optional[Any] = None
    accuracy_gauge: Optional[Any] = None
    error_counter: Optional[Any] = None
    prediction_counter: Optional[Any] = None
    feature_count_gauge: Optional[Any] = None
    weight_gauges: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.weight_gauges is None:
            self.weight_gauges = {}


class PrometheusMetricsManager:
    """
    Prometheus metrics manager for Market Regime components
    
    Provides comprehensive metrics collection, monitoring, and alerting
    capabilities for all 8 components with production-ready monitoring.
    """
    
    def __init__(self, 
                 registry: Optional[CollectorRegistry] = None,
                 metrics_port: int = 9090,
                 push_gateway_url: Optional[str] = None,
                 job_name: str = "market_regime_framework"):
        """
        Initialize Prometheus metrics manager
        
        Args:
            registry: Prometheus registry (uses default if None)
            metrics_port: Port for metrics HTTP server
            push_gateway_url: URL for push gateway (optional)
            job_name: Job name for push gateway
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client library not available")
        
        self.registry = registry or REGISTRY
        self.metrics_port = metrics_port
        self.push_gateway_url = push_gateway_url
        self.job_name = job_name
        self.logger = logging.getLogger(__name__)
        
        # Component metrics storage
        self.component_metrics: Dict[int, ComponentMetrics] = {}
        
        # Global metrics
        self._initialize_global_metrics()
        
        # HTTP server for metrics
        self._metrics_server = None
        self._server_thread = None
        
        self.logger.info("PrometheusMetricsManager initialized")
    
    def _initialize_global_metrics(self):
        """Initialize global system metrics"""
        
        # System-wide metrics
        self.system_uptime = Gauge(
            'market_regime_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        self.total_requests = Counter(
            'market_regime_total_requests',
            'Total number of analysis requests',
            ['component', 'status'],
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'market_regime_system_memory_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'market_regime_system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.data_pipeline_status = Gauge(
            'market_regime_data_pipeline_status',
            'Data pipeline health status (1=healthy, 0=unhealthy)',
            ['pipeline_stage'],
            registry=self.registry
        )
        
        # Business metrics
        self.regime_predictions = Counter(
            'market_regime_predictions_total',
            'Total regime predictions made',
            ['component', 'predicted_regime', 'actual_regime'],
            registry=self.registry
        )
        
        self.prediction_accuracy = Gauge(
            'market_regime_prediction_accuracy',
            'Current prediction accuracy by component',
            ['component', 'time_window'],
            registry=self.registry
        )
        
        self.model_confidence = Gauge(
            'market_regime_model_confidence',
            'Model confidence score',
            ['component'],
            registry=self.registry
        )
        
        # Weight tracking
        self.adaptive_weights = Gauge(
            'market_regime_adaptive_weights',
            'Current adaptive weights',
            ['component', 'weight_name'],
            registry=self.registry
        )
        
        # Performance SLA metrics
        self.sla_compliance = Gauge(
            'market_regime_sla_compliance',
            'SLA compliance percentage',
            ['component', 'sla_type'],
            registry=self.registry
        )
        
        self.logger.info("Global metrics initialized")
    
    def initialize_component_metrics(self, component_id: int, component_name: str):
        """
        Initialize metrics for a specific component
        
        Args:
            component_id: Component identifier (1-8)
            component_name: Human-readable component name
        """
        if component_id in self.component_metrics:
            self.logger.warning(f"Metrics for component {component_id} already initialized")
            return
        
        # Component-specific metrics
        metrics = ComponentMetrics(
            component_id=component_id,
            component_name=component_name,
            
            # Processing time histogram with buckets optimized for our SLAs
            processing_time_histogram=Histogram(
                f'market_regime_component_{component_id}_processing_seconds',
                f'Processing time for {component_name}',
                buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, float('inf')),
                registry=self.registry
            ),
            
            # Memory usage gauge
            memory_usage_gauge=Gauge(
                f'market_regime_component_{component_id}_memory_bytes',
                f'Memory usage for {component_name}',
                registry=self.registry
            ),
            
            # Accuracy gauge
            accuracy_gauge=Gauge(
                f'market_regime_component_{component_id}_accuracy',
                f'Current accuracy for {component_name}',
                registry=self.registry
            ),
            
            # Error counter
            error_counter=Counter(
                f'market_regime_component_{component_id}_errors_total',
                f'Total errors for {component_name}',
                ['error_type'],
                registry=self.registry
            ),
            
            # Prediction counter
            prediction_counter=Counter(
                f'market_regime_component_{component_id}_predictions_total',
                f'Total predictions for {component_name}',
                ['regime_type'],
                registry=self.registry
            ),
            
            # Feature count gauge
            feature_count_gauge=Gauge(
                f'market_regime_component_{component_id}_features_count',
                f'Number of features for {component_name}',
                registry=self.registry
            )
        )
        
        # Initialize weight gauges based on component type
        if component_id == 2:  # Component 2 Greeks Sentiment
            weight_names = ['gamma_weight', 'delta_weight', 'theta_weight', 'vega_weight']
            for weight_name in weight_names:
                metrics.weight_gauges[weight_name] = Gauge(
                    f'market_regime_component_{component_id}_{weight_name}',
                    f'{weight_name} for {component_name}',
                    registry=self.registry
                )
        
        self.component_metrics[component_id] = metrics
        self.logger.info(f"Initialized metrics for component {component_id}: {component_name}")
    
    def record_processing_time(self, component_id: int, processing_time_seconds: float):
        """Record processing time for a component"""
        if component_id in self.component_metrics:
            self.component_metrics[component_id].processing_time_histogram.observe(processing_time_seconds)
            
            # Update SLA compliance
            sla_target = self._get_sla_target(component_id, 'processing_time_ms')
            compliance = 1.0 if (processing_time_seconds * 1000) <= sla_target else 0.0
            self.sla_compliance.labels(component=f'component_{component_id}', sla_type='processing_time').set(compliance)
    
    def record_memory_usage(self, component_id: int, memory_bytes: float):
        """Record memory usage for a component"""
        if component_id in self.component_metrics:
            self.component_metrics[component_id].memory_usage_gauge.set(memory_bytes)
            
            # Update SLA compliance
            sla_target = self._get_sla_target(component_id, 'memory_budget_mb') * 1024 * 1024  # Convert to bytes
            compliance = 1.0 if memory_bytes <= sla_target else 0.0
            self.sla_compliance.labels(component=f'component_{component_id}', sla_type='memory_usage').set(compliance)
    
    def record_prediction(self, 
                         component_id: int,
                         predicted_regime: str,
                         actual_regime: Optional[str] = None,
                         accuracy: Optional[float] = None,
                         confidence: Optional[float] = None):
        """Record a prediction made by a component"""
        if component_id in self.component_metrics:
            # Record prediction count
            self.component_metrics[component_id].prediction_counter.labels(regime_type=predicted_regime).inc()
            
            # Record global prediction
            if actual_regime:
                self.regime_predictions.labels(
                    component=f'component_{component_id}',
                    predicted_regime=predicted_regime,
                    actual_regime=actual_regime
                ).inc()
            
            # Update accuracy if provided
            if accuracy is not None:
                self.component_metrics[component_id].accuracy_gauge.set(accuracy)
                self.prediction_accuracy.labels(
                    component=f'component_{component_id}',
                    time_window='current'
                ).set(accuracy)
            
            # Update confidence if provided
            if confidence is not None:
                self.model_confidence.labels(component=f'component_{component_id}').set(confidence)
    
    def record_error(self, component_id: int, error_type: str):
        """Record an error for a component"""
        if component_id in self.component_metrics:
            self.component_metrics[component_id].error_counter.labels(error_type=error_type).inc()
        
        # Update global request counter
        self.total_requests.labels(component=f'component_{component_id}', status='error').inc()
    
    def record_success(self, component_id: int):
        """Record a successful operation for a component"""
        self.total_requests.labels(component=f'component_{component_id}', status='success').inc()
    
    def update_feature_count(self, component_id: int, feature_count: int):
        """Update feature count for a component"""
        if component_id in self.component_metrics:
            self.component_metrics[component_id].feature_count_gauge.set(feature_count)
    
    def update_adaptive_weights(self, component_id: int, weights: Dict[str, float]):
        """Update adaptive weights for a component"""
        component_label = f'component_{component_id}'
        
        for weight_name, weight_value in weights.items():
            self.adaptive_weights.labels(component=component_label, weight_name=weight_name).set(weight_value)
            
            # Update component-specific weight gauges if they exist
            if (component_id in self.component_metrics and 
                weight_name in self.component_metrics[component_id].weight_gauges):
                self.component_metrics[component_id].weight_gauges[weight_name].set(weight_value)
    
    def update_system_metrics(self, uptime_seconds: float, memory_bytes: float, cpu_percent: float):
        """Update system-wide metrics"""
        self.system_uptime.set(uptime_seconds)
        self.system_memory_usage.set(memory_bytes)
        self.system_cpu_usage.set(cpu_percent)
    
    def update_data_pipeline_status(self, pipeline_stage: str, is_healthy: bool):
        """Update data pipeline health status"""
        self.data_pipeline_status.labels(pipeline_stage=pipeline_stage).set(1.0 if is_healthy else 0.0)
    
    def start_metrics_server(self):
        """Start HTTP server for metrics endpoint"""
        if self._metrics_server is not None:
            self.logger.warning("Metrics server already running")
            return
        
        try:
            self._metrics_server = start_http_server(self.metrics_port, registry=self.registry)
            self.logger.info(f"Metrics server started on port {self.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def stop_metrics_server(self):
        """Stop HTTP server for metrics endpoint"""
        if self._metrics_server is not None:
            self._metrics_server.shutdown()
            self._metrics_server = None
            self.logger.info("Metrics server stopped")
    
    def push_metrics(self):
        """Push metrics to gateway if configured"""
        if self.push_gateway_url:
            try:
                push_to_gateway(
                    gateway=self.push_gateway_url,
                    job=self.job_name,
                    registry=self.registry
                )
                self.logger.debug("Metrics pushed to gateway")
            except Exception as e:
                self.logger.error(f"Failed to push metrics to gateway: {e}")
    
    def _get_sla_target(self, component_id: int, sla_type: str) -> float:
        """Get SLA target for a component and metric type"""
        # SLA targets by component
        sla_targets = {
            1: {'processing_time_ms': 150, 'memory_budget_mb': 320},
            2: {'processing_time_ms': 120, 'memory_budget_mb': 280},
            3: {'processing_time_ms': 200, 'memory_budget_mb': 300},
            4: {'processing_time_ms': 200, 'memory_budget_mb': 250},
            5: {'processing_time_ms': 200, 'memory_budget_mb': 270},
            6: {'processing_time_ms': 180, 'memory_budget_mb': 450},
            7: {'processing_time_ms': 150, 'memory_budget_mb': 220},
            8: {'processing_time_ms': 100, 'memory_budget_mb': 180}
        }
        
        return sla_targets.get(component_id, {}).get(sla_type, 1000)  # Default fallback
    
    @contextmanager
    def timer(self, component_id: int):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            processing_time = time.time() - start_time
            self.record_processing_time(component_id, processing_time)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all current metrics"""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'components_initialized': list(self.component_metrics.keys()),
            'metrics_server_running': self._metrics_server is not None,
            'registry_metrics_count': len(list(self.registry.collect()))
        }
        
        return summary


def metrics_decorator(component_id: int):
    """
    Decorator for automatic metrics collection
    
    Args:
        component_id: Component identifier
        
    Returns:
        Decorated function with automatic metrics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics_manager = get_metrics_manager()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success
                processing_time = time.time() - start_time
                metrics_manager.record_processing_time(component_id, processing_time)
                metrics_manager.record_success(component_id)
                
                return result
                
            except Exception as e:
                # Record error
                error_type = type(e).__name__
                metrics_manager.record_error(component_id, error_type)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics_manager = get_metrics_manager()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                processing_time = time.time() - start_time
                metrics_manager.record_processing_time(component_id, processing_time)
                metrics_manager.record_success(component_id)
                
                return result
                
            except Exception as e:
                # Record error
                error_type = type(e).__name__
                metrics_manager.record_error(component_id, error_type)
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global metrics manager instance
_metrics_manager: Optional[PrometheusMetricsManager] = None


def get_metrics_manager() -> PrometheusMetricsManager:
    """Get global metrics manager instance"""
    global _metrics_manager
    
    if _metrics_manager is None:
        _metrics_manager = PrometheusMetricsManager()
    
    return _metrics_manager


def initialize_metrics_for_component(component_id: int, component_name: str):
    """Convenience function to initialize component metrics"""
    manager = get_metrics_manager()
    manager.initialize_component_metrics(component_id, component_name)


# Example Grafana dashboard query examples
GRAFANA_QUERIES = {
    'component_processing_time': """
        rate(market_regime_component_2_processing_seconds_sum[5m]) / 
        rate(market_regime_component_2_processing_seconds_count[5m])
    """,
    
    'component_error_rate': """
        rate(market_regime_component_2_errors_total[5m]) / 
        rate(market_regime_total_requests{component="component_2"}[5m]) * 100
    """,
    
    'sla_compliance': """
        avg_over_time(market_regime_sla_compliance{component="component_2"}[1h]) * 100
    """,
    
    'prediction_accuracy': """
        market_regime_prediction_accuracy{component="component_2", time_window="current"}
    """,
    
    'memory_usage_trend': """
        market_regime_component_2_memory_bytes / (1024*1024)
    """,
    
    'gamma_weight_evolution': """
        market_regime_component_2_gamma_weight
    """
}


# Example alerting rules for Prometheus
ALERTING_RULES = """
groups:
- name: market_regime_alerts
  rules:
  - alert: ComponentProcessingTimeTooHigh
    expr: rate(market_regime_component_2_processing_seconds_sum[5m]) / rate(market_regime_component_2_processing_seconds_count[5m]) > 0.12
    for: 2m
    labels:
      severity: warning
      component: "component_2"
    annotations:
      summary: "Component 2 processing time exceeds SLA"
      description: "Component 2 average processing time is {{ $value }}s, exceeding 120ms SLA"

  - alert: ComponentErrorRateHigh
    expr: rate(market_regime_component_2_errors_total[5m]) / rate(market_regime_total_requests{component="component_2"}[5m]) > 0.05
    for: 1m
    labels:
      severity: critical
      component: "component_2"
    annotations:
      summary: "Component 2 error rate too high"
      description: "Component 2 error rate is {{ $value | humanizePercentage }}"

  - alert: ComponentAccuracyLow
    expr: market_regime_prediction_accuracy{component="component_2"} < 0.85
    for: 5m
    labels:
      severity: warning
      component: "component_2"
    annotations:
      summary: "Component 2 prediction accuracy below target"
      description: "Component 2 accuracy is {{ $value | humanizePercentage }}, below 85% target"

  - alert: GammaWeightOutOfRange
    expr: market_regime_component_2_gamma_weight < 1.0 or market_regime_component_2_gamma_weight > 2.0
    for: 1m
    labels:
      severity: critical
      component: "component_2"
    annotations:
      summary: "Component 2 gamma weight out of acceptable range"
      description: "Gamma weight is {{ $value }}, outside acceptable range [1.0, 2.0]"
"""