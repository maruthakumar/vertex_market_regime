"""
Feature Store Monitoring and Observability
Implements comprehensive monitoring, alerting, and cost tracking for Feature Store operations
"""

import asyncio
import logging
import time
import yaml
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import threading

from google.cloud import monitoring_v3
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.monitoring_dashboard import v1 as dashboard_v1
import pandas as pd
import numpy as np


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics to track"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_PERFORMANCE = "cache_performance"
    COST = "cost"
    FEATURE_FRESHNESS = "feature_freshness"
    DATA_QUALITY = "data_quality"


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    description: str
    metric_type: MetricType
    threshold_value: float
    comparison_operator: str  # "GREATER_THAN", "LESS_THAN", etc.
    duration_seconds: int
    severity: AlertSeverity
    notification_channels: List[str]
    enabled: bool = True


@dataclass
class MonitoringMetric:
    """Individual monitoring metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class CostMetrics:
    """Cost tracking metrics"""
    feature_store_cost_usd: float
    bigquery_cost_usd: float
    compute_cost_usd: float
    total_monthly_cost_usd: float
    cost_per_request_usd: float
    budget_utilization_percent: float


class FeatureStoreMonitoring:
    """
    Comprehensive monitoring system for Vertex AI Feature Store
    Provides metrics collection, alerting, cost tracking, and dashboards
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "dev"):
        """Initialize monitoring system"""
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "feature_store_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Google Cloud monitoring
        self._initialize_monitoring_clients()
        
        # Metrics storage
        self.metrics_buffer: List[MonitoringMetric] = []
        self.alert_rules: List[AlertRule] = []
        self.cost_history: List[CostMetrics] = []
        
        # Threading for background monitoring
        self.monitoring_active = True
        self.metrics_lock = threading.Lock()
        
        # Initialize alert rules
        self._setup_default_alert_rules()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _initialize_monitoring_clients(self):
        """Initialize Google Cloud monitoring clients"""
        self.project_id = self.config["project_config"]["project_id"]
        self.project_name = f"projects/{self.project_id}"
        
        # Monitoring clients
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.alert_client = monitoring_v3.AlertPolicyServiceClient()
        self.dashboard_client = dashboard_v1.DashboardsServiceClient()
        
        # BigQuery client for cost analysis
        self.bigquery_client = bigquery.Client(project=self.project_id)
        
        self.logger.info(f"Initialized monitoring for project: {self.project_id}")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for Feature Store monitoring"""
        
        # Latency alerts
        self.alert_rules.extend([
            AlertRule(
                name="feature_store_high_latency_p99",
                description="P99 latency exceeds 50ms SLA",
                metric_type=MetricType.LATENCY,
                threshold_value=50.0,
                comparison_operator="GREATER_THAN",
                duration_seconds=300,  # 5 minutes
                severity=AlertSeverity.CRITICAL,
                notification_channels=["email", "slack"]
            ),
            AlertRule(
                name="feature_store_high_latency_p95",
                description="P95 latency exceeds 40ms target",
                metric_type=MetricType.LATENCY,
                threshold_value=40.0,
                comparison_operator="GREATER_THAN", 
                duration_seconds=300,
                severity=AlertSeverity.WARNING,
                notification_channels=["email"]
            )
        ])
        
        # Error rate alerts
        self.alert_rules.append(
            AlertRule(
                name="feature_store_high_error_rate",
                description="Error rate exceeds 1%",
                metric_type=MetricType.ERROR_RATE,
                threshold_value=1.0,
                comparison_operator="GREATER_THAN",
                duration_seconds=180,  # 3 minutes
                severity=AlertSeverity.ERROR,
                notification_channels=["email", "slack"]
            )
        )
        
        # Throughput alerts
        self.alert_rules.append(
            AlertRule(
                name="feature_store_low_throughput",
                description="Throughput below expected minimum",
                metric_type=MetricType.THROUGHPUT,
                threshold_value=10.0,  # 10 RPS minimum
                comparison_operator="LESS_THAN",
                duration_seconds=600,  # 10 minutes
                severity=AlertSeverity.WARNING,
                notification_channels=["email"]
            )
        )
        
        # Feature freshness alerts
        self.alert_rules.append(
            AlertRule(
                name="feature_store_stale_features",
                description="Features older than 5 minutes",
                metric_type=MetricType.FEATURE_FRESHNESS,
                threshold_value=300.0,  # 5 minutes
                comparison_operator="GREATER_THAN",
                duration_seconds=600,
                severity=AlertSeverity.WARNING,
                notification_channels=["email"]
            )
        )
        
        # Cost alerts
        self.alert_rules.append(
            AlertRule(
                name="feature_store_cost_budget_exceeded",
                description="Monthly cost budget exceeded 80%",
                metric_type=MetricType.COST,
                threshold_value=80.0,  # 80% of budget
                comparison_operator="GREATER_THAN",
                duration_seconds=3600,  # 1 hour
                severity=AlertSeverity.ERROR,
                notification_channels=["email", "slack"]
            )
        )
        
        self.logger.info(f"Initialized {len(self.alert_rules)} alert rules")
    
    def record_metric(self, name: str, value: float, unit: str, labels: Optional[Dict[str, str]] = None, description: str = ""):
        """Record a monitoring metric"""
        
        metric = MonitoringMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            description=description
        )
        
        with self.metrics_lock:
            self.metrics_buffer.append(metric)
            
            # Keep buffer size manageable
            if len(self.metrics_buffer) > 10000:
                self.metrics_buffer = self.metrics_buffer[-5000:]  # Keep last 5000
        
        # Check alert rules
        self._check_alert_rules(metric)
    
    def record_latency_metric(self, operation: str, latency_ms: float, percentile: int):
        """Record latency metric for specific operation"""
        
        self.record_metric(
            name=f"feature_store_latency_p{percentile}",
            value=latency_ms,
            unit="ms",
            labels={
                "operation": operation,
                "environment": self.environment,
                "percentile": str(percentile)
            },
            description=f"P{percentile} latency for {operation} operation"
        )
    
    def record_throughput_metric(self, operation: str, requests_per_second: float):
        """Record throughput metric"""
        
        self.record_metric(
            name="feature_store_throughput",
            value=requests_per_second,
            unit="rps",
            labels={
                "operation": operation,
                "environment": self.environment
            },
            description=f"Throughput for {operation} operation"
        )
    
    def record_error_metric(self, operation: str, error_rate_percent: float, error_type: str = "general"):
        """Record error rate metric"""
        
        self.record_metric(
            name="feature_store_error_rate",
            value=error_rate_percent,
            unit="percent",
            labels={
                "operation": operation,
                "environment": self.environment,
                "error_type": error_type
            },
            description=f"Error rate for {operation} operation"
        )
    
    def record_cache_metric(self, hit_ratio_percent: float, cache_size_mb: float):
        """Record cache performance metrics"""
        
        self.record_metric(
            name="feature_store_cache_hit_ratio",
            value=hit_ratio_percent,
            unit="percent",
            labels={"environment": self.environment},
            description="Feature cache hit ratio"
        )
        
        self.record_metric(
            name="feature_store_cache_size",
            value=cache_size_mb,
            unit="mb",
            labels={"environment": self.environment},
            description="Feature cache size in MB"
        )
    
    def record_feature_freshness_metric(self, component: str, freshness_seconds: float):
        """Record feature freshness metric"""
        
        self.record_metric(
            name="feature_store_feature_freshness",
            value=freshness_seconds,
            unit="seconds",
            labels={
                "component": component,
                "environment": self.environment
            },
            description=f"Feature freshness for {component}"
        )
    
    def _check_alert_rules(self, metric: MonitoringMetric):
        """Check if metric triggers any alert rules"""
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Match metric type
            if not self._metric_matches_rule(metric, rule):
                continue
            
            # Check threshold
            triggered = self._evaluate_threshold(metric.value, rule.threshold_value, rule.comparison_operator)
            
            if triggered:
                self._trigger_alert(rule, metric)
    
    def _metric_matches_rule(self, metric: MonitoringMetric, rule: AlertRule) -> bool:
        """Check if metric matches alert rule criteria"""
        
        metric_type_mapping = {
            MetricType.LATENCY: "latency",
            MetricType.THROUGHPUT: "throughput",
            MetricType.ERROR_RATE: "error_rate",
            MetricType.CACHE_PERFORMANCE: "cache",
            MetricType.FEATURE_FRESHNESS: "freshness",
            MetricType.COST: "cost"
        }
        
        expected_prefix = metric_type_mapping.get(rule.metric_type, "")
        return expected_prefix in metric.name.lower()
    
    def _evaluate_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate if value meets threshold criteria"""
        
        if operator == "GREATER_THAN":
            return value > threshold
        elif operator == "LESS_THAN":
            return value < threshold
        elif operator == "EQUAL":
            return abs(value - threshold) < 0.001
        elif operator == "GREATER_THAN_OR_EQUAL":
            return value >= threshold
        elif operator == "LESS_THAN_OR_EQUAL":
            return value <= threshold
        else:
            return False
    
    def _trigger_alert(self, rule: AlertRule, metric: MonitoringMetric):
        """Trigger an alert"""
        
        alert_data = {
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "metric_name": metric.name,
            "metric_value": metric.value,
            "threshold": rule.threshold_value,
            "timestamp": metric.timestamp.isoformat(),
            "environment": self.environment,
            "description": rule.description
        }
        
        # Log alert
        self.logger.warning(f"ALERT TRIGGERED: {rule.name} - {rule.description}")
        self.logger.warning(f"Metric: {metric.name}={metric.value}, Threshold: {rule.threshold_value}")
        
        # Send notifications (mock implementation)
        for channel in rule.notification_channels:
            self._send_notification(channel, alert_data)
    
    def _send_notification(self, channel: str, alert_data: Dict[str, Any]):
        """Send alert notification through specified channel"""
        
        if channel == "email":
            self.logger.info(f"📧 Email notification sent: {alert_data['rule_name']}")
        elif channel == "slack":
            self.logger.info(f"💬 Slack notification sent: {alert_data['rule_name']}")
        elif channel == "webhook":
            self.logger.info(f"🔗 Webhook notification sent: {alert_data['rule_name']}")
    
    async def collect_cost_metrics(self) -> CostMetrics:
        """Collect and calculate cost metrics"""
        
        # Get current month date range
        now = datetime.utcnow()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Query BigQuery costs (simplified)
        bigquery_cost = await self._get_bigquery_costs(start_of_month, now)
        
        # Estimate Feature Store costs
        feature_store_cost = await self._estimate_feature_store_costs()
        
        # Estimate compute costs
        compute_cost = await self._estimate_compute_costs()
        
        # Calculate totals
        total_cost = bigquery_cost + feature_store_cost + compute_cost
        
        # Estimate cost per request (if we have request metrics)
        total_requests = self._get_total_requests_this_month()
        cost_per_request = total_cost / max(total_requests, 1)
        
        # Calculate budget utilization (assuming $1000 monthly budget)
        monthly_budget = 1000.0  # USD
        budget_utilization = (total_cost / monthly_budget) * 100
        
        cost_metrics = CostMetrics(
            feature_store_cost_usd=feature_store_cost,
            bigquery_cost_usd=bigquery_cost,
            compute_cost_usd=compute_cost,
            total_monthly_cost_usd=total_cost,
            cost_per_request_usd=cost_per_request,
            budget_utilization_percent=budget_utilization
        )
        
        # Store cost history
        self.cost_history.append(cost_metrics)
        
        # Record cost metrics for alerting
        self.record_metric(
            name="feature_store_monthly_cost",
            value=total_cost,
            unit="usd",
            labels={"environment": self.environment},
            description="Total monthly Feature Store cost"
        )
        
        self.record_metric(
            name="feature_store_budget_utilization",
            value=budget_utilization,
            unit="percent",
            labels={"environment": self.environment},
            description="Monthly budget utilization percentage"
        )
        
        return cost_metrics
    
    async def _get_bigquery_costs(self, start_date: datetime, end_date: datetime) -> float:
        """Get BigQuery costs for date range"""
        
        # Mock implementation - in real scenario, query Cloud Billing API
        # or use BigQuery INFORMATION_SCHEMA.JOBS_BY_PROJECT
        
        base_daily_cost = 5.0  # $5 per day base cost
        days = (end_date - start_date).days
        return base_daily_cost * days
    
    async def _estimate_feature_store_costs(self) -> float:
        """Estimate Feature Store operational costs"""
        
        # Mock implementation - in real scenario, calculate based on:
        # - Online serving requests
        # - Storage usage 
        # - Network egress
        
        base_monthly_cost = 50.0  # Base $50/month for Feature Store
        return base_monthly_cost
    
    async def _estimate_compute_costs(self) -> float:
        """Estimate compute costs for Feature Store operations"""
        
        # Mock implementation - in real scenario, get from Cloud Billing API
        
        base_monthly_compute = 25.0  # $25/month for compute
        return base_monthly_compute
    
    def _get_total_requests_this_month(self) -> int:
        """Get total requests for current month"""
        
        # Mock implementation - count from metrics buffer
        with self.metrics_lock:
            throughput_metrics = [m for m in self.metrics_buffer if "throughput" in m.name]
            
            if not throughput_metrics:
                return 1000  # Default estimate
            
            # Simple estimation based on average throughput
            avg_rps = np.mean([m.value for m in throughput_metrics])
            days_in_month = 30
            estimated_requests = int(avg_rps * 86400 * days_in_month)  # RPS * seconds/day * days
            
            return max(estimated_requests, 1)
    
    def create_monitoring_dashboard(self) -> Dict[str, Any]:
        """Create monitoring dashboard configuration"""
        
        dashboard_config = {
            "displayName": f"Feature Store Monitoring - {self.environment.upper()}",
            "mosaicLayout": {
                "tiles": [
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Feature Store Latency",
                            "xyChart": {
                                "dataSets": [{
                                    "timeSeriesQuery": {
                                        "timeSeriesFilter": {
                                            "filter": f'resource.type="gce_instance" AND metric.type="custom.googleapis.com/feature_store_latency_p99"',
                                            "aggregation": {
                                                "alignmentPeriod": "60s",
                                                "perSeriesAligner": "ALIGN_MEAN"
                                            }
                                        }
                                    },
                                    "plotType": "LINE"
                                }],
                                "yAxis": {
                                    "label": "Latency (ms)",
                                    "scale": "LINEAR"
                                }
                            }
                        }
                    },
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Error Rate",
                            "xyChart": {
                                "dataSets": [{
                                    "timeSeriesQuery": {
                                        "timeSeriesFilter": {
                                            "filter": f'resource.type="gce_instance" AND metric.type="custom.googleapis.com/feature_store_error_rate"',
                                            "aggregation": {
                                                "alignmentPeriod": "60s",
                                                "perSeriesAligner": "ALIGN_MEAN"
                                            }
                                        }
                                    },
                                    "plotType": "STACKED_AREA"
                                }],
                                "yAxis": {
                                    "label": "Error Rate (%)",
                                    "scale": "LINEAR"
                                }
                            }
                        }
                    },
                    {
                        "width": 4,
                        "height": 3,
                        "widget": {
                            "title": "Cache Hit Ratio",
                            "scorecard": {
                                "timeSeriesQuery": {
                                    "timeSeriesFilter": {
                                        "filter": f'resource.type="gce_instance" AND metric.type="custom.googleapis.com/feature_store_cache_hit_ratio"',
                                        "aggregation": {
                                            "alignmentPeriod": "300s",
                                            "perSeriesAligner": "ALIGN_MEAN"
                                        }
                                    }
                                },
                                "sparkChartView": {
                                    "sparkChartType": "SPARK_LINE"
                                }
                            }
                        }
                    },
                    {
                        "width": 8,
                        "height": 4,
                        "widget": {
                            "title": "Monthly Cost Trend",
                            "xyChart": {
                                "dataSets": [{
                                    "timeSeriesQuery": {
                                        "timeSeriesFilter": {
                                            "filter": f'resource.type="gce_instance" AND metric.type="custom.googleapis.com/feature_store_monthly_cost"',
                                            "aggregation": {
                                                "alignmentPeriod": "3600s",
                                                "perSeriesAligner": "ALIGN_MEAN"
                                            }
                                        }
                                    },
                                    "plotType": "LINE"
                                }],
                                "yAxis": {
                                    "label": "Cost (USD)",
                                    "scale": "LINEAR"
                                }
                            }
                        }
                    }
                ]
            }
        }
        
        return dashboard_config
    
    def _start_background_monitoring(self):
        """Start background monitoring thread"""
        
        def monitoring_loop():
            """Background monitoring loop"""
            while self.monitoring_active:
                try:
                    # Collect system metrics every minute
                    asyncio.run(self._collect_system_metrics())
                    time.sleep(60)
                    
                except Exception as e:
                    self.logger.error(f"Background monitoring error: {e}")
                    time.sleep(30)  # Shorter retry on error
        
        # Start background thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        self.logger.info("Background monitoring started")
    
    async def _collect_system_metrics(self):
        """Collect system metrics in background"""
        
        # Collect cost metrics (every 10 minutes)
        if int(time.time()) % 600 == 0:  # Every 10 minutes
            try:
                cost_metrics = await self.collect_cost_metrics()
                self.logger.info(f"Cost metrics collected: ${cost_metrics.total_monthly_cost_usd:.2f} monthly")
            except Exception as e:
                self.logger.error(f"Failed to collect cost metrics: {e}")
        
        # Record synthetic metrics for demonstration
        self.record_latency_metric("online_serving", np.random.normal(30, 10), 99)
        self.record_throughput_metric("online_serving", np.random.normal(50, 15))
        self.record_cache_metric(np.random.normal(75, 10), 512)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        
        with self.metrics_lock:
            recent_metrics = [m for m in self.metrics_buffer if 
                            (datetime.utcnow() - m.timestamp).total_seconds() < 3600]  # Last hour
        
        # Calculate summaries
        latency_metrics = [m for m in recent_metrics if "latency" in m.name]
        error_metrics = [m for m in recent_metrics if "error" in m.name]
        cache_metrics = [m for m in recent_metrics if "cache" in m.name]
        
        summary = {
            "monitoring_status": {
                "active": self.monitoring_active,
                "total_metrics_collected": len(self.metrics_buffer),
                "recent_metrics_hour": len(recent_metrics),
                "alert_rules_configured": len(self.alert_rules),
                "cost_history_entries": len(self.cost_history)
            },
            "performance_summary": {
                "avg_latency_p99_ms": np.mean([m.value for m in latency_metrics]) if latency_metrics else 0,
                "avg_error_rate_percent": np.mean([m.value for m in error_metrics]) if error_metrics else 0,
                "avg_cache_hit_ratio": np.mean([m.value for m in cache_metrics]) if cache_metrics else 0
            },
            "cost_summary": {
                "latest_monthly_cost_usd": self.cost_history[-1].total_monthly_cost_usd if self.cost_history else 0,
                "budget_utilization_percent": self.cost_history[-1].budget_utilization_percent if self.cost_history else 0
            },
            "sla_compliance": {
                "latency_sla_met": np.mean([m.value for m in latency_metrics]) < 50 if latency_metrics else True,
                "error_rate_sla_met": np.mean([m.value for m in error_metrics]) < 1.0 if error_metrics else True,
                "overall_health": "healthy"  # Simplified
            }
        }
        
        return summary
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        self.logger.info("Monitoring stopped")


# Example usage
async def main():
    """Example usage of Feature Store Monitoring"""
    
    # Initialize monitoring
    monitoring = FeatureStoreMonitoring(environment="dev")
    
    # Simulate some metrics
    print("Recording sample metrics...")
    monitoring.record_latency_metric("online_serving", 35.5, 99)
    monitoring.record_throughput_metric("online_serving", 125.0)
    monitoring.record_error_metric("online_serving", 0.5)
    monitoring.record_cache_metric(78.5, 1024)
    
    # Collect cost metrics
    print("Collecting cost metrics...")
    cost_metrics = await monitoring.collect_cost_metrics()
    print(f"Monthly cost: ${cost_metrics.total_monthly_cost_usd:.2f}")
    print(f"Budget utilization: {cost_metrics.budget_utilization_percent:.1f}%")
    
    # Get monitoring summary
    summary = monitoring.get_monitoring_summary()
    print(f"\nMonitoring Summary:")
    print(f"Active: {summary['monitoring_status']['active']}")
    print(f"Metrics collected: {summary['monitoring_status']['total_metrics_collected']}")
    print(f"Avg P99 latency: {summary['performance_summary']['avg_latency_p99_ms']:.1f}ms")
    
    # Wait a bit for background monitoring
    await asyncio.sleep(2)
    
    # Stop monitoring
    monitoring.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())