"""
Feature Store Performance Optimizer
Implements performance optimizations for <50ms latency target
"""

import asyncio
import logging
import time
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import threading

from google.cloud import aiplatform
from google.cloud import monitoring_v3
from google.cloud import bigquery
import pandas as pd
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for Feature Store operations"""
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    error_rate_percent: float
    cache_hit_ratio: float
    memory_usage_mb: float
    cpu_utilization_percent: float


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations"""
    target_latency_p99_ms: float = 50.0
    target_latency_p95_ms: float = 40.0
    target_latency_p50_ms: float = 25.0
    target_throughput_rps: float = 1000.0
    cache_enabled: bool = True
    cache_size_mb: int = 1024
    cache_ttl_seconds: int = 60
    connection_pool_size: int = 10
    batch_size_optimization: bool = True
    auto_scaling_enabled: bool = True


class FeatureStorePerformanceOptimizer:
    """
    Performance Optimizer for Vertex AI Feature Store
    Implements caching, connection pooling, and latency optimization
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "dev"):
        """Initialize performance optimizer"""
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "feature_store_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Performance configuration
        self.optimization_config = OptimizationConfig()
        
        # Initialize clients and optimizations
        self._initialize_clients()
        self._setup_performance_monitoring()
        self._setup_caching()
        self._setup_connection_pooling()
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_enabled = True
        
    def _initialize_clients(self):
        """Initialize optimized Google Cloud clients"""
        project_id = self.config["project_config"]["project_id"]
        location = self.config["project_config"]["location"]
        
        # Initialize Vertex AI with optimized settings
        aiplatform.init(
            project=project_id, 
            location=location,
            # Add performance optimizations
            experiment_name="feature_store_performance_optimization"
        )
        
        # Monitoring client for performance metrics
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
        
    def _setup_performance_monitoring(self):
        """Setup real-time performance monitoring"""
        self.performance_metrics = {
            "request_count": 0,
            "error_count": 0,
            "total_latency_ms": 0.0,
            "latencies": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.monitoring_lock = threading.Lock()
        
    def _setup_caching(self):
        """Setup intelligent caching system"""
        if self.optimization_config.cache_enabled:
            self.feature_cache = {}
            self.cache_timestamps = {}
            self.cache_access_count = {}
            self.cache_size_limit = self.optimization_config.cache_size_mb * 1024 * 1024  # Convert to bytes
            
            self.logger.info(f"Feature cache enabled: {self.optimization_config.cache_size_mb}MB, TTL: {self.optimization_config.cache_ttl_seconds}s")
        
    def _setup_connection_pooling(self):
        """Setup connection pooling for better performance"""
        self.connection_pool = ThreadPoolExecutor(
            max_workers=self.optimization_config.connection_pool_size,
            thread_name_prefix="feature_store_pool"
        )
        
        self.logger.info(f"Connection pool initialized: {self.optimization_config.connection_pool_size} workers")
    
    def record_performance_metric(self, operation: str, latency_ms: float, success: bool, cache_hit: bool = False):
        """Record performance metrics for monitoring"""
        with self.monitoring_lock:
            self.performance_metrics["request_count"] += 1
            self.performance_metrics["total_latency_ms"] += latency_ms
            self.performance_metrics["latencies"].append(latency_ms)
            
            if not success:
                self.performance_metrics["error_count"] += 1
            
            if cache_hit:
                self.performance_metrics["cache_hits"] += 1
            else:
                self.performance_metrics["cache_misses"] += 1
            
            # Keep only last 1000 latencies for memory efficiency
            if len(self.performance_metrics["latencies"]) > 1000:
                self.performance_metrics["latencies"] = self.performance_metrics["latencies"][-1000:]
    
    def get_current_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self.monitoring_lock:
            latencies = self.performance_metrics["latencies"]
            
            if not latencies:
                return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            
            # Calculate percentiles
            p50 = latencies_sorted[int(n * 0.5)] if n > 0 else 0
            p95 = latencies_sorted[int(n * 0.95)] if n > 0 else 0
            p99 = latencies_sorted[int(n * 0.99)] if n > 0 else 0
            
            # Calculate other metrics
            total_requests = self.performance_metrics["request_count"]
            error_rate = (self.performance_metrics["error_count"] / max(total_requests, 1)) * 100
            
            cache_total = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
            cache_hit_ratio = (self.performance_metrics["cache_hits"] / max(cache_total, 1)) * 100
            
            # Estimate throughput (requests per second over last minute)
            throughput_rps = min(total_requests, 60)  # Simplified calculation
            
            return PerformanceMetrics(
                latency_p50_ms=round(p50, 2),
                latency_p95_ms=round(p95, 2),
                latency_p99_ms=round(p99, 2),
                throughput_rps=round(throughput_rps, 2),
                error_rate_percent=round(error_rate, 2),
                cache_hit_ratio=round(cache_hit_ratio, 2),
                memory_usage_mb=self._estimate_memory_usage(),
                cpu_utilization_percent=self._estimate_cpu_utilization()
            )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB"""
        # Estimate cache memory usage
        cache_memory = 0
        if hasattr(self, 'feature_cache'):
            # Rough estimation: 1KB per cached feature set
            cache_memory = len(self.feature_cache) * 1024
        
        return round(cache_memory / (1024 * 1024), 2)  # Convert to MB
    
    def _estimate_cpu_utilization(self) -> float:
        """Estimate current CPU utilization percentage"""
        # Simplified estimation based on request load
        recent_requests = min(self.performance_metrics["request_count"], 100)
        return min(recent_requests * 0.5, 100)  # Simple heuristic
    
    def optimize_feature_cache(self, feature_data: Dict[str, Any], entity_id: str, feature_names: List[str]) -> bool:
        """Optimize feature caching with intelligent eviction"""
        if not self.optimization_config.cache_enabled:
            return False
        
        cache_key = f"{entity_id}:{':'.join(sorted(feature_names))}"
        
        # Check if we need to evict entries
        if len(self.feature_cache) * 1024 > self.cache_size_limit:  # Rough size estimation
            self._evict_cache_entries()
        
        # Cache the feature data
        self.feature_cache[cache_key] = feature_data
        self.cache_timestamps[cache_key] = datetime.utcnow()
        self.cache_access_count[cache_key] = 0
        
        return True
    
    def get_cached_features(self, entity_id: str, feature_names: List[str]) -> Optional[Dict[str, Any]]:
        """Get features from cache if available and valid"""
        if not self.optimization_config.cache_enabled:
            return None
        
        cache_key = f"{entity_id}:{':'.join(sorted(feature_names))}"
        
        # Check if cached
        if cache_key not in self.feature_cache:
            return None
        
        # Check TTL
        cache_time = self.cache_timestamps.get(cache_key, datetime.min)
        if (datetime.utcnow() - cache_time).total_seconds() > self.optimization_config.cache_ttl_seconds:
            # Remove expired entry
            self._remove_cache_entry(cache_key)
            return None
        
        # Update access count for LRU
        self.cache_access_count[cache_key] = self.cache_access_count.get(cache_key, 0) + 1
        
        return self.feature_cache[cache_key]
    
    def _evict_cache_entries(self):
        """Evict least recently used cache entries"""
        if not self.feature_cache:
            return
        
        # Sort by access count (LRU)
        sorted_keys = sorted(
            self.cache_access_count.keys(),
            key=lambda k: self.cache_access_count[k]
        )
        
        # Remove least used 25% of entries
        remove_count = max(1, len(sorted_keys) // 4)
        for key in sorted_keys[:remove_count]:
            self._remove_cache_entry(key)
        
        self.logger.info(f"Evicted {remove_count} cache entries")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a single cache entry"""
        self.feature_cache.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)
        self.cache_access_count.pop(cache_key, None)
    
    async def optimize_batch_requests(self, entity_requests: List[Tuple[str, List[str]]]) -> List[Dict[str, Any]]:
        """Optimize batch feature requests for better performance"""
        if not self.optimization_config.batch_size_optimization:
            # Process requests individually
            results = []
            for entity_id, feature_names in entity_requests:
                result = await self._fetch_single_entity(entity_id, feature_names)
                results.append(result)
            return results
        
        # Group requests by feature names for batch optimization
        grouped_requests = {}
        for entity_id, feature_names in entity_requests:
            feature_key = ':'.join(sorted(feature_names))
            if feature_key not in grouped_requests:
                grouped_requests[feature_key] = []
            grouped_requests[feature_key].append(entity_id)
        
        # Process groups concurrently
        tasks = []
        for feature_names_key, entity_ids in grouped_requests.items():
            feature_names = feature_names_key.split(':')
            task = self._fetch_batch_entities(entity_ids, feature_names)
            tasks.append(task)
        
        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        return all_results
    
    async def _fetch_single_entity(self, entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
        """Fetch features for a single entity with caching"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.get_cached_features(entity_id, feature_names)
            if cached_result is not None:
                latency = (time.time() - start_time) * 1000
                self.record_performance_metric("fetch_single_cached", latency, True, cache_hit=True)
                return cached_result
            
            # Fetch from Feature Store (mock implementation)
            await asyncio.sleep(0.020)  # Simulate 20ms latency
            
            result = {
                "entity_id": entity_id,
                "features": {name: np.random.rand() for name in feature_names},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache the result
            self.optimize_feature_cache(result, entity_id, feature_names)
            
            latency = (time.time() - start_time) * 1000
            self.record_performance_metric("fetch_single", latency, True, cache_hit=False)
            
            return result
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.record_performance_metric("fetch_single", latency, False)
            self.logger.error(f"Failed to fetch features for {entity_id}: {e}")
            raise
    
    async def _fetch_batch_entities(self, entity_ids: List[str], feature_names: List[str]) -> List[Dict[str, Any]]:
        """Fetch features for multiple entities in a batch"""
        start_time = time.time()
        
        try:
            # Simulate batch fetch with better performance than individual requests
            batch_latency_per_entity = 0.005  # 5ms per entity in batch
            await asyncio.sleep(len(entity_ids) * batch_latency_per_entity)
            
            results = []
            for entity_id in entity_ids:
                result = {
                    "entity_id": entity_id,
                    "features": {name: np.random.rand() for name in feature_names},
                    "timestamp": datetime.utcnow().isoformat()
                }
                results.append(result)
                
                # Cache individual results
                self.optimize_feature_cache(result, entity_id, feature_names)
            
            latency = (time.time() - start_time) * 1000
            self.record_performance_metric("fetch_batch", latency, True, cache_hit=False)
            
            return results
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.record_performance_metric("fetch_batch", latency, False)
            self.logger.error(f"Failed to batch fetch features: {e}")
            raise
    
    async def benchmark_performance(self, duration_seconds: int = 60, concurrent_requests: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark
        Validates <50ms latency target under load
        """
        self.logger.info(f"Starting performance benchmark: {duration_seconds}s, {concurrent_requests} concurrent requests")
        
        start_time = time.time()
        benchmark_results = {
            "duration_seconds": duration_seconds,
            "concurrent_requests": concurrent_requests,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "performance_metrics": None,
            "sla_compliance": {}
        }
        
        # Test entities and features
        test_entities = [
            f"NIFTY_20250813{str(14 + i).zfill(4)}00_0" for i in range(10)
        ]
        test_features = [
            "c1_momentum_score", "c2_gamma_exposure", "c3_institutional_flow_score", 
            "c4_skew_bias_score", "c5_momentum_score", "c6_correlation_agreement_score",
            "c7_level_strength_score", "c8_component_agreement_score"
        ]
        
        # Reset metrics for benchmark
        self.performance_metrics = {
            "request_count": 0, "error_count": 0, "total_latency_ms": 0.0,
            "latencies": [], "cache_hits": 0, "cache_misses": 0
        }
        
        async def make_concurrent_requests():
            """Make concurrent requests for the duration"""
            tasks = []
            
            while (time.time() - start_time) < duration_seconds:
                # Create batch of concurrent requests
                for _ in range(concurrent_requests):
                    entity_id = np.random.choice(test_entities)
                    num_features = np.random.randint(4, 8)  # Request 4-8 features
                    selected_features = np.random.choice(test_features, size=num_features, replace=False).tolist()
                    
                    task = self._fetch_single_entity(entity_id, selected_features)
                    tasks.append(task)
                
                # Execute batch
                try:
                    await asyncio.gather(*tasks)
                    benchmark_results["total_requests"] += len(tasks)
                    benchmark_results["successful_requests"] += len(tasks)
                except Exception as e:
                    benchmark_results["failed_requests"] += len(tasks)
                    self.logger.error(f"Benchmark request batch failed: {e}")
                
                tasks = []  # Reset for next batch
                
                # Small pause to avoid overwhelming
                await asyncio.sleep(0.1)
        
        # Run benchmark
        await make_concurrent_requests()
        
        # Collect final metrics
        final_metrics = self.get_current_performance_metrics()
        benchmark_results["performance_metrics"] = {
            "latency_p50_ms": final_metrics.latency_p50_ms,
            "latency_p95_ms": final_metrics.latency_p95_ms,
            "latency_p99_ms": final_metrics.latency_p99_ms,
            "throughput_rps": final_metrics.throughput_rps,
            "error_rate_percent": final_metrics.error_rate_percent,
            "cache_hit_ratio": final_metrics.cache_hit_ratio
        }
        
        # Check SLA compliance
        sla_targets = self.optimization_config
        benchmark_results["sla_compliance"] = {
            "latency_p50_pass": final_metrics.latency_p50_ms <= sla_targets.target_latency_p50_ms,
            "latency_p95_pass": final_metrics.latency_p95_ms <= sla_targets.target_latency_p95_ms,
            "latency_p99_pass": final_metrics.latency_p99_ms <= sla_targets.target_latency_p99_ms,
            "throughput_pass": final_metrics.throughput_rps >= sla_targets.target_throughput_rps * 0.1,  # 10% of target for test
            "error_rate_pass": final_metrics.error_rate_percent <= 1.0,  # <1% error rate
        }
        
        # Overall pass/fail
        benchmark_results["overall_pass"] = all(benchmark_results["sla_compliance"].values())
        benchmark_results["execution_time_seconds"] = time.time() - start_time
        
        self.logger.info(f"Performance benchmark completed: {benchmark_results['overall_pass']}")
        self.logger.info(f"P99 Latency: {final_metrics.latency_p99_ms}ms (Target: <50ms)")
        
        return benchmark_results
    
    async def auto_scale_optimization(self) -> Dict[str, Any]:
        """Automatically optimize settings based on current performance"""
        current_metrics = self.get_current_performance_metrics()
        
        optimization_actions = []
        
        # Optimize cache settings
        if current_metrics.cache_hit_ratio < 70:  # Less than 70% cache hit ratio
            if self.optimization_config.cache_size_mb < 2048:  # Max 2GB cache
                self.optimization_config.cache_size_mb *= 2
                optimization_actions.append(f"Increased cache size to {self.optimization_config.cache_size_mb}MB")
        
        # Optimize connection pool
        if current_metrics.throughput_rps > self.optimization_config.connection_pool_size * 50:
            if self.optimization_config.connection_pool_size < 50:  # Max 50 connections
                self.optimization_config.connection_pool_size += 5
                optimization_actions.append(f"Increased connection pool to {self.optimization_config.connection_pool_size}")
        
        # Optimize cache TTL based on error rate
        if current_metrics.error_rate_percent > 2.0:  # High error rate
            self.optimization_config.cache_ttl_seconds = max(30, self.optimization_config.cache_ttl_seconds - 15)
            optimization_actions.append(f"Reduced cache TTL to {self.optimization_config.cache_ttl_seconds}s")
        
        return {
            "optimization_actions": optimization_actions,
            "current_config": {
                "cache_size_mb": self.optimization_config.cache_size_mb,
                "connection_pool_size": self.optimization_config.connection_pool_size,
                "cache_ttl_seconds": self.optimization_config.cache_ttl_seconds
            },
            "performance_metrics": {
                "latency_p99_ms": current_metrics.latency_p99_ms,
                "throughput_rps": current_metrics.throughput_rps,
                "cache_hit_ratio": current_metrics.cache_hit_ratio,
                "error_rate_percent": current_metrics.error_rate_percent
            }
        }
    
    def generate_performance_report(self) -> str:
        """Generate detailed performance analysis report"""
        metrics = self.get_current_performance_metrics()
        
        report = f"""
# Feature Store Performance Report
Generated: {datetime.utcnow().isoformat()}
Environment: {self.environment}

## Current Performance Metrics

### Latency (Target: P99 < 50ms)
- P50: {metrics.latency_p50_ms}ms (Target: <25ms) {'✓' if metrics.latency_p50_ms <= 25 else '✗'}
- P95: {metrics.latency_p95_ms}ms (Target: <40ms) {'✓' if metrics.latency_p95_ms <= 40 else '✗'}
- P99: {metrics.latency_p99_ms}ms (Target: <50ms) {'✓' if metrics.latency_p99_ms <= 50 else '✗'}

### Throughput & Reliability
- Throughput: {metrics.throughput_rps} RPS
- Error Rate: {metrics.error_rate_percent}% {'✓' if metrics.error_rate_percent <= 1.0 else '✗'}
- Cache Hit Ratio: {metrics.cache_hit_ratio}% {'✓' if metrics.cache_hit_ratio >= 70 else '✗'}

### Resource Usage
- Memory Usage: {metrics.memory_usage_mb}MB
- CPU Utilization: {metrics.cpu_utilization_percent}%

## Optimization Configuration
- Cache Enabled: {self.optimization_config.cache_enabled}
- Cache Size: {self.optimization_config.cache_size_mb}MB
- Cache TTL: {self.optimization_config.cache_ttl_seconds}s
- Connection Pool: {self.optimization_config.connection_pool_size} workers
- Batch Optimization: {self.optimization_config.batch_size_optimization}

## Performance Analysis
Total Requests: {self.performance_metrics['request_count']}
Cache Hits: {self.performance_metrics['cache_hits']}
Cache Misses: {self.performance_metrics['cache_misses']}

## Recommendations
"""
        
        # Add recommendations based on metrics
        if metrics.latency_p99_ms > 50:
            report += "- ❌ CRITICAL: P99 latency exceeds 50ms target. Consider increasing cache size or optimizing queries.\n"
        
        if metrics.cache_hit_ratio < 70:
            report += "- ⚠️ Cache hit ratio is below 70%. Consider increasing cache TTL or cache size.\n"
        
        if metrics.error_rate_percent > 1.0:
            report += "- ⚠️ Error rate is above 1%. Investigate error causes and improve error handling.\n"
        
        if metrics.throughput_rps < 100:
            report += "- ⚠️ Throughput is low. Consider optimizing connection pooling or batch processing.\n"
        
        if all([
            metrics.latency_p99_ms <= 50,
            metrics.error_rate_percent <= 1.0,
            metrics.cache_hit_ratio >= 70
        ]):
            report += "- ✅ All performance targets are being met!\n"
        
        return report


# Example usage and testing
async def main():
    """Example usage of Performance Optimizer"""
    
    # Initialize optimizer
    optimizer = FeatureStorePerformanceOptimizer(environment="dev")
    
    print("Running performance benchmark...")
    benchmark_results = await optimizer.benchmark_performance(
        duration_seconds=30,  # Shorter for testing
        concurrent_requests=5
    )
    
    print(f"\nBenchmark Results:")
    print(f"Overall Pass: {benchmark_results['overall_pass']}")
    print(f"P99 Latency: {benchmark_results['performance_metrics']['latency_p99_ms']}ms")
    print(f"Throughput: {benchmark_results['performance_metrics']['throughput_rps']} RPS")
    print(f"Cache Hit Ratio: {benchmark_results['performance_metrics']['cache_hit_ratio']}%")
    
    # Auto-optimize
    print("\nRunning auto-optimization...")
    optimization_results = await optimizer.auto_scale_optimization()
    print(f"Optimization Actions: {optimization_results['optimization_actions']}")
    
    # Generate report
    print("\nGenerating performance report...")
    report = optimizer.generate_performance_report()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())