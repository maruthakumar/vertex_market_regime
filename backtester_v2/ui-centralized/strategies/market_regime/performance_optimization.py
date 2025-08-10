#!/usr/bin/env python3
"""
Performance Optimization Module for Market Regime Triple Straddle Engine
Phase 1 Implementation: Memory Optimization, Intelligent Caching, and Parallel Processing

This module implements the performance optimization enhancements specified in the 
Market Regime Gaps Implementation V1.0 document:

1. Memory Usage Optimization Strategy
2. Intelligent Caching Strategy  
3. Parallel Processing Architecture

Target Performance Improvements:
- Memory usage <4GB for 50+ concurrent users
- <1 second processing time (enhanced from <3s)
- <10% memory growth over 24-hour operation
- <100ms garbage collection pauses

Author: Senior Quantitative Trading Expert
Date: June 2025
Version: 1.0 - Phase 1 Performance Optimization
"""

import logging
import psutil
import gc
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import deque, OrderedDict
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import weakref
import mmap
import pickle
import json
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class MemoryPool:
    """Memory pool management for object reuse and garbage collection optimization"""
    
    def __init__(self, max_size_gb: float = 4.0):
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.allocated_objects = weakref.WeakValueDictionary()
        self.object_pools = {
            'dataframes': deque(maxlen=100),
            'arrays': deque(maxlen=200),
            'calculations': deque(maxlen=150)
        }
        self.allocation_stats = {
            'total_allocated': 0,
            'peak_usage': 0,
            'gc_triggers': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        self._lock = threading.Lock()
    
    def get_object(self, object_type: str, size_hint: Optional[int] = None):
        """Get object from pool or create new one"""
        with self._lock:
            pool = self.object_pools.get(object_type, deque())
            
            if pool:
                self.allocation_stats['pool_hits'] += 1
                return pool.popleft()
            else:
                self.allocation_stats['pool_misses'] += 1
                return self._create_new_object(object_type, size_hint)
    
    def return_object(self, obj, object_type: str):
        """Return object to pool for reuse"""
        with self._lock:
            pool = self.object_pools.get(object_type, deque())
            if len(pool) < pool.maxlen:
                # Clear object data before returning to pool
                if hasattr(obj, 'clear'):
                    obj.clear()
                pool.append(obj)
    
    def _create_new_object(self, object_type: str, size_hint: Optional[int] = None):
        """Create new object based on type"""
        if object_type == 'dataframes':
            return pd.DataFrame()
        elif object_type == 'arrays':
            size = size_hint or 1000
            return np.zeros(size, dtype=np.float32)
        elif object_type == 'calculations':
            return {}
        else:
            return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }

class MemoryMonitor:
    """Real-time memory monitoring with automatic cleanup triggers"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring_active = False
        self.cleanup_callbacks = []
        self.memory_history = deque(maxlen=1000)
        self._monitor_thread = None
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous memory monitoring"""
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                memory_usage = self._get_memory_usage_percent()
                self.memory_history.append({
                    'timestamp': datetime.now(),
                    'usage_percent': memory_usage
                })
                
                if memory_usage > self.critical_threshold:
                    logger.warning(f"CRITICAL: Memory usage {memory_usage:.1%} > {self.critical_threshold:.1%}")
                    self._trigger_emergency_cleanup()
                elif memory_usage > self.warning_threshold:
                    logger.warning(f"WARNING: Memory usage {memory_usage:.1%} > {self.warning_threshold:.1%}")
                    self._trigger_cleanup()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)
    
    def _get_memory_usage_percent(self) -> float:
        """Get current memory usage as percentage"""
        return psutil.Process().memory_percent() / 100.0
    
    def _trigger_cleanup(self):
        """Trigger normal cleanup procedures"""
        gc.collect()
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency cleanup procedures"""
        logger.warning("Triggering emergency memory cleanup")
        gc.collect()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in emergency cleanup callback: {e}")
    
    def register_cleanup_callback(self, callback):
        """Register callback function for cleanup events"""
        self.cleanup_callbacks.append(callback)

class LRUCache:
    """Custom LRU Cache with size and TTL management"""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self._lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def get(self, key: str, default=None):
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                # Check TTL if configured
                if self.ttl_seconds and self._is_expired(key):
                    del self.cache[key]
                    del self.timestamps[key]
                    self.stats['misses'] += 1
                    return default
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return self.cache[key]
            
            self.stats['misses'] += 1
            return default
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    if oldest_key in self.timestamps:
                        del self.timestamps[oldest_key]
                    self.stats['evictions'] += 1
            
            self.cache[key] = value
            if self.ttl_seconds:
                self.timestamps[key] = time.time()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if not self.ttl_seconds or key not in self.timestamps:
            return False
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'size': len(self.cache),
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }

class TimeBasedCache:
    """Time-based cache with automatic expiration"""
    
    def __init__(self, ttl: int = 300):  # 5 minutes default TTL
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self._lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0, 'expirations': 0}
    
    def get(self, key: str, default=None):
        """Get item from cache with TTL check"""
        with self._lock:
            if key in self.cache:
                if self._is_expired(key):
                    del self.cache[key]
                    del self.timestamps[key]
                    self.stats['expirations'] += 1
                    self.stats['misses'] += 1
                    return default
                
                self.stats['hits'] += 1
                return self.cache[key]
            
            self.stats['misses'] += 1
            return default
    
    def put(self, key: str, value: Any):
        """Put item in cache with timestamp"""
        with self._lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def cleanup_expired(self):
        """Remove all expired entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                del self.timestamps[key]
                self.stats['expirations'] += 1
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                **self.stats,
                'size': len(self.cache),
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }

class IntelligentCacheManager:
    """Multi-tier intelligent caching system with ML-based cache prediction"""

    def __init__(self):
        # Multi-tier caching system
        self.l1_cache = LRUCache(maxsize=512)  # Hot data - 512 entries
        self.l2_cache = LRUCache(maxsize=2048, ttl_seconds=300)  # Warm data - 2048 entries, 5min TTL
        self.l3_cache = TimeBasedCache(ttl=1800)  # Cold data - 30min TTL

        # Cache intelligence
        self.access_patterns = deque(maxlen=10000)
        self.cache_predictor = None  # Placeholder for ML predictor
        self.invalidation_rules = {}

        # Performance metrics
        self.performance_metrics = {
            'l1_stats': {},
            'l2_stats': {},
            'l3_stats': {},
            'prediction_accuracy': 0.0,
            'total_cache_hits': 0,
            'total_cache_misses': 0
        }

    def get(self, key: str, cache_type: str = 'auto') -> Any:
        """Get item from appropriate cache tier"""
        # Record access pattern
        self.access_patterns.append({
            'key': key,
            'timestamp': time.time(),
            'cache_type': cache_type
        })

        # Try L1 cache first (hot data)
        value = self.l1_cache.get(key)
        if value is not None:
            self.performance_metrics['total_cache_hits'] += 1
            return value

        # Try L2 cache (warm data)
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1 if frequently accessed
            if self._should_promote_to_l1(key):
                self.l1_cache.put(key, value)
            self.performance_metrics['total_cache_hits'] += 1
            return value

        # Try L3 cache (cold data)
        value = self.l3_cache.get(key)
        if value is not None:
            self.performance_metrics['total_cache_hits'] += 1
            return value

        self.performance_metrics['total_cache_misses'] += 1
        return None

    def put(self, key: str, value: Any, cache_type: str = 'auto', priority: str = 'normal'):
        """Put item in appropriate cache tier based on priority and access patterns"""
        if priority == 'high' or cache_type == 'l1':
            self.l1_cache.put(key, value)
        elif priority == 'medium' or cache_type == 'l2':
            self.l2_cache.put(key, value)
        else:
            self.l3_cache.put(key, value)

    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if key should be promoted to L1 cache based on access patterns"""
        recent_accesses = [
            access for access in self.access_patterns
            if access['key'] == key and time.time() - access['timestamp'] < 300  # Last 5 minutes
        ]
        return len(recent_accesses) >= 3  # Promote if accessed 3+ times in 5 minutes

    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # This would implement pattern-based cache invalidation
        # For now, we'll implement a simple prefix match
        keys_to_remove = []

        # Check all cache tiers
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            if hasattr(cache, 'cache'):
                for key in list(cache.cache.keys()):
                    if key.startswith(pattern):
                        keys_to_remove.append((cache, key))

        # Remove matching keys
        for cache, key in keys_to_remove:
            if hasattr(cache, 'cache') and key in cache.cache:
                del cache.cache[key]
                if hasattr(cache, 'timestamps') and key in cache.timestamps:
                    del cache.timestamps[key]

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        self.performance_metrics['l1_stats'] = self.l1_cache.get_stats()
        self.performance_metrics['l2_stats'] = self.l2_cache.get_stats()
        self.performance_metrics['l3_stats'] = self.l3_cache.get_stats()

        total_hits = self.performance_metrics['total_cache_hits']
        total_misses = self.performance_metrics['total_cache_misses']
        total_requests = total_hits + total_misses

        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0

        return {
            **self.performance_metrics,
            'overall_hit_rate': overall_hit_rate,
            'total_requests': total_requests,
            'cache_efficiency': self._calculate_cache_efficiency()
        }

    def _calculate_cache_efficiency(self) -> float:
        """Calculate overall cache efficiency score"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()

        # Weighted efficiency based on cache tier performance
        l1_weight = 0.5  # L1 cache is most important
        l2_weight = 0.3  # L2 cache is moderately important
        l3_weight = 0.2  # L3 cache is least important

        efficiency = (
            l1_stats.get('hit_rate', 0) * l1_weight +
            l2_stats.get('hit_rate', 0) * l2_weight +
            l3_stats.get('hit_rate', 0) * l3_weight
        )

        return efficiency

class ComponentProcessor:
    """Individual component processor for parallel execution"""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.processing_stats = {
            'total_calculations': 0,
            'average_processing_time': 0.0,
            'last_processing_time': 0.0,
            'error_count': 0
        }

    def calculate_independent_analysis(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate independent technical analysis for component"""
        start_time = time.time()

        try:
            # Simulate component-specific analysis
            # This would be replaced with actual technical analysis logic
            analysis_result = {
                'component': self.component_name,
                'timestamp': datetime.now().isoformat(),
                'technical_indicators': self._calculate_technical_indicators(component_data),
                'signals': self._generate_signals(component_data),
                'confidence': self._calculate_confidence(component_data),
                'processing_time': 0.0  # Will be updated below
            }

            processing_time = time.time() - start_time
            analysis_result['processing_time'] = processing_time

            # Update statistics
            self.processing_stats['total_calculations'] += 1
            self.processing_stats['last_processing_time'] = processing_time
            self.processing_stats['average_processing_time'] = (
                (self.processing_stats['average_processing_time'] *
                 (self.processing_stats['total_calculations'] - 1) + processing_time) /
                self.processing_stats['total_calculations']
            )

            return analysis_result

        except Exception as e:
            self.processing_stats['error_count'] += 1
            logger.error(f"Error in {self.component_name} analysis: {e}")
            return {
                'component': self.component_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate technical indicators for component"""
        # Placeholder implementation
        return {
            'ema_20': 100.0,
            'ema_100': 95.0,
            'ema_200': 90.0,
            'vwap': 98.0,
            'pivot': 97.0
        }

    def _generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals for component"""
        # Placeholder implementation
        return {
            'trend': 'bullish',
            'momentum': 'positive',
            'volatility': 'normal',
            'strength': 0.75
        }

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for analysis"""
        # Placeholder implementation
        return 0.85

class ParallelProcessingEngine:
    """Parallel processing engine for concurrent component analysis"""

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.component_processors = {
            'atm_straddle': ComponentProcessor('atm_straddle'),
            'itm1_straddle': ComponentProcessor('itm1_straddle'),
            'otm1_straddle': ComponentProcessor('otm1_straddle'),
            'combined_straddle': ComponentProcessor('combined_straddle'),
            'atm_ce': ComponentProcessor('atm_ce'),
            'atm_pe': ComponentProcessor('atm_pe')
        }

        # Performance tracking
        self.processing_stats = {
            'total_parallel_executions': 0,
            'average_parallel_time': 0.0,
            'max_parallel_time': 0.0,
            'min_parallel_time': float('inf'),
            'failed_executions': 0,
            'worker_utilization': {}
        }

    async def process_components_parallel(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all components in parallel"""
        start_time = time.time()

        try:
            # Create tasks for each component
            tasks = []
            for component_name, processor in self.component_processors.items():
                component_data = market_data.get(component_name, {})
                task = asyncio.create_task(
                    self._process_component_async(processor, component_data)
                )
                tasks.append((component_name, task))

            # Wait for all tasks to complete
            results = {}
            for component_name, task in tasks:
                try:
                    result = await task
                    results[component_name] = result
                except Exception as e:
                    logger.error(f"Error processing {component_name}: {e}")
                    results[component_name] = {'error': str(e)}

            # Update performance statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)

            return {
                'results': results,
                'processing_time': processing_time,
                'components_processed': len(results),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.processing_stats['failed_executions'] += 1
            logger.error(f"Error in parallel processing: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    async def _process_component_async(self, processor: ComponentProcessor,
                                     component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single component asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            processor.calculate_independent_analysis,
            component_data
        )

    def _update_processing_stats(self, processing_time: float):
        """Update processing statistics"""
        self.processing_stats['total_parallel_executions'] += 1

        # Update average processing time
        total_executions = self.processing_stats['total_parallel_executions']
        current_avg = self.processing_stats['average_parallel_time']
        self.processing_stats['average_parallel_time'] = (
            (current_avg * (total_executions - 1) + processing_time) / total_executions
        )

        # Update min/max processing times
        self.processing_stats['max_parallel_time'] = max(
            self.processing_stats['max_parallel_time'], processing_time
        )
        self.processing_stats['min_parallel_time'] = min(
            self.processing_stats['min_parallel_time'], processing_time
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        component_stats = {}
        for name, processor in self.component_processors.items():
            component_stats[name] = processor.processing_stats

        return {
            'parallel_processing': self.processing_stats,
            'component_processing': component_stats,
            'worker_pool_size': self.max_workers,
            'active_workers': len(self.executor._threads) if hasattr(self.executor, '_threads') else 0
        }

    def shutdown(self):
        """Shutdown the parallel processing engine"""
        self.executor.shutdown(wait=True)
        logger.info("Parallel processing engine shutdown complete")
