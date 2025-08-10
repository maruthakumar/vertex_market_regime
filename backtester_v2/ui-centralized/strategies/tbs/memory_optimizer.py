#!/usr/bin/env python3
"""
TBS Memory Optimizer - Advanced memory management for 112-parameter processing
Target: ≤3GB memory usage (previously achieved 137MB), efficient garbage collection
"""

import gc
import psutil
import weakref
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Iterator
import logging
import functools
import threading
import time
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import tracemalloc
import sys

logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    cached_objects: int
    gc_collections: Dict[int, int]
    
    @classmethod
    def current(cls) -> 'MemoryMetrics':
        """Get current memory metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return cls(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=process.memory_percent(),
            available_mb=virtual_memory.available / 1024 / 1024,
            cached_objects=len(gc.get_objects()),
            gc_collections=dict(enumerate(gc.get_stats()))
        )

class MemoryOptimizer:
    """Advanced memory optimization for TBS strategy processing"""
    
    def __init__(self, memory_limit_gb: float = 3.0):
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_mb = memory_limit_gb * 1024
        
        # Memory monitoring
        self.metrics_history: List[MemoryMetrics] = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Optimization strategies
        self.chunk_size = 10000  # Default chunk size for DataFrame processing
        self.cache_limits = {
            'parser_cache': 50,      # Max cached parsed results
            'query_cache': 100,      # Max cached queries  
            'result_cache': 25       # Max cached results
        }
        
        # Memory pressure thresholds
        self.pressure_thresholds = {
            'low': 0.4 * self.memory_limit_mb,      # 40% of limit
            'medium': 0.7 * self.memory_limit_mb,   # 70% of limit
            'high': 0.85 * self.memory_limit_mb,    # 85% of limit
            'critical': 0.95 * self.memory_limit_mb # 95% of limit
        }
        
        # Weak reference cache for automatic cleanup
        self.weak_cache = weakref.WeakValueDictionary()
        
        # Configure garbage collection for optimal performance
        self.configure_gc()
        
    def configure_gc(self):
        """Configure garbage collection for optimal memory management"""
        # Adjust GC thresholds for better performance with large datasets
        gc.set_threshold(1000, 15, 15)  # More frequent collection
        
        # Enable automatic garbage collection
        gc.enable()
        
        logger.info("Garbage collection configured for memory optimization")
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                try:
                    metrics = MemoryMetrics.current()
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 1000 metrics to prevent memory leak
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                    # Check for memory pressure and act
                    self.handle_memory_pressure(metrics)
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    break
                    
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Memory monitoring started (interval: {interval}s)")
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
        
    def handle_memory_pressure(self, metrics: MemoryMetrics):
        """Handle memory pressure situations"""
        current_memory = metrics.rss_mb
        
        if current_memory > self.pressure_thresholds['critical']:
            logger.warning(f"CRITICAL memory pressure: {current_memory:.1f}MB (limit: {self.memory_limit_mb:.1f}MB)")
            self.emergency_cleanup()
            
        elif current_memory > self.pressure_thresholds['high']:
            logger.warning(f"HIGH memory pressure: {current_memory:.1f}MB")
            self.aggressive_cleanup()
            
        elif current_memory > self.pressure_thresholds['medium']:
            logger.info(f"MEDIUM memory pressure: {current_memory:.1f}MB")
            self.moderate_cleanup()
            
        elif current_memory > self.pressure_thresholds['low']:
            logger.debug(f"LOW memory pressure: {current_memory:.1f}MB")
            self.light_cleanup()
            
    def light_cleanup(self):
        """Light memory cleanup - periodic maintenance"""
        # Force minor garbage collection
        collected = gc.collect()
        logger.debug(f"Light cleanup: collected {collected} objects")
        
    def moderate_cleanup(self):
        """Moderate memory cleanup"""
        # Clear weak cache
        self.weak_cache.clear()
        
        # Force full garbage collection
        for generation in range(3):
            collected = gc.collect(generation)
            
        logger.info(f"Moderate cleanup completed")
        
    def aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        # Clear all caches
        self.weak_cache.clear()
        
        # Force comprehensive garbage collection
        for _ in range(3):
            for generation in range(3):
                gc.collect(generation)
        
        # Reduce chunk size for future processing
        self.chunk_size = max(1000, self.chunk_size // 2)
        
        logger.warning(f"Aggressive cleanup completed, reduced chunk size to {self.chunk_size}")
        
    def emergency_cleanup(self):
        """Emergency memory cleanup - last resort"""
        # Clear everything possible
        self.weak_cache.clear()
        
        # Multiple full GC cycles
        for _ in range(5):
            for generation in range(3):
                collected = gc.collect(generation)
                
        # Drastically reduce chunk size
        self.chunk_size = max(500, self.chunk_size // 4)
        
        # Try to release memory back to OS (platform-specific)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass
            
        logger.error(f"EMERGENCY cleanup completed, chunk size reduced to {self.chunk_size}")
        
    @contextmanager
    def memory_managed_processing(self, description: str = "Processing"):
        """Context manager for memory-managed processing"""
        start_metrics = MemoryMetrics.current()
        
        try:
            logger.info(f"Starting {description} (Memory: {start_metrics.rss_mb:.1f}MB)")
            yield
            
        except MemoryError:
            logger.error(f"MemoryError during {description}")
            self.emergency_cleanup()
            raise
            
        finally:
            end_metrics = MemoryMetrics.current()
            memory_delta = end_metrics.rss_mb - start_metrics.rss_mb
            
            logger.info(f"Completed {description} (Memory: {end_metrics.rss_mb:.1f}MB, Delta: {memory_delta:+.1f}MB)")
            
            # Cleanup if memory increased significantly
            if memory_delta > 100:  # >100MB increase
                self.moderate_cleanup()
                
    def optimize_dataframe(self, df: pd.DataFrame, 
                          optimize_dtypes: bool = True,
                          reduce_precision: bool = True,
                          categorize_strings: bool = True) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        optimized_df = df.copy()
        
        if optimize_dtypes:
            # Optimize numeric dtypes
            for col in optimized_df.select_dtypes(include=['int64']).columns:
                if optimized_df[col].min() >= -128 and optimized_df[col].max() <= 127:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif optimized_df[col].min() >= -32768 and optimized_df[col].max() <= 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif optimized_df[col].min() >= -2147483648 and optimized_df[col].max() <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
                    
            # Optimize float dtypes
            if reduce_precision:
                for col in optimized_df.select_dtypes(include=['float64']).columns:
                    optimized_df[col] = optimized_df[col].astype('float32')
                    
        if categorize_strings:
            # Convert strings with low cardinality to categories
            for col in optimized_df.select_dtypes(include=['object']).columns:
                unique_ratio = optimized_df[col].nunique() / len(optimized_df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
                    
        final_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction_percent = (1 - final_memory / initial_memory) * 100
        
        logger.info(f"DataFrame optimized: {initial_memory:.1f}MB → {final_memory:.1f}MB ({reduction_percent:.1f}% reduction)")
        
        return optimized_df
        
    def chunked_processing(self, data: Union[pd.DataFrame, List], 
                          process_func: callable,
                          chunk_size: Optional[int] = None,
                          parallel: bool = True) -> Iterator[Any]:
        """Process data in memory-efficient chunks"""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        # Adjust chunk size based on current memory pressure
        current_memory = MemoryMetrics.current().rss_mb
        if current_memory > self.pressure_thresholds['medium']:
            chunk_size = max(100, chunk_size // 2)
        elif current_memory > self.pressure_thresholds['high']:
            chunk_size = max(50, chunk_size // 4)
            
        logger.info(f"Processing {len(data)} items in chunks of {chunk_size}")
        
        if isinstance(data, pd.DataFrame):
            chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        else:
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            
        if parallel and len(chunks) > 1:
            # Parallel processing with memory monitoring
            max_workers = min(4, len(chunks))  # Limit workers to prevent memory explosion
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(process_func, chunks):
                    yield result
                    
                    # Check memory pressure after each chunk
                    current_memory = MemoryMetrics.current().rss_mb
                    if current_memory > self.pressure_thresholds['high']:
                        logger.warning("High memory pressure during parallel processing")
                        self.moderate_cleanup()
        else:
            # Sequential processing
            for chunk in chunks:
                yield process_func(chunk)
                
                # Light cleanup after each chunk
                if len(chunks) > 10:  # Only for large datasets
                    self.light_cleanup()
                    
    def memory_efficient_concat(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Memory-efficient DataFrame concatenation"""
        if not dataframes:
            return pd.DataFrame()
            
        if len(dataframes) == 1:
            return dataframes[0]
            
        # Estimate memory requirements
        total_memory_mb = sum(df.memory_usage(deep=True).sum() for df in dataframes) / 1024 / 1024
        
        if total_memory_mb > self.memory_limit_mb * 0.8:  # >80% of limit
            logger.warning(f"Large concatenation detected: {total_memory_mb:.1f}MB")
            
            # Use chunked concatenation
            result = dataframes[0].copy()
            for df in dataframes[1:]:
                result = pd.concat([result, df], ignore_index=True)
                
                # Force garbage collection between concatenations
                gc.collect()
                
                # Check memory pressure
                current_memory = MemoryMetrics.current().rss_mb
                if current_memory > self.pressure_thresholds['high']:
                    self.moderate_cleanup()
                    
            return result
        else:
            # Standard concatenation for smaller datasets
            return pd.concat(dataframes, ignore_index=True)
            
    def cached_computation(self, cache_key: str, computation_func: callable, 
                          cache_type: str = 'result_cache') -> Any:
        """Memory-aware cached computation"""
        # Check cache limit
        cache_limit = self.cache_limits.get(cache_type, 100)
        
        if cache_key in self.weak_cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self.weak_cache[cache_key]
            
        # Check memory pressure before computation
        current_memory = MemoryMetrics.current().rss_mb
        if current_memory > self.pressure_thresholds['medium']:
            logger.info("High memory pressure, skipping cache")
            return computation_func()
            
        # Perform computation
        result = computation_func()
        
        # Cache if memory allows
        if len(self.weak_cache) < cache_limit:
            try:
                self.weak_cache[cache_key] = result
                logger.debug(f"Cached result for {cache_key}")
            except:
                # Cache storage failed, continue without caching
                pass
                
        return result
        
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report"""
        current_metrics = MemoryMetrics.current()
        
        report = {
            'current_memory': {
                'rss_mb': current_metrics.rss_mb,
                'vms_mb': current_metrics.vms_mb,
                'percent': current_metrics.percent,
                'available_mb': current_metrics.available_mb
            },
            'limits_and_thresholds': {
                'memory_limit_mb': self.memory_limit_mb,
                'low_threshold_mb': self.pressure_thresholds['low'],
                'medium_threshold_mb': self.pressure_thresholds['medium'],
                'high_threshold_mb': self.pressure_thresholds['high'],
                'critical_threshold_mb': self.pressure_thresholds['critical']
            },
            'cache_status': {
                'weak_cache_size': len(self.weak_cache),
                'chunk_size': self.chunk_size,
                'cache_limits': self.cache_limits
            },
            'gc_stats': dict(enumerate(gc.get_stats())),
            'memory_pressure': self._calculate_memory_pressure(),
            'optimization_settings': {
                'monitoring_active': self.monitoring_active,
                'metrics_history_length': len(self.metrics_history)
            }
        }
        
        # Add historical statistics if available
        if self.metrics_history:
            recent_metrics = self.metrics_history[-100:]  # Last 100 measurements
            report['historical_stats'] = {
                'avg_memory_mb': sum(m.rss_mb for m in recent_metrics) / len(recent_metrics),
                'max_memory_mb': max(m.rss_mb for m in recent_metrics),
                'min_memory_mb': min(m.rss_mb for m in recent_metrics),
                'memory_trend': 'increasing' if recent_metrics[-1].rss_mb > recent_metrics[0].rss_mb else 'decreasing'
            }
            
        return report
        
    def _calculate_memory_pressure(self) -> str:
        """Calculate current memory pressure level"""
        current_memory = MemoryMetrics.current().rss_mb
        
        if current_memory > self.pressure_thresholds['critical']:
            return 'CRITICAL'
        elif current_memory > self.pressure_thresholds['high']:
            return 'HIGH'
        elif current_memory > self.pressure_thresholds['medium']:
            return 'MEDIUM'
        elif current_memory > self.pressure_thresholds['low']:
            return 'LOW'
        else:
            return 'NORMAL'
            
    def cleanup_and_exit(self):
        """Cleanup before exit"""
        self.stop_monitoring()
        self.emergency_cleanup()
        logger.info("Memory optimizer cleanup completed")

# Global memory optimizer instance
_memory_optimizer = None

def get_memory_optimizer(memory_limit_gb: float = 3.0) -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer(memory_limit_gb)
    return _memory_optimizer

# Decorators for memory management
def memory_monitored(func):
    """Decorator to monitor memory usage of functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = get_memory_optimizer()
        with optimizer.memory_managed_processing(func.__name__):
            return func(*args, **kwargs)
    return wrapper

def memory_optimized_dataframe(func):
    """Decorator to automatically optimize DataFrame returns"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            optimizer = get_memory_optimizer()
            return optimizer.optimize_dataframe(result)
        return result
    return wrapper