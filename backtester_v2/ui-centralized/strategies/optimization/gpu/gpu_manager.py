"""
GPU Manager for Strategy Optimization

Manages GPU resources and selects optimal GPU backend for different workloads.
Provides unified interface for HeavyDB and CuPy acceleration.
"""

import logging
import time
import psutil
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class GPUCapabilities:
    """GPU system capabilities"""
    has_heavydb: bool
    has_cupy: bool
    gpu_memory_gb: float
    gpu_count: int
    heavydb_version: Optional[str]
    cupy_version: Optional[str]
    cuda_version: Optional[str]

@dataclass
class WorkloadProfile:
    """Workload characteristics for GPU backend selection"""
    data_size: int
    parameter_count: int
    iteration_count: int
    batch_size: int
    algorithm_type: str
    parallel_evaluations: bool

class GPUManager:
    """
    Central GPU resource manager for optimization
    
    Automatically detects available GPU resources and selects the optimal
    backend for different workload types.
    """
    
    def __init__(self,
                 prefer_heavydb: bool = True,
                 batch_size_threshold: int = 10000,
                 memory_limit_gb: float = None,
                 enable_monitoring: bool = True):
        """
        Initialize GPU manager
        
        Args:
            prefer_heavydb: Prefer HeavyDB for large datasets
            batch_size_threshold: Threshold for choosing HeavyDB vs CuPy
            memory_limit_gb: GPU memory limit in GB
            enable_monitoring: Enable GPU monitoring
        """
        self.prefer_heavydb = prefer_heavydb
        self.batch_size_threshold = batch_size_threshold
        self.memory_limit_gb = memory_limit_gb
        self.enable_monitoring = enable_monitoring
        
        # Detect available GPU capabilities
        self.capabilities = self._detect_gpu_capabilities()
        
        # Initialize backends
        self.heavydb_backend = None
        self.cupy_backend = None
        
        if self.capabilities.has_heavydb:
            try:
                from .heavydb_acceleration import HeavyDBAcceleration
                self.heavydb_backend = HeavyDBAcceleration()
                logger.info("HeavyDB acceleration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HeavyDB acceleration: {e}")
                self.capabilities.has_heavydb = False
        
        if self.capabilities.has_cupy:
            try:
                from .cupy_acceleration import CuPyAcceleration
                self.cupy_backend = CuPyAcceleration()
                logger.info("CuPy acceleration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CuPy acceleration: {e}")
                self.capabilities.has_cupy = False
        
        # GPU monitoring
        self.gpu_stats = {
            'total_evaluations': 0,
            'heavydb_evaluations': 0,
            'cupy_evaluations': 0,
            'cpu_fallback_evaluations': 0,
            'total_gpu_time': 0.0,
            'average_speedup': 1.0
        }
        
        logger.info(f"GPU Manager initialized: HeavyDB={self.capabilities.has_heavydb}, "
                   f"CuPy={self.capabilities.has_cupy}")
    
    def _detect_gpu_capabilities(self) -> GPUCapabilities:
        """Detect available GPU capabilities"""
        has_heavydb = False
        has_cupy = False
        gpu_memory_gb = 0.0
        gpu_count = 0
        heavydb_version = None
        cupy_version = None
        cuda_version = None
        
        # Check HeavyDB availability
        try:
            import pymapd
            has_heavydb = True
            heavydb_version = getattr(pymapd, '__version__', 'unknown')
        except ImportError:
            pass
        
        # Check CuPy availability
        try:
            import cupy
            has_cupy = True
            cupy_version = cupy.__version__
            
            # Get GPU info
            gpu_count = cupy.cuda.runtime.getDeviceCount()
            if gpu_count > 0:
                for i in range(gpu_count):
                    with cupy.cuda.Device(i):
                        meminfo = cupy.cuda.runtime.memGetInfo()
                        gpu_memory_gb += meminfo[1] / (1024**3)  # Convert to GB
            
            # Get CUDA version
            cuda_version = cupy.cuda.runtime.runtimeGetVersion()
            
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error detecting CuPy capabilities: {e}")
        
        return GPUCapabilities(
            has_heavydb=has_heavydb,
            has_cupy=has_cupy,
            gpu_memory_gb=gpu_memory_gb,
            gpu_count=gpu_count,
            heavydb_version=heavydb_version,
            cupy_version=cupy_version,
            cuda_version=cuda_version
        )
    
    def select_backend(self, workload: WorkloadProfile) -> str:
        """
        Select optimal GPU backend for workload
        
        Args:
            workload: Workload characteristics
            
        Returns:
            Backend name ('heavydb', 'cupy', 'cpu')
        """
        # Check if any GPU backend is available
        if not (self.capabilities.has_heavydb or self.capabilities.has_cupy):
            return 'cpu'
        
        # Decision logic based on workload characteristics
        if workload.data_size > self.batch_size_threshold:
            # Large datasets: prefer HeavyDB
            if self.capabilities.has_heavydb and self.prefer_heavydb:
                return 'heavydb'
            elif self.capabilities.has_cupy:
                return 'cupy'
            else:
                return 'cpu'
        
        else:
            # Smaller datasets: prefer CuPy for flexibility
            if self.capabilities.has_cupy:
                return 'cupy'
            elif self.capabilities.has_heavydb:
                return 'heavydb'
            else:
                return 'cpu'
    
    def accelerate_objective_function(self,
                                    objective_function: Callable,
                                    workload: WorkloadProfile) -> Callable:
        """
        Create GPU-accelerated version of objective function
        
        Args:
            objective_function: Original objective function
            workload: Workload characteristics
            
        Returns:
            GPU-accelerated objective function
        """
        backend = self.select_backend(workload)
        
        if backend == 'heavydb' and self.heavydb_backend:
            return self.heavydb_backend.accelerate_function(objective_function, workload)
        elif backend == 'cupy' and self.cupy_backend:
            return self.cupy_backend.accelerate_function(objective_function, workload)
        else:
            # CPU fallback
            logger.info("Using CPU fallback for objective function")
            return self._cpu_fallback_wrapper(objective_function)
    
    def _cpu_fallback_wrapper(self, objective_function: Callable) -> Callable:
        """Wrap objective function with CPU monitoring"""
        def wrapped_function(*args, **kwargs):
            start_time = time.time()
            result = objective_function(*args, **kwargs)
            
            if self.enable_monitoring:
                self.gpu_stats['cpu_fallback_evaluations'] += 1
                self.gpu_stats['total_evaluations'] += 1
            
            return result
        
        return wrapped_function
    
    def batch_evaluate(self,
                      objective_function: Callable,
                      parameter_sets: List[Dict[str, float]],
                      workload: WorkloadProfile) -> List[float]:
        """
        Batch evaluate objective function on multiple parameter sets
        
        Args:
            objective_function: Objective function
            parameter_sets: List of parameter dictionaries
            workload: Workload characteristics
            
        Returns:
            List of evaluation results
        """
        backend = self.select_backend(workload)
        
        logger.info(f"Batch evaluating {len(parameter_sets)} parameter sets using {backend}")
        
        start_time = time.time()
        
        if backend == 'heavydb' and self.heavydb_backend:
            results = self.heavydb_backend.batch_evaluate(
                objective_function, parameter_sets, workload
            )
            if self.enable_monitoring:
                self.gpu_stats['heavydb_evaluations'] += len(parameter_sets)
                
        elif backend == 'cupy' and self.cupy_backend:
            results = self.cupy_backend.batch_evaluate(
                objective_function, parameter_sets, workload
            )
            if self.enable_monitoring:
                self.gpu_stats['cupy_evaluations'] += len(parameter_sets)
                
        else:
            # CPU fallback
            results = []
            for params in parameter_sets:
                results.append(objective_function(params))
            if self.enable_monitoring:
                self.gpu_stats['cpu_fallback_evaluations'] += len(parameter_sets)
        
        if self.enable_monitoring:
            self.gpu_stats['total_evaluations'] += len(parameter_sets)
            self.gpu_stats['total_gpu_time'] += time.time() - start_time
        
        return results
    
    def estimate_speedup(self, workload: WorkloadProfile) -> float:
        """
        Estimate speedup for given workload
        
        Args:
            workload: Workload characteristics
            
        Returns:
            Estimated speedup factor
        """
        backend = self.select_backend(workload)
        
        if backend == 'heavydb':
            # HeavyDB typically provides 10-100x speedup for large datasets
            base_speedup = min(50.0, workload.data_size / 1000.0)
            return max(1.0, base_speedup)
            
        elif backend == 'cupy':
            # CuPy provides 5-20x speedup for numerical computations
            base_speedup = min(15.0, workload.parameter_count * 2.0)
            return max(1.0, base_speedup)
            
        else:
            return 1.0  # No speedup for CPU
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        memory_usage = {
            'system_memory_gb': psutil.virtual_memory().used / (1024**3),
            'system_memory_percent': psutil.virtual_memory().percent
        }
        
        if self.capabilities.has_cupy:
            try:
                import cupy
                for i in range(self.capabilities.gpu_count):
                    with cupy.cuda.Device(i):
                        meminfo = cupy.cuda.runtime.memGetInfo()
                        used_gb = (meminfo[1] - meminfo[0]) / (1024**3)
                        total_gb = meminfo[1] / (1024**3)
                        memory_usage[f'gpu_{i}_used_gb'] = used_gb
                        memory_usage[f'gpu_{i}_total_gb'] = total_gb
                        memory_usage[f'gpu_{i}_percent'] = (used_gb / total_gb) * 100
            except Exception as e:
                logger.warning(f"Error getting GPU memory usage: {e}")
        
        return memory_usage
    
    def optimize_batch_size(self, workload: WorkloadProfile) -> int:
        """
        Optimize batch size based on available GPU memory
        
        Args:
            workload: Workload characteristics
            
        Returns:
            Optimal batch size
        """
        if not (self.capabilities.has_heavydb or self.capabilities.has_cupy):
            return min(workload.batch_size, 1000)  # Conservative CPU batch size
        
        # Estimate memory per evaluation
        memory_per_eval_mb = workload.parameter_count * 8 * 4  # Rough estimate
        
        # Available GPU memory
        available_memory_mb = self.capabilities.gpu_memory_gb * 1024 * 0.8  # Use 80% of GPU memory
        
        # Calculate optimal batch size
        optimal_batch_size = int(available_memory_mb / memory_per_eval_mb)
        optimal_batch_size = max(1, min(optimal_batch_size, workload.batch_size))
        
        logger.info(f"Optimized batch size: {optimal_batch_size} "
                   f"(from {workload.batch_size})")
        
        return optimal_batch_size
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU performance statistics"""
        stats = self.gpu_stats.copy()
        
        # Calculate additional metrics
        if stats['total_evaluations'] > 0:
            stats['heavydb_usage_percent'] = (stats['heavydb_evaluations'] / 
                                            stats['total_evaluations']) * 100
            stats['cupy_usage_percent'] = (stats['cupy_evaluations'] / 
                                         stats['total_evaluations']) * 100
            stats['cpu_fallback_percent'] = (stats['cpu_fallback_evaluations'] / 
                                           stats['total_evaluations']) * 100
            
            stats['average_evaluation_time'] = (stats['total_gpu_time'] / 
                                              stats['total_evaluations'])
        
        # Add capability info
        stats['capabilities'] = {
            'has_heavydb': self.capabilities.has_heavydb,
            'has_cupy': self.capabilities.has_cupy,
            'gpu_memory_gb': self.capabilities.gpu_memory_gb,
            'gpu_count': self.capabilities.gpu_count
        }
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.gpu_stats = {
            'total_evaluations': 0,
            'heavydb_evaluations': 0,
            'cupy_evaluations': 0,
            'cpu_fallback_evaluations': 0,
            'total_gpu_time': 0.0,
            'average_speedup': 1.0
        }
        
        logger.info("GPU performance statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform GPU health check"""
        health_status = {
            'overall_status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check GPU availability
        if not (self.capabilities.has_heavydb or self.capabilities.has_cupy):
            health_status['overall_status'] = 'degraded'
            health_status['issues'].append('No GPU acceleration available')
            health_status['recommendations'].append('Install CuPy or configure HeavyDB for GPU acceleration')
        
        # Check memory usage
        memory_usage = self.get_memory_usage()
        for key, value in memory_usage.items():
            if 'percent' in key and value > 90:
                health_status['issues'].append(f'High memory usage: {key} = {value:.1f}%')
                health_status['recommendations'].append('Consider reducing batch size or optimizing memory usage')
        
        # Check performance
        if self.gpu_stats['cpu_fallback_evaluations'] > self.gpu_stats['total_evaluations'] * 0.5:
            health_status['issues'].append('High CPU fallback usage')
            health_status['recommendations'].append('Check GPU backend configuration and workload characteristics')
        
        if health_status['issues']:
            health_status['overall_status'] = 'degraded' if len(health_status['issues']) < 3 else 'unhealthy'
        
        return health_status
    
    def __repr__(self) -> str:
        return (f"GPUManager(HeavyDB={self.capabilities.has_heavydb}, "
                f"CuPy={self.capabilities.has_cupy}, "
                f"GPU_Memory={self.capabilities.gpu_memory_gb:.1f}GB)")