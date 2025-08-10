"""
CuPy GPU Acceleration for Strategy Optimization

Provides CuPy-based GPU acceleration for numerical computations and optimization.
Optimized for smaller datasets and complex mathematical operations.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class CuPyConfig:
    """CuPy configuration settings"""
    memory_pool_fraction: float = 0.8
    enable_memory_pool: bool = True
    cuda_streams: int = 4
    prefer_gpu_arrays: bool = True
    batch_processing: bool = True

class CuPyAcceleration:
    """
    CuPy GPU acceleration for optimization
    
    Provides GPU-accelerated numerical computations using CuPy.
    Suitable for mathematical operations on parameter vectors.
    """
    
    def __init__(self,
                 config: Optional[CuPyConfig] = None,
                 device_id: int = 0,
                 enable_memory_pool: bool = True,
                 batch_threshold: int = 100):
        """
        Initialize CuPy acceleration
        
        Args:
            config: CuPy configuration
            device_id: CUDA device ID to use
            enable_memory_pool: Whether to use CuPy memory pool
            batch_threshold: Minimum batch size for GPU acceleration
        """
        self.config = config or CuPyConfig()
        self.device_id = device_id
        self.enable_memory_pool = enable_memory_pool
        self.batch_threshold = batch_threshold
        
        # CuPy availability
        self.cupy_available = False
        self.cp = None
        self.device = None
        self.memory_pool = None
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_fallback_operations': 0,
            'total_gpu_time': 0.0,
            'total_transfer_time': 0.0,
            'average_speedup': 1.0
        }
        
        # Initialize CuPy
        self._initialize_cupy()
        
        logger.info(f"CuPy acceleration initialized: available={self.cupy_available}")
    
    def _initialize_cupy(self):
        """Initialize CuPy and GPU resources"""
        try:
            import cupy as cp
            self.cp = cp
            
            # Set device
            cp.cuda.Device(self.device_id).use()
            self.device = cp.cuda.Device(self.device_id)
            
            # Setup memory pool if enabled
            if self.enable_memory_pool:
                self.memory_pool = cp.get_default_memory_pool()
                # Set memory pool limit (80% of GPU memory by default)
                with self.device:
                    meminfo = cp.cuda.runtime.memGetInfo()
                    total_memory = meminfo[1]
                    pool_limit = int(total_memory * self.config.memory_pool_fraction)
                    self.memory_pool.set_limit(size=pool_limit)
            
            self.cupy_available = True
            
            logger.info(f"CuPy initialized on device {self.device_id}")
            if self.memory_pool:
                logger.info(f"Memory pool configured with {self.config.memory_pool_fraction:.1%} of GPU memory")
            
        except ImportError:
            logger.warning("CuPy not available - falling back to CPU")
            self.cupy_available = False
        except Exception as e:
            logger.error(f"Failed to initialize CuPy: {e}")
            self.cupy_available = False
    
    def accelerate_function(self,
                           objective_function: Callable,
                           workload: 'WorkloadProfile') -> Callable:
        """
        Create CuPy-accelerated version of objective function
        
        Args:
            objective_function: Original objective function
            workload: Workload characteristics
            
        Returns:
            GPU-accelerated objective function
        """
        if not self.cupy_available:
            logger.warning("CuPy not available, returning original function")
            return objective_function
        
        def cupy_accelerated_function(params: Dict[str, float]) -> float:
            # For single evaluations, use vectorized evaluation if beneficial
            if self._should_use_gpu(1, workload):
                return self._gpu_single_evaluation(objective_function, params)
            else:
                return objective_function(params)
        
        return cupy_accelerated_function
    
    def batch_evaluate(self,
                      objective_function: Callable,
                      parameter_sets: List[Dict[str, float]],
                      workload: 'WorkloadProfile') -> List[float]:
        """
        Batch evaluate using CuPy GPU acceleration
        
        Args:
            objective_function: Objective function
            parameter_sets: List of parameter dictionaries
            workload: Workload characteristics
            
        Returns:
            List of evaluation results
        """
        if not self.cupy_available or len(parameter_sets) == 0:
            logger.warning("CuPy not available or empty parameter set, using CPU fallback")
            return [objective_function(params) for params in parameter_sets]
        
        if not self._should_use_gpu(len(parameter_sets), workload):
            logger.info("Using CPU for small batch")
            return [objective_function(params) for params in parameter_sets]
        
        start_time = time.time()
        
        try:
            # Convert parameter sets to GPU arrays
            gpu_params = self._parameters_to_gpu_arrays(parameter_sets)
            
            # Perform GPU-accelerated batch evaluation
            gpu_results = self._gpu_batch_evaluation(objective_function, gpu_params, workload)
            
            # Convert results back to CPU
            results = self._gpu_arrays_to_cpu(gpu_results)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats['total_operations'] += 1
            self.performance_stats['gpu_operations'] += len(parameter_sets)
            self.performance_stats['total_gpu_time'] += execution_time
            
            logger.info(f"CuPy batch evaluation: {len(parameter_sets)} parameters in {execution_time:.2f}s "
                       f"({len(parameter_sets)/execution_time:.0f} params/sec)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in CuPy batch evaluation: {e}")
            # Fallback to CPU evaluation
            self.performance_stats['cpu_fallback_operations'] += len(parameter_sets)
            return [objective_function(params) for params in parameter_sets]
    
    def _should_use_gpu(self, batch_size: int, workload: 'WorkloadProfile') -> bool:
        """Determine if GPU acceleration should be used"""
        if not self.cupy_available:
            return False
        
        # Use GPU for larger batches or complex computations
        return (batch_size >= self.batch_threshold or
                workload.parameter_count > 10 or
                workload.algorithm_type in ['genetic', 'particle_swarm', 'differential_evolution'])
    
    def _parameters_to_gpu_arrays(self, parameter_sets: List[Dict[str, float]]) -> Dict[str, Any]:
        """Convert parameter sets to GPU arrays"""
        transfer_start = time.time()
        
        # Convert to pandas DataFrame for easier processing
        params_df = pd.DataFrame(parameter_sets)
        
        # Transfer to GPU
        with self.device:
            gpu_arrays = {}
            for column in params_df.columns:
                gpu_arrays[column] = self.cp.asarray(params_df[column].values, dtype=self.cp.float32)
        
        self.performance_stats['total_transfer_time'] += time.time() - transfer_start
        return gpu_arrays
    
    def _gpu_single_evaluation(self, objective_function: Callable, params: Dict[str, float]) -> float:
        """GPU-accelerated single parameter evaluation"""
        with self.device:
            # Convert to GPU arrays
            gpu_params = {k: self.cp.asarray([v], dtype=self.cp.float32) for k, v in params.items()}
            
            # Try to vectorize the objective function
            try:
                result = self._vectorized_objective(objective_function, gpu_params)
                return float(result[0])
            except:
                # Fallback to CPU evaluation
                return objective_function(params)
    
    def _gpu_batch_evaluation(self,
                             objective_function: Callable,
                             gpu_params: Dict[str, Any],
                             workload: 'WorkloadProfile') -> Any:
        """Perform GPU-accelerated batch evaluation"""
        
        with self.device:
            try:
                # Try vectorized evaluation first
                return self._vectorized_objective(objective_function, gpu_params)
            except:
                # Fallback to element-wise evaluation on GPU
                return self._elementwise_gpu_evaluation(objective_function, gpu_params)
    
    def _vectorized_objective(self, objective_function: Callable, gpu_params: Dict[str, Any]) -> Any:
        """
        Attempt to create a vectorized version of the objective function
        
        This is a simplified approach - in practice, you'd need function-specific
        vectorization based on the mathematical operations involved.
        """
        # Get parameter names and values
        param_names = list(gpu_params.keys())
        param_arrays = list(gpu_params.values())
        
        # Get batch size
        batch_size = param_arrays[0].shape[0] if param_arrays else 1
        
        # Create common vectorized operations
        try:
            # Example vectorized operations for common optimization functions
            if hasattr(objective_function, '__name__'):
                func_name = objective_function.__name__.lower()
                
                if 'quadratic' in func_name or 'sphere' in func_name:
                    # Sum of squares: f(x) = sum(x_i^2)
                    result = self.cp.zeros(batch_size, dtype=self.cp.float32)
                    for array in param_arrays:
                        result += array ** 2
                    return result
                
                elif 'rosenbrock' in func_name:
                    # Rosenbrock function (for 2D case)
                    if len(param_arrays) == 2:
                        x, y = param_arrays[0], param_arrays[1]
                        result = 100 * (y - x**2)**2 + (1 - x)**2
                        return result
                
                elif 'ackley' in func_name:
                    # Ackley function
                    n = len(param_arrays)
                    sum_sq = self.cp.zeros(batch_size)
                    sum_cos = self.cp.zeros(batch_size)
                    
                    for array in param_arrays:
                        sum_sq += array ** 2
                        sum_cos += self.cp.cos(2 * self.cp.pi * array)
                    
                    result = (-20 * self.cp.exp(-0.2 * self.cp.sqrt(sum_sq / n)) -
                             self.cp.exp(sum_cos / n) + 20 + self.cp.e)
                    return result
                
                elif 'sharpe' in func_name:
                    # Sharpe ratio calculation (simplified)
                    # Assumes parameters represent returns or weights
                    weights = self.cp.stack(param_arrays, axis=1)
                    mean_return = self.cp.mean(weights, axis=1)
                    std_return = self.cp.std(weights, axis=1)
                    result = mean_return / (std_return + 1e-8)
                    return result
            
            # If no specific vectorization available, fall back
            raise NotImplementedError("No vectorization available for this function")
            
        except Exception as e:
            logger.debug(f"Vectorization failed: {e}")
            raise
    
    def _elementwise_gpu_evaluation(self, objective_function: Callable, gpu_params: Dict[str, Any]) -> Any:
        """Element-wise evaluation on GPU (less efficient but more general)"""
        
        param_names = list(gpu_params.keys())
        batch_size = gpu_params[param_names[0]].shape[0] if param_names else 0
        
        results = []
        
        # Evaluate each parameter set individually
        for i in range(batch_size):
            params_dict = {}
            for name in param_names:
                params_dict[name] = float(gpu_params[name][i])
            
            # Evaluate on CPU (since we can't easily run arbitrary Python functions on GPU)
            result = objective_function(params_dict)
            results.append(result)
        
        # Convert results to GPU array
        return self.cp.asarray(results, dtype=self.cp.float32)
    
    def _gpu_arrays_to_cpu(self, gpu_results: Any) -> List[float]:
        """Convert GPU results back to CPU"""
        transfer_start = time.time()
        
        with self.device:
            if isinstance(gpu_results, self.cp.ndarray):
                cpu_results = gpu_results.get().tolist()
            else:
                cpu_results = [float(gpu_results)]
        
        self.performance_stats['total_transfer_time'] += time.time() - transfer_start
        
        if isinstance(cpu_results, list):
            return cpu_results
        else:
            return [cpu_results]
    
    def vectorize_parameters(self, parameter_sets: List[Dict[str, float]]) -> Tuple[Any, List[str]]:
        """
        Convert parameter sets to vectorized GPU arrays
        
        Args:
            parameter_sets: List of parameter dictionaries
            
        Returns:
            Tuple of (GPU parameter matrix, parameter names)
        """
        if not self.cupy_available or not parameter_sets:
            return None, []
        
        # Convert to DataFrame
        params_df = pd.DataFrame(parameter_sets)
        param_names = params_df.columns.tolist()
        
        with self.device:
            # Create GPU matrix (batch_size x n_params)
            gpu_matrix = self.cp.asarray(params_df.values, dtype=self.cp.float32)
        
        return gpu_matrix, param_names
    
    def evaluate_vectorized_objective(self,
                                    vectorized_function: Callable,
                                    gpu_matrix: Any,
                                    param_names: List[str]) -> List[float]:
        """
        Evaluate vectorized objective function on GPU
        
        Args:
            vectorized_function: Vectorized objective function
            gpu_matrix: GPU parameter matrix
            param_names: Parameter names
            
        Returns:
            List of evaluation results
        """
        if not self.cupy_available:
            return []
        
        with self.device:
            try:
                # Call vectorized function
                gpu_results = vectorized_function(gpu_matrix, param_names)
                
                # Convert to CPU
                cpu_results = gpu_results.get().tolist()
                return cpu_results if isinstance(cpu_results, list) else [cpu_results]
                
            except Exception as e:
                logger.error(f"Error in vectorized evaluation: {e}")
                return []
    
    def optimize_for_workload(self, workload: 'WorkloadProfile') -> Dict[str, Any]:
        """
        Optimize CuPy settings for specific workload
        
        Args:
            workload: Workload characteristics
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'batch_threshold': self.batch_threshold,
            'memory_settings': {},
            'performance_optimizations': []
        }
        
        # Adjust batch threshold based on workload
        if workload.parameter_count > 20:
            recommendations['batch_threshold'] = max(50, self.batch_threshold)
            recommendations['performance_optimizations'].append('Reduced batch threshold for high-dimensional problems')
        
        # Memory recommendations
        estimated_memory_mb = workload.data_size * workload.parameter_count * 4 / (1024 * 1024)  # float32
        recommendations['memory_settings']['estimated_memory_mb'] = estimated_memory_mb
        
        if estimated_memory_mb > 1024:  # > 1GB
            recommendations['performance_optimizations'].append('Consider batch processing for large datasets')
        
        # Algorithm-specific optimizations
        if workload.algorithm_type in ['genetic', 'particle_swarm']:
            recommendations['performance_optimizations'].append('Population-based algorithms benefit from vectorization')
        
        return recommendations
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if not self.cupy_available:
            return {'available': False}
        
        with self.device:
            meminfo = self.cp.cuda.runtime.memGetInfo()
            free_memory = meminfo[0]
            total_memory = meminfo[1]
            used_memory = total_memory - free_memory
            
            memory_info = {
                'available': True,
                'device_id': self.device_id,
                'total_memory_mb': total_memory / (1024 * 1024),
                'used_memory_mb': used_memory / (1024 * 1024),
                'free_memory_mb': free_memory / (1024 * 1024),
                'memory_utilization_percent': (used_memory / total_memory) * 100
            }
            
            if self.memory_pool:
                pool_info = {
                    'pool_used_bytes': self.memory_pool.used_bytes(),
                    'pool_total_bytes': self.memory_pool.total_bytes(),
                    'pool_limit_bytes': getattr(self.memory_pool, 'get_limit', lambda: 0)()
                }
                memory_info['memory_pool'] = pool_info
            
            return memory_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get CuPy performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate additional metrics
        if stats['total_operations'] > 0:
            stats['gpu_usage_percent'] = (stats['gpu_operations'] / 
                                        max(1, stats['gpu_operations'] + stats['cpu_fallback_operations'])) * 100
            
            if stats['total_gpu_time'] > 0:
                stats['average_gpu_operations_per_second'] = stats['gpu_operations'] / stats['total_gpu_time']
        
        # Add memory info
        stats['memory_info'] = self.get_memory_info()
        
        # Add configuration
        stats['configuration'] = {
            'device_id': self.device_id,
            'batch_threshold': self.batch_threshold,
            'memory_pool_enabled': self.enable_memory_pool,
            'cupy_available': self.cupy_available
        }
        
        return stats
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if self.cupy_available and self.memory_pool:
            with self.device:
                self.memory_pool.free_all_blocks()
                logger.info("GPU memory pool cleaned up")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform CuPy health check"""
        health = {
            'status': 'healthy',
            'issues': [],
            'gpu_available': self.cupy_available
        }
        
        if not self.cupy_available:
            health['status'] = 'unavailable'
            health['issues'].append('CuPy not available')
            return health
        
        try:
            # Test GPU computation
            with self.device:
                test_array = self.cp.ones(100)
                result = self.cp.sum(test_array)
                expected = 100.0
                
                if abs(float(result) - expected) > 1e-6:
                    health['issues'].append('GPU computation test failed')
                    health['status'] = 'unhealthy'
            
            # Check memory
            memory_info = self.get_memory_info()
            if memory_info['memory_utilization_percent'] > 90:
                health['issues'].append('High GPU memory usage')
                health['status'] = 'degraded'
            
            # Check performance
            if self.performance_stats['cpu_fallback_operations'] > self.performance_stats['gpu_operations']:
                health['issues'].append('High CPU fallback rate')
                health['status'] = 'degraded'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['issues'].append(f'Health check failed: {e}')
        
        return health
    
    def close(self):
        """Close CuPy resources"""
        if self.cupy_available:
            try:
                self.cleanup_memory()
                logger.info("CuPy resources cleaned up")
            except Exception as e:
                logger.warning(f"Error during CuPy cleanup: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.close()