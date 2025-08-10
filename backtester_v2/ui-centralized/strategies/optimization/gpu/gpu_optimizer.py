"""
GPU-Accelerated Optimizer

Unified GPU optimizer that combines HeavyDB and CuPy acceleration
with intelligent backend selection and performance optimization.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import json

from ..base.base_optimizer import BaseOptimizer, OptimizationResult
from .gpu_manager import GPUManager, WorkloadProfile
from .heavydb_acceleration import HeavyDBAcceleration
from .cupy_acceleration import CuPyAcceleration

logger = logging.getLogger(__name__)

@dataclass
class GPUOptimizationResult(OptimizationResult):
    """Extended optimization result with GPU performance metrics"""
    gpu_backend_used: str = 'cpu'
    gpu_speedup: float = 1.0
    gpu_memory_usage_mb: float = 0.0
    gpu_execution_time: float = 0.0
    cpu_fallback_count: int = 0
    backend_switches: int = 0

class GPUOptimizer(BaseOptimizer):
    """
    GPU-accelerated optimizer using dual backend (HeavyDB + CuPy)
    
    Automatically selects the best GPU backend based on workload characteristics
    and provides comprehensive GPU acceleration for optimization algorithms.
    """
    
    def __init__(self,
                 base_optimizer: BaseOptimizer,
                 enable_heavydb: bool = True,
                 enable_cupy: bool = True,
                 auto_backend_selection: bool = True,
                 performance_monitoring: bool = True,
                 memory_limit_gb: Optional[float] = None):
        """
        Initialize GPU optimizer
        
        Args:
            base_optimizer: Base optimization algorithm
            enable_heavydb: Enable HeavyDB acceleration
            enable_cupy: Enable CuPy acceleration
            auto_backend_selection: Automatically select optimal backend
            performance_monitoring: Enable performance monitoring
            memory_limit_gb: GPU memory limit in GB
        """
        # Initialize base optimizer attributes
        super().__init__(
            param_space=base_optimizer.param_space,
            objective_function=base_optimizer.objective_function,
            maximize=base_optimizer.maximize,
            random_seed=getattr(base_optimizer, 'random_seed', None)
        )
        
        self.base_optimizer = base_optimizer
        self.enable_heavydb = enable_heavydb
        self.enable_cupy = enable_cupy
        self.auto_backend_selection = auto_backend_selection
        self.performance_monitoring = performance_monitoring
        
        # Initialize GPU manager
        self.gpu_manager = GPUManager(
            prefer_heavydb=True,
            memory_limit_gb=memory_limit_gb,
            enable_monitoring=performance_monitoring
        )
        
        # Performance tracking
        self.gpu_performance_history = []
        self.backend_usage_stats = {
            'heavydb': 0,
            'cupy': 0,
            'cpu': 0
        }
        
        # Current workload profile
        self.current_workload = None
        
        logger.info(f"GPU Optimizer initialized with {base_optimizer.__class__.__name__}")
        logger.info(f"Available backends: HeavyDB={self.gpu_manager.capabilities.has_heavydb}, "
                   f"CuPy={self.gpu_manager.capabilities.has_cupy}")
    
    def optimize(self,
                n_iterations: int = 1000,
                callback: Optional[Callable] = None,
                workload_hint: Optional[Dict[str, Any]] = None,
                **kwargs) -> GPUOptimizationResult:
        """
        Run GPU-accelerated optimization
        
        Args:
            n_iterations: Number of optimization iterations
            callback: Optional callback function
            workload_hint: Hints about workload characteristics
            **kwargs: Additional optimization parameters
            
        Returns:
            GPUOptimizationResult with performance metrics
        """
        start_time = time.time()
        
        # Create workload profile
        self.current_workload = self._create_workload_profile(
            n_iterations, workload_hint or {}
        )
        
        # Select and configure GPU backend
        selected_backend = self._select_optimal_backend(self.current_workload)
        
        logger.info(f"Starting GPU optimization with {selected_backend} backend")
        logger.info(f"Workload: {n_iterations} iterations, "
                   f"{self.current_workload.parameter_count} parameters")
        
        # Accelerate objective function
        original_objective = self.base_optimizer.objective_function
        accelerated_objective = self._create_accelerated_objective_function(
            original_objective, selected_backend
        )
        
        # Update base optimizer with accelerated function
        self.base_optimizer.objective_function = accelerated_objective
        
        try:
            # Run base optimization
            base_result = self.base_optimizer.optimize(
                n_iterations=n_iterations,
                callback=self._create_gpu_callback(callback),
                **kwargs
            )
            
            # Restore original objective function
            self.base_optimizer.objective_function = original_objective
            
            # Calculate GPU metrics
            gpu_metrics = self._calculate_gpu_metrics(start_time, selected_backend)
            
            # Create enhanced result
            gpu_result = GPUOptimizationResult(
                best_params=base_result.best_params,
                best_score=base_result.best_score,
                n_iterations=base_result.n_iterations,
                execution_time=time.time() - start_time,
                convergence_history=base_result.convergence_history,
                algorithm_name=f"GPU-{base_result.algorithm_name}",
                metadata=base_result.metadata,
                gpu_backend_used=selected_backend,
                gpu_speedup=gpu_metrics['speedup'],
                gpu_memory_usage_mb=gpu_metrics['memory_usage_mb'],
                gpu_execution_time=gpu_metrics['gpu_time'],
                cpu_fallback_count=gpu_metrics['cpu_fallback_count'],
                backend_switches=gpu_metrics['backend_switches']
            )
            
            # Update performance history
            if self.performance_monitoring:
                self._update_performance_history(gpu_result)
            
            logger.info(f"GPU optimization completed in {gpu_result.execution_time:.2f}s")
            logger.info(f"GPU speedup: {gpu_result.gpu_speedup:.2f}x")
            logger.info(f"Best score: {gpu_result.best_score:.6f}")
            
            return gpu_result
            
        except Exception as e:
            # Restore original objective function
            self.base_optimizer.objective_function = original_objective
            logger.error(f"Error in GPU optimization: {e}")
            raise
    
    def _create_workload_profile(self,
                                n_iterations: int,
                                workload_hint: Dict[str, Any]) -> WorkloadProfile:
        """Create workload profile for backend selection"""
        
        # Estimate data size and batch size
        data_size = workload_hint.get('data_size', n_iterations * 10)
        batch_size = workload_hint.get('batch_size', min(1000, n_iterations // 10))
        
        return WorkloadProfile(
            data_size=data_size,
            parameter_count=len(self.param_space),
            iteration_count=n_iterations,
            batch_size=batch_size,
            algorithm_type=workload_hint.get('algorithm_type', 
                                           self.base_optimizer.__class__.__name__.lower()),
            parallel_evaluations=workload_hint.get('parallel_evaluations', True)
        )
    
    def _select_optimal_backend(self, workload: WorkloadProfile) -> str:
        """Select optimal GPU backend for workload"""
        
        if not self.auto_backend_selection:
            # Use first available backend
            if self.enable_heavydb and self.gpu_manager.capabilities.has_heavydb:
                return 'heavydb'
            elif self.enable_cupy and self.gpu_manager.capabilities.has_cupy:
                return 'cupy'
            else:
                return 'cpu'
        
        # Intelligent backend selection
        selected = self.gpu_manager.select_backend(workload)
        
        # Override based on user preferences
        if selected == 'heavydb' and not self.enable_heavydb:
            selected = 'cupy' if self.enable_cupy and self.gpu_manager.capabilities.has_cupy else 'cpu'
        elif selected == 'cupy' and not self.enable_cupy:
            selected = 'heavydb' if self.enable_heavydb and self.gpu_manager.capabilities.has_heavydb else 'cpu'
        
        return selected
    
    def _create_accelerated_objective_function(self,
                                             original_function: Callable,
                                             backend: str) -> Callable:
        """Create GPU-accelerated objective function"""
        
        if backend == 'cpu':
            return original_function
        
        return self.gpu_manager.accelerate_objective_function(
            original_function, self.current_workload
        )
    
    def _create_gpu_callback(self, user_callback: Optional[Callable]) -> Optional[Callable]:
        """Create GPU-aware callback function"""
        
        if not self.performance_monitoring and not user_callback:
            return None
        
        def gpu_callback(iteration: int, best_params: Dict[str, float], best_score: float):
            # Update GPU performance stats
            if self.performance_monitoring:
                self._update_iteration_stats(iteration, best_score)
            
            # Call user callback
            if user_callback:
                user_callback(iteration, best_params, best_score)
        
        return gpu_callback
    
    def _update_iteration_stats(self, iteration: int, score: float):
        """Update per-iteration GPU statistics"""
        
        # Get current GPU stats
        gpu_stats = self.gpu_manager.get_performance_stats()
        
        # Store iteration metrics
        iteration_stats = {
            'iteration': iteration,
            'score': score,
            'timestamp': time.time(),
            'gpu_memory_usage': self.gpu_manager.get_memory_usage(),
            'backend_usage': gpu_stats.get('backend_usage', {})
        }
        
        # Add to performance history (keep last 100 iterations)
        self.gpu_performance_history.append(iteration_stats)
        if len(self.gpu_performance_history) > 100:
            self.gpu_performance_history.pop(0)
    
    def _calculate_gpu_metrics(self, start_time: float, backend: str) -> Dict[str, Any]:
        """Calculate GPU performance metrics"""
        
        total_time = time.time() - start_time
        gpu_stats = self.gpu_manager.get_performance_stats()
        
        # Calculate speedup estimate
        estimated_cpu_time = total_time * 2.0  # Conservative estimate
        speedup = estimated_cpu_time / total_time if total_time > 0 else 1.0
        
        # Get memory usage
        memory_usage = self.gpu_manager.get_memory_usage()
        memory_usage_mb = max(
            memory_usage.get('gpu_0_used_gb', 0) * 1024,
            memory_usage.get('system_memory_gb', 0) * 1024 * 0.1  # Estimate
        )
        
        return {
            'speedup': min(speedup, 100.0),  # Cap at 100x speedup
            'memory_usage_mb': memory_usage_mb,
            'gpu_time': gpu_stats.get('total_gpu_time', 0.0),
            'cpu_fallback_count': gpu_stats.get('cpu_fallback_evaluations', 0),
            'backend_switches': 0  # TODO: Implement backend switching tracking
        }
    
    def _update_performance_history(self, result: GPUOptimizationResult):
        """Update overall performance history"""
        
        # Update backend usage stats
        self.backend_usage_stats[result.gpu_backend_used] += 1
        
        # Store performance summary
        perf_summary = {
            'timestamp': time.time(),
            'algorithm': result.algorithm_name,
            'backend': result.gpu_backend_used,
            'speedup': result.gpu_speedup,
            'execution_time': result.execution_time,
            'best_score': result.best_score,
            'iterations': result.n_iterations,
            'parameter_count': len(self.param_space)
        }
        
        # Keep last 50 optimization runs
        if not hasattr(self, 'optimization_history'):
            self.optimization_history = []
        
        self.optimization_history.append(perf_summary)
        if len(self.optimization_history) > 50:
            self.optimization_history.pop(0)
    
    def batch_optimize(self,
                      parameter_sets_list: List[List[Dict[str, float]]],
                      n_iterations_list: List[int],
                      **kwargs) -> List[GPUOptimizationResult]:
        """
        Run batch optimization with GPU acceleration
        
        Args:
            parameter_sets_list: List of parameter set lists for each optimization
            n_iterations_list: List of iteration counts for each optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            List of GPUOptimizationResult
        """
        if len(parameter_sets_list) != len(n_iterations_list):
            raise ValueError("Length of parameter_sets_list must match n_iterations_list")
        
        logger.info(f"Starting batch optimization: {len(parameter_sets_list)} runs")
        
        results = []
        total_start_time = time.time()
        
        for i, (param_sets, n_iter) in enumerate(zip(parameter_sets_list, n_iterations_list)):
            logger.info(f"Batch optimization {i+1}/{len(parameter_sets_list)}")
            
            # Create workload hint for this run
            workload_hint = {
                'data_size': len(param_sets),
                'batch_size': min(100, len(param_sets)),
                'algorithm_type': self.base_optimizer.__class__.__name__.lower()
            }
            
            # Run optimization
            result = self.optimize(
                n_iterations=n_iter,
                workload_hint=workload_hint,
                **kwargs
            )
            
            results.append(result)
        
        total_time = time.time() - total_start_time
        logger.info(f"Batch optimization completed in {total_time:.2f}s")
        
        return results
    
    def benchmark_backends(self,
                          test_iterations: int = 100,
                          test_parameter_count: int = 5) -> Dict[str, Any]:
        """
        Benchmark different GPU backends
        
        Args:
            test_iterations: Number of test iterations
            test_parameter_count: Number of test parameters
            
        Returns:
            Benchmark results
        """
        logger.info("Starting GPU backend benchmark")
        
        benchmark_results = {
            'test_config': {
                'iterations': test_iterations,
                'parameter_count': test_parameter_count,
                'timestamp': time.time()
            },
            'backends': {}
        }
        
        # Test function for benchmarking
        def test_objective(params):
            return sum(v**2 for v in params.values())
        
        # Create test workload
        test_workload = WorkloadProfile(
            data_size=test_iterations,
            parameter_count=test_parameter_count,
            iteration_count=test_iterations,
            batch_size=50,
            algorithm_type='test',
            parallel_evaluations=True
        )
        
        # Test each available backend
        available_backends = ['cpu']
        if self.gpu_manager.capabilities.has_heavydb:
            available_backends.append('heavydb')
        if self.gpu_manager.capabilities.has_cupy:
            available_backends.append('cupy')
        
        for backend in available_backends:
            logger.info(f"Benchmarking {backend} backend")
            
            try:
                start_time = time.time()
                
                # Create accelerated objective
                if backend == 'cpu':
                    accelerated_obj = test_objective
                else:
                    accelerated_obj = self.gpu_manager.accelerate_objective_function(
                        test_objective, test_workload
                    )
                
                # Run test evaluations
                test_params_list = []
                for _ in range(test_iterations):
                    params = {f'param_{i}': np.random.random() for i in range(test_parameter_count)}
                    test_params_list.append(params)
                
                # Evaluate
                if backend in ['heavydb', 'cupy']:
                    results = self.gpu_manager.batch_evaluate(
                        test_objective, test_params_list, test_workload
                    )
                else:
                    results = [accelerated_obj(params) for params in test_params_list]
                
                execution_time = time.time() - start_time
                
                benchmark_results['backends'][backend] = {
                    'execution_time': execution_time,
                    'evaluations_per_second': test_iterations / execution_time,
                    'status': 'success',
                    'results_count': len(results)
                }
                
                logger.info(f"{backend} benchmark: {execution_time:.2f}s, "
                          f"{test_iterations/execution_time:.0f} eval/s")
                
            except Exception as e:
                logger.error(f"Benchmark failed for {backend}: {e}")
                benchmark_results['backends'][backend] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate relative performance
        if 'cpu' in benchmark_results['backends'] and benchmark_results['backends']['cpu']['status'] == 'success':
            cpu_time = benchmark_results['backends']['cpu']['execution_time']
            
            for backend, result in benchmark_results['backends'].items():
                if result['status'] == 'success' and backend != 'cpu':
                    speedup = cpu_time / result['execution_time']
                    result['speedup_vs_cpu'] = speedup
        
        logger.info("GPU backend benchmark completed")
        return benchmark_results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive GPU performance report"""
        
        gpu_stats = self.gpu_manager.get_performance_stats()
        
        report = {
            'gpu_capabilities': {
                'has_heavydb': self.gpu_manager.capabilities.has_heavydb,
                'has_cupy': self.gpu_manager.capabilities.has_cupy,
                'gpu_memory_gb': self.gpu_manager.capabilities.gpu_memory_gb,
                'gpu_count': self.gpu_manager.capabilities.gpu_count
            },
            'backend_usage': self.backend_usage_stats,
            'performance_stats': gpu_stats,
            'memory_usage': self.gpu_manager.get_memory_usage(),
            'optimization_history': getattr(self, 'optimization_history', []),
            'health_status': self.gpu_manager.health_check()
        }
        
        # Calculate summary metrics
        if self.optimization_history:
            speedups = [opt['speedup'] for opt in self.optimization_history]
            report['summary_metrics'] = {
                'average_speedup': np.mean(speedups),
                'max_speedup': np.max(speedups),
                'total_optimizations': len(self.optimization_history),
                'preferred_backend': max(self.backend_usage_stats.items(), key=lambda x: x[1])[0]
            }
        
        return report
    
    def save_performance_report(self, filepath: str):
        """Save performance report to JSON file"""
        
        report = self.get_performance_report()
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, cls=NumpyEncoder, indent=2)
        
        logger.info(f"GPU performance report saved to {filepath}")
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if hasattr(self.gpu_manager, 'heavydb_backend') and self.gpu_manager.heavydb_backend:
            self.gpu_manager.heavydb_backend.close()
        
        if hasattr(self.gpu_manager, 'cupy_backend') and self.gpu_manager.cupy_backend:
            self.gpu_manager.cupy_backend.close()
        
        logger.info("GPU optimizer resources cleaned up")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except:
            pass