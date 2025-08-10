"""
Optimization Engine - Unified Interface

Provides a single entry point for all optimization operations with intelligent
algorithm selection, performance tracking, and comprehensive result analysis.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json

from ..base.base_optimizer import BaseOptimizer, OptimizationResult
from .algorithm_registry import AlgorithmRegistry
from .algorithm_metadata import (
    AlgorithmMetadataManager, AlgorithmCategory, ProblemType, ComplexityLevel
)
from ..robust.robust_optimizer import RobustOptimizer
from ..gpu.gpu_optimizer import GPUOptimizer
from ..inversion.inversion_engine import InversionEngine

logger = logging.getLogger(__name__)

@dataclass
class OptimizationRequest:
    """Optimization request specification"""
    param_space: Dict[str, Tuple[float, float]]
    objective_function: Callable[[Dict[str, float]], float]
    algorithm_preferences: Optional[List[str]] = None
    resource_constraints: Optional[Dict[str, Any]] = None
    performance_requirements: Optional[Dict[str, Any]] = None
    optimization_mode: str = "balanced"  # 'speed', 'quality', 'balanced'
    enable_gpu: bool = True
    enable_robustness: bool = True
    enable_inversion_analysis: bool = False
    max_iterations: int = 1000
    target_improvement: Optional[float] = None
    timeout_seconds: Optional[float] = None

@dataclass
class OptimizationSummary:
    """Summary of optimization run"""
    request_id: str
    algorithm_used: str
    execution_time: float
    iterations_completed: int
    best_parameters: Dict[str, float]
    best_objective_value: float
    improvement_achieved: float
    convergence_status: str
    robustness_metrics: Optional[Dict[str, Any]] = None
    gpu_utilization: Optional[Dict[str, Any]] = None
    inversion_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchOptimizationResult:
    """Result of batch optimization"""
    total_optimizations: int
    successful_optimizations: int
    failed_optimizations: int
    best_overall_result: OptimizationSummary
    algorithm_performance: Dict[str, Dict[str, float]]
    execution_summary: Dict[str, Any]
    detailed_results: List[OptimizationSummary]

class OptimizationEngine:
    """
    Unified optimization engine with intelligent algorithm selection
    
    Features:
    - Automatic algorithm discovery and registration
    - Intelligent algorithm recommendation based on problem characteristics
    - GPU acceleration when available
    - Robust optimization with cross-validation
    - Strategy inversion analysis
    - Performance tracking and learning
    - Batch optimization support
    - Real-time monitoring and callbacks
    """
    
    def __init__(self,
                 algorithms_package: str = "strategies.optimization.algorithms",
                 metadata_file: Optional[str] = None,
                 enable_gpu: bool = True,
                 enable_parallel: bool = True,
                 max_workers: Optional[int] = None,
                 cache_results: bool = True):
        """
        Initialize optimization engine
        
        Args:
            algorithms_package: Package containing algorithm implementations
            metadata_file: Path to metadata persistence file
            enable_gpu: Enable GPU acceleration
            enable_parallel: Enable parallel optimization
            max_workers: Maximum worker threads/processes
            cache_results: Cache optimization results
        """
        self.algorithms_package = algorithms_package
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(8, (mp.cpu_count() or 1) + 4)
        self.cache_results = cache_results
        
        # Initialize components
        self.registry = AlgorithmRegistry(algorithms_package, metadata_file)
        self.metadata_manager = self.registry.metadata_manager
        
        # Specialized optimizers
        self.robust_optimizer = None
        self.gpu_optimizer = None
        self.inversion_engine = None
        
        # Performance tracking
        self.optimization_history: List[OptimizationSummary] = []
        self.algorithm_performance: Dict[str, Dict[str, float]] = {}
        self.result_cache: Dict[str, OptimizationResult] = {}
        
        # Execution state
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        self.optimization_counter = 0
        
        # Discover algorithms
        discovery_result = self.registry.discover_algorithms()
        logger.info(f"OptimizationEngine initialized with {discovery_result['total_algorithms']} algorithms")
    
    def optimize(self,
                param_space: Dict[str, Tuple[float, float]],
                objective_function: Callable[[Dict[str, float]], float],
                algorithm: Optional[str] = None,
                optimization_mode: str = "balanced",
                max_iterations: int = 1000,
                enable_gpu: Optional[bool] = None,
                enable_robustness: bool = True,
                callback: Optional[Callable] = None,
                **kwargs) -> OptimizationSummary:
        """
        Main optimization interface
        
        Args:
            param_space: Parameter space definition
            objective_function: Objective function to optimize
            algorithm: Specific algorithm to use (optional)
            optimization_mode: 'speed', 'quality', 'balanced'
            max_iterations: Maximum iterations
            enable_gpu: Override GPU setting
            enable_robustness: Enable robust optimization
            callback: Progress callback function
            **kwargs: Additional algorithm parameters
            
        Returns:
            Optimization summary with results
        """
        # Create optimization request
        request = OptimizationRequest(
            param_space=param_space,
            objective_function=objective_function,
            algorithm_preferences=[algorithm] if algorithm else None,
            optimization_mode=optimization_mode,
            enable_gpu=enable_gpu if enable_gpu is not None else self.enable_gpu,
            enable_robustness=enable_robustness,
            max_iterations=max_iterations
        )
        
        return self._execute_optimization(request, callback, **kwargs)
    
    def batch_optimize(self,
                      optimization_requests: List[OptimizationRequest],
                      parallel_execution: bool = True,
                      progress_callback: Optional[Callable] = None) -> BatchOptimizationResult:
        """
        Execute multiple optimizations in batch
        
        Args:
            optimization_requests: List of optimization requests
            parallel_execution: Execute in parallel if enabled
            progress_callback: Progress callback for batch execution
            
        Returns:
            Batch optimization results
        """
        logger.info(f"Starting batch optimization with {len(optimization_requests)} requests")
        start_time = time.time()
        
        results = []
        failed_count = 0
        
        if parallel_execution and self.enable_parallel and len(optimization_requests) > 1:
            results = self._execute_parallel_batch(optimization_requests, progress_callback)
        else:
            results = self._execute_sequential_batch(optimization_requests, progress_callback)
        
        # Analyze results
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)
        
        # Find best result
        best_result = None
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.improvement_achieved)
        
        # Algorithm performance analysis
        algorithm_performance = self._analyze_algorithm_performance(successful_results)
        
        # Execution summary
        execution_time = time.time() - start_time
        execution_summary = {
            'total_execution_time': execution_time,
            'average_time_per_optimization': execution_time / len(optimization_requests),
            'parallel_execution': parallel_execution,
            'max_workers_used': self.max_workers if parallel_execution else 1
        }
        
        return BatchOptimizationResult(
            total_optimizations=len(optimization_requests),
            successful_optimizations=len(successful_results),
            failed_optimizations=failed_count,
            best_overall_result=best_result,
            algorithm_performance=algorithm_performance,
            execution_summary=execution_summary,
            detailed_results=successful_results
        )
    
    def recommend_algorithm(self,
                          problem_characteristics: Dict[str, Any],
                          resource_constraints: Optional[Dict[str, Any]] = None,
                          performance_priority: str = "balanced") -> List[Tuple[str, float]]:
        """
        Recommend algorithms for a specific problem
        
        Args:
            problem_characteristics: Problem-specific requirements
            resource_constraints: Available computational resources
            performance_priority: 'speed', 'quality', 'balanced'
            
        Returns:
            List of (algorithm_name, score) recommendations
        """
        return self.registry.recommend_algorithms(
            problem_characteristics, resource_constraints, performance_priority
        )
    
    def benchmark_algorithms(self,
                           test_functions: List[Tuple[Dict[str, Tuple[float, float]], Callable]],
                           algorithms: Optional[List[str]] = None,
                           iterations_per_test: int = 5) -> Dict[str, Any]:
        """
        Benchmark algorithms on test functions
        
        Args:
            test_functions: List of (param_space, objective_function) tuples
            algorithms: Specific algorithms to test (all if None)
            iterations_per_test: Number of iterations per test function
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting algorithm benchmark with {len(test_functions)} test functions")
        
        algorithms_to_test = algorithms or self.registry.list_algorithms()
        benchmark_results = {
            'test_functions': len(test_functions),
            'algorithms_tested': len(algorithms_to_test),
            'iterations_per_test': iterations_per_test,
            'results': {},
            'rankings': {},
            'summary': {}
        }
        
        for algorithm in algorithms_to_test:
            algorithm_results = []
            
            for i, (param_space, objective_function) in enumerate(test_functions):
                test_name = f"test_function_{i}"
                
                # Run multiple iterations
                function_results = []
                for iteration in range(iterations_per_test):
                    try:
                        request = OptimizationRequest(
                            param_space=param_space,
                            objective_function=objective_function,
                            algorithm_preferences=[algorithm],
                            max_iterations=500
                        )
                        
                        result = self._execute_optimization(request)
                        function_results.append({
                            'best_value': result.best_objective_value,
                            'execution_time': result.execution_time,
                            'iterations': result.iterations_completed,
                            'improvement': result.improvement_achieved
                        })
                        
                    except Exception as e:
                        logger.warning(f"Benchmark failed for {algorithm} on {test_name}: {e}")
                
                # Aggregate results for this test function
                if function_results:
                    algorithm_results.append({
                        'test_name': test_name,
                        'best_value_avg': np.mean([r['best_value'] for r in function_results]),
                        'best_value_std': np.std([r['best_value'] for r in function_results]),
                        'execution_time_avg': np.mean([r['execution_time'] for r in function_results]),
                        'improvement_avg': np.mean([r['improvement'] for r in function_results]),
                        'success_rate': len(function_results) / iterations_per_test
                    })
            
            benchmark_results['results'][algorithm] = algorithm_results
        
        # Calculate rankings
        benchmark_results['rankings'] = self._calculate_benchmark_rankings(benchmark_results['results'])
        
        # Generate summary
        benchmark_results['summary'] = self._generate_benchmark_summary(benchmark_results)
        
        return benchmark_results
    
    def get_optimization_history(self, 
                               algorithm: Optional[str] = None,
                               limit: Optional[int] = None) -> List[OptimizationSummary]:
        """Get optimization history with optional filtering"""
        history = self.optimization_history
        
        if algorithm:
            history = [h for h in history if h.algorithm_used == algorithm]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        total_optimizations = len(self.optimization_history)
        
        if total_optimizations == 0:
            return {'message': 'No optimizations completed yet'}
        
        # Algorithm usage
        algorithm_usage = {}
        for result in self.optimization_history:
            alg = result.algorithm_used
            algorithm_usage[alg] = algorithm_usage.get(alg, 0) + 1
        
        # Performance metrics
        execution_times = [r.execution_time for r in self.optimization_history]
        improvements = [r.improvement_achieved for r in self.optimization_history]
        
        # Success rates
        successful_optimizations = sum(1 for r in self.optimization_history 
                                     if r.convergence_status == 'converged')
        
        return {
            'total_optimizations': total_optimizations,
            'successful_optimizations': successful_optimizations,
            'success_rate': successful_optimizations / total_optimizations,
            'algorithm_usage': algorithm_usage,
            'performance_stats': {
                'avg_execution_time': np.mean(execution_times),
                'median_execution_time': np.median(execution_times),
                'avg_improvement': np.mean(improvements),
                'best_improvement': max(improvements),
                'total_execution_time': sum(execution_times)
            },
            'registry_stats': self.registry.get_registry_statistics(),
            'cache_stats': {
                'cached_results': len(self.result_cache),
                'cache_enabled': self.cache_results
            }
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.result_cache.clear()
        self.registry.clear_cache()
        logger.info("All caches cleared")
    
    def save_optimization_history(self, file_path: str):
        """Save optimization history to file"""
        try:
            history_data = []
            for result in self.optimization_history:
                # Convert to serializable format
                history_data.append({
                    'request_id': result.request_id,
                    'algorithm_used': result.algorithm_used,
                    'execution_time': result.execution_time,
                    'iterations_completed': result.iterations_completed,
                    'best_parameters': result.best_parameters,
                    'best_objective_value': result.best_objective_value,
                    'improvement_achieved': result.improvement_achieved,
                    'convergence_status': result.convergence_status,
                    'metadata': result.metadata
                })
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            logger.info(f"Optimization history saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving optimization history: {e}")
    
    # Private methods
    
    def _execute_optimization(self,
                            request: OptimizationRequest,
                            callback: Optional[Callable] = None,
                            **kwargs) -> OptimizationSummary:
        """Execute a single optimization request"""
        self.optimization_counter += 1
        request_id = f"opt_{self.optimization_counter}_{int(time.time())}"
        
        start_time = time.time()
        logger.debug(f"Executing optimization {request_id}")
        
        try:
            # Select algorithm
            algorithm_name = self._select_algorithm(request)
            
            # Check cache
            cache_key = self._generate_cache_key(request, algorithm_name)
            if self.cache_results and cache_key in self.result_cache:
                cached_result = self.result_cache[cache_key]
                logger.debug(f"Using cached result for {request_id}")
                
                return OptimizationSummary(
                    request_id=request_id,
                    algorithm_used=algorithm_name,
                    execution_time=0.001,  # Minimal cache access time
                    iterations_completed=cached_result.iterations,
                    best_parameters=cached_result.best_parameters,
                    best_objective_value=cached_result.best_objective_value,
                    improvement_achieved=cached_result.improvement,
                    convergence_status='cached'
                )
            
            # Create optimizer instance
            optimizer = self._create_optimizer(algorithm_name, request, **kwargs)
            
            # Execute optimization
            result = self._run_optimization(optimizer, request, callback)
            
            # Post-process result
            summary = self._create_optimization_summary(
                request_id, algorithm_name, result, start_time, request
            )
            
            # Update performance tracking
            self._update_performance_tracking(algorithm_name, summary)
            
            # Cache result
            if self.cache_results:
                self.result_cache[cache_key] = result
            
            # Store in history
            self.optimization_history.append(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Optimization {request_id} failed: {e}")
            
            # Create failed summary
            execution_time = time.time() - start_time
            failed_summary = OptimizationSummary(
                request_id=request_id,
                algorithm_used="unknown",
                execution_time=execution_time,
                iterations_completed=0,
                best_parameters={},
                best_objective_value=float('inf'),
                improvement_achieved=0.0,
                convergence_status='failed',
                metadata={'error': str(e)}
            )
            
            self.optimization_history.append(failed_summary)
            return failed_summary
    
    def _select_algorithm(self, request: OptimizationRequest) -> str:
        """Select the best algorithm for the request"""
        if request.algorithm_preferences:
            # Use specified algorithm if available
            for preferred in request.algorithm_preferences:
                if preferred in self.registry.algorithm_classes:
                    return preferred
        
        # Auto-select based on problem characteristics
        problem_characteristics = {
            'dimensions': len(request.param_space),
            'problem_type': 'continuous',
            'optimization_mode': request.optimization_mode
        }
        
        recommendations = self.registry.recommend_algorithms(
            problem_characteristics,
            request.resource_constraints,
            request.optimization_mode
        )
        
        if recommendations:
            return recommendations[0][0]
        
        # Fallback to first available algorithm
        available_algorithms = self.registry.list_algorithms()
        if available_algorithms:
            return available_algorithms[0]
        
        raise RuntimeError("No algorithms available for optimization")
    
    def _create_optimizer(self,
                        algorithm_name: str,
                        request: OptimizationRequest,
                        **kwargs) -> BaseOptimizer:
        """Create optimizer instance with appropriate configuration"""
        
        # Special handling for GPU optimization
        if request.enable_gpu and self.enable_gpu:
            gpu_optimizer = self._get_gpu_optimizer()
            if gpu_optimizer:
                # Configure GPU optimizer with selected algorithm
                return gpu_optimizer
        
        # Special handling for robust optimization
        if request.enable_robustness:
            robust_optimizer = self._get_robust_optimizer()
            if robust_optimizer:
                # Configure robust optimizer with selected algorithm
                base_optimizer = self.registry.get_algorithm(
                    algorithm_name,
                    param_space=request.param_space,
                    objective_function=request.objective_function,
                    **kwargs
                )
                robust_optimizer.base_optimizer = base_optimizer
                return robust_optimizer
        
        # Standard optimizer
        return self.registry.get_algorithm(
            algorithm_name,
            param_space=request.param_space,
            objective_function=request.objective_function,
            **kwargs
        )
    
    def _run_optimization(self,
                        optimizer: BaseOptimizer,
                        request: OptimizationRequest,
                        callback: Optional[Callable] = None) -> OptimizationResult:
        """Run the actual optimization"""
        
        optimization_kwargs = {
            'n_iterations': request.max_iterations
        }
        
        if callback:
            optimization_kwargs['callback'] = callback
        
        if request.timeout_seconds:
            # TODO: Implement timeout wrapper
            pass
        
        return optimizer.optimize(**optimization_kwargs)
    
    def _create_optimization_summary(self,
                                   request_id: str,
                                   algorithm_name: str,
                                   result: OptimizationResult,
                                   start_time: float,
                                   request: OptimizationRequest) -> OptimizationSummary:
        """Create optimization summary from result"""
        
        execution_time = time.time() - start_time
        
        # Calculate improvement (assuming minimization)
        initial_value = request.objective_function(
            {k: (bounds[0] + bounds[1]) / 2 for k, bounds in request.param_space.items()}
        )
        improvement = (initial_value - result.best_objective_value) / abs(initial_value + 1e-8)
        
        return OptimizationSummary(
            request_id=request_id,
            algorithm_used=algorithm_name,
            execution_time=execution_time,
            iterations_completed=result.iterations,
            best_parameters=result.best_parameters,
            best_objective_value=result.best_objective_value,
            improvement_achieved=improvement,
            convergence_status=result.convergence_status,
            metadata={
                'optimization_mode': request.optimization_mode,
                'gpu_enabled': request.enable_gpu,
                'robustness_enabled': request.enable_robustness,
                'param_space_size': len(request.param_space)
            }
        )
    
    def _update_performance_tracking(self, algorithm_name: str, summary: OptimizationSummary):
        """Update algorithm performance tracking"""
        self.metadata_manager.update_performance_profile(
            algorithm_name,
            summary.execution_time,
            summary.improvement_achieved,
            summary.convergence_status == 'converged'
        )
    
    def _generate_cache_key(self, request: OptimizationRequest, algorithm_name: str) -> str:
        """Generate cache key for optimization request"""
        import hashlib
        
        # Create deterministic hash of request parameters
        key_data = {
            'algorithm': algorithm_name,
            'param_space': sorted(request.param_space.items()),
            'max_iterations': request.max_iterations,
            'optimization_mode': request.optimization_mode
        }
        
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_gpu_optimizer(self) -> Optional[GPUOptimizer]:
        """Get or create GPU optimizer"""
        if self.gpu_optimizer is None:
            try:
                from ..gpu.gpu_optimizer import GPUOptimizer
                # Default param space and objective for initialization
                default_param_space = {'x': (-1.0, 1.0)}
                default_objective = lambda params: sum(v**2 for v in params.values())
                
                self.gpu_optimizer = GPUOptimizer(
                    param_space=default_param_space,
                    objective_function=default_objective
                )
            except Exception as e:
                logger.warning(f"GPU optimizer not available: {e}")
                return None
        
        return self.gpu_optimizer
    
    def _get_robust_optimizer(self) -> Optional[RobustOptimizer]:
        """Get or create robust optimizer"""
        if self.robust_optimizer is None:
            try:
                from ..robust.robust_optimizer import RobustOptimizer
                # Will be configured with actual optimizer later
                self.robust_optimizer = RobustOptimizer(
                    base_optimizer=None,
                    cv_folds=3,
                    noise_levels=[0.01, 0.05, 0.1]
                )
            except Exception as e:
                logger.warning(f"Robust optimizer not available: {e}")
                return None
        
        return self.robust_optimizer
    
    def _execute_parallel_batch(self,
                              requests: List[OptimizationRequest],
                              progress_callback: Optional[Callable] = None) -> List[OptimizationSummary]:
        """Execute batch optimization in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all optimization tasks
            futures = {
                executor.submit(self._execute_optimization, request): i
                for i, request in enumerate(requests)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                request_index = futures[future]
                try:
                    result = future.result()
                    results.append((request_index, result))
                    
                    if progress_callback:
                        progress_callback(len(results), len(requests), result)
                        
                except Exception as e:
                    logger.error(f"Parallel optimization failed for request {request_index}: {e}")
                    results.append((request_index, None))
        
        # Sort results by original request order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _execute_sequential_batch(self,
                                requests: List[OptimizationRequest],
                                progress_callback: Optional[Callable] = None) -> List[OptimizationSummary]:
        """Execute batch optimization sequentially"""
        results = []
        
        for i, request in enumerate(requests):
            try:
                result = self._execute_optimization(request)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(requests), result)
                    
            except Exception as e:
                logger.error(f"Sequential optimization failed for request {i}: {e}")
                results.append(None)
        
        return results
    
    def _analyze_algorithm_performance(self, results: List[OptimizationSummary]) -> Dict[str, Dict[str, float]]:
        """Analyze algorithm performance from batch results"""
        algorithm_stats = {}
        
        for result in results:
            alg = result.algorithm_used
            if alg not in algorithm_stats:
                algorithm_stats[alg] = {
                    'count': 0,
                    'total_time': 0.0,
                    'total_improvement': 0.0,
                    'success_count': 0
                }
            
            stats = algorithm_stats[alg]
            stats['count'] += 1
            stats['total_time'] += result.execution_time
            stats['total_improvement'] += result.improvement_achieved
            
            if result.convergence_status == 'converged':
                stats['success_count'] += 1
        
        # Calculate averages
        performance = {}
        for alg, stats in algorithm_stats.items():
            performance[alg] = {
                'avg_execution_time': stats['total_time'] / stats['count'],
                'avg_improvement': stats['total_improvement'] / stats['count'],
                'success_rate': stats['success_count'] / stats['count'],
                'total_uses': stats['count']
            }
        
        return performance
    
    def _calculate_benchmark_rankings(self, results: Dict[str, List[Dict]]) -> Dict[str, List[Tuple[str, float]]]:
        """Calculate algorithm rankings from benchmark results"""
        rankings = {}
        
        metrics = ['best_value_avg', 'execution_time_avg', 'improvement_avg', 'success_rate']
        
        for metric in metrics:
            metric_scores = []
            
            for algorithm, algorithm_results in results.items():
                if algorithm_results:
                    # Average across all test functions
                    scores = [r.get(metric, 0) for r in algorithm_results]
                    avg_score = np.mean(scores)
                    metric_scores.append((algorithm, avg_score))
            
            # Sort based on metric (lower is better for time, higher for others)
            reverse = metric != 'execution_time_avg'
            metric_scores.sort(key=lambda x: x[1], reverse=reverse)
            rankings[metric] = metric_scores
        
        return rankings
    
    def _generate_benchmark_summary(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary"""
        results = benchmark_data['results']
        rankings = benchmark_data['rankings']
        
        # Overall best algorithm (weighted score)
        algorithm_scores = {}
        weights = {
            'improvement_avg': 0.4,
            'success_rate': 0.3,
            'execution_time_avg': 0.2,  # Lower is better
            'best_value_avg': 0.1
        }
        
        for algorithm in results.keys():
            score = 0.0
            for metric, weight in weights.items():
                if metric in rankings:
                    # Find algorithm position in ranking
                    ranking = rankings[metric]
                    for i, (alg, _) in enumerate(ranking):
                        if alg == algorithm:
                            # Convert position to score (1st place = highest score)
                            position_score = (len(ranking) - i) / len(ranking)
                            if metric == 'execution_time_avg':
                                position_score = 1 - position_score  # Invert for time
                            score += weight * position_score
                            break
            
            algorithm_scores[algorithm] = score
        
        # Sort by overall score
        best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1]) if algorithm_scores else ("none", 0.0)
        
        return {
            'best_overall_algorithm': best_algorithm[0],
            'best_overall_score': best_algorithm[1],
            'algorithm_scores': dict(sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)),
            'total_benchmarks_run': sum(len(results[alg]) for alg in results) * benchmark_data['iterations_per_test']
        }

# Import multiprocessing at module level to avoid issues
try:
    import multiprocessing as mp
except ImportError:
    mp = None