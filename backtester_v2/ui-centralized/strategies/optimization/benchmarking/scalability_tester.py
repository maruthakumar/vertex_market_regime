"""
Scalability Tester

Tests algorithm scalability across different problem dimensions,
iteration counts, and computational resources to identify
performance bottlenecks and scaling characteristics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class ScalabilityMetrics:
    """Metrics for algorithm scalability testing"""
    algorithm_name: str
    problem_name: str
    dimension: int
    iterations: int
    
    # Performance metrics
    execution_time: float
    memory_usage_mb: float
    cpu_utilization: float
    function_evaluations: int
    
    # Solution quality
    best_objective_value: float
    convergence_achieved: bool
    
    # Scalability indicators
    time_per_dimension: float
    time_per_iteration: float
    memory_per_dimension: float
    evaluations_per_second: float
    
    # Resource efficiency
    resource_efficiency: float  # Quality / (Time * Memory)
    computational_complexity: str  # O(n), O(n^2), etc.
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalabilityAnalysis:
    """Analysis of algorithm scalability characteristics"""
    algorithm_name: str
    
    # Scaling relationships
    dimension_scaling: Dict[str, Any]  # Time vs dimension relationship
    iteration_scaling: Dict[str, Any]  # Time vs iteration relationship
    memory_scaling: Dict[str, Any]    # Memory vs dimension relationship
    
    # Performance boundaries
    max_feasible_dimension: int
    max_feasible_iterations: int
    memory_limit_dimension: int
    time_limit_dimension: int
    
    # Efficiency analysis
    optimal_dimension_range: Tuple[int, int]
    efficiency_plateau: Optional[int]
    scalability_score: float  # 0-1 score
    
    # Bottleneck identification
    primary_bottleneck: str  # 'memory', 'computation', 'convergence'
    scaling_coefficient: float
    complexity_class: str
    
    # Recommendations
    recommendations: List[str]
    optimization_suggestions: List[str]


class ScalabilityTester:
    """
    Comprehensive scalability testing for optimization algorithms
    
    Tests algorithms across different dimensions, iteration counts,
    and resource constraints to characterize scaling behavior.
    """
    
    def __init__(self,
                 max_memory_mb: int = 8192,
                 max_execution_time: int = 600,
                 dimension_range: Tuple[int, int] = (2, 100),
                 iteration_range: Tuple[int, int] = (10, 1000),
                 enable_parallel_testing: bool = True):
        """
        Initialize scalability tester
        
        Args:
            max_memory_mb: Maximum memory usage allowed
            max_execution_time: Maximum execution time in seconds
            dimension_range: Range of dimensions to test
            iteration_range: Range of iterations to test
            enable_parallel_testing: Enable parallel test execution
        """
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time
        self.dimension_range = dimension_range
        self.iteration_range = iteration_range
        self.enable_parallel_testing = enable_parallel_testing
        
        # Test results storage
        self.scalability_data: Dict[str, List[ScalabilityMetrics]] = {}
        self.baseline_performance: Dict[str, ScalabilityMetrics] = {}
        
        # System monitoring
        self.process = psutil.Process()
        
        logger.info("ScalabilityTester initialized")
    
    def test_dimension_scaling(self,
                             algorithm_factory: Callable,
                             problem_factory: Callable,
                             dimensions: List[int],
                             fixed_iterations: int = 100,
                             runs_per_dimension: int = 3) -> Dict[str, List[ScalabilityMetrics]]:
        """
        Test algorithm scaling with problem dimension
        
        Args:
            algorithm_factory: Function that creates algorithm instance
            problem_factory: Function that creates problem instance
            dimensions: List of dimensions to test
            fixed_iterations: Fixed number of iterations for each test
            runs_per_dimension: Number of runs per dimension
            
        Returns:
            Scalability metrics for each dimension
        """
        logger.info(f"Testing dimension scaling from {min(dimensions)} to {max(dimensions)} dimensions")
        
        results = {}
        
        for dim in dimensions:
            logger.info(f"Testing dimension: {dim}")
            
            # Create problem and algorithm for this dimension
            try:
                problem = problem_factory(dim)
                
                dim_results = []
                
                for run in range(runs_per_dimension):
                    algorithm = algorithm_factory(problem.param_space, problem.objective_function)
                    
                    # Run scalability test
                    metrics = self._run_scalability_test(
                        algorithm, problem, dim, fixed_iterations, f"dimension_{dim}_run_{run}"
                    )
                    
                    if metrics:
                        dim_results.append(metrics)
                        
                        # Check resource limits
                        if (metrics.memory_usage_mb > self.max_memory_mb or 
                            metrics.execution_time > self.max_execution_time):
                            logger.warning(f"Resource limits exceeded at dimension {dim}")
                            break
                
                if dim_results:
                    results[f"dimension_{dim}"] = dim_results
                else:
                    logger.warning(f"No successful runs for dimension {dim}")
                    break
                    
            except Exception as e:
                logger.error(f"Error testing dimension {dim}: {e}")
                break
        
        return results
    
    def test_iteration_scaling(self,
                             algorithm_factory: Callable,
                             problem_factory: Callable,
                             iterations_list: List[int],
                             fixed_dimension: int = 10,
                             runs_per_iteration: int = 3) -> Dict[str, List[ScalabilityMetrics]]:
        """
        Test algorithm scaling with iteration count
        
        Args:
            algorithm_factory: Function that creates algorithm instance
            problem_factory: Function that creates problem instance
            iterations_list: List of iteration counts to test
            fixed_dimension: Fixed problem dimension
            runs_per_iteration: Number of runs per iteration count
            
        Returns:
            Scalability metrics for each iteration count
        """
        logger.info(f"Testing iteration scaling from {min(iterations_list)} to {max(iterations_list)} iterations")
        
        results = {}
        
        # Create problem once
        problem = problem_factory(fixed_dimension)
        
        for iterations in iterations_list:
            logger.info(f"Testing iterations: {iterations}")
            
            iter_results = []
            
            for run in range(runs_per_iteration):
                algorithm = algorithm_factory(problem.param_space, problem.objective_function)
                
                # Run scalability test
                metrics = self._run_scalability_test(
                    algorithm, problem, fixed_dimension, iterations, f"iterations_{iterations}_run_{run}"
                )
                
                if metrics:
                    iter_results.append(metrics)
                    
                    # Check time limits
                    if metrics.execution_time > self.max_execution_time:
                        logger.warning(f"Time limit exceeded at {iterations} iterations")
                        break
            
            if iter_results:
                results[f"iterations_{iterations}"] = iter_results
            else:
                logger.warning(f"No successful runs for {iterations} iterations")
                break
        
        return results
    
    def test_memory_scaling(self,
                          algorithm_factory: Callable,
                          problem_factory: Callable,
                          dimensions: List[int],
                          memory_monitoring_interval: float = 0.1) -> Dict[str, Any]:
        """
        Test memory usage scaling with problem dimension
        
        Args:
            algorithm_factory: Function that creates algorithm instance
            problem_factory: Function that creates problem instance
            dimensions: List of dimensions to test
            memory_monitoring_interval: Memory monitoring interval in seconds
            
        Returns:
            Memory scaling analysis
        """
        logger.info("Testing memory scaling")
        
        memory_profiles = {}
        peak_memory_usage = {}
        
        for dim in dimensions:
            logger.info(f"Testing memory usage for dimension: {dim}")
            
            try:
                problem = problem_factory(dim)
                algorithm = algorithm_factory(problem.param_space, problem.objective_function)
                
                # Monitor memory during optimization
                memory_profile = self._monitor_memory_usage(
                    algorithm, problem, dim, 50, memory_monitoring_interval
                )
                
                if memory_profile:
                    memory_profiles[dim] = memory_profile
                    peak_memory_usage[dim] = max(memory_profile['memory_usage'])
                    
                    # Check memory limits
                    if peak_memory_usage[dim] > self.max_memory_mb:
                        logger.warning(f"Memory limit exceeded at dimension {dim}")
                        break
                        
            except Exception as e:
                logger.error(f"Error in memory testing for dimension {dim}: {e}")
                break
        
        return {
            'memory_profiles': memory_profiles,
            'peak_memory_usage': peak_memory_usage,
            'memory_scaling_analysis': self._analyze_memory_scaling(peak_memory_usage)
        }
    
    def test_parallel_scaling(self,
                            algorithm_factory: Callable,
                            problem_factory: Callable,
                            worker_counts: List[int],
                            fixed_dimension: int = 20,
                            fixed_iterations: int = 100) -> Dict[str, Any]:
        """
        Test parallel scaling efficiency
        
        Args:
            algorithm_factory: Function that creates algorithm instance
            problem_factory: Function that creates problem instance
            worker_counts: List of worker counts to test
            fixed_dimension: Fixed problem dimension
            fixed_iterations: Fixed iteration count
            
        Returns:
            Parallel scaling analysis
        """
        logger.info("Testing parallel scaling efficiency")
        
        if not self.enable_parallel_testing:
            return {'error': 'Parallel testing disabled'}
        
        results = {}
        baseline_time = None
        
        for workers in worker_counts:
            logger.info(f"Testing with {workers} workers")
            
            try:
                # Run multiple instances in parallel
                execution_times = []
                
                for run in range(3):  # Multiple runs for reliability
                    start_time = time.time()
                    
                    # Create multiple optimization tasks
                    tasks = []
                    for i in range(workers):
                        problem = problem_factory(fixed_dimension)
                        algorithm = algorithm_factory(problem.param_space, problem.objective_function)
                        tasks.append((algorithm, problem, fixed_iterations))
                    
                    # Execute in parallel
                    if workers == 1:
                        # Sequential execution
                        for algorithm, problem, iterations in tasks:
                            algorithm.optimize(n_iterations=iterations)
                    else:
                        # Parallel execution
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            futures = []
                            for algorithm, problem, iterations in tasks:
                                future = executor.submit(algorithm.optimize, n_iterations=iterations)
                                futures.append(future)
                            
                            # Wait for completion
                            for future in futures:
                                future.result()
                    
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                
                avg_time = np.mean(execution_times)
                results[workers] = {
                    'execution_time': avg_time,
                    'execution_times': execution_times,
                    'speedup': baseline_time / avg_time if baseline_time else 1.0,
                    'efficiency': (baseline_time / avg_time) / workers if baseline_time else 1.0
                }
                
                if baseline_time is None:
                    baseline_time = avg_time
                    
            except Exception as e:
                logger.error(f"Error in parallel testing with {workers} workers: {e}")
                results[workers] = {'error': str(e)}
        
        return {
            'parallel_results': results,
            'parallel_analysis': self._analyze_parallel_scaling(results)
        }
    
    def analyze_scalability(self,
                          algorithm_name: str,
                          scalability_data: Dict[str, List[ScalabilityMetrics]]) -> ScalabilityAnalysis:
        """
        Comprehensive scalability analysis
        
        Args:
            algorithm_name: Name of the algorithm
            scalability_data: Collected scalability metrics
            
        Returns:
            Comprehensive scalability analysis
        """
        logger.info(f"Analyzing scalability for {algorithm_name}")
        
        # Extract dimension scaling data
        dimension_data = self._extract_dimension_scaling_data(scalability_data)
        iteration_data = self._extract_iteration_scaling_data(scalability_data)
        
        # Analyze scaling relationships
        dimension_scaling = self._analyze_dimension_scaling(dimension_data)
        iteration_scaling = self._analyze_iteration_scaling(iteration_data)
        memory_scaling = self._analyze_memory_scaling_from_data(dimension_data)
        
        # Determine performance boundaries
        max_feasible_dim = self._find_max_feasible_dimension(dimension_data)
        max_feasible_iter = self._find_max_feasible_iterations(iteration_data)
        memory_limit_dim = self._find_memory_limit_dimension(dimension_data)
        time_limit_dim = self._find_time_limit_dimension(dimension_data)
        
        # Efficiency analysis
        optimal_range = self._find_optimal_dimension_range(dimension_data)
        efficiency_plateau = self._find_efficiency_plateau(dimension_data)
        scalability_score = self._calculate_scalability_score(dimension_scaling, iteration_scaling)
        
        # Bottleneck identification
        bottleneck_analysis = self._identify_bottlenecks(dimension_data, iteration_data)
        
        # Generate recommendations
        recommendations = self._generate_scalability_recommendations(
            dimension_scaling, iteration_scaling, bottleneck_analysis
        )
        
        return ScalabilityAnalysis(
            algorithm_name=algorithm_name,
            dimension_scaling=dimension_scaling,
            iteration_scaling=iteration_scaling,
            memory_scaling=memory_scaling,
            max_feasible_dimension=max_feasible_dim,
            max_feasible_iterations=max_feasible_iter,
            memory_limit_dimension=memory_limit_dim,
            time_limit_dimension=time_limit_dim,
            optimal_dimension_range=optimal_range,
            efficiency_plateau=efficiency_plateau,
            scalability_score=scalability_score,
            primary_bottleneck=bottleneck_analysis['primary_bottleneck'],
            scaling_coefficient=bottleneck_analysis['scaling_coefficient'],
            complexity_class=bottleneck_analysis['complexity_class'],
            recommendations=recommendations['recommendations'],
            optimization_suggestions=recommendations['optimization_suggestions']
        )
    
    def compare_scalability(self,
                          algorithms_data: Dict[str, Dict[str, List[ScalabilityMetrics]]]) -> Dict[str, Any]:
        """
        Compare scalability across multiple algorithms
        
        Args:
            algorithms_data: Scalability data for multiple algorithms
            
        Returns:
            Comparative scalability analysis
        """
        logger.info(f"Comparing scalability across {len(algorithms_data)} algorithms")
        
        comparisons = {
            'algorithm_rankings': {},
            'scaling_characteristics': {},
            'bottleneck_comparison': {},
            'efficiency_comparison': {},
            'recommendations': {}
        }
        
        # Analyze each algorithm
        algorithm_analyses = {}
        for alg_name, data in algorithms_data.items():
            try:
                analysis = self.analyze_scalability(alg_name, data)
                algorithm_analyses[alg_name] = analysis
            except Exception as e:
                logger.error(f"Error analyzing {alg_name}: {e}")
                continue
        
        if not algorithm_analyses:
            return {'error': 'No successful algorithm analyses'}
        
        # Compare scaling characteristics
        comparisons['scaling_characteristics'] = self._compare_scaling_characteristics(algorithm_analyses)
        
        # Rank algorithms by different criteria
        comparisons['algorithm_rankings'] = {
            'overall_scalability': self._rank_by_scalability_score(algorithm_analyses),
            'dimension_scaling': self._rank_by_dimension_scaling(algorithm_analyses),
            'memory_efficiency': self._rank_by_memory_efficiency(algorithm_analyses),
            'time_efficiency': self._rank_by_time_efficiency(algorithm_analyses)
        }
        
        # Compare bottlenecks
        comparisons['bottleneck_comparison'] = self._compare_bottlenecks(algorithm_analyses)
        
        # Efficiency comparison
        comparisons['efficiency_comparison'] = self._compare_efficiency(algorithm_analyses)
        
        # Generate comparative recommendations
        comparisons['recommendations'] = self._generate_comparative_recommendations(algorithm_analyses)
        
        return comparisons
    
    # Private helper methods
    
    def _run_scalability_test(self,
                            algorithm,
                            problem,
                            dimension: int,
                            iterations: int,
                            test_id: str) -> Optional[ScalabilityMetrics]:
        """Run single scalability test and collect metrics"""
        try:
            # Monitor system resources
            initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_count = psutil.cpu_count()
            
            # Execute optimization
            start_time = time.time()
            result = algorithm.optimize(n_iterations=iterations)
            execution_time = time.time() - start_time
            
            # Final resource usage
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            # CPU utilization (approximate)
            cpu_utilization = psutil.cpu_percent()
            
            # Function evaluations
            function_evaluations = getattr(result, 'function_evaluations', iterations)
            
            # Calculate derived metrics
            time_per_dimension = execution_time / dimension if dimension > 0 else 0
            time_per_iteration = execution_time / iterations if iterations > 0 else 0
            memory_per_dimension = memory_usage / dimension if dimension > 0 else 0
            evaluations_per_second = function_evaluations / execution_time if execution_time > 0 else 0
            
            # Resource efficiency
            quality = 1.0 / (abs(result.best_objective_value) + 1) if hasattr(result, 'best_objective_value') else 0.5
            resource_efficiency = quality / (execution_time * (memory_usage + 1))
            
            return ScalabilityMetrics(
                algorithm_name=algorithm.__class__.__name__,
                problem_name=getattr(problem, 'name', 'unknown'),
                dimension=dimension,
                iterations=iterations,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_utilization=cpu_utilization,
                function_evaluations=function_evaluations,
                best_objective_value=getattr(result, 'best_objective_value', float('inf')),
                convergence_achieved=getattr(result, 'convergence_status', '') == 'converged',
                time_per_dimension=time_per_dimension,
                time_per_iteration=time_per_iteration,
                memory_per_dimension=memory_per_dimension,
                evaluations_per_second=evaluations_per_second,
                resource_efficiency=resource_efficiency,
                computational_complexity='unknown',
                metadata={'test_id': test_id}
            )
            
        except Exception as e:
            logger.error(f"Error in scalability test {test_id}: {e}")
            return None
    
    def _monitor_memory_usage(self,
                            algorithm,
                            problem,
                            dimension: int,
                            iterations: int,
                            monitoring_interval: float) -> Optional[Dict[str, List]]:
        """Monitor memory usage during optimization"""
        memory_profile = {
            'timestamps': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        
        # Flag to stop monitoring
        optimization_finished = threading.Event()
        
        def monitor():
            start_time = time.time()
            while not optimization_finished.is_set():
                current_time = time.time() - start_time
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                memory_profile['timestamps'].append(current_time)
                memory_profile['memory_usage'].append(memory_mb)
                memory_profile['cpu_usage'].append(cpu_percent)
                
                time.sleep(monitoring_interval)
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        
        try:
            # Run optimization
            algorithm.optimize(n_iterations=iterations)
        except Exception as e:
            logger.error(f"Error during monitored optimization: {e}")
            return None
        finally:
            # Stop monitoring
            optimization_finished.set()
            monitor_thread.join(timeout=1.0)
        
        return memory_profile
    
    def _extract_dimension_scaling_data(self, scalability_data):
        """Extract dimension scaling data from results"""
        dimension_data = {}
        
        for key, metrics_list in scalability_data.items():
            if key.startswith('dimension_'):
                dim = int(key.split('_')[1])
                if metrics_list:
                    # Average across runs
                    avg_metrics = self._average_metrics(metrics_list)
                    dimension_data[dim] = avg_metrics
        
        return dimension_data
    
    def _extract_iteration_scaling_data(self, scalability_data):
        """Extract iteration scaling data from results"""
        iteration_data = {}
        
        for key, metrics_list in scalability_data.items():
            if key.startswith('iterations_'):
                iterations = int(key.split('_')[1])
                if metrics_list:
                    # Average across runs
                    avg_metrics = self._average_metrics(metrics_list)
                    iteration_data[iterations] = avg_metrics
        
        return iteration_data
    
    def _average_metrics(self, metrics_list: List[ScalabilityMetrics]) -> ScalabilityMetrics:
        """Average metrics across multiple runs"""
        if not metrics_list:
            raise ValueError("Empty metrics list")
        
        if len(metrics_list) == 1:
            return metrics_list[0]
        
        # Calculate averages
        template = metrics_list[0]
        
        return ScalabilityMetrics(
            algorithm_name=template.algorithm_name,
            problem_name=template.problem_name,
            dimension=template.dimension,
            iterations=template.iterations,
            execution_time=np.mean([m.execution_time for m in metrics_list]),
            memory_usage_mb=np.mean([m.memory_usage_mb for m in metrics_list]),
            cpu_utilization=np.mean([m.cpu_utilization for m in metrics_list]),
            function_evaluations=int(np.mean([m.function_evaluations for m in metrics_list])),
            best_objective_value=np.mean([m.best_objective_value for m in metrics_list]),
            convergence_achieved=sum(m.convergence_achieved for m in metrics_list) > len(metrics_list) / 2,
            time_per_dimension=np.mean([m.time_per_dimension for m in metrics_list]),
            time_per_iteration=np.mean([m.time_per_iteration for m in metrics_list]),
            memory_per_dimension=np.mean([m.memory_per_dimension for m in metrics_list]),
            evaluations_per_second=np.mean([m.evaluations_per_second for m in metrics_list]),
            resource_efficiency=np.mean([m.resource_efficiency for m in metrics_list]),
            computational_complexity=template.computational_complexity,
            metadata={'averaged_runs': len(metrics_list)}
        )
    
    def _analyze_dimension_scaling(self, dimension_data):
        """Analyze dimension scaling relationship"""
        if not dimension_data:
            return {'error': 'No dimension data available'}
        
        dimensions = sorted(dimension_data.keys())
        times = [dimension_data[d].execution_time for d in dimensions]
        memories = [dimension_data[d].memory_usage_mb for d in dimensions]
        
        # Fit scaling relationship
        try:
            from scipy.optimize import curve_fit
            
            # Try different scaling models
            def linear(x, a, b):
                return a * x + b
            
            def quadratic(x, a, b, c):
                return a * x**2 + b * x + c
            
            def exponential(x, a, b, c):
                return a * np.exp(b * x) + c
            
            models = {
                'linear': linear,
                'quadratic': quadratic,
                'exponential': exponential
            }
            
            best_model = None
            best_r2 = -np.inf
            best_params = None
            
            x_data = np.array(dimensions)
            y_data = np.array(times)
            
            for model_name, model_func in models.items():
                try:
                    popt, _ = curve_fit(model_func, x_data, y_data, maxfev=1000)
                    y_pred = model_func(x_data, *popt)
                    r2 = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
                        best_params = popt
                except:
                    continue
            
            return {
                'dimensions': dimensions,
                'execution_times': times,
                'memory_usage': memories,
                'best_fit_model': best_model,
                'best_fit_r2': best_r2,
                'best_fit_params': best_params.tolist() if best_params is not None else None,
                'scaling_factor': times[-1] / times[0] if len(times) > 1 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error in dimension scaling analysis: {e}")
            return {
                'dimensions': dimensions,
                'execution_times': times,
                'memory_usage': memories,
                'error': str(e)
            }
    
    def _analyze_iteration_scaling(self, iteration_data):
        """Analyze iteration scaling relationship"""
        if not iteration_data:
            return {'error': 'No iteration data available'}
        
        iterations = sorted(iteration_data.keys())
        times = [iteration_data[i].execution_time for i in iterations]
        
        # Should be approximately linear
        if len(iterations) > 1:
            slope = (times[-1] - times[0]) / (iterations[-1] - iterations[0])
            linearity_score = 1.0 - np.std(np.diff(times)) / np.mean(np.diff(times)) if np.mean(np.diff(times)) > 0 else 0
        else:
            slope = 0
            linearity_score = 1.0
        
        return {
            'iterations': iterations,
            'execution_times': times,
            'slope': slope,
            'linearity_score': linearity_score,
            'scaling_efficiency': 1.0 / slope if slope > 0 else float('inf')
        }
    
    def _analyze_memory_scaling(self, peak_memory_usage):
        """Analyze memory scaling relationship"""
        if not peak_memory_usage:
            return {'error': 'No memory data available'}
        
        dimensions = sorted(peak_memory_usage.keys())
        memory_values = [peak_memory_usage[d] for d in dimensions]
        
        # Fit memory scaling
        if len(dimensions) > 1:
            memory_slope = (memory_values[-1] - memory_values[0]) / (dimensions[-1] - dimensions[0])
            per_dimension_memory = memory_slope
        else:
            memory_slope = 0
            per_dimension_memory = memory_values[0] / dimensions[0] if dimensions[0] > 0 else 0
        
        return {
            'dimensions': dimensions,
            'memory_usage': memory_values,
            'memory_slope': memory_slope,
            'memory_per_dimension': per_dimension_memory,
            'memory_efficiency': 1.0 / per_dimension_memory if per_dimension_memory > 0 else float('inf')
        }
    
    def _analyze_memory_scaling_from_data(self, dimension_data):
        """Analyze memory scaling from dimension data"""
        if not dimension_data:
            return {'error': 'No dimension data available'}
        
        peak_memory = {d: metrics.memory_usage_mb for d, metrics in dimension_data.items()}
        return self._analyze_memory_scaling(peak_memory)
    
    def _find_max_feasible_dimension(self, dimension_data):
        """Find maximum feasible dimension within resource limits"""
        if not dimension_data:
            return 0
        
        max_dim = 0
        for dim, metrics in dimension_data.items():
            if (metrics.execution_time <= self.max_execution_time and 
                metrics.memory_usage_mb <= self.max_memory_mb):
                max_dim = max(max_dim, dim)
        
        return max_dim
    
    def _find_max_feasible_iterations(self, iteration_data):
        """Find maximum feasible iterations within time limits"""
        if not iteration_data:
            return 0
        
        max_iter = 0
        for iterations, metrics in iteration_data.items():
            if metrics.execution_time <= self.max_execution_time:
                max_iter = max(max_iter, iterations)
        
        return max_iter
    
    def _find_memory_limit_dimension(self, dimension_data):
        """Find dimension where memory limit is reached"""
        if not dimension_data:
            return 0
        
        for dim in sorted(dimension_data.keys(), reverse=True):
            if dimension_data[dim].memory_usage_mb <= self.max_memory_mb:
                return dim
        
        return 0
    
    def _find_time_limit_dimension(self, dimension_data):
        """Find dimension where time limit is reached"""
        if not dimension_data:
            return 0
        
        for dim in sorted(dimension_data.keys(), reverse=True):
            if dimension_data[dim].execution_time <= self.max_execution_time:
                return dim
        
        return 0
    
    def _find_optimal_dimension_range(self, dimension_data):
        """Find optimal dimension range for efficiency"""
        if not dimension_data:
            return (0, 0)
        
        # Find range where resource efficiency is highest
        efficiencies = [(d, metrics.resource_efficiency) for d, metrics in dimension_data.items()]
        efficiencies.sort(key=lambda x: x[1], reverse=True)
        
        if efficiencies:
            # Take top 50% by efficiency
            top_half = efficiencies[:max(1, len(efficiencies) // 2)]
            dimensions = [d for d, _ in top_half]
            return (min(dimensions), max(dimensions))
        
        return (0, 0)
    
    def _find_efficiency_plateau(self, dimension_data):
        """Find dimension where efficiency plateaus"""
        if len(dimension_data) < 3:
            return None
        
        dimensions = sorted(dimension_data.keys())
        efficiencies = [dimension_data[d].resource_efficiency for d in dimensions]
        
        # Look for plateau (small changes in efficiency)
        for i in range(2, len(efficiencies)):
            recent_change = abs(efficiencies[i] - efficiencies[i-1])
            earlier_change = abs(efficiencies[i-1] - efficiencies[i-2])
            
            if recent_change < earlier_change * 0.1:  # 90% reduction in change
                return dimensions[i]
        
        return None
    
    def _calculate_scalability_score(self, dimension_scaling, iteration_scaling):
        """Calculate overall scalability score (0-1)"""
        score = 0.5  # Base score
        
        # Dimension scaling contribution
        if 'best_fit_r2' in dimension_scaling and dimension_scaling['best_fit_r2'] is not None:
            # Better fit = more predictable scaling
            score += 0.2 * dimension_scaling['best_fit_r2']
            
            # Prefer linear or sub-quadratic scaling
            if dimension_scaling['best_fit_model'] == 'linear':
                score += 0.15
            elif dimension_scaling['best_fit_model'] == 'quadratic':
                score += 0.05
        
        # Iteration scaling contribution
        if 'linearity_score' in iteration_scaling:
            score += 0.15 * iteration_scaling['linearity_score']
        
        return min(1.0, max(0.0, score))
    
    def _identify_bottlenecks(self, dimension_data, iteration_data):
        """Identify primary performance bottlenecks"""
        bottlenecks = {
            'primary_bottleneck': 'unknown',
            'scaling_coefficient': 1.0,
            'complexity_class': 'unknown'
        }
        
        if not dimension_data:
            return bottlenecks
        
        # Analyze resource usage patterns
        dimensions = sorted(dimension_data.keys())
        if len(dimensions) > 1:
            time_growth = dimension_data[dimensions[-1]].execution_time / dimension_data[dimensions[0]].execution_time
            memory_growth = dimension_data[dimensions[-1]].memory_usage_mb / dimension_data[dimensions[0]].memory_usage_mb
            dimension_growth = dimensions[-1] / dimensions[0]
            
            time_scaling = time_growth / dimension_growth
            memory_scaling = memory_growth / dimension_growth
            
            if memory_scaling > time_scaling * 1.5:
                bottlenecks['primary_bottleneck'] = 'memory'
                bottlenecks['scaling_coefficient'] = memory_scaling
            else:
                bottlenecks['primary_bottleneck'] = 'computation'
                bottlenecks['scaling_coefficient'] = time_scaling
            
            # Estimate complexity class
            if time_scaling < 1.5:
                bottlenecks['complexity_class'] = 'O(n)'
            elif time_scaling < 3.0:
                bottlenecks['complexity_class'] = 'O(n log n)'
            elif time_scaling < 5.0:
                bottlenecks['complexity_class'] = 'O(n^2)'
            else:
                bottlenecks['complexity_class'] = 'O(n^k), k > 2'
        
        return bottlenecks
    
    def _generate_scalability_recommendations(self, dimension_scaling, iteration_scaling, bottleneck_analysis):
        """Generate scalability recommendations"""
        recommendations = []
        optimization_suggestions = []
        
        # Based on bottleneck analysis
        if bottleneck_analysis['primary_bottleneck'] == 'memory':
            recommendations.append("Memory is the primary bottleneck - consider memory-efficient algorithms")
            optimization_suggestions.append("Implement memory pooling or streaming algorithms")
            optimization_suggestions.append("Use sparse data structures where applicable")
        
        elif bottleneck_analysis['primary_bottleneck'] == 'computation':
            recommendations.append("Computation time is the primary bottleneck")
            optimization_suggestions.append("Consider parallel or GPU acceleration")
            optimization_suggestions.append("Optimize inner loops and vectorize operations")
        
        # Based on complexity class
        complexity = bottleneck_analysis['complexity_class']
        if 'n^2' in complexity or 'k > 2' in complexity:
            recommendations.append("Algorithm shows poor scaling - consider alternative approaches")
            optimization_suggestions.append("Look for more efficient algorithms with better complexity")
        
        # Based on dimension scaling
        if 'best_fit_model' in dimension_scaling:
            if dimension_scaling['best_fit_model'] == 'exponential':
                recommendations.append("Exponential scaling detected - algorithm not suitable for high dimensions")
            elif dimension_scaling['best_fit_model'] == 'quadratic':
                recommendations.append("Quadratic scaling - reasonable for moderate dimensions")
        
        return {
            'recommendations': recommendations,
            'optimization_suggestions': optimization_suggestions
        }
    
    def _analyze_parallel_scaling(self, parallel_results):
        """Analyze parallel scaling efficiency"""
        if not parallel_results:
            return {'error': 'No parallel results'}
        
        worker_counts = sorted(parallel_results.keys())
        speedups = [parallel_results[w].get('speedup', 1.0) for w in worker_counts]
        efficiencies = [parallel_results[w].get('efficiency', 1.0) for w in worker_counts]
        
        # Find optimal worker count
        efficiency_scores = [(w, eff) for w, eff in zip(worker_counts, efficiencies) if eff > 0.5]
        optimal_workers = max(efficiency_scores, key=lambda x: x[1])[0] if efficiency_scores else worker_counts[0]
        
        return {
            'worker_counts': worker_counts,
            'speedups': speedups,
            'efficiencies': efficiencies,
            'optimal_worker_count': optimal_workers,
            'max_speedup': max(speedups) if speedups else 1.0,
            'parallel_efficiency': max(efficiencies) if efficiencies else 1.0
        }
    
    # Comparison methods
    
    def _compare_scaling_characteristics(self, algorithm_analyses):
        """Compare scaling characteristics across algorithms"""
        comparison = {}
        
        for alg_name, analysis in algorithm_analyses.items():
            comparison[alg_name] = {
                'scalability_score': analysis.scalability_score,
                'max_feasible_dimension': analysis.max_feasible_dimension,
                'primary_bottleneck': analysis.primary_bottleneck,
                'complexity_class': analysis.complexity_class
            }
        
        return comparison
    
    def _rank_by_scalability_score(self, algorithm_analyses):
        """Rank algorithms by overall scalability score"""
        scores = [(name, analysis.scalability_score) for name, analysis in algorithm_analyses.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return {name: rank + 1 for rank, (name, _) in enumerate(scores)}
    
    def _rank_by_dimension_scaling(self, algorithm_analyses):
        """Rank algorithms by dimension scaling capability"""
        max_dims = [(name, analysis.max_feasible_dimension) for name, analysis in algorithm_analyses.items()]
        max_dims.sort(key=lambda x: x[1], reverse=True)
        return {name: rank + 1 for rank, (name, _) in enumerate(max_dims)}
    
    def _rank_by_memory_efficiency(self, algorithm_analyses):
        """Rank algorithms by memory efficiency"""
        # Higher is better (more efficiency)
        mem_effs = []
        for name, analysis in algorithm_analyses.items():
            if 'memory_efficiency' in analysis.memory_scaling:
                eff = analysis.memory_scaling['memory_efficiency']
                # Cap at reasonable value to handle inf
                eff = min(eff, 1000.0) if eff != float('inf') else 1000.0
                mem_effs.append((name, eff))
        
        mem_effs.sort(key=lambda x: x[1], reverse=True)
        return {name: rank + 1 for rank, (name, _) in enumerate(mem_effs)}
    
    def _rank_by_time_efficiency(self, algorithm_analyses):
        """Rank algorithms by time efficiency"""
        time_effs = []
        for name, analysis in algorithm_analyses.items():
            if 'scaling_efficiency' in analysis.iteration_scaling:
                eff = analysis.iteration_scaling['scaling_efficiency']
                # Cap at reasonable value to handle inf
                eff = min(eff, 1000.0) if eff != float('inf') else 1000.0
                time_effs.append((name, eff))
        
        time_effs.sort(key=lambda x: x[1], reverse=True)
        return {name: rank + 1 for rank, (name, _) in enumerate(time_effs)}
    
    def _compare_bottlenecks(self, algorithm_analyses):
        """Compare bottleneck patterns across algorithms"""
        bottleneck_summary = {}
        
        for alg_name, analysis in algorithm_analyses.items():
            bottleneck_summary[alg_name] = {
                'primary_bottleneck': analysis.primary_bottleneck,
                'scaling_coefficient': analysis.scaling_coefficient,
                'complexity_class': analysis.complexity_class
            }
        
        return bottleneck_summary
    
    def _compare_efficiency(self, algorithm_analyses):
        """Compare efficiency metrics across algorithms"""
        efficiency_comparison = {}
        
        for alg_name, analysis in algorithm_analyses.items():
            efficiency_comparison[alg_name] = {
                'scalability_score': analysis.scalability_score,
                'max_feasible_dimension': analysis.max_feasible_dimension,
                'optimal_range': analysis.optimal_dimension_range,
                'efficiency_plateau': analysis.efficiency_plateau
            }
        
        return efficiency_comparison
    
    def _generate_comparative_recommendations(self, algorithm_analyses):
        """Generate recommendations based on comparative analysis"""
        recommendations = {
            'best_for_high_dimensions': None,
            'most_memory_efficient': None,
            'fastest_scaling': None,
            'most_reliable': None
        }
        
        # Find best algorithm for each category
        best_scalability = max(algorithm_analyses.items(), key=lambda x: x[1].scalability_score)
        recommendations['most_reliable'] = best_scalability[0]
        
        best_dimensions = max(algorithm_analyses.items(), key=lambda x: x[1].max_feasible_dimension)
        recommendations['best_for_high_dimensions'] = best_dimensions[0]
        
        # Memory efficiency (find algorithm with best memory scaling)
        best_memory = None
        best_memory_score = 0
        for name, analysis in algorithm_analyses.items():
            if 'memory_efficiency' in analysis.memory_scaling:
                eff = analysis.memory_scaling['memory_efficiency']
                if eff != float('inf') and eff > best_memory_score:
                    best_memory_score = eff
                    best_memory = name
        recommendations['most_memory_efficient'] = best_memory
        
        # Fastest scaling (best iteration scaling)
        best_speed = None
        best_speed_score = 0
        for name, analysis in algorithm_analyses.items():
            if 'scaling_efficiency' in analysis.iteration_scaling:
                eff = analysis.iteration_scaling['scaling_efficiency']
                if eff != float('inf') and eff > best_speed_score:
                    best_speed_score = eff
                    best_speed = name
        recommendations['fastest_scaling'] = best_speed
        
        return recommendations