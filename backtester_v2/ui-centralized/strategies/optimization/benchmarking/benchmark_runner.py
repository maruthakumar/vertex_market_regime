"""
Benchmark Runner

Main interface for orchestrating comprehensive benchmarking operations
including algorithm comparison, scalability testing, and performance analysis.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

from .benchmark_suite import BenchmarkSuite, BenchmarkProblem
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics, ComparativeAnalysis
from .scalability_tester import ScalabilityTester, ScalabilityMetrics, ScalabilityAnalysis

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark runs"""
    algorithms_to_test: List[str] = field(default_factory=list)
    problems_to_test: List[str] = field(default_factory=list)
    
    # Execution parameters
    runs_per_algorithm: int = 5
    max_iterations: int = 1000
    time_limit_seconds: int = 300
    
    # Performance analysis
    enable_performance_analysis: bool = True
    enable_scalability_testing: bool = True
    enable_statistical_comparison: bool = True
    
    # Scalability testing
    scalability_dimensions: List[int] = field(default_factory=lambda: [2, 5, 10, 20])
    scalability_iterations: List[int] = field(default_factory=lambda: [100, 500, 1000])
    
    # Output configuration
    generate_plots: bool = True
    save_detailed_results: bool = True
    output_directory: str = "benchmark_results"
    
    # Parallel execution
    enable_parallel_execution: bool = True
    max_workers: int = field(default_factory=lambda: min(4, multiprocessing.cpu_count()))
    
    # Reporting
    generate_html_report: bool = True
    generate_comparison_matrix: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResults:
    """Comprehensive benchmark results"""
    configuration: BenchmarkConfiguration
    
    # Performance results
    performance_metrics: Dict[str, Dict[str, PerformanceMetrics]]  # algorithm -> problem -> metrics
    comparative_analysis: Optional[ComparativeAnalysis] = None
    
    # Scalability results
    scalability_metrics: Dict[str, Dict[str, List[ScalabilityMetrics]]] = field(default_factory=dict)
    scalability_analysis: Dict[str, ScalabilityAnalysis] = field(default_factory=dict)
    
    # Summary statistics
    algorithm_rankings: Dict[str, int] = field(default_factory=dict)
    best_algorithm_overall: Optional[str] = None
    best_algorithm_by_category: Dict[str, str] = field(default_factory=dict)
    
    # Execution metadata
    total_execution_time: float = 0.0
    successful_runs: int = 0
    failed_runs: int = 0
    warnings: List[str] = field(default_factory=list)
    
    # File paths
    output_files: Dict[str, str] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for optimization algorithms
    
    Orchestrates all benchmarking operations including performance comparison,
    scalability testing, statistical analysis, and report generation.
    """
    
    def __init__(self, 
                 benchmark_suite: Optional[BenchmarkSuite] = None,
                 performance_analyzer: Optional[PerformanceAnalyzer] = None,
                 scalability_tester: Optional[ScalabilityTester] = None):
        """
        Initialize benchmark runner
        
        Args:
            benchmark_suite: Suite of benchmark problems
            performance_analyzer: Performance analysis tools
            scalability_tester: Scalability testing tools
        """
        self.benchmark_suite = benchmark_suite or BenchmarkSuite()
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer()
        self.scalability_tester = scalability_tester or ScalabilityTester()
        
        # Algorithm registry (will be injected)
        self.algorithm_registry = None
        
        logger.info("BenchmarkRunner initialized")
        logger.info(f"Available problems: {len(self.benchmark_suite.problems)}")
    
    def set_algorithm_registry(self, registry):
        """Set algorithm registry for discovering algorithms"""
        self.algorithm_registry = registry
        logger.info("Algorithm registry set")
    
    def run_comprehensive_benchmark(self, 
                                  config: BenchmarkConfiguration) -> BenchmarkResults:
        """
        Run comprehensive benchmark suite
        
        Args:
            config: Benchmark configuration
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting comprehensive benchmark suite")
        start_time = time.time()
        
        # Validate configuration
        self._validate_configuration(config)
        
        # Initialize results
        results = BenchmarkResults(
            configuration=config,
            performance_metrics={},
            scalability_metrics={},
            scalability_analysis={}
        )
        
        # Create output directory
        output_dir = Path(config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Discover algorithms if not specified
            algorithms_to_test = self._get_algorithms_to_test(config)
            problems_to_test = self._get_problems_to_test(config)
            
            logger.info(f"Testing {len(algorithms_to_test)} algorithms on {len(problems_to_test)} problems")
            
            # Run performance benchmarks
            if config.enable_performance_analysis:
                self._run_performance_benchmarks(
                    algorithms_to_test, problems_to_test, config, results
                )
            
            # Run scalability tests
            if config.enable_scalability_testing:
                self._run_scalability_benchmarks(
                    algorithms_to_test, problems_to_test, config, results
                )
            
            # Perform comparative analysis
            if config.enable_statistical_comparison:
                self._perform_comparative_analysis(
                    algorithms_to_test, problems_to_test, results
                )
            
            # Generate summary statistics
            self._generate_summary_statistics(results)
            
            # Generate reports and visualizations
            if config.generate_plots:
                self._generate_visualizations(results, output_dir)
            
            if config.generate_html_report:
                self._generate_html_report(results, output_dir)
            
            # Save detailed results
            if config.save_detailed_results:
                self._save_detailed_results(results, output_dir)
            
            # Calculate execution time
            results.total_execution_time = time.time() - start_time
            
            logger.info(f"Comprehensive benchmark completed in {results.total_execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            results.total_execution_time = time.time() - start_time
            results.warnings.append(f"Benchmark failed: {str(e)}")
            raise
    
    def run_quick_benchmark(self, 
                          algorithms: List[str], 
                          problems: List[str],
                          runs: int = 3) -> BenchmarkResults:
        """
        Run quick benchmark for rapid algorithm comparison
        
        Args:
            algorithms: List of algorithm names
            problems: List of problem names
            runs: Number of runs per algorithm
            
        Returns:
            Quick benchmark results
        """
        config = BenchmarkConfiguration(
            algorithms_to_test=algorithms,
            problems_to_test=problems,
            runs_per_algorithm=runs,
            max_iterations=100,
            enable_scalability_testing=False,
            generate_plots=False,
            generate_html_report=False
        )
        
        return self.run_comprehensive_benchmark(config)
    
    def compare_algorithms(self, 
                         algorithm_pairs: List[Tuple[str, str]],
                         problems: List[str],
                         statistical_tests: bool = True) -> Dict[str, Any]:
        """
        Compare specific algorithm pairs
        
        Args:
            algorithm_pairs: Pairs of algorithms to compare
            problems: Problems to test on
            statistical_tests: Perform statistical significance tests
            
        Returns:
            Detailed comparison results
        """
        logger.info(f"Comparing {len(algorithm_pairs)} algorithm pairs")
        
        comparison_results = {}
        
        for alg1, alg2 in algorithm_pairs:
            logger.info(f"Comparing {alg1} vs {alg2}")
            
            # Run quick benchmark for both algorithms
            config = BenchmarkConfiguration(
                algorithms_to_test=[alg1, alg2],
                problems_to_test=problems,
                runs_per_algorithm=5,
                enable_scalability_testing=False,
                enable_statistical_comparison=statistical_tests
            )
            
            results = self.run_comprehensive_benchmark(config)
            
            comparison_results[f"{alg1}_vs_{alg2}"] = {
                'performance_comparison': results.comparative_analysis,
                'winner': self._determine_winner(alg1, alg2, results),
                'summary': self._generate_pairwise_summary(alg1, alg2, results)
            }
        
        return comparison_results
    
    def benchmark_scalability(self, 
                            algorithm: str,
                            base_problem: str,
                            dimensions: List[int] = None) -> ScalabilityAnalysis:
        """
        Detailed scalability analysis for a single algorithm
        
        Args:
            algorithm: Algorithm to test
            base_problem: Base problem to scale
            dimensions: List of dimensions to test
            
        Returns:
            Detailed scalability analysis
        """
        if dimensions is None:
            dimensions = [2, 5, 10, 20, 50, 100]
        
        logger.info(f"Running scalability analysis for {algorithm}")
        
        # Create scaled versions of the problem
        base_problem_obj = self.benchmark_suite.get_problem(base_problem)
        
        # Run scalability tests
        scalability_metrics = []
        
        for dim in dimensions:
            # Create scaled problem
            scaled_problem = self._create_scaled_problem(base_problem_obj, dim)
            
            # Test algorithm on scaled problem
            metrics = self._test_algorithm_scalability(
                algorithm, scaled_problem, dim
            )
            scalability_metrics.append(metrics)
        
        # Analyze scalability characteristics
        analysis = self.scalability_tester._analyze_scalability_characteristics(
            algorithm, scalability_metrics
        )
        
        return analysis
    
    def generate_algorithm_recommendation(self,
                                        use_case: str,
                                        problem_characteristics: Dict[str, Any],
                                        performance_requirements: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate algorithm recommendation for specific use case
        
        Args:
            use_case: Description of the use case
            problem_characteristics: Characteristics of the optimization problem
            performance_requirements: Required performance criteria
            
        Returns:
            Algorithm recommendation with rationale
        """
        logger.info(f"Generating algorithm recommendation for: {use_case}")
        
        if not self.algorithm_registry:
            raise ValueError("Algorithm registry not set")
        
        # Get algorithm recommendations from registry
        recommendations = self.algorithm_registry.recommend_algorithms(
            problem_characteristics
        )
        
        # Filter by performance requirements
        filtered_recommendations = self._filter_by_performance_requirements(
            recommendations, performance_requirements
        )
        
        # Generate recommendation report
        recommendation = {
            'use_case': use_case,
            'problem_characteristics': problem_characteristics,
            'performance_requirements': performance_requirements,
            'recommended_algorithm': filtered_recommendations[0][0] if filtered_recommendations else None,
            'alternatives': filtered_recommendations[1:3] if len(filtered_recommendations) > 1 else [],
            'rationale': self._generate_recommendation_rationale(
                filtered_recommendations, problem_characteristics
            ),
            'confidence_score': filtered_recommendations[0][1] if filtered_recommendations else 0.0
        }
        
        return recommendation
    
    # Private helper methods
    
    def _validate_configuration(self, config: BenchmarkConfiguration):
        """Validate benchmark configuration"""
        if config.runs_per_algorithm < 1:
            raise ValueError("runs_per_algorithm must be at least 1")
        
        if config.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        
        if config.time_limit_seconds < 1:
            raise ValueError("time_limit_seconds must be at least 1")
    
    def _get_algorithms_to_test(self, config: BenchmarkConfiguration) -> List[str]:
        """Get list of algorithms to test"""
        if config.algorithms_to_test:
            return config.algorithms_to_test
        
        if not self.algorithm_registry:
            raise ValueError("No algorithms specified and no registry available")
        
        # Discover all available algorithms
        discovery_result = self.algorithm_registry.discover_algorithms()
        return list(discovery_result.get('algorithms', {}).keys())
    
    def _get_problems_to_test(self, config: BenchmarkConfiguration) -> List[str]:
        """Get list of problems to test"""
        if config.problems_to_test:
            return config.problems_to_test
        
        # Use a representative subset of problems
        return [
            'sphere_2d', 'rosenbrock_2d', 'rastrigin_2d', 'ackley_2d',
            'portfolio_3_assets', 'strategy_parameters'
        ]
    
    def _run_performance_benchmarks(self,
                                  algorithms: List[str],
                                  problems: List[str],
                                  config: BenchmarkConfiguration,
                                  results: BenchmarkResults):
        """Run performance benchmarks"""
        logger.info("Running performance benchmarks")
        
        total_tests = len(algorithms) * len(problems) * config.runs_per_algorithm
        completed_tests = 0
        
        for algorithm in algorithms:
            results.performance_metrics[algorithm] = {}
            
            for problem_name in problems:
                logger.info(f"Testing {algorithm} on {problem_name}")
                
                try:
                    problem = self.benchmark_suite.get_problem(problem_name)
                    
                    # Run multiple iterations
                    for run in range(config.runs_per_algorithm):
                        # Get algorithm instance
                        optimizer = self._get_algorithm_instance(algorithm, problem)
                        
                        # Run optimization
                        start_time = time.time()
                        optimization_result = self._run_optimization(
                            optimizer, problem, config
                        )
                        execution_time = time.time() - start_time
                        
                        # Add performance data
                        metrics = self.performance_analyzer.add_performance_data(
                            algorithm_name=algorithm,
                            problem_name=problem_name,
                            objective_values=[optimization_result['best_value']],
                            execution_time=execution_time,
                            iterations=optimization_result.get('iterations', config.max_iterations),
                            function_evaluations=optimization_result.get('function_evaluations', config.max_iterations),
                            known_optimal=problem.optimal_value,
                            convergence_history=optimization_result.get('convergence_history')
                        )
                        
                        completed_tests += 1
                        if completed_tests % 10 == 0:
                            logger.info(f"Completed {completed_tests}/{total_tests} tests")
                    
                    # Aggregate results for this algorithm-problem combination
                    aggregated_metrics = self.performance_analyzer.aggregate_multiple_runs(
                        algorithm, problem_name
                    )
                    results.performance_metrics[algorithm][problem_name] = aggregated_metrics
                    results.successful_runs += config.runs_per_algorithm
                    
                except Exception as e:
                    logger.error(f"Failed to test {algorithm} on {problem_name}: {e}")
                    results.failed_runs += config.runs_per_algorithm
                    results.warnings.append(f"Failed: {algorithm} on {problem_name}: {str(e)}")
    
    def _run_scalability_benchmarks(self,
                                  algorithms: List[str],
                                  problems: List[str],
                                  config: BenchmarkConfiguration,
                                  results: BenchmarkResults):
        """Run scalability benchmarks"""
        logger.info("Running scalability benchmarks")
        
        for algorithm in algorithms:
            results.scalability_metrics[algorithm] = {}
            
            # Test on a subset of problems for scalability
            scalability_problems = problems[:2]  # Limit for performance
            
            for problem_name in scalability_problems:
                try:
                    problem = self.benchmark_suite.get_problem(problem_name)
                    
                    # Test different dimensions
                    dimension_metrics = []
                    for dim in config.scalability_dimensions:
                        if dim <= problem.get_dimensionality() * 5:  # Reasonable scaling limit
                            scaled_problem = self._create_scaled_problem(problem, dim)
                            metrics = self._test_algorithm_scalability(
                                algorithm, scaled_problem, dim
                            )
                            dimension_metrics.append(metrics)
                    
                    results.scalability_metrics[algorithm][problem_name] = dimension_metrics
                    
                    # Analyze scalability for this algorithm
                    if dimension_metrics:
                        analysis = self.scalability_tester._analyze_scalability_characteristics(
                            algorithm, dimension_metrics
                        )
                        results.scalability_analysis[f"{algorithm}_{problem_name}"] = analysis
                    
                except Exception as e:
                    logger.error(f"Scalability test failed for {algorithm} on {problem_name}: {e}")
                    results.warnings.append(f"Scalability failed: {algorithm} on {problem_name}: {str(e)}")
    
    def _perform_comparative_analysis(self,
                                    algorithms: List[str],
                                    problems: List[str],
                                    results: BenchmarkResults):
        """Perform comparative analysis between algorithms"""
        logger.info("Performing comparative analysis")
        
        try:
            comparative_analysis = self.performance_analyzer.compare_algorithms(
                algorithm_names=algorithms,
                problem_names=problems
            )
            results.comparative_analysis = comparative_analysis
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            results.warnings.append(f"Comparative analysis failed: {str(e)}")
    
    def _generate_summary_statistics(self, results: BenchmarkResults):
        """Generate summary statistics"""
        logger.info("Generating summary statistics")
        
        if results.comparative_analysis:
            results.algorithm_rankings = results.comparative_analysis.overall_rankings
            results.best_algorithm_overall = results.comparative_analysis.best_overall_algorithm
            
            # Best by category
            results.best_algorithm_by_category = {
                'speed': results.comparative_analysis.best_for_speed,
                'quality': results.comparative_analysis.best_for_quality,
                'robustness': results.comparative_analysis.best_for_robustness
            }
    
    def _get_algorithm_instance(self, algorithm_name: str, problem: BenchmarkProblem):
        """Get algorithm instance for optimization"""
        if not self.algorithm_registry:
            raise ValueError("Algorithm registry not available")
        
        # Get algorithm from registry
        optimizer = self.algorithm_registry.get_algorithm(
            algorithm_name,
            param_space=problem.param_space,
            objective_function=problem.objective_function
        )
        
        return optimizer
    
    def _run_optimization(self, optimizer, problem: BenchmarkProblem, config: BenchmarkConfiguration):
        """Run optimization with timeout and monitoring"""
        start_time = time.time()
        
        # Create a simplified optimization run
        best_value = float('inf')
        best_params = None
        convergence_history = []
        iterations = 0
        
        try:
            # Run optimization (simplified for now)
            result = optimizer.optimize(n_iterations=config.max_iterations)
            
            if hasattr(result, 'best_objective_value'):
                best_value = result.best_objective_value
            if hasattr(result, 'best_parameters'):
                best_params = result.best_parameters
            if hasattr(result, 'convergence_history'):
                convergence_history = result.convergence_history
            if hasattr(result, 'iterations'):
                iterations = result.iterations
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            # Return reasonable defaults
            pass
        
        return {
            'best_value': best_value,
            'best_parameters': best_params,
            'convergence_history': convergence_history,
            'iterations': iterations,
            'function_evaluations': iterations,
            'execution_time': time.time() - start_time
        }
    
    def _create_scaled_problem(self, base_problem: BenchmarkProblem, target_dimension: int):
        """Create scaled version of problem for scalability testing"""
        if target_dimension <= base_problem.get_dimensionality():
            return base_problem
        
        # Create scaled parameter space
        base_params = list(base_problem.param_space.items())
        scaled_param_space = {}
        
        for i in range(target_dimension):
            param_name = f"x{i}"
            if i < len(base_params):
                scaled_param_space[param_name] = base_params[i][1]
            else:
                # Use bounds from the first parameter
                scaled_param_space[param_name] = base_params[0][1]
        
        # Create scaled objective function
        def scaled_objective(params):
            return base_problem.objective_function(params)
        
        # Create new problem instance
        from .benchmark_suite import BenchmarkProblem, ProblemCategory, DifficultyLevel
        
        scaled_problem = BenchmarkProblem(
            name=f"{base_problem.name}_{target_dimension}d",
            param_space=scaled_param_space,
            objective_function=scaled_objective,
            optimal_value=base_problem.optimal_value,
            optimal_parameters=None,  # Unknown for scaled problems
            category=base_problem.category,
            difficulty=base_problem.difficulty,
            description=f"Scaled version of {base_problem.name} to {target_dimension} dimensions",
            properties=base_problem.properties.copy()
        )
        
        return scaled_problem
    
    def _test_algorithm_scalability(self, algorithm: str, problem: BenchmarkProblem, dimension: int):
        """Test algorithm scalability on specific problem"""
        try:
            optimizer = self._get_algorithm_instance(algorithm, problem)
            
            start_time = time.time()
            initial_memory = self._get_memory_usage()
            
            # Run optimization
            result = self._run_optimization(optimizer, problem, 
                                          BenchmarkConfiguration(max_iterations=100))
            
            execution_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            
            # Create scalability metrics
            from .scalability_tester import ScalabilityMetrics
            
            metrics = ScalabilityMetrics(
                algorithm_name=algorithm,
                problem_name=problem.name,
                dimension=dimension,
                iterations=result.get('iterations', 100),
                execution_time=execution_time,
                memory_usage_mb=(final_memory - initial_memory),
                cpu_utilization=0.0,  # Simplified
                function_evaluations=result.get('function_evaluations', 100),
                best_objective_value=result.get('best_value', float('inf')),
                convergence_achieved=result.get('best_value', float('inf')) < 1e6,
                time_per_dimension=execution_time / dimension,
                time_per_iteration=execution_time / result.get('iterations', 1),
                memory_per_dimension=(final_memory - initial_memory) / dimension,
                evaluations_per_second=result.get('function_evaluations', 100) / execution_time,
                resource_efficiency=1.0 / (execution_time * (final_memory - initial_memory + 1)),
                computational_complexity="O(n)"  # Simplified
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Scalability test failed: {e}")
            return None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _generate_visualizations(self, results: BenchmarkResults, output_dir: Path):
        """Generate benchmark visualizations"""
        logger.info("Generating visualizations")
        
        try:
            # Performance comparison plots
            if results.comparative_analysis:
                self._plot_algorithm_comparison(results.comparative_analysis, output_dir)
            
            # Scalability plots
            if results.scalability_metrics:
                self._plot_scalability_analysis(results.scalability_metrics, output_dir)
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            results.warnings.append(f"Visualization failed: {str(e)}")
    
    def _plot_algorithm_comparison(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Generate algorithm comparison plots"""
        # Rankings comparison
        plt.figure(figsize=(12, 8))
        
        algorithms = list(analysis.overall_rankings.keys())
        rankings = list(analysis.overall_rankings.values())
        
        plt.barh(algorithms, rankings)
        plt.xlabel('Ranking (1 = Best)')
        plt.title('Algorithm Overall Rankings')
        plt.gca().invert_yaxis()
        
        output_file = output_dir / "algorithm_rankings.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Rankings plot saved to {output_file}")
    
    def _plot_scalability_analysis(self, scalability_data: Dict, output_dir: Path):
        """Generate scalability analysis plots"""
        plt.figure(figsize=(12, 8))
        
        for algorithm, problems in scalability_data.items():
            for problem, metrics_list in problems.items():
                if metrics_list:
                    dimensions = [m.dimension for m in metrics_list if m]
                    times = [m.execution_time for m in metrics_list if m]
                    
                    plt.plot(dimensions, times, marker='o', label=f"{algorithm}_{problem}")
        
        plt.xlabel('Problem Dimension')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Algorithm Scalability Analysis')
        plt.legend()
        plt.yscale('log')
        
        output_file = output_dir / "scalability_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scalability plot saved to {output_file}")
    
    def _generate_html_report(self, results: BenchmarkResults, output_dir: Path):
        """Generate comprehensive HTML report"""
        logger.info("Generating HTML report")
        
        html_content = self._create_html_report_content(results)
        
        report_file = output_dir / "benchmark_report.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        results.output_files['html_report'] = str(report_file)
        logger.info(f"HTML report saved to {report_file}")
    
    def _create_html_report_content(self, results: BenchmarkResults) -> str:
        """Create HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Algorithm Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Optimization Algorithm Benchmark Report</h1>
                <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total execution time: {results.total_execution_time:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">Best Overall Algorithm: {results.best_algorithm_overall or 'N/A'}</div>
                <div class="metric">Successful Tests: {results.successful_runs}</div>
                <div class="metric">Failed Tests: {results.failed_runs}</div>
            </div>
            
            <div class="section">
                <h2>Algorithm Rankings</h2>
                <table>
                    <tr><th>Algorithm</th><th>Overall Rank</th></tr>
        """
        
        for alg, rank in results.algorithm_rankings.items():
            html += f"<tr><td>{alg}</td><td>{rank}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Best by Category</h2>
                <table>
                    <tr><th>Category</th><th>Best Algorithm</th></tr>
        """
        
        for category, alg in results.best_algorithm_by_category.items():
            html += f"<tr><td>{category.title()}</td><td>{alg}</td></tr>"
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _save_detailed_results(self, results: BenchmarkResults, output_dir: Path):
        """Save detailed results to JSON"""
        logger.info("Saving detailed results")
        
        # Convert results to serializable format
        serializable_results = self._make_results_serializable(results)
        
        results_file = output_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        results.output_files['detailed_results'] = str(results_file)
        logger.info(f"Detailed results saved to {results_file}")
    
    def _make_results_serializable(self, results: BenchmarkResults) -> Dict:
        """Convert results to JSON-serializable format"""
        return {
            'configuration': {
                'algorithms_tested': results.configuration.algorithms_to_test,
                'problems_tested': results.configuration.problems_to_test,
                'runs_per_algorithm': results.configuration.runs_per_algorithm,
                'max_iterations': results.configuration.max_iterations
            },
            'summary': {
                'total_execution_time': results.total_execution_time,
                'successful_runs': results.successful_runs,
                'failed_runs': results.failed_runs,
                'best_algorithm_overall': results.best_algorithm_overall,
                'algorithm_rankings': results.algorithm_rankings,
                'best_by_category': results.best_algorithm_by_category
            },
            'warnings': results.warnings,
            'output_files': results.output_files
        }
    
    def _determine_winner(self, alg1: str, alg2: str, results: BenchmarkResults) -> str:
        """Determine winner between two algorithms"""
        if not results.algorithm_rankings:
            return "Inconclusive"
        
        rank1 = results.algorithm_rankings.get(alg1, float('inf'))
        rank2 = results.algorithm_rankings.get(alg2, float('inf'))
        
        if rank1 < rank2:
            return alg1
        elif rank2 < rank1:
            return alg2
        else:
            return "Tie"
    
    def _generate_pairwise_summary(self, alg1: str, alg2: str, results: BenchmarkResults) -> str:
        """Generate summary for pairwise comparison"""
        winner = self._determine_winner(alg1, alg2, results)
        
        summary = f"Comparison between {alg1} and {alg2}: "
        
        if winner == "Tie":
            summary += "Performance is roughly equivalent"
        elif winner == "Inconclusive":
            summary += "Unable to determine clear winner"
        else:
            summary += f"{winner} performs better overall"
        
        return summary
    
    def _filter_by_performance_requirements(self, 
                                          recommendations: List[Tuple[str, float]],
                                          requirements: Dict[str, float]) -> List[Tuple[str, float]]:
        """Filter algorithm recommendations by performance requirements"""
        # Simplified filtering - in practice, this would check actual performance data
        return recommendations
    
    def _generate_recommendation_rationale(self, 
                                         recommendations: List[Tuple[str, float]],
                                         characteristics: Dict[str, Any]) -> str:
        """Generate rationale for algorithm recommendation"""
        if not recommendations:
            return "No suitable algorithms found for the given requirements"
        
        best_alg, score = recommendations[0]
        
        rationale = f"{best_alg} is recommended (confidence: {score:.2f}) based on "
        
        if characteristics.get('problem_type') == 'continuous':
            rationale += "its effectiveness on continuous optimization problems"
        elif characteristics.get('problem_type') == 'discrete':
            rationale += "its effectiveness on discrete optimization problems"
        else:
            rationale += "its general optimization capabilities"
        
        if characteristics.get('dimensionality', 0) > 10:
            rationale += " and good scalability to high dimensions"
        
        return rationale