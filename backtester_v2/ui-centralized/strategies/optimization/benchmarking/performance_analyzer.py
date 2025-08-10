"""
Performance Analyzer

Advanced performance analysis tools for optimization algorithms including
statistical analysis, convergence analysis, and comparative performance metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import time
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for an algorithm"""
    algorithm_name: str
    problem_name: str
    
    # Solution Quality Metrics
    best_objective_value: float
    final_objective_value: float
    solution_quality_score: float  # 0-1 score based on known optimal
    convergence_achieved: bool
    
    # Efficiency Metrics
    total_iterations: int
    total_function_evaluations: int
    execution_time: float
    time_per_iteration: float
    evaluations_per_second: float
    
    # Convergence Metrics
    convergence_iteration: Optional[int]
    convergence_time: Optional[float]
    convergence_rate: float
    improvement_rate: float
    
    # Robustness Metrics
    success_rate: float  # Over multiple runs
    consistency_score: float  # Low variance across runs
    reliability_score: float  # Consistent convergence
    
    # Statistical Metrics
    mean_performance: float
    std_performance: float
    median_performance: float
    best_performance: float
    worst_performance: float
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComparativeAnalysis:
    """Comparative analysis between algorithms"""
    algorithms: List[str]
    problems: List[str]
    
    # Performance Rankings
    overall_rankings: Dict[str, int]  # algorithm -> rank
    problem_specific_rankings: Dict[str, Dict[str, int]]  # problem -> algorithm -> rank
    
    # Statistical Comparisons
    statistical_significance: Dict[Tuple[str, str], bool]  # (alg1, alg2) -> significant
    effect_sizes: Dict[Tuple[str, str], float]  # Cohen's d
    
    # Specialized Rankings
    speed_rankings: Dict[str, int]
    quality_rankings: Dict[str, int]
    robustness_rankings: Dict[str, int]
    
    # Recommendations
    best_overall_algorithm: str
    best_for_speed: str
    best_for_quality: str
    best_for_robustness: str
    
    # Detailed Analysis
    strengths_weaknesses: Dict[str, Dict[str, List[str]]]
    recommendations_by_use_case: Dict[str, str]


class PerformanceAnalyzer:
    """
    Advanced performance analyzer for optimization algorithms
    
    Provides comprehensive performance analysis including statistical testing,
    convergence analysis, scalability assessment, and comparative evaluations.
    """
    
    def __init__(self,
                 significance_level: float = 0.05,
                 min_runs_for_stats: int = 10,
                 convergence_tolerance: float = 1e-6,
                 max_stagnation_iterations: int = 50):
        """
        Initialize performance analyzer
        
        Args:
            significance_level: Alpha level for statistical tests
            min_runs_for_stats: Minimum runs required for statistical analysis
            convergence_tolerance: Tolerance for convergence detection
            max_stagnation_iterations: Max iterations without improvement
        """
        self.significance_level = significance_level
        self.min_runs_for_stats = min_runs_for_stats
        self.convergence_tolerance = convergence_tolerance
        self.max_stagnation_iterations = max_stagnation_iterations
        
        # Storage for analysis results
        self.performance_data: Dict[str, List[PerformanceMetrics]] = {}
        self.convergence_histories: Dict[str, List[List[float]]] = {}
        
        logger.info("PerformanceAnalyzer initialized")
    
    def add_performance_data(self,
                           algorithm_name: str,
                           problem_name: str,
                           objective_values: List[float],
                           execution_time: float,
                           iterations: int,
                           function_evaluations: int,
                           known_optimal: Optional[float] = None,
                           convergence_history: Optional[List[float]] = None) -> PerformanceMetrics:
        """
        Add performance data for analysis
        
        Args:
            algorithm_name: Name of the algorithm
            problem_name: Name of the problem
            objective_values: List of objective values achieved
            execution_time: Total execution time
            iterations: Number of iterations
            function_evaluations: Number of function evaluations
            known_optimal: Known optimal value (if available)
            convergence_history: History of best values per iteration
            
        Returns:
            Computed performance metrics
        """
        # Calculate metrics
        best_value = min(objective_values) if objective_values else float('inf')
        final_value = objective_values[-1] if objective_values else float('inf')
        
        # Solution quality score
        if known_optimal is not None:
            if abs(known_optimal) < 1e-10:  # Optimal is zero
                quality_score = max(0, 1 - abs(best_value) / (abs(best_value) + 1))
            else:
                quality_score = max(0, 1 - abs(best_value - known_optimal) / abs(known_optimal))
        else:
            quality_score = 0.5  # Default when optimal unknown
        
        # Convergence analysis
        convergence_achieved = False
        convergence_iteration = None
        convergence_time = None
        
        if convergence_history and known_optimal is not None:
            for i, value in enumerate(convergence_history):
                if abs(value - known_optimal) <= self.convergence_tolerance:
                    convergence_achieved = True
                    convergence_iteration = i
                    convergence_time = (i / len(convergence_history)) * execution_time
                    break
        
        # Calculate rates
        time_per_iteration = execution_time / iterations if iterations > 0 else 0
        evaluations_per_second = function_evaluations / execution_time if execution_time > 0 else 0
        
        # Improvement analysis
        if convergence_history and len(convergence_history) > 1:
            improvements = [convergence_history[i-1] - convergence_history[i] 
                          for i in range(1, len(convergence_history))
                          if convergence_history[i-1] > convergence_history[i]]
            improvement_rate = np.mean(improvements) if improvements else 0
            
            # Convergence rate (exponential decay fit)
            if len(convergence_history) > 10:
                try:
                    x = np.arange(len(convergence_history))
                    y = np.array(convergence_history)
                    # Fit exponential decay: y = a * exp(-b * x) + c
                    from scipy.optimize import curve_fit
                    
                    def exp_decay(x, a, b, c):
                        return a * np.exp(-b * x) + c
                    
                    popt, _ = curve_fit(exp_decay, x, y, maxfev=1000)
                    convergence_rate = popt[1]  # Decay rate
                except:
                    convergence_rate = 0.1  # Default
            else:
                convergence_rate = 0.1
        else:
            improvement_rate = 0
            convergence_rate = 0.1
        
        # Create metrics object
        metrics = PerformanceMetrics(
            algorithm_name=algorithm_name,
            problem_name=problem_name,
            best_objective_value=best_value,
            final_objective_value=final_value,
            solution_quality_score=quality_score,
            convergence_achieved=convergence_achieved,
            total_iterations=iterations,
            total_function_evaluations=function_evaluations,
            execution_time=execution_time,
            time_per_iteration=time_per_iteration,
            evaluations_per_second=evaluations_per_second,
            convergence_iteration=convergence_iteration,
            convergence_time=convergence_time,
            convergence_rate=convergence_rate,
            improvement_rate=improvement_rate,
            success_rate=1.0 if convergence_achieved else 0.0,  # Will be updated with multiple runs
            consistency_score=1.0,  # Will be updated with multiple runs
            reliability_score=1.0,  # Will be updated with multiple runs
            mean_performance=best_value,
            std_performance=0.0,
            median_performance=best_value,
            best_performance=best_value,
            worst_performance=best_value
        )
        
        # Store data
        key = f"{algorithm_name}_{problem_name}"
        if key not in self.performance_data:
            self.performance_data[key] = []
            self.convergence_histories[key] = []
        
        self.performance_data[key].append(metrics)
        if convergence_history:
            self.convergence_histories[key].append(convergence_history)
        
        return metrics
    
    def aggregate_multiple_runs(self,
                              algorithm_name: str,
                              problem_name: str) -> PerformanceMetrics:
        """
        Aggregate performance metrics across multiple runs
        
        Args:
            algorithm_name: Name of the algorithm
            problem_name: Name of the problem
            
        Returns:
            Aggregated performance metrics
        """
        key = f"{algorithm_name}_{problem_name}"
        
        if key not in self.performance_data or len(self.performance_data[key]) == 0:
            raise ValueError(f"No performance data found for {algorithm_name} on {problem_name}")
        
        runs = self.performance_data[key]
        
        if len(runs) == 1:
            return runs[0]
        
        # Aggregate metrics
        best_values = [run.best_objective_value for run in runs]
        execution_times = [run.execution_time for run in runs]
        iterations = [run.total_iterations for run in runs]
        quality_scores = [run.solution_quality_score for run in runs]
        convergence_flags = [run.convergence_achieved for run in runs]
        
        # Statistical metrics
        mean_performance = np.mean(best_values)
        std_performance = np.std(best_values)
        median_performance = np.median(best_values)
        best_performance = min(best_values)
        worst_performance = max(best_values)
        
        # Success and consistency metrics
        success_rate = sum(convergence_flags) / len(convergence_flags)
        consistency_score = max(0, 1 - std_performance / (abs(mean_performance) + 1e-10))
        reliability_score = success_rate * consistency_score
        
        # Use best run as template
        best_run = min(runs, key=lambda r: r.best_objective_value)
        
        # Create aggregated metrics
        aggregated = PerformanceMetrics(
            algorithm_name=algorithm_name,
            problem_name=problem_name,
            best_objective_value=best_performance,
            final_objective_value=best_run.final_objective_value,
            solution_quality_score=max(quality_scores),
            convergence_achieved=success_rate > 0.5,
            total_iterations=int(np.mean(iterations)),
            total_function_evaluations=best_run.total_function_evaluations,
            execution_time=np.mean(execution_times),
            time_per_iteration=np.mean([r.time_per_iteration for r in runs]),
            evaluations_per_second=np.mean([r.evaluations_per_second for r in runs]),
            convergence_iteration=best_run.convergence_iteration,
            convergence_time=best_run.convergence_time,
            convergence_rate=np.mean([r.convergence_rate for r in runs]),
            improvement_rate=np.mean([r.improvement_rate for r in runs]),
            success_rate=success_rate,
            consistency_score=consistency_score,
            reliability_score=reliability_score,
            mean_performance=mean_performance,
            std_performance=std_performance,
            median_performance=median_performance,
            best_performance=best_performance,
            worst_performance=worst_performance,
            metadata={'num_runs': len(runs), 'aggregated': True}
        )
        
        return aggregated
    
    def compare_algorithms(self,
                         algorithm_names: List[str],
                         problem_names: List[str],
                         metrics_to_compare: List[str] = None) -> ComparativeAnalysis:
        """
        Comprehensive comparison of algorithms across problems
        
        Args:
            algorithm_names: List of algorithm names to compare
            problem_names: List of problem names to compare on
            metrics_to_compare: Specific metrics to focus on
            
        Returns:
            Comprehensive comparative analysis
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['best_objective_value', 'execution_time', 'solution_quality_score']
        
        # Gather aggregated data
        algorithm_data = {}
        for alg in algorithm_names:
            algorithm_data[alg] = {}
            for problem in problem_names:
                try:
                    metrics = self.aggregate_multiple_runs(alg, problem)
                    algorithm_data[alg][problem] = metrics
                except ValueError:
                    logger.warning(f"No data for {alg} on {problem}")
                    continue
        
        # Calculate rankings
        overall_rankings = self._calculate_overall_rankings(algorithm_data, metrics_to_compare)
        problem_specific_rankings = self._calculate_problem_specific_rankings(algorithm_data, metrics_to_compare)
        
        # Specialized rankings
        speed_rankings = self._calculate_specialized_rankings(algorithm_data, 'execution_time', ascending=True)
        quality_rankings = self._calculate_specialized_rankings(algorithm_data, 'solution_quality_score', ascending=False)
        robustness_rankings = self._calculate_specialized_rankings(algorithm_data, 'reliability_score', ascending=False)
        
        # Statistical significance testing
        statistical_significance = self._test_statistical_significance(algorithm_data, algorithm_names)
        effect_sizes = self._calculate_effect_sizes(algorithm_data, algorithm_names)
        
        # Identify best algorithms
        best_overall = min(overall_rankings.items(), key=lambda x: x[1])[0]
        best_speed = min(speed_rankings.items(), key=lambda x: x[1])[0]
        best_quality = min(quality_rankings.items(), key=lambda x: x[1])[0]
        best_robustness = min(robustness_rankings.items(), key=lambda x: x[1])[0]
        
        # Analyze strengths and weaknesses
        strengths_weaknesses = self._analyze_strengths_weaknesses(algorithm_data, algorithm_names)
        
        # Generate use case recommendations
        use_case_recommendations = self._generate_use_case_recommendations(
            algorithm_data, algorithm_names, overall_rankings
        )
        
        return ComparativeAnalysis(
            algorithms=algorithm_names,
            problems=problem_names,
            overall_rankings=overall_rankings,
            problem_specific_rankings=problem_specific_rankings,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            speed_rankings=speed_rankings,
            quality_rankings=quality_rankings,
            robustness_rankings=robustness_rankings,
            best_overall_algorithm=best_overall,
            best_for_speed=best_speed,
            best_for_quality=best_quality,
            best_for_robustness=best_robustness,
            strengths_weaknesses=strengths_weaknesses,
            recommendations_by_use_case=use_case_recommendations
        )
    
    def analyze_convergence(self,
                          algorithm_name: str,
                          problem_name: str) -> Dict[str, Any]:
        """
        Detailed convergence analysis for an algorithm on a problem
        
        Args:
            algorithm_name: Name of the algorithm
            problem_name: Name of the problem
            
        Returns:
            Detailed convergence analysis
        """
        key = f"{algorithm_name}_{problem_name}"
        
        if key not in self.convergence_histories:
            return {'error': 'No convergence history data available'}
        
        histories = self.convergence_histories[key]
        
        if not histories:
            return {'error': 'No convergence histories found'}
        
        analysis = {
            'num_runs': len(histories),
            'convergence_characteristics': {},
            'statistical_analysis': {},
            'recommendations': []
        }
        
        # Analyze each run
        convergence_rates = []
        final_values = []
        stagnation_points = []
        
        for i, history in enumerate(histories):
            if len(history) < 2:
                continue
                
            # Calculate convergence rate
            improvements = [history[j-1] - history[j] for j in range(1, len(history)) 
                          if history[j-1] > history[j]]
            avg_improvement = np.mean(improvements) if improvements else 0
            convergence_rates.append(avg_improvement)
            
            # Final value
            final_values.append(history[-1])
            
            # Detect stagnation
            stagnation_point = self._detect_stagnation(history)
            stagnation_points.append(stagnation_point)
        
        # Statistical analysis of convergence
        analysis['convergence_characteristics'] = {
            'average_convergence_rate': np.mean(convergence_rates),
            'convergence_rate_std': np.std(convergence_rates),
            'average_final_value': np.mean(final_values),
            'final_value_std': np.std(final_values),
            'average_stagnation_point': np.mean([sp for sp in stagnation_points if sp is not None]),
            'consistency_score': 1 - (np.std(final_values) / (abs(np.mean(final_values)) + 1e-10))
        }
        
        # Recommendations based on convergence analysis
        if np.mean(convergence_rates) < 0.001:
            analysis['recommendations'].append("Slow convergence detected - consider increasing learning rate or using adaptive methods")
        
        if len([sp for sp in stagnation_points if sp is not None]) > len(histories) * 0.5:
            analysis['recommendations'].append("Frequent stagnation detected - consider restart mechanisms or diversification")
        
        if np.std(final_values) > abs(np.mean(final_values)) * 0.1:
            analysis['recommendations'].append("High variance in results - consider ensemble methods or robustness improvements")
        
        return analysis
    
    def generate_performance_report(self,
                                  algorithm_names: List[str],
                                  problem_names: List[str],
                                  output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            algorithm_names: Algorithms to include in report
            problem_names: Problems to include in report
            output_file: Optional file to save report
            
        Returns:
            Comprehensive performance report
        """
        report = {
            'executive_summary': {},
            'detailed_analysis': {},
            'comparative_analysis': {},
            'recommendations': {},
            'statistical_tests': {},
            'convergence_analysis': {},
            'metadata': {
                'algorithms_analyzed': len(algorithm_names),
                'problems_analyzed': len(problem_names),
                'total_comparisons': len(algorithm_names) * len(problem_names),
                'generation_time': time.time()
            }
        }
        
        # Executive summary
        comparative_analysis = self.compare_algorithms(algorithm_names, problem_names)
        
        report['executive_summary'] = {
            'best_overall_algorithm': comparative_analysis.best_overall_algorithm,
            'best_for_speed': comparative_analysis.best_for_speed,
            'best_for_quality': comparative_analysis.best_for_quality,
            'best_for_robustness': comparative_analysis.best_for_robustness,
            'key_findings': self._generate_key_findings(comparative_analysis)
        }
        
        # Detailed analysis for each algorithm
        for alg in algorithm_names:
            alg_analysis = {}
            for problem in problem_names:
                try:
                    metrics = self.aggregate_multiple_runs(alg, problem)
                    alg_analysis[problem] = {
                        'performance_score': metrics.solution_quality_score,
                        'execution_time': metrics.execution_time,
                        'success_rate': metrics.success_rate,
                        'consistency': metrics.consistency_score
                    }
                except ValueError:
                    continue
            
            report['detailed_analysis'][alg] = alg_analysis
        
        # Store comparative analysis
        report['comparative_analysis'] = {
            'overall_rankings': comparative_analysis.overall_rankings,
            'speed_rankings': comparative_analysis.speed_rankings,
            'quality_rankings': comparative_analysis.quality_rankings,
            'robustness_rankings': comparative_analysis.robustness_rankings,
            'strengths_weaknesses': comparative_analysis.strengths_weaknesses
        }
        
        # Statistical significance tests
        report['statistical_tests'] = {
            'significance_matrix': comparative_analysis.statistical_significance,
            'effect_sizes': comparative_analysis.effect_sizes,
            'significance_level': self.significance_level
        }
        
        # Convergence analysis
        convergence_data = {}
        for alg in algorithm_names:
            for problem in problem_names:
                conv_analysis = self.analyze_convergence(alg, problem)
                if 'error' not in conv_analysis:
                    convergence_data[f"{alg}_{problem}"] = conv_analysis
        
        report['convergence_analysis'] = convergence_data
        
        # Final recommendations
        report['recommendations'] = {
            'algorithm_selection': comparative_analysis.recommendations_by_use_case,
            'performance_improvements': self._generate_improvement_recommendations(comparative_analysis),
            'future_research': self._suggest_future_research(comparative_analysis)
        }
        
        # Save report if requested
        if output_file:
            self._save_report(report, output_file)
        
        return report
    
    # Private helper methods
    
    def _calculate_overall_rankings(self, algorithm_data, metrics_to_compare):
        """Calculate overall rankings across all problems"""
        algorithm_scores = {}
        
        for alg in algorithm_data:
            scores = []
            for problem in algorithm_data[alg]:
                metrics = algorithm_data[alg][problem]
                # Weighted score combining multiple metrics
                score = (metrics.solution_quality_score * 0.4 + 
                        (1 / (metrics.execution_time + 1)) * 0.3 + 
                        metrics.reliability_score * 0.3)
                scores.append(score)
            
            algorithm_scores[alg] = np.mean(scores) if scores else 0
        
        # Convert to rankings (1 = best)
        sorted_algs = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {alg: rank + 1 for rank, (alg, _) in enumerate(sorted_algs)}
        
        return rankings
    
    def _calculate_problem_specific_rankings(self, algorithm_data, metrics_to_compare):
        """Calculate rankings for each problem separately"""
        problem_rankings = {}
        
        for problem in set().union(*(alg_data.keys() for alg_data in algorithm_data.values())):
            problem_scores = {}
            
            for alg in algorithm_data:
                if problem in algorithm_data[alg]:
                    metrics = algorithm_data[alg][problem]
                    score = (metrics.solution_quality_score * 0.4 + 
                            (1 / (metrics.execution_time + 1)) * 0.3 + 
                            metrics.reliability_score * 0.3)
                    problem_scores[alg] = score
            
            # Convert to rankings
            sorted_algs = sorted(problem_scores.items(), key=lambda x: x[1], reverse=True)
            rankings = {alg: rank + 1 for rank, (alg, _) in enumerate(sorted_algs)}
            problem_rankings[problem] = rankings
        
        return problem_rankings
    
    def _calculate_specialized_rankings(self, algorithm_data, metric_name, ascending=True):
        """Calculate rankings based on a specific metric"""
        algorithm_scores = {}
        
        for alg in algorithm_data:
            scores = []
            for problem in algorithm_data[alg]:
                metrics = algorithm_data[alg][problem]
                score = getattr(metrics, metric_name, 0)
                scores.append(score)
            
            algorithm_scores[alg] = np.mean(scores) if scores else 0
        
        # Convert to rankings
        sorted_algs = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=not ascending)
        rankings = {alg: rank + 1 for rank, (alg, _) in enumerate(sorted_algs)}
        
        return rankings
    
    def _test_statistical_significance(self, algorithm_data, algorithm_names):
        """Test statistical significance between algorithm pairs"""
        significance_results = {}
        
        for i, alg1 in enumerate(algorithm_names):
            for j, alg2 in enumerate(algorithm_names):
                if i >= j:
                    continue
                
                # Collect performance data for both algorithms
                alg1_scores = []
                alg2_scores = []
                
                for problem in set().union(*(alg_data.keys() for alg_data in algorithm_data.values())):
                    if (problem in algorithm_data[alg1] and 
                        problem in algorithm_data[alg2]):
                        alg1_scores.append(algorithm_data[alg1][problem].solution_quality_score)
                        alg2_scores.append(algorithm_data[alg2][problem].solution_quality_score)
                
                # Perform t-test if sufficient data
                if len(alg1_scores) >= 3 and len(alg2_scores) >= 3:
                    try:
                        statistic, p_value = stats.ttest_ind(alg1_scores, alg2_scores)
                        significant = p_value < self.significance_level
                        significance_results[(alg1, alg2)] = significant
                    except:
                        significance_results[(alg1, alg2)] = False
                else:
                    significance_results[(alg1, alg2)] = False
        
        return significance_results
    
    def _calculate_effect_sizes(self, algorithm_data, algorithm_names):
        """Calculate Cohen's d effect sizes between algorithm pairs"""
        effect_sizes = {}
        
        for i, alg1 in enumerate(algorithm_names):
            for j, alg2 in enumerate(algorithm_names):
                if i >= j:
                    continue
                
                # Collect performance data
                alg1_scores = []
                alg2_scores = []
                
                for problem in set().union(*(alg_data.keys() for alg_data in algorithm_data.values())):
                    if (problem in algorithm_data[alg1] and 
                        problem in algorithm_data[alg2]):
                        alg1_scores.append(algorithm_data[alg1][problem].solution_quality_score)
                        alg2_scores.append(algorithm_data[alg2][problem].solution_quality_score)
                
                # Calculate Cohen's d
                if len(alg1_scores) >= 2 and len(alg2_scores) >= 2:
                    mean1, mean2 = np.mean(alg1_scores), np.mean(alg2_scores)
                    std1, std2 = np.std(alg1_scores, ddof=1), np.std(alg2_scores, ddof=1)
                    pooled_std = np.sqrt(((len(alg1_scores) - 1) * std1**2 + 
                                        (len(alg2_scores) - 1) * std2**2) / 
                                       (len(alg1_scores) + len(alg2_scores) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (mean1 - mean2) / pooled_std
                        effect_sizes[(alg1, alg2)] = abs(cohens_d)
                    else:
                        effect_sizes[(alg1, alg2)] = 0.0
                else:
                    effect_sizes[(alg1, alg2)] = 0.0
        
        return effect_sizes
    
    def _analyze_strengths_weaknesses(self, algorithm_data, algorithm_names):
        """Analyze strengths and weaknesses of each algorithm"""
        analysis = {}
        
        for alg in algorithm_names:
            strengths = []
            weaknesses = []
            
            if alg not in algorithm_data:
                continue
            
            # Analyze performance characteristics
            quality_scores = [metrics.solution_quality_score for metrics in algorithm_data[alg].values()]
            speed_scores = [1 / (metrics.execution_time + 1) for metrics in algorithm_data[alg].values()]
            reliability_scores = [metrics.reliability_score for metrics in algorithm_data[alg].values()]
            
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                avg_speed = np.mean(speed_scores)
                avg_reliability = np.mean(reliability_scores)
                
                # Identify strengths (above 75th percentile)
                if avg_quality > 0.75:
                    strengths.append("High solution quality")
                if avg_speed > 0.75:
                    strengths.append("Fast execution")
                if avg_reliability > 0.75:
                    strengths.append("High reliability and consistency")
                
                # Identify weaknesses (below 25th percentile)
                if avg_quality < 0.25:
                    weaknesses.append("Low solution quality")
                if avg_speed < 0.25:
                    weaknesses.append("Slow execution")
                if avg_reliability < 0.25:
                    weaknesses.append("Low reliability and high variance")
            
            analysis[alg] = {'strengths': strengths, 'weaknesses': weaknesses}
        
        return analysis
    
    def _generate_use_case_recommendations(self, algorithm_data, algorithm_names, overall_rankings):
        """Generate algorithm recommendations for different use cases"""
        recommendations = {}
        
        # High-performance computing (speed priority)
        speed_scores = {}
        for alg in algorithm_names:
            if alg in algorithm_data:
                speeds = [1 / (metrics.execution_time + 1) for metrics in algorithm_data[alg].values()]
                speed_scores[alg] = np.mean(speeds) if speeds else 0
        
        best_for_speed = max(speed_scores.items(), key=lambda x: x[1])[0] if speed_scores else algorithm_names[0]
        recommendations['high_performance_computing'] = best_for_speed
        
        # Research and development (quality priority)
        quality_scores = {}
        for alg in algorithm_names:
            if alg in algorithm_data:
                qualities = [metrics.solution_quality_score for metrics in algorithm_data[alg].values()]
                quality_scores[alg] = np.mean(qualities) if qualities else 0
        
        best_for_quality = max(quality_scores.items(), key=lambda x: x[1])[0] if quality_scores else algorithm_names[0]
        recommendations['research_development'] = best_for_quality
        
        # Production systems (reliability priority)
        reliability_scores = {}
        for alg in algorithm_names:
            if alg in algorithm_data:
                reliabilities = [metrics.reliability_score for metrics in algorithm_data[alg].values()]
                reliability_scores[alg] = np.mean(reliabilities) if reliabilities else 0
        
        best_for_reliability = max(reliability_scores.items(), key=lambda x: x[1])[0] if reliability_scores else algorithm_names[0]
        recommendations['production_systems'] = best_for_reliability
        
        # General purpose (balanced)
        best_overall = min(overall_rankings.items(), key=lambda x: x[1])[0]
        recommendations['general_purpose'] = best_overall
        
        return recommendations
    
    def _detect_stagnation(self, convergence_history):
        """Detect when algorithm stagnates"""
        if len(convergence_history) < self.max_stagnation_iterations:
            return None
        
        for i in range(len(convergence_history) - self.max_stagnation_iterations):
            window = convergence_history[i:i + self.max_stagnation_iterations]
            if max(window) - min(window) <= self.convergence_tolerance:
                return i
        
        return None
    
    def _generate_key_findings(self, comparative_analysis):
        """Generate key findings from comparative analysis"""
        findings = []
        
        # Check for dominant algorithm
        rankings = comparative_analysis.overall_rankings
        best_alg = comparative_analysis.best_overall_algorithm
        
        if rankings[best_alg] == 1:
            findings.append(f"{best_alg} consistently outperforms other algorithms")
        
        # Check for specialized strengths
        if (comparative_analysis.best_for_speed != best_alg):
            findings.append(f"{comparative_analysis.best_for_speed} excels in execution speed")
        
        if (comparative_analysis.best_for_quality != best_alg):
            findings.append(f"{comparative_analysis.best_for_quality} achieves highest solution quality")
        
        # Check for statistical significance
        significant_pairs = sum(1 for sig in comparative_analysis.statistical_significance.values() if sig)
        total_pairs = len(comparative_analysis.statistical_significance)
        
        if significant_pairs > total_pairs * 0.5:
            findings.append("Significant performance differences found between algorithms")
        
        return findings
    
    def _generate_improvement_recommendations(self, comparative_analysis):
        """Generate recommendations for algorithm improvements"""
        recommendations = []
        
        # Analyze weaknesses
        for alg, sw in comparative_analysis.strengths_weaknesses.items():
            if 'Low solution quality' in sw['weaknesses']:
                recommendations.append(f"Improve {alg} solution quality through better search mechanisms")
            if 'Slow execution' in sw['weaknesses']:
                recommendations.append(f"Optimize {alg} for better computational efficiency")
            if 'Low reliability' in sw['weaknesses']:
                recommendations.append(f"Enhance {alg} robustness and consistency")
        
        return recommendations
    
    def _suggest_future_research(self, comparative_analysis):
        """Suggest future research directions"""
        suggestions = []
        
        # Based on gaps in current algorithms
        algorithms = comparative_analysis.algorithms
        
        if len(algorithms) > 1:
            suggestions.append("Investigate hybrid approaches combining best features of top algorithms")
            suggestions.append("Develop adaptive algorithms that adjust strategy based on problem characteristics")
            suggestions.append("Research parallel and distributed optimization approaches")
        
        return suggestions
    
    def _save_report(self, report, output_file):
        """Save performance report to file"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")