"""
Robust Optimizer Wrapper

Combines all robustness components into a unified optimization framework.
Provides comprehensive robustness analysis and enhanced optimization results.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import json

from ..base.base_optimizer import BaseOptimizer, OptimizationResult
from .cross_validation import CrossValidation
from .sensitivity_analysis import SensitivityAnalysis
from .robust_estimation import RobustEstimation
from .dimension_testing import DimensionTesting

logger = logging.getLogger(__name__)

@dataclass
class RobustOptimizationResult:
    """Enhanced optimization result with robustness analysis"""
    base_result: OptimizationResult
    cv_results: Optional[Dict[str, Any]] = None
    sensitivity_results: Optional[Dict[str, Any]] = None
    robustness_analysis: Optional[Dict[str, Any]] = None
    dimension_analysis: Optional[Dict[str, Any]] = None
    overall_robustness_score: float = 0.0
    robust_parameters: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None

class RobustOptimizer:
    """
    Robust optimization framework that combines all robustness components
    
    Provides a comprehensive optimization approach with:
    - Cross-validation for temporal stability
    - Sensitivity analysis for parameter robustness
    - Robust estimation for noise handling
    - Dimension testing for optimal feature selection
    """
    
    def __init__(self,
                 base_optimizer: BaseOptimizer,
                 enable_cross_validation: bool = True,
                 enable_sensitivity_analysis: bool = True,
                 enable_robust_estimation: bool = True,
                 enable_dimension_testing: bool = False,
                 cv_config: Optional[Dict[str, Any]] = None,
                 sensitivity_config: Optional[Dict[str, Any]] = None,
                 robustness_config: Optional[Dict[str, Any]] = None,
                 dimension_config: Optional[Dict[str, Any]] = None):
        """
        Initialize robust optimizer
        
        Args:
            base_optimizer: Base optimization algorithm
            enable_cross_validation: Whether to perform cross-validation
            enable_sensitivity_analysis: Whether to perform sensitivity analysis
            enable_robust_estimation: Whether to perform robust estimation
            enable_dimension_testing: Whether to perform dimension testing
            cv_config: Cross-validation configuration
            sensitivity_config: Sensitivity analysis configuration
            robustness_config: Robust estimation configuration
            dimension_config: Dimension testing configuration
        """
        self.base_optimizer = base_optimizer
        self.enable_cross_validation = enable_cross_validation
        self.enable_sensitivity_analysis = enable_sensitivity_analysis
        self.enable_robust_estimation = enable_robust_estimation
        self.enable_dimension_testing = enable_dimension_testing
        
        # Initialize robustness components
        self.cross_validator = None
        self.sensitivity_analyzer = None
        self.robust_estimator = None
        self.dimension_tester = None
        
        if enable_cross_validation:
            cv_params = cv_config or {}
            self.cross_validator = CrossValidation(**cv_params)
        
        if enable_sensitivity_analysis:
            sens_params = sensitivity_config or {}
            self.sensitivity_analyzer = SensitivityAnalysis(**sens_params)
        
        if enable_robust_estimation:
            robust_params = robustness_config or {}
            self.robust_estimator = RobustEstimation(**robust_params)
        
        if enable_dimension_testing:
            dim_params = dimension_config or {}
            self.dimension_tester = DimensionTesting(**dim_params)
        
        logger.info(f"Initialized robust optimizer with {base_optimizer.__class__.__name__}")
        logger.info(f"Enabled components: CV={enable_cross_validation}, "
                   f"Sens={enable_sensitivity_analysis}, Robust={enable_robust_estimation}, "
                   f"Dim={enable_dimension_testing}")
    
    def optimize_with_robustness(self,
                                data: Optional[pd.DataFrame] = None,
                                date_column: str = 'date',
                                available_dimensions: Optional[List[str]] = None,
                                n_iterations: int = 1000,
                                callback: Optional[Callable] = None,
                                **kwargs) -> RobustOptimizationResult:
        """
        Run optimization with comprehensive robustness analysis
        
        Args:
            data: Time-series data for cross-validation and dimension testing
            date_column: Date column name for time-series analysis
            available_dimensions: Available dimensions for dimension testing
            n_iterations: Number of optimization iterations
            callback: Optional callback function
            **kwargs: Additional optimization parameters
            
        Returns:
            RobustOptimizationResult with comprehensive analysis
        """
        start_time = time.time()
        
        logger.info("Starting robust optimization")
        
        # Phase 1: Dimension testing (if enabled and data provided)
        dimension_analysis = None
        optimal_dimensions = None
        
        if (self.enable_dimension_testing and 
            self.dimension_tester and 
            data is not None and 
            available_dimensions):
            
            logger.info("Phase 1: Dimension testing")
            try:
                dimension_analysis = self.dimension_tester.optimize_dimensions(
                    self.base_optimizer, data, available_dimensions
                )
                optimal_dimensions = dimension_analysis['best_dimensions']
                logger.info(f"Optimal dimensions selected: {optimal_dimensions}")
            except Exception as e:
                logger.error(f"Error in dimension testing: {e}")
        
        # Phase 2: Base optimization
        logger.info("Phase 2: Base optimization")
        base_result = self.base_optimizer.optimize(
            n_iterations=n_iterations,
            callback=callback,
            **kwargs
        )
        
        logger.info(f"Base optimization completed: score={base_result.best_score:.6f}")
        
        # Phase 3: Cross-validation (if enabled and data provided)
        cv_results = None
        if self.enable_cross_validation and self.cross_validator and data is not None:
            logger.info("Phase 3: Cross-validation analysis")
            try:
                cv_results = self.cross_validator.optimize_with_cv(
                    self.base_optimizer,
                    data,
                    date_column=date_column,
                    n_iterations=n_iterations // 2  # Shorter runs for CV
                )
                logger.info(f"Cross-validation completed: "
                          f"test score={cv_results['test_mean']:.6f} ± {cv_results['test_std']:.6f}")
            except Exception as e:
                logger.error(f"Error in cross-validation: {e}")
        
        # Phase 4: Sensitivity analysis
        sensitivity_results = None
        if self.enable_sensitivity_analysis and self.sensitivity_analyzer:
            logger.info("Phase 4: Sensitivity analysis")
            try:
                sensitivity_analysis = self.sensitivity_analyzer.analyze_all_parameters(
                    self.base_optimizer,
                    base_result.best_params,
                    self.base_optimizer.param_space
                )
                sensitivity_results = self.sensitivity_analyzer.create_sensitivity_report(
                    sensitivity_analysis,
                    base_result.best_params
                )
                logger.info(f"Sensitivity analysis completed: "
                          f"{sensitivity_results['summary']['robust_parameters']} robust parameters")
            except Exception as e:
                logger.error(f"Error in sensitivity analysis: {e}")
        
        # Phase 5: Robust estimation
        robustness_analysis = None
        if self.enable_robust_estimation and self.robust_estimator:
            logger.info("Phase 5: Robust estimation")
            try:
                robustness_analysis = self.robust_estimator.create_robustness_report(
                    base_result.convergence_history
                )
                logger.info(f"Robust estimation completed: "
                          f"robustness score={robustness_analysis['overall_robustness_score']:.3f}")
            except Exception as e:
                logger.error(f"Error in robust estimation: {e}")
        
        # Phase 6: Synthesize results
        logger.info("Phase 6: Synthesizing robust results")
        
        # Calculate overall robustness score
        overall_robustness_score = self._calculate_overall_robustness_score(
            base_result, cv_results, sensitivity_results, robustness_analysis
        )
        
        # Determine robust parameters
        robust_parameters = self._determine_robust_parameters(
            base_result.best_params, cv_results, sensitivity_results
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            base_result.best_params, cv_results, sensitivity_results
        )
        
        # Create comprehensive result
        robust_result = RobustOptimizationResult(
            base_result=base_result,
            cv_results=cv_results,
            sensitivity_results=sensitivity_results,
            robustness_analysis=robustness_analysis,
            dimension_analysis=dimension_analysis,
            overall_robustness_score=overall_robustness_score,
            robust_parameters=robust_parameters,
            confidence_intervals=confidence_intervals
        )
        
        total_time = time.time() - start_time
        logger.info(f"Robust optimization completed in {total_time:.2f} seconds")
        logger.info(f"Overall robustness score: {overall_robustness_score:.3f}")
        
        return robust_result
    
    def _calculate_overall_robustness_score(self,
                                          base_result: OptimizationResult,
                                          cv_results: Optional[Dict[str, Any]],
                                          sensitivity_results: Optional[Dict[str, Any]],
                                          robustness_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate overall robustness score from all analyses"""
        
        scores = []
        weights = []
        
        # Base optimization quality (convergence)
        if base_result.convergence_history:
            convergence_score = self._assess_convergence_quality(base_result.convergence_history)
            scores.append(convergence_score)
            weights.append(0.2)
        
        # Cross-validation stability
        if cv_results:
            stability_ratio = cv_results.get('stability_ratio', 0.5)
            cv_score = min(stability_ratio, 1.0) if stability_ratio > 0 else 0.5
            scores.append(cv_score)
            weights.append(0.3)
        
        # Sensitivity robustness
        if sensitivity_results:
            total_params = sensitivity_results['summary']['total_parameters']
            robust_params = sensitivity_results['summary']['robust_parameters']
            sensitivity_score = robust_params / total_params if total_params > 0 else 0.5
            scores.append(sensitivity_score)
            weights.append(0.3)
        
        # Robust estimation score
        if robustness_analysis:
            robust_score = robustness_analysis.get('overall_robustness_score', 0.5)
            scores.append(robust_score)
            weights.append(0.2)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            overall_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            overall_score = 0.5  # Default neutral score
        
        return overall_score
    
    def _assess_convergence_quality(self, convergence_history: List[float]) -> float:
        """Assess quality of convergence from optimization history"""
        if len(convergence_history) < 10:
            return 0.5
        
        # Check for improvement trend
        first_quarter = convergence_history[:len(convergence_history)//4]
        last_quarter = convergence_history[-len(convergence_history)//4:]
        
        if len(first_quarter) > 0 and len(last_quarter) > 0:
            improvement = (np.mean(last_quarter) - np.mean(first_quarter)) / abs(np.mean(first_quarter))
            improvement_score = min(max(improvement, -1.0), 1.0) / 2.0 + 0.5
        else:
            improvement_score = 0.5
        
        # Check for stability in final convergence
        final_portion = convergence_history[-len(convergence_history)//5:]
        if len(final_portion) > 1:
            cv = np.std(final_portion) / (np.mean(final_portion) + 1e-8)
            stability_score = 1.0 / (1.0 + cv)
        else:
            stability_score = 0.5
        
        return 0.6 * improvement_score + 0.4 * stability_score
    
    def _determine_robust_parameters(self,
                                   best_params: Dict[str, float],
                                   cv_results: Optional[Dict[str, Any]],
                                   sensitivity_results: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Determine robust parameter values considering all analyses"""
        
        robust_params = best_params.copy()
        
        # Use CV results if available
        if cv_results and 'fold_results' in cv_results:
            # Average parameters across CV folds for stability
            all_fold_params = [fold['params'] for fold in cv_results['fold_results']]
            
            for param_name in best_params.keys():
                param_values = [params[param_name] for params in all_fold_params if param_name in params]
                if param_values:
                    # Use median for robustness
                    robust_params[param_name] = np.median(param_values)
        
        # Adjust based on sensitivity analysis
        if sensitivity_results and 'robust_parameters' in sensitivity_results:
            for param_name, robust_value in sensitivity_results['robust_parameters'].items():
                if param_name in robust_params:
                    # Weight between original and robust estimate
                    original_value = robust_params[param_name]
                    robust_params[param_name] = 0.7 * robust_value + 0.3 * original_value
        
        return robust_params
    
    def _calculate_confidence_intervals(self,
                                      best_params: Dict[str, float],
                                      cv_results: Optional[Dict[str, Any]],
                                      sensitivity_results: Optional[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for parameters"""
        
        confidence_intervals = {}
        
        for param_name, param_value in best_params.items():
            lower_bound = param_value
            upper_bound = param_value
            
            # Use CV results for interval estimation
            if cv_results and 'fold_results' in cv_results:
                fold_values = [
                    fold['params'].get(param_name, param_value) 
                    for fold in cv_results['fold_results']
                ]
                if len(fold_values) > 1:
                    std_error = np.std(fold_values)
                    lower_bound = param_value - 1.96 * std_error
                    upper_bound = param_value + 1.96 * std_error
            
            # Adjust based on sensitivity analysis
            if (sensitivity_results and 
                'detailed_results' in sensitivity_results and
                param_name in sensitivity_results['detailed_results']):
                
                sensitivity_info = sensitivity_results['detailed_results'][param_name]
                sensitivity_score = sensitivity_info.get('sensitivity_score', 0.0)
                
                # Wider intervals for more sensitive parameters
                uncertainty_factor = 1.0 + sensitivity_score
                range_adjustment = (upper_bound - lower_bound) * uncertainty_factor * 0.5
                
                lower_bound = param_value - range_adjustment
                upper_bound = param_value + range_adjustment
            
            # Ensure bounds respect parameter space constraints
            if param_name in self.base_optimizer.param_space:
                min_val, max_val = self.base_optimizer.param_space[param_name]
                lower_bound = max(lower_bound, min_val)
                upper_bound = min(upper_bound, max_val)
            
            confidence_intervals[param_name] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def create_robustness_report(self, result: RobustOptimizationResult) -> Dict[str, Any]:
        """Create comprehensive robustness report"""
        
        report = {
            'executive_summary': {
                'algorithm_used': result.base_result.algorithm_name,
                'best_score': result.base_result.best_score,
                'overall_robustness_score': result.overall_robustness_score,
                'total_iterations': result.base_result.n_iterations,
                'execution_time': result.base_result.execution_time,
                'robustness_level': self._classify_robustness_level(result.overall_robustness_score)
            },
            'optimization_results': {
                'best_parameters': result.robust_parameters or result.base_result.best_params,
                'confidence_intervals': result.confidence_intervals,
                'convergence_quality': self._assess_convergence_quality(result.base_result.convergence_history)
            },
            'robustness_analysis': {},
            'recommendations': []
        }
        
        # Add component-specific analysis
        if result.cv_results:
            report['robustness_analysis']['cross_validation'] = {
                'test_score_mean': result.cv_results['test_mean'],
                'test_score_std': result.cv_results['test_std'],
                'stability_ratio': result.cv_results['stability_ratio'],
                'n_folds': result.cv_results['n_folds_completed']
            }
        
        if result.sensitivity_results:
            report['robustness_analysis']['sensitivity'] = {
                'robust_parameters': result.sensitivity_results['summary']['robust_parameters'],
                'total_parameters': result.sensitivity_results['summary']['total_parameters'],
                'most_sensitive': result.sensitivity_results['rankings']['by_sensitivity'][0] if result.sensitivity_results['rankings']['by_sensitivity'] else None
            }
        
        if result.robustness_analysis:
            report['robustness_analysis']['estimation'] = {
                'robustness_score': result.robustness_analysis['overall_robustness_score'],
                'outliers_detected': result.robustness_analysis.get('outlier_analysis', {}).get('total_outliers', 0),
                'convergence_stability': result.robustness_analysis['optimization_analysis']['convergence_stability']
            }
        
        if result.dimension_analysis:
            report['robustness_analysis']['dimensions'] = {
                'optimal_dimensions': result.dimension_analysis['best_dimensions'],
                'dimension_count': len(result.dimension_analysis['best_dimensions']),
                'method_used': result.dimension_analysis['method_used']
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_robustness_recommendations(result)
        
        return report
    
    def _classify_robustness_level(self, score: float) -> str:
        """Classify robustness level based on score"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Moderate"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_robustness_recommendations(self, result: RobustOptimizationResult) -> List[str]:
        """Generate recommendations based on robustness analysis"""
        recommendations = []
        
        # Overall robustness recommendations
        if result.overall_robustness_score < 0.6:
            recommendations.append("Consider using ensemble methods to improve robustness")
            recommendations.append("Increase cross-validation folds for better stability assessment")
        
        # Cross-validation recommendations
        if result.cv_results:
            stability_ratio = result.cv_results.get('stability_ratio', 1.0)
            if stability_ratio < 0.8:
                recommendations.append("High variance between train/test scores - consider regularization")
        
        # Sensitivity recommendations
        if result.sensitivity_results:
            robust_ratio = (result.sensitivity_results['summary']['robust_parameters'] / 
                          result.sensitivity_results['summary']['total_parameters'])
            if robust_ratio < 0.5:
                recommendations.append("Many parameters are sensitive - consider parameter constraints")
        
        # Convergence recommendations
        convergence_quality = self._assess_convergence_quality(result.base_result.convergence_history)
        if convergence_quality < 0.6:
            recommendations.append("Poor convergence quality - consider increasing iterations or changing algorithm")
        
        if not recommendations:
            recommendations.append("Optimization shows good robustness characteristics")
        
        return recommendations
    
    def save_robust_results(self, result: RobustOptimizationResult, filepath: str):
        """Save robust optimization results to JSON file"""
        
        # Convert result to serializable format
        serializable_result = {
            'base_result': result.base_result.to_dict(),
            'cv_results': result.cv_results,
            'sensitivity_results': result.sensitivity_results,
            'robustness_analysis': result.robustness_analysis,
            'dimension_analysis': result.dimension_analysis,
            'overall_robustness_score': result.overall_robustness_score,
            'robust_parameters': result.robust_parameters,
            'confidence_intervals': result.confidence_intervals
        }
        
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
            json.dump(serializable_result, f, cls=NumpyEncoder, indent=2)
        
        logger.info(f"Robust optimization results saved to {filepath}")