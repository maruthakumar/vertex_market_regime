"""
Sensitivity Analysis for Strategy Optimization

Analyzes how sensitive optimization results are to parameter variations.
Adapted from the enhanced market regime optimizer.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class SensitivityResult:
    """Results from sensitivity analysis"""
    parameter_name: str
    perturbation_level: float
    original_value: float
    perturbed_values: List[float]
    score_changes: List[float]
    sensitivity_score: float
    elasticity: float

class SensitivityAnalysis:
    """
    Sensitivity analysis for optimization parameters
    
    Analyzes how robust optimization results are to parameter variations
    by systematically perturbing parameters and measuring score changes.
    """
    
    def __init__(self,
                 perturbation_levels: List[float] = None,
                 perturbation_method: str = 'relative',
                 n_samples: int = 10,
                 parallel_analysis: bool = True,
                 confidence_level: float = 0.95):
        """
        Initialize sensitivity analysis
        
        Args:
            perturbation_levels: Levels of perturbation to test
            perturbation_method: Method ('relative', 'absolute', 'percentage')
            n_samples: Number of samples per perturbation level
            parallel_analysis: Whether to run analysis in parallel
            confidence_level: Confidence level for sensitivity intervals
        """
        self.perturbation_levels = perturbation_levels or [0.01, 0.05, 0.1, 0.2]
        self.perturbation_method = perturbation_method
        self.n_samples = n_samples
        self.parallel_analysis = parallel_analysis
        self.confidence_level = confidence_level
        
        # Validate parameters
        valid_methods = ['relative', 'absolute', 'percentage']
        if perturbation_method not in valid_methods:
            raise ValueError(f"Perturbation method must be one of {valid_methods}")
        
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("Confidence level must be between 0 and 1")
        
        logger.info(f"Initialized sensitivity analysis with {len(self.perturbation_levels)} perturbation levels")
    
    def analyze_parameter_sensitivity(self,
                                    optimizer: 'BaseOptimizer',
                                    best_params: Dict[str, float],
                                    param_name: str,
                                    param_bounds: Tuple[float, float]) -> SensitivityResult:
        """
        Analyze sensitivity of a single parameter
        
        Args:
            optimizer: Optimizer instance with objective function
            best_params: Best parameters from optimization
            param_name: Name of parameter to analyze
            param_bounds: (min, max) bounds for the parameter
            
        Returns:
            SensitivityResult for the parameter
        """
        original_value = best_params[param_name]
        min_val, max_val = param_bounds
        
        all_perturbed_values = []
        all_score_changes = []
        
        # Test each perturbation level
        for perturbation_level in self.perturbation_levels:
            # Generate perturbed values
            perturbed_values = self._generate_perturbed_values(
                original_value, perturbation_level, min_val, max_val
            )
            
            # Evaluate each perturbed value
            score_changes = []
            for perturbed_value in perturbed_values:
                # Create perturbed parameter set
                perturbed_params = best_params.copy()
                perturbed_params[param_name] = perturbed_value
                
                # Evaluate
                try:
                    perturbed_score = optimizer.objective_function(perturbed_params)
                    original_score = optimizer.objective_function(best_params)
                    
                    # Calculate relative change
                    if original_score != 0:
                        score_change = (perturbed_score - original_score) / abs(original_score)
                    else:
                        score_change = perturbed_score - original_score
                    
                    score_changes.append(score_change)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating perturbed parameter {param_name}={perturbed_value}: {e}")
                    score_changes.append(0.0)
            
            all_perturbed_values.extend(perturbed_values)
            all_score_changes.extend(score_changes)
        
        # Calculate sensitivity metrics
        sensitivity_score = np.std(all_score_changes) if all_score_changes else 0.0
        
        # Calculate elasticity (average percentage change in score per percentage change in parameter)
        param_changes = [(v - original_value) / original_value for v in all_perturbed_values if original_value != 0]
        if param_changes and all_score_changes:
            elasticity = np.mean([sc / pc for sc, pc in zip(all_score_changes, param_changes) if pc != 0])
        else:
            elasticity = 0.0
        
        return SensitivityResult(
            parameter_name=param_name,
            perturbation_level=max(self.perturbation_levels),
            original_value=original_value,
            perturbed_values=all_perturbed_values,
            score_changes=all_score_changes,
            sensitivity_score=sensitivity_score,
            elasticity=elasticity
        )
    
    def _generate_perturbed_values(self,
                                  original_value: float,
                                  perturbation_level: float,
                                  min_val: float,
                                  max_val: float) -> List[float]:
        """Generate perturbed values for a parameter"""
        
        perturbed_values = []
        
        if self.perturbation_method == 'relative':
            # Relative perturbation
            delta = perturbation_level * abs(original_value)
        elif self.perturbation_method == 'absolute':
            # Absolute perturbation
            delta = perturbation_level
        elif self.perturbation_method == 'percentage':
            # Percentage of parameter range
            param_range = max_val - min_val
            delta = perturbation_level * param_range
        
        # Generate symmetric perturbations
        for i in range(self.n_samples):
            if i < self.n_samples // 2:
                # Positive perturbations
                factor = (i + 1) / (self.n_samples // 2)
                perturbed_value = original_value + factor * delta
            else:
                # Negative perturbations
                factor = (i - self.n_samples // 2 + 1) / (self.n_samples - self.n_samples // 2)
                perturbed_value = original_value - factor * delta
            
            # Clip to bounds
            perturbed_value = np.clip(perturbed_value, min_val, max_val)
            perturbed_values.append(perturbed_value)
        
        return perturbed_values
    
    def analyze_all_parameters(self,
                              optimizer: 'BaseOptimizer',
                              best_params: Dict[str, float],
                              param_space: Dict[str, Tuple[float, float]]) -> Dict[str, SensitivityResult]:
        """
        Analyze sensitivity for all parameters
        
        Args:
            optimizer: Optimizer instance
            best_params: Best parameters from optimization
            param_space: Parameter space with bounds
            
        Returns:
            Dictionary of sensitivity results by parameter name
        """
        start_time = time.time()
        
        logger.info(f"Starting sensitivity analysis for {len(param_space)} parameters")
        
        sensitivity_results = {}
        
        if self.parallel_analysis:
            # Parallel analysis
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_param = {
                    executor.submit(
                        self.analyze_parameter_sensitivity,
                        optimizer, best_params, param_name, bounds
                    ): param_name
                    for param_name, bounds in param_space.items()
                }
                
                for future in as_completed(future_to_param):
                    param_name = future_to_param[future]
                    try:
                        result = future.result()
                        sensitivity_results[param_name] = result
                        logger.info(f"Completed sensitivity analysis for {param_name}: "
                                  f"sensitivity={result.sensitivity_score:.4f}")
                    except Exception as e:
                        logger.error(f"Error in sensitivity analysis for {param_name}: {e}")
        else:
            # Sequential analysis
            for param_name, bounds in param_space.items():
                try:
                    result = self.analyze_parameter_sensitivity(
                        optimizer, best_params, param_name, bounds
                    )
                    sensitivity_results[param_name] = result
                    logger.info(f"Completed sensitivity analysis for {param_name}: "
                              f"sensitivity={result.sensitivity_score:.4f}")
                except Exception as e:
                    logger.error(f"Error in sensitivity analysis for {param_name}: {e}")
        
        analysis_time = time.time() - start_time
        logger.info(f"Sensitivity analysis completed in {analysis_time:.2f} seconds")
        
        return sensitivity_results
    
    def rank_parameter_sensitivity(self,
                                  sensitivity_results: Dict[str, SensitivityResult],
                                  ranking_method: str = 'sensitivity') -> List[Tuple[str, float]]:
        """
        Rank parameters by sensitivity
        
        Args:
            sensitivity_results: Results from analyze_all_parameters
            ranking_method: Ranking method ('sensitivity', 'elasticity', 'variance')
            
        Returns:
            List of (parameter_name, score) tuples sorted by sensitivity
        """
        rankings = []
        
        for param_name, result in sensitivity_results.items():
            if ranking_method == 'sensitivity':
                score = result.sensitivity_score
            elif ranking_method == 'elasticity':
                score = abs(result.elasticity)
            elif ranking_method == 'variance':
                score = np.var(result.score_changes) if result.score_changes else 0.0
            else:
                raise ValueError(f"Unknown ranking method: {ranking_method}")
            
            rankings.append((param_name, score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def get_robust_parameters(self,
                             sensitivity_results: Dict[str, SensitivityResult],
                             sensitivity_threshold: float = 0.1) -> Dict[str, float]:
        """
        Identify robust parameters with low sensitivity
        
        Args:
            sensitivity_results: Results from analyze_all_parameters
            sensitivity_threshold: Maximum sensitivity for robust parameters
            
        Returns:
            Dictionary of robust parameters and their values
        """
        robust_params = {}
        
        for param_name, result in sensitivity_results.items():
            if result.sensitivity_score <= sensitivity_threshold:
                robust_params[param_name] = result.original_value
        
        logger.info(f"Found {len(robust_params)} robust parameters (sensitivity <= {sensitivity_threshold})")
        
        return robust_params
    
    def analyze_parameter_interactions(self,
                                     optimizer: 'BaseOptimizer',
                                     best_params: Dict[str, float],
                                     param_pairs: List[Tuple[str, str]],
                                     param_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Analyze interactions between parameter pairs
        
        Args:
            optimizer: Optimizer instance
            best_params: Best parameters
            param_pairs: List of parameter pairs to analyze
            param_space: Parameter space with bounds
            
        Returns:
            Dictionary of interaction strengths
        """
        logger.info(f"Analyzing {len(param_pairs)} parameter interactions")
        
        interaction_strengths = {}
        
        for param1, param2 in param_pairs:
            if param1 not in param_space or param2 not in param_space:
                continue
            
            # Get baseline score
            baseline_score = optimizer.objective_function(best_params)
            
            # Test individual perturbations
            params1_perturbed = best_params.copy()
            params2_perturbed = best_params.copy()
            
            # Small perturbations
            perturbation = 0.05  # 5% perturbation
            
            # Perturb param1
            original1 = best_params[param1]
            min1, max1 = param_space[param1]
            delta1 = perturbation * (max1 - min1)
            params1_perturbed[param1] = np.clip(original1 + delta1, min1, max1)
            
            # Perturb param2
            original2 = best_params[param2]
            min2, max2 = param_space[param2]
            delta2 = perturbation * (max2 - min2)
            params2_perturbed[param2] = np.clip(original2 + delta2, min2, max2)
            
            # Test combined perturbation
            params_both_perturbed = best_params.copy()
            params_both_perturbed[param1] = params1_perturbed[param1]
            params_both_perturbed[param2] = params2_perturbed[param2]
            
            try:
                score1 = optimizer.objective_function(params1_perturbed)
                score2 = optimizer.objective_function(params2_perturbed)
                score_both = optimizer.objective_function(params_both_perturbed)
                
                # Calculate interaction effect
                expected_combined = (score1 - baseline_score) + (score2 - baseline_score) + baseline_score
                actual_combined = score_both
                
                interaction_effect = abs(actual_combined - expected_combined)
                interaction_strengths[f"{param1}_{param2}"] = interaction_effect
                
            except Exception as e:
                logger.warning(f"Error analyzing interaction {param1}-{param2}: {e}")
                interaction_strengths[f"{param1}_{param2}"] = 0.0
        
        return interaction_strengths
    
    def create_sensitivity_report(self,
                                 sensitivity_results: Dict[str, SensitivityResult],
                                 best_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Create comprehensive sensitivity analysis report
        
        Args:
            sensitivity_results: Results from analyze_all_parameters
            best_params: Best parameters from optimization
            
        Returns:
            Comprehensive sensitivity report
        """
        # Parameter rankings
        sensitivity_ranking = self.rank_parameter_sensitivity(sensitivity_results, 'sensitivity')
        elasticity_ranking = self.rank_parameter_sensitivity(sensitivity_results, 'elasticity')
        
        # Robust parameters
        robust_params = self.get_robust_parameters(sensitivity_results)
        
        # Summary statistics
        all_sensitivities = [r.sensitivity_score for r in sensitivity_results.values()]
        all_elasticities = [r.elasticity for r in sensitivity_results.values()]
        
        report = {
            'summary': {
                'total_parameters': len(sensitivity_results),
                'robust_parameters': len(robust_params),
                'mean_sensitivity': np.mean(all_sensitivities),
                'max_sensitivity': np.max(all_sensitivities),
                'mean_elasticity': np.mean(all_elasticities),
                'max_elasticity': np.max(all_elasticities)
            },
            'rankings': {
                'by_sensitivity': sensitivity_ranking,
                'by_elasticity': elasticity_ranking
            },
            'robust_parameters': robust_params,
            'detailed_results': {
                param_name: {
                    'sensitivity_score': result.sensitivity_score,
                    'elasticity': result.elasticity,
                    'original_value': result.original_value,
                    'score_variance': np.var(result.score_changes) if result.score_changes else 0.0
                }
                for param_name, result in sensitivity_results.items()
            }
        }
        
        logger.info(f"Created sensitivity report: {len(sensitivity_results)} parameters analyzed")
        logger.info(f"Most sensitive parameter: {sensitivity_ranking[0][0]} ({sensitivity_ranking[0][1]:.4f})")
        logger.info(f"Robust parameters: {len(robust_params)}/{len(sensitivity_results)}")
        
        return report