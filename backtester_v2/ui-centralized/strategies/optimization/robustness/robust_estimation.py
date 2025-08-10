"""
Robust Estimation Methods for Strategy Optimization

Statistical robustness techniques to handle outliers and noise in optimization.
Adapted from the enhanced market regime optimizer.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class RobustEstimate:
    """Results from robust estimation"""
    method: str
    estimate: float
    confidence_interval: Tuple[float, float]
    outliers_detected: int
    robust_score: float
    standard_estimate: float
    improvement: float

class RobustEstimation:
    """
    Robust statistical estimation methods for optimization
    
    Provides methods to estimate optimization results that are resistant
    to outliers and noise in the objective function evaluations.
    """
    
    def __init__(self,
                 estimation_methods: List[str] = None,
                 outlier_threshold: float = 2.5,
                 confidence_level: float = 0.95,
                 min_samples: int = 10):
        """
        Initialize robust estimation
        
        Args:
            estimation_methods: Methods to use ('median', 'trimmed_mean', 'huber', 'winsorized')
            outlier_threshold: Z-score threshold for outlier detection
            confidence_level: Confidence level for intervals
            min_samples: Minimum samples required for estimation
        """
        self.estimation_methods = estimation_methods or ['median', 'trimmed_mean', 'huber']
        self.outlier_threshold = outlier_threshold
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        
        # Validate parameters
        valid_methods = ['median', 'trimmed_mean', 'huber', 'winsorized', 'mad']
        for method in self.estimation_methods:
            if method not in valid_methods:
                raise ValueError(f"Unknown estimation method: {method}. Valid methods: {valid_methods}")
        
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("Confidence level must be between 0 and 1")
        
        logger.info(f"Initialized robust estimation with methods: {self.estimation_methods}")
    
    def detect_outliers(self, 
                       values: List[float], 
                       method: str = 'z_score') -> Tuple[List[int], List[float]]:
        """
        Detect outliers in a list of values
        
        Args:
            values: List of values to analyze
            method: Detection method ('z_score', 'iqr', 'mad')
            
        Returns:
            Tuple of (outlier_indices, outlier_values)
        """
        if len(values) < self.min_samples:
            return [], []
        
        values_array = np.array(values)
        outlier_indices = []
        outlier_values = []
        
        if method == 'z_score':
            # Z-score based detection
            z_scores = np.abs(stats.zscore(values_array))
            outlier_mask = z_scores > self.outlier_threshold
            outlier_indices = np.where(outlier_mask)[0].tolist()
            outlier_values = values_array[outlier_mask].tolist()
            
        elif method == 'iqr':
            # Interquartile range based detection
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (values_array < lower_bound) | (values_array > upper_bound)
            outlier_indices = np.where(outlier_mask)[0].tolist()
            outlier_values = values_array[outlier_mask].tolist()
            
        elif method == 'mad':
            # Median Absolute Deviation based detection
            median = np.median(values_array)
            mad = np.median(np.abs(values_array - median))
            
            if mad == 0:
                return [], []
            
            modified_z_scores = 0.6745 * (values_array - median) / mad
            outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
            outlier_indices = np.where(outlier_mask)[0].tolist()
            outlier_values = values_array[outlier_mask].tolist()
        
        return outlier_indices, outlier_values
    
    def robust_mean_estimation(self, 
                              values: List[float], 
                              method: str = 'trimmed_mean') -> RobustEstimate:
        """
        Robust estimation of mean value
        
        Args:
            values: List of values
            method: Estimation method
            
        Returns:
            RobustEstimate object
        """
        if len(values) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples for robust estimation")
        
        values_array = np.array(values)
        standard_estimate = np.mean(values_array)
        
        # Detect outliers
        outlier_indices, _ = self.detect_outliers(values)
        
        if method == 'median':
            # Median (most robust)
            robust_estimate = np.median(values_array)
            
            # Bootstrap confidence interval for median
            n_bootstrap = 1000
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(values_array, size=len(values_array), replace=True)
                bootstrap_medians.append(np.median(bootstrap_sample))
            
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))
            
        elif method == 'trimmed_mean':
            # Trimmed mean (remove extreme values)
            trim_fraction = 0.2  # Remove 20% from each tail
            robust_estimate = stats.trim_mean(values_array, trim_fraction)
            
            # Confidence interval for trimmed mean
            se = stats.sem(values_array)  # Standard error
            t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(values_array) - 1)
            margin = t_value * se
            ci_lower = robust_estimate - margin
            ci_upper = robust_estimate + margin
            
        elif method == 'huber':
            # Huber M-estimator
            try:
                from scipy.stats import huber
                huber_result = huber(values_array)
                robust_estimate = huber_result.loc
                scale = huber_result.scale
                
                # Approximate confidence interval
                se = scale / np.sqrt(len(values_array))
                t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(values_array) - 1)
                margin = t_value * se
                ci_lower = robust_estimate - margin
                ci_upper = robust_estimate + margin
                
            except ImportError:
                logger.warning("Scipy huber estimator not available, falling back to trimmed mean")
                return self.robust_mean_estimation(values, 'trimmed_mean')
            
        elif method == 'winsorized':
            # Winsorized mean (replace extreme values)
            winsorized_data = stats.mstats.winsorize(values_array, limits=[0.1, 0.1])
            robust_estimate = np.mean(winsorized_data)
            
            # Confidence interval
            se = np.std(winsorized_data) / np.sqrt(len(winsorized_data))
            t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(winsorized_data) - 1)
            margin = t_value * se
            ci_lower = robust_estimate - margin
            ci_upper = robust_estimate + margin
            
        else:
            raise ValueError(f"Unknown robust estimation method: {method}")
        
        # Calculate robust score (how much better than standard estimate)
        improvement = abs(robust_estimate - standard_estimate) / abs(standard_estimate) if standard_estimate != 0 else 0
        robust_score = 1.0 / (1.0 + len(outlier_indices) / len(values))  # Penalty for outliers
        
        return RobustEstimate(
            method=method,
            estimate=robust_estimate,
            confidence_interval=(ci_lower, ci_upper),
            outliers_detected=len(outlier_indices),
            robust_score=robust_score,
            standard_estimate=standard_estimate,
            improvement=improvement
        )
    
    def robust_optimization_score(self,
                                 optimization_history: List[float],
                                 convergence_window: int = 50) -> Dict[str, Any]:
        """
        Calculate robust optimization score from convergence history
        
        Args:
            optimization_history: List of objective function values over iterations
            convergence_window: Window size for convergence analysis
            
        Returns:
            Dictionary with robust optimization metrics
        """
        if len(optimization_history) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} iterations for robust analysis")
        
        # Analyze final convergence window
        if len(optimization_history) >= convergence_window:
            final_values = optimization_history[-convergence_window:]
        else:
            final_values = optimization_history
        
        # Get robust estimates using all methods
        robust_estimates = {}
        for method in self.estimation_methods:
            try:
                estimate = self.robust_mean_estimation(final_values, method)
                robust_estimates[method] = estimate
            except Exception as e:
                logger.warning(f"Error in robust estimation method {method}: {e}")
        
        if not robust_estimates:
            raise ValueError("No robust estimation methods succeeded")
        
        # Select best estimate (highest robust score)
        best_method = max(robust_estimates.keys(), 
                         key=lambda m: robust_estimates[m].robust_score)
        best_estimate = robust_estimates[best_method]
        
        # Calculate convergence metrics
        convergence_stability = self._calculate_convergence_stability(optimization_history)
        noise_level = self._calculate_noise_level(optimization_history)
        
        # Detect optimization plateaus
        plateaus = self._detect_plateaus(optimization_history)
        
        return {
            'best_robust_estimate': best_estimate.estimate,
            'best_method': best_method,
            'confidence_interval': best_estimate.confidence_interval,
            'outliers_detected': best_estimate.outliers_detected,
            'robust_score': best_estimate.robust_score,
            'convergence_stability': convergence_stability,
            'noise_level': noise_level,
            'plateaus_detected': len(plateaus),
            'all_estimates': {
                method: {
                    'estimate': est.estimate,
                    'confidence_interval': est.confidence_interval,
                    'robust_score': est.robust_score
                }
                for method, est in robust_estimates.items()
            }
        }
    
    def _calculate_convergence_stability(self, history: List[float]) -> float:
        """Calculate stability of convergence"""
        if len(history) < 10:
            return 0.0
        
        # Calculate moving averages
        window_size = min(10, len(history) // 4)
        moving_avg = pd.Series(history).rolling(window=window_size).mean().dropna()
        
        if len(moving_avg) < 2:
            return 0.0
        
        # Calculate coefficient of variation of moving averages
        cv = np.std(moving_avg) / np.mean(moving_avg) if np.mean(moving_avg) != 0 else 1.0
        
        # Stability score (lower CV = higher stability)
        stability = 1.0 / (1.0 + cv)
        return stability
    
    def _calculate_noise_level(self, history: List[float]) -> float:
        """Calculate noise level in optimization history"""
        if len(history) < 3:
            return 0.0
        
        # Calculate first differences
        diffs = np.diff(history)
        
        # Noise level as normalized standard deviation of differences
        noise = np.std(diffs) / (np.mean(np.abs(history)) + 1e-8)
        return noise
    
    def _detect_plateaus(self, history: List[float], 
                        min_length: int = 10, 
                        tolerance: float = 0.01) -> List[Tuple[int, int]]:
        """Detect plateau regions in optimization history"""
        if len(history) < min_length:
            return []
        
        plateaus = []
        current_plateau_start = None
        
        for i in range(1, len(history)):
            # Check if current value is within tolerance of previous
            relative_change = abs(history[i] - history[i-1]) / (abs(history[i-1]) + 1e-8)
            
            if relative_change <= tolerance:
                if current_plateau_start is None:
                    current_plateau_start = i - 1
            else:
                if current_plateau_start is not None:
                    plateau_length = i - current_plateau_start
                    if plateau_length >= min_length:
                        plateaus.append((current_plateau_start, i - 1))
                    current_plateau_start = None
        
        # Check for plateau at the end
        if current_plateau_start is not None:
            plateau_length = len(history) - current_plateau_start
            if plateau_length >= min_length:
                plateaus.append((current_plateau_start, len(history) - 1))
        
        return plateaus
    
    def robust_parameter_estimation(self,
                                   parameter_samples: Dict[str, List[float]]) -> Dict[str, RobustEstimate]:
        """
        Robust estimation for multiple parameters
        
        Args:
            parameter_samples: Dictionary of parameter name to list of sample values
            
        Returns:
            Dictionary of robust estimates for each parameter
        """
        robust_params = {}
        
        for param_name, samples in parameter_samples.items():
            if len(samples) >= self.min_samples:
                try:
                    # Use the method with highest robust score
                    best_estimate = None
                    best_score = -1
                    
                    for method in self.estimation_methods:
                        try:
                            estimate = self.robust_mean_estimation(samples, method)
                            if estimate.robust_score > best_score:
                                best_estimate = estimate
                                best_score = estimate.robust_score
                        except Exception as e:
                            logger.warning(f"Error in robust estimation for {param_name} with {method}: {e}")
                    
                    if best_estimate:
                        robust_params[param_name] = best_estimate
                    
                except Exception as e:
                    logger.error(f"Error in robust parameter estimation for {param_name}: {e}")
            else:
                logger.warning(f"Insufficient samples for robust estimation of {param_name}")
        
        return robust_params
    
    def create_robustness_report(self,
                                optimization_history: List[float],
                                parameter_samples: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Create comprehensive robustness report
        
        Args:
            optimization_history: Optimization convergence history
            parameter_samples: Optional parameter samples from multiple runs
            
        Returns:
            Comprehensive robustness report
        """
        report = {
            'optimization_analysis': self.robust_optimization_score(optimization_history),
            'outlier_analysis': {},
            'stability_metrics': {}
        }
        
        # Outlier analysis
        outlier_indices, outlier_values = self.detect_outliers(optimization_history)
        report['outlier_analysis'] = {
            'total_outliers': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(optimization_history) * 100,
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values
        }
        
        # Parameter robustness (if provided)
        if parameter_samples:
            param_robustness = self.robust_parameter_estimation(parameter_samples)
            report['parameter_robustness'] = {
                param_name: {
                    'robust_estimate': est.estimate,
                    'confidence_interval': est.confidence_interval,
                    'outliers_detected': est.outliers_detected,
                    'robust_score': est.robust_score,
                    'method_used': est.method
                }
                for param_name, est in param_robustness.items()
            }
        
        # Overall robustness score
        opt_analysis = report['optimization_analysis']
        overall_score = (
            opt_analysis['robust_score'] * 0.4 +
            opt_analysis['convergence_stability'] * 0.3 +
            (1.0 - min(opt_analysis['noise_level'], 1.0)) * 0.3
        )
        
        report['overall_robustness_score'] = overall_score
        
        logger.info(f"Robustness analysis completed. Overall score: {overall_score:.3f}")
        
        return report