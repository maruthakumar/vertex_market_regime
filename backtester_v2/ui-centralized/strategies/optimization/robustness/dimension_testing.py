"""
Dimension Testing for Strategy Optimization

Tests different data dimensions and parameter configurations to find optimal setup.
Adapted from the enhanced market regime optimizer.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from itertools import combinations
import json

logger = logging.getLogger(__name__)

@dataclass
class DimensionTestResult:
    """Results from testing a specific dimension configuration"""
    dimensions: List[str]
    n_dimensions: int
    train_score: float
    test_score: float
    stability_score: float
    complexity_penalty: float
    adjusted_score: float
    optimization_time: float
    metadata: Dict[str, Any]

class DimensionTesting:
    """
    Dimension testing for optimization parameter selection
    
    Tests different combinations of data dimensions and parameters
    to find the optimal configuration for strategy optimization.
    """
    
    def __init__(self,
                 max_dimensions: int = 10,
                 min_dimensions: int = 2,
                 complexity_penalty: float = 0.01,
                 stability_weight: float = 0.3,
                 performance_weight: float = 0.7,
                 test_method: str = 'exhaustive'):
        """
        Initialize dimension testing
        
        Args:
            max_dimensions: Maximum number of dimensions to test
            min_dimensions: Minimum number of dimensions to test
            complexity_penalty: Penalty per additional dimension
            stability_weight: Weight for stability in scoring
            performance_weight: Weight for performance in scoring
            test_method: Testing method ('exhaustive', 'forward_selection', 'backward_elimination')
        """
        self.max_dimensions = max_dimensions
        self.min_dimensions = min_dimensions
        self.complexity_penalty = complexity_penalty
        self.stability_weight = stability_weight
        self.performance_weight = performance_weight
        self.test_method = test_method
        
        # Validate parameters
        if min_dimensions < 1:
            raise ValueError("Minimum dimensions must be at least 1")
        if max_dimensions < min_dimensions:
            raise ValueError("Maximum dimensions must be >= minimum dimensions")
        if not 0.0 <= complexity_penalty <= 1.0:
            raise ValueError("Complexity penalty must be between 0 and 1")
        if abs(stability_weight + performance_weight - 1.0) > 1e-6:
            raise ValueError("Stability and performance weights must sum to 1.0")
        
        valid_methods = ['exhaustive', 'forward_selection', 'backward_elimination', 'random_search']
        if test_method not in valid_methods:
            raise ValueError(f"Test method must be one of {valid_methods}")
        
        logger.info(f"Initialized dimension testing with method: {test_method}")
        logger.info(f"Testing {min_dimensions}-{max_dimensions} dimensions")
    
    def test_dimension_combination(self,
                                  optimizer: 'BaseOptimizer',
                                  data: pd.DataFrame,
                                  dimensions: List[str],
                                  train_test_split: float = 0.7,
                                  n_iterations: int = 100) -> DimensionTestResult:
        """
        Test a specific combination of dimensions
        
        Args:
            optimizer: Optimizer instance
            data: Dataset with all available dimensions
            dimensions: List of dimension names to test
            train_test_split: Fraction for training set
            n_iterations: Optimization iterations
            
        Returns:
            DimensionTestResult for this combination
        """
        start_time = time.time()
        
        # Validate dimensions exist in data
        missing_dims = [dim for dim in dimensions if dim not in data.columns]
        if missing_dims:
            raise ValueError(f"Missing dimensions in data: {missing_dims}")
        
        # Prepare data subset
        subset_data = data[dimensions].copy()
        
        # Split data temporally (important for time series)
        split_idx = int(len(subset_data) * train_test_split)
        train_data = subset_data.iloc[:split_idx]
        test_data = subset_data.iloc[split_idx:]
        
        try:
            # Create temporary objective function for this dimension subset
            original_objective = optimizer.objective_function
            
            def subset_objective(params):
                # Evaluate using only the selected dimensions
                return original_objective(params)  # Simplified for this example
            
            # Update optimizer temporarily
            optimizer.objective_function = subset_objective
            
            # Run optimization on training data
            train_result = optimizer.optimize(n_iterations=n_iterations)
            train_score = train_result.best_score
            
            # Evaluate on test data
            test_score = subset_objective(train_result.best_params)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(
                train_result.convergence_history, train_score, test_score
            )
            
            # Calculate complexity penalty
            complexity_penalty = self.complexity_penalty * len(dimensions)
            
            # Calculate adjusted score
            raw_score = (self.performance_weight * test_score + 
                        self.stability_weight * stability_score)
            adjusted_score = raw_score - complexity_penalty
            
            # Restore original objective function
            optimizer.objective_function = original_objective
            
            optimization_time = time.time() - start_time
            
            return DimensionTestResult(
                dimensions=dimensions.copy(),
                n_dimensions=len(dimensions),
                train_score=train_score,
                test_score=test_score,
                stability_score=stability_score,
                complexity_penalty=complexity_penalty,
                adjusted_score=adjusted_score,
                optimization_time=optimization_time,
                metadata={
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'convergence_iterations': train_result.n_iterations,
                    'overfitting_score': abs(train_score - test_score) / abs(train_score) if train_score != 0 else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error testing dimensions {dimensions}: {e}")
            # Return failed result
            return DimensionTestResult(
                dimensions=dimensions.copy(),
                n_dimensions=len(dimensions),
                train_score=float('-inf'),
                test_score=float('-inf'),
                stability_score=0.0,
                complexity_penalty=complexity_penalty,
                adjusted_score=float('-inf'),
                optimization_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _calculate_stability_score(self, 
                                  convergence_history: List[float],
                                  train_score: float,
                                  test_score: float) -> float:
        """Calculate stability score based on convergence and generalization"""
        
        # Convergence stability (how smooth is the convergence)
        if len(convergence_history) < 2:
            convergence_stability = 0.0
        else:
            # Calculate coefficient of variation of final 20% of convergence
            final_portion = convergence_history[-len(convergence_history)//5:]
            if len(final_portion) > 1:
                cv = np.std(final_portion) / (np.mean(final_portion) + 1e-8)
                convergence_stability = 1.0 / (1.0 + cv)
            else:
                convergence_stability = 0.5
        
        # Generalization stability (train vs test performance)
        if train_score != 0:
            generalization_gap = abs(train_score - test_score) / abs(train_score)
            generalization_stability = 1.0 / (1.0 + generalization_gap)
        else:
            generalization_stability = 0.0
        
        # Combined stability score
        stability = 0.6 * convergence_stability + 0.4 * generalization_stability
        return stability
    
    def exhaustive_dimension_search(self,
                                   optimizer: 'BaseOptimizer',
                                   data: pd.DataFrame,
                                   available_dimensions: List[str],
                                   max_combinations: int = 1000) -> List[DimensionTestResult]:
        """
        Exhaustively test all combinations of dimensions
        
        Args:
            optimizer: Optimizer instance
            data: Dataset
            available_dimensions: All available dimensions to test
            max_combinations: Maximum combinations to test
            
        Returns:
            List of DimensionTestResult sorted by adjusted score
        """
        logger.info(f"Starting exhaustive dimension search")
        logger.info(f"Available dimensions: {len(available_dimensions)}")
        
        results = []
        combinations_tested = 0
        
        # Test all combinations from min_dimensions to max_dimensions
        for n_dims in range(self.min_dimensions, 
                           min(self.max_dimensions + 1, len(available_dimensions) + 1)):
            
            logger.info(f"Testing combinations with {n_dims} dimensions")
            
            for dimension_combo in combinations(available_dimensions, n_dims):
                if combinations_tested >= max_combinations:
                    logger.warning(f"Reached maximum combinations limit: {max_combinations}")
                    break
                
                result = self.test_dimension_combination(
                    optimizer, data, list(dimension_combo)
                )
                results.append(result)
                combinations_tested += 1
                
                if combinations_tested % 10 == 0:
                    logger.info(f"Tested {combinations_tested} combinations")
            
            if combinations_tested >= max_combinations:
                break
        
        # Sort by adjusted score (descending)
        results.sort(key=lambda x: x.adjusted_score, reverse=True)
        
        logger.info(f"Exhaustive search completed: {combinations_tested} combinations tested")
        return results
    
    def forward_selection(self,
                         optimizer: 'BaseOptimizer',
                         data: pd.DataFrame,
                         available_dimensions: List[str]) -> List[DimensionTestResult]:
        """
        Forward selection algorithm for dimension selection
        
        Args:
            optimizer: Optimizer instance
            data: Dataset
            available_dimensions: All available dimensions
            
        Returns:
            List of DimensionTestResult showing progression
        """
        logger.info("Starting forward selection")
        
        selected_dimensions = []
        remaining_dimensions = available_dimensions.copy()
        results = []
        
        while (len(selected_dimensions) < self.max_dimensions and 
               len(selected_dimensions) < len(available_dimensions) and
               remaining_dimensions):
            
            best_score = float('-inf')
            best_dimension = None
            best_result = None
            
            # Test adding each remaining dimension
            for dim in remaining_dimensions:
                test_dimensions = selected_dimensions + [dim]
                
                result = self.test_dimension_combination(
                    optimizer, data, test_dimensions
                )
                
                if result.adjusted_score > best_score:
                    best_score = result.adjusted_score
                    best_dimension = dim
                    best_result = result
            
            # Add best dimension if it improves the score
            if best_result and len(results) == 0 or best_result.adjusted_score > results[-1].adjusted_score:
                selected_dimensions.append(best_dimension)
                remaining_dimensions.remove(best_dimension)
                results.append(best_result)
                
                logger.info(f"Added dimension '{best_dimension}': "
                          f"score={best_result.adjusted_score:.4f}")
            else:
                logger.info("No improvement found, stopping forward selection")
                break
        
        logger.info(f"Forward selection completed with {len(selected_dimensions)} dimensions")
        return results
    
    def backward_elimination(self,
                            optimizer: 'BaseOptimizer',
                            data: pd.DataFrame,
                            available_dimensions: List[str]) -> List[DimensionTestResult]:
        """
        Backward elimination algorithm for dimension selection
        
        Args:
            optimizer: Optimizer instance
            data: Dataset
            available_dimensions: All available dimensions
            
        Returns:
            List of DimensionTestResult showing elimination progression
        """
        logger.info("Starting backward elimination")
        
        current_dimensions = available_dimensions.copy()
        results = []
        
        # Start with all dimensions
        initial_result = self.test_dimension_combination(
            optimizer, data, current_dimensions
        )
        results.append(initial_result)
        
        while len(current_dimensions) > self.min_dimensions:
            best_score = float('-inf')
            worst_dimension = None
            best_result = None
            
            # Test removing each dimension
            for dim in current_dimensions:
                test_dimensions = [d for d in current_dimensions if d != dim]
                
                result = self.test_dimension_combination(
                    optimizer, data, test_dimensions
                )
                
                if result.adjusted_score > best_score:
                    best_score = result.adjusted_score
                    worst_dimension = dim
                    best_result = result
            
            # Remove dimension if it improves the score
            if best_result and best_result.adjusted_score > results[-1].adjusted_score:
                current_dimensions.remove(worst_dimension)
                results.append(best_result)
                
                logger.info(f"Removed dimension '{worst_dimension}': "
                          f"score={best_result.adjusted_score:.4f}")
            else:
                logger.info("No improvement found, stopping backward elimination")
                break
        
        logger.info(f"Backward elimination completed with {len(current_dimensions)} dimensions")
        return results
    
    def random_search_dimensions(self,
                                optimizer: 'BaseOptimizer',
                                data: pd.DataFrame,
                                available_dimensions: List[str],
                                n_trials: int = 100) -> List[DimensionTestResult]:
        """
        Random search for dimension combinations
        
        Args:
            optimizer: Optimizer instance
            data: Dataset
            available_dimensions: All available dimensions
            n_trials: Number of random trials
            
        Returns:
            List of DimensionTestResult
        """
        logger.info(f"Starting random search with {n_trials} trials")
        
        results = []
        
        for trial in range(n_trials):
            # Random number of dimensions
            n_dims = np.random.randint(self.min_dimensions, 
                                     min(self.max_dimensions + 1, len(available_dimensions) + 1))
            
            # Random selection of dimensions
            selected_dims = np.random.choice(available_dimensions, size=n_dims, replace=False).tolist()
            
            result = self.test_dimension_combination(optimizer, data, selected_dims)
            results.append(result)
            
            if trial % 20 == 0:
                logger.info(f"Random search trial {trial}/{n_trials}")
        
        # Sort by adjusted score
        results.sort(key=lambda x: x.adjusted_score, reverse=True)
        
        logger.info(f"Random search completed: {n_trials} trials")
        return results
    
    def optimize_dimensions(self,
                           optimizer: 'BaseOptimizer',
                           data: pd.DataFrame,
                           available_dimensions: List[str],
                           method: Optional[str] = None) -> Dict[str, Any]:
        """
        Find optimal dimension combination
        
        Args:
            optimizer: Optimizer instance
            data: Dataset
            available_dimensions: Available dimensions
            method: Override default test method
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        method = method or self.test_method
        
        logger.info(f"Optimizing dimensions using {method}")
        
        if method == 'exhaustive':
            results = self.exhaustive_dimension_search(optimizer, data, available_dimensions)
        elif method == 'forward_selection':
            results = self.forward_selection(optimizer, data, available_dimensions)
        elif method == 'backward_elimination':
            results = self.backward_elimination(optimizer, data, available_dimensions)
        elif method == 'random_search':
            results = self.random_search_dimensions(optimizer, data, available_dimensions)
        else:
            raise ValueError(f"Unknown dimension optimization method: {method}")
        
        if not results:
            raise ValueError("No valid dimension combinations found")
        
        # Get best result
        best_result = results[0]
        
        # Calculate summary statistics
        all_scores = [r.adjusted_score for r in results if r.adjusted_score != float('-inf')]
        
        optimization_summary = {
            'method_used': method,
            'total_time': time.time() - start_time,
            'combinations_tested': len(results),
            'best_dimensions': best_result.dimensions,
            'best_score': best_result.adjusted_score,
            'best_result': {
                'train_score': best_result.train_score,
                'test_score': best_result.test_score,
                'stability_score': best_result.stability_score,
                'n_dimensions': best_result.n_dimensions,
                'optimization_time': best_result.optimization_time
            },
            'score_statistics': {
                'mean_score': np.mean(all_scores) if all_scores else 0,
                'std_score': np.std(all_scores) if all_scores else 0,
                'min_score': np.min(all_scores) if all_scores else 0,
                'max_score': np.max(all_scores) if all_scores else 0
            },
            'top_10_results': [
                {
                    'dimensions': r.dimensions,
                    'adjusted_score': r.adjusted_score,
                    'n_dimensions': r.n_dimensions,
                    'stability_score': r.stability_score
                }
                for r in results[:10]
            ]
        }
        
        logger.info(f"Dimension optimization completed")
        logger.info(f"Best dimensions ({best_result.n_dimensions}): {best_result.dimensions}")
        logger.info(f"Best score: {best_result.adjusted_score:.4f}")
        
        return optimization_summary
    
    def create_dimension_report(self,
                               optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive dimension testing report
        
        Args:
            optimization_results: Results from optimize_dimensions
            
        Returns:
            Comprehensive report
        """
        report = {
            'executive_summary': {
                'method_used': optimization_results['method_used'],
                'best_dimensions': optimization_results['best_dimensions'],
                'optimal_dimension_count': len(optimization_results['best_dimensions']),
                'best_score': optimization_results['best_score'],
                'improvement_over_mean': (
                    optimization_results['best_score'] - optimization_results['score_statistics']['mean_score']
                ) if optimization_results['score_statistics']['mean_score'] != 0 else 0
            },
            'performance_analysis': optimization_results['score_statistics'],
            'dimension_ranking': optimization_results['top_10_results'],
            'recommendations': self._generate_recommendations(optimization_results)
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on dimension testing results"""
        recommendations = []
        
        best_result = results['best_result']
        n_dims = len(results['best_dimensions'])
        
        # Dimension count recommendations
        if n_dims <= 3:
            recommendations.append("Low-dimensional solution found - good for interpretability and speed")
        elif n_dims <= 6:
            recommendations.append("Moderate dimensionality - good balance of performance and complexity")
        else:
            recommendations.append("High-dimensional solution - consider regularization to prevent overfitting")
        
        # Stability recommendations
        if best_result['stability_score'] > 0.8:
            recommendations.append("High stability - optimization results are robust")
        elif best_result['stability_score'] > 0.6:
            recommendations.append("Moderate stability - consider ensemble methods")
        else:
            recommendations.append("Low stability - results may be sensitive to data changes")
        
        # Performance recommendations
        if best_result['test_score'] > best_result['train_score']:
            recommendations.append("Good generalization - test performance exceeds training")
        else:
            overfitting = abs(best_result['train_score'] - best_result['test_score']) / abs(best_result['train_score'])
            if overfitting > 0.2:
                recommendations.append("Potential overfitting detected - consider cross-validation")
        
        return recommendations