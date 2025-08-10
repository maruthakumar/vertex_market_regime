"""
Bayesian Optimization Algorithm

Uses Gaussian Process regression to model the objective function
and acquisition functions to guide search.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using Gaussian Process regression
    
    Models the objective function with a Gaussian Process and uses
    acquisition functions to balance exploration and exploitation.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 acquisition_function: str = 'expected_improvement',
                 n_initial_points: int = 5,
                 **kwargs):
        """
        Initialize Bayesian optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            acquisition_function: Acquisition function ('expected_improvement', 'upper_confidence_bound')
            n_initial_points: Number of initial random points
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        
        # Try to import scikit-optimize for Bayesian optimization
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
            self.skopt_available = True
            self.gp_minimize = gp_minimize
            self.Real = Real
            self.use_named_args = use_named_args
        except ImportError:
            self.skopt_available = False
            logger.warning("scikit-optimize not available. Bayesian optimization will use simplified version.")
        
        # Store evaluation history for GP
        self.X_samples = []
        self.y_samples = []
        
        logger.info(f"Initialized Bayesian Optimization with {acquisition_function} acquisition")
    
    def _simple_bayesian_optimization(self, n_iterations: int, callback: Optional[Callable]) -> OptimizationResult:
        """
        Simplified Bayesian optimization without external dependencies
        Falls back to random search with some heuristics
        """
        logger.warning("Using simplified Bayesian optimization (random search with exploitation)")
        
        # Initial random sampling
        for i in range(self.n_initial_points):
            params = self._random_params()
            score = self.objective_function(params)
            
            self.X_samples.append(list(params.values()))
            self.y_samples.append(score)
            
            self._update_best(params, score)
            self.iteration_history.append(score)
            
            if callback:
                callback(i + 1, params, score)
        
        # Continue with biased sampling around best points
        remaining_iterations = n_iterations - self.n_initial_points
        
        for i in range(remaining_iterations):
            # 80% exploitation around best points, 20% exploration
            if np.random.random() < 0.8 and self.best_params:
                # Sample around best parameters
                params = self._sample_around_best()
            else:
                # Random exploration
                params = self._random_params()
            
            score = self.objective_function(params)
            
            self.X_samples.append(list(params.values()))
            self.y_samples.append(score)
            
            self._update_best(params, score)
            self.iteration_history.append(score)
            
            if callback:
                callback(self.n_initial_points + i + 1, params, score)
            
            # Check convergence
            if self._check_convergence(score):
                break
        
        return self._create_result(len(self.iteration_history))
    
    def _sample_around_best(self) -> Dict[str, float]:
        """Sample parameters around current best with Gaussian noise"""
        params = {}
        noise_scale = 0.1  # 10% of parameter range
        
        for param_name, (min_val, max_val) in self.param_space.items():
            # Get current best value
            current_best = self.best_params[param_name]
            
            # Add Gaussian noise
            param_range = max_val - min_val
            noise = np.random.normal(0, noise_scale * param_range)
            new_value = current_best + noise
            
            # Clip to bounds
            params[param_name] = np.clip(new_value, min_val, max_val)
        
        return params
    
    def optimize(self, 
                 n_iterations: int = 100,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run Bayesian optimization
        
        Args:
            n_iterations: Number of iterations
            callback: Optional callback function
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        logger.info(f"Starting Bayesian optimization with {n_iterations} iterations")
        
        if not self.skopt_available:
            return self._simple_bayesian_optimization(n_iterations, callback)
        
        # Use scikit-optimize for full Bayesian optimization
        try:
            # Define search space
            dimensions = []
            param_names = []
            
            for param_name, (min_val, max_val) in self.param_space.items():
                dimensions.append(self.Real(min_val, max_val, name=param_name))
                param_names.append(param_name)
            
            # Define objective function wrapper
            @self.use_named_args(dimensions)
            def objective_wrapper(**params):
                score = self.objective_function(params)
                
                # Store for tracking
                self.X_samples.append(list(params.values()))
                self.y_samples.append(score)
                self._update_best(params, score)
                self.iteration_history.append(score)
                
                if callback:
                    callback(len(self.iteration_history), params, score)
                
                # scikit-optimize minimizes, so negate if maximizing
                return -score if self.maximize else score
            
            # Run optimization
            result = self.gp_minimize(
                func=objective_wrapper,
                dimensions=dimensions,
                n_calls=n_iterations,
                n_initial_points=self.n_initial_points,
                acq_func=self.acquisition_function.replace('expected_improvement', 'EI')
                                                   .replace('upper_confidence_bound', 'UCB'),
                random_state=self.random_seed
            )
            
            # Create our result format
            opt_result = self._create_result(n_iterations)
            opt_result.metadata.update({
                'algorithm_params': {
                    'acquisition_function': self.acquisition_function,
                    'n_initial_points': self.n_initial_points,
                    'gp_converged': True
                },
                'skopt_result': {
                    'fun': result.fun,
                    'x': result.x,
                    'n_calls': result.n_calls
                }
            })
            
            return opt_result
            
        except Exception as e:
            logger.error(f"Error in scikit-optimize Bayesian optimization: {e}")
            logger.info("Falling back to simplified Bayesian optimization")
            return self._simple_bayesian_optimization(n_iterations, callback)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Bayesian Optimization',
            'category': 'classical',
            'description': 'Gaussian Process-based optimization with acquisition functions',
            'supports_gpu': False,
            'supports_parallel': False,  # Sequential by nature
            'complexity': 'O(n³)',  # Due to GP inference
            'best_for': [
                'expensive function evaluations',
                'continuous optimization',
                'few iterations available',
                'smooth objective functions'
            ],
            'hyperparameters': [
                {
                    'name': 'acquisition_function',
                    'type': 'categorical',
                    'choices': ['expected_improvement', 'upper_confidence_bound'],
                    'default': 'expected_improvement',
                    'description': 'Acquisition function for next point selection'
                },
                {
                    'name': 'n_initial_points',
                    'type': 'int',
                    'range': (2, 50),
                    'default': 5,
                    'description': 'Number of initial random points'
                }
            ],
            'advantages': [
                'Sample efficient',
                'Good for expensive evaluations',
                'Principled uncertainty quantification',
                'Balances exploration and exploitation'
            ],
            'disadvantages': [
                'Sequential optimization',
                'Assumes smooth objective function',
                'Computational overhead for GP inference',
                'Limited to continuous parameters'
            ],
            'requirements': [
                'scikit-optimize (optional, for full functionality)',
                'Continuous parameter space recommended'
            ],
            'references': [
                'Mockus, J. (1989). Bayesian Approach to Global Optimization',
                'Brochu, E. et al. (2010). A Tutorial on Bayesian Optimization',
                'Snoek, J. et al. (2012). Practical Bayesian Optimization'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'BayesianOptimizer':
        """Create optimizer from configuration dictionary"""
        bayes_config = config.get('bayesian', {})
        
        return cls(
            acquisition_function=bayes_config.get('acquisition_function', 'expected_improvement'),
            n_initial_points=bayes_config.get('n_initial_points', 5),
            **kwargs
        )