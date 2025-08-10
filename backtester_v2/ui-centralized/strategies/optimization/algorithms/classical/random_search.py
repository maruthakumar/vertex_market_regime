"""
Random Search Optimization Algorithm

Pure random sampling approach that explores the parameter space
by evaluating randomly sampled points.
"""

import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class RandomSearchOptimizer(BaseOptimizer):
    """
    Random Search optimization algorithm
    
    Random search explores the parameter space by evaluating randomly 
    sampled points. Despite its simplicity, it's often surprisingly 
    effective and serves as a strong baseline.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 sampling_method: str = 'uniform',
                 **kwargs):
        """
        Initialize Random Search optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            sampling_method: Sampling method ('uniform', 'latin_hypercube')
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.sampling_method = sampling_method
        
        if sampling_method not in ['uniform', 'latin_hypercube']:
            raise ValueError("Sampling method must be 'uniform' or 'latin_hypercube'")
        
        logger.info(f"Initialized Random Search with {sampling_method} sampling")
    
    def _generate_random_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """
        Generate random parameter samples
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        samples = []
        
        if self.sampling_method == 'uniform':
            # Simple uniform random sampling
            for _ in range(n_samples):
                sample = {}
                for param_name, (min_val, max_val) in self.param_space.items():
                    sample[param_name] = random.uniform(min_val, max_val)
                samples.append(sample)
                
        elif self.sampling_method == 'latin_hypercube':
            # Latin hypercube sampling for better coverage
            try:
                from scipy.stats import qmc
                
                param_names = list(self.param_space.keys())
                n_dims = len(param_names)
                
                # Generate LHS samples
                sampler = qmc.LatinHypercube(d=n_dims, seed=self.random_seed)
                unit_samples = sampler.random(n_samples)
                
                for i in range(n_samples):
                    sample = {}
                    for j, param_name in enumerate(param_names):
                        min_val, max_val = self.param_space[param_name]
                        unit_val = unit_samples[i, j]
                        sample[param_name] = min_val + unit_val * (max_val - min_val)
                    samples.append(sample)
                    
            except ImportError:
                logger.warning("scipy not available, falling back to uniform sampling")
                return self._generate_random_samples(n_samples)  # Recursive call with uniform
        
        return samples
    
    def optimize(self, 
                 n_iterations: int = 1000,
                 callback: Optional[callable] = None,
                 batch_size: Optional[int] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run random search optimization
        
        Args:
            n_iterations: Number of random samples to evaluate
            callback: Optional callback function called each iteration
            batch_size: Number of samples to generate in each batch
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        logger.info(f"Starting Random Search optimization with {n_iterations} samples")
        logger.info(f"Sampling method: {self.sampling_method}")
        
        # Set default batch size
        if batch_size is None:
            batch_size = min(100, n_iterations)
        
        evaluated_samples = 0
        batch_count = 0
        
        while evaluated_samples < n_iterations:
            # Calculate samples for this batch
            remaining_samples = n_iterations - evaluated_samples
            current_batch_size = min(batch_size, remaining_samples)
            
            # Generate batch of random samples
            samples = self._generate_random_samples(current_batch_size)
            
            batch_count += 1
            logger.debug(f"Evaluating batch {batch_count} ({current_batch_size} samples)")
            
            # Evaluate each sample in batch
            for i, sample in enumerate(samples):
                score = self.objective_function(sample)
                
                # Update best if improved
                self._update_best(sample, score)
                
                # Track progress
                self.iteration_history.append(score)
                evaluated_samples += 1
                
                # Call callback if provided
                if callback:
                    callback(evaluated_samples, sample, score)
                
                # Check for convergence
                if self._check_convergence(score):
                    logger.info(f"Random search converged after {evaluated_samples} evaluations")
                    break
                
                # Log progress
                if evaluated_samples % 100 == 0:
                    logger.info(f"Evaluated {evaluated_samples}/{n_iterations} samples. "
                              f"Best score: {self.best_score:.6f}")
            
            # Early termination if converged
            if self.stagnation_count >= self.max_stagnation:
                break
        
        # Create result
        result = self._create_result(evaluated_samples)
        result.metadata.update({
            'algorithm_params': {
                'sampling_method': self.sampling_method,
                'batch_size': batch_size,
                'total_batches': batch_count
            },
            'samples_evaluated': evaluated_samples,
            'best_score_iteration': np.argmax(self.iteration_history) + 1 if self.maximize 
                                  else np.argmin(self.iteration_history) + 1
        })
        
        logger.info(f"Random Search completed. Best score: {self.best_score:.6f}")
        logger.info(f"Evaluated {evaluated_samples} samples in {batch_count} batches")
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Random Search',
            'category': 'classical',
            'description': 'Pure random sampling of parameter space',
            'supports_gpu': True,  # Can benefit from parallel evaluation
            'supports_parallel': True,
            'complexity': 'O(n)',  # n evaluations
            'best_for': [
                'baseline comparison',
                'high-dimensional spaces',
                'parallel evaluation',
                'exploration'
            ],
            'hyperparameters': [
                {
                    'name': 'sampling_method',
                    'type': 'categorical',
                    'choices': ['uniform', 'latin_hypercube'],
                    'default': 'uniform',
                    'description': 'Method for generating random samples'
                },
                {
                    'name': 'batch_size',
                    'type': 'int',
                    'range': (1, 1000),
                    'default': 100,
                    'description': 'Number of samples per batch'
                }
            ],
            'advantages': [
                'Simple to implement and understand',
                'No assumptions about objective function',
                'Highly parallelizable',
                'Good baseline performance'
            ],
            'disadvantages': [
                'No learning from previous samples',
                'Can be inefficient for low-dimensional problems',
                'No convergence guarantees'
            ],
            'references': [
                'Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter optimization',
                'Rastrigin, L.A. (1963). The convergence of the random search method'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'RandomSearchOptimizer':
        """Create optimizer from configuration dictionary"""
        random_config = config.get('random_search', {})
        
        return cls(
            sampling_method=random_config.get('sampling_method', 'uniform'),
            **kwargs
        )