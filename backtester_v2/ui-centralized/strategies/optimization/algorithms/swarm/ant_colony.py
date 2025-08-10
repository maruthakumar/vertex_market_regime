"""
Ant Colony Optimization Algorithm

Simplified ACO for continuous optimization problems.
"""

import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class AntColonyOptimizer(BaseOptimizer):
    """
    Ant Colony Optimization (ACO) for continuous optimization
    
    Simplified ACO that discretizes the continuous search space
    and applies pheromone updates to guide search.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 n_ants: int = 20,
                 n_levels: int = 10,
                 evaporation_rate: float = 0.1,
                 pheromone_deposit: float = 1.0,
                 **kwargs):
        """
        Initialize Ant Colony optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            n_ants: Number of ants in colony
            n_levels: Number of discrete levels per parameter
            evaporation_rate: Pheromone evaporation rate
            pheromone_deposit: Base pheromone deposit amount
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.n_ants = n_ants
        self.n_levels = n_levels
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        
        # Validate parameters
        if n_ants < 1:
            raise ValueError("Number of ants must be at least 1")
        if n_levels < 2:
            raise ValueError("Number of levels must be at least 2")
        if not 0.0 <= evaporation_rate <= 1.0:
            raise ValueError("Evaporation rate must be between 0.0 and 1.0")
        
        # Discretize parameter space
        self.param_levels = self._create_parameter_levels()
        
        # Initialize pheromone matrix
        self.pheromones = self._initialize_pheromones()
        
        logger.info(f"Initialized ACO with {n_ants} ants, {n_levels} levels per parameter")
    
    def _create_parameter_levels(self) -> Dict[str, List[float]]:
        """Create discrete levels for each parameter"""
        levels = {}
        for param_name, (min_val, max_val) in self.param_space.items():
            levels[param_name] = np.linspace(min_val, max_val, self.n_levels).tolist()
        return levels
    
    def _initialize_pheromones(self) -> Dict[str, List[float]]:
        """Initialize pheromone levels for each parameter level"""
        pheromones = {}
        for param_name in self.param_space.keys():
            # Initialize with equal pheromone levels
            pheromones[param_name] = [1.0] * self.n_levels
        return pheromones
    
    def _select_parameter_level(self, param_name: str) -> Tuple[int, float]:
        """
        Select parameter level based on pheromone probabilities
        
        Args:
            param_name: Name of parameter
            
        Returns:
            Tuple of (level_index, actual_value)
        """
        pheromone_levels = self.pheromones[param_name]
        
        # Convert to probabilities
        total_pheromone = sum(pheromone_levels)
        if total_pheromone == 0:
            probabilities = [1.0 / len(pheromone_levels)] * len(pheromone_levels)
        else:
            probabilities = [p / total_pheromone for p in pheromone_levels]
        
        # Roulette wheel selection
        r = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return i, self.param_levels[param_name][i]
        
        # Fallback to last level
        return len(probabilities) - 1, self.param_levels[param_name][-1]
    
    def _construct_solution(self) -> Dict[str, float]:
        """Construct a solution using pheromone-guided selection"""
        solution = {}
        
        for param_name in self.param_space.keys():
            _, value = self._select_parameter_level(param_name)
            solution[param_name] = value
        
        return solution
    
    def _update_pheromones(self, solutions: List[Dict[str, float]], 
                          fitness_values: List[float]):
        """Update pheromone levels based on solution quality"""
        
        # Evaporation
        for param_name in self.pheromones.keys():
            for i in range(len(self.pheromones[param_name])):
                self.pheromones[param_name][i] *= (1 - self.evaporation_rate)
        
        # Find best solution(s) for reinforcement
        if self.maximize:
            best_indices = [i for i, f in enumerate(fitness_values) 
                          if f == max(fitness_values)]
        else:
            best_indices = [i for i, f in enumerate(fitness_values) 
                          if f == min(fitness_values)]
        
        # Deposit pheromones for best solutions
        for best_idx in best_indices:
            solution = solutions[best_idx]
            fitness = fitness_values[best_idx]
            
            # Calculate deposit amount based on fitness
            deposit_amount = self.pheromone_deposit * abs(fitness) if fitness != 0 else self.pheromone_deposit
            
            for param_name, value in solution.items():
                # Find closest level
                levels = self.param_levels[param_name]
                closest_idx = min(range(len(levels)), 
                                 key=lambda i: abs(levels[i] - value))
                
                # Deposit pheromone
                self.pheromones[param_name][closest_idx] += deposit_amount
    
    def optimize(self, 
                 n_iterations: int = 100,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run Ant Colony Optimization
        
        Args:
            n_iterations: Number of iterations
            callback: Optional callback function
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        logger.info(f"Starting Ant Colony Optimization")
        logger.info(f"Iterations: {n_iterations}, Ants: {self.n_ants}")
        
        # Track statistics
        best_fitness_history = []
        avg_fitness_history = []
        pheromone_diversity_history = []
        
        for iteration in range(n_iterations):
            # Construct solutions with all ants
            solutions = []
            fitness_values = []
            
            for ant in range(self.n_ants):
                solution = self._construct_solution()
                fitness = self.objective_function(solution)
                
                solutions.append(solution)
                fitness_values.append(fitness)
                
                # Update global best
                self._update_best(solution, fitness)
            
            # Calculate statistics
            best_fitness = max(fitness_values) if self.maximize else min(fitness_values)
            avg_fitness = np.mean(fitness_values)
            pheromone_diversity = self._calculate_pheromone_diversity()
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            pheromone_diversity_history.append(pheromone_diversity)
            self.iteration_history.append(best_fitness)
            
            # Call callback if provided
            if callback:
                best_solution = solutions[fitness_values.index(best_fitness)]
                callback(iteration + 1, best_solution, best_fitness)
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best={best_fitness:.6f}, "
                          f"Avg={avg_fitness:.6f}, Diversity={pheromone_diversity:.4f}")
            
            # Update pheromones
            self._update_pheromones(solutions, fitness_values)
            
            # Check convergence
            if self._check_convergence(best_fitness):
                logger.info(f"ACO converged after {iteration + 1} iterations")
                break
        
        # Create result
        result = self._create_result(iteration + 1)
        result.metadata.update({
            'algorithm_params': {
                'n_ants': self.n_ants,
                'n_levels': self.n_levels,
                'evaporation_rate': self.evaporation_rate,
                'pheromone_deposit': self.pheromone_deposit,
                'iterations_completed': iteration + 1
            },
            'colony_stats': {
                'best_fitness_history': best_fitness_history,
                'avg_fitness_history': avg_fitness_history,
                'pheromone_diversity_history': pheromone_diversity_history,
                'final_pheromone_diversity': pheromone_diversity,
                'convergence_iteration': iteration + 1 if self.stagnation_count >= self.max_stagnation else None
            }
        })
        
        logger.info(f"ACO completed. Best fitness: {self.best_score:.6f}")
        return result
    
    def _calculate_pheromone_diversity(self) -> float:
        """Calculate diversity of pheromone distribution"""
        diversities = []
        
        for param_name, pheromone_levels in self.pheromones.items():
            # Calculate coefficient of variation
            mean_pheromone = np.mean(pheromone_levels)
            if mean_pheromone > 0:
                std_pheromone = np.std(pheromone_levels)
                diversity = std_pheromone / mean_pheromone
            else:
                diversity = 0.0
            diversities.append(diversity)
        
        return np.mean(diversities)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Ant Colony Optimization',
            'category': 'swarm',
            'description': 'Pheromone-based optimization inspired by ant foraging',
            'supports_gpu': True,  # Ant evaluation can be parallelized
            'supports_parallel': True,
            'complexity': 'O(i * a * n)',  # i iterations, a ants, n evaluations
            'best_for': [
                'combinatorial optimization',
                'path finding problems',
                'discrete optimization',
                'graph-based problems'
            ],
            'hyperparameters': [
                {
                    'name': 'n_ants',
                    'type': 'int',
                    'range': (5, 100),
                    'default': 20,
                    'description': 'Number of ants in colony'
                },
                {
                    'name': 'n_levels',
                    'type': 'int',
                    'range': (5, 50),
                    'default': 10,
                    'description': 'Discretization levels per parameter'
                },
                {
                    'name': 'evaporation_rate',
                    'type': 'float',
                    'range': (0.0, 1.0),
                    'default': 0.1,
                    'description': 'Pheromone evaporation rate'
                },
                {
                    'name': 'pheromone_deposit',
                    'type': 'float',
                    'range': (0.1, 10.0),
                    'default': 1.0,
                    'description': 'Base pheromone deposit amount'
                }
            ],
            'advantages': [
                'Good for combinatorial problems',
                'Inherent parallelism',
                'Positive feedback mechanism',
                'Can find multiple good solutions'
            ],
            'disadvantages': [
                'Requires discretization for continuous problems',
                'Many parameters to tune',
                'Can converge slowly',
                'May get stuck on suboptimal paths'
            ],
            'notes': [
                'This is a simplified ACO for continuous optimization',
                'Original ACO is designed for discrete/combinatorial problems',
                'Discretization may reduce solution quality'
            ],
            'references': [
                'Dorigo, M. & Gambardella, L. M. (1997). Ant colony system',
                'Socha, K. & Dorigo, M. (2008). Ant colony optimization for continuous domains',
                'Dorigo, M. et al. (2006). Ant colony optimization'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'AntColonyOptimizer':
        """Create optimizer from configuration dictionary"""
        aco_config = config.get('ant_colony', {})
        
        return cls(
            n_ants=aco_config.get('n_ants', 20),
            n_levels=aco_config.get('n_levels', 10),
            evaporation_rate=aco_config.get('evaporation_rate', 0.1),
            pheromone_deposit=aco_config.get('pheromone_deposit', 1.0),
            **kwargs
        )