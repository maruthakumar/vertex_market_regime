"""
Differential Evolution Optimization Algorithm

Evolutionary algorithm that uses vector differences for mutation.
"""

import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class DifferentialEvolutionOptimizer(BaseOptimizer):
    """
    Differential Evolution (DE) optimization algorithm
    
    Population-based evolutionary algorithm that uses vector differences
    for mutation and binary crossover for recombination.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 population_size: int = None,
                 differential_weight: float = 0.8,
                 crossover_rate: float = 0.9,
                 strategy: str = 'rand/1/bin',
                 **kwargs):
        """
        Initialize Differential Evolution optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            population_size: Population size (default: 10 * dimensions)
            differential_weight: Differential weight F (0.0 to 2.0)
            crossover_rate: Crossover probability CR (0.0 to 1.0)
            strategy: DE strategy ('rand/1/bin', 'best/1/bin', 'current-to-best/1/bin')
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        # Set default population size based on problem dimension
        if population_size is None:
            population_size = max(10, 10 * len(param_space))
        
        self.population_size = population_size
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate
        self.strategy = strategy
        
        # Validate parameters
        if population_size < 4:
            raise ValueError("Population size must be at least 4 for DE")
        if not 0.0 <= differential_weight <= 2.0:
            raise ValueError("Differential weight must be between 0.0 and 2.0")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        
        valid_strategies = ['rand/1/bin', 'best/1/bin', 'current-to-best/1/bin']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        # Population storage
        self.population = []
        self.fitness_values = []
        self.generation = 0
        
        logger.info(f"Initialized Differential Evolution with strategy={strategy}")
        logger.info(f"Population size: {population_size}, F: {differential_weight}, CR: {crossover_rate}")
    
    def _initialize_population(self) -> Tuple[List[Dict[str, float]], List[float]]:
        """Initialize random population"""
        population = []
        fitness_values = []
        
        for _ in range(self.population_size):
            individual = self._random_params()
            population.append(individual)
            
            fitness = self.objective_function(individual)
            fitness_values.append(fitness)
            
            # Update global best
            self._update_best(individual, fitness)
        
        return population, fitness_values
    
    def _get_mutation_vector(self, target_idx: int, population: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Generate mutation vector based on DE strategy
        
        Args:
            target_idx: Index of target individual
            population: Current population
            
        Returns:
            Mutation vector
        """
        param_names = list(self.param_space.keys())
        
        if self.strategy == 'rand/1/bin':
            # Select three random individuals (different from target)
            candidates = [i for i in range(len(population)) if i != target_idx]
            r1, r2, r3 = random.sample(candidates, 3)
            
            mutant = {}
            for param_name in param_names:
                mutant[param_name] = (population[r1][param_name] + 
                                    self.differential_weight * 
                                    (population[r2][param_name] - population[r3][param_name]))
                
        elif self.strategy == 'best/1/bin':
            # Use best individual + difference of two random individuals
            best_idx = np.argmax(self.fitness_values) if self.maximize else np.argmin(self.fitness_values)
            candidates = [i for i in range(len(population)) if i != target_idx and i != best_idx]
            r1, r2 = random.sample(candidates, 2)
            
            mutant = {}
            for param_name in param_names:
                mutant[param_name] = (population[best_idx][param_name] + 
                                    self.differential_weight * 
                                    (population[r1][param_name] - population[r2][param_name]))
                
        elif self.strategy == 'current-to-best/1/bin':
            # Combination of current, best, and random individuals
            best_idx = np.argmax(self.fitness_values) if self.maximize else np.argmin(self.fitness_values)
            candidates = [i for i in range(len(population)) if i != target_idx and i != best_idx]
            r1, r2 = random.sample(candidates, 2)
            
            mutant = {}
            for param_name in param_names:
                mutant[param_name] = (population[target_idx][param_name] + 
                                    self.differential_weight * 
                                    (population[best_idx][param_name] - population[target_idx][param_name]) +
                                    self.differential_weight * 
                                    (population[r1][param_name] - population[r2][param_name]))
        
        # Clip to bounds
        for param_name in param_names:
            min_val, max_val = self.param_space[param_name]
            mutant[param_name] = np.clip(mutant[param_name], min_val, max_val)
        
        return mutant
    
    def _crossover(self, target: Dict[str, float], mutant: Dict[str, float]) -> Dict[str, float]:
        """
        Binary crossover between target and mutant vectors
        
        Args:
            target: Target individual
            mutant: Mutant vector
            
        Returns:
            Trial vector
        """
        param_names = list(self.param_space.keys())
        trial = {}
        
        # Ensure at least one parameter comes from mutant
        forced_param = random.choice(param_names)
        
        for param_name in param_names:
            if param_name == forced_param or random.random() < self.crossover_rate:
                trial[param_name] = mutant[param_name]
            else:
                trial[param_name] = target[param_name]
        
        return trial
    
    def _evolve_generation(self, population: List[Dict[str, float]], 
                          fitness_values: List[float]) -> Tuple[List[Dict[str, float]], List[float]]:
        """Evolve one generation using DE operators"""
        new_population = []
        new_fitness_values = []
        
        for i in range(len(population)):
            # Get target individual
            target = population[i]
            target_fitness = fitness_values[i]
            
            # Generate mutant vector
            mutant = self._get_mutation_vector(i, population)
            
            # Crossover to create trial vector
            trial = self._crossover(target, mutant)
            
            # Evaluate trial vector
            trial_fitness = self.objective_function(trial)
            
            # Selection: keep better individual
            is_better = (trial_fitness > target_fitness if self.maximize 
                        else trial_fitness < target_fitness)
            
            if is_better:
                new_population.append(trial)
                new_fitness_values.append(trial_fitness)
                # Update global best
                self._update_best(trial, trial_fitness)
            else:
                new_population.append(target)
                new_fitness_values.append(target_fitness)
        
        return new_population, new_fitness_values
    
    def optimize(self, 
                 n_iterations: int = 300,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run Differential Evolution optimization
        
        Args:
            n_iterations: Number of generations
            callback: Optional callback function
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        generations = n_iterations
        
        logger.info(f"Starting Differential Evolution optimization")
        logger.info(f"Generations: {generations}, Population: {self.population_size}")
        
        # Initialize population
        self.population, self.fitness_values = self._initialize_population()
        
        # Track statistics
        best_fitness_history = []
        avg_fitness_history = []
        
        # Evolution loop
        for generation in range(generations):
            self.generation = generation
            
            # Get statistics
            best_fitness = max(self.fitness_values) if self.maximize else min(self.fitness_values)
            avg_fitness = np.mean(self.fitness_values)
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            self.iteration_history.append(best_fitness)
            
            # Find best individual
            best_idx = np.argmax(self.fitness_values) if self.maximize else np.argmin(self.fitness_values)
            best_individual = self.population[best_idx]
            
            # Call callback if provided
            if callback:
                callback(generation + 1, best_individual, best_fitness)
            
            # Log progress
            if generation % 20 == 0:
                logger.info(f"Generation {generation}: Best={best_fitness:.6f}, "
                          f"Avg={avg_fitness:.6f}")
            
            # Check convergence
            if self._check_convergence(best_fitness):
                logger.info(f"Differential Evolution converged after {generation + 1} generations")
                break
            
            # Evolve to next generation
            if generation < generations - 1:
                self.population, self.fitness_values = self._evolve_generation(
                    self.population, self.fitness_values)
        
        # Create result
        result = self._create_result(generation + 1)
        result.metadata.update({
            'algorithm_params': {
                'population_size': self.population_size,
                'differential_weight': self.differential_weight,
                'crossover_rate': self.crossover_rate,
                'strategy': self.strategy,
                'generations_completed': generation + 1
            },
            'evolution_stats': {
                'best_fitness_history': best_fitness_history,
                'avg_fitness_history': avg_fitness_history,
                'convergence_generation': generation + 1 if self.stagnation_count >= self.max_stagnation else None
            }
        })
        
        logger.info(f"Differential Evolution completed. Best fitness: {self.best_score:.6f}")
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Differential Evolution',
            'category': 'evolutionary',
            'description': 'Vector difference-based evolutionary optimization',
            'supports_gpu': True,  # Population evaluation can be parallelized
            'supports_parallel': True,
            'complexity': 'O(g * p * n)',  # g generations, p population, n evaluations
            'best_for': [
                'continuous optimization',
                'global optimization',
                'multimodal problems',
                'real-valued parameters'
            ],
            'hyperparameters': [
                {
                    'name': 'population_size',
                    'type': 'int',
                    'range': (10, 1000),
                    'default': '10 * dimensions',
                    'description': 'Size of the population'
                },
                {
                    'name': 'differential_weight',
                    'type': 'float',
                    'range': (0.0, 2.0),
                    'default': 0.8,
                    'description': 'Differential weight F for mutation'
                },
                {
                    'name': 'crossover_rate',
                    'type': 'float',
                    'range': (0.0, 1.0),
                    'default': 0.9,
                    'description': 'Crossover probability CR'
                },
                {
                    'name': 'strategy',
                    'type': 'categorical',
                    'choices': ['rand/1/bin', 'best/1/bin', 'current-to-best/1/bin'],
                    'default': 'rand/1/bin',
                    'description': 'DE mutation strategy'
                }
            ],
            'advantages': [
                'Excellent for continuous optimization',
                'Self-adapting behavior',
                'Few control parameters',
                'Good convergence properties'
            ],
            'disadvantages': [
                'Sensitive to parameter settings',
                'Can converge prematurely',
                'Not well-suited for discrete problems',
                'Population size affects performance'
            ],
            'references': [
                'Storn, R. & Price, K. (1997). Differential Evolution',
                'Price, K. et al. (2005). Differential Evolution: A Practical Approach',
                'Das, S. & Suganthan, P. N. (2011). Differential Evolution: A Survey'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'DifferentialEvolutionOptimizer':
        """Create optimizer from configuration dictionary"""
        de_config = config.get('differential_evolution', {})
        
        return cls(
            population_size=de_config.get('population_size'),
            differential_weight=de_config.get('differential_weight', 0.8),
            crossover_rate=de_config.get('crossover_rate', 0.9),
            strategy=de_config.get('strategy', 'rand/1/bin'),
            **kwargs
        )