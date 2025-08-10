"""
Genetic Algorithm Optimization

Evolutionary optimization using selection, crossover, and mutation operators.
Migrated from enhanced-market-regime-optimizer and adapted to BaseOptimizer.
"""

import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class Individual:
    """Represents an individual in the genetic algorithm population"""
    
    def __init__(self, params: Dict[str, float], fitness: Optional[float] = None):
        self.params = params
        self.fitness = fitness
        self.age = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness}, params={self.params})"

class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimization
    
    Evolutionary algorithm that maintains a population of candidate solutions
    and evolves them using selection, crossover, and mutation operators.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 population_size: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 tournament_size: int = 3,
                 elitism: bool = True,
                 **kwargs):
        """
        Initialize Genetic Algorithm optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            population_size: Size of the population
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament selection
            elitism: Whether to preserve best individual
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        
        # Validate parameters
        if population_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        if tournament_size < 1:
            raise ValueError("Tournament size must be at least 1")
        
        self.population = []
        self.generation = 0
        
        logger.info(f"Initialized Genetic Algorithm with population_size={population_size}")
    
    def _create_individual(self) -> Individual:
        """Create a random individual"""
        params = self._random_params()
        return Individual(params)
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """Evaluate fitness of an individual"""
        if individual.fitness is None:
            individual.fitness = self.objective_function(individual.params)
        return individual.fitness
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = self._create_individual()
            self._evaluate_individual(individual)
            population.append(individual)
        return population
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        if self.maximize:
            return max(tournament, key=lambda ind: ind.fitness)
        else:
            return min(tournament, key=lambda ind: ind.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover between two parents
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return (Individual(parent1.params.copy()), 
                   Individual(parent2.params.copy()))
        
        # Perform uniform crossover
        child1_params = {}
        child2_params = {}
        
        for param_name in self.param_space.keys():
            if random.random() < 0.5:
                child1_params[param_name] = parent1.params[param_name]
                child2_params[param_name] = parent2.params[param_name]
            else:
                child1_params[param_name] = parent2.params[param_name]
                child2_params[param_name] = parent1.params[param_name]
        
        return Individual(child1_params), Individual(child2_params)
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual using Gaussian mutation
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated_params = individual.params.copy()
        
        for param_name, (min_val, max_val) in self.param_space.items():
            if random.random() < self.mutation_rate:
                # Gaussian mutation with 10% of parameter range as std
                param_range = max_val - min_val
                mutation_std = 0.1 * param_range
                
                current_value = mutated_params[param_name]
                new_value = current_value + random.gauss(0, mutation_std)
                
                # Clip to bounds
                mutated_params[param_name] = np.clip(new_value, min_val, max_val)
        
        return Individual(mutated_params)
    
    def _evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Evolve one generation"""
        new_population = []
        
        # Elitism - preserve best individual
        if self.elitism:
            best_individual = max(population, key=lambda ind: ind.fitness if self.maximize else -ind.fitness)
            new_population.append(Individual(best_individual.params.copy(), best_individual.fitness))
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            # Add to new population
            new_population.extend([child1, child2])
        
        # Truncate to population size if needed
        new_population = new_population[:self.population_size]
        
        # Evaluate new individuals
        for individual in new_population:
            if individual.fitness is None:
                self._evaluate_individual(individual)
        
        return new_population
    
    def optimize(self, 
                 n_iterations: int = 100,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run genetic algorithm optimization
        
        Args:
            n_iterations: Number of generations
            callback: Optional callback function
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        # Calculate generations
        generations = n_iterations
        
        logger.info(f"Starting Genetic Algorithm optimization")
        logger.info(f"Generations: {generations}, Population size: {self.population_size}")
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Track statistics
        best_fitness_history = []
        avg_fitness_history = []
        
        # Evolution loop
        for generation in range(generations):
            self.generation = generation
            
            # Get population statistics
            fitnesses = [ind.fitness for ind in self.population]
            best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
            avg_fitness = np.mean(fitnesses)
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            # Find best individual in current generation
            best_individual = max(self.population, 
                                key=lambda ind: ind.fitness if self.maximize else -ind.fitness)
            
            # Update global best
            self._update_best(best_individual.params, best_individual.fitness)
            self.iteration_history.append(best_fitness)
            
            # Call callback if provided
            if callback:
                callback(generation + 1, best_individual.params, best_individual.fitness)
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best={best_fitness:.6f}, "
                          f"Avg={avg_fitness:.6f}")
            
            # Check convergence
            if self._check_convergence(best_fitness):
                logger.info(f"Genetic Algorithm converged after {generation + 1} generations")
                break
            
            # Evolve to next generation
            if generation < generations - 1:  # Don't evolve on last generation
                self.population = self._evolve_generation(self.population)
        
        # Create result
        result = self._create_result(generation + 1)
        result.metadata.update({
            'algorithm_params': {
                'population_size': self.population_size,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'tournament_size': self.tournament_size,
                'elitism': self.elitism,
                'generations_completed': generation + 1
            },
            'evolution_stats': {
                'best_fitness_history': best_fitness_history,
                'avg_fitness_history': avg_fitness_history,
                'final_population_diversity': self._calculate_diversity(),
                'convergence_generation': generation + 1 if self.stagnation_count >= self.max_stagnation else None
            }
        })
        
        logger.info(f"Genetic Algorithm completed. Best fitness: {self.best_score:.6f}")
        return result
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity as average pairwise distance"""
        if len(self.population) < 2:
            return 0.0
        
        distances = []
        param_names = list(self.param_space.keys())
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Calculate normalized Euclidean distance
                dist = 0.0
                for param_name in param_names:
                    min_val, max_val = self.param_space[param_name]
                    param_range = max_val - min_val
                    
                    val1 = self.population[i].params[param_name]
                    val2 = self.population[j].params[param_name]
                    
                    normalized_dist = abs(val1 - val2) / param_range
                    dist += normalized_dist ** 2
                
                distances.append(np.sqrt(dist))
        
        return np.mean(distances) if distances else 0.0
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Genetic Algorithm',
            'category': 'evolutionary',
            'description': 'Population-based evolutionary optimization',
            'supports_gpu': True,  # Population evaluation can be parallelized
            'supports_parallel': True,
            'complexity': 'O(g * p * n)',  # g generations, p population, n evaluations
            'best_for': [
                'global optimization',
                'multimodal problems',
                'discrete and continuous spaces',
                'population-based search'
            ],
            'hyperparameters': [
                {
                    'name': 'population_size',
                    'type': 'int',
                    'range': (10, 1000),
                    'default': 50,
                    'description': 'Size of the population'
                },
                {
                    'name': 'crossover_rate',
                    'type': 'float',
                    'range': (0.0, 1.0),
                    'default': 0.8,
                    'description': 'Probability of crossover'
                },
                {
                    'name': 'mutation_rate',
                    'type': 'float',
                    'range': (0.0, 1.0),
                    'default': 0.2,
                    'description': 'Probability of mutation'
                },
                {
                    'name': 'tournament_size',
                    'type': 'int',
                    'range': (1, 10),
                    'default': 3,
                    'description': 'Size of tournament selection'
                }
            ],
            'advantages': [
                'Global optimization capability',
                'Handles multimodal functions well',
                'Population provides multiple solutions',
                'Robust to noise'
            ],
            'disadvantages': [
                'Can be slow to converge',
                'Many hyperparameters to tune',
                'May not find exact optimum',
                'Requires many function evaluations'
            ],
            'references': [
                'Holland, J. H. (1992). Adaptation in Natural and Artificial Systems',
                'Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning',
                'Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'GeneticAlgorithmOptimizer':
        """Create optimizer from configuration dictionary"""
        ga_config = config.get('genetic_algorithm', {})
        
        return cls(
            population_size=ga_config.get('population_size', 50),
            crossover_rate=ga_config.get('crossover_prob', 0.8),
            mutation_rate=ga_config.get('mutation_prob', 0.2),
            tournament_size=ga_config.get('tournament_size', 3),
            elitism=ga_config.get('elitism', True),
            **kwargs
        )