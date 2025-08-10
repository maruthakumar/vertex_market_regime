"""Optimization algorithms for strategy optimization"""

# Classical algorithms
from .classical.hill_climbing import HillClimbingOptimizer
from .classical.random_search import RandomSearchOptimizer
from .classical.grid_search import GridSearchOptimizer
from .classical.bayesian import BayesianOptimizer

# Evolutionary algorithms
from .evolutionary.genetic_algorithm import GeneticAlgorithmOptimizer
from .evolutionary.differential_evolution import DifferentialEvolutionOptimizer

# Swarm algorithms
from .swarm.particle_swarm import ParticleSwarmOptimizer
from .swarm.ant_colony import AntColonyOptimizer

# Physics-inspired algorithms
from .physics_inspired.simulated_annealing import SimulatedAnnealingOptimizer

# Quantum algorithms (if available)
try:
    from .quantum.qaoa import QAOAOptimizer
    from .quantum.vqe import VQEOptimizer
    from .quantum.quantum_annealing import QuantumAnnealingOptimizer
    from .quantum.quantum_walk import QuantumWalkOptimizer
    from .quantum.hybrid_classical import HybridClassicalQuantumOptimizer
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# Algorithm registry for auto-discovery
CLASSICAL_ALGORITHMS = [
    HillClimbingOptimizer,
    RandomSearchOptimizer, 
    GridSearchOptimizer,
    BayesianOptimizer
]

EVOLUTIONARY_ALGORITHMS = [
    GeneticAlgorithmOptimizer,
    DifferentialEvolutionOptimizer
]

SWARM_ALGORITHMS = [
    ParticleSwarmOptimizer,
    AntColonyOptimizer
]

PHYSICS_ALGORITHMS = [
    SimulatedAnnealingOptimizer
]

QUANTUM_ALGORITHMS = []
if QUANTUM_AVAILABLE:
    QUANTUM_ALGORITHMS = [
        QAOAOptimizer,
        VQEOptimizer,
        QuantumAnnealingOptimizer,
        QuantumWalkOptimizer,
        HybridClassicalQuantumOptimizer
    ]

ALL_ALGORITHMS = (CLASSICAL_ALGORITHMS + 
                 EVOLUTIONARY_ALGORITHMS + 
                 SWARM_ALGORITHMS + 
                 PHYSICS_ALGORITHMS + 
                 QUANTUM_ALGORITHMS)

__all__ = [algo.__name__ for algo in ALL_ALGORITHMS] + [
    "CLASSICAL_ALGORITHMS",
    "EVOLUTIONARY_ALGORITHMS", 
    "SWARM_ALGORITHMS",
    "PHYSICS_ALGORITHMS",
    "QUANTUM_ALGORITHMS",
    "ALL_ALGORITHMS",
    "QUANTUM_AVAILABLE"
]