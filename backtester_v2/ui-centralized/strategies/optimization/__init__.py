"""
Strategy Optimization Module

This module provides a comprehensive optimization framework with:
- 16 optimization algorithms (classical, evolutionary, swarm, quantum)
- Robust optimization framework with cross-validation and sensitivity analysis
- GPU acceleration via HeavyDB and CuPy
- Market regime integration
- Strategy inversion engine
- Auto-discovery algorithm registry
"""

from .engines.optimization_engine import OptimizationEngine
from .base.base_optimizer import BaseOptimizer
from .engines.algorithm_registry import AlgorithmRegistry

__version__ = "1.0.0"
__author__ = "Strategy Optimization Team"

__all__ = [
    "OptimizationEngine",
    "BaseOptimizer", 
    "AlgorithmRegistry"
]