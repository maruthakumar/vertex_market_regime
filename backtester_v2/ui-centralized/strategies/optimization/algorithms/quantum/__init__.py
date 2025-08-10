"""Quantum-inspired optimization algorithms"""

try:
    from .qaoa import QAOAOptimizer
    from .vqe import VQEOptimizer
    from .quantum_annealing import QuantumAnnealingOptimizer
    from .quantum_walk import QuantumWalkOptimizer
    from .hybrid_classical import HybridClassicalQuantumOptimizer
    
    __all__ = [
        "QAOAOptimizer",
        "VQEOptimizer",
        "QuantumAnnealingOptimizer",
        "QuantumWalkOptimizer",
        "HybridClassicalQuantumOptimizer"
    ]
    
except ImportError as e:
    # Quantum libraries not available
    __all__ = []