"""
Performance Benchmarking Module

Comprehensive benchmarking system for optimization algorithms including
performance comparison, scalability analysis, and recommendation generation.
"""

from .benchmark_suite import BenchmarkSuite
from .performance_analyzer import PerformanceAnalyzer
from .scalability_tester import ScalabilityTester
from .benchmark_runner import BenchmarkRunner

__version__ = "1.0.0"
__author__ = "Strategy Optimization Team"

__all__ = [
    "BenchmarkSuite",
    "PerformanceAnalyzer", 
    "ScalabilityTester",
    "BenchmarkRunner"
]