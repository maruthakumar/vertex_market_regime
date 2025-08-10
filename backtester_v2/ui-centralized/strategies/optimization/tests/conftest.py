"""
Test Configuration and Fixtures

Provides common test fixtures, utilities, and configuration for the
optimization module test suite.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Callable, Tuple, List
import tempfile
import os
from pathlib import Path


@pytest.fixture
def simple_param_space():
    """Simple 2D parameter space for testing"""
    return {
        'x': (-5.0, 5.0),
        'y': (-3.0, 3.0)
    }


@pytest.fixture
def complex_param_space():
    """Complex multi-dimensional parameter space"""
    return {
        'param1': (-10.0, 10.0),
        'param2': (-5.0, 5.0),
        'param3': (0.0, 1.0),
        'param4': (-2.0, 2.0),
        'param5': (-1.0, 1.0)
    }


@pytest.fixture
def sphere_function():
    """Sphere function - simple convex optimization problem"""
    def objective(params: Dict[str, float]) -> float:
        return sum(v**2 for v in params.values())
    return objective


@pytest.fixture
def rosenbrock_function():
    """Rosenbrock function - classic optimization benchmark"""
    def objective(params: Dict[str, float]) -> float:
        values = list(params.values())
        if len(values) < 2:
            return sum(v**2 for v in values)
        
        result = 0
        for i in range(len(values) - 1):
            result += 100 * (values[i+1] - values[i]**2)**2 + (1 - values[i])**2
        return result
    return objective


@pytest.fixture
def rastrigin_function():
    """Rastrigin function - multimodal optimization problem"""
    def objective(params: Dict[str, float]) -> float:
        values = list(params.values())
        A = 10
        n = len(values)
        return A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in values)
    return objective


@pytest.fixture
def noisy_function():
    """Function with added noise for robust optimization testing"""
    def objective(params: Dict[str, float]) -> float:
        # Base function (sphere)
        base_value = sum(v**2 for v in params.values())
        # Add noise
        noise = np.random.normal(0, 0.1) * base_value
        return base_value + noise
    return objective


@pytest.fixture
def sample_strategy_returns():
    """Sample strategy returns for inversion testing"""
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate correlated strategy returns
    base_returns = np.random.normal(0.001, 0.02, 252)
    
    strategies = {}
    strategies['strategy_1'] = pd.Series(base_returns, index=dates)
    strategies['strategy_2'] = pd.Series(base_returns * 0.8 + np.random.normal(0, 0.01, 252), index=dates)
    strategies['strategy_3'] = pd.Series(-base_returns * 0.5 + np.random.normal(0, 0.015, 252), index=dates)
    
    return strategies


@pytest.fixture
def sample_market_data():
    """Sample market data for regime analysis"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Simulate market index returns
    market_returns = np.random.normal(0.0005, 0.015, 252)
    cumulative_prices = (1 + pd.Series(market_returns, index=dates)).cumprod() * 1000
    
    return pd.DataFrame({
        'date': dates,
        'close': cumulative_prices,
        'returns': market_returns,
        'volume': np.random.randint(1000000, 10000000, 252)
    })


@pytest.fixture
def temp_metadata_file():
    """Temporary file for metadata storage testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability for testing"""
    return True


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailability for testing"""
    return False


class TestObjectiveFunctions:
    """Collection of test objective functions"""
    
    @staticmethod
    def linear_function(params: Dict[str, float]) -> float:
        """Simple linear function"""
        return sum(params.values())
    
    @staticmethod
    def quadratic_function(params: Dict[str, float]) -> float:
        """Quadratic function with known minimum"""
        # Minimum at x=1, y=2
        x = params.get('x', 0)
        y = params.get('y', 0)
        return (x - 1)**2 + (y - 2)**2
    
    @staticmethod
    def constrained_function(params: Dict[str, float]) -> float:
        """Function with implicit constraints"""
        x = params.get('x', 0)
        y = params.get('y', 0)
        
        # Penalty for constraint violations
        penalty = 0
        if x + y > 1:  # Constraint: x + y <= 1
            penalty = 1000 * (x + y - 1)**2
        
        return x**2 + y**2 + penalty
    
    @staticmethod
    def discontinuous_function(params: Dict[str, float]) -> float:
        """Discontinuous function for robustness testing"""
        x = params.get('x', 0)
        y = params.get('y', 0)
        
        # Step function
        if x > 0 and y > 0:
            return 10
        elif x < 0 and y < 0:
            return 5
        else:
            return x**2 + y**2


@pytest.fixture
def test_functions():
    """Collection of test functions for benchmarking"""
    return TestObjectiveFunctions()


@pytest.fixture
def benchmark_suite():
    """Standard benchmark problems for algorithm testing"""
    return [
        # (param_space, objective_function, known_minimum, description)
        (
            {'x': (-5, 5), 'y': (-5, 5)},
            lambda p: p['x']**2 + p['y']**2,
            0.0,
            "Sphere Function"
        ),
        (
            {'x': (-2, 2), 'y': (-1, 3)},
            lambda p: 100 * (p['y'] - p['x']**2)**2 + (1 - p['x'])**2,
            0.0,
            "Rosenbrock Function"
        ),
        (
            {'x': (-5.12, 5.12), 'y': (-5.12, 5.12)},
            lambda p: 20 + p['x']**2 - 10*np.cos(2*np.pi*p['x']) + p['y']**2 - 10*np.cos(2*np.pi*p['y']),
            0.0,
            "Rastrigin Function"
        ),
        (
            {'x': (-500, 500), 'y': (-500, 500)},
            lambda p: 418.9829*2 - p['x']*np.sin(np.sqrt(abs(p['x']))) - p['y']*np.sin(np.sqrt(abs(p['y']))),
            0.0,
            "Schwefel Function"
        )
    ]


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for algorithm validation"""
    return {
        'max_execution_time': 30.0,  # seconds
        'min_improvement': 0.1,  # 10% improvement
        'convergence_tolerance': 1e-6,
        'max_iterations': 1000,
        'success_rate_threshold': 0.7  # 70% success rate for multiple runs
    }


class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_optimization_result(result, param_space, min_improvement=0.01):
        """Assert that optimization result meets basic criteria"""
        assert hasattr(result, 'best_parameters')
        assert hasattr(result, 'best_objective_value')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'convergence_status')
        
        # Check parameter bounds
        for param, value in result.best_parameters.items():
            if param in param_space:
                bounds = param_space[param]
                assert bounds[0] <= value <= bounds[1], f"Parameter {param} out of bounds"
        
        # Check improvement (assuming minimization)
        if hasattr(result, 'improvement') and result.improvement is not None:
            assert result.improvement >= min_improvement, "Insufficient improvement"
    
    @staticmethod
    def run_algorithm_stress_test(algorithm, param_space, objective_function, 
                                 num_runs=5, max_time=60):
        """Run stress test on algorithm"""
        results = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            try:
                result = algorithm.optimize(n_iterations=100)
                execution_time = time.time() - start_time
                
                results.append({
                    'run': i,
                    'success': True,
                    'execution_time': execution_time,
                    'best_value': result.best_objective_value,
                    'improvement': getattr(result, 'improvement', None)
                })
                
                # Check timeout
                if execution_time > max_time:
                    break
                    
            except Exception as e:
                results.append({
                    'run': i,
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                })
        
        return results
    
    @staticmethod
    def calculate_success_metrics(results):
        """Calculate success metrics from test results"""
        successful_runs = [r for r in results if r.get('success', False)]
        
        if not successful_runs:
            return {
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'avg_improvement': 0.0,
                'total_runs': len(results)
            }
        
        return {
            'success_rate': len(successful_runs) / len(results),
            'avg_execution_time': np.mean([r['execution_time'] for r in successful_runs]),
            'avg_improvement': np.mean([r.get('improvement', 0) for r in successful_runs]),
            'best_result': min(successful_runs, key=lambda x: x.get('best_value', float('inf'))),
            'total_runs': len(results),
            'successful_runs': len(successful_runs)
        }


@pytest.fixture
def test_utilities():
    """Test utility functions"""
    return TestUtilities()


# Import time for utilities
import time