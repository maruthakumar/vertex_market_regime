"""
Test Suite for Base Optimizer

Tests the fundamental BaseOptimizer class and related components
including parameter space validation, objective function handling,
and basic optimization workflow.
"""

import pytest
import numpy as np
from typing import Dict
from unittest.mock import Mock, patch

from ..base.base_optimizer import BaseOptimizer, OptimizationResult
from ..base.parameter_space import ParameterSpace, ParameterType
from ..base.objective_functions import ObjectiveFunctionWrapper


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing BaseOptimizer functionality"""
    
    def __init__(self, param_space, objective_function, **kwargs):
        super().__init__(param_space, objective_function, **kwargs)
        self.mock_iterations = kwargs.get('mock_iterations', 10)
        self.mock_best_value = kwargs.get('mock_best_value', 0.5)
        self.mock_should_fail = kwargs.get('mock_should_fail', False)
    
    def optimize(self, n_iterations=100, callback=None, **kwargs):
        if self.mock_should_fail:
            raise RuntimeError("Mock optimization failure")
        
        # Simulate optimization process
        best_params = {}
        for param, bounds in self.param_space.items():
            # Random parameter within bounds
            best_params[param] = np.random.uniform(bounds[0], bounds[1])
        
        # Simulate callback calls
        if callback:
            for i in range(min(5, n_iterations)):
                callback(i, n_iterations, self.mock_best_value)
        
        return OptimizationResult(
            best_parameters=best_params,
            best_objective_value=self.mock_best_value,
            iterations=min(self.mock_iterations, n_iterations),
            convergence_status='converged' if self.mock_iterations < n_iterations else 'max_iterations',
            optimization_history=[self.mock_best_value] * min(self.mock_iterations, n_iterations),
            metadata={'test': True}
        )


class TestBaseOptimizer:
    """Test cases for BaseOptimizer"""
    
    def test_initialization_basic(self, simple_param_space, sphere_function):
        """Test basic optimizer initialization"""
        optimizer = MockOptimizer(simple_param_space, sphere_function)
        
        assert optimizer.param_space == simple_param_space
        assert optimizer.objective_function == sphere_function
        assert optimizer.random_seed is None
        assert not optimizer.minimize
    
    def test_initialization_with_options(self, simple_param_space, sphere_function):
        """Test optimizer initialization with options"""
        optimizer = MockOptimizer(
            simple_param_space, 
            sphere_function,
            minimize=True,
            random_seed=42,
            custom_option='test'
        )
        
        assert optimizer.minimize is True
        assert optimizer.random_seed == 42
    
    def test_parameter_space_validation(self, sphere_function):
        """Test parameter space validation"""
        # Valid parameter space
        valid_space = {'x': (-1, 1), 'y': (0, 5)}
        optimizer = MockOptimizer(valid_space, sphere_function)
        assert optimizer.param_space == valid_space
        
        # Invalid parameter space - empty
        with pytest.raises(ValueError):
            MockOptimizer({}, sphere_function)
        
        # Invalid parameter space - wrong bounds
        invalid_space = {'x': (5, 1)}  # min > max
        with pytest.raises(ValueError):
            MockOptimizer(invalid_space, sphere_function)
    
    def test_objective_function_validation(self, simple_param_space):
        """Test objective function validation"""
        # Valid function
        def valid_func(params):
            return sum(params.values())
        
        optimizer = MockOptimizer(simple_param_space, valid_func)
        assert optimizer.objective_function == valid_func
        
        # Invalid function - not callable
        with pytest.raises(TypeError):
            MockOptimizer(simple_param_space, "not_a_function")
    
    def test_parameter_bounds_checking(self, simple_param_space, sphere_function):
        """Test parameter bounds checking functionality"""
        optimizer = MockOptimizer(simple_param_space, sphere_function)
        
        # Valid parameters
        valid_params = {'x': 0.0, 'y': 1.0}
        assert optimizer._check_parameter_bounds(valid_params)
        
        # Invalid parameters - out of bounds
        invalid_params = {'x': 10.0, 'y': 1.0}
        assert not optimizer._check_parameter_bounds(invalid_params)
        
        # Missing parameters
        missing_params = {'x': 0.0}
        assert not optimizer._check_parameter_bounds(missing_params)
    
    def test_random_parameter_generation(self, simple_param_space, sphere_function):
        """Test random parameter generation"""
        optimizer = MockOptimizer(simple_param_space, sphere_function, random_seed=42)
        
        params = optimizer._generate_random_parameters()
        
        # Check all parameters are present
        assert set(params.keys()) == set(simple_param_space.keys())
        
        # Check bounds
        assert optimizer._check_parameter_bounds(params)
        
        # Test reproducibility with seed
        optimizer2 = MockOptimizer(simple_param_space, sphere_function, random_seed=42)
        params2 = optimizer2._generate_random_parameters()
        assert params == params2
    
    def test_objective_function_evaluation(self, simple_param_space, sphere_function):
        """Test objective function evaluation"""
        optimizer = MockOptimizer(simple_param_space, sphere_function)
        
        test_params = {'x': 2.0, 'y': 3.0}
        expected_value = 2.0**2 + 3.0**2  # sphere function
        
        result = optimizer._evaluate_objective(test_params)
        assert abs(result - expected_value) < 1e-10
    
    def test_optimization_result_structure(self, simple_param_space, sphere_function):
        """Test optimization result structure"""
        optimizer = MockOptimizer(simple_param_space, sphere_function, mock_best_value=1.5)
        
        result = optimizer.optimize(n_iterations=20)
        
        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'best_parameters')
        assert hasattr(result, 'best_objective_value')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'convergence_status')
        assert hasattr(result, 'optimization_history')
        assert hasattr(result, 'metadata')
        
        # Check values
        assert result.best_objective_value == 1.5
        assert result.iterations <= 20
        assert len(result.optimization_history) == result.iterations
    
    def test_optimization_callback(self, simple_param_space, sphere_function):
        """Test optimization callback functionality"""
        optimizer = MockOptimizer(simple_param_space, sphere_function)
        
        callback_calls = []
        
        def test_callback(iteration, total_iterations, current_best):
            callback_calls.append((iteration, total_iterations, current_best))
        
        result = optimizer.optimize(n_iterations=10, callback=test_callback)
        
        # Check callback was called
        assert len(callback_calls) > 0
        
        # Check callback parameters
        for iteration, total, best in callback_calls:
            assert isinstance(iteration, int)
            assert isinstance(total, int)
            assert isinstance(best, (int, float))
            assert 0 <= iteration < total
    
    def test_optimization_failure_handling(self, simple_param_space, sphere_function):
        """Test handling of optimization failures"""
        optimizer = MockOptimizer(
            simple_param_space, 
            sphere_function, 
            mock_should_fail=True
        )
        
        with pytest.raises(RuntimeError, match="Mock optimization failure"):
            optimizer.optimize()
    
    def test_minimize_maximize_mode(self, simple_param_space):
        """Test minimize vs maximize mode"""
        def test_function(params):
            return params['x'] + params['y']
        
        # Minimize mode
        min_optimizer = MockOptimizer(simple_param_space, test_function, minimize=True)
        assert min_optimizer.minimize is True
        
        # Maximize mode (default)
        max_optimizer = MockOptimizer(simple_param_space, test_function, minimize=False)
        assert max_optimizer.minimize is False
    
    def test_complex_parameter_space(self, complex_param_space, sphere_function):
        """Test with complex multi-dimensional parameter space"""
        optimizer = MockOptimizer(complex_param_space, sphere_function)
        
        # Test parameter generation
        params = optimizer._generate_random_parameters()
        assert len(params) == len(complex_param_space)
        assert optimizer._check_parameter_bounds(params)
        
        # Test optimization
        result = optimizer.optimize()
        assert len(result.best_parameters) == len(complex_param_space)


class TestParameterSpace:
    """Test cases for ParameterSpace class"""
    
    def test_parameter_space_creation(self):
        """Test ParameterSpace creation and validation"""
        param_space = ParameterSpace()
        
        # Add parameters
        param_space.add_parameter('x', ParameterType.CONTINUOUS, (-5, 5))
        param_space.add_parameter('y', ParameterType.DISCRETE, [1, 2, 3, 4, 5])
        
        assert 'x' in param_space
        assert 'y' in param_space
        assert len(param_space) == 2
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        param_space = ParameterSpace()
        
        # Valid continuous parameter
        param_space.add_parameter('continuous', ParameterType.CONTINUOUS, (-1, 1))
        
        # Valid discrete parameter
        param_space.add_parameter('discrete', ParameterType.DISCRETE, [1, 2, 3])
        
        # Invalid bounds
        with pytest.raises(ValueError):
            param_space.add_parameter('invalid', ParameterType.CONTINUOUS, (5, 1))
    
    def test_parameter_sampling(self):
        """Test parameter sampling from space"""
        param_space = ParameterSpace()
        param_space.add_parameter('x', ParameterType.CONTINUOUS, (-2, 2))
        param_space.add_parameter('y', ParameterType.DISCRETE, [10, 20, 30])
        
        sample = param_space.sample_parameters()
        
        assert 'x' in sample
        assert 'y' in sample
        assert -2 <= sample['x'] <= 2
        assert sample['y'] in [10, 20, 30]


class TestObjectiveFunctionWrapper:
    """Test cases for ObjectiveFunctionWrapper"""
    
    def test_basic_wrapping(self, sphere_function):
        """Test basic function wrapping"""
        wrapper = ObjectiveFunctionWrapper(sphere_function)
        
        test_params = {'x': 1.0, 'y': 2.0}
        expected = 1.0**2 + 2.0**2
        
        result = wrapper(test_params)
        assert abs(result - expected) < 1e-10
    
    def test_evaluation_counting(self, sphere_function):
        """Test function evaluation counting"""
        wrapper = ObjectiveFunctionWrapper(sphere_function, track_evaluations=True)
        
        assert wrapper.evaluation_count == 0
        
        wrapper({'x': 1, 'y': 2})
        assert wrapper.evaluation_count == 1
        
        wrapper({'x': 3, 'y': 4})
        assert wrapper.evaluation_count == 2
    
    def test_caching(self, sphere_function):
        """Test function result caching"""
        wrapper = ObjectiveFunctionWrapper(sphere_function, use_cache=True)
        
        params = {'x': 1.0, 'y': 2.0}
        
        # First evaluation
        result1 = wrapper(params)
        
        # Second evaluation (should be cached)
        result2 = wrapper(params)
        
        assert result1 == result2
        assert len(wrapper.cache) == 1
    
    def test_noise_addition(self, sphere_function):
        """Test noise addition to function evaluations"""
        wrapper = ObjectiveFunctionWrapper(
            sphere_function, 
            add_noise=True, 
            noise_level=0.1
        )
        
        params = {'x': 1.0, 'y': 2.0}
        clean_result = sphere_function(params)
        
        # Multiple evaluations should give different results due to noise
        results = [wrapper(params) for _ in range(10)]
        
        # Results should be different but close to clean result
        assert len(set(results)) > 1  # Different due to noise
        mean_result = np.mean(results)
        assert abs(mean_result - clean_result) < 0.5  # Should be close
    
    def test_constraint_handling(self):
        """Test constraint handling in wrapper"""
        def simple_func(params):
            return params['x']**2
        
        def constraint(params):
            return params['x'] >= 0  # x must be non-negative
        
        wrapper = ObjectiveFunctionWrapper(
            simple_func, 
            constraints=[constraint],
            penalty_value=1000
        )
        
        # Valid parameters
        valid_params = {'x': 2.0}
        result = wrapper(valid_params)
        assert result == 4.0  # x^2
        
        # Invalid parameters
        invalid_params = {'x': -2.0}
        result = wrapper(invalid_params)
        assert result == 1000  # Penalty value
    
    def test_timeout_handling(self, sphere_function):
        """Test timeout handling"""
        def slow_function(params):
            import time
            time.sleep(0.1)  # Simulate slow function
            return sphere_function(params)
        
        wrapper = ObjectiveFunctionWrapper(slow_function, timeout_seconds=0.05)
        
        params = {'x': 1.0, 'y': 2.0}
        
        # Should raise timeout exception or return penalty
        result = wrapper(params)
        # Implementation dependent - could be penalty value or exception


class TestOptimizationResult:
    """Test cases for OptimizationResult dataclass"""
    
    def test_result_creation(self):
        """Test OptimizationResult creation"""
        result = OptimizationResult(
            best_parameters={'x': 1.0, 'y': 2.0},
            best_objective_value=5.0,
            iterations=100,
            convergence_status='converged'
        )
        
        assert result.best_parameters == {'x': 1.0, 'y': 2.0}
        assert result.best_objective_value == 5.0
        assert result.iterations == 100
        assert result.convergence_status == 'converged'
    
    def test_result_with_optional_fields(self):
        """Test OptimizationResult with optional fields"""
        history = [10.0, 8.0, 6.0, 5.0]
        metadata = {'algorithm': 'test', 'time': 1.5}
        
        result = OptimizationResult(
            best_parameters={'x': 1.0},
            best_objective_value=5.0,
            iterations=4,
            convergence_status='converged',
            optimization_history=history,
            metadata=metadata
        )
        
        assert result.optimization_history == history
        assert result.metadata == metadata
    
    def test_result_improvement_calculation(self):
        """Test improvement calculation"""
        result = OptimizationResult(
            best_parameters={'x': 1.0},
            best_objective_value=5.0,
            iterations=10,
            convergence_status='converged',
            optimization_history=[10.0, 8.0, 6.0, 5.0]
        )
        
        # Should calculate improvement from start to end
        expected_improvement = (10.0 - 5.0) / 10.0  # 50% improvement
        assert abs(result.improvement - expected_improvement) < 1e-10


@pytest.mark.parametrize("algorithm_class", [MockOptimizer])
def test_algorithm_interface_compliance(algorithm_class, simple_param_space, sphere_function):
    """Test that algorithms comply with BaseOptimizer interface"""
    optimizer = algorithm_class(simple_param_space, sphere_function)
    
    # Must be instance of BaseOptimizer
    assert isinstance(optimizer, BaseOptimizer)
    
    # Must have optimize method
    assert hasattr(optimizer, 'optimize')
    assert callable(optimizer.optimize)
    
    # Optimize method must return OptimizationResult
    result = optimizer.optimize(n_iterations=5)
    assert isinstance(result, OptimizationResult)
    
    # Result must have required fields
    assert hasattr(result, 'best_parameters')
    assert hasattr(result, 'best_objective_value')
    assert hasattr(result, 'iterations')
    assert hasattr(result, 'convergence_status')