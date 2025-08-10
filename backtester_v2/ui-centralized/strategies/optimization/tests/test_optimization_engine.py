"""
Test Suite for Optimization Engine

Tests the unified OptimizationEngine interface including algorithm selection,
batch optimization, benchmarking, and performance tracking.
"""

import pytest
import tempfile
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

from ..engines.optimization_engine import (
    OptimizationEngine, OptimizationRequest, OptimizationSummary, 
    BatchOptimizationResult
)
from ..engines.algorithm_registry import AlgorithmRegistry
from ..base.base_optimizer import BaseOptimizer, OptimizationResult


class MockOptimizationEngine(OptimizationEngine):
    """Mock optimization engine for testing"""
    
    def __init__(self, **kwargs):
        # Override initialization to avoid real algorithm discovery
        self.algorithms_package = kwargs.get('algorithms_package', 'test.algorithms')
        self.enable_gpu = kwargs.get('enable_gpu', True)
        self.enable_parallel = kwargs.get('enable_parallel', True)
        self.max_workers = kwargs.get('max_workers', 2)
        self.cache_results = kwargs.get('cache_results', True)
        
        # Mock components
        self.registry = Mock(spec=AlgorithmRegistry)
        self.metadata_manager = Mock()
        
        # Mock discovery result
        self.registry.discover_algorithms.return_value = {
            'total_algorithms': 3,
            'discovered_algorithms': ['mock_alg1', 'mock_alg2', 'mock_alg3'],
            'discovery_errors': 0
        }
        
        # State
        self.optimization_history = []
        self.algorithm_performance = {}
        self.result_cache = {}
        self.active_optimizations = {}
        self.optimization_counter = 0
        
        # Specialized optimizers
        self.robust_optimizer = None
        self.gpu_optimizer = None
        self.inversion_engine = None


class TestOptimizationEngine:
    """Test cases for OptimizationEngine"""
    
    def test_engine_initialization(self):
        """Test optimization engine initialization"""
        with patch('strategies.optimization.engines.optimization_engine.AlgorithmRegistry'):
            engine = MockOptimizationEngine()
            
            assert engine.algorithms_package == 'test.algorithms'
            assert engine.enable_gpu is True
            assert engine.enable_parallel is True
            assert engine.max_workers == 2
            assert engine.cache_results is True
            assert len(engine.optimization_history) == 0
    
    def test_optimization_request_creation(self, simple_param_space, sphere_function):
        """Test optimization request creation"""
        request = OptimizationRequest(
            param_space=simple_param_space,
            objective_function=sphere_function,
            optimization_mode="balanced",
            max_iterations=500
        )
        
        assert request.param_space == simple_param_space
        assert request.objective_function == sphere_function
        assert request.optimization_mode == "balanced"
        assert request.max_iterations == 500
        assert request.enable_gpu is True  # Default
        assert request.enable_robustness is True  # Default
    
    def test_basic_optimization(self, simple_param_space, sphere_function):
        """Test basic optimization workflow"""
        engine = MockOptimizationEngine()
        
        # Mock algorithm selection and creation
        mock_algorithm = Mock(spec=BaseOptimizer)
        mock_result = OptimizationResult(
            best_parameters={'x': 1.0, 'y': 2.0},
            best_objective_value=5.0,
            iterations=50,
            convergence_status='converged'
        )
        mock_algorithm.optimize.return_value = mock_result
        
        engine.registry.list_algorithms.return_value = ['test_algorithm']
        engine.registry.get_algorithm.return_value = mock_algorithm
        engine.registry.recommend_algorithms.return_value = [('test_algorithm', 0.9)]
        
        # Run optimization
        summary = engine.optimize(
            param_space=simple_param_space,
            objective_function=sphere_function,
            algorithm='test_algorithm'
        )
        
        # Verify result
        assert isinstance(summary, OptimizationSummary)
        assert summary.algorithm_used == 'test_algorithm'
        assert summary.best_parameters == {'x': 1.0, 'y': 2.0}
        assert summary.best_objective_value == 5.0
        assert summary.convergence_status == 'converged'
        assert len(engine.optimization_history) == 1
    
    def test_algorithm_selection_with_preferences(self, simple_param_space, sphere_function):
        """Test algorithm selection with user preferences"""
        engine = MockOptimizationEngine()
        
        # Mock available algorithms
        engine.registry.algorithm_classes = {
            'preferred_alg': Mock,
            'other_alg': Mock
        }
        
        # Test with preferred algorithm
        selected = engine._select_algorithm(OptimizationRequest(
            param_space=simple_param_space,
            objective_function=sphere_function,
            algorithm_preferences=['preferred_alg']
        ))
        
        assert selected == 'preferred_alg'
    
    def test_algorithm_selection_auto(self, simple_param_space, sphere_function):
        """Test automatic algorithm selection"""
        engine = MockOptimizationEngine()
        
        # Mock recommendation system
        engine.registry.recommend_algorithms.return_value = [
            ('best_alg', 0.9),
            ('good_alg', 0.7)
        ]
        
        selected = engine._select_algorithm(OptimizationRequest(
            param_space=simple_param_space,
            objective_function=sphere_function
        ))
        
        assert selected == 'best_alg'
    
    def test_optimization_caching(self, simple_param_space, sphere_function):
        """Test optimization result caching"""
        engine = MockOptimizationEngine(cache_results=True)
        
        # First optimization
        mock_algorithm = Mock(spec=BaseOptimizer)
        mock_result = OptimizationResult(
            best_parameters={'x': 0.0, 'y': 0.0},
            best_objective_value=0.0,
            iterations=10,
            convergence_status='converged'
        )
        mock_algorithm.optimize.return_value = mock_result
        
        engine.registry.get_algorithm.return_value = mock_algorithm
        engine.registry.recommend_algorithms.return_value = [('test_alg', 0.9)]
        
        # Run first optimization
        summary1 = engine.optimize(
            param_space=simple_param_space,
            objective_function=sphere_function
        )
        
        # Run second optimization with same parameters (should be cached)
        summary2 = engine.optimize(
            param_space=simple_param_space,
            objective_function=sphere_function
        )
        
        # Second optimization should be faster (cached)
        assert summary2.execution_time < summary1.execution_time
        assert summary2.convergence_status == 'cached'
    
    def test_optimization_failure_handling(self, simple_param_space, sphere_function):
        """Test handling of optimization failures"""
        engine = MockOptimizationEngine()
        
        # Mock failing algorithm
        mock_algorithm = Mock(spec=BaseOptimizer)
        mock_algorithm.optimize.side_effect = RuntimeError("Optimization failed")
        
        engine.registry.get_algorithm.return_value = mock_algorithm
        engine.registry.recommend_algorithms.return_value = [('failing_alg', 0.9)]
        
        # Run optimization (should handle failure gracefully)
        summary = engine.optimize(
            param_space=simple_param_space,
            objective_function=sphere_function
        )
        
        assert summary.convergence_status == 'failed'
        assert 'error' in summary.metadata
        assert len(engine.optimization_history) == 1
    
    def test_batch_optimization_sequential(self, simple_param_space, sphere_function):
        """Test sequential batch optimization"""
        engine = MockOptimizationEngine(enable_parallel=False)
        
        # Create multiple requests
        requests = [
            OptimizationRequest(simple_param_space, sphere_function),
            OptimizationRequest(simple_param_space, sphere_function),
            OptimizationRequest(simple_param_space, sphere_function)
        ]
        
        # Mock successful optimization
        with patch.object(engine, '_execute_optimization') as mock_exec:
            mock_exec.return_value = OptimizationSummary(
                request_id='test',
                algorithm_used='test_alg',
                execution_time=1.0,
                iterations_completed=50,
                best_parameters={'x': 0, 'y': 0},
                best_objective_value=0.0,
                improvement_achieved=0.5,
                convergence_status='converged'
            )
            
            result = engine.batch_optimize(requests, parallel_execution=False)
            
            assert isinstance(result, BatchOptimizationResult)
            assert result.total_optimizations == 3
            assert result.successful_optimizations == 3
            assert result.failed_optimizations == 0
            assert mock_exec.call_count == 3
    
    def test_batch_optimization_parallel(self, simple_param_space, sphere_function):
        """Test parallel batch optimization"""
        engine = MockOptimizationEngine(enable_parallel=True)
        
        requests = [
            OptimizationRequest(simple_param_space, sphere_function),
            OptimizationRequest(simple_param_space, sphere_function)
        ]
        
        # Mock successful optimization
        with patch.object(engine, '_execute_optimization') as mock_exec:
            mock_exec.return_value = OptimizationSummary(
                request_id='test',
                algorithm_used='test_alg',
                execution_time=1.0,
                iterations_completed=50,
                best_parameters={'x': 0, 'y': 0},
                best_objective_value=0.0,
                improvement_achieved=0.5,
                convergence_status='converged'
            )
            
            result = engine.batch_optimize(requests, parallel_execution=True)
            
            assert isinstance(result, BatchOptimizationResult)
            assert result.total_optimizations == 2
            assert result.execution_summary['parallel_execution'] is True
    
    def test_algorithm_recommendation(self, simple_param_space):
        """Test algorithm recommendation system"""
        engine = MockOptimizationEngine()
        
        # Mock registry recommendations
        engine.registry.recommend_algorithms.return_value = [
            ('best_alg', 0.95),
            ('good_alg', 0.8),
            ('okay_alg', 0.6)
        ]
        
        problem_characteristics = {
            'dimensions': len(simple_param_space),
            'problem_type': 'continuous'
        }
        
        recommendations = engine.recommend_algorithm(problem_characteristics)
        
        assert len(recommendations) == 3
        assert recommendations[0][0] == 'best_alg'
        assert recommendations[0][1] == 0.95
    
    def test_benchmarking_algorithms(self, benchmark_suite):
        """Test algorithm benchmarking functionality"""
        engine = MockOptimizationEngine()
        
        # Mock optimization execution
        with patch.object(engine, '_execute_optimization') as mock_exec:
            mock_exec.return_value = OptimizationSummary(
                request_id='benchmark',
                algorithm_used='test_alg',
                execution_time=0.5,
                iterations_completed=100,
                best_parameters={'x': 0, 'y': 0},
                best_objective_value=0.1,
                improvement_achieved=0.8,
                convergence_status='converged'
            )
            
            # Run benchmark with subset of test functions
            test_functions = benchmark_suite[:2]  # First 2 functions
            
            results = engine.benchmark_algorithms(
                test_functions=test_functions,
                algorithms=['test_alg'],
                iterations_per_test=2
            )
            
            assert 'results' in results
            assert 'rankings' in results
            assert 'summary' in results
            assert results['algorithms_tested'] == 1
            assert results['test_functions'] == 2
    
    def test_optimization_history_tracking(self, simple_param_space, sphere_function):
        """Test optimization history tracking"""
        engine = MockOptimizationEngine()
        
        # Mock multiple optimizations
        with patch.object(engine, '_execute_optimization') as mock_exec:
            mock_exec.side_effect = [
                OptimizationSummary(
                    request_id='opt1',
                    algorithm_used='alg1',
                    execution_time=1.0,
                    iterations_completed=50,
                    best_parameters={'x': 1, 'y': 1},
                    best_objective_value=2.0,
                    improvement_achieved=0.5,
                    convergence_status='converged'
                ),
                OptimizationSummary(
                    request_id='opt2',
                    algorithm_used='alg2',
                    execution_time=2.0,
                    iterations_completed=100,
                    best_parameters={'x': 0, 'y': 0},
                    best_objective_value=0.0,
                    improvement_achieved=0.9,
                    convergence_status='converged'
                )
            ]
            
            # Run optimizations
            engine.optimize(simple_param_space, sphere_function)
            engine.optimize(simple_param_space, sphere_function)
            
            # Check history
            history = engine.get_optimization_history()
            assert len(history) == 2
            assert history[0].algorithm_used == 'alg1'
            assert history[1].algorithm_used == 'alg2'
            
            # Test filtered history
            alg1_history = engine.get_optimization_history(algorithm='alg1')
            assert len(alg1_history) == 1
            assert alg1_history[0].algorithm_used == 'alg1'
    
    def test_engine_statistics(self):
        """Test engine statistics generation"""
        engine = MockOptimizationEngine()
        
        # Add mock history
        engine.optimization_history = [
            OptimizationSummary(
                request_id='opt1',
                algorithm_used='alg1',
                execution_time=1.0,
                iterations_completed=50,
                best_parameters={},
                best_objective_value=1.0,
                improvement_achieved=0.5,
                convergence_status='converged'
            ),
            OptimizationSummary(
                request_id='opt2',
                algorithm_used='alg2',
                execution_time=2.0,
                iterations_completed=100,
                best_parameters={},
                best_objective_value=0.5,
                improvement_achieved=0.8,
                convergence_status='failed'
            )
        ]
        
        # Mock registry stats
        engine.registry.get_registry_statistics.return_value = {
            'discovery_status': {'total_algorithms': 5}
        }
        
        stats = engine.get_engine_statistics()
        
        assert stats['total_optimizations'] == 2
        assert stats['successful_optimizations'] == 1
        assert stats['success_rate'] == 0.5
        assert 'algorithm_usage' in stats
        assert 'performance_stats' in stats
    
    def test_cache_management(self):
        """Test cache clearing and management"""
        engine = MockOptimizationEngine()
        
        # Add some cached data
        engine.result_cache['key1'] = Mock()
        engine.result_cache['key2'] = Mock()
        
        # Mock registry cache
        engine.registry.clear_cache = Mock()
        
        # Clear cache
        engine.clear_cache()
        
        assert len(engine.result_cache) == 0
        engine.registry.clear_cache.assert_called_once()
    
    def test_optimization_history_persistence(self, temp_metadata_file):
        """Test saving and loading optimization history"""
        engine = MockOptimizationEngine()
        
        # Add mock history
        engine.optimization_history = [
            OptimizationSummary(
                request_id='opt1',
                algorithm_used='test_alg',
                execution_time=1.0,
                iterations_completed=50,
                best_parameters={'x': 1.0, 'y': 2.0},
                best_objective_value=5.0,
                improvement_achieved=0.5,
                convergence_status='converged',
                metadata={'test': True}
            )
        ]
        
        # Save history
        engine.save_optimization_history(temp_metadata_file)
        
        # Verify file was created and contains data
        assert Path(temp_metadata_file).exists()
        
        with open(temp_metadata_file, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]['algorithm_used'] == 'test_alg'
        assert data[0]['best_parameters'] == {'x': 1.0, 'y': 2.0}
    
    def test_gpu_optimizer_integration(self, simple_param_space, sphere_function):
        """Test GPU optimizer integration"""
        engine = MockOptimizationEngine(enable_gpu=True)
        
        # Mock GPU optimizer
        mock_gpu_optimizer = Mock()
        
        with patch.object(engine, '_get_gpu_optimizer') as mock_get_gpu:
            mock_get_gpu.return_value = mock_gpu_optimizer
            
            request = OptimizationRequest(
                param_space=simple_param_space,
                objective_function=sphere_function,
                enable_gpu=True
            )
            
            optimizer = engine._create_optimizer('test_alg', request)
            
            # Should return GPU optimizer when GPU is enabled
            assert optimizer == mock_gpu_optimizer
    
    def test_robust_optimizer_integration(self, simple_param_space, sphere_function):
        """Test robust optimizer integration"""
        engine = MockOptimizationEngine()
        
        # Mock robust optimizer
        mock_robust_optimizer = Mock()
        mock_base_optimizer = Mock()
        
        with patch.object(engine, '_get_robust_optimizer') as mock_get_robust:
            mock_get_robust.return_value = mock_robust_optimizer
            engine.registry.get_algorithm.return_value = mock_base_optimizer
            
            request = OptimizationRequest(
                param_space=simple_param_space,
                objective_function=sphere_function,
                enable_robustness=True,
                enable_gpu=False  # Disable GPU to test robust path
            )
            
            optimizer = engine._create_optimizer('test_alg', request)
            
            # Should return robust optimizer with base optimizer set
            assert optimizer == mock_robust_optimizer
            assert mock_robust_optimizer.base_optimizer == mock_base_optimizer
    
    def test_callback_functionality(self, simple_param_space, sphere_function):
        """Test optimization callback functionality"""
        engine = MockOptimizationEngine()
        
        callback_calls = []
        
        def test_callback(iteration, total, current_best):
            callback_calls.append((iteration, total, current_best))
        
        # Mock algorithm with callback support
        mock_algorithm = Mock(spec=BaseOptimizer)
        mock_result = OptimizationResult(
            best_parameters={'x': 0, 'y': 0},
            best_objective_value=0.0,
            iterations=10,
            convergence_status='converged'
        )
        
        def mock_optimize(n_iterations=100, callback=None, **kwargs):
            if callback:
                for i in range(5):
                    callback(i, n_iterations, 0.1 * i)
            return mock_result
        
        mock_algorithm.optimize = mock_optimize
        engine.registry.get_algorithm.return_value = mock_algorithm
        engine.registry.recommend_algorithms.return_value = [('test_alg', 0.9)]
        
        # Run optimization with callback
        engine.optimize(
            param_space=simple_param_space,
            objective_function=sphere_function,
            callback=test_callback
        )
        
        # Verify callback was called
        assert len(callback_calls) > 0
        
        # Verify callback parameters
        for iteration, total, best in callback_calls:
            assert isinstance(iteration, int)
            assert isinstance(total, int)
            assert isinstance(best, (int, float))


class TestOptimizationRequest:
    """Test cases for OptimizationRequest"""
    
    def test_request_defaults(self, simple_param_space, sphere_function):
        """Test optimization request default values"""
        request = OptimizationRequest(
            param_space=simple_param_space,
            objective_function=sphere_function
        )
        
        assert request.algorithm_preferences is None
        assert request.resource_constraints is None
        assert request.performance_requirements is None
        assert request.optimization_mode == "balanced"
        assert request.enable_gpu is True
        assert request.enable_robustness is True
        assert request.enable_inversion_analysis is False
        assert request.max_iterations == 1000
        assert request.target_improvement is None
        assert request.timeout_seconds is None
    
    def test_request_customization(self, simple_param_space, sphere_function):
        """Test optimization request customization"""
        request = OptimizationRequest(
            param_space=simple_param_space,
            objective_function=sphere_function,
            algorithm_preferences=['genetic_algorithm'],
            optimization_mode="speed",
            enable_gpu=False,
            max_iterations=500,
            target_improvement=0.1,
            timeout_seconds=60.0
        )
        
        assert request.algorithm_preferences == ['genetic_algorithm']
        assert request.optimization_mode == "speed"
        assert request.enable_gpu is False
        assert request.max_iterations == 500
        assert request.target_improvement == 0.1
        assert request.timeout_seconds == 60.0


class TestOptimizationSummary:
    """Test cases for OptimizationSummary"""
    
    def test_summary_creation(self):
        """Test optimization summary creation"""
        summary = OptimizationSummary(
            request_id="test_123",
            algorithm_used="test_algorithm",
            execution_time=2.5,
            iterations_completed=150,
            best_parameters={'x': 1.0, 'y': 2.0},
            best_objective_value=5.0,
            improvement_achieved=0.75,
            convergence_status='converged'
        )
        
        assert summary.request_id == "test_123"
        assert summary.algorithm_used == "test_algorithm"
        assert summary.execution_time == 2.5
        assert summary.iterations_completed == 150
        assert summary.best_parameters == {'x': 1.0, 'y': 2.0}
        assert summary.best_objective_value == 5.0
        assert summary.improvement_achieved == 0.75
        assert summary.convergence_status == 'converged'
    
    def test_summary_with_optional_fields(self):
        """Test optimization summary with optional fields"""
        robustness_metrics = {'cv_score': 0.85, 'sensitivity': 0.1}
        gpu_utilization = {'memory_used': 1024, 'compute_time': 1.2}
        metadata = {'notes': 'test optimization'}
        
        summary = OptimizationSummary(
            request_id="test",
            algorithm_used="test_alg",
            execution_time=1.0,
            iterations_completed=50,
            best_parameters={},
            best_objective_value=0.0,
            improvement_achieved=0.5,
            convergence_status='converged',
            robustness_metrics=robustness_metrics,
            gpu_utilization=gpu_utilization,
            metadata=metadata
        )
        
        assert summary.robustness_metrics == robustness_metrics
        assert summary.gpu_utilization == gpu_utilization
        assert summary.metadata == metadata


class TestBatchOptimizationResult:
    """Test cases for BatchOptimizationResult"""
    
    def test_batch_result_creation(self):
        """Test batch optimization result creation"""
        best_result = OptimizationSummary(
            request_id="best",
            algorithm_used="best_alg",
            execution_time=1.0,
            iterations_completed=50,
            best_parameters={},
            best_objective_value=0.0,
            improvement_achieved=0.9,
            convergence_status='converged'
        )
        
        algorithm_performance = {
            'alg1': {'avg_execution_time': 1.0, 'success_rate': 0.8},
            'alg2': {'avg_execution_time': 2.0, 'success_rate': 0.6}
        }
        
        execution_summary = {
            'total_execution_time': 10.0,
            'parallel_execution': True
        }
        
        batch_result = BatchOptimizationResult(
            total_optimizations=10,
            successful_optimizations=8,
            failed_optimizations=2,
            best_overall_result=best_result,
            algorithm_performance=algorithm_performance,
            execution_summary=execution_summary,
            detailed_results=[best_result]
        )
        
        assert batch_result.total_optimizations == 10
        assert batch_result.successful_optimizations == 8
        assert batch_result.failed_optimizations == 2
        assert batch_result.best_overall_result == best_result
        assert batch_result.algorithm_performance == algorithm_performance
        assert batch_result.execution_summary == execution_summary
        assert len(batch_result.detailed_results) == 1


class TestOptimizationEngineEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_no_algorithms_available(self, simple_param_space, sphere_function):
        """Test behavior when no algorithms are available"""
        engine = MockOptimizationEngine()
        
        # Mock empty algorithm registry
        engine.registry.list_algorithms.return_value = []
        engine.registry.recommend_algorithms.return_value = []
        
        with pytest.raises(RuntimeError, match="No algorithms available"):
            engine._select_algorithm(OptimizationRequest(
                param_space=simple_param_space,
                objective_function=sphere_function
            ))
    
    def test_invalid_algorithm_name(self, simple_param_space, sphere_function):
        """Test behavior with invalid algorithm name"""
        engine = MockOptimizationEngine()
        
        # Mock algorithm not found
        engine.registry.get_algorithm.side_effect = ValueError("Algorithm not found")
        
        # Should handle gracefully and create failed summary
        summary = engine.optimize(
            param_space=simple_param_space,
            objective_function=sphere_function,
            algorithm='nonexistent_algorithm'
        )
        
        assert summary.convergence_status == 'failed'
    
    def test_empty_batch_optimization(self):
        """Test batch optimization with empty request list"""
        engine = MockOptimizationEngine()
        
        result = engine.batch_optimize([])
        
        assert result.total_optimizations == 0
        assert result.successful_optimizations == 0
        assert result.failed_optimizations == 0
        assert result.best_overall_result is None