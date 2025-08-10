"""
Test Suite for Algorithm Registry

Tests the algorithm discovery, registration, metadata management,
and recommendation systems of the AlgorithmRegistry.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import importlib.util

from ..engines.algorithm_registry import AlgorithmRegistry, AlgorithmLoadError, AlgorithmValidationError
from ..engines.algorithm_metadata import (
    AlgorithmMetadata, AlgorithmCategory, ProblemType, 
    AlgorithmCapabilities, ResourceRequirements, ComplexityLevel
)
from ..base.base_optimizer import BaseOptimizer, OptimizationResult


class TestAlgorithmRegistry:
    """Test cases for AlgorithmRegistry"""
    
    def test_registry_initialization(self, temp_metadata_file):
        """Test registry initialization"""
        registry = AlgorithmRegistry(
            algorithms_package="test.algorithms",
            metadata_file=temp_metadata_file,
            enable_parallel_discovery=False
        )
        
        assert registry.algorithms_package == "test.algorithms"
        assert not registry.discovery_completed
        assert registry.enable_parallel_discovery is False
        assert len(registry.algorithm_classes) == 0
    
    def test_discovery_state_management(self):
        """Test discovery state management"""
        registry = AlgorithmRegistry(enable_parallel_discovery=False)
        
        # Initially not completed
        assert not registry.discovery_completed
        
        # Mock successful discovery
        with patch.object(registry, '_get_package_path') as mock_path, \
             patch.object(registry, '_sequential_discovery') as mock_discovery:
            
            mock_path.return_value = Path("/fake/path")
            mock_discovery.return_value = {
                'discovered_algorithms': ['test_alg'],
                'scanned_modules': 1,
                'discovery_errors': 0
            }
            
            result = registry.discover_algorithms()
            
            assert registry.discovery_completed
            assert 'total_algorithms' in result
            assert 'discovery_time' in result
    
    def test_algorithm_validation(self):
        """Test algorithm validation"""
        registry = AlgorithmRegistry()
        
        # Valid algorithm class
        class ValidAlgorithm(BaseOptimizer):
            def __init__(self, param_space, objective_function, **kwargs):
                super().__init__(param_space, objective_function, **kwargs)
            
            def optimize(self, n_iterations=100, **kwargs):
                return OptimizationResult(
                    best_parameters={},
                    best_objective_value=0.0,
                    iterations=n_iterations,
                    convergence_status='converged'
                )
        
        # Should pass validation
        assert registry.validate_algorithm(ValidAlgorithm)
        
        # Invalid algorithm - doesn't inherit from BaseOptimizer
        class InvalidAlgorithm:
            pass
        
        with pytest.raises(AlgorithmValidationError):
            registry.validate_algorithm(InvalidAlgorithm)
        
        # Invalid algorithm - missing optimize method
        class IncompleteAlgorithm(BaseOptimizer):
            def __init__(self, param_space, objective_function, **kwargs):
                super().__init__(param_space, objective_function, **kwargs)
        
        with pytest.raises(AlgorithmValidationError):
            registry.validate_algorithm(IncompleteAlgorithm)
    
    def test_algorithm_name_generation(self):
        """Test algorithm name generation"""
        registry = AlgorithmRegistry()
        
        # Test name normalization
        assert registry._generate_algorithm_name("MyOptimizer", "classical") == "my"
        assert registry._generate_algorithm_name("GeneticAlgorithm", "evolutionary") == "genetic_algorithm"
        assert registry._generate_algorithm_name("ParticleSwarmOptimization", "swarm") == "particle_swarm_optimization"
        
        # Test suffix removal
        assert registry._generate_algorithm_name("RandomSearchOptimizer", "classical") == "random_search"
        assert registry._generate_algorithm_name("BayesianAlgorithm", "classical") == "bayesian"
    
    def test_algorithm_retrieval(self):
        """Test algorithm retrieval"""
        registry = AlgorithmRegistry()
        
        # Mock a discovered algorithm
        class MockAlgorithm(BaseOptimizer):
            def __init__(self, param_space, objective_function, **kwargs):
                super().__init__(param_space, objective_function, **kwargs)
            
            def optimize(self, **kwargs):
                return OptimizationResult(
                    best_parameters={}, best_objective_value=0.0,
                    iterations=1, convergence_status='converged'
                )
        
        registry.algorithm_classes['mock_algorithm'] = MockAlgorithm
        registry.discovery_completed = True
        
        # Test successful retrieval
        algorithm = registry.get_algorithm('mock_algorithm')
        assert isinstance(algorithm, MockAlgorithm)
        
        # Test non-existent algorithm
        with pytest.raises(ValueError, match="Algorithm 'nonexistent' not found"):
            registry.get_algorithm('nonexistent')
    
    def test_algorithm_caching(self):
        """Test algorithm instance caching"""
        registry = AlgorithmRegistry(cache_algorithms=True)
        
        class CacheTestAlgorithm(BaseOptimizer):
            def __init__(self, param_space, objective_function, test_param=None, **kwargs):
                super().__init__(param_space, objective_function, **kwargs)
                self.test_param = test_param
            
            def optimize(self, **kwargs):
                return OptimizationResult(
                    best_parameters={}, best_objective_value=0.0,
                    iterations=1, convergence_status='converged'
                )
        
        registry.algorithm_classes['cache_test'] = CacheTestAlgorithm
        registry.discovery_completed = True
        
        # First retrieval
        alg1 = registry.get_algorithm('cache_test', test_param='value1')
        
        # Second retrieval with same params (should be cached)
        alg2 = registry.get_algorithm('cache_test', test_param='value1')
        
        # Third retrieval with different params (should be new instance)
        alg3 = registry.get_algorithm('cache_test', test_param='value2')
        
        assert alg1 is alg2  # Same instance (cached)
        assert alg1 is not alg3  # Different instance
        assert alg3.test_param == 'value2'
    
    def test_algorithm_listing_and_filtering(self):
        """Test algorithm listing with filtering"""
        registry = AlgorithmRegistry()
        registry.discovery_completed = True
        
        # Mock some algorithms with metadata
        test_metadata = {
            'classical_alg': AlgorithmMetadata(
                name='classical_alg',
                category=AlgorithmCategory.CLASSICAL,
                description='Test classical algorithm',
                capabilities=AlgorithmCapabilities(
                    problem_types={ProblemType.CONTINUOUS},
                    supports_gpu=False,
                    supports_parallel=True
                )
            ),
            'gpu_alg': AlgorithmMetadata(
                name='gpu_alg', 
                category=AlgorithmCategory.SWARM,
                description='Test GPU algorithm',
                capabilities=AlgorithmCapabilities(
                    problem_types={ProblemType.CONTINUOUS, ProblemType.DISCRETE},
                    supports_gpu=True,
                    supports_parallel=True
                )
            )
        }
        
        # Mock metadata manager
        registry.metadata_manager.metadata_store = test_metadata
        registry.algorithm_classes = {name: Mock for name in test_metadata.keys()}
        
        # Test unfiltered listing
        all_algorithms = registry.list_algorithms()
        assert len(all_algorithms) == 2
        
        # Test category filtering
        classical_algs = registry.list_algorithms(category=AlgorithmCategory.CLASSICAL)
        assert classical_algs == ['classical_alg']
        
        # Test GPU filtering
        gpu_algs = registry.list_algorithms(supports_gpu=True)
        assert gpu_algs == ['gpu_alg']
        
        # Test parallel filtering
        parallel_algs = registry.list_algorithms(supports_parallel=True)
        assert len(parallel_algs) == 2
    
    def test_algorithm_recommendation(self):
        """Test algorithm recommendation system"""
        registry = AlgorithmRegistry()
        registry.discovery_completed = True
        
        # Mock metadata manager recommendation
        mock_recommendations = [
            ('best_algorithm', 0.9),
            ('good_algorithm', 0.7),
            ('okay_algorithm', 0.5)
        ]
        
        registry.metadata_manager.recommend_algorithms = Mock(return_value=mock_recommendations)
        registry.algorithm_classes = {name: Mock for name, _ in mock_recommendations}
        
        problem_characteristics = {
            'dimensions': 5,
            'problem_type': 'continuous'
        }
        
        recommendations = registry.recommend_algorithms(problem_characteristics)
        
        assert len(recommendations) == 3
        assert recommendations[0][0] == 'best_algorithm'
        assert recommendations[0][1] == 0.9
        
        # Test recommendation limit
        limited_recommendations = registry.recommend_algorithms(
            problem_characteristics, max_recommendations=2
        )
        assert len(limited_recommendations) == 2
    
    def test_algorithm_info_retrieval(self):
        """Test algorithm information retrieval"""
        registry = AlgorithmRegistry()
        registry.discovery_completed = True
        
        # Mock algorithm class
        class InfoTestAlgorithm(BaseOptimizer):
            """Test algorithm for info retrieval"""
            def optimize(self, **kwargs):
                pass
        
        registry.algorithm_classes['info_test'] = InfoTestAlgorithm
        registry.algorithm_modules['info_test'] = 'test.module'
        
        # Mock metadata
        test_metadata = AlgorithmMetadata(
            name='info_test',
            category=AlgorithmCategory.CLASSICAL,
            description='Test algorithm',
            version='1.0.0',
            capabilities=AlgorithmCapabilities(supports_gpu=True),
            resource_requirements=ResourceRequirements(memory_mb=200)
        )
        registry.metadata_manager.metadata_store['info_test'] = test_metadata
        
        info = registry.get_algorithm_info('info_test')
        
        assert info['name'] == 'info_test'
        assert info['class_name'] == 'InfoTestAlgorithm'
        assert info['module'] == 'test.module'
        assert info['category'] == 'classical'
        assert info['capabilities']['supports_gpu'] is True
        assert 'docstring' in info
    
    def test_registry_statistics(self):
        """Test registry statistics generation"""
        registry = AlgorithmRegistry()
        registry.discovery_completed = True
        
        # Mock some data
        registry.algorithm_classes = {
            'classical1': Mock,
            'classical2': Mock,
            'evolutionary1': Mock
        }
        
        registry.usage_counts = {
            'classical1': 5,
            'classical2': 3,
            'evolutionary1': 7
        }
        
        registry.load_times = {
            'classical1': 0.1,
            'classical2': 0.2,
            'evolutionary1': 0.15
        }
        
        # Mock metadata for category breakdown
        mock_metadata = {
            'classical1': Mock(category=AlgorithmCategory.CLASSICAL),
            'classical2': Mock(category=AlgorithmCategory.CLASSICAL),
            'evolutionary1': Mock(category=AlgorithmCategory.EVOLUTIONARY)
        }
        registry.metadata_manager.metadata_store = mock_metadata
        
        stats = registry.get_registry_statistics()
        
        assert stats['discovery_status']['total_algorithms'] == 3
        assert stats['performance_stats']['total_usage'] == 15
        assert len(stats['performance_stats']['popular_algorithms']) <= 5
        assert 'category_breakdown' in stats
    
    def test_cache_management(self):
        """Test cache clearing and management"""
        registry = AlgorithmRegistry(cache_algorithms=True)
        
        # Add some cached instances
        registry.algorithm_instances['test1'] = Mock()
        registry.algorithm_instances['test2'] = Mock()
        
        assert len(registry.algorithm_instances) == 2
        
        # Clear cache
        registry.clear_cache()
        assert len(registry.algorithm_instances) == 0
    
    def test_algorithm_reloading(self):
        """Test algorithm reloading functionality"""
        registry = AlgorithmRegistry()
        
        # Mock algorithm data
        registry.algorithm_modules['test_alg'] = 'test.module'
        registry.algorithm_classes['test_alg'] = Mock
        registry.algorithm_instances['test_alg_123'] = Mock()
        
        # Test error case - algorithm not found
        with pytest.raises(ValueError, match="Algorithm 'nonexistent' not found"):
            registry.reload_algorithm('nonexistent')
    
    def test_parallel_vs_sequential_discovery(self):
        """Test parallel vs sequential discovery modes"""
        # Test sequential discovery
        seq_registry = AlgorithmRegistry(enable_parallel_discovery=False)
        assert seq_registry.enable_parallel_discovery is False
        
        # Test parallel discovery
        par_registry = AlgorithmRegistry(enable_parallel_discovery=True)
        assert par_registry.enable_parallel_discovery is True
    
    def test_discovery_error_handling(self):
        """Test discovery error handling"""
        registry = AlgorithmRegistry()
        
        # Mock package path error
        with patch.object(registry, '_get_package_path') as mock_path:
            mock_path.side_effect = ImportError("Package not found")
            
            with pytest.raises(AlgorithmLoadError):
                registry.discover_algorithms()
    
    def test_metadata_integration(self, temp_metadata_file):
        """Test metadata manager integration"""
        registry = AlgorithmRegistry(metadata_file=temp_metadata_file)
        
        # Test metadata manager is created
        assert registry.metadata_manager is not None
        
        # Test metadata file path is set
        assert registry.metadata_manager.metadata_file == Path(temp_metadata_file)


class TestAlgorithmRegistryPerformanceTracking:
    """Test performance tracking features"""
    
    def test_load_time_tracking(self):
        """Test algorithm load time tracking"""
        registry = AlgorithmRegistry()
        
        class SlowLoadAlgorithm(BaseOptimizer):
            def __init__(self, param_space, objective_function, **kwargs):
                import time
                time.sleep(0.01)  # Simulate slow loading
                super().__init__(param_space, objective_function, **kwargs)
            
            def optimize(self, **kwargs):
                return OptimizationResult(
                    best_parameters={}, best_objective_value=0.0,
                    iterations=1, convergence_status='converged'
                )
        
        registry.algorithm_classes['slow_alg'] = SlowLoadAlgorithm
        registry.discovery_completed = True
        
        # Get algorithm (should track load time)
        registry.get_algorithm('slow_alg')
        
        assert 'slow_alg' in registry.load_times
        assert registry.load_times['slow_alg'] > 0
    
    def test_usage_counting(self):
        """Test algorithm usage counting"""
        registry = AlgorithmRegistry()
        
        class CountTestAlgorithm(BaseOptimizer):
            def optimize(self, **kwargs):
                return OptimizationResult(
                    best_parameters={}, best_objective_value=0.0,
                    iterations=1, convergence_status='converged'
                )
        
        registry.algorithm_classes['count_test'] = CountTestAlgorithm
        registry.discovery_completed = True
        
        # Use algorithm multiple times
        registry.get_algorithm('count_test')
        registry.get_algorithm('count_test')
        registry.get_algorithm('count_test')
        
        assert registry.usage_counts['count_test'] == 3


class TestAlgorithmRegistryEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_algorithms_package(self):
        """Test behavior with empty algorithms package"""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module found")
            
            registry = AlgorithmRegistry()
            
            with pytest.raises(AlgorithmLoadError):
                registry.discover_algorithms()
    
    def test_force_rediscovery(self):
        """Test forced rediscovery of algorithms"""
        registry = AlgorithmRegistry()
        
        # Mock initial discovery
        with patch.object(registry, '_get_package_path') as mock_path, \
             patch.object(registry, '_sequential_discovery') as mock_discovery:
            
            mock_path.return_value = Path("/fake/path")
            mock_discovery.return_value = {
                'discovered_algorithms': ['alg1'],
                'scanned_modules': 1,
                'discovery_errors': 0
            }
            
            # First discovery
            registry.discover_algorithms()
            assert registry.discovery_completed
            
            # Forced rediscovery
            registry.discover_algorithms(force_rediscovery=True)
            
            # Should have been called twice
            assert mock_discovery.call_count == 2
    
    def test_algorithm_with_missing_metadata(self):
        """Test algorithm handling when metadata is missing"""
        registry = AlgorithmRegistry()
        
        class NoMetadataAlgorithm(BaseOptimizer):
            def optimize(self, **kwargs):
                return OptimizationResult(
                    best_parameters={}, best_objective_value=0.0,
                    iterations=1, convergence_status='converged'
                )
        
        registry.algorithm_classes['no_metadata'] = NoMetadataAlgorithm
        registry.algorithm_modules['no_metadata'] = 'test.module.classical'
        registry.discovery_completed = True
        
        # Should create default metadata
        registry._create_default_metadata()
        
        # Check that default metadata was created
        info = registry.get_algorithm_info('no_metadata')
        assert 'category' in info
        assert info['category'] == 'classical'  # Inferred from module path
    
    def test_concurrent_access(self):
        """Test thread safety of registry operations"""
        import threading
        import time
        
        registry = AlgorithmRegistry()
        registry.discovery_completed = True
        
        class ThreadTestAlgorithm(BaseOptimizer):
            def optimize(self, **kwargs):
                return OptimizationResult(
                    best_parameters={}, best_objective_value=0.0,
                    iterations=1, convergence_status='converged'
                )
        
        registry.algorithm_classes['thread_test'] = ThreadTestAlgorithm
        
        results = []
        errors = []
        
        def worker():
            try:
                alg = registry.get_algorithm('thread_test')
                results.append(alg)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == 10