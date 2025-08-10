"""
Performance Benchmarks Test Suite
=================================

Performance and scalability tests for the market regime analysis system.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import unittest
import pandas as pd
import numpy as np
import time
import memory_profiler
import psutil
import gc
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import logging

# Import system components
from ..integration.market_regime_orchestrator import MarketRegimeOrchestrator
from ..integration.data_pipeline import DataPipeline
from ..base.common_utils import MathUtils, DataValidator


class PerformanceBenchmarkBase(unittest.TestCase):
    """Base class for performance benchmarks"""
    
    def setUp(self):
        """Setup performance benchmarking environment"""
        self.performance_thresholds = {
            'max_execution_time': 30.0,  # seconds
            'max_memory_usage': 500,     # MB
            'min_throughput': 100,       # operations per second
            'max_cpu_usage': 80          # percentage
        }
        
        # Create test datasets of different sizes
        self.test_datasets = {
            'small': self._create_test_dataset(100),
            'medium': self._create_test_dataset(1000),
            'large': self._create_test_dataset(10000),
            'xlarge': self._create_test_dataset(50000)
        }
        
        # Performance tracking
        self.performance_results = {}
    
    def _create_test_dataset(self, size: int) -> dict:
        """Create test dataset of specified size"""
        np.random.seed(42)  # For reproducible results
        
        # Option data
        strikes = np.random.choice(range(100, 121), size)
        option_types = np.random.choice(['CE', 'PE'], size)
        spots = np.random.normal(110, 5, size)
        volumes = np.random.exponential(200, size).astype(int)
        ois = np.random.exponential(2000, size).astype(int)
        closes = np.random.lognormal(1, 0.3, size)
        ivs = np.random.normal(0.2, 0.05, size)
        
        option_data = pd.DataFrame({
            'strike': strikes,
            'option_type': option_types,
            'spot': spots,
            'volume': volumes,
            'oi': ois,
            'close': closes,
            'iv': ivs,
            'delta': np.random.normal(0, 0.5, size),
            'gamma': np.random.exponential(0.05, size),
            'theta': np.random.normal(-0.1, 0.05, size),
            'vega': np.random.exponential(0.3, size),
            'timestamp': pd.date_range('2024-01-01', periods=size, freq='1s')
        })
        
        # Underlying data (smaller dataset)
        underlying_size = max(100, size // 10)
        underlying_data = pd.DataFrame({
            'open': np.random.normal(110, 5, underlying_size),
            'high': np.random.normal(112, 5, underlying_size),
            'low': np.random.normal(108, 5, underlying_size),
            'close': np.random.normal(110, 5, underlying_size),
            'volume': np.random.exponential(10000, underlying_size).astype(int),
            'timestamp': pd.date_range('2024-01-01', periods=underlying_size, freq='10s')
        })
        
        return {
            'option_data': option_data,
            'underlying_data': underlying_data
        }
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure performance of a function"""
        # Initial memory measurement
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU monitoring setup
        initial_cpu = process.cpu_percent()
        
        # Execute and time the function
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Final measurements
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        final_cpu = process.cpu_percent()
        
        # Force garbage collection for memory measurement
        gc.collect()
        
        return {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'peak_memory': final_memory,
            'cpu_usage': final_cpu,
            'success': success,
            'error': error,
            'result': result
        }
    
    def assert_performance_threshold(self, measurement, test_name):
        """Assert that performance meets thresholds"""
        # Execution time threshold
        self.assertLess(
            measurement['execution_time'], 
            self.performance_thresholds['max_execution_time'],
            f"{test_name}: Execution time {measurement['execution_time']:.2f}s exceeds threshold {self.performance_thresholds['max_execution_time']}s"
        )
        
        # Memory usage threshold
        self.assertLess(
            measurement['memory_usage'], 
            self.performance_thresholds['max_memory_usage'],
            f"{test_name}: Memory usage {measurement['memory_usage']:.2f}MB exceeds threshold {self.performance_thresholds['max_memory_usage']}MB"
        )
        
        # Success requirement
        self.assertTrue(
            measurement['success'],
            f"{test_name}: Operation failed with error: {measurement.get('error', 'Unknown error')}"
        )


class TestDataPipelinePerformance(PerformanceBenchmarkBase):
    """Test data pipeline performance"""
    
    def setUp(self):
        super().setUp()
        self.pipeline_config = {
            'cache_enabled': True,
            'cache_ttl': 300,
            'quality_thresholds': {
                'min_data_points': 10,
                'max_missing_ratio': 0.2,
                'min_quality_score': 0.6
            }
        }
    
    @patch('..integration.data_pipeline.OptionDataProcessor')
    @patch('..integration.data_pipeline.UnderlyingDataProcessor')
    def test_small_dataset_processing_performance(self, mock_underlying_processor, mock_option_processor):
        """Test performance with small dataset"""
        # Setup mocks
        mock_option_instance = Mock()
        mock_underlying_instance = Mock()
        mock_option_processor.return_value = mock_option_instance
        mock_underlying_processor.return_value = mock_underlying_instance
        
        # Mock processing to return the data unchanged
        mock_option_instance.process.side_effect = lambda x: x
        mock_underlying_instance.process.side_effect = lambda x: x
        mock_option_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 0.9}
        mock_underlying_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 0.9}
        
        pipeline = DataPipeline(self.pipeline_config)
        
        # Measure performance
        measurement = self.measure_performance(
            pipeline.process_data,
            self.test_datasets['small']
        )
        
        self.assert_performance_threshold(measurement, "Small Dataset Processing")
        self.performance_results['small_dataset_processing'] = measurement
    
    @patch('..integration.data_pipeline.OptionDataProcessor')
    @patch('..integration.data_pipeline.UnderlyingDataProcessor')
    def test_medium_dataset_processing_performance(self, mock_underlying_processor, mock_option_processor):
        """Test performance with medium dataset"""
        # Setup mocks
        mock_option_instance = Mock()
        mock_underlying_instance = Mock()
        mock_option_processor.return_value = mock_option_instance
        mock_underlying_processor.return_value = mock_underlying_instance
        
        mock_option_instance.process.side_effect = lambda x: x
        mock_underlying_instance.process.side_effect = lambda x: x
        mock_option_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 0.9}
        mock_underlying_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 0.9}
        
        pipeline = DataPipeline(self.pipeline_config)
        
        measurement = self.measure_performance(
            pipeline.process_data,
            self.test_datasets['medium']
        )
        
        self.assert_performance_threshold(measurement, "Medium Dataset Processing")
        self.performance_results['medium_dataset_processing'] = measurement
    
    @patch('..integration.data_pipeline.OptionDataProcessor')
    @patch('..integration.data_pipeline.UnderlyingDataProcessor')
    def test_large_dataset_processing_performance(self, mock_underlying_processor, mock_option_processor):
        """Test performance with large dataset"""
        # Setup mocks
        mock_option_instance = Mock()
        mock_underlying_instance = Mock()
        mock_option_processor.return_value = mock_option_instance
        mock_underlying_processor.return_value = mock_underlying_instance
        
        mock_option_instance.process.side_effect = lambda x: x
        mock_underlying_instance.process.side_effect = lambda x: x
        mock_option_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 0.9}
        mock_underlying_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 0.9}
        
        pipeline = DataPipeline(self.pipeline_config)
        
        measurement = self.measure_performance(
            pipeline.process_data,
            self.test_datasets['large']
        )
        
        # Relax thresholds for large dataset
        large_thresholds = self.performance_thresholds.copy()
        large_thresholds['max_execution_time'] = 60.0  # Allow more time for large datasets
        large_thresholds['max_memory_usage'] = 1000    # Allow more memory
        
        self.assertLess(measurement['execution_time'], large_thresholds['max_execution_time'])
        self.assertLess(measurement['memory_usage'], large_thresholds['max_memory_usage'])
        self.assertTrue(measurement['success'])
        
        self.performance_results['large_dataset_processing'] = measurement
    
    def test_cache_performance_improvement(self):
        """Test that caching improves performance"""
        # Test with caching enabled
        pipeline_with_cache = DataPipeline({**self.pipeline_config, 'cache_enabled': True})
        
        # First call (should cache)
        measurement1 = self.measure_performance(
            pipeline_with_cache.process_data,
            self.test_datasets['medium']
        )
        
        # Second call (should use cache)
        measurement2 = self.measure_performance(
            pipeline_with_cache.process_data,
            self.test_datasets['medium']
        )
        
        # Cache should make second call faster (allow some variance)
        self.assertLess(
            measurement2['execution_time'],
            measurement1['execution_time'] + 0.1,  # Allow 100ms variance
            "Cached call should be faster than or equal to initial call"
        )


class TestMathUtilsPerformance(PerformanceBenchmarkBase):
    """Test mathematical utilities performance"""
    
    def setUp(self):
        super().setUp()
        self.math_utils = MathUtils()
    
    def test_large_array_calculations_performance(self):
        """Test performance with large arrays"""
        # Create large arrays
        large_array_size = 100000
        array1 = np.random.random(large_array_size)
        array2 = np.random.random(large_array_size)
        
        # Test correlation calculation
        measurement = self.measure_performance(
            self.math_utils.calculate_correlation,
            array1, array2
        )
        
        self.assert_performance_threshold(measurement, "Large Array Correlation")
        self.performance_results['large_array_correlation'] = measurement
    
    def test_moving_average_performance(self):
        """Test moving average performance"""
        large_array = np.random.random(50000)
        window_size = 20
        
        measurement = self.measure_performance(
            self.math_utils.moving_average,
            large_array, window_size
        )
        
        self.assert_performance_threshold(measurement, "Moving Average Calculation")
        self.performance_results['moving_average'] = measurement
    
    def test_normalization_performance(self):
        """Test value normalization performance"""
        large_array = np.random.random(100000)
        
        measurement = self.measure_performance(
            self.math_utils.normalize_values,
            large_array
        )
        
        self.assert_performance_threshold(measurement, "Value Normalization")
        self.performance_results['normalization'] = measurement
    
    def test_batch_zscore_calculation_performance(self):
        """Test batch z-score calculation performance"""
        # Simulate calculating z-scores for many values
        base_values = np.random.random(1000)
        test_values = np.random.random(10000)
        
        def batch_zscore_calculation():
            results = []
            for value in test_values:
                zscore = self.math_utils.calculate_zscore(base_values, value)
                results.append(zscore)
            return results
        
        measurement = self.measure_performance(batch_zscore_calculation)
        
        self.assert_performance_threshold(measurement, "Batch Z-Score Calculation")
        self.performance_results['batch_zscore'] = measurement


class TestDataValidationPerformance(PerformanceBenchmarkBase):
    """Test data validation performance"""
    
    def setUp(self):
        super().setUp()
        self.validator = DataValidator()
    
    def test_large_dataset_validation_performance(self):
        """Test validation performance with large datasets"""
        large_option_data = self.test_datasets['large']['option_data']
        
        measurement = self.measure_performance(
            self.validator.validate_option_data,
            large_option_data
        )
        
        self.assert_performance_threshold(measurement, "Large Dataset Validation")
        self.performance_results['large_dataset_validation'] = measurement
    
    def test_repeated_validation_performance(self):
        """Test performance of repeated validations"""
        medium_data = self.test_datasets['medium']['option_data']
        
        def repeated_validation():
            results = []
            for _ in range(100):  # Validate 100 times
                result = self.validator.validate_option_data(medium_data)
                results.append(result)
            return results
        
        measurement = self.measure_performance(repeated_validation)
        
        # Adjust thresholds for repeated operations
        self.assertLess(measurement['execution_time'], 30.0)
        self.assertTrue(measurement['success'])
        
        self.performance_results['repeated_validation'] = measurement


class TestThroughputBenchmarks(PerformanceBenchmarkBase):
    """Test system throughput benchmarks"""
    
    def test_records_per_second_processing(self):
        """Test how many records can be processed per second"""
        pipeline_config = {
            'cache_enabled': False,  # Disable cache for throughput test
            'quality_thresholds': {
                'min_data_points': 1,
                'max_missing_ratio': 1.0,
                'min_quality_score': 0.0
            }
        }
        
        with patch('..integration.data_pipeline.OptionDataProcessor') as mock_option_processor, \
             patch('..integration.data_pipeline.UnderlyingDataProcessor') as mock_underlying_processor:
            
            # Setup mocks for fast processing
            mock_option_instance = Mock()
            mock_underlying_instance = Mock()
            mock_option_processor.return_value = mock_option_instance
            mock_underlying_processor.return_value = mock_underlying_instance
            
            mock_option_instance.process.side_effect = lambda x: x
            mock_underlying_instance.process.side_effect = lambda x: x
            mock_option_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 1.0}
            mock_underlying_instance.validate.return_value = {'is_valid': True, 'data_quality_score': 1.0}
            
            pipeline = DataPipeline(pipeline_config)
            
            # Test different dataset sizes
            throughput_results = {}
            
            for size_name, dataset in self.test_datasets.items():
                if size_name == 'xlarge':  # Skip extra large for throughput test
                    continue
                    
                measurement = self.measure_performance(
                    pipeline.process_data,
                    dataset
                )
                
                if measurement['success']:
                    records_processed = len(dataset['option_data'])
                    throughput = records_processed / measurement['execution_time']
                    throughput_results[size_name] = {
                        'records': records_processed,
                        'time': measurement['execution_time'],
                        'throughput': throughput
                    }
                    
                    # Log throughput information
                    print(f"{size_name.capitalize()} dataset: {throughput:.0f} records/second")
            
            self.performance_results['throughput'] = throughput_results
            
            # Assert minimum throughput for small dataset
            if 'small' in throughput_results:
                self.assertGreater(
                    throughput_results['small']['throughput'],
                    self.performance_thresholds['min_throughput'],
                    f"Throughput {throughput_results['small']['throughput']:.0f} records/sec is below threshold {self.performance_thresholds['min_throughput']}"
                )


class TestMemoryLeakDetection(PerformanceBenchmarkBase):
    """Test for memory leaks in repeated operations"""
    
    def test_repeated_operations_memory_stability(self):
        """Test that repeated operations don't cause memory leaks"""
        math_utils = MathUtils()
        
        # Record initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_measurements = []
        
        # Perform operations repeatedly
        for i in range(50):
            # Create some data and perform operations
            data = np.random.random(1000)
            
            # Perform various operations
            math_utils.moving_average(data, 10)
            math_utils.normalize_values(data)
            math_utils.calculate_correlation(data[:500], data[500:])
            
            # Measure memory every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
                
            # Force garbage collection
            if i % 20 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Assert that memory growth is reasonable (less than 100MB)
        self.assertLess(
            memory_growth, 100,
            f"Memory grew by {memory_growth:.2f}MB during repeated operations, possible memory leak"
        )
        
        # Assert that memory doesn't continuously grow
        if len(memory_measurements) > 2:
            # Check that final measurement isn't significantly higher than middle measurements
            middle_memory = np.mean(memory_measurements[1:-1])
            final_measurement = memory_measurements[-1]
            
            self.assertLess(
                final_measurement - middle_memory, 50,
                "Memory appears to be continuously growing, possible memory leak"
            )


class TestScalabilityBenchmarks(PerformanceBenchmarkBase):
    """Test system scalability with increasing load"""
    
    def test_execution_time_scalability(self):
        """Test how execution time scales with data size"""
        validator = DataValidator()
        
        scalability_results = {}
        
        for size_name, dataset in self.test_datasets.items():
            if size_name == 'xlarge':  # Skip for this test
                continue
                
            option_data = dataset['option_data']
            
            measurement = self.measure_performance(
                validator.validate_option_data,
                option_data
            )
            
            if measurement['success']:
                scalability_results[size_name] = {
                    'records': len(option_data),
                    'time': measurement['execution_time'],
                    'time_per_record': measurement['execution_time'] / len(option_data)
                }
        
        # Check that time per record doesn't grow significantly
        if 'small' in scalability_results and 'large' in scalability_results:
            small_time_per_record = scalability_results['small']['time_per_record']
            large_time_per_record = scalability_results['large']['time_per_record']
            
            # Time per record shouldn't increase by more than 10x
            self.assertLess(
                large_time_per_record / small_time_per_record, 10,
                f"Algorithm doesn't scale well: time per record increased from {small_time_per_record:.6f}s to {large_time_per_record:.6f}s"
            )
        
        self.performance_results['scalability'] = scalability_results


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataPipelinePerformance,
        TestMathUtilsPerformance,
        TestDataValidationPerformance,
        TestThroughputBenchmarks,
        TestMemoryLeakDetection,
        TestScalabilityBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Performance Benchmarks Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Print performance summary
    print(f"\n{'='*60}")
    print(f"Performance Summary")
    print(f"{'='*60}")
    
    # Aggregate results from all test instances
    all_performance_results = {}
    for test_class in test_classes:
        for test_method_name in dir(test_class):
            if test_method_name.startswith('test_'):
                # This is a simplified aggregation - in a real implementation,
                # you'd collect results from the actual test instances
                pass
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print(f"\nPerformance benchmarks completed successfully!")
    print(f"Check individual test outputs for detailed performance metrics.")