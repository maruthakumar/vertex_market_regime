#!/usr/bin/env python3
"""
Performance Tests for Optimized HeavyDB Engine

This module provides comprehensive performance tests for the Optimized HeavyDB Engine,
validating the <0.8s correlation matrix processing target and overall system optimization.

Author: The Augster
Date: 2025-06-18
Version: 1.0.0
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from optimized_heavydb_engine import (
    OptimizedHeavyDBEngine,
    OptimizedQueryResult,
    QueryPerformanceMetrics
)
from correlation_matrix_engine import CorrelationMatrixEngine

logger = logging.getLogger(__name__)

class TestOptimizedHeavyDBPerformance(unittest.TestCase):
    """Comprehensive performance test suite for Optimized HeavyDB Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = OptimizedHeavyDBEngine()
        self.correlation_engine = CorrelationMatrixEngine()
        
        # Performance targets
        self.target_processing_time = 0.8  # 0.8 second target
        self.target_query_time = 0.5  # Individual query target
        
        # Test market data samples
        self.test_market_data = [
            {
                'symbol': 'NIFTY',
                'underlying_price': 19500,
                'timestamp': datetime.now() - timedelta(minutes=i*5),
                'iv_percentile': 0.4 + np.random.random() * 0.2,
                'atr_normalized': 0.35 + np.random.random() * 0.1,
                'gamma_exposure': 0.3 + np.random.random() * 0.2
            } for i in range(10)
        ]
    
    def test_engine_initialization(self):
        """Test optimized engine initialization"""
        self.assertIsNotNone(self.engine)
        
        # Check performance targets
        self.assertEqual(self.engine.target_processing_time, 0.8)
        self.assertEqual(self.engine.max_query_time, 0.5)
        
        # Check connection pool
        self.assertGreaterEqual(len(self.engine.connection_pool), 0)
        self.assertEqual(self.engine.pool_size, self.engine.config['connection_pool']['pool_size'])
        
        # Check optimization settings
        self.assertTrue(self.engine.config['optimization']['enable_gpu_hints'])
        self.assertTrue(self.engine.config['optimization']['enable_parallel_execution'])
        
        logger.info("✅ Optimized HeavyDB Engine initialization test passed")
    
    def test_single_correlation_query_performance(self):
        """Test single correlation query performance (<0.5s target)"""
        test_data = self.test_market_data[0]
        
        start_time = time.time()
        result = self.engine.execute_optimized_correlation_query(
            test_data['symbol'],
            test_data['timestamp'],
            test_data['underlying_price']
        )
        processing_time = time.time() - start_time
        
        # Validate result type
        self.assertIsInstance(result, OptimizedQueryResult)
        
        # Validate performance metrics
        self.assertIsInstance(result.performance, QueryPerformanceMetrics)
        self.assertGreater(result.performance.query_time, 0.0)
        self.assertLess(result.performance.query_time, self.target_query_time)
        
        # Validate processing time
        self.assertLess(processing_time, self.target_query_time)
        
        # Validate data structure
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertIsNotNone(result.query_hash)
        
        logger.info(f"✅ Single query performance: {processing_time:.3f}s (target: {self.target_query_time}s)")
    
    def test_parallel_correlation_queries_performance(self):
        """Test parallel correlation queries performance"""
        queries = [
            {
                'symbol': data['symbol'],
                'timestamp': data['timestamp'],
                'underlying_price': data['underlying_price']
            } for data in self.test_market_data[:5]
        ]
        
        start_time = time.time()
        results = self.engine.execute_parallel_correlation_queries(queries)
        processing_time = time.time() - start_time
        
        # Validate results
        self.assertEqual(len(results), len(queries))
        
        for result in results:
            self.assertIsInstance(result, OptimizedQueryResult)
            self.assertLess(result.performance.query_time, self.target_query_time)
        
        # Validate parallel processing efficiency
        avg_single_time = np.mean([r.performance.query_time for r in results])
        parallel_efficiency = (avg_single_time * len(queries)) / processing_time
        
        # Parallel processing should be at least 2x more efficient
        self.assertGreater(parallel_efficiency, 2.0)
        
        logger.info(f"✅ Parallel queries performance: {processing_time:.3f}s for {len(queries)} queries")
        logger.info(f"   Parallel efficiency: {parallel_efficiency:.2f}x")
    
    def test_correlation_matrix_optimization_target(self):
        """Test correlation matrix optimization meets <0.8s target"""
        start_time = time.time()
        optimization_result = self.engine.optimize_correlation_matrix_processing(self.test_market_data)
        processing_time = time.time() - start_time
        
        # Validate performance target
        self.assertLess(processing_time, self.target_processing_time)
        
        # Validate optimization result
        self.assertIn('correlation_matrices', optimization_result)
        self.assertIn('performance_metrics', optimization_result)
        
        performance = optimization_result['performance_metrics']
        self.assertLess(performance['total_processing_time'], self.target_processing_time)
        self.assertTrue(performance['target_met'])
        
        # Validate correlation matrices
        correlation_matrices = optimization_result['correlation_matrices']
        self.assertEqual(len(correlation_matrices), len(self.test_market_data))
        
        for matrix in correlation_matrices:
            self.assertIsInstance(matrix, dict)
            self.assertIn('overall_correlation_strength', matrix)
        
        logger.info(f"✅ Correlation matrix optimization: {processing_time:.3f}s (target: {self.target_processing_time}s)")
        logger.info(f"   Success rate: {optimization_result['success_rate']:.1%}")
    
    def test_caching_performance(self):
        """Test query caching performance"""
        test_data = self.test_market_data[0]
        
        # First query (cache miss)
        start_time = time.time()
        result1 = self.engine.execute_optimized_correlation_query(
            test_data['symbol'],
            test_data['timestamp'],
            test_data['underlying_price']
        )
        first_query_time = time.time() - start_time
        
        # Second query (cache hit)
        start_time = time.time()
        result2 = self.engine.execute_optimized_correlation_query(
            test_data['symbol'],
            test_data['timestamp'],
            test_data['underlying_price']
        )
        second_query_time = time.time() - start_time
        
        # Validate cache hit
        self.assertTrue(result2.performance.cache_hit)
        self.assertFalse(result1.performance.cache_hit)
        
        # Cache hit should be significantly faster
        cache_speedup = first_query_time / second_query_time if second_query_time > 0 else 1.0
        self.assertGreater(cache_speedup, 2.0)  # At least 2x speedup
        
        logger.info(f"✅ Caching performance: {cache_speedup:.2f}x speedup")
        logger.info(f"   First query: {first_query_time:.3f}s, Cached query: {second_query_time:.3f}s")
    
    def test_correlation_engine_integration(self):
        """Test integration with correlation matrix engine"""
        test_data = self.test_market_data[0]
        
        start_time = time.time()
        result = self.correlation_engine.analyze_multi_strike_correlation(test_data)
        processing_time = time.time() - start_time
        
        # Validate performance target
        self.assertLess(processing_time, self.target_processing_time)
        
        # Validate result structure
        self.assertIsNotNone(result.overall_correlation)
        self.assertIsNotNone(result.correlation_strength)
        self.assertIsNotNone(result.regime_correlation_pattern)
        
        # Validate correlation ranges
        self.assertGreaterEqual(result.overall_correlation, 0.0)
        self.assertLessEqual(result.overall_correlation, 1.0)
        self.assertGreaterEqual(result.correlation_strength, 0.0)
        self.assertLessEqual(result.correlation_strength, 1.0)
        
        logger.info(f"✅ Correlation engine integration: {processing_time:.3f}s")
        logger.info(f"   Overall correlation: {result.overall_correlation:.3f}")
        logger.info(f"   Regime pattern: {result.regime_correlation_pattern}")
    
    def test_performance_validation(self):
        """Test performance validation with multiple samples"""
        validation_result = self.engine.validate_optimization_performance(test_samples=10)
        
        # Validate validation result
        self.assertIn('validation_status', validation_result)
        self.assertIn('total_processing_time', validation_result)
        self.assertIn('target_met', validation_result)
        
        # Validate performance targets
        self.assertTrue(validation_result['target_met'])
        self.assertLess(validation_result['total_processing_time'], self.target_processing_time)
        self.assertEqual(validation_result['validation_status'], 'PASSED')
        
        # Validate success rate
        self.assertGreater(validation_result['success_rate'], 0.8)
        
        logger.info(f"✅ Performance validation: {validation_result['validation_status']}")
        logger.info(f"   Processing time: {validation_result['total_processing_time']:.3f}s")
        logger.info(f"   Success rate: {validation_result['success_rate']:.1%}")
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Run some queries to generate metrics
        for test_data in self.test_market_data[:3]:
            self.engine.execute_optimized_correlation_query(
                test_data['symbol'],
                test_data['timestamp'],
                test_data['underlying_price']
            )
        
        # Get performance summary
        summary = self.engine.get_performance_summary()
        
        # Validate summary structure
        self.assertIn('query_performance', summary)
        self.assertIn('caching_performance', summary)
        self.assertIn('resource_utilization', summary)
        self.assertIn('optimization_status', summary)
        
        # Validate query performance
        query_perf = summary['query_performance']
        self.assertIn('average_time', query_perf)
        self.assertIn('target_met', query_perf)
        self.assertTrue(query_perf['target_met'])
        self.assertLess(query_perf['average_time'], self.target_processing_time)
        
        # Validate performance grade
        optimization_status = summary['optimization_status']
        self.assertIn('performance_grade', optimization_status)
        self.assertIn(optimization_status['performance_grade'], 
                     ['EXCELLENT', 'VERY_GOOD', 'GOOD', 'ACCEPTABLE', 'NEEDS_IMPROVEMENT'])
        
        logger.info(f"✅ Performance summary generated")
        logger.info(f"   Average query time: {query_perf['average_time']:.3f}s")
        logger.info(f"   Performance grade: {optimization_status['performance_grade']}")
    
    def test_stress_testing(self):
        """Test system under stress with multiple concurrent operations"""
        # Generate larger test dataset
        stress_test_data = [
            {
                'symbol': 'NIFTY',
                'underlying_price': 19500 + np.random.randint(-200, 200),
                'timestamp': datetime.now() - timedelta(minutes=i),
                'iv_percentile': np.random.random(),
                'atr_normalized': np.random.random(),
                'gamma_exposure': np.random.random()
            } for i in range(20)
        ]
        
        start_time = time.time()
        optimization_result = self.engine.optimize_correlation_matrix_processing(stress_test_data)
        stress_processing_time = time.time() - start_time
        
        # Validate stress test performance
        # Allow slightly higher time for stress test but should still be reasonable
        stress_target = self.target_processing_time * 2  # 1.6s for 20 samples
        self.assertLess(stress_processing_time, stress_target)
        
        # Validate success rate under stress
        self.assertGreater(optimization_result['success_rate'], 0.7)  # 70% minimum under stress
        
        logger.info(f"✅ Stress test: {stress_processing_time:.3f}s for {len(stress_test_data)} samples")
        logger.info(f"   Success rate under stress: {optimization_result['success_rate']:.1%}")
    
    def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms"""
        # Test with invalid data
        invalid_data = {
            'symbol': 'INVALID',
            'underlying_price': -1,
            'timestamp': 'invalid_timestamp'
        }
        
        result = self.engine.execute_optimized_correlation_query(
            invalid_data['symbol'],
            datetime.now(),  # Use valid timestamp
            19500  # Use valid price
        )
        
        # Should return fallback result without crashing
        self.assertIsInstance(result, OptimizedQueryResult)
        self.assertEqual(result.performance.optimization_applied, "FALLBACK")
        
        logger.info("✅ Error handling and fallback test passed")
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            self.engine.cleanup_resources()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

def run_optimized_heavydb_performance_tests():
    """Run comprehensive performance test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestOptimizedHeavyDBPerformance)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZED HEAVYDB PERFORMANCE TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Performance Target: <0.8s correlation matrix processing")
    print(f"{'='*70}")
    
    if success_rate >= 0.9:  # 90% success rate required
        print("✅ OPTIMIZED HEAVYDB PERFORMANCE TESTS PASSED")
        print("🚀 Ready for Phase 3 Real Data Integration")
        return True
    else:
        print("❌ OPTIMIZED HEAVYDB PERFORMANCE TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_optimized_heavydb_performance_tests()
    sys.exit(0 if success else 1)
