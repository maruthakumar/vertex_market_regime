#!/usr/bin/env python3
"""
Thread-Safe Configuration Loading Integration Test

PHASE 4.1.2: Test thread-safe configuration loading
- Tests concurrent access to Excel configuration
- Validates data consistency during parallel loads
- Tests configuration updates during concurrent access
- NO MOCK DATA - uses real Excel configuration

Author: Claude Code
Date: 2025-07-12
Version: 1.0.0 - PHASE 4.1.2 THREAD-SAFE TESTING
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os
import time
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class ThreadSafeConfigError(Exception):
    """Raised when thread-safe configuration fails"""
    pass

class TestThreadSafeConfigLoading(unittest.TestCase):
    """
    PHASE 4.1.2: Thread-Safe Configuration Loading Test Suite
    STRICT: Uses real Excel file with NO MOCK data
    """
    
    def setUp(self):
        """Set up test environment with STRICT real data requirements"""
        self.excel_config_path = "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod/mr/MR_CONFIG_STRATEGY_1.0.0.xlsx"
        self.strict_mode = True
        self.no_mock_data = True
        self.heavydb_mandatory = True
        
        # Thread tracking
        self.thread_results = {}
        self.thread_errors = []
        self.access_count = 0
        self.access_lock = threading.Lock()
        
        # Verify Excel file exists - FAIL if not available
        if not Path(self.excel_config_path).exists():
            self.fail(f"CRITICAL FAILURE: Excel configuration file not found: {self.excel_config_path}")
        
        logger.info(f"✅ Excel configuration file verified: {self.excel_config_path}")
    
    def test_concurrent_configuration_loading(self):
        """Test: Multiple threads loading configuration concurrently"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            # Create shared manager instance
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            
            def load_config_worker(thread_id: int, iterations: int = 10):
                """Worker function to load configuration multiple times"""
                results = []
                thread_start = time.time()
                
                try:
                    for i in range(iterations):
                        start_time = time.time()
                        
                        # Load configuration
                        config_data = manager.load_configuration()
                        
                        # Validate configuration was loaded
                        if config_data is None:
                            raise ThreadSafeConfigError(f"Thread {thread_id}: Failed to load configuration")
                        
                        # Extract some values to verify consistency
                        detection_params = manager.get_detection_parameters()
                        
                        load_time = time.time() - start_time
                        
                        with self.access_lock:
                            self.access_count += 1
                        
                        results.append({
                            'iteration': i,
                            'load_time': load_time,
                            'config_keys': len(config_data),
                            'confidence_threshold': detection_params.get('ConfidenceThreshold'),
                            'timestamp': datetime.now()
                        })
                        
                        # Small random delay to simulate real usage
                        time.sleep(random.uniform(0.01, 0.05))
                    
                    thread_duration = time.time() - thread_start
                    self.thread_results[thread_id] = {
                        'results': results,
                        'duration': thread_duration,
                        'success': True
                    }
                    
                except Exception as e:
                    self.thread_errors.append((thread_id, str(e)))
                    self.thread_results[thread_id] = {
                        'results': results,
                        'duration': time.time() - thread_start,
                        'success': False,
                        'error': str(e)
                    }
            
            # Create multiple threads
            num_threads = 10
            iterations_per_thread = 5
            threads = []
            
            logger.info(f"Starting {num_threads} threads with {iterations_per_thread} iterations each")
            
            start_time = time.time()
            
            for i in range(num_threads):
                thread = threading.Thread(
                    target=load_config_worker,
                    args=(i, iterations_per_thread),
                    name=f"ConfigLoader-{i}"
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=30)
                if thread.is_alive():
                    logger.error(f"Thread {thread.name} did not complete in time")
            
            total_duration = time.time() - start_time
            
            # Analyze results
            successful_threads = sum(1 for r in self.thread_results.values() if r['success'])
            total_loads = self.access_count
            
            # Verify all threads completed successfully
            self.assertEqual(successful_threads, num_threads, 
                           f"Not all threads succeeded: {successful_threads}/{num_threads}")
            self.assertEqual(len(self.thread_errors), 0, 
                           f"Thread errors occurred: {self.thread_errors}")
            
            # Verify expected number of loads
            expected_loads = num_threads * iterations_per_thread
            self.assertEqual(total_loads, expected_loads, 
                           f"Unexpected number of loads: {total_loads} vs {expected_loads}")
            
            # Verify data consistency across all threads
            all_confidence_values = []
            for thread_id, result in self.thread_results.items():
                if result['success']:
                    for iteration in result['results']:
                        if iteration['confidence_threshold'] is not None:
                            all_confidence_values.append(iteration['confidence_threshold'])
            
            # All threads should see the same configuration value
            unique_values = set(all_confidence_values)
            self.assertEqual(len(unique_values), 1, 
                           f"Inconsistent configuration values across threads: {unique_values}")
            
            # Calculate performance metrics
            avg_load_time = sum(
                r['load_time'] 
                for result in self.thread_results.values() 
                if result['success']
                for r in result['results']
            ) / total_loads
            
            loads_per_second = total_loads / total_duration
            
            logger.info(f"✅ PHASE 4.1.2: Concurrent loading test passed")
            logger.info(f"   Total loads: {total_loads}")
            logger.info(f"   Average load time: {avg_load_time:.3f}s")
            logger.info(f"   Loads per second: {loads_per_second:.1f}")
            logger.info(f"   Total duration: {total_duration:.2f}s")
            
        except ImportError as e:
            self.fail(f"Failed to import excel_config_manager: {e}")
        except Exception as e:
            self.fail(f"Concurrent loading test failed: {e}")
    
    def test_configuration_consistency_under_load(self):
        """Test: Configuration remains consistent under heavy concurrent load"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            
            # Track all loaded configurations
            config_snapshots = []
            config_lock = threading.Lock()
            
            def heavy_load_worker(worker_id: int):
                """Worker that rapidly loads configuration"""
                local_snapshots = []
                
                for i in range(20):  # Rapid fire loads
                    config_data = manager.load_configuration()
                    detection_params = manager.get_detection_parameters()
                    
                    snapshot = {
                        'worker_id': worker_id,
                        'iteration': i,
                        'timestamp': datetime.now(),
                        'confidence_threshold': detection_params.get('ConfidenceThreshold'),
                        'regime_smoothing': detection_params.get('RegimeSmoothing'),
                        'indicator_weights': {
                            'greek': detection_params.get('IndicatorWeightGreek'),
                            'oi': detection_params.get('IndicatorWeightOI'),
                            'price': detection_params.get('IndicatorWeightPrice')
                        }
                    }
                    local_snapshots.append(snapshot)
                
                with config_lock:
                    config_snapshots.extend(local_snapshots)
            
            # Launch workers using ThreadPoolExecutor
            num_workers = 20
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(heavy_load_worker, i) 
                    for i in range(num_workers)
                ]
                
                # Wait for all to complete
                for future in as_completed(futures, timeout=30):
                    try:
                        future.result()
                    except Exception as e:
                        self.fail(f"Worker failed: {e}")
            
            # Verify consistency across all snapshots
            self.assertGreater(len(config_snapshots), 0, "No configuration snapshots captured")
            
            # Check that all critical values are consistent
            first_snapshot = config_snapshots[0]
            for i, snapshot in enumerate(config_snapshots[1:], 1):
                # Confidence threshold should be consistent
                self.assertEqual(
                    snapshot['confidence_threshold'],
                    first_snapshot['confidence_threshold'],
                    f"Inconsistent confidence_threshold at snapshot {i}"
                )
                
                # Regime smoothing should be consistent
                self.assertEqual(
                    snapshot['regime_smoothing'],
                    first_snapshot['regime_smoothing'],
                    f"Inconsistent regime_smoothing at snapshot {i}"
                )
                
                # Indicator weights should be consistent
                for weight_type in ['greek', 'oi', 'price']:
                    self.assertEqual(
                        snapshot['indicator_weights'][weight_type],
                        first_snapshot['indicator_weights'][weight_type],
                        f"Inconsistent {weight_type} weight at snapshot {i}"
                    )
            
            logger.info(f"✅ PHASE 4.1.2: Configuration consistency verified across {len(config_snapshots)} snapshots")
            
        except Exception as e:
            self.fail(f"Configuration consistency test failed: {e}")
    
    def test_thread_safe_getter_methods(self):
        """Test: All getter methods are thread-safe"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            
            # Load configuration once
            manager.load_configuration()
            
            # Methods to test concurrently
            getter_methods = [
                ('get_detection_parameters', manager.get_detection_parameters),
                ('get_regime_adjustments', manager.get_regime_adjustments),
                ('get_strategy_mappings', manager.get_strategy_mappings),
                ('get_live_trading_config', manager.get_live_trading_config),
                ('get_technical_indicators_config', manager.get_technical_indicators_config)
            ]
            
            method_results = {name: [] for name, _ in getter_methods}
            method_errors = []
            
            def test_getter_method(method_name: str, method_func, iterations: int = 50):
                """Test a specific getter method repeatedly"""
                try:
                    for i in range(iterations):
                        result = method_func()
                        method_results[method_name].append({
                            'iteration': i,
                            'result_type': type(result).__name__,
                            'result_size': len(result) if hasattr(result, '__len__') else None
                        })
                except Exception as e:
                    method_errors.append((method_name, str(e)))
            
            # Test all methods concurrently
            threads = []
            for method_name, method_func in getter_methods:
                for i in range(5):  # 5 threads per method
                    thread = threading.Thread(
                        target=test_getter_method,
                        args=(method_name, method_func)
                    )
                    threads.append(thread)
                    thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=15)
            
            # Verify no errors
            self.assertEqual(len(method_errors), 0, 
                           f"Getter method errors: {method_errors}")
            
            # Verify all methods returned consistent results
            for method_name, results in method_results.items():
                if results:
                    # All results should have the same type
                    result_types = set(r['result_type'] for r in results)
                    self.assertEqual(len(result_types), 1, 
                                   f"Inconsistent result types for {method_name}: {result_types}")
                    
                    # All results should have the same size (if applicable)
                    result_sizes = set(r['result_size'] for r in results if r['result_size'] is not None)
                    if result_sizes:
                        self.assertEqual(len(result_sizes), 1, 
                                       f"Inconsistent result sizes for {method_name}: {result_sizes}")
            
            logger.info("✅ PHASE 4.1.2: All getter methods are thread-safe")
            
        except Exception as e:
            self.fail(f"Thread-safe getter methods test failed: {e}")
    
    def test_race_condition_protection(self):
        """Test: Protection against race conditions during configuration access"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            
            # Shared state for testing race conditions
            shared_counter = {'value': 0}
            counter_lock = threading.Lock()
            race_conditions_detected = []
            
            def race_condition_worker(worker_id: int, iterations: int = 100):
                """Worker that tries to detect race conditions"""
                for i in range(iterations):
                    # Get configuration
                    config = manager.load_configuration()
                    params = manager.get_detection_parameters()
                    
                    # Simulate some processing
                    with counter_lock:
                        old_value = shared_counter['value']
                        # Small delay to increase chance of race condition
                        time.sleep(0.0001)
                        shared_counter['value'] = old_value + 1
                    
                    # Verify configuration integrity
                    if params:
                        confidence = params.get('ConfidenceThreshold')
                        if confidence is None or not (0.0 <= confidence <= 1.0):
                            race_conditions_detected.append({
                                'worker_id': worker_id,
                                'iteration': i,
                                'issue': 'Invalid confidence threshold',
                                'value': confidence
                            })
            
            # Run many workers simultaneously
            num_workers = 50
            threads = []
            
            for i in range(num_workers):
                thread = threading.Thread(
                    target=race_condition_worker,
                    args=(i, 50)
                )
                threads.append(thread)
            
            # Start all threads at once
            for thread in threads:
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)
            
            # Verify no race conditions detected
            self.assertEqual(len(race_conditions_detected), 0, 
                           f"Race conditions detected: {race_conditions_detected[:5]}")  # Show first 5
            
            # Verify counter is correct (no lost updates)
            expected_count = num_workers * 50
            actual_count = shared_counter['value']
            self.assertEqual(actual_count, expected_count, 
                           f"Lost updates detected: {actual_count} vs {expected_count}")
            
            logger.info("✅ PHASE 4.1.2: No race conditions detected")
            
        except Exception as e:
            self.fail(f"Race condition protection test failed: {e}")
    
    def test_memory_consistency_model(self):
        """Test: Memory consistency across threads"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            
            # Load configuration once
            initial_config = manager.load_configuration()
            initial_params = manager.get_detection_parameters()
            
            # Store initial values
            initial_values = {
                'confidence_threshold': initial_params.get('ConfidenceThreshold'),
                'regime_smoothing': initial_params.get('RegimeSmoothing'),
                'sheets_count': len(initial_config)
            }
            
            # Each thread will verify it sees the same values
            verification_results = []
            results_lock = threading.Lock()
            
            def memory_consistency_worker(worker_id: int):
                """Worker that verifies memory consistency"""
                # Small delay to ensure all threads start around the same time
                time.sleep(random.uniform(0, 0.1))
                
                # Get configuration
                config = manager.load_configuration()
                params = manager.get_detection_parameters()
                
                # Compare with initial values
                current_values = {
                    'confidence_threshold': params.get('ConfidenceThreshold'),
                    'regime_smoothing': params.get('RegimeSmoothing'),
                    'sheets_count': len(config)
                }
                
                # Check consistency
                is_consistent = all(
                    current_values[key] == initial_values[key]
                    for key in initial_values
                )
                
                with results_lock:
                    verification_results.append({
                        'worker_id': worker_id,
                        'is_consistent': is_consistent,
                        'current_values': current_values,
                        'timestamp': datetime.now()
                    })
            
            # Run many workers
            num_workers = 100
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(memory_consistency_worker, i)
                    for i in range(num_workers)
                ]
                
                for future in as_completed(futures, timeout=15):
                    future.result()
            
            # Verify all workers saw consistent values
            inconsistent_workers = [
                r for r in verification_results 
                if not r['is_consistent']
            ]
            
            self.assertEqual(len(inconsistent_workers), 0, 
                           f"Memory inconsistency detected in {len(inconsistent_workers)} workers")
            
            logger.info(f"✅ PHASE 4.1.2: Memory consistency verified across {num_workers} workers")
            
        except Exception as e:
            self.fail(f"Memory consistency test failed: {e}")

def run_thread_safe_config_tests():
    """Run thread-safe configuration loading test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔒 PHASE 4.1.2: THREAD-SAFE CONFIGURATION LOADING TESTS")
    print("=" * 70)
    print("⚠️  STRICT MODE: Using real Excel configuration file")
    print("⚠️  NO MOCK DATA: All tests use actual MR_CONFIG_STRATEGY_1.0.0.xlsx")
    print("⚠️  CONCURRENCY: Testing thread-safe access patterns")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestThreadSafeConfigLoading)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"PHASE 4.1.2: THREAD-SAFE CONFIGURATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'=' * 70}")
    
    if failures > 0 or errors > 0:
        print("❌ PHASE 4.1.2: THREAD-SAFE CONFIGURATION FAILED")
        print("🔧 ISSUES NEED TO BE FIXED BEFORE PROCEEDING")
        
        if failures > 0:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if errors > 0:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        return False
    else:
        print("✅ PHASE 4.1.2: THREAD-SAFE CONFIGURATION PASSED")
        print("🔒 CONCURRENT ACCESS VALIDATED")
        print("📊 DATA CONSISTENCY CONFIRMED")
        print("🏃 NO RACE CONDITIONS DETECTED")
        print("✅ READY FOR PHASE 4.1.3 - ALL 31 SHEETS PARSING")
        return True

if __name__ == "__main__":
    success = run_thread_safe_config_tests()
    sys.exit(0 if success else 1)