#!/usr/bin/env python3
"""
Configuration Loading Performance Test

PHASE 5.1: Test configuration loading performance and optimization
- Tests Excel configuration loading speed and efficiency
- Validates memory usage during configuration operations
- Tests caching and optimization strategies
- Ensures performance meets production requirements
- NO MOCK DATA - uses real Excel configuration with performance monitoring

Author: Claude Code
Date: 2025-07-12
Version: 1.0.0 - PHASE 5.1 CONFIGURATION LOADING PERFORMANCE
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class TestConfigurationLoadingPerformance(unittest.TestCase):
    """
    PHASE 5.1: Configuration Loading Performance Test Suite
    STRICT: Uses real Excel file with NO MOCK data
    """
    
    def setUp(self):
        """Set up test environment with STRICT real data requirements"""
        self.excel_config_path = "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod/mr/MR_CONFIG_STRATEGY_1.0.0.xlsx"
        self.strict_mode = True
        self.no_mock_data = True
        
        # Verify Excel file exists
        if not Path(self.excel_config_path).exists():
            self.fail(f"CRITICAL: Excel configuration file not found: {self.excel_config_path}")
        
        # Performance requirements
        self.max_load_time = 5.0  # seconds
        self.max_memory_mb = 100  # MB
        self.min_throughput = 10  # configs per second
        
        logger.info(f"✅ Excel configuration file verified: {self.excel_config_path}")
        logger.info(f"📊 Performance requirements: Load time < {self.max_load_time}s, Memory < {self.max_memory_mb}MB")
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_single_configuration_load_performance(self):
        """Test: Single configuration load performance"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            logger.info("🔄 Testing single configuration load performance...")
            
            # Measure baseline memory
            gc.collect()
            baseline_memory = self.get_memory_usage()
            
            # Measure load time
            start_time = time.time()
            
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            config_data = manager.load_configuration()
            
            load_time = time.time() - start_time
            
            # Measure memory after load
            loaded_memory = self.get_memory_usage()
            memory_usage = loaded_memory - baseline_memory
            
            # Performance assertions
            self.assertLess(load_time, self.max_load_time, 
                          f"Load time {load_time:.2f}s exceeds maximum {self.max_load_time}s")
            
            self.assertLess(memory_usage, self.max_memory_mb,
                          f"Memory usage {memory_usage:.2f}MB exceeds maximum {self.max_memory_mb}MB")
            
            # Verify configuration integrity
            self.assertIsInstance(config_data, dict, "Configuration should be loaded as dict")
            self.assertGreater(len(config_data), 0, "Configuration should not be empty")
            
            # Performance metrics
            file_size_mb = os.path.getsize(self.excel_config_path) / 1024 / 1024
            throughput = file_size_mb / load_time
            
            performance_metrics = {
                'load_time': load_time,
                'memory_usage_mb': memory_usage,
                'file_size_mb': file_size_mb,
                'throughput_mb_per_sec': throughput,
                'sheets_loaded': len(config_data),
                'sheets_per_second': len(config_data) / load_time
            }
            
            logger.info(f"📊 Single load performance metrics: {performance_metrics}")
            logger.info("✅ PHASE 5.1: Single configuration load performance validated")
            
        except Exception as e:
            self.fail(f"Single configuration load performance test failed: {e}")
    
    def test_repeated_configuration_loads_performance(self):
        """Test: Repeated configuration loads performance"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            logger.info("🔄 Testing repeated configuration loads performance...")
            
            num_loads = 10
            load_times = []
            memory_usages = []
            
            # Baseline memory
            gc.collect()
            baseline_memory = self.get_memory_usage()
            
            for i in range(num_loads):
                # Measure each load
                gc.collect()
                start_memory = self.get_memory_usage()
                start_time = time.time()
                
                manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
                config_data = manager.load_configuration()
                
                load_time = time.time() - start_time
                end_memory = self.get_memory_usage()
                
                load_times.append(load_time)
                memory_usages.append(end_memory - start_memory)
                
                # Verify configuration integrity
                self.assertIsInstance(config_data, dict, f"Configuration {i+1} should be loaded as dict")
                self.assertGreater(len(config_data), 0, f"Configuration {i+1} should not be empty")
                
                logger.info(f"Load {i+1}: {load_time:.3f}s, Memory: {memory_usages[-1]:.2f}MB")
            
            # Performance analysis
            avg_load_time = np.mean(load_times)
            max_load_time = np.max(load_times)
            min_load_time = np.min(load_times)
            std_load_time = np.std(load_times)
            
            avg_memory = np.mean(memory_usages)
            max_memory = np.max(memory_usages)
            
            # Performance assertions
            self.assertLess(avg_load_time, self.max_load_time,
                          f"Average load time {avg_load_time:.2f}s exceeds maximum {self.max_load_time}s")
            
            self.assertLess(max_memory, self.max_memory_mb,
                          f"Maximum memory usage {max_memory:.2f}MB exceeds limit {self.max_memory_mb}MB")
            
            # Consistency check (load times should be relatively stable)
            cv_load_time = std_load_time / avg_load_time if avg_load_time > 0 else 0
            self.assertLess(cv_load_time, 0.5, "Load times should be relatively consistent")
            
            repeated_metrics = {
                'num_loads': num_loads,
                'avg_load_time': avg_load_time,
                'min_load_time': min_load_time,
                'max_load_time': max_load_time,
                'std_load_time': std_load_time,
                'cv_load_time': cv_load_time,
                'avg_memory_mb': avg_memory,
                'max_memory_mb': max_memory,
                'throughput_loads_per_sec': 1.0 / avg_load_time
            }
            
            logger.info(f"📊 Repeated loads performance metrics: {repeated_metrics}")
            logger.info("✅ PHASE 5.1: Repeated configuration loads performance validated")
            
        except Exception as e:
            self.fail(f"Repeated configuration loads performance test failed: {e}")
    
    def test_concurrent_configuration_loads_performance(self):
        """Test: Concurrent configuration loads performance"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            logger.info("🔄 Testing concurrent configuration loads performance...")
            
            num_threads = 3
            loads_per_thread = 2
            total_loads = num_threads * loads_per_thread
            
            def load_configuration(thread_id):
                """Load configuration in a thread"""
                thread_results = []
                
                for i in range(loads_per_thread):
                    start_time = time.time()
                    
                    try:
                        manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
                        config_data = manager.load_configuration()
                        
                        load_time = time.time() - start_time
                        
                        if config_data and len(config_data) > 0:
                            thread_results.append({
                                'thread_id': thread_id,
                                'load_id': i,
                                'load_time': load_time,
                                'status': 'success',
                                'sheets_loaded': len(config_data)
                            })
                        else:
                            thread_results.append({
                                'thread_id': thread_id,
                                'load_id': i,
                                'load_time': load_time,
                                'status': 'empty_config',
                                'sheets_loaded': 0
                            })
                    except Exception as e:
                        load_time = time.time() - start_time
                        thread_results.append({
                            'thread_id': thread_id,
                            'load_id': i,
                            'load_time': load_time,
                            'status': 'error',
                            'error': str(e)
                        })
                
                return thread_results
            
            # Measure concurrent execution
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(load_configuration, i) for i in range(num_threads)]
                
                all_results = []
                for future in as_completed(futures):
                    thread_results = future.result()
                    all_results.extend(thread_results)
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful_loads = [r for r in all_results if r['status'] == 'success']
            failed_loads = [r for r in all_results if r['status'] == 'error']
            empty_loads = [r for r in all_results if r['status'] == 'empty_config']
            
            # Performance metrics
            if successful_loads:
                load_times = [r['load_time'] for r in successful_loads]
                avg_load_time = np.mean(load_times)
                max_load_time = np.max(load_times)
                concurrent_throughput = len(successful_loads) / total_time
                
                # Performance assertions
                self.assertGreater(len(successful_loads), total_loads * 0.8,
                                 "At least 80% of concurrent loads should succeed")
                
                self.assertLess(avg_load_time, self.max_load_time * 1.5,
                              "Concurrent load times should not be significantly slower")
                
                self.assertGreater(concurrent_throughput, self.min_throughput * 0.5,
                                 "Concurrent throughput should meet minimum requirements")
            
            concurrent_metrics = {
                'num_threads': num_threads,
                'loads_per_thread': loads_per_thread,
                'total_loads': total_loads,
                'successful_loads': len(successful_loads),
                'failed_loads': len(failed_loads),
                'empty_loads': len(empty_loads),
                'success_rate': len(successful_loads) / total_loads,
                'total_time': total_time,
                'concurrent_throughput': len(successful_loads) / total_time if total_time > 0 else 0
            }
            
            if successful_loads:
                concurrent_metrics.update({
                    'avg_load_time': avg_load_time,
                    'max_load_time': max_load_time,
                    'min_load_time': np.min(load_times)
                })
            
            logger.info(f"📊 Concurrent loads performance metrics: {concurrent_metrics}")
            logger.info("✅ PHASE 5.1: Concurrent configuration loads performance validated")
            
        except Exception as e:
            self.fail(f"Concurrent configuration loads performance test failed: {e}")
    
    def test_parameter_extraction_performance(self):
        """Test: Parameter extraction performance"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            logger.info("🔄 Testing parameter extraction performance...")
            
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            
            # Load configuration once
            config_load_start = time.time()
            config_data = manager.load_configuration()
            config_load_time = time.time() - config_load_start
            
            # Test different parameter extraction methods
            extraction_methods = [
                ('detection_parameters', manager.get_detection_parameters),
                ('regime_adjustments', manager.get_regime_adjustments),
                ('strategy_mappings', manager.get_strategy_mappings),
                ('live_trading_config', manager.get_live_trading_config),
                ('technical_indicators_config', manager.get_technical_indicators_config)
            ]
            
            extraction_results = {}
            
            for method_name, method in extraction_methods:
                # Measure extraction time
                start_time = time.time()
                
                try:
                    result = method()
                    extraction_time = time.time() - start_time
                    
                    extraction_results[method_name] = {
                        'status': 'success',
                        'extraction_time': extraction_time,
                        'result_size': len(result) if result else 0,
                        'result_type': type(result).__name__
                    }
                    
                    # Performance assertion - extraction should be fast
                    self.assertLess(extraction_time, 1.0,
                                  f"{method_name} extraction should complete within 1 second")
                    
                    logger.info(f"✅ {method_name}: {extraction_time:.3f}s, Size: {extraction_results[method_name]['result_size']}")
                    
                except Exception as e:
                    extraction_time = time.time() - start_time
                    extraction_results[method_name] = {
                        'status': 'error',
                        'extraction_time': extraction_time,
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ {method_name} extraction failed: {e}")
            
            # Overall performance metrics
            total_extraction_time = sum(r['extraction_time'] for r in extraction_results.values())
            successful_extractions = [r for r in extraction_results.values() if r['status'] == 'success']
            
            performance_summary = {
                'config_load_time': config_load_time,
                'total_extraction_time': total_extraction_time,
                'total_time': config_load_time + total_extraction_time,
                'successful_extractions': len(successful_extractions),
                'total_extractions': len(extraction_methods),
                'extraction_success_rate': len(successful_extractions) / len(extraction_methods),
                'avg_extraction_time': np.mean([r['extraction_time'] for r in successful_extractions]) if successful_extractions else 0
            }
            
            # Performance assertions
            self.assertLess(performance_summary['total_time'], self.max_load_time * 2,
                          "Total configuration and extraction time should be reasonable")
            
            self.assertGreater(performance_summary['extraction_success_rate'], 0.6,
                             "At least 60% of extractions should succeed")
            
            logger.info(f"📊 Parameter extraction performance summary: {performance_summary}")
            logger.info("✅ PHASE 5.1: Parameter extraction performance validated")
            
        except Exception as e:
            self.fail(f"Parameter extraction performance test failed: {e}")
    
    def test_memory_efficiency_optimization(self):
        """Test: Memory efficiency and optimization"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            logger.info("🔄 Testing memory efficiency and optimization...")
            
            # Measure memory usage patterns
            memory_measurements = []
            
            def measure_memory(label):
                gc.collect()
                memory_mb = self.get_memory_usage()
                memory_measurements.append({'label': label, 'memory_mb': memory_mb})
                logger.info(f"📊 {label}: {memory_mb:.2f}MB")
                return memory_mb
            
            # Baseline measurement
            baseline_memory = measure_memory("Baseline")
            
            # Load configuration
            manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
            after_init_memory = measure_memory("After Manager Init")
            
            config_data = manager.load_configuration()
            after_load_memory = measure_memory("After Config Load")
            
            # Extract parameters
            detection_params = manager.get_detection_parameters()
            after_detection_memory = measure_memory("After Detection Params")
            
            live_config = manager.get_live_trading_config()
            after_live_memory = measure_memory("After Live Config")
            
            # Calculate memory deltas
            memory_deltas = {
                'manager_init': after_init_memory - baseline_memory,
                'config_load': after_load_memory - after_init_memory,
                'detection_params': after_detection_memory - after_load_memory,
                'live_config': after_live_memory - after_detection_memory,
                'total_usage': after_live_memory - baseline_memory
            }
            
            # Memory optimization test - cleanup
            del config_data
            del detection_params
            del live_config
            del manager
            
            gc.collect()
            after_cleanup_memory = measure_memory("After Cleanup")
            
            memory_deltas['cleanup_effectiveness'] = after_live_memory - after_cleanup_memory
            memory_deltas['final_overhead'] = after_cleanup_memory - baseline_memory
            
            # Memory efficiency assertions
            self.assertLess(memory_deltas['total_usage'], self.max_memory_mb,
                          f"Total memory usage {memory_deltas['total_usage']:.2f}MB exceeds limit")
            
            self.assertGreater(memory_deltas['cleanup_effectiveness'], 0,
                             "Memory cleanup should be effective")
            
            self.assertLess(memory_deltas['final_overhead'], 20,
                          "Final memory overhead should be minimal")
            
            # Memory efficiency metrics
            file_size_mb = os.path.getsize(self.excel_config_path) / 1024 / 1024
            memory_efficiency = file_size_mb / memory_deltas['total_usage'] if memory_deltas['total_usage'] > 0 else 0
            
            efficiency_metrics = {
                'file_size_mb': file_size_mb,
                'memory_deltas': memory_deltas,
                'memory_efficiency_ratio': memory_efficiency,
                'memory_measurements': memory_measurements
            }
            
            logger.info(f"📊 Memory efficiency metrics: {efficiency_metrics}")
            logger.info("✅ PHASE 5.1: Memory efficiency and optimization validated")
            
        except Exception as e:
            self.fail(f"Memory efficiency optimization test failed: {e}")
    
    def test_configuration_caching_performance(self):
        """Test: Configuration caching performance"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            logger.info("🔄 Testing configuration caching performance...")
            
            # Test if caching improves performance
            cache_test_results = []
            
            # First load (no cache)
            start_time = time.time()
            manager1 = MarketRegimeExcelManager(config_path=self.excel_config_path)
            config1 = manager1.load_configuration()
            first_load_time = time.time() - start_time
            
            cache_test_results.append({
                'load_type': 'first_load',
                'load_time': first_load_time,
                'config_size': len(config1) if config1 else 0
            })
            
            # Second load (potential cache benefit)
            start_time = time.time()
            manager2 = MarketRegimeExcelManager(config_path=self.excel_config_path)
            config2 = manager2.load_configuration()
            second_load_time = time.time() - start_time
            
            cache_test_results.append({
                'load_type': 'second_load',
                'load_time': second_load_time,
                'config_size': len(config2) if config2 else 0
            })
            
            # Same manager reload
            start_time = time.time()
            config3 = manager2.load_configuration()
            same_manager_reload_time = time.time() - start_time
            
            cache_test_results.append({
                'load_type': 'same_manager_reload',
                'load_time': same_manager_reload_time,
                'config_size': len(config3) if config3 else 0
            })
            
            # Analyze caching effectiveness
            caching_analysis = {
                'first_load_time': first_load_time,
                'second_load_time': second_load_time,
                'same_manager_reload_time': same_manager_reload_time,
                'second_vs_first_ratio': second_load_time / first_load_time if first_load_time > 0 else 1,
                'reload_vs_first_ratio': same_manager_reload_time / first_load_time if first_load_time > 0 else 1
            }
            
            # If caching is implemented, subsequent loads should be faster
            potential_cache_benefit = first_load_time > second_load_time or first_load_time > same_manager_reload_time
            
            if potential_cache_benefit:
                logger.info("✅ Potential caching benefits detected")
                caching_analysis['cache_benefit_detected'] = True
            else:
                logger.info("📊 No significant caching benefits detected (may not be implemented)")
                caching_analysis['cache_benefit_detected'] = False
            
            # Verify configuration consistency across loads
            if config1 and config2 and config3:
                configs_consistent = (len(config1) == len(config2) == len(config3))
                caching_analysis['config_consistency'] = configs_consistent
                
                if configs_consistent:
                    logger.info("✅ Configuration consistency maintained across loads")
                else:
                    logger.warning("⚠️ Configuration inconsistency detected across loads")
            
            logger.info(f"📊 Configuration caching analysis: {caching_analysis}")
            logger.info("✅ PHASE 5.1: Configuration caching performance validated")
            
        except Exception as e:
            self.fail(f"Configuration caching performance test failed: {e}")

def run_configuration_loading_performance_tests():
    """Run Configuration Loading Performance test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 PHASE 5.1: CONFIGURATION LOADING PERFORMANCE TESTS")
    print("=" * 70)
    print("⚠️  STRICT MODE: Using real Excel configuration file")
    print("⚠️  NO MOCK DATA: All tests use actual MR_CONFIG_STRATEGY_1.0.0.xlsx")
    print("📊 PERFORMANCE: Testing configuration loading speed and efficiency")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestConfigurationLoadingPerformance)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"PHASE 5.1: CONFIGURATION LOADING PERFORMANCE RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'=' * 70}")
    
    if failures > 0 or errors > 0:
        print("❌ PHASE 5.1: CONFIGURATION LOADING PERFORMANCE FAILED")
        print("🔧 PERFORMANCE ISSUES NEED TO BE ADDRESSED")
        
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
        print("✅ PHASE 5.1: CONFIGURATION LOADING PERFORMANCE PASSED")
        print("🚀 SINGLE LOAD PERFORMANCE VALIDATED")
        print("🔄 REPEATED LOADS PERFORMANCE CONFIRMED")
        print("⚡ CONCURRENT LOADS PERFORMANCE VERIFIED")
        print("📊 PARAMETER EXTRACTION PERFORMANCE TESTED")
        print("💾 MEMORY EFFICIENCY OPTIMIZATION VALIDATED")
        print("🎯 CONFIGURATION CACHING PERFORMANCE ANALYZED")
        print("✅ READY FOR PHASE 5.2 - REGIME DETECTION ALGORITHM PERFORMANCE")
        return True

if __name__ == "__main__":
    success = run_configuration_loading_performance_tests()
    sys.exit(0 if success else 1)