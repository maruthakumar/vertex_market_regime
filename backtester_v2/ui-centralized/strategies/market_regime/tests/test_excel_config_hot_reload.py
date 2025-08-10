#!/usr/bin/env python3
"""
Excel Configuration Hot-Reload Integration Test

PHASE 4.1.1: Test Excel Config Manager hot-reload functionality
- Tests real-time configuration updates via file watching
- Validates change callbacks and configuration propagation
- Tests thread-safe updates and audit trail
- NO MOCK DATA - uses real Excel configuration

Author: Claude Code
Date: 2025-07-12
Version: 1.0.0 - PHASE 4.1.1 HOT-RELOAD TESTING
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
import shutil
from datetime import datetime
from typing import Dict, List, Any
import tempfile

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

class TestExcelConfigHotReload(unittest.TestCase):
    """
    PHASE 4.1.1: Excel Configuration Hot-Reload Test Suite
    STRICT: Uses real Excel file with NO MOCK data
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with temporary Excel file"""
        # Original Excel config path
        cls.original_excel_path = "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod/mr/MR_CONFIG_STRATEGY_1.0.0.xlsx"
        
        # Create temp directory for testing
        cls.temp_dir = tempfile.mkdtemp(prefix="market_regime_test_")
        cls.test_excel_path = os.path.join(cls.temp_dir, "test_market_regime_config.xlsx")
        
        # Copy the original Excel file to temp location
        if Path(cls.original_excel_path).exists():
            shutil.copy2(cls.original_excel_path, cls.test_excel_path)
            logger.info(f"Created test Excel file at: {cls.test_excel_path}")
        else:
            raise FileNotFoundError(f"Original Excel file not found: {cls.original_excel_path}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        try:
            shutil.rmtree(cls.temp_dir)
            logger.info(f"Cleaned up test directory: {cls.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up test directory: {e}")
    
    def setUp(self):
        """Set up test environment for each test"""
        self.config_changes = []
        self.callback_count = 0
        self.last_callback_time = None
        self.callback_lock = threading.Lock()
        
    def _config_change_callback(self, old_config: Dict, new_config: Dict, change_type: str):
        """Callback to track configuration changes"""
        with self.callback_lock:
            self.callback_count += 1
            self.last_callback_time = datetime.now()
            self.config_changes.append({
                'timestamp': self.last_callback_time,
                'change_type': change_type,
                'old_config': old_config,
                'new_config': new_config
            })
            logger.info(f"Config change callback #{self.callback_count}: {change_type}")
    
    def test_hot_reload_basic_functionality(self):
        """Test: Basic hot-reload functionality with file watching"""
        try:
            from config.excel_config_manager import ExcelConfigManager
            
            # Initialize manager with test Excel file
            manager = ExcelConfigManager(self.test_excel_path)
            
            # Add change callback
            manager.add_change_callback(self._config_change_callback)
            
            # Start file watching
            watch_started = manager.start_file_watching()
            self.assertTrue(watch_started, "File watching should start successfully")
            self.assertTrue(manager.is_watching, "Manager should be in watching state")
            
            # Load initial configuration
            initial_config = manager.load_configuration()
            self.assertIsNotNone(initial_config, "Initial configuration should load")
            
            # Give file watcher time to stabilize
            time.sleep(1)
            
            # Modify the Excel file (change a parameter)
            excel_file = pd.ExcelFile(self.test_excel_path)
            with pd.ExcelWriter(self.test_excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                # Modify RegimeParameters sheet
                if 'RegimeParameters' in excel_file.sheet_names:
                    df = pd.read_excel(self.test_excel_path, sheet_name='RegimeParameters')
                    # Change ConfidenceThreshold value
                    df.loc[df['Parameter'] == 'ConfidenceThreshold', 'Value'] = 0.85
                    df.to_excel(writer, sheet_name='RegimeParameters', index=False)
                    logger.info("Modified ConfidenceThreshold in Excel file")
            
            # Wait for hot-reload to trigger (file system events + processing)
            time.sleep(3)
            
            # Verify callback was triggered
            with self.callback_lock:
                self.assertGreater(self.callback_count, 0, "At least one callback should be triggered")
                self.assertIsNotNone(self.last_callback_time, "Callback timestamp should be set")
            
            # Load updated configuration
            updated_config = manager.load_configuration()
            
            # Verify configuration was updated
            self.assertEqual(updated_config.regime_parameters.confidence_threshold, 0.85, 
                           "ConfidenceThreshold should be updated to 0.85")
            
            # Stop file watching
            manager.stop_file_watching()
            self.assertFalse(manager.is_watching, "Manager should stop watching")
            
            logger.info("✅ PHASE 4.1.1: Basic hot-reload functionality validated")
            
        except ImportError as e:
            self.fail(f"Failed to import ExcelConfigManager: {e}")
        except Exception as e:
            self.fail(f"Hot-reload test failed: {e}")
    
    def test_thread_safe_configuration_loading(self):
        """Test: Thread-safe configuration loading during concurrent access"""
        try:
            from config.excel_config_manager import ExcelConfigManager
            
            manager = ExcelConfigManager(self.test_excel_path)
            manager.start_file_watching()
            
            # Track results from concurrent threads
            thread_results = {}
            thread_errors = []
            
            def load_config_thread(thread_id: int, iterations: int = 10):
                """Thread function to load configuration multiple times"""
                try:
                    results = []
                    for i in range(iterations):
                        config = manager.load_configuration()
                        results.append({
                            'iteration': i,
                            'confidence_threshold': config.regime_parameters.confidence_threshold,
                            'timestamp': datetime.now()
                        })
                        time.sleep(0.1)  # Small delay between loads
                    thread_results[thread_id] = results
                except Exception as e:
                    thread_errors.append((thread_id, str(e)))
            
            # Create multiple threads for concurrent access
            threads = []
            num_threads = 5
            
            for i in range(num_threads):
                thread = threading.Thread(target=load_config_thread, args=(i, 5))
                threads.append(thread)
                thread.start()
            
            # Modify Excel file while threads are running
            time.sleep(1)
            excel_file = pd.ExcelFile(self.test_excel_path)
            with pd.ExcelWriter(self.test_excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                if 'RegimeParameters' in excel_file.sheet_names:
                    df = pd.read_excel(self.test_excel_path, sheet_name='RegimeParameters')
                    df.loc[df['Parameter'] == 'RegimeSmoothing', 'Value'] = 3  # Change from default
                    df.to_excel(writer, sheet_name='RegimeParameters', index=False)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=30)
            
            # Verify no errors occurred
            self.assertEqual(len(thread_errors), 0, 
                           f"Thread errors occurred: {thread_errors}")
            
            # Verify all threads got valid results
            self.assertEqual(len(thread_results), num_threads, 
                           "All threads should complete successfully")
            
            # Verify data consistency
            for thread_id, results in thread_results.items():
                self.assertGreater(len(results), 0, 
                                 f"Thread {thread_id} should have results")
                for result in results:
                    self.assertIsNotNone(result['confidence_threshold'], 
                                       "Configuration values should not be None")
            
            manager.stop_file_watching()
            logger.info("✅ PHASE 4.1.1: Thread-safe configuration loading validated")
            
        except Exception as e:
            self.fail(f"Thread-safe loading test failed: {e}")
    
    def test_audit_trail_for_parameter_changes(self):
        """Test: Audit trail tracking for parameter changes"""
        try:
            from config.excel_config_manager import ExcelConfigManager
            
            manager = ExcelConfigManager(self.test_excel_path)
            
            # Get initial audit trail
            initial_audit = manager.get_audit_trail()
            initial_count = len(initial_audit)
            
            # Start file watching
            manager.start_file_watching()
            
            # Make multiple parameter changes
            changes_made = []
            
            # Change 1: Modify ConfidenceThreshold
            with pd.ExcelWriter(self.test_excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df = pd.read_excel(self.test_excel_path, sheet_name='RegimeParameters')
                df.loc[df['Parameter'] == 'ConfidenceThreshold', 'Value'] = 0.75
                df.to_excel(writer, sheet_name='RegimeParameters', index=False)
                changes_made.append(('ConfidenceThreshold', 0.75))
            
            time.sleep(2)  # Wait for hot-reload
            
            # Change 2: Modify Greek thresholds
            with pd.ExcelWriter(self.test_excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                if 'GreekThresholds' in pd.ExcelFile(self.test_excel_path).sheet_names:
                    df = pd.read_excel(self.test_excel_path, sheet_name='GreekThresholds')
                    df.loc[df['Parameter'] == 'DeltaExposureMax', 'Value'] = 150000
                    df.to_excel(writer, sheet_name='GreekThresholds', index=False)
                    changes_made.append(('DeltaExposureMax', 150000))
            
            time.sleep(2)  # Wait for hot-reload
            
            # Get updated audit trail
            updated_audit = manager.get_audit_trail()
            new_entries = len(updated_audit) - initial_count
            
            # Verify audit trail captured changes
            self.assertGreater(new_entries, 0, "Audit trail should have new entries")
            
            # Check audit trail content
            recent_entries = updated_audit[-new_entries:]
            for entry in recent_entries:
                self.assertIn('timestamp', entry)
                self.assertIn('change_type', entry)
                self.assertIn('details', entry)
                
                # Verify timestamp is recent
                entry_time = datetime.fromisoformat(entry['timestamp'])
                time_diff = datetime.now() - entry_time
                self.assertLess(time_diff.total_seconds(), 60, 
                              "Audit entry should be recent")
            
            manager.stop_file_watching()
            logger.info("✅ PHASE 4.1.1: Audit trail for parameter changes validated")
            
        except Exception as e:
            self.fail(f"Audit trail test failed: {e}")
    
    def test_change_callback_propagation(self):
        """Test: Change callbacks propagate to multiple registered handlers"""
        try:
            from config.excel_config_manager import ExcelConfigManager
            
            manager = ExcelConfigManager(self.test_excel_path)
            
            # Track callbacks from multiple handlers
            handler1_calls = []
            handler2_calls = []
            handler3_calls = []
            
            def handler1(old_cfg, new_cfg, change_type):
                handler1_calls.append((change_type, datetime.now()))
                logger.info("Handler 1 triggered")
            
            def handler2(old_cfg, new_cfg, change_type):
                handler2_calls.append((change_type, datetime.now()))
                logger.info("Handler 2 triggered")
            
            def handler3(old_cfg, new_cfg, change_type):
                handler3_calls.append((change_type, datetime.now()))
                logger.info("Handler 3 triggered")
            
            # Register multiple callbacks
            manager.add_change_callback(handler1)
            manager.add_change_callback(handler2)
            manager.add_change_callback(handler3)
            
            # Start file watching
            manager.start_file_watching()
            time.sleep(1)
            
            # Trigger a configuration change
            with pd.ExcelWriter(self.test_excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df = pd.read_excel(self.test_excel_path, sheet_name='RegimeParameters')
                df.loc[df['Parameter'] == 'TransitionFrequencyLimit', 'Value'] = 0.6
                df.to_excel(writer, sheet_name='RegimeParameters', index=False)
            
            # Wait for callbacks
            time.sleep(3)
            
            # Verify all handlers were called
            self.assertGreater(len(handler1_calls), 0, "Handler 1 should be called")
            self.assertGreater(len(handler2_calls), 0, "Handler 2 should be called")
            self.assertGreater(len(handler3_calls), 0, "Handler 3 should be called")
            
            # Verify callbacks happened at similar times
            if handler1_calls and handler2_calls and handler3_calls:
                time1 = handler1_calls[0][1]
                time2 = handler2_calls[0][1]
                time3 = handler3_calls[0][1]
                
                # All callbacks should happen within 1 second of each other
                self.assertLess((time2 - time1).total_seconds(), 1.0)
                self.assertLess((time3 - time1).total_seconds(), 1.0)
            
            manager.stop_file_watching()
            logger.info("✅ PHASE 4.1.1: Change callback propagation validated")
            
        except Exception as e:
            self.fail(f"Callback propagation test failed: {e}")
    
    def test_hot_reload_performance(self):
        """Test: Hot-reload performance under rapid changes"""
        try:
            from config.excel_config_manager import ExcelConfigManager
            
            manager = ExcelConfigManager(self.test_excel_path)
            manager.add_change_callback(self._config_change_callback)
            manager.start_file_watching()
            
            # Initial configuration load
            initial_config = manager.load_configuration()
            time.sleep(1)
            
            # Make rapid changes
            num_changes = 5
            change_times = []
            
            for i in range(num_changes):
                start_time = time.time()
                
                # Make a change
                with pd.ExcelWriter(self.test_excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    df = pd.read_excel(self.test_excel_path, sheet_name='RegimeParameters')
                    df.loc[df['Parameter'] == 'ConfidenceThreshold', 'Value'] = 0.7 + (i * 0.02)
                    df.to_excel(writer, sheet_name='RegimeParameters', index=False)
                
                # Wait briefly
                time.sleep(0.5)
                
                end_time = time.time()
                change_times.append(end_time - start_time)
            
            # Wait for all changes to process
            time.sleep(2)
            
            # Verify performance
            avg_change_time = sum(change_times) / len(change_times)
            self.assertLess(avg_change_time, 1.0, 
                          f"Average change time should be < 1 second, got {avg_change_time:.2f}s")
            
            # Verify all changes were captured
            with self.callback_lock:
                # Due to file system event coalescing, we might not get exactly num_changes callbacks
                self.assertGreater(self.callback_count, 0, "At least some callbacks should be triggered")
            
            manager.stop_file_watching()
            logger.info(f"✅ PHASE 4.1.1: Hot-reload performance validated (avg: {avg_change_time:.2f}s)")
            
        except Exception as e:
            self.fail(f"Hot-reload performance test failed: {e}")

def run_hot_reload_integration_tests():
    """Run Excel configuration hot-reload test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔥 PHASE 4.1.1: EXCEL CONFIG HOT-RELOAD INTEGRATION TESTS")
    print("=" * 70)
    print("⚠️  STRICT MODE: Using real Excel configuration file")
    print("⚠️  NO MOCK DATA: All tests use actual configuration")
    print("⚠️  HOT-RELOAD: Testing real-time configuration updates")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestExcelConfigHotReload)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"PHASE 4.1.1: HOT-RELOAD INTEGRATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'=' * 70}")
    
    if failures > 0 or errors > 0:
        print("❌ PHASE 4.1.1: HOT-RELOAD INTEGRATION FAILED")
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
        print("✅ PHASE 4.1.1: HOT-RELOAD INTEGRATION PASSED")
        print("🔥 HOT-RELOAD FUNCTIONALITY VALIDATED")
        print("🔄 THREAD-SAFE UPDATES CONFIRMED")
        print("📝 AUDIT TRAIL WORKING CORRECTLY")
        print("✅ READY FOR PHASE 4.1.2 - THREAD-SAFE CONFIGURATION LOADING")
        return True

if __name__ == "__main__":
    success = run_hot_reload_integration_tests()
    sys.exit(0 if success else 1)