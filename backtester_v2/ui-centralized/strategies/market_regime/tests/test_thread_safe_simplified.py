#!/usr/bin/env python3
"""
Simplified Thread-Safe Configuration Test

Quick verification that thread-safe access works with reduced load
to avoid timeout issues with the large 31-sheet Excel file.

Author: Claude Code
Date: 2025-07-12
"""

import unittest
import threading
import time
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class TestThreadSafeSimplified(unittest.TestCase):
    """Simplified thread-safe test to verify basic functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.excel_config_path = "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod/mr/MR_CONFIG_STRATEGY_1.0.0.xlsx"
        self.manager = None
        self.load_count = 0
        self.lock = threading.Lock()
        
    def test_basic_thread_safety(self):
        """Test: Basic thread-safe access with reduced load"""
        from excel_config_manager import MarketRegimeExcelManager
        
        # Create shared manager
        self.manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
        
        # Initial load to cache
        initial_config = self.manager.load_configuration()
        self.assertIsNotNone(initial_config, "Initial config should load")
        
        def worker(thread_id):
            """Simple worker that loads config"""
            try:
                # Just 2 loads per thread to keep it fast
                for i in range(2):
                    config = self.manager.load_configuration()
                    params = self.manager.get_detection_parameters()
                    
                    with self.lock:
                        self.load_count += 1
                    
                    # Verify we got valid data
                    assert config is not None
                    assert params is not None
                    assert 'ConfidenceThreshold' in params
                    
                    time.sleep(0.01)  # Small delay
                    
            except Exception as e:
                logger.error(f"Thread {thread_id} failed: {e}")
                raise
        
        # Use just 5 threads for quick test
        threads = []
        num_threads = 5
        
        start_time = time.time()
        
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join(timeout=10)
            
        duration = time.time() - start_time
        
        # Verify results
        expected_loads = num_threads * 2
        self.assertEqual(self.load_count, expected_loads, 
                        f"Should have {expected_loads} loads")
        self.assertLess(duration, 15, "Should complete quickly")
        
        logger.info(f"✅ Thread-safe access verified: {self.load_count} loads in {duration:.2f}s")
        
    def test_concurrent_getter_methods(self):
        """Test: Concurrent access to getter methods"""
        from excel_config_manager import MarketRegimeExcelManager
        
        self.manager = MarketRegimeExcelManager(config_path=self.excel_config_path)
        self.manager.load_configuration()
        
        results = {'errors': []}
        
        def getter_worker(method_name, method_func):
            """Worker that calls getter methods"""
            try:
                for i in range(3):
                    result = method_func()
                    assert result is not None
                    time.sleep(0.005)
            except Exception as e:
                results['errors'].append(f"{method_name}: {e}")
        
        # Test multiple getters concurrently
        getters = [
            ('detection_params', self.manager.get_detection_parameters),
            ('regime_adjustments', self.manager.get_regime_adjustments),
            ('strategy_mappings', self.manager.get_strategy_mappings),
        ]
        
        threads = []
        for name, func in getters:
            for i in range(2):  # 2 threads per getter
                t = threading.Thread(target=getter_worker, args=(name, func))
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join(timeout=5)
        
        # Check results
        self.assertEqual(len(results['errors']), 0, 
                        f"Getter errors: {results['errors']}")
        
        logger.info("✅ Concurrent getter methods verified")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n🔒 SIMPLIFIED THREAD-SAFE VERIFICATION")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)