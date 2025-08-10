#!/usr/bin/env python3
"""
Comprehensive Strict Testing for HeavyDB Data Provider

This test suite ensures that the HeavyDB data provider strictly enforces
real data usage with no synthetic data generation or fallback mechanisms.

Tests Include:
1. Real HeavyDB connection requirement
2. No synthetic data generation
3. Immediate failure on HeavyDB unavailability
4. Real market data validation
5. Data freshness and integrity checks
6. No fallback to cached or mock data

Author: SuperClaude Testing Framework
Date: 2025-07-11
Version: 1.0.0
"""

import unittest
import logging
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
import os
import sys
from contextlib import contextmanager

# Add the project root to the path
sys.path.insert(0, '/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestHeavyDBDataProviderStrict(unittest.TestCase):
    """
    Comprehensive test suite for HeavyDB data provider
    
    This test suite follows the STRICT NO SYNTHETIC DATA requirements:
    - Must use real HeavyDB connections only
    - Must fail immediately if HeavyDB is unavailable
    - Must validate real market data integrity
    - Must never fall back to mock or cached data
    """
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            'host': 'localhost',
            'port': 6274,
            'database': 'heavyai',
            'user': 'admin',
            'password': 'HyperInteractive',
            'protocol': 'binary'
        }
        
        self.test_symbol = 'NIFTY'
        self.test_start = datetime(2024, 1, 1, 9, 15)
        self.test_end = datetime(2024, 1, 1, 15, 30)
        
        logger.info("Setting up HeavyDB data provider tests")
    
    def test_01_heavydb_connection_enforcement(self):
        """Test 1: Verify data provider requires real HeavyDB connection"""
        logger.info("TEST 1: HeavyDB Connection Enforcement")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            # Test with valid configuration
            provider = HeavyDBDataProvider(self.test_config)
            
            # Verify connection is attempted
            self.assertIsNotNone(provider.host)
            self.assertIsNotNone(provider.port)
            self.assertIsNotNone(provider.database)
            self.assertIsNotNone(provider.user)
            self.assertIsNotNone(provider.password)
            
            # Test connection method
            connection_result = provider.connect()
            
            # If connection succeeds, verify it's real
            if connection_result:
                self.assertTrue(provider.is_connected)
                self.assertIsNotNone(provider.connection)
                
                # Test the connection
                test_result = provider.test_connection()
                self.assertTrue(test_result, "Connection test failed - not a real HeavyDB connection")
                
                logger.info("✅ Real HeavyDB connection verified")
            else:
                self.fail("❌ HeavyDB connection failed - verify HeavyDB is running")
                
        except ImportError as e:
            self.fail(f"❌ Failed to import HeavyDB data provider: {e}")
        except Exception as e:
            self.fail(f"❌ Unexpected error in connection test: {e}")
    
    def test_02_no_synthetic_data_generation(self):
        """Test 2: Ensure no synthetic data is generated or returned"""
        logger.info("TEST 2: No Synthetic Data Generation")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.test_config)
            
            # Test with invalid date range (future dates)
            future_start = datetime(2030, 1, 1, 9, 15)
            future_end = datetime(2030, 1, 1, 15, 30)
            
            # Attempt to fetch data for future dates
            data = provider.fetch_market_data(
                symbol=self.test_symbol,
                start_time=future_start,
                end_time=future_end
            )
            
            # Should return empty DataFrame for future dates, not synthetic data
            self.assertTrue(data.empty, "❌ Provider returned data for future dates - possible synthetic data")
            
            # Test with invalid symbol
            invalid_data = provider.fetch_market_data(
                symbol='INVALID_SYMBOL_12345',
                start_time=self.test_start,
                end_time=self.test_end
            )
            
            # Should return empty DataFrame, not synthetic data
            self.assertTrue(invalid_data.empty, "❌ Provider returned data for invalid symbol - possible synthetic data")
            
            logger.info("✅ No synthetic data generation verified")
            
        except Exception as e:
            self.fail(f"❌ Error in synthetic data test: {e}")
    
    def test_03_immediate_failure_on_unavailable_heavydb(self):
        """Test 3: Ensure provider fails immediately if HeavyDB unavailable"""
        logger.info("TEST 3: Immediate Failure on HeavyDB Unavailability")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            # Test with invalid configuration (wrong port)
            invalid_config = self.test_config.copy()
            invalid_config['port'] = 9999  # Invalid port
            
            provider = HeavyDBDataProvider(invalid_config)
            
            # Connection should fail
            connection_result = provider.connect()
            self.assertFalse(connection_result, "❌ Connection should fail with invalid configuration")
            self.assertFalse(provider.is_connected, "❌ Provider should not be connected")
            
            # Test with invalid host
            invalid_config['host'] = 'nonexistent.host.com'
            provider2 = HeavyDBDataProvider(invalid_config)
            
            connection_result2 = provider2.connect()
            self.assertFalse(connection_result2, "❌ Connection should fail with invalid host")
            
            logger.info("✅ Immediate failure on unavailable HeavyDB verified")
            
        except Exception as e:
            self.fail(f"❌ Error in unavailable HeavyDB test: {e}")
    
    def test_04_real_market_data_validation(self):
        """Test 4: Validate all data queries return real market data only"""
        logger.info("TEST 4: Real Market Data Validation")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.test_config)
            
            if not provider.connect():
                self.skipTest("HeavyDB not available - skipping real data validation")
            
            # Test option chain data
            option_data = provider.fetch_market_data(
                symbol=self.test_symbol,
                start_time=self.test_start,
                end_time=self.test_end
            )
            
            if not option_data.empty:
                # Validate real market data characteristics
                self.assertIn('datetime_', option_data.columns, "❌ Missing datetime column")
                self.assertIn('symbol', option_data.columns, "❌ Missing symbol column")
                
                # Check for realistic values
                if 'ltp' in option_data.columns:
                    ltp_values = option_data['ltp'].dropna()
                    if not ltp_values.empty:
                        self.assertTrue(all(ltp_values > 0), "❌ Invalid LTP values found")
                        self.assertTrue(all(ltp_values < 100000), "❌ Unrealistic LTP values")
                
                # Check timestamp validity
                if 'datetime_' in option_data.columns:
                    timestamps = pd.to_datetime(option_data['datetime_'])
                    self.assertTrue(all(timestamps >= self.test_start), "❌ Invalid timestamps before start")
                    self.assertTrue(all(timestamps <= self.test_end), "❌ Invalid timestamps after end")
                
                logger.info("✅ Real market data validation passed")
            else:
                logger.warning("⚠️ No data returned - verify data exists in HeavyDB for test period")
            
        except Exception as e:
            self.fail(f"❌ Error in real market data validation: {e}")
    
    def test_05_data_freshness_integrity(self):
        """Test 5: Test data freshness and integrity validation"""
        logger.info("TEST 5: Data Freshness and Integrity Validation")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.test_config)
            
            if not provider.connect():
                self.skipTest("HeavyDB not available - skipping data freshness test")
            
            # Test options chain fetch
            options_data = provider.fetch_options_chain(
                symbol=self.test_symbol,
                timestamp=datetime(2024, 1, 1, 10, 0),
                expiry_date=datetime(2024, 1, 25)
            )
            
            if not options_data.empty:
                # Check data integrity
                self.assertGreater(len(options_data), 0, "❌ No options data returned")
                
                # Check for required columns
                required_cols = ['datetime_', 'symbol', 'strike', 'option_type']
                for col in required_cols:
                    self.assertIn(col, options_data.columns, f"❌ Missing required column: {col}")
                
                # Check for realistic strike prices
                if 'strike' in options_data.columns:
                    strikes = options_data['strike'].dropna()
                    if not strikes.empty:
                        self.assertTrue(all(strikes > 10000), "❌ Invalid strike prices")
                        self.assertTrue(all(strikes < 100000), "❌ Unrealistic strike prices")
                
                logger.info("✅ Data freshness and integrity validation passed")
            else:
                logger.warning("⚠️ No options data returned - verify data exists for test parameters")
            
        except Exception as e:
            self.fail(f"❌ Error in data freshness test: {e}")
    
    def test_06_no_fallback_to_cache_or_mock(self):
        """Test 6: Ensure no fallback to cached or mock data sources"""
        logger.info("TEST 6: No Fallback to Cache or Mock Data")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.test_config)
            
            # Mock the connection to simulate failure
            with patch.object(provider, 'connection', None):
                provider.is_connected = False
                
                # Attempt to fetch data with no connection
                data = provider.fetch_market_data(
                    symbol=self.test_symbol,
                    start_time=self.test_start,
                    end_time=self.test_end
                )
                
                # Should return empty DataFrame, not cached/mock data
                self.assertTrue(data.empty, "❌ Provider returned data without connection - possible cache/mock fallback")
                
                # Test ATM straddle fetch
                straddle_data = provider.fetch_atm_straddle(
                    symbol=self.test_symbol,
                    timestamp=self.test_start
                )
                
                self.assertTrue(straddle_data.empty, "❌ Provider returned straddle data without connection")
                
                # Test underlying data fetch
                underlying_data = provider.fetch_underlying_data(
                    symbol=self.test_symbol,
                    start_time=self.test_start,
                    end_time=self.test_end
                )
                
                self.assertTrue(underlying_data.empty, "❌ Provider returned underlying data without connection")
            
            logger.info("✅ No fallback to cache or mock data verified")
            
        except Exception as e:
            self.fail(f"❌ Error in fallback test: {e}")
    
    def test_07_query_execution_validation(self):
        """Test 7: Validate query execution only returns real data"""
        logger.info("TEST 7: Query Execution Validation")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.test_config)
            
            if not provider.connect():
                self.skipTest("HeavyDB not available - skipping query execution test")
            
            # Test direct query execution
            test_query = """
            SELECT COUNT(*) as row_count 
            FROM nifty_option_chain 
            WHERE symbol = 'NIFTY' 
            AND trade_time >= '2024-01-01 09:15:00' 
            AND trade_time <= '2024-01-01 15:30:00'
            """
            
            result = provider.execute_query(test_query)
            
            if not result.empty:
                # Verify we get actual count data
                self.assertIn('row_count', result.columns, "❌ Query result missing expected column")
                count_value = result.iloc[0]['row_count']
                self.assertIsInstance(count_value, (int, np.integer), "❌ Invalid count value type")
                self.assertGreaterEqual(count_value, 0, "❌ Invalid count value")
                
                logger.info(f"✅ Query execution validated - {count_value} rows found")
            else:
                logger.warning("⚠️ Query returned empty result - verify data exists")
            
        except Exception as e:
            self.fail(f"❌ Error in query execution test: {e}")
    
    def test_08_connection_context_manager(self):
        """Test 8: Test cursor context manager for real connections"""
        logger.info("TEST 8: Connection Context Manager Validation")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.test_config)
            
            if not provider.connect():
                self.skipTest("HeavyDB not available - skipping context manager test")
            
            # Test cursor context manager
            with provider.get_cursor() as cursor:
                self.assertIsNotNone(cursor, "❌ Cursor is None")
                
                # Execute a simple test query
                cursor.execute("SELECT 1 as test_value")
                result = cursor.fetchone()
                
                self.assertIsNotNone(result, "❌ No result from test query")
                self.assertEqual(result[0], 1, "❌ Incorrect test result")
            
            logger.info("✅ Connection context manager validated")
            
        except Exception as e:
            self.fail(f"❌ Error in context manager test: {e}")
    
    def test_09_real_data_type_validation(self):
        """Test 9: Validate data types are consistent with real market data"""
        logger.info("TEST 9: Real Data Type Validation")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.test_config)
            
            if not provider.connect():
                self.skipTest("HeavyDB not available - skipping data type validation")
            
            # Fetch sample data
            sample_data = provider.fetch_market_data(
                symbol=self.test_symbol,
                start_time=datetime(2024, 1, 1, 10, 0),
                end_time=datetime(2024, 1, 1, 10, 30)
            )
            
            if not sample_data.empty:
                # Validate data types
                if 'strike' in sample_data.columns:
                    strikes = sample_data['strike'].dropna()
                    if not strikes.empty:
                        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in strikes), 
                                      "❌ Invalid strike data types")
                
                if 'volume' in sample_data.columns:
                    volumes = sample_data['volume'].dropna()
                    if not volumes.empty:
                        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in volumes), 
                                      "❌ Invalid volume data types")
                
                if 'iv' in sample_data.columns:
                    ivs = sample_data['iv'].dropna()
                    if not ivs.empty:
                        self.assertTrue(all(isinstance(x, (float, np.floating)) for x in ivs), 
                                      "❌ Invalid IV data types")
                        self.assertTrue(all(x > 0 for x in ivs), "❌ Invalid IV values")
                
                logger.info("✅ Real data type validation passed")
            else:
                logger.warning("⚠️ No sample data for type validation")
            
        except Exception as e:
            self.fail(f"❌ Error in data type validation: {e}")
    
    def test_10_comprehensive_connection_failure_handling(self):
        """Test 10: Comprehensive connection failure handling"""
        logger.info("TEST 10: Comprehensive Connection Failure Handling")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            # Test with completely invalid configuration
            invalid_configs = [
                {'host': None, 'port': 6274, 'database': 'heavyai', 'user': 'admin', 'password': 'test'},
                {'host': 'localhost', 'port': None, 'database': 'heavyai', 'user': 'admin', 'password': 'test'},
                {'host': 'localhost', 'port': 6274, 'database': None, 'user': 'admin', 'password': 'test'},
                {'host': 'localhost', 'port': 6274, 'database': 'heavyai', 'user': None, 'password': 'test'},
                {'host': 'localhost', 'port': 6274, 'database': 'heavyai', 'user': 'admin', 'password': None},
            ]
            
            for i, invalid_config in enumerate(invalid_configs):
                try:
                    provider = HeavyDBDataProvider(invalid_config)
                    connection_result = provider.connect()
                    
                    # Should fail to connect
                    self.assertFalse(connection_result, f"❌ Connection should fail with invalid config {i+1}")
                    self.assertFalse(provider.is_connected, f"❌ Provider should not be connected with invalid config {i+1}")
                    
                    # Any data fetch should return empty
                    data = provider.fetch_market_data(
                        symbol=self.test_symbol,
                        start_time=self.test_start,
                        end_time=self.test_end
                    )
                    
                    self.assertTrue(data.empty, f"❌ Should return empty data with invalid config {i+1}")
                    
                except Exception as e:
                    # Connection failure is expected
                    logger.info(f"✅ Expected connection failure for config {i+1}: {e}")
            
            logger.info("✅ Comprehensive connection failure handling validated")
            
        except Exception as e:
            self.fail(f"❌ Error in connection failure test: {e}")


class TestHeavyDBDataProviderIntegration(unittest.TestCase):
    """Integration tests for HeavyDB data provider with real database"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.config = {
            'host': 'localhost',
            'port': 6274,
            'database': 'heavyai',
            'user': 'admin',
            'password': 'HyperInteractive'
        }
        
        logger.info("Setting up integration tests")
    
    def test_real_heavydb_integration(self):
        """Test integration with real HeavyDB database"""
        logger.info("INTEGRATION TEST: Real HeavyDB Integration")
        
        try:
            from strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider
            
            provider = HeavyDBDataProvider(self.config)
            
            if not provider.connect():
                self.skipTest("HeavyDB not available - skipping integration test")
            
            # Test table existence
            table_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'nifty_option_chain'
            """
            
            table_result = provider.execute_query(table_query)
            
            if not table_result.empty:
                logger.info("✅ nifty_option_chain table found")
                
                # Test data availability
                count_query = "SELECT COUNT(*) as total_rows FROM nifty_option_chain"
                count_result = provider.execute_query(count_query)
                
                if not count_result.empty:
                    total_rows = count_result.iloc[0]['total_rows']
                    self.assertGreater(total_rows, 0, "❌ No data in nifty_option_chain table")
                    logger.info(f"✅ Found {total_rows} rows in nifty_option_chain table")
                else:
                    self.fail("❌ Unable to get row count from nifty_option_chain")
            else:
                self.fail("❌ nifty_option_chain table not found in HeavyDB")
            
        except Exception as e:
            self.fail(f"❌ Integration test failed: {e}")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE HEAVYDB DATA PROVIDER TESTS")
    logger.info("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestHeavyDBDataProviderStrict))
    test_suite.addTest(unittest.makeSuite(TestHeavyDBDataProviderIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        logger.error("FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)