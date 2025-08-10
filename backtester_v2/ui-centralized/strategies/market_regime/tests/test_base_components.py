"""
Base Components Test Suite
=========================

Comprehensive tests for base infrastructure components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import logging

# Import base components
from ..base.common_utils import (
    DataValidator, MathUtils, TimeUtils, OptionUtils, 
    ConfigUtils, ErrorHandler, CacheUtils
)


class TestDataValidator(unittest.TestCase):
    """Test DataValidator functionality"""
    
    def setUp(self):
        self.validator = DataValidator()
        
        # Create sample option data
        self.sample_option_data = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'option_type': ['CE', 'PE', 'CE', 'PE', 'CE'],
            'spot': [110, 110, 110, 110, 110],
            'volume': [100, 200, 150, 300, 50],
            'oi': [1000, 2000, 1500, 3000, 500],
            'iv': [0.20, 0.22, 0.19, 0.25, 0.18],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
        
        # Create sample price data
        self.sample_price_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1200, 800, 1500, 900],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
    
    def test_validate_option_data_valid(self):
        """Test validation with valid option data"""
        result = self.validator.validate_option_data(self.sample_option_data)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertGreater(result['data_quality_score'], 0.8)
    
    def test_validate_option_data_missing_columns(self):
        """Test validation with missing required columns"""
        incomplete_data = self.sample_option_data.drop(columns=['strike'])
        result = self.validator.validate_option_data(incomplete_data)
        
        self.assertFalse(result['is_valid'])
        self.assertIn('Missing required columns', str(result['errors']))
    
    def test_validate_option_data_negative_values(self):
        """Test validation with negative values"""
        invalid_data = self.sample_option_data.copy()
        invalid_data.loc[0, 'volume'] = -100
        
        result = self.validator.validate_option_data(invalid_data)
        self.assertIn('Negative volume values', str(result['warnings']))
    
    def test_validate_price_data_valid(self):
        """Test validation with valid price data"""
        result = self.validator.validate_price_data(self.sample_price_data)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_numerical_columns(self):
        """Test numerical columns validation"""
        result = self.validator.validate_numerical_columns(
            self.sample_option_data, ['volume', 'oi', 'iv']
        )
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['total_missing'], 0)


class TestMathUtils(unittest.TestCase):
    """Test MathUtils functionality"""
    
    def setUp(self):
        self.math_utils = MathUtils()
    
    def test_safe_divide_normal(self):
        """Test safe division with normal values"""
        result = self.math_utils.safe_divide(10, 2)
        self.assertEqual(result, 5.0)
    
    def test_safe_divide_by_zero(self):
        """Test safe division by zero"""
        result = self.math_utils.safe_divide(10, 0)
        self.assertEqual(result, 0.0)
        
        result_with_default = self.math_utils.safe_divide(10, 0, default=99.0)
        self.assertEqual(result_with_default, 99.0)
    
    def test_safe_divide_nan_values(self):
        """Test safe division with NaN values"""
        result = self.math_utils.safe_divide(10, np.nan)
        self.assertEqual(result, 0.0)
    
    def test_calculate_percentage_change(self):
        """Test percentage change calculation"""
        result = self.math_utils.calculate_percentage_change(100, 110)
        self.assertAlmostEqual(result, 10.0, places=2)
        
        result_negative = self.math_utils.calculate_percentage_change(110, 100)
        self.assertAlmostEqual(result_negative, -9.09, places=2)
    
    def test_calculate_zscore(self):
        """Test z-score calculation"""
        values = [1, 2, 3, 4, 5]
        result = self.math_utils.calculate_zscore(values, 3)
        self.assertAlmostEqual(result, 0.0, places=2)
        
        result_high = self.math_utils.calculate_zscore(values, 5)
        self.assertGreater(result_high, 1.0)
    
    def test_moving_average(self):
        """Test moving average calculation"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.math_utils.moving_average(values, window=3)
        
        expected_length = len(values) - 3 + 1
        self.assertEqual(len(result), expected_length)
        self.assertAlmostEqual(result[0], 2.0, places=2)  # (1+2+3)/3
    
    def test_calculate_correlation(self):
        """Test correlation calculation"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # Perfect positive correlation
        
        result = self.math_utils.calculate_correlation(x, y)
        self.assertAlmostEqual(result, 1.0, places=2)
    
    def test_normalize_values(self):
        """Test value normalization"""
        values = [1, 2, 3, 4, 5]
        result = self.math_utils.normalize_values(values)
        
        # Check that mean is approximately 0 and std is approximately 1
        self.assertAlmostEqual(np.mean(result), 0.0, places=5)
        self.assertAlmostEqual(np.std(result), 1.0, places=5)


class TestTimeUtils(unittest.TestCase):
    """Test TimeUtils functionality"""
    
    def setUp(self):
        self.time_utils = TimeUtils()
        self.sample_timestamp = datetime(2024, 1, 15, 9, 30, 0)
    
    def test_get_market_session(self):
        """Test market session identification"""
        # Pre-market
        pre_market_time = datetime(2024, 1, 15, 8, 30, 0)
        session = self.time_utils.get_market_session(pre_market_time)
        self.assertEqual(session, 'pre_market')
        
        # Regular trading
        regular_time = datetime(2024, 1, 15, 10, 30, 0)
        session = self.time_utils.get_market_session(regular_time)
        self.assertEqual(session, 'regular')
        
        # Post-market
        post_market_time = datetime(2024, 1, 15, 16, 30, 0)
        session = self.time_utils.get_market_session(post_market_time)
        self.assertEqual(session, 'post_market')
    
    def test_is_market_open(self):
        """Test market open check"""
        # Market open time
        market_open_time = datetime(2024, 1, 15, 10, 0, 0)  # Monday 10 AM
        self.assertTrue(self.time_utils.is_market_open(market_open_time))
        
        # Weekend
        weekend_time = datetime(2024, 1, 13, 10, 0, 0)  # Saturday
        self.assertFalse(self.time_utils.is_market_open(weekend_time))
    
    def test_calculate_dte(self):
        """Test days to expiry calculation"""
        current_date = datetime(2024, 1, 15)
        expiry_date = datetime(2024, 1, 25)
        
        dte = self.time_utils.calculate_dte(current_date, expiry_date)
        self.assertEqual(dte, 10)
    
    def test_get_next_trading_day(self):
        """Test next trading day calculation"""
        friday = datetime(2024, 1, 12)  # Friday
        next_trading = self.time_utils.get_next_trading_day(friday)
        
        # Should be Monday
        self.assertEqual(next_trading.weekday(), 0)  # Monday is 0
    
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        formatted = self.time_utils.format_timestamp(self.sample_timestamp)
        self.assertIsInstance(formatted, str)
        self.assertIn('2024-01-15', formatted)


class TestOptionUtils(unittest.TestCase):
    """Test OptionUtils functionality"""
    
    def setUp(self):
        self.option_utils = OptionUtils()
    
    def test_calculate_moneyness(self):
        """Test moneyness calculation"""
        # ATM
        moneyness = self.option_utils.calculate_moneyness(100, 100)
        self.assertAlmostEqual(moneyness, 1.0, places=2)
        
        # ITM Call / OTM Put
        moneyness = self.option_utils.calculate_moneyness(95, 100)
        self.assertAlmostEqual(moneyness, 0.95, places=2)
        
        # OTM Call / ITM Put
        moneyness = self.option_utils.calculate_moneyness(105, 100)
        self.assertAlmostEqual(moneyness, 1.05, places=2)
    
    def test_classify_option_position(self):
        """Test option position classification"""
        # ATM
        position = self.option_utils.classify_option_position(100, 100, 'CE')
        self.assertEqual(position, 'ATM')
        
        # ITM Call
        position = self.option_utils.classify_option_position(95, 100, 'CE')
        self.assertEqual(position, 'ITM')
        
        # OTM Call
        position = self.option_utils.classify_option_position(105, 100, 'CE')
        self.assertEqual(position, 'OTM')
        
        # ITM Put
        position = self.option_utils.classify_option_position(105, 100, 'PE')
        self.assertEqual(position, 'ITM')
        
        # OTM Put
        position = self.option_utils.classify_option_position(95, 100, 'PE')
        self.assertEqual(position, 'OTM')
    
    def test_filter_liquid_options(self):
        """Test liquid options filtering"""
        option_data = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'volume': [100, 5, 150, 300, 2],  # Some low volume
            'oi': [1000, 20, 1500, 3000, 10],  # Some low OI
            'option_type': ['CE', 'PE', 'CE', 'PE', 'CE']
        })
        
        filtered = self.option_utils.filter_liquid_options(
            option_data, min_volume=50, min_oi=100
        )
        
        # Should filter out rows with low volume or OI
        self.assertEqual(len(filtered), 3)  # Only rows 0, 2, 3 should remain
    
    def test_calculate_time_decay(self):
        """Test time decay calculation"""
        current_date = datetime(2024, 1, 15)
        expiry_date = datetime(2024, 1, 25)
        
        time_decay = self.option_utils.calculate_time_decay(current_date, expiry_date)
        self.assertIsInstance(time_decay, float)
        self.assertGreater(time_decay, 0)
        self.assertLess(time_decay, 1)


class TestConfigUtils(unittest.TestCase):
    """Test ConfigUtils functionality"""
    
    def setUp(self):
        self.config_utils = ConfigUtils()
        
        self.sample_config = {
            'indicators': {
                'rsi': {'period': 14, 'overbought': 70},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            'risk': {
                'max_position_size': 1.0,
                'stop_loss': 0.02
            }
        }
    
    def test_get_nested_value(self):
        """Test nested value retrieval"""
        value = self.config_utils.get_nested_value(
            self.sample_config, 'indicators.rsi.period'
        )
        self.assertEqual(value, 14)
        
        # Test with default
        value = self.config_utils.get_nested_value(
            self.sample_config, 'indicators.rsi.missing', default=20
        )
        self.assertEqual(value, 20)
    
    def test_merge_configs(self):
        """Test configuration merging"""
        base_config = {'a': 1, 'b': {'c': 2, 'd': 3}}
        override_config = {'b': {'c': 4, 'e': 5}, 'f': 6}
        
        merged = self.config_utils.merge_configs(base_config, override_config)
        
        self.assertEqual(merged['a'], 1)
        self.assertEqual(merged['b']['c'], 4)  # Overridden
        self.assertEqual(merged['b']['d'], 3)  # Preserved
        self.assertEqual(merged['b']['e'], 5)  # Added
        self.assertEqual(merged['f'], 6)  # Added
    
    def test_validate_config_structure(self):
        """Test configuration structure validation"""
        required_structure = {
            'indicators': {
                'rsi': ['period', 'overbought', 'oversold'],
                'macd': ['fast', 'slow', 'signal']
            },
            'risk': ['max_position_size']
        }
        
        # Valid config
        is_valid = self.config_utils.validate_config_structure(
            self.sample_config, required_structure
        )
        self.assertTrue(is_valid)
        
        # Invalid config (missing required field)
        invalid_config = self.sample_config.copy()
        del invalid_config['indicators']['rsi']['period']
        
        is_valid = self.config_utils.validate_config_structure(
            invalid_config, required_structure
        )
        self.assertFalse(is_valid)


class TestErrorHandler(unittest.TestCase):
    """Test ErrorHandler functionality"""
    
    def setUp(self):
        self.error_handler = ErrorHandler()
    
    def test_retry_on_failure_success(self):
        """Test retry mechanism with successful execution"""
        mock_func = Mock(return_value="success")
        
        result = self.error_handler.retry_on_failure(mock_func, max_retries=3)
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 1)
    
    def test_retry_on_failure_eventual_success(self):
        """Test retry mechanism with eventual success"""
        mock_func = Mock(side_effect=[Exception("error"), Exception("error"), "success"])
        
        result = self.error_handler.retry_on_failure(mock_func, max_retries=3)
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 3)
    
    def test_retry_on_failure_max_retries_exceeded(self):
        """Test retry mechanism when max retries exceeded"""
        mock_func = Mock(side_effect=Exception("persistent error"))
        
        with self.assertRaises(Exception):
            self.error_handler.retry_on_failure(mock_func, max_retries=2)
        
        self.assertEqual(mock_func.call_count, 2)
    
    def test_safe_execute_success(self):
        """Test safe execution with success"""
        def test_function():
            return "success"
        
        result = self.error_handler.safe_execute(test_function)
        
        self.assertEqual(result['success'], True)
        self.assertEqual(result['result'], "success")
        self.assertIsNone(result['error'])
    
    def test_safe_execute_with_error(self):
        """Test safe execution with error"""
        def failing_function():
            raise ValueError("test error")
        
        result = self.error_handler.safe_execute(failing_function)
        
        self.assertEqual(result['success'], False)
        self.assertIsNone(result['result'])
        self.assertIsInstance(result['error'], ValueError)


class TestCacheUtils(unittest.TestCase):
    """Test CacheUtils functionality"""
    
    def setUp(self):
        self.cache = CacheUtils(max_size=3, ttl=60)
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations"""
        self.cache.set('key1', 'value1')
        result = self.cache.get('key1')
        
        self.assertEqual(result, 'value1')
    
    def test_cache_miss(self):
        """Test cache miss"""
        result = self.cache.get('nonexistent_key')
        self.assertIsNone(result)
        
        # Test with default
        result = self.cache.get('nonexistent_key', default='default_value')
        self.assertEqual(result, 'default_value')
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement"""
        # Fill cache beyond limit
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')
        self.cache.set('key3', 'value3')
        self.cache.set('key4', 'value4')  # Should evict oldest
        
        # key1 should be evicted
        self.assertIsNone(self.cache.get('key1'))
        self.assertEqual(self.cache.get('key4'), 'value4')
        self.assertEqual(len(self.cache._cache), 3)
    
    def test_cache_ttl_expiry(self):
        """Test cache TTL expiry"""
        # Use a cache with very short TTL
        short_cache = CacheUtils(max_size=10, ttl=0.1)  # 0.1 seconds
        
        short_cache.set('key1', 'value1')
        
        # Should be available immediately
        self.assertEqual(short_cache.get('key1'), 'value1')
        
        # Wait for expiry and check
        import time
        time.sleep(0.2)
        self.assertIsNone(short_cache.get('key1'))
    
    def test_cache_clear(self):
        """Test cache clear operation"""
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')
        
        self.assertEqual(len(self.cache._cache), 2)
        
        self.cache.clear()
        
        self.assertEqual(len(self.cache._cache), 0)
        self.assertIsNone(self.cache.get('key1'))
    
    def test_cache_contains(self):
        """Test cache contains operation"""
        self.cache.set('key1', 'value1')
        
        self.assertTrue(self.cache.contains('key1'))
        self.assertFalse(self.cache.contains('key2'))


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataValidator,
        TestMathUtils,
        TestTimeUtils,
        TestOptionUtils,
        TestConfigUtils,
        TestErrorHandler,
        TestCacheUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Base Components Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")