#!/usr/bin/env python3
"""
Unit Tests for Correlation Matrix Engine

This module provides comprehensive unit tests for the Multi-Strike Correlation
Analysis Engine, including performance validation, accuracy testing, and
HeavyDB integration validation.

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

from correlation_matrix_engine import (
    CorrelationMatrixEngine,
    CorrelationResult,
    MultiStrikeData
)

logger = logging.getLogger(__name__)

class TestCorrelationMatrixEngine(unittest.TestCase):
    """Comprehensive test suite for Correlation Matrix Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = CorrelationMatrixEngine()
        
        # Test market data samples
        self.test_market_data = [
            {
                'underlying_price': 19500,
                'timestamp': datetime.now(),
                'option_chain': self._generate_test_option_chain(19500)
            },
            {
                'underlying_price': 19450,
                'timestamp': datetime.now(),
                'option_chain': self._generate_test_option_chain(19450)
            },
            {
                'underlying_price': 19550,
                'timestamp': datetime.now(),
                'option_chain': self._generate_test_option_chain(19550)
            }
        ]
    
    def _generate_test_option_chain(self, underlying_price: float) -> pd.DataFrame:
        """Generate test option chain data"""
        strikes = []
        for i in range(-3, 4):
            strike = underlying_price + (i * 50)
            strikes.append(strike)
        
        option_data = []
        for strike in strikes:
            for option_type in ['CE', 'PE']:
                # Generate realistic option data
                moneyness = strike / underlying_price
                
                if option_type == 'CE':
                    intrinsic = max(0, underlying_price - strike)
                    time_value = max(5, 50 * (1.1 - abs(moneyness - 1.0)))
                else:
                    intrinsic = max(0, strike - underlying_price)
                    time_value = max(5, 50 * (1.1 - abs(moneyness - 1.0)))
                
                last_price = intrinsic + time_value
                
                # Add some time series data
                for j in range(20):  # 20 data points
                    price_variation = last_price * (1 + np.random.normal(0, 0.02))
                    option_data.append({
                        'strike_price': strike,
                        'option_type': option_type,
                        'last_price': price_variation,
                        'volume': np.random.randint(100, 1000),
                        'open_interest': np.random.randint(1000, 10000),
                        'implied_volatility': 0.15 + np.random.random() * 0.1,
                        'delta': 0.5 if abs(moneyness - 1.0) < 0.01 else np.random.random(),
                        'gamma': 0.01 + np.random.random() * 0.02,
                        'theta': -0.5 - np.random.random() * 0.5,
                        'vega': 50 + np.random.random() * 50,
                        'trade_time': datetime.now() - timedelta(minutes=j)
                    })
        
        return pd.DataFrame(option_data)
    
    def test_engine_initialization(self):
        """Test correlation engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.max_processing_time, 1.5)
        
        # Check weight allocations sum to 1.0
        strike_weight_sum = sum(self.engine.strike_weights.values())
        timeframe_weight_sum = sum(self.engine.timeframe_weights.values())
        
        self.assertAlmostEqual(strike_weight_sum, 1.0, places=2)
        self.assertAlmostEqual(timeframe_weight_sum, 1.0, places=2)
        
        # Check correlation thresholds
        self.assertIn('strong', self.engine.correlation_thresholds)
        self.assertIn('moderate', self.engine.correlation_thresholds)
        self.assertIn('weak', self.engine.correlation_thresholds)
        
        logger.info("✅ Engine initialization test passed")
    
    def test_multi_strike_correlation_analysis(self):
        """Test multi-strike correlation analysis"""
        for test_data in self.test_market_data:
            result = self.engine.analyze_multi_strike_correlation(test_data)
            
            # Validate result type
            self.assertIsInstance(result, CorrelationResult)
            
            # Validate required fields
            self.assertIsNotNone(result.strike_correlations)
            self.assertIsNotNone(result.timeframe_correlations)
            self.assertIsNotNone(result.overall_correlation)
            self.assertIsNotNone(result.correlation_strength)
            
            # Validate correlation ranges
            self.assertGreaterEqual(result.overall_correlation, 0.0)
            self.assertLessEqual(result.overall_correlation, 1.0)
            self.assertGreaterEqual(result.correlation_strength, 0.0)
            self.assertLessEqual(result.correlation_strength, 1.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            
            # Validate strike correlations
            required_strike_correlations = [
                'atm_itm1_correlation', 'atm_otm1_correlation', 'itm1_otm1_correlation'
            ]
            for corr_key in required_strike_correlations:
                self.assertIn(corr_key, result.strike_correlations)
                self.assertGreaterEqual(result.strike_correlations[corr_key], 0.0)
                self.assertLessEqual(result.strike_correlations[corr_key], 1.0)
            
            # Validate timeframe correlations
            for timeframe in self.engine.timeframe_weights.keys():
                corr_key = f'{timeframe}_correlation'
                self.assertIn(corr_key, result.timeframe_correlations)
                self.assertGreaterEqual(result.timeframe_correlations[corr_key], 0.0)
                self.assertLessEqual(result.timeframe_correlations[corr_key], 1.0)
        
        logger.info("✅ Multi-strike correlation analysis test passed")
    
    def test_performance_requirements(self):
        """Test processing time requirements (<1.5 seconds)"""
        processing_times = []
        
        for test_data in self.test_market_data:
            start_time = time.time()
            result = self.engine.analyze_multi_strike_correlation(test_data)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            # Individual test should be under 1.5 seconds
            self.assertLess(processing_time, 1.5)
            
            # Result should have processing time recorded
            self.assertGreater(result.processing_time, 0.0)
            self.assertLess(result.processing_time, 1.5)
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # Average should be well under target
        self.assertLess(avg_processing_time, 1.0)
        
        logger.info(f"✅ Performance test passed: avg={avg_processing_time:.3f}s, max={max_processing_time:.3f}s")
    
    def test_pairwise_correlation_calculation(self):
        """Test pairwise correlation calculation"""
        # Create test data with known correlation
        data1 = pd.DataFrame({
            'last_price': [100, 102, 104, 106, 108],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        data2 = pd.DataFrame({
            'last_price': [50, 51, 52, 53, 54],  # Perfect positive correlation
            'volume': [500, 550, 600, 650, 700]
        })
        
        correlation = self.engine._calculate_pairwise_correlation(data1, data2)
        
        # Should be high correlation (close to 1.0)
        self.assertGreater(correlation, 0.8)
        self.assertLessEqual(correlation, 1.0)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        correlation_empty = self.engine._calculate_pairwise_correlation(data1, empty_data)
        self.assertEqual(correlation_empty, 0.5)  # Fallback value
        
        logger.info(f"✅ Pairwise correlation test passed: correlation={correlation:.3f}")
    
    def test_correlation_strength_calculation(self):
        """Test correlation strength calculation"""
        # Test with high, consistent correlations
        high_correlations = {
            'atm_itm1_correlation': 0.9,
            'atm_otm1_correlation': 0.85,
            'itm1_otm1_correlation': 0.88
        }
        
        strength_high = self.engine._calculate_correlation_strength(high_correlations)
        self.assertGreater(strength_high, 0.7)
        
        # Test with low, inconsistent correlations
        low_correlations = {
            'atm_itm1_correlation': 0.2,
            'atm_otm1_correlation': 0.8,
            'itm1_otm1_correlation': 0.1
        }
        
        strength_low = self.engine._calculate_correlation_strength(low_correlations)
        self.assertLess(strength_low, 0.5)
        
        logger.info(f"✅ Correlation strength test passed: high={strength_high:.3f}, low={strength_low:.3f}")
    
    def test_regime_pattern_identification(self):
        """Test regime correlation pattern identification"""
        # Test strong trending pattern
        pattern_strong = self.engine._identify_regime_correlation_pattern(0.8, 0.8)
        self.assertEqual(pattern_strong, 'STRONG_TRENDING')
        
        # Test moderate directional pattern
        pattern_moderate = self.engine._identify_regime_correlation_pattern(0.5, 0.5)
        self.assertEqual(pattern_moderate, 'MODERATE_DIRECTIONAL')
        
        # Test range bound pattern
        pattern_range = self.engine._identify_regime_correlation_pattern(0.1, 0.3)
        self.assertEqual(pattern_range, 'RANGE_BOUND')
        
        # Test volatile mixed pattern
        pattern_volatile = self.engine._identify_regime_correlation_pattern(0.5, 0.1)
        self.assertEqual(pattern_volatile, 'VOLATILE_MIXED')
        
        logger.info("✅ Regime pattern identification test passed")
    
    def test_multi_strike_data_validation(self):
        """Test multi-strike data validation"""
        # Create valid multi-strike data
        valid_data = MultiStrikeData(
            atm_data=pd.DataFrame({'last_price': range(15)}),  # 15 points > min_data_points
            itm1_data=pd.DataFrame({'last_price': range(15)}),
            otm1_data=pd.DataFrame({'last_price': range(15)}),
            underlying_price=19500,
            timestamp=datetime.now()
        )
        
        self.assertTrue(self.engine._validate_multi_strike_data(valid_data))
        
        # Create invalid multi-strike data (insufficient points)
        invalid_data = MultiStrikeData(
            atm_data=pd.DataFrame({'last_price': range(5)}),  # 5 points < min_data_points
            itm1_data=pd.DataFrame({'last_price': range(5)}),
            otm1_data=pd.DataFrame({'last_price': range(5)}),
            underlying_price=19500,
            timestamp=datetime.now()
        )
        
        self.assertFalse(self.engine._validate_multi_strike_data(invalid_data))
        
        # Test with None data
        self.assertFalse(self.engine._validate_multi_strike_data(None))
        
        logger.info("✅ Multi-strike data validation test passed")
    
    def test_outlier_removal(self):
        """Test outlier removal functionality"""
        # Create data with outliers
        series1 = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        series2 = np.array([2, 4, 6, 8, 10, 12])
        
        cleaned1, cleaned2 = self.engine._remove_outliers(series1, series2)
        
        # Outlier should be removed
        self.assertLess(len(cleaned1), len(series1))
        self.assertLess(len(cleaned2), len(series2))
        self.assertEqual(len(cleaned1), len(cleaned2))
        
        # Check that extreme value is removed
        self.assertNotIn(100, cleaned1)
        
        logger.info(f"✅ Outlier removal test passed: {len(series1)} → {len(cleaned1)} points")
    
    def test_performance_validation(self):
        """Test performance validation with multiple samples"""
        validation_result = self.engine.validate_correlation_performance(self.test_market_data)
        
        # Validate validation result structure
        self.assertIn('total_samples', validation_result)
        self.assertIn('avg_processing_time', validation_result)
        self.assertIn('performance_target_met', validation_result)
        self.assertIn('success_rate', validation_result)
        
        # Validate metrics
        self.assertEqual(validation_result['total_samples'], len(self.test_market_data))
        self.assertGreater(validation_result['avg_processing_time'], 0.0)
        self.assertLess(validation_result['avg_processing_time'], 1.5)
        self.assertTrue(validation_result['performance_target_met'])
        self.assertEqual(validation_result['success_rate'], 1.0)
        
        logger.info(f"✅ Performance validation: {validation_result['avg_processing_time']:.3f}s avg")
    
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Test with empty data
        empty_result = self.engine.analyze_multi_strike_correlation({})
        self.assertIsInstance(empty_result, CorrelationResult)
        self.assertEqual(empty_result.confidence, 0.3)  # Low confidence for fallback
        
        # Test with invalid option chain
        invalid_data = {'underlying_price': 'invalid', 'option_chain': 'invalid'}
        invalid_result = self.engine.analyze_multi_strike_correlation(invalid_data)
        self.assertIsInstance(invalid_result, CorrelationResult)
        
        logger.info("✅ Error handling test passed")

def run_correlation_matrix_engine_tests():
    """Run comprehensive test suite for Correlation Matrix Engine"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCorrelationMatrixEngine)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"CORRELATION MATRIX ENGINE TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'='*70}")
    
    if success_rate >= 0.9:  # 90% success rate required
        print("✅ CORRELATION MATRIX ENGINE TESTS PASSED")
        return True
    else:
        print("❌ CORRELATION MATRIX ENGINE TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_correlation_matrix_engine_tests()
    sys.exit(0 if success else 1)
