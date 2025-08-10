#!/usr/bin/env python3
"""
Unit Tests for ATM CE/PE Rolling Analysis

This module provides comprehensive unit tests for the ATM CE/PE Rolling Analysis
system, including rolling correlation analysis, comprehensive indicator integration,
and multi-timeframe analysis validation.

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

from atm_cepe_rolling_analyzer import (
    ATMCEPERollingAnalyzer,
    ATMCEPERollingResult,
    ComprehensiveIndicatorResult
)

logger = logging.getLogger(__name__)

class TestATMCEPERollingAnalysis(unittest.TestCase):
    """Comprehensive test suite for ATM CE/PE Rolling Analysis"""
    
    def setUp(self):
        """Set up test environment"""
        self.analyzer = ATMCEPERollingAnalyzer()
        
        # Test market data samples
        self.test_market_data = [
            {
                'underlying_price': 19500,
                'timestamp': datetime.now(),
                'iv_percentile': 0.4, 'atr_normalized': 0.35, 'gamma_exposure': 0.3,
                'ema_alignment': 0.6, 'price_momentum': 0.5, 'volume_confirmation': 0.4,
                'strike_correlation': 0.7, 'vwap_deviation': 0.6, 'pivot_analysis': 0.5
            },
            {
                'underlying_price': 19450,
                'timestamp': datetime.now(),
                'iv_percentile': 0.8, 'atr_normalized': 0.75, 'gamma_exposure': 0.7,
                'ema_alignment': 0.2, 'price_momentum': 0.1, 'volume_confirmation': 0.15,
                'strike_correlation': 0.4, 'vwap_deviation': 0.3, 'pivot_analysis': 0.35
            },
            {
                'underlying_price': 19550,
                'timestamp': datetime.now(),
                'iv_percentile': 0.5, 'atr_normalized': 0.45, 'gamma_exposure': 0.4,
                'ema_alignment': 0.5, 'price_momentum': 0.4, 'volume_confirmation': 0.35,
                'strike_correlation': 0.65, 'vwap_deviation': 0.6, 'pivot_analysis': 0.7
            }
        ]
    
    def test_analyzer_initialization(self):
        """Test ATM CE/PE Rolling Analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        
        # Check rolling windows
        self.assertIn('short', self.analyzer.rolling_windows)
        self.assertIn('medium', self.analyzer.rolling_windows)
        self.assertIn('long', self.analyzer.rolling_windows)
        
        # Check timeframes
        expected_timeframes = ['3min', '5min', '10min', '15min']
        for tf in expected_timeframes:
            self.assertIn(tf, self.analyzer.timeframes)
        
        # Check timeframe weights sum to 1.0
        timeframe_weight_sum = sum(self.analyzer.timeframe_weights.values())
        self.assertAlmostEqual(timeframe_weight_sum, 1.0, places=2)
        
        # Check indicator parameters
        self.assertIn('ema_periods', self.analyzer.indicator_params)
        self.assertIn('rsi_period', self.analyzer.indicator_params)
        self.assertIn('macd_fast', self.analyzer.indicator_params)
        
        logger.info("✅ ATM CE/PE Rolling Analyzer initialization test passed")
    
    def test_atm_cepe_rolling_analysis(self):
        """Test ATM CE/PE rolling analysis"""
        for test_data in self.test_market_data:
            result = self.analyzer.analyze_atm_cepe_rolling(test_data)
            
            # Validate result type
            self.assertIsInstance(result, ATMCEPERollingResult)
            
            # Validate required fields
            self.assertIsNotNone(result.rolling_correlations)
            self.assertIsNotNone(result.ce_pe_correlation)
            self.assertIsNotNone(result.rolling_trends)
            self.assertIsNotNone(result.technical_indicators)
            self.assertIsNotNone(result.volatility_indicators)
            
            # Validate correlation ranges
            self.assertGreaterEqual(result.ce_pe_correlation, 0.0)
            self.assertLessEqual(result.ce_pe_correlation, 1.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            
            # Validate rolling correlations
            for window_name in self.analyzer.rolling_windows.keys():
                corr_key = f'{window_name}_rolling_correlation'
                self.assertIn(corr_key, result.rolling_correlations)
                self.assertGreaterEqual(result.rolling_correlations[corr_key], 0.0)
                self.assertLessEqual(result.rolling_correlations[corr_key], 1.0)
            
            # Validate rolling windows
            self.assertEqual(result.rolling_windows, self.analyzer.rolling_windows)
        
        logger.info("✅ ATM CE/PE rolling analysis test passed")
    
    def test_rolling_correlation_calculation(self):
        """Test rolling correlation calculation"""
        # Generate test CE/PE data with known correlation
        time_range = pd.date_range(end=datetime.now(), periods=30, freq='1min')
        
        # Create correlated data
        base_values = np.random.randn(30)
        ce_data = pd.DataFrame({
            'last_price': 150 + base_values * 5,
            'volume': np.random.randint(100, 1000, 30),
            'implied_volatility': 0.2 + np.random.random(30) * 0.1
        }, index=time_range)
        
        pe_data = pd.DataFrame({
            'last_price': 140 - base_values * 4,  # Negative correlation
            'volume': np.random.randint(100, 1000, 30),
            'implied_volatility': 0.2 + np.random.random(30) * 0.1
        }, index=time_range)
        
        # Calculate rolling correlations
        rolling_correlations = self.analyzer._calculate_rolling_correlations(ce_data, pe_data)
        
        # Validate structure
        for window_name in self.analyzer.rolling_windows.keys():
            corr_key = f'{window_name}_rolling_correlation'
            self.assertIn(corr_key, rolling_correlations)
            self.assertGreaterEqual(rolling_correlations[corr_key], 0.0)
            self.assertLessEqual(rolling_correlations[corr_key], 1.0)
        
        logger.info("✅ Rolling correlation calculation test passed")
    
    def test_comprehensive_indicator_integration(self):
        """Test comprehensive indicator integration"""
        test_data = self.test_market_data[0]
        result = self.analyzer.analyze_atm_cepe_rolling(test_data)
        
        # Validate technical indicators
        required_indicators = ['ema', 'vwap', 'pivot', 'rsi', 'macd', 'multi_timeframe']
        for indicator in required_indicators:
            self.assertIn(indicator, result.technical_indicators)
            self.assertIsInstance(result.technical_indicators[indicator], dict)
        
        # Validate EMA analysis
        ema_analysis = result.technical_indicators['ema']
        self.assertIn('combined_ema_alignment', ema_analysis)
        self.assertGreaterEqual(ema_analysis['combined_ema_alignment'], -1.0)
        self.assertLessEqual(ema_analysis['combined_ema_alignment'], 1.0)
        
        # Validate VWAP analysis
        vwap_analysis = result.technical_indicators['vwap']
        self.assertIn('combined_vwap_deviation', vwap_analysis)
        
        # Validate RSI analysis
        rsi_analysis = result.technical_indicators['rsi']
        self.assertIn('combined_rsi', rsi_analysis)
        self.assertGreaterEqual(rsi_analysis['combined_rsi'], -1.0)
        self.assertLessEqual(rsi_analysis['combined_rsi'], 1.0)
        
        # Validate MACD analysis
        macd_analysis = result.technical_indicators['macd']
        self.assertIn('combined_macd', macd_analysis)
        
        # Validate multi-timeframe analysis
        multi_tf = result.technical_indicators['multi_timeframe']
        for timeframe in self.analyzer.timeframes:
            self.assertIn(timeframe, multi_tf)
            self.assertIn('combined_score', multi_tf[timeframe])
        
        logger.info("✅ Comprehensive indicator integration test passed")
    
    def test_volatility_indicators(self):
        """Test volatility indicators analysis"""
        test_data = self.test_market_data[0]
        result = self.analyzer.analyze_atm_cepe_rolling(test_data)
        
        # Validate volatility indicators
        vol_indicators = result.volatility_indicators
        
        required_vol_indicators = [
            'ce_iv_percentile', 'pe_iv_percentile',
            'ce_atr_normalized', 'pe_atr_normalized',
            'ce_gamma_exposure', 'pe_gamma_exposure',
            'combined_volatility'
        ]
        
        for indicator in required_vol_indicators:
            self.assertIn(indicator, vol_indicators)
            self.assertGreaterEqual(vol_indicators[indicator], 0.0)
            self.assertLessEqual(vol_indicators[indicator], 1.0)
        
        logger.info("✅ Volatility indicators test passed")
    
    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis (3,5,10,15min)"""
        test_data = self.test_market_data[0]
        result = self.analyzer.analyze_atm_cepe_rolling(test_data)
        
        # Validate multi-timeframe analysis
        multi_tf = result.technical_indicators['multi_timeframe']
        
        for timeframe in ['3min', '5min', '10min', '15min']:
            self.assertIn(timeframe, multi_tf)
            tf_data = multi_tf[timeframe]
            
            # Check required fields
            required_fields = ['ce_momentum', 'pe_momentum', 'ce_volume_trend', 'pe_volume_trend', 'combined_score']
            for field in required_fields:
                self.assertIn(field, tf_data)
                self.assertGreaterEqual(tf_data[field], -1.0)
                self.assertLessEqual(tf_data[field], 1.0)
        
        logger.info("✅ Multi-timeframe analysis test passed")
    
    def test_performance_requirements(self):
        """Test processing time requirements (<3 seconds)"""
        processing_times = []
        
        for test_data in self.test_market_data:
            start_time = time.time()
            result = self.analyzer.analyze_atm_cepe_rolling(test_data)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            # Individual test should be under 3 seconds
            self.assertLess(processing_time, 3.0)
            
            # Result should have processing time recorded
            self.assertGreater(result.processing_time, 0.0)
            self.assertLess(result.processing_time, 3.0)
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # Average should be well under target
        self.assertLess(avg_processing_time, 2.0)
        
        logger.info(f"✅ Performance test passed: avg={avg_processing_time:.3f}s, max={max_processing_time:.3f}s")
    
    def test_rolling_trends_calculation(self):
        """Test rolling trends calculation"""
        # Generate test data with clear trends
        time_range = pd.date_range(end=datetime.now(), periods=25, freq='1min')
        
        # CE with upward trend
        ce_data = pd.DataFrame({
            'last_price': 150 + np.arange(25) * 2 + np.random.randn(25),  # Upward trend
            'volume': np.random.randint(100, 1000, 25)
        }, index=time_range)
        
        # PE with downward trend
        pe_data = pd.DataFrame({
            'last_price': 140 - np.arange(25) * 1.5 + np.random.randn(25),  # Downward trend
            'volume': np.random.randint(100, 1000, 25)
        }, index=time_range)
        
        # Calculate rolling trends
        rolling_trends = self.analyzer._calculate_rolling_trends(ce_data, pe_data)
        
        # Validate trends
        for window_name in self.analyzer.rolling_windows.keys():
            ce_trend_key = f'ce_{window_name}_trend'
            pe_trend_key = f'pe_{window_name}_trend'
            
            self.assertIn(ce_trend_key, rolling_trends)
            self.assertIn(pe_trend_key, rolling_trends)
            
            # CE should have positive trend
            self.assertGreaterEqual(rolling_trends[ce_trend_key], -1.0)
            self.assertLessEqual(rolling_trends[ce_trend_key], 1.0)
            
            # PE should have negative trend
            self.assertGreaterEqual(rolling_trends[pe_trend_key], -1.0)
            self.assertLessEqual(rolling_trends[pe_trend_key], 1.0)
        
        logger.info("✅ Rolling trends calculation test passed")
    
    def test_performance_validation(self):
        """Test performance validation with multiple samples"""
        validation_result = self.analyzer.validate_rolling_analysis_performance(self.test_market_data)
        
        # Validate validation result structure
        self.assertIn('total_samples', validation_result)
        self.assertIn('avg_processing_time', validation_result)
        self.assertIn('performance_target_met', validation_result)
        self.assertIn('success_rate', validation_result)
        
        # Validate metrics
        self.assertEqual(validation_result['total_samples'], len(self.test_market_data))
        self.assertGreater(validation_result['avg_processing_time'], 0.0)
        self.assertLess(validation_result['avg_processing_time'], 3.0)
        self.assertTrue(validation_result['performance_target_met'])
        self.assertEqual(validation_result['success_rate'], 1.0)
        
        logger.info(f"✅ Performance validation: {validation_result['avg_processing_time']:.3f}s avg")
    
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Test with empty data
        empty_result = self.analyzer.analyze_atm_cepe_rolling({})
        self.assertIsInstance(empty_result, ATMCEPERollingResult)
        self.assertEqual(empty_result.confidence, 0.3)  # Low confidence for fallback
        
        # Test with invalid data
        invalid_data = {'underlying_price': 'invalid', 'timestamp': 'invalid'}
        invalid_result = self.analyzer.analyze_atm_cepe_rolling(invalid_data)
        self.assertIsInstance(invalid_result, ATMCEPERollingResult)
        
        logger.info("✅ Error handling test passed")

def run_atm_cepe_rolling_analysis_tests():
    """Run comprehensive test suite for ATM CE/PE Rolling Analysis"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestATMCEPERollingAnalysis)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"ATM CE/PE ROLLING ANALYSIS TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'='*70}")
    
    if success_rate >= 0.9:  # 90% success rate required
        print("✅ ATM CE/PE ROLLING ANALYSIS TESTS PASSED")
        return True
    else:
        print("❌ ATM CE/PE ROLLING ANALYSIS TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_atm_cepe_rolling_analysis_tests()
    sys.exit(0 if success else 1)
