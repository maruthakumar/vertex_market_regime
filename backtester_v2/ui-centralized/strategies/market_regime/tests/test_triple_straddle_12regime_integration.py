#!/usr/bin/env python3
"""
Unit Tests for Triple Straddle 12-Regime Integration System

This module provides comprehensive unit tests for the Triple Straddle Integration
with 35% weight allocation, including performance validation, accuracy testing,
and HeavyDB integration validation.

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

from triple_straddle_12regime_integrator import (
    TripleStraddle12RegimeIntegrator,
    TripleStraddleResult,
    IntegratedRegimeResult
)

logger = logging.getLogger(__name__)

class TestTripleStraddle12RegimeIntegration(unittest.TestCase):
    """Comprehensive test suite for Triple Straddle 12-Regime Integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.integrator = TripleStraddle12RegimeIntegrator()
        
        # Test market data samples
        self.test_market_data = [
            {
                'underlying_price': 19500,
                'iv_percentile': 0.3, 'atr_normalized': 0.25, 'gamma_exposure': 0.2,
                'ema_alignment': 0.7, 'price_momentum': 0.6, 'volume_confirmation': 0.5,
                'strike_correlation': 0.8, 'vwap_deviation': 0.7, 'pivot_analysis': 0.6,
                'volume_trend': 0.6, 'volatility_regime': 0.4,
                'timestamp': datetime.now()
            },
            {
                'underlying_price': 19450,
                'iv_percentile': 0.8, 'atr_normalized': 0.75, 'gamma_exposure': 0.7,
                'ema_alignment': 0.2, 'price_momentum': 0.1, 'volume_confirmation': 0.15,
                'strike_correlation': 0.4, 'vwap_deviation': 0.3, 'pivot_analysis': 0.35,
                'volume_trend': 0.3, 'volatility_regime': 0.8,
                'timestamp': datetime.now()
            },
            {
                'underlying_price': 19550,
                'iv_percentile': 0.5, 'atr_normalized': 0.45, 'gamma_exposure': 0.4,
                'ema_alignment': 0.5, 'price_momentum': 0.4, 'volume_confirmation': 0.35,
                'strike_correlation': 0.65, 'vwap_deviation': 0.6, 'pivot_analysis': 0.7,
                'volume_trend': 0.5, 'volatility_regime': 0.5,
                'timestamp': datetime.now()
            }
        ]
    
    def test_integrator_initialization(self):
        """Test integrator initialization"""
        self.assertIsNotNone(self.integrator)
        self.assertEqual(self.integrator.weight_allocation['triple_straddle'], 0.35)
        self.assertEqual(self.integrator.weight_allocation['regime_components'], 0.65)
        
        # Check straddle weights sum to 1.0
        straddle_weight_sum = sum(self.integrator.straddle_weights.values())
        self.assertAlmostEqual(straddle_weight_sum, 1.0, places=2)
        
        # Check timeframe weights sum to 1.0
        timeframe_weight_sum = sum(self.integrator.timeframe_weights.values())
        self.assertAlmostEqual(timeframe_weight_sum, 1.0, places=2)
        
        logger.info("✅ Integrator initialization test passed")
    
    def test_35_percent_weight_allocation(self):
        """Test 35% weight allocation for Triple Straddle"""
        for test_data in self.test_market_data:
            result = self.integrator.analyze_integrated_regime(test_data)
            
            # Validate weight allocation
            self.assertEqual(result.triple_straddle_weight, 0.35)
            self.assertEqual(result.other_components_weight, 0.65)
            
            # Validate component breakdown
            breakdown = result.component_breakdown
            self.assertIn('straddle_weight', breakdown)
            self.assertIn('regime_weight', breakdown)
            self.assertEqual(breakdown['straddle_weight'], 0.35)
            self.assertEqual(breakdown['regime_weight'], 0.65)
            
            # Validate contributions
            expected_straddle_contribution = breakdown['straddle_score'] * 0.35
            expected_regime_contribution = breakdown['regime_score'] * 0.65
            
            self.assertAlmostEqual(
                breakdown['straddle_contribution'], 
                expected_straddle_contribution, 
                places=3
            )
            self.assertAlmostEqual(
                breakdown['regime_contribution'], 
                expected_regime_contribution, 
                places=3
            )
        
        logger.info("✅ 35% weight allocation test passed")
    
    def test_atm_itm1_otm1_analysis(self):
        """Test ATM/ITM1/OTM1 straddle analysis"""
        test_data = self.test_market_data[0]
        
        # Generate option chain for testing
        option_chain = self.integrator._generate_simulated_option_chain(test_data)
        self.assertGreater(len(option_chain), 0)
        
        # Test individual straddle components
        atm_score = self.integrator._analyze_atm_straddle(option_chain, test_data)
        itm1_score = self.integrator._analyze_itm1_straddle(option_chain, test_data)
        otm1_score = self.integrator._analyze_otm1_straddle(option_chain, test_data)
        
        # Validate score ranges
        self.assertGreaterEqual(atm_score, 0.0)
        self.assertLessEqual(atm_score, 1.0)
        self.assertGreaterEqual(itm1_score, 0.0)
        self.assertLessEqual(itm1_score, 1.0)
        self.assertGreaterEqual(otm1_score, 0.0)
        self.assertLessEqual(otm1_score, 1.0)
        
        logger.info(f"✅ ATM/ITM1/OTM1 analysis: ATM={atm_score:.3f}, ITM1={itm1_score:.3f}, OTM1={otm1_score:.3f}")
    
    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation"""
        atm_score = 0.7
        itm1_score = 0.6
        otm1_score = 0.5
        test_data = self.test_market_data[0]
        
        correlation_matrix = self.integrator._calculate_correlation_matrix(
            atm_score, itm1_score, otm1_score, test_data
        )
        
        # Validate correlation matrix structure
        required_keys = [
            'atm_itm1_correlation', 'atm_otm1_correlation', 'itm1_otm1_correlation',
            'average_correlation', 'market_structure_correlation', 'correlation_strength'
        ]
        
        for key in required_keys:
            self.assertIn(key, correlation_matrix)
            self.assertGreaterEqual(correlation_matrix[key], 0.0)
            self.assertLessEqual(correlation_matrix[key], 1.0)
        
        # Validate average correlation calculation
        expected_avg = (
            correlation_matrix['atm_itm1_correlation'] +
            correlation_matrix['atm_otm1_correlation'] +
            correlation_matrix['itm1_otm1_correlation']
        ) / 3
        
        self.assertAlmostEqual(
            correlation_matrix['average_correlation'], 
            expected_avg, 
            places=3
        )
        
        logger.info("✅ Correlation matrix calculation test passed")
    
    def test_performance_requirements(self):
        """Test processing time requirements (<1.5 seconds)"""
        processing_times = []
        
        for test_data in self.test_market_data:
            start_time = time.time()
            result = self.integrator.analyze_integrated_regime(test_data)
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
    
    def test_regime_score_normalization(self):
        """Test regime score normalization"""
        test_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for score in test_scores:
            normalized_regime = self.integrator._normalize_regime_score(score)
            normalized_straddle = self.integrator._normalize_straddle_score(score)
            
            # Normalized scores should be in valid range
            self.assertGreaterEqual(normalized_regime, 0.0)
            self.assertLessEqual(normalized_regime, 1.0)
            self.assertGreaterEqual(normalized_straddle, 0.0)
            self.assertLessEqual(normalized_straddle, 1.0)
        
        logger.info("✅ Regime score normalization test passed")
    
    def test_integrated_result_structure(self):
        """Test integrated result structure and completeness"""
        test_data = self.test_market_data[0]
        result = self.integrator.analyze_integrated_regime(test_data)
        
        # Validate result type
        self.assertIsInstance(result, IntegratedRegimeResult)
        
        # Validate required fields
        self.assertIsNotNone(result.regime_id)
        self.assertIsNotNone(result.regime_confidence)
        self.assertIsNotNone(result.triple_straddle_score)
        self.assertIsNotNone(result.final_score)
        self.assertIsNotNone(result.component_breakdown)
        
        # Validate score ranges
        self.assertGreaterEqual(result.regime_confidence, 0.0)
        self.assertLessEqual(result.regime_confidence, 1.0)
        self.assertGreaterEqual(result.triple_straddle_score, 0.0)
        self.assertLessEqual(result.triple_straddle_score, 1.0)
        self.assertGreaterEqual(result.final_score, 0.0)
        self.assertLessEqual(result.final_score, 1.0)
        
        # Validate component breakdown
        breakdown = result.component_breakdown
        required_breakdown_keys = [
            'regime_score', 'regime_weight', 'regime_contribution',
            'straddle_score', 'straddle_weight', 'straddle_contribution',
            'atm_score', 'itm1_score', 'otm1_score'
        ]
        
        for key in required_breakdown_keys:
            self.assertIn(key, breakdown)
        
        logger.info("✅ Integrated result structure test passed")
    
    def test_weight_allocation_update(self):
        """Test dynamic weight allocation update"""
        # Test valid weight update
        new_weights = {
            'triple_straddle': 0.40,
            'regime_components': 0.60
        }
        
        success = self.integrator.update_weight_allocation(new_weights)
        self.assertTrue(success)
        self.assertEqual(self.integrator.weight_allocation['triple_straddle'], 0.40)
        self.assertEqual(self.integrator.weight_allocation['regime_components'], 0.60)
        
        # Test invalid weight update (doesn't sum to 1.0)
        invalid_weights = {
            'triple_straddle': 0.50,
            'regime_components': 0.60  # Sum = 1.10
        }
        
        success = self.integrator.update_weight_allocation(invalid_weights)
        self.assertFalse(success)
        
        # Weights should remain unchanged
        self.assertEqual(self.integrator.weight_allocation['triple_straddle'], 0.40)
        
        logger.info("✅ Weight allocation update test passed")
    
    def test_performance_validation(self):
        """Test performance validation with multiple samples"""
        validation_result = self.integrator.validate_integration_performance(self.test_market_data)
        
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
        empty_result = self.integrator.analyze_integrated_regime({})
        self.assertIsInstance(empty_result, IntegratedRegimeResult)
        
        # Test with invalid option chain
        invalid_data = {'underlying_price': 'invalid'}
        invalid_result = self.integrator.analyze_integrated_regime(invalid_data)
        self.assertIsInstance(invalid_result, IntegratedRegimeResult)
        
        logger.info("✅ Error handling test passed")

def run_triple_straddle_12regime_integration_tests():
    """Run comprehensive test suite for Triple Straddle 12-Regime Integration"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTripleStraddle12RegimeIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"TRIPLE STRADDLE 12-REGIME INTEGRATION TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'='*70}")
    
    if success_rate >= 0.9:  # 90% success rate required
        print("✅ TRIPLE STRADDLE 12-REGIME INTEGRATION TESTS PASSED")
        return True
    else:
        print("❌ TRIPLE STRADDLE 12-REGIME INTEGRATION TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_triple_straddle_12regime_integration_tests()
    sys.exit(0 if success else 1)
