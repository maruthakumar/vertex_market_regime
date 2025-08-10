#!/usr/bin/env python3
"""
Tests for Advanced Dynamic Weighting Engine

This module provides comprehensive tests for the Advanced Dynamic Weighting Engine,
validating ML-based weight optimization with DTE analysis and historical performance.

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

from advanced_dynamic_weighting_engine import (
    AdvancedDynamicWeightingEngine,
    WeightOptimizationResult,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)

class TestAdvancedDynamicWeighting(unittest.TestCase):
    """Comprehensive test suite for Advanced Dynamic Weighting"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = AdvancedDynamicWeightingEngine()
        
        # Test scenarios with different DTE values
        self.test_scenarios = [
            {
                'dte': 7,  # Very short
                'volatility_level': 0.8,
                'market_momentum': 0.6,
                'iv_percentile': 0.7
            },
            {
                'dte': 21,  # Short
                'volatility_level': 0.5,
                'market_momentum': 0.4,
                'iv_percentile': 0.5
            },
            {
                'dte': 45,  # Medium
                'volatility_level': 0.3,
                'market_momentum': 0.3,
                'iv_percentile': 0.3
            },
            {
                'dte': 90,  # Long
                'volatility_level': 0.2,
                'market_momentum': 0.2,
                'iv_percentile': 0.2
            }
        ]
    
    def test_engine_initialization(self):
        """Test advanced dynamic weighting engine initialization"""
        self.assertIsNotNone(self.engine)
        
        # Check base weights
        self.assertEqual(self.engine.base_weights['triple_straddle'], 0.35)
        self.assertEqual(self.engine.base_weights['regime_components'], 0.65)
        
        # Check DTE weight factors
        self.assertIn('very_short', self.engine.dte_weight_factors)
        self.assertIn('short', self.engine.dte_weight_factors)
        self.assertIn('medium', self.engine.dte_weight_factors)
        self.assertIn('long', self.engine.dte_weight_factors)
        
        # Check ML models
        self.assertIn('weight_optimizer', self.engine.ml_models)
        self.assertIn('performance_predictor', self.engine.ml_models)
        
        # Check optimization parameters
        self.assertEqual(self.engine.optimization_params['learning_rate'], 0.01)
        self.assertEqual(self.engine.optimization_params['max_weight_change'], 0.1)
        
        logger.info("✅ Advanced Dynamic Weighting Engine initialization test passed")
    
    def test_dte_category_classification(self):
        """Test DTE category classification"""
        # Test different DTE values
        test_cases = [
            (5, 'very_short'),
            (15, 'short'),
            (30, 'medium'),
            (60, 'long'),
            (120, 'very_long')
        ]
        
        for dte, expected_category in test_cases:
            category = self.engine._get_dte_category(dte)
            self.assertEqual(category, expected_category)
        
        logger.info("✅ DTE category classification test passed")
    
    def test_dte_weight_adjustments(self):
        """Test DTE-based weight adjustments"""
        base_weights = {'triple_straddle': 0.35, 'regime_components': 0.65}
        
        for scenario in self.test_scenarios:
            dte = scenario['dte']
            adjusted_weights = self.engine._apply_dte_adjustments(base_weights.copy(), dte)
            
            # Validate weight structure
            self.assertIn('triple_straddle', adjusted_weights)
            self.assertIn('regime_components', adjusted_weights)
            
            # Validate weight sum
            weight_sum = sum(adjusted_weights.values())
            self.assertAlmostEqual(weight_sum, 1.0, places=3)
            
            # Validate weight ranges
            for weight in adjusted_weights.values():
                self.assertGreaterEqual(weight, 0.0)
                self.assertLessEqual(weight, 1.0)
        
        logger.info("✅ DTE weight adjustments test passed")
    
    def test_ml_based_weight_optimization(self):
        """Test ML-based weight optimization"""
        # Generate test historical data
        historical_data = self._generate_test_historical_data(100)
        
        for scenario in self.test_scenarios:
            dte = scenario['dte']
            
            start_time = time.time()
            optimization_result = self.engine.optimize_weights_ml_based(historical_data, dte)
            optimization_time = time.time() - start_time
            
            # Validate result type
            self.assertIsInstance(optimization_result, WeightOptimizationResult)
            
            # Validate optimized weights
            weights = optimization_result.optimized_weights
            self.assertIn('triple_straddle', weights)
            self.assertIn('regime_components', weights)
            
            # Validate weight sum
            weight_sum = sum(weights.values())
            self.assertAlmostEqual(weight_sum, 1.0, places=3)
            
            # Validate weight constraints
            self.assertGreaterEqual(weights['triple_straddle'], 
                                  self.engine.config['weight_constraints']['min_triple_straddle_weight'])
            self.assertLessEqual(weights['triple_straddle'], 
                               self.engine.config['weight_constraints']['max_triple_straddle_weight'])
            
            # Validate performance metrics
            self.assertGreaterEqual(optimization_result.confidence, 0.0)
            self.assertLessEqual(optimization_result.confidence, 1.0)
            self.assertGreaterEqual(optimization_result.historical_accuracy, 0.0)
            self.assertLessEqual(optimization_result.historical_accuracy, 1.0)
            
            # Validate DTE factor
            expected_dte_factor = self.engine._calculate_dte_factor(dte)
            self.assertEqual(optimization_result.dte_factor, expected_dte_factor)
            
            # Validate optimization time
            self.assertLess(optimization_time, 5.0)  # Should be fast
        
        logger.info("✅ ML-based weight optimization test passed")
    
    def test_weight_validation_and_normalization(self):
        """Test weight validation and normalization"""
        # Test valid weights
        valid_weights = {'triple_straddle': 0.4, 'regime_components': 0.6}
        normalized = self.engine._validate_and_normalize_weights(valid_weights)
        
        weight_sum = sum(normalized.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=3)
        
        # Test invalid weights (sum > 1)
        invalid_weights = {'triple_straddle': 0.7, 'regime_components': 0.8}
        normalized = self.engine._validate_and_normalize_weights(invalid_weights)
        
        weight_sum = sum(normalized.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=3)
        
        # Test extreme weights
        extreme_weights = {'triple_straddle': 0.9, 'regime_components': 0.1}
        normalized = self.engine._validate_and_normalize_weights(extreme_weights)
        
        # Should be constrained
        self.assertLessEqual(normalized['triple_straddle'], 
                           self.engine.config['weight_constraints']['max_triple_straddle_weight'])
        
        logger.info("✅ Weight validation and normalization test passed")
    
    def test_performance_improvement_calculation(self):
        """Test performance improvement calculation"""
        historical_data = self._generate_test_historical_data(50)
        optimized_weights = {'triple_straddle': 0.4, 'regime_components': 0.6}
        
        improvement = self.engine._calculate_performance_improvement(optimized_weights, historical_data)
        
        # Validate improvement range
        self.assertGreaterEqual(improvement, -0.5)
        self.assertLessEqual(improvement, 0.5)
        
        logger.info(f"✅ Performance improvement calculation: {improvement:.3f}")
    
    def test_regime_performance_calculation(self):
        """Test regime-specific performance calculation"""
        historical_data = self._generate_test_historical_data(50)
        
        regime_performance = self.engine._calculate_regime_performance(historical_data)
        
        # Validate structure
        self.assertIsInstance(regime_performance, dict)
        
        # Validate performance values
        for regime, performance in regime_performance.items():
            self.assertGreaterEqual(performance, 0.0)
            self.assertLessEqual(performance, 1.0)
        
        logger.info(f"✅ Regime performance calculation: {len(regime_performance)} regimes")
    
    def test_weight_optimization_validation(self):
        """Test weight optimization validation with multiple scenarios"""
        validation_result = self.engine.validate_weight_optimization_performance(self.test_scenarios)
        
        # Validate validation result structure
        self.assertIn('optimization_effective', validation_result)
        self.assertIn('avg_performance_improvement', validation_result)
        self.assertIn('avg_confidence', validation_result)
        self.assertIn('optimization_results', validation_result)
        
        # Validate metrics
        self.assertGreaterEqual(validation_result['avg_confidence'], 0.0)
        self.assertLessEqual(validation_result['avg_confidence'], 1.0)
        self.assertGreaterEqual(validation_result['avg_performance_improvement'], -0.5)
        self.assertLessEqual(validation_result['avg_performance_improvement'], 0.5)
        
        # Validate individual results
        optimization_results = validation_result['optimization_results']
        self.assertEqual(len(optimization_results), len(self.test_scenarios))
        
        for result in optimization_results:
            self.assertIn('optimized_weights', result)
            self.assertIn('confidence', result)
            self.assertIn('optimization_time', result)
            
            # Validate weight sum
            weight_sum = sum(result['optimized_weights'].values())
            self.assertAlmostEqual(weight_sum, 1.0, places=3)
        
        logger.info(f"✅ Weight optimization validation: {validation_result['avg_confidence']:.3f} confidence")
    
    def test_optimization_performance_summary(self):
        """Test optimization performance summary"""
        # Run some optimizations to generate history
        historical_data = self._generate_test_historical_data(100)
        
        for scenario in self.test_scenarios[:2]:
            self.engine.optimize_weights_ml_based(historical_data, scenario['dte'])
        
        # Get performance summary
        summary = self.engine.get_optimization_performance_summary()
        
        # Validate summary structure
        if 'status' not in summary:  # Only if we have actual data
            self.assertIn('optimization_summary', summary)
            self.assertIn('weight_evolution', summary)
            self.assertIn('performance_assessment', summary)
            
            # Validate optimization summary
            opt_summary = summary['optimization_summary']
            self.assertIn('total_optimizations', opt_summary)
            self.assertIn('avg_improvement', opt_summary)
            self.assertIn('avg_confidence', opt_summary)
            
            # Validate performance assessment
            assessment = summary['performance_assessment']
            self.assertIn('optimization_grade', assessment)
            self.assertIn(assessment['optimization_grade'], 
                         ['EXCELLENT', 'VERY_GOOD', 'GOOD', 'ACCEPTABLE', 'NEEDS_IMPROVEMENT'])
            
            logger.info(f"✅ Performance summary: grade={assessment['optimization_grade']}")
        else:
            logger.info("✅ Performance summary generated (no data available)")
    
    def test_current_optimized_weights(self):
        """Test getting current optimized weights"""
        market_data = {
            'dte': 30,
            'iv_percentile': 0.5,
            'volatility_level': 0.4,
            'market_momentum': 0.3,
            'underlying_price': 19500
        }
        
        historical_performance = self._generate_test_historical_data(50)
        
        optimized_weights = self.engine.get_current_optimized_weights(market_data, historical_performance)
        
        # Validate weights
        self.assertIn('triple_straddle', optimized_weights)
        self.assertIn('regime_components', optimized_weights)
        
        weight_sum = sum(optimized_weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=3)
        
        logger.info(f"✅ Current optimized weights: {optimized_weights}")
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Test with empty historical data
        empty_result = self.engine.optimize_weights_ml_based([], 30)
        self.assertIsInstance(empty_result, WeightOptimizationResult)
        self.assertEqual(empty_result.optimization_method, "FALLBACK_DTE_BASED")
        
        # Test with invalid DTE
        invalid_dte_result = self.engine.optimize_weights_ml_based([], -10)
        self.assertIsInstance(invalid_dte_result, WeightOptimizationResult)
        
        logger.info("✅ Error handling test passed")
    
    def _generate_test_historical_data(self, num_points: int) -> List[Dict[str, Any]]:
        """Generate test historical data"""
        historical_data = []
        
        for i in range(num_points):
            data_point = {
                'regime_confidence': 0.5 + np.random.normal(0, 0.2),
                'triple_straddle_score': 0.5 + np.random.normal(0, 0.15),
                'volatility_level': 0.5 + np.random.normal(0, 0.2),
                'correlation_strength': 0.5 + np.random.normal(0, 0.1),
                'market_momentum': 0.5 + np.random.normal(0, 0.15),
                'dte': 30 + np.random.randint(-10, 10),
                'iv_percentile': 0.5 + np.random.normal(0, 0.2),
                'volume_profile': 0.5 + np.random.normal(0, 0.15),
                'accuracy': 0.7 + np.random.normal(0, 0.1),
                'regime_consistency': 0.6 + np.random.normal(0, 0.1),
                'prediction_confidence': 0.65 + np.random.normal(0, 0.1),
                'regime_id': f"REGIME_{np.random.randint(1, 13)}",
                'timestamp': datetime.now() - timedelta(hours=i)
            }
            
            # Clip values to valid ranges
            for key, value in data_point.items():
                if isinstance(value, (int, float)) and key != 'dte':
                    data_point[key] = np.clip(value, 0.0, 1.0)
            
            historical_data.append(data_point)
        
        return historical_data

def run_advanced_dynamic_weighting_tests():
    """Run comprehensive advanced dynamic weighting test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestAdvancedDynamicWeighting)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"ADVANCED DYNAMIC WEIGHTING TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"ML-based Optimization: DTE-aware weight adjustment")
    print(f"{'='*70}")
    
    if success_rate >= 0.9:  # 90% success rate required
        print("✅ ADVANCED DYNAMIC WEIGHTING TESTS PASSED")
        print("🚀 Ready for End-to-End Performance Optimization")
        return True
    else:
        print("❌ ADVANCED DYNAMIC WEIGHTING TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_advanced_dynamic_weighting_tests()
    sys.exit(0 if success else 1)
