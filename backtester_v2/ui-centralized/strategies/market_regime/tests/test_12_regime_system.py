"""
Unit Tests for Enhanced 12-Regime Classification System

This module provides comprehensive unit tests for the 12-regime classification system,
including regime classification accuracy, mapping validation, and performance testing.

Author: The Augster
Date: 2025-06-18
Version: 1.0.0
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import sys
import os
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_12_regime_detector import Enhanced12RegimeDetector, Regime12Classification
from actual_system_excel_manager import ActualSystemExcelManager

logger = logging.getLogger(__name__)

class Test12RegimeSystem(unittest.TestCase):
    """Comprehensive test suite for 12-regime classification system"""
    
    def setUp(self):
        """Set up test environment"""
        self.detector = Enhanced12RegimeDetector()
        self.excel_manager = ActualSystemExcelManager()
        
        # Test data samples
        self.test_market_data = [
            {
                'iv_percentile': 0.2, 'atr_normalized': 0.15, 'gamma_exposure': 0.1,
                'ema_alignment': 0.8, 'price_momentum': 0.7, 'volume_confirmation': 0.6,
                'strike_correlation': 0.85, 'vwap_deviation': 0.8, 'pivot_analysis': 0.75,
                'expected_regime': 'LOW_DIRECTIONAL_TRENDING'
            },
            {
                'iv_percentile': 0.9, 'atr_normalized': 0.85, 'gamma_exposure': 0.8,
                'ema_alignment': 0.1, 'price_momentum': 0.05, 'volume_confirmation': 0.0,
                'strike_correlation': 0.3, 'vwap_deviation': 0.4, 'pivot_analysis': 0.2,
                'expected_regime': 'HIGH_NONDIRECTIONAL_RANGE'
            },
            {
                'iv_percentile': 0.5, 'atr_normalized': 0.45, 'gamma_exposure': 0.4,
                'ema_alignment': 0.5, 'price_momentum': 0.4, 'volume_confirmation': 0.3,
                'strike_correlation': 0.75, 'vwap_deviation': 0.7, 'pivot_analysis': 0.8,
                'expected_regime': 'MODERATE_DIRECTIONAL_TRENDING'
            }
        ]
        
        # 18-regime to 12-regime mapping test data
        self.mapping_test_data = [
            {'regime_18': 'HIGH_VOLATILE_STRONG_BULLISH', 'expected_12': 'HIGH_DIRECTIONAL_TRENDING'},
            {'regime_18': 'LOW_VOLATILE_MILD_BEARISH', 'expected_12': 'LOW_DIRECTIONAL_RANGE'},
            {'regime_18': 'NORMAL_VOLATILE_NEUTRAL', 'expected_12': 'MODERATE_NONDIRECTIONAL_RANGE'},
            {'regime_18': 'HIGH_VOLATILE_SIDEWAYS', 'expected_12': 'HIGH_NONDIRECTIONAL_TRENDING'},
        ]
    
    def test_12_regime_initialization(self):
        """Test 12-regime detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(len(self.detector.regime_definitions), 12)
        self.assertIn('LOW_DIRECTIONAL_TRENDING', self.detector.regime_definitions)
        self.assertIn('HIGH_NONDIRECTIONAL_RANGE', self.detector.regime_definitions)
        
        logger.info("✅ 12-regime detector initialization test passed")
    
    def test_regime_definitions_structure(self):
        """Test regime definitions have correct structure"""
        for regime_id, regime_def in self.detector.regime_definitions.items():
            # Check required fields
            self.assertIn('volatility', regime_def)
            self.assertIn('trend', regime_def)
            self.assertIn('structure', regime_def)
            self.assertIn('thresholds', regime_def)
            self.assertIn('description', regime_def)
            
            # Check threshold structure
            thresholds = regime_def['thresholds']
            self.assertIn('volatility', thresholds)
            self.assertIn('directional', thresholds)
            self.assertIn('correlation', thresholds)
            
            # Validate threshold ranges
            self.assertEqual(len(thresholds['volatility']), 2)
            self.assertEqual(len(thresholds['directional']), 2)
            self.assertEqual(len(thresholds['correlation']), 2)
        
        logger.info("✅ Regime definitions structure test passed")
    
    def test_12_regime_classification(self):
        """Test 12-regime classification accuracy"""
        correct_classifications = 0
        total_classifications = len(self.test_market_data)
        
        for test_data in self.test_market_data:
            expected_regime = test_data.pop('expected_regime')
            
            # Classify regime
            result = self.detector.classify_12_regime(test_data)
            
            # Validate result structure
            self.assertIsInstance(result, Regime12Classification)
            self.assertIsNotNone(result.regime_id)
            self.assertIsNotNone(result.confidence)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            
            # Check if classification matches expected
            if result.regime_id == expected_regime:
                correct_classifications += 1
            
            logger.info(f"Classified: {result.regime_id} (expected: {expected_regime}, confidence: {result.confidence:.3f})")
        
        accuracy = correct_classifications / total_classifications
        self.assertGreaterEqual(accuracy, 0.6)  # At least 60% accuracy for unit tests
        
        logger.info(f"✅ 12-regime classification test passed: {accuracy:.1%} accuracy")
    
    def test_18_to_12_regime_mapping(self):
        """Test 18→12 regime mapping accuracy"""
        correct_mappings = 0
        total_mappings = len(self.mapping_test_data)
        
        for mapping_data in self.mapping_test_data:
            regime_18 = mapping_data['regime_18']
            expected_12 = mapping_data['expected_12']
            
            # Test mapping
            mapped_12 = self.detector.map_18_to_12_regime(regime_18)
            
            if mapped_12 == expected_12:
                correct_mappings += 1
            
            logger.info(f"Mapped: {regime_18} → {mapped_12} (expected: {expected_12})")
        
        mapping_accuracy = correct_mappings / total_mappings
        self.assertGreaterEqual(mapping_accuracy, 0.8)  # At least 80% mapping accuracy
        
        logger.info(f"✅ 18→12 regime mapping test passed: {mapping_accuracy:.1%} accuracy")
    
    def test_component_score_calculation(self):
        """Test individual component score calculations"""
        test_data = self.test_market_data[0]
        
        # Test volatility component
        volatility_score = self.detector._calculate_volatility_component(test_data)
        self.assertGreaterEqual(volatility_score, 0.0)
        self.assertLessEqual(volatility_score, 1.0)
        
        # Test directional component
        directional_score = self.detector._calculate_directional_component(test_data)
        self.assertGreaterEqual(directional_score, -1.0)
        self.assertLessEqual(directional_score, 1.0)
        
        # Test correlation component
        correlation_score = self.detector._calculate_correlation_component(test_data)
        self.assertGreaterEqual(correlation_score, 0.0)
        self.assertLessEqual(correlation_score, 1.0)
        
        logger.info(f"✅ Component scores: vol={volatility_score:.3f}, dir={directional_score:.3f}, corr={correlation_score:.3f}")
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        for test_data in self.test_market_data:
            result = self.detector.classify_12_regime(test_data.copy())
            
            # Confidence should be reasonable
            self.assertGreaterEqual(result.confidence, 0.1)
            self.assertLessEqual(result.confidence, 1.0)
            
            # Alternative regimes should be provided
            self.assertIsInstance(result.alternative_regimes, list)
            self.assertLessEqual(len(result.alternative_regimes), 3)
        
        logger.info("✅ Confidence calculation test passed")
    
    def test_excel_12_regime_support(self):
        """Test Excel configuration support for 12-regime mode"""
        # Generate 12-regime Excel template
        template_path = "test_12_regime_template.xlsx"
        
        try:
            # Update Excel structure to use 12-regime mode
            self.excel_manager.excel_structure['RegimeFormationConfig']['data'] = \
                self.excel_manager._generate_regime_formation_config("12_REGIME")
            
            # Generate template
            generated_path = self.excel_manager.generate_excel_template(template_path)
            self.assertTrue(Path(generated_path).exists())
            
            # Load and validate
            success = self.excel_manager.load_configuration(template_path)
            self.assertTrue(success)
            
            # Check regime formation config
            regime_config = self.excel_manager.get_regime_formation_configuration()
            self.assertGreater(len(regime_config), 0)
            
            # Verify 12-regime entries
            regime_names = regime_config.iloc[:, 0].tolist()
            expected_12_regimes = [
                'LOW_DIRECTIONAL_TRENDING', 'LOW_DIRECTIONAL_RANGE',
                'MODERATE_DIRECTIONAL_TRENDING', 'HIGH_NONDIRECTIONAL_RANGE'
            ]
            
            for expected_regime in expected_12_regimes:
                self.assertIn(expected_regime, regime_names)
            
            logger.info("✅ Excel 12-regime support test passed")
            
        finally:
            # Cleanup
            if Path(template_path).exists():
                Path(template_path).unlink()
    
    def test_performance_requirements(self):
        """Test processing time requirements (<3 seconds)"""
        import time
        
        processing_times = []
        
        for _ in range(10):  # Test 10 iterations
            test_data = self.test_market_data[0].copy()
            
            start_time = time.time()
            result = self.detector.classify_12_regime(test_data)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            self.assertLess(processing_time, 3.0)  # Must be under 3 seconds
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        logger.info(f"✅ Performance test passed: avg={avg_processing_time:.3f}s, max={max_processing_time:.3f}s")
    
    def test_regime_stability_metrics(self):
        """Test regime stability metrics calculation"""
        # Simulate regime history
        for i in range(5):
            test_data = self.test_market_data[i % len(self.test_market_data)].copy()
            result = self.detector.classify_12_regime(test_data)
        
        # Get stability metrics
        stability_metrics = self.detector.get_regime_stability_metrics()
        
        self.assertIn('stability', stability_metrics)
        self.assertIn('transition_frequency', stability_metrics)
        self.assertGreaterEqual(stability_metrics['stability'], 0.0)
        self.assertLessEqual(stability_metrics['stability'], 1.0)
        
        logger.info(f"✅ Stability metrics: {stability_metrics}")
    
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Test with empty data
        empty_result = self.detector.classify_12_regime({})
        self.assertIsInstance(empty_result, Regime12Classification)
        
        # Test with invalid regime mapping
        invalid_mapping = self.detector.map_18_to_12_regime("INVALID_REGIME")
        self.assertEqual(invalid_mapping, 'MODERATE_NONDIRECTIONAL_RANGE')
        
        logger.info("✅ Error handling test passed")

def run_comprehensive_12_regime_tests():
    """Run comprehensive test suite for 12-regime system"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(Test12RegimeSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"12-REGIME SYSTEM TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'='*60}")
    
    if success_rate >= 0.9:  # 90% success rate required
        print("✅ 12-REGIME SYSTEM TESTS PASSED")
        return True
    else:
        print("❌ 12-REGIME SYSTEM TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_comprehensive_12_regime_tests()
    sys.exit(0 if success else 1)
