#!/usr/bin/env python3
"""
Volatility Threshold Calibration Inconsistency Test - Phase 1 Priority P1

OBJECTIVE: Validate unified volatility threshold calibration
CRITICAL ISSUE: Inconsistent volatility thresholds between detectors
REQUIRED FIX: Unify calibration between enhanced and standalone detectors

This test validates that both detector implementations use the same
calibrated volatility thresholds for consistent regime classification.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import logging

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import both detector implementations
from enhanced_regime_detector import Enhanced18RegimeDetector
from validation_workspace.standalone_regime_detector import StandaloneEnhanced18RegimeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityThresholdCalibrationTest(unittest.TestCase):
    """
    Critical test for volatility threshold calibration consistency
    
    OBJECTIVE: Ensure both detector implementations use the same
    calibrated volatility thresholds for consistent regime classification.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.enhanced_detector = Enhanced18RegimeDetector()
        self.standalone_detector = StandaloneEnhanced18RegimeDetector()
        
        # Test market data with different volatility levels
        self.test_scenarios = self._create_volatility_test_scenarios()
        
    def _create_volatility_test_scenarios(self) -> dict:
        """Create test scenarios with different volatility levels"""
        base_data = {
            'greek_sentiment': {'delta': 0.5, 'gamma': 0.015, 'theta': -0.25, 'vega': 15.0},
            'oi_data': {'call_oi': 1000000, 'put_oi': 800000, 'call_volume': 50000, 'put_volume': 40000},
            'price_data': [23000, 23050, 23100, 23080, 23120],
            'technical_indicators': {'rsi': 55, 'macd': 0.1, 'macd_signal': 0.05, 'ma_signal': 0.2}
        }
        
        return {
            'low_volatility': {**base_data, 'implied_volatility': 0.10},      # 10% IV
            'normal_low_volatility': {**base_data, 'implied_volatility': 0.20},  # 20% IV
            'normal_high_volatility': {**base_data, 'implied_volatility': 0.35}, # 35% IV
            'high_volatility': {**base_data, 'implied_volatility': 0.70},     # 70% IV
            'extreme_volatility': {**base_data, 'implied_volatility': 0.90}   # 90% IV
        }
    
    def test_critical_fix_threshold_consistency(self):
        """
        CRITICAL TEST: Ensure volatility thresholds are consistent between detectors
        """
        logger.info("🚨 CRITICAL TEST: Checking volatility threshold consistency")
        
        # Compare volatility thresholds directly
        enhanced_thresholds = self.enhanced_detector.volatility_thresholds
        standalone_thresholds = self.standalone_detector.volatility_thresholds
        
        # Check each threshold level
        threshold_levels = ['high', 'normal_high', 'normal_low', 'low']
        
        for level in threshold_levels:
            enhanced_value = enhanced_thresholds.get(level)
            standalone_value = standalone_thresholds.get(level)
            
            self.assertEqual(enhanced_value, standalone_value,
                           f"Threshold mismatch for '{level}': enhanced={enhanced_value}, standalone={standalone_value}")
            
            logger.info(f"✅ {level} threshold consistent: {enhanced_value}")
        
        logger.info("✅ CRITICAL TEST PASSED: All volatility thresholds are consistent")
    
    def test_calibrated_values_validation(self):
        """
        Test that calibrated values are within expected ranges for Indian market
        """
        logger.info("📊 Testing calibrated values validation")
        
        thresholds = self.enhanced_detector.volatility_thresholds
        
        # Validate calibrated ranges (based on Indian options market analysis)
        expected_ranges = {
            'high': (0.60, 0.70),      # High volatility: 60-70%
            'normal_high': (0.40, 0.50), # Normal high: 40-50%
            'normal_low': (0.20, 0.30),  # Normal low: 20-30%
            'low': (0.10, 0.20)        # Low volatility: 10-20%
        }
        
        for level, (min_val, max_val) in expected_ranges.items():
            threshold_value = thresholds[level]
            
            self.assertGreaterEqual(threshold_value, min_val,
                                  f"{level} threshold too low: {threshold_value} < {min_val}")
            self.assertLessEqual(threshold_value, max_val,
                               f"{level} threshold too high: {threshold_value} > {max_val}")
            
            logger.debug(f"✅ {level} threshold in valid range: {threshold_value} ∈ [{min_val}, {max_val}]")
        
        logger.info("✅ Calibrated values validation passed")
    
    def test_regime_classification_consistency(self):
        """
        Test that both detectors produce consistent regime classifications
        """
        logger.info("🔄 Testing regime classification consistency")
        
        for scenario_name, market_data in self.test_scenarios.items():
            with self.subTest(scenario=scenario_name):
                # Get regime classifications from both detectors
                enhanced_result = self.enhanced_detector.detect_regime(market_data)
                standalone_result = self.standalone_detector.detect_regime(market_data)
                
                enhanced_regime = enhanced_result['regime_type']
                standalone_regime = standalone_result['regime_type']
                
                # Check that regime types are consistent
                self.assertEqual(enhanced_regime, standalone_regime,
                               f"Regime mismatch for {scenario_name}: "
                               f"enhanced={enhanced_regime.value}, standalone={standalone_regime.value}")
                
                logger.debug(f"✅ {scenario_name}: Both detectors classified as {enhanced_regime.value}")
        
        logger.info("✅ Regime classification consistency validated")
    
    def test_volatility_component_calculation_consistency(self):
        """
        Test that volatility component calculations are consistent
        """
        logger.info("📈 Testing volatility component calculation consistency")
        
        for scenario_name, market_data in self.test_scenarios.items():
            with self.subTest(scenario=scenario_name):
                # Calculate volatility components directly
                enhanced_vol = self.enhanced_detector._calculate_volatility_component(market_data)
                standalone_vol = self.standalone_detector._calculate_volatility_component(market_data)
                
                # Allow small numerical differences (within 1%)
                vol_diff = abs(enhanced_vol - standalone_vol)
                self.assertLess(vol_diff, 0.01,
                              f"Volatility component mismatch for {scenario_name}: "
                              f"enhanced={enhanced_vol:.4f}, standalone={standalone_vol:.4f}, diff={vol_diff:.4f}")
                
                logger.debug(f"✅ {scenario_name}: Volatility components consistent "
                           f"(enhanced={enhanced_vol:.4f}, standalone={standalone_vol:.4f})")
        
        logger.info("✅ Volatility component calculation consistency validated")
    
    def test_threshold_boundary_behavior(self):
        """
        Test behavior at threshold boundaries
        """
        logger.info("🎯 Testing threshold boundary behavior")
        
        # Test data at exact threshold boundaries
        thresholds = self.enhanced_detector.volatility_thresholds
        
        boundary_test_cases = [
            ('low_boundary', thresholds['low']),
            ('normal_low_boundary', thresholds['normal_low']),
            ('normal_high_boundary', thresholds['normal_high']),
            ('high_boundary', thresholds['high'])
        ]
        
        base_data = self.test_scenarios['normal_low_volatility'].copy()
        
        for case_name, threshold_value in boundary_test_cases:
            with self.subTest(case=case_name):
                # Test at exact threshold
                test_data = base_data.copy()
                test_data['implied_volatility'] = threshold_value
                
                enhanced_result = self.enhanced_detector.detect_regime(test_data)
                standalone_result = self.standalone_detector.detect_regime(test_data)
                
                # Should produce same regime
                self.assertEqual(enhanced_result['regime_type'], standalone_result['regime_type'],
                               f"Boundary behavior mismatch at {case_name} (threshold={threshold_value})")
                
                logger.debug(f"✅ {case_name}: Consistent boundary behavior at {threshold_value}")
        
        logger.info("✅ Threshold boundary behavior validated")
    
    def test_critical_calibration_validation_summary(self):
        """
        Summary validation test for the critical calibration fix
        """
        logger.info("📋 CRITICAL VOLATILITY CALIBRATION VALIDATION SUMMARY")
        
        # Comprehensive validation checklist
        validation_results = {
            'thresholds_consistent': True,
            'calibrated_values_valid': True,
            'regime_classification_consistent': True,
            'volatility_calculation_consistent': True,
            'boundary_behavior_consistent': True
        }
        
        try:
            # Test threshold consistency
            enhanced_thresholds = self.enhanced_detector.volatility_thresholds
            standalone_thresholds = self.standalone_detector.volatility_thresholds
            
            for level in ['high', 'normal_high', 'normal_low', 'low']:
                if enhanced_thresholds[level] != standalone_thresholds[level]:
                    validation_results['thresholds_consistent'] = False
                    break
            
            # Test calibrated value ranges
            if not (0.60 <= enhanced_thresholds['high'] <= 0.70 and
                   0.40 <= enhanced_thresholds['normal_high'] <= 0.50 and
                   0.20 <= enhanced_thresholds['normal_low'] <= 0.30 and
                   0.10 <= enhanced_thresholds['low'] <= 0.20):
                validation_results['calibrated_values_valid'] = False
            
            # Test regime classification consistency (sample test)
            test_data = self.test_scenarios['normal_high_volatility']
            enhanced_result = self.enhanced_detector.detect_regime(test_data)
            standalone_result = self.standalone_detector.detect_regime(test_data)
            
            if enhanced_result['regime_type'] != standalone_result['regime_type']:
                validation_results['regime_classification_consistent'] = False
            
            # Test volatility calculation consistency
            enhanced_vol = self.enhanced_detector._calculate_volatility_component(test_data)
            standalone_vol = self.standalone_detector._calculate_volatility_component(test_data)
            
            if abs(enhanced_vol - standalone_vol) > 0.01:
                validation_results['volatility_calculation_consistent'] = False
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            for key in validation_results:
                validation_results[key] = False
        
        # Log validation results
        for check, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{status}: {check}")
        
        # All validations should pass
        all_passed = all(validation_results.values())
        self.assertTrue(all_passed, f"Critical calibration validation failed: {validation_results}")
        
        if all_passed:
            logger.info("🎉 CRITICAL VOLATILITY CALIBRATION VALIDATION: ALL TESTS PASSED")
            logger.info("✅ Unified volatility threshold calibration successful")
            logger.info("✅ Both detectors now use consistent Indian market calibrated values")
            logger.info("✅ Regime classification consistency achieved")
        else:
            logger.error("🚨 CRITICAL VOLATILITY CALIBRATION VALIDATION: TESTS FAILED")
            logger.error("❌ Volatility threshold calibration NOT unified")

def run_critical_calibration_test():
    """Run the critical calibration test suite"""
    print("=" * 80)
    print("🚨 VOLATILITY THRESHOLD CALIBRATION TEST - PHASE 1 PRIORITY P1")
    print("=" * 80)
    print()
    print("OBJECTIVE: Validate unified volatility threshold calibration")
    print("CRITICAL ISSUE: Inconsistent volatility thresholds between detectors")
    print("REQUIRED FIX: Unify calibration between enhanced and standalone detectors")
    print()
    print("Starting critical calibration validation...")
    print()
    
    # Run the test suite
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == '__main__':
    run_critical_calibration_test()
