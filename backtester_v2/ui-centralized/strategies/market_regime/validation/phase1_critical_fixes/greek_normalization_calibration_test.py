#!/usr/bin/env python3
"""
Greek Sentiment Normalization Calibration Test - Phase 1 Priority P1

OBJECTIVE: Validate market-calibrated Greek normalization factors
CRITICAL ISSUE: Arbitrary normalization factors (gamma*100, theta*10, vega/10)
REQUIRED FIX: Replace with market-calibrated values for Indian options market

This test validates that the Greek normalization factors have been properly
calibrated based on market standards and enhanced-market-regime-optimizer analysis.
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

from enhanced_greek_sentiment_analysis import GreekSentimentAnalyzerAnalysis, GreekSentimentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GreekNormalizationCalibrationTest(unittest.TestCase):
    """
    Critical calibration test for Greek Sentiment Analyzer normalization factors
    
    OBJECTIVE: Ensure Greek normalization factors are market-calibrated
    instead of using arbitrary values.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'normalization_calibrated': True,
            'market_standard_factors': True,
            'lookback_period': 20,
            'gamma_threshold': 0.7,
            'delta_threshold': 0.6
        }
        self.analyzer = GreekSentimentAnalyzerAnalysis(self.config)
        
        # Create realistic Greek values for NIFTY options
        self.realistic_greeks = self._create_realistic_greek_values()
        
    def _create_realistic_greek_values(self) -> dict:
        """Create realistic Greek values based on NIFTY options market data"""
        return {
            'typical_atm_greeks': {
                'delta': 0.5,      # ATM delta around 0.5
                'gamma': 0.015,    # Typical NIFTY gamma 0.01-0.02
                'theta': -0.25,    # Daily theta decay -0.2 to -0.4
                'vega': 15.0       # NIFTY vega typically 10-25
            },
            'extreme_otm_put_greeks': {
                'delta': -0.15,    # OTM put delta
                'gamma': 0.008,    # Lower gamma for OTM
                'theta': -0.15,    # Lower theta for OTM
                'vega': 8.0        # Lower vega for OTM
            },
            'extreme_otm_call_greeks': {
                'delta': 0.15,     # OTM call delta
                'gamma': 0.008,    # Lower gamma for OTM
                'theta': -0.15,    # Lower theta for OTM
                'vega': 8.0        # Lower vega for OTM
            },
            'deep_itm_greeks': {
                'delta': 0.95,     # Deep ITM delta
                'gamma': 0.002,    # Very low gamma for deep ITM
                'theta': -0.05,    # Low theta for deep ITM
                'vega': 3.0        # Low vega for deep ITM
            }
        }
    
    def test_critical_calibration_no_arbitrary_factors(self):
        """
        CRITICAL TEST: Ensure no arbitrary normalization factors
        
        This test verifies that the problematic arbitrary factors
        (gamma*100, theta*10, vega/10) have been replaced with
        market-calibrated values.
        """
        logger.info("🚨 CRITICAL TEST: Checking for arbitrary normalization factors")
        
        # Test with typical ATM Greeks
        test_greeks = self.realistic_greeks['typical_atm_greeks']
        
        # Calculate contributions using the updated method
        contributions = self.analyzer._calculate_greek_contributions(test_greeks)
        
        # Check that contributions are reasonable (not extreme values)
        self.assertIsNotNone(contributions, "Greek contributions should not be None")
        self.assertIsInstance(contributions, dict, "Contributions should be a dictionary")
        
        # CRITICAL: Check that values are in reasonable range [-1, 1]
        for greek, contribution in contributions.items():
            self.assertGreaterEqual(contribution, -1.0, 
                                  f"{greek} contribution should be >= -1.0, got {contribution}")
            self.assertLessEqual(contribution, 1.0, 
                               f"{greek} contribution should be <= 1.0, got {contribution}")
        
        # CRITICAL: Check that gamma is not over-scaled (was gamma*100)
        # With realistic gamma of 0.015 and old factor 100, would give 1.5 (clipped to 1.0)
        # With calibrated factor 50, should give 0.75 (more reasonable)
        gamma_contribution = contributions.get('gamma', 0)
        self.assertLess(abs(gamma_contribution), 0.9, 
                       f"Gamma contribution should be more moderate, got {gamma_contribution}")
        
        logger.info("✅ CRITICAL TEST PASSED: No arbitrary normalization factors detected")
    
    def test_market_calibrated_factors_validation(self):
        """
        Test that market-calibrated factors produce reasonable results
        """
        logger.info("🔍 Testing market-calibrated factors validation")
        
        # Test all realistic Greek scenarios
        for scenario_name, greeks in self.realistic_greeks.items():
            with self.subTest(scenario=scenario_name):
                contributions = self.analyzer._calculate_greek_contributions(greeks)
                
                # Verify all contributions are in valid range
                for greek, contribution in contributions.items():
                    self.assertIsInstance(contribution, (int, float), 
                                        f"{greek} contribution should be numeric")
                    self.assertGreaterEqual(contribution, -1.0)
                    self.assertLessEqual(contribution, 1.0)
                
                logger.debug(f"Scenario {scenario_name}: {contributions}")
        
        logger.info("✅ Market-calibrated factors validation passed")
    
    def test_normalization_factor_ranges(self):
        """
        Test that normalization factors are within expected market ranges
        """
        logger.info("📊 Testing normalization factor ranges")
        
        # Test edge cases to ensure factors are reasonable
        edge_cases = {
            'high_gamma': {'delta': 0.5, 'gamma': 0.025, 'theta': -0.3, 'vega': 20.0},
            'low_gamma': {'delta': 0.5, 'gamma': 0.001, 'theta': -0.1, 'vega': 5.0},
            'high_vega': {'delta': 0.5, 'gamma': 0.015, 'theta': -0.25, 'vega': 35.0},
            'low_vega': {'delta': 0.5, 'gamma': 0.015, 'theta': -0.25, 'vega': 2.0},
            'high_theta': {'delta': 0.5, 'gamma': 0.015, 'theta': -0.5, 'vega': 15.0},
            'low_theta': {'delta': 0.5, 'gamma': 0.015, 'theta': -0.05, 'vega': 15.0}
        }
        
        for case_name, greeks in edge_cases.items():
            with self.subTest(case=case_name):
                contributions = self.analyzer._calculate_greek_contributions(greeks)
                
                # Check that extreme values don't produce unreasonable results
                for greek, contribution in contributions.items():
                    self.assertGreaterEqual(contribution, -1.0)
                    self.assertLessEqual(contribution, 1.0)
                    
                    # Check that contributions are not all maxed out (sign of poor calibration)
                    if greek != 'delta':  # Delta can legitimately be at extremes
                        self.assertLess(abs(contribution), 0.95, 
                                      f"{greek} contribution too extreme for {case_name}: {contribution}")
        
        logger.info("✅ Normalization factor ranges validation passed")
    
    def test_calibration_consistency(self):
        """
        Test that calibration produces consistent results across similar inputs
        """
        logger.info("🔄 Testing calibration consistency")
        
        # Create similar Greek values
        base_greeks = {'delta': 0.5, 'gamma': 0.015, 'theta': -0.25, 'vega': 15.0}
        
        # Test small variations
        variations = [
            {'delta': 0.51, 'gamma': 0.016, 'theta': -0.26, 'vega': 15.5},
            {'delta': 0.49, 'gamma': 0.014, 'theta': -0.24, 'vega': 14.5},
            {'delta': 0.52, 'gamma': 0.017, 'theta': -0.27, 'vega': 16.0}
        ]
        
        base_contributions = self.analyzer._calculate_greek_contributions(base_greeks)
        
        for i, variation in enumerate(variations):
            var_contributions = self.analyzer._calculate_greek_contributions(variation)
            
            # Check that small input changes produce small output changes
            for greek in base_contributions:
                if greek in var_contributions:
                    diff = abs(base_contributions[greek] - var_contributions[greek])
                    self.assertLess(diff, 0.1, 
                                  f"Variation {i}: {greek} contribution changed too much: {diff}")
        
        logger.info("✅ Calibration consistency validation passed")
    
    def test_critical_calibration_validation_summary(self):
        """
        Summary validation test for the critical calibration fix
        """
        logger.info("📋 CRITICAL CALIBRATION VALIDATION SUMMARY")
        
        # Run comprehensive calibration test
        test_greeks = self.realistic_greeks['typical_atm_greeks']
        contributions = self.analyzer._calculate_greek_contributions(test_greeks)
        
        # Validation checklist
        validation_results = {
            'contributions_calculated': contributions is not None,
            'all_greeks_present': all(greek in contributions for greek in ['delta', 'gamma', 'theta', 'vega']),
            'values_in_range': all(-1.0 <= v <= 1.0 for v in contributions.values()),
            'gamma_not_over_scaled': abs(contributions.get('gamma', 0)) < 0.9,
            'theta_reasonable': abs(contributions.get('theta', 0)) < 0.8,
            'vega_reasonable': abs(contributions.get('vega', 0)) < 0.8,
            'delta_preserved': abs(contributions.get('delta', 0) - test_greeks['delta']) < 0.1
        }
        
        # Log validation results
        for check, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{status}: {check}")
        
        # All validations should pass
        all_passed = all(validation_results.values())
        self.assertTrue(all_passed, f"Critical calibration validation failed: {validation_results}")
        
        if all_passed:
            logger.info("🎉 CRITICAL CALIBRATION VALIDATION: ALL TESTS PASSED")
            logger.info("✅ Greek Sentiment Normalization ready for production")
        else:
            logger.error("🚨 CRITICAL CALIBRATION VALIDATION: TESTS FAILED")
            logger.error("❌ Greek Sentiment Normalization NOT ready for production")

def run_critical_calibration_test():
    """Run the critical calibration test suite"""
    print("=" * 80)
    print("🚨 GREEK NORMALIZATION CALIBRATION TEST - PHASE 1 PRIORITY P1")
    print("=" * 80)
    print()
    print("OBJECTIVE: Validate market-calibrated Greek normalization factors")
    print("CRITICAL ISSUE: Arbitrary normalization factors (gamma*100, theta*10, vega/10)")
    print("REQUIRED FIX: Replace with market-calibrated values for Indian options market")
    print()
    print("Starting critical calibration validation...")
    print()
    
    # Run the test suite
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == '__main__':
    run_critical_calibration_test()
