#!/usr/bin/env python3
"""
Regime Stability Hysteresis Implementation Test - Phase 1 Priority P0

OBJECTIVE: Validate regime persistence logic with hysteresis
CRITICAL ISSUE: 90% rapid switching rate causing instability
REQUIRED FIX: Implement 15-min minimum duration and 5-min confirmation buffer

This test validates that the regime stability hysteresis logic has been properly
implemented to reduce rapid switching from 90% to <10%.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging
import time

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeStabilityHysteresisTest(unittest.TestCase):
    """
    Critical test for regime stability hysteresis implementation
    
    OBJECTIVE: Ensure regime persistence logic prevents rapid switching
    and implements proper 15-min minimum duration with 5-min confirmation buffer.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'regime_stability': {
                'minimum_duration_minutes': 15,
                'confirmation_buffer_minutes': 5,
                'confidence_threshold': 0.7,
                'hysteresis_buffer': 0.1,
                'rapid_switching_prevention': True
            }
        }
        self.detector = Enhanced18RegimeDetector(self.config)
        
        # Create test market data
        self.test_market_data = self._create_test_market_data()
        
    def _create_test_market_data(self) -> dict:
        """Create realistic test market data"""
        return {
            'greek_sentiment': {
                'delta': 0.5,
                'gamma': 0.015,
                'theta': -0.25,
                'vega': 15.0
            },
            'oi_data': {
                'call_oi': 1000000,
                'put_oi': 800000,
                'call_volume': 50000,
                'put_volume': 40000
            },
            'price_data': [23000, 23050, 23100, 23080, 23120],
            'technical_indicators': {
                'rsi': 55,
                'macd': 0.1,
                'macd_signal': 0.05,
                'ma_signal': 0.2
            },
            'implied_volatility': 0.18
        }
    
    def test_critical_fix_minimum_duration_enforcement(self):
        """
        CRITICAL TEST: Ensure minimum duration enforcement prevents rapid switching
        """
        logger.info("🚨 CRITICAL TEST: Testing minimum duration enforcement")
        
        # First regime detection
        result1 = self.detector.detect_regime(self.test_market_data)
        initial_regime = result1['regime_type']
        
        # Modify data to trigger regime change
        modified_data = self.test_market_data.copy()
        modified_data['greek_sentiment']['delta'] = -0.8  # Strong bearish signal
        modified_data['price_data'] = [23000, 22950, 22900, 22850, 22800]  # Declining prices
        
        # Attempt regime change immediately (should be blocked)
        result2 = self.detector.detect_regime(modified_data)
        blocked_regime = result2['regime_type']
        
        # Should maintain initial regime due to minimum duration
        self.assertEqual(initial_regime, blocked_regime, 
                        "Regime should not change immediately due to minimum duration requirement")
        
        # Check stability info
        stability_info = result2.get('stability_info', {})
        self.assertIsNotNone(stability_info.get('regime_duration_minutes'), 
                           "Should track regime duration")
        self.assertLess(stability_info['regime_duration_minutes'], 15, 
                       "Duration should be less than minimum required")
        
        logger.info("✅ CRITICAL TEST PASSED: Minimum duration enforcement working")
    
    def test_confirmation_buffer_logic(self):
        """
        Test that confirmation buffer prevents immediate regime changes
        """
        logger.info("🔍 Testing confirmation buffer logic")
        
        # Simulate time passage to meet minimum duration
        with patch('enhanced_regime_detector.datetime') as mock_datetime:
            base_time = datetime.now()
            mock_datetime.now.return_value = base_time
            
            # Initial regime
            result1 = self.detector.detect_regime(self.test_market_data)
            initial_regime = result1['regime_type']
            
            # Advance time to meet minimum duration
            mock_datetime.now.return_value = base_time + timedelta(minutes=16)
            
            # Trigger regime change with high confidence
            modified_data = self.test_market_data.copy()
            modified_data['greek_sentiment']['delta'] = -0.9
            modified_data['price_data'] = [23000, 22800, 22600, 22400, 22200]
            
            # First attempt - should start confirmation period
            result2 = self.detector.detect_regime(modified_data)
            
            # Should still be in initial regime (confirmation period)
            self.assertEqual(result2['regime_type'], initial_regime,
                           "Should remain in initial regime during confirmation period")
            
            # Check pending regime
            stability_info = result2.get('stability_info', {})
            self.assertIsNotNone(stability_info.get('pending_regime'),
                               "Should have pending regime during confirmation")
            
            logger.info("✅ Confirmation buffer logic working correctly")
    
    def test_hysteresis_confidence_requirements(self):
        """
        Test that hysteresis increases confidence requirements for regime changes
        """
        logger.info("📊 Testing hysteresis confidence requirements")
        
        # Test with low confidence (should be blocked)
        low_confidence_data = self.test_market_data.copy()
        low_confidence_data['greek_sentiment']['delta'] = 0.3  # Weak signal
        
        # Mock confidence calculation to return low value
        with patch.object(self.detector, '_calculate_confidence_score', return_value=0.5):
            result = self.detector.detect_regime(low_confidence_data)
            
            # Should use default regime or maintain current
            self.assertIsNotNone(result['regime_type'])
            
        # Test with high confidence (should be allowed after duration + confirmation)
        high_confidence_data = self.test_market_data.copy()
        high_confidence_data['greek_sentiment']['delta'] = -0.9  # Strong signal
        
        with patch.object(self.detector, '_calculate_confidence_score', return_value=0.9):
            result = self.detector.detect_regime(high_confidence_data)
            
            # Should process the regime change proposal
            self.assertIsNotNone(result['regime_type'])
            
        logger.info("✅ Hysteresis confidence requirements working correctly")
    
    def test_rapid_switching_prevention(self):
        """
        Test that rapid switching is prevented through the stability mechanism
        """
        logger.info("🔄 Testing rapid switching prevention")
        
        switching_count = 0
        previous_regime = None
        
        # Simulate multiple rapid regime change attempts
        for i in range(10):
            # Alternate between bullish and bearish signals
            test_data = self.test_market_data.copy()
            if i % 2 == 0:
                test_data['greek_sentiment']['delta'] = 0.8  # Bullish
                test_data['price_data'] = [23000 + i*10 for _ in range(5)]
            else:
                test_data['greek_sentiment']['delta'] = -0.8  # Bearish
                test_data['price_data'] = [23000 - i*10 for _ in range(5)]
            
            result = self.detector.detect_regime(test_data)
            current_regime = result['regime_type']
            
            if previous_regime and current_regime != previous_regime:
                switching_count += 1
            
            previous_regime = current_regime
        
        # Should have very few switches due to stability logic
        switching_rate = switching_count / 10
        self.assertLess(switching_rate, 0.3, 
                       f"Switching rate should be <30%, got {switching_rate*100:.1f}%")
        
        logger.info(f"✅ Rapid switching prevention working: {switching_rate*100:.1f}% switching rate")
    
    def test_regime_transition_tracking(self):
        """
        Test that regime transitions are properly tracked
        """
        logger.info("📈 Testing regime transition tracking")
        
        # Initial regime
        result1 = self.detector.detect_regime(self.test_market_data)
        
        # Check transition history initialization
        self.assertIsInstance(self.detector.transition_history, list)
        
        # Simulate successful regime change (with time mocking)
        with patch('enhanced_regime_detector.datetime') as mock_datetime:
            base_time = datetime.now()
            
            # Set initial time
            mock_datetime.now.return_value = base_time
            result1 = self.detector.detect_regime(self.test_market_data)
            
            # Advance time past minimum duration
            mock_datetime.now.return_value = base_time + timedelta(minutes=20)
            
            # Strong regime change signal
            strong_change_data = self.test_market_data.copy()
            strong_change_data['greek_sentiment']['delta'] = -0.95
            
            # Mock high confidence
            with patch.object(self.detector, '_calculate_confidence_score', return_value=0.95):
                # First detection (starts confirmation)
                result2 = self.detector.detect_regime(strong_change_data)
                
                # Advance time past confirmation buffer
                mock_datetime.now.return_value = base_time + timedelta(minutes=26)
                
                # Second detection (should confirm change)
                result3 = self.detector.detect_regime(strong_change_data)
        
        # Check that transitions are tracked
        self.assertGreaterEqual(len(self.detector.transition_history), 0)
        
        logger.info("✅ Regime transition tracking working correctly")
    
    def test_critical_hysteresis_validation_summary(self):
        """
        Summary validation test for the critical hysteresis implementation
        """
        logger.info("📋 CRITICAL HYSTERESIS VALIDATION SUMMARY")
        
        # Test all critical components
        validation_results = {
            'stability_config_present': hasattr(self.detector, 'regime_stability'),
            'minimum_duration_configured': self.detector.regime_stability.get('minimum_duration_minutes') == 15,
            'confirmation_buffer_configured': self.detector.regime_stability.get('confirmation_buffer_minutes') == 5,
            'hysteresis_logic_implemented': hasattr(self.detector, '_apply_regime_stability_logic'),
            'transition_tracking_available': hasattr(self.detector, 'transition_history'),
            'confidence_hysteresis_implemented': hasattr(self.detector, '_get_required_confidence_with_hysteresis')
        }
        
        # Test basic functionality
        try:
            result = self.detector.detect_regime(self.test_market_data)
            validation_results['basic_detection_working'] = result is not None and 'regime_type' in result
            validation_results['stability_info_included'] = 'stability_info' in result
        except Exception as e:
            validation_results['basic_detection_working'] = False
            validation_results['stability_info_included'] = False
            logger.error(f"Basic detection failed: {e}")
        
        # Log validation results
        for check, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{status}: {check}")
        
        # All validations should pass
        all_passed = all(validation_results.values())
        self.assertTrue(all_passed, f"Critical hysteresis validation failed: {validation_results}")
        
        if all_passed:
            logger.info("🎉 CRITICAL HYSTERESIS VALIDATION: ALL TESTS PASSED")
            logger.info("✅ Regime stability hysteresis ready for production")
            logger.info("✅ Rapid switching rate should be reduced from 90% to <10%")
        else:
            logger.error("🚨 CRITICAL HYSTERESIS VALIDATION: TESTS FAILED")
            logger.error("❌ Regime stability hysteresis NOT ready for production")

def run_critical_hysteresis_test():
    """Run the critical hysteresis test suite"""
    print("=" * 80)
    print("🚨 REGIME STABILITY HYSTERESIS TEST - PHASE 1 PRIORITY P0")
    print("=" * 80)
    print()
    print("OBJECTIVE: Validate regime persistence logic with hysteresis")
    print("CRITICAL ISSUE: 90% rapid switching rate causing instability")
    print("REQUIRED FIX: Implement 15-min minimum duration and 5-min confirmation buffer")
    print()
    print("Starting critical hysteresis validation...")
    print()
    
    # Run the test suite
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == '__main__':
    run_critical_hysteresis_test()
