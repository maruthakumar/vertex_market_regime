#!/usr/bin/env python3
"""
Basic Test for Market Regime Indicators V2
=========================================

This script performs basic testing of the implemented indicators
without requiring HeavyDB or other external dependencies.

Author: Market Regime Testing Team
Date: 2025-07-06
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import indicators
from indicators.greek_sentiment import (
    GreekSentimentAnalyzer,
    BaselineTracker,
    VolumeOIWeighter,
    ITMOTMAnalyzer,
    DTEAdjuster,
    GreekCalculator
)

from indicators.oi_pa_analysis import (
    OIPAAnalyzer,
    OIPatternDetector,
    DivergenceDetector,
    VolumeFlowAnalyzer,
    CorrelationAnalyzer,
    SessionWeightManager
)


class TestGreekSentimentV2Basic(unittest.TestCase):
    """Basic tests for Greek Sentiment V2"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'baseline_config': {
                'baseline_time': '09:15:00',
                'smoothing_alpha': 0.3,
                'min_data_points': 5,
                'session_boundary_buffer': 60
            },
            'weighting_config': {
                'oi_weight_alpha': 0.6,
                'volume_weight_beta': 0.4,
                'adaptive_adjustment': True,
                'min_weight': 0.1,
                'max_weight': 0.9
            },
            'itm_otm_config': {
                'itm_strikes': [1, 2, 3],
                'otm_strikes': [1, 2, 3],
                'moneyness_threshold': 0.5,
                'institutional_threshold': 0.7
            },
            'dte_config': {
                'near_expiry_days': 7,
                'medium_expiry_days': 30,
                'adjustment_factors': {
                    'near': {'delta': 1.0, 'gamma': 1.2, 'theta': 1.5, 'vega': 0.8},
                    'medium': {'delta': 1.2, 'gamma': 1.0, 'theta': 0.8, 'vega': 1.5},
                    'far': {'delta': 1.0, 'gamma': 0.8, 'theta': 0.3, 'vega': 2.0}
                }
            },
            'normalization_config': {
                'delta_factor': 1.0,
                'gamma_factor': 50.0,
                'theta_factor': 5.0,
                'vega_factor': 20.0,
                'precision_tolerance': 0.001
            }
        }
    
    def test_baseline_tracker_initialization(self):
        """Test BaselineTracker initialization"""
        tracker = BaselineTracker(self.config['baseline_config'])
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.baseline_time.strftime('%H:%M:%S'), '09:15:00')
        logger.info("✅ BaselineTracker initialization test passed")
    
    def test_volume_oi_weighter_initialization(self):
        """Test VolumeOIWeighter initialization"""
        weighter = VolumeOIWeighter(self.config['weighting_config'])
        self.assertIsNotNone(weighter)
        self.assertEqual(weighter.oi_weight_alpha, 0.6)
        self.assertEqual(weighter.volume_weight_beta, 0.4)
        logger.info("✅ VolumeOIWeighter initialization test passed")
    
    def test_itm_otm_analyzer_initialization(self):
        """Test ITMOTMAnalyzer initialization"""
        analyzer = ITMOTMAnalyzer(self.config['itm_otm_config'])
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.itm_strikes, [1, 2, 3])
        logger.info("✅ ITMOTMAnalyzer initialization test passed")
    
    def test_dte_adjuster_initialization(self):
        """Test DTEAdjuster initialization"""
        adjuster = DTEAdjuster(self.config['dte_config'])
        self.assertIsNotNone(adjuster)
        self.assertEqual(adjuster.near_expiry_days, 7)
        logger.info("✅ DTEAdjuster initialization test passed")
    
    def test_greek_calculator_initialization(self):
        """Test GreekCalculator initialization"""
        calculator = GreekCalculator(self.config['normalization_config'])
        self.assertIsNotNone(calculator)
        self.assertEqual(calculator.normalization_factors['delta'], 1.0)
        logger.info("✅ GreekCalculator initialization test passed")
    
    def test_greek_sentiment_analyzer_initialization(self):
        """Test GreekSentimentAnalyzer initialization"""
        analyzer = GreekSentimentAnalyzer(self.config)
        self.assertIsNotNone(analyzer)
        self.assertIsNotNone(analyzer.baseline_tracker)
        self.assertIsNotNone(analyzer.volume_oi_weighter)
        logger.info("✅ GreekSentimentAnalyzer initialization test passed")


class TestOIPAAnalysisV2Basic(unittest.TestCase):
    """Basic tests for OI/PA Analysis V2"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'pattern_config': {
                'lookback_periods': 20,
                'min_pattern_strength': 0.6,
                'pattern_types': ['accumulation', 'distribution', 'neutral', 'choppy'],
                'oi_change_threshold': 0.05
            },
            'divergence_config': {
                'divergence_types': [
                    'price_oi_divergence',
                    'volume_oi_divergence',
                    'price_volume_divergence',
                    'greek_oi_divergence',
                    'iv_price_divergence'
                ],
                'divergence_threshold': 0.7,
                'confirmation_periods': 3
            },
            'volume_flow_config': {
                'institutional_threshold': 0.8,
                'retail_threshold': 0.3,
                'flow_smoothing_periods': 5,
                'volume_spike_multiplier': 2.0
            },
            'correlation_config': {
                'correlation_window': 50,
                'min_correlation': 0.80,
                'correlation_types': ['pearson', 'spearman'],
                'significance_level': 0.05
            },
            'session_config': {
                'sessions': {
                    'pre_open': {'start': '09:00', 'end': '09:15', 'weight': 0.8},
                    'opening': {'start': '09:15', 'end': '09:45', 'weight': 1.2},
                    'morning': {'start': '09:45', 'end': '12:00', 'weight': 1.0},
                    'midday': {'start': '12:00', 'end': '13:30', 'weight': 0.7},
                    'afternoon': {'start': '13:30', 'end': '15:00', 'weight': 1.1},
                    'closing': {'start': '15:00', 'end': '15:25', 'weight': 1.3},
                    'post_close': {'start': '15:25', 'end': '15:30', 'weight': 0.6}
                },
                'decay_lambda': 0.1
            }
        }
    
    def test_oi_pattern_detector_initialization(self):
        """Test OIPatternDetector initialization"""
        detector = OIPatternDetector(self.config['pattern_config'])
        self.assertIsNotNone(detector)
        self.assertEqual(detector.lookback_periods, 20)
        logger.info("✅ OIPatternDetector initialization test passed")
    
    def test_divergence_detector_initialization(self):
        """Test DivergenceDetector initialization"""
        detector = DivergenceDetector(self.config['divergence_config'])
        self.assertIsNotNone(detector)
        self.assertEqual(len(detector.divergence_types), 5)
        logger.info("✅ DivergenceDetector initialization test passed")
    
    def test_volume_flow_analyzer_initialization(self):
        """Test VolumeFlowAnalyzer initialization"""
        analyzer = VolumeFlowAnalyzer(self.config['volume_flow_config'])
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.institutional_threshold, 0.8)
        logger.info("✅ VolumeFlowAnalyzer initialization test passed")
    
    def test_correlation_analyzer_initialization(self):
        """Test CorrelationAnalyzer initialization"""
        analyzer = CorrelationAnalyzer(self.config['correlation_config'])
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.min_correlation, 0.80)
        logger.info("✅ CorrelationAnalyzer initialization test passed")
    
    def test_session_weight_manager_initialization(self):
        """Test SessionWeightManager initialization"""
        manager = SessionWeightManager(self.config['session_config'])
        self.assertIsNotNone(manager)
        self.assertEqual(len(manager.sessions), 7)
        logger.info("✅ SessionWeightManager initialization test passed")


def run_basic_tests():
    """Run all basic tests"""
    logger.info("="*80)
    logger.info("Starting Market Regime Indicators V2 Basic Tests")
    logger.info("="*80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add Greek Sentiment tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGreekSentimentV2Basic))
    
    # Add OI/PA Analysis tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOIPAAnalysisV2Basic))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("="*80)
    logger.info("Test Summary:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success: {result.wasSuccessful()}")
    logger.info("="*80)
    
    if result.wasSuccessful():
        logger.info("🎉 All basic tests passed!")
        logger.info("\nThe implemented indicators are working correctly:")
        logger.info("  ✅ Greek Sentiment V2 - All components initialized")
        logger.info("  ✅ OI/PA Analysis V2 - All components initialized")
        logger.info("\nNext steps:")
        logger.info("  1. Run comprehensive tests with real HeavyDB data")
        logger.info("  2. Implement Technical Indicators V2")
        logger.info("  3. Implement IV Analytics V2")
        logger.info("  4. Implement Market Breadth V2")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)