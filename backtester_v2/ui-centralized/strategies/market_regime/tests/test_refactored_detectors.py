"""
Test suite for refactored regime detectors

Verifies that the refactored 12-regime and 18-regime detectors
work correctly with the new base class architecture.

Author: Market Regime System Optimizer
Date: 2025-07-07
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import refactored detectors
from enhanced_modules.refactored_12_regime_detector import Refactored12RegimeDetector
from enhanced_modules.refactored_18_regime_classifier import Refactored18RegimeClassifier


class TestRefactoredDetectors(unittest.TestCase):
    """Test cases for refactored regime detectors"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample market data
        self.sample_data = {
            'timestamp': datetime.now(),
            'underlying_price': 50000.0,
            'option_chain': pd.DataFrame({
                'strike_price': [49500, 49750, 50000, 50250, 50500],
                'option_type': ['CE', 'CE', 'CE', 'CE', 'CE'],
                'last_price': [600, 400, 250, 150, 80],
                'volume': [1000, 1200, 1500, 1100, 900],
                'implied_volatility': [18, 17, 16, 17, 19],
                'delta': [0.7, 0.6, 0.5, 0.4, 0.3]
            }),
            'indicators': {
                'rsi': 65,
                'macd_signal': 50,
                'adx': 30,
                'atr': 250,
                'bollinger_width': 500
            }
        }
        
    def test_12_regime_detector_initialization(self):
        """Test 12-regime detector initialization"""
        detector = Refactored12RegimeDetector()
        
        # Check regime count
        self.assertEqual(detector.get_regime_count(), 12)
        
        # Check regime mapping
        mapping = detector.get_regime_mapping()
        self.assertEqual(len(mapping), 12)
        self.assertIn('R1', mapping)
        self.assertIn('R12', mapping)
        
    def test_18_regime_classifier_initialization(self):
        """Test 18-regime classifier initialization"""
        classifier = Refactored18RegimeClassifier()
        
        # Check regime count
        self.assertEqual(classifier.get_regime_count(), 18)
        
        # Check regime mapping
        mapping = classifier.get_regime_mapping()
        self.assertGreaterEqual(len(mapping), 18)
        
    def test_12_regime_classification(self):
        """Test 12-regime classification"""
        detector = Refactored12RegimeDetector()
        
        # Calculate regime
        result = detector.calculate_regime(self.sample_data)
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIn(result.regime_id, [f'R{i}' for i in range(1, 13)])
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)
        
        # Check metadata
        self.assertIn('components', result.metadata)
        self.assertIn('volatility', result.metadata['components'])
        self.assertIn('trend', result.metadata['components'])
        self.assertIn('structure', result.metadata['components'])
        
    def test_18_regime_classification(self):
        """Test 18-regime classification"""
        classifier = Refactored18RegimeClassifier()
        
        # Calculate regime
        result = classifier.calculate_regime(self.sample_data)
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result.regime_id, str)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)
        
        # Check scores
        self.assertGreaterEqual(result.directional_score, -1)
        self.assertLessEqual(result.directional_score, 1)
        self.assertGreaterEqual(result.volatility_score, 0)
        self.assertLessEqual(result.volatility_score, 1)
        
    def test_caching_functionality(self):
        """Test caching in base class"""
        detector = Refactored12RegimeDetector()
        
        # First call - should miss cache
        result1 = detector.calculate_regime(self.sample_data)
        
        # Second call with same data - should hit cache
        result2 = detector.calculate_regime(self.sample_data)
        
        # Results should be the same
        self.assertEqual(result1.regime_id, result2.regime_id)
        self.assertEqual(result1.confidence, result2.confidence)
        
        # Check cache metrics
        metrics = detector.get_performance_metrics()
        self.assertGreater(metrics['cache']['hit_rate'], 0)
        
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        detector = Refactored12RegimeDetector()
        
        # Calculate regime multiple times
        for _ in range(5):
            detector.calculate_regime(self.sample_data)
            
        # Get performance metrics
        metrics = detector.get_performance_metrics()
        
        # Check metrics structure
        self.assertIn('performance', metrics)
        self.assertIn('total_calculation', metrics['performance'])
        self.assertEqual(metrics['performance']['total_calculation']['count'], 5)
        
    def test_data_validation(self):
        """Test data validation in base class"""
        detector = Refactored12RegimeDetector()
        
        # Test missing required field
        invalid_data = {'timestamp': datetime.now()}
        with self.assertRaises(ValueError):
            detector.calculate_regime(invalid_data)
            
        # Test invalid price
        invalid_data = {
            'timestamp': datetime.now(),
            'underlying_price': -100
        }
        with self.assertRaises(ValueError):
            detector.calculate_regime(invalid_data)
            
    def test_regime_smoothing(self):
        """Test regime smoothing functionality"""
        detector = Refactored12RegimeDetector(config={'regime_smoothing': True})
        
        # Create varying data to trigger different regimes
        data_variations = []
        for i in range(5):
            data = self.sample_data.copy()
            data['indicators'] = {
                'rsi': 30 + i * 10,  # Varying RSI
                'adx': 20 + i * 5,   # Varying ADX
                'atr': 200 + i * 50  # Varying ATR
            }
            data_variations.append(data)
            
        # Calculate regimes
        results = []
        for data in data_variations:
            result = detector.calculate_regime(data)
            results.append(result)
            
        # Check that smoothing is applied (metadata should indicate)
        smoothed_count = sum(1 for r in results if r.metadata.get('smoothed', False))
        self.assertGreater(smoothed_count, 0)
        
    def test_alternative_regimes(self):
        """Test alternative regime suggestions"""
        detector = Refactored12RegimeDetector()
        classifier = Refactored18RegimeClassifier()
        
        # Test 12-regime alternatives
        result_12 = detector.calculate_regime(self.sample_data)
        self.assertIsInstance(result_12.alternative_regimes, list)
        self.assertLessEqual(len(result_12.alternative_regimes), 3)
        
        # Test 18-regime alternatives
        result_18 = classifier.calculate_regime(self.sample_data)
        self.assertIsInstance(result_18.alternative_regimes, list)
        self.assertLessEqual(len(result_18.alternative_regimes), 3)
        
        # Check alternative format
        if result_12.alternative_regimes:
            alt_regime, alt_confidence = result_12.alternative_regimes[0]
            self.assertIsInstance(alt_regime, str)
            self.assertIsInstance(alt_confidence, float)
            
    def test_cache_reset(self):
        """Test cache reset functionality"""
        detector = Refactored12RegimeDetector()
        
        # Calculate regime to populate cache
        detector.calculate_regime(self.sample_data)
        
        # Reset cache
        detector.reset_cache()
        
        # Check cache is empty
        metrics = detector.get_performance_metrics()
        self.assertEqual(metrics['cache']['size'], 0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    run_tests()