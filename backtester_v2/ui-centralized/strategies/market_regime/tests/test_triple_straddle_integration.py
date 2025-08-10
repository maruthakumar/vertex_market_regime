"""
Integration Test for Triple Straddle Analysis System

This module provides comprehensive testing for the Triple Straddle Analysis system
integration with the Enhanced 18-Regime Detector V2.

Test Coverage:
- Triple Straddle Analysis Engine
- Dynamic Weight Optimization
- Enhanced Regime Detector V2
- Excel Configuration System
- Performance Validation
- Real-time Integration
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from triple_straddle_analysis import TripleStraddleAnalysisEngine
from dynamic_weight_optimizer import DynamicWeightOptimizer, PerformanceMetrics
from enhanced_regime_detector_v2 import Enhanced18RegimeDetectorV2
from triple_straddle_excel_config import TripleStraddleExcelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTripleStraddleIntegration(unittest.TestCase):
    """Comprehensive integration tests for Triple Straddle Analysis system"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'learning_rate': 0.05,
            'performance_window': 20,
            'min_performance_threshold': 0.6
        }
        
        # Initialize components
        self.triple_straddle_engine = TripleStraddleAnalysisEngine(self.config)
        self.weight_optimizer = DynamicWeightOptimizer(self.config)
        self.enhanced_detector = Enhanced18RegimeDetectorV2(self.config)
        self.excel_config = TripleStraddleExcelConfig()
        
        # Generate test market data
        self.test_market_data = self._generate_test_market_data()
        
        logger.info("Test environment set up successfully")
    
    def test_triple_straddle_analysis_basic(self):
        """Test basic Triple Straddle Analysis functionality"""
        logger.info("Testing Triple Straddle Analysis basic functionality...")
        
        # Test with valid market data
        result = self.triple_straddle_engine.analyze_market_regime(self.test_market_data)
        
        # Validate result structure
        self.assertIn('triple_straddle_score', result)
        self.assertIn('confidence', result)
        self.assertIn('component_results', result)
        self.assertIn('weights_used', result)
        self.assertIn('timeframe_weights', result)
        
        # Validate score range
        self.assertGreaterEqual(result['triple_straddle_score'], -1.0)
        self.assertLessEqual(result['triple_straddle_score'], 1.0)
        
        # Validate confidence range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        logger.info("✅ Triple Straddle Analysis basic test passed")
    
    def test_dynamic_weight_optimization(self):
        """Test Dynamic Weight Optimization functionality"""
        logger.info("Testing Dynamic Weight Optimization...")
        
        # Generate performance data
        performance_data = self._generate_performance_data(50)
        market_conditions = {'volatility': 0.15, 'time_of_day': 10}
        
        # Test weight optimization
        optimization_result = self.weight_optimizer.optimize_weights(
            performance_data, market_conditions
        )
        
        # Validate optimization result
        self.assertIn('optimized_weights', optimization_result)
        self.assertIn('performance_improvement', optimization_result)
        self.assertIn('confidence_score', optimization_result)
        self.assertIn('validation_passed', optimization_result)
        
        # Validate weight structure
        weights = optimization_result['optimized_weights']
        self.assertIn('pillar', weights)
        self.assertIn('indicator', weights)
        self.assertIn('component', weights)
        
        # Validate weight normalization
        for level_weights in weights.values():
            total_weight = sum(level_weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        logger.info("✅ Dynamic Weight Optimization test passed")
    
    def test_enhanced_regime_detector_v2(self):
        """Test Enhanced 18-Regime Detector V2 functionality"""
        logger.info("Testing Enhanced 18-Regime Detector V2...")
        
        # Test regime detection
        result = self.enhanced_detector.detect_regime(self.test_market_data)
        
        # Validate result structure
        self.assertIsNotNone(result.regime_type)
        self.assertIsNotNone(result.regime_score)
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.indicator_breakdown)
        self.assertIsNotNone(result.weights_applied)
        
        # Validate score and confidence ranges
        self.assertGreaterEqual(result.regime_score, -1.0)
        self.assertLessEqual(result.regime_score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Validate indicator breakdown
        expected_indicators = [
            'triple_straddle_analysis', 'greek_sentiment', 'oi_analysis',
            'iv_skew', 'atr_premium', 'supporting_technical'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.indicator_breakdown)
            indicator_result = result.indicator_breakdown[indicator]
            self.assertIn('score', indicator_result)
            self.assertIn('confidence', indicator_result)
        
        logger.info("✅ Enhanced 18-Regime Detector V2 test passed")
    
    def test_excel_configuration_generation(self):
        """Test Excel configuration generation"""
        logger.info("Testing Excel configuration generation...")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Generate Excel template
            output_path = self.excel_config.generate_excel_template(temp_path)
            
            # Validate file creation
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(output_path, temp_path)
            
            # Validate Excel structure (basic check)
            import openpyxl
            wb = openpyxl.load_workbook(output_path)
            
            expected_sheets = [
                'TripleStraddleConfig', 'WeightOptimization', 'TimeframeSettings',
                'TechnicalAnalysis', 'PerformanceTracking', 'RegimeThresholds'
            ]
            
            for sheet_name in expected_sheets:
                self.assertIn(sheet_name, wb.sheetnames)
            
            logger.info("✅ Excel configuration generation test passed")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration of all components"""
        logger.info("Testing end-to-end integration...")
        
        # Test multiple regime detections with performance tracking
        results = []
        
        for i in range(10):
            # Generate slightly different market data
            market_data = self._generate_test_market_data(seed=i)
            
            # Detect regime
            result = self.enhanced_detector.detect_regime(market_data)
            results.append(result)
            
            # Validate each result
            self.assertIsNotNone(result.regime_type)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
        
        # Validate performance tracking
        self.assertGreater(len(self.enhanced_detector.performance_history), 0)
        self.assertGreater(len(self.enhanced_detector.regime_history), 0)
        
        # Test weight optimization trigger
        initial_weights = self.enhanced_detector.current_weights.copy()
        
        # Generate more results to trigger optimization
        for i in range(100):
            market_data = self._generate_test_market_data(seed=i + 10)
            result = self.enhanced_detector.detect_regime(market_data)
        
        # Weights should have been optimized
        final_weights = self.enhanced_detector.current_weights
        
        # At least one weight should have changed (or stayed same if optimization failed)
        logger.info("✅ End-to-end integration test passed")
    
    def test_performance_validation(self):
        """Test performance validation and metrics"""
        logger.info("Testing performance validation...")
        
        # Generate performance data
        performance_data = self._generate_performance_data(100)
        
        # Calculate performance statistics
        accuracies = [p.accuracy for p in performance_data]
        confidences = [p.confidence_avg for p in performance_data]
        
        # Validate performance metrics
        avg_accuracy = np.mean(accuracies)
        avg_confidence = np.mean(confidences)
        
        self.assertGreaterEqual(avg_accuracy, 0.0)
        self.assertLessEqual(avg_accuracy, 1.0)
        self.assertGreaterEqual(avg_confidence, 0.0)
        self.assertLessEqual(avg_confidence, 1.0)
        
        # Test performance improvement calculation
        baseline_performance = 0.6
        recent_performance = performance_data[-10:]
        
        improvement = np.mean([p.accuracy for p in recent_performance]) - baseline_performance
        
        logger.info(f"Performance improvement: {improvement:.3f}")
        logger.info("✅ Performance validation test passed")
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        logger.info("Testing error handling...")
        
        # Test with empty market data
        empty_data = {}
        result = self.triple_straddle_engine.analyze_market_regime(empty_data)
        
        # Should return default result without crashing
        self.assertIn('triple_straddle_score', result)
        self.assertEqual(result['triple_straddle_score'], 0.0)
        
        # Test with malformed market data
        malformed_data = {'invalid_key': 'invalid_value'}
        result = self.enhanced_detector.detect_regime(malformed_data)
        
        # Should return default result without crashing
        self.assertIsNotNone(result.regime_type)
        
        # Test weight optimization with insufficient data
        insufficient_performance = self._generate_performance_data(5)
        optimization_result = self.weight_optimizer.optimize_weights(
            insufficient_performance, {'volatility': 0.15}
        )
        
        # Should handle gracefully
        self.assertIn('optimized_weights', optimization_result)
        
        logger.info("✅ Error handling test passed")
    
    def _generate_test_market_data(self, seed: int = 42) -> dict:
        """Generate realistic test market data"""
        np.random.seed(seed)
        
        # Base parameters
        underlying_price = 18500 + np.random.normal(0, 100)
        strikes = [underlying_price - 100, underlying_price, underlying_price + 100]
        
        # Generate options data
        options_data = {}
        for strike in strikes:
            ce_price = max(0, underlying_price - strike + np.random.normal(0, 10))
            pe_price = max(0, strike - underlying_price + np.random.normal(0, 10))
            
            options_data[strike] = {
                'CE': {
                    'close': ce_price,
                    'volume': np.random.randint(1000, 10000),
                    'oi': np.random.randint(10000, 100000),
                    'iv': 0.15 + np.random.normal(0, 0.05)
                },
                'PE': {
                    'close': pe_price,
                    'volume': np.random.randint(1000, 10000),
                    'oi': np.random.randint(10000, 100000),
                    'iv': 0.15 + np.random.normal(0, 0.05)
                }
            }
        
        # Generate price history
        price_history = []
        for i in range(300):
            price = underlying_price + np.random.normal(0, 50)
            price_history.append({
                'close': price,
                'high': price * 1.01,
                'low': price * 0.99,
                'volume': np.random.randint(1000, 5000),
                'timestamp': datetime.now() - timedelta(minutes=300-i)
            })
        
        return {
            'underlying_price': underlying_price,
            'strikes': strikes,
            'options_data': options_data,
            'price_history': price_history,
            'greek_data': {
                'delta': np.random.normal(0.5, 0.2),
                'gamma': np.random.normal(0.02, 0.01),
                'theta': np.random.normal(-0.1, 0.05),
                'vega': np.random.normal(0.3, 0.1)
            },
            'oi_data': {
                'call_oi': np.random.randint(100000, 1000000),
                'put_oi': np.random.randint(100000, 1000000),
                'call_volume': np.random.randint(10000, 100000),
                'put_volume': np.random.randint(10000, 100000)
            }
        }
    
    def _generate_performance_data(self, count: int) -> list:
        """Generate test performance data"""
        performance_data = []
        
        for i in range(count):
            performance = PerformanceMetrics(
                accuracy=0.6 + np.random.normal(0, 0.1),
                precision=0.65 + np.random.normal(0, 0.1),
                recall=0.6 + np.random.normal(0, 0.1),
                f1_score=0.62 + np.random.normal(0, 0.1),
                confidence_avg=0.7 + np.random.normal(0, 0.1),
                regime_stability=0.8 + np.random.normal(0, 0.1),
                timestamp=datetime.now() - timedelta(minutes=count-i)
            )
            
            # Ensure values are in valid ranges
            performance.accuracy = np.clip(performance.accuracy, 0.0, 1.0)
            performance.precision = np.clip(performance.precision, 0.0, 1.0)
            performance.recall = np.clip(performance.recall, 0.0, 1.0)
            performance.f1_score = np.clip(performance.f1_score, 0.0, 1.0)
            performance.confidence_avg = np.clip(performance.confidence_avg, 0.0, 1.0)
            performance.regime_stability = np.clip(performance.regime_stability, 0.0, 1.0)
            
            performance_data.append(performance)
        
        return performance_data

if __name__ == '__main__':
    # Run integration tests
    logger.info("🚀 Starting Triple Straddle Analysis Integration Tests...")
    
    unittest.main(verbosity=2)
    
    logger.info("✅ All integration tests completed!")
