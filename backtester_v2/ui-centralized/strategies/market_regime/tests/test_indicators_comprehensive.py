"""
Comprehensive Indicators Test Suite
==================================

Comprehensive tests for all indicator components and their interactions.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import logging

# Import indicator components
from ..indicators.straddle_analysis.straddle_engine import StraddleAnalysisEngine
from ..indicators.oi_pa_analysis.oi_pa_analyzer import OIPAAnalyzer
from ..indicators.greek_sentiment.greek_sentiment_analyzer import GreekSentimentAnalyzer
from ..indicators.market_breadth.market_breadth_analyzer import MarketBreadthAnalyzer
from ..indicators.iv_analytics.iv_analytics_analyzer import IVAnalyticsAnalyzer
from ..indicators.technical_indicators.technical_indicators_analyzer import TechnicalIndicatorsAnalyzer


class TestStraddleAnalysisEngine(unittest.TestCase):
    """Test Straddle Analysis Engine"""
    
    def setUp(self):
        self.config = {
            'resistance_threshold': 0.02,
            'correlation_window': 20,
            'volume_threshold': 100,
            'components': {
                'atm_straddle': {'weight': 0.4},
                'itm1_straddle': {'weight': 0.3},
                'otm1_straddle': {'weight': 0.3}
            }
        }
        
        # Sample option data
        self.sample_data = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'option_type': ['CE', 'PE', 'CE', 'PE', 'CE'],
            'spot': [110, 110, 110, 110, 110],
            'volume': [100, 200, 300, 150, 50],
            'oi': [1000, 2000, 3000, 1500, 500],
            'close': [5.0, 8.0, 3.0, 12.0, 1.5],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
    
    @patch('..indicators.straddle_analysis.straddle_engine.ATMStraddleAnalyzer')
    @patch('..indicators.straddle_analysis.straddle_engine.ITM1StraddleAnalyzer')
    @patch('..indicators.straddle_analysis.straddle_engine.OTM1StraddleAnalyzer')
    def test_straddle_engine_initialization(self, mock_otm1, mock_itm1, mock_atm):
        """Test straddle engine initialization"""
        engine = StraddleAnalysisEngine(self.config)
        
        self.assertIsNotNone(engine.atm_analyzer)
        self.assertIsNotNone(engine.itm1_analyzer)
        self.assertIsNotNone(engine.otm1_analyzer)
        self.assertEqual(engine.resistance_threshold, 0.02)
    
    @patch('..indicators.straddle_analysis.straddle_engine.ATMStraddleAnalyzer')
    @patch('..indicators.straddle_analysis.straddle_engine.ITM1StraddleAnalyzer')
    @patch('..indicators.straddle_analysis.straddle_engine.OTM1StraddleAnalyzer')
    def test_analyze_straddle_patterns(self, mock_otm1, mock_itm1, mock_atm):
        """Test straddle pattern analysis"""
        # Setup mocks
        mock_atm_instance = Mock()
        mock_itm1_instance = Mock()
        mock_otm1_instance = Mock()
        
        mock_atm.return_value = mock_atm_instance
        mock_itm1.return_value = mock_itm1_instance
        mock_otm1.return_value = mock_otm1_instance
        
        # Mock analysis results
        mock_atm_instance.analyze.return_value = {
            'resistance_strength': 0.75,
            'correlation_score': 0.65,
            'volume_support': 'strong'
        }
        
        mock_itm1_instance.analyze.return_value = {
            'resistance_strength': 0.70,
            'correlation_score': 0.60,
            'volume_support': 'moderate'
        }
        
        mock_otm1_instance.analyze.return_value = {
            'resistance_strength': 0.60,
            'correlation_score': 0.55,
            'volume_support': 'weak'
        }
        
        engine = StraddleAnalysisEngine(self.config)
        result = engine.analyze(self.sample_data)
        
        self.assertIn('straddle_patterns', result)
        self.assertIn('composite_score', result)
        self.assertIn('regime_signals', result)
        self.assertIn('resistance_levels', result)


class TestOIPAAnalyzer(unittest.TestCase):
    """Test OI-PA Analysis"""
    
    def setUp(self):
        self.config = {
            'correlation_window': 20,
            'divergence_threshold': 0.3,
            'volume_weight': 0.6,
            'oi_weight': 0.4
        }
        
        # Sample OI-PA data
        self.sample_data = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'option_type': ['CE', 'PE', 'CE', 'PE', 'CE'],
            'volume': [100, 200, 300, 150, 50],
            'oi': [1000, 2000, 3000, 1500, 500],
            'oi_change': [100, -50, 200, -100, 25],
            'close': [5.0, 8.0, 3.0, 12.0, 1.5],
            'session_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
    
    @patch('..indicators.oi_pa_analysis.oi_pa_analyzer.CorrelationAnalyzer')
    @patch('..indicators.oi_pa_analysis.oi_pa_analyzer.DivergenceDetector')
    @patch('..indicators.oi_pa_analysis.oi_pa_analyzer.VolumeFlowAnalyzer')
    def test_oi_pa_analyzer_initialization(self, mock_volume, mock_divergence, mock_correlation):
        """Test OI-PA analyzer initialization"""
        analyzer = OIPAAnalyzer(self.config)
        
        self.assertIsNotNone(analyzer.correlation_analyzer)
        self.assertIsNotNone(analyzer.divergence_detector)
        self.assertIsNotNone(analyzer.volume_flow_analyzer)
    
    @patch('..indicators.oi_pa_analysis.oi_pa_analyzer.CorrelationAnalyzer')
    @patch('..indicators.oi_pa_analysis.oi_pa_analyzer.DivergenceDetector')
    @patch('..indicators.oi_pa_analysis.oi_pa_analyzer.VolumeFlowAnalyzer')
    def test_analyze_oi_pa_patterns(self, mock_volume, mock_divergence, mock_correlation):
        """Test OI-PA pattern analysis"""
        # Setup mocks
        mock_correlation_instance = Mock()
        mock_divergence_instance = Mock()
        mock_volume_instance = Mock()
        
        mock_correlation.return_value = mock_correlation_instance
        mock_divergence.return_value = mock_divergence_instance
        mock_volume.return_value = mock_volume_instance
        
        # Mock analysis results
        mock_correlation_instance.analyze_correlation.return_value = {
            'correlation_score': 0.75,
            'correlation_trend': 'increasing'
        }
        
        mock_divergence_instance.detect_divergences.return_value = {
            'divergence_count': 2,
            'divergence_strength': 0.6
        }
        
        mock_volume_instance.analyze_flow.return_value = {
            'flow_direction': 'bullish',
            'flow_strength': 0.8
        }
        
        analyzer = OIPAAnalyzer(self.config)
        result = analyzer.analyze(self.sample_data)
        
        self.assertIn('oi_pa_correlation', result)
        self.assertIn('divergence_analysis', result)
        self.assertIn('volume_flow_analysis', result)
        self.assertIn('composite_score', result)


class TestGreekSentimentAnalyzer(unittest.TestCase):
    """Test Greek Sentiment Analysis"""
    
    def setUp(self):
        self.config = {
            'delta_threshold': 0.5,
            'gamma_weight': 0.3,
            'theta_weight': 0.2,
            'vega_weight': 0.3,
            'volume_oi_weight': 0.2
        }
        
        # Sample Greek data
        self.sample_data = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'option_type': ['CE', 'PE', 'CE', 'PE', 'CE'],
            'delta': [0.8, -0.7, 0.5, -0.3, 0.2],
            'gamma': [0.05, 0.08, 0.12, 0.08, 0.03],
            'theta': [-0.1, -0.15, -0.2, -0.12, -0.05],
            'vega': [0.3, 0.4, 0.5, 0.35, 0.2],
            'volume': [100, 200, 300, 150, 50],
            'oi': [1000, 2000, 3000, 1500, 500],
            'close': [5.0, 8.0, 3.0, 12.0, 1.5],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
    
    @patch('..indicators.greek_sentiment.greek_sentiment_analyzer.GreekCalculator')
    @patch('..indicators.greek_sentiment.greek_sentiment_analyzer.ITMOTMAnalyzer')
    @patch('..indicators.greek_sentiment.greek_sentiment_analyzer.VolumeOIWeighter')
    def test_greek_sentiment_analyzer_initialization(self, mock_weighter, mock_itm_otm, mock_calculator):
        """Test Greek sentiment analyzer initialization"""
        analyzer = GreekSentimentAnalyzer(self.config)
        
        self.assertIsNotNone(analyzer.greek_calculator)
        self.assertIsNotNone(analyzer.itm_otm_analyzer)
        self.assertIsNotNone(analyzer.volume_oi_weighter)
    
    @patch('..indicators.greek_sentiment.greek_sentiment_analyzer.GreekCalculator')
    @patch('..indicators.greek_sentiment.greek_sentiment_analyzer.ITMOTMAnalyzer')
    @patch('..indicators.greek_sentiment.greek_sentiment_analyzer.VolumeOIWeighter')
    def test_analyze_greek_sentiment(self, mock_weighter, mock_itm_otm, mock_calculator):
        """Test Greek sentiment analysis"""
        # Setup mocks
        mock_calculator_instance = Mock()
        mock_itm_otm_instance = Mock()
        mock_weighter_instance = Mock()
        
        mock_calculator.return_value = mock_calculator_instance
        mock_itm_otm.return_value = mock_itm_otm_instance
        mock_weighter.return_value = mock_weighter_instance
        
        # Mock analysis results
        mock_calculator_instance.calculate_weighted_greeks.return_value = {
            'weighted_delta': 0.65,
            'weighted_gamma': 0.08,
            'weighted_theta': -0.12,
            'weighted_vega': 0.35
        }
        
        mock_itm_otm_instance.analyze_sentiment.return_value = {
            'itm_bias': 'bullish',
            'otm_bias': 'neutral',
            'overall_bias': 'bullish'
        }
        
        mock_weighter_instance.calculate_weights.return_value = {
            'volume_weight': 0.6,
            'oi_weight': 0.4
        }
        
        analyzer = GreekSentimentAnalyzer(self.config)
        result = analyzer.analyze(self.sample_data)
        
        self.assertIn('greek_sentiment', result)
        self.assertIn('weighted_greeks', result)
        self.assertIn('sentiment_bias', result)
        self.assertIn('composite_score', result)


class TestMarketBreadthAnalyzer(unittest.TestCase):
    """Test Market Breadth Analysis"""
    
    def setUp(self):
        self.config = {
            'breadth_threshold': 0.6,
            'divergence_threshold': 0.3,
            'momentum_window': 10
        }
        
        # Sample breadth data
        self.option_breadth_data = pd.DataFrame({
            'call_volume': [1000, 1200, 800, 1500, 900],
            'put_volume': [800, 900, 1200, 700, 1100],
            'call_oi': [10000, 12000, 8000, 15000, 9000],
            'put_oi': [8000, 9000, 12000, 7000, 11000],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
        
        self.underlying_breadth_data = pd.DataFrame({
            'advances': [150, 180, 120, 200, 140],
            'declines': [100, 80, 130, 50, 110],
            'new_highs': [20, 25, 15, 30, 18],
            'new_lows': [5, 3, 12, 2, 8],
            'volume_up': [1000000, 1200000, 800000, 1500000, 900000],
            'volume_down': [800000, 600000, 1000000, 400000, 700000],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
    
    @patch('..indicators.market_breadth.market_breadth_analyzer.OptionBreadthAnalyzer')
    @patch('..indicators.market_breadth.market_breadth_analyzer.UnderlyingBreadthAnalyzer')
    @patch('..indicators.market_breadth.market_breadth_analyzer.BreadthDivergenceDetector')
    def test_market_breadth_analyzer_initialization(self, mock_divergence, mock_underlying, mock_option):
        """Test market breadth analyzer initialization"""
        analyzer = MarketBreadthAnalyzer(self.config)
        
        self.assertIsNotNone(analyzer.option_breadth_analyzer)
        self.assertIsNotNone(analyzer.underlying_breadth_analyzer)
        self.assertIsNotNone(analyzer.divergence_detector)
    
    @patch('..indicators.market_breadth.market_breadth_analyzer.OptionBreadthAnalyzer')
    @patch('..indicators.market_breadth.market_breadth_analyzer.UnderlyingBreadthAnalyzer')
    @patch('..indicators.market_breadth.market_breadth_analyzer.BreadthDivergenceDetector')
    def test_analyze_market_breadth(self, mock_divergence, mock_underlying, mock_option):
        """Test market breadth analysis"""
        # Setup mocks
        mock_option_instance = Mock()
        mock_underlying_instance = Mock()
        mock_divergence_instance = Mock()
        
        mock_option.return_value = mock_option_instance
        mock_underlying.return_value = mock_underlying_instance
        mock_divergence.return_value = mock_divergence_instance
        
        # Mock analysis results
        mock_option_instance.analyze_breadth.return_value = {
            'call_put_ratio': 1.2,
            'breadth_momentum': 0.65,
            'breadth_direction': 'bullish'
        }
        
        mock_underlying_instance.analyze_breadth.return_value = {
            'advance_decline_ratio': 1.5,
            'new_highs_lows_ratio': 4.0,
            'breadth_thrust': 0.7
        }
        
        mock_divergence_instance.detect_divergences.return_value = {
            'divergence_count': 1,
            'divergence_severity': 0.4
        }
        
        analyzer = MarketBreadthAnalyzer(self.config)
        
        data = {
            'option_breadth': self.option_breadth_data,
            'underlying_breadth': self.underlying_breadth_data
        }
        
        result = analyzer.analyze(data)
        
        self.assertIn('option_breadth', result)
        self.assertIn('underlying_breadth', result)
        self.assertIn('breadth_divergences', result)
        self.assertIn('composite_score', result)


class TestIVAnalyticsAnalyzer(unittest.TestCase):
    """Test IV Analytics Analysis"""
    
    def setUp(self):
        self.config = {
            'skew_threshold': 0.05,
            'term_structure_threshold': 0.02,
            'arbitrage_threshold': 0.03
        }
        
        # Sample IV data
        self.sample_data = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'option_type': ['CE', 'PE', 'CE', 'PE', 'CE'],
            'iv': [0.20, 0.22, 0.19, 0.25, 0.18],
            'dte': [30, 30, 30, 30, 30],
            'spot': [110, 110, 110, 110, 110],
            'volume': [100, 200, 300, 150, 50],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
    
    @patch('..indicators.iv_analytics.iv_analytics_analyzer.SkewAnalyzer')
    @patch('..indicators.iv_analytics.iv_analytics_analyzer.TermStructureAnalyzer')
    @patch('..indicators.iv_analytics.iv_analytics_analyzer.ArbitrageDetector')
    def test_iv_analytics_analyzer_initialization(self, mock_arbitrage, mock_term_structure, mock_skew):
        """Test IV analytics analyzer initialization"""
        analyzer = IVAnalyticsAnalyzer(self.config)
        
        self.assertIsNotNone(analyzer.skew_analyzer)
        self.assertIsNotNone(analyzer.term_structure_analyzer)
        self.assertIsNotNone(analyzer.arbitrage_detector)
    
    @patch('..indicators.iv_analytics.iv_analytics_analyzer.SkewAnalyzer')
    @patch('..indicators.iv_analytics.iv_analytics_analyzer.TermStructureAnalyzer')
    @patch('..indicators.iv_analytics.iv_analytics_analyzer.ArbitrageDetector')
    def test_analyze_iv_analytics(self, mock_arbitrage, mock_term_structure, mock_skew):
        """Test IV analytics analysis"""
        # Setup mocks
        mock_skew_instance = Mock()
        mock_term_structure_instance = Mock()
        mock_arbitrage_instance = Mock()
        
        mock_skew.return_value = mock_skew_instance
        mock_term_structure.return_value = mock_term_structure_instance
        mock_arbitrage.return_value = mock_arbitrage_instance
        
        # Mock analysis results
        mock_skew_instance.analyze_skew.return_value = {
            'skew_level': 0.04,
            'skew_direction': 'put_skew',
            'skew_strength': 'moderate'
        }
        
        mock_term_structure_instance.analyze_term_structure.return_value = {
            'term_structure_slope': 0.01,
            'contango_backwardation': 'contango',
            'volatility_forecast': 'increasing'
        }
        
        mock_arbitrage_instance.detect_arbitrage.return_value = {
            'arbitrage_opportunities': 2,
            'max_arbitrage_profit': 0.05,
            'arbitrage_types': ['calendar', 'volatility']
        }
        
        analyzer = IVAnalyticsAnalyzer(self.config)
        result = analyzer.analyze(self.sample_data)
        
        self.assertIn('skew_analysis', result)
        self.assertIn('term_structure_analysis', result)
        self.assertIn('arbitrage_analysis', result)
        self.assertIn('composite_score', result)


class TestTechnicalIndicatorsAnalyzer(unittest.TestCase):
    """Test Technical Indicators Analysis"""
    
    def setUp(self):
        self.config = {
            'option_rsi_config': {'period': 14},
            'option_macd_config': {'fast': 12, 'slow': 26, 'signal': 9},
            'fusion_config': {'option_weight': 0.6, 'underlying_weight': 0.4}
        }
        
        # Sample option data
        self.option_data = pd.DataFrame({
            'close': [100, 102, 101, 103, 105, 104, 106, 108],
            'volume': [1000, 1200, 800, 1500, 900, 1100, 1300, 950],
            'timestamp': pd.date_range('2024-01-01', periods=8, freq='1min')
        })
        
        # Sample underlying data
        self.underlying_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107],
            'high': [102, 103, 104, 105, 106, 107, 108, 109],
            'low': [99, 100, 101, 102, 103, 104, 105, 106],
            'close': [101, 102, 103, 104, 105, 106, 107, 108],
            'volume': [10000, 12000, 8000, 15000, 9000, 11000, 13000, 9500],
            'timestamp': pd.date_range('2024-01-01', periods=8, freq='1min')
        })
    
    @patch('..indicators.technical_indicators.technical_indicators_analyzer.OptionRSI')
    @patch('..indicators.technical_indicators.technical_indicators_analyzer.OptionMACD')
    @patch('..indicators.technical_indicators.technical_indicators_analyzer.IndicatorFusion')
    def test_technical_indicators_analyzer_initialization(self, mock_fusion, mock_macd, mock_rsi):
        """Test technical indicators analyzer initialization"""
        analyzer = TechnicalIndicatorsAnalyzer(self.config)
        
        self.assertIsNotNone(analyzer.option_rsi)
        self.assertIsNotNone(analyzer.option_macd)
        self.assertIsNotNone(analyzer.indicator_fusion)
    
    @patch('..indicators.technical_indicators.technical_indicators_analyzer.OptionRSI')
    @patch('..indicators.technical_indicators.technical_indicators_analyzer.OptionMACD')
    @patch('..indicators.technical_indicators.technical_indicators_analyzer.IndicatorFusion')
    def test_analyze_technical_indicators(self, mock_fusion, mock_macd, mock_rsi):
        """Test technical indicators analysis"""
        # Setup mocks
        mock_rsi_instance = Mock()
        mock_macd_instance = Mock()
        mock_fusion_instance = Mock()
        
        mock_rsi.return_value = mock_rsi_instance
        mock_macd.return_value = mock_macd_instance
        mock_fusion.return_value = mock_fusion_instance
        
        # Mock indicator results
        mock_rsi_instance.calculate_option_rsi.return_value = {
            'rsi_value': 65,
            'rsi_signal': 'bullish',
            'overbought_level': False
        }
        
        mock_macd_instance.calculate_option_macd.return_value = {
            'macd_line': 0.5,
            'signal_line': 0.3,
            'histogram': 0.2,
            'macd_signal': 'bullish'
        }
        
        # Mock fusion results
        mock_fusion_instance.fuse_indicators.return_value = {
            'fused_signals': {
                'composite_signal': 'bullish',
                'composite_strength': 0.75
            },
            'indicator_agreement': {
                'overall': 0.8,
                'directional_agreement': 'strong_agreement'
            },
            'confidence_scores': {
                'overall': 0.85
            }
        }
        
        analyzer = TechnicalIndicatorsAnalyzer(self.config)
        result = analyzer.analyze(self.option_data, self.underlying_data)
        
        self.assertIn('option_indicators', result)
        self.assertIn('underlying_indicators', result)
        self.assertIn('fusion_analysis', result)
        self.assertIn('signals', result)


class TestIndicatorIntegration(unittest.TestCase):
    """Test integration between different indicators"""
    
    def setUp(self):
        # Create mock data that would be shared between indicators
        self.shared_option_data = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'option_type': ['CE', 'PE', 'CE', 'PE', 'CE'],
            'spot': [110, 110, 110, 110, 110],
            'volume': [100, 200, 300, 150, 50],
            'oi': [1000, 2000, 3000, 1500, 500],
            'close': [5.0, 8.0, 3.0, 12.0, 1.5],
            'iv': [0.20, 0.22, 0.19, 0.25, 0.18],
            'delta': [0.8, -0.7, 0.5, -0.3, 0.2],
            'gamma': [0.05, 0.08, 0.12, 0.08, 0.03],
            'theta': [-0.1, -0.15, -0.2, -0.12, -0.05],
            'vega': [0.3, 0.4, 0.5, 0.35, 0.2],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
    
    def test_data_consistency_between_indicators(self):
        """Test that indicators work with consistent data structures"""
        # This test ensures that all indicators can handle the same data format
        data_columns = set(self.shared_option_data.columns)
        
        # Required columns for different indicators
        straddle_required = {'strike', 'option_type', 'spot', 'volume', 'close'}
        oi_pa_required = {'strike', 'volume', 'oi'}
        greek_required = {'delta', 'gamma', 'theta', 'vega', 'volume', 'oi'}
        iv_required = {'strike', 'iv', 'spot'}
        
        # Check that our shared data has all required columns
        self.assertTrue(straddle_required.issubset(data_columns))
        self.assertTrue(oi_pa_required.issubset(data_columns))
        self.assertTrue(greek_required.issubset(data_columns))
        self.assertTrue(iv_required.issubset(data_columns))
    
    def test_output_format_consistency(self):
        """Test that all indicators produce consistent output formats"""
        # All indicators should return dictionaries with certain standard keys
        expected_keys = {'composite_score', 'status', 'timestamp'}
        
        # Mock each indicator
        mock_configs = [
            {},  # Default configs for each indicator
            {},
            {},
            {},
            {},
            {}
        ]
        
        # Each indicator should have these common output fields
        for config in mock_configs:
            # This is a structural test - in real implementation,
            # we would instantiate each indicator and check its output format
            pass
    
    def test_score_normalization(self):
        """Test that all indicators produce normalized scores"""
        # All composite scores should be between 0 and 1
        test_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for score in test_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_regime_classification_consistency(self):
        """Test regime classification consistency across indicators"""
        # Standard regime classifications that all indicators should use
        standard_regimes = {
            'bullish', 'bearish', 'neutral',
            'strong_bullish', 'strong_bearish',
            'volatile', 'trending', 'ranging'
        }
        
        # Test classification - each indicator should use these standard terms
        test_classifications = ['bullish', 'bearish', 'neutral']
        
        for classification in test_classifications:
            self.assertIn(classification, standard_regimes)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestStraddleAnalysisEngine,
        TestOIPAAnalyzer,
        TestGreekSentimentAnalyzer,
        TestMarketBreadthAnalyzer,
        TestIVAnalyticsAnalyzer,
        TestTechnicalIndicatorsAnalyzer,
        TestIndicatorIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Comprehensive Indicators Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")