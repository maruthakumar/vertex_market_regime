#!/usr/bin/env python3
"""
Enhanced Integration Test for Complete Market Regime System

This script tests the complete integration of:
1. Enhanced Trending OI with PA Analysis
2. Enhanced Greek Sentiment Analysis  
3. Triple Straddle Analysis
4. Enhanced 18-Regime Detector V2

Test Coverage:
- Corrected OI pattern recognition
- Multi-timeframe analysis
- Comprehensive divergence detection
- Greek sentiment with baseline tracking
- Complete regime formation
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_trending_oi_analysis():
    """Test Enhanced Trending OI with PA Analysis"""
    logger.info("🔍 Testing Enhanced Trending OI with PA Analysis...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        # Initialize analyzer
        config = {
            'strike_range': 7,
            'primary_timeframe': 3,
            'confirmation_timeframe': 15,
            'divergence_threshold': 0.3
        }
        analyzer = EnhancedTrendingOIWithPAAnalysis(config)
        
        # Generate test market data
        market_data = generate_comprehensive_market_data()
        
        # Test analysis
        result = analyzer.analyze_trending_oi_pa(market_data)
        
        # Validate results
        assert 'oi_signal' in result, "Missing OI signal"
        assert 'confidence' in result, "Missing confidence"
        assert 'pattern_breakdown' in result, "Missing pattern breakdown"
        assert 'divergence_analysis' in result, "Missing divergence analysis"
        assert 'institutional_analysis' in result, "Missing institutional analysis"
        
        # Validate signal range
        assert -1.0 <= result['oi_signal'] <= 1.0, "OI signal out of range"
        assert 0.0 <= result['confidence'] <= 1.0, "Confidence out of range"
        
        logger.info("✅ Enhanced Trending OI Analysis test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Trending OI Analysis test FAILED: {e}")
        return False

def test_enhanced_greek_sentiment_analysis():
    """Test Enhanced Greek Sentiment Analysis"""
    logger.info("🔍 Testing Enhanced Greek Sentiment Analysis...")
    
    try:
        from enhanced_greek_sentiment_analysis import GreekSentimentAnalyzerAnalysis
        
        # Initialize analyzer
        config = {
            'learning_rate': 0.05,
            'baseline_update_frequency': 30
        }
        analyzer = GreekSentimentAnalyzerAnalysis(config)
        
        # Generate test market data with Greeks
        market_data = generate_comprehensive_market_data()
        
        # Test analysis
        result = analyzer.analyze_greek_sentiment(market_data)
        
        # Validate results
        assert 'sentiment_score' in result, "Missing sentiment score"
        assert 'sentiment_type' in result, "Missing sentiment type"
        assert 'confidence' in result, "Missing confidence"
        assert 'greek_contributions' in result, "Missing Greek contributions"
        assert 'baseline_changes' in result, "Missing baseline changes"
        
        # Validate score range
        assert -1.0 <= result['sentiment_score'] <= 1.0, "Sentiment score out of range"
        assert 0.0 <= result['confidence'] <= 1.0, "Confidence out of range"
        
        # Validate Greek contributions
        greek_contributions = result['greek_contributions']
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            assert greek in greek_contributions, f"Missing {greek} contribution"
            assert -1.0 <= greek_contributions[greek] <= 1.0, f"{greek} contribution out of range"
        
        logger.info("✅ Enhanced Greek Sentiment Analysis test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Greek Sentiment Analysis test FAILED: {e}")
        return False

def test_corrected_oi_pattern_logic():
    """Test the corrected OI pattern recognition logic"""
    logger.info("🔍 Testing Corrected OI Pattern Logic...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        analyzer = EnhancedTrendingOIWithPAAnalysis()
        
        # Test corrected pattern logic
        test_cases = [
            # (oi_velocity, price_velocity, expected_pattern)
            (0.05, 0.02, 'Long_Build_Up'),      # OI↑ + Price↑ = Bullish
            (-0.05, -0.02, 'Long_Unwinding'),   # OI↓ + Price↓ = Bearish
            (0.05, -0.02, 'Short_Build_Up'),    # OI↑ + Price↓ = Bearish
            (-0.05, 0.02, 'Short_Covering'),    # OI↓ + Price↑ = Bullish
            (0.01, 0.005, 'Neutral')            # Small changes = Neutral
        ]
        
        for oi_vel, price_vel, expected in test_cases:
            pattern = analyzer._classify_oi_pattern_corrected(oi_vel, price_vel)
            assert pattern.value == expected, f"Pattern mismatch: expected {expected}, got {pattern.value}"
        
        logger.info("✅ Corrected OI Pattern Logic test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Corrected OI Pattern Logic test FAILED: {e}")
        return False

def test_multi_timeframe_analysis():
    """Test multi-timeframe analysis (3min + 15min)"""
    logger.info("🔍 Testing Multi-Timeframe Analysis...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        analyzer = EnhancedTrendingOIWithPAAnalysis({
            'primary_timeframe': 3,
            'confirmation_timeframe': 15,
            'timeframe_weights': {3: 0.4, 15: 0.6}
        })
        
        # Generate market data with sufficient history
        market_data = generate_comprehensive_market_data(history_length=300)
        
        # Test multi-timeframe analysis
        prepared_data = analyzer._prepare_market_data(market_data)
        multi_tf_result = analyzer._analyze_multi_timeframe(prepared_data)
        
        # Validate results
        assert hasattr(multi_tf_result, 'primary_signal'), "Missing primary signal"
        assert hasattr(multi_tf_result, 'confirmation_signal'), "Missing confirmation signal"
        assert hasattr(multi_tf_result, 'combined_signal'), "Missing combined signal"
        assert hasattr(multi_tf_result, 'divergence_flag'), "Missing divergence flag"
        
        # Validate signal ranges
        assert -1.0 <= multi_tf_result.primary_signal <= 1.0, "Primary signal out of range"
        assert -1.0 <= multi_tf_result.confirmation_signal <= 1.0, "Confirmation signal out of range"
        assert -1.0 <= multi_tf_result.combined_signal <= 1.0, "Combined signal out of range"
        
        logger.info("✅ Multi-Timeframe Analysis test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-Timeframe Analysis test FAILED: {e}")
        return False

def test_divergence_detection():
    """Test comprehensive divergence detection"""
    logger.info("🔍 Testing Comprehensive Divergence Detection...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        analyzer = EnhancedTrendingOIWithPAAnalysis({'divergence_threshold': 0.3})
        
        # Generate market data
        market_data = generate_comprehensive_market_data()
        prepared_data = analyzer._prepare_market_data(market_data)
        
        # Generate pattern results
        pattern_results = analyzer._identify_oi_patterns_corrected(prepared_data)
        
        # Test divergence detection
        divergence_analysis = analyzer._detect_comprehensive_divergence(prepared_data, pattern_results)
        
        # Validate divergence analysis
        expected_keys = [
            'pattern_divergence', 'oi_price_divergence', 'call_put_divergence',
            'cross_strike_divergence', 'overall_divergence', 'divergence_flags'
        ]
        
        for key in expected_keys:
            assert key in divergence_analysis, f"Missing divergence key: {key}"
        
        # Validate divergence scores
        for key in expected_keys[:-1]:  # Exclude divergence_flags
            score = divergence_analysis[key]
            assert 0.0 <= score <= 1.0, f"Divergence score {key} out of range: {score}"
        
        logger.info("✅ Comprehensive Divergence Detection test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive Divergence Detection test FAILED: {e}")
        return False

def test_institutional_retail_detection():
    """Test institutional vs retail detection"""
    logger.info("🔍 Testing Institutional vs Retail Detection...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        analyzer = EnhancedTrendingOIWithPAAnalysis()
        
        # Generate market data with institutional characteristics
        market_data = generate_institutional_market_data()
        prepared_data = analyzer._prepare_market_data(market_data)
        
        # Test institutional analysis
        institutional_analysis = analyzer._analyze_institutional_retail(prepared_data)
        
        # Validate results
        assert 'institutional_ratio' in institutional_analysis, "Missing institutional ratio"
        assert 'positioning_bias' in institutional_analysis, "Missing positioning bias"
        assert 'confidence' in institutional_analysis, "Missing confidence"
        
        # Validate ranges
        ratio = institutional_analysis['institutional_ratio']
        assert 0.0 <= ratio <= 1.0, f"Institutional ratio out of range: {ratio}"
        
        confidence = institutional_analysis['confidence']
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
        
        logger.info("✅ Institutional vs Retail Detection test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Institutional vs Retail Detection test FAILED: {e}")
        return False

def test_complete_system_integration():
    """Test complete system integration"""
    logger.info("🔍 Testing Complete System Integration...")
    
    try:
        from enhanced_regime_detector_v2 import Enhanced18RegimeDetectorV2
        
        # Initialize complete system
        config = {
            'learning_rate': 0.05,
            'performance_window': 20,
            'strike_range': 7
        }
        detector = Enhanced18RegimeDetectorV2(config)
        
        # Generate comprehensive market data
        market_data = generate_comprehensive_market_data()
        
        # Test complete regime detection
        result = detector.detect_regime(market_data)
        
        # Validate complete result
        assert hasattr(result, 'regime_type'), "Missing regime type"
        assert hasattr(result, 'regime_score'), "Missing regime score"
        assert hasattr(result, 'confidence'), "Missing confidence"
        assert hasattr(result, 'indicator_breakdown'), "Missing indicator breakdown"
        
        # Validate indicator breakdown
        expected_indicators = [
            'triple_straddle_analysis', 'greek_sentiment', 'oi_analysis'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result.indicator_breakdown, f"Missing indicator: {indicator}"
            indicator_result = result.indicator_breakdown[indicator]
            assert 'score' in indicator_result, f"Missing score for {indicator}"
            assert 'confidence' in indicator_result, f"Missing confidence for {indicator}"
        
        # Validate score ranges
        assert -1.0 <= result.regime_score <= 1.0, "Regime score out of range"
        assert 0.0 <= result.confidence <= 1.0, "Confidence out of range"
        
        logger.info("✅ Complete System Integration test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete System Integration test FAILED: {e}")
        return False

def generate_comprehensive_market_data(history_length=100):
    """Generate comprehensive test market data"""
    np.random.seed(42)
    
    underlying_price = 18500 + np.random.normal(0, 100)
    strikes = [underlying_price - 200, underlying_price - 100, underlying_price, 
               underlying_price + 100, underlying_price + 200]
    
    # Generate options data with Greeks
    options_data = {}
    for i, strike in enumerate(strikes):
        ce_price = max(0, underlying_price - strike + np.random.normal(0, 10))
        pe_price = max(0, strike - underlying_price + np.random.normal(0, 10))
        
        options_data[strike] = {
            'CE': {
                'close': ce_price,
                'previous_close': ce_price * (1 + np.random.normal(0, 0.02)),
                'volume': np.random.randint(1000, 10000),
                'oi': np.random.randint(10000, 100000),
                'previous_oi': np.random.randint(8000, 95000),
                'iv': 0.15 + np.random.normal(0, 0.05),
                'delta': 0.5 + (i - 2) * 0.2 + np.random.normal(0, 0.05),
                'gamma': 0.02 + np.random.normal(0, 0.005),
                'theta': -0.1 + np.random.normal(0, 0.02),
                'vega': 0.3 + np.random.normal(0, 0.05)
            },
            'PE': {
                'close': pe_price,
                'previous_close': pe_price * (1 + np.random.normal(0, 0.02)),
                'volume': np.random.randint(1000, 10000),
                'oi': np.random.randint(10000, 100000),
                'previous_oi': np.random.randint(8000, 95000),
                'iv': 0.15 + np.random.normal(0, 0.05),
                'delta': -0.5 + (2 - i) * 0.2 + np.random.normal(0, 0.05),
                'gamma': 0.02 + np.random.normal(0, 0.005),
                'theta': -0.1 + np.random.normal(0, 0.02),
                'vega': 0.3 + np.random.normal(0, 0.05)
            }
        }
    
    # Generate price history
    price_history = []
    for i in range(history_length):
        price = underlying_price + np.random.normal(0, 50)
        price_history.append({
            'close': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'volume': np.random.randint(1000, 5000),
            'timestamp': datetime.now() - timedelta(minutes=history_length-i)
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
        'volatility': 0.15 + np.random.normal(0, 0.05),
        'dte': np.random.randint(1, 45),
        'timestamp': datetime.now()
    }

def generate_institutional_market_data():
    """Generate market data with institutional characteristics"""
    market_data = generate_comprehensive_market_data()
    
    # Modify to show institutional characteristics (high OI/volume ratios)
    for strike, option_data in market_data['options_data'].items():
        # Institutional: High OI, Low Volume
        option_data['CE']['oi'] = np.random.randint(50000, 200000)
        option_data['CE']['volume'] = np.random.randint(100, 2000)  # Low volume
        option_data['PE']['oi'] = np.random.randint(50000, 200000)
        option_data['PE']['volume'] = np.random.randint(100, 2000)  # Low volume
    
    return market_data

def main():
    """Main test function"""
    logger.info("🚀 Starting Enhanced Market Regime System Integration Tests...")
    
    tests = [
        ("Enhanced Trending OI Analysis", test_enhanced_trending_oi_analysis),
        ("Enhanced Greek Sentiment Analysis", test_enhanced_greek_sentiment_analysis),
        ("Corrected OI Pattern Logic", test_corrected_oi_pattern_logic),
        ("Multi-Timeframe Analysis", test_multi_timeframe_analysis),
        ("Divergence Detection", test_divergence_detection),
        ("Institutional vs Retail Detection", test_institutional_retail_detection),
        ("Complete System Integration", test_complete_system_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 ENHANCED INTEGRATION TEST SUMMARY")
    logger.info("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("="*80)
    logger.info(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL ENHANCED INTEGRATION TESTS PASSED!")
        logger.info("✅ System ready for production deployment")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please review and fix issues.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
