#!/usr/bin/env python3
"""
Test Call-Put OI Correlation Analysis

This script specifically tests the implementation of call OI vs put OI correlation
analysis for market regime formation, which was missing from the initial implementation.

Test Coverage:
- Call OI with PA correlation calculation
- Put OI with PA correlation calculation  
- Call-Put OI correlation analysis
- Regime classification based on correlation patterns
- Market regime formation logic
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_call_put_oi_correlation_calculation():
    """Test the core call-put OI correlation calculation"""
    logger.info("🔍 Testing Call-Put OI Correlation Calculation...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        analyzer = EnhancedTrendingOIWithPAAnalysis()
        
        # Test case 1: High correlation scenario
        call_oi_velocities = [0.05, 0.03, 0.07, 0.04, 0.06]  # Consistent positive OI changes
        put_oi_velocities = [0.04, 0.02, 0.06, 0.03, 0.05]   # Similar pattern
        call_price_velocities = [0.02, 0.01, 0.03, 0.015, 0.025]  # Positive price changes
        put_price_velocities = [0.018, 0.012, 0.028, 0.016, 0.024]  # Similar pattern
        
        correlation = analyzer._calculate_call_put_oi_correlation(
            call_oi_velocities, put_oi_velocities, 
            call_price_velocities, put_price_velocities
        )
        
        assert 0.0 <= abs(correlation) <= 1.0, f"Correlation out of range: {correlation}"
        assert correlation > 0.5, f"Expected high correlation, got: {correlation}"
        
        # Test case 2: Low correlation scenario
        call_oi_velocities_low = [0.05, -0.03, 0.07, -0.04, 0.06]  # Mixed pattern
        put_oi_velocities_low = [-0.04, 0.02, -0.06, 0.03, -0.05]  # Opposite pattern
        call_price_velocities_low = [0.02, -0.01, 0.03, -0.015, 0.025]  # Mixed
        put_price_velocities_low = [-0.018, 0.012, -0.028, 0.016, -0.024]  # Opposite
        
        correlation_low = analyzer._calculate_call_put_oi_correlation(
            call_oi_velocities_low, put_oi_velocities_low,
            call_price_velocities_low, put_price_velocities_low
        )
        
        assert 0.0 <= abs(correlation_low) <= 1.0, f"Low correlation out of range: {correlation_low}"
        
        logger.info(f"✅ High correlation scenario: {correlation:.3f}")
        logger.info(f"✅ Low correlation scenario: {correlation_low:.3f}")
        logger.info("✅ Call-Put OI Correlation Calculation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Call-Put OI Correlation Calculation test FAILED: {e}")
        return False

def test_oi_correlation_regime_classification():
    """Test regime classification based on OI correlation"""
    logger.info("🔍 Testing OI Correlation Regime Classification...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        analyzer = EnhancedTrendingOIWithPAAnalysis()
        
        # Test different regime scenarios
        test_cases = [
            # (call_put_divergence, signal, expected_regime_type)
            (0.1, 0.7, 'TRENDING_BULLISH_HIGH_CORRELATION'),    # High correlation, strong bullish
            (0.1, -0.7, 'TRENDING_BEARISH_HIGH_CORRELATION'),   # High correlation, strong bearish
            (0.1, 0.2, 'SIDEWAYS_HIGH_CORRELATION'),            # High correlation, weak signal
            (0.8, 0.7, 'DIVERGENT_BULLISH_LOW_CORRELATION'),    # Low correlation, strong bullish
            (0.8, -0.7, 'DIVERGENT_BEARISH_LOW_CORRELATION'),   # Low correlation, strong bearish
            (0.8, 0.2, 'UNCERTAIN_LOW_CORRELATION'),            # Low correlation, weak signal
            (0.5, 0.5, 'TRANSITIONAL_BULLISH_MEDIUM_CORRELATION'), # Medium correlation, medium bullish
            (0.5, -0.5, 'TRANSITIONAL_BEARISH_MEDIUM_CORRELATION'), # Medium correlation, medium bearish
            (0.5, 0.1, 'NEUTRAL_MEDIUM_CORRELATION')            # Medium correlation, neutral
        ]
        
        for divergence, signal, expected_regime in test_cases:
            regime = analyzer._classify_oi_correlation_regime(divergence, signal)
            assert regime == expected_regime, f"Regime mismatch: expected {expected_regime}, got {regime}"
            logger.info(f"✅ Divergence: {divergence:.1f}, Signal: {signal:.1f} → {regime}")
        
        logger.info("✅ OI Correlation Regime Classification test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ OI Correlation Regime Classification test FAILED: {e}")
        return False

def test_enhanced_call_put_divergence():
    """Test the enhanced call-put divergence with correlation analysis"""
    logger.info("🔍 Testing Enhanced Call-Put Divergence Analysis...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        analyzer = EnhancedTrendingOIWithPAAnalysis()
        
        # Generate test pattern results with correlation characteristics
        pattern_results = generate_correlated_pattern_results()
        
        # Test enhanced divergence calculation
        divergence = analyzer._calculate_call_put_divergence(pattern_results)
        
        # Validate divergence score
        assert 0.0 <= divergence <= 1.0, f"Divergence score out of range: {divergence}"
        
        logger.info(f"✅ Enhanced call-put divergence: {divergence:.3f}")
        logger.info("✅ Enhanced Call-Put Divergence Analysis test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Call-Put Divergence Analysis test FAILED: {e}")
        return False

def test_complete_oi_correlation_integration():
    """Test complete integration of OI correlation analysis"""
    logger.info("🔍 Testing Complete OI Correlation Integration...")
    
    try:
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        
        # Initialize analyzer
        analyzer = EnhancedTrendingOIWithPAAnalysis({
            'divergence_threshold': 0.3,
            'strike_range': 5
        })
        
        # Generate market data with correlation characteristics
        market_data = generate_correlation_market_data()
        
        # Run complete analysis
        result = analyzer.analyze_trending_oi_pa(market_data)
        
        # Validate results include correlation analysis
        assert 'oi_signal' in result, "Missing OI signal"
        assert 'divergence_analysis' in result, "Missing divergence analysis"
        assert 'call_put_divergence' in result['divergence_analysis'], "Missing call-put divergence"
        
        # Check if correlation regime is included
        if 'oi_correlation_regime' in result:
            regime = result['oi_correlation_regime']
            logger.info(f"✅ OI Correlation Regime: {regime}")
        
        call_put_divergence = result['divergence_analysis']['call_put_divergence']
        logger.info(f"✅ Call-Put Divergence: {call_put_divergence:.3f}")
        
        logger.info("✅ Complete OI Correlation Integration test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete OI Correlation Integration test FAILED: {e}")
        return False

def test_regime_formation_with_correlation():
    """Test market regime formation with correlation analysis"""
    logger.info("🔍 Testing Market Regime Formation with Correlation...")
    
    try:
        # Test different market scenarios
        scenarios = [
            ("High Correlation Trending", generate_high_correlation_trending_data()),
            ("Low Correlation Divergent", generate_low_correlation_divergent_data()),
            ("Medium Correlation Transitional", generate_medium_correlation_transitional_data())
        ]
        
        from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
        analyzer = EnhancedTrendingOIWithPAAnalysis()
        
        for scenario_name, market_data in scenarios:
            logger.info(f"  📊 Testing scenario: {scenario_name}")
            
            result = analyzer.analyze_trending_oi_pa(market_data)
            
            # Validate regime formation
            assert 'oi_signal' in result, f"Missing OI signal for {scenario_name}"
            assert 'confidence' in result, f"Missing confidence for {scenario_name}"
            
            oi_signal = result['oi_signal']
            confidence = result['confidence']
            
            logger.info(f"    Signal: {oi_signal:.3f}, Confidence: {confidence:.3f}")
        
        logger.info("✅ Market Regime Formation with Correlation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Market Regime Formation with Correlation test FAILED: {e}")
        return False

def generate_correlated_pattern_results():
    """Generate test pattern results with correlation characteristics"""
    from enhanced_trending_oi_pa_analysis import OIAnalysisResult, OIPattern
    
    pattern_results = {}
    
    # Generate correlated call patterns
    for i in range(3):
        strike = 18500 + (i - 1) * 100
        
        # Call pattern
        call_result = OIAnalysisResult(
            pattern=OIPattern.LONG_BUILD_UP,
            confidence=0.8,
            signal_strength=0.7,
            divergence_score=0.0,
            institutional_ratio=0.0,
            timeframe_consistency=0.0,
            supporting_metrics={
                'option_type': 'call',
                'oi_velocity': 0.05 + i * 0.01,
                'price_velocity': 0.02 + i * 0.005,
                'strike': strike
            }
        )
        pattern_results[f'{strike}_CE'] = call_result
        
        # Put pattern (correlated)
        put_result = OIAnalysisResult(
            pattern=OIPattern.SHORT_COVERING,
            confidence=0.8,
            signal_strength=0.7,
            divergence_score=0.0,
            institutional_ratio=0.0,
            timeframe_consistency=0.0,
            supporting_metrics={
                'option_type': 'put',
                'oi_velocity': 0.04 + i * 0.01,  # Similar to calls
                'price_velocity': 0.018 + i * 0.005,  # Similar to calls
                'strike': strike
            }
        )
        pattern_results[f'{strike}_PE'] = put_result
    
    return pattern_results

def generate_correlation_market_data():
    """Generate market data with correlation characteristics"""
    np.random.seed(123)  # For reproducible results
    
    underlying_price = 18500
    strikes = [18400, 18500, 18600]
    
    options_data = {}
    for strike in strikes:
        # Generate correlated OI and price changes
        base_oi_change = 0.05 if strike >= underlying_price else 0.03
        base_price_change = 0.02 if strike >= underlying_price else 0.015
        
        options_data[strike] = {
            'CE': {
                'close': max(0, underlying_price - strike + 50),
                'previous_close': max(0, underlying_price - strike + 48),
                'volume': np.random.randint(2000, 8000),
                'oi': np.random.randint(20000, 80000),
                'previous_oi': np.random.randint(18000, 75000),
                'iv': 0.16 + np.random.normal(0, 0.02)
            },
            'PE': {
                'close': max(0, strike - underlying_price + 50),
                'previous_close': max(0, strike - underlying_price + 48),
                'volume': np.random.randint(2000, 8000),
                'oi': np.random.randint(20000, 80000),
                'previous_oi': np.random.randint(18000, 75000),
                'iv': 0.16 + np.random.normal(0, 0.02)
            }
        }
    
    # Generate price history
    price_history = []
    for i in range(50):
        price = underlying_price + np.random.normal(0, 30)
        price_history.append({
            'close': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'volume': np.random.randint(1000, 5000),
            'timestamp': datetime.now() - timedelta(minutes=50-i)
        })
    
    return {
        'underlying_price': underlying_price,
        'strikes': strikes,
        'options_data': options_data,
        'price_history': price_history,
        'volatility': 0.16,
        'timestamp': datetime.now()
    }

def generate_high_correlation_trending_data():
    """Generate data showing high correlation trending pattern"""
    return generate_correlation_market_data()  # Base case is high correlation

def generate_low_correlation_divergent_data():
    """Generate data showing low correlation divergent pattern"""
    market_data = generate_correlation_market_data()
    
    # Modify to create divergent patterns
    for strike, option_data in market_data['options_data'].items():
        # Make calls and puts divergent
        option_data['CE']['previous_oi'] = option_data['CE']['oi'] * 1.1  # OI decreasing
        option_data['PE']['previous_oi'] = option_data['PE']['oi'] * 0.9  # OI increasing
        
        option_data['CE']['previous_close'] = option_data['CE']['close'] * 1.05  # Price decreasing
        option_data['PE']['previous_close'] = option_data['PE']['close'] * 0.95  # Price increasing
    
    return market_data

def generate_medium_correlation_transitional_data():
    """Generate data showing medium correlation transitional pattern"""
    market_data = generate_correlation_market_data()
    
    # Modify to create mixed correlation patterns
    for i, (strike, option_data) in enumerate(market_data['options_data'].items()):
        if i % 2 == 0:  # Even strikes - correlated
            option_data['CE']['previous_oi'] = option_data['CE']['oi'] * 0.95
            option_data['PE']['previous_oi'] = option_data['PE']['oi'] * 0.95
        else:  # Odd strikes - divergent
            option_data['CE']['previous_oi'] = option_data['CE']['oi'] * 1.05
            option_data['PE']['previous_oi'] = option_data['PE']['oi'] * 0.95
    
    return market_data

def main():
    """Main test function"""
    logger.info("🚀 Starting Call-Put OI Correlation Analysis Tests...")
    
    tests = [
        ("Call-Put OI Correlation Calculation", test_call_put_oi_correlation_calculation),
        ("OI Correlation Regime Classification", test_oi_correlation_regime_classification),
        ("Enhanced Call-Put Divergence", test_enhanced_call_put_divergence),
        ("Complete OI Correlation Integration", test_complete_oi_correlation_integration),
        ("Regime Formation with Correlation", test_regime_formation_with_correlation)
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
    logger.info("📊 CALL-PUT OI CORRELATION TEST SUMMARY")
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
        logger.info("🎉 ALL CALL-PUT OI CORRELATION TESTS PASSED!")
        logger.info("✅ Missing correlation analysis successfully implemented")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please review and fix issues.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
