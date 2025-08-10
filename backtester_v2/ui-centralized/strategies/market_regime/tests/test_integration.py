"""
Integration Tests for Backward Compatibility
===========================================

Tests to ensure the refactored system maintains backward compatibility.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add the module path
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime')

from integration.integrated_engine import analyze_market_regime, IntegratedMarketRegimeEngine

logger = logging.getLogger(__name__)

def create_test_market_data() -> pd.DataFrame:
    """Create realistic test market data"""
    # Create test data with required columns
    strikes = [19000, 19100, 19200, 19300, 19400, 19500, 19600]
    option_types = ['CE', 'PE']
    
    data = []
    base_time = datetime.now()
    
    for strike in strikes:
        for option_type in option_types:
            # Simulate realistic Greeks and market data
            if option_type == 'CE':
                delta = max(0.01, min(0.99, (19300 - strike) / 500 + 0.5))
                gamma = 0.002 * np.exp(-abs(strike - 19300) / 200)
                theta = -0.15 * gamma
                vega = 15 * gamma
                oi = np.random.randint(1000, 10000)
                volume = np.random.randint(100, 2000)
            else:  # PE
                delta = max(-0.99, min(-0.01, (strike - 19300) / 500 - 0.5))
                gamma = 0.002 * np.exp(-abs(strike - 19300) / 200)
                theta = -0.15 * gamma
                vega = 15 * gamma
                oi = np.random.randint(1000, 10000)
                volume = np.random.randint(100, 2000)
            
            data.append({
                'timestamp': base_time,
                'strike': strike,
                'option_type': option_type,
                'expiry_date': base_time + timedelta(days=15),
                'underlying_price': 19300,
                'dte': 15,
                f'{option_type.lower()}_delta': delta,
                f'{option_type.lower()}_gamma': gamma,
                f'{option_type.lower()}_theta': theta,
                f'{option_type.lower()}_vega': vega,
                f'{option_type.lower()}_oi': oi,
                f'{option_type.lower()}_volume': volume,
                f'{option_type.lower()}_close': max(1, 50 + np.random.normal(0, 10))
            })
    
    return pd.DataFrame(data)

def test_backward_compatibility():
    """Test that the original interface still works"""
    print("🧪 Testing Backward Compatibility")
    print("=" * 50)
    
    try:
        # Create test data
        market_data = create_test_market_data()
        print(f"✅ Created test market data: {len(market_data)} rows")
        
        # Test original function interface
        result = analyze_market_regime(
            market_data,
            spot_price=19300,
            dte=15,
            volatility=0.25
        )
        
        # Verify original interface structure
        required_keys = [
            'market_regime', 'regime_score', 'confidence', 
            'indicator_results', 'adaptive_weights', 'computation_time', 
            'timestamp', 'metadata'
        ]
        
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"❌ Missing keys in result: {missing_keys}")
            return False
        
        print(f"✅ All required keys present in result")
        print(f"✅ Market regime: {result['market_regime']}")
        print(f"✅ Regime score: {result['regime_score']:.3f}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        print(f"✅ Computation time: {result['computation_time']:.3f}s")
        print(f"✅ Indicators processed: {len(result['indicator_results'])}")
        
        # Verify backward compatibility flag
        if result['metadata'].get('backward_compatible') is True:
            print("✅ Backward compatibility confirmed")
        else:
            print("❌ Backward compatibility flag missing")
            return False
        
        print("\n✅ Backward Compatibility Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Backward Compatibility Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_engine():
    """Test the integrated engine directly"""
    print("\n🧪 Testing Integrated Engine")
    print("=" * 50)
    
    try:
        # Create engine instance
        engine = IntegratedMarketRegimeEngine()
        print("✅ Created IntegratedMarketRegimeEngine")
        
        # Create test data
        market_data = create_test_market_data()
        
        # Test analysis
        result = engine.analyze_market_regime(
            market_data,
            spot_price=19300,
            dte=15,
            volatility=0.25
        )
        
        print(f"✅ Analysis completed: regime={result['market_regime']}")
        print(f"✅ Score: {result['regime_score']:.3f}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        
        # Test performance summary
        performance = engine.get_performance_summary()
        print(f"✅ Performance summary: {len(performance)} indicators")
        
        # Test weight recommendations
        recommendations = engine.get_weight_recommendations()
        print(f"✅ Weight recommendations: {len(recommendations)} indicators")
        
        print("\n✅ Integrated Engine Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated Engine Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_tracking():
    """Test performance tracking functionality"""
    print("\n🧪 Testing Performance Tracking")
    print("=" * 50)
    
    try:
        engine = IntegratedMarketRegimeEngine()
        market_data = create_test_market_data()
        
        # Run multiple analyses to build performance history
        results = []
        for i in range(5):
            result = engine.analyze_market_regime(
                market_data,
                spot_price=19300 + np.random.normal(0, 50),
                dte=15,
                volatility=0.25
            )
            results.append(result)
        
        print(f"✅ Completed {len(results)} analyses")
        
        # Check performance tracking
        performance = engine.get_performance_summary()
        for indicator_name, metrics in performance.items():
            print(f"✅ {indicator_name}: accuracy={metrics.accuracy:.3f}, confidence={metrics.avg_confidence:.3f}")
        
        print("\n✅ Performance Tracking Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Performance Tracking Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_integration_tests():
    """Run all integration tests"""
    print("🚀 Running All Integration Tests")
    print("=" * 60)
    
    tests = [
        test_backward_compatibility,
        test_integrated_engine,
        test_performance_tracking
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Integration Test Summary")
    print("=" * 30)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED")
    
    return passed == total

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = run_all_integration_tests()
    
    exit(0 if success else 1)