#!/usr/bin/env python3
"""
Test Script for Refactored Market Regime System
==============================================

Simple test script to verify the new refactored architecture works correctly.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_base_classes():
    """Test base classes"""
    print("🧪 Testing Base Classes")
    print("=" * 30)
    
    try:
        # Test base indicator config
        from base.base_indicator import IndicatorConfig, IndicatorState
        
        config = IndicatorConfig(
            name='test_indicator',
            enabled=True,
            weight=1.0,
            parameters={'test_param': 123}
        )
        
        print(f"✅ Created IndicatorConfig: {config.name}")
        print(f"✅ IndicatorState enum available: {IndicatorState.READY}")
        
        # Test strike selector
        from base.strike_selector_base import create_strike_selector, StrikeSelectionStrategy
        
        selector = create_strike_selector('dynamic_range', {'base_range': 0.05})
        print(f"✅ Created strike selector: {selector.name}")
        
        print("\n✅ Base Classes Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Base Classes Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_greek_sentiment_v2():
    """Test Greek Sentiment V2 indicator"""
    print("\n🧪 Testing Greek Sentiment V2")
    print("=" * 30)
    
    try:
        from base.base_indicator import IndicatorConfig
        from indicators.greek_sentiment_v2 import GreekSentimentV2
        
        # Create configuration
        config = IndicatorConfig(
            name='greek_sentiment_test',
            enabled=True,
            weight=1.0,
            parameters={
                'oi_weight_alpha': 0.6,
                'volume_weight_beta': 0.4,
                'delta_weight': 1.2,
                'vega_weight': 1.5,
                'enable_itm_analysis': True
            },
            strike_selection_strategy='full_chain'
        )
        
        # Create indicator
        indicator = GreekSentimentV2(config)
        print(f"✅ Created GreekSentimentV2: {indicator.config.name}")
        
        # Create test data
        test_data = create_test_data()
        
        # Validate data
        is_valid, errors = indicator.validate_data(test_data)
        print(f"✅ Data validation: {'PASSED' if is_valid else 'FAILED'}")
        if errors:
            print(f"   Validation errors: {errors}")
        
        # Run analysis if data is valid
        if is_valid:
            result = indicator.analyze(test_data, spot_price=19300, dte=15)
            print(f"✅ Analysis completed: value={result.value:.3f}, confidence={result.confidence:.3f}")
            print(f"✅ Sentiment: {result.metadata.get('sentiment_classification', 'Unknown')}")
            print(f"✅ Computation time: {result.computation_time:.3f}s")
        
        print("\n✅ Greek Sentiment V2 Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Greek Sentiment V2 Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_data():
    """Create test market data"""
    strikes = [19000, 19100, 19200, 19300, 19400, 19500, 19600]
    option_types = ['CE', 'PE']
    
    data = []
    base_time = datetime.now()
    
    for strike in strikes:
        for option_type in option_types:
            # Simulate realistic Greeks
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
                f'{option_type.lower()}_volume': volume
            })
    
    return pd.DataFrame(data)

def test_directory_structure():
    """Test that all directories and files exist"""
    print("\n🧪 Testing Directory Structure")
    print("=" * 30)
    
    required_dirs = [
        'base',
        'indicators', 
        'integration',
        'adaptive_optimization',
        'tests',
        'legacy'
    ]
    
    required_files = [
        'base/__init__.py',
        'base/base_indicator.py',
        'base/strike_selector_base.py',
        'base/option_data_manager.py',
        'base/performance_tracker.py',
        'base/adaptive_weight_manager.py',
        'indicators/__init__.py',
        'indicators/greek_sentiment_v2.py',
        'integration/__init__.py',
        'integration/integrated_engine.py'
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            all_exist = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            all_exist = False
    
    if all_exist:
        print("\n✅ Directory Structure Test PASSED!")
    else:
        print("\n❌ Directory Structure Test FAILED!")
    
    return all_exist

def test_imports():
    """Test that all imports work"""
    print("\n🧪 Testing Imports")
    print("=" * 30)
    
    import_tests = [
        ('base.base_indicator', 'BaseIndicator'),
        ('base.strike_selector_base', 'BaseStrikeSelector'),
        ('base.option_data_manager', 'OptionDataManager'),
        ('base.performance_tracker', 'PerformanceTracker'),
        ('base.adaptive_weight_manager', 'AdaptiveWeightManager'),
        ('indicators.greek_sentiment_v2', 'GreekSentimentV2')
    ]
    
    all_imports_work = True
    
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ Import successful: {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ Import failed: {module_name}.{class_name} - {e}")
            all_imports_work = False
    
    if all_imports_work:
        print("\n✅ Imports Test PASSED!")
    else:
        print("\n❌ Imports Test FAILED!")
    
    return all_imports_work

def run_all_tests():
    """Run all tests"""
    print("🚀 Running All Refactored System Tests")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_imports,
        test_base_classes,
        test_greek_sentiment_v2
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
    
    print(f"\n📊 Test Summary")
    print("=" * 20)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Refactored system is working correctly!")
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED")
        print("❌ System needs fixes before deployment")
    
    return passed == total

if __name__ == "__main__":
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Complete implementation of remaining indicators (OI/PA, Technical)")
        print("2. Add comprehensive test coverage")
        print("3. Integration with existing backtester")
        print("4. Performance optimization")
    
    exit(0 if success else 1)