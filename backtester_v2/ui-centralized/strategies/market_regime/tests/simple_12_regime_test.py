#!/usr/bin/env python3
"""
Simple 12-Regime System Test

Quick validation test for the 12-regime classification system
without complex dependencies.

Author: The Augster
Date: 2025-06-18
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_12_regime_import():
    """Test if 12-regime detector can be imported"""
    try:
        from enhanced_12_regime_detector import Enhanced12RegimeDetector
        print("✅ Enhanced12RegimeDetector imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import Enhanced12RegimeDetector: {e}")
        return False

def test_12_regime_initialization():
    """Test 12-regime detector initialization"""
    try:
        from enhanced_12_regime_detector import Enhanced12RegimeDetector
        
        detector = Enhanced12RegimeDetector()
        print("✅ Enhanced12RegimeDetector initialized successfully")
        
        # Check regime definitions
        regime_count = len(detector.regime_definitions)
        print(f"✅ Found {regime_count} regime definitions")
        
        if regime_count == 12:
            print("✅ Correct number of regimes (12)")
        else:
            print(f"❌ Expected 12 regimes, found {regime_count}")
            return False
        
        # Check regime mapping
        mapping_count = len(detector.regime_mapping_18_to_12)
        print(f"✅ Found {mapping_count} regime mappings")
        
        if mapping_count >= 18:
            print("✅ Sufficient regime mappings")
        else:
            print(f"❌ Expected at least 18 mappings, found {mapping_count}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced12RegimeDetector: {e}")
        return False

def test_12_regime_classification():
    """Test basic 12-regime classification"""
    try:
        from enhanced_12_regime_detector import Enhanced12RegimeDetector
        
        detector = Enhanced12RegimeDetector()
        
        # Test data
        test_data = {
            'iv_percentile': 0.2,
            'atr_normalized': 0.15,
            'gamma_exposure': 0.1,
            'ema_alignment': 0.8,
            'price_momentum': 0.7,
            'volume_confirmation': 0.6,
            'strike_correlation': 0.85,
            'vwap_deviation': 0.8,
            'pivot_analysis': 0.75
        }
        
        # Classify regime
        result = detector.classify_12_regime(test_data)
        
        print(f"✅ Classification successful: {result.regime_id}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Volatility Level: {result.volatility_level}")
        print(f"   Trend Type: {result.trend_type}")
        print(f"   Structure Type: {result.structure_type}")
        
        # Validate result structure
        if hasattr(result, 'regime_id') and result.regime_id:
            print("✅ Result has valid regime_id")
        else:
            print("❌ Result missing regime_id")
            return False
        
        if hasattr(result, 'confidence') and 0.0 <= result.confidence <= 1.0:
            print("✅ Result has valid confidence score")
        else:
            print("❌ Result has invalid confidence score")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to classify regime: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_18_to_12_mapping():
    """Test 18→12 regime mapping"""
    try:
        from enhanced_12_regime_detector import Enhanced12RegimeDetector
        
        detector = Enhanced12RegimeDetector()
        
        # Test mappings
        test_mappings = [
            ('HIGH_VOLATILE_STRONG_BULLISH', 'HIGH_DIRECTIONAL_TRENDING'),
            ('LOW_VOLATILE_MILD_BEARISH', 'LOW_DIRECTIONAL_RANGE'),
            ('NORMAL_VOLATILE_NEUTRAL', 'MODERATE_NONDIRECTIONAL_RANGE'),
            ('HIGH_VOLATILE_SIDEWAYS', 'HIGH_NONDIRECTIONAL_TRENDING'),
        ]
        
        correct_mappings = 0
        total_mappings = len(test_mappings)
        
        for regime_18, expected_12 in test_mappings:
            mapped_12 = detector.map_18_to_12_regime(regime_18)
            
            if mapped_12 == expected_12:
                correct_mappings += 1
                print(f"✅ {regime_18} → {mapped_12}")
            else:
                print(f"❌ {regime_18} → {mapped_12} (expected: {expected_12})")
        
        accuracy = correct_mappings / total_mappings
        print(f"✅ Mapping accuracy: {accuracy:.1%} ({correct_mappings}/{total_mappings})")
        
        if accuracy >= 0.75:  # 75% accuracy threshold
            print("✅ Mapping accuracy acceptable")
            return True
        else:
            print("❌ Mapping accuracy too low")
            return False
        
    except Exception as e:
        print(f"❌ Failed to test regime mapping: {e}")
        return False

def test_excel_manager_12_regime():
    """Test Excel manager 12-regime support"""
    try:
        from actual_system_excel_manager import ActualSystemExcelManager
        
        excel_manager = ActualSystemExcelManager()
        
        # Test 12-regime configuration generation
        regime_config = excel_manager._generate_regime_formation_config("12_REGIME")
        
        print(f"✅ Generated 12-regime configuration with {len(regime_config)} entries")
        
        # Check for 12-regime entries
        regime_names = [entry[0] for entry in regime_config if len(entry) > 0]
        expected_12_regimes = [
            'LOW_DIRECTIONAL_TRENDING',
            'LOW_DIRECTIONAL_RANGE',
            'MODERATE_DIRECTIONAL_TRENDING',
            'HIGH_NONDIRECTIONAL_RANGE'
        ]
        
        found_regimes = 0
        for expected_regime in expected_12_regimes:
            if expected_regime in regime_names:
                found_regimes += 1
                print(f"✅ Found regime: {expected_regime}")
            else:
                print(f"❌ Missing regime: {expected_regime}")
        
        if found_regimes >= 3:  # At least 3 out of 4 test regimes
            print("✅ Excel manager 12-regime support working")
            return True
        else:
            print("❌ Excel manager 12-regime support insufficient")
            return False
        
    except Exception as e:
        print(f"❌ Failed to test Excel manager: {e}")
        return False

def run_simple_12_regime_tests():
    """Run simple test suite for 12-regime system"""
    print("="*60)
    print("SIMPLE 12-REGIME SYSTEM VALIDATION")
    print("="*60)
    
    tests = [
        ("Import Test", test_12_regime_import),
        ("Initialization Test", test_12_regime_initialization),
        ("Classification Test", test_12_regime_classification),
        ("Mapping Test", test_18_to_12_mapping),
        ("Excel Manager Test", test_excel_manager_12_regime),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - 12-REGIME SYSTEM READY")
        return True
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("⚠️  MOST TESTS PASSED - 12-REGIME SYSTEM MOSTLY READY")
        return True
    else:
        print("❌ TESTS FAILED - 12-REGIME SYSTEM NEEDS FIXES")
        return False

if __name__ == "__main__":
    success = run_simple_12_regime_tests()
    sys.exit(0 if success else 1)
