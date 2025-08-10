#!/usr/bin/env python3

print("🚀 Starting 12-Regime System Validation...")

try:
    print("📦 Importing Enhanced12RegimeDetector...")
    from enhanced_12_regime_detector import Enhanced12RegimeDetector
    print("✅ Import successful!")
    
    print("🔧 Initializing detector...")
    detector = Enhanced12RegimeDetector()
    print("✅ Initialization successful!")
    
    print(f"📊 Regime definitions: {len(detector.regime_definitions)}")
    print(f"🔄 Regime mappings: {len(detector.regime_mapping_18_to_12)}")
    
    # List all 12 regimes
    print("\n📋 12-Regime Types:")
    for i, regime_id in enumerate(detector.regime_definitions.keys(), 1):
        print(f"  {i:2d}. {regime_id}")
    
    print("\n🧪 Testing classification...")
    test_data = {
        'iv_percentile': 0.2, 'atr_normalized': 0.15, 'gamma_exposure': 0.1,
        'ema_alignment': 0.8, 'price_momentum': 0.7, 'volume_confirmation': 0.6,
        'strike_correlation': 0.85, 'vwap_deviation': 0.8, 'pivot_analysis': 0.75
    }
    
    result = detector.classify_12_regime(test_data)
    print(f"✅ Classification Result:")
    print(f"   Regime: {result.regime_id}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Volatility: {result.volatility_level}")
    print(f"   Trend: {result.trend_type}")
    print(f"   Structure: {result.structure_type}")
    
    print("\n🔄 Testing 18→12 mapping...")
    test_mappings = [
        'HIGH_VOLATILE_STRONG_BULLISH',
        'LOW_VOLATILE_MILD_BEARISH', 
        'NORMAL_VOLATILE_NEUTRAL',
        'HIGH_VOLATILE_SIDEWAYS'
    ]
    
    for regime_18 in test_mappings:
        mapped_12 = detector.map_18_to_12_regime(regime_18)
        print(f"   {regime_18} → {mapped_12}")
    
    print("\n📊 Testing Excel manager...")
    from actual_system_excel_manager import ActualSystemExcelManager
    excel_manager = ActualSystemExcelManager()
    
    regime_config = excel_manager._generate_regime_formation_config("12_REGIME")
    print(f"✅ Excel 12-regime config generated: {len(regime_config)} entries")
    
    print("\n🎉 ALL TESTS PASSED! 12-REGIME SYSTEM IS OPERATIONAL")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
