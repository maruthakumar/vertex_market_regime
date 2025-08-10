#!/usr/bin/env python3
"""
Phase 4: Indicator Calculation Logic Testing
============================================

Tests the market regime indicators:
1. Greek Sentiment Analysis
2. Triple Rolling Straddle
3. Trending OI/PA Patterns
4. Ensemble regime classification
"""

import requests
import json
import sys
from datetime import datetime, timedelta

print("=" * 80)
print("PHASE 4: INDICATOR CALCULATION LOGIC TESTING")
print("=" * 80)

# Base URL for API
BASE_URL = "http://localhost:8000/api/v1/market-regime"

# Test cases
test_results = {
    "greek_sentiment": False,
    "triple_straddle": False,
    "trending_oi_pa": False,
    "ensemble_classification": False,
    "real_data_calculations": False,
    "multi_timeframe": False
}

# 1. Test Current Regime Calculation (Real-time)
print("\n📊 Testing Current Regime Calculation...")
try:
    response = requests.get(f"{BASE_URL}/status")
    if response.status_code == 200:
        data = response.json()
        current_regime = data.get('current_regime', {})
        
        # Check if indicators are present
        indicators = current_regime.get('indicators', {})
        print(f"   Current regime: {current_regime.get('regime')}")
        print(f"   Data source: {current_regime.get('data_source')}")
        print(f"   Confidence: {current_regime.get('confidence')}")
        
        # Validate indicators
        if current_regime.get('data_source') == 'real_heavydb':
            print(f"   ✅ Using real HeavyDB data")
            test_results["real_data_calculations"] = True
            
            # Check indicators
            if 'vix_proxy' in indicators:
                print(f"   VIX Proxy: {indicators['vix_proxy']}")
            if 'trend_strength' in indicators:
                print(f"   Trend Strength: {indicators['trend_strength']}")
            if 'volatility_percentile' in indicators:
                print(f"   Volatility Percentile: {indicators['volatility_percentile']}")
                
            # Check sub-regimes
            sub_regimes = current_regime.get('sub_regimes', {})
            if all(k in sub_regimes for k in ['volatility', 'trend', 'structure']):
                print(f"   ✅ Sub-regimes present: {sub_regimes}")
                test_results["ensemble_classification"] = True
        else:
            print(f"   ❌ Not using real data: {current_regime.get('data_source')}")
    else:
        print(f"   ❌ Status check failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# First upload configuration
print("\n⚙️  Uploading Configuration First...")
try:
    import os
    excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') and 'MARKET_REGIME' in f]
    
    if excel_files:
        excel_file = excel_files[0]
        print(f"   Using config file: {excel_file}")
        
        with open(excel_file, 'rb') as f:
            files = {'configFile': (excel_file, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
        if response.status_code == 200:
            print(f"   ✅ Configuration uploaded successfully")
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error uploading config: {e}")

# 2. Test Historical Regime Calculation
print("\n📈 Testing Historical Regime Calculation...")
try:
    # Calculate for last 7 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    payload = {
        "start_date": start_date,
        "end_date": end_date,
        "timeframe": "1min",
        "include_confidence": True
    }
    
    response = requests.post(f"{BASE_URL}/calculate", json=payload)
    if response.status_code == 200:
        data = response.json()
        regime_results = data.get('regime_results', {})
        
        if regime_results.get('data_source') == 'real_heavydb':
            print(f"   ✅ Historical calculation using real data")
            
            # Check time series
            time_series = regime_results.get('time_series', [])
            if time_series:
                print(f"   Time series points: {len(time_series)}")
                
                # Analyze first few points
                for i, point in enumerate(time_series[:3]):
                    print(f"\n   Point {i+1}:")
                    print(f"     Timestamp: {point.get('timestamp')}")
                    print(f"     Regime: {point.get('regime')}")
                    print(f"     Confidence: {point.get('confidence')}")
                    
                    # Check indicators
                    point_indicators = point.get('indicators', {})
                    if 'greek_sentiment' in point_indicators:
                        print(f"     Greek Sentiment: {point_indicators['greek_sentiment']}")
                        test_results["greek_sentiment"] = True
                    if 'straddle_signal' in point_indicators:
                        print(f"     Straddle Signal: {point_indicators['straddle_signal']}")
                        test_results["triple_straddle"] = True
                    if 'oi_pattern' in point_indicators:
                        print(f"     OI Pattern: {point_indicators['oi_pattern']}")
                        test_results["trending_oi_pa"] = True
                
                # Check summary
                summary = regime_results.get('summary', {})
                if summary:
                    print(f"\n   Summary:")
                    print(f"     Total points: {summary.get('total_points')}")
                    print(f"     Data points analyzed: {summary.get('data_points_analyzed')}")
                    print(f"     Regime distribution: {summary.get('regime_distribution')}")
                    
                    if summary.get('total_points', 0) > 0:
                        test_results["multi_timeframe"] = True
        else:
            print(f"   ❌ Not using real data: {regime_results.get('data_source')}")
    else:
        print(f"   ❌ Calculation failed: {response.status_code}")
        if response.text:
            print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Test Excel Configuration Impact
print("\n⚙️  Testing Configuration Impact on Indicators...")
try:
    # Check if configuration is loaded
    response = requests.get(f"{BASE_URL}/status")
    if response.status_code == 200:
        data = response.json()
        if data.get('system_health', {}).get('yaml_config_available'):
            print(f"   ✅ YAML configuration is active")
        else:
            print(f"   ⚠️  YAML configuration not active")
except Exception as e:
    print(f"   ❌ Error testing configuration: {e}")

# 4. Test CSV Export with Indicators
print("\n📄 Testing CSV Export with Indicators...")
try:
    payload = {
        "period": "1_day",
        "format": "detailed",
        "include_metadata": True
    }
    
    response = requests.post(f"{BASE_URL}/generate-csv", json=payload)
    if response.status_code == 200:
        csv_content = response.text
        
        # Check if CSV contains indicator columns
        lines = csv_content.split('\n')
        if lines:
            # Check metadata
            metadata_lines = [l for l in lines if l.startswith('#')]
            if metadata_lines:
                print(f"   ✅ CSV metadata present ({len(metadata_lines)} lines)")
                for line in metadata_lines[:3]:
                    print(f"     {line}")
            
            # Check headers
            header_line = next((l for l in lines if not l.startswith('#') and l.strip()), None)
            if header_line:
                headers = header_line.split(',')
                print(f"\n   CSV Headers ({len(headers)} columns):")
                
                # Look for indicator columns
                indicator_cols = [h for h in headers if any(ind in h.lower() for ind in 
                                ['vix', 'trend', 'volatility', 'greek', 'straddle', 'oi', 'confidence'])]
                if indicator_cols:
                    print(f"   ✅ Indicator columns found: {indicator_cols}")
                else:
                    print(f"   ⚠️  No indicator columns found")
    else:
        print(f"   ❌ CSV export failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Summary
print("\n" + "=" * 80)
print("PHASE 4 TEST SUMMARY")
print("=" * 80)

for test, passed in test_results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{test.replace('_', ' ').title()}: {status}")

all_passed = all(test_results.values())
critical_passed = test_results["real_data_calculations"] and test_results["ensemble_classification"]

print("\n" + "=" * 80)
if all_passed:
    print("✅ ALL INDICATOR TESTS PASSED")
elif critical_passed:
    print("⚠️  CRITICAL TESTS PASSED BUT SOME INDICATORS MISSING")
    print("\nMissing indicators might be due to:")
    print("- Indicators not fully implemented in adapter")
    print("- Need for more complex calculations")
    print("- Placeholder values being used")
else:
    print("❌ INDICATOR CALCULATION ISSUES DETECTED")
    print("\n🚨 The system is not calculating indicators from real data properly")