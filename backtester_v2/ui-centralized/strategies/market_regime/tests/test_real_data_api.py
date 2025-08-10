#!/usr/bin/env python3
import requests
import json

# Test the API
try:
    response = requests.get("http://localhost:8000/api/v1/market-regime/status")
    data = response.json()
    
    print("=" * 60)
    print("MARKET REGIME API STATUS CHECK")
    print("=" * 60)
    
    regime = data.get('current_regime', {})
    health = data.get('system_health', {})
    
    print(f"\nCurrent Regime: {regime.get('regime')}")
    print(f"Data Source: {regime.get('data_source')}")
    print(f"Confidence: {regime.get('confidence')}")
    
    print(f"\nSystem Health:")
    print(f"  HeavyDB Connected: {health.get('heavydb_connected')}")
    print(f"  Database Status: {health.get('database_status')}")
    print(f"  Record Count: {health.get('record_count', 'N/A')}")
    print(f"  System Status: {health.get('system_status')}")
    
    if regime.get('data_source') == 'real_heavydb':
        print("\n✅ SUCCESS: Using REAL HeavyDB data!")
    else:
        print(f"\n❌ ERROR: Not using real data - source is {regime.get('data_source')}")
        
except Exception as e:
    print(f"❌ Error testing API: {e}")
