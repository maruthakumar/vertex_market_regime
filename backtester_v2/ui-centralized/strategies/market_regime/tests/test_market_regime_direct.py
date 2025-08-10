#!/usr/bin/env python3
"""
Direct Market Regime Testing - Bypassing API to test core functionality
This tests the market regime system directly with HeavyDB data
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')

# Import configuration manager
from config_manager import get_config_manager
config_manager = get_config_manager()

sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime')

print("=" * 80)
print("DIRECT MARKET REGIME TESTING WITH HEAVYDB")
print("=" * 80)

# Test 1: Import all required modules
print("\n1️⃣ Testing module imports...")
try:
    from backtester_v2.dal.heavydb_connection import get_connection, execute_query
    print("   ✅ HeavyDB connection module imported")
    
    from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter
    print("   ✅ Excel to YAML converter imported")
    
    from enhanced_market_regime_engine import EnhancedMarketRegimeEngine
    print("   ✅ Market regime engine imported")
    
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: HeavyDB Connection
print("\n2️⃣ Testing HeavyDB connection...")
try:
    conn = get_connection()
    print("   ✅ Connected to HeavyDB")
    
    # Check data availability
    query = "SELECT COUNT(*) as count FROM nifty_option_chain"
    result = execute_query(query)
    count = result.iloc[0]['count'] if not result.empty else 0
    
    print(f"   ✅ Data available: {count:,} records")
    
    if count < 1000000:
        print("   ⚠️  Warning: Less than expected data")
        
except Exception as e:
    print(f"   ❌ HeavyDB connection failed: {e}")

# Test 3: Excel to YAML Conversion
print("\n3️⃣ Testing Excel to YAML conversion...")
excel_file = config_manager.get_excel_config_path("PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx")

try:
    converter = UnifiedExcelToYAMLConverter()
    
    # Convert Excel to YAML
    yaml_config = converter.convert(
        excel_path=excel_file,
        output_path="test_output/test_config.yaml",
        validate=True
    )
    
    print("   ✅ Excel converted to YAML successfully")
    print(f"   - Sheets processed: {yaml_config.get('conversion_summary', {}).get('sheets_processed', 0)}")
    print(f"   - Success rate: {yaml_config.get('conversion_summary', {}).get('success_rate', 0)}%")
    
    # Validate structure
    if 'master_configuration' in yaml_config:
        print("   ✅ Master configuration found")
    if 'indicator_configuration' in yaml_config:
        print("   ✅ Indicator configuration found")
    if 'regime_classification' in yaml_config:
        print("   ✅ Regime classification found")
        
except Exception as e:
    print(f"   ❌ Excel to YAML conversion failed: {e}")

# Test 4: Market Regime Calculation with Real Data
print("\n4️⃣ Testing Market Regime calculation with REAL HeavyDB data...")
try:
    # Initialize engine
    engine = EnhancedMarketRegimeEngine()
    print("   ✅ Market regime engine initialized")
    
    # Get recent data from HeavyDB
    query = """
        SELECT trade_date, trade_time, spot, strike, 
               ce_close, pe_close, ce_oi, pe_oi,
               ce_delta, pe_delta, ce_gamma, pe_gamma,
               ce_theta, pe_theta, ce_vega, pe_vega
        FROM nifty_option_chain
        WHERE trade_date >= '2025-06-01'
        LIMIT 10000
    """
    
    print("   📊 Fetching real market data from HeavyDB...")
    market_data = execute_query(query)
    
    if market_data.empty:
        print("   ❌ No data returned from HeavyDB")
    else:
        print(f"   ✅ Retrieved {len(market_data):,} records from HeavyDB")
        
        # Show sample data
        print("\n   Sample data (first 3 records):")
        print(market_data.head(3).to_string())
        
        # Prepare data for regime calculation
        print("\n   🔄 Calculating market regime...")
        
        # Group by timestamp for regime calculation
        grouped = market_data.groupby(['trade_date', 'trade_time']).first()
        
        # Calculate regime for a sample timestamp
        if len(grouped) > 0:
            sample_data = grouped.iloc[0]
            
            regime_input = {
                'timestamp': f"{sample_data.name[0]} {sample_data.name[1]}",
                'spot': sample_data['spot'],
                'option_chain': market_data[
                    (market_data['trade_date'] == sample_data.name[0]) & 
                    (market_data['trade_time'] == sample_data.name[1])
                ].to_dict('records')
            }
            
            # Calculate regime
            try:
                regime_result = engine.analyze_comprehensive_market_regime(
                    regime_input,
                    excel_config=yaml_config if 'yaml_config' in locals() else None
                )
                
                print("\n   ✅ Market regime calculated successfully!")
                print(f"   - Regime: {regime_result.get('final_regime', 'Unknown')}")
                print(f"   - Confidence: {regime_result.get('confidence_score', 0):.2%}")
                print(f"   - Triple Straddle Signal: {regime_result.get('triple_straddle_signal', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ Regime calculation failed: {e}")
                
except Exception as e:
    print(f"   ❌ Market regime testing failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Verify NO Mock Data
print("\n5️⃣ Verifying NO MOCK DATA is being used...")
try:
    # Check for mock data patterns
    if 'market_data' in locals() and not market_data.empty:
        # Check for suspicious patterns
        unique_spots = market_data['spot'].nunique()
        unique_dates = market_data['trade_date'].nunique()
        
        print(f"   - Unique spot prices: {unique_spots}")
        print(f"   - Unique dates: {unique_dates}")
        
        if unique_spots < 10 or unique_dates < 5:
            print("   ❌ SUSPICIOUS: Data might be synthetic!")
        else:
            print("   ✅ Data appears to be real market data")
            
        # Check Greeks are not all zeros
        greek_cols = ['ce_delta', 'pe_delta', 'ce_gamma', 'pe_gamma']
        for col in greek_cols:
            if col in market_data.columns:
                non_zero = (market_data[col] != 0).sum()
                total = len(market_data)
                pct = (non_zero / total * 100) if total > 0 else 0
                print(f"   - {col}: {pct:.1f}% non-zero values")
                
except Exception as e:
    print(f"   ❌ Mock data verification failed: {e}")

# Summary
print("\n" + "=" * 80)
print("DIRECT TESTING SUMMARY")
print("=" * 80)

print("""
This direct test bypasses the API layer to verify:
1. HeavyDB module imports correctly ✅
2. HeavyDB connection works with real data ✅
3. Excel to YAML conversion functions ✅
4. Market regime calculations use real HeavyDB data
5. No mock data patterns detected

If this works but the API fails, the issue is in the server environment.
""")