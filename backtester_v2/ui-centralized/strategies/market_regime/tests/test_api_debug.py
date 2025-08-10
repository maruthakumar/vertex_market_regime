#!/usr/bin/env python3
"""
Debug API real data issue
"""

import sys
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes')

# Import the API module
from strategies.market_regime_api import (
    HEAVYDB_AVAILABLE, CONVERTER_AVAILABLE, REAL_ENGINE_AVAILABLE, 
    ADAPTER_AVAILABLE, REAL_ENGINES_INITIALIZED, 
    real_data_engine, market_regime_analyzer, engine_adapter,
    calculate_current_regime
)

print("=" * 80)
print("DEBUGGING MARKET REGIME API")
print("=" * 80)

print("\n1️⃣ Checking module imports:")
print(f"   HEAVYDB_AVAILABLE: {HEAVYDB_AVAILABLE}")
print(f"   CONVERTER_AVAILABLE: {CONVERTER_AVAILABLE}")
print(f"   REAL_ENGINE_AVAILABLE: {REAL_ENGINE_AVAILABLE}")
print(f"   ADAPTER_AVAILABLE: {ADAPTER_AVAILABLE}")
print(f"   REAL_ENGINES_INITIALIZED: {REAL_ENGINES_INITIALIZED}")

print("\n2️⃣ Checking engine instances:")
print(f"   real_data_engine: {real_data_engine}")
print(f"   market_regime_analyzer: {market_regime_analyzer}")
print(f"   engine_adapter: {engine_adapter}")

print("\n3️⃣ Testing calculate_current_regime function:")

async def test_regime():
    try:
        result = await calculate_current_regime()
        print(f"\n✅ Result:")
        print(f"   Regime: {result.get('regime')}")
        print(f"   Data Source: {result.get('data_source')}")
        print(f"   Confidence: {result.get('confidence')}")
        
        if result.get('error'):
            print(f"   ❌ Error: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()

# Run the test
asyncio.run(test_regime())

print("\n4️⃣ Testing direct HeavyDB query:")
if HEAVYDB_AVAILABLE:
    try:
        from backtester_v2.dal.heavydb_connection import get_connection, execute_query
        conn = get_connection()
        if conn:
            # Test simple query
            result = execute_query(conn, "SELECT COUNT(*) FROM nifty_option_chain")
            count = result.iloc[0][0] if not result.empty else 0
            print(f"   ✅ HeavyDB has {count:,} records")
            
            # Test latest data query
            query = """
            SELECT * FROM nifty_option_chain 
            ORDER BY trade_date DESC, trade_time DESC 
            LIMIT 1
            """
            latest = execute_query(conn, query)
            if not latest.empty:
                print(f"   ✅ Latest date: {latest.iloc[0]['trade_date']}")
                print(f"   ✅ Latest time: {latest.iloc[0]['trade_time']}")
        else:
            print("   ❌ No connection")
    except Exception as e:
        print(f"   ❌ Query error: {e}")
else:
    print("   ❌ HeavyDB not available")