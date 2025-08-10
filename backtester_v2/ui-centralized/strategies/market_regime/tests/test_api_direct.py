#!/usr/bin/env python3
"""Direct test of market regime calculation"""

import sys
import asyncio
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/server')
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes')

print("=" * 80)
print("TESTING MARKET REGIME CALCULATION DIRECTLY")
print("=" * 80)

# Import after setting up logging
from app.api.routes.market_regime_api import (
    HEAVYDB_AVAILABLE, REAL_ENGINES_INITIALIZED, engine_adapter,
    calculate_current_regime, yaml_config_cache
)

print(f"\n1️⃣ System Status:")
print(f"   HEAVYDB_AVAILABLE: {HEAVYDB_AVAILABLE}")
print(f"   REAL_ENGINES_INITIALIZED: {REAL_ENGINES_INITIALIZED}")
print(f"   engine_adapter exists: {engine_adapter is not None}")
print(f"   yaml_config_cache exists: {bool(yaml_config_cache)}")

async def test_calculation():
    print("\n2️⃣ Running calculate_current_regime()...")
    try:
        result = await calculate_current_regime()
        
        print(f"\n3️⃣ Result:")
        print(f"   Regime: {result.get('regime')}")
        print(f"   Data Source: {result.get('data_source')}")
        print(f"   Confidence: {result.get('confidence')}")
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            
        if result.get('data_source') == 'fallback_simulation':
            print("\n⚠️  STILL USING FALLBACK SIMULATION!")
            
        elif result.get('data_source') == 'real_heavydb':
            print("\n✅ USING REAL HEAVYDB DATA!")
            print(f"   Data points used: {result.get('data_points_used', 'unknown')}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()

# Run the test
print("\nStarting async test...")
asyncio.run(test_calculation())