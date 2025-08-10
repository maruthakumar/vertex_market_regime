#!/usr/bin/env python3
"""
Patch for market_regime_api.py to ensure ONLY real HeavyDB data is used
"""

import sys
import os

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime')

# Force import of HeavyDB connection
try:
    from backtester_v2.dal.heavydb_connection import get_connection, execute_query
    HEAVYDB_AVAILABLE = True
    print("✅ HeavyDB connection module loaded successfully")
except ImportError as e:
    print(f"❌ CRITICAL: Cannot import HeavyDB connection: {e}")
    print("   Server MUST use real data - no fallbacks allowed")
    raise SystemExit("Cannot start without HeavyDB connection")

# Test connection
try:
    conn = get_connection()
    print("✅ HeavyDB connection established")
    
    # Verify data
    result = execute_query(conn, "SELECT COUNT(*) as count FROM nifty_option_chain")
    count = result.iloc[0]['count'] if not result.empty else 0
    print(f"✅ HeavyDB has {count:,} records")
    
    if count < 1000000:
        raise ValueError("Insufficient data in HeavyDB")
        
except Exception as e:
    print(f"❌ HeavyDB connection test failed: {e}")
    raise SystemExit("Cannot start without valid HeavyDB connection")

print("✅ HeavyDB validation complete - using REAL DATA ONLY")
