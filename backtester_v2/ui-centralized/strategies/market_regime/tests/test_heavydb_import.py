#!/usr/bin/env python3
"""Quick test to check HeavyDB module availability"""

import sys
import os

# Test 1: Check if heavydb module is available
print("Testing HeavyDB module availability...")
try:
    import heavydb
    print("✅ heavydb module imported successfully")
except ImportError as e:
    print(f"❌ heavydb module not available: {e}")
    print("   Trying pymapd...")
    try:
        import pymapd
        print("✅ pymapd module available (alternative)")
    except ImportError as e2:
        print(f"❌ pymapd also not available: {e2}")

# Test 2: Check DAL import
print("\nTesting DAL import...")
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
try:
    from backtester_v2.dal.heavydb_connection import get_connection
    print("✅ DAL heavydb_connection imported successfully")
except ImportError as e:
    print(f"❌ DAL import failed: {e}")

# Test 3: Check current path
print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Python path includes:")
for p in sys.path[:5]:
    print(f"  - {p}")