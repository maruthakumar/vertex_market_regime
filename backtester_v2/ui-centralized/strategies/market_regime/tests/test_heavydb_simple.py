#!/usr/bin/env python3
"""
Simple HeavyDB connection and query test
"""

try:
    from heavydb import connect
except ImportError:
    print("Installing pymapd...")
    import os
    os.system("pip install pymapd")
    from heavydb import connect

# Connect to HeavyDB
try:
    conn = connect(
        host='localhost',
        port=6274,
        user='admin',
        password='HyperInteractive',
        dbname='heavyai'
    )
    print("✅ Connected to HeavyDB")
    
    cursor = conn.cursor()
    
    # Test 1: Simple count
    print("\n1. Testing simple count:")
    cursor.execute("SELECT COUNT(*) FROM nifty_option_chain")
    count = cursor.fetchone()[0]
    print(f"   Total records: {count:,}")
    
    # Test 2: Get column names
    print("\n2. Getting table structure:")
    cursor.execute("SELECT * FROM nifty_option_chain LIMIT 1")
    columns = [desc[0] for desc in cursor.description]
    print(f"   Columns: {columns}")
    
    # Test 3: Get sample data
    print("\n3. Sample data:")
    cursor.execute("SELECT * FROM nifty_option_chain LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(f"   {row[:5]}...")  # Show first 5 columns
    
    # Test 4: Check timestamp column type
    print("\n4. Checking timestamp data:")
    try:
        cursor.execute("SELECT timestamp FROM nifty_option_chain LIMIT 5")
        timestamps = cursor.fetchall()
        for ts in timestamps:
            print(f"   Timestamp: {ts[0]} (type: {type(ts[0])})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Try different date filtering approaches
    print("\n5. Testing date filtering:")
    
    # Approach 1: Direct comparison
    try:
        cursor.execute("SELECT COUNT(*) FROM nifty_option_chain WHERE timestamp > '2025-01-01'")
        count = cursor.fetchone()[0]
        print(f"   Records after 2025-01-01: {count}")
    except Exception as e:
        print(f"   Direct comparison failed: {e}")
    
    # Approach 2: Without date functions
    try:
        cursor.execute("SELECT COUNT(*) FROM nifty_option_chain")
        total = cursor.fetchone()[0]
        print(f"   Total records (no filter): {total}")
    except Exception as e:
        print(f"   Basic count failed: {e}")
    
    # Test 6: Check for Greeks columns
    print("\n6. Checking Greeks columns:")
    try:
        cursor.execute("SELECT delta, gamma, theta, vega FROM nifty_option_chain LIMIT 5")
        greeks = cursor.fetchall()
        for g in greeks:
            print(f"   Delta: {g[0]}, Gamma: {g[1]}, Theta: {g[2]}, Vega: {g[3]}")
    except Exception as e:
        print(f"   Greeks query failed: {e}")
        # Try individual columns
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            try:
                cursor.execute(f"SELECT {greek} FROM nifty_option_chain LIMIT 1")
                val = cursor.fetchone()[0]
                print(f"   {greek}: {val}")
            except Exception as e2:
                print(f"   {greek} not found: {e2}")
    
    conn.close()
    print("\n✅ All tests completed")
    
except Exception as e:
    print(f"❌ Connection error: {e}")
    import traceback
    traceback.print_exc()