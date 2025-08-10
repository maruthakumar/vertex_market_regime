#!/usr/bin/env python3
"""Test fixed query"""

import sys
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')

from backtester_v2.dal.heavydb_connection import get_connection, execute_query

# Test the fixed query
conn = get_connection()
if conn:
    # First get max date
    max_date_query = "SELECT MAX(trade_date) FROM nifty_option_chain"
    result = execute_query(conn, max_date_query)
    if not result.empty:
        max_date = result.iloc[0][0]
        print(f"✅ Max date: {max_date}")
        
        # Now get data for that date
        query = f"""
        SELECT * FROM nifty_option_chain 
        WHERE trade_date = '{max_date}'
        AND atm_strike IS NOT NULL
        LIMIT 100
        """
        
        data = execute_query(conn, query)
        print(f"✅ Retrieved {len(data)} records")
        
        if not data.empty:
            print(f"✅ Columns: {list(data.columns)[:10]}...")
            print(f"✅ Sample data:")
            print(f"   Trade date: {data.iloc[0]['trade_date']}")
            print(f"   Trade time: {data.iloc[0]['trade_time']}")
            print(f"   Spot: {data.iloc[0]['spot']}")
            print(f"   ATM Strike: {data.iloc[0]['atm_strike']}")