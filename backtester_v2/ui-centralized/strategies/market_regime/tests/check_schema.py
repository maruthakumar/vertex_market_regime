#!/usr/bin/env python3

import sys
sys.path.insert(0, '/srv/samba/shared/bt/backtester_stable/BTRUN')

from backtester_v2.dal.heavydb_connection import get_connection, execute_query

try:
    conn = get_connection()
    if conn:
        print("✅ Connected to HeavyDB")
        
        # Get actual table schema
        try:
            result = execute_query(conn, 'SHOW TABLES')
            print('\nAvailable tables:')
            print(result)
        except Exception as e:
            print(f"Error getting tables: {e}")
        
        # Get column names
        try:
            result = execute_query(conn, 'DESCRIBE nifty_option_chain')
            print('\nTable schema:')
            print(result.head(10))
        except Exception as e:
            print(f"Error describing table: {e}")
            
        # Get sample data to see column names
        try:
            result = execute_query(conn, 'SELECT * FROM nifty_option_chain LIMIT 1')
            print('\nColumn names:')
            print(result.columns.tolist())
        except Exception as e:
            print(f"Error getting sample data: {e}")
        
except Exception as e:
    print(f'Connection error: {e}')