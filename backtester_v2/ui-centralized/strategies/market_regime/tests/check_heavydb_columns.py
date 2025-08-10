#!/usr/bin/env python3
"""Quick check of HeavyDB table columns"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from dal.heavydb_connection import get_connection, execute_query
except ImportError:
    from backtester_v2.dal.heavydb_connection import get_connection, execute_query

# Get connection
conn = get_connection()

# Check table structure
query = """
SELECT * 
FROM nifty_option_chain 
LIMIT 1
"""

df = execute_query(conn, query)
print("Available columns:")
for col in df.columns:
    print(f"  - {col}")

# Check expiry date format
query2 = """
SELECT DISTINCT expiry_date
FROM nifty_option_chain
WHERE trade_date = '2024-01-01'
ORDER BY expiry_date
LIMIT 5
"""

df2 = execute_query(conn, query2)
print("\nExpiry dates:")
for idx, row in df2.iterrows():
    print(f"  - {row['expiry_date']}")

# Check if we have data for the date range
query3 = """
SELECT COUNT(*) as row_count, MIN(trade_date) as min_date, MAX(trade_date) as max_date
FROM nifty_option_chain
WHERE trade_date BETWEEN '2024-01-01' AND '2024-01-31'
"""

df3 = execute_query(conn, query3)
print("\nData range check:")
print(f"  - Count: {df3['row_count'].iloc[0]}")
print(f"  - Min date: {df3['min_date'].iloc[0]}")
print(f"  - Max date: {df3['max_date'].iloc[0]}")

# Test the failing subquery
query4 = """
SELECT COUNT(*) as row_count
FROM nifty_option_chain
WHERE trade_date = '2024-01-02'
    AND trade_time >= '09:15:00' AND trade_time <= '15:30:00'
    AND expiry_date = '2024-01-04'
"""

df4 = execute_query(conn, query4)
print(f"\nRows for 2024-01-02 with expiry 2024-01-04: {df4['row_count'].iloc[0]}")