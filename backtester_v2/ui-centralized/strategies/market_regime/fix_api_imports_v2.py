#!/usr/bin/env python3
"""
Fix market_regime_api.py imports and ensure real data usage
"""

import re

print("=" * 80)
print("FIXING MARKET REGIME API - VERSION 2")
print("=" * 80)

api_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/market_regime_api.py"

# Read the file
with open(api_file, 'r') as f:
    content = f.read()

# Fix 1: Update Excel converter import
print("\n1️⃣ Fixing Excel to YAML converter import...")
content = content.replace(
    "from excel_to_yaml_converter import ExcelToYamlConverter",
    "from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter as ExcelToYamlConverter"
)
print("   ✅ Fixed Excel converter import")

# Fix 2: Update adapter import
print("\n2️⃣ Fixing adapter import...")
content = content.replace(
    "from real_data_engine_adapter import RealDataEngineAdapter",
    "from backtester_v2.strategies.market_regime.real_data_engine_adapter import RealDataEngineAdapter"
)
print("   ✅ Fixed adapter import")

# Fix 3: Fix database check query
print("\n3️⃣ Fixing database check query...")
content = content.replace(
    'test_query = "SELECT COUNT(*) as count FROM nifty_option_chain LIMIT 1"',
    'test_query = "SELECT COUNT(*) FROM nifty_option_chain LIMIT 1"'
)
print("   ✅ Fixed database check query")

# Fix 4: Fix calculate_current_regime to check adapter
print("\n4️⃣ Fixing calculate_current_regime function...")

# Find the function and add check for engine_adapter
pattern = r'(async def calculate_current_regime.*?:\s*""".*?""")'
match = re.search(pattern, content, re.DOTALL)

if match:
    func_start = match.end()
    # Insert adapter check
    insert_code = """
    try:
        global real_data_engine, market_regime_analyzer, yaml_config_cache, engine_adapter
        
        # Ensure we have the adapter
        if not engine_adapter:
            logger.error("Engine adapter not initialized")
            raise RuntimeError("Engine adapter not available")
"""
    
    # Find where to insert (after the try:)
    try_pos = content.find("\n    try:", func_start)
    if try_pos > 0:
        # Replace just the "try:" line with our enhanced version
        end_of_line = content.find("\n", try_pos + 1)
        content = content[:try_pos] + insert_code + content[end_of_line:]
        print("   ✅ Added engine_adapter check")

# Fix 5: Fix the conditional check for real engines
print("\n5️⃣ Fixing conditional check for real engines...")
content = content.replace(
    "if REAL_ENGINES_INITIALIZED and real_data_engine and market_regime_analyzer:",
    "if REAL_ENGINES_INITIALIZED and engine_adapter:"
)
print("   ✅ Fixed conditional check")

# Fix 6: Fix HeavyDB query - remove timestamp column reference
print("\n6️⃣ Fixing HeavyDB queries...")
# The nifty_option_chain table doesn't have a timestamp column
old_query = """SELECT * FROM nifty_option_chain 
                    WHERE timestamp >= (
                        SELECT MAX(timestamp) - INTERVAL '1' HOUR 
                        FROM nifty_option_chain
                    )
                    ORDER BY timestamp DESC 
                    LIMIT 1000"""

new_query = """SELECT * FROM nifty_option_chain 
                    ORDER BY trade_date DESC, trade_time DESC 
                    LIMIT 1000"""

content = content.replace(old_query, new_query)
print("   ✅ Fixed HeavyDB query for current regime")

# Also fix historical query
old_hist_query = """SELECT timestamp, spot_price, ce_iv, pe_iv, 
                       ce_delta, pe_delta, ce_gamma, pe_gamma, 
                       ce_theta, pe_theta, ce_vega, pe_vega,
                       strike_price, dte, ce_ltp, pe_ltp
                FROM nifty_option_chain 
                WHERE timestamp >= '{start_dt.strftime('%Y-%m-%d %H:%M:%S')}'
                  AND timestamp <= '{end_dt.strftime('%Y-%m-%d %H:%M:%S')}'
                ORDER BY timestamp ASC"""

new_hist_query = """SELECT trade_date, trade_time, spot, ce_iv, pe_iv, 
                       ce_delta, pe_delta, ce_gamma, pe_gamma, 
                       ce_theta, pe_theta, ce_vega, pe_vega,
                       strike, dte, ce_close, pe_close
                FROM nifty_option_chain 
                WHERE trade_date >= '{start_dt.strftime('%Y-%m-%d')}'
                  AND trade_date <= '{end_dt.strftime('%Y-%m-%d')}'
                ORDER BY trade_date ASC, trade_time ASC"""

content = content.replace("spot_price", "spot")
content = content.replace("strike_price", "strike")
content = content.replace("ce_ltp", "ce_close")
content = content.replace("pe_ltp", "pe_close")

print("   ✅ Fixed column names in queries")

# Write updated file
with open(api_file, 'w') as f:
    f.write(content)
    
print("\n✅ All fixes applied to market_regime_api.py")
print("\nNext steps:")
print("1. Restart the API server")
print("2. Test the /api/market-regime/status endpoint")
print("3. Verify data_source shows 'real_heavydb'")