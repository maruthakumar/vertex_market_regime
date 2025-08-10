#!/usr/bin/env python3
"""
Fix import issues in market_regime_api.py
"""

import os
import re

print("=" * 80)
print("FIXING MARKET REGIME API IMPORTS")
print("=" * 80)

api_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/market_regime_api.py"

# Read the file
with open(api_file, 'r') as f:
    content = f.read()

# Backup original
backup_file = api_file + ".backup"
with open(backup_file, 'w') as f:
    f.write(content)
print(f"✅ Created backup: {backup_file}")

# Fix 1: Update Excel converter import
print("\n1️⃣ Fixing Excel to YAML converter import...")
old_import = "from excel_to_yaml_converter import ExcelToYamlConverter"
new_import = "from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter as ExcelToYamlConverter"

if old_import in content:
    content = content.replace(old_import, new_import)
    print("   ✅ Fixed Excel converter import")
else:
    print("   ⚠️  Excel converter import not found as expected")

# Fix 2: Update adapter import
print("\n2️⃣ Fixing adapter import...")
old_adapter = "from real_data_engine_adapter import RealDataEngineAdapter"
new_adapter = "from backtester_v2.strategies.market_regime.real_data_engine_adapter import RealDataEngineAdapter"

if old_adapter in content:
    content = content.replace(old_adapter, new_adapter)
    print("   ✅ Fixed adapter import")
else:
    # Try adding the full path
    import_section = content.find("# Import adapter for real engines")
    if import_section > 0:
        # Find the try block
        try_block = content.find("try:", import_section)
        if try_block > 0:
            # Replace the import line
            end_try = content.find("except ImportError:", try_block)
            if end_try > 0:
                new_try_block = """try:
    from backtester_v2.strategies.market_regime.real_data_engine_adapter import RealDataEngineAdapter
    ADAPTER_AVAILABLE = True"""
                
                # Replace the try block content
                old_block = content[try_block:end_try].strip()
                content = content.replace(old_block, new_try_block)
                print("   ✅ Fixed adapter import with full path")

# Fix 3: Remove fallback to simulation
print("\n3️⃣ Removing simulation fallbacks...")

# Replace simulation returns with errors
replacements = [
    ('"data_source": "real_heavydb" if REAL_ENGINES_INITIALIZED and HEAVYDB_AVAILABLE else "simulation"',
     '"data_source": "real_heavydb" if REAL_ENGINES_INITIALIZED and HEAVYDB_AVAILABLE else "error_no_heavydb"'),
    
    ('"system_status": "operational" if REAL_ENGINES_INITIALIZED else "fallback_mode"',
     '"system_status": "operational" if REAL_ENGINES_INITIALIZED else "error_no_engines"'),
    
    ('"data_source": "fallback_simulation"',
     '"data_source": "error_no_real_data"'),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"   ✅ Replaced: {old[:50]}...")

# Fix 4: Add explicit error when no real data
print("\n4️⃣ Adding explicit errors for no real data...")

# Find calculate function
calc_func = content.find("async def calculate_current_regime")
if calc_func > 0:
    # Add check at beginning
    func_start = content.find("{", calc_func)
    if func_start > 0:
        insert_pos = content.find("\n", func_start) + 1
        check_code = """    # CRITICAL: Ensure real data only
    if not HEAVYDB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="HeavyDB not available - cannot use mock data"
        )
    if not REAL_ENGINES_INITIALIZED:
        logger.error("Real engines not initialized - attempting to reinitialize")
        if not initialize_real_engines():
            raise HTTPException(
                status_code=503,
                detail="Real engines not available - cannot use mock data"
            )
"""
        # Don't add if already exists
        if "CRITICAL: Ensure real data only" not in content:
            content = content[:insert_pos] + check_code + content[insert_pos:]
            print("   ✅ Added real data enforcement")

# Write updated file
with open(api_file, 'w') as f:
    f.write(content)
    
print("\n✅ Fixed import issues in market_regime_api.py")
print("\nNext steps:")
print("1. Restart the API server")
print("2. Test the /api/market-regime/status endpoint")
print("3. Verify data_source shows 'real_heavydb'")