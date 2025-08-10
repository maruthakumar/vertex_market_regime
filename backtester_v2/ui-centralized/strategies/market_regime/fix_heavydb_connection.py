#!/usr/bin/env python3
"""
Fix HeavyDB Connection in API Server
This script diagnoses and fixes the HeavyDB connection issue
"""

import sys
import os
import subprocess

print("=" * 80)
print("FIXING HEAVYDB CONNECTION IN API SERVER")
print("=" * 80)

# Step 1: Check Python environment
print("\n1️⃣ Checking Python environment...")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Step 2: Check if heavydb is installed in server environment
print("\n2️⃣ Checking HeavyDB module installation...")
try:
    import heavydb
    print("✅ heavydb module is installed")
    print(f"   Location: {heavydb.__file__}")
except ImportError:
    print("❌ heavydb module NOT installed")
    print("   Installing heavydb (pymapd)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pymapd"])

# Step 3: Check the market regime API file
print("\n3️⃣ Checking market_regime_api.py imports...")
api_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/market_regime_api.py"

# Read the file
with open(api_file, 'r') as f:
    content = f.read()

# Check for mock data fallbacks
if "mock" in content.lower() or "simulation" in content.lower():
    print("⚠️  WARNING: Found references to mock/simulation data")
    
# Step 4: Create a patch for the API to ensure real data only
print("\n4️⃣ Creating patch to enforce real data only...")

patch_content = '''#!/usr/bin/env python3
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
'''

patch_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/heavydb_connection_patch.py"
with open(patch_file, 'w') as f:
    f.write(patch_content)
print(f"✅ Created patch file: {patch_file}")

# Step 5: Check server startup script
print("\n5️⃣ Checking server startup configuration...")
server_files = [
    "/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/main.py",
    "/srv/samba/shared/bt/backtester_stable/BTRUN/server/run_server.py",
    "/srv/samba/shared/bt/backtester_stable/BTRUN/server/start_server.py"
]

for server_file in server_files:
    if os.path.exists(server_file):
        print(f"   Found server file: {server_file}")

# Step 6: Create environment check script
print("\n6️⃣ Creating server environment check...")

env_check = '''#!/usr/bin/env python3
"""Check server environment for HeavyDB"""
import sys
print(f"Server Python: {sys.executable}")
print(f"Python Path: {sys.path[:3]}")

try:
    import heavydb
    print("✅ heavydb available in server environment")
except:
    print("❌ heavydb NOT available in server environment")
    
try:
    sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
    from backtester_v2.dal.heavydb_connection import get_connection
    print("✅ Can import HeavyDB connection")
except Exception as e:
    print(f"❌ Cannot import HeavyDB connection: {e}")
'''

env_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/server/check_server_env.py"
with open(env_file, 'w') as f:
    f.write(env_check)
print(f"✅ Created environment check: {env_file}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("""
1. Run the patch file before starting the server:
   python3 heavydb_connection_patch.py

2. Check server environment:
   cd /srv/samba/shared/bt/backtester_stable/BTRUN/server
   python3 check_server_env.py

3. Ensure server startup includes proper paths:
   export PYTHONPATH=/srv/samba/shared/bt/backtester_stable/BTRUN:$PYTHONPATH

4. Remove ANY mock data fallbacks from strategies.market_regime_api.py

5. Restart the server after fixes
""")