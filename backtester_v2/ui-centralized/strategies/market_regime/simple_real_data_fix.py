#!/usr/bin/env python3
"""
Simple fix to ensure real data usage in market_regime_api.py
"""

import re

api_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/market_regime_api.py"

# Read the file
with open(api_file, 'r') as f:
    content = f.read()

print("=" * 80)
print("APPLYING SIMPLE REAL DATA FIX")
print("=" * 80)

# Fix 1: Remove the fallback to simulation in calculate_current_regime
print("\n1️⃣ Fixing calculate_current_regime to return real data only...")

# Find the fallback section and replace it
old_fallback = '''        # Fallback to basic calculation if real engines not available
        logger.warning("⚠️ Using fallback regime calculation (real engines not available)")
        regimes = ['STRONG_BULLISH', 'MILD_BULLISH', 'NEUTRAL', 'MILD_BEARISH', 'STRONG_BEARISH', 'HIGH_VOLATILITY']
        
        return {
            "regime": np.random.choice(regimes),
            "confidence": round(np.random.uniform(0.7, 0.95), 3),
            "sub_regimes": {
                "volatility": np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                "trend": np.random.choice(['BULLISH', 'BEARISH', 'SIDEWAYS']),
                "structure": np.random.choice(['STRONG', 'WEAK', 'NEUTRAL'])
            },
            "indicators": {
                "vix_proxy": round(np.random.uniform(12, 25), 2),
                "trend_strength": round(np.random.uniform(-1, 1), 3),
                "volatility_percentile": round(np.random.uniform(0, 100), 1)
            },
            "data_source": "fallback_simulation",
            "calculation_timestamp": datetime.now().isoformat()
        }'''

new_fallback = '''        # NO FALLBACK - Return error if real data not available
        logger.error("❌ Real engines/data not available - cannot use mock data")
        return {
            "regime": "ERROR_NO_DATA",
            "confidence": 0.0,
            "sub_regimes": {
                "volatility": "UNKNOWN",
                "trend": "UNKNOWN", 
                "structure": "UNKNOWN"
            },
            "indicators": {
                "vix_proxy": 0.0,
                "trend_strength": 0.0,
                "volatility_percentile": 0.0
            },
            "data_source": "error_no_real_data",
            "error": "Real data not available - mock data not allowed",
            "calculation_timestamp": datetime.now().isoformat()
        }'''

content = content.replace(old_fallback, new_fallback)
print("   ✅ Replaced fallback with error response")

# Fix 2: Update data_source in health status
print("\n2️⃣ Fixing health status data source...")
content = content.replace(
    '"data_source": "real_heavydb" if REAL_ENGINES_INITIALIZED and HEAVYDB_AVAILABLE else "simulation"',
    '"data_source": "real_heavydb" if current_regime.get("data_source") == "real_heavydb" else "error_no_real_data"'
)
print("   ✅ Updated health status data source")

# Write the updated content
with open(api_file, 'w') as f:
    f.write(content)

print("\n✅ Simple fix applied successfully")
print("\nNext steps:")
print("1. Restart the API server")
print("2. Test /api/market-regime/status")
print("3. Verify data_source is 'real_heavydb' or 'error_no_real_data' (never 'fallback_simulation')")