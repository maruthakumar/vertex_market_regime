#!/usr/bin/env python3
"""
Phase 5: Output Generation and CSV Validation
============================================

Tests CSV/YAML output generation from market regime calculations
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import io

print("=" * 80)
print("PHASE 5: OUTPUT GENERATION AND CSV VALIDATION")
print("=" * 80)

BASE_URL = "http://localhost:8000/api/v1/market-regime"

test_results = {
    "csv_generation": False,
    "csv_format": False,
    "csv_metadata": False,
    "real_data_in_csv": False,
    "yaml_export": False,
    "data_completeness": False
}

# First, let's bypass the configuration requirement by creating a mock config
print("\n1️⃣ Creating Mock Configuration...")
try:
    # Try to get current status first
    response = requests.get(f"{BASE_URL}/status")
    if response.status_code == 200:
        print("   ✅ API is accessible")
except Exception as e:
    print(f"   ❌ API error: {e}")

# Test CSV Generation without config (if possible)
print("\n2️⃣ Testing CSV Generation...")
try:
    # First check if we can get any CSV output
    # Try different approaches
    
    # Approach 1: Try direct CSV export
    print("   Attempting direct CSV export...")
    payload = {
        "period": "1_day",
        "format": "basic",
        "include_metadata": False
    }
    
    response = requests.post(f"{BASE_URL}/generate-csv", json=payload)
    if response.status_code == 200:
        csv_content = response.text
        print(f"   ✅ CSV generated successfully!")
        test_results["csv_generation"] = True
        
        # Analyze CSV content
        try:
            # Read CSV into pandas
            df = pd.read_csv(io.StringIO(csv_content))
            print(f"   CSV Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check for required columns
            required_cols = ['timestamp', 'regime', 'confidence']
            has_required = all(col in df.columns for col in required_cols)
            if has_required:
                print(f"   ✅ Required columns present")
                test_results["csv_format"] = True
            else:
                print(f"   ❌ Missing required columns")
                
            # Check data
            if not df.empty:
                print(f"\n   Sample data (first 3 rows):")
                print(df.head(3).to_string(index=False))
                
                # Check if it's real data or mock
                if 'data_source' in df.columns:
                    sources = df['data_source'].unique()
                    print(f"\n   Data sources in CSV: {sources}")
                    if 'real_heavydb' in sources:
                        test_results["real_data_in_csv"] = True
                        
                # Check data completeness
                if len(df) > 0:
                    test_results["data_completeness"] = True
                    
        except Exception as e:
            print(f"   ❌ Error parsing CSV: {e}")
            # Print raw content for debugging
            print(f"\n   Raw CSV content (first 500 chars):")
            print(csv_content[:500])
    else:
        print(f"   ❌ CSV generation failed: {response.status_code}")
        # Try to get error details
        try:
            error_data = response.json()
            print(f"   Error: {error_data}")
        except:
            print(f"   Response: {response.text[:200]}")
            
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test CSV with metadata
print("\n3️⃣ Testing CSV with Metadata...")
try:
    payload = {
        "period": "1_day",
        "format": "enhanced",
        "include_metadata": True
    }
    
    response = requests.post(f"{BASE_URL}/generate-csv", json=payload)
    if response.status_code == 200:
        csv_content = response.text
        
        # Check for metadata lines
        lines = csv_content.split('\n')
        metadata_lines = [line for line in lines if line.startswith('#')]
        
        if metadata_lines:
            print(f"   ✅ Metadata present ({len(metadata_lines)} lines)")
            test_results["csv_metadata"] = True
            
            print("   Metadata content:")
            for line in metadata_lines[:5]:
                print(f"     {line}")
        else:
            print(f"   ⚠️  No metadata found in CSV")
            
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test different periods
print("\n4️⃣ Testing Different Time Periods...")
periods = ["1_day", "7_days", "30_days"]
for period in periods:
    try:
        payload = {
            "period": period,
            "format": "basic",
            "include_metadata": False
        }
        
        response = requests.post(f"{BASE_URL}/generate-csv", json=payload)
        if response.status_code == 200:
            csv_content = response.text
            df = pd.read_csv(io.StringIO(csv_content))
            print(f"   {period}: {len(df)} rows generated ✅")
        else:
            print(f"   {period}: Failed ({response.status_code}) ❌")
            
    except Exception as e:
        print(f"   {period}: Error - {e}")

# Test YAML export
print("\n5️⃣ Testing YAML Configuration Export...")
try:
    response = requests.get(f"{BASE_URL}/download-yaml")
    if response.status_code == 200:
        # Check if it's a file response
        content_type = response.headers.get('content-type', '')
        if 'yaml' in content_type or 'application' in content_type:
            print(f"   ✅ YAML export successful")
            test_results["yaml_export"] = True
            
            # Show sample of YAML content
            yaml_content = response.text
            print(f"   YAML content preview (first 300 chars):")
            print(f"   {yaml_content[:300]}...")
        else:
            print(f"   ⚠️  Unexpected content type: {content_type}")
    else:
        print(f"   ❌ YAML export failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test getting current config
print("\n6️⃣ Testing Configuration Status...")
try:
    response = requests.get(f"{BASE_URL}/config")
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data.get('status')}")
        if data.get('status') == 'no_config':
            print(f"   ⚠️  No configuration loaded")
        else:
            config_summary = data.get('config_summary', {})
            print(f"   Config: {config_summary}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Summary
print("\n" + "=" * 80)
print("PHASE 5 TEST SUMMARY")
print("=" * 80)

for test, passed in test_results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{test.replace('_', ' ').title()}: {status}")

print("\n" + "=" * 80)
if all(test_results.values()):
    print("✅ ALL OUTPUT GENERATION TESTS PASSED")
elif test_results["csv_generation"]:
    print("⚠️  BASIC CSV GENERATION WORKS BUT SOME FEATURES MISSING")
    print("\nIssues:")
    if not test_results["real_data_in_csv"]:
        print("- CSV may be using simulated data")
    if not test_results["csv_metadata"]:
        print("- Metadata not included in CSV")
    if not test_results["yaml_export"]:
        print("- YAML export not working")
else:
    print("❌ OUTPUT GENERATION HAS CRITICAL ISSUES")
    print("\nThe system cannot generate CSV outputs properly")
    print("This may be due to configuration requirements")