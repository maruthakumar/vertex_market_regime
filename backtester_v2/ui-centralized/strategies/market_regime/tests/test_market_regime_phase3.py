#!/usr/bin/env python3
"""
Market Regime System Testing - Phase 3: Backend Processing and Excel to YAML Conversion
Tests the complete backend workflow for market regime processing

Test Coverage:
1. Excel file upload to correct endpoint
2. Excel to YAML conversion process
3. Configuration validation
4. Backend processing pipeline
5. Real HeavyDB data usage (NO MOCK DATA)
"""

import sys
import os
import time
import json
import requests
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration manager
from config_manager import get_config_manager
config_manager = get_config_manager()


class BackendProcessingTester:
    """Tests backend processing and Excel to YAML conversion"""
    
    def __init__(self):
        self.base_url = "http://173.208.247.17:8000"
        self.api_base = f"{self.base_url}/api/market-regime"
        self.excel_file = config_manager.get_excel_config_path("PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx")
        self.test_results = {
            "upload_success": False,
            "yaml_conversion": False,
            "config_validation": False,
            "heavydb_connection": False,
            "processing_complete": False,
            "no_mock_data": False
        }
        self.upload_response = None
        
    def test_upload_endpoint(self) -> bool:
        """Test file upload to correct market regime endpoint"""
        try:
            print("\n📤 Testing Market Regime Upload Endpoint...")
            print(f"   Endpoint: {self.api_base}/upload")
            
            # Verify file exists
            if not Path(self.excel_file).exists():
                print(f"   ❌ Test file not found: {self.excel_file}")
                return False
                
            # Upload file
            with open(self.excel_file, 'rb') as f:
                files = {'configFile': (
                    'market_regime_config.xlsx', 
                    f, 
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )}
                
                response = requests.post(
                    f"{self.api_base}/upload",
                    files=files,
                    timeout=60  # Longer timeout for processing
                )
                
            print(f"   Response status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ Upload successful!")
                self.test_results["upload_success"] = True
                
                try:
                    self.upload_response = response.json()
                    print(f"   Response summary:")
                    
                    # Check validation results
                    if 'validation_results' in self.upload_response:
                        val_results = self.upload_response['validation_results']
                        print(f"     - Sheets found: {val_results.get('sheets_found', 'N/A')}")
                        print(f"     - Validation passed: {val_results.get('valid', 'N/A')}")
                        
                    # Check YAML conversion
                    if 'yaml_conversion' in self.upload_response.get('validation_results', {}):
                        yaml_conv = self.upload_response['validation_results']['yaml_conversion']
                        print(f"     - YAML conversion: {'✅' if yaml_conv.get('success') else '❌'}")
                        print(f"     - Sheets processed: {yaml_conv.get('sheets_processed', 0)}")
                        print(f"     - Success rate: {yaml_conv.get('success_rate', 0)}%")
                        
                        if yaml_conv.get('success'):
                            self.test_results["yaml_conversion"] = True
                            
                    return True
                    
                except json.JSONDecodeError:
                    print(f"   Response text: {response.text[:500]}...")
                    
            elif response.status_code == 422:
                print("   ❌ Validation error:")
                print(f"   {response.text}")
            elif response.status_code == 500:
                print("   ❌ Server error:")
                print(f"   {response.text[:500]}")
            else:
                print(f"   ❌ Unexpected response: {response.text[:500]}")
                
            return False
            
        except Exception as e:
            print(f"❌ Upload test failed: {e}")
            return False
    
    def test_yaml_conversion(self) -> bool:
        """Test Excel to YAML conversion details"""
        try:
            print("\n🔄 Testing Excel to YAML Conversion...")
            
            # Check conversion status
            response = requests.get(f"{self.api_base}/config", timeout=10)
            
            if response.status_code == 200:
                config_data = response.json()
                
                if 'yaml_config' in config_data:
                    print("   ✅ YAML configuration found")
                    
                    # Check key sections
                    yaml_config = config_data['yaml_config']
                    if isinstance(yaml_config, dict):
                        sections = list(yaml_config.keys())
                        print(f"   YAML sections found: {len(sections)}")
                        
                        # Expected sections
                        expected = [
                            'master_configuration',
                            'stability_configuration',
                            'transition_management',
                            'indicator_configuration',
                            'regime_classification'
                        ]
                        
                        for section in expected:
                            if section in yaml_config:
                                print(f"   ✅ {section}")
                            else:
                                print(f"   ❌ Missing: {section}")
                                
                        self.test_results["yaml_conversion"] = len(sections) > 20
                        
                    # Save YAML for inspection
                    output_file = Path("test_output/converted_config.yaml")
                    output_file.parent.mkdir(exist_ok=True)
                    
                    with open(output_file, 'w') as f:
                        yaml.dump(yaml_config, f, default_flow_style=False)
                        
                    print(f"\n   YAML saved to: {output_file}")
                    
                    return True
                    
                else:
                    print("   ❌ No YAML configuration in response")
                    
            else:
                print(f"   ❌ Config endpoint returned: {response.status_code}")
                
            return False
            
        except Exception as e:
            print(f"❌ YAML conversion test failed: {e}")
            return False
    
    def test_config_validation(self) -> bool:
        """Test configuration validation"""
        try:
            print("\n✅ Testing Configuration Validation...")
            
            # Validate the uploaded config
            response = requests.post(f"{self.api_base}/validate", timeout=30)
            
            if response.status_code == 200:
                validation = response.json()
                
                print("   Validation Results:")
                print(f"   - Overall valid: {validation.get('valid', False)}")
                
                if 'details' in validation:
                    details = validation['details']
                    print(f"   - Master config: {'✅' if details.get('master_configuration', {}).get('valid') else '❌'}")
                    print(f"   - Indicators: {'✅' if details.get('indicator_configuration', {}).get('valid') else '❌'}")
                    print(f"   - Regime classification: {'✅' if details.get('regime_classification', {}).get('valid') else '❌'}")
                    
                if 'warnings' in validation and validation['warnings']:
                    print(f"\n   ⚠️  Warnings:")
                    for warning in validation['warnings'][:3]:
                        print(f"      - {warning}")
                        
                if 'errors' in validation and validation['errors']:
                    print(f"\n   ❌ Errors:")
                    for error in validation['errors'][:3]:
                        print(f"      - {error}")
                        
                self.test_results["config_validation"] = validation.get('valid', False)
                return True
                
            else:
                print(f"   ❌ Validation endpoint returned: {response.status_code}")
                
            return False
            
        except Exception as e:
            print(f"❌ Config validation test failed: {e}")
            return False
    
    def test_heavydb_connection(self) -> bool:
        """Verify HeavyDB is being used (NOT mock data)"""
        try:
            print("\n🗄️  Verifying HeavyDB Connection (NO MOCK DATA)...")
            
            # Check data source status
            response = requests.get(f"{self.api_base}/status", timeout=10)
            
            if response.status_code == 200:
                status = response.json()
                
                # Check data source
                data_source = status.get('data_source', {})
                print(f"   Data source type: {data_source.get('type', 'Unknown')}")
                print(f"   Connected: {data_source.get('connected', False)}")
                
                if data_source.get('type') == 'heavydb' and data_source.get('connected'):
                    print("   ✅ Using real HeavyDB data")
                    self.test_results["heavydb_connection"] = True
                    self.test_results["no_mock_data"] = True
                    
                    # Verify data statistics
                    if 'statistics' in data_source:
                        stats = data_source['statistics']
                        print(f"   - Total records: {stats.get('total_records', 0):,}")
                        print(f"   - Latest date: {stats.get('latest_date', 'N/A')}")
                        
                elif data_source.get('type') == 'mock':
                    print("   ❌ CRITICAL: Using mock data - NOT ALLOWED!")
                    self.test_results["no_mock_data"] = False
                    return False
                    
            else:
                print(f"   ❌ Status endpoint returned: {response.status_code}")
                
            return self.test_results["heavydb_connection"]
            
        except Exception as e:
            print(f"❌ HeavyDB verification failed: {e}")
            return False
    
    def test_processing_pipeline(self) -> bool:
        """Test the complete processing pipeline"""
        try:
            print("\n⚙️  Testing Processing Pipeline...")
            
            # Trigger calculation with sample parameters
            calc_params = {
                "start_date": "2025-06-01",
                "end_date": "2025-06-18",
                "symbol": "NIFTY",
                "timeframe": "5min"
            }
            
            print(f"   Parameters: {json.dumps(calc_params, indent=2)}")
            
            response = requests.post(
                f"{self.api_base}/calculate",
                json=calc_params,
                timeout=120  # Long timeout for processing
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n   Processing Results:")
                print(f"   - Status: {result.get('status', 'Unknown')}")
                print(f"   - Records processed: {result.get('records_processed', 0):,}")
                
                if 'regime_summary' in result:
                    summary = result['regime_summary']
                    print(f"\n   Regime Distribution:")
                    for regime, count in summary.items():
                        print(f"     - {regime}: {count}")
                        
                if 'indicators' in result:
                    indicators = result['indicators']
                    print(f"\n   Indicators Calculated:")
                    for ind_name, ind_data in list(indicators.items())[:5]:
                        print(f"     - {ind_name}: {ind_data.get('status', 'N/A')}")
                        
                self.test_results["processing_complete"] = result.get('status') == 'success'
                return True
                
            elif response.status_code == 422:
                print("   ❌ Invalid parameters:")
                print(f"   {response.text}")
            else:
                print(f"   ❌ Processing failed: {response.status_code}")
                print(f"   {response.text[:500]}")
                
            return False
            
        except Exception as e:
            print(f"❌ Processing pipeline test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all backend processing tests"""
        print("=" * 80)
        print("PHASE 3: BACKEND PROCESSING AND EXCEL TO YAML CONVERSION")
        print("=" * 80)
        print(f"Testing with: {os.path.basename(self.excel_file)}")
        
        # Run tests in sequence
        if self.test_upload_endpoint():
            self.test_yaml_conversion()
            self.test_config_validation()
            
        self.test_heavydb_connection()
        
        if self.test_results["config_validation"] and self.test_results["heavydb_connection"]:
            self.test_processing_pipeline()
        
        # Summary
        print("\n" + "=" * 80)
        print("PHASE 3 TEST SUMMARY")
        print("=" * 80)
        
        for test, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            critical = " (CRITICAL)" if test in ["no_mock_data", "heavydb_connection"] else ""
            print(f"{test.replace('_', ' ').title()}: {status}{critical}")
        
        # Overall assessment
        critical_passed = (
            self.test_results["upload_success"] and
            self.test_results["yaml_conversion"] and
            self.test_results["no_mock_data"]
        )
        
        print("\n" + "=" * 80)
        if critical_passed:
            print("✅ BACKEND PROCESSING VALIDATED")
            print("   - Excel upload working")
            print("   - YAML conversion successful")
            print("   - Using REAL HeavyDB data (NO MOCK)")
            
            if self.test_results["processing_complete"]:
                print("   - Full processing pipeline operational")
            else:
                print("\n⚠️  Processing pipeline needs attention")
                
        else:
            print("❌ BACKEND PROCESSING ISSUES")
            
            if not self.test_results["no_mock_data"]:
                print("\n🚨 CRITICAL: System is using MOCK DATA!")
                print("   This violates the requirement to use ONLY real HeavyDB data")
                print("   Fix this immediately before proceeding")
                
        return self.test_results


if __name__ == "__main__":
    tester = BackendProcessingTester()
    results = tester.run_all_tests()
    
    # Exit based on critical tests
    critical_passed = (
        results["upload_success"] and
        results["yaml_conversion"] and
        results["no_mock_data"]
    )
    sys.exit(0 if critical_passed else 1)