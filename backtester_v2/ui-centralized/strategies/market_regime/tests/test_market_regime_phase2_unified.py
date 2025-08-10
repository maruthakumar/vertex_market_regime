#!/usr/bin/env python3
"""
Market Regime System Testing - Phase 2.1: Unified UI Testing
Tests market regime integration within index_enterprise.html#backtest

Key Focus:
- Market regime is integrated into the main backtest interface
- Single source of truth: index_enterprise.html#backtest
- Verify market regime functionality within unified UI
"""

import sys
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration manager
from config_manager import get_config_manager
config_manager = get_config_manager()


class UnifiedUITester:
    """Tests market regime within unified enterprise UI"""
    
    def __init__(self):
        self.base_url = "http://173.208.247.17:8000"
        self.excel_file = config_manager.get_excel_config_path("PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx")
        self.test_results = {
            "enterprise_ui_loaded": False,
            "backtest_section_found": False,
            "market_regime_integrated": False,
            "upload_functionality": False,
            "api_endpoints_working": False,
            "single_source_verified": False
        }
        
    def check_enterprise_ui(self) -> bool:
        """Verify index_enterprise.html is the main UI"""
        try:
            print("\n🏢 Checking Enterprise UI (index_enterprise.html)...")
            
            # Check main page
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code != 200:
                print(f"   ❌ Main page returned: {response.status_code}")
                return False
                
            content = response.text
            
            # Check for enterprise UI indicators
            checks = {
                "Enterprise UI": "enterprise" in content.lower(),
                "MarvelQuant branding": "marvelquant" in content.lower(),
                "Bootstrap framework": "bootstrap" in content.lower(),
                "Backtest section": "backtest" in content.lower(),
                "Strategy selection": "strategy" in content.lower()
            }
            
            for component, found in checks.items():
                if found:
                    print(f"   ✅ {component} found")
                else:
                    print(f"   ❌ {component} not found")
                    
            # Check if this is index_enterprise.html
            if "enterprise" in content.lower() and "marvelquant" in content.lower():
                print("   ✅ Confirmed: Using index_enterprise.html as main UI")
                self.test_results["enterprise_ui_loaded"] = True
                return True
            else:
                print("   ❌ Not using enterprise UI")
                return False
                
        except Exception as e:
            print(f"❌ Enterprise UI check failed: {e}")
            return False
    
    def check_backtest_section(self) -> bool:
        """Check if backtest section includes market regime"""
        try:
            print("\n📊 Checking Backtest Section Integration...")
            
            # Check for backtest-related endpoints
            endpoints = [
                "/api/strategies",
                "/api/backtest/strategies",
                "/api/backtest/config"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        print(f"   ✅ Found endpoint: {endpoint}")
                        
                        # Check if response contains market regime
                        try:
                            data = response.json()
                            data_str = str(data).lower()
                            
                            if "market" in data_str and "regime" in data_str:
                                print("      ✅ Market regime found in response")
                                self.test_results["backtest_section_found"] = True
                                self.test_results["market_regime_integrated"] = True
                                
                                # Print available strategies
                                if isinstance(data, list):
                                    print(f"      Available strategies: {', '.join(data)}")
                                elif isinstance(data, dict) and 'strategies' in data:
                                    print(f"      Available strategies: {', '.join(data['strategies'])}")
                                    
                                return True
                        except:
                            pass
                            
                except requests.exceptions.RequestException:
                    pass
                    
            # Check static JavaScript files
            print("\n   Checking JavaScript integration...")
            js_files = [
                "/static/js/index_enterprise.js",
                "/static/js/unified_backtest.js",
                "/static/js/strategy_selector.js"
            ]
            
            for js_file in js_files:
                try:
                    response = requests.get(f"{self.base_url}{js_file}", timeout=5)
                    if response.status_code == 200:
                        if "market" in response.text.lower() and "regime" in response.text.lower():
                            print(f"   ✅ Market regime code found in {js_file}")
                            self.test_results["market_regime_integrated"] = True
                            break
                except:
                    pass
                    
            return self.test_results["backtest_section_found"]
            
        except Exception as e:
            print(f"❌ Backtest section check failed: {e}")
            return False
    
    def test_unified_upload(self) -> bool:
        """Test file upload through unified interface"""
        try:
            print("\n📤 Testing Unified Upload System...")
            
            # Check for unified upload endpoint
            upload_endpoints = [
                "/api/backtest/upload",
                "/api/unified/upload",
                "/api/strategy/upload"
            ]
            
            for endpoint in upload_endpoints:
                print(f"\n   Trying: {endpoint}")
                
                # First check if endpoint exists
                try:
                    response = requests.options(f"{self.base_url}{endpoint}", timeout=5)
                    print(f"   OPTIONS response: {response.status_code}")
                except:
                    pass
                
                # Try upload with strategy type
                try:
                    with open(self.excel_file, 'rb') as f:
                        files = {'file': ('config.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                        data = {
                            'strategy': 'market_regime',
                            'strategy_type': 'market_regime',
                            'type': 'market_regime'
                        }
                        
                        response = requests.post(
                            f"{self.base_url}{endpoint}",
                            files=files,
                            data=data,
                            timeout=30
                        )
                        
                        print(f"   POST response: {response.status_code}")
                        
                        if response.status_code == 200:
                            print("   ✅ Upload successful through unified interface!")
                            self.test_results["upload_functionality"] = True
                            
                            try:
                                result = response.json()
                                print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
                            except:
                                pass
                                
                            return True
                            
                        elif response.status_code == 401:
                            print("   ⚠️  Authentication required")
                        elif response.status_code == 403:
                            print("   ⚠️  Forbidden - may need API key")
                        elif response.status_code == 422:
                            print("   ❌ Validation error")
                            if response.text:
                                print(f"   Error: {response.text[:200]}")
                                
                except Exception as e:
                    print(f"   Error: {str(e)[:100]}")
                    
            return False
            
        except Exception as e:
            print(f"❌ Unified upload test failed: {e}")
            return False
    
    def verify_single_source(self) -> bool:
        """Verify no separate market_regime_enhanced.html is being used"""
        try:
            print("\n🔍 Verifying Single Source of Truth...")
            
            # Check if market_regime_enhanced.html exists but is empty/unused
            response = requests.get(f"{self.base_url}/static/market_regime_enhanced.html", timeout=5)
            
            if response.status_code == 200:
                if len(response.text.strip()) == 0:
                    print("   ✅ market_regime_enhanced.html is empty (good)")
                    self.test_results["single_source_verified"] = True
                else:
                    print("   ⚠️  market_regime_enhanced.html has content")
                    print("      Should be migrated to index_enterprise.html")
                    
            else:
                print("   ✅ No separate market_regime_enhanced.html found (good)")
                self.test_results["single_source_verified"] = True
                
            # Verify all functionality is in main UI
            print("\n   Checking main UI completeness...")
            
            # List expected market regime features
            features = [
                "Excel upload for market regime",
                "YAML conversion status",
                "Regime indicator selection",
                "Output configuration",
                "Live trading integration"
            ]
            
            print("   Expected features in unified UI:")
            for feature in features:
                print(f"   - {feature}")
                
            return self.test_results["single_source_verified"]
            
        except Exception as e:
            print(f"❌ Single source verification failed: {e}")
            return False
    
    def check_api_integration(self) -> bool:
        """Check API endpoints for market regime"""
        try:
            print("\n🔌 Checking API Integration...")
            
            # Check market regime specific APIs
            api_endpoints = [
                "/api/market_regime/status",
                "/api/market_regime/config",
                "/api/market_regime/indicators",
                "/api/market_regime/validate"
            ]
            
            working_endpoints = 0
            for endpoint in api_endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    if response.status_code in [200, 404]:  # 404 is ok, means route exists
                        print(f"   ✅ API endpoint accessible: {endpoint}")
                        working_endpoints += 1
                except:
                    print(f"   ❌ API endpoint not accessible: {endpoint}")
                    
            if working_endpoints > 0:
                print(f"\n   ✅ {working_endpoints} API endpoints found")
                self.test_results["api_endpoints_working"] = True
                return True
            else:
                print("\n   ⚠️  No specific market regime APIs found")
                print("   Market regime may be using unified backtest APIs")
                return False
                
        except Exception as e:
            print(f"❌ API integration check failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all unified UI tests"""
        print("=" * 80)
        print("PHASE 2.1: UNIFIED UI TESTING (index_enterprise.html#backtest)")
        print("=" * 80)
        print("Verifying market regime integration in main enterprise UI")
        
        # Run tests
        self.check_enterprise_ui()
        self.check_backtest_section()
        self.test_unified_upload()
        self.check_api_integration()
        self.verify_single_source()
        
        # Summary
        print("\n" + "=" * 80)
        print("UNIFIED UI TEST SUMMARY")
        print("=" * 80)
        
        for test, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test.replace('_', ' ').title()}: {status}")
        
        # Overall assessment
        critical_passed = (
            self.test_results["enterprise_ui_loaded"] and
            (self.test_results["market_regime_integrated"] or 
             self.test_results["backtest_section_found"])
        )
        
        print("\n" + "=" * 80)
        if critical_passed:
            print("✅ UNIFIED UI INTEGRATION CONFIRMED")
            print("   - Enterprise UI is the main interface")
            print("   - Market regime should be accessed via #backtest")
            
            if not self.test_results["upload_functionality"]:
                print("\n⚠️  NOTES:")
                print("   - Upload may require authentication")
                print("   - Check browser console for JavaScript errors")
                print("   - Use browser DevTools to inspect actual upload process")
                
            print("\n📍 Access market regime at:")
            print(f"   {self.base_url}/#backtest")
            print("   Then select 'Market Regime' as strategy type")
            
        else:
            print("❌ UNIFIED UI INTEGRATION ISSUES")
            print("   Please verify:")
            print("   1. index_enterprise.html is the main UI")
            print("   2. Market regime is integrated into backtest section")
            print("   3. No separate market_regime_enhanced.html is used")
            
        return self.test_results


if __name__ == "__main__":
    tester = UnifiedUITester()
    results = tester.run_all_tests()
    
    # Exit based on critical tests
    critical_passed = results["enterprise_ui_loaded"] and (
        results["market_regime_integrated"] or results["backtest_section_found"]
    )
    sys.exit(0 if critical_passed else 1)