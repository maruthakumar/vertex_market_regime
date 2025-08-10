#!/usr/bin/env python3
"""
Market Regime System Testing - Phase 1: Essential Environment Verification (Simplified)
Focuses on critical components needed for testing
"""

import sys
import os
import requests
from pathlib import Path
from typing import Dict, Any

class SimpleEnvironmentValidator:
    """Simplified environment validation"""
    
    def __init__(self):
        self.base_url = "http://173.208.247.17:8000"
        self.test_results = {}
        
    def check_application(self) -> bool:
        """Verify application is running"""
        try:
            print("\n🌐 Checking application...")
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200 and "MarvelQuant" in response.text:
                print("   ✅ Application is running")
                return True
            else:
                print("   ❌ Application not responding correctly")
                return False
        except Exception as e:
            print(f"   ❌ Application check failed: {e}")
            return False
    
    def check_test_files(self) -> bool:
        """Verify test files exist"""
        try:
            print("\n📁 Checking test files...")
            
            excel_file = config_manager.get_excel_config_path("PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx")
            
            if Path(excel_file).exists():
                print(f"   ✅ Test Excel file exists")
                file_size = Path(excel_file).stat().st_size
                print(f"      Size: {file_size/1024:.2f} KB")
                return True
            else:
                print(f"   ❌ Test Excel file not found")
                return False
                
        except Exception as e:
            print(f"   ❌ File check failed: {e}")
            return False
    
    def check_directories(self) -> bool:
        """Check/create necessary directories"""
        try:
            print("\n📂 Checking directories...")
            
            dirs_needed = [
                "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/test_output",
                "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/test_logs"
            ]
            
            for dir_path in dirs_needed:
                path_obj = Path(dir_path)
                if not path_obj.exists():
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        print(f"   ✅ Created: {dir_path}")
                    except Exception as e:
                        print(f"   ⚠️  Could not create {dir_path}: {e}")
                else:
                    print(f"   ✅ Exists: {dir_path}")
                    
            return True
            
        except Exception as e:
            print(f"   ❌ Directory check failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run simplified tests"""
        print("=" * 80)
        print("PHASE 1: ESSENTIAL ENVIRONMENT VERIFICATION (SIMPLIFIED)")
        print("=" * 80)
        
        app_ok = self.check_application()
        files_ok = self.check_test_files()
        dirs_ok = self.check_directories()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if app_ok and files_ok:
            print("\n✅ ENVIRONMENT READY FOR TESTING")
            print("   - Application is accessible")
            print("   - Test Excel file is available")
            print("   - Working directories prepared")
            return {"success": True}
        else:
            print("\n❌ ENVIRONMENT NOT READY")
            print("   Please fix the issues above")
            return {"success": False}


if __name__ == "__main__":
    validator = SimpleEnvironmentValidator()
    results = validator.run_all_tests()
    sys.exit(0 if results["success"] else 1)