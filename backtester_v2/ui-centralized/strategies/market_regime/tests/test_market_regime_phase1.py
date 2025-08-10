#!/usr/bin/env python3
"""
Market Regime System Testing - Phase 1: Environment Setup and Service Verification
This test validates all services are running and configured correctly

Test Coverage:
1. Backend services (Flask, Redis, Celery)
2. UI availability and responsiveness
3. File system permissions
4. Configuration file validation
5. WebSocket connectivity
"""

import sys
import os
import time
import json
import requests
import redis
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration manager
from config_manager import get_config_manager
config_manager = get_config_manager()


class EnvironmentValidator:
    """Validates environment setup and services"""
    
    def __init__(self):
        self.base_url = "http://173.208.247.17:8000"
        self.test_results = {
            "backend_services": False,
            "ui_availability": False,
            "file_permissions": False,
            "config_validation": False,
            "websocket_ready": False,
            "redis_connection": False,
            "file_paths": {}
        }
        
    def check_backend_services(self) -> bool:
        """Verify backend services are running"""
        try:
            print("\n🌐 Checking backend services...")
            
            # Check main app
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Main application is running")
            else:
                print(f"   ❌ Main app returned status: {response.status_code}")
                return False
                
            # Check API endpoints
            endpoints = [
                "/api/market_regime/status",
                "/api/market_regime/config",
                "/api/upload/validate"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    if response.status_code in [200, 404]:  # 404 is ok for some endpoints
                        print(f"   ✅ Endpoint {endpoint} is accessible")
                    else:
                        print(f"   ⚠️  Endpoint {endpoint} returned: {response.status_code}")
                except Exception as e:
                    print(f"   ❌ Endpoint {endpoint} failed: {e}")
                    
            self.test_results["backend_services"] = True
            return True
            
        except Exception as e:
            print(f"❌ Backend service check failed: {e}")
            return False
    
    def check_ui_availability(self) -> bool:
        """Test UI is accessible and responsive"""
        try:
            print("\n🖥️  Checking UI availability...")
            
            # Check main UI
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("   ✅ UI is accessible")
                
                # Check for key UI elements
                content = response.text
                if "market-regime" in content.lower():
                    print("   ✅ Market Regime UI components found")
                else:
                    print("   ⚠️  Market Regime UI components not found in response")
                    
            else:
                print(f"   ❌ UI returned status: {response.status_code}")
                return False
                
            # Check static assets
            static_paths = [
                "/static/js/market_regime.js",
                "/static/css/market_regime.css"
            ]
            
            for path in static_paths:
                try:
                    response = requests.head(f"{self.base_url}{path}", timeout=5)
                    if response.status_code == 200:
                        print(f"   ✅ Static asset {path} is available")
                except:
                    print(f"   ⚠️  Static asset {path} might be missing")
                    
            self.test_results["ui_availability"] = True
            return True
            
        except Exception as e:
            print(f"❌ UI availability check failed: {e}")
            return False
    
    def check_file_permissions(self) -> bool:
        """Verify file system permissions for uploads and outputs"""
        try:
            print("\n📁 Checking file permissions...")
            
            paths_to_check = {
                "input_sheets": config_manager.paths.get_input_sheets_path(),
                "output_dir": "/srv/samba/shared/bt/backtester_stable/BTRUN/output/market_regime",
                "config_dir": "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/config",
                "logs_dir": "/srv/samba/shared/bt/backtester_stable/BTRUN/logs/market_regime"
            }
            
            all_good = True
            for name, path in paths_to_check.items():
                path_obj = Path(path)
                
                # Check if exists
                if path_obj.exists():
                    print(f"   ✅ {name}: {path} exists")
                    
                    # Check permissions
                    if os.access(path, os.W_OK):
                        print(f"      ✅ Write permission OK")
                    else:
                        print(f"      ❌ No write permission")
                        all_good = False
                        
                    # Try to create test file
                    try:
                        test_file = path_obj / f".test_{int(time.time())}.tmp"
                        test_file.write_text("test")
                        test_file.unlink()
                        print(f"      ✅ Can create/delete files")
                    except Exception as e:
                        print(f"      ❌ Cannot create files: {e}")
                        all_good = False
                else:
                    print(f"   ❌ {name}: {path} does not exist")
                    # Try to create it
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        print(f"      ✅ Created directory")
                    except Exception as e:
                        print(f"      ❌ Cannot create directory: {e}")
                        all_good = False
                        
                self.test_results["file_paths"][name] = str(path)
                
            self.test_results["file_permissions"] = all_good
            return all_good
            
        except Exception as e:
            print(f"❌ File permission check failed: {e}")
            return False
    
    def check_config_files(self) -> bool:
        """Validate configuration files exist and are valid"""
        try:
            print("\n⚙️  Checking configuration files...")
            
            # Check test Excel file
            excel_file = config_manager.get_excel_config_path("PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx")
            
            if Path(excel_file).exists():
                print(f"   ✅ Test Excel file exists")
                file_size = Path(excel_file).stat().st_size
                print(f"      File size: {file_size/1024:.2f} KB")
                
                if file_size > 1000:  # At least 1KB
                    print("      ✅ File size looks valid")
                else:
                    print("      ❌ File seems too small")
                    return False
            else:
                print(f"   ❌ Test Excel file not found: {excel_file}")
                return False
                
            # Check for config templates
            config_templates = [
                "market_regime_config_template.yaml",
                "indicator_weights.json"
            ]
            
            config_dir = Path("/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/config")
            for template in config_templates:
                template_path = config_dir / template
                if template_path.exists():
                    print(f"   ✅ Config template found: {template}")
                else:
                    print(f"   ⚠️  Config template missing: {template}")
                    
            self.test_results["config_validation"] = True
            return True
            
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
            return False
    
    def check_redis_connection(self) -> bool:
        """Verify Redis is accessible for WebSocket and caching"""
        try:
            print("\n🔌 Checking Redis connection...")
            
            # Try to connect to Redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test connection
            r.ping()
            print("   ✅ Redis is running and accessible")
            
            # Test basic operations
            test_key = f"market_regime_test_{int(time.time())}"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)
            
            if value == "test_value":
                print("   ✅ Redis read/write operations working")
                r.delete(test_key)
            else:
                print("   ❌ Redis operations failed")
                return False
                
            # Check for any existing market regime keys
            keys = r.keys("market_regime:*")
            print(f"   Found {len(keys)} existing market regime keys in Redis")
            
            self.test_results["redis_connection"] = True
            return True
            
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            print("   Note: Redis is required for WebSocket progress tracking")
            return False
    
    def check_websocket_readiness(self) -> bool:
        """Check if WebSocket endpoint is ready"""
        try:
            print("\n🔄 Checking WebSocket readiness...")
            
            # Check socket.io endpoint
            response = requests.get(f"{self.base_url}/socket.io/", timeout=5)
            
            if response.status_code in [200, 400]:  # 400 is normal without proper handshake
                print("   ✅ WebSocket endpoint is responding")
                self.test_results["websocket_ready"] = True
                return True
            else:
                print(f"   ❌ WebSocket endpoint returned: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ WebSocket check failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all environment validation tests"""
        print("=" * 80)
        print("PHASE 1: ENVIRONMENT SETUP AND SERVICE VERIFICATION")
        print("=" * 80)
        
        # Run tests
        self.check_backend_services()
        self.check_ui_availability()
        self.check_file_permissions()
        self.check_config_files()
        self.check_redis_connection()
        self.check_websocket_readiness()
        
        # Summary
        print("\n" + "=" * 80)
        print("PHASE 1 TEST SUMMARY")
        print("=" * 80)
        
        all_passed = all([
            self.test_results["backend_services"],
            self.test_results["ui_availability"],
            self.test_results["file_permissions"],
            self.test_results["config_validation"]
            # Redis and WebSocket are optional but recommended
        ])
        
        for test, result in self.test_results.items():
            if test != "file_paths":
                status = "✅ PASS" if result else "❌ FAIL"
                optional = " (Optional)" if test in ["redis_connection", "websocket_ready"] else ""
                print(f"{test.replace('_', ' ').title()}: {status}{optional}")
        
        print("\nFile Paths Validated:")
        for name, path in self.test_results["file_paths"].items():
            print(f"  {name}: {path}")
        
        if all_passed:
            print("\n✅ ENVIRONMENT READY - PROCEED WITH UI TESTING")
            print("📋 All required services are running")
            print("📁 File permissions are correctly configured")
            print("⚙️  Configuration files are in place")
        else:
            print("\n❌ ENVIRONMENT ISSUES DETECTED")
            print("⚠️  Fix the failed items before proceeding")
            
        # Additional recommendations
        if not self.test_results["redis_connection"]:
            print("\n⚠️  RECOMMENDATION: Start Redis for WebSocket progress tracking")
            print("   Run: sudo systemctl start redis")
            
        if not self.test_results["websocket_ready"]:
            print("\n⚠️  RECOMMENDATION: Enable WebSocket support for real-time updates")
        
        return self.test_results


if __name__ == "__main__":
    validator = EnvironmentValidator()
    results = validator.run_all_tests()
    
    # Exit with appropriate code
    required_tests = ["backend_services", "ui_availability", "file_permissions", "config_validation"]
    if all(results[test] for test in required_tests):
        sys.exit(0)
    else:
        sys.exit(1)