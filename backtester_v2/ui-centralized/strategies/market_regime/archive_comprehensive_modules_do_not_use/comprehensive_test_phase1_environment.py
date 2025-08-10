#!/usr/bin/env python3
"""
Phase 1: Environment Setup & Preparation
=========================================

Test all system components and service connections:
1. Verify HeavyDB connection (host: localhost, port: 6274)
2. Check MySQL archive access (host: 106.51.63.60)
3. Confirm Redis service running
4. Start backend services
5. Access UI at correct endpoints

Duration: 30 minutes
Priority: HIGH
"""

import sys
import logging
import time
import requests
import redis
from datetime import datetime
import subprocess
import socket

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')

print("=" * 80)
print("PHASE 1: ENVIRONMENT SETUP & PREPARATION")
print("=" * 80)

class EnvironmentValidator:
    """Comprehensive environment validation"""
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()
        
    def test_heavydb_connection(self) -> bool:
        """Test 1: Verify HeavyDB connection"""
        logger.info("🔍 Test 1: Verifying HeavyDB connection...")
        
        try:
            from backtester_v2.dal.heavydb_connection import get_connection, execute_query
            
            conn = get_connection()
            if not conn:
                raise RuntimeError("Failed to establish HeavyDB connection")
            
            # Test query
            result = execute_query(conn, "SELECT COUNT(*) FROM nifty_option_chain LIMIT 1")
            if result.empty:
                raise RuntimeError("HeavyDB query failed")
            
            count = result.iloc[0, 0]
            logger.info(f"✅ HeavyDB connected: {count:,} total records")
            self.validation_results['heavydb'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ HeavyDB connection failed: {e}")
            self.validation_results['heavydb'] = False
            return False
    
    def test_mysql_archive_connection(self) -> bool:
        """Test 2: Check MySQL archive access"""
        logger.info("🔍 Test 2: Checking MySQL archive access...")
        
        try:
            import mysql.connector
            
            # Archive MySQL connection details
            config = {
                'host': '106.51.63.60',
                'user': 'mahesh',
                'password': 'mahesh_123',
                'database': 'historicaldb',
                'port': 3306
            }
            
            # Test connection
            conn = mysql.connector.connect(**config)
            cursor = conn.cursor()
            
            # Test query
            cursor.execute("SELECT COUNT(*) FROM option_chain LIMIT 1")
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                logger.info(f"✅ MySQL archive connected: {result[0]:,} records")
                self.validation_results['mysql_archive'] = True
            else:
                raise RuntimeError("No data in MySQL archive")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ MySQL archive connection failed: {e}")
            self.validation_results['mysql_archive'] = False
            return False
    
    def test_local_mysql_connection(self) -> bool:
        """Test 3: Check local MySQL connection"""
        logger.info("🔍 Test 3: Checking local MySQL connection...")
        
        try:
            import mysql.connector
            
            # Local MySQL connection details
            config = {
                'host': 'localhost',
                'user': 'mahesh',
                'password': 'mahesh_123',
                'database': 'historicaldb',
                'port': 3306
            }
            
            # Test connection
            conn = mysql.connector.connect(**config)
            cursor = conn.cursor()
            
            # Test query
            cursor.execute("SELECT COUNT(*) FROM option_chain LIMIT 1")
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                logger.info(f"✅ Local MySQL connected: {result[0]:,} records")
                self.validation_results['mysql_local'] = True
            else:
                logger.warning("⚠️  Local MySQL has no data")
                self.validation_results['mysql_local'] = False
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Local MySQL connection failed: {e}")
            self.validation_results['mysql_local'] = False
            return False
    
    def test_redis_service(self) -> bool:
        """Test 4: Confirm Redis service running"""
        logger.info("🔍 Test 4: Confirming Redis service...")
        
        try:
            # Test Redis connection
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test basic operations
            test_key = 'market_regime_test'
            test_value = f'test_{int(time.time())}'
            
            r.set(test_key, test_value, ex=60)  # Expire in 60 seconds
            retrieved_value = r.get(test_key)
            
            if retrieved_value == test_value:
                logger.info("✅ Redis service operational")
                r.delete(test_key)  # Clean up
                self.validation_results['redis'] = True
                return True
            else:
                raise RuntimeError("Redis read/write test failed")
                
        except Exception as e:
            logger.error(f"❌ Redis service failed: {e}")
            self.validation_results['redis'] = False
            return False
    
    def test_backend_services(self) -> bool:
        """Test 5: Check backend API services"""
        logger.info("🔍 Test 5: Checking backend API services...")
        
        try:
            # Test main API endpoint
            base_url = "http://localhost:8000"
            
            # Test health check
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Main API service operational")
            else:
                logger.warning(f"⚠️  Main API returned status: {response.status_code}")
            
            # Test market regime API endpoints
            endpoints = [
                "/api/v1/market-regime/status",
                "/api/v1/market-regime/config"
            ]
            
            endpoint_results = []
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ Endpoint {endpoint}: operational")
                        endpoint_results.append(True)
                    else:
                        logger.warning(f"⚠️  Endpoint {endpoint}: status {response.status_code}")
                        endpoint_results.append(False)
                except Exception as e:
                    logger.warning(f"⚠️  Endpoint {endpoint}: {e}")
                    endpoint_results.append(False)
            
            # Backend is considered operational if at least one endpoint works
            backend_operational = any(endpoint_results)
            self.validation_results['backend_services'] = backend_operational
            
            if backend_operational:
                logger.info("✅ Backend services operational")
            else:
                logger.error("❌ Backend services not responding")
            
            return backend_operational
            
        except Exception as e:
            logger.error(f"❌ Backend service check failed: {e}")
            self.validation_results['backend_services'] = False
            return False
    
    def test_ui_accessibility(self) -> bool:
        """Test 6: Access UI endpoints"""
        logger.info("🔍 Test 6: Testing UI accessibility...")
        
        try:
            base_url = "http://localhost:8000"
            
            # Test UI endpoints
            ui_endpoints = [
                "/static/index_enterprise.html",
                "/static/market_regime_enhanced.html"
            ]
            
            ui_results = []
            for endpoint in ui_endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ UI endpoint {endpoint}: accessible")
                        ui_results.append(True)
                    else:
                        logger.warning(f"⚠️  UI endpoint {endpoint}: status {response.status_code}")
                        ui_results.append(False)
                except Exception as e:
                    logger.warning(f"⚠️  UI endpoint {endpoint}: {e}")
                    ui_results.append(False)
            
            ui_accessible = any(ui_results)
            self.validation_results['ui_accessibility'] = ui_accessible
            
            if ui_accessible:
                logger.info("✅ UI endpoints accessible")
            else:
                logger.error("❌ UI endpoints not accessible")
            
            return ui_accessible
            
        except Exception as e:
            logger.error(f"❌ UI accessibility check failed: {e}")
            self.validation_results['ui_accessibility'] = False
            return False
    
    def test_file_system_access(self) -> bool:
        """Test 7: Verify file system access for uploads and outputs"""
        logger.info("🔍 Test 7: Testing file system access...")
        
        try:
            import os
            import tempfile
            
            # Test key directories
            test_paths = [
                "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced",
                "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime",
                "/tmp"
            ]
            
            access_results = []
            for path in test_paths:
                try:
                    if os.path.exists(path) and os.access(path, os.R_OK | os.W_OK):
                        # Test write access
                        test_file = os.path.join(path, f"test_write_{int(time.time())}.tmp")
                        with open(test_file, 'w') as f:
                            f.write("test")
                        os.remove(test_file)
                        
                        logger.info(f"✅ Directory {path}: read/write access")
                        access_results.append(True)
                    else:
                        logger.warning(f"⚠️  Directory {path}: access denied")
                        access_results.append(False)
                except Exception as e:
                    logger.warning(f"⚠️  Directory {path}: {e}")
                    access_results.append(False)
            
            filesystem_ok = all(access_results)
            self.validation_results['filesystem_access'] = filesystem_ok
            
            if filesystem_ok:
                logger.info("✅ File system access verified")
            else:
                logger.warning("⚠️  Some file system access issues detected")
            
            return filesystem_ok
            
        except Exception as e:
            logger.error(f"❌ File system access check failed: {e}")
            self.validation_results['filesystem_access'] = False
            return False
    
    def test_network_connectivity(self) -> bool:
        """Test 8: Network connectivity for external services"""
        logger.info("🔍 Test 8: Testing network connectivity...")
        
        try:
            # Test connectivity to key services
            connectivity_tests = [
                ("localhost", 8000, "Main API server"),
                ("localhost", 6379, "Redis server"),
                ("localhost", 3306, "Local MySQL"),
                ("106.51.63.60", 3306, "Archive MySQL"),
                ("localhost", 6274, "HeavyDB server")
            ]
            
            connectivity_results = []
            for host, port, service in connectivity_tests:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        logger.info(f"✅ {service} ({host}:{port}): reachable")
                        connectivity_results.append(True)
                    else:
                        logger.warning(f"⚠️  {service} ({host}:{port}): not reachable")
                        connectivity_results.append(False)
                except Exception as e:
                    logger.warning(f"⚠️  {service} ({host}:{port}): {e}")
                    connectivity_results.append(False)
            
            # At least 70% of services should be reachable
            connectivity_rate = sum(connectivity_results) / len(connectivity_results)
            network_ok = connectivity_rate >= 0.7
            
            self.validation_results['network_connectivity'] = network_ok
            
            if network_ok:
                logger.info(f"✅ Network connectivity: {connectivity_rate:.1%}")
            else:
                logger.warning(f"⚠️  Network connectivity low: {connectivity_rate:.1%}")
            
            return network_ok
            
        except Exception as e:
            logger.error(f"❌ Network connectivity check failed: {e}")
            self.validation_results['network_connectivity'] = False
            return False
    
    def generate_environment_report(self) -> dict:
        """Generate comprehensive environment report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Calculate overall success
        all_validations = list(self.validation_results.values())
        overall_success = all(all_validations)
        success_rate = sum(all_validations) / len(all_validations) * 100 if all_validations else 0
        
        # Identify critical vs non-critical failures
        critical_services = ['heavydb', 'backend_services', 'ui_accessibility']
        critical_failures = []
        non_critical_failures = []
        
        for service, result in self.validation_results.items():
            if not result:
                if service in critical_services:
                    critical_failures.append(service)
                else:
                    non_critical_failures.append(service)
        
        report = {
            'phase': 'Phase 1: Environment Setup & Preparation',
            'duration_seconds': round(duration, 2),
            'overall_success': overall_success,
            'success_rate': round(success_rate, 1),
            'validations': self.validation_results,
            'critical_failures': critical_failures,
            'non_critical_failures': non_critical_failures,
            'timestamp': datetime.now().isoformat()
        }
        
        return report

def main():
    """Execute Phase 1 environment validation"""
    print("🚀 Starting Phase 1: Environment Setup & Preparation")
    print("📋 Testing all system components and service connections")
    
    validator = EnvironmentValidator()
    
    # Execute all validation tests
    tests = [
        validator.test_heavydb_connection,
        validator.test_mysql_archive_connection,
        validator.test_local_mysql_connection,
        validator.test_redis_service,
        validator.test_backend_services,
        validator.test_ui_accessibility,
        validator.test_file_system_access,
        validator.test_network_connectivity
    ]
    
    print("\n" + "="*60)
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Executing: {test.__name__}")
        test()
    
    # Generate final report
    report = validator.generate_environment_report()
    
    print("\n" + "="*80)
    print("PHASE 1 ENVIRONMENT RESULTS")
    print("="*80)
    
    print(f"⏱️  Duration: {report['duration_seconds']} seconds")
    print(f"📊 Success Rate: {report['success_rate']}%")
    
    print(f"\n📋 Service Status:")
    for service, result in report['validations'].items():
        status = "✅ OPERATIONAL" if result else "❌ FAILED"
        print(f"   {service.replace('_', ' ').title()}: {status}")
    
    if report['critical_failures']:
        print(f"\n❌ CRITICAL FAILURES:")
        for failure in report['critical_failures']:
            print(f"   • {failure.replace('_', ' ').title()}")
    
    if report['non_critical_failures']:
        print(f"\n⚠️  NON-CRITICAL ISSUES:")
        for failure in report['non_critical_failures']:
            print(f"   • {failure.replace('_', ' ').title()}")
    
    print(f"\n🎯 OVERALL RESULT: {'✅ PHASE 1 PASSED' if report['overall_success'] else '❌ PHASE 1 FAILED'}")
    
    if report['overall_success']:
        print("\n🚀 Environment ready - Proceeding to Phase 2: UI Upload Testing")
    elif not report['critical_failures']:
        print("\n⚠️  Environment mostly ready - Can proceed with caution")
        print("   → Some non-critical services may need attention")
    else:
        print("\n🛑 MUST fix critical service issues before proceeding")
        for failure in report['critical_failures']:
            print(f"   → Fix {failure.replace('_', ' ')}")
    
    return report['overall_success'] or len(report['critical_failures']) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)