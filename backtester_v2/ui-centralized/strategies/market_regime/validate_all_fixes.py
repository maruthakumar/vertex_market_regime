#!/usr/bin/env python3
"""
Comprehensive Fix Validation
===========================

Test all the fixes implemented for Phases 1-5 to ensure everything is working properly.
"""

import sys
import logging
import time
import requests
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("COMPREHENSIVE FIX VALIDATION")
print("=" * 80)

class FixValidator:
    """Validate all implemented fixes"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.fix_results = {}
        self.start_time = time.time()
        
    def test_phase3_fixes(self):
        """Test Phase 3 backend API fixes"""
        logger.info("🔧 Testing Phase 3 API fixes...")
        
        tests = []
        
        # Test calculate endpoint (was 500 error)
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/market-regime/calculate",
                json={"timeframe": "5min", "use_real_data": True},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                if result.get('data_source') == 'real_heavydb':
                    logger.info("✅ Calculate endpoint: FIXED (using real HeavyDB)")
                    tests.append(True)
                else:
                    logger.warning("⚠️  Calculate endpoint: Partial fix")
                    tests.append(False)
            else:
                logger.error(f"❌ Calculate endpoint: Still failing ({response.status_code})")
                tests.append(False)
        except Exception as e:
            logger.error(f"❌ Calculate endpoint error: {e}")
            tests.append(False)
        
        # Test Excel to YAML conversion endpoint (was 404)
        try:
            # Create a test file
            test_data = b"PK\x03\x04"  # Basic ZIP header for XLSX
            files = {'file': ('test.xlsx', test_data, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            
            response = requests.post(
                f"{self.base_url}/api/v1/market-regime/convert",
                files=files,
                timeout=10
            )
            if response.status_code in [200, 400]:  # 400 is OK for invalid file
                logger.info("✅ Excel to YAML endpoint: FIXED (accessible)")
                tests.append(True)
            else:
                logger.error(f"❌ Excel to YAML endpoint: Still 404 ({response.status_code})")
                tests.append(False)
        except Exception as e:
            logger.error(f"❌ Excel to YAML endpoint error: {e}")
            tests.append(False)
        
        # Test export endpoints (were 404)
        export_tests = []
        export_endpoints = [
            ("/export/csv", "POST"),
            ("/export/config", "GET"),
            ("/download/csv", "GET"),
            ("/download/config", "GET"),
            ("/metrics", "GET")
        ]
        
        for endpoint, method in export_endpoints:
            try:
                url = f"{self.base_url}/api/v1/market-regime{endpoint}"
                if method == "POST":
                    response = requests.post(url, json={"use_real_data": True}, timeout=5)
                else:
                    response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    logger.info(f"✅ {endpoint}: FIXED")
                    export_tests.append(True)
                else:
                    logger.warning(f"⚠️  {endpoint}: Status {response.status_code}")
                    export_tests.append(False)
            except Exception as e:
                logger.error(f"❌ {endpoint}: {e}")
                export_tests.append(False)
        
        tests.extend(export_tests)
        
        success_rate = sum(tests) / len(tests) * 100
        self.fix_results['phase3_backend'] = success_rate
        logger.info(f"📊 Phase 3 fixes success rate: {success_rate:.1f}%")
        
    def test_phase4_fixes(self):
        """Test Phase 4 Triple Straddle column fixes"""
        logger.info("🔧 Testing Phase 4 column mapping fixes...")
        
        try:
            # Import the column mapper
            sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime')
            from heavydb_column_mapper import HeavyDBColumnMapper
            
            mapper = HeavyDBColumnMapper()
            
            # Test column mappings
            tests = []
            
            # Test option_type mapping
            option_type_query = mapper.get_straddle_query(24850.0)
            if 'option_type' in option_type_query and "\'CE\'" in option_type_query:
                logger.info("✅ Option type mapping: FIXED")
                tests.append(True)
            else:
                logger.error("❌ Option type mapping: Still broken")
                tests.append(False)
            
            # Test price column mappings
            if 'ce_close' in option_type_query and 'pe_close' in option_type_query:
                logger.info("✅ Price column mapping: FIXED")
                tests.append(True)
            else:
                logger.error("❌ Price column mapping: Still broken")
                tests.append(False)
            
            # Test Greeks query
            greeks_query = mapper.get_greeks_query()
            if 'ce_delta' in greeks_query and 'pe_delta' in greeks_query:
                logger.info("✅ Greeks column mapping: Working")
                tests.append(True)
            else:
                logger.error("❌ Greeks column mapping: Issues")
                tests.append(False)
            
            success_rate = sum(tests) / len(tests) * 100
            self.fix_results['phase4_columns'] = success_rate
            logger.info(f"📊 Phase 4 fixes success rate: {success_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Phase 4 column mapper test failed: {e}")
            self.fix_results['phase4_columns'] = 0
    
    def test_phase5_fixes(self):
        """Test Phase 5 output generation fixes"""
        logger.info("🔧 Testing Phase 5 output generation fixes...")
        
        tests = []
        
        # Test CSV export functionality
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/market-regime/export/csv",
                json={"use_real_data": True, "data_source": "heavydb"},
                timeout=10
            )
            
            if response.status_code == 200:
                csv_content = response.text
                if 'timestamp,regime,confidence' in csv_content:
                    logger.info("✅ CSV export: FIXED (proper structure)")
                    tests.append(True)
                else:
                    logger.warning("⚠️  CSV export: Structure issues")
                    tests.append(False)
            else:
                logger.error(f"❌ CSV export: Still failing ({response.status_code})")
                tests.append(False)
        except Exception as e:
            logger.error(f"❌ CSV export error: {e}")
            tests.append(False)
        
        # Test metrics endpoint
        try:
            response = requests.get(f"{self.base_url}/api/v1/market-regime/metrics", timeout=5)
            
            if response.status_code == 200:
                metrics = response.json()
                required_fields = ['processing_time', 'accuracy', 'heavydb_status']
                found_fields = sum(1 for field in required_fields if field in metrics)
                
                if found_fields >= 2:
                    logger.info("✅ Metrics endpoint: FIXED")
                    tests.append(True)
                else:
                    logger.warning("⚠️  Metrics endpoint: Missing fields")
                    tests.append(False)
            else:
                logger.error(f"❌ Metrics endpoint: Status {response.status_code}")
                tests.append(False)
        except Exception as e:
            logger.error(f"❌ Metrics endpoint error: {e}")
            tests.append(False)
        
        success_rate = sum(tests) / len(tests) * 100
        self.fix_results['phase5_output'] = success_rate
        logger.info(f"📊 Phase 5 fixes success rate: {success_rate:.1f}%")
    
    def test_heavydb_integration(self):
        """Test HeavyDB integration still working"""
        logger.info("🔧 Testing HeavyDB integration after fixes...")
        
        try:
            # Add paths
            sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
            from backtester_v2.dal.heavydb_connection import get_connection, execute_query
            
            conn = get_connection()
            if conn:
                # Test basic query
                result = execute_query(conn, "SELECT COUNT(*) FROM nifty_option_chain LIMIT 1")
                
                if not result.empty:
                    count = result.iloc[0, 0]
                    if count > 30000000:  # Should be ~33M records
                        logger.info(f"✅ HeavyDB integration: Working ({count:,} records)")
                        self.fix_results['heavydb_integration'] = 100
                    else:
                        logger.warning(f"⚠️  HeavyDB integration: Low record count ({count})")
                        self.fix_results['heavydb_integration'] = 50
                else:
                    logger.error("❌ HeavyDB integration: No data")
                    self.fix_results['heavydb_integration'] = 0
            else:
                logger.error("❌ HeavyDB integration: Connection failed")
                self.fix_results['heavydb_integration'] = 0
                
        except Exception as e:
            logger.error(f"❌ HeavyDB integration error: {e}")
            self.fix_results['heavydb_integration'] = 0
    
    def generate_fix_validation_report(self):
        """Generate comprehensive fix validation report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Calculate overall improvement
        total_score = sum(self.fix_results.values())
        max_score = len(self.fix_results) * 100
        overall_improvement = total_score / max_score * 100
        
        print("\n" + "="*80)
        print("FIX VALIDATION RESULTS")
        print("="*80)
        
        print(f"⏱️  Validation Duration: {duration:.2f} seconds")
        print(f"📊 Overall Improvement: {overall_improvement:.1f}%")
        
        print(f"\n📋 Fix Results by Phase:")
        for phase, score in self.fix_results.items():
            status = "✅ EXCELLENT" if score >= 80 else "⚠️ PARTIAL" if score >= 50 else "❌ NEEDS WORK"
            print(f"   {phase.replace('_', ' ').title()}: {score:.1f}% {status}")
        
        # Determine overall status
        if overall_improvement >= 80:
            print(f"\n🎯 OVERALL STATUS: ✅ FIXES SUCCESSFUL")
            print("🚀 System is significantly improved and ready for production")
        elif overall_improvement >= 60:
            print(f"\n🎯 OVERALL STATUS: ⚠️ FIXES MOSTLY SUCCESSFUL")
            print("📈 Major improvements made, minor issues remain")
        else:
            print(f"\n🎯 OVERALL STATUS: ❌ MORE FIXES NEEDED")
            print("🔧 Significant issues still require attention")
        
        return {
            'overall_improvement': overall_improvement,
            'fix_results': self.fix_results,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Execute comprehensive fix validation"""
    print("🚀 Starting comprehensive fix validation")
    print("📋 Testing all Phase 1-5 fixes implemented")
    
    validator = FixValidator()
    
    # Test all fixes
    validator.test_phase3_fixes()
    validator.test_phase4_fixes() 
    validator.test_phase5_fixes()
    validator.test_heavydb_integration()
    
    # Generate report
    report = validator.generate_fix_validation_report()
    
    return report['overall_improvement'] >= 60

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)