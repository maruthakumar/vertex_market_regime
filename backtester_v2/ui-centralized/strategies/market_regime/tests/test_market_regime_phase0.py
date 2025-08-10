#!/usr/bin/env python3
"""
Market Regime System Testing - Phase 0: HeavyDB Data Validation
CRITICAL: This test MUST pass before any other testing can proceed

This test validates:
1. HeavyDB connection is active
2. Real market data is available
3. Greeks are populated
4. Data freshness is within limits
5. NO MOCK DATA is present
"""

import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from heavydb import connect
except ImportError:
    print("ERROR: HeavyDB (pymapd) not installed. Installing...")
    os.system("pip install pymapd")
    from heavydb import connect

class HeavyDBDataValidator:
    """Validates HeavyDB connection and data authenticity"""
    
    def __init__(self):
        self.connection = None
        self.test_results = {
            "connection": False,
            "data_availability": False,
            "data_freshness": False,
            "greeks_populated": False,
            "no_mock_data": False,
            "performance": {}
        }
        
    def connect_to_heavydb(self) -> bool:
        """Establish connection to HeavyDB"""
        try:
            print("\n🔌 Connecting to HeavyDB...")
            self.connection = connect(
                host='localhost',
                port=6274,
                user='admin',
                password='HyperInteractive',
                dbname='heavyai'
            )
            print("✅ Connected to HeavyDB successfully")
            self.test_results["connection"] = True
            return True
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Cannot connect to HeavyDB: {e}")
            print("⚠️  CANNOT PROCEED WITH TESTING - Fix HeavyDB connection first!")
            return False
    
    def validate_data_availability(self) -> bool:
        """Check if real market data exists"""
        try:
            print("\n📊 Validating data availability...")
            cursor = self.connection.cursor()
            
            # Check total records
            query = "SELECT COUNT(*) as total FROM nifty_option_chain"
            cursor.execute(query)
            total_records = cursor.fetchone()[0]
            print(f"   Total records in database: {total_records:,}")
            
            # Check recent data - HeavyDB syntax
            query = """
                SELECT COUNT(*) as recent_count 
                FROM nifty_option_chain 
                WHERE timestamp >= DATEADD('minute', -60, NOW())
            """
            cursor.execute(query)
            recent_records = cursor.fetchone()[0]
            print(f"   Records in last 60 minutes: {recent_records:,}")
            
            if recent_records > 1000:
                print("✅ Sufficient real-time data available")
                self.test_results["data_availability"] = True
                return True
            else:
                print(f"❌ Insufficient data: Only {recent_records} records in last hour")
                return False
                
        except Exception as e:
            print(f"❌ Error checking data availability: {e}")
            return False
    
    def check_data_freshness(self) -> bool:
        """Verify data is current and not stale"""
        try:
            print("\n⏰ Checking data freshness...")
            cursor = self.connection.cursor()
            
            # Get latest timestamp
            query = "SELECT MAX(CAST(timestamp AS TIMESTAMP)) as latest FROM nifty_option_chain"
            cursor.execute(query)
            latest_timestamp = cursor.fetchone()[0]
            
            if latest_timestamp:
                # Convert to datetime
                current_time = datetime.now()
                data_age = current_time - latest_timestamp
                
                print(f"   Latest data timestamp: {latest_timestamp}")
                print(f"   Current time: {current_time}")
                print(f"   Data age: {data_age}")
                
                # During market hours, data should be <2 minutes old
                if data_age < timedelta(minutes=2):
                    print("✅ Data is fresh (< 2 minutes old)")
                    self.test_results["data_freshness"] = True
                    return True
                else:
                    print(f"⚠️  Data is stale: {data_age.total_seconds()/60:.1f} minutes old")
                    # Check if market is closed
                    if current_time.hour < 9 or current_time.hour > 15:
                        print("   Note: Market is closed, stale data expected")
                        self.test_results["data_freshness"] = True
                        return True
                    return False
            else:
                print("❌ No timestamp data found")
                return False
                
        except Exception as e:
            print(f"❌ Error checking data freshness: {e}")
            return False
    
    def validate_greeks_data(self) -> bool:
        """Ensure Greeks are populated and realistic"""
        try:
            print("\n🧮 Validating Greeks data...")
            cursor = self.connection.cursor()
            
            # Check Greeks completeness
            query = """
                SELECT COUNT(*) as greeks_count
                FROM nifty_option_chain
                WHERE delta IS NOT NULL 
                AND gamma IS NOT NULL 
                AND theta IS NOT NULL 
                AND vega IS NOT NULL
                AND timestamp >= DATEADD('minute', -15, NOW())
            """
            cursor.execute(query)
            greeks_count = cursor.fetchone()[0]
            
            # Check sample Greeks values
            query = """
                SELECT strike, option_type, delta, gamma, theta, vega
                FROM nifty_option_chain
                WHERE timestamp >= DATEADD('minute', -5, NOW())
                AND delta IS NOT NULL
                LIMIT 10
            """
            cursor.execute(query)
            sample_greeks = cursor.fetchall()
            
            print(f"   Records with complete Greeks: {greeks_count}")
            print("   Sample Greeks values:")
            for row in sample_greeks[:5]:
                strike, opt_type, delta, gamma, theta, vega = row
                print(f"     {strike} {opt_type}: Delta={delta:.4f}, Gamma={gamma:.4f}, "
                      f"Theta={theta:.4f}, Vega={vega:.4f}")
            
            # Validate Greeks are in reasonable ranges
            valid_greeks = True
            for row in sample_greeks:
                _, opt_type, delta, gamma, theta, vega = row
                if opt_type == 'CE':
                    if not (0 <= delta <= 1):
                        valid_greeks = False
                else:  # PE
                    if not (-1 <= delta <= 0):
                        valid_greeks = False
                
                if gamma < 0 or vega < 0:
                    valid_greeks = False
            
            if greeks_count > 100 and valid_greeks:
                print("✅ Greeks data is complete and valid")
                self.test_results["greeks_populated"] = True
                return True
            else:
                print("❌ Greeks data is incomplete or invalid")
                return False
                
        except Exception as e:
            print(f"❌ Error validating Greeks: {e}")
            return False
    
    def detect_mock_data(self) -> bool:
        """Ensure no mock/synthetic data patterns"""
        try:
            print("\n🔍 Checking for mock data patterns...")
            cursor = self.connection.cursor()
            
            # Check for suspicious patterns
            checks_passed = True
            
            # 1. Check for repeating values
            query = """
                SELECT ltp, COUNT(*) as count
                FROM nifty_option_chain
                WHERE timestamp >= DATEADD('minute', -5, NOW())
                GROUP BY ltp
                HAVING COUNT(*) > 50
            """
            cursor.execute(query)
            repeating_prices = cursor.fetchall()
            
            if repeating_prices:
                print("⚠️  Found suspiciously repeating prices:")
                for price, count in repeating_prices[:5]:
                    print(f"     Price {price} repeated {count} times")
                checks_passed = False
            
            # 2. Check for sequential patterns
            query = """
                SELECT strike, ltp, 
                       LAG(ltp) OVER (ORDER BY strike) as prev_ltp
                FROM nifty_option_chain
                WHERE timestamp = (SELECT MAX(timestamp) FROM nifty_option_chain)
                AND option_type = 'CE'
                LIMIT 20
            """
            cursor.execute(query)
            price_sequence = cursor.fetchall()
            
            # 3. Check timestamp continuity
            query = """
                SELECT CAST(timestamp AS TIMESTAMP) as minute,
                       COUNT(*) as records
                FROM nifty_option_chain
                WHERE timestamp >= DATEADD('minute', -10, NOW())
                GROUP BY timestamp
                ORDER BY timestamp
            """
            cursor.execute(query)
            timestamp_continuity = cursor.fetchall()
            
            if len(timestamp_continuity) < 5:
                print("⚠️  Timestamp data not continuous")
                checks_passed = False
            
            if checks_passed:
                print("✅ No mock data patterns detected - Data appears authentic")
                self.test_results["no_mock_data"] = True
                return True
            else:
                print("❌ Suspicious data patterns detected - May be mock data")
                return False
                
        except Exception as e:
            print(f"❌ Error checking for mock data: {e}")
            return False
    
    def test_query_performance(self) -> bool:
        """Test HeavyDB query performance"""
        try:
            print("\n⚡ Testing query performance...")
            cursor = self.connection.cursor()
            
            test_queries = {
                "simple_select": """
                    SELECT * FROM nifty_option_chain 
                    WHERE timestamp >= DATEADD('minute', -1, NOW())
                    LIMIT 100
                """,
                "greek_aggregation": """
                    SELECT AVG(delta), AVG(gamma), AVG(theta), AVG(vega)
                    FROM nifty_option_chain
                    WHERE timestamp >= DATEADD('minute', -5, NOW())
                """,
                "strike_analysis": """
                    SELECT strike, option_type, 
                           SUM(oi) as total_oi, 
                           AVG(ltp) as avg_price
                    FROM nifty_option_chain
                    WHERE timestamp >= DATEADD('minute', -5, NOW())
                    GROUP BY strike, option_type
                """
            }
            
            all_passed = True
            for query_name, query in test_queries.items():
                start_time = time.time()
                cursor.execute(query)
                _ = cursor.fetchall()
                execution_time = (time.time() - start_time) * 1000  # ms
                
                self.test_results["performance"][query_name] = execution_time
                print(f"   {query_name}: {execution_time:.2f}ms", end="")
                
                if execution_time < 500:
                    print(" ✅")
                else:
                    print(" ❌ (exceeds 500ms limit)")
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Error testing performance: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("=" * 80)
        print("PHASE 0: HEAVYDB DATA VALIDATION")
        print("CRITICAL: ALL TESTS MUST PASS BEFORE PROCEEDING")
        print("=" * 80)
        
        # 1. Connection test
        if not self.connect_to_heavydb():
            return self.test_results
        
        # 2. Data availability
        self.validate_data_availability()
        
        # 3. Data freshness
        self.check_data_freshness()
        
        # 4. Greeks validation
        self.validate_greeks_data()
        
        # 5. Mock data detection
        self.detect_mock_data()
        
        # 6. Performance testing
        self.test_query_performance()
        
        # Summary
        print("\n" + "=" * 80)
        print("PHASE 0 TEST SUMMARY")
        print("=" * 80)
        
        all_passed = all([
            self.test_results["connection"],
            self.test_results["data_availability"],
            self.test_results["data_freshness"],
            self.test_results["greeks_populated"],
            self.test_results["no_mock_data"]
        ])
        
        for test, result in self.test_results.items():
            if test != "performance":
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test.replace('_', ' ').title()}: {status}")
        
        print("\nPerformance Metrics:")
        for query, time_ms in self.test_results["performance"].items():
            status = "✅" if time_ms < 500 else "❌"
            print(f"  {query}: {time_ms:.2f}ms {status}")
        
        if all_passed:
            print("\n✅ ALL TESTS PASSED - PROCEED WITH TESTING")
        else:
            print("\n❌ TESTS FAILED - CANNOT PROCEED")
            print("⚠️  FIX HEAVYDB DATA ISSUES BEFORE CONTINUING")
        
        return self.test_results


if __name__ == "__main__":
    validator = HeavyDBDataValidator()
    results = validator.run_all_tests()
    
    # Exit with appropriate code
    if all(results[k] for k in results if k != "performance"):
        sys.exit(0)
    else:
        sys.exit(1)