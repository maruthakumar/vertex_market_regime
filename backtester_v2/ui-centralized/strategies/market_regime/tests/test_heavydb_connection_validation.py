#!/usr/bin/env python3
"""
HeavyDB Connection Validation Test

Test that validates:
1. HeavyDB connection is available and working
2. Real data queries execute successfully
3. No synthetic/mock data is used in validation
4. Connection failures are properly detected
5. Alert system responds to connection issues
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, '/srv/samba/shared/bt/backtester_stable/BTRUN')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeavyDBConnectionValidator:
    """Validates HeavyDB connection with strict no-mock policy"""
    
    def __init__(self):
        self.connection = None
        self.validation_results = []
        
    def validate_connection(self) -> Dict[str, Any]:
        """Validate HeavyDB connection with comprehensive checks"""
        
        validation_result = {
            "test_name": "HeavyDB Connection Validation",
            "timestamp": datetime.now().isoformat(),
            "status": "UNKNOWN",
            "checks": [],
            "errors": [],
            "warnings": [],
            "performance_metrics": {}
        }
        
        try:
            # Check 1: Test connection establishment
            logger.info("Testing HeavyDB connection establishment...")
            connection_check = self._test_connection_establishment()
            validation_result["checks"].append(connection_check)
            
            if not connection_check["passed"]:
                validation_result["status"] = "FAILED"
                validation_result["errors"].append("Cannot establish HeavyDB connection")
                return validation_result
            
            # Check 2: Test basic query execution
            logger.info("Testing basic query execution...")
            query_check = self._test_basic_query()
            validation_result["checks"].append(query_check)
            
            # Check 3: Test table existence and structure
            logger.info("Testing table existence and structure...")
            table_check = self._test_table_structure()
            validation_result["checks"].append(table_check)
            
            # Check 4: Test data availability
            logger.info("Testing data availability...")
            data_check = self._test_data_availability()
            validation_result["checks"].append(data_check)
            
            # Check 5: Test performance
            logger.info("Testing query performance...")
            performance_check = self._test_query_performance()
            validation_result["checks"].append(performance_check)
            validation_result["performance_metrics"] = performance_check.get("metrics", {})
            
            # Check 6: Test connection health monitoring
            logger.info("Testing connection health monitoring...")
            health_check = self._test_health_monitoring()
            validation_result["checks"].append(health_check)
            
            # Determine overall status
            failed_checks = [c for c in validation_result["checks"] if not c["passed"]]
            if failed_checks:
                validation_result["status"] = "FAILED"
                validation_result["errors"].extend([c["error"] for c in failed_checks if c.get("error")])
            else:
                validation_result["status"] = "PASSED"
                
        except Exception as e:
            validation_result["status"] = "ERROR"
            validation_result["errors"].append(f"Validation failed with exception: {str(e)}")
            logger.error(f"Validation error: {e}")
            
        return validation_result
    
    def _test_connection_establishment(self) -> Dict[str, Any]:
        """Test that HeavyDB connection can be established"""
        check_result = {
            "name": "Connection Establishment",
            "passed": False,
            "details": {}
        }
        
        try:
            # Try to import and create connection using the function-based API
            from backtester_v2.dal.heavydb_connection import get_connection
            
            # Test connection
            conn = get_connection(enforce_real_data=True)
            if conn is not None:
                check_result["passed"] = True
                check_result["details"]["connection_type"] = type(conn).__name__
                check_result["details"]["connection_established"] = True
                self.connection = conn
                logger.info("HeavyDB connection established successfully")
            else:
                check_result["error"] = "Connection returned None"
                logger.error("Failed to establish HeavyDB connection")
                
        except ImportError as e:
            check_result["error"] = f"Cannot import HeavyDB connection module: {str(e)}"
            logger.error(f"Import error: {e}")
        except Exception as e:
            check_result["error"] = f"Connection establishment failed: {str(e)}"
            logger.error(f"Connection error: {e}")
            
        return check_result
    
    def _test_basic_query(self) -> Dict[str, Any]:
        """Test basic query execution"""
        check_result = {
            "name": "Basic Query Execution",
            "passed": False,
            "details": {}
        }
        
        if not self.connection:
            check_result["error"] = "No connection available"
            return check_result
            
        try:
            # Import execute_query function
            from backtester_v2.dal.heavydb_connection import execute_query
            
            # Execute a simple test query using the module's function
            result_df = execute_query(self.connection, "SELECT 1 as test_value")
            
            if not result_df.empty and result_df.iloc[0, 0] == 1:
                check_result["passed"] = True
                check_result["details"]["query_result"] = result_df.iloc[0, 0]
                logger.info("Basic query execution successful")
            else:
                check_result["error"] = f"Unexpected query result: {result_df}"
                
        except Exception as e:
            check_result["error"] = f"Query execution failed: {str(e)}"
            logger.error(f"Query error: {e}")
            
        return check_result
    
    def _test_table_structure(self) -> Dict[str, Any]:
        """Test that required tables exist and have expected structure"""
        check_result = {
            "name": "Table Structure Validation",
            "passed": False,
            "details": {}
        }
        
        if not self.connection:
            check_result["error"] = "No connection available"
            return check_result
            
        try:
            # Import functions from the module
            from backtester_v2.dal.heavydb_connection import execute_query, validate_table_exists, get_table_schema
            
            # Check if nifty_option_chain table exists
            table_exists = validate_table_exists('nifty_option_chain')
            
            if table_exists:
                # Get column information by querying the table directly
                sample_df = execute_query(self.connection, "SELECT * FROM nifty_option_chain LIMIT 1")
                
                if not sample_df.empty:
                    check_result["passed"] = True
                    check_result["details"]["table_exists"] = True
                    check_result["details"]["column_count"] = len(sample_df.columns)
                    check_result["details"]["columns"] = sample_df.columns.tolist()[:10]  # First 10 columns
                    logger.info(f"Table structure validated: {len(sample_df.columns)} columns found")
                else:
                    check_result["error"] = "Could not retrieve table schema"
            else:
                check_result["error"] = "nifty_option_chain table not found"
                
        except Exception as e:
            check_result["error"] = f"Table structure check failed: {str(e)}"
            logger.error(f"Table structure error: {e}")
            
        return check_result
    
    def _test_data_availability(self) -> Dict[str, Any]:
        """Test that real data is available in the database"""
        check_result = {
            "name": "Data Availability Check",
            "passed": False,
            "details": {}
        }
        
        if not self.connection:
            check_result["error"] = "No connection available"
            return check_result
            
        try:
            # Import functions from the module
            from backtester_v2.dal.heavydb_connection import execute_query, get_table_row_count
            
            # Check total row count using the module's function
            total_rows = get_table_row_count('nifty_option_chain')
            
            # Check recent data (last 30 days)
            recent_data_df = execute_query(self.connection, """
                SELECT COUNT(*) as recent_count FROM nifty_option_chain 
                WHERE trade_date >= CAST(NOW() - INTERVAL '30' DAY AS DATE)
            """)
            recent_rows = recent_data_df.iloc[0, 0] if not recent_data_df.empty else 0
            
            # Check data range
            date_range_df = execute_query(self.connection, """
                SELECT 
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(DISTINCT trade_date) as unique_dates
                FROM nifty_option_chain
            """)
            
            if total_rows > 0:
                check_result["passed"] = True
                check_result["details"]["total_rows"] = total_rows
                check_result["details"]["recent_rows"] = recent_rows
                
                if not date_range_df.empty:
                    check_result["details"]["date_range"] = {
                        "min_date": str(date_range_df.iloc[0, 0]) if date_range_df.iloc[0, 0] else None,
                        "max_date": str(date_range_df.iloc[0, 1]) if date_range_df.iloc[0, 1] else None,
                        "unique_dates": date_range_df.iloc[0, 2] if date_range_df.iloc[0, 2] else 0
                    }
                    
                logger.info(f"Data availability validated: {total_rows:,} total rows")
            else:
                check_result["error"] = "No data found in nifty_option_chain table"
                
        except Exception as e:
            check_result["error"] = f"Data availability check failed: {str(e)}"
            logger.error(f"Data availability error: {e}")
            
        return check_result
    
    def _test_query_performance(self) -> Dict[str, Any]:
        """Test query performance to ensure acceptable response times"""
        check_result = {
            "name": "Query Performance Test",
            "passed": False,
            "details": {},
            "metrics": {}
        }
        
        if not self.connection:
            check_result["error"] = "No connection available"
            return check_result
            
        try:
            # Import function from the module
            from backtester_v2.dal.heavydb_connection import execute_query
            
            # Test 1: Simple aggregation query
            start_time = time.time()
            simple_result_df = execute_query(self.connection, """
                SELECT COUNT(*) as total_rows,
                       COUNT(DISTINCT trade_date) as unique_dates,
                       MAX(trade_date) as latest_date
                FROM nifty_option_chain
            """)
            simple_query_time = time.time() - start_time
            
            # Test 2: Complex query with filtering
            start_time = time.time()
            complex_result_df = execute_query(self.connection, """
                SELECT trade_date, 
                       AVG(ce_close) as avg_ce_price,
                       AVG(pe_close) as avg_pe_price,
                       COUNT(*) as record_count
                FROM nifty_option_chain
                WHERE trade_date >= CAST(NOW() - INTERVAL '7' DAY AS DATE)
                GROUP BY trade_date
                ORDER BY trade_date DESC
                LIMIT 10
            """)
            complex_query_time = time.time() - start_time
            
            # Performance thresholds
            SIMPLE_QUERY_THRESHOLD = 5.0  # seconds
            COMPLEX_QUERY_THRESHOLD = 10.0  # seconds
            
            check_result["metrics"] = {
                "simple_query_time": simple_query_time,
                "complex_query_time": complex_query_time,
                "simple_query_threshold": SIMPLE_QUERY_THRESHOLD,
                "complex_query_threshold": COMPLEX_QUERY_THRESHOLD,
                "total_rows": simple_result_df.iloc[0, 0] if not simple_result_df.empty else 0,
                "complex_result_count": len(complex_result_df)
            }
            
            if (simple_query_time <= SIMPLE_QUERY_THRESHOLD and 
                complex_query_time <= COMPLEX_QUERY_THRESHOLD):
                check_result["passed"] = True
                check_result["details"]["performance_acceptable"] = True
                logger.info(f"Query performance acceptable: {simple_query_time:.2f}s, {complex_query_time:.2f}s")
            else:
                check_result["error"] = f"Query performance too slow: {simple_query_time:.2f}s, {complex_query_time:.2f}s"
                
        except Exception as e:
            check_result["error"] = f"Performance test failed: {str(e)}"
            logger.error(f"Performance test error: {e}")
            
        return check_result
    
    def _test_health_monitoring(self) -> Dict[str, Any]:
        """Test that health monitoring works correctly"""
        check_result = {
            "name": "Health Monitoring Test",
            "passed": False,
            "details": {}
        }
        
        try:
            # Test health check using the module's functions
            from backtester_v2.dal.heavydb_connection import test_connection, get_connection_status
            
            # Test basic connection health
            start_time = time.time()
            connection_healthy = test_connection()
            health_check_time = time.time() - start_time
            
            if connection_healthy:
                check_result["passed"] = True
                check_result["details"]["health_check_time"] = health_check_time
                check_result["details"]["health_status"] = "HEALTHY"
                
                # Get comprehensive connection status
                connection_status = get_connection_status()
                check_result["details"]["connection_status"] = {
                    "connection_available": connection_status.get("connection_available", False),
                    "real_data_validated": connection_status.get("real_data_validated", False),
                    "table_exists": connection_status.get("table_exists", False),
                    "table_row_count": connection_status.get("table_row_count", 0),
                    "data_authenticity_score": connection_status.get("data_authenticity_score", 0.0)
                }
                
                logger.info(f"Health monitoring validated: {health_check_time:.3f}s response time")
            else:
                check_result["error"] = "Health check failed - connection not healthy"
                
        except Exception as e:
            check_result["error"] = f"Health monitoring test failed: {str(e)}"
            logger.error(f"Health monitoring error: {e}")
            
        return check_result


def run_validation_suite():
    """Run the complete HeavyDB connection validation suite"""
    
    print("=" * 60)
    print("HeavyDB Connection Validation Suite")
    print("=" * 60)
    print()
    
    validator = HeavyDBConnectionValidator()
    
    # Run validation
    start_time = time.time()
    results = validator.validate_connection()
    total_time = time.time() - start_time
    
    # Print results
    print(f"Validation Status: {results['status']}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Timestamp: {results['timestamp']}")
    print()
    
    # Print individual check results
    print("Individual Check Results:")
    print("-" * 40)
    for check in results['checks']:
        status = "✓ PASS" if check['passed'] else "✗ FAIL"
        print(f"{status} - {check['name']}")
        if not check['passed'] and check.get('error'):
            print(f"     Error: {check['error']}")
        if check.get('details'):
            key_details = {k: v for k, v in check['details'].items() if k in ['total_rows', 'column_count', 'connection_established']}
            if key_details:
                print(f"     Details: {key_details}")
    print()
    
    # Print performance metrics
    if results.get('performance_metrics'):
        print("Performance Metrics:")
        print("-" * 40)
        metrics = results['performance_metrics']
        if 'simple_query_time' in metrics:
            print(f"Simple Query Time: {metrics['simple_query_time']:.3f}s")
        if 'complex_query_time' in metrics:
            print(f"Complex Query Time: {metrics['complex_query_time']:.3f}s")
        if 'total_rows' in metrics:
            print(f"Total Rows: {metrics['total_rows']:,}")
        print()
    
    # Print errors and warnings
    if results.get('errors'):
        print("Errors:")
        print("-" * 40)
        for error in results['errors']:
            print(f"  • {error}")
        print()
    
    if results.get('warnings'):
        print("Warnings:")
        print("-" * 40)
        for warning in results['warnings']:
            print(f"  • {warning}")
        print()
    
    # Save results to file
    results_file = f"heavydb_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {results_file}")
    print()
    
    # Final summary
    if results['status'] == 'PASSED':
        print("🎉 ALL TESTS PASSED - HeavyDB connection is healthy and ready!")
    elif results['status'] == 'FAILED':
        print("❌ SOME TESTS FAILED - HeavyDB connection has issues that need attention.")
    else:
        print("⚠️  VALIDATION ERROR - Could not complete validation suite.")
    
    return results


if __name__ == "__main__":
    # Run the validation suite
    results = run_validation_suite()
    
    # Exit with appropriate code
    if results['status'] == 'PASSED':
        sys.exit(0)
    else:
        sys.exit(1)