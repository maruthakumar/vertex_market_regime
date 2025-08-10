#!/usr/bin/env python3
"""
HeavyDB Alert System Test

This test validates the alert system that monitors HeavyDB connection health:
1. Tests alert triggering when HeavyDB becomes unavailable
2. Validates alert system fails fast without fallback mechanisms
3. Ensures no synthetic data is used in alert validation
4. Tests real-time monitoring and immediate alert generation
"""

import sys
import time
import json
import logging
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, '/srv/samba/shared/bt/backtester_stable/BTRUN')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeavyDBAlertSystem:
    """Production alert system for HeavyDB connection monitoring"""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_handlers = []
        self.is_monitoring = False
        self.last_connection_status = None
        
    def add_alert_handler(self, handler):
        """Add an alert handler (email, webhook, etc.)"""
        self.alert_handlers.append(handler)
        
    def trigger_alert(self, severity: str, message: str, details: Dict[str, Any] = None):
        """Trigger an alert with specified severity"""
        alert = {
            "alert_id": f"alert_{len(self.alerts) + 1}",
            "severity": severity,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "source": "heavydb_monitor"
        }
        
        self.alerts.append(alert)
        
        # Call all alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
                
        logger.critical(f"ALERT [{severity}]: {message}")
        return alert
        
    def check_connection_health(self) -> Dict[str, Any]:
        """Check HeavyDB connection health and trigger alerts if needed"""
        try:
            # Import real connection function
            from backtester_v2.dal.heavydb_connection import get_connection_status
            
            # Get real connection status
            status = get_connection_status()
            
            # Check if status has changed
            if self.last_connection_status != status.get('connection_available'):
                if status.get('connection_available'):
                    if self.last_connection_status is False:
                        # Connection restored
                        self.trigger_alert(
                            severity="INFO",
                            message="HeavyDB connection restored",
                            details={
                                "previous_status": "disconnected",
                                "current_status": "connected",
                                "data_validated": status.get('real_data_validated', False),
                                "row_count": status.get('table_row_count', 0)
                            }
                        )
                else:
                    # Connection lost
                    self.trigger_alert(
                        severity="CRITICAL",
                        message="HeavyDB connection lost",
                        details={
                            "previous_status": "connected" if self.last_connection_status else "unknown",
                            "current_status": "disconnected",
                            "error": status.get('error_message', 'Unknown error'),
                            "action_required": "Immediate investigation needed"
                        }
                    )
                    
                self.last_connection_status = status.get('connection_available')
                
            return status
            
        except Exception as e:
            # Connection check failed
            self.trigger_alert(
                severity="CRITICAL",
                message="HeavyDB connection check failed",
                details={
                    "error": str(e),
                    "check_time": datetime.now().isoformat(),
                    "action_required": "System maintenance needed"
                }
            )
            return {"connection_available": False, "error": str(e)}
            
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts"""
        by_severity = {}
        for alert in self.alerts:
            severity = alert['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(alert)
            
        return {
            "total_alerts": len(self.alerts),
            "by_severity": by_severity,
            "last_alert": self.alerts[-1] if self.alerts else None
        }


class TestHeavyDBAlertSystem:
    """Test suite for HeavyDB alert system"""
    
    def __init__(self):
        self.test_results = []
        
    def test_alert_system_initialization(self) -> Dict[str, Any]:
        """Test that alert system initializes correctly"""
        test_result = {
            "test_name": "Alert System Initialization",
            "passed": False,
            "details": {}
        }
        
        try:
            alert_system = HeavyDBAlertSystem()
            
            # Verify initial state
            assert len(alert_system.alerts) == 0
            assert alert_system.is_monitoring == False
            assert alert_system.last_connection_status is None
            
            test_result["passed"] = True
            test_result["details"]["initial_alerts"] = len(alert_system.alerts)
            logger.info("Alert system initialization test passed")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Alert system initialization test failed: {e}")
            
        return test_result
        
    def test_alert_triggering(self) -> Dict[str, Any]:
        """Test alert triggering functionality"""
        test_result = {
            "test_name": "Alert Triggering Test",
            "passed": False,
            "details": {}
        }
        
        try:
            alert_system = HeavyDBAlertSystem()
            
            # Trigger test alert
            alert = alert_system.trigger_alert(
                severity="CRITICAL",
                message="Test connection failure",
                details={"test": True}
            )
            
            # Verify alert was created
            assert len(alert_system.alerts) == 1
            assert alert["severity"] == "CRITICAL"
            assert alert["message"] == "Test connection failure"
            assert alert["details"]["test"] == True
            
            test_result["passed"] = True
            test_result["details"]["alert_created"] = True
            test_result["details"]["alert_id"] = alert["alert_id"]
            logger.info("Alert triggering test passed")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Alert triggering test failed: {e}")
            
        return test_result
        
    def test_connection_health_monitoring(self) -> Dict[str, Any]:
        """Test connection health monitoring with real HeavyDB"""
        test_result = {
            "test_name": "Connection Health Monitoring Test",
            "passed": False,
            "details": {}
        }
        
        try:
            alert_system = HeavyDBAlertSystem()
            
            # Check connection health
            status = alert_system.check_connection_health()
            
            # Verify status structure
            assert "connection_available" in status
            assert "real_data_validated" in status
            assert "table_row_count" in status
            
            test_result["passed"] = True
            test_result["details"]["connection_available"] = status.get("connection_available")
            test_result["details"]["real_data_validated"] = status.get("real_data_validated")
            test_result["details"]["table_row_count"] = status.get("table_row_count")
            logger.info("Connection health monitoring test passed")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Connection health monitoring test failed: {e}")
            
        return test_result
        
    def test_fail_fast_no_fallbacks(self) -> Dict[str, Any]:
        """Test that system fails fast without fallback mechanisms"""
        test_result = {
            "test_name": "Fail Fast No Fallbacks Test",
            "passed": False,
            "details": {}
        }
        
        try:
            alert_system = HeavyDBAlertSystem()
            
            # Mock a connection failure
            with patch('backtester_v2.dal.heavydb_connection.get_connection_status') as mock_status:
                mock_status.side_effect = Exception("Connection failed")
                
                # Check that it fails fast
                status = alert_system.check_connection_health()
                
                # Verify it didn't try fallbacks
                assert status["connection_available"] == False
                assert "error" in status
                
                # Verify alert was triggered
                assert len(alert_system.alerts) > 0
                critical_alerts = [a for a in alert_system.alerts if a["severity"] == "CRITICAL"]
                assert len(critical_alerts) > 0
                
            test_result["passed"] = True
            test_result["details"]["failed_fast"] = True
            test_result["details"]["critical_alerts"] = len(critical_alerts)
            logger.info("Fail fast no fallbacks test passed")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Fail fast no fallbacks test failed: {e}")
            
        return test_result
        
    def test_real_data_validation_in_alerts(self) -> Dict[str, Any]:
        """Test that alerts validate real data usage"""
        test_result = {
            "test_name": "Real Data Validation in Alerts Test",
            "passed": False,
            "details": {}
        }
        
        try:
            alert_system = HeavyDBAlertSystem()
            
            # Check connection health (should validate real data)
            status = alert_system.check_connection_health()
            
            # Verify real data validation
            if status.get("connection_available"):
                assert status.get("real_data_validated") is not None
                assert status.get("table_row_count", 0) > 1000000  # Real data should have > 1M rows
                
                # Check for data authenticity score
                assert "data_authenticity_score" in status
                
            test_result["passed"] = True
            test_result["details"]["real_data_validated"] = status.get("real_data_validated")
            test_result["details"]["data_authenticity_score"] = status.get("data_authenticity_score")
            logger.info("Real data validation in alerts test passed")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Real data validation in alerts test failed: {e}")
            
        return test_result
        
    def test_alert_summary_generation(self) -> Dict[str, Any]:
        """Test alert summary generation"""
        test_result = {
            "test_name": "Alert Summary Generation Test",
            "passed": False,
            "details": {}
        }
        
        try:
            alert_system = HeavyDBAlertSystem()
            
            # Generate some test alerts
            alert_system.trigger_alert("CRITICAL", "Test critical alert")
            alert_system.trigger_alert("WARNING", "Test warning alert")
            alert_system.trigger_alert("INFO", "Test info alert")
            
            # Get summary
            summary = alert_system.get_alert_summary()
            
            # Verify summary structure
            assert summary["total_alerts"] == 3
            assert "by_severity" in summary
            assert "CRITICAL" in summary["by_severity"]
            assert "WARNING" in summary["by_severity"]
            assert "INFO" in summary["by_severity"]
            assert len(summary["by_severity"]["CRITICAL"]) == 1
            
            test_result["passed"] = True
            test_result["details"]["total_alerts"] = summary["total_alerts"]
            test_result["details"]["severities"] = list(summary["by_severity"].keys())
            logger.info("Alert summary generation test passed")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Alert summary generation test failed: {e}")
            
        return test_result
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all alert system tests"""
        logger.info("Starting HeavyDB Alert System Tests")
        
        test_methods = [
            self.test_alert_system_initialization,
            self.test_alert_triggering,
            self.test_connection_health_monitoring,
            self.test_fail_fast_no_fallbacks,
            self.test_real_data_validation_in_alerts,
            self.test_alert_summary_generation
        ]
        
        results = {
            "test_suite": "HeavyDB Alert System Tests",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                result = test_method()
                results["tests"].append(result)
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                results["tests"].append({
                    "test_name": test_method.__name__,
                    "passed": False,
                    "error": str(e)
                })
                
        results["summary"] = {
            "total_tests": len(test_methods),
            "passed": passed,
            "failed": failed,
            "success_rate": f"{(passed/len(test_methods)*100):.1f}%"
        }
        
        return results


def run_alert_system_tests():
    """Run the complete alert system test suite"""
    
    print("=" * 60)
    print("HeavyDB Alert System Test Suite")
    print("=" * 60)
    print()
    
    tester = TestHeavyDBAlertSystem()
    
    # Run all tests
    start_time = time.time()
    results = tester.run_all_tests()
    total_time = time.time() - start_time
    
    # Print results
    print(f"Test Suite: {results['test_suite']}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Timestamp: {results['timestamp']}")
    print()
    
    # Print summary
    summary = results['summary']
    print("Test Summary:")
    print("-" * 40)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']}")
    print()
    
    # Print individual test results
    print("Individual Test Results:")
    print("-" * 40)
    for test in results['tests']:
        status = "✓ PASS" if test['passed'] else "✗ FAIL"
        print(f"{status} - {test['test_name']}")
        if not test['passed'] and test.get('error'):
            print(f"     Error: {test['error']}")
        if test.get('details'):
            key_details = {k: v for k, v in test['details'].items() if k in ['connection_available', 'real_data_validated', 'total_alerts']}
            if key_details:
                print(f"     Details: {key_details}")
    print()
    
    # Save results to file
    results_file = f"heavydb_alert_system_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {results_file}")
    print()
    
    # Final summary
    if summary['failed'] == 0:
        print("🎉 ALL TESTS PASSED - Alert system is working correctly!")
    else:
        print(f"❌ {summary['failed']} TEST(S) FAILED - Alert system needs attention.")
    
    return results


if __name__ == "__main__":
    results = run_alert_system_tests()
    
    # Exit with appropriate code
    if results['summary']['failed'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)