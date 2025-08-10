#!/usr/bin/env python3
"""
Complete HeavyDB Monitoring System Integration Test

This test demonstrates the complete monitoring system working together:
1. Real-time connection health monitoring
2. Immediate alert generation on failures
3. Comprehensive status reporting
4. Performance metrics tracking
5. Data authenticity validation

This is the final validation that the system is production-ready.
"""

import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '/srv/samba/shared/bt/backtester_stable/BTRUN')

from backtester_v2.dal.heavydb_connection import (
    get_connection, get_connection_status, test_connection,
    get_table_row_count, execute_query
)


class ProductionHeavyDBMonitor:
    """
    Production-ready HeavyDB monitoring system that demonstrates
    all validated capabilities working together.
    """
    
    def __init__(self):
        self.is_running = False
        self.monitor_thread = None
        self.status_history = []
        self.alerts = []
        self.performance_metrics = []
        
    def start_monitoring(self, duration_seconds: int = 10):
        """Start the monitoring system for a specified duration"""
        print(f"🚀 Starting HeavyDB Production Monitor (Duration: {duration_seconds}s)")
        print("-" * 60)
        
        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(duration_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("🔴 HeavyDB Production Monitor stopped")
        
    def _monitor_loop(self, duration_seconds: int):
        """Main monitoring loop with comprehensive checks"""
        start_time = time.time()
        check_count = 0
        
        while self.is_running and (time.time() - start_time) < duration_seconds:
            check_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            try:
                # 1. Connection Health Check
                print(f"[{timestamp}] Check #{check_count}: Connection Health...")
                health_start = time.time()
                connection_healthy = test_connection()
                health_time = time.time() - health_start
                
                # 2. Comprehensive Status Check
                print(f"[{timestamp}] Check #{check_count}: Status Validation...")
                status_start = time.time()
                status = get_connection_status()
                status_time = time.time() - status_start
                
                # 3. Performance Metrics
                print(f"[{timestamp}] Check #{check_count}: Performance Test...")
                perf_start = time.time()
                conn = get_connection()
                if conn:
                    # Quick performance test
                    result = execute_query(conn, "SELECT COUNT(*) FROM nifty_option_chain LIMIT 1")
                    perf_time = time.time() - perf_start
                else:
                    perf_time = -1
                
                # 4. Record Status
                status_record = {
                    "timestamp": timestamp,
                    "check_number": check_count,
                    "connection_healthy": connection_healthy,
                    "status": status,
                    "performance_metrics": {
                        "health_check_time": health_time,
                        "status_check_time": status_time,
                        "query_performance_time": perf_time
                    }
                }
                
                self.status_history.append(status_record)
                
                # 5. Display Results
                self._display_check_results(status_record)
                
                # 6. Alert Generation
                self._check_for_alerts(status_record)
                
            except Exception as e:
                error_record = {
                    "timestamp": timestamp,
                    "check_number": check_count,
                    "error": str(e),
                    "connection_healthy": False
                }
                self.status_history.append(error_record)
                print(f"[{timestamp}] ❌ ERROR: {str(e)}")
                
                # Generate critical alert
                self.alerts.append({
                    "severity": "CRITICAL",
                    "message": f"Monitoring check #{check_count} failed",
                    "error": str(e),
                    "timestamp": timestamp
                })
                
            # Wait before next check
            time.sleep(2)
            
        print(f"\\n✅ Monitoring completed: {check_count} checks performed")
        
    def _display_check_results(self, status_record: Dict[str, Any]):
        """Display formatted check results"""
        timestamp = status_record["timestamp"]
        check_num = status_record["check_number"]
        
        if status_record["connection_healthy"]:
            status_icon = "✅"
            status_text = "HEALTHY"
        else:
            status_icon = "❌"
            status_text = "UNHEALTHY"
            
        status = status_record.get("status", {})
        perf = status_record.get("performance_metrics", {})
        
        print(f"[{timestamp}] {status_icon} Check #{check_num}: {status_text}")
        print(f"              Connection: {status.get('connection_available', 'Unknown')}")
        print(f"              Real Data: {status.get('real_data_validated', 'Unknown')}")
        print(f"              Row Count: {status.get('table_row_count', 'Unknown'):,}")
        print(f"              Health Time: {perf.get('health_check_time', 0):.3f}s")
        print(f"              Query Time: {perf.get('query_performance_time', 0):.3f}s")
        print()
        
    def _check_for_alerts(self, status_record: Dict[str, Any]):
        """Check for conditions that require alerts"""
        timestamp = status_record["timestamp"]
        
        # Check for connection issues
        if not status_record["connection_healthy"]:
            self.alerts.append({
                "severity": "CRITICAL",
                "message": "HeavyDB connection unhealthy",
                "timestamp": timestamp,
                "details": status_record
            })
            
        # Check for performance issues
        perf = status_record.get("performance_metrics", {})
        if perf.get("health_check_time", 0) > 1.0:  # > 1 second
            self.alerts.append({
                "severity": "WARNING",
                "message": "Slow health check response",
                "timestamp": timestamp,
                "response_time": perf.get("health_check_time", 0)
            })
            
        # Check for data validation issues
        status = status_record.get("status", {})
        if not status.get("real_data_validated", False):
            self.alerts.append({
                "severity": "CRITICAL",
                "message": "Real data validation failed",
                "timestamp": timestamp,
                "details": status
            })
            
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        if not self.status_history:
            return {"error": "No monitoring data available"}
            
        total_checks = len(self.status_history)
        healthy_checks = sum(1 for s in self.status_history if s.get("connection_healthy", False))
        
        # Calculate average performance
        health_times = [s.get("performance_metrics", {}).get("health_check_time", 0) 
                       for s in self.status_history if "performance_metrics" in s]
        avg_health_time = sum(health_times) / len(health_times) if health_times else 0
        
        # Alert summary
        alerts_by_severity = {}
        for alert in self.alerts:
            severity = alert["severity"]
            if severity not in alerts_by_severity:
                alerts_by_severity[severity] = []
            alerts_by_severity[severity].append(alert)
            
        return {
            "monitoring_summary": {
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "unhealthy_checks": total_checks - healthy_checks,
                "health_percentage": f"{(healthy_checks/total_checks*100):.1f}%" if total_checks > 0 else "0%",
                "average_health_check_time": f"{avg_health_time:.3f}s",
                "total_alerts": len(self.alerts),
                "alerts_by_severity": alerts_by_severity
            },
            "latest_status": self.status_history[-1] if self.status_history else None,
            "performance_trend": health_times[-5:] if len(health_times) >= 5 else health_times
        }


def run_complete_monitoring_test():
    """Run the complete monitoring system test"""
    
    print("=" * 70)
    print("🔍 HeavyDB Complete Monitoring System Integration Test")
    print("=" * 70)
    print()
    
    print("This test demonstrates:")
    print("✅ Real-time connection health monitoring")
    print("✅ Immediate alert generation on failures")
    print("✅ Comprehensive status reporting")
    print("✅ Performance metrics tracking")
    print("✅ Data authenticity validation")
    print()
    
    # Create and start monitor
    monitor = ProductionHeavyDBMonitor()
    
    try:
        # Run monitoring for 10 seconds
        monitor.start_monitoring(duration_seconds=10)
        
        # Wait for monitoring to complete
        if monitor.monitor_thread:
            monitor.monitor_thread.join()
            
        # Get comprehensive summary
        summary = monitor.get_monitoring_summary()
        
        print("=" * 70)
        print("📊 MONITORING SUMMARY")
        print("=" * 70)
        
        monitoring_data = summary["monitoring_summary"]
        
        print(f"Total Checks Performed: {monitoring_data['total_checks']}")
        print(f"Healthy Checks: {monitoring_data['healthy_checks']}")
        print(f"Unhealthy Checks: {monitoring_data['unhealthy_checks']}")
        print(f"Health Percentage: {monitoring_data['health_percentage']}")
        print(f"Average Health Check Time: {monitoring_data['average_health_check_time']}")
        print(f"Total Alerts Generated: {monitoring_data['total_alerts']}")
        print()
        
        # Display alerts
        if monitoring_data['alerts_by_severity']:
            print("🚨 ALERTS GENERATED:")
            for severity, alerts in monitoring_data['alerts_by_severity'].items():
                print(f"  {severity}: {len(alerts)} alerts")
                for alert in alerts[:3]:  # Show first 3 alerts
                    print(f"    - {alert['message']} at {alert['timestamp']}")
            print()
        else:
            print("✅ NO ALERTS GENERATED - System operating normally")
            print()
            
        # Display latest status
        if summary["latest_status"]:
            latest = summary["latest_status"]
            print("📈 LATEST STATUS:")
            print(f"  Connection Healthy: {latest.get('connection_healthy', 'Unknown')}")
            if "status" in latest:
                status = latest["status"]
                print(f"  Connection Available: {status.get('connection_available', 'Unknown')}")
                print(f"  Real Data Validated: {status.get('real_data_validated', 'Unknown')}")
                print(f"  Table Row Count: {status.get('table_row_count', 'Unknown'):,}")
                print(f"  Data Authenticity Score: {status.get('data_authenticity_score', 'Unknown')}")
            print()
            
        # Performance trend
        if summary["performance_trend"]:
            print("📊 PERFORMANCE TREND (Last 5 checks):")
            for i, time_val in enumerate(summary["performance_trend"]):
                print(f"  Check {i+1}: {time_val:.3f}s")
            print()
            
        # Save detailed results
        results_file = f"complete_monitoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print(f"📁 Detailed results saved to: {results_file}")
        print()
        
        # Final assessment
        health_pct = float(monitoring_data['health_percentage'].rstrip('%'))
        total_alerts = monitoring_data['total_alerts']
        
        if health_pct >= 95 and total_alerts == 0:
            print("🎉 SYSTEM STATUS: EXCELLENT")
            print("   - Health percentage >= 95%")
            print("   - No alerts generated")
            print("   - All monitoring checks passed")
            return True
        elif health_pct >= 90:
            print("✅ SYSTEM STATUS: GOOD")
            print("   - Health percentage >= 90%")
            print("   - Minor issues detected")
            return True
        else:
            print("⚠️  SYSTEM STATUS: NEEDS ATTENTION")
            print("   - Health percentage < 90%")
            print("   - Multiple issues detected")
            return False
            
    except Exception as e:
        print(f"❌ MONITORING TEST FAILED: {str(e)}")
        return False
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    success = run_complete_monitoring_test()
    sys.exit(0 if success else 1)