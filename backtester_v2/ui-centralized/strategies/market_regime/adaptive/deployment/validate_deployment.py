#!/usr/bin/env python3
"""
Production Deployment Validation Script

This script automates the validation of the Adaptive Market Regime System deployment.
Run this after deployment to ensure everything is working correctly.
"""

import os
import sys
import json
import time
import requests
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentValidator:
    """Validates production deployment of the Adaptive Market Regime System"""
    
    def __init__(self, config_path: str = "/opt/adaptive_regime/config/production.yaml"):
        self.config_path = config_path
        self.base_url = "http://localhost:8080"
        self.results = {}
        self.start_time = datetime.now()
        
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        logger.info("="*60)
        logger.info("ADAPTIVE MARKET REGIME SYSTEM - DEPLOYMENT VALIDATION")
        logger.info("="*60)
        logger.info(f"Start Time: {self.start_time}")
        logger.info("")
        
        # Run all validation checks
        validations = [
            ("System Requirements", self.validate_system_requirements),
            ("Service Status", self.validate_service_status),
            ("API Health", self.validate_api_health),
            ("Database Connection", self.validate_database_connection),
            ("Component Status", self.validate_component_status),
            ("Performance Metrics", self.validate_performance),
            ("Security Configuration", self.validate_security),
            ("Monitoring Setup", self.validate_monitoring),
            ("Log Files", self.validate_logging),
            ("Backup Configuration", self.validate_backups)
        ]
        
        all_passed = True
        
        for name, validator in validations:
            logger.info(f"\n{name}")
            logger.info("-" * len(name))
            
            try:
                passed, details = validator()
                self.results[name] = {
                    "passed": passed,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                }
                
                if passed:
                    logger.info(f"✅ {name}: PASSED")
                else:
                    logger.error(f"❌ {name}: FAILED")
                    all_passed = False
                    
                # Log details
                for detail in details:
                    logger.info(f"  {detail}")
                    
            except Exception as e:
                logger.error(f"❌ {name}: ERROR - {str(e)}")
                self.results[name] = {
                    "passed": False,
                    "details": [f"Error: {str(e)}"],
                    "timestamp": datetime.now().isoformat()
                }
                all_passed = False
        
        # Generate summary
        self.generate_summary(all_passed)
        
        return all_passed
    
    def validate_system_requirements(self) -> Tuple[bool, List[str]]:
        """Validate system requirements"""
        details = []
        passed = True
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count >= 8:
            details.append(f"CPU Cores: {cpu_count} (minimum 8 required)")
        else:
            details.append(f"CPU Cores: {cpu_count} (INSUFFICIENT - minimum 8 required)")
            passed = False
        
        # Check Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 16:
            details.append(f"Memory: {memory_gb:.1f}GB (minimum 16GB required)")
        else:
            details.append(f"Memory: {memory_gb:.1f}GB (INSUFFICIENT - minimum 16GB required)")
            passed = False
        
        # Check Disk Space
        disk = psutil.disk_usage('/')
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 50:
            details.append(f"Free Disk Space: {disk_gb:.1f}GB (minimum 50GB required)")
        else:
            details.append(f"Free Disk Space: {disk_gb:.1f}GB (INSUFFICIENT - minimum 50GB required)")
            passed = False
        
        # Check Python Version
        python_version = sys.version.split()[0]
        if sys.version_info >= (3, 8):
            details.append(f"Python Version: {python_version}")
        else:
            details.append(f"Python Version: {python_version} (INSUFFICIENT - minimum 3.8 required)")
            passed = False
        
        # Check GPU (optional but recommended)
        try:
            nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True)
            if nvidia_smi.returncode == 0:
                gpu_name = nvidia_smi.stdout.strip()
                details.append(f"GPU: {gpu_name} detected")
            else:
                details.append("GPU: Not detected (optional but recommended)")
        except:
            details.append("GPU: nvidia-smi not available")
        
        return passed, details
    
    def validate_service_status(self) -> Tuple[bool, List[str]]:
        """Validate systemd service status"""
        details = []
        passed = True
        
        try:
            # Check service status
            result = subprocess.run(
                ['systemctl', 'is-active', 'adaptive_regime'],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip() == 'active':
                details.append("Service Status: Active")
                
                # Get service details
                result = subprocess.run(
                    ['systemctl', 'status', 'adaptive_regime', '--no-pager'],
                    capture_output=True,
                    text=True
                )
                
                # Extract key information
                for line in result.stdout.split('\n'):
                    if 'Active:' in line:
                        if 'active (running)' in line:
                            details.append("Service State: Running")
                        else:
                            details.append(f"Service State: {line.strip()}")
                            passed = False
                    elif 'Main PID:' in line:
                        details.append(f"Process {line.strip()}")
                    elif 'Memory:' in line:
                        details.append(f"Memory Usage: {line.strip().split(':')[1].strip()}")
            else:
                details.append(f"Service Status: {result.stdout.strip()} (NOT ACTIVE)")
                passed = False
                
        except Exception as e:
            details.append(f"Failed to check service status: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_api_health(self) -> Tuple[bool, List[str]]:
        """Validate API health endpoint"""
        details = []
        passed = True
        
        try:
            # Check health endpoint
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            
            if response.status_code == 200:
                details.append(f"Health Endpoint: Responding (status {response.status_code})")
                
                health_data = response.json()
                details.append(f"System Status: {health_data.get('status', 'unknown')}")
                details.append(f"Version: {health_data.get('version', 'unknown')}")
                details.append(f"Uptime: {health_data.get('uptime_seconds', 0)}s")
                
                if health_data.get('status') != 'healthy':
                    passed = False
            else:
                details.append(f"Health Endpoint: Failed (status {response.status_code})")
                passed = False
                
            # Check response time
            response_time = response.elapsed.total_seconds() * 1000
            if response_time < 100:
                details.append(f"Response Time: {response_time:.1f}ms")
            else:
                details.append(f"Response Time: {response_time:.1f}ms (SLOW - should be <100ms)")
                passed = False
                
        except requests.exceptions.ConnectionError:
            details.append("API Connection: Failed - service may not be running")
            passed = False
        except Exception as e:
            details.append(f"API Check Failed: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_database_connection(self) -> Tuple[bool, List[str]]:
        """Validate database connectivity"""
        details = []
        passed = True
        
        try:
            # Check database status through API
            response = requests.get(f"{self.base_url}/api/v1/system/database", timeout=5)
            
            if response.status_code == 200:
                db_data = response.json()
                details.append(f"Database Connection: {db_data.get('status', 'unknown')}")
                details.append(f"Database Type: {db_data.get('type', 'HeavyDB')}")
                details.append(f"Connection Pool: {db_data.get('active_connections', 0)}/{db_data.get('pool_size', 10)}")
                
                if db_data.get('status') != 'connected':
                    passed = False
            else:
                details.append("Database Status: Unable to verify")
                passed = False
                
        except Exception as e:
            details.append(f"Database Check Failed: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_component_status(self) -> Tuple[bool, List[str]]:
        """Validate all system components"""
        details = []
        passed = True
        
        try:
            response = requests.get(f"{self.base_url}/api/v1/components", timeout=5)
            
            if response.status_code == 200:
                components = response.json()
                
                critical_components = [
                    'base_regime_detector',
                    'adaptive_scoring_layer',
                    'integration_orchestrator',
                    'performance_feedback_system'
                ]
                
                for comp_name, comp_status in components.items():
                    status = comp_status.get('state', 'unknown')
                    if status == 'running':
                        details.append(f"{comp_name}: ✅ Running")
                    else:
                        details.append(f"{comp_name}: ❌ {status}")
                        if comp_name in critical_components:
                            passed = False
            else:
                details.append("Unable to retrieve component status")
                passed = False
                
        except Exception as e:
            details.append(f"Component Check Failed: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_performance(self) -> Tuple[bool, List[str]]:
        """Validate performance metrics"""
        details = []
        passed = True
        
        try:
            # Run a simple performance test
            start_time = time.time()
            
            # Make 10 requests to measure average latency
            latencies = []
            for _ in range(10):
                response = requests.get(f"{self.base_url}/api/v1/regime/current", timeout=5)
                if response.status_code == 200:
                    latencies.append(response.elapsed.total_seconds() * 1000)
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                
                if avg_latency < 100:
                    details.append(f"Average Latency: {avg_latency:.1f}ms")
                else:
                    details.append(f"Average Latency: {avg_latency:.1f}ms (HIGH - target <100ms)")
                    passed = False
                
                if max_latency < 200:
                    details.append(f"Max Latency: {max_latency:.1f}ms")
                else:
                    details.append(f"Max Latency: {max_latency:.1f}ms (HIGH - target <200ms)")
                    passed = False
            
            # Check system metrics
            response = requests.get(f"{self.base_url}/api/v1/metrics", timeout=5)
            if response.status_code == 200:
                metrics = response.json()
                
                cpu_usage = metrics.get('cpu_usage_percent', 0)
                if cpu_usage < 80:
                    details.append(f"CPU Usage: {cpu_usage:.1f}%")
                else:
                    details.append(f"CPU Usage: {cpu_usage:.1f}% (HIGH - should be <80%)")
                    passed = False
                
                memory_usage = metrics.get('memory_usage_percent', 0)
                if memory_usage < 80:
                    details.append(f"Memory Usage: {memory_usage:.1f}%")
                else:
                    details.append(f"Memory Usage: {memory_usage:.1f}% (HIGH - should be <80%)")
                    passed = False
                    
        except Exception as e:
            details.append(f"Performance Check Failed: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_security(self) -> Tuple[bool, List[str]]:
        """Validate security configuration"""
        details = []
        passed = True
        
        try:
            # Check if API requires authentication
            response = requests.get(f"{self.base_url}/api/v1/protected", timeout=5)
            
            if response.status_code == 401:
                details.append("API Authentication: Enabled")
            else:
                details.append("API Authentication: Not properly configured")
                passed = False
            
            # Check HTTPS redirect (in production should be behind nginx)
            details.append("HTTPS: Should be configured at load balancer/nginx level")
            
            # Check security headers
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            headers = response.headers
            
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block'
            }
            
            for header, expected in security_headers.items():
                if header in headers:
                    details.append(f"Security Header {header}: Present")
                else:
                    details.append(f"Security Header {header}: Missing")
                    # Not failing on missing headers as they might be set at nginx level
                    
        except Exception as e:
            details.append(f"Security Check Failed: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_monitoring(self) -> Tuple[bool, List[str]]:
        """Validate monitoring setup"""
        details = []
        passed = True
        
        try:
            # Check Prometheus metrics endpoint
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            
            if response.status_code == 200:
                details.append("Prometheus Metrics: Available")
                
                # Check for key metrics
                metrics_text = response.text
                key_metrics = [
                    'regime_detection_total',
                    'regime_detection_latency',
                    'regime_current',
                    'system_uptime'
                ]
                
                for metric in key_metrics:
                    if metric in metrics_text:
                        details.append(f"Metric '{metric}': Present")
                    else:
                        details.append(f"Metric '{metric}': Missing")
            else:
                details.append("Prometheus Metrics: Not available")
                passed = False
                
            # Check if Grafana dashboards exist (would need actual Grafana API check)
            details.append("Grafana Dashboards: Should be configured separately")
            
        except Exception as e:
            details.append(f"Monitoring Check Failed: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_logging(self) -> Tuple[bool, List[str]]:
        """Validate logging configuration"""
        details = []
        passed = True
        
        log_dir = Path("/var/log/adaptive_regime")
        
        try:
            if log_dir.exists():
                details.append(f"Log Directory: {log_dir} exists")
                
                # Check for key log files
                expected_logs = ['system.log', 'error.log', 'performance.log']
                
                for log_file in expected_logs:
                    log_path = log_dir / log_file
                    if log_path.exists():
                        size_mb = log_path.stat().st_size / (1024*1024)
                        details.append(f"Log file '{log_file}': Present ({size_mb:.1f}MB)")
                    else:
                        details.append(f"Log file '{log_file}': Missing")
                        passed = False
                        
                # Check log rotation
                if (log_dir / 'system.log.1').exists() or (log_dir / 'system.log.gz').exists():
                    details.append("Log Rotation: Configured")
                else:
                    details.append("Log Rotation: Not yet active")
                    
            else:
                details.append(f"Log Directory: {log_dir} does not exist")
                passed = False
                
        except Exception as e:
            details.append(f"Logging Check Failed: {str(e)}")
            passed = False
        
        return passed, details
    
    def validate_backups(self) -> Tuple[bool, List[str]]:
        """Validate backup configuration"""
        details = []
        passed = True
        
        backup_dir = Path("/backup/adaptive_regime")
        
        try:
            if backup_dir.exists():
                details.append(f"Backup Directory: {backup_dir} exists")
                
                # Check for recent backups
                backup_files = list(backup_dir.glob("*"))
                if backup_files:
                    latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                    age_hours = (time.time() - latest_backup.stat().st_mtime) / 3600
                    
                    if age_hours < 48:
                        details.append(f"Latest Backup: {latest_backup.name} ({age_hours:.1f} hours old)")
                    else:
                        details.append(f"Latest Backup: {latest_backup.name} ({age_hours:.1f} hours old - STALE)")
                        passed = False
                else:
                    details.append("No backups found")
                    passed = False
            else:
                details.append(f"Backup Directory: {backup_dir} does not exist")
                # Not failing as backups might not be set up yet
                
            # Check backup script
            backup_script = Path("/opt/adaptive_regime/scripts/daily_backup.sh")
            if backup_script.exists():
                details.append("Backup Script: Present")
            else:
                details.append("Backup Script: Not found")
                
        except Exception as e:
            details.append(f"Backup Check: {str(e)}")
        
        return passed, details
    
    def generate_summary(self, all_passed: bool):
        """Generate validation summary"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        # Count results
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results.values() if r['passed'])
        failed_checks = total_checks - passed_checks
        
        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {failed_checks}")
        
        # Overall status
        logger.info("")
        if all_passed:
            logger.info("🎉 DEPLOYMENT VALIDATION: PASSED 🎉")
            logger.info("The Adaptive Market Regime System is ready for production use!")
        else:
            logger.error("❌ DEPLOYMENT VALIDATION: FAILED ❌")
            logger.error("Please address the failed checks before using in production.")
            
            # List failed checks
            logger.info("\nFailed Checks:")
            for check_name, result in self.results.items():
                if not result['passed']:
                    logger.error(f"  - {check_name}")
        
        # Save results to file
        report_path = f"/tmp/deployment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': self.start_time.isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                'overall_passed': all_passed,
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        # Execution time
        duration = datetime.now() - self.start_time
        logger.info(f"Validation completed in: {duration.total_seconds():.1f} seconds")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Adaptive Market Regime System deployment')
    parser.add_argument('--config', default='/opt/adaptive_regime/config/production.yaml',
                       help='Path to configuration file')
    parser.add_argument('--url', default='http://localhost:8080',
                       help='Base URL of the service')
    
    args = parser.parse_args()
    
    # Run validation
    validator = DeploymentValidator(args.config)
    if args.url:
        validator.base_url = args.url
    
    success = validator.run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()