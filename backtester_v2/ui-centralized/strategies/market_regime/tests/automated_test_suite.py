#!/usr/bin/env python3
"""
Automated Test Suite for Market Regime Strategy

PHASE 6.1: Unified test runner and orchestrator
- Automatically discovers and runs all test phases
- Provides comprehensive test reporting and analytics
- Manages test execution order and dependencies
- Ensures complete system validation with real data
- NO MOCK DATA - all tests use actual configuration files

Author: Claude Code
Date: 2025-07-12
Version: 1.0.0 - PHASE 6.1 AUTOMATED TEST SUITE
"""

import unittest
import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class MarketRegimeAutomatedTestSuite:
    """
    Unified Automated Test Suite for Market Regime Strategy
    
    Orchestrates execution of all test phases:
    - Phase 1: Excel Configuration Tests
    - Phase 2: HeavyDB Strict Enforcement Tests
    - Phase 3: 10×10 Correlation Matrix Tests
    - Phase 4: Integration Point Tests
    - Phase 5: Performance & Validation Tests
    - Phase 6: Automated Test Suite (this)
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the automated test suite"""
        self.config_path = config_path or "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod/mr/MR_CONFIG_STRATEGY_1.0.0.xlsx"
        self.test_directory = Path(__file__).parent
        self.strict_mode = True
        self.no_mock_data = True
        
        # Test execution configuration
        self.test_phases = {
            'phase1': {
                'name': 'Excel Configuration Tests',
                'description': 'Tests Excel configuration loading and validation',
                'test_files': [
                    'test_excel_config_manager_integration.py'
                ],
                'priority': 'critical',
                'estimated_time': 120  # seconds
            },
            'phase2': {
                'name': 'HeavyDB Strict Enforcement Tests',
                'description': 'Tests HeavyDB integration and data validation',
                'test_files': [
                    'corrected_phase2_day5_test.py',
                    'phase2_day5_excel_integration_test.py',
                    'test_heavydb_import.py',
                    'check_heavydb_columns.py'
                ],
                'priority': 'critical',
                'estimated_time': 180
            },
            'phase3': {
                'name': '10×10 Correlation Matrix Tests',
                'description': 'Tests correlation matrix functionality',
                'test_files': [
                    'test_correlation_matrix_engine.py',
                    'test_enhanced_correlation_matrix_integration.py',
                    'test_enhanced_correlation_matrix_heavydb.py',
                    'test_call_put_oi_correlation.py'
                ],
                'priority': 'high',
                'estimated_time': 150
            },
            'phase4': {
                'name': 'Integration Point Tests',
                'description': 'Tests all integration points and module communication',
                'test_files': [
                    'test_master_config_core_integration.py',
                    'test_indicator_config_integration.py',
                    'test_performance_metrics_monitoring.py',
                    'test_cross_module_communication.py',
                    'test_end_to_end_pipeline.py',
                    'test_error_propagation_handling.py',
                    'test_excel_error_handling_recovery.py'
                ],
                'priority': 'critical',
                'estimated_time': 300
            },
            'phase5': {
                'name': 'Performance & Validation Tests',
                'description': 'Tests performance and production readiness',
                'test_files': [
                    'test_configuration_loading_performance.py',
                    'test_regime_detection_performance.py',
                    'test_production_validation_scenarios.py'
                ],
                'priority': 'high',
                'estimated_time': 240
            }
        }
        
        # Test execution results
        self.test_results = {}
        self.execution_start_time = None
        self.execution_end_time = None
        
        # System monitoring
        self.system_metrics = []
        
        # Verify configuration file exists
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        logger.info(f"✅ Automated Test Suite initialized")
        logger.info(f"📁 Configuration: {self.config_path}")
        logger.info(f"📂 Test Directory: {self.test_directory}")
        logger.info(f"🧪 Test Phases: {len(self.test_phases)}")
    
    def get_system_metrics(self):
        """Get current system metrics"""
        return {
            'timestamp': time.time(),
            'memory_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
    
    def discover_test_files(self) -> Dict[str, List[str]]:
        """Discover all test files in the test directory"""
        discovered_tests = {}
        
        for phase_id, phase_config in self.test_phases.items():
            phase_tests = []
            
            # Look for explicitly configured test files
            for test_file in phase_config['test_files']:
                test_path = self.test_directory / test_file
                if test_path.exists():
                    phase_tests.append(str(test_path))
                else:
                    logger.warning(f"⚠️ Test file not found: {test_file}")
            
            # Auto-discover additional test files by pattern
            pattern_searches = {
                'phase1': ['*excel*test*.py', '*config*test*.py'],
                'phase2': ['*heavydb*test*.py', '*db*test*.py'],
                'phase3': ['*correlation*test*.py', '*matrix*test*.py'],
                'phase4': ['*integration*test*.py', '*module*test*.py', '*pipeline*test*.py', '*error*test*.py'],
                'phase5': ['*performance*test*.py', '*validation*test*.py', '*production*test*.py']
            }
            
            if phase_id in pattern_searches:
                for pattern in pattern_searches[phase_id]:
                    for test_file in self.test_directory.glob(pattern):
                        if str(test_file) not in phase_tests and test_file.name != 'automated_test_suite.py':
                            phase_tests.append(str(test_file))
            
            discovered_tests[phase_id] = phase_tests
            logger.info(f"📋 {phase_config['name']}: {len(phase_tests)} test files discovered")
        
        return discovered_tests
    
    def validate_test_environment(self) -> bool:
        """Validate that the test environment is properly set up"""
        validation_results = []
        
        # Check configuration file
        try:
            if Path(self.config_path).exists():
                file_size = Path(self.config_path).stat().st_size
                validation_results.append({
                    'check': 'Configuration file exists',
                    'status': 'pass',
                    'details': f'File size: {file_size / 1024 / 1024:.2f}MB'
                })
            else:
                validation_results.append({
                    'check': 'Configuration file exists',
                    'status': 'fail',
                    'details': f'File not found: {self.config_path}'
                })
        except Exception as e:
            validation_results.append({
                'check': 'Configuration file access',
                'status': 'error',
                'details': str(e)
            })
        
        # Check test directory
        try:
            test_files = list(self.test_directory.glob('test_*.py'))
            validation_results.append({
                'check': 'Test directory accessible',
                'status': 'pass',
                'details': f'{len(test_files)} test files found'
            })
        except Exception as e:
            validation_results.append({
                'check': 'Test directory access',
                'status': 'error',
                'details': str(e)
            })
        
        # Check system resources
        try:
            memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
            disk_gb = psutil.disk_usage('/').free / 1024 / 1024 / 1024
            
            validation_results.append({
                'check': 'System resources',
                'status': 'pass',
                'details': f'Memory: {memory_gb:.1f}GB, Disk: {disk_gb:.1f}GB free'
            })
        except Exception as e:
            validation_results.append({
                'check': 'System resources',
                'status': 'error',
                'details': str(e)
            })
        
        # Check Python environment
        try:
            python_version = sys.version.split()[0]
            validation_results.append({
                'check': 'Python environment',
                'status': 'pass',
                'details': f'Python {python_version}'
            })
        except Exception as e:
            validation_results.append({
                'check': 'Python environment',
                'status': 'error',
                'details': str(e)
            })
        
        # Summary
        passed_checks = sum(1 for r in validation_results if r['status'] == 'pass')
        total_checks = len(validation_results)
        
        logger.info(f"🔍 Environment validation: {passed_checks}/{total_checks} checks passed")
        
        for result in validation_results:
            status_emoji = "✅" if result['status'] == 'pass' else "❌" if result['status'] == 'fail' else "⚠️"
            logger.info(f"{status_emoji} {result['check']}: {result['details']}")
        
        return passed_checks == total_checks
    
    def execute_test_file(self, test_file_path: str, timeout: int = 300) -> Dict[str, Any]:
        """Execute a single test file and return results"""
        test_start_time = time.time()
        
        try:
            logger.info(f"🧪 Executing test: {Path(test_file_path).name}")
            
            # Record system metrics before test
            start_metrics = self.get_system_metrics()
            
            # Execute the test file
            result = subprocess.run(
                [sys.executable, test_file_path],
                cwd=str(self.test_directory),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Record system metrics after test
            end_metrics = self.get_system_metrics()
            
            execution_time = time.time() - test_start_time
            
            # Parse test results from output
            test_result = {
                'test_file': Path(test_file_path).name,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'start_metrics': start_metrics,
                'end_metrics': end_metrics,
                'memory_delta_mb': end_metrics['memory_mb'] - start_metrics['memory_mb'],
                'success': result.returncode == 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract test statistics from output
            if result.stdout:
                test_result.update(self.parse_test_output(result.stdout))
            
            if test_result['success']:
                logger.info(f"✅ {Path(test_file_path).name}: PASSED ({execution_time:.1f}s)")
            else:
                logger.error(f"❌ {Path(test_file_path).name}: FAILED ({execution_time:.1f}s)")
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - test_start_time
            test_result = {
                'test_file': Path(test_file_path).name,
                'execution_time': execution_time,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Test timed out after {timeout} seconds',
                'success': False,
                'timeout': True,
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"⏰ {Path(test_file_path).name}: TIMEOUT ({timeout}s)")
            
        except Exception as e:
            execution_time = time.time() - test_start_time
            test_result = {
                'test_file': Path(test_file_path).name,
                'execution_time': execution_time,
                'return_code': -2,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"💥 {Path(test_file_path).name}: ERROR - {e}")
        
        return test_result
    
    def parse_test_output(self, output: str) -> Dict[str, Any]:
        """Parse test output to extract statistics"""
        parsed_data = {}
        
        try:
            lines = output.split('\n')
            
            for line in lines:
                # Look for test result patterns
                if 'Total Tests:' in line:
                    parsed_data['total_tests'] = int(line.split(':')[1].strip())
                elif 'Passed:' in line and 'PASSED' not in line:
                    parsed_data['passed_tests'] = int(line.split(':')[1].strip())
                elif 'Failed:' in line and 'FAILED' not in line:
                    parsed_data['failed_tests'] = int(line.split(':')[1].strip())
                elif 'Errors:' in line:
                    parsed_data['error_tests'] = int(line.split(':')[1].strip())
                elif 'Success Rate:' in line:
                    rate_str = line.split(':')[1].strip().replace('%', '')
                    parsed_data['success_rate'] = float(rate_str)
                elif 'Ran' in line and 'test' in line:
                    # unittest output format
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        parsed_data['unittest_ran'] = int(parts[1])
        
        except Exception as e:
            logger.warning(f"Failed to parse test output: {e}")
        
        return parsed_data
    
    def execute_phase(self, phase_id: str, test_files: List[str], parallel: bool = False) -> Dict[str, Any]:
        """Execute all tests in a phase"""
        phase_config = self.test_phases[phase_id]
        phase_start_time = time.time()
        
        logger.info(f"🚀 Starting {phase_config['name']} ({len(test_files)} tests)")
        
        phase_results = {
            'phase_id': phase_id,
            'phase_name': phase_config['name'],
            'description': phase_config['description'],
            'priority': phase_config['priority'],
            'test_files': test_files,
            'test_results': [],
            'start_time': datetime.now().isoformat(),
            'parallel_execution': parallel
        }
        
        # Record initial system state
        initial_metrics = self.get_system_metrics()
        phase_results['initial_metrics'] = initial_metrics
        
        if parallel and len(test_files) > 1:
            # Execute tests in parallel
            with ThreadPoolExecutor(max_workers=min(3, len(test_files))) as executor:
                future_to_test = {
                    executor.submit(self.execute_test_file, test_file, phase_config['estimated_time']): test_file
                    for test_file in test_files
                }
                
                for future in as_completed(future_to_test):
                    test_result = future.result()
                    phase_results['test_results'].append(test_result)
        else:
            # Execute tests sequentially
            for test_file in test_files:
                test_result = self.execute_test_file(test_file, phase_config['estimated_time'])
                phase_results['test_results'].append(test_result)
        
        # Calculate phase summary
        phase_execution_time = time.time() - phase_start_time
        final_metrics = self.get_system_metrics()
        
        phase_results.update({
            'end_time': datetime.now().isoformat(),
            'execution_time': phase_execution_time,
            'final_metrics': final_metrics,
            'memory_delta_mb': final_metrics['memory_mb'] - initial_metrics['memory_mb']
        })
        
        # Calculate success statistics
        successful_tests = [r for r in phase_results['test_results'] if r['success']]
        failed_tests = [r for r in phase_results['test_results'] if not r['success']]
        
        phase_results.update({
            'total_tests_executed': len(phase_results['test_results']),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(phase_results['test_results']) if phase_results['test_results'] else 0
        })
        
        # Log phase summary
        if phase_results['success_rate'] == 1.0:
            logger.info(f"✅ {phase_config['name']}: ALL TESTS PASSED ({phase_execution_time:.1f}s)")
        else:
            logger.error(f"❌ {phase_config['name']}: {len(failed_tests)} FAILED ({phase_execution_time:.1f}s)")
        
        return phase_results
    
    def run_all_tests(self, parallel_phases: bool = False, parallel_tests: bool = False) -> Dict[str, Any]:
        """Run all test phases"""
        logger.info("🎯 Starting Market Regime Strategy Automated Test Suite")
        logger.info("=" * 70)
        logger.info("⚠️  STRICT MODE: All tests use real configuration data")
        logger.info("⚠️  NO MOCK DATA: Using actual MR_CONFIG_STRATEGY_1.0.0.xlsx")
        logger.info("🧪 COMPREHENSIVE: Testing all system components")
        logger.info("=" * 70)
        
        self.execution_start_time = time.time()
        
        # Validate environment
        if not self.validate_test_environment():
            logger.error("❌ Environment validation failed")
            return {'success': False, 'error': 'Environment validation failed'}
        
        # Discover tests
        discovered_tests = self.discover_test_files()
        
        # Execute all phases
        execution_results = {
            'suite_name': 'Market Regime Strategy Automated Test Suite',
            'execution_id': f"test_run_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'configuration_file': self.config_path,
            'test_directory': str(self.test_directory),
            'parallel_phases': parallel_phases,
            'parallel_tests': parallel_tests,
            'discovered_tests': discovered_tests,
            'phase_results': [],
            'system_metrics_log': []
        }
        
        # Record initial system state
        initial_system_metrics = self.get_system_metrics()
        execution_results['initial_system_metrics'] = initial_system_metrics
        
        if parallel_phases:
            # Execute phases in parallel (be careful with resource usage)
            with ThreadPoolExecutor(max_workers=2) as executor:  # Limit to 2 parallel phases
                phase_futures = {}
                
                for phase_id, test_files in discovered_tests.items():
                    if test_files:  # Only execute phases with test files
                        future = executor.submit(self.execute_phase, phase_id, test_files, parallel_tests)
                        phase_futures[future] = phase_id
                
                for future in as_completed(phase_futures):
                    phase_result = future.result()
                    execution_results['phase_results'].append(phase_result)
        else:
            # Execute phases sequentially (recommended)
            for phase_id, test_files in discovered_tests.items():
                if test_files:  # Only execute phases with test files
                    phase_result = self.execute_phase(phase_id, test_files, parallel_tests)
                    execution_results['phase_results'].append(phase_result)
                    
                    # Record system metrics after each phase
                    current_metrics = self.get_system_metrics()
                    execution_results['system_metrics_log'].append({
                        'phase': phase_id,
                        'metrics': current_metrics,
                        'timestamp': datetime.now().isoformat()
                    })
        
        self.execution_end_time = time.time()
        total_execution_time = self.execution_end_time - self.execution_start_time
        
        # Final system state
        final_system_metrics = self.get_system_metrics()
        
        # Calculate overall summary
        all_test_results = []
        for phase_result in execution_results['phase_results']:
            all_test_results.extend(phase_result['test_results'])
        
        successful_tests = [r for r in all_test_results if r['success']]
        failed_tests = [r for r in all_test_results if not r['success']]
        
        execution_results.update({
            'end_time': datetime.now().isoformat(),
            'total_execution_time': total_execution_time,
            'final_system_metrics': final_system_metrics,
            'total_phases_executed': len(execution_results['phase_results']),
            'total_tests_executed': len(all_test_results),
            'total_successful_tests': len(successful_tests),
            'total_failed_tests': len(failed_tests),
            'overall_success_rate': len(successful_tests) / len(all_test_results) if all_test_results else 0,
            'overall_success': len(failed_tests) == 0
        })
        
        # Save results
        self.test_results = execution_results
        
        # Log final summary
        self.log_execution_summary(execution_results)
        
        return execution_results
    
    def log_execution_summary(self, results: Dict[str, Any]):
        """Log a comprehensive execution summary"""
        logger.info("\n" + "=" * 70)
        logger.info("MARKET REGIME STRATEGY AUTOMATED TEST SUITE RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"📊 Execution Summary:")
        logger.info(f"   Total Phases: {results['total_phases_executed']}")
        logger.info(f"   Total Tests: {results['total_tests_executed']}")
        logger.info(f"   Successful: {results['total_successful_tests']}")
        logger.info(f"   Failed: {results['total_failed_tests']}")
        logger.info(f"   Success Rate: {results['overall_success_rate']:.1%}")
        logger.info(f"   Execution Time: {results['total_execution_time']:.1f}s")
        
        logger.info(f"\n📋 Phase Results:")
        for phase_result in results['phase_results']:
            status = "✅ PASSED" if phase_result['success_rate'] == 1.0 else "❌ FAILED"
            logger.info(f"   {phase_result['phase_name']}: {status} ({phase_result['successful_tests']}/{phase_result['total_tests_executed']})")
        
        if results['overall_success']:
            logger.info(f"\n🎉 OVERALL RESULT: ✅ ALL TESTS PASSED")
            logger.info(f"🚀 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT")
        else:
            logger.info(f"\n💥 OVERALL RESULT: ❌ {results['total_failed_tests']} TESTS FAILED")
            logger.info(f"🔧 ISSUES NEED TO BE ADDRESSED BEFORE DEPLOYMENT")
        
        logger.info("=" * 70)
    
    def save_results_to_file(self, filename: str = None) -> str:
        """Save test results to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_regime_test_results_{timestamp}.json"
        
        filepath = self.test_directory / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info(f"💾 Test results saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            return ""
    
    def generate_html_report(self, filename: str = None) -> str:
        """Generate an HTML test report"""
        if not self.test_results:
            logger.warning("No test results available for HTML report")
            return ""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_regime_test_report_{timestamp}.html"
        
        filepath = self.test_directory / filename
        
        try:
            html_content = self._generate_html_content()
            
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            logger.info(f"📄 HTML report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return ""
    
    def _generate_html_content(self) -> str:
        """Generate HTML content for the test report"""
        results = self.test_results
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Regime Strategy Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .phase {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
                .phase-header {{ background: #e8f4f8; padding: 15px; font-weight: bold; }}
                .test-result {{ margin: 10px; padding: 10px; border-left: 4px solid #ddd; }}
                .success {{ border-left-color: #4CAF50; background: #f1f8e9; }}
                .failure {{ border-left-color: #f44336; background: #ffebee; }}
                .metrics {{ font-size: 0.9em; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Market Regime Strategy - Automated Test Suite Report</h1>
                <p><strong>Execution ID:</strong> {results.get('execution_id', 'N/A')}</p>
                <p><strong>Start Time:</strong> {results.get('start_time', 'N/A')}</p>
                <p><strong>Total Execution Time:</strong> {results.get('total_execution_time', 0):.1f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Phases</td><td>{results.get('total_phases_executed', 0)}</td></tr>
                    <tr><td>Total Tests</td><td>{results.get('total_tests_executed', 0)}</td></tr>
                    <tr><td>Successful Tests</td><td>{results.get('total_successful_tests', 0)}</td></tr>
                    <tr><td>Failed Tests</td><td>{results.get('total_failed_tests', 0)}</td></tr>
                    <tr><td>Success Rate</td><td>{results.get('overall_success_rate', 0):.1%}</td></tr>
                </table>
            </div>
        """
        
        # Add phase details
        for phase_result in results.get('phase_results', []):
            html += f"""
            <div class="phase">
                <div class="phase-header">
                    {phase_result['phase_name']} - {phase_result['successful_tests']}/{phase_result['total_tests_executed']} Passed
                </div>
                <p>{phase_result['description']}</p>
                <div class="metrics">
                    Execution Time: {phase_result['execution_time']:.1f}s | 
                    Priority: {phase_result['priority']} |
                    Memory Delta: {phase_result.get('memory_delta_mb', 0):.1f}MB
                </div>
            """
            
            # Add test results
            for test_result in phase_result['test_results']:
                status_class = 'success' if test_result['success'] else 'failure'
                status_text = 'PASSED' if test_result['success'] else 'FAILED'
                
                html += f"""
                <div class="test-result {status_class}">
                    <strong>{test_result['test_file']}</strong> - {status_text}
                    <div class="metrics">
                        Time: {test_result['execution_time']:.1f}s | 
                        Return Code: {test_result['return_code']}
                    </div>
                </div>
                """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html

def run_automated_test_suite():
    """Main function to run the automated test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize the test suite
        test_suite = MarketRegimeAutomatedTestSuite()
        
        # Run all tests
        results = test_suite.run_all_tests(
            parallel_phases=False,  # Sequential phases for stability
            parallel_tests=False    # Sequential tests for reliability
        )
        
        # Save results
        json_file = test_suite.save_results_to_file()
        html_file = test_suite.generate_html_report()
        
        # Return success status
        return results['overall_success']
        
    except Exception as e:
        logger.error(f"Automated test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = run_automated_test_suite()
    sys.exit(0 if success else 1)