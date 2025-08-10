"""
Master Test Runner for POS Strategy System
==========================================

This script orchestrates all POS strategy tests in the correct order and provides
a comprehensive testing workflow with detailed reporting.

Test Execution Order:
1. Pre-flight checks (environment, dependencies, files)
2. Excel configuration parsing tests
3. HeavyDB integration tests  
4. Component logic tests
5. Integration tests
6. Performance validation
7. End-to-end workflow tests
8. Validation procedures and success criteria

CRITICAL: ALL TESTS USE REAL DATA - NO MOCK DATA ALLOWED
Database: HeavyDB localhost:6274 with real NIFTY option chain data

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --quick            # Run quick validation only
    python run_all_tests.py --performance      # Focus on performance tests
    python run_all_tests.py --config-only      # Configuration tests only
    python run_all_tests.py --heavydb-only     # HeavyDB tests only
"""

import argparse
import logging
import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import traceback

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent.parent.parent))

# Import all test modules
from backtester_v2.strategies.pos.test_comprehensive_pos_suite import POSTestSuite, run_pos_comprehensive_tests
from backtester_v2.strategies.pos.test_excel_configuration_detailed import ExcelConfigurationTester, run_excel_configuration_tests
from backtester_v2.strategies.pos.test_heavydb_integration_detailed import HeavyDBIntegrationTester, run_heavydb_integration_tests
from backtester_v2.strategies.pos.test_validation_procedures import ValidationProcedures, run_validation_procedures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'/tmp/pos_test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class POSTestOrchestrator:
    """Master test orchestrator for the complete POS strategy testing suite"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.test_results = {
            'pre_flight': {'status': 'NOT_RUN', 'duration': 0},
            'excel_config': {'status': 'NOT_RUN', 'duration': 0},
            'heavydb_integration': {'status': 'NOT_RUN', 'duration': 0},
            'comprehensive_suite': {'status': 'NOT_RUN', 'duration': 0},
            'validation_procedures': {'status': 'NOT_RUN', 'duration': 0}
        }
        
        self.test_phases = [
            {
                'name': 'pre_flight',
                'description': 'Pre-flight Environment Checks',
                'function': self.run_pre_flight_checks,
                'required': True,
                'timeout_minutes': 5
            },
            {
                'name': 'excel_config',
                'description': 'Excel Configuration Parsing Tests',
                'function': self.run_excel_configuration_tests,
                'required': True,
                'timeout_minutes': 10
            },
            {
                'name': 'heavydb_integration',
                'description': 'HeavyDB Integration Tests',
                'function': self.run_heavydb_integration_tests,
                'required': True,
                'timeout_minutes': 15
            },
            {
                'name': 'comprehensive_suite',
                'description': 'Comprehensive POS Strategy Tests',
                'function': self.run_comprehensive_suite,
                'required': True,
                'timeout_minutes': 30
            },
            {
                'name': 'validation_procedures',
                'description': 'Validation Procedures and Success Criteria',
                'function': self.run_validation_procedures,
                'required': True,
                'timeout_minutes': 20
            }
        ]
        
        # Performance targets and success criteria
        self.success_criteria = {
            'overall_pass_rate': 0.85,  # 85% of tests must pass
            'critical_tests_pass_rate': 1.0,  # 100% of critical tests must pass
            'performance_threshold': 0.5,  # 50% of target performance acceptable
            'data_quality_threshold': 0.90,  # 90% data quality required
            'max_total_duration_minutes': 120  # Maximum 2 hours for full suite
        }
    
    def print_banner(self):
        """Print test suite banner"""
        banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║          POS STRATEGY COMPREHENSIVE TEST SUITE                    ║
    ║                                                                   ║
    ║  Testing all 200+ parameters with real HeavyDB data              ║
    ║  NO MOCK DATA - Real NIFTY option chain validation               ║
    ║                                                                   ║
    ║  Components:                                                      ║
    ║  • Excel Configuration Parsing (25+66+6 parameters)              ║
    ║  • HeavyDB Integration (33.19M+ rows)                            ║
    ║  • Breakeven Analysis (17 parameters)                            ║
    ║  • VIX Configuration (8 ranges)                                  ║
    ║  • Volatility Metrics (IVP, IVR, ATR)                           ║
    ║  • Greeks Calculations                                           ║
    ║  • End-to-End Workflow                                          ║
    ║  • Performance Validation (529,861 rows/sec target)             ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info("POS Strategy Comprehensive Test Suite Started")
        logger.info(f"Start time: {self.start_time}")
    
    def run_pre_flight_checks(self) -> bool:
        """Run pre-flight environment and dependency checks"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: PRE-FLIGHT CHECKS")
        logger.info("="*70)
        
        try:
            checks_passed = 0
            total_checks = 6
            
            # Check 1: Python version
            python_version = sys.version_info
            if python_version.major >= 3 and python_version.minor >= 8:
                logger.info(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
                checks_passed += 1
            else:
                logger.error(f"✗ Python version too old: {python_version}. Need Python 3.8+")
            
            # Check 2: Required Python packages
            required_packages = ['pandas', 'numpy', 'heavydb', 'pydantic', 'openpyxl']
            package_check = True
            
            for package in required_packages:
                try:
                    __import__(package)
                    logger.info(f"✓ Package {package}: Available")
                except ImportError:
                    logger.error(f"✗ Package {package}: Missing")
                    package_check = False
            
            if package_check:
                checks_passed += 1
            
            # Check 3: Configuration files
            config_files = {
                'portfolio': '/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations/data/prod/pos/POS_CONFIG_PORTFOLIO_1.0.0.xlsx',
                'strategy': '/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations/data/prod/pos/POS_CONFIG_STRATEGY_1.0.0.xlsx',
                'adjustment': '/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations/data/prod/pos/POS_CONFIG_ADJUSTMENT_1.0.0.xlsx'
            }
            
            config_check = True
            for config_type, file_path in config_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    logger.info(f"✓ Config {config_type}: Found ({file_size:.1f} KB)")
                else:
                    logger.error(f"✗ Config {config_type}: Missing - {file_path}")
                    config_check = False
            
            if config_check:
                checks_passed += 1
            
            # Check 4: HeavyDB connection
            heavydb_check = self._test_heavydb_connection()
            if heavydb_check:
                logger.info("✓ HeavyDB connection: Successful")
                checks_passed += 1
            else:
                logger.error("✗ HeavyDB connection: Failed")
            
            # Check 5: Disk space
            disk_space_gb = self._check_disk_space()
            if disk_space_gb >= 1.0:  # At least 1GB free
                logger.info(f"✓ Disk space: {disk_space_gb:.1f} GB available")
                checks_passed += 1
            else:
                logger.error(f"✗ Disk space: Only {disk_space_gb:.1f} GB available (need ≥1GB)")
            
            # Check 6: Memory
            memory_gb = self._check_available_memory()
            if memory_gb >= 2.0:  # At least 2GB RAM
                logger.info(f"✓ Available memory: {memory_gb:.1f} GB")
                checks_passed += 1
            else:
                logger.warning(f"⚠ Available memory: {memory_gb:.1f} GB (recommended ≥2GB)")
                checks_passed += 0.5  # Partial credit
            
            success_rate = checks_passed / total_checks
            success = success_rate >= 0.8  # 80% of checks must pass
            
            logger.info(f"\nPre-flight check results: {checks_passed}/{total_checks} ({success_rate:.1%})")
            
            if success:
                logger.info("✓ Pre-flight checks PASSED - environment ready for testing")
            else:
                logger.error("✗ Pre-flight checks FAILED - fix issues before proceeding")
            
            return success
            
        except Exception as e:
            logger.error(f"✗ Pre-flight checks failed with exception: {e}")
            return False
    
    def _test_heavydb_connection(self) -> bool:
        """Quick HeavyDB connection test"""
        try:
            from heavydb import connect
            conn = connect(
                host='localhost',
                port=6274,
                user='admin',
                password='HyperInteractive',
                dbname='heavyai'
            )
            
            # Quick test query
            import pandas as pd
            result = pd.read_sql("SELECT COUNT(*) as count FROM nifty_option_chain LIMIT 1", conn)
            conn.close()
            
            return not result.empty
            
        except Exception as e:
            logger.error(f"HeavyDB connection test failed: {e}")
            return False
    
    def _check_disk_space(self) -> float:
        """Check available disk space in GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage('/tmp')
            return free / (1024**3)  # Convert to GB
        except:
            return 0.0
    
    def _check_available_memory(self) -> float:
        """Check available memory in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable:' in line:
                        available_kb = int(line.split()[1])
                        return available_kb / (1024**2)  # Convert to GB
            return 0.0
        except:
            return 0.0
    
    def run_excel_configuration_tests(self) -> bool:
        """Run Excel configuration parsing tests"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: EXCEL CONFIGURATION TESTS")
        logger.info("="*70)
        
        try:
            # Run Excel configuration tests
            result = run_excel_configuration_tests()
            
            if result == 0:
                logger.info("✓ Excel configuration tests PASSED")
                return True
            else:
                logger.error("✗ Excel configuration tests FAILED")
                return False
                
        except Exception as e:
            logger.error(f"✗ Excel configuration tests failed with exception: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_heavydb_integration_tests(self) -> bool:
        """Run HeavyDB integration tests"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: HEAVYDB INTEGRATION TESTS")
        logger.info("="*70)
        
        try:
            # Run HeavyDB integration tests
            result = run_heavydb_integration_tests()
            
            if result == 0:
                logger.info("✓ HeavyDB integration tests PASSED")
                return True
            else:
                logger.error("✗ HeavyDB integration tests FAILED")
                return False
                
        except Exception as e:
            logger.error(f"✗ HeavyDB integration tests failed with exception: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_comprehensive_suite(self) -> bool:
        """Run comprehensive POS strategy test suite"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: COMPREHENSIVE POS STRATEGY TESTS")
        logger.info("="*70)
        
        try:
            # Run comprehensive test suite
            result = run_pos_comprehensive_tests()
            
            if result == 0:
                logger.info("✓ Comprehensive test suite PASSED")
                return True
            else:
                logger.error("✗ Comprehensive test suite FAILED")
                return False
                
        except Exception as e:
            logger.error(f"✗ Comprehensive test suite failed with exception: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_validation_procedures(self) -> bool:
        """Run validation procedures and success criteria checks"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 5: VALIDATION PROCEDURES")
        logger.info("="*70)
        
        try:
            # Run validation procedures
            result = run_validation_procedures()
            
            if result == 0:
                logger.info("✓ Validation procedures PASSED")
                return True
            else:
                logger.error("✗ Validation procedures FAILED")
                return False
                
        except Exception as e:
            logger.error(f"✗ Validation procedures failed with exception: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_phase(self, phase: Dict[str, Any]) -> bool:
        """Run a single test phase with timing and error handling"""
        phase_name = phase['name']
        phase_description = phase['description']
        phase_function = phase['function']
        
        logger.info(f"\nStarting: {phase_description}")
        
        phase_start = datetime.now()
        
        try:
            # Set timeout
            timeout_seconds = phase.get('timeout_minutes', 30) * 60
            
            # Run the phase
            success = phase_function()
            
            phase_end = datetime.now()
            duration = (phase_end - phase_start).total_seconds()
            
            # Update results
            self.test_results[phase_name] = {
                'status': 'PASSED' if success else 'FAILED',
                'duration': duration,
                'start_time': phase_start.isoformat(),
                'end_time': phase_end.isoformat()
            }
            
            logger.info(f"Phase {phase_description}: {'PASSED' if success else 'FAILED'} ({duration:.1f}s)")
            
            # Check if this is a required phase
            if not success and phase.get('required', False):
                logger.error(f"Required phase failed: {phase_description}")
                return False
            
            return success
            
        except Exception as e:
            phase_end = datetime.now()
            duration = (phase_end - phase_start).total_seconds()
            
            self.test_results[phase_name] = {
                'status': 'ERROR',
                'duration': duration,
                'error': str(e),
                'start_time': phase_start.isoformat(),
                'end_time': phase_end.isoformat()
            }
            
            logger.error(f"Phase {phase_description}: ERROR ({duration:.1f}s) - {e}")
            
            if phase.get('required', False):
                logger.error(f"Required phase had error: {phase_description}")
                return False
            
            return False
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final test report"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate statistics
        phases_run = len([r for r in self.test_results.values() if r['status'] != 'NOT_RUN'])
        phases_passed = len([r for r in self.test_results.values() if r['status'] == 'PASSED'])
        phases_failed = len([r for r in self.test_results.values() if r['status'] == 'FAILED'])
        phases_error = len([r for r in self.test_results.values() if r['status'] == 'ERROR'])
        
        success_rate = phases_passed / phases_run if phases_run > 0 else 0
        
        # Determine overall status
        overall_status = 'PASSED' if success_rate >= self.success_criteria['overall_pass_rate'] else 'FAILED'
        
        # Generate report
        report = {
            'test_suite': 'POS Strategy Comprehensive Test Suite',
            'execution_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'overall_status': overall_status,
                'success_rate': success_rate
            },
            'phase_statistics': {
                'phases_run': phases_run,
                'phases_passed': phases_passed,
                'phases_failed': phases_failed,
                'phases_error': phases_error
            },
            'phase_results': self.test_results,
            'success_criteria': self.success_criteria,
            'recommendations': self._generate_recommendations(),
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform,
                'working_directory': os.getcwd()
            }
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check each phase for specific recommendations
        for phase_name, result in self.test_results.items():
            if result['status'] == 'FAILED':
                if phase_name == 'excel_config':
                    recommendations.append("Review Excel configuration files for missing or invalid parameters")
                elif phase_name == 'heavydb_integration':
                    recommendations.append("Check HeavyDB connection, data availability, and query performance")
                elif phase_name == 'comprehensive_suite':
                    recommendations.append("Investigate comprehensive test failures - check component integration")
                elif phase_name == 'validation_procedures':
                    recommendations.append("Review validation criteria and ensure all business logic is correct")
            
            elif result['status'] == 'ERROR':
                recommendations.append(f"Fix runtime errors in {phase_name} phase before proceeding")
        
        # Performance recommendations
        total_duration_minutes = sum(r.get('duration', 0) for r in self.test_results.values()) / 60
        if total_duration_minutes > self.success_criteria['max_total_duration_minutes']:
            recommendations.append("Optimize test performance - execution time exceeds recommended limits")
        
        # Success recommendations
        if not recommendations:
            recommendations.append("All tests passed successfully - POS strategy system is ready for production")
            recommendations.append("Consider running full regression tests before deployment")
            recommendations.append("Monitor system performance in production environment")
        
        return recommendations
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print comprehensive final test summary"""
        print("\n" + "="*80)
        print("FINAL TEST SUITE SUMMARY")
        print("="*80)
        
        # Execution summary
        exec_summary = report['execution_summary']
        print(f"Start Time:      {exec_summary['start_time']}")
        print(f"End Time:        {exec_summary['end_time']}")
        print(f"Total Duration:  {exec_summary['total_duration_minutes']:.1f} minutes")
        print(f"Overall Status:  {exec_summary['overall_status']}")
        print(f"Success Rate:    {exec_summary['success_rate']:.1%}")
        
        print("\n" + "-"*80)
        print("PHASE RESULTS")
        print("-"*80)
        
        # Phase results
        for phase_name, result in self.test_results.items():
            if result['status'] != 'NOT_RUN':
                status_symbol = "✓" if result['status'] == 'PASSED' else "✗" if result['status'] == 'FAILED' else "⚠"
                duration = result.get('duration', 0)
                phase_description = next((p['description'] for p in self.test_phases if p['name'] == phase_name), phase_name)
                print(f"{status_symbol} {phase_description:<50} | {result['status']:<8} | {duration:>6.1f}s")
        
        # Statistics
        stats = report['phase_statistics']
        print(f"\nPhases Run: {stats['phases_run']}, Passed: {stats['phases_passed']}, Failed: {stats['phases_failed']}, Errors: {stats['phases_error']}")
        
        # Recommendations
        if report['recommendations']:
            print("\n" + "-"*80)
            print("RECOMMENDATIONS")
            print("-"*80)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Final verdict
        print("\n" + "="*80)
        if exec_summary['overall_status'] == 'PASSED':
            print("🎉 POS STRATEGY TEST SUITE: PASSED")
            print("✅ System validated and ready for production use")
        else:
            print("❌ POS STRATEGY TEST SUITE: FAILED")
            print("⚠️  Address issues before production deployment")
        print("="*80)
    
    def run_full_test_suite(self, test_filter: Optional[str] = None) -> bool:
        """Run the complete test suite with optional filtering"""
        
        self.print_banner()
        
        overall_success = True
        phases_to_run = self.test_phases
        
        # Apply test filter if specified
        if test_filter:
            if test_filter == 'quick':
                phases_to_run = [p for p in self.test_phases if p['name'] in ['pre_flight', 'excel_config']]
            elif test_filter == 'config-only':
                phases_to_run = [p for p in self.test_phases if p['name'] in ['pre_flight', 'excel_config']]
            elif test_filter == 'heavydb-only':
                phases_to_run = [p for p in self.test_phases if p['name'] in ['pre_flight', 'heavydb_integration']]
            elif test_filter == 'performance':
                phases_to_run = [p for p in self.test_phases if p['name'] in ['pre_flight', 'heavydb_integration', 'comprehensive_suite']]
        
        logger.info(f"Running {len(phases_to_run)} test phases")
        
        # Execute test phases
        for i, phase in enumerate(phases_to_run, 1):
            logger.info(f"\n[{i}/{len(phases_to_run)}] {phase['description']}")
            
            phase_success = self.run_phase(phase)
            
            if not phase_success:
                overall_success = False
                
                # Stop on critical failures
                if phase.get('required', False) and phase['name'] in ['pre_flight', 'heavydb_integration']:
                    logger.error(f"Critical phase failed: {phase['name']}. Stopping test suite.")
                    break
        
        # Generate and display final report
        report = self.generate_final_report()
        self.print_final_summary(report)
        
        # Save detailed report
        report_file = f"/tmp/pos_test_suite_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"\nDetailed report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"Could not save report file: {e}")
        
        return overall_success and report['execution_summary']['overall_status'] == 'PASSED'


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='POS Strategy Comprehensive Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all_tests.py                # Run complete test suite
    python run_all_tests.py --quick        # Quick validation (pre-flight + config)
    python run_all_tests.py --config-only  # Configuration tests only
    python run_all_tests.py --heavydb-only # HeavyDB integration tests only
    python run_all_tests.py --performance  # Performance-focused tests
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick validation tests only (pre-flight + configuration)'
    )
    
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='Run configuration parsing tests only'
    )
    
    parser.add_argument(
        '--heavydb-only',
        action='store_true',
        help='Run HeavyDB integration tests only'
    )
    
    parser.add_argument(
        '--performance',
        action='store_true',
        help='Run performance-focused tests (excludes validation procedures)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine test filter
    test_filter = None
    if args.quick:
        test_filter = 'quick'
    elif args.config_only:
        test_filter = 'config-only'
    elif args.heavydb_only:
        test_filter = 'heavydb-only'
    elif args.performance:
        test_filter = 'performance'
    
    # Run test suite
    orchestrator = POSTestOrchestrator()
    success = orchestrator.run_full_test_suite(test_filter)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()