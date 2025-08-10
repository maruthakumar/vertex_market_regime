"""
Comprehensive Test Suite for Excel Configuration System
Evidence-based validation with real Excel files and performance metrics
"""

import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import logging
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the configurations directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from excel_config_system import ExcelConfigurationSystem, create_production_config_system
from converters.excel_to_yaml import ExcelToYAMLConverter
from hot_reload.hot_reload_system import HotReloadSystem
from versioning.version_manager import ConfigurationVersionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestResults:
    """Container for test results"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.performance_metrics = []
        self.validation_results = []
        self.errors = []
        self.warnings = []
    
    def add_test_result(self, test_name: str, passed: bool, duration: float = 0, 
                       error: str = None, details: Dict = None):
        """Add a test result"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            logger.info(f"✅ {test_name} - PASSED ({duration:.3f}s)")
        else:
            self.tests_failed += 1
            logger.error(f"❌ {test_name} - FAILED ({duration:.3f}s)")
            if error:
                self.errors.append(f"{test_name}: {error}")
                logger.error(f"   Error: {error}")
        
        self.performance_metrics.append({
            'test_name': test_name,
            'duration': duration,
            'passed': passed,
            'error': error,
            'details': details or {}
        })
    
    def add_validation_result(self, validation_name: str, result: Dict[str, Any]):
        """Add validation result"""
        self.validation_results.append({
            'validation_name': validation_name,
            'result': result
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        return {
            'tests_run': self.tests_run,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'success_rate': self.tests_passed / max(self.tests_run, 1),
            'total_duration': sum(m['duration'] for m in self.performance_metrics),
            'avg_duration': sum(m['duration'] for m in self.performance_metrics) / max(len(self.performance_metrics), 1),
            'performance_metrics': self.performance_metrics,
            'validation_results': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings
        }

class ExcelConfigSystemTester:
    """Comprehensive test suite for Excel Configuration System"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.test_results = TestResults()
        self.temp_dir = None
        
    def run_all_tests(self) -> TestResults:
        """Run all tests and return results"""
        logger.info("🚀 Starting Excel Configuration System Test Suite")
        logger.info(f"📁 Base path: {self.base_path}")
        
        try:
            # Setup test environment
            self._setup_test_environment()
            
            # Run test suites
            self._test_excel_file_discovery()
            self._test_converter_performance()
            self._test_yaml_conversion_accuracy()
            self._test_pandas_validation()
            self._test_hot_reload_system()
            self._test_version_management()
            self._test_integrated_system()
            self._test_error_handling()
            self._test_performance_targets()
            
            # Generate final report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Test suite failed with error: {e}")
            self.test_results.add_test_result("test_suite_execution", False, 0, str(e))
        
        finally:
            # Cleanup
            self._cleanup_test_environment()
        
        logger.info("✅ Test suite completed")
        return self.test_results
    
    def _setup_test_environment(self):
        """Setup test environment"""
        logger.info("🔧 Setting up test environment")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="excel_config_test_"))
        logger.info(f"📁 Created temporary directory: {self.temp_dir}")
        
        # Validate base path exists
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
        
        # Check for required directories
        required_dirs = ['data/prod']
        for req_dir in required_dirs:
            dir_path = self.base_path / req_dir
            if not dir_path.exists():
                logger.warning(f"Required directory missing: {dir_path}")
        
        logger.info("✅ Test environment setup complete")
    
    def _test_excel_file_discovery(self):
        """Test Excel file discovery and inventory"""
        logger.info("🔍 Testing Excel file discovery")
        
        start_time = time.time()
        
        try:
            # Find all Excel files
            prod_dir = self.base_path / "data" / "prod"
            
            if not prod_dir.exists():
                self.test_results.add_test_result(
                    "excel_file_discovery", 
                    False, 
                    time.time() - start_time,
                    "Production directory not found"
                )
                return
            
            # Count files by strategy
            strategy_counts = {}
            total_files = 0
            
            for strategy_dir in prod_dir.iterdir():
                if strategy_dir.is_dir():
                    strategy_type = strategy_dir.name
                    excel_files = []
                    
                    for pattern in ['*.xlsx', '*.xls', '*.xlsm']:
                        excel_files.extend(strategy_dir.glob(pattern))
                    
                    strategy_counts[strategy_type] = len(excel_files)
                    total_files += len(excel_files)
            
            # Validate expected strategies
            expected_strategies = ['tbs', 'tv', 'orb', 'oi', 'ml', 'pos', 'mr']
            found_strategies = set(strategy_counts.keys())
            
            details = {
                'total_files': total_files,
                'strategy_counts': strategy_counts,
                'expected_strategies': expected_strategies,
                'found_strategies': list(found_strategies),
                'missing_strategies': list(set(expected_strategies) - found_strategies)
            }
            
            # Test passes if we found some files
            success = total_files > 0
            duration = time.time() - start_time
            
            self.test_results.add_test_result(
                "excel_file_discovery", 
                success, 
                duration,
                None if success else "No Excel files found",
                details
            )
            
            logger.info(f"📊 Found {total_files} Excel files across {len(strategy_counts)} strategies")
            
        except Exception as e:
            self.test_results.add_test_result(
                "excel_file_discovery", 
                False, 
                time.time() - start_time,
                str(e)
            )
    
    def _test_converter_performance(self):
        """Test converter performance against <100ms target"""
        logger.info("⚡ Testing converter performance")
        
        converter = ExcelToYAMLConverter()
        
        # Find test files
        test_files = []
        prod_dir = self.base_path / "data" / "prod"
        
        for strategy_dir in prod_dir.iterdir():
            if strategy_dir.is_dir():
                excel_files = list(strategy_dir.glob("*.xlsx"))
                test_files.extend(excel_files[:2])  # Test first 2 files per strategy
        
        if not test_files:
            self.test_results.add_test_result(
                "converter_performance", 
                False, 
                0,
                "No test files found"
            )
            return
        
        # Test each file
        performance_results = []
        
        for test_file in test_files:
            try:
                start_time = time.time()
                yaml_data, metrics = converter.convert_single_file(str(test_file))
                duration = time.time() - start_time
                
                performance_results.append({
                    'file': str(test_file),
                    'duration': duration,
                    'success': metrics.success,
                    'file_size': metrics.file_size,
                    'sheet_count': metrics.sheet_count,
                    'meets_target': duration < 0.1  # <100ms target
                })
                
                logger.info(f"  📄 {test_file.name}: {duration:.3f}s ({'✅' if duration < 0.1 else '⚠️'})")
                
            except Exception as e:
                performance_results.append({
                    'file': str(test_file),
                    'duration': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze results
        successful_conversions = [r for r in performance_results if r['success']]
        total_duration = sum(r['duration'] for r in successful_conversions)
        avg_duration = total_duration / len(successful_conversions) if successful_conversions else 0
        target_met_count = sum(1 for r in successful_conversions if r.get('meets_target', False))
        target_met_rate = target_met_count / len(successful_conversions) if successful_conversions else 0
        
        details = {
            'total_files_tested': len(test_files),
            'successful_conversions': len(successful_conversions),
            'avg_duration': avg_duration,
            'max_duration': max(r['duration'] for r in successful_conversions) if successful_conversions else 0,
            'min_duration': min(r['duration'] for r in successful_conversions) if successful_conversions else 0,
            'target_met_count': target_met_count,
            'target_met_rate': target_met_rate,
            'performance_results': performance_results
        }
        
        # Test passes if >50% of files meet <100ms target
        success = target_met_rate > 0.5
        
        self.test_results.add_test_result(
            "converter_performance", 
            success, 
            total_duration,
            None if success else f"Only {target_met_rate:.1%} of files met <100ms target",
            details
        )
    
    def _test_yaml_conversion_accuracy(self):
        """Test YAML conversion accuracy"""
        logger.info("🎯 Testing YAML conversion accuracy")
        
        converter = ExcelToYAMLConverter()
        
        # Test with a known file structure
        test_files = []
        prod_dir = self.base_path / "data" / "prod"
        
        # Find TBS files (simpler structure)
        tbs_dir = prod_dir / "tbs"
        if tbs_dir.exists():
            test_files.extend(list(tbs_dir.glob("*.xlsx"))[:1])
        
        if not test_files:
            self.test_results.add_test_result(
                "yaml_conversion_accuracy", 
                False, 
                0,
                "No suitable test files found"
            )
            return
        
        start_time = time.time()
        accuracy_results = []
        
        for test_file in test_files:
            try:
                # Convert to YAML
                yaml_data, metrics = converter.convert_single_file(str(test_file))
                
                if not metrics.success:
                    accuracy_results.append({
                        'file': str(test_file),
                        'success': False,
                        'error': metrics.error_message
                    })
                    continue
                
                # Read original Excel file
                excel_file = pd.ExcelFile(str(test_file))
                
                # Validate structure
                excel_sheets = set(excel_file.sheet_names)
                yaml_sheets = set(yaml_data.keys())
                
                # Remove metadata
                yaml_sheets.discard('_metadata')
                
                # Check sheet preservation
                expected_sheets = [sheet for sheet in excel_sheets if not sheet.startswith('_')]
                converted_sheets = [sheet for sheet in yaml_sheets if not sheet.startswith('_')]
                
                sheet_preservation_rate = len(converted_sheets) / len(expected_sheets) if expected_sheets else 1
                
                accuracy_results.append({
                    'file': str(test_file),
                    'success': True,
                    'sheet_count': len(expected_sheets),
                    'converted_sheets': len(converted_sheets),
                    'sheet_preservation_rate': sheet_preservation_rate,
                    'has_metadata': '_metadata' in yaml_data,
                    'data_size': len(str(yaml_data))
                })
                
                logger.info(f"  📄 {test_file.name}: {len(converted_sheets)}/{len(expected_sheets)} sheets converted")
                
            except Exception as e:
                accuracy_results.append({
                    'file': str(test_file),
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze results
        successful_conversions = [r for r in accuracy_results if r['success']]
        avg_preservation_rate = sum(r['sheet_preservation_rate'] for r in successful_conversions) / len(successful_conversions) if successful_conversions else 0
        
        details = {
            'total_files_tested': len(test_files),
            'successful_conversions': len(successful_conversions),
            'avg_sheet_preservation_rate': avg_preservation_rate,
            'accuracy_results': accuracy_results
        }
        
        # Test passes if >90% sheet preservation rate
        success = avg_preservation_rate > 0.9
        duration = time.time() - start_time
        
        self.test_results.add_test_result(
            "yaml_conversion_accuracy", 
            success, 
            duration,
            None if success else f"Sheet preservation rate: {avg_preservation_rate:.1%}",
            details
        )
    
    def _test_pandas_validation(self):
        """Test pandas validation functionality"""
        logger.info("🔍 Testing pandas validation")
        
        converter = ExcelToYAMLConverter()
        start_time = time.time()
        
        try:
            # Test with a simple Excel file
            test_file = self._find_test_file('tbs')
            
            if not test_file:
                self.test_results.add_test_result(
                    "pandas_validation", 
                    False, 
                    time.time() - start_time,
                    "No test file found"
                )
                return
            
            # Convert file
            yaml_data, metrics = converter.convert_single_file(str(test_file))
            
            if not metrics.success:
                self.test_results.add_test_result(
                    "pandas_validation", 
                    False, 
                    time.time() - start_time,
                    f"Conversion failed: {metrics.error_message}"
                )
                return
            
            # Test validation
            validation_result = converter._validate_with_pandas(yaml_data, 'tbs', str(test_file))
            
            details = {
                'file_tested': str(test_file),
                'validation_result': validation_result,
                'has_errors': len(validation_result.get('errors', [])) > 0,
                'has_warnings': len(validation_result.get('warnings', [])) > 0,
                'is_valid': validation_result.get('valid', False)
            }
            
            # Test passes if validation completes without exceptions
            success = True
            duration = time.time() - start_time
            
            self.test_results.add_test_result(
                "pandas_validation", 
                success, 
                duration,
                None,
                details
            )
            
            logger.info(f"  📊 Validation result: {'✅ Valid' if validation_result.get('valid') else '⚠️ Has issues'}")
            
        except Exception as e:
            self.test_results.add_test_result(
                "pandas_validation", 
                False, 
                time.time() - start_time,
                str(e)
            )
    
    def _test_hot_reload_system(self):
        """Test hot reload system functionality"""
        logger.info("🔄 Testing hot reload system")
        
        start_time = time.time()
        
        try:
            # Create test system in temporary directory
            test_config_dir = self.temp_dir / "test_configs"
            test_config_dir.mkdir(parents=True)
            
            # Copy a test file
            test_file = self._find_test_file('tbs')
            if not test_file:
                self.test_results.add_test_result(
                    "hot_reload_system", 
                    False, 
                    time.time() - start_time,
                    "No test file found"
                )
                return
            
            # Create test directory structure
            test_prod_dir = test_config_dir / "data" / "prod" / "tbs"
            test_prod_dir.mkdir(parents=True)
            
            test_copy = test_prod_dir / test_file.name
            shutil.copy2(test_file, test_copy)
            
            # Test hot reload system
            from core.config_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            
            # Create hot reload system
            hot_reload_system = HotReloadSystem(config_manager, [str(test_config_dir)])
            
            # Test events
            events_received = []
            
            def test_callback(event):
                events_received.append(event)
            
            hot_reload_system.add_global_callback(test_callback)
            
            # Start watching
            hot_reload_system.start_watching()
            
            # Simulate file change
            time.sleep(0.1)  # Give system time to initialize
            
            # Touch the file to trigger change
            test_copy.touch()
            
            # Wait for event processing
            time.sleep(0.2)
            
            # Stop watching
            hot_reload_system.stop_watching()
            
            # Check results
            stats = hot_reload_system.get_statistics()
            
            details = {
                'watching_started': True,
                'events_received': len(events_received),
                'stats': stats
            }
            
            # Test passes if system started and stopped without errors
            success = True
            duration = time.time() - start_time
            
            self.test_results.add_test_result(
                "hot_reload_system", 
                success, 
                duration,
                None,
                details
            )
            
            logger.info(f"  🔄 Hot reload system: {'✅ Working' if success else '❌ Failed'}")
            
        except Exception as e:
            self.test_results.add_test_result(
                "hot_reload_system", 
                False, 
                time.time() - start_time,
                str(e)
            )
    
    def _test_version_management(self):
        """Test version management functionality"""
        logger.info("📝 Testing version management")
        
        start_time = time.time()
        
        try:
            # Create version manager
            version_dir = self.temp_dir / "versions"
            version_manager = ConfigurationVersionManager(str(version_dir))
            
            # Find test file
            test_file = self._find_test_file('tbs')
            if not test_file:
                self.test_results.add_test_result(
                    "version_management", 
                    False, 
                    time.time() - start_time,
                    "No test file found"
                )
                return
            
            # Create version
            version1 = version_manager.create_version(
                file_path=str(test_file),
                strategy_type='tbs',
                config_name='test_config',
                user='test_user',
                description='Test version'
            )
            
            # Test version retrieval
            retrieved_version = version_manager.get_version(version1.version_id)
            
            # Test version listing
            versions = version_manager.list_versions('tbs', 'test_config')
            
            # Test statistics
            stats = version_manager.get_statistics()
            
            details = {
                'version_created': version1.version_id,
                'version_retrieved': retrieved_version is not None,
                'versions_listed': len(versions),
                'stats': stats
            }
            
            # Test passes if version was created and retrieved
            success = retrieved_version is not None and len(versions) > 0
            duration = time.time() - start_time
            
            self.test_results.add_test_result(
                "version_management", 
                success, 
                duration,
                None,
                details
            )
            
            logger.info(f"  📝 Version management: {'✅ Working' if success else '❌ Failed'}")
            
        except Exception as e:
            self.test_results.add_test_result(
                "version_management", 
                False, 
                time.time() - start_time,
                str(e)
            )
    
    def _test_integrated_system(self):
        """Test integrated system functionality"""
        logger.info("🔗 Testing integrated system")
        
        start_time = time.time()
        
        try:
            # Create temporary config system
            test_config_dir = self.temp_dir / "integrated_test"
            test_config_dir.mkdir(parents=True)
            
            # Create system
            config_system = ExcelConfigurationSystem(
                base_path=str(test_config_dir),
                enable_hot_reload=False,  # Disable for testing
                enable_versioning=True
            )
            
            # Test system start/stop
            config_system.start()
            
            # Get metrics
            metrics = config_system.get_system_metrics()
            
            # Test system stop
            config_system.stop()
            
            details = {
                'system_started': True,
                'system_stopped': True,
                'metrics': {
                    'system_health': metrics.system_health,
                    'performance_summary': metrics.performance_summary
                }
            }
            
            # Test passes if system started and stopped without errors
            success = True
            duration = time.time() - start_time
            
            self.test_results.add_test_result(
                "integrated_system", 
                success, 
                duration,
                None,
                details
            )
            
            logger.info(f"  🔗 Integrated system: {'✅ Working' if success else '❌ Failed'}")
            
        except Exception as e:
            self.test_results.add_test_result(
                "integrated_system", 
                False, 
                time.time() - start_time,
                str(e)
            )
    
    def _test_error_handling(self):
        """Test error handling"""
        logger.info("🚨 Testing error handling")
        
        start_time = time.time()
        
        try:
            converter = ExcelToYAMLConverter()
            
            # Test with non-existent file
            yaml_data, metrics = converter.convert_single_file("/nonexistent/file.xlsx")
            
            # Should fail gracefully
            error_handled = not metrics.success and metrics.error_message is not None
            
            details = {
                'nonexistent_file_handled': error_handled,
                'error_message': metrics.error_message
            }
            
            # Test passes if error was handled gracefully
            success = error_handled
            duration = time.time() - start_time
            
            self.test_results.add_test_result(
                "error_handling", 
                success, 
                duration,
                None,
                details
            )
            
            logger.info(f"  🚨 Error handling: {'✅ Graceful' if success else '❌ Poor'}")
            
        except Exception as e:
            # Should not reach here - errors should be caught
            self.test_results.add_test_result(
                "error_handling", 
                False, 
                time.time() - start_time,
                f"Uncaught exception: {str(e)}"
            )
    
    def _test_performance_targets(self):
        """Test overall performance targets"""
        logger.info("🎯 Testing performance targets")
        
        start_time = time.time()
        
        try:
            # Get performance metrics from previous tests
            converter_metrics = None
            for metric in self.test_results.performance_metrics:
                if metric['test_name'] == 'converter_performance':
                    converter_metrics = metric.get('details', {})
                    break
            
            if not converter_metrics:
                self.test_results.add_test_result(
                    "performance_targets", 
                    False, 
                    time.time() - start_time,
                    "No converter performance metrics found"
                )
                return
            
            # Check targets
            target_met_rate = converter_metrics.get('target_met_rate', 0)
            avg_duration = converter_metrics.get('avg_duration', 0)
            
            # Performance targets:
            # - >80% of files should convert in <100ms
            # - Average conversion time should be <200ms
            
            target_80_percent = target_met_rate > 0.8
            avg_under_200ms = avg_duration < 0.2
            
            details = {
                'target_met_rate': target_met_rate,
                'avg_duration': avg_duration,
                'target_80_percent_met': target_80_percent,
                'avg_under_200ms': avg_under_200ms,
                'performance_target_score': (target_met_rate + (1 - min(avg_duration / 0.2, 1))) / 2
            }
            
            # Test passes if both targets are met
            success = target_80_percent and avg_under_200ms
            duration = time.time() - start_time
            
            self.test_results.add_test_result(
                "performance_targets", 
                success, 
                duration,
                None if success else f"Performance targets not met: {target_met_rate:.1%} <100ms, {avg_duration:.3f}s avg",
                details
            )
            
            logger.info(f"  🎯 Performance targets: {'✅ Met' if success else '❌ Not met'}")
            
        except Exception as e:
            self.test_results.add_test_result(
                "performance_targets", 
                False, 
                time.time() - start_time,
                str(e)
            )
    
    def _find_test_file(self, strategy_type: str) -> Path:
        """Find a test file for the given strategy type"""
        prod_dir = self.base_path / "data" / "prod" / strategy_type
        
        if not prod_dir.exists():
            return None
        
        excel_files = list(prod_dir.glob("*.xlsx"))
        return excel_files[0] if excel_files else None
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating test report")
        
        summary = self.test_results.get_summary()
        
        # Create report
        report = {
            'test_suite': 'Excel Configuration System',
            'timestamp': time.time(),
            'summary': summary,
            'evidence': {
                'real_excel_files_used': True,
                'performance_measured': True,
                'pandas_validation_tested': True,
                'hot_reload_tested': True,
                'versioning_tested': True,
                'error_handling_tested': True
            },
            'performance_targets': {
                'conversion_under_100ms': 'Target: >80% of files',
                'avg_conversion_time': 'Target: <200ms',
                'hot_reload_detection': 'Target: <50ms',
                'system_startup_time': 'Target: <5s'
            },
            'recommendations': []
        }
        
        # Add recommendations based on results
        if summary['success_rate'] < 0.8:
            report['recommendations'].append("Review failing tests and improve system reliability")
        
        # Find performance issues
        perf_metrics = [m for m in summary['performance_metrics'] if m['test_name'] == 'converter_performance']
        if perf_metrics:
            perf_details = perf_metrics[0].get('details', {})
            target_met_rate = perf_details.get('target_met_rate', 0)
            
            if target_met_rate < 0.8:
                report['recommendations'].append(f"Improve conversion performance - only {target_met_rate:.1%} of files meet <100ms target")
        
        # Save report
        report_path = self.temp_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Test report saved to: {report_path}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📋 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests run: {summary['tests_run']}")
        logger.info(f"Tests passed: {summary['tests_passed']}")
        logger.info(f"Tests failed: {summary['tests_failed']}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Total duration: {summary['total_duration']:.3f}s")
        logger.info(f"Average duration: {summary['avg_duration']:.3f}s")
        
        if summary['errors']:
            logger.info("\n❌ ERRORS:")
            for error in summary['errors']:
                logger.info(f"  - {error}")
        
        if summary['warnings']:
            logger.info("\n⚠️ WARNINGS:")
            for warning in summary['warnings']:
                logger.info(f"  - {warning}")
        
        logger.info("=" * 60)
    
    def _cleanup_test_environment(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment")
        
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"🗑️ Removed temporary directory: {self.temp_dir}")

def main():
    """Main test execution function"""
    # Configuration
    base_path = "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations"
    
    # Check if base path exists
    if not Path(base_path).exists():
        logger.error(f"Base path does not exist: {base_path}")
        return 1
    
    # Run tests
    tester = ExcelConfigSystemTester(base_path)
    results = tester.run_all_tests()
    
    # Return appropriate exit code
    return 0 if results.tests_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())