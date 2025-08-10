"""
Test Runner for Optimization Module

Comprehensive test runner that executes all test suites,
generates coverage reports, and provides detailed test results.
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import logging

# Add the optimization module to path
optimization_root = Path(__file__).parent.parent
sys.path.insert(0, str(optimization_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationTestRunner:
    """
    Comprehensive test runner for the optimization module
    
    Features:
    - Runs all test suites with coverage analysis
    - Generates detailed test reports
    - Provides performance benchmarking
    - Creates HTML coverage reports
    - Validates all components
    """
    
    def __init__(self, 
                 test_directory: Optional[str] = None,
                 coverage_threshold: float = 0.8,
                 generate_html_report: bool = True,
                 run_performance_tests: bool = True):
        """
        Initialize test runner
        
        Args:
            test_directory: Directory containing tests (default: current directory)
            coverage_threshold: Minimum coverage threshold (0.0-1.0)
            generate_html_report: Generate HTML coverage report
            run_performance_tests: Run performance benchmarks
        """
        self.test_directory = Path(test_directory) if test_directory else Path(__file__).parent
        self.coverage_threshold = coverage_threshold
        self.generate_html_report = generate_html_report
        self.run_performance_tests = run_performance_tests
        
        # Test results storage
        self.test_results: Dict[str, Any] = {}
        self.coverage_results: Dict[str, Any] = {}
        self.performance_results: Dict[str, Any] = {}
        
        logger.info(f"OptimizationTestRunner initialized")
        logger.info(f"Test directory: {self.test_directory}")
        logger.info(f"Coverage threshold: {self.coverage_threshold}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test suites
        
        Returns:
            Comprehensive test results
        """
        logger.info("Starting comprehensive test suite execution")
        start_time = time.time()
        
        try:
            # Run core test suites
            self._run_core_tests()
            
            # Run integration tests
            self._run_integration_tests()
            
            # Run performance tests
            if self.run_performance_tests:
                self._run_performance_tests()
            
            # Generate coverage report
            self._generate_coverage_report()
            
            # Compile final results
            execution_time = time.time() - start_time
            final_results = self._compile_final_results(execution_time)
            
            logger.info(f"Test suite completed in {execution_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            raise
    
    def _run_core_tests(self):
        """Run core component test suites"""
        logger.info("Running core component tests")
        
        core_test_modules = [
            "test_base_optimizer.py",
            "test_algorithm_registry.py", 
            "test_optimization_engine.py",
            "test_gpu_acceleration.py",
            "test_robustness_framework.py",
            "test_inversion_system.py"
        ]
        
        for test_module in core_test_modules:
            module_name = test_module.replace('.py', '')
            logger.info(f"Running {module_name} tests")
            
            try:
                # Run pytest for specific module
                result = self._run_pytest_module(test_module)
                self.test_results[module_name] = result
                
                if result['passed']:
                    logger.info(f"✅ {module_name}: {result['summary']}")
                else:
                    logger.warning(f"❌ {module_name}: {result['summary']}")
                    
            except Exception as e:
                logger.error(f"Failed to run {module_name}: {e}")
                self.test_results[module_name] = {
                    'passed': False,
                    'error': str(e),
                    'summary': f"Failed with exception: {e}"
                }
    
    def _run_integration_tests(self):
        """Run integration tests"""
        logger.info("Running integration tests")
        
        try:
            # Create integration test suite
            integration_results = self._run_integration_test_suite()
            self.test_results['integration_tests'] = integration_results
            
            if integration_results['passed']:
                logger.info("✅ Integration tests passed")
            else:
                logger.warning("❌ Integration tests failed")
                
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            self.test_results['integration_tests'] = {
                'passed': False,
                'error': str(e)
            }
    
    def _run_performance_tests(self):
        """Run performance benchmark tests"""
        logger.info("Running performance benchmark tests")
        
        try:
            # Algorithm performance benchmarks
            algorithm_benchmarks = self._benchmark_algorithms()
            
            # System performance benchmarks
            system_benchmarks = self._benchmark_system_performance()
            
            self.performance_results = {
                'algorithm_benchmarks': algorithm_benchmarks,
                'system_benchmarks': system_benchmarks
            }
            
            logger.info("✅ Performance benchmarks completed")
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            self.performance_results = {
                'error': str(e),
                'benchmarks_completed': False
            }
    
    def _run_pytest_module(self, test_module: str) -> Dict[str, Any]:
        """Run pytest for a specific module"""
        test_file = self.test_directory / test_module
        
        if not test_file.exists():
            return {
                'passed': False,
                'error': f"Test file {test_file} not found",
                'summary': "Test file not found"
            }
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=short',
                '--cov=strategies.optimization',
                '--cov-report=term-missing',
                '--cov-append'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.test_directory.parent.parent
            )
            
            # Parse results
            passed = result.returncode == 0
            output_lines = result.stdout.split('\n')
            
            # Extract test statistics
            stats = self._parse_pytest_output(output_lines)
            
            return {
                'passed': passed,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'statistics': stats,
                'summary': self._generate_test_summary(stats)
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'summary': f"Exception during test execution: {e}"
            }
    
    def _parse_pytest_output(self, output_lines: List[str]) -> Dict[str, Any]:
        """Parse pytest output to extract statistics"""
        stats = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0,
            'execution_time': 0.0
        }
        
        for line in output_lines:
            # Look for summary line
            if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                # Parse summary line like "5 passed, 2 failed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        stats['passed'] = int(parts[i-1])
                    elif part == 'failed' and i > 0:
                        stats['failed'] = int(parts[i-1])
                    elif part == 'skipped' and i > 0:
                        stats['skipped'] = int(parts[i-1])
                    elif part == 'error' and i > 0:
                        stats['errors'] = int(parts[i-1])
                    elif 'in' in parts and i < len(parts) - 1 and 's' in parts[i+1]:
                        try:
                            stats['execution_time'] = float(parts[i+1].replace('s', ''))
                        except ValueError:
                            pass
            
            # Count warnings
            if 'warning' in line.lower():
                stats['warnings'] += 1
        
        stats['total_tests'] = stats['passed'] + stats['failed'] + stats['skipped'] + stats['errors']
        return stats
    
    def _generate_test_summary(self, stats: Dict[str, Any]) -> str:
        """Generate human-readable test summary"""
        total = stats['total_tests']
        passed = stats['passed']
        failed = stats['failed']
        skipped = stats['skipped']
        
        if total == 0:
            return "No tests found"
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        summary = f"{passed}/{total} tests passed ({success_rate:.1f}%)"
        
        if failed > 0:
            summary += f", {failed} failed"
        if skipped > 0:
            summary += f", {skipped} skipped"
        
        summary += f" in {stats['execution_time']:.2f}s"
        
        return summary
    
    def _run_integration_test_suite(self) -> Dict[str, Any]:
        """Run integration tests that verify component interactions"""
        logger.info("Executing integration test scenarios")
        
        integration_tests = [
            self._test_optimization_engine_integration,
            self._test_gpu_robustness_integration,
            self._test_inversion_risk_integration,
            self._test_end_to_end_workflow
        ]
        
        results = []
        for test_func in integration_tests:
            try:
                test_name = test_func.__name__
                logger.info(f"Running {test_name}")
                
                start_time = time.time()
                test_result = test_func()
                execution_time = time.time() - start_time
                
                results.append({
                    'test_name': test_name,
                    'passed': test_result.get('passed', False),
                    'execution_time': execution_time,
                    'details': test_result
                })
                
            except Exception as e:
                logger.error(f"Integration test {test_func.__name__} failed: {e}")
                results.append({
                    'test_name': test_func.__name__,
                    'passed': False,
                    'error': str(e),
                    'execution_time': 0.0
                })
        
        # Calculate overall success
        passed_tests = sum(1 for r in results if r['passed'])
        total_tests = len(results)
        
        return {
            'passed': passed_tests == total_tests,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'detailed_results': results
        }
    
    def _test_optimization_engine_integration(self) -> Dict[str, Any]:
        """Test optimization engine integration with registry and algorithms"""
        try:
            from ..engines.optimization_engine import OptimizationEngine
            
            # Test engine initialization
            engine = OptimizationEngine()
            
            # Test algorithm discovery
            discovery_result = engine.registry.discover_algorithms()
            
            # Test basic optimization
            param_space = {'x': (-1, 1), 'y': (-1, 1)}
            objective = lambda p: p['x']**2 + p['y']**2
            
            result = engine.optimize(param_space, objective, max_iterations=10)
            
            return {
                'passed': True,
                'algorithms_discovered': discovery_result.get('total_algorithms', 0),
                'optimization_completed': hasattr(result, 'best_parameters')
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_gpu_robustness_integration(self) -> Dict[str, Any]:
        """Test GPU acceleration with robustness framework"""
        try:
            from ..robustness.robust_optimizer import RobustOptimizer
            from ..gpu.gpu_optimizer import GPUOptimizer
            
            # Create GPU optimizer
            param_space = {'x': (-2, 2), 'y': (-2, 2)}
            objective = lambda p: p['x']**2 + p['y']**2
            
            gpu_optimizer = GPUOptimizer(param_space, objective)
            
            # Wrap with robust optimizer
            robust_optimizer = RobustOptimizer(
                base_optimizer=gpu_optimizer,
                cv_folds=2,
                noise_levels=[0.01]
            )
            
            # Run robust optimization
            result = robust_optimizer.optimize(n_iterations=10)
            
            return {
                'passed': True,
                'robustness_completed': hasattr(result, 'robustness_metrics'),
                'gpu_integration': True
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_inversion_risk_integration(self) -> Dict[str, Any]:
        """Test strategy inversion with risk analysis"""
        try:
            from ..inversion.inversion_engine import InversionEngine
            import pandas as pd
            import numpy as np
            
            # Create sample strategy data
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            strategy_data = pd.DataFrame({
                'strategy_1': np.random.normal(0.001, 0.02, 100),
                'strategy_2': np.random.normal(0.0005, 0.015, 100)
            }, index=dates)
            
            # Run inversion analysis
            engine = InversionEngine()
            result = engine.analyze_and_invert_portfolio(
                strategy_data=strategy_data,
                strategy_columns=['strategy_1', 'strategy_2']
            )
            
            return {
                'passed': True,
                'strategies_analyzed': len(result.inversion_results),
                'risk_assessments': len(result.risk_assessments),
                'recommendations_generated': len(result.recommendations)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end optimization workflow"""
        try:
            from ..engines.optimization_engine import OptimizationEngine
            
            # Initialize engine
            engine = OptimizationEngine()
            
            # Define optimization problem
            param_space = {'x': (-5, 5), 'y': (-3, 3), 'z': (-2, 2)}
            
            def complex_objective(params):
                x, y, z = params['x'], params['y'], params['z']
                return x**2 + y**2 + z**2 + 0.1*x*y + 0.05*y*z
            
            # Run optimization with all features
            result = engine.optimize(
                param_space=param_space,
                objective_function=complex_objective,
                optimization_mode='balanced',
                enable_gpu=True,
                enable_robustness=True,
                max_iterations=20
            )
            
            # Test batch optimization
            requests = [
                engine.OptimizationRequest(param_space, complex_objective)
                for _ in range(3)
            ]
            
            batch_result = engine.batch_optimize(requests)
            
            return {
                'passed': True,
                'single_optimization': hasattr(result, 'best_parameters'),
                'batch_optimization': batch_result.total_optimizations == 3,
                'success_rate': batch_result.successful_optimizations / batch_result.total_optimizations
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _benchmark_algorithms(self) -> Dict[str, Any]:
        """Benchmark algorithm performance"""
        logger.info("Benchmarking algorithm performance")
        
        try:
            from ..engines.optimization_engine import OptimizationEngine
            
            engine = OptimizationEngine()
            
            # Define benchmark problems
            benchmark_problems = [
                # Sphere function
                ({'x': (-5, 5), 'y': (-5, 5)}, lambda p: p['x']**2 + p['y']**2),
                
                # Rosenbrock function
                ({'x': (-2, 2), 'y': (-1, 3)}, 
                 lambda p: 100*(p['y'] - p['x']**2)**2 + (1 - p['x'])**2),
                
                # Rastrigin function
                ({'x': (-5.12, 5.12), 'y': (-5.12, 5.12)},
                 lambda p: 20 + p['x']**2 - 10*np.cos(2*np.pi*p['x']) + 
                          p['y']**2 - 10*np.cos(2*np.pi*p['y']))
            ]
            
            # Run benchmarks
            benchmark_results = engine.benchmark_algorithms(
                test_functions=benchmark_problems,
                algorithms=None,  # Test all available
                iterations_per_test=3
            )
            
            return {
                'completed': True,
                'algorithms_tested': benchmark_results['algorithms_tested'],
                'test_functions': benchmark_results['test_functions'],
                'summary': benchmark_results['summary']
            }
            
        except Exception as e:
            logger.error(f"Algorithm benchmarking failed: {e}")
            return {'completed': False, 'error': str(e)}
    
    def _benchmark_system_performance(self) -> Dict[str, Any]:
        """Benchmark overall system performance"""
        logger.info("Benchmarking system performance")
        
        try:
            # Memory usage benchmark
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # CPU usage benchmark
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Algorithm loading benchmark
            start_time = time.time()
            from ..engines.algorithm_registry import AlgorithmRegistry
            registry = AlgorithmRegistry()
            discovery_result = registry.discover_algorithms()
            discovery_time = time.time() - start_time
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'cpu_percent': cpu_percent,
                'algorithm_discovery_time': discovery_time,
                'algorithms_discovered': discovery_result.get('total_algorithms', 0)
            }
            
        except Exception as e:
            logger.error(f"System benchmarking failed: {e}")
            return {'error': str(e)}
    
    def _generate_coverage_report(self):
        """Generate test coverage report"""
        logger.info("Generating coverage report")
        
        try:
            # Run coverage combine and report
            coverage_cmd = [
                sys.executable, '-m', 'coverage', 'combine'
            ]
            subprocess.run(coverage_cmd, cwd=self.test_directory.parent.parent)
            
            # Generate text report
            report_cmd = [
                sys.executable, '-m', 'coverage', 'report',
                '--include=strategies/optimization/*'
            ]
            result = subprocess.run(
                report_cmd,
                capture_output=True,
                text=True,
                cwd=self.test_directory.parent.parent
            )
            
            self.coverage_results = {
                'text_report': result.stdout,
                'coverage_data': self._parse_coverage_report(result.stdout)
            }
            
            # Generate HTML report if requested
            if self.generate_html_report:
                html_cmd = [
                    sys.executable, '-m', 'coverage', 'html',
                    '--include=strategies/optimization/*',
                    '--directory=htmlcov'
                ]
                subprocess.run(html_cmd, cwd=self.test_directory.parent.parent)
                self.coverage_results['html_report_generated'] = True
            
        except Exception as e:
            logger.error(f"Coverage report generation failed: {e}")
            self.coverage_results = {'error': str(e)}
    
    def _parse_coverage_report(self, coverage_text: str) -> Dict[str, Any]:
        """Parse coverage report text to extract metrics"""
        lines = coverage_text.split('\n')
        coverage_data = {}
        
        for line in lines:
            if 'strategies/optimization' in line:
                parts = line.split()
                if len(parts) >= 4:
                    module_name = parts[0]
                    try:
                        coverage_percent = int(parts[-1].replace('%', ''))
                        coverage_data[module_name] = coverage_percent
                    except (ValueError, IndexError):
                        pass
        
        # Calculate overall coverage
        if coverage_data:
            overall_coverage = sum(coverage_data.values()) / len(coverage_data)
            coverage_data['overall_coverage'] = overall_coverage
        
        return coverage_data
    
    def _compile_final_results(self, execution_time: float) -> Dict[str, Any]:
        """Compile final comprehensive test results"""
        
        # Calculate overall test statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for module_name, results in self.test_results.items():
            if isinstance(results, dict) and 'statistics' in results:
                stats = results['statistics']
                total_tests += stats.get('total_tests', 0)
                passed_tests += stats.get('passed', 0)
                failed_tests += stats.get('failed', 0)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Check coverage threshold
        overall_coverage = self.coverage_results.get('coverage_data', {}).get('overall_coverage', 0)
        coverage_threshold_met = overall_coverage >= (self.coverage_threshold * 100)
        
        # Determine overall test status
        overall_passed = (
            success_rate >= 95 and  # At least 95% test success
            coverage_threshold_met and  # Coverage threshold met
            failed_tests == 0  # No failed tests
        )
        
        final_results = {
            'summary': {
                'overall_passed': overall_passed,
                'execution_time': execution_time,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'coverage_threshold_met': coverage_threshold_met,
                'overall_coverage': overall_coverage
            },
            'detailed_results': {
                'test_results': self.test_results,
                'coverage_results': self.coverage_results,
                'performance_results': self.performance_results
            },
            'recommendations': self._generate_recommendations(overall_passed)
        }
        
        return final_results
    
    def _generate_recommendations(self, overall_passed: bool) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not overall_passed:
            recommendations.append("❌ Overall test suite failed - review failed tests")
        else:
            recommendations.append("✅ All tests passed successfully")
        
        # Coverage recommendations
        overall_coverage = self.coverage_results.get('coverage_data', {}).get('overall_coverage', 0)
        if overall_coverage < (self.coverage_threshold * 100):
            recommendations.append(f"📊 Increase test coverage from {overall_coverage:.1f}% to {self.coverage_threshold*100}%")
        
        # Performance recommendations
        if self.performance_results and 'error' not in self.performance_results:
            algorithm_benchmarks = self.performance_results.get('algorithm_benchmarks', {})
            if algorithm_benchmarks.get('completed'):
                recommendations.append("🚀 Performance benchmarks completed successfully")
        
        # Component-specific recommendations
        for module_name, results in self.test_results.items():
            if isinstance(results, dict) and not results.get('passed', False):
                recommendations.append(f"🔧 Fix issues in {module_name}")
        
        return recommendations
    
    def save_results(self, output_file: str):
        """Save test results to file"""
        try:
            if not hasattr(self, '_final_results'):
                logger.warning("No final results to save - run tests first")
                return
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self._final_results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def run_comprehensive_tests():
    """Run comprehensive test suite with default settings"""
    runner = OptimizationTestRunner()
    results = runner.run_all_tests()
    
    # Store results for saving
    runner._final_results = results
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION MODULE TEST SUITE RESULTS")
    print("="*80)
    
    summary = results['summary']
    
    if summary['overall_passed']:
        print("🎉 OVERALL STATUS: PASSED")
    else:
        print("❌ OVERALL STATUS: FAILED")
    
    print(f"\n📊 TEST STATISTICS:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Execution Time: {summary['execution_time']:.2f}s")
    
    print(f"\n📈 COVERAGE:")
    print(f"   Overall Coverage: {summary['overall_coverage']:.1f}%")
    print(f"   Threshold Met: {'✅' if summary['coverage_threshold_met'] else '❌'}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in results['recommendations']:
        print(f"   {rec}")
    
    print("\n" + "="*80)
    
    # Save results
    results_file = runner.test_directory / "test_results.json"
    runner.save_results(str(results_file))
    
    return results


if __name__ == "__main__":
    # Run tests when script is executed directly
    import numpy as np  # Required for tests
    
    try:
        results = run_comprehensive_tests()
        
        # Exit with appropriate code
        if results['summary']['overall_passed']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        sys.exit(2)