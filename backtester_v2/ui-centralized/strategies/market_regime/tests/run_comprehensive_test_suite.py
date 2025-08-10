"""
Comprehensive Test Suite Runner
==============================

Master test runner for the entire market regime analysis system.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_base_components import *
from test_integration_layer import *
from test_indicators_comprehensive import *
from test_performance_benchmarks import *


class ComprehensiveTestRunner:
    """Comprehensive test runner with detailed reporting"""
    
    def __init__(self):
        self.test_categories = {
            'base_components': [
                'TestDataValidator',
                'TestMathUtils', 
                'TestTimeUtils',
                'TestOptionUtils',
                'TestConfigUtils',
                'TestErrorHandler',
                'TestCacheUtils'
            ],
            'integration_layer': [
                'TestMarketRegimeOrchestrator',
                'TestComponentManager',
                'TestDataPipeline',
                'TestResultAggregator'
            ],
            'indicators': [
                'TestStraddleAnalysisEngine',
                'TestOIPAAnalyzer',
                'TestGreekSentimentAnalyzer',
                'TestMarketBreadthAnalyzer',
                'TestIVAnalyticsAnalyzer',
                'TestTechnicalIndicatorsAnalyzer',
                'TestIndicatorIntegration'
            ],
            'performance': [
                'TestDataPipelinePerformance',
                'TestMathUtilsPerformance',
                'TestDataValidationPerformance',
                'TestThroughputBenchmarks',
                'TestMemoryLeakDetection',
                'TestScalabilityBenchmarks'
            ]
        }
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_category_tests(self, category: str) -> Dict[str, Any]:
        """Run tests for a specific category"""
        print(f"\n{'='*60}")
        print(f"Running {category.upper()} Tests")
        print(f"{'='*60}")
        
        category_results = {
            'category': category,
            'start_time': datetime.now(),
            'test_classes': [],
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'execution_time': 0,
            'success_rate': 0.0
        }
        
        test_suite = unittest.TestSuite()
        
        # Add tests for this category
        for test_class_name in self.test_categories[category]:
            try:
                # Get the test class from globals
                test_class = globals().get(test_class_name)
                if test_class:
                    tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                    test_suite.addTests(tests)
                    category_results['test_classes'].append(test_class_name)
                else:
                    print(f"Warning: Test class {test_class_name} not found")
            except Exception as e:
                print(f"Error loading test class {test_class_name}: {e}")
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        start_time = time.time()
        result = runner.run(test_suite)
        end_time = time.time()
        
        # Collect results
        category_results['end_time'] = datetime.now()
        category_results['execution_time'] = end_time - start_time
        category_results['total_tests'] = result.testsRun
        category_results['failed_tests'] = len(result.failures)
        category_results['error_tests'] = len(result.errors)
        category_results['passed_tests'] = result.testsRun - len(result.failures) - len(result.errors)
        
        if result.testsRun > 0:
            category_results['success_rate'] = (category_results['passed_tests'] / result.testsRun) * 100
        
        # Store detailed failure/error information
        category_results['failures'] = []
        for test, traceback in result.failures:
            category_results['failures'].append({
                'test': str(test),
                'traceback': traceback
            })
        
        category_results['errors'] = []
        for test, traceback in result.errors:
            category_results['errors'].append({
                'test': str(test),
                'traceback': traceback
            })
        
        # Print category summary
        print(f"\n{category.upper()} Category Results:")
        print(f"  Tests run: {category_results['total_tests']}")
        print(f"  Passed: {category_results['passed_tests']}")
        print(f"  Failed: {category_results['failed_tests']}")
        print(f"  Errors: {category_results['error_tests']}")
        print(f"  Success rate: {category_results['success_rate']:.1f}%")
        print(f"  Execution time: {category_results['execution_time']:.2f}s")
        
        return category_results
    
    def run_all_tests(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run all test categories"""
        self.start_time = datetime.now()
        print(f"Starting Comprehensive Test Suite at {self.start_time}")
        
        # Use all categories if none specified
        if categories is None:
            categories = list(self.test_categories.keys())
        
        # Run each category
        for category in categories:
            if category in self.test_categories:
                try:
                    category_result = self.run_category_tests(category)
                    self.results[category] = category_result
                except Exception as e:
                    print(f"Error running {category} tests: {e}")
                    self.results[category] = {
                        'category': category,
                        'error': str(e),
                        'total_tests': 0,
                        'passed_tests': 0,
                        'failed_tests': 0,
                        'error_tests': 1,
                        'success_rate': 0.0
                    }
            else:
                print(f"Warning: Unknown test category '{category}'")
        
        self.end_time = datetime.now()
        
        # Generate overall summary
        overall_summary = self.generate_summary()
        self.results['overall_summary'] = overall_summary
        
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        total_tests = sum(result.get('total_tests', 0) for result in self.results.values() if isinstance(result, dict))
        total_passed = sum(result.get('passed_tests', 0) for result in self.results.values() if isinstance(result, dict))
        total_failed = sum(result.get('failed_tests', 0) for result in self.results.values() if isinstance(result, dict))
        total_errors = sum(result.get('error_tests', 0) for result in self.results.values() if isinstance(result, dict))
        total_execution_time = sum(result.get('execution_time', 0) for result in self.results.values() if isinstance(result, dict))
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
        
        summary = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_execution_time': total_execution_time,
            'wall_clock_time': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
            'categories_run': len([r for r in self.results.values() if isinstance(r, dict) and r.get('total_tests', 0) > 0]),
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'overall_success_rate': overall_success_rate,
            'category_success_rates': {
                category: result.get('success_rate', 0.0) 
                for category, result in self.results.items() 
                if isinstance(result, dict) and 'success_rate' in result
            }
        }
        
        return summary
    
    def print_final_summary(self):
        """Print comprehensive final summary"""
        if 'overall_summary' not in self.results:
            return
        
        summary = self.results['overall_summary']
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TEST SUITE RESULTS")
        print(f"{'='*80}")
        
        print(f"Start Time: {summary['start_time']}")
        print(f"End Time: {summary['end_time']}")
        print(f"Wall Clock Time: {summary['wall_clock_time']:.2f} seconds")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds")
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Categories Run: {summary['categories_run']}")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['total_passed']}")
        print(f"  Failed: {summary['total_failed']}")
        print(f"  Errors: {summary['total_errors']}")
        print(f"  Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        
        print(f"\nCATEGORY BREAKDOWN:")
        for category, success_rate in summary['category_success_rates'].items():
            status = "✓" if success_rate == 100.0 else "⚠" if success_rate >= 80.0 else "✗"
            print(f"  {status} {category.upper()}: {success_rate:.1f}%")
        
        # Print failed tests summary
        failed_tests = []
        error_tests = []
        
        for category, result in self.results.items():
            if isinstance(result, dict):
                failed_tests.extend(result.get('failures', []))
                error_tests.extend(result.get('errors', []))
        
        if failed_tests:
            print(f"\nFAILED TESTS ({len(failed_tests)}):")
            for i, failure in enumerate(failed_tests[:5], 1):  # Show first 5
                print(f"  {i}. {failure['test']}")
            if len(failed_tests) > 5:
                print(f"  ... and {len(failed_tests) - 5} more")
        
        if error_tests:
            print(f"\nERROR TESTS ({len(error_tests)}):")
            for i, error in enumerate(error_tests[:5], 1):  # Show first 5
                print(f"  {i}. {error['test']}")
            if len(error_tests) > 5:
                print(f"  ... and {len(error_tests) - 5} more")
        
        # Overall status
        print(f"\n{'='*80}")
        if summary['overall_success_rate'] == 100.0:
            print(f"🎉 ALL TESTS PASSED! System is ready for production.")
        elif summary['overall_success_rate'] >= 95.0:
            print(f"✅ EXCELLENT! {summary['overall_success_rate']:.1f}% success rate. Minor issues to address.")
        elif summary['overall_success_rate'] >= 85.0:
            print(f"⚠️  GOOD. {summary['overall_success_rate']:.1f}% success rate. Some issues need attention.")
        elif summary['overall_success_rate'] >= 70.0:
            print(f"⚠️  FAIR. {summary['overall_success_rate']:.1f}% success rate. Significant issues to resolve.")
        else:
            print(f"❌ POOR. {summary['overall_success_rate']:.1f}% success rate. Major issues require immediate attention.")
        print(f"{'='*80}")
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_test_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_value = value.copy()
                for k, v in json_value.items():
                    if isinstance(v, datetime):
                        json_value[k] = v.isoformat()
                json_results[key] = json_value
            else:
                json_results[key] = value
        
        try:
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            print(f"\nTest results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results to {filename}: {e}")


def main():
    """Main test runner function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run comprehensive market regime test suite')
    parser.add_argument(
        '--categories', 
        nargs='+', 
        choices=['base_components', 'integration_layer', 'indicators', 'performance'],
        help='Specific test categories to run (default: all)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save test results to JSON file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Specify output file name for results'
    )
    
    args = parser.parse_args()
    
    try:
        # Run the tests
        results = runner.run_all_tests(categories=args.categories)
        
        # Print final summary
        runner.print_final_summary()
        
        # Save results if requested
        if args.save_results:
            runner.save_results(args.output_file)
        
        # Return appropriate exit code
        if 'overall_summary' in results:
            success_rate = results['overall_summary']['overall_success_rate']
            if success_rate == 100.0:
                sys.exit(0)  # Perfect success
            elif success_rate >= 95.0:
                sys.exit(0)  # Acceptable success
            else:
                sys.exit(1)  # Too many failures
        else:
            sys.exit(1)  # No results generated
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()