#!/usr/bin/env python3
"""
Run Comprehensive Tests for Market Regime Indicators V2
======================================================

This script runs comprehensive tests for the implemented indicators:
- Greek Sentiment V2
- OI/PA Analysis V2 (Trending OI with PA)

The tests use real HeavyDB data and validate against PHASE2 Excel configuration.

Author: Market Regime Testing Team
Date: 2025-07-06
"""

import sys
import os
from pathlib import Path
import logging
import time
from datetime import datetime
import json

# Setup logging
log_dir = Path(__file__).parent / "test_logs"
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"comprehensive_test_run_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import test modules
try:
    from test_greek_sentiment_v2_comprehensive import run_comprehensive_tests as run_greek_tests
    from test_oi_pa_analysis_v2_comprehensive import run_comprehensive_tests as run_oi_pa_tests
except ImportError as e:
    logger.error(f"Failed to import test modules: {e}")
    sys.exit(1)


def print_banner(text):
    """Print a formatted banner"""
    width = 80
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def run_all_tests():
    """Run all comprehensive tests and generate report"""
    print_banner("MARKET REGIME INDICATORS V2 - COMPREHENSIVE TEST SUITE")
    
    # Test results storage
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'test_suites': {},
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0
        }
    }
    
    # Configure test environment
    logger.info("Configuring test environment...")
    logger.info(f"Log file: {log_file}")
    
    # Check HeavyDB availability
    try:
        from base.heavydb_connector import HeavyDBConnector
        db = HeavyDBConnector()
        if db.connect():
            logger.info("✅ HeavyDB connection available")
            db.disconnect()
        else:
            logger.warning("⚠️ HeavyDB connection not available - tests will use mock data")
    except Exception as e:
        logger.warning(f"⚠️ Could not check HeavyDB connection: {e}")
    
    # Run Greek Sentiment V2 tests
    print("\n")
    print_banner("GREEK SENTIMENT V2 - COMPREHENSIVE TESTS")
    
    start_time = time.time()
    try:
        greek_success = run_greek_tests()
        greek_elapsed = time.time() - start_time
        
        test_results['test_suites']['greek_sentiment_v2'] = {
            'success': greek_success,
            'elapsed_time': greek_elapsed,
            'status': 'PASSED' if greek_success else 'FAILED'
        }
        
        if greek_success:
            logger.info(f"✅ Greek Sentiment V2 tests PASSED in {greek_elapsed:.2f}s")
        else:
            logger.error(f"❌ Greek Sentiment V2 tests FAILED in {greek_elapsed:.2f}s")
            
    except Exception as e:
        logger.error(f"❌ Greek Sentiment V2 tests ERROR: {e}")
        test_results['test_suites']['greek_sentiment_v2'] = {
            'success': False,
            'error': str(e),
            'status': 'ERROR'
        }
    
    # Run OI/PA Analysis V2 tests
    print("\n")
    print_banner("OI/PA ANALYSIS V2 - COMPREHENSIVE TESTS")
    
    start_time = time.time()
    try:
        oi_pa_success = run_oi_pa_tests()
        oi_pa_elapsed = time.time() - start_time
        
        test_results['test_suites']['oi_pa_analysis_v2'] = {
            'success': oi_pa_success,
            'elapsed_time': oi_pa_elapsed,
            'status': 'PASSED' if oi_pa_success else 'FAILED'
        }
        
        if oi_pa_success:
            logger.info(f"✅ OI/PA Analysis V2 tests PASSED in {oi_pa_elapsed:.2f}s")
        else:
            logger.error(f"❌ OI/PA Analysis V2 tests FAILED in {oi_pa_elapsed:.2f}s")
            
    except Exception as e:
        logger.error(f"❌ OI/PA Analysis V2 tests ERROR: {e}")
        test_results['test_suites']['oi_pa_analysis_v2'] = {
            'success': False,
            'error': str(e),
            'status': 'ERROR'
        }
    
    # Generate summary
    print("\n")
    print_banner("COMPREHENSIVE TEST SUMMARY")
    
    all_passed = all(
        suite.get('success', False) 
        for suite in test_results['test_suites'].values()
    )
    
    # Calculate totals
    total_suites = len(test_results['test_suites'])
    passed_suites = sum(1 for s in test_results['test_suites'].values() if s.get('success', False))
    failed_suites = total_suites - passed_suites
    
    test_results['summary']['total_suites'] = total_suites
    test_results['summary']['passed_suites'] = passed_suites
    test_results['summary']['failed_suites'] = failed_suites
    test_results['summary']['overall_success'] = all_passed
    
    # Display summary
    print(f"\nTest Suites Run: {total_suites}")
    print(f"Passed: {passed_suites}")
    print(f"Failed: {failed_suites}")
    print(f"\nDetailed Results:")
    
    for suite_name, result in test_results['test_suites'].items():
        status_icon = "✅" if result.get('success', False) else "❌"
        elapsed = result.get('elapsed_time', 0)
        print(f"  {status_icon} {suite_name}: {result['status']} ({elapsed:.2f}s)")
        if 'error' in result:
            print(f"     Error: {result['error']}")
    
    # Save results to JSON
    results_file = log_dir / f"test_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    print(f"Test log saved to: {log_file}")
    
    # Final verdict
    print("\n")
    if all_passed:
        print_banner("🎉 ALL TESTS PASSED! 🎉")
        print("\nThe implemented indicators are ready for use:")
        print("  ✅ Greek Sentiment V2 - Fully tested with all components")
        print("  ✅ OI/PA Analysis V2 - Fully tested with all components")
        print("\nNext step: Implement Technical Indicators V2")
    else:
        print_banner("❌ SOME TESTS FAILED ❌")
        print("\nPlease review the logs and fix any issues before proceeding.")
    
    return all_passed


def main():
    """Main entry point"""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()