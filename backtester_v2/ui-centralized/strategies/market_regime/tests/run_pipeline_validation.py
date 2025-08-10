#!/usr/bin/env python3
"""
Pipeline Validation Test Runner
===============================

This script runs all E2E pipeline validation tests to ensure
strict NO MOCK data enforcement throughout the system.

Usage:
    python3 run_pipeline_validation.py

Author: SuperClaude Testing Framework
Date: 2025-07-11
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test_suite(test_file, description):
    """Run a specific test suite"""
    logger.info(f"Running {description}...")
    logger.info(f"Test file: {test_file}")
    
    try:
        # Change to test directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(test_dir)
        
        # Run the test
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Log results
        if result.returncode == 0:
            logger.info(f"✅ {description} - PASSED")
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Main test runner"""
    logger.info("=" * 80)
    logger.info("E2E PIPELINE VALIDATION TEST RUNNER")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Test suites to run
    test_suites = [
        {
            'file': 'test_e2e_pipeline_strict_no_mock.py',
            'description': 'Strict NO MOCK Data Enforcement Test'
        },
        {
            'file': 'test_e2e_practical_no_mock.py',
            'description': 'Practical E2E Pipeline Test'
        },
        {
            'file': 'test_e2e_final_validation.py',
            'description': 'Final Comprehensive Validation Test'
        }
    ]
    
    # Run all test suites
    results = []
    for suite in test_suites:
        logger.info("-" * 60)
        result = run_test_suite(suite['file'], suite['description'])
        results.append({
            'suite': suite['description'],
            'passed': result
        })
    
    # Summary
    logger.info("=" * 80)
    logger.info("PIPELINE VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    total_suites = len(results)
    passed_suites = sum(1 for r in results if r['passed'])
    
    logger.info(f"Total test suites: {total_suites}")
    logger.info(f"Passed: {passed_suites}")
    logger.info(f"Failed: {total_suites - passed_suites}")
    logger.info(f"Success rate: {(passed_suites / total_suites) * 100:.1f}%")
    
    # Individual results
    for result in results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        logger.info(f"{status}: {result['suite']}")
    
    logger.info("=" * 80)
    
    # Final assessment
    if passed_suites == total_suites:
        logger.info("🎯 ALL PIPELINE VALIDATION TESTS PASSED")
        logger.info("🔒 NO MOCK DATA ENFORCEMENT CONFIRMED")
        logger.info("🚀 PIPELINE INTEGRITY VALIDATED")
        logger.info("📊 EXCEL → HEAVYDB → PROCESSING → OUTPUT VERIFIED")
        exit_code = 0
    else:
        logger.error("❌ SOME PIPELINE VALIDATION TESTS FAILED")
        logger.error("⚠️ PIPELINE INTEGRITY ISSUES DETECTED")
        exit_code = 1
    
    logger.info(f"Completed at: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)