#!/usr/bin/env python3
"""
Production Clean Validation Script
=================================

This script validates that all production code is clean of synthetic data generation.
It checks for np.random usage and other synthetic data patterns.

Author: Claude Code
Date: 2025-07-11
Version: 1.0.0
"""

import os
import re
import sys
import logging
from typing import Dict, List, Tuple, Set
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionCleanValidator:
    """Validates production code is clean of synthetic data generation"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.violations = []
        self.critical_files = [
            'correlation_matrix_engine.py',
            'rolling_correlation_matrix_engine.py',
            'strategy.py',
            'optimized_heavydb_engine.py',
            'unified_api.py',
            'market_regime_processor.py'
        ]
        
    def validate_all_files(self) -> Dict[str, any]:
        """Validate all Python files for synthetic data usage"""
        logger.info("🔍 Starting production clean validation...")
        
        results = {
            'total_files_checked': 0,
            'files_with_violations': 0,
            'total_violations': 0,
            'critical_files_status': {},
            'violation_details': [],
            'clean_files': [],
            'test_files_ignored': 0
        }
        
        # Check all Python files
        for py_file in self.base_path.rglob('*.py'):
            if self._is_test_file(py_file):
                results['test_files_ignored'] += 1
                continue
                
            results['total_files_checked'] += 1
            violations = self._check_file_for_violations(py_file)
            
            if violations:
                results['files_with_violations'] += 1
                results['total_violations'] += len(violations)
                results['violation_details'].extend(violations)
                
                # Check if it's a critical file
                if py_file.name in self.critical_files:
                    results['critical_files_status'][py_file.name] = 'VIOLATED'
            else:
                results['clean_files'].append(str(py_file))
                if py_file.name in self.critical_files:
                    results['critical_files_status'][py_file.name] = 'CLEAN'
        
        # Check for missing critical files
        for critical_file in self.critical_files:
            if critical_file not in results['critical_files_status']:
                results['critical_files_status'][critical_file] = 'NOT_FOUND'
        
        self._log_results(results)
        return results
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file (ignore test files)"""
        test_patterns = [
            'test_',
            '_test.py',
            '/tests/',
            'demo_',
            'example_',
            'sample_'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str.lower() for pattern in test_patterns)
    
    def _check_file_for_violations(self, file_path: Path) -> List[Dict[str, any]]:
        """Check a single file for synthetic data violations"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                violations.extend(self._check_line_for_violations(
                    file_path, line_num, line
                ))
                
        except Exception as e:
            logger.warning(f"Could not check file {file_path}: {e}")
            
        return violations
    
    def _check_line_for_violations(self, file_path: Path, line_num: int, line: str) -> List[Dict[str, any]]:
        """Check a single line for violations"""
        violations = []
        
        # Patterns to detect synthetic data generation
        violation_patterns = [
            (r'np\.random\.', 'NUMPY_RANDOM_USAGE'),
            (r'numpy\.random\.', 'NUMPY_RANDOM_USAGE'),
            (r'random\.', 'RANDOM_MODULE_USAGE'),
            (r'\.random\(', 'RANDOM_CALL'),
            (r'generate.*sample', 'SAMPLE_GENERATION'),
            (r'synthetic.*data', 'SYNTHETIC_DATA_MENTION'),
            (r'mock.*data', 'MOCK_DATA_USAGE'),
            (r'fake.*data', 'FAKE_DATA_USAGE'),
            (r'simulated.*data', 'SIMULATED_DATA_USAGE'),
        ]
        
        for pattern, violation_type in violation_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Skip if it's a comment explaining the restriction
                if 'production mode' in line.lower() and 'disabled' in line.lower():
                    continue
                    
                # Skip if it's in a docstring or comment about restrictions
                if any(keyword in line.lower() for keyword in ['no synthetic', 'disabled', 'production mode']):
                    continue
                    
                violations.append({
                    'file': str(file_path),
                    'line_number': line_num,
                    'line_content': line.strip(),
                    'violation_type': violation_type,
                    'pattern': pattern
                })
                
        return violations
    
    def _log_results(self, results: Dict[str, any]):
        """Log validation results"""
        logger.info("📊 Production Clean Validation Results:")
        logger.info(f"   📁 Total files checked: {results['total_files_checked']}")
        logger.info(f"   🧪 Test files ignored: {results['test_files_ignored']}")
        logger.info(f"   ✅ Clean files: {len(results['clean_files'])}")
        logger.info(f"   ❌ Files with violations: {results['files_with_violations']}")
        logger.info(f"   🚨 Total violations: {results['total_violations']}")
        
        logger.info("\n🔍 Critical Files Status:")
        for file_name, status in results['critical_files_status'].items():
            if status == 'CLEAN':
                logger.info(f"   ✅ {file_name}: CLEAN")
            elif status == 'VIOLATED':
                logger.error(f"   ❌ {file_name}: VIOLATIONS FOUND")
            else:
                logger.warning(f"   ⚠️ {file_name}: NOT FOUND")
        
        if results['violation_details']:
            logger.error("\n🚨 Violation Details:")
            for violation in results['violation_details']:
                logger.error(f"   File: {violation['file']}")
                logger.error(f"   Line {violation['line_number']}: {violation['line_content']}")
                logger.error(f"   Type: {violation['violation_type']}")
                logger.error("   " + "-" * 50)
    
    def is_production_clean(self) -> bool:
        """Check if all critical files are clean"""
        results = self.validate_all_files()
        
        # Check critical files
        critical_clean = all(
            results['critical_files_status'].get(file, 'NOT_FOUND') == 'CLEAN'
            for file in self.critical_files
        )
        
        # Check overall violations
        no_violations = results['total_violations'] == 0
        
        return critical_clean and no_violations

def main():
    """Main validation function"""
    base_path = '/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/strategies/market_regime'
    
    validator = ProductionCleanValidator(base_path)
    
    logger.info("🧹 Starting Production Clean Validation")
    logger.info("=" * 60)
    
    if validator.is_production_clean():
        logger.info("✅ SUCCESS: All production code is clean of synthetic data generation!")
        sys.exit(0)
    else:
        logger.error("❌ FAILURE: Production code contains synthetic data generation violations!")
        sys.exit(1)

if __name__ == "__main__":
    main()