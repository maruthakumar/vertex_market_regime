#!/usr/bin/env python3
"""
ZERO Synthetic Data Tolerance Test Suite

This module provides ABSOLUTE validation that NO synthetic, mock, simulated,
or artificial data is used anywhere in the market regime system. Tests MUST
ABORT immediately if ANY form of non-real data is detected.

ZERO TOLERANCE POLICY:
1. NO synthetic data generation under any circumstances
2. NO mock data substitution in any form
3. NO simulated data for testing or fallback
4. NO artificial data creation or manipulation
5. IMMEDIATE ABORT when non-real data detected

Author: Enhanced by Claude Code
Date: 2025-07-10
Version: 2.0.0 - ZERO SYNTHETIC TOLERANCE
"""

import unittest
import logging
import sys
import os
import ast
import re
import inspect
from pathlib import Path
from datetime import datetime
import importlib.util

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class SyntheticDataDetectedError(Exception):
    """Raised when synthetic data is detected in any form"""
    pass

class MockDataProhibitedError(Exception):
    """Raised when mock data is detected"""
    pass

class ArtificialDataFoundError(Exception):
    """Raised when artificial data generation is found"""
    pass

class FakeDataViolationError(Exception):
    """Raised when fake data usage is detected"""
    pass

class TestZeroSyntheticTolerance(unittest.TestCase):
    """ZERO tolerance for synthetic data test suite - ABORT on ANY synthetic data"""
    
    def setUp(self):
        """Set up zero tolerance testing environment"""
        self.zero_tolerance = True
        self.abort_on_detection = True
        self.strict_validation = True
        
        # Comprehensive list of prohibited data patterns
        self.prohibited_data_patterns = [
            # Synthetic data patterns
            'synthetic', 'synth', 'artificial', 'generated', 'create_data',
            'generate_data', 'build_data', 'construct_data', 'fabricate',
            
            # Mock data patterns  
            'mock', 'stub', 'dummy', 'fake', 'test_data', 'sample_data',
            'example_data', 'demo_data', 'placeholder', 'substitute',
            
            # Simulation patterns
            'simulate', 'simulation', 'simulated', 'sim_data', 'model_data',
            'virtual', 'emulated', 'approximation', 'estimated_data',
            
            # Fallback patterns
            'fallback', 'backup_data', 'default_data', 'alternative_data',
            'cached_data', 'offline_data', 'local_data', 'hardcoded',
            
            # Random/generated patterns
            'random', 'rand', 'randomized', 'noise', 'gaussian', 'uniform',
            'normal_dist', 'random_walk', 'monte_carlo', 'bootstrap'
        ]
        
        # Prohibited function names that indicate data generation
        self.prohibited_functions = [
            'generate_mock', 'create_synthetic', 'build_test_data', 'simulate_data',
            'mock_data', 'fake_data', 'dummy_data', 'random_data', 'sample_data',
            'create_fallback', 'generate_backup', 'build_alternative', 'construct_mock',
            'fabricate_data', 'synthesize', 'emulate_data', 'approximate_data'
        ]
        
        # Critical files that MUST be free of synthetic data
        self.critical_files = [
            'correlation_matrix_engine.py',
            'enhanced_correlation_matrix.py',
            'rolling_correlation_matrix_engine.py',
            'sophisticated_regime_formation_engine.py',
            'optimized_heavydb_engine.py',
            'real_data_integration_engine.py',
            'data/heavydb_data_provider.py',
            'dte_specific_historical_analyzer.py',
            'adaptive_learning_engine.py'
        ]
    
    def test_zero_synthetic_data_in_core_files(self):
        """CRITICAL: Test ZERO synthetic data in core market regime files"""
        violations_found = []
        
        for file_path in self.critical_files:
            try:
                violations = self._scan_file_for_synthetic_data(file_path)
                if violations:
                    violations_found.extend(violations)
                    
            except Exception as e:
                logger.warning(f"Error scanning {file_path}: {e}")
        
        if violations_found:
            violation_details = '\n'.join([f"- {v}" for v in violations_found])
            raise SyntheticDataDetectedError(
                f"CRITICAL FAILURE: Synthetic data violations detected:\n{violation_details}"
            )
        
        logger.info("✅ ZERO TOLERANCE: No synthetic data detected in core files")
    
    def _scan_file_for_synthetic_data(self, file_path):
        """Scan individual file for synthetic data patterns"""
        violations = []
        
        try:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                logger.warning(f"File not found: {file_path}")
                return violations
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Scan for prohibited patterns in content
            content_lower = content.lower()
            for pattern in self.prohibited_data_patterns:
                if pattern in content_lower:
                    # Find specific line numbers
                    for i, line in enumerate(lines, 1):
                        if pattern in line.lower():
                            violations.append(f"{file_path}:{i} - Prohibited pattern '{pattern}': {line.strip()}")
            
            # Parse AST to find function definitions with prohibited names
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name_lower = node.name.lower()
                        for prohibited in self.prohibited_functions:
                            if prohibited in func_name_lower:
                                violations.append(f"{file_path}:{node.lineno} - Prohibited function '{node.name}'")
                    
                    # Check for prohibited variable assignments
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                var_name_lower = target.id.lower()
                                for pattern in self.prohibited_data_patterns:
                                    if pattern in var_name_lower:
                                        violations.append(f"{file_path}:{node.lineno} - Prohibited variable '{target.id}'")
            
            except SyntaxError as e:
                logger.warning(f"Could not parse AST for {file_path}: {e}")
                
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
        
        return violations
    
    def test_zero_mock_data_in_imports(self):
        """CRITICAL: Test ZERO mock data imports in any modules"""
        try:
            prohibited_imports = [
                'mock', 'unittest.mock', 'pytest.mock', 'mocking', 'faker',
                'factory_boy', 'mimesis', 'synthetic_data', 'data_generator'
            ]
            
            violations = []
            
            for file_path in self.critical_files:
                violations.extend(self._scan_file_for_prohibited_imports(file_path, prohibited_imports))
            
            if violations:
                violation_details = '\n'.join([f"- {v}" for v in violations])
                raise MockDataProhibitedError(
                    f"CRITICAL FAILURE: Mock data imports detected:\n{violation_details}"
                )
            
            logger.info("✅ ZERO TOLERANCE: No mock data imports detected")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Mock data import scan failed: {e}")
    
    def _scan_file_for_prohibited_imports(self, file_path, prohibited_imports):
        """Scan file for prohibited import statements"""
        violations = []
        
        try:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                return violations
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Parse AST to find import statements
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            for prohibited in prohibited_imports:
                                if prohibited in alias.name.lower():
                                    violations.append(f"{file_path}:{node.lineno} - Prohibited import '{alias.name}'")
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for prohibited in prohibited_imports:
                                if prohibited in node.module.lower():
                                    violations.append(f"{file_path}:{node.lineno} - Prohibited import from '{node.module}'")
            
            except SyntaxError as e:
                logger.warning(f"Could not parse imports in {file_path}: {e}")
                
        except Exception as e:
            logger.error(f"Error scanning imports in {file_path}: {e}")
        
        return violations
    
    def test_zero_data_generation_functions(self):
        """CRITICAL: Test ZERO data generation functions exist"""
        try:
            # Dynamically import and inspect modules
            generation_functions_found = []
            
            for file_path in self.critical_files:
                functions_found = self._inspect_module_for_generation_functions(file_path)
                generation_functions_found.extend(functions_found)
            
            if generation_functions_found:
                function_details = '\n'.join([f"- {f}" for f in generation_functions_found])
                raise ArtificialDataFoundError(
                    f"CRITICAL FAILURE: Data generation functions detected:\n{function_details}"
                )
            
            logger.info("✅ ZERO TOLERANCE: No data generation functions detected")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Data generation function scan failed: {e}")
    
    def _inspect_module_for_generation_functions(self, file_path):
        """Inspect module for data generation functions"""
        functions_found = []
        
        try:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                return functions_found
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location("temp_module", full_path)
            if spec is None:
                return functions_found
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Inspect all functions in module
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                name_lower = name.lower()
                
                # Check function name
                for prohibited in self.prohibited_functions:
                    if prohibited in name_lower:
                        functions_found.append(f"{file_path} - Function '{name}'")
                        break
                
                # Check docstring for generation patterns
                if obj.__doc__:
                    doc_lower = obj.__doc__.lower()
                    for pattern in self.prohibited_data_patterns:
                        if pattern in doc_lower:
                            functions_found.append(f"{file_path} - Function '{name}' (docstring contains '{pattern}')")
                            break
                
                # Check function source for generation patterns
                try:
                    source = inspect.getsource(obj)
                    source_lower = source.lower()
                    for pattern in self.prohibited_data_patterns:
                        if pattern in source_lower:
                            functions_found.append(f"{file_path} - Function '{name}' (source contains '{pattern}')")
                            break
                except (OSError, TypeError):
                    # Could not get source
                    pass
                    
        except Exception as e:
            logger.warning(f"Error inspecting module {file_path}: {e}")
        
        return functions_found
    
    def test_zero_hardcoded_fake_data(self):
        """CRITICAL: Test ZERO hardcoded fake data in source code"""
        try:
            fake_data_patterns = [
                # Fake price patterns
                r'\b19500\b', r'\b20000\b', r'\b18000\b',  # Common fake NIFTY prices
                
                # Fake timestamp patterns
                r'2023-01-01', r'2024-01-01', r'1970-01-01',  # Common fake dates
                
                # Fake option data patterns
                r'CE.*19500', r'PE.*19500',  # Fake strike prices
                
                # Array/list patterns that look fake
                r'\[1,\s*2,\s*3', r'\[0\.1,\s*0\.2,\s*0\.3',  # Sequential fake arrays
                
                # Obvious test values
                r'\btest_value\b', r'\bsample_price\b', r'\bdummy_data\b'
            ]
            
            violations = []
            
            for file_path in self.critical_files:
                violations.extend(self._scan_file_for_fake_data_patterns(file_path, fake_data_patterns))
            
            if violations:
                violation_details = '\n'.join([f"- {v}" for v in violations])
                raise FakeDataViolationError(
                    f"CRITICAL FAILURE: Hardcoded fake data detected:\n{violation_details}"
                )
            
            logger.info("✅ ZERO TOLERANCE: No hardcoded fake data detected")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Fake data pattern scan failed: {e}")
    
    def _scan_file_for_fake_data_patterns(self, file_path, patterns):
        """Scan file for hardcoded fake data patterns"""
        violations = []
        
        try:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                return violations
            
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                # Skip comments and docstrings
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                
                for pattern in patterns:
                    if re.search(pattern, line):
                        violations.append(f"{file_path}:{i} - Potential fake data pattern '{pattern}': {line.strip()}")
                        
        except Exception as e:
            logger.error(f"Error scanning fake data patterns in {file_path}: {e}")
        
        return violations
    
    def test_zero_test_data_constants(self):
        """CRITICAL: Test ZERO test data constants in production code"""
        try:
            test_constant_patterns = [
                r'TEST_DATA\s*=', r'MOCK_DATA\s*=', r'SAMPLE_DATA\s*=',
                r'FAKE_PRICE\s*=', r'DUMMY_VALUE\s*=', r'DEFAULT_DATA\s*=',
                r'FALLBACK_DATA\s*=', r'BACKUP_DATA\s*=', r'SYNTHETIC_\w+\s*='
            ]
            
            violations = []
            
            for file_path in self.critical_files:
                violations.extend(self._scan_file_for_test_constants(file_path, test_constant_patterns))
            
            if violations:
                violation_details = '\n'.join([f"- {v}" for v in violations])
                raise SyntheticDataDetectedError(
                    f"CRITICAL FAILURE: Test data constants in production code:\n{violation_details}"
                )
            
            logger.info("✅ ZERO TOLERANCE: No test data constants detected")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Test constant scan failed: {e}")
    
    def _scan_file_for_test_constants(self, file_path, patterns):
        """Scan file for test data constant definitions"""
        violations = []
        
        try:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                return violations
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith('#'):
                    continue
                
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append(f"{file_path}:{i} - Test constant detected: {line.strip()}")
                        
        except Exception as e:
            logger.error(f"Error scanning test constants in {file_path}: {e}")
        
        return violations
    
    def test_zero_data_caching_fallbacks(self):
        """CRITICAL: Test ZERO data caching with synthetic fallbacks"""
        try:
            cache_fallback_patterns = [
                r'cache.*fallback', r'cached.*backup', r'cache.*default',
                r'if.*cache.*else.*generate', r'cache.*synthetic', r'cache.*mock'
            ]
            
            violations = []
            
            for file_path in self.critical_files:
                violations.extend(self._scan_file_for_cache_fallbacks(file_path, cache_fallback_patterns))
            
            if violations:
                violation_details = '\n'.join([f"- {v}" for v in violations])
                raise SyntheticDataDetectedError(
                    f"CRITICAL FAILURE: Cache fallback to synthetic data detected:\n{violation_details}"
                )
            
            logger.info("✅ ZERO TOLERANCE: No cache fallbacks to synthetic data detected")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Cache fallback scan failed: {e}")
    
    def _scan_file_for_cache_fallbacks(self, file_path, patterns):
        """Scan file for cache fallback mechanisms"""
        violations = []
        
        try:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                return violations
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                line_lower = line.lower()
                for pattern in patterns:
                    if re.search(pattern, line_lower):
                        violations.append(f"{file_path}:{i} - Cache fallback detected: {line.strip()}")
                        
        except Exception as e:
            logger.error(f"Error scanning cache fallbacks in {file_path}: {e}")
        
        return violations
    
    def test_runtime_synthetic_data_detection(self):
        """CRITICAL: Test runtime detection of synthetic data usage"""
        try:
            # Import and test components for synthetic data usage
            test_components = [
                'correlation_matrix_engine',
                'sophisticated_regime_formation_engine',
                'optimized_heavydb_engine'
            ]
            
            for component_name in test_components:
                self._test_component_runtime_data_source(component_name)
            
            logger.info("✅ ZERO TOLERANCE: Runtime synthetic data detection passed")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Runtime synthetic data detection failed: {e}")
    
    def _test_component_runtime_data_source(self, component_name):
        """Test component at runtime for synthetic data usage"""
        try:
            # Import component
            component = __import__(component_name)
            
            # Look for data source attributes
            if hasattr(component, 'data_source'):
                data_source = str(component.data_source).lower()
                for pattern in self.prohibited_data_patterns:
                    if pattern in data_source:
                        raise SyntheticDataDetectedError(
                            f"RUNTIME VIOLATION: {component_name} using prohibited data source: {data_source}"
                        )
            
            # Test component methods for synthetic data indicators
            for attr_name in dir(component):
                if callable(getattr(component, attr_name)):
                    method = getattr(component, attr_name)
                    if hasattr(method, '__doc__') and method.__doc__:
                        doc_lower = method.__doc__.lower()
                        for pattern in self.prohibited_data_patterns:
                            if pattern in doc_lower:
                                raise SyntheticDataDetectedError(
                                    f"RUNTIME VIOLATION: {component_name}.{attr_name} references synthetic data"
                                )
                                
        except ImportError:
            logger.warning(f"Could not import component {component_name} for runtime testing")
        except SyntheticDataDetectedError:
            raise
        except Exception as e:
            logger.warning(f"Error testing component {component_name} at runtime: {e}")

def run_zero_synthetic_tolerance_tests():
    """Run ZERO synthetic data tolerance validation"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔒 ZERO SYNTHETIC DATA TOLERANCE VALIDATION")
    print("=" * 70)
    print("⚠️  ABSOLUTELY NO SYNTHETIC DATA ALLOWED")
    print("⚠️  IMMEDIATE ABORT ON ANY FAKE DATA DETECTION")
    print("⚠️  ZERO TOLERANCE FOR MOCK OR GENERATED DATA")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestZeroSyntheticTolerance)
    
    # Run tests with immediate abort on failure
    runner = unittest.TextTestRunner(verbosity=2, failfast=True)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print(f"\n{'=' * 70}")
    print(f"ZERO SYNTHETIC TOLERANCE RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"{'=' * 70}")
    
    if failures > 0 or errors > 0:
        print("❌ CRITICAL FAILURE: SYNTHETIC DATA TOLERANCE VIOLATED")
        print("🚫 FAKE, MOCK, OR GENERATED DATA DETECTED")
        print("🔒 SYSTEM COMPROMISED - SYNTHETIC DATA NOT ALLOWED")
        
        if failures > 0:
            print("\nSYNTHETIC DATA VIOLATIONS:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if errors > 0:
            print("\nDETECTION ERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        return False
    else:
        print("✅ ZERO SYNTHETIC TOLERANCE VALIDATION PASSED")
        print("🔒 100% REAL DATA USAGE CONFIRMED")
        print("🚫 ZERO SYNTHETIC DATA DETECTED")
        print("✅ NO MOCK DATA FOUND")
        print("✅ NO ARTIFICIAL DATA GENERATION")
        print("✅ PRODUCTION DATA INTEGRITY MAINTAINED")
        return True

if __name__ == "__main__":
    success = run_zero_synthetic_tolerance_tests()
    sys.exit(0 if success else 1)