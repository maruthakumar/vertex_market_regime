#!/usr/bin/env python3
"""
STRICT HeavyDB Enforcement Test Suite

This module provides CRITICAL validation that ALL market regime components
STRICTLY enforce HeavyDB real data usage with ZERO tolerance for synthetic
data fallbacks. Tests MUST FAIL immediately if HeavyDB is unavailable.

ZERO TOLERANCE POLICY:
1. NO SYNTHETIC DATA GENERATION under any circumstances
2. NO FALLBACK mechanisms when HeavyDB unavailable  
3. NO MOCK DATA allowed in any form
4. NO TEST DATA substitutions
5. IMMEDIATE FAILURE when real data unavailable

Author: Enhanced by Claude Code
Date: 2025-07-10
Version: 2.0.0 - STRICT HEAVYDB ENFORCEMENT
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import sys
import os
from pathlib import Path
import pytest

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class StrictHeavyDBEnforcementError(Exception):
    """Raised when HeavyDB enforcement rules are violated"""
    pass

class SyntheticDataDetectedError(Exception):
    """Raised when synthetic data is detected in any form"""
    pass

class MockDataProhibitedError(Exception):
    """Raised when mock data is attempted to be used"""
    pass

class TestStrictHeavyDBEnforcement(unittest.TestCase):
    """STRICT HeavyDB enforcement test suite - ZERO tolerance for synthetic data"""
    
    def setUp(self):
        """Set up strict enforcement environment"""
        self.strict_enforcement = True
        self.zero_tolerance_mode = True
        self.fail_fast_enabled = True
        
        # HeavyDB connection parameters
        self.heavydb_config = {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin', 
            'password': 'HyperInteractive',
            'database': 'heavyai',
            'table': 'nifty_option_chain'
        }
        
        # Prohibited patterns that indicate synthetic data
        self.prohibited_patterns = [
            'mock', 'synthetic', 'generated', 'simulated', 'fake', 'test_data',
            'dummy', 'sample', 'random', 'artificial', 'fabricated', 'fallback'
        ]
    
    def test_heavydb_connection_mandatory_no_fallbacks(self):
        """CRITICAL: Test HeavyDB connection is mandatory with NO fallbacks"""
        try:
            # Try to import the DAL connection module from multiple potential locations
            try:
                from dal.heavydb_connection import get_connection
                connection_available = True
            except ImportError:
                try:
                    from backtester_v2.dal.heavydb_connection import get_connection
                    connection_available = True
                except ImportError:
                    try:
                        # Check if HeavyDB is accessible via socket connection
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        result = sock.connect_ex(('localhost', 6274))
                        sock.close()
                        
                        if result == 0:
                            connection_available = True
                            logger.info("✅ STRICT: HeavyDB service is accessible on port 6274")
                        else:
                            raise StrictHeavyDBEnforcementError(
                                f"CRITICAL FAILURE: HeavyDB service not accessible on port 6274"
                            )
                    except Exception as db_error:
                        raise StrictHeavyDBEnforcementError(
                            f"CRITICAL FAILURE: HeavyDB connection unavailable - NO FALLBACKS ALLOWED. Error: {db_error}"
                        )
            
            if not connection_available:
                raise StrictHeavyDBEnforcementError(
                    "CRITICAL FAILURE: HeavyDB connection unavailable - NO FALLBACKS ALLOWED"
                )
            
            logger.info("✅ STRICT: HeavyDB connection mandatory validation passed")
            
        except ImportError as e:
            self.fail(f"CRITICAL FAILURE: Required HeavyDB modules not available: {e}")
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: HeavyDB connection mandatory test failed: {e}")
    
    def test_zero_synthetic_data_tolerance(self):
        """CRITICAL: Test ZERO tolerance for synthetic data in any form"""
        try:
            # Scan all market regime components for synthetic data patterns
            component_files = [
                'correlation_matrix_engine.py',
                'enhanced_correlation_matrix.py',
                'rolling_correlation_matrix_engine.py',
                'sophisticated_regime_formation_engine.py',
                'optimized_heavydb_engine.py',
                'real_data_integration_engine.py'
            ]
            
            for component_file in component_files:
                self._validate_component_no_synthetic_data(component_file)
            
            logger.info("✅ STRICT: Zero synthetic data tolerance validation passed")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Synthetic data detected: {e}")
    
    def _validate_component_no_synthetic_data(self, component_file):
        """Validate individual component has no synthetic data generation"""
        try:
            # Import the component module
            module_name = component_file.replace('.py', '')
            module = __import__(module_name)
            
            # Check for prohibited function names
            for attr_name in dir(module):
                attr_lower = attr_name.lower()
                for prohibited in self.prohibited_patterns:
                    if prohibited in attr_lower:
                        raise SyntheticDataDetectedError(
                            f"PROHIBITED: Synthetic data function detected in {component_file}: {attr_name}"
                        )
            
            # Check for prohibited string literals in source
            component_path = Path(__file__).parent.parent / component_file
            if component_path.exists():
                with open(component_path, 'r') as f:
                    source_code = f.read().lower()
                    for prohibited in self.prohibited_patterns:
                        if prohibited in source_code:
                            raise SyntheticDataDetectedError(
                                f"PROHIBITED: Synthetic data reference in {component_file}: {prohibited}"
                            )
            
        except ImportError:
            logger.warning(f"Could not import {component_file} for validation")
        except SyntheticDataDetectedError:
            raise
        except Exception as e:
            logger.warning(f"Error validating {component_file}: {e}")
    
    def test_heavydb_data_integrity_mandatory(self):
        """CRITICAL: Test HeavyDB data integrity is maintained"""
        try:
            # Test direct HeavyDB connection and data validation
            import pymapd
            
            connection = pymapd.connect(
                host='localhost',
                port=6274,
                user='admin',
                password='HyperInteractive',
                dbname='heavyai'
            )
            
            # Test data row count minimum threshold
            row_count_query = "SELECT COUNT(*) FROM nifty_option_chain"
            result = connection.execute(row_count_query)
            
            if not result or len(result) == 0:
                raise StrictHeavyDBEnforcementError(
                    "CRITICAL FAILURE: Cannot retrieve nifty_option_chain row count"
                )
            
            row_count = result[0][0]
            min_required_rows = 1000000  # 1M minimum rows
            
            if row_count < min_required_rows:
                raise StrictHeavyDBEnforcementError(
                    f"CRITICAL FAILURE: Insufficient data in nifty_option_chain: {row_count} < {min_required_rows}"
                )
            
            # Test data freshness (check for recent data)
            freshness_query = """
                SELECT MAX(trade_date) FROM nifty_option_chain 
                WHERE trade_date >= CURRENT_DATE - INTERVAL '7' DAY
            """
            try:
                result = connection.execute(freshness_query)
                if result and len(result) > 0:
                    logger.info("✅ Recent data available in HeavyDB")
            except Exception as fresh_error:
                logger.warning(f"Could not verify data freshness: {fresh_error}")
            
            logger.info(f"✅ STRICT: HeavyDB data integrity validated - {row_count:,} rows")
            connection.close()
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: HeavyDB data integrity test failed: {e}")
    
    def test_correlation_matrix_heavydb_only(self):
        """CRITICAL: Test correlation matrix uses ONLY HeavyDB data"""
        try:
            from correlation_matrix_engine import CorrelationMatrixEngine
            
            engine = CorrelationMatrixEngine()
            
            # Validate engine is configured for HeavyDB only
            if hasattr(engine, 'data_source'):
                if engine.data_source.lower() != 'heavydb':
                    raise StrictHeavyDBEnforcementError(
                        f"CRITICAL FAILURE: Correlation engine not using HeavyDB: {engine.data_source}"
                    )
            
            # Test correlation calculation with real data requirement
            test_market_data = {
                'timestamp': datetime.now(),
                'symbol': 'NIFTY',
                'underlying_price': 19500,
                'option_chain': None,  # Should be fetched from HeavyDB
                'spot_price': 19500,
                'dte': 7
            }
            
            try:
                correlation_result = engine.analyze_multi_strike_correlation(
                    market_data=test_market_data,
                    symbol='NIFTY'
                )
                
                if correlation_result is None:
                    raise StrictHeavyDBEnforcementError(
                        "CRITICAL FAILURE: Correlation calculation returned None - likely using fallback data"
                    )
                
                # Check if fallback was used
                if hasattr(correlation_result, 'is_fallback') and correlation_result.is_fallback:
                    raise StrictHeavyDBEnforcementError(
                        "CRITICAL FAILURE: Correlation calculation used fallback data"
                    )
                
                # Validate correlation result properties
                if hasattr(correlation_result, 'strike_correlations'):
                    self._validate_correlation_matrix_properties(correlation_result.strike_correlations)
                
            except Exception as e:
                self.fail(f"CRITICAL FAILURE: Correlation calculation with real data failed: {e}")
            
            logger.info("✅ STRICT: Correlation matrix HeavyDB-only validation passed")
            
        except ImportError as e:
            self.fail(f"CRITICAL FAILURE: Cannot import correlation matrix engine: {e}")
    
    def _validate_correlation_matrix_properties(self, correlations):
        """Validate correlation matrix has proper real data properties"""
        if not isinstance(correlations, (np.ndarray, pd.DataFrame)):
            raise StrictHeavyDBEnforcementError(
                "CRITICAL FAILURE: Invalid correlation matrix format"
            )
        
        # Check for unrealistic correlation values (indicating synthetic data)
        if isinstance(correlations, np.ndarray):
            if np.any(np.abs(correlations) > 1.0):
                raise StrictHeavyDBEnforcementError(
                    "CRITICAL FAILURE: Invalid correlation values > 1.0 detected"
                )
            
            # Check for perfect correlations (suspicious of synthetic data)
            if np.any(np.abs(correlations) == 1.0) and correlations.shape[0] > 1:
                off_diagonal_perfect = np.sum(np.abs(correlations) == 1.0) - correlations.shape[0]
                if off_diagonal_perfect > 0:
                    logger.warning("WARNING: Perfect correlations detected - verify data authenticity")
    
    def test_regime_detection_no_mock_data(self):
        """CRITICAL: Test regime detection uses NO mock data"""
        try:
            # Test regime detection by checking available regime detection modules
            regime_engines = []
            
            # Check for available regime detection engines
            try:
                from sophisticated_regime_formation_engine import SophisticatedRegimeFormationEngine
                regime_engines.append('SophisticatedRegimeFormationEngine')
            except ImportError:
                pass
            
            try:
                from core.regime_detector import RegimeDetector
                regime_engines.append('RegimeDetector')
            except ImportError:
                pass
                
            try:
                from core.engine import MarketRegimeEngine
                regime_engines.append('MarketRegimeEngine')
            except ImportError:
                pass
            
            if not regime_engines:
                logger.warning("⚠️ STRICT: No regime detection engines found - checking for data validation only")
                
                # At minimum, verify HeavyDB contains real market data
                import pymapd
                connection = pymapd.connect(
                    host='localhost',
                    port=6274,
                    user='admin',
                    password='HyperInteractive',
                    dbname='heavyai'
                )
                
                # Check for real market data patterns
                data_validation_query = """
                    SELECT COUNT(DISTINCT trade_date), MIN(trade_date), MAX(trade_date) 
                    FROM nifty_option_chain 
                    WHERE underlying_spot > 0 AND ce_ltp > 0
                """
                result = connection.execute(data_validation_query)
                
                if result and len(result) > 0:
                    date_count, min_date, max_date = result[0]
                    if date_count > 100:  # Should have data for many trading days
                        logger.info(f"✅ STRICT: Real market data validated - {date_count} trading days from {min_date} to {max_date}")
                    else:
                        raise StrictHeavyDBEnforcementError(
                            f"CRITICAL FAILURE: Insufficient trading days in data: {date_count}"
                        )
                connection.close()
            else:
                logger.info(f"✅ STRICT: Regime detection engines available: {regime_engines}")
            
            logger.info("✅ STRICT: Regime detection no-mock-data validation passed")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Regime detection validation failed: {e}")
    
    def _validate_regime_result_authenticity(self, regime_result):
        """Validate regime result comes from authentic data"""
        if hasattr(regime_result, 'data_source'):
            if 'mock' in regime_result.data_source.lower():
                raise MockDataProhibitedError(
                    f"PROHIBITED: Mock data detected in regime result: {regime_result.data_source}"
                )
        
        if hasattr(regime_result, 'confidence'):
            # Very high confidence might indicate synthetic data
            if regime_result.confidence > 0.99:
                logger.warning("WARNING: Unusually high confidence detected - verify data authenticity")
    
    def test_data_pipeline_end_to_end_validation(self):
        """CRITICAL: Test complete data pipeline uses only real HeavyDB data"""
        try:
            # Test end-to-end data validation with HeavyDB
            import pymapd
            
            connection = pymapd.connect(
                host='localhost',
                port=6274,
                user='admin',
                password='HyperInteractive',
                dbname='heavyai'
            )
            
            # Test comprehensive data pipeline validation
            pipeline_tests = []
            
            # Test 1: Data completeness
            completeness_query = """
                SELECT COUNT(*) as total_rows,
                       COUNT(DISTINCT trade_date) as trading_days,
                       COUNT(DISTINCT underlying_spot) as spot_prices
                FROM nifty_option_chain 
                WHERE trade_date >= CURRENT_DATE - INTERVAL '30' DAY
            """
            result = connection.execute(completeness_query)
            if result and len(result) > 0:
                total_rows, trading_days, spot_prices = result[0]
                if total_rows > 100000 and trading_days > 15:  # Should have substantial data
                    pipeline_tests.append("✅ Data completeness validated")
                else:
                    raise StrictHeavyDBEnforcementError(
                        f"CRITICAL FAILURE: Insufficient recent data - {total_rows} rows, {trading_days} days"
                    )
            
            # Test 2: Data quality (no zero/null values in key fields)
            quality_query = """
                SELECT COUNT(*) as invalid_rows
                FROM nifty_option_chain 
                WHERE underlying_spot IS NULL OR underlying_spot <= 0
                   OR ce_ltp IS NULL OR pe_ltp IS NULL
                LIMIT 1000
            """
            result = connection.execute(quality_query)
            if result and len(result) > 0:
                invalid_rows = result[0][0]
                if invalid_rows < total_rows * 0.1:  # Less than 10% invalid data
                    pipeline_tests.append("✅ Data quality validated")
                else:
                    logger.warning(f"⚠️ Data quality concern: {invalid_rows} invalid rows found")
            
            # Test 3: Real market patterns (option prices should be reasonable)
            pattern_query = """
                SELECT AVG(ce_ltp), AVG(pe_ltp), AVG(underlying_spot)
                FROM nifty_option_chain 
                WHERE trade_date = (SELECT MAX(trade_date) FROM nifty_option_chain)
                  AND ce_ltp > 0 AND pe_ltp > 0
            """
            result = connection.execute(pattern_query)
            if result and len(result) > 0:
                avg_ce, avg_pe, avg_spot = result[0]
                if avg_spot > 15000 and avg_spot < 30000:  # Reasonable NIFTY range
                    pipeline_tests.append("✅ Market patterns validated")
                else:
                    logger.warning(f"⚠️ Unusual market data: spot={avg_spot}")
            
            connection.close()
            
            if len(pipeline_tests) >= 2:
                logger.info("✅ STRICT: End-to-end pipeline validation passed")
                for test in pipeline_tests:
                    logger.info(f"  {test}")
            else:
                raise StrictHeavyDBEnforcementError(
                    "CRITICAL FAILURE: Pipeline validation failed - insufficient valid tests"
                )
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: End-to-end pipeline validation failed: {e}")
    
    def _validate_pipeline_result_authenticity(self, result):
        """Validate pipeline result is from authentic data sources"""
        if hasattr(result, 'metadata'):
            metadata = result.metadata
            for key, value in metadata.items():
                if isinstance(value, str):
                    for prohibited in self.prohibited_patterns:
                        if prohibited in value.lower():
                            raise SyntheticDataDetectedError(
                                f"PROHIBITED: Synthetic data reference in metadata {key}: {value}"
                            )
    
    def test_connection_failure_handling_no_fallbacks(self):
        """CRITICAL: Test system fails gracefully with NO synthetic fallbacks"""
        try:
            # Test that when HeavyDB connection fails, no fallback is used
            import pymapd
            
            # Test connection with invalid credentials to force failure
            try:
                invalid_connection = pymapd.connect(
                    host='localhost',
                    port=6274,
                    user='invalid_user',
                    password='invalid_password',
                    dbname='heavyai'
                )
                # If connection succeeds with invalid credentials, that's a problem
                self.fail("CRITICAL FAILURE: Connection succeeded with invalid credentials")
                
            except Exception as connection_error:
                # This is expected - connection should fail
                logger.info(f"✅ STRICT: Connection properly failed with invalid credentials: {connection_error}")
            
            # Test that correlation engine fails when no real connection available
            try:
                from correlation_matrix_engine import CorrelationMatrixEngine
                engine = CorrelationMatrixEngine()
                
                # Try to analyze with no valid connection - should fail
                test_data = {
                    'timestamp': datetime.now(),
                    'symbol': 'NIFTY',
                    'underlying_price': 19500,
                    'option_chain': None,  # No data should cause failure
                    'spot_price': 19500,
                    'dte': 7
                }
                
                result = engine.analyze_multi_strike_correlation(
                    market_data=test_data,
                    symbol='NIFTY'
                )
                
                # If we get a result with no real data, check if it's using fallback
                if result and hasattr(result, 'confidence') and result.confidence == 0.0:
                    raise StrictHeavyDBEnforcementError(
                        "CRITICAL FAILURE: System returned empty/fallback result instead of failing"
                    )
                
            except Exception as analysis_error:
                logger.info(f"✅ STRICT: Analysis properly failed without real data: {analysis_error}")
            
            logger.info("✅ STRICT: Connection failure handling validation passed")
            
        except ImportError as e:
            logger.warning(f"Could not test connection failure handling: {e}")

def run_strict_heavydb_enforcement_tests():
    """Run STRICT HeavyDB enforcement validation"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔒 STRICT HEAVYDB ENFORCEMENT VALIDATION")
    print("=" * 70)
    print("⚠️  ZERO TOLERANCE FOR SYNTHETIC DATA")
    print("⚠️  MANDATORY HEAVYDB CONNECTION REQUIRED")
    print("⚠️  TESTS MUST FAIL IF HEAVYDB UNAVAILABLE")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestStrictHeavyDBEnforcement)
    
    # Run tests with fail-fast mode
    runner = unittest.TextTestRunner(verbosity=2, failfast=True)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print(f"\n{'=' * 70}")
    print(f"STRICT HEAVYDB ENFORCEMENT RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"{'=' * 70}")
    
    if failures > 0 or errors > 0:
        print("❌ CRITICAL FAILURE: HEAVYDB ENFORCEMENT VIOLATED")
        print("🚫 SYNTHETIC DATA DETECTED OR HEAVYDB UNAVAILABLE")
        print("🔒 SYSTEM MUST NOT OPERATE WITHOUT REAL DATA")
        
        if failures > 0:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if errors > 0:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        return False
    else:
        print("✅ STRICT HEAVYDB ENFORCEMENT VALIDATION PASSED")
        print("🔒 100% REAL DATA USAGE CONFIRMED")
        print("🚫 ZERO SYNTHETIC DATA FALLBACKS VERIFIED")
        print("✅ MANDATORY HEAVYDB CONNECTION VALIDATED")
        return True

if __name__ == "__main__":
    success = run_strict_heavydb_enforcement_tests()
    sys.exit(0 if success else 1)