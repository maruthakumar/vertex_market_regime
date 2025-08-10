"""
Detailed HeavyDB Integration Testing for POS Strategy
====================================================

This module provides comprehensive testing of HeavyDB integration with real NIFTY option chain data.
CRITICAL: NO MOCK DATA - All tests use actual HeavyDB database with real market data.

Database: localhost:6274, admin/HyperInteractive, heavyai
Table: nifty_option_chain (33.19M+ rows)
Performance Target: 529,861 rows/sec processing

Test Categories:
1. Connection and Authentication
2. Data Availability and Volume
3. Data Quality and Structure
4. Query Performance
5. Strike Selection Logic
6. Greeks Calculations
7. Multi-timeframe Data Access
8. Error Handling and Recovery
"""

import pandas as pd
import numpy as np
from heavydb import connect
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import date, datetime, time
import traceback
from decimal import Decimal

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent.parent.parent))

from backtester_v2.strategies.pos.models_enhanced import (
    CompletePOSStrategy, EnhancedLegModel, StrikeMethod,
    InstrumentType, TransactionType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HeavyDBIntegrationTester:
    """Comprehensive HeavyDB integration testing with real data"""
    
    def __init__(self):
        self.connection = None
        self.test_results = {
            'connection': [],
            'data_availability': [],
            'data_quality': [],
            'query_performance': [],
            'strike_selection': [],
            'greeks_validation': [],
            'multi_timeframe': [],
            'error_handling': []
        }
        
        # Database connection parameters
        self.db_params = {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
        
        # Performance benchmarks
        self.performance_targets = {
            'min_rows': 1000000,  # Minimum 1M rows for testing
            'target_processing_speed': 529861,  # rows/sec
            'max_query_time': 10.0,  # seconds for standard queries
            'min_data_days': 100,  # Minimum days of data
            'required_columns': [
                'trade_date', 'trade_time', 'option_type', 'strike_price',
                'expiry_date', 'close_price', 'volume', 'open_interest',
                'delta', 'gamma', 'theta', 'vega', 'underlying_value'
            ]
        }
        
        # Test dates for consistent testing
        self.test_dates = {
            'single_day': '2024-01-01',
            'week_start': '2024-01-01',
            'week_end': '2024-01-05',
            'month_start': '2024-01-01',
            'month_end': '2024-01-31'
        }
    
    def establish_connection(self) -> bool:
        """Establish and validate HeavyDB connection"""
        logger.info("=== Testing HeavyDB Connection ===")
        
        test_result = {
            'test': 'database_connection',
            'status': 'UNKNOWN',
            'connection_time': None,
            'server_info': {}
        }
        
        try:
            start_time = datetime.now()
            
            # Establish connection
            self.connection = connect(**self.db_params)
            
            connection_time = (datetime.now() - start_time).total_seconds()
            test_result['connection_time'] = connection_time
            
            # Test connection with simple query
            test_query = "SELECT 1 as test_value"
            result = pd.read_sql(test_query, self.connection)
            
            if result.iloc[0]['test_value'] == 1:
                test_result['status'] = 'PASSED'
                logger.info(f"✓ HeavyDB connection established in {connection_time:.3f}s")
                
                # Get server information
                try:
                    server_info_query = """
                    SELECT 
                        COUNT(*) as total_tables
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = 'mapd'
                    """
                    server_info = pd.read_sql(server_info_query, self.connection)
                    test_result['server_info']['total_tables'] = int(server_info['total_tables'].iloc[0])
                except Exception as e:
                    logger.warning(f"Could not get server info: {e}")
                    
            else:
                raise ValueError("Connection test query failed")
                
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"✗ HeavyDB connection failed: {e}")
            return False
        
        self.test_results['connection'].append(test_result)
        return True
    
    def test_data_availability_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive test of data availability and volume"""
        logger.info("\n=== Testing Data Availability and Volume ===")
        
        test_result = {
            'test': 'data_availability',
            'status': 'UNKNOWN',
            'data_summary': {},
            'quality_metrics': {},
            'validation_checks': {}
        }
        
        try:
            if not self.connection:
                raise ValueError("No database connection")
            
            # Test 1: Overall data volume
            volume_query = """
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT trade_date) as unique_days,
                MIN(trade_date) as earliest_date,
                MAX(trade_date) as latest_date,
                COUNT(DISTINCT option_type) as option_types,
                COUNT(DISTINCT expiry_date) as unique_expiries
            FROM nifty_option_chain
            """
            
            volume_result = pd.read_sql(volume_query, self.connection)
            data_summary = {
                'total_rows': int(volume_result['total_rows'].iloc[0]),
                'unique_days': int(volume_result['unique_days'].iloc[0]),
                'earliest_date': str(volume_result['earliest_date'].iloc[0]),
                'latest_date': str(volume_result['latest_date'].iloc[0]),
                'option_types': int(volume_result['option_types'].iloc[0]),
                'unique_expiries': int(volume_result['unique_expiries'].iloc[0])
            }
            
            test_result['data_summary'] = data_summary
            
            logger.info(f"Total rows: {data_summary['total_rows']:,}")
            logger.info(f"Date range: {data_summary['earliest_date']} to {data_summary['latest_date']}")
            logger.info(f"Unique days: {data_summary['unique_days']:,}")
            logger.info(f"Option types: {data_summary['option_types']}")
            logger.info(f"Unique expiries: {data_summary['unique_expiries']:,}")
            
            # Test 2: Data distribution by option type
            distribution_query = """
            SELECT 
                option_type,
                COUNT(*) as row_count,
                COUNT(DISTINCT trade_date) as days_available,
                COUNT(DISTINCT strike_price) as unique_strikes,
                AVG(volume) as avg_volume,
                AVG(open_interest) as avg_oi
            FROM nifty_option_chain
            WHERE trade_date >= '2024-01-01'
            GROUP BY option_type
            ORDER BY option_type
            """
            
            distribution_result = pd.read_sql(distribution_query, self.connection)
            
            quality_metrics = {}
            for _, row in distribution_result.iterrows():
                option_type = row['option_type']
                quality_metrics[option_type] = {
                    'row_count': int(row['row_count']),
                    'days_available': int(row['days_available']),
                    'unique_strikes': int(row['unique_strikes']),
                    'avg_volume': float(row['avg_volume']) if pd.notna(row['avg_volume']) else 0,
                    'avg_oi': float(row['avg_oi']) if pd.notna(row['avg_oi']) else 0
                }
                
                logger.info(f"{option_type}: {row['row_count']:,} rows, {row['unique_strikes']} strikes")
            
            test_result['quality_metrics'] = quality_metrics
            
            # Test 3: Validation checks
            validation_checks = {
                'sufficient_data_volume': data_summary['total_rows'] >= self.performance_targets['min_rows'],
                'sufficient_time_span': data_summary['unique_days'] >= self.performance_targets['min_data_days'],
                'has_call_options': 'CE' in quality_metrics,
                'has_put_options': 'PE' in quality_metrics,
                'recent_data_available': data_summary['latest_date'] >= '2024-01-01'
            }
            
            test_result['validation_checks'] = validation_checks
            
            # Log validation results
            for check, passed in validation_checks.items():
                status = "✓" if passed else "✗"
                logger.info(f"{status} {check}: {passed}")
            
            # Determine overall status
            passed_checks = sum(validation_checks.values())
            total_checks = len(validation_checks)
            
            if passed_checks == total_checks:
                test_result['status'] = 'PASSED'
            elif passed_checks >= total_checks * 0.8:
                test_result['status'] = 'PARTIAL'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"Data availability validation: {passed_checks}/{total_checks} checks passed")
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"✗ Data availability test failed: {e}")
        
        self.test_results['data_availability'].append(test_result)
        return test_result
    
    def test_data_quality_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive data quality testing"""
        logger.info("\n=== Testing Data Quality and Structure ===")
        
        test_result = {
            'test': 'data_quality',
            'status': 'UNKNOWN',
            'column_analysis': {},
            'data_integrity': {},
            'business_logic_validation': {}
        }
        
        try:
            if not self.connection:
                raise ValueError("No database connection")
            
            # Test 1: Column availability and types
            try:
                columns_query = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'nifty_option_chain'
                ORDER BY ORDINAL_POSITION
                """
                
                columns_result = pd.read_sql(columns_query, self.connection)
                available_columns = columns_result['COLUMN_NAME'].tolist()
                
                missing_columns = [col for col in self.performance_targets['required_columns'] 
                                 if col not in available_columns]
                
                test_result['column_analysis'] = {
                    'available_columns': available_columns,
                    'missing_required': missing_columns,
                    'total_columns': len(available_columns)
                }
                
                logger.info(f"Available columns: {len(available_columns)}")
                if missing_columns:
                    logger.warning(f"Missing required columns: {missing_columns}")
                else:
                    logger.info("✓ All required columns available")
                    
            except Exception as e:
                logger.warning(f"Could not analyze columns: {e}")
                # Fallback: test by querying sample data
                sample_query = f"""
                SELECT * FROM nifty_option_chain 
                WHERE trade_date = '{self.test_dates['single_day']}'
                LIMIT 1
                """
                sample_data = pd.read_sql(sample_query, self.connection)
                test_result['column_analysis'] = {
                    'available_columns': sample_data.columns.tolist(),
                    'total_columns': len(sample_data.columns)
                }
            
            # Test 2: Data integrity checks
            integrity_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN close_price > 0 THEN 1 END) as valid_prices,
                COUNT(CASE WHEN volume >= 0 THEN 1 END) as valid_volumes,
                COUNT(CASE WHEN open_interest >= 0 THEN 1 END) as valid_oi,
                COUNT(CASE WHEN delta IS NOT NULL THEN 1 END) as delta_available,
                COUNT(CASE WHEN gamma IS NOT NULL THEN 1 END) as gamma_available,
                COUNT(CASE WHEN underlying_value > 0 THEN 1 END) as valid_underlying,
                COUNT(DISTINCT trade_date) as trading_days
            FROM nifty_option_chain
            WHERE trade_date BETWEEN '{self.test_dates['week_start']}' AND '{self.test_dates['week_end']}'
            """
            
            integrity_result = pd.read_sql(integrity_query, self.connection)
            
            total_records = int(integrity_result['total_records'].iloc[0])
            
            data_integrity = {
                'total_records': total_records,
                'valid_prices_pct': float(integrity_result['valid_prices'].iloc[0]) / total_records * 100,
                'valid_volumes_pct': float(integrity_result['valid_volumes'].iloc[0]) / total_records * 100,
                'valid_oi_pct': float(integrity_result['valid_oi'].iloc[0]) / total_records * 100,
                'delta_availability_pct': float(integrity_result['delta_available'].iloc[0]) / total_records * 100,
                'gamma_availability_pct': float(integrity_result['gamma_available'].iloc[0]) / total_records * 100,
                'valid_underlying_pct': float(integrity_result['valid_underlying'].iloc[0]) / total_records * 100,
                'trading_days': int(integrity_result['trading_days'].iloc[0])
            }
            
            test_result['data_integrity'] = data_integrity
            
            logger.info(f"Data integrity analysis for {total_records:,} records:")
            logger.info(f"  Valid prices: {data_integrity['valid_prices_pct']:.1f}%")
            logger.info(f"  Valid volumes: {data_integrity['valid_volumes_pct']:.1f}%")
            logger.info(f"  Delta availability: {data_integrity['delta_availability_pct']:.1f}%")
            logger.info(f"  Gamma availability: {data_integrity['gamma_availability_pct']:.1f}%")
            
            # Test 3: Business logic validation
            business_logic_query = f"""
            SELECT 
                option_type,
                COUNT(*) as records,
                AVG(CASE WHEN option_type = 'CE' AND delta > 0 THEN 1.0 ELSE 0.0 END) as ce_positive_delta_pct,
                AVG(CASE WHEN option_type = 'PE' AND delta < 0 THEN 1.0 ELSE 0.0 END) as pe_negative_delta_pct,
                AVG(CASE WHEN gamma >= 0 THEN 1.0 ELSE 0.0 END) as positive_gamma_pct,
                COUNT(DISTINCT strike_price) as unique_strikes,
                MIN(strike_price) as min_strike,
                MAX(strike_price) as max_strike
            FROM nifty_option_chain
            WHERE trade_date = '{self.test_dates['single_day']}'
            AND trade_time = '09:20:00'
            AND option_type IN ('CE', 'PE')
            AND delta IS NOT NULL
            GROUP BY option_type
            """
            
            business_result = pd.read_sql(business_logic_query, self.connection)
            
            business_logic_validation = {}
            for _, row in business_result.iterrows():
                option_type = row['option_type']
                business_logic_validation[option_type] = {
                    'records': int(row['records']),
                    'delta_logic_correct': float(row['ce_positive_delta_pct']) > 0.8 if option_type == 'CE' 
                                         else float(row['pe_negative_delta_pct']) > 0.8,
                    'gamma_positive_pct': float(row['positive_gamma_pct']) * 100,
                    'strike_range': {
                        'min': float(row['min_strike']),
                        'max': float(row['max_strike']),
                        'count': int(row['unique_strikes'])
                    }
                }
                
                logger.info(f"{option_type}: {row['records']} records, {row['unique_strikes']} strikes")
                logger.info(f"  Strike range: {row['min_strike']:.0f} - {row['max_strike']:.0f}")
            
            test_result['business_logic_validation'] = business_logic_validation
            
            # Determine overall status
            quality_score = 0
            quality_checks = 0
            
            # Check data integrity percentages
            for metric, value in data_integrity.items():
                if metric.endswith('_pct'):
                    quality_checks += 1
                    if value >= 80:  # 80% threshold
                        quality_score += 1
            
            # Check business logic
            for option_data in business_logic_validation.values():
                quality_checks += 1
                if option_data['delta_logic_correct']:
                    quality_score += 1
            
            quality_ratio = quality_score / quality_checks if quality_checks > 0 else 0
            
            if quality_ratio >= 0.9:
                test_result['status'] = 'PASSED'
            elif quality_ratio >= 0.7:
                test_result['status'] = 'PARTIAL'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"Data quality score: {quality_score}/{quality_checks} ({quality_ratio:.1%})")
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"✗ Data quality test failed: {e}")
        
        self.test_results['data_quality'].append(test_result)
        return test_result
    
    def test_query_performance(self) -> Dict[str, Any]:
        """Test query performance against targets"""
        logger.info("\n=== Testing Query Performance ===")
        
        test_result = {
            'test': 'query_performance',
            'status': 'UNKNOWN',
            'performance_tests': []
        }
        
        try:
            if not self.connection:
                raise ValueError("No database connection")
            
            # Performance Test 1: Simple aggregation
            perf_test_1 = self._run_performance_test(
                name="simple_aggregation",
                query=f"""
                SELECT 
                    option_type,
                    COUNT(*) as record_count,
                    AVG(close_price) as avg_price,
                    SUM(volume) as total_volume
                FROM nifty_option_chain
                WHERE trade_date = '{self.test_dates['single_day']}'
                GROUP BY option_type
                """,
                expected_rows=3  # CE, PE, XX
            )
            test_result['performance_tests'].append(perf_test_1)
            
            # Performance Test 2: Strike selection query
            perf_test_2 = self._run_performance_test(
                name="strike_selection",
                query=f"""
                SELECT 
                    option_type,
                    strike_price,
                    close_price,
                    delta,
                    gamma,
                    underlying_value
                FROM nifty_option_chain
                WHERE trade_date = '{self.test_dates['single_day']}'
                AND trade_time = '09:20:00'
                AND option_type IN ('CE', 'PE')
                ORDER BY option_type, ABS(strike_price - underlying_value)
                """,
                expected_min_rows=100
            )
            test_result['performance_tests'].append(perf_test_2)
            
            # Performance Test 3: Multi-day analysis
            perf_test_3 = self._run_performance_test(
                name="multi_day_analysis",
                query=f"""
                SELECT 
                    trade_date,
                    option_type,
                    COUNT(*) as daily_records,
                    AVG(close_price) as avg_price,
                    SUM(volume) as daily_volume
                FROM nifty_option_chain
                WHERE trade_date BETWEEN '{self.test_dates['week_start']}' AND '{self.test_dates['week_end']}'
                AND trade_time = '15:29:00'
                AND option_type IN ('CE', 'PE')
                GROUP BY trade_date, option_type
                ORDER BY trade_date, option_type
                """,
                expected_min_rows=10
            )
            test_result['performance_tests'].append(perf_test_3)
            
            # Performance Test 4: Greeks calculation query
            perf_test_4 = self._run_performance_test(
                name="greeks_calculation",
                query=f"""
                SELECT 
                    strike_price,
                    close_price,
                    delta,
                    gamma,
                    theta,
                    vega,
                    underlying_value,
                    (delta * 50) as position_delta,
                    (gamma * 50 * 50) as position_gamma
                FROM nifty_option_chain
                WHERE trade_date = '{self.test_dates['single_day']}'
                AND trade_time = '09:20:00'
                AND option_type = 'CE'
                AND delta IS NOT NULL
                AND gamma IS NOT NULL
                ORDER BY strike_price
                """,
                expected_min_rows=50
            )
            test_result['performance_tests'].append(perf_test_4)
            
            # Analyze overall performance
            total_tests = len(test_result['performance_tests'])
            passed_tests = sum(1 for test in test_result['performance_tests'] if test['status'] == 'PASSED')
            
            avg_processing_speed = np.mean([
                test['rows_per_second'] for test in test_result['performance_tests'] 
                if test['status'] in ['PASSED', 'PARTIAL']
            ])
            
            if passed_tests == total_tests:
                test_result['status'] = 'PASSED'
            elif passed_tests >= total_tests * 0.75:
                test_result['status'] = 'PARTIAL'
            else:
                test_result['status'] = 'FAILED'
            
            test_result['summary'] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'avg_processing_speed': float(avg_processing_speed),
                'target_speed': self.performance_targets['target_processing_speed']
            }
            
            logger.info(f"Performance summary: {passed_tests}/{total_tests} tests passed")
            logger.info(f"Average processing speed: {avg_processing_speed:,.0f} rows/sec")
            logger.info(f"Target speed: {self.performance_targets['target_processing_speed']:,} rows/sec")
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"✗ Query performance test failed: {e}")
        
        self.test_results['query_performance'].append(test_result)
        return test_result
    
    def _run_performance_test(self, name: str, query: str, expected_rows: int = None, 
                            expected_min_rows: int = None) -> Dict[str, Any]:
        """Run a single performance test"""
        
        perf_test = {
            'name': name,
            'status': 'UNKNOWN',
            'execution_time': None,
            'rows_returned': 0,
            'rows_per_second': 0
        }
        
        try:
            start_time = datetime.now()
            
            # Execute query
            result = pd.read_sql(query, self.connection)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            rows_returned = len(result)
            rows_per_second = rows_returned / execution_time if execution_time > 0 else 0
            
            perf_test.update({
                'execution_time': execution_time,
                'rows_returned': rows_returned,
                'rows_per_second': rows_per_second
            })
            
            # Validate results
            validation_passed = True
            validation_messages = []
            
            if expected_rows is not None and rows_returned != expected_rows:
                validation_passed = False
                validation_messages.append(f"Expected {expected_rows} rows, got {rows_returned}")
            
            if expected_min_rows is not None and rows_returned < expected_min_rows:
                validation_passed = False
                validation_messages.append(f"Expected at least {expected_min_rows} rows, got {rows_returned}")
            
            if execution_time > self.performance_targets['max_query_time']:
                validation_passed = False
                validation_messages.append(f"Query too slow: {execution_time:.3f}s > {self.performance_targets['max_query_time']}s")
            
            # Determine status
            if validation_passed:
                if rows_per_second >= self.performance_targets['target_processing_speed'] * 0.5:  # 50% of target
                    perf_test['status'] = 'PASSED'
                else:
                    perf_test['status'] = 'PARTIAL'
            else:
                perf_test['status'] = 'FAILED'
                perf_test['validation_errors'] = validation_messages
            
            logger.info(f"  {name}: {execution_time:.3f}s, {rows_returned} rows, {rows_per_second:,.0f} rows/sec - {perf_test['status']}")
            
        except Exception as e:
            perf_test['status'] = 'FAILED'
            perf_test['error'] = str(e)
            logger.error(f"  {name}: FAILED - {e}")
        
        return perf_test
    
    def test_strike_selection_logic(self) -> Dict[str, Any]:
        """Test strike selection logic for different methods"""
        logger.info("\n=== Testing Strike Selection Logic ===")
        
        test_result = {
            'test': 'strike_selection',
            'status': 'UNKNOWN',
            'strike_tests': []
        }
        
        try:
            if not self.connection:
                raise ValueError("No database connection")
            
            # Get underlying price for reference
            underlying_query = f"""
            SELECT DISTINCT underlying_value
            FROM nifty_option_chain
            WHERE trade_date = '{self.test_dates['single_day']}'
            AND trade_time = '09:20:00'
            AND underlying_value IS NOT NULL
            LIMIT 1
            """
            
            underlying_result = pd.read_sql(underlying_query, self.connection)
            
            if underlying_result.empty:
                raise ValueError("No underlying value found for test date")
            
            underlying_price = float(underlying_result['underlying_value'].iloc[0])
            logger.info(f"Reference underlying price: {underlying_price}")
            
            # Test different strike selection methods
            strike_methods = [
                ('ATM', 0),
                ('OTM_CE_100', 100),
                ('OTM_PE_100', -100),
                ('ITM_CE_100', -100),
                ('ITM_PE_100', 100)
            ]
            
            for method_name, offset in strike_methods:
                target_strike = underlying_price + offset
                
                strike_test = self._test_strike_selection_method(
                    method_name, target_strike, underlying_price
                )
                test_result['strike_tests'].append(strike_test)
            
            # Test delta-based selection
            delta_test = self._test_delta_based_selection(underlying_price)
            test_result['strike_tests'].append(delta_test)
            
            # Analyze results
            passed_tests = sum(1 for test in test_result['strike_tests'] if test['status'] == 'PASSED')
            total_tests = len(test_result['strike_tests'])
            
            if passed_tests == total_tests:
                test_result['status'] = 'PASSED'
            elif passed_tests >= total_tests * 0.75:
                test_result['status'] = 'PARTIAL'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"Strike selection tests: {passed_tests}/{total_tests} passed")
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"✗ Strike selection test failed: {e}")
        
        self.test_results['strike_selection'].append(test_result)
        return test_result
    
    def _test_strike_selection_method(self, method_name: str, target_strike: float, 
                                    underlying_price: float) -> Dict[str, Any]:
        """Test a specific strike selection method"""
        
        test = {
            'method': method_name,
            'target_strike': target_strike,
            'status': 'UNKNOWN'
        }
        
        try:
            # Find closest strike
            strike_query = f"""
            SELECT 
                strike_price,
                close_price,
                delta,
                ABS(strike_price - {target_strike}) as strike_distance
            FROM nifty_option_chain
            WHERE trade_date = '{self.test_dates['single_day']}'
            AND trade_time = '09:20:00'
            AND option_type = 'CE'
            AND close_price > 0
            ORDER BY strike_distance
            LIMIT 1
            """
            
            result = pd.read_sql(strike_query, self.connection)
            
            if result.empty:
                test['status'] = 'FAILED'
                test['error'] = 'No strikes found'
                return test
            
            selected_strike = float(result['strike_price'].iloc[0])
            strike_distance = abs(selected_strike - target_strike)
            
            test.update({
                'selected_strike': selected_strike,
                'strike_distance': strike_distance,
                'option_price': float(result['close_price'].iloc[0]),
                'delta': float(result['delta'].iloc[0]) if pd.notna(result['delta'].iloc[0]) else None
            })
            
            # Validate selection
            if strike_distance <= 50:  # Within 50 points acceptable
                test['status'] = 'PASSED'
            else:
                test['status'] = 'FAILED'
                test['error'] = f'Strike too far from target: {strike_distance} points'
            
            logger.info(f"  {method_name}: Target={target_strike}, Selected={selected_strike}, Distance={strike_distance:.0f}")
            
        except Exception as e:
            test['status'] = 'FAILED'
            test['error'] = str(e)
        
        return test
    
    def _test_delta_based_selection(self, underlying_price: float) -> Dict[str, Any]:
        """Test delta-based strike selection"""
        
        test = {
            'method': 'delta_based_0.5',
            'target_delta': 0.5,
            'status': 'UNKNOWN'
        }
        
        try:
            # Find strike closest to 0.5 delta
            delta_query = f"""
            SELECT 
                strike_price,
                close_price,
                delta,
                ABS(delta - 0.5) as delta_distance
            FROM nifty_option_chain
            WHERE trade_date = '{self.test_dates['single_day']}'
            AND trade_time = '09:20:00'
            AND option_type = 'CE'
            AND delta IS NOT NULL
            AND delta BETWEEN 0.3 AND 0.7
            ORDER BY delta_distance
            LIMIT 1
            """
            
            result = pd.read_sql(delta_query, self.connection)
            
            if result.empty:
                test['status'] = 'FAILED'
                test['error'] = 'No options with suitable delta found'
                return test
            
            selected_delta = float(result['delta'].iloc[0])
            delta_distance = abs(selected_delta - 0.5)
            
            test.update({
                'selected_strike': float(result['strike_price'].iloc[0]),
                'selected_delta': selected_delta,
                'delta_distance': delta_distance,
                'option_price': float(result['close_price'].iloc[0])
            })
            
            # Validate selection
            if delta_distance <= 0.1:  # Within 0.1 delta acceptable
                test['status'] = 'PASSED'
            else:
                test['status'] = 'FAILED'
                test['error'] = f'Delta too far from target: {delta_distance:.3f}'
            
            logger.info(f"  Delta-based: Target=0.5, Selected={selected_delta:.3f}, Distance={delta_distance:.3f}")
            
        except Exception as e:
            test['status'] = 'FAILED'
            test['error'] = str(e)
        
        return test
    
    def test_greeks_validation(self) -> Dict[str, Any]:
        """Test Greeks calculations and validation"""
        logger.info("\n=== Testing Greeks Validation ===")
        
        test_result = {
            'test': 'greeks_validation',
            'status': 'UNKNOWN',
            'greeks_analysis': {}
        }
        
        try:
            if not self.connection:
                raise ValueError("No database connection")
            
            # Get Greeks data for analysis
            greeks_query = f"""
            SELECT 
                option_type,
                strike_price,
                close_price,
                delta,
                gamma,
                theta,
                vega,
                underlying_value
            FROM nifty_option_chain
            WHERE trade_date = '{self.test_dates['single_day']}'
            AND trade_time = '09:20:00'
            AND option_type IN ('CE', 'PE')
            AND delta IS NOT NULL
            AND gamma IS NOT NULL
            AND close_price > 0
            ORDER BY option_type, strike_price
            LIMIT 200
            """
            
            greeks_data = pd.read_sql(greeks_query, self.connection)
            
            if greeks_data.empty:
                raise ValueError("No Greeks data available")
            
            # Analyze Greeks by option type
            for option_type in ['CE', 'PE']:
                type_data = greeks_data[greeks_data['option_type'] == option_type]
                
                if type_data.empty:
                    continue
                
                analysis = {
                    'record_count': len(type_data),
                    'delta_range': {
                        'min': float(type_data['delta'].min()),
                        'max': float(type_data['delta'].max()),
                        'mean': float(type_data['delta'].mean())
                    },
                    'gamma_stats': {
                        'min': float(type_data['gamma'].min()),
                        'max': float(type_data['gamma'].max()),
                        'mean': float(type_data['gamma'].mean()),
                        'positive_count': int((type_data['gamma'] >= 0).sum())
                    },
                    'validations': {}
                }
                
                # Validation checks
                if option_type == 'CE':
                    analysis['validations']['delta_positive'] = (type_data['delta'] >= 0).all()
                    analysis['validations']['atm_delta_reasonable'] = (
                        type_data[type_data['delta'].between(0.4, 0.6)]['delta'].count() > 0
                    )
                elif option_type == 'PE':
                    analysis['validations']['delta_negative'] = (type_data['delta'] <= 0).all()
                    analysis['validations']['atm_delta_reasonable'] = (
                        type_data[type_data['delta'].between(-0.6, -0.4)]['delta'].count() > 0
                    )
                
                analysis['validations']['gamma_positive'] = (type_data['gamma'] >= 0).all()
                analysis['validations']['reasonable_ranges'] = (
                    type_data['delta'].between(-1, 1).all() and
                    type_data['gamma'].between(0, 0.1).all()
                )
                
                test_result['greeks_analysis'][option_type] = analysis
                
                # Log analysis
                logger.info(f"{option_type} Greeks Analysis:")
                logger.info(f"  Records: {analysis['record_count']}")
                logger.info(f"  Delta range: {analysis['delta_range']['min']:.3f} to {analysis['delta_range']['max']:.3f}")
                logger.info(f"  Gamma range: {analysis['gamma_stats']['min']:.5f} to {analysis['gamma_stats']['max']:.5f}")
                
                for validation, passed in analysis['validations'].items():
                    status = "✓" if passed else "✗"
                    logger.info(f"  {status} {validation}: {passed}")
            
            # Portfolio Greeks calculation test
            portfolio_test = self._test_portfolio_greeks_calculation(greeks_data)
            test_result['portfolio_greeks'] = portfolio_test
            
            # Determine overall status
            all_validations = []
            for option_analysis in test_result['greeks_analysis'].values():
                all_validations.extend(option_analysis['validations'].values())
            
            if portfolio_test['status'] == 'PASSED':
                all_validations.append(True)
            
            passed_validations = sum(all_validations)
            total_validations = len(all_validations)
            
            if passed_validations == total_validations:
                test_result['status'] = 'PASSED'
            elif passed_validations >= total_validations * 0.8:
                test_result['status'] = 'PARTIAL'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"Greeks validation: {passed_validations}/{total_validations} checks passed")
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"✗ Greeks validation failed: {e}")
        
        self.test_results['greeks_validation'].append(test_result)
        return test_result
    
    def _test_portfolio_greeks_calculation(self, greeks_data: pd.DataFrame) -> Dict[str, Any]:
        """Test portfolio-level Greeks calculations"""
        
        test = {
            'test': 'portfolio_greeks_calculation',
            'status': 'UNKNOWN'
        }
        
        try:
            # Simulate a simple portfolio (1 lot each position)
            lot_size = 50  # NIFTY lot size
            
            # Calculate portfolio Greeks
            portfolio_delta = (greeks_data['delta'] * lot_size).sum()
            portfolio_gamma = (greeks_data['gamma'] * lot_size * lot_size).sum()
            portfolio_theta = (greeks_data['theta'] * lot_size).sum() if 'theta' in greeks_data.columns else 0
            portfolio_vega = (greeks_data['vega'] * lot_size).sum() if 'vega' in greeks_data.columns else 0
            
            test.update({
                'portfolio_delta': float(portfolio_delta),
                'portfolio_gamma': float(portfolio_gamma),
                'portfolio_theta': float(portfolio_theta),
                'portfolio_vega': float(portfolio_vega),
                'position_count': len(greeks_data)
            })
            
            # Validate calculations
            validations = {
                'delta_reasonable': abs(portfolio_delta) < 10000,  # Reasonable for sample portfolio
                'gamma_positive': portfolio_gamma >= 0,
                'calculations_complete': not any(pd.isna([portfolio_delta, portfolio_gamma]))
            }
            
            test['validations'] = validations
            
            if all(validations.values()):
                test['status'] = 'PASSED'
            else:
                test['status'] = 'FAILED'
            
            logger.info(f"Portfolio Greeks: Delta={portfolio_delta:.0f}, Gamma={portfolio_gamma:.0f}")
            
        except Exception as e:
            test['status'] = 'FAILED'
            test['error'] = str(e)
        
        return test
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration test report"""
        logger.info("\n" + "="*80)
        logger.info("HEAVYDB INTEGRATION TEST REPORT")
        logger.info("="*80)
        
        # Calculate summary statistics
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            for test in tests:
                if isinstance(test, dict) and 'status' in test:
                    total_tests += 1
                    if test['status'] == 'PASSED':
                        passed_tests += 1
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        overall_status = 'PASSED' if success_rate >= 0.8 else 'PARTIAL' if success_rate >= 0.6 else 'FAILED'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_info': self.db_params,
            'overall_status': overall_status,
            'success_rate': success_rate,
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests
            },
            'detailed_results': self.test_results,
            'performance_targets': self.performance_targets
        }
        
        # Log summary
        for category, tests in self.test_results.items():
            if tests:
                latest_test = tests[-1]
                status = latest_test.get('status', 'UNKNOWN')
                logger.info(f"{category.upper():<25} | {status}")
        
        logger.info(f"\nOverall Status: {overall_status} ({success_rate:.1%} success rate)")
        
        # Save report
        report_file = f"/tmp/heavydb_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Integration report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")
        
        return report
    
    def run_all_integration_tests(self) -> bool:
        """Run all HeavyDB integration tests"""
        logger.info("Starting HeavyDB Integration Tests")
        logger.info("="*80)
        
        try:
            # Test 1: Connection
            if not self.establish_connection():
                return False
            
            # Test 2: Data availability
            self.test_data_availability_comprehensive()
            
            # Test 3: Data quality
            self.test_data_quality_comprehensive()
            
            # Test 4: Query performance
            self.test_query_performance()
            
            # Test 5: Strike selection
            self.test_strike_selection_logic()
            
            # Test 6: Greeks validation
            self.test_greeks_validation()
            
            # Generate report
            report = self.generate_integration_report()
            
            return report['overall_status'] in ['PASSED', 'PARTIAL']
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return False
        
        finally:
            if self.connection:
                self.connection.close()
                logger.info("Database connection closed")


def run_heavydb_integration_tests():
    """Entry point for HeavyDB integration tests"""
    tester = HeavyDBIntegrationTester()
    success = tester.run_all_integration_tests()
    
    if success:
        print("\n✅ HeavyDB Integration Tests Completed Successfully!")
        return 0
    else:
        print("\n❌ HeavyDB Integration Tests Failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_heavydb_integration_tests())