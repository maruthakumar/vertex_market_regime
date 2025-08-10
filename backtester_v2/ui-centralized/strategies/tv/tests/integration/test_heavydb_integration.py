#!/usr/bin/env python3
"""
Integration Tests for TV Strategy with HeavyDB - REAL DATA ONLY
Tests complete workflow from TV configuration to HeavyDB query execution
NO MOCK DATA - ONLY REAL DATABASE CONNECTIONS AND REAL INPUT SHEETS
"""

import pytest
import pandas as pd
import time
from datetime import datetime, date, timedelta
from pathlib import Path
import os
import sys

# Add strategy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from parser import TVParser
from signal_processor import SignalProcessor
from query_builder import TVQueryBuilder
from processor import TVProcessor


class TestTVHeavyDBIntegration:
    """Integration tests with REAL HeavyDB and REAL configuration files"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
        self.signal_processor = SignalProcessor()
        self.query_builder = TVQueryBuilder()
        self.processor = TVProcessor()
        
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_complete_workflow_real_config_and_db(self, real_config_files, heavydb_connection, real_nifty_data_sample):
        """Test complete TV workflow with REAL config files and REAL HeavyDB data"""
        
        # Step 1: Parse REAL TV configuration
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        assert tv_config is not None, "TV config parsing must succeed"
        assert tv_config['enabled'] is True, "TV config must be enabled"
        
        # Step 2: Parse REAL signals
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        assert len(raw_signals) == 4, "Real signals file contains 4 signals"
        
        # Step 3: Process signals with REAL configuration
        processed_signals = self.signal_processor.process_signals(raw_signals, tv_config)
        assert len(processed_signals) == 2, "Should have 2 processed trades (T001 and T002)"
        
        # Step 4: Generate HeavyDB queries for REAL data
        queries = []
        for signal in processed_signals:
            # Use a recent date that exists in HeavyDB
            signal_modified = signal.copy()
            signal_modified['entry_datetime'] = datetime(2024, 1, 1, 9, 16, 0)
            signal_modified['exit_datetime'] = datetime(2024, 1, 1, 15, 30, 0)
            
            query = self.query_builder.build_query(signal_modified, tv_config)
            assert query is not None, f"Query generation must succeed for signal {signal['trade_no']}"
            queries.append(query)
        
        # Step 5: Execute queries on REAL HeavyDB
        results = []
        for i, query in enumerate(queries):
            try:
                result = heavydb_connection.execute(query)
                assert result is not None, f"Query {i+1} must return results"
                results.append(result)
            except Exception as e:
                pytest.fail(f"HeavyDB query execution failed: {e}")
        
        # Step 6: Process results
        processed_results = []
        for i, result in enumerate(results):
            processed_result = self.processor.process_result(result, processed_signals[i])
            assert processed_result is not None, f"Result processing must succeed for query {i+1}"
            processed_results.append(processed_result)
        
        # Validate final results
        assert len(processed_results) == 2, "Should have 2 processed results"
        
        # Check result structure
        for result in processed_results:
            required_fields = ['trade_no', 'pnl', 'entry_price', 'exit_price', 'entry_time', 'exit_time']
            for field in required_fields:
                assert field in result, f"Result missing required field: {field}"
        
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_heavydb_connection_validation(self, heavydb_connection):
        """Test HeavyDB connection and data availability for TV strategy"""
        
        # Test basic connection
        assert heavydb_connection is not None, "HeavyDB connection must be available"
        
        # Test NIFTY data availability
        nifty_count_query = """
        SELECT COUNT(*) as total_rows 
        FROM nifty_option_chain 
        WHERE trade_date >= DATE '2024-01-01'
        """
        
        result = heavydb_connection.execute(nifty_count_query)
        assert result is not None, "NIFTY data query must succeed"
        assert len(result) > 0, "Must have NIFTY data results"
        
        total_rows = result[0]['total_rows']
        assert total_rows > 1000000, f"Must have substantial NIFTY data (got {total_rows:,} rows)"
        
        # Test specific date availability for TV signals
        date_check_query = """
        SELECT DISTINCT trade_date 
        FROM nifty_option_chain 
        WHERE trade_date = DATE '2024-01-01'
        """
        
        date_result = heavydb_connection.execute(date_check_query)
        assert date_result is not None and len(date_result) > 0, \
            "January 1, 2024 data must be available for TV signal testing"
        
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_tv_query_generation_real_data(self, real_config_files, heavydb_connection):
        """Test TV query generation with REAL configuration and validate against HeavyDB"""
        
        # Load real configuration
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        processed_signals = self.signal_processor.process_signals(raw_signals, tv_config)
        
        # Test query generation for each signal
        for signal in processed_signals:
            # Modify to use available date in HeavyDB
            signal_test = signal.copy()
            signal_test['entry_datetime'] = datetime(2024, 1, 1, 9, 16, 0)
            signal_test['exit_datetime'] = datetime(2024, 1, 1, 15, 30, 0)
            
            # Generate query
            query = self.query_builder.build_query(signal_test, tv_config)
            assert query is not None, f"Query generation failed for signal {signal['trade_no']}"
            assert isinstance(query, str), "Query must be string"
            assert len(query) > 100, "Query must be substantial"
            
            # Validate query contains expected elements
            assert "nifty_option_chain" in query.lower(), "Query must reference NIFTY option chain"
            assert "trade_date" in query.lower(), "Query must filter by trade date"
            assert "trade_time" in query.lower(), "Query must filter by trade time"
            
            # Test query syntax by executing on HeavyDB
            try:
                result = heavydb_connection.execute(f"EXPLAIN {query}")
                assert result is not None, f"Query syntax validation failed for {signal['trade_no']}"
            except Exception as e:
                pytest.fail(f"Generated query has syntax errors: {e}\nQuery: {query}")
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_database_exit_timing_real_data(self, real_config_files, heavydb_connection):
        """Test database exit timing feature with REAL HeavyDB data"""
        
        # Load configuration with database exit timing enabled
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        tv_config['use_db_exit_timing'] = True
        tv_config['exit_search_interval'] = 5  # 5 minutes
        tv_config['exit_price_source'] = 'SPOT'
        
        # Test database exit timing query generation
        test_signal = {
            'trade_no': 'TEST_DB_EXIT',
            'direction': 'LONG',
            'entry_datetime': datetime(2024, 1, 1, 9, 16, 0),
            'exit_datetime': datetime(2024, 1, 1, 12, 0, 0),
            'lots': 1
        }
        
        # Test precise exit time detection
        precise_exit_query = self.query_builder._build_precise_exit_query(
            test_signal, tv_config
        )
        assert precise_exit_query is not None, "Precise exit query generation must succeed"
        
        # Execute on real HeavyDB
        try:
            result = heavydb_connection.execute(precise_exit_query)
            # Result may be empty if no precise exit found, but query must be valid
            assert result is not None, "Precise exit query must execute successfully"
        except Exception as e:
            pytest.fail(f"Precise exit query failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_strike_selection_integration_real_data(self, real_config_files, heavydb_connection):
        """Test strike selection logic integration with REAL HeavyDB data"""
        
        # Load real TBS strategy configuration
        strategy_path = real_config_files['strategy']
        
        # Parse strategy configuration
        general_params = pd.read_excel(str(strategy_path), sheet_name='GeneralParameter')
        leg_params = pd.read_excel(str(strategy_path), sheet_name='LegParameter')
        
        # Validate strategy configuration structure
        assert len(general_params) > 0, "General parameters must exist"
        assert len(leg_params) > 0, "Leg parameters must exist"
        
        # Check strike selection method from real config
        first_leg = leg_params.iloc[0]
        assert first_leg['StrikeMethod'] == 'ATM', "Real config uses ATM strike method"
        assert first_leg['StrikeValue'] == 0, "ATM strike value should be 0"
        
        # Test ATM strike calculation with real data
        atm_query = """
        SELECT 
            trade_date,
            trade_time, 
            index_spot,
            MIN(ABS(strike_price - index_spot)) as atm_diff,
            MIN(CASE WHEN ABS(strike_price - index_spot) = MIN(ABS(strike_price - index_spot)) 
                THEN strike_price END) as atm_strike
        FROM nifty_option_chain
        WHERE trade_date = DATE '2024-01-01'
        AND trade_time = TIME '09:16:00'
        AND index_spot IS NOT NULL
        AND strike_price IS NOT NULL
        GROUP BY trade_date, trade_time, index_spot
        LIMIT 1
        """
        
        result = heavydb_connection.execute(atm_query)
        assert result is not None and len(result) > 0, "ATM calculation must work with real data"
        
        atm_data = result[0]
        assert 'atm_strike' in atm_data, "ATM strike must be calculated"
        assert atm_data['atm_strike'] > 0, "ATM strike must be positive"
        
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_portfolio_strategy_integration_real_files(self, real_config_files):
        """Test portfolio and strategy file integration with REAL configuration files"""
        
        # Test long portfolio integration
        long_portfolio_path = real_config_files['portfolio_long']
        strategy_settings = pd.read_excel(str(long_portfolio_path), sheet_name='StrategySetting')
        
        # Verify strategy file reference
        strategy_file_ref = strategy_settings.iloc[0]['StrategyExcelFilePath']
        assert strategy_file_ref == 'TV_CONFIG_STRATEGY_LONG_1.0.0.xlsx', \
            "Long portfolio must reference correct strategy file"
        
        # Test short portfolio integration
        short_portfolio_path = real_config_files['portfolio_short']
        short_strategy_settings = pd.read_excel(str(short_portfolio_path), sheet_name='StrategySetting')
        
        short_strategy_file_ref = short_strategy_settings.iloc[0]['StrategyExcelFilePath']
        assert isinstance(short_strategy_file_ref, str), "Short portfolio must have strategy reference"
        
        # Test manual portfolio integration
        manual_portfolio_path = real_config_files['portfolio_manual']
        manual_strategy_settings = pd.read_excel(str(manual_portfolio_path), sheet_name='StrategySetting')
        
        manual_strategy_file_ref = manual_strategy_settings.iloc[0]['StrategyExcelFilePath']
        assert isinstance(manual_strategy_file_ref, str), "Manual portfolio must have strategy reference"
        
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_6_file_dependency_chain_real_config(self, real_config_files):
        """Test complete 6-file dependency chain with REAL configuration files"""
        
        # File 1: TV Master Config
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        
        # File 2: Signals (referenced by TV Master)
        signals_file = tv_config['signal_file_path']
        assert signals_file == 'TV_CONFIG_SIGNALS_1.0.0.xlsx'
        signals_path = real_config_files['tv_master'].parent / signals_file
        assert signals_path.exists(), f"Referenced signals file must exist: {signals_file}"
        
        # File 3: Long Portfolio (referenced by TV Master)
        long_portfolio_file = tv_config['long_portfolio_file_path']
        assert long_portfolio_file == 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx'
        long_portfolio_path = real_config_files['tv_master'].parent / long_portfolio_file
        assert long_portfolio_path.exists(), f"Referenced long portfolio must exist: {long_portfolio_file}"
        
        # File 4: Short Portfolio (referenced by TV Master)
        short_portfolio_file = tv_config['short_portfolio_file_path']
        assert short_portfolio_file == 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx'
        short_portfolio_path = real_config_files['tv_master'].parent / short_portfolio_file
        assert short_portfolio_path.exists(), f"Referenced short portfolio must exist: {short_portfolio_file}"
        
        # Files 5 & 6: Strategy files (referenced by portfolios)
        long_strategy_df = pd.read_excel(str(long_portfolio_path), sheet_name='StrategySetting')
        long_strategy_file = long_strategy_df.iloc[0]['StrategyExcelFilePath']
        
        # Validate complete chain integrity
        assert all([
            tv_config['enabled'],
            len(tv_config['signal_file_path']) > 0,
            len(tv_config['long_portfolio_file_path']) > 0,
            len(tv_config['short_portfolio_file_path']) > 0,
            isinstance(long_strategy_file, str),
            len(long_strategy_file) > 0
        ]), "Complete 6-file chain must be valid"


class TestTVHeavyDBPerformance:
    """Performance tests with REAL HeavyDB and REAL configuration"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
        self.signal_processor = SignalProcessor()
        self.query_builder = TVQueryBuilder()
        self.processor = TVProcessor()
    
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_end_to_end_performance_real_data(self, real_config_files, heavydb_connection):
        """Test end-to-end performance with REAL configuration and HeavyDB"""
        
        start_time = time.time()
        
        # Step 1: Parse configuration (should be fast)
        config_start = time.time()
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        config_time = time.time() - config_start
        
        # Step 2: Process signals (should be fast)
        processing_start = time.time()
        processed_signals = self.signal_processor.process_signals(raw_signals, tv_config)
        processing_time = time.time() - processing_start
        
        # Step 3: Generate queries (should be fast)
        query_start = time.time()
        queries = []
        for signal in processed_signals:
            signal_modified = signal.copy()
            signal_modified['entry_datetime'] = datetime(2024, 1, 1, 9, 16, 0)
            signal_modified['exit_datetime'] = datetime(2024, 1, 1, 15, 30, 0)
            
            query = self.query_builder.build_query(signal_modified, tv_config)
            queries.append(query)
        query_time = time.time() - query_start
        
        # Step 4: Execute on HeavyDB (critical performance test)
        db_start = time.time()
        results = []
        for query in queries:
            result = heavydb_connection.execute(query)
            results.append(result)
        db_time = time.time() - db_start
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert config_time < 0.5, f"Config parsing too slow: {config_time:.3f}s"
        assert processing_time < 0.1, f"Signal processing too slow: {processing_time:.3f}s"
        assert query_time < 0.1, f"Query generation too slow: {query_time:.3f}s"
        assert db_time < 3.0, f"HeavyDB execution too slow: {db_time:.3f}s"
        assert total_time < 3.0, f"Total processing too slow: {total_time:.3f}s"
        
        # Validate results
        assert len(results) == len(processed_signals), "All queries must return results"
        
        print(f"Performance summary:")
        print(f"  Config parsing: {config_time:.3f}s")
        print(f"  Signal processing: {processing_time:.3f}s")
        print(f"  Query generation: {query_time:.3f}s")
        print(f"  HeavyDB execution: {db_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
    
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_heavydb_query_performance_real_data(self, heavydb_connection):
        """Test HeavyDB query performance with TV-specific query patterns"""
        
        # Test basic option chain query performance
        basic_query = """
        SELECT 
            trade_date, trade_time, index_spot, strike_price,
            call_ltp, put_ltp, call_volume, put_volume
        FROM nifty_option_chain
        WHERE trade_date = DATE '2024-01-01'
        AND trade_time BETWEEN TIME '09:15:00' AND TIME '15:30:00'
        AND strike_price BETWEEN 21000 AND 22000
        ORDER BY trade_time, strike_price
        LIMIT 1000
        """
        
        start_time = time.time()
        result = heavydb_connection.execute(basic_query)
        query_time = time.time() - start_time
        
        assert result is not None, "Basic query must succeed"
        assert query_time < 1.0, f"Basic query too slow: {query_time:.3f}s"
        
        # Test ATM calculation performance
        atm_query = """
        WITH atm_calc AS (
            SELECT 
                trade_date, trade_time, index_spot,
                strike_price,
                ABS(strike_price - index_spot) as strike_diff
            FROM nifty_option_chain
            WHERE trade_date = DATE '2024-01-01'
            AND trade_time = TIME '09:16:00'
            AND index_spot IS NOT NULL
        )
        SELECT 
            trade_date, trade_time, index_spot,
            MIN(strike_diff) as min_diff,
            MIN(CASE WHEN strike_diff = MIN(strike_diff) THEN strike_price END) as atm_strike
        FROM atm_calc
        GROUP BY trade_date, trade_time, index_spot
        """
        
        start_time = time.time()
        atm_result = heavydb_connection.execute(atm_query)
        atm_time = time.time() - start_time
        
        assert atm_result is not None, "ATM query must succeed"
        assert atm_time < 0.5, f"ATM calculation too slow: {atm_time:.3f}s"
        
        print(f"Query performance:")
        print(f"  Basic query: {query_time:.3f}s")
        print(f"  ATM calculation: {atm_time:.3f}s")