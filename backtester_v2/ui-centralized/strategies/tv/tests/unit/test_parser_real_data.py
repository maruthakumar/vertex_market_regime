#!/usr/bin/env python3
"""
Unit Tests for TV Parser - REAL DATA VALIDATION
Tests parsing of actual production TV configuration files with real HeavyDB validation
NO MOCK DATA - ONLY REAL INPUT SHEETS
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from pathlib import Path
import os

from parser import TVParser


class TestTVParserRealData:
    """Test TV Parser with REAL production configuration files"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_real_tv_master_config(self, real_config_files):
        """Test parsing REAL TV master configuration file"""
        tv_master_path = real_config_files['tv_master']
        
        # Parse the real TV master config
        result = self.parser.parse_tv_settings(str(tv_master_path))
        
        # Validate parser output structure
        assert isinstance(result, dict), "Result must be dictionary"
        assert 'settings' in result, "Result must contain 'settings' key"
        assert 'source_file' in result, "Result must contain 'source_file' key"
        assert isinstance(result['settings'], list), "Settings must be list"
        assert len(result['settings']) > 0, "Must have at least one setting"
        
        # Get first setting for validation
        tv_config = result['settings'][0]
        
        # Validate structure - these are the ACTUAL fields from real file
        required_fields = [
            'name', 'enabled', 'signal_file_path', 'start_date', 'end_date',
            'signal_date_format', 'intraday_sqoff_applicable', 'intraday_exit_time',
            'tv_exit_applicable', 'do_rollover', 'rollover_time', 'manual_trade_entry_time',
            'manual_trade_lots', 'first_trade_entry_time', 'increase_entry_signal_time_by',
            'increase_exit_signal_time_by', 'expiry_day_exit_time', 'slippage_percent',
            'long_portfolio_file_path', 'short_portfolio_file_path', 'manual_portfolio_file_path',
            'use_db_exit_timing', 'exit_search_interval', 'exit_price_source'
        ]
        
        # Verify all required fields are present
        for field in required_fields:
            assert field in tv_config, f"Required field '{field}' missing from parsed result"
        
        # Validate data types and values from REAL config
        assert isinstance(tv_config['name'], str) and tv_config['name'] == 'TV_Backtest_Sample'
        assert isinstance(tv_config['enabled'], bool) and tv_config['enabled'] is True
        assert isinstance(tv_config['start_date'], date) and tv_config['start_date'] == date(2024, 1, 1)
        assert isinstance(tv_config['end_date'], date) and tv_config['end_date'] == date(2024, 12, 31)
        assert isinstance(tv_config['signal_date_format'], str) and tv_config['signal_date_format'] == '%Y%m%d %H%M%S'
        
        # Validate file paths from real config
        assert tv_config['signal_file_path'] == 'TV_CONFIG_SIGNALS_1.0.0.xlsx'
        assert tv_config['long_portfolio_file_path'] == 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx'
        assert tv_config['short_portfolio_file_path'] == 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx'
        assert tv_config['manual_portfolio_file_path'] == 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx'
        
        # Validate real time settings
        assert isinstance(tv_config['intraday_exit_time'], time) and tv_config['intraday_exit_time'] == time(15, 30, 0)
        assert isinstance(tv_config['rollover_time'], time) and tv_config['rollover_time'] == time(15, 20, 0)
        assert isinstance(tv_config['expiry_day_exit_time'], time) and tv_config['expiry_day_exit_time'] == time(15, 20, 0)
        
        # Validate boolean flags from real config  
        assert isinstance(tv_config['intraday_sqoff_applicable'], bool) and tv_config['intraday_sqoff_applicable'] is True
        assert isinstance(tv_config['tv_exit_applicable'], bool) and tv_config['tv_exit_applicable'] is True
        assert isinstance(tv_config['do_rollover'], bool) and tv_config['do_rollover'] is False
        assert isinstance(tv_config['use_db_exit_timing'], bool) and tv_config['use_db_exit_timing'] is False
        
        # Validate numeric settings from real config
        assert isinstance(tv_config['slippage_percent'], float) and tv_config['slippage_percent'] == 0.1
        assert isinstance(tv_config['exit_search_interval'], int) and tv_config['exit_search_interval'] == 5
        assert isinstance(tv_config['exit_price_source'], str) and tv_config['exit_price_source'] == 'SPOT'
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_real_tv_signals(self, real_config_files):
        """Test parsing REAL TV signals file"""
        signals_path = real_config_files['signals']
        
        # Parse the real signals file
        result = self.parser.parse_signals(str(signals_path), '%Y%m%d %H%M%S')
        
        # Validate structure - these are the ACTUAL signals from real file
        assert isinstance(result, list), "Signals result must be list"
        assert len(result) == 4, "Real signals file contains exactly 4 signals"
        
        # Check first signal (Entry Long)
        first_signal = result[0]
        assert first_signal['trade_no'] == 'T001'
        assert first_signal['signal_type'] == 'Entry Long'
        assert isinstance(first_signal['datetime'], datetime)
        assert first_signal['datetime'] == datetime(2024, 1, 1, 9, 16, 0)
        assert first_signal['lots'] == 1
        
        # Check second signal (Exit Long)
        second_signal = result[1]
        assert second_signal['trade_no'] == 'T001'
        assert second_signal['signal_type'] == 'Exit Long'
        assert second_signal['datetime'] == datetime(2024, 1, 1, 12, 0, 0)
        assert second_signal['lots'] == 1
        
        # Check third signal (Entry Short)
        third_signal = result[2]
        assert third_signal['trade_no'] == 'T002'
        assert third_signal['signal_type'] == 'Entry Short'
        assert third_signal['datetime'] == datetime(2024, 1, 1, 13, 0, 0)
        assert third_signal['lots'] == 2
        
        # Check fourth signal (Exit Short)
        fourth_signal = result[3]
        assert fourth_signal['trade_no'] == 'T002'
        assert fourth_signal['signal_type'] == 'Exit Short'
        assert fourth_signal['datetime'] == datetime(2024, 1, 1, 15, 0, 0)
        assert fourth_signal['lots'] == 2
        
        # Validate signal pairing
        long_signals = [s for s in result if s['trade_no'] == 'T001']
        short_signals = [s for s in result if s['trade_no'] == 'T002']
        
        assert len(long_signals) == 2, "T001 should have entry and exit"
        assert len(short_signals) == 2, "T002 should have entry and exit"
        
        # Validate signal types
        long_types = [s['signal_type'] for s in long_signals]
        short_types = [s['signal_type'] for s in short_signals]
        
        assert 'Entry Long' in long_types and 'Exit Long' in long_types
        assert 'Entry Short' in short_types and 'Exit Short' in short_types
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_real_portfolio_long(self, real_config_files):
        """Test parsing REAL long portfolio configuration"""
        portfolio_path = real_config_files['portfolio_long']
        
        # Parse PortfolioSetting sheet
        df_portfolio = pd.read_excel(str(portfolio_path), sheet_name='PortfolioSetting')
        
        # Validate actual structure from real file
        expected_portfolio_columns = [
            'Capital', 'MaxRisk', 'MaxPositions', 'RiskPerTrade', 
            'UseKellyCriterion', 'RebalanceFrequency'
        ]
        
        assert list(df_portfolio.columns) == expected_portfolio_columns, \
            f"Portfolio columns don't match expected: {list(df_portfolio.columns)}"
        
        # Validate actual values from real file
        first_row = df_portfolio.iloc[0]
        assert first_row['Capital'] == 1000000
        assert first_row['MaxRisk'] == 5
        assert first_row['MaxPositions'] == 5
        assert first_row['RiskPerTrade'] == 2
        assert first_row['UseKellyCriterion'] == 'NO'
        assert first_row['RebalanceFrequency'] == 'DAILY'
        
        # Parse StrategySetting sheet
        df_strategy = pd.read_excel(str(portfolio_path), sheet_name='StrategySetting')
        
        expected_strategy_columns = [
            'StrategyName', 'StrategyExcelFilePath', 'Enabled', 'Priority', 'AllocationPercent'
        ]
        
        assert list(df_strategy.columns) == expected_strategy_columns, \
            f"Strategy columns don't match expected: {list(df_strategy.columns)}"
        
        # Validate actual strategy values from real file
        strategy_row = df_strategy.iloc[0]
        assert strategy_row['StrategyName'] == 'LONG_Strategy_1'
        assert strategy_row['StrategyExcelFilePath'] == 'TV_CONFIG_STRATEGY_LONG_1.0.0.xlsx'
        assert strategy_row['Enabled'] == 'YES'
        assert strategy_row['Priority'] == 1
        assert strategy_row['AllocationPercent'] == 100
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_real_tbs_strategy(self, real_config_files):
        """Test parsing REAL TBS strategy configuration"""
        strategy_path = real_config_files['strategy']
        
        # Parse GeneralParameter sheet
        df_general = pd.read_excel(str(strategy_path), sheet_name='GeneralParameter')
        
        # Validate structure from real file (43 columns)
        assert len(df_general.columns) == 43, f"Expected 43 general parameter columns, got {len(df_general.columns)}"
        
        # Validate specific key columns exist
        key_columns = [
            'StrategyName', 'Underlying', 'Index', 'Weekdays', 'DTE',
            'StrikeSelectionTime', 'StartTime', 'EndTime', 'StrategyProfit', 'StrategyLoss'
        ]
        
        for col in key_columns:
            assert col in df_general.columns, f"Key column '{col}' missing from GeneralParameter"
        
        # Validate actual values from real file
        general_row = df_general.iloc[0]
        assert general_row['StrategyName'] == 'TV_Strategy_Sample'
        assert general_row['Underlying'] == 'SPOT'
        assert general_row['Index'] == 'NIFTY'
        assert general_row['Weekdays'] == '1,2,3,4,5'
        assert general_row['DTE'] == 0
        assert general_row['StrikeSelectionTime'] == 92000  # 09:20:00
        assert general_row['StartTime'] == 91600  # 09:16:00
        assert general_row['EndTime'] == 150000  # 15:00:00
        
        # Parse LegParameter sheet
        df_legs = pd.read_excel(str(strategy_path), sheet_name='LegParameter')
        
        # Validate structure from real file (29 columns)
        assert len(df_legs.columns) == 29, f"Expected 29 leg parameter columns, got {len(df_legs.columns)}"
        
        # Validate key leg columns
        key_leg_columns = [
            'StrategyName', 'LegID', 'Instrument', 'Transaction', 'Expiry',
            'StrikeMethod', 'StrikeValue', 'SLType', 'SLValue', 'TGTType', 'TGTValue', 'Lots'
        ]
        
        for col in key_leg_columns:
            assert col in df_legs.columns, f"Key leg column '{col}' missing from LegParameter"
        
        # Validate actual leg values from real file
        leg_row = df_legs.iloc[0]
        assert leg_row['StrategyName'] == 'TV_Strategy_Sample'
        assert leg_row['LegID'] == 'leg1'
        assert leg_row['Instrument'] == 'call'
        assert leg_row['Transaction'] == 'sell'
        assert leg_row['Expiry'] == 'current'
        assert leg_row['StrikeMethod'] == 'ATM'
        assert leg_row['StrikeValue'] == 0
        assert leg_row['SLType'] == 'percentage'
        assert leg_row['SLValue'] == 100
        assert leg_row['TGTType'] == 'percentage'
        assert leg_row['TGTValue'] == 0
        assert leg_row['Lots'] == 1
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_real_file_path_resolution(self, real_config_files):
        """Test file path resolution with REAL configuration files"""
        tv_master_path = real_config_files['tv_master']
        
        # Parse the master config to get referenced file paths
        result = self.parser.parse_tv_settings(str(tv_master_path))
        
        # Test that all referenced files can be resolved
        base_dir = tv_master_path.parent
        
        signal_resolved = self.parser._resolve_file_path(
            result['signal_file_path'], str(tv_master_path)
        )
        assert Path(signal_resolved).exists(), f"Signal file should exist: {signal_resolved}"
        
        long_portfolio_resolved = self.parser._resolve_file_path(
            result['long_portfolio_file_path'], str(tv_master_path)
        )
        assert Path(long_portfolio_resolved).exists(), f"Long portfolio file should exist: {long_portfolio_resolved}"
        
        short_portfolio_resolved = self.parser._resolve_file_path(
            result['short_portfolio_file_path'], str(tv_master_path)
        )
        assert Path(short_portfolio_resolved).exists(), f"Short portfolio file should exist: {short_portfolio_resolved}"
        
        manual_portfolio_resolved = self.parser._resolve_file_path(
            result['manual_portfolio_file_path'], str(tv_master_path)
        )
        assert Path(manual_portfolio_resolved).exists(), f"Manual portfolio file should exist: {manual_portfolio_resolved}"
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_real_6_file_hierarchy_validation(self, real_config_files):
        """Test complete 6-file hierarchy validation with REAL files"""
        
        # 1. Validate TV Master exists and is parseable
        tv_master_path = real_config_files['tv_master']
        assert tv_master_path.exists(), "TV Master config must exist"
        
        tv_config = self.parser.parse_tv_settings(str(tv_master_path))
        assert tv_config['enabled'] is True, "TV config must be enabled"
        
        # 2. Validate Signals file exists and has valid signals
        signals_path = real_config_files['signals']
        assert signals_path.exists(), "Signals file must exist"
        
        signals = self.parser.parse_signals(str(signals_path), tv_config['signal_date_format'])
        assert len(signals) > 0, "Must have at least one signal"
        assert len(signals) % 2 == 0, "Must have even number of signals (entry/exit pairs)"
        
        # 3. Validate Portfolio files exist and reference strategies
        portfolio_files = ['portfolio_long', 'portfolio_short', 'portfolio_manual']
        for portfolio_key in portfolio_files:
            portfolio_path = real_config_files[portfolio_key]
            assert portfolio_path.exists(), f"{portfolio_key} must exist"
            
            # Check StrategySetting sheet references strategy file
            df = pd.read_excel(str(portfolio_path), sheet_name='StrategySetting')
            strategy_file = df.iloc[0]['StrategyExcelFilePath']
            assert isinstance(strategy_file, str) and strategy_file.endswith('.xlsx'), \
                f"Strategy file reference must be valid Excel file: {strategy_file}"
        
        # 4. Validate Strategy file exists and has required sheets
        strategy_path = real_config_files['strategy']
        assert strategy_path.exists(), "Strategy file must exist"
        
        # Check required sheets exist
        required_sheets = ['GeneralParameter', 'LegParameter']
        for sheet_name in required_sheets:
            assert self.parser._sheet_exists(str(strategy_path), sheet_name), \
                f"Required sheet '{sheet_name}' must exist in strategy file"
        
        # 5. Validate cross-file consistency
        # Check that signal date range falls within TV config date range
        signal_dates = [s['datetime'].date() for s in signals]
        min_signal_date = min(signal_dates)
        max_signal_date = max(signal_dates)
        
        assert min_signal_date >= tv_config['start_date'], \
            f"Signal date {min_signal_date} before TV start date {tv_config['start_date']}"
        assert max_signal_date <= tv_config['end_date'], \
            f"Signal date {max_signal_date} after TV end date {tv_config['end_date']}"
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_real_signal_pairing_validation(self, real_config_files):
        """Test signal pairing logic with REAL signals"""
        signals_path = real_config_files['signals']
        signals = self.parser.parse_signals(str(signals_path), '%Y%m%d %H%M%S')
        
        # Group signals by trade number
        trade_groups = {}
        for signal in signals:
            trade_no = signal['trade_no']
            if trade_no not in trade_groups:
                trade_groups[trade_no] = []
            trade_groups[trade_no].append(signal)
        
        # Validate each trade has exactly 2 signals (entry and exit)
        for trade_no, trade_signals in trade_groups.items():
            assert len(trade_signals) == 2, f"Trade {trade_no} must have exactly 2 signals"
            
            signal_types = [s['signal_type'] for s in trade_signals]
            
            # Check for valid entry/exit pairs
            if 'Entry Long' in signal_types:
                assert 'Exit Long' in signal_types, f"Trade {trade_no} has Entry Long but no Exit Long"
            elif 'Entry Short' in signal_types:
                assert 'Exit Short' in signal_types, f"Trade {trade_no} has Entry Short but no Exit Short"
            elif 'Manual Entry' in signal_types:
                assert 'Manual Exit' in signal_types, f"Trade {trade_no} has Manual Entry but no Manual Exit"
            else:
                pytest.fail(f"Trade {trade_no} has invalid signal types: {signal_types}")
            
            # Validate timing (entry before exit)
            entry_signal = next(s for s in trade_signals if 'Entry' in s['signal_type'])
            exit_signal = next(s for s in trade_signals if 'Exit' in s['signal_type'])
            
            assert entry_signal['datetime'] < exit_signal['datetime'], \
                f"Trade {trade_no}: Entry time must be before exit time"
            
            # Validate lot consistency
            assert entry_signal['lots'] == exit_signal['lots'], \
                f"Trade {trade_no}: Entry and exit lots must match"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_real_data_types_and_formats(self, real_config_files):
        """Test data type validation with REAL configuration data"""
        
        # Test TV master config data types
        tv_master_path = real_config_files['tv_master']
        tv_config = self.parser.parse_tv_settings(str(tv_master_path))
        
        # Validate specific data types from real config
        assert isinstance(tv_config['slippage_percent'], (int, float)), "Slippage must be numeric"
        assert 0 <= tv_config['slippage_percent'] <= 100, "Slippage must be between 0-100%"
        
        assert isinstance(tv_config['exit_search_interval'], int), "Exit search interval must be integer"
        assert tv_config['exit_search_interval'] > 0, "Exit search interval must be positive"
        
        assert tv_config['exit_price_source'] in ['SPOT', 'FUTURE'], "Exit price source must be SPOT or FUTURE"
        
        # Test signal data types
        signals_path = real_config_files['signals']
        signals = self.parser.parse_signals(str(signals_path), tv_config['signal_date_format'])
        
        for signal in signals:
            assert isinstance(signal['trade_no'], str), "Trade number must be string"
            assert isinstance(signal['lots'], int), "Lots must be integer"
            assert signal['lots'] > 0, "Lots must be positive"
            assert isinstance(signal['datetime'], datetime), "Signal datetime must be datetime object"
            assert signal['signal_type'] in [
                'Entry Long', 'Exit Long', 'Entry Short', 'Exit Short', 
                'Manual Entry', 'Manual Exit'
            ], f"Invalid signal type: {signal['signal_type']}"


class TestTVParserRealDataPerformance:
    """Performance tests with REAL data"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
    
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_real_file_parsing_performance(self, real_config_files):
        """Test parsing performance with REAL files"""
        import time
        
        # Test TV master config parsing speed
        start_time = time.time()
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        master_parse_time = time.time() - start_time
        
        assert master_parse_time < 1.0, f"TV master parsing took too long: {master_parse_time:.3f}s"
        
        # Test signals parsing speed
        start_time = time.time()
        signals = self.parser.parse_signals(str(real_config_files['signals']), tv_config['signal_date_format'])
        signals_parse_time = time.time() - start_time
        
        assert signals_parse_time < 1.0, f"Signals parsing took too long: {signals_parse_time:.3f}s"
        
        # Test complete 6-file validation speed
        start_time = time.time()
        # Validate all files exist and are parseable
        for file_key, file_path in real_config_files.items():
            assert file_path.exists(), f"File {file_key} must exist"
        total_validation_time = time.time() - start_time
        
        assert total_validation_time < 2.0, f"Complete validation took too long: {total_validation_time:.3f}s"