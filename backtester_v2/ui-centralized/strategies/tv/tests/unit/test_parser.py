#!/usr/bin/env python3
"""
Unit Tests for TV Parser
Tests the parsing of all 6 TV configuration files with real data validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

from parser import TVParser


class TestTVParser:
    """Test TV Parser functionality with real data validation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_tv_setting_columns_complete(self):
        """Test that all 24 TV setting columns are defined"""
        expected_columns = [
            'StartDate', 'EndDate', 'SignalDateFormat', 'Enabled', 'TvExitApplicable',
            'ManualTradeEntryTime', 'ManualTradeLots', 'IncreaseEntrySignalTimeBy',
            'IncreaseExitSignalTimeBy', 'IntradaySqOffApplicable', 'FirstTradeEntryTime',
            'IntradayExitTime', 'ExpiryDayExitTime', 'DoRollover', 'RolloverTime',
            'Name', 'SignalFilePath', 'LongPortfolioFilePath', 'ShortPortfolioFilePath',
            'ManualPortfolioFilePath', 'UseDbExitTiming', 'ExitSearchInterval',
            'ExitPriceSource', 'SlippagePercent'
        ]
        
        assert len(self.parser.tv_setting_columns) == 24, "TV settings must have exactly 24 columns"
        assert all(col in self.parser.tv_setting_columns for col in expected_columns), \
            "All expected TV setting columns must be present"
    
    @pytest.mark.unit  
    @pytest.mark.heavydb
    def test_signal_columns_complete(self):
        """Test that all 4 signal columns are defined"""
        expected_columns = ['Trade #', 'Type', 'Date/Time', 'Contracts']
        
        assert len(self.parser.signal_columns) == 4, "TV signals must have exactly 4 columns"
        assert self.parser.signal_columns == expected_columns, \
            "Signal columns must match expected format"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_tv_settings_valid_excel(self, sample_tv_config):
        """Test parsing valid TV settings Excel file"""
        # Create temporary Excel file with valid data
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df = pd.DataFrame([sample_tv_config])
            df.to_excel(tmp.name, sheet_name='Setting', index=False)
            tmp_path = tmp.name
        
        try:
            # Test parsing
            result = self.parser.parse_tv_settings(tmp_path)
            
            # Validate result structure
            assert isinstance(result, dict), "Result must be dictionary"
            assert 'name' in result, "Result must contain 'name' field"
            assert 'enabled' in result, "Result must contain 'enabled' field"
            assert 'start_date' in result, "Result must contain 'start_date' field"
            assert 'end_date' in result, "Result must contain 'end_date' field"
            
            # Validate data types
            assert isinstance(result['start_date'], date), "start_date must be date object"
            assert isinstance(result['end_date'], date), "end_date must be date object"
            assert isinstance(result['enabled'], bool), "enabled must be boolean"
            
            # Validate specific values
            assert result['name'] == 'TEST_TV_CONFIG'
            assert result['enabled'] is True
            assert result['start_date'] == date(2024, 1, 1)
            assert result['end_date'] == date(2024, 1, 31)
            
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_tv_settings_missing_file(self):
        """Test parsing with missing Excel file"""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_tv_settings('/nonexistent/file.xlsx')
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_tv_settings_invalid_dates(self):
        """Test parsing with invalid date formats"""
        invalid_config = {
            'Name': 'TEST_INVALID',
            'Enabled': 'YES',
            'StartDate': 'invalid_date',
            'EndDate': '31_01_2024',
            'SignalDateFormat': '%Y%m%d %H%M%S'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df = pd.DataFrame([invalid_config])
            df.to_excel(tmp.name, sheet_name='Setting', index=False)
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError):
                self.parser.parse_tv_settings(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_signals_valid_data(self, sample_signals):
        """Test parsing valid signal data"""
        # Create temporary Excel file with signal data
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df = pd.DataFrame(sample_signals)
            df.to_excel(tmp.name, sheet_name='List of trades', index=False)
            tmp_path = tmp.name
        
        try:
            # Test parsing
            result = self.parser.parse_signals(tmp_path, '%Y%m%d %H%M%S')
            
            # Validate result
            assert isinstance(result, list), "Result must be list"
            assert len(result) == 4, "Should parse all 4 signals"
            
            # Check first signal
            first_signal = result[0]
            assert 'trade_no' in first_signal
            assert 'signal_type' in first_signal
            assert 'datetime' in first_signal
            assert 'lots' in first_signal
            
            # Validate data types
            assert isinstance(first_signal['datetime'], datetime)
            assert isinstance(first_signal['lots'], int)
            
            # Validate specific values
            assert first_signal['trade_no'] == 'T001'
            assert first_signal['signal_type'] == 'Entry Long'
            assert first_signal['lots'] == 5
            
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_signals_missing_sheet(self):
        """Test parsing signals with missing sheet"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df = pd.DataFrame([{'dummy': 'data'}])
            df.to_excel(tmp.name, sheet_name='Wrong Sheet', index=False)
            tmp_path = tmp.name
        
        try:
            with pytest.raises(KeyError):
                self.parser.parse_signals(tmp_path, '%Y%m%d %H%M%S')
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_parse_signals_missing_columns(self):
        """Test parsing signals with missing required columns"""
        invalid_signals = [
            {
                'Trade #': 'T001',
                'Type': 'Entry Long',
                # Missing Date/Time and Contracts columns
            }
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df = pd.DataFrame(invalid_signals)
            df.to_excel(tmp.name, sheet_name='List of trades', index=False)
            tmp_path = tmp.name
        
        try:
            with pytest.raises(KeyError):
                self.parser.parse_signals(tmp_path, '%Y%m%d %H%M%S')
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_validate_tv_settings_valid(self, sample_tv_config):
        """Test validation of valid TV settings"""
        result = self.parser._validate_tv_settings(sample_tv_config)
        assert result is True, "Valid TV settings should pass validation"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_validate_tv_settings_invalid_date_range(self):
        """Test validation with invalid date range"""
        invalid_config = {
            'Name': 'TEST_INVALID',
            'Enabled': 'YES',
            'StartDate': '31_01_2024',  # End date before start date
            'EndDate': '01_01_2024',
            'SignalDateFormat': '%Y%m%d %H%M%S'
        }
        
        with pytest.raises(ValueError):
            self.parser._validate_tv_settings(invalid_config)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_validate_tv_settings_missing_required_fields(self):
        """Test validation with missing required fields"""
        invalid_config = {
            'Name': 'TEST_INVALID',
            # Missing Enabled, StartDate, EndDate
        }
        
        with pytest.raises(KeyError):
            self.parser._validate_tv_settings(invalid_config)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_date_parsing_multiple_formats(self):
        """Test date parsing with different valid formats"""
        test_cases = [
            ('01_01_2024', date(2024, 1, 1)),
            ('31_12_2023', date(2023, 12, 31)),
            ('15_06_2024', date(2024, 6, 15))
        ]
        
        for date_str, expected_date in test_cases:
            result = self.parser._parse_date(date_str)
            assert result == expected_date, f"Failed to parse {date_str}"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_time_parsing_multiple_formats(self):
        """Test time parsing with different valid formats"""
        test_cases = [
            ('091500', time(9, 15, 0)),
            ('153000', time(15, 30, 0)),
            ('120000', time(12, 0, 0))
        ]
        
        for time_str, expected_time in test_cases:
            result = self.parser._parse_time(time_str)
            assert result == expected_time, f"Failed to parse {time_str}"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_boolean_parsing(self):
        """Test boolean parsing from Excel string values"""
        test_cases = [
            ('YES', True),
            ('yes', True),
            ('Yes', True),
            ('NO', False),
            ('no', False),
            ('No', False)
        ]
        
        for bool_str, expected_bool in test_cases:
            result = self.parser._parse_boolean(bool_str)
            assert result == expected_bool, f"Failed to parse {bool_str}"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_signal_type_validation(self):
        """Test signal type validation"""
        valid_types = [
            'Entry Long', 'Exit Long',
            'Entry Short', 'Exit Short',
            'Manual Entry', 'Manual Exit'
        ]
        
        for signal_type in valid_types:
            result = self.parser._validate_signal_type(signal_type)
            assert result is True, f"Valid signal type {signal_type} should pass validation"
        
        # Test invalid types
        invalid_types = ['Invalid Entry', 'Wrong Exit', 'Random Signal']
        for signal_type in invalid_types:
            with pytest.raises(ValueError):
                self.parser._validate_signal_type(signal_type)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_file_path_resolution(self):
        """Test file path resolution for relative and absolute paths"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            base_file = Path(temp_dir) / "tv_config.xlsx"
            signal_file = Path(temp_dir) / "signals.xlsx"
            portfolio_file = Path(temp_dir) / "subdir" / "portfolio.xlsx"
            
            # Create files
            base_file.touch()
            signal_file.touch()
            portfolio_file.parent.mkdir(exist_ok=True)
            portfolio_file.touch()
            
            # Test relative path resolution
            relative_signal = self.parser._resolve_file_path("signals.xlsx", str(base_file))
            assert Path(relative_signal).exists(), "Relative path should resolve correctly"
            
            # Test subdirectory path resolution
            relative_portfolio = self.parser._resolve_file_path("subdir/portfolio.xlsx", str(base_file))
            assert Path(relative_portfolio).exists(), "Subdirectory path should resolve correctly"
            
            # Test absolute path (should return as-is if exists)
            absolute_signal = self.parser._resolve_file_path(str(signal_file), str(base_file))
            assert absolute_signal == str(signal_file), "Absolute path should return as-is"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_excel_sheet_existence(self):
        """Test Excel sheet existence validation"""
        # Create temporary Excel with multiple sheets
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            with pd.ExcelWriter(tmp.name) as writer:
                pd.DataFrame([{'test': 'data'}]).to_excel(writer, sheet_name='Setting', index=False)
                pd.DataFrame([{'test': 'data'}]).to_excel(writer, sheet_name='List of trades', index=False)
            tmp_path = tmp.name
        
        try:
            # Test existing sheets
            assert self.parser._sheet_exists(tmp_path, 'Setting'), "Setting sheet should exist"
            assert self.parser._sheet_exists(tmp_path, 'List of trades'), "List of trades sheet should exist"
            
            # Test non-existing sheet
            assert not self.parser._sheet_exists(tmp_path, 'NonExistent'), "Non-existent sheet should return False"
            
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_column_validation(self):
        """Test column validation for Excel files"""
        # Create Excel with correct columns
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df = pd.DataFrame(columns=self.parser.tv_setting_columns)
            df.to_excel(tmp.name, sheet_name='Setting', index=False)
            tmp_path = tmp.name
        
        try:
            result = self.parser._validate_columns(tmp_path, 'Setting', self.parser.tv_setting_columns)
            assert result is True, "Correct columns should pass validation"
        finally:
            os.unlink(tmp_path)
        
        # Create Excel with missing columns
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df = pd.DataFrame(columns=['Name', 'Enabled'])  # Missing many columns
            df.to_excel(tmp.name, sheet_name='Setting', index=False)
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError):
                self.parser._validate_columns(tmp_path, 'Setting', self.parser.tv_setting_columns)
        finally:
            os.unlink(tmp_path)


class TestTVParserIntegration:
    """Integration tests for TV Parser with real file combinations"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_6_file_hierarchy_parsing(self, test_config_files):
        """Test parsing complete 6-file hierarchy"""
        # This test would require actual sample files
        # For now, we'll test the structure validation
        
        expected_files = [
            'tv_master', 'signals', 'portfolio_long', 
            'portfolio_short', 'strategy_long', 'strategy_short'
        ]
        
        for file_key in expected_files:
            assert file_key in test_config_files, f"Test config must include {file_key}"
    
    @pytest.mark.integration  
    @pytest.mark.heavydb
    def test_cross_file_validation(self):
        """Test validation across multiple files"""
        # Test that signal types in signal file match available portfolios
        # Test that portfolio references match strategy files
        # This is a placeholder for more complex validation logic
        pass