#!/usr/bin/env python3
"""
Tests for Excel to YAML Converter - REAL DATA VALIDATION
Tests conversion of REAL TV configuration files to YAML format with validation
"""

import pytest
import yaml
import json
from pathlib import Path
import tempfile
import os
from datetime import datetime, date, time

from excel_to_yaml_converter import TVExcelToYAMLConverter


class TestTVExcelToYAMLConverter:
    """Test Excel to YAML conversion with REAL configuration files"""
    
    def setup_method(self):
        """Setup for each test"""
        self.converter = TVExcelToYAMLConverter()
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_convert_real_tv_master_config_to_yaml(self, real_config_files):
        """Test converting REAL TV master configuration to YAML"""
        
        tv_master_path = real_config_files['tv_master']
        
        # Convert to YAML
        yaml_output = self.converter.convert_tv_master_to_yaml(str(tv_master_path))
        
        # Validate YAML structure
        assert isinstance(yaml_output, dict), "YAML output must be dictionary"
        assert 'tv_configuration' in yaml_output, "Must have tv_configuration root"
        
        tv_config = yaml_output['tv_configuration']
        
        # Validate required sections
        required_sections = [
            'master_settings', 'signal_configuration', 'portfolio_mappings',
            'execution_settings', 'timing_settings', 'rollover_settings'
        ]
        
        for section in required_sections:
            assert section in tv_config, f"Missing required section: {section}"
        
        # Validate master settings from real config
        master_settings = tv_config['master_settings']
        assert master_settings['name'] == 'TV_Backtest_Sample'
        assert master_settings['enabled'] is True
        assert master_settings['date_range']['start'] == '2024-01-01'
        assert master_settings['date_range']['end'] == '2024-12-31'
        
        # Validate signal configuration from real config
        signal_config = tv_config['signal_configuration']
        assert signal_config['file_path'] == 'TV_CONFIG_SIGNALS_1.0.0.xlsx'
        assert signal_config['date_format'] == '%Y%m%d %H%M%S'
        assert signal_config['time_adjustments']['entry_offset_seconds'] == 0
        assert signal_config['time_adjustments']['exit_offset_seconds'] == 0
        
        # Validate portfolio mappings from real config
        portfolio_mappings = tv_config['portfolio_mappings']
        assert portfolio_mappings['long'] == 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx'
        assert portfolio_mappings['short'] == 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx'
        assert portfolio_mappings['manual'] == 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx'
        
        # Validate execution settings from real config
        execution_settings = tv_config['execution_settings']
        assert execution_settings['intraday_squareoff'] is True
        assert execution_settings['exit_time'] == '15:30:00'
        assert execution_settings['tv_exit_applicable'] is True
        assert execution_settings['slippage_percent'] == 0.1
        assert execution_settings['use_db_exit_timing'] is False
        assert execution_settings['exit_search_interval'] == 5
        assert execution_settings['exit_price_source'] == 'SPOT'
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_convert_real_signals_to_yaml(self, real_config_files):
        """Test converting REAL signals file to YAML"""
        
        signals_path = real_config_files['signals']
        
        # Convert signals to YAML
        yaml_output = self.converter.convert_signals_to_yaml(str(signals_path))
        
        # Validate structure
        assert 'signals' in yaml_output, "Must have signals section"
        signals_list = yaml_output['signals']
        
        assert isinstance(signals_list, list), "Signals must be list"
        assert len(signals_list) == 4, "Real signals file has 4 signals"
        
        # Validate first signal (T001 Entry Long)
        first_signal = signals_list[0]
        assert first_signal['trade_no'] == 'T001'
        assert first_signal['type'] == 'Entry Long'
        assert first_signal['datetime'] == '2024-01-01 09:16:00'
        assert first_signal['contracts'] == 1
        
        # Validate second signal (T001 Exit Long)
        second_signal = signals_list[1]
        assert second_signal['trade_no'] == 'T001'
        assert second_signal['type'] == 'Exit Long'
        assert second_signal['datetime'] == '2024-01-01 12:00:00'
        assert second_signal['contracts'] == 1
        
        # Validate signal pairing in YAML
        trade_nos = set(signal['trade_no'] for signal in signals_list)
        assert 'T001' in trade_nos and 'T002' in trade_nos
        
        # Check signal types
        signal_types = [signal['type'] for signal in signals_list]
        assert 'Entry Long' in signal_types
        assert 'Exit Long' in signal_types  
        assert 'Entry Short' in signal_types
        assert 'Exit Short' in signal_types
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_convert_real_portfolio_to_yaml(self, real_config_files):
        """Test converting REAL portfolio configuration to YAML"""
        
        portfolio_path = real_config_files['portfolio_long']
        
        # Convert portfolio to YAML
        yaml_output = self.converter.convert_portfolio_to_yaml(str(portfolio_path))
        
        # Validate structure
        assert 'portfolio_configuration' in yaml_output
        portfolio_config = yaml_output['portfolio_configuration']
        
        # Validate portfolio settings from real config
        assert 'portfolio_settings' in portfolio_config
        portfolio_settings = portfolio_config['portfolio_settings']
        assert portfolio_settings['capital'] == 1000000
        assert portfolio_settings['max_risk'] == 5
        assert portfolio_settings['max_positions'] == 5
        assert portfolio_settings['risk_per_trade'] == 2
        assert portfolio_settings['use_kelly_criterion'] is False
        assert portfolio_settings['rebalance_frequency'] == 'DAILY'
        
        # Validate strategy settings from real config
        assert 'strategy_settings' in portfolio_config
        strategy_settings = portfolio_config['strategy_settings']
        assert isinstance(strategy_settings, list)
        assert len(strategy_settings) > 0
        
        first_strategy = strategy_settings[0]
        assert first_strategy['strategy_name'] == 'LONG_Strategy_1'
        assert first_strategy['strategy_file'] == 'TV_CONFIG_STRATEGY_LONG_1.0.0.xlsx'
        assert first_strategy['enabled'] is True
        assert first_strategy['priority'] == 1
        assert first_strategy['allocation_percent'] == 100
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_convert_real_tbs_strategy_to_yaml(self, real_config_files):
        """Test converting REAL TBS strategy configuration to YAML"""
        
        strategy_path = real_config_files['strategy']
        
        # Convert strategy to YAML
        yaml_output = self.converter.convert_tbs_strategy_to_yaml(str(strategy_path))
        
        # Validate structure
        assert 'tbs_strategy_configuration' in yaml_output
        strategy_config = yaml_output['tbs_strategy_configuration']
        
        # Validate general parameters from real config
        assert 'general_parameters' in strategy_config
        general_params = strategy_config['general_parameters']
        assert general_params['strategy_name'] == 'TV_Strategy_Sample'
        assert general_params['underlying'] == 'SPOT'
        assert general_params['index'] == 'NIFTY'
        assert general_params['weekdays'] == [1, 2, 3, 4, 5]  # Converted from '1,2,3,4,5'
        assert general_params['dte'] == 0
        assert general_params['strike_selection_time'] == '09:20:00'  # Converted from 92000
        assert general_params['start_time'] == '09:16:00'  # Converted from 91600
        assert general_params['end_time'] == '15:00:00'  # Converted from 150000
        
        # Validate leg parameters from real config
        assert 'leg_parameters' in strategy_config
        leg_params = strategy_config['leg_parameters']
        assert isinstance(leg_params, list)
        assert len(leg_params) > 0
        
        first_leg = leg_params[0]
        assert first_leg['leg_id'] == 'leg1'
        assert first_leg['instrument'] == 'call'
        assert first_leg['transaction'] == 'sell'
        assert first_leg['expiry'] == 'current'
        assert first_leg['strike_method'] == 'ATM'
        assert first_leg['strike_value'] == 0
        assert first_leg['sl_type'] == 'percentage'
        assert first_leg['sl_value'] == 100
        assert first_leg['lots'] == 1
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_complete_6_file_hierarchy_to_yaml(self, real_config_files):
        """Test converting complete 6-file hierarchy to unified YAML"""
        
        # Convert complete hierarchy
        yaml_output = self.converter.convert_complete_hierarchy_to_yaml(real_config_files)
        
        # Validate top-level structure
        assert 'tv_complete_configuration' in yaml_output
        complete_config = yaml_output['tv_complete_configuration']
        
        # Validate all 6 file components are present
        expected_components = [
            'tv_master', 'signals', 'portfolio_long', 
            'portfolio_short', 'portfolio_manual', 'tbs_strategy'
        ]
        
        for component in expected_components:
            assert component in complete_config, f"Missing component: {component}"
        
        # Validate file references are preserved
        tv_master = complete_config['tv_master']
        assert tv_master['signal_configuration']['file_path'] == 'TV_CONFIG_SIGNALS_1.0.0.xlsx'
        assert tv_master['portfolio_mappings']['long'] == 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx'
        
        # Validate metadata
        assert 'metadata' in complete_config
        metadata = complete_config['metadata']
        assert 'conversion_timestamp' in metadata
        assert 'source_files' in metadata
        assert len(metadata['source_files']) == 6
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_yaml_serialization_and_deserialization(self, real_config_files):
        """Test YAML serialization and deserialization roundtrip"""
        
        # Convert to YAML
        yaml_output = self.converter.convert_tv_master_to_yaml(str(real_config_files['tv_master']))
        
        # Serialize to YAML string
        yaml_string = yaml.dump(yaml_output, default_flow_style=False, allow_unicode=True)
        assert isinstance(yaml_string, str)
        assert len(yaml_string) > 100
        
        # Deserialize back to dict
        parsed_yaml = yaml.safe_load(yaml_string)
        assert isinstance(parsed_yaml, dict)
        
        # Validate roundtrip preserved data
        assert parsed_yaml['tv_configuration']['master_settings']['name'] == 'TV_Backtest_Sample'
        assert parsed_yaml['tv_configuration']['master_settings']['enabled'] is True
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(yaml_output, tmp, default_flow_style=False, allow_unicode=True)
            tmp_path = tmp.name
        
        try:
            # Read back from file
            with open(tmp_path, 'r') as f:
                file_parsed = yaml.safe_load(f)
            
            assert file_parsed == parsed_yaml, "File roundtrip must preserve data"
            
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_data_type_conversion_validation(self, real_config_files):
        """Test proper data type conversion in YAML output"""
        
        yaml_output = self.converter.convert_tv_master_to_yaml(str(real_config_files['tv_master']))
        tv_config = yaml_output['tv_configuration']
        
        # Test boolean conversions
        assert isinstance(tv_config['master_settings']['enabled'], bool)
        assert isinstance(tv_config['execution_settings']['intraday_squareoff'], bool)
        assert isinstance(tv_config['execution_settings']['tv_exit_applicable'], bool)
        assert isinstance(tv_config['rollover_settings']['do_rollover'], bool)
        
        # Test string conversions
        assert isinstance(tv_config['master_settings']['name'], str)
        assert isinstance(tv_config['signal_configuration']['file_path'], str)
        assert isinstance(tv_config['signal_configuration']['date_format'], str)
        
        # Test numeric conversions
        assert isinstance(tv_config['execution_settings']['slippage_percent'], (int, float))
        assert isinstance(tv_config['execution_settings']['exit_search_interval'], int)
        assert isinstance(tv_config['timing_settings']['entry_offset_seconds'], int)
        assert isinstance(tv_config['timing_settings']['exit_offset_seconds'], int)
        
        # Test time conversions
        assert isinstance(tv_config['execution_settings']['exit_time'], str)
        assert tv_config['execution_settings']['exit_time'] == '15:30:00'
        
        # Test date conversions
        assert isinstance(tv_config['master_settings']['date_range']['start'], str)
        assert isinstance(tv_config['master_settings']['date_range']['end'], str)
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_yaml_schema_validation(self, real_config_files):
        """Test YAML output against defined schema"""
        
        yaml_output = self.converter.convert_tv_master_to_yaml(str(real_config_files['tv_master']))
        
        # Validate against schema (basic structure validation)
        schema = self.converter.get_yaml_schema()
        
        # Test required fields exist
        tv_config = yaml_output['tv_configuration']
        
        # Master settings schema
        master_required = ['name', 'enabled', 'date_range']
        for field in master_required:
            assert field in tv_config['master_settings'], f"Missing master setting: {field}"
        
        # Signal configuration schema
        signal_required = ['file_path', 'date_format', 'time_adjustments']
        for field in signal_required:
            assert field in tv_config['signal_configuration'], f"Missing signal setting: {field}"
        
        # Portfolio mappings schema
        portfolio_required = ['long', 'short', 'manual']
        for field in portfolio_required:
            assert field in tv_config['portfolio_mappings'], f"Missing portfolio mapping: {field}"
        
        # Execution settings schema
        execution_required = ['intraday_squareoff', 'exit_time', 'slippage_percent']
        for field in execution_required:
            assert field in tv_config['execution_settings'], f"Missing execution setting: {field}"
    
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_file_path_preservation_in_yaml(self, real_config_files):
        """Test that file paths are correctly preserved in YAML conversion"""
        
        yaml_output = self.converter.convert_complete_hierarchy_to_yaml(real_config_files)
        complete_config = yaml_output['tv_complete_configuration']
        
        # Validate file path references are preserved
        tv_master = complete_config['tv_master']
        
        # Signal file reference
        signal_file = tv_master['signal_configuration']['file_path']
        assert signal_file == 'TV_CONFIG_SIGNALS_1.0.0.xlsx'
        
        # Portfolio file references
        portfolio_mappings = tv_master['portfolio_mappings']
        assert portfolio_mappings['long'] == 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx'
        assert portfolio_mappings['short'] == 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx'
        assert portfolio_mappings['manual'] == 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx'
        
        # Strategy file references in portfolios
        long_portfolio = complete_config['portfolio_long']
        long_strategy_ref = long_portfolio['strategy_settings'][0]['strategy_file']
        assert long_strategy_ref == 'TV_CONFIG_STRATEGY_LONG_1.0.0.xlsx'
        
        # Validate metadata includes source file paths
        metadata = complete_config['metadata']
        source_files = metadata['source_files']
        assert 'tv_master' in source_files
        assert 'signals' in source_files
        assert 'portfolio_long' in source_files
        assert 'portfolio_short' in source_files
        assert 'portfolio_manual' in source_files
        assert 'tbs_strategy' in source_files
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_yaml_output_human_readable(self, real_config_files):
        """Test that YAML output is human-readable and well-formatted"""
        
        yaml_output = self.converter.convert_tv_master_to_yaml(str(real_config_files['tv_master']))
        
        # Convert to YAML string
        yaml_string = yaml.dump(
            yaml_output, 
            default_flow_style=False, 
            allow_unicode=True,
            indent=2,
            sort_keys=False
        )
        
        # Basic readability checks
        lines = yaml_string.split('\n')
        assert len(lines) > 10, "YAML should have multiple lines"
        
        # Check indentation consistency
        indented_lines = [line for line in lines if line.startswith('  ')]
        assert len(indented_lines) > 5, "Should have proper indentation"
        
        # Check for readable field names (no underscores in output)
        readable_sections = [
            'tv_configuration:', 'master_settings:', 'signal_configuration:',
            'portfolio_mappings:', 'execution_settings:'
        ]
        
        for section in readable_sections:
            assert section in yaml_string, f"Missing readable section: {section}"
        
        # Ensure no complex nested structures on single lines
        for line in lines:
            if ':' in line:
                # Lines with colons should not be overly complex
                assert len(line.strip()) < 200, f"Line too complex: {line[:100]}..."


class TestTVExcelToYAMLConverterPerformance:
    """Performance tests for Excel to YAML conversion"""
    
    def setup_method(self):
        """Setup for each test"""
        self.converter = TVExcelToYAMLConverter()
    
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_conversion_performance_real_files(self, real_config_files):
        """Test conversion performance with REAL configuration files"""
        
        import time
        
        # Test TV master conversion performance
        start_time = time.time()
        yaml_output = self.converter.convert_tv_master_to_yaml(str(real_config_files['tv_master']))
        master_time = time.time() - start_time
        
        assert master_time < 1.0, f"TV master conversion too slow: {master_time:.3f}s"
        
        # Test signals conversion performance
        start_time = time.time()
        signals_yaml = self.converter.convert_signals_to_yaml(str(real_config_files['signals']))
        signals_time = time.time() - start_time
        
        assert signals_time < 0.5, f"Signals conversion too slow: {signals_time:.3f}s"
        
        # Test complete hierarchy conversion performance
        start_time = time.time()
        complete_yaml = self.converter.convert_complete_hierarchy_to_yaml(real_config_files)
        complete_time = time.time() - start_time
        
        assert complete_time < 2.0, f"Complete conversion too slow: {complete_time:.3f}s"
        
        print(f"Conversion performance:")
        print(f"  TV master: {master_time:.3f}s")
        print(f"  Signals: {signals_time:.3f}s")
        print(f"  Complete hierarchy: {complete_time:.3f}s")
        
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_yaml_serialization_performance(self, real_config_files):
        """Test YAML serialization performance"""
        
        import time
        
        # Convert to dict
        yaml_dict = self.converter.convert_complete_hierarchy_to_yaml(real_config_files)
        
        # Test serialization performance
        start_time = time.time()
        yaml_string = yaml.dump(yaml_dict, default_flow_style=False, allow_unicode=True)
        serialization_time = time.time() - start_time
        
        assert serialization_time < 0.5, f"YAML serialization too slow: {serialization_time:.3f}s"
        
        # Test deserialization performance
        start_time = time.time()
        parsed_dict = yaml.safe_load(yaml_string)
        deserialization_time = time.time() - start_time
        
        assert deserialization_time < 0.5, f"YAML deserialization too slow: {deserialization_time:.3f}s"
        
        print(f"Serialization performance:")
        print(f"  YAML dump: {serialization_time:.3f}s")
        print(f"  YAML load: {deserialization_time:.3f}s")