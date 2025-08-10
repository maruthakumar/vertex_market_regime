#!/usr/bin/env python3
"""
Unit Tests for TV Signal Processor - REAL DATA VALIDATION  
Tests signal processing logic with real configuration files and HeavyDB validation
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, date, time as datetime_time
from unittest.mock import Mock, patch
import time

from signal_processor import SignalProcessor
from parser import TVParser


class TestSignalProcessorRealData:
    """Test Signal Processor with REAL TV configuration and signals"""
    
    def setup_method(self):
        """Setup for each test"""
        self.processor = SignalProcessor()
        self.parser = TVParser()
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_process_real_signals_basic(self, real_config_files):
        """Test processing REAL signals from configuration files"""
        
        # Load real TV config and signals
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        
        # Process the real signals
        processed_signals = self.processor.process_signals(raw_signals, tv_config)
        
        # Validate processed output
        assert isinstance(processed_signals, list), "Processed signals must be list"
        assert len(processed_signals) > 0, "Must have processed signals"
        
        # Check each processed signal has required fields
        for signal in processed_signals:
            required_fields = [
                'trade_no', 'signal_type', 'datetime', 'lots', 
                'entry_datetime', 'exit_datetime', 'direction'
            ]
            for field in required_fields:
                assert field in signal, f"Signal missing required field: {field}"
        
        # Validate signal pairing worked correctly
        # Real signals file has T001 (Long) and T002 (Short)
        trade_nos = set(signal['trade_no'] for signal in processed_signals)
        assert 'T001' in trade_nos, "T001 trade must be processed"
        assert 'T002' in trade_nos, "T002 trade must be processed"
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_signal_pairing_with_real_data(self, real_config_files):
        """Test signal pairing logic with REAL signal data"""
        
        # Load real signals
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        
        # Test the pairing function directly
        paired_signals = self.processor._pair_signals(raw_signals)
        
        # Real file has 2 trades (T001 and T002)
        assert len(paired_signals) == 2, "Should have 2 paired trades from real data"
        
        # Check T001 (Long trade)
        t001_signal = next(s for s in paired_signals if s['trade_no'] == 'T001')
        assert t001_signal['direction'] == 'LONG', "T001 should be LONG direction"
        assert t001_signal['entry_datetime'] == datetime(2024, 1, 1, 9, 16, 0)
        assert t001_signal['exit_datetime'] == datetime(2024, 1, 1, 12, 0, 0)
        assert t001_signal['lots'] == 1
        
        # Check T002 (Short trade)
        t002_signal = next(s for s in paired_signals if s['trade_no'] == 'T002')
        assert t002_signal['direction'] == 'SHORT', "T002 should be SHORT direction"
        assert t002_signal['entry_datetime'] == datetime(2024, 1, 1, 13, 0, 0)
        assert t002_signal['exit_datetime'] == datetime(2024, 1, 1, 15, 0, 0)
        assert t002_signal['lots'] == 2
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_date_range_filtering_real_config(self, real_config_files):
        """Test date range filtering with REAL configuration dates"""
        
        # Load real TV config
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        
        # Real config has start: 01_01_2024, end: 31_12_2024
        assert tv_config['start_date'] == date(2024, 1, 1)
        assert tv_config['end_date'] == date(2024, 12, 31)
        
        # Create test signals with different dates
        test_signals = [
            {
                'trade_no': 'T_BEFORE',
                'signal_type': 'Entry Long',
                'datetime': datetime(2023, 12, 31, 10, 0, 0),  # Before start
                'lots': 1
            },
            {
                'trade_no': 'T_VALID',
                'signal_type': 'Entry Long',
                'datetime': datetime(2024, 6, 15, 10, 0, 0),  # Within range
                'lots': 1
            },
            {
                'trade_no': 'T_AFTER',
                'signal_type': 'Entry Long',
                'datetime': datetime(2025, 1, 1, 10, 0, 0),  # After end
                'lots': 1
            }
        ]
        
        # Test date filtering
        filtered_signals = self.processor._filter_by_date_range(
            test_signals,
            tv_config['start_date'],
            tv_config['end_date']
        )
        
        # Only the valid signal should remain
        assert len(filtered_signals) == 1, "Only signals within date range should remain"
        assert filtered_signals[0]['trade_no'] == 'T_VALID'
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_time_adjustments_real_config(self, real_config_files):
        """Test time adjustments with REAL configuration values"""
        
        # Load real TV config
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        
        # Real config has time adjustments set to 0
        assert tv_config['increase_entry_signal_time_by'] == 0
        assert tv_config['increase_exit_signal_time_by'] == 0
        
        # Process signals (no time adjustment expected)
        processed_signals = self.processor.process_signals(raw_signals, tv_config)
        
        # Find T001 signals
        t001_signal = next(s for s in processed_signals if s['trade_no'] == 'T001')
        
        # Times should match original (no adjustment)
        assert t001_signal['entry_datetime'] == datetime(2024, 1, 1, 9, 16, 0)
        assert t001_signal['exit_datetime'] == datetime(2024, 1, 1, 12, 0, 0)
        
        # Test with time adjustments
        tv_config_adjusted = tv_config.copy()
        tv_config_adjusted['increase_entry_signal_time_by'] = 300  # 5 minutes
        tv_config_adjusted['increase_exit_signal_time_by'] = 600   # 10 minutes
        
        adjusted_signals = self.processor.process_signals(raw_signals, tv_config_adjusted)
        t001_adjusted = next(s for s in adjusted_signals if s['trade_no'] == 'T001')
        
        # Entry should be 5 minutes later
        expected_entry = datetime(2024, 1, 1, 9, 21, 0)  # 09:16 + 5 min
        assert t001_adjusted['entry_datetime'] == expected_entry
        
        # Exit should be 10 minutes later
        expected_exit = datetime(2024, 1, 1, 12, 10, 0)  # 12:00 + 10 min
        assert t001_adjusted['exit_datetime'] == expected_exit
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_intraday_exit_override_real_config(self, real_config_files):
        """Test intraday exit time override with REAL configuration"""
        
        # Load real TV config
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        
        # Real config has IntradaySqOffApplicable=YES, IntradayExitTime=153000 (15:30)
        assert tv_config['intraday_sqoff_applicable'] is True
        assert tv_config['intraday_exit_time'] == datetime_time(15, 30, 0)
        
        # Modify config to disable TV exit (force intraday exit)
        tv_config_modified = tv_config.copy()
        tv_config_modified['tv_exit_applicable'] = False
        
        processed_signals = self.processor.process_signals(raw_signals, tv_config_modified)
        
        # All signals should exit at 15:30 on their entry date
        for signal in processed_signals:
            expected_exit_time = datetime_time(15, 30, 0)
            assert signal['exit_datetime'].time() == expected_exit_time, \
                f"Signal {signal['trade_no']} should exit at 15:30"
            # Exit date should be same as entry date
            assert signal['exit_datetime'].date() == signal['entry_datetime'].date(), \
                f"Signal {signal['trade_no']} should exit on same day"
        
    @pytest.mark.unit  
    @pytest.mark.heavydb
    def test_manual_trade_generation_real_config(self, real_config_files):
        """Test manual trade generation with REAL configuration settings"""
        
        # Load real TV config
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        
        # Real config has manual trades disabled (ManualTradeEntryTime=0, ManualTradeLots=0)
        assert tv_config['manual_trade_entry_time'] == 0
        assert tv_config['manual_trade_lots'] == 0
        
        # Enable manual trades for testing
        tv_config_manual = tv_config.copy()
        tv_config_manual['manual_trade_entry_time'] = datetime_time(9, 30, 0)
        tv_config_manual['manual_trade_lots'] = 5
        
        # Generate manual signals
        manual_signals = self.processor._generate_manual_signals(tv_config_manual)
        
        # Should generate signals for the date range (2024-01-01 to 2024-12-31)
        # But filtered to trading days within the range
        assert len(manual_signals) > 0, "Should generate manual signals"
        
        # Check first manual signal
        first_manual = manual_signals[0]
        assert first_manual['signal_type'] == 'Manual Entry'
        assert first_manual['lots'] == 5
        assert first_manual['datetime'].time() == datetime_time(9, 30, 0)
        
        # Should have corresponding exit signal
        exit_signals = [s for s in manual_signals if s['signal_type'] == 'Manual Exit']
        assert len(exit_signals) > 0, "Should have manual exit signals"
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_rollover_handling_real_config(self, real_config_files):
        """Test rollover handling with REAL configuration"""
        
        # Load real TV config
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        
        # Real config has DoRollover=NO
        assert tv_config['do_rollover'] is False
        
        # Test with rollover enabled
        tv_config_rollover = tv_config.copy()
        tv_config_rollover['do_rollover'] = True
        tv_config_rollover['rollover_time'] = datetime_time(15, 20, 0)
        
        # Create test signals spanning multiple days
        test_signals = [
            {
                'trade_no': 'T_ROLLOVER',
                'signal_type': 'Entry Long',
                'datetime': datetime(2024, 1, 25, 9, 16, 0),  # Thursday
                'lots': 1
            },
            {
                'trade_no': 'T_ROLLOVER',
                'signal_type': 'Exit Long',
                'datetime': datetime(2024, 1, 29, 15, 0, 0),  # Monday next week
                'lots': 1
            }
        ]
        
        # Process with rollover
        processed_signals = self.processor.process_signals(test_signals, tv_config_rollover)
        
        # Should detect rollover scenario and handle appropriately
        rollover_signal = processed_signals[0]
        assert 'is_rollover_trade' in rollover_signal
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_signal_direction_mapping_real_data(self, real_config_files):
        """Test signal direction mapping with REAL signal types"""
        
        # Load real signals
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        
        # Test direction mapping
        for signal in raw_signals:
            direction = self.processor._get_signal_direction(signal['signal_type'])
            
            if signal['signal_type'] in ['Entry Long', 'Exit Long']:
                assert direction == 'LONG', f"Long signals should map to LONG direction"
            elif signal['signal_type'] in ['Entry Short', 'Exit Short']:
                assert direction == 'SHORT', f"Short signals should map to SHORT direction"
            elif signal['signal_type'] in ['Manual Entry', 'Manual Exit']:
                assert direction == 'MANUAL', f"Manual signals should map to MANUAL direction"
        
    @pytest.mark.unit
    @pytest.mark.heavydb
    def test_portfolio_file_resolution_real_config(self, real_config_files):
        """Test portfolio file resolution with REAL configuration paths"""
        
        # Load real TV config
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        
        # Test portfolio path resolution
        long_portfolio = self.processor._resolve_portfolio_path(
            'LONG', tv_config, str(real_config_files['tv_master'])
        )
        short_portfolio = self.processor._resolve_portfolio_path(
            'SHORT', tv_config, str(real_config_files['tv_master'])
        )
        manual_portfolio = self.processor._resolve_portfolio_path(
            'MANUAL', tv_config, str(real_config_files['tv_master'])
        )
        
        # Validate resolved paths exist
        from pathlib import Path
        assert Path(long_portfolio).exists(), f"Long portfolio should exist: {long_portfolio}"
        assert Path(short_portfolio).exists(), f"Short portfolio should exist: {short_portfolio}"
        assert Path(manual_portfolio).exists(), f"Manual portfolio should exist: {manual_portfolio}"
        
        # Validate filenames match config
        assert long_portfolio.endswith(tv_config['long_portfolio_file_path'])
        assert short_portfolio.endswith(tv_config['short_portfolio_file_path'])
        assert manual_portfolio.endswith(tv_config['manual_portfolio_file_path'])


class TestSignalProcessorRealDataPerformance:
    """Performance tests for Signal Processor with REAL data"""
    
    def setup_method(self):
        """Setup for each test"""
        self.processor = SignalProcessor()
        self.parser = TVParser()
    
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_signal_processing_performance_real_data(self, real_config_files):
        """Test signal processing performance with REAL configuration"""
        
        # Load real config and signals
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        raw_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        
        # Measure processing time
        start_time = time.time()
        processed_signals = self.processor.process_signals(raw_signals, tv_config)
        processing_time = time.time() - start_time
        
        # Performance requirements
        assert processing_time < 0.1, f"Signal processing took too long: {processing_time:.3f}s"
        assert len(processed_signals) > 0, "Must produce processed signals"
        
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_large_signal_processing_simulation(self, real_config_files):
        """Test processing performance with larger signal sets (simulated from real data)"""
        
        # Load real config and signals as template
        tv_config = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        real_signals = self.parser.parse_signals(
            str(real_config_files['signals']), 
            tv_config['signal_date_format']
        )
        
        # Create larger signal set by replicating real signals with different trade numbers
        large_signal_set = []
        for i in range(50):  # 50 trades = 100 signals
            for signal in real_signals:
                new_signal = signal.copy()
                new_signal['trade_no'] = f"{signal['trade_no']}_BATCH_{i}"
                # Adjust datetime to spread across the year
                days_offset = i * 7  # Weekly spacing
                new_signal['datetime'] = signal['datetime'] + timedelta(days=days_offset)
                large_signal_set.append(new_signal)
        
        # Measure processing time for larger set
        start_time = time.time()
        processed_signals = self.processor.process_signals(large_signal_set, tv_config)
        processing_time = time.time() - start_time
        
        # Performance requirements for larger dataset
        assert processing_time < 1.0, f"Large signal processing took too long: {processing_time:.3f}s"
        assert len(processed_signals) == 100, "Should process all 100 trades (50 pairs)"
        
        # Validate all signals processed correctly
        trade_nos = set(signal['trade_no'] for signal in processed_signals)
        assert len(trade_nos) == 100, "Should have 100 unique trade numbers"