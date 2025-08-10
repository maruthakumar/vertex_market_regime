#!/usr/bin/env python3
"""
TV Parser - Handles parsing of TradingView signal Excel input files
Based on actual column structure from /srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/tv/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, time
import logging
import os

logger = logging.getLogger(__name__)


class TVParser:
    """Parser for TV (TradingView) strategy Excel files"""
    
    def __init__(self):
        # Define expected columns based on actual files
        self.tv_setting_columns = [
            'StartDate', 'EndDate', 'SignalDateFormat', 'Enabled', 'TvExitApplicable',
            'ManualTradeEntryTime', 'ManualTradeLots', 'IncreaseEntrySignalTimeBy',
            'IncreaseExitSignalTimeBy', 'IntradaySqOffApplicable', 'FirstTradeEntryTime',
            'IntradayExitTime', 'ExpiryDayExitTime', 'DoRollover', 'RolloverTime',
            'Name', 'SignalFilePath', 'LongPortfolioFilePath', 'ShortPortfolioFilePath',
            'ManualPortfolioFilePath', 'UseDbExitTiming', 'ExitSearchInterval',
            'ExitPriceSource', 'SlippagePercent'
        ]
        
        # Signal columns from TradingView
        self.signal_columns = ['Trade #', 'Type', 'Date/Time', 'Contracts']
        
    def parse_tv_settings(self, excel_path: str) -> Dict[str, Any]:
        """
        Parse the TV settings Excel file
        
        Args:
            excel_path: Path to input_tv.xlsx
            
        Returns:
            Dictionary containing TV settings
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"TV settings file not found: {excel_path}")
        
        logger.info(f"Parsing TV settings: {excel_path}")
        
        try:
            # Read Setting sheet with explicit engine
            settings_df = pd.read_excel(excel_path, sheet_name='Setting', engine='openpyxl')
            logger.info(f"Loaded TV settings: {len(settings_df)} rows")
            
            # Filter enabled settings
            enabled_df = settings_df[settings_df['Enabled'].apply(self._parse_bool)]
            
            if enabled_df.empty:
                raise ValueError("No enabled TV settings found")
            
            # Parse all enabled settings
            tv_settings = []
            for idx, row in enabled_df.iterrows():
                setting = self._parse_tv_setting_row(row)
                tv_settings.append(setting)
            
            return {
                'settings': tv_settings,
                'source_file': excel_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing TV settings: {e}")
            raise
    
    def parse_signals(self, signal_path: str, date_format: str) -> List[Dict[str, Any]]:
        """
        Parse signal file (list of trades)
        
        Args:
            signal_path: Path to signal Excel file
            date_format: Format string for parsing datetime
            
        Returns:
            List of signal dictionaries
        """
        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
        
        logger.info(f"Parsing signals: {signal_path}")
        
        try:
            # Try to find the correct sheet
            excel_file = pd.ExcelFile(signal_path)
            sheet_name = None
            
            # Look for common sheet names
            for name in ['List of trades', 'Signals', 'Sheet1']:
                if name in excel_file.sheet_names:
                    sheet_name = name
                    break
            
            if not sheet_name and excel_file.sheet_names:
                sheet_name = excel_file.sheet_names[0]
            
            # Read signals with explicit engine
            signals_df = pd.read_excel(signal_path, sheet_name=sheet_name, engine='openpyxl')
            logger.info(f"Loaded {len(signals_df)} signals from sheet '{sheet_name}'")
            
            # Parse signals
            signals = []
            for idx, row in signals_df.iterrows():
                signal = self._parse_signal_row(row, date_format)
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error parsing signals: {e}")
            raise
    
    def _parse_tv_setting_row(self, row: pd.Series) -> Dict[str, Any]:
        """Parse a TV setting row"""
        setting = {
            'name': str(row.get('Name', 'TV_Backtest')),
            'enabled': True,  # Already filtered
            'start_date': self._parse_date(row.get('StartDate')),
            'end_date': self._parse_date(row.get('EndDate')),
            'signal_date_format': str(row.get('SignalDateFormat', '%Y%m%d %H%M%S')),
            'tv_exit_applicable': self._parse_bool(row.get('TvExitApplicable', 'YES')),
            'intraday_sqoff_applicable': self._parse_bool(row.get('IntradaySqOffApplicable', 'NO')),
            'do_rollover': self._parse_bool(row.get('DoRollover', 'NO')),
            'use_db_exit_timing': self._parse_bool(row.get('UseDbExitTiming', 'NO')),
        }
        
        # Parse time fields
        time_fields = {
            'ManualTradeEntryTime': 'manual_trade_entry_time',
            'FirstTradeEntryTime': 'first_trade_entry_time',
            'IntradayExitTime': 'intraday_exit_time',
            'ExpiryDayExitTime': 'expiry_day_exit_time',
            'RolloverTime': 'rollover_time'
        }
        
        for excel_col, field_name in time_fields.items():
            if excel_col in row and pd.notna(row[excel_col]):
                setting[field_name] = self._parse_time(row[excel_col])
        
        # Parse numeric fields
        numeric_fields = {
            'ManualTradeLots': ('manual_trade_lots', int),
            'IncreaseEntrySignalTimeBy': ('increase_entry_signal_time_by', int),
            'IncreaseExitSignalTimeBy': ('increase_exit_signal_time_by', int),
            'ExitSearchInterval': ('exit_search_interval', int),
            'SlippagePercent': ('slippage_percent', float)
        }
        
        for excel_col, (field_name, converter) in numeric_fields.items():
            if excel_col in row and pd.notna(row[excel_col]):
                setting[field_name] = converter(row[excel_col])
        
        # Parse portfolio file paths and signal file path
        file_path_fields = {
            'SignalFilePath': 'signal_file_path',
            'LongPortfolioFilePath': 'long_portfolio_file_path',
            'ShortPortfolioFilePath': 'short_portfolio_file_path',
            'ManualPortfolioFilePath': 'manual_portfolio_file_path'
        }
        
        for excel_col, field_name in file_path_fields.items():
            if excel_col in row and pd.notna(row[excel_col]):
                value = str(row[excel_col])
                # Clean Windows paths - extract just filename
                if '\\' in value:
                    # Handle Windows paths on Linux
                    value = value.split('\\')[-1]
                setting[field_name] = value
        
        # Parse exit price source
        if 'ExitPriceSource' in row and pd.notna(row['ExitPriceSource']):
            setting['exit_price_source'] = str(row['ExitPriceSource']).upper()
        
        return setting
    
    def _parse_signal_row(self, row: pd.Series, date_format: str) -> Optional[Dict[str, Any]]:
        """Parse a signal row"""
        # Check required fields
        if pd.isna(row.get('Trade #')) or pd.isna(row.get('Type')):
            return None
        
        signal = {
            'trade_no': str(row['Trade #']),
            'signal_type': row.get('Type', ''),  # Keep original format for consistency
            'lots': int(row.get('Contracts', 1)) if pd.notna(row.get('Contracts')) else 1,
        }
        
        # Parse datetime
        if 'Date/Time' in row and pd.notna(row['Date/Time']):
            signal['datetime'] = self._parse_signal_datetime(row['Date/Time'], date_format)
        else:
            logger.warning(f"Signal {signal['trade_no']} missing datetime")
            return None
        
        # Add optional fields
        if 'Signal' in row and pd.notna(row['Signal']):
            signal['signal_name'] = str(row['Signal'])
        
        if 'Price INR' in row and pd.notna(row['Price INR']):
            # Handle comma-separated numbers
            price_str = str(row['Price INR']).replace(',', '')
            try:
                signal['price'] = float(price_str)
            except ValueError:
                logger.warning(f"Could not parse price: {row['Price INR']}")
        
        return signal
    
    def _normalize_signal_type(self, signal_type: str) -> str:
        """Normalize signal type to standard format"""
        signal_type = str(signal_type).strip().upper()
        
        # Map variations to standard types
        mapping = {
            'ENTRY LONG': 'ENTRY_LONG',
            'EXIT LONG': 'EXIT_LONG',
            'ENTRY SHORT': 'ENTRY_SHORT',
            'EXIT SHORT': 'EXIT_SHORT',
            'MANUAL ENTRY': 'MANUAL_ENTRY',
            'MANUAL EXIT': 'MANUAL_EXIT',
            'BUY': 'ENTRY_LONG',
            'SELL': 'EXIT_LONG',
            'SHORT': 'ENTRY_SHORT',
            'COVER': 'EXIT_SHORT'
        }
        
        return mapping.get(signal_type, signal_type)
    
    def _resolve_file_path(self, file_path: str, base_file_path: str) -> str:
        """Resolve file path relative to base file"""
        if not file_path:
            return ""
        
        # If already absolute, return as is
        if os.path.isabs(file_path):
            return file_path
        
        # Resolve relative to base file directory
        base_dir = os.path.dirname(base_file_path)
        resolved_path = os.path.join(base_dir, file_path)
        return os.path.abspath(resolved_path)
    
    def _sheet_exists(self, excel_path: str, sheet_name: str) -> bool:
        """Check if sheet exists in Excel file"""
        try:
            excel_file = pd.ExcelFile(excel_path)
            return sheet_name in excel_file.sheet_names
        except Exception:
            return False
    
    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse date from various formats"""
        if pd.isna(value):
            return None
        
        if isinstance(value, (datetime, date)):
            return value.date() if isinstance(value, datetime) else value
        
        # Try parsing string formats
        date_str = str(value).strip()
        
        # Try DD_MM_YYYY format
        if '_' in date_str:
            try:
                parts = date_str.split('_')
                if len(parts) == 3:
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    return date(year, month, day)
            except:
                pass
        
        # Try other formats
        for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d']:
            try:
                return datetime.strptime(date_str, fmt).date()
            except:
                continue
        
        logger.warning(f"Could not parse date: {value}")
        return None
    
    def _parse_time(self, value: Any) -> Optional[time]:
        """Parse time from various formats"""
        if pd.isna(value):
            return None
        
        if isinstance(value, time):
            return value
        
        if isinstance(value, datetime):
            return value.time()
        
        # Handle zero values
        if value == 0 or value == '0':
            return time(0, 0, 0)
        
        # Convert to string
        time_str = str(value).strip()
        
        # Handle HHMMSS format (e.g., 92000 for 09:20:00)
        if time_str.isdigit() and len(time_str) in [5, 6]:
            # Pad with leading zero if needed
            time_str = time_str.zfill(6)
            try:
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    return time(hour, minute, second)
            except:
                pass
        
        # Try parsing HH:MM:SS format
        for fmt in ['%H:%M:%S', '%H:%M']:
            try:
                return datetime.strptime(time_str, fmt).time()
            except:
                continue
        
        logger.warning(f"Could not parse time: {value}")
        return None
    
    def _parse_signal_datetime(self, value: Any, date_format: str) -> Optional[datetime]:
        """Parse signal datetime using specified format"""
        if pd.isna(value):
            return None
        
        if isinstance(value, datetime):
            return value
        
        # Convert to string
        dt_str = str(value).strip()
        
        # Try the specified format first
        try:
            return datetime.strptime(dt_str, date_format)
        except:
            pass
        
        # Try common formats
        for fmt in ['%Y%m%d %H%M%S', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S']:
            try:
                return datetime.strptime(dt_str, fmt)
            except:
                continue
        
        logger.warning(f"Could not parse datetime '{value}' with format '{date_format}'")
        return None
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean from various formats"""
        if pd.isna(value):
            return False
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1', 'Y', 'T']
        
        return bool(value)