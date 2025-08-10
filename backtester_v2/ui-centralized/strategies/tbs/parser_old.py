#!/usr/bin/env python3
"""
TBS Parser - Handles parsing of TBS Excel input files
Updated to match actual column structure from /srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/tbs/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, time
import logging
import os

logger = logging.getLogger(__name__)


class TBSParser:
    """Parser for TBS strategy Excel files"""
    
    def __init__(self):
        self.required_portfolio_columns = [
            'StartDate', 'EndDate', 'Capital', 'PortfolioName'
        ]
        self.required_strategy_columns = [
            'StrategyName', 'Enabled'
        ]
        
    def parse_portfolio_excel(self, excel_path: str) -> Dict[str, Any]:
        """
        Parse the portfolio Excel file containing PortfolioSetting and StrategySetting sheets
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dictionary containing parsed portfolio and strategy data
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        logger.info(f"Parsing portfolio Excel: {excel_path}")
        
        try:
            # Read PortfolioSetting sheet
            portfolio_df = pd.read_excel(excel_path, sheet_name='PortfolioSetting')
            logger.info(f"Loaded PortfolioSetting: {len(portfolio_df)} rows")
            
            # Read StrategySetting sheet
            strategy_df = pd.read_excel(excel_path, sheet_name='StrategySetting')
            logger.info(f"Loaded StrategySetting: {len(strategy_df)} rows")
            
            # Parse portfolio settings
            portfolio_data = self._parse_portfolio_settings(portfolio_df)
            
            # Parse strategy settings
            strategies = self._parse_strategy_settings(strategy_df)
            
            return {
                'portfolio': portfolio_data,
                'strategies': strategies,
                'source_file': excel_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}")
            raise
    
    def _parse_portfolio_settings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse portfolio settings from DataFrame"""
        if df.empty:
            raise ValueError("PortfolioSetting sheet is empty")
        
        # Get first row
        row = df.iloc[0]
        
        # Parse dates
        start_date = self._parse_date(row.get('StartDate'))
        end_date = self._parse_date(row.get('EndDate'))
        
        # Parse capital
        capital = float(row.get('Capital', 1000000))
        
        # Parse other settings
        portfolio_data = {
            'portfolio_name': str(row.get('PortfolioName', 'Portfolio')),
            'start_date': start_date,
            'end_date': end_date,
            'capital': capital,
            'is_tick_bt': self._parse_bool(row.get('IsTickBT', 'no')),
            'enabled': self._parse_bool(row.get('Enabled', 'YES')),
            'index': str(row.get('Index', 'NIFTY')).upper(),
            'lot_size': int(row.get('LotSize', 50)),
            'margin': float(row.get('Margin', 0.15)),
            'sl_percent': float(row.get('SLPercent', 0)) if pd.notna(row.get('SLPercent')) else None,
            'target_percent': float(row.get('TargetPercent', 0)) if pd.notna(row.get('TargetPercent')) else None,
            'daily_max_profit': float(row.get('DailyMaxProfit', 0)) if pd.notna(row.get('DailyMaxProfit')) else None,
            'daily_max_loss': float(row.get('DailyMaxLoss', 0)) if pd.notna(row.get('DailyMaxLoss')) else None,
        }
        
        # Additional settings
        for col in df.columns:
            if col not in portfolio_data and pd.notna(row[col]):
                portfolio_data[col.lower()] = row[col]
        
        return portfolio_data
    
    def _parse_strategy_settings(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse strategy settings from DataFrame"""
        strategies = []
        
        for idx, row in df.iterrows():
            # Skip if not enabled
            if not self._parse_bool(row.get('Enabled', 'NO')):
                continue
            
            strategy = {
                'strategy_name': str(row.get('StrategyName', f'Strategy_{idx}')),
                'enabled': True,
                'strategy_index': idx,
                'legs': []
            }
            
            # Parse leg information
            leg_data = self._parse_leg_info(row)
            if leg_data:
                strategy['legs'].append(leg_data)
            
            # Add all other columns
            for col in df.columns:
                if col not in ['StrategyName', 'Enabled'] and pd.notna(row[col]):
                    strategy[self._normalize_column_name(col)] = row[col]
            
            strategies.append(strategy)
        
        return strategies
    
    def _parse_leg_info(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Parse leg information from a strategy row"""
        leg = {}
        
        # Common leg fields
        leg_fields = {
            'LegNo': 'leg_no',
            'Quantity': 'quantity',
            'OptionType': 'option_type',
            'StrikeSelection': 'strike_selection',
            'StrikeValue': 'strike_value',
            'ExpiryRule': 'expiry_rule',
            'ExpiryValue': 'expiry_value',
            'EntryTime': 'entry_time',
            'ExitTime': 'exit_time',
            'TransactionType': 'transaction_type',
            'SLPercent': 'sl_percent',
            'TargetPercent': 'target_percent'
        }
        
        has_leg_data = False
        for excel_col, field_name in leg_fields.items():
            if excel_col in row and pd.notna(row[excel_col]):
                has_leg_data = True
                if excel_col in ['EntryTime', 'ExitTime']:
                    leg[field_name] = self._parse_time(row[excel_col])
                elif excel_col in ['Quantity', 'StrikeValue', 'ExpiryValue']:
                    leg[field_name] = int(row[excel_col])
                elif excel_col in ['SLPercent', 'TargetPercent']:
                    leg[field_name] = float(row[excel_col])
                else:
                    leg[field_name] = str(row[excel_col]).upper()
        
        return leg if has_leg_data else None
    
    def parse_multi_leg_excel(self, excel_path: str) -> Dict[str, Any]:
        """
        Parse multi-leg Excel file (INPUT TBS MULTI LEGS.xlsx format)
        
        Args:
            excel_path: Path to multi-leg Excel file
            
        Returns:
            Dictionary containing general parameters and leg parameters
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Multi-leg Excel file not found: {excel_path}")
        
        logger.info(f"Parsing multi-leg Excel: {excel_path}")
        
        try:
            # Read GeneralParameter sheet
            general_df = pd.read_excel(excel_path, sheet_name='GeneralParameter')
            logger.info(f"Loaded GeneralParameter: {len(general_df)} rows")
            
            # Read LegParameter sheet
            leg_df = pd.read_excel(excel_path, sheet_name='LegParameter')
            logger.info(f"Loaded LegParameter: {len(leg_df)} rows")
            
            # Parse sheets
            general_params = self._parse_general_parameters(general_df)
            leg_params = self._parse_leg_parameters(leg_df)
            
            return {
                'general_parameters': general_params,
                'leg_parameters': leg_params,
                'source_file': excel_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing multi-leg Excel: {e}")
            raise
    
    def _parse_general_parameters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse general parameters from DataFrame"""
        params = []
        
        for idx, row in df.iterrows():
            param = {
                'strategy_name': str(row.get('StrategyName', f'Strategy_{idx}')),
                'enabled': self._parse_bool(row.get('Enabled', 'YES')),
                'index': str(row.get('Index', 'NIFTY')).upper(),
                'no_of_legs': int(row.get('NoOfLegs', 1)),
                'lot_multiplier': int(row.get('LotMultiplier', 1)),
                'position_type': str(row.get('PositionType', 'INTRADAY')).upper()
            }
            
            # Add all other columns
            for col in df.columns:
                if col not in param and pd.notna(row[col]):
                    param[self._normalize_column_name(col)] = row[col]
            
            params.append(param)
        
        return params
    
    def _parse_leg_parameters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse leg parameters from DataFrame"""
        legs = []
        
        for idx, row in df.iterrows():
            leg = {
                'strategy_name': str(row.get('StrategyName', '')),
                'leg_no': int(row.get('LegNo', 1)),
                'quantity': int(row.get('Quantity', 1)),
                'option_type': str(row.get('OptionType', 'CE')).upper(),
                'strike_selection': str(row.get('StrikeSelection', 'ATM')).upper(),
                'strike_value': int(row.get('StrikeValue', 0)),
                'expiry_rule': str(row.get('ExpiryRule', 'CW')).upper(),
                'expiry_value': int(row.get('ExpiryValue', 0)),
                'entry_time': self._parse_time(row.get('EntryTime', '09:20:00')),
                'exit_time': self._parse_time(row.get('ExitTime', '15:15:00')),
                'transaction_type': str(row.get('TransactionType', 'SELL')).upper(),
                'sl_percent': float(row.get('SLPercent', 0)) if pd.notna(row.get('SLPercent')) else None,
                'target_percent': float(row.get('TargetPercent', 0)) if pd.notna(row.get('TargetPercent')) else None,
            }
            
            # Add all other columns
            for col in df.columns:
                if col not in leg and pd.notna(row[col]):
                    leg[self._normalize_column_name(col)] = row[col]
            
            legs.append(leg)
        
        return legs
    
    # Utility methods
    def _parse_date(self, value: Any) -> date:
        """Parse date from various formats"""
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            # Try different date formats
            formats = ['%d_%m_%Y', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d%m%Y']
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {value}")
        raise ValueError(f"Invalid date type: {type(value)}")
    
    def _parse_time(self, value: Any) -> time:
        """Parse time from various formats"""
        if isinstance(value, time):
            return value
        if isinstance(value, datetime):
            return value.time()
        if isinstance(value, str):
            # Try different time formats
            formats = ['%H:%M:%S', '%H:%M', '%H%M%S', '%H%M']
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt).time()
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse time: {value}")
        raise ValueError(f"Invalid time type: {type(value)}")
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean from various formats"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1', 'Y', 'T']
        return bool(value)
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name to snake_case"""
        import re
        # Convert CamelCase to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()