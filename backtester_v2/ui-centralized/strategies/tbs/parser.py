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
    """Parser for TBS strategy Excel files - matches actual input sheet structure"""
    
    def __init__(self):
        # Define expected columns based on actual files
        self.required_portfolio_columns = [
            'StartDate', 'EndDate', 'Enabled', 'PortfolioName'
        ]
        self.required_strategy_columns = [
            'Enabled', 'PortfolioName', 'StrategyType', 'StrategyExcelFilePath'
        ]
        
    def parse_portfolio_excel(self, excel_path: str) -> Dict[str, Any]:
        """
        Parse the portfolio Excel file containing PortfolioSetting and StrategySetting sheets
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
            
            # Parse portfolio settings - only enabled ones
            portfolio_data = self._parse_portfolio_settings(portfolio_df)
            
            # Parse strategy settings - only enabled ones
            strategies = self._parse_strategy_settings(strategy_df)
            
            return {
                'portfolio': portfolio_data,
                'strategies': strategies,
                'source_file': excel_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}")
            raise
    
    def parse_multi_leg_excel(self, excel_path: str) -> Dict[str, Any]:
        """
        Parse multi-leg Excel file (TBS strategy file)
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
            strategies = self._parse_general_parameters(general_df)
            legs = self._parse_leg_parameters(leg_df)
            
            # Attach legs to strategies
            for strategy in strategies:
                strategy_name = strategy['strategy_name']
                strategy['legs'] = [leg for leg in legs if leg['strategy_name'] == strategy_name]
            
            return {
                'strategies': strategies,
                'source_file': excel_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing multi-leg Excel: {e}")
            raise
    
    def _parse_portfolio_settings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse portfolio settings from DataFrame"""
        # Filter enabled portfolios
        enabled_df = df[df['Enabled'].apply(self._parse_bool)]
        
        if enabled_df.empty:
            raise ValueError("No enabled portfolios found in PortfolioSetting")
        
        # Get first enabled portfolio
        row = enabled_df.iloc[0]
        
        portfolio_data = {
            'portfolio_name': str(row.get('PortfolioName', 'Portfolio')).upper(),
            'start_date': self._parse_date(row.get('StartDate')),
            'end_date': self._parse_date(row.get('EndDate')),
            'is_tick_bt': self._parse_bool(row.get('IsTickBT', 'no')),
            'enabled': True,
            
            # Optional fields with defaults
            'capital': float(row.get('Capital', 1000000)) if 'Capital' in df.columns else 1000000.0,
            'index': str(row.get('Index', 'NIFTY')).upper() if 'Index' in df.columns else 'NIFTY',
            'lot_size': int(row.get('LotSize', 50)) if 'LotSize' in df.columns else 50,
            'margin': float(row.get('Margin', 0.15)) if 'Margin' in df.columns else 0.15,
        }
        
        # Add all other columns as extra params
        for col in df.columns:
            col_lower = self._normalize_column_name(col)
            if col_lower not in portfolio_data and pd.notna(row[col]):
                portfolio_data[col_lower] = row[col]
        
        return portfolio_data
    
    def _parse_strategy_settings(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse strategy settings from DataFrame"""
        strategies = []
        
        for idx, row in df.iterrows():
            # Skip if not enabled
            if not self._parse_bool(row.get('Enabled', 'NO')):
                continue
            
            # Skip if not TBS type
            strategy_type = str(row.get('StrategyType', '')).upper()
            if strategy_type != 'TBS':
                continue
            
            strategy = {
                'strategy_name': f'Strategy_{idx}',  # Will be replaced when loading actual strategy
                'enabled': True,
                'strategy_index': idx,
                'portfolio_name': str(row.get('PortfolioName', '')).upper(),
                'strategy_type': strategy_type,
                'strategy_excel_file_path': str(row.get('StrategyExcelFilePath', '')),
            }
            
            strategies.append(strategy)
        
        return strategies
    
    def _parse_general_parameters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse general parameters from DataFrame"""
        strategies = []
        
        for idx, row in df.iterrows():
            strategy = {
                'strategy_name': str(row.get('StrategyName', f'Strategy_{idx}')),
                'index': str(row.get('Index', 'NIFTY')).upper(),
                'underlying': str(row.get('Underlying', 'SPOT')).upper(),
                'enabled': True,
                'legs': []  # Will be filled later
            }
            
            # Parse times
            time_fields = ['StrikeSelectionTime', 'StartTime', 'LastEntryTime', 'EndTime',
                          'PnLCalTime', 'SqOff1Time', 'SqOff2Time']
            for field in time_fields:
                if field in row and pd.notna(row[field]):
                    strategy[self._normalize_column_name(field)] = self._parse_time(row[field])
            
            # Parse numeric fields
            numeric_fields = ['DTE', 'StrategyProfit', 'StrategyLoss', 'StrategyProfitReExecuteNo',
                            'StrategyLossReExecuteNo', 'LockPercent', 'TrailPercent',
                            'SqOff1Percent', 'SqOff2Percent', 'ProfitReaches', 'LockMinProfitAt',
                            'IncreaseInProfit', 'TrailMinProfitBy', 'StoplossCheckingInterval',
                            'TargetCheckingInterval', 'ReEntryCheckingInterval']
            
            for field in numeric_fields:
                if field in row and pd.notna(row[field]):
                    normalized = self._normalize_column_name(field)
                    if field in ['DTE', 'StrategyProfitReExecuteNo', 'StrategyLossReExecuteNo',
                               'StoplossCheckingInterval', 'TargetCheckingInterval', 'ReEntryCheckingInterval']:
                        # Handle time format for interval fields
                        value = row[field]
                        if isinstance(value, str) and ':' in value:
                            # Convert HH:MM:SS to seconds
                            parts = value.split(':')
                            if len(parts) == 3:
                                strategy[normalized] = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            else:
                                strategy[normalized] = int(value)
                        else:
                            strategy[normalized] = int(value)
                    else:
                        strategy[normalized] = float(row[field])
            
            # Parse boolean fields
            bool_fields = ['MoveSlToCost', 'ConsiderHedgePnLForStgyPnL', 'OnExpiryDayTradeNextExpiry']
            for field in bool_fields:
                if field in row and pd.notna(row[field]):
                    strategy[self._normalize_column_name(field)] = self._parse_bool(row[field])
            
            # Parse string fields
            string_fields = ['Weekdays', 'StrategyTrailingType', 'TgtTrackingFrom',
                           'TgtRegisterPriceFrom', 'SlTrackingFrom', 'SlRegisterPriceFrom',
                           'PnLCalculationFrom']
            for field in string_fields:
                if field in row and pd.notna(row[field]):
                    strategy[self._normalize_column_name(field)] = str(row[field])
            
            strategies.append(strategy)
        
        return strategies
    
    def _parse_leg_parameters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse leg parameters from DataFrame"""
        legs = []
        
        for idx, row in df.iterrows():
            # Skip idle legs
            if self._parse_bool(row.get('IsIdle', 'no')):
                continue
            
            leg = {
                'strategy_name': str(row.get('StrategyName', '')),
                'leg_no': int(row.get('LegID', idx + 1)),
                'quantity': int(row.get('Lots', 1)),
                'option_type': self._convert_instrument(row.get('Instrument', 'call')),
                'strike_selection': self._convert_strike_method(row.get('StrikeMethod', 'atm')),
                'strike_value': float(row.get('StrikeValue', 0)),
                'expiry_rule': self._convert_expiry(row.get('Expiry', 'current')),
                'transaction_type': str(row.get('Transaction', 'buy')).upper(),
            }
            
            # Parse SL and Target
            if 'SLValue' in row and pd.notna(row['SLValue']):
                leg['sl_percent'] = float(row['SLValue'])
                leg['sl_type'] = str(row.get('SLType', 'percentage'))
            
            if 'TGTValue' in row and pd.notna(row['TGTValue']):
                leg['target_percent'] = float(row['TGTValue'])
                leg['tgt_type'] = str(row.get('TGTType', 'percentage'))
            
            # Parse W&T fields
            if 'W&Type' in row and pd.notna(row['W&Type']):
                leg['wait_type'] = str(row['W&Type'])
                if 'W&TValue' in row and pd.notna(row['W&TValue']):
                    leg['wait_value'] = float(row['W&TValue'])
                if 'TrailW&T' in row and pd.notna(row['TrailW&T']):
                    leg['trail_wait'] = self._parse_bool(row['TrailW&T'])
            
            # Add other fields
            optional_fields = {
                'MatchPremium': 'match_premium',
                'StrikePremiumCondition': 'strike_premium_condition',
                'TrailSLType': 'trail_sl_type',
                'SL_TrailAt': 'sl_trail_at',
                'SL_TrailBy': 'sl_trail_by',
                'ReEntryType': 'reentry_type',
                'ReEnteriesCount': 'reentries_count',
                'OpenHedge': 'open_hedge',
                'HedgeStrikeMethod': 'hedge_strike_method',
                'HedgeStrikeValue': 'hedge_strike_value',
                'HedgeStrikePremiumCondition': 'hedge_strike_premium_condition'
            }
            
            for excel_col, field_name in optional_fields.items():
                if excel_col in row and pd.notna(row[excel_col]):
                    if field_name in ['sl_trail_at', 'sl_trail_by', 'hedge_strike_value']:
                        leg[field_name] = float(row[excel_col])
                    elif field_name in ['reentries_count']:
                        leg[field_name] = int(row[excel_col])
                    elif field_name == 'open_hedge':
                        leg[field_name] = self._parse_bool(row[excel_col])
                    else:
                        leg[field_name] = str(row[excel_col])
            
            # Default values for compatibility
            leg['entry_time'] = time(9, 20)
            leg['exit_time'] = time(15, 15)
            leg['expiry_value'] = 0
            
            legs.append(leg)
        
        return legs
    
    def _convert_instrument(self, value: str) -> str:
        """Convert instrument type to standard format"""
        value = str(value).upper()
        if value in ['CALL', 'CE']:
            return 'CE'
        elif value in ['PUT', 'PE']:
            return 'PE'
        elif value == 'FUT':
            return 'FUT'
        return value
    
    def _convert_expiry(self, value: str) -> str:
        """Convert expiry to standard format"""
        value = str(value).upper()
        mapping = {
            'CURRENT': 'CW',
            'NEXT': 'NW',
            'MONTHLY': 'CM',
            'NEXT MONTHLY': 'NM'
        }
        return mapping.get(value, value)
    
    def _convert_strike_method(self, value: str) -> str:
        """Convert strike method to standard format"""
        value = str(value).upper()
        
        # Standard mappings
        if value in ['ATM', 'FIXED', 'PREMIUM', 'DELTA']:
            return value
        
        # Handle ITM/OTM formats
        if value.startswith('ITM') or value.startswith('OTM'):
            return value
        
        # Handle special cases
        mapping = {
            'ATM WIDTH': 'ATM_WIDTH',
            'STRADDLE WIDTH': 'STRADDLE_WIDTH',
            'ATM MATCH': 'ATM_MATCH',
            'ATM DIFF': 'ATM_DIFF'
        }
        
        return mapping.get(value, value)
    
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
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean from various formats"""
        if pd.isna(value):
            return False
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1', 'Y', 'T']
        
        return bool(value)
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name to snake_case"""
        import re
        # Replace special characters
        name = name.replace('&', '_and_')
        name = name.replace(' ', '_')
        # Convert CamelCase to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        result = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        # Remove double underscores
        result = re.sub('_+', '_', result)
        return result.strip('_')