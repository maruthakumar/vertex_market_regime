#!/usr/bin/env python3
"""
TBS Parser that matches actual input sheet structure from 
/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/tbs/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, time
import logging
import os

logger = logging.getLogger(__name__)


class TBSParserActual:
    """Parser for actual TBS strategy Excel files"""
    
    def __init__(self):
        # Portfolio columns from actual files
        self.portfolio_columns = [
            'StartDate', 'EndDate', 'IsTickBT', 'Enabled', 'PortfolioName',
            'PortfolioTarget', 'PortfolioStoploss', 'PortfolioTrailingType',
            'PnLCalTime', 'LockPercent', 'TrailPercent', 'SqOff1Time', 'SqOff1Percent',
            'SqOff2Time', 'SqOff2Percent', 'ProfitReaches', 'LockMinProfitAt',
            'IncreaseInProfit', 'TrailMinProfitBy', 'Multiplier', 'SlippagePercent'
        ]
        
        # Strategy setting columns
        self.strategy_columns = ['Enabled', 'PortfolioName', 'StrategyType', 'StrategyExcelFilePath']
        
        # General parameter columns
        self.general_columns = [
            'StrategyName', 'MoveSlToCost', 'Underlying', 'Index', 'Weekdays', 'DTE',
            'StrikeSelectionTime', 'StartTime', 'LastEntryTime', 'EndTime',
            'StrategyProfit', 'StrategyLoss', 'StrategyProfitReExecuteNo',
            'StrategyLossReExecuteNo', 'StrategyTrailingType', 'PnLCalTime',
            'LockPercent', 'TrailPercent', 'SqOff1Time', 'SqOff1Percent',
            'SqOff2Time', 'SqOff2Percent', 'ProfitReaches', 'LockMinProfitAt',
            'IncreaseInProfit', 'TrailMinProfitBy', 'TgtTrackingFrom',
            'TgtRegisterPriceFrom', 'SlTrackingFrom', 'SlRegisterPriceFrom',
            'PnLCalculationFrom', 'ConsiderHedgePnLForStgyPnL',
            'StoplossCheckingInterval', 'TargetCheckingInterval',
            'ReEntryCheckingInterval', 'OnExpiryDayTradeNextExpiry'
        ]
        
        # Leg parameter columns
        self.leg_columns = [
            'StrategyName', 'IsIdle', 'LegID', 'Instrument', 'Transaction', 'Expiry',
            'W&Type', 'W&TValue', 'TrailW&T', 'StrikeMethod', 'MatchPremium',
            'StrikeValue', 'StrikePremiumCondition', 'SLType', 'SLValue',
            'TGTType', 'TGTValue', 'TrailSLType', 'SL_TrailAt', 'SL_TrailBy',
            'Lots', 'ReEntryType', 'ReEnteriesCount', 'OnEntry_OpenTradeOn',
            'OnEntry_SqOffTradeOff', 'OnEntry_SqOffAllLegs', 'OnEntry_OpenTradeDelay',
            'OnEntry_SqOffDelay', 'OnExit_OpenTradeOn', 'OnExit_SqOffTradeOff',
            'OnExit_SqOffAllLegs', 'OnExit_OpenAllLegs', 'OnExit_OpenTradeDelay',
            'OnExit_SqOffDelay', 'OpenHedge', 'HedgeStrikeMethod',
            'HedgeStrikeValue', 'HedgeStrikePremiumCondition'
        ]
        
    def parse_portfolio_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse the portfolio Excel file"""
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Portfolio Excel file not found: {excel_path}")
        
        logger.info(f"Parsing portfolio Excel: {excel_path}")
        
        try:
            # Read sheets
            portfolio_df = pd.read_excel(excel_path, sheet_name='PortfolioSetting')
            strategy_df = pd.read_excel(excel_path, sheet_name='StrategySetting')
            
            # Parse portfolio settings (filter enabled)
            portfolios = []
            for idx, row in portfolio_df.iterrows():
                if self._parse_bool(row.get('Enabled', 'NO')):
                    portfolio = self._parse_portfolio_row(row)
                    portfolios.append(portfolio)
            
            # Parse strategy settings (filter enabled)
            strategies = []
            for idx, row in strategy_df.iterrows():
                if self._parse_bool(row.get('Enabled', 'NO')):
                    strategy = self._parse_strategy_row(row, idx)
                    strategies.append(strategy)
            
            return {
                'portfolios': portfolios,
                'strategies': strategies,
                'source_file': excel_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing portfolio Excel: {e}")
            raise
    
    def parse_strategy_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse TBS strategy Excel file"""
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Strategy Excel file not found: {excel_path}")
        
        logger.info(f"Parsing strategy Excel: {excel_path}")
        
        try:
            # Read sheets
            general_df = pd.read_excel(excel_path, sheet_name='GeneralParameter')
            leg_df = pd.read_excel(excel_path, sheet_name='LegParameter')
            
            # Parse general parameters
            strategies = []
            for idx, row in general_df.iterrows():
                strategy = self._parse_general_row(row)
                
                # Find legs for this strategy
                strategy_name = strategy['strategy_name']
                legs = []
                
                for leg_idx, leg_row in leg_df.iterrows():
                    if leg_row.get('StrategyName') == strategy_name:
                        # Skip idle legs
                        if not self._parse_bool(leg_row.get('IsIdle', 'no')):
                            leg = self._parse_leg_row(leg_row)
                            legs.append(leg)
                
                strategy['legs'] = legs
                strategies.append(strategy)
            
            return {
                'strategies': strategies,
                'source_file': excel_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing strategy Excel: {e}")
            raise
    
    def _parse_portfolio_row(self, row: pd.Series) -> Dict[str, Any]:
        """Parse a portfolio settings row"""
        portfolio = {
            'portfolio_name': str(row.get('PortfolioName', '')).upper(),
            'start_date': self._parse_date(row.get('StartDate')),
            'end_date': self._parse_date(row.get('EndDate')),
            'is_tick_bt': self._parse_bool(row.get('IsTickBT', 'NO')),
            'capital': float(row.get('Capital', 1000000)) if 'Capital' in row else 1000000.0,
            'index': str(row.get('Index', 'NIFTY')).upper() if 'Index' in row else 'NIFTY',
            'lot_size': int(row.get('LotSize', 50)) if 'LotSize' in row else 50,
            'margin': float(row.get('Margin', 0.15)) if 'Margin' in row else 0.15,
        }
        
        # Add optional fields
        optional_fields = {
            'PortfolioTarget': ('portfolio_target', float),
            'PortfolioStoploss': ('portfolio_stoploss', float),
            'PortfolioTrailingType': ('portfolio_trailing_type', str),
            'PnLCalTime': ('pnl_cal_time', self._parse_time),
            'LockPercent': ('lock_percent', float),
            'TrailPercent': ('trail_percent', float),
            'SqOff1Time': ('sqoff1_time', self._parse_time),
            'SqOff1Percent': ('sqoff1_percent', float),
            'SqOff2Time': ('sqoff2_time', self._parse_time),
            'SqOff2Percent': ('sqoff2_percent', float),
            'ProfitReaches': ('profit_reaches', float),
            'LockMinProfitAt': ('lock_min_profit_at', float),
            'IncreaseInProfit': ('increase_in_profit', float),
            'TrailMinProfitBy': ('trail_min_profit_by', float),
            'Multiplier': ('multiplier', float),
            'SlippagePercent': ('slippage_percent', float),
        }
        
        for excel_col, (field_name, converter) in optional_fields.items():
            if excel_col in row and pd.notna(row[excel_col]):
                portfolio[field_name] = converter(row[excel_col])
        
        return portfolio
    
    def _parse_strategy_row(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """Parse a strategy settings row"""
        return {
            'strategy_index': idx,
            'portfolio_name': str(row.get('PortfolioName', '')).upper(),
            'strategy_type': str(row.get('StrategyType', 'TBS')).upper(),
            'strategy_excel_file_path': str(row.get('StrategyExcelFilePath', '')),
            'enabled': True  # Already filtered
        }
    
    def _parse_general_row(self, row: pd.Series) -> Dict[str, Any]:
        """Parse a general parameter row"""
        strategy = {
            'strategy_name': str(row.get('StrategyName', '')),
            'index': str(row.get('Index', 'NIFTY')).upper(),
            'underlying': str(row.get('Underlying', 'SPOT')).upper(),
        }
        
        # Add optional fields
        optional_fields = {
            'MoveSlToCost': ('move_sl_to_cost', self._parse_bool),
            'Weekdays': ('weekdays', str),
            'DTE': ('dte', int),
            'StrikeSelectionTime': ('strike_selection_time', self._parse_time),
            'StartTime': ('start_time', self._parse_time),
            'LastEntryTime': ('last_entry_time', self._parse_time),
            'EndTime': ('end_time', self._parse_time),
            'StrategyProfit': ('strategy_profit', float),
            'StrategyLoss': ('strategy_loss', float),
            'StrategyProfitReExecuteNo': ('strategy_profit_reexecute_no', int),
            'StrategyLossReExecuteNo': ('strategy_loss_reexecute_no', int),
            'StrategyTrailingType': ('strategy_trailing_type', str),
            'PnLCalTime': ('pnl_cal_time', self._parse_time),
            'LockPercent': ('lock_percent', float),
            'TrailPercent': ('trail_percent', float),
            'SqOff1Time': ('sqoff1_time', self._parse_time),
            'SqOff1Percent': ('sqoff1_percent', float),
            'SqOff2Time': ('sqoff2_time', self._parse_time),
            'SqOff2Percent': ('sqoff2_percent', float),
            'ProfitReaches': ('profit_reaches', float),
            'LockMinProfitAt': ('lock_min_profit_at', float),
            'IncreaseInProfit': ('increase_in_profit', float),
            'TrailMinProfitBy': ('trail_min_profit_by', float),
            'TgtTrackingFrom': ('tgt_tracking_from', str),
            'TgtRegisterPriceFrom': ('tgt_register_price_from', str),
            'SlTrackingFrom': ('sl_tracking_from', str),
            'SlRegisterPriceFrom': ('sl_register_price_from', str),
            'PnLCalculationFrom': ('pnl_calculation_from', str),
            'ConsiderHedgePnLForStgyPnL': ('consider_hedge_pnl', self._parse_bool),
            'StoplossCheckingInterval': ('stoploss_checking_interval', int),
            'TargetCheckingInterval': ('target_checking_interval', int),
            'ReEntryCheckingInterval': ('reentry_checking_interval', int),
            'OnExpiryDayTradeNextExpiry': ('on_expiry_day_trade_next_expiry', self._parse_bool),
        }
        
        for excel_col, (field_name, converter) in optional_fields.items():
            if excel_col in row and pd.notna(row[excel_col]):
                try:
                    strategy[field_name] = converter(row[excel_col])
                except:
                    logger.warning(f"Failed to convert {excel_col}: {row[excel_col]}")
        
        return strategy
    
    def _parse_leg_row(self, row: pd.Series) -> Dict[str, Any]:
        """Parse a leg parameter row"""
        leg = {
            'leg_id': str(row.get('LegID', '')),
            'instrument': self._convert_instrument(row.get('Instrument', 'call')),
            'transaction_type': str(row.get('Transaction', 'buy')).upper(),
            'expiry': self._convert_expiry(row.get('Expiry', 'current')),
            'strike_method': self._convert_strike_method(row.get('StrikeMethod', 'atm')),
            'strike_value': float(row.get('StrikeValue', 0)),
            'lots': int(row.get('Lots', 1)),
        }
        
        # Add optional fields
        optional_fields = {
            'W&Type': ('wait_type', str),
            'W&TValue': ('wait_value', float),
            'TrailW&T': ('trail_wait', self._parse_bool),
            'MatchPremium': ('match_premium', str),
            'StrikePremiumCondition': ('strike_premium_condition', str),
            'SLType': ('sl_type', str),
            'SLValue': ('sl_value', float),
            'TGTType': ('tgt_type', str),
            'TGTValue': ('tgt_value', float),
            'TrailSLType': ('trail_sl_type', str),
            'SL_TrailAt': ('sl_trail_at', float),
            'SL_TrailBy': ('sl_trail_by', float),
            'ReEntryType': ('reentry_type', str),
            'ReEnteriesCount': ('reentries_count', int),
            'OpenHedge': ('open_hedge', self._parse_bool),
            'HedgeStrikeMethod': ('hedge_strike_method', str),
            'HedgeStrikeValue': ('hedge_strike_value', float),
            'HedgeStrikePremiumCondition': ('hedge_strike_premium_condition', str),
        }
        
        for excel_col, (field_name, converter) in optional_fields.items():
            if excel_col in row and pd.notna(row[excel_col]):
                try:
                    leg[field_name] = converter(row[excel_col])
                except:
                    logger.warning(f"Failed to convert {excel_col}: {row[excel_col]}")
        
        # Add time fields
        time_fields = [
            'OnEntry_OpenTradeOn', 'OnEntry_SqOffTradeOff',
            'OnEntry_OpenTradeDelay', 'OnEntry_SqOffDelay',
            'OnExit_OpenTradeOn', 'OnExit_SqOffTradeOff',
            'OnExit_OpenTradeDelay', 'OnExit_SqOffDelay'
        ]
        
        for field in time_fields:
            if field in row and pd.notna(row[field]):
                leg[self._normalize_column_name(field)] = self._parse_time(row[field])
        
        # Add boolean fields
        bool_fields = [
            'OnEntry_SqOffAllLegs', 'OnExit_SqOffAllLegs', 'OnExit_OpenAllLegs'
        ]
        
        for field in bool_fields:
            if field in row and pd.notna(row[field]):
                leg[self._normalize_column_name(field)] = self._parse_bool(row[field])
        
        return leg
    
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
        
        # Handle ATM with offset
        if value == 'ATM':
            return 'ATM'
        
        # Handle ITM/OTM formats
        if value.startswith('ITM'):
            return value
        elif value.startswith('OTM'):
            return value
        
        # Handle other methods
        mapping = {
            'FIXED': 'FIXED',
            'PREMIUM': 'PREMIUM',
            'ATM WIDTH': 'ATM_WIDTH',
            'STRADDLE WIDTH': 'STRADDLE_WIDTH',
            'ATM MATCH': 'ATM_MATCH',
            'ATM DIFF': 'ATM_DIFF',
            'DELTA': 'DELTA'
        }
        
        return mapping.get(value, value)
    
    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse date from various formats"""
        if pd.isna(value):
            return None
        
        if isinstance(value, (datetime, date)):
            return value.date() if isinstance(value, datetime) else value
        
        # Try parsing string formats
        date_str = str(value)
        
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
        
        # Convert to string
        time_str = str(value)
        
        # Handle HHMMSS format (e.g., 92000 for 09:20:00)
        if time_str.isdigit() and len(time_str) in [5, 6]:
            # Pad with leading zero if needed
            time_str = time_str.zfill(6)
            try:
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
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
        
        value_str = str(value).upper()
        return value_str in ['YES', 'TRUE', '1', 'Y', 'T']
    
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


def test_actual_parser():
    """Test the actual parser with real files"""
    print("Testing Actual TBS Parser")
    print("="*60)
    
    portfolio_path = '/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/tbs/input_portfolio.xlsx'
    tbs_path = '/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/tbs/input_tbs_portfolio.xlsx'
    
    parser = TBSParserActual()
    
    # Test portfolio parsing
    print("\n1. Testing Portfolio Parsing...")
    try:
        portfolio_data = parser.parse_portfolio_excel(portfolio_path)
        print(f"✅ Parsed {len(portfolio_data['portfolios'])} portfolios")
        print(f"✅ Found {len(portfolio_data['strategies'])} strategies")
        
        if portfolio_data['portfolios']:
            p = portfolio_data['portfolios'][0]
            print(f"\nFirst portfolio:")
            print(f"  Name: {p['portfolio_name']}")
            print(f"  Start: {p['start_date']}")
            print(f"  End: {p['end_date']}")
            
    except Exception as e:
        print(f"❌ Portfolio parsing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test strategy parsing
    print("\n2. Testing Strategy Parsing...")
    try:
        strategy_data = parser.parse_strategy_excel(tbs_path)
        print(f"✅ Parsed {len(strategy_data['strategies'])} strategies")
        
        for i, strategy in enumerate(strategy_data['strategies']):
            print(f"\nStrategy {i+1}: {strategy['strategy_name']}")
            print(f"  Index: {strategy['index']}")
            print(f"  Legs: {len(strategy['legs'])}")
            
            for j, leg in enumerate(strategy['legs'][:2]):  # Show first 2 legs
                print(f"\n  Leg {j+1}:")
                print(f"    ID: {leg['leg_id']}")
                print(f"    Instrument: {leg['instrument']}")
                print(f"    Transaction: {leg['transaction_type']}")
                print(f"    Strike: {leg['strike_method']} + {leg['strike_value']}")
                print(f"    Lots: {leg['lots']}")
                
    except Exception as e:
        print(f"❌ Strategy parsing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_actual_parser()