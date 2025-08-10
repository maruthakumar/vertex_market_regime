#!/usr/bin/env python3
"""
OI Parser - Handles parsing of Open Interest strategy Excel input files
Based on column mapping from /srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/oi/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date, time
import logging
import os

logger = logging.getLogger(__name__)


class OIParser:
    """Parser for OI (Open Interest) strategy Excel files"""
    
    def __init__(self):
        # Define expected columns based on column mapping
        self.general_param_columns = [
            'StrategyName', 'Timeframe', 'MaxOpenPositions', 'Underlying', 'Index', 
            'Weekdays', 'DTE', 'StrikeSelectionTime', 'StartTime', 'LastEntryTime', 
            'EndTime', 'StrategyProfit', 'StrategyLoss', 'StrategyProfitReExecuteNo',
            'StrategyLossReExecuteNo', 'StrategyTrailingType', 'PnLCalTime',
            'LockPercent', 'TrailPercent', 'SqOff1Time', 'SqOff1Percent',
            'SqOff2Time', 'SqOff2Percent', 'ProfitReaches', 'LockMinProfitAt',
            'IncreaseInProfit', 'TrailMinProfitBy', 'TgtTrackingFrom',
            'TgtRegisterPriceFrom', 'SlTrackingFrom', 'SlRegisterPriceFrom',
            'PnLCalculationFrom', 'ConsiderHedgePnLForStgyPnL',
            'StoplossCheckingInterval', 'TargetCheckingInterval',
            'ReEntryCheckingInterval', 'OnExpiryDayTradeNextExpiry'
        ]
        
        self.leg_param_columns = [
            'StrategyName', 'OiThreshold', 'LegID', 'Instrument', 'Transaction',
            'Expiry', 'MatchPremium', 'StrikeMethod', 'StrikeValue',
            'StrikePremiumCondition', 'SLType', 'SLValue', 'TGTType', 'TGTValue',
            'TrailSLType', 'SL_TrailAt', 'SL_TrailBy', 'Lots',
            'SL_ReEntryType', 'SL_ReEntryNo', 'TGT_ReEntryType', 'TGT_ReEntryNo',
            'OpenHedge', 'HedgeStrikeMethod', 'HedgeStrikeValue',
            'HedgeStrikePremiumCondition'
        ]
    
    def parse_oi_excel(self, excel_path: str) -> Dict[str, Any]:
        """
        Parse the OI strategy Excel file
        
        Args:
            excel_path: Path to input_maxoi.xlsx or similar
            
        Returns:
            Dictionary containing parsed OI strategies and legs
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"OI strategy file not found: {excel_path}")
        
        logger.info(f"Parsing OI strategy: {excel_path}")
        
        try:
            # Check which format this is
            excel_file = pd.ExcelFile(excel_path)
            sheet_names = excel_file.sheet_names
            
            # Check if this is the new format (with GeneralParameter and LegParameter sheets)
            if 'GeneralParameter' in sheet_names and 'LegParameter' in sheet_names:
                return self._parse_new_format(excel_path)
            # Check if this is the archive format (with Sheet1)
            elif 'Sheet1' in sheet_names:
                # Use the archive parser
                from .archive_parser import OIArchiveParser
                archive_parser = OIArchiveParser()
                return archive_parser.parse_archive_oi_excel(excel_path)
            else:
                raise ValueError(f"Unknown OI Excel format. Sheet names: {sheet_names}")
            
        except Exception as e:
            logger.error(f"Error parsing OI strategy: {e}")
            raise
    
    def _parse_new_format(self, excel_path: str) -> Dict[str, Any]:
        """Parse the new format with GeneralParameter and LegParameter sheets"""
        # Read both sheets
        general_df = pd.read_excel(excel_path, sheet_name='GeneralParameter')
        leg_df = pd.read_excel(excel_path, sheet_name='LegParameter')
        
        # Parse strategies
        strategies = []
        for idx, row in general_df.iterrows():
            strategy = self._parse_general_params(row)
            if strategy:
                # Get legs for this strategy
                strategy_legs = leg_df[leg_df['StrategyName'] == strategy['strategy_name']]
                legs = []
                for _, leg_row in strategy_legs.iterrows():
                    leg = self._parse_leg_params(leg_row)
                    if leg:
                        legs.append(leg)
                
                strategy['legs'] = legs
                strategies.append(strategy)
        
        return {
            'strategies': strategies,
            'source_file': excel_path,
            'format': 'new'
        }
    
    def _parse_general_params(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Parse GeneralParameter row"""
        
        strategy = {
            'strategy_name': str(row.get('StrategyName', '')),
            'timeframe': int(row.get('Timeframe', 3)),
            'max_open_positions': int(row.get('MaxOpenPositions', 1)),
            'underlying': str(row.get('Underlying', 'SPOT')).upper(),
            'index': str(row.get('Index', 'NIFTY')).upper(),
            'weekdays': self._parse_weekdays(row.get('Weekdays', '1,2,3,4,5')),
            'dte': int(row.get('DTE', 0)),
            
            # Time parameters - OI specific
            'strike_selection_time': self._parse_time(row.get('StrikeSelectionTime', 91600)),
            'start_time': self._parse_time(row.get('StartTime', 91600)),
            'last_entry_time': self._parse_time(row.get('LastEntryTime', 150000)),
            'end_time': self._parse_time(row.get('EndTime', 152000)),
            
            # Risk management
            'strategy_profit': float(row.get('StrategyProfit', 0)) if pd.notna(row.get('StrategyProfit')) else None,
            'strategy_loss': float(row.get('StrategyLoss', 0)) if pd.notna(row.get('StrategyLoss')) else None,
            'strategy_profit_reexecute_no': int(row.get('StrategyProfitReExecuteNo', 0)),
            'strategy_loss_reexecute_no': int(row.get('StrategyLossReExecuteNo', 0)),
            
            # Trailing parameters
            'strategy_trailing_type': str(row.get('StrategyTrailingType', '')),
            'profit_reaches': float(row.get('ProfitReaches', 0)) if pd.notna(row.get('ProfitReaches')) else None,
            'lock_min_profit_at': float(row.get('LockMinProfitAt', 0)) if pd.notna(row.get('LockMinProfitAt')) else None,
            'increase_in_profit': float(row.get('IncreaseInProfit', 0)) if pd.notna(row.get('IncreaseInProfit')) else None,
            'trail_min_profit_by': float(row.get('TrailMinProfitBy', 0)) if pd.notna(row.get('TrailMinProfitBy')) else None,
            
            # OI-specific partial exit parameters
            'pnl_cal_time': self._parse_time(row.get('PnLCalTime')) if pd.notna(row.get('PnLCalTime')) else None,
            'lock_percent': float(row.get('LockPercent', 0)) if pd.notna(row.get('LockPercent')) else None,
            'trail_percent': float(row.get('TrailPercent', 0)) if pd.notna(row.get('TrailPercent')) else None,
            'sq_off_1_time': self._parse_time(row.get('SqOff1Time')) if pd.notna(row.get('SqOff1Time')) else None,
            'sq_off_1_percent': float(row.get('SqOff1Percent', 0)) if pd.notna(row.get('SqOff1Percent')) else None,
            'sq_off_2_time': self._parse_time(row.get('SqOff2Time')) if pd.notna(row.get('SqOff2Time')) else None,
            'sq_off_2_percent': float(row.get('SqOff2Percent', 0)) if pd.notna(row.get('SqOff2Percent')) else None,
            
            # Tracking parameters
            'tgt_tracking_from': str(row.get('TgtTrackingFrom', 'close')).lower(),
            'tgt_register_price_from': str(row.get('TgtRegisterPriceFrom', 'tick')).lower(),
            'sl_tracking_from': str(row.get('SlTrackingFrom', 'close')).lower(),
            'sl_register_price_from': str(row.get('SlRegisterPriceFrom', 'tick')).lower(),
            'pnl_calculation_from': str(row.get('PnLCalculationFrom', 'close')).lower(),
            
            # Boolean flags
            'consider_hedge_pnl': self._parse_bool(row.get('ConsiderHedgePnLForStgyPnL', 'NO')),
            'on_expiry_day_trade_next': self._parse_bool(row.get('OnExpiryDayTradeNextExpiry', 'NO')),
            
            # Intervals (in seconds)
            'stoploss_checking_interval': int(row.get('StoplossCheckingInterval', 60)),
            'target_checking_interval': int(row.get('TargetCheckingInterval', 60)),
            'reentry_checking_interval': int(row.get('ReEntryCheckingInterval', 60))
        }
        
        return strategy
    
    def _parse_leg_params(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Parse LegParameter row"""
        
        leg = {
            'leg_id': str(row.get('LegID', '')),
            'instrument': self._normalize_instrument(row.get('Instrument', '')),
            'transaction': self._normalize_transaction(row.get('Transaction', '')),
            'expiry': self._normalize_expiry(row.get('Expiry', '')),
            'lots': int(row.get('Lots', 1)),
            
            # OI-specific parameters
            'oi_threshold': int(row.get('OiThreshold', 800000)),
            
            # Strike selection
            'strike_method': self._normalize_strike_method(row.get('StrikeMethod', 'MAXOI_1')),
            'strike_value': float(row.get('StrikeValue', 0)) if pd.notna(row.get('StrikeValue')) else 0,
            'match_premium': str(row.get('MatchPremium', '')).lower() if pd.notna(row.get('MatchPremium')) else None,
            'strike_premium_condition': str(row.get('StrikePremiumCondition', '=')),
            
            # Risk parameters
            'sl_type': str(row.get('SLType', 'percentage')).lower(),
            'sl_value': float(row.get('SLValue', 500)),  # Default 500% for SELL
            'tgt_type': str(row.get('TGTType', 'percentage')).lower(),
            'tgt_value': float(row.get('TGTValue', 100)),  # Default 100%
            
            # Trailing SL
            'trail_sl_type': str(row.get('TrailSLType', '')).lower() if pd.notna(row.get('TrailSLType')) else None,
            'sl_trail_at': float(row.get('SL_TrailAt', 0)) if pd.notna(row.get('SL_TrailAt')) else None,
            'sl_trail_by': float(row.get('SL_TrailBy', 0)) if pd.notna(row.get('SL_TrailBy')) else None,
            
            # Re-entry
            'sl_reentry_type': str(row.get('SL_ReEntryType', '')).lower() if pd.notna(row.get('SL_ReEntryType')) else None,
            'sl_reentry_no': int(row.get('SL_ReEntryNo', 0)),
            'tgt_reentry_type': str(row.get('TGT_ReEntryType', '')).lower() if pd.notna(row.get('TGT_ReEntryType')) else None,
            'tgt_reentry_no': int(row.get('TGT_ReEntryNo', 0)),
            
            # Hedge parameters
            'open_hedge': self._parse_bool(row.get('OpenHedge', 'NO')),
            'hedge_strike_method': self._normalize_strike_method(row.get('HedgeStrikeMethod', '')) if pd.notna(row.get('HedgeStrikeMethod')) else None,
            'hedge_strike_value': float(row.get('HedgeStrikeValue', 0)) if pd.notna(row.get('HedgeStrikeValue')) else None,
            'hedge_strike_premium_condition': str(row.get('HedgeStrikePremiumCondition', '=')) if pd.notna(row.get('HedgeStrikePremiumCondition')) else None
        }
        
        return leg
    
    def _parse_time(self, value: Any) -> time:
        """Parse time from various formats"""
        if pd.isna(value):
            return time(0, 0, 0)
        
        if isinstance(value, time):
            return value
        
        if isinstance(value, datetime):
            return value.time()
        
        # Handle HHMMSS integer format (common in OI sheets)
        if isinstance(value, (int, float)):
            time_str = f"{int(value):06d}"
            try:
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    return time(hour, minute, second)
            except:
                pass
        
        # Handle string formats
        time_str = str(value).strip()
        
        # Try HH:MM:SS format
        for fmt in ['%H:%M:%S', '%H:%M']:
            try:
                return datetime.strptime(time_str, fmt).time()
            except:
                continue
        
        logger.warning(f"Could not parse time: {value}, defaulting to 00:00:00")
        return time(0, 0, 0)
    
    def _parse_weekdays(self, value: Any) -> List[int]:
        """Parse weekdays string to list of integers"""
        if pd.isna(value):
            return [1, 2, 3, 4, 5]  # Default Mon-Fri
        
        weekdays_str = str(value).strip()
        try:
            return [int(d.strip()) for d in weekdays_str.split(',') if d.strip()]
        except:
            logger.warning(f"Could not parse weekdays: {value}, defaulting to Mon-Fri")
            return [1, 2, 3, 4, 5]
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean from various formats"""
        if pd.isna(value):
            return False
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1', 'Y', 'T']
        
        return bool(value)
    
    def _normalize_instrument(self, value: str) -> str:
        """Normalize instrument type"""
        instrument = str(value).upper().strip()
        
        mapping = {
            'CALL': 'CE',
            'PUT': 'PE',
            'CE': 'CE',
            'PE': 'PE',
            'FUT': 'FUT',
            'FUTURE': 'FUT'
        }
        
        return mapping.get(instrument, instrument)
    
    def _normalize_transaction(self, value: str) -> str:
        """Normalize transaction type"""
        transaction = str(value).upper().strip()
        
        mapping = {
            'BUY': 'BUY',
            'SELL': 'SELL',
            'B': 'BUY',
            'S': 'SELL'
        }
        
        return mapping.get(transaction, transaction)
    
    def _normalize_expiry(self, value: str) -> str:
        """Normalize expiry rule"""
        expiry = str(value).upper().strip()
        
        mapping = {
            'CURRENT': 'CW',
            'CW': 'CW',
            'CURRENT_WEEK': 'CW',
            'NEXT': 'NW',
            'NW': 'NW',
            'NEXT_WEEK': 'NW',
            'MONTHLY': 'CM',
            'CM': 'CM',
            'CURRENT_MONTH': 'CM',
            'NM': 'NM',
            'NEXT_MONTH': 'NM'
        }
        
        return mapping.get(expiry, expiry)
    
    def _normalize_strike_method(self, value: str) -> str:
        """Normalize strike selection method - including OI-specific methods"""
        method = str(value).upper().strip()
        
        # OI-specific methods
        oi_methods = {
            'MAXOI1': 'MAXOI_1',
            'MAXOI_1': 'MAXOI_1',
            'MAXOI2': 'MAXOI_2', 
            'MAXOI_2': 'MAXOI_2',
            'MAXOI3': 'MAXOI_3',
            'MAXOI_3': 'MAXOI_3',
            'MAXCOI1': 'MAXCOI_1',
            'MAXCOI_1': 'MAXCOI_1',
            'MAXCOI2': 'MAXCOI_2',
            'MAXCOI_2': 'MAXCOI_2',
            'MAXCOI3': 'MAXCOI_3',
            'MAXCOI_3': 'MAXCOI_3'
        }
        
        if method in oi_methods:
            return oi_methods[method]
        
        # Standard methods (from TBS/ORB)
        if method in ['ATM', 'ITM1', 'ITM2', 'ITM3', 'OTM1', 'OTM2', 'OTM3', 'FIXED']:
            return method
        
        # Alternative names
        mapping = {
            'PREMIUM': 'PREMIUM',
            'ATM WIDTH': 'ATM_WIDTH',
            'ATM_WIDTH': 'ATM_WIDTH',
            'STRADDLE WIDTH': 'STRADDLE_WIDTH',
            'STRADDLE_WIDTH': 'STRADDLE_WIDTH',
            'DELTA': 'DELTA'
        }
        
        return mapping.get(method, method)