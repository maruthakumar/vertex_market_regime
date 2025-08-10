#!/usr/bin/env python3
"""
OI Archive Parser - Handles parsing of Open Interest strategy files from archive format
Compatible with input_maxoi.xlsx format
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date, time
import logging
import os

logger = logging.getLogger(__name__)


class OIArchiveParser:
    """Parser for OI strategy files in archive format"""
    
    def parse_archive_oi_excel(self, excel_path: str) -> Dict[str, Any]:
        """
        Parse the OI strategy Excel file in archive format
        
        Args:
            excel_path: Path to input_maxoi.xlsx
            
        Returns:
            Dictionary containing parsed OI strategies
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"OI strategy file not found: {excel_path}")
        
        logger.info(f"Parsing archive OI strategy: {excel_path}")
        
        try:
            # Read the main sheet
            df = pd.read_excel(excel_path, sheet_name='Sheet1')
            
            # Parse strategies
            strategies = []
            for idx, row in df.iterrows():
                strategy = self._parse_archive_row(row)
                if strategy:
                    strategies.append(strategy)
            
            return {
                'strategies': strategies,
                'source_file': excel_path,
                'format': 'archive'
            }
            
        except Exception as e:
            logger.error(f"Error parsing archive OI strategy: {e}")
            raise
    
    def _parse_archive_row(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Parse archive format row into strategy format"""
        
        strategy_id = str(row.get('id', ''))
        strike_to_trade = str(row.get('striketotrade', '')).lower()
        
        # Determine OI method from striketotrade column
        oi_method = 'MAXOI_1'  # Default
        if 'maxoi1' in strike_to_trade:
            oi_method = 'MAXOI_1'
        elif 'maxoi2' in strike_to_trade:
            oi_method = 'MAXOI_2'
        elif 'maxcoi1' in strike_to_trade:
            oi_method = 'MAXCOI_1'
        elif 'maxcoi2' in strike_to_trade:
            oi_method = 'MAXCOI_2'
        elif 'maxoi3' in strike_to_trade:
            oi_method = 'MAXOI_3'
        elif 'maxcoi3' in strike_to_trade:
            oi_method = 'MAXCOI_3'
        
        # Parse dates
        start_date = self._parse_date(row.get('startdate'))
        end_date = self._parse_date(row.get('enddate'))
        
        # Parse times
        entry_time = self._parse_time(row.get('entrytime', 92200))
        last_entry_time = self._parse_time(row.get('lastentrytime', 152400))
        exit_time = self._parse_time(row.get('exittime', 152500))
        
        # Determine expiry format
        expiry_str = str(row.get('expiry', 'current')).lower()
        if 'current' in expiry_str or 'cw' in expiry_str:
            expiry = 'CW'
        elif 'next' in expiry_str or 'nw' in expiry_str:
            expiry = 'NW'
        elif 'cm' in expiry_str:
            expiry = 'CM'
        elif 'nm' in expiry_str:
            expiry = 'NM'
        else:
            expiry = 'CW'  # Default
        
        # Convert to standard format
        strategy = {
            'strategy_name': strategy_id,
            'timeframe': 3,  # OI strategies typically use 3-minute timeframe
            'max_open_positions': 1,
            'underlying': 'SPOT',
            'index': str(row.get('underlyingname', 'NIFTY')).upper(),
            'weekdays': [1, 2, 3, 4, 5],  # Mon-Fri default
            'dte': int(row.get('dte', 0)),
            
            # Time parameters
            'strike_selection_time': entry_time,  # OI selection at entry time
            'start_time': entry_time,
            'last_entry_time': last_entry_time,
            'end_time': exit_time,
            
            # Risk management
            'strategy_profit': float(row.get('strategymaxprofit', 0)) if row.get('strategymaxprofit') else None,
            'strategy_loss': float(row.get('strategymaxloss', 0)) if row.get('strategymaxloss') else None,
            
            # OI-specific
            'strike_count': int(row.get('noofstrikeeachside', 40)),
            'slippage_percent': float(row.get('slippagepercent', 0.1)),
            
            # Date range (for portfolio)
            'start_date': start_date,
            'end_date': end_date,
            
            # Create legs based on OI method
            'legs': self._create_oi_legs(
                oi_method=oi_method,
                lots=int(row.get('lot', 1)),
                expiry=expiry
            )
        }
        
        return strategy
    
    def _create_oi_legs(self, oi_method: str, lots: int, expiry: str) -> List[Dict[str, Any]]:
        """Create OI legs based on method"""
        legs = []
        
        # For MAXOI/MAXCOI strategies, we typically trade both CE and PE
        # Based on the strike with maximum OI
        
        if 'MAXOI' in oi_method or 'MAXCOI' in oi_method:
            # CE leg
            legs.append({
                'leg_id': 'LEG1',
                'instrument': 'CE',
                'transaction': 'SELL',  # Typically sell options at high OI strikes
                'expiry': expiry,
                'lots': lots,
                'oi_threshold': 800000,  # Default threshold
                'strike_method': oi_method,
                'strike_value': 0,
                'sl_type': 'percentage',
                'sl_value': 500,  # 500% SL for SELL
                'tgt_type': 'percentage',
                'tgt_value': 100
            })
            
            # PE leg
            legs.append({
                'leg_id': 'LEG2',
                'instrument': 'PE',
                'transaction': 'SELL',
                'expiry': expiry,
                'lots': lots,
                'oi_threshold': 800000,
                'strike_method': oi_method,
                'strike_value': 0,
                'sl_type': 'percentage',
                'sl_value': 500,
                'tgt_type': 'percentage',
                'tgt_value': 100
            })
        
        return legs
    
    def _parse_date(self, date_val) -> Optional[date]:
        """Parse date from various formats"""
        if pd.isna(date_val):
            return None
        
        # Handle YYMMDD format
        date_str = str(int(date_val))
        if len(date_str) == 6:
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            return date(year, month, day)
        
        return None
    
    def _parse_time(self, time_val) -> time:
        """Parse time from HHMMSS format"""
        if pd.isna(time_val):
            return time(9, 16, 0)  # Default
        
        time_int = int(time_val)
        hours = time_int // 10000
        minutes = (time_int % 10000) // 100
        seconds = time_int % 100
        
        return time(hours, minutes, seconds)