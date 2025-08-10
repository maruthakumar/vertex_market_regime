#!/usr/bin/env python3
"""
Fixed TBS Parser that handles actual column names from input files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date, time
import logging
import os

logger = logging.getLogger(__name__)


class TBSParserFixed:
    """Fixed parser for TBS strategy Excel files"""
    
    def __init__(self):
        # Column mappings for LegParameter sheet
        self.leg_column_mappings = {
            'LegID': 'leg_no',
            'Instrument': 'option_type',  # call/put -> CE/PE
            'Transaction': 'transaction_type',  # sell/buy -> SELL/BUY
            'Expiry': 'expiry_rule',  # current/next -> CW/NW
            'StrikeMethod': 'strike_selection',  # atm/otm/itm -> ATM/OTM/ITM
            'StrikeValue': 'strike_value',
            'SLValue': 'sl_percent',
            'TGTValue': 'target_percent',
            'Lots': 'quantity'
        }
        
    def parse_multi_leg_excel_fixed(self, excel_path: str) -> Dict[str, Any]:
        """Parse multi-leg Excel with actual column names"""
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Multi-leg Excel file not found: {excel_path}")
        
        logger.info(f"Parsing multi-leg Excel: {excel_path}")
        
        try:
            # Read sheets
            general_df = pd.read_excel(excel_path, sheet_name='GeneralParameter')
            leg_df = pd.read_excel(excel_path, sheet_name='LegParameter')
            
            # Parse general parameters
            strategies = []
            for idx, row in general_df.iterrows():
                strategy = {
                    'strategy_name': str(row.get('StrategyName', f'Strategy_{idx}')),
                    'index': str(row.get('Index', 'NIFTY')).upper(),
                    'enabled': True,  # Assume enabled if in file
                    'legs': []
                }
                
                # Add other fields
                for col in general_df.columns:
                    if col not in ['StrategyName', 'Index']:
                        strategy[self._normalize_column_name(col)] = row[col]
                
                strategies.append(strategy)
            
            # Parse leg parameters with proper mapping
            legs = []
            for idx, row in leg_df.iterrows():
                leg = {
                    'strategy_name': str(row.get('StrategyName', ''))
                }
                
                # Map columns
                for old_col, new_col in self.leg_column_mappings.items():
                    if old_col in row:
                        value = row[old_col]
                        
                        # Convert values
                        if old_col == 'Instrument':
                            # Convert call/put to CE/PE
                            value = 'CE' if str(value).lower() == 'call' else 'PE'
                        elif old_col == 'Transaction':
                            # Convert to uppercase
                            value = str(value).upper()
                        elif old_col == 'Expiry':
                            # Convert current/next to CW/NW
                            if str(value).lower() == 'current':
                                value = 'CW'
                            elif str(value).lower() == 'next':
                                value = 'NW'
                            else:
                                value = str(value).upper()
                        elif old_col == 'StrikeMethod':
                            # Convert to uppercase
                            value = str(value).upper()
                        elif old_col in ['LegID', 'StrikeValue', 'Lots']:
                            # Convert to int
                            value = int(value)
                        elif old_col in ['SLValue', 'TGTValue']:
                            # Convert to float
                            value = float(value) if pd.notna(value) else None
                        
                        leg[new_col] = value
                
                # Add default values for missing fields
                leg['entry_time'] = time(9, 20)  # Default entry time
                leg['exit_time'] = time(15, 15)  # Default exit time
                leg['expiry_value'] = 0  # Default expiry value
                
                legs.append(leg)
            
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
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name to snake_case"""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def test_fixed_parser():
    """Test the fixed parser"""
    print("Testing Fixed TBS Parser")
    print("="*60)
    
    tbs_file = '/srv/samba/shared/input_tbs_multi_legs.xlsx'
    parser = TBSParserFixed()
    
    try:
        result = parser.parse_multi_leg_excel_fixed(tbs_file)
        
        print(f"✅ Parsed successfully!")
        print(f"Strategies found: {len(result['strategies'])}")
        
        for strategy in result['strategies']:
            print(f"\nStrategy: {strategy['strategy_name']}")
            print(f"  Index: {strategy['index']}")
            print(f"  Legs: {len(strategy['legs'])}")
            
            for i, leg in enumerate(strategy['legs']):
                print(f"\n  Leg {i+1}:")
                print(f"    Type: {leg['option_type']}")
                print(f"    Transaction: {leg['transaction_type']}")
                print(f"    Strike: {leg['strike_selection']} + {leg['strike_value']}")
                print(f"    Quantity: {leg['quantity']}")
                print(f"    SL%: {leg['sl_percent']}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_parser()