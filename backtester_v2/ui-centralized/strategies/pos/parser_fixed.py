"""
Fixed Parser for POS (Positional) Strategy Excel files
Handles actual input sheet format with tabular structure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, time, date
import logging
from pathlib import Path
import re

from .models import (
    POSLegModel,
    POSPortfolioModel,
    POSStrategyModel,
    GreekLimits,
    AdjustmentRule,
    MarketRegimeFilter,
    VIXFilter,
    AdjustmentTrigger,
    AdjustmentAction,
    PositionType,
    OptionType,
    StrikeSelection
)
from .constants import (
    MAX_LEGS,
    ERROR_MESSAGES,
    DEFAULT_ENTRY_TIME,
    DEFAULT_EXIT_TIME,
    DEFAULT_GREEK_LIMITS
)

logger = logging.getLogger(__name__)


class POSParserFixed:
    """Fixed parser for actual POS strategy Excel format"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def parse_input(self, 
                   portfolio_file: str,
                   strategy_file: str,
                   adjustment_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse all input files for POS strategy
        
        Args:
            portfolio_file: Path to portfolio Excel file
            strategy_file: Path to strategy Excel file
            adjustment_file: Optional path to adjustment rules file
            
        Returns:
            Dictionary containing parsed strategy configuration
        """
        try:
            # Parse portfolio settings
            portfolio_data = self.parse_portfolio_excel(portfolio_file)
            
            # Parse strategy legs
            strategy_data = self.parse_strategy_excel(strategy_file)
            
            # Parse adjustments if provided
            adjustments = {}
            if adjustment_file:
                adjustments = self.parse_adjustment_excel(adjustment_file)
                
            # Combine all data
            result = {
                "portfolio": portfolio_data,
                "strategy": strategy_data,
                "adjustments": adjustments,
                "errors": self.errors,
                "warnings": self.warnings
            }
            
            # Create and validate complete model
            if not self.errors:
                strategy_model = self._create_strategy_model(result)
                result["model"] = strategy_model
                
            return result
            
        except Exception as e:
            logger.error(f"Error parsing POS input files: {str(e)}")
            self.errors.append(f"Parser error: {str(e)}")
            return {
                "errors": self.errors,
                "warnings": self.warnings
            }
    
    def parse_portfolio_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse portfolio settings from actual Excel format"""
        try:
            df = pd.read_excel(excel_path, sheet_name='PortfolioSetting')
            
            # Read the actual format with columns
            row = df.iloc[0] if not df.empty else pd.Series()
            
            portfolio_data = {
                'portfolio_name': str(row.get('PortfolioName', 'POS_Portfolio')),
                'strategy_name': 'Positional_Strategy',
                'strategy_type': 'CUSTOM',  # Will be determined from legs
                'start_date': pd.to_datetime(row['StartDate']).date() if 'StartDate' in row else date.today(),
                'end_date': pd.to_datetime(row['EndDate']).date() if 'EndDate' in row else date.today(),
                'index_name': str(row.get('IndexName', 'NIFTY')).upper(),
                'underlying_price_type': 'SPOT',
                'position_sizing': 'FIXED',
                'max_positions': 1,
                'position_size_value': float(row.get('Multiplier', 1)) * 50 * 100,  # lots * lot_size * multiplier
                'max_portfolio_risk': float(row.get('PortfolioStoploss', 0.02)) if 'PortfolioStoploss' in row else 0.02,
                'transaction_costs': float(row.get('SlippagePercent', 0.1)) / 100 if 'SlippagePercent' in row else 0.001,
                'slippage_value': float(row.get('SlippagePercent', 0.1)) / 100 if 'SlippagePercent' in row else 0.001,
                'use_intraday_data': bool(row.get('IsTickBT', False)),
                'calculate_greeks': True,
                'enable_adjustments': False,  # Start simple
                'enabled': bool(row.get('Enabled', True))
            }
            
            # Add default Greek limits
            portfolio_data['greek_limits'] = DEFAULT_GREEK_LIMITS
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error parsing portfolio Excel: {str(e)}")
            self.errors.append(f"Portfolio parsing error: {str(e)}")
            return {}
    
    def parse_strategy_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse strategy legs from actual Excel format"""
        try:
            # Read LegParameter sheet
            leg_df = pd.read_excel(excel_path, sheet_name='LegParameter')
            
            legs = []
            for idx, row in leg_df.iterrows():
                # Map actual columns to expected format
                leg = {
                    'leg_id': idx + 1,
                    'leg_name': str(row.get('LegID', f'leg_{idx+1}')),
                    'option_type': self._extract_option_type(row),
                    'position_type': str(row.get('Transaction', 'BUY')).upper(),  # BUY/SELL
                    'strike_selection': self._map_strike_method(row.get('StrikeMethod', 'ATM')),
                    'strike_offset': self._extract_strike_offset(row.get('StrikeMethod', '')),
                    'lots': int(row.get('Lots', 1)),
                    'lot_size': 50,  # NIFTY lot size
                    'entry_time': time(9, 20),  # Default
                    'exit_time': time(15, 20),  # Default
                    'stop_loss': float(row['StopLossValue']) if pd.notna(row.get('StopLossValue')) else None,
                    'take_profit': float(row['TargetValue']) if pd.notna(row.get('TargetValue')) else None,
                    'is_active': bool(row.get('IsActive', True))
                }
                
                # Handle specific strike price if provided
                if pd.notna(row.get('Strike')):
                    leg['strike_price'] = float(row['Strike'])
                    leg['strike_selection'] = 'STRIKE_PRICE'
                
                # Determine expiry from leg name or other fields
                leg['expiry_type'] = self._determine_expiry_type(row)
                
                legs.append(leg)
                
            # Detect strategy type from legs
            strategy_type = self._detect_strategy_type(legs)
            
            return {
                'legs': legs,
                'strategy_type': strategy_type,
                'entry_logic': 'ALL',
                'exit_logic': 'ALL',
                'settings': {}
            }
            
        except Exception as e:
            logger.error(f"Error parsing strategy Excel: {str(e)}")
            self.errors.append(f"Strategy parsing error: {str(e)}")
            return {'legs': []}
    
    def _extract_option_type(self, row: pd.Series) -> str:
        """Extract option type from row data"""
        # Check direct option type column
        if 'OptionType' in row and pd.notna(row['OptionType']):
            opt_type = str(row['OptionType']).upper()
            if 'CALL' in opt_type or 'CE' in opt_type:
                return 'CALL'
            elif 'PUT' in opt_type or 'PE' in opt_type:
                return 'PUT'
        
        # Check instrument column
        if 'Instrument' in row and pd.notna(row['Instrument']):
            instrument = str(row['Instrument']).upper()
            if 'CALL' in instrument or 'CE' in instrument:
                return 'CALL'
            elif 'PUT' in instrument or 'PE' in instrument:
                return 'PUT'
        
        # Check leg name
        if 'LegID' in row and pd.notna(row['LegID']):
            leg_id = str(row['LegID']).lower()
            if 'call' in leg_id:
                return 'CALL'
            elif 'put' in leg_id:
                return 'PUT'
        
        # Default to CALL
        self.warnings.append(f"Could not determine option type for leg, defaulting to CALL")
        return 'CALL'
    
    def _map_strike_method(self, method: str) -> str:
        """Map Excel strike method to internal format"""
        if pd.isna(method):
            return 'ATM'
            
        method_upper = str(method).upper()
        
        if 'ATM' in method_upper:
            return 'ATM'
        elif 'ITM' in method_upper:
            return 'ITM'
        elif 'OTM' in method_upper:
            return 'OTM'
        elif 'STRIKE' in method_upper or 'FIXED' in method_upper:
            return 'STRIKE_PRICE'
        elif 'DELTA' in method_upper:
            return 'DELTA_BASED'
        elif 'PERCENT' in method_upper or '%' in method_upper:
            return 'PERCENTAGE_BASED'
        else:
            return 'ATM'  # Default
    
    def _extract_strike_offset(self, method: str) -> float:
        """Extract offset value from strike method string"""
        if pd.isna(method):
            return 0
            
        # Extract number from strings like 'OTM_100' or 'ITM 50' or 'ATM+100'
        match = re.search(r'[-+]?\d+(?:\.\d+)?', str(method))
        return float(match.group()) if match else 0
    
    def _determine_expiry_type(self, row: pd.Series) -> str:
        """Determine expiry type from row data"""
        # Check expiry type column
        if 'ExpiryType' in row and pd.notna(row['ExpiryType']):
            expiry = str(row['ExpiryType']).upper()
            if 'CURRENT' in expiry and 'WEEK' in expiry:
                return 'CURRENT_WEEK'
            elif 'NEXT' in expiry and 'WEEK' in expiry:
                return 'NEXT_WEEK'
            elif 'CURRENT' in expiry and 'MONTH' in expiry:
                return 'CURRENT_MONTH'
            elif 'NEXT' in expiry and 'MONTH' in expiry:
                return 'NEXT_MONTH'
        
        # Check leg name
        if 'LegID' in row and pd.notna(row['LegID']):
            leg_id = str(row['LegID']).lower()
            if 'weekly' in leg_id or 'week' in leg_id:
                if 'current' in leg_id:
                    return 'CURRENT_WEEK'
                else:
                    return 'NEXT_WEEK'
            elif 'monthly' in leg_id or 'month' in leg_id:
                if 'current' in leg_id:
                    return 'CURRENT_MONTH'
                else:
                    return 'NEXT_MONTH'
        
        # Default to current week
        return 'CURRENT_WEEK'
    
    def _detect_strategy_type(self, legs: List[Dict]) -> str:
        """Detect strategy type from leg configuration"""
        num_legs = len(legs)
        
        if num_legs == 0:
            return 'CUSTOM'
        elif num_legs == 1:
            return 'SINGLE_LEG'
        elif num_legs == 2:
            # Check for spreads
            if legs[0]['option_type'] == legs[1]['option_type']:
                if legs[0]['position_type'] != legs[1]['position_type']:
                    return 'VERTICAL_SPREAD'
                else:
                    return 'CALENDAR_SPREAD'
            else:
                return 'STRANGLE'  # Different option types
        elif num_legs == 4:
            # Check for iron condor/butterfly
            put_legs = [l for l in legs if l['option_type'] == 'PUT']
            call_legs = [l for l in legs if l['option_type'] == 'CALL']
            
            if len(put_legs) == 2 and len(call_legs) == 2:
                # Check positions
                buy_count = sum(1 for l in legs if l['position_type'] == 'BUY')
                sell_count = sum(1 for l in legs if l['position_type'] == 'SELL')
                
                if buy_count == 2 and sell_count == 2:
                    return 'IRON_CONDOR'
        
        return 'CUSTOM'
    
    def parse_adjustment_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse adjustment rules from Excel (if provided)"""
        try:
            df = pd.read_excel(excel_path, sheet_name='AdjustmentRules')
            
            adjustments = []
            
            for idx, row in df.iterrows():
                if pd.notna(row.get('RuleID')):
                    rule = {
                        'rule_id': str(row['RuleID']),
                        'trigger_type': str(row.get('TriggerType', 'PRICE_BASED')),
                        'trigger_condition': str(row.get('TriggerCondition', '')),
                        'trigger_value': float(row.get('TriggerValue', 0)),
                        'action_type': str(row.get('ActionType', 'CLOSE_POSITION')),
                        'action_params': {},
                        'max_adjustments': int(row.get('MaxAdjustments', 3)),
                        'cooldown_period': int(row.get('CooldownPeriod', 60)),
                        'priority': int(row.get('Priority', 1))
                    }
                    adjustments.append(rule)
                    
            return {"rules": adjustments}
            
        except Exception as e:
            logger.warning(f"No adjustment rules found or error parsing: {str(e)}")
            return {"rules": []}
    
    def _create_strategy_model(self, parsed_data: Dict[str, Any]) -> POSStrategyModel:
        """Create validated strategy model from parsed data"""
        # Create portfolio model
        portfolio = POSPortfolioModel(**parsed_data['portfolio'])
        
        # Update strategy type based on detected type
        if 'strategy_type' in parsed_data['strategy']:
            portfolio.strategy_type = parsed_data['strategy']['strategy_type']
        
        # Create leg models
        legs = []
        for leg_data in parsed_data['strategy']['legs']:
            # Clean up leg data
            clean_leg_data = {k: v for k, v in leg_data.items() if k != 'expiry_type'}
            leg = POSLegModel(**clean_leg_data)
            legs.append(leg)
            
        # Create strategy model
        strategy = POSStrategyModel(
            portfolio=portfolio,
            legs=legs
        )
        
        return strategy
    
    def validate_parsed_data(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Validate parsed data for consistency and completeness"""
        validation_errors = []
        
        # Validate date range
        if parsed_data['portfolio'].get('start_date') >= parsed_data['portfolio'].get('end_date'):
            validation_errors.append("End date must be after start date")
            
        # Validate legs
        if not parsed_data['strategy']['legs']:
            validation_errors.append("No legs found in strategy")
        
        leg_ids = set()
        for leg in parsed_data['strategy']['legs']:
            if leg['leg_id'] in leg_ids:
                validation_errors.append(f"Duplicate leg ID: {leg['leg_id']}")
            leg_ids.add(leg['leg_id'])
            
        return validation_errors