"""
Parser for POS (Positional) Strategy Excel files
Handles portfolio settings and multi-leg configurations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, time, date
import logging
from pathlib import Path

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


class POSParser:
    """Parser for POS strategy Excel files"""
    
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
        """Parse portfolio settings from Excel"""
        try:
            df = pd.read_excel(excel_path, sheet_name='PortfolioSetting')
            
            portfolio_data = {}
            
            # Parse each row as key-value pair
            for idx, row in df.iterrows():
                if pd.notna(row.get('Parameter')) and pd.notna(row.get('Value')):
                    param = str(row['Parameter']).strip()
                    value = row['Value']
                    
                    # Parse specific parameters
                    if param == 'PortfolioName':
                        portfolio_data['portfolio_name'] = str(value)
                    elif param == 'StrategyName':
                        portfolio_data['strategy_name'] = str(value)
                    elif param == 'StrategyType':
                        portfolio_data['strategy_type'] = str(value).upper()
                    elif param == 'StartDate':
                        portfolio_data['start_date'] = pd.to_datetime(value).date()
                    elif param == 'EndDate':
                        portfolio_data['end_date'] = pd.to_datetime(value).date()
                    elif param == 'IndexName':
                        portfolio_data['index_name'] = str(value).upper()
                    elif param == 'UnderlyingPriceType':
                        portfolio_data['underlying_price_type'] = str(value).upper()
                    elif param == 'PositionSizing':
                        portfolio_data['position_sizing'] = str(value)
                    elif param == 'MaxPositions':
                        portfolio_data['max_positions'] = int(value)
                    elif param == 'PositionSizeValue':
                        portfolio_data['position_size_value'] = float(value)
                    elif param == 'MaxPortfolioRisk':
                        portfolio_data['max_portfolio_risk'] = float(value)
                    elif param == 'MaxDailyLoss':
                        portfolio_data['max_daily_loss'] = float(value) if value else None
                    elif param == 'MaxDrawdown':
                        portfolio_data['max_drawdown'] = float(value) if value else None
                    elif param == 'RebalanceFrequency':
                        portfolio_data['rebalance_frequency'] = str(value)
                    elif param == 'TransactionCosts':
                        portfolio_data['transaction_costs'] = float(value)
                    elif param == 'SlippageModel':
                        portfolio_data['slippage_model'] = str(value)
                    elif param == 'SlippageValue':
                        portfolio_data['slippage_value'] = float(value)
                    elif param == 'UseIntradayData':
                        portfolio_data['use_intraday_data'] = bool(value)
                    elif param == 'CalculateGreeks':
                        portfolio_data['calculate_greeks'] = bool(value)
                    elif param == 'EnableAdjustments':
                        portfolio_data['enable_adjustments'] = bool(value)
            
            # Parse Greek limits if present
            if 'GreekLimits' in df.columns:
                portfolio_data['greek_limits'] = self._parse_greek_limits(df)
                
            # Parse market filters if present
            if 'MarketFilters' in df.columns:
                portfolio_data['market_regime_filter'] = self._parse_market_filters(df)
                
            # Validate required fields
            required_fields = ['portfolio_name', 'strategy_name', 'start_date', 'end_date']
            for field in required_fields:
                if field not in portfolio_data:
                    self.errors.append(f"Missing required portfolio field: {field}")
                    
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error parsing portfolio Excel: {str(e)}")
            self.errors.append(f"Portfolio parsing error: {str(e)}")
            return {}
    
    def parse_strategy_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse strategy leg configurations from Excel"""
        try:
            # Check for LegParameter sheet
            excel_file = pd.ExcelFile(excel_path)
            sheet_name = 'LegParameter' if 'LegParameter' in excel_file.sheet_names else 'StrategySetting'
            
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            legs = []
            
            # Parse each row as a leg
            for idx, row in df.iterrows():
                if pd.notna(row.get('LegID')) or pd.notna(row.get('Leg_ID')):
                    leg_data = self._parse_leg_row(row, idx)
                    if leg_data:
                        legs.append(leg_data)
                        
            # Validate leg count
            if len(legs) == 0:
                self.errors.append("No valid legs found in strategy file")
            elif len(legs) > MAX_LEGS:
                self.errors.append(ERROR_MESSAGES["INVALID_LEG_COUNT"].format(max_legs=MAX_LEGS))
                
            # Parse strategy-level settings if present
            strategy_settings = {}
            if 'StrategySettings' in excel_file.sheet_names:
                settings_df = pd.read_excel(excel_path, sheet_name='StrategySettings')
                strategy_settings = self._parse_strategy_settings(settings_df)
                
            return {
                "legs": legs,
                "settings": strategy_settings
            }
            
        except Exception as e:
            logger.error(f"Error parsing strategy Excel: {str(e)}")
            self.errors.append(f"Strategy parsing error: {str(e)}")
            return {"legs": [], "settings": {}}
    
    def _parse_leg_row(self, row: pd.Series, idx: int) -> Optional[Dict[str, Any]]:
        """Parse a single leg configuration from Excel row"""
        try:
            leg_data = {
                "leg_id": int(row.get('LegID', row.get('Leg_ID', idx + 1))),
                "leg_name": str(row.get('LegName', row.get('Leg_Name', f'Leg_{idx + 1}'))),
                "option_type": self._parse_option_type(row),
                "position_type": self._parse_position_type(row),
                "strike_selection": self._parse_strike_selection(row),
                "lots": int(row.get('Lots', row.get('Quantity', 1))),
                "lot_size": int(row.get('LotSize', row.get('Lot_Size', 50))),
                "entry_time": self._parse_time(row.get('EntryTime', row.get('Entry_Time', DEFAULT_ENTRY_TIME))),
                "exit_time": self._parse_time(row.get('ExitTime', row.get('Exit_Time', DEFAULT_EXIT_TIME)))
            }
            
            # Parse optional fields
            if pd.notna(row.get('StrikeOffset', row.get('Strike_Offset'))):
                leg_data['strike_offset'] = float(row.get('StrikeOffset', row.get('Strike_Offset')))
            if pd.notna(row.get('StrikePrice', row.get('Strike_Price'))):
                leg_data['strike_price'] = float(row.get('StrikePrice', row.get('Strike_Price')))
            if pd.notna(row.get('DeltaTarget', row.get('Delta_Target'))):
                leg_data['delta_target'] = float(row.get('DeltaTarget', row.get('Delta_Target')))
            if pd.notna(row.get('StopLoss', row.get('Stop_Loss'))):
                leg_data['stop_loss'] = float(row.get('StopLoss', row.get('Stop_Loss')))
            if pd.notna(row.get('TakeProfit', row.get('Take_Profit'))):
                leg_data['take_profit'] = float(row.get('TakeProfit', row.get('Take_Profit')))
            if pd.notna(row.get('TrailingStop', row.get('Trailing_Stop'))):
                leg_data['trailing_stop'] = float(row.get('TrailingStop', row.get('Trailing_Stop')))
                
            # Parse conditions
            if pd.notna(row.get('EntryConditions', row.get('Entry_Conditions'))):
                leg_data['entry_conditions'] = str(row.get('EntryConditions', row.get('Entry_Conditions'))).split(';')
            if pd.notna(row.get('ExitConditions', row.get('Exit_Conditions'))):
                leg_data['exit_conditions'] = str(row.get('ExitConditions', row.get('Exit_Conditions'))).split(';')
                
            return leg_data
            
        except Exception as e:
            logger.error(f"Error parsing leg row {idx}: {str(e)}")
            self.errors.append(f"Leg {idx + 1} parsing error: {str(e)}")
            return None
    
    def _parse_option_type(self, row: pd.Series) -> str:
        """Parse option type from various formats"""
        option_type = str(row.get('OptionType', row.get('Option_Type', row.get('Type', '')))).upper()
        
        if option_type in ['CALL', 'CE', 'C']:
            return OptionType.CALL
        elif option_type in ['PUT', 'PE', 'P']:
            return OptionType.PUT
        else:
            raise ValueError(f"Invalid option type: {option_type}")
    
    def _parse_position_type(self, row: pd.Series) -> str:
        """Parse position type from various formats"""
        position_type = str(row.get('PositionType', row.get('Position_Type', row.get('Action', '')))).upper()
        
        if position_type in ['BUY', 'LONG', 'B']:
            return PositionType.BUY
        elif position_type in ['SELL', 'SHORT', 'S']:
            return PositionType.SELL
        else:
            raise ValueError(f"Invalid position type: {position_type}")
    
    def _parse_strike_selection(self, row: pd.Series) -> str:
        """Parse strike selection method"""
        strike_selection = str(row.get('StrikeSelection', row.get('Strike_Selection', 'ATM'))).upper()
        
        if strike_selection in ['ATM', 'AT_THE_MONEY']:
            return StrikeSelection.ATM
        elif strike_selection in ['ITM', 'IN_THE_MONEY']:
            return StrikeSelection.ITM
        elif strike_selection in ['OTM', 'OUT_THE_MONEY']:
            return StrikeSelection.OTM
        elif strike_selection in ['STRIKE_PRICE', 'FIXED']:
            return StrikeSelection.STRIKE_PRICE
        elif strike_selection in ['DELTA_BASED', 'DELTA']:
            return StrikeSelection.DELTA_BASED
        elif strike_selection in ['PERCENTAGE_BASED', 'PERCENTAGE']:
            return StrikeSelection.PERCENTAGE_BASED
        else:
            self.warnings.append(f"Unknown strike selection: {strike_selection}, defaulting to ATM")
            return StrikeSelection.ATM
    
    def _parse_time(self, time_value: Any) -> time:
        """Parse time from various formats"""
        if isinstance(time_value, time):
            return time_value
        elif isinstance(time_value, str):
            try:
                return datetime.strptime(time_value, "%H:%M:%S").time()
            except:
                try:
                    return datetime.strptime(time_value, "%H:%M").time()
                except:
                    logger.warning(f"Invalid time format: {time_value}, using default")
                    return time(9, 20)
        else:
            return time(9, 20)
    
    def _parse_greek_limits(self, df: pd.DataFrame) -> GreekLimits:
        """Parse Greek limits from portfolio settings"""
        limits = {}
        greek_params = ['max_delta', 'min_delta', 'max_gamma', 'min_gamma', 
                       'max_theta', 'min_theta', 'max_vega', 'min_vega']
        
        for param in greek_params:
            value = self._get_parameter_value(df, param.replace('_', ' ').title())
            if value is not None:
                limits[param] = float(value)
                
        return GreekLimits(**limits)
    
    def _parse_market_filters(self, df: pd.DataFrame) -> MarketRegimeFilter:
        """Parse market regime filters from portfolio settings"""
        # Parse VIX filter
        vix_min = self._get_parameter_value(df, 'VIX Min')
        vix_max = self._get_parameter_value(df, 'VIX Max')
        
        vix_filter = None
        if vix_min is not None or vix_max is not None:
            vix_filter = VIXFilter(
                min_vix=float(vix_min) if vix_min else None,
                max_vix=float(vix_max) if vix_max else None
            )
            
        # Parse other filters
        trend_filter = self._get_parameter_value(df, 'Trend Filter')
        volatility_regime = self._get_parameter_value(df, 'Volatility Regime')
        
        return MarketRegimeFilter(
            vix_filter=vix_filter,
            trend_filter=trend_filter,
            volatility_regime=volatility_regime
        )
    
    def _get_parameter_value(self, df: pd.DataFrame, parameter: str) -> Any:
        """Get parameter value from dataframe"""
        mask = df['Parameter'].str.contains(parameter, case=False, na=False)
        if mask.any():
            return df.loc[mask, 'Value'].iloc[0]
        return None
    
    def parse_adjustment_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse adjustment rules from Excel"""
        try:
            df = pd.read_excel(excel_path, sheet_name='AdjustmentRules')
            
            adjustments = []
            
            for idx, row in df.iterrows():
                if pd.notna(row.get('RuleID')):
                    rule = AdjustmentRule(
                        rule_id=str(row['RuleID']),
                        trigger_type=AdjustmentTrigger(row['TriggerType']),
                        trigger_condition=str(row['TriggerCondition']),
                        trigger_value=float(row['TriggerValue']),
                        action_type=AdjustmentAction(row['ActionType']),
                        action_params=eval(row['ActionParams']) if pd.notna(row.get('ActionParams')) else {},
                        max_adjustments=int(row.get('MaxAdjustments', 3)),
                        cooldown_period=int(row.get('CooldownPeriod', 60)),
                        priority=int(row.get('Priority', 1))
                    )
                    adjustments.append(rule)
                    
            return {"rules": adjustments}
            
        except Exception as e:
            logger.error(f"Error parsing adjustment Excel: {str(e)}")
            self.warnings.append(f"Adjustment parsing warning: {str(e)}")
            return {"rules": []}
    
    def _create_strategy_model(self, parsed_data: Dict[str, Any]) -> POSStrategyModel:
        """Create validated strategy model from parsed data"""
        # Create portfolio model
        portfolio = POSPortfolioModel(**parsed_data['portfolio'])
        
        # Create leg models
        legs = []
        for leg_data in parsed_data['strategy']['legs']:
            leg = POSLegModel(**leg_data)
            
            # Add adjustment rules if present
            if parsed_data.get('adjustments', {}).get('rules'):
                leg_rules = [rule for rule in parsed_data['adjustments']['rules'] 
                           if rule.rule_id.startswith(f"L{leg.leg_id}")]
                leg.adjustment_rules = leg_rules
                
            legs.append(leg)
            
        # Create strategy model
        strategy = POSStrategyModel(
            portfolio=portfolio,
            legs=legs,
            **parsed_data['strategy']['settings']
        )
        
        return strategy
    
    def validate_parsed_data(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Validate parsed data for consistency and completeness"""
        validation_errors = []
        
        # Validate date range
        if parsed_data['portfolio'].get('start_date') >= parsed_data['portfolio'].get('end_date'):
            validation_errors.append("End date must be after start date")
            
        # Validate legs
        leg_ids = set()
        for leg in parsed_data['strategy']['legs']:
            if leg['leg_id'] in leg_ids:
                validation_errors.append(f"Duplicate leg ID: {leg['leg_id']}")
            leg_ids.add(leg['leg_id'])
            
            # Validate strike selection requirements
            if leg['strike_selection'] == StrikeSelection.STRIKE_PRICE and 'strike_price' not in leg:
                validation_errors.append(f"Leg {leg['leg_id']}: strike_price required for STRIKE_PRICE selection")
            elif leg['strike_selection'] == StrikeSelection.DELTA_BASED and 'delta_target' not in leg:
                validation_errors.append(f"Leg {leg['leg_id']}: delta_target required for DELTA_BASED selection")
                
        return validation_errors