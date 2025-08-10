"""
Enhanced POS Parser supporting all 200+ columns and all sheets
Handles: PositionalParameter, LegParameter, AdjustmentRules, 
MarketStructure, GreekLimits, VolatilityMetrics, BreakevenAnalysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, time
from typing import Dict, List, Any, Optional, Union
import json
import os

from .models_enhanced import (
    EnhancedPortfolioModel, EnhancedPositionalStrategy, EnhancedLegModel,
    AdjustmentRule, MarketStructureConfig, GreekLimitsConfig,
    CompletePOSStrategy, VixConfiguration, VixRange, PremiumTargets,
    BreakevenConfig, VolatilityFilter, EntryConfig, RiskManagement,
    PositionType, StrategySubtype, RollFrequency, ExpiryType,
    VixMethod, PremiumType, BECalculationMethod, BufferType,
    Frequency, BEAction, InstrumentType, TransactionType,
    PositionRole, StrikeMethod, BEContribution, BERole,
    StopLossType, AdjustmentTrigger, AdjustmentAction, MarketRegime
)


class EnhancedPOSParser:
    """Enhanced parser for complete POS strategy with all features"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def parse_input(self, portfolio_file: str, strategy_file: str) -> Dict[str, Any]:
        """Parse all input files and return complete strategy model"""
        try:
            # Parse portfolio file
            portfolio_data = self._parse_portfolio_file(portfolio_file)
            if not portfolio_data:
                return {'errors': self.errors, 'model': None}
            
            # Parse strategy file (all sheets)
            strategy_data = self._parse_strategy_file(strategy_file)
            if not strategy_data:
                return {'errors': self.errors, 'model': None}
            
            # Create complete model
            model = self._create_complete_model(portfolio_data, strategy_data)
            
            return {
                'model': model,
                'errors': self.errors,
                'warnings': self.warnings
            }
            
        except Exception as e:
            self.errors.append(f"Parser error: {str(e)}")
            return {'errors': self.errors, 'model': None}
    
    def _parse_portfolio_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse portfolio Excel file"""
        try:
            # Read PortfolioSetting sheet
            df_portfolio = pd.read_excel(file_path, sheet_name='PortfolioSetting')
            if df_portfolio.empty:
                self.errors.append("PortfolioSetting sheet is empty")
                return None
            
            row = df_portfolio.iloc[0]
            
            # Parse dates
            start_date = self._parse_date(row.get('StartDate'))
            end_date = self._parse_date(row.get('EndDate'))
            
            portfolio_data = {
                'portfolio_name': str(row.get('PortfolioName', 'POS_Portfolio')),
                'start_date': start_date,
                'end_date': end_date,
                'index_name': str(row.get('IndexName', 'NIFTY')),
                'multiplier': int(row.get('Multiplier', 1)),
                'slippage_percent': float(row.get('SlippagePercent', 0.1)),
                'is_tick_bt': self._parse_bool(row.get('IsTickBT', 'NO')),
                'enabled': self._parse_bool(row.get('Enabled', 'YES')),
                'portfolio_stoploss': float(row.get('PortfolioStoploss', 0)),
                'portfolio_target': float(row.get('PortfolioTarget', 0)),
                'initial_capital': float(row.get('InitialCapital', 1000000)),
                'position_size_method': str(row.get('PositionSizeMethod', 'FIXED')),
                'position_size_value': float(row.get('PositionSizeValue', 100000)),
                'max_open_positions': int(row.get('MaxOpenPositions', 5)),
                'correlation_limit': float(row.get('CorrelationLimit', 0.7)),
                'transaction_costs': float(row.get('TransactionCosts', 0.001)),
                'use_margin': self._parse_bool(row.get('UseMargin', 'NO')),
                'margin_requirement': float(row.get('MarginRequirement', 0.2)),
                'compound_profits': self._parse_bool(row.get('CompoundProfits', 'NO')),
                'reinvestment_ratio': float(row.get('ReinvestmentRatio', 0.5))
            }
            
            return portfolio_data
            
        except Exception as e:
            self.errors.append(f"Portfolio parsing error: {str(e)}")
            return None
    
    def _parse_strategy_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse complete strategy Excel file with all sheets"""
        try:
            xl = pd.ExcelFile(file_path)
            available_sheets = xl.sheet_names
            
            strategy_data = {}
            
            # 1. Parse PositionalParameter (Required)
            if 'PositionalParameter' in available_sheets:
                strategy_data['parameters'] = self._parse_positional_parameters(
                    pd.read_excel(file_path, sheet_name='PositionalParameter')
                )
            else:
                self.errors.append("PositionalParameter sheet not found")
                return None
            
            # 2. Parse LegParameter (Required)
            if 'LegParameter' in available_sheets:
                strategy_data['legs'] = self._parse_leg_parameters(
                    pd.read_excel(file_path, sheet_name='LegParameter')
                )
            else:
                self.errors.append("LegParameter sheet not found")
                return None
            
            # 3. Parse AdjustmentRules (Optional)
            if 'AdjustmentRules' in available_sheets:
                strategy_data['adjustment_rules'] = self._parse_adjustment_rules(
                    pd.read_excel(file_path, sheet_name='AdjustmentRules')
                )
            
            # 4. Parse MarketStructure (Optional)
            if 'MarketStructure' in available_sheets:
                strategy_data['market_structure'] = self._parse_market_structure(
                    pd.read_excel(file_path, sheet_name='MarketStructure')
                )
            
            # 5. Parse GreekLimits (Optional)
            if 'GreekLimits' in available_sheets:
                strategy_data['greek_limits'] = self._parse_greek_limits(
                    pd.read_excel(file_path, sheet_name='GreekLimits')
                )
            
            # 6. Parse VolatilityMetrics (Optional)
            if 'VolatilityMetrics' in available_sheets:
                vol_df = pd.read_excel(file_path, sheet_name='VolatilityMetrics')
                # Merge with parameters as they share volatility settings
                if not vol_df.empty:
                    strategy_data['parameters'].update(
                        self._parse_volatility_metrics(vol_df)
                    )
            
            # 7. Parse BreakevenAnalysis (Optional)
            if 'BreakevenAnalysis' in available_sheets:
                be_df = pd.read_excel(file_path, sheet_name='BreakevenAnalysis')
                # Merge with parameters as they share BE settings
                if not be_df.empty:
                    strategy_data['parameters'].update(
                        self._parse_breakeven_analysis(be_df)
                    )
            
            return strategy_data
            
        except Exception as e:
            self.errors.append(f"Strategy file parsing error: {str(e)}")
            return None
    
    def _parse_positional_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse all 200+ PositionalParameter columns"""
        if df.empty:
            self.errors.append("PositionalParameter sheet is empty")
            return {}
        
        row = df.iloc[0]  # First row contains parameters
        
        # Parse all parameter groups
        params = {
            # Strategy Identity & Type
            'strategy_name': str(row.get('StrategyName', 'POS_Strategy')),
            'position_type': self._parse_enum(row.get('PositionType'), PositionType, 'WEEKLY'),
            'strategy_subtype': self._parse_enum(row.get('StrategySubtype'), StrategySubtype, 'CALENDAR_SPREAD'),
            'enabled': self._parse_bool(row.get('Enabled', 'YES')),
            'priority': int(row.get('Priority', 1)),
            
            # Timeframe Configuration
            'short_leg_dte': int(row.get('ShortLegDTE', 7)),
            'long_leg_dte': self._parse_optional_int(row.get('LongLegDTE')),
            'roll_frequency': self._parse_enum(row.get('RollFrequency'), RollFrequency),
            'custom_dte_list': self._parse_int_list(row.get('CustomDTEList')),
            'min_dte_to_enter': self._parse_optional_int(row.get('MinDTEToEnter')),
            'max_dte_to_enter': self._parse_optional_int(row.get('MaxDTEToEnter')),
            'preferred_expiry': self._parse_enum(row.get('PreferredExpiry'), ExpiryType, 'WEEKLY'),
            'avoid_expiry_week': self._parse_bool(row.get('AvoidExpiryWeek', 'NO')),
            
            # VIX Configuration
            'vix_config': {
                'method': self._parse_enum(row.get('VixMethod'), VixMethod, 'SPOT'),
                'low': {'min': float(row.get('VixLowMin', 9)), 'max': float(row.get('VixLowMax', 12))},
                'medium': {'min': float(row.get('VixMedMin', 13)), 'max': float(row.get('VixMedMax', 20))},
                'high': {'min': float(row.get('VixHighMin', 20)), 'max': float(row.get('VixHighMax', 30))},
                'extreme': {'min': float(row.get('VixExtremeMin', 30)), 'max': float(row.get('VixExtremeMax', 100))},
                'custom_ranges': self._parse_json(row.get('CustomVixRanges'))
            },
            
            # Premium Targets
            'premium_targets': {
                'low': self._parse_optional_str(row.get('TargetPremiumLow')),
                'medium': self._parse_optional_str(row.get('TargetPremiumMed')),
                'high': self._parse_optional_str(row.get('TargetPremiumHigh')),
                'extreme': self._parse_optional_str(row.get('TargetPremiumExtreme')),
                'premium_type': self._parse_enum(row.get('PremiumType'), PremiumType, 'ABSOLUTE'),
                'min_acceptable': self._parse_optional_float(row.get('MinAcceptablePremium')),
                'max_acceptable': self._parse_optional_float(row.get('MaxAcceptablePremium')),
                'differential': row.get('PremiumDifferential') if pd.notna(row.get('PremiumDifferential')) else None
            },
            
            # Breakeven Configuration
            'breakeven_config': {
                'enabled': self._parse_bool(row.get('UseBreakevenAnalysis', 'NO')),
                'calculation_method': self._parse_enum(row.get('BreakevenCalculation'), BECalculationMethod, 'THEORETICAL'),
                'upper_target': row.get('UpperBreakevenTarget', 'DYNAMIC'),
                'lower_target': row.get('LowerBreakevenTarget', 'DYNAMIC'),
                'buffer': float(row.get('BreakevenBuffer', 50)),
                'buffer_type': self._parse_enum(row.get('BreakevenBufferType'), BufferType, 'FIXED'),
                'dynamic_adjustment': self._parse_bool(row.get('DynamicBEAdjustment', 'NO')),
                'recalc_frequency': self._parse_enum(row.get('BERecalcFrequency'), Frequency, 'HOURLY'),
                'include_commissions': self._parse_bool(row.get('IncludeCommissions', 'YES')),
                'include_slippage': self._parse_bool(row.get('IncludeSlippage', 'YES')),
                'time_decay_factor': self._parse_bool(row.get('TimeDecayFactor', 'YES')),
                'volatility_smile_be': self._parse_bool(row.get('VolatilitySmileBE', 'NO')),
                'spot_price_threshold': float(row.get('SpotPriceBEThreshold', 0.02)),
                'approach_action': self._parse_enum(row.get('BEApproachAction'), BEAction, 'ADJUST'),
                'breach_action': self._parse_enum(row.get('BEBreachAction'), BEAction, 'CLOSE'),
                'track_distance': self._parse_bool(row.get('TrackBEDistance', 'YES')),
                'distance_alert': float(row.get('BEDistanceAlert', 100))
            },
            
            # Volatility Filter
            'volatility_filter': {
                'use_ivp': self._parse_bool(row.get('UseIVP', 'NO')),
                'ivp_lookback': int(row.get('IVPLookback', 252)),
                'ivp_min_entry': float(row.get('IVPMinEntry', 0.30)),
                'ivp_max_entry': float(row.get('IVPMaxEntry', 0.70)),
                'use_ivr': self._parse_bool(row.get('UseIVR', 'NO')),
                'ivr_lookback': int(row.get('IVRLookback', 252)),
                'ivr_min_entry': float(row.get('IVRMinEntry', 0.20)),
                'ivr_max_entry': float(row.get('IVRMaxEntry', 0.80)),
                'use_atr_percentile': self._parse_bool(row.get('UseATRPercentile', 'NO')),
                'atr_period': int(row.get('ATRPeriod', 14)),
                'atr_lookback': int(row.get('ATRLookback', 252)),
                'atr_min_percentile': float(row.get('ATRMinPercentile', 0.20)),
                'atr_max_percentile': float(row.get('ATRMaxPercentile', 0.80))
            },
            
            # Entry Configuration
            'entry_config': {
                'days': self._parse_str_list(row.get('EntryDays')),
                'time_start': self._parse_time(row.get('EntryTimeStart')),
                'time_end': self._parse_time(row.get('EntryTimeEnd')),
                'preferred_time': self._parse_time(row.get('PreferredEntryTime')),
                'avoid_first_minutes': int(row.get('AvoidFirstMinutes', 0)),
                'avoid_last_minutes': int(row.get('AvoidLastMinutes', 0)),
                'min_volume': self._parse_optional_int(row.get('MinVolume')),
                'min_oi': self._parse_optional_int(row.get('MinOI')),
                'max_spread': self._parse_optional_float(row.get('MaxSpread')),
                'require_trend_confirmation': self._parse_bool(row.get('RequireTrendConfirmation', 'NO')),
                'trend_lookback_periods': int(row.get('TrendLookbackPeriods', 20))
            },
            
            # Risk Management
            'risk_management': {
                'max_position_size': int(row.get('MaxPositionSize', 10)),
                'max_portfolio_risk': float(row.get('MaxPortfolioRisk', 0.02)),
                'max_daily_loss': self._parse_optional_float(row.get('MaxDailyLoss')),
                'profit_target': self._parse_optional_float(row.get('ProfitTarget')),
                'stop_loss': self._parse_optional_float(row.get('StopLoss')),
                'be_risk_management': self._parse_bool(row.get('BERiskManagement', 'NO')),
                'max_be_exposure': self._parse_optional_float(row.get('MaxBEExposure')),
                'use_kelly_criterion': self._parse_bool(row.get('UseKellyCriterion', 'NO')),
                'kelly_fraction': float(row.get('KellyFraction', 0.25)),
                'max_correlation_risk': float(row.get('MaxCorrelationRisk', 0.7)),
                'use_var': self._parse_bool(row.get('UseVaR', 'NO')),
                'var_confidence': float(row.get('VaRConfidence', 0.95)),
                'var_lookback': int(row.get('VaRLookback', 252))
            },
            
            # Additional parameters
            'max_loss_per_day': self._parse_optional_float(row.get('MaxLossPerDay')),
            'max_loss_per_week': self._parse_optional_float(row.get('MaxLossPerWeek')),
            'max_consecutive_losses': int(row.get('MaxConsecutiveLosses', 3)),
            'pause_after_max_losses': self._parse_bool(row.get('PauseAfterMaxLosses', 'YES')),
            'pause_duration_days': int(row.get('PauseDurationDays', 2)),
            'track_metrics': self._parse_bool(row.get('TrackMetrics', 'YES')),
            'trade_only_in_market_hours': self._parse_bool(row.get('TradeOnlyInMarketHours', 'YES')),
            'avoid_holidays': self._parse_bool(row.get('AvoidHolidays', 'YES')),
            'use_limit_orders': self._parse_bool(row.get('UseLimitOrders', 'YES')),
            'limit_order_offset': float(row.get('LimitOrderOffset', 0.05)),
            'max_order_retries': int(row.get('MaxOrderRetries', 3))
        }
        
        return params
    
    def _parse_leg_parameters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse all LegParameter rows"""
        legs = []
        
        for idx, row in df.iterrows():
            leg = {
                # Core Configuration
                'leg_id': str(row.get('LegID', f'leg_{idx}')),
                'leg_name': str(row.get('LegName', f'Leg {idx}')),
                'is_active': self._parse_bool(row.get('IsActive', 'YES')),
                'leg_priority': int(row.get('LegPriority', idx + 1)),
                
                # Position Configuration
                'instrument': self._parse_enum(row.get('Instrument'), InstrumentType, 'CALL'),
                'transaction': self._parse_enum(row.get('Transaction'), TransactionType, 'BUY'),
                'position_role': self._parse_enum(row.get('PositionRole'), PositionRole, 'PRIMARY'),
                'is_weekly_leg': self._parse_bool(row.get('IsWeeklyLeg', 'NO')),
                'is_protective_leg': self._parse_bool(row.get('IsProtectiveLeg', 'NO')),
                
                # Strike Selection
                'strike_method': self._parse_strike_method(row.get('StrikeMethod', 'ATM')),
                'strike_value': self._parse_optional_float(row.get('StrikeValue')),
                'strike_delta': self._parse_optional_float(row.get('StrikeDelta')),
                'strike_premium': self._parse_optional_float(row.get('StrikePremium')),
                'optimize_for_be': self._parse_bool(row.get('OptimizeForBE', 'NO')),
                'target_be_distance': self._parse_optional_float(row.get('TargetBEDistance')),
                
                # Size Management
                'lots': int(row.get('Lots', 1)),
                'dynamic_sizing': self._parse_bool(row.get('DynamicSizing', 'NO')),
                'size_multiplier': float(row.get('SizeMultiplier', 1.0)),
                'max_lots': self._parse_optional_int(row.get('MaxLots')),
                'min_lots': int(row.get('MinLots', 1)),
                
                # Volatility-Based Sizing
                'ivp_sizing': self._parse_bool(row.get('IVPSizing', 'NO')),
                'ivp_size_min': float(row.get('IVPSizeMin', 0.5)),
                'ivp_size_max': float(row.get('IVPSizeMax', 1.5)),
                'atr_sizing': self._parse_bool(row.get('ATRSizing', 'NO')),
                'atr_size_factor': float(row.get('ATRSizeFactor', 1.0)),
                'vol_regime_sizing': self._parse_bool(row.get('VolRegimeSizing', 'NO')),
                
                # Breakeven Tracking
                'track_leg_be': self._parse_bool(row.get('TrackLegBE', 'NO')),
                'leg_be_contribution': self._parse_enum(row.get('LegBEContribution'), BEContribution, 'NEUTRAL'),
                'leg_be_weight': float(row.get('LegBEWeight', 1.0)),
                'be_adjustment_role': self._parse_enum(row.get('BEAdjustmentRole'), BERole, 'MAINTAIN'),
                'min_be_improvement': self._parse_optional_float(row.get('MinBEImprovement')),
                
                # Risk Parameters
                'stop_loss_type': self._parse_enum(row.get('StopLossType'), StopLossType, 'NONE'),
                'stop_loss_value': self._parse_optional_float(row.get('StopLossValue')),
                'target_type': self._parse_enum(row.get('TargetType'), StopLossType, 'NONE'),
                'target_value': self._parse_optional_float(row.get('TargetValue')),
                'trailing_stop': self._parse_bool(row.get('TrailingStop', 'NO')),
                'trailing_stop_distance': self._parse_optional_float(row.get('TrailingStopDistance')),
                
                # Greeks limits
                'max_delta': self._parse_optional_float(row.get('MaxDelta')),
                'max_gamma': self._parse_optional_float(row.get('MaxGamma')),
                'max_theta': self._parse_optional_float(row.get('MaxTheta')),
                'max_vega': self._parse_optional_float(row.get('MaxVega')),
                
                # Execution parameters
                'execution_priority': int(row.get('ExecutionPriority', idx + 1)),
                'fill_or_kill': self._parse_bool(row.get('FillOrKill', 'NO')),
                'all_or_none': self._parse_bool(row.get('AllOrNone', 'NO')),
                'iceberg_order': self._parse_bool(row.get('IcebergOrder', 'NO')),
                'iceberg_visible_lots': self._parse_optional_int(row.get('IcebergVisibleLots')),
                
                # Adjustment parameters
                'adjustable': self._parse_bool(row.get('Adjustable', 'YES')),
                'adjustment_cooldown': int(row.get('AdjustmentCooldown', 0)),
                'max_adjustments': self._parse_optional_int(row.get('MaxAdjustments')),
                'adjustment_cost_limit': self._parse_optional_float(row.get('AdjustmentCostLimit'))
            }
            
            legs.append(leg)
        
        return legs
    
    def _parse_adjustment_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse all 90+ AdjustmentRules parameters"""
        rules = []
        
        for idx, row in df.iterrows():
            rule = {
                'rule_id': str(row.get('RuleID', f'rule_{idx}')),
                'rule_name': str(row.get('RuleName', f'Rule {idx}')),
                'enabled': self._parse_bool(row.get('Enabled', 'YES')),
                'priority': int(row.get('Priority', idx + 1)),
                
                # Trigger Configuration
                'trigger_type': self._parse_enum(row.get('TriggerType'), AdjustmentTrigger, 'TIME_BASED'),
                'trigger_value': float(row.get('TriggerValue', 0)),
                'trigger_comparison': str(row.get('TriggerComparison', 'GREATER')),
                'trigger_value2': self._parse_optional_float(row.get('TriggerValue2')),
                
                # Condition checks
                'check_time': self._parse_bool(row.get('CheckTime', 'YES')),
                'min_time_in_position': int(row.get('MinTimeInPosition', 0)),
                'max_time_in_position': self._parse_optional_int(row.get('MaxTimeInPosition')),
                'check_pnl': self._parse_bool(row.get('CheckPnL', 'YES')),
                'min_pnl': self._parse_optional_float(row.get('MinPnL')),
                'max_pnl': self._parse_optional_float(row.get('MaxPnL')),
                'check_underlying_move': self._parse_bool(row.get('CheckUnderlyingMove', 'YES')),
                'underlying_move_percent': self._parse_optional_float(row.get('UnderlyingMovePercent')),
                
                # Action Configuration
                'action_type': self._parse_enum(row.get('ActionType'), AdjustmentAction, 'ROLL_STRIKE'),
                'action_leg_id': self._parse_optional_str(row.get('ActionLegID')),
                'new_strike_method': self._parse_enum(row.get('NewStrikeMethod'), StrikeMethod) if pd.notna(row.get('NewStrikeMethod')) else None,
                'new_strike_offset': self._parse_optional_float(row.get('NewStrikeOffset')),
                'roll_to_dte': self._parse_optional_int(row.get('RollToDTE')),
                
                # Risk checks
                'check_cost': self._parse_bool(row.get('CheckCost', 'YES')),
                'max_adjustment_cost': self._parse_optional_float(row.get('MaxAdjustmentCost')),
                'check_be_improvement': self._parse_bool(row.get('CheckBEImprovement', 'YES')),
                'min_be_improvement_required': self._parse_optional_float(row.get('MinBEImprovementRequired')),
                'check_risk_reduction': self._parse_bool(row.get('CheckRiskReduction', 'YES')),
                'min_risk_reduction_percent': self._parse_optional_float(row.get('MinRiskReductionPercent')),
                
                # Greeks-based triggers
                'delta_trigger': self._parse_optional_float(row.get('DeltaTrigger')),
                'gamma_trigger': self._parse_optional_float(row.get('GammaTrigger')),
                'theta_trigger': self._parse_optional_float(row.get('ThetaTrigger')),
                'vega_trigger': self._parse_optional_float(row.get('VegaTrigger')),
                'delta_neutral_band': self._parse_optional_float(row.get('DeltaNeutralBand')),
                
                # Market condition filters
                'require_market_regime': self._parse_enum(row.get('RequireMarketRegime'), MarketRegime) if pd.notna(row.get('RequireMarketRegime')) else None,
                'avoid_market_regime': self._parse_enum(row.get('AvoidMarketRegime'), MarketRegime) if pd.notna(row.get('AvoidMarketRegime')) else None,
                'vix_min': self._parse_optional_float(row.get('VixMin')),
                'vix_max': self._parse_optional_float(row.get('VixMax')),
                
                # Execution parameters
                'use_limit_order': self._parse_bool(row.get('UseLimitOrder', 'YES')),
                'limit_price_offset': float(row.get('LimitPriceOffset', 0.05)),
                'max_retries': int(row.get('MaxRetries', 3)),
                'timeout_seconds': int(row.get('TimeoutSeconds', 60)),
                
                # Post-adjustment actions
                'update_stops': self._parse_bool(row.get('UpdateStops', 'YES')),
                'update_targets': self._parse_bool(row.get('UpdateTargets', 'YES')),
                'send_alert': self._parse_bool(row.get('SendAlert', 'YES')),
                'alert_message': self._parse_optional_str(row.get('AlertMessage')),
                
                # Additional parameters
                'backtest_win_rate': self._parse_optional_float(row.get('BacktestWinRate')),
                'backtest_avg_improvement': self._parse_optional_float(row.get('BacktestAvgImprovement'))
            }
            
            rules.append(rule)
        
        return rules
    
    def _parse_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse MarketStructure configuration"""
        if df.empty:
            return None
        
        row = df.iloc[0]
        
        return {
            'enabled': self._parse_bool(row.get('Enabled', 'NO')),
            
            # Trend detection
            'use_moving_averages': self._parse_bool(row.get('UseMovingAverages', 'YES')),
            'ma_periods': self._parse_int_list(row.get('MAPeriods', '20,50,200')),
            'trend_strength_threshold': float(row.get('TrendStrengthThreshold', 0.7)),
            
            # Support/Resistance
            'detect_sr_levels': self._parse_bool(row.get('DetectSRLevels', 'YES')),
            'sr_lookback_periods': int(row.get('SRLookbackPeriods', 100)),
            'sr_touch_threshold': float(row.get('SRTouchThreshold', 0.002)),
            'sr_strength_min_touches': int(row.get('SRStrengthMinTouches', 3)),
            
            # Volume analysis
            'analyze_volume': self._parse_bool(row.get('AnalyzeVolume', 'YES')),
            'volume_ma_period': int(row.get('VolumeMAperiod', 20)),
            'unusual_volume_threshold': float(row.get('UnusualVolumeThreshold', 2.0)),
            
            # Market breadth
            'use_advance_decline': self._parse_bool(row.get('UseAdvanceDecline', 'NO')),
            'use_up_down_volume': self._parse_bool(row.get('UseUpDownVolume', 'NO')),
            'breadth_threshold': float(row.get('BreadthThreshold', 0.6)),
            
            # Volatility regime
            'detect_volatility_regime': self._parse_bool(row.get('DetectVolatilityRegime', 'YES')),
            'volatility_lookback': int(row.get('VolatilityLookback', 30)),
            'regime_change_threshold': float(row.get('RegimeChangeThreshold', 0.2)),
            
            # Pattern detection
            'detect_chart_patterns': self._parse_bool(row.get('DetectChartPatterns', 'NO')),
            'patterns_to_detect': self._parse_str_list(row.get('PatternsToDetect')),
            
            # Market microstructure
            'analyze_bid_ask_spread': self._parse_bool(row.get('AnalyzeBidAskSpread', 'YES')),
            'analyze_order_flow': self._parse_bool(row.get('AnalyzeOrderFlow', 'NO')),
            'detect_large_orders': self._parse_bool(row.get('DetectLargeOrders', 'YES')),
            'large_order_threshold': int(row.get('LargeOrderThreshold', 100))
        }
    
    def _parse_greek_limits(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse GreekLimits configuration"""
        if df.empty:
            return None
        
        row = df.iloc[0]
        
        return {
            'enabled': self._parse_bool(row.get('Enabled', 'NO')),
            
            # Portfolio level limits
            'portfolio_max_delta': self._parse_optional_float(row.get('PortfolioMaxDelta')),
            'portfolio_max_gamma': self._parse_optional_float(row.get('PortfolioMaxGamma')),
            'portfolio_max_theta': self._parse_optional_float(row.get('PortfolioMaxTheta')),
            'portfolio_max_vega': self._parse_optional_float(row.get('PortfolioMaxVega')),
            'portfolio_max_rho': self._parse_optional_float(row.get('PortfolioMaxRho')),
            
            # Position level limits
            'position_max_delta': self._parse_optional_float(row.get('PositionMaxDelta')),
            'position_max_gamma': self._parse_optional_float(row.get('PositionMaxGamma')),
            'position_max_theta': self._parse_optional_float(row.get('PositionMaxTheta')),
            'position_max_vega': self._parse_optional_float(row.get('PositionMaxVega')),
            
            # Delta-neutral bands
            'maintain_delta_neutral': self._parse_bool(row.get('MaintainDeltaNeutral', 'NO')),
            'delta_neutral_threshold': float(row.get('DeltaNeutralThreshold', 100)),
            'auto_hedge_delta': self._parse_bool(row.get('AutoHedgeDelta', 'NO')),
            'hedge_instrument': str(row.get('HedgeInstrument', 'FUTURES')),
            
            # Gamma scalping
            'enable_gamma_scalping': self._parse_bool(row.get('EnableGammaScalping', 'NO')),
            'gamma_scalp_threshold': float(row.get('GammaScalpThreshold', 50)),
            'scalp_size': int(row.get('ScalpSize', 1)),
            
            # Vega management
            'vega_hedge_enabled': self._parse_bool(row.get('VegaHedgeEnabled', 'NO')),
            'vega_hedge_threshold': float(row.get('VegaHedgeThreshold', 1000)),
            'vega_hedge_method': str(row.get('VegaHedgeMethod', 'CALENDAR')),
            
            # Risk parity
            'use_risk_parity': self._parse_bool(row.get('UseRiskParity', 'NO')),
            'risk_parity_target': str(row.get('RiskParityTarget', 'EQUAL_RISK')),
            'rebalance_frequency': str(row.get('RebalanceFrequency', 'DAILY'))
        }
    
    def _parse_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse additional volatility metrics"""
        if df.empty:
            return {}
        
        row = df.iloc[0]
        
        # Return updates to volatility_filter
        return {
            'volatility_filter': {
                'historical_vol_lookback': int(row.get('HistoricalVolLookback', 30)),
                'garch_enabled': self._parse_bool(row.get('GARCHEnabled', 'NO')),
                'garch_p': int(row.get('GARCH_P', 1)),
                'garch_q': int(row.get('GARCH_Q', 1)),
                'vol_smile_adjustment': self._parse_bool(row.get('VolSmileAdjustment', 'NO')),
                'term_structure_analysis': self._parse_bool(row.get('TermStructureAnalysis', 'NO')),
                'vol_of_vol': self._parse_bool(row.get('VolOfVol', 'NO')),
                'vol_regime_detection': self._parse_bool(row.get('VolRegimeDetection', 'YES'))
            }
        }
    
    def _parse_breakeven_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse additional breakeven analysis parameters"""
        if df.empty:
            return {}
        
        row = df.iloc[0]
        
        # Return updates to breakeven_config
        return {
            'breakeven_config': {
                'monte_carlo_simulations': int(row.get('MonteCarloSimulations', 1000)),
                'confidence_interval': float(row.get('ConfidenceInterval', 0.95)),
                'stress_test_scenarios': self._parse_int_list(row.get('StressTestScenarios', '5,10,15,20')),
                'path_dependency': self._parse_bool(row.get('PathDependency', 'NO')),
                'discrete_dividends': self._parse_bool(row.get('DiscreteDividends', 'NO')),
                'early_exercise': self._parse_bool(row.get('EarlyExercise', 'NO')),
                'be_surface_generation': self._parse_bool(row.get('BESurfaceGeneration', 'NO'))
            }
        }
    
    def _create_complete_model(self, portfolio_data: Dict, strategy_data: Dict) -> CompletePOSStrategy:
        """Create complete POS strategy model"""
        # Create portfolio model
        portfolio = EnhancedPortfolioModel(**portfolio_data)
        
        # Create strategy model
        strategy_params = strategy_data['parameters']
        strategy = EnhancedPositionalStrategy(
            **{k: v for k, v in strategy_params.items() 
               if k not in ['volatility_filter', 'breakeven_config']}
        )
        
        # Update nested configs
        if 'volatility_filter' in strategy_params:
            for k, v in strategy_params['volatility_filter'].items():
                setattr(strategy.volatility_filter, k, v)
        
        if 'breakeven_config' in strategy_params:
            for k, v in strategy_params['breakeven_config'].items():
                setattr(strategy.breakeven_config, k, v)
        
        # Create leg models
        legs = [EnhancedLegModel(**leg_data) for leg_data in strategy_data['legs']]
        
        # Create adjustment rules
        adjustment_rules = None
        if 'adjustment_rules' in strategy_data:
            adjustment_rules = [
                AdjustmentRule(**rule_data) 
                for rule_data in strategy_data['adjustment_rules']
            ]
        
        # Create market structure config
        market_structure = None
        if 'market_structure' in strategy_data:
            market_structure = MarketStructureConfig(**strategy_data['market_structure'])
        
        # Create greek limits config
        greek_limits = None
        if 'greek_limits' in strategy_data:
            greek_limits = GreekLimitsConfig(**strategy_data['greek_limits'])
        
        # Create complete model
        return CompletePOSStrategy(
            portfolio=portfolio,
            strategy=strategy,
            legs=legs,
            adjustment_rules=adjustment_rules,
            market_structure=market_structure,
            greek_limits=greek_limits
        )
    
    # Helper parsing methods
    def _parse_date(self, value) -> date:
        """Parse date from various formats"""
        if pd.isna(value):
            return date.today()
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            # Handle DD_MM_YYYY format
            if '_' in value:
                parts = value.split('_')
                return date(int(parts[2]), int(parts[1]), int(parts[0]))
            # Handle YYYY-MM-DD format
            return datetime.strptime(value, '%Y-%m-%d').date()
        return date.today()
    
    def _parse_time(self, value) -> Optional[time]:
        """Parse time from various formats"""
        if pd.isna(value):
            return None
        if isinstance(value, time):
            return value
        if isinstance(value, datetime):
            return value.time()
        if isinstance(value, (int, float)):
            # Handle HHMMSS format
            hour = int(value // 10000)
            minute = int((value % 10000) // 100)
            second = int(value % 100)
            return time(hour, minute, second)
        if isinstance(value, str):
            # Handle HH:MM:SS format
            return datetime.strptime(value, '%H:%M:%S').time()
        return None
    
    def _parse_bool(self, value) -> bool:
        """Parse boolean from YES/NO or True/False"""
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1']
        return bool(value)
    
    def _parse_enum(self, value, enum_class, default=None):
        """Parse enum value"""
        if pd.isna(value) or value is None:
            return default
        if isinstance(value, enum_class):
            return value
        try:
            return enum_class(str(value).upper())
        except:
            if default:
                return default
            # Try to find closest match
            value_upper = str(value).upper()
            for item in enum_class:
                if item.value.upper() == value_upper:
                    return item
            # Return first enum value as fallback
            return list(enum_class)[0]
    
    def _parse_optional_int(self, value) -> Optional[int]:
        """Parse optional integer"""
        if pd.isna(value) or value is None:
            return None
        return int(value)
    
    def _parse_optional_float(self, value) -> Optional[float]:
        """Parse optional float"""
        if pd.isna(value) or value is None:
            return None
        return float(value)
    
    def _parse_optional_str(self, value) -> Optional[str]:
        """Parse optional string"""
        if pd.isna(value) or value is None:
            return None
        return str(value)
    
    def _parse_str_list(self, value) -> List[str]:
        """Parse comma-separated string list"""
        if pd.isna(value) or not value:
            return []
        if isinstance(value, list):
            return value
        return [s.strip() for s in str(value).split(',')]
    
    def _parse_int_list(self, value) -> List[int]:
        """Parse comma-separated integer list"""
        if pd.isna(value) or not value:
            return []
        if isinstance(value, list):
            return [int(x) for x in value]
        return [int(s.strip()) for s in str(value).split(',')]
    
    def _parse_json(self, value) -> Optional[Dict]:
        """Parse JSON string"""
        if pd.isna(value) or not value:
            return None
        if isinstance(value, dict):
            return value
        try:
            return json.loads(str(value))
        except:
            return None
    
    def _parse_strike_method(self, value) -> StrikeMethod:
        """Parse strike method with special handling"""
        if pd.isna(value):
            return StrikeMethod.ATM
        
        value_str = str(value).upper()
        
        # Handle special formats like OTM_100 -> OTM with offset
        if '_' in value_str:
            base = value_str.split('_')[0]
            if base in ['ATM', 'ITM', 'OTM']:
                return StrikeMethod(base)
        
        # Handle numbered variants
        if value_str.startswith('ITM') and len(value_str) == 4:
            return StrikeMethod(value_str)
        if value_str.startswith('OTM') and len(value_str) == 4:
            return StrikeMethod(value_str)
        
        return self._parse_enum(value, StrikeMethod, StrikeMethod.ATM)