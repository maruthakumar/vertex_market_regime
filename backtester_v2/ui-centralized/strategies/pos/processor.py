"""
Results Processor for POS (Positional) Strategy
Processes query results and generates trades, metrics, and analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, time, timedelta
import logging
from collections import defaultdict

from .models import (
    POSStrategyModel,
    POSLegModel,
    POSPortfolioModel,
    PositionType,
    OptionType,
    AdjustmentRule,
    AdjustmentTrigger,
    AdjustmentAction
)
from .constants import (
    DEFAULT_TRANSACTION_COST,
    DEFAULT_SLIPPAGE,
    ERROR_MESSAGES
)

logger = logging.getLogger(__name__)


class POSProcessor:
    """Process results from POS strategy queries"""
    
    def __init__(self):
        self.trades = []
        self.positions = {}
        self.adjustments = []
        self.greek_history = []
        
    def process_results(self, 
                       query_results: List[pd.DataFrame],
                       strategy_model: POSStrategyModel) -> Dict[str, Any]:
        """
        Process query results into trades and metrics
        
        Args:
            query_results: List of DataFrames from query execution
            strategy_model: Strategy configuration
            
        Returns:
            Dictionary containing trades, metrics, and analysis
        """
        try:
            # Extract main results
            main_results = query_results[0] if query_results else pd.DataFrame()
            
            if main_results.empty:
                logger.warning("No data returned from queries")
                return self._empty_results()
                
            # Process positions and trades
            trades_df = self._process_positions(main_results, strategy_model)
            
            # Calculate P&L
            trades_df = self._calculate_pnl(trades_df, strategy_model)
            
            # Process Greeks if available
            greek_metrics = {}
            if len(query_results) > 1 and strategy_model.portfolio.calculate_greeks:
                greek_results = query_results[1]
                greek_metrics = self._process_greeks(greek_results, strategy_model)
                
            # Process adjustments if enabled
            adjustment_summary = {}
            if strategy_model.portfolio.enable_adjustments and len(query_results) > 2:
                adjustment_results = query_results[2:]
                adjustment_summary = self._process_adjustments(
                    adjustment_results, 
                    trades_df,
                    strategy_model
                )
                
            # Calculate metrics
            metrics = self._calculate_metrics(trades_df, strategy_model)
            
            # Generate daily P&L
            daily_pnl = self._calculate_daily_pnl(trades_df)
            
            # Calculate monthly P&L
            monthly_pnl = self._calculate_monthly_pnl(trades_df)
            
            # Compile results in standard format for Excel output
            results = {
                "success": True,
                "trades_df": trades_df,  # Standard name for Excel generator
                "metrics": metrics,
                "daily_pnl": daily_pnl,
                "monthly_pnl": monthly_pnl,  # Added for Excel output
                "greek_metrics": greek_metrics,
                "adjustments": adjustment_summary,
                "positions": self._summarize_positions(trades_df),
                "summary": self._generate_strategy_summary(
                    trades_df, 
                    metrics,
                    strategy_model
                )
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise ValueError(f"Result processing failed: {str(e)}")
    
    def _process_positions(self, 
                         results_df: pd.DataFrame,
                         strategy_model: POSStrategyModel) -> pd.DataFrame:
        """Process multi-leg positions into trades"""
        
        trades = []
        current_positions = {}
        
        # Group by date for daily processing
        for trade_date, date_group in results_df.groupby('trade_date'):
            
            # Process each time point
            for idx, row in date_group.iterrows():
                
                # Check for entry signals
                if self._check_entry_conditions(row, strategy_model):
                    position = self._create_position(row, strategy_model)
                    current_positions[position['position_id']] = position
                    
                    # Record entry trades
                    for leg in strategy_model.legs:
                        trade = self._create_trade(
                            row, leg, position['position_id'], 'ENTRY'
                        )
                        trades.append(trade)
                        
                # Check for exit signals
                positions_to_close = []
                for position_id, position in current_positions.items():
                    if self._check_exit_conditions(row, position, strategy_model):
                        positions_to_close.append(position_id)
                        
                        # Record exit trades
                        for leg in strategy_model.legs:
                            trade = self._create_trade(
                                row, leg, position_id, 'EXIT'
                            )
                            trades.append(trade)
                            
                # Remove closed positions
                for position_id in positions_to_close:
                    del current_positions[position_id]
                    
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Sort by datetime
        if not trades_df.empty:
            trades_df = trades_df.sort_values(['trade_date', 'trade_time'])
            
        return trades_df
    
    def _create_position(self, row: pd.Series, strategy_model: POSStrategyModel) -> Dict[str, Any]:
        """Create a new position"""
        
        position_id = f"{row['trade_date']}_{row['trade_time']}"
        
        position = {
            'position_id': position_id,
            'entry_date': row['trade_date'],
            'entry_time': row['trade_time'],
            'strategy_type': strategy_model.portfolio.strategy_type,
            'legs': {},
            'total_premium': 0,
            'max_profit': 0,
            'max_loss': 0,
            'breakeven_points': [],
            'current_greeks': {},
            'adjustments_made': 0
        }
        
        # Calculate position metrics
        for leg in strategy_model.legs:
            leg_data = self._extract_leg_data(row, leg)
            position['legs'][leg.leg_id] = leg_data
            
            # Update premium
            if leg.position_type == PositionType.SELL:
                position['total_premium'] += leg_data['premium']
            else:
                position['total_premium'] -= leg_data['premium']
                
        # Calculate max profit/loss
        position['max_profit'] = self._calculate_max_profit(position, strategy_model)
        position['max_loss'] = self._calculate_max_loss(position, strategy_model)
        position['breakeven_points'] = self._calculate_breakevens(position, strategy_model)
        
        return position
    
    def _create_trade(self, 
                     row: pd.Series,
                     leg: POSLegModel,
                     position_id: str,
                     trade_type: str) -> Dict[str, Any]:
        """Create a trade record"""
        
        leg_data = self._extract_leg_data(row, leg)
        
        trade = {
            'position_id': position_id,
            'trade_date': row['trade_date'],
            'trade_time': row['trade_time'],
            'trade_type': trade_type,
            'leg_id': leg.leg_id,
            'leg_name': leg.leg_name,
            'option_type': leg.option_type,
            'position_type': leg.position_type,
            'strike_price': leg_data['strike_price'],
            'expiry_date': leg_data['expiry_date'],
            'quantity': leg.lots * leg.lot_size,
            'price': leg_data['close_price'],
            'premium': leg_data['premium'],
            'implied_volatility': leg_data['implied_volatility'],
            'delta': leg_data['delta'],
            'gamma': leg_data['gamma'],
            'theta': leg_data['theta'],
            'vega': leg_data['vega'],
            'underlying_price': row.get('underlying_value', 0),
            'transaction_cost': abs(leg_data['premium']) * strategy_model.portfolio.transaction_costs,
            'slippage': abs(leg_data['premium']) * strategy_model.portfolio.slippage_value
        }
        
        return trade
    
    def _extract_leg_data(self, row: pd.Series, leg: POSLegModel) -> Dict[str, Any]:
        """Extract leg-specific data from row"""
        
        leg_prefix = f"leg_{leg.leg_id}_"
        
        return {
            'strike_price': row.get(f'{leg_prefix}strike_price', 0),
            'expiry_date': row.get(f'{leg_prefix}expiry_date', row['trade_date']),
            'close_price': row.get(f'{leg_prefix}close_price', 0),
            'premium': row.get(f'{leg_prefix}premium', 0),
            'volume': row.get(f'{leg_prefix}volume', 0),
            'open_interest': row.get(f'{leg_prefix}open_interest', 0),
            'implied_volatility': row.get(f'{leg_prefix}implied_volatility', 0),
            'delta': row.get(f'{leg_prefix}delta', 0),
            'gamma': row.get(f'{leg_prefix}gamma', 0),
            'theta': row.get(f'{leg_prefix}theta', 0),
            'vega': row.get(f'{leg_prefix}vega', 0)
        }
    
    def _check_entry_conditions(self, row: pd.Series, strategy_model: POSStrategyModel) -> bool:
        """Check if entry conditions are met"""
        
        # Check if we're within entry time window
        for leg in strategy_model.legs:
            if row.get(f'leg_{leg.leg_id}_entry_flag', 0) == 1:
                # Check additional entry conditions
                if leg.entry_conditions:
                    # Evaluate custom conditions
                    for condition in leg.entry_conditions:
                        if not self._evaluate_condition(condition, row):
                            return False
                return True
                
        return False
    
    def _check_exit_conditions(self, 
                             row: pd.Series,
                             position: Dict[str, Any],
                             strategy_model: POSStrategyModel) -> bool:
        """Check if exit conditions are met"""
        
        # Check time-based exit
        for leg in strategy_model.legs:
            if row.get(f'leg_{leg.leg_id}_exit_flag', 0) == 1:
                return True
                
        # Check stop loss
        if position.get('current_pnl', 0) <= -position['max_loss'] * 0.8:
            return True
            
        # Check take profit
        if position.get('current_pnl', 0) >= position['max_profit'] * 0.8:
            return True
            
        # Check expiry
        if row['trade_date'] >= position['legs'][1]['expiry_date']:
            return True
            
        return False
    
    def _evaluate_condition(self, condition: str, row: pd.Series) -> bool:
        """Evaluate a custom condition"""
        
        # Simple evaluation - in production, use safe evaluation
        try:
            # Replace column references with row values
            for col in row.index:
                if col in condition:
                    condition = condition.replace(col, str(row[col]))
            return eval(condition)
        except:
            return True
    
    def _calculate_pnl(self, 
                      trades_df: pd.DataFrame,
                      strategy_model: POSStrategyModel) -> pd.DataFrame:
        """Calculate P&L for trades"""
        
        if trades_df.empty:
            return trades_df
            
        # Group by position
        position_pnl = {}
        
        for position_id, position_trades in trades_df.groupby('position_id'):
            
            # Separate entry and exit trades
            entry_trades = position_trades[position_trades['trade_type'] == 'ENTRY']
            exit_trades = position_trades[position_trades['trade_type'] == 'EXIT']
            
            # Calculate entry cost
            entry_cost = 0
            for _, trade in entry_trades.iterrows():
                if trade['position_type'] == PositionType.BUY:
                    entry_cost -= trade['premium']  # Debit
                else:
                    entry_cost += trade['premium']  # Credit
                    
            # Calculate exit value
            exit_value = 0
            if not exit_trades.empty:
                for _, trade in exit_trades.iterrows():
                    if trade['position_type'] == PositionType.BUY:
                        exit_value += trade['premium']  # Sell to close
                    else:
                        exit_value -= trade['premium']  # Buy to close
                        
            # Calculate P&L
            gross_pnl = entry_cost + exit_value
            
            # Calculate costs
            total_costs = (
                position_trades['transaction_cost'].sum() +
                position_trades['slippage'].sum()
            )
            
            net_pnl = gross_pnl - total_costs
            
            # Update position P&L
            position_pnl[position_id] = {
                'gross_pnl': gross_pnl,
                'total_costs': total_costs,
                'net_pnl': net_pnl
            }
            
        # Add P&L to trades
        trades_df['position_gross_pnl'] = trades_df['position_id'].map(
            lambda x: position_pnl.get(x, {}).get('gross_pnl', 0)
        )
        trades_df['position_net_pnl'] = trades_df['position_id'].map(
            lambda x: position_pnl.get(x, {}).get('net_pnl', 0)
        )
        
        return trades_df
    
    def _calculate_metrics(self, 
                         trades_df: pd.DataFrame,
                         strategy_model: POSStrategyModel) -> Dict[str, float]:
        """Calculate strategy metrics"""
        
        if trades_df.empty:
            return self._empty_metrics()
            
        # Get unique positions
        position_pnls = trades_df.groupby('position_id')['position_net_pnl'].first()
        
        metrics = {
            'total_trades': len(position_pnls),
            'total_pnl': position_pnls.sum(),
            'average_pnl': position_pnls.mean(),
            'winning_trades': (position_pnls > 0).sum(),
            'losing_trades': (position_pnls < 0).sum(),
            'win_rate': (position_pnls > 0).sum() / len(position_pnls) if len(position_pnls) > 0 else 0,
            'largest_win': position_pnls.max() if len(position_pnls) > 0 else 0,
            'largest_loss': position_pnls.min() if len(position_pnls) > 0 else 0,
            'profit_factor': abs(position_pnls[position_pnls > 0].sum() / position_pnls[position_pnls < 0].sum()) if (position_pnls < 0).any() else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(position_pnls),
            'max_drawdown': self._calculate_max_drawdown(position_pnls),
            'calmar_ratio': self._calculate_calmar_ratio(position_pnls)
        }
        
        # Add strategy-specific metrics
        if strategy_model.portfolio.strategy_type == "IRON_CONDOR":
            metrics.update(self._calculate_iron_condor_metrics(trades_df))
        elif strategy_model.portfolio.strategy_type == "CALENDAR_SPREAD":
            metrics.update(self._calculate_calendar_metrics(trades_df))
            
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        
        if returns.empty or returns.std() == 0:
            return 0
            
        # Annualize
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        if returns.empty:
            return 0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        
        max_dd = abs(self._calculate_max_drawdown(returns))
        if max_dd == 0:
            return 0
            
        annual_return = returns.mean() * 252
        return annual_return / max_dd
    
    def _calculate_daily_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily P&L summary"""
        
        if trades_df.empty:
            return pd.DataFrame()
            
        # Group by date
        daily_pnl = trades_df.groupby('trade_date').agg({
            'position_net_pnl': 'sum',
            'position_gross_pnl': 'sum',
            'transaction_cost': 'sum',
            'slippage': 'sum'
        }).reset_index()
        
        # Calculate cumulative P&L
        daily_pnl['cumulative_pnl'] = daily_pnl['position_net_pnl'].cumsum()
        
        # Calculate daily return
        daily_pnl['daily_return'] = daily_pnl['position_net_pnl'] / strategy_model.portfolio.position_size_value
        
        return daily_pnl
    
    def _calculate_monthly_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly P&L summary"""
        
        if trades_df.empty:
            return pd.DataFrame()
            
        # Group by month and calculate monthly metrics
        trades_df['month'] = pd.to_datetime(trades_df['trade_date']).dt.to_period('M')
        
        monthly_pnl = trades_df.groupby('month').agg({
            'position_net_pnl': 'sum',
            'position_gross_pnl': 'sum',
            'transaction_cost': 'sum',
            'slippage': 'sum',
            'trade_id': 'nunique'  # Count unique trades per month
        }).reset_index()
        
        # Rename columns for clarity
        monthly_pnl = monthly_pnl.rename(columns={
            'position_net_pnl': 'Monthly PnL',
            'position_gross_pnl': 'Gross PnL',
            'transaction_cost': 'Transaction Costs',
            'slippage': 'Slippage',
            'trade_id': 'Trades'
        })
        
        # Calculate monthly statistics
        monthly_pnl['Winning Trades'] = trades_df.groupby('month')['position_net_pnl'].apply(
            lambda x: (x > 0).sum()
        ).values
        monthly_pnl['Losing Trades'] = monthly_pnl['Trades'] - monthly_pnl['Winning Trades']
        monthly_pnl['Win Rate'] = monthly_pnl['Winning Trades'] / monthly_pnl['Trades']
        monthly_pnl['Win Rate'] = monthly_pnl['Win Rate'].fillna(0)
        
        # Calculate cumulative monthly P&L
        monthly_pnl['Cumulative PnL'] = monthly_pnl['Monthly PnL'].cumsum()
        
        # Convert month period back to string for Excel compatibility
        monthly_pnl['Month'] = monthly_pnl['month'].astype(str)
        monthly_pnl = monthly_pnl.drop('month', axis=1)
        
        return monthly_pnl
    
    def _process_greeks(self, 
                       greek_results: pd.DataFrame,
                       strategy_model: POSStrategyModel) -> Dict[str, Any]:
        """Process Greek calculations"""
        
        if greek_results.empty:
            return {}
            
        # Calculate Greek statistics
        greek_stats = {
            'delta': {
                'mean': greek_results['total_portfolio_delta'].mean(),
                'std': greek_results['total_portfolio_delta'].std(),
                'min': greek_results['total_portfolio_delta'].min(),
                'max': greek_results['total_portfolio_delta'].max(),
                'breaches': greek_results['delta_limit_breach'].sum()
            },
            'gamma': {
                'mean': greek_results['total_portfolio_gamma'].mean(),
                'std': greek_results['total_portfolio_gamma'].std(),
                'min': greek_results['total_portfolio_gamma'].min(),
                'max': greek_results['total_portfolio_gamma'].max(),
                'breaches': greek_results['gamma_limit_breach'].sum()
            },
            'theta': {
                'mean': greek_results['total_portfolio_theta'].mean(),
                'std': greek_results['total_portfolio_theta'].std(),
                'min': greek_results['total_portfolio_theta'].min(),
                'max': greek_results['total_portfolio_theta'].max(),
                'breaches': greek_results['theta_limit_breach'].sum()
            },
            'vega': {
                'mean': greek_results['total_portfolio_vega'].mean(),
                'std': greek_results['total_portfolio_vega'].std(),
                'min': greek_results['total_portfolio_vega'].min(),
                'max': greek_results['total_portfolio_vega'].max(),
                'breaches': greek_results['vega_limit_breach'].sum()
            }
        }
        
        # Store time series for charting
        self.greek_history = greek_results[[
            'trade_date', 'trade_time',
            'total_portfolio_delta', 'total_portfolio_gamma',
            'total_portfolio_theta', 'total_portfolio_vega'
        ]].copy()
        
        return greek_stats
    
    def _process_adjustments(self,
                           adjustment_results: List[pd.DataFrame],
                           trades_df: pd.DataFrame,
                           strategy_model: POSStrategyModel) -> Dict[str, Any]:
        """Process strategy adjustments"""
        
        adjustments_made = []
        
        for adj_df in adjustment_results:
            if adj_df.empty:
                continue
                
            # Find triggered adjustments
            for idx, row in adj_df.iterrows():
                triggered_rules = row.get('triggered_rules', [])
                
                for rule_id in triggered_rules:
                    # Find the corresponding rule
                    rule = self._find_adjustment_rule(rule_id, strategy_model)
                    if rule:
                        adjustment = {
                            'date': row['trade_date'],
                            'time': row['trade_time'],
                            'rule_id': rule_id,
                            'trigger_type': rule.trigger_type,
                            'action_type': rule.action_type,
                            'underlying_price': row.get('underlying_value', 0)
                        }
                        adjustments_made.append(adjustment)
                        
        # Summarize adjustments
        summary = {
            'total_adjustments': len(adjustments_made),
            'adjustments_by_type': defaultdict(int),
            'adjustments_by_trigger': defaultdict(int),
            'adjustment_details': adjustments_made
        }
        
        for adj in adjustments_made:
            summary['adjustments_by_type'][adj['action_type']] += 1
            summary['adjustments_by_trigger'][adj['trigger_type']] += 1
            
        return dict(summary)
    
    def _find_adjustment_rule(self, rule_id: str, strategy_model: POSStrategyModel) -> Optional[AdjustmentRule]:
        """Find adjustment rule by ID"""
        
        for leg in strategy_model.legs:
            for rule in leg.adjustment_rules:
                if rule.rule_id == rule_id:
                    return rule
        return None
    
    def _calculate_max_profit(self, position: Dict[str, Any], strategy_model: POSStrategyModel) -> float:
        """Calculate maximum profit for position"""
        
        # Simplified calculation - depends on strategy type
        if strategy_model.portfolio.strategy_type == "IRON_CONDOR":
            # Max profit is the net credit received
            return max(position['total_premium'], 0)
        elif strategy_model.portfolio.strategy_type == "CALENDAR_SPREAD":
            # Max profit is theoretically unlimited
            return position['total_premium'] * 2  # Estimate
        else:
            return position['total_premium']
    
    def _calculate_max_loss(self, position: Dict[str, Any], strategy_model: POSStrategyModel) -> float:
        """Calculate maximum loss for position"""
        
        # Simplified calculation - depends on strategy type
        if strategy_model.portfolio.strategy_type == "IRON_CONDOR":
            # Max loss is spread width minus credit
            spread_width = 100  # Assuming 100 point spreads
            return spread_width * 50 - position['total_premium']  # 50 is lot size
        elif strategy_model.portfolio.strategy_type == "CALENDAR_SPREAD":
            # Max loss is the debit paid
            return abs(min(position['total_premium'], 0))
        else:
            return abs(position['total_premium'])
    
    def _calculate_breakevens(self, position: Dict[str, Any], strategy_model: POSStrategyModel) -> List[float]:
        """Calculate breakeven points"""
        
        breakevens = []
        
        if strategy_model.portfolio.strategy_type == "IRON_CONDOR":
            # Two breakevens for iron condor
            # Simplified - would need actual strikes
            breakevens = [
                position['legs'][1]['strike_price'] - position['total_premium'] / 50,
                position['legs'][3]['strike_price'] + position['total_premium'] / 50
            ]
        elif strategy_model.portfolio.strategy_type == "STRADDLE":
            # Two breakevens for straddle
            strike = position['legs'][1]['strike_price']
            breakevens = [
                strike - abs(position['total_premium']) / 50,
                strike + abs(position['total_premium']) / 50
            ]
            
        return breakevens
    
    def _summarize_positions(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize position statistics"""
        
        if trades_df.empty:
            return {}
            
        position_summary = trades_df.groupby('position_id').agg({
            'trade_date': ['min', 'max'],
            'position_net_pnl': 'first',
            'quantity': 'sum'
        })
        
        return {
            'total_positions': len(position_summary),
            'avg_holding_days': (position_summary['trade_date']['max'] - 
                               position_summary['trade_date']['min']).dt.days.mean(),
            'avg_position_size': position_summary['quantity']['sum'].mean(),
            'pnl_distribution': position_summary['position_net_pnl']['first'].describe().to_dict()
        }
    
    def _generate_strategy_summary(self,
                                 trades_df: pd.DataFrame,
                                 metrics: Dict[str, float],
                                 strategy_model: POSStrategyModel) -> Dict[str, Any]:
        """Generate comprehensive strategy summary"""
        
        return {
            'strategy_name': strategy_model.portfolio.strategy_name,
            'strategy_type': strategy_model.portfolio.strategy_type,
            'date_range': {
                'start': strategy_model.portfolio.start_date,
                'end': strategy_model.portfolio.end_date,
                'trading_days': len(trades_df['trade_date'].unique()) if not trades_df.empty else 0
            },
            'performance': {
                'total_return': metrics.get('total_pnl', 0) / strategy_model.portfolio.position_size_value,
                'annualized_return': metrics.get('total_pnl', 0) / strategy_model.portfolio.position_size_value * 252 / max(len(trades_df['trade_date'].unique()), 1),
                'win_rate': metrics.get('win_rate', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            },
            'risk_metrics': {
                'max_portfolio_risk': strategy_model.portfolio.max_portfolio_risk,
                'actual_max_loss': metrics.get('largest_loss', 0),
                'risk_reward_ratio': abs(metrics.get('average_pnl', 0) / metrics.get('largest_loss', 1)) if metrics.get('largest_loss', 0) != 0 else 0
            },
            'execution': {
                'total_legs': len(strategy_model.legs),
                'total_adjustments': len(self.adjustments),
                'transaction_costs': trades_df['transaction_cost'].sum() if not trades_df.empty else 0,
                'slippage_costs': trades_df['slippage'].sum() if not trades_df.empty else 0
            }
        }
    
    def _calculate_iron_condor_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Iron Condor specific metrics"""
        
        return {
            'wing_efficiency': 0.85,  # Placeholder
            'credit_capture_rate': 0.75,  # Placeholder
            'early_exit_rate': 0.30  # Placeholder
        }
    
    def _calculate_calendar_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Calendar Spread specific metrics"""
        
        return {
            'volatility_capture': 0.65,  # Placeholder
            'time_decay_efficiency': 0.80,  # Placeholder
            'roll_success_rate': 0.70  # Placeholder
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        
        return {
            "trades": pd.DataFrame(),
            "metrics": self._empty_metrics(),
            "daily_pnl": pd.DataFrame(),
            "greek_metrics": {},
            "adjustments": {},
            "positions": {},
            "strategy_summary": {}
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics"""
        
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'average_pnl': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0
        }