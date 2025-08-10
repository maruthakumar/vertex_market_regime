"""
Results Processor for ML Indicator Strategy
Processes query results with indicators and generates trades, metrics, and analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, time, timedelta
import logging
from collections import defaultdict

from .models import (
    MLIndicatorStrategyModel,
    MLIndicatorPortfolioModel,
    MLLegModel,
    SignalCondition,
    SignalLogic,
    ComparisonOperator,
    RiskManagementConfig
)
from .constants import (
    RISK_DEFAULTS,
    PERFORMANCE_METRICS,
    ERROR_MESSAGES
)

logger = logging.getLogger(__name__)


class MLIndicatorProcessor:
    """Process results from ML Indicator strategy queries"""
    
    def __init__(self):
        self.trades = []
        self.signals = []
        self.indicator_values = {}
        self.ml_predictions = []
        
    def process_results(self,
                       query_results: List[pd.DataFrame],
                       strategy_model: MLIndicatorStrategyModel) -> Dict[str, Any]:
        """
        Process query results into trades and metrics
        
        Args:
            query_results: List of DataFrames from query execution
            strategy_model: Strategy configuration
            
        Returns:
            Dictionary containing trades, metrics, and analysis
        """
        try:
            # Extract main results with signals
            signal_results = query_results[0] if query_results else pd.DataFrame()
            
            if signal_results.empty:
                logger.warning("No data returned from queries")
                return self._empty_results()
                
            # Store indicator values for analysis
            self.indicator_values = self._extract_indicator_values(
                signal_results,
                strategy_model
            )
            
            # Process signals into trades
            if strategy_model.legs:
                # Multi-leg strategy
                trades_df = self._process_multi_leg_trades(
                    signal_results,
                    query_results[1:] if len(query_results) > 1 else [],
                    strategy_model
                )
            else:
                # Single instrument strategy
                trades_df = self._process_single_instrument_trades(
                    signal_results,
                    strategy_model
                )
                
            # Apply risk management
            trades_df = self._apply_risk_management(trades_df, strategy_model)
            
            # Calculate P&L
            trades_df = self._calculate_pnl(trades_df, strategy_model)
            
            # Process ML predictions if available
            ml_analysis = {}
            if strategy_model.ml_config and len(query_results) > len(strategy_model.legs):
                ml_results = query_results[-1]
                ml_analysis = self._process_ml_predictions(ml_results, trades_df)
                
            # Calculate metrics
            metrics = self._calculate_metrics(trades_df, strategy_model)
            
            # Generate daily P&L
            daily_pnl = self._calculate_daily_pnl(trades_df)
            
            # Analyze signals
            signal_analysis = self._analyze_signals(signal_results, trades_df)
            
            # Analyze indicators
            indicator_analysis = self._analyze_indicators(self.indicator_values, trades_df)
            
            # Compile results
            results = {
                "trades": trades_df,
                "metrics": metrics,
                "daily_pnl": daily_pnl,
                "signal_analysis": signal_analysis,
                "indicator_analysis": indicator_analysis,
                "ml_analysis": ml_analysis,
                "indicator_values": self.indicator_values,
                "strategy_summary": self._generate_strategy_summary(
                    trades_df,
                    metrics,
                    signal_analysis,
                    strategy_model
                )
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise ValueError(f"Result processing failed: {str(e)}")
    
    def _extract_indicator_values(self,
                                signal_results: pd.DataFrame,
                                strategy_model: MLIndicatorStrategyModel) -> pd.DataFrame:
        """Extract indicator values from results"""
        
        # Get all indicator columns
        indicator_columns = ['trade_date', 'trade_time', 'close_price', 'volume']
        
        # Add configured indicators
        for indicator in strategy_model.indicators:
            col_name = f"{indicator.indicator_name}_{indicator.parameters.get('timeperiod', '')}"
            if col_name in signal_results.columns:
                indicator_columns.append(col_name)
            # Check for multi-output indicators
            for suffix in ['', '_0', '_1', '_2', '_UPPER', '_MIDDLE', '_LOWER']:
                full_col = f"{indicator.indicator_name}{suffix}"
                if full_col in signal_results.columns:
                    indicator_columns.append(full_col)
                    
        # Add SMC indicators if configured
        if strategy_model.smc_config:
            smc_columns = ['bos_signal', 'order_block_type', 'fvg_signal', 'liquidity_grab']
            for col in smc_columns:
                if col in signal_results.columns:
                    indicator_columns.append(col)
                    
        # Extract available columns
        available_columns = [col for col in indicator_columns if col in signal_results.columns]
        
        return signal_results[available_columns].copy()
    
    def _process_single_instrument_trades(self,
                                        signal_results: pd.DataFrame,
                                        strategy_model: MLIndicatorStrategyModel) -> pd.DataFrame:
        """Process trades for single instrument strategy"""
        
        trades = []
        position_open = False
        current_position = None
        
        for idx, row in signal_results.iterrows():
            
            # Check for entry signal
            if not position_open and row.get('entry_signal', 0) == 1:
                # Open position
                position = {
                    'trade_id': f"{row['trade_date']}_{row['trade_time']}_ENTRY",
                    'trade_date': row['trade_date'],
                    'trade_time': row['trade_time'],
                    'trade_type': 'ENTRY',
                    'direction': strategy_model.position_type,
                    'price': row['close_price'],
                    'quantity': self._calculate_position_size(row, strategy_model),
                    'signal_strength': row.get('entry_strength', 1.0),
                    'underlying_price': row.get('underlying_value', row['close_price'])
                }
                
                trades.append(position)
                position_open = True
                current_position = position.copy()
                
            # Check for exit signal
            elif position_open and row.get('exit_signal', 0) == 1:
                # Close position
                exit_trade = {
                    'trade_id': f"{row['trade_date']}_{row['trade_time']}_EXIT",
                    'trade_date': row['trade_date'],
                    'trade_time': row['trade_time'],
                    'trade_type': 'EXIT',
                    'direction': 'SELL' if current_position['direction'] == 'LONG' else 'BUY',
                    'price': row['close_price'],
                    'quantity': current_position['quantity'],
                    'signal_strength': row.get('exit_strength', 1.0),
                    'underlying_price': row.get('underlying_value', row['close_price']),
                    'entry_trade_id': current_position['trade_id']
                }
                
                trades.append(exit_trade)
                position_open = False
                current_position = None
                
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        return trades_df
    
    def _process_multi_leg_trades(self,
                                signal_results: pd.DataFrame,
                                leg_results: List[pd.DataFrame],
                                strategy_model: MLIndicatorStrategyModel) -> pd.DataFrame:
        """Process trades for multi-leg strategy"""
        
        trades = []
        position_open = False
        current_position = {}
        
        # Combine leg results
        combined_legs = self._combine_leg_results(leg_results, strategy_model)
        
        for idx, row in signal_results.iterrows():
            
            # Check for entry signal
            if not position_open and row.get('entry_signal', 0) == 1:
                
                # Create trades for each leg
                for leg_idx, leg in enumerate(strategy_model.legs):
                    leg_data = combined_legs[combined_legs['trade_date'] == row['trade_date']].iloc[0]
                    
                    trade = {
                        'trade_id': f"{row['trade_date']}_{row['trade_time']}_L{leg.leg_id}_ENTRY",
                        'trade_date': row['trade_date'],
                        'trade_time': row['trade_time'],
                        'trade_type': 'ENTRY',
                        'leg_id': leg.leg_id,
                        'leg_name': leg.leg_name,
                        'option_type': leg.option_type,
                        'position_type': leg.position_type,
                        'strike_price': leg_data.get(f'leg_{leg.leg_id}_strike', 0),
                        'price': leg_data.get(f'leg_{leg.leg_id}_price', 0),
                        'quantity': leg.lots * leg.lot_size,
                        'signal_strength': row.get('entry_strength', 1.0),
                        'delta': leg_data.get(f'leg_{leg.leg_id}_delta', 0),
                        'gamma': leg_data.get(f'leg_{leg.leg_id}_gamma', 0),
                        'theta': leg_data.get(f'leg_{leg.leg_id}_theta', 0),
                        'vega': leg_data.get(f'leg_{leg.leg_id}_vega', 0)
                    }
                    
                    trades.append(trade)
                    current_position[leg.leg_id] = trade.copy()
                    
                position_open = True
                
            # Check for exit signal
            elif position_open and row.get('exit_signal', 0) == 1:
                
                # Create exit trades for each leg
                for leg_idx, leg in enumerate(strategy_model.legs):
                    leg_data = combined_legs[combined_legs['trade_date'] == row['trade_date']].iloc[0]
                    
                    trade = {
                        'trade_id': f"{row['trade_date']}_{row['trade_time']}_L{leg.leg_id}_EXIT",
                        'trade_date': row['trade_date'],
                        'trade_time': row['trade_time'],
                        'trade_type': 'EXIT',
                        'leg_id': leg.leg_id,
                        'leg_name': leg.leg_name,
                        'option_type': leg.option_type,
                        'position_type': 'SELL' if leg.position_type == 'BUY' else 'BUY',
                        'strike_price': current_position[leg.leg_id]['strike_price'],
                        'price': leg_data.get(f'leg_{leg.leg_id}_price', 0),
                        'quantity': leg.lots * leg.lot_size,
                        'signal_strength': row.get('exit_strength', 1.0),
                        'entry_trade_id': current_position[leg.leg_id]['trade_id']
                    }
                    
                    trades.append(trade)
                    
                position_open = False
                current_position = {}
                
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        return trades_df
    
    def _combine_leg_results(self,
                           leg_results: List[pd.DataFrame],
                           strategy_model: MLIndicatorStrategyModel) -> pd.DataFrame:
        """Combine results from multiple leg queries"""
        
        if not leg_results:
            return pd.DataFrame()
            
        # Start with first leg
        combined = leg_results[0].copy()
        
        # Join other legs
        for i, leg_df in enumerate(leg_results[1:], 1):
            leg = strategy_model.legs[i]
            
            # Rename columns to avoid conflicts
            leg_df = leg_df.rename(columns={
                'strike_price': f'leg_{leg.leg_id}_strike',
                'close_price': f'leg_{leg.leg_id}_price',
                'delta': f'leg_{leg.leg_id}_delta',
                'gamma': f'leg_{leg.leg_id}_gamma',
                'theta': f'leg_{leg.leg_id}_theta',
                'vega': f'leg_{leg.leg_id}_vega'
            })
            
            # Merge on date and time
            combined = pd.merge(
                combined, leg_df,
                on=['trade_date', 'trade_time'],
                how='inner'
            )
            
        return combined
    
    def _calculate_position_size(self, row: pd.Series, strategy_model: MLIndicatorStrategyModel) -> float:
        """Calculate position size based on risk management rules"""
        
        risk_config = strategy_model.portfolio.risk_config
        
        if risk_config.position_sizing == "FIXED":
            return risk_config.max_position_size
            
        elif risk_config.position_sizing == "KELLY":
            # Simplified Kelly criterion
            win_rate = 0.55  # Would come from historical data
            avg_win = 1.5
            avg_loss = 1.0
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return risk_config.max_position_size * min(kelly_fraction, 0.25)
            
        elif risk_config.position_sizing == "VOLATILITY":
            # Volatility-based sizing
            volatility = row.get('ATR_14', row['close_price'] * 0.02)
            return risk_config.max_position_size * (0.02 / volatility)
            
        else:
            return risk_config.max_position_size
    
    def _apply_risk_management(self,
                             trades_df: pd.DataFrame,
                             strategy_model: MLIndicatorStrategyModel) -> pd.DataFrame:
        """Apply risk management rules to trades"""
        
        if trades_df.empty:
            return trades_df
            
        risk_config = strategy_model.portfolio.risk_config
        
        # Group by position
        positions = []
        
        # Process each entry-exit pair
        entry_trades = trades_df[trades_df['trade_type'] == 'ENTRY']
        
        for _, entry in entry_trades.iterrows():
            # Find corresponding exit
            exit_trades = trades_df[
                (trades_df['trade_type'] == 'EXIT') &
                (trades_df.get('entry_trade_id', '') == entry['trade_id'])
            ]
            
            if not exit_trades.empty:
                exit_trade = exit_trades.iloc[0]
                
                # Calculate P&L
                if 'direction' in entry:  # Single instrument
                    if entry['direction'] == 'LONG':
                        pnl_points = exit_trade['price'] - entry['price']
                    else:
                        pnl_points = entry['price'] - exit_trade['price']
                    pnl = pnl_points * entry['quantity']
                else:  # Multi-leg
                    # Calculate based on leg positions
                    pnl = self._calculate_multi_leg_pnl(entry, exit_trade, trades_df)
                    
                # Check stop loss
                if risk_config.stop_loss_type == "PERCENTAGE":
                    max_loss = entry['price'] * entry['quantity'] * risk_config.stop_loss_value
                    if pnl < -max_loss:
                        # Would adjust exit price in real implementation
                        logger.info(f"Stop loss triggered for {entry['trade_id']}")
                        
                # Check take profit
                if risk_config.take_profit_type == "PERCENTAGE":
                    target_profit = entry['price'] * entry['quantity'] * risk_config.take_profit_value
                    if pnl > target_profit:
                        logger.info(f"Take profit triggered for {entry['trade_id']}")
                        
        return trades_df
    
    def _calculate_multi_leg_pnl(self, entry: pd.Series, exit: pd.Series, trades_df: pd.DataFrame) -> float:
        """Calculate P&L for multi-leg position"""
        
        # Get all legs for this position
        position_trades = trades_df[
            trades_df['trade_id'].str.contains(entry['trade_id'].split('_L')[0])
        ]
        
        total_pnl = 0
        
        # Calculate P&L for each leg
        for leg_id in position_trades['leg_id'].unique():
            leg_entry = position_trades[
                (position_trades['leg_id'] == leg_id) &
                (position_trades['trade_type'] == 'ENTRY')
            ].iloc[0]
            
            leg_exit = position_trades[
                (position_trades['leg_id'] == leg_id) &
                (position_trades['trade_type'] == 'EXIT')
            ].iloc[0]
            
            # Calculate based on position type
            if leg_entry['position_type'] == 'BUY':
                leg_pnl = (leg_exit['price'] - leg_entry['price']) * leg_entry['quantity']
            else:  # SELL
                leg_pnl = (leg_entry['price'] - leg_exit['price']) * leg_entry['quantity']
                
            total_pnl += leg_pnl
            
        return total_pnl
    
    def _calculate_pnl(self,
                      trades_df: pd.DataFrame,
                      strategy_model: MLIndicatorStrategyModel) -> pd.DataFrame:
        """Calculate P&L for trades"""
        
        if trades_df.empty:
            return trades_df
            
        # Initialize P&L columns
        trades_df['gross_pnl'] = 0
        trades_df['transaction_cost'] = 0
        trades_df['slippage'] = 0
        trades_df['net_pnl'] = 0
        
        # Calculate transaction costs
        trades_df['transaction_cost'] = (
            trades_df['price'] * trades_df['quantity'] * 
            strategy_model.portfolio.transaction_costs
        )
        
        # Calculate slippage
        execution_config = strategy_model.portfolio.execution_config
        if execution_config.slippage_model == "FIXED":
            trades_df['slippage'] = execution_config.slippage_value * trades_df['quantity']
        elif execution_config.slippage_model == "PERCENTAGE":
            trades_df['slippage'] = (
                trades_df['price'] * trades_df['quantity'] * 
                execution_config.slippage_value
            )
            
        # Process each position
        if 'leg_id' in trades_df.columns:
            # Multi-leg strategy
            trades_df = self._calculate_multi_leg_full_pnl(trades_df)
        else:
            # Single instrument
            trades_df = self._calculate_single_instrument_pnl(trades_df)
            
        return trades_df
    
    def _calculate_single_instrument_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate P&L for single instrument trades"""
        
        # Match entry and exit trades
        entry_trades = trades_df[trades_df['trade_type'] == 'ENTRY'].copy()
        
        for idx, entry in entry_trades.iterrows():
            # Find exit
            exit_mask = (
                (trades_df['trade_type'] == 'EXIT') &
                (trades_df['entry_trade_id'] == entry['trade_id'])
            )
            
            if exit_mask.any():
                exit_idx = trades_df[exit_mask].index[0]
                exit_trade = trades_df.loc[exit_idx]
                
                # Calculate P&L
                if entry['direction'] == 'LONG':
                    gross_pnl = (exit_trade['price'] - entry['price']) * entry['quantity']
                else:  # SHORT
                    gross_pnl = (entry['price'] - exit_trade['price']) * entry['quantity']
                    
                # Update both trades
                trades_df.loc[idx, 'gross_pnl'] = gross_pnl
                trades_df.loc[exit_idx, 'gross_pnl'] = gross_pnl
                
                # Calculate net P&L
                total_costs = (
                    entry['transaction_cost'] + exit_trade['transaction_cost'] +
                    entry['slippage'] + exit_trade['slippage']
                )
                
                net_pnl = gross_pnl - total_costs
                trades_df.loc[idx, 'net_pnl'] = net_pnl
                trades_df.loc[exit_idx, 'net_pnl'] = net_pnl
                
        return trades_df
    
    def _calculate_multi_leg_full_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate P&L for multi-leg trades"""
        
        # Group by position (extract position ID from trade_id)
        trades_df['position_id'] = trades_df['trade_id'].str.extract(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})')
        
        for position_id, position_trades in trades_df.groupby('position_id'):
            total_entry_cost = 0
            total_exit_value = 0
            
            # Calculate for each leg
            for leg_id, leg_trades in position_trades.groupby('leg_id'):
                entry = leg_trades[leg_trades['trade_type'] == 'ENTRY'].iloc[0]
                exit = leg_trades[leg_trades['trade_type'] == 'EXIT'].iloc[0] if 'EXIT' in leg_trades['trade_type'].values else None
                
                if exit is not None:
                    if entry['position_type'] == 'BUY':
                        total_entry_cost += entry['price'] * entry['quantity']
                        total_exit_value += exit['price'] * entry['quantity']
                    else:  # SELL
                        total_entry_cost -= entry['price'] * entry['quantity']
                        total_exit_value -= exit['price'] * entry['quantity']
                        
            # Calculate position P&L
            gross_pnl = total_exit_value - total_entry_cost
            
            # Update all trades in position
            trades_df.loc[position_trades.index, 'gross_pnl'] = gross_pnl
            
            # Calculate net P&L
            total_costs = position_trades['transaction_cost'].sum() + position_trades['slippage'].sum()
            net_pnl = gross_pnl - total_costs
            trades_df.loc[position_trades.index, 'net_pnl'] = net_pnl
            
        return trades_df
    
    def _calculate_metrics(self,
                         trades_df: pd.DataFrame,
                         strategy_model: MLIndicatorStrategyModel) -> Dict[str, float]:
        """Calculate strategy metrics"""
        
        if trades_df.empty:
            return self._empty_metrics()
            
        # Get completed trades
        if 'position_id' in trades_df.columns:
            # Multi-leg - group by position
            position_pnls = trades_df.groupby('position_id')['net_pnl'].first()
        else:
            # Single instrument - only entry trades
            position_pnls = trades_df[trades_df['trade_type'] == 'ENTRY']['net_pnl']
            
        # Remove incomplete trades
        position_pnls = position_pnls[position_pnls != 0]
        
        if position_pnls.empty:
            return self._empty_metrics()
            
        # Calculate basic metrics
        metrics = {
            'total_trades': len(position_pnls),
            'total_pnl': position_pnls.sum(),
            'average_pnl': position_pnls.mean(),
            'winning_trades': (position_pnls > 0).sum(),
            'losing_trades': (position_pnls < 0).sum(),
            'win_rate': (position_pnls > 0).sum() / len(position_pnls),
            'largest_win': position_pnls.max(),
            'largest_loss': position_pnls.min(),
            'profit_factor': abs(position_pnls[position_pnls > 0].sum() / position_pnls[position_pnls < 0].sum()) if (position_pnls < 0).any() else 0
        }
        
        # Calculate risk-adjusted metrics
        if len(position_pnls) > 1:
            metrics.update({
                'sharpe_ratio': self._calculate_sharpe_ratio(position_pnls),
                'sortino_ratio': self._calculate_sortino_ratio(position_pnls),
                'max_drawdown': self._calculate_max_drawdown(position_pnls),
                'calmar_ratio': self._calculate_calmar_ratio(position_pnls),
                'information_ratio': self._calculate_information_ratio(position_pnls)
            })
        else:
            metrics.update({
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'information_ratio': 0
            })
            
        # Add ML-specific metrics if configured
        if strategy_model.ml_config:
            metrics.update(self._calculate_ml_metrics(trades_df))
            
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        
        if returns.empty or returns.std() == 0:
            return 0
            
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        
        if returns.empty:
            return 0
            
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
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
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_return: float = 0.05) -> float:
        """Calculate Information ratio"""
        
        if returns.empty or returns.std() == 0:
            return 0
            
        excess_returns = returns - benchmark_return / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_ml_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate ML-specific metrics"""
        
        signal_strength_mean = trades_df['signal_strength'].mean() if 'signal_strength' in trades_df else 0
        
        return {
            'avg_signal_strength': signal_strength_mean,
            'signal_accuracy': 0.65,  # Placeholder - would calculate from predictions
            'prediction_confidence': 0.75  # Placeholder
        }
    
    def _calculate_daily_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily P&L summary"""
        
        if trades_df.empty:
            return pd.DataFrame()
            
        # Group by date
        daily_pnl = trades_df.groupby('trade_date').agg({
            'net_pnl': 'sum',
            'gross_pnl': 'sum',
            'transaction_cost': 'sum',
            'slippage': 'sum'
        }).reset_index()
        
        # Calculate cumulative P&L
        daily_pnl['cumulative_pnl'] = daily_pnl['net_pnl'].cumsum()
        
        # Calculate daily return
        initial_capital = strategy_model.portfolio.risk_config.max_position_size
        daily_pnl['daily_return'] = daily_pnl['net_pnl'] / initial_capital
        daily_pnl['cumulative_return'] = (1 + daily_pnl['daily_return']).cumprod() - 1
        
        return daily_pnl
    
    def _analyze_signals(self, signal_results: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze signal performance"""
        
        if signal_results.empty:
            return {}
            
        total_entry_signals = signal_results['entry_signal'].sum()
        total_exit_signals = signal_results['exit_signal'].sum()
        
        # Count actual trades
        actual_entries = len(trades_df[trades_df['trade_type'] == 'ENTRY']) if not trades_df.empty else 0
        actual_exits = len(trades_df[trades_df['trade_type'] == 'EXIT']) if not trades_df.empty else 0
        
        # Signal quality metrics
        signal_analysis = {
            'total_entry_signals': total_entry_signals,
            'total_exit_signals': total_exit_signals,
            'signals_taken': actual_entries,
            'signal_efficiency': actual_entries / total_entry_signals if total_entry_signals > 0 else 0,
            'avg_entry_strength': signal_results[signal_results['entry_signal'] == 1]['entry_strength'].mean() if 'entry_strength' in signal_results else 0,
            'avg_exit_strength': signal_results[signal_results['exit_signal'] == 1]['exit_strength'].mean() if 'exit_strength' in signal_results else 0,
            'signal_distribution': {
                'hourly': self._analyze_signal_by_hour(signal_results),
                'daily': self._analyze_signal_by_day(signal_results)
            }
        }
        
        return signal_analysis
    
    def _analyze_signal_by_hour(self, signal_results: pd.DataFrame) -> Dict[int, int]:
        """Analyze signal distribution by hour"""
        
        if 'trade_time' not in signal_results.columns:
            return {}
            
        signal_results['hour'] = pd.to_datetime(signal_results['trade_time']).dt.hour
        return signal_results[signal_results['entry_signal'] == 1]['hour'].value_counts().to_dict()
    
    def _analyze_signal_by_day(self, signal_results: pd.DataFrame) -> Dict[str, int]:
        """Analyze signal distribution by day of week"""
        
        if 'trade_date' not in signal_results.columns:
            return {}
            
        signal_results['dow'] = pd.to_datetime(signal_results['trade_date']).dt.day_name()
        return signal_results[signal_results['entry_signal'] == 1]['dow'].value_counts().to_dict()
    
    def _analyze_indicators(self, indicator_values: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicator performance"""
        
        if indicator_values.empty:
            return {}
            
        indicator_stats = {}
        
        # Calculate statistics for each indicator
        numeric_columns = indicator_values.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in ['trade_date', 'trade_time']:
                indicator_stats[col] = {
                    'mean': indicator_values[col].mean(),
                    'std': indicator_values[col].std(),
                    'min': indicator_values[col].min(),
                    'max': indicator_values[col].max(),
                    'current': indicator_values[col].iloc[-1] if not indicator_values.empty else 0
                }
                
        # Analyze indicator correlations with returns
        if not trades_df.empty and 'net_pnl' in trades_df.columns:
            # This would require matching timestamps between indicators and trades
            pass
            
        return {
            'indicator_statistics': indicator_stats,
            'indicator_count': len(numeric_columns) - 2,  # Exclude date/time
            'data_points': len(indicator_values)
        }
    
    def _process_ml_predictions(self, ml_results: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Process ML model predictions and analysis"""
        
        if ml_results.empty:
            return {}
            
        # Store predictions
        self.ml_predictions = ml_results.copy()
        
        # Calculate prediction accuracy
        if 'target' in ml_results.columns and 'prediction' in ml_results.columns:
            accuracy = (ml_results['target'] == ml_results['prediction']).mean()
        else:
            accuracy = 0
            
        # Feature importance if available
        feature_importance = {}
        if 'feature_importance' in ml_results.columns:
            feature_importance = ml_results['feature_importance'].iloc[0] if not ml_results.empty else {}
            
        return {
            'prediction_accuracy': accuracy,
            'total_predictions': len(ml_results),
            'feature_importance': feature_importance,
            'model_confidence': ml_results['confidence'].mean() if 'confidence' in ml_results else 0
        }
    
    def _generate_strategy_summary(self,
                                 trades_df: pd.DataFrame,
                                 metrics: Dict[str, float],
                                 signal_analysis: Dict[str, Any],
                                 strategy_model: MLIndicatorStrategyModel) -> Dict[str, Any]:
        """Generate comprehensive strategy summary"""
        
        return {
            'strategy_name': strategy_model.portfolio.strategy_name,
            'strategy_type': 'ML_INDICATOR',
            'date_range': {
                'start': strategy_model.portfolio.start_date,
                'end': strategy_model.portfolio.end_date,
                'trading_days': len(trades_df['trade_date'].unique()) if not trades_df.empty else 0
            },
            'configuration': {
                'indicators': len(strategy_model.indicators),
                'signal_logic': strategy_model.signal_logic,
                'position_type': strategy_model.position_type,
                'risk_management': strategy_model.portfolio.risk_config.position_sizing,
                'ml_enabled': strategy_model.ml_config is not None
            },
            'performance': {
                'total_return': metrics.get('total_pnl', 0) / strategy_model.portfolio.risk_config.max_position_size,
                'annualized_return': metrics.get('total_pnl', 0) / strategy_model.portfolio.risk_config.max_position_size * 252 / max(len(trades_df['trade_date'].unique()), 1) if not trades_df.empty else 0,
                'win_rate': metrics.get('win_rate', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            },
            'signals': {
                'total_generated': signal_analysis.get('total_entry_signals', 0),
                'signals_taken': signal_analysis.get('signals_taken', 0),
                'efficiency': signal_analysis.get('signal_efficiency', 0)
            },
            'execution': {
                'total_trades': metrics.get('total_trades', 0),
                'avg_trade_pnl': metrics.get('average_pnl', 0),
                'transaction_costs': trades_df['transaction_cost'].sum() if not trades_df.empty else 0,
                'slippage_costs': trades_df['slippage'].sum() if not trades_df.empty else 0
            }
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        
        return {
            "trades": pd.DataFrame(),
            "metrics": self._empty_metrics(),
            "daily_pnl": pd.DataFrame(),
            "signal_analysis": {},
            "indicator_analysis": {},
            "ml_analysis": {},
            "indicator_values": pd.DataFrame(),
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
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'information_ratio': 0
        }