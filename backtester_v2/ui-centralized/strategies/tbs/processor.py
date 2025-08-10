#!/usr/bin/env python3
"""
TBS Processor - Handles processing of TBS query results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, time
import logging

logger = logging.getLogger(__name__)


class TBSProcessor:
    """Processes results from TBS strategy queries"""
    
    def __init__(self):
        self.trade_id_counter = 0
        
    def process_results(self, query_results: List[pd.DataFrame], 
                       strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query results into final backtest output
        
        Args:
            query_results: List of DataFrames from executed queries
            strategy_params: Strategy parameters including portfolio and leg info
            
        Returns:
            Dictionary containing processed results
        """
        if not query_results:
            logger.warning("No query results to process")
            return self._empty_results()
        
        # Extract parameters
        portfolio_params = strategy_params.get('portfolio', {})
        legs = strategy_params.get('legs', [])
        initial_capital = portfolio_params.get('capital', 1000000)
        
        # Process trades for each leg
        all_trades = []
        for i, (leg, leg_results) in enumerate(zip(legs, query_results)):
            if leg_results.empty:
                logger.warning(f"No results for leg {i+1}")
                continue
            
            leg_trades = self._process_leg_trades(
                leg_results=leg_results,
                leg_params=leg,
                leg_number=i+1,
                strategy_name=leg.get('strategy_name', f'Strategy_{i+1}')
            )
            all_trades.extend(leg_trades)
        
        if not all_trades:
            logger.warning("No trades generated from results")
            return self._empty_results()
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(all_trades)
        
        # Sort by entry time
        trades_df = trades_df.sort_values('EntryDateTime').reset_index(drop=True)
        
        # Calculate cumulative P&L
        trades_df['CumulativePnL'] = trades_df['PnL'].cumsum()
        trades_df['Capital'] = initial_capital + trades_df['CumulativePnL']
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades_df, initial_capital)
        
        # Calculate daily P&L
        daily_pnl = self._calculate_daily_pnl(trades_df)
        
        # Calculate monthly P&L
        monthly_pnl = self._calculate_monthly_pnl(trades_df)
        
        return {
            'success': True,
            'trades_df': trades_df,
            'metrics': metrics,
            'daily_pnl': daily_pnl,
            'monthly_pnl': monthly_pnl,
            'summary': self._generate_summary(trades_df, metrics)
        }
    
    def _process_leg_trades(self, leg_results: pd.DataFrame, leg_params: Dict[str, Any],
                           leg_number: int, strategy_name: str) -> List[Dict[str, Any]]:
        """Process trades for a single leg"""
        trades = []
        
        # Get leg parameters
        option_type = leg_params.get('option_type', 'CE')
        quantity = leg_params.get('quantity', 1)
        transaction_type = leg_params.get('transaction_type', 'SELL')
        lot_size = leg_params.get('lot_size', 50)
        
        # Process each row as a trade
        for idx, row in leg_results.iterrows():
            self.trade_id_counter += 1
            
            trade = {
                'TradeID': self.trade_id_counter,
                'StrategyName': strategy_name,
                'LegNumber': leg_number,
                'EntryDate': row.get('trade_date'),
                'EntryTime': row.get('entry_time', time(9, 20)),
                'EntryDateTime': pd.Timestamp.combine(
                    row.get('trade_date'),
                    row.get('entry_time', time(9, 20))
                ),
                'ExitDate': row.get('trade_date'),  # Same day for intraday
                'ExitTime': row.get('exit_time', time(15, 15)),
                'ExitDateTime': pd.Timestamp.combine(
                    row.get('trade_date'),
                    row.get('exit_time', time(15, 15))
                ),
                'OptionType': option_type,
                'Strike': row.get('strike'),
                'ExpiryDate': row.get('expiry_date'),
                'TransactionType': transaction_type,
                'Quantity': quantity,
                'LotSize': lot_size,
                'EntryPrice': row.get('entry_price', 0),
                'ExitPrice': row.get('exit_price', 0),
                'ExitReason': row.get('exit_reason', 'TIME_EXIT'),
            }
            
            # Calculate P&L
            if transaction_type == 'SELL':
                trade['PnL'] = (trade['EntryPrice'] - trade['ExitPrice']) * quantity * lot_size
            else:
                trade['PnL'] = (trade['ExitPrice'] - trade['EntryPrice']) * quantity * lot_size
            
            # Calculate brokerage and net P&L
            trade['Brokerage'] = self._calculate_brokerage(trade)
            trade['NetPnL'] = trade['PnL'] - trade['Brokerage']
            
            trades.append(trade)
        
        return trades
    
    def _calculate_brokerage(self, trade: Dict[str, Any]) -> float:
        """Calculate brokerage for a trade"""
        # Simple brokerage calculation - can be enhanced
        entry_brokerage = 20  # Rs 20 per lot
        exit_brokerage = 20
        
        total_lots = trade['Quantity']
        return (entry_brokerage + exit_brokerage) * total_lots
    
    def _calculate_metrics(self, trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Calculate strategy metrics"""
        if trades_df.empty:
            return self._empty_metrics()
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PnL'] > 0])
        losing_trades = len(trades_df[trades_df['PnL'] < 0])
        
        total_pnl = trades_df['PnL'].sum()
        total_brokerage = trades_df['Brokerage'].sum()
        net_pnl = trades_df['NetPnL'].sum()
        
        max_profit = trades_df['PnL'].max() if not trades_df.empty else 0
        max_loss = trades_df['PnL'].min() if not trades_df.empty else 0
        
        avg_profit = trades_df[trades_df['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['PnL'] < 0]['PnL'].mean() if losing_trades > 0 else 0
        
        # Calculate drawdown
        cumulative_pnl = trades_df['CumulativePnL']
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max)
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        # Calculate ratios
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = abs(avg_profit * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
        
        # Calculate returns
        total_return = (net_pnl / initial_capital * 100) if initial_capital > 0 else 0
        
        # Calculate Sharpe ratio (simplified - daily)
        if len(trades_df) > 1:
            daily_returns = trades_df.groupby('EntryDate')['PnL'].sum() / initial_capital
            sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'TotalTrades': total_trades,
            'WinningTrades': winning_trades,
            'LosingTrades': losing_trades,
            'WinRate': win_rate,
            'TotalPnL': total_pnl,
            'TotalBrokerage': total_brokerage,
            'NetPnL': net_pnl,
            'MaxProfit': max_profit,
            'MaxLoss': max_loss,
            'AvgProfit': avg_profit,
            'AvgLoss': avg_loss,
            'MaxDrawdown': max_drawdown,
            'ProfitFactor': profit_factor,
            'TotalReturn': total_return,
            'SharpeRatio': sharpe_ratio,
            'InitialCapital': initial_capital,
            'FinalCapital': initial_capital + net_pnl
        }
    
    def _calculate_daily_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily P&L summary"""
        if trades_df.empty:
            return pd.DataFrame()
        
        daily_pnl = trades_df.groupby('EntryDate').agg({
            'PnL': 'sum',
            'NetPnL': 'sum',
            'TradeID': 'count'
        }).reset_index()
        
        daily_pnl.columns = ['Date', 'GrossPnL', 'NetPnL', 'TradeCount']
        daily_pnl['CumulativePnL'] = daily_pnl['NetPnL'].cumsum()
        
        return daily_pnl
    
    def _calculate_monthly_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly P&L summary"""
        if trades_df.empty:
            return pd.DataFrame()
        
        # Add month column
        trades_df['Month'] = pd.to_datetime(trades_df['EntryDate']).dt.to_period('M')
        
        monthly_pnl = trades_df.groupby('Month').agg({
            'PnL': 'sum',
            'NetPnL': 'sum',
            'TradeID': 'count'
        }).reset_index()
        
        monthly_pnl.columns = ['Month', 'GrossPnL', 'NetPnL', 'TradeCount']
        monthly_pnl['CumulativePnL'] = monthly_pnl['NetPnL'].cumsum()
        
        # Convert Month back to string for display
        monthly_pnl['Month'] = monthly_pnl['Month'].astype(str)
        
        return monthly_pnl
    
    def _generate_summary(self, trades_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy summary"""
        if trades_df.empty:
            return {'status': 'No trades executed'}
        
        return {
            'status': 'Completed',
            'total_trades': metrics['TotalTrades'],
            'net_pnl': metrics['NetPnL'],
            'win_rate': f"{metrics['WinRate']:.2f}%",
            'max_drawdown': metrics['MaxDrawdown'],
            'sharpe_ratio': f"{metrics['SharpeRatio']:.2f}",
            'start_date': trades_df['EntryDate'].min(),
            'end_date': trades_df['EntryDate'].max(),
            'trading_days': trades_df['EntryDate'].nunique()
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'success': False,
            'trades_df': pd.DataFrame(),
            'metrics': self._empty_metrics(),
            'daily_pnl': pd.DataFrame(),
            'monthly_pnl': pd.DataFrame(),
            'summary': {'status': 'No results'}
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'TotalTrades': 0,
            'WinningTrades': 0,
            'LosingTrades': 0,
            'WinRate': 0,
            'TotalPnL': 0,
            'TotalBrokerage': 0,
            'NetPnL': 0,
            'MaxProfit': 0,
            'MaxLoss': 0,
            'AvgProfit': 0,
            'AvgLoss': 0,
            'MaxDrawdown': 0,
            'ProfitFactor': 0,
            'TotalReturn': 0,
            'SharpeRatio': 0,
            'InitialCapital': 0,
            'FinalCapital': 0
        }