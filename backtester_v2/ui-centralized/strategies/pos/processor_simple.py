"""
Simplified Processor for POS Strategy Results
Processes query results into trades and metrics
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, date, time
import uuid

from .models_simple import SimplePOSStrategy, SimpleLegModel, SimplePortfolioModel, TradeResult, BacktestResult

logger = logging.getLogger(__name__)


class SimplePOSProcessor:
    """Simplified processor for POS strategies"""
    
    def __init__(self):
        self.trade_counter = 0
    
    def process_results(self, df: pd.DataFrame, strategy: SimplePOSStrategy) -> BacktestResult:
        """Process query results into trades and metrics"""
        
        if df.empty:
            return BacktestResult(
                strategy_summary=strategy.get_summary(),
                trades=[],
                metrics={'error': 'No data returned from query'},
                errors=['No data returned from query']
            )
        
        trades = []
        daily_pnl = []
        
        # Process each day's data
        for idx, row in df.iterrows():
            # Create entry trades for the day
            day_trades = self._create_daily_trades(row, strategy)
            trades.extend(day_trades)
            
            # Calculate daily P&L (simplified - assumes positions closed EOD)
            daily_result = self._calculate_daily_pnl(row, strategy)
            daily_pnl.append(daily_result)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, daily_pnl, strategy)
        
        return BacktestResult(
            strategy_summary=strategy.get_summary(),
            trades=trades,
            metrics=metrics,
            daily_pnl=daily_pnl
        )
    
    def _create_daily_trades(self, row: pd.Series, strategy: SimplePOSStrategy) -> List[TradeResult]:
        """Create trades for a single day"""
        trades = []
        
        # Entry trades at open
        for leg in strategy.legs:
            # Entry trade
            entry_trade = self._create_trade(
                row=row,
                leg=leg,
                trade_type='ENTRY',
                trade_time=leg.entry_time
            )
            if entry_trade:
                trades.append(entry_trade)
            
            # Exit trade (simplified - at close)
            exit_trade = self._create_trade(
                row=row,
                leg=leg,
                trade_type='EXIT',
                trade_time=leg.exit_time,
                is_exit=True
            )
            if exit_trade:
                trades.append(exit_trade)
        
        return trades
    
    def _create_trade(self, row: pd.Series, leg: SimpleLegModel, 
                     trade_type: str, trade_time: time, is_exit: bool = False) -> Optional[TradeResult]:
        """Create a single trade"""
        
        # Get price and Greeks from row
        price_col = f"{leg.leg_name}_price"
        strike_col = f"{leg.leg_name}_strike"
        delta_col = f"{leg.leg_name}_delta"
        gamma_col = f"{leg.leg_name}_gamma"
        theta_col = f"{leg.leg_name}_theta"
        vega_col = f"{leg.leg_name}_vega"
        
        # Check if data exists
        if price_col not in row or pd.isna(row[price_col]):
            return None
        
        price = float(row[price_col])
        strike = float(row[strike_col]) if strike_col in row else 0
        
        # For exit trades, reverse the position
        if is_exit:
            position_type = 'SELL' if leg.position_type == 'BUY' else 'BUY'
        else:
            position_type = leg.position_type
        
        # Calculate premium (negative for buys, positive for sells)
        quantity = leg.lots * leg.lot_size
        if position_type == 'BUY':
            premium = -price * quantity
        else:
            premium = price * quantity
        
        # Calculate transaction costs
        transaction_cost = abs(premium) * 0.001  # 0.1%
        
        self.trade_counter += 1
        
        return TradeResult(
            trade_id=f"T{self.trade_counter:06d}",
            trade_date=row['trade_date'],
            trade_time=trade_time,
            trade_type=trade_type,
            leg_name=leg.leg_name,
            option_type=leg.option_type,
            position_type=position_type,
            strike_price=strike,
            expiry_date=row.get('expiry_date', row['trade_date']),
            quantity=quantity if position_type == 'BUY' else -quantity,
            price=price,
            premium=premium,
            transaction_cost=transaction_cost,
            underlying_price=float(row.get('spot', 0)),
            delta=float(row.get(delta_col, 0)) if delta_col in row else None,
            gamma=float(row.get(gamma_col, 0)) if gamma_col in row else None,
            theta=float(row.get(theta_col, 0)) if theta_col in row else None,
            vega=float(row.get(vega_col, 0)) if vega_col in row else None
        )
    
    def _calculate_daily_pnl(self, row: pd.Series, strategy: SimplePOSStrategy) -> Dict[str, Any]:
        """Calculate P&L for a single day"""
        
        # Get total premium from row
        total_premium = float(row.get('total_premium', 0))
        
        # Get Greeks
        net_delta = float(row.get('net_delta', 0))
        net_gamma = float(row.get('net_gamma', 0))
        net_theta = float(row.get('net_theta', 0))
        net_vega = float(row.get('net_vega', 0))
        
        # Calculate transaction costs
        total_transaction_cost = abs(total_premium) * strategy.portfolio.transaction_costs
        
        # Net P&L (simplified - assumes positions closed at same price)
        net_pnl = total_premium - total_transaction_cost
        
        return {
            'date': row['trade_date'],
            'spot': float(row.get('spot', 0)),
            'total_premium': total_premium,
            'transaction_cost': total_transaction_cost,
            'net_pnl': net_pnl,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'net_vega': net_vega
        }
    
    def _calculate_metrics(self, trades: List[TradeResult], 
                          daily_pnl: List[Dict], strategy: SimplePOSStrategy) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        if not daily_pnl:
            return {'error': 'No daily P&L data'}
        
        # Convert to DataFrame for easier calculation
        pnl_df = pd.DataFrame(daily_pnl)
        
        # Basic metrics
        total_trades = len(trades)
        entry_trades = [t for t in trades if t.trade_type == 'ENTRY']
        exit_trades = [t for t in trades if t.trade_type == 'EXIT']
        
        # P&L metrics
        total_pnl = pnl_df['net_pnl'].sum()
        avg_daily_pnl = pnl_df['net_pnl'].mean()
        
        # Win rate (simplified)
        winning_days = (pnl_df['net_pnl'] > 0).sum()
        total_days = len(pnl_df)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Risk metrics
        pnl_df['cumulative_pnl'] = pnl_df['net_pnl'].cumsum()
        pnl_df['running_max'] = pnl_df['cumulative_pnl'].cummax()
        pnl_df['drawdown'] = pnl_df['cumulative_pnl'] - pnl_df['running_max']
        max_drawdown = pnl_df['drawdown'].min()
        
        # Greek averages
        avg_delta = pnl_df['net_delta'].mean()
        avg_gamma = pnl_df['net_gamma'].mean()
        avg_theta = pnl_df['net_theta'].mean()
        avg_vega = pnl_df['net_vega'].mean()
        
        # Return metrics
        metrics = {
            'total_trades': total_trades,
            'entry_trades': len(entry_trades),
            'exit_trades': len(exit_trades),
            'total_days': total_days,
            'total_pnl': round(total_pnl, 2),
            'avg_daily_pnl': round(avg_daily_pnl, 2),
            'win_rate': round(win_rate, 4),
            'winning_days': winning_days,
            'losing_days': total_days - winning_days,
            'max_drawdown': round(max_drawdown, 2),
            'avg_delta': round(avg_delta, 2),
            'avg_gamma': round(avg_gamma, 4),
            'avg_theta': round(avg_theta, 2),
            'avg_vega': round(avg_vega, 2),
            'total_premium_collected': round(pnl_df['total_premium'].sum(), 2),
            'total_transaction_costs': round(pnl_df['transaction_cost'].sum(), 2)
        }
        
        # Calculate Sharpe ratio if we have enough data
        if len(pnl_df) > 1:
            daily_returns = pnl_df['net_pnl'] / strategy.portfolio.position_size_value
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
            metrics['sharpe_ratio'] = round(sharpe, 2)
        
        return metrics