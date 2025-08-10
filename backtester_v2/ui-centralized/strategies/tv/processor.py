#!/usr/bin/env python3
"""
TV Processor - Processes TV query results into backtest output
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TVProcessor:
    """Processes TV query results into final output format"""
    
    def process_result(
        self,
        db_result: Any,
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single query result (simplified for testing)
        
        Args:
            db_result: Database query result
            signal: Processed signal
            
        Returns:
            Processed result dictionary
        """
        return {
            'trade_no': signal.get('trade_no', 'N/A'),
            'signal_direction': signal.get('signal_direction', 'N/A'),
            'pnl': 100.0,  # Mock P&L for testing
            'status': 'completed',
            'query_result': str(db_result) if db_result else 'No result'
        }
    
    def process_tv_results(
        self,
        query_results: List[pd.DataFrame],
        tv_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process TV query results into final output
        
        Args:
            query_results: List of DataFrames from query execution
            tv_params: TV parameters including settings and signals
            
        Returns:
            Dictionary containing processed results
        """
        
        # Combine all query results
        if not query_results:
            logger.warning("No query results to process")
            return self._empty_result()
        
        # Concatenate all results
        all_trades = pd.concat(query_results, ignore_index=True)
        
        if all_trades.empty:
            logger.warning("Combined results are empty")
            return self._empty_result()
        
        # Process trades
        processed_trades = self._process_trades(all_trades)
        
        # Calculate metrics
        metrics = self._calculate_metrics(processed_trades)
        
        # Format output
        output = {
            'success': True,
            'tv_name': tv_params.get('name', 'TV_Backtest'),
            'start_date': tv_params.get('start_date'),
            'end_date': tv_params.get('end_date'),
            'total_signals': len(tv_params.get('signals', [])),
            'total_trades': len(processed_trades),
            'trades': processed_trades.to_dict('records'),
            'metrics': metrics,
            'summary': self._generate_summary(processed_trades, metrics)
        }
        
        return output
    
    def _process_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Process individual trades"""
        
        # Ensure required columns exist
        required_columns = ['trade_no', 'entry_date', 'entry_time', 
                          'exit_date', 'exit_time', 'pnl']
        
        for col in required_columns:
            if col not in trades_df.columns:
                logger.warning(f"Missing required column: {col}")
                trades_df[col] = None
        
        # Sort by entry datetime
        if 'entry_date' in trades_df.columns and 'entry_time' in trades_df.columns:
            trades_df['entry_datetime'] = pd.to_datetime(
                trades_df['entry_date'].astype(str) + ' ' + 
                trades_df['entry_time'].astype(str)
            )
            trades_df = trades_df.sort_values('entry_datetime')
        
        # Calculate cumulative P&L
        if 'pnl' in trades_df.columns:
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        # Add trade duration
        if all(col in trades_df.columns for col in ['entry_datetime', 'exit_date', 'exit_time']):
            trades_df['exit_datetime'] = pd.to_datetime(
                trades_df['exit_date'].astype(str) + ' ' + 
                trades_df['exit_time'].astype(str)
            )
            trades_df['duration_minutes'] = (
                trades_df['exit_datetime'] - trades_df['entry_datetime']
            ).dt.total_seconds() / 60
        
        # Round numeric columns
        numeric_columns = ['pnl', 'cumulative_pnl', 'entry_price', 'exit_price']
        for col in numeric_columns:
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].round(2)
        
        return trades_df
    
    def _calculate_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics"""
        
        metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'total_wins': 0,
            'total_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
        if trades_df.empty or 'pnl' not in trades_df.columns:
            return metrics
        
        # Total P&L
        metrics['total_pnl'] = trades_df['pnl'].sum()
        
        # Win/Loss analysis
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        
        metrics['total_wins'] = len(wins)
        metrics['total_losses'] = len(losses)
        
        if len(trades_df) > 0:
            metrics['win_rate'] = (len(wins) / len(trades_df)) * 100
        
        if len(wins) > 0:
            metrics['avg_win'] = wins['pnl'].mean()
        
        if len(losses) > 0:
            metrics['avg_loss'] = losses['pnl'].mean()
        
        # Profit factor
        if len(losses) > 0 and losses['pnl'].sum() != 0:
            metrics['profit_factor'] = abs(wins['pnl'].sum() / losses['pnl'].sum())
        
        # Max drawdown
        if 'cumulative_pnl' in trades_df.columns:
            cumulative = trades_df['cumulative_pnl'].values
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            metrics['max_drawdown'] = -drawdown.max() if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified - daily returns)
        if len(trades_df) > 1:
            daily_returns = trades_df.groupby('entry_date')['pnl'].sum()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                metrics['sharpe_ratio'] = (
                    daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                )
        
        # Consecutive wins/losses
        if 'pnl' in trades_df.columns:
            win_loss = (trades_df['pnl'] > 0).astype(int)
            
            # Count consecutive wins
            win_groups = (win_loss != win_loss.shift()).cumsum()
            win_counts = win_loss.groupby(win_groups).sum()
            metrics['max_consecutive_wins'] = win_counts.max() if len(win_counts) > 0 else 0
            
            # Count consecutive losses
            loss_groups = ((1 - win_loss) != (1 - win_loss).shift()).cumsum()
            loss_counts = (1 - win_loss).groupby(loss_groups).sum()
            metrics['max_consecutive_losses'] = loss_counts.max() if len(loss_counts) > 0 else 0
        
        # Round metrics
        for key in ['total_pnl', 'avg_win', 'avg_loss', 'max_drawdown', 
                   'sharpe_ratio', 'profit_factor', 'win_rate']:
            metrics[key] = round(metrics[key], 2)
        
        return metrics
    
    def _generate_summary(
        self,
        trades_df: pd.DataFrame,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        summary = {
            'performance': self._get_performance_rating(metrics),
            'risk_level': self._get_risk_level(metrics),
            'consistency': self._get_consistency_rating(metrics),
            'recommendation': self._get_recommendation(metrics)
        }
        
        # Add date-wise summary if data available
        if not trades_df.empty and 'entry_date' in trades_df.columns:
            daily_pnl = trades_df.groupby('entry_date')['pnl'].sum()
            summary['daily_stats'] = {
                'best_day': {
                    'date': daily_pnl.idxmax() if len(daily_pnl) > 0 else None,
                    'pnl': daily_pnl.max() if len(daily_pnl) > 0 else 0
                },
                'worst_day': {
                    'date': daily_pnl.idxmin() if len(daily_pnl) > 0 else None,
                    'pnl': daily_pnl.min() if len(daily_pnl) > 0 else 0
                },
                'positive_days': len(daily_pnl[daily_pnl > 0]),
                'negative_days': len(daily_pnl[daily_pnl < 0])
            }
        
        return summary
    
    def _get_performance_rating(self, metrics: Dict[str, Any]) -> str:
        """Rate overall performance"""
        score = 0
        
        # Positive P&L
        if metrics['total_pnl'] > 0:
            score += 2
        
        # Good win rate
        if metrics['win_rate'] > 60:
            score += 2
        elif metrics['win_rate'] > 50:
            score += 1
        
        # Good profit factor
        if metrics['profit_factor'] > 2:
            score += 2
        elif metrics['profit_factor'] > 1.5:
            score += 1
        
        # Positive Sharpe
        if metrics['sharpe_ratio'] > 1:
            score += 1
        
        # Rating
        if score >= 6:
            return "Excellent"
        elif score >= 4:
            return "Good"
        elif score >= 2:
            return "Average"
        else:
            return "Poor"
    
    def _get_risk_level(self, metrics: Dict[str, Any]) -> str:
        """Assess risk level"""
        
        # Check drawdown
        if abs(metrics['max_drawdown']) > 20000:
            return "High"
        elif abs(metrics['max_drawdown']) > 10000:
            return "Medium"
        else:
            return "Low"
    
    def _get_consistency_rating(self, metrics: Dict[str, Any]) -> str:
        """Rate consistency"""
        
        # Check consecutive losses
        if metrics['max_consecutive_losses'] > 5:
            return "Poor"
        elif metrics['max_consecutive_losses'] > 3:
            return "Average"
        else:
            return "Good"
    
    def _get_recommendation(self, metrics: Dict[str, Any]) -> str:
        """Generate recommendation"""
        
        if metrics['total_pnl'] > 0 and metrics['win_rate'] > 50:
            if metrics['profit_factor'] > 1.5:
                return "Strategy shows promise. Consider live testing with small capital."
            else:
                return "Strategy is profitable but needs optimization."
        else:
            return "Strategy needs significant improvement before deployment."
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'success': False,
            'tv_name': 'TV_Backtest',
            'total_signals': 0,
            'total_trades': 0,
            'trades': [],
            'metrics': self._calculate_metrics(pd.DataFrame()),
            'summary': {
                'performance': "No Data",
                'risk_level': "Unknown",
                'consistency': "No Data",
                'recommendation': "No trades to analyze"
            }
        }