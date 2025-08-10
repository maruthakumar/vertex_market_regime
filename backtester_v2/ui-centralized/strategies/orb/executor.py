#!/usr/bin/env python3
"""
ORB Executor - Main executor class for ORB strategy
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date, time
import logging
from heavydb import connect

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from strategies.orb.parser import ORBParser
from strategies.orb.models import ORBSettingModel, ORBLegModel
from strategies.orb.processor import ORBProcessor

logger = logging.getLogger(__name__)


class ORBExecutor:
    """Main executor for ORB (Opening Range Breakout) strategies"""
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ORB executor
        
        Args:
            db_config: Database configuration
        """
        # Set up database connection
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'port': 6274,
                'user': 'admin',
                'password': 'HyperInteractive',
                'dbname': 'heavyai'
            }
        
        self.conn = connect(**db_config)
        self.parser = ORBParser()
        self.processor = ORBProcessor(self.conn)
        
        logger.info("ORB Executor initialized")
    
    def execute_orb_backtest(
        self,
        input_file: str,
        portfolio_file: str,
        start_date: date,
        end_date: date,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute ORB backtest
        
        Args:
            input_file: Path to ORB input Excel file
            portfolio_file: Path to portfolio configuration file
            start_date: Backtest start date
            end_date: Backtest end date
            output_file: Optional output file path
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info(f"Starting ORB backtest from {start_date} to {end_date}")
            
            # Step 1: Parse input files
            orb_data = self.parser.parse_orb_excel(input_file)
            strategies = orb_data['strategies']
            
            if not strategies:
                logger.warning("No strategies found in input file")
                return {'status': 'error', 'message': 'No strategies found'}
            
            logger.info(f"Loaded {len(strategies)} ORB strategies")
            
            # Step 2: Convert to models
            strategy_models = []
            for strategy_dict in strategies:
                # Create leg models
                leg_models = []
                for leg_dict in strategy_dict.get('legs', []):
                    leg_model = ORBLegModel(**leg_dict)
                    leg_models.append(leg_model)
                
                # Create strategy model
                strategy_dict['legs'] = leg_models
                strategy_model = ORBSettingModel(**strategy_dict)
                strategy_models.append(strategy_model)
            
            # Step 3: Process each strategy for each date
            all_results = []
            
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
            
            for strategy in strategy_models:
                logger.info(f"Processing strategy: {strategy.strategy_name}")
                strategy_results = []
                
                for trade_date in date_range:
                    # Check if it's a trading day
                    weekday = trade_date.weekday() + 1  # Monday=1, Sunday=7
                    if not strategy.is_trading_day(weekday):
                        continue
                    
                    # Process strategy for this date
                    result = self.processor.process_orb_strategy(
                        settings=strategy,
                        trade_date=trade_date.date()
                    )
                    
                    if result['trades']:  # Only add if trades were executed
                        strategy_results.extend(result['trades'])
                
                all_results.extend(strategy_results)
            
            # Step 4: Format results
            formatted_results = self._format_results(all_results)
            
            # Step 5: Save to output file if specified
            if output_file:
                self._save_results(formatted_results, output_file)
            
            # Step 6: Calculate summary statistics
            summary = self._calculate_summary(formatted_results)
            
            return {
                'status': 'success',
                'results': formatted_results,
                'summary': summary,
                'trade_count': len(all_results)
            }
            
        except Exception as e:
            logger.error(f"Error executing ORB backtest: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
        finally:
            # Clear processor state
            self.processor.clear_state()
    
    def _format_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Format results into DataFrame"""
        
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add calculated columns
        df['entry_datetime'] = pd.to_datetime(df['trade_date'].astype(str) + ' ' + df['entry_time'].astype(str))
        df['exit_datetime'] = pd.to_datetime(df['trade_date'].astype(str) + ' ' + df['exit_time'].astype(str))
        
        # Sort by entry time
        df = df.sort_values('entry_datetime')
        
        # Format columns
        columns_order = [
            'strategy_name', 'trade_date', 'leg_id', 'instrument', 'transaction',
            'strike', 'expiry', 'entry_time', 'entry_price', 'exit_time', 'exit_price',
            'lots', 'pnl', 'breakout_type', 'breakout_strength'
        ]
        
        # Reorder columns
        available_columns = [col for col in columns_order if col in df.columns]
        df = df[available_columns]
        
        return df
    
    def _save_results(self, results: pd.DataFrame, output_file: str):
        """Save results to file"""
        
        if results.empty:
            logger.warning("No results to save")
            return
        
        # Determine file format from extension
        ext = os.path.splitext(output_file)[1].lower()
        
        if ext == '.xlsx':
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main results
                results.to_excel(writer, sheet_name='Trades', index=False)
                
                # Summary by strategy
                strategy_summary = results.groupby('strategy_name').agg({
                    'pnl': ['sum', 'mean', 'std', 'count'],
                    'trade_date': ['min', 'max']
                }).round(2)
                strategy_summary.to_excel(writer, sheet_name='Strategy Summary')
                
                # Daily P&L
                daily_pnl = results.groupby('trade_date')['pnl'].sum().reset_index()
                daily_pnl.to_excel(writer, sheet_name='Daily PnL', index=False)
                
            logger.info(f"Results saved to {output_file}")
            
        elif ext == '.csv':
            results.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
        else:
            logger.warning(f"Unsupported output format: {ext}")
    
    def _calculate_summary(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        if results.empty:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_win': 0,
                'max_loss': 0,
                'sharpe_ratio': 0
            }
        
        # Basic statistics
        total_trades = len(results)
        total_pnl = results['pnl'].sum()
        
        # Win/Loss statistics
        winning_trades = results[results['pnl'] > 0]
        losing_trades = results[results['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        max_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
        max_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = results.groupby('trade_date')['pnl'].sum()
        if len(daily_returns) > 1:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'breakout_distribution': results['breakout_type'].value_counts().to_dict() if 'breakout_type' in results else {}
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute ORB Backtest')
    parser.add_argument('--input', '-i', required=True, help='Path to ORB input Excel file')
    parser.add_argument('--portfolio', '-p', required=True, help='Path to portfolio configuration file')
    parser.add_argument('--start-date', '-s', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', help='Output file path (Excel or CSV)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # Execute backtest
    executor = ORBExecutor()
    
    try:
        result = executor.execute_orb_backtest(
            input_file=args.input,
            portfolio_file=args.portfolio,
            start_date=start_date,
            end_date=end_date,
            output_file=args.output
        )
        
        if result['status'] == 'success':
            print(f"\nBacktest completed successfully!")
            print(f"Total trades: {result['trade_count']}")
            print(f"\nSummary:")
            for key, value in result['summary'].items():
                print(f"  {key}: {value}")
        else:
            print(f"\nBacktest failed: {result.get('message', 'Unknown error')}")
            
    finally:
        executor.close()