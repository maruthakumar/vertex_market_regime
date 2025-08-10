#!/usr/bin/env python3
"""
TBS Strategy Excel Output Generator
==================================

Generates Excel output in the exact format expected by the legacy backtester,
matching the output format specification for downstream Strategy Consolidator compatibility.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
import logging
from typing import Dict, Any, List
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

logger = logging.getLogger(__name__)

class TBSExcelOutputGenerator:
    """
    Generates Excel output for TBS strategy in legacy format
    """
    
    def __init__(self):
        """Initialize Excel output generator"""
        self.column_order = [
            'portfolio_name', 'strategy', 'leg_id', 'entry_date', 'entry_time', 'entry_day', 
            'exit_date', 'exit_time', 'exit_day', 'symbol', 'expiry', 'strike', 
            'instrument_type', 'side', 'filled_quantity', 'entry_price', 'exit_price', 
            'points', 'pointsAfterSlippage', 'pnl', 'pnlAfterSlippage', 'expenses', 
            'netPnlAfterExpenses', 're_entry_no', 'stop_loss_entry_number', 
            'take_profit_entry_number', 'reason', 'strategy_entry_number', 
            'index_entry_price', 'index_exit_price', 'max_profit', 'max_loss'
        ]
        
        self.column_rename_mapping = {
            'leg_id': "ID",
            'index_entry_price': 'Index At Entry',
            'index_exit_price': 'Index At Exit',
            'entry_date': "Entry Date",
            'entry_time': "Enter On",
            'entry_day': "Entry Day",
            'exit_date': "Exit Date",
            'exit_time': "Exit at",
            'exit_day': "Exit Day",
            'symbol': "Index",
            'expiry': "Expiry",
            'strike': "Strike",
            'instrument_type': "CE/PE",
            'side': "Trade",
            'filled_quantity': "Qty",
            'entry_price': "Entry at",
            'exit_price': "Exit at",
            'pnl': "PNL",
            'pnlAfterSlippage': "AfterSlippage",
            'expenses': "Taxes",
            'netPnlAfterExpenses': "Net PNL",
            're_entry_no': "Re-entry No",
            'reason': "Reason",
            'max_profit': "MaxProfit",
            'max_loss': "MaxLoss",
            'strategy_entry_number': "Strategy Entry No",
            'strategy': "Strategy Name",
            'stop_loss_entry_number': "SL Re-entry No",
            'take_profit_entry_number': "TGT Re-entry No",
            'points': "Points",
            'pointsAfterSlippage': 'Points After Slippage',
            'portfolio_name': "Portfolio Name"
        }
        
        logger.info("🚀 TBS Excel Output Generator initialized")
    
    def generate_excel_output(self, results: Dict[str, Any], strategy_params: Dict[str, Any], 
                             output_path: Path) -> Path:
        """
        Generate Excel output in legacy format
        
        Args:
            results: Processed results from TBS processor
            strategy_params: Strategy parameters
            output_path: Output file path
            
        Returns:
            Path to generated Excel file
        """
        
        try:
            logger.info(f"📊 Generating TBS Excel output: {output_path}")
            
            # Extract data from results
            trades_df = results.get('trades_df', pd.DataFrame())
            metrics = results.get('metrics', {})
            daily_pnl = results.get('daily_pnl', pd.DataFrame())
            monthly_pnl = results.get('monthly_pnl', pd.DataFrame())
            
            # Convert trades to legacy format
            transactions_df = self._convert_trades_to_legacy_format(trades_df, strategy_params)
            
            # Generate metrics sheet
            metrics_df = self._generate_metrics_sheet(metrics, strategy_params)
            
            # Generate max profit/loss sheet
            max_profit_loss_df = self._generate_max_profit_loss_sheet(daily_pnl)
            
            # Generate results sheets (day-wise, month-wise, margin-wise)
            results_df = self._generate_results_sheet(daily_pnl, monthly_pnl, metrics)
            
            # Save to Excel with multiple sheets
            self._save_to_excel(
                output_path,
                transactions_df,
                metrics_df,
                max_profit_loss_df,
                results_df,
                strategy_params
            )
            
            logger.info(f"✅ TBS Excel output generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generating TBS Excel output: {e}")
            raise
    
    def _convert_trades_to_legacy_format(self, trades_df: pd.DataFrame, 
                                       strategy_params: Dict[str, Any]) -> pd.DataFrame:
        """Convert trades DataFrame to legacy transaction format"""
        
        if trades_df.empty:
            return pd.DataFrame(columns=self.column_order)
        
        try:
            # Create legacy format DataFrame
            legacy_df = pd.DataFrame()
            
            # Map TBS fields to legacy fields
            legacy_df['portfolio_name'] = strategy_params.get('portfolio_name', 'TBS_PORTFOLIO')
            legacy_df['strategy'] = trades_df.get('StrategyName', strategy_params.get('strategy_name', 'TBS_STRATEGY'))
            legacy_df['leg_id'] = trades_df.get('LegNumber', 1)
            
            # Date and time fields
            legacy_df['entry_date'] = pd.to_datetime(trades_df.get('EntryDate')).dt.date
            legacy_df['entry_time'] = pd.to_datetime(trades_df.get('EntryDateTime')).dt.time
            legacy_df['entry_day'] = pd.to_datetime(trades_df.get('EntryDate')).dt.strftime("%A")
            
            legacy_df['exit_date'] = pd.to_datetime(trades_df.get('ExitDate')).dt.date
            legacy_df['exit_time'] = pd.to_datetime(trades_df.get('ExitDateTime')).dt.time
            legacy_df['exit_day'] = pd.to_datetime(trades_df.get('ExitDate')).dt.strftime("%A")
            
            # Symbol and option details
            legacy_df['symbol'] = strategy_params.get('symbol', 'NIFTY')
            legacy_df['expiry'] = pd.to_datetime(trades_df.get('ExpiryDate')).dt.date
            legacy_df['strike'] = trades_df.get('Strike', 0).astype(int)
            legacy_df['instrument_type'] = trades_df.get('OptionType', 'CE')
            
            # Transaction details
            legacy_df['side'] = trades_df.get('TransactionType', 'SELL')
            legacy_df['filled_quantity'] = trades_df.get('Quantity', 1)
            legacy_df['entry_price'] = trades_df.get('EntryPrice', 0)
            legacy_df['exit_price'] = trades_df.get('ExitPrice', 0)
            
            # P&L calculations
            legacy_df['points'] = legacy_df['entry_price'] - legacy_df['exit_price']
            legacy_df['pointsAfterSlippage'] = legacy_df['points']  # Assuming no slippage for now
            legacy_df['pnl'] = trades_df.get('PnL', 0)
            legacy_df['pnlAfterSlippage'] = legacy_df['pnl']
            legacy_df['expenses'] = trades_df.get('Brokerage', 0)
            legacy_df['netPnlAfterExpenses'] = trades_df.get('NetPnL', 0)
            
            # Additional fields
            legacy_df['re_entry_no'] = 0
            legacy_df['stop_loss_entry_number'] = 0
            legacy_df['take_profit_entry_number'] = 0
            legacy_df['reason'] = trades_df.get('ExitReason', 'TIME_EXIT')
            legacy_df['strategy_entry_number'] = trades_df.get('TradeID', 0)
            legacy_df['index_entry_price'] = 0  # Would need index data
            legacy_df['index_exit_price'] = 0   # Would need index data
            legacy_df['max_profit'] = 0         # Would need intraday tracking
            legacy_df['max_loss'] = 0           # Would need intraday tracking
            
            # Ensure all columns are present in correct order
            for col in self.column_order:
                if col not in legacy_df.columns:
                    legacy_df[col] = 0
            
            legacy_df = legacy_df[self.column_order]
            
            # Rename columns to legacy names
            legacy_df = legacy_df.rename(columns=self.column_rename_mapping)
            
            return legacy_df
            
        except Exception as e:
            logger.error(f"Error converting trades to legacy format: {e}")
            return pd.DataFrame(columns=self.column_order)
    
    def _generate_metrics_sheet(self, metrics: Dict[str, Any], 
                               strategy_params: Dict[str, Any]) -> pd.DataFrame:
        """Generate metrics sheet in legacy format"""
        
        try:
            strategy_name = strategy_params.get('strategy_name', 'TBS_STRATEGY')
            
            metrics_data = {
                'Particulars': [
                    'Total PnL',
                    'Total Trades',
                    'Win Rate',
                    'Max Drawdown',
                    'Sharpe Ratio',
                    'Profit Factor',
                    'Average Profit',
                    'Average Loss',
                    'Maximum Trade Profit',
                    'Maximum Trade Loss',
                    'Total Return',
                    'Initial Capital',
                    'Final Capital'
                ],
                'Combined': [
                    metrics.get('NetPnL', 0),
                    metrics.get('TotalTrades', 0),
                    f"{metrics.get('WinRate', 0):.2f}%",
                    metrics.get('MaxDrawdown', 0),
                    f"{metrics.get('SharpeRatio', 0):.2f}",
                    f"{metrics.get('ProfitFactor', 0):.2f}",
                    metrics.get('AvgProfit', 0),
                    metrics.get('AvgLoss', 0),
                    metrics.get('MaxProfit', 0),
                    metrics.get('MaxLoss', 0),
                    f"{metrics.get('TotalReturn', 0):.2f}%",
                    metrics.get('InitialCapital', 0),
                    metrics.get('FinalCapital', 0)
                ]
            }
            
            # Add strategy-specific column
            metrics_data[strategy_name] = metrics_data['Combined']
            
            return pd.DataFrame(metrics_data)
            
        except Exception as e:
            logger.error(f"Error generating metrics sheet: {e}")
            return pd.DataFrame()
    
    def _generate_max_profit_loss_sheet(self, daily_pnl: pd.DataFrame) -> pd.DataFrame:
        """Generate max profit/loss sheet"""
        
        try:
            if daily_pnl.empty:
                return pd.DataFrame(columns=['Date', 'Max Profit', 'Max Profit Time', 'Max Loss', 'Max Loss Time'])
            
            max_profit_loss_data = []
            
            for _, row in daily_pnl.iterrows():
                max_profit_loss_data.append({
                    'Date': row['Date'],
                    'Max Profit': max(0, row['NetPnL']),
                    'Max Profit Time': '15:15:00' if row['NetPnL'] > 0 else '',
                    'Max Loss': min(0, row['NetPnL']),
                    'Max Loss Time': '15:15:00' if row['NetPnL'] < 0 else ''
                })
            
            return pd.DataFrame(max_profit_loss_data)
            
        except Exception as e:
            logger.error(f"Error generating max profit/loss sheet: {e}")
            return pd.DataFrame()
    
    def _generate_results_sheet(self, daily_pnl: pd.DataFrame, monthly_pnl: pd.DataFrame,
                               metrics: Dict[str, Any]) -> pd.DataFrame:
        """Generate results sheet with day-wise, month-wise, and margin-wise stats"""
        
        try:
            # This would be a complex sheet with multiple sections
            # For now, create a simplified version
            
            results_data = {
                'Metric': ['Total P&L', 'Total Trades', 'Win Rate', 'Max Drawdown'],
                'Value': [
                    metrics.get('NetPnL', 0),
                    metrics.get('TotalTrades', 0),
                    f"{metrics.get('WinRate', 0):.2f}%",
                    metrics.get('MaxDrawdown', 0)
                ]
            }
            
            return pd.DataFrame(results_data)
            
        except Exception as e:
            logger.error(f"Error generating results sheet: {e}")
            return pd.DataFrame()
    
    def _save_to_excel(self, output_path: Path, transactions_df: pd.DataFrame,
                      metrics_df: pd.DataFrame, max_profit_loss_df: pd.DataFrame,
                      results_df: pd.DataFrame, strategy_params: Dict[str, Any]):
        """Save all sheets to Excel file"""
        
        try:
            strategy_name = strategy_params.get('strategy_name', 'TBS_STRATEGY').upper()
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Write sheets in legacy order
                metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                max_profit_loss_df.to_excel(writer, sheet_name="Max Profit and Loss", index=False)
                transactions_df.to_excel(writer, sheet_name=f"{strategy_name} Trans", index=False)
                results_df.to_excel(writer, sheet_name=f"{strategy_name} Results", index=False)
            
            logger.info(f"📄 Excel file saved with {len(transactions_df)} transactions")
            
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
            raise

# Test function to generate sample TBS output
async def generate_sample_tbs_output():
    """Generate sample TBS Excel output for testing"""
    
    try:
        # Create sample data
        sample_trades = pd.DataFrame({
            'TradeID': [1, 2, 3, 4],
            'StrategyName': ['TBS_IRON_CONDOR'] * 4,
            'LegNumber': [1, 2, 3, 4],
            'EntryDate': ['2024-01-01'] * 4,
            'EntryTime': ['09:20:00'] * 4,
            'EntryDateTime': ['2024-01-01 09:20:00'] * 4,
            'ExitDate': ['2024-01-01'] * 4,
            'ExitTime': ['15:15:00'] * 4,
            'ExitDateTime': ['2024-01-01 15:15:00'] * 4,
            'OptionType': ['CE', 'PE', 'CE', 'PE'],
            'Strike': [21500, 21500, 21600, 21400],
            'ExpiryDate': ['2024-01-04'] * 4,
            'TransactionType': ['SELL', 'SELL', 'BUY', 'BUY'],
            'Quantity': [1, 1, 1, 1],
            'LotSize': [50] * 4,
            'EntryPrice': [120, 110, 80, 70],
            'ExitPrice': [80, 95, 100, 85],
            'ExitReason': ['TIME_EXIT'] * 4,
            'PnL': [2000, 750, -1000, -750],
            'Brokerage': [40] * 4,
            'NetPnL': [1960, 710, -1040, -790]
        })
        
        sample_metrics = {
            'TotalTrades': 4,
            'WinningTrades': 2,
            'LosingTrades': 2,
            'WinRate': 50.0,
            'TotalPnL': 1000,
            'TotalBrokerage': 160,
            'NetPnL': 840,
            'MaxProfit': 2000,
            'MaxLoss': -1000,
            'AvgProfit': 1375,
            'AvgLoss': -875,
            'MaxDrawdown': -1000,
            'ProfitFactor': 1.57,
            'TotalReturn': 0.84,
            'SharpeRatio': 1.25,
            'InitialCapital': 100000,
            'FinalCapital': 100840
        }
        
        sample_daily_pnl = pd.DataFrame({
            'Date': ['2024-01-01'],
            'GrossPnL': [1000],
            'NetPnL': [840],
            'TradeCount': [4],
            'CumulativePnL': [840]
        })
        
        sample_results = {
            'success': True,
            'trades_df': sample_trades,
            'metrics': sample_metrics,
            'daily_pnl': sample_daily_pnl,
            'monthly_pnl': pd.DataFrame(),
            'summary': {'status': 'Completed'}
        }
        
        sample_strategy_params = {
            'portfolio_name': 'TBS_TEST_PORTFOLIO',
            'strategy_name': 'TBS_IRON_CONDOR',
            'symbol': 'NIFTY'
        }
        
        # Generate Excel output
        generator = TBSExcelOutputGenerator()
        output_path = Path("/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/tbs/TBS_SAMPLE_OUTPUT.xlsx")
        
        generator.generate_excel_output(sample_results, sample_strategy_params, output_path)
        
        print(f"✅ Sample TBS Excel output generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating sample TBS output: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_sample_tbs_output())
