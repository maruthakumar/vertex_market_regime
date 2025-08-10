#!/usr/bin/env python3
"""
POS Strategy Excel Output Generator
==================================

Generates Excel output in the exact format expected by the legacy backtester,
matching the output format specification for downstream Strategy Consolidator compatibility.
Based on TBS strategy golden format with POS-specific adaptations.
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

class POSExcelOutputGenerator:
    """
    Generates Excel output for POS strategy in legacy format
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
        
        logger.info("🚀 POS Excel Output Generator initialized")
    
    def generate_excel_output(self, results: Dict[str, Any], strategy_params: Dict[str, Any], 
                             output_path: Path) -> Path:
        """
        Generate Excel output in legacy format
        
        Args:
            results: Processed results from POS processor
            strategy_params: Strategy parameters
            output_path: Output file path
            
        Returns:
            Path to generated Excel file
        """
        
        try:
            logger.info(f"📊 Generating POS Excel output: {output_path}")
            
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
            
            logger.info(f"✅ POS Excel output generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generating POS Excel output: {e}")
            raise
    
    def _convert_trades_to_legacy_format(self, trades_df: pd.DataFrame, 
                                       strategy_params: Dict[str, Any]) -> pd.DataFrame:
        """Convert POS trades DataFrame to legacy transaction format"""
        
        if trades_df.empty:
            return pd.DataFrame(columns=self.column_order)
        
        try:
            # Create legacy format DataFrame
            legacy_df = pd.DataFrame()
            
            # Map POS fields to legacy fields
            legacy_df['portfolio_name'] = strategy_params.get('portfolio_name', 'POS_PORTFOLIO')
            legacy_df['strategy'] = trades_df.get('StrategyName', strategy_params.get('strategy_name', 'POS_STRATEGY'))
            legacy_df['leg_id'] = trades_df.get('LegID', trades_df.get('LegNumber', 1))
            
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
            
            # Transaction details - Handle multi-leg positions
            legacy_df['side'] = trades_df.get('TransactionType', trades_df.get('PositionType', 'SELL'))
            legacy_df['filled_quantity'] = trades_df.get('Quantity', trades_df.get('Lots', 1))
            legacy_df['entry_price'] = trades_df.get('EntryPrice', 0)
            legacy_df['exit_price'] = trades_df.get('ExitPrice', 0)
            
            # P&L calculations - Enhanced for multi-leg strategies
            legacy_df['points'] = trades_df.get('Points', legacy_df['entry_price'] - legacy_df['exit_price'])
            legacy_df['pointsAfterSlippage'] = trades_df.get('PointsAfterSlippage', legacy_df['points'])
            legacy_df['pnl'] = trades_df.get('PnL', 0)
            legacy_df['pnlAfterSlippage'] = trades_df.get('PnLAfterSlippage', legacy_df['pnl'])
            legacy_df['expenses'] = trades_df.get('Expenses', trades_df.get('Brokerage', 0))
            legacy_df['netPnlAfterExpenses'] = trades_df.get('NetPnL', legacy_df['pnl'] - legacy_df['expenses'])
            
            # POS-specific fields
            legacy_df['re_entry_no'] = trades_df.get('ReEntryNumber', 0)
            legacy_df['stop_loss_entry_number'] = trades_df.get('StopLossNumber', 0)
            legacy_df['take_profit_entry_number'] = trades_df.get('TakeProfitNumber', 0)
            legacy_df['reason'] = trades_df.get('ExitReason', 'TIME_EXIT')
            legacy_df['strategy_entry_number'] = trades_df.get('StrategyEntryNumber', trades_df.get('TradeID', 0))
            
            # Index tracking
            legacy_df['index_entry_price'] = trades_df.get('IndexAtEntry', 0)
            legacy_df['index_exit_price'] = trades_df.get('IndexAtExit', 0)
            
            # Performance tracking
            legacy_df['max_profit'] = trades_df.get('MaxProfit', 0)
            legacy_df['max_loss'] = trades_df.get('MaxLoss', 0)
            
            # Ensure all columns are present in correct order
            for col in self.column_order:
                if col not in legacy_df.columns:
                    legacy_df[col] = 0
            
            legacy_df = legacy_df[self.column_order]
            
            # Rename columns to legacy names
            legacy_df = legacy_df.rename(columns=self.column_rename_mapping)
            
            return legacy_df
            
        except Exception as e:
            logger.error(f"Error converting POS trades to legacy format: {e}")
            return pd.DataFrame(columns=[self.column_rename_mapping.get(col, col) for col in self.column_order])
    
    def _generate_metrics_sheet(self, metrics: Dict[str, Any], 
                               strategy_params: Dict[str, Any]) -> pd.DataFrame:
        """Generate metrics sheet with performance statistics"""
        
        try:
            # Default metrics structure
            default_metrics = {
                'Total PnL': 0,
                'Total Trades': 0,
                'Win Rate': 0,
                'Max Drawdown': 0,
                'Sharpe Ratio': 0,
                'Profit Factor': 0,
                'Average Profit': 0,
                'Average Loss': 0,
                'Maximum Trade Profit': 0,
                'Maximum Trade Loss': 0,
                'Total Return': 0,
                'Initial Capital': strategy_params.get('initial_capital', 100000),
                'Final Capital': 0,
                'Number of Winning Trades': 0,
                'Number of Losing Trades': 0,
                'Largest Winning Trade': 0,
                'Largest Losing Trade': 0,
                'Average Trade': 0,
                'Total Commission': 0,
                'Risk-Adjusted Return': 0,
                'Calmar Ratio': 0
            }
            
            # Update with actual metrics if available
            for key in default_metrics.keys():
                if key in metrics:
                    default_metrics[key] = metrics[key]
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                'Particulars': list(default_metrics.keys()),
                'Combined': list(default_metrics.values())
            })
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error generating metrics sheet: {e}")
            return pd.DataFrame({
                'Particulars': ['Total PnL', 'Total Trades'],
                'Combined': [0, 0]
            })
    
    def _generate_max_profit_loss_sheet(self, daily_pnl: pd.DataFrame) -> pd.DataFrame:
        """Generate max profit and loss tracking sheet"""
        
        try:
            if daily_pnl.empty:
                return pd.DataFrame({
                    'Date': [],
                    'Daily PnL': [],
                    'Cumulative PnL': [],
                    'Max Profit': [],
                    'Max Loss': [],
                    'Drawdown': []
                })
            
            # Calculate cumulative metrics
            daily_pnl = daily_pnl.copy()
            daily_pnl['Cumulative PnL'] = daily_pnl['Daily PnL'].cumsum()
            daily_pnl['Max Profit'] = daily_pnl['Cumulative PnL'].cummax()
            daily_pnl['Drawdown'] = daily_pnl['Cumulative PnL'] - daily_pnl['Max Profit']
            daily_pnl['Max Loss'] = daily_pnl['Drawdown'].cummin()
            
            return daily_pnl[['Date', 'Daily PnL', 'Cumulative PnL', 'Max Profit', 'Max Loss', 'Drawdown']]
            
        except Exception as e:
            logger.error(f"Error generating max profit/loss sheet: {e}")
            return pd.DataFrame({
                'Date': [],
                'Daily PnL': [],
                'Cumulative PnL': [],
                'Max Profit': [],
                'Max Loss': [],
                'Drawdown': []
            })
    
    def _generate_results_sheet(self, daily_pnl: pd.DataFrame, monthly_pnl: pd.DataFrame,
                               metrics: Dict[str, Any]) -> pd.DataFrame:
        """Generate results summary sheet"""
        
        try:
            results_data = []
            
            # Overall summary
            results_data.append({
                'Category': 'Overall Performance',
                'Period': 'Total',
                'PnL': metrics.get('Total PnL', 0),
                'Trades': metrics.get('Total Trades', 0),
                'Win Rate': f"{metrics.get('Win Rate', 0):.2%}",
                'Max Drawdown': metrics.get('Max Drawdown', 0),
                'Sharpe Ratio': metrics.get('Sharpe Ratio', 0)
            })
            
            # Monthly breakdown if available
            if not monthly_pnl.empty:
                for _, month_data in monthly_pnl.iterrows():
                    results_data.append({
                        'Category': 'Monthly Performance',
                        'Period': month_data.get('Month', 'Unknown'),
                        'PnL': month_data.get('PnL', 0),
                        'Trades': month_data.get('Trades', 0),
                        'Win Rate': f"{month_data.get('Win Rate', 0):.2%}",
                        'Max Drawdown': month_data.get('Max Drawdown', 0),
                        'Sharpe Ratio': month_data.get('Sharpe Ratio', 0)
                    })
            
            # Strategy-specific metrics
            results_data.append({
                'Category': 'Strategy Metrics',
                'Period': 'Multi-Leg Performance',
                'PnL': metrics.get('Multi Leg PnL', 0),
                'Trades': metrics.get('Multi Leg Trades', 0),
                'Win Rate': f"{metrics.get('Multi Leg Win Rate', 0):.2%}",
                'Max Drawdown': metrics.get('Multi Leg Max Drawdown', 0),
                'Sharpe Ratio': metrics.get('Multi Leg Sharpe', 0)
            })
            
            return pd.DataFrame(results_data)
            
        except Exception as e:
            logger.error(f"Error generating results sheet: {e}")
            return pd.DataFrame({
                'Category': ['Overall Performance'],
                'Period': ['Total'],
                'PnL': [0],
                'Trades': [0],
                'Win Rate': ['0.00%'],
                'Max Drawdown': [0],
                'Sharpe Ratio': [0]
            })
    
    def _save_to_excel(self, output_path: Path, transactions_df: pd.DataFrame,
                      metrics_df: pd.DataFrame, max_profit_loss_df: pd.DataFrame,
                      results_df: pd.DataFrame, strategy_params: Dict[str, Any]) -> None:
        """Save all sheets to Excel file"""
        
        try:
            strategy_name = strategy_params.get('strategy_name', 'POS_STRATEGY')
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Write sheets in specific order with name length limits
                strategy_short = strategy_name[:20] if len(strategy_name) > 20 else strategy_name
                
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                max_profit_loss_df.to_excel(writer, sheet_name='Max Profit and Loss', index=False)
                transactions_df.to_excel(writer, sheet_name=f'{strategy_short} Trans', index=False)
                results_df.to_excel(writer, sheet_name=f'{strategy_short} Results', index=False)
            
            logger.info(f"📊 Excel file saved with {len(transactions_df)} transactions")
            
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
            raise
    
    def generate_sample_output(self, output_path: Path) -> Path:
        """Generate sample POS output for reference"""
        
        try:
            logger.info(f"📊 Generating POS sample output: {output_path}")
            
            # Create sample data
            sample_trades = pd.DataFrame({
                'Portfolio Name': ['POS_IRON_CONDOR'] * 4,
                'Strategy Name': ['Iron_Condor_Conservative'] * 4,
                'ID': [1, 2, 3, 4],
                'Entry Date': ['2024-01-15'] * 4,
                'Enter On': ['10:30:00'] * 4,
                'Entry Day': ['Monday'] * 4,
                'Exit Date': ['2024-01-18'] * 4,
                'Exit at': ['15:20:00'] * 4,
                'Exit Day': ['Thursday'] * 4,
                'Index': ['NIFTY'] * 4,
                'Expiry': ['2024-01-25'] * 4,
                'Strike': [21800, 21700, 22000, 22100],
                'CE/PE': ['PE', 'PE', 'CE', 'CE'],
                'Trade': ['SELL', 'BUY', 'SELL', 'BUY'],
                'Qty': [50] * 4,
                'Entry at': [85.50, 42.25, 78.75, 35.80],
                'Exit at': [12.30, 5.15, 8.45, 2.25],
                'Points': [73.20, 37.10, 70.30, 33.55],
                'Points After Slippage': [72.70, 36.85, 69.80, 33.30],
                'PNL': [3650, 1842.5, 3515, 1677.5],
                'AfterSlippage': [3635, 1842.5, 3490, 1665],
                'Taxes': [18.25, 9.21, 17.58, 8.39],
                'Net PNL': [3631.75, 1833.29, 3472.42, 1656.61],
                'Re-entry No': [0] * 4,
                'Reason': ['TIME_EXIT'] * 4,
                'MaxProfit': [4200, 2100, 4000, 1900],
                'MaxLoss': [850, 425, 800, 380],
                'Strategy Entry No': [1] * 4,
                'SL Re-entry No': [0] * 4,
                'TGT Re-entry No': [0] * 4,
                'Index At Entry': [21900] * 4,
                'Index At Exit': [21950] * 4
            })
            
            # Sample metrics
            sample_metrics = pd.DataFrame({
                'Particulars': [
                    'Total PnL', 'Total Trades', 'Win Rate', 'Max Drawdown',
                    'Sharpe Ratio', 'Profit Factor', 'Average Profit', 'Average Loss',
                    'Maximum Trade Profit', 'Maximum Trade Loss', 'Total Return',
                    'Initial Capital', 'Final Capital'
                ],
                'Combined': [
                    10594.07, 1, 100.0, -500.0,
                    2.45, 4.25, 10594.07, 0,
                    10594.07, 0, 10.59,
                    100000, 110594.07
                ]
            })
            
            # Sample daily P&L
            sample_daily_pnl = pd.DataFrame({
                'Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18'],
                'Daily PnL': [2500, 3200, 2800, 2094.07],
                'Cumulative PnL': [2500, 5700, 8500, 10594.07],
                'Max Profit': [2500, 5700, 8500, 10594.07],
                'Max Loss': [0, 0, 0, 0],
                'Drawdown': [0, 0, 0, 0]
            })
            
            # Sample results
            sample_results = pd.DataFrame({
                'Category': ['Overall Performance', 'Strategy Metrics'],
                'Period': ['Total', 'Multi-Leg Performance'],
                'PnL': [10594.07, 10594.07],
                'Trades': [1, 1],
                'Win Rate': ['100.00%', '100.00%'],
                'Max Drawdown': [0, 0],
                'Sharpe Ratio': [2.45, 2.45]
            })
            
            # Save to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                sample_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                sample_daily_pnl.to_excel(writer, sheet_name='Max Profit and Loss', index=False)
                sample_trades.to_excel(writer, sheet_name='Iron_Condor Trans', index=False)
                sample_results.to_excel(writer, sheet_name='Iron_Condor Results', index=False)
            
            logger.info(f"✅ POS sample output generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generating POS sample output: {e}")
            raise


def main():
    """Test the POS Excel output generator"""
    generator = POSExcelOutputGenerator()
    
    # Generate sample output
    output_path = Path("POS_SAMPLE_OUTPUT.xlsx")
    generator.generate_sample_output(output_path)
    
    print(f"✅ POS sample output generated: {output_path}")


if __name__ == "__main__":
    main()