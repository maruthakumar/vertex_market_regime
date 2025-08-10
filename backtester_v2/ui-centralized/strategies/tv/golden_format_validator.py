#!/usr/bin/env python3
"""
TV Golden Format Validator
Validates TV strategy output in golden format with all 6 sheets
Compares with legacy outputs for compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date, time
from pathlib import Path
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.golden_format.tv_formatter import TVGoldenFormatter
from core.golden_format.models import GoldenFormatData, StrategyData
from strategies.tv.parser import TVParser
from strategies.tv.signal_processor import SignalProcessor
from strategies.tv.processor import TVProcessor

logger = logging.getLogger(__name__)


class TVGoldenFormatValidator:
    """Validates TV golden format output"""
    
    def __init__(self):
        """Initialize validator"""
        self.formatter = TVGoldenFormatter()
        self.parser = TVParser()
        self.signal_processor = SignalProcessor()
        self.tv_processor = TVProcessor()
        
        # Expected sheets in golden format
        self.expected_sheets = [
            'Portfolio Parameters',
            'General Parameters', 
            'Leg Parameters',
            'Trades',
            'Results',
            'TV Setting',
            'Signals'
        ]
        
        # Optional sheets based on portfolio types
        self.optional_sheets = [
            'Long Portfolio',
            'Short Portfolio'
        ]
    
    def validate_golden_format_structure(self, golden_output: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate the structure of golden format output
        
        Args:
            golden_output: Dictionary of sheet name to DataFrame
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check required sheets
        for sheet in self.expected_sheets:
            if sheet not in golden_output:
                errors.append(f"Missing required sheet: {sheet}")
        
        # Validate each sheet structure
        if 'Portfolio Parameters' in golden_output:
            portfolio_errors = self._validate_portfolio_parameters(golden_output['Portfolio Parameters'])
            errors.extend(portfolio_errors)
        
        if 'General Parameters' in golden_output:
            general_errors = self._validate_general_parameters(golden_output['General Parameters'])
            errors.extend(general_errors)
        
        if 'Leg Parameters' in golden_output:
            leg_errors = self._validate_leg_parameters(golden_output['Leg Parameters'])
            errors.extend(leg_errors)
        
        if 'Trades' in golden_output:
            trade_errors = self._validate_trades(golden_output['Trades'])
            errors.extend(trade_errors)
        
        if 'Results' in golden_output:
            result_errors = self._validate_results(golden_output['Results'])
            errors.extend(result_errors)
        
        if 'TV Setting' in golden_output:
            tv_setting_errors = self._validate_tv_settings(golden_output['TV Setting'])
            errors.extend(tv_setting_errors)
        
        if 'Signals' in golden_output:
            signal_errors = self._validate_signals(golden_output['Signals'])
            errors.extend(signal_errors)
        
        return len(errors) == 0, errors
    
    def _validate_portfolio_parameters(self, df: pd.DataFrame) -> List[str]:
        """Validate Portfolio Parameters sheet"""
        errors = []
        
        required_columns = [
            'PortfolioName', 'StrategyType', 'StartDate', 'EndDate',
            'Capital', 'MaxPositions', 'MaxRisk', 'MaxDrawdown',
            'TransactionCost', 'Slippage', 'MarginRequired'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Portfolio Parameters missing column: {col}")
        
        if len(df) == 0:
            errors.append("Portfolio Parameters sheet is empty")
        
        return errors
    
    def _validate_general_parameters(self, df: pd.DataFrame) -> List[str]:
        """Validate General Parameters sheet"""
        errors = []
        
        # Should have at least Parameter and Value columns
        if 'Parameter' not in df.columns or 'Value' not in df.columns:
            errors.append("General Parameters must have 'Parameter' and 'Value' columns")
        
        # Check for TV-specific parameters
        tv_params = ['SignalSource', 'SignalDateFormat', 'UseDbExitTiming', 
                     'ExitSearchInterval', 'ExitPriceSource']
        
        if 'Parameter' in df.columns:
            params_present = df['Parameter'].tolist()
            for param in tv_params:
                if param not in params_present:
                    errors.append(f"General Parameters missing TV parameter: {param}")
        
        return errors
    
    def _validate_leg_parameters(self, df: pd.DataFrame) -> List[str]:
        """Validate Leg Parameters sheet"""
        errors = []
        
        required_columns = [
            'LegID', 'LegName', 'Instrument', 'OptionType',
            'StrikeSelection', 'Expiry', 'Lots', 'Direction'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Leg Parameters missing column: {col}")
        
        return errors
    
    def _validate_trades(self, df: pd.DataFrame) -> List[str]:
        """Validate Trades sheet"""
        errors = []
        
        required_columns = [
            'TradeDate', 'TradeTime', 'TradedStock', 'TransactionType',
            'Quantity', 'Price'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Trades missing column: {col}")
        
        # Check for leg columns (L1-L4)
        leg_columns = ['L1', 'L2', 'L3', 'L4']
        leg_found = any(col in df.columns for col in leg_columns)
        if not leg_found:
            errors.append("Trades sheet missing leg columns (L1-L4)")
        
        return errors
    
    def _validate_results(self, df: pd.DataFrame) -> List[str]:
        """Validate Results sheet"""
        errors = []
        
        required_columns = [
            'TradeDate', 'EntryTime', 'ExitTime', 'Profit', 'Profit%'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Results missing column: {col}")
        
        # Check for leg columns
        leg_columns = ['L1', 'L2', 'L3', 'L4']
        leg_found = any(col in df.columns for col in leg_columns)
        if not leg_found:
            errors.append("Results sheet missing leg columns (L1-L4)")
        
        return errors
    
    def _validate_tv_settings(self, df: pd.DataFrame) -> List[str]:
        """Validate TV Setting sheet"""
        errors = []
        
        required_columns = [
            'Name', 'Enabled', 'SignalFilePath', 'StartDate', 'EndDate',
            'SignalDateFormat', 'IntradaySqOffApplicable', 'IntradayExitTime',
            'TvExitApplicable', 'SlippagePercent'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"TV Setting missing column: {col}")
        
        if len(df) == 0:
            errors.append("TV Setting sheet is empty")
        
        return errors
    
    def _validate_signals(self, df: pd.DataFrame) -> List[str]:
        """Validate Signals sheet"""
        errors = []
        
        required_columns = ['Trade #', 'Type', 'Date/Time', 'Contracts']
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Signals missing column: {col}")
        
        if len(df) == 0:
            errors.append("Signals sheet is empty")
        
        return errors
    
    def generate_golden_format_from_config(self, 
                                         config_files: Dict[str, Path],
                                         backtest_results: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate golden format output from TV configuration files
        
        Args:
            config_files: Dictionary of file paths for 6-file hierarchy
            backtest_results: Optional backtest results to include
            
        Returns:
            Golden format output as dictionary of DataFrames
        """
        # Parse TV configuration
        tv_config_result = self.parser.parse_tv_settings(str(config_files['tv_master']))
        tv_config = tv_config_result['settings'][0]
        
        # Parse signals
        signals = self.parser.parse_signals(str(config_files['signals']), tv_config['signal_date_format'])
        
        # Process signals
        processed_signals = self.signal_processor.process_signals(signals, tv_config)
        
        # Create input data structure for formatter
        input_data = {
            'tv_excel': str(config_files['tv_master']),
            'signal_file': str(config_files['signals']),
            'tv_config': tv_config,
            'signals': signals,
            'processed_signals': processed_signals
        }
        
        # Generate backtest ID
        backtest_id = f"TV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create processed results structure
        if backtest_results:
            processed_results = backtest_results
        else:
            # Create mock results for validation
            processed_results = self._create_mock_results(processed_signals)
        
        # Generate golden format
        golden_output = self.formatter.create_golden_format(
            processed_results=processed_results,
            input_data=input_data,
            backtest_id=backtest_id
        )
        
        return golden_output
    
    def _create_mock_results(self, processed_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create mock results for testing golden format generation"""
        trades = []
        results = []
        
        for i, signal in enumerate(processed_signals):
            # Create mock trade
            trade = {
                'trade_date': signal['entry_date'],
                'trade_time': signal['entry_time'],
                'signal_id': signal['trade_no'],
                'signal_type': signal['signal_direction'],
                'symbol': 'NIFTY',
                'transaction_type': 'BUY' if signal['signal_direction'] == 'LONG' else 'SELL',
                'quantity': signal['lots'],
                'price': 20000.0 + (i * 100),  # Mock price
                'strike': 20000,
                'expiry': date(2024, 1, 25),
                'option_type': 'CE' if signal['signal_direction'] == 'LONG' else 'PE',
                'mtm': 0
            }
            trades.append(trade)
            
            # Create mock result
            result = {
                'trade_date': signal['entry_date'],
                'entry_time': signal['entry_time'],
                'exit_time': signal['exit_time'],
                'signal_id': signal['trade_no'],
                'pnl': 1000.0 * (i + 1),  # Mock P&L
                'net_pnl': 1000.0 * (i + 1),
                'strategy_name': 'TV'
            }
            results.append(result)
        
        return {
            'trades': pd.DataFrame(trades),
            'results': pd.DataFrame(results),
            'summary': {
                'total_trades': len(trades),
                'total_pnl': sum(r['pnl'] for r in results),
                'win_rate': 60.0,
                'sharpe_ratio': 1.5
            }
        }
    
    def compare_with_legacy_output(self, 
                                  golden_output: Dict[str, pd.DataFrame],
                                  legacy_output_path: str) -> Dict[str, Any]:
        """
        Compare golden format output with legacy output
        
        Args:
            golden_output: Generated golden format output
            legacy_output_path: Path to legacy Excel output
            
        Returns:
            Comparison results
        """
        comparison = {
            'matching_sheets': [],
            'missing_sheets': [],
            'extra_sheets': [],
            'data_differences': []
        }
        
        try:
            # Read legacy output
            legacy_sheets = pd.read_excel(legacy_output_path, sheet_name=None)
            
            # Compare sheet names
            golden_sheets = set(golden_output.keys())
            legacy_sheet_names = set(legacy_sheets.keys())
            
            comparison['matching_sheets'] = list(golden_sheets & legacy_sheet_names)
            comparison['missing_sheets'] = list(legacy_sheet_names - golden_sheets)
            comparison['extra_sheets'] = list(golden_sheets - legacy_sheet_names)
            
            # Compare data in matching sheets
            for sheet in comparison['matching_sheets']:
                golden_df = golden_output[sheet]
                legacy_df = legacy_sheets[sheet]
                
                # Compare shapes
                if golden_df.shape != legacy_df.shape:
                    comparison['data_differences'].append({
                        'sheet': sheet,
                        'issue': 'Shape mismatch',
                        'golden_shape': golden_df.shape,
                        'legacy_shape': legacy_df.shape
                    })
                
                # Compare column names
                golden_cols = set(golden_df.columns)
                legacy_cols = set(legacy_df.columns)
                
                if golden_cols != legacy_cols:
                    comparison['data_differences'].append({
                        'sheet': sheet,
                        'issue': 'Column mismatch',
                        'missing_cols': list(legacy_cols - golden_cols),
                        'extra_cols': list(golden_cols - legacy_cols)
                    })
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def save_golden_format(self, golden_output: Dict[str, pd.DataFrame], output_path: str):
        """Save golden format output to Excel file"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in golden_output.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Golden format output saved to: {output_path}")
    
    def generate_validation_report(self, 
                                  golden_output: Dict[str, pd.DataFrame],
                                  validation_results: Tuple[bool, List[str]],
                                  comparison_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_passed': validation_results[0],
            'validation_errors': validation_results[1],
            'sheet_summary': {},
            'data_quality': {},
            'comparison': comparison_results
        }
        
        # Add sheet summaries
        for sheet_name, df in golden_output.items():
            report['sheet_summary'][sheet_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
        
        # Add data quality metrics
        if 'Trades' in golden_output:
            trades_df = golden_output['Trades']
            report['data_quality']['trades'] = {
                'total_trades': len(trades_df),
                'unique_dates': trades_df['TradeDate'].nunique() if 'TradeDate' in trades_df else 0,
                'null_values': trades_df.isnull().sum().to_dict()
            }
        
        if 'Results' in golden_output:
            results_df = golden_output['Results']
            if 'Profit' in results_df:
                report['data_quality']['results'] = {
                    'total_pnl': results_df['Profit'].sum(),
                    'avg_pnl': results_df['Profit'].mean(),
                    'win_rate': (results_df['Profit'] > 0).mean() * 100,
                    'max_profit': results_df['Profit'].max(),
                    'max_loss': results_df['Profit'].min()
                }
        
        return report


def main():
    """Test golden format validation"""
    print("\n" + "="*80)
    print("TV GOLDEN FORMAT VALIDATION")
    print("="*80)
    
    # Initialize validator
    validator = TVGoldenFormatValidator()
    
    # Define real configuration files
    base_path = Path('../../configurations/data/prod/tv')
    config_files = {
        'tv_master': base_path / 'TV_CONFIG_MASTER_1.0.0.xlsx',
        'signals': base_path / 'TV_CONFIG_SIGNALS_1.0.0.xlsx',
        'portfolio_long': base_path / 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx',
        'portfolio_short': base_path / 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx',
        'portfolio_manual': base_path / 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx',
        'strategy': base_path / 'TV_CONFIG_STRATEGY_1.0.0.xlsx'
    }
    
    # Generate golden format
    print("\n🔄 Generating golden format output...")
    try:
        golden_output = validator.generate_golden_format_from_config(config_files)
        print(f"✅ Generated {len(golden_output)} sheets")
        
        for sheet_name in golden_output:
            print(f"   • {sheet_name}: {len(golden_output[sheet_name])} rows")
    
    except Exception as e:
        print(f"❌ Failed to generate golden format: {e}")
        return 1
    
    # Validate structure
    print("\n🔍 Validating golden format structure...")
    is_valid, errors = validator.validate_golden_format_structure(golden_output)
    
    if is_valid:
        print("✅ Golden format structure is valid")
    else:
        print(f"❌ Validation failed with {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   • {error}")
    
    # Save golden format
    output_path = f"golden_format_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    validator.save_golden_format(golden_output, output_path)
    print(f"\n💾 Golden format saved to: {output_path}")
    
    # Generate validation report
    report = validator.generate_validation_report(golden_output, (is_valid, errors))
    
    print("\n📊 Validation Report Summary:")
    print(f"   • Validation Passed: {report['validation_passed']}")
    print(f"   • Total Sheets: {len(report['sheet_summary'])}")
    print(f"   • Total Errors: {len(report['validation_errors'])}")
    
    if 'data_quality' in report and 'results' in report['data_quality']:
        results_quality = report['data_quality']['results']
        print(f"\n📈 Results Summary:")
        print(f"   • Total P&L: ₹{results_quality.get('total_pnl', 0):,.2f}")
        print(f"   • Win Rate: {results_quality.get('win_rate', 0):.1f}%")
    
    # Save report
    report_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Validation report saved to: {report_path}")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())