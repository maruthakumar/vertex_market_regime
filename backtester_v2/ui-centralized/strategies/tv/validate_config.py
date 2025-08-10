#!/usr/bin/env python3
"""
TV Configuration Validator
Validates complete 6-file hierarchy configuration
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, '.')

from tv_unified_config import TVHierarchicalConfiguration


class ConfigValidator:
    """Validate TV 6-file hierarchy configuration"""
    
    def __init__(self):
        self.validation_results = {
            'files': {},
            'cross_file': {},
            'warnings': [],
            'errors': [],
            'info': []
        }
    
    def validate_configuration_set(self, config_dir: str) -> Dict[str, Any]:
        """Validate complete configuration set"""
        print(f"\n🔍 Validating TV configuration in: {config_dir}")
        
        # Find configuration files
        config_files = self._find_config_files(Path(config_dir))
        
        # Validate individual files
        for file_type, file_path in config_files.items():
            if file_path:
                self._validate_file(file_type, file_path)
        
        # Cross-file validation
        self._validate_cross_file_references(config_files)
        
        # Date consistency
        self._validate_date_consistency(config_files)
        
        # Portfolio allocation
        self._validate_portfolio_allocation(config_files)
        
        # Generate summary
        self.validation_results['summary'] = self._generate_summary()
        
        return self.validation_results
    
    def _find_config_files(self, config_dir: Path) -> Dict[str, Optional[Path]]:
        """Find all configuration files in directory"""
        files = {
            'tv_master': None,
            'signals': None,
            'portfolio_long': None,
            'portfolio_short': None,
            'portfolio_manual': None,
            'strategy': None
        }
        
        # Pattern matching
        patterns = {
            'tv_master': '*MASTER*.xlsx',
            'signals': '*SIGNALS*.xlsx',
            'portfolio_long': '*PORTFOLIO_LONG*.xlsx',
            'portfolio_short': '*PORTFOLIO_SHORT*.xlsx',
            'portfolio_manual': '*PORTFOLIO_MANUAL*.xlsx',
            'strategy': '*STRATEGY*.xlsx'
        }
        
        for file_type, pattern in patterns.items():
            matches = list(config_dir.glob(pattern))
            if matches:
                files[file_type] = matches[0]  # Take first match
                self.validation_results['info'].append(f"Found {file_type}: {matches[0].name}")
            else:
                self.validation_results['errors'].append(f"Missing {file_type} file")
        
        return files
    
    def _validate_file(self, file_type: str, file_path: Path):
        """Validate individual file"""
        validation = {
            'exists': file_path.exists(),
            'readable': False,
            'has_required_sheets': False,
            'has_required_columns': False,
            'data_validation': {}
        }
        
        if not validation['exists']:
            self.validation_results['errors'].append(f"{file_type} file not found: {file_path}")
            self.validation_results['files'][file_type] = validation
            return
        
        try:
            # File-specific validation
            if file_type == 'tv_master':
                validation.update(self._validate_tv_master(file_path))
            elif file_type == 'signals':
                validation.update(self._validate_signals(file_path))
            elif file_type.startswith('portfolio'):
                validation.update(self._validate_portfolio(file_path, file_type))
            elif file_type == 'strategy':
                validation.update(self._validate_strategy(file_path))
            
            validation['readable'] = True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Error reading {file_type}: {e}")
            validation['readable'] = False
        
        self.validation_results['files'][file_type] = validation
    
    def _validate_tv_master(self, file_path: Path) -> Dict[str, Any]:
        """Validate TV master configuration"""
        result = {}
        
        # Check for Setting sheet
        xl_file = pd.ExcelFile(file_path)
        if 'Setting' not in xl_file.sheet_names:
            self.validation_results['errors'].append("TV Master missing 'Setting' sheet")
            result['has_required_sheets'] = False
            return result
        
        result['has_required_sheets'] = True
        
        # Read settings
        df = pd.read_excel(file_path, sheet_name='Setting')
        
        # Check required columns
        required_cols = ['Name', 'Enabled', 'SignalFilePath', 'StartDate', 'EndDate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.validation_results['errors'].append(f"TV Master missing columns: {missing_cols}")
            result['has_required_columns'] = False
        else:
            result['has_required_columns'] = True
        
        # Data validation
        if not df.empty:
            row = df.iloc[0]
            
            # Check dates
            try:
                start_date = self._parse_date(row.get('StartDate'))
                end_date = self._parse_date(row.get('EndDate'))
                
                if start_date and end_date:
                    if start_date > end_date:
                        self.validation_results['errors'].append("Start date after end date")
                    result['date_range'] = {
                        'start': str(start_date),
                        'end': str(end_date)
                    }
            except:
                self.validation_results['warnings'].append("Could not parse dates in TV Master")
            
            # Check file references
            result['referenced_files'] = {
                'signal': row.get('SignalFilePath', ''),
                'long_portfolio': row.get('LongPortfolioFilePath', ''),
                'short_portfolio': row.get('ShortPortfolioFilePath', ''),
                'manual_portfolio': row.get('ManualPortfolioFilePath', '')
            }
        
        return result
    
    def _validate_signals(self, file_path: Path) -> Dict[str, Any]:
        """Validate signals file"""
        result = {}
        
        # Find signal sheet
        xl_file = pd.ExcelFile(file_path)
        signal_sheet = None
        
        for sheet in xl_file.sheet_names:
            if 'trade' in sheet.lower() or 'signal' in sheet.lower():
                signal_sheet = sheet
                break
        
        if not signal_sheet and xl_file.sheet_names:
            signal_sheet = xl_file.sheet_names[0]
        
        result['signal_sheet'] = signal_sheet
        result['has_required_sheets'] = True
        
        # Read signals
        df = pd.read_excel(file_path, sheet_name=signal_sheet)
        
        # Check required columns
        required_cols = ['Trade #', 'Type', 'Date/Time', 'Contracts']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.validation_results['errors'].append(f"Signals missing columns: {missing_cols}")
            result['has_required_columns'] = False
        else:
            result['has_required_columns'] = True
        
        # Data validation
        result['signal_count'] = len(df)
        result['unique_trades'] = df['Trade #'].nunique() if 'Trade #' in df.columns else 0
        
        # Check for paired signals
        if 'Type' in df.columns:
            signal_types = df['Type'].value_counts().to_dict()
            result['signal_types'] = signal_types
            
            # Simple pairing check
            entries = sum(1 for t in signal_types if 'entry' in str(t).lower())
            exits = sum(1 for t in signal_types if 'exit' in str(t).lower())
            
            if entries != exits:
                self.validation_results['warnings'].append(f"Unpaired signals: {entries} entries, {exits} exits")
        
        return result
    
    def _validate_portfolio(self, file_path: Path, portfolio_type: str) -> Dict[str, Any]:
        """Validate portfolio configuration"""
        result = {}
        
        # Check for PortfolioSetting sheet
        xl_file = pd.ExcelFile(file_path)
        if 'PortfolioSetting' not in xl_file.sheet_names:
            self.validation_results['errors'].append(f"{portfolio_type} missing 'PortfolioSetting' sheet")
            result['has_required_sheets'] = False
            return result
        
        result['has_required_sheets'] = True
        
        # Read settings
        df = pd.read_excel(file_path, sheet_name='PortfolioSetting')
        
        # Check required columns
        required_cols = ['Capital', 'MaxRisk', 'MaxPositions']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.validation_results['warnings'].append(f"{portfolio_type} missing columns: {missing_cols}")
            result['has_required_columns'] = False
        else:
            result['has_required_columns'] = True
        
        # Data validation
        if not df.empty:
            row = df.iloc[0]
            
            capital = row.get('Capital', 0)
            if capital <= 0:
                self.validation_results['errors'].append(f"{portfolio_type} has invalid capital: {capital}")
            
            result['capital'] = capital
            result['max_risk'] = row.get('MaxRisk', 0)
            result['max_positions'] = row.get('MaxPositions', 0)
        
        return result
    
    def _validate_strategy(self, file_path: Path) -> Dict[str, Any]:
        """Validate TBS strategy configuration"""
        result = {}
        
        # Check required sheets
        xl_file = pd.ExcelFile(file_path)
        required_sheets = ['GeneralParameter', 'LegParameter']
        missing_sheets = [s for s in required_sheets if s not in xl_file.sheet_names]
        
        if missing_sheets:
            self.validation_results['errors'].append(f"Strategy missing sheets: {missing_sheets}")
            result['has_required_sheets'] = False
            return result
        
        result['has_required_sheets'] = True
        
        # Read general parameters
        general_df = pd.read_excel(file_path, sheet_name='GeneralParameter')
        if not general_df.empty:
            result['strategy_name'] = general_df.iloc[0].get('StrategyName', 'Unknown')
            result['index'] = general_df.iloc[0].get('Index', 'NIFTY')
        
        # Read leg parameters
        legs_df = pd.read_excel(file_path, sheet_name='LegParameter')
        result['leg_count'] = len(legs_df)
        result['has_required_columns'] = True
        
        if result['leg_count'] == 0:
            self.validation_results['errors'].append("Strategy has no leg definitions")
        
        return result
    
    def _validate_cross_file_references(self, config_files: Dict[str, Optional[Path]]):
        """Validate cross-file references"""
        # Get TV master references
        tv_master_validation = self.validation_results['files'].get('tv_master', {})
        referenced_files = tv_master_validation.get('referenced_files', {})
        
        # Check if referenced files exist
        if referenced_files:
            # Signal file
            signal_ref = referenced_files.get('signal', '')
            if signal_ref and config_files.get('signals'):
                if Path(signal_ref).name != config_files['signals'].name:
                    self.validation_results['warnings'].append(
                        f"Signal file mismatch: TV Master references '{signal_ref}' "
                        f"but found '{config_files['signals'].name}'"
                    )
            
            # Portfolio files
            for portfolio_type in ['long', 'short', 'manual']:
                ref_key = f'{portfolio_type}_portfolio'
                file_key = f'portfolio_{portfolio_type}'
                
                portfolio_ref = referenced_files.get(ref_key, '')
                if portfolio_ref and config_files.get(file_key):
                    if Path(portfolio_ref).name != config_files[file_key].name:
                        self.validation_results['warnings'].append(
                            f"{portfolio_type.capitalize()} portfolio mismatch: "
                            f"TV Master references '{portfolio_ref}' "
                            f"but found '{config_files[file_key].name}'"
                        )
        
        self.validation_results['cross_file']['references_valid'] = len(
            [w for w in self.validation_results['warnings'] if 'mismatch' in w]
        ) == 0
    
    def _validate_date_consistency(self, config_files: Dict[str, Optional[Path]]):
        """Validate date consistency across files"""
        # Get date range from TV master
        tv_master_validation = self.validation_results['files'].get('tv_master', {})
        date_range = tv_master_validation.get('date_range', {})
        
        if date_range:
            self.validation_results['cross_file']['date_range'] = date_range
            
            # TODO: Check if signal dates fall within range
            # This would require parsing the signals which is already done elsewhere
    
    def _validate_portfolio_allocation(self, config_files: Dict[str, Optional[Path]]):
        """Validate portfolio capital allocation"""
        total_capital = 0
        portfolio_capitals = {}
        
        for portfolio_type in ['long', 'short', 'manual']:
            file_key = f'portfolio_{portfolio_type}'
            portfolio_validation = self.validation_results['files'].get(file_key, {})
            capital = portfolio_validation.get('capital', 0)
            
            if capital > 0:
                portfolio_capitals[portfolio_type] = capital
                total_capital += capital
        
        self.validation_results['cross_file']['total_capital'] = total_capital
        self.validation_results['cross_file']['portfolio_capitals'] = portfolio_capitals
        
        if total_capital == 0:
            self.validation_results['errors'].append("No capital allocated across portfolios")
    
    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if pd.isna(date_str) or not date_str:
            return None
        
        # Try DD_MM_YYYY format
        try:
            if isinstance(date_str, str) and '_' in date_str:
                parts = date_str.split('_')
                if len(parts) == 3:
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    return datetime(year, month, day)
        except:
            pass
        
        # Try pandas parsing
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        total_files = len([f for f, v in self.validation_results['files'].items() if v.get('exists', False)])
        total_errors = len(self.validation_results['errors'])
        total_warnings = len(self.validation_results['warnings'])
        
        return {
            'files_found': total_files,
            'files_expected': 6,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'is_valid': total_errors == 0 and total_files == 6,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def print_report(self):
        """Print formatted validation report"""
        print("\n" + "="*80)
        print("TV CONFIGURATION VALIDATION REPORT")
        print("="*80)
        
        summary = self.validation_results['summary']
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   • Files found: {summary['files_found']}/{summary['files_expected']}")
        print(f"   • Errors: {summary['total_errors']}")
        print(f"   • Warnings: {summary['total_warnings']}")
        print(f"   • Valid: {'✅ Yes' if summary['is_valid'] else '❌ No'}")
        
        # File validation
        print(f"\n📁 File Validation:")
        for file_type, validation in self.validation_results['files'].items():
            if validation.get('exists'):
                status = "✅" if validation.get('readable') else "❌"
                print(f"   {status} {file_type}:")
                print(f"      • Sheets: {'✓' if validation.get('has_required_sheets') else '✗'}")
                print(f"      • Columns: {'✓' if validation.get('has_required_columns') else '✗'}")
                
                # File-specific info
                if file_type == 'signals':
                    print(f"      • Signals: {validation.get('signal_count', 0)}")
                    print(f"      • Trades: {validation.get('unique_trades', 0)}")
                elif 'portfolio' in file_type:
                    capital = validation.get('capital', 0)
                    print(f"      • Capital: ₹{capital:,}")
                elif file_type == 'strategy':
                    print(f"      • Legs: {validation.get('leg_count', 0)}")
            else:
                print(f"   ❌ {file_type}: Not found")
        
        # Cross-file validation
        cross_file = self.validation_results.get('cross_file', {})
        if cross_file:
            print(f"\n🔗 Cross-File Validation:")
            print(f"   • References valid: {'✓' if cross_file.get('references_valid') else '✗'}")
            
            total_capital = cross_file.get('total_capital', 0)
            print(f"   • Total capital: ₹{total_capital:,}")
            
            if 'date_range' in cross_file:
                dr = cross_file['date_range']
                print(f"   • Date range: {dr.get('start')} to {dr.get('end')}")
        
        # Errors
        if self.validation_results['errors']:
            print(f"\n❌ Errors ({len(self.validation_results['errors'])}):")
            for error in self.validation_results['errors']:
                print(f"   • {error}")
        
        # Warnings
        if self.validation_results['warnings']:
            print(f"\n⚠️  Warnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                print(f"   • {warning}")
        
        # Info
        if self.validation_results['info']:
            print(f"\nℹ️  Information:")
            for info in self.validation_results['info']:
                print(f"   • {info}")
        
        print("\n" + "="*80)
        
        # Final verdict
        if summary['is_valid']:
            print("✅ CONFIGURATION IS VALID AND READY FOR USE")
        else:
            print("❌ CONFIGURATION HAS ERRORS THAT MUST BE FIXED")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Validate TV 6-file configuration hierarchy')
    parser.add_argument('--config-dir', required=True, help='Directory containing configuration files')
    parser.add_argument('--output-json', help='Save validation results to JSON file')
    
    args = parser.parse_args()
    
    # Create validator
    validator = ConfigValidator()
    
    # Validate configuration
    results = validator.validate_configuration_set(args.config_dir)
    
    # Print report
    validator.print_report()
    
    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Validation results saved to: {args.output_json}")
    
    # Exit with error code if invalid
    return 0 if results['summary']['is_valid'] else 1


if __name__ == "__main__":
    sys.exit(main())