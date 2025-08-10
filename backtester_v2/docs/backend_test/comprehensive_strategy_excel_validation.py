#!/usr/bin/env python3
"""
Comprehensive Strategy Excel Validation Script
Validates Excel input files for all trading strategies using pandas DataFrame analysis
"""

import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyExcelValidator:
    """Comprehensive Excel validation for all trading strategies"""
    
    def __init__(self, base_path: str = "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations/data/prod"):
        self.base_path = Path(base_path)
        self.validation_results = {}
        self.strategy_configs = {
            'TBS': {
                'files': ['TBS_CONFIG_PORTFOLIO_1.0.0.xlsx', 'TBS_CONFIG_STRATEGY_1.0.0.xlsx'],
                'expected_sheets': ['PortfolioSetting', 'GeneralParameter', 'LegParameter', 'StrategySetting']
            },
            'TV': {
                'files': ['TV_CONFIG_MASTER_1.0.0.xlsx', 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx', 
                         'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx', 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx',
                         'TV_CONFIG_SIGNALS_1.0.0.xlsx', 'TV_CONFIG_STRATEGY_1.0.0.xlsx'],
                'expected_sheets': ['TV Setting', 'TV Signals', 'PortfolioSetting', 'GeneralParameter', 'LegParameter']
            },
            'ORB': {
                'files': ['ORB_CONFIG_PORTFOLIO_1.0.0.xlsx', 'ORB_CONFIG_STRATEGY_1.0.0.xlsx'],
                'expected_sheets': ['PortfolioSetting', 'GeneralParameter', 'LegParameter']
            },
            'OI': {
                'files': ['OI_CONFIG_PORTFOLIO_1.0.0.xlsx', 'OI_CONFIG_STRATEGY_1.0.0.xlsx'],
                'expected_sheets': ['PortfolioSetting', 'GeneralParameter', 'LegParameter']
            },
            'POS': {
                'files': ['POS_CONFIG_PORTFOLIO_1.0.0.xlsx', 'POS_CONFIG_STRATEGY_1.0.0.xlsx', 'POS_CONFIG_ADJUSTMENT_1.0.0.xlsx'],
                'expected_sheets': ['PortfolioSetting', 'PositionalParameter', 'LegParameter', 'AdjustmentRules', 'MarketStructure', 'GreekLimits']
            },
            'ML': {
                'files': ['ML_CONFIG_PORTFOLIO_1.0.0.xlsx', 'ML_CONFIG_STRATEGY_1.0.0.xlsx', 'ML_CONFIG_INDICATORS_1.0.0.xlsx'],
                'expected_sheets': ['PortfolioSetting', 'GeneralParameter', 'LegParameter', 'IndicatorConfig']
            },
            'MR': {
                'files': ['MR_CONFIG_PORTFOLIO_1.0.0.xlsx', 'MR_CONFIG_STRATEGY_1.0.0.xlsx', 'MR_CONFIG_REGIME_1.0.0.xlsx', 'MR_CONFIG_OPTIMIZATION_1.0.0.xlsx'],
                'expected_sheets': ['PortfolioSetting', 'RegimeParameter', 'OptimizationParameter', 'MarketStructure']
            }
        }
    
    def validate_strategy_files(self, strategy: str) -> Dict[str, Any]:
        """Validate all Excel files for a specific strategy"""
        logger.info(f"🔍 Validating {strategy} strategy files...")
        
        strategy_path = self.base_path / strategy.lower()
        strategy_result = {
            'strategy': strategy,
            'files_found': [],
            'files_missing': [],
            'sheet_analysis': {},
            'parameter_counts': {},
            'data_validation': {},
            'total_parameters': 0
        }
        
        if not strategy_path.exists():
            logger.error(f"❌ Strategy directory not found: {strategy_path}")
            strategy_result['error'] = f"Directory not found: {strategy_path}"
            return strategy_result
        
        config = self.strategy_configs.get(strategy, {})
        expected_files = config.get('files', [])
        expected_sheets = config.get('expected_sheets', [])
        
        # Check file existence
        for file_name in expected_files:
            file_path = strategy_path / file_name
            if file_path.exists():
                strategy_result['files_found'].append(file_name)
                # Analyze Excel file
                file_analysis = self.analyze_excel_file(file_path, expected_sheets)
                strategy_result['sheet_analysis'][file_name] = file_analysis
                
                # Count parameters
                for sheet_name, sheet_data in file_analysis.get('sheets', {}).items():
                    param_count = len(sheet_data.get('columns', []))
                    strategy_result['parameter_counts'][f"{file_name}:{sheet_name}"] = param_count
                    strategy_result['total_parameters'] += param_count
            else:
                strategy_result['files_missing'].append(file_name)
                logger.warning(f"⚠️  Missing file: {file_name}")
        
        return strategy_result
    
    def analyze_excel_file(self, file_path: Path, expected_sheets: List[str]) -> Dict[str, Any]:
        """Analyze a single Excel file structure"""
        analysis = {
            'file_path': str(file_path),
            'sheets_found': [],
            'sheets_missing': [],
            'sheets': {},
            'file_accessible': True
        }
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            available_sheets = excel_file.sheet_names
            analysis['sheets_found'] = available_sheets
            
            # Check for expected sheets
            for expected_sheet in expected_sheets:
                if expected_sheet not in available_sheets:
                    analysis['sheets_missing'].append(expected_sheet)
            
            # Analyze each sheet
            for sheet_name in available_sheets:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    sheet_analysis = self.analyze_sheet_structure(df, sheet_name)
                    analysis['sheets'][sheet_name] = sheet_analysis
                except Exception as e:
                    logger.error(f"❌ Error reading sheet {sheet_name}: {e}")
                    analysis['sheets'][sheet_name] = {'error': str(e)}
            
        except Exception as e:
            logger.error(f"❌ Error reading Excel file {file_path}: {e}")
            analysis['file_accessible'] = False
            analysis['error'] = str(e)
        
        return analysis
    
    def analyze_sheet_structure(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Analyze the structure of a single sheet"""
        analysis = {
            'sheet_name': sheet_name,
            'rows': len(df),
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'data_types': {},
            'null_counts': {},
            'sample_data': {},
            'validation_issues': []
        }
        
        # Analyze data types and null values
        for col in df.columns:
            analysis['data_types'][col] = str(df[col].dtype)
            analysis['null_counts'][col] = int(df[col].isnull().sum())  # Convert to int for JSON serialization

            # Get sample data (first non-null value)
            non_null_values = df[col].dropna()
            if not non_null_values.empty:
                analysis['sample_data'][col] = str(non_null_values.iloc[0])
            else:
                analysis['sample_data'][col] = 'ALL_NULL'
                analysis['validation_issues'].append(f"Column '{col}' has all null values")
        
        # Check for completely empty rows
        empty_rows = int(df.isnull().all(axis=1).sum())  # Convert to int for JSON serialization
        if empty_rows > 0:
            analysis['validation_issues'].append(f"{empty_rows} completely empty rows found")
        
        return analysis
    
    def validate_all_strategies(self) -> Dict[str, Any]:
        """Validate Excel files for all strategies"""
        logger.info("🚀 Starting comprehensive strategy validation...")
        
        validation_summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'strategies_validated': [],
            'total_files_analyzed': 0,
            'total_parameters_found': 0,
            'validation_results': {}
        }
        
        for strategy in self.strategy_configs.keys():
            strategy_result = self.validate_strategy_files(strategy)
            validation_summary['strategies_validated'].append(strategy)
            validation_summary['total_files_analyzed'] += len(strategy_result['files_found'])
            validation_summary['total_parameters_found'] += strategy_result['total_parameters']
            validation_summary['validation_results'][strategy] = strategy_result
        
        return validation_summary
    
    def generate_validation_report(self, output_path: str = None) -> str:
        """Generate comprehensive validation report"""
        if output_path is None:
            output_path = f"strategy_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        validation_results = self.validate_all_strategies()
        
        # Save detailed results
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Generate summary report
        summary_lines = [
            "# 📊 COMPREHENSIVE STRATEGY EXCEL VALIDATION REPORT",
            f"**Generated:** {validation_results['validation_timestamp']}",
            f"**Strategies Validated:** {len(validation_results['strategies_validated'])}",
            f"**Total Files Analyzed:** {validation_results['total_files_analyzed']}",
            f"**Total Parameters Found:** {validation_results['total_parameters_found']}",
            "",
            "## 📋 STRATEGY SUMMARY",
            ""
        ]
        
        for strategy, result in validation_results['validation_results'].items():
            summary_lines.extend([
                f"### **{strategy} Strategy**",
                f"- **Files Found:** {len(result['files_found'])}/{len(self.strategy_configs[strategy]['files'])}",
                f"- **Total Parameters:** {result['total_parameters']}",
                f"- **Files:** {', '.join(result['files_found'])}",
                ""
            ])
            
            if result['files_missing']:
                summary_lines.append(f"- **⚠️  Missing Files:** {', '.join(result['files_missing'])}")
                summary_lines.append("")
        
        summary_report = "\n".join(summary_lines)
        
        # Save summary report
        summary_path = output_path.replace('.json', '_summary.md')
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"✅ Validation report saved: {output_path}")
        logger.info(f"✅ Summary report saved: {summary_path}")
        
        return summary_report

def main():
    """Main execution function"""
    validator = StrategyExcelValidator()
    
    # Generate comprehensive validation report
    report = validator.generate_validation_report()
    print(report)
    
    return validator

if __name__ == "__main__":
    validator = main()
