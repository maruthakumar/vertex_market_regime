#!/usr/bin/env python3
"""
Advanced Configuration Input Validator
=====================================

This module implements comprehensive validation for market regime configuration
files with detailed error reporting and suggestions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    ERROR = "error"      # Must fix - prevents operation
    WARNING = "warning"  # Should fix - may cause issues
    INFO = "info"       # Nice to fix - optimization suggestion


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: ValidationSeverity
    category: str
    sheet: Optional[str]
    field: Optional[str]
    message: str
    suggestion: Optional[str]
    value: Any = None
    expected: Any = None


class ConfigurationValidator:
    """Advanced configuration validator with comprehensive checks"""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.validation_rules = self._initialize_validation_rules()
        
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for each configuration aspect"""
        return {
            'indicator_rules': {
                'weight_sum': {'min': 0.95, 'max': 1.05},
                'weight_range': {'min': 0.0, 'max': 1.0},
                'lookback_range': {'min': 1, 'max': 500},
                'enabled_minimum': 3  # At least 3 indicators should be enabled
            },
            'straddle_rules': {
                'ema_periods': {'min': 5, 'max': 200},
                'vwap_periods': {'min': 10, 'max': 500},
                'volume_threshold': {'min': 0, 'max': 1000000}
            },
            'regime_rules': {
                'threshold_order': True,  # Thresholds must be in order
                'confidence_range': {'min': 0.0, 'max': 1.0},
                'min_regimes': 8,
                'max_regimes': 18
            },
            'timeframe_rules': {
                'min_timeframes': 2,
                'max_timeframes': 10,
                'valid_periods': [1, 3, 5, 10, 15, 30, 60]  # minutes
            },
            'greek_rules': {
                'delta_range': {'min': -1.0, 'max': 1.0},
                'gamma_range': {'min': 0.0, 'max': 10.0},
                'theta_range': {'min': -100.0, 'max': 0.0},
                'vega_range': {'min': 0.0, 'max': 100.0}
            }
        }
    
    def validate_excel_file(self, file_path: str) -> Tuple[bool, List[ValidationIssue], Dict[str, Any]]:
        """
        Comprehensive validation of Excel configuration file
        
        Returns:
            Tuple of (is_valid, issues, metadata)
        """
        self.issues = []
        metadata = {
            'file_path': file_path,
            'validation_timestamp': datetime.now().isoformat(),
            'sheets_analyzed': 0,
            'total_parameters': 0
        }
        
        try:
            # Load Excel file
            excel_data = pd.ExcelFile(file_path)
            sheets = excel_data.sheet_names
            metadata['sheets_analyzed'] = len(sheets)
            
            # 1. Validate sheet structure
            self._validate_sheet_structure(sheets)
            
            # 2. Validate each sheet's content
            sheet_data = {}
            for sheet in sheets:
                df = excel_data.parse(sheet)
                sheet_data[sheet] = df
                metadata['total_parameters'] += len(df) * len(df.columns)
            
            # 3. Validate IndicatorConfiguration
            if 'IndicatorConfiguration' in sheet_data:
                self._validate_indicators(sheet_data['IndicatorConfiguration'])
            
            # 4. Validate StraddleAnalysisConfig
            if 'StraddleAnalysisConfig' in sheet_data:
                self._validate_straddle_config(sheet_data['StraddleAnalysisConfig'])
            
            # 5. Validate DynamicWeightageConfig
            if 'DynamicWeightageConfig' in sheet_data:
                self._validate_dynamic_weights(sheet_data['DynamicWeightageConfig'])
            
            # 6. Validate MultiTimeframeConfig
            if 'MultiTimeframeConfig' in sheet_data:
                self._validate_timeframes(sheet_data['MultiTimeframeConfig'])
            
            # 7. Validate GreekSentimentConfig
            if 'GreekSentimentConfig' in sheet_data:
                self._validate_greek_config(sheet_data['GreekSentimentConfig'])
            
            # 8. Validate RegimeFormationConfig
            if 'RegimeFormationConfig' in sheet_data:
                self._validate_regime_config(sheet_data['RegimeFormationConfig'])
            
            # 9. Validate RegimeComplexityConfig
            if 'RegimeComplexityConfig' in sheet_data:
                self._validate_complexity_config(sheet_data['RegimeComplexityConfig'])
            
            # 10. Cross-sheet validation
            self._validate_cross_sheet_consistency(sheet_data)
            
            # Determine if configuration is valid
            error_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR)
            is_valid = error_count == 0
            
            metadata['error_count'] = error_count
            metadata['warning_count'] = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)
            metadata['info_count'] = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.INFO)
            
            return is_valid, self.issues, metadata
            
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category='file_error',
                sheet=None,
                field=None,
                message=f"Failed to read Excel file: {str(e)}",
                suggestion="Ensure the file is a valid Excel file and not corrupted"
            ))
            return False, self.issues, metadata
    
    def _validate_sheet_structure(self, sheets: List[str]):
        """Validate that all required sheets are present"""
        required_sheets = [
            'IndicatorConfiguration',
            'StraddleAnalysisConfig',
            'DynamicWeightageConfig',
            'MultiTimeframeConfig',
            'GreekSentimentConfig',
            'RegimeFormationConfig',
            'RegimeComplexityConfig'
        ]
        
        for sheet in required_sheets:
            if sheet not in sheets:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='structure',
                    sheet=None,
                    field=sheet,
                    message=f"Required sheet '{sheet}' is missing",
                    suggestion=f"Add the '{sheet}' sheet to your Excel file"
                ))
        
        # Check for extra sheets
        extra_sheets = [s for s in sheets if s not in required_sheets]
        if extra_sheets:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category='structure',
                sheet=None,
                field='extra_sheets',
                message=f"Extra sheets found: {', '.join(extra_sheets)}",
                suggestion="Remove unnecessary sheets to keep the file clean",
                value=extra_sheets
            ))
    
    def _validate_indicators(self, df: pd.DataFrame):
        """Validate indicator configuration"""
        sheet = 'IndicatorConfiguration'
        
        # Check required columns
        required_columns = ['Indicator', 'Enabled', 'Weight', 'Lookback']
        for col in required_columns:
            if col not in df.columns:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='columns',
                    sheet=sheet,
                    field=col,
                    message=f"Required column '{col}' is missing",
                    suggestion=f"Add '{col}' column to {sheet} sheet"
                ))
                return
        
        # Validate weights
        if 'Weight' in df.columns:
            total_weight = df[df['Enabled'] == True]['Weight'].sum() if 'Enabled' in df.columns else df['Weight'].sum()
            
            if not (self.validation_rules['indicator_rules']['weight_sum']['min'] <= 
                   total_weight <= 
                   self.validation_rules['indicator_rules']['weight_sum']['max']):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='parameters',
                    sheet=sheet,
                    field='Weight',
                    message=f"Total weight of enabled indicators is {total_weight:.3f}, should be close to 1.0",
                    suggestion="Adjust indicator weights so they sum to 1.0",
                    value=total_weight,
                    expected=1.0
                ))
            
            # Check individual weights
            for idx, row in df.iterrows():
                weight = row['Weight']
                if not (self.validation_rules['indicator_rules']['weight_range']['min'] <= 
                       weight <= 
                       self.validation_rules['indicator_rules']['weight_range']['max']):
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category='parameters',
                        sheet=sheet,
                        field=f"Weight[{row['Indicator']}]",
                        message=f"Weight {weight} is outside valid range [0, 1]",
                        suggestion=f"Set weight for {row['Indicator']} between 0 and 1",
                        value=weight,
                        expected="[0, 1]"
                    ))
        
        # Check minimum enabled indicators
        if 'Enabled' in df.columns:
            enabled_count = df[df['Enabled'] == True].shape[0]
            if enabled_count < self.validation_rules['indicator_rules']['enabled_minimum']:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='parameters',
                    sheet=sheet,
                    field='Enabled',
                    message=f"Only {enabled_count} indicators are enabled, recommend at least {self.validation_rules['indicator_rules']['enabled_minimum']}",
                    suggestion="Enable more indicators for better market analysis",
                    value=enabled_count,
                    expected=f">= {self.validation_rules['indicator_rules']['enabled_minimum']}"
                ))
    
    def _validate_straddle_config(self, df: pd.DataFrame):
        """Validate straddle analysis configuration"""
        sheet = 'StraddleAnalysisConfig'
        
        # Check EMA periods
        if 'EMA_Period' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get('EMA_Period')):
                    period = row['EMA_Period']
                    rules = self.validation_rules['straddle_rules']['ema_periods']
                    if not (rules['min'] <= period <= rules['max']):
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category='parameters',
                            sheet=sheet,
                            field=f"EMA_Period[{row.get('Straddle_Type', idx)}]",
                            message=f"EMA period {period} is outside recommended range [{rules['min']}, {rules['max']}]",
                            suggestion=f"Use EMA period between {rules['min']} and {rules['max']}",
                            value=period,
                            expected=f"[{rules['min']}, {rules['max']}]"
                        ))
    
    def _validate_dynamic_weights(self, df: pd.DataFrame):
        """Validate dynamic weightage configuration"""
        sheet = 'DynamicWeightageConfig'
        
        # Check learning rate
        if 'Learning_Rate' in df.columns and len(df) > 0:
            lr = df.iloc[0].get('Learning_Rate', 0)
            if not (0 < lr <= 1):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='parameters',
                    sheet=sheet,
                    field='Learning_Rate',
                    message=f"Learning rate {lr} must be between 0 and 1",
                    suggestion="Set learning rate between 0.01 and 0.5 for optimal performance",
                    value=lr,
                    expected="(0, 1]"
                ))
    
    def _validate_timeframes(self, df: pd.DataFrame):
        """Validate multi-timeframe configuration"""
        sheet = 'MultiTimeframeConfig'
        
        if 'Period_Minutes' in df.columns:
            periods = df['Period_Minutes'].tolist()
            valid_periods = self.validation_rules['timeframe_rules']['valid_periods']
            
            for period in periods:
                if period not in valid_periods:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category='parameters',
                        sheet=sheet,
                        field='Period_Minutes',
                        message=f"Period {period} minutes is non-standard",
                        suggestion=f"Consider using standard periods: {valid_periods}",
                        value=period,
                        expected=valid_periods
                    ))
    
    def _validate_greek_config(self, df: pd.DataFrame):
        """Validate Greek sentiment configuration"""
        sheet = 'GreekSentimentConfig'
        
        # Validate Greek ranges
        greek_columns = {
            'Delta': self.validation_rules['greek_rules']['delta_range'],
            'Gamma': self.validation_rules['greek_rules']['gamma_range'],
            'Theta': self.validation_rules['greek_rules']['theta_range'],
            'Vega': self.validation_rules['greek_rules']['vega_range']
        }
        
        for greek, range_rules in greek_columns.items():
            if f'{greek}_Weight' in df.columns:
                # Check if weights are reasonable
                for idx, row in df.iterrows():
                    weight = row.get(f'{greek}_Weight', 0)
                    if weight < 0 or weight > 1:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category='parameters',
                            sheet=sheet,
                            field=f"{greek}_Weight",
                            message=f"{greek} weight {weight} should be between 0 and 1",
                            suggestion=f"Adjust {greek} weight to be between 0 and 1",
                            value=weight,
                            expected="[0, 1]"
                        ))
    
    def _validate_regime_config(self, df: pd.DataFrame):
        """Validate regime formation configuration"""
        sheet = 'RegimeFormationConfig'
        
        # Check regime count
        regime_count = len(df)
        if regime_count < self.validation_rules['regime_rules']['min_regimes']:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category='parameters',
                sheet=sheet,
                field='regime_count',
                message=f"Only {regime_count} regimes defined, minimum is {self.validation_rules['regime_rules']['min_regimes']}",
                suggestion="Add more regime definitions for better market coverage",
                value=regime_count,
                expected=f">= {self.validation_rules['regime_rules']['min_regimes']}"
            ))
        
        # Validate threshold ordering
        if 'Min_Threshold' in df.columns and 'Max_Threshold' in df.columns:
            for idx, row in df.iterrows():
                if row['Min_Threshold'] >= row['Max_Threshold']:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category='logic',
                        sheet=sheet,
                        field=f"Threshold[{row.get('Regime_Name', idx)}]",
                        message=f"Min threshold ({row['Min_Threshold']}) must be less than Max threshold ({row['Max_Threshold']})",
                        suggestion="Ensure Min_Threshold < Max_Threshold for each regime",
                        value=f"Min={row['Min_Threshold']}, Max={row['Max_Threshold']}"
                    ))
    
    def _validate_complexity_config(self, df: pd.DataFrame):
        """Validate regime complexity configuration"""
        sheet = 'RegimeComplexityConfig'
        
        # Check adaptation settings
        if 'Enable_Adaptation' in df.columns and len(df) > 0:
            adaptation = df.iloc[0].get('Enable_Adaptation', False)
            if adaptation and 'Adaptation_Rate' in df.columns:
                rate = df.iloc[0].get('Adaptation_Rate', 0)
                if rate <= 0 or rate > 1:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category='parameters',
                        sheet=sheet,
                        field='Adaptation_Rate',
                        message=f"Adaptation rate {rate} should be between 0 and 1",
                        suggestion="Set adaptation rate between 0.01 and 0.1 for gradual learning",
                        value=rate,
                        expected="(0, 1]"
                    ))
    
    def _validate_cross_sheet_consistency(self, sheet_data: Dict[str, pd.DataFrame]):
        """Validate consistency across sheets"""
        
        # Check if regime names in RegimeFormationConfig match references in other sheets
        if 'RegimeFormationConfig' in sheet_data:
            regime_names = set(sheet_data['RegimeFormationConfig'].get('Regime_Name', []))
            
            # Check if complexity config references valid regimes
            if 'RegimeComplexityConfig' in sheet_data and 'Regime_Reference' in sheet_data['RegimeComplexityConfig'].columns:
                for ref in sheet_data['RegimeComplexityConfig']['Regime_Reference']:
                    if pd.notna(ref) and ref not in regime_names:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category='consistency',
                            sheet='RegimeComplexityConfig',
                            field='Regime_Reference',
                            message=f"Referenced regime '{ref}' not found in RegimeFormationConfig",
                            suggestion=f"Add '{ref}' to RegimeFormationConfig or update the reference",
                            value=ref
                        ))
    
    def generate_validation_report(self, issues: List[ValidationIssue]) -> str:
        """Generate a human-readable validation report"""
        report = []
        report.append("=" * 70)
        report.append("CONFIGURATION VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for i in issues if i.severity == ValidationSeverity.INFO)
        
        report.append("SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Issues: {len(issues)}")
        report.append(f"  - Errors: {error_count} (must fix)")
        report.append(f"  - Warnings: {warning_count} (should fix)")
        report.append(f"  - Info: {info_count} (suggestions)")
        report.append("")
        
        if error_count == 0:
            report.append("✅ Configuration is VALID (no errors found)")
        else:
            report.append("❌ Configuration is INVALID (errors found)")
        report.append("")
        
        # Group issues by severity
        for severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            severity_issues = [i for i in issues if i.severity == severity]
            if severity_issues:
                report.append(f"\n{severity.value.upper()}S ({len(severity_issues)})")
                report.append("-" * 50)
                
                for idx, issue in enumerate(severity_issues, 1):
                    report.append(f"\n{idx}. [{issue.category.upper()}] {issue.sheet or 'Global'}")
                    if issue.field:
                        report.append(f"   Field: {issue.field}")
                    report.append(f"   Issue: {issue.message}")
                    if issue.value is not None:
                        report.append(f"   Current: {issue.value}")
                    if issue.expected is not None:
                        report.append(f"   Expected: {issue.expected}")
                    if issue.suggestion:
                        report.append(f"   💡 Suggestion: {issue.suggestion}")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)


def validate_configuration_file(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate a configuration file and return results
    
    Returns:
        Tuple of (is_valid, report_text, metadata)
    """
    validator = ConfigurationValidator()
    is_valid, issues, metadata = validator.validate_excel_file(file_path)
    report = validator.generate_validation_report(issues)
    
    return is_valid, report, metadata


# Example usage
if __name__ == "__main__":
    # Test with a sample file
    test_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/sample_config.xlsx"
    
    if Path(test_file).exists():
        is_valid, report, metadata = validate_configuration_file(test_file)
        print(report)
        print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
    else:
        print(f"Test file not found: {test_file}")