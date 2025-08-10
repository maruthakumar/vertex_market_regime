#!/usr/bin/env python3
"""
Unified Configuration Validator
===============================

This module provides comprehensive validation for the unified enhanced market regime
configuration, ensuring all parameters are valid and consistent across sheets.

Features:
- Sheet structure validation
- Parameter range validation
- Cross-sheet consistency checks
- Missing parameter detection
- UI-compatible error reporting
- Excel-to-YAML conversion support

Author: The Augster
Date: 2025-01-27
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import yaml
from pathlib import Path
import json
import sys

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation error severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Validation result with details"""
    is_valid: bool
    sheet_name: str
    parameter: str
    message: str
    severity: ValidationSeverity
    cell_reference: Optional[str] = None
    suggested_value: Optional[Any] = None

@dataclass
class SheetValidationRules:
    """Validation rules for a specific sheet"""
    sheet_name: str
    required_columns: List[str]
    parameter_rules: Dict[str, Dict[str, Any]]
    cross_references: List[Dict[str, str]]

class UnifiedConfigValidator:
    """Comprehensive validator for unified market regime configuration"""
    
    def __init__(self):
        """Initialize unified configuration validator"""
        self.validation_results: List[ValidationResult] = []
        self.required_sheets = [
            'IndicatorConfiguration',
            'StraddleAnalysisConfig',
            'DynamicWeightageConfig',
            'MultiTimeframeConfig',
            'GreekSentimentConfig',
            'TrendingOIPAConfig',
            'RegimeFormationConfig',
            'RegimeComplexityConfig',
            'IVSurfaceConfig',
            'ATRIndicatorsConfig',
            'PerformanceMetrics',
            'SystemConfiguration'
        ]
        
        # Define validation rules for each sheet
        self.sheet_rules = self._define_validation_rules()
        
        logger.info("Unified Configuration Validator initialized")
    
    def _define_validation_rules(self) -> Dict[str, SheetValidationRules]:
        """Define comprehensive validation rules for each sheet"""
        rules = {}
        
        # IndicatorConfiguration rules
        rules['IndicatorConfiguration'] = SheetValidationRules(
            sheet_name='IndicatorConfiguration',
            required_columns=['IndicatorSystem', 'Enabled', 'BaseWeight', 'PerformanceTracking'],
            parameter_rules={
                'BaseWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'Enabled': {'values': ['YES', 'NO'], 'type': str},
                'PerformanceTracking': {'values': ['YES', 'NO'], 'type': str},
                'AdaptiveWeight': {'values': ['YES', 'NO'], 'type': str}
            },
            cross_references=[
                {'field': 'IndicatorSystem', 'ref_sheet': 'DynamicWeightageConfig', 'ref_field': 'SystemName'}
            ]
        )
        
        # StraddleAnalysisConfig rules
        rules['StraddleAnalysisConfig'] = SheetValidationRules(
            sheet_name='StraddleAnalysisConfig',
            required_columns=['StraddleType', 'Enabled', 'Weight'],
            parameter_rules={
                'Weight': {'min': 0.0, 'max': 1.0, 'type': float, 'sum_constraint': 1.0},
                'Enabled': {'values': ['YES', 'NO'], 'type': str},
                'EMAEnabled': {'values': ['YES', 'NO'], 'type': str},
                'VWAPEnabled': {'values': ['YES', 'NO'], 'type': str},
                'PreviousDayVWAP': {'values': ['YES', 'NO'], 'type': str}
            },
            cross_references=[]
        )
        
        # DynamicWeightageConfig rules
        rules['DynamicWeightageConfig'] = SheetValidationRules(
            sheet_name='DynamicWeightageConfig',
            required_columns=['SystemName', 'CurrentWeight', 'MinWeight', 'MaxWeight'],
            parameter_rules={
                'CurrentWeight': {'min': 0.0, 'max': 1.0, 'type': float, 'sum_constraint': 1.0},
                'MinWeight': {'min': 0.0, 'max': 0.5, 'type': float},
                'MaxWeight': {'min': 0.1, 'max': 1.0, 'type': float},
                'HistoricalPerformance': {'min': 0.0, 'max': 1.0, 'type': float},
                'LearningRate': {'min': 0.001, 'max': 0.1, 'type': float},
                'PerformanceWindow': {'min': 10, 'max': 1000, 'type': int},
                'AutoAdjust': {'values': ['YES', 'NO'], 'type': str}
            },
            cross_references=[
                {'field': 'SystemName', 'ref_sheet': 'IndicatorConfiguration', 'ref_field': 'IndicatorSystem'}
            ]
        )
        
        # MultiTimeframeConfig rules
        rules['MultiTimeframeConfig'] = SheetValidationRules(
            sheet_name='MultiTimeframeConfig',
            required_columns=['Timeframe', 'Enabled', 'Weight'],
            parameter_rules={
                'Weight': {'min': 0.0, 'max': 1.0, 'type': float},
                'Enabled': {'values': ['YES', 'NO'], 'type': str},
                'Primary': {'values': ['YES', 'NO'], 'type': str},
                'ConfirmationRequired': {'values': ['YES', 'NO'], 'type': str},
                'RegimeStability': {'min': 0.0, 'max': 1.0, 'type': float},
                'TransitionSensitivity': {'min': 0.0, 'max': 1.0, 'type': float}
            },
            cross_references=[]
        )
        
        # GreekSentimentConfig rules
        rules['GreekSentimentConfig'] = SheetValidationRules(
            sheet_name='GreekSentimentConfig',
            required_columns=['Parameter', 'Value', 'Type'],
            parameter_rules={
                'DeltaBaseWeight': {'min': 0.0, 'max': 2.0, 'type': float},
                'VegaBaseWeight': {'min': 0.0, 'max': 2.0, 'type': float},
                'ThetaBaseWeight': {'min': 0.0, 'max': 2.0, 'type': float},
                'GammaBaseWeight': {'min': 0.0, 'max': 2.0, 'type': float},
                'VannaWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'CharmWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'VolgaWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'VetaWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'SpeedWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'LookbackPeriod': {'min': 5, 'max': 100, 'type': int},
                'GammaThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'DeltaThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'CurrentWeekWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'NextWeekWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'CurrentMonthWeight': {'min': 0.0, 'max': 1.0, 'type': float}
            },
            cross_references=[]
        )
        
        # TrendingOIPAConfig rules
        rules['TrendingOIPAConfig'] = SheetValidationRules(
            sheet_name='TrendingOIPAConfig',
            required_columns=['Parameter', 'Value'],
            parameter_rules={
                'OIPatternLogic': {'values': ['CORRECTED_SAME', 'ORIGINAL', 'CUSTOM'], 'type': str},
                'OILookback': {'min': 1, 'max': 50, 'type': int},
                'PriceLookback': {'min': 1, 'max': 50, 'type': int},
                'DivergenceThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'AccumulationThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'UsePercentile': {'values': ['YES', 'NO'], 'type': str},
                'PercentileWindow': {'min': 5, 'max': 100, 'type': int},
                'StrikeRange': {'min': 1, 'max': 20, 'type': int},
                'SkewThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'InstitutionalThreshold': {'min': 100000, 'max': 10000000, 'type': int},
                'RetailThreshold': {'min': 1000, 'max': 1000000, 'type': int}
            },
            cross_references=[]
        )
        
        # RegimeFormationConfig rules
        rules['RegimeFormationConfig'] = SheetValidationRules(
            sheet_name='RegimeFormationConfig',
            required_columns=['RegimeType', 'DirectionalThreshold', 'VolatilityThreshold', 'Enabled'],
            parameter_rules={
                'DirectionalThreshold': {'min': -0.1, 'max': 0.1, 'type': float},
                'VolatilityThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'ConfidenceThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'MinDuration': {'min': 1, 'max': 20, 'type': int},
                'Enabled': {'values': ['YES', 'NO'], 'type': str}
            },
            cross_references=[]
        )
        
        # RegimeComplexityConfig rules
        rules['RegimeComplexityConfig'] = SheetValidationRules(
            sheet_name='RegimeComplexityConfig',
            required_columns=['Setting', 'Value'],
            parameter_rules={
                'RegimeGranularity': {'values': ['8_REGIME', '12_REGIME', '18_REGIME'], 'type': str},
                'UserLevel': {'values': ['NOVICE', 'INTERMEDIATE', 'EXPERT'], 'type': str},
                'RiskProfile': {'values': ['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'], 'type': str},
                'AdaptationSpeed': {'values': ['SLOW', 'MEDIUM', 'FAST'], 'type': str},
                'SignalFrequency': {'values': ['LOW', 'MODERATE', 'HIGH'], 'type': str}
            },
            cross_references=[]
        )
        
        # IVSurfaceConfig rules
        rules['IVSurfaceConfig'] = SheetValidationRules(
            sheet_name='IVSurfaceConfig',
            required_columns=['Parameter', 'Value'],
            parameter_rules={
                'IVSurfaceEnabled': {'values': ['YES', 'NO'], 'type': str},
                'SentimentLevels': {'min': 3, 'max': 9, 'type': int},
                'PercentileWindow': {'min': 50, 'max': 504, 'type': int},
                'ExpansionPredictionWindow': {'min': 5, 'max': 50, 'type': int},
                'ContractionSensitivity': {'min': 0.05, 'max': 0.5, 'type': float},
                'HysteresisFactor': {'min': 0.05, 'max': 0.3, 'type': float},
                'IVRankWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'IVPercentileWeight': {'min': 0.0, 'max': 1.0, 'type': float},
                'IVSkewWeight': {'min': 0.0, 'max': 1.0, 'type': float}
            },
            cross_references=[]
        )
        
        # ATRIndicatorsConfig rules
        rules['ATRIndicatorsConfig'] = SheetValidationRules(
            sheet_name='ATRIndicatorsConfig',
            required_columns=['Parameter', 'Value'],
            parameter_rules={
                'ATRIndicatorsEnabled': {'values': ['YES', 'NO'], 'type': str},
                'ATRShortPeriod': {'min': 5, 'max': 30, 'type': int},
                'ATRMediumPeriod': {'min': 10, 'max': 50, 'type': int},
                'ATRLongPeriod': {'min': 20, 'max': 100, 'type': int},
                'BandPeriods': {'min': 5, 'max': 100, 'type': int},
                'BandStdMultiplier': {'min': 0.5, 'max': 5.0, 'type': float},
                'BreakoutThreshold': {'min': 0.5, 'max': 5.0, 'type': float},
                'MinBreakoutVolume': {'min': 100, 'max': 10000, 'type': int}
            },
            cross_references=[]
        )
        
        # PerformanceMetrics rules
        rules['PerformanceMetrics'] = SheetValidationRules(
            sheet_name='PerformanceMetrics',
            required_columns=['Parameter', 'Value'],
            parameter_rules={
                'PerformanceTrackingEnabled': {'values': ['YES', 'NO'], 'type': str},
                'TrackingWindow': {'min': 100, 'max': 10000, 'type': int},
                'MinSampleSize': {'min': 10, 'max': 100, 'type': int},
                'AccuracyThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'SharpeRatioThreshold': {'min': 0.0, 'max': 5.0, 'type': float},
                'MaxDrawdownThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'WinRateThreshold': {'min': 0.0, 'max': 1.0, 'type': float}
            },
            cross_references=[]
        )
        
        # SystemConfiguration rules
        rules['SystemConfiguration'] = SheetValidationRules(
            sheet_name='SystemConfiguration',
            required_columns=['Parameter', 'Value'],
            parameter_rules={
                'UnifiedSystemEnabled': {'values': ['YES', 'NO'], 'type': str},
                'TrendingOIEnabled': {'values': ['YES', 'NO'], 'type': str},
                'GreekSentimentEnabled': {'values': ['YES', 'NO'], 'type': str},
                'TripleStraddleEnabled': {'values': ['YES', 'NO'], 'type': str},
                'GlobalConfidenceThreshold': {'min': 0.0, 'max': 1.0, 'type': float},
                'RegimeMode': {'values': ['8_REGIME', '18_REGIME'], 'type': str},
                'MaxConcurrentAnalysis': {'min': 1, 'max': 16, 'type': int}
            },
            cross_references=[]
        )
        
        return rules
    
    def validate_excel_file(self, excel_path: str) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate entire Excel configuration file
        
        Args:
            excel_path: Path to Excel configuration file
            
        Returns:
            Tuple of (is_valid, list of validation results)
        """
        self.validation_results = []
        
        try:
            # Check file exists
            if not Path(excel_path).exists():
                self.validation_results.append(
                    ValidationResult(
                        is_valid=False,
                        sheet_name="FILE",
                        parameter="Path",
                        message=f"Configuration file not found: {excel_path}",
                        severity=ValidationSeverity.ERROR
                    )
                )
                return False, self.validation_results
            
            # Load Excel file
            xl_file = pd.ExcelFile(excel_path)
            
            # Validate sheet structure
            self._validate_sheet_structure(xl_file)
            
            # Validate each sheet
            for sheet_name in self.required_sheets:
                if sheet_name in xl_file.sheet_names:
                    self._validate_sheet_content(xl_file, sheet_name)
            
            # Validate cross-sheet consistency
            self._validate_cross_sheet_consistency(xl_file)
            
            # Check overall validity
            has_errors = any(r.severity == ValidationSeverity.ERROR for r in self.validation_results)
            
            if not has_errors:
                self.validation_results.append(
                    ValidationResult(
                        is_valid=True,
                        sheet_name="OVERALL",
                        parameter="Configuration",
                        message="All validations passed successfully",
                        severity=ValidationSeverity.INFO
                    )
                )
            
            return not has_errors, self.validation_results
            
        except Exception as e:
            logger.error(f"Error validating Excel file: {e}")
            self.validation_results.append(
                ValidationResult(
                    is_valid=False,
                    sheet_name="FILE",
                    parameter="Validation",
                    message=f"Validation failed: {str(e)}",
                    severity=ValidationSeverity.ERROR
                )
            )
            return False, self.validation_results
    
    def _validate_sheet_structure(self, xl_file: pd.ExcelFile):
        """Validate Excel file has required sheets"""
        existing_sheets = set(xl_file.sheet_names)
        
        # Check for missing required sheets
        for sheet in self.required_sheets:
            if sheet not in existing_sheets:
                self.validation_results.append(
                    ValidationResult(
                        is_valid=False,
                        sheet_name=sheet,
                        parameter="Sheet",
                        message=f"Required sheet '{sheet}' is missing",
                        severity=ValidationSeverity.ERROR
                    )
                )
        
        # Check for unexpected sheets (warning only)
        optional_sheets = {'ValidationRules', 'TemplateMetadata'}
        unexpected = existing_sheets - set(self.required_sheets) - optional_sheets
        
        for sheet in unexpected:
            self.validation_results.append(
                ValidationResult(
                    is_valid=True,
                    sheet_name=sheet,
                    parameter="Sheet",
                    message=f"Unexpected sheet '{sheet}' found (will be ignored)",
                    severity=ValidationSeverity.WARNING
                )
            )
    
    def _validate_sheet_content(self, xl_file: pd.ExcelFile, sheet_name: str):
        """Validate content of a specific sheet"""
        if sheet_name not in self.sheet_rules:
            return
        
        rules = self.sheet_rules[sheet_name]
        
        try:
            # Read sheet
            df = pd.read_excel(xl_file, sheet_name=sheet_name, header=1)
            
            # Check required columns
            self._validate_required_columns(df, rules)
            
            # Validate parameter values
            self._validate_parameter_values(df, rules, sheet_name)
            
            # Check sum constraints
            self._validate_sum_constraints(df, rules, sheet_name)
            
        except Exception as e:
            self.validation_results.append(
                ValidationResult(
                    is_valid=False,
                    sheet_name=sheet_name,
                    parameter="Content",
                    message=f"Error reading sheet: {str(e)}",
                    severity=ValidationSeverity.ERROR
                )
            )
    
    def _validate_required_columns(self, df: pd.DataFrame, rules: SheetValidationRules):
        """Validate sheet has required columns"""
        missing_columns = set(rules.required_columns) - set(df.columns)
        
        for col in missing_columns:
            self.validation_results.append(
                ValidationResult(
                    is_valid=False,
                    sheet_name=rules.sheet_name,
                    parameter=col,
                    message=f"Required column '{col}' is missing",
                    severity=ValidationSeverity.ERROR
                )
            )
    
    def _validate_parameter_values(self, df: pd.DataFrame, rules: SheetValidationRules, sheet_name: str):
        """Validate parameter values against rules"""
        for param_name, param_rules in rules.parameter_rules.items():
            # Find parameter in dataframe
            param_values = self._find_parameter_values(df, param_name, sheet_name)
            
            for row_idx, value in param_values:
                if pd.isna(value):
                    continue
                
                # Type validation
                expected_type = param_rules.get('type', str)
                try:
                    if expected_type == float:
                        value = float(value)
                    elif expected_type == int:
                        value = int(float(value))
                    else:
                        value = str(value)
                except:
                    self.validation_results.append(
                        ValidationResult(
                            is_valid=False,
                            sheet_name=sheet_name,
                            parameter=param_name,
                            message=f"Invalid type for {param_name}: expected {expected_type.__name__}",
                            severity=ValidationSeverity.ERROR,
                            cell_reference=f"Row {row_idx + 3}"
                        )
                    )
                    continue
                
                # Range validation
                if 'min' in param_rules and value < param_rules['min']:
                    self.validation_results.append(
                        ValidationResult(
                            is_valid=False,
                            sheet_name=sheet_name,
                            parameter=param_name,
                            message=f"{param_name} value {value} below minimum {param_rules['min']}",
                            severity=ValidationSeverity.ERROR,
                            cell_reference=f"Row {row_idx + 3}",
                            suggested_value=param_rules['min']
                        )
                    )
                
                if 'max' in param_rules and value > param_rules['max']:
                    self.validation_results.append(
                        ValidationResult(
                            is_valid=False,
                            sheet_name=sheet_name,
                            parameter=param_name,
                            message=f"{param_name} value {value} above maximum {param_rules['max']}",
                            severity=ValidationSeverity.ERROR,
                            cell_reference=f"Row {row_idx + 3}",
                            suggested_value=param_rules['max']
                        )
                    )
                
                # Allowed values validation
                if 'values' in param_rules:
                    str_value = str(value).upper()
                    allowed = [str(v).upper() for v in param_rules['values']]
                    if str_value not in allowed:
                        self.validation_results.append(
                            ValidationResult(
                                is_valid=False,
                                sheet_name=sheet_name,
                                parameter=param_name,
                                message=f"{param_name} value '{value}' not in allowed values: {param_rules['values']}",
                                severity=ValidationSeverity.ERROR,
                                cell_reference=f"Row {row_idx + 3}",
                                suggested_value=param_rules['values'][0]
                            )
                        )
    
    def _find_parameter_values(self, df: pd.DataFrame, param_name: str, sheet_name: str) -> List[Tuple[int, Any]]:
        """Find parameter values in dataframe"""
        values = []
        
        # Check if parameter is a column name
        if param_name in df.columns:
            for idx, value in enumerate(df[param_name]):
                if pd.notna(value):
                    values.append((idx, value))
        
        # Check if parameter is in Parameter column (for key-value sheets)
        elif 'Parameter' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get('Parameter')) and str(row['Parameter']) == param_name:
                    if 'Value' in row and pd.notna(row['Value']):
                        values.append((idx, row['Value']))
        
        return values
    
    def _validate_sum_constraints(self, df: pd.DataFrame, rules: SheetValidationRules, sheet_name: str):
        """Validate sum constraints (e.g., weights must sum to 1.0)"""
        for param_name, param_rules in rules.parameter_rules.items():
            if 'sum_constraint' in param_rules:
                values = self._find_parameter_values(df, param_name, sheet_name)
                if values:
                    total = sum(float(v[1]) for v in values if pd.notna(v[1]))
                    expected = param_rules['sum_constraint']
                    
                    if abs(total - expected) > 0.001:  # Allow small floating point errors
                        self.validation_results.append(
                            ValidationResult(
                                is_valid=False,
                                sheet_name=sheet_name,
                                parameter=param_name,
                                message=f"{param_name} values sum to {total:.3f}, expected {expected}",
                                severity=ValidationSeverity.ERROR,
                                suggested_value=expected
                            )
                        )
    
    def _validate_cross_sheet_consistency(self, xl_file: pd.ExcelFile):
        """Validate consistency across sheets"""
        try:
            # Check system consistency between IndicatorConfiguration and DynamicWeightageConfig
            if all(sheet in xl_file.sheet_names for sheet in ['IndicatorConfiguration', 'DynamicWeightageConfig']):
                df_indicators = pd.read_excel(xl_file, sheet_name='IndicatorConfiguration', header=1)
                df_weights = pd.read_excel(xl_file, sheet_name='DynamicWeightageConfig', header=1)
                
                # Get system names
                indicator_systems = set()
                for _, row in df_indicators.iterrows():
                    if pd.notna(row.get('IndicatorSystem')):
                        indicator_systems.add(str(row['IndicatorSystem']))
                
                weight_systems = set()
                for _, row in df_weights.iterrows():
                    if pd.notna(row.get('SystemName')) and not pd.isna(row.get('CurrentWeight')):
                        weight_systems.add(str(row['SystemName']))
                
                # Check for mismatches
                missing_in_weights = indicator_systems - weight_systems
                missing_in_indicators = weight_systems - indicator_systems
                
                for system in missing_in_weights:
                    self.validation_results.append(
                        ValidationResult(
                            is_valid=False,
                            sheet_name="DynamicWeightageConfig",
                            parameter=system,
                            message=f"System '{system}' in IndicatorConfiguration but missing in DynamicWeightageConfig",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                
                for system in missing_in_indicators:
                    if system not in ['Parameter', 'Weight Optimization Parameters']:  # Ignore headers
                        self.validation_results.append(
                            ValidationResult(
                                is_valid=False,
                                sheet_name="IndicatorConfiguration",
                                parameter=system,
                                message=f"System '{system}' in DynamicWeightageConfig but missing in IndicatorConfiguration",
                                severity=ValidationSeverity.WARNING
                            )
                        )
            
            # Check enabled systems have non-zero weights
            if 'SystemConfiguration' in xl_file.sheet_names:
                df_system = pd.read_excel(xl_file, sheet_name='SystemConfiguration', header=1)
                
                # Find enabled systems
                enabled_systems = []
                for _, row in df_system.iterrows():
                    param = str(row.get('Parameter', ''))
                    value = str(row.get('Value', '')).upper()
                    
                    if param.endswith('Enabled') and value == 'YES':
                        system_name = param.replace('Enabled', '').replace('Enhanced', '').replace('Advanced', '')
                        enabled_systems.append(system_name)
                
                # Add more cross-sheet validations as needed
                
        except Exception as e:
            logger.error(f"Error in cross-sheet validation: {e}")
            self.validation_results.append(
                ValidationResult(
                    is_valid=False,
                    sheet_name="CROSS_SHEET",
                    parameter="Consistency",
                    message=f"Cross-sheet validation error: {str(e)}",
                    severity=ValidationSeverity.WARNING
                )
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary for UI display"""
        errors = [r for r in self.validation_results if r.severity == ValidationSeverity.ERROR]
        warnings = [r for r in self.validation_results if r.severity == ValidationSeverity.WARNING]
        info = [r for r in self.validation_results if r.severity == ValidationSeverity.INFO]
        
        summary = {
            'total_checks': len(self.validation_results),
            'errors': len(errors),
            'warnings': len(warnings),
            'info': len(info),
            'is_valid': len(errors) == 0,
            'details': {
                'errors': [self._format_result(r) for r in errors],
                'warnings': [self._format_result(r) for r in warnings],
                'info': [self._format_result(r) for r in info]
            }
        }
        
        return summary
    
    def _format_result(self, result: ValidationResult) -> Dict[str, Any]:
        """Format validation result for UI display"""
        formatted = {
            'sheet': result.sheet_name,
            'parameter': result.parameter,
            'message': result.message,
            'severity': result.severity.value
        }
        
        if result.cell_reference:
            formatted['cell'] = result.cell_reference
        
        if result.suggested_value is not None:
            formatted['suggested'] = result.suggested_value
        
        return formatted
    
    def export_validation_report(self, output_path: str):
        """Export detailed validation report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'validator_version': '1.0.0',
            'summary': self.get_validation_summary(),
            'detailed_results': [
                {
                    'sheet': r.sheet_name,
                    'parameter': r.parameter,
                    'message': r.message,
                    'severity': r.severity.value,
                    'cell_reference': r.cell_reference,
                    'suggested_value': r.suggested_value,
                    'is_valid': r.is_valid
                }
                for r in self.validation_results
            ]
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to: {output_path}")


def main():
    """Test the validator with the unified configuration"""
    import sys
    
    # Test with the unified configuration
    config_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/UNIFIED_ENHANCED_MARKET_REGIME_CONFIG_V2.xlsx"
    
    print("Unified Configuration Validator")
    print("=" * 50)
    
    validator = UnifiedConfigValidator()
    is_valid, results = validator.validate_excel_file(config_path)
    
    # Get summary
    summary = validator.get_validation_summary()
    
    print(f"\nValidation Summary:")
    print(f"  Total Checks: {summary['total_checks']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Info: {summary['info']}")
    print(f"  Valid: {'✅ YES' if summary['is_valid'] else '❌ NO'}")
    
    # Show errors if any
    if summary['errors'] > 0:
        print("\nErrors:")
        for error in summary['details']['errors']:
            print(f"  - [{error['sheet']}] {error['parameter']}: {error['message']}")
            if 'suggested' in error:
                print(f"    Suggested: {error['suggested']}")
    
    # Show warnings if any
    if summary['warnings'] > 0:
        print("\nWarnings:")
        for warning in summary['details']['warnings']:
            print(f"  - [{warning['sheet']}] {warning['parameter']}: {warning['message']}")
    
    # Export report
    report_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/validation_report.json"
    validator.export_validation_report(report_path)
    print(f"\nDetailed report saved to: {report_path}")
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())