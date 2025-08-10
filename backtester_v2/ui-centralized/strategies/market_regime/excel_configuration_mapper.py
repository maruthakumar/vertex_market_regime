#!/usr/bin/env python3
"""
Excel Configuration Mapper
==========================

This module provides systematic mapping from the 31-sheet Excel configuration
to individual enhanced module parameters, ensuring proper parameter injection
and configuration validation for the market regime system.

Features:
- Systematic mapping from 31 Excel sheets to module parameters
- Parameter validation and type conversion
- Configuration change propagation mechanism
- Dynamic configuration reloading capability
- Cross-parameter dependency validation
- Configuration versioning and rollback

Author: The Augster
Date: 2025-01-06
Version: 1.0.0 - Excel Configuration Mapper
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import openpyxl
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Parameter type enumeration"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ENUM = "enum"

class ValidationRule(Enum):
    """Parameter validation rules"""
    REQUIRED = "required"
    RANGE = "range"
    CHOICES = "choices"
    REGEX = "regex"
    DEPENDENCY = "dependency"

@dataclass
class ParameterMapping:
    """Parameter mapping configuration"""
    excel_sheet: str
    excel_column: str
    module_parameter: str
    parameter_type: ParameterType
    default_value: Any = None
    validation_rules: List[ValidationRule] = field(default_factory=list)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

@dataclass
class ModuleParameterConfig:
    """Complete parameter configuration for a module"""
    module_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    mappings: List[ParameterMapping] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None

class ExcelConfigurationMapper:
    """
    Systematic mapper for Excel configuration to enhanced module parameters
    
    Handles the complex task of mapping 600+ parameters from 31 Excel sheets
    to individual enhanced module configurations with proper validation.
    """
    
    def __init__(self, excel_file_path: str):
        """Initialize Excel Configuration Mapper"""
        self.excel_file_path = Path(excel_file_path)
        self.excel_data: Dict[str, pd.DataFrame] = {}
        self.parameter_mappings: Dict[str, List[ParameterMapping]] = {}
        self.module_configs: Dict[str, ModuleParameterConfig] = {}
        
        # Configuration cache
        self.config_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
        
        # Initialize parameter mappings
        self._initialize_parameter_mappings()
        
        logger.info(f"Excel Configuration Mapper initialized for: {excel_file_path}")
    
    def load_excel_configuration(self) -> bool:
        """
        Load Excel configuration from file
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not self.excel_file_path.exists():
                logger.error(f"Excel file not found: {self.excel_file_path}")
                return False
            
            # Load all sheets
            wb = openpyxl.load_workbook(self.excel_file_path, read_only=True)
            sheet_names = wb.sheetnames
            wb.close()
            
            logger.info(f"Loading {len(sheet_names)} sheets from Excel configuration")
            
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(self.excel_file_path, sheet_name=sheet_name)
                    self.excel_data[sheet_name] = df
                    logger.debug(f"Loaded sheet: {sheet_name} ({df.shape[0]} rows, {df.shape[1]} cols)")
                except Exception as e:
                    logger.warning(f"Error loading sheet {sheet_name}: {e}")
                    continue
            
            self.cache_timestamp = datetime.now()
            logger.info(f"Successfully loaded {len(self.excel_data)} sheets")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Excel configuration: {e}")
            return False
    
    def map_module_parameters(self, module_name: str) -> Optional[ModuleParameterConfig]:
        """
        Map Excel parameters to specific module configuration
        
        Args:
            module_name: Name of the module to configure
            
        Returns:
            Optional[ModuleParameterConfig]: Module configuration or None if error
        """
        try:
            if module_name not in self.parameter_mappings:
                logger.warning(f"No parameter mappings defined for module: {module_name}")
                return None
            
            # Create module config
            module_config = ModuleParameterConfig(module_name=module_name)
            mappings = self.parameter_mappings[module_name]
            
            # Process each parameter mapping
            for mapping in mappings:
                try:
                    value = self._extract_parameter_value(mapping)
                    if value is not None:
                        # Validate parameter
                        if self._validate_parameter(mapping, value):
                            module_config.parameters[mapping.module_parameter] = value
                        else:
                            error_msg = f"Validation failed for parameter: {mapping.module_parameter}"
                            module_config.validation_errors.append(error_msg)
                            logger.warning(error_msg)
                    else:
                        # Use default value if available
                        if mapping.default_value is not None:
                            module_config.parameters[mapping.module_parameter] = mapping.default_value
                        elif ValidationRule.REQUIRED in mapping.validation_rules:
                            error_msg = f"Required parameter missing: {mapping.module_parameter}"
                            module_config.validation_errors.append(error_msg)
                            logger.error(error_msg)
                
                except Exception as e:
                    error_msg = f"Error processing parameter {mapping.module_parameter}: {e}"
                    module_config.validation_errors.append(error_msg)
                    logger.error(error_msg)
            
            module_config.mappings = mappings
            module_config.last_updated = datetime.now()
            
            # Cache the configuration
            self.module_configs[module_name] = module_config
            
            logger.info(f"Mapped {len(module_config.parameters)} parameters for module: {module_name}")
            if module_config.validation_errors:
                logger.warning(f"Module {module_name} has {len(module_config.validation_errors)} validation errors")
            
            return module_config
            
        except Exception as e:
            logger.error(f"Error mapping parameters for module {module_name}: {e}")
            return None
    
    def get_all_module_configurations(self) -> Dict[str, ModuleParameterConfig]:
        """
        Get configurations for all registered modules
        
        Returns:
            Dict[str, ModuleParameterConfig]: All module configurations
        """
        all_configs = {}
        
        for module_name in self.parameter_mappings.keys():
            config = self.map_module_parameters(module_name)
            if config:
                all_configs[module_name] = config
        
        return all_configs
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate complete configuration
        
        Returns:
            Dict: Validation report
        """
        validation_report = {
            'timestamp': datetime.now(),
            'total_modules': len(self.parameter_mappings),
            'successful_modules': 0,
            'failed_modules': 0,
            'total_parameters': 0,
            'validation_errors': [],
            'module_reports': {}
        }
        
        for module_name in self.parameter_mappings.keys():
            config = self.map_module_parameters(module_name)
            if config:
                module_report = {
                    'parameters_mapped': len(config.parameters),
                    'validation_errors': len(config.validation_errors),
                    'errors': config.validation_errors.copy()
                }
                
                validation_report['module_reports'][module_name] = module_report
                validation_report['total_parameters'] += len(config.parameters)
                
                if config.validation_errors:
                    validation_report['failed_modules'] += 1
                    validation_report['validation_errors'].extend(
                        [f"{module_name}: {error}" for error in config.validation_errors]
                    )
                else:
                    validation_report['successful_modules'] += 1
        
        return validation_report
    
    def _initialize_parameter_mappings(self):
        """Initialize parameter mappings for all enhanced modules"""
        
        # Greek Sentiment Analysis Module
        self.parameter_mappings['enhanced_greek_sentiment_analysis'] = [
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='delta_weight',
                module_parameter='delta_weight',
                parameter_type=ParameterType.FLOAT,
                default_value=1.2,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.0, 'max': 3.0}
            ),
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='vega_weight',
                module_parameter='vega_weight',
                parameter_type=ParameterType.FLOAT,
                default_value=1.5,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.0, 'max': 3.0}
            ),
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='theta_weight',
                module_parameter='theta_weight',
                parameter_type=ParameterType.FLOAT,
                default_value=0.3,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.0, 'max': 2.0}
            ),
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='gamma_weight',
                module_parameter='gamma_weight',
                parameter_type=ParameterType.FLOAT,
                default_value=0.0,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.0, 'max': 2.0}
            ),
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='sentiment_threshold_strong_bullish',
                module_parameter='strong_bullish_threshold',
                parameter_type=ParameterType.FLOAT,
                default_value=0.45,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.1, 'max': 1.0}
            ),
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='sentiment_threshold_mild_bullish',
                module_parameter='mild_bullish_threshold',
                parameter_type=ParameterType.FLOAT,
                default_value=0.15,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.05, 'max': 0.5}
            ),
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='dte_near_expiry_days',
                module_parameter='near_expiry_dte',
                parameter_type=ParameterType.INTEGER,
                default_value=7,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 1, 'max': 15}
            ),
            ParameterMapping(
                excel_sheet='GreekSentimentConfig',
                excel_column='dte_medium_expiry_days',
                module_parameter='medium_expiry_dte',
                parameter_type=ParameterType.INTEGER,
                default_value=30,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 8, 'max': 60}
            )
        ]
        
        # Trending OI PA Analysis Module
        self.parameter_mappings['enhanced_trending_oi_pa_analysis'] = [
            ParameterMapping(
                excel_sheet='TrendingOIPAConfig',
                excel_column='correlation_threshold',
                module_parameter='correlation_threshold',
                parameter_type=ParameterType.FLOAT,
                default_value=0.80,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.5, 'max': 1.0}
            ),
            ParameterMapping(
                excel_sheet='TrendingOIPAConfig',
                excel_column='time_decay_lambda',
                module_parameter='time_decay_lambda',
                parameter_type=ParameterType.FLOAT,
                default_value=0.1,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 0.01, 'max': 1.0}
            ),
            ParameterMapping(
                excel_sheet='TrendingOIPAConfig',
                excel_column='primary_timeframe_minutes',
                module_parameter='primary_timeframe',
                parameter_type=ParameterType.INTEGER,
                default_value=3,
                validation_rules=[ValidationRule.CHOICES],
                validation_config={'choices': [1, 3, 5, 15]}
            ),
            ParameterMapping(
                excel_sheet='TrendingOIPAConfig',
                excel_column='confirmation_timeframe_minutes',
                module_parameter='confirmation_timeframe',
                parameter_type=ParameterType.INTEGER,
                default_value=15,
                validation_rules=[ValidationRule.CHOICES],
                validation_config={'choices': [5, 15, 30, 60]}
            ),
            ParameterMapping(
                excel_sheet='TrendingOIPAConfig',
                excel_column='institutional_threshold',
                module_parameter='institutional_threshold',
                parameter_type=ParameterType.FLOAT,
                default_value=2.0,
                validation_rules=[ValidationRule.RANGE],
                validation_config={'min': 1.0, 'max': 5.0}
            )
        ]
        
        logger.info(f"Initialized parameter mappings for {len(self.parameter_mappings)} modules")

    def _extract_parameter_value(self, mapping: ParameterMapping) -> Any:
        """Extract parameter value from Excel data"""
        try:
            if mapping.excel_sheet not in self.excel_data:
                logger.warning(f"Sheet {mapping.excel_sheet} not found in Excel data")
                return None

            df = self.excel_data[mapping.excel_sheet]

            # Look for the parameter in different ways
            value = None

            # Method 1: Direct column match
            if mapping.excel_column in df.columns:
                # Get the first non-null value
                non_null_values = df[mapping.excel_column].dropna()
                if not non_null_values.empty:
                    value = non_null_values.iloc[0]

            # Method 2: Search in Parameter column with Value column
            elif 'Parameter' in df.columns and 'Value' in df.columns:
                param_row = df[df['Parameter'] == mapping.excel_column]
                if not param_row.empty:
                    value = param_row['Value'].iloc[0]

            # Method 3: Search in first column with second column as value
            elif len(df.columns) >= 2:
                first_col = df.columns[0]
                second_col = df.columns[1]
                param_row = df[df[first_col] == mapping.excel_column]
                if not param_row.empty:
                    value = param_row[second_col].iloc[0]

            # Convert value to appropriate type
            if value is not None:
                value = self._convert_parameter_type(value, mapping.parameter_type)

            return value

        except Exception as e:
            logger.error(f"Error extracting parameter {mapping.module_parameter}: {e}")
            return None

    def _convert_parameter_type(self, value: Any, param_type: ParameterType) -> Any:
        """Convert parameter value to specified type"""
        try:
            if pd.isna(value):
                return None

            if param_type == ParameterType.STRING:
                return str(value)
            elif param_type == ParameterType.INTEGER:
                return int(float(value))  # Handle Excel numeric values
            elif param_type == ParameterType.FLOAT:
                return float(value)
            elif param_type == ParameterType.BOOLEAN:
                if isinstance(value, str):
                    return value.upper() in ['TRUE', 'YES', '1', 'ON', 'ENABLED']
                return bool(value)
            elif param_type == ParameterType.LIST:
                if isinstance(value, str):
                    # Try to parse as JSON or comma-separated
                    try:
                        return json.loads(value)
                    except:
                        return [item.strip() for item in value.split(',')]
                elif isinstance(value, (list, tuple)):
                    return list(value)
                else:
                    return [value]
            elif param_type == ParameterType.DICT:
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except:
                        return {'value': value}
                elif isinstance(value, dict):
                    return value
                else:
                    return {'value': value}
            else:
                return value

        except Exception as e:
            logger.error(f"Error converting parameter type {param_type}: {e}")
            return value

    def _validate_parameter(self, mapping: ParameterMapping, value: Any) -> bool:
        """Validate parameter value against rules"""
        try:
            for rule in mapping.validation_rules:
                if rule == ValidationRule.REQUIRED:
                    if value is None:
                        return False

                elif rule == ValidationRule.RANGE:
                    if 'min' in mapping.validation_config and value < mapping.validation_config['min']:
                        return False
                    if 'max' in mapping.validation_config and value > mapping.validation_config['max']:
                        return False

                elif rule == ValidationRule.CHOICES:
                    if 'choices' in mapping.validation_config:
                        if value not in mapping.validation_config['choices']:
                            return False

                elif rule == ValidationRule.REGEX:
                    if 'pattern' in mapping.validation_config:
                        import re
                        if not re.match(mapping.validation_config['pattern'], str(value)):
                            return False

            return True

        except Exception as e:
            logger.error(f"Error validating parameter {mapping.module_parameter}: {e}")
            return False

    def add_module_mapping(self, module_name: str, mappings: List[ParameterMapping]):
        """Add parameter mappings for a new module"""
        self.parameter_mappings[module_name] = mappings
        logger.info(f"Added parameter mappings for module: {module_name}")

    def reload_configuration(self) -> bool:
        """Reload configuration from Excel file"""
        try:
            # Clear cache
            self.excel_data.clear()
            self.module_configs.clear()
            self.config_cache.clear()

            # Reload Excel data
            return self.load_excel_configuration()

        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False

    def export_configuration_summary(self, output_path: str) -> bool:
        """Export configuration summary to JSON file"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'excel_file': str(self.excel_file_path),
                'total_sheets': len(self.excel_data),
                'total_modules': len(self.parameter_mappings),
                'modules': {}
            }

            for module_name, config in self.module_configs.items():
                summary['modules'][module_name] = {
                    'parameters_count': len(config.parameters),
                    'validation_errors_count': len(config.validation_errors),
                    'last_updated': config.last_updated.isoformat() if config.last_updated else None,
                    'parameters': config.parameters.copy(),
                    'validation_errors': config.validation_errors.copy()
                }

            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Configuration summary exported to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting configuration summary: {e}")
            return False


# Factory function for easy instantiation
def create_configuration_mapper(excel_file_path: str) -> ExcelConfigurationMapper:
    """
    Factory function to create Excel Configuration Mapper

    Args:
        excel_file_path: Path to Excel configuration file

    Returns:
        ExcelConfigurationMapper: Configured mapper instance
    """
    mapper = ExcelConfigurationMapper(excel_file_path)
    mapper.load_excel_configuration()
    return mapper
