#!/usr/bin/env python3
"""
Parameter Injector for Market Regime Analysis
============================================

Advanced parameter injection system that integrates Excel configuration
parameters into output data for complete traceability and transparency.

Features:
- Excel configuration parameter extraction
- Intelligent parameter categorization
- Dynamic parameter injection into time series
- Parameter validation and conflict resolution
- Configuration change tracking
- Multi-format parameter encoding

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field

from ..common_utils import ErrorHandler, DataValidator, ConfigUtils
from ...excel_configuration_mapper import ExcelConfigurationMapper

logger = logging.getLogger(__name__)


@dataclass
class ParameterCategory:
    """Parameter category definition"""
    name: str
    description: str
    parameters: List[str] = field(default_factory=list)
    encoding_method: str = 'json'  # json, string, numeric, boolean
    include_in_output: bool = True
    

@dataclass
class ParameterInjectionConfig:
    """Configuration for parameter injection"""
    include_all_parameters: bool = True
    include_metadata: bool = True
    parameter_prefix: str = 'param_'
    max_parameter_length: int = 1000
    categories: List[ParameterCategory] = field(default_factory=list)
    

class ParameterInjector:
    """
    Advanced parameter injection system for Excel configuration integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Parameter Injector"""
        self.config = config
        self.injection_config = self._parse_injection_config(config)
        
        # Initialize utilities
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        self.config_utils = ConfigUtils()
        
        # Excel configuration mapper
        excel_path = config.get('excel_configuration_path')
        if excel_path:
            self.excel_mapper = ExcelConfigurationMapper(excel_path)
        else:
            self.excel_mapper = None
        
        # Parameter categories
        self._initialize_parameter_categories()
        
        logger.info("Parameter Injector initialized")
    
    def inject_parameters(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        injection_mode: str = 'comprehensive'
    ) -> pd.DataFrame:
        """
        Inject Excel parameters into DataFrame with comprehensive traceability
        
        Args:
            data: DataFrame to inject parameters into
            parameters: Dictionary of parameters to inject
            injection_mode: 'comprehensive', 'essential', 'minimal'
            
        Returns:
            DataFrame with injected parameters
        """
        try:
            # Validate input data
            if data.empty:
                logger.warning("Empty DataFrame provided for parameter injection")
                return data
            
            # Create working copy
            injected_df = data.copy()
            
            # Categorize parameters
            categorized_params = self._categorize_parameters(parameters)
            
            # Inject parameters based on mode
            if injection_mode == 'comprehensive':
                injected_df = self._inject_comprehensive_parameters(
                    injected_df, categorized_params
                )
            elif injection_mode == 'essential':
                injected_df = self._inject_essential_parameters(
                    injected_df, categorized_params
                )
            elif injection_mode == 'minimal':
                injected_df = self._inject_minimal_parameters(
                    injected_df, categorized_params
                )
            else:
                raise ValueError(f"Unknown injection mode: {injection_mode}")
            
            # Add injection metadata
            if self.injection_config.include_metadata:
                injected_df = self._add_injection_metadata(
                    injected_df, parameters, injection_mode
                )
            
            # Validate injection results
            injection_summary = self._validate_injection_results(
                injected_df, parameters
            )
            
            logger.info(f"Parameter injection completed: {injection_summary}")
            return injected_df
            
        except Exception as e:
            error_msg = f"Error injecting parameters: {e}"
            self.error_handler.handle_error(error_msg, e)
            return data
    
    def _parse_injection_config(self, config: Dict[str, Any]) -> ParameterInjectionConfig:
        """
        Parse injection configuration from config dictionary
        """
        injection_config = config.get('parameter_injection', {})
        
        return ParameterInjectionConfig(
            include_all_parameters=injection_config.get('include_all_parameters', True),
            include_metadata=injection_config.get('include_metadata', True),
            parameter_prefix=injection_config.get('parameter_prefix', 'param_'),
            max_parameter_length=injection_config.get('max_parameter_length', 1000)
        )
    
    def _initialize_parameter_categories(self):
        """
        Initialize parameter categories for intelligent injection
        """
        self.parameter_categories = {
            'trading_mode': ParameterCategory(
                name='trading_mode',
                description='Trading mode configuration (intraday/positional/hybrid)',
                parameters=['trading_mode', 'timeframe_weights', 'mode_settings'],
                encoding_method='json'
            ),
            'indicators': ParameterCategory(
                name='indicators',
                description='Indicator weights and configuration',
                parameters=['indicator_weights', 'straddle_weights', 'greek_weights'],
                encoding_method='json'
            ),
            'thresholds': ParameterCategory(
                name='thresholds',
                description='Regime detection thresholds',
                parameters=['volatility_thresholds', 'directional_thresholds', 'confidence_threshold'],
                encoding_method='json'
            ),
            'optimization': ParameterCategory(
                name='optimization',
                description='Dynamic optimization parameters',
                parameters=['learning_rate', 'adaptation_period', 'optimization_enabled'],
                encoding_method='json'
            ),
            'system': ParameterCategory(
                name='system',
                description='System configuration and metadata',
                parameters=['config_file', 'version', 'symbol', 'analysis_date'],
                encoding_method='string'
            )
        }
    
    def _categorize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Categorize parameters for intelligent injection
        """
        categorized = {category: {} for category in self.parameter_categories.keys()}
        uncategorized = {}
        
        for param_name, param_value in parameters.items():
            categorized_param = False
            
            # Find matching category
            for category_name, category_config in self.parameter_categories.items():
                if any(param_name.lower().find(cat_param.lower()) != -1 
                      for cat_param in category_config.parameters):
                    categorized[category_name][param_name] = param_value
                    categorized_param = True
                    break
            
            # If not categorized, add to uncategorized
            if not categorized_param:
                uncategorized[param_name] = param_value
        
        # Add uncategorized parameters
        if uncategorized:
            categorized['other'] = uncategorized
        
        return categorized
    
    def _inject_comprehensive_parameters(self, df: pd.DataFrame, categorized_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Inject all parameters with full categorization
        """
        injected_df = df.copy()
        
        for category_name, category_params in categorized_params.items():
            if not category_params:
                continue
                
            category_config = self.parameter_categories.get(
                category_name, 
                ParameterCategory(name=category_name, description='Uncategorized parameters')
            )
            
            # Inject category parameters
            for param_name, param_value in category_params.items():
                column_name = f"{self.injection_config.parameter_prefix}{category_name}_{param_name}"
                encoded_value = self._encode_parameter_value(
                    param_value, category_config.encoding_method
                )
                injected_df[column_name] = encoded_value
        
        return injected_df
    
    def _inject_essential_parameters(self, df: pd.DataFrame, categorized_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Inject only essential parameters for trading
        """
        injected_df = df.copy()
        
        # Essential parameter list
        essential_categories = ['trading_mode', 'indicators', 'thresholds']
        essential_params = [
            'trading_mode', 'indicator_weights', 'volatility_thresholds',
            'directional_thresholds', 'confidence_threshold', 'symbol'
        ]
        
        for category_name in essential_categories:
            if category_name in categorized_params:
                category_params = categorized_params[category_name]
                category_config = self.parameter_categories[category_name]
                
                for param_name, param_value in category_params.items():
                    if any(essential_param in param_name.lower() for essential_param in essential_params):
                        column_name = f"{self.injection_config.parameter_prefix}{param_name}"
                        encoded_value = self._encode_parameter_value(
                            param_value, category_config.encoding_method
                        )
                        injected_df[column_name] = encoded_value
        
        return injected_df
    
    def _inject_minimal_parameters(self, df: pd.DataFrame, categorized_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Inject only minimal parameters for identification
        """
        injected_df = df.copy()
        
        # Minimal parameter list
        minimal_params = ['symbol', 'trading_mode', 'config_file', 'analysis_date']
        
        # Flatten categorized parameters
        all_params = {}
        for category_params in categorized_params.values():
            all_params.update(category_params)
        
        for param_name, param_value in all_params.items():
            if any(minimal_param in param_name.lower() for minimal_param in minimal_params):
                column_name = f"{self.injection_config.parameter_prefix}{param_name}"
                encoded_value = self._encode_parameter_value(param_value, 'string')
                injected_df[column_name] = encoded_value
        
        return injected_df
    
    def _encode_parameter_value(self, value: Any, encoding_method: str) -> Any:
        """
        Encode parameter value based on specified method
        """
        try:
            if encoding_method == 'json':
                if isinstance(value, (dict, list)):
                    encoded = json.dumps(value)
                    # Truncate if too long
                    if len(encoded) > self.injection_config.max_parameter_length:
                        encoded = encoded[:self.injection_config.max_parameter_length] + '...'
                    return encoded
                else:
                    return json.dumps(value)
            
            elif encoding_method == 'string':
                return str(value)
            
            elif encoding_method == 'numeric':
                if isinstance(value, (int, float)):
                    return value
                else:
                    try:
                        return float(value)
                    except:
                        return 0
            
            elif encoding_method == 'boolean':
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ['true', 'yes', '1', 'on', 'enabled']
                else:
                    return bool(value)
            
            else:
                return str(value)
                
        except Exception as e:
            logger.warning(f"Error encoding parameter value {value}: {e}")
            return str(value)
    
    def _add_injection_metadata(self, df: pd.DataFrame, parameters: Dict[str, Any], injection_mode: str) -> pd.DataFrame:
        """
        Add metadata about the parameter injection process
        """
        try:
            metadata_df = df.copy()
            
            # Injection metadata
            metadata_df['param_injection_timestamp'] = datetime.now().isoformat()
            metadata_df['param_injection_mode'] = injection_mode
            metadata_df['param_total_count'] = len(parameters)
            metadata_df['param_injection_version'] = '1.0.0'
            
            # Configuration source metadata
            if 'config_file' in parameters:
                metadata_df['param_config_source'] = parameters['config_file']
            
            # Excel mapper metadata
            if self.excel_mapper:
                metadata_df['param_excel_mapper_enabled'] = True
                metadata_df['param_excel_file'] = str(self.excel_mapper.excel_file_path)
            else:
                metadata_df['param_excel_mapper_enabled'] = False
            
            return metadata_df
            
        except Exception as e:
            logger.error(f"Error adding injection metadata: {e}")
            return df
    
    def _validate_injection_results(self, df: pd.DataFrame, original_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameter injection results
        """
        try:
            # Count injected parameter columns
            param_columns = [col for col in df.columns if col.startswith(self.injection_config.parameter_prefix)]
            
            # Calculate injection statistics
            injection_summary = {
                'total_original_parameters': len(original_parameters),
                'injected_parameter_columns': len(param_columns),
                'injection_ratio': len(param_columns) / len(original_parameters) if original_parameters else 0,
                'dataframe_columns_added': len(param_columns),
                'injection_success': len(param_columns) > 0
            }
            
            return injection_summary
            
        except Exception as e:
            logger.error(f"Error validating injection results: {e}")
            return {'injection_success': False, 'error': str(e)}
    
    def extract_excel_parameters(self, excel_file_path: str) -> Dict[str, Any]:
        """
        Extract parameters from Excel configuration file
        """
        try:
            if not self.excel_mapper:
                # Create temporary mapper
                temp_mapper = ExcelConfigurationMapper(excel_file_path)
                if not temp_mapper.load_excel_configuration():
                    return {}
                
                # Get all module configurations
                all_configs = temp_mapper.get_all_module_configurations()
                
                # Flatten parameters
                flattened_params = {}
                for module_name, module_config in all_configs.items():
                    for param_name, param_value in module_config.parameters.items():
                        flattened_params[f"{module_name}_{param_name}"] = param_value
                
                # Add Excel file metadata
                flattened_params['config_file'] = excel_file_path
                flattened_params['extraction_timestamp'] = datetime.now().isoformat()
                
                return flattened_params
            
            else:
                # Use existing mapper
                if not self.excel_mapper.load_excel_configuration():
                    return {}
                
                all_configs = self.excel_mapper.get_all_module_configurations()
                flattened_params = {}
                
                for module_name, module_config in all_configs.items():
                    for param_name, param_value in module_config.parameters.items():
                        flattened_params[f"{module_name}_{param_name}"] = param_value
                
                flattened_params['config_file'] = excel_file_path
                flattened_params['extraction_timestamp'] = datetime.now().isoformat()
                
                return flattened_params
                
        except Exception as e:
            error_msg = f"Error extracting Excel parameters: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {}
    
    def get_injection_summary(self) -> Dict[str, Any]:
        """
        Get summary of injection configuration and capabilities
        """
        return {
            'injection_config': {
                'include_all_parameters': self.injection_config.include_all_parameters,
                'include_metadata': self.injection_config.include_metadata,
                'parameter_prefix': self.injection_config.parameter_prefix,
                'max_parameter_length': self.injection_config.max_parameter_length
            },
            'parameter_categories': {
                name: {
                    'description': config.description,
                    'parameter_count': len(config.parameters),
                    'encoding_method': config.encoding_method
                }
                for name, config in self.parameter_categories.items()
            },
            'excel_mapper_available': self.excel_mapper is not None,
            'injection_modes': ['comprehensive', 'essential', 'minimal']
        }


class ParameterInjectionError(Exception):
    """Custom exception for parameter injection errors"""
    pass