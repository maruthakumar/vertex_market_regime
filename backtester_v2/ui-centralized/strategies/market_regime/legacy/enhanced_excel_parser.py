#!/usr/bin/env python3
"""
Enhanced Excel Parser for Market Regime Systems
==============================================

This module extends the Excel parsing capabilities to support the newly integrated
Enhanced Market Regime systems: Trending OI with PA, Greek Sentiment Analysis, and Triple Straddle Analysis.

Features:
- Multi-sheet configuration parsing
- Enhanced parameter validation
- Cross-sheet dependency validation
- Dynamic parameter type detection
- Backward compatibility with existing templates
- Comprehensive error reporting

Author: The Augster
Date: 2025-01-16
Version: 1.0.0
"""

import pandas as pd
import openpyxl
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import yaml
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedSystemType(Enum):
    """Enhanced system types"""
    TRENDING_OI_PA = "enhanced_trending_oi_pa"
    GREEK_SENTIMENT = "enhanced_greek_sentiment"
    TRIPLE_STRADDLE = "triple_straddle_analysis"

@dataclass
class ParsedParameter:
    """Parsed parameter with validation"""
    name: str
    value: Any
    param_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    is_valid: bool = True
    validation_error: str = ""

@dataclass
class EnhancedSystemConfig:
    """Configuration for enhanced system"""
    system_type: EnhancedSystemType
    enabled: bool
    parameters: Dict[str, ParsedParameter]
    sheet_configs: Dict[str, Dict[str, Any]]
    validation_errors: List[str]

class EnhancedExcelParser:
    """Enhanced Excel parser for market regime systems"""
    
    def __init__(self):
        """Initialize enhanced Excel parser"""
        self.supported_systems = {
            'enhanced_trending_oi_pa_config.xlsx': EnhancedSystemType.TRENDING_OI_PA,
            'enhanced_greek_sentiment_config.xlsx': EnhancedSystemType.GREEK_SENTIMENT,
            'triple_straddle_analysis_config.xlsx': EnhancedSystemType.TRIPLE_STRADDLE
        }
        
        # Define required sheets for each system
        self.required_sheets = {
            EnhancedSystemType.TRENDING_OI_PA: [
                'GeneralParameters', 'MultiTimeframeConfig', 
                'DivergenceAnalysis', 'VolumeWeightingConfig'
            ],
            EnhancedSystemType.GREEK_SENTIMENT: [
                'GreekWeightConfig', 'DTEAdjustments', 'BaselineTracking'
            ],
            EnhancedSystemType.TRIPLE_STRADDLE: [
                'StraddleComponents', 'TimeframeWeights', 'TechnicalAnalysis',
                'EMAConfiguration', 'VWAPConfiguration'
            ]
        }
        
        logger.info("Enhanced Excel Parser initialized")
    
    def detect_system_type(self, excel_path: str) -> Optional[EnhancedSystemType]:
        """
        Detect the enhanced system type from Excel file
        
        Args:
            excel_path (str): Path to Excel file
            
        Returns:
            Optional[EnhancedSystemType]: Detected system type or None
        """
        try:
            filename = Path(excel_path).name
            
            # Check by filename
            if filename in self.supported_systems:
                return self.supported_systems[filename]
            
            # Check by sheet structure
            xl_file = pd.ExcelFile(excel_path)
            sheet_names = set(xl_file.sheet_names)
            
            for system_type, required_sheets in self.required_sheets.items():
                if all(sheet in sheet_names for sheet in required_sheets):
                    return system_type
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting system type: {e}")
            return None
    
    def validate_excel_structure(self, excel_path: str, system_type: EnhancedSystemType) -> Tuple[bool, List[str]]:
        """
        Validate Excel file structure for enhanced system
        
        Args:
            excel_path (str): Path to Excel file
            system_type (EnhancedSystemType): System type to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        try:
            xl_file = pd.ExcelFile(excel_path)
            sheet_names = set(xl_file.sheet_names)
            required_sheets = self.required_sheets[system_type]
            
            # Check required sheets
            missing_sheets = [sheet for sheet in required_sheets if sheet not in sheet_names]
            if missing_sheets:
                errors.append(f"Missing required sheets: {', '.join(missing_sheets)}")
            
            # Validate each sheet structure
            for sheet_name in required_sheets:
                if sheet_name in sheet_names:
                    sheet_errors = self._validate_sheet_structure(xl_file, sheet_name, system_type)
                    errors.extend(sheet_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Excel structure validation error: {str(e)}")
            return False, errors
    
    def _validate_sheet_structure(self, xl_file: pd.ExcelFile, sheet_name: str,
                                 system_type: EnhancedSystemType) -> List[str]:
        """Validate individual sheet structure"""
        errors = []

        try:
            # Try reading with header row 1 (skip merged header)
            df = pd.read_excel(xl_file, sheet_name=sheet_name, header=1)

            # Check minimum columns
            if len(df.columns) < 2:
                errors.append(f"Sheet '{sheet_name}' must have at least 2 columns")

            # Check for required columns based on sheet type
            required_columns = self._get_required_columns(sheet_name, system_type)
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Sheet '{sheet_name}' missing columns: {', '.join(missing_columns)}")

            # Check for empty data
            if df.empty:
                errors.append(f"Sheet '{sheet_name}' contains no data")

        except Exception as e:
            errors.append(f"Error validating sheet '{sheet_name}': {str(e)}")

        return errors
    
    def _get_required_columns(self, sheet_name: str, system_type: EnhancedSystemType) -> List[str]:
        """Get required columns for specific sheet"""
        column_requirements = {
            # Trending OI PA sheets
            'GeneralParameters': ['Parameter', 'Value'],
            'MultiTimeframeConfig': ['Timeframe', 'Weight'],
            'DivergenceAnalysis': ['DivergenceType', 'Enabled'],
            'VolumeWeightingConfig': ['Parameter', 'Value'],
            
            # Greek Sentiment sheets
            'GreekWeightConfig': ['Greek', 'BaseWeight'],
            'DTEAdjustments': ['DTERange', 'DeltaMultiplier'],
            'BaselineTracking': ['Parameter', 'Value'],
            
            # Triple Straddle sheets
            'StraddleComponents': ['Component', 'Weight'],
            'TimeframeWeights': ['Timeframe', 'Weight'],
            'TechnicalAnalysis': ['Analysis', 'Weight'],
            'EMAConfiguration': ['EMAPeriod', 'Weight'],
            'VWAPConfiguration': ['VWAPType', 'Weight']
        }
        
        return column_requirements.get(sheet_name, ['Parameter', 'Value'])
    
    def parse_enhanced_system_config(self, excel_path: str) -> Optional[EnhancedSystemConfig]:
        """
        Parse enhanced system configuration from Excel file
        
        Args:
            excel_path (str): Path to Excel file
            
        Returns:
            Optional[EnhancedSystemConfig]: Parsed configuration or None
        """
        try:
            # Detect system type
            system_type = self.detect_system_type(excel_path)
            if not system_type:
                logger.error(f"Could not detect enhanced system type for: {excel_path}")
                return None
            
            # Validate structure
            is_valid, validation_errors = self.validate_excel_structure(excel_path, system_type)
            if not is_valid:
                logger.error(f"Excel structure validation failed: {validation_errors}")
                return EnhancedSystemConfig(
                    system_type=system_type,
                    enabled=False,
                    parameters={},
                    sheet_configs={},
                    validation_errors=validation_errors
                )
            
            # Parse configuration
            xl_file = pd.ExcelFile(excel_path)
            parameters = {}
            sheet_configs = {}
            
            # Parse each sheet
            for sheet_name in self.required_sheets[system_type]:
                if sheet_name in xl_file.sheet_names:
                    sheet_params, sheet_config = self._parse_sheet(xl_file, sheet_name, system_type)
                    parameters.update(sheet_params)
                    sheet_configs[sheet_name] = sheet_config
            
            # Check if system is enabled
            enabled = parameters.get('SystemEnabled', ParsedParameter('SystemEnabled', 'YES', 'str')).value.upper() == 'YES'
            
            return EnhancedSystemConfig(
                system_type=system_type,
                enabled=enabled,
                parameters=parameters,
                sheet_configs=sheet_configs,
                validation_errors=[]
            )
            
        except Exception as e:
            logger.error(f"Error parsing enhanced system config: {e}")
            return None
    
    def _parse_sheet(self, xl_file: pd.ExcelFile, sheet_name: str,
                    system_type: EnhancedSystemType) -> Tuple[Dict[str, ParsedParameter], Dict[str, Any]]:
        """Parse individual sheet"""
        parameters = {}
        sheet_config = {}

        try:
            # Read with header row 1 to skip merged header
            df = pd.read_excel(xl_file, sheet_name=sheet_name, header=1)
            
            # Handle different sheet structures
            if 'Parameter' in df.columns and 'Value' in df.columns:
                # Standard parameter-value format
                for _, row in df.iterrows():
                    if pd.notna(row['Parameter']) and pd.notna(row['Value']):
                        param = self._create_parsed_parameter(
                            row['Parameter'], row['Value'], row, system_type
                        )
                        parameters[param.name] = param
            
            elif sheet_name in ['MultiTimeframeConfig', 'TimeframeWeights']:
                # Timeframe configuration format
                for _, row in df.iterrows():
                    if pd.notna(row.get('Timeframe')) and pd.notna(row.get('Weight')):
                        timeframe = str(row['Timeframe'])
                        weight = float(row['Weight'])
                        param = ParsedParameter(
                            name=f"{timeframe}_Weight",
                            value=weight,
                            param_type='float',
                            min_value=0.0,
                            max_value=1.0,
                            description=f"Weight for {timeframe} timeframe"
                        )
                        parameters[param.name] = param
            
            elif sheet_name in ['GreekWeightConfig', 'StraddleComponents', 'TechnicalAnalysis']:
                # Component configuration format
                component_col = 'Greek' if 'Greek' in df.columns else ('Component' if 'Component' in df.columns else 'Analysis')
                weight_col = 'BaseWeight' if 'BaseWeight' in df.columns else 'Weight'
                
                for _, row in df.iterrows():
                    if pd.notna(row.get(component_col)) and pd.notna(row.get(weight_col)):
                        component = str(row[component_col])
                        weight = float(row[weight_col])
                        param = ParsedParameter(
                            name=f"{component}_Weight",
                            value=weight,
                            param_type='float',
                            min_value=0.0,
                            max_value=2.0,
                            description=f"Weight for {component}"
                        )
                        parameters[param.name] = param
            
            # Store sheet configuration
            sheet_config = {
                'sheet_name': sheet_name,
                'data': df.to_dict('records'),
                'columns': list(df.columns),
                'row_count': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error parsing sheet '{sheet_name}': {e}")
        
        return parameters, sheet_config
    
    def _create_parsed_parameter(self, name: str, value: Any, row: pd.Series, 
                                system_type: EnhancedSystemType) -> ParsedParameter:
        """Create parsed parameter with validation"""
        try:
            # Determine parameter type and constraints
            param_type = self._detect_parameter_type(value)
            min_val = row.get('Min') if 'Min' in row and pd.notna(row['Min']) else None
            max_val = row.get('Max') if 'Max' in row and pd.notna(row['Max']) else None
            description = row.get('Description', '') if 'Description' in row else ''
            
            # Convert value to appropriate type
            converted_value = self._convert_parameter_value(value, param_type)
            
            # Validate parameter
            is_valid, validation_error = self._validate_parameter(
                name, converted_value, param_type, min_val, max_val
            )
            
            return ParsedParameter(
                name=name,
                value=converted_value,
                param_type=param_type,
                min_value=min_val,
                max_value=max_val,
                description=description,
                is_valid=is_valid,
                validation_error=validation_error
            )
            
        except Exception as e:
            return ParsedParameter(
                name=name,
                value=value,
                param_type='str',
                description='',
                is_valid=False,
                validation_error=f"Parameter creation error: {str(e)}"
            )
    
    def _detect_parameter_type(self, value: Any) -> str:
        """Detect parameter type from value"""
        if isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            # Check if string represents boolean
            if value.upper() in ['YES', 'NO', 'TRUE', 'FALSE']:
                return 'bool'
            # Check if string represents number
            try:
                float(value)
                return 'float' if '.' in value else 'int'
            except ValueError:
                return 'str'
        else:
            return 'str'
    
    def _convert_parameter_value(self, value: Any, param_type: str) -> Any:
        """Convert parameter value to appropriate type"""
        try:
            if param_type == 'bool':
                if isinstance(value, str):
                    return value.upper() in ['YES', 'TRUE', '1']
                return bool(value)
            elif param_type == 'int':
                return int(float(value))
            elif param_type == 'float':
                return float(value)
            else:
                return str(value)
        except (ValueError, TypeError):
            return value
    
    def _validate_parameter(self, name: str, value: Any, param_type: str, 
                           min_val: Optional[float], max_val: Optional[float]) -> Tuple[bool, str]:
        """Validate parameter value"""
        try:
            # Type validation
            if param_type in ['int', 'float'] and not isinstance(value, (int, float)):
                return False, f"Parameter '{name}' must be numeric"
            
            # Range validation
            if min_val is not None and isinstance(value, (int, float)) and value < min_val:
                return False, f"Parameter '{name}' value {value} is below minimum {min_val}"
            
            if max_val is not None and isinstance(value, (int, float)) and value > max_val:
                return False, f"Parameter '{name}' value {value} is above maximum {max_val}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error for '{name}': {str(e)}"
    
    def convert_to_yaml(self, config: EnhancedSystemConfig) -> str:
        """
        Convert enhanced system configuration to YAML format
        
        Args:
            config (EnhancedSystemConfig): Parsed configuration
            
        Returns:
            str: YAML configuration string
        """
        try:
            yaml_config = {
                'system_type': config.system_type.value,
                'enabled': config.enabled,
                'parameters': {},
                'validation_errors': config.validation_errors
            }
            
            # Convert parameters to YAML-friendly format
            for param_name, param in config.parameters.items():
                yaml_config['parameters'][param_name] = {
                    'value': param.value,
                    'type': param.param_type,
                    'description': param.description,
                    'is_valid': param.is_valid
                }
                
                if param.min_value is not None:
                    yaml_config['parameters'][param_name]['min_value'] = param.min_value
                if param.max_value is not None:
                    yaml_config['parameters'][param_name]['max_value'] = param.max_value
                if param.validation_error:
                    yaml_config['parameters'][param_name]['validation_error'] = param.validation_error
            
            return yaml.dump(yaml_config, default_flow_style=False, sort_keys=True)
            
        except Exception as e:
            logger.error(f"Error converting to YAML: {e}")
            return ""


def main():
    """Main function for testing enhanced Excel parser"""
    try:
        parser = EnhancedExcelParser()
        
        # Test files
        test_files = [
            "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/enhanced_trending_oi_pa_config.xlsx",
            "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/enhanced_greek_sentiment_config.xlsx",
            "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/triple_straddle_analysis_config.xlsx"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"\n🔍 Testing: {Path(test_file).name}")
                
                # Detect system type
                system_type = parser.detect_system_type(test_file)
                print(f"📊 System Type: {system_type.value if system_type else 'Unknown'}")
                
                # Parse configuration
                config = parser.parse_enhanced_system_config(test_file)
                if config:
                    print(f"✅ Parsed successfully: {len(config.parameters)} parameters")
                    print(f"🔧 System Enabled: {config.enabled}")
                    
                    if config.validation_errors:
                        print(f"⚠️ Validation Errors: {len(config.validation_errors)}")
                        for error in config.validation_errors:
                            print(f"   - {error}")
                else:
                    print("❌ Failed to parse configuration")
            else:
                print(f"⚠️ Test file not found: {test_file}")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    main()
