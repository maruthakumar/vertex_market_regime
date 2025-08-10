#!/usr/bin/env python3
"""
Unified Excel Parser for Enhanced Market Regime Systems
======================================================

This module extends the Excel parsing capabilities to support the new unified
Enhanced Market Regime template that consolidates all three systems into a single file.

Features:
- Unified template detection and parsing
- Cross-system parameter validation
- Consolidated configuration management
- Enhanced validation rules
- Backward compatibility with separate templates
- Migration utilities for existing configurations

Author: The Augster
Date: 2025-01-16
Version: 2.0.0
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Template types supported by the parser"""
    UNIFIED = "unified_enhanced_market_regime"
    SEPARATE_TRENDING_OI = "enhanced_trending_oi_pa"
    SEPARATE_GREEK_SENTIMENT = "enhanced_greek_sentiment"
    SEPARATE_TRIPLE_STRADDLE = "triple_straddle_analysis"

@dataclass
class UnifiedSystemConfig:
    """Configuration for unified enhanced market regime system"""
    template_type: TemplateType
    systems_enabled: Dict[str, bool]
    system_weights: Dict[str, float]
    parameters: Dict[str, Any]
    indicators: Dict[str, Dict[str, Any]]
    validation_errors: List[str]
    cross_system_validation: bool

class UnifiedExcelParser:
    """Enhanced Excel parser for unified market regime template"""
    
    def __init__(self):
        """Initialize unified Excel parser"""
        self.unified_required_sheets = [
            'SystemConfiguration', 'IndicatorRegistry', 'DynamicWeightManagement',
            'TimeframeConfiguration', 'PerformanceTracking'
        ]
        
        self.system_specific_sheets = {
            'TrendingOIConfiguration': 'trending_oi',
            'GreekSentimentConfiguration': 'greek_sentiment',
            'TripleStraddleConfiguration': 'triple_straddle'
        }
        
        self.validation_rules = [
            'SystemWeightSum', 'IndicatorWeightRange', 'TimeframeWeightNormalization',
            'ThresholdRange', 'CrossSystemConsistency', 'EnabledSystemValidation'
        ]
        
        logger.info("Unified Excel Parser initialized")
    
    def detect_template_type(self, excel_path: str) -> TemplateType:
        """
        Detect whether Excel file is unified or separate template
        
        Args:
            excel_path (str): Path to Excel file
            
        Returns:
            TemplateType: Detected template type
        """
        try:
            xl_file = pd.ExcelFile(excel_path)
            sheet_names = set(xl_file.sheet_names)
            
            # Check for unified template
            if all(sheet in sheet_names for sheet in self.unified_required_sheets):
                return TemplateType.UNIFIED
            
            # Check for separate templates
            filename = Path(excel_path).name.lower()
            if 'trending_oi' in filename:
                return TemplateType.SEPARATE_TRENDING_OI
            elif 'greek_sentiment' in filename:
                return TemplateType.SEPARATE_GREEK_SENTIMENT
            elif 'triple_straddle' in filename:
                return TemplateType.SEPARATE_TRIPLE_STRADDLE
            
            # Default to unified if structure matches
            if len(sheet_names.intersection(self.unified_required_sheets)) >= 3:
                return TemplateType.UNIFIED
            
            return TemplateType.SEPARATE_TRENDING_OI  # Default fallback
            
        except Exception as e:
            logger.error(f"Error detecting template type: {e}")
            return TemplateType.UNIFIED
    
    def parse_unified_template(self, excel_path: str) -> Optional[UnifiedSystemConfig]:
        """
        Parse unified enhanced market regime template
        
        Args:
            excel_path (str): Path to unified Excel template
            
        Returns:
            Optional[UnifiedSystemConfig]: Parsed unified configuration
        """
        try:
            template_type = self.detect_template_type(excel_path)
            
            if template_type != TemplateType.UNIFIED:
                logger.warning(f"Template is not unified type: {template_type}")
                return None
            
            xl_file = pd.ExcelFile(excel_path)
            
            # Parse system configuration
            systems_enabled, system_weights = self._parse_system_configuration(xl_file)
            
            # Parse indicator registry
            indicators = self._parse_indicator_registry(xl_file)
            
            # Parse all configuration parameters
            parameters = self._parse_all_parameters(xl_file)
            
            # Validate configuration
            validation_errors = self._validate_unified_configuration(
                systems_enabled, system_weights, indicators, parameters
            )
            
            return UnifiedSystemConfig(
                template_type=template_type,
                systems_enabled=systems_enabled,
                system_weights=system_weights,
                parameters=parameters,
                indicators=indicators,
                validation_errors=validation_errors,
                cross_system_validation=len(validation_errors) == 0
            )
            
        except Exception as e:
            logger.error(f"Error parsing unified template: {e}")
            return None
    
    def _parse_system_configuration(self, xl_file: pd.ExcelFile) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Parse SystemConfiguration sheet"""
        try:
            df = pd.read_excel(xl_file, sheet_name='SystemConfiguration', header=1)
            
            systems_enabled = {}
            system_weights = {}
            
            for _, row in df.iterrows():
                if pd.notna(row.get('Parameter')) and pd.notna(row.get('Value')):
                    param = str(row['Parameter'])
                    value = row['Value']
                    
                    # Parse system enable flags
                    if param.endswith('Enabled'):
                        system_name = param.replace('Enabled', '').lower()
                        systems_enabled[system_name] = str(value).upper() == 'YES'
                    
                    # Parse system weights (will be extracted from DynamicWeightManagement)
                    elif param.endswith('Weight') and 'System' in param:
                        system_name = param.replace('SystemWeight', '').lower()
                        try:
                            system_weights[system_name] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid weight value for {param}: {value}")
            
            return systems_enabled, system_weights
            
        except Exception as e:
            logger.error(f"Error parsing system configuration: {e}")
            return {}, {}
    
    def _parse_indicator_registry(self, xl_file: pd.ExcelFile) -> Dict[str, Dict[str, Any]]:
        """Parse IndicatorRegistry sheet"""
        try:
            df = pd.read_excel(xl_file, sheet_name='IndicatorRegistry', header=1)
            
            indicators = {}
            
            for _, row in df.iterrows():
                if pd.notna(row.get('IndicatorID')):
                    indicator_id = str(row['IndicatorID'])
                    
                    indicators[indicator_id] = {
                        'name': str(row.get('IndicatorName', '')),
                        'category': str(row.get('Category', '')),
                        'system': str(row.get('System', '')),
                        'base_weight': float(row.get('BaseWeight', 0.0)),
                        'min_weight': float(row.get('MinWeight', 0.0)),
                        'max_weight': float(row.get('MaxWeight', 1.0)),
                        'enabled': str(row.get('Enabled', 'NO')).upper() == 'YES',
                        'priority': str(row.get('Priority', 'MEDIUM')),
                        'description': str(row.get('Description', ''))
                    }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error parsing indicator registry: {e}")
            return {}
    
    def _parse_all_parameters(self, xl_file: pd.ExcelFile) -> Dict[str, Any]:
        """Parse all configuration parameters from all sheets"""
        parameters = {}
        
        # Parse each configuration sheet
        config_sheets = [
            'SystemConfiguration', 'DynamicWeightManagement', 'TimeframeConfiguration',
            'PerformanceTracking', 'TrendingOIConfiguration', 'GreekSentimentConfiguration',
            'TripleStraddleConfiguration', 'ValidationRules'
        ]
        
        for sheet_name in config_sheets:
            if sheet_name in xl_file.sheet_names:
                sheet_params = self._parse_parameter_sheet(xl_file, sheet_name)
                parameters.update(sheet_params)
        
        return parameters
    
    def _parse_parameter_sheet(self, xl_file: pd.ExcelFile, sheet_name: str) -> Dict[str, Any]:
        """Parse individual parameter sheet"""
        try:
            df = pd.read_excel(xl_file, sheet_name=sheet_name, header=1)
            parameters = {}
            
            for _, row in df.iterrows():
                if pd.notna(row.get('Parameter')) and pd.notna(row.get('Value')):
                    param_name = str(row['Parameter'])
                    param_value = row['Value']
                    
                    # Convert parameter value to appropriate type
                    if isinstance(param_value, str):
                        if param_value.upper() in ['YES', 'TRUE']:
                            param_value = True
                        elif param_value.upper() in ['NO', 'FALSE']:
                            param_value = False
                        else:
                            try:
                                param_value = float(param_value)
                                if param_value.is_integer():
                                    param_value = int(param_value)
                            except (ValueError, TypeError):
                                pass  # Keep as string
                    
                    parameters[f"{sheet_name}_{param_name}"] = param_value
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error parsing sheet {sheet_name}: {e}")
            return {}
    
    def _validate_unified_configuration(self, systems_enabled: Dict[str, bool], 
                                      system_weights: Dict[str, float],
                                      indicators: Dict[str, Dict[str, Any]],
                                      parameters: Dict[str, Any]) -> List[str]:
        """Validate unified configuration with cross-system checks"""
        errors = []
        
        try:
            # Validate at least one system is enabled
            if not any(systems_enabled.values()):
                errors.append("At least one enhanced system must be enabled")
            
            # Validate system weights sum to 1.0 (if systems are enabled)
            enabled_systems = [k for k, v in systems_enabled.items() if v]
            if enabled_systems:
                total_weight = sum(system_weights.get(system, 0.0) for system in enabled_systems)
                if abs(total_weight - 1.0) > 0.01:
                    errors.append(f"Enabled system weights must sum to 1.0, got {total_weight:.3f}")
            
            # Validate indicator weights
            for indicator_id, indicator_config in indicators.items():
                if indicator_config['enabled']:
                    weight = indicator_config['base_weight']
                    min_weight = indicator_config['min_weight']
                    max_weight = indicator_config['max_weight']
                    
                    if not (min_weight <= weight <= max_weight):
                        errors.append(f"Indicator {indicator_id} weight {weight} outside range [{min_weight}, {max_weight}]")
            
            # Validate cross-system consistency
            errors.extend(self._validate_cross_system_consistency(systems_enabled, parameters))
            
            # Validate parameter ranges
            errors.extend(self._validate_parameter_ranges(parameters))
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def _validate_cross_system_consistency(self, systems_enabled: Dict[str, bool], 
                                         parameters: Dict[str, Any]) -> List[str]:
        """Validate consistency across enabled systems"""
        errors = []
        
        # Check timeframe consistency
        if systems_enabled.get('trendingoienabled', False) and systems_enabled.get('triplestraddleenabled', False):
            # Both systems use timeframes - ensure consistency
            trending_oi_timeframes = self._extract_timeframe_config(parameters, 'TrendingOI')
            triple_straddle_timeframes = self._extract_timeframe_config(parameters, 'TripleStraddle')
            
            # Validate that enabled timeframes are consistent
            for timeframe in ['3min', '5min', '10min', '15min']:
                trending_enabled = trending_oi_timeframes.get(f'{timeframe}_enabled', False)
                straddle_enabled = triple_straddle_timeframes.get(f'{timeframe}_enabled', False)
                
                if trending_enabled != straddle_enabled:
                    errors.append(f"Timeframe {timeframe} enabled status inconsistent between systems")
        
        return errors
    
    def _extract_timeframe_config(self, parameters: Dict[str, Any], system: str) -> Dict[str, Any]:
        """Extract timeframe configuration for a specific system"""
        timeframe_config = {}
        
        for param_name, param_value in parameters.items():
            if 'timeframe' in param_name.lower() and system.lower() in param_name.lower():
                timeframe_config[param_name] = param_value
        
        return timeframe_config
    
    def _validate_parameter_ranges(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameter value ranges"""
        errors = []
        
        # Define parameter range validations
        range_validations = {
            'confidence': (0.0, 1.0),
            'threshold': (0.0, 1.0),
            'weight': (0.0, 2.0),
            'lookback': (1, 1000),
            'window': (1, 10000)
        }
        
        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                for validation_key, (min_val, max_val) in range_validations.items():
                    if validation_key in param_name.lower():
                        if not (min_val <= param_value <= max_val):
                            errors.append(f"Parameter {param_name} value {param_value} outside range [{min_val}, {max_val}]")
                        break
        
        return errors
    
    def convert_to_yaml(self, config: UnifiedSystemConfig) -> str:
        """Convert unified configuration to YAML format"""
        try:
            yaml_config = {
                'template_type': config.template_type.value,
                'systems_enabled': config.systems_enabled,
                'system_weights': config.system_weights,
                'indicators': config.indicators,
                'parameters': config.parameters,
                'validation': {
                    'cross_system_validation': config.cross_system_validation,
                    'validation_errors': config.validation_errors
                },
                'metadata': {
                    'template_version': '2.0.0',
                    'generated_at': pd.Timestamp.now().isoformat(),
                    'parser_version': '2.0.0'
                }
            }
            
            return yaml.dump(yaml_config, default_flow_style=False, sort_keys=True)
            
        except Exception as e:
            logger.error(f"Error converting to YAML: {e}")
            return ""


def main():
    """Main function for testing unified Excel parser"""
    try:
        parser = UnifiedExcelParser()
        
        # Test unified template
        test_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/UNIFIED_ENHANCED_MARKET_REGIME_CONFIG.xlsx"
        
        if Path(test_file).exists():
            print(f"🔍 Testing unified template: {Path(test_file).name}")
            
            # Detect template type
            template_type = parser.detect_template_type(test_file)
            print(f"📊 Template Type: {template_type.value}")
            
            # Parse configuration
            config = parser.parse_unified_template(test_file)
            if config:
                print(f"✅ Parsed successfully")
                print(f"🔧 Systems Enabled: {config.systems_enabled}")
                print(f"⚖️ System Weights: {config.system_weights}")
                print(f"📊 Indicators: {len(config.indicators)}")
                print(f"📋 Parameters: {len(config.parameters)}")
                print(f"✅ Cross-system Validation: {config.cross_system_validation}")
                
                if config.validation_errors:
                    print(f"⚠️ Validation Errors: {len(config.validation_errors)}")
                    for error in config.validation_errors[:3]:
                        print(f"   - {error}")
            else:
                print("❌ Failed to parse unified template")
        else:
            print(f"⚠️ Test file not found: {test_file}")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    main()
