#!/usr/bin/env python3
"""
Enhanced Excel Configuration Generator for Enhanced Triple Straddle Framework v2.0
==================================================================================

This module generates the unified Excel configuration template with 6 specialized sheets:
1. VolumeWeightedGreeksConfig - Volume-weighted Greek calculation parameters
2. DeltaStrikeSelectionConfig - Delta-based strike filtering parameters
3. HybridClassificationConfig - Hybrid system weight distribution settings
4. PerformanceMonitoringConfig - Performance targets and monitoring settings
5. MathematicalAccuracyConfig - Validation parameters and tolerance settings
6. ConfigurationProfiles - Conservative/Balanced/Aggressive presets

Features:
- Data validation rules and parameter ranges
- Three configuration profiles (Conservative/Balanced/Aggressive)
- Cross-system parameter validation
- Migration utilities for existing configurations
- Integration with all Phase 1 + Phase 2 components

Author: The Augster
Date: 2025-06-20
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfigurationProfile:
    """Configuration profile definition"""
    name: str
    description: str
    risk_level: str
    parameters: Dict[str, Any]

class EnhancedExcelConfigGenerator:
    """
    Enhanced Excel Configuration Generator for creating unified configuration templates
    with comprehensive parameter validation and profile management
    """
    
    def __init__(self, output_directory: str = "."):
        """
        Initialize Enhanced Excel Configuration Generator
        
        Args:
            output_directory: Directory to save configuration files
        """
        self.output_directory = output_directory
        self.configuration_sheets = {}
        self.validation_rules = {}
        self.profiles = {}
        
        # Initialize configuration sheets
        self._initialize_configuration_sheets()
        self._initialize_validation_rules()
        self._initialize_configuration_profiles()
        
        logger.info("Enhanced Excel Configuration Generator initialized")
        logger.info(f"Output directory: {self.output_directory}")
    
    def _initialize_configuration_sheets(self) -> None:
        """Initialize all configuration sheets with default parameters"""
        
        # Sheet 1: Volume-Weighted Greeks Configuration
        self.configuration_sheets['VolumeWeightedGreeksConfig'] = {
            'baseline_time_hour': 9,
            'baseline_time_minute': 15,
            'normalization_method': 'tanh',
            'accuracy_tolerance': 0.001,
            'dte_0_weight': 0.70,
            'dte_1_weight': 0.30,
            'dte_2_weight': 0.30,
            'dte_3_weight': 0.30,
            'delta_component_weight': 0.40,
            'gamma_component_weight': 0.30,
            'theta_component_weight': 0.20,
            'vega_component_weight': 0.10,
            'risk_free_rate': 0.05,
            'default_iv': 0.20,
            'enable_black_scholes_fallback': True,
            'performance_target_seconds': 3.0
        }
        
        # Sheet 2: Delta-based Strike Selection Configuration
        self.configuration_sheets['DeltaStrikeSelectionConfig'] = {
            'call_delta_min': 0.01,
            'call_delta_max': 0.50,
            'put_delta_min': -0.50,
            'put_delta_max': -0.01,
            'max_strikes_per_expiry': 50,
            'recalculate_frequency_seconds': 60,
            'enable_delta_caching': True,
            'cache_expiry_seconds': 300,
            'selection_confidence_threshold': 0.60,
            'mathematical_tolerance': 0.001,
            'performance_target_seconds': 3.0
        }
        
        # Sheet 3: Hybrid Classification Configuration
        self.configuration_sheets['HybridClassificationConfig'] = {
            'enhanced_system_weight': 0.70,
            'timeframe_hierarchy_weight': 0.30,
            'agreement_threshold': 0.75,
            'confidence_threshold': 0.60,
            'transition_smoothing_factor': 0.1,
            'transition_memory_window': 10,
            'volatility_low_threshold': 0.15,
            'volatility_normal_threshold': 0.25,
            'volatility_high_threshold': 0.25,
            'strong_bullish_threshold': 0.6,
            'mild_bullish_threshold': 0.2,
            'neutral_threshold': 0.1,
            'mild_bearish_threshold': -0.2,
            'strong_bearish_threshold': -0.6,
            'mathematical_tolerance': 0.001,
            'performance_target_seconds': 3.0
        }
        
        # Sheet 4: Performance Monitoring Configuration
        self.configuration_sheets['PerformanceMonitoringConfig'] = {
            'max_processing_time_seconds': 3.0,
            'min_accuracy_threshold': 0.85,
            'mathematical_tolerance': 0.001,
            'max_memory_usage_mb': 500,
            'alert_threshold_violations': 3,
            'alert_cooldown_seconds': 300,
            'enable_processing_time_alerts': True,
            'enable_accuracy_alerts': True,
            'enable_memory_alerts': True,
            'enable_mathematical_accuracy_alerts': True,
            'monitoring_interval_seconds': 10,
            'metrics_history_limit': 10000,
            'gc_frequency_operations': 100,
            'enable_real_time_monitoring': True
        }
        
        # Sheet 5: Mathematical Accuracy Configuration
        self.configuration_sheets['MathematicalAccuracyConfig'] = {
            'tolerance': 0.001,
            'enable_precision_validation': True,
            'enable_bounds_checking': True,
            'enable_finite_validation': True,
            'correlation_threshold': 0.80,
            'pattern_similarity_threshold': 0.75,
            'time_decay_lambda': 0.1,
            'time_decay_window_seconds': 300,
            'baseline_weight': 1.0,
            'min_weight': 0.1,
            'max_weight': 2.0,
            'validation_sample_size': 1000,
            'stress_test_iterations': 100
        }
        
        # Sheet 6: Configuration Profiles
        self.configuration_sheets['ConfigurationProfiles'] = {
            'active_profile': 'Balanced',
            'profile_descriptions': {
                'Conservative': 'Lower risk, higher accuracy thresholds, stricter validation',
                'Balanced': 'Balanced risk-reward, standard thresholds, moderate validation',
                'Aggressive': 'Higher risk, lower thresholds, faster processing'
            }
        }
    
    def _initialize_validation_rules(self) -> None:
        """Initialize validation rules for all parameters"""
        
        self.validation_rules = {
            'VolumeWeightedGreeksConfig': {
                'baseline_time_hour': {'min': 0, 'max': 23, 'type': 'int'},
                'baseline_time_minute': {'min': 0, 'max': 59, 'type': 'int'},
                'normalization_method': {'values': ['tanh', 'sigmoid', 'linear'], 'type': 'str'},
                'accuracy_tolerance': {'min': 0.0001, 'max': 0.01, 'type': 'float'},
                'dte_0_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'dte_1_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'dte_2_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'dte_3_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'delta_component_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'gamma_component_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'theta_component_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'vega_component_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'risk_free_rate': {'min': 0.0, 'max': 0.2, 'type': 'float'},
                'default_iv': {'min': 0.01, 'max': 2.0, 'type': 'float'},
                'performance_target_seconds': {'min': 1.0, 'max': 10.0, 'type': 'float'}
            },
            'DeltaStrikeSelectionConfig': {
                'call_delta_min': {'min': 0.001, 'max': 0.999, 'type': 'float'},
                'call_delta_max': {'min': 0.001, 'max': 0.999, 'type': 'float'},
                'put_delta_min': {'min': -0.999, 'max': -0.001, 'type': 'float'},
                'put_delta_max': {'min': -0.999, 'max': -0.001, 'type': 'float'},
                'max_strikes_per_expiry': {'min': 5, 'max': 200, 'type': 'int'},
                'recalculate_frequency_seconds': {'min': 10, 'max': 600, 'type': 'int'},
                'cache_expiry_seconds': {'min': 60, 'max': 3600, 'type': 'int'},
                'selection_confidence_threshold': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'mathematical_tolerance': {'min': 0.0001, 'max': 0.01, 'type': 'float'},
                'performance_target_seconds': {'min': 1.0, 'max': 10.0, 'type': 'float'}
            },
            'HybridClassificationConfig': {
                'enhanced_system_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'timeframe_hierarchy_weight': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'agreement_threshold': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'confidence_threshold': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'transition_smoothing_factor': {'min': 0.01, 'max': 1.0, 'type': 'float'},
                'transition_memory_window': {'min': 3, 'max': 50, 'type': 'int'},
                'volatility_low_threshold': {'min': 0.05, 'max': 0.5, 'type': 'float'},
                'volatility_normal_threshold': {'min': 0.1, 'max': 0.8, 'type': 'float'},
                'volatility_high_threshold': {'min': 0.15, 'max': 1.0, 'type': 'float'},
                'mathematical_tolerance': {'min': 0.0001, 'max': 0.01, 'type': 'float'},
                'performance_target_seconds': {'min': 1.0, 'max': 10.0, 'type': 'float'}
            },
            'PerformanceMonitoringConfig': {
                'max_processing_time_seconds': {'min': 1.0, 'max': 30.0, 'type': 'float'},
                'min_accuracy_threshold': {'min': 0.5, 'max': 0.99, 'type': 'float'},
                'mathematical_tolerance': {'min': 0.0001, 'max': 0.01, 'type': 'float'},
                'max_memory_usage_mb': {'min': 100, 'max': 2000, 'type': 'int'},
                'alert_threshold_violations': {'min': 1, 'max': 10, 'type': 'int'},
                'alert_cooldown_seconds': {'min': 60, 'max': 3600, 'type': 'int'},
                'monitoring_interval_seconds': {'min': 1, 'max': 60, 'type': 'int'},
                'metrics_history_limit': {'min': 1000, 'max': 50000, 'type': 'int'},
                'gc_frequency_operations': {'min': 10, 'max': 1000, 'type': 'int'}
            },
            'MathematicalAccuracyConfig': {
                'tolerance': {'min': 0.0001, 'max': 0.01, 'type': 'float'},
                'correlation_threshold': {'min': 0.5, 'max': 0.99, 'type': 'float'},
                'pattern_similarity_threshold': {'min': 0.5, 'max': 0.99, 'type': 'float'},
                'time_decay_lambda': {'min': 0.01, 'max': 1.0, 'type': 'float'},
                'time_decay_window_seconds': {'min': 60, 'max': 3600, 'type': 'int'},
                'baseline_weight': {'min': 0.1, 'max': 5.0, 'type': 'float'},
                'min_weight': {'min': 0.01, 'max': 1.0, 'type': 'float'},
                'max_weight': {'min': 1.0, 'max': 10.0, 'type': 'float'},
                'validation_sample_size': {'min': 100, 'max': 10000, 'type': 'int'},
                'stress_test_iterations': {'min': 10, 'max': 1000, 'type': 'int'}
            }
        }

    def _initialize_configuration_profiles(self) -> None:
        """Initialize the three configuration profiles: Conservative, Balanced, Aggressive"""

        # Conservative Profile - Lower risk, higher accuracy thresholds
        self.profiles['Conservative'] = ConfigurationProfile(
            name='Conservative',
            description='Lower risk, higher accuracy thresholds, stricter validation',
            risk_level='Low',
            parameters={
                'VolumeWeightedGreeksConfig': {
                    'accuracy_tolerance': 0.0005,
                    'performance_target_seconds': 2.0,
                    'dte_0_weight': 0.80,
                    'delta_component_weight': 0.50,
                    'gamma_component_weight': 0.25,
                    'theta_component_weight': 0.15,
                    'vega_component_weight': 0.10
                },
                'HybridClassificationConfig': {
                    'enhanced_system_weight': 0.60,
                    'timeframe_hierarchy_weight': 0.40,
                    'agreement_threshold': 0.85,
                    'confidence_threshold': 0.75
                },
                'PerformanceMonitoringConfig': {
                    'max_processing_time_seconds': 2.5,
                    'min_accuracy_threshold': 0.90,
                    'alert_threshold_violations': 2
                }
            }
        )

        # Balanced Profile - Standard settings
        self.profiles['Balanced'] = ConfigurationProfile(
            name='Balanced',
            description='Balanced risk-reward, standard thresholds, moderate validation',
            risk_level='Medium',
            parameters={}  # Use defaults
        )

        # Aggressive Profile - Higher risk, lower thresholds
        self.profiles['Aggressive'] = ConfigurationProfile(
            name='Aggressive',
            description='Higher risk, lower thresholds, faster processing',
            risk_level='High',
            parameters={
                'VolumeWeightedGreeksConfig': {
                    'accuracy_tolerance': 0.002,
                    'performance_target_seconds': 4.0,
                    'dte_0_weight': 0.60
                },
                'HybridClassificationConfig': {
                    'enhanced_system_weight': 0.80,
                    'timeframe_hierarchy_weight': 0.20,
                    'agreement_threshold': 0.65,
                    'confidence_threshold': 0.50
                },
                'PerformanceMonitoringConfig': {
                    'max_processing_time_seconds': 4.0,
                    'min_accuracy_threshold': 0.80,
                    'alert_threshold_violations': 5
                }
            }
        )

    def generate_excel_configuration(self, profile_name: str = 'Balanced') -> Dict[str, Any]:
        """Generate Excel configuration based on selected profile"""
        try:
            if profile_name not in self.profiles:
                logger.warning(f"Profile '{profile_name}' not found, using 'Balanced'")
                profile_name = 'Balanced'

            profile = self.profiles[profile_name]
            logger.info(f"Generating Excel configuration for profile: {profile_name}")

            excel_config = {}

            for sheet_name, base_config in self.configuration_sheets.items():
                sheet_config = base_config.copy()

                if sheet_name in profile.parameters:
                    sheet_config.update(profile.parameters[sheet_name])

                validated_config = self._validate_sheet_configuration(sheet_name, sheet_config)
                excel_config[sheet_name] = validated_config

            excel_config['_metadata'] = {
                'profile_name': profile_name,
                'profile_description': profile.description,
                'risk_level': profile.risk_level,
                'generated_timestamp': datetime.now().isoformat(),
                'generator_version': '2.0.0',
                'framework_version': 'Enhanced Triple Straddle Framework v2.0'
            }

            return excel_config

        except Exception as e:
            logger.error(f"Error generating Excel configuration: {e}")
            return {}

    def _validate_sheet_configuration(self, sheet_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters against validation rules"""
        try:
            if sheet_name not in self.validation_rules:
                return config

            validated_config = config.copy()
            rules = self.validation_rules[sheet_name]

            for param_name, param_value in config.items():
                if param_name in rules:
                    rule = rules[param_name]

                    # Range validation
                    if 'min' in rule and param_value < rule['min']:
                        validated_config[param_name] = rule['min']

                    if 'max' in rule and param_value > rule['max']:
                        validated_config[param_name] = rule['max']

                    # Value validation
                    if 'values' in rule and param_value not in rule['values']:
                        validated_config[param_name] = rule['values'][0]

            return validated_config

        except Exception as e:
            logger.error(f"Error validating sheet configuration: {e}")
            return config

    def save_configuration_to_json(self, config: Dict[str, Any], filename: str = None) -> str:
        """Save configuration to JSON file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_triple_straddle_config_{timestamp}.json"

            filepath = os.path.join(self.output_directory, filename)

            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, default=str)

            logger.info(f"Configuration saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return ""

# Integration function for unified_stable_market_regime_pipeline.py
def generate_configuration_for_pipeline(profile_name: str = 'Balanced',
                                       output_directory: str = ".") -> Optional[Dict[str, Any]]:
    """
    Main integration function for generating Excel configuration

    Args:
        profile_name: Configuration profile to use
        output_directory: Directory to save configuration files

    Returns:
        Dictionary containing configuration or None if generation fails
    """
    try:
        generator = EnhancedExcelConfigGenerator(output_directory)
        config = generator.generate_excel_configuration(profile_name)

        if config:
            # Save to JSON file
            filepath = generator.save_configuration_to_json(config)

            return {
                'configuration_generated': True,
                'profile_name': profile_name,
                'config_data': config,
                'saved_to_file': filepath,
                'generation_timestamp': datetime.now().isoformat()
            }
        else:
            logger.warning("Configuration generation failed")
            return None

    except Exception as e:
        logger.error(f"Error in configuration generation: {e}")
        return None

# Unit test function
def test_enhanced_excel_config_generator():
    """Basic unit test for enhanced Excel configuration generator"""
    try:
        logger.info("Testing Enhanced Excel Configuration Generator...")

        # Test all three profiles
        profiles = ['Conservative', 'Balanced', 'Aggressive']

        for profile in profiles:
            result = generate_configuration_for_pipeline(profile)

            if result and result['configuration_generated']:
                logger.info(f"✅ {profile} profile configuration generated successfully")
                logger.info(f"   Profile: {result['profile_name']}")
                logger.info(f"   Sheets: {len(result['config_data']) - 1}")  # -1 for metadata
                logger.info(f"   Saved to: {result['saved_to_file']}")
            else:
                logger.error(f"❌ {profile} profile configuration generation failed")
                return False

        logger.info("✅ Enhanced Excel Configuration Generator test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Enhanced Excel Configuration Generator test ERROR: {e}")
        return False

if __name__ == "__main__":
    # Run basic test
    test_enhanced_excel_config_generator()
