"""
Adaptive Regime Configuration Manager

This module manages Excel-based configuration for the adaptive market regime
formation system, supporting 8, 12, or 18 regime configurations with
intelligent parameter validation and template generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import json
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeCount(Enum):
    """Supported regime count configurations"""
    EIGHT = 8
    TWELVE = 12
    EIGHTEEN = 18


@dataclass
class AdaptiveRegimeConfig:
    """Configuration data class for adaptive regime parameters"""
    regime_count: int
    historical_lookback_days: int
    intraday_window: str
    transition_sensitivity: float
    adaptive_learning_rate: float
    min_regime_duration: int
    noise_filter_window: int
    enable_asl: bool
    enable_hysteresis: bool
    confidence_threshold: float
    
    # Additional adaptive parameters
    clustering_algorithm: str = "kmeans"
    feature_selection_method: str = "auto"
    optimization_frequency: str = "daily"
    performance_lookback: int = 30
    
    # Profile settings
    profile_name: str = "balanced"
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.regime_count not in [8, 12, 18]:
            raise ValueError(f"Invalid regime_count: {self.regime_count}. Must be 8, 12, or 18")
        
        if not (30 <= self.historical_lookback_days <= 180):
            raise ValueError(f"historical_lookback_days must be between 30 and 180")
        
        if self.intraday_window not in ["3min", "5min", "10min", "15min"]:
            raise ValueError(f"Invalid intraday_window: {self.intraday_window}")
        
        if not (0.0 <= self.transition_sensitivity <= 1.0):
            raise ValueError("transition_sensitivity must be between 0.0 and 1.0")
        
        if not (0.01 <= self.adaptive_learning_rate <= 0.2):
            raise ValueError("adaptive_learning_rate must be between 0.01 and 0.2")
        
        if not (5 <= self.min_regime_duration <= 60):
            raise ValueError("min_regime_duration must be between 5 and 60 minutes")
        
        if not (3 <= self.noise_filter_window <= 10):
            raise ValueError("noise_filter_window must be between 3 and 10")
        
        if not (0.5 <= self.confidence_threshold <= 0.9):
            raise ValueError("confidence_threshold must be between 0.5 and 0.9")
        
        return True


class AdaptiveRegimeConfigManager:
    """
    Manages Excel-based configuration for adaptive regime formation system
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        'regime_count': 12,
        'historical_lookback_days': 90,
        'intraday_window': '5min',
        'transition_sensitivity': 0.7,
        'adaptive_learning_rate': 0.05,
        'min_regime_duration': 15,
        'noise_filter_window': 5,
        'enable_asl': True,
        'enable_hysteresis': True,
        'confidence_threshold': 0.65,
        'clustering_algorithm': 'kmeans',
        'feature_selection_method': 'auto',
        'optimization_frequency': 'daily',
        'performance_lookback': 30,
        'profile_name': 'balanced'
    }
    
    # Profile presets
    PROFILE_PRESETS = {
        'conservative': {
            'transition_sensitivity': 0.8,
            'adaptive_learning_rate': 0.02,
            'min_regime_duration': 30,
            'noise_filter_window': 7,
            'confidence_threshold': 0.75
        },
        'balanced': DEFAULT_CONFIG.copy(),
        'aggressive': {
            'transition_sensitivity': 0.6,
            'adaptive_learning_rate': 0.1,
            'min_regime_duration': 10,
            'noise_filter_window': 3,
            'confidence_threshold': 0.55
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to Excel configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[AdaptiveRegimeConfig] = None
        self._load_defaults()
        
        if self.config_path and self.config_path.exists():
            try:
                self.load_configuration()
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                logger.info("Using default configuration")
    
    def _load_defaults(self):
        """Load default configuration"""
        self.config = AdaptiveRegimeConfig(**self.DEFAULT_CONFIG)
    
    def load_configuration(self) -> AdaptiveRegimeConfig:
        """
        Load configuration from Excel file
        
        Returns:
            AdaptiveRegimeConfig object with loaded parameters
        """
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            # Read the Adaptive Regime Formation sheet
            df = pd.read_excel(
                self.config_path, 
                sheet_name='Adaptive Regime Formation',
                engine='openpyxl'
            )
            
            # Parse configuration parameters
            config_dict = {}
            for _, row in df.iterrows():
                param_name = row['Parameter']
                value = row['Value']
                
                # Convert parameter names to lowercase with underscores
                param_key = param_name.replace(' ', '_').lower()
                
                # Type conversion based on parameter
                if param_key in ['regime_count', 'historical_lookback_days', 
                               'min_regime_duration', 'noise_filter_window',
                               'performance_lookback']:
                    value = int(value)
                elif param_key in ['transition_sensitivity', 'adaptive_learning_rate',
                                 'confidence_threshold']:
                    value = float(value)
                elif param_key in ['enable_asl', 'enable_hysteresis']:
                    value = str(value).upper() == 'YES'
                
                config_dict[param_key] = value
            
            # Apply profile presets if specified
            profile_name = config_dict.get('profile_name', 'balanced')
            if profile_name in self.PROFILE_PRESETS:
                profile_config = self.PROFILE_PRESETS[profile_name].copy()
                profile_config.update(config_dict)
                config_dict = profile_config
            
            # Create configuration object
            self.config = AdaptiveRegimeConfig(**config_dict)
            self.config.validate()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            logger.info(f"Regime count: {self.config.regime_count}")
            logger.info(f"Profile: {self.config.profile_name}")
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            True if all parameters are valid
        """
        try:
            config = AdaptiveRegimeConfig(**parameters)
            return config.validate()
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def generate_template(self, output_path: Optional[str] = None) -> Path:
        """
        Generate Excel configuration template
        
        Args:
            output_path: Path for output file (optional)
            
        Returns:
            Path to generated template file
        """
        if not output_path:
            output_path = Path("adaptive_regime_config_template.xlsx")
        else:
            output_path = Path(output_path)
        
        # Create template data
        template_data = []
        
        # Basic parameters
        template_data.extend([
            ['Regime_Count', 12, 'Number of regimes (8/12/18)', 'int', 8, 18],
            ['Historical_Lookback_Days', 90, 'Days of historical data for analysis', 'int', 30, 180],
            ['Intraday_Window', '5min', 'Primary timeframe for intraday analysis', 'str', '3min', '15min'],
            ['Transition_Sensitivity', 0.7, 'Sensitivity to regime changes (0-1)', 'float', 0.0, 1.0],
            ['Adaptive_Learning_Rate', 0.05, 'ASL weight update rate', 'float', 0.01, 0.2],
            ['Min_Regime_Duration', 15, 'Minimum regime duration in minutes', 'int', 5, 60],
            ['Noise_Filter_Window', 5, 'Window size for noise filtering', 'int', 3, 10],
            ['Enable_ASL', 'YES', 'Enable Adaptive Scoring Layer', 'bool', 'YES', 'NO'],
            ['Enable_Hysteresis', 'YES', 'Enable transition hysteresis', 'bool', 'YES', 'NO'],
            ['Confidence_Threshold', 0.65, 'Minimum confidence for regime classification', 'float', 0.5, 0.9],
        ])
        
        # Advanced parameters
        template_data.extend([
            ['Clustering_Algorithm', 'kmeans', 'Algorithm for regime clustering', 'str', 'kmeans', 'hierarchical'],
            ['Feature_Selection_Method', 'auto', 'Method for feature selection', 'str', 'auto', 'manual'],
            ['Optimization_Frequency', 'daily', 'Frequency of parameter optimization', 'str', 'hourly', 'weekly'],
            ['Performance_Lookback', 30, 'Days to look back for performance metrics', 'int', 7, 90],
            ['Profile_Name', 'balanced', 'Configuration profile (conservative/balanced/aggressive)', 'str', 'conservative', 'aggressive'],
        ])
        
        # Component weights
        template_data.extend([
            ['Weight_Triple_Straddle', 0.25, 'Weight for Triple Straddle component', 'float', 0.0, 1.0],
            ['Weight_Greek_Sentiment', 0.20, 'Weight for Greek Sentiment component', 'float', 0.0, 1.0],
            ['Weight_OI_Analysis', 0.20, 'Weight for OI Analysis component', 'float', 0.0, 1.0],
            ['Weight_Technical', 0.15, 'Weight for Technical Indicators', 'float', 0.0, 1.0],
            ['Weight_ML_Ensemble', 0.20, 'Weight for ML Ensemble predictions', 'float', 0.0, 1.0],
        ])
        
        # Regime-specific parameters
        template_data.extend([
            ['Volatility_Threshold_Low', 0.25, 'Threshold for low volatility classification', 'float', 0.0, 0.5],
            ['Volatility_Threshold_High', 0.75, 'Threshold for high volatility classification', 'float', 0.5, 1.0],
            ['Trend_Threshold_Bullish', 0.3, 'Threshold for bullish trend classification', 'float', 0.0, 1.0],
            ['Trend_Threshold_Bearish', -0.3, 'Threshold for bearish trend classification', 'float', -1.0, 0.0],
        ])
        
        # Create DataFrame
        df = pd.DataFrame(template_data, columns=[
            'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue'
        ])
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write main configuration sheet
            df.to_excel(writer, sheet_name='Adaptive Regime Formation', index=False)
            
            # Add profile examples sheet
            profile_data = []
            for profile_name, profile_config in self.PROFILE_PRESETS.items():
                for param, value in profile_config.items():
                    profile_data.append([profile_name, param, value])
            
            profile_df = pd.DataFrame(profile_data, columns=['Profile', 'Parameter', 'Value'])
            profile_df.to_excel(writer, sheet_name='Profile Examples', index=False)
            
            # Add regime mapping sheet for different counts
            regime_mapping_data = []
            
            # 8-regime mapping
            regime_mapping_data.extend([
                [8, 0, 'Strong Bullish', 'High momentum bullish trend'],
                [8, 1, 'Moderate Bullish', 'Steady bullish movement'],
                [8, 2, 'Weak Bullish', 'Mild bullish bias'],
                [8, 3, 'Neutral Range', 'Sideways range-bound'],
                [8, 4, 'Volatile Neutral', 'High volatility sideways'],
                [8, 5, 'Weak Bearish', 'Mild bearish bias'],
                [8, 6, 'Moderate Bearish', 'Steady bearish movement'],
                [8, 7, 'Strong Bearish', 'High momentum bearish trend'],
            ])
            
            # 12-regime mapping
            regime_mapping_data.extend([
                [12, 0, 'Low Vol Bullish Trending', 'Stable bullish trend'],
                [12, 1, 'Low Vol Bullish Range', 'Bullish bias in range'],
                [12, 2, 'Low Vol Neutral Trending', 'Stable sideways trend'],
                [12, 3, 'Low Vol Neutral Range', 'Quiet consolidation'],
                [12, 4, 'Med Vol Bullish Trending', 'Normal bullish trend'],
                [12, 5, 'Med Vol Bullish Range', 'Bullish with chop'],
                [12, 6, 'Med Vol Neutral Trending', 'Active sideways'],
                [12, 7, 'Med Vol Neutral Range', 'Normal range trading'],
                [12, 8, 'High Vol Bullish Trending', 'Volatile bullish'],
                [12, 9, 'High Vol Bullish Range', 'Chaotic bullish'],
                [12, 10, 'High Vol Bearish Trending', 'Volatile bearish'],
                [12, 11, 'High Vol Bearish Range', 'Chaotic bearish'],
            ])
            
            # 18-regime mapping (simplified for space)
            for i in range(18):
                vol_level = ['Low', 'Normal', 'High'][i // 6]
                trend = ['Strong Bullish', 'Mild Bullish', 'Neutral', 'Sideways', 'Mild Bearish', 'Strong Bearish'][i % 6]
                regime_mapping_data.append([18, i, f"{vol_level} Vol {trend}", f"{vol_level} volatility with {trend.lower()} characteristics"])
            
            regime_df = pd.DataFrame(regime_mapping_data, columns=['Regime_Count', 'Regime_ID', 'Regime_Name', 'Description'])
            regime_df.to_excel(writer, sheet_name='Regime Mappings', index=False)
            
            # Format the Excel file
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # Set column widths
                worksheet.column_dimensions['A'].width = 30
                worksheet.column_dimensions['B'].width = 15
                worksheet.column_dimensions['C'].width = 50
                
                if sheet_name == 'Adaptive Regime Formation':
                    worksheet.column_dimensions['D'].width = 12
                    worksheet.column_dimensions['E'].width = 12
                    worksheet.column_dimensions['F'].width = 12
        
        logger.info(f"Configuration template generated at {output_path}")
        return output_path
    
    def reload_configuration(self) -> AdaptiveRegimeConfig:
        """
        Reload configuration from file
        
        Returns:
            Updated configuration object
        """
        if not self.config_path:
            logger.warning("No configuration path set, using defaults")
            return self.config
        
        return self.load_configuration()
    
    def get_regime_specific_config(self, regime_count: int) -> Dict[str, Any]:
        """
        Get configuration specific to regime count
        
        Args:
            regime_count: Number of regimes (8, 12, or 18)
            
        Returns:
            Dictionary of regime-specific parameters
        """
        if regime_count not in [8, 12, 18]:
            raise ValueError(f"Invalid regime count: {regime_count}")
        
        # Base configuration
        config = self.config.__dict__.copy() if self.config else self.DEFAULT_CONFIG.copy()
        
        # Regime-specific adjustments
        if regime_count == 8:
            config.update({
                'min_regime_duration': 20,  # Longer duration for fewer regimes
                'transition_sensitivity': 0.75,  # Higher threshold
                'noise_filter_window': 7  # More filtering
            })
        elif regime_count == 18:
            config.update({
                'min_regime_duration': 10,  # Shorter duration for more regimes
                'transition_sensitivity': 0.65,  # Lower threshold
                'noise_filter_window': 3  # Less filtering
            })
        
        return config
    
    def export_config_summary(self) -> Dict[str, Any]:
        """
        Export configuration summary as dictionary
        
        Returns:
            Dictionary containing configuration summary
        """
        if not self.config:
            self._load_defaults()
        
        return {
            'regime_count': self.config.regime_count,
            'profile': self.config.profile_name,
            'lookback_days': self.config.historical_lookback_days,
            'intraday_window': self.config.intraday_window,
            'asl_enabled': self.config.enable_asl,
            'hysteresis_enabled': self.config.enable_hysteresis,
            'optimization_frequency': self.config.optimization_frequency,
            'last_updated': datetime.now().isoformat()
        }
    
    def save_configuration(self, config: AdaptiveRegimeConfig, output_path: Optional[str] = None):
        """
        Save configuration to Excel file
        
        Args:
            config: Configuration object to save
            output_path: Output file path (uses current path if not specified)
        """
        output_path = Path(output_path) if output_path else self.config_path
        
        if not output_path:
            raise ValueError("No output path specified")
        
        # Convert config to DataFrame
        data = []
        for key, value in config.__dict__.items():
            param_name = key.replace('_', ' ').title()
            if isinstance(value, bool):
                value = 'YES' if value else 'NO'
            
            data.append([param_name, value, '', '', '', ''])
        
        df = pd.DataFrame(data, columns=[
            'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue'
        ])
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Adaptive Regime Formation', index=False)
        
        logger.info(f"Configuration saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate template
    manager = AdaptiveRegimeConfigManager()
    template_path = manager.generate_template("adaptive_regime_template.xlsx")
    print(f"Template generated at: {template_path}")
    
    # Load configuration
    # manager = AdaptiveRegimeConfigManager("config.xlsx")
    # config = manager.load_configuration()
    # print(f"Loaded config: {config}")
    
    # Get regime-specific config
    config_8 = manager.get_regime_specific_config(8)
    print(f"\n8-regime config adjustments: {config_8}")