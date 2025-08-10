#!/usr/bin/env python3
"""
Adaptive Configuration Parser for Excel-based Trading Mode Configuration
Reads and processes the Adaptive Timeframe Configuration sheet from Excel

This module parses the Excel configuration to extract trading mode settings
and timeframe configurations for the adaptive market regime system.

Author: The Augster
Date: 2025-01-10
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class AdaptiveConfigParser:
    """
    Parses adaptive timeframe configuration from Excel template
    """
    
    def __init__(self, excel_path: str = None):
        """
        Initialize configuration parser
        
        Args:
            excel_path: Path to Excel configuration file
        """
        self.excel_path = excel_path
        self.config_data = {}
        self.mode_configs = {}
        self.timeframe_configs = {}
        
    def parse_adaptive_config(self, excel_path: str = None) -> Dict[str, Any]:
        """
        Parse the Adaptive Timeframe Configuration sheet
        
        Args:
            excel_path: Path to Excel file (uses instance path if not provided)
            
        Returns:
            Parsed configuration dictionary
        """
        try:
            path = excel_path or self.excel_path
            if not path:
                raise ValueError("No Excel path provided")
            
            # Read Excel file
            df_dict = pd.read_excel(path, sheet_name=None, engine='openpyxl')
            
            # Find adaptive timeframe sheet
            adaptive_sheet = None
            for sheet_name in df_dict.keys():
                if 'adaptive' in sheet_name.lower() and 'timeframe' in sheet_name.lower():
                    adaptive_sheet = df_dict[sheet_name]
                    break
            
            if adaptive_sheet is None:
                logger.warning("No Adaptive Timeframe Configuration sheet found")
                return {}
            
            # Parse different sections
            self._parse_mode_selection(adaptive_sheet)
            self._parse_timeframe_configuration(adaptive_sheet)
            self._parse_custom_configuration(adaptive_sheet)
            self._parse_performance_optimization(adaptive_sheet)
            
            # Compile final configuration
            config = {
                'current_mode': self._extract_current_mode(adaptive_sheet),
                'mode_configs': self.mode_configs,
                'timeframe_configs': self.timeframe_configs,
                'custom_config': self.config_data.get('custom', {}),
                'performance_data': self.config_data.get('performance', {})
            }
            
            logger.info("Successfully parsed adaptive configuration")
            return config
            
        except Exception as e:
            logger.error(f"Error parsing adaptive configuration: {e}")
            return {}
    
    def _parse_mode_selection(self, df: pd.DataFrame):
        """Parse trading mode selection section"""
        try:
            # Find mode selection section
            mode_start_idx = None
            for idx, row in df.iterrows():
                if any('Trading Mode Selection' in str(cell) for cell in row if pd.notna(cell)):
                    mode_start_idx = idx
                    break
            
            if mode_start_idx is None:
                return
            
            # Parse mode data (typically 5 rows after header)
            for i in range(mode_start_idx + 2, min(mode_start_idx + 7, len(df))):
                row = df.iloc[i]
                
                # Extract mode data
                if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                    mode_name = str(row.iloc[0]).lower()
                    
                    self.mode_configs[mode_name] = {
                        'description': str(row.iloc[1]) if pd.notna(row.iloc[1]) else '',
                        'risk_multiplier': float(row.iloc[2]) if pd.notna(row.iloc[2]) else 1.0,
                        'transition_threshold': float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0.75,
                        'stability_window': int(row.iloc[4]) if pd.notna(row.iloc[4]) else 15
                    }
            
            logger.debug(f"Parsed {len(self.mode_configs)} mode configurations")
            
        except Exception as e:
            logger.error(f"Error parsing mode selection: {e}")
    
    def _parse_timeframe_configuration(self, df: pd.DataFrame):
        """Parse timeframe configuration by mode"""
        try:
            # Find timeframe configuration section
            tf_start_idx = None
            for idx, row in df.iterrows():
                if any('Timeframe Configuration by Trading Mode' in str(cell) for cell in row if pd.notna(cell)):
                    tf_start_idx = idx
                    break
            
            if tf_start_idx is None:
                return
            
            # Parse timeframe data
            # Headers are typically 2 rows after section title
            header_idx = tf_start_idx + 2
            
            # Initialize mode columns mapping
            mode_columns = {
                'intraday': {'enable': None, 'weight': None},
                'positional': {'enable': None, 'weight': None},
                'hybrid': {'enable': None, 'weight': None},
                'custom': {'enable': None, 'weight': None}
            }
            
            # Map column indices for each mode
            header_row = df.iloc[header_idx]
            for col_idx, header in enumerate(header_row):
                if pd.notna(header):
                    header_str = str(header).lower()
                    for mode in mode_columns:
                        if mode in header_str:
                            if 'enable' in header_str:
                                mode_columns[mode]['enable'] = col_idx
                            elif 'weight' in header_str:
                                mode_columns[mode]['weight'] = col_idx
            
            # Parse timeframe rows
            for i in range(header_idx + 1, min(header_idx + 10, len(df))):
                row = df.iloc[i]
                
                if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                    timeframe = str(row.iloc[0]).replace(' ', '').lower()
                    
                    # Initialize timeframe config
                    if timeframe not in self.timeframe_configs:
                        self.timeframe_configs[timeframe] = {}
                    
                    # Parse each mode's configuration
                    for mode, cols in mode_columns.items():
                        if cols['enable'] is not None and cols['weight'] is not None:
                            enabled = str(row.iloc[cols['enable']]).upper() == 'YES' if pd.notna(row.iloc[cols['enable']]) else False
                            weight = float(row.iloc[cols['weight']]) if pd.notna(row.iloc[cols['weight']]) else 0.0
                            
                            self.timeframe_configs[timeframe][mode] = {
                                'enabled': enabled,
                                'weight': weight
                            }
            
            logger.debug(f"Parsed timeframe configurations for {len(self.timeframe_configs)} timeframes")
            
        except Exception as e:
            logger.error(f"Error parsing timeframe configuration: {e}")
    
    def _parse_custom_configuration(self, df: pd.DataFrame):
        """Parse custom mode configuration"""
        try:
            # Find custom configuration section
            custom_start_idx = None
            for idx, row in df.iterrows():
                if any('Custom Mode Configuration' in str(cell) for cell in row if pd.notna(cell)):
                    custom_start_idx = idx
                    break
            
            if custom_start_idx is None:
                return
            
            custom_config = {}
            
            # Parse custom parameters (typically 6 rows)
            for i in range(custom_start_idx + 2, min(custom_start_idx + 8, len(df))):
                row = df.iloc[i]
                
                if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                    param_name = str(row.iloc[0]).strip()
                    param_value = row.iloc[1]
                    
                    # Map parameter names
                    if 'Mode Name' in param_name:
                        custom_config['name'] = str(param_value)
                    elif 'Description' in param_name:
                        custom_config['description'] = str(param_value)
                    elif 'Risk Multiplier' in param_name:
                        custom_config['risk_multiplier'] = float(param_value)
                    elif 'Transition Threshold' in param_name:
                        custom_config['transition_threshold'] = float(param_value)
                    elif 'Stability Window' in param_name:
                        custom_config['regime_stability_window'] = int(param_value)
            
            self.config_data['custom'] = custom_config
            logger.debug("Parsed custom configuration")
            
        except Exception as e:
            logger.error(f"Error parsing custom configuration: {e}")
    
    def _parse_performance_optimization(self, df: pd.DataFrame):
        """Parse performance-based optimization data"""
        try:
            # Find performance section
            perf_start_idx = None
            for idx, row in df.iterrows():
                if any('Performance-Based Weight Optimization' in str(cell) for cell in row if pd.notna(cell)):
                    perf_start_idx = idx
                    break
            
            if perf_start_idx is None:
                return
            
            performance_data = {}
            
            # Parse performance rows
            for i in range(perf_start_idx + 2, min(perf_start_idx + 10, len(df))):
                row = df.iloc[i]
                
                if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                    timeframe = str(row.iloc[0]).replace(' ', '').lower()
                    
                    performance_data[timeframe] = {
                        'historical_accuracy': float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0.5,
                        'current_weight': float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0.0,
                        'optimized_weight': float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0.0,
                        'status': str(row.iloc[4]) if pd.notna(row.iloc[4]) else 'Unknown'
                    }
            
            self.config_data['performance'] = performance_data
            logger.debug(f"Parsed performance data for {len(performance_data)} timeframes")
            
        except Exception as e:
            logger.error(f"Error parsing performance optimization: {e}")
    
    def _extract_current_mode(self, df: pd.DataFrame) -> str:
        """Extract current active mode from sheet"""
        try:
            # Look for "Current Active Mode" cell
            for idx, row in df.iterrows():
                for col_idx, cell in enumerate(row):
                    if pd.notna(cell) and 'Current Active Mode' in str(cell):
                        # Next cell should contain the mode
                        if col_idx + 1 < len(row):
                            mode = str(row.iloc[col_idx + 1]).lower().strip()
                            return mode
            
            return 'hybrid'  # Default
            
        except Exception as e:
            logger.error(f"Error extracting current mode: {e}")
            return 'hybrid'
    
    def generate_adaptive_config(self, mode: str) -> Dict[str, Any]:
        """
        Generate adaptive configuration for specific mode
        
        Args:
            mode: Trading mode to generate config for
            
        Returns:
            Configuration dictionary for AdaptiveTimeframeManager
        """
        if not self.timeframe_configs:
            logger.warning("No timeframe configurations loaded")
            return {}
        
        # Build timeframes configuration
        timeframes = {}
        for tf_name, tf_modes in self.timeframe_configs.items():
            if mode in tf_modes:
                tf_config = tf_modes[mode]
                timeframes[tf_name] = {
                    'enabled': tf_config['enabled'],
                    'weight': tf_config['weight'],
                    'min_data_points': self._get_min_data_points(tf_name),
                    'description': self._get_timeframe_description(tf_name)
                }
        
        # Get mode configuration
        mode_config = self.mode_configs.get(mode, {})
        
        # Build final configuration
        config = {
            'mode': mode,
            'description': mode_config.get('description', ''),
            'timeframes': timeframes,
            'risk_multiplier': mode_config.get('risk_multiplier', 1.0),
            'transition_threshold': mode_config.get('transition_threshold', 0.75),
            'regime_stability_window': mode_config.get('stability_window', 15)
        }
        
        return config
    
    def _get_min_data_points(self, timeframe: str) -> int:
        """Get minimum data points for timeframe"""
        mapping = {
            '3minutes': 20,
            '3min': 20,
            '5minutes': 12,
            '5min': 12,
            '10minutes': 6,
            '10min': 6,
            '15minutes': 4,
            '15min': 4,
            '30minutes': 2,
            '30min': 2,
            '1hour': 1,
            '1hr': 1,
            '4hours': 1,
            '4hr': 1
        }
        return mapping.get(timeframe, 1)
    
    def _get_timeframe_description(self, timeframe: str) -> str:
        """Get description for timeframe"""
        mapping = {
            '3minutes': 'Ultra short-term scalping',
            '3min': 'Ultra short-term scalping',
            '5minutes': 'Short-term entry signals',
            '5min': 'Short-term entry signals',
            '10minutes': 'Trend confirmation',
            '10min': 'Trend confirmation',
            '15minutes': 'Standard intraday',
            '15min': 'Standard intraday',
            '30minutes': 'Medium-term trend',
            '30min': 'Medium-term trend',
            '1hour': 'Positional trend',
            '1hr': 'Positional trend',
            '4hours': 'Multi-day context',
            '4hr': 'Multi-day context'
        }
        return mapping.get(timeframe, '')

# Example usage
if __name__ == "__main__":
    # Test with the template file
    parser = AdaptiveConfigParser()
    
    # Parse configuration
    config = parser.parse_adaptive_config(
        "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/market_regime_config_template.xlsx"
    )
    
    print("Parsed Adaptive Configuration:")
    print(json.dumps(config, indent=2))
    
    # Generate config for specific mode
    print("\n" + "="*60)
    print("Generated config for intraday mode:")
    intraday_config = parser.generate_adaptive_config('intraday')
    print(json.dumps(intraday_config, indent=2))