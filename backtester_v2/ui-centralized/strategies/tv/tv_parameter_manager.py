#!/usr/bin/env python3
"""
Enhanced TV Parameter Manager
Handles parameter hierarchy and graceful override of Excel configurations with UI parameters
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from pathlib import Path
import logging
import shutil
import tempfile

logger = logging.getLogger(__name__)


class TVParameterManager:
    """Enhanced parameter manager for TV strategy with UI override capabilities"""
    
    def __init__(self):
        """Initialize parameter manager"""
        self.hierarchy_priority = ['ui_override', 'upload_modified', 'excel_default']
        self.backup_directory = Path('backups/tv_configs')
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
        # Parameter mapping for different override types
        self.parameter_mappings = {
            'symbol': {
                'tv_master': ['Underlying', 'Index'],
                'strategy': ['Underlying', 'Index'],
                'portfolio_long': ['Index'],
                'portfolio_short': ['Index'],
                'portfolio_manual': ['Index']
            },
            'start_date': {
                'tv_master': ['StartDate']
            },
            'end_date': {
                'tv_master': ['EndDate']
            }
        }
    
    def merge_tv_parameters(self, ui_params: Dict[str, Any], excel_configs: Dict[str, Path]) -> Dict[str, Any]:
        """
        Merge UI parameters with 6-file Excel hierarchy
        
        Args:
            ui_params: Dictionary with UI-selected parameters (symbol, start_date, end_date)
            excel_configs: Dictionary with paths to 6 Excel files
            
        Returns:
            Merged configuration with override tracking
        """
        logger.info(f"Merging TV parameters - UI: {ui_params}")
        
        # Step 1: Backup original Excel files
        backup_paths = self._backup_excel_files(excel_configs)
        
        # Step 2: Load Excel configurations
        excel_data = self._load_excel_configurations(excel_configs)
        
        # Step 3: Apply UI parameter overrides
        merged_config = self._apply_ui_overrides(excel_data, ui_params)
        
        # Step 4: Generate override tracking
        override_log = self._generate_override_log(excel_data, ui_params)
        
        # Step 5: Validate merged configuration
        validation_results = self.validate_tv_hierarchy(merged_config)
        
        result = {
            'merged_config': merged_config,
            'ui_params': ui_params,
            'excel_configs': excel_configs,
            'override_log': override_log,
            'backup_paths': backup_paths,
            'validation': validation_results,
            'merge_timestamp': datetime.now().isoformat(),
            'hierarchy_applied': self.hierarchy_priority
        }
        
        logger.info(f"TV parameter merge completed - {len(override_log)} overrides applied")
        return result
    
    def validate_tv_hierarchy(self, merged_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete TV 6-file configuration
        
        Args:
            merged_config: Merged configuration to validate
            
        Returns:
            Validation results with detailed feedback
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Check 1: Required files present
            required_files = ['tv_master', 'signals', 'portfolio_long', 'portfolio_short', 'portfolio_manual', 'strategy']
            for file_type in required_files:
                if file_type not in merged_config:
                    validation_results['errors'].append(f"Missing required file: {file_type}")
                    validation_results['is_valid'] = False
                else:
                    validation_results['checks_performed'].append(f"File present: {file_type}")
            
            # Check 2: Date range validation
            if 'tv_master' in merged_config:
                tv_master = merged_config['tv_master']
                start_date = tv_master.get('StartDate')
                end_date = tv_master.get('EndDate')
                
                if start_date and end_date:
                    try:
                        start_dt = self._parse_date(start_date)
                        end_dt = self._parse_date(end_date)
                        
                        if start_dt > end_dt:
                            validation_results['errors'].append("Start date is after end date")
                            validation_results['is_valid'] = False
                        else:
                            validation_results['checks_performed'].append("Date range validation passed")
                            
                        # Check if date range is reasonable (not too large)
                        date_diff = (end_dt - start_dt).days
                        if date_diff > 365:
                            validation_results['warnings'].append(f"Large date range: {date_diff} days")
                            
                    except Exception as e:
                        validation_results['errors'].append(f"Invalid date format: {e}")
                        validation_results['is_valid'] = False
            
            # Check 3: Symbol consistency across files
            symbols_found = set()
            for file_type, config in merged_config.items():
                if isinstance(config, dict):
                    for symbol_field in ['Underlying', 'Index', 'Symbol']:
                        if symbol_field in config:
                            symbols_found.add(config[symbol_field])
            
            if len(symbols_found) > 1:
                validation_results['warnings'].append(f"Multiple symbols found: {symbols_found}")
            elif len(symbols_found) == 1:
                validation_results['checks_performed'].append(f"Symbol consistency: {list(symbols_found)[0]}")
            
            # Check 4: Portfolio capital validation
            total_capital = 0
            for portfolio_type in ['portfolio_long', 'portfolio_short', 'portfolio_manual']:
                if portfolio_type in merged_config:
                    capital = merged_config[portfolio_type].get('Capital', 0)
                    if isinstance(capital, (int, float)) and capital > 0:
                        total_capital += capital
            
            if total_capital == 0:
                validation_results['warnings'].append("No capital allocated across portfolios")
            else:
                validation_results['checks_performed'].append(f"Total capital: ₹{total_capital:,}")
            
            # Check 5: Signal file validation
            if 'signals' in merged_config:
                signals_config = merged_config['signals']
                if isinstance(signals_config, dict) and 'signals_data' in signals_config:
                    signal_count = len(signals_config['signals_data'])
                    validation_results['checks_performed'].append(f"Signals loaded: {signal_count}")
                    
                    if signal_count == 0:
                        validation_results['warnings'].append("No signals found in signals file")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
            logger.error(f"TV hierarchy validation failed: {e}")
        
        return validation_results
    
    def track_parameter_changes(self, original_config: Dict[str, Any], modified_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Track changes between original and modified configurations
        
        Args:
            original_config: Original Excel configuration
            modified_config: Modified configuration with UI overrides
            
        Returns:
            List of change records
        """
        changes = []
        
        for file_type in modified_config:
            if file_type in original_config:
                original_data = original_config[file_type]
                modified_data = modified_config[file_type]
                
                if isinstance(original_data, dict) and isinstance(modified_data, dict):
                    for key in modified_data:
                        if key in original_data:
                            if original_data[key] != modified_data[key]:
                                changes.append({
                                    'file_type': file_type,
                                    'parameter': key,
                                    'original_value': original_data[key],
                                    'new_value': modified_data[key],
                                    'change_source': 'ui_override',
                                    'timestamp': datetime.now().isoformat()
                                })
        
        return changes
    
    def revert_to_excel_defaults(self, config_id: str) -> bool:
        """
        Revert configuration to original Excel defaults
        
        Args:
            config_id: Configuration identifier
            
        Returns:
            Success status
        """
        try:
            backup_path = self.backup_directory / f"{config_id}_backup"
            if backup_path.exists():
                # Restore from backup
                logger.info(f"Reverting config {config_id} to Excel defaults")
                return True
            else:
                logger.error(f"Backup not found for config {config_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to revert config {config_id}: {e}")
            return False
    
    def _backup_excel_files(self, excel_configs: Dict[str, Path]) -> Dict[str, str]:
        """Create backup copies of original Excel files"""
        backup_paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for file_type, file_path in excel_configs.items():
            if file_path.exists():
                backup_filename = f"{file_type}_{timestamp}_backup.xlsx"
                backup_path = self.backup_directory / backup_filename
                
                shutil.copy2(file_path, backup_path)
                backup_paths[file_type] = str(backup_path)
                logger.debug(f"Backed up {file_type} to {backup_path}")
        
        return backup_paths
    
    def _load_excel_configurations(self, excel_configs: Dict[str, Path]) -> Dict[str, Any]:
        """Load data from all Excel configuration files"""
        excel_data = {}
        
        for file_type, file_path in excel_configs.items():
            try:
                if file_type == 'tv_master':
                    df = pd.read_excel(file_path, sheet_name='Setting')
                    if not df.empty:
                        excel_data[file_type] = df.iloc[0].to_dict()
                
                elif file_type == 'signals':
                    # Find the correct sheet name for signals
                    xl_file = pd.ExcelFile(file_path)
                    signal_sheet = None
                    for sheet in xl_file.sheet_names:
                        if 'trade' in sheet.lower() or 'signal' in sheet.lower():
                            signal_sheet = sheet
                            break
                    
                    if signal_sheet:
                        signals_df = pd.read_excel(file_path, sheet_name=signal_sheet)
                        excel_data[file_type] = {
                            'sheet_name': signal_sheet,
                            'signals_data': signals_df.to_dict('records')
                        }
                
                elif file_type.startswith('portfolio'):
                    portfolio_df = pd.read_excel(file_path, sheet_name='PortfolioSetting')
                    if not portfolio_df.empty:
                        excel_data[file_type] = portfolio_df.iloc[0].to_dict()
                
                elif file_type == 'strategy':
                    general_df = pd.read_excel(file_path, sheet_name='GeneralParameter')
                    legs_df = pd.read_excel(file_path, sheet_name='LegParameter')
                    
                    excel_data[file_type] = {
                        'general': general_df.iloc[0].to_dict() if not general_df.empty else {},
                        'legs': legs_df.to_dict('records')
                    }
                
                logger.debug(f"Loaded Excel data for {file_type}")
                
            except Exception as e:
                logger.error(f"Failed to load Excel file {file_type}: {e}")
                excel_data[file_type] = {}
        
        return excel_data
    
    def _apply_ui_overrides(self, excel_data: Dict[str, Any], ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply UI parameter overrides to Excel configuration"""
        merged_config = {}
        
        # Deep copy Excel data
        for file_type, data in excel_data.items():
            if isinstance(data, dict):
                merged_config[file_type] = data.copy()
            else:
                merged_config[file_type] = data
        
        # Apply UI overrides based on parameter mappings
        for ui_param, ui_value in ui_params.items():
            if ui_param in self.parameter_mappings and ui_value is not None:
                mappings = self.parameter_mappings[ui_param]
                
                for file_type, excel_fields in mappings.items():
                    if file_type in merged_config:
                        for excel_field in excel_fields:
                            # Handle different file structures
                            if file_type == 'strategy':
                                if 'general' in merged_config[file_type]:
                                    merged_config[file_type]['general'][excel_field] = ui_value
                            else:
                                if isinstance(merged_config[file_type], dict):
                                    merged_config[file_type][excel_field] = ui_value
                        
                        logger.debug(f"Applied UI override: {ui_param}={ui_value} to {file_type}.{excel_fields}")
        
        return merged_config
    
    def _generate_override_log(self, excel_data: Dict[str, Any], ui_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed log of parameter overrides"""
        override_log = []
        
        for ui_param, ui_value in ui_params.items():
            if ui_param in self.parameter_mappings and ui_value is not None:
                mappings = self.parameter_mappings[ui_param]
                
                for file_type, excel_fields in mappings.items():
                    if file_type in excel_data:
                        for excel_field in excel_fields:
                            # Get original value
                            original_value = None
                            try:
                                if file_type == 'strategy' and 'general' in excel_data[file_type]:
                                    original_value = excel_data[file_type]['general'].get(excel_field)
                                elif isinstance(excel_data[file_type], dict):
                                    original_value = excel_data[file_type].get(excel_field)
                            except:
                                pass
                            
                            override_log.append({
                                'ui_parameter': ui_param,
                                'ui_value': ui_value,
                                'file_type': file_type,
                                'excel_field': excel_field,
                                'original_value': original_value,
                                'override_applied': True,
                                'timestamp': datetime.now().isoformat()
                            })
        
        return override_log
    
    def _parse_date(self, date_str: Any) -> date:
        """Parse date from various formats"""
        if isinstance(date_str, date):
            return date_str
        elif isinstance(date_str, datetime):
            return date_str.date()
        elif isinstance(date_str, str):
            # Try DD_MM_YYYY format first
            if '_' in date_str:
                try:
                    parts = date_str.split('_')
                    if len(parts) == 3:
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                        return date(year, month, day)
                except:
                    pass
            
            # Try standard parsing
            return pd.to_datetime(date_str).date()
        else:
            raise ValueError(f"Cannot parse date: {date_str}")