#!/usr/bin/env python3
"""
Enhanced TV YAML Converter
Extends the base TVExcelToYAMLConverter with UI parameter injection and automatic conversion
"""

import yaml
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date, time
from pathlib import Path
import logging

# Import the base converter
from excel_to_yaml_converter import TVExcelToYAMLConverter

logger = logging.getLogger(__name__)


class EnhancedTVYAMLConverter(TVExcelToYAMLConverter):
    """Enhanced YAML converter with UI parameter injection and automatic conversion"""
    
    def __init__(self):
        """Initialize enhanced converter"""
        super().__init__()
        self.conversion_metadata = {}
        
    def auto_convert_with_ui_override(self, excel_files: Dict[str, Path], ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically convert Excel to YAML with UI parameter injection
        
        Args:
            excel_files: Dictionary of Excel file paths (6-file hierarchy)
            ui_params: UI-selected parameters (symbol, start_date, end_date)
            
        Returns:
            Complete YAML configuration with UI overrides applied
        """
        logger.info(f"Starting auto-conversion with UI overrides: {ui_params}")
        
        try:
            # Step 1: Convert base Excel configuration to YAML
            base_yaml = self.convert_complete_hierarchy_to_yaml(excel_files)
            
            # Step 2: Apply UI parameter injections
            enhanced_yaml = self._inject_ui_parameters(base_yaml, ui_params)
            
            # Step 3: Add conversion metadata
            enhanced_yaml = self._add_conversion_metadata(enhanced_yaml, ui_params, excel_files)
            
            # Step 4: Validate enhanced YAML structure
            validation_results = self._validate_enhanced_yaml(enhanced_yaml)
            if not validation_results['is_valid']:
                logger.warning(f"YAML validation warnings: {validation_results['warnings']}")
            
            # Step 5: Generate parameter difference report
            diff_report = self._generate_parameter_diff_report(base_yaml, enhanced_yaml, ui_params)
            enhanced_yaml['parameter_diff_report'] = diff_report
            
            logger.info("Auto-conversion with UI overrides completed successfully")
            return enhanced_yaml
            
        except Exception as e:
            logger.error(f"Auto-conversion failed: {e}")
            raise
    
    def preserve_excel_integrity(self, original_files: Dict[str, Path]) -> Dict[str, str]:
        """
        Preserve original Excel file integrity with backup and version management
        
        Args:
            original_files: Dictionary of original Excel file paths
            
        Returns:
            Dictionary of backup file paths
        """
        backup_paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path('backups/excel_originals')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file_type, file_path in original_files.items():
            if file_path.exists():
                backup_filename = f"{file_type}_{timestamp}_original.xlsx"
                backup_path = backup_dir / backup_filename
                
                # Create backup
                import shutil
                shutil.copy2(file_path, backup_path)
                backup_paths[file_type] = str(backup_path)
                
                logger.debug(f"Preserved Excel integrity: {file_type} → {backup_path}")
        
        return backup_paths
    
    def generate_diff_report(self, original_yaml: Dict[str, Any], modified_yaml: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed difference report between original and modified YAML
        
        Args:
            original_yaml: Original YAML configuration
            modified_yaml: Modified YAML with UI overrides
            
        Returns:
            Detailed difference report
        """
        diff_report = {
            'differences_found': 0,
            'parameter_changes': [],
            'new_parameters': [],
            'removed_parameters': [],
            'structural_changes': [],
            'generation_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Compare main configuration sections
            self._compare_yaml_sections(original_yaml, modified_yaml, diff_report, path_prefix='')
            
            diff_report['differences_found'] = (
                len(diff_report['parameter_changes']) + 
                len(diff_report['new_parameters']) + 
                len(diff_report['removed_parameters'])
            )
            
            logger.info(f"Diff report generated: {diff_report['differences_found']} differences found")
            
        except Exception as e:
            logger.error(f"Failed to generate diff report: {e}")
            diff_report['error'] = str(e)
        
        return diff_report
    
    def export_enhanced_yaml(self, yaml_config: Dict[str, Any], output_path: Path) -> Path:
        """
        Export enhanced YAML configuration to file
        
        Args:
            yaml_config: Complete YAML configuration
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"Enhanced YAML exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export enhanced YAML: {e}")
            raise
    
    def _inject_ui_parameters(self, base_yaml: Dict[str, Any], ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Inject UI parameters into YAML configuration"""
        enhanced_yaml = self._deep_copy_yaml(base_yaml)
        
        # Track UI parameter injections
        ui_injections = []
        
        try:
            # Navigate to TV configuration section
            if 'tv_complete_configuration' in enhanced_yaml:
                tv_config = enhanced_yaml['tv_complete_configuration']
                
                # Inject symbol parameter
                if ui_params.get('symbol'):
                    symbol = ui_params['symbol']
                    
                    # Update in various sections
                    if 'tv_master' in tv_config:
                        if 'master_settings' in tv_config['tv_master']:
                            # Add symbol to master settings
                            tv_config['tv_master']['master_settings']['selected_symbol'] = symbol
                            ui_injections.append('tv_master.master_settings.selected_symbol')
                    
                    if 'tbs_strategy' in tv_config:
                        if 'general' in tv_config['tbs_strategy']:
                            tv_config['tbs_strategy']['general']['Underlying'] = symbol
                            tv_config['tbs_strategy']['general']['Index'] = symbol
                            ui_injections.append('tbs_strategy.general.Underlying')
                            ui_injections.append('tbs_strategy.general.Index')
                
                # Inject date parameters
                if ui_params.get('start_date'):
                    start_date = ui_params['start_date']
                    if 'tv_master' in tv_config and 'master_settings' in tv_config['tv_master']:
                        if 'date_range' in tv_config['tv_master']['master_settings']:
                            tv_config['tv_master']['master_settings']['date_range']['start'] = start_date
                            ui_injections.append('tv_master.master_settings.date_range.start')
                
                if ui_params.get('end_date'):
                    end_date = ui_params['end_date']
                    if 'tv_master' in tv_config and 'master_settings' in tv_config['tv_master']:
                        if 'date_range' in tv_config['tv_master']['master_settings']:
                            tv_config['tv_master']['master_settings']['date_range']['end'] = end_date
                            ui_injections.append('tv_master.master_settings.date_range.end')
            
            # Add UI injection tracking
            enhanced_yaml['ui_injection_log'] = {
                'injected_parameters': ui_injections,
                'ui_params_applied': ui_params,
                'injection_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"UI parameters injected: {len(ui_injections)} parameters modified")
            
        except Exception as e:
            logger.error(f"Failed to inject UI parameters: {e}")
            enhanced_yaml['ui_injection_error'] = str(e)
        
        return enhanced_yaml
    
    def _add_conversion_metadata(self, yaml_config: Dict[str, Any], ui_params: Dict[str, Any], excel_files: Dict[str, Path]) -> Dict[str, Any]:
        """Add comprehensive conversion metadata"""
        metadata = {
            'conversion_info': {
                'converter_version': '2.0.0_enhanced',
                'conversion_timestamp': datetime.now().isoformat(),
                'conversion_type': 'auto_with_ui_override',
                'excel_files_processed': len(excel_files),
                'ui_parameters_applied': len([v for v in ui_params.values() if v is not None])
            },
            'source_files': {
                file_type: {
                    'path': str(path),
                    'exists': path.exists(),
                    'size_bytes': path.stat().st_size if path.exists() else 0,
                    'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None
                }
                for file_type, path in excel_files.items()
            },
            'ui_override_summary': {
                'parameters_overridden': list(ui_params.keys()),
                'override_values': ui_params,
                'priority_level': 'highest'
            },
            'conversion_statistics': {
                'total_sections_processed': len(yaml_config.get('tv_complete_configuration', {})),
                'total_parameters_converted': self._count_yaml_parameters(yaml_config),
                'conversion_success': True
            }
        }
        
        yaml_config['conversion_metadata'] = metadata
        return yaml_config
    
    def _validate_enhanced_yaml(self, enhanced_yaml: Dict[str, Any]) -> Dict[str, Any]:
        """Validate enhanced YAML structure and content"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Check 1: Required top-level sections
            required_sections = ['tv_complete_configuration', 'conversion_metadata', 'ui_injection_log']
            for section in required_sections:
                if section in enhanced_yaml:
                    validation['checks_performed'].append(f"Required section present: {section}")
                else:
                    validation['warnings'].append(f"Optional section missing: {section}")
            
            # Check 2: TV configuration structure
            if 'tv_complete_configuration' in enhanced_yaml:
                tv_config = enhanced_yaml['tv_complete_configuration']
                expected_subsections = ['tv_master', 'signals', 'portfolio_long', 'portfolio_short', 'portfolio_manual', 'tbs_strategy']
                
                for subsection in expected_subsections:
                    if subsection in tv_config:
                        validation['checks_performed'].append(f"TV subsection present: {subsection}")
                    else:
                        validation['warnings'].append(f"TV subsection missing: {subsection}")
            
            # Check 3: UI injection validation
            if 'ui_injection_log' in enhanced_yaml:
                injection_log = enhanced_yaml['ui_injection_log']
                if 'injected_parameters' in injection_log:
                    param_count = len(injection_log['injected_parameters'])
                    validation['checks_performed'].append(f"UI parameters injected: {param_count}")
                    
                    if param_count == 0:
                        validation['warnings'].append("No UI parameters were injected")
            
            # Check 4: Metadata completeness
            if 'conversion_metadata' in enhanced_yaml:
                metadata = enhanced_yaml['conversion_metadata']
                if 'conversion_info' in metadata and 'source_files' in metadata:
                    validation['checks_performed'].append("Conversion metadata complete")
                else:
                    validation['warnings'].append("Incomplete conversion metadata")
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            validation['is_valid'] = False
        
        return validation
    
    def _generate_parameter_diff_report(self, base_yaml: Dict[str, Any], enhanced_yaml: Dict[str, Any], ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed parameter difference report"""
        diff_report = {
            'ui_overrides_applied': [],
            'parameter_changes': [],
            'new_sections_added': [],
            'values_changed': 0,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Track UI parameter applications
            for param_name, param_value in ui_params.items():
                if param_value is not None:
                    diff_report['ui_overrides_applied'].append({
                        'parameter': param_name,
                        'ui_value': param_value,
                        'applied_to_sections': self._find_parameter_locations(enhanced_yaml, param_name, param_value)
                    })
            
            # Compare specific sections for changes
            if 'tv_complete_configuration' in both base_yaml and enhanced_yaml:
                base_tv = base_yaml['tv_complete_configuration']
                enhanced_tv = enhanced_yaml['tv_complete_configuration']
                
                self._compare_tv_sections(base_tv, enhanced_tv, diff_report)
            
            diff_report['values_changed'] = len(diff_report['parameter_changes'])
            
        except Exception as e:
            diff_report['error'] = str(e)
            logger.error(f"Failed to generate parameter diff report: {e}")
        
        return diff_report
    
    def _deep_copy_yaml(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of YAML configuration"""
        import copy
        return copy.deepcopy(yaml_config)
    
    def _count_yaml_parameters(self, yaml_config: Dict[str, Any]) -> int:
        """Count total parameters in YAML configuration"""
        count = 0
        
        def count_recursive(obj):
            nonlocal count
            if isinstance(obj, dict):
                count += len(obj)
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        count_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        count_recursive(item)
        
        count_recursive(yaml_config)
        return count
    
    def _compare_yaml_sections(self, original: Dict[str, Any], modified: Dict[str, Any], diff_report: Dict[str, Any], path_prefix: str):
        """Compare YAML sections recursively"""
        for key in modified:
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            
            if key not in original:
                diff_report['new_parameters'].append({
                    'path': current_path,
                    'value': modified[key]
                })
            elif original[key] != modified[key]:
                diff_report['parameter_changes'].append({
                    'path': current_path,
                    'original_value': original[key],
                    'new_value': modified[key]
                })
                
                # Recursive comparison for nested structures
                if isinstance(original[key], dict) and isinstance(modified[key], dict):
                    self._compare_yaml_sections(original[key], modified[key], diff_report, current_path)
    
    def _find_parameter_locations(self, yaml_config: Dict[str, Any], param_name: str, param_value: Any) -> List[str]:
        """Find all locations where a parameter was applied"""
        locations = []
        
        def search_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if value == param_value and param_name.lower() in key.lower():
                        locations.append(current_path)
                    if isinstance(value, (dict, list)):
                        search_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    if isinstance(item, (dict, list)):
                        search_recursive(item, current_path)
        
        search_recursive(yaml_config)
        return locations
    
    def _compare_tv_sections(self, base_tv: Dict[str, Any], enhanced_tv: Dict[str, Any], diff_report: Dict[str, Any]):
        """Compare TV configuration sections specifically"""
        tv_sections = ['tv_master', 'signals', 'portfolio_long', 'portfolio_short', 'portfolio_manual', 'tbs_strategy']
        
        for section in tv_sections:
            if section in enhanced_tv:
                if section not in base_tv:
                    diff_report['new_sections_added'].append(section)
                else:
                    # Compare section contents
                    self._compare_yaml_sections(base_tv[section], enhanced_tv[section], diff_report, section)