#!/usr/bin/env python3
"""
Hot-Reloading Configuration System
Phase 2 Day 5: Excel Configuration Integration

This module provides real-time configuration updates without system restart
for the DTE Enhanced Triple Straddle Rolling Analysis Framework.

Author: The Augster
Date: 2025-06-20
Version: 5.0.0 (Phase 2 Day 5 Hot-Reload System)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
import hashlib
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConfigurationChange:
    """Represents a configuration change event"""
    timestamp: datetime
    parameter: str
    old_value: Any
    new_value: Any
    sheet_name: str
    skill_level: str
    hot_reload_enabled: bool
    validation_result: bool
    error_message: Optional[str] = None

@dataclass
class HotReloadConfig:
    """Configuration for hot-reload system"""
    enabled: bool = True
    check_interval: float = 1.0
    max_retries: int = 3
    validation_enabled: bool = True
    backup_enabled: bool = True
    change_log_enabled: bool = True
    notification_callbacks: List[Callable] = field(default_factory=list)

class ConfigurationValidator:
    """Validates configuration changes before applying them"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for different parameter types"""
        
        return {
            'DTE_LEARNING_ENABLED': {
                'type': bool,
                'required': True,
                'description': 'Must be True or False'
            },
            'DTE_RANGE_MIN': {
                'type': int,
                'min_value': 0,
                'max_value': 30,
                'description': 'Must be integer between 0 and 30'
            },
            'DTE_RANGE_MAX': {
                'type': int,
                'min_value': 0,
                'max_value': 30,
                'description': 'Must be integer between 0 and 30'
            },
            'DTE_FOCUS_RANGE_MIN': {
                'type': int,
                'min_value': 0,
                'max_value': 4,
                'description': 'Must be integer between 0 and 4'
            },
            'DTE_FOCUS_RANGE_MAX': {
                'type': int,
                'min_value': 0,
                'max_value': 4,
                'description': 'Must be integer between 0 and 4'
            },
            'ATM_BASE_WEIGHT': {
                'type': float,
                'min_value': 0.05,
                'max_value': 0.80,
                'description': 'Must be float between 0.05 and 0.80'
            },
            'ITM1_BASE_WEIGHT': {
                'type': float,
                'min_value': 0.05,
                'max_value': 0.80,
                'description': 'Must be float between 0.05 and 0.80'
            },
            'OTM1_BASE_WEIGHT': {
                'type': float,
                'min_value': 0.05,
                'max_value': 0.80,
                'description': 'Must be float between 0.05 and 0.80'
            },
            'TARGET_PROCESSING_TIME': {
                'type': float,
                'min_value': 0.1,
                'max_value': 60.0,
                'description': 'Must be float between 0.1 and 60.0 seconds'
            },
            'CONFIDENCE_THRESHOLD': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'description': 'Must be float between 0.0 and 1.0'
            },
            'ACCURACY_TARGET': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'description': 'Must be float between 0.0 and 1.0'
            }
        }
    
    def validate_parameter(self, parameter: str, value: Any, sheet_name: str) -> tuple[bool, Optional[str]]:
        """Validate a single parameter change"""
        
        try:
            if parameter not in self.validation_rules:
                return True, None  # No specific validation rule
            
            rule = self.validation_rules[parameter]
            
            # Type validation
            expected_type = rule.get('type')
            if expected_type and not isinstance(value, expected_type):
                try:
                    # Try to convert
                    if expected_type == bool:
                        if isinstance(value, str):
                            value = value.lower() in ['true', '1', 'yes', 'on']
                        else:
                            value = bool(value)
                    elif expected_type == int:
                        value = int(float(value))
                    elif expected_type == float:
                        value = float(value)
                except (ValueError, TypeError):
                    return False, f"Invalid type for {parameter}. {rule.get('description', '')}"
            
            # Range validation
            if 'min_value' in rule and value < rule['min_value']:
                return False, f"{parameter} value {value} below minimum {rule['min_value']}"
            
            if 'max_value' in rule and value > rule['max_value']:
                return False, f"{parameter} value {value} above maximum {rule['max_value']}"
            
            # Required validation
            if rule.get('required', False) and value is None:
                return False, f"{parameter} is required"
            
            # Custom validation for weight parameters
            if parameter.endswith('_WEIGHT'):
                return self._validate_weight_parameter(parameter, value)
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error for {parameter}: {str(e)}"
    
    def _validate_weight_parameter(self, parameter: str, value: float) -> tuple[bool, Optional[str]]:
        """Special validation for weight parameters"""
        
        if not 0.05 <= value <= 0.80:
            return False, f"Weight {parameter} must be between 0.05 and 0.80"
        
        return True, None
    
    def validate_weight_sum(self, atm_weight: float, itm1_weight: float, otm1_weight: float) -> tuple[bool, Optional[str]]:
        """Validate that weights sum to approximately 1.0"""
        
        total = atm_weight + itm1_weight + otm1_weight
        if not 0.99 <= total <= 1.01:
            return False, f"Weights must sum to 1.0 (current sum: {total:.3f})"
        
        return True, None

class ExcelConfigurationWatcher(FileSystemEventHandler):
    """Watches Excel configuration files for changes"""
    
    def __init__(self, hot_reload_system):
        self.hot_reload_system = hot_reload_system
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events"""
        
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's an Excel configuration file
        if file_path.suffix.lower() in ['.xlsx', '.xls'] and 'CONFIGURATION' in file_path.name.upper():
            current_time = time.time()
            
            # Debounce rapid file changes
            if file_path in self.last_modified:
                if current_time - self.last_modified[file_path] < 2.0:
                    return
            
            self.last_modified[file_path] = current_time
            
            logger.info(f"📁 Configuration file changed: {file_path}")
            
            # Trigger hot reload
            threading.Thread(
                target=self.hot_reload_system.reload_configuration,
                args=(str(file_path),),
                daemon=True
            ).start()

class HotReloadConfigurationSystem:
    """
    Hot-Reloading Configuration System for DTE Enhanced Framework
    
    Provides real-time configuration updates without system restart
    """
    
    def __init__(self, config_file_path: str, hot_reload_config: Optional[HotReloadConfig] = None):
        """Initialize hot-reload configuration system"""
        
        self.config_file_path = Path(config_file_path)
        self.hot_reload_config = hot_reload_config or HotReloadConfig()
        
        # Configuration state
        self.current_config = {}
        self.config_hash = ""
        self.change_history = []
        self.validator = ConfigurationValidator()
        
        # Threading
        self.reload_lock = threading.Lock()
        self.observer = None
        self.monitoring_thread = None
        self.is_running = False
        
        # Callbacks
        self.change_callbacks = []
        
        # Backup system
        self.backup_dir = self.config_file_path.parent / "config_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info("🔄 Hot-Reload Configuration System initialized")
        logger.info(f"📁 Monitoring: {self.config_file_path}")
        logger.info(f"⚙️ Hot-reload enabled: {self.hot_reload_config.enabled}")
    
    def start_monitoring(self):
        """Start monitoring configuration file for changes"""
        
        if not self.hot_reload_config.enabled:
            logger.info("⚠️ Hot-reload disabled, skipping monitoring")
            return
        
        try:
            # Load initial configuration
            self.reload_configuration()
            
            # Start file system watcher
            self.observer = Observer()
            event_handler = ExcelConfigurationWatcher(self)
            self.observer.schedule(
                event_handler,
                str(self.config_file_path.parent),
                recursive=False
            )
            self.observer.start()
            
            # Start monitoring thread
            self.is_running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info("✅ Hot-reload monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start hot-reload monitoring: {e}")
            raise
    
    def stop_monitoring(self):
        """Stop monitoring configuration file"""
        
        try:
            self.is_running = False
            
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            
            logger.info("🛑 Hot-reload monitoring stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping hot-reload monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.is_running:
            try:
                time.sleep(self.hot_reload_config.check_interval)
                
                # Check for configuration changes
                if self._has_configuration_changed():
                    self.reload_configuration()
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _has_configuration_changed(self) -> bool:
        """Check if configuration file has changed"""
        
        try:
            if not self.config_file_path.exists():
                return False
            
            # Calculate current file hash
            current_hash = self._calculate_file_hash()
            
            if current_hash != self.config_hash:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking configuration changes: {e}")
            return False
    
    def _calculate_file_hash(self) -> str:
        """Calculate hash of configuration file"""
        
        try:
            with open(self.config_file_path, 'rb') as f:
                file_content = f.read()
                return hashlib.md5(file_content).hexdigest()
        except Exception:
            return ""

    def reload_configuration(self, file_path: Optional[str] = None):
        """Reload configuration from Excel file"""

        with self.reload_lock:
            try:
                config_path = Path(file_path) if file_path else self.config_file_path

                if not config_path.exists():
                    logger.warning(f"⚠️ Configuration file not found: {config_path}")
                    return

                logger.info(f"🔄 Reloading configuration from: {config_path}")

                # Create backup if enabled
                if self.hot_reload_config.backup_enabled:
                    self._create_backup()

                # Load new configuration
                new_config = self._load_excel_configuration(config_path)

                # Validate configuration changes
                changes = self._detect_configuration_changes(new_config)

                if changes:
                    # Validate all changes
                    valid_changes = []
                    invalid_changes = []

                    for change in changes:
                        if change.hot_reload_enabled:
                            is_valid, error_msg = self.validator.validate_parameter(
                                change.parameter, change.new_value, change.sheet_name
                            )

                            if is_valid:
                                change.validation_result = True
                                valid_changes.append(change)
                            else:
                                change.validation_result = False
                                change.error_message = error_msg
                                invalid_changes.append(change)
                        else:
                            logger.info(f"⚠️ Parameter {change.parameter} not hot-reloadable, skipping")

                    # Apply valid changes
                    if valid_changes:
                        self._apply_configuration_changes(valid_changes)
                        self.current_config = new_config
                        self.config_hash = self._calculate_file_hash()

                        logger.info(f"✅ Applied {len(valid_changes)} configuration changes")

                        # Notify callbacks
                        self._notify_change_callbacks(valid_changes)

                    # Log invalid changes
                    if invalid_changes:
                        logger.warning(f"⚠️ {len(invalid_changes)} invalid configuration changes:")
                        for change in invalid_changes:
                            logger.warning(f"   {change.parameter}: {change.error_message}")

                    # Store change history
                    if self.hot_reload_config.change_log_enabled:
                        self.change_history.extend(valid_changes + invalid_changes)
                        self._trim_change_history()

                else:
                    logger.info("ℹ️ No configuration changes detected")

            except Exception as e:
                logger.error(f"❌ Error reloading configuration: {e}")
                raise

    def _load_excel_configuration(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from Excel file"""

        try:
            config = {}

            # Read all sheets
            excel_file = pd.ExcelFile(config_path)

            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(config_path, sheet_name=sheet_name)

                    # Convert to dictionary format
                    sheet_config = {}
                    for _, row in df.iterrows():
                        param_name = row.get('Parameter')
                        param_value = row.get('Value')

                        if pd.notna(param_name) and pd.notna(param_value):
                            sheet_config[param_name] = param_value

                    config[sheet_name] = sheet_config

                except Exception as e:
                    logger.warning(f"⚠️ Error reading sheet {sheet_name}: {e}")

            return config

        except Exception as e:
            logger.error(f"❌ Error loading Excel configuration: {e}")
            raise

    def _detect_configuration_changes(self, new_config: Dict[str, Any]) -> List[ConfigurationChange]:
        """Detect changes between current and new configuration"""

        changes = []

        try:
            for sheet_name, sheet_config in new_config.items():
                current_sheet = self.current_config.get(sheet_name, {})

                for param_name, new_value in sheet_config.items():
                    current_value = current_sheet.get(param_name)

                    if current_value != new_value:
                        # Get additional metadata
                        skill_level, hot_reload_enabled = self._get_parameter_metadata(
                            sheet_name, param_name
                        )

                        change = ConfigurationChange(
                            timestamp=datetime.now(),
                            parameter=param_name,
                            old_value=current_value,
                            new_value=new_value,
                            sheet_name=sheet_name,
                            skill_level=skill_level,
                            hot_reload_enabled=hot_reload_enabled,
                            validation_result=False
                        )

                        changes.append(change)

            return changes

        except Exception as e:
            logger.error(f"❌ Error detecting configuration changes: {e}")
            return []

    def _get_parameter_metadata(self, sheet_name: str, param_name: str) -> tuple[str, bool]:
        """Get parameter metadata (skill level, hot-reload enabled)"""

        try:
            # Read the sheet to get metadata
            df = pd.read_excel(self.config_file_path, sheet_name=sheet_name)

            param_row = df[df['Parameter'] == param_name]

            if not param_row.empty:
                skill_level = param_row.iloc[0].get('Skill_Level', 'Novice')
                hot_reload = param_row.iloc[0].get('Hot_Reload', True)
                return skill_level, hot_reload

            return 'Novice', True

        except Exception:
            return 'Novice', True

    def _apply_configuration_changes(self, changes: List[ConfigurationChange]):
        """Apply validated configuration changes"""

        try:
            for change in changes:
                logger.info(f"🔄 Applying: {change.parameter} = {change.new_value} (was: {change.old_value})")

                # Update current configuration
                if change.sheet_name not in self.current_config:
                    self.current_config[change.sheet_name] = {}

                self.current_config[change.sheet_name][change.parameter] = change.new_value

        except Exception as e:
            logger.error(f"❌ Error applying configuration changes: {e}")
            raise

    def _notify_change_callbacks(self, changes: List[ConfigurationChange]):
        """Notify registered callbacks about configuration changes"""

        try:
            for callback in self.change_callbacks:
                try:
                    callback(changes)
                except Exception as e:
                    logger.error(f"❌ Error in change callback: {e}")

        except Exception as e:
            logger.error(f"❌ Error notifying change callbacks: {e}")

    def _create_backup(self):
        """Create backup of current configuration"""

        try:
            if self.config_file_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{self.config_file_path.stem}_backup_{timestamp}{self.config_file_path.suffix}"
                backup_path = self.backup_dir / backup_name

                import shutil
                shutil.copy2(self.config_file_path, backup_path)

                logger.info(f"💾 Configuration backup created: {backup_path}")

                # Clean old backups (keep last 10)
                self._cleanup_old_backups()

        except Exception as e:
            logger.error(f"❌ Error creating configuration backup: {e}")

    def _cleanup_old_backups(self):
        """Clean up old backup files"""

        try:
            backup_files = list(self.backup_dir.glob(f"{self.config_file_path.stem}_backup_*"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the 10 most recent backups
            for old_backup in backup_files[10:]:
                old_backup.unlink()
                logger.info(f"🗑️ Removed old backup: {old_backup}")

        except Exception as e:
            logger.error(f"❌ Error cleaning up old backups: {e}")

    def _trim_change_history(self):
        """Trim change history to prevent memory issues"""

        max_history = 1000
        if len(self.change_history) > max_history:
            self.change_history = self.change_history[-max_history:]

    def register_change_callback(self, callback: Callable[[List[ConfigurationChange]], None]):
        """Register callback for configuration changes"""

        self.change_callbacks.append(callback)
        logger.info(f"📞 Registered configuration change callback: {callback.__name__}")

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""

        return self.current_config.copy()

    def get_parameter_value(self, sheet_name: str, parameter: str, default: Any = None) -> Any:
        """Get specific parameter value"""

        return self.current_config.get(sheet_name, {}).get(parameter, default)

    def get_change_history(self, limit: int = 100) -> List[ConfigurationChange]:
        """Get recent configuration change history"""

        return self.change_history[-limit:]

    def force_reload(self):
        """Force reload configuration"""

        logger.info("🔄 Force reloading configuration...")
        self.reload_configuration()

    def validate_current_config(self) -> Dict[str, List[str]]:
        """Validate current configuration and return any issues"""

        issues = {}

        try:
            for sheet_name, sheet_config in self.current_config.items():
                sheet_issues = []

                for param_name, value in sheet_config.items():
                    is_valid, error_msg = self.validator.validate_parameter(
                        param_name, value, sheet_name
                    )

                    if not is_valid:
                        sheet_issues.append(f"{param_name}: {error_msg}")

                if sheet_issues:
                    issues[sheet_name] = sheet_issues

            # Special validation for weight parameters
            dte_config = self.current_config.get('DTE_Learning_Config', {})
            atm_weight = dte_config.get('ATM_BASE_WEIGHT', 0.5)
            itm1_weight = dte_config.get('ITM1_BASE_WEIGHT', 0.3)
            otm1_weight = dte_config.get('OTM1_BASE_WEIGHT', 0.2)

            is_valid, error_msg = self.validator.validate_weight_sum(atm_weight, itm1_weight, otm1_weight)
            if not is_valid:
                if 'DTE_Learning_Config' not in issues:
                    issues['DTE_Learning_Config'] = []
                issues['DTE_Learning_Config'].append(error_msg)

            return issues

        except Exception as e:
            logger.error(f"❌ Error validating configuration: {e}")
            return {'validation_error': [str(e)]}

# Example usage and testing
def example_change_callback(changes: List[ConfigurationChange]):
    """Example callback for configuration changes"""

    logger.info(f"📢 Configuration changed! {len(changes)} parameters updated:")
    for change in changes:
        logger.info(f"   {change.parameter}: {change.old_value} → {change.new_value}")

if __name__ == "__main__":
    # Example usage
    config_file = "excel_config_templates/DTE_ENHANCED_CONFIGURATION_TEMPLATE.xlsx"

    if Path(config_file).exists():
        # Initialize hot-reload system
        hot_reload_config = HotReloadConfig(
            enabled=True,
            check_interval=2.0,
            validation_enabled=True,
            backup_enabled=True,
            change_log_enabled=True
        )

        system = HotReloadConfigurationSystem(config_file, hot_reload_config)

        # Register callback
        system.register_change_callback(example_change_callback)

        # Start monitoring
        system.start_monitoring()

        logger.info("🔄 Hot-reload system running. Modify the Excel file to test...")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Stopping hot-reload system...")
            system.stop_monitoring()
    else:
        logger.error(f"❌ Configuration file not found: {config_file}")
        logger.info("💡 Run create_excel_config_templates.py first to create the template")
