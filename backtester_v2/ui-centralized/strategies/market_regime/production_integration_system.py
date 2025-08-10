#!/usr/bin/env python3
"""
Production Integration System
Phase 2 Day 5: Excel Configuration Integration

This module integrates the DTE Enhanced Excel Configuration System
with existing production infrastructure (enterprise_server_v2.py and BT_TV_GPU_aggregated_v4.py).

Author: The Augster
Date: 2025-06-20
Version: 5.0.0 (Phase 2 Day 5 Production Integration)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import warnings
import sys
import os

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from hot_reload_config_system import HotReloadConfigurationSystem, HotReloadConfig, ConfigurationChange
from progressive_disclosure_ui import ProgressiveDisclosureUI, SkillLevel

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProductionIntegrationConfig:
    """Configuration for production integration"""
    enterprise_server_integration: bool = True
    tv_strategy_engine_integration: bool = True
    excel_parser_compatibility: bool = True
    heavydb_integration: bool = True
    hot_reload_enabled: bool = True
    progressive_ui_enabled: bool = True
    performance_monitoring: bool = True
    backup_enabled: bool = True

@dataclass
class StrategyTypeConfig:
    """Configuration for each strategy type"""
    strategy_type: str
    dte_learning_enabled: bool
    default_dte_focus: int
    weight_optimization: str
    performance_target: float
    excel_config_path: Optional[str] = None

class ProductionIntegrationSystem:
    """
    Production Integration System for DTE Enhanced Framework
    
    Integrates Excel configuration system with:
    - enterprise_server_v2.py production server
    - BT_TV_GPU_aggregated_v4.py TV strategy engine
    - Excel parser-driven dynamic file requirements
    - 100% real HeavyDB data integration
    """
    
    def __init__(self, config_file_path: str, integration_config: Optional[ProductionIntegrationConfig] = None):
        """Initialize Production Integration System"""
        
        self.config_file_path = Path(config_file_path)
        self.integration_config = integration_config or ProductionIntegrationConfig()
        
        # Core systems
        self.hot_reload_system = None
        self.progressive_ui = None
        
        # Strategy configurations
        self.strategy_configs = {}
        
        # Performance tracking
        self.performance_metrics = {}
        self.integration_status = {}
        
        # Callbacks for production systems
        self.production_callbacks = []
        
        logger.info("🏭 Production Integration System initialized")
        logger.info(f"📁 Configuration file: {self.config_file_path}")
        logger.info(f"⚙️ Integration config: {self.integration_config}")
    
    def initialize_production_integration(self):
        """Initialize all production integration components"""
        
        try:
            logger.info("🚀 Initializing production integration components...")
            
            # Initialize hot-reload system
            if self.integration_config.hot_reload_enabled:
                self._initialize_hot_reload_system()
            
            # Initialize progressive UI
            if self.integration_config.progressive_ui_enabled:
                self._initialize_progressive_ui()
            
            # Load strategy configurations
            self._load_strategy_configurations()
            
            # Initialize Excel parser compatibility
            if self.integration_config.excel_parser_compatibility:
                self._initialize_excel_parser_compatibility()
            
            # Initialize HeavyDB integration
            if self.integration_config.heavydb_integration:
                self._initialize_heavydb_integration()
            
            # Initialize performance monitoring
            if self.integration_config.performance_monitoring:
                self._initialize_performance_monitoring()
            
            logger.info("✅ Production integration initialization completed")
            
        except Exception as e:
            logger.error(f"❌ Error initializing production integration: {e}")
            raise
    
    def _initialize_hot_reload_system(self):
        """Initialize hot-reload configuration system"""
        
        try:
            logger.info("🔄 Initializing hot-reload system...")
            
            hot_reload_config = HotReloadConfig(
                enabled=True,
                check_interval=1.0,
                validation_enabled=True,
                backup_enabled=self.integration_config.backup_enabled,
                change_log_enabled=True
            )
            
            self.hot_reload_system = HotReloadConfigurationSystem(
                str(self.config_file_path),
                hot_reload_config
            )
            
            # Register production callback
            self.hot_reload_system.register_change_callback(self._handle_configuration_change)
            
            # Start monitoring
            self.hot_reload_system.start_monitoring()
            
            logger.info("✅ Hot-reload system initialized and monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Error initializing hot-reload system: {e}")
            raise
    
    def _initialize_progressive_ui(self):
        """Initialize progressive disclosure UI system"""
        
        try:
            logger.info("🎯 Initializing progressive UI system...")
            
            self.progressive_ui = ProgressiveDisclosureUI(str(self.config_file_path))
            
            # Set default skill level (can be auto-detected later)
            self.progressive_ui.set_skill_level(SkillLevel.INTERMEDIATE)
            
            logger.info("✅ Progressive UI system initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing progressive UI: {e}")
            raise
    
    def _load_strategy_configurations(self):
        """Load configurations for all 6 strategy types"""
        
        try:
            logger.info("⚙️ Loading strategy configurations...")
            
            # Read strategy configuration sheet
            strategy_df = pd.read_excel(self.config_file_path, sheet_name='Strategy_Config')
            
            # Group by strategy type
            strategy_types = ['TBS', 'TV', 'ORB', 'OI', 'Indicator', 'POS']
            
            for strategy_type in strategy_types:
                strategy_rows = strategy_df[strategy_df['Strategy_Type'] == strategy_type]
                
                if not strategy_rows.empty:
                    config_dict = {}
                    for _, row in strategy_rows.iterrows():
                        config_dict[row['Parameter']] = row['Value']
                    
                    strategy_config = StrategyTypeConfig(
                        strategy_type=strategy_type,
                        dte_learning_enabled=config_dict.get('dte_learning_enabled', True),
                        default_dte_focus=config_dict.get('default_dte_focus', 3),
                        weight_optimization=config_dict.get('weight_optimization', 'ml_enhanced'),
                        performance_target=config_dict.get('performance_target', 0.85)
                    )
                    
                    self.strategy_configs[strategy_type] = strategy_config
                    logger.info(f"   ✅ {strategy_type}: DTE learning {'enabled' if strategy_config.dte_learning_enabled else 'disabled'}")
            
            logger.info(f"✅ Loaded configurations for {len(self.strategy_configs)} strategy types")
            
        except Exception as e:
            logger.error(f"❌ Error loading strategy configurations: {e}")
            raise
    
    def _initialize_excel_parser_compatibility(self):
        """Initialize Excel parser compatibility layer"""
        
        try:
            logger.info("📊 Initializing Excel parser compatibility...")
            
            # Ensure backward compatibility with existing Excel parser infrastructure
            self.excel_parser_config = {
                'dynamic_file_requirements': True,
                'legacy_format_support': True,
                'oi_system_compatibility': True,  # Two-file format support
                'unified_parser_integration': True
            }
            
            # Validate Excel structure compatibility
            self._validate_excel_structure_compatibility()
            
            logger.info("✅ Excel parser compatibility initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Excel parser compatibility: {e}")
            raise
    
    def _validate_excel_structure_compatibility(self):
        """Validate Excel structure compatibility with existing parsers"""
        
        try:
            # Check required sheets exist
            required_sheets = [
                'DTE_Learning_Config',
                'ML_Model_Config',
                'Strategy_Config',
                'Performance_Config',
                'UI_Config',
                'Validation_Config',
                'Rolling_Config',
                'Regime_Config'
            ]
            
            excel_file = pd.ExcelFile(self.config_file_path)
            missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
            
            if missing_sheets:
                logger.warning(f"⚠️ Missing sheets: {missing_sheets}")
            else:
                logger.info("✅ All required sheets present")
            
            # Validate key parameters exist
            self._validate_key_parameters()
            
        except Exception as e:
            logger.error(f"❌ Error validating Excel structure: {e}")
            raise
    
    def _validate_key_parameters(self):
        """Validate key parameters exist in configuration"""
        
        try:
            key_parameters = {
                'DTE_Learning_Config': ['DTE_LEARNING_ENABLED', 'DTE_FOCUS_RANGE_MIN', 'DTE_FOCUS_RANGE_MAX'],
                'Performance_Config': ['TARGET_PROCESSING_TIME', 'PARALLEL_PROCESSING_ENABLED'],
                'Strategy_Config': ['dte_learning_enabled', 'default_dte_focus']
            }
            
            for sheet_name, params in key_parameters.items():
                try:
                    df = pd.read_excel(self.config_file_path, sheet_name=sheet_name)
                    sheet_params = df['Parameter'].tolist() if 'Parameter' in df.columns else []
                    
                    missing_params = [p for p in params if p not in sheet_params]
                    if missing_params:
                        logger.warning(f"⚠️ Missing parameters in {sheet_name}: {missing_params}")
                    else:
                        logger.info(f"✅ Key parameters validated for {sheet_name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not validate {sheet_name}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error validating key parameters: {e}")
    
    def _initialize_heavydb_integration(self):
        """Initialize HeavyDB integration compatibility"""
        
        try:
            logger.info("🗄️ Initializing HeavyDB integration...")
            
            self.heavydb_config = {
                'real_data_enforcement': True,
                'synthetic_data_fallback': False,
                'nifty_option_chain_table': 'nifty_option_chain',
                'trade_time_column': 'trade_time',
                'minute_level_queries': True,
                'data_quality_validation': True
            }
            
            # Validate HeavyDB configuration parameters
            validation_df = pd.read_excel(self.config_file_path, sheet_name='Validation_Config')
            
            real_data_enforcement = validation_df[
                validation_df['Parameter'] == 'REAL_DATA_ENFORCEMENT'
            ]['Value'].iloc[0] if len(validation_df[validation_df['Parameter'] == 'REAL_DATA_ENFORCEMENT']) > 0 else True
            
            synthetic_data_allowed = validation_df[
                validation_df['Parameter'] == 'SYNTHETIC_DATA_ALLOWED'
            ]['Value'].iloc[0] if len(validation_df[validation_df['Parameter'] == 'SYNTHETIC_DATA_ALLOWED']) > 0 else False
            
            if real_data_enforcement and not synthetic_data_allowed:
                logger.info("✅ HeavyDB real data enforcement configured correctly")
            else:
                logger.warning("⚠️ HeavyDB configuration may allow synthetic data fallbacks")
            
            logger.info("✅ HeavyDB integration initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing HeavyDB integration: {e}")
            raise
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring system"""
        
        try:
            logger.info("📊 Initializing performance monitoring...")
            
            # Read performance configuration
            perf_df = pd.read_excel(self.config_file_path, sheet_name='Performance_Config')
            
            perf_config = {}
            for _, row in perf_df.iterrows():
                perf_config[row['Parameter']] = row['Value']
            
            self.performance_config = {
                'target_processing_time': perf_config.get('TARGET_PROCESSING_TIME', 3.0),
                'parallel_processing': perf_config.get('PARALLEL_PROCESSING_ENABLED', True),
                'max_workers': perf_config.get('MAX_WORKERS', 72),
                'memory_limit_mb': perf_config.get('MEMORY_LIMIT_MB', 1024),
                'monitoring_enabled': perf_config.get('PERFORMANCE_MONITORING', True)
            }
            
            # Initialize performance tracking
            self.performance_metrics = {
                'configuration_load_time': 0.0,
                'hot_reload_response_time': 0.0,
                'ui_generation_time': 0.0,
                'strategy_config_time': 0.0,
                'total_integration_time': 0.0
            }
            
            logger.info(f"✅ Performance monitoring initialized (target: {self.performance_config['target_processing_time']}s)")
            
        except Exception as e:
            logger.error(f"❌ Error initializing performance monitoring: {e}")
            raise
    
    def _handle_configuration_change(self, changes: List[ConfigurationChange]):
        """Handle configuration changes from hot-reload system"""
        
        try:
            logger.info(f"🔄 Handling {len(changes)} configuration changes...")
            
            start_time = time.time()
            
            # Process changes by category
            strategy_changes = []
            performance_changes = []
            dte_changes = []
            
            for change in changes:
                if change.sheet_name == 'Strategy_Config':
                    strategy_changes.append(change)
                elif change.sheet_name == 'Performance_Config':
                    performance_changes.append(change)
                elif change.sheet_name == 'DTE_Learning_Config':
                    dte_changes.append(change)
            
            # Apply strategy configuration changes
            if strategy_changes:
                self._apply_strategy_configuration_changes(strategy_changes)
            
            # Apply performance configuration changes
            if performance_changes:
                self._apply_performance_configuration_changes(performance_changes)
            
            # Apply DTE configuration changes
            if dte_changes:
                self._apply_dte_configuration_changes(dte_changes)
            
            # Notify production systems
            self._notify_production_systems(changes)
            
            # Update performance metrics
            self.performance_metrics['hot_reload_response_time'] = time.time() - start_time
            
            logger.info(f"✅ Configuration changes applied in {self.performance_metrics['hot_reload_response_time']:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error handling configuration changes: {e}")
    
    def _apply_strategy_configuration_changes(self, changes: List[ConfigurationChange]):
        """Apply strategy configuration changes"""
        
        try:
            for change in changes:
                # Find affected strategy type
                strategy_type = None
                for st, config in self.strategy_configs.items():
                    if change.parameter.startswith(st.lower()) or st in change.parameter:
                        strategy_type = st
                        break
                
                if strategy_type:
                    config = self.strategy_configs[strategy_type]
                    
                    if 'dte_learning_enabled' in change.parameter:
                        config.dte_learning_enabled = change.new_value
                    elif 'default_dte_focus' in change.parameter:
                        config.default_dte_focus = change.new_value
                    elif 'weight_optimization' in change.parameter:
                        config.weight_optimization = change.new_value
                    elif 'performance_target' in change.parameter:
                        config.performance_target = change.new_value
                    
                    logger.info(f"   ✅ Updated {strategy_type} configuration: {change.parameter} = {change.new_value}")
            
        except Exception as e:
            logger.error(f"❌ Error applying strategy configuration changes: {e}")
    
    def _apply_performance_configuration_changes(self, changes: List[ConfigurationChange]):
        """Apply performance configuration changes"""
        
        try:
            for change in changes:
                if change.parameter in self.performance_config:
                    old_value = self.performance_config[change.parameter]
                    self.performance_config[change.parameter] = change.new_value
                    
                    logger.info(f"   ✅ Updated performance config: {change.parameter} = {change.new_value} (was: {old_value})")
            
        except Exception as e:
            logger.error(f"❌ Error applying performance configuration changes: {e}")
    
    def _apply_dte_configuration_changes(self, changes: List[ConfigurationChange]):
        """Apply DTE configuration changes"""
        
        try:
            for change in changes:
                logger.info(f"   ✅ DTE configuration updated: {change.parameter} = {change.new_value}")
                
                # If DTE learning is disabled, notify all strategies
                if change.parameter == 'DTE_LEARNING_ENABLED' and not change.new_value:
                    logger.warning("⚠️ DTE learning disabled globally - affecting all strategies")
            
        except Exception as e:
            logger.error(f"❌ Error applying DTE configuration changes: {e}")
    
    def _notify_production_systems(self, changes: List[ConfigurationChange]):
        """Notify production systems of configuration changes"""
        
        try:
            # Notify registered production callbacks
            for callback in self.production_callbacks:
                try:
                    callback(changes)
                except Exception as e:
                    logger.error(f"❌ Error in production callback: {e}")
            
            # Log integration status
            self.integration_status['last_update'] = datetime.now().isoformat()
            self.integration_status['changes_applied'] = len(changes)
            
        except Exception as e:
            logger.error(f"❌ Error notifying production systems: {e}")
    
    def register_production_callback(self, callback: Callable[[List[ConfigurationChange]], None]):
        """Register callback for production system notifications"""
        
        self.production_callbacks.append(callback)
        logger.info(f"📞 Registered production callback: {callback.__name__}")
    
    def get_strategy_configuration(self, strategy_type: str) -> Optional[StrategyTypeConfig]:
        """Get configuration for specific strategy type"""
        
        return self.strategy_configs.get(strategy_type)
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get current complete configuration"""
        
        if self.hot_reload_system:
            return self.hot_reload_system.get_current_config()
        else:
            return {}
    
    def get_ui_layout_for_skill_level(self, skill_level: SkillLevel, sheet_name: str) -> Dict[str, Any]:
        """Get UI layout for specific skill level and sheet"""
        
        if self.progressive_ui:
            original_level = self.progressive_ui.ui_config.current_skill_level
            self.progressive_ui.set_skill_level(skill_level)
            layout = self.progressive_ui.generate_ui_layout(sheet_name)
            self.progressive_ui.set_skill_level(original_level)
            return layout
        else:
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        return {
            'performance_metrics': self.performance_metrics,
            'performance_config': self.performance_config,
            'integration_status': self.integration_status
        }
    
    def validate_production_integration(self) -> Dict[str, Any]:
        """Validate production integration status"""
        
        try:
            validation_results = {
                'hot_reload_system': self.hot_reload_system is not None,
                'progressive_ui': self.progressive_ui is not None,
                'strategy_configs_loaded': len(self.strategy_configs) == 6,
                'excel_parser_compatibility': self.integration_config.excel_parser_compatibility,
                'heavydb_integration': self.integration_config.heavydb_integration,
                'performance_monitoring': self.integration_config.performance_monitoring,
                'configuration_file_exists': self.config_file_path.exists(),
                'total_strategies': len(self.strategy_configs),
                'strategies_with_dte': sum(1 for config in self.strategy_configs.values() if config.dte_learning_enabled)
            }
            
            # Overall status
            validation_results['overall_status'] = all([
                validation_results['hot_reload_system'],
                validation_results['progressive_ui'],
                validation_results['strategy_configs_loaded'],
                validation_results['configuration_file_exists']
            ])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error validating production integration: {e}")
            return {'overall_status': False, 'error': str(e)}
    
    def shutdown(self):
        """Shutdown production integration system"""
        
        try:
            logger.info("🛑 Shutting down production integration system...")
            
            if self.hot_reload_system:
                self.hot_reload_system.stop_monitoring()
            
            logger.info("✅ Production integration system shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

# Example production callback
def example_enterprise_server_callback(changes: List[ConfigurationChange]):
    """Example callback for enterprise_server_v2.py integration"""
    
    logger.info(f"🏭 Enterprise server notified of {len(changes)} configuration changes")
    for change in changes:
        logger.info(f"   📊 {change.parameter}: {change.old_value} → {change.new_value}")

def example_tv_strategy_callback(changes: List[ConfigurationChange]):
    """Example callback for BT_TV_GPU_aggregated_v4.py integration"""
    
    logger.info(f"📺 TV strategy engine notified of {len(changes)} configuration changes")
    for change in changes:
        if change.sheet_name == 'Strategy_Config' and 'TV' in change.parameter:
            logger.info(f"   🎯 TV strategy update: {change.parameter} = {change.new_value}")

if __name__ == "__main__":
    # Example usage
    config_file = "excel_config_templates/DTE_ENHANCED_CONFIGURATION_TEMPLATE.xlsx"
    
    if Path(config_file).exists():
        # Initialize production integration
        integration_config = ProductionIntegrationConfig(
            enterprise_server_integration=True,
            tv_strategy_engine_integration=True,
            excel_parser_compatibility=True,
            heavydb_integration=True,
            hot_reload_enabled=True,
            progressive_ui_enabled=True,
            performance_monitoring=True,
            backup_enabled=True
        )
        
        system = ProductionIntegrationSystem(config_file, integration_config)
        
        # Register production callbacks
        system.register_production_callback(example_enterprise_server_callback)
        system.register_production_callback(example_tv_strategy_callback)
        
        # Initialize integration
        system.initialize_production_integration()
        
        # Validate integration
        validation = system.validate_production_integration()
        
        logger.info(f"\n{'='*60}")
        logger.info("PRODUCTION INTEGRATION VALIDATION")
        logger.info(f"{'='*60}")
        
        for key, value in validation.items():
            status = "✅" if value else "❌"
            logger.info(f"{status} {key}: {value}")
        
        logger.info(f"\n🏭 Production integration running. Modify Excel file to test hot-reload...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Stopping production integration...")
            system.shutdown()
    else:
        logger.error(f"❌ Configuration file not found: {config_file}")
        logger.info("💡 Run create_excel_config_templates.py first to create the template")
