#!/usr/bin/env python3
"""
UI Integration Manager
=====================

This module provides UI integration components for the enhanced market regime system,
enabling seamless integration with the existing UI infrastructure while providing
enhanced configuration management and real-time monitoring capabilities.

Features:
- Parameter management interface integration
- Real-time monitoring dashboard components
- Configuration validation UI components
- Module status indicators and health monitoring
- Excel configuration upload and management
- Performance metrics visualization

Author: The Augster
Date: 2025-01-06
Version: 1.0.0 - UI Integration Manager
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class UIComponentConfig:
    """Configuration for UI components"""
    enable_parameter_management: bool = True
    enable_real_time_monitoring: bool = True
    enable_configuration_validation: bool = True
    enable_performance_metrics: bool = True
    enable_module_status_indicators: bool = True
    
    # UI refresh settings
    monitoring_refresh_interval_seconds: int = 30
    metrics_refresh_interval_seconds: int = 60
    status_refresh_interval_seconds: int = 10
    
    # Display settings
    max_log_entries: int = 1000
    max_performance_history: int = 100
    enable_detailed_tooltips: bool = True

@dataclass
class UIIntegrationStatus:
    """Status information for UI integration"""
    is_active: bool = False
    last_update: Optional[datetime] = None
    active_components: List[str] = field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Component status
    parameter_management_status: str = "inactive"
    monitoring_dashboard_status: str = "inactive"
    configuration_validation_status: str = "inactive"
    
    # Performance metrics
    ui_response_time: float = 0.0
    data_refresh_rate: float = 0.0

class UIIntegrationManager:
    """
    UI Integration Manager for Enhanced Market Regime System
    
    Provides seamless integration with existing UI infrastructure while
    adding enhanced configuration management and monitoring capabilities.
    """
    
    def __init__(self, config: Optional[UIComponentConfig] = None):
        """Initialize UI Integration Manager"""
        self.config = config or UIComponentConfig()
        self.status = UIIntegrationStatus()
        
        # UI component registry
        self.ui_components: Dict[str, Any] = {}
        self.component_handlers: Dict[str, callable] = {}
        
        # Data caches for UI
        self.parameter_cache: Dict[str, Any] = {}
        self.monitoring_cache: Dict[str, Any] = {}
        self.performance_cache: List[Dict[str, Any]] = []
        
        # Integration hooks
        self.engine_reference: Optional[Any] = None
        self.config_mapper_reference: Optional[Any] = None
        
        logger.info("UI Integration Manager initialized")
    
    def register_engine_reference(self, engine_instance: Any) -> bool:
        """Register reference to the unified engine"""
        try:
            self.engine_reference = engine_instance
            logger.info("Engine reference registered with UI Integration Manager")
            return True
        except Exception as e:
            logger.error(f"Error registering engine reference: {e}")
            return False
    
    def register_config_mapper_reference(self, config_mapper_instance: Any) -> bool:
        """Register reference to the configuration mapper"""
        try:
            self.config_mapper_reference = config_mapper_instance
            logger.info("Configuration mapper reference registered with UI Integration Manager")
            return True
        except Exception as e:
            logger.error(f"Error registering config mapper reference: {e}")
            return False
    
    def initialize_ui_components(self) -> bool:
        """Initialize all UI components"""
        try:
            logger.info("Initializing UI components...")
            
            success_count = 0
            total_components = 0
            
            # Initialize parameter management interface
            if self.config.enable_parameter_management:
                total_components += 1
                if self._initialize_parameter_management():
                    success_count += 1
                    self.status.active_components.append("parameter_management")
                    self.status.parameter_management_status = "active"
            
            # Initialize real-time monitoring dashboard
            if self.config.enable_real_time_monitoring:
                total_components += 1
                if self._initialize_monitoring_dashboard():
                    success_count += 1
                    self.status.active_components.append("monitoring_dashboard")
                    self.status.monitoring_dashboard_status = "active"
            
            # Initialize configuration validation UI
            if self.config.enable_configuration_validation:
                total_components += 1
                if self._initialize_configuration_validation():
                    success_count += 1
                    self.status.active_components.append("configuration_validation")
                    self.status.configuration_validation_status = "active"
            
            # Initialize performance metrics visualization
            if self.config.enable_performance_metrics:
                total_components += 1
                if self._initialize_performance_metrics():
                    success_count += 1
                    self.status.active_components.append("performance_metrics")
            
            # Initialize module status indicators
            if self.config.enable_module_status_indicators:
                total_components += 1
                if self._initialize_module_status_indicators():
                    success_count += 1
                    self.status.active_components.append("module_status_indicators")
            
            self.status.is_active = success_count > 0
            self.status.last_update = datetime.now()
            
            logger.info(f"UI components initialized: {success_count}/{total_components} successful")
            return success_count == total_components
            
        except Exception as e:
            self.status.last_error = str(e)
            self.status.error_count += 1
            logger.error(f"Error initializing UI components: {e}")
            return False
    
    def get_parameter_management_data(self) -> Dict[str, Any]:
        """Get data for parameter management interface"""
        try:
            if not self.config_mapper_reference:
                return {"error": "Configuration mapper not available"}
            
            # Get all module configurations
            all_configs = self.config_mapper_reference.get_all_module_configurations()
            
            parameter_data = {
                "timestamp": datetime.now().isoformat(),
                "total_modules": len(all_configs),
                "modules": {}
            }
            
            for module_name, module_config in all_configs.items():
                parameter_data["modules"][module_name] = {
                    "parameters": module_config.parameters.copy(),
                    "validation_errors": module_config.validation_errors.copy(),
                    "last_updated": module_config.last_updated.isoformat() if module_config.last_updated else None,
                    "parameter_count": len(module_config.parameters),
                    "error_count": len(module_config.validation_errors)
                }
            
            # Cache the data
            self.parameter_cache = parameter_data
            
            return parameter_data
            
        except Exception as e:
            logger.error(f"Error getting parameter management data: {e}")
            return {"error": str(e)}
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time monitoring dashboard"""
        try:
            monitoring_data = {
                "timestamp": datetime.now().isoformat(),
                "engine_status": {},
                "module_status": {},
                "performance_metrics": {},
                "system_health": {}
            }
            
            # Get engine status
            if self.engine_reference:
                try:
                    engine_status = self.engine_reference.get_engine_status()
                    monitoring_data["engine_status"] = engine_status
                except Exception as e:
                    monitoring_data["engine_status"] = {"error": str(e)}
            
            # Get integration manager status
            if (self.engine_reference and 
                hasattr(self.engine_reference, 'integration_manager') and 
                self.engine_reference.integration_manager):
                try:
                    integration_status = self.engine_reference.integration_manager.get_integration_status()
                    monitoring_data["module_status"] = integration_status
                except Exception as e:
                    monitoring_data["module_status"] = {"error": str(e)}
            
            # Calculate system health score
            monitoring_data["system_health"] = self._calculate_system_health(monitoring_data)
            
            # Cache the data
            self.monitoring_cache = monitoring_data
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Error getting monitoring dashboard data: {e}")
            return {"error": str(e)}
    
    def get_configuration_validation_data(self) -> Dict[str, Any]:
        """Get data for configuration validation interface"""
        try:
            if not self.config_mapper_reference:
                return {"error": "Configuration mapper not available"}
            
            # Get validation report
            validation_report = self.config_mapper_reference.validate_configuration()
            
            validation_data = {
                "timestamp": datetime.now().isoformat(),
                "validation_summary": {
                    "total_modules": validation_report.get("total_modules", 0),
                    "successful_modules": validation_report.get("successful_modules", 0),
                    "failed_modules": validation_report.get("failed_modules", 0),
                    "total_parameters": validation_report.get("total_parameters", 0),
                    "total_errors": len(validation_report.get("validation_errors", []))
                },
                "validation_details": validation_report.get("module_reports", {}),
                "validation_errors": validation_report.get("validation_errors", [])
            }
            
            return validation_data
            
        except Exception as e:
            logger.error(f"Error getting configuration validation data: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics_data(self) -> Dict[str, Any]:
        """Get data for performance metrics visualization"""
        try:
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": {},
                "historical_metrics": self.performance_cache.copy(),
                "performance_summary": {}
            }
            
            # Get current performance metrics
            if self.engine_reference:
                try:
                    engine_status = self.engine_reference.get_engine_status()
                    performance_metrics = engine_status.get("performance_metrics", {})
                    
                    performance_data["current_metrics"] = performance_metrics
                    
                    # Add to historical cache
                    historical_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "metrics": performance_metrics.copy()
                    }
                    
                    self.performance_cache.append(historical_entry)
                    
                    # Limit cache size
                    if len(self.performance_cache) > self.config.max_performance_history:
                        self.performance_cache = self.performance_cache[-self.config.max_performance_history:]
                    
                    # Calculate performance summary
                    performance_data["performance_summary"] = self._calculate_performance_summary(performance_metrics)
                    
                except Exception as e:
                    performance_data["current_metrics"] = {"error": str(e)}
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting performance metrics data: {e}")
            return {"error": str(e)}
    
    def get_module_status_data(self) -> Dict[str, Any]:
        """Get data for module status indicators"""
        try:
            status_data = {
                "timestamp": datetime.now().isoformat(),
                "module_statuses": {},
                "integration_health": {},
                "error_summary": {}
            }
            
            # Get module statuses from integration manager
            if (self.engine_reference and 
                hasattr(self.engine_reference, 'integration_manager') and 
                self.engine_reference.integration_manager):
                try:
                    integration_status = self.engine_reference.integration_manager.get_integration_status()
                    
                    status_data["module_statuses"] = integration_status.get("modules", {})
                    status_data["integration_health"] = {
                        "integration_active": integration_status.get("integration_active", False),
                        "total_modules": integration_status.get("metrics", {}).get("total_modules", 0),
                        "active_modules": integration_status.get("metrics", {}).get("active_modules", 0),
                        "error_modules": integration_status.get("metrics", {}).get("error_modules", 0)
                    }
                    
                except Exception as e:
                    status_data["module_statuses"] = {"error": str(e)}
            
            return status_data
            
        except Exception as e:
            logger.error(f"Error getting module status data: {e}")
            return {"error": str(e)}

    def _initialize_parameter_management(self) -> bool:
        """Initialize parameter management interface"""
        try:
            # Register parameter management handlers
            self.component_handlers["get_parameters"] = self.get_parameter_management_data

            logger.info("Parameter management interface initialized")
            return True

        except Exception as e:
            logger.error(f"Error initializing parameter management: {e}")
            return False

    def _initialize_monitoring_dashboard(self) -> bool:
        """Initialize real-time monitoring dashboard"""
        try:
            # Register monitoring dashboard handlers
            self.component_handlers["get_monitoring_data"] = self.get_monitoring_dashboard_data

            logger.info("Monitoring dashboard initialized")
            return True

        except Exception as e:
            logger.error(f"Error initializing monitoring dashboard: {e}")
            return False

    def _initialize_configuration_validation(self) -> bool:
        """Initialize configuration validation UI"""
        try:
            # Register validation handlers
            self.component_handlers["get_validation_data"] = self.get_configuration_validation_data

            logger.info("Configuration validation UI initialized")
            return True

        except Exception as e:
            logger.error(f"Error initializing configuration validation: {e}")
            return False

    def _initialize_performance_metrics(self) -> bool:
        """Initialize performance metrics visualization"""
        try:
            # Register performance metrics handlers
            self.component_handlers["get_performance_data"] = self.get_performance_metrics_data

            logger.info("Performance metrics visualization initialized")
            return True

        except Exception as e:
            logger.error(f"Error initializing performance metrics: {e}")
            return False

    def _initialize_module_status_indicators(self) -> bool:
        """Initialize module status indicators"""
        try:
            # Register status indicator handlers
            self.component_handlers["get_module_status"] = self.get_module_status_data

            logger.info("Module status indicators initialized")
            return True

        except Exception as e:
            logger.error(f"Error initializing module status indicators: {e}")
            return False

    def _calculate_system_health(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score"""
        try:
            health_score = 0.0
            health_factors = []

            # Engine health (30%)
            engine_status = monitoring_data.get("engine_status", {})
            if engine_status.get("is_initialized") and engine_status.get("is_active"):
                health_factors.append(("engine", 0.3, 1.0))
            else:
                health_factors.append(("engine", 0.3, 0.0))

            # Module health (40%)
            module_status = monitoring_data.get("module_status", {})
            if "metrics" in module_status:
                metrics = module_status["metrics"]
                total_modules = metrics.get("total_modules", 1)
                active_modules = metrics.get("active_modules", 0)
                module_health = active_modules / total_modules if total_modules > 0 else 0
                health_factors.append(("modules", 0.4, module_health))
            else:
                health_factors.append(("modules", 0.4, 0.0))

            # Performance health (30%)
            performance_metrics = engine_status.get("performance_metrics", {})
            if performance_metrics:
                # Check if processing time is within acceptable limits
                avg_time = performance_metrics.get("average_processing_time", 0)
                max_time = 3.0  # 3 second target
                performance_health = max(0, 1 - (avg_time / max_time)) if avg_time > 0 else 1.0
                health_factors.append(("performance", 0.3, performance_health))
            else:
                health_factors.append(("performance", 0.3, 0.5))

            # Calculate weighted health score
            for factor_name, weight, score in health_factors:
                health_score += weight * score

            return {
                "overall_health_score": health_score,
                "health_grade": self._get_health_grade(health_score),
                "health_factors": {name: {"weight": weight, "score": score} for name, weight, score in health_factors}
            }

        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {"overall_health_score": 0.0, "health_grade": "Unknown", "error": str(e)}

    def _get_health_grade(self, health_score: float) -> str:
        """Convert health score to grade"""
        if health_score >= 0.9:
            return "Excellent"
        elif health_score >= 0.8:
            return "Good"
        elif health_score >= 0.7:
            return "Fair"
        elif health_score >= 0.5:
            return "Poor"
        else:
            return "Critical"

    def _calculate_performance_summary(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance summary statistics"""
        try:
            summary = {
                "status": "unknown",
                "key_metrics": {},
                "alerts": []
            }

            # Analyze key metrics
            total_analyses = performance_metrics.get("total_analyses", 0)
            successful_analyses = performance_metrics.get("successful_analyses", 0)
            avg_processing_time = performance_metrics.get("average_processing_time", 0)

            # Calculate success rate
            success_rate = (successful_analyses / total_analyses) if total_analyses > 0 else 0

            summary["key_metrics"] = {
                "success_rate": success_rate,
                "total_analyses": total_analyses,
                "avg_processing_time": avg_processing_time
            }

            # Determine status
            if success_rate >= 0.95 and avg_processing_time <= 3.0:
                summary["status"] = "excellent"
            elif success_rate >= 0.9 and avg_processing_time <= 5.0:
                summary["status"] = "good"
            elif success_rate >= 0.8:
                summary["status"] = "fair"
            else:
                summary["status"] = "poor"

            return summary

        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {"status": "error", "error": str(e)}

    def get_ui_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive UI integration status"""
        try:
            return {
                "is_active": self.status.is_active,
                "last_update": self.status.last_update.isoformat() if self.status.last_update else None,
                "active_components": self.status.active_components.copy(),
                "error_count": self.status.error_count,
                "last_error": self.status.last_error,
                "component_status": {
                    "parameter_management": self.status.parameter_management_status,
                    "monitoring_dashboard": self.status.monitoring_dashboard_status,
                    "configuration_validation": self.status.configuration_validation_status
                },
                "available_handlers": list(self.component_handlers.keys())
            }

        except Exception as e:
            logger.error(f"Error getting UI integration status: {e}")
            return {"error": str(e)}


# Factory function for easy instantiation
def create_ui_integration_manager(config: Optional[UIComponentConfig] = None) -> UIIntegrationManager:
    """
    Factory function to create UI Integration Manager

    Args:
        config: Optional UI component configuration

    Returns:
        UIIntegrationManager: Configured UI integration manager instance
    """
    return UIIntegrationManager(config)
