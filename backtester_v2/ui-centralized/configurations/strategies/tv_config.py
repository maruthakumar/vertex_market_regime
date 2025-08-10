"""
TV (TradingView) Configuration
"""

from typing import Dict, Any, List, Optional
from datetime import time
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class TVConfiguration(BaseConfiguration):
    """
    Configuration for TradingView Strategy (TV)
    
    This configuration handles all TV-specific parameters including
    webhook settings, signal configurations, position management,
    and alert integration.
    """
    
    def __init__(self, strategy_name: str):
        """Initialize TV configuration"""
        super().__init__("tv", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate TV configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate webhook settings
        webhook_errors = self._validate_webhook_settings()
        if webhook_errors:
            errors['webhook_settings'] = webhook_errors
        
        # Validate signal configuration
        signal_errors = self._validate_signal_configuration()
        if signal_errors:
            errors['signal_configuration'] = signal_errors
        
        # Validate position management
        position_errors = self._validate_position_management()
        if position_errors:
            errors['position_management'] = position_errors
        
        # Validate alert settings
        alert_errors = self._validate_alert_settings()
        if alert_errors:
            errors['alert_settings'] = alert_errors
        
        if errors:
            raise ValidationError("TV configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for TV configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["webhook_settings", "signal_configuration", "position_management"],
            "properties": {
                "webhook_settings": {
                    "type": "object",
                    "required": ["enabled", "secret_key"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "secret_key": {"type": "string", "minLength": 8},
                        "endpoint_url": {"type": "string", "format": "uri"},
                        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                        "retry_count": {"type": "integer", "minimum": 0, "maximum": 5}
                    }
                },
                "signal_configuration": {
                    "type": "object",
                    "required": ["signal_type", "entry_conditions"],
                    "properties": {
                        "signal_type": {
                            "type": "string",
                            "enum": ["SIMPLE", "COMPLEX", "MULTI_TIMEFRAME", "INDICATOR_BASED"]
                        },
                        "entry_conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "indicator": {"type": "string"},
                                    "condition": {"type": "string", "enum": ["ABOVE", "BELOW", "CROSSES_ABOVE", "CROSSES_BELOW", "EQUALS"]},
                                    "value": {"type": "number"},
                                    "timeframe": {"type": "string"}
                                }
                            }
                        },
                        "exit_conditions": {"type": "array"},
                        "confirmation_required": {"type": "boolean"},
                        "signal_validity_minutes": {"type": "integer", "minimum": 1}
                    }
                },
                "position_management": {
                    "type": "object",
                    "required": ["position_type", "option_type"],
                    "properties": {
                        "position_type": {"type": "string", "enum": ["BUY", "SELL", "BOTH"]},
                        "option_type": {"type": "string", "enum": ["CE", "PE", "BOTH", "DYNAMIC"]},
                        "strike_selection": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["ATM", "OTM", "ITM", "SIGNAL_BASED"]},
                                "offset": {"type": "integer"},
                                "dynamic_adjustment": {"type": "boolean"}
                            }
                        },
                        "quantity_method": {"type": "string", "enum": ["FIXED", "CAPITAL_BASED", "RISK_BASED"]},
                        "lots": {"type": "integer", "minimum": 1},
                        "stop_loss": {"type": "number", "minimum": 0},
                        "target": {"type": "number", "minimum": 0}
                    }
                },
                "alert_settings": {
                    "type": "object",
                    "properties": {
                        "alert_frequency": {"type": "string", "enum": ["ONCE_PER_BAR", "ONCE_PER_BAR_CLOSE", "ONCE"]},
                        "expiration_time": {"type": "string", "format": "date-time"},
                        "alert_name": {"type": "string"},
                        "message_template": {"type": "string"}
                    }
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "max_positions": {"type": "integer", "minimum": 1},
                        "max_loss_per_signal": {"type": "number"},
                        "daily_loss_limit": {"type": "number"},
                        "position_sizing": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["FIXED", "PERCENTAGE", "KELLY", "VOLATILITY_BASED"]},
                                "value": {"type": "number"}
                            }
                        }
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for TV configuration"""
        return {
            "webhook_settings": {
                "enabled": True,
                "secret_key": "your-secret-key-here",
                "endpoint_url": "https://api.yourdomain.com/tradingview/webhook",
                "timeout_seconds": 30,
                "retry_count": 3,
                "validate_ip": True,
                "allowed_ips": ["52.89.214.238", "34.212.75.30", "54.218.53.128", "52.32.178.7"]
            },
            "signal_configuration": {
                "signal_type": "INDICATOR_BASED",
                "entry_conditions": [
                    {
                        "indicator": "RSI",
                        "condition": "BELOW",
                        "value": 30,
                        "timeframe": "5m"
                    }
                ],
                "exit_conditions": [
                    {
                        "indicator": "RSI",
                        "condition": "ABOVE",
                        "value": 70,
                        "timeframe": "5m"
                    }
                ],
                "confirmation_required": False,
                "signal_validity_minutes": 5,
                "signal_cooldown_minutes": 15
            },
            "position_management": {
                "position_type": "BUY",
                "option_type": "CE",
                "strike_selection": {
                    "method": "ATM",
                    "offset": 0,
                    "dynamic_adjustment": False,
                    "adjustment_threshold": 50
                },
                "quantity_method": "FIXED",
                "lots": 1,
                "stop_loss": 30,
                "target": 50,
                "trailing_stop": False,
                "partial_exit": {
                    "enabled": False,
                    "levels": []
                }
            },
            "alert_settings": {
                "alert_frequency": "ONCE_PER_BAR_CLOSE",
                "expiration_time": None,
                "alert_name": "TradingView Strategy Alert",
                "message_template": "{{strategy.order.action}} {{ticker}} @ {{close}}",
                "include_chart_snapshot": False
            },
            "risk_management": {
                "max_positions": 3,
                "max_loss_per_signal": 5000,
                "daily_loss_limit": 15000,
                "position_sizing": {
                    "method": "FIXED",
                    "value": 1
                },
                "correlation_check": {
                    "enabled": False,
                    "max_correlation": 0.7
                }
            },
            "advanced_features": {
                "multi_timeframe_confirmation": False,
                "volume_confirmation": False,
                "market_regime_filter": False,
                "news_filter": False,
                "vix_filter": {
                    "enabled": False,
                    "min_vix": 12,
                    "max_vix": 30
                }
            }
        }
    
    def _validate_webhook_settings(self) -> List[str]:
        """Validate webhook settings"""
        errors = []
        webhook = self.get("webhook_settings", {})
        
        if webhook.get("enabled", False):
            # Check secret key
            secret_key = webhook.get("secret_key", "")
            if len(secret_key) < 8:
                errors.append("Secret key must be at least 8 characters")
            
            # Check endpoint URL
            endpoint = webhook.get("endpoint_url", "")
            if not endpoint or not endpoint.startswith(("http://", "https://")):
                errors.append("Valid endpoint URL required when webhook is enabled")
            
            # Check timeout
            timeout = webhook.get("timeout_seconds", 30)
            if not 1 <= timeout <= 300:
                errors.append("Timeout must be between 1 and 300 seconds")
        
        return errors
    
    def _validate_signal_configuration(self) -> List[str]:
        """Validate signal configuration"""
        errors = []
        signals = self.get("signal_configuration", {})
        
        # Check entry conditions
        entry_conditions = signals.get("entry_conditions", [])
        if not entry_conditions:
            errors.append("At least one entry condition required")
        
        for i, condition in enumerate(entry_conditions):
            if not condition.get("indicator"):
                errors.append(f"Entry condition {i+1}: Indicator required")
            if not condition.get("condition"):
                errors.append(f"Entry condition {i+1}: Condition required")
            if "value" not in condition:
                errors.append(f"Entry condition {i+1}: Value required")
        
        # Check signal validity
        validity = signals.get("signal_validity_minutes", 0)
        if validity <= 0:
            errors.append("Signal validity must be positive")
        
        return errors
    
    def _validate_position_management(self) -> List[str]:
        """Validate position management settings"""
        errors = []
        position = self.get("position_management", {})
        
        # Check quantity method
        qty_method = position.get("quantity_method", "FIXED")
        if qty_method == "FIXED":
            lots = position.get("lots", 0)
            if lots <= 0:
                errors.append("Lots must be positive for fixed quantity method")
        
        # Check stop loss and target
        stop_loss = position.get("stop_loss", 0)
        target = position.get("target", 0)
        
        if stop_loss < 0:
            errors.append("Stop loss cannot be negative")
        if target < 0:
            errors.append("Target cannot be negative")
        
        # Check strike selection
        strike = position.get("strike_selection", {})
        if strike.get("method") in ["OTM", "ITM"] and not strike.get("offset"):
            errors.append("Strike offset required for OTM/ITM selection")
        
        return errors
    
    def _validate_alert_settings(self) -> List[str]:
        """Validate alert settings"""
        errors = []
        alerts = self.get("alert_settings", {})
        
        # Check message template
        template = alerts.get("message_template", "")
        if not template:
            errors.append("Alert message template required")
        
        # Check alert name
        alert_name = alerts.get("alert_name", "")
        if not alert_name:
            errors.append("Alert name required")
        
        return errors
    
    def get_webhook_config(self) -> Dict[str, Any]:
        """Get webhook configuration"""
        return self.get("webhook_settings", {})
    
    def get_signal_conditions(self, signal_type: str = "entry") -> List[Dict[str, Any]]:
        """Get signal conditions"""
        signal_config = self.get("signal_configuration", {})
        
        if signal_type == "entry":
            return signal_config.get("entry_conditions", [])
        elif signal_type == "exit":
            return signal_config.get("exit_conditions", [])
        else:
            return []
    
    def get_position_config(self) -> Dict[str, Any]:
        """Get position management configuration"""
        return self.get("position_management", {})
    
    def is_multi_timeframe(self) -> bool:
        """Check if strategy uses multiple timeframes"""
        conditions = self.get("signal_configuration.entry_conditions", [])
        timeframes = set()
        
        for condition in conditions:
            if "timeframe" in condition:
                timeframes.add(condition["timeframe"])
        
        return len(timeframes) > 1
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk management limits"""
        risk = self.get("risk_management", {})
        
        return {
            "max_positions": risk.get("max_positions", 3),
            "max_loss_per_signal": risk.get("max_loss_per_signal", 5000),
            "daily_loss_limit": risk.get("daily_loss_limit", 15000),
            "position_sizing": risk.get("position_sizing", {"method": "FIXED", "value": 1})
        }
    
    def get_alert_template(self) -> str:
        """Get alert message template"""
        return self.get("alert_settings.message_template", "{{strategy.order.action}} {{ticker}}")
    
    def __str__(self) -> str:
        """String representation"""
        return f"TV Configuration: {self.strategy_name} (Signal Type: {self.get('signal_configuration.signal_type', 'UNKNOWN')})"