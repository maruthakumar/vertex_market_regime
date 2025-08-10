"""
TBS (Trade Builder Strategy) Configuration
"""

from typing import Dict, Any, List, Optional
from datetime import time
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class TBSConfiguration(BaseConfiguration):
    """
    Configuration for Trade Builder Strategy (TBS)
    
    This configuration handles all TBS-specific parameters including
    portfolio settings, strategy parameters, enhancement parameters,
    and risk management settings.
    """
    
    def __init__(self, strategy_name: str):
        """Initialize TBS configuration"""
        super().__init__("tbs", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate TBS configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate portfolio settings
        portfolio_errors = self._validate_portfolio_settings()
        if portfolio_errors:
            errors['portfolio_settings'] = portfolio_errors
        
        # Validate strategy parameters
        strategy_errors = self._validate_strategy_parameters()
        if strategy_errors:
            errors['strategy_parameters'] = strategy_errors
        
        # Validate enhancement parameters
        enhancement_errors = self._validate_enhancement_parameters()
        if enhancement_errors:
            errors['enhancement_parameters'] = enhancement_errors
        
        # Validate risk management
        risk_errors = self._validate_risk_management()
        if risk_errors:
            errors['risk_management'] = risk_errors
        
        if errors:
            raise ValidationError("TBS configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for TBS configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["portfolio_settings", "strategy_parameters"],
            "properties": {
                "portfolio_settings": {
                    "type": "object",
                    "required": ["capital", "strategy_type"],
                    "properties": {
                        "capital": {"type": "number", "minimum": 0},
                        "strategy_type": {"type": "string", "enum": ["TBS"]},
                        "capital_per_set": {"type": "number", "minimum": 0},
                        "max_trades": {"type": "integer", "minimum": 1},
                        "product_type": {"type": "string", "enum": ["MIS", "NRML", "CNC"]},
                        "order_type": {"type": "string", "enum": ["MARKET", "LIMIT"]}
                    }
                },
                "strategy_parameters": {
                    "type": "object",
                    "required": ["entry_time", "exit_time"],
                    "properties": {
                        "entry_time": {"type": "string", "format": "time"},
                        "exit_time": {"type": "string", "format": "time"},
                        "square_off_time": {"type": "string", "format": "time"},
                        "strike_selection_method": {
                            "type": "string",
                            "enum": ["ATM", "OTM", "ITM", "STRIKE_PRICE", "PREMIUM_BASED"]
                        },
                        "position_type": {"type": "string", "enum": ["BUY", "SELL"]},
                        "option_type": {"type": "string", "enum": ["CE", "PE", "BOTH"]},
                        "expiry_type": {"type": "string", "enum": ["WEEKLY", "MONTHLY", "NEXT_WEEKLY"]},
                        "stop_loss": {"type": "number", "minimum": 0},
                        "target": {"type": "number", "minimum": 0},
                        "trailing_stop_loss": {"type": "boolean"}
                    }
                },
                "enhancement_parameters": {
                    "type": "object",
                    "properties": {
                        "reentry_enabled": {"type": "boolean"},
                        "reentry_count": {"type": "integer", "minimum": 0},
                        "multileg_enabled": {"type": "boolean"},
                        "adjustment_enabled": {"type": "boolean"},
                        "vix_based_adjustment": {"type": "boolean"},
                        "dynamic_position_sizing": {"type": "boolean"}
                    }
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "max_loss_per_day": {"type": "number"},
                        "max_profit_per_day": {"type": "number"},
                        "max_trades_per_day": {"type": "integer", "minimum": 1},
                        "position_size_limit": {"type": "number", "minimum": 0, "maximum": 1},
                        "risk_per_trade": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for TBS configuration"""
        return {
            "portfolio_settings": {
                "capital": 100000,
                "strategy_type": "TBS",
                "capital_per_set": 25000,
                "max_trades": 4,
                "product_type": "MIS",
                "order_type": "MARKET"
            },
            "strategy_parameters": {
                "entry_time": "09:30:00",
                "exit_time": "15:00:00",
                "square_off_time": "15:15:00",
                "strike_selection_method": "ATM",
                "position_type": "SELL",
                "option_type": "BOTH",
                "expiry_type": "WEEKLY",
                "stop_loss": 30,
                "target": 50,
                "trailing_stop_loss": False,
                "strike_offset": 0,
                "lots": 1
            },
            "enhancement_parameters": {
                "reentry_enabled": False,
                "reentry_count": 2,
                "reentry_delay_minutes": 5,
                "multileg_enabled": False,
                "multileg_config": {},
                "adjustment_enabled": False,
                "adjustment_rules": {},
                "vix_based_adjustment": False,
                "vix_threshold": 20,
                "dynamic_position_sizing": False
            },
            "risk_management": {
                "max_loss_per_day": 5000,
                "max_profit_per_day": 10000,
                "max_trades_per_day": 10,
                "position_size_limit": 0.25,
                "risk_per_trade": 0.02,
                "stop_trading_on_loss": True,
                "stop_trading_on_profit": False
            }
        }
    
    def _validate_portfolio_settings(self) -> List[str]:
        """Validate portfolio settings"""
        errors = []
        portfolio = self.get("portfolio_settings", {})
        
        # Check capital
        capital = portfolio.get("capital", 0)
        if capital <= 0:
            errors.append("Capital must be greater than 0")
        
        # Check capital per set
        capital_per_set = portfolio.get("capital_per_set", 0)
        if capital_per_set > capital:
            errors.append("Capital per set cannot exceed total capital")
        
        # Check strategy type
        if portfolio.get("strategy_type") != "TBS":
            errors.append("Strategy type must be TBS")
        
        return errors
    
    def _validate_strategy_parameters(self) -> List[str]:
        """Validate strategy parameters"""
        errors = []
        params = self.get("strategy_parameters", {})
        
        # Validate times
        entry_time = params.get("entry_time")
        exit_time = params.get("exit_time")
        square_off_time = params.get("square_off_time")
        
        if entry_time and exit_time:
            try:
                entry = time.fromisoformat(entry_time)
                exit = time.fromisoformat(exit_time)
                
                if entry >= exit:
                    errors.append("Entry time must be before exit time")
                
                if square_off_time:
                    square_off = time.fromisoformat(square_off_time)
                    if square_off <= exit:
                        errors.append("Square off time must be after exit time")
                        
            except ValueError:
                errors.append("Invalid time format. Use HH:MM:SS")
        
        # Validate stop loss and target
        stop_loss = params.get("stop_loss", 0)
        target = params.get("target", 0)
        
        if stop_loss < 0:
            errors.append("Stop loss cannot be negative")
        
        if target < 0:
            errors.append("Target cannot be negative")
        
        return errors
    
    def _validate_enhancement_parameters(self) -> List[str]:
        """Validate enhancement parameters"""
        errors = []
        enhancements = self.get("enhancement_parameters", {})
        
        # Validate reentry settings
        if enhancements.get("reentry_enabled", False):
            reentry_count = enhancements.get("reentry_count", 0)
            if reentry_count <= 0:
                errors.append("Reentry count must be positive when reentry is enabled")
        
        # Validate VIX settings
        if enhancements.get("vix_based_adjustment", False):
            vix_threshold = enhancements.get("vix_threshold", 0)
            if vix_threshold <= 0:
                errors.append("VIX threshold must be positive")
        
        return errors
    
    def _validate_risk_management(self) -> List[str]:
        """Validate risk management settings"""
        errors = []
        risk = self.get("risk_management", {})
        
        # Validate max loss/profit
        max_loss = risk.get("max_loss_per_day", 0)
        max_profit = risk.get("max_profit_per_day", 0)
        
        if max_loss < 0:
            errors.append("Max loss per day cannot be negative")
        
        if max_profit < 0:
            errors.append("Max profit per day cannot be negative")
        
        # Validate position size limit
        position_limit = risk.get("position_size_limit", 0)
        if not 0 <= position_limit <= 1:
            errors.append("Position size limit must be between 0 and 1")
        
        # Validate risk per trade
        risk_per_trade = risk.get("risk_per_trade", 0)
        if not 0 <= risk_per_trade <= 1:
            errors.append("Risk per trade must be between 0 and 1")
        
        return errors
    
    def get_strike_selection_config(self) -> Dict[str, Any]:
        """Get strike selection configuration"""
        params = self.get("strategy_parameters", {})
        
        return {
            "method": params.get("strike_selection_method", "ATM"),
            "offset": params.get("strike_offset", 0),
            "option_type": params.get("option_type", "BOTH"),
            "position_type": params.get("position_type", "SELL")
        }
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk management limits"""
        risk = self.get("risk_management", {})
        portfolio = self.get("portfolio_settings", {})
        
        capital = portfolio.get("capital", 100000)
        
        return {
            "max_loss_amount": risk.get("max_loss_per_day", 5000),
            "max_profit_amount": risk.get("max_profit_per_day", 10000),
            "max_position_value": capital * risk.get("position_size_limit", 0.25),
            "risk_per_trade_amount": capital * risk.get("risk_per_trade", 0.02)
        }
    
    def is_multileg_strategy(self) -> bool:
        """Check if this is a multileg strategy"""
        return self.get("enhancement_parameters.multileg_enabled", False)
    
    def get_multileg_config(self) -> Dict[str, Any]:
        """Get multileg configuration if enabled"""
        if self.is_multileg_strategy():
            return self.get("enhancement_parameters.multileg_config", {})
        return {}
    
    def __str__(self) -> str:
        """String representation"""
        return f"TBS Configuration: {self.strategy_name} (Capital: {self.get('portfolio_settings.capital', 0)})"