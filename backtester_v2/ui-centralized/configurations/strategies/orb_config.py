"""
ORB (Opening Range Breakout) Configuration
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import time
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class ORBConfiguration(BaseConfiguration):
    """
    Configuration for Opening Range Breakout Strategy (ORB)
    
    This configuration handles all ORB-specific parameters including
    opening range definition, breakout conditions, position management,
    and time-based rules.
    """
    
    def __init__(self, strategy_name: str):
        """Initialize ORB configuration"""
        super().__init__("orb", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate ORB configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate opening range settings
        range_errors = self._validate_opening_range()
        if range_errors:
            errors['opening_range'] = range_errors
        
        # Validate breakout conditions
        breakout_errors = self._validate_breakout_conditions()
        if breakout_errors:
            errors['breakout_conditions'] = breakout_errors
        
        # Validate time settings
        time_errors = self._validate_time_settings()
        if time_errors:
            errors['time_settings'] = time_errors
        
        # Validate position management
        position_errors = self._validate_position_management()
        if position_errors:
            errors['position_management'] = position_errors
        
        if errors:
            raise ValidationError("ORB configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for ORB configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["opening_range", "breakout_conditions", "time_settings", "position_management"],
            "properties": {
                "opening_range": {
                    "type": "object",
                    "required": ["start_time", "end_time", "range_type"],
                    "properties": {
                        "start_time": {"type": "string", "format": "time"},
                        "end_time": {"type": "string", "format": "time"},
                        "range_type": {"type": "string", "enum": ["HIGH_LOW", "VWAP_BASED", "VOLUME_WEIGHTED", "CUSTOM"]},
                        "buffer_percentage": {"type": "number", "minimum": 0, "maximum": 5},
                        "minimum_range_points": {"type": "number", "minimum": 0},
                        "maximum_range_points": {"type": "number", "minimum": 0}
                    }
                },
                "breakout_conditions": {
                    "type": "object",
                    "required": ["breakout_type", "confirmation_method"],
                    "properties": {
                        "breakout_type": {"type": "string", "enum": ["RANGE_HIGH", "RANGE_LOW", "BOTH", "FIRST_BREAKOUT"]},
                        "confirmation_method": {"type": "string", "enum": ["IMMEDIATE", "CLOSE_BASED", "VOLUME_BASED", "TIME_BASED"]},
                        "confirmation_candles": {"type": "integer", "minimum": 1, "maximum": 5},
                        "volume_multiplier": {"type": "number", "minimum": 1},
                        "retest_allowed": {"type": "boolean"},
                        "false_breakout_filter": {"type": "boolean"}
                    }
                },
                "time_settings": {
                    "type": "object",
                    "required": ["entry_start_time", "entry_end_time", "exit_time"],
                    "properties": {
                        "entry_start_time": {"type": "string", "format": "time"},
                        "entry_end_time": {"type": "string", "format": "time"},
                        "exit_time": {"type": "string", "format": "time"},
                        "no_trade_zones": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string", "format": "time"},
                                    "end": {"type": "string", "format": "time"}
                                }
                            }
                        }
                    }
                },
                "position_management": {
                    "type": "object",
                    "required": ["position_type", "option_type", "quantity"],
                    "properties": {
                        "position_type": {"type": "string", "enum": ["BUY", "SELL", "DIRECTIONAL"]},
                        "option_type": {"type": "string", "enum": ["CE", "PE", "BOTH", "BREAKOUT_DIRECTION"]},
                        "strike_selection": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["ATM", "OTM", "ITM", "RANGE_BASED"]},
                                "offset": {"type": "integer"},
                                "range_based_offset": {"type": "number"}
                            }
                        },
                        "quantity": {"type": "integer", "minimum": 1},
                        "stop_loss_type": {"type": "string", "enum": ["POINTS", "PERCENTAGE", "RANGE_BASED", "TRAILING"]},
                        "stop_loss_value": {"type": "number", "minimum": 0},
                        "target_type": {"type": "string", "enum": ["POINTS", "PERCENTAGE", "RANGE_BASED", "RISK_REWARD"]},
                        "target_value": {"type": "number", "minimum": 0}
                    }
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "gap_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "min_gap_percentage": {"type": "number"},
                                "max_gap_percentage": {"type": "number"}
                            }
                        },
                        "volume_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "min_volume": {"type": "integer"},
                                "volume_sma_period": {"type": "integer"}
                            }
                        },
                        "volatility_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "min_atr": {"type": "number"},
                                "max_atr": {"type": "number"}
                            }
                        },
                        "trend_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "trend_period": {"type": "integer"},
                                "trend_direction": {"type": "string", "enum": ["BULLISH", "BEARISH", "ANY"]}
                            }
                        }
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for ORB configuration"""
        return {
            "opening_range": {
                "start_time": "09:15:00",
                "end_time": "09:30:00",
                "range_type": "HIGH_LOW",
                "buffer_percentage": 0.1,
                "minimum_range_points": 20,
                "maximum_range_points": 200,
                "include_pre_market": False,
                "range_extension_allowed": False
            },
            "breakout_conditions": {
                "breakout_type": "BOTH",
                "confirmation_method": "CLOSE_BASED",
                "confirmation_candles": 2,
                "volume_multiplier": 1.5,
                "retest_allowed": True,
                "false_breakout_filter": True,
                "breakout_threshold_points": 5,
                "momentum_confirmation": False
            },
            "time_settings": {
                "entry_start_time": "09:30:00",
                "entry_end_time": "14:30:00",
                "exit_time": "15:15:00",
                "no_trade_zones": [
                    {"start": "12:00:00", "end": "12:30:00"}
                ],
                "friday_early_exit": True,
                "friday_exit_time": "14:30:00"
            },
            "position_management": {
                "position_type": "BUY",
                "option_type": "BREAKOUT_DIRECTION",
                "strike_selection": {
                    "method": "ATM",
                    "offset": 0,
                    "range_based_offset": 0.5
                },
                "quantity": 1,
                "stop_loss_type": "RANGE_BASED",
                "stop_loss_value": 0.5,
                "target_type": "RISK_REWARD",
                "target_value": 2.0,
                "trailing_stop": {
                    "enabled": False,
                    "activation_points": 30,
                    "trail_points": 15
                },
                "partial_exit": {
                    "enabled": False,
                    "levels": [
                        {"percentage": 50, "target_multiplier": 1.0},
                        {"percentage": 50, "target_multiplier": 2.0}
                    ]
                }
            },
            "filters": {
                "gap_filter": {
                    "enabled": True,
                    "min_gap_percentage": -2.0,
                    "max_gap_percentage": 2.0,
                    "gap_direction": "ANY"
                },
                "volume_filter": {
                    "enabled": True,
                    "min_volume": 1000000,
                    "volume_sma_period": 20,
                    "volume_multiplier": 1.2
                },
                "volatility_filter": {
                    "enabled": False,
                    "min_atr": 50,
                    "max_atr": 200,
                    "atr_period": 14
                },
                "trend_filter": {
                    "enabled": False,
                    "trend_period": 20,
                    "trend_direction": "ANY",
                    "trend_strength": 0.5
                },
                "market_condition_filter": {
                    "enabled": False,
                    "avoid_trending_days": False,
                    "avoid_range_days": False
                }
            },
            "risk_management": {
                "max_trades_per_day": 2,
                "max_loss_per_day": 5000,
                "max_profit_per_day": 10000,
                "consecutive_loss_limit": 3,
                "risk_per_trade": 0.02,
                "correlation_check": False
            }
        }
    
    def _validate_opening_range(self) -> List[str]:
        """Validate opening range settings"""
        errors = []
        orb_range = self.get("opening_range", {})
        
        # Validate times
        start_time = orb_range.get("start_time")
        end_time = orb_range.get("end_time")
        
        if start_time and end_time:
            try:
                start = time.fromisoformat(start_time)
                end = time.fromisoformat(end_time)
                
                if start >= end:
                    errors.append("Opening range start time must be before end time")
                
                # Check if range is reasonable (15-60 minutes typical)
                duration_minutes = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
                if duration_minutes < 5:
                    errors.append("Opening range duration too short (minimum 5 minutes)")
                elif duration_minutes > 120:
                    errors.append("Opening range duration too long (maximum 120 minutes)")
                    
            except ValueError:
                errors.append("Invalid time format. Use HH:MM:SS")
        
        # Validate range limits
        min_range = orb_range.get("minimum_range_points", 0)
        max_range = orb_range.get("maximum_range_points", 0)
        
        if min_range < 0:
            errors.append("Minimum range points cannot be negative")
        if max_range > 0 and max_range < min_range:
            errors.append("Maximum range points must be greater than minimum")
        
        return errors
    
    def _validate_breakout_conditions(self) -> List[str]:
        """Validate breakout conditions"""
        errors = []
        breakout = self.get("breakout_conditions", {})
        
        # Validate confirmation method
        confirmation = breakout.get("confirmation_method")
        if confirmation in ["CLOSE_BASED", "TIME_BASED"]:
            candles = breakout.get("confirmation_candles", 0)
            if candles <= 0:
                errors.append(f"Confirmation candles must be positive for {confirmation} method")
        
        elif confirmation == "VOLUME_BASED":
            multiplier = breakout.get("volume_multiplier", 0)
            if multiplier <= 1:
                errors.append("Volume multiplier must be greater than 1")
        
        # Validate breakout threshold
        threshold = breakout.get("breakout_threshold_points", 0)
        if threshold < 0:
            errors.append("Breakout threshold cannot be negative")
        
        return errors
    
    def _validate_time_settings(self) -> List[str]:
        """Validate time settings"""
        errors = []
        times = self.get("time_settings", {})
        
        # Validate entry window
        entry_start = times.get("entry_start_time")
        entry_end = times.get("entry_end_time")
        exit_time = times.get("exit_time")
        
        if entry_start and entry_end and exit_time:
            try:
                start = time.fromisoformat(entry_start)
                end = time.fromisoformat(entry_end)
                exit_t = time.fromisoformat(exit_time)
                
                if start >= end:
                    errors.append("Entry start time must be before entry end time")
                if end >= exit_t:
                    errors.append("Entry end time must be before exit time")
                    
            except ValueError:
                errors.append("Invalid time format in time settings")
        
        # Validate no-trade zones
        no_trade_zones = times.get("no_trade_zones", [])
        for i, zone in enumerate(no_trade_zones):
            if "start" in zone and "end" in zone:
                try:
                    zone_start = time.fromisoformat(zone["start"])
                    zone_end = time.fromisoformat(zone["end"])
                    if zone_start >= zone_end:
                        errors.append(f"No-trade zone {i+1}: Start time must be before end time")
                except ValueError:
                    errors.append(f"No-trade zone {i+1}: Invalid time format")
        
        return errors
    
    def _validate_position_management(self) -> List[str]:
        """Validate position management settings"""
        errors = []
        position = self.get("position_management", {})
        
        # Validate quantity
        quantity = position.get("quantity", 0)
        if quantity <= 0:
            errors.append("Position quantity must be positive")
        
        # Validate stop loss
        sl_type = position.get("stop_loss_type")
        sl_value = position.get("stop_loss_value", 0)
        
        if sl_value < 0:
            errors.append("Stop loss value cannot be negative")
        
        if sl_type == "RANGE_BASED" and sl_value > 2:
            errors.append("Range-based stop loss multiplier seems too high (>2)")
        
        # Validate target
        target_type = position.get("target_type")
        target_value = position.get("target_value", 0)
        
        if target_value < 0:
            errors.append("Target value cannot be negative")
        
        if target_type == "RISK_REWARD" and target_value < 0.5:
            errors.append("Risk-reward ratio too low (minimum 0.5)")
        
        return errors
    
    def get_opening_range_times(self) -> Tuple[time, time]:
        """Get opening range start and end times"""
        orb_range = self.get("opening_range", {})
        start = time.fromisoformat(orb_range.get("start_time", "09:15:00"))
        end = time.fromisoformat(orb_range.get("end_time", "09:30:00"))
        return start, end
    
    def get_entry_window(self) -> Tuple[time, time]:
        """Get entry window times"""
        times = self.get("time_settings", {})
        start = time.fromisoformat(times.get("entry_start_time", "09:30:00"))
        end = time.fromisoformat(times.get("entry_end_time", "14:30:00"))
        return start, end
    
    def get_breakout_config(self) -> Dict[str, Any]:
        """Get breakout configuration"""
        return self.get("breakout_conditions", {})
    
    def get_active_filters(self) -> List[str]:
        """Get list of active filters"""
        filters = self.get("filters", {})
        active = []
        
        for filter_name, filter_config in filters.items():
            if isinstance(filter_config, dict) and filter_config.get("enabled", False):
                active.append(filter_name)
        
        return active
    
    def get_stop_loss_points(self, range_size: float) -> float:
        """Calculate stop loss in points based on configuration"""
        position = self.get("position_management", {})
        sl_type = position.get("stop_loss_type", "POINTS")
        sl_value = position.get("stop_loss_value", 0)
        
        if sl_type == "POINTS":
            return sl_value
        elif sl_type == "RANGE_BASED":
            return range_size * sl_value
        else:
            return sl_value  # For percentage, return as-is
    
    def get_target_points(self, range_size: float, stop_loss_points: float) -> float:
        """Calculate target in points based on configuration"""
        position = self.get("position_management", {})
        target_type = position.get("target_type", "POINTS")
        target_value = position.get("target_value", 0)
        
        if target_type == "POINTS":
            return target_value
        elif target_type == "RANGE_BASED":
            return range_size * target_value
        elif target_type == "RISK_REWARD":
            return stop_loss_points * target_value
        else:
            return target_value  # For percentage, return as-is
    
    def __str__(self) -> str:
        """String representation"""
        start, end = self.get_opening_range_times()
        return f"ORB Configuration: {self.strategy_name} (Range: {start}-{end})"