"""
OI (Open Interest) Configuration
"""

from typing import Dict, Any, List, Optional
from datetime import time
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class OIConfiguration(BaseConfiguration):
    """
    Configuration for Open Interest Strategy (OI)
    
    This configuration handles all OI-specific parameters including
    OI analysis methods, COI (Change in OI) calculations, PCR analysis,
    and OI-based position management.
    """
    
    def __init__(self, strategy_name: str):
        """Initialize OI configuration"""
        super().__init__("oi", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate OI configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate OI analysis settings
        oi_errors = self._validate_oi_analysis()
        if oi_errors:
            errors['oi_analysis'] = oi_errors
        
        # Validate PCR settings
        pcr_errors = self._validate_pcr_settings()
        if pcr_errors:
            errors['pcr_settings'] = pcr_errors
        
        # Validate strike selection
        strike_errors = self._validate_strike_selection()
        if strike_errors:
            errors['strike_selection'] = strike_errors
        
        # Validate position rules
        position_errors = self._validate_position_rules()
        if position_errors:
            errors['position_rules'] = position_errors
        
        if errors:
            raise ValidationError("OI configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for OI configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["oi_analysis", "pcr_settings", "strike_selection", "position_rules"],
            "properties": {
                "oi_analysis": {
                    "type": "object",
                    "required": ["analysis_method", "timeframe", "threshold_settings"],
                    "properties": {
                        "analysis_method": {
                            "type": "string",
                            "enum": ["MAX_OI", "MAX_COI", "OI_BUILDUP", "COMBINED", "WEIGHTED"]
                        },
                        "timeframe": {"type": "string", "enum": ["1min", "3min", "5min", "15min", "30min"]},
                        "lookback_periods": {"type": "integer", "minimum": 1, "maximum": 100},
                        "threshold_settings": {
                            "type": "object",
                            "properties": {
                                "min_oi": {"type": "integer", "minimum": 0},
                                "min_coi_percentage": {"type": "number", "minimum": 0},
                                "significant_level_multiplier": {"type": "number", "minimum": 1}
                            }
                        },
                        "weighted_factors": {
                            "type": "object",
                            "properties": {
                                "oi_weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "coi_weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "volume_weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "price_weight": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    }
                },
                "pcr_settings": {
                    "type": "object",
                    "required": ["pcr_type", "calculation_method"],
                    "properties": {
                        "pcr_type": {"type": "string", "enum": ["OI_PCR", "VOLUME_PCR", "COMBINED_PCR"]},
                        "calculation_method": {"type": "string", "enum": ["TOTAL", "ATM_ONLY", "RANGE_BASED", "WEIGHTED"]},
                        "pcr_thresholds": {
                            "type": "object",
                            "properties": {
                                "oversold": {"type": "number", "minimum": 0},
                                "neutral_low": {"type": "number", "minimum": 0},
                                "neutral_high": {"type": "number", "minimum": 0},
                                "overbought": {"type": "number", "minimum": 0}
                            }
                        },
                        "strike_range": {"type": "integer", "minimum": 1, "maximum": 20},
                        "smoothing_period": {"type": "integer", "minimum": 1}
                    }
                },
                "strike_selection": {
                    "type": "object",
                    "required": ["primary_method", "strike_range"],
                    "properties": {
                        "primary_method": {
                            "type": "string",
                            "enum": ["MAX_OI", "MAX_COI", "SUPPORT_RESISTANCE", "PCR_BASED", "DYNAMIC"]
                        },
                        "secondary_method": {"type": "string", "enum": ["MAX_OI", "MAX_COI", "VOLUME_BASED", "NONE"]},
                        "strike_range": {"type": "integer", "minimum": 3, "maximum": 20},
                        "dynamic_adjustment": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "adjustment_frequency": {"type": "string", "enum": ["1min", "5min", "15min", "30min"]},
                                "threshold_percentage": {"type": "number", "minimum": 0}
                            }
                        },
                        "support_resistance_levels": {
                            "type": "object",
                            "properties": {
                                "lookback_days": {"type": "integer", "minimum": 1},
                                "level_strength": {"type": "string", "enum": ["STRONG", "MEDIUM", "WEAK", "ALL"]}
                            }
                        }
                    }
                },
                "position_rules": {
                    "type": "object",
                    "required": ["entry_conditions", "exit_conditions"],
                    "properties": {
                        "entry_conditions": {
                            "type": "object",
                            "properties": {
                                "oi_buildup_type": {"type": "string", "enum": ["LONG", "SHORT", "BOTH", "DYNAMIC"]},
                                "min_oi_change": {"type": "number", "minimum": 0},
                                "pcr_condition": {"type": "string", "enum": ["ABOVE", "BELOW", "BETWEEN", "ANY"]},
                                "pcr_value": {"type": "number", "minimum": 0},
                                "confirmation_required": {"type": "boolean"},
                                "entry_time_start": {"type": "string", "format": "time"},
                                "entry_time_end": {"type": "string", "format": "time"}
                            }
                        },
                        "exit_conditions": {
                            "type": "object",
                            "properties": {
                                "oi_reversal": {"type": "boolean"},
                                "pcr_reversal": {"type": "boolean"},
                                "time_based_exit": {"type": "string", "format": "time"},
                                "stop_loss_percentage": {"type": "number", "minimum": 0},
                                "target_percentage": {"type": "number", "minimum": 0},
                                "trailing_stop": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "activation_percentage": {"type": "number"},
                                        "trail_percentage": {"type": "number"}
                                    }
                                }
                            }
                        },
                        "position_sizing": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["FIXED", "OI_BASED", "PCR_BASED", "VOLATILITY_BASED"]},
                                "base_lots": {"type": "integer", "minimum": 1},
                                "max_lots": {"type": "integer", "minimum": 1},
                                "oi_multiplier": {"type": "number", "minimum": 0}
                            }
                        }
                    }
                },
                "advanced_settings": {
                    "type": "object",
                    "properties": {
                        "iv_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "min_iv_percentile": {"type": "number", "minimum": 0, "maximum": 100},
                                "max_iv_percentile": {"type": "number", "minimum": 0, "maximum": 100}
                            }
                        },
                        "greek_filters": {
                            "type": "object",
                            "properties": {
                                "delta_filter": {"type": "boolean"},
                                "gamma_filter": {"type": "boolean"},
                                "min_delta": {"type": "number"},
                                "max_gamma": {"type": "number"}
                            }
                        },
                        "market_hours_only": {"type": "boolean"},
                        "exclude_expiry_day": {"type": "boolean"},
                        "exclude_holidays": {"type": "boolean"}
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for OI configuration"""
        return {
            "oi_analysis": {
                "analysis_method": "COMBINED",
                "timeframe": "5min",
                "lookback_periods": 20,
                "threshold_settings": {
                    "min_oi": 10000,
                    "min_coi_percentage": 5.0,
                    "significant_level_multiplier": 1.5,
                    "noise_filter": True
                },
                "weighted_factors": {
                    "oi_weight": 0.4,
                    "coi_weight": 0.3,
                    "volume_weight": 0.2,
                    "price_weight": 0.1
                },
                "normalization": {
                    "enabled": True,
                    "method": "Z_SCORE",
                    "window": 20
                }
            },
            "pcr_settings": {
                "pcr_type": "OI_PCR",
                "calculation_method": "RANGE_BASED",
                "pcr_thresholds": {
                    "oversold": 0.5,
                    "neutral_low": 0.7,
                    "neutral_high": 1.3,
                    "overbought": 1.5
                },
                "strike_range": 5,
                "smoothing_period": 3,
                "use_ema": False,
                "intraday_reset": True
            },
            "strike_selection": {
                "primary_method": "MAX_OI",
                "secondary_method": "MAX_COI",
                "strike_range": 10,
                "dynamic_adjustment": {
                    "enabled": True,
                    "adjustment_frequency": "5min",
                    "threshold_percentage": 2.0,
                    "max_adjustments_per_day": 3
                },
                "support_resistance_levels": {
                    "lookback_days": 5,
                    "level_strength": "STRONG",
                    "confluence_required": True
                },
                "strike_filtering": {
                    "min_oi": 5000,
                    "min_volume": 100,
                    "exclude_deep_otm": True,
                    "otm_threshold": 5
                }
            },
            "position_rules": {
                "entry_conditions": {
                    "oi_buildup_type": "BOTH",
                    "min_oi_change": 10.0,
                    "pcr_condition": "BETWEEN",
                    "pcr_value": 0.7,
                    "pcr_range": [0.7, 1.3],
                    "confirmation_required": True,
                    "entry_time_start": "09:30:00",
                    "entry_time_end": "14:30:00",
                    "min_time_after_open": 15
                },
                "exit_conditions": {
                    "oi_reversal": True,
                    "oi_reversal_threshold": 5.0,
                    "pcr_reversal": True,
                    "pcr_reversal_threshold": 0.2,
                    "time_based_exit": "15:15:00",
                    "stop_loss_percentage": 30,
                    "target_percentage": 50,
                    "trailing_stop": {
                        "enabled": False,
                        "activation_percentage": 30,
                        "trail_percentage": 15
                    },
                    "partial_exit": {
                        "enabled": False,
                        "levels": [
                            {"percentage": 50, "profit": 30}
                        ]
                    }
                },
                "position_sizing": {
                    "method": "OI_BASED",
                    "base_lots": 1,
                    "max_lots": 5,
                    "oi_multiplier": 0.0001,
                    "pcr_adjustment": True,
                    "volatility_adjustment": False
                }
            },
            "advanced_settings": {
                "iv_filter": {
                    "enabled": False,
                    "min_iv_percentile": 20,
                    "max_iv_percentile": 80,
                    "iv_lookback_days": 30
                },
                "greek_filters": {
                    "delta_filter": False,
                    "gamma_filter": False,
                    "min_delta": 0.2,
                    "max_gamma": 0.1,
                    "theta_consideration": False
                },
                "market_hours_only": True,
                "exclude_expiry_day": True,
                "exclude_holidays": True,
                "special_events_handling": {
                    "enabled": False,
                    "events": ["RBI_POLICY", "BUDGET", "EARNINGS"]
                }
            },
            "risk_management": {
                "max_positions": 3,
                "max_loss_per_day": 10000,
                "max_loss_per_position": 5000,
                "correlation_limit": 0.7,
                "portfolio_heat": 0.1
            }
        }
    
    def _validate_oi_analysis(self) -> List[str]:
        """Validate OI analysis settings"""
        errors = []
        oi_analysis = self.get("oi_analysis", {})
        
        # Validate weighted factors
        factors = oi_analysis.get("weighted_factors", {})
        if factors:
            total_weight = sum(factors.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point differences
                errors.append(f"Weighted factors must sum to 1.0 (current: {total_weight})")
        
        # Validate thresholds
        thresholds = oi_analysis.get("threshold_settings", {})
        min_oi = thresholds.get("min_oi", 0)
        if min_oi < 0:
            errors.append("Minimum OI cannot be negative")
        
        min_coi = thresholds.get("min_coi_percentage", 0)
        if min_coi < 0:
            errors.append("Minimum COI percentage cannot be negative")
        
        return errors
    
    def _validate_pcr_settings(self) -> List[str]:
        """Validate PCR settings"""
        errors = []
        pcr = self.get("pcr_settings", {})
        
        # Validate PCR thresholds
        thresholds = pcr.get("pcr_thresholds", {})
        if thresholds:
            oversold = thresholds.get("oversold", 0)
            neutral_low = thresholds.get("neutral_low", 0)
            neutral_high = thresholds.get("neutral_high", 0)
            overbought = thresholds.get("overbought", 0)
            
            if not (0 < oversold < neutral_low < neutral_high < overbought):
                errors.append("PCR thresholds must be in ascending order: oversold < neutral_low < neutral_high < overbought")
        
        # Validate strike range
        strike_range = pcr.get("strike_range", 0)
        if strike_range < 1:
            errors.append("PCR strike range must be at least 1")
        
        return errors
    
    def _validate_strike_selection(self) -> List[str]:
        """Validate strike selection settings"""
        errors = []
        strike = self.get("strike_selection", {})
        
        # Validate strike range
        strike_range = strike.get("strike_range", 0)
        if strike_range < 3:
            errors.append("Strike range must be at least 3 for proper analysis")
        
        # Validate dynamic adjustment
        dynamic = strike.get("dynamic_adjustment", {})
        if dynamic.get("enabled", False):
            threshold = dynamic.get("threshold_percentage", 0)
            if threshold <= 0:
                errors.append("Dynamic adjustment threshold must be positive")
        
        return errors
    
    def _validate_position_rules(self) -> List[str]:
        """Validate position rules"""
        errors = []
        rules = self.get("position_rules", {})
        
        # Validate entry conditions
        entry = rules.get("entry_conditions", {})
        min_oi_change = entry.get("min_oi_change", 0)
        if min_oi_change < 0:
            errors.append("Minimum OI change cannot be negative")
        
        # Validate PCR condition
        pcr_condition = entry.get("pcr_condition")
        if pcr_condition == "BETWEEN":
            pcr_range = entry.get("pcr_range", [])
            if len(pcr_range) != 2 or pcr_range[0] >= pcr_range[1]:
                errors.append("PCR range must have two values with min < max")
        
        # Validate position sizing
        sizing = rules.get("position_sizing", {})
        base_lots = sizing.get("base_lots", 0)
        max_lots = sizing.get("max_lots", 0)
        
        if base_lots <= 0:
            errors.append("Base lots must be positive")
        if max_lots < base_lots:
            errors.append("Max lots must be greater than or equal to base lots")
        
        return errors
    
    def get_oi_analysis_config(self) -> Dict[str, Any]:
        """Get OI analysis configuration"""
        return self.get("oi_analysis", {})
    
    def get_pcr_config(self) -> Dict[str, Any]:
        """Get PCR configuration"""
        return self.get("pcr_settings", {})
    
    def get_strike_selection_method(self) -> str:
        """Get primary strike selection method"""
        return self.get("strike_selection.primary_method", "MAX_OI")
    
    def get_position_signal(self, oi_data: Dict[str, Any]) -> Optional[str]:
        """
        Determine position signal based on OI data
        
        Args:
            oi_data: Dictionary containing OI metrics
            
        Returns:
            'LONG', 'SHORT', or None
        """
        rules = self.get("position_rules.entry_conditions", {})
        buildup_type = rules.get("oi_buildup_type", "BOTH")
        
        # Check OI change threshold
        min_change = rules.get("min_oi_change", 0)
        if abs(oi_data.get("coi_percentage", 0)) < min_change:
            return None
        
        # Determine signal based on buildup type
        if buildup_type == "LONG" and oi_data.get("long_buildup", False):
            return "LONG"
        elif buildup_type == "SHORT" and oi_data.get("short_buildup", False):
            return "SHORT"
        elif buildup_type == "BOTH":
            if oi_data.get("long_buildup", False):
                return "LONG"
            elif oi_data.get("short_buildup", False):
                return "SHORT"
        
        return None
    
    def calculate_position_size(self, oi_value: float, base_capital: float) -> int:
        """
        Calculate position size based on OI
        
        Args:
            oi_value: Current OI value
            base_capital: Base capital for position sizing
            
        Returns:
            Number of lots
        """
        sizing = self.get("position_rules.position_sizing", {})
        method = sizing.get("method", "FIXED")
        base_lots = sizing.get("base_lots", 1)
        max_lots = sizing.get("max_lots", 5)
        
        if method == "FIXED":
            return base_lots
        elif method == "OI_BASED":
            multiplier = sizing.get("oi_multiplier", 0.0001)
            calculated_lots = int(base_lots + (oi_value * multiplier))
            return min(calculated_lots, max_lots)
        
        return base_lots
    
    def __str__(self) -> str:
        """String representation"""
        return f"OI Configuration: {self.strategy_name} (Method: {self.get('oi_analysis.analysis_method', 'UNKNOWN')})"