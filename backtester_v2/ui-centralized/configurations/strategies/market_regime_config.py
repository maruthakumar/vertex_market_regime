"""
Market Regime Configuration
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import time, datetime
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class MarketRegimeConfiguration(BaseConfiguration):
    """
    Configuration for Market Regime Strategy
    
    This configuration handles the most sophisticated strategy with
    18-regime classification system using Volatility×Trend×Structure
    architecture. It includes enhanced 12-regime detector, comprehensive
    Excel-based configuration, and advanced market microstructure analysis.
    """
    
    def __init__(self, strategy_name: str):
        """Initialize Market Regime configuration"""
        super().__init__("market_regime", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate Market Regime configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate regime detection
        regime_errors = self._validate_regime_detection()
        if regime_errors:
            errors['regime_detection'] = regime_errors
        
        # Validate regime parameters
        param_errors = self._validate_regime_parameters()
        if param_errors:
            errors['regime_parameters'] = param_errors
        
        # Validate strategy mapping
        strategy_errors = self._validate_strategy_mapping()
        if strategy_errors:
            errors['strategy_mapping'] = strategy_errors
        
        # Validate transition rules
        transition_errors = self._validate_transition_rules()
        if transition_errors:
            errors['transition_rules'] = transition_errors
        
        if errors:
            raise ValidationError("Market Regime configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Market Regime configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["regime_detection", "regime_parameters", "strategy_mapping", "transition_rules"],
            "properties": {
                "regime_detection": {
                    "type": "object",
                    "required": ["classification_method", "detection_frequency", "lookback_periods"],
                    "properties": {
                        "classification_method": {
                            "type": "string",
                            "enum": ["VOLATILITY_TREND_STRUCTURE", "ENHANCED_12_REGIME", "ML_BASED", "HYBRID"]
                        },
                        "detection_frequency": {
                            "type": "string",
                            "enum": ["REALTIME", "1min", "5min", "15min", "30min", "DAILY"]
                        },
                        "lookback_periods": {
                            "type": "object",
                            "properties": {
                                "short": {"type": "integer", "minimum": 5, "maximum": 50},
                                "medium": {"type": "integer", "minimum": 20, "maximum": 100},
                                "long": {"type": "integer", "minimum": 50, "maximum": 500}
                            }
                        },
                        "volatility_calculation": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["STANDARD_DEV", "ATR", "GARCH", "EWMA", "PARKINSON"]},
                                "annualization_factor": {"type": "number", "minimum": 1},
                                "outlier_treatment": {"type": "string", "enum": ["WINSORIZE", "TRIM", "NONE"]}
                            }
                        },
                        "trend_identification": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["SMA", "EMA", "LINEAR_REGRESSION", "HODRICK_PRESCOTT"]},
                                "strength_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                "confirmation_periods": {"type": "integer", "minimum": 1}
                            }
                        },
                        "market_structure": {
                            "type": "object",
                            "properties": {
                                "metrics": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["HURST_EXPONENT", "FRACTAL_DIMENSION", "ENTROPY", "CORRELATION_STRUCTURE"]
                                    }
                                },
                                "microstructure_features": {"type": "boolean"}
                            }
                        }
                    }
                },
                "regime_parameters": {
                    "type": "object",
                    "required": ["regime_definitions", "thresholds"],
                    "properties": {
                        "regime_definitions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "regime_id": {"type": "integer", "minimum": 1, "maximum": 18},
                                    "name": {"type": "string"},
                                    "volatility_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                                    "trend_direction": {"type": "string", "enum": ["BULLISH", "NEUTRAL", "BEARISH"]},
                                    "structure_type": {"type": "string", "enum": ["TRENDING", "RANGING", "TRANSITIONAL"]},
                                    "characteristics": {
                                        "type": "object",
                                        "properties": {
                                            "vol_percentile_min": {"type": "number", "minimum": 0, "maximum": 100},
                                            "vol_percentile_max": {"type": "number", "minimum": 0, "maximum": 100},
                                            "trend_strength_min": {"type": "number", "minimum": -1, "maximum": 1},
                                            "trend_strength_max": {"type": "number", "minimum": -1, "maximum": 1}
                                        }
                                    }
                                }
                            }
                        },
                        "thresholds": {
                            "type": "object",
                            "properties": {
                                "volatility_breakpoints": {"type": "array", "items": {"type": "number"}},
                                "trend_breakpoints": {"type": "array", "items": {"type": "number"}},
                                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        },
                        "triple_straddle_analysis": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "rolling_windows": {"type": "array", "items": {"type": "integer"}},
                                "straddle_strikes": {"type": "array", "items": {"type": "string"}},
                                "weight_scheme": {"type": "string", "enum": ["EQUAL", "VOLATILITY_WEIGHTED", "TIME_DECAY_WEIGHTED"]}
                            }
                        }
                    }
                },
                "strategy_mapping": {
                    "type": "object",
                    "required": ["regime_strategies"],
                    "properties": {
                        "regime_strategies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "regime_id": {"type": "integer"},
                                    "primary_strategy": {"type": "string", "enum": ["TREND_FOLLOWING", "MEAN_REVERSION", "VOLATILITY_ARBITRAGE", "MARKET_NEUTRAL", "PROTECTIVE"]},
                                    "position_bias": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL", "DYNAMIC"]},
                                    "option_strategy": {"type": "string", "enum": ["NAKED", "SPREAD", "STRADDLE", "STRANGLE", "BUTTERFLY", "CONDOR", "RATIO"]},
                                    "trade_frequency": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW", "VERY_LOW"]},
                                    "risk_parameters": {
                                        "type": "object",
                                        "properties": {
                                            "position_size_multiplier": {"type": "number", "minimum": 0.1, "maximum": 3},
                                            "stop_loss_multiplier": {"type": "number", "minimum": 0.5, "maximum": 2},
                                            "target_multiplier": {"type": "number", "minimum": 0.5, "maximum": 5}
                                        }
                                    }
                                }
                            }
                        },
                        "dynamic_adjustment": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "adjustment_factors": {
                                    "type": "object",
                                    "properties": {
                                        "volatility_scaling": {"type": "boolean"},
                                        "trend_strength_scaling": {"type": "boolean"},
                                        "regime_confidence_scaling": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                },
                "transition_rules": {
                    "type": "object",
                    "required": ["confirmation_required", "transition_handling"],
                    "properties": {
                        "confirmation_required": {"type": "boolean"},
                        "confirmation_periods": {"type": "integer", "minimum": 1, "maximum": 10},
                        "transition_handling": {
                            "type": "object",
                            "properties": {
                                "position_adjustment": {"type": "string", "enum": ["IMMEDIATE", "GRADUAL", "AT_EXPIRY", "CONDITIONAL"]},
                                "hedge_during_transition": {"type": "boolean"},
                                "reduce_size_during_transition": {"type": "boolean"},
                                "transition_buffer_percentage": {"type": "number", "minimum": 0, "maximum": 50}
                            }
                        },
                        "regime_persistence": {
                            "type": "object",
                            "properties": {
                                "min_regime_duration": {"type": "integer", "minimum": 1},
                                "regime_stability_check": {"type": "boolean"},
                                "false_signal_filter": {"type": "boolean"}
                            }
                        }
                    }
                },
                "advanced_features": {
                    "type": "object",
                    "properties": {
                        "ml_enhancement": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "model_type": {"type": "string", "enum": ["RANDOM_FOREST", "XG_BOOST", "NEURAL_NETWORK", "ENSEMBLE"]},
                                "feature_importance_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                "retraining_frequency": {"type": "string", "enum": ["DAILY", "WEEKLY", "MONTHLY"]}
                            }
                        },
                        "cross_asset_signals": {
                            "type": "object",
                            "properties": {
                                "use_vix": {"type": "boolean"},
                                "use_dollar_index": {"type": "boolean"},
                                "use_bond_yields": {"type": "boolean"},
                                "use_commodity_indices": {"type": "boolean"}
                            }
                        },
                        "sentiment_integration": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "sources": {"type": "array", "items": {"type": "string"}},
                                "weight_in_regime_detection": {"type": "number", "minimum": 0, "maximum": 0.5}
                            }
                        }
                    }
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "regime_specific_limits": {
                            "type": "object",
                            "properties": {
                                "high_volatility_exposure_limit": {"type": "number", "minimum": 0, "maximum": 1},
                                "crisis_regime_exposure_limit": {"type": "number", "minimum": 0, "maximum": 0.5},
                                "max_regime_concentration": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        },
                        "portfolio_protection": {
                            "type": "object",
                            "properties": {
                                "tail_hedge_allocation": {"type": "number", "minimum": 0, "maximum": 0.2},
                                "crisis_regime_protection": {"type": "string", "enum": ["PUT_OPTIONS", "VIX_CALLS", "CASH", "MIXED"]},
                                "drawdown_circuit_breaker": {"type": "number", "minimum": 0, "maximum": 0.5}
                            }
                        }
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for Market Regime configuration"""
        return {
            "regime_detection": {
                "classification_method": "VOLATILITY_TREND_STRUCTURE",
                "detection_frequency": "5min",
                "lookback_periods": {
                    "short": 20,
                    "medium": 50,
                    "long": 200
                },
                "volatility_calculation": {
                    "method": "EWMA",
                    "annualization_factor": 252,
                    "outlier_treatment": "WINSORIZE",
                    "decay_factor": 0.94,
                    "min_periods": 20
                },
                "trend_identification": {
                    "method": "EMA",
                    "strength_threshold": 0.3,
                    "confirmation_periods": 3,
                    "slope_calculation_periods": 10,
                    "detrending_method": "LINEAR"
                },
                "market_structure": {
                    "metrics": ["HURST_EXPONENT", "FRACTAL_DIMENSION", "ENTROPY"],
                    "microstructure_features": True,
                    "order_flow_imbalance": True,
                    "tick_data_analysis": False
                },
                "regime_smoothing": {
                    "apply_smoothing": True,
                    "smoothing_window": 3,
                    "min_confidence": 0.6
                }
            },
            "regime_parameters": {
                "regime_definitions": [
                    # Low Volatility Regimes (1-6)
                    {
                        "regime_id": 1,
                        "name": "Low Vol Bullish Trending",
                        "volatility_level": "LOW",
                        "trend_direction": "BULLISH",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 0,
                            "vol_percentile_max": 33,
                            "trend_strength_min": 0.5,
                            "trend_strength_max": 1.0,
                            "hurst_min": 0.6,
                            "hurst_max": 1.0
                        }
                    },
                    {
                        "regime_id": 2,
                        "name": "Low Vol Neutral Trending",
                        "volatility_level": "LOW",
                        "trend_direction": "NEUTRAL",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 0,
                            "vol_percentile_max": 33,
                            "trend_strength_min": -0.2,
                            "trend_strength_max": 0.2,
                            "hurst_min": 0.5,
                            "hurst_max": 0.6
                        }
                    },
                    {
                        "regime_id": 3,
                        "name": "Low Vol Bearish Trending",
                        "volatility_level": "LOW",
                        "trend_direction": "BEARISH",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 0,
                            "vol_percentile_max": 33,
                            "trend_strength_min": -1.0,
                            "trend_strength_max": -0.5,
                            "hurst_min": 0.6,
                            "hurst_max": 1.0
                        }
                    },
                    {
                        "regime_id": 4,
                        "name": "Low Vol Bullish Ranging",
                        "volatility_level": "LOW",
                        "trend_direction": "BULLISH",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 0,
                            "vol_percentile_max": 33,
                            "trend_strength_min": 0.2,
                            "trend_strength_max": 0.5,
                            "hurst_min": 0.3,
                            "hurst_max": 0.5
                        }
                    },
                    {
                        "regime_id": 5,
                        "name": "Low Vol Neutral Ranging",
                        "volatility_level": "LOW",
                        "trend_direction": "NEUTRAL",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 0,
                            "vol_percentile_max": 33,
                            "trend_strength_min": -0.2,
                            "trend_strength_max": 0.2,
                            "hurst_min": 0.4,
                            "hurst_max": 0.6
                        }
                    },
                    {
                        "regime_id": 6,
                        "name": "Low Vol Bearish Ranging",
                        "volatility_level": "LOW",
                        "trend_direction": "BEARISH",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 0,
                            "vol_percentile_max": 33,
                            "trend_strength_min": -0.5,
                            "trend_strength_max": -0.2,
                            "hurst_min": 0.3,
                            "hurst_max": 0.5
                        }
                    },
                    # Medium Volatility Regimes (7-12)
                    {
                        "regime_id": 7,
                        "name": "Medium Vol Bullish Trending",
                        "volatility_level": "MEDIUM",
                        "trend_direction": "BULLISH",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 33,
                            "vol_percentile_max": 67,
                            "trend_strength_min": 0.5,
                            "trend_strength_max": 1.0,
                            "hurst_min": 0.6,
                            "hurst_max": 1.0
                        }
                    },
                    {
                        "regime_id": 8,
                        "name": "Medium Vol Neutral Trending",
                        "volatility_level": "MEDIUM",
                        "trend_direction": "NEUTRAL",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 33,
                            "vol_percentile_max": 67,
                            "trend_strength_min": -0.2,
                            "trend_strength_max": 0.2,
                            "hurst_min": 0.5,
                            "hurst_max": 0.6
                        }
                    },
                    {
                        "regime_id": 9,
                        "name": "Medium Vol Bearish Trending",
                        "volatility_level": "MEDIUM",
                        "trend_direction": "BEARISH",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 33,
                            "vol_percentile_max": 67,
                            "trend_strength_min": -1.0,
                            "trend_strength_max": -0.5,
                            "hurst_min": 0.6,
                            "hurst_max": 1.0
                        }
                    },
                    {
                        "regime_id": 10,
                        "name": "Medium Vol Bullish Ranging",
                        "volatility_level": "MEDIUM",
                        "trend_direction": "BULLISH",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 33,
                            "vol_percentile_max": 67,
                            "trend_strength_min": 0.2,
                            "trend_strength_max": 0.5,
                            "hurst_min": 0.3,
                            "hurst_max": 0.5
                        }
                    },
                    {
                        "regime_id": 11,
                        "name": "Medium Vol Neutral Ranging",
                        "volatility_level": "MEDIUM",
                        "trend_direction": "NEUTRAL",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 33,
                            "vol_percentile_max": 67,
                            "trend_strength_min": -0.2,
                            "trend_strength_max": 0.2,
                            "hurst_min": 0.4,
                            "hurst_max": 0.6
                        }
                    },
                    {
                        "regime_id": 12,
                        "name": "Medium Vol Bearish Ranging",
                        "volatility_level": "MEDIUM",
                        "trend_direction": "BEARISH",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 33,
                            "vol_percentile_max": 67,
                            "trend_strength_min": -0.5,
                            "trend_strength_max": -0.2,
                            "hurst_min": 0.3,
                            "hurst_max": 0.5
                        }
                    },
                    # High Volatility Regimes (13-18)
                    {
                        "regime_id": 13,
                        "name": "High Vol Bullish Trending",
                        "volatility_level": "HIGH",
                        "trend_direction": "BULLISH",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 67,
                            "vol_percentile_max": 100,
                            "trend_strength_min": 0.5,
                            "trend_strength_max": 1.0,
                            "hurst_min": 0.6,
                            "hurst_max": 1.0
                        }
                    },
                    {
                        "regime_id": 14,
                        "name": "High Vol Neutral Trending",
                        "volatility_level": "HIGH",
                        "trend_direction": "NEUTRAL",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 67,
                            "vol_percentile_max": 100,
                            "trend_strength_min": -0.2,
                            "trend_strength_max": 0.2,
                            "hurst_min": 0.5,
                            "hurst_max": 0.6
                        }
                    },
                    {
                        "regime_id": 15,
                        "name": "High Vol Bearish Trending",
                        "volatility_level": "HIGH",
                        "trend_direction": "BEARISH",
                        "structure_type": "TRENDING",
                        "characteristics": {
                            "vol_percentile_min": 67,
                            "vol_percentile_max": 100,
                            "trend_strength_min": -1.0,
                            "trend_strength_max": -0.5,
                            "hurst_min": 0.6,
                            "hurst_max": 1.0
                        }
                    },
                    {
                        "regime_id": 16,
                        "name": "High Vol Bullish Ranging",
                        "volatility_level": "HIGH",
                        "trend_direction": "BULLISH",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 67,
                            "vol_percentile_max": 100,
                            "trend_strength_min": 0.2,
                            "trend_strength_max": 0.5,
                            "hurst_min": 0.3,
                            "hurst_max": 0.5
                        }
                    },
                    {
                        "regime_id": 17,
                        "name": "High Vol Neutral Ranging",
                        "volatility_level": "HIGH",
                        "trend_direction": "NEUTRAL",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 67,
                            "vol_percentile_max": 100,
                            "trend_strength_min": -0.2,
                            "trend_strength_max": 0.2,
                            "hurst_min": 0.4,
                            "hurst_max": 0.6
                        }
                    },
                    {
                        "regime_id": 18,
                        "name": "High Vol Bearish Ranging (Crisis)",
                        "volatility_level": "HIGH",
                        "trend_direction": "BEARISH",
                        "structure_type": "RANGING",
                        "characteristics": {
                            "vol_percentile_min": 67,
                            "vol_percentile_max": 100,
                            "trend_strength_min": -0.5,
                            "trend_strength_max": -0.2,
                            "hurst_min": 0.3,
                            "hurst_max": 0.5
                        }
                    }
                ],
                "thresholds": {
                    "volatility_breakpoints": [33, 67],
                    "trend_breakpoints": [-0.3, 0.3],
                    "confidence_threshold": 0.7,
                    "regime_change_sensitivity": 0.8
                },
                "triple_straddle_analysis": {
                    "enabled": True,
                    "rolling_windows": [3, 5, 10, 15],
                    "straddle_strikes": ["ATM", "ATM-1", "ATM+1"],
                    "weight_scheme": "TIME_DECAY_WEIGHTED",
                    "decay_factor": 0.9,
                    "min_option_volume": 100
                }
            },
            "strategy_mapping": {
                "regime_strategies": [
                    # Low Volatility Strategies
                    {
                        "regime_id": 1,
                        "primary_strategy": "TREND_FOLLOWING",
                        "position_bias": "LONG",
                        "option_strategy": "NAKED",
                        "trade_frequency": "HIGH",
                        "risk_parameters": {
                            "position_size_multiplier": 1.5,
                            "stop_loss_multiplier": 0.8,
                            "target_multiplier": 2.0
                        }
                    },
                    {
                        "regime_id": 2,
                        "primary_strategy": "MEAN_REVERSION",
                        "position_bias": "NEUTRAL",
                        "option_strategy": "SPREAD",
                        "trade_frequency": "MEDIUM",
                        "risk_parameters": {
                            "position_size_multiplier": 1.0,
                            "stop_loss_multiplier": 1.0,
                            "target_multiplier": 1.5
                        }
                    },
                    {
                        "regime_id": 3,
                        "primary_strategy": "TREND_FOLLOWING",
                        "position_bias": "SHORT",
                        "option_strategy": "NAKED",
                        "trade_frequency": "HIGH",
                        "risk_parameters": {
                            "position_size_multiplier": 1.5,
                            "stop_loss_multiplier": 0.8,
                            "target_multiplier": 2.0
                        }
                    },
                    {
                        "regime_id": 4,
                        "primary_strategy": "MEAN_REVERSION",
                        "position_bias": "LONG",
                        "option_strategy": "SPREAD",
                        "trade_frequency": "HIGH",
                        "risk_parameters": {
                            "position_size_multiplier": 1.2,
                            "stop_loss_multiplier": 0.9,
                            "target_multiplier": 1.5
                        }
                    },
                    {
                        "regime_id": 5,
                        "primary_strategy": "VOLATILITY_ARBITRAGE",
                        "position_bias": "NEUTRAL",
                        "option_strategy": "STRADDLE",
                        "trade_frequency": "LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.8,
                            "stop_loss_multiplier": 1.2,
                            "target_multiplier": 1.8
                        }
                    },
                    {
                        "regime_id": 6,
                        "primary_strategy": "MEAN_REVERSION",
                        "position_bias": "SHORT",
                        "option_strategy": "SPREAD",
                        "trade_frequency": "HIGH",
                        "risk_parameters": {
                            "position_size_multiplier": 1.2,
                            "stop_loss_multiplier": 0.9,
                            "target_multiplier": 1.5
                        }
                    },
                    # Medium Volatility Strategies
                    {
                        "regime_id": 7,
                        "primary_strategy": "TREND_FOLLOWING",
                        "position_bias": "LONG",
                        "option_strategy": "SPREAD",
                        "trade_frequency": "MEDIUM",
                        "risk_parameters": {
                            "position_size_multiplier": 1.2,
                            "stop_loss_multiplier": 1.0,
                            "target_multiplier": 2.5
                        }
                    },
                    {
                        "regime_id": 8,
                        "primary_strategy": "MARKET_NEUTRAL",
                        "position_bias": "NEUTRAL",
                        "option_strategy": "BUTTERFLY",
                        "trade_frequency": "LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.8,
                            "stop_loss_multiplier": 1.2,
                            "target_multiplier": 1.8
                        }
                    },
                    {
                        "regime_id": 9,
                        "primary_strategy": "TREND_FOLLOWING",
                        "position_bias": "SHORT",
                        "option_strategy": "SPREAD",
                        "trade_frequency": "MEDIUM",
                        "risk_parameters": {
                            "position_size_multiplier": 1.2,
                            "stop_loss_multiplier": 1.0,
                            "target_multiplier": 2.5
                        }
                    },
                    {
                        "regime_id": 10,
                        "primary_strategy": "MEAN_REVERSION",
                        "position_bias": "LONG",
                        "option_strategy": "CONDOR",
                        "trade_frequency": "MEDIUM",
                        "risk_parameters": {
                            "position_size_multiplier": 1.0,
                            "stop_loss_multiplier": 1.1,
                            "target_multiplier": 2.0
                        }
                    },
                    {
                        "regime_id": 11,
                        "primary_strategy": "VOLATILITY_ARBITRAGE",
                        "position_bias": "NEUTRAL",
                        "option_strategy": "STRANGLE",
                        "trade_frequency": "LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.7,
                            "stop_loss_multiplier": 1.3,
                            "target_multiplier": 2.2
                        }
                    },
                    {
                        "regime_id": 12,
                        "primary_strategy": "MEAN_REVERSION",
                        "position_bias": "SHORT",
                        "option_strategy": "CONDOR",
                        "trade_frequency": "MEDIUM",
                        "risk_parameters": {
                            "position_size_multiplier": 1.0,
                            "stop_loss_multiplier": 1.1,
                            "target_multiplier": 2.0
                        }
                    },
                    # High Volatility Strategies
                    {
                        "regime_id": 13,
                        "primary_strategy": "TREND_FOLLOWING",
                        "position_bias": "LONG",
                        "option_strategy": "RATIO",
                        "trade_frequency": "LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.8,
                            "stop_loss_multiplier": 1.5,
                            "target_multiplier": 3.0
                        }
                    },
                    {
                        "regime_id": 14,
                        "primary_strategy": "MARKET_NEUTRAL",
                        "position_bias": "NEUTRAL",
                        "option_strategy": "STRANGLE",
                        "trade_frequency": "VERY_LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.5,
                            "stop_loss_multiplier": 1.8,
                            "target_multiplier": 2.5
                        }
                    },
                    {
                        "regime_id": 15,
                        "primary_strategy": "TREND_FOLLOWING",
                        "position_bias": "SHORT",
                        "option_strategy": "RATIO",
                        "trade_frequency": "LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.8,
                            "stop_loss_multiplier": 1.5,
                            "target_multiplier": 3.0
                        }
                    },
                    {
                        "regime_id": 16,
                        "primary_strategy": "VOLATILITY_ARBITRAGE",
                        "position_bias": "LONG",
                        "option_strategy": "STRADDLE",
                        "trade_frequency": "VERY_LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.6,
                            "stop_loss_multiplier": 1.6,
                            "target_multiplier": 2.8
                        }
                    },
                    {
                        "regime_id": 17,
                        "primary_strategy": "PROTECTIVE",
                        "position_bias": "NEUTRAL",
                        "option_strategy": "BUTTERFLY",
                        "trade_frequency": "VERY_LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.4,
                            "stop_loss_multiplier": 2.0,
                            "target_multiplier": 2.0
                        }
                    },
                    {
                        "regime_id": 18,
                        "primary_strategy": "PROTECTIVE",
                        "position_bias": "SHORT",
                        "option_strategy": "SPREAD",
                        "trade_frequency": "VERY_LOW",
                        "risk_parameters": {
                            "position_size_multiplier": 0.3,
                            "stop_loss_multiplier": 2.0,
                            "target_multiplier": 4.0
                        }
                    }
                ],
                "dynamic_adjustment": {
                    "enabled": True,
                    "adjustment_factors": {
                        "volatility_scaling": True,
                        "trend_strength_scaling": True,
                        "regime_confidence_scaling": True,
                        "market_impact_scaling": True
                    },
                    "scaling_parameters": {
                        "volatility_scale_factor": 0.8,
                        "trend_scale_factor": 1.2,
                        "confidence_scale_factor": 1.0
                    }
                }
            },
            "transition_rules": {
                "confirmation_required": True,
                "confirmation_periods": 3,
                "transition_handling": {
                    "position_adjustment": "GRADUAL",
                    "hedge_during_transition": True,
                    "reduce_size_during_transition": True,
                    "transition_buffer_percentage": 20,
                    "max_transition_time": 15
                },
                "regime_persistence": {
                    "min_regime_duration": 5,
                    "regime_stability_check": True,
                    "false_signal_filter": True,
                    "stability_threshold": 0.7
                },
                "emergency_transitions": {
                    "gap_threshold": 3,
                    "volatility_spike_threshold": 2.5,
                    "immediate_exit_regimes": [18]
                }
            },
            "advanced_features": {
                "ml_enhancement": {
                    "enabled": False,
                    "model_type": "XG_BOOST",
                    "feature_importance_threshold": 0.05,
                    "retraining_frequency": "WEEKLY",
                    "validation_window": 60,
                    "min_accuracy": 0.65
                },
                "cross_asset_signals": {
                    "use_vix": True,
                    "use_dollar_index": True,
                    "use_bond_yields": True,
                    "use_commodity_indices": False,
                    "signal_weights": {
                        "vix": 0.3,
                        "dollar_index": 0.2,
                        "bond_yields": 0.2
                    }
                },
                "sentiment_integration": {
                    "enabled": False,
                    "sources": ["NEWS_SENTIMENT", "OPTIONS_FLOW", "SOCIAL_MEDIA"],
                    "weight_in_regime_detection": 0.2,
                    "sentiment_smoothing": 5
                },
                "microstructure_analysis": {
                    "order_flow_toxicity": True,
                    "price_discovery_metrics": True,
                    "market_maker_positioning": False
                }
            },
            "risk_management": {
                "regime_specific_limits": {
                    "high_volatility_exposure_limit": 0.5,
                    "crisis_regime_exposure_limit": 0.2,
                    "max_regime_concentration": 0.7,
                    "transition_period_limit": 0.3
                },
                "portfolio_protection": {
                    "tail_hedge_allocation": 0.05,
                    "crisis_regime_protection": "PUT_OPTIONS",
                    "drawdown_circuit_breaker": 0.15,
                    "recovery_time_requirement": 5
                },
                "stress_testing": {
                    "scenarios": [
                        "2008_CRISIS",
                        "COVID_CRASH",
                        "FLASH_CRASH",
                        "VOLATILITY_SPIKE"
                    ],
                    "max_acceptable_loss": 0.25
                },
                "correlation_limits": {
                    "max_strategy_correlation": 0.6,
                    "max_regime_correlation": 0.8
                }
            },
            "monitoring": {
                "regime_metrics": [
                    "regime_stability",
                    "transition_frequency",
                    "prediction_accuracy",
                    "strategy_performance"
                ],
                "alert_conditions": {
                    "rapid_regime_changes": 3,
                    "low_confidence_duration": 10,
                    "performance_deviation": 0.3
                },
                "reporting": {
                    "frequency": "HOURLY",
                    "include_regime_analysis": True,
                    "include_performance_attribution": True
                }
            }
        }
    
    def _validate_regime_detection(self) -> List[str]:
        """Validate regime detection settings"""
        errors = []
        detection = self.get("regime_detection", {})
        
        # Validate lookback periods
        lookback = detection.get("lookback_periods", {})
        short = lookback.get("short", 0)
        medium = lookback.get("medium", 0)
        long = lookback.get("long", 0)
        
        if not (short < medium < long):
            errors.append("Lookback periods must be in ascending order: short < medium < long")
        
        if short < 5:
            errors.append("Short lookback period too small (minimum 5)")
        
        # Validate volatility calculation
        vol_calc = detection.get("volatility_calculation", {})
        annualization = vol_calc.get("annualization_factor", 0)
        if annualization < 1:
            errors.append("Annualization factor must be at least 1")
        
        # Validate trend identification
        trend = detection.get("trend_identification", {})
        threshold = trend.get("strength_threshold", 0)
        if not 0 <= threshold <= 1:
            errors.append("Trend strength threshold must be between 0 and 1")
        
        return errors
    
    def _validate_regime_parameters(self) -> List[str]:
        """Validate regime parameter definitions"""
        errors = []
        params = self.get("regime_parameters", {})
        
        # Validate regime definitions
        regimes = params.get("regime_definitions", [])
        if len(regimes) != 18:
            errors.append(f"Expected 18 regime definitions, got {len(regimes)}")
        
        regime_ids = set()
        for regime in regimes:
            regime_id = regime.get("regime_id")
            if regime_id in regime_ids:
                errors.append(f"Duplicate regime ID: {regime_id}")
            regime_ids.add(regime_id)
            
            # Validate characteristics
            chars = regime.get("characteristics", {})
            vol_min = chars.get("vol_percentile_min", 0)
            vol_max = chars.get("vol_percentile_max", 100)
            
            if not 0 <= vol_min <= vol_max <= 100:
                errors.append(f"Regime {regime_id}: Invalid volatility percentile range")
            
            trend_min = chars.get("trend_strength_min", -1)
            trend_max = chars.get("trend_strength_max", 1)
            
            if not -1 <= trend_min <= trend_max <= 1:
                errors.append(f"Regime {regime_id}: Invalid trend strength range")
        
        # Validate thresholds
        thresholds = params.get("thresholds", {})
        confidence = thresholds.get("confidence_threshold", 0)
        if not 0 <= confidence <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Validate triple straddle analysis
        tsa = params.get("triple_straddle_analysis", {})
        if tsa.get("enabled"):
            windows = tsa.get("rolling_windows", [])
            if not windows:
                errors.append("Rolling windows required for triple straddle analysis")
            elif windows != sorted(windows):
                errors.append("Rolling windows must be in ascending order")
        
        return errors
    
    def _validate_strategy_mapping(self) -> List[str]:
        """Validate strategy mapping"""
        errors = []
        mapping = self.get("strategy_mapping", {})
        
        # Validate regime strategies
        strategies = mapping.get("regime_strategies", [])
        if len(strategies) != 18:
            errors.append(f"Expected 18 regime strategies, got {len(strategies)}")
        
        mapped_regimes = set()
        for strategy in strategies:
            regime_id = strategy.get("regime_id")
            if regime_id in mapped_regimes:
                errors.append(f"Duplicate strategy mapping for regime {regime_id}")
            mapped_regimes.add(regime_id)
            
            # Validate risk parameters
            risk_params = strategy.get("risk_parameters", {})
            size_mult = risk_params.get("position_size_multiplier", 1)
            if not 0.1 <= size_mult <= 3:
                errors.append(f"Regime {regime_id}: Position size multiplier out of range")
            
            sl_mult = risk_params.get("stop_loss_multiplier", 1)
            if not 0.5 <= sl_mult <= 2:
                errors.append(f"Regime {regime_id}: Stop loss multiplier out of range")
            
            target_mult = risk_params.get("target_multiplier", 1)
            if not 0.5 <= target_mult <= 5:
                errors.append(f"Regime {regime_id}: Target multiplier out of range")
        
        return errors
    
    def _validate_transition_rules(self) -> List[str]:
        """Validate transition rules"""
        errors = []
        transitions = self.get("transition_rules", {})
        
        # Validate confirmation periods
        confirmation = transitions.get("confirmation_periods", 0)
        if confirmation < 1:
            errors.append("Confirmation periods must be at least 1")
        
        # Validate transition handling
        handling = transitions.get("transition_handling", {})
        buffer = handling.get("transition_buffer_percentage", 0)
        if not 0 <= buffer <= 50:
            errors.append("Transition buffer percentage must be between 0 and 50")
        
        # Validate regime persistence
        persistence = transitions.get("regime_persistence", {})
        min_duration = persistence.get("min_regime_duration", 0)
        if min_duration < 1:
            errors.append("Minimum regime duration must be at least 1")
        
        return errors
    
    def get_current_regime(self, market_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Determine current market regime
        
        Args:
            market_data: Current market data including price, volume, volatility
            
        Returns:
            Tuple of (regime_id, confidence_score)
        """
        # This is a simplified implementation
        # In production, this would use the full regime detection logic
        
        volatility = market_data.get("volatility", 0)
        trend = market_data.get("trend_strength", 0)
        structure = market_data.get("hurst_exponent", 0.5)
        
        # Determine volatility level
        vol_breakpoints = self.get("regime_parameters.thresholds.volatility_breakpoints", [33, 67])
        if volatility < vol_breakpoints[0]:
            vol_level = "LOW"
            regime_base = 1
        elif volatility < vol_breakpoints[1]:
            vol_level = "MEDIUM"
            regime_base = 7
        else:
            vol_level = "HIGH"
            regime_base = 13
        
        # Determine trend direction
        trend_breakpoints = self.get("regime_parameters.thresholds.trend_breakpoints", [-0.3, 0.3])
        if trend < trend_breakpoints[0]:
            trend_offset = 2  # Bearish
        elif trend < trend_breakpoints[1]:
            trend_offset = 1  # Neutral
        else:
            trend_offset = 0  # Bullish
        
        # Determine structure
        if structure > 0.6:
            structure_offset = 0  # Trending
        else:
            structure_offset = 3  # Ranging
        
        regime_id = regime_base + trend_offset + structure_offset
        
        # Calculate confidence (simplified)
        confidence = min(0.95, abs(trend) + (1 - abs(structure - 0.5)) / 2)
        
        return regime_id, confidence
    
    def get_regime_strategy(self, regime_id: int) -> Dict[str, Any]:
        """
        Get strategy configuration for a specific regime
        
        Args:
            regime_id: Regime identifier (1-18)
            
        Returns:
            Strategy configuration for the regime
        """
        strategies = self.get("strategy_mapping.regime_strategies", [])
        
        for strategy in strategies:
            if strategy.get("regime_id") == regime_id:
                return strategy
        
        # Return default strategy if not found
        return {
            "primary_strategy": "MARKET_NEUTRAL",
            "position_bias": "NEUTRAL",
            "option_strategy": "SPREAD",
            "trade_frequency": "LOW",
            "risk_parameters": {
                "position_size_multiplier": 0.5,
                "stop_loss_multiplier": 1.5,
                "target_multiplier": 2.0
            }
        }
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get human-readable name for regime"""
        regimes = self.get("regime_parameters.regime_definitions", [])
        
        for regime in regimes:
            if regime.get("regime_id") == regime_id:
                return regime.get("name", f"Regime {regime_id}")
        
        return f"Unknown Regime {regime_id}"
    
    def should_transition(self, current_regime: int, new_regime: int, confidence: float) -> bool:
        """
        Determine if regime transition should occur
        
        Args:
            current_regime: Current regime ID
            new_regime: Potential new regime ID
            confidence: Confidence score for new regime
            
        Returns:
            True if transition should occur
        """
        if current_regime == new_regime:
            return False
        
        # Check confidence threshold
        min_confidence = self.get("regime_parameters.thresholds.confidence_threshold", 0.7)
        if confidence < min_confidence:
            return False
        
        # Check if it's an emergency transition
        emergency_regimes = self.get("transition_rules.emergency_transitions.immediate_exit_regimes", [])
        if new_regime in emergency_regimes:
            return True
        
        # Normal transition logic would include confirmation periods
        # This is simplified for the example
        return True
    
    def get_risk_limits(self, regime_id: int) -> Dict[str, float]:
        """
        Get risk limits for specific regime
        
        Args:
            regime_id: Regime identifier
            
        Returns:
            Risk limits dictionary
        """
        # Determine volatility level from regime
        if regime_id <= 6:
            vol_level = "LOW"
        elif regime_id <= 12:
            vol_level = "MEDIUM"
        else:
            vol_level = "HIGH"
        
        # Get base limits
        risk_mgmt = self.get("risk_management", {})
        
        # Adjust based on volatility
        if vol_level == "HIGH":
            exposure_limit = risk_mgmt.get("regime_specific_limits.high_volatility_exposure_limit", 0.5)
        elif regime_id == 18:  # Crisis regime
            exposure_limit = risk_mgmt.get("regime_specific_limits.crisis_regime_exposure_limit", 0.2)
        else:
            exposure_limit = 1.0
        
        return {
            "max_exposure": exposure_limit,
            "max_positions": 5 if vol_level == "LOW" else 3 if vol_level == "MEDIUM" else 2,
            "position_size_limit": 0.2 * exposure_limit,
            "stop_loss_multiplier": self.get_regime_strategy(regime_id).get("risk_parameters", {}).get("stop_loss_multiplier", 1.0)
        }
    
    def __str__(self) -> str:
        """String representation"""
        method = self.get("regime_detection.classification_method", "UNKNOWN")
        return f"Market Regime Configuration: {self.strategy_name} (Method: {method}, 18 Regimes)"