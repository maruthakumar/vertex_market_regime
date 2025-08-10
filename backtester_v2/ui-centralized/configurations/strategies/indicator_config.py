"""
Indicator Strategy Configuration
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import time
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class IndicatorConfiguration(BaseConfiguration):
    """
    Configuration for Technical Indicator Strategy
    
    This configuration handles indicator-based trading strategies including:
    - TA-Lib indicators (RSI, MACD, Bollinger Bands, etc.)
    - Smart Money Concepts (BOS, CHOCH, Order Blocks, etc.)
    - Candlestick patterns (Doji, Hammer, Engulfing, etc.)
    - Multi-timeframe analysis with weighted consensus
    - Complex signal conditions with AND/OR/NOT logic
    """
    
    def __init__(self, strategy_name: str):
        """Initialize Indicator configuration"""
        super().__init__("indicator", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate Indicator configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate indicator configurations
        indicator_errors = self._validate_indicators()
        if indicator_errors:
            errors['indicators'] = indicator_errors
        
        # Validate signal conditions
        signal_errors = self._validate_signal_conditions()
        if signal_errors:
            errors['signal_conditions'] = signal_errors
        
        # Validate risk management
        risk_errors = self._validate_risk_management()
        if risk_errors:
            errors['risk_management'] = risk_errors
        
        # Validate timeframe settings
        timeframe_errors = self._validate_timeframe_settings()
        if timeframe_errors:
            errors['timeframe_settings'] = timeframe_errors
        
        if errors:
            raise ValidationError("Indicator configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Indicator configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["indicators", "signal_conditions", "risk_management", "timeframe_settings"],
            "properties": {
                "indicators": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type", "category", "timeframe", "enabled"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["TALIB", "SMC", "PATTERN", "CUSTOM"]},
                            "category": {
                                "type": "string",
                                "enum": ["MOMENTUM", "TREND", "VOLATILITY", "VOLUME", "STRUCTURE", "CANDLESTICK"]
                            },
                            "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]},
                            "period": {"type": "integer", "minimum": 1, "maximum": 200},
                            "enabled": {"type": "boolean"},
                            "weight": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
                            "threshold_upper": {"type": "number"},
                            "threshold_lower": {"type": "number"},
                            "signal_type": {
                                "type": "string",
                                "enum": [
                                    "OVERBOUGHT_OVERSOLD", "CROSSOVER", "PRICE_CROSSOVER",
                                    "TREND_STRENGTH", "BAND_SQUEEZE", "VOLATILITY_FILTER",
                                    "VOLUME_CONFIRMATION", "STRUCTURE_BREAK", "STRUCTURE_CHANGE",
                                    "SUPPORT_RESISTANCE", "REVERSAL_PATTERN"
                                ]
                            },
                            "parameters": {"type": "object"}
                        }
                    }
                },
                "signal_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["condition_id", "condition_type", "logic_operator"],
                        "properties": {
                            "condition_id": {"type": "string"},
                            "condition_type": {
                                "type": "string",
                                "enum": ["ENTRY", "EXIT", "STOP_LOSS", "TAKE_PROFIT"]
                            },
                            "logic_operator": {"type": "string", "enum": ["AND", "OR", "NOT"]},
                            "indicator1_name": {"type": "string"},
                            "operator1": {
                                "type": "string",
                                "enum": [
                                    "LESS_THAN", "GREATER_THAN", "EQUALS",
                                    "CROSSOVER_ABOVE", "CROSSOVER_BELOW",
                                    "PRICE_ABOVE", "PRICE_BELOW", "MULTIPLIER"
                                ]
                            },
                            "value1": {"type": "number"},
                            "indicator2_name": {"type": "string"},
                            "operator2": {"type": "string"},
                            "value2": {"type": "number"},
                            "condition_weight": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
                            "condition_enabled": {"type": "boolean"}
                        }
                    }
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "position_sizing_method": {
                            "type": "string",
                            "enum": ["FIXED_AMOUNT", "PERCENTAGE_RISK", "ATR_BASED", "VOLATILITY_ADJUSTED"]
                        },
                        "fixed_position_size": {"type": "number", "minimum": 1000, "maximum": 10000000},
                        "risk_percentage": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                        "max_positions": {"type": "integer", "minimum": 1, "maximum": 10},
                        "max_daily_loss": {"type": "number", "minimum": 100, "maximum": 100000},
                        "max_drawdown": {"type": "number", "minimum": 1.0, "maximum": 50.0},
                        "stop_loss_type": {
                            "type": "string",
                            "enum": ["FIXED", "PERCENTAGE", "ATR_BASED", "INDICATOR_BASED"]
                        },
                        "stop_loss_value": {"type": "number", "minimum": 0},
                        "take_profit_type": {
                            "type": "string",
                            "enum": ["FIXED", "PERCENTAGE", "ATR_BASED", "RISK_REWARD"]
                        },
                        "take_profit_value": {"type": "number", "minimum": 0},
                        "trailing_stop": {"type": "boolean"},
                        "trailing_stop_activation": {"type": "number", "minimum": 0},
                        "trailing_stop_distance": {"type": "number", "minimum": 0}
                    }
                },
                "timeframe_settings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["timeframe", "enabled", "weight", "purpose"],
                        "properties": {
                            "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]},
                            "enabled": {"type": "boolean"},
                            "weight": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
                            "purpose": {
                                "type": "string",
                                "enum": [
                                    "ENTRY_TIMING", "PRIMARY_SIGNALS", "TREND_CONFIRMATION",
                                    "MAJOR_TREND", "LONG_TERM_BIAS", "MARKET_STRUCTURE"
                                ]
                            },
                            "indicator_set": {
                                "type": "string",
                                "enum": [
                                    "FULL", "TREND", "MOMENTUM", "VOLATILITY",
                                    "VOLUME", "SMC", "PATTERN", "BASIC", "SCALPING"
                                ]
                            },
                            "priority": {"type": "integer", "minimum": 1, "maximum": 10}
                        }
                    }
                },
                "entry_rules": {
                    "type": "object",
                    "properties": {
                        "entry_start_time": {"type": "string", "format": "time"},
                        "entry_end_time": {"type": "string", "format": "time"},
                        "min_indicators_agreement": {"type": "integer", "minimum": 1},
                        "confirmation_candles": {"type": "integer", "minimum": 0, "maximum": 5},
                        "entry_on_close": {"type": "boolean"},
                        "limit_order_offset": {"type": "number", "minimum": 0},
                        "max_entries_per_day": {"type": "integer", "minimum": 1}
                    }
                },
                "exit_rules": {
                    "type": "object",
                    "properties": {
                        "exit_on_opposite_signal": {"type": "boolean"},
                        "time_based_exit": {"type": "string", "format": "time"},
                        "max_holding_period": {"type": "integer", "minimum": 1},
                        "partial_exit_enabled": {"type": "boolean"},
                        "partial_exit_levels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "percentage": {"type": "number", "minimum": 0, "maximum": 100},
                                    "trigger": {"type": "number", "minimum": 0}
                                }
                            }
                        }
                    }
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "volume_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "min_volume": {"type": "integer", "minimum": 0},
                                "volume_ma_period": {"type": "integer", "minimum": 1},
                                "volume_multiplier": {"type": "number", "minimum": 0}
                            }
                        },
                        "volatility_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "min_atr": {"type": "number", "minimum": 0},
                                "max_atr": {"type": "number", "minimum": 0},
                                "atr_period": {"type": "integer", "minimum": 1}
                            }
                        },
                        "trend_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "trend_indicator": {"type": "string"},
                                "trend_period": {"type": "integer", "minimum": 1},
                                "min_trend_strength": {"type": "number", "minimum": 0}
                            }
                        },
                        "time_filter": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "trading_sessions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "string", "format": "time"},
                                            "end": {"type": "string", "format": "time"}
                                        }
                                    }
                                },
                                "avoid_news_time": {"type": "boolean"},
                                "news_buffer_minutes": {"type": "integer", "minimum": 0}
                            }
                        }
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for Indicator configuration"""
        return {
            "indicators": [
                # Momentum Indicators
                {
                    "name": "RSI",
                    "type": "TALIB",
                    "category": "MOMENTUM",
                    "timeframe": "5m",
                    "period": 14,
                    "enabled": True,
                    "weight": 0.15,
                    "threshold_upper": 70,
                    "threshold_lower": 30,
                    "signal_type": "OVERBOUGHT_OVERSOLD",
                    "parameters": {
                        "price_type": "close"
                    }
                },
                {
                    "name": "MACD",
                    "type": "TALIB",
                    "category": "MOMENTUM",
                    "timeframe": "5m",
                    "period": 0,  # Uses default 12,26,9
                    "enabled": True,
                    "weight": 0.15,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "CROSSOVER",
                    "parameters": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    }
                },
                # Trend Indicators
                {
                    "name": "EMA",
                    "type": "TALIB",
                    "category": "TREND",
                    "timeframe": "5m",
                    "period": 20,
                    "enabled": True,
                    "weight": 0.10,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "PRICE_CROSSOVER",
                    "parameters": {}
                },
                {
                    "name": "ADX",
                    "type": "TALIB",
                    "category": "TREND",
                    "timeframe": "15m",
                    "period": 14,
                    "enabled": True,
                    "weight": 0.10,
                    "threshold_upper": 25,
                    "threshold_lower": 20,
                    "signal_type": "TREND_STRENGTH",
                    "parameters": {}
                },
                # Volatility Indicators
                {
                    "name": "BBANDS",
                    "type": "TALIB",
                    "category": "VOLATILITY",
                    "timeframe": "5m",
                    "period": 20,
                    "enabled": True,
                    "weight": 0.10,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "BAND_SQUEEZE",
                    "parameters": {
                        "nb_dev_up": 2,
                        "nb_dev_dn": 2,
                        "ma_type": 0
                    }
                },
                {
                    "name": "ATR",
                    "type": "TALIB",
                    "category": "VOLATILITY",
                    "timeframe": "5m",
                    "period": 14,
                    "enabled": True,
                    "weight": 0.05,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "VOLATILITY_FILTER",
                    "parameters": {}
                },
                # Volume Indicators
                {
                    "name": "OBV",
                    "type": "TALIB",
                    "category": "VOLUME",
                    "timeframe": "5m",
                    "period": 0,
                    "enabled": True,
                    "weight": 0.10,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "VOLUME_CONFIRMATION",
                    "parameters": {}
                },
                # Smart Money Concepts
                {
                    "name": "BOS_DETECTION",
                    "type": "SMC",
                    "category": "STRUCTURE",
                    "timeframe": "15m",
                    "period": 20,
                    "enabled": True,
                    "weight": 0.15,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "STRUCTURE_BREAK",
                    "parameters": {
                        "swing_length": 5,
                        "confirmation_candles": 2
                    }
                },
                {
                    "name": "ORDER_BLOCK",
                    "type": "SMC",
                    "category": "STRUCTURE",
                    "timeframe": "15m",
                    "period": 10,
                    "enabled": True,
                    "weight": 0.10,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "SUPPORT_RESISTANCE",
                    "parameters": {
                        "lookback": 50,
                        "min_touches": 2
                    }
                },
                # Candlestick Patterns
                {
                    "name": "HAMMER",
                    "type": "PATTERN",
                    "category": "CANDLESTICK",
                    "timeframe": "5m",
                    "period": 0,
                    "enabled": False,
                    "weight": 0.00,
                    "threshold_upper": 0,
                    "threshold_lower": 0,
                    "signal_type": "REVERSAL_PATTERN",
                    "parameters": {}
                }
            ],
            "signal_conditions": [
                # Entry Conditions
                {
                    "condition_id": "ENTRY_001",
                    "condition_type": "ENTRY",
                    "logic_operator": "AND",
                    "indicator1_name": "RSI",
                    "operator1": "LESS_THAN",
                    "value1": 30,
                    "indicator2_name": "MACD",
                    "operator2": "CROSSOVER_ABOVE",
                    "value2": 0,
                    "condition_weight": 0.5,
                    "condition_enabled": True
                },
                {
                    "condition_id": "ENTRY_002",
                    "condition_type": "ENTRY",
                    "logic_operator": "AND",
                    "indicator1_name": "BOS_DETECTION",
                    "operator1": "EQUALS",
                    "value1": 1,
                    "indicator2_name": "ADX",
                    "operator2": "GREATER_THAN",
                    "value2": 25,
                    "condition_weight": 0.5,
                    "condition_enabled": True
                },
                # Exit Conditions
                {
                    "condition_id": "EXIT_001",
                    "condition_type": "EXIT",
                    "logic_operator": "OR",
                    "indicator1_name": "RSI",
                    "operator1": "GREATER_THAN",
                    "value1": 70,
                    "indicator2_name": "MACD",
                    "operator2": "CROSSOVER_BELOW",
                    "value2": 0,
                    "condition_weight": 1.0,
                    "condition_enabled": True
                },
                # Stop Loss Condition
                {
                    "condition_id": "SL_001",
                    "condition_type": "STOP_LOSS",
                    "logic_operator": "AND",
                    "indicator1_name": "ATR",
                    "operator1": "MULTIPLIER",
                    "value1": 2.0,
                    "indicator2_name": None,
                    "operator2": None,
                    "value2": None,
                    "condition_weight": 1.0,
                    "condition_enabled": True
                },
                # Take Profit Condition
                {
                    "condition_id": "TP_001",
                    "condition_type": "TAKE_PROFIT",
                    "logic_operator": "AND",
                    "indicator1_name": "ATR",
                    "operator1": "MULTIPLIER",
                    "value1": 3.0,
                    "indicator2_name": None,
                    "operator2": None,
                    "value2": None,
                    "condition_weight": 1.0,
                    "condition_enabled": True
                }
            ],
            "risk_management": {
                "position_sizing_method": "FIXED_AMOUNT",
                "fixed_position_size": 100000,
                "risk_percentage": 2.0,
                "max_positions": 3,
                "max_daily_loss": 5000,
                "max_drawdown": 10.0,
                "stop_loss_type": "ATR_BASED",
                "stop_loss_value": 2.0,
                "take_profit_type": "RISK_REWARD",
                "take_profit_value": 2.0,
                "trailing_stop": False,
                "trailing_stop_activation": 1.5,
                "trailing_stop_distance": 0.5,
                "use_kelly_criterion": False,
                "kelly_fraction": 0.25
            },
            "timeframe_settings": [
                {
                    "timeframe": "1m",
                    "enabled": False,
                    "weight": 0.10,
                    "purpose": "ENTRY_TIMING",
                    "indicator_set": "SCALPING",
                    "priority": 6
                },
                {
                    "timeframe": "5m",
                    "enabled": True,
                    "weight": 0.40,
                    "purpose": "PRIMARY_SIGNALS",
                    "indicator_set": "FULL",
                    "priority": 1
                },
                {
                    "timeframe": "15m",
                    "enabled": True,
                    "weight": 0.30,
                    "purpose": "TREND_CONFIRMATION",
                    "indicator_set": "TREND",
                    "priority": 2
                },
                {
                    "timeframe": "1h",
                    "enabled": True,
                    "weight": 0.20,
                    "purpose": "MAJOR_TREND",
                    "indicator_set": "TREND",
                    "priority": 3
                },
                {
                    "timeframe": "4h",
                    "enabled": False,
                    "weight": 0.00,
                    "purpose": "LONG_TERM_BIAS",
                    "indicator_set": "BASIC",
                    "priority": 4
                },
                {
                    "timeframe": "1d",
                    "enabled": False,
                    "weight": 0.00,
                    "purpose": "MARKET_STRUCTURE",
                    "indicator_set": "BASIC",
                    "priority": 5
                }
            ],
            "entry_rules": {
                "entry_start_time": "09:30:00",
                "entry_end_time": "15:00:00",
                "min_indicators_agreement": 2,
                "confirmation_candles": 1,
                "entry_on_close": True,
                "limit_order_offset": 0,
                "max_entries_per_day": 5,
                "re_entry_cooldown": 30,
                "avoid_high_impact_news": True
            },
            "exit_rules": {
                "exit_on_opposite_signal": True,
                "time_based_exit": "15:15:00",
                "max_holding_period": 240,  # minutes
                "partial_exit_enabled": False,
                "partial_exit_levels": [
                    {"percentage": 50, "trigger": 1.0},
                    {"percentage": 30, "trigger": 1.5},
                    {"percentage": 20, "trigger": 2.0}
                ],
                "break_even_enabled": True,
                "break_even_trigger": 1.0
            },
            "filters": {
                "volume_filter": {
                    "enabled": True,
                    "min_volume": 1000000,
                    "volume_ma_period": 20,
                    "volume_multiplier": 1.5,
                    "volume_profile_analysis": False
                },
                "volatility_filter": {
                    "enabled": True,
                    "min_atr": 50,
                    "max_atr": 200,
                    "atr_period": 14,
                    "vix_filter": False,
                    "vix_threshold": 30
                },
                "trend_filter": {
                    "enabled": True,
                    "trend_indicator": "EMA",
                    "trend_period": 50,
                    "min_trend_strength": 0.5,
                    "trend_alignment_required": True
                },
                "time_filter": {
                    "enabled": True,
                    "trading_sessions": [
                        {"start": "09:30:00", "end": "11:30:00"},
                        {"start": "13:00:00", "end": "15:00:00"}
                    ],
                    "avoid_news_time": True,
                    "news_buffer_minutes": 30,
                    "avoid_rollover": True,
                    "avoid_expiry_day": True
                }
            },
            "monitoring": {
                "performance_tracking": True,
                "indicator_performance": True,
                "condition_hit_rate": True,
                "timeframe_contribution": True,
                "real_time_alerts": False
            }
        }
    
    def _validate_indicators(self) -> List[str]:
        """Validate indicator configurations"""
        errors = []
        indicators = self.get("indicators", [])
        
        if not indicators:
            errors.append("At least one indicator must be configured")
            return errors
        
        # Track indicator names for duplicate check
        names = []
        total_weight = 0
        
        # Valid indicator names by type
        valid_talib = ['RSI', 'MACD', 'EMA', 'SMA', 'STOCH', 'CCI', 'WILLIAMS_R', 
                       'ADX', 'BBANDS', 'ATR', 'OBV', 'AD']
        valid_smc = ['BOS_DETECTION', 'CHOCH_DETECTION', 'ORDER_BLOCK', 
                     'FAIR_VALUE_GAP', 'LIQUIDITY_SWEEP']
        valid_pattern = ['DOJI', 'HAMMER', 'ENGULFING', 'SHOOTING_STAR', 
                         'MORNING_STAR', 'EVENING_STAR']
        
        for i, indicator in enumerate(indicators):
            name = indicator.get("name", "")
            ind_type = indicator.get("type", "")
            
            # Check for duplicates
            if name in names:
                errors.append(f"Duplicate indicator name: {name}")
            names.append(name)
            
            # Validate indicator name against type
            if ind_type == "TALIB" and name not in valid_talib:
                errors.append(f"Invalid TA-Lib indicator: {name}")
            elif ind_type == "SMC" and name not in valid_smc:
                errors.append(f"Invalid SMC indicator: {name}")
            elif ind_type == "PATTERN" and name not in valid_pattern:
                errors.append(f"Invalid pattern indicator: {name}")
            
            # Validate period for specific indicators
            period = indicator.get("period", 0)
            if name in ['RSI', 'STOCH', 'CCI', 'WILLIAMS_R', 'ADX', 'ATR'] and period > 0:
                if not 2 <= period <= 100:
                    errors.append(f"Period for {name} must be between 2-100")
            
            # Validate thresholds
            if indicator.get("signal_type") == "OVERBOUGHT_OVERSOLD":
                upper = indicator.get("threshold_upper", 0)
                lower = indicator.get("threshold_lower", 0)
                if upper <= lower:
                    errors.append(f"{name}: Upper threshold must be greater than lower threshold")
            
            # Sum weights for enabled indicators
            if indicator.get("enabled", False):
                total_weight += indicator.get("weight", 0)
        
        # Validate total weight
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            errors.append(f"Enabled indicator weights must sum to 1.0, got {total_weight}")
        
        return errors
    
    def _validate_signal_conditions(self) -> List[str]:
        """Validate signal condition logic"""
        errors = []
        conditions = self.get("signal_conditions", [])
        
        if not conditions:
            errors.append("At least one signal condition must be configured")
            return errors
        
        # Get available indicator names
        indicators = self.get("indicators", [])
        available_indicators = [ind.get("name") for ind in indicators if ind.get("enabled")]
        
        # Check for required condition types
        condition_types = [c.get("condition_type") for c in conditions]
        if "ENTRY" not in condition_types:
            errors.append("At least one ENTRY condition is required")
        
        # Validate each condition
        for i, condition in enumerate(conditions):
            cond_id = condition.get("condition_id", f"COND_{i}")
            
            # Validate indicator references
            ind1 = condition.get("indicator1_name")
            if ind1 and ind1 not in available_indicators:
                errors.append(f"{cond_id}: Indicator '{ind1}' not found or not enabled")
            
            ind2 = condition.get("indicator2_name")
            if ind2 and ind2 not in available_indicators:
                errors.append(f"{cond_id}: Indicator '{ind2}' not found or not enabled")
            
            # Validate operator logic
            operator1 = condition.get("operator1")
            if operator1 == "MULTIPLIER" and not ind1:
                errors.append(f"{cond_id}: MULTIPLIER operator requires indicator1")
        
        # Validate weights
        entry_conditions = [c for c in conditions if c.get("condition_type") == "ENTRY"]
        if entry_conditions:
            total_weight = sum(c.get("condition_weight", 0) for c in entry_conditions 
                             if c.get("condition_enabled", True))
            if total_weight > 1.0:
                errors.append(f"Total entry condition weights exceed 1.0: {total_weight}")
        
        return errors
    
    def _validate_risk_management(self) -> List[str]:
        """Validate risk management settings"""
        errors = []
        risk = self.get("risk_management", {})
        
        # Validate position sizing
        method = risk.get("position_sizing_method")
        if method == "FIXED_AMOUNT":
            size = risk.get("fixed_position_size", 0)
            if size <= 0:
                errors.append("Fixed position size must be positive")
        elif method == "PERCENTAGE_RISK":
            pct = risk.get("risk_percentage", 0)
            if not 0 < pct <= 10:
                errors.append("Risk percentage must be between 0 and 10")
        
        # Validate stop loss and take profit
        sl_type = risk.get("stop_loss_type")
        tp_type = risk.get("take_profit_type")
        
        if sl_type == "ATR_BASED":
            multiplier = risk.get("stop_loss_value", 0)
            if multiplier <= 0:
                errors.append("ATR multiplier for stop loss must be positive")
        
        if tp_type == "RISK_REWARD":
            ratio = risk.get("take_profit_value", 0)
            if ratio <= 0:
                errors.append("Risk-reward ratio must be positive")
        
        # Validate trailing stop
        if risk.get("trailing_stop"):
            activation = risk.get("trailing_stop_activation", 0)
            distance = risk.get("trailing_stop_distance", 0)
            if activation <= 0 or distance <= 0:
                errors.append("Trailing stop activation and distance must be positive")
        
        return errors
    
    def _validate_timeframe_settings(self) -> List[str]:
        """Validate multi-timeframe configuration"""
        errors = []
        timeframes = self.get("timeframe_settings", [])
        
        if not timeframes:
            errors.append("At least one timeframe must be configured")
            return errors
        
        enabled_tf = [tf for tf in timeframes if tf.get("enabled")]
        if len(enabled_tf) < 1:
            errors.append("At least one timeframe must be enabled")
        
        # Check weight distribution
        total_weight = sum(tf.get("weight", 0) for tf in enabled_tf)
        if abs(total_weight - 1.0) > 0.0001:
            errors.append(f"Enabled timeframe weights must sum to 1.0, got {total_weight}")
        
        # Check priority uniqueness
        priorities = [tf.get("priority") for tf in enabled_tf]
        if len(priorities) != len(set(priorities)):
            errors.append("Duplicate priorities found in enabled timeframes")
        
        # Validate timeframe hierarchy
        tf_order = ['1m', '5m', '15m', '1h', '4h', '1d']
        enabled_order = [tf.get("timeframe") for tf in enabled_tf 
                        if tf.get("timeframe") in tf_order]
        
        if len(enabled_order) < 1:
            errors.append("At least one standard timeframe must be enabled")
        
        return errors
    
    def get_enabled_indicators(self) -> List[Dict[str, Any]]:
        """Get list of enabled indicators"""
        indicators = self.get("indicators", [])
        return [ind for ind in indicators if ind.get("enabled", False)]
    
    def get_signal_conditions(self, condition_type: str = None) -> List[Dict[str, Any]]:
        """Get signal conditions, optionally filtered by type"""
        conditions = self.get("signal_conditions", [])
        if condition_type:
            return [c for c in conditions 
                   if c.get("condition_type") == condition_type and c.get("condition_enabled", True)]
        return [c for c in conditions if c.get("condition_enabled", True)]
    
    def get_timeframe_weights(self) -> Dict[str, float]:
        """Get timeframe weights for multi-timeframe analysis"""
        timeframes = self.get("timeframe_settings", [])
        weights = {}
        for tf in timeframes:
            if tf.get("enabled"):
                weights[tf.get("timeframe")] = tf.get("weight", 0)
        return weights
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return self.get("risk_management", {})
    
    def get_active_filters(self) -> List[str]:
        """Get list of active filters"""
        filters = self.get("filters", {})
        active = []
        for filter_name, filter_config in filters.items():
            if isinstance(filter_config, dict) and filter_config.get("enabled", False):
                active.append(filter_name)
        return active
    
    def __str__(self) -> str:
        """String representation"""
        indicator_count = len(self.get_enabled_indicators())
        timeframe_count = len([tf for tf in self.get("timeframe_settings", []) if tf.get("enabled")])
        return f"Indicator Configuration: {self.strategy_name} ({indicator_count} indicators, {timeframe_count} timeframes)"