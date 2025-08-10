"""
POS (Positional) Configuration
"""

from typing import Dict, Any, List, Optional
from datetime import time, datetime
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class POSConfiguration(BaseConfiguration):
    """
    Configuration for Positional Strategy with Greeks (POS)
    
    This configuration handles all POS-specific parameters including
    Greeks-based position management, multi-leg strategies, hedging,
    and advanced risk management using option Greeks.
    """
    
    def __init__(self, strategy_name: str):
        """Initialize POS configuration"""
        super().__init__("pos", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate POS configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate Greeks settings
        greeks_errors = self._validate_greeks_settings()
        if greeks_errors:
            errors['greeks_settings'] = greeks_errors
        
        # Validate position structure
        position_errors = self._validate_position_structure()
        if position_errors:
            errors['position_structure'] = position_errors
        
        # Validate hedging parameters
        hedging_errors = self._validate_hedging_parameters()
        if hedging_errors:
            errors['hedging_parameters'] = hedging_errors
        
        # Validate risk management
        risk_errors = self._validate_risk_management()
        if risk_errors:
            errors['risk_management'] = risk_errors
        
        if errors:
            raise ValidationError("POS configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for POS configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["greeks_settings", "position_structure", "risk_management"],
            "properties": {
                "greeks_settings": {
                    "type": "object",
                    "required": ["calculation_method", "update_frequency", "thresholds"],
                    "properties": {
                        "calculation_method": {
                            "type": "string",
                            "enum": ["BLACK_SCHOLES", "BINOMIAL", "MONTE_CARLO", "SABR"]
                        },
                        "update_frequency": {
                            "type": "string",
                            "enum": ["REALTIME", "1min", "5min", "15min", "30min"]
                        },
                        "implied_volatility": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "enum": ["MARKET", "HISTORICAL", "GARCH", "EWMA"]},
                                "adjustment_factor": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                                "smile_adjustment": {"type": "boolean"}
                            }
                        },
                        "thresholds": {
                            "type": "object",
                            "properties": {
                                "delta": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number", "minimum": -1, "maximum": 0},
                                        "max": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                },
                                "gamma": {
                                    "type": "object",
                                    "properties": {
                                        "max": {"type": "number", "minimum": 0}
                                    }
                                },
                                "vega": {
                                    "type": "object",
                                    "properties": {
                                        "max": {"type": "number", "minimum": 0}
                                    }
                                },
                                "theta": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number", "maximum": 0}
                                    }
                                }
                            }
                        }
                    }
                },
                "position_structure": {
                    "type": "object",
                    "required": ["strategy_type", "legs"],
                    "properties": {
                        "strategy_type": {
                            "type": "string",
                            "enum": ["NAKED", "SPREAD", "BUTTERFLY", "CONDOR", "STRADDLE", "STRANGLE", "RATIO", "CALENDAR", "DIAGONAL", "CUSTOM"]
                        },
                        "legs": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "position": {"type": "string", "enum": ["BUY", "SELL"]},
                                    "option_type": {"type": "string", "enum": ["CE", "PE"]},
                                    "strike_selection": {
                                        "type": "object",
                                        "properties": {
                                            "method": {"type": "string", "enum": ["ATM", "OTM", "ITM", "DELTA_BASED", "PERCENTAGE_BASED"]},
                                            "offset": {"type": "integer"},
                                            "delta_target": {"type": "number", "minimum": 0, "maximum": 1}
                                        }
                                    },
                                    "quantity_ratio": {"type": "integer", "minimum": 1}
                                }
                            }
                        },
                        "expiry_selection": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["CURRENT_WEEK", "NEXT_WEEK", "MONTHLY", "DAYS_TO_EXPIRY"]},
                                "days_to_expiry": {"type": "integer", "minimum": 1, "maximum": 90},
                                "roll_days_before": {"type": "integer", "minimum": 0, "maximum": 7}
                            }
                        }
                    }
                },
                "hedging_parameters": {
                    "type": "object",
                    "properties": {
                        "delta_hedging": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "threshold": {"type": "number", "minimum": 0},
                                "hedge_instrument": {"type": "string", "enum": ["FUTURES", "OPTIONS", "SPOT"]},
                                "rebalance_frequency": {"type": "string", "enum": ["CONTINUOUS", "DISCRETE", "THRESHOLD_BASED"]}
                            }
                        },
                        "gamma_hedging": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "threshold": {"type": "number", "minimum": 0},
                                "hedge_method": {"type": "string", "enum": ["STRADDLE", "STRANGLE", "BUTTERFLY"]}
                            }
                        },
                        "vega_hedging": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "threshold": {"type": "number", "minimum": 0},
                                "vix_hedge": {"type": "boolean"}
                            }
                        }
                    }
                },
                "entry_rules": {
                    "type": "object",
                    "properties": {
                        "entry_time_start": {"type": "string", "format": "time"},
                        "entry_time_end": {"type": "string", "format": "time"},
                        "iv_conditions": {
                            "type": "object",
                            "properties": {
                                "min_iv_percentile": {"type": "number", "minimum": 0, "maximum": 100},
                                "max_iv_percentile": {"type": "number", "minimum": 0, "maximum": 100},
                                "iv_rank_threshold": {"type": "number", "minimum": 0, "maximum": 100}
                            }
                        },
                        "market_conditions": {
                            "type": "object",
                            "properties": {
                                "trend_filter": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL", "ANY"]},
                                "volatility_regime": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "ANY"]},
                                "avoid_events": {"type": "boolean"}
                            }
                        }
                    }
                },
                "exit_rules": {
                    "type": "object",
                    "properties": {
                        "profit_target": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["PERCENTAGE", "POINTS", "PREMIUM_BASED"]},
                                "value": {"type": "number", "minimum": 0}
                            }
                        },
                        "stop_loss": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["PERCENTAGE", "POINTS", "PREMIUM_BASED", "GREEK_BASED"]},
                                "value": {"type": "number", "minimum": 0}
                            }
                        },
                        "time_based": {
                            "type": "object",
                            "properties": {
                                "days_to_expiry_exit": {"type": "integer", "minimum": 0},
                                "theta_decay_exit": {"type": "boolean"},
                                "exit_time": {"type": "string", "format": "time"}
                            }
                        },
                        "greek_based_exits": {
                            "type": "object",
                            "properties": {
                                "delta_breach": {"type": "boolean"},
                                "gamma_breach": {"type": "boolean"},
                                "vega_breach": {"type": "boolean"}
                            }
                        }
                    }
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "position_sizing": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["FIXED", "KELLY", "RISK_PARITY", "GREEK_BASED"]},
                                "base_size": {"type": "number", "minimum": 0},
                                "max_risk_per_position": {"type": "number", "minimum": 0}
                            }
                        },
                        "portfolio_greeks": {
                            "type": "object",
                            "properties": {
                                "max_portfolio_delta": {"type": "number", "minimum": 0},
                                "max_portfolio_gamma": {"type": "number", "minimum": 0},
                                "max_portfolio_vega": {"type": "number", "minimum": 0},
                                "max_portfolio_theta": {"type": "number", "maximum": 0}
                            }
                        },
                        "margin_management": {
                            "type": "object",
                            "properties": {
                                "max_margin_utilization": {"type": "number", "minimum": 0, "maximum": 1},
                                "margin_buffer": {"type": "number", "minimum": 0},
                                "span_calculation": {"type": "boolean"}
                            }
                        }
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for POS configuration"""
        return {
            "greeks_settings": {
                "calculation_method": "BLACK_SCHOLES",
                "update_frequency": "1min",
                "implied_volatility": {
                    "source": "MARKET",
                    "adjustment_factor": 1.0,
                    "smile_adjustment": True,
                    "term_structure": True
                },
                "thresholds": {
                    "delta": {
                        "min": -0.7,
                        "max": 0.7,
                        "warning": 0.5
                    },
                    "gamma": {
                        "max": 0.1,
                        "warning": 0.05
                    },
                    "vega": {
                        "max": 1000,
                        "warning": 500
                    },
                    "theta": {
                        "min": -500,
                        "warning": -300
                    }
                },
                "second_order_greeks": {
                    "calculate_vanna": True,
                    "calculate_charm": True,
                    "calculate_vomma": False,
                    "calculate_speed": False
                }
            },
            "position_structure": {
                "strategy_type": "SPREAD",
                "legs": [
                    {
                        "position": "SELL",
                        "option_type": "CE",
                        "strike_selection": {
                            "method": "OTM",
                            "offset": 1,
                            "delta_target": 0.3
                        },
                        "quantity_ratio": 1
                    },
                    {
                        "position": "SELL",
                        "option_type": "PE",
                        "strike_selection": {
                            "method": "OTM",
                            "offset": 1,
                            "delta_target": 0.3
                        },
                        "quantity_ratio": 1
                    }
                ],
                "expiry_selection": {
                    "method": "CURRENT_WEEK",
                    "days_to_expiry": 7,
                    "roll_days_before": 1,
                    "avoid_expiry_day": True
                },
                "adjustment_rules": {
                    "enabled": True,
                    "adjustment_triggers": [
                        {"type": "DELTA_BREACH", "threshold": 0.5},
                        {"type": "PROFIT_TARGET", "threshold": 0.5}
                    ]
                }
            },
            "hedging_parameters": {
                "delta_hedging": {
                    "enabled": False,
                    "threshold": 0.1,
                    "hedge_instrument": "FUTURES",
                    "rebalance_frequency": "THRESHOLD_BASED",
                    "transaction_cost": 0.0005
                },
                "gamma_hedging": {
                    "enabled": False,
                    "threshold": 0.05,
                    "hedge_method": "STRADDLE",
                    "hedge_ratio": 0.5
                },
                "vega_hedging": {
                    "enabled": False,
                    "threshold": 500,
                    "vix_hedge": False,
                    "hedge_ratio": 0.3
                },
                "tail_risk_hedging": {
                    "enabled": False,
                    "method": "FAR_OTM_PUTS",
                    "allocation_percentage": 2
                }
            },
            "entry_rules": {
                "entry_time_start": "09:30:00",
                "entry_time_end": "15:00:00",
                "iv_conditions": {
                    "min_iv_percentile": 20,
                    "max_iv_percentile": 80,
                    "iv_rank_threshold": 30,
                    "iv_term_structure": "NORMAL"
                },
                "market_conditions": {
                    "trend_filter": "ANY",
                    "volatility_regime": "ANY",
                    "avoid_events": True,
                    "event_days_buffer": 2
                },
                "technical_filters": {
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "use_support_resistance": True
                },
                "position_limits": {
                    "max_positions": 5,
                    "max_positions_per_expiry": 3,
                    "max_positions_per_underlying": 2
                }
            },
            "exit_rules": {
                "profit_target": {
                    "type": "PERCENTAGE",
                    "value": 50,
                    "partial_exit": {
                        "enabled": True,
                        "levels": [
                            {"profit_percentage": 25, "exit_percentage": 50}
                        ]
                    }
                },
                "stop_loss": {
                    "type": "PERCENTAGE",
                    "value": 30,
                    "trailing_stop": {
                        "enabled": False,
                        "activation": 20,
                        "trail_percentage": 10
                    }
                },
                "time_based": {
                    "days_to_expiry_exit": 1,
                    "theta_decay_exit": True,
                    "theta_threshold": -100,
                    "exit_time": "15:15:00",
                    "friday_early_exit": True
                },
                "greek_based_exits": {
                    "delta_breach": True,
                    "gamma_breach": True,
                    "vega_breach": False,
                    "exit_on_iv_spike": True,
                    "iv_spike_threshold": 50
                },
                "market_based_exits": {
                    "exit_on_gap": True,
                    "gap_threshold": 2,
                    "exit_on_trend_reversal": False
                }
            },
            "risk_management": {
                "position_sizing": {
                    "method": "FIXED",
                    "base_size": 1,
                    "max_risk_per_position": 5000,
                    "kelly_fraction": 0.25,
                    "vol_adjustment": True
                },
                "portfolio_greeks": {
                    "max_portfolio_delta": 5,
                    "max_portfolio_gamma": 0.5,
                    "max_portfolio_vega": 5000,
                    "max_portfolio_theta": -2000,
                    "rebalance_frequency": "DAILY"
                },
                "margin_management": {
                    "max_margin_utilization": 0.6,
                    "margin_buffer": 0.2,
                    "span_calculation": True,
                    "stress_test_scenarios": True
                },
                "concentration_limits": {
                    "max_single_position_risk": 0.2,
                    "max_sector_exposure": 0.4,
                    "max_expiry_concentration": 0.5
                },
                "drawdown_control": {
                    "max_drawdown": 0.15,
                    "consecutive_loss_limit": 3,
                    "pause_after_drawdown": True,
                    "recovery_period_days": 2
                }
            },
            "monitoring": {
                "real_time_greeks": True,
                "alert_thresholds": {
                    "delta_alert": 0.6,
                    "gamma_alert": 0.08,
                    "vega_alert": 800,
                    "margin_alert": 0.8
                },
                "performance_metrics": [
                    "sharpe_ratio",
                    "sortino_ratio",
                    "calmar_ratio",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor"
                ],
                "reporting_frequency": "DAILY"
            }
        }
    
    def _validate_greeks_settings(self) -> List[str]:
        """Validate Greeks settings"""
        errors = []
        greeks = self.get("greeks_settings", {})
        
        # Validate thresholds
        thresholds = greeks.get("thresholds", {})
        
        # Delta validation
        delta = thresholds.get("delta", {})
        if delta:
            min_delta = delta.get("min", -1)
            max_delta = delta.get("max", 1)
            
            if not -1 <= min_delta <= 0:
                errors.append("Minimum delta must be between -1 and 0")
            if not 0 <= max_delta <= 1:
                errors.append("Maximum delta must be between 0 and 1")
            if min_delta > max_delta:
                errors.append("Minimum delta cannot be greater than maximum delta")
        
        # Gamma validation
        gamma = thresholds.get("gamma", {})
        if gamma and gamma.get("max", 0) < 0:
            errors.append("Maximum gamma cannot be negative")
        
        # Vega validation
        vega = thresholds.get("vega", {})
        if vega and vega.get("max", 0) < 0:
            errors.append("Maximum vega cannot be negative")
        
        # Theta validation
        theta = thresholds.get("theta", {})
        if theta and theta.get("min", 0) > 0:
            errors.append("Minimum theta should be negative or zero")
        
        return errors
    
    def _validate_position_structure(self) -> List[str]:
        """Validate position structure"""
        errors = []
        structure = self.get("position_structure", {})
        
        # Validate strategy type and legs consistency
        strategy_type = structure.get("strategy_type")
        legs = structure.get("legs", [])
        
        if not legs:
            errors.append("At least one leg required")
        
        # Validate leg configurations
        total_buy_ratio = 0
        total_sell_ratio = 0
        
        for i, leg in enumerate(legs):
            if not leg.get("position"):
                errors.append(f"Leg {i+1}: Position (BUY/SELL) required")
            if not leg.get("option_type"):
                errors.append(f"Leg {i+1}: Option type (CE/PE) required")
            
            ratio = leg.get("quantity_ratio", 1)
            if ratio <= 0:
                errors.append(f"Leg {i+1}: Quantity ratio must be positive")
            
            if leg.get("position") == "BUY":
                total_buy_ratio += ratio
            else:
                total_sell_ratio += ratio
        
        # Strategy-specific validations
        if strategy_type == "SPREAD" and len(legs) < 2:
            errors.append("Spread strategy requires at least 2 legs")
        elif strategy_type == "BUTTERFLY" and len(legs) != 3:
            errors.append("Butterfly strategy requires exactly 3 legs")
        elif strategy_type == "CONDOR" and len(legs) != 4:
            errors.append("Condor strategy requires exactly 4 legs")
        
        # Validate expiry selection
        expiry = structure.get("expiry_selection", {})
        days_to_expiry = expiry.get("days_to_expiry", 0)
        if days_to_expiry <= 0:
            errors.append("Days to expiry must be positive")
        
        return errors
    
    def _validate_hedging_parameters(self) -> List[str]:
        """Validate hedging parameters"""
        errors = []
        hedging = self.get("hedging_parameters", {})
        
        # Validate delta hedging
        delta_hedge = hedging.get("delta_hedging", {})
        if delta_hedge.get("enabled") and delta_hedge.get("threshold", 0) < 0:
            errors.append("Delta hedging threshold cannot be negative")
        
        # Validate gamma hedging
        gamma_hedge = hedging.get("gamma_hedging", {})
        if gamma_hedge.get("enabled") and gamma_hedge.get("threshold", 0) < 0:
            errors.append("Gamma hedging threshold cannot be negative")
        
        # Validate vega hedging
        vega_hedge = hedging.get("vega_hedging", {})
        if vega_hedge.get("enabled") and vega_hedge.get("threshold", 0) < 0:
            errors.append("Vega hedging threshold cannot be negative")
        
        return errors
    
    def _validate_risk_management(self) -> List[str]:
        """Validate risk management settings"""
        errors = []
        risk = self.get("risk_management", {})
        
        # Validate position sizing
        sizing = risk.get("position_sizing", {})
        if sizing.get("base_size", 0) <= 0:
            errors.append("Base position size must be positive")
        
        max_risk = sizing.get("max_risk_per_position", 0)
        if max_risk < 0:
            errors.append("Maximum risk per position cannot be negative")
        
        # Validate portfolio Greeks limits
        portfolio_greeks = risk.get("portfolio_greeks", {})
        if portfolio_greeks:
            if portfolio_greeks.get("max_portfolio_delta", 0) < 0:
                errors.append("Maximum portfolio delta cannot be negative")
            if portfolio_greeks.get("max_portfolio_gamma", 0) < 0:
                errors.append("Maximum portfolio gamma cannot be negative")
            if portfolio_greeks.get("max_portfolio_vega", 0) < 0:
                errors.append("Maximum portfolio vega cannot be negative")
            if portfolio_greeks.get("max_portfolio_theta", 0) > 0:
                errors.append("Maximum portfolio theta should be negative or zero")
        
        # Validate margin management
        margin = risk.get("margin_management", {})
        utilization = margin.get("max_margin_utilization", 0)
        if not 0 < utilization <= 1:
            errors.append("Maximum margin utilization must be between 0 and 1")
        
        return errors
    
    def get_strategy_legs(self) -> List[Dict[str, Any]]:
        """Get configured strategy legs"""
        return self.get("position_structure.legs", [])
    
    def get_greek_thresholds(self) -> Dict[str, Any]:
        """Get Greek thresholds"""
        return self.get("greeks_settings.thresholds", {})
    
    def get_hedging_config(self) -> Dict[str, Any]:
        """Get hedging configuration"""
        return self.get("hedging_parameters", {})
    
    def get_position_size(self, portfolio_value: float, current_risk: float) -> int:
        """
        Calculate position size based on risk management rules
        
        Args:
            portfolio_value: Current portfolio value
            current_risk: Current portfolio risk
            
        Returns:
            Number of lots
        """
        sizing = self.get("risk_management.position_sizing", {})
        method = sizing.get("method", "FIXED")
        base_size = sizing.get("base_size", 1)
        max_risk = sizing.get("max_risk_per_position", 5000)
        
        if method == "FIXED":
            return int(base_size)
        elif method == "RISK_PARITY":
            # Calculate based on equal risk allocation
            risk_allocation = portfolio_value * 0.02  # 2% risk per position
            position_size = risk_allocation / max_risk
            return max(1, int(position_size))
        elif method == "KELLY":
            kelly_fraction = sizing.get("kelly_fraction", 0.25)
            return max(1, int(base_size * kelly_fraction))
        
        return int(base_size)
    
    def should_hedge(self, current_greeks: Dict[str, float]) -> Dict[str, bool]:
        """
        Determine which Greeks need hedging
        
        Args:
            current_greeks: Current portfolio Greeks
            
        Returns:
            Dictionary of hedge requirements
        """
        hedging = self.get("hedging_parameters", {})
        thresholds = self.get("greeks_settings.thresholds", {})
        
        hedge_requirements = {}
        
        # Check delta hedging
        delta_hedge = hedging.get("delta_hedging", {})
        if delta_hedge.get("enabled"):
            delta_threshold = delta_hedge.get("threshold", 0.1)
            current_delta = abs(current_greeks.get("delta", 0))
            hedge_requirements["delta"] = current_delta > delta_threshold
        
        # Check gamma hedging
        gamma_hedge = hedging.get("gamma_hedging", {})
        if gamma_hedge.get("enabled"):
            gamma_threshold = gamma_hedge.get("threshold", 0.05)
            current_gamma = abs(current_greeks.get("gamma", 0))
            hedge_requirements["gamma"] = current_gamma > gamma_threshold
        
        # Check vega hedging
        vega_hedge = hedging.get("vega_hedging", {})
        if vega_hedge.get("enabled"):
            vega_threshold = vega_hedge.get("threshold", 500)
            current_vega = abs(current_greeks.get("vega", 0))
            hedge_requirements["vega"] = current_vega > vega_threshold
        
        return hedge_requirements
    
    def get_exit_conditions(self) -> Dict[str, Any]:
        """Get all exit conditions"""
        return self.get("exit_rules", {})
    
    def __str__(self) -> str:
        """String representation"""
        strategy_type = self.get("position_structure.strategy_type", "UNKNOWN")
        return f"POS Configuration: {self.strategy_name} (Strategy: {strategy_type})"