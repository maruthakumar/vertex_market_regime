"""
Strategy Consolidation Configuration
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import time, datetime
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class StrategyConsolidationConfiguration(BaseConfiguration):
    """
    Configuration for Strategy Consolidation and Optimization
    
    This configuration handles unified strategy management including:
    - Combining multiple strategies with dynamic weights
    - Performance-based weight allocation
    - Cross-strategy risk management
    - Portfolio-level optimization
    - Strategy correlation analysis
    - Adaptive parameter tuning
    """
    
    def __init__(self, strategy_name: str):
        """Initialize Strategy Consolidation configuration"""
        super().__init__("strategy_consolidation", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate Strategy Consolidation configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate strategy configurations
        strategy_errors = self._validate_strategies()
        if strategy_errors:
            errors['strategies'] = strategy_errors
        
        # Validate consolidation rules
        consolidation_errors = self._validate_consolidation_rules()
        if consolidation_errors:
            errors['consolidation_rules'] = consolidation_errors
        
        # Validate optimization settings
        optimization_errors = self._validate_optimization()
        if optimization_errors:
            errors['optimization'] = optimization_errors
        
        # Validate portfolio management
        portfolio_errors = self._validate_portfolio_management()
        if portfolio_errors:
            errors['portfolio_management'] = portfolio_errors
        
        if errors:
            raise ValidationError("Strategy Consolidation configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Strategy Consolidation configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["strategies", "consolidation_rules", "optimization", "portfolio_management"],
            "properties": {
                "strategies": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "required": ["strategy_id", "strategy_type", "enabled", "weight"],
                        "properties": {
                            "strategy_id": {"type": "string"},
                            "strategy_type": {
                                "type": "string",
                                "enum": ["TBS", "TV", "ORB", "OI", "ML", "POS", "MARKET_REGIME", "INDICATOR", "ML_TRIPLE_STRADDLE"]
                            },
                            "enabled": {"type": "boolean"},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "config_file": {"type": "string"},
                            "performance_metrics": {
                                "type": "object",
                                "properties": {
                                    "sharpe_ratio": {"type": "number"},
                                    "win_rate": {"type": "number", "minimum": 0, "maximum": 1},
                                    "profit_factor": {"type": "number", "minimum": 0},
                                    "max_drawdown": {"type": "number", "minimum": 0, "maximum": 1},
                                    "avg_return": {"type": "number"}
                                }
                            },
                            "constraints": {
                                "type": "object",
                                "properties": {
                                    "max_allocation": {"type": "number", "minimum": 0, "maximum": 1},
                                    "min_allocation": {"type": "number", "minimum": 0, "maximum": 1},
                                    "correlation_limit": {"type": "number", "minimum": -1, "maximum": 1},
                                    "conditional_activation": {
                                        "type": "object",
                                        "properties": {
                                            "market_condition": {"type": "string"},
                                            "volatility_range": {
                                                "type": "object",
                                                "properties": {
                                                    "min": {"type": "number", "minimum": 0},
                                                    "max": {"type": "number", "minimum": 0}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "consolidation_rules": {
                    "type": "object",
                    "required": ["weight_allocation", "signal_aggregation", "conflict_resolution"],
                    "properties": {
                        "weight_allocation": {
                            "type": "object",
                            "properties": {
                                "method": {
                                    "type": "string",
                                    "enum": ["FIXED", "DYNAMIC", "PERFORMANCE_BASED", "RISK_PARITY", "EQUAL_WEIGHT", "OPTIMIZATION_BASED"]
                                },
                                "update_frequency": {"type": "string", "enum": ["REALTIME", "DAILY", "WEEKLY", "MONTHLY"]},
                                "lookback_period": {"type": "integer", "minimum": 1, "maximum": 252},
                                "min_performance_period": {"type": "integer", "minimum": 20},
                                "rebalancing_threshold": {"type": "number", "minimum": 0, "maximum": 0.5}
                            }
                        },
                        "signal_aggregation": {
                            "type": "object",
                            "properties": {
                                "method": {
                                    "type": "string",
                                    "enum": ["MAJORITY_VOTE", "WEIGHTED_AVERAGE", "UNANIMOUS", "THRESHOLD_BASED", "ML_ENSEMBLE"]
                                },
                                "min_strategies_agreement": {"type": "integer", "minimum": 1},
                                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                "signal_strength_calculation": {"type": "string", "enum": ["AVERAGE", "WEIGHTED", "MAX", "MEDIAN"]}
                            }
                        },
                        "conflict_resolution": {
                            "type": "object",
                            "properties": {
                                "method": {
                                    "type": "string",
                                    "enum": ["HIGHEST_WEIGHT", "HIGHEST_CONFIDENCE", "MOST_RECENT", "PERFORMANCE_BASED", "CUSTOM_RULES"]
                                },
                                "priority_order": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "veto_strategies": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "correlation_management": {
                            "type": "object",
                            "properties": {
                                "max_strategy_correlation": {"type": "number", "minimum": -1, "maximum": 1},
                                "correlation_window": {"type": "integer", "minimum": 20},
                                "decorrelation_method": {"type": "string", "enum": ["REDUCE_WEIGHTS", "DISABLE_STRATEGY", "ADJUST_TIMING"]}
                            }
                        }
                    }
                },
                "optimization": {
                    "type": "object",
                    "required": ["optimization_method", "objective_function", "constraints"],
                    "properties": {
                        "optimization_method": {
                            "type": "string",
                            "enum": ["GRID_SEARCH", "RANDOM_SEARCH", "BAYESIAN", "GENETIC_ALGORITHM", "GRADIENT_BASED", "NONE"]
                        },
                        "objective_function": {
                            "type": "string",
                            "enum": ["SHARPE_RATIO", "SORTINO_RATIO", "CALMAR_RATIO", "PROFIT_FACTOR", "CUSTOM", "MULTI_OBJECTIVE"]
                        },
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "min_sharpe_ratio": {"type": "number"},
                                "max_drawdown": {"type": "number", "minimum": 0, "maximum": 1},
                                "min_win_rate": {"type": "number", "minimum": 0, "maximum": 1},
                                "max_volatility": {"type": "number", "minimum": 0},
                                "min_trades_per_day": {"type": "integer", "minimum": 0}
                            }
                        },
                        "parameter_ranges": {
                            "type": "object",
                            "properties": {
                                "weight_range": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number", "minimum": 0, "maximum": 1},
                                        "max": {"type": "number", "minimum": 0, "maximum": 1},
                                        "step": {"type": "number", "minimum": 0.01}
                                    }
                                }
                            }
                        },
                        "optimization_settings": {
                            "type": "object",
                            "properties": {
                                "n_iterations": {"type": "integer", "minimum": 10},
                                "n_jobs": {"type": "integer", "minimum": -1},
                                "random_state": {"type": "integer"},
                                "convergence_tolerance": {"type": "number", "minimum": 0},
                                "early_stopping": {"type": "boolean"},
                                "patience": {"type": "integer", "minimum": 1}
                            }
                        }
                    }
                },
                "portfolio_management": {
                    "type": "object",
                    "required": ["risk_limits", "position_sizing", "execution"],
                    "properties": {
                        "risk_limits": {
                            "type": "object",
                            "properties": {
                                "max_portfolio_risk": {"type": "number", "minimum": 0, "maximum": 1},
                                "max_strategy_risk": {"type": "number", "minimum": 0, "maximum": 1},
                                "max_correlation_risk": {"type": "number", "minimum": 0, "maximum": 1},
                                "var_limit": {"type": "number", "minimum": 0},
                                "stress_test_limit": {"type": "number", "minimum": 0}
                            }
                        },
                        "position_sizing": {
                            "type": "object",
                            "properties": {
                                "method": {
                                    "type": "string",
                                    "enum": ["EQUAL_WEIGHT", "VOLATILITY_PARITY", "RISK_PARITY", "KELLY", "FIXED_FRACTIONAL"]
                                },
                                "base_size_per_strategy": {"type": "number", "minimum": 0, "maximum": 1},
                                "scale_by_confidence": {"type": "boolean"},
                                "scale_by_volatility": {"type": "boolean"},
                                "max_position_per_strategy": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        },
                        "execution": {
                            "type": "object",
                            "properties": {
                                "order_routing": {
                                    "type": "string",
                                    "enum": ["SEQUENTIAL", "PARALLEL", "SMART_ROUTING", "PRIORITY_BASED"]
                                },
                                "execution_lag": {"type": "integer", "minimum": 0},
                                "slippage_model": {"type": "string", "enum": ["LINEAR", "SQUARE_ROOT", "MARKET_IMPACT"]},
                                "partial_fill_handling": {"type": "string", "enum": ["WAIT", "CANCEL", "MARKET_ORDER"]}
                            }
                        },
                        "rebalancing": {
                            "type": "object",
                            "properties": {
                                "frequency": {"type": "string", "enum": ["DAILY", "WEEKLY", "MONTHLY", "THRESHOLD_BASED"]},
                                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                "constraints": {
                                    "type": "object",
                                    "properties": {
                                        "min_trade_size": {"type": "number", "minimum": 0},
                                        "max_turnover": {"type": "number", "minimum": 0, "maximum": 2},
                                        "tax_aware": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                },
                "performance_tracking": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["TOTAL_RETURN", "SHARPE_RATIO", "SORTINO_RATIO", "MAX_DRAWDOWN", 
                                        "WIN_RATE", "PROFIT_FACTOR", "STRATEGY_CONTRIBUTION", "CORRELATION_MATRIX"]
                            }
                        },
                        "attribution": {
                            "type": "object",
                            "properties": {
                                "strategy_level": {"type": "boolean"},
                                "factor_decomposition": {"type": "boolean"},
                                "time_based_analysis": {"type": "boolean"}
                            }
                        },
                        "reporting": {
                            "type": "object",
                            "properties": {
                                "frequency": {"type": "string", "enum": ["REALTIME", "DAILY", "WEEKLY", "MONTHLY"]},
                                "format": {"type": "string", "enum": ["JSON", "CSV", "HTML", "PDF"]},
                                "recipients": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                },
                "adaptive_learning": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "learning_rate": {"type": "number", "minimum": 0, "maximum": 1},
                        "adaptation_method": {
                            "type": "string",
                            "enum": ["ONLINE_LEARNING", "BATCH_LEARNING", "REINFORCEMENT_LEARNING"]
                        },
                        "feature_importance": {"type": "boolean"},
                        "regime_detection": {"type": "boolean"},
                        "parameter_evolution": {"type": "boolean"}
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for Strategy Consolidation configuration"""
        return {
            "strategies": [
                {
                    "strategy_id": "TBS_001",
                    "strategy_type": "TBS",
                    "enabled": True,
                    "weight": 0.25,
                    "config_file": "tbs_config_001.xlsx",
                    "performance_metrics": {
                        "sharpe_ratio": 1.5,
                        "win_rate": 0.55,
                        "profit_factor": 1.8,
                        "max_drawdown": 0.12,
                        "avg_return": 0.002
                    },
                    "constraints": {
                        "max_allocation": 0.40,
                        "min_allocation": 0.10,
                        "correlation_limit": 0.7,
                        "conditional_activation": {
                            "market_condition": "TRENDING",
                            "volatility_range": {
                                "min": 0.10,
                                "max": 0.30
                            }
                        }
                    }
                },
                {
                    "strategy_id": "ORB_001",
                    "strategy_type": "ORB",
                    "enabled": True,
                    "weight": 0.20,
                    "config_file": "orb_config_001.xlsx",
                    "performance_metrics": {
                        "sharpe_ratio": 1.2,
                        "win_rate": 0.52,
                        "profit_factor": 1.6,
                        "max_drawdown": 0.15,
                        "avg_return": 0.0015
                    },
                    "constraints": {
                        "max_allocation": 0.30,
                        "min_allocation": 0.05,
                        "correlation_limit": 0.5,
                        "conditional_activation": {
                            "market_condition": "RANGING",
                            "volatility_range": {
                                "min": 0.05,
                                "max": 0.20
                            }
                        }
                    }
                },
                {
                    "strategy_id": "ML_001",
                    "strategy_type": "ML_TRIPLE_STRADDLE",
                    "enabled": True,
                    "weight": 0.35,
                    "config_file": "ml_triple_straddle_001.xlsx",
                    "performance_metrics": {
                        "sharpe_ratio": 2.0,
                        "win_rate": 0.62,
                        "profit_factor": 2.2,
                        "max_drawdown": 0.10,
                        "avg_return": 0.003
                    },
                    "constraints": {
                        "max_allocation": 0.50,
                        "min_allocation": 0.20,
                        "correlation_limit": 0.6,
                        "conditional_activation": None
                    }
                },
                {
                    "strategy_id": "MR_001",
                    "strategy_type": "MARKET_REGIME",
                    "enabled": True,
                    "weight": 0.20,
                    "config_file": "market_regime_001.xlsx",
                    "performance_metrics": {
                        "sharpe_ratio": 1.8,
                        "win_rate": 0.58,
                        "profit_factor": 2.0,
                        "max_drawdown": 0.08,
                        "avg_return": 0.0025
                    },
                    "constraints": {
                        "max_allocation": 0.35,
                        "min_allocation": 0.10,
                        "correlation_limit": 0.5,
                        "conditional_activation": None
                    }
                }
            ],
            "consolidation_rules": {
                "weight_allocation": {
                    "method": "PERFORMANCE_BASED",
                    "update_frequency": "WEEKLY",
                    "lookback_period": 60,
                    "min_performance_period": 30,
                    "rebalancing_threshold": 0.05,
                    "weight_bounds": {
                        "min": 0.05,
                        "max": 0.50
                    }
                },
                "signal_aggregation": {
                    "method": "WEIGHTED_AVERAGE",
                    "min_strategies_agreement": 2,
                    "confidence_threshold": 0.60,
                    "signal_strength_calculation": "WEIGHTED",
                    "consensus_required": False,
                    "signal_decay": 0.95
                },
                "conflict_resolution": {
                    "method": "PERFORMANCE_BASED",
                    "priority_order": ["ML_TRIPLE_STRADDLE", "MARKET_REGIME", "TBS", "ORB"],
                    "veto_strategies": [],
                    "conflict_threshold": 0.3,
                    "resolution_window": 5
                },
                "correlation_management": {
                    "max_strategy_correlation": 0.7,
                    "correlation_window": 60,
                    "decorrelation_method": "REDUCE_WEIGHTS",
                    "correlation_check_frequency": "DAILY",
                    "correlation_matrix_update": "ROLLING"
                }
            },
            "optimization": {
                "optimization_method": "BAYESIAN",
                "objective_function": "SHARPE_RATIO",
                "constraints": {
                    "min_sharpe_ratio": 1.0,
                    "max_drawdown": 0.20,
                    "min_win_rate": 0.50,
                    "max_volatility": 0.25,
                    "min_trades_per_day": 2,
                    "max_correlation": 0.8
                },
                "parameter_ranges": {
                    "weight_range": {
                        "min": 0.05,
                        "max": 0.50,
                        "step": 0.05
                    },
                    "lookback_range": {
                        "min": 20,
                        "max": 120,
                        "step": 10
                    },
                    "threshold_range": {
                        "min": 0.50,
                        "max": 0.90,
                        "step": 0.05
                    }
                },
                "optimization_settings": {
                    "n_iterations": 100,
                    "n_jobs": -1,
                    "random_state": 42,
                    "convergence_tolerance": 0.001,
                    "early_stopping": True,
                    "patience": 10,
                    "cv_folds": 5,
                    "warm_start": True
                }
            },
            "portfolio_management": {
                "risk_limits": {
                    "max_portfolio_risk": 0.25,
                    "max_strategy_risk": 0.10,
                    "max_correlation_risk": 0.15,
                    "var_limit": 0.05,
                    "stress_test_limit": 0.30,
                    "max_leverage": 2.0,
                    "margin_buffer": 0.20
                },
                "position_sizing": {
                    "method": "RISK_PARITY",
                    "base_size_per_strategy": 0.25,
                    "scale_by_confidence": True,
                    "scale_by_volatility": True,
                    "max_position_per_strategy": 0.40,
                    "min_position_size": 0.01,
                    "position_rounding": "LOT"
                },
                "execution": {
                    "order_routing": "SMART_ROUTING",
                    "execution_lag": 100,  # milliseconds
                    "slippage_model": "SQUARE_ROOT",
                    "partial_fill_handling": "WAIT",
                    "max_order_size": 0.05,
                    "order_splitting": True,
                    "iceberg_orders": False
                },
                "rebalancing": {
                    "frequency": "WEEKLY",
                    "threshold": 0.05,
                    "constraints": {
                        "min_trade_size": 0.001,
                        "max_turnover": 0.50,
                        "tax_aware": False,
                        "transaction_cost": 0.001
                    },
                    "rebalance_time": "09:30:00",
                    "emergency_rebalance": True
                }
            },
            "performance_tracking": {
                "metrics": [
                    "TOTAL_RETURN", "SHARPE_RATIO", "SORTINO_RATIO", 
                    "MAX_DRAWDOWN", "WIN_RATE", "PROFIT_FACTOR", 
                    "STRATEGY_CONTRIBUTION", "CORRELATION_MATRIX"
                ],
                "attribution": {
                    "strategy_level": True,
                    "factor_decomposition": True,
                    "time_based_analysis": True,
                    "regime_based_analysis": True
                },
                "reporting": {
                    "frequency": "DAILY",
                    "format": "JSON",
                    "recipients": [],
                    "include_charts": True,
                    "include_recommendations": True
                },
                "alerts": {
                    "drawdown_alert": 0.10,
                    "correlation_alert": 0.85,
                    "performance_deviation": 0.20,
                    "strategy_failure": True
                }
            },
            "adaptive_learning": {
                "enabled": True,
                "learning_rate": 0.01,
                "adaptation_method": "ONLINE_LEARNING",
                "feature_importance": True,
                "regime_detection": True,
                "parameter_evolution": True,
                "adaptation_frequency": "DAILY",
                "min_data_points": 100,
                "confidence_threshold": 0.95
            },
            "monitoring": {
                "health_checks": True,
                "strategy_diagnostics": True,
                "system_metrics": True,
                "alert_channels": ["email", "slack"],
                "monitoring_frequency": "REALTIME"
            }
        }
    
    def _validate_strategies(self) -> List[str]:
        """Validate strategy configurations"""
        errors = []
        strategies = self.get("strategies", [])
        
        if len(strategies) < 2:
            errors.append("At least 2 strategies required for consolidation")
            return errors
        
        # Track strategy IDs for uniqueness
        strategy_ids = []
        total_weight = 0
        enabled_count = 0
        
        for strategy in strategies:
            strategy_id = strategy.get("strategy_id", "")
            
            # Check for duplicate IDs
            if strategy_id in strategy_ids:
                errors.append(f"Duplicate strategy ID: {strategy_id}")
            strategy_ids.append(strategy_id)
            
            # Sum weights for enabled strategies
            if strategy.get("enabled", False):
                enabled_count += 1
                weight = strategy.get("weight", 0)
                total_weight += weight
                
                # Validate constraints
                constraints = strategy.get("constraints", {})
                max_alloc = constraints.get("max_allocation", 1)
                min_alloc = constraints.get("min_allocation", 0)
                
                if min_alloc > max_alloc:
                    errors.append(f"{strategy_id}: min_allocation > max_allocation")
                
                if weight > max_alloc:
                    errors.append(f"{strategy_id}: weight exceeds max_allocation")
                
                if weight < min_alloc:
                    errors.append(f"{strategy_id}: weight below min_allocation")
        
        if enabled_count < 2:
            errors.append("At least 2 strategies must be enabled")
        
        if enabled_count > 0 and abs(total_weight - 1.0) > 0.01:
            errors.append(f"Enabled strategy weights must sum to 1.0, got {total_weight}")
        
        return errors
    
    def _validate_consolidation_rules(self) -> List[str]:
        """Validate consolidation rules"""
        errors = []
        rules = self.get("consolidation_rules", {})
        
        # Validate weight allocation
        weight_alloc = rules.get("weight_allocation", {})
        method = weight_alloc.get("method", "")
        
        if method == "PERFORMANCE_BASED":
            lookback = weight_alloc.get("lookback_period", 0)
            min_period = weight_alloc.get("min_performance_period", 0)
            
            if min_period > lookback:
                errors.append("min_performance_period cannot exceed lookback_period")
        
        # Validate signal aggregation
        signal_agg = rules.get("signal_aggregation", {})
        min_agreement = signal_agg.get("min_strategies_agreement", 0)
        enabled_strategies = len([s for s in self.get("strategies", []) if s.get("enabled")])
        
        if min_agreement > enabled_strategies:
            errors.append(f"min_strategies_agreement ({min_agreement}) exceeds enabled strategies ({enabled_strategies})")
        
        # Validate correlation management
        corr_mgmt = rules.get("correlation_management", {})
        max_corr = corr_mgmt.get("max_strategy_correlation", 1)
        
        if not -1 <= max_corr <= 1:
            errors.append("max_strategy_correlation must be between -1 and 1")
        
        return errors
    
    def _validate_optimization(self) -> List[str]:
        """Validate optimization settings"""
        errors = []
        optimization = self.get("optimization", {})
        
        # Validate constraints
        constraints = optimization.get("constraints", {})
        max_dd = constraints.get("max_drawdown", 0)
        
        if max_dd < 0 or max_dd > 1:
            errors.append("max_drawdown must be between 0 and 1")
        
        # Validate parameter ranges
        param_ranges = optimization.get("parameter_ranges", {})
        weight_range = param_ranges.get("weight_range", {})
        
        if weight_range:
            w_min = weight_range.get("min", 0)
            w_max = weight_range.get("max", 1)
            
            if w_min >= w_max:
                errors.append("weight_range min must be less than max")
            
            if w_min < 0 or w_max > 1:
                errors.append("weight_range must be between 0 and 1")
        
        # Validate optimization settings
        settings = optimization.get("optimization_settings", {})
        n_iter = settings.get("n_iterations", 0)
        
        if n_iter < 10:
            errors.append("n_iterations should be at least 10 for meaningful optimization")
        
        return errors
    
    def _validate_portfolio_management(self) -> List[str]:
        """Validate portfolio management settings"""
        errors = []
        portfolio = self.get("portfolio_management", {})
        
        # Validate risk limits
        risk_limits = portfolio.get("risk_limits", {})
        max_portfolio_risk = risk_limits.get("max_portfolio_risk", 0)
        max_strategy_risk = risk_limits.get("max_strategy_risk", 0)
        
        if max_strategy_risk > max_portfolio_risk:
            errors.append("max_strategy_risk cannot exceed max_portfolio_risk")
        
        # Validate position sizing
        pos_sizing = portfolio.get("position_sizing", {})
        base_size = pos_sizing.get("base_size_per_strategy", 0)
        max_pos = pos_sizing.get("max_position_per_strategy", 0)
        
        if base_size > max_pos:
            errors.append("base_size_per_strategy cannot exceed max_position_per_strategy")
        
        # Validate rebalancing
        rebalancing = portfolio.get("rebalancing", {})
        max_turnover = rebalancing.get("constraints", {}).get("max_turnover", 0)
        
        if max_turnover < 0 or max_turnover > 2:
            errors.append("max_turnover should be between 0 and 2 (200% turnover)")
        
        return errors
    
    def get_enabled_strategies(self) -> List[Dict[str, Any]]:
        """Get list of enabled strategies"""
        strategies = self.get("strategies", [])
        return [s for s in strategies if s.get("enabled", False)]
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get strategy weights"""
        strategies = self.get_enabled_strategies()
        return {s["strategy_id"]: s["weight"] for s in strategies}
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration"""
        return self.get("optimization", {})
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get portfolio risk limits"""
        return self.get("portfolio_management.risk_limits", {})
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        strategies = self.get_enabled_strategies()
        
        if not strategies:
            return {}
        
        # Calculate weighted averages
        total_weight = sum(s["weight"] for s in strategies)
        weighted_sharpe = sum(s["weight"] * s.get("performance_metrics", {}).get("sharpe_ratio", 0) 
                             for s in strategies) / total_weight
        weighted_return = sum(s["weight"] * s.get("performance_metrics", {}).get("avg_return", 0) 
                             for s in strategies) / total_weight
        
        # Max drawdown is not additive, take weighted max
        max_drawdown = max(s.get("performance_metrics", {}).get("max_drawdown", 0) 
                          for s in strategies)
        
        return {
            "portfolio_sharpe_ratio": weighted_sharpe,
            "portfolio_return": weighted_return,
            "portfolio_max_drawdown": max_drawdown,
            "strategy_count": len(strategies),
            "total_weight": total_weight
        }
    
    def get_correlation_matrix(self) -> Optional[List[List[float]]]:
        """Get strategy correlation matrix if available"""
        # This would be populated by the system based on historical performance
        return self.get("performance_tracking.correlation_matrix", None)
    
    def __str__(self) -> str:
        """String representation"""
        enabled_count = len(self.get_enabled_strategies())
        method = self.get("consolidation_rules.weight_allocation.method", "UNKNOWN")
        return f"Strategy Consolidation Configuration: {self.strategy_name} ({enabled_count} strategies, {method} allocation)"