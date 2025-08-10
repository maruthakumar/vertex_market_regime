"""
ML Triple Straddle Configuration
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import time, datetime
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class MLTripleStraddleConfiguration(BaseConfiguration):
    """
    Configuration for ML Triple Rolling Straddle Strategy
    
    This configuration handles the most sophisticated ML-based options trading
    strategy with 300+ parameters across 26 configuration sheets including:
    - 5 ML models (LightGBM, CatBoost, TabNet, LSTM, Transformer)
    - 160+ engineered features across 6 categories
    - Professional risk management with Kelly criterion
    - Triple straddle signal generation (ATM/ITM/OTM)
    - HeavyDB integration for 33M+ rows of data
    """
    
    def __init__(self, strategy_name: str):
        """Initialize ML Triple Straddle configuration"""
        super().__init__("ml_triple_straddle", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate ML Triple Straddle configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate ML models
        model_errors = self._validate_ml_models()
        if model_errors:
            errors['ml_models'] = model_errors
        
        # Validate feature engineering
        feature_errors = self._validate_features()
        if feature_errors:
            errors['features'] = feature_errors
        
        # Validate risk management
        risk_errors = self._validate_risk_management()
        if risk_errors:
            errors['risk_management'] = risk_errors
        
        # Validate signal generation
        signal_errors = self._validate_signal_generation()
        if signal_errors:
            errors['signal_generation'] = signal_errors
        
        if errors:
            raise ValidationError("ML Triple Straddle configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for ML Triple Straddle configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["ml_models", "features", "risk_management", "signal_generation"],
            "properties": {
                "ml_models": {
                    "type": "object",
                    "required": ["lightgbm", "catboost", "tabnet", "lstm", "transformer", "ensemble"],
                    "properties": {
                        "lightgbm": {
                            "type": "object",
                            "properties": {
                                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "device_type": {"type": "string", "enum": ["CPU", "GPU"]},
                                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000},
                                "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                                "max_depth": {"type": "integer", "minimum": -1, "maximum": 100},
                                "feature_fraction": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                                "bagging_fraction": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                                "lambda_l1": {"type": "number", "minimum": 0},
                                "lambda_l2": {"type": "number", "minimum": 0},
                                "gpu_use_dp": {"type": "boolean"}
                            }
                        },
                        "catboost": {
                            "type": "object",
                            "properties": {
                                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "task_type": {"type": "string", "enum": ["CPU", "GPU"]},
                                "iterations": {"type": "integer", "minimum": 10, "maximum": 10000},
                                "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                                "depth": {"type": "integer", "minimum": 1, "maximum": 16},
                                "l2_leaf_reg": {"type": "number", "minimum": 0},
                                "auto_class_weights": {"type": "string", "enum": ["None", "Balanced", "SqrtBalanced"]},
                                "has_time": {"type": "boolean"}
                            }
                        },
                        "tabnet": {
                            "type": "object",
                            "properties": {
                                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "n_d": {"type": "integer", "minimum": 8, "maximum": 256},
                                "n_a": {"type": "integer", "minimum": 8, "maximum": 256},
                                "n_steps": {"type": "integer", "minimum": 1, "maximum": 10},
                                "gamma": {"type": "number", "minimum": 1.0, "maximum": 2.0},
                                "n_independent": {"type": "integer", "minimum": 1, "maximum": 5},
                                "n_shared": {"type": "integer", "minimum": 1, "maximum": 5},
                                "mask_type": {"type": "string", "enum": ["sparsemax", "entmax"]},
                                "virtual_batch_size": {"type": "integer", "minimum": 16, "maximum": 512}
                            }
                        },
                        "lstm": {
                            "type": "object",
                            "properties": {
                                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "sequence_length": {"type": "integer", "minimum": 5, "maximum": 100},
                                "units": {"type": "integer", "minimum": 32, "maximum": 512},
                                "layers": {"type": "integer", "minimum": 1, "maximum": 5},
                                "dropout": {"type": "number", "minimum": 0, "maximum": 0.5},
                                "bidirectional": {"type": "boolean"}
                            }
                        },
                        "transformer": {
                            "type": "object",
                            "properties": {
                                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "num_heads": {"type": "integer", "minimum": 2, "maximum": 16},
                                "head_size": {"type": "integer", "minimum": 64, "maximum": 512},
                                "ff_dim": {"type": "integer", "minimum": 128, "maximum": 2048},
                                "num_layers": {"type": "integer", "minimum": 1, "maximum": 6},
                                "dropout": {"type": "number", "minimum": 0, "maximum": 0.5}
                            }
                        },
                        "ensemble": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["weighted_average", "voting", "stacking", "blending"]},
                                "confidence_threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                                "dynamic_weight_adjustment": {"type": "boolean"},
                                "min_model_agreement": {"type": "integer", "minimum": 1, "maximum": 5}
                            }
                        }
                    }
                },
                "features": {
                    "type": "object",
                    "required": ["market_regime", "greeks", "iv", "oi", "technical", "microstructure"],
                    "properties": {
                        "market_regime": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "lookback_period": {"type": "integer", "minimum": 5, "maximum": 60},
                                "normalization": {"type": "string", "enum": ["zscore", "minmax", "robust"]},
                                "update_frequency": {"type": "integer", "minimum": 1, "maximum": 60},
                                "features": {
                                    "type": "object",
                                    "properties": {
                                        "ema_crossover": {"type": "boolean"},
                                        "vwap_deviation": {"type": "boolean"},
                                        "greek_sentiment": {"type": "boolean"},
                                        "iv_skew": {"type": "boolean"},
                                        "oi_flow": {"type": "boolean"},
                                        "regime_transitions": {"type": "boolean"}
                                    }
                                }
                            }
                        },
                        "greeks": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "calculation_method": {"type": "string", "enum": ["black_scholes", "binomial", "monte_carlo"]},
                                "normalization": {"type": "string", "enum": ["minmax", "zscore", "none"]},
                                "smoothing_window": {"type": "integer", "minimum": 1, "maximum": 10},
                                "moneyness_levels": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ["ATM", "ITM", "OTM"]}
                                }
                            }
                        },
                        "iv": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "calculation_window": {"type": "integer", "minimum": 20, "maximum": 252},
                                "smoothing": {"type": "string", "enum": ["exponential", "simple", "none"]},
                                "outlier_handling": {"type": "string", "enum": ["clip", "winsorize", "none"]},
                                "strike_surface": {
                                    "type": "array",
                                    "items": {"type": "integer"}
                                },
                                "term_structure": {
                                    "type": "array",
                                    "items": {"type": "integer"}
                                }
                            }
                        },
                        "oi": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "rolling_window": {"type": "integer", "minimum": 5, "maximum": 60},
                                "change_calculation": {"type": "string", "enum": ["absolute", "relative", "both"]},
                                "filtering": {"type": "string", "enum": ["volume_weighted", "none"]},
                                "smart_money_detection": {"type": "boolean"}
                            }
                        },
                        "technical": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "period_length": {"type": "integer", "minimum": 5, "maximum": 50},
                                "calculation_method": {"type": "string", "enum": ["standard", "adaptive"]},
                                "smoothing": {"type": "boolean"},
                                "indicators": {
                                    "type": "object",
                                    "properties": {
                                        "price_relative": {"type": "boolean"},
                                        "volatility": {"type": "boolean"},
                                        "momentum": {"type": "boolean"},
                                        "trend": {"type": "boolean"},
                                        "support_resistance": {"type": "boolean"}
                                    }
                                }
                            }
                        },
                        "microstructure": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "window": {"type": "integer", "minimum": 1, "maximum": 10},
                                "aggregation": {"type": "string", "enum": ["volume_weighted", "time_weighted"]},
                                "filtering": {"type": "string", "enum": ["outlier_robust", "none"]},
                                "features": {
                                    "type": "object",
                                    "properties": {
                                        "order_flow": {"type": "boolean"},
                                        "spread_analysis": {"type": "boolean"},
                                        "market_depth": {"type": "boolean"},
                                        "trade_analytics": {"type": "boolean"},
                                        "market_impact": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                },
                "risk_management": {
                    "type": "object",
                    "required": ["position_sizing", "risk_limits", "stop_loss", "circuit_breaker"],
                    "properties": {
                        "position_sizing": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["kelly", "fixed", "volatility_based", "risk_parity"]},
                                "kelly_fraction": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                                "max_position_size": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                                "max_positions": {"type": "integer", "minimum": 1, "maximum": 20},
                                "position_correlation_limit": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        },
                        "risk_limits": {
                            "type": "object",
                            "properties": {
                                "max_portfolio_risk": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                                "max_daily_loss": {"type": "number", "minimum": 0.01, "maximum": 0.2},
                                "max_drawdown": {"type": "number", "minimum": 0.05, "maximum": 0.5},
                                "var_confidence_level": {"type": "number", "minimum": 0.9, "maximum": 0.99},
                                "stress_test_scenarios": {"type": "boolean"}
                            }
                        },
                        "stop_loss": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["atr_based", "percentage", "fixed", "dynamic"]},
                                "atr_multiplier": {"type": "number", "minimum": 0.5, "maximum": 5.0},
                                "trailing_stop": {"type": "boolean"},
                                "profit_target_multiple": {"type": "number", "minimum": 1.0, "maximum": 5.0},
                                "max_holding_time": {"type": "integer", "minimum": 1, "maximum": 1440}
                            }
                        },
                        "circuit_breaker": {
                            "type": "object",
                            "properties": {
                                "loss_threshold": {"type": "number", "minimum": 0.01, "maximum": 0.1},
                                "cooldown_period": {"type": "integer", "minimum": 15, "maximum": 240},
                                "volatility_trigger": {"type": "number", "minimum": 0.02, "maximum": 0.2},
                                "manual_override": {"type": "boolean"}
                            }
                        }
                    }
                },
                "signal_generation": {
                    "type": "object",
                    "required": ["straddle_config", "signal_filters", "signal_processing"],
                    "properties": {
                        "straddle_config": {
                            "type": "object",
                            "properties": {
                                "atm": {
                                    "type": "object",
                                    "properties": {
                                        "confidence_threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                                        "weight": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                },
                                "itm": {
                                    "type": "object",
                                    "properties": {
                                        "confidence_threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                                        "weight": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                },
                                "otm": {
                                    "type": "object",
                                    "properties": {
                                        "confidence_threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                                        "weight": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                },
                                "signal_aggregation_window": {"type": "integer", "minimum": 1, "maximum": 30},
                                "consensus_threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                                "multi_timeframe_analysis": {"type": "boolean"}
                            }
                        },
                        "signal_filters": {
                            "type": "object",
                            "properties": {
                                "min_volume": {"type": "integer", "minimum": 100},
                                "max_spread": {"type": "number", "minimum": 0.001, "maximum": 0.05},
                                "market_hours_only": {"type": "boolean"},
                                "min_dte": {"type": "integer", "minimum": 0},
                                "max_dte": {"type": "integer", "minimum": 1, "maximum": 90},
                                "iv_percentile_range": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number", "minimum": 0, "maximum": 100},
                                        "max": {"type": "number", "minimum": 0, "maximum": 100}
                                    }
                                }
                            }
                        },
                        "signal_processing": {
                            "type": "object",
                            "properties": {
                                "signal_latency_target": {"type": "integer", "minimum": 10, "maximum": 1000},
                                "smoothing_window": {"type": "integer", "minimum": 1, "maximum": 10},
                                "outlier_detection": {"type": "boolean"},
                                "cross_validation": {"type": "boolean"},
                                "real_time_processing": {"type": "boolean"}
                            }
                        }
                    }
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "training_period": {
                            "type": "object",
                            "properties": {
                                "start_year": {"type": "integer", "minimum": 2015},
                                "end_year": {"type": "integer", "minimum": 2020}
                            }
                        },
                        "validation_split": {"type": "number", "minimum": 0.1, "maximum": 0.4},
                        "cross_validation_folds": {"type": "integer", "minimum": 2, "maximum": 10},
                        "time_series_split": {"type": "boolean"},
                        "feature_selection": {"type": "string", "enum": ["mutual_info", "importance", "correlation", "all"]},
                        "early_stopping": {"type": "boolean"},
                        "mixed_precision": {"type": "boolean"},
                        "gpu_optimization": {"type": "boolean"},
                        "hyperparameter_optimization": {"type": "string", "enum": ["optuna", "grid_search", "random_search", "none"]}
                    }
                },
                "database": {
                    "type": "object",
                    "properties": {
                        "heavydb": {
                            "type": "object",
                            "properties": {
                                "host": {"type": "string"},
                                "port": {"type": "integer"},
                                "database": {"type": "string"},
                                "gpu_queries": {"type": "boolean"},
                                "connection_pooling": {"type": "boolean"},
                                "query_cache_ttl": {"type": "integer", "minimum": 0}
                            }
                        }
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for ML Triple Straddle configuration"""
        return {
            "ml_models": {
                "lightgbm": {
                    "weight": 0.30,
                    "device_type": "GPU",
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "lambda_l1": 0.1,
                    "lambda_l2": 0.1,
                    "min_child_samples": 20,
                    "num_leaves": 31,
                    "gpu_use_dp": False,
                    "gpu_platform_id": 0,
                    "gpu_device_id": 0
                },
                "catboost": {
                    "weight": 0.25,
                    "task_type": "GPU",
                    "iterations": 1000,
                    "learning_rate": 0.03,
                    "depth": 8,
                    "l2_leaf_reg": 3.0,
                    "border_count": 254,
                    "auto_class_weights": "Balanced",
                    "has_time": True,
                    "gpu_ram_part": 0.5,
                    "max_ctr_complexity": 4
                },
                "tabnet": {
                    "weight": 0.25,
                    "n_d": 64,
                    "n_a": 64,
                    "n_steps": 5,
                    "gamma": 1.5,
                    "n_independent": 2,
                    "n_shared": 2,
                    "mask_type": "entmax",
                    "virtual_batch_size": 128,
                    "momentum": 0.98,
                    "epsilon": 1e-15
                },
                "lstm": {
                    "weight": 0.10,
                    "sequence_length": 20,
                    "units": 128,
                    "layers": 2,
                    "dropout": 0.2,
                    "recurrent_dropout": 0.2,
                    "bidirectional": True,
                    "stateful": False
                },
                "transformer": {
                    "weight": 0.10,
                    "num_heads": 8,
                    "head_size": 256,
                    "ff_dim": 512,
                    "num_layers": 4,
                    "dropout": 0.2,
                    "use_positional_encoding": True
                },
                "ensemble": {
                    "method": "weighted_average",
                    "confidence_threshold": 0.75,
                    "dynamic_weight_adjustment": True,
                    "min_model_agreement": 3,
                    "weight_decay_factor": 0.95
                }
            },
            "features": {
                "market_regime": {
                    "enabled": True,
                    "lookback_period": 15,
                    "normalization": "zscore",
                    "update_frequency": 1,
                    "features": {
                        "ema_crossover": True,
                        "vwap_deviation": True,
                        "greek_sentiment": True,
                        "iv_skew": True,
                        "oi_flow": True,
                        "regime_transitions": True,
                        "market_microstructure": True
                    },
                    "regime_count": 18,
                    "transition_smoothing": 3
                },
                "greeks": {
                    "enabled": True,
                    "calculation_method": "black_scholes",
                    "normalization": "minmax",
                    "smoothing_window": 5,
                    "moneyness_levels": ["ATM", "ITM", "OTM"],
                    "greeks_calculated": ["delta", "gamma", "vega", "theta", "rho"],
                    "sentiment_calculation": True
                },
                "iv": {
                    "enabled": True,
                    "calculation_window": 252,
                    "smoothing": "exponential",
                    "outlier_handling": "clip",
                    "outlier_threshold": 3,
                    "strike_surface": [-10, -5, 0, 5, 10],
                    "term_structure": [7, 14, 30, 60, 90],
                    "percentile_calculation": True,
                    "skew_calculation": True
                },
                "oi": {
                    "enabled": True,
                    "rolling_window": 20,
                    "change_calculation": "both",
                    "filtering": "volume_weighted",
                    "smart_money_detection": True,
                    "institutional_threshold": 0.7,
                    "concentration_analysis": True,
                    "max_pain_calculation": True
                },
                "technical": {
                    "enabled": True,
                    "period_length": 14,
                    "calculation_method": "standard",
                    "smoothing": True,
                    "indicators": {
                        "price_relative": True,
                        "volatility": True,
                        "momentum": True,
                        "trend": True,
                        "support_resistance": True,
                        "volume_analysis": True
                    },
                    "custom_indicators": []
                },
                "microstructure": {
                    "enabled": True,
                    "window": 5,
                    "aggregation": "volume_weighted",
                    "filtering": "outlier_robust",
                    "features": {
                        "order_flow": True,
                        "spread_analysis": True,
                        "market_depth": True,
                        "trade_analytics": True,
                        "market_impact": True,
                        "kyle_lambda": True
                    },
                    "tick_aggregation": 100
                }
            },
            "risk_management": {
                "position_sizing": {
                    "method": "kelly",
                    "kelly_fraction": 0.25,
                    "max_position_size": 0.20,
                    "max_positions": 5,
                    "position_correlation_limit": 0.30,
                    "min_position_size": 0.01,
                    "sizing_frequency": "per_signal"
                },
                "risk_limits": {
                    "max_portfolio_risk": 0.30,
                    "max_daily_loss": 0.05,
                    "max_drawdown": 0.20,
                    "var_confidence_level": 0.95,
                    "stress_test_scenarios": True,
                    "max_leverage": 2.0,
                    "margin_call_level": 0.75
                },
                "stop_loss": {
                    "method": "atr_based",
                    "atr_multiplier": 2.0,
                    "trailing_stop": True,
                    "profit_target_multiple": 2.0,
                    "max_holding_time": 1440,
                    "time_based_exit": "15:15:00",
                    "breakeven_activation": 1.5
                },
                "circuit_breaker": {
                    "loss_threshold": 0.03,
                    "cooldown_period": 60,
                    "volatility_trigger": 0.05,
                    "manual_override": True,
                    "consecutive_loss_limit": 3,
                    "recovery_check_period": 30
                }
            },
            "signal_generation": {
                "straddle_config": {
                    "atm": {
                        "confidence_threshold": 0.75,
                        "weight": 0.40,
                        "max_positions": 2
                    },
                    "itm": {
                        "confidence_threshold": 0.80,
                        "weight": 0.35,
                        "max_positions": 2,
                        "strike_distance": 1
                    },
                    "otm": {
                        "confidence_threshold": 0.70,
                        "weight": 0.25,
                        "max_positions": 1,
                        "strike_distance": 1
                    },
                    "signal_aggregation_window": 10,
                    "consensus_threshold": 0.60,
                    "multi_timeframe_analysis": True,
                    "rolling_window": True
                },
                "signal_filters": {
                    "min_volume": 1000,
                    "max_spread": 0.01,
                    "market_hours_only": True,
                    "min_dte": 1,
                    "max_dte": 45,
                    "iv_percentile_range": {
                        "min": 10,
                        "max": 90
                    },
                    "moneyness_filter": {
                        "min": -0.1,
                        "max": 0.1
                    },
                    "liquidity_score_min": 0.7
                },
                "signal_processing": {
                    "signal_latency_target": 100,
                    "smoothing_window": 5,
                    "outlier_detection": True,
                    "cross_validation": True,
                    "real_time_processing": True,
                    "signal_decay_factor": 0.95,
                    "min_signal_strength": 0.6
                }
            },
            "training": {
                "training_period": {
                    "start_year": 2020,
                    "end_year": 2024
                },
                "validation_split": 0.20,
                "cross_validation_folds": 5,
                "time_series_split": True,
                "feature_selection": "mutual_info",
                "feature_importance_threshold": 0.01,
                "early_stopping": True,
                "early_stopping_patience": 10,
                "mixed_precision": True,
                "gpu_optimization": True,
                "hyperparameter_optimization": "optuna",
                "n_trials": 100,
                "ensemble_training": True
            },
            "backtesting": {
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "slippage_model": "linear",
                "execution_delay": 50,
                "partial_fills": True,
                "market_impact_model": True,
                "realistic_queue_position": True
            },
            "database": {
                "heavydb": {
                    "host": "localhost",
                    "port": 6274,
                    "database": "heavyai",
                    "user": "admin",
                    "password": "HyperInteractive",
                    "gpu_queries": True,
                    "connection_pooling": True,
                    "pool_size": 10,
                    "query_cache_ttl": 300,
                    "compression": True
                },
                "data_source": {
                    "primary_table": "nifty_option_chain",
                    "data_quality_checks": True,
                    "real_time_processing": True,
                    "cache_ttl": 300,
                    "compression": True,
                    "partitioning": "daily"
                }
            },
            "monitoring": {
                "performance_tracking": True,
                "metrics": [
                    "signal_latency",
                    "prediction_accuracy",
                    "sharpe_ratio",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor"
                ],
                "alert_thresholds": {
                    "latency_ms": 200,
                    "accuracy_min": 0.60,
                    "drawdown_max": 0.25
                },
                "logging_level": "INFO",
                "checkpoint_frequency": "hourly"
            }
        }
    
    def _validate_ml_models(self) -> List[str]:
        """Validate ML model configurations"""
        errors = []
        models = self.get("ml_models", {})
        
        # Validate model weights sum to 1.0
        weights = []
        for model_name in ["lightgbm", "catboost", "tabnet", "lstm", "transformer"]:
            model_config = models.get(model_name, {})
            weight = model_config.get("weight", 0)
            weights.append(weight)
        
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Model weights must sum to 1.0, got {total_weight}")
        
        # Validate LightGBM settings
        lgb = models.get("lightgbm", {})
        if lgb.get("device_type") == "GPU" and not lgb.get("gpu_use_dp", False):
            # This is actually good - FP16 is faster
            pass
        
        # Validate ensemble settings
        ensemble = models.get("ensemble", {})
        min_agreement = ensemble.get("min_model_agreement", 1)
        if min_agreement > 5:
            errors.append("Minimum model agreement cannot exceed number of models (5)")
        
        return errors
    
    def _validate_features(self) -> List[str]:
        """Validate feature engineering settings"""
        errors = []
        features = self.get("features", {})
        
        # Check if at least some features are enabled
        enabled_count = 0
        for feature_type in ["market_regime", "greeks", "iv", "oi", "technical", "microstructure"]:
            if features.get(feature_type, {}).get("enabled", False):
                enabled_count += 1
        
        if enabled_count == 0:
            errors.append("At least one feature category must be enabled")
        
        # Validate IV surface strikes
        iv = features.get("iv", {})
        strikes = iv.get("strike_surface", [])
        if strikes and strikes != sorted(strikes):
            errors.append("IV strike surface must be in ascending order")
        
        # Validate OI settings
        oi = features.get("oi", {})
        if oi.get("smart_money_detection") and not oi.get("institutional_threshold"):
            errors.append("Institutional threshold required when smart money detection is enabled")
        
        return errors
    
    def _validate_risk_management(self) -> List[str]:
        """Validate risk management settings"""
        errors = []
        risk = self.get("risk_management", {})
        
        # Validate position sizing
        sizing = risk.get("position_sizing", {})
        if sizing.get("method") == "kelly":
            kelly_fraction = sizing.get("kelly_fraction", 0)
            if not 0 < kelly_fraction <= 1:
                errors.append("Kelly fraction must be between 0 and 1")
        
        # Validate risk limits
        limits = risk.get("risk_limits", {})
        max_daily_loss = limits.get("max_daily_loss", 0)
        max_drawdown = limits.get("max_drawdown", 0)
        
        if max_daily_loss >= max_drawdown:
            errors.append("Daily loss limit should be less than maximum drawdown")
        
        # Validate stop loss
        stop_loss = risk.get("stop_loss", {})
        if stop_loss.get("method") == "atr_based":
            multiplier = stop_loss.get("atr_multiplier", 0)
            if multiplier <= 0:
                errors.append("ATR multiplier must be positive")
        
        return errors
    
    def _validate_signal_generation(self) -> List[str]:
        """Validate signal generation settings"""
        errors = []
        signals = self.get("signal_generation", {})
        
        # Validate straddle weights
        straddle = signals.get("straddle_config", {})
        weights = []
        for straddle_type in ["atm", "itm", "otm"]:
            config = straddle.get(straddle_type, {})
            weights.append(config.get("weight", 0))
        
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Straddle weights must sum to 1.0, got {total_weight}")
        
        # Validate signal filters
        filters = signals.get("signal_filters", {})
        min_dte = filters.get("min_dte", 0)
        max_dte = filters.get("max_dte", 45)
        
        if min_dte >= max_dte:
            errors.append("Minimum DTE must be less than maximum DTE")
        
        # Validate IV percentile range
        iv_range = filters.get("iv_percentile_range", {})
        if iv_range:
            min_iv = iv_range.get("min", 0)
            max_iv = iv_range.get("max", 100)
            if min_iv >= max_iv:
                errors.append("Minimum IV percentile must be less than maximum")
        
        return errors
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights for ensemble"""
        models = self.get("ml_models", {})
        weights = {}
        
        for model_name in ["lightgbm", "catboost", "tabnet", "lstm", "transformer"]:
            model_config = models.get(model_name, {})
            weights[model_name] = model_config.get("weight", 0)
        
        return weights
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled feature categories"""
        features = self.get("features", {})
        enabled = []
        
        for feature_type in ["market_regime", "greeks", "iv", "oi", "technical", "microstructure"]:
            if features.get(feature_type, {}).get("enabled", False):
                enabled.append(feature_type)
        
        return enabled
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return self.get("risk_management", {})
    
    def get_straddle_config(self) -> Dict[str, Any]:
        """Get straddle configuration"""
        return self.get("signal_generation.straddle_config", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get("database", {})
    
    def calculate_feature_count(self) -> int:
        """Calculate total number of features"""
        count = 0
        features = self.get("features", {})
        
        # Market regime features
        if features.get("market_regime", {}).get("enabled"):
            count += 38  # As per documentation
        
        # Greek features
        if features.get("greeks", {}).get("enabled"):
            count += 21  # 5 greeks × 3 moneyness + 6 aggregated
        
        # IV features
        if features.get("iv", {}).get("enabled"):
            count += 30  # Strike surface + term structure + analytics
        
        # OI features
        if features.get("oi", {}).get("enabled"):
            count += 20  # Levels + changes + analytics
        
        # Technical features
        if features.get("technical", {}).get("enabled"):
            count += 31  # Various technical indicators
        
        # Microstructure features
        if features.get("microstructure", {}).get("enabled"):
            count += 20  # Order flow + spread + depth + impact
        
        return count
    
    def __str__(self) -> str:
        """String representation"""
        model_count = len([m for m, w in self.get_model_weights().items() if w > 0])
        feature_count = self.calculate_feature_count()
        return f"ML Triple Straddle Configuration: {self.strategy_name} ({model_count} models, {feature_count} features)"