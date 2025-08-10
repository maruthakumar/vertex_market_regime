"""
ML (Machine Learning) Configuration
"""

from typing import Dict, Any, List, Optional
from datetime import time
import json

from ..core.base_config import BaseConfiguration
from ..core.exceptions import ValidationError

class MLConfiguration(BaseConfiguration):
    """
    Configuration for Machine Learning Strategy (ML)
    
    This configuration handles all ML-specific parameters including
    model selection, feature engineering, training parameters,
    prediction settings, and ensemble configurations.
    """
    
    def __init__(self, strategy_name: str):
        """Initialize ML configuration"""
        super().__init__("ml", strategy_name)
        
        # Set default values
        self._data = self.get_default_values()
    
    def validate(self) -> bool:
        """
        Validate ML configuration
        
        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = {}
        
        # Validate model configuration
        model_errors = self._validate_model_config()
        if model_errors:
            errors['model_config'] = model_errors
        
        # Validate feature engineering
        feature_errors = self._validate_feature_engineering()
        if feature_errors:
            errors['feature_engineering'] = feature_errors
        
        # Validate training parameters
        training_errors = self._validate_training_parameters()
        if training_errors:
            errors['training_parameters'] = training_errors
        
        # Validate prediction settings
        prediction_errors = self._validate_prediction_settings()
        if prediction_errors:
            errors['prediction_settings'] = prediction_errors
        
        if errors:
            raise ValidationError("ML configuration validation failed", errors)
        
        self._is_validated = True
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for ML configuration"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["model_config", "feature_engineering", "training_parameters", "prediction_settings"],
            "properties": {
                "model_config": {
                    "type": "object",
                    "required": ["primary_model", "model_parameters"],
                    "properties": {
                        "primary_model": {
                            "type": "string",
                            "enum": ["LIGHTGBM", "CATBOOST", "XGBOOST", "RANDOM_FOREST", "NEURAL_NETWORK", "LSTM", "TRANSFORMER", "ENSEMBLE"]
                        },
                        "model_parameters": {
                            "type": "object",
                            "properties": {
                                "n_estimators": {"type": "integer", "minimum": 10},
                                "max_depth": {"type": "integer", "minimum": 1},
                                "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                                "subsample": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                                "colsample_bytree": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                                "reg_alpha": {"type": "number", "minimum": 0},
                                "reg_lambda": {"type": "number", "minimum": 0}
                            }
                        },
                        "ensemble_config": {
                            "type": "object",
                            "properties": {
                                "models": {"type": "array", "items": {"type": "string"}},
                                "voting_method": {"type": "string", "enum": ["HARD", "SOFT", "WEIGHTED"]},
                                "weights": {"type": "array", "items": {"type": "number"}}
                            }
                        },
                        "neural_network_config": {
                            "type": "object",
                            "properties": {
                                "architecture": {"type": "array", "items": {"type": "integer"}},
                                "activation": {"type": "string", "enum": ["relu", "tanh", "sigmoid", "elu"]},
                                "dropout_rate": {"type": "number", "minimum": 0, "maximum": 0.9},
                                "batch_normalization": {"type": "boolean"}
                            }
                        }
                    }
                },
                "feature_engineering": {
                    "type": "object",
                    "required": ["feature_groups", "preprocessing"],
                    "properties": {
                        "feature_groups": {
                            "type": "object",
                            "properties": {
                                "price_features": {"type": "boolean"},
                                "volume_features": {"type": "boolean"},
                                "technical_indicators": {"type": "boolean"},
                                "market_microstructure": {"type": "boolean"},
                                "options_greeks": {"type": "boolean"},
                                "sentiment_features": {"type": "boolean"},
                                "time_features": {"type": "boolean"},
                                "custom_features": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "technical_indicators": {
                            "type": "object",
                            "properties": {
                                "moving_averages": {"type": "array", "items": {"type": "integer"}},
                                "rsi_periods": {"type": "array", "items": {"type": "integer"}},
                                "bollinger_bands": {"type": "boolean"},
                                "macd": {"type": "boolean"},
                                "atr": {"type": "boolean"},
                                "stochastic": {"type": "boolean"},
                                "custom_indicators": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "preprocessing": {
                            "type": "object",
                            "properties": {
                                "scaling_method": {"type": "string", "enum": ["STANDARD", "MINMAX", "ROBUST", "NONE"]},
                                "handle_missing": {"type": "string", "enum": ["DROP", "FORWARD_FILL", "INTERPOLATE", "MEAN"]},
                                "outlier_method": {"type": "string", "enum": ["IQR", "ZSCORE", "ISOLATION_FOREST", "NONE"]},
                                "feature_selection": {
                                    "type": "object",
                                    "properties": {
                                        "method": {"type": "string", "enum": ["IMPORTANCE", "CORRELATION", "MUTUAL_INFO", "NONE"]},
                                        "top_features": {"type": "integer", "minimum": 1}
                                    }
                                }
                            }
                        }
                    }
                },
                "training_parameters": {
                    "type": "object",
                    "required": ["data_split", "cross_validation", "optimization"],
                    "properties": {
                        "data_split": {
                            "type": "object",
                            "properties": {
                                "train_ratio": {"type": "number", "minimum": 0.1, "maximum": 0.9},
                                "validation_ratio": {"type": "number", "minimum": 0.1, "maximum": 0.5},
                                "test_ratio": {"type": "number", "minimum": 0.1, "maximum": 0.5},
                                "time_based_split": {"type": "boolean"},
                                "purged_window": {"type": "integer", "minimum": 0}
                            }
                        },
                        "cross_validation": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["TIME_SERIES", "PURGED_KF", "WALK_FORWARD", "NONE"]},
                                "n_splits": {"type": "integer", "minimum": 2, "maximum": 20},
                                "gap": {"type": "integer", "minimum": 0}
                            }
                        },
                        "optimization": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["GRID_SEARCH", "RANDOM_SEARCH", "BAYESIAN", "OPTUNA", "NONE"]},
                                "metric": {"type": "string", "enum": ["ACCURACY", "F1", "PRECISION", "RECALL", "AUC", "SHARPE", "PROFIT"]},
                                "n_trials": {"type": "integer", "minimum": 1},
                                "early_stopping": {"type": "boolean"},
                                "patience": {"type": "integer", "minimum": 1}
                            }
                        },
                        "training_window": {
                            "type": "object",
                            "properties": {
                                "lookback_days": {"type": "integer", "minimum": 30},
                                "retrain_frequency": {"type": "string", "enum": ["DAILY", "WEEKLY", "MONTHLY", "QUARTERLY"]},
                                "incremental_learning": {"type": "boolean"}
                            }
                        }
                    }
                },
                "prediction_settings": {
                    "type": "object",
                    "required": ["prediction_type", "target_definition", "execution"],
                    "properties": {
                        "prediction_type": {"type": "string", "enum": ["CLASSIFICATION", "REGRESSION", "MULTI_CLASS", "MULTI_OUTPUT"]},
                        "target_definition": {
                            "type": "object",
                            "properties": {
                                "target_variable": {"type": "string"},
                                "prediction_horizon": {"type": "integer", "minimum": 1},
                                "threshold": {"type": "number"},
                                "classes": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "execution": {
                            "type": "object",
                            "properties": {
                                "position_type": {"type": "string", "enum": ["BUY", "SELL", "BOTH", "PREDICTED"]},
                                "option_type": {"type": "string", "enum": ["CE", "PE", "BOTH", "PREDICTED"]},
                                "strike_selection": {
                                    "type": "object",
                                    "properties": {
                                        "method": {"type": "string", "enum": ["ATM", "PREDICTED", "VOLATILITY_BASED"]},
                                        "offset": {"type": "integer"}
                                    }
                                },
                                "quantity_method": {"type": "string", "enum": ["FIXED", "CONFIDENCE_BASED", "KELLY", "RISK_PARITY"]},
                                "stop_loss": {"type": "number", "minimum": 0},
                                "target": {"type": "number", "minimum": 0}
                            }
                        }
                    }
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "max_positions": {"type": "integer", "minimum": 1},
                        "position_sizing": {
                            "type": "object",
                            "properties": {
                                "base_size": {"type": "number", "minimum": 0},
                                "max_size": {"type": "number", "minimum": 0},
                                "confidence_scaling": {"type": "boolean"}
                            }
                        },
                        "portfolio_constraints": {
                            "type": "object",
                            "properties": {
                                "max_sector_exposure": {"type": "number", "minimum": 0, "maximum": 1},
                                "max_correlation": {"type": "number", "minimum": 0, "maximum": 1},
                                "diversification_ratio": {"type": "number", "minimum": 1}
                            }
                        }
                    }
                },
                "monitoring": {
                    "type": "object",
                    "properties": {
                        "model_drift": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "method": {"type": "string", "enum": ["PSI", "KS", "WASSERSTEIN", "CUSTOM"]},
                                "threshold": {"type": "number", "minimum": 0}
                            }
                        },
                        "performance_tracking": {
                            "type": "object",
                            "properties": {
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "evaluation_window": {"type": "integer", "minimum": 1},
                                "alert_threshold": {"type": "number"}
                            }
                        }
                    }
                }
            }
        }
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for ML configuration"""
        return {
            "model_config": {
                "primary_model": "LIGHTGBM",
                "model_parameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "min_child_samples": 20,
                    "num_leaves": 31
                },
                "ensemble_config": {
                    "models": ["LIGHTGBM", "CATBOOST", "XGBOOST"],
                    "voting_method": "SOFT",
                    "weights": [0.4, 0.3, 0.3],
                    "meta_learner": None
                },
                "neural_network_config": {
                    "architecture": [128, 64, 32],
                    "activation": "relu",
                    "dropout_rate": 0.2,
                    "batch_normalization": True,
                    "optimizer": "adam",
                    "learning_rate_schedule": "cosine"
                },
                "tabnet_config": {
                    "n_steps": 3,
                    "gamma": 1.3,
                    "n_independent": 2,
                    "n_shared": 2,
                    "momentum": 0.02
                }
            },
            "feature_engineering": {
                "feature_groups": {
                    "price_features": True,
                    "volume_features": True,
                    "technical_indicators": True,
                    "market_microstructure": True,
                    "options_greeks": True,
                    "sentiment_features": False,
                    "time_features": True,
                    "custom_features": []
                },
                "technical_indicators": {
                    "moving_averages": [5, 10, 20, 50],
                    "rsi_periods": [14, 21],
                    "bollinger_bands": True,
                    "macd": True,
                    "atr": True,
                    "stochastic": True,
                    "adx": True,
                    "cci": True,
                    "williams_r": True,
                    "custom_indicators": []
                },
                "market_microstructure": {
                    "bid_ask_spread": True,
                    "order_imbalance": True,
                    "trade_intensity": True,
                    "price_impact": True,
                    "liquidity_measures": True
                },
                "preprocessing": {
                    "scaling_method": "ROBUST",
                    "handle_missing": "FORWARD_FILL",
                    "outlier_method": "IQR",
                    "outlier_threshold": 3,
                    "feature_selection": {
                        "method": "IMPORTANCE",
                        "top_features": 50,
                        "importance_threshold": 0.01
                    },
                    "dimensionality_reduction": {
                        "method": "PCA",
                        "n_components": 0.95
                    }
                }
            },
            "training_parameters": {
                "data_split": {
                    "train_ratio": 0.7,
                    "validation_ratio": 0.15,
                    "test_ratio": 0.15,
                    "time_based_split": True,
                    "purged_window": 5,
                    "embargo_period": 1
                },
                "cross_validation": {
                    "method": "PURGED_KF",
                    "n_splits": 5,
                    "gap": 10,
                    "max_train_size": None,
                    "test_size": 0.2
                },
                "optimization": {
                    "method": "OPTUNA",
                    "metric": "SHARPE",
                    "n_trials": 100,
                    "early_stopping": True,
                    "patience": 10,
                    "parallel_jobs": -1,
                    "seed": 42
                },
                "training_window": {
                    "lookback_days": 365,
                    "retrain_frequency": "WEEKLY",
                    "incremental_learning": False,
                    "warm_start": True,
                    "min_samples": 1000
                },
                "class_balancing": {
                    "method": "SMOTE",
                    "sampling_strategy": "auto",
                    "random_state": 42
                }
            },
            "prediction_settings": {
                "prediction_type": "CLASSIFICATION",
                "target_definition": {
                    "target_variable": "direction",
                    "prediction_horizon": 5,
                    "threshold": 0.002,
                    "classes": ["UP", "DOWN", "NEUTRAL"],
                    "class_weights": "balanced"
                },
                "confidence_threshold": 0.6,
                "ensemble_predictions": {
                    "use_ensemble": True,
                    "aggregation_method": "WEIGHTED_AVERAGE",
                    "confidence_weighting": True
                },
                "execution": {
                    "position_type": "PREDICTED",
                    "option_type": "PREDICTED",
                    "strike_selection": {
                        "method": "VOLATILITY_BASED",
                        "offset": 0,
                        "volatility_multiplier": 1.0
                    },
                    "quantity_method": "CONFIDENCE_BASED",
                    "base_quantity": 1,
                    "max_quantity": 5,
                    "stop_loss": 30,
                    "target": 50,
                    "time_based_exit": "15:15:00"
                }
            },
            "risk_management": {
                "max_positions": 5,
                "position_sizing": {
                    "base_size": 0.02,
                    "max_size": 0.1,
                    "confidence_scaling": True,
                    "kelly_fraction": 0.25
                },
                "portfolio_constraints": {
                    "max_sector_exposure": 0.3,
                    "max_correlation": 0.7,
                    "diversification_ratio": 2.0,
                    "max_drawdown": 0.15
                },
                "risk_metrics": {
                    "var_confidence": 0.95,
                    "cvar_confidence": 0.95,
                    "max_leverage": 2.0
                }
            },
            "monitoring": {
                "model_drift": {
                    "enabled": True,
                    "method": "PSI",
                    "threshold": 0.2,
                    "check_frequency": "DAILY",
                    "features_to_monitor": ["price", "volume", "volatility"]
                },
                "performance_tracking": {
                    "metrics": ["accuracy", "precision", "recall", "f1", "sharpe", "max_drawdown"],
                    "evaluation_window": 20,
                    "alert_threshold": -0.2,
                    "comparison_baseline": "buy_and_hold"
                },
                "feature_importance": {
                    "track_importance": True,
                    "update_frequency": "WEEKLY",
                    "importance_change_threshold": 0.1
                }
            }
        }
    
    def _validate_model_config(self) -> List[str]:
        """Validate model configuration"""
        errors = []
        model_config = self.get("model_config", {})
        
        # Validate model parameters
        params = model_config.get("model_parameters", {})
        
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if not 0.001 <= lr <= 1.0:
                errors.append("Learning rate must be between 0.001 and 1.0")
        
        if "max_depth" in params:
            depth = params["max_depth"]
            if depth < 1:
                errors.append("Max depth must be at least 1")
        
        # Validate ensemble config
        ensemble = model_config.get("ensemble_config", {})
        if ensemble:
            models = ensemble.get("models", [])
            weights = ensemble.get("weights", [])
            
            if models and weights and len(models) != len(weights):
                errors.append("Number of models must match number of weights in ensemble")
            
            if weights and abs(sum(weights) - 1.0) > 0.01:
                errors.append("Ensemble weights must sum to 1.0")
        
        return errors
    
    def _validate_feature_engineering(self) -> List[str]:
        """Validate feature engineering settings"""
        errors = []
        features = self.get("feature_engineering", {})
        
        # Validate technical indicators
        indicators = features.get("technical_indicators", {})
        ma_periods = indicators.get("moving_averages", [])
        
        for period in ma_periods:
            if period <= 0:
                errors.append("Moving average periods must be positive")
                break
        
        # Validate preprocessing
        preprocessing = features.get("preprocessing", {})
        feature_selection = preprocessing.get("feature_selection", {})
        
        if feature_selection.get("method") == "IMPORTANCE":
            top_features = feature_selection.get("top_features", 0)
            if top_features <= 0:
                errors.append("Top features must be positive when using importance-based selection")
        
        return errors
    
    def _validate_training_parameters(self) -> List[str]:
        """Validate training parameters"""
        errors = []
        training = self.get("training_parameters", {})
        
        # Validate data split
        split = training.get("data_split", {})
        train = split.get("train_ratio", 0)
        val = split.get("validation_ratio", 0)
        test = split.get("test_ratio", 0)
        
        if abs((train + val + test) - 1.0) > 0.01:
            errors.append("Train, validation, and test ratios must sum to 1.0")
        
        # Validate cross-validation
        cv = training.get("cross_validation", {})
        if cv.get("method") != "NONE":
            n_splits = cv.get("n_splits", 0)
            if n_splits < 2:
                errors.append("Cross-validation requires at least 2 splits")
        
        # Validate training window
        window = training.get("training_window", {})
        lookback = window.get("lookback_days", 0)
        if lookback < 30:
            errors.append("Training lookback should be at least 30 days")
        
        return errors
    
    def _validate_prediction_settings(self) -> List[str]:
        """Validate prediction settings"""
        errors = []
        prediction = self.get("prediction_settings", {})
        
        # Validate confidence threshold
        confidence = prediction.get("confidence_threshold", 0)
        if not 0 <= confidence <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Validate target definition
        target = prediction.get("target_definition", {})
        horizon = target.get("prediction_horizon", 0)
        if horizon <= 0:
            errors.append("Prediction horizon must be positive")
        
        # Validate execution
        execution = prediction.get("execution", {})
        stop_loss = execution.get("stop_loss", 0)
        target_profit = execution.get("target", 0)
        
        if stop_loss < 0:
            errors.append("Stop loss cannot be negative")
        if target_profit < 0:
            errors.append("Target profit cannot be negative")
        
        return errors
    
    def get_model_type(self) -> str:
        """Get primary model type"""
        return self.get("model_config.primary_model", "LIGHTGBM")
    
    def get_feature_list(self) -> List[str]:
        """Get list of all enabled features"""
        features = []
        feature_groups = self.get("feature_engineering.feature_groups", {})
        
        # Add feature group names
        for group, enabled in feature_groups.items():
            if enabled and group != "custom_features":
                features.append(group)
        
        # Add custom features
        custom = feature_groups.get("custom_features", [])
        features.extend(custom)
        
        return features
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for training"""
        return self.get("model_config.model_parameters", {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get complete training configuration"""
        return {
            "model_type": self.get_model_type(),
            "model_params": self.get_model_parameters(),
            "data_split": self.get("training_parameters.data_split", {}),
            "cross_validation": self.get("training_parameters.cross_validation", {}),
            "optimization": self.get("training_parameters.optimization", {})
        }
    
    def get_prediction_threshold(self) -> float:
        """Get confidence threshold for predictions"""
        return self.get("prediction_settings.confidence_threshold", 0.6)
    
    def requires_retraining(self, last_train_date: datetime, current_date: datetime) -> bool:
        """
        Check if model requires retraining
        
        Args:
            last_train_date: Date of last training
            current_date: Current date
            
        Returns:
            True if retraining is needed
        """
        frequency = self.get("training_parameters.training_window.retrain_frequency", "WEEKLY")
        
        days_diff = (current_date - last_train_date).days
        
        if frequency == "DAILY":
            return days_diff >= 1
        elif frequency == "WEEKLY":
            return days_diff >= 7
        elif frequency == "MONTHLY":
            return days_diff >= 30
        elif frequency == "QUARTERLY":
            return days_diff >= 90
        
        return False
    
    def __str__(self) -> str:
        """String representation"""
        return f"ML Configuration: {self.strategy_name} (Model: {self.get_model_type()})"