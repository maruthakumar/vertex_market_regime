"""
5-Model ML Ensemble for Ultra-Sophisticated Pattern Scoring

Implements a comprehensive ensemble of 5 specialized machine learning models:
1. LightGBM - Gradient boosting for pattern feature relationships
2. CatBoost - Categorical boosting for regime-based patterns  
3. TabNet - Deep tabular learning for complex feature interactions
4. LSTM - Recurrent neural network for temporal pattern sequences
5. Transformer - Attention-based model for multi-timeframe analysis

The ensemble combines predictions using advanced stacking and meta-learning
to achieve superior pattern scoring accuracy and robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .pattern_repository import PatternSchema

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Individual model prediction result"""
    model_name: str
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    prediction_time: float
    model_version: str


@dataclass
class EnsemblePrediction:
    """Complete ensemble prediction result"""
    pattern_id: str
    timestamp: datetime
    
    # Individual model predictions
    lightgbm_prediction: Optional[ModelPrediction]
    catboost_prediction: Optional[ModelPrediction]
    tabnet_prediction: Optional[ModelPrediction]
    lstm_prediction: Optional[ModelPrediction]
    transformer_prediction: Optional[ModelPrediction]
    
    # Ensemble results
    ensemble_score: float
    ensemble_confidence: float
    prediction_variance: float
    model_agreement: float
    
    # Meta-features
    volatility_regime_score: float
    trend_regime_score: float
    pattern_complexity_score: float
    temporal_consistency_score: float
    
    # Quality metrics
    prediction_quality: str  # 'excellent', 'good', 'fair', 'poor'
    recommendation: str      # 'strong_buy', 'buy', 'hold', 'avoid'
    risk_score: float


class LSTMPatternModel(nn.Module):
    """LSTM model for temporal pattern analysis"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPatternModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last time step
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = torch.sigmoid(self.fc3(out))
        
        return out


class TransformerPatternModel(nn.Module):
    """Transformer model for multi-timeframe attention analysis"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super(TransformerPatternModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class AdvancedMLEnsemble:
    """
    Advanced 5-Model ML Ensemble for Pattern Scoring
    
    Combines multiple state-of-the-art machine learning models:
    1. LightGBM - Fast gradient boosting with optimal splits
    2. CatBoost - Handles categorical features natively
    3. TabNet - Deep learning for tabular data
    4. LSTM - Temporal sequence modeling
    5. Transformer - Multi-head attention for timeframes
    
    Uses meta-learning and stacking for ensemble predictions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ML Ensemble
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or self._get_default_config()
        
        # Model availability flags
        self.models_available = {
            'lightgbm': LIGHTGBM_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE,
            'tabnet': TABNET_AVAILABLE,
            'lstm': PYTORCH_AVAILABLE,
            'transformer': PYTORCH_AVAILABLE
        }
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        
        # Meta-learner for ensemble
        self.meta_learner = None
        
        # Model performance tracking
        self.model_performances = {}
        self.training_history = []
        
        # Device for PyTorch models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = logging.getLogger(f"{__name__}.AdvancedMLEnsemble")
        self.logger.info("Advanced ML Ensemble initialized")
        self.logger.info(f"Available models: {[name for name, available in self.models_available.items() if available]}")
        
        # Initialize models
        self._initialize_models()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default ensemble configuration"""
        return {
            # Training parameters
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            
            # LightGBM config
            'lgb_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            
            # CatBoost config
            'cb_params': {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.03,
                'random_seed': 42,
                'verbose': False
            },
            
            # TabNet config
            'tabnet_params': {
                'n_d': 64,
                'n_a': 64,
                'n_steps': 5,
                'gamma': 1.3,
                'lambda_sparse': 1e-3,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': {'lr': 2e-2},
                'mask_type': 'entmax'
            },
            
            # LSTM config
            'lstm_params': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'sequence_length': 10,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            
            # Transformer config
            'transformer_params': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'sequence_length': 10,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            
            # Ensemble config
            'ensemble_method': 'stacking',  # 'simple_average', 'weighted_average', 'stacking'
            'meta_learner_params': {
                'alpha': 1.0,
                'random_state': 42
            }
        }
    
    def _initialize_models(self):
        """Initialize all available models"""
        try:
            # Initialize LightGBM
            if self.models_available['lightgbm']:
                self.models['lightgbm'] = None  # Will be created during training
                self.logger.info("LightGBM model initialized")
            
            # Initialize CatBoost
            if self.models_available['catboost']:
                self.models['catboost'] = None  # Will be created during training
                self.logger.info("CatBoost model initialized")
            
            # Initialize TabNet
            if self.models_available['tabnet']:
                self.models['tabnet'] = None  # Will be created during training
                self.logger.info("TabNet model initialized")
            
            # Initialize LSTM
            if self.models_available['lstm']:
                self.models['lstm'] = None  # Will be created during training
                self.logger.info("LSTM model initialized")
            
            # Initialize Transformer
            if self.models_available['transformer']:
                self.models['transformer'] = None  # Will be created during training
                self.logger.info("Transformer model initialized")
            
            # Initialize meta-learner
            self.meta_learner = Ridge(**self.config['meta_learner_params'])
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def prepare_features(self, patterns: List[PatternSchema]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix from patterns
        
        Args:
            patterns: List of pattern schemas
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        try:
            feature_data = []
            targets = []
            
            for pattern in patterns:
                # Extract pattern features
                pattern_features = self._extract_pattern_features(pattern)
                if pattern_features is not None:
                    feature_data.append(pattern_features)
                    
                    # Target is success rate or performance score
                    target = pattern.historical_performance.get('success_rate', 0.0)
                    targets.append(target)
            
            if not feature_data:
                raise ValueError("No valid features extracted from patterns")
            
            # Convert to arrays
            features = np.array(feature_data)
            targets = np.array(targets)
            
            # Generate feature names
            feature_names = self._generate_feature_names()
            
            self.logger.info(f"Prepared features: {features.shape[0]} patterns, {features.shape[1]} features")
            
            return features, targets, feature_names
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    def _extract_pattern_features(self, pattern: PatternSchema) -> Optional[np.ndarray]:
        """Extract numerical features from a pattern"""
        try:
            features = []
            
            # Basic pattern features
            features.extend([
                len(pattern.components),
                len(pattern.timeframe_analysis),
                pattern.historical_performance.get('success_rate', 0.0),
                pattern.historical_performance.get('avg_return', 0.0),
                pattern.historical_performance.get('max_drawdown', 0.0),
                getattr(pattern, 'total_occurrences', 0)
            ])
            
            # Timeframe analysis features
            timeframe_scores = []
            timeframe_strengths = []
            for tf, tf_data in pattern.timeframe_analysis.items():
                timeframe_scores.append(tf_data.get('validation_score', 0.0))
                timeframe_strengths.append(tf_data.get('strength', 0.0))
            
            # Pad or truncate to fixed size
            timeframe_scores = self._pad_or_truncate(timeframe_scores, 4)
            timeframe_strengths = self._pad_or_truncate(timeframe_strengths, 4)
            features.extend(timeframe_scores)
            features.extend(timeframe_strengths)
            
            # Component features
            component_strengths = []
            component_actions = []
            for comp_name, comp_data in pattern.components.items():
                component_strengths.append(comp_data.get('strength', 0.0))
                
                # Encode action as number
                action = comp_data.get('action', 'neutral')
                action_encoded = self._encode_action(action)
                component_actions.append(action_encoded)
            
            # Pad or truncate to fixed size (10 components)
            component_strengths = self._pad_or_truncate(component_strengths, 10)
            component_actions = self._pad_or_truncate(component_actions, 10)
            features.extend(component_strengths)
            features.extend(component_actions)
            
            # Cross-timeframe confluence features
            confluence_data = pattern.cross_timeframe_confluence
            features.extend([
                confluence_data.get('alignment_score', 0.0),
                confluence_data.get('consistency_score', 0.0),
                confluence_data.get('volatility_factor', 0.0)
            ])
            
            # Market context features
            market_context = pattern.market_context
            features.extend([
                self._encode_regime(market_context.get('preferred_regime', 'neutral')),
                self._encode_volatility(market_context.get('volatility_environment', 'medium')),
                self._encode_trend(market_context.get('trend_environment', 'neutral'))
            ])
            
            # Statistical features (if available)
            statistical_data = getattr(pattern, 'statistical_significance', {})
            features.extend([
                statistical_data.get('t_test', {}).get('p_value', 1.0),
                statistical_data.get('chi_square', {}).get('p_value', 1.0),
                statistical_data.get('effect_size', 0.0)
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error extracting features from pattern {pattern.pattern_id}: {e}")
            return None
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for interpretability"""
        names = [
            'num_components', 'num_timeframes', 'success_rate', 'avg_return', 
            'max_drawdown', 'total_occurrences'
        ]
        
        # Timeframe features
        for i in range(4):
            names.append(f'tf_{i}_score')
        for i in range(4):
            names.append(f'tf_{i}_strength')
        
        # Component features
        for i in range(10):
            names.append(f'comp_{i}_strength')
        for i in range(10):
            names.append(f'comp_{i}_action')
        
        # Confluence features
        names.extend(['alignment_score', 'consistency_score', 'volatility_factor'])
        
        # Market context features
        names.extend(['regime_encoded', 'volatility_encoded', 'trend_encoded'])
        
        # Statistical features
        names.extend(['t_test_p_value', 'chi_square_p_value', 'effect_size'])
        
        return names
    
    def _pad_or_truncate(self, data: List[float], target_length: int) -> List[float]:
        """Pad or truncate list to target length"""
        if len(data) >= target_length:
            return data[:target_length]
        else:
            return data + [0.0] * (target_length - len(data))
    
    def _encode_action(self, action: str) -> float:
        """Encode action string to number"""
        action_map = {
            'rejection': 1.0,
            'support': 0.8,
            'bounce': 0.6,
            'breakout': 0.4,
            'breakdown': 0.2,
            'neutral': 0.0
        }
        return action_map.get(action.lower(), 0.0)
    
    def _encode_regime(self, regime: str) -> float:
        """Encode regime string to number"""
        regime_map = {
            'bullish': 1.0,
            'bearish': -1.0,
            'neutral': 0.0,
            'volatile': 0.5,
            'consolidation': -0.5
        }
        return regime_map.get(regime.lower(), 0.0)
    
    def _encode_volatility(self, volatility: str) -> float:
        """Encode volatility string to number"""
        volatility_map = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'extreme': 1.0
        }
        return volatility_map.get(volatility.lower(), 0.5)
    
    def _encode_trend(self, trend: str) -> float:
        """Encode trend string to number"""
        trend_map = {
            'strong_bullish': 1.0,
            'bullish': 0.5,
            'neutral': 0.0,
            'bearish': -0.5,
            'strong_bearish': -1.0
        }
        return trend_map.get(trend.lower(), 0.0)
    
    def train(self, patterns: List[PatternSchema], validation_patterns: Optional[List[PatternSchema]] = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble
        
        Args:
            patterns: Training patterns
            validation_patterns: Optional validation patterns
            
        Returns:
            Training results and performance metrics
        """
        try:
            start_time = datetime.now()
            
            # Prepare features
            X, y, feature_names = self.prepare_features(patterns)
            self.feature_names = feature_names
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )
            
            # Scale features
            self.scalers['standard'] = StandardScaler()
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            # Store original data for sequence models
            X_train_seq = self._prepare_sequence_data(X_train_scaled)
            X_test_seq = self._prepare_sequence_data(X_test_scaled)
            
            # Training results
            training_results = {
                'training_time': 0.0,
                'model_performances': {},
                'ensemble_performance': {},
                'feature_importance': {}
            }
            
            # Train individual models
            base_predictions_train = []
            base_predictions_test = []
            
            # 1. Train LightGBM
            if self.models_available['lightgbm']:
                lgb_results = self._train_lightgbm(X_train, y_train, X_test, y_test)
                training_results['model_performances']['lightgbm'] = lgb_results
                base_predictions_train.append(lgb_results['train_predictions'])
                base_predictions_test.append(lgb_results['test_predictions'])
            
            # 2. Train CatBoost
            if self.models_available['catboost']:
                cb_results = self._train_catboost(X_train, y_train, X_test, y_test)
                training_results['model_performances']['catboost'] = cb_results
                base_predictions_train.append(cb_results['train_predictions'])
                base_predictions_test.append(cb_results['test_predictions'])
            
            # 3. Train TabNet
            if self.models_available['tabnet']:
                tabnet_results = self._train_tabnet(X_train_scaled, y_train, X_test_scaled, y_test)
                training_results['model_performances']['tabnet'] = tabnet_results
                base_predictions_train.append(tabnet_results['train_predictions'])
                base_predictions_test.append(tabnet_results['test_predictions'])
            
            # 4. Train LSTM
            if self.models_available['lstm']:
                lstm_results = self._train_lstm(X_train_seq, y_train, X_test_seq, y_test)
                training_results['model_performances']['lstm'] = lstm_results
                base_predictions_train.append(lstm_results['train_predictions'])
                base_predictions_test.append(lstm_results['test_predictions'])
            
            # 5. Train Transformer
            if self.models_available['transformer']:
                transformer_results = self._train_transformer(X_train_seq, y_train, X_test_seq, y_test)
                training_results['model_performances']['transformer'] = transformer_results
                base_predictions_train.append(transformer_results['train_predictions'])
                base_predictions_test.append(transformer_results['test_predictions'])
            
            # Train meta-learner
            if base_predictions_train:
                base_predictions_train = np.column_stack(base_predictions_train)
                base_predictions_test = np.column_stack(base_predictions_test)
                
                self.meta_learner.fit(base_predictions_train, y_train)
                
                # Ensemble predictions
                ensemble_train_pred = self.meta_learner.predict(base_predictions_train)
                ensemble_test_pred = self.meta_learner.predict(base_predictions_test)
                
                # Ensemble performance
                training_results['ensemble_performance'] = {
                    'train_rmse': np.sqrt(mean_squared_error(y_train, ensemble_train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_test_pred)),
                    'train_r2': r2_score(y_train, ensemble_train_pred),
                    'test_r2': r2_score(y_test, ensemble_test_pred),
                    'train_mae': mean_absolute_error(y_train, ensemble_train_pred),
                    'test_mae': mean_absolute_error(y_test, ensemble_test_pred)
                }
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            training_results['training_time'] = training_time
            
            # Store training history
            self.training_history.append({
                'timestamp': start_time,
                'num_patterns': len(patterns),
                'training_time': training_time,
                'performance': training_results['ensemble_performance']
            })
            
            self.logger.info(f"Ensemble training completed in {training_time:.2f} seconds")
            self.logger.info(f"Ensemble test R²: {training_results['ensemble_performance'].get('test_r2', 0):.4f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            raise
    
    def _prepare_sequence_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare sequence data for LSTM and Transformer models"""
        try:
            sequence_length = self.config['lstm_params']['sequence_length']
            
            # Create overlapping sequences
            if len(X) < sequence_length:
                # Pad if insufficient data
                padding = np.zeros((sequence_length - len(X), X.shape[1]))
                X_padded = np.vstack([padding, X])
                return X_padded.reshape(1, sequence_length, X.shape[1])
            
            sequences = []
            for i in range(len(X) - sequence_length + 1):
                sequences.append(X[i:i + sequence_length])
            
            return np.array(sequences)
            
        except Exception as e:
            self.logger.warning(f"Error preparing sequence data: {e}")
            # Fallback: repeat last sample
            seq_len = self.config['lstm_params']['sequence_length']
            repeated = np.repeat(X[-1:], seq_len, axis=0)
            return repeated.reshape(1, seq_len, X.shape[1])
    
    def _train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM model"""
        try:
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Train model
            self.models['lightgbm'] = lgb.train(
                self.config['lgb_params'],
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            # Predictions
            train_pred = self.models['lightgbm'].predict(X_train)
            test_pred = self.models['lightgbm'].predict(X_test)
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names, 
                self.models['lightgbm'].feature_importance()
            ))
            
            return {
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.warning(f"Error training LightGBM: {e}")
            return self._create_dummy_results(len(y_train), len(y_test))
    
    def _train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train CatBoost model"""
        try:
            # Initialize model
            self.models['catboost'] = cb.CatBoostRegressor(**self.config['cb_params'])
            
            # Train model
            self.models['catboost'].fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Predictions
            train_pred = self.models['catboost'].predict(X_train)
            test_pred = self.models['catboost'].predict(X_test)
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.models['catboost'].feature_importances_
            ))
            
            return {
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.warning(f"Error training CatBoost: {e}")
            return self._create_dummy_results(len(y_train), len(y_test))
    
    def _train_tabnet(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train TabNet model"""
        try:
            # Initialize model
            self.models['tabnet'] = TabNetRegressor(**self.config['tabnet_params'])
            
            # Train model
            self.models['tabnet'].fit(
                X_train, y_train.reshape(-1, 1),
                eval_set=[(X_test, y_test.reshape(-1, 1))],
                max_epochs=200,
                patience=20,
                batch_size=256,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            
            # Predictions
            train_pred = self.models['tabnet'].predict(X_train).flatten()
            test_pred = self.models['tabnet'].predict(X_test).flatten()
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.models['tabnet'].feature_importances_
            ))
            
            return {
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.warning(f"Error training TabNet: {e}")
            return self._create_dummy_results(len(y_train), len(y_test))
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model"""
        try:
            # Model parameters
            input_size = X_train.shape[-1]
            lstm_params = self.config['lstm_params']
            
            # Initialize model
            self.models['lstm'] = LSTMPatternModel(
                input_size=input_size,
                hidden_size=lstm_params['hidden_size'],
                num_layers=lstm_params['num_layers'],
                dropout=lstm_params['dropout']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.models['lstm'].parameters(), lr=lstm_params['learning_rate'])
            
            # Prepare data
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
                X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
            
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            # Training loop
            self.models['lstm'].train()
            for epoch in range(lstm_params['epochs']):
                optimizer.zero_grad()
                outputs = self.models['lstm'](X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Predictions
            self.models['lstm'].eval()
            with torch.no_grad():
                train_pred = self.models['lstm'](X_train_tensor).cpu().numpy().flatten()
                test_pred = self.models['lstm'](X_test_tensor).cpu().numpy().flatten()
            
            return {
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'feature_importance': {}  # LSTM doesn't provide feature importance
            }
            
        except Exception as e:
            self.logger.warning(f"Error training LSTM: {e}")
            return self._create_dummy_results(len(y_train), len(y_test))
    
    def _train_transformer(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Transformer model"""
        try:
            # Model parameters
            input_size = X_train.shape[-1]
            transformer_params = self.config['transformer_params']
            
            # Initialize model
            self.models['transformer'] = TransformerPatternModel(
                input_size=input_size,
                d_model=transformer_params['d_model'],
                nhead=transformer_params['nhead'],
                num_layers=transformer_params['num_layers'],
                dropout=transformer_params['dropout']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.models['transformer'].parameters(), lr=transformer_params['learning_rate'])
            
            # Prepare data
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
                X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
            
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            # Training loop
            self.models['transformer'].train()
            for epoch in range(transformer_params['epochs']):
                optimizer.zero_grad()
                outputs = self.models['transformer'](X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Predictions
            self.models['transformer'].eval()
            with torch.no_grad():
                train_pred = self.models['transformer'](X_train_tensor).cpu().numpy().flatten()
                test_pred = self.models['transformer'](X_test_tensor).cpu().numpy().flatten()
            
            return {
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'feature_importance': {}  # Transformer doesn't provide traditional feature importance
            }
            
        except Exception as e:
            self.logger.warning(f"Error training Transformer: {e}")
            return self._create_dummy_results(len(y_train), len(y_test))
    
    def _create_dummy_results(self, train_size: int, test_size: int) -> Dict[str, Any]:
        """Create dummy results for failed model training"""
        return {
            'train_predictions': np.full(train_size, 0.5),
            'test_predictions': np.full(test_size, 0.5),
            'train_rmse': 1.0,
            'test_rmse': 1.0,
            'train_r2': 0.0,
            'test_r2': 0.0,
            'feature_importance': {}
        }
    
    def predict(self, pattern: PatternSchema, 
               market_data: Optional[Dict[str, Any]] = None) -> EnsemblePrediction:
        """
        Make ensemble prediction for a pattern
        
        Args:
            pattern: Pattern to score
            market_data: Current market data for context
            
        Returns:
            Complete ensemble prediction with individual model results
        """
        try:
            start_time = datetime.now()
            
            # Extract features
            pattern_features = self._extract_pattern_features(pattern)
            if pattern_features is None:
                return self._create_failed_prediction(pattern.pattern_id, "Feature extraction failed")
            
            # Scale features
            if 'standard' in self.scalers:
                features_scaled = self.scalers['standard'].transform(pattern_features.reshape(1, -1))
            else:
                features_scaled = pattern_features.reshape(1, -1)
            
            # Prepare sequence data
            features_seq = self._prepare_sequence_data(features_scaled)
            
            # Individual model predictions
            model_predictions = {}
            base_predictions = []
            
            # 1. LightGBM prediction
            if 'lightgbm' in self.models and self.models['lightgbm'] is not None:
                lgb_pred = self._predict_lightgbm(pattern_features.reshape(1, -1))
                model_predictions['lightgbm'] = lgb_pred
                base_predictions.append(lgb_pred.prediction)
            
            # 2. CatBoost prediction
            if 'catboost' in self.models and self.models['catboost'] is not None:
                cb_pred = self._predict_catboost(pattern_features.reshape(1, -1))
                model_predictions['catboost'] = cb_pred
                base_predictions.append(cb_pred.prediction)
            
            # 3. TabNet prediction
            if 'tabnet' in self.models and self.models['tabnet'] is not None:
                tabnet_pred = self._predict_tabnet(features_scaled)
                model_predictions['tabnet'] = tabnet_pred
                base_predictions.append(tabnet_pred.prediction)
            
            # 4. LSTM prediction
            if 'lstm' in self.models and self.models['lstm'] is not None:
                lstm_pred = self._predict_lstm(features_seq)
                model_predictions['lstm'] = lstm_pred
                base_predictions.append(lstm_pred.prediction)
            
            # 5. Transformer prediction
            if 'transformer' in self.models and self.models['transformer'] is not None:
                transformer_pred = self._predict_transformer(features_seq)
                model_predictions['transformer'] = transformer_pred
                base_predictions.append(transformer_pred.prediction)
            
            # Ensemble prediction
            if base_predictions and self.meta_learner is not None:
                base_pred_array = np.array(base_predictions).reshape(1, -1)
                ensemble_score = float(self.meta_learner.predict(base_pred_array)[0])
            else:
                # Fallback to simple average
                ensemble_score = float(np.mean(base_predictions)) if base_predictions else 0.5
            
            # Calculate ensemble metrics
            ensemble_confidence = self._calculate_ensemble_confidence(base_predictions)
            prediction_variance = self._calculate_prediction_variance(base_predictions)
            model_agreement = self._calculate_model_agreement(base_predictions)
            
            # Calculate meta-features
            volatility_regime_score = self._calculate_volatility_regime_score(pattern, market_data)
            trend_regime_score = self._calculate_trend_regime_score(pattern, market_data)
            pattern_complexity_score = self._calculate_pattern_complexity_score(pattern)
            temporal_consistency_score = self._calculate_temporal_consistency_score(pattern)
            
            # Quality assessment
            prediction_quality = self._assess_prediction_quality(
                ensemble_confidence, model_agreement, prediction_variance
            )
            recommendation = self._generate_recommendation(ensemble_score, ensemble_confidence)
            risk_score = self._calculate_risk_score(pattern, ensemble_score, prediction_variance)
            
            # Create ensemble prediction result
            result = EnsemblePrediction(
                pattern_id=pattern.pattern_id,
                timestamp=datetime.now(),
                lightgbm_prediction=model_predictions.get('lightgbm'),
                catboost_prediction=model_predictions.get('catboost'),
                tabnet_prediction=model_predictions.get('tabnet'),
                lstm_prediction=model_predictions.get('lstm'),
                transformer_prediction=model_predictions.get('transformer'),
                ensemble_score=ensemble_score,
                ensemble_confidence=ensemble_confidence,
                prediction_variance=prediction_variance,
                model_agreement=model_agreement,
                volatility_regime_score=volatility_regime_score,
                trend_regime_score=trend_regime_score,
                pattern_complexity_score=pattern_complexity_score,
                temporal_consistency_score=temporal_consistency_score,
                prediction_quality=prediction_quality,
                recommendation=recommendation,
                risk_score=risk_score
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return self._create_failed_prediction(pattern.pattern_id, str(e))
    
    def _predict_lightgbm(self, features: np.ndarray) -> ModelPrediction:
        """Make LightGBM prediction"""
        try:
            start_time = datetime.now()
            prediction = float(self.models['lightgbm'].predict(features)[0])
            
            # Feature importance from model
            feature_importance = dict(zip(
                self.feature_names,
                self.models['lightgbm'].feature_importance()
            ))
            
            # Calculate confidence based on prediction certainty
            confidence = self._calculate_lgb_confidence(prediction, features)
            
            return ModelPrediction(
                model_name="lightgbm",
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                prediction_time=(datetime.now() - start_time).total_seconds(),
                model_version="1.0.0"
            )
            
        except Exception as e:
            self.logger.warning(f"Error in LightGBM prediction: {e}")
            return self._create_failed_model_prediction("lightgbm")
    
    def _predict_catboost(self, features: np.ndarray) -> ModelPrediction:
        """Make CatBoost prediction"""
        try:
            start_time = datetime.now()
            prediction = float(self.models['catboost'].predict(features)[0])
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.models['catboost'].feature_importances_
            ))
            
            # Calculate confidence
            confidence = self._calculate_cb_confidence(prediction, features)
            
            return ModelPrediction(
                model_name="catboost",
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                prediction_time=(datetime.now() - start_time).total_seconds(),
                model_version="1.0.0"
            )
            
        except Exception as e:
            self.logger.warning(f"Error in CatBoost prediction: {e}")
            return self._create_failed_model_prediction("catboost")
    
    def _predict_tabnet(self, features: np.ndarray) -> ModelPrediction:
        """Make TabNet prediction"""
        try:
            start_time = datetime.now()
            prediction = float(self.models['tabnet'].predict(features)[0])
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.models['tabnet'].feature_importances_
            ))
            
            # Calculate confidence
            confidence = self._calculate_tabnet_confidence(prediction)
            
            return ModelPrediction(
                model_name="tabnet",
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                prediction_time=(datetime.now() - start_time).total_seconds(),
                model_version="1.0.0"
            )
            
        except Exception as e:
            self.logger.warning(f"Error in TabNet prediction: {e}")
            return self._create_failed_model_prediction("tabnet")
    
    def _predict_lstm(self, features: np.ndarray) -> ModelPrediction:
        """Make LSTM prediction"""
        try:
            start_time = datetime.now()
            
            # Prepare tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Make prediction
            self.models['lstm'].eval()
            with torch.no_grad():
                prediction = float(self.models['lstm'](features_tensor).cpu().numpy()[0])
            
            # LSTM doesn't provide feature importance
            feature_importance = {}
            
            # Calculate confidence based on prediction stability
            confidence = self._calculate_lstm_confidence(prediction)
            
            return ModelPrediction(
                model_name="lstm",
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                prediction_time=(datetime.now() - start_time).total_seconds(),
                model_version="1.0.0"
            )
            
        except Exception as e:
            self.logger.warning(f"Error in LSTM prediction: {e}")
            return self._create_failed_model_prediction("lstm")
    
    def _predict_transformer(self, features: np.ndarray) -> ModelPrediction:
        """Make Transformer prediction"""
        try:
            start_time = datetime.now()
            
            # Prepare tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Make prediction
            self.models['transformer'].eval()
            with torch.no_grad():
                prediction = float(self.models['transformer'](features_tensor).cpu().numpy()[0])
            
            # Transformer doesn't provide traditional feature importance
            feature_importance = {}
            
            # Calculate confidence
            confidence = self._calculate_transformer_confidence(prediction)
            
            return ModelPrediction(
                model_name="transformer",
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                prediction_time=(datetime.now() - start_time).total_seconds(),
                model_version="1.0.0"
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Transformer prediction: {e}")
            return self._create_failed_model_prediction("transformer")
    
    def _calculate_ensemble_confidence(self, predictions: List[float]) -> float:
        """Calculate ensemble confidence based on prediction agreement"""
        if not predictions or len(predictions) < 2:
            return 0.5
        
        # Calculate coefficient of variation (lower = higher confidence)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        if mean_pred == 0:
            return 0.5
        
        cv = std_pred / mean_pred
        confidence = max(0.0, 1.0 - cv)
        
        return min(1.0, confidence)
    
    def _calculate_prediction_variance(self, predictions: List[float]) -> float:
        """Calculate prediction variance across models"""
        if not predictions:
            return 1.0
        
        return float(np.var(predictions))
    
    def _calculate_model_agreement(self, predictions: List[float]) -> float:
        """Calculate model agreement score"""
        if not predictions or len(predictions) < 2:
            return 0.5
        
        # Calculate pairwise correlations
        agreements = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                # Simple agreement: 1 - abs(diff)
                agreement = 1.0 - abs(predictions[i] - predictions[j])
                agreements.append(agreement)
        
        return float(np.mean(agreements)) if agreements else 0.5
    
    def _calculate_volatility_regime_score(self, pattern: PatternSchema, 
                                         market_data: Optional[Dict]) -> float:
        """Calculate volatility regime compatibility score"""
        try:
            # Pattern volatility preference
            pattern_vol = pattern.market_context.get('volatility_environment', 'medium')
            
            # Current market volatility
            if market_data:
                current_vol = market_data.get('volatility', 0.02)
                if current_vol > 0.03:
                    current_vol_cat = 'high'
                elif current_vol > 0.015:
                    current_vol_cat = 'medium'
                else:
                    current_vol_cat = 'low'
            else:
                current_vol_cat = 'medium'
            
            # Compatibility matrix
            compatibility = {
                ('low', 'low'): 1.0,
                ('low', 'medium'): 0.7,
                ('low', 'high'): 0.3,
                ('medium', 'low'): 0.7,
                ('medium', 'medium'): 1.0,
                ('medium', 'high'): 0.7,
                ('high', 'low'): 0.3,
                ('high', 'medium'): 0.7,
                ('high', 'high'): 1.0
            }
            
            return compatibility.get((pattern_vol, current_vol_cat), 0.5)
            
        except Exception:
            return 0.5
    
    def _calculate_trend_regime_score(self, pattern: PatternSchema,
                                    market_data: Optional[Dict]) -> float:
        """Calculate trend regime compatibility score"""
        try:
            pattern_trend = pattern.market_context.get('trend_environment', 'neutral')
            
            if market_data:
                trend_strength = market_data.get('trend_strength', 0.0)
                if trend_strength > 0.3:
                    current_trend = 'strong_bullish'
                elif trend_strength > 0.1:
                    current_trend = 'bullish'
                elif trend_strength > -0.1:
                    current_trend = 'neutral'
                elif trend_strength > -0.3:
                    current_trend = 'bearish'
                else:
                    current_trend = 'strong_bearish'
            else:
                current_trend = 'neutral'
            
            # Trend compatibility scoring
            if pattern_trend == current_trend:
                return 1.0
            elif pattern_trend == 'neutral' or current_trend == 'neutral':
                return 0.7
            elif (pattern_trend in ['bullish', 'strong_bullish'] and 
                  current_trend in ['bullish', 'strong_bullish']):
                return 0.8
            elif (pattern_trend in ['bearish', 'strong_bearish'] and 
                  current_trend in ['bearish', 'strong_bearish']):
                return 0.8
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _calculate_pattern_complexity_score(self, pattern: PatternSchema) -> float:
        """Calculate pattern complexity score"""
        try:
            # Factors that increase complexity
            complexity_factors = []
            
            # Number of components
            num_components = len(pattern.components)
            complexity_factors.append(min(1.0, num_components / 10))
            
            # Number of timeframes
            num_timeframes = len(pattern.timeframe_analysis)
            complexity_factors.append(min(1.0, num_timeframes / 4))
            
            # Cross-timeframe confluence
            confluence_score = pattern.cross_timeframe_confluence.get('alignment_score', 0.5)
            complexity_factors.append(confluence_score)
            
            # Market context specificity
            context_specificity = 0.0
            if pattern.market_context.get('preferred_regime') != 'any':
                context_specificity += 0.25
            if pattern.market_context.get('volatility_environment') != 'any':
                context_specificity += 0.25
            if pattern.market_context.get('trend_environment') != 'any':
                context_specificity += 0.25
            if pattern.market_context.get('time_of_day_preference') != 'any':
                context_specificity += 0.25
            
            complexity_factors.append(context_specificity)
            
            return float(np.mean(complexity_factors))
            
        except Exception:
            return 0.5
    
    def _calculate_temporal_consistency_score(self, pattern: PatternSchema) -> float:
        """Calculate temporal consistency score across timeframes"""
        try:
            timeframe_scores = []
            timeframe_strengths = []
            
            for tf_data in pattern.timeframe_analysis.values():
                timeframe_scores.append(tf_data.get('validation_score', 0.0))
                timeframe_strengths.append(tf_data.get('strength', 0.0))
            
            if not timeframe_scores:
                return 0.5
            
            # Consistency = 1 - coefficient of variation
            score_std = np.std(timeframe_scores)
            score_mean = np.mean(timeframe_scores)
            
            if score_mean == 0:
                return 0.5
            
            cv = score_std / score_mean
            consistency = max(0.0, 1.0 - cv)
            
            # Bonus for high average strength
            avg_strength = np.mean(timeframe_strengths)
            strength_bonus = avg_strength * 0.2
            
            return min(1.0, consistency + strength_bonus)
            
        except Exception:
            return 0.5
    
    def _assess_prediction_quality(self, confidence: float, agreement: float, variance: float) -> str:
        """Assess overall prediction quality"""
        # Weighted quality score
        quality_score = (confidence * 0.4 + agreement * 0.4 + (1.0 - variance) * 0.2)
        
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.65:
            return 'good'
        elif quality_score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_recommendation(self, ensemble_score: float, confidence: float) -> str:
        """Generate trading recommendation"""
        # Adjust score by confidence
        adjusted_score = ensemble_score * confidence
        
        if adjusted_score >= 0.8:
            return 'strong_buy'
        elif adjusted_score >= 0.65:
            return 'buy'
        elif adjusted_score >= 0.4:
            return 'hold'
        else:
            return 'avoid'
    
    def _calculate_risk_score(self, pattern: PatternSchema, ensemble_score: float, variance: float) -> float:
        """Calculate risk score for the pattern"""
        try:
            # Base risk from historical performance
            max_drawdown = abs(pattern.historical_performance.get('max_drawdown', 0.1))
            volatility = pattern.historical_performance.get('return_std', 0.1)
            
            # Risk from prediction uncertainty
            prediction_risk = variance * 2.0  # Scale variance
            
            # Risk from pattern complexity
            complexity_risk = self._calculate_pattern_complexity_score(pattern) * 0.3
            
            # Combined risk (higher = riskier)
            total_risk = (max_drawdown * 0.4 + volatility * 0.3 + 
                         prediction_risk * 0.2 + complexity_risk * 0.1)
            
            return min(1.0, total_risk)
            
        except Exception:
            return 0.5
    
    def _calculate_lgb_confidence(self, prediction: float, features: np.ndarray) -> float:
        """Calculate LightGBM prediction confidence"""
        # Simple confidence based on prediction extremity
        confidence = abs(prediction - 0.5) * 2.0
        return min(1.0, max(0.1, confidence))
    
    def _calculate_cb_confidence(self, prediction: float, features: np.ndarray) -> float:
        """Calculate CatBoost prediction confidence"""
        confidence = abs(prediction - 0.5) * 2.0
        return min(1.0, max(0.1, confidence))
    
    def _calculate_tabnet_confidence(self, prediction: float) -> float:
        """Calculate TabNet prediction confidence"""
        confidence = abs(prediction - 0.5) * 2.0
        return min(1.0, max(0.1, confidence))
    
    def _calculate_lstm_confidence(self, prediction: float) -> float:
        """Calculate LSTM prediction confidence"""
        confidence = abs(prediction - 0.5) * 2.0
        return min(1.0, max(0.1, confidence))
    
    def _calculate_transformer_confidence(self, prediction: float) -> float:
        """Calculate Transformer prediction confidence"""
        confidence = abs(prediction - 0.5) * 2.0
        return min(1.0, max(0.1, confidence))
    
    def _create_failed_model_prediction(self, model_name: str) -> ModelPrediction:
        """Create failed model prediction"""
        return ModelPrediction(
            model_name=model_name,
            prediction=0.5,
            confidence=0.0,
            feature_importance={},
            prediction_time=0.0,
            model_version="error"
        )
    
    def _create_failed_prediction(self, pattern_id: str, error_message: str) -> EnsemblePrediction:
        """Create failed ensemble prediction"""
        return EnsemblePrediction(
            pattern_id=pattern_id,
            timestamp=datetime.now(),
            lightgbm_prediction=None,
            catboost_prediction=None,
            tabnet_prediction=None,
            lstm_prediction=None,
            transformer_prediction=None,
            ensemble_score=0.5,
            ensemble_confidence=0.0,
            prediction_variance=1.0,
            model_agreement=0.0,
            volatility_regime_score=0.5,
            trend_regime_score=0.5,
            pattern_complexity_score=0.5,
            temporal_consistency_score=0.5,
            prediction_quality='poor',
            recommendation='avoid',
            risk_score=1.0
        )
    
    def save_models(self, save_path: str):
        """Save trained models to disk"""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save individual models
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = save_path / f"{model_name}_model.pkl"
                    
                    if model_name in ['lightgbm']:
                        model.save_model(str(model_path))
                    elif model_name in ['catboost']:
                        model.save_model(str(model_path))
                    elif model_name in ['tabnet']:
                        model.save_model(str(model_path))
                    elif model_name in ['lstm', 'transformer']:
                        torch.save(model.state_dict(), str(model_path))
            
            # Save meta-learner
            if self.meta_learner is not None:
                joblib.dump(self.meta_learner, save_path / "meta_learner.pkl")
            
            # Save scalers
            joblib.dump(self.scalers, save_path / "scalers.pkl")
            
            # Save feature names
            joblib.dump(self.feature_names, save_path / "feature_names.pkl")
            
            # Save config
            joblib.dump(self.config, save_path / "config.pkl")
            
            self.logger.info(f"Models saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, load_path: str):
        """Load trained models from disk"""
        try:
            load_path = Path(load_path)
            
            # Load config
            config_path = load_path / "config.pkl"
            if config_path.exists():
                self.config = joblib.load(config_path)
            
            # Load feature names
            feature_names_path = load_path / "feature_names.pkl"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
            
            # Load scalers
            scalers_path = load_path / "scalers.pkl"
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
            
            # Load individual models
            for model_name in self.models.keys():
                model_path = load_path / f"{model_name}_model.pkl"
                
                if model_path.exists():
                    if model_name == 'lightgbm' and self.models_available['lightgbm']:
                        self.models[model_name] = lgb.Booster(model_file=str(model_path))
                    elif model_name == 'catboost' and self.models_available['catboost']:
                        self.models[model_name] = cb.CatBoostRegressor()
                        self.models[model_name].load_model(str(model_path))
                    elif model_name == 'tabnet' and self.models_available['tabnet']:
                        self.models[model_name] = TabNetRegressor()
                        self.models[model_name].load_model(str(model_path))
                    elif model_name in ['lstm', 'transformer'] and self.models_available[model_name]:
                        # Recreate model architecture
                        if model_name == 'lstm':
                            input_size = len(self.feature_names) if self.feature_names else 50
                            self.models[model_name] = LSTMPatternModel(input_size).to(self.device)
                        else:  # transformer
                            input_size = len(self.feature_names) if self.feature_names else 50
                            self.models[model_name] = TransformerPatternModel(input_size).to(self.device)
                        
                        self.models[model_name].load_state_dict(torch.load(str(model_path), map_location=self.device))
                        self.models[model_name].eval()
            
            # Load meta-learner
            meta_learner_path = load_path / "meta_learner.pkl"
            if meta_learner_path.exists():
                self.meta_learner = joblib.load(meta_learner_path)
            
            self.logger.info(f"Models loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        try:
            available_models = [name for name, available in self.models_available.items() if available]
            trained_models = [name for name, model in self.models.items() if model is not None]
            
            return {
                "ensemble_type": "5-Model Advanced ML Ensemble",
                "available_models": available_models,
                "trained_models": trained_models,
                "model_count": len(trained_models),
                "feature_count": len(self.feature_names),
                "meta_learner": "Ridge Regression" if self.meta_learner else None,
                "device": str(self.device),
                "training_history_length": len(self.training_history),
                "last_training": self.training_history[-1] if self.training_history else None,
                "model_performances": self.model_performances,
                "config": {
                    "ensemble_method": self.config.get('ensemble_method'),
                    "test_size": self.config.get('test_size'),
                    "cv_folds": self.config.get('cv_folds')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble summary: {e}")
            return {"error": str(e)}