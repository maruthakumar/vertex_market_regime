"""
Baseline Model Components for Market Regime Training Pipeline
Implementation of TabNet, XGBoost, LSTM, and Temporal Fusion Transformer models

This module provides:
- TabNet regime classifier (primary model)
- XGBoost alternative classifier (baseline comparison)
- LSTM transition forecaster (sequence modeling)
- Temporal Fusion Transformer (advanced forecasting)
- Model selection and configuration framework
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from datetime import datetime
import json


class BaseMarketRegimeModel(ABC):
    """Abstract base class for market regime models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        self.feature_importance = {}
        self.model_metadata = {
            "model_type": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "config": config
        }
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass
    
    def save_model(self, path: str) -> str:
        """Save model to disk"""
        try:
            model_data = {
                "model": self.model,
                "config": self.config,
                "training_metrics": self.training_metrics,
                "feature_importance": self.feature_importance,
                "model_metadata": self.model_metadata,
                "is_trained": self.is_trained
            }
            
            joblib.dump(model_data, path)
            self.logger.info(f"Model saved to: {path}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load model from disk"""
        try:
            model_data = joblib.load(path)
            self.model = model_data["model"]
            self.config = model_data["config"]
            self.training_metrics = model_data["training_metrics"]
            self.feature_importance = model_data["feature_importance"]
            self.model_metadata = model_data["model_metadata"]
            self.is_trained = model_data["is_trained"]
            
            self.logger.info(f"Model loaded from: {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise


class TabNetRegimeClassifier(BaseMarketRegimeModel):
    """TabNet architecture for market regime classification"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # TabNet specific configuration
        self.tabnet_config = {
            "n_d": config.get("n_d", 64),
            "n_a": config.get("n_a", 64),
            "n_steps": config.get("n_steps", 5),
            "gamma": config.get("gamma", 1.3),
            "lambda_sparse": config.get("lambda_sparse", 1e-3),
            "optimizer_fn": getattr(optim, config.get("optimizer_fn", "Adam")),
            "optimizer_params": config.get("optimizer_params", {"lr": 2e-2}),
            "scheduler_params": config.get("scheduler_params", {"step_size": 50, "gamma": 0.9}),
            "mask_type": config.get("mask_type", "entmax"),
            "device_name": config.get("device_name", "auto"),
            "verbose": config.get("verbose", 1)
        }
        
        # Training configuration
        self.max_epochs = config.get("max_epochs", 200)
        self.patience = config.get("patience", 20)
        self.batch_size = config.get("batch_size", 1024)
        self.eval_metric = config.get("eval_metric", ["accuracy", "logloss"])
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train TabNet model"""
        
        self.logger.info("Starting TabNet training...")
        training_start_time = datetime.now()
        
        try:
            # Initialize TabNet classifier
            self.model = TabNetClassifier(**self.tabnet_config)
            
            # Prepare validation data
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            else:
                # Split training data for validation
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                X_train, y_train = X_train_split, y_train_split
                eval_set = [(X_val_split, y_val_split)]
            
            # Train the model
            self.model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=eval_set,
                eval_name=["validation"],
                eval_metric=self.eval_metric,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                virtual_batch_size=self.batch_size // 4,
                num_workers=0,
                drop_last=False
            )
            
            self.is_trained = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_f1 = f1_score(y_train, train_predictions, average='weighted')
            
            if X_val is not None:
                val_predictions = self.model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_predictions)
                val_f1 = f1_score(y_val, val_predictions, average='weighted')
            else:
                val_predictions = self.model.predict(X_val_split)
                val_accuracy = accuracy_score(y_val_split, val_predictions)
                val_f1 = f1_score(y_val_split, val_predictions, average='weighted')
            
            training_time = (datetime.now() - training_start_time).total_seconds()
            
            self.training_metrics = {
                "train_accuracy": float(train_accuracy),
                "train_f1_score": float(train_f1),
                "val_accuracy": float(val_accuracy),
                "val_f1_score": float(val_f1),
                "training_time_seconds": training_time,
                "best_epoch": int(self.model.best_epoch),
                "best_cost": float(self.model.best_cost)
            }
            
            # Get feature importance
            self.feature_importance = self._calculate_feature_importance()
            
            self.logger.info(f"TabNet training completed in {training_time:.1f} seconds")
            self.logger.info(f"Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"TabNet training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance from TabNet"""
        try:
            # Get feature importance from TabNet
            importance = self.model.feature_importances_
            
            # Create feature importance dictionary
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
            
            return importance_dict
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {str(e)}")
            return {}


class XGBoostRegimeClassifier(BaseMarketRegimeModel):
    """XGBoost alternative for market regime classification"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # XGBoost specific configuration
        self.xgb_config = {
            "max_depth": config.get("max_depth", 6),
            "learning_rate": config.get("learning_rate", 0.1),
            "n_estimators": config.get("n_estimators", 1000),
            "subsample": config.get("subsample", 0.8),
            "colsample_bytree": config.get("colsample_bytree", 0.8),
            "reg_alpha": config.get("reg_alpha", 0.1),
            "reg_lambda": config.get("reg_lambda", 0.1),
            "objective": config.get("objective", "multi:softprob"),
            "eval_metric": config.get("eval_metric", "mlogloss"),
            "random_state": config.get("random_state", 42),
            "n_jobs": config.get("n_jobs", -1)
        }
        
        # Training configuration
        self.early_stopping_rounds = config.get("early_stopping_rounds", 50)
        self.verbose = config.get("verbose", False)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train XGBoost model"""
        
        self.logger.info("Starting XGBoost training...")
        training_start_time = datetime.now()
        
        try:
            # Initialize XGBoost classifier
            self.model = xgb.XGBClassifier(**self.xgb_config)
            
            # Prepare validation data
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            else:
                # Split training data for validation
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                X_train, y_train = X_train_split, y_train_split
                eval_set = [(X_val_split, y_val_split)]
            
            # Train the model
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=self.verbose
            )
            
            self.is_trained = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_f1 = f1_score(y_train, train_predictions, average='weighted')
            
            if X_val is not None:
                val_predictions = self.model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_predictions)
                val_f1 = f1_score(y_val, val_predictions, average='weighted')
            else:
                val_predictions = self.model.predict(X_val_split)
                val_accuracy = accuracy_score(y_val_split, val_predictions)
                val_f1 = f1_score(y_val_split, val_predictions, average='weighted')
            
            training_time = (datetime.now() - training_start_time).total_seconds()
            
            self.training_metrics = {
                "train_accuracy": float(train_accuracy),
                "train_f1_score": float(train_f1),
                "val_accuracy": float(val_accuracy),
                "val_f1_score": float(val_f1),
                "training_time_seconds": training_time,
                "best_iteration": int(self.model.best_iteration),
                "best_score": float(self.model.best_score)
            }
            
            # Get feature importance
            self.feature_importance = self._calculate_feature_importance()
            
            self.logger.info(f"XGBoost training completed in {training_time:.1f} seconds")
            self.logger.info(f"Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance from XGBoost"""
        try:
            # Get feature importance from XGBoost
            importance = self.model.feature_importances_
            
            # Create feature importance dictionary
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
            
            return importance_dict
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {str(e)}")
            return {}


class LSTMTransitionForecaster(BaseMarketRegimeModel):
    """LSTM-based transition forecaster for market regime changes"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # LSTM specific configuration
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.sequence_length = config.get("sequence_length", 60)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 15)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
    
    def _create_model(self, input_size: int, num_classes: int) -> nn.Module:
        """Create LSTM model"""
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, num_classes)
                
            def forward(self, x):
                batch_size = x.size(0)
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                
                out, (hn, cn) = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])  # Take last output
                out = self.fc(out)
                return out
        
        return LSTMModel(input_size, self.hidden_size, self.num_layers, num_classes, self.dropout)
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train LSTM model"""
        
        self.logger.info("Starting LSTM training...")
        training_start_time = datetime.now()
        
        try:
            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            num_classes = len(self.label_encoder.classes_)
            
            # Prepare sequences
            X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train_encoded)
            
            if X_val is not None and y_val is not None:
                y_val_encoded = self.label_encoder.transform(y_val)
                X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val_encoded)
            else:
                # Split training data for validation
                split_idx = int(len(X_train_seq) * 0.8)
                X_val_seq = X_train_seq[split_idx:]
                y_val_seq = y_train_seq[split_idx:]
                X_train_seq = X_train_seq[:split_idx]
                y_train_seq = y_train_seq[:split_idx]
            
            # Create model
            input_size = X_train_seq.shape[2]
            self.model = self._create_model(input_size, num_classes).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_seq),
                torch.LongTensor(y_train_seq)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq),
                torch.LongTensor(y_val_seq)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            best_val_acc = 0
            patience_counter = 0
            train_losses = []
            val_accuracies = []
            
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                train_losses.append(avg_loss)
                
                # Validation phase
                self.model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_acc = val_correct / val_total
                val_accuracies.append(val_acc)
                
                scheduler.step(1 - val_acc)  # ReduceLROnPlateau expects loss, so use 1-accuracy
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
            
            self.is_trained = True
            
            # Calculate final metrics
            training_time = (datetime.now() - training_start_time).total_seconds()
            
            self.training_metrics = {
                "best_val_accuracy": float(best_val_acc),
                "final_train_loss": float(train_losses[-1]),
                "training_time_seconds": training_time,
                "epochs_trained": epoch + 1,
                "sequence_length": self.sequence_length
            }
            
            self.logger.info(f"LSTM training completed in {training_time:.1f} seconds")
            self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare sequences
        X_seq = []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
        
        if len(X_seq) == 0:
            raise ValueError(f"Input data must have at least {self.sequence_length} samples")
        
        X_seq = np.array(X_seq)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels = self.label_encoder.inverse_transform(predicted.cpu().numpy())
        
        return predicted_labels
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare sequences
        X_seq = []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
        
        if len(X_seq) == 0:
            raise ValueError(f"Input data must have at least {self.sequence_length} samples")
        
        X_seq = np.array(X_seq)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (placeholder for LSTM)"""
        # LSTM doesn't provide direct feature importance
        return {"feature_importance": "Not available for LSTM models"}


class TemporalFusionTransformer(BaseMarketRegimeModel):
    """Temporal Fusion Transformer for advanced transition forecasting (Optional)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # TFT specific configuration
        self.hidden_size = config.get("hidden_size", 128)
        self.attention_heads = config.get("attention_heads", 4)
        self.dropout = config.get("dropout", 0.1)
        self.sequence_length = config.get("sequence_length", 60)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.max_epochs = config.get("max_epochs", 100)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def _create_model(self, input_size: int, num_classes: int) -> nn.Module:
        """Create simplified TFT model (placeholder implementation)"""
        
        class SimplifiedTFT(nn.Module):
            def __init__(self, input_size, hidden_size, attention_heads, num_classes, dropout):
                super(SimplifiedTFT, self).__init__()
                self.hidden_size = hidden_size
                self.attention_heads = attention_heads
                
                # Encoder layers
                self.input_linear = nn.Linear(input_size, hidden_size)
                self.attention = nn.MultiheadAttention(hidden_size, attention_heads, dropout=dropout)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                
                # Feed forward
                self.feed_forward = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                
                # Output layer
                self.output_linear = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # Input projection
                x = self.input_linear(x)
                
                # Self-attention
                x_transposed = x.transpose(0, 1)  # (seq_len, batch, hidden)
                attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
                x = self.norm1(x_transposed + attn_output)
                
                # Feed forward
                ff_output = self.feed_forward(x)
                x = self.norm2(x + ff_output)
                
                # Take last timestep for classification
                x = x[-1, :, :]  # (batch, hidden)
                x = self.dropout(x)
                output = self.output_linear(x)
                
                return output
        
        return SimplifiedTFT(input_size, self.hidden_size, self.attention_heads, num_classes, self.dropout)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train TFT model (simplified implementation)"""
        
        self.logger.info("Starting TFT training...")
        training_start_time = datetime.now()
        
        # Note: This is a simplified implementation
        # In production, would use a proper TFT implementation like pytorch-forecasting
        
        try:
            # For now, return placeholder metrics
            # Full TFT implementation would require additional dependencies
            training_time = (datetime.now() - training_start_time).total_seconds()
            
            self.training_metrics = {
                "note": "TFT implementation is a placeholder",
                "training_time_seconds": training_time,
                "implementation_status": "simplified_placeholder"
            }
            
            self.is_trained = True
            self.logger.info("TFT training completed (placeholder implementation)")
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"TFT training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (placeholder)"""
        # Placeholder implementation
        return np.random.randint(0, 3, size=len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (placeholder)"""
        # Placeholder implementation
        num_classes = 3  # Assume 3 regime classes
        return np.random.rand(len(X), num_classes)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (placeholder)"""
        return {"feature_importance": "Not available for TFT models (placeholder)"}


class ModelSelectionFramework:
    """Framework for model selection and configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.available_models = {
            "tabnet": TabNetRegimeClassifier,
            "xgboost": XGBoostRegimeClassifier,
            "lstm": LSTMTransitionForecaster,
            "tft": TemporalFusionTransformer
        }
        
    def create_model(self, model_type: str, model_config: Dict[str, Any]) -> BaseMarketRegimeModel:
        """Create model instance based on type"""
        
        if model_type not in self.available_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.available_models[model_type]
        return model_class(model_config)
    
    def get_default_config(self, model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type"""
        
        default_configs = {
            "tabnet": {
                "n_d": 64,
                "n_a": 64,
                "n_steps": 5,
                "gamma": 1.3,
                "lambda_sparse": 1e-3,
                "optimizer_fn": "Adam",
                "optimizer_params": {"lr": 2e-2},
                "scheduler_params": {"step_size": 50, "gamma": 0.9},
                "mask_type": "entmax",
                "max_epochs": 200,
                "patience": 20,
                "batch_size": 1024
            },
            "xgboost": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 1000,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "early_stopping_rounds": 50
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "sequence_length": 60,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "patience": 15
            }
        }
        
        return default_configs.get(model_type, {})
    
    def validate_config(self, model_type: str, config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        
        try:
            default_config = self.get_default_config(model_type)
            
            # Basic validation - check if required parameters exist
            for key in default_config:
                if key not in config:
                    self.logger.warning(f"Missing parameter {key} for {model_type}, using default")
                    config[key] = default_config[key]
            
            return True
        except Exception as e:
            self.logger.error(f"Config validation failed: {str(e)}")
            return False


# Utility functions for KFP component integration
def train_baseline_model(
    model_type: str,
    training_data_path: str,
    validation_data_path: str,
    model_config: Dict[str, Any],
    output_model_path: str
) -> Dict[str, Any]:
    """
    Train baseline model for KFP component
    
    This function serves as the entry point for model training in KFP components
    """
    
    try:
        # Load data
        train_data = pd.read_parquet(training_data_path)
        val_data = pd.read_parquet(validation_data_path)
        
        # Separate features and target
        feature_columns = [col for col in train_data.columns if col not in ['target', 'timestamp']]
        X_train = train_data[feature_columns].values
        y_train = train_data['target'].values
        X_val = val_data[feature_columns].values
        y_val = val_data['target'].values
        
        # Create model
        framework = ModelSelectionFramework({})
        model = framework.create_model(model_type, model_config)
        
        # Train model
        training_metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model.save_model(output_model_path)
        
        return {
            "status": "success",
            "model_type": model_type,
            "training_metrics": training_metrics,
            "model_path": output_model_path,
            "feature_importance": model.get_feature_importance()
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }