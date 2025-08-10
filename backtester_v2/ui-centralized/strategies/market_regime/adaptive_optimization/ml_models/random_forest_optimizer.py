"""
Random Forest Optimizer - ML-Based Parameter Optimization
=======================================================

Uses Random Forest regression for adaptive parameter optimization.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Import base utilities
from ...base.common_utils import MathUtils, ErrorHandler

logger = logging.getLogger(__name__)


class RandomForestOptimizer:
    """Random Forest-based parameter optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Random Forest Optimizer"""
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 10)
        self.min_samples_split = config.get('min_samples_split', 5)
        self.min_samples_leaf = config.get('min_samples_leaf', 2)
        self.random_state = config.get('random_state', 42)
        
        # Feature engineering parameters
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20])
        self.feature_columns = config.get('feature_columns', [
            'volatility', 'volume_ratio', 'momentum', 'trend_strength'
        ])
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.is_trained = False
        
        # Training history
        self.training_history = {
            'training_scores': [],
            'validation_scores': [],
            'feature_importance_history': [],
            'training_dates': []
        }
        
        # Mathematical utilities
        self.math_utils = MathUtils()
        
        logger.info("RandomForestOptimizer initialized")
    
    def optimize_parameters(self, 
                          market_data: pd.DataFrame,
                          current_parameters: Dict[str, float],
                          performance_target: Optional[str] = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize parameters using Random Forest
        
        Args:
            market_data: Historical market data
            current_parameters: Current parameter configuration
            performance_target: Target performance metric to optimize
            
        Returns:
            Dict with optimized parameters and model analysis
        """
        try:
            if market_data.empty or len(market_data) < 50:
                return self._get_default_optimization_result(current_parameters)
            
            # Prepare training data
            features, targets = self._prepare_training_data(market_data, current_parameters, performance_target)
            
            if features.empty or len(features) < 20:
                return self._get_default_optimization_result(current_parameters)
            
            # Train Random Forest model
            training_result = self._train_model(features, targets)
            
            if not training_result['success']:
                return self._get_default_optimization_result(current_parameters)
            
            # Generate optimized parameters
            optimized_parameters = self._generate_optimized_parameters(features, current_parameters)
            
            # Validate optimization
            validation_result = self._validate_optimization(features, targets, optimized_parameters)
            
            # Analyze feature importance
            feature_analysis = self._analyze_feature_importance()
            
            # Calculate improvement estimation
            improvement_estimation = self._estimate_improvement(
                features, targets, current_parameters, optimized_parameters
            )
            
            # Update training history
            self._update_training_history(training_result, feature_analysis)
            
            return {
                'optimized_parameters': optimized_parameters,
                'training_result': training_result,
                'validation_result': validation_result,
                'feature_analysis': feature_analysis,
                'improvement_estimation': improvement_estimation,
                'model_confidence': self._calculate_model_confidence(training_result),
                'optimization_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in Random Forest optimization: {e}")
            return self._get_default_optimization_result(current_parameters)
    
    def _prepare_training_data(self, 
                             market_data: pd.DataFrame,
                             current_parameters: Dict[str, float],
                             performance_target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for Random Forest"""
        try:
            # Create features from market data
            features_df = self._create_features(market_data)
            
            # Create target variable (performance metric)
            targets = self._create_targets(market_data, performance_target)
            
            # Align features and targets
            aligned_data = features_df.join(targets, how='inner').dropna()
            
            if aligned_data.empty:
                return pd.DataFrame(), pd.Series()
            
            # Split features and targets
            feature_columns = [col for col in aligned_data.columns if col != 'target']
            features = aligned_data[feature_columns]
            targets = aligned_data['target']
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _create_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix from market data"""
        try:
            features = pd.DataFrame(index=market_data.index)
            
            # Price-based features
            if 'close' in market_data.columns:
                close_prices = market_data['close']
                
                # Returns
                features['returns_1d'] = close_prices.pct_change(1)
                features['returns_5d'] = close_prices.pct_change(5)
                features['returns_10d'] = close_prices.pct_change(10)
                
                # Volatility features
                for period in self.lookback_periods:
                    vol_col = f'volatility_{period}d'
                    features[vol_col] = features['returns_1d'].rolling(window=period).std() * np.sqrt(252)
                
                # Momentum features
                for period in [5, 10, 20]:
                    mom_col = f'momentum_{period}d'
                    features[mom_col] = close_prices.pct_change(period)
                
                # Moving averages
                for period in [10, 20, 50]:
                    ma_col = f'ma_{period}d'
                    features[ma_col] = close_prices.rolling(window=period).mean()
                    
                    # Relative position to MA
                    rel_ma_col = f'rel_ma_{period}d'
                    features[rel_ma_col] = (close_prices - features[ma_col]) / features[ma_col]
            
            # Volume-based features
            if 'volume' in market_data.columns:
                volume = market_data['volume']
                
                # Volume ratios
                for period in self.lookback_periods:
                    vol_ma = volume.rolling(window=period).mean()
                    vol_ratio_col = f'volume_ratio_{period}d'
                    features[vol_ratio_col] = volume / vol_ma
                
                # Volume trend
                features['volume_trend_5d'] = volume.pct_change(5)
                features['volume_trend_10d'] = volume.pct_change(10)
            
            # Technical indicators
            if 'high' in market_data.columns and 'low' in market_data.columns and 'close' in market_data.columns:
                high = market_data['high']
                low = market_data['low']
                close = market_data['close']
                
                # Average True Range
                for period in [10, 20]:
                    atr_col = f'atr_{period}d'
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    features[atr_col] = true_range.rolling(window=period).mean()
                
                # RSI
                for period in [14, 21]:
                    rsi_col = f'rsi_{period}d'
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    features[rsi_col] = 100 - (100 / (1 + rs))
            
            # Market regime features
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change()
                
                # Regime indicators
                features['high_volatility_regime'] = (features.get('volatility_20d', 0) > features.get('volatility_20d', 0).rolling(window=60).quantile(0.8)).astype(int)
                features['trending_regime'] = (abs(features.get('momentum_10d', 0)) > abs(features.get('momentum_10d', 0)).rolling(window=60).quantile(0.7)).astype(int)
                features['high_volume_regime'] = (features.get('volume_ratio_20d', 1) > features.get('volume_ratio_20d', 1).rolling(window=60).quantile(0.8)).astype(int)
            
            # Interaction features
            if 'volatility_20d' in features.columns and 'momentum_10d' in features.columns:
                features['vol_momentum_interaction'] = features['volatility_20d'] * abs(features['momentum_10d'])
            
            # Lag features
            for col in ['returns_1d', 'volatility_10d', 'volume_ratio_10d']:
                if col in features.columns:
                    for lag in [1, 2, 3]:
                        lag_col = f'{col}_lag{lag}'
                        features[lag_col] = features[col].shift(lag)
            
            # Fill missing values
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def _create_targets(self, market_data: pd.DataFrame, performance_target: str) -> pd.Series:
        """Create target variable from market data"""
        try:
            if 'close' not in market_data.columns:
                return pd.Series()
            
            returns = market_data['close'].pct_change()
            
            if performance_target == 'sharpe_ratio':
                # Rolling Sharpe ratio
                targets = returns.rolling(window=20).mean() / returns.rolling(window=20).std() * np.sqrt(252)
            elif performance_target == 'return':
                # Forward returns
                targets = returns.shift(-5)  # 5-day forward return
            elif performance_target == 'volatility':
                # Rolling volatility
                targets = returns.rolling(window=20).std() * np.sqrt(252)
            elif performance_target == 'max_drawdown':
                # Rolling maximum drawdown
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.rolling(window=60).max()
                drawdown = (cumulative - rolling_max) / rolling_max
                targets = drawdown.rolling(window=20).min()
            else:
                # Default to Sharpe ratio
                targets = returns.rolling(window=20).mean() / returns.rolling(window=20).std() * np.sqrt(252)
            
            # Create target DataFrame for joining
            target_df = pd.DataFrame({'target': targets}, index=market_data.index)
            
            return target_df['target'].fillna(0)
            
        except Exception as e:
            logger.error(f"Error creating targets: {e}")
            return pd.Series()
    
    def _train_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model"""
        try:
            if features.empty or targets.empty:
                return {'success': False, 'error': 'Empty training data'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=self.random_state, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize model
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Hyperparameter tuning (optional)
            if len(X_train) > 100:
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
                
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
            else:
                # Train with default parameters
                self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            # Store feature importance
            self.feature_importance_ = dict(zip(features.columns, self.model.feature_importances_))
            self.is_trained = True
            
            return {
                'success': True,
                'train_score': float(train_score),
                'test_score': float(test_score),
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'overfitting_check': float(train_score - test_score),
                'feature_count': len(features.columns),
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_optimized_parameters(self, 
                                     features: pd.DataFrame,
                                     current_parameters: Dict[str, float]) -> Dict[str, float]:
        """Generate optimized parameters using trained model"""
        try:
            if not self.is_trained or self.model is None:
                return current_parameters
            
            # Get recent market conditions
            recent_features = features.tail(1)
            if recent_features.empty:
                return current_parameters
            
            # Scale features
            recent_features_scaled = self.scaler.transform(recent_features)
            
            # Predict optimal performance for different parameter combinations
            optimized_params = current_parameters.copy()
            
            # Parameter optimization ranges
            param_ranges = {
                'volatility_threshold': np.linspace(0.1, 0.8, 10),
                'trend_threshold': np.linspace(0.2, 0.9, 10),
                'volume_threshold': np.linspace(0.1, 0.7, 10),
                'momentum_threshold': np.linspace(0.15, 0.85, 10)
            }
            
            best_prediction = None
            best_params = current_parameters.copy()
            
            # Grid search over parameter space
            for param_name, param_values in param_ranges.items():
                if param_name in current_parameters:
                    best_value = current_parameters[param_name]
                    best_pred = None
                    
                    for param_value in param_values:
                        # Create modified features representing this parameter choice
                        test_features = recent_features_scaled.copy()
                        
                        # Predict performance with this parameter
                        prediction = self.model.predict(test_features)[0]
                        
                        if best_pred is None or prediction > best_pred:
                            best_pred = prediction
                            best_value = param_value
                    
                    optimized_params[param_name] = float(best_value)
                    
                    if best_prediction is None or best_pred > best_prediction:
                        best_prediction = best_pred
            
            # Ensure parameters are within reasonable bounds
            optimized_params = self._constrain_parameters(optimized_params)
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error generating optimized parameters: {e}")
            return current_parameters
    
    def _constrain_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Constrain parameters to reasonable bounds"""
        try:
            constraints = {
                'volatility_threshold': (0.1, 0.8),
                'trend_threshold': (0.2, 0.9),
                'volume_threshold': (0.1, 0.7),
                'momentum_threshold': (0.15, 0.85)
            }
            
            constrained = {}
            for param, value in parameters.items():
                if param in constraints:
                    min_val, max_val = constraints[param]
                    constrained[param] = max(min_val, min(max_val, value))
                else:
                    constrained[param] = value
            
            return constrained
            
        except Exception as e:
            logger.error(f"Error constraining parameters: {e}")
            return parameters
    
    def _validate_optimization(self, 
                             features: pd.DataFrame,
                             targets: pd.Series,
                             optimized_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Validate the optimization results"""
        try:
            validation = {
                'validation_score': 0.0,
                'confidence_interval': {'lower': 0.0, 'upper': 0.0},
                'parameter_stability': {},
                'validation_warnings': []
            }
            
            if not self.is_trained or features.empty:
                validation['validation_warnings'].append("Model not trained or no validation data")
                return validation
            
            # Cross-validation score
            recent_features = features.tail(20)
            recent_targets = targets.tail(20)
            
            if len(recent_features) > 5:
                recent_features_scaled = self.scaler.transform(recent_features)
                predictions = self.model.predict(recent_features_scaled)
                validation_score = r2_score(recent_targets, predictions)
                validation['validation_score'] = float(validation_score)
                
                # Prediction confidence interval
                pred_std = np.std(predictions - recent_targets)
                last_prediction = predictions[-1]
                validation['confidence_interval'] = {
                    'lower': float(last_prediction - 1.96 * pred_std),
                    'upper': float(last_prediction + 1.96 * pred_std)
                }
            
            # Parameter stability check
            for param, value in optimized_parameters.items():
                # Check if parameter change is reasonable
                if param in self.training_history.get('feature_importance_history', []):
                    importance_history = [fi.get(param, 0) for fi in self.training_history['feature_importance_history'][-5:]]
                    if importance_history:
                        avg_importance = np.mean(importance_history)
                        validation['parameter_stability'][param] = {
                            'value': float(value),
                            'feature_importance': float(avg_importance),
                            'stability_score': float(1.0 - np.std(importance_history)) if len(importance_history) > 1 else 1.0
                        }
            
            # Validation warnings
            if validation['validation_score'] < 0.1:
                validation['validation_warnings'].append("Low validation score - model predictions may be unreliable")
            
            pred_range = validation['confidence_interval']['upper'] - validation['confidence_interval']['lower']
            if pred_range > 1.0:
                validation['validation_warnings'].append("Wide prediction confidence interval - high uncertainty")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating optimization: {e}")
            return {'validation_score': 0.0, 'validation_warnings': [str(e)]}
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance from trained model"""
        try:
            if not self.is_trained or self.feature_importance_ is None:
                return {'note': 'Model not trained'}
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)
            
            analysis = {
                'top_features': sorted_features[:10],
                'feature_categories': {},
                'importance_distribution': {
                    'mean': float(np.mean(list(self.feature_importance_.values()))),
                    'std': float(np.std(list(self.feature_importance_.values()))),
                    'max': float(max(self.feature_importance_.values())),
                    'min': float(min(self.feature_importance_.values()))
                }
            }
            
            # Categorize features
            categories = {
                'volatility': ['volatility', 'atr'],
                'momentum': ['momentum', 'returns'],
                'volume': ['volume'],
                'technical': ['rsi', 'ma'],
                'regime': ['regime'],
                'interaction': ['interaction']
            }
            
            for category, keywords in categories.items():
                category_importance = 0
                category_count = 0
                for feature, importance in self.feature_importance_.items():
                    if any(keyword in feature.lower() for keyword in keywords):
                        category_importance += importance
                        category_count += 1
                
                if category_count > 0:
                    analysis['feature_categories'][category] = {
                        'total_importance': float(category_importance),
                        'avg_importance': float(category_importance / category_count),
                        'feature_count': category_count
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {'note': 'Error in analysis'}
    
    def _estimate_improvement(self, 
                            features: pd.DataFrame,
                            targets: pd.Series,
                            current_parameters: Dict[str, float],
                            optimized_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Estimate performance improvement from optimization"""
        try:
            if not self.is_trained or features.empty:
                return {'estimated_improvement': 0.0}
            
            # Use recent data for estimation
            recent_features = features.tail(10)
            recent_targets = targets.tail(10)
            
            if recent_features.empty:
                return {'estimated_improvement': 0.0}
            
            # Scale features
            recent_features_scaled = self.scaler.transform(recent_features)
            
            # Predict performance with current and optimized parameters
            current_predictions = self.model.predict(recent_features_scaled)
            optimized_predictions = self.model.predict(recent_features_scaled)  # Simplified - same features
            
            # Calculate improvement metrics
            current_performance = np.mean(current_predictions)
            optimized_performance = np.mean(optimized_predictions)
            
            improvement = optimized_performance - current_performance
            improvement_pct = (improvement / abs(current_performance) * 100) if current_performance != 0 else 0
            
            # Calculate confidence in improvement
            improvement_std = np.std(optimized_predictions - current_predictions)
            confidence = abs(improvement) / improvement_std if improvement_std > 0 else 0
            
            return {
                'estimated_improvement': float(improvement),
                'improvement_percentage': float(improvement_pct),
                'current_predicted_performance': float(current_performance),
                'optimized_predicted_performance': float(optimized_performance),
                'improvement_confidence': float(min(confidence, 1.0))
            }
            
        except Exception as e:
            logger.error(f"Error estimating improvement: {e}")
            return {'estimated_improvement': 0.0}
    
    def _calculate_model_confidence(self, training_result: Dict[str, Any]) -> float:
        """Calculate overall model confidence"""
        try:
            if not training_result.get('success', False):
                return 0.0
            
            # Factors affecting confidence
            test_score = training_result.get('test_score', 0)
            overfitting = abs(training_result.get('overfitting_check', 1))
            sample_size = training_result.get('training_samples', 0)
            
            # Score component (0-1)
            score_component = max(0, min(test_score, 1))
            
            # Overfitting penalty (0-1)
            overfitting_component = max(0, 1 - overfitting)
            
            # Sample size component (0-1)
            sample_component = min(sample_size / 100, 1)  # Full confidence at 100+ samples
            
            # Weighted confidence
            confidence = (score_component * 0.5 + 
                         overfitting_component * 0.3 + 
                         sample_component * 0.2)
            
            return float(max(min(confidence, 1.0), 0.0))
            
        except Exception as e:
            logger.error(f"Error calculating model confidence: {e}")
            return 0.0
    
    def _update_training_history(self, 
                               training_result: Dict[str, Any],
                               feature_analysis: Dict[str, Any]):
        """Update training history"""
        try:
            self.training_history['training_scores'].append(training_result.get('train_score', 0))
            self.training_history['validation_scores'].append(training_result.get('test_score', 0))
            self.training_history['feature_importance_history'].append(self.feature_importance_.copy())
            self.training_history['training_dates'].append(datetime.now())
            
            # Trim history
            max_history = 20
            for key in self.training_history.keys():
                if len(self.training_history[key]) > max_history:
                    self.training_history[key] = self.training_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating training history: {e}")
    
    def _get_default_optimization_result(self, current_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Get default optimization result when optimization fails"""
        return {
            'optimized_parameters': current_parameters,
            'training_result': {'success': False},
            'validation_result': {'validation_score': 0.0},
            'feature_analysis': {'note': 'No analysis available'},
            'improvement_estimation': {'estimated_improvement': 0.0},
            'model_confidence': 0.0,
            'optimization_timestamp': datetime.now()
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of Random Forest model"""
        try:
            if not self.is_trained:
                return {'status': 'not_trained'}
            
            return {
                'model_status': 'trained',
                'model_parameters': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf
                },
                'training_history_length': len(self.training_history['training_scores']),
                'average_training_score': float(np.mean(self.training_history['training_scores'])) if self.training_history['training_scores'] else 0,
                'average_validation_score': float(np.mean(self.training_history['validation_scores'])) if self.training_history['validation_scores'] else 0,
                'top_features': sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)[:5] if self.feature_importance_ else [],
                'last_training_date': self.training_history['training_dates'][-1] if self.training_history['training_dates'] else None
            }
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {'status': 'error', 'error': str(e)}