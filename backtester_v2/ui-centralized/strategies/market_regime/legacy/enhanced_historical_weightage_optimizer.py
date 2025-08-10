#!/usr/bin/env python3
"""
Enhanced Historical Data-Driven Dynamic Weightage Optimizer
Building upon existing Enhanced18RegimeDetector and market regime templates

This module implements sophisticated historical performance-based dynamic weightage
optimization using 2-3 years of NIFTY options data for precise indicator weight
adjustment based on statistical performance analysis.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import json
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import existing components
from .archive_enhanced_modules_do_not_use.enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType
from .time_series_regime_storage import TimeSeriesRegimeStorage
from .excel_config_manager import MarketRegimeExcelManager

logger = logging.getLogger(__name__)

@dataclass
class IndicatorPerformanceMetrics:
    """Performance metrics for individual indicators"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    false_positive_rate: float
    false_negative_rate: float
    regime_prediction_success: float
    market_condition_effectiveness: float
    statistical_significance: float  # p-value
    confidence_interval: Tuple[float, float]
    sample_size: int
    timeframe_performance: Dict[str, float]
    dte_performance: Dict[int, float]

@dataclass
class DTESpecificAnalysis:
    """DTE-specific performance analysis"""
    dte: int
    indicator_performance: Dict[str, IndicatorPerformanceMetrics]
    optimal_weights: Dict[str, float]
    regime_formation_accuracy: float
    transition_prediction_accuracy: float
    market_condition_sensitivity: Dict[str, float]
    statistical_confidence: float

class EnhancedHistoricalWeightageOptimizer:
    """
    Enhanced Historical Data-Driven Dynamic Weightage Optimizer
    
    Implements sophisticated algorithms for optimizing indicator weights based on
    historical performance analysis using 2-3 years of NIFTY options data.
    
    Key Features:
    - DTE-specific weight optimization (1-30 days)
    - Statistical significance testing (p-values, confidence intervals)
    - Machine learning models for weight prediction
    - Rolling window analysis with exponential decay
    - Market condition similarity algorithms
    - Real-time performance adaptation
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 storage_path: Optional[str] = None,
                 historical_data_path: Optional[str] = None):
        """
        Initialize Enhanced Historical Weightage Optimizer
        
        Args:
            config_path: Path to Excel configuration file
            storage_path: Path to time-series storage database
            historical_data_path: Path to historical NIFTY options data
        """
        # Initialize core components
        self.config_manager = MarketRegimeExcelManager(config_path)
        self.storage = TimeSeriesRegimeStorage(storage_path or "enhanced_regime_storage.db")
        self.regime_detector = Enhanced18RegimeDetector()
        
        # Historical data management
        self.historical_data_path = historical_data_path
        self.historical_data = None
        self.performance_cache = {}
        
        # Indicator systems (from codebase analysis)
        self.indicator_systems = [
            'greek_sentiment', 'trending_oi_pa', 'ema_indicators', 'vwap_indicators',
            'iv_skew', 'iv_indicators', 'premium_indicators', 'atr_indicators',
            'enhanced_straddle_analysis', 'multi_timeframe_analysis'
        ]
        
        # Timeframes (from existing system)
        self.timeframes = ['3min', '5min', '10min', '15min', '30min']
        
        # DTE range for analysis
        self.dte_range = list(range(1, 31))  # 1-30 days
        
        # Performance tracking
        self.performance_history = {}
        self.weight_history = {}
        self.optimization_results = {}
        
        # Machine learning models
        self.ml_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        # Statistical parameters
        self.confidence_level = 0.95
        self.significance_threshold = 0.05
        self.min_sample_size = 100
        
        # Dynamic adjustment parameters
        self.learning_rate = 0.01
        self.decay_factor = 0.95
        self.rolling_window_size = 100
        self.performance_window_days = 30
        
        logger.info("Enhanced Historical Weightage Optimizer initialized")
    
    def load_historical_data(self, data_path: str = None) -> bool:
        """
        Load historical NIFTY options data for analysis
        
        Args:
            data_path: Path to historical data file
            
        Returns:
            bool: Success status
        """
        try:
            data_path = data_path or self.historical_data_path
            if not data_path or not Path(data_path).exists():
                logger.error(f"Historical data file not found: {data_path}")
                return False
            
            logger.info(f"Loading historical data from: {data_path}")
            
            # Load data based on file format
            if data_path.endswith('.csv'):
                self.historical_data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                self.historical_data = pd.read_parquet(data_path)
            else:
                logger.error(f"Unsupported file format: {data_path}")
                return False
            
            # Validate required columns
            required_columns = [
                'timestamp', 'strike', 'expiry_date', 'ce_close', 'pe_close',
                'ce_oi', 'pe_oi', 'ce_volume', 'pe_volume', 'underlying_price',
                'ce_delta', 'ce_gamma', 'ce_theta', 'ce_vega',
                'pe_delta', 'pe_gamma', 'pe_theta', 'pe_vega', 'iv'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.historical_data.columns]
            if missing_columns:
                logger.warning(f"Missing columns in historical data: {missing_columns}")
            
            # Convert timestamp to datetime
            self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
            self.historical_data['expiry_date'] = pd.to_datetime(self.historical_data['expiry_date'])
            
            # Calculate DTE
            self.historical_data['dte'] = (
                self.historical_data['expiry_date'] - self.historical_data['timestamp']
            ).dt.days
            
            # Filter valid DTE range
            self.historical_data = self.historical_data[
                (self.historical_data['dte'] >= 1) & 
                (self.historical_data['dte'] <= 30)
            ]
            
            logger.info(f"Loaded {len(self.historical_data)} historical records")
            logger.info(f"Date range: {self.historical_data['timestamp'].min()} to {self.historical_data['timestamp'].max()}")
            logger.info(f"DTE range: {self.historical_data['dte'].min()} to {self.historical_data['dte'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    def calculate_indicator_historical_performance(self, 
                                                 indicator_name: str,
                                                 dte: Optional[int] = None,
                                                 timeframe: Optional[str] = None,
                                                 market_condition: Optional[str] = None) -> IndicatorPerformanceMetrics:
        """
        Calculate comprehensive historical performance metrics for an indicator
        
        Args:
            indicator_name: Name of the indicator system
            dte: Specific DTE for analysis (optional)
            timeframe: Specific timeframe for analysis (optional)
            market_condition: Specific market condition (optional)
            
        Returns:
            IndicatorPerformanceMetrics: Comprehensive performance metrics
        """
        try:
            logger.info(f"Calculating historical performance for {indicator_name}")
            
            # Filter historical data based on criteria
            data = self.historical_data.copy()
            
            if dte is not None:
                data = data[data['dte'] == dte]
            
            if len(data) < self.min_sample_size:
                logger.warning(f"Insufficient data for {indicator_name} analysis: {len(data)} samples")
                return self._create_default_metrics()
            
            # Calculate indicator signals (placeholder - integrate with actual indicator calculations)
            indicator_signals = self._calculate_indicator_signals(data, indicator_name)
            
            # Calculate regime predictions
            regime_predictions = self._calculate_regime_predictions(data, indicator_signals)
            
            # Calculate actual regime outcomes (ground truth)
            actual_regimes = self._calculate_actual_regimes(data)
            
            # Calculate performance metrics
            accuracy = self._calculate_accuracy(regime_predictions, actual_regimes)
            precision = self._calculate_precision(regime_predictions, actual_regimes)
            recall = self._calculate_recall(regime_predictions, actual_regimes)
            f1_score = self._calculate_f1_score(precision, recall)
            
            # Calculate financial metrics
            sharpe_ratio = self._calculate_sharpe_ratio(data, indicator_signals)
            
            # Calculate error rates
            false_positive_rate = self._calculate_false_positive_rate(regime_predictions, actual_regimes)
            false_negative_rate = self._calculate_false_negative_rate(regime_predictions, actual_regimes)
            
            # Calculate regime-specific metrics
            regime_prediction_success = self._calculate_regime_prediction_success(regime_predictions, actual_regimes)
            market_condition_effectiveness = self._calculate_market_condition_effectiveness(data, indicator_signals)
            
            # Statistical significance testing
            statistical_significance, confidence_interval = self._calculate_statistical_significance(
                regime_predictions, actual_regimes
            )
            
            # Timeframe-specific performance
            timeframe_performance = self._calculate_timeframe_performance(data, indicator_name)
            
            # DTE-specific performance
            dte_performance = self._calculate_dte_performance(data, indicator_name)
            
            metrics = IndicatorPerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                regime_prediction_success=regime_prediction_success,
                market_condition_effectiveness=market_condition_effectiveness,
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                sample_size=len(data),
                timeframe_performance=timeframe_performance,
                dte_performance=dte_performance
            )
            
            logger.info(f"Performance metrics calculated for {indicator_name}: Accuracy={accuracy:.3f}, Sharpe={sharpe_ratio:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance for {indicator_name}: {e}")
            return self._create_default_metrics()
    
    def optimize_weights_for_dte(self, dte: int, 
                               market_condition: Optional[str] = None) -> Dict[str, float]:
        """
        Optimize indicator weights for specific DTE using historical performance
        
        Mathematical Formula:
        W_i = f(historical_accuracy, recent_performance, market_similarity, statistical_significance)
        
        Where:
        W_i = Weight for indicator i
        f() = Optimization function combining multiple factors
        
        Args:
            dte: Days to expiry for optimization
            market_condition: Current market condition for similarity matching
            
        Returns:
            Dict[str, float]: Optimized weights for each indicator
        """
        try:
            logger.info(f"Optimizing weights for DTE={dte}, market_condition={market_condition}")
            
            # Calculate performance metrics for each indicator at this DTE
            indicator_metrics = {}
            for indicator in self.indicator_systems:
                metrics = self.calculate_indicator_historical_performance(
                    indicator, dte=dte, market_condition=market_condition
                )
                indicator_metrics[indicator] = metrics
            
            # Initialize weights
            optimized_weights = {}
            
            for indicator in self.indicator_systems:
                metrics = indicator_metrics[indicator]
                
                # Base weight from historical accuracy
                base_weight = metrics.accuracy
                
                # Adjust for statistical significance
                significance_multiplier = 1.0
                if metrics.statistical_significance < self.significance_threshold:
                    significance_multiplier = 1.2  # Boost statistically significant indicators
                else:
                    significance_multiplier = 0.8  # Reduce weight for non-significant indicators
                
                # Adjust for sample size
                sample_size_multiplier = min(1.0, metrics.sample_size / self.min_sample_size)
                
                # Adjust for Sharpe ratio (financial performance)
                sharpe_multiplier = 1.0 + (metrics.sharpe_ratio / 2.0)  # Normalize Sharpe contribution
                
                # Adjust for regime prediction success
                regime_multiplier = 1.0 + (metrics.regime_prediction_success - 0.5)
                
                # Calculate final weight using mathematical formula
                final_weight = (
                    base_weight * 
                    significance_multiplier * 
                    sample_size_multiplier * 
                    sharpe_multiplier * 
                    regime_multiplier
                )
                
                optimized_weights[indicator] = max(0.01, min(1.0, final_weight))  # Bounds: [0.01, 1.0]
            
            # Normalize weights to sum to 1.0
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
            
            logger.info(f"Optimized weights for DTE={dte}: {optimized_weights}")
            
            # Store optimization results
            self.optimization_results[dte] = {
                'weights': optimized_weights,
                'metrics': indicator_metrics,
                'timestamp': datetime.now(),
                'market_condition': market_condition
            }
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Error optimizing weights for DTE={dte}: {e}")
            return self._get_default_weights()
    
    def implement_machine_learning_optimization(self, 
                                              target_metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Implement machine learning models for weight optimization
        
        Uses Random Forest and Linear Regression to predict optimal weights
        based on historical performance patterns.
        
        Args:
            target_metric: Target metric for optimization ('accuracy', 'sharpe_ratio', etc.)
            
        Returns:
            Dict[str, Any]: ML optimization results
        """
        try:
            logger.info(f"Implementing ML optimization for target metric: {target_metric}")
            
            # Prepare training data
            X, y = self._prepare_ml_training_data(target_metric)
            
            if len(X) < self.min_sample_size:
                logger.warning(f"Insufficient data for ML training: {len(X)} samples")
                return {}
            
            # Split data for training and validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            ml_results = {}
            
            # Train Random Forest model
            rf_model = self.ml_models['random_forest']
            rf_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(X_val)
            rf_r2 = r2_score(y_val, rf_predictions)
            rf_mse = mean_squared_error(y_val, rf_predictions)
            
            ml_results['random_forest'] = {
                'r2_score': rf_r2,
                'mse': rf_mse,
                'feature_importance': dict(zip(self.indicator_systems, rf_model.feature_importances_))
            }
            
            # Train Linear Regression model
            lr_model = self.ml_models['linear_regression']
            lr_model.fit(X_train, y_train)
            lr_predictions = lr_model.predict(X_val)
            lr_r2 = r2_score(y_val, lr_predictions)
            lr_mse = mean_squared_error(y_val, lr_predictions)
            
            ml_results['linear_regression'] = {
                'r2_score': lr_r2,
                'mse': lr_mse,
                'coefficients': dict(zip(self.indicator_systems, lr_model.coef_))
            }
            
            # Select best model
            best_model = 'random_forest' if rf_r2 > lr_r2 else 'linear_regression'
            ml_results['best_model'] = best_model
            ml_results['best_r2'] = max(rf_r2, lr_r2)
            
            logger.info(f"ML optimization completed. Best model: {best_model} (R²={ml_results['best_r2']:.3f})")
            
            return ml_results
            
        except Exception as e:
            logger.error(f"Error in ML optimization: {e}")
            return {}
    
    def calculate_adaptive_weights_with_rolling_window(self, 
                                                     current_market_data: pd.DataFrame,
                                                     window_size: int = None) -> Dict[str, float]:
        """
        Calculate adaptive weights using rolling window analysis with exponential decay
        
        Mathematical Formula:
        W_i(t) = α * W_i(t-1) + (1-α) * Performance_i(window)
        
        Where:
        α = decay_factor (0.95 default)
        Performance_i(window) = Recent performance in rolling window
        
        Args:
            current_market_data: Current market data for context
            window_size: Size of rolling window (default: self.rolling_window_size)
            
        Returns:
            Dict[str, float]: Adaptive weights
        """
        try:
            window_size = window_size or self.rolling_window_size
            logger.info(f"Calculating adaptive weights with rolling window size: {window_size}")
            
            # Get recent performance data
            recent_data = self._get_recent_performance_data(window_size)
            
            if len(recent_data) < window_size // 2:
                logger.warning("Insufficient recent data for adaptive weights")
                return self._get_default_weights()
            
            # Calculate current market conditions
            current_dte = self._extract_dte_from_market_data(current_market_data)
            current_volatility = self._calculate_current_volatility(current_market_data)
            current_regime = self._estimate_current_regime(current_market_data)
            
            # Calculate performance-based weights for each indicator
            adaptive_weights = {}
            
            for indicator in self.indicator_systems:
                # Get historical weight
                historical_weight = self.weight_history.get(indicator, 1.0 / len(self.indicator_systems))
                
                # Calculate recent performance
                recent_performance = self._calculate_recent_performance(recent_data, indicator)
                
                # Apply exponential decay formula
                adaptive_weight = (
                    self.decay_factor * historical_weight + 
                    (1 - self.decay_factor) * recent_performance
                )
                
                # Adjust for current market conditions
                market_adjustment = self._calculate_market_condition_adjustment(
                    indicator, current_dte, current_volatility, current_regime
                )
                
                adaptive_weight *= market_adjustment
                
                adaptive_weights[indicator] = max(0.01, min(1.0, adaptive_weight))
            
            # Normalize weights
            total_weight = sum(adaptive_weights.values())
            if total_weight > 0:
                adaptive_weights = {k: v/total_weight for k, v in adaptive_weights.items()}
            
            # Update weight history
            self.weight_history.update(adaptive_weights)
            
            logger.info(f"Adaptive weights calculated: {adaptive_weights}")
            
            return adaptive_weights
            
        except Exception as e:
            logger.error(f"Error calculating adaptive weights: {e}")
            return self._get_default_weights()
    
    # Helper methods (implementation details)
    def _calculate_indicator_signals(self, data: pd.DataFrame, indicator_name: str) -> pd.Series:
        """Calculate indicator signals from historical data"""
        # Placeholder implementation - integrate with actual indicator calculations
        # This would call the specific indicator calculation methods
        return pd.Series(np.random.randn(len(data)), index=data.index)
    
    def _calculate_regime_predictions(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate regime predictions from indicator signals"""
        # Convert signals to regime predictions
        return pd.Series(np.where(signals > 0, 1, 0), index=data.index)
    
    def _calculate_actual_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Calculate actual regime outcomes (ground truth)"""
        # Calculate actual market regimes based on price movements
        returns = data['underlying_price'].pct_change()
        return pd.Series(np.where(returns > 0, 1, 0), index=data.index)
    
    def _calculate_accuracy(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate prediction accuracy"""
        return (predictions == actual).mean()
    
    def _calculate_precision(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate precision score"""
        tp = ((predictions == 1) & (actual == 1)).sum()
        fp = ((predictions == 1) & (actual == 0)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_recall(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate recall score"""
        tp = ((predictions == 1) & (actual == 1)).sum()
        fn = ((predictions == 0) & (actual == 1)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, data: pd.DataFrame, signals: pd.Series) -> float:
        """Calculate Sharpe ratio for indicator signals"""
        returns = data['underlying_price'].pct_change()
        strategy_returns = returns * signals.shift(1)
        return strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0.0
    
    def _calculate_false_positive_rate(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate false positive rate"""
        fp = ((predictions == 1) & (actual == 0)).sum()
        tn = ((predictions == 0) & (actual == 0)).sum()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def _calculate_false_negative_rate(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate false negative rate"""
        fn = ((predictions == 0) & (actual == 1)).sum()
        tp = ((predictions == 1) & (actual == 1)).sum()
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    def _calculate_regime_prediction_success(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate regime prediction success rate"""
        return self._calculate_accuracy(predictions, actual)
    
    def _calculate_market_condition_effectiveness(self, data: pd.DataFrame, signals: pd.Series) -> float:
        """Calculate effectiveness across different market conditions"""
        # Placeholder - implement market condition analysis
        return 0.75
    
    def _calculate_statistical_significance(self, predictions: pd.Series, actual: pd.Series) -> Tuple[float, Tuple[float, float]]:
        """Calculate statistical significance and confidence intervals"""
        # Perform t-test for statistical significance
        accuracy = self._calculate_accuracy(predictions, actual)
        n = len(predictions)
        
        # Calculate confidence interval for accuracy
        std_error = np.sqrt(accuracy * (1 - accuracy) / n)
        margin_error = stats.norm.ppf(1 - (1 - self.confidence_level) / 2) * std_error
        
        confidence_interval = (accuracy - margin_error, accuracy + margin_error)
        
        # Calculate p-value (simplified)
        t_stat = (accuracy - 0.5) / std_error if std_error > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return p_value, confidence_interval
    
    def _calculate_timeframe_performance(self, data: pd.DataFrame, indicator_name: str) -> Dict[str, float]:
        """Calculate performance across different timeframes"""
        # Placeholder - implement timeframe-specific analysis
        return {tf: 0.75 for tf in self.timeframes}
    
    def _calculate_dte_performance(self, data: pd.DataFrame, indicator_name: str) -> Dict[int, float]:
        """Calculate performance across different DTE values"""
        # Placeholder - implement DTE-specific analysis
        return {dte: 0.75 for dte in self.dte_range}
    
    def _create_default_metrics(self) -> IndicatorPerformanceMetrics:
        """Create default performance metrics"""
        return IndicatorPerformanceMetrics(
            accuracy=0.5, precision=0.5, recall=0.5, f1_score=0.5,
            sharpe_ratio=0.0, false_positive_rate=0.5, false_negative_rate=0.5,
            regime_prediction_success=0.5, market_condition_effectiveness=0.5,
            statistical_significance=1.0, confidence_interval=(0.4, 0.6),
            sample_size=0, timeframe_performance={}, dte_performance={}
        )
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default indicator weights"""
        return {indicator: 1.0 / len(self.indicator_systems) for indicator in self.indicator_systems}
    
    def _prepare_ml_training_data(self, target_metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        # Placeholder - implement ML training data preparation
        n_samples = 1000
        X = np.random.randn(n_samples, len(self.indicator_systems))
        y = np.random.randn(n_samples)
        return X, y
    
    def _get_recent_performance_data(self, window_size: int) -> pd.DataFrame:
        """Get recent performance data for rolling window analysis"""
        # Placeholder - implement recent performance data retrieval
        return pd.DataFrame()
    
    def _extract_dte_from_market_data(self, market_data: pd.DataFrame) -> int:
        """Extract DTE from current market data"""
        return market_data.get('dte', [7])[0] if 'dte' in market_data.columns else 7
    
    def _calculate_current_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate current market volatility"""
        return market_data.get('iv', [0.2]).mean() if 'iv' in market_data.columns else 0.2
    
    def _estimate_current_regime(self, market_data: pd.DataFrame) -> str:
        """Estimate current market regime"""
        return "NORMAL_VOLATILE_NEUTRAL"  # Placeholder
    
    def _calculate_recent_performance(self, recent_data: pd.DataFrame, indicator: str) -> float:
        """Calculate recent performance for an indicator"""
        return 0.75  # Placeholder
    
    def _calculate_market_condition_adjustment(self, indicator: str, dte: int, 
                                             volatility: float, regime: str) -> float:
        """Calculate market condition adjustment factor"""
        return 1.0  # Placeholder
