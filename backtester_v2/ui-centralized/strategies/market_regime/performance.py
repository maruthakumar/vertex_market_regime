"""
Market Regime Performance Tracking

This module handles performance tracking and adaptive weight optimization
for market regime indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
import logging

from .models import RegimeConfig, PerformanceMetrics

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Track performance of regime predictions and optimize weights
    
    This class monitors the accuracy of regime predictions and adjusts
    indicator weights based on historical performance.
    """
    
    def __init__(self, config: RegimeConfig):
        """
        Initialize the performance tracker
        
        Args:
            config (RegimeConfig): Configuration for performance tracking
        """
        self.config = config
        self.performance_history = {}
        self.weight_history = {}
        self.trade_results = []
        
        logger.info("PerformanceTracker initialized")
    
    def update_performance(self, regime_results: pd.DataFrame, market_data: pd.DataFrame):
        """
        Update performance metrics based on regime predictions vs market outcomes
        
        Args:
            regime_results (pd.DataFrame): Regime classification results
            market_data (pd.DataFrame): Actual market data for validation
        """
        try:
            if regime_results.empty or market_data.empty:
                return
            
            # Calculate actual market regime based on price movements
            actual_regimes = self._calculate_actual_regimes(market_data)
            
            # Align predictions with actuals
            aligned_data = self._align_predictions_with_actuals(regime_results, actual_regimes)
            
            if aligned_data.empty:
                return
            
            # Update indicator performance
            self._update_indicator_performance(aligned_data)
            
            # Update weights if adaptive learning is enabled
            if self.config.learning_rate > 0:
                self._update_adaptive_weights()
            
            logger.info(f"Updated performance metrics for {len(aligned_data)} data points")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _calculate_actual_regimes(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate actual market regimes based on price movements"""
        try:
            actual_regimes = pd.DataFrame(index=market_data.index)
            
            # Calculate returns
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change()
                
                # Calculate rolling statistics
                rolling_return = returns.rolling(window=20).mean()
                rolling_volatility = returns.rolling(window=20).std()
                
                # Classify actual regimes based on returns and volatility
                conditions = [
                    (rolling_return > 0.002) & (rolling_volatility < 0.02),  # Strong Bullish
                    (rolling_return > 0.001) & (rolling_volatility < 0.025), # Moderate Bullish
                    (rolling_return > 0.0005),                               # Weak Bullish
                    (rolling_return >= -0.0005) & (rolling_return <= 0.0005), # Neutral
                    (rolling_return > -0.001),                               # Weak Bearish
                    (rolling_return > -0.002),                               # Moderate Bearish
                ]
                
                choices = [
                    'STRONG_BULLISH',
                    'MODERATE_BULLISH', 
                    'WEAK_BULLISH',
                    'NEUTRAL',
                    'WEAK_BEARISH',
                    'MODERATE_BEARISH'
                ]
                
                actual_regimes['actual_regime'] = np.select(conditions, choices, default='STRONG_BEARISH')
                
                # Add volatility-based regimes
                high_vol_threshold = rolling_volatility.quantile(0.8)
                low_vol_threshold = rolling_volatility.quantile(0.2)
                
                actual_regimes.loc[rolling_volatility > high_vol_threshold, 'actual_regime'] = 'HIGH_VOLATILITY'
                actual_regimes.loc[rolling_volatility < low_vol_threshold, 'actual_regime'] = 'LOW_VOLATILITY'
                
                # Add market metrics
                actual_regimes['actual_return'] = rolling_return
                actual_regimes['actual_volatility'] = rolling_volatility
                actual_regimes['price_change'] = returns
            
            return actual_regimes
            
        except Exception as e:
            logger.error(f"Error calculating actual regimes: {e}")
            return pd.DataFrame()
    
    def _align_predictions_with_actuals(self, predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
        """Align prediction and actual data"""
        try:
            # Merge on timestamp
            aligned = predictions.join(actuals, how='inner')
            
            # Remove rows with missing data
            aligned = aligned.dropna(subset=['regime_type', 'actual_regime'])
            
            return aligned
            
        except Exception as e:
            logger.error(f"Error aligning predictions with actuals: {e}")
            return pd.DataFrame()
    
    def _update_indicator_performance(self, aligned_data: pd.DataFrame):
        """Update performance metrics for each indicator"""
        try:
            current_date = date.today()
            
            # Get indicator columns
            indicator_columns = [col for col in aligned_data.columns if col.endswith('_signal')]
            
            for col in indicator_columns:
                indicator_id = col.replace('_signal', '')
                
                # Calculate hit rate
                hit_rate = self._calculate_hit_rate(aligned_data, col)
                
                # Calculate Sharpe ratio
                sharpe_ratio = self._calculate_sharpe_ratio(aligned_data, col)
                
                # Calculate information ratio
                information_ratio = self._calculate_information_ratio(aligned_data, col)
                
                # Get current weight
                current_weight = aligned_data.get(f'{indicator_id}_weight', pd.Series([0.1] * len(aligned_data))).iloc[0]
                
                # Calculate performance score
                performance_score = self._calculate_performance_score(hit_rate, sharpe_ratio, information_ratio)
                
                # Create performance metrics
                metrics = PerformanceMetrics(
                    indicator_id=indicator_id,
                    evaluation_date=current_date,
                    hit_rate=hit_rate,
                    sharpe_ratio=sharpe_ratio,
                    information_ratio=information_ratio,
                    current_weight=current_weight,
                    performance_score=performance_score,
                    trade_count=len(aligned_data),
                    win_rate=hit_rate,  # Same as hit rate for regime prediction
                    avg_return=aligned_data['actual_return'].mean() if 'actual_return' in aligned_data.columns else 0.0,
                    max_drawdown=0.0  # TODO: Calculate actual drawdown
                )
                
                # Store in history
                if indicator_id not in self.performance_history:
                    self.performance_history[indicator_id] = []
                
                self.performance_history[indicator_id].append(metrics)
                
                # Keep only recent history
                max_history = self.config.performance_window
                if len(self.performance_history[indicator_id]) > max_history:
                    self.performance_history[indicator_id] = self.performance_history[indicator_id][-max_history:]
            
        except Exception as e:
            logger.error(f"Error updating indicator performance: {e}")
    
    def _calculate_hit_rate(self, data: pd.DataFrame, signal_col: str) -> float:
        """Calculate hit rate for an indicator"""
        try:
            if signal_col not in data.columns or 'actual_regime' not in data.columns:
                return 0.5
            
            # Convert regime predictions to numeric
            predicted_numeric = self._regime_to_numeric(data['regime_type'])
            actual_numeric = self._regime_to_numeric(data['actual_regime'])
            
            # Calculate directional accuracy
            predicted_direction = np.sign(predicted_numeric)
            actual_direction = np.sign(actual_numeric)
            
            correct_predictions = (predicted_direction == actual_direction).sum()
            total_predictions = len(data)
            
            return correct_predictions / total_predictions if total_predictions > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating hit rate: {e}")
            return 0.5
    
    def _calculate_sharpe_ratio(self, data: pd.DataFrame, signal_col: str) -> float:
        """Calculate Sharpe ratio for an indicator"""
        try:
            if 'actual_return' not in data.columns:
                return 0.0
            
            # Get indicator signals
            signals = data[signal_col] if signal_col in data.columns else pd.Series([0] * len(data))
            returns = data['actual_return']
            
            # Calculate strategy returns (signal * actual return)
            strategy_returns = signals.shift(1) * returns  # Use previous signal
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return 0.0
            
            # Calculate Sharpe ratio
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()
            
            # Annualize (assuming minute data)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252 * 24 * 60) if std_return > 0 else 0.0
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, data: pd.DataFrame, signal_col: str) -> float:
        """Calculate information ratio for an indicator"""
        try:
            if 'actual_return' not in data.columns:
                return 0.0
            
            # Get indicator signals and benchmark (market returns)
            signals = data[signal_col] if signal_col in data.columns else pd.Series([0] * len(data))
            returns = data['actual_return']
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            strategy_returns = strategy_returns.dropna()
            
            # Calculate excess returns (vs zero benchmark)
            excess_returns = strategy_returns
            
            if len(excess_returns) == 0 or excess_returns.std() == 0:
                return 0.0
            
            # Calculate information ratio
            mean_excess = excess_returns.mean()
            std_excess = excess_returns.std()
            
            information_ratio = mean_excess / std_excess if std_excess > 0 else 0.0
            
            return information_ratio
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    def _calculate_performance_score(self, hit_rate: float, sharpe_ratio: float, information_ratio: float) -> float:
        """Calculate overall performance score"""
        try:
            # Weighted combination of metrics
            score = (
                0.4 * hit_rate +
                0.3 * min(max(sharpe_ratio / 2.0, -1.0), 1.0) +  # Normalize Sharpe to [-1, 1]
                0.3 * min(max(information_ratio / 2.0, -1.0), 1.0)  # Normalize IR to [-1, 1]
            )
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    def _regime_to_numeric(self, regime_series: pd.Series) -> pd.Series:
        """Convert regime labels to numeric values"""
        regime_map = {
            'STRONG_BULLISH': 2.0,
            'MODERATE_BULLISH': 1.0,
            'WEAK_BULLISH': 0.5,
            'NEUTRAL': 0.0,
            'SIDEWAYS': 0.0,
            'WEAK_BEARISH': -0.5,
            'MODERATE_BEARISH': -1.0,
            'STRONG_BEARISH': -2.0,
            'HIGH_VOLATILITY': 0.0,
            'LOW_VOLATILITY': 0.0,
            'TRANSITION': 0.0
        }
        
        return regime_series.map(regime_map).fillna(0.0)
    
    def _update_adaptive_weights(self):
        """Update indicator weights based on performance"""
        try:
            if not self.performance_history:
                return
            
            # Calculate new weights based on recent performance
            new_weights = {}
            total_performance = 0.0
            
            for indicator_id, metrics_list in self.performance_history.items():
                if not metrics_list:
                    continue
                
                # Use recent performance (last few evaluations)
                recent_metrics = metrics_list[-5:]  # Last 5 evaluations
                avg_performance = np.mean([m.performance_score for m in recent_metrics])
                
                new_weights[indicator_id] = avg_performance
                total_performance += avg_performance
            
            # Normalize weights
            if total_performance > 0:
                for indicator_id in new_weights:
                    new_weights[indicator_id] /= total_performance
                
                # Apply learning rate
                for indicator_id, new_weight in new_weights.items():
                    if indicator_id in self.weight_history:
                        current_weight = self.weight_history[indicator_id][-1] if self.weight_history[indicator_id] else 0.1
                    else:
                        current_weight = 0.1
                    
                    # Gradual weight adjustment
                    adjusted_weight = current_weight + self.config.learning_rate * (new_weight - current_weight)
                    
                    # Store weight history
                    if indicator_id not in self.weight_history:
                        self.weight_history[indicator_id] = []
                    
                    self.weight_history[indicator_id].append(adjusted_weight)
                    
                    # Keep only recent history
                    if len(self.weight_history[indicator_id]) > 100:
                        self.weight_history[indicator_id] = self.weight_history[indicator_id][-100:]
            
        except Exception as e:
            logger.error(f"Error updating adaptive weights: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all indicators"""
        try:
            summary = {}
            
            for indicator_id, metrics_list in self.performance_history.items():
                if not metrics_list:
                    continue
                
                latest_metrics = metrics_list[-1]
                
                summary[indicator_id] = {
                    'hit_rate': latest_metrics.hit_rate,
                    'sharpe_ratio': latest_metrics.sharpe_ratio,
                    'information_ratio': latest_metrics.information_ratio,
                    'performance_score': latest_metrics.performance_score,
                    'current_weight': latest_metrics.current_weight,
                    'evaluation_count': len(metrics_list),
                    'last_evaluation': latest_metrics.evaluation_date.isoformat()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current adaptive weights"""
        try:
            current_weights = {}
            
            for indicator_id, weight_history in self.weight_history.items():
                if weight_history:
                    current_weights[indicator_id] = weight_history[-1]
            
            return current_weights
            
        except Exception as e:
            logger.error(f"Error getting current weights: {e}")
            return {}
