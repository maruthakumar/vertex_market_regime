"""
Performance Evaluator - Advanced Performance Evaluation System
===========================================================

Evaluates and tracks performance of market regime detection systems.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import base utilities
from ...base.common_utils import MathUtils, TimeUtils, ErrorHandler

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """Advanced performance evaluation for market regime systems"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Performance Evaluator"""
        self.evaluation_window = config.get('evaluation_window', 252)  # 1 year
        self.benchmark_metrics = config.get('benchmark_metrics', ['sharpe_ratio', 'max_drawdown', 'win_rate'])
        self.risk_free_rate = config.get('risk_free_rate', 0.05)  # 5% annual
        
        # Performance tracking
        self.performance_history = {
            'daily_returns': [],
            'regime_predictions': [],
            'actual_regimes': [],
            'timestamps': [],
            'portfolio_values': []
        }
        
        # Benchmark comparisons
        self.benchmark_data = {
            'market_returns': [],
            'regime_benchmark': [],
            'strategy_returns': []
        }
        
        # Mathematical utilities
        self.math_utils = MathUtils()
        self.time_utils = TimeUtils()
        
        logger.info("PerformanceEvaluator initialized with comprehensive evaluation metrics")
    
    def evaluate_performance(self, 
                           strategy_returns: pd.Series,
                           regime_predictions: pd.Series,
                           actual_regimes: Optional[pd.Series] = None,
                           benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive performance evaluation
        
        Args:
            strategy_returns: Strategy returns time series
            regime_predictions: Predicted market regimes
            actual_regimes: Actual market regimes (if available)
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dict with comprehensive performance analysis
        """
        try:
            if strategy_returns.empty:
                return self._get_default_performance_evaluation()
            
            # Basic performance metrics
            basic_metrics = self._calculate_basic_metrics(strategy_returns)
            
            # Risk-adjusted metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(strategy_returns, benchmark_returns)
            
            # Regime prediction accuracy (if actual regimes available)
            regime_accuracy = self._evaluate_regime_accuracy(regime_predictions, actual_regimes)
            
            # Portfolio analytics
            portfolio_analytics = self._calculate_portfolio_analytics(strategy_returns)
            
            # Benchmark comparison
            benchmark_comparison = self._compare_to_benchmark(strategy_returns, benchmark_returns)
            
            # Rolling performance analysis
            rolling_analysis = self._analyze_rolling_performance(strategy_returns)
            
            # Regime-specific performance
            regime_performance = self._analyze_regime_performance(strategy_returns, regime_predictions)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(strategy_returns)
            
            # Performance attribution
            attribution_analysis = self._calculate_performance_attribution(strategy_returns, regime_predictions)
            
            # Update performance history
            self._update_performance_history(strategy_returns, regime_predictions, actual_regimes)
            
            return {
                'evaluation_timestamp': datetime.now(),
                'basic_metrics': basic_metrics,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'regime_accuracy': regime_accuracy,
                'portfolio_analytics': portfolio_analytics,
                'benchmark_comparison': benchmark_comparison,
                'rolling_analysis': rolling_analysis,
                'regime_performance': regime_performance,
                'risk_metrics': risk_metrics,
                'attribution_analysis': attribution_analysis,
                'overall_score': self._calculate_overall_performance_score(basic_metrics, risk_adjusted_metrics, regime_accuracy)
            }
            
        except Exception as e:
            logger.error(f"Error in performance evaluation: {e}")
            return self._get_default_performance_evaluation()
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        try:
            if returns.empty:
                return {}
            
            # Total return
            total_return = (1 + returns).prod() - 1
            
            # Annualized return
            periods = len(returns)
            years = periods / 252  # Assuming daily data
            annualized_return = ((1 + total_return) ** (1/years)) - 1 if years > 0 else 0
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Win rate
            win_rate = (returns > 0).sum() / len(returns)
            
            # Average win/loss
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            
            avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
            
            # Profit factor
            total_gains = winning_returns.sum() if len(winning_returns) > 0 else 0
            total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 1
            profit_factor = self.math_utils.safe_divide(total_gains, total_losses, 0)
            
            return {
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'volatility': float(volatility),
                'win_rate': float(win_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor),
                'best_day': float(returns.max()),
                'worst_day': float(returns.min())
            }
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_risk_adjusted_metrics(self, 
                                       returns: pd.Series, 
                                       benchmark_returns: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        try:
            if returns.empty:
                return {}
            
            # Sharpe ratio
            excess_returns = returns - (self.risk_free_rate / 252)
            sharpe_ratio = self.math_utils.safe_divide(
                excess_returns.mean() * 252, 
                returns.std() * np.sqrt(252), 
                0
            )
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
            sortino_ratio = self.math_utils.safe_divide(
                excess_returns.mean() * 252,
                downside_volatility,
                0
            )
            
            # Calmar ratio
            max_drawdown = self._calculate_max_drawdown(returns)
            calmar_ratio = self.math_utils.safe_divide(
                returns.mean() * 252,
                abs(max_drawdown),
                0
            )
            
            # Information ratio (vs benchmark)
            information_ratio = 0.0
            if benchmark_returns is not None and not benchmark_returns.empty:
                aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
                tracking_error = (returns - aligned_benchmark).std() * np.sqrt(252)
                information_ratio = self.math_utils.safe_divide(
                    (returns - aligned_benchmark).mean() * 252,
                    tracking_error,
                    0
                )
            
            # Beta (vs benchmark)
            beta = 1.0
            if benchmark_returns is not None and not benchmark_returns.empty:
                aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
                if aligned_benchmark.var() != 0:
                    beta = np.cov(returns, aligned_benchmark)[0, 1] / aligned_benchmark.var()
            
            # Alpha (vs benchmark)
            alpha = 0.0
            if benchmark_returns is not None and not benchmark_returns.empty:
                aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
                alpha = returns.mean() * 252 - (self.risk_free_rate + beta * (aligned_benchmark.mean() * 252 - self.risk_free_rate))
            
            return {
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'information_ratio': float(information_ratio),
                'beta': float(beta),
                'alpha': float(alpha)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return float(drawdown.min())
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _evaluate_regime_accuracy(self, 
                                regime_predictions: pd.Series,
                                actual_regimes: Optional[pd.Series]) -> Dict[str, float]:
        """Evaluate regime prediction accuracy"""
        try:
            if actual_regimes is None or actual_regimes.empty:
                return {'regime_accuracy': 0.0, 'note': 'No actual regime data available'}
            
            # Align predictions and actual regimes
            aligned_actual = actual_regimes.reindex(regime_predictions.index).fillna(0)
            
            # Calculate accuracy metrics
            accuracy = accuracy_score(aligned_actual, regime_predictions)
            
            # Calculate precision, recall, F1 for each regime
            unique_regimes = np.unique(np.concatenate([regime_predictions.unique(), aligned_actual.unique()]))
            
            precision_scores = {}
            recall_scores = {}
            f1_scores = {}
            
            for regime in unique_regimes:
                try:
                    # Convert to binary classification for each regime
                    y_true_binary = (aligned_actual == regime).astype(int)
                    y_pred_binary = (regime_predictions == regime).astype(int)
                    
                    if y_true_binary.sum() > 0:  # Only if regime exists in actual data
                        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                        
                        precision_scores[f'regime_{regime}'] = float(precision)
                        recall_scores[f'regime_{regime}'] = float(recall)
                        f1_scores[f'regime_{regime}'] = float(f1)
                except:
                    continue
            
            # Regime transition accuracy
            transition_accuracy = self._calculate_transition_accuracy(regime_predictions, aligned_actual)
            
            return {
                'overall_accuracy': float(accuracy),
                'precision_scores': precision_scores,
                'recall_scores': recall_scores,
                'f1_scores': f1_scores,
                'transition_accuracy': float(transition_accuracy),
                'regime_consistency': self._calculate_regime_consistency(regime_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating regime accuracy: {e}")
            return {'regime_accuracy': 0.0}
    
    def _calculate_transition_accuracy(self, 
                                     predictions: pd.Series, 
                                     actual: pd.Series) -> float:
        """Calculate accuracy of regime transitions"""
        try:
            pred_transitions = predictions != predictions.shift(1)
            actual_transitions = actual != actual.shift(1)
            
            # Accuracy of transition timing (within 1-2 periods)
            transition_matches = 0
            total_transitions = actual_transitions.sum()
            
            if total_transitions == 0:
                return 1.0
            
            for i in range(1, len(actual_transitions)):
                if actual_transitions.iloc[i]:  # Actual transition occurred
                    # Check if prediction had transition within ±2 periods
                    window_start = max(0, i-2)
                    window_end = min(len(pred_transitions), i+3)
                    if pred_transitions.iloc[window_start:window_end].any():
                        transition_matches += 1
            
            return transition_matches / total_transitions
            
        except Exception as e:
            logger.error(f"Error calculating transition accuracy: {e}")
            return 0.0
    
    def _calculate_regime_consistency(self, regime_predictions: pd.Series) -> float:
        """Calculate regime prediction consistency"""
        try:
            if len(regime_predictions) == 0:
                return 0.0
            
            # Calculate frequency of regime changes
            regime_changes = (regime_predictions != regime_predictions.shift(1)).sum()
            change_frequency = regime_changes / len(regime_predictions)
            
            # Lower change frequency indicates higher consistency
            consistency_score = max(0, 1 - change_frequency * 2)  # Penalize high change frequency
            
            return float(consistency_score)
            
        except Exception as e:
            logger.error(f"Error calculating regime consistency: {e}")
            return 0.0
    
    def _calculate_portfolio_analytics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate detailed portfolio analytics"""
        try:
            if returns.empty:
                return {}
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            
            max_drawdown = float(drawdowns.min())
            avg_drawdown = float(drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0
            
            # Drawdown duration
            drawdown_periods = []
            in_drawdown = False
            drawdown_start = None
            
            for i, dd in enumerate(drawdowns):
                if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= -0.001 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    if drawdown_start is not None:
                        drawdown_periods.append(i - drawdown_start)
            
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            
            # Return distribution analysis
            return_stats = {
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'var_95': float(np.percentile(returns, 5)),
                'var_99': float(np.percentile(returns, 1)),
                'cvar_95': float(returns[returns <= np.percentile(returns, 5)].mean()),
                'cvar_99': float(returns[returns <= np.percentile(returns, 1)].mean())
            }
            
            # Rolling metrics
            rolling_sharpe = self._calculate_rolling_sharpe(returns, window=60)
            rolling_volatility = returns.rolling(window=30).std() * np.sqrt(252)
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'avg_drawdown_duration': float(avg_drawdown_duration),
                'max_drawdown_duration': float(max_drawdown_duration),
                'return_distribution': return_stats,
                'rolling_sharpe_current': float(rolling_sharpe.iloc[-1]) if not rolling_sharpe.empty else 0,
                'rolling_volatility_current': float(rolling_volatility.iloc[-1]) if not rolling_volatility.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio analytics: {e}")
            return {}
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        try:
            rolling_returns = returns.rolling(window=window).mean() * 252
            rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
            rolling_sharpe = (rolling_returns - self.risk_free_rate) / rolling_volatility
            return rolling_sharpe.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating rolling Sharpe: {e}")
            return pd.Series([0] * len(returns))
    
    def _compare_to_benchmark(self, 
                            strategy_returns: pd.Series,
                            benchmark_returns: Optional[pd.Series]) -> Dict[str, Any]:
        """Compare strategy performance to benchmark"""
        try:
            if benchmark_returns is None or benchmark_returns.empty:
                return {'note': 'No benchmark data available'}
            
            # Align data
            aligned_benchmark = benchmark_returns.reindex(strategy_returns.index).fillna(0)
            
            # Cumulative performance comparison
            strategy_cumulative = (1 + strategy_returns).cumprod()
            benchmark_cumulative = (1 + aligned_benchmark).cumprod()
            
            # Performance metrics comparison
            strategy_total_return = strategy_cumulative.iloc[-1] - 1
            benchmark_total_return = benchmark_cumulative.iloc[-1] - 1
            
            strategy_volatility = strategy_returns.std() * np.sqrt(252)
            benchmark_volatility = aligned_benchmark.std() * np.sqrt(252)
            
            # Outperformance analysis
            excess_returns = strategy_returns - aligned_benchmark
            outperformance_rate = (excess_returns > 0).sum() / len(excess_returns)
            
            # Tracking error
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # Maximum relative drawdown
            relative_performance = strategy_cumulative / benchmark_cumulative
            relative_rolling_max = relative_performance.expanding().max()
            relative_drawdown = (relative_performance - relative_rolling_max) / relative_rolling_max
            max_relative_drawdown = float(relative_drawdown.min())
            
            return {
                'strategy_total_return': float(strategy_total_return),
                'benchmark_total_return': float(benchmark_total_return),
                'excess_return': float(strategy_total_return - benchmark_total_return),
                'strategy_volatility': float(strategy_volatility),
                'benchmark_volatility': float(benchmark_volatility),
                'outperformance_rate': float(outperformance_rate),
                'tracking_error': float(tracking_error),
                'max_relative_drawdown': max_relative_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error comparing to benchmark: {e}")
            return {}
    
    def _analyze_rolling_performance(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze rolling performance metrics"""
        try:
            rolling_windows = [30, 60, 90, 180]
            rolling_analysis = {}
            
            for window in rolling_windows:
                if len(returns) >= window:
                    rolling_returns = returns.rolling(window=window).sum()
                    rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
                    rolling_sharpe = self._calculate_rolling_sharpe(returns, window)
                    
                    rolling_analysis[f'window_{window}'] = {
                        'current_return': float(rolling_returns.iloc[-1]),
                        'avg_return': float(rolling_returns.mean()),
                        'current_volatility': float(rolling_volatility.iloc[-1]),
                        'avg_volatility': float(rolling_volatility.mean()),
                        'current_sharpe': float(rolling_sharpe.iloc[-1]),
                        'avg_sharpe': float(rolling_sharpe.mean())
                    }
            
            # Performance trend analysis
            trend_analysis = self._analyze_performance_trend(returns)
            rolling_analysis['trend_analysis'] = trend_analysis
            
            return rolling_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing rolling performance: {e}")
            return {}
    
    def _analyze_performance_trend(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance trend"""
        try:
            if len(returns) < 60:
                return {'trend': 'insufficient_data'}
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Linear trend over last 60 days
            recent_cumulative = cumulative_returns.tail(60)
            x = np.arange(len(recent_cumulative))
            trend_slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_cumulative)
            
            # Classify trend
            if trend_slope > 0.001 and p_value < 0.05:
                trend_direction = 'upward'
            elif trend_slope < -0.001 and p_value < 0.05:
                trend_direction = 'downward'
            else:
                trend_direction = 'sideways'
            
            # Trend strength
            trend_strength = abs(r_value)
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': float(trend_strength),
                'trend_slope': float(trend_slope),
                'trend_significance': float(1 - p_value)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trend: {e}")
            return {'trend': 'unknown'}
    
    def _analyze_regime_performance(self, 
                                  returns: pd.Series, 
                                  regime_predictions: pd.Series) -> Dict[str, Any]:
        """Analyze performance by regime"""
        try:
            regime_performance = {}
            
            for regime in regime_predictions.unique():
                regime_mask = regime_predictions == regime
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_performance[f'regime_{regime}'] = {
                        'total_return': float((1 + regime_returns).prod() - 1),
                        'avg_return': float(regime_returns.mean()),
                        'volatility': float(regime_returns.std() * np.sqrt(252)),
                        'sharpe_ratio': float(self.math_utils.safe_divide(
                            regime_returns.mean() * 252,
                            regime_returns.std() * np.sqrt(252),
                            0
                        )),
                        'win_rate': float((regime_returns > 0).sum() / len(regime_returns)),
                        'max_drawdown': float(self._calculate_max_drawdown(regime_returns)),
                        'periods': len(regime_returns)
                    }
            
            # Regime transition performance
            transition_performance = self._analyze_transition_performance(returns, regime_predictions)
            regime_performance['transition_analysis'] = transition_performance
            
            return regime_performance
            
        except Exception as e:
            logger.error(f"Error analyzing regime performance: {e}")
            return {}
    
    def _analyze_transition_performance(self, 
                                      returns: pd.Series, 
                                      regime_predictions: pd.Series) -> Dict[str, Any]:
        """Analyze performance during regime transitions"""
        try:
            regime_changes = regime_predictions != regime_predictions.shift(1)
            transition_periods = regime_changes[regime_changes].index
            
            if len(transition_periods) == 0:
                return {'note': 'No regime transitions detected'}
            
            # Performance around transitions
            transition_returns = []
            for transition_date in transition_periods:
                # Get returns in window around transition
                window_start = max(0, returns.index.get_loc(transition_date) - 5)
                window_end = min(len(returns), returns.index.get_loc(transition_date) + 6)
                window_returns = returns.iloc[window_start:window_end]
                transition_returns.extend(window_returns.tolist())
            
            if transition_returns:
                return {
                    'avg_transition_return': float(np.mean(transition_returns)),
                    'transition_volatility': float(np.std(transition_returns) * np.sqrt(252)),
                    'transition_frequency': float(len(transition_periods) / len(returns) * 252),  # Annualized
                    'transition_count': len(transition_periods)
                }
            else:
                return {'note': 'No transition return data available'}
                
        except Exception as e:
            logger.error(f"Error analyzing transition performance: {e}")
            return {}
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            if returns.empty:
                return {}
            
            # Value at Risk (VaR)
            var_95 = float(np.percentile(returns, 5))
            var_99 = float(np.percentile(returns, 1))
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95
            cvar_99 = float(returns[returns <= var_99].mean()) if (returns <= var_99).any() else var_99
            
            # Maximum consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            for ret in returns:
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            # Ulcer Index
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            ulcer_index = np.sqrt((drawdowns ** 2).mean())
            
            # Tail ratio
            upside_tail = returns[returns > np.percentile(returns, 95)].mean()
            downside_tail = abs(returns[returns < np.percentile(returns, 5)].mean())
            tail_ratio = self.math_utils.safe_divide(upside_tail, downside_tail, 1)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_consecutive_losses': max_consecutive_losses,
                'ulcer_index': float(ulcer_index),
                'tail_ratio': float(tail_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_performance_attribution(self, 
                                         returns: pd.Series, 
                                         regime_predictions: pd.Series) -> Dict[str, Any]:
        """Calculate performance attribution by regime"""
        try:
            total_return = (1 + returns).prod() - 1
            attribution = {}
            
            for regime in regime_predictions.unique():
                regime_mask = regime_predictions == regime
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_contribution = (1 + regime_returns).prod() - 1
                    weight = len(regime_returns) / len(returns)
                    
                    attribution[f'regime_{regime}'] = {
                        'contribution': float(regime_contribution),
                        'weight': float(weight),
                        'attribution_percent': float(regime_contribution / total_return * 100) if total_return != 0 else 0
                    }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating performance attribution: {e}")
            return {}
    
    def _calculate_overall_performance_score(self, 
                                           basic_metrics: Dict[str, float],
                                           risk_adjusted_metrics: Dict[str, float],
                                           regime_accuracy: Dict[str, float]) -> float:
        """Calculate overall performance score"""
        try:
            # Weighted scoring system
            sharpe_score = min(risk_adjusted_metrics.get('sharpe_ratio', 0) / 2.0, 1.0) * 0.3
            sortino_score = min(risk_adjusted_metrics.get('sortino_ratio', 0) / 2.5, 1.0) * 0.2
            return_score = min(basic_metrics.get('annualized_return', 0) / 0.15, 1.0) * 0.2  # Normalize to 15%
            drawdown_score = max(0, 1 + basic_metrics.get('max_drawdown', -0.2) / 0.2) * 0.15  # Penalize >20% drawdown
            accuracy_score = regime_accuracy.get('overall_accuracy', 0) * 0.15
            
            overall_score = sharpe_score + sortino_score + return_score + drawdown_score + accuracy_score
            return float(max(min(overall_score, 1.0), 0.0))
            
        except Exception as e:
            logger.error(f"Error calculating overall performance score: {e}")
            return 0.5
    
    def _update_performance_history(self, 
                                  returns: pd.Series,
                                  regime_predictions: pd.Series,
                                  actual_regimes: Optional[pd.Series]):
        """Update performance history"""
        try:
            # Add to history
            self.performance_history['daily_returns'].extend(returns.tolist())
            self.performance_history['regime_predictions'].extend(regime_predictions.tolist())
            self.performance_history['timestamps'].extend(returns.index.tolist())
            
            if actual_regimes is not None:
                aligned_actual = actual_regimes.reindex(returns.index).fillna(0)
                self.performance_history['actual_regimes'].extend(aligned_actual.tolist())
            
            # Calculate portfolio values
            if self.performance_history['portfolio_values']:
                last_value = self.performance_history['portfolio_values'][-1]
            else:
                last_value = 100  # Starting value
            
            for ret in returns:
                last_value *= (1 + ret)
                self.performance_history['portfolio_values'].append(last_value)
            
            # Trim history to reasonable size
            max_history = 2520  # ~10 years daily data
            for key in self.performance_history.keys():
                if len(self.performance_history[key]) > max_history:
                    excess = len(self.performance_history[key]) - max_history
                    self.performance_history[key] = self.performance_history[key][excess:]
                    
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    def _get_default_performance_evaluation(self) -> Dict[str, Any]:
        """Get default performance evaluation when evaluation fails"""
        return {
            'evaluation_timestamp': datetime.now(),
            'basic_metrics': {},
            'risk_adjusted_metrics': {},
            'regime_accuracy': {'regime_accuracy': 0.0},
            'portfolio_analytics': {},
            'benchmark_comparison': {},
            'rolling_analysis': {},
            'regime_performance': {},
            'risk_metrics': {},
            'attribution_analysis': {},
            'overall_score': 0.0,
            'status': 'evaluation_failed'
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance evaluation system"""
        try:
            if not self.performance_history['daily_returns']:
                return {'status': 'no_performance_history'}
            
            total_periods = len(self.performance_history['daily_returns'])
            total_return = (np.array(self.performance_history['portfolio_values'])[-1] / 
                          np.array(self.performance_history['portfolio_values'])[0] - 1) if self.performance_history['portfolio_values'] else 0
            
            return {
                'total_periods': total_periods,
                'total_return': float(total_return),
                'current_portfolio_value': float(self.performance_history['portfolio_values'][-1]) if self.performance_history['portfolio_values'] else 100,
                'evaluation_config': {
                    'evaluation_window': self.evaluation_window,
                    'risk_free_rate': self.risk_free_rate,
                    'benchmark_metrics': self.benchmark_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'status': 'error', 'error': str(e)}