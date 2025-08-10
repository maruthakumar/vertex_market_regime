"""
Strategy Inverter - Core Inversion Logic

Implements sophisticated strategy inversion algorithms to transform 
underperforming strategies into profitable ones.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class InversionType(Enum):
    """Types of strategy inversions"""
    SIMPLE = "simple"                    # Basic sign flip
    ULTA = "ulta"                       # Underperforming Long Term Average
    CONDITIONAL = "conditional"          # Based on market conditions
    ROLLING = "rolling"                 # Rolling window based
    SMART = "smart"                     # AI-driven intelligent inversion
    RISK_ADJUSTED = "risk_adjusted"     # Risk-return optimized inversion

@dataclass
class InversionResult:
    """Result from strategy inversion"""
    original_strategy: str
    inverted_strategy: str
    inversion_type: InversionType
    original_returns: pd.Series
    inverted_returns: pd.Series
    original_metrics: Dict[str, float]
    inverted_metrics: Dict[str, float]
    improvement_score: float
    confidence_score: float
    risk_score: float
    metadata: Dict[str, Any]

class StrategyInverter:
    """
    Advanced strategy inversion system
    
    Implements multiple inversion algorithms to transform underperforming
    strategies into profitable ones through sophisticated analysis.
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 min_improvement_threshold: float = 0.1,
                 confidence_threshold: float = 0.7,
                 lookback_window: int = 252,
                 enable_advanced_inversions: bool = True):
        """
        Initialize strategy inverter
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculations
            min_improvement_threshold: Minimum improvement required for inversion
            confidence_threshold: Minimum confidence required for inversion
            lookback_window: Lookback window for rolling calculations
            enable_advanced_inversions: Enable advanced inversion algorithms
        """
        self.risk_free_rate = risk_free_rate
        self.min_improvement_threshold = min_improvement_threshold
        self.confidence_threshold = confidence_threshold
        self.lookback_window = lookback_window
        self.enable_advanced_inversions = enable_advanced_inversions
        
        # Inversion statistics
        self.inversion_stats = {
            'total_analyzed': 0,
            'inversions_applied': 0,
            'total_improvement': 0.0,
            'success_rate': 0.0,
            'inversion_types_used': {}
        }
        
        logger.info(f"StrategyInverter initialized with improvement_threshold={min_improvement_threshold}")
    
    def analyze_strategy_for_inversion(self, 
                                     returns: pd.Series,
                                     strategy_name: str,
                                     market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze if a strategy is a candidate for inversion
        
        Args:
            returns: Strategy returns
            strategy_name: Name of the strategy
            market_data: Optional market data for advanced analysis
            
        Returns:
            Analysis results with inversion recommendations
        """
        self.inversion_stats['total_analyzed'] += 1
        
        # Calculate base metrics
        original_metrics = self._calculate_strategy_metrics(returns)
        
        # Determine inversion candidates
        inversion_analysis = {
            'strategy_name': strategy_name,
            'original_metrics': original_metrics,
            'inversion_candidates': [],
            'recommended_inversion': None,
            'confidence_scores': {},
            'risk_analysis': {}
        }
        
        # Test different inversion types
        inversion_types = [InversionType.SIMPLE, InversionType.ULTA]
        
        if self.enable_advanced_inversions:
            inversion_types.extend([
                InversionType.CONDITIONAL,
                InversionType.ROLLING,
                InversionType.RISK_ADJUSTED
            ])
        
        for inversion_type in inversion_types:
            try:
                inverted_returns = self._apply_inversion(
                    returns, inversion_type, market_data
                )
                
                if inverted_returns is not None:
                    inverted_metrics = self._calculate_strategy_metrics(inverted_returns)
                    improvement_score = self._calculate_improvement_score(
                        original_metrics, inverted_metrics
                    )
                    confidence_score = self._calculate_confidence_score(
                        returns, inverted_returns, inversion_type
                    )
                    
                    candidate = {
                        'type': inversion_type,
                        'improvement_score': improvement_score,
                        'confidence_score': confidence_score,
                        'inverted_metrics': inverted_metrics,
                        'meets_threshold': improvement_score >= self.min_improvement_threshold,
                        'meets_confidence': confidence_score >= self.confidence_threshold
                    }
                    
                    inversion_analysis['inversion_candidates'].append(candidate)
                    inversion_analysis['confidence_scores'][inversion_type.value] = confidence_score
                    
            except Exception as e:
                logger.warning(f"Error testing {inversion_type} for {strategy_name}: {e}")
        
        # Select best inversion candidate
        viable_candidates = [
            c for c in inversion_analysis['inversion_candidates']
            if c['meets_threshold'] and c['meets_confidence']
        ]
        
        if viable_candidates:
            # Sort by improvement score and confidence
            viable_candidates.sort(
                key=lambda x: (x['improvement_score'] * x['confidence_score']), 
                reverse=True
            )
            inversion_analysis['recommended_inversion'] = viable_candidates[0]
        
        return inversion_analysis
    
    def invert_strategy(self,
                       returns: pd.Series,
                       strategy_name: str,
                       inversion_type: Optional[InversionType] = None,
                       market_data: Optional[pd.DataFrame] = None,
                       force_inversion: bool = False) -> Optional[InversionResult]:
        """
        Invert a strategy using specified or optimal inversion method
        
        Args:
            returns: Strategy returns
            strategy_name: Name of the strategy
            inversion_type: Specific inversion type to use
            market_data: Optional market data for advanced analysis
            force_inversion: Force inversion even if not recommended
            
        Returns:
            InversionResult if successful, None otherwise
        """
        logger.info(f"Inverting strategy: {strategy_name}")
        
        # Analyze strategy if no specific inversion type provided
        if inversion_type is None:
            analysis = self.analyze_strategy_for_inversion(
                returns, strategy_name, market_data
            )
            
            if not force_inversion and analysis['recommended_inversion'] is None:
                logger.info(f"No viable inversion found for {strategy_name}")
                return None
            
            if analysis['recommended_inversion']:
                inversion_type = analysis['recommended_inversion']['type']
            else:
                inversion_type = InversionType.SIMPLE  # Default fallback
        
        # Apply inversion
        try:
            original_metrics = self._calculate_strategy_metrics(returns)
            inverted_returns = self._apply_inversion(returns, inversion_type, market_data)
            
            if inverted_returns is None:
                logger.warning(f"Inversion failed for {strategy_name}")
                return None
            
            inverted_metrics = self._calculate_strategy_metrics(inverted_returns)
            improvement_score = self._calculate_improvement_score(
                original_metrics, inverted_metrics
            )
            confidence_score = self._calculate_confidence_score(
                returns, inverted_returns, inversion_type
            )
            risk_score = self._calculate_risk_score(inverted_returns)
            
            # Create inversion result
            result = InversionResult(
                original_strategy=strategy_name,
                inverted_strategy=f"{strategy_name}_inverted_{inversion_type.value}",
                inversion_type=inversion_type,
                original_returns=returns.copy(),
                inverted_returns=inverted_returns,
                original_metrics=original_metrics,
                inverted_metrics=inverted_metrics,
                improvement_score=improvement_score,
                confidence_score=confidence_score,
                risk_score=risk_score,
                metadata={
                    'inversion_timestamp': pd.Timestamp.now(),
                    'inversion_parameters': self._get_inversion_parameters(inversion_type),
                    'market_conditions': self._analyze_market_conditions(market_data) if market_data is not None else {}
                }
            )
            
            # Update statistics
            self.inversion_stats['inversions_applied'] += 1
            self.inversion_stats['total_improvement'] += improvement_score
            
            inversion_type_key = inversion_type.value
            if inversion_type_key not in self.inversion_stats['inversion_types_used']:
                self.inversion_stats['inversion_types_used'][inversion_type_key] = 0
            self.inversion_stats['inversion_types_used'][inversion_type_key] += 1
            
            self.inversion_stats['success_rate'] = (
                self.inversion_stats['inversions_applied'] / 
                max(1, self.inversion_stats['total_analyzed'])
            )
            
            logger.info(f"Successfully inverted {strategy_name} using {inversion_type.value} "
                       f"(improvement: {improvement_score:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error inverting strategy {strategy_name}: {e}")
            return None
    
    def _apply_inversion(self,
                        returns: pd.Series,
                        inversion_type: InversionType,
                        market_data: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
        """Apply specific inversion type to returns"""
        
        if inversion_type == InversionType.SIMPLE:
            return self._simple_inversion(returns)
        
        elif inversion_type == InversionType.ULTA:
            return self._ulta_inversion(returns)
        
        elif inversion_type == InversionType.CONDITIONAL:
            return self._conditional_inversion(returns, market_data)
        
        elif inversion_type == InversionType.ROLLING:
            return self._rolling_inversion(returns)
        
        elif inversion_type == InversionType.RISK_ADJUSTED:
            return self._risk_adjusted_inversion(returns)
        
        elif inversion_type == InversionType.SMART:
            return self._smart_inversion(returns, market_data)
        
        else:
            logger.warning(f"Unknown inversion type: {inversion_type}")
            return None
    
    def _simple_inversion(self, returns: pd.Series) -> pd.Series:
        """Simple sign flip inversion"""
        # Clean the data first
        cleaned_returns = self._clean_returns(returns)
        return -cleaned_returns
    
    def _ulta_inversion(self, returns: pd.Series) -> pd.Series:
        """
        Underperforming Long Term Average (ULTA) inversion
        
        Inverts strategy if cumulative performance is below long-term average
        """
        cleaned_returns = self._clean_returns(returns)
        
        # Calculate rolling mean for long-term average
        rolling_mean = cleaned_returns.rolling(
            window=min(self.lookback_window, len(cleaned_returns) // 2),
            min_periods=20
        ).mean()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + cleaned_returns).cumprod()
        expected_cumulative = (1 + rolling_mean).cumprod()
        
        # Create inversion mask where performance is below expected
        underperforming_mask = cumulative_returns < expected_cumulative
        
        # Apply conditional inversion
        inverted_returns = cleaned_returns.copy()
        inverted_returns[underperforming_mask] = -inverted_returns[underperforming_mask]
        
        return inverted_returns
    
    def _conditional_inversion(self, 
                              returns: pd.Series, 
                              market_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Conditional inversion based on market conditions
        
        Inverts based on market regime, volatility, or other conditions
        """
        cleaned_returns = self._clean_returns(returns)
        
        if market_data is None:
            # Fallback to volatility-based conditional inversion
            return self._volatility_conditional_inversion(cleaned_returns)
        
        # Use market regime information if available
        if 'Market_Regime' in market_data.columns:
            return self._regime_conditional_inversion(cleaned_returns, market_data)
        
        # Use volatility-based approach as fallback
        return self._volatility_conditional_inversion(cleaned_returns)
    
    def _volatility_conditional_inversion(self, returns: pd.Series) -> pd.Series:
        """Invert based on volatility conditions"""
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20, min_periods=10).std()
        vol_threshold = rolling_vol.quantile(0.7)  # High volatility threshold
        
        # Invert during high volatility periods
        inverted_returns = returns.copy()
        high_vol_mask = rolling_vol > vol_threshold
        inverted_returns[high_vol_mask] = -inverted_returns[high_vol_mask]
        
        return inverted_returns
    
    def _regime_conditional_inversion(self,
                                    returns: pd.Series,
                                    market_data: pd.DataFrame) -> pd.Series:
        """Invert based on market regime"""
        inverted_returns = returns.copy()
        
        # Align indices
        aligned_data = market_data.reindex(returns.index, method='ffill')
        
        if 'Market_Regime' in aligned_data.columns:
            # Define regimes that typically benefit from inversion
            inversion_regimes = ['Bearish', 'High_Volatility', 'Trend_Down']
            
            for regime in inversion_regimes:
                regime_mask = aligned_data['Market_Regime'].str.contains(
                    regime, case=False, na=False
                )
                inverted_returns[regime_mask] = -inverted_returns[regime_mask]
        
        return inverted_returns
    
    def _rolling_inversion(self, returns: pd.Series) -> pd.Series:
        """
        Rolling window inversion based on performance windows
        
        Inverts strategy in windows where it underperforms
        """
        cleaned_returns = self._clean_returns(returns)
        window_size = min(60, len(cleaned_returns) // 4)  # Quarterly windows
        
        inverted_returns = cleaned_returns.copy()
        
        for i in range(window_size, len(cleaned_returns), window_size):
            window_start = max(0, i - window_size)
            window_end = i
            
            window_returns = cleaned_returns.iloc[window_start:window_end]
            window_performance = window_returns.mean()
            
            # Invert if window performance is negative
            if window_performance < 0:
                inverted_returns.iloc[window_start:window_end] = -window_returns
        
        return inverted_returns
    
    def _risk_adjusted_inversion(self, returns: pd.Series) -> pd.Series:
        """
        Risk-adjusted inversion that considers Sharpe ratio optimization
        
        Inverts parts of strategy to maximize risk-adjusted returns
        """
        cleaned_returns = self._clean_returns(returns)
        
        # Calculate rolling Sharpe ratio
        rolling_mean = cleaned_returns.rolling(window=60, min_periods=20).mean()
        rolling_std = cleaned_returns.rolling(window=60, min_periods=20).std()
        rolling_sharpe = (rolling_mean - self.risk_free_rate/252) / (rolling_std + 1e-8)
        
        # Calculate rolling Sharpe for inverted returns
        inverted_returns_temp = -cleaned_returns
        rolling_mean_inv = inverted_returns_temp.rolling(window=60, min_periods=20).mean()
        rolling_std_inv = inverted_returns_temp.rolling(window=60, min_periods=20).std()
        rolling_sharpe_inv = (rolling_mean_inv - self.risk_free_rate/252) / (rolling_std_inv + 1e-8)
        
        # Use inversion where it improves Sharpe ratio
        inversion_mask = rolling_sharpe_inv > rolling_sharpe
        
        inverted_returns = cleaned_returns.copy()
        inverted_returns[inversion_mask] = -inverted_returns[inversion_mask]
        
        return inverted_returns
    
    def _smart_inversion(self,
                        returns: pd.Series,
                        market_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        AI-driven smart inversion using multiple signals
        
        Combines multiple inversion strategies intelligently
        """
        # This is a placeholder for more advanced ML-based inversion
        # For now, combine multiple simple approaches
        
        simple_inv = self._simple_inversion(returns)
        ulta_inv = self._ulta_inversion(returns)
        rolling_inv = self._rolling_inversion(returns)
        
        # Weighted combination based on historical performance
        # This could be replaced with ML models in the future
        weights = [0.3, 0.4, 0.3]  # Equal-ish weighting
        
        combined_inv = (
            weights[0] * simple_inv +
            weights[1] * ulta_inv +
            weights[2] * rolling_inv
        )
        
        return combined_inv
    
    def _clean_returns(self, returns: pd.Series) -> pd.Series:
        """Clean returns data by handling outliers and missing values"""
        cleaned = returns.copy()
        
        # Handle infinite values
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with zero
        cleaned = cleaned.fillna(0)
        
        # Clip extreme outliers (beyond 3 standard deviations)
        std_threshold = 3
        mean_return = cleaned.mean()
        std_return = cleaned.std()
        
        lower_bound = mean_return - std_threshold * std_return
        upper_bound = mean_return + std_threshold * std_return
        
        cleaned = np.clip(cleaned, lower_bound, upper_bound)
        
        return pd.Series(cleaned, index=returns.index)
    
    def _calculate_strategy_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive strategy metrics"""
        cleaned_returns = self._clean_returns(returns)
        
        if len(cleaned_returns) == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'calmar_ratio': 0.0
            }
        
        # Basic returns
        total_return = cleaned_returns.sum()
        annual_return = cleaned_returns.mean() * 252
        volatility = cleaned_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / (volatility + 1e-8)
        
        # Maximum drawdown
        cumulative = (1 + cleaned_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Win rate
        positive_returns = cleaned_returns > 0
        win_rate = positive_returns.mean() if len(cleaned_returns) > 0 else 0
        
        # Profit factor
        gross_profit = cleaned_returns[cleaned_returns > 0].sum()
        gross_loss = abs(cleaned_returns[cleaned_returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        # Calmar ratio
        calmar_ratio = annual_return / (max_drawdown + 1e-8)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_improvement_score(self,
                                   original_metrics: Dict[str, float],
                                   inverted_metrics: Dict[str, float]) -> float:
        """Calculate improvement score from inversion"""
        
        # Weight different metrics
        metric_weights = {
            'sharpe_ratio': 0.4,
            'calmar_ratio': 0.3,
            'annual_return': 0.2,
            'max_drawdown': -0.1  # Negative weight (lower is better)
        }
        
        improvement_score = 0.0
        
        for metric, weight in metric_weights.items():
            if metric in original_metrics and metric in inverted_metrics:
                original_val = original_metrics[metric]
                inverted_val = inverted_metrics[metric]
                
                if metric == 'max_drawdown':
                    # For drawdown, improvement is reduction (original - inverted)
                    improvement = (original_val - inverted_val) / (abs(original_val) + 1e-8)
                else:
                    # For other metrics, improvement is increase
                    improvement = (inverted_val - original_val) / (abs(original_val) + 1e-8)
                
                improvement_score += weight * improvement
        
        return improvement_score
    
    def _calculate_confidence_score(self,
                                  original_returns: pd.Series,
                                  inverted_returns: pd.Series,
                                  inversion_type: InversionType) -> float:
        """Calculate confidence score for inversion"""
        
        # Base confidence on consistency of improvement
        original_metrics = self._calculate_strategy_metrics(original_returns)
        inverted_metrics = self._calculate_strategy_metrics(inverted_returns)
        
        # Rolling window analysis
        window_size = min(60, len(original_returns) // 4)
        consistency_scores = []
        
        for i in range(window_size, len(original_returns), window_size//2):
            start_idx = max(0, i - window_size)
            end_idx = i
            
            orig_window = original_returns.iloc[start_idx:end_idx]
            inv_window = inverted_returns.iloc[start_idx:end_idx]
            
            if len(orig_window) > 10:  # Minimum window size
                orig_sharpe = self._calculate_strategy_metrics(orig_window)['sharpe_ratio']
                inv_sharpe = self._calculate_strategy_metrics(inv_window)['sharpe_ratio']
                
                consistency_scores.append(1.0 if inv_sharpe > orig_sharpe else 0.0)
        
        consistency_rate = np.mean(consistency_scores) if consistency_scores else 0.5
        
        # Combine with overall improvement magnitude
        overall_improvement = inverted_metrics['sharpe_ratio'] - original_metrics['sharpe_ratio']
        improvement_confidence = min(1.0, max(0.0, overall_improvement + 1.0))
        
        # Weight by inversion type complexity
        type_confidence_weights = {
            InversionType.SIMPLE: 0.8,
            InversionType.ULTA: 0.9,
            InversionType.CONDITIONAL: 0.85,
            InversionType.ROLLING: 0.75,
            InversionType.RISK_ADJUSTED: 0.95,
            InversionType.SMART: 0.7
        }
        
        type_weight = type_confidence_weights.get(inversion_type, 0.8)
        
        return (0.4 * consistency_rate + 0.4 * improvement_confidence + 0.2 * type_weight)
    
    def _calculate_risk_score(self, returns: pd.Series) -> float:
        """Calculate risk score for inverted strategy"""
        metrics = self._calculate_strategy_metrics(returns)
        
        # Normalize risk components
        vol_score = min(1.0, metrics['volatility'] / 0.3)  # 30% vol = max risk
        dd_score = min(1.0, metrics['max_drawdown'] / 0.5)  # 50% dd = max risk
        negative_skew = max(0.0, -returns.skew()) / 2.0 if len(returns) > 30 else 0.5
        
        # Combined risk score (0 = low risk, 1 = high risk)
        risk_score = 0.4 * vol_score + 0.4 * dd_score + 0.2 * negative_skew
        
        return min(1.0, risk_score)
    
    def _get_inversion_parameters(self, inversion_type: InversionType) -> Dict[str, Any]:
        """Get parameters used for specific inversion type"""
        return {
            'inversion_type': inversion_type.value,
            'lookback_window': self.lookback_window,
            'risk_free_rate': self.risk_free_rate,
            'min_improvement_threshold': self.min_improvement_threshold,
            'confidence_threshold': self.confidence_threshold
        }
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions during inversion period"""
        conditions = {}
        
        if 'Market_Regime' in market_data.columns:
            regime_counts = market_data['Market_Regime'].value_counts()
            conditions['dominant_regime'] = regime_counts.index[0] if len(regime_counts) > 0 else 'Unknown'
            conditions['regime_distribution'] = regime_counts.to_dict()
        
        return conditions
    
    def get_inversion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive inversion statistics"""
        return {
            'performance_stats': self.inversion_stats.copy(),
            'average_improvement': (
                self.inversion_stats['total_improvement'] / 
                max(1, self.inversion_stats['inversions_applied'])
            ),
            'configuration': {
                'risk_free_rate': self.risk_free_rate,
                'min_improvement_threshold': self.min_improvement_threshold,
                'confidence_threshold': self.confidence_threshold,
                'lookback_window': self.lookback_window,
                'enable_advanced_inversions': self.enable_advanced_inversions
            }
        }
    
    def reset_statistics(self):
        """Reset inversion statistics"""
        self.inversion_stats = {
            'total_analyzed': 0,
            'inversions_applied': 0,
            'total_improvement': 0.0,
            'success_rate': 0.0,
            'inversion_types_used': {}
        }
        logger.info("Inversion statistics reset")