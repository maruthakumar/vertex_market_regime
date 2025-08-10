"""
Inversion Analyzer - Advanced Analysis for Strategy Inversions

Provides sophisticated analysis of strategy performance patterns 
to identify optimal inversion opportunities and strategies.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

@dataclass
class PerformancePattern:
    """Identified performance pattern in strategy"""
    pattern_type: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    severity_score: float
    pattern_metrics: Dict[str, float]
    suggested_inversion: str
    confidence: float

@dataclass 
class InversionOpportunity:
    """Identified inversion opportunity"""
    strategy_name: str
    opportunity_type: str
    time_period: Tuple[pd.Timestamp, pd.Timestamp]
    expected_improvement: float
    risk_level: str
    confidence_score: float
    supporting_evidence: List[str]
    recommended_parameters: Dict[str, Any]

class InversionAnalyzer:
    """
    Advanced analyzer for strategy inversion opportunities
    
    Uses statistical analysis, pattern recognition, and machine learning
    to identify optimal inversion strategies and timing.
    """
    
    def __init__(self,
                 analysis_window: int = 252,
                 pattern_detection_threshold: float = 0.05,
                 clustering_features: List[str] = None,
                 enable_ml_analysis: bool = True):
        """
        Initialize inversion analyzer
        
        Args:
            analysis_window: Days to analyze for patterns
            pattern_detection_threshold: Threshold for pattern significance
            clustering_features: Features to use for clustering analysis
            enable_ml_analysis: Enable machine learning based analysis
        """
        self.analysis_window = analysis_window
        self.pattern_detection_threshold = pattern_detection_threshold
        self.enable_ml_analysis = enable_ml_analysis
        
        # Default clustering features
        self.clustering_features = clustering_features or [
            'returns', 'volatility', 'skewness', 'kurtosis', 
            'trend', 'momentum', 'mean_reversion'
        ]
        
        # Analysis cache
        self.analysis_cache = {}
        
        logger.info("InversionAnalyzer initialized")
    
    def analyze_strategy_portfolio(self,
                                 strategy_data: pd.DataFrame,
                                 strategy_columns: List[str],
                                 market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze entire portfolio for inversion opportunities
        
        Args:
            strategy_data: DataFrame with strategy returns
            strategy_columns: List of strategy column names
            market_data: Optional market data for context
            
        Returns:
            Comprehensive portfolio analysis with inversion opportunities
        """
        logger.info(f"Analyzing portfolio of {len(strategy_columns)} strategies")
        
        portfolio_analysis = {
            'strategy_analyses': {},
            'portfolio_patterns': {},
            'inversion_opportunities': [],
            'correlation_analysis': {},
            'risk_analysis': {},
            'recommendations': []
        }
        
        # Analyze individual strategies
        for strategy in strategy_columns:
            if strategy in strategy_data.columns:
                strategy_analysis = self.analyze_individual_strategy(
                    strategy_data[strategy], strategy, market_data
                )
                portfolio_analysis['strategy_analyses'][strategy] = strategy_analysis
        
        # Portfolio-level analysis
        portfolio_analysis['correlation_analysis'] = self._analyze_portfolio_correlations(
            strategy_data[strategy_columns]
        )
        
        portfolio_analysis['portfolio_patterns'] = self._detect_portfolio_patterns(
            strategy_data[strategy_columns], market_data
        )
        
        # Identify inversion opportunities
        portfolio_analysis['inversion_opportunities'] = self._identify_inversion_opportunities(
            portfolio_analysis['strategy_analyses']
        )
        
        # Risk analysis
        portfolio_analysis['risk_analysis'] = self._analyze_inversion_risks(
            strategy_data[strategy_columns], portfolio_analysis['inversion_opportunities']
        )
        
        # Generate recommendations
        portfolio_analysis['recommendations'] = self._generate_portfolio_recommendations(
            portfolio_analysis
        )
        
        return portfolio_analysis
    
    def analyze_individual_strategy(self,
                                   returns: pd.Series,
                                   strategy_name: str,
                                   market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze individual strategy for inversion patterns
        
        Args:
            returns: Strategy returns
            strategy_name: Name of the strategy
            market_data: Optional market data for context
            
        Returns:
            Detailed strategy analysis
        """
        logger.debug(f"Analyzing strategy: {strategy_name}")
        
        # Check cache
        cache_key = f"{strategy_name}_{hash(tuple(returns.values))}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        analysis = {
            'strategy_name': strategy_name,
            'basic_metrics': self._calculate_basic_metrics(returns),
            'performance_patterns': self._detect_performance_patterns(returns),
            'regime_analysis': self._analyze_regime_performance(returns, market_data),
            'drawdown_analysis': self._analyze_drawdown_patterns(returns),
            'seasonality_analysis': self._analyze_seasonality(returns),
            'volatility_analysis': self._analyze_volatility_patterns(returns),
            'inversion_candidates': self._identify_strategy_inversion_candidates(returns),
            'ml_analysis': {}
        }
        
        # Advanced ML analysis if enabled
        if self.enable_ml_analysis:
            analysis['ml_analysis'] = self._ml_based_analysis(returns, market_data)
        
        # Cache result
        self.analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic strategy metrics"""
        cleaned_returns = self._clean_returns(returns)
        
        if len(cleaned_returns) == 0:
            return {}
        
        metrics = {
            'total_return': cleaned_returns.sum(),
            'mean_return': cleaned_returns.mean(),
            'volatility': cleaned_returns.std(),
            'skewness': stats.skew(cleaned_returns.dropna()),
            'kurtosis': stats.kurtosis(cleaned_returns.dropna()),
            'sharpe_ratio': self._calculate_sharpe_ratio(cleaned_returns),
            'max_drawdown': self._calculate_max_drawdown(cleaned_returns),
            'win_rate': (cleaned_returns > 0).mean(),
            'profit_factor': self._calculate_profit_factor(cleaned_returns),
            'calmar_ratio': self._calculate_calmar_ratio(cleaned_returns)
        }
        
        return metrics
    
    def _detect_performance_patterns(self, returns: pd.Series) -> List[PerformancePattern]:
        """Detect specific performance patterns that suggest inversion opportunities"""
        patterns = []
        cleaned_returns = self._clean_returns(returns)
        
        if len(cleaned_returns) < 30:  # Need minimum data
            return patterns
        
        # Detect consistent underperformance periods
        patterns.extend(self._detect_underperformance_periods(cleaned_returns))
        
        # Detect volatility spikes with negative returns
        patterns.extend(self._detect_volatility_spike_patterns(cleaned_returns))
        
        # Detect trend reversal patterns
        patterns.extend(self._detect_trend_reversal_patterns(cleaned_returns))
        
        # Detect regime-specific underperformance
        patterns.extend(self._detect_regime_underperformance(cleaned_returns))
        
        return patterns
    
    def _detect_underperformance_periods(self, returns: pd.Series) -> List[PerformancePattern]:
        """Detect periods of consistent underperformance"""
        patterns = []
        
        # Rolling cumulative returns
        window = min(60, len(returns) // 4)
        rolling_cumret = returns.rolling(window=window).sum()
        
        # Find periods where rolling return is consistently negative
        negative_periods = rolling_cumret < -self.pattern_detection_threshold
        
        # Group consecutive periods
        period_groups = []
        current_group = []
        
        for i, is_negative in enumerate(negative_periods):
            if is_negative:
                current_group.append(i)
            else:
                if len(current_group) >= 10:  # Minimum period length
                    period_groups.append(current_group)
                current_group = []
        
        # Add final group if exists
        if len(current_group) >= 10:
            period_groups.append(current_group)
        
        # Create pattern objects
        for group in period_groups:
            start_idx, end_idx = group[0], group[-1]
            start_date = returns.index[start_idx]
            end_date = returns.index[end_idx]
            
            period_returns = returns.iloc[start_idx:end_idx+1]
            severity = abs(period_returns.sum())
            
            pattern = PerformancePattern(
                pattern_type="underperformance_period",
                start_date=start_date,
                end_date=end_date,
                duration_days=(end_date - start_date).days,
                severity_score=severity,
                pattern_metrics={
                    'cumulative_loss': period_returns.sum(),
                    'worst_day': period_returns.min(),
                    'negative_days_pct': (period_returns < 0).mean()
                },
                suggested_inversion="rolling",
                confidence=min(0.9, severity * 2)
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_volatility_spike_patterns(self, returns: pd.Series) -> List[PerformancePattern]:
        """Detect high volatility periods with poor performance"""
        patterns = []
        
        # Calculate rolling volatility
        vol_window = 20
        rolling_vol = returns.rolling(window=vol_window).std()
        vol_threshold = rolling_vol.quantile(0.8)  # Top 20% volatility
        
        # Find high volatility periods with negative returns
        high_vol_periods = rolling_vol > vol_threshold
        neg_return_periods = returns.rolling(window=vol_window).mean() < 0
        
        problem_periods = high_vol_periods & neg_return_periods
        
        # Group consecutive periods
        consecutive_periods = self._group_consecutive_periods(problem_periods, min_length=5)
        
        for start_idx, end_idx in consecutive_periods:
            start_date = returns.index[start_idx]
            end_date = returns.index[end_idx]
            
            period_returns = returns.iloc[start_idx:end_idx+1]
            period_vol = rolling_vol.iloc[start_idx:end_idx+1]
            
            pattern = PerformancePattern(
                pattern_type="volatility_spike",
                start_date=start_date,
                end_date=end_date,
                duration_days=(end_date - start_date).days,
                severity_score=period_vol.mean(),
                pattern_metrics={
                    'avg_volatility': period_vol.mean(),
                    'avg_return': period_returns.mean(),
                    'vol_to_return_ratio': period_vol.mean() / (abs(period_returns.mean()) + 1e-8)
                },
                suggested_inversion="conditional",
                confidence=0.75
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_trend_reversal_patterns(self, returns: pd.Series) -> List[PerformancePattern]:
        """Detect trend reversal patterns that might benefit from inversion"""
        patterns = []
        
        # Calculate moving averages
        short_ma = returns.rolling(window=10).mean()
        long_ma = returns.rolling(window=30).mean()
        
        # Detect trend changes
        trend_signal = short_ma - long_ma
        trend_changes = trend_signal.diff()
        
        # Find significant trend reversals
        reversal_threshold = trend_changes.std() * 1.5
        significant_reversals = abs(trend_changes) > reversal_threshold
        
        reversal_dates = returns.index[significant_reversals]
        
        for reversal_date in reversal_dates:
            # Analyze period around reversal
            reversal_idx = returns.index.get_loc(reversal_date)
            start_idx = max(0, reversal_idx - 15)
            end_idx = min(len(returns) - 1, reversal_idx + 15)
            
            period_returns = returns.iloc[start_idx:end_idx+1]
            
            if period_returns.sum() < -self.pattern_detection_threshold:
                pattern = PerformancePattern(
                    pattern_type="trend_reversal",
                    start_date=returns.index[start_idx],
                    end_date=returns.index[end_idx],
                    duration_days=30,
                    severity_score=abs(trend_changes.loc[reversal_date]),
                    pattern_metrics={
                        'reversal_magnitude': trend_changes.loc[reversal_date],
                        'period_return': period_returns.sum(),
                        'pre_reversal_trend': trend_signal.loc[reversal_date]
                    },
                    suggested_inversion="smart",
                    confidence=0.65
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_regime_underperformance(self, returns: pd.Series) -> List[PerformancePattern]:
        """Detect regime-specific underperformance using clustering"""
        patterns = []
        
        if not self.enable_ml_analysis or len(returns) < 100:
            return patterns
        
        try:
            # Prepare features for clustering
            features_df = self._prepare_clustering_features(returns)
            
            if features_df.empty:
                return patterns
            
            # Perform clustering to identify regimes
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df.dropna())
            
            # Use 3-5 clusters for regime identification
            n_clusters = min(4, max(2, len(scaled_features) // 30))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(scaled_features)
            
            # Analyze performance by regime
            regime_df = pd.DataFrame({
                'regime': regime_labels,
                'returns': returns.iloc[:len(regime_labels)]
            })
            
            regime_performance = regime_df.groupby('regime')['returns'].agg([
                'mean', 'std', 'count'
            ])
            
            # Find underperforming regimes
            underperforming_regimes = regime_performance[
                regime_performance['mean'] < -self.pattern_detection_threshold
            ]
            
            for regime_id in underperforming_regimes.index:
                regime_mask = regime_df['regime'] == regime_id
                regime_periods = self._get_regime_periods(regime_mask)
                
                for start_idx, end_idx in regime_periods:
                    if end_idx - start_idx >= 5:  # Minimum period length
                        start_date = returns.index[start_idx]
                        end_date = returns.index[end_idx]
                        
                        pattern = PerformancePattern(
                            pattern_type="regime_underperformance",
                            start_date=start_date,
                            end_date=end_date,
                            duration_days=(end_date - start_date).days,
                            severity_score=abs(regime_performance.loc[regime_id, 'mean']),
                            pattern_metrics={
                                'regime_id': int(regime_id),
                                'regime_mean_return': regime_performance.loc[regime_id, 'mean'],
                                'regime_volatility': regime_performance.loc[regime_id, 'std'],
                                'regime_frequency': regime_performance.loc[regime_id, 'count']
                            },
                            suggested_inversion="conditional",
                            confidence=0.8
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"Error in regime underperformance detection: {e}")
        
        return patterns
    
    def _analyze_regime_performance(self,
                                  returns: pd.Series,
                                  market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""
        regime_analysis = {}
        
        if market_data is None or 'Market_Regime' not in market_data.columns:
            return regime_analysis
        
        # Align data
        aligned_data = market_data.reindex(returns.index, method='ffill')
        
        if 'Market_Regime' in aligned_data.columns:
            regime_performance = {}
            
            for regime in aligned_data['Market_Regime'].unique():
                if pd.notna(regime):
                    regime_mask = aligned_data['Market_Regime'] == regime
                    regime_returns = returns[regime_mask]
                    
                    if len(regime_returns) > 5:
                        regime_performance[regime] = {
                            'mean_return': regime_returns.mean(),
                            'volatility': regime_returns.std(),
                            'sharpe_ratio': self._calculate_sharpe_ratio(regime_returns),
                            'max_drawdown': self._calculate_max_drawdown(regime_returns),
                            'win_rate': (regime_returns > 0).mean(),
                            'observation_count': len(regime_returns)
                        }
            
            regime_analysis['regime_performance'] = regime_performance
            
            # Identify worst performing regimes
            if regime_performance:
                worst_regimes = sorted(
                    regime_performance.items(),
                    key=lambda x: x[1]['mean_return']
                )[:2]
                
                regime_analysis['worst_regimes'] = [
                    {'regime': regime, 'metrics': metrics}
                    for regime, metrics in worst_regimes
                    if metrics['mean_return'] < 0
                ]
        
        return regime_analysis
    
    def _analyze_drawdown_patterns(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze drawdown patterns for inversion insights"""
        # Calculate drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if i - start_idx >= 5:  # Minimum duration
                    drawdown_periods.append((start_idx, i))
        
        # Analyze drawdown characteristics
        drawdown_analysis = {
            'max_drawdown': abs(drawdown.min()),
            'avg_drawdown_duration': 0,
            'drawdown_frequency': 0,
            'recovery_times': [],
            'deep_drawdowns': []
        }
        
        if drawdown_periods:
            durations = [end - start for start, end in drawdown_periods]
            drawdown_analysis['avg_drawdown_duration'] = np.mean(durations)
            drawdown_analysis['drawdown_frequency'] = len(drawdown_periods) / len(returns) * 252
            
            # Analyze recovery times
            for start, end in drawdown_periods:
                dd_returns = returns.iloc[start:end+1]
                min_point = dd_returns.cumsum().idxmin()
                
                # Find recovery (if any)
                recovery_idx = None
                cum_from_min = dd_returns.loc[min_point:].cumsum()
                
                for idx, cum_ret in cum_from_min.items():
                    if cum_ret >= 0:
                        recovery_idx = returns.index.get_loc(idx)
                        break
                
                if recovery_idx:
                    recovery_time = recovery_idx - returns.index.get_loc(min_point)
                    drawdown_analysis['recovery_times'].append(recovery_time)
                
                # Identify deep drawdowns (>10%)
                if abs(dd_returns.sum()) > 0.1:
                    drawdown_analysis['deep_drawdowns'].append({
                        'start_date': returns.index[start],
                        'end_date': returns.index[end],
                        'magnitude': abs(dd_returns.sum()),
                        'duration': end - start
                    })
        
        return drawdown_analysis
    
    def _analyze_seasonality(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal patterns in returns"""
        seasonality_analysis = {}
        
        if not isinstance(returns.index, pd.DatetimeIndex):
            return seasonality_analysis
        
        # Monthly analysis
        monthly_returns = returns.groupby(returns.index.month).agg([
            'mean', 'std', 'count'
        ])
        
        # Find worst performing months
        worst_months = monthly_returns.nsmallest(3, 'mean')
        
        seasonality_analysis['monthly_performance'] = monthly_returns.to_dict()
        seasonality_analysis['worst_months'] = worst_months.index.tolist()
        
        # Day of week analysis (if sufficient data)
        if len(returns) > 100:
            dow_returns = returns.groupby(returns.index.dayofweek).agg([
                'mean', 'std', 'count'
            ])
            seasonality_analysis['day_of_week_performance'] = dow_returns.to_dict()
        
        return seasonality_analysis
    
    def _analyze_volatility_patterns(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze volatility patterns for inversion insights"""
        # Rolling volatility
        vol_window = 20
        rolling_vol = returns.rolling(window=vol_window).std()
        
        volatility_analysis = {
            'mean_volatility': rolling_vol.mean(),
            'volatility_of_volatility': rolling_vol.std(),
            'high_vol_periods': [],
            'vol_clustering': False
        }
        
        # Identify high volatility periods
        vol_threshold = rolling_vol.quantile(0.8)
        high_vol_mask = rolling_vol > vol_threshold
        
        high_vol_periods = self._group_consecutive_periods(high_vol_mask, min_length=3)
        
        for start_idx, end_idx in high_vol_periods:
            period_returns = returns.iloc[start_idx:end_idx+1]
            period_vol = rolling_vol.iloc[start_idx:end_idx+1]
            
            volatility_analysis['high_vol_periods'].append({
                'start_date': returns.index[start_idx],
                'end_date': returns.index[end_idx],
                'avg_volatility': period_vol.mean(),
                'period_return': period_returns.sum(),
                'duration': end_idx - start_idx + 1
            })
        
        # Check for volatility clustering
        vol_autocorr = rolling_vol.autocorr(lag=1)
        volatility_analysis['vol_clustering'] = vol_autocorr > 0.3
        volatility_analysis['vol_autocorrelation'] = vol_autocorr
        
        return volatility_analysis
    
    def _identify_strategy_inversion_candidates(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """Identify specific inversion candidates for the strategy"""
        candidates = []
        
        # Based on overall performance
        total_return = returns.sum()
        if total_return < -0.05:  # Losing strategy
            candidates.append({
                'type': 'simple_inversion',
                'reason': 'Overall negative performance',
                'priority': 'high',
                'expected_improvement': abs(total_return),
                'confidence': 0.8
            })
        
        # Based on Sharpe ratio
        sharpe = self._calculate_sharpe_ratio(returns)
        if sharpe < -0.5:
            candidates.append({
                'type': 'risk_adjusted_inversion',
                'reason': 'Poor risk-adjusted returns',
                'priority': 'high',
                'expected_improvement': abs(sharpe),
                'confidence': 0.85
            })
        
        # Based on drawdown
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd > 0.3:  # High drawdown
            candidates.append({
                'type': 'drawdown_optimized_inversion',
                'reason': 'Excessive drawdowns',
                'priority': 'medium',
                'expected_improvement': max_dd * 0.5,
                'confidence': 0.7
            })
        
        return candidates
    
    def _ml_based_analysis(self,
                          returns: pd.Series,
                          market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Machine learning based analysis for inversion opportunities"""
        ml_analysis = {}
        
        try:
            # Feature engineering
            features_df = self._prepare_clustering_features(returns)
            
            if len(features_df) < 50:  # Need sufficient data
                return ml_analysis
            
            # Clustering analysis
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df.dropna())
            
            # Multiple clustering approaches
            ml_analysis['clustering_results'] = {}
            
            for n_clusters in [2, 3, 4]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # Analyze cluster performance
                cluster_performance = {}
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_returns = returns.iloc[:len(cluster_labels)][cluster_mask]
                    
                    if len(cluster_returns) > 5:
                        cluster_performance[cluster_id] = {
                            'mean_return': cluster_returns.mean(),
                            'count': len(cluster_returns),
                            'performance_rank': 0  # Will be set later
                        }
                
                # Rank clusters by performance
                sorted_clusters = sorted(
                    cluster_performance.items(),
                    key=lambda x: x[1]['mean_return'],
                    reverse=True
                )
                
                for rank, (cluster_id, perf) in enumerate(sorted_clusters):
                    cluster_performance[cluster_id]['performance_rank'] = rank
                
                ml_analysis['clustering_results'][f'{n_clusters}_clusters'] = {
                    'cluster_performance': cluster_performance,
                    'silhouette_score': self._calculate_silhouette_score(scaled_features, cluster_labels)
                }
        
        except Exception as e:
            logger.warning(f"Error in ML analysis: {e}")
        
        return ml_analysis
    
    def _prepare_clustering_features(self, returns: pd.Series) -> pd.DataFrame:
        """Prepare features for clustering analysis"""
        features_data = {}
        
        # Basic return features
        features_data['returns'] = returns
        features_data['abs_returns'] = abs(returns)
        features_data['squared_returns'] = returns ** 2
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features_data[f'mean_{window}d'] = returns.rolling(window).mean()
            features_data[f'std_{window}d'] = returns.rolling(window).std()
            features_data[f'min_{window}d'] = returns.rolling(window).min()
            features_data[f'max_{window}d'] = returns.rolling(window).max()
        
        # Technical indicators
        features_data['momentum_5d'] = returns.rolling(5).sum()
        features_data['momentum_20d'] = returns.rolling(20).sum()
        features_data['mean_reversion'] = returns - returns.rolling(20).mean()
        
        # Volatility features
        features_data['volatility'] = returns.rolling(20).std()
        features_data['volatility_ratio'] = (
            returns.rolling(5).std() / returns.rolling(20).std()
        )
        
        return pd.DataFrame(features_data)
    
    def _identify_inversion_opportunities(self,
                                        strategy_analyses: Dict[str, Any]) -> List[InversionOpportunity]:
        """Identify portfolio-wide inversion opportunities"""
        opportunities = []
        
        for strategy_name, analysis in strategy_analyses.items():
            # Check basic metrics
            basic_metrics = analysis.get('basic_metrics', {})
            
            if basic_metrics.get('total_return', 0) < -0.1:  # 10% loss threshold
                opportunity = InversionOpportunity(
                    strategy_name=strategy_name,
                    opportunity_type="underperforming_strategy",
                    time_period=(None, None),  # Entire period
                    expected_improvement=abs(basic_metrics.get('total_return', 0)),
                    risk_level=self._assess_inversion_risk_level(basic_metrics),
                    confidence_score=0.8,
                    supporting_evidence=[
                        f"Total return: {basic_metrics.get('total_return', 0):.3f}",
                        f"Sharpe ratio: {basic_metrics.get('sharpe_ratio', 0):.3f}"
                    ],
                    recommended_parameters={
                        'inversion_type': 'simple',
                        'apply_immediately': True
                    }
                )
                opportunities.append(opportunity)
            
            # Check patterns
            patterns = analysis.get('performance_patterns', [])
            for pattern in patterns:
                if pattern.confidence > 0.7:
                    opportunity = InversionOpportunity(
                        strategy_name=strategy_name,
                        opportunity_type=f"pattern_based_{pattern.pattern_type}",
                        time_period=(pattern.start_date, pattern.end_date),
                        expected_improvement=pattern.severity_score,
                        risk_level="medium",
                        confidence_score=pattern.confidence,
                        supporting_evidence=[
                            f"Pattern type: {pattern.pattern_type}",
                            f"Duration: {pattern.duration_days} days",
                            f"Severity: {pattern.severity_score:.3f}"
                        ],
                        recommended_parameters={
                            'inversion_type': pattern.suggested_inversion,
                            'pattern_specific': True
                        }
                    )
                    opportunities.append(opportunity)
        
        # Sort by expected improvement
        opportunities.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        return opportunities
    
    def _analyze_portfolio_correlations(self, strategy_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between strategies"""
        corr_matrix = strategy_data.corr()
        
        correlation_analysis = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': [],
            'diversification_opportunities': []
        }
        
        # Find high correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    correlation_analysis['high_correlations'].append({
                        'strategy1': corr_matrix.columns[i],
                        'strategy2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return correlation_analysis
    
    def _detect_portfolio_patterns(self,
                                 strategy_data: pd.DataFrame,
                                 market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Detect portfolio-wide patterns"""
        portfolio_patterns = {}
        
        # Portfolio-wide performance metrics
        portfolio_returns = strategy_data.mean(axis=1)
        portfolio_patterns['portfolio_metrics'] = self._calculate_basic_metrics(portfolio_returns)
        
        # Sector/strategy concentration analysis
        portfolio_patterns['concentration_analysis'] = {
            'strategy_count': len(strategy_data.columns),
            'avg_individual_weight': 1.0 / len(strategy_data.columns),
            'performance_dispersion': strategy_data.std(axis=1).mean()
        }
        
        return portfolio_patterns
    
    def _analyze_inversion_risks(self,
                               strategy_data: pd.DataFrame,
                               opportunities: List[InversionOpportunity]) -> Dict[str, Any]:
        """Analyze risks associated with proposed inversions"""
        risk_analysis = {
            'portfolio_risk_metrics': {},
            'inversion_risks': [],
            'risk_mitigation_suggestions': []
        }
        
        # Calculate portfolio-level risk metrics
        portfolio_returns = strategy_data.mean(axis=1)
        risk_analysis['portfolio_risk_metrics'] = {
            'portfolio_volatility': portfolio_returns.std() * np.sqrt(252),
            'portfolio_var_95': portfolio_returns.quantile(0.05),
            'portfolio_max_drawdown': self._calculate_max_drawdown(portfolio_returns)
        }
        
        # Analyze risks for each opportunity
        for opp in opportunities:
            if opp.strategy_name in strategy_data.columns:
                strategy_returns = strategy_data[opp.strategy_name]
                
                # Estimate post-inversion risk
                inverted_returns = -strategy_returns  # Simple inversion for estimation
                
                risk_assessment = {
                    'opportunity_id': f"{opp.strategy_name}_{opp.opportunity_type}",
                    'pre_inversion_volatility': strategy_returns.std() * np.sqrt(252),
                    'post_inversion_volatility': inverted_returns.std() * np.sqrt(252),
                    'risk_change': 'minimal',  # Will be updated
                    'portfolio_impact': 'low'   # Will be updated
                }
                
                # Assess risk change
                vol_change = abs(risk_assessment['post_inversion_volatility'] - 
                               risk_assessment['pre_inversion_volatility'])
                
                if vol_change > 0.1:
                    risk_assessment['risk_change'] = 'high'
                elif vol_change > 0.05:
                    risk_assessment['risk_change'] = 'moderate'
                
                risk_analysis['inversion_risks'].append(risk_assessment)
        
        return risk_analysis
    
    def _generate_portfolio_recommendations(self, portfolio_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # High-priority opportunities
        high_priority_opps = [
            opp for opp in portfolio_analysis['inversion_opportunities']
            if opp.confidence_score > 0.8 and opp.expected_improvement > 0.1
        ]
        
        if high_priority_opps:
            recommendations.append(
                f"Consider inverting {len(high_priority_opps)} high-confidence strategies "
                f"with expected improvement > 10%"
            )
        
        # Risk management
        high_risk_strategies = len([
            analysis for analysis in portfolio_analysis['strategy_analyses'].values()
            if analysis.get('basic_metrics', {}).get('max_drawdown', 0) > 0.3
        ])
        
        if high_risk_strategies > 0:
            recommendations.append(
                f"Review {high_risk_strategies} strategies with high drawdowns (>30%)"
            )
        
        # Diversification
        high_corr_pairs = len(portfolio_analysis['correlation_analysis']['high_correlations'])
        if high_corr_pairs > 2:
            recommendations.append(
                f"Consider diversification: {high_corr_pairs} highly correlated strategy pairs found"
            )
        
        return recommendations
    
    # Utility methods
    def _clean_returns(self, returns: pd.Series) -> pd.Series:
        """Clean returns data"""
        cleaned = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        return cleaned
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / (volatility + 1e-8)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return gross_profit / (gross_loss + 1e-8)
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        max_dd = self._calculate_max_drawdown(returns)
        return annual_return / (max_dd + 1e-8)
    
    def _group_consecutive_periods(self, mask: pd.Series, min_length: int = 1) -> List[Tuple[int, int]]:
        """Group consecutive True values in boolean mask"""
        periods = []
        start_idx = None
        
        for i, value in enumerate(mask):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                if i - start_idx >= min_length:
                    periods.append((start_idx, i - 1))
                start_idx = None
        
        # Handle case where period extends to end
        if start_idx is not None and len(mask) - start_idx >= min_length:
            periods.append((start_idx, len(mask) - 1))
        
        return periods
    
    def _get_regime_periods(self, regime_mask: pd.Series) -> List[Tuple[int, int]]:
        """Get periods for a specific regime"""
        return self._group_consecutive_periods(regime_mask, min_length=3)
    
    def _assess_inversion_risk_level(self, metrics: Dict[str, float]) -> str:
        """Assess risk level for inversion"""
        max_dd = metrics.get('max_drawdown', 0)
        volatility = metrics.get('volatility', 0)
        
        if max_dd > 0.5 or volatility > 0.4:
            return 'high'
        elif max_dd > 0.3 or volatility > 0.25:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(features, labels)
        except ImportError:
            return 0.0  # Fallback if sklearn not available
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")