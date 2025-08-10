"""
Risk Analyzer for Strategy Inversions

Provides comprehensive risk analysis for strategy inversions including
portfolio risk, individual strategy risk, and post-inversion risk assessment.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for a strategy or portfolio"""
    volatility: float
    max_drawdown: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk
    downside_deviation: float
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None

@dataclass
class InversionRiskAssessment:
    """Risk assessment for a strategy inversion"""
    strategy_name: str
    pre_inversion_risk: RiskMetrics
    post_inversion_risk: RiskMetrics
    risk_change_score: float
    risk_level: str  # 'low', 'medium', 'high'
    risk_factors: List[str]
    recommendations: List[str]
    portfolio_impact: Dict[str, float]

class RiskAnalyzer:
    """
    Comprehensive risk analyzer for strategy inversions
    
    Analyzes risk at multiple levels: individual strategy, portfolio,
    and post-inversion to provide comprehensive risk assessment.
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95,
                 benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize risk analyzer
        
        Args:
            risk_free_rate: Risk-free rate for calculations
            confidence_level: Confidence level for VaR calculations
            benchmark_returns: Benchmark returns for beta calculation
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.benchmark_returns = benchmark_returns
        
        # Risk thresholds
        self.risk_thresholds = {
            'volatility': {'low': 0.15, 'medium': 0.25, 'high': 0.4},
            'max_drawdown': {'low': 0.1, 'medium': 0.2, 'high': 0.3},
            'var_95': {'low': 0.02, 'medium': 0.04, 'high': 0.06}
        }
        
        logger.info("RiskAnalyzer initialized")
    
    def analyze_inversion_risk(self,
                             original_returns: pd.Series,
                             inverted_returns: pd.Series,
                             strategy_name: str,
                             portfolio_returns: Optional[pd.Series] = None) -> InversionRiskAssessment:
        """
        Comprehensive risk analysis for strategy inversion
        
        Args:
            original_returns: Original strategy returns
            inverted_returns: Inverted strategy returns
            strategy_name: Name of the strategy
            portfolio_returns: Portfolio returns for context analysis
            
        Returns:
            Comprehensive risk assessment
        """
        logger.debug(f"Analyzing inversion risk for {strategy_name}")
        
        # Calculate risk metrics for both versions
        pre_inversion_risk = self.calculate_risk_metrics(original_returns, portfolio_returns)
        post_inversion_risk = self.calculate_risk_metrics(inverted_returns, portfolio_returns)
        
        # Calculate risk change score
        risk_change_score = self._calculate_risk_change_score(
            pre_inversion_risk, post_inversion_risk
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(post_inversion_risk, risk_change_score)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            original_returns, inverted_returns, post_inversion_risk
        )
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            pre_inversion_risk, post_inversion_risk, risk_factors, risk_level
        )
        
        # Analyze portfolio impact
        portfolio_impact = self._analyze_portfolio_impact(
            original_returns, inverted_returns, portfolio_returns
        )
        
        return InversionRiskAssessment(
            strategy_name=strategy_name,
            pre_inversion_risk=pre_inversion_risk,
            post_inversion_risk=post_inversion_risk,
            risk_change_score=risk_change_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations,
            portfolio_impact=portfolio_impact
        )
    
    def calculate_risk_metrics(self,
                             returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a return series
        
        Args:
            returns: Return series
            benchmark_returns: Optional benchmark for relative metrics
            
        Returns:
            Comprehensive risk metrics
        """
        if len(returns) == 0:
            return RiskMetrics(0, 0, 0, 0, 0)
        
        # Clean returns
        cleaned_returns = self._clean_returns(returns)
        
        # Basic risk metrics
        volatility = cleaned_returns.std() * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(cleaned_returns)
        var_95 = self._calculate_var(cleaned_returns, self.confidence_level)
        cvar_95 = self._calculate_cvar(cleaned_returns, self.confidence_level)
        downside_deviation = self._calculate_downside_deviation(cleaned_returns)
        
        # Relative metrics (if benchmark available)
        beta = None
        tracking_error = None
        information_ratio = None
        
        benchmark = benchmark_returns if benchmark_returns is not None else self.benchmark_returns
        
        if benchmark is not None and len(benchmark) > 0:
            # Align returns
            aligned_returns, aligned_benchmark = self._align_series(cleaned_returns, benchmark)
            
            if len(aligned_returns) > 20:  # Need sufficient data
                beta = self._calculate_beta(aligned_returns, aligned_benchmark)
                tracking_error = self._calculate_tracking_error(aligned_returns, aligned_benchmark)
                information_ratio = self._calculate_information_ratio(
                    aligned_returns, aligned_benchmark, tracking_error
                )
        
        return RiskMetrics(
            volatility=volatility,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            downside_deviation=downside_deviation,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    def analyze_portfolio_risk_impact(self,
                                    original_portfolio: pd.DataFrame,
                                    inverted_strategies: List[str],
                                    inverted_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Analyze risk impact of inversions on portfolio level
        
        Args:
            original_portfolio: Original portfolio data
            inverted_strategies: List of strategies being inverted
            inverted_returns: Dictionary of inverted return series
            
        Returns:
            Portfolio risk impact analysis
        """
        logger.info("Analyzing portfolio-level risk impact")
        
        # Create modified portfolio with inversions
        modified_portfolio = original_portfolio.copy()
        
        for strategy in inverted_strategies:
            if strategy in modified_portfolio.columns and strategy in inverted_returns:
                aligned_returns = inverted_returns[strategy].reindex(
                    modified_portfolio.index, fill_value=0
                )
                modified_portfolio[strategy] = aligned_returns
        
        # Calculate portfolio returns
        original_portfolio_returns = original_portfolio.mean(axis=1)
        modified_portfolio_returns = modified_portfolio.mean(axis=1)
        
        # Risk metrics for both portfolios
        original_risk = self.calculate_risk_metrics(original_portfolio_returns)
        modified_risk = self.calculate_risk_metrics(modified_portfolio_returns)
        
        # Correlation analysis
        original_corr_matrix = original_portfolio.corr()
        modified_corr_matrix = modified_portfolio.corr()
        
        # Calculate diversification metrics
        original_diversification = self._calculate_diversification_ratio(original_portfolio)
        modified_diversification = self._calculate_diversification_ratio(modified_portfolio)
        
        return {
            'original_portfolio_risk': original_risk,
            'modified_portfolio_risk': modified_risk,
            'risk_improvement': {
                'volatility_change': modified_risk.volatility - original_risk.volatility,
                'max_drawdown_change': modified_risk.max_drawdown - original_risk.max_drawdown,
                'var_change': modified_risk.var_95 - original_risk.var_95
            },
            'diversification_impact': {
                'original_diversification_ratio': original_diversification,
                'modified_diversification_ratio': modified_diversification,
                'diversification_change': modified_diversification - original_diversification
            },
            'correlation_analysis': {
                'avg_correlation_original': np.mean(original_corr_matrix.values[np.triu_indices_from(original_corr_matrix.values, k=1)]),
                'avg_correlation_modified': np.mean(modified_corr_matrix.values[np.triu_indices_from(modified_corr_matrix.values, k=1)]),
                'max_correlation_original': np.max(original_corr_matrix.values[np.triu_indices_from(original_corr_matrix.values, k=1)]),
                'max_correlation_modified': np.max(modified_corr_matrix.values[np.triu_indices_from(modified_corr_matrix.values, k=1)])
            },
            'inverted_strategies_count': len(inverted_strategies),
            'portfolio_strategies_count': len(original_portfolio.columns)
        }
    
    def stress_test_inversions(self,
                             strategy_returns: Dict[str, pd.Series],
                             inversion_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stress test different inversion scenarios
        
        Args:
            strategy_returns: Dictionary of strategy return series
            inversion_scenarios: List of inversion scenarios to test
            
        Returns:
            Stress test results
        """
        logger.info(f"Running stress tests for {len(inversion_scenarios)} scenarios")
        
        stress_test_results = {
            'scenarios': {},
            'worst_case_scenario': None,
            'best_case_scenario': None,
            'risk_summary': {}
        }
        
        scenario_risks = []
        
        for i, scenario in enumerate(inversion_scenarios):
            scenario_name = scenario.get('name', f'scenario_{i}')
            strategies_to_invert = scenario.get('strategies', [])
            
            try:
                # Create scenario portfolio
                scenario_portfolio = pd.DataFrame(strategy_returns)
                
                # Apply inversions
                for strategy in strategies_to_invert:
                    if strategy in scenario_portfolio.columns:
                        scenario_portfolio[strategy] = -scenario_portfolio[strategy]
                
                # Calculate scenario risk
                scenario_returns = scenario_portfolio.mean(axis=1)
                scenario_risk = self.calculate_risk_metrics(scenario_returns)
                
                # Stress factors
                stress_factors = {
                    'market_crash': self._simulate_market_crash(scenario_returns),
                    'volatility_spike': self._simulate_volatility_spike(scenario_returns),
                    'liquidity_crisis': self._simulate_liquidity_crisis(scenario_returns)
                }
                
                scenario_results = {
                    'scenario_name': scenario_name,
                    'strategies_inverted': strategies_to_invert,
                    'baseline_risk': scenario_risk,
                    'stress_factors': stress_factors,
                    'overall_stress_score': np.mean(list(stress_factors.values()))
                }
                
                stress_test_results['scenarios'][scenario_name] = scenario_results
                scenario_risks.append((scenario_name, scenario_risk.volatility))
                
            except Exception as e:
                logger.warning(f"Error in stress test scenario {scenario_name}: {e}")
        
        # Identify best and worst scenarios
        if scenario_risks:
            worst_scenario = max(scenario_risks, key=lambda x: x[1])
            best_scenario = min(scenario_risks, key=lambda x: x[1])
            
            stress_test_results['worst_case_scenario'] = worst_scenario[0]
            stress_test_results['best_case_scenario'] = best_scenario[0]
            
            # Risk summary
            volatilities = [risk[1] for risk in scenario_risks]
            stress_test_results['risk_summary'] = {
                'min_volatility': min(volatilities),
                'max_volatility': max(volatilities),
                'avg_volatility': np.mean(volatilities),
                'volatility_range': max(volatilities) - min(volatilities)
            }
        
        return stress_test_results
    
    # Private methods for calculations
    
    def _calculate_risk_change_score(self,
                                   pre_risk: RiskMetrics,
                                   post_risk: RiskMetrics) -> float:
        """Calculate overall risk change score"""
        
        # Weight different risk components
        weights = {
            'volatility': 0.3,
            'max_drawdown': 0.4,
            'var_95': 0.2,
            'downside_deviation': 0.1
        }
        
        changes = {
            'volatility': (post_risk.volatility - pre_risk.volatility) / (pre_risk.volatility + 1e-8),
            'max_drawdown': (post_risk.max_drawdown - pre_risk.max_drawdown) / (pre_risk.max_drawdown + 1e-8),
            'var_95': (post_risk.var_95 - pre_risk.var_95) / (abs(pre_risk.var_95) + 1e-8),
            'downside_deviation': (post_risk.downside_deviation - pre_risk.downside_deviation) / (pre_risk.downside_deviation + 1e-8)
        }
        
        # Calculate weighted risk change score
        risk_change_score = sum(weights[metric] * change for metric, change in changes.items())
        
        return risk_change_score
    
    def _determine_risk_level(self, risk_metrics: RiskMetrics, risk_change_score: float) -> str:
        """Determine overall risk level"""
        
        risk_scores = []
        
        # Volatility risk
        if risk_metrics.volatility > self.risk_thresholds['volatility']['high']:
            risk_scores.append(3)
        elif risk_metrics.volatility > self.risk_thresholds['volatility']['medium']:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # Drawdown risk
        if risk_metrics.max_drawdown > self.risk_thresholds['max_drawdown']['high']:
            risk_scores.append(3)
        elif risk_metrics.max_drawdown > self.risk_thresholds['max_drawdown']['medium']:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # VaR risk
        if abs(risk_metrics.var_95) > self.risk_thresholds['var_95']['high']:
            risk_scores.append(3)
        elif abs(risk_metrics.var_95) > self.risk_thresholds['var_95']['medium']:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # Risk change score impact
        if risk_change_score > 0.5:  # Significant risk increase
            risk_scores.append(3)
        elif risk_change_score > 0.2:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # Determine overall level
        avg_risk_score = np.mean(risk_scores)
        
        if avg_risk_score >= 2.5:
            return 'high'
        elif avg_risk_score >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _identify_risk_factors(self,
                             original_returns: pd.Series,
                             inverted_returns: pd.Series,
                             post_risk: RiskMetrics) -> List[str]:
        """Identify specific risk factors"""
        
        risk_factors = []
        
        # High volatility
        if post_risk.volatility > 0.3:
            risk_factors.append("High volatility after inversion")
        
        # Large drawdowns
        if post_risk.max_drawdown > 0.25:
            risk_factors.append("Excessive maximum drawdown")
        
        # Negative skewness
        skewness = stats.skew(inverted_returns.dropna())
        if skewness < -1.0:
            risk_factors.append("Highly negative skew in returns")
        
        # High kurtosis (fat tails)
        kurtosis = stats.kurtosis(inverted_returns.dropna())
        if kurtosis > 5.0:
            risk_factors.append("Fat tails in return distribution")
        
        # Increased correlation with market stress
        if len(inverted_returns) > 50:
            stress_periods = inverted_returns < inverted_returns.quantile(0.1)
            if stress_periods.sum() > len(inverted_returns) * 0.2:
                risk_factors.append("High frequency of stress periods")
        
        # Volatility clustering
        vol_series = inverted_returns.rolling(20).std()
        vol_autocorr = vol_series.autocorr(lag=1)
        if vol_autocorr > 0.5:
            risk_factors.append("Volatility clustering present")
        
        return risk_factors
    
    def _generate_risk_recommendations(self,
                                     pre_risk: RiskMetrics,
                                     post_risk: RiskMetrics,
                                     risk_factors: List[str],
                                     risk_level: str) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        # General recommendations based on risk level
        if risk_level == 'high':
            recommendations.append("Consider reducing position size due to high risk")
            recommendations.append("Implement strict stop-loss mechanisms")
            recommendations.append("Monitor closely for early warning signs")
        elif risk_level == 'medium':
            recommendations.append("Regular monitoring recommended")
            recommendations.append("Consider partial position sizing")
        
        # Specific recommendations based on risk factors
        if "High volatility after inversion" in risk_factors:
            recommendations.append("Consider volatility-targeting position sizing")
        
        if "Excessive maximum drawdown" in risk_factors:
            recommendations.append("Implement dynamic drawdown controls")
        
        if "Highly negative skew in returns" in risk_factors:
            recommendations.append("Consider tail risk hedging strategies")
        
        if "Fat tails in return distribution" in risk_factors:
            recommendations.append("Use robust risk measures (VaR, CVaR)")
        
        if "Volatility clustering present" in risk_factors:
            recommendations.append("Adjust position sizing during high volatility periods")
        
        # Improvement recommendations
        if post_risk.max_drawdown < pre_risk.max_drawdown:
            recommendations.append("Drawdown risk improved - maintain current approach")
        
        if post_risk.volatility < pre_risk.volatility:
            recommendations.append("Volatility reduced - consider gradual position increase")
        
        return recommendations
    
    def _analyze_portfolio_impact(self,
                                original_returns: pd.Series,
                                inverted_returns: pd.Series,
                                portfolio_returns: Optional[pd.Series]) -> Dict[str, float]:
        """Analyze impact on portfolio risk"""
        
        portfolio_impact = {
            'correlation_change': 0.0,
            'contribution_to_portfolio_risk': 0.0,
            'diversification_impact': 0.0
        }
        
        if portfolio_returns is None or len(portfolio_returns) == 0:
            return portfolio_impact
        
        # Align series
        aligned_original, aligned_portfolio = self._align_series(original_returns, portfolio_returns)
        aligned_inverted, _ = self._align_series(inverted_returns, portfolio_returns)
        
        if len(aligned_original) > 20:
            # Correlation change
            original_corr = aligned_original.corr(aligned_portfolio)
            inverted_corr = aligned_inverted.corr(aligned_portfolio)
            portfolio_impact['correlation_change'] = inverted_corr - original_corr
            
            # Risk contribution estimate
            portfolio_vol = aligned_portfolio.std()
            strategy_vol = aligned_inverted.std()
            contribution = (inverted_corr * strategy_vol) / portfolio_vol
            portfolio_impact['contribution_to_portfolio_risk'] = contribution
        
        return portfolio_impact
    
    # Risk calculation methods
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return returns.quantile(1 - confidence_level)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk"""
        var_threshold = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var_threshold]
        return tail_returns.mean() if len(tail_returns) > 0 else var_threshold
    
    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0.0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target] - target
        return np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
    
    def _calculate_beta(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate beta"""
        if len(returns) != len(benchmark) or len(returns) < 2:
            return 1.0
        
        covariance = np.cov(returns, benchmark)[0, 1]
        benchmark_variance = np.var(benchmark)
        
        return covariance / (benchmark_variance + 1e-8)
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate tracking error"""
        if len(returns) != len(benchmark):
            return 0.0
        
        excess_returns = returns - benchmark
        return excess_returns.std() * np.sqrt(252)
    
    def _calculate_information_ratio(self,
                                   returns: pd.Series,
                                   benchmark: pd.Series,
                                   tracking_error: float) -> float:
        """Calculate information ratio"""
        if tracking_error == 0:
            return 0.0
        
        excess_returns = returns - benchmark
        excess_return_mean = excess_returns.mean() * 252
        
        return excess_return_mean / tracking_error
    
    def _calculate_diversification_ratio(self, portfolio: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        if len(portfolio.columns) < 2:
            return 1.0
        
        # Individual volatilities
        individual_vols = portfolio.std() * np.sqrt(252)
        
        # Portfolio volatility
        portfolio_returns = portfolio.mean(axis=1)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Weighted average of individual volatilities (equal weights)
        weighted_avg_vol = individual_vols.mean()
        
        return weighted_avg_vol / (portfolio_vol + 1e-8)
    
    # Stress testing methods
    
    def _simulate_market_crash(self, returns: pd.Series) -> float:
        """Simulate market crash scenario"""
        # Apply -20% shock to worst 5% of returns
        crash_threshold = returns.quantile(0.05)
        crashed_returns = returns.copy()
        crash_mask = returns <= crash_threshold
        crashed_returns[crash_mask] *= 1.2  # 20% worse
        
        # Calculate risk increase
        original_var = self._calculate_var(returns, 0.95)
        crashed_var = self._calculate_var(crashed_returns, 0.95)
        
        return abs(crashed_var - original_var) / abs(original_var + 1e-8)
    
    def _simulate_volatility_spike(self, returns: pd.Series) -> float:
        """Simulate volatility spike scenario"""
        # Double volatility for 20% of periods
        vol_spike_returns = returns.copy()
        spike_indices = np.random.choice(len(returns), size=int(len(returns) * 0.2), replace=False)
        vol_spike_returns.iloc[spike_indices] *= 2
        
        # Calculate volatility increase
        original_vol = returns.std()
        spiked_vol = vol_spike_returns.std()
        
        return (spiked_vol - original_vol) / (original_vol + 1e-8)
    
    def _simulate_liquidity_crisis(self, returns: pd.Series) -> float:
        """Simulate liquidity crisis scenario"""
        # Increase negative skewness and kurtosis
        crisis_returns = returns.copy()
        
        # Make worst returns even worse
        worst_percentile = returns.quantile(0.1)
        crisis_mask = returns <= worst_percentile
        crisis_returns[crisis_mask] *= 1.5
        
        # Calculate distribution change
        original_skew = stats.skew(returns.dropna())
        crisis_skew = stats.skew(crisis_returns.dropna())
        
        return abs(crisis_skew - original_skew)
    
    # Utility methods
    
    def _clean_returns(self, returns: pd.Series) -> pd.Series:
        """Clean returns data"""
        cleaned = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        return cleaned
    
    def _align_series(self, series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two series by index"""
        common_index = series1.index.intersection(series2.index)
        return series1.loc[common_index], series2.loc[common_index]