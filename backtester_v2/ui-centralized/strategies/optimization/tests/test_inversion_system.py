"""
Test Suite for Strategy Inversion System

Tests strategy inversion logic, pattern detection, risk analysis,
and the complete inversion workflow.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from scipy import stats

from ..inversion.strategy_inverter import StrategyInverter, InversionType, InversionResult
from ..inversion.pattern_detector import PatternDetector, PatternType, DetectedPattern
from ..inversion.risk_analyzer import RiskAnalyzer, RiskMetrics, InversionRiskAssessment
from ..inversion.inversion_analyzer import InversionAnalyzer, InversionRecommendation
from ..inversion.inversion_engine import InversionEngine, InversionEngineResult


class TestStrategyInverter:
    """Test cases for StrategyInverter"""
    
    def test_strategy_inverter_initialization(self):
        """Test strategy inverter initialization"""
        inverter = StrategyInverter(
            inversion_types=[InversionType.SIMPLE, InversionType.ULTA],
            threshold_configs={'ulta_threshold': 0.1}
        )
        
        assert InversionType.SIMPLE in inverter.inversion_types
        assert InversionType.ULTA in inverter.inversion_types
        assert inverter.threshold_configs['ulta_threshold'] == 0.1
    
    def test_simple_inversion(self, sample_strategy_returns):
        """Test simple strategy inversion"""
        inverter = StrategyInverter()
        
        original_returns = sample_strategy_returns['strategy_1']
        
        result = inverter.invert_strategy(
            returns=original_returns,
            inversion_type=InversionType.SIMPLE,
            strategy_name='test_strategy'
        )
        
        assert isinstance(result, InversionResult)
        assert result.inversion_type == InversionType.SIMPLE
        assert result.strategy_name == 'test_strategy'
        assert len(result.inverted_returns) == len(original_returns)
        
        # Simple inversion should be exact opposite
        np.testing.assert_array_almost_equal(
            result.inverted_returns.values,
            -original_returns.values
        )
    
    def test_ulta_inversion(self, sample_strategy_returns):
        """Test ULTA (Underperforming Long Term Average) inversion"""
        inverter = StrategyInverter(threshold_configs={'ulta_threshold': 0.05})
        
        original_returns = sample_strategy_returns['strategy_1']
        
        result = inverter.invert_strategy(
            returns=original_returns,
            inversion_type=InversionType.ULTA,
            strategy_name='ulta_test'
        )
        
        assert isinstance(result, InversionResult)
        assert result.inversion_type == InversionType.ULTA
        
        # ULTA inversion should only invert periods below threshold
        long_term_avg = original_returns.expanding().mean()
        underperforming_mask = original_returns < (long_term_avg - 0.05)
        
        # Check that only underperforming periods were inverted
        expected_inverted = original_returns.copy()
        expected_inverted[underperforming_mask] *= -1
        
        pd.testing.assert_series_equal(result.inverted_returns, expected_inverted)
    
    def test_conditional_inversion(self, sample_strategy_returns):
        """Test conditional inversion based on market conditions"""
        inverter = StrategyInverter()
        
        original_returns = sample_strategy_returns['strategy_1']
        
        # Create mock market condition (high volatility periods)
        market_vol = original_returns.rolling(20).std()
        high_vol_threshold = market_vol.quantile(0.8)
        
        condition_func = lambda idx: market_vol.iloc[idx] > high_vol_threshold
        
        result = inverter.invert_strategy(
            returns=original_returns,
            inversion_type=InversionType.CONDITIONAL,
            strategy_name='conditional_test',
            condition_function=condition_func
        )
        
        assert isinstance(result, InversionResult)
        assert result.inversion_type == InversionType.CONDITIONAL
        
        # Should only invert during high volatility periods
        assert len(result.inverted_returns) == len(original_returns)
    
    def test_rolling_inversion(self, sample_strategy_returns):
        """Test rolling window inversion"""
        inverter = StrategyInverter(threshold_configs={'rolling_window': 30})
        
        original_returns = sample_strategy_returns['strategy_1']
        
        result = inverter.invert_strategy(
            returns=original_returns,
            inversion_type=InversionType.ROLLING,
            strategy_name='rolling_test'
        )
        
        assert isinstance(result, InversionResult)
        assert result.inversion_type == InversionType.ROLLING
        
        # Rolling inversion should adapt over time
        assert len(result.inverted_returns) == len(original_returns)
        
        # Check that inversion decision changes over time
        inversion_decisions = result.metadata.get('inversion_decisions', [])
        assert len(set(inversion_decisions)) > 1  # Should have different decisions
    
    def test_risk_adjusted_inversion(self, sample_strategy_returns):
        """Test risk-adjusted inversion"""
        inverter = StrategyInverter()
        
        original_returns = sample_strategy_returns['strategy_1']
        
        result = inverter.invert_strategy(
            returns=original_returns,
            inversion_type=InversionType.RISK_ADJUSTED,
            strategy_name='risk_adjusted_test'
        )
        
        assert isinstance(result, InversionResult)
        assert result.inversion_type == InversionType.RISK_ADJUSTED
        
        # Risk-adjusted inversion should consider volatility
        assert 'risk_metrics' in result.metadata
        assert 'volatility_analysis' in result.metadata
    
    def test_smart_inversion(self, sample_strategy_returns):
        """Test smart inversion with ML-based decision making"""
        inverter = StrategyInverter()
        
        original_returns = sample_strategy_returns['strategy_1']
        
        # Provide market data for smart inversion
        market_data = pd.DataFrame({
            'market_returns': np.random.normal(0.001, 0.02, len(original_returns)),
            'volume': np.random.randint(1000000, 10000000, len(original_returns))
        }, index=original_returns.index)
        
        result = inverter.invert_strategy(
            returns=original_returns,
            inversion_type=InversionType.SMART,
            strategy_name='smart_test',
            market_data=market_data
        )
        
        assert isinstance(result, InversionResult)
        assert result.inversion_type == InversionType.SMART
        
        # Smart inversion should use ML features
        assert 'ml_features' in result.metadata
        assert 'prediction_confidence' in result.metadata
    
    def test_inversion_performance_metrics(self, sample_strategy_returns):
        """Test inversion performance metrics calculation"""
        inverter = StrategyInverter()
        
        original_returns = sample_strategy_returns['strategy_1']
        
        result = inverter.invert_strategy(
            returns=original_returns,
            inversion_type=InversionType.SIMPLE,
            strategy_name='metrics_test'
        )
        
        # Should calculate performance metrics
        assert hasattr(result, 'performance_metrics')
        metrics = result.performance_metrics
        
        assert 'original_sharpe' in metrics
        assert 'inverted_sharpe' in metrics
        assert 'improvement_ratio' in metrics
        assert 'max_drawdown_improvement' in metrics
    
    def test_batch_inversion(self, sample_strategy_returns):
        """Test batch inversion of multiple strategies"""
        inverter = StrategyInverter()
        
        # Create portfolio DataFrame
        portfolio_returns = pd.DataFrame(sample_strategy_returns)
        
        results = inverter.batch_invert_strategies(
            portfolio_returns=portfolio_returns,
            inversion_type=InversionType.SIMPLE
        )
        
        assert isinstance(results, dict)
        assert len(results) == len(sample_strategy_returns)
        
        for strategy_name, result in results.items():
            assert isinstance(result, InversionResult)
            assert result.strategy_name == strategy_name


class TestPatternDetector:
    """Test cases for PatternDetector"""
    
    def test_pattern_detector_initialization(self):
        """Test pattern detector initialization"""
        detector = PatternDetector(
            pattern_types=[PatternType.CYCLICAL, PatternType.TREND],
            detection_params={'min_cycle_length': 20}
        )
        
        assert PatternType.CYCLICAL in detector.pattern_types
        assert PatternType.TREND in detector.pattern_types
        assert detector.detection_params['min_cycle_length'] == 20
    
    def test_cyclical_pattern_detection(self):
        """Test cyclical pattern detection"""
        detector = PatternDetector()
        
        # Create synthetic cyclical data
        t = np.arange(252)
        cyclical_returns = 0.02 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.01, 252)
        returns_series = pd.Series(cyclical_returns)
        
        patterns = detector.detect_patterns(returns_series)
        
        # Should detect cyclical pattern
        cyclical_patterns = [p for p in patterns if p.pattern_type == PatternType.CYCLICAL]
        assert len(cyclical_patterns) > 0
        
        for pattern in cyclical_patterns:
            assert hasattr(pattern, 'cycle_length')
            assert hasattr(pattern, 'amplitude')
            assert hasattr(pattern, 'phase')
    
    def test_trend_pattern_detection(self):
        """Test trend pattern detection"""
        detector = PatternDetector()
        
        # Create data with clear trend
        trend_returns = np.linspace(-0.001, 0.001, 252) + np.random.normal(0, 0.005, 252)
        returns_series = pd.Series(trend_returns)
        
        patterns = detector.detect_patterns(returns_series)
        
        # Should detect trend pattern
        trend_patterns = [p for p in patterns if p.pattern_type == PatternType.TREND]
        assert len(trend_patterns) > 0
        
        for pattern in trend_patterns:
            assert hasattr(pattern, 'trend_direction')
            assert hasattr(pattern, 'trend_strength')
    
    def test_volatility_clustering_detection(self, sample_strategy_returns):
        """Test volatility clustering pattern detection"""
        detector = PatternDetector()
        
        returns = sample_strategy_returns['strategy_1']
        patterns = detector.detect_patterns(returns)
        
        # Look for volatility clustering
        vol_patterns = [p for p in patterns if p.pattern_type == PatternType.VOLATILITY_CLUSTERING]
        
        if vol_patterns:
            for pattern in vol_patterns:
                assert hasattr(pattern, 'clustering_strength')
                assert hasattr(pattern, 'persistence')
    
    def test_change_point_detection(self):
        """Test change point detection"""
        detector = PatternDetector()
        
        # Create data with change point
        before_change = np.random.normal(0.001, 0.01, 100)
        after_change = np.random.normal(-0.002, 0.02, 152)
        returns_with_change = np.concatenate([before_change, after_change])
        returns_series = pd.Series(returns_with_change)
        
        patterns = detector.detect_patterns(returns_series)
        
        # Should detect change point
        change_patterns = [p for p in patterns if p.pattern_type == PatternType.CHANGE_POINT]
        
        if change_patterns:
            for pattern in change_patterns:
                assert hasattr(pattern, 'change_location')
                assert hasattr(pattern, 'change_magnitude')
    
    def test_momentum_pattern_detection(self, sample_strategy_returns):
        """Test momentum pattern detection"""
        detector = PatternDetector()
        
        returns = sample_strategy_returns['strategy_1']
        patterns = detector.detect_patterns(returns)
        
        # Look for momentum patterns
        momentum_patterns = [p for p in patterns if p.pattern_type == PatternType.MOMENTUM]
        
        if momentum_patterns:
            for pattern in momentum_patterns:
                assert hasattr(pattern, 'momentum_strength')
                assert hasattr(pattern, 'persistence_length')
    
    def test_mean_reversion_detection(self, sample_strategy_returns):
        """Test mean reversion pattern detection"""
        detector = PatternDetector()
        
        returns = sample_strategy_returns['strategy_1']
        patterns = detector.detect_patterns(returns)
        
        # Look for mean reversion patterns
        reversion_patterns = [p for p in patterns if p.pattern_type == PatternType.MEAN_REVERSION]
        
        if reversion_patterns:
            for pattern in reversion_patterns:
                assert hasattr(pattern, 'reversion_speed')
                assert hasattr(pattern, 'equilibrium_level')
    
    def test_outlier_detection(self, sample_strategy_returns):
        """Test outlier pattern detection"""
        detector = PatternDetector()
        
        returns = sample_strategy_returns['strategy_1']
        
        # Add some outliers
        outlier_returns = returns.copy()
        outlier_returns.iloc[50] = 0.1  # Large positive outlier
        outlier_returns.iloc[100] = -0.08  # Large negative outlier
        
        patterns = detector.detect_patterns(outlier_returns)
        
        # Should detect outliers
        outlier_patterns = [p for p in patterns if p.pattern_type == PatternType.OUTLIERS]
        assert len(outlier_patterns) > 0
        
        for pattern in outlier_patterns:
            assert hasattr(pattern, 'outlier_indices')
            assert hasattr(pattern, 'outlier_values')


class TestRiskAnalyzer:
    """Test cases for RiskAnalyzer"""
    
    def test_risk_analyzer_initialization(self):
        """Test risk analyzer initialization"""
        analyzer = RiskAnalyzer(
            risk_free_rate=0.03,
            confidence_level=0.99
        )
        
        assert analyzer.risk_free_rate == 0.03
        assert analyzer.confidence_level == 0.99
    
    def test_risk_metrics_calculation(self, sample_strategy_returns):
        """Test comprehensive risk metrics calculation"""
        analyzer = RiskAnalyzer()
        
        returns = sample_strategy_returns['strategy_1']
        metrics = analyzer.calculate_risk_metrics(returns)
        
        assert isinstance(metrics, RiskMetrics)
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'var_95')
        assert hasattr(metrics, 'cvar_95')
        assert hasattr(metrics, 'downside_deviation')
        
        # Risk metrics should be reasonable
        assert metrics.volatility > 0
        assert 0 <= metrics.max_drawdown <= 1
        assert metrics.var_95 < 0  # VaR should be negative
        assert metrics.cvar_95 <= metrics.var_95  # CVaR should be worse than VaR
    
    def test_inversion_risk_assessment(self, sample_strategy_returns):
        """Test inversion risk assessment"""
        analyzer = RiskAnalyzer()
        
        original_returns = sample_strategy_returns['strategy_1']
        inverted_returns = -original_returns  # Simple inversion
        
        assessment = analyzer.analyze_inversion_risk(
            original_returns=original_returns,
            inverted_returns=inverted_returns,
            strategy_name='test_strategy'
        )
        
        assert isinstance(assessment, InversionRiskAssessment)
        assert assessment.strategy_name == 'test_strategy'
        assert hasattr(assessment, 'pre_inversion_risk')
        assert hasattr(assessment, 'post_inversion_risk')
        assert hasattr(assessment, 'risk_change_score')
        assert hasattr(assessment, 'risk_level')
        assert hasattr(assessment, 'risk_factors')
        assert hasattr(assessment, 'recommendations')
    
    def test_portfolio_risk_impact_analysis(self, sample_strategy_returns):
        """Test portfolio-level risk impact analysis"""
        analyzer = RiskAnalyzer()
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        inverted_strategies = ['strategy_1']
        inverted_returns = {'strategy_1': -sample_strategy_returns['strategy_1']}
        
        impact_analysis = analyzer.analyze_portfolio_risk_impact(
            original_portfolio=portfolio_df,
            inverted_strategies=inverted_strategies,
            inverted_returns=inverted_returns
        )
        
        assert isinstance(impact_analysis, dict)
        assert 'original_portfolio_risk' in impact_analysis
        assert 'modified_portfolio_risk' in impact_analysis
        assert 'risk_improvement' in impact_analysis
        assert 'diversification_impact' in impact_analysis
        assert 'correlation_analysis' in impact_analysis
    
    def test_stress_testing(self, sample_strategy_returns):
        """Test stress testing of inversions"""
        analyzer = RiskAnalyzer()
        
        scenarios = [
            {'name': 'invert_all', 'strategies': list(sample_strategy_returns.keys())},
            {'name': 'invert_one', 'strategies': ['strategy_1']},
            {'name': 'no_inversion', 'strategies': []}
        ]
        
        stress_results = analyzer.stress_test_inversions(
            strategy_returns=sample_strategy_returns,
            inversion_scenarios=scenarios
        )
        
        assert isinstance(stress_results, dict)
        assert 'scenarios' in stress_results
        assert 'worst_case_scenario' in stress_results
        assert 'best_case_scenario' in stress_results
        assert 'risk_summary' in stress_results
        
        # Should have tested all scenarios
        assert len(stress_results['scenarios']) == 3
    
    def test_var_and_cvar_calculation(self):
        """Test VaR and CVaR calculation accuracy"""
        analyzer = RiskAnalyzer(confidence_level=0.95)
        
        # Generate known distribution
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
        var_95 = analyzer._calculate_var(returns, 0.95)
        cvar_95 = analyzer._calculate_cvar(returns, 0.95)
        
        # VaR should be approximately at 5th percentile
        expected_var = returns.quantile(0.05)
        assert abs(var_95 - expected_var) < 1e-10
        
        # CVaR should be worse (more negative) than VaR
        assert cvar_95 <= var_95
    
    def test_beta_calculation_with_benchmark(self, sample_strategy_returns, sample_market_data):
        """Test beta calculation with benchmark"""
        analyzer = RiskAnalyzer(benchmark_returns=sample_market_data['returns'])
        
        strategy_returns = sample_strategy_returns['strategy_1']
        
        # Align series
        aligned_strategy, aligned_benchmark = analyzer._align_series(
            strategy_returns, sample_market_data['returns']
        )
        
        beta = analyzer._calculate_beta(aligned_strategy, aligned_benchmark)
        
        # Beta should be a reasonable number
        assert isinstance(beta, float)
        assert -5 <= beta <= 5  # Reasonable range for beta


class TestInversionAnalyzer:
    """Test cases for InversionAnalyzer"""
    
    def test_inversion_analyzer_initialization(self):
        """Test inversion analyzer initialization"""
        analyzer = InversionAnalyzer(
            min_observation_period=60,
            confidence_threshold=0.8
        )
        
        assert analyzer.min_observation_period == 60
        assert analyzer.confidence_threshold == 0.8
    
    def test_strategy_prioritization(self, sample_strategy_returns):
        """Test strategy prioritization for inversion"""
        analyzer = InversionAnalyzer()
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        
        prioritization = analyzer.prioritize_strategies_for_inversion(portfolio_df)
        
        assert isinstance(prioritization, list)
        assert len(prioritization) <= len(sample_strategy_returns)
        
        # Each item should have strategy name and priority score
        for item in prioritization:
            assert 'strategy_name' in item
            assert 'priority_score' in item
            assert 'reasoning' in item
    
    def test_inversion_recommendation(self, sample_strategy_returns):
        """Test inversion recommendation generation"""
        analyzer = InversionAnalyzer()
        
        strategy_returns = sample_strategy_returns['strategy_1']
        
        recommendation = analyzer.generate_inversion_recommendation(
            strategy_returns=strategy_returns,
            strategy_name='test_strategy'
        )
        
        assert isinstance(recommendation, InversionRecommendation)
        assert recommendation.strategy_name == 'test_strategy'
        assert hasattr(recommendation, 'recommended_type')
        assert hasattr(recommendation, 'confidence_score')
        assert hasattr(recommendation, 'expected_improvement')
        assert hasattr(recommendation, 'reasoning')
    
    def test_portfolio_wide_analysis(self, sample_strategy_returns):
        """Test portfolio-wide inversion analysis"""
        analyzer = InversionAnalyzer()
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        
        analysis = analyzer.analyze_strategy_portfolio(
            strategy_data=portfolio_df,
            strategy_columns=list(portfolio_df.columns)
        )
        
        assert isinstance(analysis, dict)
        assert 'portfolio_summary' in analysis
        assert 'individual_analyses' in analysis
        assert 'correlation_matrix' in analysis
        assert 'diversification_metrics' in analysis
        assert 'inversion_recommendations' in analysis


class TestInversionEngine:
    """Test cases for InversionEngine"""
    
    def test_inversion_engine_initialization(self):
        """Test inversion engine initialization"""
        engine = InversionEngine(
            supported_types=[InversionType.SIMPLE, InversionType.ULTA],
            enable_risk_analysis=True,
            enable_pattern_detection=True
        )
        
        assert InversionType.SIMPLE in engine.supported_types
        assert InversionType.ULTA in engine.supported_types
        assert engine.enable_risk_analysis is True
        assert engine.enable_pattern_detection is True
    
    def test_complete_inversion_workflow(self, sample_strategy_returns):
        """Test complete inversion analysis workflow"""
        engine = InversionEngine()
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        strategy_columns = list(portfolio_df.columns)
        
        result = engine.analyze_and_invert_portfolio(
            strategy_data=portfolio_df,
            strategy_columns=strategy_columns
        )
        
        assert isinstance(result, InversionEngineResult)
        assert hasattr(result, 'analysis_summary')
        assert hasattr(result, 'prioritization_results')
        assert hasattr(result, 'risk_assessments')
        assert hasattr(result, 'inversion_results')
        assert hasattr(result, 'performance_analysis')
        assert hasattr(result, 'recommendations')
        
        # Should have analyzed all strategies
        assert len(result.inversion_results) <= len(strategy_columns)
    
    def test_inversion_with_market_data(self, sample_strategy_returns, sample_market_data):
        """Test inversion analysis with market data"""
        engine = InversionEngine(enable_smart_inversion=True)
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        
        result = engine.analyze_and_invert_portfolio(
            strategy_data=portfolio_df,
            strategy_columns=list(portfolio_df.columns),
            market_data=sample_market_data
        )
        
        # Should incorporate market data in analysis
        assert isinstance(result, InversionEngineResult)
        assert 'market_regime_analysis' in result.analysis_summary
    
    def test_selective_inversion(self, sample_strategy_returns):
        """Test selective strategy inversion"""
        engine = InversionEngine()
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        selected_strategies = ['strategy_1', 'strategy_2']  # Only invert subset
        
        result = engine.analyze_and_invert_portfolio(
            strategy_data=portfolio_df,
            strategy_columns=selected_strategies
        )
        
        # Should only analyze selected strategies
        assert len(result.inversion_results) == len(selected_strategies)
        
        for strategy_name in selected_strategies:
            assert strategy_name in result.inversion_results
    
    def test_inversion_performance_comparison(self, sample_strategy_returns):
        """Test performance comparison of inversions"""
        engine = InversionEngine()
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        
        result = engine.analyze_and_invert_portfolio(
            strategy_data=portfolio_df,
            strategy_columns=list(portfolio_df.columns)
        )
        
        performance_analysis = result.performance_analysis
        
        assert 'portfolio_improvement' in performance_analysis
        assert 'individual_improvements' in performance_analysis
        assert 'risk_adjusted_metrics' in performance_analysis
        
        # Should provide clear performance metrics
        for strategy_name in portfolio_df.columns:
            if strategy_name in result.inversion_results:
                assert strategy_name in performance_analysis['individual_improvements']


class TestInversionSystemIntegration:
    """Integration tests for the complete inversion system"""
    
    def test_end_to_end_inversion_pipeline(self, sample_strategy_returns):
        """Test complete end-to-end inversion pipeline"""
        # Initialize all components
        engine = InversionEngine(
            supported_types=[InversionType.SIMPLE, InversionType.ULTA, InversionType.SMART],
            enable_risk_analysis=True,
            enable_pattern_detection=True
        )
        
        portfolio_df = pd.DataFrame(sample_strategy_returns)
        
        # Run complete analysis
        result = engine.analyze_and_invert_portfolio(
            strategy_data=portfolio_df,
            strategy_columns=list(portfolio_df.columns)
        )
        
        # Verify all phases completed
        assert result.analysis_summary['phase_1_completed'] is True
        assert result.analysis_summary['phase_2_completed'] is True
        assert result.analysis_summary['phase_3_completed'] is True
        assert result.analysis_summary['phase_4_completed'] is True
        assert result.analysis_summary['phase_5_completed'] is True
        assert result.analysis_summary['phase_6_completed'] is True
        assert result.analysis_summary['phase_7_completed'] is True
        
        # Should have comprehensive results
        assert len(result.prioritization_results) > 0
        assert len(result.risk_assessments) > 0
        assert len(result.inversion_results) > 0
        assert len(result.recommendations) > 0
    
    def test_inversion_with_different_types(self, sample_strategy_returns):
        """Test inversion with different inversion types"""
        engine = InversionEngine()
        
        strategy_returns = sample_strategy_returns['strategy_1']
        
        # Test different inversion types
        inversion_types = [
            InversionType.SIMPLE,
            InversionType.ULTA,
            InversionType.CONDITIONAL,
            InversionType.ROLLING,
            InversionType.RISK_ADJUSTED,
            InversionType.SMART
        ]
        
        results = {}
        for inv_type in inversion_types:
            try:
                inverter = StrategyInverter()
                result = inverter.invert_strategy(
                    returns=strategy_returns,
                    inversion_type=inv_type,
                    strategy_name=f'test_{inv_type.value}'
                )
                results[inv_type] = result
            except Exception as e:
                # Some types might require additional parameters
                pass
        
        # Should successfully invert with multiple types
        assert len(results) >= 3  # At least simple, ULTA, and one other
        
        # All results should be valid
        for inv_type, result in results.items():
            assert isinstance(result, InversionResult)
            assert result.inversion_type == inv_type
    
    def test_performance_comparison_across_types(self, sample_strategy_returns):
        """Test performance comparison across inversion types"""
        engine = InversionEngine()
        
        strategy_returns = sample_strategy_returns['strategy_1']
        
        # Compare multiple inversion types
        comparison_results = engine.compare_inversion_types(
            strategy_returns=strategy_returns,
            strategy_name='comparison_test',
            types_to_compare=[InversionType.SIMPLE, InversionType.ULTA, InversionType.ROLLING]
        )
        
        assert isinstance(comparison_results, dict)
        assert 'performance_comparison' in comparison_results
        assert 'best_performing_type' in comparison_results
        assert 'detailed_results' in comparison_results
        
        # Should rank different inversion types
        performance_comparison = comparison_results['performance_comparison']
        assert len(performance_comparison) == 3


class TestInversionSystemEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_inversion_with_insufficient_data(self):
        """Test inversion with insufficient data"""
        inverter = StrategyInverter()
        
        # Very short return series
        short_returns = pd.Series([0.01, -0.02, 0.015])
        
        result = inverter.invert_strategy(
            returns=short_returns,
            inversion_type=InversionType.SIMPLE,
            strategy_name='short_data_test'
        )
        
        # Should handle gracefully
        assert isinstance(result, InversionResult)
        assert len(result.inverted_returns) == len(short_returns)
    
    def test_inversion_with_extreme_returns(self):
        """Test inversion with extreme return values"""
        inverter = StrategyInverter()
        
        # Create returns with extreme values
        extreme_returns = pd.Series([0.5, -0.8, 0.3, -0.9, 0.1])
        
        result = inverter.invert_strategy(
            returns=extreme_returns,
            inversion_type=InversionType.SIMPLE,
            strategy_name='extreme_test'
        )
        
        # Should handle extreme values
        assert isinstance(result, InversionResult)
        assert not np.any(np.isnan(result.inverted_returns))
        assert not np.any(np.isinf(result.inverted_returns))
    
    def test_inversion_with_missing_data(self):
        """Test inversion with missing data points"""
        inverter = StrategyInverter()
        
        # Create returns with NaN values
        returns_with_nan = pd.Series([0.01, np.nan, -0.02, 0.015, np.nan])
        
        result = inverter.invert_strategy(
            returns=returns_with_nan,
            inversion_type=InversionType.SIMPLE,
            strategy_name='missing_data_test'
        )
        
        # Should handle missing data appropriately
        assert isinstance(result, InversionResult)
        assert len(result.inverted_returns) == len(returns_with_nan)
    
    def test_risk_analysis_with_constant_returns(self):
        """Test risk analysis with constant return series"""
        analyzer = RiskAnalyzer()
        
        # Constant returns (no volatility)
        constant_returns = pd.Series([0.01] * 100)
        
        metrics = analyzer.calculate_risk_metrics(constant_returns)
        
        # Should handle zero volatility case
        assert isinstance(metrics, RiskMetrics)
        assert metrics.volatility == 0.0
        assert metrics.max_drawdown == 0.0
    
    def test_pattern_detection_with_noise(self):
        """Test pattern detection with high noise"""
        detector = PatternDetector()
        
        # High noise, low signal data
        noise_returns = pd.Series(np.random.normal(0, 0.1, 252))
        
        patterns = detector.detect_patterns(noise_returns)
        
        # Should handle noisy data gracefully
        assert isinstance(patterns, list)
        # May or may not detect patterns in pure noise