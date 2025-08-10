"""
Test Suite for Robustness Framework

Tests cross-validation, sensitivity analysis, dimension testing,
robust estimation, and the overall robust optimizer framework.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from ..robustness.robust_optimizer import RobustOptimizer, RobustnessResult
from ..robustness.cross_validation import TimeSeriesCrossValidator, ValidationResult
from ..robustness.sensitivity_analysis import SensitivityAnalyzer, SensitivityResult
from ..robustness.dimension_testing import DimensionTester, DimensionTestResult
from ..robustness.robust_estimation import RobustEstimator, RobustMetrics
from ..base.base_optimizer import BaseOptimizer, OptimizationResult


class MockBaseOptimizer(BaseOptimizer):
    """Mock base optimizer for testing"""
    
    def __init__(self, param_space, objective_function, mock_result_value=1.0, **kwargs):
        super().__init__(param_space, objective_function, **kwargs)
        self.mock_result_value = mock_result_value
        self.call_count = 0
    
    def optimize(self, n_iterations=100, **kwargs):
        self.call_count += 1
        
        # Generate random parameters within bounds
        best_params = {}
        for param, bounds in self.param_space.items():
            best_params[param] = np.random.uniform(bounds[0], bounds[1])
        
        return OptimizationResult(
            best_parameters=best_params,
            best_objective_value=self.mock_result_value + np.random.normal(0, 0.1),
            iterations=min(n_iterations, 50),
            convergence_status='converged'
        )


class TestRobustOptimizer:
    """Test cases for RobustOptimizer"""
    
    def test_robust_optimizer_initialization(self, simple_param_space, sphere_function):
        """Test robust optimizer initialization"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=3,
            noise_levels=[0.01, 0.05],
            enable_sensitivity_analysis=True
        )
        
        assert robust_optimizer.base_optimizer == base_optimizer
        assert robust_optimizer.cv_folds == 3
        assert robust_optimizer.noise_levels == [0.01, 0.05]
        assert robust_optimizer.enable_sensitivity_analysis is True
    
    def test_robust_optimization_basic(self, simple_param_space, sphere_function):
        """Test basic robust optimization workflow"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function, mock_result_value=2.5)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=3,
            noise_levels=[0.01]
        )
        
        result = robust_optimizer.optimize(n_iterations=20)
        
        # Should return RobustnessResult
        assert isinstance(result, RobustnessResult)
        assert hasattr(result, 'best_parameters')
        assert hasattr(result, 'best_objective_value')
        assert hasattr(result, 'robustness_metrics')
        assert hasattr(result, 'cv_results')
        
        # Base optimizer should have been called multiple times (CV + noise tests)
        assert base_optimizer.call_count > 1
    
    def test_cross_validation_integration(self, simple_param_space, sphere_function):
        """Test cross-validation integration"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=5,
            noise_levels=[]  # Disable noise testing for this test
        )
        
        result = robust_optimizer.optimize(n_iterations=10)
        
        # Should have CV results
        assert result.cv_results is not None
        assert hasattr(result.cv_results, 'fold_scores')
        assert hasattr(result.cv_results, 'mean_score')
        assert hasattr(result.cv_results, 'std_score')
        
        # Should have performed 5-fold CV
        assert len(result.cv_results.fold_scores) == 5
    
    def test_noise_robustness_testing(self, simple_param_space, sphere_function):
        """Test noise robustness testing"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=0,  # Disable CV for this test
            noise_levels=[0.01, 0.05, 0.1]
        )
        
        result = robust_optimizer.optimize(n_iterations=10)
        
        # Should have noise test results
        assert hasattr(result, 'noise_analysis')
        assert result.noise_analysis is not None
        
        # Should test multiple noise levels
        assert len(result.noise_analysis) == 3
    
    def test_sensitivity_analysis_integration(self, simple_param_space, sphere_function):
        """Test sensitivity analysis integration"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            enable_sensitivity_analysis=True,
            cv_folds=0,  # Disable CV
            noise_levels=[]  # Disable noise testing
        )
        
        result = robust_optimizer.optimize(n_iterations=10)
        
        # Should have sensitivity analysis results
        assert hasattr(result, 'sensitivity_results')
        assert result.sensitivity_results is not None
    
    def test_dimension_testing_integration(self, complex_param_space, sphere_function):
        """Test dimension testing integration"""
        base_optimizer = MockBaseOptimizer(complex_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            enable_dimension_testing=True,
            cv_folds=0,
            noise_levels=[]
        )
        
        result = robust_optimizer.optimize(n_iterations=10)
        
        # Should have dimension test results
        assert hasattr(result, 'dimension_analysis')
        assert result.dimension_analysis is not None
    
    def test_robust_metrics_calculation(self, simple_param_space, sphere_function):
        """Test robust metrics calculation"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=3,
            noise_levels=[0.01, 0.05]
        )
        
        result = robust_optimizer.optimize(n_iterations=10)
        
        metrics = result.robustness_metrics
        
        # Should have robustness metrics
        assert hasattr(metrics, 'stability_score')
        assert hasattr(metrics, 'consistency_score')
        assert hasattr(metrics, 'reliability_score')
        assert hasattr(metrics, 'overall_robustness')
        
        # Scores should be between 0 and 1
        assert 0 <= metrics.stability_score <= 1
        assert 0 <= metrics.consistency_score <= 1
        assert 0 <= metrics.reliability_score <= 1
        assert 0 <= metrics.overall_robustness <= 1


class TestTimeSeriesCrossValidator:
    """Test cases for TimeSeriesCrossValidator"""
    
    def test_cv_initialization(self):
        """Test cross-validator initialization"""
        cv = TimeSeriesCrossValidator(n_folds=5, test_size=0.2)
        
        assert cv.n_folds == 5
        assert cv.test_size == 0.2
        assert cv.preserve_order is True
    
    def test_time_series_split(self):
        """Test time series splitting"""
        cv = TimeSeriesCrossValidator(n_folds=3, test_size=0.3)
        
        # Create mock time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'value': np.random.randn(100),
            'date': dates
        })
        
        splits = cv.split(data)
        
        assert len(splits) == 3
        
        for train_data, test_data in splits:
            # Test data should come after train data (time series property)
            assert train_data['date'].max() < test_data['date'].min()
            
            # Test size should be approximately correct
            expected_test_size = int(len(data) * 0.3)
            assert abs(len(test_data) - expected_test_size) <= 2
    
    def test_cross_validation_execution(self, simple_param_space, sphere_function):
        """Test cross-validation execution"""
        cv = TimeSeriesCrossValidator(n_folds=3)
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        # Create mock validation data
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        validation_data = pd.DataFrame({
            'returns': np.random.randn(60) * 0.02,
            'date': dates
        })
        
        result = cv.cross_validate(base_optimizer, validation_data)
        
        assert isinstance(result, ValidationResult)
        assert len(result.fold_scores) == 3
        assert hasattr(result, 'mean_score')
        assert hasattr(result, 'std_score')
        
        # All fold scores should be numeric
        assert all(isinstance(score, (int, float)) for score in result.fold_scores)
    
    def test_cv_with_insufficient_data(self):
        """Test cross-validation with insufficient data"""
        cv = TimeSeriesCrossValidator(n_folds=5, test_size=0.3)
        
        # Very small dataset
        small_data = pd.DataFrame({
            'value': [1, 2, 3],
            'date': pd.date_range('2023-01-01', periods=3, freq='D')
        })
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError, match="Insufficient data"):
            cv.split(small_data)
    
    def test_custom_scoring_function(self, simple_param_space, sphere_function):
        """Test custom scoring function"""
        def custom_scorer(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))  # MAE
        
        cv = TimeSeriesCrossValidator(n_folds=2, scoring_function=custom_scorer)
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        validation_data = pd.DataFrame({
            'returns': np.random.randn(30) * 0.02,
            'date': pd.date_range('2023-01-01', periods=30, freq='D')
        })
        
        result = cv.cross_validate(base_optimizer, validation_data)
        
        # Should use custom scoring function
        assert isinstance(result, ValidationResult)
        assert len(result.fold_scores) == 2


class TestSensitivityAnalyzer:
    """Test cases for SensitivityAnalyzer"""
    
    def test_sensitivity_analyzer_initialization(self):
        """Test sensitivity analyzer initialization"""
        analyzer = SensitivityAnalyzer(
            perturbation_levels=[0.01, 0.05, 0.1],
            analysis_methods=['local', 'global']
        )
        
        assert analyzer.perturbation_levels == [0.01, 0.05, 0.1]
        assert analyzer.analysis_methods == ['local', 'global']
    
    def test_local_sensitivity_analysis(self, simple_param_space, sphere_function):
        """Test local sensitivity analysis"""
        analyzer = SensitivityAnalyzer(analysis_methods=['local'])
        
        base_params = {'x': 1.0, 'y': 2.0}
        
        result = analyzer.analyze_sensitivity(
            sphere_function, base_params, simple_param_space
        )
        
        assert isinstance(result, SensitivityResult)
        assert hasattr(result, 'parameter_sensitivities')
        assert hasattr(result, 'total_sensitivity')
        
        # Should have sensitivity for each parameter
        assert 'x' in result.parameter_sensitivities
        assert 'y' in result.parameter_sensitivities
    
    def test_global_sensitivity_analysis(self, simple_param_space, sphere_function):
        """Test global sensitivity analysis"""
        analyzer = SensitivityAnalyzer(
            analysis_methods=['global'],
            global_samples=100
        )
        
        base_params = {'x': 1.0, 'y': 2.0}
        
        result = analyzer.analyze_sensitivity(
            sphere_function, base_params, simple_param_space
        )
        
        assert isinstance(result, SensitivityResult)
        assert hasattr(result, 'global_sensitivity_indices')
        
        # Global sensitivity should provide Sobol indices
        if result.global_sensitivity_indices:
            assert 'first_order' in result.global_sensitivity_indices
            assert 'total_order' in result.global_sensitivity_indices
    
    def test_parameter_ranking(self, complex_param_space, sphere_function):
        """Test parameter importance ranking"""
        analyzer = SensitivityAnalyzer()
        
        base_params = {param: 0.0 for param in complex_param_space.keys()}
        
        result = analyzer.analyze_sensitivity(
            sphere_function, base_params, complex_param_space
        )
        
        # Should provide parameter ranking
        assert hasattr(result, 'parameter_ranking')
        assert isinstance(result.parameter_ranking, list)
        
        # Ranking should include all parameters
        ranked_params = [item[0] for item in result.parameter_ranking]
        assert set(ranked_params) == set(complex_param_space.keys())
    
    def test_sensitivity_with_noisy_function(self, simple_param_space, noisy_function):
        """Test sensitivity analysis with noisy function"""
        analyzer = SensitivityAnalyzer(
            perturbation_levels=[0.05],
            noise_robust=True
        )
        
        base_params = {'x': 0.5, 'y': 1.0}
        
        result = analyzer.analyze_sensitivity(
            noisy_function, base_params, simple_param_space
        )
        
        # Should handle noisy function gracefully
        assert isinstance(result, SensitivityResult)
        assert result.analysis_quality['noise_level'] is not None


class TestDimensionTester:
    """Test cases for DimensionTester"""
    
    def test_dimension_tester_initialization(self):
        """Test dimension tester initialization"""
        tester = DimensionTester(
            max_dimensions=10,
            dimension_increments=[1, 2, 5]
        )
        
        assert tester.max_dimensions == 10
        assert tester.dimension_increments == [1, 2, 5]
    
    def test_curse_of_dimensionality_analysis(self, sphere_function):
        """Test curse of dimensionality analysis"""
        tester = DimensionTester(max_dimensions=8)
        
        # Create base optimizer for different dimensions
        def create_optimizer(n_dims):
            param_space = {f'x{i}': (-5, 5) for i in range(n_dims)}
            return MockBaseOptimizer(param_space, sphere_function)
        
        result = tester.test_dimensionality_scaling(
            create_optimizer, dimensions_to_test=[2, 4, 6]
        )
        
        assert isinstance(result, DimensionTestResult)
        assert hasattr(result, 'dimension_performance')
        assert hasattr(result, 'scaling_analysis')
        
        # Should test all specified dimensions
        assert len(result.dimension_performance) == 3
        assert 2 in result.dimension_performance
        assert 4 in result.dimension_performance
        assert 6 in result.dimension_performance
    
    def test_effective_dimensionality_estimation(self, complex_param_space, sphere_function):
        """Test effective dimensionality estimation"""
        tester = DimensionTester()
        base_optimizer = MockBaseOptimizer(complex_param_space, sphere_function)
        
        result = tester.estimate_effective_dimensionality(base_optimizer)
        
        assert hasattr(result, 'effective_dimensions')
        assert hasattr(result, 'parameter_importance')
        
        # Effective dimensions should be <= total dimensions
        total_dims = len(complex_param_space)
        assert result.effective_dimensions <= total_dims
    
    def test_dimension_reduction_suggestions(self, complex_param_space, sphere_function):
        """Test dimension reduction suggestions"""
        tester = DimensionTester()
        base_optimizer = MockBaseOptimizer(complex_param_space, sphere_function)
        
        result = tester.analyze_parameter_importance(base_optimizer)
        
        assert hasattr(result, 'low_importance_parameters')
        assert hasattr(result, 'reduction_suggestions')
        
        if result.low_importance_parameters:
            assert all(param in complex_param_space for param in result.low_importance_parameters)


class TestRobustEstimator:
    """Test cases for RobustEstimator"""
    
    def test_robust_estimator_initialization(self):
        """Test robust estimator initialization"""
        estimator = RobustEstimator(
            outlier_threshold=2.5,
            confidence_level=0.95
        )
        
        assert estimator.outlier_threshold == 2.5
        assert estimator.confidence_level == 0.95
    
    def test_outlier_detection(self):
        """Test outlier detection in optimization results"""
        estimator = RobustEstimator()
        
        # Create sample results with outliers
        results = [1.0, 1.1, 0.9, 1.2, 0.8, 10.0, 1.0, 0.95]  # 10.0 is outlier
        
        outliers = estimator.detect_outliers(results)
        
        # Should detect the outlier
        assert 10.0 in outliers
        assert len(outliers) >= 1
    
    def test_robust_mean_estimation(self):
        """Test robust mean estimation"""
        estimator = RobustEstimator()
        
        # Data with outliers
        data = [1.0, 1.1, 0.9, 1.2, 0.8, 10.0, 1.0, 0.95, 1.05]
        
        robust_mean = estimator.calculate_robust_mean(data)
        regular_mean = np.mean(data)
        
        # Robust mean should be less affected by outlier
        assert abs(robust_mean - 1.0) < abs(regular_mean - 1.0)
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        estimator = RobustEstimator(confidence_level=0.95)
        
        data = np.random.normal(5.0, 1.0, 100)  # Normal distribution
        
        ci_lower, ci_upper = estimator.calculate_confidence_interval(data)
        
        # Confidence interval should contain the true mean (5.0) most of the time
        assert ci_lower < 5.0 < ci_upper
        
        # 95% CI should be wider than 90% CI
        estimator_90 = RobustEstimator(confidence_level=0.90)
        ci_90_lower, ci_90_upper = estimator_90.calculate_confidence_interval(data)
        
        assert (ci_upper - ci_lower) > (ci_90_upper - ci_90_lower)
    
    def test_robust_metrics_computation(self):
        """Test robust metrics computation"""
        estimator = RobustEstimator()
        
        # Mock optimization results
        cv_scores = [0.8, 0.85, 0.75, 0.9, 0.82]
        noise_results = {
            0.01: [0.85, 0.83, 0.87],
            0.05: [0.80, 0.78, 0.82],
            0.1: [0.70, 0.68, 0.72]
        }
        
        metrics = estimator.compute_robust_metrics(cv_scores, noise_results)
        
        assert isinstance(metrics, RobustMetrics)
        assert hasattr(metrics, 'stability_score')
        assert hasattr(metrics, 'consistency_score')
        assert hasattr(metrics, 'reliability_score')
        assert hasattr(metrics, 'overall_robustness')
        
        # All scores should be between 0 and 1
        assert 0 <= metrics.stability_score <= 1
        assert 0 <= metrics.consistency_score <= 1
        assert 0 <= metrics.reliability_score <= 1
        assert 0 <= metrics.overall_robustness <= 1


class TestRobustnessFrameworkIntegration:
    """Integration tests for the robustness framework"""
    
    def test_full_robustness_pipeline(self, simple_param_space, sphere_function):
        """Test complete robustness analysis pipeline"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=3,
            noise_levels=[0.01, 0.05],
            enable_sensitivity_analysis=True,
            enable_dimension_testing=True
        )
        
        result = robust_optimizer.optimize(n_iterations=20)
        
        # Should have all robustness components
        assert isinstance(result, RobustnessResult)
        assert result.cv_results is not None
        assert result.noise_analysis is not None
        assert result.sensitivity_results is not None
        assert result.robustness_metrics is not None
        
        # Base optimizer should have been called multiple times
        assert base_optimizer.call_count > 5
    
    def test_robustness_with_different_optimizers(self, simple_param_space, sphere_function):
        """Test robustness framework with different base optimizers"""
        # Test with different mock behaviors
        stable_optimizer = MockBaseOptimizer(simple_param_space, sphere_function, mock_result_value=1.0)
        unstable_optimizer = MockBaseOptimizer(simple_param_space, sphere_function, mock_result_value=5.0)
        
        # Stable optimizer
        robust_stable = RobustOptimizer(
            base_optimizer=stable_optimizer,
            cv_folds=3,
            noise_levels=[0.01]
        )
        
        result_stable = robust_stable.optimize(n_iterations=10)
        
        # Unstable optimizer  
        robust_unstable = RobustOptimizer(
            base_optimizer=unstable_optimizer,
            cv_folds=3,
            noise_levels=[0.01]
        )
        
        result_unstable = robust_unstable.optimize(n_iterations=10)
        
        # Stable optimizer should have better robustness scores
        assert result_stable.robustness_metrics.overall_robustness >= result_unstable.robustness_metrics.overall_robustness
    
    def test_robustness_reporting(self, simple_param_space, sphere_function):
        """Test robustness reporting and summary generation"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=3,
            noise_levels=[0.01, 0.05]
        )
        
        result = robust_optimizer.optimize(n_iterations=15)
        
        # Generate robustness report
        report = result.generate_robustness_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'cross_validation' in report
        assert 'noise_analysis' in report
        assert 'robustness_metrics' in report
        
        # Summary should contain key metrics
        summary = report['summary']
        assert 'overall_robustness_score' in summary
        assert 'recommendation' in summary
    
    def test_performance_with_time_series_data(self, sample_strategy_returns):
        """Test robustness framework with time series data"""
        def time_series_objective(params):
            # Simulate strategy performance based on parameters
            strategy_returns = sample_strategy_returns['strategy_1']
            return strategy_returns.std() * params.get('risk_factor', 1.0)
        
        param_space = {'risk_factor': (0.5, 2.0), 'lookback': (10, 50)}
        base_optimizer = MockBaseOptimizer(param_space, time_series_objective)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=3,
            time_series_data=sample_strategy_returns['strategy_1']
        )
        
        result = robust_optimizer.optimize(n_iterations=10)
        
        # Should handle time series data appropriately
        assert isinstance(result, RobustnessResult)
        assert result.cv_results is not None
        
        # CV should respect time series order
        assert hasattr(result.cv_results, 'time_series_validation')


class TestRobustnessFrameworkEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_robustness_with_failing_optimizer(self, simple_param_space, sphere_function):
        """Test robustness framework with failing base optimizer"""
        class FailingOptimizer(BaseOptimizer):
            def optimize(self, **kwargs):
                raise RuntimeError("Optimization failed")
        
        failing_optimizer = FailingOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=failing_optimizer,
            cv_folds=2,
            handle_failures=True
        )
        
        # Should handle failures gracefully
        result = robust_optimizer.optimize(n_iterations=5)
        
        # Should still return a result, possibly with degraded metrics
        assert isinstance(result, RobustnessResult)
    
    def test_robustness_with_extreme_noise(self, simple_param_space, sphere_function):
        """Test robustness with extreme noise levels"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            noise_levels=[0.5, 1.0, 2.0]  # Very high noise
        )
        
        result = robust_optimizer.optimize(n_iterations=10)
        
        # Should handle extreme noise
        assert isinstance(result, RobustnessResult)
        assert result.noise_analysis is not None
        
        # Robustness scores should reflect poor performance under high noise
        assert result.robustness_metrics.overall_robustness < 0.8
    
    def test_robustness_with_minimal_cv_folds(self, simple_param_space, sphere_function):
        """Test robustness with minimal cross-validation folds"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=1  # Minimal CV
        )
        
        result = robust_optimizer.optimize(n_iterations=5)
        
        # Should handle minimal CV gracefully
        assert isinstance(result, RobustnessResult)
        assert result.cv_results is not None
    
    def test_robustness_with_no_validation_data(self, simple_param_space, sphere_function):
        """Test robustness when no validation data is available"""
        base_optimizer = MockBaseOptimizer(simple_param_space, sphere_function)
        
        robust_optimizer = RobustOptimizer(
            base_optimizer=base_optimizer,
            cv_folds=0,  # No CV
            noise_levels=[],  # No noise testing
            enable_sensitivity_analysis=False
        )
        
        result = robust_optimizer.optimize(n_iterations=5)
        
        # Should still work with minimal robustness testing
        assert isinstance(result, RobustnessResult)
        assert result.robustness_metrics is not None