"""
Statistical Pattern Validator for Ultra-High Precision Pattern Validation

Implements comprehensive statistical testing for pattern significance validation:
- T-tests for mean differences
- Chi-square tests for independence  
- Bootstrap validation for robustness
- Effect size calculations
- Confidence interval analysis

This validator is part of Layer 4 of the 7-layer validation system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

from .pattern_repository import PatternSchema

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result of a single statistical test"""
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: float
    degrees_of_freedom: Optional[int]
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significant: bool


@dataclass 
class StatisticalValidationResult:
    """Complete statistical validation result"""
    pattern_id: str
    validation_timestamp: datetime
    
    # Test results
    t_test_result: Optional[StatisticalTestResult]
    chi_square_result: Optional[StatisticalTestResult]
    bootstrap_result: Optional[StatisticalTestResult]
    
    # Summary statistics
    sample_size: int
    effect_size_cohen_d: float
    confidence_level: float
    overall_significance: float
    
    # Quality metrics
    statistical_power: float
    minimum_detectable_effect: float
    recommended_sample_size: int


class StatisticalPatternValidator:
    """
    Statistical Pattern Validator for Layer 4 Validation
    
    Implements rigorous statistical testing to ensure pattern significance:
    - Minimum sample size requirements
    - Multiple statistical test validation
    - Effect size analysis  
    - Bootstrap resampling validation
    - Confidence interval analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Statistical Pattern Validator
        
        Args:
            config: Validator configuration
        """
        self.config = config or self._get_default_config()
        
        # Statistical test parameters
        self.alpha = self.config.get('alpha', 0.01)  # 99% confidence level
        self.min_sample_size = self.config.get('min_sample_size', 100)
        self.bootstrap_iterations = self.config.get('bootstrap_iterations', 1000)
        self.effect_size_threshold = self.config.get('effect_size_threshold', 0.5)
        
        # Test configuration
        self.enable_t_test = self.config.get('enable_t_test', True)
        self.enable_chi_square = self.config.get('enable_chi_square', True)
        self.enable_bootstrap = self.config.get('enable_bootstrap', True)
        
        self.logger = logging.getLogger(f"{__name__}.StatisticalPatternValidator")
        self.logger.info("Statistical Pattern Validator initialized")
        self.logger.info(f"Alpha level: {self.alpha}, Min sample size: {self.min_sample_size}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default statistical validator configuration"""
        return {
            'alpha': 0.01,  # 99% confidence level
            'min_sample_size': 100,
            'bootstrap_iterations': 1000,
            'effect_size_threshold': 0.5,
            'enable_t_test': True,
            'enable_chi_square': True,
            'enable_bootstrap': True,
            'power_analysis': True,
            'bonferroni_correction': True
        }
    
    def validate_pattern_significance(self, pattern: PatternSchema, 
                                    historical_data: Optional[Dict[str, Any]]) -> StatisticalValidationResult:
        """
        Perform comprehensive statistical significance validation
        
        Args:
            pattern: Pattern to validate
            historical_data: Historical performance data
            
        Returns:
            StatisticalValidationResult with detailed test results
        """
        try:
            validation_timestamp = datetime.now()
            
            # Extract pattern performance data
            performance_data = self._extract_performance_data(pattern, historical_data)
            
            if not performance_data or len(performance_data) < self.min_sample_size:
                return self._create_insufficient_data_result(pattern.pattern_id, validation_timestamp)
            
            # Prepare data for statistical tests
            returns = performance_data['returns']
            success_indicators = performance_data['success_indicators']
            sample_size = len(returns)
            
            # Test 1: T-test for mean return significance
            t_test_result = None
            if self.enable_t_test:
                t_test_result = self._perform_t_test(returns, pattern.pattern_id)
            
            # Test 2: Chi-square test for success rate independence
            chi_square_result = None
            if self.enable_chi_square:
                chi_square_result = self._perform_chi_square_test(success_indicators, pattern.pattern_id)
            
            # Test 3: Bootstrap validation for robustness
            bootstrap_result = None
            if self.enable_bootstrap:
                bootstrap_result = self._perform_bootstrap_validation(returns, pattern.pattern_id)
            
            # Calculate effect sizes
            effect_size_cohen_d = self._calculate_cohens_d(returns)
            
            # Calculate overall significance
            overall_significance = self._calculate_overall_significance([
                t_test_result, chi_square_result, bootstrap_result
            ])
            
            # Power analysis
            statistical_power = self._calculate_statistical_power(sample_size, effect_size_cohen_d)
            minimum_detectable_effect = self._calculate_minimum_detectable_effect(sample_size)
            recommended_sample_size = self._calculate_recommended_sample_size(effect_size_cohen_d)
            
            result = StatisticalValidationResult(
                pattern_id=pattern.pattern_id,
                validation_timestamp=validation_timestamp,
                t_test_result=t_test_result,
                chi_square_result=chi_square_result,
                bootstrap_result=bootstrap_result,
                sample_size=sample_size,
                effect_size_cohen_d=effect_size_cohen_d,
                confidence_level=1 - self.alpha,
                overall_significance=overall_significance,
                statistical_power=statistical_power,
                minimum_detectable_effect=minimum_detectable_effect,
                recommended_sample_size=recommended_sample_size
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in statistical validation: {e}")
            return self._create_error_result(pattern.pattern_id, str(e), datetime.now())
    
    def _extract_performance_data(self, pattern: PatternSchema, 
                                historical_data: Optional[Dict[str, Any]]) -> Optional[Dict]:
        """Extract performance data for statistical analysis"""
        try:
            if not historical_data:
                return None
            
            # Get performance history
            performance_history = historical_data.get('performance_history', [])
            if not performance_history:
                return None
            
            # Extract returns and success indicators
            returns = []
            success_indicators = []
            
            for trade in performance_history:
                trade_return = trade.get('return', 0.0)
                trade_success = 1 if trade_return > 0 else 0
                
                returns.append(trade_return)
                success_indicators.append(trade_success)
            
            if not returns:
                return None
            
            return {
                'returns': np.array(returns),
                'success_indicators': np.array(success_indicators),
                'sample_size': len(returns)
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting performance data: {e}")
            return None
    
    def _perform_t_test(self, returns: np.ndarray, pattern_id: str) -> StatisticalTestResult:
        """Perform one-sample t-test against zero mean"""
        try:
            # One-sample t-test against zero (no return)
            t_statistic, p_value = stats.ttest_1samp(returns, 0.0)
            
            # Degrees of freedom
            df = len(returns) - 1
            
            # Critical value for two-tailed test
            critical_value = stats.t.ppf(1 - self.alpha/2, df)
            
            # Effect size (Cohen's d)
            effect_size = np.mean(returns) / np.std(returns, ddof=1)
            
            # Confidence interval
            mean_return = np.mean(returns)
            std_error = stats.sem(returns)
            ci_lower = mean_return - critical_value * std_error
            ci_upper = mean_return + critical_value * std_error
            
            # Interpretation
            significant = p_value < self.alpha
            interpretation = self._interpret_t_test(t_statistic, p_value, significant)
            
            return StatisticalTestResult(
                test_name="one_sample_t_test",
                test_statistic=t_statistic,
                p_value=p_value,
                critical_value=critical_value,
                degrees_of_freedom=df,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                interpretation=interpretation,
                significant=significant
            )
            
        except Exception as e:
            self.logger.warning(f"Error in t-test: {e}")
            return self._create_failed_test_result("one_sample_t_test", str(e))
    
    def _perform_chi_square_test(self, success_indicators: np.ndarray, pattern_id: str) -> StatisticalTestResult:
        """Perform chi-square test for success rate vs random chance"""
        try:
            # Expected frequencies (50% success rate = random chance)
            n_trades = len(success_indicators)
            n_successes = np.sum(success_indicators)
            n_failures = n_trades - n_successes
            
            # Observed vs expected frequencies
            observed = np.array([n_successes, n_failures])
            expected = np.array([n_trades * 0.5, n_trades * 0.5])
            
            # Chi-square test
            chi2_statistic, p_value = stats.chisquare(observed, expected)
            
            # Degrees of freedom
            df = 1  # 2 categories - 1
            
            # Critical value
            critical_value = stats.chi2.ppf(1 - self.alpha, df)
            
            # Effect size (Cramer's V)
            effect_size = np.sqrt(chi2_statistic / n_trades)
            
            # Confidence interval for proportion
            success_rate = n_successes / n_trades
            ci_lower, ci_upper = self._proportion_confidence_interval(n_successes, n_trades)
            
            # Interpretation
            significant = p_value < self.alpha
            interpretation = self._interpret_chi_square(chi2_statistic, p_value, significant, success_rate)
            
            return StatisticalTestResult(
                test_name="chi_square_goodness_of_fit",
                test_statistic=chi2_statistic,
                p_value=p_value,
                critical_value=critical_value,
                degrees_of_freedom=df,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                interpretation=interpretation,
                significant=significant
            )
            
        except Exception as e:
            self.logger.warning(f"Error in chi-square test: {e}")
            return self._create_failed_test_result("chi_square_goodness_of_fit", str(e))
    
    def _perform_bootstrap_validation(self, returns: np.ndarray, pattern_id: str) -> StatisticalTestResult:
        """Perform bootstrap validation for robustness"""
        try:
            # Bootstrap resampling
            def bootstrap_mean(data):
                return np.mean(data)
            
            # Create bootstrap distribution
            bootstrap_means = []
            for _ in range(self.bootstrap_iterations):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Calculate statistics
            original_mean = np.mean(returns)
            bootstrap_std = np.std(bootstrap_means)
            
            # Bootstrap confidence interval
            ci_lower = np.percentile(bootstrap_means, (self.alpha/2) * 100)
            ci_upper = np.percentile(bootstrap_means, (1 - self.alpha/2) * 100)
            
            # P-value: proportion of bootstrap means <= 0
            p_value = np.sum(bootstrap_means <= 0) / len(bootstrap_means)
            
            # Effect size
            effect_size = original_mean / bootstrap_std
            
            # Test statistic (standardized mean)
            test_statistic = original_mean / bootstrap_std
            
            # Critical value (approximate)
            critical_value = stats.norm.ppf(1 - self.alpha/2)
            
            # Interpretation
            significant = p_value < self.alpha and ci_lower > 0
            interpretation = self._interpret_bootstrap(original_mean, p_value, significant, ci_lower, ci_upper)
            
            return StatisticalTestResult(
                test_name="bootstrap_validation",
                test_statistic=test_statistic,
                p_value=p_value,
                critical_value=critical_value,
                degrees_of_freedom=None,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                interpretation=interpretation,
                significant=significant
            )
            
        except Exception as e:
            self.logger.warning(f"Error in bootstrap validation: {e}")
            return self._create_failed_test_result("bootstrap_validation", str(e))
    
    def _calculate_cohens_d(self, returns: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            # Cohen's d = (mean - 0) / std
            cohens_d = mean_return / std_return if std_return > 0 else 0.0
            
            return abs(cohens_d)
            
        except Exception:
            return 0.0
    
    def _calculate_overall_significance(self, test_results: List[Optional[StatisticalTestResult]]) -> float:
        """Calculate overall significance score from all tests"""
        try:
            valid_tests = [test for test in test_results if test is not None]
            
            if not valid_tests:
                return 0.0
            
            # Count significant tests
            significant_tests = sum(1 for test in valid_tests if test.significant)
            
            # Calculate significance score
            significance_score = significant_tests / len(valid_tests)
            
            # Bonus for low p-values
            avg_p_value = np.mean([test.p_value for test in valid_tests])
            p_value_bonus = max(0.0, (self.alpha - avg_p_value) / self.alpha * 0.2)
            
            # Final score
            overall_significance = min(1.0, significance_score + p_value_bonus)
            
            return overall_significance
            
        except Exception:
            return 0.0
    
    def _calculate_statistical_power(self, sample_size: int, effect_size: float) -> float:
        """Calculate statistical power for current sample size and effect size"""
        try:
            # Approximate power calculation for t-test
            # Based on non-central t-distribution
            
            ncp = effect_size * np.sqrt(sample_size)  # Non-centrality parameter
            critical_t = stats.t.ppf(1 - self.alpha/2, sample_size - 1)
            
            # Power = P(|T| > critical_t | H1 is true)
            power = 1 - stats.nct.cdf(critical_t, sample_size - 1, ncp) + stats.nct.cdf(-critical_t, sample_size - 1, ncp)
            
            return min(1.0, max(0.0, power))
            
        except Exception:
            return 0.5
    
    def _calculate_minimum_detectable_effect(self, sample_size: int) -> float:
        """Calculate minimum detectable effect size for current sample size"""
        try:
            # For 80% power
            desired_power = 0.8
            
            # Approximate calculation
            critical_t = stats.t.ppf(1 - self.alpha/2, sample_size - 1)
            power_t = stats.t.ppf(desired_power, sample_size - 1)
            
            mde = (critical_t + power_t) / np.sqrt(sample_size)
            
            return max(0.0, mde)
            
        except Exception:
            return 1.0
    
    def _calculate_recommended_sample_size(self, effect_size: float) -> int:
        """Calculate recommended sample size for adequate power"""
        try:
            if effect_size <= 0:
                return self.min_sample_size * 2
            
            # For 80% power and current effect size
            desired_power = 0.8
            
            # Approximate sample size calculation
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(desired_power)
            
            n_required = ((z_alpha + z_beta) / effect_size) ** 2
            
            return max(self.min_sample_size, int(np.ceil(n_required)))
            
        except Exception:
            return self.min_sample_size * 2
    
    def _proportion_confidence_interval(self, n_successes: int, n_total: int) -> Tuple[float, float]:
        """Calculate confidence interval for proportion"""
        try:
            p = n_successes / n_total
            z_score = stats.norm.ppf(1 - self.alpha/2)
            
            margin_error = z_score * np.sqrt(p * (1 - p) / n_total)
            
            ci_lower = max(0.0, p - margin_error)
            ci_upper = min(1.0, p + margin_error)
            
            return ci_lower, ci_upper
            
        except Exception:
            return 0.0, 1.0
    
    def _interpret_t_test(self, t_statistic: float, p_value: float, significant: bool) -> str:
        """Interpret t-test results"""
        if significant:
            direction = "positive" if t_statistic > 0 else "negative"
            return f"Significant {direction} mean return (t={t_statistic:.3f}, p={p_value:.4f})"
        else:
            return f"Non-significant mean return (t={t_statistic:.3f}, p={p_value:.4f})"
    
    def _interpret_chi_square(self, chi2_statistic: float, p_value: float, 
                            significant: bool, success_rate: float) -> str:
        """Interpret chi-square test results"""
        if significant:
            if success_rate > 0.5:
                return f"Success rate ({success_rate:.1%}) significantly higher than chance (χ²={chi2_statistic:.3f}, p={p_value:.4f})"
            else:
                return f"Success rate ({success_rate:.1%}) significantly lower than chance (χ²={chi2_statistic:.3f}, p={p_value:.4f})"
        else:
            return f"Success rate ({success_rate:.1%}) not significantly different from chance (χ²={chi2_statistic:.3f}, p={p_value:.4f})"
    
    def _interpret_bootstrap(self, mean_return: float, p_value: float, significant: bool,
                           ci_lower: float, ci_upper: float) -> str:
        """Interpret bootstrap validation results"""
        if significant:
            return f"Bootstrap validation confirms significant positive returns (mean={mean_return:.4f}, 99% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
        else:
            return f"Bootstrap validation shows non-significant returns (mean={mean_return:.4f}, p={p_value:.4f})"
    
    def _create_failed_test_result(self, test_name: str, error_message: str) -> StatisticalTestResult:
        """Create failed test result"""
        return StatisticalTestResult(
            test_name=test_name,
            test_statistic=0.0,
            p_value=1.0,
            critical_value=0.0,
            degrees_of_freedom=None,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            interpretation=f"Test failed: {error_message}",
            significant=False
        )
    
    def _create_insufficient_data_result(self, pattern_id: str, 
                                       validation_timestamp: datetime) -> StatisticalValidationResult:
        """Create result for insufficient data"""
        return StatisticalValidationResult(
            pattern_id=pattern_id,
            validation_timestamp=validation_timestamp,
            t_test_result=None,
            chi_square_result=None,
            bootstrap_result=None,
            sample_size=0,
            effect_size_cohen_d=0.0,
            confidence_level=0.0,
            overall_significance=0.0,
            statistical_power=0.0,
            minimum_detectable_effect=0.0,
            recommended_sample_size=self.min_sample_size
        )
    
    def _create_error_result(self, pattern_id: str, error_message: str,
                           validation_timestamp: datetime) -> StatisticalValidationResult:
        """Create error result"""
        return StatisticalValidationResult(
            pattern_id=pattern_id,
            validation_timestamp=validation_timestamp,
            t_test_result=self._create_failed_test_result("error", error_message),
            chi_square_result=None,
            bootstrap_result=None,
            sample_size=0,
            effect_size_cohen_d=0.0,
            confidence_level=0.0,
            overall_significance=0.0,
            statistical_power=0.0,
            minimum_detectable_effect=0.0,
            recommended_sample_size=self.min_sample_size
        )
    
    def get_validator_summary(self) -> Dict[str, Any]:
        """Get statistical validator summary"""
        return {
            "alpha_level": self.alpha,
            "confidence_level": 1 - self.alpha,
            "min_sample_size": self.min_sample_size,
            "bootstrap_iterations": self.bootstrap_iterations,
            "effect_size_threshold": self.effect_size_threshold,
            "tests_enabled": {
                "t_test": self.enable_t_test,
                "chi_square": self.enable_chi_square,
                "bootstrap": self.enable_bootstrap
            },
            "validator_type": "StatisticalPatternValidator",
            "version": "1.0.0"
        }