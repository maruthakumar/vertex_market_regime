"""
System Validator

This module implements comprehensive validation for the adaptive market regime
formation system, ensuring all components meet quality standards and work
correctly together before production deployment.

Key Features:
- Component-level validation tests
- Integration validation across modules
- Performance benchmarking and validation
- Data quality validation
- Configuration validation
- Production readiness checks
- Automated test report generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
import time
from pathlib import Path
import hashlib
import traceback

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"


class ValidationStatus(Enum):
    """Validation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"


class ComponentType(Enum):
    """System components to validate"""
    # Phase 1 - Core Infrastructure
    BASE_REGIME_DETECTOR = "base_regime_detector"
    CONFIGURATION_MANAGER = "configuration_manager"
    HISTORICAL_ANALYZER = "historical_analyzer"
    
    # Phase 2 - Adaptive Components
    ADAPTIVE_SCORING_LAYER = "adaptive_scoring_layer"
    TRANSITION_MATRIX_ANALYZER = "transition_matrix_analyzer"
    DYNAMIC_BOUNDARY_OPTIMIZER = "dynamic_boundary_optimizer"
    
    # Phase 3 - Intelligence Layer
    INTELLIGENT_TRANSITION_MANAGER = "intelligent_transition_manager"
    REGIME_STABILITY_MONITOR = "regime_stability_monitor"
    ADAPTIVE_NOISE_FILTER = "adaptive_noise_filter"
    
    # Phase 4 - Optimization & Feedback
    PERFORMANCE_FEEDBACK_SYSTEM = "performance_feedback_system"
    CONTINUOUS_LEARNING_ENGINE = "continuous_learning_engine"
    REGIME_OPTIMIZATION_SCHEDULER = "regime_optimization_scheduler"
    
    # Integration
    INTEGRATED_SYSTEM = "integrated_system"


@dataclass
class ValidationTest:
    """Individual validation test"""
    test_id: str
    test_name: str
    component: ComponentType
    validation_level: ValidationLevel
    test_function: Callable
    timeout: float = 60.0
    critical: bool = True
    dependencies: List[str] = field(default_factory=list)
    expected_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_id: str
    test_name: str
    component: ComponentType
    status: ValidationStatus
    execution_time: float
    timestamp: datetime
    metrics: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    timestamp: datetime
    validation_level: ValidationLevel
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    skipped_tests: int
    overall_status: ValidationStatus
    component_results: Dict[ComponentType, List[ValidationResult]]
    performance_metrics: Dict[str, Any]
    data_quality_metrics: Dict[str, Any]
    integration_metrics: Dict[str, Any]
    production_readiness_score: float
    recommendations: List[str]
    detailed_results: List[ValidationResult]


@dataclass
class PerformanceBenchmark:
    """Performance benchmark criteria"""
    metric_name: str
    target_value: float
    tolerance: float
    unit: str
    critical: bool = True


@dataclass
class DataQualityCheck:
    """Data quality validation criteria"""
    check_name: str
    validation_function: Callable
    threshold: float
    description: str


class SystemValidator:
    """
    Comprehensive system validation framework
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize system validator
        
        Args:
            validation_level: Level of validation to perform
        """
        self.validation_level = validation_level
        
        # Test registry
        self.validation_tests: Dict[str, ValidationTest] = {}
        self.test_results: List[ValidationResult] = []
        
        # Performance benchmarks
        self.performance_benchmarks: Dict[str, PerformanceBenchmark] = {}
        self._initialize_performance_benchmarks()
        
        # Data quality checks
        self.data_quality_checks: List[DataQualityCheck] = []
        self._initialize_data_quality_checks()
        
        # Module references (will be set during validation)
        self.modules: Dict[ComponentType, Any] = {}
        
        # Validation state
        self.validation_start_time: Optional[datetime] = None
        self.validation_end_time: Optional[datetime] = None
        self.market_data: Optional[pd.DataFrame] = None
        
        # Initialize validation tests
        self._register_validation_tests()
        
        logger.info(f"SystemValidator initialized with level: {validation_level.value}")
    
    def _initialize_performance_benchmarks(self):
        """Initialize performance benchmark criteria"""
        
        benchmarks = [
            # Latency benchmarks
            PerformanceBenchmark("regime_prediction_latency", 10.0, 2.0, "ms", True),
            PerformanceBenchmark("asl_scoring_latency", 5.0, 1.0, "ms", True),
            PerformanceBenchmark("transition_analysis_latency", 20.0, 5.0, "ms", False),
            PerformanceBenchmark("optimization_cycle_time", 60.0, 10.0, "seconds", False),
            
            # Accuracy benchmarks
            PerformanceBenchmark("regime_prediction_accuracy", 0.7, 0.05, "ratio", True),
            PerformanceBenchmark("transition_detection_accuracy", 0.75, 0.05, "ratio", True),
            PerformanceBenchmark("noise_filter_effectiveness", 0.8, 0.1, "ratio", False),
            
            # Resource usage benchmarks
            PerformanceBenchmark("cpu_usage_average", 0.7, 0.1, "ratio", False),
            PerformanceBenchmark("memory_usage_peak", 0.8, 0.1, "ratio", True),
            PerformanceBenchmark("thread_pool_utilization", 0.6, 0.2, "ratio", False),
            
            # Throughput benchmarks
            PerformanceBenchmark("market_data_throughput", 1000.0, 100.0, "records/sec", True),
            PerformanceBenchmark("learning_examples_per_second", 100.0, 20.0, "examples/sec", False),
            PerformanceBenchmark("optimization_tasks_per_hour", 50.0, 10.0, "tasks/hour", False)
        ]
        
        for benchmark in benchmarks:
            self.performance_benchmarks[benchmark.metric_name] = benchmark
    
    def _initialize_data_quality_checks(self):
        """Initialize data quality validation checks"""
        
        self.data_quality_checks = [
            DataQualityCheck(
                "completeness",
                self._check_data_completeness,
                0.95,
                "Check for missing values in critical fields"
            ),
            DataQualityCheck(
                "consistency",
                self._check_data_consistency,
                0.98,
                "Check for logical consistency in data"
            ),
            DataQualityCheck(
                "timeliness",
                self._check_data_timeliness,
                0.99,
                "Check for data freshness and timestamp validity"
            ),
            DataQualityCheck(
                "accuracy",
                self._check_data_accuracy,
                0.95,
                "Check for data accuracy and reasonable values"
            ),
            DataQualityCheck(
                "uniqueness",
                self._check_data_uniqueness,
                0.999,
                "Check for duplicate records"
            )
        ]
    
    def _register_validation_tests(self):
        """Register all validation tests"""
        
        # Component validation tests
        self._register_component_tests()
        
        # Integration validation tests
        self._register_integration_tests()
        
        # Performance validation tests
        self._register_performance_tests()
        
        # Production readiness tests
        if self.validation_level == ValidationLevel.PRODUCTION:
            self._register_production_tests()
    
    def _register_component_tests(self):
        """Register component-level validation tests"""
        
        # Phase 2 - Adaptive Components
        self.register_test(ValidationTest(
            test_id="asl_001",
            test_name="ASL Weight Evolution",
            component=ComponentType.ADAPTIVE_SCORING_LAYER,
            validation_level=ValidationLevel.BASIC,
            test_function=self._validate_asl_weights,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="tma_001",
            test_name="Transition Matrix Properties",
            component=ComponentType.TRANSITION_MATRIX_ANALYZER,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_transition_matrix,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="dbo_001",
            test_name="Boundary Optimization Convergence",
            component=ComponentType.DYNAMIC_BOUNDARY_OPTIMIZER,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_boundary_optimization,
            critical=True
        ))
        
        # Phase 3 - Intelligence Layer
        self.register_test(ValidationTest(
            test_id="itm_001",
            test_name="Transition Filtering Effectiveness",
            component=ComponentType.INTELLIGENT_TRANSITION_MANAGER,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_transition_filtering,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="rsm_001",
            test_name="Stability Monitoring Accuracy",
            component=ComponentType.REGIME_STABILITY_MONITOR,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_stability_monitoring,
            critical=False
        ))
        
        self.register_test(ValidationTest(
            test_id="anf_001",
            test_name="Noise Filter Performance",
            component=ComponentType.ADAPTIVE_NOISE_FILTER,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_noise_filtering,
            critical=False
        ))
        
        # Phase 4 - Optimization & Feedback
        self.register_test(ValidationTest(
            test_id="pfs_001",
            test_name="Performance Feedback Accuracy",
            component=ComponentType.PERFORMANCE_FEEDBACK_SYSTEM,
            validation_level=ValidationLevel.COMPREHENSIVE,
            test_function=self._validate_performance_feedback,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="cle_001",
            test_name="Learning Engine Adaptation",
            component=ComponentType.CONTINUOUS_LEARNING_ENGINE,
            validation_level=ValidationLevel.COMPREHENSIVE,
            test_function=self._validate_learning_adaptation,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="ros_001",
            test_name="Scheduler Task Management",
            component=ComponentType.REGIME_OPTIMIZATION_SCHEDULER,
            validation_level=ValidationLevel.COMPREHENSIVE,
            test_function=self._validate_scheduler_efficiency,
            critical=False
        ))
    
    def _register_integration_tests(self):
        """Register integration validation tests"""
        
        self.register_test(ValidationTest(
            test_id="int_001",
            test_name="End-to-End Prediction Pipeline",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_prediction_pipeline,
            critical=True,
            timeout=120.0
        ))
        
        self.register_test(ValidationTest(
            test_id="int_002",
            test_name="Cross-Component Data Flow",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.COMPREHENSIVE,
            test_function=self._validate_data_flow,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="int_003",
            test_name="Feedback Loop Integration",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.COMPREHENSIVE,
            test_function=self._validate_feedback_loops,
            critical=False
        ))
    
    def _register_performance_tests(self):
        """Register performance validation tests"""
        
        self.register_test(ValidationTest(
            test_id="perf_001",
            test_name="System Latency Benchmarks",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_system_latency,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="perf_002",
            test_name="Resource Usage Limits",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.STANDARD,
            test_function=self._validate_resource_usage,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="perf_003",
            test_name="Throughput Benchmarks",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.COMPREHENSIVE,
            test_function=self._validate_throughput,
            critical=False
        ))
    
    def _register_production_tests(self):
        """Register production readiness tests"""
        
        self.register_test(ValidationTest(
            test_id="prod_001",
            test_name="Failover and Recovery",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.PRODUCTION,
            test_function=self._validate_failover_recovery,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="prod_002",
            test_name="Data Persistence and Recovery",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.PRODUCTION,
            test_function=self._validate_data_persistence,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="prod_003",
            test_name="Security and Access Control",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.PRODUCTION,
            test_function=self._validate_security,
            critical=True
        ))
        
        self.register_test(ValidationTest(
            test_id="prod_004",
            test_name="Monitoring and Alerting",
            component=ComponentType.INTEGRATED_SYSTEM,
            validation_level=ValidationLevel.PRODUCTION,
            test_function=self._validate_monitoring,
            critical=False
        ))
    
    def register_test(self, test: ValidationTest):
        """Register a validation test"""
        
        self.validation_tests[test.test_id] = test
        logger.debug(f"Registered validation test: {test.test_name} ({test.test_id})")
    
    def set_modules(self, modules: Dict[ComponentType, Any]):
        """Set module references for validation"""
        
        self.modules.update(modules)
        logger.info(f"Set {len(modules)} module references for validation")
    
    def set_market_data(self, market_data: pd.DataFrame):
        """Set market data for validation"""
        
        self.market_data = market_data
        logger.info(f"Set market data with {len(market_data)} records for validation")
    
    def validate_system(self, components: Optional[List[ComponentType]] = None) -> ValidationReport:
        """
        Perform comprehensive system validation
        
        Args:
            components: Specific components to validate (None = all)
            
        Returns:
            Comprehensive validation report
        """
        self.validation_start_time = datetime.now()
        self.test_results = []
        
        logger.info(f"Starting system validation at level: {self.validation_level.value}")
        
        # Filter tests by validation level
        tests_to_run = [
            test for test in self.validation_tests.values()
            if test.validation_level.value <= self.validation_level.value
        ]
        
        # Filter by components if specified
        if components:
            tests_to_run = [
                test for test in tests_to_run
                if test.component in components
            ]
        
        # Sort tests by dependencies
        tests_to_run = self._sort_tests_by_dependencies(tests_to_run)
        
        # Execute validation tests
        for test in tests_to_run:
            result = self._execute_validation_test(test)
            self.test_results.append(result)
        
        # Validate data quality
        data_quality_metrics = self._validate_data_quality()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Calculate integration metrics
        integration_metrics = self._calculate_integration_metrics()
        
        # Generate validation report
        report = self._generate_validation_report(
            data_quality_metrics,
            performance_metrics,
            integration_metrics
        )
        
        self.validation_end_time = datetime.now()
        
        logger.info(f"System validation completed: {report.overall_status.value}")
        
        return report
    
    def _execute_validation_test(self, test: ValidationTest) -> ValidationResult:
        """Execute a single validation test"""
        
        logger.debug(f"Executing test: {test.test_name}")
        start_time = time.time()
        
        try:
            # Check dependencies
            if not self._check_test_dependencies(test):
                return ValidationResult(
                    test_id=test.test_id,
                    test_name=test.test_name,
                    component=test.component,
                    status=ValidationStatus.SKIPPED,
                    execution_time=0.0,
                    timestamp=datetime.now(),
                    metrics={},
                    error_message="Dependencies not met"
                )
            
            # Execute test with timeout
            metrics = test.test_function()
            
            # Evaluate results
            status, warnings, recommendations = self._evaluate_test_results(
                metrics, test.expected_metrics
            )
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                test_id=test.test_id,
                test_name=test.test_name,
                component=test.component,
                status=status,
                execution_time=execution_time,
                timestamp=datetime.now(),
                metrics=metrics,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            
            return ValidationResult(
                test_id=test.test_id,
                test_name=test.test_name,
                component=test.component,
                status=ValidationStatus.FAILED,
                execution_time=execution_time,
                timestamp=datetime.now(),
                metrics={},
                error_message=error_msg
            )
    
    def _validate_asl_weights(self) -> Dict[str, Any]:
        """Validate ASL weight evolution"""
        
        if ComponentType.ADAPTIVE_SCORING_LAYER not in self.modules:
            raise ValueError("ASL module not available")
        
        asl = self.modules[ComponentType.ADAPTIVE_SCORING_LAYER]
        metrics = {}
        
        # Check weight normalization
        weights = asl.weights
        weight_sum = sum(weights.values())
        metrics['weight_normalization'] = abs(weight_sum - 1.0) < 0.001
        
        # Check weight bounds
        metrics['weights_in_bounds'] = all(
            asl.config.min_weight <= w <= asl.config.max_weight 
            for w in weights.values()
        )
        
        # Check weight diversity
        weight_variance = np.var(list(weights.values()))
        metrics['weight_diversity'] = weight_variance > 0.01
        
        # Check learning capability
        initial_weights = weights.copy()
        
        # Simulate some predictions and updates
        for _ in range(10):
            dummy_scores = {i: np.random.random() for i in range(12)}
            actual_regime = np.random.randint(0, 12)
            asl.update_weights_based_on_performance(dummy_scores, actual_regime)
        
        # Check if weights changed
        weight_changes = [
            abs(weights[k] - initial_weights[k]) 
            for k in weights.keys()
        ]
        metrics['weight_adaptation'] = max(weight_changes) > 0.001
        
        # Restore initial weights
        asl.weights = initial_weights
        
        return metrics
    
    def _validate_transition_matrix(self) -> Dict[str, Any]:
        """Validate transition matrix properties"""
        
        if ComponentType.TRANSITION_MATRIX_ANALYZER not in self.modules:
            raise ValueError("Transition Matrix Analyzer not available")
        
        analyzer = self.modules[ComponentType.TRANSITION_MATRIX_ANALYZER]
        metrics = {}
        
        # Generate test sequence
        test_sequence = np.random.randint(0, 12, 1000).tolist()
        timestamps = pd.date_range(start='2024-01-01', periods=1000, freq='5min').tolist()
        
        # Analyze transitions
        results = analyzer.analyze_transitions(test_sequence, timestamps)
        
        # Check matrix properties
        matrix = results['transition_matrix']
        
        # Row stochasticity
        row_sums = matrix.sum(axis=1)
        metrics['row_stochastic'] = np.allclose(row_sums, 1.0, atol=0.001)
        
        # Matrix completeness
        metrics['matrix_complete'] = matrix.shape == (12, 12)
        
        # Pattern detection
        metrics['patterns_found'] = len(results['transition_patterns']) > 0
        
        # Markov properties
        if 'markov_analysis' in results:
            markov = results['markov_analysis']
            metrics['is_ergodic'] = markov.is_ergodic
            metrics['has_stationary_dist'] = markov.stationary_distribution is not None
        
        return metrics
    
    def _validate_boundary_optimization(self) -> Dict[str, Any]:
        """Validate boundary optimization"""
        
        if ComponentType.DYNAMIC_BOUNDARY_OPTIMIZER not in self.modules:
            raise ValueError("Boundary Optimizer not available")
        
        optimizer = self.modules[ComponentType.DYNAMIC_BOUNDARY_OPTIMIZER]
        metrics = {}
        
        # Create test performance data
        performance_data = []
        for i in range(100):
            performance_data.append({
                'predicted_regime': np.random.randint(0, 12),
                'actual_regime': np.random.randint(0, 12),
                'timestamp': datetime.now() - timedelta(minutes=i*5)
            })
        
        market_conditions = {
            'volatility': 0.2,
            'trend': 0.01,
            'volume_ratio': 1.0
        }
        
        # Test optimization
        result = optimizer.optimize_boundaries(performance_data, market_conditions)
        
        # Check optimization results
        metrics['optimization_converged'] = result.convergence_status
        metrics['iterations_reasonable'] = 1 <= result.iterations <= 100
        metrics['improvement_achieved'] = result.improvement > 0
        metrics['time_reasonable'] = result.optimization_time < 60.0
        
        # Check boundaries
        boundaries = optimizer.current_boundaries
        metrics['boundaries_valid'] = len(boundaries) > 0
        
        return metrics
    
    def _validate_transition_filtering(self) -> Dict[str, Any]:
        """Validate transition filtering effectiveness"""
        
        if ComponentType.INTELLIGENT_TRANSITION_MANAGER not in self.modules:
            raise ValueError("Transition Manager not available")
        
        manager = self.modules[ComponentType.INTELLIGENT_TRANSITION_MANAGER]
        metrics = {}
        
        # Test noise filtering
        noise_filtered = 0
        total_transitions = 50
        
        for i in range(total_transitions):
            # Create noisy transition
            current_regime = i % 12
            proposed_regime = (i + 1) % 12
            
            # Add noise to some transitions
            if i % 5 == 0:  # 20% noisy
                regime_scores = {j: 0.08 + np.random.random() * 0.02 for j in range(12)}
            else:
                regime_scores = {j: 0.05 for j in range(12)}
                regime_scores[proposed_regime] = 0.45
            
            # Normalize
            total = sum(regime_scores.values())
            regime_scores = {k: v/total for k, v in regime_scores.items()}
            
            market_data = {
                'volatility': 0.15 + np.random.random() * 0.1,
                'volume_ratio': 0.8 + np.random.random() * 0.4
            }
            
            decision = manager.evaluate_transition(
                proposed_regime, regime_scores, market_data
            )
            
            if not decision.approved and i % 5 == 0:
                noise_filtered += 1
        
        metrics['noise_filter_rate'] = noise_filtered / (total_transitions * 0.2)
        metrics['false_positive_prevention'] = manager.false_positives_prevented
        metrics['filter_effectiveness'] = metrics['noise_filter_rate'] > 0.7
        
        return metrics
    
    def _validate_stability_monitoring(self) -> Dict[str, Any]:
        """Validate stability monitoring accuracy"""
        
        if ComponentType.REGIME_STABILITY_MONITOR not in self.modules:
            raise ValueError("Stability Monitor not available")
        
        monitor = self.modules[ComponentType.REGIME_STABILITY_MONITOR]
        metrics = {}
        
        # Test stability detection
        for regime_id in range(5):  # Test subset
            # Stable regime
            for _ in range(20):
                regime_scores = {i: 0.05 for i in range(12)}
                regime_scores[regime_id] = 0.5
                
                market_data = {
                    'volatility': 0.15,
                    'trend': 0.01,
                    'volume_ratio': 1.0
                }
                
                monitor.update_regime_data(regime_id, regime_scores, market_data)
        
        # Get stability report
        report = monitor.get_stability_report()
        
        metrics['stability_monitoring_active'] = len(report['regime_metrics']) > 0
        metrics['anomaly_detection_working'] = 'anomaly_summary' in report
        metrics['system_stability_reasonable'] = 0.0 <= report['system_stability_score'] <= 1.0
        
        return metrics
    
    def _validate_noise_filtering(self) -> Dict[str, Any]:
        """Validate noise filter performance"""
        
        if ComponentType.ADAPTIVE_NOISE_FILTER not in self.modules:
            raise ValueError("Noise Filter not available")
        
        noise_filter = self.modules[ComponentType.ADAPTIVE_NOISE_FILTER]
        metrics = {}
        
        # Test different noise types
        noise_detected_count = 0
        total_tests = 30
        
        for i in range(total_tests):
            # Create regime scores with various noise patterns
            if i % 3 == 0:  # Microstructure noise
                regime_scores = {j: 0.08 + np.random.random() * 0.01 for j in range(12)}
            elif i % 3 == 1:  # Volume anomaly
                regime_scores = {j: np.random.random() for j in range(12)}
            else:  # Clean signal
                regime_scores = {j: 0.05 for j in range(12)}
                regime_scores[5] = 0.45
            
            # Normalize
            total = sum(regime_scores.values())
            regime_scores = {k: v/total for k, v in regime_scores.items()}
            
            market_data = {
                'volume': 1000 if i % 3 != 1 else 50,  # Low volume for anomaly
                'volume_ratio': 1.0 if i % 3 != 1 else 0.1,
                'volatility': 0.2
            }
            
            result = noise_filter.filter_regime_signal(
                regime_scores, market_data, 5
            )
            
            if result.has_noise and i % 3 != 2:
                noise_detected_count += 1
        
        metrics['noise_detection_rate'] = noise_detected_count / (total_tests * 2/3)
        metrics['noise_types_detected'] = len(set(
            nt.value for nt in result.noise_types
        )) if result.has_noise else 0
        metrics['filter_effectiveness'] = metrics['noise_detection_rate'] > 0.6
        
        return metrics
    
    def _validate_performance_feedback(self) -> Dict[str, Any]:
        """Validate performance feedback accuracy"""
        
        if ComponentType.PERFORMANCE_FEEDBACK_SYSTEM not in self.modules:
            raise ValueError("Performance Feedback System not available")
        
        feedback_system = self.modules[ComponentType.PERFORMANCE_FEEDBACK_SYSTEM]
        metrics = {}
        
        # Test performance tracking
        test_predictions = np.random.randint(0, 12, 100).tolist()
        test_actuals = test_predictions.copy()
        
        # Add some errors
        for i in range(20):
            idx = np.random.randint(0, 100)
            test_actuals[idx] = (test_actuals[idx] + 1) % 12
        
        # Evaluate performance
        perf_metrics = feedback_system.evaluate_component_performance(
            component=ComponentType.INTEGRATED_SYSTEM,
            predictions=test_predictions,
            actual_values=test_actuals,
            context={'test': True}
        )
        
        # Check metrics
        metrics['accuracy_calculated'] = 'accuracy' in perf_metrics
        metrics['accuracy_reasonable'] = 0.7 <= perf_metrics.get('accuracy', 0) <= 0.85
        metrics['feedback_actions_generated'] = len(
            feedback_system.get_pending_feedback_actions()
        ) > 0
        
        # Get report
        report = feedback_system.generate_performance_report()
        metrics['report_generated'] = report is not None
        metrics['system_score_valid'] = 0.0 <= report.system_performance_score <= 1.0
        
        return metrics
    
    def _validate_learning_adaptation(self) -> Dict[str, Any]:
        """Validate learning engine adaptation"""
        
        if ComponentType.CONTINUOUS_LEARNING_ENGINE not in self.modules:
            raise ValueError("Learning Engine not available")
        
        learning_engine = self.modules[ComponentType.CONTINUOUS_LEARNING_ENGINE]
        metrics = {}
        
        # Test learning capability
        from ..optimization.continuous_learning_engine import LearningExample
        
        # Add learning examples
        for i in range(100):
            features = np.random.randn(10)
            target = 0 if features[0] > 0 else 1  # Simple pattern
            
            example = LearningExample(
                features=features,
                target=target,
                timestamp=datetime.now(),
                context={'iteration': i}
            )
            
            learning_engine.add_learning_example(example)
        
        # Test predictions
        correct_predictions = 0
        total_predictions = 20
        
        for _ in range(total_predictions):
            test_features = np.random.randn(10)
            expected = 0 if test_features[0] > 0 else 1
            
            prediction, confidence = learning_engine.predict(test_features)
            
            if prediction == expected:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        
        # Get statistics
        stats = learning_engine.get_learning_statistics()
        
        metrics['learning_active'] = stats['total_examples'] > 0
        metrics['models_created'] = stats['active_models'] > 0
        metrics['learning_accuracy'] = accuracy
        metrics['accuracy_reasonable'] = accuracy > 0.6  # Should learn simple pattern
        
        return metrics
    
    def _validate_scheduler_efficiency(self) -> Dict[str, Any]:
        """Validate scheduler task management"""
        
        if ComponentType.REGIME_OPTIMIZATION_SCHEDULER not in self.modules:
            raise ValueError("Scheduler not available")
        
        scheduler = self.modules[ComponentType.REGIME_OPTIMIZATION_SCHEDULER]
        metrics = {}
        
        # Start scheduler
        scheduler.start_scheduler()
        
        try:
            # Schedule test tasks
            from ..optimization.regime_optimization_scheduler import OptimizationTask, TaskType, TaskPriority
            
            test_tasks = []
            for i in range(5):
                task = OptimizationTask(
                    task_id=f"validation_task_{i}",
                    task_type=TaskType.MODEL_VALIDATION,
                    priority=TaskPriority.MEDIUM,
                    parameters={'test': True},
                    dependencies=[],
                    estimated_duration=1.0,
                    resource_requirements={'cpu': 0.1, 'memory': 0.05},
                    deadline=None,
                    created_at=datetime.now()
                )
                
                scheduler.schedule_task(task)
                test_tasks.append(task)
            
            # Wait for execution
            time.sleep(5)
            
            # Get status
            status = scheduler.get_scheduler_status()
            
            metrics['scheduler_running'] = status['scheduler_running']
            metrics['tasks_scheduled'] = status['statistics']['total_scheduled'] >= 5
            metrics['tasks_completed'] = status['statistics']['total_completed'] > 0
            metrics['success_rate'] = status['statistics']['success_rate']
            metrics['efficiency'] = metrics['success_rate'] > 0.5
            
        finally:
            scheduler.stop_scheduler()
        
        return metrics
    
    def _validate_prediction_pipeline(self) -> Dict[str, Any]:
        """Validate end-to-end prediction pipeline"""
        
        metrics = {}
        
        if not self.market_data.empty and len(self.modules) >= 5:
            # Test prediction pipeline
            predictions_made = 0
            errors = 0
            
            for i in range(min(50, len(self.market_data))):
                try:
                    row = self.market_data.iloc[i]
                    
                    # Prepare market data
                    market_data_point = {
                        'regime_count': 12,
                        'volatility': row.get('realized_vol', 0.2),
                        'trend': row.get('trend', 0.0),
                        'volume_ratio': row.get('volume_ratio', 1.0)
                    }
                    
                    # ASL scoring (if available)
                    if ComponentType.ADAPTIVE_SCORING_LAYER in self.modules:
                        asl = self.modules[ComponentType.ADAPTIVE_SCORING_LAYER]
                        regime_scores = asl.calculate_regime_scores(market_data_point)
                        predictions_made += 1
                    
                except Exception as e:
                    errors += 1
                    logger.debug(f"Pipeline error: {e}")
            
            metrics['predictions_successful'] = predictions_made
            metrics['pipeline_errors'] = errors
            metrics['pipeline_success_rate'] = predictions_made / (predictions_made + errors) if (predictions_made + errors) > 0 else 0
            metrics['pipeline_functional'] = metrics['pipeline_success_rate'] > 0.8
        else:
            metrics['pipeline_functional'] = False
            metrics['error'] = "Insufficient modules or market data"
        
        return metrics
    
    def _validate_data_flow(self) -> Dict[str, Any]:
        """Validate cross-component data flow"""
        
        metrics = {}
        
        # Check module connectivity
        connected_pairs = []
        
        # ASL -> Transition Manager
        if (ComponentType.ADAPTIVE_SCORING_LAYER in self.modules and 
            ComponentType.INTELLIGENT_TRANSITION_MANAGER in self.modules):
            connected_pairs.append("asl_to_transition_manager")
        
        # Transition Analyzer -> Boundary Optimizer
        if (ComponentType.TRANSITION_MATRIX_ANALYZER in self.modules and 
            ComponentType.DYNAMIC_BOUNDARY_OPTIMIZER in self.modules):
            connected_pairs.append("analyzer_to_optimizer")
        
        # Stability Monitor -> Performance Feedback
        if (ComponentType.REGIME_STABILITY_MONITOR in self.modules and 
            ComponentType.PERFORMANCE_FEEDBACK_SYSTEM in self.modules):
            connected_pairs.append("monitor_to_feedback")
        
        metrics['connected_components'] = len(connected_pairs)
        metrics['data_flow_paths'] = connected_pairs
        metrics['integration_level'] = len(connected_pairs) / 3.0  # Normalize to ratio
        metrics['data_flow_adequate'] = metrics['integration_level'] > 0.5
        
        return metrics
    
    def _validate_feedback_loops(self) -> Dict[str, Any]:
        """Validate feedback loop integration"""
        
        metrics = {}
        
        feedback_loops_active = []
        
        # Performance -> Learning
        if (ComponentType.PERFORMANCE_FEEDBACK_SYSTEM in self.modules and 
            ComponentType.CONTINUOUS_LEARNING_ENGINE in self.modules):
            feedback_loops_active.append("performance_to_learning")
        
        # Learning -> ASL
        if (ComponentType.CONTINUOUS_LEARNING_ENGINE in self.modules and 
            ComponentType.ADAPTIVE_SCORING_LAYER in self.modules):
            feedback_loops_active.append("learning_to_asl")
        
        # Scheduler -> All Components
        if ComponentType.REGIME_OPTIMIZATION_SCHEDULER in self.modules:
            feedback_loops_active.append("scheduler_orchestration")
        
        metrics['active_feedback_loops'] = len(feedback_loops_active)
        metrics['feedback_loop_types'] = feedback_loops_active
        metrics['feedback_coverage'] = len(feedback_loops_active) / 3.0
        metrics['feedback_adequate'] = metrics['feedback_coverage'] > 0.5
        
        return metrics
    
    def _validate_system_latency(self) -> Dict[str, Any]:
        """Validate system latency benchmarks"""
        
        metrics = {}
        latency_tests = []
        
        # Test ASL latency
        if ComponentType.ADAPTIVE_SCORING_LAYER in self.modules:
            asl = self.modules[ComponentType.ADAPTIVE_SCORING_LAYER]
            
            start_time = time.time()
            for _ in range(100):
                scores = asl.calculate_regime_scores({'regime_count': 12})
            asl_latency = (time.time() - start_time) / 100 * 1000  # ms
            
            latency_tests.append(('asl_scoring_latency', asl_latency))
        
        # Evaluate against benchmarks
        for metric_name, measured_value in latency_tests:
            if metric_name in self.performance_benchmarks:
                benchmark = self.performance_benchmarks[metric_name]
                within_tolerance = abs(measured_value - benchmark.target_value) <= benchmark.tolerance
                metrics[f"{metric_name}_ms"] = measured_value
                metrics[f"{metric_name}_meets_target"] = within_tolerance
        
        metrics['latency_tests_passed'] = all(
            metrics.get(f"{name}_meets_target", False) 
            for name, _ in latency_tests
        )
        
        return metrics
    
    def _validate_resource_usage(self) -> Dict[str, Any]:
        """Validate resource usage limits"""
        
        import psutil
        import os
        
        metrics = {}
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # CPU usage
        cpu_percent = process.cpu_percent(interval=1.0) / 100.0
        metrics['cpu_usage'] = cpu_percent
        metrics['cpu_within_limit'] = cpu_percent < self.performance_benchmarks['cpu_usage_average'].target_value
        
        # Memory usage
        memory_info = process.memory_info()
        memory_percent = process.memory_percent() / 100.0
        metrics['memory_usage'] = memory_percent
        metrics['memory_within_limit'] = memory_percent < self.performance_benchmarks['memory_usage_peak'].target_value
        
        # Thread count
        thread_count = process.num_threads()
        metrics['thread_count'] = thread_count
        metrics['thread_count_reasonable'] = thread_count < 50
        
        metrics['resource_usage_acceptable'] = (
            metrics['cpu_within_limit'] and 
            metrics['memory_within_limit'] and 
            metrics['thread_count_reasonable']
        )
        
        return metrics
    
    def _validate_throughput(self) -> Dict[str, Any]:
        """Validate throughput benchmarks"""
        
        metrics = {}
        
        # Test data processing throughput
        if self.market_data is not None and not self.market_data.empty:
            sample_size = min(1000, len(self.market_data))
            
            start_time = time.time()
            processed = 0
            
            for i in range(sample_size):
                row = self.market_data.iloc[i]
                # Simulate processing
                processed += 1
            
            elapsed_time = time.time() - start_time
            throughput = processed / elapsed_time if elapsed_time > 0 else 0
            
            metrics['market_data_throughput'] = throughput
            metrics['throughput_meets_target'] = throughput > self.performance_benchmarks['market_data_throughput'].target_value
        else:
            metrics['throughput_meets_target'] = False
            metrics['error'] = "No market data available"
        
        return metrics
    
    def _validate_failover_recovery(self) -> Dict[str, Any]:
        """Validate failover and recovery mechanisms"""
        
        metrics = {}
        
        # Test module recovery
        recovery_tests = []
        
        # Test scheduler recovery
        if ComponentType.REGIME_OPTIMIZATION_SCHEDULER in self.modules:
            scheduler = self.modules[ComponentType.REGIME_OPTIMIZATION_SCHEDULER]
            
            # Start, stop, restart
            scheduler.start_scheduler()
            time.sleep(1)
            scheduler.stop_scheduler()
            time.sleep(1)
            scheduler.start_scheduler()
            time.sleep(1)
            
            status = scheduler.get_scheduler_status()
            scheduler_recovered = status['scheduler_running']
            recovery_tests.append(('scheduler', scheduler_recovered))
            
            scheduler.stop_scheduler()
        
        metrics['recovery_tests'] = recovery_tests
        metrics['recovery_success_rate'] = sum(1 for _, success in recovery_tests if success) / len(recovery_tests) if recovery_tests else 0
        metrics['failover_capability'] = metrics['recovery_success_rate'] > 0.8
        
        return metrics
    
    def _validate_data_persistence(self) -> Dict[str, Any]:
        """Validate data persistence and recovery"""
        
        metrics = {}
        persistence_tests = []
        
        # Test learning engine persistence
        if ComponentType.CONTINUOUS_LEARNING_ENGINE in self.modules:
            import tempfile
            
            learning_engine = self.modules[ComponentType.CONTINUOUS_LEARNING_ENGINE]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save models
                learning_engine.save_models(temp_dir)
                
                # Check if files were created
                saved_files = list(Path(temp_dir).glob("*.joblib"))
                persistence_tests.append(('learning_models_saved', len(saved_files) > 0))
                
                # Test loading
                try:
                    from ..optimization.continuous_learning_engine import ContinuousLearningEngine, LearningConfiguration
                    new_engine = ContinuousLearningEngine(LearningConfiguration())
                    new_engine.load_models(temp_dir)
                    persistence_tests.append(('learning_models_loaded', True))
                except:
                    persistence_tests.append(('learning_models_loaded', False))
        
        metrics['persistence_tests'] = persistence_tests
        metrics['persistence_success_rate'] = sum(1 for _, success in persistence_tests if success) / len(persistence_tests) if persistence_tests else 0
        metrics['data_persistence_functional'] = metrics['persistence_success_rate'] > 0.8
        
        return metrics
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security and access control"""
        
        metrics = {}
        
        # Basic security checks
        security_checks = []
        
        # Check for hardcoded credentials
        import inspect
        
        for component_type, module in self.modules.items():
            source = inspect.getsource(module.__class__)
            
            # Look for common credential patterns
            has_hardcoded_creds = any(
                pattern in source.lower() 
                for pattern in ['password=', 'api_key=', 'secret=']
            )
            
            security_checks.append((f"{component_type.value}_no_hardcoded_creds", not has_hardcoded_creds))
        
        # Check for secure random generation
        security_checks.append(('uses_secure_random', 'secrets' in str(self.modules) or 'random' in str(self.modules)))
        
        metrics['security_checks'] = security_checks
        metrics['security_check_pass_rate'] = sum(1 for _, passed in security_checks if passed) / len(security_checks) if security_checks else 0
        metrics['basic_security_adequate'] = metrics['security_check_pass_rate'] > 0.8
        
        return metrics
    
    def _validate_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring and alerting capabilities"""
        
        metrics = {}
        
        monitoring_capabilities = []
        
        # Check performance monitoring
        if ComponentType.PERFORMANCE_FEEDBACK_SYSTEM in self.modules:
            feedback_system = self.modules[ComponentType.PERFORMANCE_FEEDBACK_SYSTEM]
            report = feedback_system.generate_performance_report()
            monitoring_capabilities.append(('performance_monitoring', report is not None))
        
        # Check stability monitoring
        if ComponentType.REGIME_STABILITY_MONITOR in self.modules:
            monitor = self.modules[ComponentType.REGIME_STABILITY_MONITOR]
            stability_report = monitor.get_stability_report()
            monitoring_capabilities.append(('stability_monitoring', 'system_stability_score' in stability_report))
        
        # Check scheduler monitoring
        if ComponentType.REGIME_OPTIMIZATION_SCHEDULER in self.modules:
            scheduler = self.modules[ComponentType.REGIME_OPTIMIZATION_SCHEDULER]
            status = scheduler.get_scheduler_status()
            monitoring_capabilities.append(('task_monitoring', 'statistics' in status))
        
        metrics['monitoring_capabilities'] = monitoring_capabilities
        metrics['monitoring_coverage'] = sum(1 for _, active in monitoring_capabilities if active) / len(monitoring_capabilities) if monitoring_capabilities else 0
        metrics['monitoring_adequate'] = metrics['monitoring_coverage'] > 0.7
        
        return metrics
    
    def _check_data_completeness(self, data: pd.DataFrame) -> float:
        """Check data completeness"""
        
        if data is None or data.empty:
            return 0.0
        
        total_values = data.size
        non_null_values = data.count().sum()
        
        return non_null_values / total_values if total_values > 0 else 0.0
    
    def _check_data_consistency(self, data: pd.DataFrame) -> float:
        """Check data consistency"""
        
        if data is None or data.empty:
            return 0.0
        
        consistency_checks = []
        
        # Check price consistency
        if 'spot_price' in data.columns:
            valid_prices = (data['spot_price'] > 0).sum()
            consistency_checks.append(valid_prices / len(data))
        
        # Check volume consistency
        if 'volume' in data.columns:
            valid_volumes = (data['volume'] >= 0).sum()
            consistency_checks.append(valid_volumes / len(data))
        
        return np.mean(consistency_checks) if consistency_checks else 1.0
    
    def _check_data_timeliness(self, data: pd.DataFrame) -> float:
        """Check data timeliness"""
        
        if data is None or data.empty:
            return 0.0
        
        if 'timestamp' in data.columns:
            try:
                timestamps = pd.to_datetime(data['timestamp'])
                # Check if timestamps are sequential
                time_diffs = timestamps.diff().dropna()
                consistent_intervals = (time_diffs == time_diffs.mode()[0]).sum()
                return consistent_intervals / len(time_diffs) if len(time_diffs) > 0 else 1.0
            except:
                return 0.0
        
        return 1.0
    
    def _check_data_accuracy(self, data: pd.DataFrame) -> float:
        """Check data accuracy"""
        
        if data is None or data.empty:
            return 0.0
        
        accuracy_checks = []
        
        # Check for reasonable values
        if 'volatility' in data.columns:
            reasonable_vol = ((data['volatility'] >= 0) & (data['volatility'] < 2.0)).sum()
            accuracy_checks.append(reasonable_vol / len(data))
        
        if 'returns' in data.columns:
            reasonable_returns = ((data['returns'] > -0.5) & (data['returns'] < 0.5)).sum()
            accuracy_checks.append(reasonable_returns / len(data))
        
        return np.mean(accuracy_checks) if accuracy_checks else 1.0
    
    def _check_data_uniqueness(self, data: pd.DataFrame) -> float:
        """Check data uniqueness"""
        
        if data is None or data.empty:
            return 0.0
        
        total_rows = len(data)
        unique_rows = len(data.drop_duplicates())
        
        return unique_rows / total_rows if total_rows > 0 else 1.0
    
    def _validate_data_quality(self) -> Dict[str, Any]:
        """Validate overall data quality"""
        
        metrics = {}
        
        if self.market_data is not None:
            for check in self.data_quality_checks:
                score = check.validation_function(self.market_data)
                metrics[check.check_name] = {
                    'score': score,
                    'passed': score >= check.threshold,
                    'threshold': check.threshold
                }
        
        # Overall data quality score
        if metrics:
            scores = [m['score'] for m in metrics.values()]
            metrics['overall_score'] = np.mean(scores)
            metrics['overall_passed'] = all(m['passed'] for m in metrics.values())
        else:
            metrics['overall_score'] = 0.0
            metrics['overall_passed'] = False
        
        return metrics
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        
        metrics = {}
        
        # Aggregate test execution times
        execution_times = [r.execution_time for r in self.test_results]
        if execution_times:
            metrics['avg_test_execution_time'] = np.mean(execution_times)
            metrics['max_test_execution_time'] = np.max(execution_times)
            metrics['total_validation_time'] = sum(execution_times)
        
        # Component coverage
        tested_components = set(r.component for r in self.test_results)
        metrics['components_tested'] = len(tested_components)
        metrics['component_coverage'] = len(tested_components) / len(ComponentType)
        
        # Test success rates by component
        component_success = defaultdict(lambda: {'passed': 0, 'total': 0})
        for result in self.test_results:
            component_success[result.component]['total'] += 1
            if result.status == ValidationStatus.PASSED:
                component_success[result.component]['passed'] += 1
        
        metrics['component_success_rates'] = {
            comp.value: data['passed'] / data['total'] if data['total'] > 0 else 0.0
            for comp, data in component_success.items()
        }
        
        return metrics
    
    def _calculate_integration_metrics(self) -> Dict[str, Any]:
        """Calculate integration quality metrics"""
        
        metrics = {}
        
        # Integration test results
        integration_tests = [
            r for r in self.test_results 
            if r.component == ComponentType.INTEGRATED_SYSTEM
        ]
        
        if integration_tests:
            passed = sum(1 for r in integration_tests if r.status == ValidationStatus.PASSED)
            metrics['integration_test_pass_rate'] = passed / len(integration_tests)
            metrics['integration_test_count'] = len(integration_tests)
        else:
            metrics['integration_test_pass_rate'] = 0.0
            metrics['integration_test_count'] = 0
        
        # Cross-component dependencies
        dependency_count = sum(
            len(test.dependencies) 
            for test in self.validation_tests.values()
        )
        metrics['total_dependencies'] = dependency_count
        metrics['avg_dependencies_per_test'] = dependency_count / len(self.validation_tests) if self.validation_tests else 0
        
        return metrics
    
    def _sort_tests_by_dependencies(self, tests: List[ValidationTest]) -> List[ValidationTest]:
        """Sort tests by dependencies using topological sort"""
        
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        test_map = {test.test_id: test for test in tests}
        
        for test in tests:
            in_degree[test.test_id] = len(test.dependencies)
            for dep in test.dependencies:
                if dep in test_map:
                    graph[dep].append(test.test_id)
        
        # Topological sort
        queue = [test_id for test_id, degree in in_degree.items() if degree == 0]
        sorted_tests = []
        
        while queue:
            test_id = queue.pop(0)
            sorted_tests.append(test_map[test_id])
            
            for dependent in graph[test_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Add any remaining tests (cycles)
        for test in tests:
            if test not in sorted_tests:
                sorted_tests.append(test)
        
        return sorted_tests
    
    def _check_test_dependencies(self, test: ValidationTest) -> bool:
        """Check if test dependencies are satisfied"""
        
        for dep_id in test.dependencies:
            # Check if dependency test passed
            dep_results = [
                r for r in self.test_results 
                if r.test_id == dep_id and r.status == ValidationStatus.PASSED
            ]
            
            if not dep_results:
                return False
        
        return True
    
    def _evaluate_test_results(self, metrics: Dict[str, Any], 
                             expected: Dict[str, Any]) -> Tuple[ValidationStatus, List[str], List[str]]:
        """Evaluate test results against expected metrics"""
        
        warnings = []
        recommendations = []
        
        # If no expected metrics, just check if test completed
        if not expected:
            return ValidationStatus.PASSED, warnings, recommendations
        
        # Compare against expected values
        all_passed = True
        
        for key, expected_value in expected.items():
            if key not in metrics:
                warnings.append(f"Missing expected metric: {key}")
                all_passed = False
                continue
            
            actual_value = metrics[key]
            
            # Boolean comparison
            if isinstance(expected_value, bool):
                if actual_value != expected_value:
                    all_passed = False
                    recommendations.append(f"Expected {key}={expected_value}, got {actual_value}")
            
            # Numeric comparison with tolerance
            elif isinstance(expected_value, (int, float)):
                tolerance = 0.1 * abs(expected_value)  # 10% tolerance
                if abs(actual_value - expected_value) > tolerance:
                    warnings.append(f"{key}: {actual_value} outside tolerance of {expected_value}")
        
        if all_passed and not warnings:
            return ValidationStatus.PASSED, warnings, recommendations
        elif all_passed or len(warnings) < len(expected) / 2:
            return ValidationStatus.WARNING, warnings, recommendations
        else:
            return ValidationStatus.FAILED, warnings, recommendations
    
    def _generate_validation_report(self, data_quality_metrics: Dict[str, Any],
                                   performance_metrics: Dict[str, Any],
                                   integration_metrics: Dict[str, Any]) -> ValidationReport:
        """Generate comprehensive validation report"""
        
        # Count test results
        passed = sum(1 for r in self.test_results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in self.test_results if r.status == ValidationStatus.WARNING)
        skipped = sum(1 for r in self.test_results if r.status == ValidationStatus.SKIPPED)
        
        # Determine overall status
        if failed > 0:
            overall_status = ValidationStatus.FAILED
        elif warnings > passed / 2:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        # Group results by component
        component_results = defaultdict(list)
        for result in self.test_results:
            component_results[result.component].append(result)
        
        # Calculate production readiness score
        readiness_factors = []
        
        # Test pass rate
        test_pass_rate = passed / len(self.test_results) if self.test_results else 0
        readiness_factors.append(test_pass_rate)
        
        # Data quality
        if data_quality_metrics.get('overall_passed'):
            readiness_factors.append(1.0)
        else:
            readiness_factors.append(data_quality_metrics.get('overall_score', 0.0))
        
        # Performance metrics
        if performance_metrics.get('component_coverage', 0) > 0.8:
            readiness_factors.append(1.0)
        else:
            readiness_factors.append(performance_metrics.get('component_coverage', 0.0))
        
        # Integration quality
        readiness_factors.append(integration_metrics.get('integration_test_pass_rate', 0.0))
        
        production_readiness_score = np.mean(readiness_factors)
        
        # Generate recommendations
        recommendations = []
        
        if production_readiness_score < 0.8:
            recommendations.append("System needs improvement before production deployment")
        
        if failed > 0:
            failed_components = set(
                r.component.value for r in self.test_results 
                if r.status == ValidationStatus.FAILED
            )
            recommendations.append(f"Fix failures in: {', '.join(failed_components)}")
        
        if data_quality_metrics.get('overall_score', 0) < 0.9:
            recommendations.append("Improve data quality validation")
        
        if performance_metrics.get('component_coverage', 0) < 1.0:
            recommendations.append("Increase test coverage for all components")
        
        # Generate report ID
        report_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        return ValidationReport(
            report_id=report_id,
            timestamp=datetime.now(),
            validation_level=self.validation_level,
            total_tests=len(self.test_results),
            passed_tests=passed,
            failed_tests=failed,
            warning_tests=warnings,
            skipped_tests=skipped,
            overall_status=overall_status,
            component_results=dict(component_results),
            performance_metrics=performance_metrics,
            data_quality_metrics=data_quality_metrics,
            integration_metrics=integration_metrics,
            production_readiness_score=production_readiness_score,
            recommendations=recommendations,
            detailed_results=self.test_results
        )
    
    def export_validation_report(self, report: ValidationReport, filepath: str):
        """Export validation report to file"""
        
        export_data = {
            'report_id': report.report_id,
            'timestamp': report.timestamp.isoformat(),
            'validation_level': report.validation_level.value,
            'summary': {
                'total_tests': report.total_tests,
                'passed': report.passed_tests,
                'failed': report.failed_tests,
                'warnings': report.warning_tests,
                'skipped': report.skipped_tests,
                'overall_status': report.overall_status.value,
                'production_readiness_score': report.production_readiness_score
            },
            'component_results': {
                comp.value: [
                    {
                        'test_id': r.test_id,
                        'test_name': r.test_name,
                        'status': r.status.value,
                        'execution_time': r.execution_time,
                        'error': r.error_message
                    }
                    for r in results
                ]
                for comp, results in report.component_results.items()
            },
            'metrics': {
                'performance': report.performance_metrics,
                'data_quality': report.data_quality_metrics,
                'integration': report.integration_metrics
            },
            'recommendations': report.recommendations
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Validation report exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create validator
    validator = SystemValidator(validation_level=ValidationLevel.COMPREHENSIVE)
    
    # Mock modules for testing
    class MockModule:
        def __init__(self, name):
            self.name = name
            self.weights = {i: 0.2 for i in range(5)}
            self.config = type('Config', (), {'min_weight': 0.05, 'max_weight': 0.8})()
        
        def calculate_regime_scores(self, data):
            return {i: np.random.random() for i in range(12)}
        
        def update_weights_based_on_performance(self, scores, actual):
            # Simulate weight update
            for key in self.weights:
                self.weights[key] += np.random.uniform(-0.01, 0.01)
    
    # Set mock modules
    modules = {
        ComponentType.ADAPTIVE_SCORING_LAYER: MockModule("ASL"),
        ComponentType.TRANSITION_MATRIX_ANALYZER: MockModule("TMA"),
        ComponentType.PERFORMANCE_FEEDBACK_SYSTEM: MockModule("PFS")
    }
    
    validator.set_modules(modules)
    
    # Create mock market data
    market_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='5min'),
        'spot_price': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 5000, 100),
        'volatility': np.random.uniform(0.1, 0.3, 100),
        'trend': np.random.uniform(-0.02, 0.02, 100),
        'volume_ratio': np.random.uniform(0.5, 2.0, 100)
    })
    
    validator.set_market_data(market_data)
    
    # Run validation
    report = validator.validate_system()
    
    # Display results
    print(f"Validation Report: {report.report_id}")
    print(f"Overall Status: {report.overall_status.value}")
    print(f"Production Readiness: {report.production_readiness_score:.2%}")
    print(f"Tests: {report.passed_tests}/{report.total_tests} passed")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")