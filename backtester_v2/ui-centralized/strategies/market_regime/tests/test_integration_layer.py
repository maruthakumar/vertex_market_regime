"""
Integration Layer Test Suite
===========================

Comprehensive tests for integration layer components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import logging

# Import integration components
from ..integration.market_regime_orchestrator import MarketRegimeOrchestrator
from ..integration.component_manager import ComponentManager
from ..integration.data_pipeline import DataPipeline
from ..integration.result_aggregator import ResultAggregator


class TestMarketRegimeOrchestrator(unittest.TestCase):
    """Test MarketRegimeOrchestrator functionality"""
    
    def setUp(self):
        self.config = {
            'execution_mode': 'parallel',
            'timeout': 30.0,
            'component_weights': {
                'straddle_analysis': 0.25,
                'oi_pa_analysis': 0.20,
                'greek_sentiment': 0.15,
                'market_breadth': 0.25,
                'iv_analytics': 0.10,
                'technical_indicators': 0.05
            }
        }
        
        # Create sample input data
        self.sample_data = {
            'option_data': pd.DataFrame({
                'strike': [100, 105, 110],
                'option_type': ['CE', 'PE', 'CE'],
                'volume': [100, 200, 150],
                'oi': [1000, 2000, 1500]
            }),
            'underlying_data': pd.DataFrame({
                'close': [110, 111, 112],
                'volume': [10000, 12000, 8000]
            })
        }
    
    @patch('..integration.market_regime_orchestrator.ComponentManager')
    @patch('..integration.market_regime_orchestrator.DataPipeline')
    @patch('..integration.market_regime_orchestrator.ResultAggregator')
    def test_orchestrator_initialization(self, mock_aggregator, mock_pipeline, mock_manager):
        """Test orchestrator initialization"""
        orchestrator = MarketRegimeOrchestrator(self.config)
        
        self.assertIsNotNone(orchestrator.component_manager)
        self.assertIsNotNone(orchestrator.data_pipeline)
        self.assertIsNotNone(orchestrator.result_aggregator)
        self.assertEqual(orchestrator.execution_mode, 'parallel')
    
    @patch('..integration.market_regime_orchestrator.ComponentManager')
    @patch('..integration.market_regime_orchestrator.DataPipeline')
    @patch('..integration.market_regime_orchestrator.ResultAggregator')
    def test_orchestrate_analysis_success(self, mock_aggregator, mock_pipeline, mock_manager):
        """Test successful analysis orchestration"""
        # Setup mocks
        mock_manager_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_aggregator_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_pipeline.return_value = mock_pipeline_instance
        mock_aggregator.return_value = mock_aggregator_instance
        
        # Mock processed data
        mock_pipeline_instance.process_data.return_value = {
            'processed_data': self.sample_data,
            'status': 'success'
        }
        
        # Mock component execution results
        mock_manager_instance.execute_component.return_value = {
            'status': 'success',
            'composite_score': 0.75,
            'regime_classification': 'bullish'
        }
        
        # Mock aggregated results
        mock_aggregator_instance.aggregate_results.return_value = {
            'primary_aggregation': {'aggregated_score': 0.70},
            'status': 'success'
        }
        
        orchestrator = MarketRegimeOrchestrator(self.config)
        result = orchestrator.orchestrate_analysis(self.sample_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('processed_data', result)
        self.assertIn('component_results', result)
        self.assertIn('aggregated_results', result)
    
    @patch('..integration.market_regime_orchestrator.ComponentManager')
    @patch('..integration.market_regime_orchestrator.DataPipeline')
    @patch('..integration.market_regime_orchestrator.ResultAggregator')
    def test_orchestrate_analysis_with_component_failure(self, mock_aggregator, mock_pipeline, mock_manager):
        """Test analysis orchestration with component failure"""
        # Setup mocks
        mock_manager_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_aggregator_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_pipeline.return_value = mock_pipeline_instance
        mock_aggregator.return_value = mock_aggregator_instance
        
        # Mock processed data
        mock_pipeline_instance.process_data.return_value = {
            'processed_data': self.sample_data,
            'status': 'success'
        }
        
        # Mock component execution with some failures
        def mock_execute_component(component_name, data, timeout=None):
            if component_name == 'straddle_analysis':
                return {'status': 'error', 'error': 'Component failed'}
            else:
                return {
                    'status': 'success',
                    'composite_score': 0.75,
                    'regime_classification': 'bullish'
                }
        
        mock_manager_instance.execute_component.side_effect = mock_execute_component
        
        # Mock aggregated results
        mock_aggregator_instance.aggregate_results.return_value = {
            'primary_aggregation': {'aggregated_score': 0.60},
            'status': 'partial_success'
        }
        
        orchestrator = MarketRegimeOrchestrator(self.config)
        result = orchestrator.orchestrate_analysis(self.sample_data)
        
        self.assertEqual(result['status'], 'partial_success')
        self.assertIn('failed_components', result)


class TestComponentManager(unittest.TestCase):
    """Test ComponentManager functionality"""
    
    def setUp(self):
        self.config = {
            'auto_load_components': False,
            'component_timeout': 30.0,
            'health_check_interval': 300
        }
        
        # Mock component class
        self.mock_component_class = Mock()
        self.mock_component_instance = Mock()
        self.mock_component_class.return_value = self.mock_component_instance
        
        # Setup mock component instance
        self.mock_component_instance.analyze.return_value = {
            'status': 'success',
            'composite_score': 0.75
        }
        self.mock_component_instance.get_health_status.return_value = {
            'status': 'healthy'
        }
        self.mock_component_instance.get_metadata.return_value = {
            'version': '2.0.0'
        }
    
    def test_component_manager_initialization(self):
        """Test component manager initialization"""
        manager = ComponentManager(self.config)
        
        self.assertEqual(manager.component_timeout, 30.0)
        self.assertEqual(manager.health_check_interval, 300)
        self.assertFalse(manager.auto_load_components)
    
    def test_register_component_success(self):
        """Test successful component registration"""
        manager = ComponentManager(self.config)
        
        result = manager.register_component(
            'test_component',
            self.mock_component_class,
            'test_category',
            dependencies=[],
            config={'param1': 'value1'}
        )
        
        self.assertTrue(result)
        self.assertIn('test_component', manager.component_registry)
        self.assertEqual(manager.component_registry['test_component']['category'], 'test_category')
    
    def test_load_component_success(self):
        """Test successful component loading"""
        manager = ComponentManager(self.config)
        
        # Register component first
        manager.register_component('test_component', self.mock_component_class)
        
        # Load component
        result = manager.load_component('test_component')
        
        self.assertTrue(result)
        self.assertIn('test_component', manager.component_instances)
    
    def test_execute_component_success(self):
        """Test successful component execution"""
        manager = ComponentManager(self.config)
        
        # Register and load component
        manager.register_component('test_component', self.mock_component_class)
        manager.load_component('test_component')
        
        # Execute component
        data = {'test_data': 'value'}
        result = manager.execute_component('test_component', data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('execution_metadata', result)
        self.mock_component_instance.analyze.assert_called_once_with(data)
    
    def test_perform_health_checks(self):
        """Test health checks functionality"""
        manager = ComponentManager(self.config)
        
        # Register and load component
        manager.register_component('test_component', self.mock_component_class)
        manager.load_component('test_component')
        
        # Perform health checks
        health_results = manager.perform_health_checks()
        
        self.assertEqual(health_results['overall_health'], 'healthy')
        self.assertIn('test_component', health_results['component_health'])
    
    def test_unload_component(self):
        """Test component unloading"""
        manager = ComponentManager(self.config)
        
        # Register and load component
        manager.register_component('test_component', self.mock_component_class)
        manager.load_component('test_component')
        
        # Unload component
        result = manager.unload_component('test_component')
        
        self.assertTrue(result)
        self.assertNotIn('test_component', manager.component_instances)


class TestDataPipeline(unittest.TestCase):
    """Test DataPipeline functionality"""
    
    def setUp(self):
        self.config = {
            'cache_enabled': True,
            'cache_ttl': 300,
            'quality_thresholds': {
                'min_data_points': 10,
                'max_missing_ratio': 0.2,
                'min_quality_score': 0.6
            }
        }
        
        # Sample data
        self.sample_data = {
            'option_data': pd.DataFrame({
                'strike': [100, 105, 110, 115, 120],
                'option_type': ['CE', 'PE', 'CE', 'PE', 'CE'],
                'volume': [100, 200, 150, 300, 50],
                'oi': [1000, 2000, 1500, 3000, 500],
                'spot': [110, 110, 110, 110, 110]
            }),
            'underlying_data': pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [102, 103, 104, 105, 106],
                'low': [99, 100, 101, 102, 103],
                'close': [101, 102, 103, 104, 105],
                'volume': [1000, 1200, 800, 1500, 900]
            })
        }
    
    def test_data_pipeline_initialization(self):
        """Test data pipeline initialization"""
        pipeline = DataPipeline(self.config)
        
        self.assertTrue(pipeline.cache_enabled)
        self.assertEqual(pipeline.cache_ttl, 300)
        self.assertIn('option', pipeline.processors)
        self.assertIn('underlying', pipeline.processors)
    
    def test_process_data_success(self):
        """Test successful data processing"""
        pipeline = DataPipeline(self.config)
        
        result = pipeline.process_data(self.sample_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('processed_data', result)
        self.assertIn('validation_results', result)
        self.assertIn('data_quality', result)
        self.assertIn('overall_quality', result)
        
        # Check that processed data contains expected data types
        self.assertIn('option_data', result['processed_data'])
        self.assertIn('underlying_data', result['processed_data'])
    
    def test_data_validation(self):
        """Test data validation functionality"""
        pipeline = DataPipeline(self.config)
        
        # Test with valid data
        validation_result = pipeline._validate_input_data(
            self.sample_data['option_data'], 'option'
        )
        self.assertTrue(validation_result['is_valid'])
        
        # Test with invalid data (empty DataFrame)
        empty_data = pd.DataFrame()
        validation_result = pipeline._validate_input_data(empty_data, 'option')
        self.assertFalse(validation_result['is_valid'])
    
    def test_data_quality_calculation(self):
        """Test data quality metrics calculation"""
        pipeline = DataPipeline(self.config)
        
        quality_metrics = pipeline._calculate_data_quality(
            self.sample_data['option_data'], 'option'
        )
        
        self.assertIn('overall_score', quality_metrics)
        self.assertIn('completeness', quality_metrics)
        self.assertIn('consistency', quality_metrics)
        self.assertIn('accuracy', quality_metrics)
        
        # Quality scores should be between 0 and 1
        self.assertGreaterEqual(quality_metrics['overall_score'], 0)
        self.assertLessEqual(quality_metrics['overall_score'], 1)
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        pipeline = DataPipeline(self.config)
        
        # First call should process data
        result1 = pipeline.process_data(self.sample_data)
        
        # Second call should return cached result (if caching is working)
        with patch.object(pipeline, '_process_data_type') as mock_process:
            result2 = pipeline.process_data(self.sample_data)
            
            # If cache is working, _process_data_type should not be called again
            if pipeline.cache_enabled:
                mock_process.assert_not_called()


class TestResultAggregator(unittest.TestCase):
    """Test ResultAggregator functionality"""
    
    def setUp(self):
        self.config = {
            'default_strategy': 'weighted_average',
            'component_weights': {
                'component1': 0.4,
                'component2': 0.3,
                'component3': 0.3
            },
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }
        
        # Sample component results
        self.sample_results = {
            'component1': {
                'composite_score': 0.75,
                'regime_classification': 'bullish',
                'confidence': 0.8,
                'status': 'success'
            },
            'component2': {
                'composite_score': 0.65,
                'regime_classification': 'bullish',
                'confidence': 0.7,
                'status': 'success'
            },
            'component3': {
                'composite_score': 0.55,
                'regime_classification': 'neutral',
                'confidence': 0.6,
                'status': 'success'
            }
        }
    
    def test_result_aggregator_initialization(self):
        """Test result aggregator initialization"""
        aggregator = ResultAggregator(self.config)
        
        self.assertEqual(aggregator.default_strategy, 'weighted_average')
        self.assertIn('weighted_average', aggregator.aggregation_strategies)
        self.assertIn('ensemble', aggregator.aggregation_strategies)
    
    def test_aggregate_results_weighted_average(self):
        """Test result aggregation using weighted average strategy"""
        aggregator = ResultAggregator(self.config)
        
        result = aggregator.aggregate_results(
            self.sample_results,
            aggregation_strategy='weighted_average'
        )
        
        self.assertEqual(result['primary_aggregation']['aggregation_method'], 'weighted_average')
        self.assertIn('aggregated_score', result['primary_aggregation'])
        self.assertIn('confidence_analysis', result)
        self.assertIn('consensus_analysis', result)
        self.assertIn('quality_metrics', result)
    
    def test_aggregate_results_ensemble_strategy(self):
        """Test result aggregation using ensemble strategy"""
        aggregator = ResultAggregator(self.config)
        
        result = aggregator.aggregate_results(
            self.sample_results,
            aggregation_strategy='ensemble'
        )
        
        self.assertEqual(result['primary_aggregation']['aggregation_method'], 'ensemble_voting')
        self.assertIn('regime_classification', result['primary_aggregation'])
        self.assertIn('regime_confidence', result['primary_aggregation'])
    
    def test_confidence_analysis(self):
        """Test confidence analysis calculation"""
        aggregator = ResultAggregator(self.config)
        
        result = aggregator.aggregate_results(self.sample_results)
        confidence_analysis = result['confidence_analysis']
        
        self.assertIn('overall_confidence', confidence_analysis)
        self.assertIn('component_agreement', confidence_analysis)
        self.assertIn('confidence_level', confidence_analysis)
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(confidence_analysis['overall_confidence'], 0)
        self.assertLessEqual(confidence_analysis['overall_confidence'], 1)
    
    def test_consensus_analysis(self):
        """Test consensus analysis"""
        aggregator = ResultAggregator(self.config)
        
        result = aggregator.aggregate_results(self.sample_results)
        consensus_analysis = result['consensus_analysis']
        
        self.assertIn('consensus_score', consensus_analysis)
        self.assertIn('majority_view', consensus_analysis)
        self.assertIn('dissenting_components', consensus_analysis)
        self.assertIn('consensus_strength', consensus_analysis)
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        aggregator = ResultAggregator(self.config)
        
        # Add an outlier component
        outlier_results = self.sample_results.copy()
        outlier_results['outlier_component'] = {
            'composite_score': 0.1,  # Very low score
            'regime_classification': 'bearish',
            'confidence': 0.9,
            'status': 'success'
        }
        
        result = aggregator.aggregate_results(outlier_results)
        anomaly_detection = result['anomaly_detection']
        
        self.assertIn('outlier_components', anomaly_detection)
        self.assertIn('anomaly_score', anomaly_detection)
        self.assertGreater(anomaly_detection['anomaly_score'], 0)
    
    def test_quality_metrics(self):
        """Test quality metrics calculation"""
        aggregator = ResultAggregator(self.config)
        
        result = aggregator.aggregate_results(self.sample_results)
        quality_metrics = result['quality_metrics']
        
        self.assertIn('overall_quality', quality_metrics)
        self.assertIn('component_quality', quality_metrics)
        self.assertIn('data_completeness', quality_metrics)
        self.assertIn('quality_grade', quality_metrics)
        
        # Quality should be between 0 and 1
        self.assertGreaterEqual(quality_metrics['overall_quality'], 0)
        self.assertLessEqual(quality_metrics['overall_quality'], 1)
    
    def test_insights_generation(self):
        """Test insights generation"""
        aggregator = ResultAggregator(self.config)
        
        result = aggregator.aggregate_results(self.sample_results)
        insights = result['insights']
        
        self.assertIn('key_insights', insights)
        self.assertIn('recommendations', insights)
        self.assertIn('risk_factors', insights)
        self.assertIn('opportunities', insights)
        
        # Should have some insights
        self.assertGreater(len(insights['key_insights']), 0)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMarketRegimeOrchestrator,
        TestComponentManager,
        TestDataPipeline,
        TestResultAggregator
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Integration Layer Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")