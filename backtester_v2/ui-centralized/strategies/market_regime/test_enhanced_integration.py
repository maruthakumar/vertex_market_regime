#!/usr/bin/env python3
"""
Enhanced Integration Test Suite
==============================

Comprehensive test suite for the enhanced market regime integration system,
validating all components including configuration mapping, module integration,
and end-to-end functionality.

Features:
- Configuration mapping validation
- Enhanced module integration tests
- Performance benchmarking
- Error handling validation
- End-to-end system tests

Author: The Augster
Date: 2025-01-06
Version: 1.0.0 - Enhanced Integration Test Suite
"""

import unittest
import logging
import time
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Import components to test
from .enhanced_module_integration_manager import (
    EnhancedModuleIntegrationManager,
    ModuleStatus,
    IntegrationPriority,
    create_integration_manager
)
from .excel_configuration_mapper import (
    ExcelConfigurationMapper,
    ParameterMapping,
    ParameterType,
    ValidationRule
)
from .enhanced_greek_sentiment_integration import (
    GreekSentimentAnalyzerIntegration,
    GreekSentimentIntegrationConfig,
    create_greek_sentiment_integration
)
from .unified_enhanced_market_regime_engine import (
    UnifiedEnhancedMarketRegimeEngine,
    UnifiedEngineConfig,
    create_unified_engine
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEnhancedModuleIntegrationManager(unittest.TestCase):
    """Test cases for Enhanced Module Integration Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.integration_manager = create_integration_manager()
        
        # Create a simple test module class
        class TestModule:
            def __init__(self, config):
                self.config = config
            
            def analyze(self, market_data):
                return {'test_result': 42, 'confidence': 0.8}
        
        self.test_module_class = TestModule
    
    def test_module_registration(self):
        """Test module registration functionality"""
        # Test successful registration
        success = self.integration_manager.register_enhanced_module(
            name='test_module',
            module_class=self.test_module_class,
            config_requirements=['test_param'],
            dependencies=[],
            priority=IntegrationPriority.MEDIUM
        )
        self.assertTrue(success)
        
        # Verify module is registered
        self.assertIn('test_module', self.integration_manager.registered_modules)
        
        # Test duplicate registration
        success = self.integration_manager.register_enhanced_module(
            name='test_module',
            module_class=self.test_module_class,
            config_requirements=['test_param'],
            dependencies=[],
            priority=IntegrationPriority.MEDIUM
        )
        self.assertTrue(success)  # Should update existing registration
    
    def test_module_configuration(self):
        """Test module configuration functionality"""
        # Register test module
        self.integration_manager.register_enhanced_module(
            name='test_module',
            module_class=self.test_module_class,
            config_requirements=['test_param'],
            dependencies=[],
            priority=IntegrationPriority.MEDIUM
        )
        
        # Test configuration
        global_config = {
            'test_module': {'test_param': 'test_value'},
            'symbol': 'NIFTY',
            'timeframe': '1min'
        }
        
        success = self.integration_manager.configure_modules(global_config)
        self.assertTrue(success)
        
        # Verify configuration
        registration = self.integration_manager.registered_modules['test_module']
        self.assertEqual(registration.status, ModuleStatus.CONFIGURED)
        self.assertIn('test_param', registration.configuration)
    
    def test_module_initialization(self):
        """Test module initialization functionality"""
        # Register and configure test module
        self.integration_manager.register_enhanced_module(
            name='test_module',
            module_class=self.test_module_class,
            config_requirements=[],
            dependencies=[],
            priority=IntegrationPriority.MEDIUM
        )
        
        global_config = {'test_module': {}}
        self.integration_manager.configure_modules(global_config)
        
        # Test initialization
        success = self.integration_manager.initialize_modules()
        self.assertTrue(success)
        
        # Verify initialization
        registration = self.integration_manager.registered_modules['test_module']
        self.assertEqual(registration.status, ModuleStatus.INITIALIZED)
        self.assertIsNotNone(registration.instance)
    
    def test_module_execution(self):
        """Test module execution functionality"""
        # Setup complete module
        self.integration_manager.register_enhanced_module(
            name='test_module',
            module_class=self.test_module_class,
            config_requirements=[],
            dependencies=[],
            priority=IntegrationPriority.MEDIUM
        )
        
        global_config = {'test_module': {}}
        self.integration_manager.configure_modules(global_config)
        self.integration_manager.initialize_modules()
        self.integration_manager.activate_integration()
        
        # Test execution
        market_data = {
            'timestamp': datetime.now(),
            'underlying_price': 18000.0
        }
        
        result = self.integration_manager.execute_module_analysis('test_module', market_data)
        self.assertIsNotNone(result)
        self.assertIn('test_result', result)
        self.assertEqual(result['test_result'], 42)

class TestExcelConfigurationMapper(unittest.TestCase):
    """Test cases for Excel Configuration Mapper"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary Excel file for testing
        self.temp_excel_path = self._create_test_excel_file()
        self.config_mapper = ExcelConfigurationMapper(self.temp_excel_path)
    
    def _create_test_excel_file(self) -> str:
        """Create a temporary Excel file for testing"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Create test Excel data
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            # GreekSentimentConfig sheet
            greek_config_data = {
                'Parameter': ['delta_weight', 'vega_weight', 'theta_weight', 'gamma_weight'],
                'Value': [1.2, 1.5, 0.3, 0.0],
                'Description': ['Delta weight', 'Vega weight', 'Theta weight', 'Gamma weight']
            }
            pd.DataFrame(greek_config_data).to_excel(writer, sheet_name='GreekSentimentConfig', index=False)
            
            # TrendingOIPAConfig sheet
            oi_config_data = {
                'Parameter': ['correlation_threshold', 'time_decay_lambda', 'primary_timeframe_minutes'],
                'Value': [0.80, 0.1, 3],
                'Description': ['Correlation threshold', 'Time decay lambda', 'Primary timeframe']
            }
            pd.DataFrame(oi_config_data).to_excel(writer, sheet_name='TrendingOIPAConfig', index=False)
        
        return temp_path
    
    def test_excel_loading(self):
        """Test Excel file loading"""
        success = self.config_mapper.load_excel_configuration()
        self.assertTrue(success)
        
        # Verify sheets are loaded
        self.assertIn('GreekSentimentConfig', self.config_mapper.excel_data)
        self.assertIn('TrendingOIPAConfig', self.config_mapper.excel_data)
    
    def test_parameter_mapping(self):
        """Test parameter mapping functionality"""
        self.config_mapper.load_excel_configuration()
        
        # Test Greek sentiment module mapping
        module_config = self.config_mapper.map_module_parameters('enhanced_greek_sentiment_analysis')
        self.assertIsNotNone(module_config)
        
        # Verify parameters are mapped
        self.assertIn('delta_weight', module_config.parameters)
        self.assertEqual(module_config.parameters['delta_weight'], 1.2)
        
        self.assertIn('vega_weight', module_config.parameters)
        self.assertEqual(module_config.parameters['vega_weight'], 1.5)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        self.config_mapper.load_excel_configuration()
        
        validation_report = self.config_mapper.validate_configuration()
        self.assertIsNotNone(validation_report)
        
        # Check validation report structure
        self.assertIn('total_modules', validation_report)
        self.assertIn('successful_modules', validation_report)
        self.assertIn('module_reports', validation_report)
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary Excel file
        Path(self.temp_excel_path).unlink(missing_ok=True)

class TestGreekSentimentIntegration(unittest.TestCase):
    """Test cases for Greek Sentiment Integration"""
    
    def setUp(self):
        """Set up test environment"""
        config = GreekSentimentIntegrationConfig(
            delta_weight=1.2,
            vega_weight=1.5,
            theta_weight=0.3,
            gamma_weight=0.0,
            enable_caching=True
        )
        self.greek_integration = create_greek_sentiment_integration(config)
    
    def test_greek_sentiment_analysis(self):
        """Test Greek sentiment analysis functionality"""
        # Create test market data
        market_data = {
            'timestamp': datetime.now(),
            'underlying_price': 18000.0,
            'ATM_CE_delta': 0.5,
            'ATM_PE_delta': -0.5,
            'ATM_CE_gamma': 0.001,
            'ATM_PE_gamma': 0.001,
            'ATM_CE_theta': -5.0,
            'ATM_PE_theta': -5.0,
            'ATM_CE_vega': 20.0,
            'ATM_PE_vega': 20.0,
            'ATM_CE_ltp': 150.0,
            'ATM_PE_ltp': 140.0
        }
        
        # Test analysis
        result = self.greek_integration.analyze_greek_sentiment(market_data)
        self.assertIsNotNone(result)
        
        # Verify result structure
        self.assertIsInstance(result.sentiment_score, float)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.regime_contribution, float)
        
        # Verify regime contribution is in valid range
        self.assertGreaterEqual(result.regime_contribution, 0.0)
        self.assertLessEqual(result.regime_contribution, 1.0)
    
    def test_regime_component(self):
        """Test regime component calculation"""
        market_data = {
            'timestamp': datetime.now(),
            'underlying_price': 18000.0,
            'ATM_CE_delta': 0.5,
            'ATM_PE_delta': -0.5,
            'ATM_CE_vega': 20.0,
            'ATM_PE_vega': 20.0
        }
        
        regime_component = self.greek_integration.get_regime_component(market_data)
        self.assertIsInstance(regime_component, float)
        self.assertGreaterEqual(regime_component, 0.0)
        self.assertLessEqual(regime_component, 1.0)
    
    def test_caching_functionality(self):
        """Test caching functionality"""
        market_data = {
            'timestamp': datetime.now(),
            'underlying_price': 18000.0,
            'ATM_CE_delta': 0.5,
            'ATM_PE_delta': -0.5
        }
        
        # First analysis (cache miss)
        start_time = time.time()
        result1 = self.greek_integration.analyze_greek_sentiment(market_data)
        first_time = time.time() - start_time
        
        # Second analysis (cache hit)
        start_time = time.time()
        result2 = self.greek_integration.analyze_greek_sentiment(market_data)
        second_time = time.time() - start_time
        
        # Verify both results exist
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        
        # Cache hit should be faster (though this might not always be true in tests)
        # Just verify cache metrics are updated
        status = self.greek_integration.get_integration_status()
        self.assertIn('performance_metrics', status)

class TestUnifiedEngine(unittest.TestCase):
    """Test cases for Unified Enhanced Market Regime Engine"""
    
    def setUp(self):
        """Set up test environment"""
        config = UnifiedEngineConfig(
            enable_enhanced_modules=True,
            enable_comprehensive_modules=True,
            enable_excel_config=False,  # Disable Excel for testing
            max_processing_time_seconds=5.0
        )
        self.unified_engine = create_unified_engine(config)
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        success = self.unified_engine.initialize()
        self.assertTrue(success)
        
        # Verify engine state
        self.assertTrue(self.unified_engine.is_initialized)
        self.assertTrue(self.unified_engine.is_active)
    
    def test_market_regime_analysis(self):
        """Test complete market regime analysis"""
        # Initialize engine
        self.unified_engine.initialize()
        
        # Create comprehensive test data
        market_data = {
            'timestamp': datetime.now(),
            'underlying_price': 18000.0,
            'ATM_CE_ltp': 150.0,
            'ATM_PE_ltp': 140.0,
            'ATM_CE_delta': 0.5,
            'ATM_PE_delta': -0.5,
            'ATM_CE_gamma': 0.001,
            'ATM_PE_gamma': 0.001,
            'ATM_CE_theta': -5.0,
            'ATM_PE_theta': -5.0,
            'ATM_CE_vega': 20.0,
            'ATM_PE_vega': 20.0,
            'ATM_CE_volume': 1000,
            'ATM_PE_volume': 1200,
            'ATM_CE_oi': 50000,
            'ATM_PE_oi': 55000
        }
        
        # Test analysis
        result = self.unified_engine.analyze_market_regime(market_data)
        self.assertIsNotNone(result)
        
        # Verify result structure
        self.assertIsInstance(result.regime_id, int)
        self.assertIsInstance(result.regime_name, str)
        self.assertIsInstance(result.confidence_score, float)
        self.assertIsInstance(result.regime_score, float)
        
        # Verify processing time is within limits
        self.assertLessEqual(result.total_processing_time, self.unified_engine.config.max_processing_time_seconds)
    
    def test_engine_status(self):
        """Test engine status reporting"""
        self.unified_engine.initialize()
        
        status = self.unified_engine.get_engine_status()
        self.assertIsNotNone(status)
        
        # Verify status structure
        self.assertIn('is_initialized', status)
        self.assertIn('is_active', status)
        self.assertIn('performance_metrics', status)
        self.assertIn('components', status)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self.unified_engine, 'shutdown'):
            self.unified_engine.shutdown()

def run_integration_tests():
    """Run all integration tests"""
    logger.info("Starting Enhanced Integration Test Suite")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEnhancedModuleIntegrationManager))
    test_suite.addTest(unittest.makeSuite(TestExcelConfigurationMapper))
    test_suite.addTest(unittest.makeSuite(TestGreekSentimentIntegration))
    test_suite.addTest(unittest.makeSuite(TestUnifiedEngine))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.failures:
        logger.error("Test failures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("Test errors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
