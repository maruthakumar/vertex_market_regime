#!/usr/bin/env python3
"""
Simple Integration Test
=======================

Simple test to validate the enhanced market regime integration components
without complex import dependencies.

Author: The Augster
Date: 2025-01-06
Version: 1.0.0 - Simple Integration Test
"""

import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_module_integration_manager():
    """Test Enhanced Module Integration Manager"""
    try:
        logger.info("Testing Enhanced Module Integration Manager...")
        
        from enhanced_module_integration_manager import (
            EnhancedModuleIntegrationManager,
            ModuleStatus,
            IntegrationPriority,
            create_integration_manager
        )
        
        # Create integration manager
        manager = create_integration_manager()
        logger.info("✅ Integration manager created successfully")
        
        # Create a simple test module
        class TestModule:
            def __init__(self, config):
                self.config = config
            
            def analyze(self, market_data):
                return {'test_result': 42, 'confidence': 0.8}
        
        # Register test module
        success = manager.register_enhanced_module(
            name='test_module',
            module_class=TestModule,
            config_requirements=[],
            dependencies=[],
            priority=IntegrationPriority.MEDIUM
        )
        
        if success:
            logger.info("✅ Module registration successful")
        else:
            logger.error("❌ Module registration failed")
            return False
        
        # Configure modules
        global_config = {'test_module': {}, 'symbol': 'NIFTY'}
        success = manager.configure_modules(global_config)
        
        if success:
            logger.info("✅ Module configuration successful")
        else:
            logger.error("❌ Module configuration failed")
            return False
        
        # Initialize modules
        success = manager.initialize_modules()
        
        if success:
            logger.info("✅ Module initialization successful")
        else:
            logger.error("❌ Module initialization failed")
            return False
        
        # Activate integration
        success = manager.activate_integration()
        
        if success:
            logger.info("✅ Integration activation successful")
        else:
            logger.error("❌ Integration activation failed")
            return False
        
        # Test module execution
        market_data = {
            'timestamp': datetime.now(),
            'underlying_price': 18000.0
        }
        
        result = manager.execute_module_analysis('test_module', market_data)
        
        if result and 'test_result' in result:
            logger.info("✅ Module execution successful")
            logger.info(f"   Result: {result}")
        else:
            logger.error("❌ Module execution failed")
            return False
        
        # Get status
        status = manager.get_integration_status()
        logger.info(f"✅ Integration status: {status['integration_active']}")
        logger.info(f"   Active modules: {status['metrics']['active_modules']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Module Integration Manager test failed: {e}")
        return False

def test_excel_configuration_mapper():
    """Test Excel Configuration Mapper"""
    try:
        logger.info("Testing Excel Configuration Mapper...")
        
        from excel_configuration_mapper import (
            ExcelConfigurationMapper,
            ParameterMapping,
            ParameterType
        )
        
        # Test without actual Excel file (just class creation)
        mapper = ExcelConfigurationMapper("dummy_path.xlsx")
        logger.info("✅ Configuration mapper created successfully")
        
        # Test parameter mappings initialization
        if hasattr(mapper, 'parameter_mappings') and mapper.parameter_mappings:
            logger.info(f"✅ Parameter mappings initialized for {len(mapper.parameter_mappings)} modules")
            
            # List available modules
            for module_name in mapper.parameter_mappings.keys():
                logger.info(f"   - {module_name}")
        else:
            logger.warning("⚠️  No parameter mappings found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Excel Configuration Mapper test failed: {e}")
        return False

def test_greek_sentiment_integration():
    """Test Greek Sentiment Integration"""
    try:
        logger.info("Testing Greek Sentiment Integration...")
        
        from enhanced_greek_sentiment_integration import (
            GreekSentimentAnalyzerIntegration,
            GreekSentimentIntegrationConfig,
            create_greek_sentiment_integration
        )
        
        # Create configuration
        config = GreekSentimentIntegrationConfig(
            delta_weight=1.2,
            vega_weight=1.5,
            theta_weight=0.3,
            gamma_weight=0.0,
            enable_caching=True
        )
        
        # Create integration
        integration = create_greek_sentiment_integration(config)
        logger.info("✅ Greek sentiment integration created successfully")
        
        # Test with sample market data
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
        
        # Test regime component calculation
        regime_component = integration.get_regime_component(market_data)
        
        if isinstance(regime_component, float) and 0.0 <= regime_component <= 1.0:
            logger.info(f"✅ Greek sentiment regime component: {regime_component:.3f}")
        else:
            logger.error(f"❌ Invalid regime component: {regime_component}")
            return False
        
        # Test integration status
        status = integration.get_integration_status()
        if 'total_analyses' in status:
            logger.info(f"✅ Integration status retrieved: {status['total_analyses']} analyses")
        else:
            logger.warning("⚠️  Integration status incomplete")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Greek Sentiment Integration test failed: {e}")
        return False

def test_unified_engine():
    """Test Unified Enhanced Market Regime Engine"""
    try:
        logger.info("Testing Unified Enhanced Market Regime Engine...")
        
        from unified_enhanced_market_regime_engine import (
            UnifiedEnhancedMarketRegimeEngine,
            UnifiedEngineConfig,
            create_unified_engine
        )
        
        # Create configuration
        config = UnifiedEngineConfig(
            enable_enhanced_modules=True,
            enable_comprehensive_modules=True,
            enable_excel_config=False,  # Disable Excel for testing
            max_processing_time_seconds=5.0
        )
        
        # Create unified engine
        engine = create_unified_engine(config)
        logger.info("✅ Unified engine created successfully")
        
        # Test initialization
        start_time = time.time()
        success = engine.initialize()
        init_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ Engine initialization successful ({init_time:.3f}s)")
        else:
            logger.error("❌ Engine initialization failed")
            return False
        
        # Test market regime analysis
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
            'ATM_PE_vega': 20.0
        }
        
        start_time = time.time()
        result = engine.analyze_market_regime(market_data)
        analysis_time = time.time() - start_time
        
        if result:
            logger.info(f"✅ Market regime analysis successful ({analysis_time:.3f}s)")
            logger.info(f"   Regime: {result.regime_name} (ID: {result.regime_id})")
            logger.info(f"   Confidence: {result.confidence_score:.3f}")
            logger.info(f"   Regime Score: {result.regime_score:.3f}")
            logger.info(f"   Greek Sentiment: {result.greek_sentiment_contribution:.3f}")
            
            # Verify processing time
            if analysis_time <= config.max_processing_time_seconds:
                logger.info(f"✅ Processing time within target ({analysis_time:.3f}s <= {config.max_processing_time_seconds}s)")
            else:
                logger.warning(f"⚠️  Processing time exceeded target ({analysis_time:.3f}s > {config.max_processing_time_seconds}s)")
        else:
            logger.error("❌ Market regime analysis failed")
            return False
        
        # Test engine status
        status = engine.get_engine_status()
        if status and 'is_initialized' in status:
            logger.info(f"✅ Engine status: Initialized={status['is_initialized']}, Active={status['is_active']}")
            logger.info(f"   Performance: {status['performance_metrics']['total_analyses']} analyses")
        else:
            logger.warning("⚠️  Engine status incomplete")
        
        # Cleanup
        engine.shutdown()
        logger.info("✅ Engine shutdown successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unified Engine test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    logger.info("=" * 60)
    logger.info("ENHANCED MARKET REGIME INTEGRATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Enhanced Module Integration Manager", test_enhanced_module_integration_manager),
        ("Excel Configuration Mapper", test_excel_configuration_mapper),
        ("Greek Sentiment Integration", test_greek_sentiment_integration),
        ("Unified Enhanced Engine", test_unified_engine)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.error(f"💥 {failed} TESTS FAILED!")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
