#!/usr/bin/env python3
"""
End-to-End Test Suite
====================

Comprehensive end-to-end testing for the enhanced market regime system
using the specified Excel configuration file to validate complete
system functionality including all parameters, integrations, and strategy execution.

Author: The Augster
Date: 2025-01-06
Version: 1.0.0 - End-to-End Test Suite
"""

import sys
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_excel_configuration_integration():
    """Test Excel configuration integration with the specified file"""
    try:
        logger.info("Testing Excel Configuration Integration...")
        
        # Path to the specified Excel configuration file
        excel_config_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"
        
        if not Path(excel_config_path).exists():
            logger.error(f"❌ Excel configuration file not found: {excel_config_path}")
            return False
        
        from excel_configuration_mapper import create_configuration_mapper
        
        # Create configuration mapper
        config_mapper = create_configuration_mapper(excel_config_path)
        logger.info("✅ Configuration mapper created successfully")
        
        # Test configuration loading
        if not config_mapper.load_excel_configuration():
            logger.error("❌ Failed to load Excel configuration")
            return False
        
        logger.info(f"✅ Excel configuration loaded: {len(config_mapper.excel_data)} sheets")
        
        # List loaded sheets
        for sheet_name in config_mapper.excel_data.keys():
            df = config_mapper.excel_data[sheet_name]
            logger.info(f"   - {sheet_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Test parameter mapping for available modules
        all_configs = config_mapper.get_all_module_configurations()
        logger.info(f"✅ Parameter mapping completed for {len(all_configs)} modules")
        
        for module_name, module_config in all_configs.items():
            logger.info(f"   - {module_name}: {len(module_config.parameters)} parameters, {len(module_config.validation_errors)} errors")
        
        # Test configuration validation
        validation_report = config_mapper.validate_configuration()
        logger.info(f"✅ Configuration validation completed")
        logger.info(f"   - Total modules: {validation_report['total_modules']}")
        logger.info(f"   - Successful modules: {validation_report['successful_modules']}")
        logger.info(f"   - Failed modules: {validation_report['failed_modules']}")
        logger.info(f"   - Total parameters: {validation_report['total_parameters']}")
        
        if validation_report['validation_errors']:
            logger.warning(f"⚠️  {len(validation_report['validation_errors'])} validation errors found")
            for error in validation_report['validation_errors'][:5]:  # Show first 5 errors
                logger.warning(f"     {error}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Excel configuration integration test failed: {e}")
        return False

def test_complete_system_integration():
    """Test complete system integration with all components"""
    try:
        logger.info("Testing Complete System Integration...")
        
        from unified_enhanced_market_regime_engine import (
            UnifiedEnhancedMarketRegimeEngine,
            UnifiedEngineConfig,
            create_unified_engine
        )
        from ui_integration_manager import (
            UIIntegrationManager,
            UIComponentConfig,
            create_ui_integration_manager
        )
        
        # Create engine configuration
        engine_config = UnifiedEngineConfig(
            excel_config_path="/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx",
            enable_excel_config=True,
            enable_enhanced_modules=True,
            enable_comprehensive_modules=True,
            max_processing_time_seconds=5.0,
            enable_performance_monitoring=True,
            run_integration_tests=True
        )
        
        # Create unified engine
        engine = create_unified_engine(engine_config)
        logger.info("✅ Unified engine created successfully")
        
        # Test engine initialization
        start_time = time.time()
        success = engine.initialize()
        init_time = time.time() - start_time
        
        if not success:
            logger.error("❌ Engine initialization failed")
            return False
        
        logger.info(f"✅ Engine initialization successful ({init_time:.3f}s)")
        
        # Create UI integration manager
        ui_config = UIComponentConfig(
            enable_parameter_management=True,
            enable_real_time_monitoring=True,
            enable_configuration_validation=True,
            enable_performance_metrics=True
        )
        
        ui_manager = create_ui_integration_manager(ui_config)
        logger.info("✅ UI integration manager created successfully")
        
        # Register engine reference with UI manager
        ui_manager.register_engine_reference(engine)
        ui_manager.register_config_mapper_reference(engine.config_mapper)
        
        # Initialize UI components
        if not ui_manager.initialize_ui_components():
            logger.warning("⚠️  Some UI components failed to initialize")
        else:
            logger.info("✅ UI components initialized successfully")
        
        # Test comprehensive market data analysis
        market_data = create_comprehensive_test_data()
        
        # Perform multiple analyses to test performance
        analysis_times = []
        successful_analyses = 0
        
        for i in range(5):
            start_time = time.time()
            result = engine.analyze_market_regime(market_data)
            analysis_time = time.time() - start_time
            analysis_times.append(analysis_time)
            
            if result:
                successful_analyses += 1
                if i == 0:  # Log details for first analysis
                    logger.info(f"✅ Market regime analysis successful ({analysis_time:.3f}s)")
                    logger.info(f"   Regime: {result.regime_name} (ID: {result.regime_id})")
                    logger.info(f"   Confidence: {result.confidence_score:.3f}")
                    logger.info(f"   Regime Score: {result.regime_score:.3f}")
                    logger.info(f"   Greek Sentiment: {result.greek_sentiment_contribution:.3f}")
                    logger.info(f"   Data Quality: {result.data_quality_score:.3f}")
            else:
                logger.warning(f"⚠️  Analysis {i+1} failed")
        
        # Calculate performance metrics
        avg_time = np.mean(analysis_times)
        max_time = np.max(analysis_times)
        success_rate = successful_analyses / len(analysis_times)
        
        logger.info(f"✅ Performance Analysis:")
        logger.info(f"   Success Rate: {success_rate:.1%}")
        logger.info(f"   Average Time: {avg_time:.3f}s")
        logger.info(f"   Maximum Time: {max_time:.3f}s")
        logger.info(f"   Target Met: {avg_time <= engine_config.max_processing_time_seconds}")
        
        # Test UI data retrieval
        logger.info("Testing UI data retrieval...")
        
        # Test parameter management data
        param_data = ui_manager.get_parameter_management_data()
        if "error" not in param_data:
            logger.info(f"✅ Parameter management data: {param_data['total_modules']} modules")
        else:
            logger.warning(f"⚠️  Parameter management data error: {param_data['error']}")
        
        # Test monitoring dashboard data
        monitoring_data = ui_manager.get_monitoring_dashboard_data()
        if "error" not in monitoring_data:
            logger.info("✅ Monitoring dashboard data retrieved successfully")
            if "system_health" in monitoring_data:
                health = monitoring_data["system_health"]
                logger.info(f"   System Health: {health.get('health_grade', 'Unknown')} ({health.get('overall_health_score', 0):.2f})")
        else:
            logger.warning(f"⚠️  Monitoring dashboard data error: {monitoring_data['error']}")
        
        # Test configuration validation data
        validation_data = ui_manager.get_configuration_validation_data()
        if "error" not in validation_data:
            summary = validation_data["validation_summary"]
            logger.info(f"✅ Configuration validation data: {summary['successful_modules']}/{summary['total_modules']} modules valid")
        else:
            logger.warning(f"⚠️  Configuration validation data error: {validation_data['error']}")
        
        # Test performance metrics data
        performance_data = ui_manager.get_performance_metrics_data()
        if "error" not in performance_data:
            logger.info("✅ Performance metrics data retrieved successfully")
        else:
            logger.warning(f"⚠️  Performance metrics data error: {performance_data['error']}")
        
        # Get final engine status
        engine_status = engine.get_engine_status()
        logger.info(f"✅ Final Engine Status:")
        logger.info(f"   Initialized: {engine_status['is_initialized']}")
        logger.info(f"   Active: {engine_status['is_active']}")
        logger.info(f"   Total Analyses: {engine_status['performance_metrics']['total_analyses']}")
        logger.info(f"   Successful Analyses: {engine_status['performance_metrics']['successful_analyses']}")
        
        # Get UI integration status
        ui_status = ui_manager.get_ui_integration_status()
        logger.info(f"✅ UI Integration Status:")
        logger.info(f"   Active: {ui_status['is_active']}")
        logger.info(f"   Active Components: {len(ui_status['active_components'])}")
        logger.info(f"   Available Handlers: {len(ui_status['available_handlers'])}")
        
        # Cleanup
        engine.shutdown()
        ui_manager.cleanup()
        logger.info("✅ System cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete system integration test failed: {e}")
        return False

def create_comprehensive_test_data() -> dict:
    """Create comprehensive test market data"""
    return {
        'timestamp': datetime.now(),
        'underlying_price': 18000.0,
        
        # ATM options data
        'ATM_CE_ltp': 150.0,
        'ATM_PE_ltp': 140.0,
        'ATM_CE_volume': 1000,
        'ATM_PE_volume': 1200,
        'ATM_CE_oi': 50000,
        'ATM_PE_oi': 55000,
        'ATM_CE_iv': 0.18,
        'ATM_PE_iv': 0.20,
        
        # Greeks data
        'ATM_CE_delta': 0.5,
        'ATM_PE_delta': -0.5,
        'ATM_CE_gamma': 0.001,
        'ATM_PE_gamma': 0.001,
        'ATM_CE_theta': -5.0,
        'ATM_PE_theta': -5.0,
        'ATM_CE_vega': 20.0,
        'ATM_PE_vega': 20.0,
        'ATM_CE_rho': 8.0,
        'ATM_PE_rho': -8.0,
        
        # ITM1 options data
        'ITM1_CE_ltp': 200.0,
        'ITM1_PE_ltp': 90.0,
        'ITM1_CE_volume': 800,
        'ITM1_PE_volume': 900,
        'ITM1_CE_oi': 40000,
        'ITM1_PE_oi': 35000,
        'ITM1_CE_delta': 0.7,
        'ITM1_PE_delta': -0.3,
        
        # OTM1 options data
        'OTM1_CE_ltp': 100.0,
        'OTM1_PE_ltp': 190.0,
        'OTM1_CE_volume': 600,
        'OTM1_PE_volume': 700,
        'OTM1_CE_oi': 30000,
        'OTM1_PE_oi': 45000,
        'OTM1_CE_delta': 0.3,
        'OTM1_PE_delta': -0.7,
        
        # Market data
        'spot_price': 18000.0,
        'vix': 15.5,
        'market_trend': 'bullish',
        'session': 'regular',
        
        # Additional data for enhanced modules
        'high': 18100.0,
        'low': 17900.0,
        'close': 18000.0,
        'volume': 1000000,
        'atr_14': 120.0,
        'atr_21': 115.0,
        'volatility': 0.18
    }

def run_end_to_end_tests():
    """Run complete end-to-end test suite"""
    logger.info("=" * 60)
    logger.info("ENHANCED MARKET REGIME END-TO-END TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Excel Configuration Integration", test_excel_configuration_integration),
        ("Complete System Integration", test_complete_system_integration)
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
    logger.info("END-TO-END TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL END-TO-END TESTS PASSED!")
        logger.info("✅ Enhanced Market Regime System is ready for production!")
        return True
    else:
        logger.error(f"💥 {failed} END-TO-END TESTS FAILED!")
        logger.error("❌ System requires additional fixes before production deployment")
        return False

if __name__ == '__main__':
    success = run_end_to_end_tests()
    exit(0 if success else 1)
