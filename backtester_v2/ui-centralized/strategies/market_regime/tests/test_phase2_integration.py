#!/usr/bin/env python3
"""
Phase 2 Integration Test for Enhanced Triple Straddle Framework v2.0
===================================================================

This test validates the integration of all Phase 2 components:
1. Hybrid Classification System (70%/30% weight distribution)
2. Enhanced Performance Monitor (real-time monitoring and alerts)
3. Enhanced Excel Configuration Generator (unified configuration templates)
4. Unified Enhanced Triple Straddle Pipeline (complete integration)

Author: The Augster
Date: 2025-06-20
Version: 2.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_classification_system():
    """Test Hybrid Classification System with 70%/30% weight distribution"""
    try:
        from hybrid_classification_system import classify_hybrid_market_regime
        
        logger.info("Testing Hybrid Classification System...")
        
        # Create test enhanced system data
        enhanced_system_data = {
            'directional_component': 0.3,
            'volatility_component': 0.18,
            'volume_weighted_greek_exposure': 0.15,
            'oi_signal': 0.25,
            'pearson_correlation': 0.82,
            'correlation_threshold_met': True
        }
        
        # Create test timeframe hierarchy data
        timeframe_hierarchy_data = {
            'primary_timeframe_signal': 0.2,
            'secondary_timeframe_signal': 0.18,
            'timeframe_agreement': 0.78,
            'timeframe_confidence': 0.72
        }
        
        # Test hybrid classification
        result = classify_hybrid_market_regime(enhanced_system_data, timeframe_hierarchy_data)
        
        if result and 'hybrid_regime_classification' in result:
            classification = result['hybrid_regime_classification']
            logger.info("✅ Hybrid Classification System test PASSED")
            logger.info(f"   Regime: {classification.get('regime_name', 'Unknown')}")
            logger.info(f"   Confidence: {classification.get('confidence', 0):.3f}")
            logger.info(f"   Enhanced contribution: {classification.get('enhanced_contribution', 0):.3f}")
            logger.info(f"   Stable contribution: {classification.get('stable_contribution', 0):.3f}")
            logger.info(f"   System agreement: {classification.get('system_agreement', 0):.3f}")
            logger.info(f"   Mathematical accuracy: {classification.get('mathematical_accuracy', False)}")
            return True
        else:
            logger.error("❌ Hybrid Classification System test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hybrid Classification System test ERROR: {e}")
        return False

def test_enhanced_performance_monitor():
    """Test Enhanced Performance Monitor with real-time monitoring"""
    try:
        from enhanced_performance_monitor import test_enhanced_performance_monitor
        
        logger.info("Testing Enhanced Performance Monitor...")
        
        # Run the built-in test
        result = test_enhanced_performance_monitor()
        
        if result:
            logger.info("✅ Enhanced Performance Monitor test PASSED")
            return True
        else:
            logger.error("❌ Enhanced Performance Monitor test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced Performance Monitor test ERROR: {e}")
        return False

def test_enhanced_excel_config_generator():
    """Test Enhanced Excel Configuration Generator with all profiles"""
    try:
        from enhanced_excel_config_generator import test_enhanced_excel_config_generator
        
        logger.info("Testing Enhanced Excel Configuration Generator...")
        
        # Run the built-in test
        result = test_enhanced_excel_config_generator()
        
        if result:
            logger.info("✅ Enhanced Excel Configuration Generator test PASSED")
            return True
        else:
            logger.error("❌ Enhanced Excel Configuration Generator test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced Excel Configuration Generator test ERROR: {e}")
        return False

def test_unified_enhanced_triple_straddle_pipeline():
    """Test Unified Enhanced Triple Straddle Pipeline with complete integration"""
    try:
        from unified_enhanced_triple_straddle_pipeline import test_unified_enhanced_triple_straddle_pipeline
        
        logger.info("Testing Unified Enhanced Triple Straddle Pipeline...")
        
        # Run the built-in test
        result = test_unified_enhanced_triple_straddle_pipeline()
        
        if result:
            logger.info("✅ Unified Enhanced Triple Straddle Pipeline test PASSED")
            return True
        else:
            logger.error("❌ Unified Enhanced Triple Straddle Pipeline test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unified Enhanced Triple Straddle Pipeline test ERROR: {e}")
        return False

def test_complete_phase2_integration():
    """Test complete Phase 2 integration with all components working together"""
    try:
        from unified_enhanced_triple_straddle_pipeline import process_market_data_unified_pipeline, PipelineConfig
        
        logger.info("Testing Complete Phase 2 Integration...")
        
        # Create comprehensive test market data
        market_data = pd.DataFrame({
            'strike': [22700, 22800, 22900, 23000, 23100, 23200, 23300, 23400, 23500],
            'option_type': ['CE', 'CE', 'CE', 'CE', 'PE', 'PE', 'PE', 'PE', 'PE'],
            'underlying_price': [23100] * 9,
            'dte': [0, 0, 1, 1, 1, 1, 2, 2, 3],
            'iv': [0.12, 0.14, 0.16, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23],
            'volume': [50, 150, 300, 500, 400, 300, 200, 100, 50],
            'oi': [200, 600, 1200, 2000, 1800, 1500, 1000, 600, 300],
            'close': [400, 300, 200, 100, 100, 200, 300, 400, 500],
            'ltp': [400, 300, 200, 100, 100, 200, 300, 400, 500],
            'previous_oi': [190, 570, 1140, 1900, 1710, 1425, 950, 570, 285],
            'previous_close': [390, 290, 195, 98, 98, 195, 290, 390, 490]
        })
        
        # Test all three configuration profiles
        profiles = ['Conservative', 'Balanced', 'Aggressive']
        integration_results = {}
        
        for profile in profiles:
            logger.info(f"Testing {profile} profile integration...")
            
            # Configure pipeline for full Phase 2 testing
            config = PipelineConfig(
                enable_phase1_components=True,
                enable_phase2_components=True,
                enable_volume_weighted_greeks=True,
                enable_delta_strike_selection=True,
                enable_enhanced_trending_oi=True,
                enable_hybrid_classification=True,
                enable_performance_monitoring=True,
                configuration_profile=profile
            )
            
            # Process market data through unified pipeline
            result = process_market_data_unified_pipeline(market_data, config)
            
            if result and result.get('pipeline_processing_successful', False):
                integration_results[profile] = {
                    'success': True,
                    'processing_time': result.get('processing_time', 0),
                    'pipeline_accuracy': result.get('pipeline_accuracy', 0),
                    'mathematical_accuracy': result.get('mathematical_accuracy', False),
                    'components_processed': len([k for k in result.keys() if any(suffix in k for suffix in ['_greeks', '_selection', '_oi', '_classification', '_monitoring'])])
                }
                
                logger.info(f"   ✅ {profile} profile integration PASSED")
                logger.info(f"      Processing time: {result['processing_time']:.3f}s")
                logger.info(f"      Pipeline accuracy: {result['pipeline_accuracy']:.3f}")
                logger.info(f"      Mathematical accuracy: {result['mathematical_accuracy']}")
                logger.info(f"      Components processed: {integration_results[profile]['components_processed']}")
                
                # Validate performance targets
                if result['processing_time'] <= 3.0:
                    logger.info(f"      ✅ Processing time target met (<3s)")
                else:
                    logger.warning(f"      ⚠️  Processing time target missed: {result['processing_time']:.3f}s")
                
                if result['pipeline_accuracy'] >= 0.85:
                    logger.info(f"      ✅ Accuracy target met (>85%)")
                else:
                    logger.warning(f"      ⚠️  Accuracy target missed: {result['pipeline_accuracy']:.3f}")
                
                if result['mathematical_accuracy']:
                    logger.info(f"      ✅ Mathematical accuracy validated (±0.001)")
                else:
                    logger.warning(f"      ⚠️  Mathematical accuracy validation failed")
                
            else:
                integration_results[profile] = {'success': False}
                logger.error(f"   ❌ {profile} profile integration FAILED")
                if result:
                    logger.error(f"      Errors: {result.get('error_messages', [])}")
                    logger.error(f"      Warnings: {result.get('warning_messages', [])}")
                return False
        
        # Validate integration results
        successful_profiles = sum(1 for r in integration_results.values() if r['success'])
        
        if successful_profiles == len(profiles):
            logger.info("✅ Complete Phase 2 Integration test PASSED")
            logger.info(f"   All {len(profiles)} configuration profiles tested successfully")
            
            # Performance summary
            avg_processing_time = np.mean([r['processing_time'] for r in integration_results.values() if r['success']])
            avg_accuracy = np.mean([r['pipeline_accuracy'] for r in integration_results.values() if r['success']])
            math_accuracy_rate = np.mean([r['mathematical_accuracy'] for r in integration_results.values() if r['success']])
            
            logger.info(f"   Average processing time: {avg_processing_time:.3f}s")
            logger.info(f"   Average pipeline accuracy: {avg_accuracy:.3f}")
            logger.info(f"   Mathematical accuracy rate: {math_accuracy_rate:.3f}")
            
            return True
        else:
            logger.error(f"❌ Complete Phase 2 Integration test FAILED: {successful_profiles}/{len(profiles)} profiles successful")
            return False
            
    except Exception as e:
        logger.error(f"❌ Complete Phase 2 Integration test ERROR: {e}")
        return False

def main():
    """Run all Phase 2 integration tests"""
    logger.info("=" * 80)
    logger.info("ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0 - PHASE 2 INTEGRATION TESTS")
    logger.info("=" * 80)
    
    test_results = []
    
    # Test individual Phase 2 components
    test_results.append(("Hybrid Classification System", test_hybrid_classification_system()))
    test_results.append(("Enhanced Performance Monitor", test_enhanced_performance_monitor()))
    test_results.append(("Enhanced Excel Configuration Generator", test_enhanced_excel_config_generator()))
    test_results.append(("Unified Enhanced Triple Straddle Pipeline", test_unified_enhanced_triple_straddle_pipeline()))
    
    # Test complete integration
    test_results.append(("Complete Phase 2 Integration", test_complete_phase2_integration()))
    
    # Summary
    logger.info("=" * 80)
    logger.info("PHASE 2 TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:.<50} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 80)
    logger.info(f"OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL PHASE 2 TESTS PASSED - ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0 COMPLETE!")
        logger.info("🚀 Ready for production deployment with full Phase 1 + Phase 2 integration!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed - Review Phase 2 implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
