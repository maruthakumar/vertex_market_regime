#!/usr/bin/env python3
"""
Comprehensive HeavyDB Integration Fix Validation

This module validates that all HeavyDB connection issues have been resolved
and all market regime components work correctly with both real HeavyDB
connections and appropriate fallback mechanisms.

Author: The Augster
Date: 2025-06-18
Version: 1.0.0
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import sys
import os
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class TestHeavyDBIntegrationFix(unittest.TestCase):
    """Comprehensive test suite for HeavyDB integration fix validation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_market_data = {
            'underlying_price': 19500,
            'timestamp': datetime.now(),
            'symbol': 'NIFTY',
            'dte': 30,
            'iv_percentile': 0.5,
            'atr_normalized': 0.4,
            'gamma_exposure': 0.3,
            'ema_alignment': 0.6,
            'price_momentum': 0.5,
            'volume_confirmation': 0.4,
            'strike_correlation': 0.7,
            'vwap_deviation': 0.6,
            'pivot_analysis': 0.5
        }
    
    def test_heavydb_connection_module(self):
        """Test the new HeavyDB connection module"""
        try:
            from dal.heavydb_connection import (
                get_connection, execute_query, test_connection, 
                get_connection_status, validate_table_exists,
                generate_mock_option_chain_data
            )
            
            # Test connection
            conn = get_connection()
            self.assertIsNotNone(conn, "HeavyDB connection should be available")
            
            # Test simple query
            result = execute_query(conn, "SELECT 1 as test")
            self.assertFalse(result.empty, "Simple query should return results")
            self.assertEqual(result.iloc[0, 0], 1, "Query result should be 1")
            
            # Test connection status
            status = get_connection_status()
            self.assertIsInstance(status, dict, "Status should be a dictionary")
            self.assertIn('connection_available', status)
            self.assertIn('table_exists', status)
            
            # Test table validation
            table_exists = validate_table_exists('nifty_option_chain')
            self.assertTrue(table_exists, "nifty_option_chain table should exist")
            
            # Test mock data generation (for fallback scenarios)
            mock_data = generate_mock_option_chain_data('NIFTY', 19500, 50)
            self.assertFalse(mock_data.empty, "Mock data should be generated")
            self.assertGreater(len(mock_data), 0, "Mock data should have records")
            
            # Validate mock data structure
            required_columns = [
                'trade_time', 'symbol', 'strike_price', 'option_type', 
                'last_price', 'volume', 'open_interest', 'implied_volatility'
            ]
            for col in required_columns:
                self.assertIn(col, mock_data.columns, f"Mock data should have {col} column")
            
            logger.info("✅ HeavyDB connection module test passed")
            
        except Exception as e:
            self.fail(f"HeavyDB connection module test failed: {e}")
    
    def test_enhanced_12_regime_detector_with_heavydb(self):
        """Test Enhanced 12-Regime Detector with HeavyDB integration"""
        try:
            from enhanced_12_regime_detector import Enhanced12RegimeDetector
            
            detector = Enhanced12RegimeDetector()
            self.assertIsNotNone(detector, "Detector should be initialized")
            
            # Test regime classification
            result = detector.classify_12_regime(self.test_market_data)
            
            # Validate result structure
            self.assertIsNotNone(result.regime_id, "Regime ID should be set")
            self.assertGreaterEqual(result.confidence, 0.0, "Confidence should be >= 0")
            self.assertLessEqual(result.confidence, 1.0, "Confidence should be <= 1")
            self.assertGreater(result.processing_time, 0.0, "Processing time should be > 0")
            
            logger.info(f"✅ 12-Regime Detector test passed: {result.regime_id}")
            
        except Exception as e:
            self.fail(f"Enhanced 12-Regime Detector test failed: {e}")
    
    def test_atm_cepe_rolling_analyzer_with_heavydb(self):
        """Test ATM CE/PE Rolling Analyzer with HeavyDB integration"""
        try:
            from atm_cepe_rolling_analyzer import ATMCEPERollingAnalyzer
            
            analyzer = ATMCEPERollingAnalyzer()
            self.assertIsNotNone(analyzer, "Analyzer should be initialized")
            
            # Test rolling analysis
            result = analyzer.analyze_atm_cepe_rolling(self.test_market_data, 'NIFTY')
            
            # Validate result structure
            self.assertIsNotNone(result.ce_pe_correlation, "CE/PE correlation should be set")
            self.assertGreaterEqual(result.ce_pe_correlation, 0.0, "Correlation should be >= 0")
            self.assertLessEqual(result.ce_pe_correlation, 1.0, "Correlation should be <= 1")
            self.assertGreater(result.processing_time, 0.0, "Processing time should be > 0")
            self.assertIsInstance(result.rolling_correlations, dict, "Rolling correlations should be dict")
            self.assertIsInstance(result.technical_indicators, dict, "Technical indicators should be dict")
            
            logger.info(f"✅ ATM CE/PE Rolling Analyzer test passed: correlation={result.ce_pe_correlation:.3f}")
            
        except Exception as e:
            self.fail(f"ATM CE/PE Rolling Analyzer test failed: {e}")
    
    def test_optimized_heavydb_engine(self):
        """Test Optimized HeavyDB Engine"""
        try:
            from optimized_heavydb_engine import OptimizedHeavyDBEngine
            
            engine = OptimizedHeavyDBEngine()
            self.assertIsNotNone(engine, "Engine should be initialized")
            
            # Test optimization
            test_data = [self.test_market_data]
            result = engine.optimize_correlation_matrix_processing(test_data)
            
            # Validate result structure
            self.assertIn('correlation_matrices', result)
            self.assertIn('performance_metrics', result)
            self.assertIsInstance(result['correlation_matrices'], list)
            
            # Validate performance
            performance = result['performance_metrics']
            self.assertIn('total_processing_time', performance)
            self.assertLess(performance['total_processing_time'], 5.0, "Processing should be fast")
            
            logger.info(f"✅ Optimized HeavyDB Engine test passed: {performance['total_processing_time']:.3f}s")
            
        except Exception as e:
            self.fail(f"Optimized HeavyDB Engine test failed: {e}")
    
    def test_real_data_integration_engine(self):
        """Test Real Data Integration Engine"""
        try:
            from real_data_integration_engine import RealDataIntegrationEngine
            
            engine = RealDataIntegrationEngine()
            self.assertIsNotNone(engine, "Engine should be initialized")
            
            # Test real data integration
            result = engine.integrate_real_production_data(
                'NIFTY', datetime.now(), 19500, 60
            )
            
            # Validate result structure
            self.assertIsNotNone(result.data_source, "Data source should be set")
            self.assertIsNotNone(result.validation_result, "Validation result should be set")
            self.assertGreater(result.processing_time, 0.0, "Processing time should be > 0")
            
            # Validate data quality
            validation = result.validation_result
            self.assertGreaterEqual(validation.data_quality_score, 0.0, "Quality score should be >= 0")
            self.assertLessEqual(validation.data_quality_score, 1.0, "Quality score should be <= 1")
            
            logger.info(f"✅ Real Data Integration Engine test passed: quality={validation.data_quality_score:.3f}")
            
        except Exception as e:
            self.fail(f"Real Data Integration Engine test failed: {e}")
    
    def test_advanced_dynamic_weighting_engine(self):
        """Test Advanced Dynamic Weighting Engine"""
        try:
            from advanced_dynamic_weighting_engine import AdvancedDynamicWeightingEngine
            
            engine = AdvancedDynamicWeightingEngine()
            self.assertIsNotNone(engine, "Engine should be initialized")
            
            # Generate test historical data
            historical_data = []
            for i in range(50):
                data_point = {
                    'regime_confidence': 0.7 + np.random.normal(0, 0.1),
                    'triple_straddle_score': 0.6 + np.random.normal(0, 0.1),
                    'accuracy': 0.75 + np.random.normal(0, 0.1),
                    'dte': 30,
                    'timestamp': datetime.now() - timedelta(hours=i)
                }
                historical_data.append(data_point)
            
            # Test weight optimization
            result = engine.optimize_weights_ml_based(historical_data, 30)
            
            # Validate result structure
            self.assertIsInstance(result.optimized_weights, dict)
            self.assertIn('triple_straddle', result.optimized_weights)
            self.assertIn('regime_components', result.optimized_weights)
            
            # Validate weight sum
            weight_sum = sum(result.optimized_weights.values())
            self.assertAlmostEqual(weight_sum, 1.0, places=2, msg="Weights should sum to 1.0")
            
            logger.info(f"✅ Advanced Dynamic Weighting Engine test passed: weights={result.optimized_weights}")
            
        except Exception as e:
            self.fail(f"Advanced Dynamic Weighting Engine test failed: {e}")
    
    def test_triple_straddle_12regime_integrator(self):
        """Test Triple Straddle 12-Regime Integrator with all components"""
        try:
            from triple_straddle_12regime_integrator import TripleStraddle12RegimeIntegrator
            
            integrator = TripleStraddle12RegimeIntegrator()
            self.assertIsNotNone(integrator, "Integrator should be initialized")
            
            # Test integrated analysis
            result = integrator.analyze_integrated_regime(self.test_market_data)
            
            # Validate result structure
            self.assertIsNotNone(result.regime_id, "Regime ID should be set")
            self.assertIsNotNone(result.final_score, "Final score should be set")
            self.assertGreaterEqual(result.confidence, 0.0, "Confidence should be >= 0")
            self.assertLessEqual(result.confidence, 1.0, "Confidence should be <= 1")
            self.assertGreater(result.processing_time, 0.0, "Processing time should be > 0")
            
            # Validate performance target
            self.assertLess(result.processing_time, 5.0, "Processing should be under 5 seconds")
            
            logger.info(f"✅ Triple Straddle Integrator test passed: {result.regime_id}, score={result.final_score:.3f}")
            
        except Exception as e:
            self.fail(f"Triple Straddle 12-Regime Integrator test failed: {e}")
    
    def test_correlation_matrix_engine_with_optimization(self):
        """Test Correlation Matrix Engine with optimization"""
        try:
            from correlation_matrix_engine import CorrelationMatrixEngine
            
            engine = CorrelationMatrixEngine()
            self.assertIsNotNone(engine, "Engine should be initialized")
            
            # Test correlation analysis
            result = engine.analyze_multi_strike_correlation(self.test_market_data, 'NIFTY')
            
            # Validate result structure
            self.assertIsNotNone(result.overall_correlation, "Overall correlation should be set")
            self.assertGreaterEqual(result.overall_correlation, 0.0, "Correlation should be >= 0")
            self.assertLessEqual(result.overall_correlation, 1.0, "Correlation should be <= 1")
            self.assertGreater(result.processing_time, 0.0, "Processing time should be > 0")
            
            # Validate performance target (should be optimized)
            self.assertLess(result.processing_time, 2.0, "Processing should be optimized")
            
            logger.info(f"✅ Correlation Matrix Engine test passed: correlation={result.overall_correlation:.3f}")
            
        except Exception as e:
            self.fail(f"Correlation Matrix Engine test failed: {e}")
    
    def test_end_to_end_system_performance(self):
        """Test end-to-end system performance with all components"""
        try:
            from triple_straddle_12regime_integrator import TripleStraddle12RegimeIntegrator
            
            integrator = TripleStraddle12RegimeIntegrator()
            
            # Test multiple scenarios for performance validation
            test_scenarios = [
                {'underlying_price': 19500, 'dte': 7},
                {'underlying_price': 19450, 'dte': 21},
                {'underlying_price': 19550, 'dte': 45}
            ]
            
            total_start_time = time.time()
            results = []
            
            for scenario in test_scenarios:
                test_data = self.test_market_data.copy()
                test_data.update(scenario)
                
                scenario_start = time.time()
                result = integrator.analyze_integrated_regime(test_data)
                scenario_time = time.time() - scenario_start
                
                results.append({
                    'scenario': scenario,
                    'result': result,
                    'processing_time': scenario_time
                })
                
                # Validate individual scenario performance
                self.assertLess(scenario_time, 5.0, f"Scenario processing should be under 5s: {scenario_time:.3f}s")
            
            total_time = time.time() - total_start_time
            avg_time = total_time / len(test_scenarios)
            
            # Validate overall performance
            self.assertLess(avg_time, 3.0, f"Average processing time should be under 3s: {avg_time:.3f}s")
            
            # Validate all results are valid
            for result_data in results:
                result = result_data['result']
                self.assertIsNotNone(result.regime_id, "All results should have regime ID")
                self.assertGreater(result.confidence, 0.0, "All results should have confidence > 0")
            
            logger.info(f"✅ End-to-end performance test passed: avg_time={avg_time:.3f}s")
            
        except Exception as e:
            self.fail(f"End-to-end system performance test failed: {e}")

def run_heavydb_integration_fix_tests():
    """Run comprehensive HeavyDB integration fix validation"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔍 HEAVYDB INTEGRATION FIX VALIDATION")
    print("="*70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestHeavyDBIntegrationFix)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"HEAVYDB INTEGRATION FIX VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'='*70}")
    
    if success_rate == 1.0:  # 100% success rate required
        print("✅ ALL HEAVYDB INTEGRATION TESTS PASSED")
        print("🎉 SYSTEM IS READY FOR PRODUCTION")
        return True
    else:
        print("❌ SOME HEAVYDB INTEGRATION TESTS FAILED")
        if failures > 0:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if errors > 0:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        return False

if __name__ == "__main__":
    success = run_heavydb_integration_fix_tests()
    sys.exit(0 if success else 1)
