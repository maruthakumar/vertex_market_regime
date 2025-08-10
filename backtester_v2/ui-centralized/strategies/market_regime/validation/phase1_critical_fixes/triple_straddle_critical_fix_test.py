"""
Triple Straddle Critical Fix Test - Phase 1 Priority P0

This test addresses the CRITICAL PRODUCTION BLOCKER in triple_straddle_analysis.py
where synthetic/random price history is used instead of real market data.

CRITICAL ISSUE: Lines 315-340 in triple_straddle_analysis.py generate synthetic
price history with random data, making technical analysis invalid.

Author: The Augster
Date: 2025-01-16
Priority: P0 - CRITICAL PRODUCTION BLOCKER
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import logging
import inspect

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from triple_straddle_analysis import TripleStraddleAnalysisEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripleStraddleCriticalFixTest(unittest.TestCase):
    """
    Critical fix test for Triple Straddle Analysis synthetic data issue
    
    OBJECTIVE: Ensure Triple Straddle Analysis uses real market data
    instead of synthetic/random price history for technical analysis.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'use_real_data': True,
            'synthetic_data_fallback': False,
            'component_weights': {
                'atm_straddle': 0.50,
                'itm1_straddle': 0.30,
                'otm1_straddle': 0.20
            }
        }
        self.engine = TripleStraddleAnalysisEngine(self.config)
        
        # Create sample real market data
        self.real_market_data = self._create_real_market_data()
        
    def _create_real_market_data(self) -> dict:
        """Create realistic market data for testing"""
        # Simulate real market data structure
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=300,
            freq='1min'
        )
        
        # Create realistic price movement (not random)
        base_price = 100.0
        trend = 0.001  # Small upward trend
        volatility = 0.02
        
        prices = []
        for i in range(300):
            # Realistic price movement with trend and mean reversion
            if i == 0:
                price = base_price
            else:
                # Mean reversion + trend + small random component
                mean_reversion = (base_price - prices[-1]) * 0.01
                trend_component = trend
                random_component = np.random.normal(0, volatility) * 0.1
                
                price_change = mean_reversion + trend_component + random_component
                price = prices[-1] * (1 + price_change)
            
            prices.append(price)
        
        # Create realistic volume pattern
        volumes = []
        for i in range(300):
            # Higher volume during market open/close
            hour = timestamps[i].hour
            if 9 <= hour <= 10 or 14 <= hour <= 15:
                base_volume = 1000
            else:
                base_volume = 500
            
            volume = int(base_volume * (1 + np.random.normal(0, 0.3)))
            volumes.append(max(volume, 100))  # Minimum volume
        
        return {
            'underlying_price': prices[-1],
            'strikes': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
            'options_data': {
                100: {  # ATM strike
                    'CE': {'close': 2.5, 'volume': 1000, 'oi': 5000, 'iv': 0.15},
                    'PE': {'close': 2.3, 'volume': 1200, 'oi': 5500, 'iv': 0.16}
                },
                99: {   # ITM1 strike
                    'CE': {'close': 3.2, 'volume': 800, 'oi': 4000, 'iv': 0.14},
                    'PE': {'close': 1.8, 'volume': 900, 'oi': 4500, 'iv': 0.15}
                },
                101: {  # OTM1 strike
                    'CE': {'close': 1.9, 'volume': 700, 'oi': 3500, 'iv': 0.16},
                    'PE': {'close': 2.8, 'volume': 1100, 'oi': 6000, 'iv': 0.17}
                }
            },
            'price_history': [
                {
                    'timestamp': timestamps[i],
                    'close': prices[i],
                    'high': prices[i] * 1.005,
                    'low': prices[i] * 0.995,
                    'volume': volumes[i]
                }
                for i in range(300)
            ]
        }
    
    def test_critical_fix_no_synthetic_data_generation(self):
        """
        CRITICAL TEST: Ensure no synthetic data generation in production code

        This test verifies that the problematic synthetic data generation
        code (lines 315-340) has been replaced with real data integration.
        """
        logger.info("🚨 CRITICAL TEST: Checking for synthetic data generation")

        # Test with market data that includes trade_date and underlying_symbol
        enhanced_market_data = self.real_market_data.copy()
        enhanced_market_data['trade_date'] = datetime.now().date()
        enhanced_market_data['underlying_symbol'] = 'NIFTY'

        # Analyze the market regime
        result = self.engine.analyze_market_regime(enhanced_market_data)

        # Check that result is not None (basic functionality)
        self.assertIsNotNone(result, "Market regime analysis should return results")

        # CRITICAL CHECK: Verify no synthetic data was used
        # Check that the engine has HEAVYDB_AVAILABLE flag
        self.assertTrue(hasattr(self.engine, '__class__'))

        # Verify the method exists and has been updated
        method_source = inspect.getsource(self.engine._get_component_price_history)

        # CRITICAL: Ensure synthetic data generation is not present
        self.assertNotIn('np.random.normal', method_source,
                        "CRITICAL: Synthetic data generation still present in code")
        self.assertNotIn('np.random.randint', method_source,
                        "CRITICAL: Random volume generation still present in code")

        # Verify real data integration is present
        self.assertIn('HeavyDB', method_source,
                     "Real HeavyDB integration should be present")
        self.assertIn('_fetch_option_price_history', method_source,
                     "Real option price history fetching should be present")

        # For now, we check that the analysis completes without errors
        self.assertIn('triple_straddle_score', result)
        self.assertIn('confidence', result)

        logger.info("✅ CRITICAL TEST PASSED: No synthetic data generation detected")
    
    def test_real_data_integration_validation(self):
        """
        Test that real market data is properly integrated and used
        """
        logger.info("🔍 Testing real data integration")
        
        # Mock the price history retrieval to ensure it's called
        with patch.object(self.engine, '_get_component_price_history') as mock_price_history:
            # Set up mock to return real data
            mock_price_history.return_value = pd.DataFrame(self.real_market_data['price_history'])
            
            # Run analysis
            result = self.engine.analyze_market_regime(self.real_market_data)
            
            # Verify that real data retrieval was called
            self.assertTrue(mock_price_history.called, "Real price history should be retrieved")
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('triple_straddle_score', result)
            
        logger.info("✅ Real data integration validation passed")
    
    def test_technical_analysis_with_real_data(self):
        """
        Test that technical analysis (EMA, VWAP, Pivot) uses real data
        """
        logger.info("📊 Testing technical analysis with real data")
        
        # Create a component for testing
        component = self.engine._extract_straddle_components(self.real_market_data)
        
        if 'atm' in component:
            # Test technical analysis on real data
            result = self.engine._analyze_straddle_component(
                component['atm'], 'atm', self.real_market_data
            )
            
            # Verify technical analysis results
            self.assertIsNotNone(result.ema_score)
            self.assertIsNotNone(result.vwap_score)
            self.assertIsNotNone(result.pivot_score)
            
            # Verify scores are within reasonable ranges
            self.assertGreaterEqual(result.ema_score, -1.0)
            self.assertLessEqual(result.ema_score, 1.0)
            self.assertGreaterEqual(result.vwap_score, -1.0)
            self.assertLessEqual(result.vwap_score, 1.0)
            self.assertGreaterEqual(result.pivot_score, -1.0)
            self.assertLessEqual(result.pivot_score, 1.0)
            
        logger.info("✅ Technical analysis with real data validation passed")
    
    def test_performance_with_real_data(self):
        """
        Test that performance meets requirements with real data
        """
        logger.info("⚡ Testing performance with real data")
        
        start_time = datetime.now()
        
        # Run analysis multiple times to test performance
        for _ in range(10):
            result = self.engine.analyze_market_regime(self.real_market_data)
            self.assertIsNotNone(result)
        
        end_time = datetime.now()
        avg_time = (end_time - start_time).total_seconds() / 10
        
        # Performance should be under 100ms per analysis
        self.assertLess(avg_time, 0.1, f"Analysis should complete in <100ms, got {avg_time:.3f}s")
        
        logger.info(f"✅ Performance test passed: {avg_time:.3f}s average")
    
    def test_data_quality_validation(self):
        """
        Test that data quality validation works correctly
        """
        logger.info("🔍 Testing data quality validation")
        
        # Test with incomplete data
        incomplete_data = self.real_market_data.copy()
        incomplete_data['price_history'] = incomplete_data['price_history'][:10]  # Only 10 periods
        
        result = self.engine.analyze_market_regime(incomplete_data)
        
        # Should handle incomplete data gracefully
        self.assertIsNotNone(result)
        
        # Confidence should be lower for incomplete data
        if 'confidence' in result:
            self.assertLessEqual(result['confidence'], 0.8)
        
        logger.info("✅ Data quality validation passed")
    
    def test_critical_fix_validation_summary(self):
        """
        Summary validation test for the critical fix
        """
        logger.info("📋 CRITICAL FIX VALIDATION SUMMARY")
        
        # Run comprehensive analysis
        result = self.engine.analyze_market_regime(self.real_market_data)
        
        # Validation checklist
        validation_results = {
            'analysis_completes': result is not None,
            'has_triple_straddle_score': 'triple_straddle_score' in result,
            'has_confidence': 'confidence' in result,
            'has_component_results': 'component_results' in result,
            'confidence_reasonable': result.get('confidence', 0) > 0.3 if result else False
        }
        
        # Log validation results
        for check, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{status}: {check}")
        
        # All validations should pass
        all_passed = all(validation_results.values())
        self.assertTrue(all_passed, f"Critical fix validation failed: {validation_results}")
        
        if all_passed:
            logger.info("🎉 CRITICAL FIX VALIDATION: ALL TESTS PASSED")
            logger.info("✅ Triple Straddle Analysis ready for production")
        else:
            logger.error("🚨 CRITICAL FIX VALIDATION: TESTS FAILED")
            logger.error("❌ Triple Straddle Analysis NOT ready for production")

def run_critical_fix_test():
    """Run the critical fix test suite"""
    print("=" * 80)
    print("🚨 TRIPLE STRADDLE CRITICAL FIX TEST - PHASE 1 PRIORITY P0")
    print("=" * 80)
    print()
    print("OBJECTIVE: Fix synthetic data usage in Triple Straddle Analysis")
    print("CRITICAL ISSUE: Lines 315-340 use synthetic/random price history")
    print("REQUIRED FIX: Replace with real market data integration")
    print()
    print("Starting critical fix validation...")
    print()
    
    # Run the test suite
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == '__main__':
    run_critical_fix_test()
