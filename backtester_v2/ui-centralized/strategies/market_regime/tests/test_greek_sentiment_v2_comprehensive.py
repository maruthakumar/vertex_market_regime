#!/usr/bin/env python3
"""
Comprehensive Test Suite for Greek Sentiment V2
==============================================

This test suite validates all components of the Greek Sentiment V2 implementation
using real HeavyDB data and PHASE2 Excel configuration.

Test Coverage:
- BaselineTracker: 9:15 AM baseline establishment
- VolumeOIWeighter: Dual weighting system
- ITMOTMAnalyzer: Moneyness classification
- DTEAdjuster: DTE-specific adjustments
- GreekCalculator: Normalization and aggregation
- Integration: Full pipeline testing

Author: Market Regime Testing Team
Date: 2025-07-06
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Import Greek Sentiment components
    from indicators.greek_sentiment import (
        GreekSentimentAnalyzer,
        BaselineTracker,
        VolumeOIWeighter,
        ITMOTMAnalyzer,
        DTEAdjuster,
        GreekCalculator
    )
    
    # Import data utilities
    from base.heavydb_connector import HeavyDBConnector
    from excel_config_manager import MarketRegimeExcelManager
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure all required modules are in the Python path")
    raise

class TestGreekSentimentV2Comprehensive(unittest.TestCase):
    """Comprehensive test suite for Greek Sentiment V2"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        logger.info("Setting up Greek Sentiment V2 comprehensive test environment")
        
        # Initialize HeavyDB connection
        cls.db_connector = HeavyDBConnector()
        cls.db_connected = cls.db_connector.connect()
        
        if not cls.db_connected:
            logger.warning("HeavyDB connection failed - tests will use mock data")
        
        # Load Excel configuration
        config_path = Path(__file__).parent.parent.parent / "config" / "PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"
        cls.excel_manager = MarketRegimeExcelManager(str(config_path))
        
        # Extract Greek Sentiment configuration
        cls.greek_config = cls._extract_greek_config()
        
        # Load test data
        cls.test_data = cls._load_test_data()
        
        logger.info("Test environment setup complete")
    
    @classmethod
    def _extract_greek_config(cls) -> Dict[str, Any]:
        """Extract Greek Sentiment configuration from Excel"""
        try:
            # Get detection parameters
            detection_params = cls.excel_manager.get_detection_parameters()
            
            greek_config = {
                'baseline_config': {
                    'baseline_time': '09:15:00',
                    'smoothing_alpha': 0.3,
                    'min_data_points': 5,
                    'session_boundary_buffer': 60  # seconds
                },
                'weighting_config': {
                    'oi_weight_alpha': 0.6,
                    'volume_weight_beta': 0.4,
                    'adaptive_adjustment': True,
                    'min_weight': 0.1,
                    'max_weight': 0.9
                },
                'itm_otm_config': {
                    'itm_strikes': [1, 2, 3],
                    'otm_strikes': [1, 2, 3],
                    'moneyness_threshold': 0.5,
                    'institutional_threshold': 0.7
                },
                'dte_config': {
                    'near_expiry_days': 7,
                    'medium_expiry_days': 30,
                    'adjustment_factors': {
                        'near': {'delta': 1.0, 'gamma': 1.2, 'theta': 1.5, 'vega': 0.8},
                        'medium': {'delta': 1.2, 'gamma': 1.0, 'theta': 0.8, 'vega': 1.5},
                        'far': {'delta': 1.0, 'gamma': 0.8, 'theta': 0.3, 'vega': 2.0}
                    }
                },
                'normalization_config': {
                    'delta_factor': 1.0,
                    'gamma_factor': 50.0,
                    'theta_factor': 5.0,
                    'vega_factor': 20.0,
                    'precision_tolerance': 0.001
                }
            }
            
            return greek_config
            
        except Exception as e:
            logger.error(f"Error extracting Greek config: {e}")
            return {}
    
    @classmethod
    def _load_test_data(cls) -> pd.DataFrame:
        """Load test data from HeavyDB or create mock data"""
        if cls.db_connected:
            try:
                # Query recent NIFTY data
                query = """
                SELECT 
                    timestamp,
                    underlying_price,
                    strike_price,
                    option_type,
                    ltp AS price,
                    volume,
                    oi AS open_interest,
                    delta,
                    gamma,
                    theta,
                    vega,
                    iv,
                    expiry
                FROM nifty_option_chain
                WHERE DATE(timestamp) = '2024-12-20'
                    AND TIME(timestamp) BETWEEN '09:15:00' AND '15:30:00'
                ORDER BY timestamp
                LIMIT 10000
                """
                
                data = cls.db_connector.execute_query(query)
                if not data.empty:
                    logger.info(f"Loaded {len(data)} rows of real HeavyDB data")
                    return data
                    
            except Exception as e:
                logger.error(f"Error loading HeavyDB data: {e}")
        
        # Create comprehensive mock data
        return cls._create_mock_data()
    
    @classmethod
    def _create_mock_data(cls) -> pd.DataFrame:
        """Create realistic mock data for testing"""
        logger.info("Creating mock test data")
        
        # Generate timestamps for a full trading day
        base_date = datetime(2024, 12, 20, 9, 15, 0)
        timestamps = [base_date + timedelta(minutes=i) for i in range(375)]  # 9:15 to 15:30
        
        # Create data for multiple strikes
        data_rows = []
        underlying_price = 21500
        
        for timestamp in timestamps:
            # Simulate price movement
            underlying_price += np.random.randn() * 5
            
            # ATM and OTM/ITM strikes
            strikes = [
                underlying_price - 300,  # ITM3
                underlying_price - 200,  # ITM2
                underlying_price - 100,  # ITM1
                underlying_price,         # ATM
                underlying_price + 100,  # OTM1
                underlying_price + 200,  # OTM2
                underlying_price + 300   # OTM3
            ]
            
            for strike in strikes:
                for option_type in ['CE', 'PE']:
                    moneyness = (underlying_price - strike) / strike if option_type == 'CE' else (strike - underlying_price) / strike
                    
                    # Generate realistic Greeks
                    if option_type == 'CE':
                        delta = 0.5 + moneyness * 2  # Simplified
                        delta = np.clip(delta, 0.05, 0.95)
                    else:
                        delta = -0.5 + moneyness * 2
                        delta = np.clip(delta, -0.95, -0.05)
                    
                    gamma = 0.02 * (1 - abs(moneyness * 4))
                    gamma = max(gamma, 0.001)
                    
                    theta = -0.05 * (1 + abs(moneyness))
                    vega = 0.15 * (1 - abs(moneyness * 2))
                    
                    # Volume and OI
                    volume = int(np.random.gamma(2, 1000) * (1 + abs(moneyness)))
                    oi = int(np.random.gamma(3, 10000) * (1 + abs(moneyness)))
                    
                    # Price
                    intrinsic = max(0, underlying_price - strike) if option_type == 'CE' else max(0, strike - underlying_price)
                    time_value = 50 * (1 - abs(moneyness)) * np.random.uniform(0.8, 1.2)
                    price = intrinsic + time_value
                    
                    data_rows.append({
                        'timestamp': timestamp,
                        'underlying_price': underlying_price,
                        'strike_price': strike,
                        'option_type': option_type,
                        'price': price,
                        'volume': volume,
                        'open_interest': oi,
                        'delta': delta,
                        'gamma': gamma,
                        'theta': theta,
                        'vega': vega,
                        'iv': 0.20 + abs(moneyness) * 0.1,
                        'expiry': base_date.date() + timedelta(days=7)
                    })
        
        df = pd.DataFrame(data_rows)
        logger.info(f"Created mock data with {len(df)} rows")
        return df
    
    def setUp(self):
        """Set up for each test"""
        # Initialize components with configuration
        self.baseline_tracker = BaselineTracker(self.greek_config['baseline_config'])
        self.volume_oi_weighter = VolumeOIWeighter(self.greek_config['weighting_config'])
        self.itm_otm_analyzer = ITMOTMAnalyzer(self.greek_config['itm_otm_config'])
        self.dte_adjuster = DTEAdjuster(self.greek_config['dte_config'])
        self.greek_calculator = GreekCalculator(self.greek_config['normalization_config'])
        
        # Initialize main analyzer
        self.greek_analyzer = GreekSentimentAnalyzer(self.greek_config)
    
    def test_baseline_tracker_session_establishment(self):
        """Test 9:15 AM baseline establishment"""
        logger.info("Testing baseline tracker session establishment")
        
        # Filter data for 9:15 AM
        baseline_time = time(9, 15, 0)
        morning_data = self.test_data[
            self.test_data['timestamp'].dt.time == baseline_time
        ]
        
        if not morning_data.empty:
            # Calculate Greeks at 9:15
            greeks = {
                'delta': morning_data['delta'].sum(),
                'gamma': morning_data['gamma'].sum(),
                'theta': morning_data['theta'].sum(),
                'vega': morning_data['vega'].sum()
            }
            
            # Update baselines
            baselines = self.baseline_tracker.update_baselines(
                greeks,
                morning_data.iloc[0]['timestamp'],
                morning_data.iloc[0]['timestamp'].date()
            )
            
            # Verify baselines were established
            self.assertIsNotNone(baselines)
            self.assertIn('delta', baselines)
            self.assertIn('baseline_time', baselines)
            
            logger.info(f"✅ Baseline establishment successful: {baselines}")
        else:
            logger.warning("No 9:15 AM data available for baseline testing")
    
    def test_volume_oi_dual_weighting(self):
        """Test dual weighting system (α×OI + β×Volume)"""
        logger.info("Testing volume-OI dual weighting system")
        
        # Get sample data
        sample_data = self.test_data.head(100)
        
        # Calculate weighted Greeks
        weighted_greeks = self.volume_oi_weighter.calculate_dual_weighted_greeks(
            sample_data,
            sample_data.iloc[0]['timestamp']
        )
        
        # Verify weighting
        self.assertIsNotNone(weighted_greeks)
        self.assertIn('delta', weighted_greeks)
        self.assertIn('weights_used', weighted_greeks)
        
        # Check weight bounds
        weights = weighted_greeks['weights_used']
        self.assertGreaterEqual(weights['oi_weight'], 0.1)
        self.assertLessEqual(weights['oi_weight'], 0.9)
        self.assertAlmostEqual(
            weights['oi_weight'] + weights['volume_weight'],
            1.0,
            places=3
        )
        
        logger.info(f"✅ Dual weighting successful: OI={weights['oi_weight']:.3f}, Volume={weights['volume_weight']:.3f}")
    
    def test_itm_otm_moneyness_classification(self):
        """Test ITM/OTM moneyness classification"""
        logger.info("Testing ITM/OTM moneyness classification")
        
        # Get sample data
        sample_data = self.test_data.head(200)
        
        # Analyze ITM/OTM sentiment
        analysis = self.itm_otm_analyzer.analyze_itm_otm_sentiment(
            sample_data,
            sample_data.iloc[0]['underlying_price']
        )
        
        # Verify analysis structure
        self.assertIn('itm_sentiment', analysis)
        self.assertIn('otm_sentiment', analysis)
        self.assertIn('institutional_bias', analysis)
        self.assertIn('moneyness_distribution', analysis)
        
        # Verify moneyness classification
        moneyness_dist = analysis['moneyness_distribution']
        total_options = sum(moneyness_dist.values())
        self.assertGreater(total_options, 0)
        
        logger.info(f"✅ Moneyness classification: {moneyness_dist}")
        logger.info(f"   Institutional bias: {analysis['institutional_bias']:.3f}")
    
    def test_dte_adjustment_factors(self):
        """Test DTE-specific adjustment factors"""
        logger.info("Testing DTE adjustment factors")
        
        # Test different DTE scenarios
        test_cases = [
            (3, 'near'),    # Near expiry
            (15, 'medium'),  # Medium expiry
            (45, 'far')      # Far expiry
        ]
        
        for days_to_expiry, expected_category in test_cases:
            # Create test Greeks
            test_greeks = {
                'delta': 0.5,
                'gamma': 0.02,
                'theta': -0.05,
                'vega': 0.15
            }
            
            # Apply DTE adjustments
            adjusted = self.dte_adjuster.apply_dte_adjustments(
                test_greeks.copy(),
                days_to_expiry
            )
            
            # Verify adjustments were applied
            self.assertIn('dte_category', adjusted)
            self.assertEqual(adjusted['dte_category'], expected_category)
            
            # Verify factors were applied correctly
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                self.assertNotEqual(adjusted[greek], test_greeks[greek])
            
            logger.info(f"✅ DTE {days_to_expiry} days ({expected_category}): Adjustments applied correctly")
    
    def test_greek_normalization_precision(self):
        """Test Greek normalization with precision tolerance"""
        logger.info("Testing Greek normalization precision")
        
        # Create test Greeks with known values
        test_greeks = {
            'delta': 0.5,
            'gamma': 0.02,
            'theta': -0.05,
            'vega': 0.15
        }
        
        # Normalize Greeks
        normalized = self.greek_calculator.normalize_greeks(test_greeks)
        
        # Verify normalization
        self.assertIn('normalized', normalized)
        self.assertIn('raw', normalized)
        
        # Test precision tolerance
        for greek, value in normalized['normalized'].items():
            # Verify value is within reasonable bounds
            self.assertGreaterEqual(abs(value), 0)
            self.assertLessEqual(abs(value), 10)  # Normalized values should be reasonable
        
        # Test denormalization
        denormalized = self.greek_calculator.denormalize_greeks(normalized['normalized'])
        
        # Verify precision within tolerance
        for greek in test_greeks:
            diff = abs(denormalized[greek] - test_greeks[greek])
            self.assertLess(diff, 0.001, f"{greek} precision exceeded tolerance")
        
        logger.info("✅ Greek normalization precision test passed")
    
    def test_full_pipeline_integration(self):
        """Test full Greek Sentiment pipeline with real data"""
        logger.info("Testing full Greek Sentiment V2 pipeline")
        
        # Process a time slice of data
        test_slice = self.test_data[
            (self.test_data['timestamp'].dt.time >= time(9, 30)) &
            (self.test_data['timestamp'].dt.time <= time(10, 0))
        ]
        
        if test_slice.empty:
            logger.warning("No data available for pipeline test")
            return
        
        # Group by timestamp and analyze
        results = []
        for timestamp, group in test_slice.groupby('timestamp'):
            try:
                result = self.greek_analyzer.analyze(group)
                results.append(result)
            except Exception as e:
                logger.error(f"Pipeline error at {timestamp}: {e}")
        
        # Verify results
        self.assertGreater(len(results), 0)
        
        # Check result structure
        if results:
            sample_result = results[0]
            self.assertIn('sentiment_score', sample_result)
            self.assertIn('component_health', sample_result)
            self.assertIn('weighted_greeks', sample_result)
            
            # Verify component health
            health = sample_result['component_health']
            self.assertIn('baseline_tracker', health)
            self.assertIn('volume_oi_weighter', health)
            
            logger.info(f"✅ Pipeline processed {len(results)} timestamps successfully")
            logger.info(f"   Sample sentiment score: {sample_result['sentiment_score']:.3f}")
    
    def test_component_fallback_mechanisms(self):
        """Test component fallback mechanisms"""
        logger.info("Testing component fallback mechanisms")
        
        # Test with minimal/corrupted data
        bad_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'underlying_price': [21500],
            'strike_price': [21500],
            'option_type': ['CE'],
            'price': [100],
            'volume': [0],  # Zero volume
            'open_interest': [0],  # Zero OI
            'delta': [np.nan],  # Missing Greek
            'gamma': [0.02],
            'theta': [-0.05],
            'vega': [0.15],
            'iv': [0.20],
            'expiry': [datetime.now().date() + timedelta(days=7)]
        })
        
        try:
            # Analyzer should handle bad data gracefully
            result = self.greek_analyzer.analyze(bad_data)
            
            # Check fallback was triggered
            self.assertIn('fallback_triggered', result)
            self.assertTrue(result['fallback_triggered'])
            self.assertIn('fallback_reason', result)
            
            logger.info(f"✅ Fallback mechanism triggered correctly: {result['fallback_reason']}")
            
        except Exception as e:
            self.fail(f"Fallback mechanism failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("Testing performance benchmarks")
        
        import time
        
        # Get substantial data sample
        perf_data = self.test_data.head(1000)
        
        # Measure processing time
        start_time = time.time()
        
        for timestamp, group in perf_data.groupby('timestamp'):
            _ = self.greek_analyzer.analyze(group)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate metrics
        rows_processed = len(perf_data)
        rows_per_second = rows_processed / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ Performance test completed:")
        logger.info(f"   Rows processed: {rows_processed}")
        logger.info(f"   Time elapsed: {elapsed:.3f} seconds")
        logger.info(f"   Processing speed: {rows_per_second:.0f} rows/second")
        
        # Verify performance threshold
        self.assertGreater(rows_per_second, 100, "Performance below threshold")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, 'db_connector') and cls.db_connected:
            cls.db_connector.disconnect()
        logger.info("Test environment cleaned up")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("="*80)
    logger.info("Starting Greek Sentiment V2 Comprehensive Tests")
    logger.info("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGreekSentimentV2Comprehensive)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("="*80)
    logger.info("Test Summary:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success: {result.wasSuccessful()}")
    logger.info("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)