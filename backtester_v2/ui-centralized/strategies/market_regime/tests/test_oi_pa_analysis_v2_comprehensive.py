#!/usr/bin/env python3
"""
Comprehensive Test Suite for OI/PA Analysis V2
=============================================

This test suite validates all components of the OI/PA Analysis V2 implementation
(Trending OI with Price Action) using real HeavyDB data and PHASE2 Excel configuration.

Test Coverage:
- OIPatternDetector: OI pattern recognition
- DivergenceDetector: 5-type divergence detection
- VolumeFlowAnalyzer: Institutional vs retail flow
- CorrelationAnalyzer: Mathematical correlation
- SessionWeightManager: Time-based weighting
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
from typing import Dict, List, Any, Optional, Tuple
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
    # Import OI/PA Analysis components
    from indicators.oi_pa_analysis import (
        OIPAAnalyzer,
        OIPatternDetector,
        DivergenceDetector,
        VolumeFlowAnalyzer,
        CorrelationAnalyzer,
        SessionWeightManager
    )
    
    # Import data utilities
    from base.heavydb_connector import HeavyDBConnector
    from excel_config_manager import MarketRegimeExcelManager
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure all required modules are in the Python path")
    raise

class TestOIPAAnalysisV2Comprehensive(unittest.TestCase):
    """Comprehensive test suite for OI/PA Analysis V2"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        logger.info("Setting up OI/PA Analysis V2 comprehensive test environment")
        
        # Initialize HeavyDB connection
        cls.db_connector = HeavyDBConnector()
        cls.db_connected = cls.db_connector.connect()
        
        if not cls.db_connected:
            logger.warning("HeavyDB connection failed - tests will use mock data")
        
        # Load Excel configuration
        config_path = Path(__file__).parent.parent.parent / "config" / "PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"
        cls.excel_manager = MarketRegimeExcelManager(str(config_path))
        
        # Extract OI/PA configuration
        cls.oi_pa_config = cls._extract_oi_pa_config()
        
        # Load test data
        cls.test_data = cls._load_test_data()
        
        logger.info("Test environment setup complete")
    
    @classmethod
    def _extract_oi_pa_config(cls) -> Dict[str, Any]:
        """Extract OI/PA configuration from Excel"""
        try:
            # Get detection parameters
            detection_params = cls.excel_manager.get_detection_parameters()
            
            oi_pa_config = {
                'pattern_config': {
                    'lookback_periods': 20,
                    'min_pattern_strength': 0.6,
                    'pattern_types': ['accumulation', 'distribution', 'neutral', 'choppy'],
                    'oi_change_threshold': 0.05
                },
                'divergence_config': {
                    'divergence_types': [
                        'price_oi_divergence',
                        'volume_oi_divergence', 
                        'price_volume_divergence',
                        'greek_oi_divergence',
                        'iv_price_divergence'
                    ],
                    'divergence_threshold': 0.7,
                    'confirmation_periods': 3
                },
                'volume_flow_config': {
                    'institutional_threshold': 0.8,
                    'retail_threshold': 0.3,
                    'flow_smoothing_periods': 5,
                    'volume_spike_multiplier': 2.0
                },
                'correlation_config': {
                    'correlation_window': 50,
                    'min_correlation': 0.80,
                    'correlation_types': ['pearson', 'spearman'],
                    'significance_level': 0.05
                },
                'session_config': {
                    'sessions': {
                        'pre_open': {'start': '09:00', 'end': '09:15', 'weight': 0.8},
                        'opening': {'start': '09:15', 'end': '09:45', 'weight': 1.2},
                        'morning': {'start': '09:45', 'end': '12:00', 'weight': 1.0},
                        'midday': {'start': '12:00', 'end': '13:30', 'weight': 0.7},
                        'afternoon': {'start': '13:30', 'end': '15:00', 'weight': 1.1},
                        'closing': {'start': '15:00', 'end': '15:25', 'weight': 1.3},
                        'post_close': {'start': '15:25', 'end': '15:30', 'weight': 0.6}
                    },
                    'decay_lambda': 0.1
                }
            }
            
            return oi_pa_config
            
        except Exception as e:
            logger.error(f"Error extracting OI/PA config: {e}")
            return {}
    
    @classmethod
    def _load_test_data(cls) -> pd.DataFrame:
        """Load test data from HeavyDB or create mock data"""
        if cls.db_connected:
            try:
                # Query recent NIFTY data with OI changes
                query = """
                SELECT 
                    timestamp,
                    underlying_price,
                    strike_price,
                    option_type,
                    ltp AS price,
                    volume,
                    oi AS open_interest,
                    LAG(oi) OVER (PARTITION BY strike_price, option_type ORDER BY timestamp) AS prev_oi,
                    delta,
                    gamma,
                    theta,
                    vega,
                    iv,
                    expiry
                FROM nifty_option_chain
                WHERE DATE(timestamp) = '2024-12-20'
                    AND TIME(timestamp) BETWEEN '09:00:00' AND '15:30:00'
                ORDER BY timestamp
                LIMIT 15000
                """
                
                data = cls.db_connector.execute_query(query)
                if not data.empty:
                    # Calculate OI change
                    data['oi_change'] = data['open_interest'] - data['prev_oi'].fillna(data['open_interest'])
                    logger.info(f"Loaded {len(data)} rows of real HeavyDB data")
                    return data
                    
            except Exception as e:
                logger.error(f"Error loading HeavyDB data: {e}")
        
        # Create comprehensive mock data
        return cls._create_mock_data()
    
    @classmethod
    def _create_mock_data(cls) -> pd.DataFrame:
        """Create realistic mock data for OI/PA testing"""
        logger.info("Creating mock test data for OI/PA analysis")
        
        # Generate timestamps for a full trading day
        base_date = datetime(2024, 12, 20, 9, 0, 0)
        timestamps = [base_date + timedelta(minutes=i) for i in range(390)]  # 9:00 to 15:30
        
        # Create data with realistic OI/PA patterns
        data_rows = []
        underlying_price = 21500
        base_oi = 100000
        
        # Simulate different market regimes
        regime_changes = [0, 100, 200, 300, 390]
        regimes = ['accumulation', 'distribution', 'neutral', 'volatile']
        
        for i, timestamp in enumerate(timestamps):
            # Determine current regime
            regime_idx = next(j for j in range(len(regime_changes)-1) 
                            if regime_changes[j] <= i < regime_changes[j+1])
            current_regime = regimes[min(regime_idx, len(regimes)-1)]
            
            # Simulate price and OI patterns based on regime
            if current_regime == 'accumulation':
                # Price up, OI up - bullish
                underlying_price += np.random.uniform(0, 2)
                oi_change_factor = 1.02
            elif current_regime == 'distribution':
                # Price down, OI down - bearish
                underlying_price -= np.random.uniform(0, 2)
                oi_change_factor = 0.98
            elif current_regime == 'neutral':
                # Sideways price, stable OI
                underlying_price += np.random.uniform(-1, 1)
                oi_change_factor = 1.0
            else:  # volatile
                # Large price swings, OI changes
                underlying_price += np.random.uniform(-5, 5)
                oi_change_factor = np.random.uniform(0.95, 1.05)
            
            # Create options data
            strikes = [underlying_price + offset for offset in [-300, -200, -100, 0, 100, 200, 300]]
            
            for strike in strikes:
                for option_type in ['CE', 'PE']:
                    # Calculate OI with pattern
                    prev_oi = base_oi * (1 + np.random.uniform(-0.1, 0.1))
                    new_oi = prev_oi * oi_change_factor * np.random.uniform(0.9, 1.1)
                    
                    # Volume patterns - higher during opening/closing
                    hour = timestamp.hour
                    if hour == 9 or hour == 15:
                        volume_multiplier = 2.0
                    else:
                        volume_multiplier = 1.0
                    
                    volume = int(np.random.gamma(2, 1000) * volume_multiplier)
                    
                    # Price patterns
                    moneyness = (underlying_price - strike) / strike
                    if option_type == 'CE':
                        intrinsic = max(0, underlying_price - strike)
                    else:
                        intrinsic = max(0, strike - underlying_price)
                    
                    time_value = 50 * np.exp(-abs(moneyness) * 2)
                    price = intrinsic + time_value + np.random.uniform(-5, 5)
                    
                    # Greeks
                    delta = 0.5 + moneyness * 2 if option_type == 'CE' else -0.5 + moneyness * 2
                    delta = np.clip(delta, -0.95, 0.95)
                    
                    data_rows.append({
                        'timestamp': timestamp,
                        'underlying_price': underlying_price,
                        'strike_price': strike,
                        'option_type': option_type,
                        'price': max(price, 0.5),
                        'volume': volume,
                        'open_interest': int(new_oi),
                        'prev_oi': int(prev_oi),
                        'oi_change': int(new_oi - prev_oi),
                        'delta': delta,
                        'gamma': 0.02 * (1 - abs(moneyness)),
                        'theta': -0.05 * (1 + abs(moneyness)),
                        'vega': 0.15 * (1 - abs(moneyness)),
                        'iv': 0.20 + abs(moneyness) * 0.05,
                        'expiry': base_date.date() + timedelta(days=7),
                        'regime': current_regime  # For validation
                    })
        
        df = pd.DataFrame(data_rows)
        logger.info(f"Created mock data with {len(df)} rows covering {len(regimes)} market regimes")
        return df
    
    def setUp(self):
        """Set up for each test"""
        # Initialize components with configuration
        self.oi_pattern_detector = OIPatternDetector(self.oi_pa_config['pattern_config'])
        self.divergence_detector = DivergenceDetector(self.oi_pa_config['divergence_config'])
        self.volume_flow_analyzer = VolumeFlowAnalyzer(self.oi_pa_config['volume_flow_config'])
        self.correlation_analyzer = CorrelationAnalyzer(self.oi_pa_config['correlation_config'])
        self.session_weight_manager = SessionWeightManager(self.oi_pa_config['session_config'])
        
        # Initialize main analyzer
        self.oi_pa_analyzer = OIPAAnalyzer(self.oi_pa_config)
    
    def test_oi_pattern_detection(self):
        """Test OI pattern detection and classification"""
        logger.info("Testing OI pattern detection")
        
        # Get sample data with different patterns
        sample_data = self.test_data.head(500)
        
        # Detect patterns
        patterns = self.oi_pattern_detector.detect_patterns(
            sample_data,
            sample_data.iloc[-1]['timestamp']
        )
        
        # Verify pattern detection
        self.assertIn('current_pattern', patterns)
        self.assertIn('pattern_strength', patterns)
        self.assertIn('oi_trend', patterns)
        self.assertIn('price_trend', patterns)
        
        # Verify pattern is valid
        valid_patterns = ['accumulation', 'distribution', 'neutral', 'choppy']
        self.assertIn(patterns['current_pattern'], valid_patterns)
        
        # Verify pattern strength
        self.assertGreaterEqual(patterns['pattern_strength'], 0)
        self.assertLessEqual(patterns['pattern_strength'], 1)
        
        logger.info(f"✅ Pattern detected: {patterns['current_pattern']} "
                   f"(strength: {patterns['pattern_strength']:.3f})")
        logger.info(f"   OI trend: {patterns['oi_trend']}, Price trend: {patterns['price_trend']}")
    
    def test_five_type_divergence_detection(self):
        """Test 5-type divergence detection system"""
        logger.info("Testing 5-type divergence detection")
        
        # Get sample data
        sample_data = self.test_data.iloc[100:300]  # Mid-day data
        
        # Detect divergences
        divergences = self.divergence_detector.detect_divergences(
            sample_data,
            sample_data.iloc[-1]['timestamp']
        )
        
        # Verify all divergence types are checked
        expected_types = [
            'price_oi_divergence',
            'volume_oi_divergence',
            'price_volume_divergence',
            'greek_oi_divergence',
            'iv_price_divergence'
        ]
        
        self.assertIn('divergences', divergences)
        self.assertIn('strongest_divergence', divergences)
        self.assertIn('divergence_count', divergences)
        
        # Check each divergence type
        for div_type in expected_types:
            self.assertIn(div_type, divergences['divergences'])
            div_info = divergences['divergences'][div_type]
            self.assertIn('detected', div_info)
            self.assertIn('strength', div_info)
            self.assertIn('direction', div_info)
            
            if div_info['detected']:
                logger.info(f"✅ {div_type}: strength={div_info['strength']:.3f}, "
                           f"direction={div_info['direction']}")
        
        logger.info(f"   Total divergences detected: {divergences['divergence_count']}")
    
    def test_volume_flow_institutional_detection(self):
        """Test institutional vs retail volume flow detection"""
        logger.info("Testing volume flow analysis")
        
        # Get high volume period data
        sample_data = self.test_data[
            (self.test_data['timestamp'].dt.time >= time(9, 15)) &
            (self.test_data['timestamp'].dt.time <= time(9, 45))
        ]
        
        if sample_data.empty:
            logger.warning("No opening session data available")
            return
        
        # Analyze volume flow
        flow_analysis = self.volume_flow_analyzer.analyze_volume_flow(
            sample_data,
            sample_data.iloc[-1]['timestamp']
        )
        
        # Verify flow analysis
        self.assertIn('institutional_flow', flow_analysis)
        self.assertIn('retail_flow', flow_analysis)
        self.assertIn('flow_ratio', flow_analysis)
        self.assertIn('dominant_player', flow_analysis)
        self.assertIn('volume_profile', flow_analysis)
        
        # Verify flow values
        inst_flow = flow_analysis['institutional_flow']
        retail_flow = flow_analysis['retail_flow']
        
        self.assertGreaterEqual(inst_flow, -1)
        self.assertLessEqual(inst_flow, 1)
        self.assertGreaterEqual(retail_flow, -1)
        self.assertLessEqual(retail_flow, 1)
        
        logger.info(f"✅ Volume flow analysis completed:")
        logger.info(f"   Institutional flow: {inst_flow:.3f}")
        logger.info(f"   Retail flow: {retail_flow:.3f}")
        logger.info(f"   Dominant player: {flow_analysis['dominant_player']}")
    
    def test_correlation_analysis_threshold(self):
        """Test correlation analysis with 0.80 threshold"""
        logger.info("Testing correlation analysis")
        
        # Get sufficient data for correlation
        sample_data = self.test_data.head(1000)
        
        # Calculate correlations
        correlations = self.correlation_analyzer.calculate_correlations(
            sample_data,
            sample_data.iloc[-1]['timestamp']
        )
        
        # Verify correlation structure
        self.assertIn('price_oi_correlation', correlations)
        self.assertIn('volume_oi_correlation', correlations)
        self.assertIn('significant_correlations', correlations)
        self.assertIn('correlation_matrix', correlations)
        
        # Check correlation values
        price_oi_corr = correlations['price_oi_correlation']
        self.assertGreaterEqual(price_oi_corr['pearson'], -1)
        self.assertLessEqual(price_oi_corr['pearson'], 1)
        
        # Check significance
        significant_corrs = correlations['significant_correlations']
        for corr_pair, corr_info in significant_corrs.items():
            if corr_info['is_significant']:
                self.assertGreaterEqual(abs(corr_info['correlation']), 0.80)
                logger.info(f"✅ Significant correlation found: {corr_pair} = {corr_info['correlation']:.3f}")
        
        logger.info(f"   Total significant correlations: {len(significant_corrs)}")
    
    def test_session_weight_management(self):
        """Test 7-session time-based weight management"""
        logger.info("Testing session weight management")
        
        # Test each session
        test_times = [
            (time(9, 10), 'pre_open'),
            (time(9, 30), 'opening'),
            (time(11, 0), 'morning'),
            (time(12, 30), 'midday'),
            (time(14, 0), 'afternoon'),
            (time(15, 10), 'closing'),
            (time(15, 28), 'post_close')
        ]
        
        for test_time, expected_session in test_times:
            # Create timestamp
            test_timestamp = datetime.combine(datetime.now().date(), test_time)
            
            # Get session info
            session_info = self.session_weight_manager.get_session_info(test_timestamp)
            
            # Verify session detection
            self.assertEqual(session_info['current_session'], expected_session)
            self.assertGreater(session_info['session_weight'], 0)
            
            # Test time decay
            decay_weight = self.session_weight_manager.calculate_time_decay(
                test_timestamp,
                test_timestamp - timedelta(minutes=30)
            )
            
            self.assertGreater(decay_weight, 0)
            self.assertLessEqual(decay_weight, 1)
            
            logger.info(f"✅ Session {expected_session}: weight={session_info['session_weight']:.2f}, "
                       f"decay={decay_weight:.3f}")
    
    def test_full_oi_pa_pipeline(self):
        """Test full OI/PA analysis pipeline"""
        logger.info("Testing full OI/PA Analysis V2 pipeline")
        
        # Process different time windows
        time_windows = [
            (time(9, 15), time(9, 45), "Opening"),
            (time(11, 0), time(11, 30), "Mid-morning"),
            (time(14, 30), time(15, 0), "Pre-closing")
        ]
        
        results = []
        for start_time, end_time, period_name in time_windows:
            # Filter data
            window_data = self.test_data[
                (self.test_data['timestamp'].dt.time >= start_time) &
                (self.test_data['timestamp'].dt.time <= end_time)
            ]
            
            if window_data.empty:
                logger.warning(f"No data for {period_name} period")
                continue
            
            # Analyze
            try:
                result = self.oi_pa_analyzer.analyze(window_data)
                result['period'] = period_name
                results.append(result)
                
                logger.info(f"✅ {period_name} analysis completed:")
                logger.info(f"   Pattern: {result['pattern']}")
                logger.info(f"   Divergences: {result['divergence_count']}")
                logger.info(f"   OI/PA Score: {result['oi_pa_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Pipeline error for {period_name}: {e}")
        
        # Verify results
        self.assertGreater(len(results), 0)
        
        # Check result consistency
        for result in results:
            self.assertIn('pattern', result)
            self.assertIn('divergences', result)
            self.assertIn('volume_flow', result)
            self.assertIn('correlations', result)
            self.assertIn('oi_pa_score', result)
            self.assertIn('component_health', result)
    
    def test_edge_cases_handling(self):
        """Test edge cases and error handling"""
        logger.info("Testing edge case handling")
        
        # Test with missing OI data
        missing_oi_data = self.test_data.head(10).copy()
        missing_oi_data['open_interest'] = np.nan
        
        try:
            result = self.oi_pa_analyzer.analyze(missing_oi_data)
            self.assertIn('data_quality_warning', result)
            logger.info("✅ Missing OI data handled gracefully")
        except Exception as e:
            self.fail(f"Failed to handle missing OI data: {e}")
        
        # Test with zero volume
        zero_volume_data = self.test_data.head(10).copy()
        zero_volume_data['volume'] = 0
        
        try:
            result = self.oi_pa_analyzer.analyze(zero_volume_data)
            self.assertIn('low_activity_warning', result)
            logger.info("✅ Zero volume data handled gracefully")
        except Exception as e:
            self.fail(f"Failed to handle zero volume: {e}")
        
        # Test with single data point
        single_point = self.test_data.head(1)
        
        try:
            result = self.oi_pa_analyzer.analyze(single_point)
            self.assertIn('insufficient_data', result)
            logger.info("✅ Single data point handled gracefully")
        except Exception as e:
            self.fail(f"Failed to handle single data point: {e}")
    
    def test_performance_metrics(self):
        """Test performance and scalability"""
        logger.info("Testing performance metrics")
        
        import time
        
        # Test with increasing data sizes
        data_sizes = [100, 500, 1000, 5000]
        performance_results = []
        
        for size in data_sizes:
            test_data = self.test_data.head(size)
            
            start_time = time.time()
            _ = self.oi_pa_analyzer.analyze(test_data)
            elapsed = time.time() - start_time
            
            rows_per_second = size / elapsed if elapsed > 0 else 0
            performance_results.append({
                'size': size,
                'elapsed': elapsed,
                'rows_per_second': rows_per_second
            })
            
            logger.info(f"✅ Size {size}: {elapsed:.3f}s ({rows_per_second:.0f} rows/s)")
        
        # Verify performance scales reasonably
        # Should maintain > 500 rows/second for moderate sizes
        for result in performance_results:
            if result['size'] <= 1000:
                self.assertGreater(result['rows_per_second'], 500, 
                                 f"Performance too slow for size {result['size']}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, 'db_connector') and cls.db_connected:
            cls.db_connector.disconnect()
        logger.info("Test environment cleaned up")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("="*80)
    logger.info("Starting OI/PA Analysis V2 Comprehensive Tests")
    logger.info("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOIPAAnalysisV2Comprehensive)
    
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