#!/usr/bin/env python3
"""
Comprehensive Test of Refactored Triple Straddle Analysis with Real HeavyDB Data

This test validates the new modular straddle analysis system using:
- Real HeavyDB NIFTY option data
- ML Triple Rolling Straddle input configuration
- Performance validation (<3 seconds target)
- Accuracy comparison with old system

Author: Claude Code
Date: 2025-07-06
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import time
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from concurrent.futures import ThreadPoolExecutor

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2')

# Import new refactored components
from strategies.market_regime.indicators.straddle_analysis import (
    TripleStraddleEngine,
    TripleStraddleAnalysisResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('refactored_straddle_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HeavyDBDataProvider:
    """Provides real HeavyDB data for testing"""
    
    def __init__(self):
        self.connection = None
        self.data_cache = {}
        
    def connect(self) -> bool:
        """Connect to HeavyDB"""
        try:
            import pymapd
            self.connection = pymapd.connect(
                host='localhost',
                port=6274,
                user='admin',
                password='HyperInteractive',
                dbname='heavyai'
            )
            logger.info("Connected to HeavyDB successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to HeavyDB: {e}")
            return False
    
    def get_sample_data(self, date: str = '2024-12-01', limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample NIFTY option data from HeavyDB"""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            # Query to get comprehensive option data
            query = f"""
            SELECT 
                timestamp,
                underlying_value as underlying_price,
                atm_ce_ltp as ATM_CE,
                atm_pe_ltp as ATM_PE,
                itm1_ce_ltp as ITM1_CE,
                itm1_pe_ltp as ITM1_PE,
                otm1_ce_ltp as OTM1_CE,
                otm1_pe_ltp as OTM1_PE,
                atm_ce_volume,
                atm_pe_volume,
                atm_ce_oi,
                atm_pe_oi,
                atm_ce_delta,
                atm_pe_delta,
                atm_ce_gamma,
                atm_pe_gamma,
                atm_ce_theta,
                atm_pe_theta,
                atm_ce_vega,
                atm_pe_vega,
                vix_value,
                iv_surface_atm
            FROM nifty_option_chain 
            WHERE DATE(timestamp) = '{date}'
            AND underlying_value > 0
            AND atm_ce_ltp > 0 
            AND atm_pe_ltp > 0
            ORDER BY timestamp
            LIMIT {limit}
            """
            
            df = pd.read_sql(query, self.connection)
            
            if df.empty:
                logger.warning(f"No data found for date {date}")
                return []
            
            # Convert to list of dictionaries
            data_list = []
            for _, row in df.iterrows():
                data_point = {
                    'timestamp': pd.Timestamp(row['timestamp']),
                    'underlying_price': float(row['underlying_price']),
                    'spot_price': float(row['underlying_price']),
                    
                    # Option prices
                    'ATM_CE': float(row['ATM_CE']) if pd.notna(row['ATM_CE']) else 0,
                    'ATM_PE': float(row['ATM_PE']) if pd.notna(row['ATM_PE']) else 0,
                    'ITM1_CE': float(row['ITM1_CE']) if pd.notna(row['ITM1_CE']) else 0,
                    'ITM1_PE': float(row['ITM1_PE']) if pd.notna(row['ITM1_PE']) else 0,
                    'OTM1_CE': float(row['OTM1_CE']) if pd.notna(row['OTM1_CE']) else 0,
                    'OTM1_PE': float(row['OTM1_PE']) if pd.notna(row['OTM1_PE']) else 0,
                    
                    # Volume data
                    'volume': float(row['atm_ce_volume']) if pd.notna(row['atm_ce_volume']) else 0,
                    'atm_ce_volume': float(row['atm_ce_volume']) if pd.notna(row['atm_ce_volume']) else 0,
                    'atm_pe_volume': float(row['atm_pe_volume']) if pd.notna(row['atm_pe_volume']) else 0,
                    
                    # Open Interest
                    'open_interest': float(row['atm_ce_oi']) if pd.notna(row['atm_ce_oi']) else 0,
                    'atm_ce_oi': float(row['atm_ce_oi']) if pd.notna(row['atm_ce_oi']) else 0,
                    'atm_pe_oi': float(row['atm_pe_oi']) if pd.notna(row['atm_pe_oi']) else 0,
                    
                    # Greeks
                    'atm_ce_delta': float(row['atm_ce_delta']) if pd.notna(row['atm_ce_delta']) else 0.5,
                    'atm_pe_delta': float(row['atm_pe_delta']) if pd.notna(row['atm_pe_delta']) else -0.5,
                    'atm_ce_gamma': float(row['atm_ce_gamma']) if pd.notna(row['atm_ce_gamma']) else 0.02,
                    'atm_pe_gamma': float(row['atm_pe_gamma']) if pd.notna(row['atm_pe_gamma']) else 0.02,
                    'atm_ce_theta': float(row['atm_ce_theta']) if pd.notna(row['atm_ce_theta']) else -20,
                    'atm_pe_theta': float(row['atm_pe_theta']) if pd.notna(row['atm_pe_theta']) else -20,
                    'atm_ce_vega': float(row['atm_ce_vega']) if pd.notna(row['atm_ce_vega']) else 50,
                    'atm_pe_vega': float(row['atm_pe_vega']) if pd.notna(row['atm_pe_vega']) else 50,
                    
                    # Market data
                    'vix': float(row['vix_value']) if pd.notna(row['vix_value']) else 15,
                    'implied_volatility': float(row['iv_surface_atm']) if pd.notna(row['iv_surface_atm']) else 0.15,
                    
                    # OHLC approximation
                    'high': float(row['underlying_price']) * 1.001,
                    'low': float(row['underlying_price']) * 0.999,
                    'open': float(row['underlying_price']),
                    'close': float(row['underlying_price'])
                }
                data_list.append(data_point)
            
            logger.info(f"Retrieved {len(data_list)} data points from HeavyDB")
            return data_list
            
        except Exception as e:
            logger.error(f"Error querying HeavyDB: {e}")
            return []
    
    def close(self):
        """Close HeavyDB connection"""
        if self.connection:
            self.connection.close()
            logger.info("HeavyDB connection closed")


class RefactoredStraddleValidator:
    """Validates the refactored straddle analysis system"""
    
    def __init__(self):
        self.data_provider = HeavyDBDataProvider()
        self.engine = None
        self.test_results = {
            'initialization': {},
            'performance': {},
            'accuracy': {},
            'functionality': {},
            'edge_cases': {}
        }
        self.performance_target = 3.0  # 3 seconds
        
    def setup(self) -> bool:
        """Setup test environment"""
        logger.info("Setting up Refactored Straddle Validator")
        
        try:
            # Connect to HeavyDB
            if not self.data_provider.connect():
                logger.error("Failed to connect to HeavyDB")
                return False
            
            # Initialize the new engine with ML config
            config_path = '/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/ML_tripple_rolling_straddle/ML_Triple_Rolling_Straddle_COMPLETE_CONFIG.xlsx'
            
            start_time = time.time()
            self.engine = TripleStraddleEngine(config_path)
            init_time = time.time() - start_time
            
            self.test_results['initialization'] = {
                'time': init_time,
                'success': True,
                'config_loaded': True
            }
            
            logger.info(f"Engine initialized successfully in {init_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            self.test_results['initialization'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_single_analysis_performance(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Test single analysis performance"""
        logger.info("Testing single analysis performance")
        
        timestamp = data_point['timestamp']
        
        # Run analysis multiple times for accurate timing
        times = []
        results = []
        
        for i in range(5):  # 5 runs for average
            start_time = time.time()
            result = self.engine.analyze(data_point, timestamp)
            analysis_time = time.time() - start_time
            
            times.append(analysis_time)
            results.append(result)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        performance_result = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'target_met': avg_time <= self.performance_target,
            'success_rate': sum(1 for r in results if r is not None) / len(results),
            'valid_results': len([r for r in results if r is not None])
        }
        
        logger.info(f"Performance: {avg_time:.2f}s avg, target {'MET' if performance_result['target_met'] else 'MISSED'}")
        return performance_result
    
    def test_batch_analysis(self, data_points: List[Dict[str, Any]], batch_size: int = 20) -> Dict[str, Any]:
        """Test batch analysis for consistency"""
        logger.info(f"Testing batch analysis with {len(data_points)} data points")
        
        batch_results = []
        analysis_times = []
        regimes = []
        confidences = []
        
        for i, data_point in enumerate(data_points[:batch_size]):
            timestamp = data_point['timestamp']
            
            start_time = time.time()
            result = self.engine.analyze(data_point, timestamp)
            analysis_time = time.time() - start_time
            
            analysis_times.append(analysis_time)
            
            if result:
                batch_results.append(result)
                regimes.append(result.market_regime)
                confidences.append(result.regime_confidence)
            
            # Progress logging
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{batch_size} analyses")
        
        # Calculate statistics
        avg_time = np.mean(analysis_times)
        success_rate = len(batch_results) / batch_size
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Regime distribution
        regime_distribution = pd.Series(regimes).value_counts().to_dict() if regimes else {}
        
        batch_result = {
            'total_processed': batch_size,
            'successful_analyses': len(batch_results),
            'success_rate': success_rate,
            'avg_analysis_time': avg_time,
            'avg_regime_confidence': avg_confidence,
            'regime_distribution': regime_distribution,
            'within_target_rate': sum(1 for t in analysis_times if t <= self.performance_target) / len(analysis_times)
        }
        
        logger.info(f"Batch results: {success_rate:.1%} success, {avg_time:.2f}s avg time")
        return batch_result
    
    def test_component_functionality(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual component functionality"""
        logger.info("Testing component functionality")
        
        timestamp = data_point['timestamp']
        result = self.engine.analyze(data_point, timestamp)
        
        if not result:
            return {'success': False, 'error': 'Analysis failed'}
        
        functionality_test = {
            'straddle_analysis': {
                'available': result.straddle_result is not None,
                'atm_straddle': result.straddle_result.atm_result is not None if result.straddle_result else False,
                'itm1_straddle': result.straddle_result.itm1_result is not None if result.straddle_result else False,
                'otm1_straddle': result.straddle_result.otm1_result is not None if result.straddle_result else False,
                'combined_price': result.straddle_result.combined_price if result.straddle_result else 0
            },
            'resistance_analysis': {
                'available': result.resistance_result is not None,
                'support_levels': len(result.resistance_result.support_levels) if result.resistance_result else 0,
                'resistance_levels': len(result.resistance_result.resistance_levels) if result.resistance_result else 0
            },
            'correlation_analysis': {
                'available': result.correlation_result is not None,
                'matrix_populated': bool(result.correlation_result) if result.correlation_result else False
            },
            'regime_detection': {
                'regime_identified': bool(result.market_regime),
                'confidence_level': result.regime_confidence,
                'regime_type': result.market_regime
            },
            'recommendations': {
                'position_provided': bool(result.position_recommendations),
                'risk_calculated': bool(result.risk_parameters)
            }
        }
        
        return functionality_test
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling"""
        logger.info("Testing edge cases")
        
        edge_tests = {}
        
        # Test 1: Missing option data
        incomplete_data = {
            'timestamp': pd.Timestamp.now(),
            'underlying_price': 21500,
            'ATM_CE': 150,
            'ATM_PE': 150
            # Missing ITM1 and OTM1 data
        }
        
        result = self.engine.analyze(incomplete_data)
        edge_tests['missing_options'] = {
            'handled': result is not None,
            'graceful_degradation': True if result else False
        }
        
        # Test 2: Extreme market conditions
        extreme_data = {
            'timestamp': pd.Timestamp.now(),
            'underlying_price': 21500,
            'ATM_CE': 500,  # Very expensive options
            'ATM_PE': 500,
            'ITM1_CE': 600,
            'ITM1_PE': 600,
            'OTM1_CE': 50,
            'OTM1_PE': 50,
            'implied_volatility': 0.5  # 50% IV
        }
        
        result = self.engine.analyze(extreme_data)
        edge_tests['extreme_conditions'] = {
            'handled': result is not None,
            'reasonable_regime': result.market_regime if result else None
        }
        
        # Test 3: Invalid data types
        try:
            invalid_data = {
                'timestamp': 'invalid_timestamp',
                'underlying_price': 'not_a_number'
            }
            result = self.engine.analyze(invalid_data)
            edge_tests['invalid_data'] = {
                'handled': False,  # Should fail gracefully
                'error_caught': False
            }
        except Exception:
            edge_tests['invalid_data'] = {
                'handled': True,
                'error_caught': True
            }
        
        return edge_tests
    
    def compare_with_old_system(self, data_points: List[Dict[str, Any]], sample_size: int = 5) -> Dict[str, Any]:
        """Compare results with old system (if available)"""
        logger.info(f"Comparing with old system using {sample_size} samples")
        
        # Note: This would require the old system to be available
        # For now, we'll just validate consistency of our new system
        
        new_results = []
        for data_point in data_points[:sample_size]:
            result = self.engine.analyze(data_point)
            if result:
                new_results.append({
                    'regime': result.market_regime,
                    'confidence': result.regime_confidence,
                    'combined_price': result.straddle_result.combined_price if result.straddle_result else 0
                })
        
        # Check consistency
        regimes = [r['regime'] for r in new_results]
        confidences = [r['confidence'] for r in new_results]
        
        comparison = {
            'new_system_consistency': {
                'regime_stability': len(set(regimes)) / len(regimes) if regimes else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'results_count': len(new_results)
            },
            'validation': {
                'reasonable_regimes': all(isinstance(r, str) and r for r in regimes),
                'confidence_range': all(0 <= c <= 1 for c in confidences)
            }
        }
        
        return comparison
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'system_info': {
                'engine_version': '3.0.0',
                'config_source': 'ML_Triple_Rolling_Straddle_COMPLETE_CONFIG.xlsx',
                'data_source': 'HeavyDB NIFTY Option Chain'
            },
            'test_results': self.test_results,
            'summary': {
                'overall_success': all([
                    self.test_results.get('initialization', {}).get('success', False),
                    self.test_results.get('performance', {}).get('target_met', False),
                    self.test_results.get('functionality', {}).get('success', False)
                ]),
                'performance_target_met': self.test_results.get('performance', {}).get('target_met', False),
                'ready_for_production': False  # Will be set based on all tests
            },
            'recommendations': []
        }
        
        # Add recommendations based on results
        if not report['summary']['performance_target_met']:
            report['recommendations'].append('Optimize performance to meet <3 second target')
        
        if self.test_results.get('functionality', {}).get('success', False):
            report['recommendations'].append('System ready for integration testing')
        
        # Set production readiness
        report['summary']['ready_for_production'] = (
            report['summary']['overall_success'] and
            report['summary']['performance_target_met']
        )
        
        return report
    
    def run_comprehensive_test(self):
        """Run the complete test suite"""
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE REFACTORED STRADDLE ANALYSIS TEST")
        logger.info("=" * 80)
        
        # Setup
        if not self.setup():
            logger.error("Test setup failed. Exiting.")
            return
        
        try:
            # Get real data from HeavyDB
            logger.info("\nFetching real data from HeavyDB...")
            data_points = self.data_provider.get_sample_data(date='2024-12-01', limit=50)
            
            if not data_points:
                logger.error("No data available from HeavyDB")
                return
            
            logger.info(f"Retrieved {len(data_points)} data points")
            
            # Test 1: Single Analysis Performance
            logger.info("\n--- Test 1: Single Analysis Performance ---")
            perf_result = self.test_single_analysis_performance(data_points[0])
            self.test_results['performance'] = perf_result
            
            # Test 2: Batch Analysis
            logger.info("\n--- Test 2: Batch Analysis ---")
            batch_result = self.test_batch_analysis(data_points, batch_size=20)
            self.test_results['batch'] = batch_result
            
            # Test 3: Component Functionality
            logger.info("\n--- Test 3: Component Functionality ---")
            func_result = self.test_component_functionality(data_points[0])
            self.test_results['functionality'] = func_result
            
            # Test 4: Edge Cases
            logger.info("\n--- Test 4: Edge Cases ---")
            edge_result = self.test_edge_cases()
            self.test_results['edge_cases'] = edge_result
            
            # Test 5: System Comparison
            logger.info("\n--- Test 5: System Consistency ---")
            comparison_result = self.compare_with_old_system(data_points, sample_size=10)
            self.test_results['comparison'] = comparison_result
            
            # Generate Report
            logger.info("\n--- Generating Comprehensive Report ---")
            report = self.generate_comprehensive_report()
            
            # Save report
            report_path = 'refactored_straddle_test_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Print Summary
            self.print_test_summary(report)
            
            logger.info(f"\nDetailed report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
        
        finally:
            # Cleanup
            self.data_provider.close()
    
    def print_test_summary(self, report: Dict[str, Any]):
        """Print test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        # Performance Summary
        perf = self.test_results.get('performance', {})
        logger.info(f"\nPerformance:")
        logger.info(f"  Average Time: {perf.get('avg_time', 0):.3f} seconds")
        logger.info(f"  Target (<3s): {'✓ MET' if perf.get('target_met', False) else '✗ MISSED'}")
        logger.info(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
        
        # Batch Results
        batch = self.test_results.get('batch', {})
        logger.info(f"\nBatch Analysis ({batch.get('total_processed', 0)} samples):")
        logger.info(f"  Success Rate: {batch.get('success_rate', 0):.1%}")
        logger.info(f"  Avg Time: {batch.get('avg_analysis_time', 0):.3f}s")
        logger.info(f"  Avg Confidence: {batch.get('avg_regime_confidence', 0):.1%}")
        
        # Functionality
        func = self.test_results.get('functionality', {})
        if func:
            logger.info(f"\nFunctionality:")
            straddle = func.get('straddle_analysis', {})
            logger.info(f"  Straddle Analysis: {'✓' if straddle.get('available') else '✗'}")
            logger.info(f"  ATM/ITM1/OTM1: {straddle.get('atm_straddle')}/{straddle.get('itm1_straddle')}/{straddle.get('otm1_straddle')}")
            logger.info(f"  Resistance Analysis: {'✓' if func.get('resistance_analysis', {}).get('available') else '✗'}")
            logger.info(f"  Regime Detection: {'✓' if func.get('regime_detection', {}).get('regime_identified') else '✗'}")
        
        # Edge Cases
        edge = self.test_results.get('edge_cases', {})
        logger.info(f"\nEdge Cases:")
        for test_name, result in edge.items():
            status = '✓' if result.get('handled', False) else '✗'
            logger.info(f"  {test_name}: {status}")
        
        # Overall Result
        overall = report['summary']['ready_for_production']
        logger.info(f"\nOVERALL RESULT: {'✅ READY FOR PRODUCTION' if overall else '❌ NEEDS ATTENTION'}")
        
        if report['recommendations']:
            logger.info(f"\nRecommendations:")
            for rec in report['recommendations']:
                logger.info(f"  • {rec}")
        
        logger.info("=" * 80)


if __name__ == "__main__":
    validator = RefactoredStraddleValidator()
    validator.run_comprehensive_test()