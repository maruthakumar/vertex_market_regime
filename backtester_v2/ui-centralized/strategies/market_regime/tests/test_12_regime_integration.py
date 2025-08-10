#!/usr/bin/env python3
"""
Comprehensive 12-Regime System Integration Test

This module provides comprehensive integration testing for the 12-regime system
including Excel configuration, regime classification, mapping validation,
and performance benchmarking.

Author: The Augster
Date: 2025-06-18
Version: 1.0.0
"""

import time
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Comprehensive12RegimeIntegrationTest:
    """Comprehensive integration test suite for 12-regime system"""
    
    def __init__(self):
        """Initialize integration test suite"""
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = datetime.now()
        
        logger.info("🚀 Initializing 12-Regime Integration Test Suite")
    
    def test_12_regime_detector_integration(self) -> bool:
        """Test 12-regime detector integration"""
        try:
            logger.info("📦 Testing 12-regime detector integration...")
            
            from enhanced_12_regime_detector import Enhanced12RegimeDetector
            
            # Initialize detector
            detector = Enhanced12RegimeDetector()
            
            # Validate initialization
            assert len(detector.regime_definitions) == 12, f"Expected 12 regimes, got {len(detector.regime_definitions)}"
            assert len(detector.regime_mapping_18_to_12) >= 18, f"Expected ≥18 mappings, got {len(detector.regime_mapping_18_to_12)}"
            
            # Test classification with multiple scenarios
            test_scenarios = [
                {
                    'name': 'Low Volatility Directional Trending',
                    'data': {
                        'iv_percentile': 0.15, 'atr_normalized': 0.10, 'gamma_exposure': 0.05,
                        'ema_alignment': 0.85, 'price_momentum': 0.80, 'volume_confirmation': 0.75,
                        'strike_correlation': 0.90, 'vwap_deviation': 0.85, 'pivot_analysis': 0.80
                    },
                    'expected_volatility': 'LOW',
                    'expected_trend': 'DIRECTIONAL',
                    'expected_structure': 'TRENDING'
                },
                {
                    'name': 'High Volatility Non-Directional Range',
                    'data': {
                        'iv_percentile': 0.95, 'atr_normalized': 0.90, 'gamma_exposure': 0.85,
                        'ema_alignment': 0.05, 'price_momentum': 0.10, 'volume_confirmation': 0.15,
                        'strike_correlation': 0.30, 'vwap_deviation': 0.25, 'pivot_analysis': 0.35
                    },
                    'expected_volatility': 'HIGH',
                    'expected_trend': 'NONDIRECTIONAL',
                    'expected_structure': 'RANGE'
                },
                {
                    'name': 'Moderate Volatility Directional Range',
                    'data': {
                        'iv_percentile': 0.50, 'atr_normalized': 0.45, 'gamma_exposure': 0.40,
                        'ema_alignment': 0.60, 'price_momentum': 0.55, 'volume_confirmation': 0.50,
                        'strike_correlation': 0.45, 'vwap_deviation': 0.40, 'pivot_analysis': 0.50
                    },
                    'expected_volatility': 'MODERATE',
                    'expected_trend': 'DIRECTIONAL',
                    'expected_structure': 'RANGE'
                }
            ]
            
            classification_results = []
            
            for scenario in test_scenarios:
                start_time = time.time()
                result = detector.classify_12_regime(scenario['data'])
                processing_time = time.time() - start_time
                
                # Validate result structure
                assert hasattr(result, 'regime_id'), "Result missing regime_id"
                assert hasattr(result, 'confidence'), "Result missing confidence"
                assert 0.0 <= result.confidence <= 1.0, f"Invalid confidence: {result.confidence}"
                
                # Validate component classification
                assert result.volatility_level == scenario['expected_volatility'], \
                    f"Expected {scenario['expected_volatility']}, got {result.volatility_level}"
                assert result.trend_type == scenario['expected_trend'], \
                    f"Expected {scenario['expected_trend']}, got {result.trend_type}"
                assert result.structure_type == scenario['expected_structure'], \
                    f"Expected {scenario['expected_structure']}, got {result.structure_type}"
                
                # Performance validation
                assert processing_time < 3.0, f"Processing time {processing_time:.3f}s exceeds 3s limit"
                
                classification_results.append({
                    'scenario': scenario['name'],
                    'regime': result.regime_id,
                    'confidence': result.confidence,
                    'processing_time': processing_time
                })
                
                logger.info(f"✅ {scenario['name']}: {result.regime_id} (confidence: {result.confidence:.3f}, time: {processing_time:.3f}s)")
            
            self.test_results['12_regime_detector'] = {
                'status': 'PASSED',
                'classification_results': classification_results,
                'total_scenarios': len(test_scenarios)
            }
            
            logger.info("✅ 12-regime detector integration test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ 12-regime detector integration test FAILED: {e}")
            self.test_results['12_regime_detector'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_18_to_12_mapping_accuracy(self) -> bool:
        """Test 18→12 regime mapping accuracy"""
        try:
            logger.info("🔄 Testing 18→12 regime mapping accuracy...")
            
            from enhanced_12_regime_detector import Enhanced12RegimeDetector
            detector = Enhanced12RegimeDetector()
            
            # Comprehensive mapping test cases
            mapping_test_cases = [
                ('HIGH_VOLATILE_STRONG_BULLISH', 'HIGH_DIRECTIONAL_TRENDING'),
                ('HIGH_VOLATILE_MILD_BULLISH', 'HIGH_DIRECTIONAL_RANGE'),
                ('NORMAL_VOLATILE_STRONG_BULLISH', 'MODERATE_DIRECTIONAL_TRENDING'),
                ('NORMAL_VOLATILE_MILD_BULLISH', 'MODERATE_DIRECTIONAL_RANGE'),
                ('LOW_VOLATILE_STRONG_BULLISH', 'LOW_DIRECTIONAL_TRENDING'),
                ('LOW_VOLATILE_MILD_BULLISH', 'LOW_DIRECTIONAL_RANGE'),
                ('HIGH_VOLATILE_STRONG_BEARISH', 'HIGH_DIRECTIONAL_TRENDING'),
                ('HIGH_VOLATILE_MILD_BEARISH', 'HIGH_DIRECTIONAL_RANGE'),
                ('NORMAL_VOLATILE_STRONG_BEARISH', 'MODERATE_DIRECTIONAL_TRENDING'),
                ('NORMAL_VOLATILE_MILD_BEARISH', 'MODERATE_DIRECTIONAL_RANGE'),
                ('LOW_VOLATILE_STRONG_BEARISH', 'LOW_DIRECTIONAL_TRENDING'),
                ('LOW_VOLATILE_MILD_BEARISH', 'LOW_DIRECTIONAL_RANGE'),
                ('HIGH_VOLATILE_NEUTRAL', 'HIGH_NONDIRECTIONAL_RANGE'),
                ('NORMAL_VOLATILE_NEUTRAL', 'MODERATE_NONDIRECTIONAL_RANGE'),
                ('LOW_VOLATILE_NEUTRAL', 'LOW_NONDIRECTIONAL_RANGE'),
                ('HIGH_VOLATILE_SIDEWAYS', 'HIGH_NONDIRECTIONAL_TRENDING'),
                ('NORMAL_VOLATILE_SIDEWAYS', 'MODERATE_NONDIRECTIONAL_TRENDING'),
                ('LOW_VOLATILE_SIDEWAYS', 'LOW_NONDIRECTIONAL_TRENDING'),
            ]
            
            correct_mappings = 0
            total_mappings = len(mapping_test_cases)
            mapping_results = []
            
            for regime_18, expected_12 in mapping_test_cases:
                mapped_12 = detector.map_18_to_12_regime(regime_18)
                is_correct = mapped_12 == expected_12
                
                if is_correct:
                    correct_mappings += 1
                
                mapping_results.append({
                    'regime_18': regime_18,
                    'expected_12': expected_12,
                    'mapped_12': mapped_12,
                    'correct': is_correct
                })
                
                status = "✅" if is_correct else "❌"
                logger.info(f"{status} {regime_18} → {mapped_12} (expected: {expected_12})")
            
            mapping_accuracy = correct_mappings / total_mappings
            
            # Require >95% mapping accuracy
            assert mapping_accuracy >= 0.95, f"Mapping accuracy {mapping_accuracy:.1%} below 95% threshold"
            
            self.test_results['18_to_12_mapping'] = {
                'status': 'PASSED',
                'accuracy': mapping_accuracy,
                'correct_mappings': correct_mappings,
                'total_mappings': total_mappings,
                'mapping_results': mapping_results
            }
            
            logger.info(f"✅ 18→12 mapping accuracy test PASSED: {mapping_accuracy:.1%} ({correct_mappings}/{total_mappings})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 18→12 mapping accuracy test FAILED: {e}")
            self.test_results['18_to_12_mapping'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_excel_configuration_integration(self) -> bool:
        """Test Excel configuration integration for 12-regime system"""
        try:
            logger.info("📊 Testing Excel configuration integration...")
            
            from actual_system_excel_manager import ActualSystemExcelManager
            
            excel_manager = ActualSystemExcelManager()
            
            # Test 12-regime configuration generation
            regime_config = excel_manager._generate_regime_formation_config("12_REGIME")
            complexity_config = excel_manager._generate_regime_complexity_config()
            
            # Validate regime configuration
            assert len(regime_config) >= 13, f"Expected ≥13 regime config entries, got {len(regime_config)}"
            
            # Check for 12-regime entries
            regime_names = [entry[0] for entry in regime_config if len(entry) > 0 and entry[0] != 'REGIME_COMPLEXITY']
            expected_12_regimes = [
                'LOW_DIRECTIONAL_TRENDING', 'LOW_DIRECTIONAL_RANGE',
                'LOW_NONDIRECTIONAL_TRENDING', 'LOW_NONDIRECTIONAL_RANGE',
                'MODERATE_DIRECTIONAL_TRENDING', 'MODERATE_DIRECTIONAL_RANGE',
                'MODERATE_NONDIRECTIONAL_TRENDING', 'MODERATE_NONDIRECTIONAL_RANGE',
                'HIGH_DIRECTIONAL_TRENDING', 'HIGH_DIRECTIONAL_RANGE',
                'HIGH_NONDIRECTIONAL_TRENDING', 'HIGH_NONDIRECTIONAL_RANGE'
            ]
            
            found_regimes = 0
            for expected_regime in expected_12_regimes:
                if expected_regime in regime_names:
                    found_regimes += 1
                else:
                    logger.warning(f"Missing expected regime: {expected_regime}")
            
            assert found_regimes >= 10, f"Found only {found_regimes}/12 expected regimes"
            
            # Validate complexity configuration
            complexity_params = [entry[0] for entry in complexity_config]
            required_params = ['REGIME_COMPLEXITY', 'TRIPLE_STRADDLE_WEIGHT', 'REGIME_MAPPING_12']
            
            for param in required_params:
                assert param in complexity_params, f"Missing required parameter: {param}"
            
            # Test Excel template generation
            template_path = "test_12_regime_integration.xlsx"
            
            try:
                # Update Excel structure for 12-regime
                excel_manager.excel_structure['RegimeFormationConfig']['data'] = regime_config
                excel_manager.excel_structure['RegimeComplexityConfig']['data'] = complexity_config
                
                # Generate template
                generated_path = excel_manager.generate_excel_template(template_path)
                assert Path(generated_path).exists(), f"Template not created: {generated_path}"
                
                # Load and validate
                success = excel_manager.load_configuration(template_path)
                assert success, "Failed to load generated template"
                
                # Validate loaded configuration
                loaded_regime_config = excel_manager.get_regime_formation_configuration()
                assert len(loaded_regime_config) > 0, "No regime configuration loaded"
                
                logger.info(f"✅ Excel template generated and validated: {template_path}")
                
            finally:
                # Cleanup
                if Path(template_path).exists():
                    Path(template_path).unlink()
            
            self.test_results['excel_configuration'] = {
                'status': 'PASSED',
                'regime_config_entries': len(regime_config),
                'found_regimes': found_regimes,
                'expected_regimes': len(expected_12_regimes)
            }
            
            logger.info("✅ Excel configuration integration test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Excel configuration integration test FAILED: {e}")
            self.test_results['excel_configuration'] = {'status': 'FAILED', 'error': str(e)}
            return False

    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks for 12-regime system"""
        try:
            logger.info("⚡ Testing performance benchmarks...")

            from enhanced_12_regime_detector import Enhanced12RegimeDetector
            detector = Enhanced12RegimeDetector()

            # Performance test data
            test_data = {
                'iv_percentile': 0.5, 'atr_normalized': 0.4, 'gamma_exposure': 0.3,
                'ema_alignment': 0.6, 'price_momentum': 0.5, 'volume_confirmation': 0.4,
                'strike_correlation': 0.7, 'vwap_deviation': 0.6, 'pivot_analysis': 0.5
            }

            # Test processing time over multiple iterations
            processing_times = []
            num_iterations = 100

            for i in range(num_iterations):
                start_time = time.time()
                result = detector.classify_12_regime(test_data.copy())
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Validate each result
                assert hasattr(result, 'regime_id'), f"Iteration {i}: Missing regime_id"
                assert hasattr(result, 'confidence'), f"Iteration {i}: Missing confidence"

            # Calculate performance metrics
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)

            # Performance requirements
            assert avg_processing_time < 1.0, f"Average processing time {avg_processing_time:.3f}s exceeds 1s"
            assert max_processing_time < 3.0, f"Max processing time {max_processing_time:.3f}s exceeds 3s"
            assert min_processing_time > 0.0, f"Min processing time {min_processing_time:.3f}s invalid"

            # Test mapping performance
            mapping_times = []
            test_regimes = ['HIGH_VOLATILE_STRONG_BULLISH', 'LOW_VOLATILE_NEUTRAL', 'NORMAL_VOLATILE_SIDEWAYS']

            for regime_18 in test_regimes:
                start_time = time.time()
                mapped_12 = detector.map_18_to_12_regime(regime_18)
                mapping_time = time.time() - start_time
                mapping_times.append(mapping_time)

                assert mapped_12 is not None, f"Mapping failed for {regime_18}"

            avg_mapping_time = sum(mapping_times) / len(mapping_times)
            assert avg_mapping_time < 0.1, f"Average mapping time {avg_mapping_time:.3f}s exceeds 0.1s"

            self.performance_metrics = {
                'classification': {
                    'avg_time': avg_processing_time,
                    'max_time': max_processing_time,
                    'min_time': min_processing_time,
                    'iterations': num_iterations
                },
                'mapping': {
                    'avg_time': avg_mapping_time,
                    'test_cases': len(test_regimes)
                }
            }

            self.test_results['performance_benchmarks'] = {
                'status': 'PASSED',
                'metrics': self.performance_metrics
            }

            logger.info(f"✅ Performance benchmarks PASSED:")
            logger.info(f"   Classification: avg={avg_processing_time:.3f}s, max={max_processing_time:.3f}s")
            logger.info(f"   Mapping: avg={avg_mapping_time:.3f}s")
            return True

        except Exception as e:
            logger.error(f"❌ Performance benchmarks test FAILED: {e}")
            self.test_results['performance_benchmarks'] = {'status': 'FAILED', 'error': str(e)}
            return False

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report"""
        try:
            end_time = datetime.now()
            total_duration = (end_time - self.start_time).total_seconds()

            # Count test results
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
            failed_tests = total_tests - passed_tests
            success_rate = passed_tests / total_tests if total_tests > 0 else 0

            # Generate report
            report = f"""
{'='*80}
12-REGIME SYSTEM COMPREHENSIVE INTEGRATION TEST REPORT
{'='*80}

Test Execution Summary:
- Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- Total Duration: {total_duration:.2f} seconds
- Total Tests: {total_tests}
- Passed Tests: {passed_tests}
- Failed Tests: {failed_tests}
- Success Rate: {success_rate:.1%}

{'='*80}
DETAILED TEST RESULTS
{'='*80}

"""

            for test_name, result in self.test_results.items():
                status_icon = "✅" if result['status'] == 'PASSED' else "❌"
                report += f"{status_icon} {test_name.upper().replace('_', ' ')}: {result['status']}\n"

                if result['status'] == 'PASSED':
                    if test_name == '12_regime_detector':
                        report += f"   - Scenarios Tested: {result['total_scenarios']}\n"
                        report += f"   - All Classifications Successful\n"
                    elif test_name == '18_to_12_mapping':
                        report += f"   - Mapping Accuracy: {result['accuracy']:.1%}\n"
                        report += f"   - Correct Mappings: {result['correct_mappings']}/{result['total_mappings']}\n"
                    elif test_name == 'excel_configuration':
                        report += f"   - Regime Config Entries: {result['regime_config_entries']}\n"
                        report += f"   - Found Regimes: {result['found_regimes']}/{result['expected_regimes']}\n"
                    elif test_name == 'performance_benchmarks':
                        metrics = result['metrics']
                        report += f"   - Avg Classification Time: {metrics['classification']['avg_time']:.3f}s\n"
                        report += f"   - Max Classification Time: {metrics['classification']['max_time']:.3f}s\n"
                        report += f"   - Avg Mapping Time: {metrics['mapping']['avg_time']:.3f}s\n"
                else:
                    report += f"   - Error: {result.get('error', 'Unknown error')}\n"

                report += "\n"

            report += f"""
{'='*80}
OVERALL ASSESSMENT
{'='*80}

"""

            if success_rate >= 1.0:
                report += "🎉 EXCELLENT: All tests passed! 12-regime system is fully operational.\n"
                assessment = "PRODUCTION_READY"
            elif success_rate >= 0.8:
                report += "✅ GOOD: Most tests passed. 12-regime system is mostly operational.\n"
                assessment = "MOSTLY_READY"
            elif success_rate >= 0.6:
                report += "⚠️  FAIR: Some tests failed. 12-regime system needs improvements.\n"
                assessment = "NEEDS_IMPROVEMENT"
            else:
                report += "❌ POOR: Many tests failed. 12-regime system needs significant fixes.\n"
                assessment = "NEEDS_MAJOR_FIXES"

            report += f"\nSystem Status: {assessment}\n"
            report += f"Ready for Phase 2: {'YES' if success_rate >= 0.8 else 'NO'}\n"

            report += f"\n{'='*80}\n"

            # Save report to file
            report_path = f"12_regime_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report)

            logger.info(f"📄 Comprehensive report saved: {report_path}")

            return report

        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return f"Error generating report: {e}"

    def run_comprehensive_integration_tests(self) -> bool:
        """Run all comprehensive integration tests"""
        logger.info("🚀 Starting Comprehensive 12-Regime Integration Tests")
        logger.info("="*80)

        # Define test sequence
        tests = [
            ("12-Regime Detector Integration", self.test_12_regime_detector_integration),
            ("18→12 Mapping Accuracy", self.test_18_to_12_mapping_accuracy),
            ("Excel Configuration Integration", self.test_excel_configuration_integration),
            ("Performance Benchmarks", self.test_performance_benchmarks),
        ]

        # Execute tests
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} ---")
            try:
                test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} encountered unexpected error: {e}")
                self.test_results[test_name.lower().replace(' ', '_').replace('→', '_to_')] = {
                    'status': 'FAILED',
                    'error': f"Unexpected error: {e}"
                }

        # Generate and display report
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*80)

        report = self.generate_comprehensive_report()
        print(report)

        # Determine overall success
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        return success_rate >= 0.8  # 80% success rate required

def main():
    """Main execution function"""
    test_suite = Comprehensive12RegimeIntegrationTest()
    success = test_suite.run_comprehensive_integration_tests()

    if success:
        print("\n🎉 12-REGIME SYSTEM INTEGRATION TESTS PASSED")
        print("✅ System is ready for Phase 2 implementation")
        return 0
    else:
        print("\n❌ 12-REGIME SYSTEM INTEGRATION TESTS FAILED")
        print("🔧 System needs fixes before proceeding to Phase 2")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
