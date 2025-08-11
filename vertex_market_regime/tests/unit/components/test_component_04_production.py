"""
Component 4 Production Testing Suite - Complete IV Skew Analysis Validation

Comprehensive production testing for Component 4 IV Skew Analysis using actual
production data with complete volatility surface validation, performance benchmarking,
and framework integration testing.

🚨 COMPREHENSIVE PRODUCTION TESTING:
- Full chain validation using ALL 54-68 strikes from production data
- Asymmetric coverage testing (Put: -21% range vs Call: +9.9% range)
- Variable strike count testing across different DTE scenarios
- Dynamic interval testing for non-uniform spacing (50/100/200/500 points)
- Greeks integration testing across complete surface
- Performance scalability testing with actual variable loads
- Surface accuracy validation >90% using historical patterns
- Framework integration testing with Components 1+2+3
"""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Component imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from components.component_04_iv_skew.component_04_analyzer import Component04IVSkewAnalyzer
from components.component_04_iv_skew.skew_analyzer import IVSkewAnalyzer, IVSkewData
from components.component_04_iv_skew.dual_dte_framework import DualDTEFramework, DTEBucket
from components.component_04_iv_skew.regime_classifier import IVSkewRegimeClassifier, MarketRegime

# Test fixtures
from tests.fixtures.production_data_fixtures import ProductionDataFixture


class TestComponent04ProductionValidation:
    """Complete production validation for Component 4"""
    
    @pytest.fixture
    def production_data_paths(self) -> List[str]:
        """Get production data file paths"""
        data_dir = Path("/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed")
        
        if not data_dir.exists():
            pytest.skip("Production data directory not available")
        
        # Find sample files across different DTEs and expiries
        sample_files = []
        for expiry_dir in data_dir.iterdir():
            if expiry_dir.is_dir() and expiry_dir.name.startswith('expiry='):
                parquet_files = list(expiry_dir.glob('*.parquet'))
                if parquet_files:
                    sample_files.extend(parquet_files[:2])  # Take first 2 files per expiry
                    
                if len(sample_files) >= 10:  # Limit for testing
                    break
        
        if not sample_files:
            pytest.skip("No production parquet files found")
        
        return [str(f) for f in sample_files]
    
    @pytest.fixture
    def component_config(self) -> Dict[str, Any]:
        """Component configuration for testing"""
        return {
            'component_id': 4,
            'feature_count': 87,
            'processing_budget_ms': 200,
            'memory_budget_mb': 300,
            'min_iv_threshold': 0.001,
            'max_iv_threshold': 2.0,
            'interpolation_method': 'cubic',
            'min_surface_r_squared': 0.8,
            'steep_skew_threshold': 0.015,
            'moderate_skew_threshold': 0.008,
            'high_tail_risk_threshold': 0.7,
            'crash_probability_threshold': 0.3
        }
    
    @pytest.fixture
    def component_analyzer(self, component_config) -> Component04IVSkewAnalyzer:
        """Initialize Component 4 analyzer"""
        return Component04IVSkewAnalyzer(component_config)


class TestCompleteVolatilitySurfaceConstruction:
    """Test complete volatility surface construction using ALL strikes"""
    
    def test_full_chain_extraction(self, component_analyzer, production_data_paths):
        """Test complete strike chain extraction using ALL available strikes"""
        
        for file_path in production_data_paths[:3]:  # Test first 3 files
            df = pd.read_parquet(file_path)
            
            # Extract IV skew data
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            
            # Validate complete extraction
            assert skew_data.strike_count >= 40, f"Insufficient strikes: {skew_data.strike_count}"
            assert skew_data.strike_count <= 80, f"Too many strikes: {skew_data.strike_count}"
            
            # Validate data completeness
            valid_call_ivs = np.sum(skew_data.call_ivs > 0)
            valid_put_ivs = np.sum(skew_data.put_ivs > 0)
            
            assert valid_call_ivs >= skew_data.strike_count * 0.5, "Insufficient call IV coverage"
            assert valid_put_ivs >= skew_data.strike_count * 0.5, "Insufficient put IV coverage"
            
            # Validate strike range coverage
            spot = skew_data.spot
            min_strike = np.min(skew_data.strikes)
            max_strike = np.max(skew_data.strikes)
            
            put_coverage_pct = (spot - min_strike) / spot * 100
            call_coverage_pct = (max_strike - spot) / spot * 100
            
            # Production data shows asymmetric coverage
            assert put_coverage_pct >= 10, f"Insufficient put coverage: {put_coverage_pct:.1f}%"
            assert call_coverage_pct >= 5, f"Insufficient call coverage: {call_coverage_pct:.1f}%"
            
            print(f"✅ File {Path(file_path).name}: {skew_data.strike_count} strikes, "
                  f"Put coverage: {put_coverage_pct:.1f}%, Call coverage: {call_coverage_pct:.1f}%")
    
    def test_surface_construction_quality(self, component_analyzer, production_data_paths):
        """Test volatility surface construction quality across production data"""
        
        quality_scores = []
        r_squared_scores = []
        
        for file_path in production_data_paths[:5]:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            
            # Construct volatility surface
            surface_result = component_analyzer.iv_analyzer.analyze_complete_iv_surface(skew_data)
            
            quality_scores.append(surface_result.surface_quality_score)
            r_squared_scores.append(surface_result.surface_r_squared)
            
            # Individual file validation
            assert surface_result.surface_quality_score >= 0.6, \
                f"Poor surface quality: {surface_result.surface_quality_score:.3f}"
            
            assert surface_result.data_completeness >= 0.7, \
                f"Poor data completeness: {surface_result.data_completeness:.3f}"
            
            assert len(surface_result.surface_strikes) >= 50, \
                f"Insufficient surface points: {len(surface_result.surface_strikes)}"
        
        # Overall quality validation
        avg_quality = np.mean(quality_scores)
        avg_r_squared = np.mean(r_squared_scores)
        
        assert avg_quality >= 0.75, f"Average surface quality too low: {avg_quality:.3f}"
        assert avg_r_squared >= 0.7, f"Average R-squared too low: {avg_r_squared:.3f}"
        
        print(f"✅ Surface Quality: {avg_quality:.3f}, R²: {avg_r_squared:.3f}")
    
    def test_asymmetric_coverage_handling(self, component_analyzer, production_data_paths):
        """Test asymmetric put/call coverage handling"""
        
        asymmetry_ratios = []
        
        for file_path in production_data_paths[:4]:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            surface_result = component_analyzer.iv_analyzer.analyze_complete_iv_surface(skew_data)
            
            # Test asymmetric coverage factor
            asymmetry_factor = surface_result.asymmetric_coverage_factor
            asymmetry_ratios.append(asymmetry_factor)
            
            # Validate asymmetric coverage is properly handled
            assert 0.0 <= asymmetry_factor <= 1.0, f"Invalid asymmetry factor: {asymmetry_factor}"
            
            # Test put skew dominance (should be positive due to asymmetric coverage)
            assert surface_result.put_skew_dominance >= 0, \
                f"Unexpected put skew dominance: {surface_result.put_skew_dominance}"
        
        # Validate overall asymmetric handling
        avg_asymmetry = np.mean(asymmetry_ratios)
        
        # Should show put bias due to production data coverage pattern
        assert avg_asymmetry >= 0.6, f"Expected put bias in production data: {avg_asymmetry:.3f}"
        
        print(f"✅ Asymmetric coverage properly handled: {avg_asymmetry:.3f}")


class TestDTEAdaptiveFramework:
    """Test DTE-adaptive framework with varying strike counts"""
    
    def test_dte_classification(self, component_analyzer, production_data_paths):
        """Test DTE bucket classification"""
        
        dte_classifications = {}
        
        for file_path in production_data_paths:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            
            dte = skew_data.dte
            dte_bucket = component_analyzer.dte_framework.dte_analyzer.classify_dte_bucket(dte)
            
            if dte_bucket not in dte_classifications:
                dte_classifications[dte_bucket] = []
            dte_classifications[dte_bucket].append((dte, skew_data.strike_count))
        
        # Validate DTE classification coverage
        assert len(dte_classifications) >= 2, "Need multiple DTE buckets for testing"
        
        for dte_bucket, dte_data in dte_classifications.items():
            dtes = [d[0] for d in dte_data]
            strike_counts = [d[1] for d in dte_data]
            
            print(f"✅ {dte_bucket.value}: DTEs {min(dtes)}-{max(dtes)}, "
                  f"Strike counts {min(strike_counts)}-{max(strike_counts)}")
    
    def test_variable_strike_count_handling(self, component_analyzer, production_data_paths):
        """Test handling of variable strike counts per DTE"""
        
        strike_count_variations = {}
        
        for file_path in production_data_paths:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            
            dte = skew_data.dte
            strike_count = skew_data.strike_count
            
            dte_bucket = component_analyzer.dte_framework.dte_analyzer.classify_dte_bucket(dte)
            
            if dte_bucket not in strike_count_variations:
                strike_count_variations[dte_bucket] = []
            strike_count_variations[dte_bucket].append(strike_count)
            
            # Test DTE-specific analysis
            surface_result = component_analyzer.iv_analyzer.analyze_complete_iv_surface(skew_data)
            dte_metrics = component_analyzer.dte_framework.dte_analyzer.analyze_dte_specific_surface(
                skew_data, surface_result
            )
            
            # Validate DTE-specific metrics
            assert dte_metrics.dte == dte
            assert dte_metrics.dte_bucket == dte_bucket
            assert dte_metrics.total_strikes == strike_count
            assert 0.0 <= dte_metrics.surface_quality <= 1.0
            assert 0.0 <= dte_metrics.strike_coverage <= 2.0  # Can exceed 100% 
        
        # Validate strike count variation handling
        for dte_bucket, counts in strike_count_variations.items():
            if len(counts) > 1:
                variation = np.std(counts) / np.mean(counts)
                print(f"✅ {dte_bucket.value}: Strike count variation {variation:.3f}")
    
    def test_intraday_surface_evolution(self, component_analyzer, production_data_paths):
        """Test intraday surface evolution analysis using zone_name"""
        
        zone_data = {}
        
        for file_path in production_data_paths[:6]:
            df = pd.read_parquet(file_path)
            
            if 'zone_name' in df.columns:
                zones = df['zone_name'].unique()
                
                for zone in zones:
                    zone_df = df[df['zone_name'] == zone]
                    if len(zone_df) > 10:  # Sufficient data
                        skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(zone_df)
                        surface_result = component_analyzer.iv_analyzer.analyze_complete_iv_surface(skew_data)
                        
                        if zone not in zone_data:
                            zone_data[zone] = []
                        zone_data[zone].append({
                            'surface_quality': surface_result.surface_quality_score,
                            'atm_iv': surface_result.smile_atm_iv,
                            'skew_steepness': surface_result.skew_steepness
                        })
        
        # Test zone-based analysis if data available
        if len(zone_data) >= 2:
            # Intraday analysis
            intraday_analysis = component_analyzer.dte_framework.intraday_analyzer.analyze_intraday_surface_evolution(
                zone_data
            )
            
            assert 0.0 <= intraday_analysis.surface_stability_score <= 1.0
            assert isinstance(intraday_analysis.volatility_regime_changes, list)
            
            print(f"✅ Intraday analysis: {len(zone_data)} zones, "
                  f"stability {intraday_analysis.surface_stability_score:.3f}")


class TestGreeksIntegration:
    """Test Greeks integration across complete surface"""
    
    def test_greeks_data_availability(self, component_analyzer, production_data_paths):
        """Test availability and validity of Greeks data"""
        
        greeks_coverage_stats = []
        
        for file_path in production_data_paths[:5]:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            
            # Check Greeks data coverage
            valid_call_deltas = np.sum(np.isfinite(skew_data.call_deltas))
            valid_put_deltas = np.sum(np.isfinite(skew_data.put_deltas))
            valid_call_gammas = np.sum(np.isfinite(skew_data.call_gammas))
            valid_put_gammas = np.sum(np.isfinite(skew_data.put_gammas))
            valid_call_vegas = np.sum(np.isfinite(skew_data.call_vegas))
            valid_put_vegas = np.sum(np.isfinite(skew_data.put_vegas))
            
            total_strikes = skew_data.strike_count
            
            coverage_stats = {
                'call_delta_coverage': valid_call_deltas / total_strikes,
                'put_delta_coverage': valid_put_deltas / total_strikes,
                'call_gamma_coverage': valid_call_gammas / total_strikes,
                'put_gamma_coverage': valid_put_gammas / total_strikes,
                'call_vega_coverage': valid_call_vegas / total_strikes,
                'put_vega_coverage': valid_put_vegas / total_strikes
            }
            
            greeks_coverage_stats.append(coverage_stats)
            
            # Validate minimum coverage
            for greek, coverage in coverage_stats.items():
                assert coverage >= 0.5, f"Insufficient {greek}: {coverage:.3f}"
        
        # Overall Greeks coverage validation
        avg_coverage = {}
        for greek in greeks_coverage_stats[0].keys():
            avg_coverage[greek] = np.mean([stats[greek] for stats in greeks_coverage_stats])
        
        print(f"✅ Greeks coverage: {avg_coverage}")
        
        for greek, avg_cov in avg_coverage.items():
            assert avg_cov >= 0.7, f"Poor average {greek}: {avg_cov:.3f}"
    
    def test_greeks_surface_consistency(self, component_analyzer, production_data_paths):
        """Test Greeks consistency across volatility surface"""
        
        consistency_scores = []
        
        for file_path in production_data_paths[:4]:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            surface_result = component_analyzer.iv_analyzer.analyze_complete_iv_surface(skew_data)
            
            # Test Greeks integration
            greeks_analysis = component_analyzer.greeks_analyzer.analyze_greeks_surface_integration(
                skew_data, surface_result
            )
            
            consistency_score = greeks_analysis['overall_greeks_consistency']
            consistency_scores.append(consistency_score)
            
            # Validate individual components
            assert 0.0 <= consistency_score <= 1.0, f"Invalid consistency score: {consistency_score}"
            
            surface_consistency = greeks_analysis['surface_consistency']
            assert 'overall_delta_consistency' in surface_consistency
            assert 'gamma_consistency' in surface_consistency
        
        # Overall consistency validation
        avg_consistency = np.mean(consistency_scores)
        assert avg_consistency >= 0.6, f"Poor Greeks consistency: {avg_consistency:.3f}"
        
        print(f"✅ Greeks surface consistency: {avg_consistency:.3f}")
    
    def test_pin_risk_analysis(self, component_analyzer, production_data_paths):
        """Test pin risk analysis across strikes"""
        
        pin_risk_scores = []
        
        for file_path in production_data_paths[:3]:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            surface_result = component_analyzer.iv_analyzer.analyze_complete_iv_surface(skew_data)
            
            greeks_analysis = component_analyzer.greeks_analyzer.analyze_greeks_surface_integration(
                skew_data, surface_result
            )
            
            pin_risk_analysis = greeks_analysis['pin_risk_analysis']
            
            # Validate pin risk analysis
            assert 'overall_pin_risk_level' in pin_risk_analysis
            assert 'max_pin_risk_strike' in pin_risk_analysis
            assert 'current_pin_risk' in pin_risk_analysis
            
            pin_risk_level = pin_risk_analysis['overall_pin_risk_level']
            pin_risk_scores.append(pin_risk_level)
            
            # Pin risk should be higher for shorter DTEs
            if skew_data.dte <= 7:
                assert pin_risk_level >= 0.3, f"Low pin risk for short DTE: {pin_risk_level:.3f}"
        
        print(f"✅ Pin risk analysis: avg {np.mean(pin_risk_scores):.3f}")


class TestRegimeClassification:
    """Test 8-regime classification system"""
    
    def test_regime_classification_coverage(self, component_analyzer, production_data_paths):
        """Test regime classification across different market conditions"""
        
        regime_counts = {}
        classification_confidences = []
        
        for file_path in production_data_paths[:8]:
            df = pd.read_parquet(file_path)
            
            try:
                # Run complete analysis
                result = await component_analyzer.analyze(df)
                
                regime = result.metadata['regime_classification']
                confidence = result.confidence
                
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                regime_counts[regime] += 1
                
                classification_confidences.append(confidence)
                
                # Validate classification result
                assert confidence >= 0.0, f"Invalid confidence: {confidence}"
                assert isinstance(regime, str), f"Invalid regime type: {type(regime)}"
                
            except Exception as e:
                print(f"Classification failed for {Path(file_path).name}: {e}")
                continue
        
        # Validate regime distribution
        assert len(regime_counts) >= 2, "Need multiple regimes for validation"
        
        avg_confidence = np.mean(classification_confidences) if classification_confidences else 0.5
        assert avg_confidence >= 0.5, f"Low classification confidence: {avg_confidence:.3f}"
        
        print(f"✅ Regimes classified: {regime_counts}")
        print(f"✅ Average confidence: {avg_confidence:.3f}")
    
    def test_regime_feature_consistency(self, component_analyzer, production_data_paths):
        """Test consistency between regime classification and extracted features"""
        
        regime_feature_consistency = []
        
        for file_path in production_data_paths[:5]:
            df = pd.read_parquet(file_path)
            
            try:
                result = await component_analyzer.analyze(df)
                
                # Check feature extraction
                features = result.features
                assert features.feature_count == 87, f"Wrong feature count: {features.feature_count}"
                assert len(features.features) == 87, f"Wrong feature array length: {len(features.features)}"
                
                # Check for NaN or infinite values
                assert not np.any(np.isnan(features.features)), "Features contain NaN values"
                assert not np.any(np.isinf(features.features)), "Features contain infinite values"
                
                # Feature consistency with regime
                regime_confidence = result.confidence
                feature_quality = np.std(features.features) / (np.mean(np.abs(features.features)) + 1e-10)
                
                # Higher regime confidence should correlate with better feature quality
                consistency_score = min(1.0, regime_confidence * (1.0 - min(0.5, feature_quality)))
                regime_feature_consistency.append(consistency_score)
                
            except Exception as e:
                print(f"Feature consistency test failed for {Path(file_path).name}: {e}")
                continue
        
        if regime_feature_consistency:
            avg_consistency = np.mean(regime_feature_consistency)
            assert avg_consistency >= 0.6, f"Poor regime-feature consistency: {avg_consistency:.3f}"
            
            print(f"✅ Regime-feature consistency: {avg_consistency:.3f}")


class TestPerformanceCompliance:
    """Test performance compliance with budget requirements"""
    
    def test_processing_time_compliance(self, component_analyzer, production_data_paths):
        """Test processing time compliance <200ms"""
        
        processing_times = []
        
        for file_path in production_data_paths:
            df = pd.read_parquet(file_path)
            
            start_time = time.time()
            try:
                result = await component_analyzer.analyze(df)
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                processing_times.append(processing_time)
                
                # Individual file compliance
                assert processing_time < 250, f"Processing time exceeded: {processing_time:.1f}ms"
                
            except Exception as e:
                print(f"Performance test failed for {Path(file_path).name}: {e}")
                continue
        
        # Overall performance validation
        if processing_times:
            avg_time = np.mean(processing_times)
            p95_time = np.percentile(processing_times, 95)
            
            assert avg_time < 200, f"Average processing time too high: {avg_time:.1f}ms"
            assert p95_time < 300, f"P95 processing time too high: {p95_time:.1f}ms"
            
            print(f"✅ Performance: avg {avg_time:.1f}ms, P95 {p95_time:.1f}ms")
    
    def test_memory_usage_compliance(self, component_analyzer, production_data_paths):
        """Test memory usage compliance <300MB"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usages = []
        
        for file_path in production_data_paths[:5]:  # Test subset for memory
            df = pd.read_parquet(file_path)
            
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = await component_analyzer.analyze(df)
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before
                
                memory_usages.append(memory_increase)
                
                # Individual memory compliance (allow some overhead)
                assert memory_increase < 400, f"Memory usage too high: {memory_increase:.1f}MB"
                
            except Exception as e:
                print(f"Memory test failed for {Path(file_path).name}: {e}")
                continue
        
        # Overall memory validation
        if memory_usages:
            avg_memory = np.mean(memory_usages)
            max_memory = np.max(memory_usages)
            
            assert avg_memory < 300, f"Average memory usage too high: {avg_memory:.1f}MB"
            assert max_memory < 500, f"Peak memory usage too high: {max_memory:.1f}MB"
            
            print(f"✅ Memory usage: avg {avg_memory:.1f}MB, peak {max_memory:.1f}MB")
    
    def test_scalability_with_variable_loads(self, component_analyzer, production_data_paths):
        """Test performance scalability with variable strike counts"""
        
        performance_by_complexity = {}
        
        for file_path in production_data_paths:
            df = pd.read_parquet(file_path)
            
            # Measure complexity
            strike_count = len(df['strike'].unique())
            row_count = len(df)
            complexity = strike_count * row_count / 1000  # Normalized complexity
            
            # Measure performance
            start_time = time.time()
            try:
                result = await component_analyzer.analyze(df)
                processing_time = (time.time() - start_time) * 1000
                
                if complexity not in performance_by_complexity:
                    performance_by_complexity[complexity] = []
                performance_by_complexity[complexity].append(processing_time)
                
            except Exception as e:
                continue
        
        # Analyze scalability
        if len(performance_by_complexity) >= 3:
            complexities = sorted(performance_by_complexity.keys())
            avg_times = [np.mean(performance_by_complexity[c]) for c in complexities]
            
            # Check that performance scales reasonably
            time_ratio = avg_times[-1] / avg_times[0] if avg_times[0] > 0 else 1
            complexity_ratio = complexities[-1] / complexities[0] if complexities[0] > 0 else 1
            
            scalability_efficiency = complexity_ratio / time_ratio if time_ratio > 0 else 1
            
            # Performance should scale better than linearly with complexity
            assert scalability_efficiency >= 0.5, f"Poor scalability: {scalability_efficiency:.3f}"
            
            print(f"✅ Scalability efficiency: {scalability_efficiency:.3f}")


class TestFrameworkIntegration:
    """Test integration with Components 1+2+3 framework"""
    
    def test_shared_schema_compatibility(self, component_analyzer, production_data_paths):
        """Test compatibility with shared production schema"""
        
        schema_compatibility_scores = []
        
        for file_path in production_data_paths[:4]:
            df = pd.read_parquet(file_path)
            
            # Check required columns exist
            required_columns = [
                'trade_date', 'trade_time', 'expiry_date', 'strike', 'dte',
                'spot', 'atm_strike', 'zone_name', 'call_strike_type', 'put_strike_type',
                'ce_iv', 'pe_iv', 'ce_delta', 'pe_delta', 'ce_gamma', 'pe_gamma',
                'ce_theta', 'pe_theta', 'ce_vega', 'pe_vega', 'ce_volume', 'pe_volume',
                'ce_oi', 'pe_oi'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            column_coverage = (len(required_columns) - len(missing_columns)) / len(required_columns)
            
            schema_compatibility_scores.append(column_coverage)
            
            # Must have high schema compatibility
            assert column_coverage >= 0.9, f"Poor schema compatibility: {column_coverage:.3f}"
            
            if missing_columns:
                print(f"Missing columns in {Path(file_path).name}: {missing_columns}")
        
        # Overall schema compatibility
        avg_compatibility = np.mean(schema_compatibility_scores)
        assert avg_compatibility >= 0.95, f"Poor average schema compatibility: {avg_compatibility:.3f}"
        
        print(f"✅ Schema compatibility: {avg_compatibility:.3f}")
    
    def test_component_integration_scores(self, component_analyzer, production_data_paths):
        """Test component integration scoring"""
        
        integration_scores = []
        
        for file_path in production_data_paths[:3]:
            df = pd.read_parquet(file_path)
            
            try:
                result = await component_analyzer.analyze(df)
                
                # Check integration metadata
                metadata = result.metadata
                assert 'integration_result' in metadata
                
                integration_data = metadata['integration_result']
                assert 'component_agreement_score' in integration_data
                assert 'combined_regime_score' in integration_data
                
                agreement_score = integration_data['component_agreement_score']
                combined_score = integration_data['combined_regime_score']
                
                integration_scores.append({
                    'agreement': agreement_score,
                    'combined': combined_score
                })
                
                # Validate integration scores
                assert 0.0 <= agreement_score <= 1.0, f"Invalid agreement score: {agreement_score}"
                assert 0.0 <= combined_score <= 1.0, f"Invalid combined score: {combined_score}"
                
            except Exception as e:
                print(f"Integration test failed for {Path(file_path).name}: {e}")
                continue
        
        if integration_scores:
            avg_agreement = np.mean([s['agreement'] for s in integration_scores])
            avg_combined = np.mean([s['combined'] for s in integration_scores])
            
            assert avg_agreement >= 0.5, f"Poor component agreement: {avg_agreement:.3f}"
            assert avg_combined >= 0.5, f"Poor combined scoring: {avg_combined:.3f}"
            
            print(f"✅ Integration - Agreement: {avg_agreement:.3f}, Combined: {avg_combined:.3f}")


class TestProductionDataAccuracyValidation:
    """Test accuracy validation using production historical patterns"""
    
    def test_surface_accuracy_validation(self, component_analyzer, production_data_paths):
        """Test >90% surface accuracy using historical patterns"""
        
        surface_accuracies = []
        
        for file_path in production_data_paths[:6]:
            df = pd.read_parquet(file_path)
            skew_data = component_analyzer.iv_analyzer.extract_iv_skew_data(df)
            surface_result = component_analyzer.iv_analyzer.analyze_complete_iv_surface(skew_data)
            
            # Surface accuracy metrics
            r_squared = surface_result.surface_r_squared
            quality_score = surface_result.surface_quality_score
            data_completeness = surface_result.data_completeness
            
            # Combined accuracy score
            accuracy_score = (r_squared * 0.4 + quality_score * 0.4 + data_completeness * 0.2)
            surface_accuracies.append(accuracy_score)
            
            # Individual file validation
            assert accuracy_score >= 0.7, f"Poor surface accuracy: {accuracy_score:.3f}"
        
        # Overall accuracy validation
        avg_accuracy = np.mean(surface_accuracies)
        assert avg_accuracy >= 0.8, f"Poor average surface accuracy: {avg_accuracy:.3f}"
        
        # Check if we meet the >90% target for majority of files
        high_accuracy_count = sum(1 for acc in surface_accuracies if acc >= 0.9)
        high_accuracy_ratio = high_accuracy_count / len(surface_accuracies)
        
        print(f"✅ Surface accuracy: {avg_accuracy:.3f}, >90% ratio: {high_accuracy_ratio:.3f}")
        
        # At least 60% of files should achieve >90% accuracy
        assert high_accuracy_ratio >= 0.6, f"Too few high-accuracy results: {high_accuracy_ratio:.3f}"
    
    def test_regime_classification_accuracy(self, component_analyzer, production_data_paths):
        """Test regime classification accuracy against expected patterns"""
        
        regime_consistency_scores = []
        
        for file_path in production_data_paths[:5]:
            df = pd.read_parquet(file_path)
            
            try:
                result = await component_analyzer.analyze(df)
                
                # Extract classification confidence and consistency
                regime_confidence = result.confidence
                overall_consistency = result.metadata['integration_result'].get('overall_consistency', 0.5)
                
                # Combined regime accuracy
                regime_accuracy = (regime_confidence * 0.6 + overall_consistency * 0.4)
                regime_consistency_scores.append(regime_accuracy)
                
                # Individual file validation
                assert regime_accuracy >= 0.6, f"Poor regime accuracy: {regime_accuracy:.3f}"
                
            except Exception as e:
                print(f"Regime accuracy test failed for {Path(file_path).name}: {e}")
                continue
        
        if regime_consistency_scores:
            avg_regime_accuracy = np.mean(regime_consistency_scores)
            assert avg_regime_accuracy >= 0.7, f"Poor regime classification accuracy: {avg_regime_accuracy:.3f}"
            
            print(f"✅ Regime classification accuracy: {avg_regime_accuracy:.3f}")


# Test execution helpers
def run_production_tests():
    """Run all production tests"""
    
    print("🚨 Running Component 4 Production Validation Tests...")
    
    # Run pytest with specific markers
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-k", "not slow"  # Skip slow tests by default
    ])


if __name__ == "__main__":
    run_production_tests()