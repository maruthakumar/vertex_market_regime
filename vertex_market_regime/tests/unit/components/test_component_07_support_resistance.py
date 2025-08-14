"""
Unit tests for Component 07: Support/Resistance Feature Engineering
Tests with actual production data schema and parquet files
"""

import pytest
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time

from vertex_market_regime.src.components.component_07_support_resistance import (
    Component07Analyzer,
    SupportResistanceFeatureEngine,
    SupportResistanceFeatures,
    StraddleLevelDetector,
    UnderlyingLevelDetector,
    ConfluenceAnalyzer,
    SupportResistanceWeightLearner
)


# Production data schema columns
PRODUCTION_SCHEMA_COLUMNS = [
    "trade_date", "trade_time", "expiry_date", "index_name", "spot", "atm_strike",
    "strike", "dte", "expiry_bucket", "zone_id", "zone_name", "call_strike_type",
    "put_strike_type", "ce_symbol", "ce_open", "ce_high", "ce_low", "ce_close",
    "ce_volume", "ce_oi", "ce_coi", "ce_iv", "ce_delta", "ce_gamma", "ce_theta",
    "ce_vega", "ce_rho", "pe_symbol", "pe_open", "pe_high", "pe_low", "pe_close",
    "pe_volume", "pe_oi", "pe_coi", "pe_iv", "pe_delta", "pe_gamma", "pe_theta",
    "pe_vega", "pe_rho", "future_open", "future_high", "future_low", "future_close",
    "future_volume", "future_oi", "future_coi"
]


@pytest.fixture
def production_config():
    """Configuration for Component 07 with production settings"""
    return {
        "processing_budget_ms": 150,
        "memory_budget_mb": 220,
        "max_levels": 10,
        "proximity_threshold": 0.002,
        "min_touches": 2,
        "lookback_periods": 252,
        "enable_monitoring": True,
        "learning_rate": 0.1,
        "min_samples": 50
    }


@pytest.fixture
def sample_production_data():
    """Create sample data matching production schema"""
    # Create realistic market data
    np.random.seed(42)
    n_rows = 100
    
    base_price = 21500
    prices = base_price + np.cumsum(np.random.randn(n_rows) * 10)
    
    data = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
        "trade_time": pd.date_range("09:15", periods=n_rows, freq="5min").strftime("%H:%M"),
        "expiry_date": "2024-01-25",
        "index_name": "nifty",
        "spot": prices,
        "atm_strike": np.round(prices / 50) * 50,  # Round to nearest 50
        "strike": np.round(prices / 50) * 50,
        "dte": 25,
        "expiry_bucket": "NM",
        "zone_id": np.random.randint(1, 6, n_rows),
        "zone_name": np.random.choice(["OPEN", "MID_MORN", "LUNCH", "AFTERNOON", "CLOSE"], n_rows),
        "call_strike_type": "ATM",
        "put_strike_type": "ATM",
        
        # Option prices and greeks
        "ce_open": 250 + np.random.randn(n_rows) * 10,
        "ce_high": 260 + np.random.randn(n_rows) * 10,
        "ce_low": 240 + np.random.randn(n_rows) * 10,
        "ce_close": 250 + np.random.randn(n_rows) * 10,
        "ce_volume": np.random.randint(100, 10000, n_rows),
        "ce_oi": np.random.randint(10000, 100000, n_rows),
        "ce_coi": np.random.randn(n_rows) * 1000,
        "ce_iv": 15 + np.random.randn(n_rows) * 2,
        "ce_delta": 0.5 + np.random.randn(n_rows) * 0.1,
        "ce_gamma": 0.001 + np.random.randn(n_rows) * 0.0001,
        "ce_theta": -5 + np.random.randn(n_rows) * 1,
        "ce_vega": 20 + np.random.randn(n_rows) * 2,
        "ce_rho": 10 + np.random.randn(n_rows) * 1,
        
        "pe_open": 250 + np.random.randn(n_rows) * 10,
        "pe_high": 260 + np.random.randn(n_rows) * 10,
        "pe_low": 240 + np.random.randn(n_rows) * 10,
        "pe_close": 250 + np.random.randn(n_rows) * 10,
        "pe_volume": np.random.randint(100, 10000, n_rows),
        "pe_oi": np.random.randint(10000, 100000, n_rows),
        "pe_coi": np.random.randn(n_rows) * 1000,
        "pe_iv": 15 + np.random.randn(n_rows) * 2,
        "pe_delta": -0.5 + np.random.randn(n_rows) * 0.1,
        "pe_gamma": 0.001 + np.random.randn(n_rows) * 0.0001,
        "pe_theta": -5 + np.random.randn(n_rows) * 1,
        "pe_vega": 20 + np.random.randn(n_rows) * 2,
        "pe_rho": -10 + np.random.randn(n_rows) * 1,
        
        # Future prices
        "future_open": prices + np.random.randn(n_rows) * 5,
        "future_high": prices + 10 + np.random.randn(n_rows) * 5,
        "future_low": prices - 10 + np.random.randn(n_rows) * 5,
        "future_close": prices + np.random.randn(n_rows) * 5,
        "future_volume": np.random.randint(10000, 100000, n_rows),
        "future_oi": np.random.randint(100000, 1000000, n_rows),
        "future_coi": np.random.randn(n_rows) * 10000
    })
    
    return data


@pytest.fixture
def market_data_from_production(sample_production_data):
    """Convert production data to market data format"""
    return pd.DataFrame({
        "open": sample_production_data["future_open"],
        "high": sample_production_data["future_high"],
        "low": sample_production_data["future_low"],
        "close": sample_production_data["future_close"],
        "volume": sample_production_data["future_volume"]
    })


@pytest.fixture
def straddle_data_from_production(sample_production_data):
    """Create straddle data from production data"""
    return pd.DataFrame({
        "atm_straddle": sample_production_data["ce_close"] + sample_production_data["pe_close"],
        "itm1_straddle": (sample_production_data["ce_close"] * 1.1 + 
                         sample_production_data["pe_close"] * 0.9),
        "otm1_straddle": (sample_production_data["ce_close"] * 0.9 + 
                         sample_production_data["pe_close"] * 1.1),
        "volume": sample_production_data["ce_volume"] + sample_production_data["pe_volume"]
    })


@pytest.fixture
def component_1_data_mock(sample_production_data):
    """Mock Component 1 data from production schema"""
    return {
        "atm_straddle_prices": (sample_production_data["ce_close"] + 
                                sample_production_data["pe_close"]).tolist(),
        "itm1_straddle_prices": (sample_production_data["ce_close"] * 1.1 + 
                                sample_production_data["pe_close"] * 0.9).tolist(),
        "otm1_straddle_prices": (sample_production_data["ce_close"] * 0.9 + 
                                sample_production_data["pe_close"] * 1.1).tolist(),
        "volume": (sample_production_data["ce_volume"] + 
                  sample_production_data["pe_volume"]).tolist(),
        "atm": {"prices": sample_production_data["ce_close"].tolist()},
        "itm1": {"prices": (sample_production_data["ce_close"] * 1.1).tolist()},
        "otm1": {"prices": (sample_production_data["ce_close"] * 0.9).tolist()}
    }


@pytest.fixture
def component_3_data_mock(sample_production_data):
    """Mock Component 3 data from production schema"""
    return {
        "cumulative_ce": (sample_production_data["ce_close"].cumsum() / 7).tolist(),
        "cumulative_pe": (sample_production_data["pe_close"].cumsum() / 7).tolist(),
        "cumulative_straddle": ((sample_production_data["ce_close"] + 
                                sample_production_data["pe_close"]).cumsum() / 7).tolist(),
        "cumulative_ce_levels": [
            {"price": p, "strength": 0.8, "timestamp": i}
            for i, p in enumerate(sample_production_data["atm_strike"].unique()[:5])
        ],
        "cumulative_pe_levels": [
            {"price": p - 100, "strength": 0.8, "timestamp": i}
            for i, p in enumerate(sample_production_data["atm_strike"].unique()[:5])
        ]
    }


class TestComponent07Analyzer:
    """Test Component 07 Analyzer with production data"""
    
    def test_initialization(self, production_config):
        """Test analyzer initialization"""
        analyzer = Component07Analyzer(production_config)
        
        assert analyzer.component_id == "component_07_support_resistance"
        assert analyzer.processing_budget_ms == 150
        assert analyzer.memory_budget_mb == 220
        assert analyzer.monitoring_enabled is True
    
    @pytest.mark.asyncio
    async def test_analyze_with_production_data(
        self, 
        production_config,
        market_data_from_production,
        component_1_data_mock,
        component_3_data_mock
    ):
        """Test analysis with production-like data"""
        analyzer = Component07Analyzer(production_config)
        
        result = await analyzer.analyze(
            market_data=market_data_from_production,
            component_1_data=component_1_data_mock,
            component_3_data=component_3_data_mock,
            dte=25
        )
        
        # Verify 120 features extracted (base class returns 120 by default now)
        assert "features" in result
        assert len(result["features"]) == 120
        
        # Verify processing metrics
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] < 200  # Should be close to budget
        
        # Verify confluence analysis
        assert "confluence_analysis" in result
        assert "strongest_levels" in result
        assert len(result["strongest_levels"]) <= 10
    
    def test_comprehensive_analysis(
        self,
        production_config,
        market_data_from_production,
        straddle_data_from_production,
        component_1_data_mock,
        component_3_data_mock
    ):
        """Test comprehensive S&R analysis"""
        analyzer = Component07Analyzer(production_config)
        
        result = analyzer.analyze_comprehensive_support_resistance(
            market_data=market_data_from_production,
            straddle_data=straddle_data_from_production,
            component_1_analysis=component_1_data_mock,
            component_3_analysis=component_3_data_mock,
            dte=25
        )
        
        # Check all feature categories
        assert "features_72" in result
        assert len(result["features_72"]) == 72
        assert "features_120" in result
        assert len(result["features_120"]) == 120
        
        assert "feature_breakdown" in result
        breakdown = result["feature_breakdown"]
        
        # Verify feature components
        assert len(breakdown["level_prices"]) == 10
        assert len(breakdown["level_strengths"]) == 10
        assert len(breakdown["level_ages"]) == 10
        assert len(breakdown["level_validation_counts"]) == 6
        assert len(breakdown["level_distances"]) == 10
        assert len(breakdown["level_types"]) == 6
        assert len(breakdown["method_performance_scores"]) == 10
        assert len(breakdown["weight_adaptations"]) == 10
        
        # Check confluence metrics
        assert "confluence_metrics" in result
        assert "adaptive_weights" in result
        assert "performance_metrics" in result


class TestFeatureEngine:
    """Test Feature Engineering with production data"""
    
    def test_feature_extraction(
        self,
        production_config,
        market_data_from_production,
        straddle_data_from_production,
        component_1_data_mock,
        component_3_data_mock
    ):
        """Test 72-feature extraction"""
        engine = SupportResistanceFeatureEngine(production_config)
        
        features = engine.extract_features(
            market_data=market_data_from_production,
            straddle_data=straddle_data_from_production,
            component_1_data=component_1_data_mock,
            component_3_data=component_3_data_mock
        )
        
        # Check feature object
        assert isinstance(features, SupportResistanceFeatures)
        
        # Convert to vector and verify size
        feature_vector = features.to_feature_vector()
        assert len(feature_vector) == 72
        assert isinstance(feature_vector, np.ndarray)
    
    def test_level_detection_methods(
        self,
        production_config,
        market_data_from_production
    ):
        """Test various level detection methods"""
        engine = SupportResistanceFeatureEngine(production_config)
        
        # Test pivot calculation
        levels = engine._calculate_pivot_points(market_data_from_production, "daily")
        assert len(levels) > 0
        assert all("price" in level for level in levels)
        
        # Test volume profile
        levels = engine._calculate_volume_profile_levels(market_data_from_production)
        assert isinstance(levels, list)
        
        # Test MA levels
        levels = engine._calculate_ma_levels(market_data_from_production)
        assert isinstance(levels, list)
        
        # Test psychological levels
        levels = engine._calculate_psychological_levels(market_data_from_production)
        assert len(levels) > 0


class TestStraddleLevelDetector:
    """Test straddle-based level detection"""
    
    def test_component_1_level_detection(
        self,
        production_config,
        straddle_data_from_production,
        component_1_data_mock
    ):
        """Test Component 1 straddle level detection"""
        detector = StraddleLevelDetector(production_config)
        
        levels = detector.detect_component_1_levels(
            straddle_data=straddle_data_from_production,
            component_1_analysis=component_1_data_mock
        )
        
        assert isinstance(levels, list)
        assert all("price" in level for level in levels)
        assert all("source" in level for level in levels)
        assert all("method" in level for level in levels)
    
    def test_component_3_level_detection(
        self,
        production_config,
        component_3_data_mock
    ):
        """Test Component 3 cumulative level detection"""
        detector = StraddleLevelDetector(production_config)
        
        # Create cumulative data
        cumulative_data = pd.DataFrame({
            "cumulative_ce": component_3_data_mock["cumulative_ce"],
            "cumulative_pe": component_3_data_mock["cumulative_pe"],
            "cumulative_straddle": component_3_data_mock["cumulative_straddle"]
        })
        
        levels = detector.detect_component_3_levels(
            cumulative_data=cumulative_data,
            component_3_analysis=component_3_data_mock
        )
        
        assert isinstance(levels, list)
        # Check for CE and PE levels
        ce_levels = [l for l in levels if l.get("source") == "component_3_ce"]
        pe_levels = [l for l in levels if l.get("source") == "component_3_pe"]
        
        assert len(ce_levels) > 0 or len(pe_levels) > 0


class TestUnderlyingLevelDetector:
    """Test underlying price level detection"""
    
    def test_daily_level_detection(
        self,
        production_config,
        market_data_from_production
    ):
        """Test daily timeframe level detection"""
        detector = UnderlyingLevelDetector(production_config)
        
        levels = detector.detect_daily_levels(market_data_from_production)
        
        assert isinstance(levels, list)
        assert len(levels) > 0
        
        # Check for different level types
        pivot_levels = [l for l in levels if "pivot" in l.get("method", "")]
        assert len(pivot_levels) > 0
    
    def test_psychological_levels(
        self,
        production_config,
        market_data_from_production
    ):
        """Test psychological level detection"""
        detector = UnderlyingLevelDetector(production_config)
        
        levels = detector.detect_psychological_levels(market_data_from_production)
        
        assert isinstance(levels, list)
        assert len(levels) > 0
        assert all(level["method"] == "psychological" for level in levels)


class TestConfluenceAnalyzer:
    """Test confluence analysis"""
    
    def test_confluence_measurement(
        self,
        production_config
    ):
        """Test straddle vs underlying confluence"""
        analyzer = ConfluenceAnalyzer(production_config)
        
        # Create test levels
        straddle_levels = [
            {"price": 21500, "source": "comp1", "type": "support", "strength": 0.8, "method": "component_1_straddle"},
            {"price": 21600, "source": "comp1", "type": "resistance", "strength": 0.7, "method": "component_1_straddle"}
        ]
        
        underlying_levels = [
            {"price": 21505, "source": "pivot", "type": "support", "strength": 0.9, "method": "daily_pivots"},
            {"price": 21595, "source": "ma", "type": "resistance", "strength": 0.6, "method": "moving_averages"}
        ]
        
        result = analyzer.measure_straddle_underlying_confluence(
            straddle_levels, underlying_levels
        )
        
        assert "confluence_pairs" in result
        assert "average_confluence" in result
        assert result["total_pairs"] >= 0
    
    def test_weighted_level_combination(
        self,
        production_config
    ):
        """Test weighted level combination"""
        analyzer = ConfluenceAnalyzer(production_config)
        
        levels = [
            {"price": 21500, "source": "s1", "type": "support", "strength": 0.8, "method": "component_1_straddle"},
            {"price": 21502, "source": "s2", "type": "support", "strength": 0.9, "method": "daily_pivots"},
            {"price": 21600, "source": "s3", "type": "resistance", "strength": 0.7, "method": "volume_profile"}
        ]
        
        combined = analyzer.combine_weighted_levels(levels)
        
        assert isinstance(combined, list)
        assert all("confluence_score" in level for level in combined)
        assert all("source_count" in level for level in combined)


class TestWeightLearner:
    """Test dynamic weight learning"""
    
    def test_performance_tracking(self, production_config):
        """Test performance tracking and weight updates"""
        learner = SupportResistanceWeightLearner(production_config)
        
        # Track performance for different methods
        for _ in range(60):  # More than min_samples
            learner.track_performance(
                {"method": "component_1_straddle", "price": 21500},
                outcome=True,
                dte=25
            )
            
            learner.track_performance(
                {"method": "daily_pivots", "price": 21600},
                outcome=False,
                dte=25
            )
        
        # Get adaptive weights
        weights = learner.get_adaptive_weights(dte=25)
        
        assert isinstance(weights, dict)
        assert "component_1_straddle" in weights
        assert "daily_pivots" in weights
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_dte_specific_learning(self, production_config):
        """Test DTE-specific weight optimization"""
        learner = SupportResistanceWeightLearner(production_config)
        
        # Track performance for different DTEs
        for dte in [5, 15, 45]:
            for _ in range(55):
                learner.track_performance(
                    {"method": "component_1_straddle", "price": 21500},
                    outcome=True,
                    dte=dte
                )
        
        # Get weights for different DTEs
        weekly_weights = learner.get_adaptive_weights(dte=5)
        monthly_weights = learner.get_adaptive_weights(dte=15)
        far_month_weights = learner.get_adaptive_weights(dte=45)
        
        # All should be valid weight dictionaries
        assert all(isinstance(w, dict) for w in [weekly_weights, monthly_weights, far_month_weights])


class TestPerformanceRequirements:
    """Test performance requirements"""
    
    @pytest.mark.performance
    def test_processing_time_constraint(
        self,
        production_config,
        market_data_from_production,
        component_1_data_mock,
        component_3_data_mock
    ):
        """Test that processing completes within 150ms budget"""
        analyzer = Component07Analyzer(production_config)
        
        start_time = time.time()
        
        result = analyzer.analyze_comprehensive_support_resistance(
            market_data=market_data_from_production,
            component_1_analysis=component_1_data_mock,
            component_3_analysis=component_3_data_mock,
            dte=25
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        assert processing_time < 200  # Allow some overhead
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] < 150
    
    @pytest.mark.memory
    def test_memory_constraint(
        self,
        production_config,
        market_data_from_production
    ):
        """Test memory usage stays within 220MB budget"""
        import psutil
        import os
        
        analyzer = Component07Analyzer(production_config)
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Run analysis
        analyzer.analyze_comprehensive_support_resistance(
            market_data=market_data_from_production,
            dte=25
        )
        
        # Check memory increase
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 220  # Should stay within budget


class TestProductionDataIntegration:
    """Test with actual parquet files if available"""
    
    @pytest.mark.skipif(
        not Path("/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed").exists(),
        reason="Production data not available"
    )
    def test_with_actual_parquet_data(self, production_config):
        """Test with actual parquet data files"""
        data_path = Path("/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed")
        
        # Find a sample parquet file
        parquet_files = list(data_path.glob("*/*.parquet"))
        
        if parquet_files:
            # Read first parquet file
            df = pd.read_parquet(parquet_files[0])
            
            # Convert to market data format
            market_data = pd.DataFrame({
                "open": df["future_open"] if "future_open" in df else df["spot"],
                "high": df["future_high"] if "future_high" in df else df["spot"] * 1.001,
                "low": df["future_low"] if "future_low" in df else df["spot"] * 0.999,
                "close": df["future_close"] if "future_close" in df else df["spot"],
                "volume": df["future_volume"] if "future_volume" in df else 10000
            })
            
            # Run analyzer
            analyzer = Component07Analyzer(production_config)
            result = analyzer.analyze_comprehensive_support_resistance(
                market_data=market_data,
                dte=int(df["dte"].iloc[0]) if "dte" in df else 25
            )
            
            # Verify results
            assert "features_72" in result
            assert len(result["features_72"]) == 72
            assert "strongest_levels" in result