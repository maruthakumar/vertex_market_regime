"""
Component 07: Support/Resistance Analyzer
Master integration and analysis engine for 72-feature S&R detection
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime

from ..base_component import BaseMarketRegimeComponent as BaseComponent
from .feature_engine import SupportResistanceFeatureEngine, SupportResistanceFeatures
from .straddle_level_detector import StraddleLevelDetector
from .underlying_level_detector import UnderlyingLevelDetector
from .confluence_analyzer import ConfluenceAnalyzer
from .weight_learning_engine import SupportResistanceWeightLearner
from .advanced_pattern_detector import AdvancedPatternDetector
from .momentum_level_detector import MomentumLevelDetector, MomentumLevelResult, MomentumLevelData

logger = logging.getLogger(__name__)


class Component07Analyzer(BaseComponent):
    """
    Main analyzer for Component 7: Support/Resistance Feature Engineering
    Generates 130 raw features for ML consumption without hard-coded classifications (Phase 2 Enhanced)
    Phase 1: 120 features (72 base + 48 advanced)
    Phase 2: +10 momentum-based features = 130 total
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Component 07 Analyzer
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Component configuration
        self.component_id = "component_07_support_resistance"
        self.processing_budget_ms = config.get("processing_budget_ms", 150)
        self.memory_budget_mb = config.get("memory_budget_mb", 220)
        
        # Initialize sub-components
        self.feature_engine = SupportResistanceFeatureEngine(config)
        self.straddle_detector = StraddleLevelDetector(config)
        self.underlying_detector = UnderlyingLevelDetector(config)
        self.confluence_analyzer = ConfluenceAnalyzer(config)
        self.weight_learner = SupportResistanceWeightLearner(config)
        self.advanced_detector = AdvancedPatternDetector(config)
        
        # Phase 2: Momentum-based level detector
        self.momentum_level_detector = MomentumLevelDetector(config)
        
        # Performance monitoring
        self.last_processing_time = 0
        self.last_memory_usage = 0
        self.level_cache = {}
        
        # Real-time monitoring
        self.monitoring_enabled = config.get("enable_monitoring", True)
        self.monitored_levels = []
        
        logger.info(f"Initialized Component07Analyzer Phase 2 with 130 features (120 + 10 momentum), {self.processing_budget_ms}ms budget")
    
    async def analyze(
        self,
        market_data: pd.DataFrame,
        component_1_data: Optional[Dict[str, Any]] = None,
        component_3_data: Optional[Dict[str, Any]] = None,
        dte: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze market data and extract 72 S&R features
        
        Args:
            market_data: DataFrame with OHLCV data
            component_1_data: Optional Component 1 triple straddle analysis
            component_3_data: Optional Component 3 cumulative ATM±7 analysis
            dte: Days to expiry for context-specific analysis
            
        Returns:
            Dictionary with analysis results and 72 features
        """
        start_time = time.time()
        
        try:
            # Prepare straddle and cumulative data if available
            straddle_data = self._prepare_straddle_data(component_1_data) if component_1_data else None
            cumulative_data = self._prepare_cumulative_data(component_3_data) if component_3_data else None
            
            # Get adaptive weights for current DTE
            adaptive_weights = self.weight_learner.get_adaptive_weights(dte)
            
            # Detect levels from all sources
            straddle_levels = await self._detect_straddle_levels(
                straddle_data, component_1_data
            )
            
            underlying_levels = await self._detect_underlying_levels(market_data)
            
            # Component-specific levels
            component_1_levels = []
            component_3_levels = []
            
            if component_1_data:
                component_1_levels = self.straddle_detector.detect_component_1_levels(
                    straddle_data, component_1_data
                )
            
            if component_3_data:
                component_3_levels = self.straddle_detector.detect_component_3_levels(
                    cumulative_data, component_3_data
                )
            
            # Analyze confluence
            confluence_analysis = self.confluence_analyzer.analyze_confluence(
                straddle_levels,
                underlying_levels,
                component_1_levels,
                component_3_levels
            )
            
            # Extract 72 features
            features = self.feature_engine.extract_features(
                market_data,
                straddle_data,
                cumulative_data,
                component_1_data,
                component_3_data
            )
            
            # Monitor levels in real-time
            if self.monitoring_enabled:
                self._update_monitored_levels(
                    confluence_analysis["combined_levels"][:10]
                )
            
            # Calculate processing metrics
            processing_time = (time.time() - start_time) * 1000
            memory_usage = self._get_memory_usage()
            
            # Prepare results
            results = {
                "features": features.to_feature_vector().tolist(),
                "feature_object": features,
                "confluence_analysis": confluence_analysis,
                "strongest_levels": self.confluence_analyzer.get_strongest_levels(
                    confluence_analysis["combined_levels"], 10
                ),
                "adaptive_weights": adaptive_weights,
                "processing_time_ms": processing_time,
                "memory_usage_mb": memory_usage,
                "timestamp": datetime.now().isoformat(),
                "dte": dte
            }
            
            # Update performance metrics
            self.last_processing_time = processing_time
            self.last_memory_usage = memory_usage
            
            # Check performance constraints
            if processing_time > self.processing_budget_ms:
                logger.warning(
                    f"Processing time {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms"
                )
            
            if memory_usage > self.memory_budget_mb:
                logger.warning(
                    f"Memory usage {memory_usage:.2f}MB exceeded budget {self.memory_budget_mb}MB"
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Component07 analysis: {str(e)}")
            raise
    
    def analyze_comprehensive_support_resistance(
        self,
        market_data: pd.DataFrame,
        straddle_data: Optional[pd.DataFrame] = None,
        cumulative_data: Optional[pd.DataFrame] = None,
        component_1_analysis: Optional[Dict[str, Any]] = None,
        component_3_analysis: Optional[Dict[str, Any]] = None,
        dte: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive S&R analysis with all detection methods
        Master method for complete 72-feature extraction
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        start_time = time.time()
        
        # Detect levels from all sources
        all_levels = []
        
        # Component 1 straddle-based levels
        if straddle_data is not None:
            comp1_levels = self.straddle_detector.detect_component_1_levels(
                straddle_data, component_1_analysis
            )
            all_levels.extend(comp1_levels)
        
        # Component 3 cumulative-based levels  
        if cumulative_data is not None:
            comp3_levels = self.straddle_detector.detect_component_3_levels(
                cumulative_data, component_3_analysis
            )
            all_levels.extend(comp3_levels)
        
        # Traditional underlying levels
        underlying_levels = self.underlying_detector.detect_all_levels(market_data)
        all_levels.extend(underlying_levels)
        
        # Validate historical levels
        validated_levels = self.underlying_detector.validate_historical_levels(
            all_levels, market_data
        )
        
        # Analyze confluence
        confluence_analysis = self.confluence_analyzer.analyze_confluence(
            comp1_levels if straddle_data is not None else [],
            underlying_levels,
            comp1_levels if straddle_data is not None else [],
            comp3_levels if cumulative_data is not None else []
        )
        
        # Extract 72 base features
        features = self.feature_engine.extract_features(
            market_data,
            straddle_data,
            cumulative_data,
            component_1_analysis,
            component_3_analysis
        )
        
        # Extract 48 advanced pattern features
        advanced_features_dict = self.advanced_detector.extract_all_advanced_features(
            market_data=market_data,
            options_data=market_data if "ce_oi" in market_data.columns else None,
            greeks_data=component_1_analysis.get("greeks") if component_1_analysis else None,
            component_data=component_1_analysis
        )
        
        # Create 48-feature vector from advanced patterns
        advanced_features = self.advanced_detector.create_feature_vector(advanced_features_dict)
        
        # Create 120-feature vector
        features_120 = features.to_extended_feature_vector(advanced_features)
        
        # Get adaptive weights
        adaptive_weights = self.weight_learner.get_adaptive_weights(dte)
        
        # Get performance metrics
        performance_metrics = self.weight_learner.get_performance_metrics()
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "features_72": features.to_feature_vector().tolist(),
            "features_120": features_120.tolist(),
            "feature_breakdown": {
                # Base 72 features
                "level_prices": features.level_prices.tolist(),
                "level_strengths": features.level_strengths.tolist(),
                "level_ages": features.level_ages.tolist(),
                "level_validation_counts": features.level_validation_counts.tolist(),
                "level_distances": features.level_distances.tolist(),
                "level_types": features.level_types.tolist(),
                "method_performance_scores": features.method_performance_scores.tolist(),
                "weight_adaptations": features.weight_adaptations.tolist(),
                "historical_accuracy_rates": features.historical_accuracy_rates.tolist(),
                "cross_validation_metrics": features.cross_validation_metrics.tolist(),
                # Advanced 48 features
                "advanced_patterns": advanced_features_dict
            },
            "strongest_levels": confluence_analysis["combined_levels"][:10],
            "confluence_metrics": {
                "straddle_underlying": confluence_analysis["straddle_underlying_confluence"],
                "component_validation": confluence_analysis["component_validation"],
                "timeframe_agreement": confluence_analysis["timeframe_agreement"],
                "statistics": confluence_analysis["confluence_statistics"]
            },
            "adaptive_weights": adaptive_weights,
            "performance_metrics": performance_metrics,
            "total_levels_detected": len(all_levels),
            "validated_levels": len(validated_levels),
            "processing_time_ms": processing_time,
            "dte": dte,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _detect_straddle_levels(
        self,
        straddle_data: Optional[pd.DataFrame],
        component_1_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from straddle data asynchronously
        """
        if straddle_data is None:
            return []
        
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.straddle_detector.detect_component_1_levels,
            straddle_data,
            component_1_data
        )
    
    async def _detect_underlying_levels(
        self,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from underlying price asynchronously
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.underlying_detector.detect_all_levels,
            market_data
        )
    
    def _prepare_straddle_data(
        self,
        component_1_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Prepare straddle data from Component 1 analysis
        """
        # Extract straddle prices from Component 1
        data = {}
        
        if "atm_straddle_prices" in component_1_data:
            data["atm_straddle"] = component_1_data["atm_straddle_prices"]
        
        if "itm1_straddle_prices" in component_1_data:
            data["itm1_straddle"] = component_1_data["itm1_straddle_prices"]
        
        if "otm1_straddle_prices" in component_1_data:
            data["otm1_straddle"] = component_1_data["otm1_straddle_prices"]
        
        if "volume" in component_1_data:
            data["volume"] = component_1_data["volume"]
        
        if data:
            return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    def _prepare_cumulative_data(
        self,
        component_3_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Prepare cumulative data from Component 3 analysis
        """
        data = {}
        
        if "cumulative_ce" in component_3_data:
            data["cumulative_ce"] = component_3_data["cumulative_ce"]
        
        if "cumulative_pe" in component_3_data:
            data["cumulative_pe"] = component_3_data["cumulative_pe"]
        
        if "cumulative_straddle" in component_3_data:
            data["cumulative_straddle"] = component_3_data["cumulative_straddle"]
        
        if data:
            return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    def _update_monitored_levels(
        self,
        levels: List[Dict[str, Any]]
    ) -> None:
        """
        Update real-time monitored levels
        """
        self.monitored_levels = levels
        
        # Log significant changes
        for level in levels[:5]:  # Top 5 levels
            logger.debug(
                f"Monitoring level at {level['price']:.2f} "
                f"(Type: {level.get('type', 'neutral')}, "
                f"Confluence: {level.get('confluence_score', 0):.2f})"
            )
    
    def update_level_performance(
        self,
        level: Dict[str, Any],
        outcome: bool,
        dte: Optional[int] = None
    ) -> None:
        """
        Update performance tracking for a level
        
        Args:
            level: Level that was tested
            outcome: Whether level held
            dte: Days to expiry
        """
        # Update feature engine performance
        self.feature_engine.update_performance(level, outcome)
        
        # Update weight learner
        self.weight_learner.track_performance(level, outcome, dte)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get component health metrics
        
        Returns:
            Dictionary with health metrics
        """
        return {
            "component_id": self.component_id,
            "status": "healthy" if self.last_processing_time < self.processing_budget_ms else "degraded",
            "last_processing_time_ms": self.last_processing_time,
            "last_memory_usage_mb": self.last_memory_usage,
            "processing_budget_ms": self.processing_budget_ms,
            "memory_budget_mb": self.memory_budget_mb,
            "monitored_levels_count": len(self.monitored_levels),
            "weight_learner_updates": self.weight_learner.update_counter,
            "cache_size": len(self.level_cache)
        }
    
    def optimize_weights(self) -> None:
        """
        Trigger weight optimization
        """
        self.weight_learner.update_weights()
        logger.info("Triggered weight optimization")
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def extract_features(self, market_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Extract features from market data (required by base class)
        
        Args:
            market_data: DataFrame with market data
            **kwargs: Additional parameters
            
        Returns:
            Feature vector as numpy array (120 features)
        """
        result = self.analyze_comprehensive_support_resistance(
            market_data=market_data,
            dte=kwargs.get("dte", None)
        )
        # Return 120 features by default
        return np.array(result.get("features_120", result["features_72"]))
    
    def update_weights(self, performance_data: List[Dict[str, Any]]) -> None:
        """
        Update component weights based on performance (required by base class)
        
        Args:
            performance_data: List of performance records
        """
        self.weight_learner.optimize_weights_batch(performance_data)