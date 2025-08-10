"""
Market Regime Engine - Main Orchestration Module

This module serves as the central orchestration point for the market regime strategy system.
It coordinates all components and provides a unified interface for regime analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# Import existing modules for compatibility
try:
    from ..archive_enhanced_modules_do_not_use.enhanced_market_regime_engine import EnhancedMarketRegimeEngine
    from ..archive_comprehensive_modules_do_not_use.comprehensive_triple_straddle_engine import StraddleAnalysisEngine
    LEGACY_MODULES_AVAILABLE = True
except ImportError:
    LEGACY_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RegimeAnalysisResult:
    """Result of regime analysis"""
    regime: str
    confidence: float
    signals: Dict[str, Any]
    indicators: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketRegimeEngine:
    """
    Market Regime Engine - Central orchestration for regime analysis
    
    This engine coordinates:
    1. Data loading and preprocessing
    2. Indicator calculations
    3. Regime detection and classification
    4. Signal generation
    5. Result aggregation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the market regime engine
        
        Args:
            config: Configuration dictionary from ConfigManager
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self._initialize_components()
        
        # Use legacy modules if available for backward compatibility
        if LEGACY_MODULES_AVAILABLE and config.get('use_legacy_modules', True):
            self._initialize_legacy_modules()
        
    def _initialize_components(self):
        """Initialize all core components"""
        # Initialize analyzer
        from .analyzer import MarketRegimeAnalyzer
        self.analyzer = MarketRegimeAnalyzer(self.config)
        
        # Initialize classifier
        from .regime_classifier import RegimeClassifier
        self.classifier = RegimeClassifier(self.config)
        
        # Initialize detector
        from .regime_detector import RegimeDetector
        self.detector = RegimeDetector(self.config)
        
        # Initialize indicators
        self._initialize_indicators()
        
        # Initialize data components
        self._initialize_data_components()
        
    def _initialize_indicators(self):
        """Initialize indicator modules"""
        self.indicators = {}
        
        # ATR Analysis
        if self.config.get('indicators', {}).get('atr', {}).get('enabled', True):
            from ..indicators.atr_analysis import ATRAnalysis
            self.indicators['atr'] = ATRAnalysis(self.config['indicators']['atr'])
        
        # IV Analysis
        if self.config.get('indicators', {}).get('iv', {}).get('enabled', True):
            from ..indicators.iv_analysis import IVAnalysis
            self.indicators['iv'] = IVAnalysis(self.config['indicators']['iv'])
        
        # Greek Sentiment
        if self.config.get('indicators', {}).get('greek', {}).get('enabled', True):
            from ..indicators.greek_sentiment import GreekSentiment
            self.indicators['greek'] = GreekSentiment(self.config['indicators']['greek'])
        
        # OI Price Action
        if self.config.get('indicators', {}).get('oi_pa', {}).get('enabled', True):
            from ..indicators.oi_price_action import OIPriceAction
            self.indicators['oi_pa'] = OIPriceAction(self.config['indicators']['oi_pa'])
        
        # Straddle Analysis
        if self.config.get('indicators', {}).get('straddle', {}).get('enabled', True):
            from ..indicators.straddle_analysis import StraddleAnalysis
            self.indicators['straddle'] = StraddleAnalysis(self.config['indicators']['straddle'])
            
    def _initialize_data_components(self):
        """Initialize data layer components"""
        from ..data.loaders import DataLoader
        from ..data.processors import DataProcessor
        from ..data.cache_manager import CacheManager
        
        self.data_loader = DataLoader(self.config.get('data', {}))
        self.data_processor = DataProcessor(self.config.get('data', {}))
        self.cache_manager = CacheManager(self.config.get('cache', {}))
        
    def _initialize_legacy_modules(self):
        """Initialize legacy modules for backward compatibility"""
        try:
            self.legacy_engine = EnhancedMarketRegimeEngine(
                self.config.get('legacy_config', {})
            )
            self.legacy_straddle = StraddleAnalysisEngine(
                self.config.get('straddle_config', {})
            )
            self.logger.info("Legacy modules initialized for backward compatibility")
        except Exception as e:
            self.logger.warning(f"Failed to initialize legacy modules: {e}")
            self.legacy_engine = None
            self.legacy_straddle = None
    
    def analyze(self, 
                market_data: pd.DataFrame,
                use_cache: bool = True) -> RegimeAnalysisResult:
        """
        Perform complete market regime analysis
        
        Args:
            market_data: DataFrame with market data
            use_cache: Whether to use cached results
            
        Returns:
            RegimeAnalysisResult with complete analysis
        """
        try:
            # Check cache first
            if use_cache:
                cache_key = self._generate_cache_key(market_data)
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    self.logger.debug("Using cached result")
                    return cached_result
            
            # Preprocess data
            processed_data = self.data_processor.process(market_data)
            
            # Calculate indicators
            indicator_results = self._calculate_indicators(processed_data)
            
            # Detect regime
            detection_result = self.detector.detect(processed_data, indicator_results)
            
            # Classify regime
            regime_classification = self.classifier.classify(detection_result)
            
            # Generate signals
            signals = self._generate_signals(
                processed_data, 
                indicator_results, 
                regime_classification
            )
            
            # Aggregate results
            result = RegimeAnalysisResult(
                regime=regime_classification['regime'],
                confidence=regime_classification['confidence'],
                signals=signals,
                indicators=indicator_results,
                timestamp=datetime.now(),
                metadata={
                    'data_points': len(market_data),
                    'processing_time': self._get_processing_time(),
                    'regime_details': regime_classification
                }
            )
            
            # Cache result
            if use_cache:
                self.cache_manager.set(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in market regime analysis: {e}")
            raise
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all enabled indicators"""
        results = {}
        
        for name, indicator in self.indicators.items():
            try:
                results[name] = indicator.calculate(data)
            except Exception as e:
                self.logger.error(f"Error calculating {name} indicator: {e}")
                results[name] = None
                
        return results
    
    def _generate_signals(self,
                         data: pd.DataFrame,
                         indicators: Dict[str, Any],
                         regime: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on indicators and regime"""
        signals = {
            'action': 'HOLD',
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Get signals from each indicator
        for name, indicator in self.indicators.items():
            if indicators.get(name):
                try:
                    indicator_signals = indicator.get_signals(indicators[name])
                    signals['indicators'][name] = indicator_signals
                except Exception as e:
                    self.logger.error(f"Error getting signals from {name}: {e}")
        
        # Aggregate signals based on regime
        signals = self.analyzer.aggregate_signals(signals, regime)
        
        return signals
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key for data"""
        # Simple implementation - can be enhanced
        return f"regime_{len(data)}_{data.index[-1]}"
    
    def _get_processing_time(self) -> float:
        """Get processing time (placeholder)"""
        return 0.0
    
    def get_regime_history(self, 
                          start_date: datetime,
                          end_date: datetime) -> List[RegimeAnalysisResult]:
        """Get historical regime analysis results"""
        # Implementation depends on storage strategy
        pass
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration dynamically"""
        self.config.update(new_config)
        self._initialize_components()
        self.logger.info("Configuration updated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'status': 'operational',
            'indicators': list(self.indicators.keys()),
            'legacy_modules': LEGACY_MODULES_AVAILABLE,
            'cache_size': self.cache_manager.size(),
            'config_version': self.config.get('version', 'unknown')
        }