"""
Multi-Timeframe Pattern Detector for 10-Component Analysis

Detects patterns across all 10 components with multiple timeframe analysis:
- 6 individual components (ATM/ITM1/OTM1 × CE/PE)
- 3 individual straddles (ATM, ITM1, OTM1)
- 1 combined triple straddle
- 4 timeframes (3, 5, 10, 15 minutes)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .pattern_repository import PatternRepository, PatternSchema
from ..core.calculation_engine import CalculationEngine
from ..rolling.window_manager import RollingWindowManager

logger = logging.getLogger(__name__)


@dataclass
class PatternDetectionResult:
    """Result of pattern detection analysis"""
    timestamp: datetime
    detected_patterns: List[PatternSchema]
    confluence_score: float
    timeframe_alignment: Dict[str, float]
    component_analysis: Dict[str, Dict[str, Any]]
    risk_assessment: Dict[str, float]


class MultiTimeframePatternDetector:
    """
    Advanced Multi-Timeframe Pattern Detector for 10 Components
    
    Detects and validates patterns across:
    - All 10 components simultaneously
    - 4 timeframes (3, 5, 10, 15 minutes) 
    - Technical indicator confluences
    - Support/resistance interactions
    - Cross-component correlations
    
    Features:
    - Real-time pattern detection
    - Multi-timeframe confluence validation
    - Statistical significance testing
    - Risk-adjusted pattern scoring
    """
    
    def __init__(self, 
                 pattern_repository: PatternRepository,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager,
                 config: Optional[Dict] = None):
        """
        Initialize Multi-Timeframe Pattern Detector
        
        Args:
            pattern_repository: Pattern storage repository
            calculation_engine: Calculation engine for indicators
            window_manager: Rolling window manager
            config: Detector configuration
        """
        self.pattern_repository = pattern_repository
        self.calculation_engine = calculation_engine
        self.window_manager = window_manager
        self.config = config or self._get_default_config()
        
        # All 10 components
        self.all_components = [
            'ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE',
            'ATM_STRADDLE', 'ITM1_STRADDLE', 'OTM1_STRADDLE', 'COMBINED_TRIPLE_STRADDLE'
        ]
        
        # Individual components (6)
        self.individual_components = [
            'ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE'
        ]
        
        # Straddle components (4 = 3 individual + 1 combined)
        self.straddle_components = [
            'ATM_STRADDLE', 'ITM1_STRADDLE', 'OTM1_STRADDLE', 'COMBINED_TRIPLE_STRADDLE'
        ]
        
        # Timeframes
        self.timeframes = [3, 5, 10, 15]  # minutes
        
        # Technical indicators to analyze
        self.technical_indicators = {
            "trend": ["EMA_20", "EMA_100", "EMA_200", "SMA_50"],
            "momentum": ["RSI", "MACD", "Stochastic", "Williams_R"],
            "volume": ["VWAP", "Volume_Profile", "OBV", "VWMA"],
            "volatility": ["ATR", "Bollinger_Bands", "Keltner_Channel"],
            "support_resistance": ["Pivot_Points", "Fibonacci", "Supply_Demand_Zones"]
        }
        
        # Pattern types to detect
        self.pattern_types = {
            "single_component": self.individual_components,
            "individual_straddle": self.straddle_components[:3],  # Exclude combined
            "combined_triple": ["COMBINED_TRIPLE_STRADDLE"],
            "cross_component": "multiple"
        }
        
        # Detection thresholds
        self.detection_thresholds = {
            "confluence_threshold": 0.85,
            "timeframe_alignment": 0.80,
            "volume_confirmation": 0.75,
            "statistical_significance": 0.95,
            "min_pattern_strength": 0.70
        }
        
        # Pattern cache for real-time detection
        self._active_patterns = {}
        self._pattern_history = []
        
        self.logger = logging.getLogger(f"{__name__}.MultiTimeframePatternDetector")
        self.logger.info(f"Pattern detector initialized for {len(self.all_components)} components")
        self.logger.info(f"Monitoring {len(self.timeframes)} timeframes: {self.timeframes}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default detector configuration"""
        return {
            'enable_real_time': True,
            'enable_statistical_validation': True,
            'enable_ml_scoring': True,
            'enable_risk_assessment': True,
            'confluence_required': True,
            'timeframe_validation': True,
            'min_data_points': 50,
            'max_patterns_per_detection': 10
        }
    
    def detect_patterns(self, market_data: Dict[str, Any], 
                       timestamp: Optional[datetime] = None) -> PatternDetectionResult:
        """
        Detect patterns across all components and timeframes
        
        Args:
            market_data: Real-time market data for all components
            timestamp: Current timestamp
            
        Returns:
            PatternDetectionResult with detected patterns
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Extract component data for all 10 components
            component_data = self._extract_component_data(market_data)
            
            # Detect patterns for each component type
            detected_patterns = []
            
            # 1. Single component patterns (6 individual components)
            single_patterns = self._detect_single_component_patterns(component_data, timestamp)
            detected_patterns.extend(single_patterns)
            
            # 2. Individual straddle patterns (3 straddles)
            straddle_patterns = self._detect_individual_straddle_patterns(component_data, timestamp)
            detected_patterns.extend(straddle_patterns)
            
            # 3. Combined triple straddle patterns (1 combined)
            triple_patterns = self._detect_combined_triple_patterns(component_data, timestamp)
            detected_patterns.extend(triple_patterns)
            
            # 4. Cross-component patterns (multiple components)
            cross_patterns = self._detect_cross_component_patterns(component_data, timestamp)
            detected_patterns.extend(cross_patterns)
            
            # Validate and score patterns across timeframes
            validated_patterns = self._validate_multi_timeframe_patterns(detected_patterns, component_data)
            
            # Calculate confluence scores
            confluence_score = self._calculate_overall_confluence(validated_patterns)
            
            # Calculate timeframe alignment
            timeframe_alignment = self._calculate_timeframe_alignment(validated_patterns)
            
            # Perform component analysis
            component_analysis = self._analyze_components(component_data, timestamp)
            
            # Risk assessment
            risk_assessment = self._calculate_risk_assessment(validated_patterns, component_analysis)
            
            result = PatternDetectionResult(
                timestamp=timestamp,
                detected_patterns=validated_patterns,
                confluence_score=confluence_score,
                timeframe_alignment=timeframe_alignment,
                component_analysis=component_analysis,
                risk_assessment=risk_assessment
            )
            
            # Store validated patterns
            self._store_detected_patterns(validated_patterns)
            
            # Update active patterns
            self._update_active_patterns(validated_patterns)
            
            self.logger.debug(f"Detected {len(validated_patterns)} patterns at {timestamp}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {e}")
            return PatternDetectionResult(
                timestamp=timestamp,
                detected_patterns=[],
                confluence_score=0.0,
                timeframe_alignment={},
                component_analysis={},
                risk_assessment={}
            )
    
    def _extract_component_data(self, market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract data for all 10 components"""
        component_data = {}
        
        # Extract individual component data (6)
        for component in self.individual_components:
            component_data[component] = {
                'price': market_data.get(component, 0.0),
                'volume': market_data.get(f'{component}_volume', 0),
                'high': market_data.get(f'{component}_high', market_data.get(component, 0.0)),
                'low': market_data.get(f'{component}_low', market_data.get(component, 0.0)),
                'open': market_data.get(f'{component}_open', market_data.get(component, 0.0)),
                'timestamp': market_data.get('timestamp', datetime.now())
            }
        
        # Calculate straddle data (3 individual straddles)
        component_data['ATM_STRADDLE'] = {
            'price': component_data['ATM_CE']['price'] + component_data['ATM_PE']['price'],
            'volume': component_data['ATM_CE']['volume'] + component_data['ATM_PE']['volume'],
            'correlation': self._calculate_component_correlation('ATM_CE', 'ATM_PE'),
            'spread_ratio': self._calculate_spread_ratio('ATM_CE', 'ATM_PE'),
            'timestamp': market_data.get('timestamp', datetime.now())
        }
        
        component_data['ITM1_STRADDLE'] = {
            'price': component_data['ITM1_CE']['price'] + component_data['ITM1_PE']['price'],
            'volume': component_data['ITM1_CE']['volume'] + component_data['ITM1_PE']['volume'],
            'correlation': self._calculate_component_correlation('ITM1_CE', 'ITM1_PE'),
            'spread_ratio': self._calculate_spread_ratio('ITM1_CE', 'ITM1_PE'),
            'timestamp': market_data.get('timestamp', datetime.now())
        }
        
        component_data['OTM1_STRADDLE'] = {
            'price': component_data['OTM1_CE']['price'] + component_data['OTM1_PE']['price'],
            'volume': component_data['OTM1_CE']['volume'] + component_data['OTM1_PE']['volume'],
            'correlation': self._calculate_component_correlation('OTM1_CE', 'OTM1_PE'),
            'spread_ratio': self._calculate_spread_ratio('OTM1_CE', 'OTM1_PE'),
            'timestamp': market_data.get('timestamp', datetime.now())
        }
        
        # Calculate combined triple straddle data
        component_data['COMBINED_TRIPLE_STRADDLE'] = {
            'price': (component_data['ATM_STRADDLE']['price'] + 
                     component_data['ITM1_STRADDLE']['price'] + 
                     component_data['OTM1_STRADDLE']['price']),
            'volume': (component_data['ATM_STRADDLE']['volume'] + 
                      component_data['ITM1_STRADDLE']['volume'] + 
                      component_data['OTM1_STRADDLE']['volume']),
            'weighted_correlation': self._calculate_weighted_correlation(),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(component_data),
            'timestamp': market_data.get('timestamp', datetime.now())
        }
        
        return component_data
    
    def _detect_single_component_patterns(self, component_data: Dict[str, Dict[str, Any]], 
                                        timestamp: datetime) -> List[PatternSchema]:
        """Detect patterns for individual components (6)"""
        patterns = []
        
        for component in self.individual_components:
            if component not in component_data:
                continue
            
            comp_data = component_data[component]
            
            # Detect patterns across all timeframes
            for timeframe in self.timeframes:
                timeframe_patterns = self._detect_component_timeframe_patterns(
                    component, comp_data, timeframe, timestamp
                )
                patterns.extend(timeframe_patterns)
        
        return patterns
    
    def _detect_component_timeframe_patterns(self, component: str, 
                                           comp_data: Dict[str, Any],
                                           timeframe: int,
                                           timestamp: datetime) -> List[PatternSchema]:
        """Detect patterns for a specific component and timeframe"""
        patterns = []
        
        try:
            # Get rolling data for this timeframe
            rolling_data = self.window_manager.get_window_data(component.lower(), timeframe)
            if not rolling_data or len(rolling_data[0]) < self.config['min_data_points']:
                return patterns
            
            # Calculate technical indicators for this timeframe
            indicators = self._calculate_timeframe_indicators(component, timeframe)
            
            # Detect specific pattern types
            pattern_detections = {
                'ema_rejection': self._detect_ema_rejection_pattern(component, timeframe, indicators),
                'pivot_support': self._detect_pivot_support_pattern(component, timeframe, indicators),
                'vwap_bounce': self._detect_vwap_bounce_pattern(component, timeframe, indicators),
                'volume_confirmation': self._detect_volume_confirmation_pattern(component, timeframe, indicators),
                'momentum_divergence': self._detect_momentum_divergence_pattern(component, timeframe, indicators)
            }
            
            # Create patterns for each detection
            for pattern_name, detection_result in pattern_detections.items():
                if detection_result and detection_result.get('strength', 0) > self.detection_thresholds['min_pattern_strength']:
                    pattern = self._create_pattern_schema(
                        component, timeframe, pattern_name, detection_result, timestamp
                    )
                    if pattern:
                        patterns.append(pattern)
        
        except Exception as e:
            self.logger.warning(f"Error detecting patterns for {component} {timeframe}min: {e}")
        
        return patterns
    
    def _detect_ema_rejection_pattern(self, component: str, timeframe: int, 
                                    indicators: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect EMA rejection pattern"""
        try:
            current_price = indicators.get('current_price', 0)
            ema_20 = indicators.get(f'ema_20_{timeframe}min', 0)
            ema_100 = indicators.get(f'ema_100_{timeframe}min', 0)
            ema_200 = indicators.get(f'ema_200_{timeframe}min', 0)
            volume = indicators.get('volume_ratio', 1.0)
            
            # Check for rejection at key EMA levels
            rejection_signals = []
            
            # EMA 200 rejection (strongest signal)
            if abs(current_price - ema_200) / ema_200 < 0.005 and volume > 1.5:
                rejection_signals.append({
                    'level': 'EMA_200',
                    'strength': 0.9,
                    'volume_confirmation': volume > 1.5
                })
            
            # EMA 100 rejection
            if abs(current_price - ema_100) / ema_100 < 0.005 and volume > 1.3:
                rejection_signals.append({
                    'level': 'EMA_100',
                    'strength': 0.8,
                    'volume_confirmation': volume > 1.3
                })
            
            # EMA 20 rejection
            if abs(current_price - ema_20) / ema_20 < 0.003 and volume > 1.2:
                rejection_signals.append({
                    'level': 'EMA_20',
                    'strength': 0.7,
                    'volume_confirmation': volume > 1.2
                })
            
            if rejection_signals:
                return {
                    'pattern_type': 'ema_rejection',
                    'signals': rejection_signals,
                    'strength': max(signal['strength'] for signal in rejection_signals),
                    'volume_confirmation': any(signal['volume_confirmation'] for signal in rejection_signals),
                    'timeframe': timeframe,
                    'component': component
                }
            
        except Exception as e:
            self.logger.warning(f"Error detecting EMA rejection: {e}")
        
        return None
    
    def _detect_pivot_support_pattern(self, component: str, timeframe: int,
                                    indicators: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect pivot support/resistance pattern"""
        try:
            current_price = indicators.get('current_price', 0)
            pivot_s1 = indicators.get('pivot_S1', 0)
            pivot_r1 = indicators.get('pivot_R1', 0)
            pivot_point = indicators.get('pivot_PP', 0)
            volume = indicators.get('volume_ratio', 1.0)
            
            support_signals = []
            
            # Pivot S1 support
            if abs(current_price - pivot_s1) / pivot_s1 < 0.002 and volume > 1.4:
                support_signals.append({
                    'level': 'PIVOT_S1',
                    'strength': 0.85,
                    'price_level': pivot_s1
                })
            
            # Pivot R1 resistance
            if abs(current_price - pivot_r1) / pivot_r1 < 0.002 and volume > 1.4:
                support_signals.append({
                    'level': 'PIVOT_R1',
                    'strength': 0.85,
                    'price_level': pivot_r1
                })
            
            # Pivot Point support/resistance
            if abs(current_price - pivot_point) / pivot_point < 0.002 and volume > 1.3:
                support_signals.append({
                    'level': 'PIVOT_PP',
                    'strength': 0.80,
                    'price_level': pivot_point
                })
            
            if support_signals:
                return {
                    'pattern_type': 'pivot_support',
                    'signals': support_signals,
                    'strength': max(signal['strength'] for signal in support_signals),
                    'volume_confirmation': volume > 1.3,
                    'timeframe': timeframe,
                    'component': component
                }
            
        except Exception as e:
            self.logger.warning(f"Error detecting pivot support: {e}")
        
        return None
    
    def _detect_vwap_bounce_pattern(self, component: str, timeframe: int,
                                  indicators: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect VWAP bounce pattern"""
        try:
            current_price = indicators.get('current_price', 0)
            vwap = indicators.get(f'vwap_{timeframe}min', 0)
            volume = indicators.get('volume_ratio', 1.0)
            rsi = indicators.get('rsi', 50)
            
            if vwap == 0:
                return None
            
            vwap_distance = abs(current_price - vwap) / vwap
            
            # VWAP bounce criteria
            if (vwap_distance < 0.003 and volume > 1.5 and 
                (rsi < 35 or rsi > 65)):  # Oversold/overbought at VWAP
                
                return {
                    'pattern_type': 'vwap_bounce',
                    'strength': 0.8 if volume > 2.0 else 0.7,
                    'vwap_level': vwap,
                    'distance_ratio': vwap_distance,
                    'volume_confirmation': volume > 1.5,
                    'rsi_confirmation': rsi < 35 or rsi > 65,
                    'timeframe': timeframe,
                    'component': component
                }
            
        except Exception as e:
            self.logger.warning(f"Error detecting VWAP bounce: {e}")
        
        return None
    
    def _detect_volume_confirmation_pattern(self, component: str, timeframe: int,
                                          indicators: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect volume confirmation pattern"""
        try:
            volume_ratio = indicators.get('volume_ratio', 1.0)
            price_change_pct = indicators.get('price_change_percent', 0)
            
            # Volume surge with price movement
            if volume_ratio > 2.0 and abs(price_change_pct) > 1.0:
                return {
                    'pattern_type': 'volume_confirmation',
                    'strength': min(0.9, volume_ratio / 3.0),
                    'volume_ratio': volume_ratio,
                    'price_change_percent': price_change_pct,
                    'timeframe': timeframe,
                    'component': component
                }
            
        except Exception as e:
            self.logger.warning(f"Error detecting volume confirmation: {e}")
        
        return None
    
    def _detect_momentum_divergence_pattern(self, component: str, timeframe: int,
                                          indicators: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect momentum divergence pattern"""
        try:
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            price_change_pct = indicators.get('price_change_percent', 0)
            
            # RSI divergence
            if ((rsi < 30 and price_change_pct < -2) or 
                (rsi > 70 and price_change_pct > 2)):
                
                return {
                    'pattern_type': 'momentum_divergence',
                    'strength': 0.75,
                    'rsi': rsi,
                    'macd': macd,
                    'price_change_percent': price_change_pct,
                    'divergence_type': 'bullish' if rsi < 30 else 'bearish',
                    'timeframe': timeframe,
                    'component': component
                }
            
        except Exception as e:
            self.logger.warning(f"Error detecting momentum divergence: {e}")
        
        return None