"""
Enhanced Resistance Analyzer for 10-Component Multi-Timeframe Analysis

Provides comprehensive support/resistance analysis across all 10 components:
- 6 Individual: ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
- 3 Individual Straddles: ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE
- 1 Combined Triple: COMBINED_TRIPLE_STRADDLE

Features multi-timeframe (3,5,10,15 min) resistance level detection with
pattern recognition integration for enhanced trading signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from numba import jit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ResistanceLevel:
    """Individual resistance/support level"""
    level: float
    level_type: str  # 'support', 'resistance', 'pivot'
    strength: float  # 0-1 strength score
    touch_count: int
    first_touch: datetime
    last_touch: datetime
    volume_profile: Dict[str, float]
    timeframes_active: List[int]
    components_involved: List[str]
    confidence: float
    pattern_relevance: float  # Relevance for pattern detection


@dataclass
class ComponentResistanceResult:
    """Resistance analysis result for a single component"""
    component_name: str
    timestamp: datetime
    timeframe: int
    
    # Current levels
    support_levels: List[ResistanceLevel]
    resistance_levels: List[ResistanceLevel]
    pivot_levels: List[ResistanceLevel]
    
    # Level interactions
    current_price: float
    nearest_support: Optional[ResistanceLevel]
    nearest_resistance: Optional[ResistanceLevel]
    level_interactions: Dict[str, Any]
    
    # Pattern indicators
    breakout_probability: float
    reversal_probability: float
    consolidation_score: float
    trend_direction: str
    
    # Multi-timeframe analysis
    timeframe_consistency: Dict[int, float]
    cross_timeframe_levels: List[ResistanceLevel]


@dataclass
class EnhancedResistanceAnalysisResult:
    """Comprehensive resistance analysis for all 10 components"""
    timestamp: datetime
    analysis_duration: float
    
    # Component-wise results
    component_results: Dict[str, ComponentResistanceResult]
    
    # Cross-component analysis
    confluence_zones: List[Dict[str, Any]]
    multi_component_levels: List[ResistanceLevel]
    correlation_with_levels: Dict[str, float]
    
    # Pattern integration
    pattern_relevant_levels: List[ResistanceLevel]
    pattern_confluence_score: float
    regime_indicators: Dict[str, float]
    
    # Trading signals
    breakout_signals: List[Dict[str, Any]]
    reversal_signals: List[Dict[str, Any]]
    consolidation_signals: List[Dict[str, Any]]


class Enhanced10ComponentResistanceAnalyzer:
    """
    Enhanced Resistance Analyzer for All 10 Components
    
    Analyzes support/resistance levels across:
    - Individual Components (6): ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
    - Individual Straddles (3): ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE
    - Combined Triple (1): COMBINED_TRIPLE_STRADDLE
    
    Features:
    - Multi-timeframe level detection (3, 5, 10, 15 minutes)
    - Volume-weighted level validation
    - Cross-component level confluence
    - Pattern-relevant level identification
    - Real-time breakout/reversal signal generation
    """
    
    def __init__(self, window_manager=None, config: Optional[Dict] = None):
        """
        Initialize Enhanced Resistance Analyzer
        
        Args:
            window_manager: Rolling window manager for data access
            config: Analyzer configuration
        """
        self.window_manager = window_manager
        self.config = config or self._get_default_config()
        
        # All 10 component definitions
        self.all_components = [
            'ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE',  # Individual (6)
            'ATM_STRADDLE', 'ITM1_STRADDLE', 'OTM1_STRADDLE',  # Individual Straddles (3)
            'COMBINED_TRIPLE_STRADDLE'  # Combined Triple (1)
        ]
        
        # Component categories
        self.individual_components = self.all_components[:6]
        self.straddle_components = self.all_components[6:9]
        self.combined_component = self.all_components[9:]
        
        # Multi-timeframe analysis
        self.timeframes = [3, 5, 10, 15]  # minutes
        
        # Level detection parameters
        self.level_detection_params = {
            'min_touches': 2,
            'touch_tolerance': 0.001,  # 0.1% tolerance
            'min_strength': 0.3,
            'volume_weight': 0.4,
            'time_decay_factor': 0.95,
            'confluence_distance': 0.005  # 0.5% for confluence zones
        }
        
        # Pattern integration parameters
        self.pattern_params = {
            'pattern_relevance_threshold': 0.7,
            'confluence_strength_threshold': 0.8,
            'breakout_volume_multiplier': 1.5,
            'reversal_rsi_threshold': [30, 70]
        }
        
        # Storage for resistance levels by component and timeframe
        self.resistance_history = {}
        self.confluence_zones_history = []
        
        # Initialize storage
        for component in self.all_components:
            self.resistance_history[component] = {}
            for timeframe in self.timeframes:
                self.resistance_history[component][timeframe] = {
                    'support_levels': [],
                    'resistance_levels': [],
                    'pivot_levels': []
                }
        
        # Performance tracking
        self.analysis_count = 0
        self.pattern_signals_generated = 0
        
        self.logger = logging.getLogger(f"{__name__}.Enhanced10ComponentResistanceAnalyzer")
        self.logger.info(f"Enhanced Resistance Analyzer initialized for {len(self.all_components)} components")
        self.logger.info(f"Multi-timeframe analysis: {self.timeframes}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default analyzer configuration"""
        return {
            'enable_volume_weighting': True,
            'enable_multi_timeframe': True,
            'enable_pattern_integration': True,
            'enable_confluence_detection': True,
            'min_data_points': 50,
            'max_levels_per_component': 10,
            'level_expiry_hours': 24,
            'real_time_updates': True
        }
    
    def analyze_all_components(self, market_data: Dict[str, Any], 
                             timestamp: Optional[datetime] = None) -> EnhancedResistanceAnalysisResult:
        """
        Perform comprehensive resistance analysis for all 10 components
        
        Args:
            market_data: Real-time market data for all components
            timestamp: Analysis timestamp
            
        Returns:
            EnhancedResistanceAnalysisResult with complete analysis
        """
        start_time = datetime.now()
        if timestamp is None:
            timestamp = start_time
        
        try:
            # Extract component prices
            component_prices = self._extract_component_prices(market_data)
            
            # Analyze each component across all timeframes
            component_results = {}
            
            for component in self.all_components:
                if component in component_prices:
                    component_result = self._analyze_component_resistance(
                        component, component_prices[component], timestamp
                    )
                    if component_result:
                        component_results[component] = component_result
            
            # Cross-component analysis
            confluence_zones = self._detect_confluence_zones(component_results, timestamp)
            
            # Multi-component level detection
            multi_component_levels = self._detect_multi_component_levels(component_results)
            
            # Correlation with resistance levels
            correlation_with_levels = self._calculate_level_correlations(component_results)
            
            # Pattern integration
            pattern_relevant_levels = self._identify_pattern_relevant_levels(component_results)
            pattern_confluence_score = self._calculate_pattern_confluence_score(
                pattern_relevant_levels, confluence_zones
            )
            
            # Regime indicators
            regime_indicators = self._calculate_resistance_regime_indicators(component_results)
            
            # Generate trading signals
            breakout_signals = self._generate_breakout_signals(component_results, confluence_zones)
            reversal_signals = self._generate_reversal_signals(component_results, confluence_zones)
            consolidation_signals = self._generate_consolidation_signals(component_results)
            
            # Calculate analysis duration
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = EnhancedResistanceAnalysisResult(
                timestamp=timestamp,
                analysis_duration=analysis_duration,
                component_results=component_results,
                confluence_zones=confluence_zones,
                multi_component_levels=multi_component_levels,
                correlation_with_levels=correlation_with_levels,
                pattern_relevant_levels=pattern_relevant_levels,
                pattern_confluence_score=pattern_confluence_score,
                regime_indicators=regime_indicators,
                breakout_signals=breakout_signals,
                reversal_signals=reversal_signals,
                consolidation_signals=consolidation_signals
            )
            
            # Update history
            self._update_resistance_history(result)
            
            # Update performance tracking
            self.analysis_count += 1
            self.pattern_signals_generated += len(breakout_signals) + len(reversal_signals)
            
            self.logger.debug(f"Resistance analysis completed in {analysis_duration:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in resistance analysis: {e}")
            # Return empty result on error
            return EnhancedResistanceAnalysisResult(
                timestamp=timestamp,
                analysis_duration=0.0,
                component_results={},
                confluence_zones=[],
                multi_component_levels=[],
                correlation_with_levels={},
                pattern_relevant_levels=[],
                pattern_confluence_score=0.0,
                regime_indicators={},
                breakout_signals=[],
                reversal_signals=[],
                consolidation_signals=[]
            )
    
    def _extract_component_prices(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract current prices for all 10 components"""
        component_prices = {}
        
        # Extract individual component prices (6)
        for component in self.individual_components:
            price = market_data.get(component, 0.0)
            if price > 0:
                component_prices[component] = price
        
        # Calculate straddle prices (3)
        if 'ATM_CE' in component_prices and 'ATM_PE' in component_prices:
            component_prices['ATM_STRADDLE'] = component_prices['ATM_CE'] + component_prices['ATM_PE']
        
        if 'ITM1_CE' in component_prices and 'ITM1_PE' in component_prices:
            component_prices['ITM1_STRADDLE'] = component_prices['ITM1_CE'] + component_prices['ITM1_PE']
        
        if 'OTM1_CE' in component_prices and 'OTM1_PE' in component_prices:
            component_prices['OTM1_STRADDLE'] = component_prices['OTM1_CE'] + component_prices['OTM1_PE']
        
        # Calculate combined triple straddle price
        straddle_prices = [
            component_prices.get('ATM_STRADDLE', 0),
            component_prices.get('ITM1_STRADDLE', 0),
            component_prices.get('OTM1_STRADDLE', 0)
        ]
        
        if all(price > 0 for price in straddle_prices):
            component_prices['COMBINED_TRIPLE_STRADDLE'] = sum(straddle_prices)
        
        return component_prices
    
    def _analyze_component_resistance(self, component: str, current_price: float, 
                                    timestamp: datetime) -> Optional[ComponentResistanceResult]:
        """Analyze resistance levels for a single component across all timeframes"""
        try:
            # Get multi-timeframe data for this component
            timeframe_results = {}
            
            for timeframe in self.timeframes:
                timeframe_result = self._analyze_component_timeframe(
                    component, current_price, timeframe, timestamp
                )
                if timeframe_result:
                    timeframe_results[timeframe] = timeframe_result
            
            if not timeframe_results:
                return None
            
            # Aggregate results across timeframes
            all_support_levels = []
            all_resistance_levels = []
            all_pivot_levels = []
            
            for tf_result in timeframe_results.values():
                all_support_levels.extend(tf_result['support_levels'])
                all_resistance_levels.extend(tf_result['resistance_levels'])
                all_pivot_levels.extend(tf_result['pivot_levels'])
            
            # Find strongest levels (remove duplicates and rank by strength)
            support_levels = self._consolidate_levels(all_support_levels)
            resistance_levels = self._consolidate_levels(all_resistance_levels)
            pivot_levels = self._consolidate_levels(all_pivot_levels)
            
            # Find nearest levels
            nearest_support = self._find_nearest_level(current_price, support_levels, 'below')
            nearest_resistance = self._find_nearest_level(current_price, resistance_levels, 'above')
            
            # Calculate level interactions
            level_interactions = self._calculate_level_interactions(
                current_price, support_levels, resistance_levels, pivot_levels
            )
            
            # Calculate probabilities
            breakout_probability = self._calculate_breakout_probability(
                current_price, resistance_levels, timeframe_results
            )
            reversal_probability = self._calculate_reversal_probability(
                current_price, support_levels, resistance_levels, timeframe_results
            )
            consolidation_score = self._calculate_consolidation_score(
                current_price, support_levels, resistance_levels
            )
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(
                current_price, support_levels, resistance_levels, timeframe_results
            )
            
            # Calculate timeframe consistency
            timeframe_consistency = self._calculate_timeframe_consistency(timeframe_results)
            
            # Cross-timeframe levels (levels that appear in multiple timeframes)
            cross_timeframe_levels = self._find_cross_timeframe_levels(timeframe_results)
            
            return ComponentResistanceResult(
                component_name=component,
                timestamp=timestamp,
                timeframe=0,  # Aggregated across all timeframes
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                pivot_levels=pivot_levels,
                current_price=current_price,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                level_interactions=level_interactions,
                breakout_probability=breakout_probability,
                reversal_probability=reversal_probability,
                consolidation_score=consolidation_score,
                trend_direction=trend_direction,
                timeframe_consistency=timeframe_consistency,
                cross_timeframe_levels=cross_timeframe_levels
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing resistance for {component}: {e}")
            return None
    
    def _analyze_component_timeframe(self, component: str, current_price: float,
                                   timeframe: int, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Analyze resistance levels for a component in a specific timeframe"""
        if not self.window_manager:
            return None
        
        try:
            # Get price data for this timeframe
            component_data, timestamps = self.window_manager.get_window_data(
                component.lower(), timeframe
            )
            
            if not component_data or len(component_data) < self.config['min_data_points']:
                return None
            
            # Extract price and volume data
            prices = [dp.get('close', dp.get('price', 0)) for dp in component_data]
            volumes = [dp.get('volume', 0) for dp in component_data]
            highs = [dp.get('high', dp.get('close', dp.get('price', 0))) for dp in component_data]
            lows = [dp.get('low', dp.get('close', dp.get('price', 0))) for dp in component_data]
            
            # Detect support levels
            support_levels = self._detect_support_levels(
                prices, volumes, timestamps, timeframe, component
            )
            
            # Detect resistance levels
            resistance_levels = self._detect_resistance_levels(
                prices, volumes, timestamps, timeframe, component
            )
            
            # Detect pivot levels
            pivot_levels = self._detect_pivot_levels(
                highs, lows, prices, volumes, timestamps, timeframe, component
            )
            
            # Calculate volume profile for levels
            for level_list in [support_levels, resistance_levels, pivot_levels]:
                for level in level_list:
                    level.volume_profile = self._calculate_volume_profile_for_level(
                        level.level, prices, volumes
                    )
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'pivot_levels': pivot_levels,
                'timeframe': timeframe,
                'data_points': len(prices)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing {component} {timeframe}min timeframe: {e}")
            return None
    
    @staticmethod
    @jit(nopython=True)
    def _fast_peak_detection(prices: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Fast peak and trough detection using Numba"""
        n = len(prices)
        peaks = np.zeros(n, dtype=np.bool_)
        troughs = np.zeros(n, dtype=np.bool_)
        
        for i in range(window, n - window):
            # Check for peak
            is_peak = True
            for j in range(i - window, i + window + 1):
                if j != i and prices[j] >= prices[i]:
                    is_peak = False
                    break
            peaks[i] = is_peak
            
            # Check for trough
            is_trough = True
            for j in range(i - window, i + window + 1):
                if j != i and prices[j] <= prices[i]:
                    is_trough = False
                    break
            troughs[i] = is_trough
        
        return peaks, troughs
    
    def _detect_support_levels(self, prices: List[float], volumes: List[float],
                             timestamps: List[datetime], timeframe: int,
                             component: str) -> List[ResistanceLevel]:
        """Detect support levels using multiple methods"""
        support_levels = []
        
        try:
            prices_array = np.array(prices)
            volumes_array = np.array(volumes)
            
            # Method 1: Local minima detection
            _, troughs = self._fast_peak_detection(prices_array)
            trough_indices = np.where(troughs)[0]
            
            # Group nearby troughs into support levels
            level_groups = self._group_nearby_levels(
                prices_array[trough_indices], trough_indices, self.level_detection_params['touch_tolerance']
            )
            
            for group_prices, group_indices in level_groups:
                if len(group_prices) >= self.level_detection_params['min_touches']:
                    level_price = np.mean(group_prices)
                    
                    # Calculate level strength
                    strength = self._calculate_level_strength(
                        level_price, group_prices, volumes_array[group_indices], 'support'
                    )
                    
                    if strength >= self.level_detection_params['min_strength']:
                        support_level = ResistanceLevel(
                            level=level_price,
                            level_type='support',
                            strength=strength,
                            touch_count=len(group_prices),
                            first_touch=timestamps[group_indices[0]],
                            last_touch=timestamps[group_indices[-1]],
                            volume_profile={},
                            timeframes_active=[timeframe],
                            components_involved=[component],
                            confidence=min(0.9, strength + 0.1),
                            pattern_relevance=self._calculate_pattern_relevance(level_price, component)
                        )
                        support_levels.append(support_level)
            
            # Method 2: Volume-based support detection
            volume_support_levels = self._detect_volume_based_levels(
                prices_array, volumes_array, timestamps, 'support', timeframe, component
            )
            support_levels.extend(volume_support_levels)
            
        except Exception as e:
            self.logger.warning(f"Error detecting support levels: {e}")
        
        return support_levels
    
    def _detect_resistance_levels(self, prices: List[float], volumes: List[float],
                                timestamps: List[datetime], timeframe: int,
                                component: str) -> List[ResistanceLevel]:
        """Detect resistance levels using multiple methods"""
        resistance_levels = []
        
        try:
            prices_array = np.array(prices)
            volumes_array = np.array(volumes)
            
            # Method 1: Local maxima detection
            peaks, _ = self._fast_peak_detection(prices_array)
            peak_indices = np.where(peaks)[0]
            
            # Group nearby peaks into resistance levels
            level_groups = self._group_nearby_levels(
                prices_array[peak_indices], peak_indices, self.level_detection_params['touch_tolerance']
            )
            
            for group_prices, group_indices in level_groups:
                if len(group_prices) >= self.level_detection_params['min_touches']:
                    level_price = np.mean(group_prices)
                    
                    # Calculate level strength
                    strength = self._calculate_level_strength(
                        level_price, group_prices, volumes_array[group_indices], 'resistance'
                    )
                    
                    if strength >= self.level_detection_params['min_strength']:
                        resistance_level = ResistanceLevel(
                            level=level_price,
                            level_type='resistance',
                            strength=strength,
                            touch_count=len(group_prices),
                            first_touch=timestamps[group_indices[0]],
                            last_touch=timestamps[group_indices[-1]],
                            volume_profile={},
                            timeframes_active=[timeframe],
                            components_involved=[component],
                            confidence=min(0.9, strength + 0.1),
                            pattern_relevance=self._calculate_pattern_relevance(level_price, component)
                        )
                        resistance_levels.append(resistance_level)
            
            # Method 2: Volume-based resistance detection
            volume_resistance_levels = self._detect_volume_based_levels(
                prices_array, volumes_array, timestamps, 'resistance', timeframe, component
            )
            resistance_levels.extend(volume_resistance_levels)
            
        except Exception as e:
            self.logger.warning(f"Error detecting resistance levels: {e}")
        
        return resistance_levels
    
    def _detect_pivot_levels(self, highs: List[float], lows: List[float], 
                           prices: List[float], volumes: List[float],
                           timestamps: List[datetime], timeframe: int,
                           component: str) -> List[ResistanceLevel]:
        """Detect pivot point levels"""
        pivot_levels = []
        
        try:
            if len(prices) < 3:
                return pivot_levels
            
            # Calculate standard pivot points from recent high/low data
            recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            recent_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            recent_close = prices[-1]
            
            # Standard pivot calculation
            pivot_point = (recent_high + recent_low + recent_close) / 3
            
            # Support and resistance levels
            r1 = 2 * pivot_point - recent_low
            s1 = 2 * pivot_point - recent_high
            r2 = pivot_point + (recent_high - recent_low)
            s2 = pivot_point - (recent_high - recent_low)
            
            # Create pivot levels
            pivot_data = [
                (pivot_point, 'pivot'),
                (r1, 'resistance'),
                (s1, 'support'),
                (r2, 'resistance'),
                (s2, 'support')
            ]
            
            for level_price, level_type in pivot_data:
                if level_price > 0:
                    # Calculate strength based on historical price interaction
                    strength = self._calculate_pivot_level_strength(level_price, prices, volumes)
                    
                    if strength >= self.level_detection_params['min_strength']:
                        pivot_level = ResistanceLevel(
                            level=level_price,
                            level_type=level_type,
                            strength=strength,
                            touch_count=1,  # New pivot level
                            first_touch=timestamps[-1] if timestamps else datetime.now(),
                            last_touch=timestamps[-1] if timestamps else datetime.now(),
                            volume_profile={},
                            timeframes_active=[timeframe],
                            components_involved=[component],
                            confidence=0.7,  # Standard confidence for pivot levels
                            pattern_relevance=self._calculate_pattern_relevance(level_price, component)
                        )
                        pivot_levels.append(pivot_level)
            
        except Exception as e:
            self.logger.warning(f"Error detecting pivot levels: {e}")
        
        return pivot_levels
    
    def _group_nearby_levels(self, prices: np.ndarray, indices: np.ndarray, 
                           tolerance: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Group nearby price levels together"""
        if len(prices) == 0:
            return []
        
        groups = []
        sorted_indices = np.argsort(prices)
        sorted_prices = prices[sorted_indices]
        sorted_original_indices = indices[sorted_indices]
        
        current_group_prices = [sorted_prices[0]]
        current_group_indices = [sorted_original_indices[0]]
        
        for i in range(1, len(sorted_prices)):
            price_diff = abs(sorted_prices[i] - sorted_prices[i-1]) / sorted_prices[i-1]
            
            if price_diff <= tolerance:
                # Add to current group
                current_group_prices.append(sorted_prices[i])
                current_group_indices.append(sorted_original_indices[i])
            else:
                # Start new group
                if len(current_group_prices) > 0:
                    groups.append((np.array(current_group_prices), np.array(current_group_indices)))
                current_group_prices = [sorted_prices[i]]
                current_group_indices = [sorted_original_indices[i]]
        
        # Add last group
        if len(current_group_prices) > 0:
            groups.append((np.array(current_group_prices), np.array(current_group_indices)))
        
        return groups
    
    def _calculate_level_strength(self, level_price: float, touch_prices: np.ndarray,
                                touch_volumes: np.ndarray, level_type: str) -> float:
        """Calculate strength of a resistance/support level"""
        try:
            # Base strength from number of touches
            touch_strength = min(1.0, len(touch_prices) / 5.0)
            
            # Volume weighting
            avg_volume = np.mean(touch_volumes) if len(touch_volumes) > 0 else 0
            volume_weight = min(1.0, avg_volume / (np.mean(touch_volumes) + 1e-6))
            
            # Price consistency (lower variance = higher strength)
            price_variance = np.var(touch_prices) if len(touch_prices) > 1 else 0
            relative_variance = price_variance / (level_price ** 2) if level_price > 0 else 1
            consistency_strength = max(0, 1 - relative_variance * 100)
            
            # Combine factors
            total_strength = (
                touch_strength * 0.4 + 
                volume_weight * self.level_detection_params['volume_weight'] +
                consistency_strength * 0.3
            )
            
            return min(1.0, total_strength)
            
        except Exception as e:
            self.logger.warning(f"Error calculating level strength: {e}")
            return 0.5
    
    def _calculate_pivot_level_strength(self, level_price: float, prices: List[float],
                                      volumes: List[float]) -> float:
        """Calculate strength of pivot levels based on historical interaction"""
        try:
            prices_array = np.array(prices)
            
            # Find how many times price came close to this level
            tolerance = level_price * self.level_detection_params['touch_tolerance']
            close_touches = np.sum(np.abs(prices_array - level_price) <= tolerance)
            
            # Base strength from touches
            touch_strength = min(1.0, close_touches / 3.0)
            
            # Volume at touches
            volumes_array = np.array(volumes) if volumes else np.ones(len(prices))
            touch_mask = np.abs(prices_array - level_price) <= tolerance
            avg_touch_volume = np.mean(volumes_array[touch_mask]) if np.any(touch_mask) else 0
            avg_total_volume = np.mean(volumes_array) if len(volumes_array) > 0 else 1
            
            volume_strength = min(1.0, avg_touch_volume / (avg_total_volume + 1e-6))
            
            # Combine strengths
            total_strength = touch_strength * 0.6 + volume_strength * 0.4
            
            return max(0.3, min(1.0, total_strength))  # Minimum 0.3 for pivot levels
            
        except Exception as e:
            self.logger.warning(f"Error calculating pivot strength: {e}")
            return 0.5
    
    def _calculate_pattern_relevance(self, level_price: float, component: str) -> float:
        """Calculate relevance of level for pattern detection"""
        try:
            # Base relevance
            relevance = 0.5
            
            # Higher relevance for straddle components
            if 'STRADDLE' in component:
                relevance += 0.2
            
            # Higher relevance for combined triple
            if 'COMBINED' in component:
                relevance += 0.3
            
            # Price-based relevance (levels at round numbers are more relevant)
            if level_price > 0:
                # Check if close to round numbers
                round_distances = [
                    abs(level_price - round(level_price, -1)),  # Nearest 10
                    abs(level_price - round(level_price, -2)),  # Nearest 100
                    abs(level_price - round(level_price / 50) * 50),  # Nearest 50
                    abs(level_price - round(level_price / 25) * 25)   # Nearest 25
                ]
                
                min_round_distance = min(round_distances)
                if min_round_distance / level_price < 0.01:  # Within 1%
                    relevance += 0.1
            
            return min(1.0, relevance)
            
        except Exception as e:
            self.logger.warning(f"Error calculating pattern relevance: {e}")
            return 0.5
    
    def _detect_volume_based_levels(self, prices: np.ndarray, volumes: np.ndarray,
                                  timestamps: List[datetime], level_type: str,
                                  timeframe: int, component: str) -> List[ResistanceLevel]:
        """Detect levels based on volume profile analysis"""
        levels = []
        
        try:
            if len(prices) < 10 or len(volumes) < 10:
                return levels
            
            # Create price-volume bins
            price_min, price_max = np.min(prices), np.max(prices)
            n_bins = min(20, len(prices) // 3)
            
            if price_min == price_max:
                return levels
            
            bin_edges = np.linspace(price_min, price_max, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate volume in each price bin
            bin_volumes = np.zeros(n_bins)
            
            for i in range(len(prices)):
                bin_idx = np.searchsorted(bin_edges[1:], prices[i])
                bin_idx = min(bin_idx, n_bins - 1)
                bin_volumes[bin_idx] += volumes[i]
            
            # Find high volume areas
            volume_threshold = np.percentile(bin_volumes, 75)  # Top 25% volume
            
            for i, volume in enumerate(bin_volumes):
                if volume > volume_threshold:
                    level_price = bin_centers[i]
                    
                    # Calculate strength based on volume
                    strength = min(1.0, volume / np.max(bin_volumes))
                    
                    if strength >= self.level_detection_params['min_strength']:
                        volume_level = ResistanceLevel(
                            level=level_price,
                            level_type=level_type,
                            strength=strength,
                            touch_count=1,
                            first_touch=timestamps[0] if timestamps else datetime.now(),
                            last_touch=timestamps[-1] if timestamps else datetime.now(),
                            volume_profile={'total_volume': volume, 'bin_volume': volume},
                            timeframes_active=[timeframe],
                            components_involved=[component],
                            confidence=strength * 0.8,  # Slightly lower confidence for volume levels
                            pattern_relevance=self._calculate_pattern_relevance(level_price, component)
                        )
                        levels.append(volume_level)
            
        except Exception as e:
            self.logger.warning(f"Error detecting volume-based levels: {e}")
        
        return levels
    
    def _consolidate_levels(self, levels: List[ResistanceLevel]) -> List[ResistanceLevel]:
        """Consolidate similar levels and remove duplicates"""
        if not levels:
            return []
        
        # Sort by strength (highest first)
        sorted_levels = sorted(levels, key=lambda x: x.strength, reverse=True)
        consolidated = []
        
        for level in sorted_levels:
            # Check if similar level already exists
            similar_exists = False
            
            for existing in consolidated:
                price_diff = abs(level.level - existing.level) / existing.level
                if price_diff < self.level_detection_params['confluence_distance']:
                    # Merge with existing level (keep stronger one)
                    if level.strength > existing.strength:
                        # Replace with stronger level
                        consolidated.remove(existing)
                        consolidated.append(level)
                    similar_exists = True
                    break
            
            if not similar_exists:
                consolidated.append(level)
        
        # Limit number of levels
        max_levels = self.config.get('max_levels_per_component', 10)
        return consolidated[:max_levels]
    
    def _find_nearest_level(self, current_price: float, levels: List[ResistanceLevel],
                          direction: str) -> Optional[ResistanceLevel]:
        """Find nearest level in specified direction"""
        if not levels:
            return None
        
        candidate_levels = []
        
        for level in levels:
            if direction == 'above' and level.level > current_price:
                candidate_levels.append(level)
            elif direction == 'below' and level.level < current_price:
                candidate_levels.append(level)
        
        if not candidate_levels:
            return None
        
        # Return level with minimum distance
        return min(candidate_levels, key=lambda x: abs(x.level - current_price))
    
    def _calculate_level_interactions(self, current_price: float,
                                    support_levels: List[ResistanceLevel],
                                    resistance_levels: List[ResistanceLevel],
                                    pivot_levels: List[ResistanceLevel]) -> Dict[str, Any]:
        """Calculate various level interaction metrics"""
        interactions = {
            'price_level_ratio': 0.5,
            'support_distance': 0.0,
            'resistance_distance': 0.0,
            'level_density': 0.0,
            'channel_width': 0.0
        }
        
        try:
            all_levels = support_levels + resistance_levels + pivot_levels
            
            if not all_levels:
                return interactions
            
            # Calculate distances to nearest levels
            nearest_support = self._find_nearest_level(current_price, support_levels, 'below')
            nearest_resistance = self._find_nearest_level(current_price, resistance_levels, 'above')
            
            if nearest_support:
                interactions['support_distance'] = abs(current_price - nearest_support.level) / current_price
            
            if nearest_resistance:
                interactions['resistance_distance'] = abs(nearest_resistance.level - current_price) / current_price
            
            # Calculate position within channel
            if nearest_support and nearest_resistance:
                channel_width = nearest_resistance.level - nearest_support.level
                position_in_channel = (current_price - nearest_support.level) / channel_width
                interactions['price_level_ratio'] = position_in_channel
                interactions['channel_width'] = channel_width / current_price
            
            # Level density (levels per price range)
            level_prices = [level.level for level in all_levels]
            price_range = max(level_prices) - min(level_prices) if level_prices else 1
            interactions['level_density'] = len(all_levels) / (price_range / current_price)
            
        except Exception as e:
            self.logger.warning(f"Error calculating level interactions: {e}")
        
        return interactions
    
    def _detect_confluence_zones(self, component_results: Dict[str, ComponentResistanceResult],
                               timestamp: datetime) -> List[Dict[str, Any]]:
        """Detect confluence zones where multiple components have levels"""
        confluence_zones = []
        
        try:
            # Collect all levels from all components
            all_levels_by_price = {}
            
            for component, result in component_results.items():
                all_levels = result.support_levels + result.resistance_levels + result.pivot_levels
                
                for level in all_levels:
                    price_key = round(level.level, 2)  # Group by price (2 decimal places)
                    
                    if price_key not in all_levels_by_price:
                        all_levels_by_price[price_key] = []
                    
                    all_levels_by_price[price_key].append((component, level))
            
            # Find confluence zones (multiple components at same price level)
            for price, component_levels in all_levels_by_price.items():
                if len(component_levels) >= 2:  # At least 2 components
                    components_involved = [comp for comp, _ in component_levels]
                    levels_involved = [level for _, level in component_levels]
                    
                    # Calculate confluence strength
                    avg_strength = np.mean([level.strength for level in levels_involved])
                    confluence_strength = avg_strength * (len(component_levels) / len(self.all_components))
                    
                    confluence_zone = {
                        'price_level': price,
                        'components_involved': components_involved,
                        'levels_involved': levels_involved,
                        'confluence_strength': confluence_strength,
                        'total_components': len(component_levels),
                        'level_types': [level.level_type for level in levels_involved],
                        'timestamp': timestamp
                    }
                    
                    confluence_zones.append(confluence_zone)
            
            # Sort by confluence strength
            confluence_zones.sort(key=lambda x: x['confluence_strength'], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error detecting confluence zones: {e}")
        
        return confluence_zones
    
    def _generate_breakout_signals(self, component_results: Dict[str, ComponentResistanceResult],
                                 confluence_zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate breakout signals based on resistance analysis"""
        breakout_signals = []
        
        try:
            for component, result in component_results.items():
                if result.breakout_probability > 0.7:  # High breakout probability
                    
                    # Check for confluence support
                    confluence_support = any(
                        component in zone['components_involved'] and zone['confluence_strength'] > 0.7
                        for zone in confluence_zones
                    )
                    
                    signal = {
                        'component': component,
                        'signal_type': 'breakout',
                        'current_price': result.current_price,
                        'target_level': result.nearest_resistance.level if result.nearest_resistance else 0,
                        'probability': result.breakout_probability,
                        'confidence': result.nearest_resistance.confidence if result.nearest_resistance else 0.5,
                        'confluence_support': confluence_support,
                        'timestamp': result.timestamp,
                        'trend_direction': result.trend_direction
                    }
                    
                    breakout_signals.append(signal)
            
        except Exception as e:
            self.logger.warning(f"Error generating breakout signals: {e}")
        
        return breakout_signals
    
    def _generate_reversal_signals(self, component_results: Dict[str, ComponentResistanceResult],
                                 confluence_zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate reversal signals based on resistance analysis"""
        reversal_signals = []
        
        try:
            for component, result in component_results.items():
                if result.reversal_probability > 0.7:  # High reversal probability
                    
                    # Determine reversal direction
                    if result.nearest_resistance and abs(result.current_price - result.nearest_resistance.level) / result.current_price < 0.01:
                        reversal_type = 'resistance_reversal'
                        target_level = result.nearest_support.level if result.nearest_support else result.current_price * 0.95
                    elif result.nearest_support and abs(result.current_price - result.nearest_support.level) / result.current_price < 0.01:
                        reversal_type = 'support_reversal'
                        target_level = result.nearest_resistance.level if result.nearest_resistance else result.current_price * 1.05
                    else:
                        continue
                    
                    signal = {
                        'component': component,
                        'signal_type': 'reversal',
                        'reversal_type': reversal_type,
                        'current_price': result.current_price,
                        'target_level': target_level,
                        'probability': result.reversal_probability,
                        'confidence': 0.8,
                        'timestamp': result.timestamp
                    }
                    
                    reversal_signals.append(signal)
            
        except Exception as e:
            self.logger.warning(f"Error generating reversal signals: {e}")
        
        return reversal_signals
    
    def _calculate_breakout_probability(self, current_price: float, 
                                       resistance_levels: List[ResistanceLevel],
                                       timeframe_results: Dict[int, Dict]) -> float:
        """Calculate probability of breakout"""
        if not resistance_levels:
            return 0.5
        
        nearest_resistance = self._find_nearest_level(current_price, resistance_levels, 'above')
        if not nearest_resistance:
            return 0.3  # Low probability if no resistance above
        
        # Distance to resistance
        distance_ratio = abs(nearest_resistance.level - current_price) / current_price
        
        # Volume analysis
        volume_factor = 1.0
        if hasattr(nearest_resistance, 'volume_profile') and nearest_resistance.volume_profile:
            volume_factor = min(2.0, nearest_resistance.volume_profile.get('total_volume', 1) / 1000)
        
        # Strength factor (weaker resistance = higher breakout probability)
        strength_factor = 1.0 - nearest_resistance.strength
        
        # Calculate probability
        breakout_prob = (
            (1.0 - min(1.0, distance_ratio * 100)) * 0.4 +  # Closer = higher prob
            volume_factor * 0.3 +  # Higher volume = higher prob
            strength_factor * 0.3   # Weaker resistance = higher prob
        )
        
        return min(0.95, max(0.05, breakout_prob))
    
    def _calculate_reversal_probability(self, current_price: float,
                                      support_levels: List[ResistanceLevel],
                                      resistance_levels: List[ResistanceLevel],
                                      timeframe_results: Dict[int, Dict]) -> float:
        """Calculate probability of reversal"""
        # Check if near support or resistance
        nearest_support = self._find_nearest_level(current_price, support_levels, 'below')
        nearest_resistance = self._find_nearest_level(current_price, resistance_levels, 'above')
        
        reversal_factors = []
        
        # Near support level
        if nearest_support:
            support_distance = abs(current_price - nearest_support.level) / current_price
            if support_distance < 0.02:  # Within 2%
                reversal_factors.append(nearest_support.strength * 0.8)
        
        # Near resistance level
        if nearest_resistance:
            resistance_distance = abs(nearest_resistance.level - current_price) / current_price
            if resistance_distance < 0.02:  # Within 2%
                reversal_factors.append(nearest_resistance.strength * 0.8)
        
        # Multi-timeframe confirmation
        if len(timeframe_results) > 1:
            reversal_factors.append(0.2)  # Bonus for multi-timeframe analysis
        
        return np.mean(reversal_factors) if reversal_factors else 0.3
    
    def _calculate_consolidation_score(self, current_price: float,
                                     support_levels: List[ResistanceLevel],
                                     resistance_levels: List[ResistanceLevel]) -> float:
        """Calculate consolidation score"""
        if not support_levels or not resistance_levels:
            return 0.2
        
        nearest_support = self._find_nearest_level(current_price, support_levels, 'below')
        nearest_resistance = self._find_nearest_level(current_price, resistance_levels, 'above')
        
        if not nearest_support or not nearest_resistance:
            return 0.2
        
        # Channel width
        channel_width = abs(nearest_resistance.level - nearest_support.level) / current_price
        
        # Position in channel
        channel_position = (current_price - nearest_support.level) / (nearest_resistance.level - nearest_support.level)
        
        # Level strength
        avg_strength = (nearest_support.strength + nearest_resistance.strength) / 2
        
        # Narrow channel with strong levels = high consolidation score
        consolidation_score = (
            avg_strength * 0.5 +
            (1.0 - min(1.0, channel_width * 20)) * 0.3 +  # Narrow channel bonus
            (0.5 - abs(channel_position - 0.5)) * 0.2  # Middle position bonus
        )
        
        return min(1.0, max(0.0, consolidation_score))
    
    def _determine_trend_direction(self, current_price: float,
                                 support_levels: List[ResistanceLevel],
                                 resistance_levels: List[ResistanceLevel],
                                 timeframe_results: Dict[int, Dict]) -> str:
        """Determine trend direction"""
        if not support_levels and not resistance_levels:
            return 'NEUTRAL'
        
        # Count levels above and below current price
        levels_above = len([r for r in resistance_levels if r.level > current_price])
        levels_below = len([s for s in support_levels if s.level < current_price])
        
        # Strength of levels
        strength_above = sum([r.strength for r in resistance_levels if r.level > current_price])
        strength_below = sum([s.strength for s in support_levels if s.level < current_price])
        
        # Determine trend
        if strength_above > strength_below * 1.5:
            return 'BEARISH'
        elif strength_below > strength_above * 1.5:
            return 'BULLISH'
        elif levels_above > levels_below + 2:
            return 'BEARISH'
        elif levels_below > levels_above + 2:
            return 'BULLISH'
        else:
            return 'NEUTRAL'
    
    def _calculate_timeframe_consistency(self, timeframe_results: Dict[int, Dict]) -> Dict[int, float]:
        """Calculate consistency across timeframes"""
        consistency = {}
        
        if len(timeframe_results) < 2:
            return {tf: 1.0 for tf in timeframe_results.keys()}
        
        # Compare each timeframe with others
        for tf1 in timeframe_results.keys():
            consistency_scores = []
            
            for tf2 in timeframe_results.keys():
                if tf1 != tf2:
                    result1 = timeframe_results[tf1]
                    result2 = timeframe_results[tf2]
                    
                    # Compare number of levels
                    levels1 = len(result1.get('support_levels', [])) + len(result1.get('resistance_levels', []))
                    levels2 = len(result2.get('support_levels', [])) + len(result2.get('resistance_levels', []))
                    
                    level_consistency = 1.0 - abs(levels1 - levels2) / max(levels1 + levels2, 1)
                    consistency_scores.append(level_consistency)
            
            consistency[tf1] = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return consistency
    
    def _find_cross_timeframe_levels(self, timeframe_results: Dict[int, Dict]) -> List[ResistanceLevel]:
        """Find levels that appear across multiple timeframes"""
        cross_tf_levels = []
        
        if len(timeframe_results) < 2:
            return cross_tf_levels
        
        # Collect all levels from all timeframes
        all_levels = []
        for tf, result in timeframe_results.items():
            tf_levels = result.get('support_levels', []) + result.get('resistance_levels', [])
            for level in tf_levels:
                level.timeframes_active = [tf]  # Set timeframe
                all_levels.append(level)
        
        # Find levels that appear in multiple timeframes
        level_groups = {}
        tolerance = 0.005  # 0.5% tolerance
        
        for level in all_levels:
            price_key = round(level.level / (level.level * tolerance)) * (level.level * tolerance)
            
            if price_key not in level_groups:
                level_groups[price_key] = []
            level_groups[price_key].append(level)
        
        # Keep groups with multiple timeframes
        for group in level_groups.values():
            if len(group) >= 2:
                # Merge levels from different timeframes
                merged_level = group[0]  # Start with first level
                merged_level.timeframes_active = list(set([tf for level in group for tf in level.timeframes_active]))
                merged_level.strength = np.mean([level.strength for level in group])
                merged_level.touch_count = sum([level.touch_count for level in group])
                
                cross_tf_levels.append(merged_level)
        
        return cross_tf_levels
    
    def _detect_multi_component_levels(self, component_results: Dict[str, ComponentResistanceResult]) -> List[ResistanceLevel]:
        """Detect levels that appear across multiple components"""
        multi_comp_levels = []
        
        # Collect all levels from all components
        all_levels = []
        for component, result in component_results.items():
            comp_levels = result.support_levels + result.resistance_levels + result.pivot_levels
            for level in comp_levels:
                if component not in level.components_involved:
                    level.components_involved.append(component)
                all_levels.append(level)
        
        # Group levels by price
        level_groups = {}
        tolerance = 0.005  # 0.5% tolerance
        
        for level in all_levels:
            price_key = round(level.level / (level.level * tolerance)) * (level.level * tolerance)
            
            if price_key not in level_groups:
                level_groups[price_key] = []
            level_groups[price_key].append(level)
        
        # Keep groups with multiple components
        for group in level_groups.values():
            components_in_group = set()
            for level in group:
                components_in_group.update(level.components_involved)
            
            if len(components_in_group) >= 2:
                # Create merged multi-component level
                merged_level = group[0]  # Start with first level
                merged_level.components_involved = list(components_in_group)
                merged_level.strength = np.mean([level.strength for level in group])
                merged_level.confidence = min(0.95, merged_level.strength + 0.1 * len(components_in_group))
                merged_level.pattern_relevance = min(1.0, merged_level.pattern_relevance + 0.1 * len(components_in_group))
                
                multi_comp_levels.append(merged_level)
        
        return multi_comp_levels
    
    def _calculate_level_correlations(self, component_results: Dict[str, ComponentResistanceResult]) -> Dict[str, float]:
        """Calculate correlations between resistance levels"""
        correlations = {}
        
        try:
            # Extract level data for correlation calculation
            component_level_data = {}
            
            for component, result in component_results.items():
                all_levels = result.support_levels + result.resistance_levels
                level_prices = [level.level for level in all_levels]
                component_level_data[component] = level_prices
            
            # Calculate correlations between component level sets
            components = list(component_level_data.keys())
            
            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    levels1 = component_level_data[comp1]
                    levels2 = component_level_data[comp2]
                    
                    if len(levels1) > 0 and len(levels2) > 0:
                        # Simple correlation based on level count similarity
                        level_diff = abs(len(levels1) - len(levels2))
                        max_levels = max(len(levels1), len(levels2))
                        correlation = 1.0 - (level_diff / max_levels) if max_levels > 0 else 0.0
                        
                        correlations[f"{comp1}_{comp2}"] = correlation
            
        except Exception as e:
            self.logger.warning(f"Error calculating level correlations: {e}")
        
        return correlations
    
    def _identify_pattern_relevant_levels(self, component_results: Dict[str, ComponentResistanceResult]) -> List[ResistanceLevel]:
        """Identify levels most relevant for pattern detection"""
        pattern_levels = []
        
        for component, result in component_results.items():
            all_levels = result.support_levels + result.resistance_levels + result.pivot_levels
            
            # Filter levels by pattern relevance threshold
            relevant_levels = [
                level for level in all_levels 
                if level.pattern_relevance >= self.pattern_params['pattern_relevance_threshold']
            ]
            
            pattern_levels.extend(relevant_levels)
        
        # Sort by pattern relevance and strength
        pattern_levels.sort(key=lambda x: (x.pattern_relevance, x.strength), reverse=True)
        
        return pattern_levels[:20]  # Top 20 most relevant levels
    
    def _calculate_pattern_confluence_score(self, pattern_levels: List[ResistanceLevel],
                                          confluence_zones: List[Dict[str, Any]]) -> float:
        """Calculate pattern confluence score"""
        if not pattern_levels or not confluence_zones:
            return 0.0
        
        confluence_score = 0.0
        
        # Score based on pattern levels in confluence zones
        for zone in confluence_zones:
            zone_price = zone['price_level']
            zone_strength = zone['confluence_strength']
            
            # Check if any pattern-relevant levels are in this zone
            levels_in_zone = [
                level for level in pattern_levels
                if abs(level.level - zone_price) / zone_price < 0.01  # Within 1%
            ]
            
            if levels_in_zone:
                zone_pattern_score = zone_strength * len(levels_in_zone) / len(pattern_levels)
                confluence_score += zone_pattern_score
        
        return min(1.0, confluence_score)
    
    def _calculate_resistance_regime_indicators(self, component_results: Dict[str, ComponentResistanceResult]) -> Dict[str, float]:
        """Calculate regime indicators based on resistance analysis"""
        regime_indicators = {}
        
        try:
            # Aggregate metrics across all components
            total_support_levels = 0
            total_resistance_levels = 0
            avg_breakout_prob = []
            avg_reversal_prob = []
            avg_consolidation_score = []
            
            for component, result in component_results.items():
                total_support_levels += len(result.support_levels)
                total_resistance_levels += len(result.resistance_levels)
                avg_breakout_prob.append(result.breakout_probability)
                avg_reversal_prob.append(result.reversal_probability)
                avg_consolidation_score.append(result.consolidation_score)
            
            # Calculate regime indicators
            regime_indicators['level_density'] = (total_support_levels + total_resistance_levels) / len(component_results)
            regime_indicators['breakout_tendency'] = np.mean(avg_breakout_prob) if avg_breakout_prob else 0.5
            regime_indicators['reversal_tendency'] = np.mean(avg_reversal_prob) if avg_reversal_prob else 0.5
            regime_indicators['consolidation_strength'] = np.mean(avg_consolidation_score) if avg_consolidation_score else 0.5
            
            # Support/resistance balance
            total_levels = total_support_levels + total_resistance_levels
            if total_levels > 0:
                regime_indicators['support_resistance_ratio'] = total_support_levels / total_levels
            else:
                regime_indicators['support_resistance_ratio'] = 0.5
            
            # Overall regime strength
            regime_indicators['overall_regime_strength'] = np.mean([
                regime_indicators['level_density'] / 10,  # Normalize
                regime_indicators['consolidation_strength']
            ])
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime indicators: {e}")
            regime_indicators = {'overall_regime_strength': 0.5}
        
        return regime_indicators
    
    def _generate_consolidation_signals(self, component_results: Dict[str, ComponentResistanceResult]) -> List[Dict[str, Any]]:
        """Generate consolidation signals"""
        consolidation_signals = []
        
        try:
            for component, result in component_results.items():
                if result.consolidation_score > 0.7:  # High consolidation score
                    
                    signal = {
                        'component': component,
                        'signal_type': 'consolidation',
                        'current_price': result.current_price,
                        'consolidation_score': result.consolidation_score,
                        'support_level': result.nearest_support.level if result.nearest_support else 0,
                        'resistance_level': result.nearest_resistance.level if result.nearest_resistance else 0,
                        'confidence': result.consolidation_score,
                        'timestamp': result.timestamp
                    }
                    
                    consolidation_signals.append(signal)
            
        except Exception as e:
            self.logger.warning(f"Error generating consolidation signals: {e}")
        
        return consolidation_signals
    
    def _calculate_volume_profile_for_level(self, level_price: float, prices: List[float], volumes: List[float]) -> Dict[str, float]:
        """Calculate volume profile for a specific level"""
        if not prices or not volumes:
            return {}
        
        try:
            tolerance = level_price * 0.005  # 0.5% tolerance
            
            total_volume = 0
            touch_volume = 0
            touch_count = 0
            
            for i, price in enumerate(prices):
                volume = volumes[i] if i < len(volumes) else 0
                total_volume += volume
                
                if abs(price - level_price) <= tolerance:
                    touch_volume += volume
                    touch_count += 1
            
            volume_profile = {
                'total_volume': total_volume,
                'touch_volume': touch_volume,
                'touch_count': touch_count,
                'volume_ratio': touch_volume / total_volume if total_volume > 0 else 0,
                'avg_touch_volume': touch_volume / touch_count if touch_count > 0 else 0
            }
            
            return volume_profile
            
        except Exception as e:
            self.logger.warning(f"Error calculating volume profile: {e}")
            return {}
    
    def _update_resistance_history(self, result: EnhancedResistanceAnalysisResult):
        """Update resistance analysis history"""
        try:
            # Store confluence zones
            self.confluence_zones_history.append({
                'timestamp': result.timestamp,
                'zones': result.confluence_zones,
                'pattern_confluence_score': result.pattern_confluence_score
            })
            
            # Maintain history size (keep last 100 results)
            if len(self.confluence_zones_history) > 100:
                self.confluence_zones_history = self.confluence_zones_history[-100:]
            
            # Update component-specific history
            for component, comp_result in result.component_results.items():
                if component in self.resistance_history:
                    for timeframe in self.timeframes:
                        if timeframe in self.resistance_history[component]:
                            # Store recent levels
                            history = self.resistance_history[component][timeframe]
                            
                            # Keep recent levels only
                            cutoff_time = result.timestamp - timedelta(hours=self.config.get('level_expiry_hours', 24))
                            
                            for level_type in ['support_levels', 'resistance_levels', 'pivot_levels']:
                                if level_type in history:
                                    history[level_type] = [
                                        level for level in history[level_type]
                                        if level.last_touch > cutoff_time
                                    ]
        
        except Exception as e:
            self.logger.warning(f"Error updating resistance history: {e}")
    
    def get_analyzer_summary(self) -> Dict[str, Any]:
        """Get comprehensive analyzer summary"""
        return {
            'total_analyses': self.analysis_count,
            'signals_generated': self.pattern_signals_generated,
            'components_tracked': len(self.all_components),
            'timeframes_analyzed': self.timeframes,
            'level_detection_params': self.level_detection_params,
            'pattern_integration': self.pattern_params,
            'performance_metrics': {
                'avg_signals_per_analysis': self.pattern_signals_generated / max(1, self.analysis_count),
                'analysis_success_rate': 1.0  # Placeholder
            }
        }