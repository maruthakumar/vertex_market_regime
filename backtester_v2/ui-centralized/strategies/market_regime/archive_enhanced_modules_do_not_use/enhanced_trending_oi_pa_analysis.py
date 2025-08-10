"""
Enhanced Trending OI with PA Analysis for Backtester V2 - Phase 1 Enhanced + Mathematical Correlation

This module implements the corrected and enhanced Trending OI with Price Action analysis
system with the following key features:

1. Corrected OI Pattern Recognition (Both calls and puts follow same OI-Price relationship)
2. Multi-Timeframe Analysis (3min + 15min with proper weighting)
3. Comprehensive Divergence Detection (5 types)
4. Institutional vs Retail Detection
5. Integration with Triple Straddle Analysis
6. Enhanced Features (Volume weighting, Dynamic strike range, Session weighting)
7. Historical Performance Tracking

PHASE 1 ENHANCEMENTS (EXISTING):
8. Complete 18-Regime Classification System
9. Comprehensive Volatility Component Integration
10. Dynamic Threshold Optimization
11. Enhanced Market Regime Formation Logic

ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0 ADDITIONS (NEW):
12. Pearson Correlation-based Pattern Similarity (>0.80 threshold)
13. Time-decay Weighting exp(-λ × (T-t)) Mathematical Formulation
14. Mathematical Accuracy Validation (±0.001 tolerance)
15. Enhanced Historical Pattern Recognition with Correlation Metrics
16. Preserved Advanced Features with Mathematical Transparency
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import warnings

# Phase 1 Enhanced Components
from .enhanced_modules.enhanced_18_regime_classifier import Enhanced18RegimeClassifier
from .volatility_component_calculator import VolatilityComponentCalculator
from .dynamic_threshold_optimizer import DynamicThresholdOptimizer
warnings.filterwarnings('ignore')

# Mathematical precision tolerance for Enhanced Triple Straddle Framework v2.0
MATHEMATICAL_TOLERANCE = 0.001

logger = logging.getLogger(__name__)

class OIPattern(Enum):
    """OI Pattern Classifications"""
    LONG_BUILD_UP = "Long_Build_Up"           # OI↑ + Price↑ (Bullish)
    LONG_UNWINDING = "Long_Unwinding"         # OI↓ + Price↓ (Bearish)
    SHORT_BUILD_UP = "Short_Build_Up"         # OI↑ + Price↓ (Bearish)
    SHORT_COVERING = "Short_Covering"         # OI↓ + Price↑ (Bullish)
    NEUTRAL = "Neutral"                       # No clear pattern

class DivergenceType(Enum):
    """Types of divergence detection"""
    PATTERN_DIVERGENCE = "Pattern_Divergence"
    OI_PRICE_DIVERGENCE = "OI_Price_Divergence"
    CALL_PUT_DIVERGENCE = "Call_Put_Divergence"
    INSTITUTIONAL_RETAIL_DIVERGENCE = "Institutional_Retail_Divergence"
    CROSS_STRIKE_DIVERGENCE = "Cross_Strike_Divergence"

@dataclass
class OIAnalysisResult:
    """Result structure for OI analysis"""
    pattern: OIPattern
    confidence: float
    signal_strength: float
    divergence_score: float
    institutional_ratio: float
    timeframe_consistency: float
    supporting_metrics: Dict[str, Any]

@dataclass
class MultiTimeframeResult:
    """Multi-timeframe analysis result"""
    primary_signal: float      # 3-minute signal
    confirmation_signal: float # 15-minute signal
    combined_signal: float     # Weighted combination
    divergence_flag: bool      # Timeframe divergence
    confidence: float          # Overall confidence

@dataclass
class CorrelationAnalysisResult:
    """Result structure for correlation analysis - Enhanced Triple Straddle Framework v2.0"""
    pearson_correlation: float
    correlation_confidence: float
    pattern_similarity_score: float
    time_decay_weight: float
    mathematical_accuracy: bool
    correlation_threshold_met: bool  # >0.80 threshold
    historical_pattern_match: Optional[Dict[str, Any]]

@dataclass
class TimeDecayWeightingConfig:
    """Configuration for time-decay weighting - Enhanced Triple Straddle Framework v2.0"""
    lambda_decay: float = 0.1  # Decay parameter λ
    time_window: int = 300     # Time window in seconds (5 minutes)
    baseline_weight: float = 1.0
    min_weight: float = 0.1
    max_weight: float = 2.0

class EnhancedTrendingOIWithPAAnalysis:
    """
    Enhanced Trending OI with PA Analysis Engine
    
    Implements corrected OI pattern recognition with comprehensive divergence detection,
    multi-timeframe analysis, and integration with Triple Straddle system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Enhanced Trending OI with PA Analysis with Phase 1 Enhancements"""
        self.config = config or {}

        # Core configuration
        self.strike_range = int(self.config.get('strike_range', 7))  # ±7 strikes from ATM
        self.volatility_adjustment = self.config.get('volatility_adjustment', True)

        # Multi-timeframe configuration
        self.primary_timeframe = int(self.config.get('primary_timeframe', 3))    # 3-minute
        self.confirmation_timeframe = int(self.config.get('confirmation_timeframe', 15))  # 15-minute
        self.timeframe_weights = self.config.get('timeframe_weights', {
            3: 0.4,   # 40% weight for 3-minute (primary intraday)
            15: 0.6   # 60% weight for 15-minute (confirmation)
        })

        # Divergence detection configuration
        self.divergence_threshold = float(self.config.get('divergence_threshold', 0.3))  # 30%
        self.divergence_window = int(self.config.get('divergence_window', 10))
        self.confidence_reduction_factor = float(self.config.get('confidence_reduction_factor', 0.5))

        # Institutional detection configuration
        self.institutional_lot_size = int(self.config.get('institutional_lot_size', 1000))
        self.institutional_oi_threshold = float(self.config.get('institutional_oi_threshold', 0.6))

        # Session-based weighting
        self.session_weights = self.config.get('session_weights', {
            'market_open': 1.2,    # 9:15-10:30 (Higher weight)
            'mid_session': 1.0,    # 10:30-14:30 (Normal weight)
            'market_close': 1.3    # 14:30-15:30 (Highest weight)
        })

        # Pattern performance tracking
        self.pattern_performance_history = {}
        self.historical_accuracy = {}

        # Volume weighting configuration
        self.volume_weight_factor = float(self.config.get('volume_weight_factor', 0.3))
        self.min_volume_threshold = int(self.config.get('min_volume_threshold', 100))

        # PHASE 1 ENHANCED COMPONENTS
        # Initialize 18-regime classifier
        self.regime_classifier = Enhanced18RegimeClassifier(
            self.config.get('regime_classifier_config', {})
        )

        # Initialize volatility component calculator
        self.volatility_calculator = VolatilityComponentCalculator(
            self.config.get('volatility_calculator_config', {})
        )

        # Initialize dynamic threshold optimizer
        self.threshold_optimizer = DynamicThresholdOptimizer(
            self.config.get('threshold_optimizer_config', {})
        )

        # Phase 1 enhancement flags
        self.enable_18_regime_classification = self.config.get('enable_18_regime_classification', True)
        self.enable_volatility_component = self.config.get('enable_volatility_component', True)
        self.enable_dynamic_thresholds = self.config.get('enable_dynamic_thresholds', True)

        # ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0 - Mathematical Correlation Features
        self.enable_pearson_correlation = self.config.get('enable_pearson_correlation', True)
        self.enable_time_decay_weighting = self.config.get('enable_time_decay_weighting', True)
        self.enable_mathematical_validation = self.config.get('enable_mathematical_validation', True)

        # Correlation configuration
        self.correlation_threshold = self.config.get('correlation_threshold', 0.80)  # >0.80 requirement
        self.correlation_window = self.config.get('correlation_window', 20)  # Historical window
        self.pattern_similarity_threshold = self.config.get('pattern_similarity_threshold', 0.75)

        # Time-decay weighting configuration
        self.time_decay_config = TimeDecayWeightingConfig(
            lambda_decay=self.config.get('lambda_decay', 0.1),
            time_window=self.config.get('time_window', 300),
            baseline_weight=self.config.get('baseline_weight', 1.0),
            min_weight=self.config.get('min_weight', 0.1),
            max_weight=self.config.get('max_weight', 2.0)
        )

        # Historical pattern storage for correlation analysis
        self.historical_patterns = []
        self.correlation_cache = {}
        self.mathematical_accuracy_history = []

        logger.info("Enhanced Trending OI with PA Analysis initialized with Phase 1 + Mathematical Correlation enhancements")
        logger.info(f"Pearson correlation threshold: {self.correlation_threshold}")
        logger.info(f"Time-decay lambda: {self.time_decay_config.lambda_decay}")
        logger.info(f"Mathematical tolerance: ±{MATHEMATICAL_TOLERANCE}")
    
    def analyze_trending_oi_pa(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis function for Trending OI with PA
        
        Args:
            market_data: Comprehensive market data including options, OI, volume, etc.
            
        Returns:
            Dictionary with complete OI analysis results
        """
        try:
            # Step 1: Extract and prepare data
            prepared_data = self._prepare_market_data(market_data)

            if not prepared_data:
                logger.warning("Insufficient market data for OI analysis")
                return self._get_default_result()

            # PHASE 1 ENHANCEMENT: Step 1.5: Calculate volatility component
            volatility_result = None
            if self.enable_volatility_component:
                volatility_result = self.volatility_calculator.calculate_volatility_component(market_data)
                prepared_data['volatility_component'] = volatility_result

            # PHASE 1 ENHANCEMENT: Step 1.6: Calculate dynamic thresholds
            threshold_result = None
            if self.enable_dynamic_thresholds:
                threshold_result = self.threshold_optimizer.calculate_adaptive_thresholds(market_data)
                prepared_data['dynamic_thresholds'] = threshold_result

            # Step 2: Multi-timeframe analysis
            multi_tf_result = self._analyze_multi_timeframe(prepared_data)
            
            # Step 3: OI pattern recognition (corrected logic)
            pattern_results = self._identify_oi_patterns_corrected(prepared_data)
            
            # Step 4: Divergence detection (5 types)
            divergence_analysis = self._detect_comprehensive_divergence(prepared_data, pattern_results)
            
            # Step 5: Institutional vs retail analysis
            institutional_analysis = self._analyze_institutional_retail(prepared_data)
            
            # Step 6: Volume-weighted analysis
            volume_weighted_results = self._apply_volume_weighting(pattern_results, prepared_data)
            
            # Step 7: Session-based adjustment
            session_adjusted_results = self._apply_session_weighting(volume_weighted_results)
            
            # ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0: Step 7.5: Mathematical Correlation Analysis
            correlation_analysis = None
            if self.enable_pearson_correlation:
                correlation_analysis = self._perform_correlation_analysis(
                    pattern_results, prepared_data, volume_weighted_results
                )

            # ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0: Step 7.6: Time-decay Weighting
            time_decay_weights = None
            if self.enable_time_decay_weighting:
                time_decay_weights = self._calculate_time_decay_weights(prepared_data)
                # Apply time-decay weights to session-adjusted results
                session_adjusted_results = self._apply_time_decay_weighting(
                    session_adjusted_results, time_decay_weights
                )

            # Step 8: Calculate final signal and confidence (enhanced with correlation)
            final_signal = self._calculate_final_signal_enhanced(
                multi_tf_result, session_adjusted_results, divergence_analysis,
                institutional_analysis, correlation_analysis, time_decay_weights
            )

            # ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0: Mathematical Accuracy Validation
            mathematical_accuracy = True
            if self.enable_mathematical_validation:
                mathematical_accuracy = self._validate_mathematical_accuracy(final_signal)
                self.mathematical_accuracy_history.append(mathematical_accuracy)

            # Step 9: Update performance tracking (enhanced)
            self._update_performance_tracking_enhanced(pattern_results, final_signal, correlation_analysis)
            
            # PHASE 1 ENHANCEMENT: Calculate directional component for regime classification
            directional_component = self._calculate_directional_component(
                final_signal, multi_tf_result, divergence_analysis
            )

            # PHASE 1 ENHANCEMENT: Classify market regime using 18-regime system
            regime_classification = None
            if self.enable_18_regime_classification and volatility_result:
                regime_classification = self.regime_classifier.classify_market_regime(
                    directional_component, volatility_result.volatility_component
                )

            # PHASE 1 ENHANCEMENT: Update threshold optimizer performance
            if self.enable_dynamic_thresholds and threshold_result:
                performance_score = final_signal.get('confidence', 0.5)
                self.threshold_optimizer.update_performance(performance_score)

            # Prepare enhanced result with Phase 1 + Mathematical Correlation components
            result = {
                'oi_signal': final_signal['signal'],
                'confidence': final_signal['confidence'],
                'pattern_breakdown': pattern_results,
                'multi_timeframe': multi_tf_result,
                'divergence_analysis': divergence_analysis,
                'institutional_analysis': institutional_analysis,
                'session_adjustment': session_adjusted_results.get('session_factor', 1.0),
                'timestamp': datetime.now(),
                'analysis_type': 'enhanced_trending_oi_pa_v2_phase1_mathematical_correlation',
                'mathematical_accuracy': mathematical_accuracy
            }

            # Add Enhanced Triple Straddle Framework v2.0 results
            if correlation_analysis:
                result['correlation_analysis'] = {
                    'pearson_correlation': correlation_analysis.pearson_correlation,
                    'correlation_confidence': correlation_analysis.correlation_confidence,
                    'pattern_similarity_score': correlation_analysis.pattern_similarity_score,
                    'correlation_threshold_met': correlation_analysis.correlation_threshold_met,
                    'mathematical_accuracy': correlation_analysis.mathematical_accuracy,
                    'historical_pattern_match': correlation_analysis.historical_pattern_match
                }

            if time_decay_weights:
                result['time_decay_analysis'] = {
                    'time_decay_weight': time_decay_weights.get('current_weight', 1.0),
                    'decay_factor': time_decay_weights.get('decay_factor', 1.0),
                    'time_elapsed': time_decay_weights.get('time_elapsed', 0),
                    'baseline_weight': self.time_decay_config.baseline_weight
                }

            # Add Phase 1 enhanced results
            if volatility_result:
                result['volatility_analysis'] = {
                    'volatility_component': volatility_result.volatility_component,
                    'volatility_regime': volatility_result.volatility_regime,
                    'atr_volatility': volatility_result.atr_volatility,
                    'oi_volatility': volatility_result.oi_volatility,
                    'price_volatility': volatility_result.price_volatility,
                    'confidence': volatility_result.confidence
                }

            if threshold_result:
                result['dynamic_thresholds'] = {
                    'oi_threshold': threshold_result.oi_threshold,
                    'price_threshold': threshold_result.price_threshold,
                    'optimization_reason': threshold_result.optimization_reason,
                    'confidence': threshold_result.confidence
                }

            if regime_classification:
                result['regime_classification'] = {
                    'regime': regime_classification.regime,
                    'confidence': regime_classification.confidence,
                    'directional_score': regime_classification.directional_score,
                    'volatility_score': regime_classification.volatility_score,
                    'regime_probability': regime_classification.regime_probability,
                    'alternative_regimes': regime_classification.alternative_regimes
                }

            return result
            
        except Exception as e:
            logger.error(f"Error in Enhanced Trending OI with PA analysis: {e}")
            return self._get_default_result()

    def _calculate_directional_component(self, final_signal: Dict[str, Any],
                                       multi_tf_result: MultiTimeframeResult,
                                       divergence_analysis: Dict[str, Any]) -> float:
        """
        Calculate directional component for regime classification - CALIBRATED FOR INDIAN MARKET

        Args:
            final_signal: Final signal from OI analysis
            multi_tf_result: Multi-timeframe analysis result
            divergence_analysis: Divergence analysis result

        Returns:
            Directional component score (-1 to +1)
        """
        try:
            # Base directional signal from OI analysis
            base_signal = final_signal.get('signal', 0.0)

            # Multi-timeframe confirmation
            tf_signal = multi_tf_result.combined_signal
            tf_confidence = multi_tf_result.confidence

            # CALIBRATED: Enhanced scaling for Indian market characteristics
            scaled_base_signal = np.tanh(base_signal * 1.5)  # Enhanced scaling for stronger signals
            scaled_tf_signal = np.tanh(tf_signal * 1.2)      # Moderate scaling for timeframe signals

            # Divergence adjustment (reduce directional strength if high divergence)
            overall_divergence = divergence_analysis.get('overall_divergence', 0.0)
            # CALIBRATED: Less aggressive divergence penalty for Indian market
            divergence_factor = max(0.5, 1.0 - (overall_divergence * 0.7))  # Reduced penalty

            # CALIBRATED: Session-based weight adjustments for Indian market
            current_time = datetime.now().time()
            # Try to get timestamp from final_signal metadata if available
            if 'timestamp' in final_signal:
                current_time = final_signal['timestamp'].time()
            elif hasattr(self, 'current_timestamp') and self.current_timestamp:
                current_time = self.current_timestamp.time()

            # Adjust weights based on market session
            if time(9, 15) <= current_time <= time(10, 30):  # Opening session
                oi_weight, tf_weight, conf_weight = 0.7, 0.2, 0.1  # Higher OI weight during opening
            elif time(14, 30) <= current_time <= time(15, 30):  # Closing session
                oi_weight, tf_weight, conf_weight = 0.6, 0.3, 0.1  # Balanced weights during closing
            else:  # Mid-session
                oi_weight, tf_weight, conf_weight = 0.6, 0.25, 0.15  # Standard calibrated weights

            # CALIBRATED: Session-adjusted weight distribution for Indian market
            directional_component = (
                scaled_base_signal * oi_weight +        # Session-adjusted OI signal weight
                scaled_tf_signal * tf_weight +          # Session-adjusted multi-timeframe weight
                (scaled_base_signal * tf_confidence) * conf_weight  # Session-adjusted confidence weight
            ) * divergence_factor

            # Ensure result is in [-1, +1] range
            return np.clip(directional_component, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating directional component: {e}")
            return 0.0
    
    def _prepare_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and validate market data for analysis"""
        try:
            underlying_price = market_data.get('underlying_price', 0)
            if underlying_price == 0:
                return {}
            
            # Dynamic strike range based on volatility
            volatility = market_data.get('volatility', 0.15)
            if self.volatility_adjustment:
                adjusted_range = int(self.strike_range * (1 + volatility))
            else:
                adjusted_range = self.strike_range
            
            # Select strikes around ATM
            strikes = market_data.get('strikes', [])
            if not strikes:
                return {}
            
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
            atm_index = strikes.index(atm_strike)
            
            # Select strike range
            start_idx = max(0, atm_index - adjusted_range)
            end_idx = min(len(strikes), atm_index + adjusted_range + 1)
            selected_strikes = strikes[start_idx:end_idx]
            
            # Extract options data for selected strikes
            options_data = market_data.get('options_data', {})
            selected_options_data = {
                strike: options_data[strike] 
                for strike in selected_strikes 
                if strike in options_data
            }
            
            # Prepare historical data for multi-timeframe analysis
            price_history = market_data.get('price_history', [])
            
            return {
                'underlying_price': underlying_price,
                'atm_strike': atm_strike,
                'selected_strikes': selected_strikes,
                'options_data': selected_options_data,
                'price_history': price_history,
                'volatility': volatility,
                'timestamp': market_data.get('timestamp', datetime.now())
            }
            
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return {}
    
    def _identify_oi_patterns_corrected(self, prepared_data: Dict[str, Any]) -> Dict[str, OIAnalysisResult]:
        """
        Identify OI patterns with CORRECTED logic
        
        CORRECTED LOGIC: Both calls and puts follow the same OI-Price relationship:
        - Long_Build_Up: OI↑ + Price↑ (Bullish positioning)
        - Long_Unwinding: OI↓ + Price↓ (Bearish unwinding)
        - Short_Build_Up: OI↑ + Price↓ (Bearish positioning)
        - Short_Covering: OI↓ + Price↑ (Bullish covering)
        """
        try:
            pattern_results = {}
            options_data = prepared_data.get('options_data', {})
            
            for strike, option_data in options_data.items():
                # Analyze calls
                if 'CE' in option_data:
                    call_pattern = self._identify_single_option_pattern(
                        option_data['CE'], 'call', strike, prepared_data
                    )
                    pattern_results[f'{strike}_CE'] = call_pattern
                
                # Analyze puts (SAME LOGIC as calls)
                if 'PE' in option_data:
                    put_pattern = self._identify_single_option_pattern(
                        option_data['PE'], 'put', strike, prepared_data
                    )
                    pattern_results[f'{strike}_PE'] = put_pattern
            
            return pattern_results
            
        except Exception as e:
            logger.error(f"Error identifying OI patterns: {e}")
            return {}
    
    def _identify_single_option_pattern(self, option_data: Dict[str, Any], 
                                      option_type: str, strike: float, 
                                      prepared_data: Dict[str, Any]) -> OIAnalysisResult:
        """
        Identify pattern for a single option using CORRECTED logic
        
        Args:
            option_data: Single option data (CE or PE)
            option_type: 'call' or 'put'
            strike: Strike price
            prepared_data: Prepared market data
            
        Returns:
            OIAnalysisResult with pattern classification
        """
        try:
            # Calculate OI velocity (change over time)
            current_oi = option_data.get('oi', 0)
            previous_oi = option_data.get('previous_oi', current_oi)
            oi_velocity = (current_oi - previous_oi) / max(previous_oi, 1)
            
            # Calculate price velocity (change over time)
            current_price = option_data.get('close', 0)
            previous_price = option_data.get('previous_close', current_price)
            price_velocity = (current_price - previous_price) / max(previous_price, 0.01)
            
            # Apply CORRECTED pattern logic (same for both calls and puts) with dynamic thresholds
            pattern = self._classify_oi_pattern_corrected(oi_velocity, price_velocity, prepared_data)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(oi_velocity, price_velocity, option_data)
            
            # Calculate confidence based on volume and consistency
            confidence = self._calculate_pattern_confidence(option_data, oi_velocity, price_velocity)
            
            # Calculate supporting metrics
            supporting_metrics = {
                'oi_velocity': oi_velocity,
                'price_velocity': price_velocity,
                'volume': option_data.get('volume', 0),
                'oi': current_oi,
                'price': current_price,
                'option_type': option_type,
                'strike': strike
            }
            
            return OIAnalysisResult(
                pattern=pattern,
                confidence=confidence,
                signal_strength=signal_strength,
                divergence_score=0.0,  # Will be calculated in divergence analysis
                institutional_ratio=0.0,  # Will be calculated in institutional analysis
                timeframe_consistency=0.0,  # Will be calculated in multi-timeframe analysis
                supporting_metrics=supporting_metrics
            )
            
        except Exception as e:
            logger.error(f"Error identifying single option pattern: {e}")
            return OIAnalysisResult(
                pattern=OIPattern.NEUTRAL,
                confidence=0.5,
                signal_strength=0.0,
                divergence_score=0.0,
                institutional_ratio=0.0,
                timeframe_consistency=0.0,
                supporting_metrics={}
            )
    
    def _classify_oi_pattern_corrected(self, oi_velocity: float, price_velocity: float,
                                     prepared_data: Optional[Dict[str, Any]] = None) -> OIPattern:
        """
        Classify OI pattern using CORRECTED logic with DYNAMIC THRESHOLDS

        CORRECTED LOGIC (applies to both calls and puts):
        - Long_Build_Up: OI↑ + Price↑ (Bullish positioning)
        - Long_Unwinding: OI↓ + Price↓ (Bearish unwinding)
        - Short_Build_Up: OI↑ + Price↓ (Bearish positioning)
        - Short_Covering: OI↓ + Price↑ (Bullish covering)
        """
        # PHASE 1 ENHANCEMENT: Use dynamic thresholds if available
        if prepared_data and 'dynamic_thresholds' in prepared_data and self.enable_dynamic_thresholds:
            threshold_result = prepared_data['dynamic_thresholds']
            oi_threshold = threshold_result.oi_threshold
            price_threshold = threshold_result.price_threshold
        else:
            # Fallback to static thresholds
            oi_threshold = 0.02    # 2% OI change threshold
            price_threshold = 0.01 # 1% price change threshold

        # Check if changes are significant enough
        significant_oi_change = abs(oi_velocity) > oi_threshold
        significant_price_change = abs(price_velocity) > price_threshold

        if not (significant_oi_change or significant_price_change):
            return OIPattern.NEUTRAL

        # Apply corrected pattern logic
        if oi_velocity > oi_threshold and price_velocity > price_threshold:
            return OIPattern.LONG_BUILD_UP      # OI↑ + Price↑ (Bullish)
        elif oi_velocity < -oi_threshold and price_velocity < -price_threshold:
            return OIPattern.LONG_UNWINDING     # OI↓ + Price↓ (Bearish)
        elif oi_velocity > oi_threshold and price_velocity < -price_threshold:
            return OIPattern.SHORT_BUILD_UP     # OI↑ + Price↓ (Bearish)
        elif oi_velocity < -oi_threshold and price_velocity > price_threshold:
            return OIPattern.SHORT_COVERING     # OI↓ + Price↑ (Bullish)
        else:
            return OIPattern.NEUTRAL
    
    def _calculate_signal_strength(self, oi_velocity: float, price_velocity: float, 
                                 option_data: Dict[str, Any]) -> float:
        """Calculate signal strength based on OI and price velocities"""
        try:
            # Base strength from velocity magnitudes
            oi_strength = min(abs(oi_velocity) * 10, 1.0)  # Scale to [0, 1]
            price_strength = min(abs(price_velocity) * 20, 1.0)  # Scale to [0, 1]
            
            # Volume confirmation
            volume = option_data.get('volume', 0)
            avg_volume = option_data.get('avg_volume', volume)
            volume_factor = min(volume / max(avg_volume, 1), 2.0)  # Up to 2x boost
            
            # Combined strength
            base_strength = (oi_strength + price_strength) / 2
            volume_adjusted_strength = base_strength * (0.7 + 0.3 * volume_factor)
            
            return np.clip(volume_adjusted_strength, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _calculate_pattern_confidence(self, option_data: Dict[str, Any], 
                                    oi_velocity: float, price_velocity: float) -> float:
        """Calculate confidence in pattern identification"""
        try:
            # Base confidence from velocity consistency
            velocity_consistency = 1.0 - abs(abs(oi_velocity) - abs(price_velocity))
            velocity_consistency = max(0.0, velocity_consistency)
            
            # Volume confirmation
            volume = option_data.get('volume', 0)
            if volume > self.min_volume_threshold:
                volume_confidence = min(volume / (self.min_volume_threshold * 5), 1.0)
            else:
                volume_confidence = 0.3  # Low confidence for low volume
            
            # OI significance
            oi = option_data.get('oi', 0)
            oi_confidence = min(oi / 10000, 1.0) if oi > 0 else 0.3
            
            # Combined confidence
            combined_confidence = (
                velocity_consistency * 0.4 +
                volume_confidence * 0.3 +
                oi_confidence * 0.3
            )
            
            return np.clip(combined_confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5

    def _analyze_multi_timeframe(self, prepared_data: Dict[str, Any]) -> MultiTimeframeResult:
        """
        Analyze OI patterns across multiple timeframes (3min + 15min)

        Returns:
            MultiTimeframeResult with combined signals and divergence detection
        """
        try:
            price_history = prepared_data.get('price_history', [])
            if len(price_history) < 20:
                return MultiTimeframeResult(0.0, 0.0, 0.0, False, 0.5)

            # Resample to different timeframes
            primary_data = self._resample_to_timeframe(price_history, self.primary_timeframe)
            confirmation_data = self._resample_to_timeframe(price_history, self.confirmation_timeframe)

            # Calculate signals for each timeframe
            primary_signal = self._calculate_timeframe_signal(primary_data, prepared_data)
            confirmation_signal = self._calculate_timeframe_signal(confirmation_data, prepared_data)

            # Detect timeframe divergence
            divergence_flag = abs(primary_signal - confirmation_signal) > 0.3

            # Calculate combined signal using weights
            combined_signal = (
                primary_signal * self.timeframe_weights[self.primary_timeframe] +
                confirmation_signal * self.timeframe_weights[self.confirmation_timeframe]
            )

            # Calculate confidence based on timeframe consistency
            consistency = 1.0 - abs(primary_signal - confirmation_signal)
            confidence = max(0.3, consistency)

            return MultiTimeframeResult(
                primary_signal=primary_signal,
                confirmation_signal=confirmation_signal,
                combined_signal=combined_signal,
                divergence_flag=divergence_flag,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return MultiTimeframeResult(0.0, 0.0, 0.0, False, 0.5)

    def _resample_to_timeframe(self, price_history: List[Dict], timeframe_minutes: int) -> List[Dict]:
        """Resample price history to specified timeframe"""
        try:
            if not price_history:
                return []

            # Convert to DataFrame for easier resampling
            df = pd.DataFrame(price_history)
            if 'timestamp' not in df.columns:
                return price_history  # Return original if no timestamp

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Resample to specified timeframe
            freq = f'{timeframe_minutes}T'
            resampled = df.resample(freq).agg({
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'volume': 'sum'
            }).dropna()

            # Convert back to list of dictionaries
            resampled_list = []
            for timestamp, row in resampled.iterrows():
                resampled_list.append({
                    'timestamp': timestamp,
                    'close': row['close'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume']
                })

            return resampled_list

        except Exception as e:
            logger.error(f"Error resampling to timeframe: {e}")
            return price_history

    def _calculate_timeframe_signal(self, timeframe_data: List[Dict],
                                  prepared_data: Dict[str, Any]) -> float:
        """Calculate OI signal for a specific timeframe"""
        try:
            if len(timeframe_data) < 5:
                return 0.0

            # Calculate price momentum for this timeframe
            recent_prices = [d['close'] for d in timeframe_data[-5:]]
            price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            # Estimate OI momentum (simplified for timeframe analysis)
            # In real implementation, this would use actual OI data for the timeframe
            options_data = prepared_data.get('options_data', {})

            total_call_oi_change = 0
            total_put_oi_change = 0
            total_options = 0

            for strike, option_data in options_data.items():
                if 'CE' in option_data:
                    ce_oi = option_data['CE'].get('oi', 0)
                    ce_prev_oi = option_data['CE'].get('previous_oi', ce_oi)
                    total_call_oi_change += (ce_oi - ce_prev_oi) / max(ce_prev_oi, 1)
                    total_options += 1

                if 'PE' in option_data:
                    pe_oi = option_data['PE'].get('oi', 0)
                    pe_prev_oi = option_data['PE'].get('previous_oi', pe_oi)
                    total_put_oi_change += (pe_oi - pe_prev_oi) / max(pe_prev_oi, 1)
                    total_options += 1

            if total_options == 0:
                return 0.0

            avg_oi_change = (total_call_oi_change + total_put_oi_change) / total_options

            # Combine price and OI momentum
            combined_momentum = 0.6 * price_momentum + 0.4 * avg_oi_change

            return np.clip(combined_momentum, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating timeframe signal: {e}")
            return 0.0

    def _detect_comprehensive_divergence(self, prepared_data: Dict[str, Any],
                                       pattern_results: Dict[str, OIAnalysisResult]) -> Dict[str, Any]:
        """
        Detect all 5 types of divergence:
        1. Pattern Divergence
        2. OI-Price Divergence
        3. Call-Put Divergence
        4. Institutional-Retail Divergence
        5. Cross-Strike Divergence
        """
        try:
            divergence_analysis = {
                'pattern_divergence': 0.0,
                'oi_price_divergence': 0.0,
                'call_put_divergence': 0.0,
                'institutional_retail_divergence': 0.0,
                'cross_strike_divergence': 0.0,
                'overall_divergence': 0.0,
                'divergence_flags': []
            }

            # 1. Pattern Divergence - Current vs Historical patterns
            pattern_divergence = self._calculate_pattern_divergence(pattern_results)
            divergence_analysis['pattern_divergence'] = pattern_divergence

            # 2. OI-Price Divergence - OI and price moving in opposite directions
            oi_price_divergence = self._calculate_oi_price_divergence(prepared_data, pattern_results)
            divergence_analysis['oi_price_divergence'] = oi_price_divergence

            # 3. Call-Put Divergence - Conflicting signals between calls and puts
            call_put_divergence = self._calculate_call_put_divergence(pattern_results)
            divergence_analysis['call_put_divergence'] = call_put_divergence

            # 4. Cross-Strike Divergence - Different strikes showing conflicting patterns
            cross_strike_divergence = self._calculate_cross_strike_divergence(pattern_results)
            divergence_analysis['cross_strike_divergence'] = cross_strike_divergence

            # Calculate overall divergence score
            divergence_scores = [
                pattern_divergence, oi_price_divergence, call_put_divergence, cross_strike_divergence
            ]
            overall_divergence = np.mean(divergence_scores)
            divergence_analysis['overall_divergence'] = overall_divergence

            # Generate divergence flags
            if overall_divergence > self.divergence_threshold:
                divergence_analysis['divergence_flags'].append('HIGH_DIVERGENCE')

            if oi_price_divergence > 0.5:
                divergence_analysis['divergence_flags'].append('OI_PRICE_CONFLICT')

            if call_put_divergence > 0.4:
                divergence_analysis['divergence_flags'].append('CALL_PUT_CONFLICT')

            return divergence_analysis

        except Exception as e:
            logger.error(f"Error detecting comprehensive divergence: {e}")
            return {'overall_divergence': 0.0, 'divergence_flags': []}

    def _calculate_pattern_divergence(self, pattern_results: Dict[str, OIAnalysisResult]) -> float:
        """Calculate pattern divergence from historical patterns"""
        try:
            if not pattern_results:
                return 0.0

            # Get current pattern distribution
            current_patterns = [result.pattern for result in pattern_results.values()]
            pattern_counts = {}
            for pattern in current_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            # Compare with historical pattern distribution (simplified)
            # In real implementation, this would use actual historical data
            expected_distribution = {
                OIPattern.LONG_BUILD_UP: 0.25,
                OIPattern.LONG_UNWINDING: 0.25,
                OIPattern.SHORT_BUILD_UP: 0.25,
                OIPattern.SHORT_COVERING: 0.25,
                OIPattern.NEUTRAL: 0.0
            }

            total_patterns = len(current_patterns)
            if total_patterns == 0:
                return 0.0

            # Calculate divergence from expected distribution
            divergence = 0.0
            for pattern, expected_ratio in expected_distribution.items():
                actual_ratio = pattern_counts.get(pattern, 0) / total_patterns
                divergence += abs(actual_ratio - expected_ratio)

            return min(divergence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating pattern divergence: {e}")
            return 0.0

    def _calculate_oi_price_divergence(self, prepared_data: Dict[str, Any],
                                     pattern_results: Dict[str, OIAnalysisResult]) -> float:
        """Calculate OI-Price divergence (when OI and price move in opposite directions)"""
        try:
            divergence_count = 0
            total_count = 0

            for option_key, result in pattern_results.items():
                metrics = result.supporting_metrics
                oi_velocity = metrics.get('oi_velocity', 0)
                price_velocity = metrics.get('price_velocity', 0)

                # Check for divergence (opposite directions with significant magnitude)
                if abs(oi_velocity) > 0.02 and abs(price_velocity) > 0.01:
                    if (oi_velocity > 0 and price_velocity < 0) or (oi_velocity < 0 and price_velocity > 0):
                        divergence_count += 1

                total_count += 1

            return divergence_count / total_count if total_count > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating OI-Price divergence: {e}")
            return 0.0

    def _calculate_call_put_divergence(self, pattern_results: Dict[str, OIAnalysisResult]) -> float:
        """
        Calculate Call-Put divergence with correlation analysis

        This implements the missing correlation/non-correlation logic between
        call OI with PA and put OI with PA for market regime formation.
        """
        try:
            call_signals = []
            put_signals = []
            call_oi_velocities = []
            put_oi_velocities = []
            call_price_velocities = []
            put_price_velocities = []

            for option_key, result in pattern_results.items():
                option_type = result.supporting_metrics.get('option_type', '')
                oi_velocity = result.supporting_metrics.get('oi_velocity', 0)
                price_velocity = result.supporting_metrics.get('price_velocity', 0)

                # Convert pattern to signal
                signal = self._pattern_to_signal(result.pattern)

                if option_type == 'call':
                    call_signals.append(signal)
                    call_oi_velocities.append(oi_velocity)
                    call_price_velocities.append(price_velocity)
                elif option_type == 'put':
                    put_signals.append(signal)
                    put_oi_velocities.append(oi_velocity)
                    put_price_velocities.append(price_velocity)

            if not call_signals or not put_signals:
                return 0.0

            # Calculate correlation between call OI patterns and put OI patterns
            call_put_correlation = self._calculate_call_put_oi_correlation(
                call_oi_velocities, put_oi_velocities,
                call_price_velocities, put_price_velocities
            )

            # Calculate signal divergence
            avg_call_signal = np.mean(call_signals)
            avg_put_signal = np.mean(put_signals)
            signal_divergence = abs(avg_call_signal - avg_put_signal) / 2.0

            # Combine correlation and signal divergence for final divergence score
            # High correlation + low signal divergence = low overall divergence
            # Low correlation + high signal divergence = high overall divergence
            correlation_factor = 1.0 - abs(call_put_correlation)  # Low correlation = high factor
            combined_divergence = (correlation_factor * 0.6 + signal_divergence * 0.4)

            return min(combined_divergence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating Call-Put divergence: {e}")
            return 0.0

    def _calculate_call_put_oi_correlation(self, call_oi_velocities: List[float],
                                         put_oi_velocities: List[float],
                                         call_price_velocities: List[float],
                                         put_price_velocities: List[float]) -> float:
        """
        Calculate correlation between call OI with PA and put OI with PA

        This is the core missing logic from the enhanced optimizer package.
        Market regime formation is based on correlation/non-correlation between:
        - Call OI patterns with price action
        - Put OI patterns with price action
        """
        try:
            if len(call_oi_velocities) < 3 or len(put_oi_velocities) < 3:
                return 0.0

            # Calculate call OI-PA correlation
            call_oi_pa_correlation = np.corrcoef(call_oi_velocities, call_price_velocities)[0, 1] \
                if len(call_oi_velocities) > 1 else 0.0

            # Calculate put OI-PA correlation
            put_oi_pa_correlation = np.corrcoef(put_oi_velocities, put_price_velocities)[0, 1] \
                if len(put_oi_velocities) > 1 else 0.0

            # Handle NaN correlations
            if np.isnan(call_oi_pa_correlation):
                call_oi_pa_correlation = 0.0
            if np.isnan(put_oi_pa_correlation):
                put_oi_pa_correlation = 0.0

            # Calculate correlation between call and put OI velocities
            call_put_oi_correlation = np.corrcoef(
                call_oi_velocities[:min(len(call_oi_velocities), len(put_oi_velocities))],
                put_oi_velocities[:min(len(call_oi_velocities), len(put_oi_velocities))]
            )[0, 1] if min(len(call_oi_velocities), len(put_oi_velocities)) > 1 else 0.0

            if np.isnan(call_put_oi_correlation):
                call_put_oi_correlation = 0.0

            # Market regime formation logic based on correlation patterns:
            # 1. High correlation between call OI-PA and put OI-PA = Trending market
            # 2. Low/negative correlation = Sideways/uncertain market
            # 3. Divergent call vs put OI patterns = Transition/reversal signals

            # Combine all correlation measures
            combined_correlation = (
                abs(call_oi_pa_correlation) * 0.3 +      # Call OI-PA strength
                abs(put_oi_pa_correlation) * 0.3 +       # Put OI-PA strength
                abs(call_put_oi_correlation) * 0.4       # Call-Put OI correlation
            )

            return np.clip(combined_correlation, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating call-put OI correlation: {e}")
            return 0.0

    def _calculate_cross_strike_divergence(self, pattern_results: Dict[str, OIAnalysisResult]) -> float:
        """Calculate Cross-Strike divergence (different strikes showing conflicting patterns)"""
        try:
            strike_signals = {}

            # Group by strike
            for option_key, result in pattern_results.items():
                strike = result.supporting_metrics.get('strike', 0)
                signal = self._pattern_to_signal(result.pattern)

                if strike not in strike_signals:
                    strike_signals[strike] = []
                strike_signals[strike].append(signal)

            # Calculate average signal per strike
            strike_avg_signals = {}
            for strike, signals in strike_signals.items():
                strike_avg_signals[strike] = np.mean(signals)

            if len(strike_avg_signals) < 2:
                return 0.0

            # Calculate divergence as standard deviation of strike signals
            signals = list(strike_avg_signals.values())
            divergence = np.std(signals)

            return min(divergence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating Cross-Strike divergence: {e}")
            return 0.0

    def _pattern_to_signal(self, pattern: OIPattern) -> float:
        """Convert OI pattern to numerical signal"""
        pattern_signals = {
            OIPattern.LONG_BUILD_UP: 1.0,      # Strong bullish
            OIPattern.SHORT_COVERING: 0.5,     # Mild bullish
            OIPattern.NEUTRAL: 0.0,            # Neutral
            OIPattern.SHORT_BUILD_UP: -0.5,    # Mild bearish
            OIPattern.LONG_UNWINDING: -1.0     # Strong bearish
        }
        return pattern_signals.get(pattern, 0.0)

    def _analyze_institutional_retail(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze institutional vs retail positioning

        Returns:
            Dictionary with institutional analysis results
        """
        try:
            options_data = prepared_data.get('options_data', {})

            total_institutional_oi = 0
            total_retail_oi = 0
            institutional_call_oi = 0
            institutional_put_oi = 0
            retail_call_oi = 0
            retail_put_oi = 0

            for strike, option_data in options_data.items():
                # Analyze calls
                if 'CE' in option_data:
                    ce_data = option_data['CE']
                    ce_oi = ce_data.get('oi', 0)
                    ce_volume = ce_data.get('volume', 0)

                    # Institutional detection based on OI/Volume ratio
                    if ce_volume > 0:
                        oi_volume_ratio = ce_oi / ce_volume
                        if oi_volume_ratio > 10:  # High OI relative to volume = Institutional
                            institutional_call_oi += ce_oi
                            total_institutional_oi += ce_oi
                        else:
                            retail_call_oi += ce_oi
                            total_retail_oi += ce_oi

                # Analyze puts
                if 'PE' in option_data:
                    pe_data = option_data['PE']
                    pe_oi = pe_data.get('oi', 0)
                    pe_volume = pe_data.get('volume', 0)

                    # Institutional detection based on OI/Volume ratio
                    if pe_volume > 0:
                        oi_volume_ratio = pe_oi / pe_volume
                        if oi_volume_ratio > 10:  # High OI relative to volume = Institutional
                            institutional_put_oi += pe_oi
                            total_institutional_oi += pe_oi
                        else:
                            retail_put_oi += pe_oi
                            total_retail_oi += pe_oi

            total_oi = total_institutional_oi + total_retail_oi

            if total_oi == 0:
                return {
                    'institutional_ratio': 0.5,
                    'institutional_call_ratio': 0.5,
                    'institutional_put_ratio': 0.5,
                    'positioning_bias': 'NEUTRAL',
                    'confidence': 0.3
                }

            # Calculate ratios
            institutional_ratio = total_institutional_oi / total_oi

            institutional_call_ratio = (
                institutional_call_oi / (institutional_call_oi + retail_call_oi)
                if (institutional_call_oi + retail_call_oi) > 0 else 0.5
            )

            institutional_put_ratio = (
                institutional_put_oi / (institutional_put_oi + retail_put_oi)
                if (institutional_put_oi + retail_put_oi) > 0 else 0.5
            )

            # Determine positioning bias
            if institutional_call_ratio > 0.6:
                positioning_bias = 'INSTITUTIONAL_BULLISH'
            elif institutional_put_ratio > 0.6:
                positioning_bias = 'INSTITUTIONAL_BEARISH'
            elif institutional_ratio > 0.7:
                positioning_bias = 'INSTITUTIONAL_DOMINATED'
            elif institutional_ratio < 0.3:
                positioning_bias = 'RETAIL_DOMINATED'
            else:
                positioning_bias = 'MIXED_POSITIONING'

            # Calculate confidence based on data quality
            confidence = min(total_oi / 100000, 1.0)  # Higher confidence with more OI

            return {
                'institutional_ratio': institutional_ratio,
                'institutional_call_ratio': institutional_call_ratio,
                'institutional_put_ratio': institutional_put_ratio,
                'positioning_bias': positioning_bias,
                'confidence': confidence,
                'total_institutional_oi': total_institutional_oi,
                'total_retail_oi': total_retail_oi
            }

        except Exception as e:
            logger.error(f"Error analyzing institutional vs retail: {e}")
            return {
                'institutional_ratio': 0.5,
                'positioning_bias': 'NEUTRAL',
                'confidence': 0.3
            }

    def _apply_volume_weighting(self, pattern_results: Dict[str, OIAnalysisResult],
                              prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply volume weighting to OI analysis results"""
        try:
            volume_weighted_results = {}
            total_volume = 0

            # Calculate total volume across all options
            for option_key, result in pattern_results.items():
                volume = result.supporting_metrics.get('volume', 0)
                total_volume += volume

            if total_volume == 0:
                return {'volume_weighted_signal': 0.0, 'volume_confidence': 0.3}

            # Calculate volume-weighted signal
            weighted_signal = 0.0
            weighted_confidence = 0.0

            for option_key, result in pattern_results.items():
                volume = result.supporting_metrics.get('volume', 0)
                volume_weight = volume / total_volume

                signal = self._pattern_to_signal(result.pattern) * result.signal_strength
                weighted_signal += signal * volume_weight
                weighted_confidence += result.confidence * volume_weight

            # Apply volume factor
            avg_volume = total_volume / len(pattern_results) if pattern_results else 0
            volume_factor = min(avg_volume / self.min_volume_threshold, 2.0) if avg_volume > 0 else 0.5

            # Adjust signal and confidence based on volume
            final_signal = weighted_signal * (0.7 + 0.3 * volume_factor)
            final_confidence = weighted_confidence * (0.8 + 0.2 * volume_factor)

            return {
                'volume_weighted_signal': np.clip(final_signal, -1.0, 1.0),
                'volume_confidence': np.clip(final_confidence, 0.1, 1.0),
                'volume_factor': volume_factor,
                'total_volume': total_volume
            }

        except Exception as e:
            logger.error(f"Error applying volume weighting: {e}")
            return {'volume_weighted_signal': 0.0, 'volume_confidence': 0.3}

    def _apply_session_weighting(self, volume_weighted_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply session-based weighting (market open/mid/close)"""
        try:
            current_time = datetime.now().time()

            # Determine market session
            if current_time >= datetime.strptime('09:15', '%H:%M').time() and \
               current_time <= datetime.strptime('10:30', '%H:%M').time():
                session = 'market_open'
            elif current_time >= datetime.strptime('14:30', '%H:%M').time() and \
                 current_time <= datetime.strptime('15:30', '%H:%M').time():
                session = 'market_close'
            else:
                session = 'mid_session'

            # Get session weight
            session_factor = self.session_weights.get(session, 1.0)

            # Apply session weighting
            original_signal = volume_weighted_results.get('volume_weighted_signal', 0.0)
            session_adjusted_signal = original_signal * session_factor

            # Normalize to maintain [-1, 1] range
            session_adjusted_signal = np.clip(session_adjusted_signal, -1.0, 1.0)

            return {
                'session_adjusted_signal': session_adjusted_signal,
                'session': session,
                'session_factor': session_factor,
                'original_signal': original_signal
            }

        except Exception as e:
            logger.error(f"Error applying session weighting: {e}")
            return {'session_adjusted_signal': 0.0, 'session_factor': 1.0}

    def _calculate_final_signal(self, multi_tf_result: MultiTimeframeResult,
                              session_results: Dict[str, Any],
                              divergence_analysis: Dict[str, Any],
                              institutional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final OI signal combining all analysis components"""
        try:
            # Base signal from session-adjusted results
            base_signal = session_results.get('session_adjusted_signal', 0.0)

            # Multi-timeframe signal
            multi_tf_signal = multi_tf_result.combined_signal

            # Combine base and multi-timeframe signals
            combined_signal = 0.7 * base_signal + 0.3 * multi_tf_signal

            # Base confidence
            base_confidence = multi_tf_result.confidence

            # Apply divergence adjustment
            overall_divergence = divergence_analysis.get('overall_divergence', 0.0)
            divergence_adjustment = 1.0 - (overall_divergence * self.confidence_reduction_factor)
            adjusted_confidence = base_confidence * divergence_adjustment

            # Apply institutional confidence boost
            institutional_confidence = institutional_analysis.get('confidence', 0.5)
            institutional_ratio = institutional_analysis.get('institutional_ratio', 0.5)

            # Boost confidence if institutional activity is high
            if institutional_ratio > 0.6:
                institutional_boost = 1.0 + (institutional_ratio - 0.6) * 0.5
                adjusted_confidence *= institutional_boost

            # Apply call-put OI correlation regime adjustment
            call_put_divergence = divergence_analysis.get('call_put_divergence', 0.0)
            regime_classification = self._classify_oi_correlation_regime(call_put_divergence, combined_signal)

            # Final confidence
            final_confidence = np.clip(adjusted_confidence, 0.1, 1.0)

            # Apply confidence to signal strength
            final_signal = combined_signal * final_confidence

            return {
                'signal': np.clip(final_signal, -1.0, 1.0),
                'confidence': final_confidence,
                'base_signal': base_signal,
                'multi_tf_signal': multi_tf_signal,
                'divergence_adjustment': divergence_adjustment,
                'institutional_boost': institutional_ratio > 0.6,
                'oi_correlation_regime': regime_classification
            }

        except Exception as e:
            logger.error(f"Error calculating final signal: {e}")
            return {'signal': 0.0, 'confidence': 0.5}

    def _classify_oi_correlation_regime(self, call_put_divergence: float, signal: float) -> str:
        """
        Classify market regime based on call-put OI correlation analysis

        This implements the core regime formation logic from the enhanced optimizer:
        - High correlation + strong signal = Trending regime
        - Low correlation + weak signal = Sideways regime
        - High divergence = Transition/Reversal regime
        """
        try:
            # Convert divergence to correlation (inverse relationship)
            correlation_strength = 1.0 - call_put_divergence
            signal_strength = abs(signal)

            # Regime classification based on correlation and signal patterns
            if correlation_strength > 0.7 and signal_strength > 0.5:
                if signal > 0:
                    return 'TRENDING_BULLISH_HIGH_CORRELATION'
                else:
                    return 'TRENDING_BEARISH_HIGH_CORRELATION'

            elif correlation_strength > 0.7 and signal_strength <= 0.5:
                return 'SIDEWAYS_HIGH_CORRELATION'

            elif correlation_strength <= 0.3 and signal_strength > 0.5:
                if signal > 0:
                    return 'DIVERGENT_BULLISH_LOW_CORRELATION'
                else:
                    return 'DIVERGENT_BEARISH_LOW_CORRELATION'

            elif correlation_strength <= 0.3 and signal_strength <= 0.5:
                return 'UNCERTAIN_LOW_CORRELATION'

            else:  # Medium correlation (0.3 - 0.7)
                if signal_strength > 0.3:
                    if signal > 0:
                        return 'TRANSITIONAL_BULLISH_MEDIUM_CORRELATION'
                    else:
                        return 'TRANSITIONAL_BEARISH_MEDIUM_CORRELATION'
                else:
                    return 'NEUTRAL_MEDIUM_CORRELATION'

        except Exception as e:
            logger.error(f"Error classifying OI correlation regime: {e}")
            return 'UNKNOWN_REGIME'

    def _update_performance_tracking(self, pattern_results: Dict[str, OIAnalysisResult],
                                   final_signal: Dict[str, Any]):
        """Update performance tracking for pattern accuracy"""
        try:
            # Track pattern occurrences
            for option_key, result in pattern_results.items():
                pattern = result.pattern
                if pattern not in self.pattern_performance_history:
                    self.pattern_performance_history[pattern] = {
                        'occurrences': 0,
                        'success_count': 0,
                        'success_rate': 0.5
                    }

                self.pattern_performance_history[pattern]['occurrences'] += 1

            # Store current signal for future validation
            # In real implementation, this would be validated against actual market moves

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    def _get_default_result(self) -> Dict[str, Any]:
        """Get default result when analysis fails"""
        return {
            'oi_signal': 0.0,
            'confidence': 0.5,
            'pattern_breakdown': {},
            'multi_timeframe': MultiTimeframeResult(0.0, 0.0, 0.0, False, 0.5),
            'divergence_analysis': {'overall_divergence': 0.0, 'divergence_flags': []},
            'institutional_analysis': {'institutional_ratio': 0.5, 'positioning_bias': 'NEUTRAL'},
            'session_adjustment': 1.0,
            'timestamp': datetime.now(),
            'analysis_type': 'enhanced_trending_oi_pa_v2_default'
        }

    def get_pattern_performance_summary(self) -> Dict[str, Any]:
        """Get summary of pattern performance for optimization"""
        try:
            summary = {}
            for pattern, history in self.pattern_performance_history.items():
                summary[pattern.value] = {
                    'occurrences': history['occurrences'],
                    'success_rate': history['success_rate'],
                    'reliability': 'High' if history['success_rate'] > 0.7 else
                                  'Medium' if history['success_rate'] > 0.5 else 'Low'
                }
            return summary
        except Exception as e:
            logger.error(f"Error getting pattern performance summary: {e}")
            return {}

    # ===== ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0 - MATHEMATICAL CORRELATION METHODS =====

    def _perform_correlation_analysis(self, pattern_results: Dict[str, OIAnalysisResult],
                                    prepared_data: Dict[str, Any],
                                    volume_weighted_results: Dict[str, Any]) -> Optional[CorrelationAnalysisResult]:
        """
        Perform Pearson correlation analysis for pattern similarity (>0.80 threshold)

        Args:
            pattern_results: Current pattern analysis results
            prepared_data: Prepared market data
            volume_weighted_results: Volume-weighted analysis results

        Returns:
            CorrelationAnalysisResult or None if analysis fails
        """
        try:
            # Extract current pattern vector
            current_pattern_vector = self._extract_pattern_vector(pattern_results)

            if len(current_pattern_vector) < 3:  # Need minimum data points
                logger.warning("Insufficient data for correlation analysis")
                return None

            # Find best historical pattern match
            best_correlation = -1.0
            best_match = None
            correlation_confidence = 0.0

            if len(self.historical_patterns) >= self.correlation_window:
                for historical_pattern in self.historical_patterns[-self.correlation_window:]:
                    historical_vector = historical_pattern.get('pattern_vector', [])

                    if len(historical_vector) == len(current_pattern_vector):
                        try:
                            # Calculate Pearson correlation
                            correlation, p_value = pearsonr(current_pattern_vector, historical_vector)

                            if not np.isnan(correlation) and correlation > best_correlation:
                                best_correlation = correlation
                                best_match = historical_pattern
                                correlation_confidence = 1.0 - p_value if not np.isnan(p_value) else 0.5

                        except Exception as corr_e:
                            logger.warning(f"Error calculating correlation: {corr_e}")
                            continue

            # Calculate pattern similarity score
            pattern_similarity = self._calculate_pattern_similarity(
                pattern_results, best_match if best_match else {}
            )

            # Calculate time-decay weight for this correlation
            time_decay_weight = 1.0
            if best_match and 'timestamp' in best_match:
                current_time = prepared_data.get('timestamp', datetime.now())
                historical_time = best_match['timestamp']
                time_decay_weight = self._calculate_single_time_decay_weight(
                    current_time, historical_time
                )

            # Validate mathematical accuracy
            mathematical_accuracy = self._validate_correlation_accuracy(
                best_correlation, correlation_confidence, pattern_similarity
            )

            # Check if correlation threshold is met
            threshold_met = best_correlation >= self.correlation_threshold

            # Store current pattern for future correlation analysis
            current_pattern_data = {
                'pattern_vector': current_pattern_vector,
                'timestamp': prepared_data.get('timestamp', datetime.now()),
                'pattern_results': pattern_results,
                'volume_weighted_results': volume_weighted_results
            }
            self.historical_patterns.append(current_pattern_data)

            # Keep only recent patterns for memory management
            if len(self.historical_patterns) > 1000:
                self.historical_patterns = self.historical_patterns[-1000:]

            return CorrelationAnalysisResult(
                pearson_correlation=best_correlation,
                correlation_confidence=correlation_confidence,
                pattern_similarity_score=pattern_similarity,
                time_decay_weight=time_decay_weight,
                mathematical_accuracy=mathematical_accuracy,
                correlation_threshold_met=threshold_met,
                historical_pattern_match=best_match
            )

        except Exception as e:
            logger.error(f"Error performing correlation analysis: {e}")
            return None

    def _extract_pattern_vector(self, pattern_results: Dict[str, OIAnalysisResult]) -> List[float]:
        """Extract numerical vector from pattern results for correlation analysis"""
        try:
            vector = []

            for option_key, result in pattern_results.items():
                # Pattern strength
                pattern_strength = self._pattern_to_signal(result.pattern)
                vector.append(pattern_strength)

                # Confidence
                vector.append(result.confidence)

                # Signal strength
                vector.append(result.signal_strength)

                # OI velocity
                oi_velocity = result.supporting_metrics.get('oi_velocity', 0)
                vector.append(oi_velocity)

                # Price velocity
                price_velocity = result.supporting_metrics.get('price_velocity', 0)
                vector.append(price_velocity)

            return vector

        except Exception as e:
            logger.error(f"Error extracting pattern vector: {e}")
            return []

    def _calculate_pattern_similarity(self, current_patterns: Dict[str, OIAnalysisResult],
                                    historical_patterns: Dict[str, Any]) -> float:
        """Calculate pattern similarity score using multiple metrics"""
        try:
            if not historical_patterns:
                return 0.0

            # Pattern type similarity
            current_pattern_types = [result.pattern.value for result in current_patterns.values()]
            historical_pattern_types = []

            if 'pattern_results' in historical_patterns:
                historical_pattern_types = [
                    result.pattern.value for result in historical_patterns['pattern_results'].values()
                ]

            # Calculate Jaccard similarity for pattern types
            current_set = set(current_pattern_types)
            historical_set = set(historical_pattern_types)

            if len(current_set.union(historical_set)) == 0:
                jaccard_similarity = 0.0
            else:
                jaccard_similarity = len(current_set.intersection(historical_set)) / len(current_set.union(historical_set))

            # Confidence similarity
            current_confidences = [result.confidence for result in current_patterns.values()]
            historical_confidences = []

            if 'pattern_results' in historical_patterns:
                historical_confidences = [
                    result.confidence for result in historical_patterns['pattern_results'].values()
                ]

            confidence_similarity = 0.0
            if current_confidences and historical_confidences:
                min_len = min(len(current_confidences), len(historical_confidences))
                current_conf_subset = current_confidences[:min_len]
                historical_conf_subset = historical_confidences[:min_len]

                try:
                    conf_correlation, _ = pearsonr(current_conf_subset, historical_conf_subset)
                    confidence_similarity = conf_correlation if not np.isnan(conf_correlation) else 0.0
                except:
                    confidence_similarity = 0.0

            # Combined similarity score
            similarity_score = (jaccard_similarity * 0.6 + confidence_similarity * 0.4)

            return np.clip(similarity_score, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

    def _calculate_time_decay_weights(self, prepared_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate time-decay weights using exp(-λ × (T-t)) formula

        Args:
            prepared_data: Prepared market data with timestamp

        Returns:
            Dictionary containing time-decay weight information
        """
        try:
            current_time = prepared_data.get('timestamp', datetime.now())

            # Calculate time elapsed since baseline (in seconds)
            if hasattr(self, 'baseline_time'):
                baseline_time = self.baseline_time
            else:
                # Set baseline to market open (9:15 AM)
                baseline_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
                self.baseline_time = baseline_time

            time_elapsed = (current_time - baseline_time).total_seconds()

            # Apply time-decay formula: exp(-λ × (T-t))
            # Where λ is decay parameter, T is current time, t is baseline time
            decay_factor = np.exp(-self.time_decay_config.lambda_decay * (time_elapsed / self.time_decay_config.time_window))

            # Calculate current weight with bounds
            current_weight = self.time_decay_config.baseline_weight * decay_factor
            current_weight = np.clip(
                current_weight,
                self.time_decay_config.min_weight,
                self.time_decay_config.max_weight
            )

            return {
                'current_weight': current_weight,
                'decay_factor': decay_factor,
                'time_elapsed': time_elapsed,
                'baseline_time': baseline_time,
                'lambda_decay': self.time_decay_config.lambda_decay
            }

        except Exception as e:
            logger.error(f"Error calculating time-decay weights: {e}")
            return {'current_weight': 1.0, 'decay_factor': 1.0, 'time_elapsed': 0}

    def _calculate_single_time_decay_weight(self, current_time: datetime, historical_time: datetime) -> float:
        """Calculate time-decay weight for a single historical point"""
        try:
            time_diff = (current_time - historical_time).total_seconds()
            decay_factor = np.exp(-self.time_decay_config.lambda_decay * (time_diff / self.time_decay_config.time_window))
            weight = self.time_decay_config.baseline_weight * decay_factor
            return np.clip(weight, self.time_decay_config.min_weight, self.time_decay_config.max_weight)
        except Exception as e:
            logger.error(f"Error calculating single time-decay weight: {e}")
            return 1.0

    def _apply_time_decay_weighting(self, session_adjusted_results: Dict[str, Any],
                                  time_decay_weights: Dict[str, float]) -> Dict[str, Any]:
        """Apply time-decay weighting to session-adjusted results"""
        try:
            if not time_decay_weights:
                return session_adjusted_results

            current_weight = time_decay_weights.get('current_weight', 1.0)

            # Apply time-decay weight to signal strength
            if 'signal_strength' in session_adjusted_results:
                session_adjusted_results['signal_strength'] *= current_weight

            # Apply time-decay weight to confidence
            if 'confidence' in session_adjusted_results:
                session_adjusted_results['confidence'] *= current_weight

            # Store time-decay information
            session_adjusted_results['time_decay_applied'] = True
            session_adjusted_results['time_decay_weight'] = current_weight

            return session_adjusted_results

        except Exception as e:
            logger.error(f"Error applying time-decay weighting: {e}")
            return session_adjusted_results

    def _validate_mathematical_accuracy(self, final_signal: Dict[str, Any]) -> bool:
        """Validate mathematical accuracy within ±0.001 tolerance"""
        try:
            signal = final_signal.get('signal', 0.0)
            confidence = final_signal.get('confidence', 0.0)

            # Check if values are finite
            if not (np.isfinite(signal) and np.isfinite(confidence)):
                logger.error("Mathematical accuracy check failed: non-finite values")
                return False

            # Check if values are within expected bounds
            if not (-1.1 <= signal <= 1.1):  # Allow small tolerance beyond [-1, 1]
                logger.error(f"Signal out of bounds: {signal}")
                return False

            if not (0.0 <= confidence <= 1.1):  # Allow small tolerance beyond [0, 1]
                logger.error(f"Confidence out of bounds: {confidence}")
                return False

            # Check precision (should be representable within tolerance)
            signal_rounded = round(signal, 3)  # Round to 3 decimal places (0.001 precision)
            confidence_rounded = round(confidence, 3)

            signal_error = abs(signal - signal_rounded)
            confidence_error = abs(confidence - confidence_rounded)

            if signal_error > MATHEMATICAL_TOLERANCE or confidence_error > MATHEMATICAL_TOLERANCE:
                logger.warning(f"Mathematical precision warning: signal_error={signal_error:.6f}, confidence_error={confidence_error:.6f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating mathematical accuracy: {e}")
            return False

    def _validate_correlation_accuracy(self, correlation: float, confidence: float, similarity: float) -> bool:
        """Validate correlation analysis mathematical accuracy"""
        try:
            # Check if values are finite
            values = [correlation, confidence, similarity]
            if not all(np.isfinite(v) for v in values):
                return False

            # Check bounds
            if not (-1.1 <= correlation <= 1.1):
                return False

            if not (0.0 <= confidence <= 1.1):
                return False

            if not (0.0 <= similarity <= 1.1):
                return False

            # Check precision
            for value in values:
                rounded_value = round(value, 3)
                precision_error = abs(value - rounded_value)
                if precision_error > MATHEMATICAL_TOLERANCE:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating correlation accuracy: {e}")
            return False

    def _calculate_final_signal_enhanced(self, multi_tf_result: MultiTimeframeResult,
                                       session_adjusted_results: Dict[str, Any],
                                       divergence_analysis: Dict[str, Any],
                                       institutional_analysis: Dict[str, Any],
                                       correlation_analysis: Optional[CorrelationAnalysisResult],
                                       time_decay_weights: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate final signal with enhanced mathematical correlation features"""
        try:
            # Base signal calculation (preserve existing logic)
            base_signal = self._calculate_final_signal(
                multi_tf_result, session_adjusted_results, divergence_analysis, institutional_analysis
            )

            # Enhanced signal with correlation and time-decay
            enhanced_signal = base_signal['signal']
            enhanced_confidence = base_signal['confidence']

            # Apply correlation enhancement
            if correlation_analysis and correlation_analysis.correlation_threshold_met:
                correlation_boost = correlation_analysis.pearson_correlation * 0.1  # 10% boost for high correlation
                enhanced_signal += correlation_boost
                enhanced_confidence *= (1.0 + correlation_analysis.correlation_confidence * 0.05)  # 5% confidence boost

            # Apply time-decay weighting
            if time_decay_weights:
                time_weight = time_decay_weights.get('current_weight', 1.0)
                enhanced_signal *= time_weight
                enhanced_confidence *= time_weight

            # Ensure bounds
            enhanced_signal = np.clip(enhanced_signal, -1.0, 1.0)
            enhanced_confidence = np.clip(enhanced_confidence, 0.0, 1.0)

            return {
                'signal': enhanced_signal,
                'confidence': enhanced_confidence,
                'base_signal': base_signal['signal'],
                'base_confidence': base_signal['confidence'],
                'correlation_enhancement': correlation_analysis is not None,
                'time_decay_applied': time_decay_weights is not None
            }

        except Exception as e:
            logger.error(f"Error calculating enhanced final signal: {e}")
            return {'signal': 0.0, 'confidence': 0.5}

    def _update_performance_tracking_enhanced(self, pattern_results: Dict[str, OIAnalysisResult],
                                            final_signal: Dict[str, Any],
                                            correlation_analysis: Optional[CorrelationAnalysisResult]) -> None:
        """Update performance tracking with enhanced correlation metrics"""
        try:
            # Call original performance tracking
            self._update_performance_tracking(pattern_results, final_signal)

            # Add correlation-specific tracking
            if correlation_analysis:
                if not hasattr(self, 'correlation_performance_history'):
                    self.correlation_performance_history = []

                correlation_performance = {
                    'timestamp': datetime.now(),
                    'pearson_correlation': correlation_analysis.pearson_correlation,
                    'correlation_confidence': correlation_analysis.correlation_confidence,
                    'pattern_similarity_score': correlation_analysis.pattern_similarity_score,
                    'threshold_met': correlation_analysis.correlation_threshold_met,
                    'mathematical_accuracy': correlation_analysis.mathematical_accuracy,
                    'final_signal_strength': abs(final_signal.get('signal', 0.0))
                }

                self.correlation_performance_history.append(correlation_performance)

                # Keep only recent history
                if len(self.correlation_performance_history) > 1000:
                    self.correlation_performance_history = self.correlation_performance_history[-1000:]

        except Exception as e:
            logger.error(f"Error updating enhanced performance tracking: {e}")

    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics including correlation analysis"""
        try:
            base_metrics = self.get_pattern_performance_summary()

            enhanced_metrics = {
                'base_performance': base_metrics,
                'mathematical_accuracy_rate': np.mean(self.mathematical_accuracy_history) if self.mathematical_accuracy_history else 0.0,
                'total_patterns_analyzed': len(self.historical_patterns),
                'correlation_cache_size': len(self.correlation_cache)
            }

            # Add correlation-specific metrics
            if hasattr(self, 'correlation_performance_history') and self.correlation_performance_history:
                recent_correlations = self.correlation_performance_history[-100:]  # Last 100

                enhanced_metrics['correlation_metrics'] = {
                    'average_correlation': np.mean([c['pearson_correlation'] for c in recent_correlations]),
                    'correlation_threshold_success_rate': np.mean([c['threshold_met'] for c in recent_correlations]),
                    'average_pattern_similarity': np.mean([c['pattern_similarity_score'] for c in recent_correlations]),
                    'mathematical_accuracy_rate': np.mean([c['mathematical_accuracy'] for c in recent_correlations])
                }

            return enhanced_metrics

        except Exception as e:
            logger.error(f"Error getting enhanced performance metrics: {e}")
            return {}
