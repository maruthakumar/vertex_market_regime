"""
Enhanced 18-Regime Detector V2 with Triple Straddle Analysis

This module implements the enhanced version of the 18-regime detector that integrates
the sophisticated Triple Straddle Analysis system along with all other indicators
from the enhanced-market-regime-optimizer system.

Features:
- Triple Straddle Analysis (ATM, ITM1, OTM1)
- Enhanced Greek Sentiment Analysis
- Advanced OI Pattern Recognition
- IV Skew Analysis
- ATR/Premium Analysis
- Dynamic Weight Optimization
- Multi-timeframe Analysis
- Real-time Performance Tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import sys
import os

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from correct locations
try:
    from .enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType
except ImportError:
    from enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType

try:
    from ..triple_straddle_analysis import TripleStraddleAnalysisEngine
except ImportError:
    from triple_straddle_analysis import TripleStraddleAnalysisEngine

try:
    from ..dynamic_weight_optimizer import DynamicWeightOptimizer, PerformanceMetrics
except ImportError:
    from dynamic_weight_optimizer import DynamicWeightOptimizer, PerformanceMetrics

# Import enhanced local implementations
try:
    from .enhanced_trending_oi_pa_analysis import OIPriceActionAnalyzer as EnhancedTrendingOIWithPAAnalysis
except ImportError:
    from enhanced_trending_oi_pa_analysis import OIPriceActionAnalyzer as EnhancedTrendingOIWithPAAnalysis

try:
    from .enhanced_greek_sentiment_analysis import GreekSentimentAnalyzerAnalysis
except ImportError:
    from enhanced_greek_sentiment_analysis import GreekSentimentAnalyzerAnalysis

# Try to import enhanced optimizer components as fallback
try:
    from utils.feature_engineering.iv_skew.iv_skew_analysis import IVSkewAnalysis
    from utils.feature_engineering.atr_indicators import ATRIndicators
    from utils.feature_engineering.iv_indicators import IVIndicators
    from utils.feature_engineering.premium_indicators import PremiumIndicators
except ImportError as e:
    logger.warning(f"Could not import enhanced optimizer components: {e}")
    # Fallback to basic implementations
    IVSkewAnalysis = None
    ATRIndicators = None
    IVIndicators = None
    PremiumIndicators = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRegimeResult:
    """Enhanced regime detection result"""
    regime_type: Enhanced18RegimeType
    regime_score: float
    confidence: float
    indicator_breakdown: Dict[str, Any]
    weights_applied: Dict[str, Dict[str, float]]
    performance_metrics: PerformanceMetrics
    timestamp: datetime

class Enhanced18RegimeDetectorV2:
    """
    Enhanced 18-Regime Detector V2 with Triple Straddle Analysis
    
    This detector integrates:
    1. Triple Straddle Analysis (20% weight) - NEW CORE COMPONENT
    2. Enhanced Greek Sentiment (25% weight)
    3. Advanced OI Analysis (15% weight)
    4. IV Skew Analysis (15% weight)
    5. ATR/Premium Analysis (10% weight)
    6. Supporting Technical (15% weight)
    
    With dynamic weight optimization based on historical performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Enhanced 18-Regime Detector V2"""
        self.config = config or {}
        
        # Initialize base detector for regime classification
        self.base_detector = Enhanced18RegimeDetector(config)
        
        # Initialize core analysis engines
        self.triple_straddle_engine = TripleStraddleAnalysisEngine(config)
        self.weight_optimizer = DynamicWeightOptimizer(config)
        
        # Initialize enhanced indicator systems
        self._initialize_indicator_systems()
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.regime_history: List[EnhancedRegimeResult] = []
        
        # Current weights (will be dynamically optimized)
        self.current_weights = self.weight_optimizer.get_current_weights()
        
        logger.info("Enhanced 18-Regime Detector V2 initialized with Triple Straddle Analysis")
    
    def _initialize_indicator_systems(self):
        """Initialize all indicator systems"""
        try:
            # Enhanced Greek Sentiment (Local Implementation)
            self.greek_sentiment = GreekSentimentAnalyzerAnalysis(self.config.get('greek_sentiment', {}))

            # Enhanced OI Analysis (Local Implementation)
            self.oi_analysis = EnhancedTrendingOIWithPAAnalysis(self.config.get('oi_analysis', {}))
            
            # IV Skew Analysis
            if IVSkewAnalysis:
                self.iv_skew = IVSkewAnalysis(self.config.get('iv_skew', {}))
            else:
                self.iv_skew = None
                logger.warning("IV Skew Analysis not available")
            
            # ATR Indicators
            if ATRIndicators:
                self.atr_indicators = ATRIndicators(self.config.get('atr', {}))
            else:
                self.atr_indicators = None
                logger.warning("ATR Indicators not available")
            
            # Premium Indicators
            if PremiumIndicators:
                self.premium_indicators = PremiumIndicators(self.config.get('premium', {}))
            else:
                self.premium_indicators = None
                logger.warning("Premium Indicators not available")
            
        except Exception as e:
            logger.error(f"Error initializing indicator systems: {e}")
    
    def detect_regime(self, market_data: Dict[str, Any]) -> EnhancedRegimeResult:
        """
        Enhanced regime detection with Triple Straddle Analysis
        
        Args:
            market_data: Comprehensive market data including options, Greeks, OI, etc.
            
        Returns:
            EnhancedRegimeResult with detailed regime analysis
        """
        try:
            # Calculate all indicator scores
            indicator_scores = self._calculate_all_indicator_scores(market_data)
            
            # Get current optimized weights
            current_weights = self.weight_optimizer.get_current_weights()
            
            # Calculate final regime score
            final_score = self._calculate_final_regime_score(indicator_scores, current_weights)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(indicator_scores, current_weights)
            
            # Classify into 18 regimes using base detector
            regime_type = self.base_detector._classify_18_regimes(
                final_score, self._estimate_volatility(market_data)
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                regime_type, final_score, overall_confidence, market_data
            )
            
            # Create enhanced result
            result = EnhancedRegimeResult(
                regime_type=regime_type,
                regime_score=final_score,
                confidence=overall_confidence,
                indicator_breakdown=indicator_scores,
                weights_applied=current_weights,
                performance_metrics=performance_metrics,
                timestamp=datetime.now()
            )
            
            # Update performance tracking
            self._update_performance_tracking(result)
            
            # Trigger weight optimization if needed
            self._trigger_weight_optimization_if_needed()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced regime detection: {e}")
            return self._get_default_result()
    
    def _calculate_all_indicator_scores(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate scores from all indicator systems"""
        indicator_scores = {}
        
        try:
            # 1. Triple Straddle Analysis (NEW CORE COMPONENT)
            triple_straddle_result = self.triple_straddle_engine.analyze_market_regime(market_data)
            indicator_scores['triple_straddle_analysis'] = {
                'score': triple_straddle_result['triple_straddle_score'],
                'confidence': triple_straddle_result['confidence'],
                'details': triple_straddle_result
            }
            
            # 2. Enhanced Greek Sentiment
            if self.greek_sentiment:
                greek_result = self._analyze_greek_sentiment(market_data)
                indicator_scores['greek_sentiment'] = greek_result
            else:
                indicator_scores['greek_sentiment'] = {'score': 0.0, 'confidence': 0.5}
            
            # 3. Advanced OI Analysis
            if self.oi_analysis:
                oi_result = self._analyze_oi_patterns(market_data)
                indicator_scores['oi_analysis'] = oi_result
            else:
                indicator_scores['oi_analysis'] = {'score': 0.0, 'confidence': 0.5}
            
            # 4. IV Skew Analysis
            if self.iv_skew:
                iv_skew_result = self._analyze_iv_skew(market_data)
                indicator_scores['iv_skew'] = iv_skew_result
            else:
                indicator_scores['iv_skew'] = {'score': 0.0, 'confidence': 0.5}
            
            # 5. ATR/Premium Analysis
            atr_premium_result = self._analyze_atr_premium(market_data)
            indicator_scores['atr_premium'] = atr_premium_result
            
            # 6. Supporting Technical Analysis
            supporting_technical_result = self._analyze_supporting_technical(market_data)
            indicator_scores['supporting_technical'] = supporting_technical_result
            
        except Exception as e:
            logger.error(f"Error calculating indicator scores: {e}")
            # Return default scores
            for indicator in ['triple_straddle_analysis', 'greek_sentiment', 'oi_analysis', 
                            'iv_skew', 'atr_premium', 'supporting_technical']:
                indicator_scores[indicator] = {'score': 0.0, 'confidence': 0.5}
        
        return indicator_scores
    
    def _analyze_greek_sentiment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Greek sentiment using enhanced local system"""
        try:
            # Use the enhanced local Greek sentiment analyzer
            sentiment_result = self.greek_sentiment.analyze_greek_sentiment(market_data)

            # Extract score and confidence
            sentiment_score = sentiment_result.get('sentiment_score', 0.0)
            confidence = sentiment_result.get('confidence', 0.5)
            sentiment_type = sentiment_result.get('sentiment_type', 'Neutral')

            return {
                'score': sentiment_score,
                'confidence': confidence,
                'sentiment_classification': sentiment_type,
                'details': sentiment_result
            }

        except Exception as e:
            logger.error(f"Error analyzing Greek sentiment: {e}")
            return {'score': 0.0, 'confidence': 0.5}
    
    def _analyze_oi_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze OI patterns using enhanced local system"""
        try:
            # Use the enhanced local OI analyzer
            oi_result = self.oi_analysis.analyze_trending_oi_pa(market_data)

            # Extract pattern information
            oi_signal = oi_result.get('oi_signal', 0.0)
            confidence = oi_result.get('confidence', 0.5)
            pattern_breakdown = oi_result.get('pattern_breakdown', {})

            return {
                'score': oi_signal,
                'confidence': confidence,
                'pattern_breakdown': pattern_breakdown,
                'details': oi_result
            }

        except Exception as e:
            logger.error(f"Error analyzing OI patterns: {e}")
            return {'score': 0.0, 'confidence': 0.5}
    
    def _analyze_iv_skew(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze IV skew using enhanced system"""
        try:
            # Convert market data for IV skew analysis
            iv_df = self._convert_to_iv_dataframe(market_data)
            
            if iv_df.empty:
                return {'score': 0.0, 'confidence': 0.5}
            
            # Analyze IV skew
            skew_result = self.iv_skew.analyze_iv_skew(iv_df)
            
            if skew_result.empty:
                return {'score': 0.0, 'confidence': 0.5}
            
            # Extract skew information
            latest_result = skew_result.iloc[-1]
            skew_score = latest_result.get('IV_Skew_Score', 0.0)
            confidence = latest_result.get('IV_Skew_Confidence', 0.5)
            
            return {
                'score': np.clip(skew_score, -1.0, 1.0),
                'confidence': confidence,
                'skew_type': latest_result.get('IV_Skew_Type', 'Neutral'),
                'details': latest_result.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing IV skew: {e}")
            return {'score': 0.0, 'confidence': 0.5}
    
    def _analyze_atr_premium(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ATR and premium indicators"""
        try:
            # Calculate ATR-based volatility score
            price_data = market_data.get('price_history', [])
            if len(price_data) < 20:
                atr_score = 0.0
            else:
                # Simple ATR calculation
                highs = [p.get('high', 0) for p in price_data[-20:]]
                lows = [p.get('low', 0) for p in price_data[-20:]]
                closes = [p.get('close', 0) for p in price_data[-20:]]
                
                true_ranges = []
                for i in range(1, len(highs)):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    )
                    true_ranges.append(tr)
                
                atr = np.mean(true_ranges) if true_ranges else 0
                current_price = closes[-1] if closes else 1
                atr_percentage = atr / current_price if current_price > 0 else 0
                
                # Convert ATR to score
                if atr_percentage > 0.03:  # High volatility
                    atr_score = 0.5
                elif atr_percentage > 0.015:  # Medium volatility
                    atr_score = 0.0
                else:  # Low volatility
                    atr_score = -0.5
            
            # Calculate premium-based score
            options_data = market_data.get('options_data', {})
            if options_data:
                # Simple premium analysis based on ATM straddle
                atm_premium_score = self._calculate_premium_score(options_data)
            else:
                atm_premium_score = 0.0
            
            # Combine ATR and premium scores
            combined_score = 0.6 * atr_score + 0.4 * atm_premium_score
            
            return {
                'score': np.clip(combined_score, -1.0, 1.0),
                'confidence': 0.7,
                'atr_score': atr_score,
                'premium_score': atm_premium_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ATR/Premium: {e}")
            return {'score': 0.0, 'confidence': 0.5}
    
    def _analyze_supporting_technical(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supporting technical indicators"""
        try:
            # Simple technical analysis based on price action
            price_data = market_data.get('price_history', [])
            
            if len(price_data) < 50:
                return {'score': 0.0, 'confidence': 0.5}
            
            closes = [p.get('close', 0) for p in price_data[-50:]]
            
            # Simple moving average analysis
            ma_20 = np.mean(closes[-20:])
            ma_50 = np.mean(closes[-50:])
            current_price = closes[-1]
            
            # Technical score based on MA position
            if current_price > ma_20 > ma_50:
                technical_score = 0.5  # Bullish
            elif current_price < ma_20 < ma_50:
                technical_score = -0.5  # Bearish
            else:
                technical_score = 0.0  # Neutral
            
            return {
                'score': technical_score,
                'confidence': 0.6,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error analyzing supporting technical: {e}")
            return {'score': 0.0, 'confidence': 0.5}
    
    def _calculate_final_regime_score(self, indicator_scores: Dict[str, Any], 
                                    weights: Dict[str, Dict[str, float]]) -> float:
        """Calculate final regime score using dynamic weights"""
        try:
            indicator_weights = weights['indicator']
            
            final_score = 0.0
            total_weight = 0.0
            
            for indicator, weight in indicator_weights.items():
                if indicator in indicator_scores:
                    score = indicator_scores[indicator]['score']
                    final_score += score * weight
                    total_weight += weight
            
            return final_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating final regime score: {e}")
            return 0.0
    
    def _calculate_overall_confidence(self, indicator_scores: Dict[str, Any], 
                                    weights: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall confidence using dynamic weights"""
        try:
            indicator_weights = weights['indicator']
            
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for indicator, weight in indicator_weights.items():
                if indicator in indicator_scores:
                    confidence = indicator_scores[indicator]['confidence']
                    weighted_confidence += confidence * weight
                    total_weight += weight
            
            return weighted_confidence / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def _estimate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Estimate current market volatility"""
        try:
            # Use ATR as volatility proxy
            price_data = market_data.get('price_history', [])
            if len(price_data) < 10:
                return 0.15  # Default volatility
            
            closes = [p.get('close', 0) for p in price_data[-10:]]
            returns = [np.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            return min(1.0, max(0.05, volatility))
            
        except Exception as e:
            logger.error(f"Error estimating volatility: {e}")
            return 0.15
    
    def _calculate_performance_metrics(self, regime_type: Enhanced18RegimeType, 
                                     score: float, confidence: float, 
                                     market_data: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate performance metrics for the current prediction"""
        # Simplified performance metrics calculation
        return PerformanceMetrics(
            accuracy=confidence,  # Use confidence as proxy for accuracy
            precision=confidence * 0.9,
            recall=confidence * 0.8,
            f1_score=confidence * 0.85,
            confidence_avg=confidence,
            regime_stability=0.8,  # Default stability
            timestamp=datetime.now()
        )
    
    def _update_performance_tracking(self, result: EnhancedRegimeResult):
        """Update performance tracking with latest result"""
        self.performance_history.append(result.performance_metrics)
        self.regime_history.append(result)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
    
    def _trigger_weight_optimization_if_needed(self):
        """Trigger weight optimization if conditions are met"""
        # Trigger optimization every 100 predictions
        if len(self.performance_history) % 100 == 0 and len(self.performance_history) >= 100:
            try:
                market_conditions = self._get_current_market_conditions()
                optimization_result = self.weight_optimizer.optimize_weights(
                    self.performance_history[-50:], market_conditions
                )
                
                if optimization_result.validation_passed:
                    self.current_weights = optimization_result.optimized_weights
                    logger.info(f"Weights optimized with {optimization_result.performance_improvement:.3f} improvement")
                
            except Exception as e:
                logger.error(f"Error in weight optimization: {e}")
    
    def _get_current_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions for weight optimization"""
        if not self.regime_history:
            return {'volatility': 0.15}
        
        recent_regimes = self.regime_history[-10:]
        volatility = np.mean([r.performance_metrics.regime_stability for r in recent_regimes])
        
        return {
            'volatility': 1.0 - volatility,  # Lower stability = higher volatility
            'time_of_day': datetime.now().hour
        }
    
    # Helper methods for data conversion
    def _convert_to_options_dataframe(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert market data to options DataFrame for Greek analysis"""
        # Simplified conversion - would need actual implementation
        return pd.DataFrame()
    
    def _convert_to_oi_dataframe(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert market data to OI DataFrame"""
        # Simplified conversion - would need actual implementation
        return pd.DataFrame()
    
    def _convert_to_iv_dataframe(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert market data to IV DataFrame"""
        # Simplified conversion - would need actual implementation
        return pd.DataFrame()
    
    def _convert_oi_pattern_to_score(self, pattern: str) -> float:
        """Convert OI pattern to numerical score"""
        pattern_scores = {
            'Strong_Bullish': 1.0,
            'Mild_Bullish': 0.5,
            'Sideways_To_Bullish': 0.25,
            'Neutral': 0.0,
            'Sideways': 0.0,
            'Sideways_To_Bearish': -0.25,
            'Mild_Bearish': -0.5,
            'Strong_Bearish': -1.0
        }
        return pattern_scores.get(pattern, 0.0)
    
    def _calculate_premium_score(self, options_data: Dict[str, Any]) -> float:
        """Calculate premium-based score"""
        # Simplified premium analysis
        return 0.0
    
    def _get_default_result(self) -> EnhancedRegimeResult:
        """Get default result when detection fails"""
        return EnhancedRegimeResult(
            regime_type=Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL,
            regime_score=0.0,
            confidence=0.5,
            indicator_breakdown={},
            weights_applied=self.weight_optimizer.get_current_weights(),
            performance_metrics=PerformanceMetrics(
                accuracy=0.5, precision=0.5, recall=0.5, f1_score=0.5,
                confidence_avg=0.5, regime_stability=0.5, timestamp=datetime.now()
            ),
            timestamp=datetime.now()
        )
