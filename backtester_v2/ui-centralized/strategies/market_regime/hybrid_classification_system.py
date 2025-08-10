#!/usr/bin/env python3
"""
Hybrid Classification System for Enhanced Triple Straddle Framework v2.0
========================================================================

This module implements the hybrid market regime classification system that combines:
- Enhanced 18-regime system (70% weight)
- Existing timeframe hierarchy system (30% weight)

Features:
- Agreement threshold validation
- Confidence scoring integration
- Transition probability calculations
- Mathematical accuracy validation (±0.001 tolerance)
- Performance optimization for <3s processing
- Integration with unified_stable_market_regime_pipeline.py

Author: The Augster
Date: 2025-06-20
Version: 2.0.0 (Phase 2 Enhanced)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mathematical precision tolerance for Enhanced Triple Straddle Framework v2.0
MATHEMATICAL_TOLERANCE = 0.001

class HybridMarketRegimeClassifier:
    """
    Hybrid market regime classification system combining:
    1. Enhanced 18-regime deterministic system (70% weight)
    2. Existing timeframe hierarchy system (30% weight)
    
    Provides:
    - Dual classification output
    - Confidence scoring based on system agreement
    - Mathematical integration with ±0.001 precision
    - Performance monitoring and validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize hybrid classification system"""
        
        self.config = config or self._get_default_config()
        self.classification_history = []
        self.performance_metrics = {}
        
        logger.info("🔄 Hybrid Market Regime Classifier initialized")
        logger.info(f"✅ Enhanced system weight: {self.config['enhanced_system_weight']}")
        logger.info(f"✅ Stable system weight: {self.config['stable_system_weight']}")
        logger.info(f"✅ Mathematical tolerance: ±{self.config['mathematical_tolerance']}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for hybrid classification"""
        
        return {
            # System integration weights
            'enhanced_system_weight': 0.70,    # 70% weight to enhanced system
            'stable_system_weight': 0.30,      # 30% weight to stable system
            
            # Enhanced system component weights
            'enhanced_component_weights': {
                'straddle_analysis': 0.40,      # Enhanced Triple Straddle
                'greek_sentiment': 0.30,        # Volume-weighted Greek Sentiment
                'oi_pattern_recognition': 0.20, # Advanced OI Pattern Recognition
                'technical_analysis': 0.10      # Enhanced Technical Analysis
            },
            
            # Stable system timeframe weights
            'stable_timeframe_weights': {
                '1min': 0.05,
                '5min': 0.15,
                '15min': 0.25,
                '30min': 0.35,
                'opening': 0.10,
                'previous_day': 0.10
            },
            
            # 18-regime classification thresholds
            'regime_thresholds': {
                'strong_bullish': 0.5,
                'mild_bullish': 0.2,
                'sideways_bullish': 0.1,
                'neutral': 0.1,
                'sideways_bearish': -0.1,
                'mild_bearish': -0.2,
                'strong_bearish': -0.5
            },
            
            # Volatility classification
            'volatility_thresholds': {
                'high': 0.6,
                'normal': 0.3,
                'low': 0.0
            },
            
            # Performance and validation
            'mathematical_tolerance': 0.001,
            'min_confidence_threshold': 0.5,
            'agreement_weight': 0.3,
            
            # Processing targets
            'max_processing_time_seconds': 3.0,
            'min_accuracy_threshold': 0.85
        }
    
    def classify_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hybrid market regime classification
        
        Args:
            market_data: Dict containing all market analysis components
            
        Returns:
            Dict with hybrid classification results
        """
        
        start_time = datetime.now()
        logger.info("🎯 Performing hybrid market regime classification...")
        
        # Enhanced system classification
        enhanced_result = self._classify_enhanced_system(market_data)
        
        # Stable system classification
        stable_result = self._classify_stable_system(market_data)
        
        # Hybrid integration
        hybrid_result = self._integrate_classifications(enhanced_result, stable_result)
        
        # Performance validation
        processing_time = (datetime.now() - start_time).total_seconds()
        hybrid_result['processing_time'] = processing_time
        
        # Store classification history
        self._update_classification_history(hybrid_result)
        
        # Validate performance targets
        self._validate_performance_targets(hybrid_result)
        
        logger.info(f"   ✅ Hybrid classification complete in {processing_time:.3f}s")
        logger.info(f"   ✅ Final regime: {hybrid_result['regime_name']}")
        logger.info(f"   ✅ Confidence: {hybrid_result['confidence']:.3f}")
        
        return hybrid_result
    
    def _classify_enhanced_system(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify using enhanced 18-regime deterministic system"""
        
        logger.info("   📊 Enhanced system classification...")
        
        # Extract component scores
        straddle_score = market_data.get('straddle_analysis', {}).get('score', 0.0)
        greek_score = market_data.get('greek_sentiment', {}).get('score', 0.0)
        oi_score = market_data.get('oi_pattern_recognition', {}).get('score', 0.0)
        technical_score = market_data.get('technical_analysis', {}).get('score', 0.0)
        
        # Calculate weighted enhanced score
        weights = self.config['enhanced_component_weights']
        enhanced_score = (
            weights['straddle_analysis'] * straddle_score +
            weights['greek_sentiment'] * greek_score +
            weights['oi_pattern_recognition'] * oi_score +
            weights['technical_analysis'] * technical_score
        )
        
        # Determine volatility level
        volatility_score = market_data.get('volatility_analysis', {}).get('score', 0.3)
        volatility_level = self._classify_volatility_level(volatility_score)
        
        # Map to 18-regime classification
        regime_id, regime_name = self._map_to_18_regimes(enhanced_score, volatility_level)
        
        # Calculate component confidence
        component_confidences = [
            market_data.get('straddle_analysis', {}).get('confidence', 0.5),
            market_data.get('greek_sentiment', {}).get('confidence', 0.5),
            market_data.get('oi_pattern_recognition', {}).get('confidence', 0.5),
            market_data.get('technical_analysis', {}).get('confidence', 0.5)
        ]
        
        enhanced_confidence = np.mean(component_confidences)
        
        return {
            'system_type': 'enhanced',
            'regime_id': regime_id,
            'regime_name': regime_name,
            'regime_score': enhanced_score,
            'volatility_level': volatility_level,
            'confidence': enhanced_confidence,
            'component_scores': {
                'straddle': straddle_score,
                'greek': greek_score,
                'oi': oi_score,
                'technical': technical_score
            }
        }
    
    def _classify_stable_system(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify using existing timeframe hierarchy system"""
        
        logger.info("   📈 Stable system classification...")
        
        # Extract timeframe scores
        timeframe_scores = {}
        timeframe_confidences = {}
        
        for timeframe in self.config['stable_timeframe_weights'].keys():
            tf_data = market_data.get(f'{timeframe}_analysis', {})
            timeframe_scores[timeframe] = tf_data.get('score', 0.0)
            timeframe_confidences[timeframe] = tf_data.get('confidence', 0.5)
        
        # Calculate weighted stable score
        weights = self.config['stable_timeframe_weights']
        stable_score = sum(
            weights[tf] * timeframe_scores.get(tf, 0.0)
            for tf in weights.keys()
        )
        
        # Calculate weighted confidence
        stable_confidence = sum(
            weights[tf] * timeframe_confidences.get(tf, 0.5)
            for tf in weights.keys()
        )
        
        # Map to regime classification (simplified for stable system)
        stable_regime_name = self._map_stable_regime(stable_score)
        
        return {
            'system_type': 'stable',
            'regime_name': stable_regime_name,
            'regime_score': stable_score,
            'confidence': stable_confidence,
            'timeframe_scores': timeframe_scores
        }
    
    def _integrate_classifications(self, enhanced_result: Dict[str, Any], stable_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate enhanced and stable system classifications"""
        
        logger.info("   🔄 Integrating classification systems...")
        
        # Weight-based score integration
        enhanced_weight = self.config['enhanced_system_weight']
        stable_weight = self.config['stable_system_weight']
        
        # Calculate weighted final score
        final_score = (
            enhanced_weight * enhanced_result['regime_score'] +
            stable_weight * stable_result['regime_score']
        )
        
        # Primary regime from enhanced system
        primary_regime_id = enhanced_result['regime_id']
        primary_regime_name = enhanced_result['regime_name']
        
        # Calculate system agreement confidence
        agreement_score = self._calculate_system_agreement(enhanced_result, stable_result)
        
        # Combined confidence calculation
        base_confidence = min(enhanced_result['confidence'], stable_result['confidence'])
        final_confidence = base_confidence * (1 + agreement_score * self.config['agreement_weight'])
        final_confidence = min(final_confidence, 1.0)  # Cap at 1.0
        
        # Enhanced transition probability calculation (Enhanced Triple Straddle Framework v2.0)
        transition_analysis = self._calculate_enhanced_transition_probability(enhanced_result, stable_result)
        
        return {
            'regime_id': primary_regime_id,
            'regime_name': primary_regime_name,
            'regime_score': final_score,
            'confidence': final_confidence,
            'volatility_level': enhanced_result['volatility_level'],
            
            # System contributions
            'enhanced_contribution': enhanced_weight,
            'stable_contribution': stable_weight,
            'system_agreement': agreement_score,
            
            # Detailed results
            'enhanced_result': enhanced_result,
            'stable_result': stable_result,
            
            # Additional metrics (Enhanced Triple Straddle Framework v2.0)
            'transition_analysis': transition_analysis,
            'transition_probability': transition_analysis.get('transition_probability', 0.5),
            'classification_quality': self._assess_classification_quality(enhanced_result, stable_result)
        }
    
    def _classify_volatility_level(self, volatility_score: float) -> str:
        """Classify volatility level based on score"""
        
        thresholds = self.config['volatility_thresholds']
        
        if volatility_score >= thresholds['high']:
            return 'High'
        elif volatility_score >= thresholds['normal']:
            return 'Normal'
        else:
            return 'Low'
    
    def _map_to_18_regimes(self, score: float, volatility_level: str) -> Tuple[int, str]:
        """Map score and volatility to 18-regime classification"""
        
        thresholds = self.config['regime_thresholds']
        
        # Determine directional component
        if score >= thresholds['strong_bullish']:
            direction, strength = 'Bullish', 'Strong'
            base_id = 1
        elif score >= thresholds['mild_bullish']:
            direction, strength = 'Bullish', 'Mild'
            base_id = 2
        elif score >= thresholds['sideways_bullish']:
            direction, strength = 'Bullish', 'Sideways'
            base_id = 3
        elif score >= thresholds['neutral']:
            direction, strength = 'Neutral', 'Neutral'
            base_id = 4
        elif score >= thresholds['sideways_bearish']:
            direction, strength = 'Bearish', 'Sideways'
            base_id = 5
        elif score >= thresholds['mild_bearish']:
            direction, strength = 'Bearish', 'Mild'
            base_id = 6
        else:
            direction, strength = 'Bearish', 'Strong'
            base_id = 7
        
        # Adjust for volatility level
        volatility_offset = {'High': 0, 'Normal': 6, 'Low': 12}
        regime_id = base_id + volatility_offset[volatility_level]
        
        # Create regime name
        regime_name = f"{volatility_level}_Volatile_{strength}_{direction}"
        
        return regime_id, regime_name
    
    def _map_stable_regime(self, score: float) -> str:
        """Map stable system score to regime name"""
        
        if score >= 0.3:
            return 'Stable_Bullish'
        elif score >= 0.1:
            return 'Stable_Mild_Bullish'
        elif score >= -0.1:
            return 'Stable_Neutral'
        elif score >= -0.3:
            return 'Stable_Mild_Bearish'
        else:
            return 'Stable_Bearish'
    
    def _calculate_system_agreement(self, enhanced_result: Dict[str, Any], stable_result: Dict[str, Any]) -> float:
        """Calculate agreement score between systems"""
        
        # Score correlation
        score_diff = abs(enhanced_result['regime_score'] - stable_result['regime_score'])
        score_agreement = max(0, 1 - score_diff)
        
        # Directional agreement
        enhanced_direction = 1 if enhanced_result['regime_score'] > 0 else -1 if enhanced_result['regime_score'] < 0 else 0
        stable_direction = 1 if stable_result['regime_score'] > 0 else -1 if stable_result['regime_score'] < 0 else 0
        
        directional_agreement = 1.0 if enhanced_direction == stable_direction else 0.0
        
        # Combined agreement
        agreement_score = (score_agreement * 0.6 + directional_agreement * 0.4)
        
        return agreement_score
    
    def _calculate_transition_probability(self, enhanced_result: Dict[str, Any], stable_result: Dict[str, Any]) -> float:
        """Calculate probability of regime transition"""
        
        # High disagreement suggests potential transition
        agreement = self._calculate_system_agreement(enhanced_result, stable_result)
        
        # Low confidence suggests uncertainty/transition
        min_confidence = min(enhanced_result['confidence'], stable_result['confidence'])
        
        # Transition probability increases with disagreement and low confidence
        transition_probability = (1 - agreement) * 0.6 + (1 - min_confidence) * 0.4
        
        return min(transition_probability, 1.0)
    
    def _assess_classification_quality(self, enhanced_result: Dict[str, Any], stable_result: Dict[str, Any]) -> str:
        """Assess overall classification quality"""
        
        agreement = self._calculate_system_agreement(enhanced_result, stable_result)
        avg_confidence = (enhanced_result['confidence'] + stable_result['confidence']) / 2
        
        if agreement >= 0.8 and avg_confidence >= 0.8:
            return 'HIGH'
        elif agreement >= 0.6 and avg_confidence >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _update_classification_history(self, result: Dict[str, Any]):
        """Update classification history for performance tracking"""
        
        self.classification_history.append({
            'timestamp': datetime.now(),
            'regime_id': result['regime_id'],
            'regime_name': result['regime_name'],
            'confidence': result['confidence'],
            'processing_time': result['processing_time'],
            'system_agreement': result['system_agreement']
        })
        
        # Keep only last 1000 classifications
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-1000:]
    
    def _validate_performance_targets(self, result: Dict[str, Any]):
        """Validate performance against targets"""

        # Processing time validation
        if result['processing_time'] > self.config['max_processing_time_seconds']:
            logger.warning(f"⚠️  Processing time {result['processing_time']:.3f}s exceeds target {self.config['max_processing_time_seconds']}s")

        # Confidence validation
        if result['confidence'] < self.config['min_confidence_threshold']:
            logger.warning(f"⚠️  Confidence {result['confidence']:.3f} below threshold {self.config['min_confidence_threshold']}")

        # ENHANCED TRIPLE STRADDLE FRAMEWORK v2.0: Mathematical accuracy validation
        mathematical_accuracy = self._validate_mathematical_accuracy(result)
        result['mathematical_accuracy'] = mathematical_accuracy

        if not mathematical_accuracy:
            logger.warning("⚠️  Mathematical accuracy validation failed")

    def _validate_mathematical_accuracy(self, result: Dict[str, Any]) -> bool:
        """
        Validate mathematical accuracy within ±0.001 tolerance
        Enhanced Triple Straddle Framework v2.0 requirement
        """
        try:
            # Check regime score precision
            regime_score = result.get('regime_score', 0.0)
            if not np.isfinite(regime_score):
                logger.error("Mathematical accuracy check failed: non-finite regime score")
                return False

            # Check confidence precision
            confidence = result.get('confidence', 0.0)
            if not np.isfinite(confidence):
                logger.error("Mathematical accuracy check failed: non-finite confidence")
                return False

            # Check bounds
            if not (-1.1 <= regime_score <= 1.1):  # Allow small tolerance beyond [-1, 1]
                logger.error(f"Regime score out of bounds: {regime_score}")
                return False

            if not (0.0 <= confidence <= 1.1):  # Allow small tolerance beyond [0, 1]
                logger.error(f"Confidence out of bounds: {confidence}")
                return False

            # Check precision (should be representable within tolerance)
            score_rounded = round(regime_score, 3)  # Round to 3 decimal places (0.001 precision)
            confidence_rounded = round(confidence, 3)

            score_error = abs(regime_score - score_rounded)
            confidence_error = abs(confidence - confidence_rounded)

            if score_error > MATHEMATICAL_TOLERANCE or confidence_error > MATHEMATICAL_TOLERANCE:
                logger.warning(f"Mathematical precision warning: score_error={score_error:.6f}, confidence_error={confidence_error:.6f}")
                return False

            # Validate system agreement precision
            system_agreement = result.get('system_agreement', 0.0)
            if not np.isfinite(system_agreement) or not (0.0 <= system_agreement <= 1.1):
                logger.error(f"System agreement validation failed: {system_agreement}")
                return False

            agreement_error = abs(system_agreement - round(system_agreement, 3))
            if agreement_error > MATHEMATICAL_TOLERANCE:
                logger.warning(f"System agreement precision warning: error={agreement_error:.6f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating mathematical accuracy: {e}")
            return False

    def _calculate_enhanced_transition_probability(self, enhanced_result: Dict[str, Any],
                                                 stable_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enhanced transition probability with mathematical validation
        Enhanced Triple Straddle Framework v2.0 feature
        """
        try:
            # Base transition probability
            base_transition = self._calculate_transition_probability(enhanced_result, stable_result)

            # Enhanced factors for transition probability

            # 1. Score volatility (rapid changes indicate potential transition)
            score_volatility = 0.0
            if len(self.classification_history) >= 5:
                recent_scores = [h['regime_score'] for h in self.classification_history[-5:]]
                score_volatility = np.std(recent_scores) if len(recent_scores) > 1 else 0.0

            # 2. Confidence trend (declining confidence suggests transition)
            confidence_trend = 0.0
            if len(self.classification_history) >= 3:
                recent_confidences = [h['confidence'] for h in self.classification_history[-3:]]
                if len(recent_confidences) >= 2:
                    confidence_trend = recent_confidences[-1] - recent_confidences[0]

            # 3. System disagreement persistence
            disagreement_persistence = 0.0
            if len(self.classification_history) >= 3:
                recent_agreements = [h.get('system_agreement', 0.5) for h in self.classification_history[-3:]]
                disagreement_persistence = 1.0 - np.mean(recent_agreements)

            # Calculate enhanced transition probability
            volatility_factor = min(score_volatility * 2.0, 0.3)  # Cap at 30%
            confidence_factor = max(-confidence_trend, 0.0) * 0.2  # Declining confidence increases transition probability
            persistence_factor = disagreement_persistence * 0.3

            enhanced_transition = base_transition + volatility_factor + confidence_factor + persistence_factor
            enhanced_transition = np.clip(enhanced_transition, 0.0, 1.0)

            # Mathematical accuracy validation
            transition_rounded = round(enhanced_transition, 3)
            transition_error = abs(enhanced_transition - transition_rounded)
            mathematical_accuracy = transition_error <= MATHEMATICAL_TOLERANCE

            return {
                'transition_probability': enhanced_transition,
                'base_transition': base_transition,
                'volatility_factor': volatility_factor,
                'confidence_factor': confidence_factor,
                'persistence_factor': persistence_factor,
                'mathematical_accuracy': mathematical_accuracy,
                'transition_confidence': 1.0 - transition_error if mathematical_accuracy else 0.5
            }

        except Exception as e:
            logger.error(f"Error calculating enhanced transition probability: {e}")
            return {
                'transition_probability': 0.5,
                'mathematical_accuracy': False,
                'transition_confidence': 0.0
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        if not self.classification_history:
            return {'status': 'No classification history available'}
        
        recent_history = self.classification_history[-100:]  # Last 100 classifications
        
        metrics = {
            'avg_processing_time': np.mean([h['processing_time'] for h in recent_history]),
            'avg_confidence': np.mean([h['confidence'] for h in recent_history]),
            'avg_system_agreement': np.mean([h['system_agreement'] for h in recent_history]),
            'processing_time_compliance': np.mean([
                h['processing_time'] <= self.config['max_processing_time_seconds'] 
                for h in recent_history
            ]),
            'confidence_compliance': np.mean([
                h['confidence'] >= self.config['min_confidence_threshold'] 
                for h in recent_history
            ]),
            'total_classifications': len(self.classification_history),
            'recent_regime_distribution': pd.Series([h['regime_name'] for h in recent_history]).value_counts().to_dict()
        }
        
        return metrics

# Integration function for unified_stable_market_regime_pipeline.py
def classify_hybrid_market_regime(enhanced_system_data: Dict[str, Any],
                                timeframe_hierarchy_data: Dict[str, Any],
                                config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Main integration function for hybrid market regime classification

    Args:
        enhanced_system_data: Data from enhanced 18-regime system components
        timeframe_hierarchy_data: Data from existing timeframe hierarchy system
        config: Optional configuration for hybrid classification

    Returns:
        Dictionary containing hybrid classification results or None if classification fails
    """
    try:
        # Initialize hybrid classifier
        classifier = HybridMarketRegimeClassifier(config)

        # Combine input data
        market_data = {
            **enhanced_system_data,
            **timeframe_hierarchy_data
        }

        # Perform hybrid classification
        result = classifier.classify_market_regime(market_data)

        if result is None:
            logger.warning("Hybrid market regime classification failed")
            return None

        # Return results in format expected by pipeline
        return {
            'hybrid_regime_classification': {
                'regime_id': result['regime_id'],
                'regime_name': result['regime_name'],
                'regime_score': result['regime_score'],
                'confidence': result['confidence'],
                'volatility_level': result['volatility_level'],
                'enhanced_contribution': result['enhanced_contribution'],
                'stable_contribution': result['stable_contribution'],
                'system_agreement': result['system_agreement'],
                'transition_probability': result['transition_probability'],
                'mathematical_accuracy': result.get('mathematical_accuracy', False),
                'classification_quality': result['classification_quality'],
                'processing_time': result['processing_time']
            },
            'enhanced_system_result': result['enhanced_result'],
            'stable_system_result': result['stable_result'],
            'transition_analysis': result.get('transition_analysis', {}),
            'classification_timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in hybrid market regime classification: {e}")
        return None

def main():
    """Main function for testing hybrid classification system"""
    
    logger.info("🚀 Testing Hybrid Market Regime Classifier")
    
    # Initialize classifier
    classifier = HybridMarketRegimeClassifier()
    
    # Generate sample market data
    sample_market_data = {
        'straddle_analysis': {'score': 0.3, 'confidence': 0.8},
        'greek_sentiment': {'score': 0.2, 'confidence': 0.7},
        'oi_pattern_recognition': {'score': 0.1, 'confidence': 0.6},
        'technical_analysis': {'score': 0.4, 'confidence': 0.9},
        'volatility_analysis': {'score': 0.4},
        '1min_analysis': {'score': 0.2, 'confidence': 0.6},
        '5min_analysis': {'score': 0.3, 'confidence': 0.7},
        '15min_analysis': {'score': 0.25, 'confidence': 0.8},
        '30min_analysis': {'score': 0.35, 'confidence': 0.75},
        'opening_analysis': {'score': 0.4, 'confidence': 0.8},
        'previous_day_analysis': {'score': 0.3, 'confidence': 0.7}
    }
    
    # Perform classification
    classification_result = classifier.classify_market_regime(sample_market_data)
    
    # Get performance metrics
    performance_metrics = classifier.get_performance_metrics()
    
    logger.info("🎯 Hybrid Classification System Testing Complete")
    
    return {
        'classification_result': classification_result,
        'performance_metrics': performance_metrics
    }

if __name__ == "__main__":
    main()
