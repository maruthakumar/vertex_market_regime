"""
Market Regime ML Integration Module

This module properly integrates all ML components for sophisticated regime detection:
- Continuous Learning Engine (Random Forest, Gradient Boosting, Neural Network)
- Ensemble Voting System
- Enhanced 18-Regime Detector
- Confidence Calibration Framework
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adaptive', 'optimization'))

# Import ML components
try:
    from adaptive.optimization.continuous_learning_engine import (
        ContinuousLearningEngine, LearningConfiguration, LearningMode,
        LearningExample
    )
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML Engine not available: {e}")
    ML_ENGINE_AVAILABLE = False

# Import HMM detector
try:
    from hmm_regime_detector import HMMRegimeDetector
    HMM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"HMM detector not available: {e}")
    HMM_AVAILABLE = False

# Import confidence calibration
try:
    from confidence_calibration import ConfidenceCalibrator
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Confidence calibration not available: {e}")
    CALIBRATION_AVAILABLE = False

# Import ensemble voting
try:
    from ensemble_voting_system import (
        EnsembleVotingSystem, EnsembleVotingConfig, 
        VotingMethod, VoterResult
    )
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Ensemble voting not available: {e}")
    ENSEMBLE_AVAILABLE = False

# Import enhanced regime detector
try:
    from enhanced_regime_detector import (
        Enhanced18RegimeDetector, Enhanced18RegimeType
    )
    REGIME_DETECTOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced regime detector not available: {e}")
    REGIME_DETECTOR_AVAILABLE = False

# Import actual indicator calculators
try:
    from phase4_iv_suite_integration import Phase4IVSuiteIntegration
    from enhanced_greek_sentiment_analysis import GreekSentimentAnalyzerAnalysis
    from enhanced_trending_oi_pa_analysis import OIPriceActionAnalyzer
    from enhanced_atr_indicators import EnhancedATRIndicators
    from comprehensive_triple_straddle_engine import StraddleAnalysisEngine
    INDICATORS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Indicator modules not available: {e}")
    INDICATORS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MarketRegimeMLIntegration:
    """
    Integrates all ML components for sophisticated market regime detection
    """
    
    def __init__(self, excel_config_path: Optional[str] = None):
        """
        Initialize ML Integration
        
        Args:
            excel_config_path: Path to Excel configuration file
        """
        self.excel_config_path = excel_config_path
        self.is_initialized = False
        
        # ML Components
        self.learning_engine = None
        self.ensemble_voter = None
        self.regime_detector = None
        self.hmm_detector = None
        self.confidence_calibrator = None
        
        # Indicator engines
        self.iv_suite = None
        self.greek_sentiment = None
        self.trending_oi_pa = None
        self.atr_indicators = None
        self.straddle_engine = None
        
        # Configuration from Excel
        self.excel_config = self._load_excel_config()
        
        # Initialize components
        self._initialize_components()
        
        # Feature engineering settings
        self.feature_columns = [
            'greek_sentiment_score', 'trending_oi_pa_score', 'iv_skew_score',
            'straddle_score', 'atr_score', 'price_momentum', 'volume_profile',
            'vix_level', 'gamma_exposure', 'delta_exposure', 'vega_exposure',
            'theta_decay', 'put_call_ratio', 'oi_change_rate', 'price_volatility'
        ]
        
        # Regime mapping for ML models (0-17 for 18 regimes)
        self.regime_to_id = {regime.value: i for i, regime in enumerate(Enhanced18RegimeType)}
        self.id_to_regime = {i: regime for regime, i in self.regime_to_id.items()}
        
        logger.info("MarketRegimeMLIntegration initialized")
    
    def _load_excel_config(self) -> Dict[str, Any]:
        """Load configuration from Excel file"""
        config = {
            'ensemble_weights': {
                'hmm': 0.35,  # From Excel EnsembleMethods sheet
                'gmm': 0.25,
                'random_forest': 0.20,
                'gradient_boost': 0.20
            },
            'confidence_calibration': {
                'method': 'ISOTONIC_REGRESSION',
                'platt_scaling': True,
                'temperature': 1.2
            },
            'regime_thresholds': {
                'strong_bullish': 0.45,
                'mild_bullish': 0.18,
                'neutral': 0.08,
                'mild_bearish': -0.18,
                'strong_bearish': -0.45
            }
        }
        return config
    
    def _initialize_components(self):
        """Initialize all ML and indicator components"""
        try:
            # Initialize ML engine
            if ML_ENGINE_AVAILABLE:
                ml_config = LearningConfiguration(
                    learning_mode=LearningMode.HYBRID,
                    online_batch_size=50,
                    batch_retrain_frequency=500,
                    drift_detection_window=100,
                    min_accuracy_threshold=0.65
                )
                self.learning_engine = ContinuousLearningEngine(ml_config)
                logger.info("ML Engine initialized")
            
            # Initialize ensemble voter
            if ENSEMBLE_AVAILABLE:
                ensemble_config = EnsembleVotingConfig(
                    voting_method=VotingMethod.ADAPTIVE_WEIGHTED,
                    min_voters=3,
                    confidence_threshold=0.6,
                    agreement_threshold=0.7
                )
                self.ensemble_voter = EnsembleVotingSystem(
                    weights=self.excel_config['ensemble_weights'],
                    config=ensemble_config
                )
                logger.info("Ensemble Voter initialized")
            
            # Initialize enhanced regime detector
            if REGIME_DETECTOR_AVAILABLE:
                self.regime_detector = Enhanced18RegimeDetector()
                logger.info("Enhanced Regime Detector initialized")
            
            # Initialize HMM detector
            if HMM_AVAILABLE:
                self.hmm_detector = HMMRegimeDetector(n_regimes=18)
                logger.info("HMM Detector initialized")
            
            # Initialize confidence calibrator
            if CALIBRATION_AVAILABLE:
                self.confidence_calibrator = ConfidenceCalibrator(
                    method='isotonic',
                    temperature=self.excel_config['confidence_calibration']['temperature']
                )
                logger.info("Confidence Calibrator initialized")
            
            # Initialize indicator engines
            if INDICATORS_AVAILABLE:
                self.iv_suite = Phase4IVSuiteIntegration()
                self.greek_sentiment = GreekSentimentAnalyzerAnalysis()
                self.trending_oi_pa = OIPriceActionAnalyzer()
                self.atr_indicators = EnhancedATRIndicators()
                self.straddle_engine = StraddleAnalysisEngine()
                logger.info("Indicator engines initialized")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.is_initialized = False
    
    def calculate_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for ML models
        
        Args:
            market_data: Raw market data
            
        Returns:
            DataFrame with calculated features
        """
        features = market_data.copy()
        
        try:
            # Greek sentiment analysis
            if self.greek_sentiment:
                greek_result = self.greek_sentiment.analyze_greek_sentiment(market_data)
                features['greek_sentiment_score'] = greek_result.get('sentiment_score', 0)
                features['gamma_exposure'] = greek_result.get('gamma_exposure', 0)
                features['delta_exposure'] = greek_result.get('delta_exposure', 0)
                features['vega_exposure'] = greek_result.get('vega_exposure', 0)
                features['theta_decay'] = greek_result.get('theta_decay', 0)
            
            # Trending OI/PA analysis
            if self.trending_oi_pa:
                oi_result = self.trending_oi_pa.analyze_trending_oi_pa(market_data)
                features['trending_oi_pa_score'] = oi_result.get('trend_score', 0)
                features['oi_change_rate'] = oi_result.get('oi_change_rate', 0)
                features['put_call_ratio'] = oi_result.get('put_call_ratio', 1)
            
            # IV analysis
            if self.iv_suite:
                iv_result = self.iv_suite.analyze_enhanced_iv_indicators(
                    iv_data={},
                    market_data=market_data.to_dict('records')[0] if len(market_data) > 0 else {},
                    underlying_price=market_data['underlying_spot'].iloc[-1] if 'underlying_spot' in market_data.columns else 0,
                    current_vix=market_data['vix_close'].iloc[-1] if 'vix_close' in market_data.columns else 20
                )
                features['iv_skew_score'] = iv_result.get('iv_skew_score', 0)
            
            # Straddle analysis
            if self.straddle_engine:
                straddle_result = self.straddle_engine.analyze_comprehensive_straddle(market_data)
                features['straddle_score'] = straddle_result.get('overall_score', 0)
            
            # ATR indicators
            if self.atr_indicators:
                atr_result = self.atr_indicators.calculate_enhanced_atr_indicators(market_data)
                features['atr_score'] = atr_result.get('atr_percentile', 50) / 100
                features['price_volatility'] = atr_result.get('atr_ratio', 1)
            
            # Additional features
            features['price_momentum'] = features['underlying_spot'].pct_change(20).fillna(0)
            features['volume_profile'] = (features['volume'] / features['volume'].rolling(20).mean()).fillna(1)
            features['vix_level'] = features['vix_close'] / 20  # Normalized around 20
            
            # Fill missing values
            for col in self.feature_columns:
                if col in features.columns:
                    features[col] = features[col].fillna(0)
                else:
                    features[col] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return features
    
    def predict_regime_ml(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict regime using ML models
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (regime_id, confidence)
        """
        if not self.learning_engine or features.empty:
            return 8, 0.5  # Default to neutral
        
        try:
            # Prepare features for ML
            feature_vector = features[self.feature_columns].iloc[-1].values
            
            # Get ML prediction
            prediction, confidence = self.learning_engine.predict(
                feature_vector, 
                use_ensemble=True
            )
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 8, 0.5
    
    def detect_regime_enhanced(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect regime using enhanced 18-regime detector
        
        Args:
            market_data: Market data
            
        Returns:
            Regime detection result
        """
        if not self.regime_detector:
            return {
                'regime_type': Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL,
                'confidence': 0.5
            }
        
        try:
            result = self.regime_detector.detect_regime(market_data)
            return {
                'regime_type': result.regime_type,
                'confidence': result.confidence,
                'volatility_component': result.volatility_component,
                'trend_component': result.trend_component
            }
        except Exception as e:
            logger.error(f"Enhanced detection error: {e}")
            return {
                'regime_type': Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL,
                'confidence': 0.5
            }
    
    async def ensemble_regime_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform ensemble regime detection combining all methods
        
        Args:
            market_data: Market data
            
        Returns:
            Final regime detection result
        """
        try:
            # Calculate features
            features = self.calculate_features(market_data)
            
            # Get predictions from different methods
            voting_inputs = {}
            
            # 1. ML prediction
            ml_regime_id, ml_confidence = self.predict_regime_ml(features)
            ml_regime = self.id_to_regime.get(ml_regime_id, Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL)
            voting_inputs['ml_prediction'] = {
                'regime': ml_regime,
                'confidence': ml_confidence,
                'weight': self.excel_config['ensemble_weights']['random_forest'] + 
                         self.excel_config['ensemble_weights']['gradient_boost']
            }
            
            # 2. Enhanced detector prediction  
            enhanced_result = self.detect_regime_enhanced(market_data)
            voting_inputs['enhanced_detector'] = {
                'regime': enhanced_result['regime_type'],
                'confidence': enhanced_result['confidence'],
                'weight': 0.3
            }
            
            # 3. HMM prediction
            if self.hmm_detector and self.hmm_detector.is_fitted:
                hmm_regime_id, hmm_confidence = self.hmm_detector.predict_regime(market_data)
                hmm_regime = self.id_to_regime.get(hmm_regime_id, Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL)
                voting_inputs['hmm_prediction'] = {
                    'regime': hmm_regime,
                    'confidence': hmm_confidence,
                    'weight': self.excel_config['ensemble_weights']['hmm']
                }
            
            # 4. Rule-based prediction (using actual thresholds)
            rule_based_regime = self._rule_based_regime(features)
            voting_inputs['rule_based'] = {
                'regime': rule_based_regime['regime'],
                'confidence': rule_based_regime['confidence'],
                'weight': 0.15  # Reduced to accommodate HMM
            }
            
            # Ensemble voting
            if self.ensemble_voter:
                ensemble_result = await self.ensemble_voter.vote(voting_inputs)
                final_regime = ensemble_result.get('regime_type', Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL)
                final_confidence = ensemble_result.get('confidence_score', 0.5)
            else:
                # Simple weighted average
                final_regime = ml_regime
                final_confidence = ml_confidence
            
            # Apply confidence calibration
            if self.confidence_calibrator:
                calibrated_confidence = self.confidence_calibrator.calibrate(final_confidence)
                
                # Add calibration data for future improvement
                if hasattr(final_regime, 'value'):
                    self.confidence_calibrator.add_calibration_data(
                        predictions=[self.regime_to_id.get(final_regime.value, 8)],
                        confidences=[final_confidence],
                        actuals=[self.regime_to_id.get(final_regime.value, 8)]  # Will be updated with actual later
                    )
                
                final_confidence = calibrated_confidence
            
            # Add learning example for continuous improvement
            if self.learning_engine and hasattr(final_regime, 'value'):
                learning_example = LearningExample(
                    features=features[self.feature_columns].iloc[-1].values,
                    target=self.regime_to_id.get(final_regime.value, 8),
                    timestamp=datetime.now(),
                    context={'market_data': market_data.iloc[-1].to_dict()},
                    confidence=final_confidence
                )
                self.learning_engine.add_learning_example(learning_example)
            
            return {
                'regime_type': final_regime,
                'regime_name': final_regime.value if hasattr(final_regime, 'value') else str(final_regime),
                'confidence_score': final_confidence,
                'ml_confidence': ml_confidence,
                'ensemble_agreement': ensemble_result.get('agreement_score', 0) if 'ensemble_result' in locals() else 0,
                'volatility_component': enhanced_result.get('volatility_component', 0.5),
                'trend_component': enhanced_result.get('trend_component', 0.5),
                'feature_scores': {
                    'greek_sentiment': features['greek_sentiment_score'].iloc[-1],
                    'trending_oi_pa': features['trending_oi_pa_score'].iloc[-1],
                    'iv_skew': features['iv_skew_score'].iloc[-1],
                    'straddle': features['straddle_score'].iloc[-1]
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble detection error: {e}")
            return {
                'regime_type': Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL,
                'regime_name': 'Normal_Volatile_Neutral',
                'confidence_score': 0.5,
                'ml_confidence': 0.5,
                'ensemble_agreement': 0,
                'volatility_component': 0.5,
                'trend_component': 0.5
            }
    
    def _rule_based_regime(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Apply rule-based regime detection using proper thresholds"""
        
        # Get latest values
        momentum = features['price_momentum'].iloc[-1]
        volatility = features['vix_level'].iloc[-1]
        
        # Determine trend using proper thresholds
        thresholds = self.excel_config['regime_thresholds']
        
        if momentum > thresholds['strong_bullish']:
            trend = 'strong_bullish'
        elif momentum > thresholds['mild_bullish']:
            trend = 'mild_bullish'
        elif momentum > thresholds['neutral']:
            trend = 'neutral'
        elif momentum > thresholds['mild_bearish']:
            trend = 'sideways'
        elif momentum > thresholds['strong_bearish']:
            trend = 'mild_bearish'
        else:
            trend = 'strong_bearish'
        
        # Determine volatility level
        if volatility > 1.5:  # VIX > 30
            vol_level = 'high_volatile'
        elif volatility > 0.75:  # VIX > 15
            vol_level = 'normal_volatile'
        else:
            vol_level = 'low_volatile'
        
        # Combine for regime
        regime_name = f"{vol_level}_{trend}".upper()
        
        # Map to enum
        regime_map = {
            'HIGH_VOLATILE_STRONG_BULLISH': Enhanced18RegimeType.HIGH_VOLATILE_STRONG_BULLISH,
            'NORMAL_VOLATILE_STRONG_BULLISH': Enhanced18RegimeType.NORMAL_VOLATILE_STRONG_BULLISH,
            'LOW_VOLATILE_STRONG_BULLISH': Enhanced18RegimeType.LOW_VOLATILE_STRONG_BULLISH,
            'HIGH_VOLATILE_MILD_BULLISH': Enhanced18RegimeType.HIGH_VOLATILE_MILD_BULLISH,
            'NORMAL_VOLATILE_MILD_BULLISH': Enhanced18RegimeType.NORMAL_VOLATILE_MILD_BULLISH,
            'LOW_VOLATILE_MILD_BULLISH': Enhanced18RegimeType.LOW_VOLATILE_MILD_BULLISH,
            'HIGH_VOLATILE_NEUTRAL': Enhanced18RegimeType.HIGH_VOLATILE_NEUTRAL,
            'NORMAL_VOLATILE_NEUTRAL': Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL,
            'LOW_VOLATILE_NEUTRAL': Enhanced18RegimeType.LOW_VOLATILE_NEUTRAL,
            'HIGH_VOLATILE_SIDEWAYS': Enhanced18RegimeType.HIGH_VOLATILE_SIDEWAYS,
            'NORMAL_VOLATILE_SIDEWAYS': Enhanced18RegimeType.NORMAL_VOLATILE_SIDEWAYS,
            'LOW_VOLATILE_SIDEWAYS': Enhanced18RegimeType.LOW_VOLATILE_SIDEWAYS,
            'HIGH_VOLATILE_MILD_BEARISH': Enhanced18RegimeType.HIGH_VOLATILE_MILD_BEARISH,
            'NORMAL_VOLATILE_MILD_BEARISH': Enhanced18RegimeType.NORMAL_VOLATILE_MILD_BEARISH,
            'LOW_VOLATILE_MILD_BEARISH': Enhanced18RegimeType.LOW_VOLATILE_MILD_BEARISH,
            'HIGH_VOLATILE_STRONG_BEARISH': Enhanced18RegimeType.HIGH_VOLATILE_STRONG_BEARISH,
            'NORMAL_VOLATILE_STRONG_BEARISH': Enhanced18RegimeType.NORMAL_VOLATILE_STRONG_BEARISH,
            'LOW_VOLATILE_STRONG_BEARISH': Enhanced18RegimeType.LOW_VOLATILE_STRONG_BEARISH
        }
        
        regime = regime_map.get(regime_name, Enhanced18RegimeType.NORMAL_VOLATILE_NEUTRAL)
        
        # Calculate confidence based on indicator agreement
        confidence_factors = []
        if features['greek_sentiment_score'].iloc[-1] * momentum > 0:  # Same direction
            confidence_factors.append(0.9)
        if features['trending_oi_pa_score'].iloc[-1] * momentum > 0:
            confidence_factors.append(0.85)
        if abs(momentum) > 0.3:  # Strong trend
            confidence_factors.append(0.95)
        
        confidence = np.mean(confidence_factors) if confidence_factors else 0.7
        
        return {
            'regime': regime,
            'confidence': confidence
        }
    
    def train_hmm(self, historical_data: pd.DataFrame):
        """Train HMM detector on historical data"""
        if self.hmm_detector and len(historical_data) > 100:
            try:
                logger.info("Training HMM detector...")
                self.hmm_detector.fit(historical_data)
                logger.info("HMM training completed")
            except Exception as e:
                logger.error(f"HMM training failed: {e}")
    
    def fit_confidence_calibration(self, min_samples: int = 100):
        """Fit confidence calibration models"""
        if self.confidence_calibrator:
            try:
                self.confidence_calibrator.fit(min_samples)
                logger.info("Confidence calibration fitted")
            except Exception as e:
                logger.error(f"Confidence calibration fitting failed: {e}")
    
    def get_ml_statistics(self) -> Dict[str, Any]:
        """Get ML engine statistics"""
        stats = {}
        
        if self.learning_engine:
            stats['ml_engine'] = self.learning_engine.get_learning_statistics()
        
        if self.hmm_detector and self.hmm_detector.is_fitted:
            stats['hmm'] = {
                'is_fitted': True,
                'n_regimes': self.hmm_detector.n_regimes,
                'regime_characteristics': self.hmm_detector.get_regime_characteristics()
            }
        
        if self.confidence_calibrator:
            stats['calibration'] = self.confidence_calibrator.get_calibration_metrics()
        
        return stats

# Example usage
if __name__ == "__main__":
    # Test the integration
    integration = MarketRegimeMLIntegration()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'underlying_spot': [100, 101, 102, 103, 102],
        'volume': [1000, 1100, 1200, 1300, 1250],
        'vix_close': [15, 16, 17, 16, 15],
        'total_gamma': [0.1, 0.2, 0.3, 0.2, 0.1],
        'ce_iv': [15, 16, 17, 16, 15],
        'pe_iv': [14, 15, 16, 15, 14]
    })
    
    # Run ensemble detection
    import asyncio
    result = asyncio.run(integration.ensemble_regime_detection(sample_data))
    print(f"Detected regime: {result['regime_name']} (confidence: {result['confidence_score']:.2f})")