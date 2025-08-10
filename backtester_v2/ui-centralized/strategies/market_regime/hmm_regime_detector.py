"""
Hidden Markov Model (HMM) Regime Detector

Implements HMM for market regime detection as specified in Excel configuration
with 35% weight in ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    logging.warning("hmmlearn not installed. Install with: pip install hmmlearn")
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)

class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection
    
    Uses Gaussian HMM to model market regimes based on returns and volatility
    """
    
    def __init__(self, n_regimes: int = 18, n_iter: int = 100):
        """
        Initialize HMM Regime Detector
        
        Args:
            n_regimes: Number of hidden states (18 for full regime system)
            n_iter: Number of iterations for EM algorithm
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.model = None
        self.is_fitted = False
        
        # Feature configuration
        self.feature_names = [
            'returns',
            'volatility', 
            'volume_ratio',
            'iv_level',
            'gamma_exposure',
            'oi_change_rate'
        ]
        
        # Initialize model if available
        if HMM_AVAILABLE:
            self._initialize_model()
        else:
            logger.error("HMM not available - hmmlearn package not installed")
    
    def _initialize_model(self):
        """Initialize the HMM model"""
        if not HMM_AVAILABLE:
            return
        
        # Gaussian HMM with full covariance
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
            init_params="stmc"  # Initialize all parameters
        )
        
        # Set initial transition matrix (slightly sticky states)
        trans_mat = np.full((self.n_regimes, self.n_regimes), 0.05)
        np.fill_diagonal(trans_mat, 0.95 - (self.n_regimes - 1) * 0.05)
        trans_mat = trans_mat / trans_mat.sum(axis=1)[:, np.newaxis]
        self.model.transmat_ = trans_mat
        
        logger.info(f"HMM initialized with {self.n_regimes} regimes")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Feature matrix for HMM
        """
        features = pd.DataFrame(index=data.index)
        
        # Price returns
        features['returns'] = data['underlying_spot'].pct_change().fillna(0)
        
        # Volatility (rolling std of returns)
        features['volatility'] = features['returns'].rolling(20).std().fillna(
            features['returns'].std()
        )
        
        # Volume ratio
        if 'volume' in data.columns:
            features['volume_ratio'] = (
                data['volume'] / data['volume'].rolling(20).mean()
            ).fillna(1)
        else:
            features['volume_ratio'] = 1
        
        # IV level (normalized VIX)
        if 'vix_close' in data.columns:
            features['iv_level'] = data['vix_close'] / 20
        else:
            features['iv_level'] = 1
        
        # Gamma exposure
        if 'total_gamma' in data.columns:
            features['gamma_exposure'] = data['total_gamma'].fillna(0)
        else:
            features['gamma_exposure'] = 0
        
        # OI change rate
        if 'total_oi' in data.columns:
            oi_change = data['total_oi'].pct_change().fillna(0)
            features['oi_change_rate'] = oi_change.rolling(5).mean().fillna(0)
        else:
            features['oi_change_rate'] = 0
        
        # Scale features
        feature_matrix = features[self.feature_names].values
        
        # Standardize features
        means = np.nanmean(feature_matrix, axis=0)
        stds = np.nanstd(feature_matrix, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        
        feature_matrix = (feature_matrix - means) / stds
        
        return feature_matrix
    
    def fit(self, data: pd.DataFrame):
        """
        Fit HMM model to historical data
        
        Args:
            data: Historical market data
        """
        if not HMM_AVAILABLE or self.model is None:
            logger.error("Cannot fit HMM - model not available")
            return
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Remove any NaN values
            valid_idx = ~np.any(np.isnan(features), axis=1)
            features_clean = features[valid_idx]
            
            if len(features_clean) < 100:
                logger.warning("Insufficient data for HMM fitting")
                return
            
            # Fit the model
            logger.info(f"Fitting HMM with {len(features_clean)} samples")
            self.model.fit(features_clean)
            
            self.is_fitted = True
            logger.info("HMM fitting completed")
            
            # Log transition matrix
            logger.debug(f"Transition matrix:\n{self.model.transmat_}")
            
        except Exception as e:
            logger.error(f"Error fitting HMM: {e}")
            self.is_fitted = False
    
    def predict_regime(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict current regime using HMM
        
        Args:
            data: Recent market data (at least 20 rows)
            
        Returns:
            Tuple of (regime_id, confidence)
        """
        if not HMM_AVAILABLE or not self.is_fitted:
            return 8, 0.5  # Default to neutral regime
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Get most recent valid features
            valid_features = features[~np.any(np.isnan(features), axis=1)]
            
            if len(valid_features) == 0:
                return 8, 0.5
            
            # Predict hidden states
            hidden_states = self.model.predict(valid_features)
            
            # Get posterior probabilities for confidence
            _, posteriors = self.model.score_samples(valid_features)
            
            # Current regime is the last predicted state
            current_regime = hidden_states[-1]
            
            # Confidence is the posterior probability of current state
            confidence = posteriors[-1, current_regime]
            
            return int(current_regime), float(confidence)
            
        except Exception as e:
            logger.error(f"Error predicting regime with HMM: {e}")
            return 8, 0.5
    
    def get_regime_sequence(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get full sequence of regime predictions
        
        Args:
            data: Market data
            
        Returns:
            DataFrame with regime predictions and probabilities
        """
        if not HMM_AVAILABLE or not self.is_fitted:
            return pd.DataFrame()
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Remove NaN
            valid_idx = ~np.any(np.isnan(features), axis=1)
            valid_features = features[valid_idx]
            valid_index = data.index[valid_idx]
            
            # Predict states
            hidden_states = self.model.predict(valid_features)
            
            # Get posterior probabilities
            _, posteriors = self.model.score_samples(valid_features)
            
            # Create results DataFrame
            results = pd.DataFrame(index=valid_index)
            results['hmm_regime'] = hidden_states
            results['hmm_confidence'] = [
                posteriors[i, hidden_states[i]] for i in range(len(hidden_states))
            ]
            
            # Add regime probabilities for top 3 regimes
            for i in range(min(3, self.n_regimes)):
                results[f'regime_{i}_prob'] = posteriors[:, i]
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting regime sequence: {e}")
            return pd.DataFrame()
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the learned transition matrix"""
        if self.is_fitted and self.model is not None:
            return self.model.transmat_
        return np.array([])
    
    def get_regime_means(self) -> np.ndarray:
        """Get the learned regime means"""
        if self.is_fitted and self.model is not None:
            return self.model.means_
        return np.array([])
    
    def get_regime_characteristics(self) -> Dict[int, Dict[str, float]]:
        """
        Get characteristics of each regime based on learned parameters
        
        Returns:
            Dictionary mapping regime ID to characteristics
        """
        if not self.is_fitted or self.model is None:
            return {}
        
        characteristics = {}
        means = self.model.means_
        
        for regime_id in range(self.n_regimes):
            regime_means = means[regime_id]
            
            characteristics[regime_id] = {
                'avg_return': regime_means[0] if len(regime_means) > 0 else 0,
                'avg_volatility': regime_means[1] if len(regime_means) > 1 else 0,
                'avg_volume_ratio': regime_means[2] if len(regime_means) > 2 else 1,
                'avg_iv_level': regime_means[3] if len(regime_means) > 3 else 1,
                'avg_gamma': regime_means[4] if len(regime_means) > 4 else 0,
                'avg_oi_change': regime_means[5] if len(regime_means) > 5 else 0
            }
        
        return characteristics
    
    def save_model(self, filepath: str):
        """Save fitted HMM model"""
        if not self.is_fitted:
            logger.warning("Model not fitted, cannot save")
            return
        
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"HMM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load fitted HMM model"""
        import joblib
        try:
            self.model = joblib.load(filepath)
            self.is_fitted = True
            logger.info(f"HMM model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading HMM model: {e}")
            self.is_fitted = False


# Example usage
if __name__ == "__main__":
    # Check if HMM is available
    if HMM_AVAILABLE:
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        
        # Simulate regime-switching data
        regimes = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
        
        # Different characteristics for each regime
        returns = np.zeros(1000)
        volatility = np.zeros(1000)
        
        for i in range(1000):
            if regimes[i] == 0:  # Low vol regime
                returns[i] = np.random.normal(0.0001, 0.005)
                volatility[i] = 0.005
            elif regimes[i] == 1:  # Medium vol regime
                returns[i] = np.random.normal(0.0002, 0.01)
                volatility[i] = 0.01
            else:  # High vol regime
                returns[i] = np.random.normal(-0.0001, 0.02)
                volatility[i] = 0.02
        
        # Create DataFrame
        data = pd.DataFrame({
            'underlying_spot': 100 * (1 + returns).cumprod(),
            'volume': np.random.poisson(1000, 1000),
            'vix_close': volatility * 100 * np.sqrt(252),
            'total_gamma': np.random.normal(0, 0.1, 1000),
            'total_oi': np.random.poisson(10000, 1000).cumsum()
        }, index=dates)
        
        # Test HMM
        hmm_detector = HMMRegimeDetector(n_regimes=3)
        
        # Fit model
        hmm_detector.fit(data[:800])
        
        # Predict on test data
        regime, confidence = hmm_detector.predict_regime(data[800:850])
        print(f"Predicted regime: {regime}, Confidence: {confidence:.2f}")
        
        # Get full sequence
        regime_sequence = hmm_detector.get_regime_sequence(data[800:])
        print("\nRegime sequence (first 10):")
        print(regime_sequence.head(10))
        
        # Get regime characteristics
        characteristics = hmm_detector.get_regime_characteristics()
        print("\nRegime characteristics:")
        for regime_id, chars in characteristics.items():
            print(f"Regime {regime_id}: {chars}")
    else:
        print("HMM not available. Install hmmlearn package.")