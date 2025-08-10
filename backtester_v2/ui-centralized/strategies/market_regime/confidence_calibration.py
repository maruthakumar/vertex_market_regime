"""
Confidence Calibration Module

Implements confidence calibration methods:
- Isotonic Regression
- Platt Scaling
- Temperature Scaling
- Histogram Binning

As specified in Excel ConfidenceCalibration sheet
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ConfidenceCalibrator:
    """
    Implements various confidence calibration methods for regime predictions
    """
    
    def __init__(self, method: str = 'isotonic', temperature: float = 1.2):
        """
        Initialize Confidence Calibrator
        
        Args:
            method: Calibration method ('isotonic', 'platt', 'temperature', 'histogram')
            temperature: Temperature parameter for temperature scaling
        """
        self.method = method
        self.temperature = temperature
        
        # Calibration models
        self.isotonic_model = None
        self.platt_model = None
        self.histogram_bins = None
        self.bin_calibrations = None
        
        # Calibration data
        self.calibration_data = []
        self.is_calibrated = False
        
        # Configuration from Excel
        self.config = {
            'confidence_floor': 0.6,     # From Excel
            'confidence_ceiling': 0.99,  # From Excel
            'calibration_bins': 20,      # From Excel
            'platt_scaling_enabled': True,  # From Excel
            'isotonic_regression_enabled': True  # Implied from Excel
        }
        
        logger.info(f"ConfidenceCalibrator initialized with method: {method}")
    
    def add_calibration_data(self, predictions: np.ndarray, 
                           confidences: np.ndarray, 
                           actuals: np.ndarray):
        """
        Add data for calibration
        
        Args:
            predictions: Predicted regime classes
            confidences: Raw confidence scores
            actuals: Actual regime classes
        """
        for pred, conf, actual in zip(predictions, confidences, actuals):
            self.calibration_data.append({
                'prediction': pred,
                'confidence': conf,
                'actual': actual,
                'correct': pred == actual
            })
    
    def fit(self, min_samples: int = 100):
        """
        Fit calibration models
        
        Args:
            min_samples: Minimum samples required for calibration
        """
        if len(self.calibration_data) < min_samples:
            logger.warning(f"Insufficient calibration data: {len(self.calibration_data)} < {min_samples}")
            return
        
        # Convert to arrays
        data_df = pd.DataFrame(self.calibration_data)
        confidences = data_df['confidence'].values
        correct = data_df['correct'].values
        
        try:
            if self.method == 'isotonic' or self.config['isotonic_regression_enabled']:
                self._fit_isotonic(confidences, correct)
            
            if self.method == 'platt' or self.config['platt_scaling_enabled']:
                self._fit_platt(confidences, correct)
            
            if self.method == 'temperature':
                self._fit_temperature(confidences, correct)
            
            if self.method == 'histogram':
                self._fit_histogram(confidences, correct)
            
            self.is_calibrated = True
            logger.info(f"Calibration completed with {len(self.calibration_data)} samples")
            
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            self.is_calibrated = False
    
    def _fit_isotonic(self, confidences: np.ndarray, correct: np.ndarray):
        """Fit isotonic regression calibration"""
        try:
            self.isotonic_model = IsotonicRegression(
                y_min=0, 
                y_max=1,
                out_of_bounds='clip'
            )
            self.isotonic_model.fit(confidences, correct)
            logger.info("Isotonic regression calibration fitted")
        except Exception as e:
            logger.error(f"Isotonic regression fitting failed: {e}")
    
    def _fit_platt(self, confidences: np.ndarray, correct: np.ndarray):
        """Fit Platt scaling (sigmoid) calibration"""
        try:
            # Reshape for sklearn
            X = confidences.reshape(-1, 1)
            
            # Fit logistic regression
            self.platt_model = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            self.platt_model.fit(X, correct)
            logger.info("Platt scaling calibration fitted")
        except Exception as e:
            logger.error(f"Platt scaling fitting failed: {e}")
    
    def _fit_temperature(self, confidences: np.ndarray, correct: np.ndarray):
        """Fit temperature scaling"""
        try:
            # Temperature scaling uses a single parameter
            # Optimize temperature to minimize NLL
            from scipy.optimize import minimize_scalar
            
            def nll_loss(T):
                # Apply temperature scaling
                scaled = confidences / T
                # Clip to avoid numerical issues
                scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
                # Negative log likelihood
                nll = -np.mean(correct * np.log(scaled) + (1 - correct) * np.log(1 - scaled))
                return nll
            
            result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
            logger.info(f"Temperature scaling fitted: T={self.temperature:.3f}")
        except Exception as e:
            logger.error(f"Temperature scaling fitting failed: {e}")
    
    def _fit_histogram(self, confidences: np.ndarray, correct: np.ndarray):
        """Fit histogram binning calibration"""
        try:
            n_bins = self.config['calibration_bins']
            
            # Create bins
            self.histogram_bins = np.linspace(0, 1, n_bins + 1)
            self.bin_calibrations = {}
            
            # Calculate calibration for each bin
            for i in range(n_bins):
                bin_mask = (confidences >= self.histogram_bins[i]) & \
                          (confidences < self.histogram_bins[i + 1])
                
                if np.sum(bin_mask) > 0:
                    # Calibrated confidence is accuracy in this bin
                    self.bin_calibrations[i] = np.mean(correct[bin_mask])
                else:
                    # Use bin center as default
                    self.bin_calibrations[i] = (self.histogram_bins[i] + self.histogram_bins[i + 1]) / 2
            
            logger.info(f"Histogram calibration fitted with {n_bins} bins")
        except Exception as e:
            logger.error(f"Histogram calibration fitting failed: {e}")
    
    def calibrate(self, confidence: float, method: Optional[str] = None) -> float:
        """
        Calibrate a confidence score
        
        Args:
            confidence: Raw confidence score
            method: Override calibration method
            
        Returns:
            Calibrated confidence score
        """
        if not self.is_calibrated and method != 'temperature':
            # Can still use temperature scaling without fitting
            return self._apply_temperature_scaling(confidence)
        
        calibration_method = method or self.method
        
        if calibration_method == 'isotonic' and self.isotonic_model is not None:
            calibrated = self.isotonic_model.predict([confidence])[0]
        
        elif calibration_method == 'platt' and self.platt_model is not None:
            calibrated = self.platt_model.predict_proba([[confidence]])[0, 1]
        
        elif calibration_method == 'temperature':
            calibrated = self._apply_temperature_scaling(confidence)
        
        elif calibration_method == 'histogram' and self.bin_calibrations is not None:
            calibrated = self._apply_histogram_calibration(confidence)
        
        else:
            calibrated = confidence
        
        # Apply floor and ceiling
        calibrated = np.clip(
            calibrated,
            self.config['confidence_floor'],
            self.config['confidence_ceiling']
        )
        
        return float(calibrated)
    
    def _apply_temperature_scaling(self, confidence: float) -> float:
        """Apply temperature scaling to confidence"""
        # Apply temperature
        scaled = confidence / self.temperature
        
        # Convert to probability using sigmoid if needed
        if scaled > 1 or scaled < 0:
            # Use logit transform and sigmoid
            logit = np.log(confidence / (1 - confidence + 1e-7))
            scaled_logit = logit / self.temperature
            scaled = 1 / (1 + np.exp(-scaled_logit))
        
        return scaled
    
    def _apply_histogram_calibration(self, confidence: float) -> float:
        """Apply histogram binning calibration"""
        if self.histogram_bins is None or self.bin_calibrations is None:
            return confidence
        
        # Find bin
        bin_idx = np.digitize(confidence, self.histogram_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.bin_calibrations) - 1)
        
        return self.bin_calibrations.get(bin_idx, confidence)
    
    def calibrate_batch(self, confidences: np.ndarray, 
                       method: Optional[str] = None) -> np.ndarray:
        """
        Calibrate a batch of confidence scores
        
        Args:
            confidences: Array of raw confidence scores
            method: Override calibration method
            
        Returns:
            Array of calibrated confidence scores
        """
        return np.array([self.calibrate(conf, method) for conf in confidences])
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get calibration quality metrics"""
        if not self.calibration_data:
            return {}
        
        data_df = pd.DataFrame(self.calibration_data)
        confidences = data_df['confidence'].values
        correct = data_df['correct'].values
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(confidences, correct)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(confidences, correct)
        
        # Brier Score
        brier_score = np.mean((confidences - correct) ** 2)
        
        metrics = {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'n_samples': len(self.calibration_data),
            'accuracy': np.mean(correct),
            'avg_confidence': np.mean(confidences)
        }
        
        # Add calibration curve data
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                correct, confidences, n_bins=10, strategy='uniform'
            )
            metrics['calibration_curve'] = {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        except:
            pass
        
        return metrics
    
    def _calculate_ece(self, confidences: np.ndarray, correct: np.ndarray, 
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & \
                      (confidences < bin_boundaries[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(correct[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                bin_weight = np.sum(bin_mask) / len(confidences)
                
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _calculate_mce(self, confidences: np.ndarray, correct: np.ndarray,
                      n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0
        
        for i in range(n_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & \
                      (confidences < bin_boundaries[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(correct[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                
                mce = max(mce, abs(bin_accuracy - bin_confidence))
        
        return mce
    
    def save_calibration(self, filepath: str):
        """Save calibration models"""
        import joblib
        
        calibration_state = {
            'method': self.method,
            'temperature': self.temperature,
            'isotonic_model': self.isotonic_model,
            'platt_model': self.platt_model,
            'histogram_bins': self.histogram_bins,
            'bin_calibrations': self.bin_calibrations,
            'config': self.config,
            'is_calibrated': self.is_calibrated
        }
        
        joblib.dump(calibration_state, filepath)
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration models"""
        import joblib
        
        try:
            calibration_state = joblib.load(filepath)
            
            self.method = calibration_state['method']
            self.temperature = calibration_state['temperature']
            self.isotonic_model = calibration_state['isotonic_model']
            self.platt_model = calibration_state['platt_model']
            self.histogram_bins = calibration_state['histogram_bins']
            self.bin_calibrations = calibration_state['bin_calibrations']
            self.config = calibration_state['config']
            self.is_calibrated = calibration_state['is_calibrated']
            
            logger.info(f"Calibration loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")


# Example usage
if __name__ == "__main__":
    # Create calibrator
    calibrator = ConfidenceCalibrator(method='isotonic')
    
    # Generate synthetic calibration data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate overconfident predictions
    true_probs = np.random.beta(2, 2, n_samples)
    raw_confidences = np.clip(true_probs + np.random.normal(0, 0.1, n_samples), 0, 1)
    raw_confidences = raw_confidences ** 0.5  # Make overconfident
    
    # Generate outcomes
    outcomes = np.random.binomial(1, true_probs)
    predictions = (raw_confidences > 0.5).astype(int)
    
    # Add calibration data
    calibrator.add_calibration_data(predictions, raw_confidences, outcomes)
    
    # Fit calibration
    calibrator.fit()
    
    # Test calibration
    test_confidences = [0.3, 0.5, 0.7, 0.9, 0.95]
    print("Calibration results:")
    for conf in test_confidences:
        calibrated = calibrator.calibrate(conf)
        print(f"Raw: {conf:.2f} → Calibrated: {calibrated:.2f}")
    
    # Get metrics
    metrics = calibrator.get_calibration_metrics()
    print(f"\nCalibration metrics:")
    print(f"ECE: {metrics['ece']:.3f}")
    print(f"MCE: {metrics['mce']:.3f}")
    print(f"Brier Score: {metrics['brier_score']:.3f}")