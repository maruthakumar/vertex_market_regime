"""
Continuous Learning Engine

This module implements automated continuous learning capabilities for the
adaptive market regime formation system, enabling real-time adaptation
and improvement based on market conditions and performance feedback.

Key Features:
- Online learning algorithms for real-time adaptation
- Multi-model ensemble with dynamic weighting
- Concept drift detection and handling
- Automated model retraining and selection
- Knowledge transfer between different market conditions
- Performance-based learning rate adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import tempfile
import os

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for the engine"""
    ONLINE = "online"
    BATCH = "batch"
    HYBRID = "hybrid"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class ModelType(Enum):
    """Types of learning models"""
    SGD_CLASSIFIER = "sgd_classifier"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class ConceptDriftType(Enum):
    """Types of concept drift"""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"
    NONE = "none"


@dataclass
class LearningExample:
    """Individual learning example"""
    features: np.ndarray
    target: Any
    timestamp: datetime
    context: Dict[str, Any]
    weight: float = 1.0
    confidence: float = 1.0
    market_regime: Optional[int] = None


@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    model_id: str
    accuracy: float
    log_loss_score: float
    prediction_confidence: float
    stability: float
    recent_performance: List[float]
    training_time: float
    prediction_time: float
    last_updated: datetime
    sample_count: int


@dataclass
class ConceptDriftDetection:
    """Concept drift detection result"""
    drift_detected: bool
    drift_type: ConceptDriftType
    confidence: float
    affected_features: List[str]
    timestamp: datetime
    severity: float
    recommended_actions: List[str]


@dataclass
class LearningConfiguration:
    """Configuration for continuous learning"""
    # Learning modes
    learning_mode: LearningMode = LearningMode.HYBRID
    online_batch_size: int = 50
    batch_retrain_frequency: int = 1000
    
    # Model management
    max_models: int = 5
    model_selection_metric: str = "accuracy"
    ensemble_method: str = "weighted_voting"
    
    # Drift detection
    drift_detection_window: int = 200
    drift_threshold: float = 0.05
    drift_sensitivity: float = 0.1
    
    # Learning rates
    base_learning_rate: float = 0.01
    adaptive_learning_rate: bool = True
    lr_decay_factor: float = 0.95
    min_learning_rate: float = 0.001
    
    # Performance thresholds
    min_accuracy_threshold: float = 0.6
    performance_window: int = 100
    retraining_threshold: float = 0.05
    
    # Feature management
    feature_selection: bool = True
    max_features: int = 50
    feature_importance_threshold: float = 0.01


class ContinuousLearningEngine:
    """
    Continuous learning engine for adaptive regime formation
    """
    
    def __init__(self, config: Optional[LearningConfiguration] = None):
        """
        Initialize continuous learning engine
        
        Args:
            config: Learning configuration
        """
        self.config = config or LearningConfiguration()
        
        # Model management
        self.models: Dict[str, Any] = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.active_models: List[str] = []
        self.best_model_id: Optional[str] = None
        
        # Learning data
        self.learning_buffer = deque(maxlen=self.config.drift_detection_window * 2)
        self.feature_buffer = deque(maxlen=1000)
        self.target_buffer = deque(maxlen=1000)
        
        # Drift detection
        self.drift_detector = ConceptDriftDetector(
            window_size=self.config.drift_detection_window,
            threshold=self.config.drift_threshold,
            sensitivity=self.config.drift_sensitivity
        )
        
        # Feature management
        self.feature_scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self.selected_features: List[int] = []
        
        # Learning state
        self.total_examples = 0
        self.last_batch_training = datetime.now()
        self.last_model_update = datetime.now()
        self.learning_rate = self.config.base_learning_rate
        
        # Performance tracking
        self.accuracy_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=self.config.performance_window)
        
        # Knowledge base
        self.market_condition_models: Dict[str, str] = {}  # condition -> model_id
        self.transfer_learning_weights: Dict[str, np.ndarray] = {}
        
        # Initialize base models
        self._initialize_models()
        
        logger.info("ContinuousLearningEngine initialized")
    
    def _initialize_models(self):
        """Initialize the learning models"""
        
        # Online learning models
        self.models["sgd"] = SGDClassifier(
            learning_rate='adaptive',
            eta0=self.learning_rate,
            random_state=42
        )
        
        self.models["passive_aggressive"] = PassiveAggressiveClassifier(
            C=1.0,
            random_state=42
        )
        
        # Batch learning models (will be created on demand)
        self.batch_model_configs = {
            "random_forest": {
                "class": RandomForestClassifier,
                "params": {"n_estimators": 50, "random_state": 42, "n_jobs": -1}
            },
            "gradient_boosting": {
                "class": GradientBoostingClassifier,
                "params": {"n_estimators": 50, "random_state": 42}
            },
            "neural_network": {
                "class": MLPClassifier,
                "params": {"hidden_layer_sizes": (50, 25), "random_state": 42, "max_iter": 200}
            }
        }
        
        # Initialize model performances
        for model_id in self.models.keys():
            self.model_performances[model_id] = ModelPerformance(
                model_id=model_id,
                accuracy=0.0,
                log_loss_score=float('inf'),
                prediction_confidence=0.0,
                stability=0.0,
                recent_performance=[],
                training_time=0.0,
                prediction_time=0.0,
                last_updated=datetime.now(),
                sample_count=0
            )
        
        self.active_models = list(self.models.keys())
    
    def add_learning_example(self, example: LearningExample):
        """
        Add a new learning example to the system
        
        Args:
            example: Learning example to add
        """
        # Store example
        self.learning_buffer.append(example)
        self.feature_buffer.append(example.features)
        self.target_buffer.append(example.target)
        
        self.total_examples += 1
        
        # Check for concept drift
        if len(self.learning_buffer) >= self.config.drift_detection_window:
            drift_result = self.drift_detector.detect_drift(
                list(self.feature_buffer)[-self.config.drift_detection_window:],
                list(self.target_buffer)[-self.config.drift_detection_window:]
            )
            
            if drift_result.drift_detected:
                self._handle_concept_drift(drift_result)
        
        # Online learning update
        if self.config.learning_mode in [LearningMode.ONLINE, LearningMode.HYBRID]:
            self._update_online_models([example])
        
        # Batch learning check
        if (self.config.learning_mode in [LearningMode.BATCH, LearningMode.HYBRID] and
            self.total_examples % self.config.batch_retrain_frequency == 0):
            self._perform_batch_learning()
        
        # Model selection update
        if self.total_examples % self.config.online_batch_size == 0:
            self._update_model_selection()
    
    def _update_online_models(self, examples: List[LearningExample]):
        """Update online learning models with new examples"""
        
        start_time = datetime.now()
        
        for example in examples:
            features = example.features.reshape(1, -1)
            target = [example.target]
            
            # Scale features if scaler is fitted
            if hasattr(self.feature_scaler, 'scale_'):
                features = self.feature_scaler.transform(features)
            
            # Update online models
            for model_id in ["sgd", "passive_aggressive"]:
                if model_id in self.models:
                    try:
                        # Partial fit for online learning
                        if hasattr(self.models[model_id], 'partial_fit'):
                            classes = np.unique(list(self.target_buffer)) if len(self.target_buffer) > 0 else None
                            if classes is not None:
                                self.models[model_id].partial_fit(features, target, classes=classes)
                            else:
                                # First example, use a reasonable class range
                                classes = np.arange(0, 12)  # Assume 12 possible regimes
                                self.models[model_id].partial_fit(features, target, classes=classes)
                        
                        # Update performance
                        self._update_model_performance(model_id, features, target)
                        
                    except Exception as e:
                        logger.warning(f"Failed to update model {model_id}: {e}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Online model update completed in {training_time:.3f}s")
    
    def _perform_batch_learning(self):
        """Perform batch learning with accumulated examples"""
        
        logger.info("Starting batch learning...")
        start_time = datetime.now()
        
        # Prepare data
        if len(self.learning_buffer) < 100:
            logger.warning("Insufficient data for batch learning")
            return
        
        features = np.array([ex.features for ex in self.learning_buffer])
        targets = np.array([ex.target for ex in self.learning_buffer])
        weights = np.array([ex.weight for ex in self.learning_buffer])
        
        # Feature scaling
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Feature selection
        if self.config.feature_selection:
            self._update_feature_selection(features_scaled, targets)
            if self.selected_features:
                features_scaled = features_scaled[:, self.selected_features]
        
        # Create and train batch models
        for model_name, config in self.batch_model_configs.items():
            try:
                model = config["class"](**config["params"])
                model.fit(features_scaled, targets, sample_weight=weights)
                
                # Store model
                self.models[model_name] = model
                
                # Initialize performance if not exists
                if model_name not in self.model_performances:
                    self.model_performances[model_name] = ModelPerformance(
                        model_id=model_name,
                        accuracy=0.0,
                        log_loss_score=float('inf'),
                        prediction_confidence=0.0,
                        stability=0.0,
                        recent_performance=[],
                        training_time=0.0,
                        prediction_time=0.0,
                        last_updated=datetime.now(),
                        sample_count=0
                    )
                
                # Evaluate model
                predictions = model.predict(features_scaled)
                accuracy = accuracy_score(targets, predictions)
                
                self.model_performances[model_name].accuracy = accuracy
                self.model_performances[model_name].last_updated = datetime.now()
                self.model_performances[model_name].sample_count = len(targets)
                
                if model_name not in self.active_models:
                    self.active_models.append(model_name)
                
                logger.debug(f"Batch model {model_name} trained with accuracy: {accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train batch model {model_name}: {e}")
        
        self.last_batch_training = datetime.now()
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Batch learning completed in {training_time:.3f}s")
    
    def _update_feature_selection(self, features: np.ndarray, targets: np.ndarray):
        """Update feature selection based on importance"""
        
        try:
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=20, random_state=42)
            rf.fit(features, targets)
            
            importances = rf.feature_importances_
            
            # Select features above threshold
            important_features = []
            for i, importance in enumerate(importances):
                if importance >= self.config.feature_importance_threshold:
                    important_features.append(i)
            
            # Limit to max features
            if len(important_features) > self.config.max_features:
                # Sort by importance and take top features
                feature_importance_pairs = [(i, importances[i]) for i in important_features]
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                important_features = [pair[0] for pair in feature_importance_pairs[:self.config.max_features]]
            
            self.selected_features = important_features
            
            # Update feature importances
            for i, importance in enumerate(importances):
                feature_name = f"feature_{i}"
                self.feature_importances[feature_name] = importance
            
            logger.debug(f"Selected {len(important_features)} important features")
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            self.selected_features = list(range(min(features.shape[1], self.config.max_features)))
    
    def _update_model_performance(self, model_id: str, features: np.ndarray, targets: List[Any]):
        """Update performance metrics for a model"""
        
        if model_id not in self.models:
            return
        
        try:
            model = self.models[model_id]
            
            # Make prediction
            start_time = datetime.now()
            predictions = model.predict(features)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            accuracy = accuracy_score(targets, predictions)
            
            # Get prediction probabilities if available
            confidence = 0.5
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features)
                    confidence = np.mean(np.max(probabilities, axis=1))
                except:
                    pass
            
            # Update performance
            performance = self.model_performances[model_id]
            performance.accuracy = accuracy
            performance.prediction_confidence = confidence
            performance.prediction_time = prediction_time
            performance.last_updated = datetime.now()
            performance.sample_count += len(targets)
            
            # Update recent performance
            performance.recent_performance.append(accuracy)
            if len(performance.recent_performance) > self.config.performance_window:
                performance.recent_performance.pop(0)
            
            # Calculate stability
            if len(performance.recent_performance) >= 10:
                performance.stability = 1.0 - np.std(performance.recent_performance)
            
        except Exception as e:
            logger.warning(f"Failed to update performance for model {model_id}: {e}")
    
    def _update_model_selection(self):
        """Update model selection based on current performance"""
        
        if not self.active_models:
            return
        
        # Calculate combined scores for each model
        model_scores = {}
        
        for model_id in self.active_models:
            if model_id in self.model_performances:
                perf = self.model_performances[model_id]
                
                # Combined score: accuracy + stability + confidence
                score = (perf.accuracy * 0.5 + 
                        perf.stability * 0.3 + 
                        perf.prediction_confidence * 0.2)
                
                model_scores[model_id] = score
        
        # Select best model
        if model_scores:
            self.best_model_id = max(model_scores, key=model_scores.get)
            logger.debug(f"Best model selected: {self.best_model_id} (score: {model_scores[self.best_model_id]:.3f})")
    
    def predict(self, features: np.ndarray, use_ensemble: bool = True) -> Tuple[Any, float]:
        """
        Make prediction using the learning engine
        
        Args:
            features: Input features
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.active_models:
            logger.warning("No active models available for prediction")
            return 0, 0.0
        
        features = features.reshape(1, -1)
        
        # Scale features
        if hasattr(self.feature_scaler, 'scale_'):
            features_scaled = self.feature_scaler.transform(features)
        else:
            features_scaled = features
        
        # Apply feature selection
        if self.selected_features and features_scaled.shape[1] > len(self.selected_features):
            features_scaled = features_scaled[:, self.selected_features]
        
        if use_ensemble and len(self.active_models) > 1:
            return self._ensemble_predict(features_scaled)
        else:
            return self._single_model_predict(features_scaled)
    
    def _single_model_predict(self, features: np.ndarray) -> Tuple[Any, float]:
        """Make prediction using the best single model"""
        
        model_id = self.best_model_id or self.active_models[0]
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return 0, 0.0
        
        try:
            model = self.models[model_id]
            prediction = model.predict(features)[0]
            
            # Get confidence
            confidence = 0.5
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features)
                    confidence = np.max(probabilities[0])
                except:
                    pass
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_id}: {e}")
            return 0, 0.0
    
    def _ensemble_predict(self, features: np.ndarray) -> Tuple[Any, float]:
        """Make ensemble prediction using multiple models"""
        
        predictions = []
        confidences = []
        weights = []
        
        for model_id in self.active_models:
            if model_id in self.models:
                try:
                    model = self.models[model_id]
                    pred = model.predict(features)[0]
                    
                    # Get confidence and weight
                    confidence = 0.5
                    if hasattr(model, 'predict_proba'):
                        try:
                            probabilities = model.predict_proba(features)
                            confidence = np.max(probabilities[0])
                        except:
                            pass
                    
                    # Weight by model performance
                    model_weight = self.model_performances[model_id].accuracy
                    
                    predictions.append(pred)
                    confidences.append(confidence)
                    weights.append(model_weight)
                    
                except Exception as e:
                    logger.warning(f"Failed to get prediction from model {model_id}: {e}")
        
        if not predictions:
            return 0, 0.0
        
        # Weighted voting
        if self.config.ensemble_method == "weighted_voting":
            # Convert to numpy arrays
            predictions = np.array(predictions)
            weights = np.array(weights)
            confidences = np.array(confidences)
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            # Get unique predictions and their weighted votes
            unique_preds = np.unique(predictions)
            vote_scores = {}
            
            for pred in unique_preds:
                mask = predictions == pred
                vote_scores[pred] = np.sum(weights[mask] * confidences[mask])
            
            # Select prediction with highest weighted vote
            best_prediction = max(vote_scores, key=vote_scores.get)
            ensemble_confidence = vote_scores[best_prediction]
            
            return best_prediction, ensemble_confidence
        
        else:  # Simple majority voting
            from collections import Counter
            pred_counts = Counter(predictions)
            best_prediction = pred_counts.most_common(1)[0][0]
            ensemble_confidence = np.mean(confidences)
            
            return best_prediction, ensemble_confidence
    
    def _handle_concept_drift(self, drift_result: ConceptDriftDetection):
        """Handle detected concept drift"""
        
        logger.warning(f"Concept drift detected: {drift_result.drift_type.value} "
                      f"(confidence: {drift_result.confidence:.3f})")
        
        if drift_result.drift_type == ConceptDriftType.SUDDEN:
            # Aggressive retraining for sudden drift
            self._aggressive_retraining()
        elif drift_result.drift_type == ConceptDriftType.GRADUAL:
            # Adaptive learning rate adjustment
            self._adjust_learning_rate(factor=1.2)
        elif drift_result.drift_type == ConceptDriftType.INCREMENTAL:
            # Increase online learning frequency
            self.config.online_batch_size = max(10, self.config.online_batch_size // 2)
        
        # Update drift detection sensitivity
        if drift_result.severity > 0.7:
            self.drift_detector.sensitivity *= 0.9  # More sensitive
        else:
            self.drift_detector.sensitivity *= 1.1  # Less sensitive
    
    def _aggressive_retraining(self):
        """Perform aggressive retraining in response to sudden drift"""
        
        logger.info("Performing aggressive retraining due to concept drift")
        
        # Reset online models
        self._initialize_models()
        
        # Immediate batch retraining with recent data
        recent_examples = list(self.learning_buffer)[-self.config.batch_retrain_frequency//2:]
        if len(recent_examples) >= 50:
            # Temporarily store recent examples
            temp_buffer = self.learning_buffer.copy()
            self.learning_buffer.clear()
            self.learning_buffer.extend(recent_examples)
            
            # Perform batch learning
            self._perform_batch_learning()
            
            # Restore full buffer
            self.learning_buffer = temp_buffer
    
    def _adjust_learning_rate(self, factor: float):
        """Adjust learning rate for online models"""
        
        new_lr = max(self.config.min_learning_rate, self.learning_rate * factor)
        self.learning_rate = new_lr
        
        # Update SGD model learning rate
        if "sgd" in self.models:
            self.models["sgd"].set_params(eta0=new_lr)
        
        logger.debug(f"Learning rate adjusted to {new_lr:.6f}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        
        stats = {
            'total_examples': self.total_examples,
            'active_models': len(self.active_models),
            'best_model': self.best_model_id,
            'current_learning_rate': self.learning_rate,
            'last_batch_training': self.last_batch_training.isoformat(),
            'model_performances': {
                model_id: {
                    'accuracy': perf.accuracy,
                    'confidence': perf.prediction_confidence,
                    'stability': perf.stability,
                    'sample_count': perf.sample_count
                }
                for model_id, perf in self.model_performances.items()
            },
            'feature_info': {
                'total_features': len(self.feature_names) or len(self.selected_features),
                'selected_features': len(self.selected_features),
                'top_features': sorted(
                    self.feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10] if self.feature_importances else []
            },
            'drift_detection': {
                'window_size': self.drift_detector.window_size,
                'sensitivity': self.drift_detector.sensitivity,
                'recent_drift_count': len([d for d in self.drift_detector.drift_history 
                                         if (datetime.now() - d.timestamp).total_seconds() < 3600])
            }
        }
        
        return stats
    
    def save_models(self, directory: str):
        """Save all models to directory"""
        
        os.makedirs(directory, exist_ok=True)
        
        for model_id, model in self.models.items():
            try:
                model_path = os.path.join(directory, f"{model_id}.joblib")
                joblib.dump(model, model_path)
                logger.debug(f"Saved model {model_id} to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model {model_id}: {e}")
        
        # Save metadata
        metadata = {
            'config': self.config.__dict__,
            'model_performances': {
                k: {
                    'model_id': v.model_id,
                    'accuracy': v.accuracy,
                    'stability': v.stability,
                    'sample_count': v.sample_count,
                    'last_updated': v.last_updated.isoformat()
                }
                for k, v in self.model_performances.items()
            },
            'active_models': self.active_models,
            'best_model_id': self.best_model_id,
            'selected_features': self.selected_features,
            'learning_rate': self.learning_rate
        }
        
        metadata_path = os.path.join(directory, "metadata.json")
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models and metadata saved to {directory}")
    
    def load_models(self, directory: str):
        """Load models from directory"""
        
        # Load metadata
        metadata_path = os.path.join(directory, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            
            self.active_models = metadata.get('active_models', [])
            self.best_model_id = metadata.get('best_model_id')
            self.selected_features = metadata.get('selected_features', [])
            self.learning_rate = metadata.get('learning_rate', self.config.base_learning_rate)
            
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
        
        # Load models
        for model_id in self.active_models:
            try:
                model_path = os.path.join(directory, f"{model_id}.joblib")
                if os.path.exists(model_path):
                    self.models[model_id] = joblib.load(model_path)
                    logger.debug(f"Loaded model {model_id} from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
        
        logger.info(f"Models loaded from {directory}")


class ConceptDriftDetector:
    """Concept drift detection using statistical methods"""
    
    def __init__(self, window_size: int = 200, threshold: float = 0.05, sensitivity: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.drift_history: List[ConceptDriftDetection] = []
        
        # Drift detection state
        self.reference_window: Optional[np.ndarray] = None
        self.detection_window: Optional[np.ndarray] = None
    
    def detect_drift(self, features: List[np.ndarray], targets: List[Any]) -> ConceptDriftDetection:
        """Detect concept drift in the data stream"""
        
        if len(features) < self.window_size:
            return ConceptDriftDetection(
                drift_detected=False,
                drift_type=ConceptDriftType.NONE,
                confidence=0.0,
                affected_features=[],
                timestamp=datetime.now(),
                severity=0.0,
                recommended_actions=[]
            )
        
        # Convert to numpy arrays
        feature_matrix = np.array(features)
        target_array = np.array(targets)
        
        # Split into reference and detection windows
        mid_point = len(features) // 2
        reference_features = feature_matrix[:mid_point]
        detection_features = feature_matrix[mid_point:]
        reference_targets = target_array[:mid_point]
        detection_targets = target_array[mid_point:]
        
        # Feature-based drift detection
        feature_drift_scores = []
        affected_features = []
        
        for i in range(feature_matrix.shape[1]):
            ref_feature = reference_features[:, i]
            det_feature = detection_features[:, i]
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, p_value = stats.ks_2samp(ref_feature, det_feature)
                if p_value < self.threshold:
                    feature_drift_scores.append(ks_stat)
                    affected_features.append(f"feature_{i}")
            except:
                pass
        
        # Target distribution drift
        try:
            target_ks_stat, target_p_value = stats.ks_2samp(reference_targets, detection_targets)
            target_drift = target_p_value < self.threshold
        except:
            target_drift = False
            target_ks_stat = 0.0
        
        # Overall drift assessment
        feature_drift = len(affected_features) > 0
        drift_detected = feature_drift or target_drift
        
        if drift_detected:
            # Classify drift type
            drift_type = self._classify_drift_type(feature_drift_scores, target_ks_stat)
            
            # Calculate confidence and severity
            if feature_drift_scores:
                confidence = np.mean(feature_drift_scores)
                severity = max(feature_drift_scores)
            else:
                confidence = target_ks_stat
                severity = target_ks_stat
            
            # Generate recommendations
            recommendations = self._generate_drift_recommendations(drift_type, severity)
        else:
            drift_type = ConceptDriftType.NONE
            confidence = 0.0
            severity = 0.0
            recommendations = []
        
        # Create drift detection result
        drift_result = ConceptDriftDetection(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=confidence,
            affected_features=affected_features,
            timestamp=datetime.now(),
            severity=severity,
            recommended_actions=recommendations
        )
        
        # Store in history
        self.drift_history.append(drift_result)
        
        return drift_result
    
    def _classify_drift_type(self, feature_scores: List[float], target_score: float) -> ConceptDriftType:
        """Classify the type of concept drift"""
        
        if not feature_scores and target_score < 0.3:
            return ConceptDriftType.GRADUAL
        elif feature_scores and max(feature_scores) > 0.7:
            return ConceptDriftType.SUDDEN
        elif len(feature_scores) > 3:
            return ConceptDriftType.INCREMENTAL
        else:
            return ConceptDriftType.GRADUAL
    
    def _generate_drift_recommendations(self, drift_type: ConceptDriftType, severity: float) -> List[str]:
        """Generate recommendations for handling drift"""
        
        recommendations = []
        
        if drift_type == ConceptDriftType.SUDDEN:
            recommendations.extend([
                "immediate_retraining",
                "reset_online_models",
                "increase_learning_rate"
            ])
        elif drift_type == ConceptDriftType.GRADUAL:
            recommendations.extend([
                "adaptive_learning_rate",
                "increase_update_frequency",
                "monitor_performance"
            ])
        elif drift_type == ConceptDriftType.INCREMENTAL:
            recommendations.extend([
                "incremental_learning",
                "feature_selection_update",
                "model_ensemble_reweight"
            ])
        
        if severity > 0.8:
            recommendations.append("emergency_recalibration")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Create continuous learning engine
    config = LearningConfiguration(
        learning_mode=LearningMode.HYBRID,
        online_batch_size=25,
        batch_retrain_frequency=500,
        drift_detection_window=100
    )
    
    learning_engine = ContinuousLearningEngine(config)
    
    # Simulate learning examples
    for i in range(1000):
        # Generate synthetic features
        features = np.random.randn(10)
        
        # Generate target (regime) with some pattern
        if i < 500:
            target = 0 if features[0] > 0 else 1
        else:
            # Concept drift: reverse the pattern
            target = 1 if features[0] > 0 else 0
        
        # Create learning example
        example = LearningExample(
            features=features,
            target=target,
            timestamp=datetime.now(),
            context={'iteration': i},
            confidence=0.8
        )
        
        # Add to learning engine
        learning_engine.add_learning_example(example)
        
        # Make predictions periodically
        if i % 50 == 0:
            prediction, confidence = learning_engine.predict(features)
            accuracy = 1.0 if prediction == target else 0.0
            print(f"Iteration {i}: Prediction={prediction}, Actual={target}, "
                  f"Accuracy={accuracy}, Confidence={confidence:.3f}")
    
    # Get learning statistics
    stats = learning_engine.get_learning_statistics()
    print("\nLearning Statistics:")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Active models: {stats['active_models']}")
    print(f"Best model: {stats['best_model']}")
    print(f"Recent drift events: {stats['drift_detection']['recent_drift_count']}")
    
    # Save models
    with tempfile.TemporaryDirectory() as temp_dir:
        learning_engine.save_models(temp_dir)
        print(f"Models saved to temporary directory: {temp_dir}")