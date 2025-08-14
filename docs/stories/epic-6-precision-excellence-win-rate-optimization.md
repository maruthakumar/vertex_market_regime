# Epic 6: Precision Excellence & Win Rate Optimization
**Duration:** Weeks 7-10 (can start partially in Week 5)  
**Status:** Proposed  
**Priority:** CRITICAL - Achieving 95% Win Rate Target  
**Dependencies:** Epic 3 Stories 0-3 (Labels, Training, Calibration)  
**Investment:** ~$50K compute + 2-3 engineers for 4 weeks  

## Epic Goal
Transform the system from baseline 87% accuracy to 95% win rate through selective inference optimization, ensemble meta-learning, advanced feature engineering, and precision-focused model enhancements while maintaining sub-600ms latency.

## Epic Description
This epic implements targeted enhancements identified through systematic analysis to achieve the 95% win rate target. The approach focuses on high-confidence predictions through selective inference, sophisticated ensemble methods, cross-component feature engineering, and temporal pattern recognition. Each story has quantified impact on the win rate target.

### Key Innovations
- **Selective Inference**: Only predict when confidence exceeds regime-specific thresholds
- **Ensemble Meta-Learning**: Combine 5+ diverse models with neural network meta-learner
- **Cross-Component Intelligence**: Leverage interactions between all 8 components
- **Temporal Mining**: LSTM-based pattern extraction for regime transitions
- **Robustness by Design**: Component dropout training for system resilience

### Success Metrics
- **Primary**: Precision@threshold ≥ 95% per regime (measured on rolling 1000-prediction window)
- **Coverage**: ≥ 40% of trading minutes (acceptable trade-off for precision)
- **Latency**: P95 ≤ 600ms with all enhancements
- **Stability**: Win rate variance < 2% over rolling window
- **Robustness**: >90% accuracy with any single component failure

## Epic Stories

---

### Story 1: Cross-Component Feature Interaction Engineering
**Duration:** 3 days  
**Priority:** HIGH  
**Impact:** +2-3% accuracy  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to engineer cross-component interaction features  
**So that** the model captures complex relationships between components that indicate regime changes

**Acceptance Criteria:**
- [ ] Generate top 50 polynomial interaction features (Component pairs)
- [ ] Create 30 temporal interaction features (time-lagged products)
- [ ] Implement 20 ratio-based features across components
- [ ] Feature importance analysis shows >10% of top features are interactions
- [ ] Performance impact < 50ms additional latency
- [ ] Feature versioning and documentation in feature registry

**Technical Implementation:**
```python
class CrossComponentFeatureEngineer:
    """
    Engineer interaction features between components for enhanced regime detection
    """
    
    def __init__(self):
        self.interaction_configs = {
            # Polynomial Interactions (2nd and 3rd order)
            'polynomial_features': {
                'c1_c2_interaction': 'c1.straddle_signal * c2.gamma_exposure',
                'c1_c3_interaction': 'c1.atm_premium * c3.oi_flow_score',
                'c2_c4_interaction': 'c2.vega_score * c4.iv_skew_bias',
                'c3_c5_interaction': 'c3.institutional_flow * c5.atr_regime',
                'c4_c6_interaction': 'c4.term_structure * c6.correlation_strength',
                'c1_c2_c3_triple': 'c1.signal * c2.gamma * c3.oi_score',
                # ... 44 more polynomial interactions
            },
            
            # Temporal Lag Interactions
            'temporal_interactions': {
                'c1_momentum': 'c1.signal_t - c1.signal_t-5',
                'c1_c6_lag': 'c1.signal_t * c6.correlation_t-1',
                'c3_oi_acceleration': '(c3.oi_t - c3.oi_t-5) / 5',
                'c4_iv_momentum': 'c4.iv_percentile_t - c4.iv_percentile_t-10',
                'regime_persistence': 'regime_t-1 * confidence_t-1',
                # ... 25 more temporal features
            },
            
            # Ratio-Based Features
            'ratio_features': {
                'straddle_to_atr': 'c1.atm_straddle / c5.atr_value',
                'iv_to_realized': 'c4.iv_30d / c5.realized_vol',
                'gamma_to_volume': 'c2.net_gamma / c3.total_volume',
                'correlation_dispersion': 'c6.max_corr / c6.min_corr',
                'skew_normalized': 'c4.put_call_skew / c4.atm_iv',
                # ... 15 more ratio features
            }
        }
    
    def engineer_features(self, component_data: dict) -> pd.DataFrame:
        """
        Generate all interaction features with validation
        """
        features = pd.DataFrame()
        
        # Polynomial interactions with null handling
        for name, formula in self.interaction_configs['polynomial_features'].items():
            features[name] = self._safe_evaluate(formula, component_data)
        
        # Temporal interactions with lag validation
        for name, formula in self.interaction_configs['temporal_interactions'].items():
            features[name] = self._compute_temporal(formula, component_data)
        
        # Ratio features with zero-division protection
        for name, formula in self.interaction_configs['ratio_features'].items():
            features[name] = self._safe_ratio(formula, component_data)
        
        # Validate no information leakage
        self._validate_no_lookahead(features)
        
        return features
```

**Validation Tests:**
```python
def test_interaction_features():
    # Test polynomial interactions preserve causality
    # Test temporal lags don't leak future information
    # Test ratios handle edge cases (zero denominators)
    # Test feature importance > baseline
    pass
```

---

### Story 2: Ensemble Meta-Learner Implementation (Component 8 Enhancement)
**Duration:** 5 days  
**Priority:** CRITICAL  
**Impact:** +3-4% accuracy  
**Assignee:** Senior ML Engineer  

**As a** ML engineer  
**I want** to implement a stacking ensemble with meta-learner  
**So that** multiple models' strengths are combined optimally for maximum accuracy

**Acceptance Criteria:**
- [ ] Train 5 diverse base models (XGBoost, LightGBM, CatBoost, TabNet, Random Forest)
- [ ] Implement 2-layer neural network meta-learner with dropout
- [ ] Include model confidence scores and disagreement metrics as meta-features
- [ ] Cross-validation shows >3% improvement over best single model
- [ ] Model distillation reduces serving latency to <100ms
- [ ] Implement model versioning and A/B testing capability

**Technical Architecture:**
```python
class EnsembleMetaLearner:
    """
    Sophisticated ensemble with neural network meta-learner for 95% win rate
    """
    
    def __init__(self):
        # Base Models Configuration
        self.base_models = {
            'xgboost': {
                'model': XGBClassifier(
                    n_estimators=1000,
                    max_depth=8,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=1,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    scale_pos_weight=self._calculate_class_weights()
                ),
                'weight': 0.25,
                'features': 'all'
            },
            
            'lightgbm': {
                'model': LGBMClassifier(
                    num_leaves=127,
                    learning_rate=0.03,
                    n_estimators=800,
                    subsample=0.9,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_samples=20,
                    class_weight='balanced'
                ),
                'weight': 0.25,
                'features': 'all'
            },
            
            'catboost': {
                'model': CatBoostClassifier(
                    iterations=1000,
                    depth=10,
                    learning_rate=0.03,
                    l2_leaf_reg=3,
                    bootstrap_type='Bayesian',
                    bagging_temperature=0.8,
                    class_weights=self._calculate_class_weights(),
                    verbose=False
                ),
                'weight': 0.20,
                'features': 'all'
            },
            
            'tabnet': {
                'model': TabNetClassifier(
                    n_d=64,
                    n_a=64,
                    n_steps=5,
                    gamma=1.5,
                    n_independent=2,
                    n_shared=2,
                    lambda_sparse=1e-4,
                    momentum=0.3,
                    clip_value=2.0,
                    optimizer_params={'lr': 0.02},
                    scheduler_params={'gamma': 0.95}
                ),
                'weight': 0.20,
                'features': 'normalized'
            },
            
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    class_weight='balanced_subsample',
                    n_jobs=-1
                ),
                'weight': 0.10,
                'features': 'all'
            }
        }
        
        # Meta-Learner Architecture
        self.meta_learner = self._build_meta_learner()
        
    def _build_meta_learner(self):
        """
        Neural network meta-learner for combining base model predictions
        """
        import tensorflow as tf
        
        # Input: 5 models × 8 regimes = 40 base predictions
        # Plus: 5 confidence scores, 8 component signals, 10 meta-features
        input_dim = 40 + 5 + 8 + 10  # 63 total
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')  # 8 regimes
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """
        Two-stage training: base models then meta-learner
        """
        # Stage 1: Train base models with cross-validation
        base_predictions_train = []
        base_predictions_val = []
        
        for name, config in self.base_models.items():
            print(f"Training {name}...")
            
            # Get appropriate features
            X_train_model = self._prepare_features(X_train, config['features'])
            X_val_model = self._prepare_features(X_val, config['features'])
            
            # Train with early stopping
            if hasattr(config['model'], 'fit'):
                config['model'].fit(
                    X_train_model, y_train,
                    eval_set=[(X_val_model, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            
            # Get predictions and confidence
            train_pred = config['model'].predict_proba(X_train_model)
            val_pred = config['model'].predict_proba(X_val_model)
            
            base_predictions_train.append(train_pred)
            base_predictions_val.append(val_pred)
        
        # Stage 2: Train meta-learner
        meta_features_train = self._create_meta_features(
            base_predictions_train, X_train
        )
        meta_features_val = self._create_meta_features(
            base_predictions_val, X_val
        )
        
        # Train meta-learner with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.meta_learner.fit(
            meta_features_train, y_train,
            validation_data=(meta_features_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
    def predict_with_confidence(self, X):
        """
        Generate predictions with confidence scores
        """
        # Get base model predictions
        base_predictions = []
        for name, config in self.base_models.items():
            X_model = self._prepare_features(X, config['features'])
            pred = config['model'].predict_proba(X_model)
            base_predictions.append(pred)
        
        # Create meta features
        meta_features = self._create_meta_features(base_predictions, X)
        
        # Get final predictions
        final_predictions = self.meta_learner.predict(meta_features)
        confidence_scores = np.max(final_predictions, axis=1)
        
        return final_predictions, confidence_scores
```

**Model Distillation for Production:**
```python
class DistilledEnsemble:
    """
    Compress ensemble into single fast model for production
    """
    
    def distill_ensemble(self, ensemble, X_train, temperature=3.0):
        """
        Knowledge distillation to create compact serving model
        """
        # Get soft labels from ensemble
        soft_labels = ensemble.predict_with_confidence(X_train)[0]
        soft_labels = self._apply_temperature(soft_labels, temperature)
        
        # Train compact student model
        student = XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05
        )
        
        student.fit(X_train, soft_labels)
        
        return student
```

---

### Story 3: Selective Inference Optimization
**Duration:** 4 days  
**Priority:** CRITICAL  
**Impact:** +5-7% precision  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to optimize confidence thresholds per regime  
**So that** we only make predictions when confidence is high enough to ensure 95% precision

**Acceptance Criteria:**
- [ ] Implement per-regime dynamic confidence thresholds
- [ ] Create abstention mechanism with informative "no-signal" responses
- [ ] Optimize precision-coverage trade-off curves for each regime
- [ ] Document coverage metrics for each threshold setting
- [ ] API returns structured response with confidence and reasoning
- [ ] Implement threshold adjustment based on recent performance

**Implementation:**
```python
class SelectiveInferenceOptimizer:
    """
    Optimize thresholds for 95% precision with maximum coverage
    """
    
    def __init__(self):
        # Initial thresholds (to be optimized)
        self.regime_thresholds = {
            'LVLD': 0.92,  # Low Volatility Low Delta
            'HVC': 0.90,   # High Volatility Consolidation
            'VCPE': 0.91,  # Volatility Compression Peak Exit
            'TBVE': 0.93,  # Trending Bullish Volatility Expansion
            'TBVS': 0.93,  # Trending Bearish Volatility Squeeze
            'SCGS': 0.94,  # Strong Counter-Gamma Squeeze
            'PSED': 0.94,  # Put Spread Expansion Dominant
            'CBV': 0.95    # Complex Breakout Volatility
        }
        
        # Performance tracking
        self.performance_buffer = deque(maxlen=1000)
        self.threshold_optimizer = ThresholdOptimizer()
        
    def optimize_thresholds(self, X_val, y_val, target_precision=0.95):
        """
        Find optimal thresholds using validation data
        """
        from sklearn.metrics import precision_recall_curve
        
        optimized_thresholds = {}
        coverage_stats = {}
        
        for regime_idx, regime_name in enumerate(self.regime_thresholds.keys()):
            # Get predictions for this regime
            y_true_regime = (y_val == regime_idx).astype(int)
            y_scores = self.model.predict_proba(X_val)[:, regime_idx]
            
            # Calculate precision-recall curve
            precisions, recalls, thresholds = precision_recall_curve(
                y_true_regime, y_scores
            )
            
            # Find threshold for target precision
            valid_idx = np.where(precisions >= target_precision)[0]
            
            if len(valid_idx) > 0:
                # Choose threshold with maximum recall at target precision
                best_idx = valid_idx[np.argmax(recalls[valid_idx])]
                optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.99
                achieved_precision = precisions[best_idx]
                achieved_recall = recalls[best_idx]
            else:
                # If target precision not achievable, use maximum precision
                best_idx = np.argmax(precisions)
                optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.99
                achieved_precision = precisions[best_idx]
                achieved_recall = recalls[best_idx]
            
            optimized_thresholds[regime_name] = float(optimal_threshold)
            coverage_stats[regime_name] = {
                'threshold': float(optimal_threshold),
                'precision': float(achieved_precision),
                'recall': float(achieved_recall),
                'coverage': float(achieved_recall),  # Proportion of regime instances predicted
                'f1_score': float(2 * achieved_precision * achieved_recall / 
                                 (achieved_precision + achieved_recall + 1e-10))
            }
            
            print(f"{regime_name}: threshold={optimal_threshold:.3f}, "
                  f"precision={achieved_precision:.3f}, coverage={achieved_recall:.3f}")
        
        self.regime_thresholds = optimized_thresholds
        self.coverage_stats = coverage_stats
        
        return optimized_thresholds, coverage_stats
    
    def predict_with_abstention(self, X, return_all_scores=False):
        """
        Make predictions only when confidence exceeds threshold
        """
        # Get raw predictions and probabilities
        probabilities = self.model.predict_proba(X)
        max_probs = np.max(probabilities, axis=1)
        predicted_regimes = np.argmax(probabilities, axis=1)
        
        results = []
        
        for i in range(len(X)):
            regime_idx = predicted_regimes[i]
            regime_name = list(self.regime_thresholds.keys())[regime_idx]
            confidence = max_probs[i]
            threshold = self.regime_thresholds[regime_name]
            
            if confidence >= threshold:
                # Make prediction
                result = {
                    'prediction': regime_name,
                    'confidence': float(confidence),
                    'threshold': float(threshold),
                    'margin': float(confidence - threshold),
                    'status': 'predicted'
                }
            else:
                # Abstain from prediction
                result = {
                    'prediction': None,
                    'confidence': float(confidence),
                    'threshold': float(threshold),
                    'margin': float(confidence - threshold),
                    'status': 'no_signal',
                    'reason': f'Confidence {confidence:.3f} below threshold {threshold:.3f}',
                    'likely_regime': regime_name  # For monitoring
                }
            
            if return_all_scores:
                result['all_scores'] = {
                    name: float(probabilities[i, j]) 
                    for j, name in enumerate(self.regime_thresholds.keys())
                }
            
            results.append(result)
            
            # Track performance for dynamic adjustment
            self.performance_buffer.append({
                'regime': regime_name,
                'confidence': confidence,
                'threshold': threshold,
                'predicted': confidence >= threshold
            })
        
        return results
    
    def adjust_thresholds_dynamically(self):
        """
        Adjust thresholds based on recent performance
        """
        if len(self.performance_buffer) < 100:
            return  # Not enough data
        
        recent_performance = pd.DataFrame(self.performance_buffer)
        
        for regime in self.regime_thresholds.keys():
            regime_data = recent_performance[recent_performance['regime'] == regime]
            
            if len(regime_data) > 10:
                # Calculate actual precision
                predicted_data = regime_data[regime_data['predicted']]
                if len(predicted_data) > 0:
                    # This would need ground truth labels in production
                    # For now, using confidence as proxy
                    avg_confidence = predicted_data['confidence'].mean()
                    
                    # Adjust threshold if precision proxy is off target
                    if avg_confidence < 0.95:
                        # Increase threshold
                        self.regime_thresholds[regime] = min(
                            0.99, 
                            self.regime_thresholds[regime] * 1.02
                        )
                    elif avg_confidence > 0.97:
                        # Can decrease threshold slightly for more coverage
                        self.regime_thresholds[regime] = max(
                            0.85,
                            self.regime_thresholds[regime] * 0.98
                        )
```

**API Response Structure:**
```python
def format_api_response(predictions):
    """
    Structure API response with selective inference
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'predictions': [
            {
                'minute': pred.get('minute'),
                'regime': pred.get('prediction'),  # None if no_signal
                'confidence': pred.get('confidence'),
                'status': pred.get('status'),  # 'predicted' or 'no_signal'
                'reason': pred.get('reason'),  # Explanation if no_signal
                'metadata': {
                    'threshold_used': pred.get('threshold'),
                    'confidence_margin': pred.get('margin'),
                    'all_regime_scores': pred.get('all_scores', {})
                }
            }
            for pred in predictions
        ],
        'summary': {
            'total_minutes': len(predictions),
            'predicted_minutes': sum(1 for p in predictions if p['status'] == 'predicted'),
            'coverage_rate': sum(1 for p in predictions if p['status'] == 'predicted') / len(predictions),
            'abstention_rate': sum(1 for p in predictions if p['status'] == 'no_signal') / len(predictions)
        }
    }
```

---

### Story 4: Advanced Probability Calibration Pipeline
**Duration:** 3 days  
**Priority:** HIGH  
**Impact:** +2-3% precision  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to implement sophisticated probability calibration  
**So that** model confidence scores accurately reflect true probabilities

**Acceptance Criteria:**
- [ ] Implement Platt scaling, isotonic regression, and temperature scaling
- [ ] Create per-regime calibration curves with validation
- [ ] Ensemble calibration for combined predictions
- [ ] Calibration plots show near-diagonal alignment
- [ ] ECE (Expected Calibration Error) < 0.02 for all regimes
- [ ] Beta calibration for extreme probabilities

**Implementation:**
```python
class AdvancedCalibrationPipeline:
    """
    Multi-method calibration for accurate probability estimates
    """
    
    def __init__(self):
        self.calibrators = {}
        self.calibration_methods = ['platt', 'isotonic', 'temperature', 'beta']
        self.regime_calibrators = {}
        
    def fit_calibration(self, X_cal, y_cal, probabilities):
        """
        Fit multiple calibration methods and select best
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        for regime_idx, regime_name in enumerate(self.regime_names):
            regime_probs = probabilities[:, regime_idx]
            regime_labels = (y_cal == regime_idx).astype(int)
            
            calibrators = {}
            
            # Platt Scaling
            platt = LogisticRegression()
            platt.fit(regime_probs.reshape(-1, 1), regime_labels)
            calibrators['platt'] = platt
            
            # Isotonic Regression
            isotonic = IsotonicRegression(out_of_bounds='clip')
            isotonic.fit(regime_probs, regime_labels)
            calibrators['isotonic'] = isotonic
            
            # Temperature Scaling
            temperature = self._fit_temperature_scaling(regime_probs, regime_labels)
            calibrators['temperature'] = temperature
            
            # Beta Calibration
            beta = self._fit_beta_calibration(regime_probs, regime_labels)
            calibrators['beta'] = beta
            
            # Select best calibrator based on ECE
            best_method, best_calibrator = self._select_best_calibrator(
                calibrators, regime_probs, regime_labels
            )
            
            self.regime_calibrators[regime_name] = {
                'method': best_method,
                'calibrator': best_calibrator,
                'ece_before': self._calculate_ece(regime_probs, regime_labels),
                'ece_after': self._calculate_ece(
                    self._apply_calibration(regime_probs, best_calibrator, best_method),
                    regime_labels
                )
            }
    
    def _fit_temperature_scaling(self, probabilities, labels):
        """
        Fit temperature parameter for scaling
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        class TemperatureScaling(nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = nn.Parameter(torch.ones(1))
            
            def forward(self, logits):
                return logits / self.temperature
        
        # Convert to torch tensors
        logits = torch.from_numpy(np.log(probabilities + 1e-10)).float()
        labels = torch.from_numpy(labels).long()
        
        # Optimize temperature
        model = TemperatureScaling()
        optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled = model(logits.unsqueeze(1))
            loss = criterion(scaled, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return float(model.temperature.item())
    
    def _fit_beta_calibration(self, probabilities, labels):
        """
        Fit beta calibration for extreme probabilities
        """
        from scipy.optimize import minimize
        from scipy.stats import beta
        
        def beta_log_loss(params, probs, labels):
            a, b = params
            # Map probabilities through beta CDF
            calibrated = beta.cdf(probs, a, b)
            # Calculate log loss
            epsilon = 1e-10
            calibrated = np.clip(calibrated, epsilon, 1 - epsilon)
            return -np.mean(
                labels * np.log(calibrated) + 
                (1 - labels) * np.log(1 - calibrated)
            )
        
        # Optimize beta parameters
        result = minimize(
            beta_log_loss,
            x0=[1, 1],
            args=(probabilities, labels),
            bounds=[(0.1, 10), (0.1, 10)],
            method='L-BFGS-B'
        )
        
        return result.x
    
    def _calculate_ece(self, probabilities, labels, n_bins=10):
        """
        Calculate Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def calibrate_probabilities(self, probabilities):
        """
        Apply calibration to raw probabilities
        """
        calibrated = np.zeros_like(probabilities)
        
        for regime_idx, regime_name in enumerate(self.regime_names):
            calibrator_info = self.regime_calibrators[regime_name]
            method = calibrator_info['method']
            calibrator = calibrator_info['calibrator']
            
            regime_probs = probabilities[:, regime_idx]
            
            if method == 'platt':
                calibrated[:, regime_idx] = calibrator.predict_proba(
                    regime_probs.reshape(-1, 1)
                )[:, 1]
            elif method == 'isotonic':
                calibrated[:, regime_idx] = calibrator.transform(regime_probs)
            elif method == 'temperature':
                calibrated[:, regime_idx] = self._apply_temperature(
                    regime_probs, calibrator
                )
            elif method == 'beta':
                from scipy.stats import beta
                a, b = calibrator
                calibrated[:, regime_idx] = beta.cdf(regime_probs, a, b)
        
        # Renormalize to ensure probabilities sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated
    
    def plot_calibration_curves(self, probabilities, labels, save_path='calibration_plots.png'):
        """
        Generate calibration plots for validation
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for regime_idx, (regime_name, ax) in enumerate(zip(self.regime_names, axes)):
            regime_probs = probabilities[:, regime_idx]
            regime_labels = (labels == regime_idx).astype(int)
            
            # Calculate calibration curve
            fraction_pos, mean_pred = calibration_curve(
                regime_labels, regime_probs, n_bins=10
            )
            
            # Plot
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax.plot(mean_pred, fraction_pos, 'o-', label='Before calibration')
            
            # Plot after calibration
            if regime_name in self.regime_calibrators:
                calibrated_probs = self.calibrate_probabilities(probabilities)[:, regime_idx]
                fraction_pos_cal, mean_pred_cal = calibration_curve(
                    regime_labels, calibrated_probs, n_bins=10
                )
                ax.plot(mean_pred_cal, fraction_pos_cal, 's-', label='After calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'{regime_name} Calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
```

---

### Story 5: Temporal Pattern Mining with LSTM Features
**Duration:** 4 days  
**Priority:** MEDIUM  
**Impact:** +2% accuracy on transitions  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to extract temporal patterns using LSTM  
**So that** regime transitions are better predicted

**Acceptance Criteria:**
- [ ] Train LSTM autoencoder on component time sequences
- [ ] Extract 30 temporal embedding features
- [ ] Attention weights identify important historical timepoints
- [ ] Transition prediction accuracy improves >2%
- [ ] Features integrated into ensemble pipeline
- [ ] Latency impact < 30ms

**Implementation:**
```python
class TemporalPatternMiner:
    """
    LSTM-based temporal pattern extraction for regime transitions
    """
    
    def __init__(self, sequence_length=20, embedding_dim=30):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_encoder = None
        self.attention_model = None
        
    def build_lstm_autoencoder(self, input_dim):
        """
        Build LSTM autoencoder for temporal pattern extraction
        """
        import tensorflow as tf
        
        # Encoder
        encoder_input = tf.keras.Input(shape=(self.sequence_length, input_dim))
        
        # Bidirectional LSTM layers
        lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(encoder_input)
        
        lstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(lstm1)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=64
        )(lstm2, lstm2)
        
        # Encoding layer
        encoding = tf.keras.layers.LSTM(self.embedding_dim, name='encoding')(attention)
        
        # Decoder
        repeat = tf.keras.layers.RepeatVector(self.sequence_length)(encoding)
        
        lstm3 = tf.keras.layers.LSTM(64, return_sequences=True)(repeat)
        lstm4 = tf.keras.layers.LSTM(128, return_sequences=True)(lstm3)
        
        decoder_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(input_dim)
        )(lstm4)
        
        # Models
        autoencoder = tf.keras.Model(encoder_input, decoder_output)
        encoder = tf.keras.Model(encoder_input, encoding)
        
        # Compile
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.autoencoder = autoencoder
        self.encoder = encoder
        
        # Build attention extraction model
        attention_model = tf.keras.Model(
            encoder_input,
            attention
        )
        self.attention_model = attention_model
        
        return autoencoder, encoder
    
    def train_temporal_model(self, X_sequences, epochs=50):
        """
        Train LSTM autoencoder on historical sequences
        """
        # Train autoencoder
        history = self.autoencoder.fit(
            X_sequences, X_sequences,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        return history
    
    def extract_temporal_features(self, X_sequences):
        """
        Extract temporal embeddings and attention weights
        """
        # Get LSTM embeddings
        embeddings = self.encoder.predict(X_sequences, batch_size=32)
        
        # Get attention weights
        attention_outputs = self.attention_model.predict(X_sequences, batch_size=32)
        
        # Calculate temporal statistics
        temporal_features = np.concatenate([
            embeddings,  # Raw embeddings (30 features)
            np.mean(attention_outputs, axis=1),  # Attention statistics
            np.std(attention_outputs, axis=1),
            np.max(attention_outputs, axis=1),
            np.min(attention_outputs, axis=1)
        ], axis=1)
        
        return temporal_features
    
    def identify_transition_patterns(self, X_sequences, y_transitions):
        """
        Identify patterns that precede regime transitions
        """
        # Extract features for transition and non-transition sequences
        transition_idx = np.where(y_transitions == 1)[0]
        stable_idx = np.where(y_transitions == 0)[0]
        
        transition_features = self.extract_temporal_features(X_sequences[transition_idx])
        stable_features = self.extract_temporal_features(X_sequences[stable_idx])
        
        # Statistical test for significant differences
        from scipy import stats
        
        significant_features = []
        for i in range(transition_features.shape[1]):
            _, p_value = stats.ttest_ind(
                transition_features[:, i],
                stable_features[:, i]
            )
            if p_value < 0.01:  # Significant difference
                significant_features.append(i)
        
        print(f"Found {len(significant_features)} significant temporal features for transitions")
        
        return significant_features
    
    def create_sequence_features(self, component_data, lookback=20):
        """
        Create sequences from component data for LSTM processing
        """
        sequences = []
        
        for i in range(lookback, len(component_data)):
            sequence = component_data[i-lookback:i]
            sequences.append(sequence)
        
        return np.array(sequences)
```

**Transition Detection Enhancement:**
```python
class RegimeTransitionDetector:
    """
    Specialized model for detecting regime transitions
    """
    
    def __init__(self, temporal_miner):
        self.temporal_miner = temporal_miner
        self.transition_model = None
        
    def build_transition_model(self, input_dim):
        """
        Build specialized model for transition detection
        """
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')  # Binary: transition or stable
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.transition_model = model
        return model
    
    def predict_transitions(self, X, threshold=0.7):
        """
        Predict regime transitions with confidence
        """
        # Extract temporal features
        sequences = self.temporal_miner.create_sequence_features(X)
        temporal_features = self.temporal_miner.extract_temporal_features(sequences)
        
        # Predict transitions
        transition_probs = self.transition_model.predict(temporal_features)[:, 1]
        
        # Apply threshold
        transitions = transition_probs > threshold
        
        return transitions, transition_probs
```

---

### Story 6: Component Dropout Robustness Training
**Duration:** 2 days  
**Priority:** MEDIUM  
**Impact:** +1% robustness  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to train models with component dropout  
**So that** the system is robust to component failures

**Acceptance Criteria:**
- [ ] Implement training with 10-20% random component dropout
- [ ] Models maintain >90% accuracy with any single component missing
- [ ] Document performance degradation curves
- [ ] API handles missing components gracefully
- [ ] Implement component health monitoring
- [ ] Fallback strategies for each component

**Implementation:**
```python
class RobustTrainingPipeline:
    """
    Training with component dropout for robustness
    """
    
    def __init__(self, dropout_rate=0.15):
        self.dropout_rate = dropout_rate
        self.component_importance = {}
        self.degradation_curves = {}
        
    def train_with_dropout(self, X, y, component_groups):
        """
        Train with random component dropout
        """
        from sklearn.model_selection import KFold
        
        models = []
        dropout_patterns = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Generate dropout patterns for this fold
            n_patterns = 10
            for pattern_idx in range(n_patterns):
                # Randomly select components to drop
                dropout_mask = self._generate_dropout_mask(component_groups)
                
                # Apply dropout
                X_train_dropped = self._apply_component_dropout(
                    X_train, component_groups, dropout_mask
                )
                X_val_dropped = self._apply_component_dropout(
                    X_val, component_groups, dropout_mask
                )
                
                # Train model with dropout
                model = XGBClassifier(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05
                )
                
                model.fit(X_train_dropped, y_train)
                
                # Evaluate
                val_score = model.score(X_val_dropped, y_val)
                
                models.append(model)
                dropout_patterns.append({
                    'pattern': dropout_mask,
                    'validation_score': val_score,
                    'fold': fold_idx
                })
                
                print(f"Fold {fold_idx}, Pattern {pattern_idx}: "
                      f"Dropped {sum(dropout_mask)} components, "
                      f"Val accuracy: {val_score:.3f}")
        
        # Analyze component importance through dropout impact
        self._analyze_component_importance(dropout_patterns)
        
        return models, dropout_patterns
    
    def _generate_dropout_mask(self, component_groups):
        """
        Generate random dropout mask for components
        """
        n_components = len(component_groups)
        dropout_mask = np.random.binomial(1, self.dropout_rate, n_components)
        
        # Ensure at least one component is kept
        if dropout_mask.sum() == n_components:
            dropout_mask[np.random.randint(n_components)] = 0
        
        return dropout_mask
    
    def _apply_component_dropout(self, X, component_groups, dropout_mask):
        """
        Apply dropout by setting component features to baseline values
        """
        X_dropped = X.copy()
        
        for comp_idx, (comp_name, feature_indices) in enumerate(component_groups.items()):
            if dropout_mask[comp_idx]:
                # Set to median values (or zeros)
                X_dropped[:, feature_indices] = np.median(
                    X[:, feature_indices], axis=0
                )
        
        return X_dropped
    
    def test_single_component_failure(self, X_test, y_test, component_groups, base_model):
        """
        Test performance with each component individually failed
        """
        degradation_results = {}
        base_score = base_model.score(X_test, y_test)
        
        for comp_name, feature_indices in component_groups.items():
            # Create copy with component failed
            X_failed = X_test.copy()
            X_failed[:, feature_indices] = np.median(
                X_test[:, feature_indices], axis=0
            )
            
            # Evaluate
            failed_score = base_model.score(X_failed, y_test)
            degradation = base_score - failed_score
            
            degradation_results[comp_name] = {
                'base_score': base_score,
                'failed_score': failed_score,
                'degradation': degradation,
                'relative_degradation': degradation / base_score * 100
            }
            
            print(f"Component {comp_name} failure: "
                  f"{failed_score:.3f} ({-degradation*100:.1f}% degradation)")
        
        self.degradation_curves = degradation_results
        return degradation_results
    
    def implement_fallback_strategies(self):
        """
        Define fallback strategies for component failures
        """
        fallback_strategies = {
            'component_1_straddle': {
                'detection': 'check_straddle_data_freshness',
                'fallback': 'use_previous_straddle_values',
                'confidence_penalty': 0.1
            },
            'component_2_greeks': {
                'detection': 'validate_greek_calculations',
                'fallback': 'use_simplified_greeks',
                'confidence_penalty': 0.15
            },
            'component_3_oi_pa': {
                'detection': 'check_oi_data_completeness',
                'fallback': 'use_volume_weighted_estimates',
                'confidence_penalty': 0.12
            },
            'component_4_iv_skew': {
                'detection': 'validate_iv_surface',
                'fallback': 'use_atm_iv_only',
                'confidence_penalty': 0.18
            },
            'component_5_atr_ema': {
                'detection': 'check_price_data_continuity',
                'fallback': 'use_simple_moving_averages',
                'confidence_penalty': 0.08
            },
            'component_6_correlation': {
                'detection': 'validate_correlation_matrix',
                'fallback': 'use_historical_correlations',
                'confidence_penalty': 0.20
            },
            'component_7_sr': {
                'detection': 'check_level_calculations',
                'fallback': 'use_pivot_points_only',
                'confidence_penalty': 0.10
            },
            'component_8_integration': {
                'detection': 'validate_component_signals',
                'fallback': 'use_majority_voting',
                'confidence_penalty': 0.25
            }
        }
        
        return fallback_strategies
```

---

### Story 7: Active Learning Pipeline for Edge Cases
**Duration:** 3 days  
**Priority:** LOW  
**Impact:** +1-2% on edge cases  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to implement active learning for difficult samples  
**So that** edge cases are continuously improved

**Acceptance Criteria:**
- [ ] Identify lowest-confidence predictions daily
- [ ] Create human-in-the-loop labeling interface
- [ ] Implement uncertainty sampling strategies
- [ ] Retrain on accumulated edge cases weekly
- [ ] Track improvement metrics on difficult samples
- [ ] Version control for labeled edge cases

**Implementation:**
```python
class ActiveLearningPipeline:
    """
    Active learning for continuous improvement on edge cases
    """
    
    def __init__(self, uncertainty_threshold=0.3):
        self.uncertainty_threshold = uncertainty_threshold
        self.edge_case_buffer = []
        self.labeling_queue = deque(maxlen=1000)
        self.improvement_tracker = {}
        
    def identify_edge_cases(self, X, predictions, probabilities):
        """
        Identify samples needing human review
        """
        edge_cases = []
        
        # Strategy 1: Uncertainty Sampling (low confidence)
        max_probs = np.max(probabilities, axis=1)
        uncertain_idx = np.where(max_probs < (1 - self.uncertainty_threshold))[0]
        
        # Strategy 2: Margin Sampling (close second choice)
        sorted_probs = np.sort(probabilities, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        margin_idx = np.where(margins < 0.1)[0]
        
        # Strategy 3: Entropy-based sampling
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        entropy_idx = np.where(entropy > np.percentile(entropy, 90))[0]
        
        # Combine strategies
        edge_case_indices = np.unique(np.concatenate([
            uncertain_idx, margin_idx, entropy_idx
        ]))
        
        for idx in edge_case_indices:
            edge_cases.append({
                'index': idx,
                'features': X[idx],
                'prediction': predictions[idx],
                'probabilities': probabilities[idx],
                'confidence': max_probs[idx],
                'margin': margins[idx] if idx < len(margins) else 0,
                'entropy': entropy[idx],
                'timestamp': datetime.now().isoformat(),
                'strategy': self._determine_strategy(idx, uncertain_idx, margin_idx, entropy_idx)
            })
        
        return edge_cases
    
    def create_labeling_interface(self):
        """
        Web interface for human labeling of edge cases
        """
        from flask import Flask, render_template, request, jsonify
        
        app = Flask(__name__)
        
        @app.route('/label_queue')
        def get_labeling_queue():
            # Return next unlabeled edge case
            if self.labeling_queue:
                case = self.labeling_queue[0]
                return jsonify({
                    'case_id': case['index'],
                    'features': case['features'].tolist(),
                    'model_prediction': case['prediction'],
                    'confidence': float(case['confidence']),
                    'all_probabilities': case['probabilities'].tolist()
                })
            return jsonify({'message': 'No cases to label'})
        
        @app.route('/submit_label', methods=['POST'])
        def submit_label():
            data = request.json
            case_id = data['case_id']
            human_label = data['label']
            confidence = data.get('confidence', 1.0)
            notes = data.get('notes', '')
            
            # Store labeled case
            self._store_labeled_case(case_id, human_label, confidence, notes)
            
            # Remove from queue
            self.labeling_queue.popleft()
            
            return jsonify({'success': True, 'remaining': len(self.labeling_queue)})
        
        @app.route('/dashboard')
        def dashboard():
            stats = {
                'total_labeled': len(self.edge_case_buffer),
                'queue_size': len(self.labeling_queue),
                'improvement_metrics': self.improvement_tracker
            }
            return jsonify(stats)
        
        return app
    
    def retrain_with_edge_cases(self, base_model, min_cases=100):
        """
        Retrain model incorporating labeled edge cases
        """
        if len(self.edge_case_buffer) < min_cases:
            print(f"Only {len(self.edge_case_buffer)} cases available, "
                  f"waiting for {min_cases}")
            return base_model
        
        # Prepare augmented training data
        edge_X = np.array([case['features'] for case in self.edge_case_buffer])
        edge_y = np.array([case['human_label'] for case in self.edge_case_buffer])
        edge_weights = np.array([case['confidence'] for case in self.edge_case_buffer])
        
        # Combine with original training data (with lower weight)
        X_combined = np.vstack([self.X_train, edge_X])
        y_combined = np.hstack([self.y_train, edge_y])
        
        # Weight edge cases higher
        weights = np.ones(len(X_combined))
        weights[-len(edge_X):] *= 2.0  # Double weight for edge cases
        weights[-len(edge_X):] *= edge_weights  # Apply confidence weights
        
        # Retrain model
        model = clone(base_model)
        model.fit(X_combined, y_combined, sample_weight=weights)
        
        # Evaluate improvement on edge cases
        edge_improvement = self._evaluate_edge_case_improvement(
            base_model, model, edge_X, edge_y
        )
        
        self.improvement_tracker[datetime.now().isoformat()] = {
            'n_edge_cases': len(edge_X),
            'base_accuracy': edge_improvement['base_accuracy'],
            'new_accuracy': edge_improvement['new_accuracy'],
            'improvement': edge_improvement['improvement']
        }
        
        print(f"Retrained with {len(edge_X)} edge cases. "
              f"Improvement: {edge_improvement['improvement']:.2%}")
        
        return model
    
    def _evaluate_edge_case_improvement(self, base_model, new_model, X_edge, y_edge):
        """
        Measure improvement on edge cases
        """
        base_score = base_model.score(X_edge, y_edge)
        new_score = new_model.score(X_edge, y_edge)
        
        return {
            'base_accuracy': base_score,
            'new_accuracy': new_score,
            'improvement': new_score - base_score
        }
```

---

### Story 8: Data Augmentation for Rare Regimes
**Duration:** 2 days  
**Priority:** MEDIUM  
**Impact:** +1-2% on rare regimes  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to augment data for underrepresented regimes  
**So that** rare regimes are better classified

**Acceptance Criteria:**
- [ ] Implement SMOTE for minority regime oversampling
- [ ] Create synthetic regime transitions using VAE
- [ ] Time-series specific augmentation (jittering, scaling, time-warping)
- [ ] Validation shows improved minority class F1 scores >5%
- [ ] No degradation on majority classes
- [ ] Augmentation reproducibility with seeds

**Implementation:**
```python
class RegimeDataAugmenter:
    """
    Sophisticated data augmentation for regime imbalance
    """
    
    def __init__(self, target_ratio=0.2):
        self.target_ratio = target_ratio
        self.augmentation_stats = {}
        self.vae_model = None
        
    def augment_rare_regimes(self, X, y, regime_names):
        """
        Main augmentation pipeline
        """
        # Analyze regime distribution
        regime_counts = pd.Series(y).value_counts()
        print("Original regime distribution:")
        for regime_idx, count in regime_counts.items():
            print(f"  {regime_names[regime_idx]}: {count} ({count/len(y):.2%})")
        
        # Identify rare regimes (< target_ratio)
        total_samples = len(y)
        rare_regimes = [
            idx for idx, count in regime_counts.items()
            if count / total_samples < self.target_ratio
        ]
        
        augmented_X = X.copy()
        augmented_y = y.copy()
        
        for regime_idx in rare_regimes:
            regime_name = regime_names[regime_idx]
            current_count = regime_counts[regime_idx]
            target_count = int(total_samples * self.target_ratio)
            samples_needed = target_count - current_count
            
            print(f"\nAugmenting {regime_name}: {current_count} → {target_count}")
            
            # Get regime samples
            regime_mask = y == regime_idx
            X_regime = X[regime_mask]
            
            # Apply multiple augmentation strategies
            synthetic_samples = []
            
            # Strategy 1: SMOTE (30% of needed samples)
            smote_samples = self._apply_smote(
                X_regime, int(samples_needed * 0.3)
            )
            synthetic_samples.append(smote_samples)
            
            # Strategy 2: VAE generation (30%)
            vae_samples = self._generate_vae_samples(
                X_regime, int(samples_needed * 0.3)
            )
            synthetic_samples.append(vae_samples)
            
            # Strategy 3: Time-series augmentation (40%)
            ts_samples = self._apply_timeseries_augmentation(
                X_regime, int(samples_needed * 0.4)
            )
            synthetic_samples.append(ts_samples)
            
            # Combine synthetic samples
            X_synthetic = np.vstack(synthetic_samples)
            y_synthetic = np.full(len(X_synthetic), regime_idx)
            
            # Add to augmented dataset
            augmented_X = np.vstack([augmented_X, X_synthetic])
            augmented_y = np.hstack([augmented_y, y_synthetic])
            
            self.augmentation_stats[regime_name] = {
                'original': current_count,
                'target': target_count,
                'generated': len(X_synthetic),
                'methods': {
                    'smote': len(smote_samples),
                    'vae': len(vae_samples),
                    'timeseries': len(ts_samples)
                }
            }
        
        return augmented_X, augmented_y
    
    def _apply_smote(self, X, n_samples):
        """
        Apply SMOTE for synthetic sample generation
        """
        from imblearn.over_sampling import SMOTE
        
        if len(X) < 6:  # SMOTE needs at least 6 samples
            # Use random oversampling for very rare cases
            indices = np.random.choice(len(X), n_samples, replace=True)
            return X[indices] + np.random.normal(0, 0.01, X[indices].shape)
        
        # Create temporary binary problem for SMOTE
        X_temp = np.vstack([X, np.random.randn(len(X) * 2, X.shape[1])])
        y_temp = np.hstack([np.ones(len(X)), np.zeros(len(X) * 2)])
        
        smote = SMOTE(k_neighbors=min(5, len(X)-1), random_state=42)
        X_resampled, _ = smote.fit_resample(X_temp, y_temp)
        
        # Extract synthetic samples
        synthetic_mask = np.arange(len(X_resampled)) >= len(X_temp)
        X_synthetic = X_resampled[synthetic_mask][:n_samples]
        
        return X_synthetic
    
    def _generate_vae_samples(self, X, n_samples):
        """
        Generate samples using Variational Autoencoder
        """
        import tensorflow as tf
        from tensorflow.keras import layers
        
        if self.vae_model is None:
            # Build VAE
            latent_dim = 20
            input_dim = X.shape[1]
            
            # Encoder
            encoder_input = layers.Input(shape=(input_dim,))
            x = layers.Dense(128, activation='relu')(encoder_input)
            x = layers.Dense(64, activation='relu')(x)
            z_mean = layers.Dense(latent_dim)(x)
            z_log_var = layers.Dense(latent_dim)(x)
            
            # Sampling layer
            def sampling(args):
                z_mean, z_log_var = args
                epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            z = layers.Lambda(sampling)([z_mean, z_log_var])
            
            # Decoder
            decoder_input = layers.Input(shape=(latent_dim,))
            x = layers.Dense(64, activation='relu')(decoder_input)
            x = layers.Dense(128, activation='relu')(x)
            decoder_output = layers.Dense(input_dim)(x)
            
            # Models
            encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z])
            decoder = tf.keras.Model(decoder_input, decoder_output)
            
            # VAE
            vae_output = decoder(encoder(encoder_input)[2])
            vae = tf.keras.Model(encoder_input, vae_output)
            
            # Custom loss
            reconstruction_loss = tf.keras.losses.mse(encoder_input, vae_output)
            reconstruction_loss *= input_dim
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss) * -0.5
            vae_loss = reconstruction_loss + kl_loss
            
            vae.add_loss(vae_loss)
            vae.compile(optimizer='adam')
            
            # Train VAE
            vae.fit(X, epochs=50, batch_size=32, verbose=0)
            
            self.vae_model = {
                'encoder': encoder,
                'decoder': decoder,
                'vae': vae,
                'latent_dim': latent_dim
            }
        
        # Generate samples
        latent_samples = np.random.normal(0, 1, (n_samples, self.vae_model['latent_dim']))
        synthetic_samples = self.vae_model['decoder'].predict(latent_samples, verbose=0)
        
        return synthetic_samples
    
    def _apply_timeseries_augmentation(self, X, n_samples):
        """
        Time-series specific augmentation techniques
        """
        augmented = []
        
        for _ in range(n_samples):
            # Randomly select a sample to augment
            idx = np.random.randint(len(X))
            sample = X[idx].copy()
            
            # Apply random augmentation
            aug_type = np.random.choice(['jitter', 'scale', 'warp', 'combine'])
            
            if aug_type == 'jitter':
                # Add small random noise
                noise = np.random.normal(0, 0.02, sample.shape)
                sample += noise
                
            elif aug_type == 'scale':
                # Random scaling
                scale = np.random.uniform(0.9, 1.1)
                sample *= scale
                
            elif aug_type == 'warp':
                # Time warping (simulate different market speeds)
                # This would need sequence data, so we simulate it
                warp_factor = np.random.uniform(0.8, 1.2)
                indices = np.linspace(0, len(sample)-1, len(sample))
                warped_indices = indices * warp_factor
                warped_indices = np.clip(warped_indices, 0, len(sample)-1).astype(int)
                sample = sample[warped_indices]
                
            elif aug_type == 'combine':
                # Combine two samples
                idx2 = np.random.randint(len(X))
                alpha = np.random.uniform(0.7, 0.9)
                sample = alpha * sample + (1 - alpha) * X[idx2]
            
            augmented.append(sample)
        
        return np.array(augmented)
    
    def validate_augmentation(self, X_original, y_original, X_augmented, y_augmented, model):
        """
        Validate that augmentation improves performance
        """
        from sklearn.metrics import classification_report
        
        # Train on original
        model_original = clone(model)
        model_original.fit(X_original, y_original)
        
        # Train on augmented
        model_augmented = clone(model)
        model_augmented.fit(X_augmented, y_augmented)
        
        # Evaluate on test set (should be separate)
        X_test, y_test = self.get_test_data()  # Implement this
        
        print("\nOriginal Data Performance:")
        y_pred_original = model_original.predict(X_test)
        print(classification_report(y_test, y_pred_original))
        
        print("\nAugmented Data Performance:")
        y_pred_augmented = model_augmented.predict(X_test)
        print(classification_report(y_test, y_pred_augmented))
        
        # Calculate improvement per regime
        improvements = {}
        for regime_idx in range(8):
            mask = y_test == regime_idx
            if mask.sum() > 0:
                original_acc = (y_pred_original[mask] == y_test[mask]).mean()
                augmented_acc = (y_pred_augmented[mask] == y_test[mask]).mean()
                improvements[regime_idx] = augmented_acc - original_acc
        
        return improvements
```

---

### Story 9: Feature Selection with mRMR
**Duration:** 2 days  
**Priority:** MEDIUM  
**Impact:** Latency improvement + 1% accuracy  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to select optimal features using mRMR  
**So that** we use only the most informative features

**Acceptance Criteria:**
- [ ] Implement minimum Redundancy Maximum Relevance algorithm
- [ ] Reduce feature set from 1000+ to 500 optimal features
- [ ] No accuracy loss, latency improvement >20%
- [ ] Document selected features with importance scores
- [ ] Create feature dependency graph
- [ ] Implement incremental feature selection

**Implementation:**
```python
class MRMRFeatureSelector:
    """
    Minimum Redundancy Maximum Relevance feature selection
    """
    
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.selected_features = []
        self.feature_scores = {}
        self.redundancy_matrix = None
        
    def fit(self, X, y, feature_names):
        """
        Select features using mRMR algorithm
        """
        from sklearn.feature_selection import mutual_info_classif
        from scipy.stats import pearsonr
        
        n_total_features = X.shape[1]
        
        # Calculate relevance (mutual information with target)
        relevance = mutual_info_classif(X, y)
        
        # Calculate redundancy matrix (correlation between features)
        print("Calculating redundancy matrix...")
        redundancy_matrix = np.zeros((n_total_features, n_total_features))
        
        for i in range(n_total_features):
            for j in range(i+1, n_total_features):
                corr, _ = pearsonr(X[:, i], X[:, j])
                redundancy_matrix[i, j] = abs(corr)
                redundancy_matrix[j, i] = abs(corr)
        
        self.redundancy_matrix = redundancy_matrix
        
        # mRMR selection
        selected_indices = []
        remaining_indices = list(range(n_total_features))
        
        # Select first feature (highest relevance)
        first_feature = np.argmax(relevance)
        selected_indices.append(first_feature)
        remaining_indices.remove(first_feature)
        
        # Iteratively select features
        for _ in range(min(self.n_features - 1, n_total_features - 1)):
            scores = []
            
            for idx in remaining_indices:
                # Relevance term
                rel = relevance[idx]
                
                # Redundancy term (average correlation with selected features)
                if selected_indices:
                    red = np.mean([redundancy_matrix[idx, s] for s in selected_indices])
                else:
                    red = 0
                
                # mRMR score
                score = rel - red
                scores.append(score)
            
            # Select feature with highest score
            best_idx = remaining_indices[np.argmax(scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            if len(selected_indices) % 50 == 0:
                print(f"Selected {len(selected_indices)} features...")
        
        self.selected_features = selected_indices
        self.feature_names = [feature_names[i] for i in selected_indices]
        
        # Calculate final scores
        for idx, name in zip(selected_indices, self.feature_names):
            self.feature_scores[name] = {
                'relevance': relevance[idx],
                'avg_redundancy': np.mean([redundancy_matrix[idx, s] 
                                          for s in selected_indices if s != idx]),
                'mrmr_score': relevance[idx] - np.mean([redundancy_matrix[idx, s] 
                                                        for s in selected_indices if s != idx])
            }
        
        return self
    
    def transform(self, X):
        """
        Transform data to selected features
        """
        return X[:, self.selected_features]
    
    def analyze_feature_groups(self):
        """
        Analyze selected features by component
        """
        component_distribution = {}
        
        for feature_name in self.feature_names:
            # Extract component from feature name
            component = feature_name.split('_')[0]  # Assuming naming convention
            
            if component not in component_distribution:
                component_distribution[component] = []
            
            component_distribution[component].append(feature_name)
        
        print("\nFeature distribution by component:")
        for component, features in component_distribution.items():
            print(f"  {component}: {len(features)} features")
        
        return component_distribution
    
    def create_feature_dependency_graph(self):
        """
        Create graph showing feature relationships
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.Graph()
        
        # Add nodes
        for feature in self.feature_names[:50]:  # Top 50 for visualization
            G.add_node(feature)
        
        # Add edges based on redundancy
        threshold = 0.5  # Correlation threshold for edge
        
        for i, feature1 in enumerate(self.feature_names[:50]):
            for j, feature2 in enumerate(self.feature_names[:50]):
                if i < j:
                    correlation = self.redundancy_matrix[
                        self.selected_features[i], 
                        self.selected_features[j]
                    ]
                    if correlation > threshold:
                        G.add_edge(feature1, feature2, weight=correlation)
        
        # Visualize
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes colored by component
        node_colors = []
        for node in G.nodes():
            component = node.split('_')[0]
            # Assign colors based on component
            color_map = {
                'c1': 'red', 'c2': 'blue', 'c3': 'green', 
                'c4': 'yellow', 'c5': 'purple', 'c6': 'orange',
                'c7': 'brown', 'c8': 'pink'
            }
            node_colors.append(color_map.get(component, 'gray'))
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Draw edges with thickness based on correlation
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
        
        plt.title("Feature Dependency Graph (Top 50 Features)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('feature_dependency_graph.png')
        plt.show()
        
        return G
    
    def incremental_selection(self, X, y, increment=50):
        """
        Incrementally select features to find optimal number
        """
        from sklearn.model_selection import cross_val_score
        
        performance_curve = []
        
        for n_features in range(increment, min(self.n_features, X.shape[1]), increment):
            # Select n_features
            X_subset = X[:, self.selected_features[:n_features]]
            
            # Evaluate performance
            model = XGBClassifier(n_estimators=100, max_depth=6)
            scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy')
            
            performance_curve.append({
                'n_features': n_features,
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std()
            })
            
            print(f"Features: {n_features}, Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Find optimal number (knee point)
        accuracies = [p['mean_accuracy'] for p in performance_curve]
        optimal_idx = self._find_knee_point(accuracies)
        optimal_n_features = performance_curve[optimal_idx]['n_features']
        
        print(f"\nOptimal number of features: {optimal_n_features}")
        
        return performance_curve, optimal_n_features
    
    def _find_knee_point(self, values):
        """
        Find knee point in curve (point of diminishing returns)
        """
        from scipy.spatial.distance import euclidean
        
        # Normalize values
        values = np.array(values)
        x = np.arange(len(values))
        
        # Create line from first to last point
        line_start = np.array([x[0], values[0]])
        line_end = np.array([x[-1], values[-1]])
        
        # Find point with maximum distance to line
        distances = []
        for i, value in enumerate(values):
            point = np.array([x[i], value])
            # Distance from point to line
            distance = np.abs(np.cross(line_end - line_start, line_start - point)) / \
                      np.linalg.norm(line_end - line_start)
            distances.append(distance)
        
        return np.argmax(distances)
```

---

### Story 10: Meta-Feature Engineering
**Duration:** 2 days  
**Priority:** LOW  
**Impact:** +1% accuracy  
**Assignee:** ML Engineer  

**As a** ML engineer  
**I want** to engineer meta-features from model behaviors  
**So that** component disagreements and model uncertainties inform predictions

**Acceptance Criteria:**
- [ ] Create component disagreement scores (8 features)
- [ ] Engineer confidence-weighted signal aggregations (8 features)
- [ ] Add regime stability metrics (5 features)
- [ ] Implement regime transition indicators (5 features)
- [ ] Features show >0.1 mutual information with target
- [ ] Document meta-feature definitions and rationale

**Implementation:**
```python
class MetaFeatureEngineer:
    """
    Engineer features from model and component behaviors
    """
    
    def __init__(self):
        self.meta_features = {}
        self.feature_importance = {}
        
    def engineer_meta_features(self, component_outputs, model_predictions, historical_data):
        """
        Create meta-features from system behavior
        """
        meta_features = {}
        
        # Component Disagreement Features
        disagreement_features = self._calculate_component_disagreement(component_outputs)
        meta_features.update(disagreement_features)
        
        # Confidence-Weighted Signals
        weighted_signals = self._create_weighted_signals(component_outputs)
        meta_features.update(weighted_signals)
        
        # Regime Stability Metrics
        stability_metrics = self._calculate_regime_stability(model_predictions, historical_data)
        meta_features.update(stability_metrics)
        
        # Transition Indicators
        transition_indicators = self._detect_transition_patterns(model_predictions, historical_data)
        meta_features.update(transition_indicators)
        
        # System Coherence Metrics
        coherence_metrics = self._calculate_system_coherence(component_outputs)
        meta_features.update(coherence_metrics)
        
        return pd.DataFrame([meta_features])
    
    def _calculate_component_disagreement(self, component_outputs):
        """
        Measure disagreement between components
        """
        features = {}
        
        # Extract component signals
        signals = [comp['signal'] for comp in component_outputs.values()]
        
        # Pairwise disagreement
        disagreements = []
        for i in range(len(signals)):
            for j in range(i+1, len(signals)):
                disagreements.append(abs(signals[i] - signals[j]))
        
        features['mean_component_disagreement'] = np.mean(disagreements)
        features['max_component_disagreement'] = np.max(disagreements)
        features['std_component_disagreement'] = np.std(disagreements)
        
        # Component-specific disagreement
        for i, (comp_name, comp_data) in enumerate(component_outputs.items()):
            other_signals = [s for j, s in enumerate(signals) if j != i]
            features[f'{comp_name}_disagreement'] = np.mean([
                abs(comp_data['signal'] - s) for s in other_signals
            ])
        
        # Voting disagreement (for classification)
        if 'regime_vote' in list(component_outputs.values())[0]:
            votes = [comp['regime_vote'] for comp in component_outputs.values()]
            vote_counts = pd.Series(votes).value_counts()
            features['vote_entropy'] = -sum([
                (c/len(votes)) * np.log(c/len(votes) + 1e-10) 
                for c in vote_counts
            ])
            features['vote_consensus'] = vote_counts.iloc[0] / len(votes)
        
        return features
    
    def _create_weighted_signals(self, component_outputs):
        """
        Create confidence-weighted aggregate signals
        """
        features = {}
        
        # Extract signals and confidences
        signals = []
        confidences = []
        
        for comp_name, comp_data in component_outputs.items():
            signals.append(comp_data['signal'])
            confidences.append(comp_data.get('confidence', 1.0))
        
        signals = np.array(signals)
        confidences = np.array(confidences)
        
        # Weighted aggregations
        features['confidence_weighted_mean'] = np.sum(signals * confidences) / np.sum(confidences)
        features['confidence_weighted_std'] = np.sqrt(
            np.sum(confidences * (signals - features['confidence_weighted_mean'])**2) / 
            np.sum(confidences)
        )
        
        # Trimmed weighted mean (exclude lowest confidence)
        if len(confidences) > 3:
            trim_idx = np.argsort(confidences)[1:]  # Exclude lowest
            features['trimmed_weighted_mean'] = np.sum(signals[trim_idx] * confidences[trim_idx]) / \
                                               np.sum(confidences[trim_idx])
        
        # Confidence statistics
        features['mean_confidence'] = np.mean(confidences)
        features['min_confidence'] = np.min(confidences)
        features['confidence_range'] = np.max(confidences) - np.min(confidences)
        
        # Per-component weighted signals
        for comp_name, comp_data in component_outputs.items():
            features[f'{comp_name}_weighted'] = comp_data['signal'] * comp_data.get('confidence', 1.0)
        
        return features
    
    def _calculate_regime_stability(self, predictions, historical_data):
        """
        Calculate regime stability metrics
        """
        features = {}
        
        if len(historical_data) < 2:
            return {
                'regime_duration': 0,
                'regime_switches_recent': 0,
                'regime_stability_score': 1.0,
                'regime_persistence_probability': 0.5,
                'regime_volatility': 0
            }
        
        # Current regime duration
        current_regime = predictions[-1]['regime']
        duration = 1
        for i in range(len(predictions) - 2, -1, -1):
            if predictions[i]['regime'] == current_regime:
                duration += 1
            else:
                break
        
        features['regime_duration'] = duration
        
        # Recent regime switches (last 20 periods)
        recent_window = min(20, len(predictions))
        recent_regimes = [p['regime'] for p in predictions[-recent_window:]]
        switches = sum([1 for i in range(1, len(recent_regimes)) 
                       if recent_regimes[i] != recent_regimes[i-1]])
        features['regime_switches_recent'] = switches
        
        # Stability score (inverse of switch frequency)
        features['regime_stability_score'] = 1.0 / (1 + switches / recent_window)
        
        # Persistence probability (based on historical durations)
        if current_regime in historical_data.get('regime_durations', {}):
            historical_durations = historical_data['regime_durations'][current_regime]
            if historical_durations:
                avg_duration = np.mean(historical_durations)
                features['regime_persistence_probability'] = min(1.0, duration / avg_duration)
            else:
                features['regime_persistence_probability'] = 0.5
        else:
            features['regime_persistence_probability'] = 0.5
        
        # Regime volatility (frequency of changes)
        features['regime_volatility'] = switches / recent_window
        
        return features
    
    def _detect_transition_patterns(self, predictions, historical_data):
        """
        Detect patterns indicating regime transitions
        """
        features = {}
        
        if len(predictions) < 5:
            return {
                'confidence_declining': 0,
                'disagreement_increasing': 0,
                'volatility_spike': 0,
                'transition_probability': 0,
                'pre_transition_pattern': 0
            }
        
        # Confidence trend
        recent_confidences = [p.get('confidence', 1.0) for p in predictions[-5:]]
        confidence_trend = np.polyfit(range(5), recent_confidences, 1)[0]
        features['confidence_declining'] = 1 if confidence_trend < -0.02 else 0
        
        # Disagreement trend
        if 'component_disagreement' in predictions[-1]:
            recent_disagreements = [p.get('component_disagreement', 0) for p in predictions[-5:]]
            disagreement_trend = np.polyfit(range(5), recent_disagreements, 1)[0]
            features['disagreement_increasing'] = 1 if disagreement_trend > 0.02 else 0
        else:
            features['disagreement_increasing'] = 0
        
        # Volatility spike detection
        if 'volatility' in predictions[-1]:
            recent_volatility = [p.get('volatility', 0) for p in predictions[-10:]]
            if len(recent_volatility) > 5:
                vol_mean = np.mean(recent_volatility[:-1])
                vol_std = np.std(recent_volatility[:-1])
                current_vol = recent_volatility[-1]
                features['volatility_spike'] = 1 if current_vol > vol_mean + 2*vol_std else 0
            else:
                features['volatility_spike'] = 0
        else:
            features['volatility_spike'] = 0
        
        # Transition probability based on patterns
        transition_score = (
            features['confidence_declining'] * 0.3 +
            features['disagreement_increasing'] * 0.3 +
            features['volatility_spike'] * 0.4
        )
        features['transition_probability'] = transition_score
        
        # Pre-transition pattern detection
        if historical_data.get('transition_patterns'):
            # Check if current pattern matches historical pre-transition patterns
            current_pattern = [
                features['confidence_declining'],
                features['disagreement_increasing'],
                features['volatility_spike']
            ]
            
            pattern_matches = 0
            for historical_pattern in historical_data['transition_patterns']:
                if np.array_equal(current_pattern, historical_pattern[:3]):
                    pattern_matches += 1
            
            features['pre_transition_pattern'] = min(1.0, pattern_matches / 
                                                    max(1, len(historical_data['transition_patterns'])))
        else:
            features['pre_transition_pattern'] = 0
        
        return features
    
    def _calculate_system_coherence(self, component_outputs):
        """
        Calculate overall system coherence metrics
        """
        features = {}
        
        # Signal coherence (how aligned are component signals)
        signals = [comp['signal'] for comp in component_outputs.values()]
        features['signal_coherence'] = 1.0 / (1 + np.std(signals))
        
        # Confidence coherence
        confidences = [comp.get('confidence', 1.0) for comp in component_outputs.values()]
        features['confidence_coherence'] = 1.0 / (1 + np.std(confidences))
        
        # Directional agreement (for components with direction)
        if 'direction' in list(component_outputs.values())[0]:
            directions = [comp['direction'] for comp in component_outputs.values()]
            same_direction = sum([1 for d in directions if d == directions[0]])
            features['directional_agreement'] = same_direction / len(directions)
        
        # Component correlation matrix determinant (system independence)
        if len(signals) > 1:
            signal_matrix = np.array(signals).reshape(-1, 1)
            if signal_matrix.shape[0] > signal_matrix.shape[1]:
                corr_matrix = np.corrcoef(signal_matrix.T)
                if corr_matrix.shape[0] > 1:
                    features['system_independence'] = np.linalg.det(corr_matrix)
                else:
                    features['system_independence'] = 1.0
            else:
                features['system_independence'] = 1.0
        else:
            features['system_independence'] = 1.0
        
        return features
    
    def validate_meta_features(self, meta_features, y_true):
        """
        Validate meta-features using mutual information
        """
        from sklearn.feature_selection import mutual_info_classif
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(meta_features, y_true)
        
        # Create importance dictionary
        feature_names = meta_features.columns
        for name, score in zip(feature_names, mi_scores):
            self.feature_importance[name] = score
        
        # Filter features with MI > 0.1
        significant_features = [name for name, score in self.feature_importance.items() 
                               if score > 0.1]
        
        print(f"\nSignificant meta-features ({len(significant_features)}/{len(feature_names)}):")
        for name in significant_features:
            print(f"  {name}: MI = {self.feature_importance[name]:.3f}")
        
        return significant_features
```

---

## Epic Validation & Testing

### Performance Testing Framework
```python
class Epic6PerformanceValidator:
    """
    Comprehensive testing for Epic 6 implementations
    """
    
    def __init__(self):
        self.test_results = {}
        self.performance_baselines = {
            'accuracy': 0.87,
            'precision': 0.87,
            'latency_ms': 800,
            'memory_mb': 3700
        }
        
    def run_comprehensive_tests(self):
        """
        Run all Epic 6 validation tests
        """
        print("="*50)
        print("EPIC 6 VALIDATION SUITE")
        print("="*50)
        
        # Story-by-story validation
        self.test_results['story_1'] = self.test_feature_interactions()
        self.test_results['story_2'] = self.test_ensemble_performance()
        self.test_results['story_3'] = self.test_selective_inference()
        self.test_results['story_4'] = self.test_calibration()
        self.test_results['story_5'] = self.test_temporal_patterns()
        self.test_results['story_6'] = self.test_robustness()
        self.test_results['story_7'] = self.test_active_learning()
        self.test_results['story_8'] = self.test_augmentation()
        self.test_results['story_9'] = self.test_feature_selection()
        self.test_results['story_10'] = self.test_meta_features()
        
        # Integration tests
        self.test_results['integration'] = self.test_end_to_end()
        
        # Performance benchmarks
        self.test_results['benchmarks'] = self.run_performance_benchmarks()
        
        # Generate report
        self.generate_validation_report()
        
    def test_selective_inference(self):
        """
        Validate selective inference achieves 95% precision
        """
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Test with and without selective inference
        results_without = self.model.predict(X_test)
        results_with = self.selective_inference.predict_with_abstention(X_test)
        
        # Calculate metrics
        predictions_made = [r for r in results_with if r['status'] == 'predicted']
        coverage = len(predictions_made) / len(results_with)
        
        # Precision calculation (would need ground truth)
        precision = self.calculate_precision(predictions_made, y_test)
        
        return {
            'coverage': coverage,
            'precision': precision,
            'target_precision': 0.95,
            'passed': precision >= 0.95
        }
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report
        """
        report = []
        report.append("# Epic 6 Validation Report\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # Summary
        passed_tests = sum(1 for r in self.test_results.values() if r.get('passed', False))
        total_tests = len(self.test_results)
        
        report.append(f"## Summary\n")
        report.append(f"- Tests Passed: {passed_tests}/{total_tests}\n")
        report.append(f"- Overall Status: {'✅ PASSED' if passed_tests == total_tests else '❌ FAILED'}\n\n")
        
        # Detailed results
        report.append("## Detailed Results\n\n")
        
        for test_name, results in self.test_results.items():
            status = '✅' if results.get('passed', False) else '❌'
            report.append(f"### {test_name} {status}\n")
            
            for key, value in results.items():
                if key != 'passed':
                    report.append(f"- {key}: {value}\n")
            
            report.append("\n")
        
        # Performance metrics
        if 'benchmarks' in self.test_results:
            report.append("## Performance Benchmarks\n\n")
            benchmarks = self.test_results['benchmarks']
            
            report.append(f"- Accuracy: {benchmarks['accuracy']:.3f} ")
            report.append(f"(baseline: {self.performance_baselines['accuracy']:.3f})\n")
            
            report.append(f"- Precision: {benchmarks['precision']:.3f} ")
            report.append(f"(target: 0.95)\n")
            
            report.append(f"- Latency: {benchmarks['latency_ms']:.0f}ms ")
            report.append(f"(target: <600ms)\n")
            
            report.append(f"- Memory: {benchmarks['memory_mb']:.0f}MB ")
            report.append(f"(target: <3700MB)\n")
        
        # Save report
        with open('epic6_validation_report.md', 'w') as f:
            f.writelines(report)
        
        print("Validation report saved to epic6_validation_report.md")
```

---

## Monitoring & Maintenance

### Production Monitoring Dashboard
```yaml
monitoring:
  dashboards:
    - name: "Epic 6 - Win Rate Monitor"
      panels:
        - precision_by_regime:
            query: "avg(precision) by (regime) over 1h"
            alert_threshold: 0.93
        - coverage_rate:
            query: "sum(predictions_made) / sum(total_minutes)"
            target: 0.40
        - latency_percentiles:
            query: "histogram_quantile(0.95, latency_ms)"
            alert_threshold: 600
        - component_health:
            query: "up{component=~'c[1-8]'}"
            alert_on: any_down
        
  alerts:
    - name: "Precision Drop"
      condition: "precision < 0.93 for 10m"
      severity: "critical"
      action: "increase_confidence_thresholds"
    
    - name: "Coverage Too Low"
      condition: "coverage < 0.30 for 30m"
      severity: "warning"
      action: "review_thresholds"
    
    - name: "Model Drift Detected"
      condition: "accuracy_rolling_1h < accuracy_baseline - 0.03"
      severity: "warning"
      action: "trigger_retraining"
```

---

## Success Criteria & Validation Gates

### Epic 6 Completion Criteria
- [ ] **Precision**: ≥95% on test set with selective inference
- [ ] **Coverage**: ≥40% of trading minutes classified
- [ ] **Latency**: P95 ≤ 600ms with all enhancements
- [ ] **Robustness**: >90% accuracy with single component failure
- [ ] **Stability**: <2% variance in rolling 1000-prediction window
- [ ] **All Stories**: 10/10 stories completed with acceptance criteria met
- [ ] **Integration**: End-to-end tests passing
- [ ] **Documentation**: Architecture, API, and runbooks updated

### Risk Mitigation
1. **Overfitting**: Aggressive regularization, proper time-series CV
2. **Latency**: Model distillation, feature caching, GPU inference
3. **Complexity**: Incremental rollout, comprehensive testing
4. **Drift**: Online learning, continuous monitoring

## Definition of Done
- [ ] All 10 stories implemented and tested
- [ ] 95% precision achieved with acceptable coverage
- [ ] Performance targets met (latency, memory)
- [ ] Models versioned and registered in Vertex AI
- [ ] Monitoring dashboards live
- [ ] Documentation complete
- [ ] Production deployment approved