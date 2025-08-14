# Epic 7: DTE-Adaptive Reversal Pattern Learning System

**Duration:** Weeks 13-16 (After Epic 6 completion)  
**Status:** Planned  
**Priority:** HIGH - Advanced ML pattern recognition for precision trading  
**Prerequisites:** Epics 1-6 must be completed first

## Epic Goal
Implement a sophisticated DTE-adaptive reversal pattern learning system that combines all 824 features from Components 1-8 with candlestick patterns (15), SMC concepts (5), and DTE-specific machine learning models (DTE 0-30) using Vertex AI infrastructure to achieve >80% reversal prediction accuracy with precise timing predictions.

## Epic Description

### Vision
Create a revolutionary reversal detection system that understands how reversal patterns behave differently at each DTE (Days to Expiry) level. By training 31 separate models (DTE 0 through DTE 30) and leveraging the full power of our 8-component system plus traditional technical patterns, we'll achieve unprecedented accuracy in reversal prediction and timing.

### Core Innovation: DTE Reversal Personalities
Each DTE has distinct reversal characteristics:
- **DTE 0 (Expiry)**: Gamma-driven pin-risk reversals at strikes (Component 2 dominant)
- **DTE 1-7 (Weekly)**: IV skew percentile reversals at VWAP levels (Component 4 dominant)
- **DTE 8-30 (Monthly)**: Correlation breakdown institutional reversals (Component 6 dominant)

### Technical Architecture
```yaml
Total Features: 844
  Component Features: 824
    - Component 1 (Triple Straddle): 120 features
    - Component 2 (Greeks γ=1.5): 98 features
    - Component 3 (OI-PA): 105 features
    - Component 4 (IV Skew Percentiles): 87 features
    - Component 5 (ATR-EMA-CPR): 94 features
    - Component 6 (Correlation): 200+ features
    - Component 7 (Support/Resistance): 72 features
    - Component 8 (Master Integration): 48 features
  
  Candlestick Patterns: 15
    - Bullish: Hammer, Morning Star, Bullish Engulfing, Piercing, Harami, Marubozu, Dragonfly Doji
    - Bearish: Shooting Star, Evening Star, Bearish Engulfing, Dark Cloud, Harami, Marubozu, Gravestone Doji
    - Neutral: Doji
    
  SMC Concepts: 5
    - Break of Structure (BOS)
    - Change of Character (CHoCH)
    - Order Blocks
    - Fair Value Gaps (FVG)
    - Liquidity Pools
    
ML Models: 31 DTE-specific models
  - Individual models for DTE 0, 1, 2, ..., 30
  - Each using 5-model ensemble (LightGBM, CatBoost, TabNet, LSTM, Transformer)
```

### Integration with Existing Infrastructure
- **Data Pipeline**: Leverages Epic 2's BigQuery and Feature Store
- **ML Infrastructure**: Uses Epic 3's Vertex AI training and serving
- **Feature Engineering**: Builds on Epic 1's 824 component features
- **Real-time Serving**: Epic 4's production endpoints
- **Monitoring**: Epic 5's observability framework

## Success Criteria
- **Accuracy**: >80% reversal detection accuracy for DTE 0-1, >70% for DTE 2-7, >65% for DTE 8-30
- **Timing Precision**: Predict reversal timing within 5 minutes for DTE 0, 15 minutes for DTE 1-7
- **Performance**: <200ms inference latency with Feature Store integration
- **Reliability**: 99.9% uptime with automatic failover
- **Backtesting**: Validated on 7 years of historical data

## Epic Stories

### Story 1: Candlestick Pattern Feature Engineering (15 Patterns)
**As a** quantitative developer  
**I want** to implement comprehensive candlestick pattern detection integrated with component signals  
**So that** traditional reversal patterns are enhanced with options market intelligence

**Acceptance Criteria:**
- [ ] Implement 15 candlestick patterns using TA-Lib or custom logic
- [ ] Create pattern strength scoring based on volume and volatility context
- [ ] Integrate pattern signals with Component 7 S&R levels for confluence
- [ ] Generate pattern occurrence features for each DTE bucket
- [ ] Validate patterns against 7 years of historical data
- [ ] Performance: Pattern detection <50ms per timeframe

**Technical Details:**
```python
CANDLESTICK_FEATURES = {
    'bullish_patterns': [
        'hammer_score',           # 0-1 strength with volume confirmation
        'morning_star_score',      # Multi-candle pattern score
        'bullish_engulfing_score', # Size and volume weighted
        'piercing_pattern_score',  # Penetration depth factor
        'bullish_harami_score',    # Inside bar consideration
        'bullish_marubozu_score',  # Trending strength
        'dragonfly_doji_score'     # Reversal at support
    ],
    'bearish_patterns': [
        'shooting_star_score',
        'evening_star_score',
        'bearish_engulfing_score',
        'dark_cloud_score',
        'bearish_harami_score',
        'bearish_marubozu_score',
        'gravestone_doji_score'
    ],
    'neutral_patterns': ['doji_score']
}
```

---

### Story 2: SMC Concepts Integration (5 Indicators)
**As a** trading system developer  
**I want** Smart Money Concepts integrated with component-based analysis  
**So that** institutional trading patterns enhance reversal detection

**Acceptance Criteria:**
- [ ] Implement Break of Structure (BOS) detection with multi-timeframe validation
- [ ] Implement Change of Character (CHoCH) with trend transition scoring
- [ ] Identify Order Blocks with volume profile integration
- [ ] Detect Fair Value Gaps (FVG) with fill probability scoring
- [ ] Map Liquidity Pools to Component 7 S&R levels
- [ ] Create SMC confluence score for reversal validation

**Integration with Components:**
```python
SMC_COMPONENT_MAPPING = {
    'BOS': {
        'primary_component': 'Component_7_SR',  # S&R break validation
        'secondary_component': 'Component_3_OI',  # Institutional confirmation
        'weight_by_dte': {
            'dte_0': 0.3,    # Less reliable at expiry
            'dte_1_7': 0.5,  # Most reliable weekly
            'dte_8_30': 0.4  # Moderate monthly
        }
    },
    'CHoCH': {
        'primary_component': 'Component_6_Correlation',  # Regime change
        'secondary_component': 'Component_1_Straddle',   # Momentum shift
    },
    'Order_Blocks': {
        'primary_component': 'Component_3_OI_PA',  # Institutional levels
        'validation': 'Component_7_SR_confluence'
    }
}
```

---

### Story 3: DTE-Specific Feature Engineering (31 DTE Buckets)
**As a** ML engineer  
**I want** DTE-specific feature sets optimized for each expiry timeframe  
**So that** models learn unique reversal characteristics at each DTE level

**Acceptance Criteria:**
- [ ] Create 31 distinct feature sets (DTE 0 through DTE 30)
- [ ] Implement DTE-specific feature weighting schemes
- [ ] Generate temporal features (time to expiry, decay curves)
- [ ] Create DTE transition features (DTE 1→0, 8→7, etc.)
- [ ] Implement feature importance ranking per DTE
- [ ] Document optimal feature subsets for each DTE range

**DTE Feature Architecture:**
```python
DTE_FEATURE_WEIGHTS = {
    'dte_0': {  # Expiry day
        'component_2_gamma': 0.25,      # Highest weight
        'component_7_sr': 0.20,         # Pin risk levels
        'component_1_straddle': 0.15,   # Straddle price action
        'candlestick_patterns': 0.10,   # Quick reversals
        'other_components': 0.30
    },
    'dte_1_7': {  # Weekly options
        'component_4_iv_skew': 0.20,    # IV percentile regimes
        'component_1_straddle': 0.18,   # Straddle momentum
        'component_5_atr_ema': 0.15,    # Volatility regimes
        'candlestick_patterns': 0.15,   # Pattern reliability
        'other_components': 0.32
    },
    'dte_8_30': {  # Monthly options
        'component_6_correlation': 0.22, # Correlation breakdowns
        'component_3_oi_pa': 0.20,      # Institutional flow
        'component_8_integration': 0.15, # Regime transitions
        'smc_concepts': 0.13,           # Smart money patterns
        'other_components': 0.30
    }
}
```

---

### Story 4: Component Cascade Detection System
**As a** system architect  
**I want** to detect how reversals cascade through components in sequence  
**So that** we can predict reversal timing with high precision

**Acceptance Criteria:**
- [ ] Implement cascade sequence detection for each DTE range
- [ ] Create cascade velocity measurements (component to component lag)
- [ ] Build cascade confidence scoring system
- [ ] Implement early warning system based on cascade initiation
- [ ] Validate cascade patterns across different market regimes
- [ ] Create cascade visualization for monitoring

**Cascade Sequences:**
```python
REVERSAL_CASCADE_SEQUENCES = {
    'dte_0_cascade': [
        ('component_2_gamma_spike', 0),        # T+0 minutes
        ('component_7_sr_penetration', 2),     # T+2 minutes
        ('component_1_straddle_reversal', 5),  # T+5 minutes
        ('price_reversal_confirmed', 8)        # T+8 minutes
    ],
    'dte_1_7_cascade': [
        ('component_4_iv_regime_shift', 0),    # T+0 minutes
        ('component_6_correlation_break', 10), # T+10 minutes
        ('component_3_flow_reversal', 20),     # T+20 minutes
        ('price_reversal_confirmed', 30)       # T+30 minutes
    ],
    'dte_8_30_cascade': [
        ('component_6_correlation_breakdown', 0),  # T+0 minutes
        ('component_3_institutional_shift', 30),   # T+30 minutes
        ('component_5_volatility_regime', 60),     # T+60 minutes
        ('price_reversal_confirmed', 90)           # T+90 minutes
    ]
}
```

---

### Story 5: 5-Model ML Ensemble Implementation
**As a** ML engineer  
**I want** to implement the 5-model ensemble for each DTE bucket  
**So that** we leverage multiple ML architectures for robust predictions

**Acceptance Criteria:**
- [ ] Implement LightGBM models for gradient boosting (feature relationships)
- [ ] Implement CatBoost models for categorical features (regime classification)
- [ ] Implement TabNet for deep tabular learning (complex interactions)
- [ ] Implement LSTM for temporal sequences (pattern evolution)
- [ ] Implement Transformer for multi-timeframe attention (timeframe confluence)
- [ ] Create ensemble stacking with meta-learner
- [ ] Implement model versioning and A/B testing framework

**Model Configuration:**
```python
ENSEMBLE_CONFIG = {
    'lightgbm': {
        'purpose': 'Feature relationship learning',
        'hyperparameters': {
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'feature_fraction': 0.8
        }
    },
    'catboost': {
        'purpose': 'Categorical pattern recognition',
        'categorical_features': ['dte_bucket', 'regime_type', 'pattern_type'],
        'hyperparameters': {
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.03
        }
    },
    'tabnet': {
        'purpose': 'Deep feature interaction learning',
        'hyperparameters': {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'momentum': 0.98
        }
    },
    'lstm': {
        'purpose': 'Temporal pattern sequence modeling',
        'architecture': {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'sequence_length': 60  # 60 minutes lookback
        }
    },
    'transformer': {
        'purpose': 'Multi-timeframe attention analysis',
        'architecture': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 512
        }
    }
}
```

---

### Story 6: Reversal Fingerprint Learning System
**As a** pattern recognition specialist  
**I want** to learn unique fingerprints for different reversal types  
**So that** each reversal pattern is identified with high confidence

**Acceptance Criteria:**
- [ ] Define reversal taxonomy (V-reversal, double top/bottom, failed breakout, gap reversal)
- [ ] Create fingerprint feature extraction for each reversal type
- [ ] Implement fingerprint matching algorithm with similarity scoring
- [ ] Build fingerprint database with historical examples
- [ ] Create confidence scoring based on fingerprint match quality
- [ ] Implement online learning for new fingerprint patterns

**Reversal Fingerprints:**
```python
REVERSAL_FINGERPRINTS = {
    'v_reversal': {
        'characteristics': {
            'speed': 'fast',  # <15 minutes
            'component_signature': {
                'component_2_gamma': 'spike',
                'component_7_sr': 'sharp_penetration',
                'component_1_straddle': 'momentum_flip',
                'candlestick': 'hammer_or_shooting_star'
            },
            'volume_profile': 'climactic',
            'retracement': None  # No retracement
        }
    },
    'double_top_bottom': {
        'characteristics': {
            'speed': 'slow',  # >60 minutes
            'component_signature': {
                'component_6_correlation': 'gradual_breakdown',
                'component_5_atr_ema': 'divergence',
                'component_3_oi': 'distribution',
                'candlestick': 'evening_morning_star'
            },
            'volume_profile': 'declining',
            'retracement': '38.2_to_50_percent'
        }
    },
    'failed_breakout_reversal': {
        'characteristics': {
            'speed': 'medium',  # 30-45 minutes
            'component_signature': {
                'component_7_sr': 'false_break',
                'component_3_oi': 'trap_pattern',
                'component_4_iv': 'volatility_collapse',
                'smc': 'liquidity_grab'
            },
            'volume_profile': 'exhaustion',
            'retracement': 'full_retracement'
        }
    }
}
```

---

### Story 7: Vertex AI Training Pipeline
**As a** ML operations engineer  
**I want** automated training pipelines for DTE-specific models  
**So that** models are continuously improved with new data

**Acceptance Criteria:**
- [ ] Create Vertex AI Pipeline for automated training
- [ ] Implement data versioning and feature store integration
- [ ] Set up hyperparameter tuning with Vertex AI Vizier
- [ ] Configure distributed training for large models
- [ ] Implement model evaluation and comparison framework
- [ ] Create automated model promotion based on performance metrics

**Training Pipeline Specification:**
```yaml
pipeline:
  name: dte_reversal_training_pipeline
  
  steps:
    1_data_preparation:
      - source: BigQuery training_dataset
      - feature_selection: DTE-specific subsets
      - train_test_split: 80/20 with time-based validation
      - scaling: StandardScaler with feature store stats
      
    2_model_training:
      - parallel_training: 31 DTE models
      - ensemble_models: 5 per DTE
      - total_models: 155 (31 × 5)
      - compute: GPU-enabled for deep learning
      
    3_model_evaluation:
      - metrics: accuracy, precision, recall, timing_error
      - backtesting: 7 years historical data
      - cross_validation: Time series CV
      
    4_model_registration:
      - registry: Vertex AI Model Registry
      - versioning: Semantic versioning
      - metadata: Performance metrics, feature importance
      
    5_deployment_decision:
      - criteria: >80% accuracy for DTE 0-1
      - comparison: Against current production
      - rollout: Gradual with A/B testing
```

---

### Story 8: Real-time Inference Architecture
**As a** systems engineer  
**I want** low-latency inference for reversal predictions  
**So that** traders receive timely reversal signals

**Acceptance Criteria:**
- [ ] Design inference pipeline with <200ms latency
- [ ] Implement feature caching for hot features
- [ ] Create batch prediction for multiple DTEs
- [ ] Build fallback mechanism for model failures
- [ ] Implement prediction explanation (SHAP values)
- [ ] Create confidence calibration for predictions

**Inference Architecture:**
```python
INFERENCE_PIPELINE = {
    'feature_retrieval': {
        'online_features': {
            'source': 'Vertex AI Feature Store',
            'features': 32,  # Core features
            'latency': '<50ms'
        },
        'cached_features': {
            'source': 'Redis Cache',
            'features': 100,  # Frequently used
            'latency': '<10ms'
        },
        'computed_features': {
            'source': 'Real-time computation',
            'features': 712,  # Remaining features
            'latency': '<140ms'
        }
    },
    'model_serving': {
        'endpoint': 'Vertex AI Endpoint',
        'auto_scaling': {
            'min_replicas': 2,
            'max_replicas': 10,
            'target_utilization': 70
        },
        'batch_size': 32,  # Process multiple DTEs together
        'timeout': 500  # milliseconds
    },
    'post_processing': {
        'confidence_calibration': 'Isotonic regression',
        'explanation': 'SHAP values for top features',
        'filtering': 'Minimum confidence threshold'
    }
}
```

---

### Story 9: Backtesting Framework
**As a** quantitative analyst  
**I want** comprehensive backtesting of reversal predictions  
**So that** we validate model performance across market conditions

**Acceptance Criteria:**
- [ ] Implement event-based backtesting framework
- [ ] Create DTE-specific performance metrics
- [ ] Build regime-based performance analysis
- [ ] Implement transaction cost modeling
- [ ] Create performance attribution analysis
- [ ] Generate backtesting reports and visualizations

**Backtesting Metrics:**
```python
BACKTEST_METRICS = {
    'accuracy_metrics': {
        'reversal_detection_rate': 'True positives / Total reversals',
        'false_positive_rate': 'False signals / Total signals',
        'timing_accuracy': 'Average timing error in minutes'
    },
    'trading_metrics': {
        'win_rate': 'Profitable reversals / Total trades',
        'risk_reward_ratio': 'Average win / Average loss',
        'maximum_drawdown': 'Largest peak to trough',
        'sharpe_ratio': 'Risk-adjusted returns'
    },
    'dte_specific_metrics': {
        'dte_0_accuracy': 'Target >80%',
        'dte_1_7_accuracy': 'Target >70%',
        'dte_8_30_accuracy': 'Target >65%',
        'timing_precision': {
            'dte_0': '±5 minutes',
            'dte_1_7': '±15 minutes',
            'dte_8_30': '±60 minutes'
        }
    }
}
```

---

### Story 10: Monitoring and Alerting System
**As a** trading operations manager  
**I want** real-time monitoring of reversal predictions  
**So that** we can detect and respond to model degradation quickly

**Acceptance Criteria:**
- [ ] Create real-time performance monitoring dashboards
- [ ] Implement drift detection for features and predictions
- [ ] Set up alerting for accuracy degradation
- [ ] Build model health checks and diagnostics
- [ ] Create audit trail for all predictions
- [ ] Implement feedback loop for model improvement

**Monitoring Framework:**
```yaml
monitoring:
  dashboards:
    - reversal_accuracy_dashboard:
        metrics: [accuracy, precision, recall, f1_score]
        grouping: [dte_bucket, reversal_type, timeframe]
        refresh: real_time
        
    - timing_precision_dashboard:
        metrics: [timing_error, early_signals, late_signals]
        grouping: [dte_bucket, component_cascade]
        refresh: 5_minutes
        
    - model_health_dashboard:
        metrics: [latency, throughput, error_rate]
        grouping: [model_type, dte_bucket]
        refresh: 1_minute
        
  alerts:
    - accuracy_degradation:
        condition: accuracy < threshold - 10%
        severity: high
        action: notify_team + investigate
        
    - feature_drift:
        condition: KL_divergence > 0.1
        severity: medium
        action: retrain_model
        
    - cascade_anomaly:
        condition: cascade_sequence_broken
        severity: high
        action: investigate_components
```

---

### Story 11: A/B Testing Framework
**As a** product manager  
**I want** to test new reversal models safely in production  
**So that** we can validate improvements before full rollout

**Acceptance Criteria:**
- [ ] Implement traffic splitting for model comparison
- [ ] Create performance comparison framework
- [ ] Build statistical significance testing
- [ ] Implement gradual rollout mechanism
- [ ] Create rollback capabilities
- [ ] Generate A/B test reports

---

### Story 12: Documentation and Training
**As a** team lead  
**I want** comprehensive documentation of the reversal system  
**So that** the team can maintain and improve the system

**Acceptance Criteria:**
- [ ] Create technical documentation for all components
- [ ] Build user guide for traders
- [ ] Document model training procedures
- [ ] Create troubleshooting guide
- [ ] Build knowledge base of reversal patterns
- [ ] Develop training materials for new team members

---

## Technical Architecture

### System Components
```
┌─────────────────────────────────────────────────────────────┐
│                 DTE-Adaptive Reversal System                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ Feature Layer   │  │  ML Layer       │  │ Serving    │ │
│  │                 │  │                 │  │ Layer      │ │
│  │ • 824 Component │→ │ • 31 DTE Models │→ │ • Vertex   │ │
│  │ • 15 Candles    │  │ • 5-Model       │  │   AI       │ │
│  │ • 5 SMC         │  │   Ensemble      │  │ • <200ms   │ │
│  └─────────────────┘  └─────────────────┘  └────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Performance Targets                     │   │
│  │  • DTE 0: >80% accuracy, ±5 min timing             │   │
│  │  • DTE 1-7: >70% accuracy, ±15 min timing          │   │
│  │  • DTE 8-30: >65% accuracy, ±60 min timing         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
Market Data → Components 1-8 → Feature Engineering (844 features)
                                        ↓
                            DTE Bucketing (0-30)
                                        ↓
                        DTE-Specific Model Selection
                                        ↓
                            5-Model Ensemble Prediction
                                        ↓
                        Reversal Signal + Timing + Confidence
```

## Dependencies

### Prerequisites (Must be completed first)
- **Epic 1**: Feature Engineering Foundation (824 component features)
- **Epic 2**: Data Pipeline Modernization (BigQuery, Feature Store)
- **Epic 3**: ML Pipeline and Model Deployment
- **Epic 4**: Production Readiness
- **Epic 5**: Monitoring and Observability
- **Epic 6**: Strategy Integration

### External Dependencies
- Historical data: 7 years of options and futures data
- Vertex AI quotas: Sufficient for 155 models
- BigQuery storage: ~500GB for training data
- Feature Store: Configured for 844 features

## Risk Mitigation

### Technical Risks
1. **Model Overfitting**
   - Mitigation: Rigorous cross-validation, regularization, ensemble diversity
   
2. **Latency Requirements**
   - Mitigation: Feature caching, model optimization, edge deployment

3. **Data Quality**
   - Mitigation: Comprehensive data validation, outlier detection

### Business Risks
1. **False Signals**
   - Mitigation: Confidence thresholds, human oversight for large trades

2. **Market Regime Changes**
   - Mitigation: Continuous learning, regime detection, adaptive models

## Success Metrics

### Technical Metrics
- Model accuracy by DTE bucket
- Timing precision in minutes
- Inference latency (p50, p95, p99)
- Feature importance stability
- Model drift indicators

### Business Metrics
- Trading win rate improvement
- Risk-adjusted returns (Sharpe ratio)
- Maximum drawdown reduction
- False signal cost reduction

## Timeline

### Phase 1: Foundation (Weeks 13-14)
- Stories 1-3: Pattern features and DTE engineering
- Stories 4-6: Cascade and fingerprint systems

### Phase 2: ML Implementation (Weeks 15-16)
- Stories 7-8: Training and inference
- Stories 9-10: Backtesting and monitoring

### Phase 3: Production (Post Epic 6)
- Stories 11-12: A/B testing and documentation
- Production rollout with gradual scaling

## Budget Estimation

### Compute Resources
- Training: 155 models × $50 = $7,750 initial
- Retraining: $2,000/month ongoing
- Inference: $3,000/month (auto-scaling)

### Storage
- BigQuery: $500/month
- Feature Store: $1,000/month
- Model Registry: $200/month

### Total Monthly: ~$6,700

## Definition of Done

- [ ] All 31 DTE models trained and deployed
- [ ] 844 features integrated and validated
- [ ] Backtesting shows target accuracy achieved
- [ ] Inference latency <200ms confirmed
- [ ] Monitoring dashboards operational
- [ ] Documentation complete
- [ ] Team trained on system operation
- [ ] Production rollout successful with positive ROI

## Future Enhancements (Post-Epic 7)

1. **Cross-Asset Reversal Learning**
   - Apply to BANKNIFTY, FINNIFTY
   - Currency and commodity options

2. **Advanced Architectures**
   - Graph Neural Networks for component relationships
   - Reinforcement Learning for timing optimization

3. **Real-time Adaptation**
   - Online learning from live trades
   - Dynamic confidence calibration

4. **Integration Expansions**
   - Connect to execution systems
   - Risk management integration

## Appendix: Component Integration Details

### Component-Specific Reversal Signals

#### Component 1 (Triple Straddle)
- Straddle momentum reversal patterns
- ATM/ITM/OTM divergence signals
- EMA/VWAP confluence on straddle prices

#### Component 2 (Greeks Sentiment γ=1.5)
- Gamma explosion at expiry (DTE 0)
- Pin risk reversal zones
- Vanna/Charm effects near strikes

#### Component 3 (OI-PA Trending)
- Institutional flow reversals
- Cumulative ATM±7 divergence
- Volume-weighted position changes

#### Component 4 (IV Skew Percentiles)
- 7-regime IV classification shifts
- Skew inversion signals
- Term structure reversals

#### Component 5 (ATR-EMA-CPR)
- ATR band penetration reversals
- EMA confluence breaks
- CPR width changes

#### Component 6 (Correlation/Predictive)
- Correlation breakdown (<0.3)
- Gap prediction accuracy
- Cross-component validation failures

#### Component 7 (Support/Resistance)
- 72 features for level identification
- Touch count exhaustion
- Multi-timeframe confluence breaks

#### Component 8 (Master Integration)
- Regime transition signals
- Component weight shifts
- Confidence score changes

---

*This epic represents the culmination of our market regime system, bringing together all components with advanced ML to achieve unprecedented reversal detection accuracy. Implementation should begin only after Epics 1-6 are successfully completed and validated.*