-- Component 8: Master Integration Features
-- Total features: 48
-- Purpose: Integration and orchestration of all components

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c8_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c8_component_agreement_score FLOAT64,
  c8_integration_confidence FLOAT64,
  c8_transition_probability_hint FLOAT64,
  
  -- Offline-only features
  -- Component weights
  c8_weight_c1 FLOAT64,
  c8_weight_c2 FLOAT64,
  c8_weight_c3 FLOAT64,
  c8_weight_c4 FLOAT64,
  c8_weight_c5 FLOAT64,
  c8_weight_c6 FLOAT64,
  c8_weight_c7 FLOAT64,
  
  -- Component scores
  c8_score_c1 FLOAT64,
  c8_score_c2 FLOAT64,
  c8_score_c3 FLOAT64,
  c8_score_c4 FLOAT64,
  c8_score_c5 FLOAT64,
  c8_score_c6 FLOAT64,
  c8_score_c7 FLOAT64,
  
  -- Regime classification
  c8_regime_classification INT64,  -- 1=Trending, 2=Volatile, 3=Ranging, 4=Breakout
  c8_regime_probability_trending FLOAT64,
  c8_regime_probability_volatile FLOAT64,
  c8_regime_probability_ranging FLOAT64,
  c8_regime_probability_breakout FLOAT64,
  
  -- Component agreement metrics
  c8_component_correlation FLOAT64,
  c8_component_divergence FLOAT64,
  c8_unanimous_signal_strength FLOAT64,
  c8_conflicting_signals_count INT64,
  
  -- Integration quality metrics
  c8_data_quality_score FLOAT64,
  c8_feature_completeness FLOAT64,
  c8_computation_latency_ms FLOAT64,
  c8_integration_health_score FLOAT64,
  
  -- Ensemble predictions
  c8_ensemble_direction_prediction INT64,  -- -1=Down, 0=Neutral, 1=Up
  c8_ensemble_magnitude_prediction FLOAT64,
  c8_ensemble_confidence FLOAT64,
  c8_prediction_disagreement_score FLOAT64,
  
  -- Adaptive learning metrics
  c8_learning_rate FLOAT64,
  c8_adaptation_speed FLOAT64,
  c8_model_drift_score FLOAT64,
  c8_retraining_urgency FLOAT64,
  
  -- System performance metrics
  c8_total_processing_time_ms FLOAT64,
  c8_memory_usage_mb FLOAT64,
  c8_cpu_utilization_pct FLOAT64,
  c8_gpu_utilization_pct FLOAT64,
  
  -- Meta-features
  c8_feature_importance_variance FLOAT64,
  c8_component_contribution_entropy FLOAT64,
  c8_signal_noise_ratio FLOAT64,
  c8_information_gain FLOAT64,
  
  -- Alert generation
  c8_alert_level INT64,  -- 0=None, 1=Low, 2=Medium, 3=High, 4=Critical
  c8_alert_type STRING,  -- JSON array of alert types
  c8_action_recommendation STRING,  -- Recommended action based on analysis
  
  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 8: Master Integration - 48 features for component orchestration",
  labels=[("component", "c8"), ("feature_count", "48"), ("version", "1.0")]
);