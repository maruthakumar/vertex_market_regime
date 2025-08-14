-- Training Dataset: Joined view/table across all component features (Phase 2 Enhanced)
-- Total features: 932 from all 8 components (Phase 1: 872 + Phase 2: 60 momentum enhancements)
-- Purpose: Denormalized dataset for ML model training with momentum-correlation-support/resistance synergy

-- Option 1: Create as View (for always-fresh data)
CREATE OR REPLACE VIEW `arched-bot-269016.market_regime_{env}.training_dataset` AS
SELECT 
  -- Common columns
  c1.symbol,
  c1.ts_minute,
  c1.date,
  c1.dte,
  c1.zone_name,
  
  -- Component 1 features (150 features - includes 30 momentum features)
  c1.c1_momentum_score,
  c1.c1_vol_compression,
  c1.c1_breakout_probability,
  c1.c1_straddle_pct_chg_0dte,
  c1.c1_straddle_pct_chg_3dte,
  c1.c1_straddle_pct_chg_7dte,
  c1.c1_straddle_pct_chg_14dte,
  c1.c1_straddle_pct_chg_21dte,
  c1.c1_straddle_pct_chg_30dte,
  c1.c1_straddle_volume_0dte,
  c1.c1_straddle_volume_3dte,
  c1.c1_straddle_volume_7dte,
  c1.c1_straddle_volume_14dte,
  c1.c1_straddle_volume_21dte,
  c1.c1_straddle_volume_30dte,
  c1.c1_straddle_oi_0dte,
  c1.c1_straddle_oi_3dte,
  c1.c1_straddle_oi_7dte,
  c1.c1_straddle_oi_14dte,
  c1.c1_straddle_oi_21dte,
  c1.c1_straddle_oi_30dte,
  c1.c1_straddle_ma_5min,
  c1.c1_straddle_ma_15min,
  c1.c1_straddle_ma_30min,
  c1.c1_straddle_ma_60min,
  c1.c1_correlation_0_3dte,
  c1.c1_correlation_0_7dte,
  c1.c1_correlation_3_7dte,
  c1.c1_correlation_7_14dte,
  c1.c1_correlation_14_21dte,
  c1.c1_correlation_21_30dte,
  c1.c1_dte_momentum_score,
  c1.c1_dte_divergence_indicator,
  c1.c1_dte_volatility_term_structure,
  c1.c1_zone_momentum_open,
  c1.c1_zone_momentum_mid_morn,
  c1.c1_zone_momentum_lunch,
  c1.c1_zone_momentum_afternoon,
  c1.c1_zone_momentum_close,
  c1.c1_breakout_strength,
  c1.c1_consolidation_score,
  c1.c1_range_expansion_probability,
  c1.c1_volume_intensity,
  c1.c1_volume_weighted_momentum,
  c1.c1_abnormal_volume_flag,
  c1.c1_bid_ask_spread_avg,
  c1.c1_market_depth_score,
  c1.c1_liquidity_score,
  c1.c1_regime_stability_score,
  c1.c1_transition_probability,
  c1.c1_regime_confidence,
  c1.c1_z_score_5min,
  c1.c1_z_score_15min,
  c1.c1_z_score_60min,
  c1.c1_percentile_rank_5min,
  c1.c1_percentile_rank_15min,
  c1.c1_percentile_rank_60min,
  c1.c1_rsi_5min,
  c1.c1_rsi_15min,
  c1.c1_macd_signal,
  c1.c1_macd_histogram,
  c1.c1_realized_vol_5min,
  c1.c1_realized_vol_15min,
  c1.c1_realized_vol_60min,
  c1.c1_vol_of_vol,
  c1.c1_beta_to_nifty,
  c1.c1_beta_to_banknifty,
  c1.c1_correlation_to_vix,
  c1.c1_event_impact_score,
  c1.c1_pre_event_positioning,
  c1.c1_post_event_momentum,
  c1.c1_entropy_score,
  c1.c1_fractal_dimension,
  c1.c1_hurst_exponent,
  c1.c1_lyapunov_exponent,
  c1.c1_order_flow_imbalance,
  c1.c1_toxic_flow_probability,
  c1.c1_smart_money_indicator,
  c1.c1_day_of_week_effect,
  c1.c1_time_of_day_effect,
  c1.c1_monthly_seasonality,
  c1.c1_expiry_effect,
  c1.c1_var_95,
  c1.c1_cvar_95,
  c1.c1_max_drawdown,
  c1.c1_sharpe_ratio,
  c1.c1_ml_momentum_prediction,
  c1.c1_ml_volatility_forecast,
  c1.c1_ml_regime_probability,
  c1.c1_ml_confidence_score,
  
  -- Component 1 Phase 2: Momentum features (30 features)
  c1.c1_rsi_3min_trend,
  c1.c1_rsi_3min_strength,
  c1.c1_rsi_3min_signal,
  c1.c1_rsi_3min_normalized,
  c1.c1_rsi_5min_trend,
  c1.c1_rsi_5min_strength,
  c1.c1_rsi_5min_signal,
  c1.c1_rsi_5min_normalized,
  c1.c1_rsi_10min_trend,
  c1.c1_rsi_10min_strength,
  c1.c1_rsi_10min_signal,
  c1.c1_rsi_10min_normalized,
  c1.c1_rsi_15min_trend,
  c1.c1_rsi_15min_strength,
  c1.c1_rsi_combined_consensus,
  c1.c1_macd_3min_signal,
  c1.c1_macd_3min_histogram,
  c1.c1_macd_3min_crossover,
  c1.c1_macd_5min_signal,
  c1.c1_macd_5min_histogram,
  c1.c1_macd_5min_crossover,
  c1.c1_macd_10min_signal,
  c1.c1_macd_10min_histogram,
  c1.c1_macd_15min_signal,
  c1.c1_macd_consensus_strength,
  c1.c1_momentum_3min_5min_divergence,
  c1.c1_momentum_5min_10min_divergence,
  c1.c1_momentum_10min_15min_divergence,
  c1.c1_momentum_consensus_score,
  c1.c1_momentum_regime_strength,
  
  -- Component 2 features (98 features)
  c2.c2_gamma_exposure,
  c2.c2_sentiment_level,
  c2.c2_pin_risk_score,
  c2.c2_gamma_weighted,
  c2.c2_delta,
  c2.c2_vega,
  c2.c2_theta,
  c2.c2_rho,
  c2.c2_vanna,
  c2.c2_charm,
  c2.c2_volga,
  c2.c2_veta,
  c2.c2_speed,
  c2.c2_zomma,
  c2.c2_color,
  -- (continuing with all c2 features... abbreviated for readability)
  
  -- Component 3 features (105 features)
  c3.c3_institutional_flow_score,
  c3.c3_divergence_type,
  c3.c3_range_expansion_score,
  -- (all c3 features... abbreviated for readability)
  
  -- Component 4 features (87 features)
  c4.c4_skew_bias_score,
  c4.c4_term_structure_signal,
  c4.c4_iv_regime_level,
  -- (all c4 features... abbreviated for readability)
  
  -- Component 5 features (94 features)
  c5.c5_momentum_score,
  c5.c5_volatility_regime_score,
  c5.c5_confluence_score,
  -- (all c5 features... abbreviated for readability)
  
  -- Component 6 features (220 features - includes 20 momentum-enhanced correlation features)
  c6.c6_correlation_agreement_score,
  c6.c6_breakdown_alert,
  c6.c6_system_stability_score,
  -- (all c6 features including correlation matrix... abbreviated for readability)
  
  -- Component 6 Phase 2: Momentum-enhanced correlation features (20 features)
  c6.c6_rsi_cross_correlation_3min,
  c6.c6_rsi_cross_correlation_5min,
  c6.c6_rsi_price_agreement_3min,
  c6.c6_rsi_price_agreement_5min,
  c6.c6_rsi_regime_coherence_3min,
  c6.c6_rsi_regime_coherence_5min,
  c6.c6_rsi_divergence_3min_5min,
  c6.c6_rsi_divergence_5min_10min,
  c6.c6_macd_signal_correlation_3min,
  c6.c6_macd_signal_correlation_5min,
  c6.c6_macd_histogram_convergence_3min,
  c6.c6_macd_histogram_convergence_5min,
  c6.c6_macd_trend_agreement_3min,
  c6.c6_macd_trend_agreement_5min,
  c6.c6_macd_momentum_strength_3min,
  c6.c6_macd_momentum_strength_5min,
  c6.c6_multi_timeframe_rsi_consensus,
  c6.c6_multi_timeframe_macd_consensus,
  c6.c6_cross_component_momentum_agreement,
  c6.c6_overall_momentum_system_coherence,
  
  -- Component 7 features (130 features - includes 10 momentum-based level features)
  c7.c7_level_strength_score,
  c7.c7_breakout_probability,
  -- (all c7 features... abbreviated for readability)
  
  -- Component 7 Phase 2: Momentum-based level detection features (10 features)
  c7.c7_rsi_overbought_resistance_strength,
  c7.c7_rsi_oversold_support_strength,
  c7.c7_rsi_neutral_zone_level_density,
  c7.c7_rsi_level_convergence_strength,
  c7.c7_macd_crossover_level_strength,
  c7.c7_macd_histogram_reversal_strength,
  c7.c7_macd_momentum_consensus_validation,
  c7.c7_rsi_price_divergence_exhaustion,
  c7.c7_macd_momentum_exhaustion,
  c7.c7_multi_timeframe_exhaustion_consensus,
  
  -- Component 8 features (48 features)
  c8.c8_component_agreement_score,
  c8.c8_integration_confidence,
  c8.c8_transition_probability_hint,
  c8.c8_weight_c1,
  c8.c8_weight_c2,
  c8.c8_weight_c3,
  c8.c8_weight_c4,
  c8.c8_weight_c5,
  c8.c8_weight_c6,
  c8.c8_weight_c7,
  c8.c8_score_c1,
  c8.c8_score_c2,
  c8.c8_score_c3,
  c8.c8_score_c4,
  c8.c8_score_c5,
  c8.c8_score_c6,
  c8.c8_score_c7,
  c8.c8_regime_classification,
  c8.c8_regime_probability_trending,
  c8.c8_regime_probability_volatile,
  c8.c8_regime_probability_ranging,
  c8.c8_regime_probability_breakout,
  c8.c8_component_correlation,
  c8.c8_component_divergence,
  c8.c8_unanimous_signal_strength,
  c8.c8_conflicting_signals_count,
  c8.c8_data_quality_score,
  c8.c8_feature_completeness,
  c8.c8_computation_latency_ms,
  c8.c8_integration_health_score,
  c8.c8_ensemble_direction_prediction,
  c8.c8_ensemble_magnitude_prediction,
  c8.c8_ensemble_confidence,
  c8.c8_prediction_disagreement_score,
  c8.c8_learning_rate,
  c8.c8_adaptation_speed,
  c8.c8_model_drift_score,
  c8.c8_retraining_urgency,
  c8.c8_total_processing_time_ms,
  c8.c8_memory_usage_mb,
  c8.c8_cpu_utilization_pct,
  c8.c8_gpu_utilization_pct,
  c8.c8_feature_importance_variance,
  c8.c8_component_contribution_entropy,
  c8.c8_signal_noise_ratio,
  c8.c8_information_gain,
  c8.c8_alert_level,
  c8.c8_alert_type,
  c8.c8_action_recommendation,
  
  -- Label columns for supervised learning (to be added when available)
  -- NULL as regime_label,  -- Will be populated from labeled data
  -- NULL as transition_label,  -- Will be populated from labeled data
  -- NULL as gap_direction_label,  -- Will be populated from labeled data
  
  -- Metadata
  CURRENT_TIMESTAMP() as query_timestamp

FROM `arched-bot-269016.market_regime_{env}.c1_features` c1
INNER JOIN `arched-bot-269016.market_regime_{env}.c2_features` c2
  ON c1.symbol = c2.symbol 
  AND c1.ts_minute = c2.ts_minute 
  AND c1.dte = c2.dte
INNER JOIN `arched-bot-269016.market_regime_{env}.c3_features` c3
  ON c1.symbol = c3.symbol 
  AND c1.ts_minute = c3.ts_minute 
  AND c1.dte = c3.dte
INNER JOIN `arched-bot-269016.market_regime_{env}.c4_features` c4
  ON c1.symbol = c4.symbol 
  AND c1.ts_minute = c4.ts_minute 
  AND c1.dte = c4.dte
INNER JOIN `arched-bot-269016.market_regime_{env}.c5_features` c5
  ON c1.symbol = c5.symbol 
  AND c1.ts_minute = c5.ts_minute 
  AND c1.dte = c5.dte
INNER JOIN `arched-bot-269016.market_regime_{env}.c6_features` c6
  ON c1.symbol = c6.symbol 
  AND c1.ts_minute = c6.ts_minute 
  AND c1.dte = c6.dte
INNER JOIN `arched-bot-269016.market_regime_{env}.c7_features` c7
  ON c1.symbol = c7.symbol 
  AND c1.ts_minute = c7.ts_minute 
  AND c1.dte = c7.dte
INNER JOIN `arched-bot-269016.market_regime_{env}.c8_features` c8
  ON c1.symbol = c8.symbol 
  AND c1.ts_minute = c8.ts_minute 
  AND c1.dte = c8.dte;

-- Option 2: Create as Materialized Table (for performance)
-- Uncomment below to create as table instead of view

/*
CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.training_dataset` 
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Training dataset with all 824+ features from 8 components",
  labels=[("dataset", "training"), ("feature_count", "824+"), ("version", "1.0")]
)
AS 
SELECT ... (same SELECT statement as above)
*/

-- Option 3: Create audit table for data quality monitoring
CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.mr_load_audit` (
  load_id STRING NOT NULL,
  load_timestamp TIMESTAMP NOT NULL,
  table_name STRING NOT NULL,
  row_count INT64,
  null_check_passed BOOL,
  schema_validation_passed BOOL,
  error_message STRING,
  load_duration_seconds FLOAT64,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(load_timestamp)
OPTIONS(
  description="Audit table for tracking data loads and quality checks",
  labels=[("type", "audit"), ("version", "1.0")]
);