-- Component 6: Correlation & Predictive Features + Momentum Enhancement (Phase 2)
-- Total features: 220 (Phase 1: 200 + Phase 2: 20 momentum-enhanced correlation features)
-- Purpose: Correlation matrix analysis and predictive indicators with momentum synergy (GPU-intensive)

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c6_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c6_correlation_agreement_score FLOAT64,
  c6_breakdown_alert INT64,
  c6_system_stability_score FLOAT64,
  
  -- Offline-only features
  -- 30x30 Correlation Matrix (top 30 correlations)
  c6_corr_matrix_1 FLOAT64,
  c6_corr_matrix_2 FLOAT64,
  c6_corr_matrix_3 FLOAT64,
  c6_corr_matrix_4 FLOAT64,
  c6_corr_matrix_5 FLOAT64,
  c6_corr_matrix_6 FLOAT64,
  c6_corr_matrix_7 FLOAT64,
  c6_corr_matrix_8 FLOAT64,
  c6_corr_matrix_9 FLOAT64,
  c6_corr_matrix_10 FLOAT64,
  c6_corr_matrix_11 FLOAT64,
  c6_corr_matrix_12 FLOAT64,
  c6_corr_matrix_13 FLOAT64,
  c6_corr_matrix_14 FLOAT64,
  c6_corr_matrix_15 FLOAT64,
  c6_corr_matrix_16 FLOAT64,
  c6_corr_matrix_17 FLOAT64,
  c6_corr_matrix_18 FLOAT64,
  c6_corr_matrix_19 FLOAT64,
  c6_corr_matrix_20 FLOAT64,
  c6_corr_matrix_21 FLOAT64,
  c6_corr_matrix_22 FLOAT64,
  c6_corr_matrix_23 FLOAT64,
  c6_corr_matrix_24 FLOAT64,
  c6_corr_matrix_25 FLOAT64,
  c6_corr_matrix_26 FLOAT64,
  c6_corr_matrix_27 FLOAT64,
  c6_corr_matrix_28 FLOAT64,
  c6_corr_matrix_29 FLOAT64,
  c6_corr_matrix_30 FLOAT64,
  
  -- Rolling correlations (multiple windows)
  c6_rolling_corr_5min FLOAT64,
  c6_rolling_corr_15min FLOAT64,
  c6_rolling_corr_30min FLOAT64,
  c6_rolling_corr_60min FLOAT64,
  c6_rolling_corr_daily FLOAT64,
  
  -- Cross-asset correlations
  c6_corr_nifty_banknifty FLOAT64,
  c6_corr_nifty_vix FLOAT64,
  c6_corr_nifty_usdinr FLOAT64,
  c6_corr_nifty_gold FLOAT64,
  c6_corr_nifty_crude FLOAT64,
  c6_corr_banknifty_vix FLOAT64,
  c6_corr_sector_index FLOAT64,
  
  -- Component correlations
  c6_corr_c1_c2 FLOAT64,
  c6_corr_c1_c3 FLOAT64,
  c6_corr_c1_c4 FLOAT64,
  c6_corr_c1_c5 FLOAT64,
  c6_corr_c2_c3 FLOAT64,
  c6_corr_c2_c4 FLOAT64,
  c6_corr_c2_c5 FLOAT64,
  c6_corr_c3_c4 FLOAT64,
  c6_corr_c3_c5 FLOAT64,
  c6_corr_c4_c5 FLOAT64,
  
  -- Correlation breakdown detection
  c6_correlation_breakdown_score FLOAT64,
  c6_breakdown_duration INT64,
  c6_breakdown_magnitude FLOAT64,
  c6_recovery_probability FLOAT64,
  
  -- Correlation stability metrics
  c6_correlation_stability FLOAT64,
  c6_correlation_mean_reversion FLOAT64,
  c6_correlation_regime INT64,
  c6_correlation_transition_prob FLOAT64,
  
  -- Eigenvalue decomposition (PCA)
  c6_eigen_value_1 FLOAT64,
  c6_eigen_value_2 FLOAT64,
  c6_eigen_value_3 FLOAT64,
  c6_eigen_vector_1 STRING,  -- JSON array
  c6_eigen_vector_2 STRING,  -- JSON array
  c6_eigen_vector_3 STRING,  -- JSON array
  c6_explained_variance_ratio FLOAT64,
  
  -- Network analysis metrics
  c6_network_centrality FLOAT64,
  c6_network_clustering_coef FLOAT64,
  c6_network_modularity FLOAT64,
  c6_network_connectivity FLOAT64,
  
  -- Predictive indicators
  c6_price_prediction_1min FLOAT64,
  c6_price_prediction_5min FLOAT64,
  c6_price_prediction_15min FLOAT64,
  c6_volatility_prediction_5min FLOAT64,
  c6_volatility_prediction_15min FLOAT64,
  c6_regime_prediction_prob FLOAT64,
  
  -- Feature importance scores
  c6_feature_importance_1 FLOAT64,
  c6_feature_importance_2 FLOAT64,
  c6_feature_importance_3 FLOAT64,
  c6_feature_importance_4 FLOAT64,
  c6_feature_importance_5 FLOAT64,
  c6_feature_importance_6 FLOAT64,
  c6_feature_importance_7 FLOAT64,
  c6_feature_importance_8 FLOAT64,
  c6_feature_importance_9 FLOAT64,
  c6_feature_importance_10 FLOAT64,
  
  -- Dynamic factor model
  c6_factor_1_loading FLOAT64,
  c6_factor_2_loading FLOAT64,
  c6_factor_3_loading FLOAT64,
  c6_idiosyncratic_risk FLOAT64,
  c6_systematic_risk FLOAT64,
  
  -- Copula-based dependencies
  c6_tail_dependence_upper FLOAT64,
  c6_tail_dependence_lower FLOAT64,
  c6_copula_parameter FLOAT64,
  
  -- Information theory metrics
  c6_mutual_information FLOAT64,
  c6_transfer_entropy FLOAT64,
  c6_conditional_entropy FLOAT64,
  
  -- Granger causality tests
  c6_granger_cause_to_nifty FLOAT64,
  c6_granger_cause_from_nifty FLOAT64,
  c6_granger_cause_to_vix FLOAT64,
  c6_granger_cause_from_vix FLOAT64,
  
  -- Cointegration analysis
  c6_cointegration_score FLOAT64,
  c6_error_correction_term FLOAT64,
  c6_long_run_equilibrium FLOAT64,
  
  -- Wavelet correlations
  c6_wavelet_corr_high_freq FLOAT64,
  c6_wavelet_corr_mid_freq FLOAT64,
  c6_wavelet_corr_low_freq FLOAT64,
  
  -- Lead-lag relationships
  c6_lead_lag_indicator FLOAT64,
  c6_optimal_lag INT64,
  c6_lag_correlation_peak FLOAT64,
  
  -- Regime-specific correlations
  c6_bull_regime_correlation FLOAT64,
  c6_bear_regime_correlation FLOAT64,
  c6_neutral_regime_correlation FLOAT64,
  
  -- Correlation anomaly detection
  c6_correlation_zscore FLOAT64,
  c6_correlation_outlier_score FLOAT64,
  c6_anomaly_detection_flag INT64,
  
  -- System risk metrics
  c6_systemic_risk_score FLOAT64,
  c6_contagion_risk FLOAT64,
  c6_diversification_ratio FLOAT64,
  
  -- Advanced predictive features
  c6_ensemble_prediction FLOAT64,
  c6_prediction_confidence FLOAT64,
  c6_prediction_disagreement FLOAT64,
  
  -- Time-varying correlations
  c6_dcc_correlation FLOAT64,  -- Dynamic Conditional Correlation
  c6_realized_correlation FLOAT64,
  c6_correlation_forecast FLOAT64,
  
  -- Non-linear dependencies
  c6_distance_correlation FLOAT64,
  c6_maximal_information_coef FLOAT64,
  c6_hoeffding_d_statistic FLOAT64,
  
  -- Additional correlation features (to reach 200+)
  c6_correlation_skew FLOAT64,
  c6_correlation_kurtosis FLOAT64,
  c6_correlation_entropy FLOAT64,
  c6_correlation_complexity FLOAT64,
  c6_partial_correlation_1 FLOAT64,
  c6_partial_correlation_2 FLOAT64,
  c6_partial_correlation_3 FLOAT64,
  c6_partial_correlation_4 FLOAT64,
  c6_partial_correlation_5 FLOAT64,
  c6_semi_correlation_positive FLOAT64,
  c6_semi_correlation_negative FLOAT64,
  c6_rank_correlation FLOAT64,
  c6_kendall_tau FLOAT64,
  c6_correlation_half_life FLOAT64,
  c6_correlation_momentum FLOAT64,
  c6_correlation_mean FLOAT64,
  c6_correlation_std FLOAT64,
  c6_correlation_min FLOAT64,
  c6_correlation_max FLOAT64,
  c6_correlation_range FLOAT64,
  
  -- ML-enhanced correlation features
  c6_ml_correlation_forecast FLOAT64,
  c6_ml_breakdown_probability FLOAT64,
  c6_ml_regime_classification INT64,
  c6_ml_feature_selection STRING,  -- JSON array of selected features
  
  -- GPU computation metrics
  c6_gpu_computation_time_ms FLOAT64,
  c6_matrix_condition_number FLOAT64,
  c6_numerical_stability_score FLOAT64,
  
  -- Phase 2: Momentum-Enhanced Correlation Features (20 features)
  -- RSI-Correlation Features (8 features)
  c6_rsi_cross_correlation_3min FLOAT64,
  c6_rsi_cross_correlation_5min FLOAT64,
  c6_rsi_price_agreement_3min FLOAT64,
  c6_rsi_price_agreement_5min FLOAT64,
  c6_rsi_regime_coherence_3min FLOAT64,
  c6_rsi_regime_coherence_5min FLOAT64,
  c6_rsi_divergence_3min_5min FLOAT64,
  c6_rsi_divergence_5min_10min FLOAT64,

  -- MACD-Correlation Features (8 features)
  c6_macd_signal_correlation_3min FLOAT64,
  c6_macd_signal_correlation_5min FLOAT64,
  c6_macd_histogram_convergence_3min FLOAT64,
  c6_macd_histogram_convergence_5min FLOAT64,
  c6_macd_trend_agreement_3min FLOAT64,
  c6_macd_trend_agreement_5min FLOAT64,
  c6_macd_momentum_strength_3min FLOAT64,
  c6_macd_momentum_strength_5min FLOAT64,

  -- Momentum Consensus Features (4 features)
  c6_multi_timeframe_rsi_consensus FLOAT64,
  c6_multi_timeframe_macd_consensus FLOAT64,
  c6_cross_component_momentum_agreement FLOAT64,
  c6_overall_momentum_system_coherence FLOAT64,

  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)

  -- Phase 2: Momentum-Enhanced Correlation Features (20 total)
  -- RSI Correlation Features (8 features)
  c6_rsi_cross_correlation_3min FLOAT64,
  c6_rsi_cross_correlation_5min FLOAT64,
  c6_rsi_price_agreement_3min FLOAT64,
  c6_rsi_price_agreement_5min FLOAT64,
  c6_rsi_regime_coherence_3min FLOAT64,
  c6_rsi_regime_coherence_5min FLOAT64,
  c6_rsi_divergence_3min_5min FLOAT64,
  c6_rsi_divergence_5min_10min FLOAT64,
  
  -- MACD Correlation Features (8 features)
  c6_macd_signal_correlation_3min FLOAT64,
  c6_macd_signal_correlation_5min FLOAT64,
  c6_macd_histogram_convergence_3min FLOAT64,
  c6_macd_histogram_convergence_5min FLOAT64,
  c6_macd_trend_agreement_3min FLOAT64,
  c6_macd_trend_agreement_5min FLOAT64,
  c6_macd_momentum_strength_3min FLOAT64,
  c6_macd_momentum_strength_5min FLOAT64,
  
  -- Momentum Consensus Features (4 features)
  c6_multi_timeframe_rsi_consensus FLOAT64,
  c6_multi_timeframe_macd_consensus FLOAT64,
  c6_cross_component_momentum_agreement FLOAT64,
  c6_overall_momentum_system_coherence FLOAT64,
  
  -- Additional correlation features to reach 220 (20 more)
  c6_correlation_matrix_stability FLOAT64,
  c6_cross_symbol_correlation_nifty_bank FLOAT64,
  c6_cross_symbol_correlation_volatility FLOAT64,
  c6_temporal_correlation_consistency FLOAT64,
  c6_regime_correlation_shift FLOAT64,
  c6_correlation_breakdown_risk FLOAT64,
  c6_correlation_recovery_speed FLOAT64,
  c6_correlation_quality_score FLOAT64,
  c6_correlation_confidence_level FLOAT64,
  c6_correlation_prediction_accuracy FLOAT64,
  c6_correlation_adaptive_weight FLOAT64,
  c6_correlation_performance_score FLOAT64,
  c6_correlation_validation_count FLOAT64,
  c6_correlation_historical_accuracy FLOAT64,
  c6_correlation_trend_consistency FLOAT64,
  c6_correlation_volatility_adjustment FLOAT64,
  c6_correlation_regime_sensitivity FLOAT64,
  c6_correlation_cross_validation FLOAT64,
  c6_correlation_ensemble_agreement FLOAT64,
  c6_correlation_system_health FLOAT64,
)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 6: Correlation & Predictive + Momentum Enhancement - 220 features (200 + 20 momentum correlation)",
  labels=[("component", "c6"), ("feature_count", "220"), ("version", "2.0"), ("phase", "momentum_enhanced"), ("gpu_required", "true")]
);