-- Component 4: IV Skew Analysis Features
-- Total features: 87
-- Purpose: Implied Volatility skew and percentile analysis

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c4_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c4_skew_bias_score FLOAT64,
  c4_term_structure_signal INT64,
  c4_iv_regime_level INT64,
  
  -- Offline-only features
  -- Core IV metrics
  c4_iv_atm FLOAT64,
  c4_iv_otm_call FLOAT64,
  c4_iv_otm_put FLOAT64,
  c4_iv_itm_call FLOAT64,
  c4_iv_itm_put FLOAT64,
  
  -- IV by DTE
  c4_iv_0dte FLOAT64,
  c4_iv_3dte FLOAT64,
  c4_iv_7dte FLOAT64,
  c4_iv_14dte FLOAT64,
  c4_iv_21dte FLOAT64,
  c4_iv_30dte FLOAT64,
  
  -- Skew metrics
  c4_skew_25delta FLOAT64,
  c4_skew_10delta FLOAT64,
  c4_skew_slope FLOAT64,
  c4_skew_curvature FLOAT64,
  c4_skew_asymmetry FLOAT64,
  
  -- IV percentile analysis
  c4_iv_percentile_1d FLOAT64,
  c4_iv_percentile_5d FLOAT64,
  c4_iv_percentile_20d FLOAT64,
  c4_iv_percentile_60d FLOAT64,
  c4_iv_percentile_252d FLOAT64,
  
  -- Skew percentile analysis
  c4_skew_percentile_1d FLOAT64,
  c4_skew_percentile_5d FLOAT64,
  c4_skew_percentile_20d FLOAT64,
  c4_skew_percentile_60d FLOAT64,
  
  -- Term structure
  c4_term_structure_slope FLOAT64,
  c4_term_structure_convexity FLOAT64,
  c4_calendar_spread FLOAT64,
  c4_forward_volatility FLOAT64,
  
  -- IV momentum
  c4_iv_momentum_5min FLOAT64,
  c4_iv_momentum_15min FLOAT64,
  c4_iv_momentum_60min FLOAT64,
  c4_iv_acceleration FLOAT64,
  
  -- Dual DTE framework
  c4_short_term_iv_trend FLOAT64,
  c4_long_term_iv_trend FLOAT64,
  c4_dual_dte_divergence FLOAT64,
  c4_dte_crossover_signal INT64,
  
  -- Regime classification
  c4_iv_regime_low INT64,
  c4_iv_regime_normal INT64,
  c4_iv_regime_elevated INT64,
  c4_iv_regime_extreme INT64,
  c4_regime_transition_probability FLOAT64,
  
  -- IV surface analysis
  c4_surface_smoothness FLOAT64,
  c4_surface_arbitrage_score FLOAT64,
  c4_surface_stability FLOAT64,
  c4_surface_richness FLOAT64,
  
  -- Risk reversal and butterfly
  c4_risk_reversal_25d FLOAT64,
  c4_risk_reversal_10d FLOAT64,
  c4_butterfly_25d FLOAT64,
  c4_butterfly_10d FLOAT64,
  
  -- IV-RV analysis
  c4_iv_rv_spread FLOAT64,
  c4_iv_rv_ratio FLOAT64,
  c4_variance_risk_premium FLOAT64,
  
  -- Zone-based IV analysis
  c4_iv_zone_open FLOAT64,
  c4_iv_zone_mid_morn FLOAT64,
  c4_iv_zone_lunch FLOAT64,
  c4_iv_zone_afternoon FLOAT64,
  c4_iv_zone_close FLOAT64,
  
  -- Smile dynamics
  c4_smile_width FLOAT64,
  c4_smile_center FLOAT64,
  c4_smile_stability FLOAT64,
  
  -- Event volatility
  c4_event_iv_premium FLOAT64,
  c4_earnings_iv_cycle FLOAT64,
  c4_weekend_theta_decay FLOAT64,
  
  -- Cross-asset IV
  c4_iv_beta_to_vix FLOAT64,
  c4_iv_correlation_index FLOAT64,
  c4_iv_dispersion FLOAT64,
  
  -- Advanced skew metrics
  c4_skew_momentum FLOAT64,
  c4_skew_mean_reversion FLOAT64,
  c4_skew_regime_score INT64,
  
  -- ML-enhanced features
  c4_ml_iv_forecast FLOAT64,
  c4_ml_skew_prediction FLOAT64,
  c4_ml_regime_probability FLOAT64,
  c4_ml_percentile_forecast FLOAT64,
  
  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)

  -- Additional features to reach target count
  c4_feature_79 FLOAT64,
  c4_feature_80 FLOAT64,
  c4_feature_81 FLOAT64,
  c4_feature_82 FLOAT64,
  c4_feature_83 FLOAT64,
  c4_feature_84 FLOAT64,
  c4_feature_85 FLOAT64,
  c4_feature_86 FLOAT64,
  c4_feature_87 FLOAT64,

)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 4: IV Skew Analysis - 87 features for implied volatility and skew analysis",
  labels=[("component", "c4"), ("feature_count", "87"), ("version", "1.0")]
);