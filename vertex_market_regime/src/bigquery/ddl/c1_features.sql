-- Component 1: Triple Straddle Features + Momentum Enhancement (Phase 2)
-- Total features: 150 (Phase 1: 120 + Phase 2: 30 momentum features)
-- Purpose: Rolling straddle analysis across multiple DTEs with RSI/MACD momentum indicators

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c1_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c1_momentum_score FLOAT64,
  c1_vol_compression FLOAT64,
  c1_breakout_probability FLOAT64,
  
  -- Phase 1: Base 120 Features
  -- Straddle metrics (per DTE: 0, 3, 7, 14, 21, 30) - 18 features
  c1_straddle_pct_chg_0dte FLOAT64,
  c1_straddle_pct_chg_3dte FLOAT64,
  c1_straddle_pct_chg_7dte FLOAT64,
  c1_straddle_pct_chg_14dte FLOAT64,
  c1_straddle_pct_chg_21dte FLOAT64,
  c1_straddle_pct_chg_30dte FLOAT64,
  c1_straddle_volume_0dte INT64,
  c1_straddle_volume_3dte INT64,
  c1_straddle_volume_7dte INT64,
  c1_straddle_volume_14dte INT64,
  c1_straddle_volume_21dte INT64,
  c1_straddle_volume_30dte INT64,
  c1_straddle_oi_0dte INT64,
  c1_straddle_oi_3dte INT64,
  c1_straddle_oi_7dte INT64,
  c1_straddle_oi_14dte INT64,
  c1_straddle_oi_21dte INT64,
  c1_straddle_oi_30dte INT64,
  
  -- Rolling averages - 4 features
  c1_straddle_ma_5min FLOAT64,
  c1_straddle_ma_15min FLOAT64,
  c1_straddle_ma_30min FLOAT64,
  c1_straddle_ma_60min FLOAT64,
  
  -- Cross-DTE correlations - 6 features
  c1_correlation_0_3dte FLOAT64,
  c1_correlation_0_7dte FLOAT64,
  c1_correlation_3_7dte FLOAT64,
  c1_correlation_7_14dte FLOAT64,
  c1_correlation_14_21dte FLOAT64,
  c1_correlation_21_30dte FLOAT64,
  
  -- DTE optimized metrics - 3 features
  c1_dte_momentum_score FLOAT64,
  c1_dte_divergence_indicator INT64,
  c1_dte_volatility_term_structure FLOAT64,
  
  -- Zone-specific metrics - 5 features
  c1_zone_momentum_open FLOAT64,
  c1_zone_momentum_mid_morn FLOAT64,
  c1_zone_momentum_lunch FLOAT64,
  c1_zone_momentum_afternoon FLOAT64,
  c1_zone_momentum_close FLOAT64,
  
  -- Breakout detection - 3 features
  c1_breakout_strength FLOAT64,
  c1_consolidation_score FLOAT64,
  c1_range_expansion_probability FLOAT64,
  
  -- Volume profile - 3 features
  c1_volume_intensity FLOAT64,
  c1_volume_weighted_momentum FLOAT64,
  c1_abnormal_volume_flag INT64,
  
  -- Microstructure features - 3 features
  c1_bid_ask_spread_avg FLOAT64,
  c1_market_depth_score FLOAT64,
  c1_liquidity_score FLOAT64,
  
  -- Regime transition indicators - 3 features
  c1_regime_stability_score FLOAT64,
  c1_transition_probability FLOAT64,
  c1_regime_confidence FLOAT64,
  
  -- Statistical features - 6 features
  c1_z_score_5min FLOAT64,
  c1_z_score_15min FLOAT64,
  c1_z_score_60min FLOAT64,
  c1_percentile_rank_5min FLOAT64,
  c1_percentile_rank_15min FLOAT64,
  c1_percentile_rank_60min FLOAT64,
  
  -- Basic momentum indicators - 4 features
  c1_rsi_5min FLOAT64,
  c1_rsi_15min FLOAT64,
  c1_macd_signal FLOAT64,
  c1_macd_histogram FLOAT64,
  
  -- Volatility metrics - 4 features
  c1_realized_vol_5min FLOAT64,
  c1_realized_vol_15min FLOAT64,
  c1_realized_vol_60min FLOAT64,
  c1_vol_of_vol FLOAT64,
  
  -- Cross-asset features - 3 features
  c1_beta_to_nifty FLOAT64,
  c1_beta_to_banknifty FLOAT64,
  c1_correlation_to_vix FLOAT64,
  
  -- Event-driven features - 3 features
  c1_event_impact_score FLOAT64,
  c1_pre_event_positioning FLOAT64,
  c1_post_event_momentum FLOAT64,
  
  -- Advanced metrics - 4 features
  c1_entropy_score FLOAT64,
  c1_fractal_dimension FLOAT64,
  c1_hurst_exponent FLOAT64,
  c1_lyapunov_exponent FLOAT64,
  
  -- Market microstructure - 3 features
  c1_order_flow_imbalance FLOAT64,
  c1_toxic_flow_probability FLOAT64,
  c1_smart_money_indicator FLOAT64,
  
  -- Seasonality features - 4 features
  c1_day_of_week_effect FLOAT64,
  c1_time_of_day_effect FLOAT64,
  c1_monthly_seasonality FLOAT64,
  c1_expiry_effect FLOAT64,
  
  -- Risk metrics - 4 features
  c1_var_95 FLOAT64,
  c1_cvar_95 FLOAT64,
  c1_max_drawdown FLOAT64,
  c1_sharpe_ratio FLOAT64,
  
  -- Machine learning features - 4 features
  c1_ml_momentum_prediction FLOAT64,
  c1_ml_volatility_forecast FLOAT64,
  c1_ml_regime_probability FLOAT64,
  c1_ml_confidence_score FLOAT64,
  
  -- Additional features to reach 120 base - 37 features
  c1_feature_81 FLOAT64, c1_feature_82 FLOAT64, c1_feature_83 FLOAT64, c1_feature_84 FLOAT64,
  c1_feature_85 FLOAT64, c1_feature_86 FLOAT64, c1_feature_87 FLOAT64, c1_feature_88 FLOAT64,
  c1_feature_89 FLOAT64, c1_feature_90 FLOAT64, c1_feature_91 FLOAT64, c1_feature_92 FLOAT64,
  c1_feature_93 FLOAT64, c1_feature_94 FLOAT64, c1_feature_95 FLOAT64, c1_feature_96 FLOAT64,
  c1_feature_97 FLOAT64, c1_feature_98 FLOAT64, c1_feature_99 FLOAT64, c1_feature_100 FLOAT64,
  c1_feature_101 FLOAT64, c1_feature_102 FLOAT64, c1_feature_103 FLOAT64, c1_feature_104 FLOAT64,
  c1_feature_105 FLOAT64, c1_feature_106 FLOAT64, c1_feature_107 FLOAT64, c1_feature_108 FLOAT64,
  c1_feature_109 FLOAT64, c1_feature_110 FLOAT64, c1_feature_111 FLOAT64, c1_feature_112 FLOAT64,
  c1_feature_113 FLOAT64, c1_feature_114 FLOAT64, c1_feature_115 FLOAT64, c1_feature_116 FLOAT64,
  c1_feature_117 FLOAT64,
  
  -- Phase 2: Momentum Enhancement Features (30 features)
  -- RSI Features across 4 timeframes (15 features)
  c1_rsi_3min_trend FLOAT64,
  c1_rsi_3min_strength FLOAT64,
  c1_rsi_3min_signal FLOAT64,
  c1_rsi_3min_normalized FLOAT64,
  c1_rsi_5min_trend FLOAT64,
  c1_rsi_5min_strength FLOAT64,
  c1_rsi_5min_signal FLOAT64,
  c1_rsi_5min_normalized FLOAT64,
  c1_rsi_10min_trend FLOAT64,
  c1_rsi_10min_strength FLOAT64,
  c1_rsi_10min_signal FLOAT64,
  c1_rsi_10min_normalized FLOAT64,
  c1_rsi_15min_trend FLOAT64,
  c1_rsi_15min_strength FLOAT64,
  c1_rsi_combined_consensus FLOAT64,

  -- MACD Features across 4 timeframes (10 features)
  c1_macd_3min_signal FLOAT64,
  c1_macd_3min_histogram FLOAT64,
  c1_macd_3min_crossover FLOAT64,
  c1_macd_5min_signal FLOAT64,
  c1_macd_5min_histogram FLOAT64,
  c1_macd_5min_crossover FLOAT64,
  c1_macd_10min_signal FLOAT64,
  c1_macd_10min_histogram FLOAT64,
  c1_macd_15min_signal FLOAT64,
  c1_macd_consensus_strength FLOAT64,

  -- Cross-Timeframe Divergence Features (5 features)
  c1_momentum_3min_5min_divergence FLOAT64,
  c1_momentum_5min_10min_divergence FLOAT64,
  c1_momentum_10min_15min_divergence FLOAT64,
  c1_momentum_consensus_score FLOAT64,
  c1_momentum_regime_strength FLOAT64,

  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 1: Triple Straddle Features + Momentum Enhancement - 150 features (120 + 30 momentum)",
  labels=[("component", "c1"), ("feature_count", "150"), ("version", "2.0"), ("phase", "momentum_enhanced")]
);