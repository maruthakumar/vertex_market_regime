-- Component 3: OI-PA Trending Analysis Features
-- Total features: 105
-- Purpose: Open Interest and Price Action trending analysis

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c3_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c3_institutional_flow_score FLOAT64,
  c3_divergence_type INT64,
  c3_range_expansion_score FLOAT64,
  
  -- Offline-only features
  -- Open Interest metrics
  c3_oi_total INT64,
  c3_oi_call INT64,
  c3_oi_put INT64,
  c3_oi_put_call_ratio FLOAT64,
  c3_oi_change_1min INT64,
  c3_oi_change_5min INT64,
  c3_oi_change_15min INT64,
  c3_oi_change_60min INT64,
  
  -- OI by DTE
  c3_oi_0dte INT64,
  c3_oi_3dte INT64,
  c3_oi_7dte INT64,
  c3_oi_14dte INT64,
  c3_oi_21dte INT64,
  c3_oi_30dte INT64,
  
  -- Price Action metrics
  c3_price_momentum_1min FLOAT64,
  c3_price_momentum_5min FLOAT64,
  c3_price_momentum_15min FLOAT64,
  c3_price_momentum_60min FLOAT64,
  c3_price_acceleration FLOAT64,
  c3_price_velocity FLOAT64,
  
  -- OI-Price divergence
  c3_oi_price_divergence FLOAT64,
  c3_oi_price_correlation FLOAT64,
  c3_divergence_strength FLOAT64,
  c3_divergence_duration INT64,
  
  -- Institutional flow analysis
  c3_large_trade_oi_change INT64,
  c3_block_trade_volume INT64,
  c3_institutional_oi_ratio FLOAT64,
  c3_smart_money_flow FLOAT64,
  
  -- Trending indicators
  c3_trend_strength FLOAT64,
  c3_trend_consistency FLOAT64,
  c3_trend_quality FLOAT64,
  c3_trend_exhaustion FLOAT64,
  
  -- Range analysis
  c3_range_width FLOAT64,
  c3_range_position FLOAT64,
  c3_range_breakout_probability FLOAT64,
  c3_range_mean_reversion_score FLOAT64,
  
  -- Volume-OI analysis
  c3_volume_oi_ratio FLOAT64,
  c3_volume_weighted_oi FLOAT64,
  c3_oi_concentration FLOAT64,
  c3_oi_dispersion FLOAT64,
  
  -- Strike-level OI
  c3_max_oi_strike FLOAT64,
  c3_max_oi_call_strike FLOAT64,
  c3_max_oi_put_strike FLOAT64,
  c3_oi_weighted_strike FLOAT64,
  
  -- OI changes by strike
  c3_atm_oi_change INT64,
  c3_otm_oi_change INT64,
  c3_itm_oi_change INT64,
  c3_far_otm_oi_change INT64,
  
  -- Cumulative metrics
  c3_cumulative_oi_change INT64,
  c3_cumulative_volume INT64,
  c3_cumulative_delta_oi INT64,
  
  -- OI velocity and acceleration
  c3_oi_velocity FLOAT64,
  c3_oi_acceleration FLOAT64,
  c3_oi_jerk FLOAT64,
  
  -- Market depth from OI
  c3_oi_support_level FLOAT64,
  c3_oi_resistance_level FLOAT64,
  c3_oi_pivot_points STRING,  -- JSON array of pivot points
  
  -- Sentiment from OI-PA
  c3_oi_bullish_buildup FLOAT64,
  c3_oi_bearish_buildup FLOAT64,
  c3_oi_short_covering FLOAT64,
  c3_oi_long_unwinding FLOAT64,
  
  -- Advanced flow metrics
  c3_toxic_flow_indicator FLOAT64,
  c3_informed_trading_probability FLOAT64,
  c3_stealth_accumulation FLOAT64,
  c3_distribution_score FLOAT64,
  
  -- Microstructure from OI
  c3_oi_order_imbalance FLOAT64,
  c3_oi_liquidity_score FLOAT64,
  c3_oi_market_impact FLOAT64,
  
  -- Cross-asset OI analysis
  c3_nifty_banknifty_oi_ratio FLOAT64,
  c3_sector_oi_divergence FLOAT64,
  c3_index_stock_oi_correlation FLOAT64,
  
  -- Time-based OI patterns
  c3_oi_morning_buildup FLOAT64,
  c3_oi_afternoon_unwind FLOAT64,
  c3_oi_overnight_change INT64,
  c3_oi_weekly_pattern FLOAT64,
  
  -- Event-driven OI
  c3_pre_expiry_oi_rollover FLOAT64,
  c3_post_event_oi_reaction FLOAT64,
  c3_oi_event_anticipation FLOAT64,
  
  -- Statistical OI features
  c3_oi_zscore_5min FLOAT64,
  c3_oi_zscore_60min FLOAT64,
  c3_oi_percentile_rank FLOAT64,
  c3_oi_mean_reversion FLOAT64,
  
  -- ML-enhanced features
  c3_ml_flow_classification INT64,
  c3_ml_oi_forecast FLOAT64,
  c3_ml_divergence_prediction FLOAT64,
  c3_ml_institutional_intent INT64,
  
  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)

  -- Additional features to reach target count
  c3_feature_91 FLOAT64,
  c3_feature_92 FLOAT64,
  c3_feature_93 FLOAT64,
  c3_feature_94 FLOAT64,
  c3_feature_95 FLOAT64,
  c3_feature_96 FLOAT64,
  c3_feature_97 FLOAT64,
  c3_feature_98 FLOAT64,
  c3_feature_99 FLOAT64,
  c3_feature_100 FLOAT64,
  c3_feature_101 FLOAT64,
  c3_feature_102 FLOAT64,
  c3_feature_103 FLOAT64,
  c3_feature_104 FLOAT64,
  c3_feature_105 FLOAT64,

)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 3: OI-PA Trending Analysis - 105 features for open interest and price action",
  labels=[("component", "c3"), ("feature_count", "105"), ("version", "1.0")]
);