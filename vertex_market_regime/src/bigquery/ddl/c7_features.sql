-- Component 7: Support/Resistance Features + Momentum Enhancement (Phase 2)
-- Total features: 130 (120 + 10 momentum-based level features)
-- Purpose: Support and resistance level detection with momentum validation

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c7_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c7_level_strength_score FLOAT64,
  c7_breakout_probability FLOAT64,
  
  -- Offline-only features
  -- Primary support/resistance levels
  c7_support_1 FLOAT64,
  c7_support_2 FLOAT64,
  c7_support_3 FLOAT64,
  c7_resistance_1 FLOAT64,
  c7_resistance_2 FLOAT64,
  c7_resistance_3 FLOAT64,
  
  -- Level strength metrics
  c7_support_1_strength FLOAT64,
  c7_support_2_strength FLOAT64,
  c7_support_3_strength FLOAT64,
  c7_resistance_1_strength FLOAT64,
  c7_resistance_2_strength FLOAT64,
  c7_resistance_3_strength FLOAT64,
  
  -- Dynamic levels
  c7_dynamic_support FLOAT64,
  c7_dynamic_resistance FLOAT64,
  c7_pivot_point FLOAT64,
  c7_fibonacci_38_2 FLOAT64,
  c7_fibonacci_50_0 FLOAT64,
  c7_fibonacci_61_8 FLOAT64,
  
  -- Volume at levels
  c7_volume_at_support FLOAT64,
  c7_volume_at_resistance FLOAT64,
  c7_volume_profile_poc FLOAT64,  -- Point of Control
  c7_volume_weighted_support FLOAT64,
  c7_volume_weighted_resistance FLOAT64,
  
  -- Level testing metrics
  c7_support_tests_count INT64,
  c7_resistance_tests_count INT64,
  c7_support_holds_ratio FLOAT64,
  c7_resistance_holds_ratio FLOAT64,
  
  -- Confluence detection
  c7_confluence_support_score FLOAT64,
  c7_confluence_resistance_score FLOAT64,
  c7_multi_timeframe_agreement FLOAT64,
  c7_technical_confluence_count INT64,
  
  -- Breakout/breakdown metrics
  c7_breakout_momentum FLOAT64,
  c7_breakdown_momentum FLOAT64,
  c7_false_breakout_probability FLOAT64,
  c7_breakout_volume_confirmation FLOAT64,
  
  -- Price action at levels
  c7_price_to_support_distance FLOAT64,
  c7_price_to_resistance_distance FLOAT64,
  c7_price_level_magnetism FLOAT64,
  c7_level_rejection_score FLOAT64,
  
  -- Historical level analysis
  c7_historical_support_reliability FLOAT64,
  c7_historical_resistance_reliability FLOAT64,
  c7_level_age_days INT64,
  c7_level_significance_score FLOAT64,
  
  -- Market profile levels
  c7_value_area_high FLOAT64,
  c7_value_area_low FLOAT64,
  c7_value_area_poc FLOAT64,
  c7_initial_balance_high FLOAT64,
  c7_initial_balance_low FLOAT64,
  
  -- Options-based S/R
  c7_max_pain_level FLOAT64,
  c7_call_wall FLOAT64,
  c7_put_wall FLOAT64,
  c7_gamma_flip_level FLOAT64,
  
  -- Trend channel boundaries
  c7_upper_channel FLOAT64,
  c7_lower_channel FLOAT64,
  c7_channel_midpoint FLOAT64,
  c7_channel_width FLOAT64,
  
  -- Level clustering
  c7_support_cluster_center FLOAT64,
  c7_resistance_cluster_center FLOAT64,
  c7_cluster_density_score FLOAT64,
  
  -- Time-based levels
  c7_daily_high FLOAT64,
  c7_daily_low FLOAT64,
  c7_weekly_high FLOAT64,
  c7_weekly_low FLOAT64,
  c7_monthly_high FLOAT64,
  c7_monthly_low FLOAT64,
  
  -- ML-enhanced features
  c7_ml_support_prediction FLOAT64,
  c7_ml_resistance_prediction FLOAT64,
  c7_ml_breakout_classification INT64,
  c7_ml_level_reliability_score FLOAT64,
  
  -- Advanced Pattern Features (48 additional)
  -- Advanced Reversal Patterns
  c7_hammer_pattern_strength FLOAT64,
  c7_shooting_star_strength FLOAT64,
  c7_doji_reversal_score FLOAT64,
  c7_engulfing_pattern_score FLOAT64,
  c7_harami_pattern_strength FLOAT64,
  c7_piercing_line_score FLOAT64,
  c7_dark_cloud_cover_score FLOAT64,
  c7_morning_star_strength FLOAT64,
  c7_evening_star_strength FLOAT64,
  c7_three_white_soldiers_score FLOAT64,
  c7_three_black_crows_score FLOAT64,
  c7_inside_bar_pattern_score FLOAT64,
  
  -- Advanced Breakout Patterns
  c7_flag_pattern_score FLOAT64,
  c7_pennant_pattern_score FLOAT64,
  c7_triangle_pattern_score FLOAT64,
  c7_rectangle_pattern_score FLOAT64,
  c7_wedge_pattern_score FLOAT64,
  c7_cup_handle_pattern_score FLOAT64,
  c7_head_shoulders_score FLOAT64,
  c7_inverse_head_shoulders_score FLOAT64,
  c7_double_top_score FLOAT64,
  c7_double_bottom_score FLOAT64,
  c7_triple_top_score FLOAT64,
  c7_triple_bottom_score FLOAT64,
  
  -- Advanced Volume Analysis
  c7_accumulation_distribution FLOAT64,
  c7_on_balance_volume FLOAT64,
  c7_volume_rate_of_change FLOAT64,
  c7_money_flow_index FLOAT64,
  c7_chaikin_money_flow FLOAT64,
  c7_ease_of_movement FLOAT64,
  c7_force_index FLOAT64,
  c7_negative_volume_index FLOAT64,
  c7_positive_volume_index FLOAT64,
  c7_price_volume_trend FLOAT64,
  
  -- Advanced Momentum Indicators
  c7_williams_percent_r FLOAT64,
  c7_commodity_channel_index FLOAT64,
  c7_detrended_price_oscillator FLOAT64,
  c7_know_sure_thing FLOAT64,
  c7_percentage_price_oscillator FLOAT64,
  c7_rate_of_change FLOAT64,
  c7_relative_strength_ratio FLOAT64,
  c7_stochastic_oscillator FLOAT64,
  c7_true_strength_index FLOAT64,
  c7_ultimate_oscillator FLOAT64,
  
  -- Advanced Volatility Measures
  c7_average_true_range_ratio FLOAT64,
  c7_bollinger_band_width FLOAT64,
  c7_keltner_channel_width FLOAT64,
  c7_donchian_channel_width FLOAT64,
  c7_volatility_system_score FLOAT64,
  c7_choppiness_index FLOAT64,
  
  -- Phase 2: Momentum-Based Level Detection Features (10 features)
  -- RSI Level Confluence Features (4 features)
  c7_rsi_overbought_resistance_strength FLOAT64,
  c7_rsi_oversold_support_strength FLOAT64,
  c7_rsi_neutral_zone_level_density FLOAT64,
  c7_rsi_level_convergence_strength FLOAT64,

  -- MACD Level Validation Features (3 features)
  c7_macd_crossover_level_strength FLOAT64,
  c7_macd_histogram_reversal_strength FLOAT64,
  c7_macd_momentum_consensus_validation FLOAT64,

  -- Momentum Exhaustion Level Features (3 features)
  c7_rsi_price_divergence_exhaustion FLOAT64,
  c7_macd_momentum_exhaustion FLOAT64,
  c7_multi_timeframe_exhaustion_consensus FLOAT64,

  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)

  -- Phase 2: Momentum-Based Level Detection Features (10 total)
  -- RSI Level Confluence Features (4 features)
  c7_rsi_overbought_resistance_strength FLOAT64,
  c7_rsi_oversold_support_strength FLOAT64,
  c7_rsi_neutral_zone_level_density FLOAT64,
  c7_rsi_level_convergence_strength FLOAT64,
  
  -- MACD Level Validation Features (3 features)
  c7_macd_crossover_level_strength FLOAT64,
  c7_macd_histogram_reversal_strength FLOAT64,
  c7_macd_momentum_consensus_validation FLOAT64,
  
  -- Momentum Exhaustion Features (3 features)
  c7_rsi_price_divergence_exhaustion FLOAT64,
  c7_macd_momentum_exhaustion FLOAT64,
  c7_multi_timeframe_exhaustion_consensus FLOAT64,
)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 7: Support/Resistance + Momentum Enhancement - 130 features (120 + 10 momentum-based levels)",
  labels=[("component", "c7"), ("feature_count", "130"), ("version", "2.0"), ("phase", "momentum_enhanced")]
);