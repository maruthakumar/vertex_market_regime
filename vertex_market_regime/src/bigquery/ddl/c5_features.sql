-- Component 5: ATR-EMA-CPR Features
-- Total features: 94
-- Purpose: Technical indicators - ATR, EMA, CPR analysis

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c5_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c5_momentum_score FLOAT64,
  c5_volatility_regime_score INT64,
  c5_confluence_score FLOAT64,
  
  -- Offline-only features
  -- ATR (Average True Range) metrics
  c5_atr_5min FLOAT64,
  c5_atr_15min FLOAT64,
  c5_atr_30min FLOAT64,
  c5_atr_60min FLOAT64,
  c5_atr_daily FLOAT64,
  
  -- ATR percentiles
  c5_atr_percentile_5d FLOAT64,
  c5_atr_percentile_20d FLOAT64,
  c5_atr_percentile_60d FLOAT64,
  
  -- EMA (Exponential Moving Average) metrics
  c5_ema_9 FLOAT64,
  c5_ema_21 FLOAT64,
  c5_ema_50 FLOAT64,
  c5_ema_100 FLOAT64,
  c5_ema_200 FLOAT64,
  
  -- EMA crossovers
  c5_ema_9_21_crossover INT64,
  c5_ema_21_50_crossover INT64,
  c5_ema_50_200_crossover INT64,
  c5_golden_cross INT64,
  c5_death_cross INT64,
  
  -- CPR (Central Pivot Range) metrics
  c5_cpr_pivot FLOAT64,
  c5_cpr_tc FLOAT64,  -- Top Central
  c5_cpr_bc FLOAT64,  -- Bottom Central
  c5_cpr_width FLOAT64,
  c5_cpr_width_percentile FLOAT64,
  
  -- CPR support/resistance levels
  c5_cpr_r1 FLOAT64,
  c5_cpr_r2 FLOAT64,
  c5_cpr_r3 FLOAT64,
  c5_cpr_s1 FLOAT64,
  c5_cpr_s2 FLOAT64,
  c5_cpr_s3 FLOAT64,
  
  -- Dual asset analysis (NIFTY & BANKNIFTY)
  c5_nifty_atr FLOAT64,
  c5_banknifty_atr FLOAT64,
  c5_nifty_momentum FLOAT64,
  c5_banknifty_momentum FLOAT64,
  c5_inter_market_divergence FLOAT64,
  c5_correlation_nifty_banknifty FLOAT64,
  
  -- Price position relative to indicators
  c5_price_to_ema9_ratio FLOAT64,
  c5_price_to_ema21_ratio FLOAT64,
  c5_price_to_ema50_ratio FLOAT64,
  c5_price_to_cpr_position FLOAT64,
  
  -- Volatility regime metrics
  c5_volatility_expansion FLOAT64,
  c5_volatility_contraction FLOAT64,
  c5_volatility_regime_change INT64,
  c5_volatility_percentile FLOAT64,
  
  -- Momentum indicators
  c5_rate_of_change_5min FLOAT64,
  c5_rate_of_change_15min FLOAT64,
  c5_rate_of_change_60min FLOAT64,
  c5_momentum_oscillator FLOAT64,
  
  -- Confluence detection
  c5_ema_confluence_count INT64,
  c5_cpr_ema_confluence FLOAT64,
  c5_technical_agreement_score FLOAT64,
  c5_signal_strength FLOAT64,
  
  -- Trend strength
  c5_trend_strength_ema FLOAT64,
  c5_trend_consistency FLOAT64,
  c5_trend_quality_score FLOAT64,
  
  -- Range analysis
  c5_daily_range_position FLOAT64,
  c5_range_expansion_probability FLOAT64,
  c5_range_contraction_score FLOAT64,
  
  -- VWAP integration
  c5_vwap FLOAT64,
  c5_price_to_vwap_ratio FLOAT64,
  c5_vwap_deviation FLOAT64,
  
  -- Bollinger Bands
  c5_bb_upper FLOAT64,
  c5_bb_middle FLOAT64,
  c5_bb_lower FLOAT64,
  c5_bb_width FLOAT64,
  c5_bb_position FLOAT64,
  
  -- Additional technical indicators
  c5_adx FLOAT64,
  c5_plus_di FLOAT64,
  c5_minus_di FLOAT64,
  c5_cci FLOAT64,
  c5_williams_r FLOAT64,
  
  -- Market breadth from technicals
  c5_advance_decline_ratio FLOAT64,
  c5_mcclellan_oscillator FLOAT64,
  c5_market_breadth_score FLOAT64,
  
  -- Seasonality adjustments
  c5_seasonal_atr_adjustment FLOAT64,
  c5_expiry_week_effect FLOAT64,
  c5_monthly_pivot_levels STRING,  -- JSON array
  
  -- ML-enhanced features
  c5_ml_trend_prediction FLOAT64,
  c5_ml_volatility_forecast FLOAT64,
  c5_ml_confluence_probability FLOAT64,
  c5_ml_regime_classification INT64,
  
  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)

  -- Additional features to reach target count
  c5_feature_85 FLOAT64,
  c5_feature_86 FLOAT64,
  c5_feature_87 FLOAT64,
  c5_feature_88 FLOAT64,
  c5_feature_89 FLOAT64,
  c5_feature_90 FLOAT64,
  c5_feature_91 FLOAT64,
  c5_feature_92 FLOAT64,
  c5_feature_93 FLOAT64,
  c5_feature_94 FLOAT64,

)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 5: ATR-EMA-CPR - 94 features for technical analysis",
  labels=[("component", "c5"), ("feature_count", "94"), ("version", "1.0")]
);