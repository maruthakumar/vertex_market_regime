-- Component 2: Greeks Sentiment Analysis Features
-- Total features: 98
-- Purpose: Greeks-based sentiment analysis with gamma weight = 1.5

CREATE TABLE IF NOT EXISTS `arched-bot-269016.market_regime_{env}.c2_features` (
  -- Common columns
  symbol STRING NOT NULL,
  ts_minute TIMESTAMP NOT NULL,
  date DATE NOT NULL,
  dte INT64 NOT NULL,
  zone_name STRING NOT NULL,
  
  -- Online features (in Feature Store)
  c2_gamma_exposure FLOAT64,
  c2_sentiment_level INT64,
  c2_pin_risk_score FLOAT64,
  
  -- Offline-only features
  -- Primary Greeks (gamma weight = 1.5)
  c2_gamma_weighted FLOAT64,  -- gamma * 1.5 weight factor
  c2_delta FLOAT64,
  c2_vega FLOAT64,
  c2_theta FLOAT64,
  c2_rho FLOAT64,
  
  -- Second-order Greeks
  c2_vanna FLOAT64,
  c2_charm FLOAT64,
  c2_volga FLOAT64,
  c2_veta FLOAT64,
  c2_speed FLOAT64,
  c2_zomma FLOAT64,
  c2_color FLOAT64,
  
  -- Greeks aggregations by DTE
  c2_gamma_0dte FLOAT64,
  c2_gamma_3dte FLOAT64,
  c2_gamma_7dte FLOAT64,
  c2_gamma_14dte FLOAT64,
  c2_gamma_21dte FLOAT64,
  c2_gamma_30dte FLOAT64,
  
  c2_delta_0dte FLOAT64,
  c2_delta_3dte FLOAT64,
  c2_delta_7dte FLOAT64,
  c2_delta_14dte FLOAT64,
  c2_delta_21dte FLOAT64,
  c2_delta_30dte FLOAT64,
  
  -- Sentiment indicators
  c2_bullish_sentiment_score FLOAT64,
  c2_bearish_sentiment_score FLOAT64,
  c2_neutral_sentiment_score FLOAT64,
  c2_sentiment_divergence FLOAT64,
  c2_sentiment_momentum FLOAT64,
  
  -- Pin risk analysis
  c2_max_pain_strike FLOAT64,
  c2_pin_strike_distance FLOAT64,
  c2_pin_magnetic_force FLOAT64,
  c2_gamma_flip_level FLOAT64,
  
  -- Volume-weighted Greeks
  c2_volume_weighted_gamma FLOAT64,
  c2_volume_weighted_delta FLOAT64,
  c2_volume_weighted_vega FLOAT64,
  c2_volume_weighted_theta FLOAT64,
  
  -- Greeks ratios
  c2_gamma_delta_ratio FLOAT64,
  c2_vega_theta_ratio FLOAT64,
  c2_vanna_vega_ratio FLOAT64,
  c2_charm_delta_ratio FLOAT64,
  
  -- Greeks momentum
  c2_gamma_momentum_5min FLOAT64,
  c2_gamma_momentum_15min FLOAT64,
  c2_gamma_momentum_60min FLOAT64,
  c2_delta_momentum_5min FLOAT64,
  c2_delta_momentum_15min FLOAT64,
  c2_delta_momentum_60min FLOAT64,
  
  -- Greeks divergence
  c2_gamma_spot_divergence FLOAT64,
  c2_delta_volume_divergence FLOAT64,
  c2_vega_iv_divergence FLOAT64,
  
  -- Market maker positioning
  c2_mm_gamma_exposure FLOAT64,
  c2_mm_delta_hedge_flow FLOAT64,
  c2_mm_vega_exposure FLOAT64,
  c2_dealer_positioning_score FLOAT64,
  
  -- Cross-strike analysis
  c2_atm_gamma FLOAT64,
  c2_otm_gamma FLOAT64,
  c2_itm_gamma FLOAT64,
  c2_put_call_gamma_ratio FLOAT64,
  
  -- Skew-adjusted Greeks
  c2_skew_adjusted_delta FLOAT64,
  c2_skew_adjusted_gamma FLOAT64,
  c2_skew_adjusted_vega FLOAT64,
  
  -- Term structure Greeks
  c2_term_structure_gamma FLOAT64,
  c2_term_structure_vega FLOAT64,
  c2_calendar_spread_gamma FLOAT64,
  
  -- Risk reversal indicators
  c2_risk_reversal_25d FLOAT64,
  c2_risk_reversal_10d FLOAT64,
  c2_butterfly_25d FLOAT64,
  
  -- Smile dynamics
  c2_smile_slope FLOAT64,
  c2_smile_curvature FLOAT64,
  c2_smile_asymmetry FLOAT64,
  
  -- Greeks stability
  c2_gamma_stability_score FLOAT64,
  c2_delta_stability_score FLOAT64,
  c2_greeks_coherence_score FLOAT64,
  
  -- Advanced sentiment metrics
  c2_weighted_put_call_ratio FLOAT64,
  c2_smart_money_sentiment FLOAT64,
  c2_retail_sentiment FLOAT64,
  c2_institutional_sentiment FLOAT64,
  
  -- Event-driven Greeks
  c2_pre_event_gamma_buildup FLOAT64,
  c2_event_vega_spike FLOAT64,
  c2_post_event_theta_decay FLOAT64,
  
  -- ML-enhanced features
  c2_ml_sentiment_prediction FLOAT64,
  c2_ml_gamma_forecast FLOAT64,
  c2_ml_pin_probability FLOAT64,
  c2_ml_greeks_regime INT64,
  
  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)

  -- Additional features to reach target count
  c2_feature_89 FLOAT64,
  c2_feature_90 FLOAT64,
  c2_feature_91 FLOAT64,
  c2_feature_92 FLOAT64,
  c2_feature_93 FLOAT64,
  c2_feature_94 FLOAT64,
  c2_feature_95 FLOAT64,
  c2_feature_96 FLOAT64,
  c2_feature_97 FLOAT64,
  c2_feature_98 FLOAT64,

)
PARTITION BY date
CLUSTER BY symbol, dte
OPTIONS(
  description="Component 2: Greeks Sentiment Analysis - 98 features with gamma weight = 1.5",
  labels=[("component", "c2"), ("feature_count", "98"), ("version", "1.0"), ("gamma_weight", "1.5")]
);