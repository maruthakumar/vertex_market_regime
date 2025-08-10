#!/usr/bin/env python3
"""
Debug Regime Detector Classification
"""

import numpy as np
from enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType

# Create test case
detector = Enhanced18RegimeDetector()

# Test scenario: Should be NORMAL_VOLATILE_MILD_BEARISH
test_data = {
    'underlying_price': 50000,
    'greek_sentiment': {'delta': -0.3, 'gamma': 0.03, 'theta': -0.015, 'vega': 0.10},
    'oi_data': {'call_oi': 10000, 'put_oi': 12000, 'call_volume': 3000, 'put_volume': 4000},
    'price_data': [50200, 50150, 50100, 50050, 50000],
    'technical_indicators': {'rsi': 35, 'macd': -20, 'macd_signal': -10},
    'implied_volatility': 0.35,
    'atr': 300
}

print("🔍 Debug Regime Detection")
print("=" * 50)

# Get result
result = detector.detect_regime(test_data)

print(f"\nExpected: NORMAL_VOLATILE_MILD_BEARISH")
print(f"Detected: {result['regime_type'].value}")
print(f"Confidence: {result['confidence']:.3f}")

# Debug components
print(f"\nComponents:")
print(f"  Directional: {result['components']['directional']:.3f}")
print(f"  Volatility: {result['components']['volatility']:.3f}")

# Manual calculation check
print("\n📊 Manual Calculation Check:")

# Greek sentiment
delta = -0.3
gamma = 0.03
theta = -0.015
vega = 0.10

greek_score = delta * 0.4 + np.sign(delta) * gamma * 0.3 - theta * 0.2 + vega * 0.1
print(f"Greek Score: {greek_score:.3f}")

# OI analysis
call_oi = 10000
put_oi = 12000
call_volume = 3000
put_volume = 4000

pcr_oi = call_oi / put_oi  # 0.833
pcr_volume = call_volume / put_volume  # 0.75
oi_score = (pcr_oi - 1.0) * 0.5 + (pcr_volume - 1.0) * 0.5
print(f"OI Score: {oi_score:.3f}")

# Price action
prices = [50200, 50150, 50100, 50050, 50000]
momentum = (prices[-1] - prices[0]) / prices[0]  # -0.004
price_score = np.tanh(momentum * 10)
print(f"Price Score: {price_score:.3f}")

# Technical indicators
rsi = 35
macd = -20
macd_signal = -10

rsi_score = (rsi - 50) / 50  # -0.3
macd_diff = macd - macd_signal  # -10
macd_score = np.tanh(macd_diff)  # close to -1
tech_score = (rsi_score + macd_score) / 2
print(f"Tech Score: {tech_score:.3f}")

# Weighted directional
weights = detector.indicator_weights
directional = (greek_score * weights['greek_sentiment'] + 
               oi_score * weights['oi_analysis'] + 
               price_score * weights['price_action'] + 
               tech_score * weights['technical_indicators'])
total_weight = sum([weights['greek_sentiment'], weights['oi_analysis'], 
                   weights['price_action'], weights['technical_indicators']])
directional /= total_weight

print(f"\nWeighted Directional: {directional:.3f}")

# Check thresholds
print(f"\nDirectional Thresholds:")
for key, value in detector.directional_thresholds.items():
    print(f"  {key}: {value}")

# Volatility calculation
iv = 0.35
atr_vol = 300 / 50000  # 0.006
price_vol = np.std(np.diff(prices) / np.array(prices[:-1])) * np.sqrt(252)
volatility = np.mean([iv, price_vol])

print(f"\nVolatility: {volatility:.3f}")
print(f"Volatility Thresholds:")
for key, value in detector.volatility_thresholds.items():
    print(f"  {key}: {value}")

# Classification
if volatility >= 0.70:
    vol_cat = "HIGH"
elif volatility >= 0.45:
    vol_cat = "NORMAL"
else:
    vol_cat = "LOW"

if directional >= 0.45:
    dir_cat = "STRONG_BULLISH"
elif directional >= 0.18:
    dir_cat = "MILD_BULLISH"
elif directional >= 0.08:
    dir_cat = "NEUTRAL"
elif directional >= 0.05:
    dir_cat = "SIDEWAYS"
elif directional >= -0.18:
    dir_cat = "MILD_BEARISH"
else:
    dir_cat = "STRONG_BEARISH"

print(f"\nClassification: {vol_cat}_VOLATILE_{dir_cat}")

# List all regime types
print("\n📋 All Enhanced18RegimeType values:")
for regime in Enhanced18RegimeType:
    print(f"  {regime.value}")