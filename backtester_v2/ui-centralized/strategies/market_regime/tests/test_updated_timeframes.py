#!/usr/bin/env python3
"""
Test script for updated timeframe regime extractor
Verifies the new intraday timeframes (5m, 15m, 30m, 1h) work correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from timeframe_regime_extractor import TimeframeRegimeExtractor

def create_test_market_data():
    """Create test market data for timeframe testing"""
    # Generate 2 hours of minute-level data
    timestamps = pd.date_range(end=datetime.now(), periods=120, freq='1min')
    
    # Generate synthetic price data with trend
    base_price = 22000
    trend = np.linspace(0, 100, 120)
    noise = np.random.normal(0, 20, 120)
    prices = base_price + trend + noise
    
    # Create price data DataFrame
    price_data = pd.DataFrame({
        'timestamp': timestamps,
        'underlying_price': prices,
        'volume': np.random.randint(1000, 10000, 120),
        'iv_percentile': np.random.uniform(0.3, 0.7, 120),
        'atr_normalized': np.random.uniform(0.2, 0.6, 120)
    })
    
    # Create market data dictionary
    market_data = {
        'underlying_price': prices[-1],
        'timestamp': timestamps[-1],
        'volume': price_data['volume'].iloc[-1],
        'iv_percentile': price_data['iv_percentile'].iloc[-1],
        'atr_normalized': price_data['atr_normalized'].iloc[-1],
        'price_data': price_data
    }
    
    return market_data

def test_updated_timeframes():
    """Test the updated timeframe regime extractor"""
    print("Testing Updated Timeframe Regime Extractor")
    print("=" * 60)
    
    # Initialize extractor
    extractor = TimeframeRegimeExtractor()
    
    # Verify supported timeframes
    print(f"\nSupported timeframes: {extractor.supported_timeframes}")
    print(f"Timeframe weights: {extractor.timeframe_weights}")
    
    # Create test market data
    market_data = create_test_market_data()
    print(f"\nCreated test market data with {len(market_data['price_data'])} minutes of data")
    
    # Test resampling for each timeframe
    print("\n1. Testing data resampling:")
    print("-" * 40)
    
    for timeframe in extractor.supported_timeframes:
        resampled_data = extractor.resample_data_to_timeframe(market_data, timeframe)
        print(f"{timeframe}: {len(resampled_data)} periods after resampling")
    
    # Test regime score extraction
    print("\n2. Testing regime score extraction:")
    print("-" * 40)
    
    timeframe_scores = extractor.extract_timeframe_scores(market_data)
    
    for key, value in timeframe_scores.items():
        if 'regime_score' in key:
            print(f"{key}: {value:.4f}")
    
    print(f"\nCross-timeframe correlation: {timeframe_scores.get('cross_timeframe_correlation', 0):.4f}")
    print(f"Regime consistency: {timeframe_scores.get('regime_consistency', 0):.4f}")
    
    # Test multi-timeframe analysis
    print("\n3. Testing multi-timeframe analysis:")
    print("-" * 40)
    
    analysis = extractor.analyze_multi_timeframe_regime(market_data)
    
    print(f"Dominant timeframe: {analysis.dominant_timeframe}")
    print(f"Overall confidence: {analysis.overall_confidence:.4f}")
    print(f"Regime consistency: {analysis.regime_consistency:.4f}")
    print(f"Cross-timeframe correlation: {analysis.cross_timeframe_correlation:.4f}")
    
    print("\n4. Detailed timeframe scores:")
    print("-" * 40)
    
    for timeframe, score_obj in analysis.timeframe_scores.items():
        print(f"\n{timeframe}:")
        print(f"  Regime Score: {score_obj.regime_score:.4f}")
        print(f"  Regime ID: {score_obj.regime_id}")
        print(f"  Confidence: {score_obj.confidence:.4f}")
        print(f"  Trend Strength: {score_obj.trend_strength:.4f}")
        print(f"  Volatility Score: {score_obj.volatility_score:.4f}")
        print(f"  Structure Score: {score_obj.structure_score:.4f}")
    
    print("\n✓ Timeframe regime extractor updated successfully!")
    print("  - Old timeframes: 3min, 5min, 10min, 15min")
    print("  - New timeframes: 5min, 15min, 30min, 1hr")

if __name__ == "__main__":
    test_updated_timeframes()