#!/usr/bin/env python3
"""
Test script for adaptive timeframe functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategies.market_regime.timeframe_regime_extractor import TimeframeRegimeExtractor
from strategies.market_regime.adaptive_timeframe_manager import AdaptiveTimeframeManager
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_adaptive_modes():
    """Test different trading modes"""
    print("="*60)
    print("Testing Adaptive Timeframe System")
    print("="*60)
    
    # Test 1: Intraday Mode
    print("\n1. INTRADAY MODE TEST")
    print("-"*40)
    extractor_intraday = TimeframeRegimeExtractor(trading_mode='intraday')
    config = extractor_intraday.get_adaptive_configuration()
    print(f"Active timeframes: {config['active_timeframes']}")
    print(f"Timeframe weights: {config['timeframe_weights']}")
    print(f"Mode parameters: {config['mode_parameters']}")
    
    # Test 2: Positional Mode
    print("\n2. POSITIONAL MODE TEST")
    print("-"*40)
    extractor_positional = TimeframeRegimeExtractor(trading_mode='positional')
    config = extractor_positional.get_adaptive_configuration()
    print(f"Active timeframes: {config['active_timeframes']}")
    print(f"Timeframe weights: {config['timeframe_weights']}")
    print(f"Mode parameters: {config['mode_parameters']}")
    
    # Test 3: Hybrid Mode
    print("\n3. HYBRID MODE TEST")
    print("-"*40)
    extractor_hybrid = TimeframeRegimeExtractor(trading_mode='hybrid')
    config = extractor_hybrid.get_adaptive_configuration()
    print(f"Active timeframes: {config['active_timeframes']}")
    print(f"Timeframe weights: {config['timeframe_weights']}")
    print(f"Mode parameters: {config['mode_parameters']}")
    
    # Test 4: Custom Mode
    print("\n4. CUSTOM MODE TEST")
    print("-"*40)
    custom_config = {
        'description': 'Ultra-fast scalping configuration',
        'timeframes': {
            '3min': {'enabled': True, 'weight': 0.50},
            '5min': {'enabled': True, 'weight': 0.35},
            '10min': {'enabled': True, 'weight': 0.15}
        },
        'risk_multiplier': 0.5,
        'transition_threshold': 0.6
    }
    extractor_custom = TimeframeRegimeExtractor(trading_mode='custom', adaptive_config=custom_config)
    config = extractor_custom.get_adaptive_configuration()
    print(f"Active timeframes: {config['active_timeframes']}")
    print(f"Timeframe weights: {config['timeframe_weights']}")
    print(f"Mode parameters: {config['mode_parameters']}")
    
    # Test 5: Mode Switching
    print("\n5. MODE SWITCHING TEST")
    print("-"*40)
    print("Starting with hybrid mode...")
    extractor = TimeframeRegimeExtractor(trading_mode='hybrid')
    print(f"Initial timeframes: {extractor.supported_timeframes}")
    
    print("\nSwitching to intraday mode...")
    extractor.switch_trading_mode('intraday')
    print(f"New timeframes: {extractor.supported_timeframes}")
    
    print("\nSwitching to positional mode...")
    extractor.switch_trading_mode('positional')
    print(f"New timeframes: {extractor.supported_timeframes}")
    
    # Test 6: Sample Data Processing
    print("\n6. SAMPLE DATA PROCESSING TEST")
    print("-"*40)
    # Create sample market data
    sample_data = {
        'underlying_price': 19500,
        'timestamp': datetime.now(),
        'volume': 1000000,
        'iv_percentile': 0.45,
        'atr_normalized': 0.35
    }
    
    try:
        # Extract timeframe scores (will use default values without real data)
        scores = extractor_hybrid.extract_timeframe_scores(sample_data)
        print("Timeframe scores extracted successfully:")
        for key, value in scores.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Note: Score extraction requires real market data connection: {e}")
    
    print("\n" + "="*60)
    print("Adaptive Timeframe System Test Complete!")
    print("="*60)

def test_timeframe_manager():
    """Test the adaptive timeframe manager directly"""
    print("\n\nTESTING ADAPTIVE TIMEFRAME MANAGER")
    print("="*60)
    
    manager = AdaptiveTimeframeManager()
    
    # Test configuration validation
    print("\n1. CONFIGURATION VALIDATION")
    print("-"*40)
    is_valid, errors = manager.validate_configuration()
    print(f"Configuration valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test performance optimization
    print("\n2. PERFORMANCE-BASED OPTIMIZATION")
    print("-"*40)
    print("Original weights:", manager.get_timeframe_weights())
    
    # Simulate performance data
    performance_data = {
        '5min': 0.85,   # High accuracy
        '15min': 0.75,  # Good accuracy
        '30min': 0.65,  # Moderate accuracy
        '1hr': 0.60     # Lower accuracy
    }
    
    manager.optimize_weights_from_performance(performance_data)
    print("Optimized weights:", manager.get_timeframe_weights())
    
    # Test mode suggestion
    print("\n3. MODE SUGGESTION")
    print("-"*40)
    suggested_mode = manager.suggest_mode_from_strategy(
        holding_period_minutes=30,
        trade_frequency='high'
    )
    print(f"Suggested mode for 30-min holding, high frequency: {suggested_mode}")
    
    suggested_mode = manager.suggest_mode_from_strategy(
        holding_period_minutes=300,
        trade_frequency='low'
    )
    print(f"Suggested mode for 300-min holding, low frequency: {suggested_mode}")
    
    # Test configuration export/import
    print("\n4. CONFIGURATION EXPORT/IMPORT")
    print("-"*40)
    exported_config = manager.export_configuration()
    print(f"Exported configuration: {exported_config['mode']}")
    
    # Import back
    success = manager.import_configuration(exported_config)
    print(f"Import successful: {success}")

if __name__ == "__main__":
    test_adaptive_modes()
    test_timeframe_manager()