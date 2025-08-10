#!/usr/bin/env python3
"""
Create Sample Configuration Excel File
=====================================

This script creates a sample Market Regime configuration Excel file
with all required sheets and example data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_sample_config():
    """Create a sample configuration Excel file"""
    
    # Define file path
    output_file = "MARKET_REGIME_SAMPLE_CONFIG.xlsx"
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # 1. IndicatorConfiguration
        indicator_data = {
            'Indicator': [
                'RSI', 'MACD', 'Bollinger Bands', 'ATR', 'Volume Profile',
                'Moving Average', 'Stochastic', 'OBV', 'ADX', 'Fibonacci'
            ],
            'Enabled': [True, True, True, True, True, False, True, False, True, False],
            'Weight': [0.15, 0.20, 0.15, 0.10, 0.10, 0.00, 0.15, 0.00, 0.15, 0.00],
            'Lookback': [14, 26, 20, 14, 30, 50, 14, 20, 14, 50],
            'Parameters': [
                '14,30,70', '12,26,9', '20,2', '14', '30',
                '50,200', '14,3,3', '20', '14', 'Auto'
            ],
            'Description': [
                'Relative Strength Index', 
                'Moving Average Convergence Divergence',
                'Volatility bands',
                'Average True Range',
                'Volume at price levels',
                'Simple/Exponential MA',
                'Momentum oscillator',
                'On Balance Volume',
                'Average Directional Index',
                'Retracement levels'
            ]
        }
        pd.DataFrame(indicator_data).to_excel(writer, sheet_name='IndicatorConfiguration', index=False)
        
        # 2. StraddleAnalysisConfig
        straddle_data = {
            'Straddle_Type': [
                'ATM', 'ITM1', 'ITM2', 'OTM1', 'OTM2', 
                'Symmetric', 'Weighted'
            ],
            'Enabled': [True, True, False, True, False, True, True],
            'Weight': [0.30, 0.20, 0.00, 0.20, 0.00, 0.15, 0.15],
            'EMA_Period': [20, 20, 20, 20, 20, 30, 25],
            'VWAP_Period': [30, 30, 30, 30, 30, 40, 35],
            'Volume_Threshold': [1000, 800, 600, 800, 600, 1200, 1000],
            'Premium_Adjustment': [1.0, 0.95, 0.90, 0.95, 0.90, 1.0, 1.05]
        }
        pd.DataFrame(straddle_data).to_excel(writer, sheet_name='StraddleAnalysisConfig', index=False)
        
        # 3. DynamicWeightageConfig
        dynamic_data = {
            'Parameter': [
                'Enable_Dynamic_Weights', 'Learning_Rate', 'Adaptation_Period',
                'Performance_Window', 'Min_Weight', 'Max_Weight',
                'Rebalance_Frequency', 'Decay_Factor'
            ],
            'Value': [
                'TRUE', 0.05, 100, 500, 0.05, 0.40, 60, 0.95
            ],
            'Description': [
                'Enable adaptive weight adjustment',
                'Rate of weight adaptation',
                'Bars for adaptation calculation',
                'Performance evaluation window',
                'Minimum allowed weight',
                'Maximum allowed weight',
                'Minutes between rebalancing',
                'Weight decay for old performance'
            ]
        }
        pd.DataFrame(dynamic_data).to_excel(writer, sheet_name='DynamicWeightageConfig', index=False)
        
        # 4. MultiTimeframeConfig
        timeframe_data = {
            'Timeframe': ['1min', '3min', '5min', '15min', '30min'],
            'Period_Minutes': [1, 3, 5, 15, 30],
            'Weight': [0.10, 0.15, 0.35, 0.25, 0.15],
            'Enabled': [True, True, True, True, True],
            'Aggregation_Method': ['Last', 'Average', 'Weighted', 'Weighted', 'Average'],
            'Lookback_Multiplier': [1, 1, 1, 2, 3]
        }
        pd.DataFrame(timeframe_data).to_excel(writer, sheet_name='MultiTimeframeConfig', index=False)
        
        # 5. GreekSentimentConfig
        greek_data = {
            'Greek_Type': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
            'Enabled': [True, True, True, True, False],
            'Weight': [0.35, 0.25, 0.20, 0.20, 0.00],
            'Sentiment_Threshold': [0.3, 0.05, -0.02, 0.15, 0.01],
            'Normalization': ['MinMax', 'ZScore', 'MinMax', 'ZScore', 'None'],
            'Aggregation': ['Sum', 'Average', 'Sum', 'Average', 'Sum'],
            'Strike_Range': [3, 3, 3, 3, 0],
            'Time_Decay_Factor': [0.95, 0.90, 0.85, 0.92, 1.0],
            'Directional_Bias': ['Both', 'Both', 'Negative', 'Both', 'Positive'],
            'Min_OI_Filter': [100, 100, 100, 100, 0],
            'Smoothing_Period': [5, 3, 5, 5, 0],
            'Alert_Threshold': [0.5, 0.1, -0.05, 0.25, 0.02],
            'Regime_Sensitivity': ['High', 'Medium', 'High', 'Medium', 'Low'],
            'Description': [
                'Price sensitivity',
                'Rate of delta change',
                'Time decay',
                'Volatility sensitivity',
                'Interest rate sensitivity'
            ]
        }
        pd.DataFrame(greek_data).to_excel(writer, sheet_name='GreekSentimentConfig', index=False)
        
        # 6. RegimeFormationConfig (18 Regimes)
        regime_data = {
            'Regime_Name': [
                'High_Volatile_Strong_Bullish', 'Normal_Volatile_Strong_Bullish', 'Low_Volatile_Strong_Bullish',
                'High_Volatile_Mild_Bullish', 'Normal_Volatile_Mild_Bullish', 'Low_Volatile_Mild_Bullish',
                'High_Volatile_Neutral', 'Normal_Volatile_Neutral', 'Low_Volatile_Neutral',
                'High_Volatile_Sideways', 'Normal_Volatile_Sideways', 'Low_Volatile_Sideways',
                'High_Volatile_Mild_Bearish', 'Normal_Volatile_Mild_Bearish', 'Low_Volatile_Mild_Bearish',
                'High_Volatile_Strong_Bearish', 'Normal_Volatile_Strong_Bearish', 'Low_Volatile_Strong_Bearish'
            ],
            'Regime_ID': list(range(1, 19)),
            'Directional_Component': [
                0.7, 0.7, 0.7,    # Strong Bullish
                0.3, 0.3, 0.3,    # Mild Bullish
                0.0, 0.0, 0.0,    # Neutral
                0.0, 0.0, 0.0,    # Sideways
                -0.3, -0.3, -0.3, # Mild Bearish
                -0.7, -0.7, -0.7  # Strong Bearish
            ],
            'Volatility_Component': [
                0.8, 0.4, 0.15,   # Strong Bullish (High, Normal, Low)
                0.8, 0.4, 0.15,   # Mild Bullish
                0.8, 0.4, 0.15,   # Neutral
                0.8, 0.4, 0.15,   # Sideways
                0.8, 0.4, 0.15,   # Mild Bearish
                0.8, 0.4, 0.15    # Strong Bearish
            ],
            'Min_Threshold': [
                0.45, 0.45, 0.45,     # Strong Bullish
                0.18, 0.18, 0.18,     # Mild Bullish
                0.08, 0.08, 0.08,     # Neutral
                -0.08, -0.08, -0.08,  # Sideways
                -0.45, -0.45, -0.45,  # Mild Bearish
                -1.0, -1.0, -1.0      # Strong Bearish
            ],
            'Max_Threshold': [
                1.0, 1.0, 1.0,        # Strong Bullish
                0.45, 0.45, 0.45,     # Mild Bullish
                0.18, 0.18, 0.18,     # Neutral
                0.08, 0.08, 0.08,     # Sideways
                -0.18, -0.18, -0.18,  # Mild Bearish
                -0.45, -0.45, -0.45   # Strong Bearish
            ],
            'Color_Code': [
                '#00FF00', '#32CD32', '#90EE90',  # Greens for Bullish
                '#7FFF00', '#ADFF2F', '#F0E68C',  # Yellow-Greens
                '#FFD700', '#FFA500', '#FFFFE0',  # Yellows for Neutral
                '#F0E68C', '#EEE8AA', '#FAFAD2',  # Light Yellows
                '#FFA07A', '#FF8C00', '#FFE4B5',  # Oranges
                '#FF0000', '#DC143C', '#FFB6C1'   # Reds for Bearish
            ]
        }
        pd.DataFrame(regime_data).to_excel(writer, sheet_name='RegimeFormationConfig', index=False)
        
        # 7. RegimeComplexityConfig
        complexity_data = {
            'Parameter': [
                'Regime_Mode', 'Enable_Adaptation', 'Adaptation_Rate',
                'Transition_Smoothing', 'Confidence_Threshold', 'Min_Regime_Duration',
                'Hysteresis_Buffer', 'Enable_ML_Enhancement', 'Feature_Selection',
                'Regime_History_Window', 'Anomaly_Detection', 'Alert_Sensitivity'
            ],
            'Value': [
                18, 'TRUE', 0.03, 
                5, 0.75, 12,
                0.08, 'FALSE', 'Auto',
                100, 'TRUE', 'Medium'
            ],
            'Description': [
                'Number of regime classifications (8 or 18)',
                'Enable adaptive regime detection',
                'Rate of regime adaptation',
                'Minutes for transition smoothing',
                'Minimum confidence for regime change',
                'Minimum minutes in regime',
                'Buffer to prevent rapid switching',
                'Use ML for regime enhancement',
                'Feature selection method',
                'Bars to keep in regime history',
                'Enable anomaly detection',
                'Alert sensitivity level'
            ]
        }
        pd.DataFrame(complexity_data).to_excel(writer, sheet_name='RegimeComplexityConfig', index=False)
        
    print(f"✅ Sample configuration file created: {output_file}")
    print(f"📍 Location: {os.path.abspath(output_file)}")
    
    # Create validation summary
    validation_info = {
        'Configuration Summary': {
            'Total Sheets': 7,
            'Indicators': '10 defined, 7 enabled',
            'Straddle Types': '7 defined, 5 enabled',
            'Timeframes': '5 (1min to 30min)',
            'Greeks': '5 defined, 4 enabled',
            'Regimes': '18 (full complexity)',
            'Adaptation': 'Enabled',
            'Total Parameters': 'Approximately 150+'
        },
        'Key Features': {
            'Dynamic Weights': 'Enabled with 0.05 learning rate',
            'Multi-timeframe': '5 timeframes with weighted aggregation',
            'Greek Analysis': 'Delta, Gamma, Theta, Vega enabled',
            'Regime Detection': '18-regime system with adaptation',
            'ML Enhancement': 'Disabled (can be enabled)',
            'Anomaly Detection': 'Enabled'
        },
        'Validation Points': {
            'Indicator Weights': 'Sum to 1.0 (valid)',
            'Straddle Weights': 'Sum to 1.0 (valid)',
            'Timeframe Weights': 'Sum to 1.0 (valid)',
            'Greek Weights': 'Sum to 1.0 (valid)',
            'Regime Thresholds': 'Properly ordered',
            'Learning Rate': '0.05 (valid range)',
            'Confidence Threshold': '0.75 (optimal)'
        }
    }
    
    # Save validation info
    with open('SAMPLE_CONFIG_INFO.json', 'w') as f:
        import json
        json.dump(validation_info, f, indent=2)
    
    print("\n📋 Configuration Details saved to: SAMPLE_CONFIG_INFO.json")
    
    return output_file


if __name__ == "__main__":
    create_sample_config()
    
    # Test validation
    print("\n🔍 Testing configuration validation...")
    from advanced_config_validator import validate_configuration_file
    
    is_valid, report, metadata = validate_configuration_file("MARKET_REGIME_SAMPLE_CONFIG.xlsx")
    
    print("\n" + report)
    
    if is_valid:
        print("\n✅ Sample configuration is VALID!")
    else:
        print("\n❌ Sample configuration has issues - check report above")