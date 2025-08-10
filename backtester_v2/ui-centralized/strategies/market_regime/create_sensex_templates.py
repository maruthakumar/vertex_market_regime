#!/usr/bin/env python3
"""
SENSEX Sample Template Generator
Creates missing SENSEX 0DTE, 1DTE, 3DTE Excel configuration templates
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

def create_sensex_0dte_template():
    """Create SENSEX 0DTE specific template"""
    
    # SENSEX-specific parameters for 0DTE (same-day expiry)
    sensex_config = {
        'symbol': 'SENSEX',
        'lot_size': 10,
        'tick_size': 0.05,
        'dte': 0,
        'trading_hours': '09:15-15:30',
        'settlement': 'Cash_Settled'
    }
    
    # Sheet 1: IndicatorConfiguration - Optimized for 0DTE SENSEX
    indicator_config = pd.DataFrame({
        'indicator_id': [
            'enhanced_straddle_analysis', 'greek_sentiment', 'oi_analysis', 
            'supporting_technical', 'iv_skew', 'atr_premium',
            'price_action_sentiment', 'volume_profile', 'momentum_oscillator', 'volatility_surface'
        ],
        'indicator_name': [
            'Enhanced Straddle Analysis', 'Greek Sentiment', 'OI Analysis',
            'Supporting Technical', 'IV Skew', 'ATR Premium', 
            'Price Action Sentiment', 'Volume Profile', 'Momentum Oscillator', 'Volatility Surface'
        ],
        'base_weight': [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.03, 0.02, 0.02],
        'min_weight': [0.10, 0.08, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01],
        'max_weight': [0.40, 0.35, 0.30, 0.25, 0.25, 0.20, 0.15, 0.10, 0.08, 0.08],
        'enabled': [True] * 10,
        'dte_adaptation': [True] * 10,
        'sensex_optimized': [True] * 10
    })
    
    # Sheet 2: StraddleAnalysisConfig - SENSEX 0DTE specific
    straddle_config = pd.DataFrame({
        'straddle_type': [
            'atm_straddle', 'itm1_straddle', 'otm1_straddle', 'combined_straddle',
            'atm_ce', 'atm_pe', 'dynamic_weighted_straddle'
        ],
        'base_weight': [0.35, 0.25, 0.20, 0.10, 0.04, 0.04, 0.02],
        'ema_periods': [
            '5,10,20', '10,20,50', '10,20,50', '5,10,20,50',
            '5,10', '5,10', '20,50'
        ],
        'vwap_types': [
            'current,previous', 'current', 'current', 'current,previous,weekly',
            'current', 'current', 'previous'
        ],
        'timeframes': [
            '1,3,5', '3,5,10', '3,5,10', '1,3,5,10',
            '1,3', '1,3', '5,10'
        ],
        'dte_0_multiplier': [1.5, 1.3, 1.2, 1.4, 1.1, 1.1, 1.0],
        'sensex_lot_adjustment': [True] * 7
    })
    
    # Sheet 3: DynamicWeightageConfig - Performance-based adjustments
    dynamic_config = pd.DataFrame({
        'component': [
            'enhanced_straddle_analysis', 'greek_sentiment', 'oi_analysis',
            'supporting_technical', 'iv_skew', 'atr_premium'
        ],
        'performance_lookback_minutes': [30, 60, 45, 90, 60, 30],
        'min_performance_threshold': [0.6, 0.55, 0.5, 0.45, 0.5, 0.55],
        'weight_increase_factor': [1.2, 1.15, 1.1, 1.05, 1.1, 1.15],
        'weight_decrease_factor': [0.8, 0.85, 0.9, 0.95, 0.9, 0.85],
        'adaptation_speed': [0.1, 0.08, 0.06, 0.04, 0.06, 0.08],
        'sensex_0dte_enabled': [True] * 6
    })
    
    # Sheet 4: MultiTimeframeConfig - SENSEX 0DTE timeframes
    timeframe_config = pd.DataFrame({
        'timeframe_minutes': [1, 3, 5, 10, 15],
        'timeframe_name': ['1min', '3min', '5min', '10min', '15min'],
        'base_weight': [0.30, 0.25, 0.20, 0.15, 0.10],
        'dte_0_weight_multiplier': [2.0, 1.5, 1.2, 0.8, 0.5],
        'update_frequency_seconds': [30, 60, 120, 300, 600],
        'sensex_optimized': [True] * 5,
        'intraday_focus': [True, True, True, False, False]
    })
    
    # Sheet 5: GreekSentimentConfig - SENSEX Greeks
    greek_config = pd.DataFrame({
        'greek_parameter': [
            'delta_atm', 'gamma_atm', 'theta_atm', 'vega_atm',
            'delta_skew', 'gamma_acceleration', 'theta_decay_rate',
            'vega_surface_curvature', 'implied_volatility_atm',
            'iv_skew_25delta', 'iv_skew_10delta', 'term_structure_slope', 'volatility_smile_asymmetry'
        ],
        'base_weight': [0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.01, 0.01],
        'sensex_adjustment_factor': [1.1, 1.2, 1.0, 0.9, 1.1, 1.3, 1.0, 0.8, 1.0, 0.9, 0.8, 0.7, 0.7],
        'dte_0_sensitivity': [1.5, 2.0, 3.0, 1.2, 1.3, 2.5, 2.8, 1.0, 1.4, 1.1, 1.0, 0.8, 0.8],
        'enabled': [True] * 13,
        'real_time_update': [True] * 13
    })
    
    # Sheet 6: RegimeFormationConfig - SENSEX-specific regime types
    regime_config = pd.DataFrame({
        'regime_id': list(range(1, 19)),
        'regime_name': [
            'Strong_Bullish_SENSEX_0DTE', 'Moderate_Bullish_SENSEX_0DTE', 'Weak_Bullish_SENSEX_0DTE',
            'Bullish_Consolidation_SENSEX_0DTE', 'Neutral_Balanced_SENSEX_0DTE', 'Neutral_Volatile_SENSEX_0DTE',
            'Neutral_Range_SENSEX_0DTE', 'Bearish_Consolidation_SENSEX_0DTE', 'Weak_Bearish_SENSEX_0DTE',
            'Moderate_Bearish_SENSEX_0DTE', 'Strong_Bearish_SENSEX_0DTE', 'High_Vol_Bullish_SENSEX_0DTE',
            'High_Vol_Neutral_SENSEX_0DTE', 'High_Vol_Bearish_SENSEX_0DTE', 'Low_Vol_Trending_SENSEX_0DTE',
            'Low_Vol_Ranging_SENSEX_0DTE', 'Transition_Regime_SENSEX_0DTE', 'Undefined_SENSEX_0DTE'
        ],
        'directional_threshold': [
            0.6, 0.3, 0.1, 0.05, 0.02, 0.02, 0.02, -0.05, -0.1, -0.3, -0.6,
            0.4, 0.0, -0.4, 0.2, 0.0, 0.0, 0.0
        ],
        'volatility_threshold': [
            0.25, 0.25, 0.25, 0.15, 0.15, 0.35, 0.15, 0.15, 0.25, 0.25, 0.25,
            0.50, 0.50, 0.50, 0.10, 0.10, 0.30, 0.30
        ],
        'confidence_threshold': [0.8] * 18,
        'sensex_calibrated': [True] * 18
    })
    
    # Sheet 7: RegimeComplexityConfig - SENSEX 0DTE complexity settings
    complexity_config = pd.DataFrame({
        'parameter': [
            'regime_mode', 'confidence_threshold', 'regime_smoothing',
            'update_frequency_seconds', 'lookback_minutes', 'transition_buffer',
            'sensex_lot_multiplier', 'dte_0_speed_factor', 'intraday_optimization',
            'real_time_adaptation'
        ],
        'value': [
            '18_REGIME', 0.75, 3, 60, 120, 0.1,
            10, 2.0, True, True
        ],
        'description': [
            'Use 18-regime classification for granular analysis',
            'Minimum confidence for regime classification',
            'Smoothing factor to prevent rapid regime switching',
            'Frequency of regime updates in seconds',
            'Historical lookback period in minutes',
            'Buffer to prevent oscillation between regimes',
            'SENSEX lot size multiplier for position sizing',
            'Speed factor for 0DTE same-day expiry',
            'Enable intraday optimization',
            'Enable real-time weight adaptation'
        ],
        'sensex_specific': [True] * 10
    })
    
    return {
        'IndicatorConfiguration': indicator_config,
        'StraddleAnalysisConfig': straddle_config,
        'DynamicWeightageConfig': dynamic_config,
        'MultiTimeframeConfig': timeframe_config,
        'GreekSentimentConfig': greek_config,
        'RegimeFormationConfig': regime_config,
        'RegimeComplexityConfig': complexity_config
    }

def create_sensex_1dte_template():
    """Create SENSEX 1DTE specific template"""
    config = create_sensex_0dte_template()
    
    # Adjust for 1DTE
    config['IndicatorConfiguration']['base_weight'] = [0.22, 0.18, 0.16, 0.12, 0.12, 0.08, 0.05, 0.03, 0.02, 0.02]
    config['StraddleAnalysisConfig']['dte_0_multiplier'] = [1.2, 1.1, 1.0, 1.1, 1.0, 1.0, 0.9]
    config['MultiTimeframeConfig']['dte_0_weight_multiplier'] = [1.5, 1.2, 1.0, 0.9, 0.7]
    config['GreekSentimentConfig']['dte_0_sensitivity'] = [1.2, 1.5, 2.0, 1.0, 1.1, 1.8, 2.0, 0.9, 1.2, 1.0, 0.9, 0.8, 0.7]
    
    # Update regime names for 1DTE
    regime_names = [name.replace('0DTE', '1DTE') for name in config['RegimeFormationConfig']['regime_name']]
    config['RegimeFormationConfig']['regime_name'] = regime_names
    
    # Update complexity config
    config['RegimeComplexityConfig'].loc[config['RegimeComplexityConfig']['parameter'] == 'dte_0_speed_factor', 'value'] = 1.5
    config['RegimeComplexityConfig'].loc[config['RegimeComplexityConfig']['parameter'] == 'update_frequency_seconds', 'value'] = 90
    
    return config

def create_sensex_3dte_template():
    """Create SENSEX 3DTE specific template"""
    config = create_sensex_0dte_template()
    
    # Adjust for 3DTE
    config['IndicatorConfiguration']['base_weight'] = [0.20, 0.16, 0.15, 0.14, 0.14, 0.08, 0.06, 0.03, 0.02, 0.02]
    config['StraddleAnalysisConfig']['dte_0_multiplier'] = [1.0, 0.9, 0.8, 0.9, 0.8, 0.8, 0.7]
    config['MultiTimeframeConfig']['dte_0_weight_multiplier'] = [1.0, 1.0, 1.0, 1.0, 1.0]
    config['GreekSentimentConfig']['dte_0_sensitivity'] = [1.0, 1.2, 1.5, 0.9, 1.0, 1.4, 1.6, 0.8, 1.0, 0.9, 0.8, 0.8, 0.7]
    
    # Update regime names for 3DTE
    regime_names = [name.replace('0DTE', '3DTE') for name in config['RegimeFormationConfig']['regime_name']]
    config['RegimeFormationConfig']['regime_name'] = regime_names
    
    # Update complexity config
    config['RegimeComplexityConfig'].loc[config['RegimeComplexityConfig']['parameter'] == 'dte_0_speed_factor', 'value'] = 1.0
    config['RegimeComplexityConfig'].loc[config['RegimeComplexityConfig']['parameter'] == 'update_frequency_seconds', 'value'] = 120
    config['RegimeComplexityConfig'].loc[config['RegimeComplexityConfig']['parameter'] == 'lookback_minutes', 'value'] = 180
    
    return config

def save_excel_template(config, filename):
    """Save configuration to Excel file"""
    try:
        output_path = f"/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/static/templates/{filename}"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in config.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Created: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error creating {filename}: {e}")
        return False

def main():
    """Main function to create all SENSEX sample templates"""
    print("🔧 Creating SENSEX Sample Templates...")
    
    # Create templates
    templates = {
        'SENSEX_0DTE_TEMPLATE.xlsx': create_sensex_0dte_template(),
        'SENSEX_1DTE_TEMPLATE.xlsx': create_sensex_1dte_template(),
        'SENSEX_3DTE_TEMPLATE.xlsx': create_sensex_3dte_template()
    }
    
    success_count = 0
    for filename, config in templates.items():
        if save_excel_template(config, filename):
            success_count += 1
    
    print(f"\n📊 SENSEX Template Creation Summary:")
    print(f"✅ Successfully created: {success_count}/{len(templates)} templates")
    
    if success_count == len(templates):
        print("🎉 All SENSEX sample templates created successfully!")
        print("\n📁 Files created:")
        for filename in templates.keys():
            print(f"  - /srv/samba/shared/bt/backtester_stable/BTRUN/server/app/static/templates/{filename}")
    else:
        print("⚠️ Some templates failed to create. Check errors above.")

if __name__ == "__main__":
    main()