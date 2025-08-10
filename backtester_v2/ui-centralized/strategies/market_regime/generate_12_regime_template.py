#!/usr/bin/env python3
"""
Generate 12-Regime Excel Template

Creates a comprehensive Excel template for the 12-regime system
with all required sheets and configurations.

Author: The Augster
Date: 2025-06-18
Version: 1.0.0
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_12_regime_excel_template():
    """Create comprehensive 12-regime Excel template"""
    
    template_path = "12_regime_market_regime_config.xlsx"
    
    try:
        with pd.ExcelWriter(template_path, engine='openpyxl') as writer:
            
            # 1. Regime Formation Configuration (12-Regime)
            regime_formation_data = [
                ['REGIME_COMPLEXITY', '12_REGIME', 'N/A', 'N/A', 'N/A', True, 'Regime complexity: 8_REGIME, 12_REGIME, or 18_REGIME'],
                
                # Low Volatility Regimes
                ['LOW_DIRECTIONAL_TRENDING', 0.25, 0.30, 0.70, 5, True, 'Low volatility directional trending market'],
                ['LOW_DIRECTIONAL_RANGE', 0.25, 0.30, 0.50, 5, True, 'Low volatility directional range-bound market'],
                ['LOW_NONDIRECTIONAL_TRENDING', 0.25, 0.15, 0.70, 5, True, 'Low volatility non-directional trending market'],
                ['LOW_NONDIRECTIONAL_RANGE', 0.25, 0.15, 0.50, 5, True, 'Low volatility non-directional range-bound market'],
                
                # Moderate Volatility Regimes
                ['MODERATE_DIRECTIONAL_TRENDING', 0.50, 0.50, 0.70, 4, True, 'Moderate volatility directional trending market'],
                ['MODERATE_DIRECTIONAL_RANGE', 0.50, 0.50, 0.50, 4, True, 'Moderate volatility directional range-bound market'],
                ['MODERATE_NONDIRECTIONAL_TRENDING', 0.50, 0.20, 0.70, 4, True, 'Moderate volatility non-directional trending market'],
                ['MODERATE_NONDIRECTIONAL_RANGE', 0.50, 0.20, 0.50, 4, True, 'Moderate volatility non-directional range-bound market'],
                
                # High Volatility Regimes
                ['HIGH_DIRECTIONAL_TRENDING', 0.80, 0.70, 0.70, 3, True, 'High volatility directional trending market'],
                ['HIGH_DIRECTIONAL_RANGE', 0.80, 0.70, 0.50, 3, True, 'High volatility directional range-bound market'],
                ['HIGH_NONDIRECTIONAL_TRENDING', 0.80, 0.25, 0.70, 3, True, 'High volatility non-directional trending market'],
                ['HIGH_NONDIRECTIONAL_RANGE', 0.80, 0.25, 0.50, 3, True, 'High volatility non-directional range-bound market'],
            ]
            
            regime_formation_df = pd.DataFrame(
                regime_formation_data,
                columns=['RegimeType', 'VolatilityThreshold', 'DirectionalThreshold', 'CorrelationThreshold', 'StabilityPeriods', 'Enabled', 'Description']
            )
            regime_formation_df.to_excel(writer, sheet_name='RegimeFormationConfig', index=False)
            
            # 2. Regime Complexity Configuration
            regime_complexity_data = [
                ['REGIME_COMPLEXITY', '12_REGIME', '8_REGIME,12_REGIME,18_REGIME', 'Choose regime complexity level', 'Determines number of regime types'],
                ['VOLATILITY_LEVELS', '3', '2,3', 'Number of volatility levels (High/Moderate/Low)', 'Affects regime granularity'],
                ['DIRECTIONAL_LEVELS', '2', '2,4,6', 'Number of directional levels', 'Directional/Non-directional (12-regime)'],
                ['STRUCTURE_LEVELS', '2', '1,2', 'Number of structure levels (12-regime only)', 'Trending/Range-bound structure analysis'],
                ['AUTO_SIMPLIFY', 'False', 'True,False', 'Auto-simplify to 8 regimes if needed', 'Fallback for performance'],
                ['REGIME_MAPPING_8', 'ENABLED', 'ENABLED,DISABLED', 'Enable 8-regime mapping', 'Maps 12 regimes to 8 for compatibility'],
                ['REGIME_MAPPING_12', 'ENABLED', 'ENABLED,DISABLED', 'Enable 12-regime system', 'Primary 12-regime classification'],
                ['CONFIDENCE_BOOST_12', '0.03', '0.0,0.1', 'Confidence boost for 12-regime mode', 'Balanced granularity bonus'],
                ['TRANSITION_SMOOTHING', 'ENHANCED', 'BASIC,ENHANCED', 'Regime transition smoothing', 'Reduces false regime changes'],
                ['PERFORMANCE_TRACKING', 'PER_REGIME', 'AGGREGATE,PER_REGIME', 'Performance tracking granularity', 'Individual vs combined tracking'],
                ['TRIPLE_STRADDLE_WEIGHT', '0.35', '0.2,0.5', '12-regime Triple Straddle weight allocation', 'Weight for Triple Straddle Analysis in 12-regime system']
            ]
            
            regime_complexity_df = pd.DataFrame(
                regime_complexity_data,
                columns=['Parameter', 'Value', 'Options', 'Description', 'Notes']
            )
            regime_complexity_df.to_excel(writer, sheet_name='RegimeComplexityConfig', index=False)
            
            # 3. Dynamic Weightage Configuration (12-Regime Optimized)
            dynamic_weightage_data = [
                ['triple_straddle_analysis', 0.35, 0.20, 0.50, True, True, 'Triple Straddle Analysis (ATM/ITM1/OTM1)'],
                ['technical_indicators', 0.30, 0.15, 0.45, True, True, 'EMA/VWAP/Pivot Technical Analysis'],
                ['volatility_indicators', 0.20, 0.10, 0.35, True, True, 'IV Percentile/ATR/Gamma Exposure'],
                ['market_microstructure', 0.15, 0.05, 0.25, True, True, 'Volume/OI/Bid-Ask Analysis'],
            ]
            
            dynamic_weightage_df = pd.DataFrame(
                dynamic_weightage_data,
                columns=['SystemName', 'CurrentWeight', 'MinWeight', 'MaxWeight', 'Enabled', 'AutoAdjust', 'Description']
            )
            dynamic_weightage_df.to_excel(writer, sheet_name='DynamicWeightageConfig', index=False)
            
            # 4. Multi-Timeframe Configuration
            timeframe_data = [
                ['3min', 0.25, True, 'Short-term momentum analysis'],
                ['5min', 0.30, True, 'Primary trend identification'],
                ['10min', 0.25, True, 'Medium-term structure analysis'],
                ['15min', 0.20, True, 'Long-term regime validation'],
            ]
            
            timeframe_df = pd.DataFrame(
                timeframe_data,
                columns=['Timeframe', 'Weight', 'Enabled', 'Description']
            )
            timeframe_df.to_excel(writer, sheet_name='MultiTimeframeConfig', index=False)
            
            # 5. Triple Straddle Configuration (12-Regime Integration)
            straddle_data = [
                ['ATM_STRADDLE', 0.50, True, 'At-the-money straddle analysis'],
                ['ITM1_STRADDLE', 0.25, True, 'In-the-money 1 strike analysis'],
                ['OTM1_STRADDLE', 0.25, True, 'Out-of-the-money 1 strike analysis'],
                ['CORRELATION_MATRIX', 0.80, True, 'Strike correlation threshold'],
                ['VOLUME_THRESHOLD', 100, True, 'Minimum volume for analysis'],
                ['IV_NORMALIZATION', True, True, 'Normalize IV for extreme values'],
                ['REGIME_INTEGRATION', True, True, 'Integrate with 12-regime system'],
            ]
            
            straddle_df = pd.DataFrame(
                straddle_data,
                columns=['Component', 'Weight_or_Value', 'Enabled', 'Description']
            )
            straddle_df.to_excel(writer, sheet_name='TripleStraddleConfig', index=False)
            
            # 6. Indicator Configuration (12-Regime Optimized)
            indicator_data = [
                ['IV_PERCENTILE', 'VOLATILITY', 0.15, 0.05, 0.25, True, True, 20, 'IV percentile analysis'],
                ['ATR_NORMALIZED', 'VOLATILITY', 0.12, 0.05, 0.20, True, True, 14, 'Normalized ATR'],
                ['GAMMA_EXPOSURE', 'VOLATILITY', 0.08, 0.03, 0.15, True, True, 10, 'Gamma exposure analysis'],
                ['EMA_ALIGNMENT', 'TECHNICAL', 0.15, 0.08, 0.25, True, True, 20, 'EMA alignment score'],
                ['PRICE_MOMENTUM', 'TECHNICAL', 0.12, 0.05, 0.20, True, True, 14, 'Price momentum indicator'],
                ['VOLUME_CONFIRMATION', 'TECHNICAL', 0.08, 0.03, 0.15, True, True, 10, 'Volume confirmation'],
                ['STRIKE_CORRELATION', 'STRADDLE', 0.15, 0.10, 0.25, True, True, 15, 'Strike correlation analysis'],
                ['VWAP_DEVIATION', 'STRADDLE', 0.10, 0.05, 0.18, True, True, 12, 'VWAP deviation analysis'],
                ['PIVOT_ANALYSIS', 'STRADDLE', 0.05, 0.02, 0.12, True, True, 8, 'Pivot point analysis'],
            ]
            
            indicator_df = pd.DataFrame(
                indicator_data,
                columns=['IndicatorName', 'Category', 'BaseWeight', 'MinWeight', 'MaxWeight', 'Enabled', 'Adaptive', 'LookbackPeriods', 'Description']
            )
            indicator_df.to_excel(writer, sheet_name='IndicatorConfiguration', index=False)
            
            # 7. 18→12 Regime Mapping
            mapping_data = [
                # Bullish → DIRECTIONAL_TRENDING/RANGE
                ['HIGH_VOLATILE_STRONG_BULLISH', 'HIGH_DIRECTIONAL_TRENDING', 0.95, 'Strong bullish to directional trending'],
                ['HIGH_VOLATILE_MILD_BULLISH', 'HIGH_DIRECTIONAL_RANGE', 0.90, 'Mild bullish to directional range'],
                ['NORMAL_VOLATILE_STRONG_BULLISH', 'MODERATE_DIRECTIONAL_TRENDING', 0.95, 'Normal strong bullish to moderate trending'],
                ['NORMAL_VOLATILE_MILD_BULLISH', 'MODERATE_DIRECTIONAL_RANGE', 0.90, 'Normal mild bullish to moderate range'],
                ['LOW_VOLATILE_STRONG_BULLISH', 'LOW_DIRECTIONAL_TRENDING', 0.95, 'Low strong bullish to low trending'],
                ['LOW_VOLATILE_MILD_BULLISH', 'LOW_DIRECTIONAL_RANGE', 0.90, 'Low mild bullish to low range'],
                
                # Bearish → DIRECTIONAL_TRENDING/RANGE
                ['HIGH_VOLATILE_STRONG_BEARISH', 'HIGH_DIRECTIONAL_TRENDING', 0.95, 'Strong bearish to directional trending'],
                ['HIGH_VOLATILE_MILD_BEARISH', 'HIGH_DIRECTIONAL_RANGE', 0.90, 'Mild bearish to directional range'],
                ['NORMAL_VOLATILE_STRONG_BEARISH', 'MODERATE_DIRECTIONAL_TRENDING', 0.95, 'Normal strong bearish to moderate trending'],
                ['NORMAL_VOLATILE_MILD_BEARISH', 'MODERATE_DIRECTIONAL_RANGE', 0.90, 'Normal mild bearish to moderate range'],
                ['LOW_VOLATILE_STRONG_BEARISH', 'LOW_DIRECTIONAL_TRENDING', 0.95, 'Low strong bearish to low trending'],
                ['LOW_VOLATILE_MILD_BEARISH', 'LOW_DIRECTIONAL_RANGE', 0.90, 'Low mild bearish to low range'],
                
                # Neutral/Sideways → NONDIRECTIONAL_TRENDING/RANGE
                ['HIGH_VOLATILE_NEUTRAL', 'HIGH_NONDIRECTIONAL_RANGE', 0.85, 'High neutral to non-directional range'],
                ['NORMAL_VOLATILE_NEUTRAL', 'MODERATE_NONDIRECTIONAL_RANGE', 0.85, 'Normal neutral to moderate range'],
                ['LOW_VOLATILE_NEUTRAL', 'LOW_NONDIRECTIONAL_RANGE', 0.85, 'Low neutral to low range'],
                ['HIGH_VOLATILE_SIDEWAYS', 'HIGH_NONDIRECTIONAL_TRENDING', 0.80, 'High sideways to non-directional trending'],
                ['NORMAL_VOLATILE_SIDEWAYS', 'MODERATE_NONDIRECTIONAL_TRENDING', 0.80, 'Normal sideways to moderate trending'],
                ['LOW_VOLATILE_SIDEWAYS', 'LOW_NONDIRECTIONAL_TRENDING', 0.80, 'Low sideways to low trending'],
            ]
            
            mapping_df = pd.DataFrame(
                mapping_data,
                columns=['Source18Regime', 'Target12Regime', 'MappingConfidence', 'Description']
            )
            mapping_df.to_excel(writer, sheet_name='RegimeMapping18to12', index=False)
            
        logger.info(f"✅ 12-regime Excel template created: {template_path}")
        return template_path
        
    except Exception as e:
        logger.error(f"❌ Error creating Excel template: {e}")
        raise

def validate_12_regime_template(template_path: str):
    """Validate the created 12-regime template"""
    try:
        # Read all sheets
        excel_data = pd.read_excel(template_path, sheet_name=None)
        
        expected_sheets = [
            'RegimeFormationConfig',
            'RegimeComplexityConfig', 
            'DynamicWeightageConfig',
            'MultiTimeframeConfig',
            'TripleStraddleConfig',
            'IndicatorConfiguration',
            'RegimeMapping18to12'
        ]
        
        # Check all sheets exist
        for sheet in expected_sheets:
            if sheet in excel_data:
                logger.info(f"✅ Sheet '{sheet}' found with {len(excel_data[sheet])} rows")
            else:
                logger.error(f"❌ Missing sheet: {sheet}")
                return False
        
        # Validate regime formation config
        regime_config = excel_data['RegimeFormationConfig']
        regime_count = len(regime_config[regime_config['RegimeType'] != 'REGIME_COMPLEXITY'])
        
        if regime_count == 12:
            logger.info(f"✅ Correct number of regimes: {regime_count}")
        else:
            logger.error(f"❌ Expected 12 regimes, found {regime_count}")
            return False
        
        # Validate mapping
        mapping_config = excel_data['RegimeMapping18to12']
        mapping_count = len(mapping_config)
        
        if mapping_count >= 18:
            logger.info(f"✅ Sufficient regime mappings: {mapping_count}")
        else:
            logger.error(f"❌ Expected at least 18 mappings, found {mapping_count}")
            return False
        
        logger.info("✅ 12-regime template validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating template: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating 12-Regime Excel Template...")
    
    try:
        template_path = create_12_regime_excel_template()
        print(f"✅ Template created: {template_path}")
        
        print("🔍 Validating template...")
        if validate_12_regime_template(template_path):
            print("✅ Template validation passed")
            print("🎉 12-REGIME EXCEL TEMPLATE READY FOR USE")
        else:
            print("❌ Template validation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
