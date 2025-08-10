"""
Create test configuration Excel files for validation testing

Generates various valid and invalid configuration files to test
the configuration validation system comprehensively.

Author: Market Regime System Optimizer
Date: 2025-07-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def create_valid_config():
    """Create a valid configuration Excel file"""
    sheets = {}
    
    # Master Configuration
    sheets['Master_Config'] = pd.DataFrame({
        'Parameter': [
            'version', 'regime_count', 'confidence_threshold',
            'cache_enabled', 'smoothing_enabled', 'smoothing_window'
        ],
        'Value': ['2.0.0', '12', '0.85', 'True', 'True', '3'],
        'Description': [
            'Configuration version',
            'Number of regimes (12 or 18)',
            'Minimum confidence threshold',
            'Enable caching',
            'Enable regime smoothing',
            'Smoothing window size'
        ]
    })
    
    # Indicator Weights
    sheets['Indicator_Weights'] = pd.DataFrame({
        'Indicator': [
            'GreekSentimentAnalysis',
            'TrendingOIPA',
            'StraddleAnalysis',
            'IVSurfaceAnalysis',
            'ATRIndicators',
            'MultiTimeframeConsensus'
        ],
        'Weight': [0.20, 0.15, 0.15, 0.10, 0.08, 0.07],
        'Enabled': [True, True, True, True, True, True],
        'Category': ['Greeks', 'OI', 'Straddle', 'Volatility', 'Technical', 'Consensus'],
        'Lookback_Period': [30, 60, 45, 30, 14, 60]
    })
    
    # Add remaining indicators to sum to 1.0
    additional_indicators = pd.DataFrame({
        'Indicator': ['RSI', 'MACD', 'BollingerBands', 'VWAP', 'MarketBreadth'],
        'Weight': [0.05, 0.05, 0.05, 0.05, 0.05],
        'Enabled': [True, True, True, True, True],
        'Category': ['Technical', 'Technical', 'Technical', 'Volume', 'Breadth'],
        'Lookback_Period': [14, 26, 20, 60, 30]
    })
    sheets['Indicator_Weights'] = pd.concat([sheets['Indicator_Weights'], additional_indicators], 
                                           ignore_index=True)
    
    # 12-Regime Configuration
    sheets['Regime_Thresholds_12'] = pd.DataFrame({
        'Regime_ID': [f'R{i}' for i in range(1, 13)],
        'Regime_Name': [
            'Low Vol Directional Trending',
            'Low Vol Directional Range',
            'Low Vol Non-Directional Trending',
            'Low Vol Non-Directional Range',
            'Moderate Vol Directional Trending',
            'Moderate Vol Directional Range',
            'Moderate Vol Non-Directional Trending',
            'Moderate Vol Non-Directional Range',
            'High Vol Directional Trending',
            'High Vol Directional Range',
            'High Vol Non-Directional Trending',
            'High Vol Non-Directional Range'
        ],
        'Volatility_Level': ['LOW']*4 + ['MODERATE']*4 + ['HIGH']*4,
        'Trend_Type': ['DIRECTIONAL', 'DIRECTIONAL', 'NONDIRECTIONAL', 'NONDIRECTIONAL']*3,
        'Structure_Type': ['TRENDING', 'RANGE']*6,
        'Min_Confidence': [0.6]*12
    })
    
    # Timeframe Configuration
    sheets['Timeframe_Config'] = pd.DataFrame({
        'Timeframe_Minutes': [1, 3, 5, 10, 15, 30],
        'Weight': [0.10, 0.15, 0.35, 0.20, 0.15, 0.05],
        'Lookback_Periods': [120, 100, 60, 30, 20, 10],
        'Min_Data_Points': [60, 50, 30, 15, 10, 5]
    })
    
    # Straddle Configuration
    sheets['Straddle_Config'] = pd.DataFrame({
        'Component': [
            'ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 
            'OTM1_CE', 'OTM1_PE', 'ATM_STRADDLE', 
            'ITM1_STRADDLE', 'OTM1_STRADDLE', 'COMBINED_TRIPLE'
        ],
        'Weight': [0.1]*10,
        'Enabled': [True]*10,
        'Rolling_Windows': ['3,5,10,15']*10,
        'Calculation_Method': ['Independent']*6 + ['Combined']*4
    })
    
    # Correlation Matrix Config
    sheets['Correlation_Config'] = pd.DataFrame({
        'Parameter': [
            'matrix_size', 'calculation_method', 'min_correlation',
            'max_correlation', 'pattern_threshold', 'update_frequency'
        ],
        'Value': ['10x10', 'pearson', '-1.0', '1.0', '0.85', '1min'],
        'Type': ['string', 'string', 'float', 'float', 'float', 'string']
    })
    
    # Greek Parameters
    sheets['Greek_Config'] = pd.DataFrame({
        'Greek': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
        'Weight': [0.30, 0.25, 0.20, 0.20, 0.05],
        'Normalization': ['tanh', 'sigmoid', 'linear', 'sigmoid', 'linear'],
        'Min_Value': [-1.0, 0.0, -100.0, 0.0, -50.0],
        'Max_Value': [1.0, 10.0, 0.0, 100.0, 50.0]
    })
    
    # Ensemble Methods
    sheets['Ensemble_Config'] = pd.DataFrame({
        'Method': ['HMM', 'GMM', 'RandomForest', 'GradientBoost'],
        'Weight': [0.35, 0.25, 0.20, 0.20],
        'Enabled': [True, True, True, True],
        'Parameters': [
            'n_states=12,covariance_type=full',
            'n_components=12,covariance_type=tied',
            'n_estimators=100,max_depth=10',
            'n_estimators=50,learning_rate=0.1'
        ]
    })
    
    return sheets


def create_invalid_configs():
    """Create various invalid configuration files for testing"""
    invalid_configs = []
    
    # Config 1: Weights don't sum to 1
    config1 = {
        'name': 'invalid_weights_sum',
        'sheets': {
            'Indicator_Weights': pd.DataFrame({
                'Indicator': ['Greek', 'OI', 'Straddle'],
                'Weight': [0.3, 0.3, 0.3],  # Sum = 0.9
                'Enabled': [True, True, True]
            })
        }
    }
    invalid_configs.append(config1)
    
    # Config 2: Invalid regime count
    config2 = {
        'name': 'invalid_regime_count',
        'sheets': {
            'Master_Config': pd.DataFrame({
                'Parameter': ['regime_count'],
                'Value': ['25']  # Outside valid range
            })
        }
    }
    invalid_configs.append(config2)
    
    # Config 3: Negative weights
    config3 = {
        'name': 'negative_weights',
        'sheets': {
            'Greek_Config': pd.DataFrame({
                'Greek': ['Delta', 'Gamma'],
                'Weight': [-0.5, 1.5],  # Negative weight
                'Min_Value': [-1.0, 0.0],
                'Max_Value': [1.0, 10.0]
            })
        }
    }
    invalid_configs.append(config3)
    
    # Config 4: Missing required columns
    config4 = {
        'name': 'missing_columns',
        'sheets': {
            'Timeframe_Config': pd.DataFrame({
                'Timeframe': [1, 5, 15],
                # Missing 'Weight' column
                'Lookback': [100, 200, 300]
            })
        }
    }
    invalid_configs.append(config4)
    
    # Config 5: Invalid parameter ranges
    config5 = {
        'name': 'invalid_ranges',
        'sheets': {
            'Greek_Config': pd.DataFrame({
                'Greek': ['Delta', 'Theta'],
                'Min_Value': [-2.0, 50.0],  # Invalid ranges
                'Max_Value': [2.0, 100.0]   # Theta should be negative
            })
        }
    }
    invalid_configs.append(config5)
    
    return invalid_configs


def create_edge_case_configs():
    """Create edge case configurations for testing"""
    edge_cases = []
    
    # Edge case 1: Minimum viable config
    case1 = {
        'name': 'minimum_config',
        'sheets': {
            'Master_Config': pd.DataFrame({
                'Parameter': ['regime_count'],
                'Value': ['12']
            }),
            'Indicator_Weights': pd.DataFrame({
                'Indicator': ['AllInOne'],
                'Weight': [1.0],
                'Enabled': [True]
            })
        }
    }
    edge_cases.append(case1)
    
    # Edge case 2: Maximum complexity
    indicators = [f'Indicator_{i}' for i in range(50)]
    case2 = {
        'name': 'maximum_complexity',
        'sheets': {
            'Indicator_Weights': pd.DataFrame({
                'Indicator': indicators,
                'Weight': [1.0/50]*50,
                'Enabled': [True]*50,
                'Lookback_Period': list(range(10, 510, 10))
            })
        }
    }
    edge_cases.append(case2)
    
    # Edge case 3: Unicode and special characters
    case3 = {
        'name': 'unicode_names',
        'sheets': {
            'Custom_Indicators': pd.DataFrame({
                'Indicator': ['Индикатор_1', '指标_2', 'Indicateur_3'],
                'Weight': [0.333, 0.333, 0.334],
                'Enabled': [True, True, True]
            })
        }
    }
    edge_cases.append(case3)
    
    return edge_cases


def save_config(sheets, filename, output_dir):
    """Save configuration to Excel file"""
    output_path = os.path.join(output_dir, filename)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for column in df:
                column_width = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2
                
    print(f"Created: {output_path}")


def main():
    """Generate all test configuration files"""
    # Create output directory
    output_dir = Path(__file__).parent / 'config_fixtures'
    output_dir.mkdir(exist_ok=True)
    
    # Create valid configuration
    print("Creating valid configuration...")
    valid_sheets = create_valid_config()
    save_config(valid_sheets, 'valid_config.xlsx', output_dir)
    
    # Create invalid configurations
    print("\nCreating invalid configurations...")
    for config in create_invalid_configs():
        save_config(config['sheets'], f"{config['name']}.xlsx", output_dir)
        
    # Create edge case configurations
    print("\nCreating edge case configurations...")
    for config in create_edge_case_configs():
        save_config(config['sheets'], f"{config['name']}.xlsx", output_dir)
        
    print(f"\nAll configuration files created in: {output_dir}")


if __name__ == '__main__':
    main()