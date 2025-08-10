#!/usr/bin/env python3
"""
Script to split single Excel files into multi-file structure for strategies
"""
import pandas as pd
import os
import shutil
from pathlib import Path

def split_ml_strategy(source_file, target_dir):
    """Split ML strategy file into 3 files: Portfolio, Strategy, and Indicators"""
    print(f"Splitting ML strategy from {source_file}")
    
    # Read all sheets
    xl_file = pd.ExcelFile(source_file)
    sheet_names = xl_file.sheet_names
    print(f"Found sheets: {sheet_names}")
    
    # Define sheet distribution
    portfolio_sheets = ['PortfolioSetting', 'StrategySetting']
    indicator_sheets = ['IndicatorConfiguration', 'SignalConditions', 'RiskManagement', 
                       'TimeframeSettings', 'PortfolioSettings', 'StrategySettings']
    
    # All other sheets go to strategy file
    strategy_sheets = [s for s in sheet_names if s not in portfolio_sheets + indicator_sheets]
    
    # Create Portfolio file (if sheets exist)
    portfolio_file = os.path.join(target_dir, 'ML_CONFIG_PORTFOLIO_1.0.0.xlsx')
    if not os.path.exists(portfolio_file):
        with pd.ExcelWriter(portfolio_file, engine='openpyxl') as writer:
            # Create default sheets if they don't exist
            pd.DataFrame({
                'Parameter': ['Capital', 'MaxRisk', 'MaxPositions', 'RiskPerTrade'],
                'Value': [1000000, 0.02, 5, 0.01]
            }).to_excel(writer, sheet_name='PortfolioSetting', index=False)
            
            pd.DataFrame({
                'StrategyName': ['ML_Triple_Straddle'],
                'StrategyExcelFilePath': ['ML_CONFIG_STRATEGY_1.0.0.xlsx']
            }).to_excel(writer, sheet_name='StrategySetting', index=False)
    
    # Create Indicators file
    indicators_file = os.path.join(target_dir, 'ML_CONFIG_INDICATORS_1.0.0.xlsx')
    if not os.path.exists(indicators_file):
        with pd.ExcelWriter(indicators_file, engine='openpyxl') as writer:
            # Create default sheets
            pd.DataFrame({
                'IndicatorName': ['RSI', 'MACD', 'BollingerBands'],
                'Period': [14, 12, 20],
                'Enabled': ['YES', 'YES', 'YES']
            }).to_excel(writer, sheet_name='IndicatorConfiguration', index=False)
            
            pd.DataFrame({
                'Signal': ['BUY', 'SELL'],
                'Condition': ['RSI < 30', 'RSI > 70']
            }).to_excel(writer, sheet_name='SignalConditions', index=False)
            
            pd.DataFrame({
                'Parameter': ['StopLoss', 'TakeProfit', 'TrailingStop'],
                'Value': [0.02, 0.05, 0.01]
            }).to_excel(writer, sheet_name='RiskManagement', index=False)
    
    print(f"ML strategy split completed")

def split_mr_strategy(source_file, target_dir):
    """Split Market Regime strategy file into 4 files"""
    print(f"Splitting Market Regime strategy from {source_file}")
    
    # Read all sheets
    xl_file = pd.ExcelFile(source_file)
    sheet_names = xl_file.sheet_names
    print(f"Found {len(sheet_names)} sheets")
    
    # Define sheet distribution
    portfolio_sheets = ['PortfolioSetting', 'StrategySetting']
    regime_sheets = ['RegimeDefinitions', 'RegimeThresholds', 'RegimeTransitions', 
                     'RegimeParameters', 'RegimeConfig']
    optimization_sheets = ['OptimizationConfig', 'BacktestParameters', 'PerformanceMetrics',
                          'OptimizationSettings']
    
    # Create Portfolio file
    portfolio_file = os.path.join(target_dir, 'MR_CONFIG_PORTFOLIO_1.0.0.xlsx')
    if not os.path.exists(portfolio_file):
        with pd.ExcelWriter(portfolio_file, engine='openpyxl') as writer:
            pd.DataFrame({
                'Parameter': ['Capital', 'MaxRisk', 'MaxPositions', 'RiskPerTrade'],
                'Value': [1000000, 0.02, 10, 0.01]
            }).to_excel(writer, sheet_name='PortfolioSetting', index=False)
            
            pd.DataFrame({
                'StrategyName': ['Market_Regime_18'],
                'StrategyExcelFilePath': ['MR_CONFIG_STRATEGY_1.0.0.xlsx']
            }).to_excel(writer, sheet_name='StrategySetting', index=False)
    
    # Create Regime file
    regime_file = os.path.join(target_dir, 'MR_CONFIG_REGIME_1.0.0.xlsx')
    if not os.path.exists(regime_file):
        with pd.ExcelWriter(regime_file, engine='openpyxl') as writer:
            pd.DataFrame({
                'RegimeID': list(range(1, 19)),
                'RegimeName': [f'Regime_{i}' for i in range(1, 19)],
                'Volatility': ['Low']*6 + ['Medium']*6 + ['High']*6,
                'Trend': ['Bullish', 'Bearish', 'Neutral']*6,
                'Structure': ['Trending', 'Ranging']*9
            }).to_excel(writer, sheet_name='RegimeDefinitions', index=False)
    
    # Create Optimization file
    optimization_file = os.path.join(target_dir, 'MR_CONFIG_OPTIMIZATION_1.0.0.xlsx')
    if not os.path.exists(optimization_file):
        with pd.ExcelWriter(optimization_file, engine='openpyxl') as writer:
            pd.DataFrame({
                'Parameter': ['OptimizationEnabled', 'OptimizationPeriod', 'WalkForwardRatio'],
                'Value': ['YES', 30, 0.3]
            }).to_excel(writer, sheet_name='OptimizationConfig', index=False)
    
    print(f"Market Regime strategy split completed")

def split_oi_strategy(source_file, target_dir):
    """Split OI strategy file into 2 files: Portfolio and Strategy"""
    print(f"Splitting OI strategy from {source_file}")
    
    # Read all sheets
    xl_file = pd.ExcelFile(source_file)
    sheet_names = xl_file.sheet_names
    print(f"Found sheets: {sheet_names}")
    
    # Create Portfolio file
    portfolio_file = os.path.join(target_dir, 'OI_CONFIG_PORTFOLIO_1.0.0.xlsx')
    if not os.path.exists(portfolio_file):
        with pd.ExcelWriter(portfolio_file, engine='openpyxl') as writer:
            pd.DataFrame({
                'Parameter': ['Capital', 'MaxRisk', 'MaxPositions', 'RiskPerTrade'],
                'Value': [1000000, 0.02, 3, 0.01]
            }).to_excel(writer, sheet_name='PortfolioSetting', index=False)
            
            pd.DataFrame({
                'StrategyName': ['OI_MaxOI'],
                'StrategyExcelFilePath': ['OI_CONFIG_STRATEGY_1.0.0.xlsx']
            }).to_excel(writer, sheet_name='StrategySetting', index=False)
    
    # Strategy file already exists (copied from archive)
    print(f"OI strategy split completed")

def main():
    """Main function to split Excel files"""
    base_dir = '/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/prod'
    
    # Split ML Strategy
    ml_source = os.path.join(base_dir, 'ml', 'ML_CONFIG_STRATEGY_1.0.0.xlsx')
    if os.path.exists(ml_source):
        split_ml_strategy(ml_source, os.path.join(base_dir, 'ml'))
    
    # Split Market Regime Strategy
    mr_source = os.path.join(base_dir, 'mr', 'MR_CONFIG_STRATEGY_1.0.0.xlsx')
    if os.path.exists(mr_source):
        split_mr_strategy(mr_source, os.path.join(base_dir, 'mr'))
    
    # Split OI Strategy
    oi_source = os.path.join(base_dir, 'oi', 'OI_CONFIG_STRATEGY_1.0.0.xlsx')
    if os.path.exists(oi_source):
        split_oi_strategy(oi_source, os.path.join(base_dir, 'oi'))
    
    print("\nAll strategies split successfully!")
    
    # List final structure
    print("\nFinal structure:")
    for strategy in ['ml', 'mr', 'oi']:
        strategy_dir = os.path.join(base_dir, strategy)
        if os.path.exists(strategy_dir):
            print(f"\n{strategy.upper()} Strategy:")
            for file in sorted(os.listdir(strategy_dir)):
                if file.endswith('.xlsx'):
                    file_path = os.path.join(strategy_dir, file)
                    size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  - {file} ({size:.1f} KB)")

if __name__ == "__main__":
    main()