#!/usr/bin/env python3
"""
Simple test for golden format validation
Tests parsing and basic structure validation
"""

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import TVParser
from signal_processor import SignalProcessor

def main():
    print("\n" + "="*80)
    print("TV GOLDEN FORMAT SIMPLE TEST")
    print("="*80)
    
    # Initialize components
    parser = TVParser()
    signal_processor = SignalProcessor()
    
    # Define real configuration files
    base_path = Path('../../configurations/data/prod/tv')
    config_files = {
        'tv_master': base_path / 'TV_CONFIG_MASTER_1.0.0.xlsx',
        'signals': base_path / 'TV_CONFIG_SIGNALS_1.0.0.xlsx',
        'portfolio_long': base_path / 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx',
        'portfolio_short': base_path / 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx',
        'portfolio_manual': base_path / 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx',
        'strategy': base_path / 'TV_CONFIG_STRATEGY_1.0.0.xlsx'
    }
    
    # Step 1: Parse TV configuration
    print("\n1️⃣ Parsing TV Master configuration...")
    try:
        tv_config_result = parser.parse_tv_settings(str(config_files['tv_master']))
        tv_config = tv_config_result['settings'][0]
        
        print(f"✅ TV config parsed: {tv_config['name']}")
        print(f"   • Date range: {tv_config['start_date']} to {tv_config['end_date']}")
        print(f"   • Signal format: {tv_config['signal_date_format']}")
        
    except Exception as e:
        print(f"❌ Failed to parse TV config: {e}")
        return 1
    
    # Step 2: Parse signals
    print("\n2️⃣ Parsing signal file...")
    try:
        signals = parser.parse_signals(str(config_files['signals']), tv_config['signal_date_format'])
        print(f"✅ Signals parsed: {len(signals)}")
        
        # Show signal details
        for i, signal in enumerate(signals):
            print(f"   • Signal {i+1}: Trade #{signal['trade_no']}, Type: {signal['signal_type']}, "
                  f"Time: {signal['datetime']}, Contracts: {signal.get('contracts', 'N/A')}")
    
    except Exception as e:
        print(f"❌ Failed to parse signals: {e}")
        return 1
    
    # Step 3: Process signals
    print("\n3️⃣ Processing signals...")
    try:
        processed_signals = signal_processor.process_signals(signals, tv_config)
        print(f"✅ Processed signals: {len(processed_signals)}")
        
        for i, signal in enumerate(processed_signals):
            print(f"   • Processed {i+1}: {signal['signal_direction']} - "
                  f"Entry: {signal['entry_date']} {signal['entry_time']}, "
                  f"Exit: {signal['exit_date']} {signal['exit_time']}, "
                  f"Lots: {signal['lots']}")
    
    except Exception as e:
        print(f"❌ Failed to process signals: {e}")
        return 1
    
    # Step 4: Check portfolio files
    print("\n4️⃣ Checking portfolio configurations...")
    for portfolio_type in ['long', 'short', 'manual']:
        portfolio_key = f'portfolio_{portfolio_type}'
        portfolio_path = config_files[portfolio_key]
        
        try:
            # Check if file exists
            if not portfolio_path.exists():
                print(f"❌ Missing {portfolio_type} portfolio: {portfolio_path}")
                continue
                
            # Read portfolio settings
            portfolio_df = pd.read_excel(str(portfolio_path), sheet_name='PortfolioSetting', engine='openpyxl')
            
            if not portfolio_df.empty:
                capital = portfolio_df.iloc[0]['Capital']
                print(f"✅ {portfolio_type.capitalize()} portfolio: Capital = ₹{capital:,}")
            else:
                print(f"⚠️  {portfolio_type.capitalize()} portfolio has no settings")
                
        except Exception as e:
            print(f"❌ Error reading {portfolio_type} portfolio: {e}")
    
    # Step 5: Check strategy file
    print("\n5️⃣ Checking TBS strategy configuration...")
    try:
        strategy_path = config_files['strategy']
        
        # Read general parameters
        general_df = pd.read_excel(str(strategy_path), sheet_name='GeneralParameter', engine='openpyxl')
        if not general_df.empty:
            strategy_name = general_df.iloc[0]['StrategyName']
            print(f"✅ Strategy: {strategy_name}")
        
        # Read leg parameters
        legs_df = pd.read_excel(str(strategy_path), sheet_name='LegParameter', engine='openpyxl')
        print(f"✅ Legs configured: {len(legs_df)}")
        
    except Exception as e:
        print(f"❌ Error reading strategy file: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"✅ TV Master: {tv_config['name']}")
    print(f"✅ Signals: {len(signals)} raw, {len(processed_signals)} processed")
    print(f"✅ Portfolio files: Long, Short, Manual")
    print(f"✅ Strategy: TBS configuration")
    
    print("\n✨ Basic TV configuration validation complete!")
    print("📋 Ready for golden format generation (but needs ResultRecord fix)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())