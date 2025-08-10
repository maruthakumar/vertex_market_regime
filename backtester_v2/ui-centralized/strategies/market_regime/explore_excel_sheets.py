#!/usr/bin/env python3
"""
Explore all sheets in the Excel configuration file to understand parameters
"""

import pandas as pd
from pathlib import Path
import json

excel_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"

# Load Excel file
excel_file = pd.ExcelFile(excel_path)

print(f"Excel File: {Path(excel_path).name}")
print(f"Total Sheets: {len(excel_file.sheet_names)}")
print("="*80)

# Explore each sheet
sheet_details = {}

for sheet_name in excel_file.sheet_names:
    print(f"\n📊 Sheet: {sheet_name}")
    print("-"*60)
    
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Get sheet info
        num_rows = len(df)
        num_cols = len(df.columns)
        columns = list(df.columns)
        
        print(f"Shape: {num_rows} rows × {num_cols} columns")
        print(f"Columns: {', '.join(columns[:5])}{' ...' if len(columns) > 5 else ''}")
        
        # Store sheet details
        sheet_details[sheet_name] = {
            'rows': num_rows,
            'columns': num_cols,
            'column_names': columns,
            'sample_data': df.head(3).to_dict() if num_rows > 0 else {}
        }
        
        # Show sample values for key sheets
        if sheet_name in ['MasterConfiguration', 'IndicatorConfiguration', 'RegimeClassification']:
            print("\nSample Data:")
            print(df.head(3).to_string(index=False, max_cols=5))
            
    except Exception as e:
        print(f"Error reading sheet: {e}")
        sheet_details[sheet_name] = {'error': str(e)}

# Save detailed exploration
output_path = "excel_sheet_exploration.json"
with open(output_path, 'w') as f:
    json.dump(sheet_details, f, indent=2, default=str)
    
print(f"\n\nDetailed exploration saved to: {output_path}")

# Summary of key configuration parameters
print("\n" + "="*80)
print("📋 KEY CONFIGURATION PARAMETERS FOUND:")
print("="*80)

# Check MasterConfiguration
if 'MasterConfiguration' in sheet_details:
    print("\n1. Master Configuration:")
    master_df = pd.read_excel(excel_file, sheet_name='MasterConfiguration')
    for idx, row in master_df.iterrows():
        if pd.notna(row.get('Parameter')):
            print(f"   - {row.get('Parameter')}: {row.get('Value')}")

# Check IndicatorConfiguration  
if 'IndicatorConfiguration' in sheet_details:
    print("\n2. Indicator Weights:")
    ind_df = pd.read_excel(excel_file, sheet_name='IndicatorConfiguration')
    for idx, row in ind_df.iterrows():
        if pd.notna(row.get('Indicator')):
            print(f"   - {row.get('Indicator')}: Base Weight={row.get('Base_Weight')}")

# Check RegimeClassification
if 'RegimeClassification' in sheet_details:
    print("\n3. Regime Classification Rules:")
    regime_df = pd.read_excel(excel_file, sheet_name='RegimeClassification')
    print(f"   - Total Regime Rules: {len(regime_df)}")
    print(f"   - Regime Types: {regime_df['Regime_Name'].unique()[:5].tolist() if 'Regime_Name' in regime_df.columns else 'N/A'}")