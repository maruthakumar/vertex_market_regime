#!/usr/bin/env python3
"""
Deep analysis of Excel configuration sheets
"""

import pandas as pd
from pathlib import Path

excel_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"
excel_file = pd.ExcelFile(excel_path)

print("📊 COMPLETE EXCEL CONFIGURATION ANALYSIS")
print("="*80)

# 1. Master Configuration
print("\n1️⃣ MASTER CONFIGURATION")
print("-"*60)
master_df = pd.read_excel(excel_file, sheet_name='MasterConfiguration', skiprows=1)
master_df.columns = ['Parameter', 'Value', 'Options', 'Default', 'Description']
for idx, row in master_df.iterrows():
    if pd.notna(row['Parameter']):
        print(f"• {row['Parameter']}: {row['Value']}")

# 2. Indicator Configuration  
print("\n2️⃣ INDICATOR CONFIGURATION (6 Components)")
print("-"*60)
ind_df = pd.read_excel(excel_file, sheet_name='IndicatorConfiguration', skiprows=1)
ind_df.columns = ['IndicatorID', 'IndicatorName', 'BaseWeight', 'MinWeight', 'MaxWeight', 'AdaptiveEnabled', 'RequiresData', 'Description']
for idx, row in ind_df.iterrows():
    if pd.notna(row['IndicatorName']):
        print(f"• {row['IndicatorName']}: Base Weight = {row['BaseWeight']}")

# 3. Regime Classification (18 Regimes)
print("\n3️⃣ REGIME CLASSIFICATION (18 Regimes)")  
print("-"*60)
regime_df = pd.read_excel(excel_file, sheet_name='RegimeClassification')
print(f"Total Regimes: {len(regime_df)}")
for idx, row in regime_df.head(5).iterrows():
    if pd.notna(row.get('RegimeName')):
        print(f"• {row.get('RegimeName')}: {row.get('Description', '')}")
print("... and more")

# 4. Greek Sentiment Configuration
print("\n4️⃣ GREEK SENTIMENT CONFIGURATION")
print("-"*60)
greek_df = pd.read_excel(excel_file, sheet_name='GreekSentimentConfig')
for idx, row in greek_df.head(5).iterrows():
    if pd.notna(row.get('Parameter')):
        print(f"• {row.get('Parameter')}: {row.get('Value')}")

# 5. Straddle Analysis Configuration  
print("\n5️⃣ STRADDLE ANALYSIS CONFIGURATION")
print("-"*60)
straddle_df = pd.read_excel(excel_file, sheet_name='StraddleAnalysisConfig')
for idx, row in straddle_df.head(5).iterrows():
    if pd.notna(row.get('Parameter')):
        print(f"• {row.get('Parameter')}: {row.get('Value')}")

# 6. OI/PA Configuration
print("\n6️⃣ TRENDING OI/PA CONFIGURATION")
print("-"*60)
oi_df = pd.read_excel(excel_file, sheet_name='TrendingOIPAConfig')
for idx, row in oi_df.head(5).iterrows():
    if pd.notna(row.get('Parameter')):
        print(f"• {row.get('Parameter')}: {row.get('Value')}")

# 7. Dynamic Weightage Configuration
print("\n7️⃣ DYNAMIC WEIGHTAGE CONFIGURATION")
print("-"*60)
weight_df = pd.read_excel(excel_file, sheet_name='DynamicWeightageConfig')
for idx, row in weight_df.iterrows():
    if pd.notna(row.get('Component')):
        print(f"• {row.get('Component')}: Base={row.get('BaseWeight')}, Min={row.get('MinWeight')}, Max={row.get('MaxWeight')}")

# 8. Output Format
print("\n8️⃣ OUTPUT FORMAT (CSV Columns)")
print("-"*60)
output_df = pd.read_excel(excel_file, sheet_name='OutputFormat')
print(f"Total Output Columns: {len(output_df)}")
required_cols = output_df[output_df['Required'] == True]['ColumnName'].tolist()
print(f"Required Columns ({len(required_cols)}): {', '.join(required_cols[:10])}...")

# 9. Multi-Timeframe Configuration
print("\n9️⃣ MULTI-TIMEFRAME CONFIGURATION")
print("-"*60)
timeframe_df = pd.read_excel(excel_file, sheet_name='MultiTimeframeConfig')
for idx, row in timeframe_df.iterrows():
    if pd.notna(row.get('Timeframe')):
        print(f"• {row.get('Timeframe')}: Weight={row.get('Weight')}, Window={row.get('Window')}")

# 10. Intraday Settings
print("\n🔟 INTRADAY SETTINGS")
print("-"*60)
intraday_df = pd.read_excel(excel_file, sheet_name='IntradaySettings')
for idx, row in intraday_df.head(5).iterrows():
    if pd.notna(row.get('TimeSlot')):
        print(f"• {row.get('TimeSlot')}: {row.get('StartTime')} - {row.get('EndTime')}, Sensitivity={row.get('Sensitivity')}")

print("\n" + "="*80)
print("✅ Complete Excel configuration analyzed!")
print("This configuration controls ALL aspects of the market regime system.")