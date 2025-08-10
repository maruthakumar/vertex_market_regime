import pandas as pd
import openpyxl
from pathlib import Path
import json

# Path to the Excel file
file_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"

def analyze_excel_structure():
    """Analyze the structure of the market regime Excel file"""
    
    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    print(f"=== EXCEL FILE ANALYSIS ===")
    print(f"File: {Path(file_path).name}")
    print(f"Total number of sheets: {len(sheet_names)}")
    print("\n=== SHEET NAMES ===")
    for i, sheet_name in enumerate(sheet_names, 1):
        print(f"{i}. {sheet_name}")
    
    print("\n=== DETAILED SHEET ANALYSIS ===\n")
    
    sheet_analysis = {}
    
    for sheet_name in sheet_names:
        print(f"\n{'='*60}")
        print(f"SHEET: {sheet_name}")
        print(f"{'='*60}")
        
        try:
            # Read the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Basic info
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"\nColumn Headers:")
            for col in df.columns:
                print(f"  - {col}")
            
            # Show first few rows
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string(max_cols=10, max_colwidth=30))
            
            # Identify data types
            print(f"\nData Types:")
            for col, dtype in df.dtypes.items():
                print(f"  - {col}: {dtype}")
            
            # Check for any special patterns or metadata
            if sheet_name == "METADATA" or "metadata" in sheet_name.lower():
                print(f"\nMetadata content:")
                print(df.to_string())
            
            # Store analysis
            sheet_analysis[sheet_name] = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "purpose": identify_sheet_purpose(sheet_name, df)
            }
            
        except Exception as e:
            print(f"Error reading sheet: {e}")
            sheet_analysis[sheet_name] = {"error": str(e)}
    
    # Summary
    print(f"\n\n{'='*60}")
    print("=== SUMMARY ===")
    print(f"{'='*60}")
    
    # Group sheets by category
    regime_sheets = [s for s in sheet_names if "REGIME" in s.upper()]
    config_sheets = [s for s in sheet_names if any(x in s.upper() for x in ["CONFIG", "METADATA", "PARAMETERS"])]
    weight_sheets = [s for s in sheet_names if "WEIGHT" in s.upper()]
    other_sheets = [s for s in sheet_names if s not in regime_sheets + config_sheets + weight_sheets]
    
    print(f"\nSheet Categories:")
    print(f"  - Regime Definition Sheets: {len(regime_sheets)}")
    for s in regime_sheets[:5]:  # Show first 5
        print(f"    • {s}")
    if len(regime_sheets) > 5:
        print(f"    ... and {len(regime_sheets) - 5} more")
    
    print(f"\n  - Configuration/Metadata Sheets: {len(config_sheets)}")
    for s in config_sheets:
        print(f"    • {s}")
    
    print(f"\n  - Weight/Parameter Sheets: {len(weight_sheets)}")
    for s in weight_sheets:
        print(f"    • {s}")
    
    if other_sheets:
        print(f"\n  - Other Sheets: {len(other_sheets)}")
        for s in other_sheets:
            print(f"    • {s}")
    
    # Look for version info
    print(f"\n=== VERSION INFORMATION ===")
    version_info = extract_version_info(excel_file, sheet_names)
    if version_info:
        for key, value in version_info.items():
            print(f"  {key}: {value}")
    else:
        print("  No explicit version information found in sheets")
    
    # Extract from filename
    filename = Path(file_path).name
    if "20250627_195625_20250628_104335" in filename:
        print(f"\n  From filename:")
        print(f"    - Creation Date: 2025-06-27 19:56:25")
        print(f"    - Last Modified: 2025-06-28 10:43:35")

def identify_sheet_purpose(sheet_name, df):
    """Identify the purpose of a sheet based on its name and content"""
    sheet_upper = sheet_name.upper()
    
    if "METADATA" in sheet_upper:
        return "Configuration metadata and version information"
    elif "REGIME" in sheet_upper and "DEFINITION" in sheet_upper:
        return "Market regime definitions and parameters"
    elif "WEIGHT" in sheet_upper:
        return "Weightage configuration for regime detection"
    elif "INDICATOR" in sheet_upper:
        return "Technical indicator configuration"
    elif "THRESHOLD" in sheet_upper:
        return "Threshold values for regime classification"
    elif "CONFIG" in sheet_upper:
        return "General configuration parameters"
    elif "VALIDATION" in sheet_upper:
        return "Validation rules and constraints"
    else:
        # Try to infer from columns
        cols = [str(c).upper() for c in df.columns]
        if any("VOLATILITY" in c for c in cols):
            return "Volatility-related parameters"
        elif any("TREND" in c for c in cols):
            return "Trend-related parameters"
        elif any("STRUCTURE" in c for c in cols):
            return "Market structure parameters"
        else:
            return "Unknown/General data"

def extract_version_info(excel_file, sheet_names):
    """Extract version information from the Excel file"""
    version_info = {}
    
    # Check for metadata sheet
    metadata_sheets = [s for s in sheet_names if "METADATA" in s.upper()]
    
    for sheet_name in metadata_sheets:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # Look for version-related columns or rows
            for col in df.columns:
                if any(v in str(col).upper() for v in ["VERSION", "RELEASE", "DATE", "UPDATED"]):
                    version_info[col] = df[col].iloc[0] if len(df) > 0 else "N/A"
            
            # Check if data is in key-value format
            if len(df.columns) == 2:
                for _, row in df.iterrows():
                    key = str(row.iloc[0]).strip()
                    value = str(row.iloc[1]).strip()
                    if any(v in key.upper() for v in ["VERSION", "RELEASE", "DATE", "UPDATED"]):
                        version_info[key] = value
        except:
            pass
    
    return version_info

if __name__ == "__main__":
    analyze_excel_structure()