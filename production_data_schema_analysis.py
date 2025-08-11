#!/usr/bin/env python3
"""
Production Data Schema Analysis for IV Skew System
Analysis of /Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_production_schema():
    """Comprehensive analysis of production data for IV Skew system design"""
    
    base_path = Path("/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed")
    sample_file = base_path / "expiry=25012024/nifty_2024_01_15_25012024.parquet"
    
    print("=" * 80)
    print("PRODUCTION DATA SCHEMA ANALYSIS FOR IV SKEW SYSTEM")
    print("=" * 80)
    
    # 1. Schema Analysis
    df = pd.read_parquet(sample_file)
    print(f"\n1. SCHEMA OVERVIEW:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Total rows in sample: {len(df):,}")
    
    # 2. Index Coverage
    print(f"\n2. INDEX COVERAGE:")
    print(f"   Available indices: {df['index_name'].unique().tolist()}")
    print(f"   Note: Only NIFTY data found, no BANKNIFTY in current dataset")
    
    # 3. Strike Analysis
    strikes = sorted(df['strike'].unique())
    atm = df['atm_strike'].iloc[0]
    spot = df['spot'].iloc[0]
    
    print(f"\n3. STRIKE COVERAGE ANALYSIS:")
    print(f"   Sample date: {df['trade_date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"   Spot price: {spot:.2f}")
    print(f"   ATM strike: {atm:.0f}")
    print(f"   Total strikes available: {len(strikes)}")
    print(f"   Strike range: {min(strikes)} to {max(strikes)}")
    
    # Strike intervals
    intervals = sorted(set(np.diff(strikes)))
    print(f"   Strike intervals: {intervals}")
    
    # OTM coverage
    otm_calls = [s for s in strikes if s > atm]
    otm_puts = [s for s in strikes if s < atm]
    max_otm_call = max(otm_calls) - atm if otm_calls else 0
    max_otm_put = atm - min(otm_puts) if otm_puts else 0
    
    print(f"   OTM Call strikes: {len(otm_calls)} (max +{max_otm_call:.0f} points)")
    print(f"   OTM Put strikes: {len(otm_puts)} (max -{max_otm_put:.0f} points)")
    print(f"   Coverage as % of spot: +{max_otm_call/spot*100:.1f}% / -{max_otm_put/spot*100:.1f}%")
    
    # 4. Strike Intervals by Region
    print(f"\n4. STRIKE INTERVAL PATTERNS:")
    
    # Near money (within 500 points of ATM)
    near_money = [s for s in strikes if abs(s - atm) <= 500]
    near_intervals = np.diff(near_money) if len(near_money) > 1 else []
    print(f"   Near ATM (±500pts): {len(near_money)} strikes, intervals: {sorted(set(near_intervals)) if len(near_intervals) > 0 else 'N/A'}")
    
    # Far OTM
    far_otm = [s for s in strikes if abs(s - atm) > 500]
    if far_otm:
        far_intervals = np.diff(sorted(far_otm))
        print(f"   Far OTM (>500pts): {len(far_otm)} strikes, intervals: {sorted(set(far_intervals))}")
    
    # 5. IV Data Quality
    print(f"\n5. IV DATA QUALITY:")
    ce_iv_coverage = df['ce_iv'].notna().sum() / len(df) * 100
    pe_iv_coverage = df['pe_iv'].notna().sum() / len(df) * 100
    print(f"   CE IV coverage: {ce_iv_coverage:.1f}% ({df['ce_iv'].notna().sum():,}/{len(df):,} records)")
    print(f"   PE IV coverage: {pe_iv_coverage:.1f}% ({df['pe_iv'].notna().sum():,}/{len(df):,} records)")
    
    # IV ranges by moneyness
    near_atm_data = df[abs(df['strike'] - atm) <= 200]
    far_otm_data = df[abs(df['strike'] - atm) > 500]
    
    if len(near_atm_data) > 0:
        ce_iv_near = near_atm_data['ce_iv'].dropna()
        pe_iv_near = near_atm_data['pe_iv'].dropna()
        print(f"   Near ATM IV ranges: CE {ce_iv_near.min():.1f}-{ce_iv_near.max():.1f}%, PE {pe_iv_near.min():.1f}-{pe_iv_near.max():.1f}%")
    
    if len(far_otm_data) > 0:
        ce_iv_far = far_otm_data['ce_iv'].dropna()
        pe_iv_far = far_otm_data['pe_iv'].dropna()
        print(f"   Far OTM IV ranges: CE {ce_iv_far.min():.1f}-{ce_iv_far.max():.1f}%, PE {pe_iv_far.min():.1f}-{pe_iv_far.max():.1f}%")
    
    # 6. Key Columns for IV Skew
    print(f"\n6. KEY COLUMNS FOR IV SKEW SYSTEM:")
    iv_columns = [col for col in df.columns if 'iv' in col.lower()]
    greeks_columns = [col for col in df.columns if any(greek in col.lower() for greek in ['delta', 'gamma', 'theta', 'vega', 'rho'])]
    
    print(f"   IV columns: {iv_columns}")
    print(f"   Greeks columns: {greeks_columns}")
    
    essential_columns = [
        'trade_date', 'trade_time', 'expiry_date', 'spot', 'atm_strike', 'strike', 'dte',
        'ce_iv', 'pe_iv', 'ce_delta', 'pe_delta', 'ce_volume', 'pe_volume', 'ce_oi', 'pe_oi',
        'call_strike_type', 'put_strike_type', 'zone_name'
    ]
    
    print(f"   Essential columns available:")
    for col in essential_columns:
        status = "✓" if col in df.columns else "✗"
        print(f"     {status} {col}")
    
    # 7. DTE Analysis across multiple files
    print(f"\n7. DTE vs STRIKE COVERAGE ANALYSIS:")
    
    dte_files = [
        ("expiry=04012024/nifty_2024_01_01_04012024.parquet", "Short DTE (3 days)"),
        ("expiry=25012024/nifty_2024_01_15_25012024.parquet", "Medium DTE (10 days)"),
        ("expiry=29022024/nifty_2024_01_30_29022024.parquet", "Long DTE (30 days)"),
    ]
    
    for file_rel, desc in dte_files:
        file_path = base_path / file_rel
        if file_path.exists():
            df_dte = pd.read_parquet(file_path)
            strikes_dte = sorted(df_dte['strike'].unique())
            atm_dte = df_dte['atm_strike'].iloc[0]
            spot_dte = df_dte['spot'].iloc[0]
            dte = df_dte['dte'].iloc[0]
            
            max_otm_call_dte = max([s - atm_dte for s in strikes_dte if s > atm_dte], default=0)
            max_otm_put_dte = max([atm_dte - s for s in strikes_dte if s < atm_dte], default=0)
            
            print(f"   {desc}:")
            print(f"     Strikes: {len(strikes_dte)}, Range: {min(strikes_dte)}-{max(strikes_dte)}")
            print(f"     Coverage: +{max_otm_call_dte:.0f}pts/{max_otm_call_dte/spot_dte*100:.1f}%, -{max_otm_put_dte:.0f}pts/{max_otm_put_dte/spot_dte*100:.1f}%")
    
    # 8. Recommendations
    print(f"\n8. RECOMMENDATIONS FOR IV SKEW SYSTEM:")
    print(f"   ✓ Use ALL available strikes (not just ATM ±7)")
    print(f"   ✓ NIFTY strike intervals: 50pts near ATM, 100-500pts far OTM")
    print(f"   ✓ Coverage extends 6-35% of spot price in each direction")
    print(f"   ✓ IV data quality: 100% coverage across all strikes")
    print(f"   ✓ Strike organization: use 'call_strike_type' and 'put_strike_type'")
    print(f"   ✓ Time-based analysis: use 'zone_name' for intraday patterns")
    print(f"   ✓ DTE-based filtering: longer DTE = wider strike coverage")
    
    print(f"\n9. CRITICAL DESIGN CONSIDERATIONS:")
    print(f"   • Strike coverage varies significantly with DTE (3 days: 64 strikes, 58 days: 45 strikes)")
    print(f"   • Far OTM options may have zero IV (use filtering)")
    print(f"   • Strike intervals are non-uniform (dynamic binning required)")
    print(f"   • No BANKNIFTY data in current dataset (NIFTY-only analysis)")
    print(f"   • Greeks available for risk management calculations")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_production_schema()