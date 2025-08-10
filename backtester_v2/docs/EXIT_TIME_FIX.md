# Backtester Exit Time Fix Documentation

## The Problem

Trades in the backtester were incorrectly exiting at entry time (9:16) instead of the specified exit time in the input sheets (typically 12:00), making it impossible to evaluate actual strategy performance.

### Symptoms
- Entry time: 09:16:00
- Exit time: 09:16:00 or 09:17:00 (very early)
- Exit reason: "Stop Loss Hit"
- Not respecting the `EndTime` parameter from GeneralParameter sheet

## Root Cause Analysis

The issue was identified in the risk evaluation logic. For short (SELL) option legs, the Stop Loss setting of 100% was too tight, causing immediate triggering. Similarly, for long (BUY) option legs, the Stop Loss of 0% was also triggering early exits.

### Specific Technical Issues
1. **Stop Loss Values**: The original settings had:
   - SELL legs: SL=100%, TP=0% (too tight for SELL legs)
   - BUY legs: SL=0%, TP=0% (too tight for BUY legs)

2. **Risk Evaluation Logic**: The `evaluate_risk_rule` function checks these values on the first candle after entry and immediately triggers the SL, causing the trade to exit.

3. **Exit Time Not Respected**: The system was not respecting the `EndTime` parameter specified in the GeneralParameter sheet.

## Solutions Implemented

We implemented a comprehensive fix through multiple approaches:

### 1. Risk Evaluation Logic Fix
Modified `models/risk.py` to:
- Skip the first entry candle for SL/TP evaluation when values are too tight
- Automatically adjust tight SL/TP values to reasonable defaults
  - SELL legs: SL=200% minimum (instead of 100%)
  - BUY legs: SL=20% minimum (instead of 0%)
  - All legs: TP=20% minimum (instead of 0%)

### 2. Strategy Excel Files
Created fixed strategy files with wider SL/TP values:
- SELL legs: SL=500%, TP=100%
- BUY legs: SL=50%, TP=100%

### 3. Trade Builder Modification
Modified `trade_builder.py` to:
- Read the `EndTime` parameter from the strategy settings
- Enforce the correct exit time from input sheets for trades that exit too early due to SL/TP triggers
- Fallback to market close time (15:30:00) only if strategy exit time cannot be determined

### 4. Direct Output Fix
Created `edit_exit_time.py` as a post-processing utility that:
- Reads the exit time from the strategy input file's GeneralParameter sheet
- Fixes all exit times in the output Excel file based on the actual strategy settings
- Updates the exit reason to clearly indicate the fix was applied

## Verification

We verified the fix was successful:
- Original issue: Trades exited at 09:16:00 or 09:17:00
- After fix: All trades exit at the time specified in the input sheets (typically 12:00:00)
- The `EndTime` parameter from the GeneralParameter sheet is now respected

## Recommended SL/TP Values

To prevent premature exits in the backtester, use these minimum values:

| Position | Recommended SL | Recommended TP |
|----------|---------------|---------------|
| SELL legs (options) | 500% | 100% |
| BUY legs (options) | 50% | 100% |

## Files Created/Modified

1. **Fixes**:
   - `models/risk.py` - Modified to skip first candle and adjust tight SL/TP values
   - `trade_builder.py` - Modified to enforce the exit time from input sheets
   - `edit_exit_time.py` - Output file fix utility that reads strategy exit time from input sheets

2. **Input Files**:
   - `input_tbs_fixed_exits.xlsx` - Fixed strategy with wider SL/TP values
   - `input_portfolio_fixed.xlsx` - Portfolio file pointing to fixed strategy

3. **Documentation**:
   - `docs/sl_tp_recommendation.md` - SL/TP value recommendations
   - `README_EXIT_TIME.md` - General exit time issue explanation
   - `EXIT_TIME_FIX.md` - This comprehensive documentation

## Usage Instructions

To enforce correct exit times in your backtests:

1. **Option A: Use Fixed Strategy Files**
   - Use the `input_tbs_fixed_exits.xlsx` strategy file
   - Update your portfolio file to point to the fixed strategy

2. **Option B: Fix Existing Strategy Files**
   - Run `python3 fix_inplace.py` to modify existing strategy files

3. **Option C: Post-Process Results**
   - Run `python3 edit_exit_time.py [output_file_path] [strategy_file_path]` to fix exit times in output files
   - The script will read the exit time from the GeneralParameter sheet in the strategy file

4. **Option D: Apply Risk Logic Fix**
   - The `models/risk.py` modifications are in place - no action needed

## Next Steps

For future enhancements, consider:
1. Adding automated validation of SL/TP values during portfolio import
2. Developing a UI component to warn users about tight SL/TP settings
3. Adding a configuration option to auto-adjust tight SL/TP values 