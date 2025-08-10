# Understanding Exit Timing in the Backtester

This guide explains how trades exit in the backtesting system, particularly the interaction between scheduled exit times and Stop Loss (SL) / Take Profit (TP) conditions.

## Exit Priority

Trades exit based on the following priority rules:

1. **SL/TP Trigger**: If price reaches SL or TP level before EndTime
2. **Scheduled Exit**: At EndTime, all remaining positions are closed  
3. **Partial Exits**: At SqOff1Time/SqOff2Time if configured
4. **Strategy-level SL/TP**: If strategy profit/loss thresholds are hit

## Key Time Parameters

| Sheet | Parameter | Description | Format | Example | Notes |
|----|-----|----|-----|---|----|
| GeneralParameter | StrikeSelectionTime | When strikes are selected | HHMMSS | 91600 | 9:16 AM |
| GeneralParameter | StartTime | When trade entry begins | HHMMSS | 91600 | 9:16 AM |
| GeneralParameter | LastEntryTime | Latest time to enter trades | HHMMSS | 120000 | 12:00 PM |
| GeneralParameter | EndTime | When trades are exited | HHMMSS | 120000 | 12:00 PM |

In many strategies, StartTime and StrikeSelectionTime are both set to 91600 (9:16 AM), and EndTime is set to 120000 (12:00 PM).

## Common Problem: Premature Exits

If trades are exiting at entry time (9:16) instead of at the scheduled exit time (12:00), this is usually caused by:

### Tight SL/TP values triggering immediately

This is the most common issue - SL/TP values that are too tight can trigger immediately after entry.

**Problematic Settings:**
```
Leg 1 (CALL SELL): SL=percentage 100, TP=percentage 0
Leg 2 (PUT SELL): SL=percentage 100, TP=percentage 0
```

With these settings, a small price movement can trigger the stop loss immediately.

**Recommended Settings:**
```
Leg 1 (CALL SELL): SL=percentage 500, TP=percentage 100
Leg 2 (PUT SELL): SL=percentage 500, TP=percentage 100
Leg 3 (CALL BUY): SL=percentage 50, TP=percentage 100
Leg 4 (PUT BUY): SL=percentage 50, TP=percentage 100
```

These wider values provide enough room for normal price movements while still protecting against extreme moves.

### EndTime equals entry time

If EndTime is incorrectly set to the same value as StartTime (e.g., both 91600), trades will exit immediately after entry.

**Solution:** Ensure EndTime > StartTime.

## Time Format Handling

Time values are stored as integers in HHMMSS format (no colons) in Excel but are converted to proper time format (HH:MM:SS) when used in database queries.

Examples:
- 91600 → 09:16:00
- 120000 → 12:00:00
- 153000 → 15:30:00

## Other Exit Settings

The LegParameter sheet contains additional fields that affect exit behavior:

| Parameter | Description | Notes |
|-----|----|----|
| SLType | Stop loss type | `percentage` / `point` / etc. |
| SLValue | Stop loss value | Amount for SL |
| TGTType | Target type | `percentage` / `point` / etc. |
| TGTValue | Target value | Amount for TP |
| TrailSLType | Trailing SL type | For trailing stops |
| SL_TrailAt | When to start trailing | Profit level to start trailing |
| SL_TrailBy | Trail step | Amount to trail by |

## Verifying Exit Behavior

After running a backtest, check the exit reasons in the output Excel file:

- **Exit Time Hit**: Normal exit at the scheduled EndTime
- **Stop Loss Hit**: SL triggered before EndTime
- **Target Hit**: TP triggered before EndTime
- **Trail SL Hit**: Trailing SL triggered

## Troubleshooting Steps

1. Check EndTime parameter (should be different from StartTime)
2. Check SL/TP values (should be wide enough not to trigger immediately)
3. Verify risk rule evaluation in the code (models/risk.py)
4. Look for any tick data filtering issues that might affect exit decisions

## Code Implementation

The exit logic is implemented in these key files:
- heavydb_trade_processing.py (evaluate_trade_exit function)
- models/risk.py (evaluate_risk_rule function)
- trade_builder.py (build_trade_record function)

# Recommended SL/TP Values for Backtesting

This document provides guidance on suitable Stop Loss (SL) and Take Profit (TP) values for different strategy types to prevent premature exit while maintaining risk management.

## General Recommendations

| Position Type | Recommended SL | Recommended TP | Notes |
|---------------|----------------|----------------|-------|
| SELL (options) | 500% | 100% | Wide SL for volatile options |
| BUY (options) | 50% | 100% | Moderate SL, reasonable TP |
| SELL (futures) | 5% | 5% | Tighter values for less volatile futures |
| BUY (futures) | 5% | 5% | Tighter values for less volatile futures |

## Strategy-Specific Recommendations

### Short Straddle/Strangle (SELL ATM/OTM options)

```
CALL SELL: SL=percentage 500, TP=percentage 100
PUT SELL: SL=percentage 500, TP=percentage 100
```

These values provide enough room for intraday volatility while still protecting against extreme moves.

### Iron Condor/Iron Butterfly

```
ATM/ITM CALL SELL: SL=percentage 500, TP=percentage 100
ATM/ITM PUT SELL: SL=percentage 500, TP=percentage 100
OTM CALL BUY: SL=percentage 50, TP=percentage 200
OTM PUT BUY: SL=percentage 50, TP=percentage 200
```

The wider TP on long wings allows for more potential profit in extreme moves.

### Directional Long Options

```
CALL/PUT BUY: SL=percentage 50, TP=percentage 200
```

Higher TP for long options as they can see significant percentage gains.

### Calendar Spreads

```
FRONT MONTH SELL: SL=percentage 300, TP=percentage 100
BACK MONTH BUY: SL=percentage 50, TP=percentage 100
```

Calendar spreads have different dynamics due to time decay differences.

## Using These Values in Backtesting

To implement these recommendations:

1. Open your strategy Excel file
2. Go to the LegParameter sheet
3. Update the SLType, SLValue, TGTType, and TGTValue columns with the recommended values
4. Save the file and run your backtest

## Testing Your SL/TP Values

You can use the `test_sl_tp.py` script to test different SL/TP values with simulated price movements:

```bash
python3 test_sl_tp.py
```

Adjust the test cases in the script to test your specific SL/TP combinations. 