# Backtester Exit Time Fix

This document explains the issue with trades exiting at entry time (9:16) instead of the scheduled exit time (12:00) and the solution we implemented.

## The Problem

Trades were exiting at entry time (9:16) instead of waiting until the scheduled exit time (12:00) due to Stop Loss (SL) and Take Profit (TP) values that were too tight.

### Original Configuration

In the original `input_tbs_multi_legs.xlsx` file, the LegParameter sheet had these settings:

```
Leg 1 (CALL SELL): SL=percentage 100, TP=percentage 0
Leg 2 (PUT SELL): SL=percentage 100, TP=percentage 0
Leg 3 (CALL BUY): SL=percentage 0, TP=percentage 0
Leg 4 (PUT BUY): SL=percentage 0, TP=percentage 0
```

These values caused issues:
1. For SELL legs, a 100% SL means that if the price doubles, the SL triggers. This can happen very quickly with volatile options.
2. For BUY legs, a 0% SL/TP means effectively no SL or TP protection.
3. For SELL legs, a 0% TP means effectively no take profit target.

## The Solution

We created an updated strategy file (`input_tbs_fixed_exits.xlsx`) with wider SL/TP values:

```
Leg 1 (CALL SELL): SL=percentage 500, TP=percentage 100
Leg 2 (PUT SELL): SL=percentage 500, TP=percentage 100
Leg 3 (CALL BUY): SL=percentage 50, TP=percentage 100
Leg 4 (PUT BUY): SL=percentage 50, TP=percentage 100
```

These changes:
1. For SELL legs, widened the SL to 500% (price needs to increase 5x to trigger SL)
2. For SELL legs, added a 100% TP (price needs to decrease to 0 to trigger TP)
3. For BUY legs, added a 50% SL (price needs to decrease by half to trigger SL)
4. For BUY legs, added a 100% TP (price needs to double to trigger TP)

We also updated the portfolio file (`input_portfolio_fixed.xlsx`) to use the new strategy file.

## Testing and Verification

We developed a test script (`test_sl_tp.py`) to verify that the wider SL/TP values prevent premature exit. The test results:

```
Test: SELL leg with tight SL (original)
SL: 2%, TP: 0%
Exit time: 09:16:00, Price: 100.00, Reason: Target Hit
❌ EARLY EXIT: Trade exited early due to Target Hit

Test: SELL leg with wide SL (fixed)
SL: 500%, TP: 100%
Exit time: 12:00:00, Price: 97.00, Reason: Exit Time Hit
✅ SUCCESS: Trade exited at scheduled time (12:00)

Test: BUY leg with tight SL
SL: 2%, TP: 0%
Exit time: 09:16:00, Price: 100.00, Reason: Target Hit
❌ EARLY EXIT: Trade exited early due to Target Hit

Test: BUY leg with wide SL (fixed)
SL: 50%, TP: 100%
Exit time: 12:00:00, Price: 97.00, Reason: Exit Time Hit
✅ SUCCESS: Trade exited at scheduled time (12:00)
```

## Exit Priority

Trades exit based on the following priority rules:

1. **SL/TP Trigger**: If price reaches SL or TP level before EndTime
2. **Scheduled Exit**: At EndTime, all remaining positions are closed  
3. **Partial Exits**: At SqOff1Time/SqOff2Time if configured
4. **Strategy-level SL/TP**: If strategy profit/loss thresholds are hit

## Additional Documentation

For more details, see:
- `docs/exit_timing_guide.md` - Comprehensive guide to exit timing in the backtester
- `docs/sl_tp_recommendation.md` - Recommended SL/TP values for different strategies 