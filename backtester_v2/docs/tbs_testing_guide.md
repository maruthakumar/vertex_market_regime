# TBS (Time-Based Strategy) Testing Guide

This document provides comprehensive guidance for testing the Time-Based Strategy (TBS) component of the HeavyDB backtester, including test frameworks, test data, and execution procedures.

## Test Framework Overview

The TBS testing framework consists of several components:

1. **Standalone Unit Tests** - Test individual components without database dependencies
   - Strike selection tests
   - Premium differential tests
   - Exit time behavior tests

2. **HeavyDB Integration Tests** - Test SQL queries against actual database
   - Validate SQL patterns work with real data
   - Verify result sets and data integrity

3. **Test Runner** - Unified execution of all TBS tests

## Test Types and Their Purpose

### Strike Selection Tests

Tests all strike selection methods implemented in the TBS strategy, including:

- **ATM**: At-The-Money strike selection
- **ITM/OTM**: In-The-Money and Out-Of-The-Money strike selection
- **FIXED**: Fixed strike selection
- **PREMIUM**: Premium-based strike selection
- **ATM WIDTH**: Strike selection based on ATM straddle width
- **DELTA**: Delta-based strike selection

These tests verify that the SQL generation for each selection type is correct and properly handles all specified parameters.

### Premium Differential Tests

Tests the premium differential calculation features:

- **Premium diff abs**: Absolute difference between call and put prices
- **Premium diff percent**: Percentage difference between call and put prices
- **Call higher/Put higher**: Specific scenario where one option type has higher premium
- **Change strike**: Logic to change selected strike based on premium difference

### Exit Time Tests

Tests trade exit behavior to ensure:

- Trades exit at the specified exit time when SL/TP not hit
- SL (stop loss) is triggered correctly before exit time
- TP (take profit) is triggered correctly before exit time
- Different transaction types (BUY/SELL) exit correctly

### Database Integration Tests

Tests the actual SQL queries against a real HeavyDB database to verify:

- Queries are syntactically correct
- Queries return expected data
- Strike selection works with real data
- Premium differentials are calculated correctly

## Input Test Data

Each test type requires specific input data:

### Strike Selection Test Data
- File: `test_data/tbs_strike_selection_test.xlsx`
- Contains legs with different strike selection methods:
  - ATM, ITM1, ITM2, ITM3, OTM1, OTM2, OTM3
  - FIXED with values (19000, 19500, 20000)
  - PREMIUM with conditions (<=, >=) and values (50, 100, 150)
  - ATM WIDTH with multipliers (0.5, 1.0, 1.5)
  - DELTA with targets (0.25, 0.5, 0.75)

### Premium Differential Test Data
- File: `test_data/tbs_premium_diff_test.xlsx`
- Contains strategies with premium differential parameters:
  - premium_diff_abs with values (30, 50, 100)
  - premium_diff_call_higher/put_higher with values (20, 50)
  - premium_diff_percent with values (10%, 20%, 30%)
  - With/without strike adjustments

### Exit Time Test Data
- File: `test_data/tbs_exit_time_test.xlsx`
- Contains trades with different exit configurations:
  - No SL/TP, exit at specific time (9:16-12:00, 9:30-15:30)
  - Various SL percentages (20%, 50%, 100%, 500%)
  - Various TP percentages (20%, 50%, 100%)
  - BUY and SELL trades

## Running the Tests

### Running All Tests

The fastest way to run all TBS tests is with the test runner script:

```bash
cd /srv/samba/shared
bt/backtester_stable/BTRUN/run_tbs_tests.sh
```

### Options for the Test Runner

The test runner script supports several options:

```bash
bt/backtester_stable/BTRUN/run_tbs_tests.sh [--with-db] [--mock-db] [-v|--verbose LEVEL]
```

| Option | Description |
|--------|-------------|
| `--with-db` | Run tests against a real HeavyDB database |
| `--mock-db` | Run tests using a mock database (no real connection required) |
| `-v, --verbose LEVEL` | Set verbosity level (1-3) |
| `-h, --help` | Show help message |

### Using the Mock Database

For development and testing without a real HeavyDB connection, you can use the mock database:

```bash
# Run tests with mock database
bt/backtester_stable/BTRUN/run_tbs_tests.sh --mock-db
```

### Running with the Real Database

To run tests against the actual HeavyDB database:

```bash
# Run tests with real database
bt/backtester_stable/BTRUN/run_tbs_tests.sh --with-db
```

### Python Module Approach

You can also use the Python module directly:

```bash
# Run standalone tests only
python3 -m bt.backtester_stable.BTRUN.tests.run_tbs_tests

# Run database tests too
python3 -m bt.backtester_stable.BTRUN.tests.run_tbs_tests -d

# Control verbosity
python3 -m bt.backtester_stable.BTRUN.tests.run_tbs_tests -v 3
```

## Test Results

The test framework has been verified to work with:

1. **Standalone Tests**: All 27 tests pass
2. **Mock Database Tests**: All 9 database tests pass with mock data
3. **Real Database Tests**: All 9 database tests pass with actual HeavyDB database

In total, 36 tests have been implemented and verified to pass against both mock and real data.

### Database Test Coverage

The database integration tests verify:

1. ATM strike selection with real option chain data
2. ITM/OTM strike selection matching call/put_strike_type columns
3. Fixed strike selection using numeric strike values
4. Premium-based selection finding options with specific premium levels
5. Delta-based selection finding options with target delta
6. Premium differential calculation at ATM strike
7. Multiple time period checks across trading hours
8. Trading day queries including DTE filtering

## Extending the Test Framework

To add additional tests:

1. Create a new test file in `bt/backtester_stable/BTRUN/tests/`
2. Add the new file to the test runner in `run_tbs_tests.py`
3. Update the shell script `run_tbs_tests.sh` if needed

## HeavyDB Connection Information

The tests connect to HeavyDB using the following parameters:

- **Host**: `127.0.0.1`
- **Port**: `6274`
- **User**: `admin`
- **Database**: `heavyai`

These can be overridden using environment variables:

- `HEAVYDB_HOST`
- `HEAVYDB_PORT`
- `HEAVYDB_USER`
- `HEAVYDB_PASSWORD`
- `HEAVYDB_DBNAME`

## Troubleshooting

### Common Issues

1. **Database Connection Failures**:
   - Check HeavyDB is running
   - Verify connection parameters
   - Try the mock database with `--mock-db`

2. **Missing Tables**:
   - Verify nifty_option_chain exists
   - Check column names match test expectations

3. **Column Name Mismatches**:
   - The tests now handle both attribute and index-based access
   - Check column_names initialization in heavydb_tbs_test.py

4. **Import Errors**:
   - Run with the -m flag: `python3 -m bt.backtester_stable.BTRUN.tests.run_tbs_tests`
   - Ensure Python path includes parent directory

## SQL Query Patterns

Key SQL patterns used in TBS strategy:

### ATM Strike Selection
```sql
SELECT * 
FROM nifty_option_chain
WHERE trade_date = DATE '2025-04-01'
  AND trade_time = TIME '09:16:00'
  AND strike = atm_strike
  AND ce_symbol IS NOT NULL
```

### ITM/OTM Strike Selection
```sql
-- ITM1 for CALL
SELECT * 
FROM nifty_option_chain
WHERE trade_date = DATE '2025-04-01'
  AND trade_time = TIME '09:16:00'
  AND call_strike_type = 'ITM1'
  AND ce_symbol IS NOT NULL

-- OTM2 for PUT
SELECT * 
FROM nifty_option_chain
WHERE trade_date = DATE '2025-04-01'
  AND trade_time = TIME '09:16:00'
  AND put_strike_type = 'OTM2'
  AND pe_symbol IS NOT NULL
```

### Premium-Based Selection
```sql
SELECT * 
FROM nifty_option_chain
WHERE trade_date = DATE '2025-04-01'
  AND trade_time = TIME '09:16:00'
  AND ce_close <= 100
ORDER BY ABS(ce_close - 100) ASC
LIMIT 1
```

### Premium Differential Check
```sql
WITH atm_premium AS (
  SELECT 
    strike AS atm_strike,
    ce_close AS call_price,
    pe_close AS put_price,
    ABS(ce_close - pe_close) AS premium_diff
  FROM nifty_option_chain
  WHERE trade_date = DATE '2025-04-01'
    AND trade_time = TIME '09:16:00'
    AND strike = atm_strike
)
SELECT * FROM atm_premium
WHERE premium_diff >= 50
```

## Future Enhancements

Planned enhancements to the TBS testing framework:

1. End-to-end backtest execution tests with full portfolio Excel
2. Performance benchmarking of TBS SQL patterns
3. Automated test coverage reporting
4. Regression test suite with historical data 