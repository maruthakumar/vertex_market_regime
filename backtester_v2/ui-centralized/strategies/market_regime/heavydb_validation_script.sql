-- HeavyDB Market Regime Validation Script
-- Generated: 2025-07-08T01:54:25.581922
-- Database: heavyai
-- Table: nifty_option_chain

-- 1. Verify table structure and data availability
\d nifty_option_chain

-- 2. Check data statistics
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT DATE(timestamp)) as unique_days,
    MIN(timestamp) as earliest_date,
    MAX(timestamp) as latest_date,
    AVG(underlying_price) as avg_underlying_price,
    COUNT(DISTINCT strike_price) as unique_strikes
FROM nifty_option_chain
WHERE symbol = 'NIFTY';

-- 3. Extract sample data for market regime analysis
SELECT 
    timestamp,
    underlying_price,
    strike_price,
    option_type,
    last_price,
    volume,
    open_interest,
    implied_volatility,
    delta_calculated,
    gamma_calculated,
    theta_calculated,
    vega_calculated
FROM nifty_option_chain
WHERE 
    symbol = 'NIFTY'
    AND timestamp >= '2024-12-01'
    AND timestamp < '2024-12-02'
ORDER BY timestamp, strike_price
LIMIT 100;

-- 4. Calculate Triple Rolling Straddle
WITH straddle_calc AS (
    SELECT 
        timestamp,
        underlying_price,
        strike_price,
        SUM(CASE WHEN option_type = 'CE' THEN last_price ELSE 0 END) as ce_price,
        SUM(CASE WHEN option_type = 'PE' THEN last_price ELSE 0 END) as pe_price
    FROM nifty_option_chain
    WHERE 
        symbol = 'NIFTY'
        AND timestamp >= '2024-12-01 09:15:00'
        AND timestamp <= '2024-12-01 15:30:00'
    GROUP BY timestamp, underlying_price, strike_price
)
SELECT 
    timestamp,
    underlying_price,
    strike_price,
    ce_price,
    pe_price,
    (ce_price + pe_price) as straddle_price,
    ABS(strike_price - underlying_price) as strike_distance
FROM straddle_calc
WHERE ce_price > 0 AND pe_price > 0
ORDER BY timestamp, strike_distance
LIMIT 50;

-- 5. Export results to CSV (HeavyDB specific)
-- Note: Use \copy command in heavysql client or COPY TO in Python
