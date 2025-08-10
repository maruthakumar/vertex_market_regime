# HeavyDB Guardrails Documentation

## Overview

The HeavyDB Guardrails module provides a comprehensive system to enforce best practices for HeavyDB queries, prevent performance issues, and optimize query execution. It helps protect against common mistakes that can lead to excessive resource usage or crash the GPU database.

## Key Features

- **Query Analysis**: Automatically detects risky patterns in SQL queries
- **Query Optimization**: Automatically improves queries by adding limits, replacing SELECT *, etc.
- **Performance Tracking**: Measures and logs query execution times
- **Risk Classification**: Categorizes queries by risk level (LOW, MEDIUM, HIGH, CRITICAL)
- **Enforcement**: Can block execution of high-risk queries
- **Metrics Collection**: Tracks query patterns and performance over time

## Quick Start

### Using the Enhanced Connection

```python
from bt.backtester_stable.BTRUN.dal.heavydb_connection_enhanced import get_connection

# Get a connection with guardrails enabled
conn = get_connection()

# Execute a query - it will be analyzed and optimized automatically
result = conn.execute("SELECT * FROM nifty_option_chain WHERE trade_date = '2025-04-01'")

# Get metrics about past query execution
metrics = conn.get_global_metrics()
print(f"Executed {metrics['total_queries']} queries, optimized {metrics['optimized_queries']}")
```

### Using the Decorator

```python
from bt.backtester_stable.BTRUN.dal.heavydb_guardrails import query_guardrails
from bt.backtester_stable.BTRUN.dal.heavydb_connection import get_connection

@query_guardrails(warn_only=True)
def fetch_option_data(trade_date):
    conn = get_connection()
    query = f"SELECT * FROM nifty_option_chain WHERE trade_date = '{trade_date}'"
    return conn.execute(query)

# The decorator will analyze and optimize the query
result = fetch_option_data('2025-04-01')
```

### Using the Command-line Analyzer

```bash
# Analyze a query
python -m bt.backtester_stable.BTRUN.scripts.query_analyzer --query "SELECT * FROM nifty_option_chain"

# Analyze and optimize a query stored in a file
python -m bt.backtester_stable.BTRUN.scripts.query_analyzer --file path/to/query.sql --optimize

# Benchmark a query's performance
python -m bt.backtester_stable.BTRUN.scripts.query_analyzer --file path/to/query.sql --benchmark
```

## Configuration Options

The guardrails system can be configured through environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HEAVYDB_ENFORCE_GUARDRAILS` | `true` | Whether to enable guardrails |
| `HEAVYDB_WARN_ONLY` | `false` | If true, issues warnings for risky queries but still executes them |
| `HEAVYDB_QUERY_TIMEOUT_MS` | `60000` | Query timeout in milliseconds |
| `HEAVYDB_COLLECT_METRICS` | `true` | Whether to collect performance metrics |

You can also set these options when creating a connection:

```python
conn = get_connection(
    enforce_guardrails=True,
    warn_only=False,
    collect_metrics=True,
    query_timeout_ms=30000  # 30 seconds
)
```

## Best Practices Enforced

The guardrails system enforces these best practices:

### Critical Issues (Blocks Execution by Default)

1. **Missing Date Filter**: Queries on large tables like `nifty_option_chain` must include a `trade_date` filter.
2. **No WHERE Clause**: Queries on large tables must have a WHERE clause.
3. **Unfiltered Joins**: Joins between large tables must have appropriate filters.

### Medium Issues (Warnings by Default)

1. **SELECT ***: Avoid using SELECT * to prevent excessive data transfer.
2. **Missing GPU Hint**: Multi-table joins should include `/*+ gpu_enable(true) */` hint.
3. **ORDER BY without LIMIT**: Sorting without a LIMIT can be expensive.

### Low Issues (Informational)

1. **Suboptimal JOIN Conditions**: Suggest indexes for join conditions.
2. **Missing LIMIT Clause**: All queries should have a LIMIT clause.

## Automatic Optimizations

The system automatically applies these optimizations:

1. **SELECT * Replacement**: Replaces SELECT * with a specific list of commonly needed columns
2. **Adding LIMIT**: Adds a LIMIT clause to unbounded queries
3. **Adding GPU Hint**: Adds `/*+ gpu_enable(true) */` to multi-table joins 

## Risk Levels

| Level | Description | Default Action |
|-------|-------------|---------------|
| `LOW` | Minor inefficiency or style issue | Log only |
| `MEDIUM` | Potential performance issue | Log warning, continue execution |
| `HIGH` | Likely performance problem | Block execution, can be overridden with warn_only |
| `CRITICAL` | Severe risk of system overload | Block execution, can be overridden with warn_only |

## Metrics Collection

When `collect_metrics=True`, the enhanced connection collects:

- Total queries executed
- Number of optimized queries
- Execution times
- Risk level statistics

Access these metrics with `conn.get_global_metrics()`.

## Using for Advanced Analysis

### Analyze a Single Query

```python
from bt.backtester_stable.BTRUN.dal.heavydb_guardrails import analyze_query

query = "SELECT * FROM nifty_option_chain"
analysis = analyze_query(query)

print(f"Risk level: {analysis.highest_risk}")
print(f"Tables referenced: {analysis.tables}")
print(f"Has date filter: {analysis.has_date_filter}")
```

### Optimize a Query

```python
from bt.backtester_stable.BTRUN.dal.heavydb_guardrails import optimize_query

query = "SELECT * FROM nifty_option_chain WHERE trade_date = '2025-04-01'"
optimized_query = optimize_query(query)

print(f"Original: {query}")
print(f"Optimized: {optimized_query}")
```

### Generate Index Recommendations

```python
from bt.backtester_stable.BTRUN.dal.heavydb_guardrails import recommend_indexes
from bt.backtester_stable.BTRUN.dal.heavydb_connection import get_connection

conn = get_connection()
query = "SELECT * FROM nifty_option_chain WHERE trade_date = '2025-04-01' AND strike = 22000"
index_recommendations = recommend_indexes(conn, query)

for idx in index_recommendations:
    print(idx)  # Prints CREATE INDEX statements
```

## Examples

### Analyzing Historical Data Safely

```python
from bt.backtester_stable.BTRUN.dal.heavydb_connection_enhanced import get_connection

conn = get_connection()

# Safe query - has date filter, specific columns, and LIMIT
safe_query = """
SELECT 
    trade_date, trade_time, expiry_date, strike, 
    ce_close, pe_close, underlying_price
FROM nifty_option_chain 
WHERE trade_date = '2025-04-01' 
  AND trade_time BETWEEN '09:15:00' AND '15:30:00'
  AND strike BETWEEN 22000 AND 23000
LIMIT 10000
"""

result = conn.execute(safe_query)
data = result.fetchall()
print(f"Retrieved {len(data)} rows safely")
```

### Working with Multiple Tables

```python
from bt.backtester_stable.BTRUN.dal.heavydb_connection_enhanced import get_connection

conn = get_connection()

# Safe join query
join_query = """
/*+ gpu_enable(true) */
SELECT 
    a.trade_date, a.trade_time, a.strike, 
    a.ce_close, a.pe_close, a.underlying_price,
    b.volume
FROM nifty_option_chain a
JOIN nifty_greeks b 
  ON a.trade_date = b.trade_date 
 AND a.strike = b.strike
 AND a.trade_time = b.trade_time
WHERE a.trade_date = '2025-04-01'
  AND a.trade_time BETWEEN '09:15:00' AND '15:30:00'
LIMIT 10000
"""

result = conn.execute(join_query)
data = result.fetchall()
print(f"Retrieved {len(data)} rows from join query")
```

## Troubleshooting

### Query Blocked by Guardrails

If your query is blocked:

1. Check the error message for specific issues
2. Add proper date filters (especially `trade_date`)
3. Add LIMIT clauses
4. Replace SELECT * with specific columns
5. If necessary, temporarily override with `warn_only=True`

### Performance Issues

If your queries are still slow:

1. Use `--benchmark` with query_analyzer.py to measure performance
2. Check the query plan with `explain_query(conn, query)`
3. Consider using CTE (WITH clause) for complex operations
4. Use the QUALIFY clause instead of nested subqueries
5. Add appropriate indexes using recommendations from `recommend_indexes()` 