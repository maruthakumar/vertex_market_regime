# BTRUN GPU Acceleration

This package provides GPU-accelerated backtesting utilities for the BTRUN framework. It leverages NVIDIA GPUs through the [RAPIDS](https://rapids.ai/) ecosystem (specifically cuDF) to accelerate DataFrame operations for large-scale backtests.

## Features

- Seamless CPU/GPU switching with automatic fallback to pandas
- GPU-accelerated statistics calculation
- GPU-accelerated data transformations
- Backward compatibility with legacy BTRUN code
- Optimized I/O operations
- High-level runtime functions for backtesting

## Requirements

- Python 3.8+
- pandas
- numpy
- RAPIDS cuDF (optional, for GPU acceleration)
- CUDA Toolkit 11.0+ (for RAPIDS)
- An NVIDIA GPU with compute capability 5.0+ (for GPU acceleration)

## Installation

The package comes pre-installed as part of the BTRUN framework. To enable GPU acceleration, ensure you have the RAPIDS ecosystem installed:

```bash
# Install RAPIDS (adjust version as needed)
conda install -c rapidsai -c conda-forge cudf=23.04 python=3.10 cudatoolkit=11.8
```

## Usage

### Basic Usage

```python
from BTRUN import stats, gpu_helpers

# Create a DataFrame with trade data
trades_df = pd.DataFrame({
    'entryDate': ['2023-01-01', '2023-01-02'],
    'bookedPnL': [100, -50]
})

# Calculate statistics with GPU acceleration if available
backtest_stats = stats.get_backtest_stats(trades_df, initial_capital=100000)
```

### Controlling GPU Usage

You can enable or disable GPU acceleration in several ways:

1. Environment variable:
   ```bash
   export BT_USE_GPU=true  # or false
   ```

2. Programmatically:
   ```python
   from BTRUN import gpu_helpers
   
   # Force CPU mode
   gpu_helpers.force_cpu_mode()
   
   # Force GPU mode (if available)
   gpu_helpers.force_gpu_mode()
   
   # Temporary CPU mode
   with gpu_helpers.cpu_only():
       # Code here will run on CPU even if GPU is available
       df = process_data(data)
   ```

### Key Modules

- `gpu_helpers`: Utilities for GPU/CPU operations
- `stats`: GPU-accelerated backtest statistics calculation
- `builders`: GPU-accelerated data transformation
- `io`: GPU-aware file operations
- `runtime`: High-level functions for running backtests
- `util_legacy_shim`: Backward compatibility with legacy BTRUN

## Example

```python
from BTRUN import runtime

# Run a backtest
bt_response = runtime.run_backtest(bt_params={
    'strategy': 'example',
    'start_date': '230101',
    'end_date': '230131',
    # other parameters...
})

# Process results with GPU acceleration
order_df, metrics_df, trans_dfs, day_stats, month_stats, daily_max_pl_df = runtime.process_backtest_results(
    bt_response=bt_response,
    slippage_percent=0.1,
    initial_capital=100000
)

# Save results
runtime.save_backtest_results(
    output_base_path='output/example_backtest',
    metrics_df=metrics_df,
    transaction_dfs=trans_dfs,
    day_stats=day_stats,
    month_stats=month_stats,
    margin_stats={"portfolio": pd.DataFrame()},
    daily_max_pl_df=daily_max_pl_df
)
```

## Performance Tips

1. For large backtests, ensure your GPU has sufficient memory
2. Use `gpu_helpers.get_gpu_memory_info()` to monitor GPU memory usage
3. For very large datasets, consider processing in chunks
4. Use `gpu_helpers.cpu_only()` for operations that are faster on CPU
5. Set environment variable `BT_USE_GPU=false` to disable GPU acceleration if needed

## Compatibility

This package is designed to be backward compatible with existing BTRUN code. Legacy code that uses `BTRUN.Util` functions will automatically use the GPU-accelerated versions via the compatibility layer in `util_legacy_shim.py`.

## For Developers

When extending this package:

1. Ensure all functions have appropriate docstrings
2. Use the `gh.ensure_cudf()` and `gh.to_pandas()` functions for DataFrame conversions
3. Use type hints that include both pandas and cuDF types
4. Add appropriate error handling and fallback mechanisms
5. Test with both CPU and GPU execution paths

# HeavyDB Backtester Toolkit

A GPU-accelerated backtesting system for financial strategies.

## Overview

The HeavyDB Backtester Toolkit provides a robust framework for backtesting financial strategies using HeavyDB's GPU-accelerated query processing. The system is designed to efficiently process large datasets of financial data, enabling fast and accurate backtesting of complex trading strategies.

## Latest Updates

### Phase 1.D: HeavyDB Guard-rails

We have implemented a comprehensive set of guard-rails for HeavyDB queries that:

- **Analyze Queries**: Automatically detect risky patterns like missing WHERE clauses, SELECT *, etc.
- **Optimize Queries**: Automatically improve queries by adding limits, replacing SELECT *, etc.
- **Performance Tracking**: Measure and log query execution times
- **Risk Classification**: Categorize queries by risk level (LOW, MEDIUM, HIGH, CRITICAL)
- **Enforce Best Practices**: Block execution of dangerous queries to protect system resources

Documentation for the guardrails implementation can be found in `docs/heavydb_guardrails.md`.

## Components

- **DAL**: Data Access Layer for efficient data retrieval and processing
- **Models**: Pydantic models for strict type validation
- **Excel Parser**: Parses Excel input files into validated models
- **HeavyDB Guardrails**: Ensures safe and efficient SQL queries
- **Backtester**: Core backtesting engine

## Getting Started

1. Ensure you have a working HeavyDB installation
2. Set up environment variables:
   ```
   export HEAVYDB_HOST=127.0.0.1
   export HEAVYDB_PORT=6274
   export HEAVYDB_USER=admin
   export HEAVYDB_PASSWORD=HyperInteractive
   export HEAVYDB_DATABASE=heavyai
   ```
3. Run a basic backtest:
   ```
   python BTRunPortfolio_GPU.py -i input_portfolio.xlsx
   ```

## HeavyDB Guardrails Usage

To use the HeavyDB guardrails in your code:

```python
from bt.backtester_stable.BTRUN.dal.heavydb_connection_enhanced import get_connection

# Get a connection with guardrails enabled
conn = get_connection()

# Execute a query - it will be analyzed and optimized automatically
result = conn.execute("SELECT * FROM nifty_option_chain WHERE trade_date = '2025-04-01'")
```

You can customize the guardrails behavior:

```python
# Only warn about issues, don't block execution
conn = get_connection(warn_only=True)

# Disable guardrails completely
conn = get_connection(enforce_guardrails=False)

# Enable metrics collection for performance tracking
conn = get_connection(collect_metrics=True)
metrics = conn.get_global_metrics()
```

## Development

### Testing

To run the unit tests:

```
python -m unittest discover -s bt/backtester_stable/BTRUN/tests
```

### Documentation

API documentation is available in the `docs/` directory. 