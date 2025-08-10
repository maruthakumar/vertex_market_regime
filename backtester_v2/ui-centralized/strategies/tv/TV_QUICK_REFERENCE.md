# TV Strategy Quick Reference Guide

## Quick Start

```bash
# Run TV strategy with 6-file configuration
python processor.py --config-dir /path/to/tv/configs

# Debug signal file
python debug_signals.py --signal-file TV_CONFIG_SIGNALS_1.0.0.xlsx

# Validate configuration
python validate_config.py --config-dir /path/to/tv/configs
```

## 6-File Configuration Structure

1. **TV_CONFIG_MASTER_*.xlsx** - Main configuration
   - Sheet: `Setting`
   - Key fields: StartDate, EndDate, SignalFilePath, Portfolio paths

2. **TV_CONFIG_SIGNALS_*.xlsx** - Trading signals
   - Sheet: `List of trades`
   - Columns: Trade #, Type, Date/Time, Contracts

3. **TV_CONFIG_PORTFOLIO_LONG_*.xlsx** - Long portfolio
   - Sheet: `PortfolioSetting`
   - Key fields: Capital, MaxRisk, MaxPositions

4. **TV_CONFIG_PORTFOLIO_SHORT_*.xlsx** - Short portfolio
   - Sheet: `PortfolioSetting`
   - Key fields: Capital, MaxRisk, MaxPositions

5. **TV_CONFIG_PORTFOLIO_MANUAL_*.xlsx** - Manual trades
   - Sheet: `PortfolioSetting`
   - Key fields: Capital, Manual trade settings

6. **TV_CONFIG_STRATEGY_*.xlsx** - TBS strategy
   - Sheets: `GeneralParameter`, `LegParameter`
   - Defines: Legs, strikes, expiry

## Signal Types

- `Entry Long` - Open long position
- `Exit Long` - Close long position
- `Entry Short` - Open short position
- `Exit Short` - Close short position

## Common Commands

### Testing
```bash
# Run unit tests
python test_tv_unit.py

# Run workflow tests (no HeavyDB)
python run_workflow_tests.py

# Run integration tests (with HeavyDB)
python run_integration_tests.py

# Run end-to-end tests
python test_e2e_complete.py
```

### Debugging
```bash
# Debug signals
python debug_signals.py --signal-file signals.xlsx --output-json analysis.json

# Validate configuration
python validate_config.py --config-dir ./configs --output-json validation.json

# Generate golden format
python test_golden_format_direct.py
```

### Conversion
```bash
# Convert Excel to YAML
python excel_to_yaml_converter.py --hierarchy-dir ./configs --output config.yaml

# Export unified configuration
python tv_unified_config.py
```

## Key Classes

### TVParser
```python
parser = TVParser()
tv_config = parser.parse_tv_settings("TV_CONFIG_MASTER.xlsx")
signals = parser.parse_signals("TV_CONFIG_SIGNALS.xlsx", "%Y%m%d %H%M%S")
```

### SignalProcessor
```python
processor = SignalProcessor()
processed_signals = processor.process_signals(signals, tv_config)
```

### TVQueryBuilder
```python
builder = TVQueryBuilder()
query = builder.build_query(signal, config)
```

### TVProcessor
```python
processor = TVProcessor()
results = processor.process(config, heavydb_connection)
```

## Configuration Parameters

### TV Master Settings
- `SignalDateFormat`: Format for parsing signal dates (e.g., "%Y%m%d %H%M%S")
- `IntradaySqOffApplicable`: YES/NO for intraday square-off
- `IntradayExitTime`: Time for intraday exit (e.g., "15:30:00")
- `SlippagePercent`: Slippage percentage (e.g., 0.1)
- `UseDbExitTiming`: YES/NO for database-based exit timing
- `ExitSearchInterval`: Minutes to search for exit (e.g., 5)

### Portfolio Settings
- `Capital`: Portfolio capital amount
- `MaxRisk`: Maximum risk percentage
- `MaxPositions`: Maximum concurrent positions
- `RiskPerTrade`: Risk per trade percentage
- `UseKellyCriterion`: YES/NO for Kelly criterion

### TBS Strategy Parameters
- `StrategyName`: Name of the strategy
- `Index`: Index to trade (e.g., "NIFTY")
- `DTE`: Days to expiry
- `StrikeMethod`: Strike selection method (e.g., "ATM")
- `Lots`: Number of lots per leg

## Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| File Parsing | < 500ms | ~8ms |
| Signal Processing | < 200ms | ~0.01ms |
| Total Workflow | < 3s | ~400ms |
| Memory Usage | < 500MB | ~100MB |

## Error Codes

- **TV001**: Missing configuration file
- **TV002**: Invalid date format
- **TV003**: Unpaired signals
- **TV004**: HeavyDB connection failed
- **TV005**: Missing required columns
- **TV006**: Invalid portfolio allocation
- **TV007**: No enabled configuration
- **TV008**: Signal date out of range

## Troubleshooting

### Signal not processed
1. Check signal file has correct columns
2. Verify date format matches configuration
3. Ensure entry/exit pairs exist

### Configuration not loading
1. Verify all 6 files exist
2. Check sheet names are correct
3. Ensure at least one config is enabled

### HeavyDB errors
1. Check connection parameters
2. Verify table exists
3. Check data for signal dates

## Golden Format Sheets

1. **Portfolio Parameters** - Portfolio configuration
2. **General Parameters** - Strategy parameters
3. **Leg Parameters** - Option leg definitions
4. **Trades** - All executed trades
5. **Results** - Trade-wise P&L
6. **TV Setting** - TV configuration
7. **Signals** - Original signals

## Environment Variables

```bash
# HeavyDB Connection
export HEAVYDB_HOST=localhost
export HEAVYDB_PORT=6274
export HEAVYDB_USER=admin
export HEAVYDB_PASSWORD=HyperInteractive
export HEAVYDB_DATABASE=heavyai

# TV Strategy
export TV_CONFIG_DIR=/path/to/configs
export TV_OUTPUT_DIR=/path/to/output
export TV_LOG_LEVEL=INFO
```

## Tips & Best Practices

1. **Always validate configuration before running**
   ```bash
   python validate_config.py --config-dir ./configs
   ```

2. **Debug signals before processing**
   ```bash
   python debug_signals.py --signal-file signals.xlsx
   ```

3. **Use parallel processing for multiple signal files**
   ```python
   processor = TVParallelProcessor(max_workers=4)
   ```

4. **Monitor performance metrics**
   - Check execution time < 3 seconds
   - Monitor memory usage < 500MB

5. **Handle errors gracefully**
   - Log all errors with context
   - Implement retry logic for transient failures

---

*For detailed documentation, see TV_STRATEGY_DOCUMENTATION.md*