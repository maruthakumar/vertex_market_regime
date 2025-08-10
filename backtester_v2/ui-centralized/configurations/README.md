# Unified Configuration Management System

## Overview

The Unified Configuration Management System provides centralized, standardized configuration management for all strategy types in Backtester V2. It replaces the scattered Excel files with a robust, scalable solution that supports:

- 🎯 **7 Strategy Types**: TBS, TV, ORB, OI, ML, POS, Market Regime
- 📊 **700+ Parameters** per strategy
- 🔄 **Hot-reloading** without system restart
- 📝 **Version Control** with full history
- 🔐 **Validation** at multiple levels
- 🌐 **API Access** for programmatic control
- 📁 **Multiple Formats**: Excel, JSON, YAML

## Architecture

```
configurations/
├── core/               # Core configuration classes
├── parsers/           # File parsers (Excel, JSON, etc.)
├── validators/        # Validation framework
├── converters/        # Format converters
├── schemas/           # JSON schemas for validation
├── templates/         # Configuration templates
├── api/              # REST/WebSocket APIs
├── storage/          # Storage backends
└── monitoring/       # Change tracking and monitoring
```

## Quick Start

### 1. Import and Initialize

```python
from configurations import ConfigurationManager

# Get the singleton instance
config_manager = ConfigurationManager()
```

### 2. Load a Configuration

```python
# Load from Excel
config = config_manager.load_configuration(
    strategy_type="tbs",
    file_path="/path/to/tbs_config.xlsx",
    config_name="my_tbs_strategy"
)

# Load from JSON
config = config_manager.load_configuration(
    strategy_type="tv",
    file_path="/path/to/tv_config.json"
)
```

### 3. Access Configuration Values

```python
# Get single value
capital = config.get("portfolio_settings.capital")

# Get nested value
strike_method = config.get("strategy_parameters.strike_selection_method")

# Get with default
reentry_enabled = config.get("enhancement_parameters.reentry_enabled", False)
```

### 4. Update Configuration

```python
# Update single value
config.set("portfolio_settings.capital", 200000)

# Update multiple values
config.update({
    "strategy_parameters.stop_loss": 40,
    "strategy_parameters.target": 60
})

# Save changes
config_manager.save_configuration(config)
```

### 5. Validate Configuration

```python
# Validate and get results
validation_result = config_manager.validate_configuration(config)

if validation_result["valid"]:
    print("Configuration is valid!")
else:
    print("Validation errors:", validation_result["errors"])
```

## Strategy-Specific Configurations

### TBS (Trade Builder Strategy)

```python
from configurations.strategies import TBSConfiguration

tbs_config = TBSConfiguration("my_tbs_strategy")

# Access TBS-specific methods
strike_config = tbs_config.get_strike_selection_config()
risk_limits = tbs_config.get_risk_limits()
is_multileg = tbs_config.is_multileg_strategy()
```

### Market Regime

```python
from configurations.strategies import MarketRegimeConfiguration

mr_config = MarketRegimeConfiguration("market_regime_v1")

# Access Market Regime specific settings
regime_count = mr_config.get("regime_settings.regime_count", 18)
rolling_windows = mr_config.get("analysis_settings.rolling_windows", [3, 5, 10, 15])
```

## Advanced Features

### 1. Configuration Watching

```python
# Watch for changes
def on_config_change(strategy_type, config_name, event):
    print(f"Configuration {config_name} was {event}")

watch_id = config_manager.watch_configuration("tbs", "my_strategy", on_config_change)

# Stop watching
config_manager.unwatch_configuration(watch_id)
```

### 2. Configuration Cloning

```python
# Clone existing configuration
cloned = config_manager.clone_configuration(
    strategy_type="tbs",
    config_name="original_strategy",
    new_name="cloned_strategy"
)
```

### 3. Configuration Merging

```python
# Merge two configurations
merged = config_manager.merge_configurations(
    config1=base_config,
    config2=override_config,
    merged_name="merged_strategy",
    strategy="override"  # or "keep", "error"
)
```

### 4. Export/Import

```python
# Export to different formats
path = config_manager.export_configuration(config, format="yaml")
path = config_manager.export_configuration(config, format="excel")

# Import from file
imported = config_manager.import_configuration(
    file_path="/path/to/config.xlsx",
    strategy_type="tbs"
)
```

### 5. Bulk Operations

```python
# List all configurations
all_configs = config_manager.list_configurations()
# {"tbs": ["strategy1", "strategy2"], "tv": ["tv_strat1"], ...}

# Reload all configurations asynchronously
import asyncio
results = asyncio.run(config_manager.reload_all_async())
```

## API Usage

### REST API Endpoints

```bash
# Create configuration
POST /api/v1/configurations/tbs
Content-Type: multipart/form-data
Body: Excel file

# Get configuration
GET /api/v1/configurations/tbs/my_strategy

# Update configuration
PUT /api/v1/configurations/tbs/my_strategy
Content-Type: application/json
Body: {"portfolio_settings": {"capital": 200000}}

# Delete configuration
DELETE /api/v1/configurations/tbs/my_strategy

# Validate configuration
POST /api/v1/configurations/tbs/validate
Body: Configuration data

# Get template
GET /api/v1/configurations/tbs/template?complexity=advanced
```

### WebSocket Events

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/configurations');

// Listen for events
ws.on('message', (data) => {
    const event = JSON.parse(data);
    switch(event.type) {
        case 'config.created':
        case 'config.updated':
        case 'config.deleted':
            console.log(`Configuration ${event.config_name} was ${event.type}`);
            break;
    }
});
```

## Validation

### Multi-Level Validation

1. **Schema Validation**: JSON Schema validation for structure
2. **Business Rules**: Strategy-specific business logic
3. **Range Validation**: Parameter value ranges
4. **Dependency Validation**: Cross-parameter dependencies
5. **Custom Validation**: User-defined validation rules

### Example Validation

```python
# Add custom validator
def validate_capital_allocation(config):
    total_capital = config.get("portfolio_settings.capital")
    capital_per_set = config.get("portfolio_settings.capital_per_set")
    max_trades = config.get("portfolio_settings.max_trades")
    
    if capital_per_set * max_trades > total_capital:
        return False, "Total allocation exceeds available capital"
    
    return True, None

# Register validator
config.add_validator(validate_capital_allocation)
```

## Migration from Old System

### 1. Analyze Existing Sheets

```python
from configurations.migration import analyze_input_sheets

analysis = analyze_input_sheets("/path/to/input_sheets")
print(f"Found {analysis['total_files']} Excel files")
print(f"Strategies: {analysis['strategies']}")
```

### 2. Migrate Configurations

```python
from configurations.migration import migrate_excel_configs

results = migrate_excel_configs(
    source_dir="/path/to/input_sheets",
    target_dir="/path/to/configurations/data"
)

print(f"Migrated {results['success']} configurations")
print(f"Failed: {results['failed']}")
```

### 3. Validate Migrated Data

```python
# Validate all migrated configurations
for strategy_type, config_names in config_manager.list_configurations().items():
    for config_name in config_names:
        config = config_manager.get_configuration(strategy_type, config_name)
        result = config_manager.validate_configuration(config)
        
        if not result["valid"]:
            print(f"Invalid: {strategy_type}/{config_name}")
            print(f"Errors: {result['errors']}")
```

## Performance Considerations

### Caching

The system implements intelligent caching:
- Parsed configurations are cached in memory
- File system changes trigger cache invalidation
- Redis backend available for distributed caching

### Async Operations

For bulk operations, use async methods:

```python
# Async configuration loading
async def load_multiple_configs(config_list):
    tasks = []
    for strategy_type, file_path in config_list:
        task = config_manager.load_configuration_async(strategy_type, file_path)
        tasks.append(task)
    
    return await asyncio.gather(*tasks)
```

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   ```python
   if not config_manager.get_configuration("tbs", "my_strategy"):
       print("Configuration not found. Available:", 
             config_manager.list_configurations("tbs"))
   ```

2. **Validation Errors**
   ```python
   errors = config.get_validation_errors()
   for field, messages in errors.items():
       print(f"{field}: {', '.join(messages)}")
   ```

3. **Import Failures**
   ```python
   try:
       config = config_manager.load_configuration("tbs", "config.xlsx")
   except ParsingError as e:
       print(f"Parsing failed: {e}")
       print(f"File: {e.file_path}")
   ```

## Best Practices

1. **Always Validate**: Validate configurations before using in production
2. **Use Templates**: Start with templates for consistency
3. **Version Control**: Track configuration changes
4. **Document Changes**: Use meaningful commit messages
5. **Test Configurations**: Test in sandbox before production
6. **Monitor Changes**: Set up watchers for critical configurations
7. **Regular Backups**: Export configurations regularly

## Extension

### Adding New Strategy Type

1. Create configuration class:
```python
from configurations.core import BaseConfiguration

class MyStrategyConfiguration(BaseConfiguration):
    def validate(self) -> bool:
        # Implementation
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        # Return JSON schema
        pass
```

2. Register with manager:
```python
from configurations import ConfigurationManager

manager = ConfigurationManager()
manager.register_configuration_class("my_strategy", MyStrategyConfiguration)
```

3. Create parser if needed:
```python
from configurations.parsers import BaseParser

class MyStrategyParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        # Implementation
        pass
```

## Support

For issues or questions:
1. Check the logs in `configurations/logs/`
2. Run validation diagnostics
3. Review the API documentation
4. Contact the development team

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Maintained By**: Backtester Development Team