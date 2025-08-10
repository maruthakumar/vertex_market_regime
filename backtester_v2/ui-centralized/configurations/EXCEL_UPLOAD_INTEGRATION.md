# Excel Upload Integration with Existing System

## Overview

The system already implements a sophisticated multi-file Excel configuration structure where the Portfolio file serves as the master configuration that references other Excel files through the `StrategySetting` sheet.

## How It Works

### 1. Portfolio File Structure

Each strategy has a Portfolio Excel file (e.g., `TV_CONFIG_PORTFOLIO_1.0.0.xlsx`) that contains:

- **PortfolioSetting Sheet**: General portfolio parameters
- **StrategySetting Sheet**: References to other Excel files

### 2. StrategySetting Sheet Format

The `StrategySetting` sheet contains these columns:
- `StrategyName`: Name of the strategy configuration
- `StrategyExcelFilePath`: Path to the strategy Excel file
- `Enabled`: Whether this strategy is enabled
- `Priority`: Execution priority
- `AllocationPercent`: Capital allocation percentage

Example:
```
| StrategyName | StrategyExcelFilePath | Enabled | Priority | AllocationPercent |
|--------------|----------------------|---------|----------|-------------------|
| TV_Strategy_1 | TV_CONFIG_STRATEGY_1.0.0.xlsx | YES | 1 | 100 |
```

### 3. File Dependencies by Strategy

Based on our implementation:

- **TV (TradingView)**: 2-6 files
  - Portfolio file (master)
  - Strategy file (required)
  - Optional: Signals, Symbols, Filters, Execution files

- **TBS**: 2 files
  - Portfolio file
  - Strategy file

- **POS**: 3 files
  - Portfolio file
  - Strategy file
  - Adjustment/Greeks file

- **OI**: 2 files
  - Portfolio file
  - Strategy file

- **ORB**: 2 files
  - Portfolio file
  - Strategy file

- **ML Indicator**: 3 files
  - Portfolio file
  - Strategy file
  - Indicators file

- **Market Regime**: 4 files
  - Portfolio file
  - Strategy file
  - Regime file
  - Optimization file

## Client-Side Implementation

The enhanced UI (`integrate_existing_excel_upload.js`) provides:

1. **Smart Upload Flow**:
   - User uploads Portfolio file first
   - System reads `StrategySetting` sheet
   - Dynamically shows required file upload fields
   - Validates completeness before allowing backtest

2. **File Validation**:
   - Checks file names match expected patterns
   - Validates all required files are uploaded
   - Shows progress and status

3. **Integration Points**:
   ```javascript
   // When strategy is selected
   updateFileUploadSection()
   
   // When portfolio file is uploaded
   handlePortfolioUpload(file)
   
   // Extract required files from StrategySetting
   extractStrategySettings(excelData)
   
   // Show dynamic upload fields
   showRequiredFiles(strategySettings)
   ```

## Backend Processing

The backend should:

1. **Receive Portfolio File**:
   ```python
   def process_portfolio_file(file):
       # Read StrategySetting sheet
       strategy_settings = read_strategy_settings(file)
       
       # Return required files list
       return {
           'required_files': [
               {
                   'name': setting['StrategyName'],
                   'path': setting['StrategyExcelFilePath'],
                   'enabled': setting['Enabled']
               }
               for setting in strategy_settings
           ]
       }
   ```

2. **Validate File Set**:
   ```python
   def validate_file_set(portfolio_file, additional_files):
       # Get expected files from portfolio
       expected = extract_expected_files(portfolio_file)
       
       # Check all required files are present
       missing = []
       for expected_file in expected:
           if expected_file['enabled'] and not find_file(expected_file['path'], additional_files):
               missing.append(expected_file['path'])
       
       return {
           'valid': len(missing) == 0,
           'missing_files': missing
       }
   ```

3. **Process Configuration**:
   ```python
   def process_configuration(strategy_type, files):
       # Load portfolio configuration
       portfolio_config = load_excel(files['portfolio'])
       
       # Load referenced configurations
       strategy_configs = {}
       for ref in portfolio_config['StrategySetting']:
           if ref['Enabled']:
               config_file = find_file(ref['StrategyExcelFilePath'], files)
               strategy_configs[ref['StrategyName']] = load_excel(config_file)
       
       # Merge and validate complete configuration
       return merge_configurations(portfolio_config, strategy_configs)
   ```

## API Endpoints

### 1. Scan Portfolio File
```http
POST /api/v1/config/scan-portfolio
Content-Type: multipart/form-data

file: <portfolio.xlsx>
```

Response:
```json
{
    "strategy_type": "tv",
    "required_files": [
        {
            "name": "TV_Strategy_1",
            "path": "TV_CONFIG_STRATEGY_1.0.0.xlsx",
            "enabled": true,
            "required": true
        }
    ]
}
```

### 2. Validate File Set
```http
POST /api/v1/config/validate
Content-Type: multipart/form-data

portfolio: <portfolio.xlsx>
file1: <strategy.xlsx>
file2: <signals.xlsx>
```

Response:
```json
{
    "valid": true,
    "files_received": 3,
    "files_expected": 3,
    "validation_details": {
        "portfolio": "valid",
        "strategy": "valid",
        "signals": "valid"
    }
}
```

### 3. Process and Start Backtest
```http
POST /api/v1/backtest/start
Content-Type: multipart/form-data

strategy_type: "tv"
portfolio: <portfolio.xlsx>
file1: <strategy.xlsx>
file2: <signals.xlsx>
start_date: "2024-01-01"
end_date: "2024-12-31"
```

## Integration with index_enterprise.html

Add this script tag to load the integration:
```html
<!-- After existing scripts -->
<script src="/static/js/integrate_existing_excel_upload.js"></script>
```

The integration automatically:
- Hooks into the existing strategy selector
- Updates the file upload tab content
- Manages the upload workflow
- Validates file completeness

## Benefits

1. **Leverages Existing Structure**: Works with current Excel format
2. **Smart File Detection**: Automatically knows what files are needed
3. **User-Friendly**: Clear workflow with validation
4. **Extensible**: Easy to add new strategies or file types
5. **Maintains Compatibility**: No changes to existing Excel files needed

## Testing

Test the integration with:
```javascript
// Check if integration is loaded
console.log(window.existingExcelUpload);

// Manually trigger file section update
window.existingExcelUpload.updateFileUploadSection();

// Check uploaded files
console.log(window.existingExcelUpload.uploadedFiles);
console.log(window.existingExcelUpload.requiredFiles);
```