# Market Regime Strategy - Clean Architecture

## Overview

The Market Regime Strategy system has been refactored to follow a clean, modular architecture that eliminates duplication, provides clear separation of concerns, and enables easy maintenance and extension.

## Directory Structure

```
strategies/market_regime/
├── core/                      # Core business logic
│   ├── __init__.py
│   ├── engine.py             # Main orchestration engine
│   ├── analyzer.py           # Core analysis logic
│   ├── regime_classifier.py  # 18-regime classification system
│   └── regime_detector.py    # Regime detection logic
│
├── indicators/               # Technical indicators
│   ├── __init__.py
│   ├── atr_analysis.py      # ATR (Average True Range) analysis
│   ├── iv_analysis.py       # IV surface, skew, percentile analysis
│   ├── greek_sentiment.py   # Greek sentiment analysis (Delta, Gamma, Theta, Vega)
│   ├── oi_price_action.py   # OI with price action analysis
│   └── straddle_analysis.py # Triple straddle strategies
│
├── adaptive/                 # Adaptive ML system (already migrated)
│   ├── __init__.py
│   ├── core/                # Core adaptive logic
│   ├── intelligence/        # ML models and intelligence
│   ├── optimization/        # Performance optimization
│   └── validation/          # Adaptive validation
│
├── data/                    # Data layer
│   ├── __init__.py
│   ├── loaders.py          # Data loading utilities
│   ├── processors.py       # Data preprocessing
│   ├── cache_manager.py    # Caching strategy
│   └── heavydb_adapter.py  # HeavyDB integration
│
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── config_manager.py   # Centralized config management
│   ├── excel_parser.py     # Excel configuration parsing
│   └── schemas/            # Configuration schemas
│       ├── regime_config.json
│       └── indicator_config.json
│
├── integration/            # External integrations
│   ├── __init__.py
│   ├── api_routes.py      # API endpoints
│   ├── ui_bridge.py       # UI integration layer
│   └── consolidator.py    # Strategy consolidator integration
│
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── constants.py       # System constants
│   ├── helpers.py         # Helper functions
│   └── validators.py      # Input validation
│
├── docs/                   # Documentation
│   └── ...                # Migrated documentation files
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
│
├── enhanced_modules/       # Enhanced implementations (existing)
├── comprehensive_modules/  # Comprehensive implementations (existing)
└── csv_handlers/          # CSV utilities (existing)
```

## Module Organization

### 1. Core Modules (`core/`)

The core directory contains the essential business logic:

- **engine.py**: Main orchestration engine that coordinates all components
- **analyzer.py**: Core analysis logic that processes market data
- **regime_classifier.py**: Implements the 18-regime classification system (3×3×2 matrix)
- **regime_detector.py**: Detects market regimes based on indicators

### 2. Indicators (`indicators/`)

Each indicator module is self-contained and follows a standard interface:

```python
class IndicatorBase:
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicator values"""
        pass
    
    def get_signals(self, values: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading signals"""
        pass
```

### 3. Adaptive System (`adaptive/`)

The adaptive system provides ML-based continuous learning:

- Online learning capabilities
- Performance tracking
- Dynamic parameter optimization
- Regime transition prediction

### 4. Data Layer (`data/`)

Handles all data operations:

- **loaders.py**: Load data from various sources (HeavyDB, CSV, etc.)
- **processors.py**: Preprocess and normalize data
- **cache_manager.py**: Implement caching for performance
- **heavydb_adapter.py**: HeavyDB-specific operations

### 5. Configuration (`config/`)

Centralized configuration management:

- Single source of truth for all parameters
- Excel-based configuration support
- JSON schema validation
- Dynamic configuration updates

### 6. Integration (`integration/`)

External system integrations:

- REST API endpoints
- UI websocket communication
- Strategy consolidator integration
- Order management system hooks

## Parameter Mapping

Based on the 181 unique parameters from the Excel configuration:

### Core Parameters
- **Regime Classification**: 18 regimes (Volatility × Trend × Structure)
- **Time Intervals**: 3min, 5min, 15min, 30min, 60min
- **DTE Ranges**: 0-2, 3-7, 8-15, 16-30, 31-45, 46+

### Indicator Parameters
1. **ATR Analysis** (3 parameters)
   - Periods: 14, 21, 50
   - Multipliers for bands
   - Breakout thresholds

2. **IV Analysis** (12 parameters)
   - Surface parameters
   - Skew calculations
   - Percentile ranges
   - Term structure

3. **Greek Sentiment** (15 parameters)
   - Delta, Gamma, Theta, Vega weights
   - DTE-specific adjustments
   - Cross-Greek correlations

4. **OI Price Action** (8 parameters)
   - Multi-timeframe settings
   - Divergence thresholds
   - Institutional detection

5. **Straddle Analysis** (20 parameters)
   - Entry/exit conditions
   - Position sizing
   - Risk management
   - Correlation matrix (6×6)

## Design Principles

### 1. Single Responsibility
Each module has one clear purpose and responsibility.

### 2. Dependency Inversion
High-level modules don't depend on low-level modules. Both depend on abstractions.

### 3. Open/Closed Principle
Modules are open for extension but closed for modification.

### 4. Interface Segregation
Modules expose only the interfaces they need.

### 5. Don't Repeat Yourself (DRY)
No duplicate code or functionality across modules.

## Usage Example

```python
from backtester_v2.strategies.market_regime.core.engine import MarketRegimeEngine
from backtester_v2.strategies.market_regime.config.config_manager import ConfigManager

# Initialize configuration
config = ConfigManager.load_from_excel("path/to/config.xlsx")

# Create engine
engine = MarketRegimeEngine(config)

# Analyze market data
result = engine.analyze(market_data)

# Get regime classification
regime = result.regime  # e.g., "HIGH_VOLATILITY_BULLISH_TRENDING"
signals = result.signals  # Trading signals from all indicators
```

## Migration Notes

1. **Import Updates**: All imports have been updated from `backtester_v2.market_regime` to `backtester_v2.strategies.market_regime`

2. **Backward Compatibility**: The existing `enhanced_modules/` and `comprehensive_modules/` directories are preserved for backward compatibility during transition.

3. **Gradual Migration**: Modules will be gradually migrated from `enhanced_modules/` and `comprehensive_modules/` to the new structure.

## Testing Strategy

1. **Unit Tests**: Test each module in isolation
2. **Integration Tests**: Test module interactions
3. **E2E Tests**: Test complete workflows with real data
4. **Performance Tests**: Ensure no performance degradation

## Future Enhancements

1. **Plugin Architecture**: Allow dynamic loading of new indicators
2. **Event-Driven Architecture**: Implement pub/sub for real-time updates
3. **Microservices**: Split into separate services for scalability
4. **GraphQL API**: Modern API for flexible data queries