# Excel to Backend Parameter Mapping - Indicator Strategy

**Strategy Type**: ML Indicator Strategy  
**Document Version**: 1.0  
**Date**: 2025-01-28  
**Purpose**: Comprehensive mapping of Excel configuration parameters to backend implementation for the ML Indicator trading strategy

## 📁 File Structure Overview

The ML Indicator strategy uses a sophisticated configuration system with **3 Excel files** containing **30+ sheets** total:

```
/configurations/data/prod/ml/
├── ML_CONFIG_INDICATORS_1.0.0.xlsx    # Technical indicators configuration
├── ML_CONFIG_PORTFOLIO_1.0.0.xlsx     # Portfolio and risk management settings  
└── ML_CONFIG_STRATEGY_1.0.0.xlsx      # ML model and strategy parameters
```

### Additional Template Files
```
/configurations/data/prod/indicator/
├── advanced_6sheet_template.xlsx      # 6-sheet advanced indicator template
└── traditional_2sheet_template.xlsx   # 2-sheet traditional template
```

## 🎯 Strategy Overview

The ML Indicator strategy combines traditional technical analysis with machine learning to create predictive trading signals. It supports:

- **197+ parameters** across all Excel configurations
- **Technical Indicators**: 40+ TA-Lib indicators with custom parameters
- **Smart Money Concepts (SMC)**: Advanced market structure analysis
- **Machine Learning Models**: 7 model types (XGBoost, LightGBM, CatBoost, etc.)
- **Multi-timeframe Analysis**: 1m to 1d timeframes
- **Signal Generation**: Complex conditional logic with ML predictions

## 📊 Sheet Structure Analysis

### ML_CONFIG_INDICATORS_1.0.0.xlsx (18 sheets)
1. **Indicators** - Main technical indicator configuration
2. **SMC** - Smart Money Concepts settings
3. **Signals** - Signal generation logic
4. **Timeframes** - Multi-timeframe analysis settings
5. **TA-Lib_Trend** - Trend indicators (SMA, EMA, MACD, etc.)
6. **TA-Lib_Momentum** - Momentum indicators (RSI, STOCH, CCI, etc.)
7. **TA-Lib_Volatility** - Volatility indicators (ATR, BBANDS, etc.)
8. **TA-Lib_Volume** - Volume indicators (OBV, MFI, AD, etc.)
9. **TA-Lib_Pattern** - Candlestick pattern recognition
10. **SMC_BOS** - Break of Structure configuration
11. **SMC_CHOCH** - Change of Character settings
12. **SMC_OrderBlock** - Order block detection
13. **SMC_FVG** - Fair Value Gap analysis
14. **SMC_Liquidity** - Liquidity pool identification
15. **Feature_Engineering** - ML feature creation
16. **Signal_Conditions** - Complex signal logic
17. **Backtesting** - Historical testing parameters
18. **Validation** - Model validation settings

### ML_CONFIG_PORTFOLIO_1.0.0.xlsx (8 sheets)
1. **PortfolioSetting** - Basic portfolio configuration
2. **RiskManagement** - Risk control parameters
3. **PositionSizing** - Dynamic position sizing
4. **Execution** - Order execution settings
5. **TimeFrames** - Trading session configuration
6. **WalkForward** - Walk-forward analysis
7. **Performance** - Performance tracking
8. **Reporting** - Output and reporting settings

### ML_CONFIG_STRATEGY_1.0.0.xlsx (5 sheets)
1. **MLModels** - Machine learning model configuration
2. **Features** - Feature selection and engineering
3. **Training** - Model training parameters
4. **Prediction** - Prediction generation settings
5. **Ensemble** - Model ensemble configuration

## 🔧 Backend Implementation Mapping

### Core Parser Location
**File**: `backtester_v2/strategies/ml_indicator/parser.py`  
**Class**: `MLIndicatorParser`  
**Main Method**: `parse_input(portfolio_file, indicator_file, ml_model_file, signal_file)`

### Model Definitions
**File**: `backtester_v2/strategies/ml_indicator/models.py`  
**Key Models**:
- `MLIndicatorStrategyModel` - Main strategy container
- `IndicatorConfig` - Individual indicator settings
- `MLModelConfig` - ML model parameters
- `SignalCondition` - Trading signal logic

## 📋 Detailed Parameter Mapping

### 1. Portfolio Configuration (PortfolioSetting Sheet)

| Excel Parameter | Backend Field | Type | Description | Parser Location |
|----------------|---------------|------|-------------|-----------------|
| PortfolioName | portfolio_name | str | Portfolio identifier | `parse_portfolio_excel()` line 127 |
| StrategyName | strategy_name | str | Strategy identifier | `parse_portfolio_excel()` line 129 |
| StartDate | start_date | date | Backtest start date | `parse_portfolio_excel()` line 131 |
| EndDate | end_date | date | Backtest end date | `parse_portfolio_excel()` line 133 |
| IndexName | index_name | str | NIFTY/BANKNIFTY/SENSEX | `parse_portfolio_excel()` line 135 |
| UnderlyingPriceType | underlying_price_type | str | SPOT/FUTURES | `parse_portfolio_excel()` line 137 |
| TransactionCosts | transaction_costs | float | Brokerage and fees | `parse_portfolio_excel()` line 139 |
| UseWalkForward | use_walk_forward | bool | Enable walk-forward analysis | `parse_portfolio_excel()` line 141 |
| WalkForwardWindow | walk_forward_window | int | Window size for analysis | `parse_portfolio_excel()` line 143 |
| TrackFeatureImportance | track_feature_importance | bool | Monitor feature importance | `parse_portfolio_excel()` line 145 |
| TrackSignalAccuracy | track_signal_accuracy | bool | Monitor signal accuracy | `parse_portfolio_excel()` line 147 |
| SavePredictions | save_predictions | bool | Save ML predictions | `parse_portfolio_excel()` line 149 |

### 2. Technical Indicators (Indicators Sheet)

| Excel Parameter | Backend Field | Type | Description | Constants Reference |
|----------------|---------------|------|-------------|-------------------|
| IndicatorName | indicator_type | IndicatorType | SMA/EMA/RSI/MACD/etc. | `constants.py` INDICATOR_CATEGORIES |
| Enabled | enabled | bool | Enable/disable indicator | `_parse_indicators()` |
| Timeframe | timeframe | Timeframe | 1m/5m/15m/30m/1h/4h/1d | `constants.py` TIMEFRAME_MINUTES |
| Parameters | parameters | Dict | Indicator-specific params | `constants.py` TALIB_PARAMS |
| Weight | weight | float | Signal weighting (0.0-1.0) | `_parse_indicators()` |
| MinConfidence | min_confidence | float | Minimum signal confidence | `_parse_indicators()` |
| MaxLookback | max_lookback | int | Historical data required | `_parse_indicators()` |

#### TA-Lib Indicator Parameters

**Trend Indicators** (`constants.py` lines 60-73):
- **SMA**: timeperiod (default: 20)
- **EMA**: timeperiod (default: 20)  
- **MACD**: fastperiod (12), slowperiod (26), signalperiod (9)
- **BBANDS**: timeperiod (20), nbdevup (2), nbdevdn (2), matype (0)
- **SAR**: acceleration (0.02), maximum (0.2)

**Momentum Indicators**:
- **RSI**: timeperiod (default: 14)
- **STOCH**: fastk_period (5), slowk_period (3), slowd_period (3)
- **ADX**: timeperiod (default: 14)
- **CCI**: timeperiod (default: 20)
- **MFI**: timeperiod (default: 14)
- **WILLR**: timeperiod (default: 14)

**Volatility Indicators**:
- **ATR**: timeperiod (default: 14)
- **NATR**: timeperiod (default: 14)
- **TRANGE**: No parameters

### 3. Smart Money Concepts (SMC Sheet)

| Excel Parameter | Backend Field | Type | Description | Constants Reference |
|----------------|---------------|------|-------------|-------------------|
| BOS_Enabled | bos_enabled | bool | Break of Structure detection | `constants.py` SMC_PARAMS |
| BOS_Lookback | bos_lookback | int | BOS detection period (20) | `constants.py` line 78 |
| BOS_MinSwingStrength | bos_min_swing_strength | int | Minimum swing strength (3) | `constants.py` line 79 |
| BOS_ConfirmationCandles | bos_confirmation_candles | int | Confirmation candles (2) | `constants.py` line 80 |
| CHOCH_Enabled | choch_enabled | bool | Change of Character detection | `constants.py` SMC_PARAMS |
| CHOCH_Lookback | choch_lookback | int | CHOCH detection period (20) | `constants.py` line 83 |
| CHOCH_TrendLookback | choch_trend_lookback | int | Trend analysis period (50) | `constants.py` line 84 |
| CHOCH_MinReversalStrength | choch_min_reversal_strength | float | Minimum reversal (0.005) | `constants.py` line 85 |
| OrderBlock_Enabled | order_block_enabled | bool | Order block detection | `constants.py` SMC_PARAMS |
| OrderBlock_Lookback | order_block_lookback | int | Order block period (50) | `constants.py` line 87 |
| OrderBlock_MinVolumeMultiplier | order_block_min_volume_multiplier | float | Volume threshold (1.5) | `constants.py` line 89 |
| OrderBlock_MaxRetestCount | order_block_max_retest_count | int | Maximum retests (3) | `constants.py` line 90 |
| FVG_Enabled | fvg_enabled | bool | Fair Value Gap detection | `constants.py` SMC_PARAMS |
| FVG_MinGapSize | fvg_min_gap_size | float | Minimum gap size (0.001) | `constants.py` line 93 |
| FVG_MaxFillRatio | fvg_max_fill_ratio | float | Maximum fill ratio (0.5) | `constants.py` line 94 |
| FVG_Lookback | fvg_lookback | int | FVG detection period (20) | `constants.py` line 95 |
| LiquidityPool_Enabled | liquidity_pool_enabled | bool | Liquidity pool detection | `constants.py` SMC_PARAMS |
| LiquidityPool_Lookback | liquidity_pool_lookback | int | Liquidity period (100) | `constants.py` line 98 |
| LiquidityPool_ClusterThreshold | liquidity_pool_cluster_threshold | float | Cluster threshold (0.002) | `constants.py` line 99 |
| LiquidityPool_MinTouches | liquidity_pool_min_touches | int | Minimum touches (2) | `constants.py` line 100 |

### 4. Machine Learning Models (MLModels Sheet)

| Excel Parameter | Backend Field | Type | Description | Constants Reference |
|----------------|---------------|------|-------------|-------------------|
| ModelType | model_type | MLModelType | XGBOOST/LIGHTGBM/CATBOOST/etc. | `constants.py` ML_MODELS |
| Enabled | enabled | bool | Enable/disable model | `parse_ml_model_config()` |
| TrainingWindow | training_window | int | Training data window | `parse_ml_model_config()` |
| PredictionHorizon | prediction_horizon | int | Prediction horizon (5) | `constants.py` DEFAULT_PREDICTION_HORIZON |
| ConfidenceThreshold | confidence_threshold | float | Min confidence (0.6) | `constants.py` DEFAULT_CONFIDENCE_THRESHOLD |
| MaxFeatures | max_features | int | Maximum features to use | `parse_ml_model_config()` |
| CrossValidationFolds | cv_folds | int | Cross-validation folds | `parse_ml_model_config()` |
| EarlyStoppingRounds | early_stopping_rounds | int | Early stopping patience | `parse_ml_model_config()` |

#### XGBoost Parameters (`constants.py` lines 120-129):
- **n_estimators**: 100
- **max_depth**: 6
- **learning_rate**: 0.1
- **subsample**: 0.8
- **colsample_bytree**: 0.8
- **objective**: "binary:logistic"
- **eval_metric**: "logloss"

#### LightGBM Parameters (`constants.py` lines 130-137):
- **num_leaves**: 31
- **max_depth**: -1
- **learning_rate**: 0.1
- **n_estimators**: 100
- **objective**: "binary"
- **metric**: "binary_logloss"

### 5. Signal Generation (Signals Sheet)

| Excel Parameter | Backend Field | Type | Description | Parser Location |
|----------------|---------------|------|-------------|-----------------|
| SignalName | signal_name | str | Signal identifier | `_parse_signals()` |
| Enabled | enabled | bool | Enable/disable signal | `_parse_signals()` |
| Condition | condition | str | Signal logic expression | `_parse_signals()` |
| ComparisonOperator | operator | ComparisonOperator | GT/LT/EQ/etc. | `models.py` ComparisonOperator |
| Threshold | threshold | float | Signal threshold value | `_parse_signals()` |
| LogicOperator | logic | SignalLogic | AND/OR/NOT | `models.py` SignalLogic |
| Weight | weight | float | Signal weight (0.0-1.0) | `_parse_signals()` |
| MinConfidence | min_confidence | float | Minimum confidence | `_parse_signals()` |
| CooldownPeriod | cooldown_period | int | Signal cooldown (bars) | `_parse_signals()` |

### 6. Risk Management (RiskManagement Sheet)

| Excel Parameter | Backend Field | Type | Description | Parser Location |
|----------------|---------------|------|-------------|-----------------|
| MaxPositions | max_positions | int | Maximum concurrent positions | `_parse_risk_config()` |
| MaxRiskPercent | max_risk_percent | float | Max portfolio risk (2.0%) | `constants.py` DEFAULT_RISK_PERCENT |
| StopLossPercent | stop_loss_percent | float | Stop loss percentage | `_parse_risk_config()` |
| TakeProfitPercent | take_profit_percent | float | Take profit percentage | `_parse_risk_config()` |
| PositionSizeMethod | position_size_method | str | FIXED/PERCENT/KELLY | `_parse_risk_config()` |
| MaxDrawdownPercent | max_drawdown_percent | float | Maximum drawdown limit | `_parse_risk_config()` |
| VolatilityAdjustment | volatility_adjustment | bool | Adjust for volatility | `_parse_risk_config()` |
| CorrelationLimit | correlation_limit | float | Position correlation limit | `_parse_risk_config()` |

### 7. Execution Settings (Execution Sheet)

| Excel Parameter | Backend Field | Type | Description | Parser Location |
|----------------|---------------|------|-------------|-----------------|
| OrderType | order_type | str | MARKET/LIMIT/STOP | `_parse_execution_config()` |
| SlippagePercent | slippage_percent | float | Expected slippage | `_parse_execution_config()` |
| FillProbability | fill_probability | float | Order fill probability | `_parse_execution_config()` |
| PartialFillAllowed | partial_fill_allowed | bool | Allow partial fills | `_parse_execution_config()` |
| MaxOrderSize | max_order_size | int | Maximum order size | `_parse_execution_config()` |
| MinOrderSize | min_order_size | int | Minimum order size | `_parse_execution_config()` |
| TimeInForce | time_in_force | str | DAY/GTC/IOC/FOK | `_parse_execution_config()` |
| ExecutionAlgorithm | execution_algorithm | str | TWAP/VWAP/POV | `_parse_execution_config()` |

## 🔄 Backend Processing Pipeline

### 1. Parser Entry Point
```python
# File: backtester_v2/strategies/ml_indicator/parser.py
def parse_input(portfolio_file, indicator_file, ml_model_file, signal_file):
    # Parse all Excel files and combine into unified configuration
    # Returns: Dict with parsed strategy configuration
```

### 2. Model Creation
```python
# File: backtester_v2/strategies/ml_indicator/models.py
class MLIndicatorStrategyModel:
    # Main strategy model containing all configuration
    portfolio: MLIndicatorPortfolioModel
    indicators: List[IndicatorConfig]
    ml_models: List[MLModelConfig]
    signals: List[SignalCondition]
```

### 3. Strategy Execution
```python
# File: backtester_v2/strategies/ml_indicator/strategy.py
class MLIndicatorStrategy:
    # Main strategy execution engine
    def run_backtest(self, config: MLIndicatorStrategyModel)
```

## 🎯 Parameter Gap Analysis

Based on parser analysis, the ML Indicator strategy supports **197+ parameters** across:

### Parameter Distribution
- **Portfolio Settings**: 15 parameters
- **Technical Indicators**: 80+ parameters (40+ indicators × 2+ params each)
- **SMC Configuration**: 20 parameters (5 concepts × 4 params each)
- **ML Models**: 35+ parameters (7 models × 5+ params each)
- **Signal Generation**: 25+ parameters
- **Risk Management**: 12 parameters
- **Execution Settings**: 10 parameters

### Excel vs Parser Gaps
- **Excel Configuration**: Supports all 197+ parameters across 30+ sheets
- **Parser Implementation**: Complete support for all parameter categories
- **Backend Models**: Full validation and type checking
- **Strategy Execution**: Comprehensive parameter utilization

## 🔧 Implementation Files Reference

### Core Strategy Files
```
backtester_v2/strategies/ml_indicator/
├── parser.py              # Excel parsing logic
├── models.py              # Data models and validation
├── strategy.py            # Main strategy execution
├── processor.py           # Data processing pipeline
├── constants.py           # Parameter definitions
└── indicators/            # Technical indicator implementations
    ├── talib_wrapper.py   # TA-Lib indicator interface
    └── smc_indicators.py  # Smart Money Concepts
```

### Configuration Files
```
backtester_v2/configurations/data/prod/ml/
├── ML_CONFIG_INDICATORS_1.0.0.xlsx   # 18 sheets
├── ML_CONFIG_PORTFOLIO_1.0.0.xlsx    # 8 sheets
└── ML_CONFIG_STRATEGY_1.0.0.xlsx     # 5 sheets
```

## 📊 Validation and Testing

### Parameter Validation
- **Type Checking**: All parameters validated against model definitions
- **Range Validation**: Numeric parameters checked against valid ranges
- **Cross-Parameter Validation**: Complex dependencies verified
- **Configuration Completeness**: Required parameters checked

### Testing Framework
- **Unit Tests**: Individual parser functions tested
- **Integration Tests**: End-to-end Excel parsing
- **Validation Tests**: Parameter boundary testing
- **Performance Tests**: Large configuration handling

## 🚀 Usage Examples

### Basic Configuration
```python
# Parse ML Indicator configuration
parser = MLIndicatorParser()
config = parser.parse_input(
    portfolio_file="ML_CONFIG_PORTFOLIO_1.0.0.xlsx",
    indicator_file="ML_CONFIG_INDICATORS_1.0.0.xlsx", 
    ml_model_file="ML_CONFIG_STRATEGY_1.0.0.xlsx"
)

# Execute strategy
strategy = MLIndicatorStrategy()
results = strategy.run_backtest(config["model"])
```

### Advanced Multi-Model Setup
```python
# Configure ensemble of ML models
config = {
    "portfolio": {...},
    "indicators": [
        {"type": "RSI", "timeframe": "15m", "parameters": {"timeperiod": 14}},
        {"type": "MACD", "timeframe": "15m", "parameters": {"fastperiod": 12}},
        {"type": "BOS", "enabled": True, "parameters": {"lookback": 20}}
    ],
    "ml_models": [
        {"type": "XGBOOST", "weight": 0.3},
        {"type": "LIGHTGBM", "weight": 0.3}, 
        {"type": "CATBOOST", "weight": 0.4}
    ]
}
```

## 🔍 Troubleshooting Common Issues

### Parameter Validation Errors
- **Invalid Indicator**: Check indicator name against `INDICATOR_CATEGORIES`
- **Missing Parameters**: Verify required parameters in Excel sheets
- **Type Mismatches**: Ensure correct data types in Excel cells
- **Range Violations**: Check parameter values against valid ranges

### Performance Optimization
- **Large Configurations**: Use pagination for large indicator sets
- **Memory Usage**: Monitor memory with many ML models
- **Processing Speed**: Optimize indicator calculations for real-time use
- **Validation Time**: Cache validation results for repeated configurations

---

**Document Maintenance**: This mapping document should be updated whenever:
- New indicators are added to the strategy
- ML model parameters are modified
- Excel sheet structures change
- Parser implementation is updated
- Backend models are extended

**Version Control**: Track changes to this document alongside code changes to maintain synchronization between Excel configurations and backend implementation.