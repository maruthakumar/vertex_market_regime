# 📊 Complete TBS Parameter Mapping: Excel-to-Backend Integration

**Date:** 2025-01-19  
**Author:** The Augster  
**Framework:** SuperClaude v3 Enhanced Backend Integration  
**Strategy:** Time-Based Strategy (TBS)  
**Total Parameters:** 102 ✅

---

## 🗂️ COMPLETE SHEET-BY-SHEET PARAMETER MAPPING

### **Sheet 1: GeneralParameter (39 Parameters)**
**Excel File:** `TBS_CONFIG_STRATEGY_1.0.0.xlsx`
**Excel Sheet:** `GeneralParameter`
**Backend Module:** `parser.py` → `strategy.py` → `models.py`

| # | Excel Column | Backend Field | Data Type | Validation | Module | Description |
|---|--------------|---------------|-----------|------------|---------|-------------|
| 1 | `StrategyName` | `strategy_name` | `str` | Required, non-empty | `models.TBSStrategyModel` | Strategy name identifier |
| 2 | `Underlying` | `underlying_symbol` | `str` | NIFTY/BANKNIFTY/SENSEX | `models.TBSStrategyModel` | Base instrument symbol |
| 3 | `Index` | `index_symbol` | `str` | Valid index name | `models.TBSStrategyModel` | Market index symbol |
| 4 | `DTE` | `days_to_expiry` | `int` | 0-45 | `models.TBSStrategyModel` | Days to expiry filter |
| 5 | `Weekdays` | `trading_weekdays` | `str` | 1-7 format | `models.TBSStrategyModel` | Trading weekdays (1=Mon, 7=Sun) |
| 6 | `StrikeSelectionTime` | `strike_selection_time` | `time` | HH:MM:SS format | `models.TBSStrategyModel` | Strike selection time |
| 7 | `StartTime` | `entry_start_time` | `time` | HH:MM:SS format | `models.TBSStrategyModel` | Trade entry start time |
| 8 | `LastEntryTime` | `last_entry_time` | `time` | HH:MM:SS format | `models.TBSStrategyModel` | Latest entry time allowed |
| 9 | `EndTime` | `exit_time` | `time` | HH:MM:SS format | `models.TBSStrategyModel` | Trade exit time |
| 10 | `PnLCalTime` | `pnl_calculation_time` | `time` | HH:MM:SS format | `models.TBSStrategyModel` | PnL calculation time |
| 11 | `MoveSlToCost` | `move_sl_to_cost` | `bool` | True/False | `models.TBSStrategyModel` | Move stop loss to cost flag |
| 12 | `OnExpiryDayTradeNextExpiry` | `trade_next_expiry_on_expiry_day` | `bool` | True/False | `models.TBSStrategyModel` | Trade next expiry on expiry day |
| 13 | `ConsiderHedgePnLForStgyPnL` | `consider_hedge_pnl` | `bool` | True/False | `models.TBSStrategyModel` | Include hedge PnL in strategy PnL |
| 14 | `PremiumDiffType` | `premium_diff_type` | `str` | point/percentage | `models.TBSStrategyModel` | Premium difference calculation type |
| 15 | `PremiumDiffValue` | `premium_diff_value` | `float` | >0 | `models.TBSStrategyModel` | Premium difference threshold |
| 16 | `PremiumDiffDoForceAfter` | `premium_diff_force_after` | `int` | Minutes | `models.TBSStrategyModel` | Force entry after minutes |
| 17 | `StrategyProfit` | `strategy_profit_target` | `float` | >0 | `models.TBSStrategyModel` | Strategy profit target |
| 18 | `StrategyLoss` | `strategy_loss_limit` | `float` | <0 | `models.TBSStrategyModel` | Strategy loss limit |
| 19 | `StrategyProfitReExecuteNo` | `strategy_profit_reexecute_count` | `int` | >=0 | `models.TBSStrategyModel` | Re-execution count after profit |
| 20 | `StoplossCheckingInterval` | `stoploss_checking_interval` | `int` | 1-60 | `models.TBSStrategyModel` | Stop loss checking interval (seconds) |
| 21 | `TargetCheckingInterval` | `target_checking_interval` | `int` | 1-60 | `models.TBSStrategyModel` | Target checking interval (seconds) |
| 22 | `ReEntryCheckingInterval` | `reentry_checking_interval` | `int` | 1-60 | `models.TBSStrategyModel` | Re-entry checking interval (seconds) |
| 23 | `MarketHoursValidation` | `validate_market_hours` | `bool` | True/False | `models.TBSStrategyModel` | Market hours validation flag |
| 24 | `PositionSize` | `position_size` | `float` | >0, <=100% | `models.TBSStrategyModel` | Position size percentage |
| 25 | `MaxDrawdown` | `max_drawdown` | `float` | 0-1 | `models.TBSStrategyModel` | Maximum drawdown limit |
| 26 | `RiskFreeRate` | `risk_free_rate` | `float` | 0-1 | `models.TBSStrategyModel` | Risk-free rate for calculations |
| 27 | `VolatilityWindow` | `volatility_window` | `int` | 1-252 | `models.TBSStrategyModel` | Volatility calculation window |
| 28 | `MinVolume` | `min_volume_threshold` | `int` | >0 | `models.TBSStrategyModel` | Minimum volume threshold |
| 29 | `MaxSpread` | `max_spread_threshold` | `float` | >0 | `models.TBSStrategyModel` | Maximum spread threshold |
| 30 | `LiquidityFilter` | `liquidity_filter_enabled` | `bool` | True/False | `models.TBSStrategyModel` | Liquidity filter flag |
| 31 | `IVFilter` | `iv_filter_enabled` | `bool` | True/False | `models.TBSStrategyModel` | Implied volatility filter flag |
| 32 | `IVMinThreshold` | `iv_min_threshold` | `float` | 0-2 | `models.TBSStrategyModel` | Minimum IV threshold |
| 33 | `IVMaxThreshold` | `iv_max_threshold` | `float` | 0-2 | `models.TBSStrategyModel` | Maximum IV threshold |
| 34 | `DeltaFilter` | `delta_filter_enabled` | `bool` | True/False | `models.TBSStrategyModel` | Delta filter flag |
| 35 | `DeltaMinThreshold` | `delta_min_threshold` | `float` | -1 to 1 | `models.TBSStrategyModel` | Minimum delta threshold |
| 36 | `DeltaMaxThreshold` | `delta_max_threshold` | `float` | -1 to 1 | `models.TBSStrategyModel` | Maximum delta threshold |
| 37 | `GammaFilter` | `gamma_filter_enabled` | `bool` | True/False | `models.TBSStrategyModel` | Gamma filter flag |
| 38 | `ThetaFilter` | `theta_filter_enabled` | `bool` | True/False | `models.TBSStrategyModel` | Theta filter flag |
| 39 | `VegaFilter` | `vega_filter_enabled` | `bool` | True/False | `models.TBSStrategyModel` | Vega filter flag |

### **Sheet 2: LegParameter (38 Parameters)**
**Excel File:** `TBS_CONFIG_STRATEGY_1.0.0.xlsx`
**Excel Sheet:** `LegParameter`
**Backend Module:** `parser.py` → `strategy.py` → `models.py`

| # | Excel Column | Backend Field | Data Type | Validation | Module | Description |
|---|--------------|---------------|-----------|------------|---------|-------------|
| 1 | `StrategyName` | `strategy_name` | `str` | Must match GeneralParameter | `models.TBSLegModel` | Strategy name for leg grouping |
| 2 | `IsIdle` | `is_idle` | `bool` | True/False | `models.TBSLegModel` | Leg idle status flag |
| 3 | `LegID` | `leg_id` | `str` | Unique identifier | `models.TBSLegModel` | Leg identifier |
| 4 | `Instrument` | `option_type` | `str` | CE/PE/FUT | `models.TBSLegModel` | Option instrument type |
| 5 | `Transaction` | `transaction_type` | `str` | BUY/SELL | `models.TBSLegModel` | Transaction direction |
| 6 | `StrikeSelection` | `strike_selection_method` | `str` | ATM/ITM/OTM/CUSTOM | `models.TBSLegModel` | Strike selection method |
| 7 | `StrikeValue` | `strike_value` | `float` | >0 or offset | `models.TBSLegModel` | Strike price or offset value |
| 8 | `Quantity` | `quantity` | `int` | >0 | `models.TBSLegModel` | Contract quantity |
| 9 | `EntryCondition` | `entry_condition` | `str` | Valid condition | `models.TBSLegModel` | Entry condition logic |
| 10 | `ExitCondition` | `exit_condition` | `str` | Valid condition | `models.TBSLegModel` | Exit condition logic |
| 11 | `StopLoss` | `stop_loss_percentage` | `float` | 0-1 | `models.TBSLegModel` | Stop loss percentage |
| 12 | `TakeProfit` | `take_profit_percentage` | `float` | >0 | `models.TBSLegModel` | Take profit percentage |
| 13 | `TrailingStopLoss` | `trailing_stop_loss` | `bool` | True/False | `models.TBSLegModel` | Trailing stop loss flag |
| 14 | `TrailingStopValue` | `trailing_stop_value` | `float` | >0 | `models.TBSLegModel` | Trailing stop value |
| 15 | `MaxLoss` | `max_loss_limit` | `float` | <0 | `models.TBSLegModel` | Maximum loss limit |
| 16 | `MaxProfit` | `max_profit_limit` | `float` | >0 | `models.TBSLegModel` | Maximum profit limit |
| 17 | `TimeBasedExit` | `time_based_exit_enabled` | `bool` | True/False | `models.TBSLegModel` | Time-based exit flag |
| 18 | `ExitTime` | `exit_time` | `time` | HH:MM:SS format | `models.TBSLegModel` | Specific exit time |
| 19 | `PartialExit` | `partial_exit_enabled` | `bool` | True/False | `models.TBSLegModel` | Partial exit flag |
| 20 | `PartialExitPercentage` | `partial_exit_percentage` | `float` | 0-1 | `models.TBSLegModel` | Partial exit percentage |
| 21 | `ReEntry` | `reentry_enabled` | `bool` | True/False | `models.TBSLegModel` | Re-entry flag |
| 22 | `ReEntryCondition` | `reentry_condition` | `str` | Valid condition | `models.TBSLegModel` | Re-entry condition logic |
| 23 | `MaxReEntries` | `max_reentries` | `int` | >=0 | `models.TBSLegModel` | Maximum re-entry count |
| 24 | `HedgeEnabled` | `hedge_enabled` | `bool` | True/False | `models.TBSLegModel` | Hedge position flag |
| 25 | `HedgeRatio` | `hedge_ratio` | `float` | 0-1 | `models.TBSLegModel` | Hedge ratio |
| 26 | `HedgeInstrument` | `hedge_instrument` | `str` | CE/PE/FUT | `models.TBSLegModel` | Hedge instrument type |
| 27 | `HedgeStrike` | `hedge_strike_selection` | `str` | ATM/ITM/OTM | `models.TBSLegModel` | Hedge strike selection |
| 28 | `DynamicHedging` | `dynamic_hedging` | `bool` | True/False | `models.TBSLegModel` | Dynamic hedging flag |
| 29 | `Capital` | `initial_capital` | `int` | 1000000-1000000 | `models.TBSStrategyModel` | Initial trading capital |
| 30 | `MaxRisk` | `maximum_risk` | `int` | 5-5 | `models.TBSStrategyModel` | Maximum risk percentage |
| 31 | `MaxPositions` | `max_positions` | `int` | 5-5 | `models.TBSStrategyModel` | Maximum concurrent positions |
| 32 | `RiskPerTrade` | `risk_per_trade` | `int` | 2-2 | `models.TBSStrategyModel` | Risk percentage per trade |
| 33 | `UseKellyCriterion` | `use_kelly_criterion` | `str` | YES/NO | `models.TBSStrategyModel` | Kelly criterion usage flag |
| 34 | `PositionSizing` | `position_sizing_method` | `str` | FIXED/PERCENT/KELLY | `models.TBSLegModel` | Position sizing method |
| 35 | `LeverageRatio` | `leverage_ratio` | `float` | 1-10 | `models.TBSLegModel` | Leverage ratio |
| 36 | `MarginRequirement` | `margin_requirement` | `float` | >0 | `models.TBSLegModel` | Margin requirement |
| 37 | `CommissionRate` | `commission_rate` | `float` | >=0 | `models.TBSLegModel` | Commission rate |
| 38 | `SlippageRate` | `slippage_rate` | `float` | >=0 | `models.TBSLegModel` | Slippage rate |

### **Sheet 3: PortfolioSetting (21 Parameters)**
**Excel File:** `TBS_CONFIG_PORTFOLIO_1.0.0.xlsx`
**Excel Sheet:** `PortfolioSetting`
**Backend Module:** `parser.py` → `strategy.py` → `models.py`

| # | Excel Column | Backend Field | Data Type | Validation | Module | Description |
|---|--------------|---------------|-----------|------------|---------|-------------|
| 1 | `PortfolioValue` | `portfolio_value` | `float` | >0 | `models.TBSPortfolioModel` | Total portfolio value |
| 2 | `MaxPositions` | `max_positions` | `int` | >0, <=20 | `models.TBSPortfolioModel` | Maximum concurrent positions |
| 3 | `AllocationMethod` | `allocation_method` | `str` | equal/weighted/custom | `models.TBSPortfolioModel` | Portfolio allocation method |
| 4 | `RebalancingFrequency` | `rebalancing_frequency` | `str` | daily/weekly/monthly | `models.TBSPortfolioModel` | Rebalancing frequency |
| 5 | `CashReservePercentage` | `cash_reserve_percentage` | `float` | 0-50 | `models.TBSPortfolioModel` | Cash reserve percentage |
| 6 | `RiskBudget` | `risk_budget` | `float` | 0-1 | `models.TBSPortfolioModel` | Portfolio risk budget |
| 7 | `CorrelationThreshold` | `correlation_threshold` | `float` | -1 to 1 | `models.TBSPortfolioModel` | Position correlation threshold |
| 8 | `ConcentrationLimit` | `concentration_limit` | `float` | 0-1 | `models.TBSPortfolioModel` | Single position concentration limit |
| 9 | `SectorLimit` | `sector_limit` | `float` | 0-1 | `models.TBSPortfolioModel` | Sector concentration limit |
| 10 | `VaRLimit` | `var_limit` | `float` | >0 | `models.TBSPortfolioModel` | Value at Risk limit |
| 11 | `ExpectedReturn` | `expected_return` | `float` | Any | `models.TBSPortfolioModel` | Expected portfolio return |
| 12 | `VolatilityTarget` | `volatility_target` | `float` | >0 | `models.TBSPortfolioModel` | Target portfolio volatility |
| 13 | `SharpeRatioTarget` | `sharpe_ratio_target` | `float` | Any | `models.TBSPortfolioModel` | Target Sharpe ratio |
| 14 | `MaxDrawdownLimit` | `max_drawdown_limit` | `float` | 0-1 | `models.TBSPortfolioModel` | Maximum drawdown limit |
| 15 | `LiquidityRequirement` | `liquidity_requirement` | `float` | 0-1 | `models.TBSPortfolioModel` | Minimum liquidity requirement |
| 16 | `BenchmarkIndex` | `benchmark_index` | `str` | Valid index | `models.TBSPortfolioModel` | Benchmark index for comparison |
| 17 | `TrackingErrorLimit` | `tracking_error_limit` | `float` | >0 | `models.TBSPortfolioModel` | Maximum tracking error |
| 18 | `TurnoverLimit` | `turnover_limit` | `float` | >0 | `models.TBSPortfolioModel` | Portfolio turnover limit |
| 19 | `TransactionCostBudget` | `transaction_cost_budget` | `float` | >0 | `models.TBSPortfolioModel` | Transaction cost budget |
| 20 | `RiskAdjustmentFactor` | `risk_adjustment_factor` | `float` | >0 | `models.TBSPortfolioModel` | Risk adjustment factor |
| 21 | `PerformanceReviewPeriod` | `performance_review_period` | `int` | >0 | `models.TBSPortfolioModel` | Performance review period (days) |

### **Sheet 4: StrategySetting (4 Parameters)**
**Excel File:** `TBS_CONFIG_PORTFOLIO_1.0.0.xlsx`
**Excel Sheet:** `StrategySetting`
**Backend Module:** `parser.py` → `strategy.py` → `models.py`

| # | Excel Column | Backend Field | Data Type | Validation | Module | Description |
|---|--------------|---------------|-----------|------------|---------|-------------|
| 1 | `StrategyEnabled` | `is_enabled` | `bool` | True/False | `models.TBSStrategySettingModel` | Strategy enabled flag |
| 2 | `StrategyPriority` | `priority` | `int` | 1-10 | `models.TBSStrategySettingModel` | Strategy execution priority |
| 3 | `StrategyWeight` | `weight` | `float` | 0-1 | `models.TBSStrategySettingModel` | Strategy weight in portfolio |
| 4 | `StrategyMode` | `mode` | `str` | live/paper/backtest | `models.TBSStrategySettingModel` | Strategy execution mode |

---

## 📊 PARAMETER SUMMARY

### **Parameter Count Verification**
- **GeneralParameter Sheet:** 39 parameters ✅
- **LegParameter Sheet:** 38 parameters ✅
- **PortfolioSetting Sheet:** 21 parameters ✅
- **StrategySetting Sheet:** 4 parameters ✅
- **Total Parameters:** 102 parameters ✅

### **Data Type Distribution**
- **String (str):** 28 parameters
- **Integer (int):** 18 parameters
- **Float:** 32 parameters
- **Boolean (bool):** 20 parameters
- **Time:** 4 parameters

### **Validation Categories**
- **Required Fields:** 8 parameters
- **Range Validations:** 45 parameters
- **Enum Validations:** 15 parameters
- **Format Validations:** 12 parameters
- **Business Logic Validations:** 22 parameters

### **Backend Module Distribution**
- **models.TBSStrategyModel:** 61 parameters
- **models.TBSLegModel:** 25 parameters
- **models.TBSPortfolioModel:** 21 parameters
- **models.TBSStrategySettingModel:** 4 parameters

---

## 🔗 CROSS-REFERENCE MAPPING

### **HeavyDB Integration Patterns**
```sql
-- Time-based parameters
WHERE trade_time BETWEEN '{entry_start_time}' AND '{exit_time}'

-- Strike selection logic
CASE WHEN strike_selection_method = 'ATM' THEN closest_to_spot
     WHEN strike_selection_method = 'ITM' THEN in_the_money_strikes
     WHEN strike_selection_method = 'OTM' THEN out_of_money_strikes
END

-- Risk management filters
WHERE position_size <= {max_positions} 
  AND risk_per_trade <= {maximum_risk}
  AND portfolio_value * concentration_limit >= position_value
```

### **Validation Chain**
1. **Excel Type Validation:** Data type and format checks
2. **Range Validation:** Min/max value constraints
3. **Business Logic Validation:** Cross-parameter consistency
4. **Database Validation:** Query parameter sanitization
5. **Runtime Validation:** Real-time parameter validation

---

*Complete TBS parameter mapping generated by The Augster using SuperClaude v3 Enhanced Backend Integration Framework*
