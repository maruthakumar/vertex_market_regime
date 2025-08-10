# BMAD Validation System - Detailed Parameter Breakdown

## Overview

This document provides a comprehensive breakdown of all parameters for each trading strategy, organized by Excel file and sheet. This granular view enables precise validation tracking and ensures complete parameter coverage.

## TBS (Time-Based Strategy)
**Total Parameters: 83** | **Files: 2** | **Sheets: 4**

### TBS_CONFIG_STRATEGY_1.0.0.xlsx
#### GeneralParameter Sheet (43 parameters)
- Core strategy settings: StrategyName, MoveSlToCost, Underlying, Index, Weekdays, DTE
- Timing parameters: StrikeSelectionTime, StartTime, LastEntryTime, EndTime
- P&L settings: StrategyProfit, StrategyLoss, StrategyProfitReExecuteNo, StrategyLossReExecuteNo
- Trailing settings: StrategyTrailingType, LockPercent, TrailPercent
- Square-off settings: SqOff1Time, SqOff1Percent, SqOff2Time, SqOff2Percent
- Premium difference settings: CheckPremiumDiffCondition, PremiumDiffType, PremiumDiffValue
- Interval settings: StoplossCheckingInterval, TargetCheckingInterval, ReEntryCheckingInterval

#### LegParameter Sheet (29 parameters)
- Option leg configuration for entry/exit conditions
- Strike selection methods
- Premium conditions
- Re-entry parameters

### TBS_CONFIG_PORTFOLIO_1.0.0.xlsx
#### PortfolioSetting Sheet (6 parameters)
- Capital, MaxRisk, MaxPositions, RiskPerTrade, UseKellyCriterion, RebalanceFrequency

#### StrategySetting Sheet (5 parameters)
- StrategyName, StrategyExcelFilePath, Enabled, Priority, AllocationPercent

---

## TV (TradingView)
**Total Parameters: 133** | **Files: 6** | **Sheets: 11**

### TV_CONFIG_STRATEGY_1.0.0.xlsx
#### GeneralParameter Sheet (43 parameters)
- Similar structure to TBS GeneralParameter
- TradingView specific signal integration settings

#### LegParameter Sheet (29 parameters)
- Option leg configuration
- Signal-based entry/exit parameters

### TV_CONFIG_SIGNALS_1.0.0.xlsx
#### List of trades Sheet (4 parameters)
- Trade #, Type, Date/Time, Contracts

### TV_CONFIG_MASTER_1.0.0.xlsx
#### Setting Sheet (24 parameters)
- Master configuration for TradingView integration
- Signal processing settings
- Execution parameters

### Portfolio Files (MANUAL/LONG/SHORT)
Each contains:
- PortfolioSetting (6 parameters)
- StrategySetting (5 parameters)

---

## OI (Open Interest)
**Total Parameters: 142** | **Files: 2** | **Sheets: 8**

### OI_CONFIG_STRATEGY_1.0.0.xlsx
#### GeneralParameter Sheet (45 parameters)
- Extended strategy settings for OI analysis
- OI threshold parameters
- Market microstructure settings

#### LegParameter Sheet (35 parameters)
- Enhanced leg parameters for OI-based entries

#### WeightConfiguration Sheet (25 parameters)
- Dynamic weight adjustment settings
- Factor-based weight calculations

#### FactorParameters Sheet (16 parameters)
- FactorName, FactorType, BaseWeight, MinWeight, MaxWeight
- LookbackPeriod, SmoothingFactor, ThresholdType, ThresholdValue

#### PortfolioSetting Sheet (5 parameters)
- Portfolio-specific OI settings

#### StrategySetting Sheet (5 parameters)
- Strategy-level OI configuration

### OI_CONFIG_PORTFOLIO_1.0.0.xlsx
- Standard portfolio and strategy settings (11 parameters)

---

## ORB (Opening Range Breakout)
**Total Parameters: 19** | **Files: 2** | **Sheets: 3**

### ORB_CONFIG_STRATEGY_1.0.0.xlsx
#### MainSetting Sheet (8 parameters)
- StrategyName, Enabled, Index
- OpeningRangeStart, OpeningRangeEnd
- BreakoutThreshold, StopLoss, Target

### ORB_CONFIG_PORTFOLIO_1.0.0.xlsx
- PortfolioSetting (6 parameters)
- StrategySetting (5 parameters)

---

## POS (Positional)
**Total Parameters: 156** | **Files: 3** | **Sheets: 7**

### POS_CONFIG_PORTFOLIO_1.0.0.xlsx
#### PortfolioSetting Sheet (25 parameters)
- Extended portfolio configuration
- Position sizing rules
- Risk parameters

#### StrategySetting Sheet (7 parameters)
- StrategyName, StrategyType, PortfolioName
- StrategyExcelFilePath, Enabled, Priority, AllocationPercent

#### RiskManagement Sheet (13 parameters)
- Greek limits: MaxPortfolioDelta, MaxPortfolioGamma, MaxPortfolioVega, MaxPortfolioTheta
- Hedging settings: DeltaHedgingEnabled, GammaHedgingEnabled, VegaHedgingEnabled
- Risk metrics: StressTestEnabled, VaRCalculationMethod, LiquidityRiskEnabled

#### MarketFilters Sheet (10 parameters)
- TrendFilter, VolatilityRegime, MarketStructure
- AvoidEvents, EventDaysBuffer
- LiquidityFilter, VolumeTrendFilter
- ImpliedVolatilityFilter, OpenInterestFilter, PCRFilter

### POS_CONFIG_ADJUSTMENT_1.0.0.xlsx
#### AdjustmentRules Sheet (8 parameters)
- RuleID, RuleName, TriggerType, TriggerThreshold
- ActionType, ActionParameters, Enabled, Priority

### POS_CONFIG_STRATEGY_1.0.0.xlsx
#### PositionalParameter Sheet (66 parameters)
- Comprehensive positional strategy settings

#### LegParameter Sheet (27 parameters)
- Position-specific leg configuration

---

## ML (Machine Learning) - ENHANCED VALIDATION
**Total Parameters: 439** | **Files: 3** | **Sheets: 33**

### ML_CONFIG_PORTFOLIO_1.0.0.xlsx
- Standard portfolio configuration (11 parameters)

### ML_CONFIG_INDICATORS_1.0.0.xlsx
#### IndicatorConfig Sheet (3 parameters)
- IndicatorName, Enabled, Parameters

### ML_CONFIG_STRATEGY_1.0.0.xlsx (30 sheets!)

#### Model Configuration Sheets
- **01_LightGBM_Config** (25 parameters)
- **02_CatBoost_Config** (25 parameters)
- **03_TabNet_Config** (25 parameters)
- **04_LSTM_Config** (18 parameters)
- **05_Transformer_Config** (18 parameters)
- **06_Ensemble_Config** (15 parameters)

#### Feature Engineering Sheets (7 parameters each)
- **07_Market_Regime_Features**
- **08_Greek_Features**
- **09_IV_Features**
- **10_OI_Features**
- **11_Technical_Features**
- **12_Microstructure_Features**
- **13_Rejection_Candle_Features**
- **14_Volume_Profile_Features**
- **15_Enhanced_VWAP_Features**

#### Risk & Position Management
- **16_Position_Sizing** (15 parameters)
- **17_Risk_Limits** (15 parameters)
- **18_Pattern_Based_Stops** (20 parameters)
- **19_Circuit_Breaker** (15 parameters)

#### Signal Processing
- **20_Straddle_Config** (18 parameters)
- **21_Signal_Filters** (20 parameters)
- **22_Signal_Processing** (18 parameters)

#### Training & Infrastructure
- **23_Training_Config** (20 parameters)
- **24_Model_Training** (22 parameters)
- **25_Backtesting_Config** (20 parameters)
- **26_HeavyDB_Connection** (18 parameters)
- **27_Data_Source** (18 parameters)

#### System Overview
- **28_System_Overview** (6 parameters)
- **29_Performance_Targets** (5 parameters)

---

## MR (Market Regime) - ENHANCED VALIDATION
**Total Parameters: 267** | **Files: 4** | **Sheets: 43**

### MR_CONFIG_STRATEGY_1.0.0.xlsx (31 sheets!)

#### Core Configuration
- **StabilityConfiguration** (26 parameters)
- **TransitionManagement** (21 parameters)
- **NoiseFiltering** (18 parameters)

#### Indicator Configurations
- **GreekSentimentConfig** (18 parameters)
- **TrendingOIPAConfig** (16 parameters)
- **StraddleAnalysisConfig** (19 parameters)
- **IVSurfaceConfig** (18 parameters)
- **ATRIndicatorsConfig** (20 parameters)

#### Regime Management
- **RegimeClassification** (8 parameters)
- **RegimeParameters** (8 parameters)
- **TransitionRules** (8 parameters)

#### Multi-Timeframe & Adaptive
- **MultiTimeframeConfig** (7 parameters)
- **DynamicWeightageConfig** (7 parameters)
- **AdaptiveTuning** (18 parameters)

#### Performance & Validation
- **PerformanceMetrics** (7 parameters)
- **ValidationRules** (7 parameters)
- **IntradaySettings** (7 parameters)

### MR_CONFIG_PORTFOLIO_1.0.0.xlsx
- Standard portfolio configuration (11 parameters)

### MR_CONFIG_OPTIMIZATION_1.0.0.xlsx
- OptimizationSettings (3 parameters)

### MR_CONFIG_REGIME_1.0.0.xlsx
- Regime-specific settings (12 parameters)

---

## IND (Indicator)
**Total Parameters: 197** (from column mapping)

---

## OPT (Optimization)
**Total Parameters: 283** (from Excel analysis)

---

## Summary Statistics

| Strategy | Parameters | Files | Sheets | Enhanced | Complexity |
|----------|------------|-------|--------|----------|------------|
| TBS      | 83         | 2     | 4      | No       | Medium     |
| TV       | 133        | 6     | 11     | No       | Medium     |
| OI       | 142        | 2     | 8      | No       | High       |
| ORB      | 19         | 2     | 3      | No       | Low        |
| POS      | 156        | 3     | 7      | No       | High       |
| ML       | 439        | 3     | 33     | Yes      | Very High  |
| MR       | 267        | 4     | 43     | Yes      | Very High  |
| IND      | 197        | -     | -      | No       | High       |
| OPT      | 283        | -     | -      | No       | High       |

**Total: 1,719 parameters across all strategies**

## Validation Approach by Complexity

### Low Complexity (ORB)
- Simple parameter validation
- Direct mapping verification
- Basic range checks

### Medium Complexity (TBS, TV)
- Parameter interdependency checks
- Signal integration validation
- Timing constraint verification

### High Complexity (OI, POS, IND, OPT)
- Multi-sheet coordination
- Complex dependency validation
- Performance optimization required

### Very High Complexity (ML, MR) - Enhanced Validation
- Double validation protocol
- Statistical anomaly detection
- Cross-reference validation
- Model-specific constraints
- Feature engineering validation
- 30+ sheets coordination