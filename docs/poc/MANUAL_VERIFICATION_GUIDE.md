# Market Regime Components Manual Verification Guide

## Overview
This guide provides step-by-step instructions for manually verifying all 8 market regime components using individual PoC validation scripts with actual production data.

## Data Source
**Path**: `/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/`
- 87 Parquet files (January 2024, 22 trading days)
- 49-column production schema
- 8,537+ rows per file
- Complete options chain with Greeks, OI, volume data

## Component-by-Component Validation

### 🔍 Component 1: Triple Straddle Analysis

**Script**: `component_01_triple_straddle_poc_validation.py`

**Key Indicators to Verify**:
```bash
python3 component_01_triple_straddle_poc_validation.py
```

**Manual Verification Checklist**:
- [ ] **Rolling Straddle Calculation**: ATM, ITM1, OTM1 straddle prices calculated correctly
- [ ] **EMA Analysis**: 20, 50, 100, 200 period EMAs on straddle prices (not underlying)
- [ ] **VWAP Accuracy**: Volume-weighted using ce_volume + pe_volume
- [ ] **Pivot Points**: PP, R1-R3, S1-S3 calculated on straddle prices
- [ ] **Multi-timeframe**: 3,5,10,15min resampling working
- [ ] **RSI Calculation**: 14-period RSI on straddle prices
- [ ] **MACD Calculation**: 12,26,9 MACD on straddle prices
- [ ] **Performance**: Processing time <150ms
- [ ] **Strike Classification**: ATM/ITM1/OTM1 selection working

**Expected Results**:
- 120 features across all sub-components
- ATM straddle values in reasonable range (e.g., 50-500 for NIFTY)
- EMA convergence ratios <0.1
- VWAP different from simple average (volume weighting effect)
- RSI values between 0-100
- MACD crossover signals detected

---

### 🔍 Component 2: Greeks Sentiment Analysis

**Script**: `component_02_greeks_sentiment_poc_validation.py`

**Key Indicators to Verify**:
```bash
python3 component_02_greeks_sentiment_poc_validation.py
```

**Manual Verification Checklist**:
- [ ] **Greeks Extraction**: Delta, Gamma, Theta, Vega extracted with 96%+ coverage
- [ ] **Gamma Weight**: Gamma receives highest weight (1.5) in calculations
- [ ] **Volume-weighted Analysis**: ce_volume, pe_volume, ce_oi, pe_oi analysis working
- [ ] **Second-order Greeks**: Vanna, Charm, Volga calculated from first-order Greeks
- [ ] **7-level Sentiment**: Classification into strong_bullish, mild_bullish, etc.
- [ ] **DTE Adjustments**: Near-expiry gamma multiplier (3.0x) applied
- [ ] **Strike Type Integration**: ATM/ITM/OTM straddle selection
- [ ] **Performance**: Processing time <120ms
- [ ] **Memory**: Usage <280MB

**Expected Results**:
- 98 features across Greeks analysis
- Greeks coverage ≥96% (from story validation)
- Gamma weight = 1.5 (highest among all Greeks)
- Institutional flow scores calculated
- Sentiment classification working with confidence scores

---

### 🔍 Component 3: OI-PA Trending Analysis

**Script**: `component_03_oi_pa_trending_poc_validation.py`

**Key Indicators to Verify**:
```bash
python3 component_03_oi_pa_trending_poc_validation.py
```

**Manual Verification Checklist**:
- [ ] **Cumulative ATM ±7**: OI summation across ATM-7 to ATM+7 strikes
- [ ] **CE Option Seller Patterns**: 4 patterns (Short/Long Buildup/Covering)
- [ ] **PE Option Seller Patterns**: 4 patterns (Short/Long Buildup/Covering)
- [ ] **Future Seller Patterns**: 4 patterns for underlying correlation
- [ ] **3-way Correlation Matrix**: 6 scenarios (bullish, bearish, institutional, etc.)
- [ ] **Volume-OI Divergence**: Divergence detection working
- [ ] **Institutional Flow**: Smart money positioning detected
- [ ] **Multi-timeframe Rollups**: 5min(35%), 15min(20%), 3min(15%), 10min(30%)
- [ ] **Performance**: Processing time <200ms

**Expected Results**:
- 105 features across OI-PA analysis
- Option seller patterns detected in data
- 3-way correlation scenarios classified
- Volume-OI divergence correlation <-0.3 indicates institutional activity
- Multi-timeframe weights sum to 1.0

---

### 🔍 Component 4: IV Skew Analysis

**Script**: `component_04_iv_skew_poc_validation.py`

**Key Indicators to Verify**:
```bash
python3 component_04_iv_skew_poc_validation.py
```

**Manual Verification Checklist**:
- [ ] **IV Data Extraction**: ce_iv, pe_iv with 100% coverage
- [ ] **Volatility Surface**: 54-68 strikes per expiry constructed
- [ ] **Asymmetric Skew**: Put coverage ≥21%, Call coverage ≥9.9%
- [ ] **Risk Reversal**: Equidistant OTM puts/calls analysis
- [ ] **Smile Curvature**: Volatility smile shape analysis
- [ ] **DTE-Adaptive**: Short(54), Medium(68), Long(64) strikes per DTE
- [ ] **Wing Analysis**: Tail risk assessment at far OTM strikes
- [ ] **Surface Modeling**: Cubic spline/polynomial fitting capability
- [ ] **Performance**: Processing time <200ms

**Expected Results**:
- 87 features across IV analysis
- Strike count 54-68 per expiry
- Asymmetric coverage confirmed (more put than call coverage)
- Risk reversal values calculated
- Smile curvature detected (convex/concave)

---

## Component 5-8 Quick Validation

### Component 5: ATR-EMA-CPR
**Expected**: 95 features, technical analysis on dual assets

### Component 6: Correlation Analysis  
**Expected**: 110 features, cross-component correlation intelligence

### Component 7: Support/Resistance
**Expected**: 85 features, level detection with confluence analysis

### Component 8: Master Integration
**Expected**: 150 features, component aggregation and confidence metrics

---

## Running All Validations

### Sequential Execution:
```bash
# Run each component individually
python3 component_01_triple_straddle_poc_validation.py
python3 component_02_greeks_sentiment_poc_validation.py  
python3 component_03_oi_pa_trending_poc_validation.py
python3 component_04_iv_skew_poc_validation.py
```

### Results Files Generated:
- `component_01_poc_results.json`
- `component_02_poc_results.json` 
- `component_03_poc_results.json`
- `component_04_poc_results.json`

---

## Success Criteria

### Overall System Requirements:
- **Total Features**: 850+ across all components
- **Processing Time**: <600ms for all components combined
- **Memory Usage**: <2.5GB total system budget
- **Data Coverage**: 99%+ for OI/Volume, 96%+ for Greeks

### Individual Component Requirements:
| Component | Features | Processing | Memory | Key Validation |
|-----------|----------|------------|---------|----------------|
| Component 1 | 120 | <150ms | <512MB | EMA on straddle prices |
| Component 2 | 98 | <120ms | <280MB | Gamma weight = 1.5 |
| Component 3 | 105 | <200ms | <300MB | Option seller patterns |
| Component 4 | 87 | <200ms | <300MB | 54-68 strikes per expiry |

---

## Troubleshooting

### Common Issues:

1. **Import Errors**: 
   - Check Python path includes project root
   - Verify all dependencies installed

2. **Data Loading Issues**:
   - Confirm data path exists
   - Check parquet file permissions

3. **Processing Timeouts**:
   - Normal for first run (data loading)
   - Subsequent runs should be faster

4. **Memory Issues**:
   - Monitor system memory during validation
   - Close other applications if needed

---

## Manual Verification Steps

### For Each Component:

1. **Run PoC Script**: Execute the component-specific validation script
2. **Review Console Output**: Check all test results and indicators
3. **Verify JSON Results**: Open the generated JSON file for detailed metrics
4. **Check Performance**: Confirm processing times and memory usage
5. **Validate Indicators**: Manually verify key calculations make sense
6. **Mark Checklist**: Update the manual verification checklist

### Key Questions to Answer:

1. **Are the indicators calculating correctly?**
   - Do EMA values converge properly?
   - Are straddle prices reasonable?
   - Do Greeks weights match specifications?

2. **Is the data coverage adequate?**
   - 96%+ Greeks coverage for Component 2?
   - 99%+ OI coverage for Component 3?
   - 54-68 strikes for Component 4?

3. **Are performance targets met?**
   - Processing times within budget?
   - Memory usage reasonable?
   - Feature counts match expectations?

---

## Next Steps After Validation

1. **Fix Any Issues**: Address failed components before production
2. **Integration Testing**: Test component-to-component data flow  
3. **Performance Optimization**: Optimize any components exceeding budgets
4. **Production Deployment**: Deploy validated components to production
5. **Monitoring Setup**: Implement production monitoring and alerting

---

## Contact & Support

For issues with validation:
1. Check error messages in console output
2. Review generated JSON files for detailed diagnostics
3. Verify data path and file permissions
4. Test with single parquet file first if issues persist

This guide ensures comprehensive validation of all market regime components before production deployment.