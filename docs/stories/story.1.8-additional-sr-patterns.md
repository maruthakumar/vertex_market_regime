# Additional Support & Resistance Patterns for Component 7
## Advanced OI-Based and Cross-Component S&R Formation

## 🎯 NEW PATTERN CATEGORIES DISCOVERED

### Enhanced VWAP & Pivot Integration

#### **Daily VWAP and Previous Day VWAP on Straddles**
```python
def calculate_vwap_sr_levels(self, straddle_data):
    """
    Daily and Previous Day VWAP create strong S&R levels
    """
    # Daily VWAP (resets at 9:15 AM)
    daily_vwap = calculate_vwap(
        straddle_data['price'],
        straddle_data['volume'],
        reset='daily'
    )
    
    # Previous Day VWAP (final value from yesterday)
    prev_day_vwap = straddle_data['prev_day_final_vwap']
    
    vwap_levels = [
        {
            'level': daily_vwap,
            'type': 'daily_vwap_sr',
            'strength': 0.8,
            'description': 'Current day VWAP acts as dynamic S&R'
        },
        {
            'level': prev_day_vwap,
            'type': 'prev_day_vwap_sr',
            'strength': 0.7,
            'description': 'Previous day VWAP acts as reference S&R'
        }
    ]
    
    # VWAP bands (1 & 2 standard deviations)
    vwap_std = calculate_vwap_std(straddle_data)
    vwap_levels.extend([
        {'level': daily_vwap + vwap_std, 'type': 'vwap_upper_band_1std'},
        {'level': daily_vwap + 2*vwap_std, 'type': 'vwap_upper_band_2std'},
        {'level': daily_vwap - vwap_std, 'type': 'vwap_lower_band_1std'},
        {'level': daily_vwap - 2*vwap_std, 'type': 'vwap_lower_band_2std'}
    ])
    
    return vwap_levels
```

#### **Comprehensive Day Level S&R**
```python
def calculate_day_level_sr(self, straddle_data):
    """
    Current and Previous Day High/Low/Close on straddle prices
    """
    day_levels = []
    
    # Current Day Levels (dynamic, updates throughout the day)
    day_levels.extend([
        {
            'level': straddle_data['current_day_high'],
            'type': 'current_day_high',
            'strength': 0.9,
            'description': 'Intraday resistance level'
        },
        {
            'level': straddle_data['current_day_low'],
            'type': 'current_day_low',
            'strength': 0.9,
            'description': 'Intraday support level'
        }
    ])
    
    # Previous Day Levels (static reference points)
    day_levels.extend([
        {
            'level': straddle_data['prev_day_high'],
            'type': 'prev_day_high',
            'strength': 0.85,
            'description': 'Yesterday high as resistance'
        },
        {
            'level': straddle_data['prev_day_low'],
            'type': 'prev_day_low',
            'strength': 0.85,
            'description': 'Yesterday low as support'
        },
        {
            'level': straddle_data['prev_day_close'],
            'type': 'prev_day_close',
            'strength': 0.75,
            'description': 'Yesterday close as pivot reference'
        }
    ])
    
    # Opening levels
    day_levels.append({
        'level': straddle_data['today_open'],
        'type': 'today_open',
        'strength': 0.7,
        'description': 'Opening level as S&R reference'
    })
    
    return day_levels
```

#### **Extended Pivot Points (R3/S3 included)**
```python
def calculate_extended_pivots(self, high, low, close):
    """
    Calculate full range of pivot points including R3/S3
    """
    # Standard Pivot Point
    pp = (high + low + close) / 3
    
    # All resistance and support levels
    r1 = (2 * pp) - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)  # Extended resistance
    
    s1 = (2 * pp) - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)  # Extended support
    
    return {
        'R3': r3,  # Extreme resistance
        'R2': r2,  # Strong resistance
        'R1': r1,  # Moderate resistance
        'PP': pp,  # Central pivot
        'S1': s1,  # Moderate support
        'S2': s2,  # Strong support
        'S3': s3   # Extreme support
    }
```

### 1. OI-BASED DYNAMIC S&R PATTERNS

#### **Pattern A: OI Concentration Walls**
```
HIGH OI CONCENTRATION = STRONG S&R LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strike    CE OI     PE OI     S&R Type
19800     50,000    15,000    -
19850     75,000    20,000    -
19900     120,000   25,000    ← RESISTANCE (High CE OI)
19950     85,000    45,000    -
20000     60,000    60,000    ← PIVOT (Balanced)
20050     45,000    85,000    -
20100     25,000    120,000   ← SUPPORT (High PE OI)
20150     20,000    75,000    -
20200     15,000    50,000    -

RULE: 
• CE OI > 85th percentile → RESISTANCE
• PE OI > 85th percentile → SUPPORT
• Balanced high OI → PIVOT ZONE
```

#### **Pattern B: Max Pain Migration Levels**
```python
def detect_max_pain_migration_sr(self, oi_data, time_window='5min'):
    """
    Max Pain acts as a magnetic S&R level that moves dynamically
    """
    # Calculate max pain every 5 minutes
    max_pain_series = calculate_rolling_max_pain(oi_data, window=time_window)
    
    # When max pain stays stable → Strong S&R
    # When max pain migrates → Previous levels become S&R
    
    migration_levels = []
    for i in range(1, len(max_pain_series)):
        prev_mp = max_pain_series[i-1]
        curr_mp = max_pain_series[i]
        
        if abs(curr_mp - prev_mp) > 50:  # Max pain shifted
            # Previous max pain becomes S&R
            migration_levels.append({
                'level': prev_mp,
                'type': 'abandoned_max_pain_sr',
                'strength': 0.75
            })
    
    return migration_levels
```

#### **Pattern C: OI Flow Velocity S&R**
```
OI FLOW VELOCITY CREATES DYNAMIC LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Time     Strike   CE OI Change   PE OI Change   Result
10:00    20000    +5000          -2000          → Building Resistance
10:05    20000    +8000          -1000          → Stronger Resistance
10:10    20000    +12000         +500           → MAJOR RESISTANCE FORMED

10:00    19900    -1000          +6000          → Building Support  
10:05    19900    -500           +9000          → Stronger Support
10:10    19900    +200           +15000         → MAJOR SUPPORT FORMED

PATTERN: Rapid OI accumulation at specific strikes = Future S&R levels
```

### 2. ROLLING STRADDLE ADVANCED PATTERNS

#### **Pattern D: Triple Straddle Divergence Levels**
```
WHEN STRADDLES DIVERGE → S&R FORMS AT DIVERGENCE POINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Time    ATM Straddle   ITM1 Straddle   OTM1 Straddle   Pattern
9:30    300            305             305             → Aligned
9:45    310            315             308             → Starting to diverge
10:00   320            330             310             → DIVERGENCE!
10:15   315            325             312             → ITM1 leads (Bullish)

S&R FORMATION:
• ATM straddle at 320 = Resistance (reversal point)
• ITM1 straddle at 330 = Extended target
• OTM1 straddle at 310 = Support (lagging indicator)
```

#### **Pattern E: Straddle Momentum Exhaustion Levels**
```python
def detect_straddle_momentum_exhaustion_sr(self, straddle_data):
    """
    When straddle momentum exhausts → Strong S&R forms
    """
    # Calculate straddle price momentum
    atm_momentum = calculate_momentum(straddle_data['atm_prices'], period=10)
    
    exhaustion_levels = []
    
    # Detect momentum exhaustion patterns
    for i in range(20, len(atm_momentum)):
        curr_momentum = atm_momentum[i]
        prev_momentum = atm_momentum[i-5:i].mean()
        
        # Momentum exhaustion conditions
        if prev_momentum > 0.5 and curr_momentum < 0.1:
            # Bullish exhaustion → Resistance
            exhaustion_levels.append({
                'price': straddle_data['atm_prices'][i],
                'type': 'resistance',
                'pattern': 'bullish_exhaustion'
            })
        elif prev_momentum < -0.5 and curr_momentum > -0.1:
            # Bearish exhaustion → Support
            exhaustion_levels.append({
                'price': straddle_data['atm_prices'][i],
                'type': 'support',
                'pattern': 'bearish_exhaustion'
            })
    
    return exhaustion_levels
```

### 3. CROSS-COMPONENT SYNERGY PATTERNS

#### **Pattern F: Greeks + OI Confluence S&R**
```
COMPONENT 2 (GREEKS) + COMPONENT 3 (OI) = POWERFUL S&R
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strike   Gamma    Delta    OI CE    OI PE    Result
19900    Low      0.3      50K      30K      -
19950    Medium   0.4      80K      60K      -
20000    HIGH     0.5      150K     150K     ← MAJOR S&R (Gamma + OI Peak)
20050    Medium   0.6      80K      60K      -
20100    Low      0.7      50K      30K      -

CONFLUENCE PATTERN:
• High Gamma + High OI = Pin Risk S&R Level
• Greeks sentiment shift + OI wall = Strong reversal level
```

#### **Pattern G: IV Skew Asymmetry S&R**
```python
def detect_iv_skew_asymmetry_sr(self, iv_data):
    """
    Component 4 IV Skew creates S&R at asymmetry extremes
    """
    skew_levels = []
    
    for strike in iv_data['strikes']:
        put_iv = iv_data['put_iv'][strike]
        call_iv = iv_data['call_iv'][strike]
        skew = put_iv - call_iv
        
        # Extreme put skew = Support forming
        if skew > iv_data['skew_90th_percentile']:
            skew_levels.append({
                'strike': strike,
                'type': 'support',
                'strength': min(1.0, skew / iv_data['skew_99th_percentile']),
                'pattern': 'extreme_put_skew'
            })
        
        # Extreme call skew = Resistance forming
        elif skew < iv_data['skew_10th_percentile']:
            skew_levels.append({
                'strike': strike,
                'type': 'resistance',
                'strength': min(1.0, abs(skew) / abs(iv_data['skew_1st_percentile'])),
                'pattern': 'extreme_call_skew'
            })
    
    return skew_levels
```

#### **Pattern H: Multi-Timeframe CPR Straddle Confluence**
```
COMPONENT 5 CPR ON STRADDLES + UNDERLYING = CONFLUENCE S&R
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                Daily CPR    Weekly CPR    Monthly CPR
Underlying:     19950        19900         19850
ATM Straddle:   295          290           285
ITM1 Straddle:  300          295           290

CONFLUENCE ZONES:
• 290-295 straddle price → Major support (3 timeframes align)
• Corresponds to 19900-19950 underlying → Validated S&R zone
```

### 4. ADVANCED COMPOSITE PATTERNS

#### **Pattern I: Volume-Weighted OI Profile S&R**
```python
def calculate_volume_weighted_oi_sr(self, oi_data, volume_data):
    """
    Combine OI concentration with volume to find true S&R
    """
    vw_oi_levels = []
    
    for strike in oi_data['strikes']:
        ce_oi = oi_data['ce_oi'][strike]
        pe_oi = oi_data['pe_oi'][strike]
        ce_volume = volume_data['ce_volume'][strike]
        pe_volume = volume_data['pe_volume'][strike]
        
        # Volume-weighted OI score
        ce_score = (ce_oi * ce_volume) / (ce_oi + pe_oi + 1)
        pe_score = (pe_oi * pe_volume) / (ce_oi + pe_oi + 1)
        
        if ce_score > np.percentile(all_ce_scores, 85):
            vw_oi_levels.append({
                'strike': strike,
                'type': 'resistance',
                'strength': ce_score,
                'pattern': 'volume_weighted_ce_oi'
            })
        
        if pe_score > np.percentile(all_pe_scores, 85):
            vw_oi_levels.append({
                'strike': strike,
                'type': 'support',
                'strength': pe_score,
                'pattern': 'volume_weighted_pe_oi'
            })
    
    return vw_oi_levels
```

#### **Pattern J: Gamma Flip Zones**
```
GAMMA FLIP POINTS = CRITICAL S&R LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strike    Net Gamma    MM Position    S&R Impact
19900     -500         Short          → Accelerates down
19950     -200         Short          → Slows descent
20000     0            FLIP POINT     → MAJOR S&R LEVEL
20050     +200         Long           → Slows ascent
20100     +500         Long           → Accelerates up

PATTERN: Zero gamma line = Natural S&R pivot point
```

#### **Pattern K: Correlation Breakdown S&R (Component 6)**
```python
def detect_correlation_breakdown_sr(self, correlation_data):
    """
    When correlations break → S&R levels form at breakdown points
    """
    breakdown_levels = []
    
    # Component 6 correlation matrices
    ce_pe_corr = correlation_data['ce_pe_correlation']
    straddle_underlying_corr = correlation_data['straddle_underlying_correlation']
    
    for i in range(20, len(ce_pe_corr)):
        # Detect correlation breakdown
        if ce_pe_corr[i-20:i].mean() > 0.7 and ce_pe_corr[i] < 0.3:
            # Correlation breakdown = potential reversal S&R
            breakdown_levels.append({
                'index': i,
                'price': correlation_data['price_at_index'][i],
                'type': 'reversal_sr',
                'pattern': 'correlation_breakdown',
                'strength': 0.8
            })
    
    return breakdown_levels
```

## 🔧 INTEGRATION WITH COMPONENT 7

### Enhanced Feature Engineering (72 → 120+ features)

```python
class EnhancedSupportResistanceFeatureEngine:
    """
    Expanded S&R detection with all patterns
    """
    
    def extract_comprehensive_sr_features(self, all_data):
        features = {}
        
        # Original Component 7 features (72)
        features.update(self.extract_base_sr_features(all_data))
        
        # New OI-based features (20)
        features.update(self.extract_oi_concentration_features(all_data))
        features.update(self.extract_max_pain_migration_features(all_data))
        features.update(self.extract_oi_flow_velocity_features(all_data))
        
        # New straddle pattern features (15)
        features.update(self.extract_straddle_divergence_features(all_data))
        features.update(self.extract_straddle_exhaustion_features(all_data))
        
        # Cross-component features (20)
        features.update(self.extract_greeks_oi_confluence_features(all_data))
        features.update(self.extract_iv_skew_sr_features(all_data))
        features.update(self.extract_multi_cpr_confluence_features(all_data))
        
        # Advanced composite features (15)
        features.update(self.extract_gamma_flip_features(all_data))
        features.update(self.extract_correlation_breakdown_features(all_data))
        features.update(self.extract_volume_weighted_oi_features(all_data))
        
        return features  # Now 120+ total features
```

## 📊 PRIORITY RANKING OF NEW PATTERNS

### High Priority (Implement First):
1. **OI Concentration Walls** - Most reliable for Indian markets
2. **Max Pain Migration** - Dynamic and adaptive
3. **Triple Straddle Divergence** - Unique to our approach
4. **Greeks + OI Confluence** - Cross-validation power

### Medium Priority:
5. **IV Skew Asymmetry S&R** - Good for volatility regimes
6. **Gamma Flip Zones** - Critical for expiry days
7. **Volume-Weighted OI Profile** - Enhanced accuracy
8. **Multi-Timeframe CPR Confluence** - Robust validation

### Lower Priority:
9. **Correlation Breakdown S&R** - Complex but powerful
10. **Straddle Momentum Exhaustion** - Timing-sensitive
11. **OI Flow Velocity** - Requires tick data

## 💡 KEY INSIGHTS

1. **OI is crucial for S&R**: High OI concentrations act as magnets
2. **Rolling straddles need dynamic adjustment**: As ATM changes, levels shift
3. **Cross-component validation**: Strongest S&R comes from multiple confirmations
4. **Time-sensitive patterns**: Some patterns only work near expiry (gamma flip)
5. **Volume validation**: Always confirm OI levels with volume

## 🚀 IMPLEMENTATION RECOMMENDATION

Start with OI Concentration Walls and Max Pain Migration as they provide the most reliable additional S&R levels. Then layer in the cross-component patterns for validation. This will transform Component 7 from 72 features to 120+ features, significantly improving S&R detection accuracy.