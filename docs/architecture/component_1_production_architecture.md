# Component 1 Production Architecture - Rolling Straddle Feature Engineering

## Architecture Overview

```mermaid
graph TB
    subgraph "Production Data Source"
        GCS[("Google Cloud Storage<br/>6.76 GB Parquet<br/>49 columns, 1-min timeseries")]
        LOCAL[("Local Production Dataset<br/>/data/nifty_validation/<br/>87 Parquet files")]
    end
    
    subgraph "Data Loading Layer"
        PARQUET["Parquet Loader<br/>49-column schema validation<br/>Multi-expiry handling"]
        ARROW["Apache Arrow<br/>Memory mapping<br/>Zero-copy operations"]
    end
    
    subgraph "Rolling Straddle Engine"
        TIME["Time-Series Processor<br/>Minute-by-minute rolling<br/>Dynamic strike selection"]
        FILTER["Strike Type Filter<br/>call_strike_type/put_strike_type<br/>ATM, ITM1, OTM1 classification"]
        CALC["Straddle Calculator<br/>ATM: CE+PE (ATM/ATM)<br/>ITM1: CE+PE (ITM1/OTM1)<br/>OTM1: CE+PE (OTM1/ITM1)"]
    end
    
    subgraph "Multi-Timeframe Processing"
        RESAMPLE["Timeframe Resampler<br/>1min → 3min, 5min, 10min, 15min<br/>OHLC generation"]
        SYNC["Temporal Synchronization<br/>Alignment across timeframes<br/>Missing data handling"]
    end
    
    subgraph "GPU Processing Layer"
        CUDF["RAPIDS cuDF<br/>GPU-accelerated operations<br/>Memory budget: <512MB"]
        CACHE["Cache System<br/>TTL controls<br/>Rolling window optimization"]
    end
    
    subgraph "Technical Analysis Engine"
        EMA["EMA Analysis<br/>20, 50, 100, 200 periods<br/>Applied to STRADDLE prices"]
        VWAP["VWAP Analysis<br/>Combined volume weighting<br/>ce_volume + pe_volume"]
        PIVOT["Pivot Analysis<br/>PP, R1-R3, S1-S3 on straddles<br/>CPR on underlying futures"]
    end
    
    subgraph "Feature Generation"
        WEIGHT["10-Component Weighting<br/>ATM/ITM1/OTM1 straddles<br/>Individual CE/PE components"]
        DTE["DTE-Specific Framework<br/>Weight adjustment by expiry<br/>Performance tracking"]
        FEATURES["120 Feature Vector<br/>All values [-1.0, 1.0]<br/>Component 1 output"]
    end
    
    subgraph "Validation & Testing"
        PARQUET_TEST["Parquet Production Tests<br/>80% testing effort<br/>8 expiry cycles validation"]
        CSV_TEST["CSV Cross-validation<br/>20% testing effort<br/>Reference data debugging"]
        PERF_TEST["Performance Validation<br/><150ms per file<br/><512MB memory usage"]
    end
    
    subgraph "Edge Case Handling"
        MISSING["Missing Data Handler<br/>Forward fill, interpolation<br/>Fallback strike logic"]
        EXPIRY["Multi-Expiry Logic<br/>Nearest DTE selection<br/>Rollover handling"]
        VOLUME["Volume Validation<br/>Zero volume detection<br/>Outlier filtering"]
    end
    
    %% Data Flow
    GCS --> PARQUET
    LOCAL --> PARQUET
    PARQUET --> ARROW
    ARROW --> TIME
    TIME --> FILTER
    FILTER --> CALC
    CALC --> RESAMPLE
    RESAMPLE --> SYNC
    SYNC --> CUDF
    CUDF --> CACHE
    CACHE --> EMA
    CACHE --> VWAP
    CACHE --> PIVOT
    EMA --> WEIGHT
    VWAP --> WEIGHT
    PIVOT --> WEIGHT
    WEIGHT --> DTE
    DTE --> FEATURES
    
    %% Edge Case Integration
    TIME --> MISSING
    FILTER --> EXPIRY
    CALC --> VOLUME
    MISSING --> CUDF
    EXPIRY --> CUDF
    VOLUME --> CUDF
    
    %% Testing Integration
    FEATURES --> PARQUET_TEST
    FEATURES --> CSV_TEST
    CUDF --> PERF_TEST
    
    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef gpu fill:#e8f5e8
    classDef output fill:#fff3e0
    classDef testing fill:#fce4ec
    
    class GCS,LOCAL dataSource
    class TIME,FILTER,CALC,RESAMPLE,SYNC processing
    class CUDF,CACHE gpu
    class FEATURES output
    class PARQUET_TEST,CSV_TEST,PERF_TEST testing
```

## Production Data Schema

### Parquet Schema (49 columns - Production Standard)
```
Core Identification:
├── trade_date (datetime64[ns])
├── trade_time (object) 
├── expiry_date (datetime64[ns])
├── index_name (object)

Market Data:
├── spot (float64)
├── atm_strike (float64) 
├── strike (int64)
├── dte (int64)

Strike Classification (KEY FOR ROLLING STRADDLE):
├── call_strike_type (object)  # ATM, ITM1-ITM32, OTM1-OTM32
├── put_strike_type (object)   # ATM, ITM1-ITM32, OTM1-OTM32

Call Options (14 columns):
├── ce_symbol, ce_open, ce_high, ce_low, ce_close
├── ce_volume, ce_oi, ce_coi, ce_iv
└── ce_delta, ce_gamma, ce_theta, ce_vega, ce_rho

Put Options (14 columns):
├── pe_symbol, pe_open, pe_high, pe_low, pe_close
├── pe_volume, pe_oi, pe_coi, pe_iv  
└── pe_delta, pe_gamma, pe_theta, pe_vega, pe_rho

Futures Data (7 columns):
├── future_open, future_high, future_low, future_close
└── future_volume, future_oi, future_coi

Metadata:
├── zone_id, zone_name
├── expiry_bucket
└── dte_bucket
```

## Rolling Straddle Implementation Logic

### Time-Series Rolling Mechanism
```python
def process_rolling_straddles(parquet_df):
    """
    Process minute-by-minute rolling straddle evolution
    """
    rolling_results = {}
    
    for timestamp in parquet_df['trade_time'].unique():
        minute_data = parquet_df[parquet_df['trade_time'] == timestamp]
        
        # Step 1: Use nearest expiry only (handle multiple expiries)
        nearest_expiry = minute_data.loc[minute_data['dte'].idxmin(), 'expiry_date']
        filtered_data = minute_data[minute_data['expiry_date'] == nearest_expiry]
        
        # Step 2: Extract rolling straddles using database classification
        straddles = {
            'atm': extract_straddle(filtered_data, 'ATM', 'ATM'),
            'itm1': extract_straddle(filtered_data, 'ITM1', 'OTM1'), 
            'otm1': extract_straddle(filtered_data, 'OTM1', 'ITM1')
        }
        
        # Step 3: Calculate combined volumes for VWAP
        for straddle_type, data in straddles.items():
            if data is not None:
                data['combined_volume'] = data['ce_volume'] + data['pe_volume']
                data['straddle_price'] = data['ce_close'] + data['pe_close']
        
        rolling_results[timestamp] = straddles
    
    return rolling_results

def extract_straddle(data, call_type, put_type):
    """Extract straddle data using strike type classification"""
    candidates = data[
        (data['call_strike_type'] == call_type) & 
        (data['put_strike_type'] == put_type)
    ]
    
    if len(candidates) > 0:
        return candidates.iloc[0].to_dict()
    else:
        # Fallback logic for missing strike types
        return handle_missing_strike_data(data, call_type, put_type)
```

## Performance Specifications

### Memory Budget
| Component | Budget | Usage |
|-----------|--------|-------|
| Parquet Loading | <128MB | Arrow memory mapping |
| Rolling Straddle Processing | <192MB | Time-series operations |
| Multi-timeframe Resampling | <96MB | OHLC generation |
| Technical Analysis | <96MB | EMA/VWAP/Pivot calculations |
| **Total Component 1** | **<512MB** | **Within 3.7GB system budget** |

### Processing Speed
| Operation | Target | Validation |
|-----------|--------|------------|
| Parquet file loading | <30ms | Per 8,537+ row file |
| Rolling straddle extraction | <50ms | Time-series processing |
| Multi-timeframe resampling | <40ms | 4 timeframes generation |
| Feature generation | <30ms | 120 features creation |
| **Total Component 1** | **<150ms** | **Per production file** |

## Testing Architecture

### Production Testing Hierarchy (Parquet Primary)
```
Production Testing (80% effort)
├── Parquet Pipeline Tests
│   ├── Multi-expiry file loading (8 expiry cycles)
│   ├── 49-column schema validation 
│   ├── Memory efficiency (<512MB)
│   └── Processing speed (<150ms per file)
├── Rolling Straddle Core Tests  
│   ├── Time-series rolling behavior
│   ├── Strike type classification usage
│   ├── Volume combination logic
│   └── Missing data edge cases
└── Feature Generation Tests
    ├── Exactly 120 features per timestamp
    ├── Value range validation [-1.0, 1.0]
    └── Multi-timeframe synchronization

Reference Testing (20% effort)  
├── CSV Cross-validation
│   ├── Parquet vs CSV consistency
│   ├── Same date result validation
│   └── 49-column vs 48-column mapping
└── Debug Support
    ├── Reference data verification
    └── Development troubleshooting
```

## Data Flow Summary

1. **Production Parquet** (87 files) → **Arrow Memory Mapping** → **GPU Processing**
2. **Time-Series Rolling** → **Strike Classification** → **Straddle Calculation** 
3. **Multi-Timeframe Resampling** → **Technical Analysis** → **Feature Generation**
4. **Edge Case Handling** → **Validation** → **120 Feature Output**
5. **Cross-Component Integration** → **Component 2 Handoff**

This architecture ensures Component 1 is optimized for the **actual production Parquet pipeline** with comprehensive validation using **real market data**.