# **6. Technical Architecture**

## **6.1 Modern Data Pipeline Architecture**

### **Excel → YAML → Parquet → Arrow → GPU Flow**
```mermaid
graph TD
    A[Excel Configuration Files] --> B[YAML Converter]
    B --> C[YAML Configuration]
    C --> D[Parquet Data Processor]
    D --> E[Arrow Format Optimizer]
    E --> F[GPU Processing Engine]
    F --> G[8-Component Analysis]
    G --> H[Vertex AI Enhancement]
    H --> I[8-Regime Classification]
    I --> J[Real-time Results API]
```

### **Component Processing Architecture**
```mermaid
graph TD
    A[Modern Data Pipeline] --> B[Parallel Component Processing]
    
    B --> C1[Component 1: Rolling Straddle<br/>120 features, <150ms]
    B --> C2[Component 2: Greeks Sentiment<br/>98 features, <120ms]
    B --> C3[Component 3: OI-PA Trending<br/>105 features, <200ms]
    B --> C4[Component 4: IV Skew<br/>87 features, <200ms]
    B --> C5[Component 5: ATR-EMA-CPR<br/>94 features, <200ms]
    B --> C6[Component 6: Correlation<br/>150 features, <180ms]
    B --> C7[Component 7: Support/Resistance<br/>72 features, <150ms]
    
    C1 --> ALO[Adaptive Learning Orchestrator]
    C2 --> ALO
    C3 --> ALO
    C4 --> ALO
    C5 --> ALO
    C6 --> ALO
    C7 --> ALO
    
    ALO --> C8[Component 8: Master Integration<br/>48 features, <100ms]
    ALO --> VX[Vertex AI ML Pipeline]
    VX --> C8
    
    C8 --> API[Enhanced API v2]
    API --> UI[Trading Dashboard]
```

## **6.2 Implementation Architecture**

### **Directory Structure Enhancement**
```
/Users/maruth/projects/market_regime/vertex_market_regime/
├── configs/
│   ├── excel/                          # ✅ EXISTS - Excel bridge files
│   │   ├── excel_parser.py            # 🔴 NEEDS FULL IMPLEMENTATION
│   │   └── *.xlsx                     # ✅ EXISTS - Configuration files
│   └── yaml/                          # 🔴 NEEDS IMPLEMENTATION - YAML output
│
├── src/
│   ├── data/                          # 🔴 CRITICAL - Modern pipeline needed
│   │   ├── parquet_processor.py      # 🔴 NOT IMPLEMENTED
│   │   ├── arrow_optimizer.py        # 🔴 NOT IMPLEMENTED
│   │   └── pipeline_orchestrator.py  # 🔴 NOT IMPLEMENTED
│   │
│   ├── components/                    # 🔴 CRITICAL - Full implementation needed
│   │   ├── component_01_triple_straddle/  # 🔴 STUB ONLY
│   │   ├── component_02_greeks_sentiment/ # 🔴 CRITICAL gamma fix needed
│   │   ├── component_03_oi_pa_trending/   # 🔴 STUB ONLY
│   │   ├── component_04_iv_skew/          # 🔴 STUB ONLY
│   │   ├── component_05_atr_ema_cpr/      # 🔴 STUB ONLY
│   │   ├── component_06_correlation/      # 🔴 STUB ONLY
│   │   ├── component_07_support_resistance/ # 🔴 STUB ONLY
│   │   └── component_08_master_integration/ # 🔴 STUB ONLY
│   │
│   ├── cloud/                         # 🔴 NEEDS VERTEX AI IMPLEMENTATION
│   │   ├── vertex_ai_client.py       # 🔴 NOT IMPLEMENTED
│   │   ├── bigquery_client.py        # 🔴 NOT IMPLEMENTED
│   │   └── model_serving.py          # 🔴 NOT IMPLEMENTED
│   │
│   └── ml/                            # 🔴 NEEDS FULL IMPLEMENTATION
│       ├── adaptive_learning.py      # 🔴 NOT IMPLEMENTED
│       ├── feature_engineering.py    # 🔴 NOT IMPLEMENTED
│       └── model_training.py         # 🔴 NOT IMPLEMENTED
```

---
