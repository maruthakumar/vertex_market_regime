# **3. Product Requirements**

## Component Specification Links
- [Component 1: Triple Rolling Straddle](market_regime/mr_tripple_rolling_straddle_component1.md)
- [Component 2: Greeks Sentiment Analysis](market_regime/mr_greeks_sentiment_analysis_component2.md)
- [Component 3: OI-PA Trending Analysis](market_regime/mr_oi_pa_trending_analysis_component3.md)
- [Component 4: IV Skew Analysis](market_regime/mr_iv_skew_analysis_component4.md)
- [Component 5: ATR-EMA-CPR Integration](market_regime/mr_atr_ema_cpr_component5.md)
- [Component 6: Correlation & Non-Correlation Framework](market_regime/mr_correlation_noncorelation_component6.md)
- [Component 7: Support/Resistance Formation Logic](market_regime/mr_support_resistance_component7.md)
- [Component 8: DTE-Adaptive Master Integration](market_regime/mr_dte_adaptive_overlay_component8.md)

## **3.1 Core Functional Requirements**

### **R1: Modern Data Pipeline Implementation** (HIGH PRIORITY)
- **Excel → YAML Conversion**: Automatic conversion of 600+ Excel parameters to YAML configurations
- **YAML → Parquet Processing**: Efficient data processing pipeline using Parquet format
- **Arrow Integration**: GPU-optimized Arrow format for high-performance computing
- **Vertex AI Pipeline**: Integration with Google Vertex AI for ML processing

### **R1b: Vertex AI Feature Engineering (All Components) - REQUIRED**
- All eight components’ features (total 774) must be engineered via Vertex AI Pipelines
- Features must be stored and served via Vertex AI Feature Store with strict training/serving parity
- Data source: GCS Parquet; processing: Apache Arrow + RAPIDS cuDF
- Enforce schema versioning and lineage for features used in training and inference

### **R2: 8-Component Adaptive Learning System** (HIGH PRIORITY)  
- **Component 1**: Revolutionary rolling straddle overlay with EMA/VWAP/Pivot analysis
- **Component 2**: Greeks sentiment with CRITICAL gamma_weight=1.5 fix
- **Component 3**: OI-PA trending with cumulative ATM ±7 strikes analysis
- **Component 4**: IV skew analysis with dual DTE framework
- **Component 5**: ATR-EMA-CPR integration with dual asset analysis
- **Component 6**: Ultra-comprehensive 30x30 correlation framework
- **Component 7**: Dynamic support/resistance logic with multi-method confluence
- **Component 8**: DTE-adaptive master integration with 8-regime classification

### **R3: Performance Enhancement** (HIGH PRIORITY)
- **Processing Time**: <800ms total analysis (73% improvement)
- **Memory Usage**: <3.7GB total system memory
- **Feature Engineering**: 774 expert-optimized features
- **Real-time Processing**: Concurrent component processing

### **R4: Vertex AI Integration** (MEDIUM PRIORITY)
- **Model Training**: Automated ML model training on historical data
- **Real-time Inference**: Low-latency prediction serving
- **Adaptive Learning**: Continuous model improvement
- **Hyperparameter Optimization**: Automatic parameter tuning

## **3.2 Non-Functional Requirements**

### **R5: Backward Compatibility** (CRITICAL)
- **API Compatibility**: 100% compatibility with existing v1 endpoints
- **Configuration Preservation**: All Excel configurations maintained
- **Data Format**: Existing data formats supported during transition
- **Deployment**: Gradual migration with rollback capability

### **R6: Scalability & Reliability** 
- **Auto-scaling**: Automatic resource scaling based on demand
- **Fault Tolerance**: Graceful degradation on component failures
- **Monitoring**: Comprehensive system health monitoring
- **Alerting**: Proactive issue detection and notification

---
