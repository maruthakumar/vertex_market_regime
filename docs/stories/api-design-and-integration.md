# API Design and Integration

## API Integration Strategy

**API Integration Strategy:** Extension pattern - new ML-enhanced endpoints added alongside existing API framework with version-controlled progressive enhancement

**Authentication:** Leverage existing API key and session management system used by backtester_v2

**Versioning:** Semantic versioning with backward compatibility (/api/v1/ maintained, /api/v2/ for enhanced features)

## New API Endpoints

### **Enhanced Regime Analysis**
- **Method:** POST
- **Endpoint:** `/api/v2/regime/analyze`
- **Purpose:** 8-component adaptive learning analysis with ML enhancement
- **Integration:** Extends existing regime analysis with component-level insights

**Request:**
```json
{
  "symbol": "NIFTY",
  "timestamp": "2025-08-10T14:30:00Z",
  "dte_filter": [0, 7, 30],
  "component_weights": {
    "triple_straddle": 0.15,
    "greeks_sentiment": 0.14,
    "oi_pa_trending": 0.13,
    "iv_skew": 0.12,
    "atr_ema_cpr": 0.11,
    "correlation": 0.10,
    "support_resistance": 0.10,
    "master_integration": 0.15
  },
  "use_adaptive_weights": true,
  "enable_ml_enhancement": true
}
```

**Response:**
```json
{
  "analysis_id": "uuid-12345",
  "symbol": "NIFTY",
  "timestamp": "2025-08-10T14:30:03.250Z",
  "master_regime": "VCPE",
  "regime_name": "Volatility Contraction Price Expansion",
  "master_confidence": 0.89,
  "processing_time_ms": 750,
  "component_results": {
    "triple_straddle": {
      "regime_contribution": "BULLISH_EXPANSION",
      "confidence": 0.85,
      "adaptive_weight": 0.16,
      "correlation_matrix": "10x10_matrix_data"
    },
    "greeks_sentiment": {
      "sentiment_level": "MODERATELY_BULLISH", 
      "confidence": 0.78,
      "delta_sentiment": 0.65,
      "gamma_exposure": 1.23
    }
  },
  "ml_enhancement": {
    "vertex_ai_prediction": "VCPE",
    "ml_confidence": 0.91,
    "model_version": "mr-adaptive-v1.2.3",
    "feature_importance": "top_10_features"
  },
  "system_health": {
    "component_agreement": 0.82,
    "correlation_breakdown_alerts": [],
    "performance_within_targets": true
  }
}
```

### **Adaptive Weight Management**
- **Method:** PUT  
- **Endpoint:** `/api/v2/regime/weights`
- **Purpose:** Update and monitor adaptive weight evolution across all components
- **Integration:** New capability for real-time weight optimization

**Request:**
```json
{
  "component_id": "triple_straddle",
  "dte_bucket": "0-7",
  "performance_feedback": {
    "actual_regime": "VCPE",
    "predicted_regime": "VCPE", 
    "accuracy_score": 0.95,
    "timing_accuracy_minutes": 3.2
  },
  "update_strategy": "exponential_smoothing"
}
```

**Response:**
```json
{
  "component_id": "triple_straddle",
  "old_weight": 0.15,
  "new_weight": 0.162,
  "weight_change": 0.012,
  "performance_improvement": 0.03,
  "update_timestamp": "2025-08-10T14:35:00Z",
  "learning_convergence": "improving"
}
```

### **Component Health Monitoring**
- **Method:** GET
- **Endpoint:** `/api/v2/regime/health`  
- **Purpose:** Real-time monitoring of all 8 components and system health
- **Integration:** New monitoring capability for system reliability

**Response:**
```json
{
  "system_status": "HEALTHY",
  "total_processing_time_ms": 750,
  "memory_usage_gb": 2.1,
  "component_status": {
    "triple_straddle": {
      "status": "HEALTHY",
      "last_update": "2025-08-10T14:30:00Z",
      "processing_time_ms": 140,
      "accuracy_7d": 0.87
    },
    "correlation_framework": {
      "status": "WARNING", 
      "issue": "correlation_breakdown_detected",
      "affected_pairs": ["greeks-oi_pa", "iv_skew-atr"],
      "processing_time_ms": 185
    }
  },
  "adaptive_learning": {
    "weights_updated_last_hour": 12,
    "convergence_status": "stable",
    "market_structure_changes": 0
  },
  "vertex_ai_integration": {
    "model_serving_status": "online",
    "prediction_latency_ms": 45,
    "model_drift_detected": false
  }
}
```
