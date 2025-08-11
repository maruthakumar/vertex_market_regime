# Data Models and Schema Changes

## New Data Models

### **ComponentAnalysisResult**
**Purpose:** Store results from individual component analysis for adaptive learning feedback
**Integration:** New table in HeavyDB with foreign key relationships to existing options_chain data

**Key Attributes:**
- `analysis_id`: UUID - Unique identifier for each analysis run
- `component_id`: INT - Component identifier (1-8)
- `timestamp`: TIMESTAMP - Analysis execution time
- `regime_prediction`: STRING - Individual component regime prediction
- `confidence_score`: FLOAT - Component confidence (0.0-1.0)
- `processing_time_ms`: INT - Component processing time
- `weight_factor`: FLOAT - Current adaptive weight for this component

**Relationships:**
- **With Existing:** Links to existing options_chain data via trade_date and symbol
- **With New:** Aggregated by MasterRegimeAnalysis for final classification

### **AdaptiveLearningWeights**
**Purpose:** Store historical weight evolution for each component's adaptive learning system
**Integration:** Time-series data for weight optimization and performance tracking

**Key Attributes:**
- `weight_id`: UUID - Unique weight record identifier
- `component_id`: INT - Component being weighted
- `dte_bucket`: STRING - DTE range (0-7, 8-30, 31+)
- `regime_context`: STRING - Market regime when weight was applied
- `weight_value`: FLOAT - Adaptive weight value
- `performance_metric`: FLOAT - Historical performance score
- `last_updated`: TIMESTAMP - Weight update timestamp

**Relationships:**
- **With Existing:** Performance metrics calculated against actual market moves
- **With New:** Used by ComponentAnalysisResult for real-time weight application

### **MasterRegimeAnalysis**
**Purpose:** Final integrated analysis results from all 8 components with master classification
**Integration:** Primary results table for the enhanced 8-component system

**Key Attributes:**
- `analysis_id`: UUID - Links to individual component analyses
- `symbol`: STRING - Asset symbol (NIFTY, BANKNIFTY, etc.)
- `timestamp`: TIMESTAMP - Analysis timestamp
- `master_regime`: STRING - Final regime classification (LVLD, HVC, VCPE, etc.)
- `master_confidence`: FLOAT - Overall system confidence
- `component_agreement`: FLOAT - Inter-component correlation score
- `processing_time_total_ms`: INT - Total 8-component processing time
- `vertex_ai_model_version`: STRING - ML model version used

**Relationships:**
- **With Existing:** Replaces/enhances current regime classification in trading decisions
- **With New:** Aggregates all ComponentAnalysisResult records per analysis cycle

## Schema Integration Strategy

**Database Changes Required:**
- **New Tables:** component_analysis_results, adaptive_learning_weights, master_regime_analysis, ml_model_metadata
- **Modified Tables:** Enhanced options_chain with regime_analysis_id foreign key (optional)
- **New Indexes:** Time-based indexes on component results, symbol-based clustering for performance
- **Migration Strategy:** Progressive schema deployment with fallback compatibility

**Backward Compatibility:**
- All existing tables and schemas remain unchanged
- New tables use separate namespace to avoid conflicts  
- Existing regime classification API maintains current response format with optional enhancement fields
