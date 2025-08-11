"""
Component 3: OI-PA Trending Analysis

This component implements comprehensive Open Interest - Price Action (OI-PA) trending analysis
with cumulative ATM ±7 strikes methodology and institutional flow detection using production
Parquet data.

Key Features:
- Production OI data extraction with 99.98% coverage validation
- Cumulative multi-strike OI analysis across ATM ±7 strikes
- Institutional flow detection using volume-OI divergence analysis
- Complete option seller framework (CE/PE/Future patterns)
- 3-way correlation matrix (CE+PE+Future) analysis
- 8-regime comprehensive market classification system
- Component 6 integration for system-wide correlation intelligence

Performance Targets:
- Processing time: <200ms per component
- Memory usage: <300MB per component
- OI-PA trending accuracy: >85% using institutional flow methodology
- Integration with Components 1+2 framework using shared schema

As per Story 1.4 requirements:
- Extract ce_oi (column 20), pe_oi (column 34) from production Parquet with 99.98% coverage
- Integrate ce_volume (column 19), pe_volume (column 33) for institutional flow analysis
- Implement dynamic ATM ±7 strikes range using call_strike_type/put_strike_type columns
- Build multi-timeframe rollups (5min, 15min, 3min, 10min) with weighted analysis
- Create comprehensive market regime formation using complete CE/PE correlation matrix
"""

from .production_oi_extractor import ProductionOIExtractor
from .cumulative_multistrike_analyzer import (
    CumulativeMultiStrikeAnalyzer,
    CumulativeOIMetrics
)
from .institutional_flow_detector import (
    InstitutionalFlowDetector,
    InstitutionalFlowMetrics,
    InstitutionalFlowType,
    DivergenceType
)
from .oi_pa_trending_engine import (
    OIPATrendingEngine,
    OIPATrendingMetrics,
    CEOptionSellerPattern,
    PEOptionSellerPattern,
    FutureSellerPattern,
    ThreeWayCorrelationPattern,
    ComprehensiveMarketRegime
)

# Component version
__version__ = "1.0.0"

# Component metadata
COMPONENT_INFO = {
    "name": "Component 3: OI-PA Trending Analysis",
    "version": __version__,
    "description": "Open Interest - Price Action trending analysis with institutional flow detection",
    "features": [
        "Production OI data extraction (99.98% coverage)",
        "Cumulative ATM ±7 strikes analysis",
        "Institutional flow detection",
        "Option seller correlation framework",
        "3-way correlation matrix (CE+PE+Future)",
        "8-regime market classification",
        "Component 6 integration"
    ],
    "performance_targets": {
        "processing_time_ms": 200,
        "memory_usage_mb": 300,
        "accuracy_target": 0.85
    }
}

# Export main classes for component usage
__all__ = [
    # Main component classes
    'ProductionOIExtractor',
    'CumulativeMultiStrikeAnalyzer', 
    'InstitutionalFlowDetector',
    'OIPATrendingEngine',
    
    # Data classes
    'CumulativeOIMetrics',
    'InstitutionalFlowMetrics',
    'OIPATrendingMetrics',
    
    # Enums
    'InstitutionalFlowType',
    'DivergenceType',
    'CEOptionSellerPattern',
    'PEOptionSellerPattern', 
    'FutureSellerPattern',
    'ThreeWayCorrelationPattern',
    'ComprehensiveMarketRegime',
    
    # Component metadata
    'COMPONENT_INFO',
    '__version__'
]