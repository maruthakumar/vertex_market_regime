"""
Component 5: ATR-EMA-CPR Dual-Asset Analysis System

Complete implementation of Component 5 for the Market Regime Master Framework,
providing sophisticated volatility-trend-pivot analysis through dual-asset 
ATR-EMA-CPR analysis with comprehensive DTE framework integration.

Key Features:
- Dual-asset analysis: Rolling straddle prices AND underlying prices
- Multi-timeframe analysis: Daily, weekly, monthly underlying analysis
- Cross-asset validation: Trend, volatility, and level validation
- Dual DTE framework: Specific DTE and DTE range analysis
- Enhanced regime classification: 8-regime system with institutional flow detection
- Production performance: <200ms processing, <500MB memory, 94 features
"""

# Main component analyzer
from .component_05_analyzer import (
    Component05Analyzer,
    Component05IntegrationResult, 
    Component05PerformanceMetrics,
    Component05PerformanceMonitor
)

# Core analysis engines
from .dual_asset_data_extractor import (
    DualAssetDataExtractor,
    DualAssetExtractionResult,
    StraddlePriceData,
    UnderlyingPriceData
)

from .straddle_atr_ema_cpr_engine import (
    StraddleATREMACPREngine,
    StraddleAnalysisResult,
    StraddleATRResult,
    StraddleEMAResult,
    StraddleCPRResult
)

from .underlying_atr_ema_cpr_engine import (
    UnderlyingATREMACPREngine,
    UnderlyingAnalysisResult,
    UnderlyingATRResult,
    UnderlyingEMAResult,
    UnderlyingCPRResult
)

# Advanced frameworks
from .dual_dte_framework import (
    DualDTEFramework,
    DTEIntegratedResult,
    DTESpecificAnalysis,
    DTERangeAnalysis
)

from .cross_asset_integration import (
    CrossAssetIntegrationEngine,
    CrossAssetIntegrationResult,
    TrendDirectionValidation,
    VolatilityRegimeValidation,
    SupportResistanceValidation,
    CrossAssetConfidenceResult,
    DynamicWeightingResult
)

from .enhanced_regime_classifier import (
    EnhancedRegimeClassificationEngine,
    EnhancedRegimeClassificationResult,
    RegimeFeatures,
    RegimeTransitionResult,
    InstitutionalFlowResult,
    MultiLayeredConfidenceResult
)

# Component metadata
COMPONENT_INFO = {
    'id': 5,
    'name': 'ATR-EMA-CPR Dual-Asset Analysis',
    'version': '1.0.0',
    'description': 'Sophisticated volatility-trend-pivot analysis using dual-asset ATR-EMA-CPR methodology',
    'features': 94,
    'performance_budget_ms': 200,
    'memory_budget_mb': 500,
    'regimes': 8,
    'analysis_types': [
        'straddle_atr_ema_cpr',
        'underlying_multi_timeframe',
        'cross_asset_validation',
        'dual_dte_framework',
        'enhanced_regime_classification'
    ],
    'key_capabilities': [
        'Dual-asset analysis (straddle + underlying)',
        'Multi-timeframe underlying analysis (daily/weekly/monthly)',
        'Cross-asset validation and integration',
        'Dual DTE framework (specific + range)',
        'Institutional flow detection',
        'Multi-layered confidence scoring',
        '8-regime enhanced classification',
        'Production performance optimization'
    ]
}

# Export main analyzer as default
__all__ = [
    # Main component
    'Component05Analyzer',
    'COMPONENT_INFO',
    
    # Results and metrics
    'Component05IntegrationResult',
    'Component05PerformanceMetrics', 
    'Component05PerformanceMonitor',
    
    # Data extraction
    'DualAssetDataExtractor',
    'DualAssetExtractionResult',
    'StraddlePriceData',
    'UnderlyingPriceData',
    
    # Analysis engines
    'StraddleATREMACPREngine',
    'UnderlyingATREMACPREngine',
    'StraddleAnalysisResult',
    'UnderlyingAnalysisResult',
    
    # Advanced frameworks
    'DualDTEFramework',
    'CrossAssetIntegrationEngine',
    'EnhancedRegimeClassificationEngine',
    
    # Framework results
    'DTEIntegratedResult',
    'CrossAssetIntegrationResult',
    'EnhancedRegimeClassificationResult',
    
    # Analysis components
    'StraddleATRResult',
    'StraddleEMAResult',
    'StraddleCPRResult',
    'UnderlyingATRResult',
    'UnderlyingEMAResult',
    'UnderlyingCPRResult',
    
    # Advanced results
    'DTESpecificAnalysis',
    'DTERangeAnalysis',
    'TrendDirectionValidation',
    'VolatilityRegimeValidation',
    'SupportResistanceValidation',
    'CrossAssetConfidenceResult',
    'DynamicWeightingResult',
    'RegimeFeatures',
    'RegimeTransitionResult',
    'InstitutionalFlowResult',
    'MultiLayeredConfidenceResult'
]

# Default component instance creation function
def create_component_05(config: dict = None) -> Component05Analyzer:
    """
    Create Component 5 ATR-EMA-CPR Dual-Asset Analysis instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Component05Analyzer instance
    """
    if config is None:
        config = {
            'component_id': 5,
            'feature_count': 94,
            'processing_budget_ms': 200,
            'memory_budget_mb': 500,
            'gpu_enabled': False,
            'learning_enabled': True,
            'fallback_enabled': True,
            'error_recovery_enabled': True
        }
    
    return Component05Analyzer(config)


# Component registration for factory
def register_component():
    """Register Component 5 with the component factory"""
    from ..base_component import ComponentFactory
    ComponentFactory.register_component(5, Component05Analyzer)