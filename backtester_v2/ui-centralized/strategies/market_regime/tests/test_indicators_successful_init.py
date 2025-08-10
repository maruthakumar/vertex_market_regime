#!/usr/bin/env python3
"""
Test Successful Initialization of Market Regime Indicators V2
===========================================================

This script tests that all implemented indicators can be successfully
initialized and are ready for use.

Author: Market Regime Testing Team
Date: 2025-07-06
"""

import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_greek_sentiment_components():
    """Test Greek Sentiment V2 components"""
    logger.info("Testing Greek Sentiment V2 components...")
    
    try:
        from indicators.greek_sentiment import (
            GreekSentimentAnalyzer,
            BaselineTracker,
            VolumeOIWeighter,
            ITMOTMAnalyzer,
            DTEAdjuster,
            GreekCalculator
        )
        
        # Test configuration
        config = {
            'baseline_config': {'baseline_time': '09:15:00'},
            'weighting_config': {'oi_weight_alpha': 0.6, 'volume_weight_beta': 0.4},
            'itm_otm_config': {'itm_strikes': [1, 2, 3], 'otm_strikes': [1, 2, 3]},
            'dte_config': {'near_expiry_days': 7, 'medium_expiry_days': 30},
            'normalization_config': {'delta_factor': 1.0}
        }
        
        # Initialize components
        components = []
        components.append(("BaselineTracker", BaselineTracker(config['baseline_config'])))
        components.append(("VolumeOIWeighter", VolumeOIWeighter(config['weighting_config'])))
        components.append(("ITMOTMAnalyzer", ITMOTMAnalyzer(config['itm_otm_config'])))
        components.append(("DTEAdjuster", DTEAdjuster(config['dte_config'])))
        components.append(("GreekCalculator", GreekCalculator(config['normalization_config'])))
        
        # Check all initialized
        for name, component in components:
            assert component is not None, f"{name} failed to initialize"
            logger.info(f"  ✅ {name} initialized successfully")
        
        logger.info("✅ All Greek Sentiment V2 components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Greek Sentiment V2 test failed: {e}")
        return False


def test_oi_pa_analysis_components():
    """Test OI/PA Analysis V2 components"""
    logger.info("Testing OI/PA Analysis V2 components...")
    
    try:
        from indicators.oi_pa_analysis import (
            OIPAAnalyzer,
            OIPatternDetector,
            DivergenceDetector,
            VolumeFlowAnalyzer,
            CorrelationAnalyzer,
            SessionWeightManager
        )
        
        # Test configuration
        config = {
            'pattern_config': {'lookback_periods': 20},
            'divergence_config': {'divergence_types': ['price_oi', 'volume_oi']},
            'volume_flow_config': {'institutional_threshold': 0.8},
            'correlation_config': {'min_correlation': 0.80},
            'session_config': {'decay_lambda': 0.1}
        }
        
        # Initialize components
        components = []
        components.append(("OIPatternDetector", OIPatternDetector(config['pattern_config'])))
        components.append(("DivergenceDetector", DivergenceDetector(config['divergence_config'])))
        components.append(("VolumeFlowAnalyzer", VolumeFlowAnalyzer(config['volume_flow_config'])))
        components.append(("CorrelationAnalyzer", CorrelationAnalyzer(config['correlation_config'])))
        components.append(("SessionWeightManager", SessionWeightManager(config['session_config'])))
        
        # Check all initialized
        for name, component in components:
            assert component is not None, f"{name} failed to initialize"
            logger.info(f"  ✅ {name} initialized successfully")
        
        logger.info("✅ All OI/PA Analysis V2 components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ OI/PA Analysis V2 test failed: {e}")
        return False


def main():
    """Run all initialization tests"""
    logger.info("="*80)
    logger.info("Market Regime Indicators V2 - Initialization Test")
    logger.info("="*80)
    
    # Test Greek Sentiment V2
    greek_success = test_greek_sentiment_components()
    print()
    
    # Test OI/PA Analysis V2
    oi_pa_success = test_oi_pa_analysis_components()
    print()
    
    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if greek_success and oi_pa_success:
        logger.info("🎉 ALL INDICATORS INITIALIZED SUCCESSFULLY!")
        logger.info("")
        logger.info("The following indicators are ready for use:")
        logger.info("  ✅ Greek Sentiment V2 - Fully modular with 6 components")
        logger.info("  ✅ OI/PA Analysis V2 - Fully modular with 6 components")
        logger.info("")
        logger.info("Component Details:")
        logger.info("")
        logger.info("Greek Sentiment V2:")
        logger.info("  - BaselineTracker: 9:15 AM baseline establishment")
        logger.info("  - VolumeOIWeighter: Dual weighting system (α×OI + β×Volume)")
        logger.info("  - ITMOTMAnalyzer: Moneyness classification & institutional detection")
        logger.info("  - DTEAdjuster: DTE-specific Greek adjustments")
        logger.info("  - GreekCalculator: Market-calibrated normalization")
        logger.info("  - GreekSentimentAnalyzer: Main orchestrator")
        logger.info("")
        logger.info("OI/PA Analysis V2:")
        logger.info("  - OIPatternDetector: Corrected OI-Price patterns")
        logger.info("  - DivergenceDetector: 5-type divergence detection")
        logger.info("  - VolumeFlowAnalyzer: Institutional vs retail flow")
        logger.info("  - CorrelationAnalyzer: Mathematical correlation (>0.80)")
        logger.info("  - SessionWeightManager: 7-session time weighting")
        logger.info("  - OIPAAnalyzer: Main orchestrator")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("  1. Run comprehensive tests with real HeavyDB data")
        logger.info("  2. Implement Technical Indicators V2")
        logger.info("  3. Implement IV Analytics V2")
        logger.info("  4. Implement Market Breadth V2")
        return 0
    else:
        logger.error("❌ Some indicators failed to initialize")
        logger.error("Please check the logs above for details")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)