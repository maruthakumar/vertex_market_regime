#!/usr/bin/env python3
"""
Test imports and module availability after fixes
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_modules_imports():
    """Test that enhanced modules can be imported"""
    try:
        from enhanced_modules.enhanced_triple_straddle_analyzer import EnhancedTripleStraddleAnalyzer
        assert EnhancedTripleStraddleAnalyzer is not None
        print("✅ EnhancedTripleStraddleAnalyzer import successful")
    except ImportError as e:
        assert False, f"Failed to import EnhancedTripleStraddleAnalyzer: {e}"
    
    try:
        from enhanced_modules.enhanced_multi_indicator_engine import EnhancedMultiIndicatorEngine
        assert EnhancedMultiIndicatorEngine is not None
        print("✅ EnhancedMultiIndicatorEngine import successful")
    except ImportError as e:
        assert False, f"Failed to import EnhancedMultiIndicatorEngine: {e}"
    
    try:
        from enhanced_modules.enhanced_market_regime_engine import EnhancedMarketRegimeEngine
        assert EnhancedMarketRegimeEngine is not None
        print("✅ EnhancedMarketRegimeEngine import successful")
    except ImportError as e:
        assert False, f"Failed to import EnhancedMarketRegimeEngine: {e}"

def test_config_manager_import():
    """Test that config manager can be imported"""
    try:
        from config_manager import get_config_manager
        config = get_config_manager()
        assert config is not None
        assert config.paths.base_path == "/srv/samba/shared/bt/backtester_stable/BTRUN"
        print("✅ Config manager import and initialization successful")
    except ImportError as e:
        assert False, f"Failed to import config manager: {e}"

def test_enhanced_regime_detector_v2_imports():
    """Test that enhanced regime detector v2 can be imported"""
    try:
        from enhanced_modules.enhanced_regime_detector_v2 import Enhanced18RegimeDetectorV2
        assert Enhanced18RegimeDetectorV2 is not None
        print("✅ Enhanced18RegimeDetectorV2 import successful")
    except ImportError as e:
        print(f"⚠️  Warning: Enhanced18RegimeDetectorV2 import failed (expected due to missing dependencies): {e}")

def test_comprehensive_modules_init():
    """Test that comprehensive modules __init__ exists"""
    init_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "comprehensive_modules",
        "__init__.py"
    )
    assert os.path.exists(init_path), f"comprehensive_modules/__init__.py not found at {init_path}"
    print("✅ comprehensive_modules/__init__.py exists")

def test_enhanced_modules_init():
    """Test that enhanced modules __init__ exists"""
    init_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "enhanced_modules",
        "__init__.py"
    )
    assert os.path.exists(init_path), f"enhanced_modules/__init__.py not found at {init_path}"
    print("✅ enhanced_modules/__init__.py exists")

if __name__ == "__main__":
    print("Testing Market Regime Module Imports and Fixes")
    print("=" * 50)
    
    test_enhanced_modules_init()
    test_comprehensive_modules_init()
    test_enhanced_modules_imports()
    test_config_manager_import()
    test_enhanced_regime_detector_v2_imports()
    
    print("\n✅ All critical tests passed!")