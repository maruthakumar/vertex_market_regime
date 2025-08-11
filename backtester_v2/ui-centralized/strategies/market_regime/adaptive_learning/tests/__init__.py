"""
Test Suite for Adaptive Learning Framework

Comprehensive test coverage for:
- Schema registry validation (774 features across 8 components)
- Transform utilities and GPU memory management
- Deterministic function validation
- Cache performance and TTL behavior
- Fallback mechanisms when GPU/external services unavailable
- Performance benchmarks within <800ms system budget
"""

import logging
import pytest
from pathlib import Path

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_CACHE_DIR = Path(__file__).parent / "test_cache"

# Create test directories
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_CACHE_DIR.mkdir(exist_ok=True)

# Performance test budgets
FRAMEWORK_PERFORMANCE_BUDGET_MS = 50
COMPONENT_PERFORMANCE_BUDGETS = {
    "component_01_triple_straddle": 100,
    "component_02_greeks_sentiment": 80,
    "component_03_oi_pa_trending": 120,
    "component_04_iv_skew": 90,
    "component_05_atr_ema_cpr": 110,
    "component_06_correlation": 150,
    "component_07_support_resistance": 85,
    "component_08_master_integration": 50
}

TOTAL_SYSTEM_BUDGET_MS = 800

logger.info("Adaptive Learning Framework test suite initialized")