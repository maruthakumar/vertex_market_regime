"""
TV Strategy Test Suite
Comprehensive testing for TradingView strategy implementation
"""

# Test configuration constants
HEAVYDB_TEST_CONFIG = {
    'host': 'localhost',
    'port': 6274,
    'user': 'admin',
    'password': 'HyperInteractive',
    'database': 'heavyai'
}

# Real data validation - NO MOCK DATA ALLOWED
MOCK_DATA_FORBIDDEN = True
REQUIRE_HEAVYDB_CONNECTION = True