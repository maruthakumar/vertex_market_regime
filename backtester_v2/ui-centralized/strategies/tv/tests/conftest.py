#!/usr/bin/env python3
"""
TV Strategy Test Configuration and Fixtures
"""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add strategy path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests import HEAVYDB_TEST_CONFIG, MOCK_DATA_FORBIDDEN, REQUIRE_HEAVYDB_CONNECTION

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def heavydb_connection():
    """
    HeavyDB connection fixture - MANDATORY for all tests
    NO MOCK DATA ALLOWED
    """
    if MOCK_DATA_FORBIDDEN:
        try:
            # Import HeavyDB connection
            sys.path.append('/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-tv/backtester_v2')
            from dal.heavydb_connection import HeavyDBConnection
            
            conn = HeavyDBConnection(
                host=HEAVYDB_TEST_CONFIG['host'],
                port=HEAVYDB_TEST_CONFIG['port'],
                user=HEAVYDB_TEST_CONFIG['user'],
                password=HEAVYDB_TEST_CONFIG['password'],
                database=HEAVYDB_TEST_CONFIG['database']
            )
            
            # Test connection
            result = conn.execute("SELECT COUNT(*) as count FROM nifty_option_chain LIMIT 1")
            assert result is not None, "HeavyDB connection failed"
            logger.info(f"HeavyDB connection established - {result[0]['count']} rows available")
            
            yield conn
            
        except Exception as e:
            pytest.fail(f"CRITICAL: HeavyDB connection required but failed: {e}")
    else:
        pytest.fail("MOCK DATA IS FORBIDDEN - Real HeavyDB connection required")

@pytest.fixture
def sample_tv_config():
    """Sample TV configuration for testing"""
    return {
        'Name': 'TEST_TV_CONFIG',
        'Enabled': 'YES',
        'StartDate': '01_01_2024',
        'EndDate': '31_01_2024',
        'SignalDateFormat': '%Y%m%d %H%M%S',
        'SignalFilePath': 'test_signals.xlsx',
        'LongPortfolioFilePath': 'portfolio_long.xlsx',
        'ShortPortfolioFilePath': 'portfolio_short.xlsx',
        'IntradaySqOffApplicable': 'YES',
        'IntradayExitTime': '153000',
        'TvExitApplicable': 'YES',
        'DoRollover': 'NO',
        'SlippagePercent': 0.05,
        'UseDbExitTiming': 'NO',
        'ExitSearchInterval': 5,
        'ExitPriceSource': 'SPOT'
    }

@pytest.fixture
def sample_signals():
    """Sample TV signals for testing"""
    return [
        {
            'Trade #': 'T001',
            'Type': 'Entry Long',
            'Date/Time': '20240101 091500',
            'Contracts': 5
        },
        {
            'Trade #': 'T001',
            'Type': 'Exit Long',
            'Date/Time': '20240101 153000',
            'Contracts': 5
        },
        {
            'Trade #': 'T002',
            'Type': 'Entry Short',
            'Date/Time': '20240102 091500',
            'Contracts': 10
        },
        {
            'Trade #': 'T002',
            'Type': 'Exit Short',
            'Date/Time': '20240102 153000',
            'Contracts': 10
        }
    ]

@pytest.fixture
def real_config_files():
    """Paths to REAL production TV configuration files - NO MOCK DATA"""
    base_path = Path(__file__).parent.parent.parent.parent / "configurations" / "data" / "prod" / "tv"
    
    # Verify all real files exist
    real_files = {
        'tv_master': base_path / "TV_CONFIG_MASTER_1.0.0.xlsx",
        'signals': base_path / "TV_CONFIG_SIGNALS_1.0.0.xlsx", 
        'portfolio_long': base_path / "TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx",
        'portfolio_short': base_path / "TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx",
        'portfolio_manual': base_path / "TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx",
        'strategy': base_path / "TV_CONFIG_STRATEGY_1.0.0.xlsx"
    }
    
    # Ensure all real files exist
    for file_key, file_path in real_files.items():
        assert file_path.exists(), f"REAL config file missing: {file_path}"
    
    return real_files

@pytest.fixture
def real_nifty_data_sample(heavydb_connection):
    """
    Get real NIFTY data sample from HeavyDB
    NO MOCK DATA - ALWAYS REAL DATABASE
    """
    try:
        query = """
        SELECT 
            trade_date, trade_time, index_spot, expiry_date,
            strike_price, call_ltp, put_ltp, call_volume, put_volume
        FROM nifty_option_chain 
        WHERE trade_date = DATE '2024-01-01'
        AND trade_time BETWEEN TIME '09:15:00' AND TIME '09:30:00'
        AND strike_price BETWEEN 21000 AND 22000
        ORDER BY trade_time, strike_price
        LIMIT 100
        """
        
        result = heavydb_connection.execute(query)
        assert result is not None and len(result) > 0, "No real NIFTY data found"
        
        # Convert to DataFrame for easier testing
        df = pd.DataFrame(result)
        logger.info(f"Retrieved {len(df)} real NIFTY data rows for testing")
        
        return df
        
    except Exception as e:
        pytest.fail(f"Failed to get real NIFTY data: {e}")

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "heavydb: mark test as requiring HeavyDB connection"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )

def pytest_collection_modifyitems(config, items):
    """Add markers based on test names and paths"""
    for item in items:
        # Add heavydb marker for all tests (since we require real data)
        item.add_marker(pytest.mark.heavydb)
        
        # Add markers based on path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker for performance tests
        if "performance" in str(item.fspath) or "performance" in item.name:
            item.add_marker(pytest.mark.performance)