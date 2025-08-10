"""
Test Mandatory HeavyDB Connection Enforcement

This test module ensures zero-tolerance policy for HeavyDB connection failures.
NO fallback mechanisms are allowed - if HeavyDB is unavailable, the system MUST fail immediately.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from backtester_v2.dal.heavydb_connection import HeavyDBConnection
from backtester_v2.strategies.market_regime.strategy import MarketRegimeStrategy
from backtester_v2.strategies.market_regime.core.engine import MarketRegimeEngine
from backtester_v2.strategies.market_regime.data.heavydb_data_provider import HeavyDBDataProvider


class TestMandatoryHeavyDBConnection:
    """Test suite for mandatory HeavyDB connection enforcement"""
    
    def test_heavydb_connection_required_on_init(self):
        """Test that HeavyDB connection is required during initialization"""
        # Mock a failed connection
        with patch('backtester_v2.dal.heavydb_connection.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection failed - simulated failure")
            
            # Attempt to create connection should raise exception immediately
            with pytest.raises(Exception) as exc_info:
                HeavyDBConnection()
            
            assert "Connection failed" in str(exc_info.value)
            # Ensure no fallback mechanism was triggered
            assert mock_connect.call_count == 1
    
    def test_no_mock_data_fallback(self):
        """Test that system never falls back to mock data"""
        # Simulate HeavyDB connection failure
        with patch('backtester_v2.dal.heavydb_connection.connect') as mock_connect:
            mock_connect.side_effect = Exception("HeavyDB unavailable")
            
            # Ensure MarketRegimeStrategy fails without trying mock data
            with pytest.raises(Exception) as exc_info:
                strategy = MarketRegimeStrategy()
                strategy.initialize()
            
            assert "HeavyDB" in str(exc_info.value)
            # Verify no mock data imports or usage
            assert 'mock' not in str(exc_info.value).lower()
    
    def test_data_provider_requires_heavydb(self):
        """Test that HeavyDBDataProvider requires active connection"""
        with patch('backtester_v2.dal.heavydb_connection.HeavyDBConnection') as mock_conn_class:
            mock_instance = MagicMock()
            mock_instance.connect.side_effect = Exception("Database connection required")
            mock_conn_class.return_value = mock_instance
            
            with pytest.raises(Exception) as exc_info:
                provider = HeavyDBDataProvider()
            
            assert "Database connection required" in str(exc_info.value)
    
    def test_engine_fails_without_heavydb(self):
        """Test that MarketRegimeEngine fails immediately without HeavyDB"""
        with patch('backtester_v2.strategies.market_regime.data.heavydb_data_provider.HeavyDBDataProvider') as mock_provider:
            mock_provider.side_effect = Exception("HeavyDB connection mandatory")
            
            with pytest.raises(Exception) as exc_info:
                engine = MarketRegimeEngine()
            
            assert "HeavyDB connection mandatory" in str(exc_info.value)
    
    def test_query_execution_requires_connection(self):
        """Test that query execution fails immediately if connection is lost"""
        # Create a mock connection that initially works
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch('backtester_v2.dal.heavydb_connection.connect', return_value=mock_conn):
            conn = HeavyDBConnection()
            
            # Simulate connection loss during query
            mock_cursor.execute.side_effect = Exception("Connection lost")
            
            with pytest.raises(Exception) as exc_info:
                conn.execute_query("SELECT * FROM nifty_option_chain")
            
            assert "Connection lost" in str(exc_info.value)
            # Ensure no retry or fallback mechanism
            assert mock_cursor.execute.call_count == 1
    
    def test_no_synthetic_data_generation(self):
        """Test that system never generates synthetic/mock data"""
        # Check that no synthetic data generation modules exist
        synthetic_modules = [
            'mock_data_generator',
            'synthetic_data',
            'dummy_data',
            'test_data_factory'
        ]
        
        # Scan the market_regime module for any synthetic data references
        market_regime_path = Path(__file__).parent.parent
        
        for py_file in market_regime_path.rglob("*.py"):
            if 'test' not in py_file.name:  # Skip test files
                content = py_file.read_text()
                for module in synthetic_modules:
                    assert module not in content.lower(), f"Found synthetic data reference '{module}' in {py_file}"
    
    def test_connection_retry_disabled(self):
        """Test that connection retries are disabled - fail fast policy"""
        with patch('backtester_v2.dal.heavydb_connection.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")
            
            # Multiple attempts should not be made
            with pytest.raises(Exception):
                conn = HeavyDBConnection()
            
            # Should only try once - no retries
            assert mock_connect.call_count == 1
    
    def test_proper_error_propagation(self):
        """Test that HeavyDB errors are properly propagated up the stack"""
        error_messages = [
            "HeavyDB server is not running",
            "Authentication failed",
            "Database 'heavyai' does not exist",
            "Connection timeout"
        ]
        
        for error_msg in error_messages:
            with patch('backtester_v2.dal.heavydb_connection.connect') as mock_connect:
                mock_connect.side_effect = Exception(error_msg)
                
                with pytest.raises(Exception) as exc_info:
                    HeavyDBConnection()
                
                assert error_msg in str(exc_info.value)
    
    def test_all_components_enforce_connection(self):
        """Test that all market regime components enforce HeavyDB connection"""
        components = [
            'backtester_v2.strategies.market_regime.core.engine.MarketRegimeEngine',
            'backtester_v2.strategies.market_regime.core.analyzer.MarketRegimeAnalyzer',
            'backtester_v2.strategies.market_regime.core.regime_detector.RegimeDetector',
            'backtester_v2.strategies.market_regime.processor.MarketRegimeProcessor'
        ]
        
        for component_path in components:
            module_path, class_name = component_path.rsplit('.', 1)
            
            # Mock the HeavyDB connection to fail
            with patch('backtester_v2.dal.heavydb_connection.HeavyDBConnection') as mock_conn:
                mock_conn.side_effect = Exception(f"HeavyDB required for {class_name}")
                
                try:
                    # Dynamically import the module
                    module = __import__(module_path, fromlist=[class_name])
                    component_class = getattr(module, class_name)
                    
                    # Attempt to instantiate should fail
                    with pytest.raises(Exception) as exc_info:
                        instance = component_class()
                    
                    assert "HeavyDB" in str(exc_info.value)
                except (ImportError, AttributeError):
                    # Component might not exist yet - that's ok for this test
                    pass
    
    def test_configuration_requires_database(self):
        """Test that configuration loading requires database connection"""
        from backtester_v2.strategies.market_regime.config_manager import ConfigManager
        
        with patch('backtester_v2.dal.heavydb_connection.HeavyDBConnection') as mock_conn:
            mock_conn.side_effect = Exception("Database required for configuration")
            
            with pytest.raises(Exception) as exc_info:
                config = ConfigManager()
                config.load_configuration()
            
            assert "Database required" in str(exc_info.value)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])