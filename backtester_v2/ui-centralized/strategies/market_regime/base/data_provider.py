"""
Data Provider Interface for Market Regime System

Provides dependency injection pattern for data access to avoid
circular dependencies and improve testability.

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 1.0.0
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataQuery:
    """Standard data query parameters"""
    symbol: str
    start_time: datetime
    end_time: datetime
    data_type: str  # 'option_chain', 'underlying', 'indicators'
    filters: Optional[Dict[str, Any]] = None
    columns: Optional[List[str]] = None


class DataProviderInterface(ABC):
    """
    Abstract interface for data providers
    
    Implementations can provide data from:
    - HeavyDB
    - MySQL
    - Mock/Test data
    - Real-time feeds
    """
    
    @abstractmethod
    def get_option_chain(self, query: DataQuery) -> pd.DataFrame:
        """Get option chain data"""
        pass
        
    @abstractmethod
    def get_underlying_data(self, query: DataQuery) -> pd.DataFrame:
        """Get underlying price data"""
        pass
        
    @abstractmethod
    def get_indicators(self, query: DataQuery) -> Dict[str, Any]:
        """Get technical indicators"""
        pass
        
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if data provider is connected"""
        pass
        
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        pass


class HeavyDBDataProvider(DataProviderInterface):
    """HeavyDB data provider implementation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HeavyDB provider"""
        self.config = config or self._get_default_config()
        self.connection = None
        self._connect()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default HeavyDB configuration"""
        return {
            'host': 'localhost',
            'port': 6274,
            'database': 'heavyai',
            'user': 'admin',
            'password': 'HyperInteractive',
            'table': 'nifty_option_chain'
        }
        
    def _connect(self):
        """Establish HeavyDB connection"""
        try:
            # Import HeavyDB connection module
            from ...dal.heavydb_connection import get_connection
            self.connection = get_connection()
            logger.info("HeavyDB connection established")
        except Exception as e:
            logger.error(f"Failed to connect to HeavyDB: {e}")
            self.connection = None
            
    def get_option_chain(self, query: DataQuery) -> pd.DataFrame:
        """Get option chain data from HeavyDB"""
        if not self.is_connected():
            logger.warning("HeavyDB not connected, returning empty DataFrame")
            return pd.DataFrame()
            
        try:
            # Build query
            sql = f"""
            SELECT * FROM {self.config['table']}
            WHERE symbol = '{query.symbol}'
            AND trade_time >= '{query.start_time}'
            AND trade_time <= '{query.end_time}'
            """
            
            # Add filters if provided
            if query.filters:
                for key, value in query.filters.items():
                    sql += f" AND {key} = '{value}'"
                    
            # Select specific columns if provided
            if query.columns:
                columns = ", ".join(query.columns)
                sql = sql.replace("SELECT *", f"SELECT {columns}")
                
            # Execute query
            from ...dal.heavydb_connection import execute_query
            result = execute_query(self.connection, sql)
            
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return pd.DataFrame()
            
    def get_underlying_data(self, query: DataQuery) -> pd.DataFrame:
        """Get underlying price data"""
        # For HeavyDB, underlying data might be in a different table
        # or extracted from option chain data
        if not self.is_connected():
            return pd.DataFrame()
            
        try:
            # Simple implementation - get spot prices from option data
            sql = f"""
            SELECT DISTINCT trade_time, underlying_price
            FROM {self.config['table']}
            WHERE symbol = '{query.symbol}'
            AND trade_time >= '{query.start_time}'
            AND trade_time <= '{query.end_time}'
            ORDER BY trade_time
            """
            
            from ...dal.heavydb_connection import execute_query
            result = execute_query(self.connection, sql)
            
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching underlying data: {e}")
            return pd.DataFrame()
            
    def get_indicators(self, query: DataQuery) -> Dict[str, Any]:
        """Get technical indicators (computed from data)"""
        # This would typically compute indicators from underlying data
        underlying_data = self.get_underlying_data(query)
        
        if underlying_data.empty:
            return {}
            
        try:
            indicators = {}
            
            # Simple RSI calculation (example)
            if len(underlying_data) >= 14:
                prices = underlying_data['underlying_price'].values
                deltas = np.diff(prices)
                gains = deltas[deltas > 0].mean() if len(deltas[deltas > 0]) > 0 else 0
                losses = -deltas[deltas < 0].mean() if len(deltas[deltas < 0]) > 0 else 0
                
                if losses != 0:
                    rs = gains / losses
                    indicators['rsi'] = 100 - (100 / (1 + rs))
                else:
                    indicators['rsi'] = 100
                    
            # ATR calculation (example)
            if len(underlying_data) >= 14:
                high_low = underlying_data['underlying_price'].rolling(2).max() - \
                          underlying_data['underlying_price'].rolling(2).min()
                indicators['atr'] = high_low.mean()
                
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
            
    def is_connected(self) -> bool:
        """Check if HeavyDB is connected"""
        return self.connection is not None
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            'provider': 'HeavyDB',
            'host': self.config['host'],
            'port': self.config['port'],
            'database': self.config['database'],
            'table': self.config['table'],
            'connected': self.is_connected()
        }


class MockDataProvider(DataProviderInterface):
    """Mock data provider for testing"""
    
    def __init__(self, mock_data: Optional[Dict[str, Any]] = None):
        """Initialize with optional mock data"""
        self.mock_data = mock_data or {}
        
    def get_option_chain(self, query: DataQuery) -> pd.DataFrame:
        """Return mock option chain data"""
        if 'option_chain' in self.mock_data:
            return self.mock_data['option_chain']
            
        # Generate simple mock data
        strikes = [49500, 49750, 50000, 50250, 50500]
        data = []
        
        for strike in strikes:
            for opt_type in ['CE', 'PE']:
                data.append({
                    'strike_price': strike,
                    'option_type': opt_type,
                    'last_price': abs(50000 - strike) / 10,
                    'volume': 1000,
                    'implied_volatility': 16 + abs(50000 - strike) / 1000,
                    'delta': 0.5 if opt_type == 'CE' else -0.5
                })
                
        return pd.DataFrame(data)
        
    def get_underlying_data(self, query: DataQuery) -> pd.DataFrame:
        """Return mock underlying data"""
        if 'underlying' in self.mock_data:
            return self.mock_data['underlying']
            
        # Generate simple time series
        times = pd.date_range(query.start_time, query.end_time, freq='1min')
        prices = [50000 + i * 10 for i in range(len(times))]
        
        return pd.DataFrame({
            'trade_time': times,
            'underlying_price': prices
        })
        
    def get_indicators(self, query: DataQuery) -> Dict[str, Any]:
        """Return mock indicators"""
        if 'indicators' in self.mock_data:
            return self.mock_data['indicators']
            
        return {
            'rsi': 55,
            'atr': 250,
            'adx': 25,
            'macd_signal': 10,
            'bollinger_width': 400
        }
        
    def is_connected(self) -> bool:
        """Mock provider is always connected"""
        return True
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Get mock provider info"""
        return {
            'provider': 'Mock',
            'data_keys': list(self.mock_data.keys()),
            'connected': True
        }


class DataProviderRegistry:
    """Registry for managing data providers"""
    
    _instance = None
    _providers = {}
    _default_provider = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def register_provider(self, name: str, provider: DataProviderInterface):
        """Register a data provider"""
        self._providers[name] = provider
        logger.info(f"Registered data provider: {name}")
        
        # Set as default if first provider
        if self._default_provider is None:
            self._default_provider = name
            
    def get_provider(self, name: Optional[str] = None) -> DataProviderInterface:
        """Get a data provider by name or default"""
        if name is None:
            name = self._default_provider
            
        if name not in self._providers:
            raise ValueError(f"Unknown data provider: {name}")
            
        return self._providers[name]
        
    def set_default_provider(self, name: str):
        """Set default data provider"""
        if name not in self._providers:
            raise ValueError(f"Unknown data provider: {name}")
            
        self._default_provider = name
        logger.info(f"Set default data provider: {name}")
        
    def list_providers(self) -> List[str]:
        """List available providers"""
        return list(self._providers.keys())
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all providers"""
        info = {
            'default': self._default_provider,
            'providers': {}
        }
        
        for name, provider in self._providers.items():
            info['providers'][name] = provider.get_provider_info()
            
        return info


# Convenience functions
def get_data_provider_registry() -> DataProviderRegistry:
    """Get the global data provider registry"""
    return DataProviderRegistry()


def register_default_providers():
    """Register default data providers"""
    registry = get_data_provider_registry()
    
    # Register HeavyDB provider
    try:
        heavydb_provider = HeavyDBDataProvider()
        registry.register_provider('heavydb', heavydb_provider)
    except Exception as e:
        logger.warning(f"Failed to register HeavyDB provider: {e}")
        
    # Always register mock provider for testing
    mock_provider = MockDataProvider()
    registry.register_provider('mock', mock_provider)
    
    # Set default based on availability
    if 'heavydb' in registry.list_providers():
        registry.set_default_provider('heavydb')
    else:
        registry.set_default_provider('mock')


# Import numpy for indicator calculations
import numpy as np