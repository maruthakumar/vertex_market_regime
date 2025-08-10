"""
HeavyDB Integration for ML Indicator Strategy
Provides comprehensive database connectivity, connection pooling, and query execution
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import pandas as pd
from queue import Queue, Empty
import pymapd
from pymapd import connect, Connection

from .models import MLIndicatorStrategyModel, MLSignal, MLTrade
from .query_builder import MLIndicatorQueryBuilder
from .constants import ERROR_MESSAGES

logger = logging.getLogger(__name__)


class HeavyDBConnectionPool:
    """Connection pool for HeavyDB connections with automatic management"""
    
    def __init__(self, host: str = "localhost", port: int = 6274, 
                 user: str = "admin", password: str = "HyperInteractive", 
                 dbname: str = "heavyai", pool_size: int = 5):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.pool_size = pool_size
        
        self._pool = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created_connections = 0
        
        # Initialize pool with connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                self._pool.put(conn)
                self._created_connections += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def _create_connection(self) -> Connection:
        """Create a new HeavyDB connection"""
        try:
            conn = connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname
            )
            logger.info(f"Created new HeavyDB connection to {self.host}:{self.port}")
            return conn
        except Exception as e:
            logger.error(f"Failed to create HeavyDB connection: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)"""
        conn = None
        try:
            # Try to get connection from pool
            try:
                conn = self._pool.get(timeout=30)
            except Empty:
                # Pool is empty, create new connection if under limit
                with self._lock:
                    if self._created_connections < self.pool_size * 2:  # Allow overflow
                        conn = self._create_connection()
                        self._created_connections += 1
                    else:
                        raise Exception("Connection pool exhausted")
            
            # Test connection
            if not self._test_connection(conn):
                conn = self._create_connection()
            
            yield conn
            
        except Exception as e:
            logger.error(f"Error with HeavyDB connection: {e}")
            raise
        finally:
            # Return connection to pool
            if conn:
                try:
                    self._pool.put(conn, timeout=1)
                except:
                    # Pool is full, close connection
                    try:
                        conn.close()
                    except:
                        pass
    
    def _test_connection(self, conn: Connection) -> bool:
        """Test if connection is still valid"""
        try:
            conn.execute("SELECT 1")
            return True
        except:
            return False
    
    def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except:
                pass


class MLHeavyDBIntegration:
    """Comprehensive HeavyDB integration for ML Indicator Strategy"""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """Initialize HeavyDB integration"""
        self.connection_params = connection_params or {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
        
        self.connection_pool = HeavyDBConnectionPool(**self.connection_params)
        self.query_builder = MLIndicatorQueryBuilder()
        self.query_cache = {}
        self.performance_metrics = {
            'total_queries': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def execute_query(self, query: str, use_cache: bool = True) -> pd.DataFrame:
        """Execute SQL query with caching and performance tracking"""
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache and query in self.query_cache:
                self.performance_metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for query: {query[:100]}...")
                return self.query_cache[query].copy()
            
            # Execute query
            with self.connection_pool.get_connection() as conn:
                result_df = pd.read_sql(query, conn)
                
                # Cache result if enabled
                if use_cache:
                    self.query_cache[query] = result_df.copy()
                    self.performance_metrics['cache_misses'] += 1
                
                execution_time = time.time() - start_time
                self.performance_metrics['total_queries'] += 1
                self.performance_metrics['total_execution_time'] += execution_time
                
                logger.info(f"Query executed in {execution_time:.3f}s, returned {len(result_df)} rows")
                return result_df
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.3f}s: {e}")
            logger.error(f"Failed query: {query}")
            raise
    
    def get_market_data(self, strategy_model: MLIndicatorStrategyModel, 
                       start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get market data for ML strategy analysis"""
        try:
            # Build market data query
            query = self.query_builder.build_market_data_query(
                strategy_model, start_date, end_date
            )
            
            # Execute query
            market_data = self.execute_query(query)
            
            # Validate data
            if market_data.empty:
                raise ValueError(f"No market data found for period {start_date} to {end_date}")
            
            logger.info(f"Retrieved {len(market_data)} market data records")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            raise
    
    def get_option_chain_data(self, symbol: str, expiry_date: datetime, 
                             strike_range: Optional[tuple] = None) -> pd.DataFrame:
        """Get option chain data for analysis"""
        try:
            # Build option chain query
            query = self.query_builder.build_option_chain_query(
                symbol, expiry_date, strike_range
            )
            
            # Execute query
            option_data = self.execute_query(query)
            
            logger.info(f"Retrieved {len(option_data)} option chain records")
            return option_data
            
        except Exception as e:
            logger.error(f"Failed to get option chain data: {e}")
            raise
    
    def get_historical_indicators(self, strategy_model: MLIndicatorStrategyModel,
                                 lookback_days: int = 30) -> pd.DataFrame:
        """Get historical indicator data for ML training"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Build indicators query
            query = self.query_builder.build_indicators_query(
                strategy_model, start_date, end_date
            )
            
            # Execute query
            indicators_data = self.execute_query(query)
            
            logger.info(f"Retrieved {len(indicators_data)} indicator records")
            return indicators_data
            
        except Exception as e:
            logger.error(f"Failed to get historical indicators: {e}")
            raise
    
    def store_ml_signals(self, signals: List[MLSignal]) -> bool:
        """Store ML signals in database"""
        try:
            if not signals:
                return True
            
            # Build insert query
            query = self.query_builder.build_signals_insert_query(signals)
            
            # Execute insert
            with self.connection_pool.get_connection() as conn:
                conn.execute(query)
                
            logger.info(f"Stored {len(signals)} ML signals in database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store ML signals: {e}")
            return False
    
    def store_ml_trades(self, trades: List[MLTrade]) -> bool:
        """Store ML trades in database"""
        try:
            if not trades:
                return True
            
            # Build insert query
            query = self.query_builder.build_trades_insert_query(trades)
            
            # Execute insert
            with self.connection_pool.get_connection() as conn:
                conn.execute(query)
                
            logger.info(f"Stored {len(trades)} ML trades in database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store ML trades: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get HeavyDB integration performance metrics"""
        metrics = self.performance_metrics.copy()
        
        if metrics['total_queries'] > 0:
            metrics['avg_execution_time'] = metrics['total_execution_time'] / metrics['total_queries']
            metrics['cache_hit_rate'] = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])
        else:
            metrics['avg_execution_time'] = 0.0
            metrics['cache_hit_rate'] = 0.0
        
        return metrics
    
    def optimize_queries(self, strategy_model: MLIndicatorStrategyModel) -> Dict[str, str]:
        """Generate optimized queries for the strategy"""
        try:
            optimized_queries = {}
            
            # Market data query
            optimized_queries['market_data'] = self.query_builder.build_optimized_market_data_query(
                strategy_model
            )
            
            # Indicators query
            optimized_queries['indicators'] = self.query_builder.build_optimized_indicators_query(
                strategy_model
            )
            
            # Option chain query
            optimized_queries['option_chain'] = self.query_builder.build_optimized_option_chain_query(
                strategy_model
            )
            
            logger.info("Generated optimized queries for ML strategy")
            return optimized_queries
            
        except Exception as e:
            logger.error(f"Failed to optimize queries: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test HeavyDB connection"""
        try:
            with self.connection_pool.get_connection() as conn:
                result = conn.execute("SELECT COUNT(*) FROM nifty_option_chain LIMIT 1")
                logger.info("HeavyDB connection test successful")
                return True
        except Exception as e:
            logger.error(f"HeavyDB connection test failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def close(self):
        """Close all connections and cleanup"""
        self.connection_pool.close_all()
        self.clear_cache()
        logger.info("HeavyDB integration closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Singleton instance for global access
_ml_heavydb_integration = None

def get_ml_heavydb_integration(connection_params: Optional[Dict[str, Any]] = None) -> MLHeavyDBIntegration:
    """Get singleton instance of ML HeavyDB integration"""
    global _ml_heavydb_integration
    
    if _ml_heavydb_integration is None:
        _ml_heavydb_integration = MLHeavyDBIntegration(connection_params)
    
    return _ml_heavydb_integration
