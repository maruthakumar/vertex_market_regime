#!/usr/bin/env python3
"""
HeavyDB Query Optimizer - High-performance GPU-accelerated query optimization
Target: >200K rows/sec processing (previously achieved 207,760 rows/sec)
"""

import pymapd
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for HeavyDB queries"""
    query_id: str
    query_hash: str
    execution_time_ms: float
    rows_processed: int
    rows_per_second: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    query_plan_cost: float
    cache_hit: bool
    timestamp: float
    
    @classmethod
    def create(cls, query_id: str, query: str, execution_time_ms: float, 
              rows_processed: int, memory_usage_mb: float = 0.0,
              gpu_utilization_percent: float = 0.0) -> 'QueryPerformanceMetrics':
        """Create performance metrics"""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        rows_per_second = (rows_processed / execution_time_ms) * 1000 if execution_time_ms > 0 else 0
        
        return cls(
            query_id=query_id,
            query_hash=query_hash,
            execution_time_ms=execution_time_ms,
            rows_processed=rows_processed,
            rows_per_second=rows_per_second,
            memory_usage_mb=memory_usage_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            query_plan_cost=0.0,
            cache_hit=False,
            timestamp=time.time()
        )

class HeavyDBOptimizer:
    """High-performance HeavyDB query optimizer for TBS strategy"""
    
    def __init__(self, connection_params: Dict[str, Any] = None):
        self.connection_params = connection_params or {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin', 
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
        
        # Connection pool management
        self.connection_pool = []
        self.pool_size = 4
        self.pool_lock = threading.Lock()
        
        # Query optimization settings
        self.gpu_optimization_hints = {
            'force_gpu': "/*+ cpu_mode=false */",
            'disable_watchdog': "/*+ watchdog_max_size=0 */",
            'parallel_processing': "/*+ parallel_top_min=1000 */",
            'large_group_buffer': "/*+ group_by_buffer_size=4000000000 */",  # 4GB
            'columnar_output': "/*+ enable_columnar_output=true */",
            'lazy_fetch': "/*+ enable_lazy_fetch=true */",
            'hash_join': "/*+ use_hash_join=true */",
            'fragment_skipping': "/*+ enable_fragment_skipping=true */"
        }
        
        # Performance tracking
        self.query_metrics: List[QueryPerformanceMetrics] = []
        self.query_cache = {}
        self.performance_targets = {
            'min_rows_per_second': 200000,
            'max_execution_time_ms': 5000,
            'target_gpu_utilization': 80.0
        }
        
        # Table metadata cache
        self.table_metadata = {}
        self.column_statistics = {}
        
        # Initialize connection pool
        self._initialize_connection_pool()
        
    def _initialize_connection_pool(self):
        """Initialize connection pool for parallel query execution"""
        logger.info(f"Initializing HeavyDB connection pool (size: {self.pool_size})")
        
        for i in range(self.pool_size):
            try:
                connection = pymapd.connect(**self.connection_params)
                self.connection_pool.append(connection)
                logger.debug(f"Connection pool [{i+1}/{self.pool_size}] established")
            except Exception as e:
                logger.error(f"Failed to create connection {i+1}: {e}")
                
        if not self.connection_pool:
            raise RuntimeError("Failed to establish any HeavyDB connections")
            
        logger.info(f"Connection pool initialized with {len(self.connection_pool)} connections")
        
    def get_connection(self) -> pymapd.Connection:
        """Get connection from pool"""
        with self.pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            else:
                # Create new connection if pool is empty
                return pymapd.connect(**self.connection_params)
                
    def return_connection(self, connection: pymapd.Connection):
        """Return connection to pool"""
        with self.pool_lock:
            if len(self.connection_pool) < self.pool_size:
                self.connection_pool.append(connection)
            else:
                connection.close()
                
    def analyze_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Analyze table metadata for optimization"""
        if table_name in self.table_metadata:
            return self.table_metadata[table_name]
            
        connection = self.get_connection()
        try:
            # Get table information
            table_info_query = f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            
            cursor = connection.cursor()
            cursor.execute(table_info_query)
            columns_info = cursor.fetchall()
            
            # Get table statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT expiry_date) as unique_expiries
            FROM {table_name}
            """
            
            cursor.execute(stats_query)
            stats = cursor.fetchone()
            
            metadata = {
                'columns': [
                    {
                        'name': col[0],
                        'type': col[1], 
                        'nullable': col[2],
                        'default': col[3]
                    } for col in columns_info
                ],
                'total_rows': stats[0] if stats else 0,
                'date_range': {
                    'min_date': stats[1] if stats else None,
                    'max_date': stats[2] if stats else None
                },
                'unique_symbols': stats[3] if stats else 0,
                'unique_expiries': stats[4] if stats else 0,
                'analysis_timestamp': time.time()
            }
            
            self.table_metadata[table_name] = metadata
            logger.info(f"Analyzed metadata for table {table_name}: {metadata['total_rows']:,} rows")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze table metadata: {e}")
            return {}
        finally:
            self.return_connection(connection)
            
    def optimize_query(self, query: str, estimated_rows: int = 0) -> str:
        """Optimize query with GPU-specific hints and optimizations"""
        
        # Analyze query characteristics
        query_analysis = self._analyze_query(query)
        
        # Apply base GPU optimizations
        optimized_query = query
        
        # Add GPU optimization hints based on query characteristics
        hints = []
        
        # Always use GPU mode for TBS queries
        hints.append(self.gpu_optimization_hints['force_gpu'])
        hints.append(self.gpu_optimization_hints['disable_watchdog'])
        
        # Add parallel processing for large datasets
        if estimated_rows > 10000 or query_analysis['has_aggregation']:
            hints.append(self.gpu_optimization_hints['parallel_processing'])
            
        # Large group buffer for GROUP BY operations
        if query_analysis['has_group_by']:
            hints.append(self.gpu_optimization_hints['large_group_buffer'])
            
        # Enable columnar output for better performance
        hints.append(self.gpu_optimization_hints['columnar_output'])
        hints.append(self.gpu_optimization_hints['lazy_fetch'])
        
        # Use hash joins for multi-table queries
        if query_analysis['join_count'] > 0:
            hints.append(self.gpu_optimization_hints['hash_join'])
            
        # Enable fragment skipping for date-based filtering
        if query_analysis['has_date_filter']:
            hints.append(self.gpu_optimization_hints['fragment_skipping'])
            
        # Combine hints and add to query
        if hints:
            hint_string = " ".join(hints)
            if not optimized_query.strip().startswith('/*+'):
                optimized_query = f"{hint_string}\n{optimized_query}"
                
        # Apply additional query-specific optimizations
        optimized_query = self._apply_query_optimizations(optimized_query, query_analysis)
        
        return optimized_query
        
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics for optimization"""
        query_upper = query.upper()
        
        analysis = {
            'has_aggregation': any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV']),
            'has_group_by': 'GROUP BY' in query_upper,
            'has_order_by': 'ORDER BY' in query_upper,
            'has_window_functions': any(func in query_upper for func in ['ROW_NUMBER', 'RANK', 'OVER']),
            'has_date_filter': any(col in query_upper for col in ['TRADE_DATE', 'EXPIRY_DATE']),
            'join_count': query_upper.count('JOIN'),
            'subquery_count': query_upper.count('WITH') + query_upper.count('SELECT') - 1,
            'complexity_score': 0
        }
        
        # Calculate complexity score
        analysis['complexity_score'] = (
            analysis['has_aggregation'] * 2 +
            analysis['has_group_by'] * 3 +
            analysis['has_window_functions'] * 4 +
            analysis['join_count'] * 2 +
            analysis['subquery_count'] * 1
        )
        
        return analysis
        
    def _apply_query_optimizations(self, query: str, analysis: Dict[str, Any]) -> str:
        """Apply specific query optimizations based on analysis"""
        optimized = query
        
        # Optimize date filtering
        if analysis['has_date_filter']:
            # Ensure date literals are properly formatted for fragment skipping
            optimized = re.sub(
                r"trade_date\s*BETWEEN\s*'([^']+)'\s*AND\s*'([^']+)'",
                r"trade_date BETWEEN DATE '\1' AND DATE '\2'",
                optimized,
                flags=re.IGNORECASE
            )
            
        # Optimize JOIN order for better performance
        if analysis['join_count'] > 1:
            # Add hint for join reordering
            optimized = re.sub(
                r"JOIN\s+(\w+)",
                r"/*+ join_order(auto) */ JOIN \1",
                optimized,
                count=1,
                flags=re.IGNORECASE
            )
            
        # Optimize window functions
        if analysis['has_window_functions']:
            # Add hint for window function optimization
            optimized = optimized.replace(
                "OVER (",
                "/*+ window_function_optimization=true */ OVER ("
            )
            
        return optimized
        
    def execute_optimized_query(self, query: str, query_id: str = None) -> Tuple[pd.DataFrame, QueryPerformanceMetrics]:
        """Execute optimized query with performance tracking"""
        if query_id is None:
            query_id = f"query_{int(time.time())}"
            
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.query_cache:
            cached_result, cached_metrics = self.query_cache[query_hash]
            cached_metrics.cache_hit = True
            logger.info(f"Cache hit for query {query_id} ({query_hash[:8]})")
            return cached_result.copy(), cached_metrics
            
        # Get connection from pool
        connection = self.get_connection()
        start_time = time.perf_counter()
        
        try:
            cursor = connection.cursor()
            
            # Execute query with timing
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            execution_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Convert to DataFrame
            if results and column_names:
                df = pd.DataFrame(results, columns=column_names)
                rows_processed = len(df)
            else:
                df = pd.DataFrame()
                rows_processed = 0
                
            # Create performance metrics
            metrics = QueryPerformanceMetrics.create(
                query_id=query_id,
                query=query,
                execution_time_ms=execution_time,
                rows_processed=rows_processed
            )
            
            # Cache result if performance is good
            if metrics.rows_per_second >= self.performance_targets['min_rows_per_second']:
                self.query_cache[query_hash] = (df.copy(), metrics)
                
            self.query_metrics.append(metrics)
            
            logger.info(f"Query {query_id} executed: {execution_time:.2f}ms, "
                       f"{rows_processed:,} rows, {metrics.rows_per_second:,.0f} rows/sec")
            
            return df, metrics
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Query execution failed: {e}")
            
            # Create error metrics
            metrics = QueryPerformanceMetrics.create(
                query_id=query_id,
                query=query,
                execution_time_ms=execution_time,
                rows_processed=0
            )
            self.query_metrics.append(metrics)
            
            raise
        finally:
            self.return_connection(connection)
            
    def execute_parallel_queries(self, queries: List[Tuple[str, str]]) -> List[Tuple[pd.DataFrame, QueryPerformanceMetrics]]:
        """Execute multiple queries in parallel"""
        if not queries:
            return []
            
        logger.info(f"Executing {len(queries)} queries in parallel")
        
        results = []
        with ThreadPoolExecutor(max_workers=min(len(queries), self.pool_size)) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.execute_optimized_query, query, query_id): (query, query_id)
                for query, query_id in queries
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_query):
                query, query_id = future_to_query[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel query {query_id} failed: {e}")
                    # Add empty result for failed query
                    empty_df = pd.DataFrame()
                    error_metrics = QueryPerformanceMetrics.create(
                        query_id=query_id,
                        query=query,
                        execution_time_ms=0,
                        rows_processed=0
                    )
                    results.append((empty_df, error_metrics))
                    
        logger.info(f"Completed {len(results)} parallel queries")
        return results
        
    def batch_execute_with_optimization(self, queries: List[str]) -> List[pd.DataFrame]:
        """Execute queries in batches with automatic optimization"""
        if not queries:
            return []
            
        # Optimize all queries
        optimized_queries = []
        for i, query in enumerate(queries):
            query_id = f"batch_query_{i+1}"
            
            # Estimate rows for optimization
            estimated_rows = self._estimate_query_rows(query)
            optimized_query = self.optimize_query(query, estimated_rows)
            
            optimized_queries.append((optimized_query, query_id))
            
        # Execute in parallel
        results = self.execute_parallel_queries(optimized_queries)
        
        # Return just the DataFrames
        return [result[0] for result in results]
        
    def _estimate_query_rows(self, query: str) -> int:
        """Estimate number of rows a query will return"""
        # Simple heuristic based on query characteristics
        query_upper = query.upper()
        
        # Base estimate
        estimated_rows = 10000
        
        # Adjust based on date range
        if 'BETWEEN' in query_upper and 'TRADE_DATE' in query_upper:
            # Try to extract date range
            date_match = re.search(r"BETWEEN\s*'([^']+)'\s*AND\s*'([^']+)'", query_upper)
            if date_match:
                # Rough estimate: 50K rows per day for NIFTY
                try:
                    from datetime import datetime
                    start_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    end_date = datetime.strptime(date_match.group(2), '%Y-%m-%d')
                    days = (end_date - start_date).days + 1
                    estimated_rows = days * 50000
                except:
                    pass
                    
        # Adjust for aggregation
        if 'GROUP BY' in query_upper:
            estimated_rows = min(estimated_rows // 10, 10000)
        elif any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG']):
            estimated_rows = min(estimated_rows // 100, 1000)
            
        return max(estimated_rows, 100)  # Minimum estimate
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.query_metrics:
            return {'error': 'No query metrics available'}
            
        # Calculate aggregate statistics
        total_queries = len(self.query_metrics)
        successful_queries = [m for m in self.query_metrics if m.rows_processed > 0]
        
        if successful_queries:
            avg_execution_time = sum(m.execution_time_ms for m in successful_queries) / len(successful_queries)
            avg_rows_per_second = sum(m.rows_per_second for m in successful_queries) / len(successful_queries)
            max_rows_per_second = max(m.rows_per_second for m in successful_queries)
            total_rows_processed = sum(m.rows_processed for m in successful_queries)
        else:
            avg_execution_time = 0
            avg_rows_per_second = 0  
            max_rows_per_second = 0
            total_rows_processed = 0
            
        # Performance analysis
        target_meeting_queries = [
            m for m in successful_queries 
            if m.rows_per_second >= self.performance_targets['min_rows_per_second']
        ]
        
        cache_hits = sum(1 for m in self.query_metrics if m.cache_hit)
        
        report = {
            'summary': {
                'total_queries': total_queries,
                'successful_queries': len(successful_queries),
                'failed_queries': total_queries - len(successful_queries),
                'cache_hits': cache_hits,
                'cache_hit_rate': (cache_hits / total_queries) * 100 if total_queries > 0 else 0
            },
            'performance_metrics': {
                'avg_execution_time_ms': avg_execution_time,
                'avg_rows_per_second': avg_rows_per_second,
                'max_rows_per_second': max_rows_per_second,
                'total_rows_processed': total_rows_processed,
                'queries_meeting_target': len(target_meeting_queries),
                'target_achievement_rate': (len(target_meeting_queries) / len(successful_queries)) * 100 if successful_queries else 0
            },
            'targets': self.performance_targets,
            'connection_pool': {
                'pool_size': self.pool_size,
                'available_connections': len(self.connection_pool),
                'cached_queries': len(self.query_cache)
            }
        }
        
        # Add detailed metrics for recent queries
        recent_queries = self.query_metrics[-10:] if len(self.query_metrics) > 10 else self.query_metrics
        report['recent_queries'] = [
            {
                'query_id': m.query_id,
                'query_hash': m.query_hash,
                'execution_time_ms': m.execution_time_ms,
                'rows_processed': m.rows_processed,
                'rows_per_second': m.rows_per_second,
                'cache_hit': m.cache_hit
            }
            for m in recent_queries
        ]
        
        return report
        
    def cleanup(self):
        """Cleanup connections and resources"""
        logger.info("Cleaning up HeavyDB optimizer resources")
        
        with self.pool_lock:
            for connection in self.connection_pool:
                try:
                    connection.close()
                except:
                    pass
            self.connection_pool.clear()
            
        self.query_cache.clear()
        logger.info("HeavyDB optimizer cleanup completed")
        
    def __del__(self):
        """Destructor to cleanup resources"""
        try:
            self.cleanup()
        except:
            pass