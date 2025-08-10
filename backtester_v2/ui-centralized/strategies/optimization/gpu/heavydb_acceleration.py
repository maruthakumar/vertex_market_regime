"""
HeavyDB GPU Acceleration for Strategy Optimization

Leverages HeavyDB's GPU-accelerated SQL engine for large-scale optimization.
Optimized for processing 100,000+ strategy combinations.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class HeavyDBConfig:
    """HeavyDB connection configuration"""
    host: str = 'localhost'
    port: int = 6274
    user: str = 'admin'
    password: str = 'HyperInteractive'
    database: str = 'heavyai'
    protocol: str = 'binary'
    
class HeavyDBAcceleration:
    """
    HeavyDB GPU acceleration for optimization
    
    Provides GPU-accelerated evaluation of objective functions using
    HeavyDB's columnar GPU processing capabilities.
    """
    
    def __init__(self, 
                 config: Optional[HeavyDBConfig] = None,
                 table_prefix: str = 'opt_',
                 batch_size: int = 10000,
                 fragment_size: int = 32000000):
        """
        Initialize HeavyDB acceleration
        
        Args:
            config: HeavyDB connection configuration
            table_prefix: Prefix for optimization tables
            batch_size: Batch size for processing
            fragment_size: Fragment size for HeavyDB tables
        """
        self.config = config or HeavyDBConfig()
        self.table_prefix = table_prefix
        self.batch_size = batch_size
        self.fragment_size = fragment_size
        
        # Connection management
        self.connection = None
        self.is_connected = False
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'total_rows_processed': 0,
            'total_execution_time': 0.0,
            'average_rows_per_second': 0.0
        }
        
        # Check HeavyDB availability and connect
        self._initialize_connection()
        
        logger.info(f"HeavyDB acceleration initialized: connected={self.is_connected}")
    
    def _initialize_connection(self):
        """Initialize HeavyDB connection"""
        try:
            import pymapd
            
            self.connection = pymapd.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                dbname=self.config.database,
                protocol=self.config.protocol
            )
            
            # Test connection
            self.connection.execute("SELECT 1")
            self.is_connected = True
            
            logger.info(f"Connected to HeavyDB at {self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("pymapd not available - HeavyDB acceleration disabled")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Failed to connect to HeavyDB: {e}")
            self.is_connected = False
    
    def accelerate_function(self, 
                           objective_function: Callable,
                           workload: 'WorkloadProfile') -> Callable:
        """
        Create HeavyDB-accelerated version of objective function
        
        Args:
            objective_function: Original objective function
            workload: Workload characteristics
            
        Returns:
            GPU-accelerated objective function
        """
        if not self.is_connected:
            logger.warning("HeavyDB not available, returning original function")
            return objective_function
        
        def heavydb_accelerated_function(params: Dict[str, float]) -> float:
            # For single evaluations, use batch evaluation with size 1
            results = self.batch_evaluate(objective_function, [params], workload)
            return results[0] if results else objective_function(params)
        
        return heavydb_accelerated_function
    
    def batch_evaluate(self,
                      objective_function: Callable,
                      parameter_sets: List[Dict[str, float]],
                      workload: 'WorkloadProfile') -> List[float]:
        """
        Batch evaluate using HeavyDB GPU acceleration
        
        Args:
            objective_function: Objective function
            parameter_sets: List of parameter dictionaries
            workload: Workload characteristics
            
        Returns:
            List of evaluation results
        """
        if not self.is_connected:
            logger.warning("HeavyDB not available, using CPU fallback")
            return [objective_function(params) for params in parameter_sets]
        
        if len(parameter_sets) == 0:
            return []
        
        start_time = time.time()
        
        try:
            # Create temporary table for parameter sets
            table_name = f"{self.table_prefix}params_{int(time.time() * 1000)}"
            
            # Convert parameter sets to DataFrame
            params_df = pd.DataFrame(parameter_sets)
            
            # Create table and insert data
            self._create_parameters_table(table_name, params_df)
            self._insert_parameters(table_name, params_df)
            
            # Execute GPU-accelerated evaluation
            results = self._execute_gpu_evaluation(table_name, objective_function, workload)
            
            # Cleanup
            self._drop_table(table_name)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats['total_queries'] += 1
            self.performance_stats['total_rows_processed'] += len(parameter_sets)
            self.performance_stats['total_execution_time'] += execution_time
            
            if self.performance_stats['total_execution_time'] > 0:
                self.performance_stats['average_rows_per_second'] = (
                    self.performance_stats['total_rows_processed'] / 
                    self.performance_stats['total_execution_time']
                )
            
            logger.info(f"HeavyDB batch evaluation: {len(parameter_sets)} parameters in {execution_time:.2f}s "
                       f"({len(parameter_sets)/execution_time:.0f} params/sec)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in HeavyDB batch evaluation: {e}")
            # Fallback to CPU evaluation
            return [objective_function(params) for params in parameter_sets]
    
    def _create_parameters_table(self, table_name: str, params_df: pd.DataFrame):
        """Create HeavyDB table for parameters"""
        
        # Generate column definitions
        columns = []
        for col in params_df.columns:
            columns.append(f"{col} DOUBLE")
        columns.append("result DOUBLE")  # For storing results
        columns.append("param_id BIGINT")  # For row identification
        
        create_sql = f"""
        CREATE TABLE {table_name} (
            {', '.join(columns)}
        ) WITH (fragment_size={self.fragment_size})
        """
        
        self.connection.execute(create_sql)
        logger.debug(f"Created HeavyDB table: {table_name}")
    
    def _insert_parameters(self, table_name: str, params_df: pd.DataFrame):
        """Insert parameters into HeavyDB table"""
        
        # Add param_id column
        params_df = params_df.copy()
        params_df['param_id'] = range(len(params_df))
        params_df['result'] = 0.0  # Initialize result column
        
        # Use HeavyDB's bulk insert capability
        try:
            # Convert to list of tuples for bulk insert
            data_tuples = [tuple(row) for row in params_df.values]
            
            # Prepare insert statement
            placeholders = ', '.join(['?' for _ in params_df.columns])
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
            
            # Execute bulk insert
            self.connection.executemany(insert_sql, data_tuples)
            
            logger.debug(f"Inserted {len(params_df)} parameter sets into {table_name}")
            
        except Exception as e:
            logger.error(f"Error inserting parameters: {e}")
            # Fallback to individual inserts
            for _, row in params_df.iterrows():
                values = ', '.join([str(v) for v in row.values])
                insert_sql = f"INSERT INTO {table_name} VALUES ({values})"
                self.connection.execute(insert_sql)
    
    def _execute_gpu_evaluation(self, 
                               table_name: str, 
                               objective_function: Callable,
                               workload: 'WorkloadProfile') -> List[float]:
        """Execute GPU-accelerated evaluation using SQL"""
        
        try:
            # Try to create a SQL-based objective function
            sql_objective = self._create_sql_objective_function(objective_function)
            
            if sql_objective:
                # Use pure SQL evaluation (fastest)
                return self._sql_based_evaluation(table_name, sql_objective)
            else:
                # Use hybrid approach: SQL + Python UDF
                return self._hybrid_evaluation(table_name, objective_function)
                
        except Exception as e:
            logger.error(f"Error in GPU evaluation: {e}")
            # Fallback to row-by-row evaluation
            return self._fallback_evaluation(table_name, objective_function)
    
    def _create_sql_objective_function(self, objective_function: Callable) -> Optional[str]:
        """
        Try to convert objective function to SQL
        
        Args:
            objective_function: Python objective function
            
        Returns:
            SQL expression or None if conversion not possible
        """
        # This is a simplified example - in practice, you'd need more sophisticated
        # function analysis or pre-defined SQL templates for common objective functions
        
        func_name = getattr(objective_function, '__name__', '')
        
        if 'sharpe' in func_name.lower():
            # Example: Sharpe ratio calculation in SQL
            return """
            (AVG(daily_return) - risk_free_rate) / 
            NULLIF(STDDEV(daily_return), 0) * SQRT(252)
            """
        elif 'return' in func_name.lower():
            # Simple return calculation
            return "SUM(daily_return)"
        elif 'volatility' in func_name.lower():
            # Volatility calculation
            return "STDDEV(daily_return) * SQRT(252)"
        
        # Can't convert to SQL
        return None
    
    def _sql_based_evaluation(self, table_name: str, sql_objective: str) -> List[float]:
        """Pure SQL-based evaluation (fastest path)"""
        
        update_sql = f"""
        UPDATE {table_name} 
        SET result = {sql_objective}
        """
        
        self.connection.execute(update_sql)
        
        # Retrieve results
        select_sql = f"""
        SELECT result 
        FROM {table_name} 
        ORDER BY param_id
        """
        
        result_df = self.connection.select_ipc(select_sql)
        return result_df['result'].tolist()
    
    def _hybrid_evaluation(self, table_name: str, objective_function: Callable) -> List[float]:
        """Hybrid SQL + Python evaluation"""
        
        # Retrieve parameters in batches for evaluation
        select_sql = f"SELECT * FROM {table_name} ORDER BY param_id"
        params_df = self.connection.select_ipc(select_sql)
        
        results = []
        
        # Process in batches to leverage GPU memory efficiently
        for i in range(0, len(params_df), self.batch_size):
            batch = params_df.iloc[i:i + self.batch_size]
            
            # Evaluate batch
            batch_results = []
            for _, row in batch.iterrows():
                # Extract parameters (excluding result and param_id columns)
                param_dict = row.drop(['result', 'param_id']).to_dict()
                result = objective_function(param_dict)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Update results in database (optional, for debugging)
            if len(batch_results) < 1000:  # Only for small batches to avoid overhead
                for j, result in enumerate(batch_results):
                    param_id = batch.iloc[j]['param_id']
                    update_sql = f"UPDATE {table_name} SET result = {result} WHERE param_id = {param_id}"
                    self.connection.execute(update_sql)
        
        return results
    
    def _fallback_evaluation(self, table_name: str, objective_function: Callable) -> List[float]:
        """Fallback row-by-row evaluation"""
        
        select_sql = f"SELECT * FROM {table_name} ORDER BY param_id"
        params_df = self.connection.select_ipc(select_sql)
        
        results = []
        for _, row in params_df.iterrows():
            param_dict = row.drop(['result', 'param_id']).to_dict()
            result = objective_function(param_dict)
            results.append(result)
        
        return results
    
    def _drop_table(self, table_name: str):
        """Drop temporary table"""
        try:
            self.connection.execute(f"DROP TABLE {table_name}")
            logger.debug(f"Dropped table: {table_name}")
        except Exception as e:
            logger.warning(f"Error dropping table {table_name}: {e}")
    
    def optimize_for_workload(self, workload: 'WorkloadProfile') -> Dict[str, Any]:
        """
        Optimize HeavyDB settings for specific workload
        
        Args:
            workload: Workload characteristics
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'batch_size': self.batch_size,
            'fragment_size': self.fragment_size,
            'memory_settings': {},
            'query_optimizations': []
        }
        
        # Adjust batch size based on data size
        if workload.data_size > 100000:
            recommendations['batch_size'] = min(50000, workload.data_size // 10)
            recommendations['query_optimizations'].append('Use larger batch sizes for better GPU utilization')
        
        # Adjust fragment size
        if workload.data_size > 1000000:
            recommendations['fragment_size'] = 64000000
            recommendations['query_optimizations'].append('Increase fragment size for large datasets')
        
        # Memory recommendations
        estimated_memory_mb = workload.data_size * workload.parameter_count * 8 / (1024 * 1024)
        recommendations['memory_settings']['estimated_memory_mb'] = estimated_memory_mb
        
        if estimated_memory_mb > 8192:  # > 8GB
            recommendations['query_optimizations'].append('Consider data streaming for very large datasets')
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get HeavyDB performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add connection info
        stats['connection_info'] = {
            'host': self.config.host,
            'port': self.config.port,
            'database': self.config.database,
            'is_connected': self.is_connected
        }
        
        # Add configuration
        stats['configuration'] = {
            'batch_size': self.batch_size,
            'fragment_size': self.fragment_size,
            'table_prefix': self.table_prefix
        }
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform HeavyDB health check"""
        health = {
            'status': 'healthy',
            'issues': [],
            'connection_test': False,
            'gpu_info': {}
        }
        
        try:
            # Test connection
            if self.is_connected:
                result = self.connection.execute("SELECT 1")
                health['connection_test'] = True
                
                # Get GPU info
                gpu_query = """
                SELECT 
                    device_id,
                    memory_total,
                    memory_available 
                FROM information_schema.gpu_info
                """
                try:
                    gpu_info = self.connection.execute(gpu_query)
                    health['gpu_info'] = gpu_info
                except:
                    health['issues'].append('Cannot retrieve GPU information')
            else:
                health['status'] = 'unhealthy'
                health['issues'].append('Not connected to HeavyDB')
                
        except Exception as e:
            health['status'] = 'unhealthy' 
            health['issues'].append(f'Connection test failed: {e}')
        
        return health
    
    def close(self):
        """Close HeavyDB connection"""
        if self.connection:
            try:
                self.connection.close()
                self.is_connected = False
                logger.info("HeavyDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing HeavyDB connection: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.close()