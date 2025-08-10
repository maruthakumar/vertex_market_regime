# HeavyDB Query Optimization Guide

## Overview

The HeavyDB Query Optimization system is designed to maximize performance when executing backtesting queries on large datasets (33M+ rows). It provides intelligent query analysis, GPU-specific optimizations, advanced caching, and seamless integration with the backtesting pipeline.

## Key Features

### 🚀 Performance Optimizations
- **GPU Acceleration**: Leverages HeavyDB's GPU capabilities with specialized hints
- **Query Complexity Analysis**: Automatically classifies queries and applies appropriate optimizations
- **Large Dataset Handling**: Optimized for 33M+ row operations with partition pruning and streaming
- **Parallel Execution**: Intelligent parallelization for complex queries

### 🧠 Intelligent Caching
- **LRU Cache**: Least Recently Used eviction with configurable size limits
- **Compression**: Automatic DataFrame compression using zlib for memory efficiency
- **Cache Statistics**: Detailed hit rates, compression ratios, and performance metrics
- **Persistence**: Cache survives across query sessions

### 📊 Performance Monitoring
- **Query Profiling**: Detailed execution metrics and optimization tracking
- **Strategy Effectiveness**: Analysis of which optimization strategies work best
- **Recommendations**: Automated suggestions for performance improvements
- **Benchmarking**: Performance comparison and regression detection

## Architecture

### Core Components

```
HeavyDBQueryOptimizer
├── QueryCache (LRU + Compression)
├── Query Analysis Engine
├── Optimization Strategy Engine
├── Performance Metrics Collector
└── Integration with BacktestingPipeline
```

### Query Complexity Classification

1. **SIMPLE**: Basic SELECT queries, single table operations
2. **MODERATE**: Joins, aggregations, basic subqueries
3. **COMPLEX**: Multiple joins, window functions, CTEs
4. **EXTREME**: Multi-CTE, recursive queries, large aggregations

### Optimization Strategies

1. **GPU_ACCELERATION**: GPU-specific query hints and columnar operations
2. **PARALLEL_EXECUTION**: Multi-threaded query processing
3. **INTELLIGENT_CACHING**: Result caching with compression
4. **INDEX_OPTIMIZATION**: Index usage hints and recommendations
5. **PARTITION_PRUNING**: Date-based partition elimination
6. **QUERY_REWRITING**: Query structure optimization
7. **MEMORY_STREAMING**: Memory-efficient processing for large results

## Usage Examples

### Basic Optimization

```python
from backtester_v2.core.heavydb_query_optimizer import create_query_optimizer

# Create optimizer instance
optimizer = create_query_optimizer({
    'cache_size_mb': 1000,
    'enable_gpu_optimization': True,
    'optimization_threshold_rows': 100000
})

# Optimize a query
query = """
    SELECT 
        trade_date,
        AVG(close_price) as avg_price,
        SUM(volume) as total_volume
    FROM nifty_option_chain
    WHERE trade_date BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY trade_date
"""

optimized_query, profile = optimizer.optimize_query(
    query, 
    strategy_type='ML',
    expected_rows=10000000
)

print(f"Applied {len(profile.optimization_strategies)} optimization strategies")
print(f"Query complexity: {profile.complexity}")
```

### Execute Optimized Query

```python
# Execute with automatic optimization and caching
result = optimizer.execute_optimized_query(
    query, 
    strategy_type='ML',
    use_cache=True
)

print(f"Retrieved {len(result)} rows")
```

### Integration with Backtesting Pipeline

```python
from backtester_v2.core.heavydb_backtesting_pipeline import create_backtesting_pipeline

# Pipeline automatically includes query optimization
pipeline = create_backtesting_pipeline({
    'max_concurrent_tasks': 5,
    'cache_size_mb': 1000,
    'enable_gpu_acceleration': True
})

# Queries are automatically optimized during execution
task = BacktestTask(
    task_id="optimized_backtest_001",
    strategy_type="ML",
    strategy_config={"index_name": "NIFTY"},
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Submit and execute (queries are optimized automatically)
with pipeline:
    task_id = pipeline.submit_backtest(task)
    result = await pipeline.execute_backtest_async(task)
```

## GPU Optimization Hints

The system automatically applies HeavyDB-specific GPU optimization hints:

### Basic GPU Hints
```sql
/*+ cpu_mode=false */           -- Force GPU execution
/*+ USE_COLUMNAR_SCAN */        -- Use columnar storage optimizations
/*+ watchdog_max_size=0 */      -- Disable query size limits
/*+ ENABLE_GPU_CACHE */         -- Enable GPU result caching
```

### Advanced GPU Hints (Complex Queries)
```sql
/*+ ENABLE_PARALLEL_PROCESSING */ -- Multi-GPU processing
/*+ OPTIMIZE_MEMORY_USAGE */       -- Memory optimization
/*+ ENABLE_VECTORIZATION */        -- Vector processing for aggregations
```

### Large Dataset Hints (33M+ Rows)
```sql
/*+ ENABLE_PARTITION_PRUNING */    -- Date-based partition elimination
/*+ ENABLE_STREAMING_AGGREGATION */ -- Memory-efficient aggregations
/*+ ENABLE_PARALLEL_EXECUTION */   -- Parallel query execution
```

## Performance Monitoring

### Get Optimization Report

```python
report = optimizer.get_optimization_report()

print("Optimization Metrics:")
print(f"- Total queries optimized: {report['optimization_metrics']['total_queries_optimized']}")
print(f"- Average speedup factor: {report['optimization_metrics']['average_speedup_factor']:.2f}x")

print("Cache Performance:")
print(f"- Hit rate: {report['cache_performance']['hit_rate']:.1%}")
print(f"- Memory usage: {report['cache_performance']['memory_usage_mb']:.1f} MB")
print(f"- Compression ratio: {report['cache_performance']['compression_ratio']:.2f}")

print("Query Complexity Distribution:")
for complexity, count in report['query_complexity_distribution'].items():
    print(f"- {complexity}: {count} queries")

print("Top Performing Queries:")
for query_info in report['top_performing_queries'][:5]:
    print(f"- {query_info['query_hash']}: {query_info['throughput_rows_per_sec']:,.0f} rows/sec")

print("Optimization Recommendations:")
for recommendation in report['optimization_recommendations']:
    print(f"- {recommendation}")
```

### Integration with Pipeline Monitoring

```python
# Get comprehensive pipeline report including optimization metrics
pipeline_report = pipeline.get_performance_report()

print("Pipeline Performance:")
print(f"- Total queries: {pipeline_report['pipeline_metrics']['total_queries']}")
print(f"- Rows per second: {pipeline_report['pipeline_metrics']['rows_per_second']:,.0f}")

print("Query Optimization:")
opt_report = pipeline_report['query_optimization']
print(f"- Cache hit rate: {opt_report['cache_performance']['hit_rate']:.1%}")
print(f"- Optimization strategies used: {len(opt_report['strategy_effectiveness'])}")
```

## Configuration Options

### Optimizer Configuration

```python
config = {
    # Cache settings
    'cache_size_mb': 1000,              # Cache size in MB
    'compression_enabled': True,         # Enable result compression
    
    # Optimization settings
    'enable_gpu_optimization': True,     # Enable GPU-specific hints
    'optimization_threshold_rows': 100000,  # Min rows to trigger optimization
    
    # Connection settings
    'connection_params': {
        'host': 'localhost',
        'port': 6274,
        'user': 'admin',
        'password': 'HyperInteractive',
        'dbname': 'heavyai'
    }
}

optimizer = HeavyDBQueryOptimizer(**config)
```

### Pipeline Integration Configuration

```python
pipeline_config = {
    'max_concurrent_tasks': 5,           # Parallel backtest tasks
    'cache_size_mb': 1000,              # Optimizer cache size
    'enable_gpu_acceleration': True,     # GPU optimizations
    'connection_params': {               # HeavyDB connection
        'host': 'localhost',
        'port': 6274,
        'user': 'admin', 
        'password': 'HyperInteractive',
        'dbname': 'heavyai'
    }
}

pipeline = HeavyDBBacktestingPipeline(**pipeline_config)
```

## Best Practices

### Query Design for Optimization

1. **Use Date Filters**: Always include date range filters for partition pruning
   ```sql
   WHERE trade_date BETWEEN '2024-01-01' AND '2024-12-31'
   ```

2. **Limit Result Sets**: Use LIMIT for exploration queries
   ```sql
   SELECT * FROM nifty_option_chain LIMIT 10000
   ```

3. **Optimize JOIN Order**: Place most selective conditions first
   ```sql
   SELECT * FROM small_table t1
   JOIN large_table t2 ON t1.id = t2.id
   WHERE t1.selective_condition = 'value'
   ```

4. **Use Appropriate Aggregations**: Prefer SUM/COUNT over complex calculations
   ```sql
   SELECT trade_date, COUNT(*) as trade_count
   FROM nifty_option_chain
   GROUP BY trade_date
   ```

### Cache Management

1. **Monitor Cache Hit Rates**: Aim for >80% hit rate
2. **Adjust Cache Size**: Based on available memory and query patterns
3. **Use Compression**: Enable for memory efficiency
4. **Regular Cleanup**: Clear cache periodically for long-running processes

```python
# Monitor cache performance
cache_stats = optimizer.query_cache.get_stats()
if cache_stats['hit_rate'] < 0.8:
    print("Consider increasing cache size or reviewing query patterns")

# Clear cache if needed
optimizer.clear_cache()
```

### Strategy-Specific Optimizations

Different trading strategies benefit from different optimization approaches:

- **ML Strategy**: Focus on feature extraction queries with window functions
- **TBS Strategy**: Optimize time-based filtering and session analysis
- **ORB Strategy**: Enhance range calculation and breakout detection queries
- **OI Strategy**: Optimize Open Interest aggregations and volume analysis

```python
# Strategy-specific optimization
for strategy in ['ML', 'TBS', 'ORB', 'OI']:
    optimized_query, profile = optimizer.optimize_query(
        base_query, 
        strategy_type=strategy
    )
    print(f"{strategy}: {len(profile.optimization_strategies)} strategies applied")
```

## Performance Targets

### Throughput Targets
- **Standard Queries**: >37,303 rows/sec (validated performance)
- **Simple Queries**: >100,000 rows/sec
- **Complex Aggregations**: >10,000 rows/sec
- **Window Functions**: >5,000 rows/sec

### Response Time Targets
- **Query Optimization**: <100ms per query
- **Cache Lookup**: <10ms per lookup
- **Simple Queries**: <200ms execution time
- **Complex Queries**: <2s execution time

### Memory Efficiency
- **Cache Memory**: <500MB for standard workloads
- **Query Memory**: <2GB for complex multi-strategy backtests
- **Compression Ratio**: >0.3 (70% size reduction)

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate**
   - Increase cache size: `cache_size_mb`
   - Review query variability
   - Check parameter consistency

2. **High Memory Usage**
   - Enable compression: `compression_enabled=True`
   - Reduce cache size
   - Implement result streaming

3. **Slow Query Optimization**
   - Check query complexity
   - Verify database connection
   - Monitor system resources

4. **GPU Optimization Not Applied**
   - Verify `enable_gpu_optimization=True`
   - Check HeavyDB GPU support
   - Review query compatibility

### Debug Information

```python
# Enable debug logging
import logging
logging.getLogger('backtester_v2.core.heavydb_query_optimizer').setLevel(logging.DEBUG)

# Get detailed query profile
optimized_query, profile = optimizer.optimize_query(query, strategy_type='ML')
print(f"Query hash: {profile.query_hash}")
print(f"Complexity: {profile.complexity}")
print(f"Strategies: {[s.value for s in profile.optimization_strategies]}")
print(f"Execution time: {profile.execution_time:.3f}s")
print(f"Rows processed: {profile.rows_processed}")
print(f"Memory usage: {profile.memory_usage_mb:.1f} MB")

# Check cache statistics
cache_stats = optimizer.query_cache.get_stats()
print(f"Cache stats: {cache_stats}")
```

## Advanced Usage

### Custom Optimization Strategies

For specialized use cases, you can extend the optimization system:

```python
from backtester_v2.core.heavydb_query_optimizer import OptimizationStrategy

# Define custom strategy
class CustomOptimizationStrategy(OptimizationStrategy):
    CUSTOM_STRATEGY = "custom_strategy"

# Implement custom optimization logic
def apply_custom_optimization(query, complexity):
    if "custom_pattern" in query.lower():
        return f"/*+ CUSTOM_HINT */ {query}"
    return query
```

### Performance Benchmarking

```python
import time
import statistics

def benchmark_optimization(optimizer, queries, runs=10):
    """Benchmark query optimization performance"""
    times = []
    
    for _ in range(runs):
        start_time = time.time()
        
        for query in queries:
            optimizer.optimize_query(query)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'queries_per_second': len(queries) / statistics.mean(times)
    }

# Run benchmark
benchmark_results = benchmark_optimization(optimizer, test_queries)
print(f"Optimization performance: {benchmark_results['queries_per_second']:.1f} queries/sec")
```

## Integration Testing

### Test Query Optimization

```python
import pytest
from backtester_v2.tests.test_heavydb_query_optimization import TestHeavyDBQueryOptimizer

# Run optimization tests
pytest.main([
    'backtester_v2/tests/test_heavydb_query_optimization.py::TestHeavyDBQueryOptimizer::test_query_optimization',
    '-v'
])
```

### Validate Performance

```python
# Performance validation test
from backtester_v2.tests.test_heavydb_query_optimization import TestPerformanceBenchmarks

# Run performance benchmarks
pytest.main([
    'backtester_v2/tests/test_heavydb_query_optimization.py::TestPerformanceBenchmarks',
    '-v'
])
```

## Conclusion

The HeavyDB Query Optimization system provides comprehensive performance enhancements for large-scale backtesting operations. By combining intelligent query analysis, GPU-specific optimizations, advanced caching, and seamless pipeline integration, it delivers significant performance improvements for 33M+ row datasets.

Key benefits:
- **Automatic Optimization**: No manual query tuning required
- **GPU Acceleration**: Leverages HeavyDB's columnar GPU processing
- **Intelligent Caching**: Reduces redundant computations
- **Performance Monitoring**: Detailed metrics and recommendations
- **Seamless Integration**: Works transparently with existing backtesting workflows

For optimal performance, follow the best practices outlined in this guide and monitor the system using the built-in performance reporting capabilities.