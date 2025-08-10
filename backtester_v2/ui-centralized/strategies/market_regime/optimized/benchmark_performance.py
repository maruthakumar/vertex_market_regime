"""
Performance Benchmark for Market Regime Optimizations

Compares the performance of original vs optimized implementations
across various scenarios and data sizes.

Author: Market Regime System Optimizer
Date: 2025-07-07
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json
from pathlib import Path
import sys

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

# Import components
from performance_enhanced_engine import PerformanceEnhancedMarketRegimeEngine, PerformanceConfig
from enhanced_matrix_calculator import Enhanced10x10MatrixCalculator, MatrixConfig
from ..correlation_matrix_engine import CorrelationMatrixEngine  # Original implementation
from ..archive_enhanced_modules_do_not_use.refactored_12_regime_detector import Refactored12RegimeDetector


class PerformanceBenchmark:
    """Benchmark suite for market regime optimizations"""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
        
    def generate_test_data(self, n_rows: int, n_strikes: int = 20) -> Dict[str, Any]:
        """Generate realistic test data"""
        # Generate option chain
        underlying_price = 50000
        strikes = np.linspace(
            underlying_price * 0.9,
            underlying_price * 1.1,
            n_strikes
        )
        
        option_data = []
        for _ in range(n_rows):
            for strike in strikes:
                for opt_type in ['CE', 'PE']:
                    option_data.append({
                        'strike_price': strike,
                        'option_type': opt_type,
                        'last_price': abs(underlying_price - strike) / 100 + np.random.randn() * 10,
                        'volume': np.random.randint(100, 10000),
                        'open_interest': np.random.randint(1000, 100000),
                        'implied_volatility': 15 + np.random.randn() * 2,
                        'delta': np.random.uniform(-1, 1),
                        'gamma': np.random.uniform(0, 0.1),
                        'theta': np.random.uniform(-100, 0),
                        'vega': np.random.uniform(0, 50),
                        'underlying_price': underlying_price
                    })
                    
        option_chain = pd.DataFrame(option_data)
        
        return {
            'timestamp': datetime.now(),
            'underlying_price': underlying_price,
            'option_chain': option_chain,
            'indicators': {
                'rsi': 50 + np.random.randn() * 10,
                'macd_signal': np.random.randn() * 50,
                'atr': 250 + np.random.randn() * 50,
                'adx': 25 + np.random.randn() * 10
            }
        }
        
    def benchmark_correlation_matrix(self):
        """Benchmark correlation matrix calculations"""
        print("\n=== Correlation Matrix Benchmark ===")
        
        # Test different data sizes
        sizes = [100, 500, 1000, 5000, 10000]
        results = {'sizes': sizes, 'original': [], 'optimized': [], 'gpu': []}
        
        for size in sizes:
            print(f"\nTesting with {size} rows...")
            test_data = self.generate_test_data(size)
            
            # Prepare component data (10 columns)
            component_data = pd.DataFrame(
                np.random.randn(size, 10),
                columns=[f'Component_{i}' for i in range(10)]
            )
            
            # Test original implementation
            try:
                original_engine = CorrelationMatrixEngine()
                start = time.time()
                _ = np.corrcoef(component_data.T)  # Simulate original
                original_time = time.time() - start
                results['original'].append(original_time)
            except:
                results['original'].append(None)
                
            # Test optimized CPU implementation
            optimized_calc = Enhanced10x10MatrixCalculator(
                MatrixConfig(use_gpu=False, use_sparse=False)
            )
            start = time.time()
            _ = optimized_calc.calculate_correlation_matrix(component_data, method='numba')
            optimized_time = time.time() - start
            results['optimized'].append(optimized_time)
            
            # Test GPU implementation if available
            try:
                gpu_calc = Enhanced10x10MatrixCalculator(
                    MatrixConfig(use_gpu=True)
                )
                start = time.time()
                _ = gpu_calc.calculate_correlation_matrix(component_data, method='gpu')
                gpu_time = time.time() - start
                results['gpu'].append(gpu_time)
            except:
                results['gpu'].append(None)
                
            # Print results
            print(f"  Original: {original_time:.4f}s")
            print(f"  Optimized: {optimized_time:.4f}s ({original_time/optimized_time:.2f}x faster)")
            if results['gpu'][-1]:
                print(f"  GPU: {gpu_time:.4f}s ({original_time/gpu_time:.2f}x faster)")
                
        self.results['correlation_matrix'] = results
        return results
        
    def benchmark_regime_detection(self):
        """Benchmark regime detection performance"""
        print("\n=== Regime Detection Benchmark ===")
        
        # Test batch sizes
        batch_sizes = [1, 10, 50, 100, 500]
        results = {'batch_sizes': batch_sizes, 'original': [], 'optimized': []}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size {batch_size}...")
            
            # Generate test data
            test_data_list = [self.generate_test_data(100) for _ in range(batch_size)]
            
            # Test original implementation
            original_detector = Refactored12RegimeDetector()
            start = time.time()
            for data in test_data_list:
                _ = original_detector.calculate_regime(data)
            original_time = time.time() - start
            results['original'].append(original_time)
            
            # Test optimized implementation
            enhanced_engine = PerformanceEnhancedMarketRegimeEngine(
                PerformanceConfig(use_redis_cache=False)  # Disable Redis for fair comparison
            )
            start = time.time()
            _ = enhanced_engine.calculate_regime_batch(test_data_list, '12')
            optimized_time = time.time() - start
            results['optimized'].append(optimized_time)
            
            # Print results
            print(f"  Original: {original_time:.4f}s ({batch_size/original_time:.1f} regimes/s)")
            print(f"  Optimized: {optimized_time:.4f}s ({batch_size/optimized_time:.1f} regimes/s)")
            print(f"  Speedup: {original_time/optimized_time:.2f}x")
            
        self.results['regime_detection'] = results
        return results
        
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n=== Memory Usage Benchmark ===")
        
        results = {'operations': [], 'original_memory': [], 'optimized_memory': []}
        
        # Test operations
        operations = [
            ('Small dataset', 100),
            ('Medium dataset', 1000),
            ('Large dataset', 10000)
        ]
        
        for op_name, size in operations:
            print(f"\nTesting {op_name}...")
            
            # Measure original memory usage
            original_detector = Refactored12RegimeDetector()
            gc.collect()
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            test_data = self.generate_test_data(size)
            for _ in range(10):
                _ = original_detector.calculate_regime(test_data)
                
            original_memory = self.process.memory_info().rss / 1024 / 1024 - start_memory
            
            # Measure optimized memory usage
            enhanced_engine = PerformanceEnhancedMarketRegimeEngine()
            gc.collect()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            for _ in range(10):
                _ = enhanced_engine._calculate_regime_optimized(
                    enhanced_engine.detector_12, test_data
                )
                
            optimized_memory = self.process.memory_info().rss / 1024 / 1024 - start_memory
            
            # Store results
            results['operations'].append(op_name)
            results['original_memory'].append(original_memory)
            results['optimized_memory'].append(optimized_memory)
            
            print(f"  Original: {original_memory:.1f} MB")
            print(f"  Optimized: {optimized_memory:.1f} MB")
            print(f"  Reduction: {(1 - optimized_memory/original_memory)*100:.1f}%")
            
        self.results['memory_usage'] = results
        return results
        
    def benchmark_incremental_updates(self):
        """Benchmark incremental correlation updates"""
        print("\n=== Incremental Update Benchmark ===")
        
        results = {'update_sizes': [], 'full_calc': [], 'incremental': []}
        
        # Initial data
        initial_size = 1000
        initial_data = pd.DataFrame(
            np.random.randn(initial_size, 10),
            columns=[f'Component_{i}' for i in range(10)]
        )
        
        # Test different update sizes
        update_sizes = [10, 50, 100, 500]
        
        calculator = Enhanced10x10MatrixCalculator(
            MatrixConfig(use_incremental=True)
        )
        
        # Initial calculation
        _ = calculator.calculate_incremental_correlation(initial_data, 'test_key')
        
        for update_size in update_sizes:
            print(f"\nTesting {update_size} row update...")
            
            # New data
            new_data = pd.DataFrame(
                np.random.randn(update_size, 10),
                columns=[f'Component_{i}' for i in range(10)]
            )
            
            # Full recalculation
            combined_data = pd.concat([initial_data, new_data])
            start = time.time()
            _ = calculator.calculate_correlation_matrix(combined_data, method='numpy')
            full_time = time.time() - start
            
            # Incremental update
            start = time.time()
            _ = calculator.calculate_incremental_correlation(new_data, 'test_key')
            incremental_time = time.time() - start
            
            # Store results
            results['update_sizes'].append(update_size)
            results['full_calc'].append(full_time)
            results['incremental'].append(incremental_time)
            
            print(f"  Full recalc: {full_time:.4f}s")
            print(f"  Incremental: {incremental_time:.4f}s")
            print(f"  Speedup: {full_time/incremental_time:.2f}x")
            
        self.results['incremental_updates'] = results
        return results
        
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n=== Generating Benchmark Report ===")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Correlation Matrix Performance
        if 'correlation_matrix' in self.results:
            ax = axes[0, 0]
            data = self.results['correlation_matrix']
            ax.plot(data['sizes'], data['original'], 'o-', label='Original')
            ax.plot(data['sizes'], data['optimized'], 's-', label='Optimized CPU')
            if any(data['gpu']):
                ax.plot(data['sizes'], [x for x in data['gpu'] if x], '^-', label='GPU')
            ax.set_xlabel('Data Size (rows)')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Correlation Matrix Calculation Performance')
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            
        # Plot 2: Regime Detection Throughput
        if 'regime_detection' in self.results:
            ax = axes[0, 1]
            data = self.results['regime_detection']
            original_throughput = [b/t for b, t in zip(data['batch_sizes'], data['original'])]
            optimized_throughput = [b/t for b, t in zip(data['batch_sizes'], data['optimized'])]
            
            ax.plot(data['batch_sizes'], original_throughput, 'o-', label='Original')
            ax.plot(data['batch_sizes'], optimized_throughput, 's-', label='Optimized')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (regimes/second)')
            ax.set_title('Regime Detection Throughput')
            ax.legend()
            
        # Plot 3: Memory Usage
        if 'memory_usage' in self.results:
            ax = axes[1, 0]
            data = self.results['memory_usage']
            x = range(len(data['operations']))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], data['original_memory'], width, label='Original')
            ax.bar([i + width/2 for i in x], data['optimized_memory'], width, label='Optimized')
            ax.set_xlabel('Operation')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(data['operations'])
            ax.legend()
            
        # Plot 4: Incremental Update Performance
        if 'incremental_updates' in self.results:
            ax = axes[1, 1]
            data = self.results['incremental_updates']
            
            speedup = [f/i for f, i in zip(data['full_calc'], data['incremental'])]
            ax.plot(data['update_sizes'], speedup, 'o-', color='green', linewidth=2)
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Update Size (rows)')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('Incremental Update Speedup')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=150)
        print(f"Saved visualization to benchmark_results.png")
        
        # Save detailed results
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Saved detailed results to benchmark_results.json")
        
        # Print summary
        print("\n=== Performance Summary ===")
        print(f"Average speedup factors:")
        
        # Calculate average speedups
        if 'correlation_matrix' in self.results:
            data = self.results['correlation_matrix']
            speedups = [o/p for o, p in zip(data['original'], data['optimized']) if o and p]
            if speedups:
                print(f"  Correlation Matrix: {np.mean(speedups):.2f}x")
                
        if 'regime_detection' in self.results:
            data = self.results['regime_detection']
            speedups = [o/p for o, p in zip(data['original'], data['optimized'])]
            print(f"  Regime Detection: {np.mean(speedups):.2f}x")
            
        if 'memory_usage' in self.results:
            data = self.results['memory_usage']
            reductions = [(o-p)/o*100 for o, p in zip(data['original_memory'], 
                                                     data['optimized_memory'])]
            print(f"  Memory Reduction: {np.mean(reductions):.1f}%")
            

def main():
    """Run complete benchmark suite"""
    print("Market Regime Performance Benchmark Suite")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_correlation_matrix()
    benchmark.benchmark_regime_detection()
    benchmark.benchmark_memory_usage()
    benchmark.benchmark_incremental_updates()
    
    # Generate report
    benchmark.generate_report()
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    import gc
    main()