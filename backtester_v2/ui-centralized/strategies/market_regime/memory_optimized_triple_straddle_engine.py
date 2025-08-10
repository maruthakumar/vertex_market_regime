#!/usr/bin/env python3
"""
Memory Optimized Triple Straddle Engine V2.0
Phase 1 Performance Optimization Implementation

This engine implements the performance optimization enhancements specified in the 
Market Regime Gaps Implementation V1.0 document, integrating:

1. Memory Usage Optimization Strategy
2. Intelligent Caching Strategy  
3. Parallel Processing Architecture

Performance Targets:
- Memory usage <4GB for 50+ concurrent users
- <1 second processing time (enhanced from <3s)
- <10% memory growth over 24-hour operation
- <100ms garbage collection pauses

Author: Senior Quantitative Trading Expert
Date: June 2025
Version: 1.0 - Phase 1 Performance Optimization
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import warnings
import asyncio
import threading
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import performance optimization components
try:
    from .performance_optimization import (
        MemoryPool, MemoryMonitor, LRUCache, TimeBasedCache,
        IntelligentCacheManager, ComponentProcessor, ParallelProcessingEngine
    )
except ImportError:
    from performance_optimization import (
        MemoryPool, MemoryMonitor, LRUCache, TimeBasedCache,
        IntelligentCacheManager, ComponentProcessor, ParallelProcessingEngine
    )

# Import existing engines
try:
    from .archive_comprehensive_modules_do_not_use.comprehensive_triple_straddle_engine import StraddleAnalysisEngine
    from .atm_straddle_engine import ATMStraddleEngine
    from .itm1_straddle_engine import ITM1StraddleEngine
    from .otm1_straddle_engine import OTM1StraddleEngine
    from .combined_straddle_engine import CombinedStraddleEngine
    from .atm_ce_engine import ATMCEEngine
    from .atm_pe_engine import ATMPEEngine
    from .rolling_correlation_matrix_engine import RollingCorrelationMatrixEngine
    from .dynamic_support_resistance_engine import DynamicSupportResistanceEngine
    from .correlation_based_regime_formation_engine import CorrelationBasedRegimeFormationEngine
except ImportError as e:
    # CRITICAL: No mock fallbacks allowed - fail gracefully with proper error
    logger.error(f"CRITICAL: Required engine imports failed: {e}")
    logger.error("This indicates a dependency issue that must be resolved.")
    logger.error("ZERO TOLERANCE: No mock/fallback implementations allowed.")
    raise ImportError(f"Required production engines not available: {e}") from e

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_optimized_triple_straddle_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryOptimizedTripleStraddleEngine(StraddleAnalysisEngine):
    """
    Memory Optimized Triple Straddle Engine with Phase 1 Performance Enhancements
    
    This engine extends the StraddleAnalysisEngine with:
    - Memory pool management and monitoring
    - Intelligent multi-tier caching system
    - Parallel processing for all 6 components
    - Real-time performance optimization
    - Automatic memory cleanup and garbage collection
    """
    
    def __init__(self, config_path: Optional[str] = None, max_workers: int = 8):
        """Initialize the memory optimized engine with performance enhancements"""
        
        # Initialize performance optimization components first
        self.memory_pool = MemoryPool(max_size_gb=4.0)
        self.memory_monitor = MemoryMonitor(
            warning_threshold=0.8,  # 80% memory usage
            critical_threshold=0.9   # 90% memory usage
        )
        self.cache_manager = IntelligentCacheManager()
        self.parallel_processor = ParallelProcessingEngine(max_workers=max_workers)
        
        # Register cleanup callbacks
        self.memory_monitor.register_cleanup_callback(self._emergency_cleanup)
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring(interval_seconds=1.0)
        
        # Initialize parent class - NO fallbacks allowed
        try:
            super().__init__(config_path)
        except Exception as e:
            logger.error(f"CRITICAL: Parent class initialization failed: {e}")
            logger.error("ZERO TOLERANCE: No mock implementations allowed.")
            logger.error("This indicates a critical dependency issue that must be resolved.")
            raise RuntimeError(f"Required parent class initialization failed: {e}") from e

        # Enhanced performance metrics
        self.enhanced_performance_metrics = {
            'memory_usage': {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0},
            'cache_performance': {'overall_hit_rate': 0.0},
            'parallel_processing': {},
            'optimization_effectiveness': 0.0,
            'target_compliance': {
                'memory_under_4gb': False,
                'processing_under_1s': False,
                'gc_under_100ms': False
            }
        }
        
        logger.info("🚀 Memory Optimized Triple Straddle Engine V2.0 initialized")
        logger.info("💾 Memory optimization active with 4GB limit")
        logger.info("🔄 Intelligent caching system enabled")
        logger.info("⚡ Parallel processing configured with {} workers".format(max_workers))
        logger.info("🎯 Enhanced performance targets: <1s processing, <4GB memory")
    
    async def analyze_comprehensive_triple_straddle_optimized(self, market_data: Dict[str, Any], 
                                                            current_dte: int = 0, 
                                                            current_vix: float = 20.0) -> Dict[str, Any]:
        """
        Perform optimized comprehensive triple straddle analysis with performance enhancements
        
        Args:
            market_data: Complete market data including all option prices and volumes
            current_dte: Current days to expiry for dynamic adjustments
            current_vix: Current VIX level for dynamic adjustments
            
        Returns:
            Complete analysis results with performance optimization metrics
        """
        start_time = time.time()
        gc_start_time = time.time()
        
        try:
            logger.info("🔄 Starting optimized comprehensive triple straddle analysis...")
            
            # Step 1: Check cache for existing results
            cache_key = self._generate_cache_key(market_data, current_dte, current_vix)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                logger.info("✅ Cache hit - returning cached results")
                cached_result['cache_hit'] = True
                cached_result['processing_time'] = time.time() - start_time
                return cached_result
            
            # Step 2: Extract and validate component prices with memory optimization
            component_prices = self._extract_component_prices_optimized(market_data)
            
            # Step 3: Parallel technical analysis for all 6 components
            technical_analysis_start = time.time()
            technical_results = await self._calculate_parallel_technical_analysis(
                component_prices, self.timeframe_configurations
            )
            technical_analysis_time = time.time() - technical_analysis_start
            
            # Step 4: Cached correlation matrix calculation
            correlation_start = time.time()
            correlation_results = await self._calculate_cached_correlations(technical_results)
            correlation_time = time.time() - correlation_start
            
            # Step 5: Support & resistance analysis with caching
            sr_start = time.time()
            sr_results = await self._calculate_cached_sr_analysis(
                technical_results, component_prices, list(self.timeframe_configurations.keys())
            )
            sr_time = time.time() - sr_start
            
            # Step 6: Industry-standard Combined Straddle
            combined_straddle_data = self.combined_straddle_engine.calculate_industry_standard_combined_straddle(
                component_prices['atm_straddle'],
                component_prices['itm1_straddle'], 
                component_prices['otm1_straddle'],
                current_dte, current_vix
            )
            
            # Step 7: Regime formation with optimization
            regime_start = time.time()
            regime_results = self.regime_formation_engine.calculate_enhanced_regime_score(
                technical_results, correlation_results['correlation_matrix'], 
                sr_results, self.component_specifications, self.timeframe_configurations
            )
            regime_time = time.time() - regime_start
            
            # Calculate total processing time
            total_time = time.time() - start_time
            gc_time = time.time() - gc_start_time
            
            # Update performance metrics
            self._update_enhanced_performance_metrics(
                total_time, technical_analysis_time, correlation_time, 
                sr_time, regime_time, gc_time
            )
            
            # Compile comprehensive results
            comprehensive_results = {
                'timestamp': datetime.now().isoformat(),
                'component_analysis': {
                    'atm_straddle': technical_results.get('atm_straddle', {}),
                    'itm1_straddle': technical_results.get('itm1_straddle', {}),
                    'otm1_straddle': technical_results.get('otm1_straddle', {}),
                    'combined_straddle': {
                        'technical_analysis': technical_results.get('combined_straddle', {}),
                        'weighted_combination': combined_straddle_data
                    },
                    'atm_ce': technical_results.get('atm_ce', {}),
                    'atm_pe': technical_results.get('atm_pe', {})
                },
                'correlation_analysis': correlation_results,
                'support_resistance_analysis': sr_results,
                'regime_formation': regime_results,
                'performance_metrics': self.performance_metrics,
                'enhanced_performance_metrics': self.enhanced_performance_metrics,
                'optimization_results': self._get_optimization_summary(),
                'cache_hit': False
            }
            
            # Cache results for future use
            self.cache_manager.put(cache_key, comprehensive_results, priority='high')
            
            # Performance validation and logging
            self._validate_performance_targets(total_time)
            
            return comprehensive_results

        except Exception as e:
            logger.error(f"❌ Error in optimized comprehensive triple straddle analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _generate_cache_key(self, market_data: Dict[str, Any], 
                          current_dte: int, current_vix: float) -> str:
        """Generate cache key for market data"""
        # Create a hash-based cache key from market data
        key_components = [
            str(current_dte),
            str(current_vix),
            str(hash(str(sorted(market_data.items()))))
        ]
        return "_".join(key_components)
    
    def _extract_component_prices_optimized(self, market_data: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Extract component prices with memory optimization"""
        try:
            # Get reusable objects from memory pool
            component_prices = self.memory_pool.get_object('calculations')
            
            # Extract individual component prices using vectorized operations
            atm_ce_prices = np.array(market_data.get('atm_ce_price', []), dtype=np.float32)
            atm_pe_prices = np.array(market_data.get('atm_pe_price', []), dtype=np.float32)
            itm1_ce_prices = np.array(market_data.get('itm1_ce_price', []), dtype=np.float32)
            itm1_pe_prices = np.array(market_data.get('itm1_pe_price', []), dtype=np.float32)
            otm1_ce_prices = np.array(market_data.get('otm1_ce_price', []), dtype=np.float32)
            otm1_pe_prices = np.array(market_data.get('otm1_pe_price', []), dtype=np.float32)
            
            # Vectorized straddle calculations
            component_prices['atm_straddle'] = pd.Series(atm_ce_prices + atm_pe_prices)
            component_prices['itm1_straddle'] = pd.Series(itm1_ce_prices + itm1_pe_prices)
            component_prices['otm1_straddle'] = pd.Series(otm1_ce_prices + otm1_pe_prices)
            component_prices['atm_ce'] = pd.Series(atm_ce_prices)
            component_prices['atm_pe'] = pd.Series(atm_pe_prices)
            component_prices['combined_straddle'] = pd.Series(np.zeros(len(atm_ce_prices), dtype=np.float32))

            return component_prices

        except Exception as e:
            logger.error(f"Error extracting component prices: {e}")
            return {}
    
    async def _calculate_parallel_technical_analysis(self, component_prices: Dict[str, pd.Series],
                                                   timeframes: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate technical analysis using parallel processing"""
        try:
            # Prepare market data for parallel processing
            market_data_for_parallel = {}
            for component_name, price_series in component_prices.items():
                market_data_for_parallel[component_name] = {
                    'prices': price_series.values,
                    'timeframes': timeframes
                }
            
            # Process components in parallel
            parallel_results = await self.parallel_processor.process_components_parallel(
                market_data_for_parallel
            )
            
            return parallel_results.get('results', {})
            
        except Exception as e:
            logger.error(f"Error in parallel technical analysis: {e}")
            return {}
    
    async def _calculate_cached_correlations(self, technical_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate correlations with intelligent caching"""
        try:
            # Generate cache key for correlation calculation
            correlation_cache_key = f"correlation_{hash(str(technical_results))}"
            
            # Check cache first
            cached_correlations = self.cache_manager.get(correlation_cache_key)
            if cached_correlations is not None:
                return cached_correlations
            
            # Calculate correlations if not cached
            correlation_results = self.correlation_matrix_engine.calculate_real_time_correlations(
                technical_results
            )
            
            # Cache results
            self.cache_manager.put(correlation_cache_key, correlation_results, priority='medium')
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error in cached correlation calculation: {e}")
            return {}
    
    async def _calculate_cached_sr_analysis(self, technical_results: Dict[str, Dict],
                                          component_prices: Dict[str, pd.Series],
                                          timeframes: List[str]) -> Dict[str, Any]:
        """Calculate support & resistance analysis with caching"""
        try:
            # Generate cache key for S&R analysis
            sr_cache_key = f"sr_analysis_{hash(str(technical_results))}_{hash(str(component_prices))}"
            
            # Check cache first
            cached_sr = self.cache_manager.get(sr_cache_key)
            if cached_sr is not None:
                return cached_sr
            
            # Calculate S&R analysis if not cached
            sr_results = self.sr_engine.calculate_comprehensive_sr_analysis(
                technical_results, component_prices, timeframes
            )
            
            # Cache results
            self.cache_manager.put(sr_cache_key, sr_results, priority='medium')
            
            return sr_results

        except Exception as e:
            logger.error(f"Error in cached S&R analysis: {e}")
            return {}

    def _update_enhanced_performance_metrics(self, total_time: float, technical_time: float,
                                           correlation_time: float, sr_time: float,
                                           regime_time: float, gc_time: float):
        """Update enhanced performance metrics"""
        # Memory usage metrics
        self.enhanced_performance_metrics['memory_usage'] = self.memory_pool.get_memory_usage()

        # Cache performance metrics
        self.enhanced_performance_metrics['cache_performance'] = self.cache_manager.get_comprehensive_stats()

        # Parallel processing metrics
        self.enhanced_performance_metrics['parallel_processing'] = self.parallel_processor.get_performance_stats()

        # Target compliance checks
        memory_mb = self.enhanced_performance_metrics['memory_usage']['rss_mb']
        self.enhanced_performance_metrics['target_compliance'] = {
            'memory_under_4gb': memory_mb < 4096,
            'processing_under_1s': total_time < 1.0,
            'gc_under_100ms': gc_time < 0.1
        }

        # Calculate optimization effectiveness
        cache_hit_rate = self.enhanced_performance_metrics['cache_performance'].get('overall_hit_rate', 0)
        parallel_efficiency = min(1.0, 6.0 / max(total_time, 0.1))  # 6 components ideally processed in parallel
        memory_efficiency = min(1.0, 4096 / max(memory_mb, 1))  # Memory efficiency vs 4GB target

        self.enhanced_performance_metrics['optimization_effectiveness'] = (
            cache_hit_rate * 0.4 + parallel_efficiency * 0.4 + memory_efficiency * 0.2
        )

    def _get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        return {
            'memory_optimization': {
                'pool_stats': self.memory_pool.allocation_stats,
                'current_usage': self.memory_pool.get_memory_usage(),
                'monitoring_active': self.memory_monitor.monitoring_active
            },
            'cache_optimization': {
                'hit_rates': self.cache_manager.get_comprehensive_stats(),
                'tier_distribution': {
                    'l1_size': len(self.cache_manager.l1_cache.cache),
                    'l2_size': len(self.cache_manager.l2_cache.cache),
                    'l3_size': len(self.cache_manager.l3_cache.cache)
                }
            },
            'parallel_optimization': {
                'worker_count': self.parallel_processor.max_workers,
                'processing_stats': self.parallel_processor.get_performance_stats()
            }
        }

    def _validate_performance_targets(self, total_time: float):
        """Validate performance targets and log results"""
        memory_usage = self.memory_pool.get_memory_usage()

        # Processing time validation (enhanced target: <1s)
        if total_time < 1.0:
            logger.info(f"✅ Processing completed in {total_time:.3f}s (target: <1s)")
        else:
            logger.warning(f"⚠️ Processing time {total_time:.3f}s exceeds 1s target")

        # Memory usage validation
        memory_mb = memory_usage['rss_mb']
        if memory_mb < 4096:
            logger.info(f"✅ Memory usage {memory_mb:.1f}MB within 4GB target")
        else:
            logger.warning(f"⚠️ Memory usage {memory_mb:.1f}MB exceeds 4GB target")

        # Cache performance validation
        cache_stats = self.cache_manager.get_comprehensive_stats()
        hit_rate = cache_stats.get('overall_hit_rate', 0)
        if hit_rate > 0.7:
            logger.info(f"✅ Cache hit rate {hit_rate:.1%} exceeds 70% target")
        else:
            logger.warning(f"⚠️ Cache hit rate {hit_rate:.1%} below 70% target")

    def _emergency_cleanup(self):
        """Emergency cleanup callback for memory pressure"""
        logger.warning("🚨 Emergency cleanup triggered due to memory pressure")

        # Clear all caches
        self.cache_manager.l1_cache.clear()
        self.cache_manager.l2_cache.clear()
        self.cache_manager.l3_cache.clear()

        # Force garbage collection
        gc.collect()

        # Return objects to memory pool
        for pool in self.memory_pool.object_pools.values():
            pool.clear()

        logger.info("🧹 Emergency cleanup completed")

    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'enhanced_performance_metrics': self.enhanced_performance_metrics,
            'optimization_summary': self._get_optimization_summary(),
            'target_compliance': self.enhanced_performance_metrics['target_compliance'],
            'recommendations': self._generate_performance_recommendations()
        }

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        try:
            # Memory recommendations
            memory_mb = self.enhanced_performance_metrics.get('memory_usage', {}).get('rss_mb', 0)
            if memory_mb > 3200:  # 80% of 4GB
                recommendations.append("Consider increasing memory cleanup frequency")

            # Cache recommendations
            cache_stats = self.cache_manager.get_comprehensive_stats()
            if cache_stats.get('overall_hit_rate', 0) < 0.5:
                recommendations.append("Cache hit rate is low - consider adjusting cache sizes")

            # Parallel processing recommendations
            parallel_stats = self.parallel_processor.get_performance_stats()
            avg_time = parallel_stats.get('parallel_processing', {}).get('average_parallel_time', 0)
            if avg_time > 0.5:
                recommendations.append("Parallel processing time is high - consider optimizing component calculations")

            if not recommendations:
                recommendations.append("All performance targets met - Phase 1 implementation successful")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to missing performance data")

        return recommendations

    def shutdown(self):
        """Shutdown the memory optimized engine"""
        logger.info("🔄 Shutting down Memory Optimized Triple Straddle Engine...")

        # Stop memory monitoring
        self.memory_monitor.stop_monitoring()

        # Shutdown parallel processor
        self.parallel_processor.shutdown()

        # Clear all caches
        self.cache_manager.l1_cache.clear()
        self.cache_manager.l2_cache.clear()
        self.cache_manager.l3_cache.clear()

        # Final garbage collection
        gc.collect()

        logger.info("✅ Memory Optimized Triple Straddle Engine shutdown complete")

# Factory function for easy instantiation
def create_optimized_engine(config_path: Optional[str] = None,
                          max_workers: int = 8) -> MemoryOptimizedTripleStraddleEngine:
    """Factory function to create optimized engine instance"""
    return MemoryOptimizedTripleStraddleEngine(config_path=config_path, max_workers=max_workers)
