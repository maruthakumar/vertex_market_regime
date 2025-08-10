#!/usr/bin/env python3
"""
Triple Straddle Analysis System Demonstration

This script demonstrates the complete functionality of the Triple Straddle Analysis
system including regime detection, weight optimization, and Excel configuration.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_triple_straddle_system():
    """Demonstrate the complete Triple Straddle Analysis system"""
    
    logger.info("🚀 Triple Straddle Analysis System Demonstration")
    logger.info("=" * 60)
    
    # Step 1: Initialize the system
    logger.info("📋 Step 1: Initializing Triple Straddle Analysis System...")
    
    from triple_straddle_analysis import TripleStraddleAnalysisEngine
    from dynamic_weight_optimizer import DynamicWeightOptimizer
    from triple_straddle_excel_config import TripleStraddleExcelConfig
    
    # Configuration
    config = {
        'learning_rate': 0.05,
        'performance_window': 50,
        'min_performance_threshold': 0.6,
        'weight_bounds': (0.05, 0.60)
    }
    
    # Initialize components
    engine = TripleStraddleAnalysisEngine(config)
    optimizer = DynamicWeightOptimizer(config)
    excel_config = TripleStraddleExcelConfig()
    
    logger.info("✅ System initialized successfully")
    
    # Step 2: Generate Excel configuration
    logger.info("\n📊 Step 2: Generating Excel Configuration Template...")
    
    try:
        config_path = excel_config.generate_excel_template("demo_triple_straddle_config.xlsx")
        logger.info(f"✅ Excel template generated: {config_path}")
    except Exception as e:
        logger.warning(f"⚠️ Excel generation skipped (missing openpyxl): {e}")
    
    # Step 3: Demonstrate market data analysis
    logger.info("\n🔍 Step 3: Analyzing Market Data with Triple Straddle System...")
    
    # Generate realistic market scenarios
    scenarios = [
        ("Bullish High Volatility", generate_bullish_high_vol_data()),
        ("Bearish Low Volatility", generate_bearish_low_vol_data()),
        ("Sideways Normal Volatility", generate_sideways_normal_vol_data()),
        ("Volatile Uncertain", generate_volatile_uncertain_data())
    ]
    
    results = []
    
    for scenario_name, market_data in scenarios:
        logger.info(f"\n  📈 Analyzing: {scenario_name}")
        
        # Analyze with Triple Straddle Engine
        result = engine.analyze_market_regime(market_data)
        
        # Display results
        logger.info(f"    Score: {result['triple_straddle_score']:.3f}")
        logger.info(f"    Confidence: {result['confidence']:.3f}")
        
        # Component breakdown
        if 'component_results' in result:
            logger.info("    Component Breakdown:")
            for component, comp_result in result['component_results'].items():
                logger.info(f"      {component}: {comp_result.component_score:.3f}")
        
        results.append((scenario_name, result))
    
    # Step 4: Demonstrate weight optimization
    logger.info("\n⚖️ Step 4: Demonstrating Dynamic Weight Optimization...")
    
    # Generate performance history
    performance_data = generate_performance_history(100)
    market_conditions = {'volatility': 0.18, 'time_of_day': 11}
    
    # Get initial weights
    initial_weights = optimizer.get_current_weights()
    logger.info("Initial Weights:")
    logger.info(f"  Component: {initial_weights['component']}")
    
    # Optimize weights
    optimization_result = optimizer.optimize_weights(performance_data, market_conditions)
    
    if optimization_result.validation_passed:
        logger.info("✅ Weight optimization successful!")
        logger.info(f"  Performance improvement: {optimization_result.performance_improvement:.3f}")
        logger.info(f"  Confidence: {optimization_result.confidence_score:.3f}")
        
        optimized_weights = optimization_result.optimized_weights
        logger.info("Optimized Weights:")
        logger.info(f"  Component: {optimized_weights['component']}")
    else:
        logger.info("⚠️ Weight optimization validation failed")
    
    # Step 5: Performance comparison
    logger.info("\n📊 Step 5: Performance Analysis Summary...")
    
    # Calculate statistics
    scores = [result[1]['triple_straddle_score'] for result in results]
    confidences = [result[1]['confidence'] for result in results]
    
    logger.info(f"Score Statistics:")
    logger.info(f"  Mean: {np.mean(scores):.3f}")
    logger.info(f"  Std: {np.std(scores):.3f}")
    logger.info(f"  Range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    
    logger.info(f"Confidence Statistics:")
    logger.info(f"  Mean: {np.mean(confidences):.3f}")
    logger.info(f"  Std: {np.std(confidences):.3f}")
    logger.info(f"  Range: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")
    
    # Step 6: System capabilities summary
    logger.info("\n🎯 Step 6: System Capabilities Summary...")
    
    capabilities = [
        "✅ ATM/ITM1/OTM1 Straddle Analysis",
        "✅ Multi-timeframe Analysis (3, 5, 10, 15 min)",
        "✅ EMA/VWAP/Pivot Technical Analysis",
        "✅ Dynamic Weight Optimization",
        "✅ Performance Tracking & Validation",
        "✅ Excel-based Configuration",
        "✅ Real-time Market Adaptation",
        "✅ Comprehensive Error Handling"
    ]
    
    for capability in capabilities:
        logger.info(f"  {capability}")
    
    logger.info("\n🎉 Triple Straddle Analysis System Demonstration Complete!")
    logger.info("=" * 60)
    
    return results

def generate_bullish_high_vol_data():
    """Generate bullish high volatility market data"""
    np.random.seed(1)
    underlying_price = 18600  # Higher price
    strikes = [18500, 18600, 18700]  # ITM1, ATM, OTM1
    
    options_data = {
        18500: {  # ITM1
            'CE': {'close': 125.5, 'volume': 8000, 'oi': 60000, 'iv': 0.22},
            'PE': {'close': 25.3, 'volume': 4000, 'oi': 35000, 'iv': 0.20}
        },
        18600: {  # ATM
            'CE': {'close': 85.2, 'volume': 12000, 'oi': 90000, 'iv': 0.21},
            'PE': {'close': 85.8, 'volume': 11000, 'oi': 85000, 'iv': 0.21}
        },
        18700: {  # OTM1
            'CE': {'close': 45.1, 'volume': 6000, 'oi': 50000, 'iv': 0.23},
            'PE': {'close': 145.4, 'volume': 3000, 'oi': 30000, 'iv': 0.22}
        }
    }
    
    # Generate upward trending price history
    price_history = []
    base_price = 18500
    for i in range(300):
        trend = i * 0.3  # Upward trend
        noise = np.random.normal(0, 30)  # High volatility
        price = base_price + trend + noise
        price_history.append({
            'close': price,
            'high': price * 1.015,
            'low': price * 0.985,
            'volume': np.random.randint(2000, 8000),
            'timestamp': datetime.now() - timedelta(minutes=300-i)
        })
    
    return {
        'underlying_price': underlying_price,
        'strikes': strikes,
        'options_data': options_data,
        'price_history': price_history,
        'greek_data': {'delta': 0.65, 'gamma': 0.035, 'theta': -0.15, 'vega': 0.35},
        'oi_data': {'call_oi': 600000, 'put_oi': 400000, 'call_volume': 35000, 'put_volume': 20000}
    }

def generate_bearish_low_vol_data():
    """Generate bearish low volatility market data"""
    np.random.seed(2)
    underlying_price = 18400  # Lower price
    strikes = [18300, 18400, 18500]  # ITM1, ATM, OTM1
    
    options_data = {
        18300: {  # ITM1
            'CE': {'close': 115.5, 'volume': 3000, 'oi': 40000, 'iv': 0.12},
            'PE': {'close': 15.3, 'volume': 2000, 'oi': 25000, 'iv': 0.11}
        },
        18400: {  # ATM
            'CE': {'close': 65.2, 'volume': 5000, 'oi': 60000, 'iv': 0.13},
            'PE': {'close': 65.8, 'volume': 6000, 'oi': 65000, 'iv': 0.13}
        },
        18500: {  # OTM1
            'CE': {'close': 25.1, 'volume': 2000, 'oi': 30000, 'iv': 0.14},
            'PE': {'close': 125.4, 'volume': 4000, 'oi': 45000, 'iv': 0.12}
        }
    }
    
    # Generate downward trending price history
    price_history = []
    base_price = 18500
    for i in range(300):
        trend = -i * 0.2  # Downward trend
        noise = np.random.normal(0, 15)  # Low volatility
        price = base_price + trend + noise
        price_history.append({
            'close': price,
            'high': price * 1.005,
            'low': price * 0.995,
            'volume': np.random.randint(1000, 3000),
            'timestamp': datetime.now() - timedelta(minutes=300-i)
        })
    
    return {
        'underlying_price': underlying_price,
        'strikes': strikes,
        'options_data': options_data,
        'price_history': price_history,
        'greek_data': {'delta': 0.35, 'gamma': 0.015, 'theta': -0.08, 'vega': 0.18},
        'oi_data': {'call_oi': 300000, 'put_oi': 500000, 'call_volume': 15000, 'put_volume': 30000}
    }

def generate_sideways_normal_vol_data():
    """Generate sideways normal volatility market data"""
    np.random.seed(3)
    underlying_price = 18500  # Stable price
    strikes = [18400, 18500, 18600]  # ITM1, ATM, OTM1
    
    options_data = {
        18400: {  # ITM1
            'CE': {'close': 110.5, 'volume': 5000, 'oi': 50000, 'iv': 0.16},
            'PE': {'close': 10.3, 'volume': 3000, 'oi': 30000, 'iv': 0.15}
        },
        18500: {  # ATM
            'CE': {'close': 75.2, 'volume': 8000, 'oi': 80000, 'iv': 0.17},
            'PE': {'close': 75.8, 'volume': 7500, 'oi': 75000, 'iv': 0.17}
        },
        18600: {  # OTM1
            'CE': {'close': 35.1, 'volume': 4000, 'oi': 40000, 'iv': 0.18},
            'PE': {'close': 135.4, 'volume': 3500, 'oi': 35000, 'iv': 0.16}
        }
    }
    
    # Generate sideways price history
    price_history = []
    base_price = 18500
    for i in range(300):
        noise = np.random.normal(0, 25)  # Normal volatility
        price = base_price + noise
        price_history.append({
            'close': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'volume': np.random.randint(1500, 4000),
            'timestamp': datetime.now() - timedelta(minutes=300-i)
        })
    
    return {
        'underlying_price': underlying_price,
        'strikes': strikes,
        'options_data': options_data,
        'price_history': price_history,
        'greek_data': {'delta': 0.50, 'gamma': 0.025, 'theta': -0.12, 'vega': 0.28},
        'oi_data': {'call_oi': 450000, 'put_oi': 450000, 'call_volume': 25000, 'put_volume': 25000}
    }

def generate_volatile_uncertain_data():
    """Generate volatile uncertain market data"""
    np.random.seed(4)
    underlying_price = 18520  # Slightly higher
    strikes = [18450, 18520, 18590]  # ITM1, ATM, OTM1
    
    options_data = {
        18450: {  # ITM1
            'CE': {'close': 95.5, 'volume': 7000, 'oi': 55000, 'iv': 0.28},
            'PE': {'close': 25.3, 'volume': 5000, 'oi': 40000, 'iv': 0.26}
        },
        18520: {  # ATM
            'CE': {'close': 78.2, 'volume': 10000, 'oi': 85000, 'iv': 0.29},
            'PE': {'close': 78.8, 'volume': 9500, 'oi': 80000, 'iv': 0.29}
        },
        18590: {  # OTM1
            'CE': {'close': 42.1, 'volume': 6000, 'oi': 45000, 'iv': 0.31},
            'PE': {'close': 112.4, 'volume': 4000, 'oi': 38000, 'iv': 0.27}
        }
    }
    
    # Generate volatile uncertain price history
    price_history = []
    base_price = 18500
    for i in range(300):
        # Random walk with high volatility
        if i == 0:
            price = base_price
        else:
            change = np.random.choice([-1, 1]) * np.random.exponential(20)
            price = price_history[i-1]['close'] + change
        
        price_history.append({
            'close': price,
            'high': price * 1.02,
            'low': price * 0.98,
            'volume': np.random.randint(3000, 10000),
            'timestamp': datetime.now() - timedelta(minutes=300-i)
        })
    
    return {
        'underlying_price': underlying_price,
        'strikes': strikes,
        'options_data': options_data,
        'price_history': price_history,
        'greek_data': {'delta': 0.48, 'gamma': 0.040, 'theta': -0.18, 'vega': 0.42},
        'oi_data': {'call_oi': 520000, 'put_oi': 480000, 'call_volume': 40000, 'put_volume': 35000}
    }

def generate_performance_history(count):
    """Generate realistic performance history"""
    from dynamic_weight_optimizer import PerformanceMetrics
    
    performance_data = []
    for i in range(count):
        # Simulate improving performance over time
        base_accuracy = 0.65 + (i / count) * 0.15  # Improve from 65% to 80%
        
        perf = PerformanceMetrics(
            accuracy=base_accuracy + np.random.normal(0, 0.05),
            precision=base_accuracy + np.random.normal(0, 0.05),
            recall=base_accuracy + np.random.normal(0, 0.05),
            f1_score=base_accuracy + np.random.normal(0, 0.05),
            confidence_avg=0.7 + np.random.normal(0, 0.1),
            regime_stability=0.8 + np.random.normal(0, 0.1),
            timestamp=datetime.now() - timedelta(minutes=count-i)
        )
        
        # Clip to valid ranges
        perf.accuracy = np.clip(perf.accuracy, 0.0, 1.0)
        perf.precision = np.clip(perf.precision, 0.0, 1.0)
        perf.recall = np.clip(perf.recall, 0.0, 1.0)
        perf.f1_score = np.clip(perf.f1_score, 0.0, 1.0)
        perf.confidence_avg = np.clip(perf.confidence_avg, 0.0, 1.0)
        perf.regime_stability = np.clip(perf.regime_stability, 0.0, 1.0)
        
        performance_data.append(perf)
    
    return performance_data

if __name__ == '__main__':
    # Run demonstration
    results = demonstrate_triple_straddle_system()
    
    logger.info("\n🎯 Demonstration completed successfully!")
    logger.info(f"Analyzed {len(results)} market scenarios with Triple Straddle Analysis")
