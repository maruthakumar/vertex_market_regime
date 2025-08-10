#!/usr/bin/env python3
"""
Triple Straddle Analysis System Validation

This script validates the implementation of the Triple Straddle Analysis system
without complex imports, focusing on core functionality validation.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_triple_straddle_analysis():
    """Validate Triple Straddle Analysis Engine"""
    logger.info("🔍 Validating Triple Straddle Analysis Engine...")
    
    try:
        from triple_straddle_analysis import TripleStraddleAnalysisEngine
        
        # Initialize engine
        config = {'learning_rate': 0.05, 'performance_window': 20}
        engine = TripleStraddleAnalysisEngine(config)
        
        # Generate test market data
        test_data = generate_test_market_data()
        
        # Test analysis
        result = engine.analyze_market_regime(test_data)
        
        # Validate result structure
        required_keys = ['triple_straddle_score', 'confidence', 'component_results', 'weights_used']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Validate score ranges
        assert -1.0 <= result['triple_straddle_score'] <= 1.0, "Score out of range"
        assert 0.0 <= result['confidence'] <= 1.0, "Confidence out of range"
        
        logger.info("✅ Triple Straddle Analysis Engine validation PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Triple Straddle Analysis Engine validation FAILED: {e}")
        return False

def validate_dynamic_weight_optimizer():
    """Validate Dynamic Weight Optimizer"""
    logger.info("🔍 Validating Dynamic Weight Optimizer...")
    
    try:
        from dynamic_weight_optimizer import DynamicWeightOptimizer, PerformanceMetrics
        
        # Initialize optimizer
        config = {'learning_rate': 0.05, 'performance_window': 20}
        optimizer = DynamicWeightOptimizer(config)
        
        # Generate test performance data
        performance_data = []
        for i in range(20):
            perf = PerformanceMetrics(
                accuracy=0.7 + np.random.normal(0, 0.1),
                precision=0.65 + np.random.normal(0, 0.1),
                recall=0.6 + np.random.normal(0, 0.1),
                f1_score=0.62 + np.random.normal(0, 0.1),
                confidence_avg=0.7 + np.random.normal(0, 0.1),
                regime_stability=0.8 + np.random.normal(0, 0.1),
                timestamp=datetime.now() - timedelta(minutes=20-i)
            )
            # Clip values to valid ranges
            perf.accuracy = np.clip(perf.accuracy, 0.0, 1.0)
            perf.precision = np.clip(perf.precision, 0.0, 1.0)
            perf.recall = np.clip(perf.recall, 0.0, 1.0)
            perf.f1_score = np.clip(perf.f1_score, 0.0, 1.0)
            perf.confidence_avg = np.clip(perf.confidence_avg, 0.0, 1.0)
            perf.regime_stability = np.clip(perf.regime_stability, 0.0, 1.0)
            
            performance_data.append(perf)
        
        # Test optimization
        market_conditions = {'volatility': 0.15, 'time_of_day': 10}
        result = optimizer.optimize_weights(performance_data, market_conditions)
        
        # Validate result structure (result is a WeightOptimizationResult object)
        assert hasattr(result, 'optimized_weights'), "Missing optimized_weights attribute"
        assert hasattr(result, 'performance_improvement'), "Missing performance_improvement attribute"
        assert hasattr(result, 'confidence_score'), "Missing confidence_score attribute"
        assert hasattr(result, 'validation_passed'), "Missing validation_passed attribute"

        # Validate weight structure
        weights = result.optimized_weights
        assert 'pillar' in weights, "Missing pillar weights"
        assert 'indicator' in weights, "Missing indicator weights"
        assert 'component' in weights, "Missing component weights"
        
        # Validate weight normalization
        for level_weights in weights.values():
            total_weight = sum(level_weights.values())
            assert abs(total_weight - 1.0) < 0.1, f"Weights not normalized: {total_weight}"
        
        logger.info("✅ Dynamic Weight Optimizer validation PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dynamic Weight Optimizer validation FAILED: {e}")
        return False

def validate_excel_config():
    """Validate Excel Configuration System"""
    logger.info("🔍 Validating Excel Configuration System...")
    
    try:
        from triple_straddle_excel_config import TripleStraddleExcelConfig
        
        # Initialize config
        excel_config = TripleStraddleExcelConfig()
        
        # Test default configuration
        default_config = excel_config._get_default_configuration()
        
        # Validate structure
        required_sections = ['triple_straddle', 'timeframes', 'technical_analysis', 'performance_tracking']
        for section in required_sections:
            assert section in default_config, f"Missing config section: {section}"
        
        # Validate triple straddle weights
        ts_config = default_config['triple_straddle']
        weight_sum = ts_config['atm_weight'] + ts_config['itm1_weight'] + ts_config['otm1_weight']
        assert abs(weight_sum - 1.0) < 0.01, f"Triple straddle weights don't sum to 1: {weight_sum}"
        
        # Validate timeframe weights
        tf_weights = default_config['timeframes']
        tf_sum = sum(tf_weights.values())
        assert abs(tf_sum - 1.0) < 0.01, f"Timeframe weights don't sum to 1: {tf_sum}"
        
        logger.info("✅ Excel Configuration System validation PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Excel Configuration System validation FAILED: {e}")
        return False

def validate_system_integration():
    """Validate overall system integration"""
    logger.info("🔍 Validating System Integration...")
    
    try:
        # Test component interaction
        from triple_straddle_analysis import TripleStraddleAnalysisEngine
        from dynamic_weight_optimizer import DynamicWeightOptimizer
        
        # Initialize components
        config = {'learning_rate': 0.05, 'performance_window': 20}
        engine = TripleStraddleAnalysisEngine(config)
        optimizer = DynamicWeightOptimizer(config)
        
        # Test data flow
        test_data = generate_test_market_data()
        
        # Analyze with engine
        analysis_result = engine.analyze_market_regime(test_data)
        
        # Get current weights from optimizer
        current_weights = optimizer.get_current_weights()
        
        # Validate integration
        assert 'component' in current_weights, "Missing component weights"
        assert 'atm_straddle' in current_weights['component'], "Missing ATM straddle weight"
        
        # Test weight application
        component_weights = current_weights['component']
        assert abs(sum(component_weights.values()) - 1.0) < 0.1, "Component weights not normalized"
        
        logger.info("✅ System Integration validation PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ System Integration validation FAILED: {e}")
        return False

def generate_test_market_data():
    """Generate realistic test market data"""
    np.random.seed(42)
    
    # Base parameters
    underlying_price = 18500 + np.random.normal(0, 100)
    strikes = [underlying_price - 100, underlying_price, underlying_price + 100]
    
    # Generate options data
    options_data = {}
    for strike in strikes:
        ce_price = max(0, underlying_price - strike + np.random.normal(0, 10))
        pe_price = max(0, strike - underlying_price + np.random.normal(0, 10))
        
        options_data[strike] = {
            'CE': {
                'close': ce_price,
                'volume': np.random.randint(1000, 10000),
                'oi': np.random.randint(10000, 100000),
                'iv': 0.15 + np.random.normal(0, 0.05)
            },
            'PE': {
                'close': pe_price,
                'volume': np.random.randint(1000, 10000),
                'oi': np.random.randint(10000, 100000),
                'iv': 0.15 + np.random.normal(0, 0.05)
            }
        }
    
    # Generate price history
    price_history = []
    for i in range(300):
        price = underlying_price + np.random.normal(0, 50)
        price_history.append({
            'close': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'volume': np.random.randint(1000, 5000),
            'timestamp': datetime.now() - timedelta(minutes=300-i)
        })
    
    return {
        'underlying_price': underlying_price,
        'strikes': strikes,
        'options_data': options_data,
        'price_history': price_history,
        'greek_data': {
            'delta': np.random.normal(0.5, 0.2),
            'gamma': np.random.normal(0.02, 0.01),
            'theta': np.random.normal(-0.1, 0.05),
            'vega': np.random.normal(0.3, 0.1)
        },
        'oi_data': {
            'call_oi': np.random.randint(100000, 1000000),
            'put_oi': np.random.randint(100000, 1000000),
            'call_volume': np.random.randint(10000, 100000),
            'put_volume': np.random.randint(10000, 100000)
        }
    }

def main():
    """Main validation function"""
    logger.info("🚀 Starting Triple Straddle Analysis System Validation...")
    
    # Run all validations
    validations = [
        ("Triple Straddle Analysis Engine", validate_triple_straddle_analysis),
        ("Dynamic Weight Optimizer", validate_dynamic_weight_optimizer),
        ("Excel Configuration System", validate_excel_config),
        ("System Integration", validate_system_integration)
    ]
    
    results = []
    for name, validation_func in validations:
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ {name} validation FAILED with exception: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{name}: {status}")
        if result:
            passed += 1
    
    logger.info("="*60)
    logger.info(f"Overall Result: {passed}/{total} validations passed")
    
    if passed == total:
        logger.info("🎉 ALL VALIDATIONS PASSED! System is ready for deployment.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} validations failed. Please review and fix issues.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
