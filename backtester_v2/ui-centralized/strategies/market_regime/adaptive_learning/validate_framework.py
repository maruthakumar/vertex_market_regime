#!/usr/bin/env python3
"""
Simple validation script for the adaptive learning framework.
Tests core functionality without external dependencies.
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic framework imports."""
    print("Testing basic imports...")
    
    try:
        from . import (
            FRAMEWORK_VERSION,
            PERFORMANCE_BUDGET_MS,
            COMPONENT_FEATURE_COUNT,
            TOTAL_FEATURES,
            get_framework_info,
            validate_performance_budget
        )
        print(f"✓ Core framework imports successful")
        print(f"  Framework version: {FRAMEWORK_VERSION}")
        print(f"  Total features: {TOTAL_FEATURES}")
        return True
    except Exception as e:
        print(f"✗ Core framework imports failed: {str(e)}")
        return False

def test_schema_registry():
    """Test schema registry functionality."""
    print("\nTesting schema registry...")
    
    try:
        from .schema_registry import FeatureDefinition, ComponentSchema, SchemaRegistry
        
        # Test feature definition
        feature = FeatureDefinition(
            name="test_feature",
            data_type="float", 
            description="Test feature"
        )
        print(f"✓ FeatureDefinition creation successful")
        
        # Test schema registry with temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = SchemaRegistry(registry_path=Path(temp_dir))
            schemas = registry.get_all_schemas()
            
            print(f"✓ SchemaRegistry created with {len(schemas)} schemas")
            
            # Validate feature counts
            total_features = sum(schema.feature_count for schema in schemas.values())
            if total_features == 774:
                print(f"✓ Total feature count validated: {total_features}")
                return True
            else:
                print(f"✗ Feature count mismatch: {total_features} != 774")
                return False
                
    except Exception as e:
        print(f"✗ Schema registry test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_base_component():
    """Test base component functionality.""" 
    print("\nTesting base component...")
    
    try:
        from .base_component import AdaptiveComponent, ComponentConfig, AnalysisResult
        
        # Test component config
        config = ComponentConfig(
            component_id="test_component",
            feature_count=10,
            processing_budget_ms=100,
            memory_budget_mb=256
        )
        print(f"✓ ComponentConfig creation successful")
        
        # Test analysis result
        result = AnalysisResult(
            component_id="test_component",
            features={"test": 1.0},
            metadata={},
            processing_time_ms=50.0,
            memory_usage_mb=10.0,
            timestamp=time.time(),
            confidence_score=0.95,
            success=True
        )
        print(f"✓ AnalysisResult creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Base component test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_transform_utilities():
    """Test transform utilities."""
    print("\nTesting transform utilities...")
    
    try:
        from .utils.transforms import ArrowToCuDFConverter, MultiTimeframeAggregator, get_optimal_chunk_size
        
        # Test converter creation
        converter = ArrowToCuDFConverter(enable_gpu=False)  # Disable GPU for testing
        print(f"✓ ArrowToCuDFConverter creation successful")
        
        # Test aggregator
        aggregator = MultiTimeframeAggregator(converter)
        print(f"✓ MultiTimeframeAggregator creation successful")
        
        # Test chunk size calculation
        chunk_size = get_optimal_chunk_size(2.0)
        print(f"✓ Optimal chunk size calculation: {chunk_size:,} rows")
        
        return True
        
    except Exception as e:
        print(f"✗ Transform utilities test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_system():
    """Test cache system."""
    print("\nTesting cache system...")
    
    try:
        from .cache.local_cache import LocalFeatureCache, CachePolicy
        
        # Test cache policy
        policy = CachePolicy(ttl_minutes=10, max_size_mb=32)
        print(f"✓ CachePolicy creation successful")
        
        # Test cache creation with temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LocalFeatureCache(
                component_id="test_component",
                ttl_minutes=5,
                cache_dir=Path(temp_dir)
            )
            print(f"✓ LocalFeatureCache creation successful")
            
            # Test basic cache operations
            test_key = "test_key"
            test_data = {"test": "data", "value": 42}
            
            # Put data
            success = cache.put(test_key, test_data)
            if success:
                print(f"✓ Cache put operation successful")
            
            # Get data
            retrieved = cache.get(test_key)
            if retrieved == test_data:
                print(f"✓ Cache get operation successful")
                return True
            else:
                print(f"✗ Cache get operation failed: {retrieved} != {test_data}")
                return False
        
    except Exception as e:
        print(f"✗ Cache system test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance within budgets."""
    print("\nTesting performance...")
    
    try:
        from . import validate_performance_budget
        
        # Test performance validation
        start_time = time.time()
        time.sleep(0.01)  # 10ms operation
        
        # Should pass within 50ms budget
        validate_performance_budget(start_time, 50, "test_operation")
        print(f"✓ Performance validation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {str(e)}")
        return False

def main():
    """Run all validation tests."""
    print("Adaptive Learning Framework Validation")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_schema_registry,
        test_base_component,
        test_transform_utilities,
        test_cache_system,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready for development.")
        return 0
    else:
        print("❌ Some tests failed. Please review implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())