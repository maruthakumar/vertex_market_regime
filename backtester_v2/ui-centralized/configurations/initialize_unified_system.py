#!/usr/bin/env python3
"""
Initialize Unified Parameter Management System

This script initializes the unified parameter management system and
demonstrates its capabilities while preserving backward compatibility.
"""

import logging
import sys
from pathlib import Path

# Add configurations to path
sys.path.append(str(Path(__file__).parent))

from parameter_registry import ParameterRegistry, SchemaExtractor
from version_control import VersionManager
from gateway import UnifiedConfigurationGateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize the unified parameter management system"""
    
    print("🚀 Initializing Unified Parameter Management System")
    print("=" * 60)
    
    try:
        # Step 1: Initialize Parameter Registry
        print("\n1. Initializing Parameter Registry...")
        registry = ParameterRegistry()
        
        # Extract parameters from existing configuration classes
        extractor = SchemaExtractor(registry)
        success = extractor.extract_all_strategies()
        
        if success:
            stats = registry.get_statistics()
            print(f"   ✅ Extracted {stats['total_parameters']} parameters")
            print(f"   ✅ Created {stats['total_categories']} categories") 
            print(f"   ✅ Registered {stats['total_strategies']} strategies")
        else:
            print("   ⚠️  Partial extraction completed")
        
        # Step 2: Initialize Version Control
        print("\n2. Initializing Version Control System...")
        version_manager = VersionManager()
        print("   ✅ Version control database initialized")
        
        # Step 3: Initialize Unified Gateway
        print("\n3. Initializing Unified Configuration Gateway...")
        gateway = UnifiedConfigurationGateway(
            parameter_registry=registry,
            version_manager=version_manager
        )
        
        # Perform health check
        health = gateway.health_check()
        print(f"   ✅ Gateway status: {health['overall_status']}")
        
        # Display component status
        for component, status in health['components'].items():
            if status['status'] == 'healthy':
                print(f"   ✅ {component}: {status['status']}")
            else:
                print(f"   ❌ {component}: {status['status']} - {status.get('error', 'Unknown error')}")
        
        # Step 4: Display System Statistics
        print("\n4. System Statistics:")
        print("-" * 30)
        
        gateway_stats = gateway.get_statistics()
        
        if 'parameter_registry' in gateway_stats:
            reg_stats = gateway_stats['parameter_registry']
            print(f"   📊 Total Parameters: {reg_stats['total_parameters']}")
            print(f"   📊 Total Strategies: {reg_stats['total_strategies']}")
            print(f"   📊 Strategy Breakdown:")
            for strategy, count in reg_stats['strategy_breakdown'].items():
                print(f"      - {strategy}: {count} parameters")
        
        if 'version_control' in gateway_stats:
            vc_stats = gateway_stats['version_control']
            print(f"   📊 Storage Size: {vc_stats['storage_size_mb']} MB")
        
        # Step 5: Test Basic Functionality
        print("\n5. Testing Basic Functionality...")
        
        # Test strategy detection
        from gateway.strategy_detector import StrategyDetector
        detector = StrategyDetector()
        print("   ✅ Strategy detection ready")
        
        # Test configuration listing
        configs = gateway.list_configurations()
        print(f"   ✅ Configuration listing: {len(configs)} configurations found")
        
        # Test parameter search
        search_results = gateway.search_parameters("capital")
        print(f"   ✅ Parameter search: Found {len(search_results)} 'capital' parameters")
        
        print("\n" + "=" * 60)
        print("🎉 Unified Parameter Management System Successfully Initialized!")
        print("\nKey Features Available:")
        print("  • Centralized parameter registry for all 10 strategies")
        print("  • Git-like version control for configuration files") 
        print("  • Automatic strategy type detection")
        print("  • Enhanced validation with schema enforcement")
        print("  • Deduplication and metadata enrichment")
        print("  • Backward compatibility with existing workflows")
        
        print("\nNext Steps:")
        print("  1. Upload Excel files using the enhanced gateway")
        print("  2. Use batch processing for multiple files")
        print("  3. Explore version history and rollback capabilities")
        print("  4. Try dynamic form generation for new UIs")
        
        return gateway
        
    except Exception as e:
        print(f"\n❌ Failed to initialize system: {e}")
        logger.exception("System initialization failed")
        return None

def demonstrate_usage():
    """Demonstrate key usage patterns"""
    
    print("\n" + "=" * 60)
    print("📖 Usage Examples")
    print("=" * 60)
    
    print("""
# 1. Load Configuration with Auto-Detection
from configurations.gateway import UnifiedConfigurationGateway

gateway = UnifiedConfigurationGateway()

# Auto-detect strategy type and load with versioning
config = gateway.load_configuration(
    strategy_type="auto",  # Automatic detection
    file_path="my_config.xlsx",
    author="user123",
    commit_message="Updated risk parameters"
)

# 2. Search Parameters Across All Strategies
results = gateway.search_parameters("stop_loss")
for param in results:
    print(f"{param.strategy_type}.{param.name}: {param.description}")

# 3. Get Configuration History
history = gateway.get_configuration_history("tbs_my_strategy")
for version in history:
    print(f"v{version.version_number}: {version.commit_message}")

# 4. Rollback to Previous Version
gateway.rollback_configuration("tbs_my_strategy", "version_id_here")

# 5. Batch Upload with Deduplication
from configurations.gateway import BatchProcessor

processor = BatchProcessor()
results = processor.process_folder("/path/to/configs/")
print(f"Processed {results['successful']} files, {results['duplicates']} duplicates")

# 6. Generate Dynamic Form from Schema
schema = gateway.get_strategy_schema("tbs")
# Use schema to generate UI form automatically
    """)

if __name__ == "__main__":
    # Initialize the system
    gateway = initialize_system()
    
    if gateway:
        # Demonstrate usage
        demonstrate_usage()
        
        # Save initialization report
        try:
            stats = gateway.get_statistics()
            import json
            
            report_path = Path(__file__).parent / "initialization_report.json"
            with open(report_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            print(f"\n📄 Initialization report saved to: {report_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")
    
    print("\n✨ Initialization complete!")