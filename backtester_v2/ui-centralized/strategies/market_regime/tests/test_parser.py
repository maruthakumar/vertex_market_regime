#!/usr/bin/env python3
"""
Simple Parser Test
=================

Quick test to verify the enhanced Excel parser is working correctly.
"""

import sys
import os
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime')

from enhanced_excel_parser import EnhancedExcelParser, EnhancedSystemType

def test_parser():
    """Test the enhanced Excel parser"""
    print("🧪 Testing Enhanced Excel Parser")
    print("=" * 50)
    
    parser = EnhancedExcelParser()
    
    # Test files
    test_files = [
        "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/enhanced_trending_oi_pa_config.xlsx",
        "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/enhanced_greek_sentiment_config.xlsx",
        "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/enhanced/triple_straddle_analysis_config.xlsx"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            filename = os.path.basename(test_file)
            print(f"\n🔍 Testing: {filename}")
            
            # Detect system type
            system_type = parser.detect_system_type(test_file)
            print(f"📊 System Type: {system_type.value if system_type else 'Unknown'}")
            
            if system_type:
                # Validate structure
                is_valid, errors = parser.validate_excel_structure(test_file, system_type)
                print(f"✅ Structure Valid: {is_valid}")
                if errors:
                    print(f"⚠️ Validation Errors: {len(errors)}")
                    for error in errors[:3]:  # Show first 3 errors
                        print(f"   - {error}")
                
                # Parse configuration
                config = parser.parse_enhanced_system_config(test_file)
                if config:
                    print(f"📋 Parameters Parsed: {len(config.parameters)}")
                    print(f"🔧 System Enabled: {config.enabled}")
                    
                    # Show first few parameters
                    param_names = list(config.parameters.keys())[:5]
                    if param_names:
                        print(f"📄 Sample Parameters: {', '.join(param_names)}")
                else:
                    print("❌ Failed to parse configuration")
            else:
                print("❌ Could not detect system type")
        else:
            print(f"⚠️ File not found: {test_file}")
    
    print("\n" + "=" * 50)
    print("✅ Parser test completed!")

if __name__ == "__main__":
    test_parser()
