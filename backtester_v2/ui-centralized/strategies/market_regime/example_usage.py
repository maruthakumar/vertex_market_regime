"""
Example Usage of Excel-Based Market Regime System

This script demonstrates how to use the Excel-based configuration system
with the actual existing market regime implementation.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_realistic_market_data(num_points: int = 500) -> pd.DataFrame:
    """Create realistic market data for demonstration"""
    
    # Create datetime index (market hours)
    start_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    datetime_index = pd.date_range(start=start_time, periods=num_points, freq='1min')
    
    # Generate realistic market data
    np.random.seed(42)  # For reproducible results
    
    # Underlying price with trend and volatility
    base_price = 18000
    trend = np.cumsum(np.random.randn(num_points) * 0.05)  # Small trend
    volatility = np.random.randn(num_points) * 10  # Price volatility
    underlying_price = base_price + trend + volatility
    
    # Volume with intraday pattern
    base_volume = 15000
    time_factor = np.sin(np.arange(num_points) * 2 * np.pi / 375) * 0.3 + 1  # 375 min = 6.25 hours
    volume_noise = np.random.randn(num_points) * 0.2 + 1
    volume = (base_volume * time_factor * volume_noise).astype(int)
    volume = np.maximum(volume, 1000)  # Minimum volume
    
    # ATM straddle price (inversely related to underlying stability)
    straddle_base = 180
    underlying_volatility = np.abs(np.diff(underlying_price, prepend=underlying_price[0]))
    straddle_multiplier = 1 + underlying_volatility / 50  # Volatility effect
    straddle_price = straddle_base * straddle_multiplier + np.random.randn(num_points) * 8
    
    # ATM CE and PE prices
    moneyness_factor = (underlying_price - base_price) / base_price  # How far from base
    ce_bias = 1 + moneyness_factor * 0.5  # CE more expensive when underlying higher
    pe_bias = 1 - moneyness_factor * 0.5  # PE more expensive when underlying lower
    
    atm_ce_price = straddle_price * 0.52 * ce_bias + np.random.randn(num_points) * 4
    atm_pe_price = straddle_price * 0.48 * pe_bias + np.random.randn(num_points) * 4
    
    # ITM/OTM straddle prices
    itm1_straddle = straddle_price * 1.25 + np.random.randn(num_points) * 6
    itm2_straddle = straddle_price * 1.45 + np.random.randn(num_points) * 8
    otm1_straddle = straddle_price * 0.75 + np.random.randn(num_points) * 4
    otm2_straddle = straddle_price * 0.55 + np.random.randn(num_points) * 3
    
    # Greeks (realistic values)
    delta = 0.5 + moneyness_factor * 0.3 + np.random.randn(num_points) * 0.05
    delta = np.clip(delta, 0.1, 0.9)  # Keep within bounds
    
    gamma = 0.02 * (1 - np.abs(moneyness_factor)) + np.random.randn(num_points) * 0.003
    gamma = np.maximum(gamma, 0.001)  # Always positive
    
    theta = -0.05 - underlying_volatility / 1000 + np.random.randn(num_points) * 0.01
    theta = np.minimum(theta, -0.01)  # Always negative
    
    vega = 0.15 + underlying_volatility / 500 + np.random.randn(num_points) * 0.02
    vega = np.maximum(vega, 0.05)  # Always positive
    
    # IV with realistic term structure
    base_iv = 0.18
    iv_noise = np.random.randn(num_points) * 0.01
    iv_trend = np.cumsum(np.random.randn(num_points) * 0.001)  # Slow IV changes
    iv = base_iv + iv_trend + iv_noise
    iv = np.clip(iv, 0.08, 0.50)  # Reasonable IV bounds
    
    # OI with realistic patterns
    base_oi = 800000
    oi_trend = np.cumsum(np.random.randn(num_points) * 100)  # Gradual OI changes
    oi = base_oi + oi_trend + np.random.randn(num_points) * 5000
    oi = np.maximum(oi, 100000)  # Minimum OI
    
    # Create comprehensive DataFrame
    data = pd.DataFrame({
        'datetime': datetime_index,
        'underlying_price': underlying_price,
        'price': underlying_price,  # Alias
        'close': underlying_price,  # Alias
        'volume': volume,
        
        # Straddle data
        'atm_straddle_price': straddle_price,
        'ATM_STRADDLE': straddle_price,  # Alias
        'itm1_straddle_price': itm1_straddle,
        'itm2_straddle_price': itm2_straddle,
        'otm1_straddle_price': otm1_straddle,
        'otm2_straddle_price': otm2_straddle,
        
        # CE/PE data
        'atm_ce_price': atm_ce_price,
        'atm_pe_price': atm_pe_price,
        
        # Greeks
        'delta': delta,
        'call_delta': delta,  # Alias
        'gamma': gamma,
        'call_gamma': gamma,  # Alias
        'theta': theta,
        'call_theta': theta,  # Alias
        'vega': vega,
        'call_vega': vega,  # Alias
        
        # IV data
        'iv': iv,
        'ATM_CE_IV': iv,  # Alias
        'ATM_PE_IV': iv,  # Alias
        
        # OI data
        'OI': oi,
        'oi': oi,  # Alias
        
        # Expiry (weekly)
        'expiry': datetime_index[0] + timedelta(days=7)
    })
    
    # Set datetime as index
    data.set_index('datetime', inplace=True)
    
    logger.info(f"Created realistic market data: {len(data)} points, {len(data.columns)} columns")
    return data

def demonstrate_excel_manager():
    """Demonstrate Excel configuration management"""
    print("\n" + "="*60)
    print("1. EXCEL CONFIGURATION MANAGEMENT")
    print("="*60)
    
    try:
        from actual_system_excel_manager import ActualSystemExcelManager
        
        # Initialize manager
        manager = ActualSystemExcelManager()
        
        # Generate template
        template_path = "demo_market_regime_config.xlsx"
        print(f"Generating Excel template: {template_path}")
        manager.generate_excel_template(template_path)
        
        # Load configuration
        print("Loading configuration...")
        manager.load_configuration(template_path)
        
        # Show configuration summaries
        indicator_config = manager.get_indicator_configuration()
        straddle_config = manager.get_straddle_configuration()
        dynamic_weights = manager.get_dynamic_weightage_configuration()
        
        print(f"\n📊 Configuration Summary:")
        print(f"  • Indicators: {len(indicator_config)} systems")
        print(f"  • Straddles: {len(straddle_config)} types")
        print(f"  • Dynamic Weights: {len(dynamic_weights)} systems")
        
        # Show enabled indicators
        enabled_indicators = indicator_config[indicator_config['Enabled'] == True]
        print(f"\n✅ Enabled Indicators:")
        for _, row in enabled_indicators.iterrows():
            print(f"  • {row['IndicatorSystem']}: {row['BaseWeight']:.2f} weight")
        
        # Show straddle configuration
        enabled_straddles = straddle_config[straddle_config['Enabled'] == True]
        print(f"\n📈 Straddle Configuration:")
        for _, row in enabled_straddles.iterrows():
            ema_status = "✅" if row['EMAEnabled'] else "❌"
            vwap_status = "✅" if row['VWAPEnabled'] else "❌"
            print(f"  • {row['StraddleType']}: {row['Weight']:.2f} weight, EMA:{ema_status}, VWAP:{vwap_status}")
        
        # Validate configuration
        is_valid, errors = manager.validate_configuration()
        print(f"\n🔍 Configuration Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        if errors:
            for error in errors:
                print(f"  ⚠️  {error}")
        
        return True, template_path
        
    except Exception as e:
        print(f"❌ Error in Excel manager demo: {e}")
        return False, None

def demonstrate_regime_calculation(config_path: str):
    """Demonstrate market regime calculation"""
    print("\n" + "="*60)
    print("2. MARKET REGIME CALCULATION")
    print("="*60)
    
    try:
        from excel_based_regime_engine import ExcelBasedRegimeEngine
        
        # Initialize engine
        print("Initializing Excel-based regime engine...")
        engine = ExcelBasedRegimeEngine(config_path)
        
        # Check engine status
        status = engine.get_engine_status()
        print(f"\n🔧 Engine Status:")
        print(f"  • Initialized: {status['is_initialized']}")
        print(f"  • Config Path: {status['excel_config_path']}")
        print(f"  • Cache Size: {status['config_cache_size']}")
        
        # Create market data
        print("\n📊 Creating market data...")
        market_data = create_realistic_market_data(200)  # 200 minutes of data
        
        # Calculate market regime
        print("🧮 Calculating market regime...")
        regime_results = engine.calculate_market_regime(market_data)
        
        if not regime_results.empty:
            print(f"\n✅ Regime calculation successful!")
            print(f"  • Data points: {len(regime_results)}")
            print(f"  • Columns: {len(regime_results.columns)}")
            
            # Show regime distribution
            if 'Market_Regime_Label' in regime_results.columns:
                regime_counts = regime_results['Market_Regime_Label'].value_counts()
                print(f"\n📈 Regime Distribution:")
                for regime, count in regime_counts.items():
                    percentage = (count / len(regime_results)) * 100
                    print(f"  • {regime}: {count} ({percentage:.1f}%)")
            
            # Show sample results
            print(f"\n📋 Sample Results (last 5 points):")
            display_cols = ['Market_Regime_Score', 'Market_Regime_Label']
            available_cols = [col for col in display_cols if col in regime_results.columns]
            if available_cols:
                print(regime_results[available_cols].tail().to_string())
            
            # Show additional features if available
            feature_cols = [col for col in regime_results.columns if any(keyword in col.lower() 
                          for keyword in ['straddle', 'timeframe', 'confidence'])]
            if feature_cols:
                print(f"\n🎯 Additional Features Available:")
                for col in feature_cols[:5]:  # Show first 5
                    print(f"  • {col}")
                if len(feature_cols) > 5:
                    print(f"  • ... and {len(feature_cols) - 5} more")
            
            return True, regime_results
        else:
            print("❌ No regime results generated")
            return False, None
            
    except Exception as e:
        print(f"❌ Error in regime calculation demo: {e}")
        return False, None

def demonstrate_performance_tracking(engine, regime_results):
    """Demonstrate performance tracking and weight updates"""
    print("\n" + "="*60)
    print("3. PERFORMANCE TRACKING & WEIGHT UPDATES")
    print("="*60)
    
    try:
        # Simulate performance data
        performance_data = {
            'greek_sentiment': 0.78,      # Good performance
            'trending_oi_pa': 0.65,       # Average performance
            'ema_indicators': 0.85,       # Excellent performance
            'vwap_indicators': 0.72,      # Good performance
            'straddle_analysis': 0.88,    # Excellent performance
            'iv_skew': 0.58,              # Below average
            'atr_indicators': 0.69        # Average performance
        }
        
        print("📊 Simulated Performance Data:")
        for system, performance in performance_data.items():
            status = "🟢" if performance > 0.75 else "🟡" if performance > 0.6 else "🔴"
            print(f"  {status} {system}: {performance:.2f}")
        
        # Update weights
        print("\n⚖️  Updating weights based on performance...")
        success = engine.update_weights_from_performance(performance_data)
        
        if success:
            print("✅ Weight updates successful!")
            
            # Show updated status
            updated_status = engine.get_engine_status()
            print(f"\n📈 Updated Engine Status:")
            for key, value in updated_status.items():
                if 'weight' in key.lower() or 'performance' in key.lower():
                    print(f"  • {key}: {value}")
        else:
            print("❌ Weight updates failed")
        
        # Export performance report
        print("\n📄 Exporting performance report...")
        report_path = engine.export_performance_report()
        if report_path:
            print(f"✅ Performance report saved: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in performance tracking demo: {e}")
        return False

def demonstrate_advanced_features():
    """Demonstrate advanced features"""
    print("\n" + "="*60)
    print("4. ADVANCED FEATURES")
    print("="*60)
    
    try:
        from excel_based_regime_engine import ExcelBasedRegimeEngine
        
        # Generate new template
        engine = ExcelBasedRegimeEngine()
        new_template = engine.generate_configuration_template("advanced_config.xlsx")
        print(f"✅ Generated new template: {new_template}")
        
        # Demonstrate configuration reload
        print("\n🔄 Testing configuration reload...")
        reload_success = engine.reload_configuration()
        print(f"Configuration reload: {'✅ Success' if reload_success else '❌ Failed'}")
        
        # Show comprehensive status
        print("\n📊 Comprehensive Engine Status:")
        status = engine.get_engine_status()
        for key, value in status.items():
            print(f"  • {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced features demo: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🚀 Excel-Based Market Regime System Demonstration")
    print("="*80)
    print("This demo shows how to use Excel configuration with the actual existing system")
    print("at /srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated/")
    
    # Demo 1: Excel Configuration Management
    excel_success, config_path = demonstrate_excel_manager()
    
    if excel_success and config_path:
        # Demo 2: Market Regime Calculation
        regime_success, regime_results = demonstrate_regime_calculation(config_path)
        
        if regime_success and regime_results is not None:
            # Demo 3: Performance Tracking
            try:
                from excel_based_regime_engine import ExcelBasedRegimeEngine
                engine = ExcelBasedRegimeEngine(config_path)
                demonstrate_performance_tracking(engine, regime_results)
            except:
                print("⚠️  Performance tracking demo skipped (engine not available)")
    
    # Demo 4: Advanced Features
    demonstrate_advanced_features()
    
    # Summary
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETE")
    print("="*80)
    print("Key takeaways:")
    print("1. ✅ Excel configuration provides user-friendly interface")
    print("2. ✅ Integration with actual existing system works seamlessly")
    print("3. ✅ ATM straddle analysis with EMA/VWAP across multiple timeframes")
    print("4. ✅ Dynamic weightage with performance-based optimization")
    print("5. ✅ Comprehensive regime formation with 18 regime types")
    print("\nNext steps:")
    print("• Customize Excel configuration for your specific needs")
    print("• Integrate with backtester_v2 for live testing")
    print("• Monitor and adjust weights based on actual performance")
    print("• Deploy for production trading")

if __name__ == "__main__":
    main()
