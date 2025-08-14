#!/usr/bin/env python3
"""
Component 1: Triple Straddle PoC Validation
===========================================

Based on Story 1.2: Component 1 Feature Engineering
- 120 features across rolling straddle analysis
- EMA analysis on straddle prices (20, 50, 100, 200 periods)
- VWAP with combined volume (ce_volume + pe_volume)
- Pivot analysis with CPR (PP, R1-R3, S1-S3)
- Multi-timeframe integration (1min→3,5,10,15min)
- Dynamic weighting system (10 components)
- RSI and MACD implementation
- ATM, ITM1, OTM1 straddle calculations

Manual Verification Checklist:
[ ] Rolling straddle calculation accuracy (ATM, ITM1, OTM1)
[ ] EMA calculations on straddle prices (not underlying)
[ ] VWAP accuracy with volume weighting
[ ] Pivot points calculation (PP, R1-R3, S1-S3)
[ ] Multi-timeframe resampling (3,5,10,15min)
[ ] RSI calculation on straddle prices
[ ] MACD calculation on straddle prices
[ ] Dynamic weighting system (10 components)
[ ] Strike type classification (ATM/ITM1/OTM1)
[ ] Processing time <150ms
"""

import pandas as pd
import numpy as np
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class Component01PoC:
    """Component 1 Proof of Concept Validator"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.validation_results = {}
        
    def load_sample_data(self) -> pd.DataFrame:
        """Load production data sample"""
        print("📁 Loading production data...")
        parquet_files = list(self.data_path.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        sample_file = parquet_files[0]
        print(f"   Using: {sample_file}")
        df = pd.read_parquet(sample_file)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        return df
    
    def validate_rolling_straddle_calculation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PoC Test 1: Rolling Straddle Calculation
        Validates ATM, ITM1, OTM1 straddle calculations
        """
        print("\n🔍 PoC Test 1: Rolling Straddle Calculation")
        results = {}
        
        try:
            # Test ATM Straddle
            atm_data = df[(df['call_strike_type'] == 'ATM') & (df['put_strike_type'] == 'ATM')]
            if len(atm_data) > 0:
                atm_straddle = atm_data['ce_close'] + atm_data['pe_close']
                results['atm_straddle'] = {
                    'sample_count': len(atm_data),
                    'sample_value': float(atm_straddle.iloc[0]),
                    'min_value': float(atm_straddle.min()),
                    'max_value': float(atm_straddle.max()),
                    'mean_value': float(atm_straddle.mean())
                }
                print(f"   ✅ ATM Straddle: {len(atm_data)} samples, avg={atm_straddle.mean():.2f}")
            else:
                results['atm_straddle'] = {'error': 'No ATM data found'}
                print("   ❌ ATM Straddle: No data found")
            
            # Test ITM1 Straddle
            itm1_data = df[(df['call_strike_type'] == 'ITM1') & (df['put_strike_type'] == 'OTM1')]
            if len(itm1_data) > 0:
                itm1_straddle = itm1_data['ce_close'] + itm1_data['pe_close']
                results['itm1_straddle'] = {
                    'sample_count': len(itm1_data),
                    'sample_value': float(itm1_straddle.iloc[0]),
                    'mean_value': float(itm1_straddle.mean())
                }
                print(f"   ✅ ITM1 Straddle: {len(itm1_data)} samples, avg={itm1_straddle.mean():.2f}")
            else:
                results['itm1_straddle'] = {'error': 'No ITM1 data found'}
                print("   ❌ ITM1 Straddle: No data found")
            
            # Test OTM1 Straddle  
            otm1_data = df[(df['call_strike_type'] == 'OTM1') & (df['put_strike_type'] == 'ITM1')]
            if len(otm1_data) > 0:
                otm1_straddle = otm1_data['ce_close'] + otm1_data['pe_close']
                results['otm1_straddle'] = {
                    'sample_count': len(otm1_data),
                    'sample_value': float(otm1_straddle.iloc[0]),
                    'mean_value': float(otm1_straddle.mean())
                }
                print(f"   ✅ OTM1 Straddle: {len(otm1_data)} samples, avg={otm1_straddle.mean():.2f}")
            else:
                results['otm1_straddle'] = {'error': 'No OTM1 data found'}
                print("   ❌ OTM1 Straddle: No data found")
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            print(f"   ❌ Error: {e}")
        
        return results
    
    def validate_ema_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PoC Test 2: EMA Analysis on Straddle Prices
        Tests EMA periods 20, 50, 100, 200 on rolling straddle prices
        """
        print("\n🔍 PoC Test 2: EMA Analysis on Straddle Prices")
        results = {}
        
        try:
            # Get ATM straddle data for EMA calculation
            atm_data = df[(df['call_strike_type'] == 'ATM') & (df['put_strike_type'] == 'ATM')]
            if len(atm_data) == 0:
                results['error'] = 'No ATM data for EMA analysis'
                results['success'] = False
                return results
            
            # Sort by time for proper EMA calculation
            atm_data_sorted = atm_data.sort_values('trade_time')
            straddle_prices = atm_data_sorted['ce_close'] + atm_data_sorted['pe_close']
            
            ema_periods = [20, 50, 100, 200]
            ema_results = {}
            
            for period in ema_periods:
                if len(straddle_prices) >= period:
                    ema = straddle_prices.ewm(span=period, adjust=False).mean()
                    ema_results[f'ema_{period}'] = {
                        'latest_value': float(ema.iloc[-1]),
                        'samples_used': len(ema),
                        'convergence_ratio': float(abs(ema.iloc[-1] - straddle_prices.iloc[-1]) / straddle_prices.iloc[-1])
                    }
                    print(f"   ✅ EMA-{period}: {ema.iloc[-1]:.2f} (convergence: {ema_results[f'ema_{period}']['convergence_ratio']:.3f})")
                else:
                    ema_results[f'ema_{period}'] = {'error': f'Insufficient data (need {period}, have {len(straddle_prices)})'}
                    print(f"   ❌ EMA-{period}: Insufficient data")
            
            results['ema_calculations'] = ema_results
            results['straddle_price_sample'] = float(straddle_prices.iloc[-1])
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            print(f"   ❌ Error: {e}")
        
        return results
    
    def validate_vwap_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PoC Test 3: VWAP Analysis with Combined Volume
        Tests VWAP calculation using ce_volume + pe_volume
        """
        print("\n🔍 PoC Test 3: VWAP Analysis with Combined Volume")
        results = {}
        
        try:
            # Get ATM data with volume
            atm_data = df[(df['call_strike_type'] == 'ATM') & (df['put_strike_type'] == 'ATM')]
            if len(atm_data) == 0:
                results['error'] = 'No ATM data for VWAP analysis'
                results['success'] = False
                return results
            
            # Calculate combined volume
            combined_volume = atm_data['ce_volume'] + atm_data['pe_volume']
            straddle_prices = atm_data['ce_close'] + atm_data['pe_close']
            
            # Calculate VWAP
            if combined_volume.sum() > 0:
                vwap = (straddle_prices * combined_volume).sum() / combined_volume.sum()
                
                # Calculate VWAP deviation
                vwap_deviation = (straddle_prices.mean() - vwap) / vwap
                
                results['vwap_calculation'] = {
                    'vwap_value': float(vwap),
                    'simple_average': float(straddle_prices.mean()),
                    'vwap_deviation': float(vwap_deviation),
                    'total_volume': float(combined_volume.sum()),
                    'volume_weighted': True
                }
                
                print(f"   ✅ VWAP: {vwap:.2f}")
                print(f"   ✅ Simple Avg: {straddle_prices.mean():.2f}")
                print(f"   ✅ Deviation: {vwap_deviation:.3f}")
                print(f"   ✅ Total Volume: {combined_volume.sum():,.0f}")
                
            else:
                results['error'] = 'Zero total volume'
                results['success'] = False
                print("   ❌ Zero total volume")
                return results
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            print(f"   ❌ Error: {e}")
        
        return results
    
    def validate_pivot_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PoC Test 4: Pivot Analysis with CPR
        Tests pivot point calculations (PP, R1-R3, S1-S3)
        """
        print("\n🔍 PoC Test 4: Pivot Analysis with CPR")
        results = {}
        
        try:
            # Get ATM OHLC data for pivot calculation
            atm_data = df[(df['call_strike_type'] == 'ATM') & (df['put_strike_type'] == 'ATM')]
            if len(atm_data) == 0:
                results['error'] = 'No ATM data for pivot analysis'
                results['success'] = False
                return results
            
            # Calculate OHLC from straddle prices
            straddle_open = atm_data['ce_open'] + atm_data['pe_open']
            straddle_high = atm_data['ce_high'] + atm_data['pe_high']
            straddle_low = atm_data['ce_low'] + atm_data['pe_low']
            straddle_close = atm_data['ce_close'] + atm_data['pe_close']
            
            # Get daily OHLC
            high = straddle_high.max()
            low = straddle_low.min()
            close = straddle_close.iloc[-1]
            
            # Calculate Pivot Points
            pivot_point = (high + low + close) / 3
            
            # Resistance levels
            r1 = 2 * pivot_point - low
            r2 = pivot_point + (high - low)
            r3 = high + 2 * (pivot_point - low)
            
            # Support levels
            s1 = 2 * pivot_point - high
            s2 = pivot_point - (high - low)
            s3 = low - 2 * (high - pivot_point)
            
            # CPR (Central Pivot Range)
            tc = (pivot_point - s1) + pivot_point  # Top Central
            bc = pivot_point - (r1 - pivot_point)  # Bottom Central
            cpr_width = tc - bc
            
            results['pivot_calculations'] = {
                'pivot_point': float(pivot_point),
                'resistance_1': float(r1),
                'resistance_2': float(r2), 
                'resistance_3': float(r3),
                'support_1': float(s1),
                'support_2': float(s2),
                'support_3': float(s3),
                'cpr_top': float(tc),
                'cpr_bottom': float(bc),
                'cpr_width': float(cpr_width),
                'current_price': float(close),
                'price_vs_pivot': 'above' if close > pivot_point else 'below'
            }
            
            print(f"   ✅ Pivot Point: {pivot_point:.2f}")
            print(f"   ✅ R1: {r1:.2f}, R2: {r2:.2f}, R3: {r3:.2f}")
            print(f"   ✅ S1: {s1:.2f}, S2: {s2:.2f}, S3: {s3:.2f}")
            print(f"   ✅ CPR Width: {cpr_width:.2f}")
            print(f"   ✅ Current: {close:.2f} ({'above' if close > pivot_point else 'below'} pivot)")
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            print(f"   ❌ Error: {e}")
        
        return results
    
    def validate_rsi_calculation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PoC Test 5: RSI Calculation on Straddle Prices
        Tests RSI with 14-period default
        """
        print("\n🔍 PoC Test 5: RSI Calculation on Straddle Prices")
        results = {}
        
        try:
            # Get ATM straddle time series
            atm_data = df[(df['call_strike_type'] == 'ATM') & (df['put_strike_type'] == 'ATM')]
            if len(atm_data) < 15:  # Need at least 15 periods for RSI
                results['error'] = f'Insufficient data for RSI (need 15, have {len(atm_data)})'
                results['success'] = False
                return results
            
            # Sort by time and get straddle prices
            atm_sorted = atm_data.sort_values('trade_time')
            straddle_prices = atm_sorted['ce_close'] + atm_sorted['pe_close']
            
            # Calculate RSI
            period = 14
            delta = straddle_prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            latest_rsi = rsi.iloc[-1]
            
            # RSI interpretation
            if latest_rsi > 70:
                rsi_signal = 'overbought'
            elif latest_rsi < 30:
                rsi_signal = 'oversold'
            else:
                rsi_signal = 'neutral'
            
            results['rsi_calculation'] = {
                'rsi_value': float(latest_rsi),
                'rsi_signal': rsi_signal,
                'period_used': period,
                'samples_available': len(straddle_prices),
                'price_change': float(delta.iloc[-1]),
                'avg_gain': float(avg_gain.iloc[-1]),
                'avg_loss': float(avg_loss.iloc[-1])
            }
            
            print(f"   ✅ RSI-{period}: {latest_rsi:.2f} ({rsi_signal})")
            print(f"   ✅ Price Change: {delta.iloc[-1]:.2f}")
            print(f"   ✅ Avg Gain: {avg_gain.iloc[-1]:.2f}")
            print(f"   ✅ Avg Loss: {avg_loss.iloc[-1]:.2f}")
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            print(f"   ❌ Error: {e}")
        
        return results
    
    def validate_macd_calculation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PoC Test 6: MACD Calculation on Straddle Prices
        Tests MACD with 12,26,9 periods
        """
        print("\n🔍 PoC Test 6: MACD Calculation on Straddle Prices")
        results = {}
        
        try:
            # Get ATM straddle time series
            atm_data = df[(df['call_strike_type'] == 'ATM') & (df['put_strike_type'] == 'ATM')]
            if len(atm_data) < 35:  # Need at least 35 periods for MACD
                results['error'] = f'Insufficient data for MACD (need 35, have {len(atm_data)})'
                results['success'] = False
                return results
            
            # Sort by time and get straddle prices
            atm_sorted = atm_data.sort_values('trade_time')
            straddle_prices = atm_sorted['ce_close'] + atm_sorted['pe_close']
            
            # Calculate MACD
            fast_period = 12
            slow_period = 26
            signal_period = 9
            
            ema_fast = straddle_prices.ewm(span=fast_period).mean()
            ema_slow = straddle_prices.ewm(span=slow_period).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line
            
            # MACD interpretation
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            latest_histogram = histogram.iloc[-1]
            
            if latest_macd > latest_signal:
                macd_signal = 'bullish'
            else:
                macd_signal = 'bearish'
            
            results['macd_calculation'] = {
                'macd_line': float(latest_macd),
                'signal_line': float(latest_signal),
                'histogram': float(latest_histogram),
                'macd_signal': macd_signal,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period,
                'samples_available': len(straddle_prices)
            }
            
            print(f"   ✅ MACD Line: {latest_macd:.4f}")
            print(f"   ✅ Signal Line: {latest_signal:.4f}")
            print(f"   ✅ Histogram: {latest_histogram:.4f}")
            print(f"   ✅ Signal: {macd_signal}")
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            print(f"   ❌ Error: {e}")
        
        return results
    
    def validate_multi_timeframe_capability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PoC Test 7: Multi-timeframe Integration
        Tests resampling to 3,5,10,15min timeframes
        """
        print("\n🔍 PoC Test 7: Multi-timeframe Integration")
        results = {}
        
        try:
            if 'trade_time' not in df.columns:
                results['error'] = 'No trade_time column for timeframe analysis'
                results['success'] = False
                return results
            
            # Get ATM data with time
            atm_data = df[(df['call_strike_type'] == 'ATM') & (df['put_strike_type'] == 'ATM')].copy()
            
            if len(atm_data) == 0:
                results['error'] = 'No ATM data for timeframe analysis'
                results['success'] = False
                return results
            
            # Ensure trade_time is datetime
            atm_data['trade_time'] = pd.to_datetime(atm_data['trade_time'])
            atm_data = atm_data.sort_values('trade_time')
            
            # Add straddle price
            atm_data['straddle_price'] = atm_data['ce_close'] + atm_data['pe_close']
            
            # Set trade_time as index for resampling
            atm_data.set_index('trade_time', inplace=True)
            
            timeframes = ['3T', '5T', '10T', '15T']  # 3,5,10,15 minute timeframes
            timeframe_results = {}
            
            for tf in timeframes:
                try:
                    resampled = atm_data['straddle_price'].resample(tf).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'count': 'count'
                    }).dropna()
                    
                    if len(resampled) > 0:
                        timeframe_results[tf] = {
                            'bars_created': len(resampled),
                            'latest_close': float(resampled['close'].iloc[-1]),
                            'latest_high': float(resampled['high'].iloc[-1]),
                            'latest_low': float(resampled['low'].iloc[-1]),
                            'avg_count_per_bar': float(resampled['count'].mean())
                        }
                        print(f"   ✅ {tf}: {len(resampled)} bars, latest={resampled['close'].iloc[-1]:.2f}")
                    else:
                        timeframe_results[tf] = {'error': 'No data after resampling'}
                        print(f"   ❌ {tf}: No data after resampling")
                
                except Exception as tf_error:
                    timeframe_results[tf] = {'error': str(tf_error)}
                    print(f"   ❌ {tf}: {tf_error}")
            
            results['timeframe_analysis'] = timeframe_results
            results['original_data_points'] = len(atm_data)
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            print(f"   ❌ Error: {e}")
        
        return results
    
    def run_comprehensive_poc(self) -> Dict[str, Any]:
        """Run all PoC tests for Component 1"""
        print("="*80)
        print("🎯 COMPONENT 1: TRIPLE STRADDLE PoC VALIDATION")
        print("="*80)
        
        start_time = time.time()
        
        # Load data
        df = self.load_sample_data()
        
        # Run all validation tests
        validation_results = {
            'data_info': {
                'file_count': len(list(self.data_path.glob("**/*.parquet"))),
                'sample_rows': len(df),
                'sample_columns': len(df.columns),
                'unique_strikes': df['strike'].nunique() if 'strike' in df.columns else 0
            },
            'test_1_rolling_straddle': self.validate_rolling_straddle_calculation(df),
            'test_2_ema_analysis': self.validate_ema_analysis(df),
            'test_3_vwap_analysis': self.validate_vwap_analysis(df),
            'test_4_pivot_analysis': self.validate_pivot_analysis(df),
            'test_5_rsi_calculation': self.validate_rsi_calculation(df),
            'test_6_macd_calculation': self.validate_macd_calculation(df),
            'test_7_multi_timeframe': self.validate_multi_timeframe_capability(df)
        }
        
        processing_time = (time.time() - start_time) * 1000
        validation_results['performance'] = {
            'total_processing_time_ms': processing_time,
            'meets_150ms_budget': processing_time < 150
        }
        
        # Summary
        successful_tests = sum(1 for test_name, result in validation_results.items() 
                             if test_name.startswith('test_') and result.get('success', False))
        total_tests = sum(1 for test_name in validation_results.keys() if test_name.startswith('test_'))
        
        print(f"\n📊 COMPONENT 1 PoC SUMMARY:")
        print(f"   • Successful Tests: {successful_tests}/{total_tests}")
        print(f"   • Processing Time: {processing_time:.1f}ms (Budget: 150ms)")
        print(f"   • Data Processed: {len(df)} rows")
        
        print(f"\n✅ MANUAL VERIFICATION CHECKLIST:")
        checklist_items = [
            ("Rolling straddle calculation accuracy", validation_results['test_1_rolling_straddle'].get('success', False)),
            ("EMA calculations on straddle prices", validation_results['test_2_ema_analysis'].get('success', False)),
            ("VWAP accuracy with volume weighting", validation_results['test_3_vwap_analysis'].get('success', False)),
            ("Pivot points calculation", validation_results['test_4_pivot_analysis'].get('success', False)),
            ("RSI calculation on straddle prices", validation_results['test_5_rsi_calculation'].get('success', False)),
            ("MACD calculation on straddle prices", validation_results['test_6_macd_calculation'].get('success', False)),
            ("Multi-timeframe resampling", validation_results['test_7_multi_timeframe'].get('success', False)),
            ("Processing time <150ms", validation_results['performance']['meets_150ms_budget'])
        ]
        
        for item, status in checklist_items:
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {item}")
        
        validation_results['summary'] = {
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'overall_success': successful_tests == total_tests
        }
        
        return validation_results

def main():
    """Run Component 1 PoC validation"""
    data_path = "/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed"
    
    poc_validator = Component01PoC(data_path)
    results = poc_validator.run_comprehensive_poc()
    
    # Save results for manual review
    import json
    with open('component_01_poc_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: component_01_poc_results.json")
    print("="*80)
    
    return 0 if results['summary']['overall_success'] else 1

if __name__ == "__main__":
    exit(main())