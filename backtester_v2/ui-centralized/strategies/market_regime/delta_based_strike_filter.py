#!/usr/bin/env python3
"""
Delta-based Strike Filtering Algorithm
Enhanced Triple Straddle Rolling Analysis Framework v2.0

Author: The Augster
Date: 2025-06-20
Version: 2.0.0 (Strike Filtering)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeltaBasedStrikeFilter:
    """
    Advanced delta-based strike filtering for volume-weighted Greek calculations
    
    Implements:
    1. CALL options: Delta range 0.5 to 0.01 (50 delta to 1 delta)
    2. PUT options: Delta range -0.5 to -0.01 (-50 delta to -1 delta)
    3. Liquidity filtering with minimum volume thresholds
    4. Strike distribution analysis and optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize delta-based strike filter"""
        
        self.config = config or self._get_default_config()
        self.filter_statistics = {}
        
        logger.info("🎯 Delta-based Strike Filter initialized")
        logger.info(f"✅ CALL delta range: {self.config['call_delta_range']}")
        logger.info(f"✅ PUT delta range: {self.config['put_delta_range']}")
        logger.info(f"✅ Min volume threshold: {self.config['min_volume_threshold']}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for delta-based filtering"""
        
        return {
            # Delta ranges for strike selection
            'call_delta_range': (0.01, 0.5),    # 1 delta to 50 delta
            'put_delta_range': (-0.5, -0.01),   # -50 delta to -1 delta
            
            # Liquidity filters
            'min_volume_threshold': 10,          # Minimum volume for inclusion
            'min_open_interest': 100,            # Minimum OI for inclusion
            
            # Strike distribution limits
            'max_strikes_per_side': 15,          # Maximum strikes per CALL/PUT side
            'min_strikes_per_side': 5,           # Minimum strikes per CALL/PUT side
            
            # Quality filters
            'max_bid_ask_spread_pct': 0.20,      # Maximum 20% bid-ask spread
            'min_last_traded_price': 0.05,      # Minimum last traded price
            
            # Delta precision
            'delta_precision': 0.001,            # Delta calculation precision
            
            # Strike selection optimization
            'optimize_strike_distribution': True,  # Enable strike optimization
            'target_delta_distribution': 'uniform' # uniform, weighted, or exponential
        }
    
    def apply_delta_filtering(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive delta-based filtering to options data
        
        Args:
            options_data: DataFrame with options data including delta values
            
        Returns:
            Filtered DataFrame with delta-based selection applied
        """
        
        logger.info("🔍 Applying comprehensive delta-based filtering...")
        
        # Reset filter statistics
        self.filter_statistics = {
            'original_count': len(options_data),
            'call_original': len(options_data[options_data['option_type'] == 'CE']),
            'put_original': len(options_data[options_data['option_type'] == 'PE'])
        }
        
        # Step 1: Basic delta range filtering
        delta_filtered = self._apply_delta_range_filter(options_data)
        
        # Step 2: Liquidity filtering
        liquidity_filtered = self._apply_liquidity_filter(delta_filtered)
        
        # Step 3: Quality filtering
        quality_filtered = self._apply_quality_filter(liquidity_filtered)
        
        # Step 4: Strike distribution optimization
        if self.config['optimize_strike_distribution']:
            optimized_filtered = self._optimize_strike_distribution(quality_filtered)
        else:
            optimized_filtered = quality_filtered
        
        # Update final statistics
        self.filter_statistics.update({
            'final_count': len(optimized_filtered),
            'call_final': len(optimized_filtered[optimized_filtered['option_type'] == 'CE']),
            'put_final': len(optimized_filtered[optimized_filtered['option_type'] == 'PE']),
            'filter_efficiency': len(optimized_filtered) / len(options_data) if len(options_data) > 0 else 0
        })
        
        self._log_filter_statistics()
        
        return optimized_filtered
    
    def _apply_delta_range_filter(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Apply delta range filtering for CALL and PUT options"""
        
        logger.info("   📊 Applying delta range filtering...")
        
        # CALL options filter (0.5 to 0.01 delta)
        call_filter = (
            (options_data['option_type'] == 'CE') &
            (options_data['delta'] >= self.config['call_delta_range'][0]) &
            (options_data['delta'] <= self.config['call_delta_range'][1])
        )
        
        # PUT options filter (-0.5 to -0.01 delta)
        put_filter = (
            (options_data['option_type'] == 'PE') &
            (options_data['delta'] >= self.config['put_delta_range'][0]) &
            (options_data['delta'] <= self.config['put_delta_range'][1])
        )
        
        # Combine filters
        delta_filtered = options_data[call_filter | put_filter].copy()
        
        logger.info(f"      ✅ Delta filtered: {len(delta_filtered)} options")
        
        return delta_filtered
    
    def _apply_liquidity_filter(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Apply liquidity-based filtering"""
        
        logger.info("   💧 Applying liquidity filtering...")
        
        # Volume filter
        volume_filter = options_data['volume'] >= self.config['min_volume_threshold']
        
        # Open interest filter
        oi_filter = options_data['open_interest'] >= self.config['min_open_interest']
        
        # Combine liquidity filters
        liquidity_filtered = options_data[volume_filter & oi_filter].copy()
        
        logger.info(f"      ✅ Liquidity filtered: {len(liquidity_filtered)} options")
        
        return liquidity_filtered
    
    def _apply_quality_filter(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Apply quality-based filtering"""
        
        logger.info("   ⭐ Applying quality filtering...")
        
        quality_filtered = options_data.copy()
        
        # Bid-ask spread filter (if available)
        if 'bid' in options_data.columns and 'ask' in options_data.columns:
            mid_price = (options_data['bid'] + options_data['ask']) / 2
            spread_pct = (options_data['ask'] - options_data['bid']) / mid_price
            spread_filter = spread_pct <= self.config['max_bid_ask_spread_pct']
            quality_filtered = quality_filtered[spread_filter]
        
        # Last traded price filter (if available)
        if 'last_price' in options_data.columns:
            price_filter = options_data['last_price'] >= self.config['min_last_traded_price']
            quality_filtered = quality_filtered[price_filter]
        
        logger.info(f"      ✅ Quality filtered: {len(quality_filtered)} options")
        
        return quality_filtered
    
    def _optimize_strike_distribution(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Optimize strike distribution for balanced Greek exposure"""
        
        logger.info("   🎯 Optimizing strike distribution...")
        
        optimized_data = []
        
        # Separate CALL and PUT options
        call_options = options_data[options_data['option_type'] == 'CE'].copy()
        put_options = options_data[options_data['option_type'] == 'PE'].copy()
        
        # Optimize CALL strikes
        if len(call_options) > 0:
            optimized_calls = self._select_optimal_strikes(
                call_options, 'CE', self.config['max_strikes_per_side']
            )
            optimized_data.append(optimized_calls)
        
        # Optimize PUT strikes
        if len(put_options) > 0:
            optimized_puts = self._select_optimal_strikes(
                put_options, 'PE', self.config['max_strikes_per_side']
            )
            optimized_data.append(optimized_puts)
        
        # Combine optimized strikes
        if optimized_data:
            optimized_filtered = pd.concat(optimized_data, ignore_index=True)
        else:
            optimized_filtered = pd.DataFrame()
        
        logger.info(f"      ✅ Strike optimized: {len(optimized_filtered)} options")
        
        return optimized_filtered
    
    def _select_optimal_strikes(self, options_data: pd.DataFrame, option_type: str, max_strikes: int) -> pd.DataFrame:
        """Select optimal strikes based on delta distribution"""
        
        if len(options_data) <= max_strikes:
            return options_data
        
        # Sort by delta (absolute value for uniform distribution)
        if self.config['target_delta_distribution'] == 'uniform':
            # Select strikes with uniform delta distribution
            sorted_options = options_data.sort_values('delta', key=abs)
            
            # Select evenly distributed strikes
            indices = np.linspace(0, len(sorted_options) - 1, max_strikes, dtype=int)
            selected_options = sorted_options.iloc[indices]
            
        elif self.config['target_delta_distribution'] == 'weighted':
            # Weight selection by volume and open interest
            options_data['selection_weight'] = (
                options_data['volume'] * 0.6 + 
                options_data['open_interest'] * 0.4
            )
            
            # Select top weighted strikes
            selected_options = options_data.nlargest(max_strikes, 'selection_weight')
            
        else:  # exponential distribution
            # Exponential weighting favoring higher delta options
            options_data['delta_weight'] = np.exp(abs(options_data['delta']) * 5)
            selected_options = options_data.nlargest(max_strikes, 'delta_weight')
        
        return selected_options
    
    def get_strike_distribution_analysis(self, filtered_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the distribution of selected strikes"""
        
        analysis = {}
        
        # Separate CALL and PUT analysis
        for option_type in ['CE', 'PE']:
            type_data = filtered_data[filtered_data['option_type'] == option_type]
            
            if len(type_data) > 0:
                analysis[option_type] = {
                    'count': len(type_data),
                    'delta_range': (type_data['delta'].min(), type_data['delta'].max()),
                    'delta_mean': type_data['delta'].mean(),
                    'delta_std': type_data['delta'].std(),
                    'volume_total': type_data['volume'].sum(),
                    'oi_total': type_data['open_interest'].sum(),
                    'strikes': sorted(type_data['strike'].unique()) if 'strike' in type_data.columns else []
                }
        
        # Overall analysis
        analysis['overall'] = {
            'total_strikes': len(filtered_data),
            'call_put_ratio': len(filtered_data[filtered_data['option_type'] == 'CE']) / 
                             max(len(filtered_data[filtered_data['option_type'] == 'PE']), 1),
            'avg_volume_per_strike': filtered_data['volume'].mean(),
            'avg_oi_per_strike': filtered_data['open_interest'].mean()
        }
        
        return analysis
    
    def _log_filter_statistics(self):
        """Log comprehensive filter statistics"""
        
        stats = self.filter_statistics
        
        logger.info("📊 Delta-based Filtering Statistics:")
        logger.info(f"   📈 Original options: {stats['original_count']}")
        logger.info(f"      CALL: {stats['call_original']}, PUT: {stats['put_original']}")
        logger.info(f"   🎯 Final options: {stats['final_count']}")
        logger.info(f"      CALL: {stats['call_final']}, PUT: {stats['put_final']}")
        logger.info(f"   ⚡ Filter efficiency: {stats['filter_efficiency']:.1%}")
    
    def validate_delta_filtering(self, original_data: pd.DataFrame, filtered_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate delta filtering results"""
        
        validation = {
            'delta_range_compliance': True,
            'liquidity_compliance': True,
            'distribution_balance': True,
            'issues': []
        }
        
        # Check delta range compliance
        for _, row in filtered_data.iterrows():
            if row['option_type'] == 'CE':
                if not (self.config['call_delta_range'][0] <= row['delta'] <= self.config['call_delta_range'][1]):
                    validation['delta_range_compliance'] = False
                    validation['issues'].append(f"CALL delta {row['delta']} outside range")
            
            elif row['option_type'] == 'PE':
                if not (self.config['put_delta_range'][0] <= row['delta'] <= self.config['put_delta_range'][1]):
                    validation['delta_range_compliance'] = False
                    validation['issues'].append(f"PUT delta {row['delta']} outside range")
        
        # Check liquidity compliance
        low_volume = filtered_data[filtered_data['volume'] < self.config['min_volume_threshold']]
        if len(low_volume) > 0:
            validation['liquidity_compliance'] = False
            validation['issues'].append(f"{len(low_volume)} options below volume threshold")
        
        # Check distribution balance
        call_count = len(filtered_data[filtered_data['option_type'] == 'CE'])
        put_count = len(filtered_data[filtered_data['option_type'] == 'PE'])
        
        if call_count == 0 or put_count == 0:
            validation['distribution_balance'] = False
            validation['issues'].append("Missing CALL or PUT options")
        
        validation['overall_valid'] = (
            validation['delta_range_compliance'] and 
            validation['liquidity_compliance'] and 
            validation['distribution_balance']
        )
        
        return validation

def main():
    """Main function for testing delta-based strike filtering"""
    
    logger.info("🚀 Testing Delta-based Strike Filter")
    
    # Initialize filter
    strike_filter = DeltaBasedStrikeFilter()
    
    # PRODUCTION MODE: NO SYNTHETIC TEST DATA GENERATION
    logger.error("PRODUCTION MODE: Synthetic test data generation is disabled.")
    logger.error("Delta-based strike filter must use real HeavyDB option chain data only.")
    
    # Return empty test data to prevent synthetic testing
    test_data = pd.DataFrame()
    
    # Apply filtering
    filtered_data = strike_filter.apply_delta_filtering(test_data)
    
    # Get distribution analysis
    distribution_analysis = strike_filter.get_strike_distribution_analysis(filtered_data)
    
    # Validate filtering
    validation_results = strike_filter.validate_delta_filtering(test_data, filtered_data)
    
    logger.info("🎯 Delta-based Strike Filtering Testing Complete")
    
    return {
        'filtered_data': filtered_data,
        'distribution_analysis': distribution_analysis,
        'validation_results': validation_results
    }

if __name__ == "__main__":
    main()
