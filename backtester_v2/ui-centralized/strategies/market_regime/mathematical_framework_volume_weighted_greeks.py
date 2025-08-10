#!/usr/bin/env python3
"""
Mathematical Framework for Volume-weighted Greek Calculations
Enhanced Triple Straddle Rolling Analysis Framework v2.0

Author: The Augster
Date: 2025-06-20
Version: 2.0.0 (Mathematical Framework)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VolumeWeightedGreekCalculator:
    """
    Mathematical framework for volume-weighted Greek calculations
    
    Implements:
    1. Delta-based strike selection (CALL: 0.5→0.01, PUT: -0.5→-0.01)
    2. Volume-weighted cumulative exposure calculation
    3. Portfolio-level Greek aggregation with ±0.001 mathematical accuracy
    4. 9:15 AM baseline establishment and real-time change tracking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize volume-weighted Greek calculator"""
        
        self.config = config or self._get_default_config()
        self.opening_baseline = {}
        self.mathematical_tolerance = 0.001
        
        logger.info("🧮 Volume-weighted Greek Calculator initialized")
        logger.info(f"✅ Mathematical tolerance: ±{self.mathematical_tolerance}")
        logger.info(f"✅ Delta ranges: CALL {self.config['call_delta_range']}, PUT {self.config['put_delta_range']}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for volume-weighted calculations"""
        
        return {
            # Delta-based strike selection
            'call_delta_range': (0.01, 0.5),    # 1 delta to 50 delta
            'put_delta_range': (-0.5, -0.01),   # -50 delta to -1 delta
            
            # Greek component weights
            'greek_weights': {
                'delta': 0.40,    # 40% weight - directional bias
                'gamma': 0.30,    # 30% weight - acceleration risk
                'theta': 0.20,    # 20% weight - time decay (critical for 0 DTE)
                'vega': 0.10      # 10% weight - volatility sensitivity
            },
            
            # Normalization factors for tanh
            'normalization_factors': {
                'delta': 100000,  # Normalize delta exposure
                'gamma': 50000,   # Normalize gamma exposure
                'theta': 10000,   # Normalize theta exposure
                'vega': 20000     # Normalize vega exposure
            },
            
            # Contract specifications
            'option_multiplier': 50,  # NIFTY lot size
            'min_volume_threshold': 10,  # Minimum volume for inclusion
            
            # Expiry weighting
            'expiry_weights': {
                0: 0.70,    # 0 DTE: 70% weight
                1: 0.20,    # 1 DTE: 20% weight
                2: 0.07,    # 2 DTE: 7% weight
                3: 0.03     # 3 DTE: 3% weight
            }
        }
    
    def apply_delta_based_strike_selection(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply delta-based strike selection for volume-weighted calculations
        
        Args:
            options_data: DataFrame with options data including delta values
            
        Returns:
            Filtered DataFrame with delta-based selection applied
        """
        
        logger.info("🎯 Applying delta-based strike selection...")
        
        # Filter CALL options (0.5 to 0.01 delta)
        call_filter = (
            (options_data['option_type'] == 'CE') &
            (options_data['delta'] >= self.config['call_delta_range'][0]) &
            (options_data['delta'] <= self.config['call_delta_range'][1]) &
            (options_data['volume'] >= self.config['min_volume_threshold'])
        )
        
        # Filter PUT options (-0.5 to -0.01 delta)
        put_filter = (
            (options_data['option_type'] == 'PE') &
            (options_data['delta'] >= self.config['put_delta_range'][0]) &
            (options_data['delta'] <= self.config['put_delta_range'][1]) &
            (options_data['volume'] >= self.config['min_volume_threshold'])
        )
        
        # Combine filters
        filtered_data = options_data[call_filter | put_filter].copy()
        
        logger.info(f"   ✅ Original options: {len(options_data)}")
        logger.info(f"   ✅ Delta-filtered options: {len(filtered_data)}")
        logger.info(f"   ✅ CALL options: {len(filtered_data[filtered_data['option_type'] == 'CE'])}")
        logger.info(f"   ✅ PUT options: {len(filtered_data[filtered_data['option_type'] == 'PE'])}")
        
        return filtered_data
    
    def calculate_volume_weights(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume weights for each option
        
        Args:
            options_data: DataFrame with volume data
            
        Returns:
            DataFrame with volume_weight column added
        """
        
        # Calculate volume weights
        max_volume = max(options_data['volume'].max(), 1)
        options_data['volume_weight'] = options_data['volume'] / max_volume
        
        logger.info(f"   ✅ Max volume: {max_volume:,}")
        logger.info(f"   ✅ Volume weights calculated for {len(options_data)} options")
        
        return options_data
    
    def calculate_portfolio_greek_exposure(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio-level Greek exposure using volume weighting
        
        Mathematical Formula:
        Portfolio_Greek_Exposure = Σ(i=1 to n) [Greek_i × OI_i × Volume_Weight_i × Option_Multiplier]
        
        Args:
            options_data: DataFrame with options data including Greeks, OI, and volume
            
        Returns:
            Dict with portfolio Greek exposures
        """
        
        logger.info("📊 Calculating portfolio Greek exposure...")
        
        # Apply delta-based filtering
        filtered_data = self.apply_delta_based_strike_selection(options_data)
        
        # Calculate volume weights
        weighted_data = self.calculate_volume_weights(filtered_data)
        
        # Apply expiry weighting if DTE column exists
        if 'dte' in weighted_data.columns:
            weighted_data = self._apply_expiry_weighting(weighted_data)
        
        # Calculate portfolio exposure for each Greek
        portfolio_exposure = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            if greek in weighted_data.columns:
                # Volume-weighted Greek exposure calculation
                greek_exposure = (
                    weighted_data[greek] * 
                    weighted_data['open_interest'] * 
                    weighted_data['volume_weight'] * 
                    self.config['option_multiplier']
                ).sum()
                
                portfolio_exposure[greek] = greek_exposure
                
                logger.info(f"   ✅ Portfolio {greek.upper()} exposure: {greek_exposure:,.0f}")
        
        return portfolio_exposure
    
    def _apply_expiry_weighting(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Apply expiry-based weighting (0 DTE: 70%, 1-3 DTE: 30%)"""
        
        options_data['expiry_weight'] = options_data['dte'].map(
            lambda dte: self.config['expiry_weights'].get(dte, 0.0)
        )
        
        # Apply expiry weighting to volume weight
        options_data['volume_weight'] *= options_data['expiry_weight']
        
        return options_data
    
    def establish_opening_baseline(self, opening_data: pd.DataFrame) -> Dict[str, float]:
        """
        Establish 9:15 AM opening baseline for Greek exposure
        
        Args:
            opening_data: DataFrame with 9:15 AM options data
            
        Returns:
            Dict with opening Greek exposure baseline
        """
        
        logger.info("📈 Establishing 9:15 AM opening baseline...")
        
        # Calculate opening portfolio exposure
        opening_exposure = self.calculate_portfolio_greek_exposure(opening_data)
        
        # Store as baseline
        self.opening_baseline = opening_exposure.copy()
        
        logger.info("   ✅ Opening baseline established:")
        for greek, exposure in opening_exposure.items():
            logger.info(f"      {greek.upper()}: {exposure:,.0f}")
        
        return opening_exposure
    
    def calculate_greek_changes_from_baseline(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Greek changes from 9:15 AM baseline
        
        Args:
            current_data: DataFrame with current options data
            
        Returns:
            Dict with Greek changes and normalized components
        """
        
        if not self.opening_baseline:
            raise ValueError("Opening baseline not established. Call establish_opening_baseline() first.")
        
        # Calculate current portfolio exposure
        current_exposure = self.calculate_portfolio_greek_exposure(current_data)
        
        # Calculate changes from baseline
        greek_changes = {}
        normalized_components = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            if greek in current_exposure and greek in self.opening_baseline:
                # Raw change
                change = current_exposure[greek] - self.opening_baseline[greek]
                greek_changes[f'{greek}_change'] = change
                
                # Normalized component using tanh
                normalization_factor = self.config['normalization_factors'][greek]
                normalized = np.tanh(change / normalization_factor)
                normalized_components[f'{greek}_component'] = normalized
        
        # Calculate weighted Greek sentiment score
        weights = self.config['greek_weights']
        greek_sentiment_score = sum(
            weights[greek] * normalized_components.get(f'{greek}_component', 0)
            for greek in weights.keys()
        )
        
        # Combine all results
        result = {
            # Opening baseline
            **{f'opening_{greek}': self.opening_baseline[greek] for greek in self.opening_baseline},
            
            # Current exposure
            **{f'current_{greek}': current_exposure[greek] for greek in current_exposure},
            
            # Changes from baseline
            **greek_changes,
            
            # Normalized components
            **normalized_components,
            
            # Final sentiment score
            'greek_sentiment_score': greek_sentiment_score
        }
        
        return result
    
    def validate_mathematical_accuracy(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate mathematical accuracy with ±0.001 tolerance
        
        Args:
            test_data: DataFrame with test options data
            
        Returns:
            Dict with validation results
        """
        
        logger.info("🧪 Validating mathematical accuracy...")
        
        validation_results = {}
        
        # Manual calculation for validation
        manual_exposure = self._manual_portfolio_calculation(test_data)
        
        # System calculation
        system_exposure = self.calculate_portfolio_greek_exposure(test_data)
        
        # Compare results
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            if greek in manual_exposure and greek in system_exposure:
                manual_value = manual_exposure[greek]
                system_value = system_exposure[greek]
                difference = abs(manual_value - system_value)
                
                validation_results[greek] = {
                    'manual_calculation': manual_value,
                    'system_calculation': system_value,
                    'difference': difference,
                    'within_tolerance': difference <= self.mathematical_tolerance,
                    'accuracy_percentage': (1 - difference/abs(manual_value)) * 100 if manual_value != 0 else 100
                }
        
        # Overall validation status
        all_accurate = all(
            result['within_tolerance'] for result in validation_results.values()
        )
        
        validation_summary = {
            'all_calculations_accurate': all_accurate,
            'tolerance': self.mathematical_tolerance,
            'detailed_results': validation_results
        }
        
        logger.info(f"   ✅ Mathematical accuracy validation: {'PASS' if all_accurate else 'FAIL'}")
        
        return validation_summary
    
    def _manual_portfolio_calculation(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Manual calculation for validation purposes"""
        
        # Apply same filtering as system
        filtered_data = self.apply_delta_based_strike_selection(options_data)
        weighted_data = self.calculate_volume_weights(filtered_data)
        
        # Manual calculation
        manual_exposure = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            if greek in weighted_data.columns:
                total_exposure = 0
                
                for _, row in weighted_data.iterrows():
                    exposure = (
                        row[greek] * 
                        row['open_interest'] * 
                        row['volume_weight'] * 
                        self.config['option_multiplier']
                    )
                    total_exposure += exposure
                
                manual_exposure[greek] = total_exposure
        
        return manual_exposure

def main():
    """Main function for testing the mathematical framework"""
    
    logger.info("🚀 Testing Volume-weighted Greek Calculator")
    
    # Initialize calculator
    calculator = VolumeWeightedGreekCalculator()
    
    # Generate sample test data
    test_data = pd.DataFrame({
        'option_type': ['CE', 'CE', 'PE', 'PE'] * 5,
        'delta': [0.45, 0.25, -0.35, -0.15] * 5,
        'gamma': [0.005, 0.008, 0.005, 0.008] * 5,
        'theta': [-1.2, -0.8, -1.1, -0.7] * 5,
        'vega': [0.9, 1.2, 0.8, 1.1] * 5,
        'open_interest': [10000, 15000, 12000, 8000] * 5,
        'volume': [1500, 2000, 1200, 800] * 5,
        'dte': [0, 0, 0, 0] * 5
    })
    
    # Test portfolio exposure calculation
    portfolio_exposure = calculator.calculate_portfolio_greek_exposure(test_data)
    
    # Test mathematical accuracy validation
    validation_results = calculator.validate_mathematical_accuracy(test_data)
    
    logger.info("🎯 Mathematical Framework Testing Complete")
    
    return {
        'portfolio_exposure': portfolio_exposure,
        'validation_results': validation_results
    }

if __name__ == "__main__":
    main()
