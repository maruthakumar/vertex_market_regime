"""
Volume Profile Analyzer
======================

Analyzes volume distribution across price levels to identify market structure
and potential support/resistance levels for regime detection.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VolumeProfileData:
    """Container for volume profile analysis results"""
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float
    hvn_levels: List[float]  # High Volume Nodes
    lvn_levels: List[float]  # Low Volume Nodes
    volume_concentration: float  # 0-1 score
    profile_skew: float  # Distribution skew
    regime_signal: float  # -1 to 1


class VolumeProfileAnalyzer:
    """
    Analyzes volume distribution patterns for market regime detection
    
    This is one of the 9 active components in the enhanced market regime system.
    Base weight: 0.08 (8% of total regime signal)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the volume profile analyzer"""
        self.config = config or {}
        
        # Configuration parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.price_bins = self.config.get('price_bins', 50)
        self.value_area_pct = self.config.get('value_area_pct', 0.70)  # 70% of volume
        self.hvn_threshold = self.config.get('hvn_threshold', 1.5)  # 1.5x average
        self.lvn_threshold = self.config.get('lvn_threshold', 0.5)  # 0.5x average
        
        # Cache for performance
        self._cache = {}
        self._last_calculation = None
        
        logger.info(f"VolumeProfileAnalyzer initialized with {self.price_bins} price bins")
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main analysis method for volume profile
        
        Args:
            market_data: DataFrame with columns:
                - datetime_
                - close or underlying_close
                - volume
                - option columns if available
                
        Returns:
            Dict with volume profile analysis results
        """
        try:
            # Extract price and volume data
            if 'underlying_close' in market_data.columns:
                prices = market_data['underlying_close'].values
            else:
                prices = market_data['close'].values
            
            volumes = market_data['volume'].values
            
            # Build volume profile
            profile_data = self._build_volume_profile(prices, volumes)
            
            # Calculate regime signal
            regime_signal = self._calculate_regime_signal(profile_data, prices[-1])
            
            # Prepare results
            results = {
                'volume_profile_score': regime_signal,
                'poc_price': profile_data.poc_price,
                'value_area_high': profile_data.value_area_high,
                'value_area_low': profile_data.value_area_low,
                'volume_concentration': profile_data.volume_concentration,
                'profile_skew': profile_data.profile_skew,
                'hvn_count': len(profile_data.hvn_levels),
                'lvn_count': len(profile_data.lvn_levels),
                'price_position': self._calculate_price_position(
                    prices[-1], profile_data
                ),
                'volume_trend': self._calculate_volume_trend(volumes),
                'timestamp': datetime.now()
            }
            
            # Cache results
            self._cache['last_results'] = results
            self._last_calculation = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {e}")
            return self._get_default_results()
    
    def _build_volume_profile(self, 
                            prices: np.ndarray, 
                            volumes: np.ndarray) -> VolumeProfileData:
        """
        Build volume profile from price and volume data
        
        Args:
            prices: Array of prices
            volumes: Array of volumes
            
        Returns:
            VolumeProfileData object
        """
        # Create price bins
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min
        
        # Add small buffer to ensure all prices fit
        price_min -= price_range * 0.01
        price_max += price_range * 0.01
        
        bins = np.linspace(price_min, price_max, self.price_bins + 1)
        
        # Accumulate volume in each price bin
        volume_profile = np.zeros(self.price_bins)
        
        for i in range(len(prices)):
            bin_idx = np.digitize(prices[i], bins) - 1
            if 0 <= bin_idx < self.price_bins:
                volume_profile[bin_idx] += volumes[i]
        
        # Calculate bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Find Point of Control (POC)
        poc_idx = np.argmax(volume_profile)
        poc_price = bin_centers[poc_idx]
        
        # Calculate Value Area
        value_area = self._calculate_value_area(
            bin_centers, volume_profile, self.value_area_pct
        )
        
        # Identify HVN and LVN
        avg_volume = np.mean(volume_profile[volume_profile > 0])
        hvn_mask = volume_profile > (avg_volume * self.hvn_threshold)
        lvn_mask = (volume_profile > 0) & (volume_profile < (avg_volume * self.lvn_threshold))
        
        hvn_levels = bin_centers[hvn_mask].tolist()
        lvn_levels = bin_centers[lvn_mask].tolist()
        
        # Calculate concentration (how much volume is in top 20% of bins)
        sorted_volumes = np.sort(volume_profile)[::-1]
        top_20_pct_bins = int(self.price_bins * 0.2)
        concentration = np.sum(sorted_volumes[:top_20_pct_bins]) / np.sum(volume_profile)
        
        # Calculate skew
        total_volume = np.sum(volume_profile)
        if total_volume > 0:
            weighted_prices = np.sum(bin_centers * volume_profile) / total_volume
            weighted_variance = np.sum(((bin_centers - weighted_prices) ** 2) * volume_profile) / total_volume
            weighted_skew = np.sum(((bin_centers - weighted_prices) ** 3) * volume_profile) / (total_volume * weighted_variance ** 1.5)
            profile_skew = np.clip(weighted_skew, -2, 2)
        else:
            profile_skew = 0.0
        
        return VolumeProfileData(
            poc_price=poc_price,
            value_area_high=value_area[1],
            value_area_low=value_area[0],
            hvn_levels=hvn_levels,
            lvn_levels=lvn_levels,
            volume_concentration=concentration,
            profile_skew=profile_skew,
            regime_signal=0.0  # Will be calculated separately
        )
    
    def _calculate_value_area(self, 
                            prices: np.ndarray,
                            volumes: np.ndarray,
                            target_pct: float) -> Tuple[float, float]:
        """
        Calculate value area containing target percentage of volume
        
        Args:
            prices: Bin center prices
            volumes: Volume at each price level
            target_pct: Target percentage (e.g., 0.70 for 70%)
            
        Returns:
            Tuple of (value_area_low, value_area_high)
        """
        total_volume = np.sum(volumes)
        target_volume = total_volume * target_pct
        
        # Start from POC and expand outward
        poc_idx = np.argmax(volumes)
        accumulated_volume = volumes[poc_idx]
        
        low_idx = poc_idx
        high_idx = poc_idx
        
        while accumulated_volume < target_volume:
            # Check which side to expand
            expand_low = low_idx > 0
            expand_high = high_idx < len(volumes) - 1
            
            if expand_low and expand_high:
                # Expand side with higher volume
                if volumes[low_idx - 1] > volumes[high_idx + 1]:
                    low_idx -= 1
                    accumulated_volume += volumes[low_idx]
                else:
                    high_idx += 1
                    accumulated_volume += volumes[high_idx]
            elif expand_low:
                low_idx -= 1
                accumulated_volume += volumes[low_idx]
            elif expand_high:
                high_idx += 1
                accumulated_volume += volumes[high_idx]
            else:
                break
        
        return (prices[low_idx], prices[high_idx])
    
    def _calculate_regime_signal(self, 
                               profile_data: VolumeProfileData,
                               current_price: float) -> float:
        """
        Calculate regime signal from volume profile data
        
        Args:
            profile_data: Volume profile analysis results
            current_price: Current market price
            
        Returns:
            float: Regime signal between -1 and 1
        """
        signal_components = []
        
        # 1. Price position relative to POC
        poc_distance = (current_price - profile_data.poc_price) / profile_data.poc_price
        poc_signal = np.tanh(poc_distance * 10)  # Normalize to -1 to 1
        signal_components.append(('poc_position', poc_signal, 0.3))
        
        # 2. Price position within value area
        if profile_data.value_area_low <= current_price <= profile_data.value_area_high:
            # Inside value area - neutral/consolidation
            va_signal = 0.0
        elif current_price > profile_data.value_area_high:
            # Above value area - bullish
            va_distance = (current_price - profile_data.value_area_high) / profile_data.value_area_high
            va_signal = min(va_distance * 5, 1.0)
        else:
            # Below value area - bearish
            va_distance = (profile_data.value_area_low - current_price) / profile_data.value_area_low
            va_signal = max(-va_distance * 5, -1.0)
        
        signal_components.append(('value_area', va_signal, 0.3))
        
        # 3. Volume concentration signal
        # High concentration = strong trend, low concentration = ranging
        concentration_signal = (profile_data.volume_concentration - 0.5) * 2
        signal_components.append(('concentration', concentration_signal, 0.2))
        
        # 4. Profile skew signal
        # Positive skew = bullish, negative skew = bearish
        skew_signal = np.clip(profile_data.profile_skew, -1, 1)
        signal_components.append(('skew', skew_signal, 0.2))
        
        # Combine signals with weights
        total_signal = 0.0
        for name, signal, weight in signal_components:
            total_signal += signal * weight
            logger.debug(f"Volume profile {name}: {signal:.3f} (weight: {weight})")
        
        return np.clip(total_signal, -1.0, 1.0)
    
    def _calculate_price_position(self, 
                                current_price: float,
                                profile_data: VolumeProfileData) -> str:
        """Determine price position relative to volume profile"""
        if current_price > profile_data.value_area_high:
            return "above_value_area"
        elif current_price < profile_data.value_area_low:
            return "below_value_area"
        elif abs(current_price - profile_data.poc_price) / profile_data.poc_price < 0.01:
            return "at_poc"
        else:
            return "in_value_area"
    
    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Calculate volume trend (increasing/decreasing)"""
        if len(volumes) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(volumes))
        coeffs = np.polyfit(x, volumes, 1)
        
        # Normalize slope
        avg_volume = np.mean(volumes)
        if avg_volume > 0:
            normalized_slope = coeffs[0] / avg_volume
            return np.clip(normalized_slope * 10, -1, 1)
        
        return 0.0
    
    def get_support_resistance_levels(self, 
                                    market_data: pd.DataFrame,
                                    num_levels: int = 5) -> Dict[str, List[float]]:
        """
        Get support and resistance levels from volume profile
        
        Args:
            market_data: Market data DataFrame
            num_levels: Number of levels to return
            
        Returns:
            Dict with 'support' and 'resistance' lists
        """
        try:
            results = self.analyze(market_data)
            current_price = market_data['underlying_close'].iloc[-1]
            
            # HVN levels act as support/resistance
            hvn_levels = self._cache.get('hvn_levels', [])
            
            support_levels = [lvl for lvl in hvn_levels if lvl < current_price]
            resistance_levels = [lvl for lvl in hvn_levels if lvl > current_price]
            
            # Sort by distance from current price
            support_levels.sort(reverse=True)  # Closest first
            resistance_levels.sort()  # Closest first
            
            return {
                'support': support_levels[:num_levels],
                'resistance': resistance_levels[:num_levels]
            }
            
        except Exception as e:
            logger.error(f"Error getting support/resistance levels: {e}")
            return {'support': [], 'resistance': []}
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results when analysis fails"""
        return {
            'volume_profile_score': 0.0,
            'poc_price': 0.0,
            'value_area_high': 0.0,
            'value_area_low': 0.0,
            'volume_concentration': 0.5,
            'profile_skew': 0.0,
            'hvn_count': 0,
            'lvn_count': 0,
            'price_position': 'unknown',
            'volume_trend': 0.0,
            'timestamp': datetime.now()
        }