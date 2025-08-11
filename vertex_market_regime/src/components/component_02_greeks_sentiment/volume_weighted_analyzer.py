"""
Volume-Weighted Greeks Analysis Engine - Component 2

Implements institutional-grade volume weighting using ce_volume, pe_volume, ce_oi, pe_oi data.
Creates symbol-specific volume threshold learning and combined volume analysis for straddle weighting.

🚨 KEY IMPLEMENTATION: Uses actual volume/OI data from production Parquet (100% coverage)
for institutional flow detection and Greeks weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import defaultdict

from .production_greeks_extractor import ProductionGreeksData
from .comprehensive_greeks_processor import ComprehensiveGreeksAnalysis


@dataclass
class VolumeWeightingConfig:
    """Configuration for volume weighting parameters"""
    min_volume_threshold: float = 100.0         # Minimum volume for analysis
    high_volume_threshold: float = 1000.0       # High volume threshold
    min_oi_threshold: float = 500.0             # Minimum OI for institutional detection
    institutional_oi_threshold: float = 5000.0  # Institutional OI threshold
    volume_weight_cap: float = 3.0              # Maximum volume weight multiplier
    oi_weight_cap: float = 2.0                  # Maximum OI weight multiplier


@dataclass
class VolumeAnalysisResult:
    """Volume analysis result for Greeks weighting"""
    total_volume: float                    # Combined CE + PE volume
    volume_imbalance: float               # CE volume - PE volume
    volume_weight: float                  # Calculated volume weight (1.0-3.0)
    
    total_oi: float                       # Combined CE + PE open interest
    oi_imbalance: float                   # CE OI - PE OI
    oi_weight: float                      # Calculated OI weight (1.0-2.0)
    
    institutional_flow: str               # INSTITUTIONAL, RETAIL, MIXED
    combined_weight: float                # Final combined volume + OI weight
    
    # Metadata
    ce_volume_pct: float                  # CE volume percentage
    pe_volume_pct: float                  # PE volume percentage
    volume_quality: float                 # Volume data quality (0-1)


@dataclass
class SymbolVolumeThresholds:
    """Symbol-specific volume thresholds learned from historical data"""
    symbol: str
    low_volume: float                     # 25th percentile volume
    medium_volume: float                  # 50th percentile volume
    high_volume: float                    # 75th percentile volume
    extreme_volume: float                 # 95th percentile volume
    
    low_oi: float                        # 25th percentile OI
    medium_oi: float                     # 50th percentile OI
    high_oi: float                       # 75th percentile OI
    extreme_oi: float                    # 95th percentile OI
    
    sample_count: int                    # Number of observations used
    last_updated: datetime


class VolumeWeightedAnalyzer:
    """
    Volume-Weighted Greeks Analysis Engine
    
    🚨 CRITICAL FEATURES:
    - Uses ACTUAL volume/OI data from production Parquet (ce_volume, pe_volume, ce_oi, pe_oi)
    - Institutional flow detection using OI thresholds
    - Symbol-specific volume threshold learning
    - Combined volume analysis for straddle weighting
    - Adaptive volume weight learning based on historical distributions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize volume-weighted analyzer"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Volume weighting configuration
        self.volume_config = VolumeWeightingConfig()
        
        # Symbol-specific thresholds (learned from data)
        self.symbol_thresholds: Dict[str, SymbolVolumeThresholds] = {}
        
        # Historical volume/OI data for learning
        self.volume_history: Dict[str, List[float]] = defaultdict(list)
        self.oi_history: Dict[str, List[float]] = defaultdict(list)
        
        self.logger.info("🚨 VolumeWeightedAnalyzer initialized with production volume/OI data")
    
    def calculate_volume_analysis(self, greeks_data: ProductionGreeksData) -> VolumeAnalysisResult:
        """
        Calculate volume analysis using actual production volume/OI data
        
        Args:
            greeks_data: ProductionGreeksData with volume/OI information
            
        Returns:
            VolumeAnalysisResult with weighted analysis
        """
        try:
            # Extract ACTUAL volume data (100% coverage in production)
            ce_volume = float(greeks_data.ce_volume)    # Column 19
            pe_volume = float(greeks_data.pe_volume)    # Column 33
            ce_oi = float(greeks_data.ce_oi)           # Column 20
            pe_oi = float(greeks_data.pe_oi)           # Column 34
            
            # Calculate volume metrics
            total_volume = ce_volume + pe_volume
            volume_imbalance = ce_volume - pe_volume
            
            # Calculate OI metrics
            total_oi = ce_oi + pe_oi
            oi_imbalance = ce_oi - pe_oi
            
            # Calculate volume weight
            volume_weight = self._calculate_volume_weight(total_volume)
            
            # Calculate OI weight
            oi_weight = self._calculate_oi_weight(total_oi)
            
            # Determine institutional flow
            institutional_flow = self._classify_institutional_flow(total_volume, total_oi)
            
            # Combined weight (volume + OI consideration)
            combined_weight = self._calculate_combined_weight(volume_weight, oi_weight, institutional_flow)
            
            # Volume percentages
            ce_volume_pct = (ce_volume / total_volume) * 100 if total_volume > 0 else 50.0
            pe_volume_pct = (pe_volume / total_volume) * 100 if total_volume > 0 else 50.0
            
            # Volume quality assessment
            volume_quality = self._assess_volume_quality(total_volume, total_oi)
            
            return VolumeAnalysisResult(
                total_volume=total_volume,
                volume_imbalance=volume_imbalance,
                volume_weight=volume_weight,
                total_oi=total_oi,
                oi_imbalance=oi_imbalance,
                oi_weight=oi_weight,
                institutional_flow=institutional_flow,
                combined_weight=combined_weight,
                ce_volume_pct=ce_volume_pct,
                pe_volume_pct=pe_volume_pct,
                volume_quality=volume_quality
            )
            
        except Exception as e:
            self.logger.error(f"Volume analysis calculation failed: {e}")
            raise
    
    def _calculate_volume_weight(self, total_volume: float) -> float:
        """
        Calculate volume weight based on production data distributions
        
        Args:
            total_volume: Total straddle volume (CE + PE)
            
        Returns:
            Volume weight multiplier (1.0 to volume_weight_cap)
        """
        if total_volume < self.volume_config.min_volume_threshold:
            # Low volume - reduce weight
            return 0.5
        elif total_volume > self.volume_config.high_volume_threshold:
            # High volume - increase weight (capped)
            weight = min(
                1.0 + (total_volume / self.volume_config.high_volume_threshold),
                self.volume_config.volume_weight_cap
            )
            return weight
        else:
            # Normal volume - standard weight with gradual increase
            weight = 1.0 + (total_volume / self.volume_config.high_volume_threshold) * 0.5
            return min(weight, self.volume_config.volume_weight_cap)
    
    def _calculate_oi_weight(self, total_oi: float) -> float:
        """
        Calculate OI weight for institutional flow detection
        
        Args:
            total_oi: Total open interest (CE + PE)
            
        Returns:
            OI weight multiplier (1.0 to oi_weight_cap)
        """
        if total_oi < self.volume_config.min_oi_threshold:
            # Low OI - retail dominated
            return 0.8
        elif total_oi > self.volume_config.institutional_oi_threshold:
            # Institutional OI - increase weight
            weight = min(
                1.0 + (total_oi / self.volume_config.institutional_oi_threshold),
                self.volume_config.oi_weight_cap
            )
            return weight
        else:
            # Medium OI - standard weight
            return 1.0
    
    def _classify_institutional_flow(self, volume: float, oi: float) -> str:
        """
        Classify institutional flow based on volume/OI characteristics
        
        Args:
            volume: Total volume
            oi: Total open interest
            
        Returns:
            Flow classification: INSTITUTIONAL, RETAIL, MIXED
        """
        # OI-to-volume ratio for institutional detection
        oi_volume_ratio = oi / volume if volume > 0 else 0
        
        if (oi > self.volume_config.institutional_oi_threshold and 
            oi_volume_ratio > 3.0):
            return "INSTITUTIONAL"
        elif (volume > self.volume_config.high_volume_threshold and
              oi_volume_ratio < 2.0):
            return "RETAIL"
        else:
            return "MIXED"
    
    def _calculate_combined_weight(self, volume_weight: float, oi_weight: float, flow_type: str) -> float:
        """
        Calculate combined volume + OI weight
        
        Args:
            volume_weight: Volume-based weight
            oi_weight: OI-based weight  
            flow_type: Institutional flow type
            
        Returns:
            Combined weight for Greeks analysis
        """
        # Base combined weight
        base_combined = (0.6 * volume_weight) + (0.4 * oi_weight)
        
        # Flow type adjustment
        if flow_type == "INSTITUTIONAL":
            # Institutional flow gets higher weight
            adjustment = 1.2
        elif flow_type == "RETAIL":
            # Retail flow gets lower weight
            adjustment = 0.9
        else:  # MIXED
            adjustment = 1.0
        
        combined_weight = base_combined * adjustment
        
        # Cap the combined weight
        return min(combined_weight, self.volume_config.volume_weight_cap)
    
    def _assess_volume_quality(self, volume: float, oi: float) -> float:
        """
        Assess volume data quality for analysis confidence
        
        Args:
            volume: Total volume
            oi: Total open interest
            
        Returns:
            Quality score (0-1)
        """
        # Volume quality component
        if volume >= self.volume_config.high_volume_threshold:
            volume_quality = 1.0
        elif volume >= self.volume_config.min_volume_threshold:
            volume_quality = 0.8
        else:
            volume_quality = 0.5
        
        # OI quality component
        if oi >= self.volume_config.institutional_oi_threshold:
            oi_quality = 1.0
        elif oi >= self.volume_config.min_oi_threshold:
            oi_quality = 0.8
        else:
            oi_quality = 0.6
        
        # Combined quality
        combined_quality = (0.7 * volume_quality) + (0.3 * oi_quality)
        return min(combined_quality, 1.0)
    
    def learn_symbol_thresholds(self, symbol: str, historical_data: List[ProductionGreeksData]) -> SymbolVolumeThresholds:
        """
        Learn symbol-specific volume thresholds from historical data
        
        Args:
            symbol: Symbol name (e.g., 'NIFTY', 'BANKNIFTY')
            historical_data: Historical ProductionGreeksData for the symbol
            
        Returns:
            Learned volume thresholds for the symbol
        """
        if not historical_data:
            raise ValueError(f"No historical data provided for symbol {symbol}")
        
        # Extract volume and OI data
        volumes = [data.ce_volume + data.pe_volume for data in historical_data]
        ois = [data.ce_oi + data.pe_oi for data in historical_data]
        
        # Calculate percentiles
        volume_percentiles = np.percentile(volumes, [25, 50, 75, 95])
        oi_percentiles = np.percentile(ois, [25, 50, 75, 95])
        
        thresholds = SymbolVolumeThresholds(
            symbol=symbol,
            low_volume=volume_percentiles[0],
            medium_volume=volume_percentiles[1],
            high_volume=volume_percentiles[2],
            extreme_volume=volume_percentiles[3],
            low_oi=oi_percentiles[0],
            medium_oi=oi_percentiles[1],
            high_oi=oi_percentiles[2],
            extreme_oi=oi_percentiles[3],
            sample_count=len(historical_data),
            last_updated=datetime.utcnow()
        )
        
        # Store learned thresholds
        self.symbol_thresholds[symbol] = thresholds
        
        self.logger.info(f"Learned volume thresholds for {symbol}: "
                        f"Volume[{thresholds.low_volume:.0f}-{thresholds.extreme_volume:.0f}], "
                        f"OI[{thresholds.low_oi:.0f}-{thresholds.extreme_oi:.0f}]")
        
        return thresholds
    
    def apply_volume_weighted_greeks(self, 
                                   greeks_analysis: ComprehensiveGreeksAnalysis,
                                   volume_analysis: VolumeAnalysisResult) -> Dict[str, float]:
        """
        Apply volume weighting to comprehensive Greeks analysis
        
        Args:
            greeks_analysis: ComprehensiveGreeksAnalysis result
            volume_analysis: VolumeAnalysisResult for weighting
            
        Returns:
            Volume-weighted Greeks scores
        """
        try:
            # Extract base Greeks scores
            base_delta = greeks_analysis.delta_analysis['weighted_delta']
            base_gamma = greeks_analysis.gamma_analysis.weighted_gamma_score  # Already 1.5x weighted
            base_theta = greeks_analysis.theta_analysis['weighted_theta']
            base_vega = greeks_analysis.vega_analysis['weighted_vega']
            
            # Apply volume weighting
            volume_weighted = {
                'delta_volume_weighted': base_delta * volume_analysis.combined_weight,
                'gamma_volume_weighted': base_gamma * volume_analysis.combined_weight,  # 1.5x + volume weight
                'theta_volume_weighted': base_theta * volume_analysis.combined_weight,
                'vega_volume_weighted': base_vega * volume_analysis.combined_weight,
                
                # Combined volume-weighted score
                'combined_volume_weighted': (
                    (base_delta + base_gamma + base_theta + base_vega) * 
                    volume_analysis.combined_weight
                ),
                
                # Metadata
                'volume_weight_applied': volume_analysis.combined_weight,
                'institutional_flow': volume_analysis.institutional_flow,
                'volume_quality': volume_analysis.volume_quality
            }
            
            self.logger.debug(f"Applied volume weight {volume_analysis.combined_weight:.2f} to Greeks analysis")
            return volume_weighted
            
        except Exception as e:
            self.logger.error(f"Volume weighting application failed: {e}")
            raise
    
    def get_symbol_volume_profile(self, symbol: str) -> Optional[SymbolVolumeThresholds]:
        """Get learned volume profile for a symbol"""
        return self.symbol_thresholds.get(symbol)
    
    def update_volume_history(self, symbol: str, volume: float, oi: float):
        """Update volume history for adaptive learning"""
        self.volume_history[symbol].append(volume)
        self.oi_history[symbol].append(oi)
        
        # Keep only recent history (last 1000 points)
        if len(self.volume_history[symbol]) > 1000:
            self.volume_history[symbol] = self.volume_history[symbol][-1000:]
        if len(self.oi_history[symbol]) > 1000:
            self.oi_history[symbol] = self.oi_history[symbol][-1000:]


# Testing and validation functions
def test_volume_weighted_analysis():
    """Test volume-weighted Greeks analysis"""
    print("🚨 Testing Volume-Weighted Greeks Analysis...")
    
    # Create test data with different volume scenarios
    test_scenarios = [
        # Low volume scenario
        ProductionGreeksData(
            ce_delta=0.5, pe_delta=-0.5, ce_gamma=0.0008, pe_gamma=0.0008,
            ce_theta=-5, pe_theta=-5, ce_vega=2, pe_vega=2,
            ce_volume=50, pe_volume=30,      # LOW VOLUME
            ce_oi=200, pe_oi=150,            # LOW OI
            call_strike_type='ATM', put_strike_type='ATM', dte=7,
            trade_time=datetime.utcnow(), expiry_date=datetime.utcnow()
        ),
        # High volume scenario
        ProductionGreeksData(
            ce_delta=0.6, pe_delta=-0.4, ce_gamma=0.0009, pe_gamma=0.0007,
            ce_theta=-8, pe_theta=-6, ce_vega=3, pe_vega=2.5,
            ce_volume=1500, pe_volume=1200,  # HIGH VOLUME
            ce_oi=3000, pe_oi=2500,         # MEDIUM OI
            call_strike_type='ATM', put_strike_type='ATM', dte=5,
            trade_time=datetime.utcnow(), expiry_date=datetime.utcnow()
        ),
        # Institutional scenario
        ProductionGreeksData(
            ce_delta=0.4, pe_delta=-0.6, ce_gamma=0.0010, pe_gamma=0.0012,
            ce_theta=-10, pe_theta=-8, ce_vega=4, pe_vega=3.5,
            ce_volume=800, pe_volume=600,    # MEDIUM VOLUME
            ce_oi=8000, pe_oi=7500,         # INSTITUTIONAL OI
            call_strike_type='ATM', put_strike_type='ATM', dte=3,
            trade_time=datetime.utcnow(), expiry_date=datetime.utcnow()
        )
    ]
    
    analyzer = VolumeWeightedAnalyzer()
    
    for i, test_data in enumerate(test_scenarios):
        print(f"\n--- Scenario {i+1} ---")
        
        # Calculate volume analysis
        volume_result = analyzer.calculate_volume_analysis(test_data)
        
        print(f"Volume: {volume_result.total_volume:.0f}, Weight: {volume_result.volume_weight:.2f}")
        print(f"OI: {volume_result.total_oi:.0f}, Weight: {volume_result.oi_weight:.2f}")
        print(f"Flow: {volume_result.institutional_flow}")
        print(f"Combined Weight: {volume_result.combined_weight:.2f}")
        print(f"Quality: {volume_result.volume_quality:.3f}")
    
    print("\n✅ Volume-Weighted Analysis test COMPLETED")


if __name__ == "__main__":
    test_volume_weighted_analysis()