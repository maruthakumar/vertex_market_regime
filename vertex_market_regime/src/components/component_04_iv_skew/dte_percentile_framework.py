"""
DTE-Adaptive IV Percentile Analysis Framework - Component 4 Enhancement

Advanced DTE-specific percentile analysis with both bucket-level (Near/Medium/Far)
and individual DTE-level tracking (dte=0, dte=1...dte=58), cross-DTE percentile
comparison, expiry-specific percentile handling, and dynamic percentile thresholds
for sophisticated market regime determination.

This module provides institutional-grade DTE-adaptive percentile analysis
with maximum granularity and precision for superior IV analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
from enum import Enum

from .iv_percentile_analyzer import IVPercentileData, IVPercentileResult
from .historical_percentile_database import HistoricalPercentileDatabase, PercentileDistribution


class DTEBucket(Enum):
    """DTE bucket classification"""
    NEAR = "near"      # 0-7 days
    MEDIUM = "medium"  # 8-30 days  
    FAR = "far"        # 31+ days


@dataclass
class DTEPercentileMetrics:
    """DTE-specific percentile metrics"""
    
    # Core DTE information
    dte: int
    dte_bucket: DTEBucket
    
    # Individual DTE percentile metrics
    dte_iv_percentile: float
    dte_historical_rank: float
    dte_confidence_score: float
    
    # Cross-DTE comparison
    relative_to_near_dte: Optional[float] = None
    relative_to_medium_dte: Optional[float] = None  
    relative_to_far_dte: Optional[float] = None
    
    # Dynamic thresholds
    dynamic_low_threshold: float = 25.0
    dynamic_high_threshold: float = 75.0
    regime_classification: str = "normal"
    
    # ATM-specific metrics
    atm_dte_percentile: float = 50.0
    atm_relative_strength: float = 0.0
    
    # Quality metrics
    data_sufficiency: float = 0.0
    calculation_confidence: float = 0.0
    
    def get_regime_classification(self) -> str:
        """Get regime classification based on percentile and thresholds"""
        
        if self.dte_iv_percentile >= 95:
            return "extremely_high"
        elif self.dte_iv_percentile >= self.dynamic_high_threshold:
            return "high" 
        elif self.dte_iv_percentile >= 60:
            return "above_normal"
        elif self.dte_iv_percentile >= 40:
            return "normal"
        elif self.dte_iv_percentile >= self.dynamic_low_threshold:
            return "below_normal"
        elif self.dte_iv_percentile >= 5:
            return "low"
        else:
            return "extremely_low"


@dataclass
class CrossDTEAnalysisResult:
    """Cross-DTE percentile comparison analysis"""
    
    # Term structure analysis
    term_structure_slope: float
    term_structure_curvature: float
    term_structure_level: float
    
    # Percentile differentials
    near_medium_differential: float
    medium_far_differential: float
    near_far_differential: float
    
    # Regime transition probabilities
    regime_transition_probability: float
    volatility_clustering_score: float
    
    # Cross-DTE regime consistency
    regime_consistency_score: float
    dominant_regime: str
    
    # Statistical significance
    statistical_significance: float
    sample_size_adequacy: float
    
    def get_term_structure_signal(self) -> str:
        """Get term structure signal classification"""
        
        if self.term_structure_slope > 2.0:
            return "steep_contango"
        elif self.term_structure_slope > 0.5:
            return "contango"
        elif self.term_structure_slope > -0.5:
            return "flat"
        elif self.term_structure_slope > -2.0:
            return "backwardation"
        else:
            return "steep_backwardation"


@dataclass 
class ExpirySpecificAnalysis:
    """Expiry-specific percentile analysis"""
    
    # Expiry information
    expiry_date: datetime
    days_to_expiry: int
    expiry_bucket: str
    
    # Expiry-specific percentiles
    expiry_iv_percentile: float
    expiry_historical_rank: int
    expiry_percentile_band: str
    
    # ATM analysis for this expiry
    expiry_atm_percentile: float
    atm_moneyness_effect: float
    
    # Volatility clustering for this expiry
    expiry_vol_clustering: float
    clustering_persistence: float
    
    # Risk assessment
    expiry_risk_score: float
    pin_risk_proximity: float
    
    def get_expiry_regime(self) -> str:
        """Get regime classification for this specific expiry"""
        
        if self.expiry_iv_percentile >= 90:
            return "high_vol_expiry"
        elif self.expiry_iv_percentile >= 75:
            return "elevated_vol_expiry"
        elif self.expiry_iv_percentile >= 25:
            return "normal_vol_expiry"
        elif self.expiry_iv_percentile >= 10:
            return "low_vol_expiry"
        else:
            return "extremely_low_vol_expiry"


class DTEPercentileFramework:
    """
    Advanced DTE-Adaptive IV Percentile Analysis Framework with both
    bucket-specific and individual DTE-level analysis capabilities.
    
    Features:
    - Individual DTE tracking (dte=0, dte=1, dte=2...dte=58)
    - DTE bucket analysis (Near: 0-7, Medium: 8-30, Far: 31+)
    - Cross-DTE percentile comparison and regime transition detection
    - Expiry-specific percentile calculation with adaptive scenarios
    - Dynamic percentile thresholds based on market conditions
    - ATM-relative percentile analysis with moneyness effects
    - Statistical significance testing for percentile reliability
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # DTE bucket configuration
        self.dte_buckets = {
            DTEBucket.NEAR: (0, 7),
            DTEBucket.MEDIUM: (8, 30), 
            DTEBucket.FAR: (31, 365)
        }
        
        # Individual DTE tracking configuration
        self.max_individual_dte = config.get('max_individual_dte', 58)
        self.min_data_points = config.get('min_data_points_dte', 20)
        
        # Dynamic threshold configuration
        self.adaptive_thresholds = config.get('use_adaptive_thresholds', True)
        self.volatility_regime_factor = config.get('volatility_regime_factor', 0.15)
        
        # Statistical significance thresholds
        self.min_significance_level = config.get('min_significance_level', 0.05)
        self.min_sample_size = config.get('min_sample_size', 30)
        
        # Performance configuration
        self.processing_budget_ms = config.get('processing_budget_ms', 100)  # Sub-budget for DTE analysis
        
        self.logger.info(f"DTE Percentile Framework initialized with individual tracking up to DTE {self.max_individual_dte}")
    
    def analyze_dte_specific_percentiles(self, iv_data: IVPercentileData,
                                       historical_db: HistoricalPercentileDatabase) -> DTEPercentileMetrics:
        """
        Analyze DTE-specific percentiles with both bucket and individual DTE analysis
        
        Args:
            iv_data: Current IV percentile data
            historical_db: Historical percentile database
            
        Returns:
            DTEPercentileMetrics with comprehensive DTE analysis
        """
        start_time = time.time()
        
        try:
            current_dte = iv_data.dte
            dte_bucket = self._classify_dte_bucket(current_dte)
            
            # Get historical percentile distribution for this specific DTE
            dte_distribution = historical_db.get_dte_percentile_distribution(current_dte)
            
            if dte_distribution is None:
                self.logger.warning(f"No historical distribution available for DTE {current_dte}")
                return self._get_default_dte_metrics(current_dte, dte_bucket)
            
            # Calculate current IV metrics for percentile comparison
            current_atm_iv = self._calculate_atm_iv(iv_data)
            current_surface_iv = self._calculate_surface_average_iv(iv_data)
            
            # Calculate individual DTE percentile
            dte_iv_percentile = dte_distribution.calculate_percentile_rank(current_atm_iv)
            
            # Calculate historical rank (position in distribution)
            dte_historical_rank = self._calculate_historical_rank(current_atm_iv, dte_distribution)
            
            # Calculate confidence score based on data quality
            dte_confidence_score = self._calculate_dte_confidence(dte_distribution, iv_data.data_completeness)
            
            # Dynamic threshold calculation
            dynamic_thresholds = self._calculate_dynamic_thresholds(
                dte_distribution, dte_bucket, current_dte
            )
            
            # ATM-specific analysis
            atm_analysis = self._analyze_atm_dte_percentiles(iv_data, dte_distribution)
            
            # Cross-DTE comparison (if individual DTE <= max tracking)
            cross_dte_comparison = None
            if current_dte <= self.max_individual_dte:
                cross_dte_comparison = self._calculate_cross_dte_comparison(
                    current_dte, current_atm_iv, historical_db
                )
            
            # Create DTE metrics result
            dte_metrics = DTEPercentileMetrics(
                dte=current_dte,
                dte_bucket=dte_bucket,
                dte_iv_percentile=dte_iv_percentile,
                dte_historical_rank=dte_historical_rank,
                dte_confidence_score=dte_confidence_score,
                dynamic_low_threshold=dynamic_thresholds['low'],
                dynamic_high_threshold=dynamic_thresholds['high'],
                regime_classification=self._classify_dte_regime(dte_iv_percentile, dynamic_thresholds),
                atm_dte_percentile=atm_analysis['atm_percentile'],
                atm_relative_strength=atm_analysis['relative_strength'],
                data_sufficiency=min(1.0, dte_distribution.count / self.min_sample_size),
                calculation_confidence=dte_confidence_score
            )
            
            # Add cross-DTE comparison if available
            if cross_dte_comparison:
                dte_metrics.relative_to_near_dte = cross_dte_comparison.get('near_differential')
                dte_metrics.relative_to_medium_dte = cross_dte_comparison.get('medium_differential')  
                dte_metrics.relative_to_far_dte = cross_dte_comparison.get('far_differential')
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.debug(f"DTE-specific analysis completed for DTE {current_dte}: "
                            f"Percentile={dte_iv_percentile:.1f}%, "
                            f"Regime={dte_metrics.regime_classification}, "
                            f"Time={processing_time:.2f}ms")
            
            return dte_metrics
            
        except Exception as e:
            self.logger.error(f"DTE-specific percentile analysis failed: {e}")
            return self._get_default_dte_metrics(iv_data.dte, self._classify_dte_bucket(iv_data.dte))
    
    def analyze_cross_dte_percentiles(self, target_dte: int, current_atm_iv: float,
                                    historical_db: HistoricalPercentileDatabase) -> CrossDTEAnalysisResult:
        """
        Analyze percentile differences across DTE buckets and individual DTEs
        for regime transition detection
        
        Args:
            target_dte: Target DTE for analysis
            current_atm_iv: Current ATM IV for comparison
            historical_db: Historical percentile database
            
        Returns:
            CrossDTEAnalysisResult with cross-DTE comparison analysis
        """
        start_time = time.time()
        
        try:
            # Get percentile distributions for different DTE ranges
            near_dtes = list(range(0, 8))   # 0-7 days
            medium_dtes = list(range(8, 31))  # 8-30 days
            far_dtes = list(range(31, min(59, self.max_individual_dte + 1)))  # 31-58 days
            
            # Calculate representative percentiles for each bucket
            bucket_percentiles = {}
            
            for bucket_name, dte_list in [('near', near_dtes), ('medium', medium_dtes), ('far', far_dtes)]:
                bucket_percentiles[bucket_name] = self._calculate_bucket_percentile(
                    dte_list, current_atm_iv, historical_db
                )
            
            # Calculate differentials
            near_medium_diff = (bucket_percentiles['near'] - bucket_percentiles['medium']) 
            medium_far_diff = (bucket_percentiles['medium'] - bucket_percentiles['far'])
            near_far_diff = (bucket_percentiles['near'] - bucket_percentiles['far'])
            
            # Term structure analysis
            term_structure = self._analyze_term_structure(bucket_percentiles)
            
            # Regime transition probability
            regime_transition_prob = self._calculate_regime_transition_probability(
                bucket_percentiles, near_medium_diff, medium_far_diff
            )
            
            # Volatility clustering analysis
            vol_clustering = self._analyze_volatility_clustering(bucket_percentiles)
            
            # Regime consistency across DTEs
            regime_consistency = self._calculate_regime_consistency(bucket_percentiles)
            
            # Determine dominant regime
            dominant_regime = self._determine_dominant_regime(bucket_percentiles)
            
            # Statistical significance assessment
            significance_metrics = self._assess_statistical_significance(
                bucket_percentiles, historical_db
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return CrossDTEAnalysisResult(
                term_structure_slope=term_structure['slope'],
                term_structure_curvature=term_structure['curvature'],
                term_structure_level=term_structure['level'],
                near_medium_differential=near_medium_diff,
                medium_far_differential=medium_far_diff,
                near_far_differential=near_far_diff,
                regime_transition_probability=regime_transition_prob,
                volatility_clustering_score=vol_clustering,
                regime_consistency_score=regime_consistency['score'],
                dominant_regime=dominant_regime,
                statistical_significance=significance_metrics['significance'],
                sample_size_adequacy=significance_metrics['sample_adequacy']
            )
            
        except Exception as e:
            self.logger.error(f"Cross-DTE percentile analysis failed: {e}")
            return self._get_default_cross_dte_result()
    
    def analyze_expiry_specific_percentiles(self, iv_data: IVPercentileData,
                                          historical_db: HistoricalPercentileDatabase) -> ExpirySpecificAnalysis:
        """
        Handle varying expiry scenarios using expiry_date and expiry_bucket columns
        at individual DTE granularity
        
        Args:
            iv_data: IV percentile data with expiry information
            historical_db: Historical percentile database
            
        Returns:
            ExpirySpecificAnalysis with expiry-specific percentile metrics
        """
        start_time = time.time()
        
        try:
            # Extract expiry information
            expiry_date = iv_data.expiry_date
            current_dte = iv_data.dte
            expiry_bucket = iv_data.expiry_bucket
            
            # Calculate current IV metrics
            current_atm_iv = self._calculate_atm_iv(iv_data)
            current_surface_iv = self._calculate_surface_average_iv(iv_data)
            
            # Get historical distribution for this specific DTE
            dte_distribution = historical_db.get_dte_percentile_distribution(current_dte)
            
            if dte_distribution is None:
                return self._get_default_expiry_analysis(iv_data)
            
            # Calculate expiry-specific percentile
            expiry_iv_percentile = dte_distribution.calculate_percentile_rank(current_atm_iv)
            
            # Historical rank calculation
            expiry_historical_rank = int(dte_distribution.count * (expiry_iv_percentile / 100))
            
            # Percentile band classification
            expiry_percentile_band = self._classify_percentile_band(expiry_iv_percentile)
            
            # ATM analysis for this expiry
            atm_analysis = self._analyze_expiry_atm_percentiles(iv_data, dte_distribution)
            
            # Volatility clustering analysis
            vol_clustering = self._analyze_expiry_volatility_clustering(
                current_atm_iv, dte_distribution, current_dte
            )
            
            # Risk assessment
            risk_assessment = self._assess_expiry_risk(iv_data, expiry_iv_percentile, current_dte)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ExpirySpecificAnalysis(
                expiry_date=expiry_date,
                days_to_expiry=current_dte,
                expiry_bucket=expiry_bucket,
                expiry_iv_percentile=expiry_iv_percentile,
                expiry_historical_rank=expiry_historical_rank,
                expiry_percentile_band=expiry_percentile_band,
                expiry_atm_percentile=atm_analysis['atm_percentile'],
                atm_moneyness_effect=atm_analysis['moneyness_effect'],
                expiry_vol_clustering=vol_clustering['clustering_score'],
                clustering_persistence=vol_clustering['persistence'],
                expiry_risk_score=risk_assessment['risk_score'],
                pin_risk_proximity=risk_assessment['pin_risk']
            )
            
        except Exception as e:
            self.logger.error(f"Expiry-specific percentile analysis failed: {e}")
            return self._get_default_expiry_analysis(iv_data)
    
    def calculate_adaptive_percentile_thresholds(self, dte: int, dte_bucket: DTEBucket,
                                               distribution: PercentileDistribution) -> Dict[str, float]:
        """
        Calculate dynamic percentile thresholds based on market conditions
        and volatility clustering per DTE
        
        Args:
            dte: Days to expiry
            dte_bucket: DTE bucket classification
            distribution: Historical percentile distribution
            
        Returns:
            Dictionary with adaptive threshold levels
        """
        try:
            # Base thresholds by DTE bucket
            base_thresholds = {
                DTEBucket.NEAR: {'low': 20, 'high': 80},    # More sensitive for near-term
                DTEBucket.MEDIUM: {'low': 25, 'high': 75},  # Standard sensitivity
                DTEBucket.FAR: {'low': 30, 'high': 70}     # Less sensitive for far-term
            }
            
            base_low = base_thresholds[dte_bucket]['low']
            base_high = base_thresholds[dte_bucket]['high']
            
            if not self.adaptive_thresholds:
                return {'low': base_low, 'high': base_high}
            
            # Volatility regime adjustment
            vol_regime_factor = self._calculate_volatility_regime_factor(distribution)
            
            # DTE-specific adjustment
            dte_adjustment = self._calculate_dte_specific_adjustment(dte)
            
            # Market condition adjustment
            market_condition_adj = self._calculate_market_condition_adjustment(distribution)
            
            # Combined adjustment
            total_adjustment = (vol_regime_factor + dte_adjustment + market_condition_adj) / 3
            
            # Apply adjustments
            adjusted_low = base_low + (total_adjustment * self.volatility_regime_factor * 100)
            adjusted_high = base_high - (total_adjustment * self.volatility_regime_factor * 100)
            
            # Ensure reasonable bounds
            adjusted_low = max(5.0, min(35.0, adjusted_low))
            adjusted_high = max(65.0, min(95.0, adjusted_high))
            
            return {
                'low': float(adjusted_low),
                'high': float(adjusted_high),
                'base_low': float(base_low),
                'base_high': float(base_high),
                'adjustment_factor': float(total_adjustment),
                'volatility_regime_factor': float(vol_regime_factor)
            }
            
        except Exception as e:
            self.logger.error(f"Adaptive threshold calculation failed: {e}")
            return {'low': 25.0, 'high': 75.0}
    
    def _classify_dte_bucket(self, dte: int) -> DTEBucket:
        """Classify DTE into bucket category"""
        
        for bucket, (min_dte, max_dte) in self.dte_buckets.items():
            if min_dte <= dte <= max_dte:
                return bucket
        
        return DTEBucket.FAR  # Default for DTEs > 365
    
    def _calculate_atm_iv(self, iv_data: IVPercentileData) -> float:
        """Calculate ATM IV from IV data"""
        
        atm_idx = np.argmin(np.abs(iv_data.strikes - iv_data.atm_strike))
        atm_ce_iv = iv_data.ce_iv[atm_idx] if not np.isnan(iv_data.ce_iv[atm_idx]) else 0.0
        atm_pe_iv = iv_data.pe_iv[atm_idx] if not np.isnan(iv_data.pe_iv[atm_idx]) else 0.0
        
        if atm_ce_iv > 0 and atm_pe_iv > 0:
            return float((atm_ce_iv + atm_pe_iv) / 2)
        elif atm_ce_iv > 0:
            return float(atm_ce_iv)
        elif atm_pe_iv > 0:
            return float(atm_pe_iv)
        else:
            return 0.0
    
    def _calculate_surface_average_iv(self, iv_data: IVPercentileData) -> float:
        """Calculate surface average IV"""
        
        valid_ivs = np.concatenate([
            iv_data.ce_iv[~np.isnan(iv_data.ce_iv) & (iv_data.ce_iv > 0)],
            iv_data.pe_iv[~np.isnan(iv_data.pe_iv) & (iv_data.pe_iv > 0)]
        ])
        
        return float(np.mean(valid_ivs)) if len(valid_ivs) > 0 else 0.0
    
    def _calculate_historical_rank(self, current_value: float, distribution: PercentileDistribution) -> float:
        """Calculate historical rank position"""
        
        if distribution.count == 0:
            return 0.0
        
        percentile = distribution.calculate_percentile_rank(current_value)
        return float(percentile / 100.0 * distribution.count)
    
    def _calculate_dte_confidence(self, distribution: PercentileDistribution, data_completeness: float) -> float:
        """Calculate confidence score for DTE analysis"""
        
        factors = []
        
        # Sample size factor
        sample_factor = min(1.0, distribution.count / self.min_sample_size)
        factors.append(sample_factor)
        
        # Data completeness factor
        factors.append(data_completeness)
        
        # Distribution quality factor (based on standard deviation)
        if distribution.std > 0:
            cv = distribution.std / (distribution.mean + 1e-10)
            quality_factor = 1.0 / (1.0 + cv)  # Lower CV = higher quality
            factors.append(quality_factor)
        
        return float(np.mean(factors))
    
    def _calculate_dynamic_thresholds(self, distribution: PercentileDistribution,
                                    dte_bucket: DTEBucket, dte: int) -> Dict[str, float]:
        """Calculate dynamic percentile thresholds"""
        
        return self.calculate_adaptive_percentile_thresholds(dte, dte_bucket, distribution)
    
    def _classify_dte_regime(self, percentile: float, thresholds: Dict[str, float]) -> str:
        """Classify DTE regime based on percentile and thresholds"""
        
        low_threshold = thresholds.get('low', 25.0)
        high_threshold = thresholds.get('high', 75.0)
        
        if percentile >= 95:
            return "extremely_high"
        elif percentile >= high_threshold:
            return "high"
        elif percentile >= 60:
            return "above_normal"
        elif percentile >= 40:
            return "normal"
        elif percentile >= low_threshold:
            return "below_normal"
        elif percentile >= 5:
            return "low"
        else:
            return "extremely_low"
    
    def _analyze_atm_dte_percentiles(self, iv_data: IVPercentileData,
                                   distribution: PercentileDistribution) -> Dict[str, float]:
        """Analyze ATM-specific DTE percentiles"""
        
        current_atm_iv = self._calculate_atm_iv(iv_data)
        atm_percentile = distribution.calculate_percentile_rank(current_atm_iv)
        
        # Calculate relative strength vs surface
        surface_iv = self._calculate_surface_average_iv(iv_data)
        if surface_iv > 0:
            relative_strength = (current_atm_iv - surface_iv) / surface_iv
        else:
            relative_strength = 0.0
        
        return {
            'atm_percentile': float(atm_percentile),
            'relative_strength': float(relative_strength),
            'current_atm_iv': float(current_atm_iv)
        }
    
    def _calculate_cross_dte_comparison(self, target_dte: int, current_atm_iv: float,
                                      historical_db: HistoricalPercentileDatabase) -> Dict[str, float]:
        """Calculate cross-DTE comparison metrics"""
        
        comparisons = {}
        
        # Define representative DTEs for each bucket
        representative_dtes = {
            'near': [0, 1, 3, 7],
            'medium': [14, 21, 30], 
            'far': [45, 60, 90] if self.max_individual_dte >= 90 else [45]
        }
        
        target_bucket = self._classify_dte_bucket(target_dte)
        
        for bucket_name, dte_list in representative_dtes.items():
            if bucket_name == target_bucket.value:
                continue
            
            bucket_percentiles = []
            for dte in dte_list:
                if dte <= self.max_individual_dte:
                    dist = historical_db.get_dte_percentile_distribution(dte)
                    if dist:
                        percentile = dist.calculate_percentile_rank(current_atm_iv)
                        bucket_percentiles.append(percentile)
            
            if bucket_percentiles:
                avg_percentile = np.mean(bucket_percentiles)
                # Calculate target DTE percentile
                target_dist = historical_db.get_dte_percentile_distribution(target_dte)
                if target_dist:
                    target_percentile = target_dist.calculate_percentile_rank(current_atm_iv)
                    differential = target_percentile - avg_percentile
                    comparisons[f'{bucket_name}_differential'] = float(differential)
        
        return comparisons
    
    def _calculate_bucket_percentile(self, dte_list: List[int], current_atm_iv: float,
                                   historical_db: HistoricalPercentileDatabase) -> float:
        """Calculate representative percentile for a DTE bucket"""
        
        percentiles = []
        
        for dte in dte_list:
            if dte <= self.max_individual_dte:
                distribution = historical_db.get_dte_percentile_distribution(dte)
                if distribution:
                    percentile = distribution.calculate_percentile_rank(current_atm_iv)
                    percentiles.append(percentile)
        
        return float(np.mean(percentiles)) if percentiles else 50.0
    
    def _analyze_term_structure(self, bucket_percentiles: Dict[str, float]) -> Dict[str, float]:
        """Analyze term structure using percentile levels"""
        
        near = bucket_percentiles.get('near', 50.0)
        medium = bucket_percentiles.get('medium', 50.0)
        far = bucket_percentiles.get('far', 50.0)
        
        # Calculate slope (far - near)
        slope = far - near
        
        # Calculate curvature (convexity)
        curvature = medium - (near + far) / 2
        
        # Calculate level (average)
        level = (near + medium + far) / 3
        
        return {
            'slope': float(slope),
            'curvature': float(curvature),
            'level': float(level)
        }
    
    def _calculate_regime_transition_probability(self, bucket_percentiles: Dict[str, float],
                                               near_med_diff: float, med_far_diff: float) -> float:
        """Calculate regime transition probability"""
        
        # High differential suggests regime instability
        max_differential = max(abs(near_med_diff), abs(med_far_diff))
        
        # Transition probability increases with differential magnitude
        transition_prob = min(1.0, max_differential / 50.0)  # Normalize to 0-1
        
        return float(transition_prob)
    
    def _analyze_volatility_clustering(self, bucket_percentiles: Dict[str, float]) -> float:
        """Analyze volatility clustering score"""
        
        percentiles = list(bucket_percentiles.values())
        
        if len(percentiles) < 2:
            return 0.5
        
        # Higher variance indicates less clustering
        variance = np.var(percentiles)
        clustering_score = 1.0 / (1.0 + variance / 100)  # Normalize
        
        return float(clustering_score)
    
    def _calculate_regime_consistency(self, bucket_percentiles: Dict[str, float]) -> Dict[str, float]:
        """Calculate regime consistency across DTEs"""
        
        percentiles = list(bucket_percentiles.values())
        
        if len(percentiles) < 2:
            return {'score': 0.5, 'variance': 0.0}
        
        variance = np.var(percentiles)
        consistency_score = 1.0 / (1.0 + variance / 100)
        
        return {
            'score': float(consistency_score),
            'variance': float(variance)
        }
    
    def _determine_dominant_regime(self, bucket_percentiles: Dict[str, float]) -> str:
        """Determine dominant regime across DTEs"""
        
        avg_percentile = np.mean(list(bucket_percentiles.values()))
        
        if avg_percentile >= 80:
            return "high_volatility"
        elif avg_percentile >= 60:
            return "elevated_volatility"
        elif avg_percentile >= 40:
            return "normal_volatility"
        elif avg_percentile >= 20:
            return "low_volatility"
        else:
            return "very_low_volatility"
    
    def _assess_statistical_significance(self, bucket_percentiles: Dict[str, float],
                                       historical_db: HistoricalPercentileDatabase) -> Dict[str, float]:
        """Assess statistical significance of percentile differences"""
        
        # Simple significance assessment based on sample sizes
        total_samples = 0
        for bucket in ['near', 'medium', 'far']:
            # Estimate samples (would need actual counting in production)
            total_samples += 100  # Placeholder
        
        sample_adequacy = min(1.0, total_samples / (self.min_sample_size * 3))
        
        # Significance based on percentile variance
        percentiles = list(bucket_percentiles.values())
        variance = np.var(percentiles)
        significance = min(1.0, variance / 50)  # Higher variance = more significant
        
        return {
            'significance': float(significance),
            'sample_adequacy': float(sample_adequacy)
        }
    
    def _classify_percentile_band(self, percentile: float) -> str:
        """Classify percentile into bands"""
        
        if percentile >= 90:
            return "very_high"
        elif percentile >= 75:
            return "high"
        elif percentile >= 60:
            return "above_average"
        elif percentile >= 40:
            return "average"
        elif percentile >= 25:
            return "below_average"
        elif percentile >= 10:
            return "low"
        else:
            return "very_low"
    
    def _analyze_expiry_atm_percentiles(self, iv_data: IVPercentileData,
                                      distribution: PercentileDistribution) -> Dict[str, float]:
        """Analyze ATM percentiles for specific expiry"""
        
        current_atm_iv = self._calculate_atm_iv(iv_data)
        atm_percentile = distribution.calculate_percentile_rank(current_atm_iv)
        
        # Moneyness effect calculation
        atm_distance = abs(iv_data.atm_strike - iv_data.spot) / iv_data.spot
        moneyness_effect = atm_distance * 100  # Convert to percentage
        
        return {
            'atm_percentile': float(atm_percentile),
            'moneyness_effect': float(moneyness_effect)
        }
    
    def _analyze_expiry_volatility_clustering(self, current_atm_iv: float,
                                            distribution: PercentileDistribution,
                                            dte: int) -> Dict[str, float]:
        """Analyze volatility clustering for specific expiry"""
        
        # Clustering based on how far current IV is from mean
        if distribution.mean > 0:
            deviation = abs(current_atm_iv - distribution.mean) / distribution.mean
            clustering_score = 1.0 / (1.0 + deviation)
        else:
            clustering_score = 0.5
        
        # Persistence based on DTE (shorter DTE = less persistent)
        persistence = max(0.1, min(1.0, dte / 30))
        
        return {
            'clustering_score': float(clustering_score),
            'persistence': float(persistence)
        }
    
    def _assess_expiry_risk(self, iv_data: IVPercentileData, percentile: float, dte: int) -> Dict[str, float]:
        """Assess risk for specific expiry"""
        
        # Risk increases with extreme percentiles and short DTE
        percentile_risk = max(abs(percentile - 50), 0) / 50  # 0-1 scale
        dte_risk = max(0.1, 1.0 - dte / 30)  # Higher risk for shorter DTE
        
        risk_score = (percentile_risk + dte_risk) / 2
        
        # Pin risk calculation (distance from round numbers)
        spot = iv_data.spot
        nearest_round = round(spot / 100) * 100
        pin_risk = max(0, 1.0 - abs(spot - nearest_round) / 100)
        
        return {
            'risk_score': float(risk_score),
            'pin_risk': float(pin_risk)
        }
    
    def _calculate_volatility_regime_factor(self, distribution: PercentileDistribution) -> float:
        """Calculate volatility regime factor for threshold adjustment"""
        
        if distribution.mean == 0:
            return 0.0
        
        # Higher coefficient of variation indicates more volatile regime
        cv = distribution.std / distribution.mean
        regime_factor = np.tanh(cv)  # Bounded between -1 and 1
        
        return float(regime_factor)
    
    def _calculate_dte_specific_adjustment(self, dte: int) -> float:
        """Calculate DTE-specific adjustment factor"""
        
        # Shorter DTEs need more sensitive thresholds
        if dte <= 7:
            return 0.2  # More sensitive
        elif dte <= 30:
            return 0.0  # Standard
        else:
            return -0.1  # Less sensitive
    
    def _calculate_market_condition_adjustment(self, distribution: PercentileDistribution) -> float:
        """Calculate market condition adjustment"""
        
        # Based on recent distribution characteristics
        # Higher standard deviation suggests more volatile market
        if distribution.std > 0:
            normalized_std = min(1.0, distribution.std / 20)  # Normalize
            return normalized_std * 0.15  # Small adjustment
        
        return 0.0
    
    def _get_default_dte_metrics(self, dte: int, dte_bucket: DTEBucket) -> DTEPercentileMetrics:
        """Get default DTE metrics when data is insufficient"""
        
        return DTEPercentileMetrics(
            dte=dte,
            dte_bucket=dte_bucket,
            dte_iv_percentile=50.0,
            dte_historical_rank=0.0,
            dte_confidence_score=0.0,
            regime_classification="normal",
            atm_dte_percentile=50.0,
            atm_relative_strength=0.0,
            data_sufficiency=0.0,
            calculation_confidence=0.0
        )
    
    def _get_default_cross_dte_result(self) -> CrossDTEAnalysisResult:
        """Get default cross-DTE result when analysis fails"""
        
        return CrossDTEAnalysisResult(
            term_structure_slope=0.0,
            term_structure_curvature=0.0,
            term_structure_level=50.0,
            near_medium_differential=0.0,
            medium_far_differential=0.0,
            near_far_differential=0.0,
            regime_transition_probability=0.5,
            volatility_clustering_score=0.5,
            regime_consistency_score=0.5,
            dominant_regime="normal_volatility",
            statistical_significance=0.0,
            sample_size_adequacy=0.0
        )
    
    def _get_default_expiry_analysis(self, iv_data: IVPercentileData) -> ExpirySpecificAnalysis:
        """Get default expiry analysis when data is insufficient"""
        
        return ExpirySpecificAnalysis(
            expiry_date=iv_data.expiry_date,
            days_to_expiry=iv_data.dte,
            expiry_bucket=iv_data.expiry_bucket,
            expiry_iv_percentile=50.0,
            expiry_historical_rank=0,
            expiry_percentile_band="average",
            expiry_atm_percentile=50.0,
            atm_moneyness_effect=0.0,
            expiry_vol_clustering=0.5,
            clustering_persistence=0.5,
            expiry_risk_score=0.5,
            pin_risk_proximity=0.0
        )