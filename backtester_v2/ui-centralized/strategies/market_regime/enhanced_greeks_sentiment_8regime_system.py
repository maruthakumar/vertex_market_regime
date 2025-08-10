#!/usr/bin/env python3
"""
Enhanced Greeks Sentiment Analysis for 8-Regime System
======================================================

This module implements comprehensive Greeks sentiment analysis aligned with the
8-regime classification system, featuring volume-weighted calculations, DTE-specific
adjustments, and cross-Greek correlation analysis.

Key Features:
- Volume-weighted Greeks calculations from 49-column dataset
- 7-level sentiment classification system
- DTE-specific Greek adjustments and time decay analysis
- Cross-Greek correlation and interaction analysis
- Regime-specific Greeks interpretation
- Real-time processing with <3 second requirement
- Excel configuration integration

Author: Mary - Business Analyst
Date: 2025-01-09
Version: 1.0.0 - Enhanced Greeks Sentiment for 8-Regime System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GreekSentimentLevel(Enum):
    """7-Level Greek sentiment classification"""
    EXTREMELY_BULLISH = 3.0
    VERY_BULLISH = 2.0
    MODERATELY_BULLISH = 1.0
    NEUTRAL = 0.0
    MODERATELY_BEARISH = -1.0
    VERY_BEARISH = -2.0
    EXTREMELY_BEARISH = -3.0

class MarketRegime(Enum):
    """8-Regime classification system"""
    LVLD = "Low_Volatility_Linear_Decay"
    HVC = "High_Volatility_Clustering"
    VCPE = "Volatility_Crush_Post_Event"
    TBVE = "Trending_Bull_Volatility_Expansion"
    TBVS = "Trending_Bear_Volatility_Spike"
    SCGS = "Sideways_Choppy_Gamma_Scalping"
    PSED = "Premium_Spike_Event_Driven"
    CBV = "Correlation_Breakdown_Volatility"

@dataclass
class GreekAnalysisResult:
    """Result structure for individual Greek analysis"""
    greek_name: str
    volume_weighted_value: float
    sentiment_level: GreekSentimentLevel
    confidence: float
    dte_adjusted_value: float
    percentile_rank: float
    cross_correlations: Dict[str, float]
    time_decay_impact: float
    regime_interpretation: Dict[str, float]

@dataclass
class GreeksSentimentResult:
    """Complete Greeks sentiment analysis result"""
    timestamp: datetime
    individual_greeks: Dict[str, GreekAnalysisResult]
    combined_sentiment: GreekSentimentLevel
    combined_confidence: float
    portfolio_greeks: Dict[str, float]
    risk_metrics: Dict[str, float]
    regime_adjustments: Dict[str, float]
    feature_scores: Dict[str, float]

class EnhancedGreeksSentiment8RegimeSystem:
    """
    Enhanced Greeks Sentiment Analysis for 8-Regime Classification
    
    This system provides comprehensive Greeks analysis with:
    - Volume-weighted calculations from options data
    - DTE-specific adjustments and time decay modeling
    - Cross-Greek correlation and interaction analysis
    - Regime-specific Greeks interpretation
    - 7-level sentiment classification
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced Greeks sentiment system"""
        
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # Greek weights for combined analysis
        self.greek_weights = {
            'delta': 0.25,  # Directional exposure
            'gamma': 0.30,  # Convexity and pin risk
            'theta': 0.20,  # Time decay impact
            'vega': 0.20,   # Volatility sensitivity
            'rho': 0.05     # Interest rate sensitivity
        }
        
        # DTE adjustment parameters
        self.dte_adjustments = {
            'delta': {'type': 'linear', 'decay_rate': 0.02},
            'gamma': {'type': 'exponential', 'peak_dte': 7, 'decay_rate': 0.15},
            'theta': {'type': 'accelerating', 'acceleration_dte': 15},
            'vega': {'type': 'iv_based', 'sensitivity_factor': 0.8},
            'rho': {'type': 'interest_rate', 'base_rate': 0.05}
        }
        
        # Sentiment level thresholds
        self.sentiment_thresholds = {
            'EXTREMELY_BULLISH': 0.8,
            'VERY_BULLISH': 0.6,
            'MODERATELY_BULLISH': 0.3,
            'NEUTRAL': 0.0,
            'MODERATELY_BEARISH': -0.3,
            'VERY_BEARISH': -0.6,
            'EXTREMELY_BEARISH': -0.8
        }
        
        # Regime-specific Greek interpretations
        self.regime_interpretations = self._initialize_regime_interpretations()
        
        # Historical data for percentile calculations
        self.greek_history = {greek: [] for greek in self.greek_weights.keys()}
        
        logger.info("Enhanced Greeks Sentiment 8-Regime System initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from Excel file or use defaults"""
        
        if self.config_path:
            try:
                # Load Excel configuration (implementation would read actual Excel)
                logger.info(f"Loading Greeks configuration from {self.config_path}")
                return self._load_excel_config(self.config_path)
            except Exception as e:
                logger.warning(f"Failed to load Excel config: {e}")
        
        # Default configuration
        return {
            'volume_weight_threshold': 0.01,  # Minimum volume for weighting
            'confidence_threshold': 0.7,
            'correlation_window': 50,
            'percentile_window': 252,  # 1 year of trading days
            'outlier_threshold': 3.0   # Z-score threshold for outliers
        }
    
    def _load_excel_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from Excel file"""
        # Implementation would read actual Excel file
        return self.config if hasattr(self, 'config') else {}
    
    def _initialize_regime_interpretations(self) -> Dict[str, Dict[str, float]]:
        """Initialize regime-specific Greek interpretations"""
        
        return {
            'LVLD': {  # Low Volatility Linear Decay
                'delta_importance': 0.7,    # Moderate directional focus
                'gamma_importance': 0.5,    # Low gamma risk
                'theta_importance': 1.0,    # High theta capture
                'vega_importance': 0.3,     # Low volatility sensitivity
                'risk_multiplier': 1.2      # Can take more risk
            },
            'HVC': {  # High Volatility Clustering
                'delta_importance': 0.6,
                'gamma_importance': 0.9,    # High gamma risk
                'theta_importance': 0.7,
                'vega_importance': 0.8,     # High volatility sensitivity
                'risk_multiplier': 0.7      # Reduce risk
            },
            'VCPE': {  # Volatility Crush Post-Event
                'delta_importance': 0.5,
                'gamma_importance': 0.4,    # Gamma normalizing
                'theta_importance': 1.0,    # Maximum theta benefit
                'vega_importance': 1.0,     # Vega crush benefit
                'risk_multiplier': 1.3      # Can be aggressive
            },
            'TBVE': {  # Trending Bull with Volatility Expansion
                'delta_importance': 0.8,    # High directional importance
                'gamma_importance': 0.7,
                'theta_importance': 0.6,
                'vega_importance': 0.7,
                'risk_multiplier': 1.0
            },
            'TBVS': {  # Trending Bear with Volatility Spike
                'delta_importance': 0.9,    # Critical directional risk
                'gamma_importance': 1.0,    # Maximum gamma risk
                'theta_importance': 0.3,    # Theta overwhelmed
                'vega_importance': 1.0,     # High volatility impact
                'risk_multiplier': 0.3      # Minimal risk
            },
            'SCGS': {  # Sideways Choppy Gamma Scalping
                'delta_importance': 0.4,
                'gamma_importance': 1.0,    # Gamma is key
                'theta_importance': 0.8,
                'vega_importance': 0.6,
                'risk_multiplier': 0.8
            },
            'PSED': {  # Premium Spike Event-Driven
                'delta_importance': 0.6,
                'gamma_importance': 0.8,
                'theta_importance': 0.9,    # Time decay important
                'vega_importance': 0.9,     # Event volatility
                'risk_multiplier': 1.1
            },
            'CBV': {  # Correlation Breakdown Volatility
                'delta_importance': 0.5,
                'gamma_importance': 0.8,
                'theta_importance': 0.5,
                'vega_importance': 0.8,
                'risk_multiplier': 0.6      # Uncertain environment
            }
        }
    
    def analyze_greeks_sentiment(self, data: pd.DataFrame, 
                               current_regime: Optional[MarketRegime] = None) -> GreeksSentimentResult:
        """
        Complete Greeks sentiment analysis
        
        Args:
            data: DataFrame with 49-column options data
            current_regime: Current market regime for context
            
        Returns:
            GreeksSentimentResult with comprehensive analysis
        """
        
        try:
            logger.info("Starting Greeks sentiment analysis")
            
            # Calculate volume-weighted Greeks
            volume_weighted_greeks = self._calculate_volume_weighted_greeks(data)
            
            # Analyze individual Greeks
            individual_results = {}
            for greek_name in self.greek_weights.keys():
                result = self._analyze_individual_greek(
                    data, greek_name, volume_weighted_greeks, current_regime
                )
                individual_results[greek_name] = result
            
            # Calculate combined sentiment
            combined_sentiment, combined_confidence = self._calculate_combined_sentiment(
                individual_results
            )
            
            # Calculate portfolio Greeks
            portfolio_greeks = self._calculate_portfolio_greeks(data, individual_results)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(individual_results, current_regime)
            
            # Regime-specific adjustments
            regime_adjustments = self._calculate_regime_adjustments(
                individual_results, current_regime
            )
            
            # Feature scores for 8-regime classification
            feature_scores = self._calculate_feature_scores(individual_results)
            
            # Update historical data
            self._update_greek_history(volume_weighted_greeks)
            
            result = GreeksSentimentResult(
                timestamp=datetime.now(),
                individual_greeks=individual_results,
                combined_sentiment=combined_sentiment,
                combined_confidence=combined_confidence,
                portfolio_greeks=portfolio_greeks,
                risk_metrics=risk_metrics,
                regime_adjustments=regime_adjustments,
                feature_scores=feature_scores
            )
            
            logger.info(f"Greeks analysis completed: {combined_sentiment.name} (confidence: {combined_confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Greeks sentiment analysis: {e}")
            raise
    
    def _calculate_volume_weighted_greeks(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-weighted Greeks from options data"""
        
        try:
            volume_weighted = {}
            
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                # Call Greeks
                ce_greek_col = f'ce_{greek}'
                ce_volume_col = 'ce_volume'
                
                # Put Greeks
                pe_greek_col = f'pe_{greek}'
                pe_volume_col = 'pe_volume'
                
                if all(col in data.columns for col in [ce_greek_col, ce_volume_col, pe_greek_col, pe_volume_col]):
                    # Filter out zero volume
                    ce_data = data[data[ce_volume_col] > self.config['volume_weight_threshold']]
                    pe_data = data[data[pe_volume_col] > self.config['volume_weight_threshold']]
                    
                    if len(ce_data) > 0 and len(pe_data) > 0:
                        # Volume-weighted calculation for calls
                        ce_vw = (ce_data[ce_greek_col] * ce_data[ce_volume_col]).sum() / ce_data[ce_volume_col].sum()
                        
                        # Volume-weighted calculation for puts  
                        pe_vw = (pe_data[pe_greek_col] * pe_data[pe_volume_col]).sum() / pe_data[pe_volume_col].sum()
                        
                        # Combined volume-weighted Greek
                        total_volume = ce_data[ce_volume_col].sum() + pe_data[pe_volume_col].sum()
                        
                        if total_volume > 0:
                            volume_weighted[greek] = (
                                (ce_vw * ce_data[ce_volume_col].sum() + pe_vw * pe_data[pe_volume_col].sum()) /
                                total_volume
                            )
                        else:
                            volume_weighted[greek] = 0.0
                    else:
                        volume_weighted[greek] = 0.0
                else:
                    logger.warning(f"Missing columns for Greek {greek}")
                    volume_weighted[greek] = 0.0
            
            logger.info(f"Calculated volume-weighted Greeks: {volume_weighted}")
            return volume_weighted
            
        except Exception as e:
            logger.error(f"Error calculating volume-weighted Greeks: {e}")
            return {greek: 0.0 for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']}
    
    def _analyze_individual_greek(self, data: pd.DataFrame, greek_name: str,
                                volume_weighted_greeks: Dict[str, float],
                                current_regime: Optional[MarketRegime]) -> GreekAnalysisResult:
        """Analyze individual Greek with comprehensive metrics"""
        
        try:
            vw_value = volume_weighted_greeks.get(greek_name, 0.0)
            
            # DTE adjustment
            dte_adjusted = self._apply_dte_adjustment(data, greek_name, vw_value)
            
            # Sentiment classification
            sentiment_level = self._classify_greek_sentiment(greek_name, dte_adjusted)
            
            # Confidence calculation
            confidence = self._calculate_greek_confidence(data, greek_name, vw_value)
            
            # Percentile rank
            percentile_rank = self._calculate_percentile_rank(greek_name, vw_value)
            
            # Cross-correlations with other Greeks
            cross_correlations = self._calculate_cross_correlations(data, greek_name)
            
            # Time decay impact (especially important for theta)
            time_decay_impact = self._calculate_time_decay_impact(data, greek_name)
            
            # Regime-specific interpretation
            regime_interpretation = self._get_regime_interpretation(greek_name, current_regime)
            
            return GreekAnalysisResult(
                greek_name=greek_name,
                volume_weighted_value=vw_value,
                sentiment_level=sentiment_level,
                confidence=confidence,
                dte_adjusted_value=dte_adjusted,
                percentile_rank=percentile_rank,
                cross_correlations=cross_correlations,
                time_decay_impact=time_decay_impact,
                regime_interpretation=regime_interpretation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing Greek {greek_name}: {e}")
            return GreekAnalysisResult(
                greek_name=greek_name,
                volume_weighted_value=0.0,
                sentiment_level=GreekSentimentLevel.NEUTRAL,
                confidence=0.0,
                dte_adjusted_value=0.0,
                percentile_rank=0.5,
                cross_correlations={},
                time_decay_impact=0.0,
                regime_interpretation={}
            )
    
    def _apply_dte_adjustment(self, data: pd.DataFrame, greek_name: str, 
                            base_value: float) -> float:
        """Apply DTE-specific adjustments to Greek values"""
        
        try:
            if 'dte' not in data.columns or len(data) == 0:
                return base_value
            
            avg_dte = data['dte'].mean()
            adjustment_config = self.dte_adjustments.get(greek_name, {})
            adjustment_type = adjustment_config.get('type', 'linear')
            
            if adjustment_type == 'linear':
                # Linear decay for delta
                decay_rate = adjustment_config.get('decay_rate', 0.02)
                adjustment = 1.0 - (decay_rate * max(0, 30 - avg_dte) / 30)
                
            elif adjustment_type == 'exponential':
                # Exponential for gamma (peaks around 7 DTE)
                peak_dte = adjustment_config.get('peak_dte', 7)
                decay_rate = adjustment_config.get('decay_rate', 0.15)
                adjustment = np.exp(-decay_rate * abs(avg_dte - peak_dte))
                
            elif adjustment_type == 'accelerating':
                # Accelerating for theta
                acceleration_dte = adjustment_config.get('acceleration_dte', 15)
                if avg_dte <= acceleration_dte:
                    adjustment = 1.0 + (acceleration_dte - avg_dte) / acceleration_dte
                else:
                    adjustment = 1.0
                    
            elif adjustment_type == 'iv_based':
                # IV-based for vega
                sensitivity = adjustment_config.get('sensitivity_factor', 0.8)
                if 'ce_iv' in data.columns:
                    avg_iv = data['ce_iv'].mean()
                    adjustment = 1.0 + sensitivity * (avg_iv - 0.2)  # Assume 20% base IV
                else:
                    adjustment = 1.0
                    
            elif adjustment_type == 'interest_rate':
                # Interest rate based for rho
                base_rate = adjustment_config.get('base_rate', 0.05)
                adjustment = 1.0 + (base_rate * avg_dte / 365)  # Annualized
                
            else:
                adjustment = 1.0
            
            adjusted_value = base_value * max(0.1, adjustment)  # Prevent negative adjustments
            
            return adjusted_value
            
        except Exception as e:
            logger.error(f"Error in DTE adjustment for {greek_name}: {e}")
            return base_value
    
    def _classify_greek_sentiment(self, greek_name: str, adjusted_value: float) -> GreekSentimentLevel:
        """Classify Greek sentiment based on adjusted value"""
        
        try:
            # Normalize the value based on Greek type
            if greek_name == 'delta':
                # Delta: -1 to +1, with 0 being neutral
                normalized = adjusted_value
            elif greek_name == 'gamma':
                # Gamma: Always positive, higher is more bullish for volatility
                normalized = min(adjusted_value / 0.1, 1.0)  # Assume 0.1 as high gamma
            elif greek_name == 'theta':
                # Theta: Usually negative, less negative is better for sellers
                normalized = -adjusted_value / 10.0  # Assume -10 as high theta
            elif greek_name == 'vega':
                # Vega: Positive, higher means more vol sensitivity
                normalized = min(adjusted_value / 20.0, 1.0)  # Assume 20 as high vega
            elif greek_name == 'rho':
                # Rho: Can be positive or negative, generally small
                normalized = adjusted_value / 5.0  # Assume 5 as significant rho
            else:
                normalized = 0.0
            
            # Classify based on thresholds
            if normalized >= self.sentiment_thresholds['EXTREMELY_BULLISH']:
                return GreekSentimentLevel.EXTREMELY_BULLISH
            elif normalized >= self.sentiment_thresholds['VERY_BULLISH']:
                return GreekSentimentLevel.VERY_BULLISH
            elif normalized >= self.sentiment_thresholds['MODERATELY_BULLISH']:
                return GreekSentimentLevel.MODERATELY_BULLISH
            elif normalized >= self.sentiment_thresholds['MODERATELY_BEARISH']:
                return GreekSentimentLevel.NEUTRAL
            elif normalized >= self.sentiment_thresholds['VERY_BEARISH']:
                return GreekSentimentLevel.MODERATELY_BEARISH
            elif normalized >= self.sentiment_thresholds['EXTREMELY_BEARISH']:
                return GreekSentimentLevel.VERY_BEARISH
            else:
                return GreekSentimentLevel.EXTREMELY_BEARISH
                
        except Exception as e:
            logger.error(f"Error classifying sentiment for {greek_name}: {e}")
            return GreekSentimentLevel.NEUTRAL
    
    def _calculate_greek_confidence(self, data: pd.DataFrame, greek_name: str, 
                                  vw_value: float) -> float:
        """Calculate confidence in Greek analysis"""
        
        try:
            confidence_factors = []
            
            # Data quality factor
            total_volume = 0
            if f'ce_volume' in data.columns and f'pe_volume' in data.columns:
                total_volume = data['ce_volume'].sum() + data['pe_volume'].sum()
            
            volume_factor = min(total_volume / 10000.0, 1.0)  # Assume 10K as good volume
            confidence_factors.append(volume_factor)
            
            # Consistency factor (low variance in Greek values)
            if f'ce_{greek_name}' in data.columns and f'pe_{greek_name}' in data.columns:
                ce_values = data[f'ce_{greek_name}'].dropna()
                pe_values = data[f'pe_{greek_name}'].dropna()
                
                if len(ce_values) > 1 and len(pe_values) > 1:
                    ce_consistency = 1.0 / (1.0 + ce_values.std())
                    pe_consistency = 1.0 / (1.0 + pe_values.std())
                    consistency_factor = (ce_consistency + pe_consistency) / 2.0
                    confidence_factors.append(consistency_factor)
            
            # Outlier factor (penalize extreme values)
            if abs(vw_value) < self.config['outlier_threshold']:
                outlier_factor = 1.0
            else:
                outlier_factor = 1.0 / (1.0 + abs(vw_value) / self.config['outlier_threshold'])
            confidence_factors.append(outlier_factor)
            
            # Combined confidence
            if confidence_factors:
                confidence = np.mean(confidence_factors)
            else:
                confidence = 0.5  # Default moderate confidence
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence for {greek_name}: {e}")
            return 0.5
    
    def _calculate_percentile_rank(self, greek_name: str, current_value: float) -> float:
        """Calculate percentile rank of current Greek value"""
        
        try:
            if greek_name not in self.greek_history:
                return 0.5  # Default to median
            
            history = self.greek_history[greek_name]
            
            if len(history) < 10:
                return 0.5
            
            # Calculate percentile rank
            rank = stats.percentileofscore(history, current_value) / 100.0
            
            return rank
            
        except Exception as e:
            logger.error(f"Error calculating percentile rank for {greek_name}: {e}")
            return 0.5
    
    def _calculate_cross_correlations(self, data: pd.DataFrame, greek_name: str) -> Dict[str, float]:
        """Calculate correlations between Greeks"""
        
        try:
            correlations = {}
            
            if f'ce_{greek_name}' not in data.columns:
                return correlations
            
            base_greek = data[f'ce_{greek_name}'] + data[f'pe_{greek_name}']
            
            for other_greek in self.greek_weights.keys():
                if other_greek != greek_name and f'ce_{other_greek}' in data.columns:
                    other_values = data[f'ce_{other_greek}'] + data[f'pe_{other_greek}']
                    
                    if len(base_greek) > 1 and len(other_values) > 1:
                        correlation = base_greek.corr(other_values)
                        correlations[other_greek] = correlation if not np.isnan(correlation) else 0.0
                    else:
                        correlations[other_greek] = 0.0
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating cross-correlations for {greek_name}: {e}")
            return {}
    
    def _calculate_time_decay_impact(self, data: pd.DataFrame, greek_name: str) -> float:
        """Calculate time decay impact, especially for theta"""
        
        try:
            if 'dte' not in data.columns or len(data) == 0:
                return 0.0
            
            avg_dte = data['dte'].mean()
            
            if greek_name == 'theta':
                # Theta accelerates as expiration approaches
                if avg_dte <= 7:
                    impact = 1.0  # High impact
                elif avg_dte <= 30:
                    impact = 0.5 + (30 - avg_dte) / 46  # Medium to high
                else:
                    impact = 0.1  # Low impact
                    
            elif greek_name == 'gamma':
                # Gamma peaks around 7-10 DTE for ATM options
                if 5 <= avg_dte <= 15:
                    impact = 1.0
                elif avg_dte <= 30:
                    impact = 0.7
                else:
                    impact = 0.3
                    
            elif greek_name == 'delta':
                # Delta becomes more sensitive near expiration
                if avg_dte <= 5:
                    impact = 0.8
                else:
                    impact = 0.5
                    
            elif greek_name == 'vega':
                # Vega decreases as expiration approaches
                if avg_dte <= 7:
                    impact = 0.2
                elif avg_dte <= 30:
                    impact = 0.6
                else:
                    impact = 1.0
                    
            else:  # rho
                # Rho generally increases with time to expiration
                impact = min(avg_dte / 365.0, 1.0)
            
            return impact
            
        except Exception as e:
            logger.error(f"Error calculating time decay impact for {greek_name}: {e}")
            return 0.0
    
    def _get_regime_interpretation(self, greek_name: str, 
                                 current_regime: Optional[MarketRegime]) -> Dict[str, float]:
        """Get regime-specific interpretation for Greek"""
        
        try:
            if current_regime is None:
                return {}
            
            regime_config = self.regime_interpretations.get(current_regime.name, {})
            
            return {
                'importance': regime_config.get(f'{greek_name}_importance', 0.5),
                'risk_multiplier': regime_config.get('risk_multiplier', 1.0),
                'regime_factor': regime_config.get(f'{greek_name}_importance', 0.5) * 
                               regime_config.get('risk_multiplier', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting regime interpretation for {greek_name}: {e}")
            return {}
    
    def _calculate_combined_sentiment(self, 
                                    individual_results: Dict[str, GreekAnalysisResult]) -> Tuple[GreekSentimentLevel, float]:
        """Calculate combined sentiment from individual Greeks"""
        
        try:
            weighted_scores = []
            total_weight = 0.0
            confidences = []
            
            for greek_name, result in individual_results.items():
                weight = self.greek_weights.get(greek_name, 0.0)
                score = result.sentiment_level.value
                confidence = result.confidence
                
                weighted_scores.append(score * weight * confidence)
                total_weight += weight * confidence
                confidences.append(confidence)
            
            if total_weight > 0:
                combined_score = sum(weighted_scores) / total_weight
            else:
                combined_score = 0.0
            
            # Convert score back to sentiment level
            if combined_score >= 2.5:
                combined_sentiment = GreekSentimentLevel.EXTREMELY_BULLISH
            elif combined_score >= 1.5:
                combined_sentiment = GreekSentimentLevel.VERY_BULLISH
            elif combined_score >= 0.5:
                combined_sentiment = GreekSentimentLevel.MODERATELY_BULLISH
            elif combined_score >= -0.5:
                combined_sentiment = GreekSentimentLevel.NEUTRAL
            elif combined_score >= -1.5:
                combined_sentiment = GreekSentimentLevel.MODERATELY_BEARISH
            elif combined_score >= -2.5:
                combined_sentiment = GreekSentimentLevel.VERY_BEARISH
            else:
                combined_sentiment = GreekSentimentLevel.EXTREMELY_BEARISH
            
            # Combined confidence
            combined_confidence = np.mean(confidences) if confidences else 0.0
            
            return combined_sentiment, combined_confidence
            
        except Exception as e:
            logger.error(f"Error calculating combined sentiment: {e}")
            return GreekSentimentLevel.NEUTRAL, 0.0
    
    def _calculate_portfolio_greeks(self, data: pd.DataFrame,
                                  individual_results: Dict[str, GreekAnalysisResult]) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        
        try:
            portfolio_greeks = {}
            
            # Aggregate Greeks by position
            for greek_name in self.greek_weights.keys():
                if greek_name in individual_results:
                    result = individual_results[greek_name]
                    portfolio_greeks[f'total_{greek_name}'] = result.volume_weighted_value
                    portfolio_greeks[f'dte_adjusted_{greek_name}'] = result.dte_adjusted_value
            
            # Calculate exposure metrics
            if 'delta' in individual_results:
                delta_result = individual_results['delta']
                portfolio_greeks['delta_exposure'] = abs(delta_result.volume_weighted_value)
                portfolio_greeks['directional_bias'] = np.sign(delta_result.volume_weighted_value)
            
            if 'gamma' in individual_results:
                gamma_result = individual_results['gamma']
                portfolio_greeks['gamma_risk'] = gamma_result.volume_weighted_value
                portfolio_greeks['pin_risk'] = min(gamma_result.volume_weighted_value * 100, 1.0)
            
            if 'theta' in individual_results:
                theta_result = individual_results['theta']
                portfolio_greeks['theta_capture'] = abs(theta_result.volume_weighted_value)
                portfolio_greeks['time_decay_benefit'] = -theta_result.volume_weighted_value  # Negative theta is good for sellers
            
            return portfolio_greeks
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            return {}
    
    def _calculate_risk_metrics(self, individual_results: Dict[str, GreekAnalysisResult],
                              current_regime: Optional[MarketRegime]) -> Dict[str, float]:
        """Calculate risk metrics based on Greeks"""
        
        try:
            risk_metrics = {}
            
            # Delta risk
            if 'delta' in individual_results:
                delta_result = individual_results['delta']
                risk_metrics['directional_risk'] = abs(delta_result.volume_weighted_value)
                risk_metrics['delta_confidence'] = delta_result.confidence
            
            # Gamma risk
            if 'gamma' in individual_results:
                gamma_result = individual_results['gamma']
                risk_metrics['convexity_risk'] = gamma_result.volume_weighted_value
                risk_metrics['gamma_confidence'] = gamma_result.confidence
            
            # Volatility risk (vega)
            if 'vega' in individual_results:
                vega_result = individual_results['vega']
                risk_metrics['volatility_risk'] = abs(vega_result.volume_weighted_value)
                risk_metrics['vega_confidence'] = vega_result.confidence
            
            # Time decay benefit (theta)
            if 'theta' in individual_results:
                theta_result = individual_results['theta']
                risk_metrics['time_decay_benefit'] = abs(theta_result.volume_weighted_value)
                risk_metrics['theta_confidence'] = theta_result.confidence
            
            # Overall risk score
            risk_components = [
                risk_metrics.get('directional_risk', 0) * 0.3,
                risk_metrics.get('convexity_risk', 0) * 0.4,
                risk_metrics.get('volatility_risk', 0) * 0.3
            ]
            risk_metrics['overall_risk_score'] = sum(risk_components)
            
            # Regime adjustment
            if current_regime and current_regime.name in self.regime_interpretations:
                regime_config = self.regime_interpretations[current_regime.name]
                risk_multiplier = regime_config.get('risk_multiplier', 1.0)
                risk_metrics['regime_adjusted_risk'] = risk_metrics['overall_risk_score'] * risk_multiplier
            else:
                risk_metrics['regime_adjusted_risk'] = risk_metrics['overall_risk_score']
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_regime_adjustments(self, individual_results: Dict[str, GreekAnalysisResult],
                                    current_regime: Optional[MarketRegime]) -> Dict[str, float]:
        """Calculate regime-specific adjustments"""
        
        try:
            if current_regime is None:
                return {}
            
            adjustments = {}
            regime_config = self.regime_interpretations.get(current_regime.name, {})
            
            for greek_name, result in individual_results.items():
                importance = regime_config.get(f'{greek_name}_importance', 0.5)
                risk_multiplier = regime_config.get('risk_multiplier', 1.0)
                
                adjustments[f'{greek_name}_importance'] = importance
                adjustments[f'{greek_name}_adjusted_impact'] = result.volume_weighted_value * importance
            
            adjustments['regime_risk_multiplier'] = regime_config.get('risk_multiplier', 1.0)
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating regime adjustments: {e}")
            return {}
    
    def _calculate_feature_scores(self, individual_results: Dict[str, GreekAnalysisResult]) -> Dict[str, float]:
        """Calculate feature scores for 8-regime classification"""
        
        try:
            features = {}
            
            # Individual Greek features
            for greek_name, result in individual_results.items():
                features[f'{greek_name}_sentiment'] = result.sentiment_level.value
                features[f'{greek_name}_confidence'] = result.confidence
                features[f'{greek_name}_percentile'] = result.percentile_rank
                features[f'{greek_name}_time_decay'] = result.time_decay_impact
            
            # Cross-Greek features
            if len(individual_results) >= 2:
                # Delta-Gamma interaction
                if 'delta' in individual_results and 'gamma' in individual_results:
                    delta_val = individual_results['delta'].volume_weighted_value
                    gamma_val = individual_results['gamma'].volume_weighted_value
                    features['delta_gamma_interaction'] = delta_val * gamma_val
                
                # Theta-Vega balance
                if 'theta' in individual_results and 'vega' in individual_results:
                    theta_val = individual_results['theta'].volume_weighted_value
                    vega_val = individual_results['vega'].volume_weighted_value
                    features['theta_vega_balance'] = abs(theta_val) / (abs(vega_val) + 1e-6)
            
            # Overall Greeks sentiment strength
            sentiment_values = [result.sentiment_level.value for result in individual_results.values()]
            if sentiment_values:
                features['greeks_sentiment_strength'] = np.std(sentiment_values)
                features['greeks_sentiment_direction'] = np.mean(sentiment_values)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating feature scores: {e}")
            return {}
    
    def _update_greek_history(self, volume_weighted_greeks: Dict[str, float]):
        """Update historical Greek values for percentile calculations"""
        
        try:
            for greek_name, value in volume_weighted_greeks.items():
                if greek_name in self.greek_history:
                    self.greek_history[greek_name].append(value)
                    
                    # Keep only recent history
                    max_history = self.config.get('percentile_window', 252)
                    if len(self.greek_history[greek_name]) > max_history:
                        self.greek_history[greek_name] = self.greek_history[greek_name][-max_history:]
                        
        except Exception as e:
            logger.error(f"Error updating Greek history: {e}")
    
    def get_greek_interpretation(self, greek_name: str, sentiment_level: GreekSentimentLevel,
                               current_regime: Optional[MarketRegime] = None) -> str:
        """Get human-readable interpretation of Greek analysis"""
        
        try:
            base_interpretations = {
                'delta': {
                    GreekSentimentLevel.EXTREMELY_BULLISH: "Very strong bullish directional bias",
                    GreekSentimentLevel.VERY_BULLISH: "Strong bullish directional bias",
                    GreekSentimentLevel.MODERATELY_BULLISH: "Moderate bullish directional bias",
                    GreekSentimentLevel.NEUTRAL: "Neutral directional exposure",
                    GreekSentimentLevel.MODERATELY_BEARISH: "Moderate bearish directional bias",
                    GreekSentimentLevel.VERY_BEARISH: "Strong bearish directional bias",
                    GreekSentimentLevel.EXTREMELY_BEARISH: "Very strong bearish directional bias"
                },
                'gamma': {
                    GreekSentimentLevel.EXTREMELY_BULLISH: "Extremely high convexity risk",
                    GreekSentimentLevel.VERY_BULLISH: "Very high convexity risk",
                    GreekSentimentLevel.MODERATELY_BULLISH: "Moderate convexity risk",
                    GreekSentimentLevel.NEUTRAL: "Low convexity risk",
                    GreekSentimentLevel.MODERATELY_BEARISH: "Very low convexity risk",
                    GreekSentimentLevel.VERY_BEARISH: "Minimal convexity risk",
                    GreekSentimentLevel.EXTREMELY_BEARISH: "No convexity risk"
                },
                'theta': {
                    GreekSentimentLevel.EXTREMELY_BULLISH: "Maximum time decay benefit",
                    GreekSentimentLevel.VERY_BULLISH: "High time decay benefit",
                    GreekSentimentLevel.MODERATELY_BULLISH: "Moderate time decay benefit",
                    GreekSentimentLevel.NEUTRAL: "Neutral time decay",
                    GreekSentimentLevel.MODERATELY_BEARISH: "Moderate time decay cost",
                    GreekSentimentLevel.VERY_BEARISH: "High time decay cost",
                    GreekSentimentLevel.EXTREMELY_BEARISH: "Maximum time decay cost"
                },
                'vega': {
                    GreekSentimentLevel.EXTREMELY_BULLISH: "Extremely high volatility sensitivity",
                    GreekSentimentLevel.VERY_BULLISH: "Very high volatility sensitivity",
                    GreekSentimentLevel.MODERATELY_BULLISH: "Moderate volatility sensitivity",
                    GreekSentimentLevel.NEUTRAL: "Low volatility sensitivity",
                    GreekSentimentLevel.MODERATELY_BEARISH: "Very low volatility sensitivity",
                    GreekSentimentLevel.VERY_BEARISH: "Minimal volatility sensitivity",
                    GreekSentimentLevel.EXTREMELY_BEARISH: "No volatility sensitivity"
                },
                'rho': {
                    GreekSentimentLevel.EXTREMELY_BULLISH: "Very high interest rate sensitivity",
                    GreekSentimentLevel.VERY_BULLISH: "High interest rate sensitivity",
                    GreekSentimentLevel.MODERATELY_BULLISH: "Moderate interest rate sensitivity",
                    GreekSentimentLevel.NEUTRAL: "Low interest rate sensitivity",
                    GreekSentimentLevel.MODERATELY_BEARISH: "Moderate negative rate sensitivity",
                    GreekSentimentLevel.VERY_BEARISH: "High negative rate sensitivity",
                    GreekSentimentLevel.EXTREMELY_BEARISH: "Very high negative rate sensitivity"
                }
            }
            
            base_interpretation = base_interpretations.get(greek_name, {}).get(
                sentiment_level, "Unknown sentiment level"
            )
            
            if current_regime:
                regime_context = f" (in {current_regime.value} regime)"
                return base_interpretation + regime_context
            
            return base_interpretation
            
        except Exception as e:
            logger.error(f"Error getting Greek interpretation: {e}")
            return "Unable to interpret"

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = EnhancedGreeksSentiment8RegimeSystem()
    
    # Example analysis (would normally use real data)
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='5min'),
        'dte': np.random.randint(1, 30, 100),
        'ce_delta': np.random.normal(0.5, 0.2, 100),
        'pe_delta': np.random.normal(-0.5, 0.2, 100),
        'ce_gamma': np.random.normal(0.05, 0.02, 100),
        'pe_gamma': np.random.normal(0.05, 0.02, 100),
        'ce_theta': np.random.normal(-2.0, 0.5, 100),
        'pe_theta': np.random.normal(-2.0, 0.5, 100),
        'ce_vega': np.random.normal(10.0, 3.0, 100),
        'pe_vega': np.random.normal(10.0, 3.0, 100),
        'ce_rho': np.random.normal(2.0, 1.0, 100),
        'pe_rho': np.random.normal(-2.0, 1.0, 100),
        'ce_volume': np.random.randint(100, 1000, 100),
        'pe_volume': np.random.randint(100, 1000, 100),
        'ce_iv': np.random.normal(0.25, 0.05, 100),
        'pe_iv': np.random.normal(0.27, 0.05, 100)
    })
    
    try:
        result = system.analyze_greeks_sentiment(sample_data, MarketRegime.LVLD)
        print(f"Greeks analysis completed successfully")
        print(f"Combined sentiment: {result.combined_sentiment.name}")
        print(f"Combined confidence: {result.combined_confidence:.3f}")
        print(f"Individual Greeks analyzed: {len(result.individual_greeks)}")
        
        # Print individual Greek interpretations
        for greek_name, greek_result in result.individual_greeks.items():
            interpretation = system.get_greek_interpretation(
                greek_name, greek_result.sentiment_level, MarketRegime.LVLD
            )
            print(f"{greek_name.upper()}: {interpretation}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")