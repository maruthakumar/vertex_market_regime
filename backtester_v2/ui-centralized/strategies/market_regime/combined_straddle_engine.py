#!/usr/bin/env python3
"""
Combined Straddle Engine - Industry-Standard Weighted Combination
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides industry-standard weighted combination of ATM, ITM1, and OTM1 straddles
with dynamic DTE and VIX adjustments, plus independent technical analysis:
- Industry-standard base weights: ATM 50%, ITM1 30%, OTM1 20%
- Dynamic DTE-based adjustments for 0-4 DTE focus
- Dynamic VIX-based adjustments for volatility conditions
- Independent EMA/VWAP/Pivot analysis on combined straddle
- Multi-timeframe rolling windows (3, 5, 10, 15 minutes)

Author: The Augster
Date: 2025-06-23
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CombinedStraddleEngine:
    """
    Combined Straddle Engine with industry-standard weighted combination
    
    Provides industry-standard weighted combination of ATM, ITM1, and OTM1 straddles
    with dynamic adjustments and independent technical analysis on the combined result.
    """
    
    def __init__(self, component_config: Dict[str, Any]):
        """Initialize Combined Straddle Engine"""
        self.config = component_config
        self.component_name = "Combined_Straddle"
        self.weight = component_config.get('weight', 0.20)
        self.priority = component_config.get('priority', 'high')
        
        # Industry-standard base weights
        self.base_weights = component_config.get('base_weights', {
            'atm': 0.50,    # 50-60% (highest liquidity and sensitivity)
            'itm1': 0.30,   # 25-30% (directional bias component)
            'otm1': 0.20    # 15-20% (tail risk component)
        })
        
        # Technical indicator parameters
        self.ema_periods = [20, 100, 200]
        self.vwap_periods = [1, 5, 15]  # Days
        self.rolling_windows = component_config.get('rolling_windows', [3, 5, 10, 15])
        
        logger.info(f"Combined Straddle Engine initialized - Weight: {self.weight}, Priority: {self.priority}")
        logger.info(f"Base weights - ATM: {self.base_weights['atm']}, ITM1: {self.base_weights['itm1']}, OTM1: {self.base_weights['otm1']}")
    
    def calculate_industry_standard_combined_straddle(self, atm_straddle: pd.Series,
                                                    itm1_straddle: pd.Series, 
                                                    otm1_straddle: pd.Series,
                                                    current_dte: int, 
                                                    current_vix: float) -> Dict[str, Any]:
        """
        Calculate industry-standard weighted combined straddle with dynamic adjustments
        
        Args:
            atm_straddle: ATM straddle prices
            itm1_straddle: ITM1 straddle prices
            otm1_straddle: OTM1 straddle prices
            current_dte: Current days to expiry
            current_vix: Current VIX level
            
        Returns:
            Combined straddle data with weights and technical analysis
        """
        try:
            logger.debug(f"Calculating combined straddle - DTE: {current_dte}, VIX: {current_vix}")
            
            # Calculate dynamic adjustments
            dte_adjustments = self._calculate_dte_adjustments(current_dte)
            vix_adjustments = self._calculate_vix_adjustments(current_vix)
            
            # Calculate adjusted weights
            adjusted_weights = self._calculate_adjusted_weights(dte_adjustments, vix_adjustments)
            
            # Calculate combined straddle
            combined_straddle = (
                atm_straddle * adjusted_weights['atm'] +
                itm1_straddle * adjusted_weights['itm1'] +
                otm1_straddle * adjusted_weights['otm1']
            )
            
            # Calculate technical analysis on combined straddle
            technical_analysis = self._calculate_combined_technical_analysis(combined_straddle)
            
            # Calculate combination metrics
            combination_metrics = self._calculate_combination_metrics(
                atm_straddle, itm1_straddle, otm1_straddle, combined_straddle, adjusted_weights
            )
            
            return {
                'combined_straddle_prices': combined_straddle,
                'weights': {
                    'base_weights': self.base_weights,
                    'dte_adjustments': dte_adjustments,
                    'vix_adjustments': vix_adjustments,
                    'final_weights': adjusted_weights
                },
                'technical_analysis': technical_analysis,
                'combination_metrics': combination_metrics,
                'market_conditions': {
                    'current_dte': current_dte,
                    'current_vix': current_vix,
                    'dte_category': self._categorize_dte(current_dte),
                    'vix_category': self._categorize_vix(current_vix)
                },
                'calculation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating combined straddle: {e}")
            return self._get_default_combined_results()
    
    def _calculate_dte_adjustments(self, current_dte: int) -> Dict[str, float]:
        """Calculate DTE-based weight adjustments"""
        try:
            if current_dte <= 1:
                # 0-1 DTE: Emphasize ATM (higher gamma sensitivity)
                return {'atm': 1.15, 'itm1': 0.95, 'otm1': 0.90}
            elif 2 <= current_dte <= 4:
                # 2-4 DTE: Balanced approach
                return {'atm': 1.00, 'itm1': 1.00, 'otm1': 1.00}
            elif 5 <= current_dte <= 7:
                # 5-7 DTE: Slightly emphasize ITM1/OTM1
                return {'atm': 0.95, 'itm1': 1.05, 'otm1': 1.02}
            else:
                # 8+ DTE: Emphasize ITM1/OTM1 (higher time value)
                return {'atm': 0.90, 'itm1': 1.10, 'otm1': 1.05}
                
        except Exception as e:
            logger.error(f"Error calculating DTE adjustments: {e}")
            return {'atm': 1.00, 'itm1': 1.00, 'otm1': 1.00}
    
    def _calculate_vix_adjustments(self, current_vix: float) -> Dict[str, float]:
        """Calculate VIX-based weight adjustments"""
        try:
            if current_vix > 30:
                # Very High VIX: Emphasize OTM for tail risk
                return {'atm': 0.90, 'itm1': 0.95, 'otm1': 1.30}
            elif current_vix > 25:
                # High VIX: Emphasize OTM for tail risk
                return {'atm': 0.95, 'itm1': 1.00, 'otm1': 1.20}
            elif current_vix < 12:
                # Very Low VIX: Emphasize ATM for precision
                return {'atm': 1.15, 'itm1': 1.00, 'otm1': 0.85}
            elif current_vix < 15:
                # Low VIX: Emphasize ATM for precision
                return {'atm': 1.10, 'itm1': 1.00, 'otm1': 0.90}
            else:
                # Normal VIX (15-25): Balanced approach
                return {'atm': 1.00, 'itm1': 1.00, 'otm1': 1.00}
                
        except Exception as e:
            logger.error(f"Error calculating VIX adjustments: {e}")
            return {'atm': 1.00, 'itm1': 1.00, 'otm1': 1.00}
    
    def _calculate_adjusted_weights(self, dte_adjustments: Dict[str, float], 
                                  vix_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Calculate final adjusted weights"""
        try:
            adjusted_weights = {}
            
            # Apply adjustments to base weights
            for component in ['atm', 'itm1', 'otm1']:
                adjusted_weights[component] = (
                    self.base_weights[component] * 
                    dte_adjustments[component] * 
                    vix_adjustments[component]
                )
            
            # Renormalize weights to sum to 1.0
            total_weight = sum(adjusted_weights.values())
            normalized_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Error calculating adjusted weights: {e}")
            return self.base_weights
    
    def _calculate_combined_technical_analysis(self, combined_straddle: pd.Series) -> Dict[str, Any]:
        """Calculate independent technical analysis on combined straddle"""
        try:
            technical_results = {}
            
            # EMA Analysis
            ema_results = {}
            for period in self.ema_periods:
                ema_key = f'ema_{period}'
                ema_values = combined_straddle.ewm(span=period).mean()
                ema_results[ema_key] = ema_values
                ema_results[f'{ema_key}_position'] = (combined_straddle / ema_values - 1).fillna(0)
                ema_results[f'above_{ema_key}'] = (combined_straddle > ema_values).astype(int)
            
            # EMA alignment
            ema_20 = ema_results['ema_20']
            ema_100 = ema_results['ema_100']
            ema_200 = ema_results['ema_200']
            
            ema_results['ema_alignment_bullish'] = (
                (ema_20 > ema_100) & (ema_100 > ema_200)
            ).astype(int)
            ema_results['ema_alignment_bearish'] = (
                (ema_20 < ema_100) & (ema_100 < ema_200)
            ).astype(int)
            
            technical_results['ema_analysis'] = ema_results
            
            # VWAP Analysis (simplified without volume)
            vwap_results = {}
            vwap_current = combined_straddle.rolling(window=20).mean()  # Simplified VWAP
            vwap_results['vwap_current'] = vwap_current
            vwap_results['vwap_position'] = (combined_straddle / vwap_current - 1).fillna(0)
            vwap_results['above_vwap_current'] = (combined_straddle > vwap_current).astype(int)
            
            technical_results['vwap_analysis'] = vwap_results
            
            # Pivot Analysis
            pivot_results = {}
            window_size = 75
            daily_high = combined_straddle.rolling(window=window_size).max()
            daily_low = combined_straddle.rolling(window=window_size).min()
            pivot_current = (daily_high + daily_low + combined_straddle) / 3
            
            pivot_results['pivot_current'] = pivot_current
            pivot_results['pivot_position'] = (combined_straddle / pivot_current - 1).fillna(0)
            pivot_results['above_pivot_current'] = (combined_straddle > pivot_current).astype(int)
            
            technical_results['pivot_analysis'] = pivot_results
            
            # Combined-specific indicators
            combined_results = {}
            combined_results['volatility'] = combined_straddle.rolling(window=20).std().fillna(0)
            combined_results['momentum_1'] = combined_straddle.pct_change(1).fillna(0)
            combined_results['momentum_5'] = combined_straddle.pct_change(5).fillna(0)
            combined_results['trend_strength'] = abs(
                combined_straddle.rolling(window=10).mean() / combined_straddle.rolling(window=30).mean() - 1
            ).fillna(0)
            
            technical_results['combined_specific_analysis'] = combined_results
            
            return technical_results
            
        except Exception as e:
            logger.error(f"Error calculating combined technical analysis: {e}")
            return {}
    
    def _calculate_combination_metrics(self, atm_straddle: pd.Series, itm1_straddle: pd.Series,
                                     otm1_straddle: pd.Series, combined_straddle: pd.Series,
                                     weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate metrics about the combination quality"""
        try:
            metrics = {}
            
            # Component correlations
            metrics['correlations'] = {
                'atm_itm1': atm_straddle.corr(itm1_straddle) if len(atm_straddle) > 1 else 0.0,
                'atm_otm1': atm_straddle.corr(otm1_straddle) if len(atm_straddle) > 1 else 0.0,
                'itm1_otm1': itm1_straddle.corr(otm1_straddle) if len(itm1_straddle) > 1 else 0.0
            }
            
            # Weight distribution
            metrics['weight_distribution'] = {
                'weight_entropy': -sum(w * np.log(w) for w in weights.values() if w > 0),
                'weight_concentration': max(weights.values()),
                'weight_balance': 1 - abs(max(weights.values()) - min(weights.values()))
            }
            
            # Combination effectiveness
            individual_volatilities = [
                atm_straddle.std() if len(atm_straddle) > 0 else 0,
                itm1_straddle.std() if len(itm1_straddle) > 0 else 0,
                otm1_straddle.std() if len(otm1_straddle) > 0 else 0
            ]
            combined_volatility = combined_straddle.std() if len(combined_straddle) > 0 else 0
            
            metrics['combination_effectiveness'] = {
                'volatility_reduction': 1 - (combined_volatility / np.mean(individual_volatilities)) if np.mean(individual_volatilities) > 0 else 0,
                'diversification_ratio': combined_volatility / max(individual_volatilities) if max(individual_volatilities) > 0 else 1,
                'signal_to_noise': abs(combined_straddle.mean()) / combined_volatility if combined_volatility > 0 else 0
            }
            
            # Component contributions
            metrics['component_contributions'] = {
                'atm_contribution': weights['atm'] * (atm_straddle.mean() if len(atm_straddle) > 0 else 0),
                'itm1_contribution': weights['itm1'] * (itm1_straddle.mean() if len(itm1_straddle) > 0 else 0),
                'otm1_contribution': weights['otm1'] * (otm1_straddle.mean() if len(otm1_straddle) > 0 else 0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating combination metrics: {e}")
            return {}
    
    def _categorize_dte(self, dte: int) -> str:
        """Categorize DTE for analysis"""
        if dte <= 1:
            return "Ultra_Short_Term"
        elif dte <= 4:
            return "Short_Term"
        elif dte <= 7:
            return "Medium_Term"
        else:
            return "Long_Term"
    
    def _categorize_vix(self, vix: float) -> str:
        """Categorize VIX for analysis"""
        if vix < 12:
            return "Very_Low_Volatility"
        elif vix < 15:
            return "Low_Volatility"
        elif vix < 20:
            return "Normal_Volatility"
        elif vix < 25:
            return "Elevated_Volatility"
        elif vix < 30:
            return "High_Volatility"
        else:
            return "Very_High_Volatility"
    
    def calculate_independent_technical_analysis(self, combined_straddle_prices: pd.Series,
                                               timeframe: str = '5min') -> Dict[str, Any]:
        """
        Calculate independent technical analysis for Combined Straddle
        
        Args:
            combined_straddle_prices: Combined straddle prices
            timeframe: Analysis timeframe
            
        Returns:
            Complete technical analysis results
        """
        try:
            logger.debug(f"Calculating Combined Straddle technical analysis for {timeframe}")
            
            if combined_straddle_prices.empty:
                logger.warning("Empty combined straddle price data")
                return self._get_default_results()
            
            # Calculate technical analysis
            technical_analysis = self._calculate_combined_technical_analysis(combined_straddle_prices)
            
            # Calculate summary metrics
            summary_metrics = self._calculate_summary_metrics(technical_analysis)
            
            # Combine results
            technical_results = {
                'component': self.component_name,
                'timeframe': timeframe,
                'weight': self.weight,
                'technical_analysis': technical_analysis,
                'summary_metrics': summary_metrics,
                'calculation_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(combined_straddle_prices)
            }
            
            return technical_results
            
        except Exception as e:
            logger.error(f"Error calculating Combined Straddle technical analysis: {e}")
            return self._get_default_results()
    
    def _calculate_summary_metrics(self, technical_analysis: Dict) -> Dict[str, float]:
        """Calculate summary metrics for Combined Straddle analysis"""
        try:
            summary = {}
            
            # Overall bullish/bearish bias
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # EMA signals
            ema_analysis = technical_analysis.get('ema_analysis', {})
            if 'ema_alignment_bullish' in ema_analysis:
                bullish_signals += ema_analysis['ema_alignment_bullish'].iloc[-1] if len(ema_analysis['ema_alignment_bullish']) > 0 else 0
                bearish_signals += ema_analysis['ema_alignment_bearish'].iloc[-1] if len(ema_analysis['ema_alignment_bearish']) > 0 else 0
                total_signals += 1
            
            # VWAP signals
            vwap_analysis = technical_analysis.get('vwap_analysis', {})
            if 'above_vwap_current' in vwap_analysis:
                bullish_signals += vwap_analysis['above_vwap_current'].iloc[-1] if len(vwap_analysis['above_vwap_current']) > 0 else 0
                bearish_signals += (1 - vwap_analysis['above_vwap_current'].iloc[-1]) if len(vwap_analysis['above_vwap_current']) > 0 else 0
                total_signals += 1
            
            # Pivot signals
            pivot_analysis = technical_analysis.get('pivot_analysis', {})
            if 'above_pivot_current' in pivot_analysis:
                bullish_signals += pivot_analysis['above_pivot_current'].iloc[-1] if len(pivot_analysis['above_pivot_current']) > 0 else 0
                bearish_signals += (1 - pivot_analysis['above_pivot_current'].iloc[-1]) if len(pivot_analysis['above_pivot_current']) > 0 else 0
                total_signals += 1
            
            # Calculate summary scores
            summary['bullish_score'] = bullish_signals / total_signals if total_signals > 0 else 0.5
            summary['bearish_score'] = bearish_signals / total_signals if total_signals > 0 else 0.5
            summary['neutral_score'] = 1 - summary['bullish_score'] - summary['bearish_score']
            
            # Overall signal strength
            summary['signal_strength'] = abs(summary['bullish_score'] - summary['bearish_score'])
            summary['signal_direction'] = 1 if summary['bullish_score'] > summary['bearish_score'] else -1
            
            # Confidence based on signal alignment
            summary['confidence'] = summary['signal_strength']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating Combined summary metrics: {e}")
            return {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0}
    
    def _assess_data_quality(self, prices: pd.Series) -> Dict[str, Any]:
        """Assess data quality for Combined Straddle analysis"""
        try:
            quality_metrics = {
                'data_points': len(prices),
                'missing_values': prices.isna().sum(),
                'zero_values': (prices == 0).sum(),
                'data_completeness': (len(prices) - prices.isna().sum()) / len(prices) if len(prices) > 0 else 0,
                'price_range': {
                    'min': float(prices.min()) if len(prices) > 0 else 0,
                    'max': float(prices.max()) if len(prices) > 0 else 0,
                    'mean': float(prices.mean()) if len(prices) > 0 else 0,
                    'std': float(prices.std()) if len(prices) > 0 else 0
                },
                'quality_score': 1.0 if len(prices) > 0 and prices.isna().sum() == 0 else 0.5
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing Combined data quality: {e}")
            return {'quality_score': 0.0}
    
    def _get_default_combined_results(self) -> Dict[str, Any]:
        """Get default combined straddle results when calculation fails"""
        return {
            'combined_straddle_prices': pd.Series([]),
            'weights': {
                'base_weights': self.base_weights,
                'dte_adjustments': {'atm': 1.0, 'itm1': 1.0, 'otm1': 1.0},
                'vix_adjustments': {'atm': 1.0, 'itm1': 1.0, 'otm1': 1.0},
                'final_weights': self.base_weights
            },
            'technical_analysis': {},
            'combination_metrics': {},
            'market_conditions': {},
            'calculation_timestamp': datetime.now().isoformat(),
            'error': 'Combined straddle calculation failed'
        }
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results when calculation fails"""
        return {
            'component': self.component_name,
            'timeframe': 'unknown',
            'weight': self.weight,
            'technical_analysis': {},
            'summary_metrics': {'bullish_score': 0.5, 'bearish_score': 0.5, 'confidence': 0.0},
            'calculation_timestamp': datetime.now().isoformat(),
            'data_quality': {'quality_score': 0.0},
            'error': 'Combined calculation failed'
        }
