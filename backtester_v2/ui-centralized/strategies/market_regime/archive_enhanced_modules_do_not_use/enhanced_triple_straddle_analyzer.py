"""
Enhanced Triple Straddle Analyzer

This module provides comprehensive analysis of ATM, ITM1, and OTM1 straddles
with independent calculations and correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedTripleStraddleAnalyzer:
    """
    Enhanced Triple Straddle Analyzer for comprehensive options analysis
    
    Features:
    - Independent ATM, ITM1, OTM1 straddle calculations
    - Cross-straddle correlation analysis
    - Dynamic greek calculations
    - Real-time performance tracking
    """
    
    def __init__(self):
        """Initialize the triple straddle analyzer"""
        self.straddle_types = ['ATM', 'ITM1', 'OTM1']
        self.timeframes = [3, 5, 10, 15]  # minutes
        
        # Component weights
        self.straddle_weights = {
            'ATM': 0.40,
            'ITM1': 0.35,
            'OTM1': 0.25
        }
        
        # Greek weights for sentiment
        self.greek_weights = {
            'delta': 0.25,
            'gamma': 0.20,
            'theta': 0.20,
            'vega': 0.20,
            'rho': 0.15
        }
        
        logger.info("Enhanced Triple Straddle Analyzer initialized")
    
    def calculate_straddle_metrics(self, market_data: pd.DataFrame, straddle_type: str) -> Dict[str, Any]:
        """Calculate metrics for a specific straddle type"""
        try:
            metrics = {
                'price': 0.0,
                'volume': 0.0,
                'oi': 0.0,
                'iv': 0.0,
                'greeks': {},
                'momentum': 0.0,
                'sentiment': 0.0
            }
            
            # Filter data for straddle type
            ce_col_prefix = f'{straddle_type}_CE_'
            pe_col_prefix = f'{straddle_type}_PE_'
            
            # Calculate straddle price
            if f'{ce_col_prefix}ltp' in market_data.columns and f'{pe_col_prefix}ltp' in market_data.columns:
                metrics['price'] = market_data[f'{ce_col_prefix}ltp'].iloc[-1] + market_data[f'{pe_col_prefix}ltp'].iloc[-1]
            
            # Calculate volume
            if f'{ce_col_prefix}volume' in market_data.columns and f'{pe_col_prefix}volume' in market_data.columns:
                metrics['volume'] = market_data[f'{ce_col_prefix}volume'].iloc[-1] + market_data[f'{pe_col_prefix}volume'].iloc[-1]
            
            # Calculate OI
            if f'{ce_col_prefix}oi' in market_data.columns and f'{pe_col_prefix}oi' in market_data.columns:
                metrics['oi'] = market_data[f'{ce_col_prefix}oi'].iloc[-1] + market_data[f'{pe_col_prefix}oi'].iloc[-1]
            
            # Calculate IV
            if f'{ce_col_prefix}iv' in market_data.columns and f'{pe_col_prefix}iv' in market_data.columns:
                ce_iv = market_data[f'{ce_col_prefix}iv'].iloc[-1]
                pe_iv = market_data[f'{pe_col_prefix}iv'].iloc[-1]
                metrics['iv'] = (ce_iv + pe_iv) / 2
            
            # Calculate Greeks
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                ce_greek_col = f'{ce_col_prefix}{greek}'
                pe_greek_col = f'{pe_col_prefix}{greek}'
                
                if ce_greek_col in market_data.columns and pe_greek_col in market_data.columns:
                    ce_val = market_data[ce_greek_col].iloc[-1]
                    pe_val = market_data[pe_greek_col].iloc[-1]
                    metrics['greeks'][greek] = ce_val + pe_val
            
            # Calculate momentum (rate of change)
            if len(market_data) > 1:
                current_price = metrics['price']
                prev_price = market_data[f'{ce_col_prefix}ltp'].iloc[-2] + market_data[f'{pe_col_prefix}ltp'].iloc[-2]
                metrics['momentum'] = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            
            # Calculate sentiment based on greeks
            metrics['sentiment'] = self._calculate_greek_sentiment(metrics['greeks'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating {straddle_type} straddle metrics: {e}")
            return {}
    
    def _calculate_greek_sentiment(self, greeks: Dict[str, float]) -> float:
        """Calculate sentiment score from greeks"""
        if not greeks:
            return 0.0
        
        sentiment = 0.0
        total_weight = 0.0
        
        for greek, value in greeks.items():
            if greek in self.greek_weights:
                weight = self.greek_weights[greek]
                
                # Normalize greek values
                if greek == 'delta':
                    # Delta ranges from -1 to 1
                    normalized = value
                elif greek == 'gamma':
                    # Gamma is always positive, higher means more volatile
                    normalized = min(value / 0.1, 1.0)  # Normalize to 0-1
                elif greek == 'theta':
                    # Theta is usually negative, more negative means faster decay
                    normalized = max(value / -50, -1.0)  # Normalize to -1 to 0
                elif greek == 'vega':
                    # Vega is positive, higher means more IV sensitive
                    normalized = min(value / 100, 1.0)  # Normalize to 0-1
                elif greek == 'rho':
                    # Rho impact is usually small
                    normalized = value / 100  # Normalize
                else:
                    normalized = 0
                
                sentiment += normalized * weight
                total_weight += weight
        
        return sentiment / total_weight if total_weight > 0 else 0.0
    
    def calculate_triple_straddle_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive triple straddle analysis"""
        try:
            analysis = {
                'timestamp': datetime.now(),
                'straddles': {},
                'correlations': {},
                'combined_metrics': {},
                'regime_signals': {}
            }
            
            # Calculate metrics for each straddle
            for straddle_type in self.straddle_types:
                analysis['straddles'][straddle_type] = self.calculate_straddle_metrics(
                    market_data, straddle_type
                )
            
            # Calculate cross-straddle correlations
            analysis['correlations'] = self._calculate_straddle_correlations(market_data)
            
            # Calculate combined metrics
            analysis['combined_metrics'] = self._calculate_combined_metrics(
                analysis['straddles']
            )
            
            # Generate regime signals
            analysis['regime_signals'] = self._generate_regime_signals(
                analysis['straddles'],
                analysis['correlations'],
                analysis['combined_metrics']
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in triple straddle analysis: {e}")
            return {}
    
    def _calculate_straddle_correlations(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between straddles"""
        correlations = {}
        
        try:
            # Get price series for each straddle
            price_series = {}
            for straddle_type in self.straddle_types:
                ce_col = f'{straddle_type}_CE_ltp'
                pe_col = f'{straddle_type}_PE_ltp'
                
                if ce_col in market_data.columns and pe_col in market_data.columns:
                    price_series[straddle_type] = market_data[ce_col] + market_data[pe_col]
            
            # Calculate pairwise correlations
            for i, straddle1 in enumerate(self.straddle_types):
                for j, straddle2 in enumerate(self.straddle_types):
                    if i < j and straddle1 in price_series and straddle2 in price_series:
                        corr = price_series[straddle1].corr(price_series[straddle2])
                        correlations[f'{straddle1}_{straddle2}'] = corr if not np.isnan(corr) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating straddle correlations: {e}")
        
        return correlations
    
    def _calculate_combined_metrics(self, straddles: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate combined metrics from all straddles"""
        combined = {
            'weighted_price': 0.0,
            'total_volume': 0.0,
            'total_oi': 0.0,
            'avg_iv': 0.0,
            'weighted_sentiment': 0.0,
            'weighted_momentum': 0.0
        }
        
        try:
            total_weight = 0.0
            
            for straddle_type, metrics in straddles.items():
                if straddle_type in self.straddle_weights and metrics:
                    weight = self.straddle_weights[straddle_type]
                    
                    combined['weighted_price'] += metrics.get('price', 0) * weight
                    combined['total_volume'] += metrics.get('volume', 0)
                    combined['total_oi'] += metrics.get('oi', 0)
                    combined['avg_iv'] += metrics.get('iv', 0) * weight
                    combined['weighted_sentiment'] += metrics.get('sentiment', 0) * weight
                    combined['weighted_momentum'] += metrics.get('momentum', 0) * weight
                    
                    total_weight += weight
            
            # Normalize weighted values
            if total_weight > 0:
                for key in ['weighted_price', 'avg_iv', 'weighted_sentiment', 'weighted_momentum']:
                    combined[key] /= total_weight
            
        except Exception as e:
            logger.error(f"Error calculating combined metrics: {e}")
        
        return combined
    
    def _generate_regime_signals(self, straddles: Dict, correlations: Dict, combined: Dict) -> Dict[str, Any]:
        """Generate regime signals from triple straddle analysis"""
        signals = {
            'volatility_regime': 'NORMAL',
            'trend_strength': 0.0,
            'regime_confidence': 0.0,
            'recommendations': []
        }
        
        try:
            # Determine volatility regime from IV and correlations
            avg_iv = combined.get('avg_iv', 0)
            
            if avg_iv > 25:
                signals['volatility_regime'] = 'HIGH_VOLATILITY'
            elif avg_iv > 18:
                signals['volatility_regime'] = 'ELEVATED_VOLATILITY'
            elif avg_iv < 12:
                signals['volatility_regime'] = 'LOW_VOLATILITY'
            
            # Calculate trend strength from momentum and sentiment
            signals['trend_strength'] = (
                combined.get('weighted_momentum', 0) * 0.6 +
                combined.get('weighted_sentiment', 0) * 0.4
            )
            
            # Calculate regime confidence
            # Higher correlation between straddles = higher confidence
            avg_correlation = np.mean(list(correlations.values())) if correlations else 0.5
            signals['regime_confidence'] = min(avg_correlation * 1.2, 1.0)
            
            # Generate recommendations
            if signals['volatility_regime'] == 'HIGH_VOLATILITY':
                signals['recommendations'].append('Consider hedging strategies')
                signals['recommendations'].append('Reduce position sizes')
            
            if abs(signals['trend_strength']) > 0.3:
                direction = 'bullish' if signals['trend_strength'] > 0 else 'bearish'
                signals['recommendations'].append(f'Strong {direction} trend detected')
            
        except Exception as e:
            logger.error(f"Error generating regime signals: {e}")
        
        return signals
    
    def analyze_multi_timeframe(self, market_data: pd.DataFrame, timeframes: List[int] = None) -> Dict[str, Any]:
        """Analyze straddles across multiple timeframes"""
        if timeframes is None:
            timeframes = self.timeframes
        
        mtf_analysis = {
            'timeframes': {},
            'consensus': {}
        }
        
        try:
            for tf in timeframes:
                # Get data for specific timeframe
                tf_data = market_data.tail(tf) if len(market_data) >= tf else market_data
                
                # Analyze for this timeframe
                mtf_analysis['timeframes'][f'{tf}min'] = self.calculate_triple_straddle_analysis(tf_data)
            
            # Calculate consensus across timeframes
            mtf_analysis['consensus'] = self._calculate_timeframe_consensus(
                mtf_analysis['timeframes']
            )
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
        
        return mtf_analysis
    
    def _calculate_timeframe_consensus(self, timeframe_data: Dict) -> Dict[str, Any]:
        """Calculate consensus signals across timeframes"""
        consensus = {
            'volatility_regime': 'MIXED',
            'trend_direction': 'NEUTRAL',
            'confidence': 0.0
        }
        
        try:
            # Collect signals from all timeframes
            volatility_regimes = []
            trend_strengths = []
            confidences = []
            
            for tf, data in timeframe_data.items():
                if 'regime_signals' in data:
                    signals = data['regime_signals']
                    volatility_regimes.append(signals.get('volatility_regime', 'NORMAL'))
                    trend_strengths.append(signals.get('trend_strength', 0))
                    confidences.append(signals.get('regime_confidence', 0))
            
            # Determine consensus volatility regime
            if volatility_regimes:
                from collections import Counter
                regime_counts = Counter(volatility_regimes)
                consensus['volatility_regime'] = regime_counts.most_common(1)[0][0]
            
            # Determine trend direction
            if trend_strengths:
                avg_trend = np.mean(trend_strengths)
                if avg_trend > 0.2:
                    consensus['trend_direction'] = 'BULLISH'
                elif avg_trend < -0.2:
                    consensus['trend_direction'] = 'BEARISH'
            
            # Calculate overall confidence
            if confidences:
                consensus['confidence'] = np.mean(confidences)
            
        except Exception as e:
            logger.error(f"Error calculating timeframe consensus: {e}")
        
        return consensus