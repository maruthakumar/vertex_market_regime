"""
Option RSI - Relative Strength Index for Options
===============================================

Calculates RSI on option prices, implied volatility, and Greeks
to identify overbought/oversold conditions in option markets.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OptionRSI:
    """
    Option-specific RSI implementation
    
    Features:
    - Price-based RSI for options
    - IV-based RSI for volatility regime
    - Greek-based RSI for sensitivity analysis
    - Put-Call RSI divergence detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Option RSI calculator"""
        # RSI periods
        self.price_rsi_period = config.get('price_rsi_period', 14)
        self.iv_rsi_period = config.get('iv_rsi_period', 14)
        self.greek_rsi_period = config.get('greek_rsi_period', 14)
        
        # Thresholds
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.extreme_overbought = config.get('extreme_overbought', 80)
        self.extreme_oversold = config.get('extreme_oversold', 20)
        
        # Advanced settings
        self.enable_divergence_detection = config.get('enable_divergence_detection', True)
        self.enable_multi_timeframe = config.get('enable_multi_timeframe', True)
        self.smoothing_factor = config.get('smoothing_factor', 3)
        
        # History tracking
        self.rsi_history = {
            'price': [],
            'iv': [],
            'greek': [],
            'composite': []
        }
        
        logger.info(f"OptionRSI initialized: price_period={self.price_rsi_period}")
    
    def calculate_option_rsi(self, 
                           option_data: pd.DataFrame,
                           option_type: str = 'both') -> Dict[str, Any]:
        """
        Calculate comprehensive option RSI
        
        Args:
            option_data: DataFrame with option prices, IV, Greeks
            option_type: 'CE', 'PE', or 'both'
            
        Returns:
            Dict with RSI values and signals
        """
        try:
            results = {
                'price_rsi': {},
                'iv_rsi': {},
                'greek_rsi': {},
                'composite_rsi': {},
                'signals': {},
                'divergences': [],
                'regime': None
            }
            
            # Calculate for each option type
            if option_type in ['CE', 'both']:
                ce_data = option_data[option_data['option_type'] == 'CE']
                if not ce_data.empty:
                    results['price_rsi']['CE'] = self._calculate_price_rsi(ce_data)
                    results['iv_rsi']['CE'] = self._calculate_iv_rsi(ce_data)
                    results['greek_rsi']['CE'] = self._calculate_greek_rsi(ce_data)
            
            if option_type in ['PE', 'both']:
                pe_data = option_data[option_data['option_type'] == 'PE']
                if not pe_data.empty:
                    results['price_rsi']['PE'] = self._calculate_price_rsi(pe_data)
                    results['iv_rsi']['PE'] = self._calculate_iv_rsi(pe_data)
                    results['greek_rsi']['PE'] = self._calculate_greek_rsi(pe_data)
            
            # Calculate composite RSI
            results['composite_rsi'] = self._calculate_composite_rsi(results)
            
            # Generate signals
            results['signals'] = self._generate_rsi_signals(results)
            
            # Detect divergences
            if self.enable_divergence_detection:
                results['divergences'] = self._detect_rsi_divergences(
                    option_data, results
                )
            
            # Classify regime
            results['regime'] = self._classify_rsi_regime(results)
            
            # Update history
            self._update_rsi_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating option RSI: {e}")
            return self._get_default_results()
    
    def _calculate_price_rsi(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate RSI on option prices"""
        try:
            # Aggregate prices by timestamp
            price_series = data.groupby('timestamp')['price'].mean()
            
            if len(price_series) < self.price_rsi_period:
                return {'value': 50.0, 'status': 'insufficient_data'}
            
            # Calculate price changes
            delta = price_series.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=self.price_rsi_period).mean()
            avg_loss = losses.rolling(window=self.price_rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get latest RSI value
            latest_rsi = rsi.iloc[-1]
            
            # Apply smoothing if enabled
            if self.smoothing_factor > 1:
                latest_rsi = rsi.rolling(window=self.smoothing_factor).mean().iloc[-1]
            
            return {
                'value': float(latest_rsi),
                'trend': self._calculate_rsi_trend(rsi),
                'strength': self._calculate_rsi_strength(latest_rsi),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating price RSI: {e}")
            return {'value': 50.0, 'status': 'error'}
    
    def _calculate_iv_rsi(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate RSI on implied volatility"""
        try:
            # Aggregate IV by timestamp
            iv_series = data.groupby('timestamp')['iv'].mean()
            
            if len(iv_series) < self.iv_rsi_period:
                return {'value': 50.0, 'status': 'insufficient_data'}
            
            # Calculate IV changes
            delta = iv_series.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=self.iv_rsi_period).mean()
            avg_loss = losses.rolling(window=self.iv_rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get latest RSI value
            latest_rsi = rsi.iloc[-1]
            
            return {
                'value': float(latest_rsi),
                'volatility_regime': self._classify_iv_regime(latest_rsi),
                'iv_trend': self._calculate_rsi_trend(rsi),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV RSI: {e}")
            return {'value': 50.0, 'status': 'error'}
    
    def _calculate_greek_rsi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI on Greeks"""
        try:
            results = {}
            
            # Calculate RSI for each Greek
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                if greek in data.columns:
                    greek_series = data.groupby('timestamp')[greek].mean()
                    
                    if len(greek_series) >= self.greek_rsi_period:
                        # Calculate changes
                        delta = greek_series.diff()
                        
                        # Separate gains and losses
                        gains = delta.where(delta > 0, 0)
                        losses = -delta.where(delta < 0, 0)
                        
                        # Calculate average gains and losses
                        avg_gain = gains.rolling(window=self.greek_rsi_period).mean()
                        avg_loss = losses.rolling(window=self.greek_rsi_period).mean()
                        
                        # Calculate RS and RSI
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        results[greek] = float(rsi.iloc[-1])
            
            # Calculate composite Greek RSI
            if results:
                composite_greek_rsi = np.mean(list(results.values()))
                results['composite'] = composite_greek_rsi
                results['greek_sentiment'] = self._classify_greek_sentiment(
                    composite_greek_rsi
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating Greek RSI: {e}")
            return {}
    
    def _calculate_composite_rsi(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate composite RSI combining all components"""
        try:
            composite_values = {}
            
            # Calculate for each option type
            for option_type in ['CE', 'PE']:
                values = []
                weights = []
                
                # Price RSI (highest weight)
                if option_type in results['price_rsi']:
                    values.append(results['price_rsi'][option_type]['value'])
                    weights.append(0.4)
                
                # IV RSI
                if option_type in results['iv_rsi']:
                    values.append(results['iv_rsi'][option_type]['value'])
                    weights.append(0.3)
                
                # Greek RSI
                if option_type in results['greek_rsi'] and 'composite' in results['greek_rsi'][option_type]:
                    values.append(results['greek_rsi'][option_type]['composite'])
                    weights.append(0.3)
                
                # Calculate weighted composite
                if values:
                    composite = np.average(values, weights=weights[:len(values)])
                    composite_values[option_type] = composite
            
            # Calculate overall composite
            if composite_values:
                overall_composite = np.mean(list(composite_values.values()))
                composite_values['overall'] = overall_composite
                composite_values['signal_strength'] = self._calculate_signal_strength(
                    overall_composite
                )
            
            return composite_values
            
        except Exception as e:
            logger.error(f"Error calculating composite RSI: {e}")
            return {}
    
    def _generate_rsi_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from RSI values"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'overbought_options': [],
                'oversold_options': [],
                'divergence_signals': []
            }
            
            # Check composite RSI
            if 'overall' in results['composite_rsi']:
                composite_rsi = results['composite_rsi']['overall']
                
                if composite_rsi >= self.extreme_overbought:
                    signals['primary_signal'] = 'extreme_overbought'
                    signals['signal_strength'] = -1.0
                elif composite_rsi >= self.overbought_threshold:
                    signals['primary_signal'] = 'overbought'
                    signals['signal_strength'] = -0.7
                elif composite_rsi <= self.extreme_oversold:
                    signals['primary_signal'] = 'extreme_oversold'
                    signals['signal_strength'] = 1.0
                elif composite_rsi <= self.oversold_threshold:
                    signals['primary_signal'] = 'oversold'
                    signals['signal_strength'] = 0.7
                else:
                    signals['primary_signal'] = 'neutral'
                    signals['signal_strength'] = (composite_rsi - 50) / 50
            
            # Check individual option types
            for option_type in ['CE', 'PE']:
                if option_type in results['composite_rsi']:
                    rsi_value = results['composite_rsi'][option_type]
                    
                    if rsi_value >= self.overbought_threshold:
                        signals['overbought_options'].append(option_type)
                    elif rsi_value <= self.oversold_threshold:
                        signals['oversold_options'].append(option_type)
            
            # Add divergence signals
            if results['divergences']:
                signals['divergence_signals'] = [
                    d['type'] for d in results['divergences']
                ]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            return {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            }
    
    def _detect_rsi_divergences(self, 
                               option_data: pd.DataFrame,
                               rsi_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect RSI divergences"""
        try:
            divergences = []
            
            # Price vs RSI divergence
            price_div = self._check_price_rsi_divergence(option_data, rsi_results)
            if price_div:
                divergences.append(price_div)
            
            # IV vs RSI divergence
            iv_div = self._check_iv_rsi_divergence(option_data, rsi_results)
            if iv_div:
                divergences.append(iv_div)
            
            # Put-Call RSI divergence
            pc_div = self._check_put_call_divergence(rsi_results)
            if pc_div:
                divergences.append(pc_div)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergences: {e}")
            return []
    
    def _check_price_rsi_divergence(self,
                                   option_data: pd.DataFrame,
                                   rsi_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for price-RSI divergence"""
        # Simplified implementation
        # In production, would check actual price trends vs RSI trends
        return None
    
    def _check_iv_rsi_divergence(self,
                                option_data: pd.DataFrame,
                                rsi_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for IV-RSI divergence"""
        # Simplified implementation
        return None
    
    def _check_put_call_divergence(self, rsi_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for put-call RSI divergence"""
        try:
            if 'CE' in rsi_results['composite_rsi'] and 'PE' in rsi_results['composite_rsi']:
                ce_rsi = rsi_results['composite_rsi']['CE']
                pe_rsi = rsi_results['composite_rsi']['PE']
                
                divergence = abs(ce_rsi - pe_rsi)
                
                if divergence > 20:  # Significant divergence
                    return {
                        'type': 'put_call_divergence',
                        'ce_rsi': ce_rsi,
                        'pe_rsi': pe_rsi,
                        'divergence': divergence,
                        'signal': 'bullish' if ce_rsi < pe_rsi else 'bearish'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking put-call divergence: {e}")
            return None
    
    def _classify_rsi_regime(self, results: Dict[str, Any]) -> str:
        """Classify market regime based on RSI"""
        try:
            if 'overall' not in results['composite_rsi']:
                return 'undefined'
            
            rsi = results['composite_rsi']['overall']
            
            if rsi >= self.extreme_overbought:
                return 'extreme_overbought_regime'
            elif rsi >= self.overbought_threshold:
                return 'overbought_regime'
            elif rsi <= self.extreme_oversold:
                return 'extreme_oversold_regime'
            elif rsi <= self.oversold_threshold:
                return 'oversold_regime'
            elif 45 <= rsi <= 55:
                return 'neutral_regime'
            elif rsi > 55:
                return 'bullish_momentum_regime'
            else:
                return 'bearish_momentum_regime'
                
        except Exception as e:
            logger.error(f"Error classifying RSI regime: {e}")
            return 'undefined'
    
    def _calculate_rsi_trend(self, rsi_series: pd.Series) -> str:
        """Calculate RSI trend direction"""
        try:
            if len(rsi_series) < 3:
                return 'neutral'
            
            recent_rsi = rsi_series.tail(3).values
            
            if recent_rsi[-1] > recent_rsi[-2] > recent_rsi[-3]:
                return 'rising'
            elif recent_rsi[-1] < recent_rsi[-2] < recent_rsi[-3]:
                return 'falling'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _calculate_rsi_strength(self, rsi_value: float) -> str:
        """Calculate RSI signal strength"""
        if rsi_value >= self.extreme_overbought or rsi_value <= self.extreme_oversold:
            return 'extreme'
        elif rsi_value >= self.overbought_threshold or rsi_value <= self.oversold_threshold:
            return 'strong'
        elif 45 <= rsi_value <= 55:
            return 'neutral'
        else:
            return 'moderate'
    
    def _classify_iv_regime(self, iv_rsi: float) -> str:
        """Classify IV regime based on RSI"""
        if iv_rsi >= 70:
            return 'high_iv_expansion'
        elif iv_rsi <= 30:
            return 'low_iv_contraction'
        else:
            return 'normal_iv_regime'
    
    def _classify_greek_sentiment(self, greek_rsi: float) -> str:
        """Classify Greek sentiment"""
        if greek_rsi >= 70:
            return 'extreme_greek_bullish'
        elif greek_rsi >= 60:
            return 'greek_bullish'
        elif greek_rsi <= 30:
            return 'extreme_greek_bearish'
        elif greek_rsi <= 40:
            return 'greek_bearish'
        else:
            return 'greek_neutral'
    
    def _calculate_signal_strength(self, rsi_value: float) -> float:
        """Calculate signal strength from RSI value"""
        if rsi_value >= self.overbought_threshold:
            return -(rsi_value - self.overbought_threshold) / (100 - self.overbought_threshold)
        elif rsi_value <= self.oversold_threshold:
            return (self.oversold_threshold - rsi_value) / self.oversold_threshold
        else:
            return (rsi_value - 50) / 50 * 0.5  # Scaled down for neutral zone
    
    def _update_rsi_history(self, results: Dict[str, Any]):
        """Update RSI history for tracking"""
        try:
            # Update price RSI history
            if 'overall' in results['composite_rsi']:
                self.rsi_history['composite'].append({
                    'timestamp': datetime.now(),
                    'value': results['composite_rsi']['overall'],
                    'regime': results['regime']
                })
            
            # Keep only recent history
            max_history = 100
            for key in self.rsi_history:
                if len(self.rsi_history[key]) > max_history:
                    self.rsi_history[key] = self.rsi_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating RSI history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'price_rsi': {},
            'iv_rsi': {},
            'greek_rsi': {},
            'composite_rsi': {'overall': 50.0},
            'signals': {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            },
            'divergences': [],
            'regime': 'undefined'
        }
    
    def get_rsi_summary(self) -> Dict[str, Any]:
        """Get summary of RSI analysis"""
        try:
            if not self.rsi_history['composite']:
                return {'status': 'no_history'}
            
            recent_values = [h['value'] for h in self.rsi_history['composite'][-20:]]
            
            return {
                'current_rsi': recent_values[-1] if recent_values else 50.0,
                'average_rsi': np.mean(recent_values),
                'rsi_trend': self._calculate_rsi_trend(pd.Series(recent_values)),
                'regime_distribution': self._calculate_regime_distribution(),
                'total_calculations': len(self.rsi_history['composite'])
            }
            
        except Exception as e:
            logger.error(f"Error getting RSI summary: {e}")
            return {'status': 'error'}
    
    def _calculate_regime_distribution(self) -> Dict[str, float]:
        """Calculate distribution of RSI regimes"""
        try:
            if not self.rsi_history['composite']:
                return {}
            
            regimes = [h['regime'] for h in self.rsi_history['composite']]
            total = len(regimes)
            
            distribution = {}
            for regime in set(regimes):
                count = regimes.count(regime)
                distribution[regime] = count / total
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating regime distribution: {e}")
            return {}