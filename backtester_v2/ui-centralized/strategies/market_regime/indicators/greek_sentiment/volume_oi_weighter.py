"""
Volume OI Weighter - Dual Weighting System
==========================================

Implements the α×OI + β×Volume dual weighting system for Greek calculations.
This is a key enhancement that combines both OI and Volume for better sentiment
analysis, especially important for capturing institutional vs retail activity.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Dual Weighting Implementation
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VolumeOIWeighter:
    """
    Dual weighting system implementing α×OI + β×Volume
    
    Key Features:
    - Configurable alpha (OI weight) and beta (Volume weight)
    - Adaptive weighting based on market conditions
    - Strike-level weight calculation
    - Quality assessment for weighting reliability
    - Support for institutional vs retail detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Volume OI Weighter"""
        self.config = config or {}
        
        # Dual weighting parameters
        self.oi_weight_alpha = self.config.get('oi_weight_alpha', 0.6)
        self.volume_weight_beta = self.config.get('volume_weight_beta', 0.4)
        
        # Validation that weights sum close to 1.0
        total_weight = self.oi_weight_alpha + self.volume_weight_beta
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Alpha + Beta = {total_weight:.3f}, should be close to 1.0")
        
        # Advanced weighting parameters
        self.adaptive_weighting = self.config.get('adaptive_weighting', True)
        self.min_weight_threshold = self.config.get('min_weight_threshold', 0.001)
        self.outlier_detection = self.config.get('outlier_detection', True)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)  # Standard deviations
        
        # Institutional vs Retail detection
        self.institutional_oi_threshold = self.config.get('institutional_oi_threshold', 10000)
        self.institutional_volume_threshold = self.config.get('institutional_volume_threshold', 5000)
        
        # Weight calculation history for adaptation
        self.weight_calculation_history = []
        self.market_condition_adaptations = {}
        
        logger.info(f"VolumeOIWeighter initialized: α={self.oi_weight_alpha}, β={self.volume_weight_beta}")
    
    def calculate_dual_weighted_greeks(self, 
                                     market_data: pd.DataFrame,
                                     selected_strikes: List,
                                     spot_price: float,
                                     market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate Greeks with dual weighting (α × OI + β × Volume)
        
        Args:
            market_data: Market data DataFrame
            selected_strikes: List of selected strikes with metadata
            spot_price: Current spot price
            market_conditions: Current market conditions for adaptive weighting
            
        Returns:
            Dict[str, float]: Dual-weighted Greeks
        """
        try:
            # Initialize weighted Greeks
            weighted_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            total_weight = 0
            weight_details = []
            
            # Get adaptive weights if enabled
            if self.adaptive_weighting and market_conditions:
                adaptive_alpha, adaptive_beta = self._get_adaptive_weights(market_conditions)
            else:
                adaptive_alpha, adaptive_beta = self.oi_weight_alpha, self.volume_weight_beta
            
            # Process each selected strike
            for strike_info in selected_strikes:
                strike = strike_info.strike
                option_type = strike_info.option_type
                strike_weight = strike_info.weight
                
                # Get data for this strike and option type
                strike_data = market_data[
                    (market_data['strike'] == strike) & 
                    (market_data['option_type'] == option_type)
                ]
                
                if strike_data.empty:
                    continue
                
                row = strike_data.iloc[0]
                
                # Extract Greeks and market data
                greeks, oi, volume = self._extract_strike_data(row, option_type)
                
                if greeks is None:
                    continue
                
                # Calculate dual weight for this strike
                dual_weight = self._calculate_strike_dual_weight(
                    oi, volume, adaptive_alpha, adaptive_beta, strike_weight
                )
                
                # Apply outlier detection if enabled
                if self.outlier_detection:
                    dual_weight = self._apply_outlier_filtering(dual_weight, weight_details)
                
                # Record weight details for analysis
                weight_details.append({
                    'strike': strike,
                    'option_type': option_type,
                    'oi': oi,
                    'volume': volume,
                    'dual_weight': dual_weight,
                    'strike_weight': strike_weight,
                    'alpha_used': adaptive_alpha,
                    'beta_used': adaptive_beta
                })
                
                # Apply dual weight to Greeks
                if dual_weight > self.min_weight_threshold:
                    weighted_greeks['delta'] += greeks['delta'] * dual_weight
                    weighted_greeks['gamma'] += greeks['gamma'] * dual_weight
                    weighted_greeks['theta'] += greeks['theta'] * dual_weight
                    weighted_greeks['vega'] += greeks['vega'] * dual_weight
                    total_weight += dual_weight
            
            # Normalize by total weight
            if total_weight > 0:
                for greek in weighted_greeks:
                    weighted_greeks[greek] /= total_weight
            
            # Store calculation details for future adaptation
            self._record_calculation_details(weight_details, total_weight, market_conditions)
            
            logger.debug(f"Dual weighting completed: total_weight={total_weight:.3f}, "
                        f"α={adaptive_alpha:.3f}, β={adaptive_beta:.3f}")
            
            return weighted_greeks
            
        except Exception as e:
            logger.error(f"Error in dual weighted Greeks calculation: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _extract_strike_data(self, row: pd.Series, option_type: str) -> Tuple[Optional[Dict], float, float]:
        """Extract Greeks, OI, and Volume from data row"""
        try:
            # Extract Greeks based on option type
            if option_type == 'CE':
                greeks = {
                    'delta': row.get('ce_delta', 0),
                    'gamma': row.get('ce_gamma', 0),
                    'theta': row.get('ce_theta', 0),
                    'vega': row.get('ce_vega', 0)
                }
                oi = row.get('ce_oi', 0)
                volume = row.get('ce_volume', 0)
            else:  # PE
                greeks = {
                    'delta': row.get('pe_delta', 0),
                    'gamma': row.get('pe_gamma', 0),
                    'theta': row.get('pe_theta', 0),
                    'vega': row.get('pe_vega', 0)
                }
                oi = row.get('pe_oi', 0)
                volume = row.get('pe_volume', 0)
            
            # Validate Greeks data
            if all(abs(v) < 1e-10 for v in greeks.values()):
                return None, 0, 0
            
            return greeks, float(oi), float(volume)
            
        except Exception as e:
            logger.error(f"Error extracting strike data: {e}")
            return None, 0, 0
    
    def _calculate_strike_dual_weight(self, 
                                    oi: float, 
                                    volume: float,
                                    alpha: float,
                                    beta: float,
                                    strike_weight: float) -> float:
        """Calculate dual weight for a single strike"""
        try:
            # Basic dual weight calculation: α × OI + β × Volume
            dual_weight = alpha * oi + beta * volume
            
            # Apply strike-level weight multiplier
            dual_weight *= strike_weight
            
            # Ensure non-negative weight
            dual_weight = max(0, dual_weight)
            
            return dual_weight
            
        except Exception as e:
            logger.error(f"Error calculating strike dual weight: {e}")
            return 0.0
    
    def _get_adaptive_weights(self, market_conditions: Dict[str, Any]) -> Tuple[float, float]:
        """Get adaptive alpha and beta based on market conditions"""
        try:
            base_alpha = self.oi_weight_alpha
            base_beta = self.volume_weight_beta
            
            # Adapt based on volatility
            volatility = market_conditions.get('volatility', 0.2)
            if volatility > 0.3:  # High volatility - favor volume
                alpha = base_alpha * 0.9
                beta = base_beta * 1.1
            elif volatility < 0.15:  # Low volatility - favor OI
                alpha = base_alpha * 1.1
                beta = base_beta * 0.9
            else:
                alpha = base_alpha
                beta = base_beta
            
            # Adapt based on time of day
            current_hour = market_conditions.get('hour', 12)
            if 9 <= current_hour <= 10:  # Market open - favor volume
                alpha *= 0.95
                beta *= 1.05
            elif 14 <= current_hour <= 15:  # Market close - favor OI
                alpha *= 1.05
                beta *= 0.95
            
            # Normalize to maintain total weight
            total = alpha + beta
            if total > 0:
                alpha /= total
                beta /= total
            
            return alpha, beta
            
        except Exception as e:
            logger.error(f"Error getting adaptive weights: {e}")
            return self.oi_weight_alpha, self.volume_weight_beta
    
    def _apply_outlier_filtering(self, weight: float, weight_history: List[Dict]) -> float:
        """Apply outlier filtering to weight values"""
        try:
            if len(weight_history) < 3:  # Need some history for outlier detection
                return weight
            
            # Get recent weights
            recent_weights = [w['dual_weight'] for w in weight_history[-10:]]
            
            # Calculate statistics
            mean_weight = np.mean(recent_weights)
            std_weight = np.std(recent_weights)
            
            # Check if current weight is an outlier
            if std_weight > 0:
                z_score = abs(weight - mean_weight) / std_weight
                if z_score > self.outlier_threshold:
                    # Cap the weight to reduce outlier impact
                    capped_weight = mean_weight + np.sign(weight - mean_weight) * self.outlier_threshold * std_weight
                    logger.debug(f"Outlier detected: weight={weight:.3f}, capped to {capped_weight:.3f}")
                    return capped_weight
            
            return weight
            
        except Exception as e:
            logger.error(f"Error in outlier filtering: {e}")
            return weight
    
    def _record_calculation_details(self, 
                                  weight_details: List[Dict],
                                  total_weight: float,
                                  market_conditions: Optional[Dict[str, Any]]):
        """Record calculation details for future adaptation"""
        try:
            calculation_record = {
                'timestamp': pd.Timestamp.now(),
                'total_weight': total_weight,
                'strike_count': len(weight_details),
                'alpha_used': weight_details[0]['alpha_used'] if weight_details else self.oi_weight_alpha,
                'beta_used': weight_details[0]['beta_used'] if weight_details else self.volume_weight_beta,
                'market_conditions': market_conditions or {},
                'weight_distribution': {
                    'min_weight': min([w['dual_weight'] for w in weight_details]) if weight_details else 0,
                    'max_weight': max([w['dual_weight'] for w in weight_details]) if weight_details else 0,
                    'avg_weight': np.mean([w['dual_weight'] for w in weight_details]) if weight_details else 0
                }
            }
            
            self.weight_calculation_history.append(calculation_record)
            
            # Keep only last 100 calculations
            if len(self.weight_calculation_history) > 100:
                self.weight_calculation_history = self.weight_calculation_history[-100:]
                
        except Exception as e:
            logger.error(f"Error recording calculation details: {e}")
    
    def analyze_institutional_vs_retail_flow(self, 
                                           market_data: pd.DataFrame,
                                           selected_strikes: List) -> Dict[str, Any]:
        """
        Analyze institutional vs retail flow based on OI and Volume patterns
        
        Args:
            market_data: Market data DataFrame
            selected_strikes: Selected strikes for analysis
            
        Returns:
            Dict: Analysis of institutional vs retail activity
        """
        try:
            institutional_flow = {'calls': 0, 'puts': 0, 'total': 0}
            retail_flow = {'calls': 0, 'puts': 0, 'total': 0}
            
            for strike_info in selected_strikes:
                strike_data = market_data[
                    (market_data['strike'] == strike_info.strike) & 
                    (market_data['option_type'] == strike_info.option_type)
                ]
                
                if strike_data.empty:
                    continue
                
                row = strike_data.iloc[0]
                option_type = strike_info.option_type
                
                # Get OI and Volume
                if option_type == 'CE':
                    oi = row.get('ce_oi', 0)
                    volume = row.get('ce_volume', 0)
                    flow_type = 'calls'
                else:
                    oi = row.get('pe_oi', 0)
                    volume = row.get('pe_volume', 0)
                    flow_type = 'puts'
                
                # Classify as institutional or retail based on thresholds
                if (oi > self.institutional_oi_threshold or 
                    volume > self.institutional_volume_threshold):
                    institutional_flow[flow_type] += oi + volume
                    institutional_flow['total'] += oi + volume
                else:
                    retail_flow[flow_type] += oi + volume
                    retail_flow['total'] += oi + volume
            
            # Calculate ratios and sentiment
            total_flow = institutional_flow['total'] + retail_flow['total']
            
            analysis = {
                'institutional_flow': institutional_flow,
                'retail_flow': retail_flow,
                'institutional_ratio': institutional_flow['total'] / total_flow if total_flow > 0 else 0,
                'retail_ratio': retail_flow['total'] / total_flow if total_flow > 0 else 0,
                'institutional_sentiment': self._calculate_flow_sentiment(institutional_flow),
                'retail_sentiment': self._calculate_flow_sentiment(retail_flow),
                'flow_divergence': self._calculate_flow_divergence(institutional_flow, retail_flow)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing institutional vs retail flow: {e}")
            return {}
    
    def _calculate_flow_sentiment(self, flow: Dict[str, float]) -> float:
        """Calculate sentiment from flow data"""
        try:
            total_flow = flow['calls'] + flow['puts']
            if total_flow > 0:
                # Positive for call bias, negative for put bias
                sentiment = (flow['calls'] - flow['puts']) / total_flow
                return np.clip(sentiment, -1.0, 1.0)
            return 0.0
        except:
            return 0.0
    
    def _calculate_flow_divergence(self, 
                                 institutional_flow: Dict[str, float],
                                 retail_flow: Dict[str, float]) -> float:
        """Calculate divergence between institutional and retail flow"""
        try:
            inst_sentiment = self._calculate_flow_sentiment(institutional_flow)
            retail_sentiment = self._calculate_flow_sentiment(retail_flow)
            
            # Divergence is the absolute difference in sentiment
            divergence = abs(inst_sentiment - retail_sentiment)
            return divergence
            
        except:
            return 0.0
    
    def get_weighting_summary(self) -> Dict[str, Any]:
        """Get summary of weighting system performance"""
        try:
            if not self.weight_calculation_history:
                return {'status': 'no_data'}
            
            recent_calculations = self.weight_calculation_history[-20:]
            
            summary = {
                'total_calculations': len(self.weight_calculation_history),
                'recent_calculations': len(recent_calculations),
                'current_weights': {
                    'alpha_oi': self.oi_weight_alpha,
                    'beta_volume': self.volume_weight_beta
                },
                'recent_performance': {
                    'avg_total_weight': np.mean([calc['total_weight'] for calc in recent_calculations]),
                    'avg_strike_count': np.mean([calc['strike_count'] for calc in recent_calculations]),
                    'weight_stability': np.std([calc['total_weight'] for calc in recent_calculations])
                },
                'adaptive_weighting_enabled': self.adaptive_weighting,
                'outlier_detection_enabled': self.outlier_detection
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating weighting summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_weighting_parameters(self, 
                                  new_alpha: Optional[float] = None,
                                  new_beta: Optional[float] = None):
        """Update weighting parameters"""
        try:
            if new_alpha is not None:
                self.oi_weight_alpha = np.clip(new_alpha, 0.0, 1.0)
            
            if new_beta is not None:
                self.volume_weight_beta = np.clip(new_beta, 0.0, 1.0)
            
            # Normalize if both provided
            if new_alpha is not None and new_beta is not None:
                total = self.oi_weight_alpha + self.volume_weight_beta
                if total > 0:
                    self.oi_weight_alpha /= total
                    self.volume_weight_beta /= total
            
            logger.info(f"Updated weighting parameters: α={self.oi_weight_alpha:.3f}, β={self.volume_weight_beta:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating weighting parameters: {e}")
    
    def get_current_weighting_config(self) -> Dict[str, Any]:
        """Get current weighting configuration"""
        return {
            'oi_weight_alpha': self.oi_weight_alpha,
            'volume_weight_beta': self.volume_weight_beta,
            'adaptive_weighting': self.adaptive_weighting,
            'min_weight_threshold': self.min_weight_threshold,
            'outlier_detection': self.outlier_detection,
            'outlier_threshold': self.outlier_threshold,
            'institutional_thresholds': {
                'oi_threshold': self.institutional_oi_threshold,
                'volume_threshold': self.institutional_volume_threshold
            }
        }