"""
DTE Adjuster - Days to Expiry Specific Adjustments
=================================================

Handles DTE-specific adjustments for Greek values, preserving the original
logic while adding enhanced DTE analysis capabilities.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Enhanced DTE Analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DTEAdjuster:
    """
    DTE-specific Greek adjustments with enhanced analysis
    
    Preserves original DTE adjustment logic while adding:
    - Enhanced DTE classification
    - Time decay impact analysis
    - Volatility-DTE interaction effects
    - Multi-expiry coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DTE Adjuster"""
        self.config = config or {}
        
        # DTE category thresholds (preserved from original)
        self.near_expiry_threshold = self.config.get('near_expiry_threshold', 7)  # 0-7 DTE
        self.medium_expiry_threshold = self.config.get('medium_expiry_threshold', 30)  # 8-30 DTE
        # far_expiry is 30+ DTE
        
        # Original DTE adjustments (preserved)
        self.dte_adjustments = {
            'near_expiry': {    # 0-7 DTE
                'delta': self.config.get('near_delta_adj', 1.0),
                'vega': self.config.get('near_vega_adj', 0.8),
                'theta': self.config.get('near_theta_adj', 1.5),  # Higher theta impact near expiry
                'gamma': self.config.get('near_gamma_adj', 1.2)
            },
            'medium_expiry': {  # 8-30 DTE
                'delta': self.config.get('medium_delta_adj', 1.2),
                'vega': self.config.get('medium_vega_adj', 1.5),    # Balanced vega impact
                'theta': self.config.get('medium_theta_adj', 0.8),
                'gamma': self.config.get('medium_gamma_adj', 1.0)
            },
            'far_expiry': {     # 30+ DTE
                'delta': self.config.get('far_delta_adj', 1.0),
                'vega': self.config.get('far_vega_adj', 2.0),      # Higher vega impact far from expiry
                'theta': self.config.get('far_theta_adj', 0.3),
                'gamma': self.config.get('far_gamma_adj', 0.8)
            }
        }
        
        # Enhanced DTE analysis
        self.enable_enhanced_analysis = self.config.get('enable_enhanced_analysis', True)
        self.volatility_dte_interaction = self.config.get('volatility_dte_interaction', True)
        self.multi_expiry_analysis = self.config.get('multi_expiry_analysis', True)
        
        # Time decay modeling
        self.time_decay_model = self.config.get('time_decay_model', 'exponential')  # exponential, linear
        self.weekend_adjustment = self.config.get('weekend_adjustment', True)
        
        # Advanced thresholds
        self.very_near_threshold = self.config.get('very_near_threshold', 2)  # 0-2 DTE (ultra near)
        self.very_far_threshold = self.config.get('very_far_threshold', 60)   # 60+ DTE (very far)
        
        # Analysis history
        self.adjustment_history = []
        
        logger.info(f"DTEAdjuster initialized: near≤{self.near_expiry_threshold}, medium≤{self.medium_expiry_threshold}")
    
    def apply_dte_adjustments(self, 
                            baseline_changes: Dict[str, float],
                            dte: int,
                            market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Apply DTE-specific adjustments to Greek baseline changes
        
        Args:
            baseline_changes: Greek changes from baseline
            dte: Days to expiry
            market_conditions: Current market conditions for enhanced analysis
            
        Returns:
            Dict[str, float]: DTE-adjusted Greek changes
        """
        try:
            # Classify DTE category (preserved logic)
            dte_category = self._classify_dte_category(dte)
            
            # Get base adjustments (preserved)
            base_adjustments = self.dte_adjustments[dte_category]
            
            # Apply enhanced adjustments if enabled
            if self.enable_enhanced_analysis and market_conditions:
                enhanced_adjustments = self._calculate_enhanced_adjustments(
                    dte, dte_category, market_conditions
                )
                # Combine base and enhanced adjustments
                final_adjustments = self._combine_adjustments(base_adjustments, enhanced_adjustments)
            else:
                final_adjustments = base_adjustments
            
            # Apply adjustments to baseline changes
            adjusted_greeks = {}
            for greek, change in baseline_changes.items():
                adjustment = final_adjustments.get(greek, 1.0)
                adjusted_greeks[greek] = change * adjustment
            
            # Record adjustment for analysis
            self._record_adjustment(dte, dte_category, baseline_changes, adjusted_greeks, final_adjustments)
            
            logger.debug(f"DTE adjustments applied: DTE={dte} ({dte_category}), adjustments={final_adjustments}")
            
            return adjusted_greeks
            
        except Exception as e:
            logger.error(f"Error applying DTE adjustments: {e}")
            return baseline_changes
    
    def _classify_dte_category(self, dte: int) -> str:
        """Classify DTE into categories (preserved original logic)"""
        if dte <= self.near_expiry_threshold:
            return 'near_expiry'
        elif dte <= self.medium_expiry_threshold:
            return 'medium_expiry'
        else:
            return 'far_expiry'
    
    def _calculate_enhanced_adjustments(self, 
                                      dte: int,
                                      dte_category: str,
                                      market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate enhanced DTE adjustments based on market conditions"""
        try:
            enhanced_adjustments = {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
            
            # Volatility-DTE interaction effects
            if self.volatility_dte_interaction:
                volatility_adjustments = self._calculate_volatility_dte_effects(
                    dte, market_conditions.get('volatility', 0.2)
                )
                enhanced_adjustments = self._merge_adjustments(enhanced_adjustments, volatility_adjustments)
            
            # Time decay acceleration near expiry
            time_decay_adjustments = self._calculate_time_decay_effects(dte)
            enhanced_adjustments = self._merge_adjustments(enhanced_adjustments, time_decay_adjustments)
            
            # Weekend/holiday adjustments
            if self.weekend_adjustment:
                weekend_adjustments = self._calculate_weekend_effects(dte, market_conditions)
                enhanced_adjustments = self._merge_adjustments(enhanced_adjustments, weekend_adjustments)
            
            # Ultra-near expiry special handling
            if dte <= self.very_near_threshold:
                ultra_near_adjustments = self._calculate_ultra_near_effects(dte)
                enhanced_adjustments = self._merge_adjustments(enhanced_adjustments, ultra_near_adjustments)
            
            return enhanced_adjustments
            
        except Exception as e:
            logger.error(f"Error calculating enhanced adjustments: {e}")
            return {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
    
    def _calculate_volatility_dte_effects(self, dte: int, volatility: float) -> Dict[str, float]:
        """Calculate volatility-DTE interaction effects"""
        try:
            adjustments = {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
            
            # High volatility effects
            if volatility > 0.3:  # High volatility
                if dte <= 7:  # Near expiry + high vol = enhanced gamma/theta
                    adjustments['gamma'] *= 1.3
                    adjustments['theta'] *= 1.2
                    adjustments['vega'] *= 0.9  # Reduced vega sensitivity
                elif dte >= 30:  # Far expiry + high vol = enhanced vega
                    adjustments['vega'] *= 1.4
                    adjustments['gamma'] *= 0.9
            
            # Low volatility effects
            elif volatility < 0.15:  # Low volatility
                if dte <= 7:  # Near expiry + low vol = reduced gamma sensitivity
                    adjustments['gamma'] *= 0.8
                    adjustments['theta'] *= 1.1
                elif dte >= 30:  # Far expiry + low vol = reduced vega
                    adjustments['vega'] *= 0.7
                    adjustments['delta'] *= 1.1
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating volatility-DTE effects: {e}")
            return {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
    
    def _calculate_time_decay_effects(self, dte: int) -> Dict[str, float]:
        """Calculate time decay acceleration effects"""
        try:
            adjustments = {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
            
            if self.time_decay_model == 'exponential':
                # Exponential time decay acceleration
                if dte <= 7:
                    # Accelerating theta decay near expiry
                    theta_multiplier = 1.0 + (7 - dte) * 0.2  # Up to 40% increase at 0 DTE
                    adjustments['theta'] *= theta_multiplier
                    
                    # Gamma acceleration
                    gamma_multiplier = 1.0 + (7 - dte) * 0.15  # Up to 30% increase at 0 DTE
                    adjustments['gamma'] *= gamma_multiplier
                
                elif dte >= 45:
                    # Reduced time decay sensitivity far from expiry
                    adjustments['theta'] *= 0.7
                    adjustments['gamma'] *= 0.8
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating time decay effects: {e}")
            return {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
    
    def _calculate_weekend_effects(self, dte: int, market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weekend/holiday time decay effects"""
        try:
            adjustments = {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
            
            # Check if weekend is approaching
            current_day = market_conditions.get('weekday', 2)  # 0=Monday, 4=Friday
            
            if current_day == 4 and dte <= 7:  # Friday with near expiry
                # Weekend time decay acceleration
                adjustments['theta'] *= 1.3  # Weekend decay effect
                adjustments['gamma'] *= 1.1
            
            elif current_day == 0 and dte <= 7:  # Monday with near expiry
                # Post-weekend adjustment
                adjustments['theta'] *= 0.9
                adjustments['vega'] *= 1.1
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating weekend effects: {e}")
            return {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
    
    def _calculate_ultra_near_effects(self, dte: int) -> Dict[str, float]:
        """Calculate ultra-near expiry effects (0-2 DTE)"""
        try:
            adjustments = {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
            
            if dte == 0:  # Expiry day
                adjustments['gamma'] *= 2.0  # Extreme gamma risk
                adjustments['theta'] *= 3.0  # Extreme time decay
                adjustments['vega'] *= 0.3   # Minimal vega sensitivity
                adjustments['delta'] *= 1.2  # Enhanced delta sensitivity
            
            elif dte == 1:  # 1 DTE
                adjustments['gamma'] *= 1.6
                adjustments['theta'] *= 2.0
                adjustments['vega'] *= 0.5
                adjustments['delta'] *= 1.1
            
            elif dte == 2:  # 2 DTE
                adjustments['gamma'] *= 1.3
                adjustments['theta'] *= 1.5
                adjustments['vega'] *= 0.7
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating ultra-near effects: {e}")
            return {'delta': 1.0, 'gamma': 1.0, 'theta': 1.0, 'vega': 1.0}
    
    def _combine_adjustments(self, 
                           base_adjustments: Dict[str, float],
                           enhanced_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Combine base and enhanced adjustments"""
        try:
            combined = {}
            
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                base_adj = base_adjustments.get(greek, 1.0)
                enhanced_adj = enhanced_adjustments.get(greek, 1.0)
                
                # Weighted combination (60% base, 40% enhanced)
                combined[greek] = 0.6 * base_adj + 0.4 * enhanced_adj
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining adjustments: {e}")
            return base_adjustments
    
    def _merge_adjustments(self, 
                         current: Dict[str, float],
                         new: Dict[str, float]) -> Dict[str, float]:
        """Merge adjustment dictionaries"""
        try:
            merged = current.copy()
            
            for greek, adjustment in new.items():
                if greek in merged:
                    merged[greek] *= adjustment
                else:
                    merged[greek] = adjustment
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging adjustments: {e}")
            return current
    
    def analyze_multi_expiry_effects(self, 
                                   market_data_by_expiry: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze effects across multiple expiries"""
        try:
            if not self.multi_expiry_analysis or len(market_data_by_expiry) < 2:
                return {}
            
            expiry_analysis = {}
            
            # Analyze each expiry
            for dte, market_data in market_data_by_expiry.items():
                category = self._classify_dte_category(dte)
                adjustments = self.dte_adjustments[category]
                
                expiry_analysis[dte] = {
                    'category': category,
                    'adjustments': adjustments,
                    'weight': self._calculate_expiry_weight(dte, list(market_data_by_expiry.keys()))
                }
            
            # Cross-expiry interactions
            interactions = self._calculate_cross_expiry_interactions(expiry_analysis)
            
            return {
                'expiry_analysis': expiry_analysis,
                'cross_expiry_interactions': interactions,
                'dominant_expiry': self._identify_dominant_expiry(expiry_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing multi-expiry effects: {e}")
            return {}
    
    def _calculate_expiry_weight(self, dte: int, all_dtes: List[int]) -> float:
        """Calculate weight for this expiry relative to others"""
        try:
            # Weight based on activity concentration and time proximity
            if dte <= 7:
                return 1.5  # Higher weight for near expiry
            elif dte <= 30:
                return 1.0  # Standard weight
            else:
                return 0.7  # Lower weight for far expiry
        except:
            return 1.0
    
    def _calculate_cross_expiry_interactions(self, expiry_analysis: Dict[int, Dict]) -> Dict[str, Any]:
        """Calculate interactions between different expiries"""
        try:
            interactions = {
                'calendar_spread_effects': {},
                'volatility_term_structure': {},
                'time_decay_coordination': {}
            }
            
            # Placeholder for complex cross-expiry analysis
            return interactions
            
        except Exception as e:
            logger.error(f"Error calculating cross-expiry interactions: {e}")
            return {}
    
    def _identify_dominant_expiry(self, expiry_analysis: Dict[int, Dict]) -> Optional[int]:
        """Identify the expiry with highest impact"""
        try:
            if not expiry_analysis:
                return None
            
            # Simple implementation - return the one with highest weight
            dominant_dte = max(expiry_analysis.keys(), 
                             key=lambda dte: expiry_analysis[dte]['weight'])
            return dominant_dte
            
        except Exception as e:
            logger.error(f"Error identifying dominant expiry: {e}")
            return None
    
    def _record_adjustment(self, 
                         dte: int,
                         dte_category: str,
                         baseline_changes: Dict[str, float],
                         adjusted_greeks: Dict[str, float],
                         adjustments: Dict[str, float]):
        """Record adjustment details for analysis"""
        try:
            record = {
                'timestamp': datetime.now(),
                'dte': dte,
                'dte_category': dte_category,
                'baseline_changes': baseline_changes.copy(),
                'adjusted_greeks': adjusted_greeks.copy(),
                'adjustments_applied': adjustments.copy(),
                'adjustment_magnitude': sum(abs(v) for v in adjustments.values())
            }
            
            self.adjustment_history.append(record)
            
            # Keep only last 100 adjustments
            if len(self.adjustment_history) > 100:
                self.adjustment_history = self.adjustment_history[-100:]
                
        except Exception as e:
            logger.error(f"Error recording adjustment: {e}")
    
    def get_dte_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of DTE analysis performance"""
        try:
            if not self.adjustment_history:
                return {'status': 'no_data'}
            
            recent_adjustments = self.adjustment_history[-20:]
            
            # Category distribution
            category_counts = {}
            for record in recent_adjustments:
                category = record['dte_category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Average adjustments by category
            category_adjustments = {}
            for category in ['near_expiry', 'medium_expiry', 'far_expiry']:
                category_records = [r for r in recent_adjustments if r['dte_category'] == category]
                if category_records:
                    avg_adjustments = {}
                    for greek in ['delta', 'gamma', 'theta', 'vega']:
                        avg_adjustments[greek] = np.mean([
                            r['adjustments_applied'].get(greek, 1.0) for r in category_records
                        ])
                    category_adjustments[category] = avg_adjustments
            
            return {
                'total_adjustments': len(self.adjustment_history),
                'recent_adjustments': len(recent_adjustments),
                'category_distribution': category_counts,
                'average_adjustments_by_category': category_adjustments,
                'dte_thresholds': {
                    'near_expiry': self.near_expiry_threshold,
                    'medium_expiry': self.medium_expiry_threshold
                },
                'enhanced_analysis_enabled': self.enable_enhanced_analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating DTE analysis summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_dte_adjustments(self, 
                             category: str,
                             greek: str,
                             new_adjustment: float):
        """Update DTE adjustment parameters"""
        try:
            if category in self.dte_adjustments and greek in self.dte_adjustments[category]:
                old_value = self.dte_adjustments[category][greek]
                self.dte_adjustments[category][greek] = new_adjustment
                
                logger.info(f"Updated {category} {greek} adjustment: {old_value:.3f} -> {new_adjustment:.3f}")
            else:
                logger.warning(f"Invalid category/greek combination: {category}/{greek}")
                
        except Exception as e:
            logger.error(f"Error updating DTE adjustments: {e}")
    
    def get_current_dte_config(self) -> Dict[str, Any]:
        """Get current DTE configuration"""
        return {
            'dte_adjustments': self.dte_adjustments.copy(),
            'thresholds': {
                'near_expiry': self.near_expiry_threshold,
                'medium_expiry': self.medium_expiry_threshold,
                'very_near': self.very_near_threshold,
                'very_far': self.very_far_threshold
            },
            'enhanced_features': {
                'enhanced_analysis': self.enable_enhanced_analysis,
                'volatility_dte_interaction': self.volatility_dte_interaction,
                'multi_expiry_analysis': self.multi_expiry_analysis,
                'weekend_adjustment': self.weekend_adjustment
            },
            'time_decay_model': self.time_decay_model
        }