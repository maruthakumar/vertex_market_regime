"""
DTE-Specific OI Expiry Analysis Module

Analyzes OI patterns based on Days-to-Expiry (DTE) for detecting:
- Near-expiry gamma exposure and pin risk
- Cross-expiry OI flow patterns during roll periods
- Expiry-specific regime transitions
- Time decay weighted OI metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExpiryEffect(Enum):
    """Types of expiry effects on OI patterns"""
    GAMMA_SQUEEZE = "gamma_squeeze"
    PIN_RISK = "pin_risk"
    ROLL_PERIOD = "roll_period"
    NORMAL = "normal"
    EXPIRY_MAGNET = "expiry_magnet"


@dataclass
class DTEOIMetrics:
    """DTE-specific OI analysis metrics"""
    dte_bucket: str
    oi_concentration: float
    gamma_exposure: float
    pin_risk_probability: float
    time_decay_impact: float
    roll_intensity: float
    expiry_effect_type: ExpiryEffect
    vega_crush_potential: float
    expiry_volatility_score: float
    cross_expiry_flow: float
    regime_transition_probability: float


class DTEOIExpiryAnalyzer:
    """
    Analyzes OI patterns specific to DTE buckets for expiry effects
    """
    
    def __init__(self, expiry_thresholds: List[int] = None):
        """
        Initialize DTE OI Expiry Analyzer
        
        Args:
            expiry_thresholds: DTE bucket boundaries
        """
        self.expiry_thresholds = expiry_thresholds or [0, 3, 7, 15, 30, 45]
        self.dte_buckets = self._create_dte_buckets()
        logger.info(f"Initialized DTEOIExpiryAnalyzer with buckets: {self.dte_buckets}")
    
    def _create_dte_buckets(self) -> Dict[str, Tuple[int, int]]:
        """Create DTE bucket definitions"""
        buckets = {}
        for i in range(len(self.expiry_thresholds) - 1):
            bucket_name = f"dte_{self.expiry_thresholds[i]}_{self.expiry_thresholds[i+1]}"
            buckets[bucket_name] = (self.expiry_thresholds[i], self.expiry_thresholds[i+1])
        
        # Add far month bucket
        buckets[f"dte_{self.expiry_thresholds[-1]}+"] = (self.expiry_thresholds[-1], float('inf'))
        return buckets
    
    def analyze_dte_specific_oi(self, df: pd.DataFrame, 
                                dte_column: str = 'dte') -> Dict[str, DTEOIMetrics]:
        """
        Analyze OI patterns for each DTE bucket
        
        Args:
            df: Production data with DTE and OI columns
            dte_column: Name of DTE column
            
        Returns:
            DTE-specific OI metrics by bucket
        """
        if dte_column not in df.columns:
            logger.warning(f"DTE column '{dte_column}' not found")
            return {}
        
        metrics_by_bucket = {}
        
        for bucket_name, (min_dte, max_dte) in self.dte_buckets.items():
            # Filter data for this DTE bucket
            if max_dte == float('inf'):
                bucket_data = df[df[dte_column] >= min_dte]
            else:
                bucket_data = df[(df[dte_column] >= min_dte) & (df[dte_column] < max_dte)]
            
            if bucket_data.empty:
                continue
            
            # Calculate bucket-specific metrics
            metrics = self._calculate_bucket_metrics(bucket_data, bucket_name, min_dte, max_dte)
            metrics_by_bucket[bucket_name] = metrics
        
        return metrics_by_bucket
    
    def _calculate_bucket_metrics(self, data: pd.DataFrame, bucket_name: str,
                                 min_dte: int, max_dte: float) -> DTEOIMetrics:
        """Calculate metrics for a specific DTE bucket"""
        
        # OI concentration in this bucket
        ce_oi = data['ce_oi'].sum() if 'ce_oi' in data.columns else 0
        pe_oi = data['pe_oi'].sum() if 'pe_oi' in data.columns else 0
        total_oi = ce_oi + pe_oi
        
        # Gamma exposure calculation (higher near expiry)
        avg_dte = data['dte'].mean() if 'dte' in data.columns else 15
        gamma_exposure = self._calculate_gamma_exposure(avg_dte, total_oi)
        
        # Pin risk probability (highest for 0-3 DTE)
        pin_risk = self._calculate_pin_risk(min_dte, max_dte, total_oi)
        
        # Time decay impact
        time_decay = self._calculate_time_decay_impact(avg_dte)
        
        # Roll intensity (for roll periods)
        roll_intensity = self._calculate_roll_intensity(data, min_dte, max_dte)
        
        # Expiry effect classification
        expiry_effect = self._classify_expiry_effect(min_dte, max_dte, gamma_exposure, total_oi)
        
        # Vega crush potential
        vega_crush = self._calculate_vega_crush_potential(avg_dte, data)
        
        # Expiry volatility score
        expiry_vol = self._calculate_expiry_volatility(data, avg_dte)
        
        # Cross-expiry flow
        cross_flow = self._estimate_cross_expiry_flow(data, min_dte)
        
        # Regime transition probability
        regime_transition = self._calculate_regime_transition_probability(data, avg_dte)
        
        return DTEOIMetrics(
            dte_bucket=bucket_name,
            oi_concentration=total_oi,
            gamma_exposure=gamma_exposure,
            pin_risk_probability=pin_risk,
            time_decay_impact=time_decay,
            roll_intensity=roll_intensity,
            expiry_effect_type=expiry_effect,
            vega_crush_potential=vega_crush,
            expiry_volatility_score=expiry_vol,
            cross_expiry_flow=cross_flow,
            regime_transition_probability=regime_transition
        )
    
    def _calculate_gamma_exposure(self, avg_dte: float, total_oi: float) -> float:
        """Calculate gamma exposure based on DTE"""
        if avg_dte <= 0:
            return 1.0
        
        # Gamma increases exponentially as expiry approaches
        gamma_multiplier = np.exp(-avg_dte / 10)  # Exponential decay with DTE
        normalized_gamma = min(gamma_multiplier * (total_oi / 1000000), 1.0)
        
        return normalized_gamma
    
    def _calculate_pin_risk(self, min_dte: int, max_dte: float, total_oi: float) -> float:
        """Calculate pin risk probability"""
        if min_dte == 0 and max_dte <= 3:
            # Highest pin risk for 0-3 DTE
            base_risk = 0.7
        elif min_dte < 7:
            # Moderate pin risk for 3-7 DTE
            base_risk = 0.4
        else:
            # Low pin risk for >7 DTE
            base_risk = 0.1
        
        # Adjust by OI concentration
        oi_factor = min(total_oi / 10000000, 1.0)
        return base_risk * (1 + 0.3 * oi_factor)
    
    def _calculate_time_decay_impact(self, avg_dte: float) -> float:
        """Calculate time decay (theta) impact"""
        if avg_dte <= 0:
            return 1.0
        
        # Theta accelerates as expiry approaches
        theta_impact = 1.0 - np.exp(-5 / avg_dte) if avg_dte > 0 else 1.0
        return min(theta_impact, 1.0)
    
    def _calculate_roll_intensity(self, data: pd.DataFrame, min_dte: int, max_dte: float) -> float:
        """Calculate roll period intensity"""
        # Roll periods typically occur 3-7 days before expiry
        if min_dte >= 3 and max_dte <= 7:
            # Check for OI migration patterns
            if 'ce_oi' in data.columns and 'pe_oi' in data.columns:
                ce_change = data['ce_oi'].diff().abs().mean()
                pe_change = data['pe_oi'].diff().abs().mean()
                
                avg_oi = (data['ce_oi'].mean() + data['pe_oi'].mean()) / 2
                if avg_oi > 0:
                    roll_intensity = (ce_change + pe_change) / avg_oi
                    return min(roll_intensity, 1.0)
        
        return 0.0
    
    def _classify_expiry_effect(self, min_dte: int, max_dte: float, 
                               gamma_exposure: float, total_oi: float) -> ExpiryEffect:
        """Classify the type of expiry effect"""
        if min_dte == 0 and max_dte <= 3:
            if gamma_exposure > 0.7:
                return ExpiryEffect.GAMMA_SQUEEZE
            elif total_oi > 5000000:
                return ExpiryEffect.PIN_RISK
            else:
                return ExpiryEffect.EXPIRY_MAGNET
        elif min_dte >= 3 and max_dte <= 7:
            return ExpiryEffect.ROLL_PERIOD
        else:
            return ExpiryEffect.NORMAL
    
    def _calculate_vega_crush_potential(self, avg_dte: float, data: pd.DataFrame) -> float:
        """Calculate potential for vega crush near expiry"""
        if avg_dte > 7:
            return 0.0
        
        # Vega crush intensifies near expiry
        base_crush = (7 - avg_dte) / 7
        
        # Check for high IV conditions
        if 'ce_iv' in data.columns and 'pe_iv' in data.columns:
            avg_iv = (data['ce_iv'].mean() + data['pe_iv'].mean()) / 2
            if avg_iv > 20:  # High IV condition
                base_crush *= 1.5
        
        return min(base_crush, 1.0)
    
    def _calculate_expiry_volatility(self, data: pd.DataFrame, avg_dte: float) -> float:
        """Calculate expiry-specific volatility score"""
        if avg_dte > 30:
            return 0.2  # Low volatility for far expiries
        
        # Calculate price volatility if available
        volatility = 0.5  # Default moderate volatility
        
        if 'ce_close' in data.columns and 'pe_close' in data.columns:
            ce_vol = data['ce_close'].pct_change().std()
            pe_vol = data['pe_close'].pct_change().std()
            
            if not np.isnan(ce_vol) and not np.isnan(pe_vol):
                avg_vol = (ce_vol + pe_vol) / 2
                volatility = min(avg_vol * 10, 1.0)  # Normalize to 0-1
        
        # Adjust for DTE
        dte_factor = 1.0 - (avg_dte / 30)
        return volatility * (1 + 0.5 * dte_factor)
    
    def _estimate_cross_expiry_flow(self, data: pd.DataFrame, min_dte: int) -> float:
        """Estimate cross-expiry OI flow"""
        if min_dte > 7:
            return 0.0  # No significant cross-expiry flow for far expiries
        
        # Look for OI changes that indicate roll activity
        flow_score = 0.0
        
        if 'ce_oi' in data.columns and 'pe_oi' in data.columns:
            ce_changes = data['ce_oi'].diff()
            pe_changes = data['pe_oi'].diff()
            
            # Large opposite changes indicate roll activity
            correlation = ce_changes.corr(pe_changes)
            if correlation < -0.3:  # Negative correlation suggests rolling
                flow_score = abs(correlation)
        
        return flow_score
    
    def _calculate_regime_transition_probability(self, data: pd.DataFrame, avg_dte: float) -> float:
        """Calculate probability of regime transition near expiry"""
        if avg_dte > 15:
            return 0.1  # Low transition probability for far expiries
        
        # Higher probability near expiry
        base_probability = (15 - avg_dte) / 15 * 0.5
        
        # Check for OI concentration changes
        if 'ce_oi' in data.columns and 'pe_oi' in data.columns:
            pcr = data['pe_oi'].sum() / (data['ce_oi'].sum() + 1)
            
            # Extreme PCR values indicate potential regime change
            if pcr < 0.5 or pcr > 2.0:
                base_probability *= 1.5
        
        return min(base_probability, 0.9)
    
    def detect_expiry_effects(self, df: pd.DataFrame, 
                            near_expiry_threshold: int = 3) -> Dict[str, Any]:
        """
        Detect specific expiry effects like gamma squeeze and pin risk
        
        Args:
            df: Production data
            near_expiry_threshold: DTE threshold for near-expiry classification
            
        Returns:
            Dictionary of detected expiry effects
        """
        effects = {
            'gamma_squeeze_detected': False,
            'pin_risk_detected': False,
            'roll_period_active': False,
            'expiry_magnet_active': False,
            'effects_summary': []
        }
        
        # Filter near-expiry data
        if 'dte' in df.columns:
            near_expiry = df[df['dte'] <= near_expiry_threshold]
            
            if not near_expiry.empty:
                # Check for gamma squeeze conditions
                if self._detect_gamma_squeeze(near_expiry):
                    effects['gamma_squeeze_detected'] = True
                    effects['effects_summary'].append("Gamma squeeze conditions detected")
                
                # Check for pin risk
                if self._detect_pin_risk(near_expiry):
                    effects['pin_risk_detected'] = True
                    effects['effects_summary'].append("Pin risk at major strikes")
                
                # Check for expiry magnet effect
                if self._detect_expiry_magnet(near_expiry):
                    effects['expiry_magnet_active'] = True
                    effects['effects_summary'].append("Expiry magnet effect active")
        
        # Check for roll period
        if 'dte' in df.columns:
            roll_period = df[(df['dte'] >= 3) & (df['dte'] <= 7)]
            if not roll_period.empty and self._detect_roll_period(roll_period):
                effects['roll_period_active'] = True
                effects['effects_summary'].append("Roll period activity detected")
        
        return effects
    
    def _detect_gamma_squeeze(self, data: pd.DataFrame) -> bool:
        """Detect gamma squeeze conditions"""
        if 'ce_oi' not in data.columns or 'pe_oi' not in data.columns:
            return False
        
        # High OI concentration near expiry
        total_oi = data['ce_oi'].sum() + data['pe_oi'].sum()
        
        # Check for concentrated strikes
        if 'atm_strike' in data.columns:
            atm_oi = data[data['call_strike_type'] == 'ATM']['ce_oi'].sum() if 'call_strike_type' in data.columns else 0
            concentration = atm_oi / (total_oi + 1)
            
            return concentration > 0.3 and total_oi > 1000000
        
        return False
    
    def _detect_pin_risk(self, data: pd.DataFrame) -> bool:
        """Detect pin risk conditions"""
        if 'ce_oi' not in data.columns or 'pe_oi' not in data.columns:
            return False
        
        # Look for high OI at specific strikes
        if 'strike' in data.columns:
            strike_oi = data.groupby('strike')[['ce_oi', 'pe_oi']].sum()
            max_oi = strike_oi.sum(axis=1).max()
            total_oi = strike_oi.sum().sum()
            
            if total_oi > 0:
                concentration = max_oi / total_oi
                return concentration > 0.2  # 20% concentration at single strike
        
        return False
    
    def _detect_expiry_magnet(self, data: pd.DataFrame) -> bool:
        """Detect expiry magnet effect"""
        # Check if price is converging to max pain or high OI strikes
        if 'spot' in data.columns and 'atm_strike' in data.columns:
            price_to_strike_ratio = data['spot'].iloc[-1] / data['atm_strike'].iloc[-1]
            
            # Price very close to strike indicates magnet effect
            return 0.99 < price_to_strike_ratio < 1.01
        
        return False
    
    def _detect_roll_period(self, data: pd.DataFrame) -> bool:
        """Detect roll period activity"""
        if 'ce_oi' not in data.columns or 'pe_oi' not in data.columns:
            return False
        
        # Look for significant OI changes
        ce_changes = data['ce_oi'].diff().abs().mean()
        pe_changes = data['pe_oi'].diff().abs().mean()
        
        avg_oi = (data['ce_oi'].mean() + data['pe_oi'].mean()) / 2
        
        if avg_oi > 0:
            change_rate = (ce_changes + pe_changes) / avg_oi
            return change_rate > 0.1  # 10% change rate indicates rolling
        
        return False
    
    def analyze_cross_expiry_flows(self, current_expiry_df: pd.DataFrame,
                                  next_expiry_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze OI flows between current and next expiry
        
        Args:
            current_expiry_df: Current expiry data
            next_expiry_df: Next expiry data
            
        Returns:
            Cross-expiry flow metrics
        """
        flows = {
            'roll_ratio': 0.0,
            'flow_direction': 0.0,  # Positive = rolling forward, Negative = unwinding
            'hedging_activity': 0.0,
            'calendar_spread_activity': 0.0
        }
        
        if current_expiry_df.empty or next_expiry_df.empty:
            return flows
        
        # Calculate OI totals
        if 'ce_oi' in current_expiry_df.columns and 'pe_oi' in current_expiry_df.columns:
            current_oi = current_expiry_df['ce_oi'].sum() + current_expiry_df['pe_oi'].sum()
            next_oi = next_expiry_df['ce_oi'].sum() + next_expiry_df['pe_oi'].sum()
            
            # Roll ratio
            if current_oi > 0:
                flows['roll_ratio'] = next_oi / current_oi
            
            # Flow direction
            current_change = current_expiry_df['ce_oi'].diff().sum() + current_expiry_df['pe_oi'].diff().sum()
            next_change = next_expiry_df['ce_oi'].diff().sum() + next_expiry_df['pe_oi'].diff().sum()
            
            if current_change < 0 and next_change > 0:
                flows['flow_direction'] = 1.0  # Rolling forward
            elif current_change > 0 and next_change < 0:
                flows['flow_direction'] = -1.0  # Unwinding
            
            # Hedging activity (opposite changes in CE/PE)
            current_ce_change = current_expiry_df['ce_oi'].diff().sum()
            current_pe_change = current_expiry_df['pe_oi'].diff().sum()
            
            if current_ce_change * current_pe_change < 0:  # Opposite signs
                flows['hedging_activity'] = abs(current_ce_change - current_pe_change) / (current_oi + 1)
            
            # Calendar spread activity
            if abs(current_change) > 0 and abs(next_change) > 0:
                flows['calendar_spread_activity'] = min(abs(current_change), abs(next_change)) / max(current_oi, next_oi)
        
        return flows
    
    def calculate_dte_weighted_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate time decay weighted OI metrics
        
        Args:
            df: Production data with DTE information
            
        Returns:
            DTE-weighted metrics
        """
        metrics = {
            'theta_weighted_oi': 0.0,
            'gamma_weighted_oi': 0.0,
            'vega_weighted_oi': 0.0,
            'charm_effect': 0.0,
            'vanna_impact': 0.0,
            'time_value_erosion': 0.0
        }
        
        if 'dte' not in df.columns:
            return metrics
        
        # Calculate weighted metrics for each DTE group
        for dte_value in df['dte'].unique():
            dte_data = df[df['dte'] == dte_value]
            
            if 'ce_oi' in dte_data.columns and 'pe_oi' in dte_data.columns:
                total_oi = dte_data['ce_oi'].sum() + dte_data['pe_oi'].sum()
                
                # Theta weighting (time decay)
                theta_weight = np.exp(-dte_value / 30)  # Exponential decay
                metrics['theta_weighted_oi'] += total_oi * theta_weight
                
                # Gamma weighting (acceleration)
                gamma_weight = np.exp(-dte_value / 10) if dte_value < 10 else 0.1
                metrics['gamma_weighted_oi'] += total_oi * gamma_weight
                
                # Vega weighting (volatility sensitivity)
                vega_weight = 1.0 - np.exp(-dte_value / 20)  # Higher for longer DTE
                metrics['vega_weighted_oi'] += total_oi * vega_weight
                
                # Charm effect (delta decay)
                metrics['charm_effect'] += total_oi * theta_weight * gamma_weight
                
                # Vanna impact (delta-vega correlation)
                metrics['vanna_impact'] += total_oi * gamma_weight * vega_weight
                
                # Time value erosion
                if dte_value < 7:
                    erosion_rate = (7 - dte_value) / 7
                    metrics['time_value_erosion'] += total_oi * erosion_rate
        
        # Normalize metrics
        total_oi_sum = df['ce_oi'].sum() + df['pe_oi'].sum() if 'ce_oi' in df.columns else 1
        for key in metrics:
            if total_oi_sum > 0:
                metrics[key] /= total_oi_sum
            metrics[key] = min(metrics[key], 1.0)  # Cap at 1.0
        
        return metrics
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract DTE-specific features for Component 3
        
        Args:
            df: Production data
            
        Returns:
            Array of DTE-specific features
        """
        features = []
        
        # Get DTE bucket metrics
        dte_metrics = self.analyze_dte_specific_oi(df)
        
        # Extract features from each bucket
        for bucket_name, metrics in dte_metrics.items():
            features.extend([
                metrics.gamma_exposure,
                metrics.pin_risk_probability,
                metrics.time_decay_impact,
                metrics.roll_intensity,
                metrics.vega_crush_potential,
                metrics.expiry_volatility_score,
                metrics.cross_expiry_flow,
                metrics.regime_transition_probability
            ])
        
        # Add expiry effects
        expiry_effects = self.detect_expiry_effects(df)
        features.extend([
            float(expiry_effects['gamma_squeeze_detected']),
            float(expiry_effects['pin_risk_detected']),
            float(expiry_effects['roll_period_active']),
            float(expiry_effects['expiry_magnet_active'])
        ])
        
        # Add weighted metrics
        weighted_metrics = self.calculate_dte_weighted_metrics(df)
        features.extend(list(weighted_metrics.values()))
        
        return np.array(features)