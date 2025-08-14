"""
Advanced Pattern Detector for Support/Resistance
Additional 48+ features to expand from 72 to 120+ features
Based on story.1.8-additional-sr-patterns.md
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AdvancedPatternDetector:
    """
    Detects advanced S&R patterns including OI-based patterns,
    straddle divergences, and cross-component synergies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize advanced pattern detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # OI pattern parameters
        self.oi_concentration_percentile = config.get("oi_concentration_percentile", 85)
        self.max_pain_lookback = config.get("max_pain_lookback", 5)
        self.oi_flow_threshold = config.get("oi_flow_threshold", 0.2)  # 20% change
        
        # Straddle pattern parameters
        self.divergence_threshold = config.get("divergence_threshold", 0.05)  # 5%
        self.momentum_period = config.get("momentum_period", 10)
        self.exhaustion_threshold = config.get("exhaustion_threshold", 0.5)
        
        # Greeks pattern parameters
        self.gamma_threshold = config.get("gamma_threshold", 0.001)
        self.skew_percentile = config.get("skew_percentile", 90)
        
        logger.info("Initialized AdvancedPatternDetector for 48+ additional features")
    
    def extract_all_advanced_features(
        self,
        market_data: pd.DataFrame,
        options_data: Optional[pd.DataFrame] = None,
        greeks_data: Optional[Dict[str, Any]] = None,
        component_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract all advanced pattern features (48+ features)
        
        Returns:
            Dictionary containing feature arrays for each pattern type
        """
        features = {}
        
        # OI-based patterns (28 features total)
        if options_data is not None:
            features["oi_concentration_walls"] = self.detect_oi_concentration_walls(options_data)  # 8 features
            features["max_pain_migration"] = self.detect_max_pain_migration(options_data)  # 6 features
            features["oi_flow_velocity"] = self.detect_oi_flow_velocity(options_data)  # 6 features
            features["volume_weighted_oi"] = self.calculate_volume_weighted_oi_profile(options_data)  # 8 features
        
        # Advanced straddle patterns (10 features total)
        if component_data is not None:
            features["triple_divergence"] = self.detect_triple_straddle_divergence(component_data)  # 6 features
            features["momentum_exhaustion"] = self.detect_straddle_momentum_exhaustion(component_data)  # 4 features
        
        # Cross-component synergies (10 features total)
        if greeks_data is not None and options_data is not None:
            features["greeks_oi_confluence"] = self.detect_greeks_oi_confluence(greeks_data, options_data)  # 6 features
            features["iv_skew_asymmetry"] = self.detect_iv_skew_asymmetry(options_data)  # 4 features
        
        return features
    
    def detect_oi_concentration_walls(self, options_data: pd.DataFrame) -> np.ndarray:
        """
        Detect OI concentration walls that form strong S&R
        CE OI > 85th percentile = Resistance
        PE OI > 85th percentile = Support
        
        Returns:
            8-feature array
        """
        features = np.zeros(8)
        
        if "ce_oi" in options_data.columns and "pe_oi" in options_data.columns:
            # Calculate OI percentiles
            ce_oi_percentile = np.percentile(options_data["ce_oi"], self.oi_concentration_percentile)
            pe_oi_percentile = np.percentile(options_data["pe_oi"], self.oi_concentration_percentile)
            
            # Find concentration walls
            ce_walls = options_data[options_data["ce_oi"] > ce_oi_percentile]
            pe_walls = options_data[options_data["pe_oi"] > pe_oi_percentile]
            
            if len(ce_walls) > 0:
                # Top 3 CE walls (resistance)
                top_ce_walls = ce_walls.nlargest(3, "ce_oi")
                for i, (idx, row) in enumerate(top_ce_walls.iterrows()):
                    if i < 3:
                        features[i] = row["strike"] if "strike" in row else row["atm_strike"]
                        features[i + 3] = row["ce_oi"] / options_data["ce_oi"].max()  # Normalized strength
            
            if len(pe_walls) > 0:
                # Top 2 PE walls (support)
                top_pe_walls = pe_walls.nlargest(2, "pe_oi")
                for i, (idx, row) in enumerate(top_pe_walls.iterrows()):
                    if i < 2:
                        features[6 + i] = row["strike"] if "strike" in row else row["atm_strike"]
        
        return features
    
    def detect_max_pain_migration(self, options_data: pd.DataFrame) -> np.ndarray:
        """
        Track max pain point migration for S&R levels
        Previous max pain points become future S&R
        
        Returns:
            6-feature array
        """
        features = np.zeros(6)
        
        if "ce_oi" in options_data.columns and "pe_oi" in options_data.columns:
            # Calculate max pain for different lookback periods
            strikes = options_data["strike"].unique() if "strike" in options_data else options_data["atm_strike"].unique()
            
            max_pain_points = []
            for strike in strikes[:20]:  # Limit computation
                strike_data = options_data[options_data.get("strike", options_data.get("atm_strike")) == strike]
                if len(strike_data) > 0:
                    # Simple max pain calculation
                    ce_oi = strike_data["ce_oi"].sum()
                    pe_oi = strike_data["pe_oi"].sum()
                    pain_value = abs(ce_oi - pe_oi)
                    max_pain_points.append((strike, pain_value))
            
            # Sort by pain value and extract top levels
            max_pain_points.sort(key=lambda x: x[1], reverse=True)
            
            for i, (strike, pain_value) in enumerate(max_pain_points[:3]):
                if i < 3:
                    features[i] = strike
                    features[i + 3] = min(1.0, pain_value / 1000000)  # Normalized pain value
        
        return features
    
    def detect_oi_flow_velocity(self, options_data: pd.DataFrame) -> np.ndarray:
        """
        Detect rapid OI accumulation that creates future S&R
        
        Returns:
            6-feature array
        """
        features = np.zeros(6)
        
        if "ce_coi" in options_data.columns and "pe_coi" in options_data.columns:
            # Calculate OI change velocity
            ce_velocity = options_data["ce_coi"].abs()
            pe_velocity = options_data["pe_coi"].abs()
            
            # Find rapid accumulation points
            ce_rapid = options_data[ce_velocity > ce_velocity.quantile(0.9)]
            pe_rapid = options_data[pe_velocity > pe_velocity.quantile(0.9)]
            
            if len(ce_rapid) > 0:
                # Top 2 CE rapid accumulation
                top_ce = ce_rapid.nlargest(2, "ce_coi", keep='first')
                for i, (idx, row) in enumerate(top_ce.iterrows()):
                    if i < 2:
                        features[i] = row.get("strike", row.get("atm_strike", 0))
                        features[i + 2] = abs(row["ce_coi"]) / 10000  # Normalized velocity
            
            if len(pe_rapid) > 0:
                # Top 2 PE rapid accumulation
                top_pe = pe_rapid.nlargest(2, "pe_coi", keep='first')
                for i, (idx, row) in enumerate(top_pe.iterrows()):
                    if i < 2:
                        features[4 + i] = row.get("strike", row.get("atm_strike", 0))
        
        return features
    
    def calculate_volume_weighted_oi_profile(self, options_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate volume-weighted OI profile for true S&R
        
        Returns:
            8-feature array
        """
        features = np.zeros(8)
        
        if all(col in options_data.columns for col in ["ce_volume", "ce_oi", "pe_volume", "pe_oi"]):
            # Calculate volume-weighted OI
            ce_vwoi = options_data["ce_volume"] * options_data["ce_oi"]
            pe_vwoi = options_data["pe_volume"] * options_data["pe_oi"]
            
            # Find peaks in VWOI
            ce_peaks = options_data[ce_vwoi > ce_vwoi.quantile(0.8)]
            pe_peaks = options_data[pe_vwoi > pe_vwoi.quantile(0.8)]
            
            if len(ce_peaks) > 0:
                top_ce = ce_peaks.nlargest(3, "ce_volume")
                for i, (idx, row) in enumerate(top_ce.iterrows()):
                    if i < 3:
                        features[i] = row.get("strike", row.get("atm_strike", 0))
                        features[i + 4] = min(1.0, ce_vwoi.iloc[idx] / ce_vwoi.max())
            
            if len(pe_peaks) > 0:
                top_pe = pe_peaks.nlargest(1, "pe_volume")
                for i, (idx, row) in enumerate(top_pe.iterrows()):
                    if i < 1:
                        features[3] = row.get("strike", row.get("atm_strike", 0))
                        features[7] = min(1.0, pe_vwoi.iloc[idx] / pe_vwoi.max())
        
        return features
    
    def detect_triple_straddle_divergence(self, component_data: Dict[str, Any]) -> np.ndarray:
        """
        Detect divergence between ATM/ITM1/OTM1 straddles
        
        Returns:
            6-feature array
        """
        features = np.zeros(6)
        
        if all(key in component_data for key in ["atm", "itm1", "otm1"]):
            atm_prices = np.array(component_data["atm"].get("prices", []))
            itm1_prices = np.array(component_data["itm1"].get("prices", []))
            otm1_prices = np.array(component_data["otm1"].get("prices", []))
            
            if len(atm_prices) > 10 and len(itm1_prices) > 10 and len(otm1_prices) > 10:
                # Calculate divergence metrics
                recent_atm = np.mean(atm_prices[-10:])
                recent_itm1 = np.mean(itm1_prices[-10:])
                recent_otm1 = np.mean(otm1_prices[-10:])
                
                # Divergence from ATM
                itm1_divergence = (recent_itm1 - recent_atm) / recent_atm if recent_atm > 0 else 0
                otm1_divergence = (recent_otm1 - recent_atm) / recent_atm if recent_atm > 0 else 0
                
                # Feature 0-1: Divergence levels
                features[0] = recent_atm if abs(itm1_divergence) > self.divergence_threshold else 0
                features[1] = recent_itm1 if abs(itm1_divergence) > self.divergence_threshold else 0
                features[2] = recent_otm1 if abs(otm1_divergence) > self.divergence_threshold else 0
                
                # Feature 3-5: Divergence strengths
                features[3] = min(1.0, abs(itm1_divergence) / 0.1)
                features[4] = min(1.0, abs(otm1_divergence) / 0.1)
                features[5] = 1.0 if itm1_divergence > otm1_divergence else 0.0  # Directional bias
        
        return features
    
    def detect_straddle_momentum_exhaustion(self, component_data: Dict[str, Any]) -> np.ndarray:
        """
        Detect momentum exhaustion in straddle prices
        
        Returns:
            4-feature array
        """
        features = np.zeros(4)
        
        if "atm_straddle_prices" in component_data:
            prices = np.array(component_data["atm_straddle_prices"])
            
            if len(prices) > self.momentum_period * 2:
                # Calculate momentum
                momentum = np.diff(prices)
                
                # Smooth momentum
                smoothed_momentum = pd.Series(momentum).rolling(self.momentum_period).mean().values
                
                # Find exhaustion points
                for i in range(self.momentum_period, len(smoothed_momentum) - 1):
                    prev_momentum = smoothed_momentum[i - 5:i].mean()
                    curr_momentum = smoothed_momentum[i]
                    
                    # Bullish exhaustion
                    if prev_momentum > self.exhaustion_threshold and abs(curr_momentum) < 0.1:
                        features[0] = prices[i]  # Resistance level
                        features[1] = min(1.0, prev_momentum)  # Exhaustion strength
                        break
                
                # Bearish exhaustion
                for i in range(self.momentum_period, len(smoothed_momentum) - 1):
                    prev_momentum = smoothed_momentum[i - 5:i].mean()
                    curr_momentum = smoothed_momentum[i]
                    
                    if prev_momentum < -self.exhaustion_threshold and abs(curr_momentum) < 0.1:
                        features[2] = prices[i]  # Support level
                        features[3] = min(1.0, abs(prev_momentum))  # Exhaustion strength
                        break
        
        return features
    
    def detect_greeks_oi_confluence(
        self,
        greeks_data: Dict[str, Any],
        options_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Detect confluence between high gamma and high OI (pin risk)
        
        Returns:
            6-feature array
        """
        features = np.zeros(6)
        
        if "gamma" in greeks_data and all(col in options_data.columns for col in ["ce_oi", "pe_oi"]):
            gamma_values = greeks_data.get("gamma", {})
            
            # Find high gamma strikes
            high_gamma_strikes = []
            for strike, gamma in gamma_values.items():
                if gamma > self.gamma_threshold:
                    high_gamma_strikes.append(strike)
            
            # Check OI at high gamma strikes
            confluence_levels = []
            for strike in high_gamma_strikes[:10]:
                strike_data = options_data[options_data.get("strike", options_data.get("atm_strike")) == strike]
                if len(strike_data) > 0:
                    total_oi = strike_data["ce_oi"].sum() + strike_data["pe_oi"].sum()
                    gamma = gamma_values.get(strike, 0)
                    
                    # Pin risk score
                    pin_risk = gamma * total_oi / 1000000
                    confluence_levels.append((strike, pin_risk))
            
            # Sort by pin risk
            confluence_levels.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top 3 pin risk levels
            for i, (strike, pin_risk) in enumerate(confluence_levels[:3]):
                if i < 3:
                    features[i] = strike
                    features[i + 3] = min(1.0, pin_risk)
        
        return features
    
    def detect_iv_skew_asymmetry(self, options_data: pd.DataFrame) -> np.ndarray:
        """
        Detect S&R from extreme IV skew asymmetry
        
        Returns:
            4-feature array
        """
        features = np.zeros(4)
        
        if "ce_iv" in options_data.columns and "pe_iv" in options_data.columns:
            # Calculate skew
            skew = options_data["pe_iv"] - options_data["ce_iv"]
            
            # Find extreme skew points
            skew_percentile = np.percentile(skew.abs(), self.skew_percentile)
            extreme_skew = options_data[skew.abs() > skew_percentile]
            
            if len(extreme_skew) > 0:
                # Extreme put skew (support)
                put_skew = extreme_skew[skew > 0].nlargest(1, "pe_iv")
                if len(put_skew) > 0:
                    row = put_skew.iloc[0]
                    features[0] = row.get("strike", row.get("atm_strike", 0))
                    features[1] = min(1.0, abs(skew.iloc[put_skew.index[0]]) / 10)
                
                # Extreme call skew (resistance)
                call_skew = extreme_skew[skew < 0].nsmallest(1, "ce_iv")
                if len(call_skew) > 0:
                    row = call_skew.iloc[0]
                    features[2] = row.get("strike", row.get("atm_strike", 0))
                    features[3] = min(1.0, abs(skew.iloc[call_skew.index[0]]) / 10)
        
        return features
    
    def create_feature_vector(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine all advanced features into a single vector
        
        Returns:
            48-feature vector
        """
        all_features = []
        
        # Add features in consistent order
        feature_order = [
            ("oi_concentration_walls", 8),
            ("max_pain_migration", 6),
            ("oi_flow_velocity", 6),
            ("volume_weighted_oi", 8),
            ("triple_divergence", 6),
            ("momentum_exhaustion", 4),
            ("greeks_oi_confluence", 6),
            ("iv_skew_asymmetry", 4)
        ]
        
        for feature_name, expected_size in feature_order:
            if feature_name in feature_dict:
                features = feature_dict[feature_name]
                if len(features) != expected_size:
                    # Pad or truncate to expected size
                    if len(features) < expected_size:
                        features = np.pad(features, (0, expected_size - len(features)))
                    else:
                        features = features[:expected_size]
                all_features.extend(features)
            else:
                # Add zeros if feature not available
                all_features.extend(np.zeros(expected_size))
        
        return np.array(all_features)