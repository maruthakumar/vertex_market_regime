"""
Straddle-Based Level Detection Engine
Integrates Component 1 (Triple Straddle) and Component 3 (Cumulative ATM±7)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StraddleLevelDetector:
    """
    Detects support/resistance levels from straddle price patterns
    Integrates Component 1 and Component 3 analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize straddle level detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Component 1 parameters (10-parameter system)
        self.component_1_params = {
            "atm_weight": config.get("atm_weight", 0.10),
            "itm1_weight": config.get("itm1_weight", 0.10),
            "otm1_weight": config.get("otm1_weight", 0.10),
            "ce_atm_weight": config.get("ce_atm_weight", 0.10),
            "ce_itm1_weight": config.get("ce_itm1_weight", 0.10),
            "ce_otm1_weight": config.get("ce_otm1_weight", 0.10),
            "pe_atm_weight": config.get("pe_atm_weight", 0.10),
            "pe_itm1_weight": config.get("pe_itm1_weight", 0.10),
            "pe_otm1_weight": config.get("pe_otm1_weight", 0.10),
            "correlation_weight": config.get("correlation_weight", 0.10)
        }
        
        # Component 3 parameters
        self.component_3_params = {
            "cumulative_ce_weight": config.get("cumulative_ce_weight", 0.30),
            "cumulative_pe_weight": config.get("cumulative_pe_weight", 0.30),
            "combined_straddle_weight": config.get("combined_straddle_weight", 0.40),
            "rolling_5min_weight": config.get("rolling_5min_weight", 0.35),
            "rolling_15min_weight": config.get("rolling_15min_weight", 0.20),
            "rolling_3min_weight": config.get("rolling_3min_weight", 0.25),
            "rolling_10min_weight": config.get("rolling_10min_weight", 0.20)
        }
        
        # EMA periods for overlay analysis
        self.ema_periods = [20, 50, 100, 200]
        
        # Multi-timeframe settings
        self.timeframes = {
            "intraday": ["5min", "15min", "30min", "60min"],
            "daily": ["1D"],
            "weekly": ["1W"],
            "monthly": ["1M"]
        }
        
        logger.info("Initialized StraddleLevelDetector with Component 1 & 3 integration")
    
    def detect_component_1_levels(
        self,
        straddle_data: pd.DataFrame,
        component_1_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from Component 1 triple straddle system
        
        Args:
            straddle_data: DataFrame with straddle prices
            component_1_analysis: Optional pre-computed Component 1 analysis
            
        Returns:
            List of detected levels from Component 1
        """
        levels = []
        
        # Extract ATM straddle levels
        atm_levels = self._detect_atm_straddle_levels(straddle_data, component_1_analysis)
        levels.extend(atm_levels)
        
        # Extract ITM1 straddle levels (Bullish bias)
        itm1_levels = self._detect_itm1_straddle_levels(straddle_data, component_1_analysis)
        levels.extend(itm1_levels)
        
        # Extract OTM1 straddle levels (Bearish bias)
        otm1_levels = self._detect_otm1_straddle_levels(straddle_data, component_1_analysis)
        levels.extend(otm1_levels)
        
        # Apply EMA overlay analysis
        ema_levels = self._detect_ema_overlay_levels(straddle_data)
        levels.extend(ema_levels)
        
        # Apply VWAP analysis
        vwap_levels = self._detect_vwap_levels(straddle_data)
        levels.extend(vwap_levels)
        
        # Apply pivot point analysis
        pivot_levels = self._detect_pivot_levels(straddle_data)
        levels.extend(pivot_levels)
        
        return levels
    
    def detect_component_3_levels(
        self,
        cumulative_data: pd.DataFrame,
        component_3_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from Component 3 cumulative ATM±7 system
        
        Args:
            cumulative_data: DataFrame with cumulative ATM±7 data
            component_3_analysis: Optional pre-computed Component 3 analysis
            
        Returns:
            List of detected levels from Component 3
        """
        levels = []
        
        # Extract cumulative CE levels (Resistance)
        ce_levels = self._detect_cumulative_ce_levels(cumulative_data, component_3_analysis)
        levels.extend(ce_levels)
        
        # Extract cumulative PE levels (Support)
        pe_levels = self._detect_cumulative_pe_levels(cumulative_data, component_3_analysis)
        levels.extend(pe_levels)
        
        # Extract combined straddle levels
        combined_levels = self._detect_combined_straddle_levels(cumulative_data, component_3_analysis)
        levels.extend(combined_levels)
        
        # Apply rolling timeframe analysis
        rolling_levels = self._detect_rolling_timeframe_levels(cumulative_data)
        levels.extend(rolling_levels)
        
        return levels
    
    def _detect_atm_straddle_levels(
        self,
        straddle_data: pd.DataFrame,
        component_1_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from ATM straddle prices
        
        Returns:
            List of ATM straddle-based levels
        """
        levels = []
        
        if "atm_straddle" in straddle_data.columns:
            atm_prices = straddle_data["atm_straddle"].values
            
            # Find local extrema
            extrema = self._find_local_extrema(atm_prices)
            
            for idx, price, is_max in extrema:
                levels.append({
                    "price": price,
                    "source": "component_1_atm",
                    "type": "resistance" if is_max else "support",
                    "strength": self.component_1_params["atm_weight"],
                    "timestamp": straddle_data.index[idx] if hasattr(straddle_data, 'index') else idx,
                    "method": "component_1_straddle"
                })
        
        return levels
    
    def _detect_itm1_straddle_levels(
        self,
        straddle_data: pd.DataFrame,
        component_1_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from ITM1 straddle prices (Bullish bias indicator)
        ITM1 rising = Bullish market, forms support levels
        
        Returns:
            List of ITM1 straddle-based levels
        """
        levels = []
        
        if "itm1_straddle" in straddle_data.columns:
            itm1_prices = straddle_data["itm1_straddle"].values
            
            # Find local extrema
            extrema = self._find_local_extrema(itm1_prices)
            
            for idx, price, is_max in extrema:
                # ITM1 local minima form support (bullish accumulation)
                # ITM1 local maxima form resistance (bullish exhaustion)
                levels.append({
                    "price": price,
                    "source": "component_1_itm1",
                    "type": "resistance" if is_max else "support",
                    "strength": self.component_1_params["itm1_weight"],
                    "timestamp": straddle_data.index[idx] if hasattr(straddle_data, 'index') else idx,
                    "method": "component_1_straddle",
                    "bias": "bullish"
                })
        
        return levels
    
    def _detect_otm1_straddle_levels(
        self,
        straddle_data: pd.DataFrame,
        component_1_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from OTM1 straddle prices (Bearish bias indicator)
        OTM1 rising = Bearish market, forms resistance levels
        
        Returns:
            List of OTM1 straddle-based levels
        """
        levels = []
        
        if "otm1_straddle" in straddle_data.columns:
            otm1_prices = straddle_data["otm1_straddle"].values
            
            # Find local extrema
            extrema = self._find_local_extrema(otm1_prices)
            
            for idx, price, is_max in extrema:
                # OTM1 local maxima form resistance (bearish pressure)
                # OTM1 local minima form support (bearish exhaustion)
                levels.append({
                    "price": price,
                    "source": "component_1_otm1",
                    "type": "resistance" if is_max else "support",
                    "strength": self.component_1_params["otm1_weight"],
                    "timestamp": straddle_data.index[idx] if hasattr(straddle_data, 'index') else idx,
                    "method": "component_1_straddle",
                    "bias": "bearish"
                })
        
        return levels
    
    def _detect_ema_overlay_levels(self, straddle_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect levels from EMA overlay on straddle prices
        
        Returns:
            List of EMA-based levels
        """
        levels = []
        
        for col in ["atm_straddle", "itm1_straddle", "otm1_straddle"]:
            if col in straddle_data.columns:
                prices = straddle_data[col]
                current_price = prices.iloc[-1] if len(prices) > 0 else 0
                
                for period in self.ema_periods:
                    if len(prices) >= period:
                        ema = prices.ewm(span=period, adjust=False).mean()
                        ema_value = ema.iloc[-1]
                        
                        # EMA acts as support if price above, resistance if below
                        level_type = "support" if current_price > ema_value else "resistance"
                        
                        levels.append({
                            "price": ema_value,
                            "source": f"ema_{period}_{col}",
                            "type": level_type,
                            "strength": 0.7,
                            "method": "component_1_ema",
                            "straddle_type": col
                        })
        
        return levels
    
    def _detect_vwap_levels(self, straddle_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect levels from VWAP on straddle price/volume
        
        Returns:
            List of VWAP-based levels
        """
        levels = []
        
        for col in ["atm_straddle", "itm1_straddle", "otm1_straddle"]:
            if col in straddle_data.columns and "volume" in straddle_data.columns:
                prices = straddle_data[col]
                volumes = straddle_data["volume"]
                
                if len(prices) > 0 and volumes.sum() > 0:
                    # Calculate VWAP
                    cumulative_pv = (prices * volumes).cumsum()
                    cumulative_volume = volumes.cumsum()
                    vwap = cumulative_pv / cumulative_volume
                    
                    # Current VWAP
                    current_vwap = vwap.iloc[-1]
                    current_price = prices.iloc[-1]
                    
                    # Previous day VWAP (if available)
                    if len(vwap) > 390:  # Assuming 390 5-min bars per day
                        prev_vwap = vwap.iloc[-390]
                        
                        levels.append({
                            "price": prev_vwap,
                            "source": f"prev_vwap_{col}",
                            "type": "support" if current_price > prev_vwap else "resistance",
                            "strength": 0.8,
                            "method": "component_1_vwap",
                            "straddle_type": col
                        })
                    
                    # Current VWAP
                    levels.append({
                        "price": current_vwap,
                        "source": f"vwap_{col}",
                        "type": "support" if current_price > current_vwap else "resistance",
                        "strength": 0.9,
                        "method": "component_1_vwap",
                        "straddle_type": col
                    })
        
        return levels
    
    def _detect_pivot_levels(self, straddle_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect pivot point levels from straddle prices
        
        Returns:
            List of pivot-based levels
        """
        levels = []
        
        for col in ["atm_straddle", "itm1_straddle", "otm1_straddle"]:
            if col in straddle_data.columns:
                if len(straddle_data) > 0:
                    # Get daily high, low, close for straddle
                    high = straddle_data[col].max()
                    low = straddle_data[col].min()
                    close = straddle_data[col].iloc[-1]
                    
                    # Calculate pivot points
                    pivot = (high + low + close) / 3
                    
                    # Resistance levels
                    r1 = 2 * pivot - low
                    r2 = pivot + (high - low)
                    r3 = high + 2 * (pivot - low)
                    
                    # Support levels
                    s1 = 2 * pivot - high
                    s2 = pivot - (high - low)
                    s3 = low - 2 * (high - pivot)
                    
                    # Add pivot levels
                    pivot_points = [
                        (r3, "R3", "resistance", 0.6),
                        (r2, "R2", "resistance", 0.8),
                        (r1, "R1", "resistance", 1.0),
                        (pivot, "PP", "neutral", 1.0),
                        (s1, "S1", "support", 1.0),
                        (s2, "S2", "support", 0.8),
                        (s3, "S3", "support", 0.6)
                    ]
                    
                    for price, label, level_type, strength in pivot_points:
                        levels.append({
                            "price": price,
                            "source": f"{label}_{col}",
                            "type": level_type,
                            "strength": strength,
                            "method": "component_1_pivot",
                            "straddle_type": col
                        })
        
        return levels
    
    def _detect_cumulative_ce_levels(
        self,
        cumulative_data: pd.DataFrame,
        component_3_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect resistance levels from cumulative CE across ATM±7
        
        Returns:
            List of cumulative CE-based resistance levels
        """
        levels = []
        
        if "cumulative_ce" in cumulative_data.columns:
            ce_prices = cumulative_data["cumulative_ce"].values
            
            # Find local maxima (resistance points)
            extrema = self._find_local_extrema(ce_prices)
            
            for idx, price, is_max in extrema:
                if is_max:  # CE accumulation forms resistance
                    levels.append({
                        "price": price,
                        "source": "component_3_ce",
                        "type": "resistance",
                        "strength": self.component_3_params["cumulative_ce_weight"],
                        "timestamp": cumulative_data.index[idx] if hasattr(cumulative_data, 'index') else idx,
                        "method": "component_3_cumulative"
                    })
        
        return levels
    
    def _detect_cumulative_pe_levels(
        self,
        cumulative_data: pd.DataFrame,
        component_3_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect support levels from cumulative PE across ATM±7
        
        Returns:
            List of cumulative PE-based support levels
        """
        levels = []
        
        if "cumulative_pe" in cumulative_data.columns:
            pe_prices = cumulative_data["cumulative_pe"].values
            
            # Find local maxima (support points)
            extrema = self._find_local_extrema(pe_prices)
            
            for idx, price, is_max in extrema:
                if is_max:  # PE accumulation forms support
                    levels.append({
                        "price": price,
                        "source": "component_3_pe",
                        "type": "support",
                        "strength": self.component_3_params["cumulative_pe_weight"],
                        "timestamp": cumulative_data.index[idx] if hasattr(cumulative_data, 'index') else idx,
                        "method": "component_3_cumulative"
                    })
        
        return levels
    
    def _detect_combined_straddle_levels(
        self,
        cumulative_data: pd.DataFrame,
        component_3_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from combined cumulative straddle
        
        Returns:
            List of combined straddle levels
        """
        levels = []
        
        if "cumulative_straddle" in cumulative_data.columns:
            straddle_prices = cumulative_data["cumulative_straddle"].values
            
            # Find all extrema
            extrema = self._find_local_extrema(straddle_prices)
            
            for idx, price, is_max in extrema:
                levels.append({
                    "price": price,
                    "source": "component_3_combined",
                    "type": "resistance" if is_max else "support",
                    "strength": self.component_3_params["combined_straddle_weight"],
                    "timestamp": cumulative_data.index[idx] if hasattr(cumulative_data, 'index') else idx,
                    "method": "component_3_cumulative"
                })
        
        return levels
    
    def _detect_rolling_timeframe_levels(
        self,
        cumulative_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from rolling timeframe analysis
        
        Returns:
            List of rolling timeframe-based levels
        """
        levels = []
        
        # Define rolling windows (in periods)
        rolling_windows = {
            "5min": 5,
            "15min": 15,
            "3min": 3,
            "10min": 10
        }
        
        for window_name, window_size in rolling_windows.items():
            if "cumulative_straddle" in cumulative_data.columns:
                if len(cumulative_data) >= window_size:
                    # Calculate rolling mean
                    rolling_mean = cumulative_data["cumulative_straddle"].rolling(window_size).mean()
                    
                    # Find extrema in rolling mean
                    if len(rolling_mean.dropna()) > 0:
                        extrema = self._find_local_extrema(rolling_mean.dropna().values)
                        
                        weight_key = f"rolling_{window_name}_weight"
                        weight = self.component_3_params.get(weight_key, 0.2)
                        
                        for idx, price, is_max in extrema:
                            levels.append({
                                "price": price,
                                "source": f"component_3_rolling_{window_name}",
                                "type": "resistance" if is_max else "support",
                                "strength": weight,
                                "method": "component_3_rolling",
                                "timeframe": window_name
                            })
        
        return levels
    
    def _find_local_extrema(
        self,
        prices: np.ndarray,
        window: int = 5
    ) -> List[Tuple[int, float, bool]]:
        """
        Find local extrema in price series
        
        Args:
            prices: Price array
            window: Window size for extrema detection
            
        Returns:
            List of (index, price, is_maximum) tuples
        """
        extrema = []
        
        if len(prices) < window * 2 + 1:
            return extrema
        
        for i in range(window, len(prices) - window):
            window_slice = prices[i - window:i + window + 1]
            
            if prices[i] == np.max(window_slice):
                extrema.append((i, prices[i], True))  # Maximum
            elif prices[i] == np.min(window_slice):
                extrema.append((i, prices[i], False))  # Minimum
        
        return extrema
    
    def calculate_multi_timeframe_consensus(
        self,
        levels: List[Dict[str, Any]],
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate consensus scores across multiple timeframes
        
        Args:
            levels: List of detected levels
            timeframes: Optional list of timeframes to consider
            
        Returns:
            Dictionary of consensus scores by price level
        """
        if timeframes is None:
            timeframes = ["5min", "15min", "30min", "60min", "daily"]
        
        # Group levels by price (with tolerance)
        price_groups = {}
        tolerance = 0.002  # 0.2% tolerance
        
        for level in levels:
            price = level["price"]
            
            # Find existing group or create new one
            matched = False
            for group_price in price_groups:
                if abs(price - group_price) / group_price < tolerance:
                    price_groups[group_price].append(level)
                    matched = True
                    break
            
            if not matched:
                price_groups[price] = [level]
        
        # Calculate consensus scores
        consensus_scores = {}
        
        for group_price, group_levels in price_groups.items():
            # Count unique timeframes
            unique_timeframes = set()
            for level in group_levels:
                if "timeframe" in level:
                    unique_timeframes.add(level["timeframe"])
            
            # Calculate consensus score
            consensus_score = len(unique_timeframes) / len(timeframes)
            consensus_scores[group_price] = consensus_score
        
        return consensus_scores