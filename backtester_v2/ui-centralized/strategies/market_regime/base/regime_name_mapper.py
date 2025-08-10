"""
Regime Name Mapper - Maps regime IDs to proper names from Excel configuration
===========================================================================

This module provides mapping between regime IDs (0-34) and their proper names
as defined in the Excel configuration. Handles all 35 regime classifications
including main regimes and sub-classifications.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RegimeNameMapper:
    """
    Maps regime IDs to proper descriptive names based on Excel configuration.
    
    Handles all 35 regime classifications with proper naming conventions:
    - Directional: Strong_Bullish, Bullish, Neutral, Bearish, Strong_Bearish
    - Volatility: High_Vol, Med_Vol, Low_Vol
    - Special: Sideways variations and transition states
    """
    
    def __init__(self):
        """Initialize regime name mappings"""
        
        # Complete mapping of all 35 regimes from Excel RegimeClassification sheet
        self.regime_id_to_name: Dict[int, str] = {
            # Strong Bullish variations (0-2)
            0: "Strong_Bullish_High_Vol",
            1: "Strong_Bullish_Med_Vol", 
            2: "Strong_Bullish_Low_Vol",
            
            # Bullish variations (3-5)
            3: "Bullish_High_Vol",
            4: "Bullish_Med_Vol",
            5: "Bullish_Low_Vol",
            
            # Neutral variations (6-8)
            6: "Neutral_High_Vol",
            7: "Neutral_Med_Vol",
            8: "Neutral_Low_Vol",
            
            # Bearish variations (9-11)
            9: "Bearish_High_Vol",
            10: "Bearish_Med_Vol",
            11: "Bearish_Low_Vol",
            
            # Strong Bearish variations (12-14)
            12: "Strong_Bearish_High_Vol",
            13: "Strong_Bearish_Med_Vol",
            14: "Strong_Bearish_Low_Vol",
            
            # Sideways variations (15-17)
            15: "Sideways_High_Vol",
            16: "Sideways_Med_Vol",
            17: "Sideways_Low_Vol",
            
            # Extended classifications (18-34)
            18: "Bullish_Transitioning_High_Vol",
            19: "Bearish_Transitioning_High_Vol",
            20: "Neutral_Expanding_Vol",
            21: "Neutral_Contracting_Vol",
            22: "Bullish_Momentum_Building",
            23: "Bearish_Momentum_Building",
            24: "Range_Bound_Tight",
            25: "Range_Bound_Wide",
            26: "Breakout_Pending_Up",
            27: "Breakout_Pending_Down",
            28: "Trend_Exhaustion_Bull",
            29: "Trend_Exhaustion_Bear",
            30: "Accumulation_Phase",
            31: "Distribution_Phase",
            32: "Volatility_Expansion",
            33: "Volatility_Contraction",
            34: "Market_Indecision"
        }
        
        # Reverse mapping for name to ID lookup
        self.regime_name_to_id: Dict[str, int] = {
            name: idx for idx, name in self.regime_id_to_name.items()
        }
        
        # Regime categories for grouping
        self.regime_categories = {
            "strong_bullish": [0, 1, 2],
            "bullish": [3, 4, 5, 18, 22, 26],
            "neutral": [6, 7, 8, 20, 21, 34],
            "bearish": [9, 10, 11, 19, 23, 27],
            "strong_bearish": [12, 13, 14],
            "sideways": [15, 16, 17, 24, 25],
            "transitional": [18, 19, 20, 21, 28, 29, 30, 31],
            "volatility_based": [32, 33]
        }
        
        # Regime colors for visualization (matching Excel configuration)
        self.regime_colors = {
            # Strong Bullish - Dark Green shades
            0: "#00FF00", 1: "#32CD32", 2: "#90EE90",
            # Bullish - Green shades  
            3: "#228B22", 4: "#3CB371", 5: "#98FB98",
            # Neutral - Yellow/Gray shades
            6: "#FFD700", 7: "#F0E68C", 8: "#FFFACD",
            # Bearish - Orange/Red shades
            9: "#FF4500", 10: "#FF6347", 11: "#FFA07A",
            # Strong Bearish - Dark Red shades
            12: "#8B0000", 13: "#DC143C", 14: "#CD5C5C",
            # Sideways - Blue shades
            15: "#0000FF", 16: "#4169E1", 17: "#87CEEB",
            # Extended classifications - Various
            18: "#7FFF00", 19: "#FF8C00", 20: "#DDA0DD",
            21: "#D8BFD8", 22: "#00FA9A", 23: "#FF69B4",
            24: "#B0C4DE", 25: "#778899", 26: "#ADFF2F",
            27: "#FFA500", 28: "#F0FFF0", 29: "#FFF0F5",
            30: "#E0FFFF", 31: "#FFE4E1", 32: "#FF1493",
            33: "#9370DB", 34: "#C0C0C0"
        }
        
        logger.info(f"RegimeNameMapper initialized with {len(self.regime_id_to_name)} regimes")
    
    def get_regime_name(self, regime_id: int) -> str:
        """
        Get regime name from ID
        
        Args:
            regime_id: Regime ID (0-34)
            
        Returns:
            str: Regime name or 'Unknown_Regime' if not found
        """
        if regime_id in self.regime_id_to_name:
            return self.regime_id_to_name[regime_id]
        else:
            logger.warning(f"Unknown regime ID: {regime_id}")
            return f"Unknown_Regime_{regime_id}"
    
    def get_regime_id(self, regime_name: str) -> Optional[int]:
        """
        Get regime ID from name
        
        Args:
            regime_name: Regime name string
            
        Returns:
            Optional[int]: Regime ID or None if not found
        """
        return self.regime_name_to_id.get(regime_name)
    
    def get_regime_category(self, regime_id: int) -> Optional[str]:
        """
        Get category for a regime ID
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Optional[str]: Category name or None
        """
        for category, ids in self.regime_categories.items():
            if regime_id in ids:
                return category
        return None
    
    def get_regime_color(self, regime_id: int) -> str:
        """
        Get color code for regime visualization
        
        Args:
            regime_id: Regime ID
            
        Returns:
            str: Hex color code
        """
        return self.regime_colors.get(regime_id, "#808080")  # Gray for unknown
    
    def get_regime_description(self, regime_id: int) -> str:
        """
        Get detailed description for a regime
        
        Args:
            regime_id: Regime ID
            
        Returns:
            str: Detailed regime description
        """
        descriptions = {
            0: "Strong uptrend with high volatility - aggressive longs",
            1: "Strong uptrend with medium volatility - confident longs",
            2: "Strong uptrend with low volatility - steady accumulation",
            3: "Uptrend with high volatility - cautious longs",
            4: "Uptrend with medium volatility - standard bullish",
            5: "Uptrend with low volatility - quiet accumulation",
            6: "Neutral market with high volatility - avoid directional",
            7: "Neutral market with medium volatility - range trading",
            8: "Neutral market with low volatility - wait for direction",
            9: "Downtrend with high volatility - cautious shorts",
            10: "Downtrend with medium volatility - standard bearish",
            11: "Downtrend with low volatility - quiet distribution",
            12: "Strong downtrend with high volatility - aggressive shorts",
            13: "Strong downtrend with medium volatility - confident shorts",
            14: "Strong downtrend with low volatility - steady decline",
            15: "Sideways market with high volatility - range extremes",
            16: "Sideways market with medium volatility - range trading",
            17: "Sideways market with low volatility - tight range",
            18: "Bullish transition with increasing volatility",
            19: "Bearish transition with increasing volatility",
            20: "Neutral market with expanding volatility",
            21: "Neutral market with contracting volatility",
            22: "Building bullish momentum - early trend",
            23: "Building bearish momentum - early decline",
            24: "Tight range-bound market - breakout pending",
            25: "Wide range-bound market - trade the range",
            26: "Upward breakout imminent - position for longs",
            27: "Downward breakout imminent - position for shorts",
            28: "Bull trend showing exhaustion signs",
            29: "Bear trend showing exhaustion signs",
            30: "Smart money accumulation phase",
            31: "Smart money distribution phase",
            32: "Volatility expanding rapidly - hedge positions",
            33: "Volatility contracting - prepare for move",
            34: "Market indecision - no clear direction"
        }
        return descriptions.get(regime_id, "No description available")
    
    def get_all_regime_names(self) -> List[str]:
        """Get list of all regime names"""
        return list(self.regime_name_to_id.keys())
    
    def get_regime_count(self) -> int:
        """Get total number of regimes"""
        return len(self.regime_id_to_name)
    
    def get_regime_info(self, regime_id: int) -> Dict[str, any]:
        """
        Get complete information for a regime
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dict containing name, category, color, and description
        """
        return {
            'id': regime_id,
            'name': self.get_regime_name(regime_id),
            'category': self.get_regime_category(regime_id),
            'color': self.get_regime_color(regime_id),
            'description': self.get_regime_description(regime_id)
        }
    
    def validate_regime_sequence(self, regime_ids: List[int]) -> List[Tuple[int, str]]:
        """
        Validate a sequence of regime IDs and return any issues
        
        Args:
            regime_ids: List of regime IDs
            
        Returns:
            List of (index, issue) tuples
        """
        issues = []
        
        for i, regime_id in enumerate(regime_ids):
            if regime_id not in self.regime_id_to_name:
                issues.append((i, f"Invalid regime ID: {regime_id}"))
            
            # Check for impossible transitions
            if i > 0:
                prev_id = regime_ids[i-1]
                if self._is_impossible_transition(prev_id, regime_id):
                    issues.append((i, f"Impossible transition: {prev_id} -> {regime_id}"))
        
        return issues
    
    def _is_impossible_transition(self, from_id: int, to_id: int) -> bool:
        """
        Check if a regime transition is impossible
        
        Args:
            from_id: Source regime ID
            to_id: Target regime ID
            
        Returns:
            bool: True if transition is impossible
        """
        # Example: Can't jump from Strong_Bullish_Low_Vol to Strong_Bearish_Low_Vol
        from_name = self.get_regime_name(from_id)
        to_name = self.get_regime_name(to_id)
        
        # Impossible to jump between extreme opposites without transition
        impossible_pairs = [
            ("Strong_Bullish", "Strong_Bearish"),
            ("Strong_Bearish", "Strong_Bullish"),
        ]
        
        for pair in impossible_pairs:
            if (pair[0] in from_name and pair[1] in to_name) or \
               (pair[1] in from_name and pair[0] in to_name):
                if "Transitioning" not in to_name:
                    return True
        
        return False


# Singleton instance
_regime_mapper = None

def get_regime_mapper() -> RegimeNameMapper:
    """Get singleton instance of RegimeNameMapper"""
    global _regime_mapper
    if _regime_mapper is None:
        _regime_mapper = RegimeNameMapper()
    return _regime_mapper