"""
Strike Arbitrage - Strike-based Arbitrage Detection
==================================================

Detects arbitrage opportunities across strikes.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StrikeArbitrage:
    """Strike-based arbitrage detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.butterfly_threshold = config.get('butterfly_threshold', 0.001)
        logger.info("StrikeArbitrage initialized")
    
    def detect_strike_arbitrage(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect arbitrage opportunities across strikes"""
        try:
            opportunities = []
            
            # Group by expiry and option type
            for (dte, option_type), group in option_data.groupby(['dte', 'option_type']):
                if len(group) < 3:
                    continue
                
                # Sort by strike
                group = group.sort_values('strike')
                
                # Check butterfly spread arbitrage
                butterfly_violations = self._check_butterfly_arbitrage(group)
                opportunities.extend(butterfly_violations)
            
            return {
                'arbitrage_opportunities': opportunities,
                'total_opportunities': len(opportunities)
            }
            
        except Exception as e:
            logger.error(f"Error detecting strike arbitrage: {e}")
            return {'arbitrage_opportunities': [], 'total_opportunities': 0}
    
    def _check_butterfly_arbitrage(self, option_group: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for butterfly spread arbitrage violations"""
        violations = []
        
        try:
            for i in range(1, len(option_group) - 1):
                left = option_group.iloc[i-1]
                center = option_group.iloc[i]
                right = option_group.iloc[i+1]
                
                # Butterfly spread: center should be convex combination
                expected_center_iv = (left['iv'] + right['iv']) / 2
                actual_center_iv = center['iv']
                
                violation = actual_center_iv - expected_center_iv
                
                if abs(violation) > self.butterfly_threshold:
                    violations.append({
                        'type': 'butterfly_arbitrage',
                        'center_strike': center['strike'],
                        'violation': float(violation),
                        'dte': center['dte'],
                        'option_type': center['option_type']
                    })
        
        except Exception as e:
            logger.error(f"Error checking butterfly arbitrage: {e}")
        
        return violations