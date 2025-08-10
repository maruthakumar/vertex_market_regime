"""
Calendar Arbitrage - Calendar Spread Arbitrage Detection
========================================================

Detects calendar spread arbitrage opportunities.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class CalendarArbitrage:
    """Calendar spread arbitrage detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_arbitrage_threshold = config.get('min_arbitrage_threshold', 0.02)
        logger.info("CalendarArbitrage initialized")
    
    def detect_calendar_arbitrage(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect calendar arbitrage opportunities"""
        try:
            opportunities = []
            
            # Group by strike
            for strike, strike_group in option_data.groupby('strike'):
                if len(strike_group) < 2:
                    continue
                
                # Sort by expiry
                strike_group = strike_group.sort_values('dte')
                
                # Check for arbitrage between adjacent expiries
                for i in range(len(strike_group) - 1):
                    near_option = strike_group.iloc[i]
                    far_option = strike_group.iloc[i + 1]
                    
                    # Calendar arbitrage check: far vol should be >= near vol
                    vol_diff = far_option['iv'] - near_option['iv']
                    
                    if vol_diff < -self.min_arbitrage_threshold:
                        opportunities.append({
                            'type': 'calendar_arbitrage',
                            'strike': strike,
                            'near_dte': near_option['dte'],
                            'far_dte': far_option['dte'],
                            'vol_difference': float(vol_diff),
                            'severity': abs(vol_diff)
                        })
            
            return {
                'arbitrage_opportunities': opportunities,
                'total_opportunities': len(opportunities),
                'max_severity': max([op['severity'] for op in opportunities]) if opportunities else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error detecting calendar arbitrage: {e}")
            return {'arbitrage_opportunities': [], 'total_opportunities': 0}