"""
Vol Arbitrage Scanner - Comprehensive Volatility Arbitrage Scanner
=================================================================

Main scanner that coordinates all arbitrage detection components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class VolArbitrageScanner:
    """Comprehensive volatility arbitrage scanner"""
    
    def __init__(self, config: Dict[str, Any]):
        self.scan_enabled = config.get('scan_enabled', True)
        self.min_opportunity_size = config.get('min_opportunity_size', 0.01)
        logger.info("VolArbitrageScanner initialized")
    
    def scan_arbitrage_opportunities(self, 
                                   calendar_results: Dict[str, Any],
                                   strike_results: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for all types of arbitrage opportunities"""
        try:
            all_opportunities = []
            
            # Combine calendar arbitrage opportunities
            if 'arbitrage_opportunities' in calendar_results:
                all_opportunities.extend(calendar_results['arbitrage_opportunities'])
            
            # Combine strike arbitrage opportunities
            if 'arbitrage_opportunities' in strike_results:
                all_opportunities.extend(strike_results['arbitrage_opportunities'])
            
            # Filter by minimum size
            significant_opportunities = [
                op for op in all_opportunities
                if op.get('severity', 0) > self.min_opportunity_size or
                   abs(op.get('violation', 0)) > self.min_opportunity_size
            ]
            
            # Rank opportunities
            ranked_opportunities = self._rank_opportunities(significant_opportunities)
            
            return {
                'total_opportunities': len(all_opportunities),
                'significant_opportunities': len(significant_opportunities),
                'ranked_opportunities': ranked_opportunities,
                'arbitrage_score': self._calculate_arbitrage_score(significant_opportunities)
            }
            
        except Exception as e:
            logger.error(f"Error scanning arbitrage opportunities: {e}")
            return {'total_opportunities': 0, 'arbitrage_score': 0.0}
    
    def _rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank arbitrage opportunities by attractiveness"""
        try:
            for op in opportunities:
                # Calculate attractiveness score
                severity = op.get('severity', abs(op.get('violation', 0)))
                op['attractiveness_score'] = severity
            
            # Sort by attractiveness
            return sorted(opportunities, key=lambda x: x.get('attractiveness_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error ranking opportunities: {e}")
            return opportunities
    
    def _calculate_arbitrage_score(self, opportunities: List[Dict[str, Any]]) -> float:
        """Calculate overall arbitrage score for the market"""
        try:
            if not opportunities:
                return 0.0
            
            # Sum of all opportunity severities
            total_severity = sum(
                op.get('severity', abs(op.get('violation', 0)))
                for op in opportunities
            )
            
            # Normalize by number of opportunities
            arbitrage_score = min(total_severity / len(opportunities), 1.0)
            
            return float(arbitrage_score)
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage score: {e}")
            return 0.0