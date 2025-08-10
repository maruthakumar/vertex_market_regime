"""
Forward Vol Calculator - Forward Volatility Calculation
======================================================

Calculates forward volatilities from the term structure.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ForwardVolCalculator:
    """Forward volatility calculation"""
    
    def __init__(self, config: Dict[str, Any]):
        logger.info("ForwardVolCalculator initialized")
    
    def calculate_forward_vols(self, term_structure: Dict[str, float]) -> Dict[str, Any]:
        """Calculate forward volatilities"""
        try:
            forward_vols = {}
            
            sorted_points = sorted(term_structure.items())
            
            for i in range(1, len(sorted_points)):
                t1, vol1 = sorted_points[i-1]
                t2, vol2 = sorted_points[i]
                
                # Calculate forward vol
                if t2 > t1:
                    var1 = vol1 ** 2 * t1
                    var2 = vol2 ** 2 * t2
                    forward_var = (var2 - var1) / (t2 - t1)
                    
                    if forward_var > 0:
                        forward_vol = np.sqrt(forward_var)
                        forward_vols[f"{t1}_{t2}"] = float(forward_vol)
            
            return {
                'forward_vols': forward_vols,
                'calculation_success': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating forward vols: {e}")
            return {'calculation_success': False}