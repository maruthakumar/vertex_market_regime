"""
Term Structure Analyzer - Volatility Term Structure Analysis
===========================================================

Analyzes the volatility term structure for trading signals and regime detection.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TermStructureAnalyzer:
    """
    Volatility term structure analysis
    
    Features:
    - Term structure slope analysis
    - Contango/backwardation detection
    - Historical comparison
    - Trading signal generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Term Structure Analyzer"""
        self.contango_threshold = config.get('contango_threshold', 0.02)
        self.backwardation_threshold = config.get('backwardation_threshold', -0.02)
        
        # History tracking
        self.term_structure_history = []
        
        logger.info("TermStructureAnalyzer initialized")
    
    def analyze_term_structure(self, iv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility term structure"""
        try:
            results = {
                'term_structure_slope': 0.0,
                'structure_type': 'flat',
                'signals': [],
                'regime': 'normal',
                'steepness': 0.0
            }
            
            # Calculate term structure
            term_structure = self._build_term_structure(iv_data)
            
            # Analyze slope
            results['term_structure_slope'] = self._calculate_slope(term_structure)
            
            # Classify structure type
            results['structure_type'] = self._classify_structure_type(results['term_structure_slope'])
            
            # Generate signals
            results['signals'] = self._generate_term_structure_signals(results)
            
            # Update history
            self._update_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing term structure: {e}")
            return {'structure_type': 'unknown'}
    
    def _build_term_structure(self, iv_data: pd.DataFrame) -> Dict[str, float]:
        """Build the term structure from IV data"""
        try:
            # Group by DTE and calculate average IV
            if 'dte' in iv_data.columns:
                term_structure = iv_data.groupby('dte')['iv'].mean().to_dict()
            else:
                # Fallback to single point
                term_structure = {30: iv_data['iv'].mean()}
            
            return term_structure
            
        except Exception as e:
            logger.error(f"Error building term structure: {e}")
            return {}
    
    def _calculate_slope(self, term_structure: Dict[str, float]) -> float:
        """Calculate term structure slope"""
        try:
            if len(term_structure) < 2:
                return 0.0
            
            # Sort by expiry
            sorted_points = sorted(term_structure.items())
            if len(sorted_points) < 2:
                return 0.0
            
            # Calculate slope between first and last points
            first_dte, first_iv = sorted_points[0]
            last_dte, last_iv = sorted_points[-1]
            
            if last_dte != first_dte:
                slope = (last_iv - first_iv) / (last_dte - first_dte)
                return float(slope)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            return 0.0
    
    def _classify_structure_type(self, slope: float) -> str:
        """Classify term structure type"""
        if slope > self.contango_threshold:
            return 'contango'
        elif slope < self.backwardation_threshold:
            return 'backwardation'
        else:
            return 'flat'
    
    def _generate_term_structure_signals(self, results: Dict[str, Any]) -> List[str]:
        """Generate trading signals from term structure"""
        signals = []
        
        structure_type = results['structure_type']
        slope = results['term_structure_slope']
        
        if structure_type == 'contango' and abs(slope) > 0.05:
            signals.append('strong_contango')
        elif structure_type == 'backwardation' and abs(slope) > 0.05:
            signals.append('strong_backwardation')
        
        return signals
    
    def _update_history(self, results: Dict[str, Any]):
        """Update term structure history"""
        try:
            self.term_structure_history.append({
                'timestamp': datetime.now(),
                'slope': results['term_structure_slope'],
                'type': results['structure_type']
            })
            
            # Keep only recent history
            if len(self.term_structure_history) > 100:
                self.term_structure_history = self.term_structure_history[-100:]
                
        except Exception as e:
            logger.error(f"Error updating history: {e}")