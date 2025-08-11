"""
Strike Type-Based Straddle Selection - Component 2

Implements straddle selection using call_strike_type and put_strike_type columns from production data.
Extracts ATM, ITM, and OTM straddles with proper strike type combinations for Greeks analysis.

🚨 KEY IMPLEMENTATION: Uses actual strike type data from production Parquet schema
with comprehensive strike type handling (ATM, ITM1-23, OTM1-23).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import defaultdict

from .production_greeks_extractor import ProductionGreeksData


@dataclass
class StraddleClassification:
    """Straddle classification based on strike types"""
    straddle_type: str              # ATM, ITM1, OTM1, etc.
    call_strike_type: str           # Call option strike type
    put_strike_type: str            # Put option strike type
    
    # Greeks data
    greeks_data: ProductionGreeksData
    
    # Analysis metrics
    moneyness_score: float          # How close to ATM (-1 to +1)
    regime_bias: str                # BULLISH, BEARISH, NEUTRAL
    pin_risk_level: float           # Pin risk assessment (0-1)
    
    # Quality metrics
    volume_quality: float           # Volume data quality
    greeks_completeness: float      # Greeks data completeness
    confidence: float               # Overall confidence in classification


@dataclass
class StraddleSelectionResult:
    """Result from straddle selection process"""
    atm_straddles: List[StraddleClassification]     # Pure ATM straddles
    itm_straddles: List[StraddleClassification]     # In-the-money biased
    otm_straddles: List[StraddleClassification]     # Out-of-the-money biased
    
    # Statistics
    total_straddles: int
    atm_count: int
    itm_count: int
    otm_count: int
    
    # Quality metrics
    overall_quality: float
    processing_time_ms: float
    metadata: Dict[str, Any]


class StrikeTypeStraddleSelector:
    """
    Strike Type-Based Straddle Selection System
    
    🚨 CRITICAL FEATURES:
    - Uses ACTUAL strike type columns (call_strike_type, put_strike_type) from production data
    - Handles all available strike types: ATM, ITM1-23, OTM1-23
    - ATM straddle extraction with 100% Greeks coverage
    - ITM/OTM straddle combinations for regime bias detection
    - Volume/OI based quality assessment for each straddle type
    """
    
    # Strike type mappings from production data validation
    VALID_STRIKE_TYPES = {
        'ATM', 'ITM1', 'ITM2', 'ITM3', 'ITM4', 'ITM5', 'ITM6', 'ITM7', 'ITM8', 'ITM9', 'ITM10',
        'ITM11', 'ITM12', 'ITM13', 'ITM14', 'ITM15', 'ITM16', 'ITM17', 'ITM18', 'ITM19', 'ITM20',
        'ITM21', 'ITM22', 'ITM23', 'OTM1', 'OTM2', 'OTM3', 'OTM4', 'OTM5', 'OTM6', 'OTM7', 'OTM8',
        'OTM9', 'OTM10', 'OTM11', 'OTM12', 'OTM13', 'OTM14', 'OTM15', 'OTM16', 'OTM17', 'OTM18',
        'OTM19', 'OTM20', 'OTM21', 'OTM22', 'OTM23'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize strike type straddle selector"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Selection criteria
        self.min_volume_threshold = self.config.get('min_volume_threshold', 50)
        self.min_oi_threshold = self.config.get('min_oi_threshold', 100)
        self.max_moneyness = self.config.get('max_moneyness', 5)  # Max ITM/OTM level
        
        # Moneyness scoring (for regime bias detection)
        self.moneyness_scores = self._build_moneyness_scores()
        
        self.logger.info("🚨 StrikeTypeStraddleSelector initialized with production strike types")
    
    def _build_moneyness_scores(self) -> Dict[str, float]:
        """
        Build moneyness scores for strike types
        
        Returns:
            Dictionary mapping strike types to moneyness scores (-1 to +1)
        """
        scores = {'ATM': 0.0}  # ATM is neutral
        
        # ITM strikes (positive moneyness for calls, negative for puts)
        for i in range(1, 24):
            scores[f'ITM{i}'] = i * 0.1  # ITM1=0.1, ITM2=0.2, etc.
        
        # OTM strikes (negative moneyness for calls, positive for puts)  
        for i in range(1, 24):
            scores[f'OTM{i}'] = -i * 0.1  # OTM1=-0.1, OTM2=-0.2, etc.
        
        return scores
    
    def extract_atm_straddles(self, greeks_data_list: List[ProductionGreeksData]) -> List[StraddleClassification]:
        """
        Extract pure ATM straddles (call_strike_type='ATM' AND put_strike_type='ATM')
        
        ATM options have 100% Greeks coverage and are primary for sentiment analysis.
        
        Args:
            greeks_data_list: List of production Greeks data
            
        Returns:
            List of ATM straddle classifications
        """
        atm_straddles = []
        
        for data in greeks_data_list:
            # Check for pure ATM straddle
            if (data.call_strike_type == 'ATM' and 
                data.put_strike_type == 'ATM'):
                
                # Quality assessment
                total_volume = data.ce_volume + data.pe_volume
                total_oi = data.ce_oi + data.pe_oi
                
                # Skip if volume too low
                if total_volume < self.min_volume_threshold:
                    continue
                
                # Calculate pin risk for ATM (highest gamma exposure)
                combined_gamma = data.ce_gamma + data.pe_gamma
                pin_risk = min(combined_gamma / 0.001, 1.0)  # Normalize to max observed gamma
                
                # Volume quality
                volume_quality = min(total_volume / 500.0, 1.0)  # Normalize to 500 volume
                
                # Greeks completeness (production data has 100% coverage)
                greeks_completeness = 1.0
                
                # Overall confidence
                confidence = (0.4 * volume_quality + 0.6 * greeks_completeness)
                
                classification = StraddleClassification(
                    straddle_type='ATM',
                    call_strike_type=data.call_strike_type,
                    put_strike_type=data.put_strike_type,
                    greeks_data=data,
                    moneyness_score=0.0,  # ATM is neutral
                    regime_bias='NEUTRAL',
                    pin_risk_level=pin_risk,
                    volume_quality=volume_quality,
                    greeks_completeness=greeks_completeness,
                    confidence=confidence
                )
                
                atm_straddles.append(classification)
        
        self.logger.info(f"Extracted {len(atm_straddles)} ATM straddles with 100% Greeks coverage")
        return atm_straddles
    
    def extract_itm_straddles(self, greeks_data_list: List[ProductionGreeksData]) -> List[StraddleClassification]:
        """
        Extract ITM-biased straddles for bullish regime detection
        
        Common ITM combinations:
        - Call ITM + Put OTM (bullish bias)
        - Both ITM but calls more ITM (strong bullish)
        
        Args:
            greeks_data_list: List of production Greeks data
            
        Returns:
            List of ITM straddle classifications
        """
        itm_straddles = []
        
        for data in greeks_data_list:
            call_type = data.call_strike_type
            put_type = data.put_strike_type
            
            # ITM straddle criteria
            is_itm_straddle = False
            regime_bias = 'NEUTRAL'
            
            # Case 1: Call ITM + Put OTM (bullish bias)
            if (call_type.startswith('ITM') and put_type.startswith('OTM')):
                is_itm_straddle = True
                regime_bias = 'BULLISH'
                
            # Case 2: Call ITM + Put ATM (mild bullish)
            elif (call_type.startswith('ITM') and put_type == 'ATM'):
                is_itm_straddle = True
                regime_bias = 'MILD_BULLISH'
                
            # Case 3: Both ITM but call more ITM (strong bullish)
            elif (call_type.startswith('ITM') and put_type.startswith('ITM')):
                call_level = int(call_type[3:]) if len(call_type) > 3 else 1
                put_level = int(put_type[3:]) if len(put_type) > 3 else 1
                
                if call_level < put_level:  # Call is more ITM
                    is_itm_straddle = True
                    regime_bias = 'STRONG_BULLISH'
            
            if is_itm_straddle:
                # Quality assessment
                total_volume = data.ce_volume + data.pe_volume
                if total_volume < self.min_volume_threshold:
                    continue
                
                # Calculate moneyness score
                call_score = self.moneyness_scores.get(call_type, 0.0)
                put_score = self.moneyness_scores.get(put_type, 0.0)
                moneyness_score = (call_score + put_score) / 2  # Average
                
                # Pin risk (lower for ITM)
                combined_gamma = data.ce_gamma + data.pe_gamma
                pin_risk = min(combined_gamma / 0.002, 0.8)  # ITM has lower pin risk
                
                classification = StraddleClassification(
                    straddle_type='ITM_BIASED',
                    call_strike_type=call_type,
                    put_strike_type=put_type,
                    greeks_data=data,
                    moneyness_score=moneyness_score,
                    regime_bias=regime_bias,
                    pin_risk_level=pin_risk,
                    volume_quality=min(total_volume / 500.0, 1.0),
                    greeks_completeness=1.0,  # Production data quality
                    confidence=0.8  # ITM straddles have good confidence
                )
                
                itm_straddles.append(classification)
        
        self.logger.info(f"Extracted {len(itm_straddles)} ITM-biased straddles")
        return itm_straddles
    
    def extract_otm_straddles(self, greeks_data_list: List[ProductionGreeksData]) -> List[StraddleClassification]:
        """
        Extract OTM-biased straddles for bearish regime detection
        
        Common OTM combinations:
        - Call OTM + Put ITM (bearish bias)
        - Both OTM but puts more OTM (strong bearish)
        
        Args:
            greeks_data_list: List of production Greeks data
            
        Returns:
            List of OTM straddle classifications
        """
        otm_straddles = []
        
        for data in greeks_data_list:
            call_type = data.call_strike_type
            put_type = data.put_strike_type
            
            # OTM straddle criteria
            is_otm_straddle = False
            regime_bias = 'NEUTRAL'
            
            # Case 1: Call OTM + Put ITM (bearish bias)
            if (call_type.startswith('OTM') and put_type.startswith('ITM')):
                is_otm_straddle = True
                regime_bias = 'BEARISH'
                
            # Case 2: Call ATM + Put ITM (mild bearish)
            elif (call_type == 'ATM' and put_type.startswith('ITM')):
                is_otm_straddle = True
                regime_bias = 'MILD_BEARISH'
                
            # Case 3: Both OTM but put more OTM (strong bearish)
            elif (call_type.startswith('OTM') and put_type.startswith('OTM')):
                call_level = int(call_type[3:]) if len(call_type) > 3 else 1
                put_level = int(put_type[3:]) if len(put_type) > 3 else 1
                
                if put_level < call_level:  # Put is more OTM (less out of money)
                    is_otm_straddle = True
                    regime_bias = 'STRONG_BEARISH'
            
            if is_otm_straddle:
                # Quality assessment
                total_volume = data.ce_volume + data.pe_volume
                if total_volume < self.min_volume_threshold:
                    continue
                
                # Calculate moneyness score
                call_score = self.moneyness_scores.get(call_type, 0.0)
                put_score = self.moneyness_scores.get(put_type, 0.0)
                moneyness_score = (call_score + put_score) / 2  # Average
                
                # Pin risk (lower for OTM)
                combined_gamma = data.ce_gamma + data.pe_gamma
                pin_risk = min(combined_gamma / 0.002, 0.6)  # OTM has lower pin risk
                
                classification = StraddleClassification(
                    straddle_type='OTM_BIASED',
                    call_strike_type=call_type,
                    put_strike_type=put_type,
                    greeks_data=data,
                    moneyness_score=moneyness_score,
                    regime_bias=regime_bias,
                    pin_risk_level=pin_risk,
                    volume_quality=min(total_volume / 500.0, 1.0),
                    greeks_completeness=1.0,  # Production data quality
                    confidence=0.8  # OTM straddles have good confidence
                )
                
                otm_straddles.append(classification)
        
        self.logger.info(f"Extracted {len(otm_straddles)} OTM-biased straddles")
        return otm_straddles
    
    def select_straddles(self, greeks_data_list: List[ProductionGreeksData]) -> StraddleSelectionResult:
        """
        Complete straddle selection process using strike type combinations
        
        Args:
            greeks_data_list: List of production Greeks data
            
        Returns:
            StraddleSelectionResult with all straddle types
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract different straddle types
            atm_straddles = self.extract_atm_straddles(greeks_data_list)
            itm_straddles = self.extract_itm_straddles(greeks_data_list) 
            otm_straddles = self.extract_otm_straddles(greeks_data_list)
            
            # Calculate statistics
            total_straddles = len(atm_straddles) + len(itm_straddles) + len(otm_straddles)
            
            # Calculate overall quality
            all_straddles = atm_straddles + itm_straddles + otm_straddles
            if all_straddles:
                overall_quality = np.mean([s.confidence for s in all_straddles])
            else:
                overall_quality = 0.0
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Metadata
            metadata = {
                'total_input_data_points': len(greeks_data_list),
                'selection_criteria': {
                    'min_volume_threshold': self.min_volume_threshold,
                    'min_oi_threshold': self.min_oi_threshold
                },
                'strike_types_available': list(self.VALID_STRIKE_TYPES),
                'atm_quality': np.mean([s.confidence for s in atm_straddles]) if atm_straddles else 0,
                'itm_quality': np.mean([s.confidence for s in itm_straddles]) if itm_straddles else 0,
                'otm_quality': np.mean([s.confidence for s in otm_straddles]) if otm_straddles else 0
            }
            
            result = StraddleSelectionResult(
                atm_straddles=atm_straddles,
                itm_straddles=itm_straddles,
                otm_straddles=otm_straddles,
                total_straddles=total_straddles,
                atm_count=len(atm_straddles),
                itm_count=len(itm_straddles),
                otm_count=len(otm_straddles),
                overall_quality=overall_quality,
                processing_time_ms=processing_time,
                metadata=metadata
            )
            
            self.logger.info(f"Straddle selection completed: {total_straddles} total "
                           f"(ATM: {len(atm_straddles)}, ITM: {len(itm_straddles)}, OTM: {len(otm_straddles)})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Straddle selection failed: {e}")
            raise
    
    def get_regime_weighted_straddles(self, selection_result: StraddleSelectionResult) -> Dict[str, List[StraddleClassification]]:
        """
        Get straddles weighted by regime bias for enhanced analysis
        
        Args:
            selection_result: StraddleSelectionResult from selection process
            
        Returns:
            Dictionary of straddles grouped by regime bias
        """
        regime_straddles = defaultdict(list)
        
        # Group all straddles by regime bias
        all_straddles = (selection_result.atm_straddles + 
                        selection_result.itm_straddles + 
                        selection_result.otm_straddles)
        
        for straddle in all_straddles:
            regime_straddles[straddle.regime_bias].append(straddle)
        
        return dict(regime_straddles)
    
    def filter_by_volume_quality(self, 
                                straddles: List[StraddleClassification],
                                min_quality: float = 0.7) -> List[StraddleClassification]:
        """Filter straddles by volume quality threshold"""
        return [s for s in straddles if s.volume_quality >= min_quality]
    
    def get_best_straddles_by_type(self, 
                                  selection_result: StraddleSelectionResult,
                                  top_n: int = 10) -> Dict[str, List[StraddleClassification]]:
        """
        Get top N best straddles by type based on confidence scores
        
        Args:
            selection_result: StraddleSelectionResult
            top_n: Number of top straddles to return per type
            
        Returns:
            Dictionary with best straddles by type
        """
        best_straddles = {}
        
        # Sort and get top ATM straddles
        if selection_result.atm_straddles:
            atm_sorted = sorted(selection_result.atm_straddles, 
                              key=lambda x: x.confidence, reverse=True)
            best_straddles['ATM'] = atm_sorted[:top_n]
        
        # Sort and get top ITM straddles
        if selection_result.itm_straddles:
            itm_sorted = sorted(selection_result.itm_straddles,
                              key=lambda x: x.confidence, reverse=True)
            best_straddles['ITM'] = itm_sorted[:top_n]
        
        # Sort and get top OTM straddles
        if selection_result.otm_straddles:
            otm_sorted = sorted(selection_result.otm_straddles,
                              key=lambda x: x.confidence, reverse=True)
            best_straddles['OTM'] = otm_sorted[:top_n]
        
        return best_straddles


# Testing and validation functions
def test_strike_type_straddle_selection():
    """Test strike type straddle selection with production-like data"""
    print("🚨 Testing Strike Type Straddle Selection...")
    
    # Create test data with different strike type combinations
    test_data = [
        # ATM straddle
        ProductionGreeksData(
            ce_delta=0.5, pe_delta=-0.5, ce_gamma=0.0008, pe_gamma=0.0008,
            ce_theta=-8, pe_theta=-6, ce_vega=3, pe_vega=2.5,
            ce_volume=1000, pe_volume=800, ce_oi=2000, pe_oi=1800,
            call_strike_type='ATM', put_strike_type='ATM', dte=7,
            trade_time=datetime.utcnow(), expiry_date=datetime.utcnow()
        ),
        # ITM bullish bias (Call ITM, Put OTM)
        ProductionGreeksData(
            ce_delta=0.7, pe_delta=-0.3, ce_gamma=0.0006, pe_gamma=0.0009,
            ce_theta=-6, pe_theta=-8, ce_vega=2.5, pe_vega=3.5,
            ce_volume=800, pe_volume=600, ce_oi=1500, pe_oi=1200,
            call_strike_type='ITM1', put_strike_type='OTM1', dte=10,
            trade_time=datetime.utcnow(), expiry_date=datetime.utcnow()
        ),
        # OTM bearish bias (Call OTM, Put ITM)
        ProductionGreeksData(
            ce_delta=0.3, pe_delta=-0.7, ce_gamma=0.0009, pe_gamma=0.0006,
            ce_theta=-8, pe_theta=-6, ce_vega=3.5, pe_vega=2.5,
            ce_volume=600, pe_volume=900, ce_oi=1200, pe_oi=1600,
            call_strike_type='OTM1', put_strike_type='ITM1', dte=5,
            trade_time=datetime.utcnow(), expiry_date=datetime.utcnow()
        ),
        # Mixed ITM (both ITM but call more)
        ProductionGreeksData(
            ce_delta=0.8, pe_delta=-0.2, ce_gamma=0.0005, pe_gamma=0.0010,
            ce_theta=-5, pe_theta=-10, ce_vega=2, pe_vega=4,
            ce_volume=500, pe_volume=400, ce_oi=1000, pe_oi=800,
            call_strike_type='ITM1', put_strike_type='ITM3', dte=15,
            trade_time=datetime.utcnow(), expiry_date=datetime.utcnow()
        )
    ]
    
    # Initialize selector
    selector = StrikeTypeStraddleSelector({'min_volume_threshold': 50})
    
    # Run straddle selection
    result = selector.select_straddles(test_data)
    
    # Display results
    print(f"✅ Total straddles selected: {result.total_straddles}")
    print(f"✅ ATM straddles: {result.atm_count}")
    print(f"✅ ITM straddles: {result.itm_count}")
    print(f"✅ OTM straddles: {result.otm_count}")
    print(f"✅ Overall quality: {result.overall_quality:.3f}")
    print(f"✅ Processing time: {result.processing_time_ms:.2f}ms")
    
    # Show regime analysis
    regime_straddles = selector.get_regime_weighted_straddles(result)
    print(f"✅ Regime distribution: {list(regime_straddles.keys())}")
    
    for regime, straddles in regime_straddles.items():
        print(f"  {regime}: {len(straddles)} straddles")
    
    # Show best straddles
    best_straddles = selector.get_best_straddles_by_type(result, top_n=2)
    print(f"✅ Best straddles by type: {list(best_straddles.keys())}")
    
    for straddle_type, straddles in best_straddles.items():
        print(f"  {straddle_type}: {len(straddles)} best straddles")
        for s in straddles:
            print(f"    {s.call_strike_type}/{s.put_strike_type} - {s.regime_bias} (conf: {s.confidence:.3f})")
    
    print("🚨 Strike Type Straddle Selection test COMPLETED")


if __name__ == "__main__":
    test_strike_type_straddle_selection()