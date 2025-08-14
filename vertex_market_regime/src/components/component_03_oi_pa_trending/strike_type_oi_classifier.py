"""
Strike Type OI Classifier Module

Classifies and analyzes OI patterns based on strike types (ATM/ITM/OTM) to detect:
- Strike-specific institutional positioning
- OI migration patterns across strike types
- Gamma hedging activity
- Strike concentration shifts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrikePositioning(Enum):
    """Types of strike-based positioning patterns"""
    ATM_CONCENTRATION = "atm_concentration"
    OTM_ACCUMULATION = "otm_accumulation"
    ITM_PROTECTION = "itm_protection"
    BALANCED_DISTRIBUTION = "balanced_distribution"
    WING_PROTECTION = "wing_protection"
    STRANGLE_SETUP = "strangle_setup"


@dataclass
class StrikeTypeOIMetrics:
    """Strike type-based OI analysis metrics"""
    atm_oi_concentration: float
    itm_oi_concentration: float
    otm_oi_concentration: float
    atm_ce_pe_ratio: float
    itm_ce_pe_ratio: float
    otm_ce_pe_ratio: float
    strike_migration_score: float
    gamma_concentration_level: float
    institutional_strike_bias: float
    retail_strike_preference: float
    strike_positioning_pattern: StrikePositioning
    max_pain_distance: float
    strike_skewness: float
    strike_dispersion: float
    hedging_activity_score: float


class StrikeTypeOIClassifier:
    """
    Classifies and analyzes OI patterns based on strike types
    """
    
    def __init__(self, strike_categories: List[str] = None):
        """
        Initialize Strike Type OI Classifier
        
        Args:
            strike_categories: Categories of strikes to analyze
        """
        self.strike_categories = strike_categories or ['ATM', 'ITM1', 'ITM2', 'OTM1', 'OTM2', 'OTM4']
        self.strike_history = {}
        logger.info(f"Initialized StrikeTypeOIClassifier with categories: {self.strike_categories}")
    
    def analyze_strike_types(self, df: pd.DataFrame) -> Dict[str, StrikeTypeOIMetrics]:
        """
        Analyze OI patterns across different strike types
        
        Args:
            df: Production data with strike type columns
            
        Returns:
            Strike type metrics by category
        """
        metrics = {}
        
        # Analyze overall strike distribution
        overall_metrics = self._analyze_overall_distribution(df)
        metrics["overall"] = overall_metrics
        
        # Analyze by specific strike types
        for strike_type in ['ATM', 'ITM', 'OTM']:
            type_metrics = self._analyze_strike_type_group(df, strike_type)
            if type_metrics:
                metrics[strike_type.lower()] = type_metrics
        
        # Analyze cross-strike patterns
        cross_strike_metrics = self._analyze_cross_strike_patterns(df)
        metrics["cross_strike"] = cross_strike_metrics
        
        return metrics
    
    def _analyze_overall_distribution(self, df: pd.DataFrame) -> StrikeTypeOIMetrics:
        """Analyze overall OI distribution across all strike types"""
        
        # Calculate OI concentrations
        atm_concentration = self._calculate_strike_concentration(df, 'ATM')
        itm_concentration = self._calculate_strike_concentration(df, ['ITM1', 'ITM2', 'ITM4'])
        otm_concentration = self._calculate_strike_concentration(df, ['OTM1', 'OTM2', 'OTM4'])
        
        # Calculate CE/PE ratios by strike type
        atm_ratio = self._calculate_ce_pe_ratio(df, 'ATM')
        itm_ratio = self._calculate_ce_pe_ratio(df, ['ITM1', 'ITM2', 'ITM4'])
        otm_ratio = self._calculate_ce_pe_ratio(df, ['OTM1', 'OTM2', 'OTM4'])
        
        # Strike migration analysis
        migration_score = self._calculate_strike_migration(df)
        
        # Gamma concentration (highest at ATM)
        gamma_concentration = self._calculate_gamma_concentration(df)
        
        # Institutional vs retail bias
        institutional_bias = self._calculate_institutional_strike_bias(df)
        retail_preference = 1.0 - institutional_bias
        
        # Strike positioning pattern
        positioning_pattern = self._classify_strike_positioning(
            atm_concentration, itm_concentration, otm_concentration
        )
        
        # Max pain analysis
        max_pain_distance = self._calculate_max_pain_distance(df)
        
        # Strike distribution metrics
        strike_skewness = self._calculate_strike_skewness(df)
        strike_dispersion = self._calculate_strike_dispersion(df)
        
        # Hedging activity
        hedging_score = self._calculate_hedging_activity(df)
        
        return StrikeTypeOIMetrics(
            atm_oi_concentration=atm_concentration,
            itm_oi_concentration=itm_concentration,
            otm_oi_concentration=otm_concentration,
            atm_ce_pe_ratio=atm_ratio,
            itm_ce_pe_ratio=itm_ratio,
            otm_ce_pe_ratio=otm_ratio,
            strike_migration_score=migration_score,
            gamma_concentration_level=gamma_concentration,
            institutional_strike_bias=institutional_bias,
            retail_strike_preference=retail_preference,
            strike_positioning_pattern=positioning_pattern,
            max_pain_distance=max_pain_distance,
            strike_skewness=strike_skewness,
            strike_dispersion=strike_dispersion,
            hedging_activity_score=hedging_score
        )
    
    def _calculate_strike_concentration(self, df: pd.DataFrame, 
                                       strike_types: Any) -> float:
        """Calculate OI concentration for specific strike types"""
        
        if isinstance(strike_types, str):
            strike_types = [strike_types]
        
        total_oi = 0
        strike_oi = 0
        
        if 'call_strike_type' in df.columns and 'ce_oi' in df.columns:
            for strike_type in strike_types:
                mask = df['call_strike_type'] == strike_type
                strike_oi += df.loc[mask, 'ce_oi'].sum()
            total_oi += df['ce_oi'].sum()
        
        if 'put_strike_type' in df.columns and 'pe_oi' in df.columns:
            for strike_type in strike_types:
                mask = df['put_strike_type'] == strike_type
                strike_oi += df.loc[mask, 'pe_oi'].sum()
            total_oi += df['pe_oi'].sum()
        
        if total_oi > 0:
            return strike_oi / total_oi
        
        return 0.0
    
    def _calculate_ce_pe_ratio(self, df: pd.DataFrame, strike_types: Any) -> float:
        """Calculate CE/PE OI ratio for specific strike types"""
        
        if isinstance(strike_types, str):
            strike_types = [strike_types]
        
        ce_oi = 0
        pe_oi = 0
        
        for strike_type in strike_types:
            if 'call_strike_type' in df.columns and 'ce_oi' in df.columns:
                mask = df['call_strike_type'] == strike_type
                ce_oi += df.loc[mask, 'ce_oi'].sum()
            
            if 'put_strike_type' in df.columns and 'pe_oi' in df.columns:
                mask = df['put_strike_type'] == strike_type
                pe_oi += df.loc[mask, 'pe_oi'].sum()
        
        if pe_oi > 0:
            return ce_oi / pe_oi
        elif ce_oi > 0:
            return 10.0  # High ratio when PE is zero
        else:
            return 1.0  # Neutral when both are zero
    
    def _calculate_strike_migration(self, df: pd.DataFrame) -> float:
        """Calculate OI migration between strike types"""
        
        migration_score = 0.0
        
        if 'call_strike_type' not in df.columns or 'ce_oi' not in df.columns:
            return 0.0
        
        # Track OI changes across strike types over time
        for strike_type in self.strike_categories:
            mask = df['call_strike_type'] == strike_type
            strike_oi = df.loc[mask, 'ce_oi']
            
            if len(strike_oi) > 1:
                # Calculate rate of change
                oi_change = strike_oi.pct_change().mean()
                
                # Store in history
                if strike_type not in self.strike_history:
                    self.strike_history[strike_type] = []
                self.strike_history[strike_type].append(oi_change)
                
                # Migration detected when some strikes gain while others lose
                migration_score += abs(oi_change)
        
        # Normalize migration score
        return min(migration_score / len(self.strike_categories), 1.0)
    
    def _calculate_gamma_concentration(self, df: pd.DataFrame) -> float:
        """Calculate gamma concentration level (highest at ATM)"""
        
        if 'call_strike_type' not in df.columns:
            return 0.0
        
        # ATM has highest gamma
        atm_mask = df['call_strike_type'] == 'ATM'
        
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            atm_oi = df.loc[atm_mask, 'ce_oi'].sum() + df.loc[atm_mask, 'pe_oi'].sum()
            total_oi = df['ce_oi'].sum() + df['pe_oi'].sum()
            
            if total_oi > 0:
                # Higher concentration at ATM means higher gamma exposure
                gamma_concentration = atm_oi / total_oi
                
                # Adjust for DTE if available (gamma increases near expiry)
                if 'dte' in df.columns:
                    avg_dte = df['dte'].mean()
                    if avg_dte < 7:
                        gamma_concentration *= 1.5
                    elif avg_dte < 15:
                        gamma_concentration *= 1.2
                
                return min(gamma_concentration, 1.0)
        
        return 0.0
    
    def _calculate_institutional_strike_bias(self, df: pd.DataFrame) -> float:
        """Calculate institutional preference for certain strike types"""
        
        institutional_score = 0.0
        
        # Institutions typically prefer ATM and near-the-money strikes
        atm_concentration = self._calculate_strike_concentration(df, 'ATM')
        near_money_concentration = self._calculate_strike_concentration(df, ['ITM1', 'OTM1'])
        
        # High concentration in ATM and near strikes indicates institutional
        if atm_concentration > 0.3:
            institutional_score += 0.4
        
        if near_money_concentration > 0.3:
            institutional_score += 0.3
        
        # Check for systematic hedging patterns
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            # Balanced CE/PE at ATM indicates hedging
            atm_mask = (df['call_strike_type'] == 'ATM') | (df['put_strike_type'] == 'ATM')
            if atm_mask.any():
                ce_atm = df.loc[atm_mask, 'ce_oi'].sum() if 'ce_oi' in df.columns else 0
                pe_atm = df.loc[atm_mask, 'pe_oi'].sum() if 'pe_oi' in df.columns else 0
                
                if ce_atm > 0 and pe_atm > 0:
                    balance_ratio = min(ce_atm, pe_atm) / max(ce_atm, pe_atm)
                    if balance_ratio > 0.7:  # Well balanced
                        institutional_score += 0.3
        
        return min(institutional_score, 1.0)
    
    def _classify_strike_positioning(self, atm_conc: float, itm_conc: float, 
                                    otm_conc: float) -> StrikePositioning:
        """Classify the strike positioning pattern"""
        
        # ATM concentration pattern
        if atm_conc > 0.5:
            return StrikePositioning.ATM_CONCENTRATION
        
        # OTM accumulation (speculation)
        if otm_conc > 0.4:
            return StrikePositioning.OTM_ACCUMULATION
        
        # ITM protection (hedging)
        if itm_conc > 0.4:
            return StrikePositioning.ITM_PROTECTION
        
        # Wing protection (tail hedging)
        if itm_conc > 0.2 and otm_conc > 0.2 and atm_conc < 0.3:
            return StrikePositioning.WING_PROTECTION
        
        # Strangle setup
        if otm_conc > 0.3 and itm_conc < 0.2:
            return StrikePositioning.STRANGLE_SETUP
        
        # Balanced distribution
        return StrikePositioning.BALANCED_DISTRIBUTION
    
    def _calculate_max_pain_distance(self, df: pd.DataFrame) -> float:
        """Calculate distance from current price to max pain strike"""
        
        if 'strike' not in df.columns or 'spot' not in df.columns:
            return 0.0
        
        # Calculate max pain (strike with maximum OI)
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            strike_oi = df.groupby('strike')[['ce_oi', 'pe_oi']].sum()
            total_oi = strike_oi.sum(axis=1)
            
            if not total_oi.empty:
                max_pain_strike = total_oi.idxmax()
                current_spot = df['spot'].iloc[-1] if len(df) > 0 else 0
                
                if current_spot > 0:
                    distance = (max_pain_strike - current_spot) / current_spot
                    return distance
        
        return 0.0
    
    def _calculate_strike_skewness(self, df: pd.DataFrame) -> float:
        """Calculate skewness of OI distribution across strikes"""
        
        if 'strike' not in df.columns:
            return 0.0
        
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            strike_oi = df.groupby('strike')[['ce_oi', 'pe_oi']].sum()
            total_oi_by_strike = strike_oi.sum(axis=1)
            
            if len(total_oi_by_strike) > 2:
                # Calculate skewness
                mean_oi = total_oi_by_strike.mean()
                std_oi = total_oi_by_strike.std()
                
                if std_oi > 0:
                    skewness = ((total_oi_by_strike - mean_oi) ** 3).mean() / (std_oi ** 3)
                    return np.clip(skewness / 3, -1.0, 1.0)  # Normalize to [-1, 1]
        
        return 0.0
    
    def _calculate_strike_dispersion(self, df: pd.DataFrame) -> float:
        """Calculate dispersion of OI across strikes"""
        
        if 'strike' not in df.columns:
            return 0.0
        
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            strike_oi = df.groupby('strike')[['ce_oi', 'pe_oi']].sum()
            total_oi_by_strike = strike_oi.sum(axis=1)
            
            if len(total_oi_by_strike) > 1:
                # Calculate coefficient of variation
                mean_oi = total_oi_by_strike.mean()
                std_oi = total_oi_by_strike.std()
                
                if mean_oi > 0:
                    dispersion = std_oi / mean_oi
                    return min(dispersion, 1.0)
        
        return 0.0
    
    def _calculate_hedging_activity(self, df: pd.DataFrame) -> float:
        """Calculate hedging activity score based on strike patterns"""
        
        hedging_score = 0.0
        
        # Balanced CE/PE at multiple strikes indicates hedging
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            if 'strike' in df.columns:
                strike_ratios = []
                
                for strike in df['strike'].unique():
                    strike_mask = df['strike'] == strike
                    ce_oi = df.loc[strike_mask, 'ce_oi'].sum()
                    pe_oi = df.loc[strike_mask, 'pe_oi'].sum()
                    
                    if ce_oi > 0 and pe_oi > 0:
                        ratio = min(ce_oi, pe_oi) / max(ce_oi, pe_oi)
                        strike_ratios.append(ratio)
                
                if strike_ratios:
                    # High average ratio indicates hedging
                    avg_ratio = np.mean(strike_ratios)
                    if avg_ratio > 0.7:
                        hedging_score = 0.8
                    elif avg_ratio > 0.5:
                        hedging_score = 0.5
                    else:
                        hedging_score = 0.2
        
        # ITM protection adds to hedging score
        itm_concentration = self._calculate_strike_concentration(df, ['ITM1', 'ITM2'])
        if itm_concentration > 0.2:
            hedging_score = min(hedging_score + 0.2, 1.0)
        
        return hedging_score
    
    def _analyze_strike_type_group(self, df: pd.DataFrame, 
                                  strike_group: str) -> Optional[StrikeTypeOIMetrics]:
        """Analyze specific strike type group (ATM, ITM, OTM)"""
        
        if strike_group == 'ATM':
            strike_types = ['ATM']
        elif strike_group == 'ITM':
            strike_types = ['ITM1', 'ITM2', 'ITM4']
        elif strike_group == 'OTM':
            strike_types = ['OTM1', 'OTM2', 'OTM4']
        else:
            return None
        
        # Filter data for this strike group
        group_mask = False
        if 'call_strike_type' in df.columns:
            for st in strike_types:
                group_mask = group_mask | (df['call_strike_type'] == st)
        
        if not isinstance(group_mask, pd.Series) or group_mask.sum() == 0:
            return None
        
        group_df = df[group_mask]
        
        # Analyze this specific group
        return self._analyze_overall_distribution(group_df)
    
    def _analyze_cross_strike_patterns(self, df: pd.DataFrame) -> StrikeTypeOIMetrics:
        """Analyze patterns across different strike types"""
        
        # This provides a holistic view of cross-strike dynamics
        return self._analyze_overall_distribution(df)
    
    def detect_strike_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect specific strike-based patterns
        
        Args:
            df: Production data
            
        Returns:
            Detected strike patterns
        """
        patterns = {
            'gamma_squeeze_risk': False,
            'institutional_hedging': False,
            'retail_speculation': False,
            'max_pain_magnet': False,
            'strike_migration_active': False,
            'pattern_details': []
        }
        
        metrics = self.analyze_strike_types(df)
        overall = metrics.get("overall")
        
        if overall:
            # Gamma squeeze risk (high ATM concentration)
            if overall.gamma_concentration_level > 0.6:
                patterns['gamma_squeeze_risk'] = True
                patterns['pattern_details'].append("High gamma concentration at ATM")
            
            # Institutional hedging
            if overall.institutional_strike_bias > 0.7 and overall.hedging_activity_score > 0.6:
                patterns['institutional_hedging'] = True
                patterns['pattern_details'].append("Institutional hedging patterns detected")
            
            # Retail speculation (high OTM concentration)
            if overall.otm_oi_concentration > 0.4 and overall.retail_strike_preference > 0.6:
                patterns['retail_speculation'] = True
                patterns['pattern_details'].append("Retail speculation in OTM strikes")
            
            # Max pain magnet
            if abs(overall.max_pain_distance) < 0.02:  # Within 2% of max pain
                patterns['max_pain_magnet'] = True
                patterns['pattern_details'].append("Price near max pain strike")
            
            # Strike migration
            if overall.strike_migration_score > 0.3:
                patterns['strike_migration_active'] = True
                patterns['pattern_details'].append("Active OI migration across strikes")
        
        return patterns
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract strike type features for Component 3
        
        Args:
            df: Production data
            
        Returns:
            Array of strike type features
        """
        features = []
        
        # Analyze strike types
        metrics = self.analyze_strike_types(df)
        
        # Extract features from overall metrics
        overall = metrics.get("overall")
        if overall:
            features.extend([
                overall.atm_oi_concentration,
                overall.itm_oi_concentration,
                overall.otm_oi_concentration,
                np.log(overall.atm_ce_pe_ratio + 1) / 3,  # Normalized log ratio
                np.log(overall.itm_ce_pe_ratio + 1) / 3,
                np.log(overall.otm_ce_pe_ratio + 1) / 3,
                overall.strike_migration_score,
                overall.gamma_concentration_level,
                overall.institutional_strike_bias,
                overall.retail_strike_preference,
                overall.max_pain_distance,
                overall.strike_skewness,
                overall.strike_dispersion,
                overall.hedging_activity_score
            ])
        
        # Detect patterns
        patterns = self.detect_strike_patterns(df)
        features.extend([
            float(patterns['gamma_squeeze_risk']),
            float(patterns['institutional_hedging']),
            float(patterns['retail_speculation']),
            float(patterns['max_pain_magnet']),
            float(patterns['strike_migration_active'])
        ])
        
        return np.array(features)