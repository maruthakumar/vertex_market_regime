"""
Gap Analysis Engine for Component 6

Implements comprehensive gap analysis and overnight factor integration
for correlation adjustment and predictive feature engineering.

Features:
- Previous day to opening gap measurement
- Overnight factor integration (6-factor system)
- Gap-adjusted correlation weights
- Current strike correlation analysis (option trader approach)
- Intraday gap evolution tracking
- Gap prediction feature extraction

🎯 PURE MEASUREMENT AND CALCULATION - NO PREDICTION LOGIC
All gap metrics are raw measurements for ML model consumption.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
import logging
import time
import warnings
from scipy import stats
from collections import deque

warnings.filterwarnings('ignore')


@dataclass
class GapMetrics:
    """Gap measurement metrics"""
    gap_size_percent: float
    gap_direction: float  # 1.0 for up, -1.0 for down, 0.0 for no gap
    gap_category: str     # 'no_gap', 'small_gap', 'medium_gap', 'large_gap', 'extreme_gap'
    absolute_gap_points: float
    previous_close: float
    current_open: float
    timestamp: datetime


@dataclass
class OvernightFactorData:
    """Overnight factor measurements"""
    sgx_nifty_change: float
    dow_jones_change: float
    vix_change: float
    usd_inr_change: float
    news_sentiment_score: float
    commodity_changes: Dict[str, float]
    global_sentiment: float
    timestamp: datetime


@dataclass
class GapCorrelationWeights:
    """Gap-adjusted correlation weights"""
    base_correlation_weight: float
    sgx_nifty_weight: float
    dow_jones_weight: float
    news_sentiment_weight: float
    vix_weight: float
    usd_inr_weight: float
    commodity_weight: float
    gap_category: str
    total_adjustment_factor: float


@dataclass
class StrikeCorrelationAnalysis:
    """Current strike correlation analysis (option trader approach)"""
    current_atm_strike: float
    current_itm1_strike: float
    current_otm1_strike: float
    previous_atm_strike: float
    previous_itm1_strike: float
    previous_otm1_strike: float
    
    atm_correlation: float
    itm1_correlation: float
    otm1_correlation: float
    
    strike_migration_impact: float
    moneyness_adjustment_factor: float


@dataclass
class GapAnalysisResult:
    """Complete gap analysis result"""
    gap_metrics: GapMetrics
    overnight_factors: OvernightFactorData
    correlation_weights: GapCorrelationWeights
    strike_analysis: StrikeCorrelationAnalysis
    
    # Feature arrays for ML consumption
    gap_direction_features: np.ndarray      # 8 features
    gap_magnitude_features: np.ndarray      # 7 features
    overnight_factor_features: np.ndarray   # 15 features
    strike_correlation_features: np.ndarray # 10 features
    
    processing_time_ms: float
    confidence_score: float
    timestamp: datetime


class GapAnalysisEngine:
    """
    Comprehensive gap analysis engine for Component 6
    
    Implements gap measurement, overnight factor integration, and gap-adjusted
    correlation weight calculation for enhanced correlation analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Gap category thresholds
        self.gap_thresholds = {
            'no_gap': 0.2,      # ±0.2%
            'small_gap': 0.5,   # ±0.5%
            'medium_gap': 1.0,  # ±1.0%
            'large_gap': 2.0,   # ±2.0%
            # extreme_gap: >2.0%
        }
        
        # Correlation weight adjustments by gap category
        self.correlation_weight_adjustments = {
            'no_gap': 1.0,      # Full correlation weight
            'small_gap': 0.8,   # Slight adjustment
            'medium_gap': 0.6,  # Moderate adjustment
            'large_gap': 0.4,   # Significant adjustment
            'extreme_gap': 0.2  # Major adjustment
        }
        
        # Overnight factor weights (6-factor system)
        self.overnight_factor_weights = {
            'sgx_nifty': 0.15,      # Direct NIFTY gap prediction
            'dow_jones': 0.10,      # Global sentiment
            'news_sentiment': 0.20, # India-specific events
            'vix_change': 0.20,     # Global volatility
            'usd_inr': 0.15,        # FII flow impact
            'commodities': 0.10,    # Sector rotation
            'other': 0.10           # Miscellaneous factors
        }
        
        # Strike correlation decay factors
        self.strike_decay_factors = {
            'no_gap': 0.95,      # Minimal decay
            'small_gap': 0.85,   # Light decay
            'medium_gap': 0.70,  # Moderate decay
            'large_gap': 0.50,   # Significant decay
            'extreme_gap': 0.25  # Major decay
        }
        
        # Historical gap data storage
        self.gap_history = deque(maxlen=252)  # 1 year of gap data
        self.overnight_history = deque(maxlen=100)  # Recent overnight factors
        
        self.logger.info("Gap Analysis Engine initialized with 6-factor overnight system")

    def analyze_comprehensive_gap(self, 
                                market_data: pd.DataFrame,
                                overnight_data: Dict[str, float],
                                previous_day_data: Optional[pd.DataFrame] = None) -> GapAnalysisResult:
        """
        Perform comprehensive gap analysis with overnight factor integration
        
        Args:
            market_data: Current day market data
            overnight_data: Overnight factor data
            previous_day_data: Previous trading day data (optional)
            
        Returns:
            GapAnalysisResult with complete gap analysis
        """
        start_time = time.time()
        
        try:
            # 1. Calculate basic gap metrics
            gap_metrics = self._calculate_gap_metrics(market_data, previous_day_data)
            
            # 2. Process overnight factors
            overnight_factors = self._process_overnight_factors(overnight_data)
            
            # 3. Calculate gap-adjusted correlation weights
            correlation_weights = self._calculate_correlation_weights(gap_metrics, overnight_factors)
            
            # 4. Perform strike correlation analysis
            strike_analysis = self._analyze_strike_correlations(
                market_data, previous_day_data, gap_metrics
            )
            
            # 5. Extract features for ML consumption
            gap_direction_features = self._extract_gap_direction_features(gap_metrics, overnight_factors)
            gap_magnitude_features = self._extract_gap_magnitude_features(gap_metrics, overnight_factors)
            overnight_factor_features = self._extract_overnight_factor_features(overnight_factors)
            strike_correlation_features = self._extract_strike_correlation_features(strike_analysis)
            
            # 6. Calculate confidence score
            confidence_score = self._calculate_gap_analysis_confidence(
                gap_metrics, overnight_factors, len(market_data)
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Store in history for future analysis
            self.gap_history.append(gap_metrics)
            self.overnight_history.append(overnight_factors)
            
            return GapAnalysisResult(
                gap_metrics=gap_metrics,
                overnight_factors=overnight_factors,
                correlation_weights=correlation_weights,
                strike_analysis=strike_analysis,
                gap_direction_features=gap_direction_features,
                gap_magnitude_features=gap_magnitude_features,
                overnight_factor_features=overnight_factor_features,
                strike_correlation_features=strike_correlation_features,
                processing_time_ms=processing_time,
                confidence_score=confidence_score,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Gap analysis failed: {e}")
            
            # Return minimal result on failure
            return self._create_minimal_gap_result(processing_time)

    def _calculate_gap_metrics(self, 
                              market_data: pd.DataFrame,
                              previous_day_data: Optional[pd.DataFrame]) -> GapMetrics:
        """Calculate basic gap measurement metrics"""
        
        try:
            if len(market_data) == 0:
                return self._create_default_gap_metrics()
            
            # Get current open price (first price of the day)
            if 'open' in market_data.columns:
                current_open = market_data['open'].iloc[0]
            elif 'close' in market_data.columns:
                current_open = market_data['close'].iloc[0]
            else:
                current_open = market_data.iloc[0, 0] if len(market_data.columns) > 0 else 0.0
            
            # Get previous close
            if previous_day_data is not None and len(previous_day_data) > 0:
                if 'close' in previous_day_data.columns:
                    previous_close = previous_day_data['close'].iloc[-1]
                else:
                    previous_close = previous_day_data.iloc[-1, 0] if len(previous_day_data.columns) > 0 else current_open
            else:
                # Fallback: use second data point as "previous close"
                if len(market_data) > 1 and 'close' in market_data.columns:
                    previous_close = market_data['close'].iloc[1]
                else:
                    previous_close = current_open
            
            # Calculate gap metrics
            if previous_close != 0:
                gap_size_percent = ((current_open - previous_close) / previous_close) * 100
                absolute_gap_points = abs(current_open - previous_close)
            else:
                gap_size_percent = 0.0
                absolute_gap_points = 0.0
            
            # Determine gap direction
            if gap_size_percent > 0.1:
                gap_direction = 1.0  # Gap up
            elif gap_size_percent < -0.1:
                gap_direction = -1.0  # Gap down
            else:
                gap_direction = 0.0  # No significant gap
            
            # Categorize gap size
            abs_gap = abs(gap_size_percent)
            if abs_gap <= self.gap_thresholds['no_gap']:
                gap_category = 'no_gap'
            elif abs_gap <= self.gap_thresholds['small_gap']:
                gap_category = 'small_gap'
            elif abs_gap <= self.gap_thresholds['medium_gap']:
                gap_category = 'medium_gap'
            elif abs_gap <= self.gap_thresholds['large_gap']:
                gap_category = 'large_gap'
            else:
                gap_category = 'extreme_gap'
            
            return GapMetrics(
                gap_size_percent=gap_size_percent,
                gap_direction=gap_direction,
                gap_category=gap_category,
                absolute_gap_points=absolute_gap_points,
                previous_close=previous_close,
                current_open=current_open,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating gap metrics: {e}")
            return self._create_default_gap_metrics()

    def _process_overnight_factors(self, overnight_data: Dict[str, float]) -> OvernightFactorData:
        """Process and normalize overnight factor data"""
        
        try:
            # Extract overnight factors with defaults
            sgx_nifty_change = overnight_data.get('sgx_nifty', 0.0)
            dow_jones_change = overnight_data.get('dow_jones', 0.0)
            vix_change = overnight_data.get('vix_change', 0.0)
            usd_inr_change = overnight_data.get('usd_inr', 0.0)
            news_sentiment_score = overnight_data.get('news_sentiment', 0.0)
            
            # Extract commodity changes
            commodity_changes = {}
            for commodity in ['oil', 'gold', 'copper', 'silver']:
                commodity_changes[commodity] = overnight_data.get(f'{commodity}_change', 0.0)
            
            # Calculate global sentiment composite
            global_sentiment = (
                sgx_nifty_change * 0.3 + 
                dow_jones_change * 0.3 +
                -abs(vix_change) * 0.2 +  # Higher VIX = lower sentiment
                news_sentiment_score * 0.2
            )
            
            return OvernightFactorData(
                sgx_nifty_change=sgx_nifty_change,
                dow_jones_change=dow_jones_change,
                vix_change=vix_change,
                usd_inr_change=usd_inr_change,
                news_sentiment_score=news_sentiment_score,
                commodity_changes=commodity_changes,
                global_sentiment=global_sentiment,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error processing overnight factors: {e}")
            return OvernightFactorData(
                sgx_nifty_change=0.0,
                dow_jones_change=0.0,
                vix_change=0.0,
                usd_inr_change=0.0,
                news_sentiment_score=0.0,
                commodity_changes={},
                global_sentiment=0.0,
                timestamp=datetime.utcnow()
            )

    def _calculate_correlation_weights(self, 
                                     gap_metrics: GapMetrics,
                                     overnight_factors: OvernightFactorData) -> GapCorrelationWeights:
        """Calculate gap-adjusted correlation weights"""
        
        try:
            # Base correlation weight from gap category
            base_weight = self.correlation_weight_adjustments.get(gap_metrics.gap_category, 1.0)
            
            # Individual overnight factor weights
            sgx_weight = self.overnight_factor_weights['sgx_nifty'] * (1.0 + abs(overnight_factors.sgx_nifty_change) * 0.1)
            dow_weight = self.overnight_factor_weights['dow_jones'] * (1.0 + abs(overnight_factors.dow_jones_change) * 0.05)
            news_weight = self.overnight_factor_weights['news_sentiment'] * (1.0 + abs(overnight_factors.news_sentiment_score) * 0.1)
            vix_weight = self.overnight_factor_weights['vix_change'] * (1.0 + abs(overnight_factors.vix_change) * 0.1)
            usd_inr_weight = self.overnight_factor_weights['usd_inr'] * (1.0 + abs(overnight_factors.usd_inr_change) * 0.05)
            commodity_weight = self.overnight_factor_weights['commodities']
            
            # Total adjustment factor
            total_adjustment = (sgx_weight + dow_weight + news_weight + vix_weight + 
                              usd_inr_weight + commodity_weight) * base_weight
            
            return GapCorrelationWeights(
                base_correlation_weight=base_weight,
                sgx_nifty_weight=sgx_weight,
                dow_jones_weight=dow_weight,
                news_sentiment_weight=news_weight,
                vix_weight=vix_weight,
                usd_inr_weight=usd_inr_weight,
                commodity_weight=commodity_weight,
                gap_category=gap_metrics.gap_category,
                total_adjustment_factor=min(2.0, max(0.1, total_adjustment))
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation weights: {e}")
            return GapCorrelationWeights(
                base_correlation_weight=1.0,
                sgx_nifty_weight=0.15,
                dow_jones_weight=0.10,
                news_sentiment_weight=0.20,
                vix_weight=0.20,
                usd_inr_weight=0.15,
                commodity_weight=0.10,
                gap_category='no_gap',
                total_adjustment_factor=1.0
            )

    def _analyze_strike_correlations(self, 
                                   market_data: pd.DataFrame,
                                   previous_day_data: Optional[pd.DataFrame],
                                   gap_metrics: GapMetrics) -> StrikeCorrelationAnalysis:
        """
        Analyze current strike correlation (option trader approach)
        
        Always correlate current day's ATM, ITM1, OTM1 with previous day's ATM, ITM1, OTM1
        regardless of actual strike values (handles gap-induced strike migration)
        """
        
        try:
            # Extract current day strikes
            current_spot = market_data['spot'].iloc[0] if 'spot' in market_data.columns else gap_metrics.current_open
            current_atm = self._find_atm_strike(current_spot)
            current_itm1 = current_atm - 50  # Assuming 50-point intervals
            current_otm1 = current_atm + 50
            
            # Extract previous day strikes
            if previous_day_data is not None and len(previous_day_data) > 0:
                previous_spot = previous_day_data['spot'].iloc[-1] if 'spot' in previous_day_data.columns else gap_metrics.previous_close
                previous_atm = self._find_atm_strike(previous_spot)
                previous_itm1 = previous_atm - 50
                previous_otm1 = previous_atm + 50
            else:
                # Fallback to current strikes adjusted for gap
                previous_atm = current_atm - (gap_metrics.absolute_gap_points)
                previous_itm1 = previous_atm - 50
                previous_otm1 = previous_atm + 50
            
            # Calculate correlations between current and previous strikes
            atm_correlation = self._calculate_strike_correlation(
                current_atm, previous_atm, market_data, previous_day_data, 'atm'
            )
            itm1_correlation = self._calculate_strike_correlation(
                current_itm1, previous_itm1, market_data, previous_day_data, 'itm1'
            )
            otm1_correlation = self._calculate_strike_correlation(
                current_otm1, previous_otm1, market_data, previous_day_data, 'otm1'
            )
            
            # Calculate strike migration impact
            strike_migration_impact = abs(current_atm - previous_atm) / previous_atm if previous_atm != 0 else 0.0
            
            # Calculate moneyness adjustment factor
            decay_factor = self.strike_decay_factors.get(gap_metrics.gap_category, 1.0)
            moneyness_adjustment = decay_factor * (1.0 - min(0.5, strike_migration_impact))
            
            return StrikeCorrelationAnalysis(
                current_atm_strike=current_atm,
                current_itm1_strike=current_itm1,
                current_otm1_strike=current_otm1,
                previous_atm_strike=previous_atm,
                previous_itm1_strike=previous_itm1,
                previous_otm1_strike=previous_otm1,
                atm_correlation=atm_correlation,
                itm1_correlation=itm1_correlation,
                otm1_correlation=otm1_correlation,
                strike_migration_impact=strike_migration_impact,
                moneyness_adjustment_factor=moneyness_adjustment
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing strike correlations: {e}")
            return StrikeCorrelationAnalysis(
                current_atm_strike=20000.0,
                current_itm1_strike=19950.0,
                current_otm1_strike=20050.0,
                previous_atm_strike=20000.0,
                previous_itm1_strike=19950.0,
                previous_otm1_strike=20050.0,
                atm_correlation=0.5,
                itm1_correlation=0.5,
                otm1_correlation=0.5,
                strike_migration_impact=0.0,
                moneyness_adjustment_factor=1.0
            )

    def _find_atm_strike(self, spot_price: float) -> float:
        """Find ATM strike for given spot price (assuming 50-point intervals)"""
        return round(spot_price / 50) * 50

    def _calculate_strike_correlation(self, 
                                    current_strike: float,
                                    previous_strike: float,
                                    market_data: pd.DataFrame,
                                    previous_day_data: Optional[pd.DataFrame],
                                    strike_type: str) -> float:
        """Calculate correlation between current and previous strike premiums"""
        
        try:
            # This is a simplified calculation - in production would need actual option data
            # For now, return correlation based on strike proximity
            strike_difference = abs(current_strike - previous_strike)
            
            if strike_difference == 0:
                return 1.0  # Perfect correlation for same strike
            elif strike_difference <= 50:
                return 0.8  # High correlation for nearby strikes
            elif strike_difference <= 100:
                return 0.6  # Moderate correlation
            elif strike_difference <= 200:
                return 0.4  # Low correlation
            else:
                return 0.2  # Very low correlation for distant strikes
                
        except Exception as e:
            self.logger.error(f"Error calculating strike correlation: {e}")
            return 0.5

    def _extract_gap_direction_features(self, 
                                      gap_metrics: GapMetrics,
                                      overnight_factors: OvernightFactorData) -> np.ndarray:
        """Extract gap direction predictor features (8 features)"""
        
        features = []
        
        try:
            # Feature 1: Gap direction (normalized)
            features.append(gap_metrics.gap_direction)
            
            # Feature 2: Gap size magnitude (log-normalized)
            gap_magnitude = min(5.0, abs(gap_metrics.gap_size_percent))  # Cap at 5%
            features.append(gap_magnitude / 5.0)
            
            # Feature 3: SGX NIFTY directional impact
            sgx_direction = 1.0 if overnight_factors.sgx_nifty_change > 0 else -1.0
            features.append(sgx_direction * min(1.0, abs(overnight_factors.sgx_nifty_change) / 2.0))
            
            # Feature 4: Dow Jones sentiment direction
            dow_direction = 1.0 if overnight_factors.dow_jones_change > 0 else -1.0
            features.append(dow_direction * min(1.0, abs(overnight_factors.dow_jones_change) / 3.0))
            
            # Feature 5: VIX impact (inverse - higher VIX suggests negative sentiment)
            vix_impact = -overnight_factors.vix_change / 10.0  # Normalize VIX change
            features.append(max(-1.0, min(1.0, vix_impact)))
            
            # Feature 6: News sentiment impact
            news_impact = max(-1.0, min(1.0, overnight_factors.news_sentiment_score))
            features.append(news_impact)
            
            # Feature 7: USD-INR impact (FII flow proxy)
            usd_inr_impact = -overnight_factors.usd_inr_change / 2.0  # Stronger USD = outflow
            features.append(max(-1.0, min(1.0, usd_inr_impact)))
            
            # Feature 8: Global sentiment composite
            global_sentiment = max(-1.0, min(1.0, overnight_factors.global_sentiment / 2.0))
            features.append(global_sentiment)
            
        except Exception as e:
            self.logger.error(f"Error extracting gap direction features: {e}")
            features = [0.0] * 8
        
        # Ensure exactly 8 features
        while len(features) < 8:
            features.append(0.0)
            
        return np.array(features[:8], dtype=np.float32)

    def _extract_gap_magnitude_features(self, 
                                      gap_metrics: GapMetrics,
                                      overnight_factors: OvernightFactorData) -> np.ndarray:
        """Extract gap magnitude predictor features (7 features)"""
        
        features = []
        
        try:
            # Feature 1: Absolute gap size (normalized)
            abs_gap = min(5.0, abs(gap_metrics.gap_size_percent))
            features.append(abs_gap / 5.0)
            
            # Feature 2: Gap category encoding
            category_encoding = {
                'no_gap': 0.0, 'small_gap': 0.2, 'medium_gap': 0.4,
                'large_gap': 0.6, 'extreme_gap': 0.8
            }
            features.append(category_encoding.get(gap_metrics.gap_category, 0.0))
            
            # Feature 3: SGX NIFTY magnitude influence
            sgx_magnitude = min(1.0, abs(overnight_factors.sgx_nifty_change) / 3.0)
            features.append(sgx_magnitude)
            
            # Feature 4: VIX magnitude influence
            vix_magnitude = min(1.0, abs(overnight_factors.vix_change) / 5.0)
            features.append(vix_magnitude)
            
            # Feature 5: Combined overnight factor magnitude
            combined_magnitude = (
                abs(overnight_factors.sgx_nifty_change) * 0.3 +
                abs(overnight_factors.dow_jones_change) * 0.2 +
                abs(overnight_factors.vix_change) * 0.2 +
                abs(overnight_factors.usd_inr_change) * 0.15 +
                abs(overnight_factors.news_sentiment_score) * 0.15
            )
            features.append(min(1.0, combined_magnitude / 3.0))
            
            # Feature 6: Historical gap percentile (if history available)
            if len(self.gap_history) > 10:
                historical_gaps = [abs(g.gap_size_percent) for g in self.gap_history]
                gap_percentile = stats.percentileofscore(historical_gaps, abs_gap) / 100.0
                features.append(gap_percentile)
            else:
                features.append(0.5)  # Default percentile
            
            # Feature 7: Gap volatility context
            if len(self.gap_history) >= 5:
                recent_gaps = [abs(g.gap_size_percent) for g in list(self.gap_history)[-5:]]
                gap_volatility = np.std(recent_gaps)
                normalized_volatility = min(1.0, gap_volatility / 2.0)
                features.append(normalized_volatility)
            else:
                features.append(0.3)  # Default volatility
                
        except Exception as e:
            self.logger.error(f"Error extracting gap magnitude features: {e}")
            features = [0.0] * 7
        
        # Ensure exactly 7 features
        while len(features) < 7:
            features.append(0.0)
            
        return np.array(features[:7], dtype=np.float32)

    def _extract_overnight_factor_features(self, overnight_factors: OvernightFactorData) -> np.ndarray:
        """Extract overnight factor features (15 features)"""
        
        features = []
        
        try:
            # Features 1-6: Individual overnight factors (normalized)
            features.append(max(-1.0, min(1.0, overnight_factors.sgx_nifty_change / 3.0)))
            features.append(max(-1.0, min(1.0, overnight_factors.dow_jones_change / 3.0)))
            features.append(max(-1.0, min(1.0, overnight_factors.vix_change / 10.0)))
            features.append(max(-1.0, min(1.0, overnight_factors.usd_inr_change / 2.0)))
            features.append(max(-1.0, min(1.0, overnight_factors.news_sentiment_score)))
            features.append(max(-1.0, min(1.0, overnight_factors.global_sentiment / 2.0)))
            
            # Features 7-10: Commodity factors
            commodities = ['oil', 'gold', 'copper', 'silver']
            for commodity in commodities:
                commodity_change = overnight_factors.commodity_changes.get(commodity, 0.0)
                normalized_change = max(-1.0, min(1.0, commodity_change / 5.0))  # Normalize to ±5%
                features.append(normalized_change)
            
            # Features 11-15: Factor interactions and composites
            # Feature 11: Risk-on/Risk-off composite
            risk_on_composite = (
                overnight_factors.dow_jones_change * 0.4 +
                -overnight_factors.vix_change * 0.3 +  # Lower VIX = risk on
                overnight_factors.commodity_changes.get('oil', 0.0) * 0.3
            )
            features.append(max(-1.0, min(1.0, risk_on_composite / 3.0)))
            
            # Feature 12: FII flow proxy
            fii_flow_proxy = (
                -overnight_factors.usd_inr_change * 0.6 +  # Stronger USD = outflow
                overnight_factors.dow_jones_change * 0.4   # US market performance
            )
            features.append(max(-1.0, min(1.0, fii_flow_proxy / 2.0)))
            
            # Feature 13: Volatility regime indicator
            volatility_regime = min(1.0, abs(overnight_factors.vix_change) / 5.0)
            features.append(volatility_regime)
            
            # Feature 14: News sentiment reliability
            news_reliability = min(1.0, abs(overnight_factors.news_sentiment_score) * 0.8)
            features.append(news_reliability)
            
            # Feature 15: Overall overnight uncertainty
            uncertainty_factors = [
                abs(overnight_factors.vix_change) / 10.0,
                abs(overnight_factors.usd_inr_change) / 2.0,
                1.0 - abs(overnight_factors.news_sentiment_score),  # Neutral news = uncertainty
            ]
            overnight_uncertainty = min(1.0, np.mean(uncertainty_factors))
            features.append(overnight_uncertainty)
            
        except Exception as e:
            self.logger.error(f"Error extracting overnight factor features: {e}")
            features = [0.0] * 15
        
        # Ensure exactly 15 features
        while len(features) < 15:
            features.append(0.0)
            
        return np.array(features[:15], dtype=np.float32)

    def _extract_strike_correlation_features(self, strike_analysis: StrikeCorrelationAnalysis) -> np.ndarray:
        """Extract strike correlation features (10 features)"""
        
        features = []
        
        try:
            # Features 1-3: Strike correlations
            features.append(strike_analysis.atm_correlation)
            features.append(strike_analysis.itm1_correlation)
            features.append(strike_analysis.otm1_correlation)
            
            # Feature 4: Average strike correlation
            avg_correlation = (strike_analysis.atm_correlation + 
                             strike_analysis.itm1_correlation + 
                             strike_analysis.otm1_correlation) / 3.0
            features.append(avg_correlation)
            
            # Feature 5: Strike correlation stability (1 - std)
            correlation_std = np.std([strike_analysis.atm_correlation, 
                                    strike_analysis.itm1_correlation, 
                                    strike_analysis.otm1_correlation])
            correlation_stability = max(0.0, 1.0 - correlation_std)
            features.append(correlation_stability)
            
            # Feature 6: Strike migration impact
            features.append(min(1.0, strike_analysis.strike_migration_impact))
            
            # Feature 7: Moneyness adjustment factor
            features.append(strike_analysis.moneyness_adjustment_factor)
            
            # Feature 8: ATM vs ITM correlation differential
            atm_itm_diff = abs(strike_analysis.atm_correlation - strike_analysis.itm1_correlation)
            features.append(atm_itm_diff)
            
            # Feature 9: ATM vs OTM correlation differential
            atm_otm_diff = abs(strike_analysis.atm_correlation - strike_analysis.otm1_correlation)
            features.append(atm_otm_diff)
            
            # Feature 10: Strike range correlation decay
            if strike_analysis.current_atm_strike != strike_analysis.current_itm1_strike:
                strike_range = abs(strike_analysis.current_otm1_strike - strike_analysis.current_itm1_strike)
                correlation_decay = 1.0 - min(1.0, strike_range / 200.0)  # Normalize by 200 points
                features.append(correlation_decay)
            else:
                features.append(1.0)
                
        except Exception as e:
            self.logger.error(f"Error extracting strike correlation features: {e}")
            features = [0.5] * 10
        
        # Ensure exactly 10 features
        while len(features) < 10:
            features.append(0.5)
            
        return np.array(features[:10], dtype=np.float32)

    def _calculate_gap_analysis_confidence(self, 
                                         gap_metrics: GapMetrics,
                                         overnight_factors: OvernightFactorData,
                                         data_size: int) -> float:
        """Calculate confidence score for gap analysis"""
        
        try:
            # Data availability confidence
            data_confidence = min(1.0, data_size / 100.0)  # Full confidence at 100+ data points
            
            # Gap measurement confidence (higher for clearer gaps)
            gap_confidence = min(1.0, abs(gap_metrics.gap_size_percent) / 2.0 + 0.3)
            
            # Overnight factor reliability
            factor_reliability = (
                min(1.0, abs(overnight_factors.sgx_nifty_change) / 3.0 + 0.5) * 0.3 +
                min(1.0, abs(overnight_factors.news_sentiment_score) + 0.5) * 0.3 +
                min(1.0, abs(overnight_factors.global_sentiment) / 2.0 + 0.5) * 0.4
            )
            
            # Combined confidence
            overall_confidence = (data_confidence * 0.3 + 
                                gap_confidence * 0.4 + 
                                factor_reliability * 0.3)
            
            return max(0.1, min(1.0, overall_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating gap analysis confidence: {e}")
            return 0.5

    def _create_default_gap_metrics(self) -> GapMetrics:
        """Create default gap metrics for fallback"""
        return GapMetrics(
            gap_size_percent=0.0,
            gap_direction=0.0,
            gap_category='no_gap',
            absolute_gap_points=0.0,
            previous_close=20000.0,
            current_open=20000.0,
            timestamp=datetime.utcnow()
        )

    def _create_minimal_gap_result(self, processing_time: float) -> GapAnalysisResult:
        """Create minimal gap analysis result for fallback"""
        return GapAnalysisResult(
            gap_metrics=self._create_default_gap_metrics(),
            overnight_factors=OvernightFactorData(0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, datetime.utcnow()),
            correlation_weights=GapCorrelationWeights(1.0, 0.15, 0.10, 0.20, 0.20, 0.15, 0.10, 'no_gap', 1.0),
            strike_analysis=StrikeCorrelationAnalysis(20000, 19950, 20050, 20000, 19950, 20050, 0.5, 0.5, 0.5, 0.0, 1.0),
            gap_direction_features=np.zeros(8, dtype=np.float32),
            gap_magnitude_features=np.zeros(7, dtype=np.float32),
            overnight_factor_features=np.zeros(15, dtype=np.float32),
            strike_correlation_features=np.zeros(10, dtype=np.float32),
            processing_time_ms=processing_time,
            confidence_score=0.5,
            timestamp=datetime.utcnow()
        )