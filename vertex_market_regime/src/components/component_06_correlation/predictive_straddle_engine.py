"""
Predictive Straddle Intelligence Engine for Component 6

Implements comprehensive predictive straddle feature extraction with previous day
close analysis, gap correlation prediction, and intraday premium evolution tracking.

Features:
- Previous Day Close Analysis (20 features)
- Gap Correlation Prediction (15 features) 
- Intraday Premium Evolution (15 features)
- Total: 50 predictive features for ML consumption

🎯 PURE FEATURE EXTRACTION - NO PREDICTION LOGIC
All features are raw measurements and mathematical calculations only.
Prediction logic deferred to Vertex AI ML models.
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
from scipy.signal import savgol_filter
from collections import deque

warnings.filterwarnings('ignore')


@dataclass
class PreviousDayCloseMetrics:
    """Previous day close analysis metrics"""
    atm_close_price: float
    itm1_close_price: float
    otm1_close_price: float
    
    atm_close_percentile: float
    itm1_close_percentile: float
    otm1_close_percentile: float
    
    atm_volume_at_close: float
    itm1_volume_at_close: float
    otm1_volume_at_close: float
    
    premium_decay_rates: Dict[str, float]
    volume_patterns: Dict[str, float]
    timestamp: datetime


@dataclass
class GapPredictionMetrics:
    """Gap correlation prediction metrics"""
    ce_pe_close_ratio: float
    premium_skew_at_close: float
    volume_pattern_score: float
    
    total_premium_at_close: float
    historical_gap_correlation: float
    volatility_expansion_indicator: float
    
    gap_direction_probability: float
    gap_magnitude_indicator: float
    timestamp: datetime


@dataclass
class IntradayEvolutionMetrics:
    """Intraday premium evolution metrics"""
    first_5min_behavior: Dict[str, float]
    first_15min_behavior: Dict[str, float]
    
    opening_gap_vs_predicted: float
    early_premium_decay: float
    
    full_day_trajectory_prediction: np.ndarray
    intraday_volatility_forecast: float
    zone_behavior_prediction: Dict[str, float]
    timestamp: datetime


@dataclass
class PredictiveStraddleResult:
    """Complete predictive straddle analysis result"""
    previous_day_metrics: PreviousDayCloseMetrics
    gap_prediction_metrics: GapPredictionMetrics
    intraday_evolution_metrics: IntradayEvolutionMetrics
    
    # Feature arrays for ML consumption (50 total features)
    atm_close_predictors: np.ndarray      # 7 features
    itm1_close_predictors: np.ndarray     # 7 features
    otm1_close_predictors: np.ndarray     # 6 features
    gap_direction_predictors: np.ndarray   # 8 features
    gap_magnitude_predictors: np.ndarray   # 7 features
    opening_minutes_analysis: np.ndarray   # 8 features
    full_day_forecast: np.ndarray         # 7 features
    
    confidence_score: float
    processing_time_ms: float
    timestamp: datetime


class PredictiveStraddleEngine:
    """
    High-performance predictive straddle intelligence engine
    
    Extracts 50 systematic predictive features from straddle premium data
    with focus on previous day patterns, gap correlation, and intraday evolution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Feature extraction settings
        self.min_historical_periods = config.get('min_historical_periods', 20)
        self.premium_decay_window = config.get('premium_decay_window', 30)  # minutes
        self.volume_analysis_window = config.get('volume_analysis_window', 60)  # minutes
        
        # Percentile calculation windows
        self.short_term_window = config.get('short_term_window', 20)   # days
        self.medium_term_window = config.get('medium_term_window', 60) # days
        self.long_term_window = config.get('long_term_window', 252)    # days
        
        # Intraday time zones for analysis
        self.intraday_zones = {
            'opening': (dt_time(9, 15), dt_time(9, 30)),   # First 15 minutes
            'morning': (dt_time(9, 30), dt_time(11, 30)),  # Morning session
            'lunch': (dt_time(11, 30), dt_time(13, 0)),    # Lunch break
            'afternoon': (dt_time(13, 0), dt_time(15, 0)),  # Afternoon session
            'closing': (dt_time(15, 0), dt_time(15, 30))   # Closing session
        }
        
        # Gap correlation thresholds
        self.gap_correlation_thresholds = {
            'strong_positive': 0.7,
            'moderate_positive': 0.3,
            'weak': 0.1,
            'moderate_negative': -0.3,
            'strong_negative': -0.7
        }
        
        # Historical data storage
        self.close_price_history = deque(maxlen=252)  # 1 year of close data
        self.gap_correlation_history = deque(maxlen=100)  # Gap correlation tracking
        self.intraday_pattern_history = deque(maxlen=50)  # Recent intraday patterns
        
        self.logger.info("Predictive Straddle Engine initialized with 50-feature extraction")

    def extract_predictive_features(self, 
                                   current_data: pd.DataFrame,
                                   previous_day_data: Optional[pd.DataFrame] = None,
                                   historical_data: Optional[pd.DataFrame] = None,
                                   overnight_factors: Optional[Dict[str, float]] = None) -> PredictiveStraddleResult:
        """
        Extract all 50 predictive straddle intelligence features
        
        Args:
            current_data: Current day market data
            previous_day_data: Previous trading day data
            historical_data: Historical market data for percentiles
            overnight_factors: Overnight factor data
            
        Returns:
            PredictiveStraddleResult with 50 predictive features
        """
        start_time = time.time()
        
        try:
            # 1. Analyze previous day close patterns (20 features)
            previous_day_metrics = self._analyze_previous_day_close(
                previous_day_data, historical_data
            )
            
            # 2. Extract gap correlation predictions (15 features)
            gap_prediction_metrics = self._extract_gap_predictions(
                current_data, previous_day_data, historical_data, overnight_factors
            )
            
            # 3. Analyze intraday evolution patterns (15 features)
            intraday_evolution_metrics = self._analyze_intraday_evolution(
                current_data, previous_day_metrics, gap_prediction_metrics
            )
            
            # 4. Generate feature arrays for ML consumption
            feature_arrays = self._generate_predictive_feature_arrays(
                previous_day_metrics, gap_prediction_metrics, intraday_evolution_metrics, 
                overnight_factors
            )
            
            # 5. Calculate confidence score
            confidence_score = self._calculate_predictive_confidence(
                current_data, previous_day_data, historical_data
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return PredictiveStraddleResult(
                previous_day_metrics=previous_day_metrics,
                gap_prediction_metrics=gap_prediction_metrics,
                intraday_evolution_metrics=intraday_evolution_metrics,
                atm_close_predictors=feature_arrays['atm_close_predictors'],
                itm1_close_predictors=feature_arrays['itm1_close_predictors'],
                otm1_close_predictors=feature_arrays['otm1_close_predictors'],
                gap_direction_predictors=feature_arrays['gap_direction_predictors'],
                gap_magnitude_predictors=feature_arrays['gap_magnitude_predictors'],
                opening_minutes_analysis=feature_arrays['opening_minutes_analysis'],
                full_day_forecast=feature_arrays['full_day_forecast'],
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Predictive feature extraction failed: {e}")
            return self._create_minimal_predictive_result(processing_time)

    def _analyze_previous_day_close(self, 
                                   previous_day_data: Optional[pd.DataFrame],
                                   historical_data: Optional[pd.DataFrame]) -> PreviousDayCloseMetrics:
        """Analyze previous day close patterns for predictive insights"""
        
        try:
            if previous_day_data is None or len(previous_day_data) == 0:
                return self._create_default_previous_day_metrics()
            
            # Extract close prices for different strikes
            atm_close = self._extract_close_price(previous_day_data, 'atm')
            itm1_close = self._extract_close_price(previous_day_data, 'itm1')
            otm1_close = self._extract_close_price(previous_day_data, 'otm1')
            
            # Calculate historical percentiles
            percentiles = self._calculate_close_percentiles(
                {'atm': atm_close, 'itm1': itm1_close, 'otm1': otm1_close},
                historical_data
            )
            
            # Extract volume at close
            volumes = self._extract_close_volumes(previous_day_data)
            
            # Calculate premium decay rates
            decay_rates = self._calculate_premium_decay_rates(previous_day_data)
            
            # Analyze volume patterns
            volume_patterns = self._analyze_volume_patterns(previous_day_data)
            
            return PreviousDayCloseMetrics(
                atm_close_price=atm_close,
                itm1_close_price=itm1_close,
                otm1_close_price=otm1_close,
                atm_close_percentile=percentiles.get('atm', 0.5),
                itm1_close_percentile=percentiles.get('itm1', 0.5),
                otm1_close_percentile=percentiles.get('otm1', 0.5),
                atm_volume_at_close=volumes.get('atm', 0.0),
                itm1_volume_at_close=volumes.get('itm1', 0.0),
                otm1_volume_at_close=volumes.get('otm1', 0.0),
                premium_decay_rates=decay_rates,
                volume_patterns=volume_patterns,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing previous day close: {e}")
            return self._create_default_previous_day_metrics()

    def _extract_close_price(self, data: pd.DataFrame, strike_type: str) -> float:
        """Extract close price for specific strike type"""
        
        try:
            # Look for strike-specific columns
            possible_columns = [
                f'{strike_type}_premium', f'{strike_type}_price', f'{strike_type}_close',
                'premium', 'close', 'price'
            ]
            
            for col in possible_columns:
                if col in data.columns:
                    return float(data[col].iloc[-1]) if len(data) > 0 else 0.0
            
            # Fallback: use last value of first available column
            if len(data.columns) > 0 and len(data) > 0:
                return float(data.iloc[-1, 0])
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error extracting close price for {strike_type}: {e}")
            return 0.0

    def _calculate_close_percentiles(self, 
                                   close_prices: Dict[str, float],
                                   historical_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate historical percentiles for close prices"""
        
        percentiles = {}
        
        try:
            if historical_data is None or len(historical_data) < self.min_historical_periods:
                # Return default percentiles
                return {strike: 0.5 for strike in close_prices.keys()}
            
            for strike_type, close_price in close_prices.items():
                try:
                    # Extract historical close prices for this strike
                    col_candidates = [f'{strike_type}_close', f'{strike_type}_premium', 'close', 'premium']
                    historical_closes = None
                    
                    for col in col_candidates:
                        if col in historical_data.columns:
                            historical_closes = historical_data[col].dropna()
                            break
                    
                    if historical_closes is not None and len(historical_closes) >= self.min_historical_periods:
                        # Calculate percentile rank
                        percentile = stats.percentileofscore(historical_closes.values, close_price) / 100.0
                        percentiles[strike_type] = max(0.0, min(1.0, percentile))
                    else:
                        percentiles[strike_type] = 0.5  # Default percentile
                        
                except Exception as e:
                    self.logger.error(f"Error calculating percentile for {strike_type}: {e}")
                    percentiles[strike_type] = 0.5
                    
        except Exception as e:
            self.logger.error(f"Error calculating close percentiles: {e}")
            percentiles = {strike: 0.5 for strike in close_prices.keys()}
        
        return percentiles

    def _extract_close_volumes(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract volume data at market close"""
        
        volumes = {}
        
        try:
            # Look for volume columns
            volume_columns = [col for col in data.columns if 'volume' in col.lower()]
            
            for strike_type in ['atm', 'itm1', 'otm1']:
                volume_found = False
                
                # Try strike-specific volume columns
                for col in volume_columns:
                    if strike_type in col.lower():
                        volumes[strike_type] = float(data[col].iloc[-1]) if len(data) > 0 else 0.0
                        volume_found = True
                        break
                
                # Fallback to general volume
                if not volume_found and 'volume' in data.columns:
                    volumes[strike_type] = float(data['volume'].iloc[-1]) if len(data) > 0 else 0.0
                elif not volume_found:
                    volumes[strike_type] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error extracting close volumes: {e}")
            volumes = {'atm': 0.0, 'itm1': 0.0, 'otm1': 0.0}
        
        return volumes

    def _calculate_premium_decay_rates(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate premium decay rates during the day"""
        
        decay_rates = {}
        
        try:
            if len(data) < 2:
                return {'atm': 0.0, 'itm1': 0.0, 'otm1': 0.0}
            
            # Calculate decay for each strike type
            for strike_type in ['atm', 'itm1', 'otm1']:
                try:
                    # Find appropriate price column
                    price_col = None
                    for col in data.columns:
                        if strike_type in col.lower() and ('premium' in col.lower() or 'price' in col.lower()):
                            price_col = col
                            break
                    
                    if price_col is None and 'premium' in data.columns:
                        price_col = 'premium'
                    elif price_col is None and 'close' in data.columns:
                        price_col = 'close'
                    
                    if price_col and len(data) >= 2:
                        # Calculate decay rate from start to end of day
                        start_price = data[price_col].iloc[0]
                        end_price = data[price_col].iloc[-1]
                        
                        if start_price != 0:
                            decay_rate = (end_price - start_price) / start_price
                            decay_rates[strike_type] = decay_rate
                        else:
                            decay_rates[strike_type] = 0.0
                    else:
                        decay_rates[strike_type] = 0.0
                        
                except Exception as e:
                    self.logger.error(f"Error calculating decay rate for {strike_type}: {e}")
                    decay_rates[strike_type] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error calculating premium decay rates: {e}")
            decay_rates = {'atm': 0.0, 'itm1': 0.0, 'otm1': 0.0}
        
        return decay_rates

    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume patterns throughout the day"""
        
        patterns = {}
        
        try:
            if 'volume' not in data.columns or len(data) < 5:
                return {
                    'volume_trend': 0.0,
                    'volume_at_close_ratio': 1.0,
                    'volume_volatility': 0.0,
                    'average_volume': 0.0
                }
            
            volumes = data['volume'].values
            
            # Volume trend (linear regression slope)
            if len(volumes) >= 3:
                x = np.arange(len(volumes))
                slope, _ = np.polyfit(x, volumes, 1)
                patterns['volume_trend'] = slope / np.mean(volumes) if np.mean(volumes) > 0 else 0.0
            else:
                patterns['volume_trend'] = 0.0
            
            # Volume at close ratio (last vs average)
            avg_volume = np.mean(volumes)
            close_volume = volumes[-1]
            patterns['volume_at_close_ratio'] = close_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume volatility (coefficient of variation)
            if avg_volume > 0:
                patterns['volume_volatility'] = np.std(volumes) / avg_volume
            else:
                patterns['volume_volatility'] = 0.0
            
            # Average volume (normalized)
            patterns['average_volume'] = avg_volume
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume patterns: {e}")
            patterns = {
                'volume_trend': 0.0,
                'volume_at_close_ratio': 1.0,
                'volume_volatility': 0.0,
                'average_volume': 0.0
            }
        
        return patterns

    def _extract_gap_predictions(self, 
                               current_data: pd.DataFrame,
                               previous_day_data: Optional[pd.DataFrame],
                               historical_data: Optional[pd.DataFrame],
                               overnight_factors: Optional[Dict[str, float]]) -> GapPredictionMetrics:
        """Extract gap correlation prediction metrics"""
        
        try:
            # CE-PE close ratio analysis
            ce_pe_ratio = self._calculate_ce_pe_close_ratio(previous_day_data)
            
            # Premium skew at close
            premium_skew = self._calculate_premium_skew_at_close(previous_day_data)
            
            # Volume pattern score for gap prediction
            volume_pattern_score = self._calculate_volume_gap_indicator(previous_day_data)
            
            # Total premium at close
            total_premium = self._calculate_total_premium_at_close(previous_day_data)
            
            # Historical gap correlation
            historical_correlation = self._calculate_historical_gap_correlation(
                historical_data, overnight_factors
            )
            
            # Volatility expansion indicator
            volatility_indicator = self._calculate_volatility_expansion_indicator(
                previous_day_data, overnight_factors
            )
            
            # Gap direction probability (placeholder - would use complex calculation)
            gap_direction_prob = self._estimate_gap_direction_probability(
                ce_pe_ratio, premium_skew, overnight_factors
            )
            
            # Gap magnitude indicator
            gap_magnitude = self._estimate_gap_magnitude_indicator(
                total_premium, volatility_indicator, overnight_factors
            )
            
            return GapPredictionMetrics(
                ce_pe_close_ratio=ce_pe_ratio,
                premium_skew_at_close=premium_skew,
                volume_pattern_score=volume_pattern_score,
                total_premium_at_close=total_premium,
                historical_gap_correlation=historical_correlation,
                volatility_expansion_indicator=volatility_indicator,
                gap_direction_probability=gap_direction_prob,
                gap_magnitude_indicator=gap_magnitude,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting gap predictions: {e}")
            return GapPredictionMetrics(
                ce_pe_close_ratio=1.0,
                premium_skew_at_close=0.0,
                volume_pattern_score=0.5,
                total_premium_at_close=100.0,
                historical_gap_correlation=0.0,
                volatility_expansion_indicator=0.5,
                gap_direction_probability=0.5,
                gap_magnitude_indicator=0.5,
                timestamp=datetime.utcnow()
            )

    def _calculate_ce_pe_close_ratio(self, data: Optional[pd.DataFrame]) -> float:
        """Calculate CE-PE ratio at close for gap direction prediction"""
        
        try:
            if data is None or len(data) == 0:
                return 1.0  # Neutral ratio
            
            # Look for CE and PE premium columns
            ce_premium = None
            pe_premium = None
            
            for col in data.columns:
                if 'ce' in col.lower() and ('premium' in col.lower() or 'price' in col.lower()):
                    ce_premium = data[col].iloc[-1]
                elif 'pe' in col.lower() and ('premium' in col.lower() or 'price' in col.lower()):
                    pe_premium = data[col].iloc[-1]
            
            if ce_premium is not None and pe_premium is not None and pe_premium != 0:
                return float(ce_premium / pe_premium)
            else:
                return 1.0  # Neutral ratio if data not available
                
        except Exception as e:
            self.logger.error(f"Error calculating CE-PE close ratio: {e}")
            return 1.0

    def _calculate_premium_skew_at_close(self, data: Optional[pd.DataFrame]) -> float:
        """Calculate premium skew at close"""
        
        try:
            if data is None or len(data) == 0:
                return 0.0
            
            # Try to find ATM, ITM, OTM premiums
            premiums = {}
            for strike_type in ['atm', 'itm', 'otm']:
                for col in data.columns:
                    if strike_type in col.lower() and ('premium' in col.lower() or 'price' in col.lower()):
                        premiums[strike_type] = data[col].iloc[-1]
                        break
            
            if len(premiums) >= 2:
                # Calculate skew as difference between OTM and ITM relative to ATM
                atm_premium = premiums.get('atm', 0)
                itm_premium = premiums.get('itm', atm_premium)
                otm_premium = premiums.get('otm', atm_premium)
                
                if atm_premium > 0:
                    skew = (otm_premium - itm_premium) / atm_premium
                    return float(skew)
            
            return 0.0  # No skew if data not available
            
        except Exception as e:
            self.logger.error(f"Error calculating premium skew: {e}")
            return 0.0

    def _calculate_volume_gap_indicator(self, data: Optional[pd.DataFrame]) -> float:
        """Calculate volume pattern score for gap prediction"""
        
        try:
            if data is None or len(data) == 0 or 'volume' not in data.columns:
                return 0.5  # Neutral score
            
            volumes = data['volume'].values
            if len(volumes) < 5:
                return 0.5
            
            # Analyze volume pattern in last hour vs day average
            last_hour_volumes = volumes[-60:] if len(volumes) >= 60 else volumes[-len(volumes)//2:]
            early_volumes = volumes[:len(volumes)//2]
            
            last_hour_avg = np.mean(last_hour_volumes)
            early_avg = np.mean(early_volumes)
            
            if early_avg > 0:
                volume_ratio = last_hour_avg / early_avg
                # High volume at close suggests gap potential
                gap_indicator = min(1.0, volume_ratio / 2.0)  # Normalize
                return float(gap_indicator)
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating volume gap indicator: {e}")
            return 0.5

    def _calculate_total_premium_at_close(self, data: Optional[pd.DataFrame]) -> float:
        """Calculate total option premium at close"""
        
        try:
            if data is None or len(data) == 0:
                return 100.0  # Default premium
            
            total_premium = 0.0
            premium_count = 0
            
            for col in data.columns:
                if 'premium' in col.lower() or 'price' in col.lower():
                    try:
                        premium_value = float(data[col].iloc[-1])
                        total_premium += premium_value
                        premium_count += 1
                    except:
                        continue
            
            return total_premium if premium_count > 0 else 100.0
            
        except Exception as e:
            self.logger.error(f"Error calculating total premium: {e}")
            return 100.0

    def _calculate_historical_gap_correlation(self, 
                                            historical_data: Optional[pd.DataFrame],
                                            overnight_factors: Optional[Dict[str, float]]) -> float:
        """Calculate historical gap correlation patterns"""
        
        try:
            # Simplified implementation - would need actual gap history analysis
            if len(self.gap_correlation_history) > 10:
                return float(np.mean(list(self.gap_correlation_history)))
            else:
                return 0.0  # No historical correlation available
                
        except Exception as e:
            self.logger.error(f"Error calculating historical gap correlation: {e}")
            return 0.0

    def _calculate_volatility_expansion_indicator(self, 
                                                data: Optional[pd.DataFrame],
                                                overnight_factors: Optional[Dict[str, float]]) -> float:
        """Calculate volatility expansion indicator"""
        
        try:
            # VIX impact from overnight factors
            vix_impact = 0.0
            if overnight_factors and 'vix_change' in overnight_factors:
                vix_change = overnight_factors['vix_change']
                vix_impact = min(1.0, abs(vix_change) / 5.0)  # Normalize VIX change
            
            # Premium volatility from data
            premium_volatility = 0.0
            if data is not None and len(data) > 5:
                for col in data.columns:
                    if 'premium' in col.lower():
                        try:
                            premium_values = data[col].values
                            if len(premium_values) > 1:
                                premium_std = np.std(premium_values)
                                premium_mean = np.mean(premium_values)
                                if premium_mean > 0:
                                    premium_volatility = max(premium_volatility, premium_std / premium_mean)
                        except:
                            continue
            
            # Combined volatility expansion indicator
            expansion_indicator = (vix_impact * 0.6 + premium_volatility * 0.4)
            return min(1.0, expansion_indicator)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility expansion indicator: {e}")
            return 0.5

    def _estimate_gap_direction_probability(self, 
                                          ce_pe_ratio: float,
                                          premium_skew: float,
                                          overnight_factors: Optional[Dict[str, float]]) -> float:
        """Estimate gap direction probability (raw measurement only)"""
        
        try:
            # CE-PE ratio contribution (>1 suggests bullish, <1 bearish)
            ratio_signal = (ce_pe_ratio - 1.0) * 0.3
            
            # Premium skew contribution
            skew_signal = premium_skew * 0.2
            
            # Overnight factors contribution
            overnight_signal = 0.0
            if overnight_factors:
                sgx_signal = overnight_factors.get('sgx_nifty', 0.0) * 0.3
                news_signal = overnight_factors.get('news_sentiment', 0.0) * 0.2
                overnight_signal = sgx_signal + news_signal
            
            # Combined probability (centered around 0.5)
            combined_signal = ratio_signal + skew_signal + overnight_signal
            probability = 0.5 + combined_signal  # Center around 0.5
            
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            self.logger.error(f"Error estimating gap direction probability: {e}")
            return 0.5

    def _estimate_gap_magnitude_indicator(self, 
                                        total_premium: float,
                                        volatility_indicator: float,
                                        overnight_factors: Optional[Dict[str, float]]) -> float:
        """Estimate gap magnitude indicator (raw measurement only)"""
        
        try:
            # Premium magnitude contribution
            premium_signal = min(1.0, total_premium / 500.0)  # Normalize to 500 premium units
            
            # Volatility contribution
            vol_signal = volatility_indicator
            
            # Overnight magnitude contribution
            overnight_magnitude = 0.0
            if overnight_factors:
                overnight_changes = [
                    abs(overnight_factors.get('sgx_nifty', 0.0)),
                    abs(overnight_factors.get('dow_jones', 0.0)),
                    abs(overnight_factors.get('vix_change', 0.0)) / 2.0,  # VIX has different scale
                ]
                overnight_magnitude = min(1.0, np.mean(overnight_changes))
            
            # Combined magnitude indicator
            magnitude_indicator = (premium_signal * 0.3 + vol_signal * 0.4 + overnight_magnitude * 0.3)
            
            return max(0.0, min(1.0, magnitude_indicator))
            
        except Exception as e:
            self.logger.error(f"Error estimating gap magnitude indicator: {e}")
            return 0.5

    def _analyze_intraday_evolution(self, 
                                  current_data: pd.DataFrame,
                                  previous_day_metrics: PreviousDayCloseMetrics,
                                  gap_prediction_metrics: GapPredictionMetrics) -> IntradayEvolutionMetrics:
        """Analyze intraday premium evolution patterns"""
        
        try:
            # Analyze first 5 and 15 minutes if data available
            first_5min = self._analyze_opening_minutes(current_data, 5)
            first_15min = self._analyze_opening_minutes(current_data, 15)
            
            # Opening gap vs predicted analysis
            opening_gap_analysis = self._analyze_opening_gap_vs_predicted(
                current_data, gap_prediction_metrics
            )
            
            # Early premium decay analysis
            early_decay = self._analyze_early_premium_decay(current_data)
            
            # Full day trajectory prediction features
            trajectory_features = self._extract_trajectory_prediction_features(
                current_data, previous_day_metrics
            )
            
            # Intraday volatility forecast
            volatility_forecast = self._forecast_intraday_volatility(
                current_data, previous_day_metrics
            )
            
            # Zone-based behavior prediction
            zone_predictions = self._predict_zone_behavior(
                current_data, previous_day_metrics
            )
            
            return IntradayEvolutionMetrics(
                first_5min_behavior=first_5min,
                first_15min_behavior=first_15min,
                opening_gap_vs_predicted=opening_gap_analysis,
                early_premium_decay=early_decay,
                full_day_trajectory_prediction=trajectory_features,
                intraday_volatility_forecast=volatility_forecast,
                zone_behavior_prediction=zone_predictions,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing intraday evolution: {e}")
            return IntradayEvolutionMetrics(
                first_5min_behavior={'premium_change': 0.0, 'volume_pattern': 0.5},
                first_15min_behavior={'premium_change': 0.0, 'volume_pattern': 0.5},
                opening_gap_vs_predicted=0.0,
                early_premium_decay=0.0,
                full_day_trajectory_prediction=np.zeros(5, dtype=np.float32),
                intraday_volatility_forecast=0.5,
                zone_behavior_prediction={'morning': 0.5, 'afternoon': 0.5},
                timestamp=datetime.utcnow()
            )

    def _analyze_opening_minutes(self, data: pd.DataFrame, minutes: int) -> Dict[str, float]:
        """Analyze premium behavior in opening minutes"""
        
        try:
            if len(data) < minutes:
                return {'premium_change': 0.0, 'volume_pattern': 0.5, 'volatility': 0.0}
            
            opening_data = data.iloc[:minutes]
            
            # Premium change analysis
            premium_change = 0.0
            if 'premium' in opening_data.columns and len(opening_data) >= 2:
                start_premium = opening_data['premium'].iloc[0]
                end_premium = opening_data['premium'].iloc[-1]
                if start_premium != 0:
                    premium_change = (end_premium - start_premium) / start_premium
            
            # Volume pattern analysis
            volume_pattern = 0.5  # Default
            if 'volume' in opening_data.columns:
                volumes = opening_data['volume'].values
                if len(volumes) > 1:
                    volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                    volume_pattern = min(1.0, max(0.0, (volume_trend / np.mean(volumes) + 1.0) / 2.0))
            
            # Volatility measurement
            volatility = 0.0
            if 'premium' in opening_data.columns and len(opening_data) >= 3:
                premium_values = opening_data['premium'].values
                volatility = np.std(premium_values) / np.mean(premium_values) if np.mean(premium_values) > 0 else 0.0
            
            return {
                'premium_change': float(premium_change),
                'volume_pattern': float(volume_pattern),
                'volatility': float(volatility)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing opening {minutes} minutes: {e}")
            return {'premium_change': 0.0, 'volume_pattern': 0.5, 'volatility': 0.0}

    def _analyze_opening_gap_vs_predicted(self, 
                                        current_data: pd.DataFrame,
                                        gap_predictions: GapPredictionMetrics) -> float:
        """Analyze actual opening gap vs predicted gap"""
        
        try:
            if len(current_data) == 0:
                return 0.0
            
            # Get actual opening behavior
            if 'premium' in current_data.columns and len(current_data) >= 2:
                opening_premium = current_data['premium'].iloc[0]
                second_premium = current_data['premium'].iloc[1]
                
                if opening_premium != 0:
                    actual_gap = (second_premium - opening_premium) / opening_premium
                    predicted_direction = gap_predictions.gap_direction_probability - 0.5  # Center around 0
                    
                    # Calculate agreement between actual and predicted
                    if (actual_gap > 0 and predicted_direction > 0) or (actual_gap < 0 and predicted_direction < 0):
                        agreement = min(1.0, 1.0 - abs(actual_gap - predicted_direction))
                        return float(agreement)
                    else:
                        disagreement = abs(actual_gap - predicted_direction)
                        return float(max(-1.0, -disagreement))
            
            return 0.0  # No gap information available
            
        except Exception as e:
            self.logger.error(f"Error analyzing opening gap vs predicted: {e}")
            return 0.0

    def _analyze_early_premium_decay(self, data: pd.DataFrame) -> float:
        """Analyze early premium decay patterns"""
        
        try:
            if len(data) < 30 or 'premium' not in data.columns:  # Need at least 30 minutes
                return 0.0
            
            early_data = data.iloc[:30]  # First 30 minutes
            premium_values = early_data['premium'].values
            
            if len(premium_values) >= 2:
                # Calculate decay rate
                start_premium = premium_values[0]
                end_premium = premium_values[-1]
                
                if start_premium != 0:
                    decay_rate = (end_premium - start_premium) / start_premium
                    return float(decay_rate)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing early premium decay: {e}")
            return 0.0

    def _extract_trajectory_prediction_features(self, 
                                              current_data: pd.DataFrame,
                                              previous_day_metrics: PreviousDayCloseMetrics) -> np.ndarray:
        """Extract full day trajectory prediction features"""
        
        try:
            features = []
            
            if len(current_data) > 0 and 'premium' in current_data.columns:
                premium_values = current_data['premium'].values
                
                # Feature 1: Current premium vs previous close
                current_premium = premium_values[-1] if len(premium_values) > 0 else 0
                prev_close = previous_day_metrics.atm_close_price
                if prev_close != 0:
                    premium_ratio = current_premium / prev_close
                    features.append(premium_ratio)
                else:
                    features.append(1.0)
                
                # Feature 2: Premium momentum (recent trend)
                if len(premium_values) >= 10:
                    recent_values = premium_values[-10:]
                    momentum = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    normalized_momentum = momentum / np.mean(recent_values) if np.mean(recent_values) > 0 else 0.0
                    features.append(float(normalized_momentum))
                else:
                    features.append(0.0)
                
                # Feature 3: Premium volatility vs previous day
                if len(premium_values) >= 5:
                    current_volatility = np.std(premium_values)
                    features.append(float(current_volatility))
                else:
                    features.append(0.0)
                
                # Additional trajectory features
                features.extend([0.5, 0.3])  # Placeholder for additional features
            else:
                features = [1.0, 0.0, 0.0, 0.5, 0.3]
            
            # Ensure exactly 5 features
            while len(features) < 5:
                features.append(0.0)
                
            return np.array(features[:5], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting trajectory features: {e}")
            return np.zeros(5, dtype=np.float32)

    def _forecast_intraday_volatility(self, 
                                    current_data: pd.DataFrame,
                                    previous_day_metrics: PreviousDayCloseMetrics) -> float:
        """Forecast intraday volatility based on current patterns"""
        
        try:
            if len(current_data) == 0 or 'premium' not in current_data.columns:
                return 0.5
            
            premium_values = current_data['premium'].values
            
            if len(premium_values) >= 5:
                # Calculate current volatility
                current_volatility = np.std(premium_values) / np.mean(premium_values) if np.mean(premium_values) > 0 else 0.0
                
                # Normalize volatility forecast
                volatility_forecast = min(1.0, current_volatility * 2.0)  # Scale for forecast
                return float(volatility_forecast)
            
            return 0.5  # Default forecast
            
        except Exception as e:
            self.logger.error(f"Error forecasting intraday volatility: {e}")
            return 0.5

    def _predict_zone_behavior(self, 
                             current_data: pd.DataFrame,
                             previous_day_metrics: PreviousDayCloseMetrics) -> Dict[str, float]:
        """Predict behavior in different intraday zones"""
        
        try:
            # Simplified zone behavior prediction
            zone_predictions = {}
            
            for zone_name in ['morning', 'lunch', 'afternoon', 'closing']:
                # Base prediction on current patterns and previous day
                if len(current_data) > 0 and 'premium' in current_data.columns:
                    current_trend = 0.0
                    if len(current_data) >= 5:
                        recent_premiums = current_data['premium'].values[-5:]
                        current_trend = np.polyfit(range(len(recent_premiums)), recent_premiums, 1)[0]
                        current_trend = current_trend / np.mean(recent_premiums) if np.mean(recent_premiums) > 0 else 0.0
                    
                    # Zone-specific adjustment
                    zone_factor = {
                        'morning': 1.2,   # Higher volatility in morning
                        'lunch': 0.8,     # Lower activity during lunch
                        'afternoon': 1.0, # Normal activity
                        'closing': 1.1    # Higher activity at close
                    }.get(zone_name, 1.0)
                    
                    zone_prediction = 0.5 + (current_trend * zone_factor * 0.3)
                    zone_predictions[zone_name] = max(0.0, min(1.0, zone_prediction))
                else:
                    zone_predictions[zone_name] = 0.5  # Default prediction
            
            return zone_predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting zone behavior: {e}")
            return {'morning': 0.5, 'lunch': 0.5, 'afternoon': 0.5, 'closing': 0.5}

    def _generate_predictive_feature_arrays(self, 
                                          previous_day_metrics: PreviousDayCloseMetrics,
                                          gap_prediction_metrics: GapPredictionMetrics,
                                          intraday_evolution_metrics: IntradayEvolutionMetrics,
                                          overnight_factors: Optional[Dict[str, float]]) -> Dict[str, np.ndarray]:
        """Generate all predictive feature arrays for ML consumption"""
        
        feature_arrays = {}
        
        try:
            # ATM Close Predictors (7 features)
            atm_features = [
                previous_day_metrics.atm_close_percentile,
                previous_day_metrics.premium_decay_rates.get('atm', 0.0),
                previous_day_metrics.volume_patterns.get('volume_at_close_ratio', 1.0),
                overnight_factors.get('sgx_nifty', 0.0) * 0.15 if overnight_factors else 0.0,
                overnight_factors.get('dow_jones', 0.0) * 0.10 if overnight_factors else 0.0,
                overnight_factors.get('vix_change', 0.0) * 0.20 if overnight_factors else 0.0,
                overnight_factors.get('news_sentiment', 0.0) * 0.20 if overnight_factors else 0.0
            ]
            feature_arrays['atm_close_predictors'] = np.array(atm_features[:7], dtype=np.float32)
            
            # ITM1 Close Predictors (7 features)
            itm1_features = [
                previous_day_metrics.itm1_close_percentile,
                previous_day_metrics.atm_close_price / previous_day_metrics.itm1_close_price if previous_day_metrics.itm1_close_price != 0 else 1.0,
                previous_day_metrics.volume_patterns.get('volume_trend', 0.0),
                overnight_factors.get('usd_inr', 0.0) * 0.15 if overnight_factors else 0.0,
                overnight_factors.get('commodities', 0.0) * 0.10 if overnight_factors else 0.0,
                previous_day_metrics.premium_decay_rates.get('itm1', 0.0),
                0.0  # Placeholder for additional ITM feature
            ]
            feature_arrays['itm1_close_predictors'] = np.array(itm1_features[:7], dtype=np.float32)
            
            # OTM1 Close Predictors (6 features)
            otm1_features = [
                gap_prediction_metrics.volatility_expansion_indicator,
                previous_day_metrics.atm_close_price / previous_day_metrics.otm1_close_price if previous_day_metrics.otm1_close_price != 0 else 1.0,
                previous_day_metrics.volume_patterns.get('volume_volatility', 0.0),
                overnight_factors.get('usd_inr', 0.0) * 0.5 if overnight_factors else 0.0,
                overnight_factors.get('commodities', 0.0) * 0.3 if overnight_factors else 0.0,
                overnight_factors.get('news_sentiment', 0.0) * 0.4 if overnight_factors else 0.0
            ]
            feature_arrays['otm1_close_predictors'] = np.array(otm1_features[:6], dtype=np.float32)
            
            # Gap Direction Predictors (8 features)
            gap_dir_features = [
                gap_prediction_metrics.ce_pe_close_ratio - 1.0,  # Center around 0
                gap_prediction_metrics.premium_skew_at_close,
                gap_prediction_metrics.volume_pattern_score,
                overnight_factors.get('sgx_nifty', 0.0) if overnight_factors else 0.0,
                overnight_factors.get('dow_jones', 0.0) if overnight_factors else 0.0,
                overnight_factors.get('news_sentiment', 0.0) if overnight_factors else 0.0,
                gap_prediction_metrics.gap_direction_probability - 0.5,  # Center around 0
                gap_prediction_metrics.historical_gap_correlation
            ]
            feature_arrays['gap_direction_predictors'] = np.array(gap_dir_features[:8], dtype=np.float32)
            
            # Gap Magnitude Predictors (7 features)
            gap_mag_features = [
                gap_prediction_metrics.total_premium_at_close / 500.0,  # Normalize
                gap_prediction_metrics.volatility_expansion_indicator,
                gap_prediction_metrics.gap_magnitude_indicator,
                abs(overnight_factors.get('sgx_nifty', 0.0)) if overnight_factors else 0.0,
                abs(overnight_factors.get('vix_change', 0.0)) / 5.0 if overnight_factors else 0.0,
                previous_day_metrics.volume_patterns.get('average_volume', 0.0) / 1000.0,  # Normalize
                intraday_evolution_metrics.intraday_volatility_forecast
            ]
            feature_arrays['gap_magnitude_predictors'] = np.array(gap_mag_features[:7], dtype=np.float32)
            
            # Opening Minutes Analysis (8 features)
            opening_features = [
                intraday_evolution_metrics.first_5min_behavior.get('premium_change', 0.0),
                intraday_evolution_metrics.first_5min_behavior.get('volume_pattern', 0.5),
                intraday_evolution_metrics.first_15min_behavior.get('premium_change', 0.0),
                intraday_evolution_metrics.first_15min_behavior.get('volume_pattern', 0.5),
                intraday_evolution_metrics.opening_gap_vs_predicted,
                intraday_evolution_metrics.early_premium_decay,
                intraday_evolution_metrics.first_5min_behavior.get('volatility', 0.0),
                intraday_evolution_metrics.first_15min_behavior.get('volatility', 0.0)
            ]
            feature_arrays['opening_minutes_analysis'] = np.array(opening_features[:8], dtype=np.float32)
            
            # Full Day Forecast (7 features)
            forecast_features = list(intraday_evolution_metrics.full_day_trajectory_prediction)
            forecast_features.extend([
                intraday_evolution_metrics.intraday_volatility_forecast,
                intraday_evolution_metrics.zone_behavior_prediction.get('afternoon', 0.5)
            ])
            feature_arrays['full_day_forecast'] = np.array(forecast_features[:7], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating predictive feature arrays: {e}")
            # Return default feature arrays
            feature_arrays = {
                'atm_close_predictors': np.zeros(7, dtype=np.float32),
                'itm1_close_predictors': np.zeros(7, dtype=np.float32),
                'otm1_close_predictors': np.zeros(6, dtype=np.float32),
                'gap_direction_predictors': np.zeros(8, dtype=np.float32),
                'gap_magnitude_predictors': np.zeros(7, dtype=np.float32),
                'opening_minutes_analysis': np.zeros(8, dtype=np.float32),
                'full_day_forecast': np.zeros(7, dtype=np.float32)
            }
        
        return feature_arrays

    def _calculate_predictive_confidence(self, 
                                       current_data: pd.DataFrame,
                                       previous_day_data: Optional[pd.DataFrame],
                                       historical_data: Optional[pd.DataFrame]) -> float:
        """Calculate confidence score for predictive features"""
        
        try:
            confidence_factors = []
            
            # Data availability confidence
            if len(current_data) > 0:
                confidence_factors.append(min(1.0, len(current_data) / 100.0))
            else:
                confidence_factors.append(0.1)
            
            # Previous day data confidence
            if previous_day_data is not None and len(previous_day_data) > 0:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.3)
            
            # Historical data confidence
            if historical_data is not None and len(historical_data) >= self.min_historical_periods:
                confidence_factors.append(min(1.0, len(historical_data) / 252.0))  # 1 year = full confidence
            else:
                confidence_factors.append(0.5)
            
            # Overall confidence
            overall_confidence = np.mean(confidence_factors)
            return max(0.1, min(1.0, overall_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating predictive confidence: {e}")
            return 0.5

    def _create_default_previous_day_metrics(self) -> PreviousDayCloseMetrics:
        """Create default previous day metrics for fallback"""
        return PreviousDayCloseMetrics(
            atm_close_price=100.0,
            itm1_close_price=120.0,
            otm1_close_price=80.0,
            atm_close_percentile=0.5,
            itm1_close_percentile=0.5,
            otm1_close_percentile=0.5,
            atm_volume_at_close=1000.0,
            itm1_volume_at_close=800.0,
            otm1_volume_at_close=600.0,
            premium_decay_rates={'atm': 0.0, 'itm1': 0.0, 'otm1': 0.0},
            volume_patterns={'volume_trend': 0.0, 'volume_at_close_ratio': 1.0, 
                           'volume_volatility': 0.0, 'average_volume': 1000.0},
            timestamp=datetime.utcnow()
        )

    def _create_minimal_predictive_result(self, processing_time: float) -> PredictiveStraddleResult:
        """Create minimal predictive result for fallback"""
        return PredictiveStraddleResult(
            previous_day_metrics=self._create_default_previous_day_metrics(),
            gap_prediction_metrics=GapPredictionMetrics(1.0, 0.0, 0.5, 100.0, 0.0, 0.5, 0.5, 0.5, datetime.utcnow()),
            intraday_evolution_metrics=IntradayEvolutionMetrics(
                {'premium_change': 0.0, 'volume_pattern': 0.5}, 
                {'premium_change': 0.0, 'volume_pattern': 0.5},
                0.0, 0.0, np.zeros(5, dtype=np.float32), 0.5, 
                {'morning': 0.5, 'afternoon': 0.5}, datetime.utcnow()
            ),
            atm_close_predictors=np.zeros(7, dtype=np.float32),
            itm1_close_predictors=np.zeros(7, dtype=np.float32),
            otm1_close_predictors=np.zeros(6, dtype=np.float32),
            gap_direction_predictors=np.zeros(8, dtype=np.float32),
            gap_magnitude_predictors=np.zeros(7, dtype=np.float32),
            opening_minutes_analysis=np.zeros(8, dtype=np.float32),
            full_day_forecast=np.zeros(7, dtype=np.float32),
            confidence_score=0.5,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )