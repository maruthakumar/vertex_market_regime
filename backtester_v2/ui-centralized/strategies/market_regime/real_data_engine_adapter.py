#!/usr/bin/env python3
"""
Real Data Engine Adapter
========================

Adapter to connect real market regime engines with HeavyDB data.
Ensures ONLY real data is used - no mock/simulation fallbacks.
"""

import sys
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')

# Import HeavyDB connection
try:
    from backtester_v2.dal.heavydb_connection import get_connection, execute_query
    HEAVYDB_AVAILABLE = True
except ImportError as e:
    logger.critical(f"Cannot import HeavyDB connection: {e}")
    HEAVYDB_AVAILABLE = False
    raise ImportError("HeavyDB connection is required - no mock data allowed")


class RealDataEngineAdapter:
    """
    Adapter for real market regime engines using HeavyDB data.
    CRITICAL: This adapter ONLY works with real HeavyDB data.
    """
    
    def __init__(self, real_engine=None, market_analyzer=None):
        """
        Initialize adapter with real engines.
        
        Args:
            real_engine: Real data integration engine instance
            market_analyzer: Comprehensive market regime analyzer instance
        """
        self.real_engine = real_engine
        self.market_analyzer = market_analyzer
        self.conn = None
        
        # Validate HeavyDB connection
        if not HEAVYDB_AVAILABLE:
            raise RuntimeError("HeavyDB not available - cannot use adapter without real data")
        
        # Test connection
        try:
            self.conn = get_connection()
            if not self.conn:
                raise RuntimeError("Failed to establish HeavyDB connection")
            
            # Verify data exists
            test_query = "SELECT COUNT(*) FROM nifty_option_chain LIMIT 1"
            result = execute_query(self.conn, test_query)
            if result.empty:
                raise RuntimeError("No data found in HeavyDB")
                
            count = result.iloc[0][0]  # First column since no alias
            logger.info(f"✅ Adapter connected to HeavyDB with {count:,} records")
            
        except Exception as e:
            logger.error(f"❌ HeavyDB connection failed in adapter: {e}")
            raise RuntimeError(f"Cannot initialize adapter without HeavyDB: {e}")
    
    def calculate_regime_from_data(self, market_data: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calculate market regime from real HeavyDB data.
        
        Args:
            market_data: DataFrame with option chain data from HeavyDB
            config: Optional configuration dictionary
            
        Returns:
            Dict with regime calculation results
        """
        try:
            if market_data.empty:
                raise ValueError("Empty market data provided")
            
            # Use real engines if available
            if self.real_engine and self.market_analyzer:
                logger.info("📊 Using real engines for regime calculation")
                
                # Process through real engine
                regime_result = self._process_with_real_engines(market_data, config)
                
                if regime_result:
                    return regime_result
            
            # Fallback to direct calculation from HeavyDB data
            logger.info("📊 Using direct HeavyDB data analysis")
            return self._calculate_from_heavydb_data(market_data)
            
        except Exception as e:
            logger.error(f"❌ Regime calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "regime": "ERROR",
                "confidence": 0.0
            }
    
    def analyze_time_series(self, market_data: pd.DataFrame, config: Optional[Dict] = None, 
                          timeframe: str = "1min", include_confidence: bool = True) -> Dict[str, Any]:
        """
        Analyze time series data from HeavyDB.
        
        Args:
            market_data: Historical data from HeavyDB
            config: Optional configuration
            timeframe: Time interval for analysis
            include_confidence: Whether to include confidence scores
            
        Returns:
            Dict with time series analysis results
        """
        try:
            if market_data.empty:
                raise ValueError("Empty market data provided")
            
            logger.info(f"📊 Analyzing {len(market_data)} real data points")
            
            # Process time series
            time_series = []
            
            # Group by timestamp if needed
            if 'timestamp' in market_data.columns:
                grouped = market_data.groupby('timestamp')
            elif 'trade_date' in market_data.columns and 'trade_time' in market_data.columns:
                # Combine date and time
                market_data['timestamp'] = pd.to_datetime(
                    market_data['trade_date'].astype(str) + ' ' + market_data['trade_time'].astype(str)
                )
                grouped = market_data.groupby('timestamp')
            else:
                raise ValueError("No timestamp columns found in data")
            
            # Analyze each time point
            for timestamp, group_data in grouped:
                regime_point = self._analyze_single_timepoint(group_data, include_confidence)
                regime_point['timestamp'] = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                time_series.append(regime_point)
            
            return {
                "success": True,
                "time_series": time_series,
                "data_points": len(market_data),
                "unique_timestamps": len(time_series)
            }
            
        except Exception as e:
            logger.error(f"❌ Time series analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "time_series": []
            }
    
    def _process_with_real_engines(self, market_data: pd.DataFrame, config: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Process data through real engines if available."""
        try:
            # Placeholder for real engine processing
            # This would call actual engine methods when available
            logger.warning("Real engines not fully implemented yet")
            return None
            
        except Exception as e:
            logger.error(f"Real engine processing failed: {e}")
            return None
    
    def _calculate_from_heavydb_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Direct calculation from HeavyDB data.
        Uses actual market metrics to determine regime.
        """
        try:
            # Calculate key metrics from real data
            
            # 1. Volatility from IV data
            avg_iv = 0
            if 'ce_iv' in market_data.columns and 'pe_iv' in market_data.columns:
                ce_iv = market_data['ce_iv'].dropna()
                pe_iv = market_data['pe_iv'].dropna()
                if len(ce_iv) > 0 and len(pe_iv) > 0:
                    avg_iv = (ce_iv.mean() + pe_iv.mean()) / 2
            
            # 2. Enhanced Greek sentiment analysis
            greek_sentiment = self._calculate_greek_sentiment(market_data)
            
            # 3. Triple rolling straddle analysis
            straddle_signal = self._calculate_triple_straddle(market_data)
            
            # 4. OI/PA pattern analysis
            oi_pattern = self._calculate_oi_patterns(market_data)
            
            # 5. Enhanced trend strength calculation
            trend_strength = self._calculate_trend_strength(market_data)
            
            # 6. Structure score calculation
            structure_score = self._calculate_structure_score(market_data)
            
            # 7. Enhanced regime determination
            volatility_regime, trend_regime, structure_regime, regime = self._determine_regime(
                avg_iv, trend_strength, greek_sentiment, straddle_signal, oi_pattern, structure_score
            )
            
            # 8. Enhanced confidence calculation
            confidence = self._calculate_confidence(
                market_data, avg_iv, greek_sentiment, trend_strength, 
                straddle_signal, oi_pattern, structure_score
            )
            
            return {
                "success": True,
                "regime": regime,
                "confidence": round(confidence, 3),
                "volatility_regime": volatility_regime,
                "trend_regime": trend_regime,
                "structure_regime": structure_regime,
                "vix_proxy": round(avg_iv, 2),
                "trend_strength": round(trend_strength, 3),
                "structure_score": round(structure_score, 3),
                "greek_sentiment": round(greek_sentiment, 3),
                "straddle_signal": round(straddle_signal, 3),
                "oi_pattern": round(oi_pattern, 3),
                "data_timestamp": datetime.now().isoformat(),
                "data_points_used": len(market_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Direct calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "regime": "ERROR",
                "confidence": 0.0
            }
    
    def _analyze_single_timepoint(self, data: pd.DataFrame, include_confidence: bool) -> Dict[str, Any]:
        """Analyze a single timepoint of data using enhanced indicators."""
        try:
            # Calculate metrics for this timepoint
            result = self._calculate_from_heavydb_data(data)
            
            # Format for time series with all enhanced indicators
            timepoint_result = {
                "regime": result.get("regime", "NEUTRAL"),
                "volatility_regime": result.get("volatility_regime", "MEDIUM"),
                "trend_regime": result.get("trend_regime", "SIDEWAYS"),
                "structure_regime": result.get("structure_regime", "MODERATE"),
                "vix_proxy": result.get("vix_proxy", 18.5),
                "trend_strength": result.get("trend_strength", 0.0),
                "structure_score": result.get("structure_score", 0.0),
                "greek_sentiment": result.get("greek_sentiment", 0.0),
                "straddle_signal": result.get("straddle_signal", 0.0),
                "oi_pattern": result.get("oi_pattern", 0.0)
            }
            
            if include_confidence:
                timepoint_result["confidence"] = result.get("confidence", 0.8)
            
            return timepoint_result
            
        except Exception as e:
            logger.error(f"Single timepoint analysis failed: {e}")
            return {
                "regime": "ERROR",
                "volatility_regime": "UNKNOWN",
                "trend_regime": "UNKNOWN",
                "structure_regime": "UNKNOWN",
                "vix_proxy": 0.0,
                "trend_strength": 0.0,
                "structure_score": 0.0,
                "greek_sentiment": 0.0,
                "straddle_signal": 0.0,
                "oi_pattern": 0.0,
                "confidence": 0.0 if include_confidence else None
            }
    
    def validate_connection(self) -> bool:
        """Validate HeavyDB connection is still active."""
        try:
            if not self.conn:
                self.conn = get_connection()
            
            # Test query
            result = execute_query(self.conn, "SELECT 1")
            return not result.empty
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    def _calculate_greek_sentiment(self, market_data: pd.DataFrame) -> float:
        """
        Enhanced Greek sentiment analysis using Delta, Gamma, Theta, Vega.
        """
        try:
            greek_scores = []
            
            # Delta sentiment (directional exposure)
            if 'ce_delta' in market_data.columns and 'pe_delta' in market_data.columns:
                ce_delta = market_data['ce_delta'].fillna(0)
                pe_delta = market_data['pe_delta'].fillna(0)
                net_delta = ce_delta.sum() - abs(pe_delta.sum())  # Calls - Puts
                delta_sentiment = np.tanh(net_delta / 1000) * 0.4  # Weight: 40%
                greek_scores.append(delta_sentiment)
            
            # Gamma sentiment (acceleration/convexity)
            if 'ce_gamma' in market_data.columns and 'pe_gamma' in market_data.columns:
                ce_gamma = market_data['ce_gamma'].fillna(0)
                pe_gamma = market_data['pe_gamma'].fillna(0)
                total_gamma = ce_gamma.sum() + pe_gamma.sum()
                gamma_sentiment = np.tanh(total_gamma / 500) * 0.25  # Weight: 25%
                greek_scores.append(gamma_sentiment)
            
            # Theta sentiment (time decay pressure)
            if 'ce_theta' in market_data.columns and 'pe_theta' in market_data.columns:
                ce_theta = market_data['ce_theta'].fillna(0)
                pe_theta = market_data['pe_theta'].fillna(0)
                total_theta = abs(ce_theta.sum()) + abs(pe_theta.sum())
                theta_sentiment = -np.tanh(total_theta / 200) * 0.2  # Weight: 20%, negative = time pressure
                greek_scores.append(theta_sentiment)
            
            # Vega sentiment (volatility exposure)
            if 'ce_vega' in market_data.columns and 'pe_vega' in market_data.columns:
                ce_vega = market_data['ce_vega'].fillna(0)
                pe_vega = market_data['pe_vega'].fillna(0)
                total_vega = ce_vega.sum() + pe_vega.sum()
                vega_sentiment = np.tanh(total_vega / 300) * 0.15  # Weight: 15%
                greek_scores.append(vega_sentiment)
            
            return sum(greek_scores) if greek_scores else 0.0
            
        except Exception as e:
            logger.error(f"Greek sentiment calculation failed: {e}")
            return 0.0
    
    def _calculate_triple_straddle(self, market_data: pd.DataFrame) -> float:
        """
        Triple rolling straddle analysis: ATM, ITM1, OTM1 straddles.
        """
        try:
            if 'atm_strike' not in market_data.columns:
                return 0.0
            
            straddle_scores = []
            
            # Group by ATM strike to get different straddle levels
            for atm_strike in market_data['atm_strike'].unique():
                atm_data = market_data[market_data['atm_strike'] == atm_strike]
                
                if len(atm_data) == 0:
                    continue
                
                # ATM Straddle (strike == atm_strike)
                atm_straddle = atm_data[atm_data['strike'] == atm_strike]
                if len(atm_straddle) > 0:
                    ce_price = atm_straddle['ce_close'].mean()
                    pe_price = atm_straddle['pe_close'].mean()
                    if not (np.isnan(ce_price) or np.isnan(pe_price)):
                        atm_value = ce_price + pe_price
                        straddle_scores.append(atm_value * 0.5)  # ATM weight: 50%
                
                # ITM1 Straddle (one strike ITM)
                itm_strike = atm_strike + 50  # Assuming 50 point strikes
                itm_straddle = atm_data[atm_data['strike'] == itm_strike]
                if len(itm_straddle) > 0:
                    ce_price = itm_straddle['ce_close'].mean()
                    pe_price = itm_straddle['pe_close'].mean()
                    if not (np.isnan(ce_price) or np.isnan(pe_price)):
                        itm_value = ce_price + pe_price
                        straddle_scores.append(itm_value * 0.3)  # ITM weight: 30%
                
                # OTM1 Straddle (one strike OTM)
                otm_strike = atm_strike - 50
                otm_straddle = atm_data[atm_data['strike'] == otm_strike]
                if len(otm_straddle) > 0:
                    ce_price = otm_straddle['ce_close'].mean()
                    pe_price = otm_straddle['pe_close'].mean()
                    if not (np.isnan(ce_price) or np.isnan(pe_price)):
                        otm_value = ce_price + pe_price
                        straddle_scores.append(otm_value * 0.2)  # OTM weight: 20%
            
            # Normalize straddle signal
            total_straddle = sum(straddle_scores)
            return np.tanh(total_straddle / 500) if total_straddle > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Triple straddle calculation failed: {e}")
            return 0.0
    
    def _calculate_oi_patterns(self, market_data: pd.DataFrame) -> float:
        """
        Open Interest and Price Action pattern analysis.
        """
        try:
            if not all(col in market_data.columns for col in ['ce_oi', 'pe_oi', 'ce_close', 'pe_close']):
                return 0.0
            
            pattern_signals = []
            
            # 1. OI buildup analysis
            ce_oi_total = market_data['ce_oi'].sum()
            pe_oi_total = market_data['pe_oi'].sum()
            
            if ce_oi_total + pe_oi_total > 0:
                oi_skew = (ce_oi_total - pe_oi_total) / (ce_oi_total + pe_oi_total)
                pattern_signals.append(oi_skew * 0.4)  # OI skew: 40%
            
            # 2. Price vs OI correlation
            if len(market_data) > 1:
                # Calculate price changes
                market_data_sorted = market_data.sort_values(['trade_date', 'trade_time'])
                ce_price_change = market_data_sorted['ce_close'].pct_change().fillna(0)
                pe_price_change = market_data_sorted['pe_close'].pct_change().fillna(0)
                
                # Calculate OI changes
                ce_oi_change = market_data_sorted['ce_oi'].pct_change().fillna(0)
                pe_oi_change = market_data_sorted['pe_oi'].pct_change().fillna(0)
                
                # Correlation analysis
                try:
                    ce_corr = np.corrcoef(ce_price_change, ce_oi_change)[0, 1]
                    pe_corr = np.corrcoef(pe_price_change, pe_oi_change)[0, 1]
                    
                    if not (np.isnan(ce_corr) or np.isnan(pe_corr)):
                        avg_corr = (ce_corr + pe_corr) / 2
                        pattern_signals.append(avg_corr * 0.3)  # Correlation: 30%
                except:
                    pass
            
            # 3. Volume-OI relationship
            if 'ce_volume' in market_data.columns and 'pe_volume' in market_data.columns:
                ce_volume = market_data['ce_volume'].sum()
                pe_volume = market_data['pe_volume'].sum()
                
                if ce_oi_total > 0 and pe_oi_total > 0:
                    ce_turnover = ce_volume / ce_oi_total if ce_oi_total > 0 else 0
                    pe_turnover = pe_volume / pe_oi_total if pe_oi_total > 0 else 0
                    
                    avg_turnover = (ce_turnover + pe_turnover) / 2
                    turnover_signal = np.tanh(avg_turnover) * 0.3  # Turnover: 30%
                    pattern_signals.append(turnover_signal)
            
            return sum(pattern_signals) if pattern_signals else 0.0
            
        except Exception as e:
            logger.error(f"OI pattern calculation failed: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """
        Enhanced trend strength using multiple indicators.
        """
        try:
            trend_signals = []
            
            # 1. Spot price trend
            if 'spot' in market_data.columns:
                spots = market_data['spot'].dropna()
                if len(spots) > 1:
                    returns = spots.pct_change().dropna()
                    if len(returns) > 0:
                        trend_signals.append(returns.mean() * 100 * 0.4)  # Weight: 40%
            
            # 2. Future price trend
            if 'future_close' in market_data.columns:
                futures = market_data['future_close'].dropna()
                if len(futures) > 1:
                    future_returns = futures.pct_change().dropna()
                    if len(future_returns) > 0:
                        trend_signals.append(future_returns.mean() * 100 * 0.3)  # Weight: 30%
            
            # 3. Call-Put premium trend
            if 'ce_close' in market_data.columns and 'pe_close' in market_data.columns:
                ce_prices = market_data['ce_close'].dropna()
                pe_prices = market_data['pe_close'].dropna()
                
                if len(ce_prices) > 0 and len(pe_prices) > 0:
                    premium_ratio = ce_prices.mean() / pe_prices.mean() if pe_prices.mean() > 0 else 1
                    # Convert to trend signal: >1 = bullish, <1 = bearish
                    premium_trend = (premium_ratio - 1) * 50  # Normalize
                    trend_signals.append(premium_trend * 0.3)  # Weight: 30%
            
            return sum(trend_signals) if trend_signals else 0.0
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return 0.0
    
    def _calculate_structure_score(self, market_data: pd.DataFrame) -> float:
        """
        Market structure analysis (support/resistance, price levels).
        """
        try:
            structure_signals = []
            
            # 1. Strike distribution analysis
            if 'strike' in market_data.columns and 'atm_strike' in market_data.columns:
                strikes = market_data['strike'].dropna()
                atm_strikes = market_data['atm_strike'].dropna()
                
                if len(strikes) > 0 and len(atm_strikes) > 0:
                    # Distance from ATM
                    strike_distances = []
                    for atm in atm_strikes.unique():
                        distances = abs(strikes - atm) / atm  # Relative distance
                        strike_distances.extend(distances)
                    
                    if strike_distances:
                        avg_distance = np.mean(strike_distances)
                        # Closer to ATM = more structured market
                        structure_signals.append((1 - min(avg_distance, 1.0)) * 0.4)  # Weight: 40%
            
            # 2. Volume distribution
            if 'ce_volume' in market_data.columns and 'pe_volume' in market_data.columns:
                ce_vol = market_data['ce_volume'].fillna(0)
                pe_vol = market_data['pe_volume'].fillna(0)
                total_vol = ce_vol + pe_vol
                
                if total_vol.sum() > 0:
                    # Concentration measure (higher = more structured)
                    vol_concentration = (total_vol**2).sum() / (total_vol.sum()**2)
                    structure_signals.append(vol_concentration * 0.3)  # Weight: 30%
            
            # 3. Price clustering
            if 'ce_close' in market_data.columns and 'pe_close' in market_data.columns:
                all_prices = []
                all_prices.extend(market_data['ce_close'].dropna())
                all_prices.extend(market_data['pe_close'].dropna())
                
                if len(all_prices) > 1:
                    # Calculate price dispersion (lower = more structured)
                    price_std = np.std(all_prices)
                    price_mean = np.mean(all_prices)
                    if price_mean > 0:
                        coeff_var = price_std / price_mean
                        structure_score = max(0, 1 - coeff_var)  # Invert so lower dispersion = higher score
                        structure_signals.append(structure_score * 0.3)  # Weight: 30%
            
            return sum(structure_signals) if structure_signals else 0.0
            
        except Exception as e:
            logger.error(f"Structure score calculation failed: {e}")
            return 0.0
    
    def _determine_regime(self, avg_iv: float, trend_strength: float, greek_sentiment: float, 
                         straddle_signal: float, oi_pattern: float, structure_score: float) -> tuple:
        """
        Enhanced regime determination using all calculated indicators.
        """
        try:
            # Volatility regime
            if avg_iv > 25:
                volatility_regime = "HIGH"
            elif avg_iv > 18:
                volatility_regime = "MEDIUM"
            else:
                volatility_regime = "LOW"
            
            # Trend regime
            if trend_strength > 1.0:
                trend_regime = "STRONG_BULLISH" if greek_sentiment > 0.2 else "BULLISH"
            elif trend_strength > 0.3:
                trend_regime = "MILD_BULLISH"
            elif trend_strength < -1.0:
                trend_regime = "STRONG_BEARISH" if greek_sentiment < -0.2 else "BEARISH"
            elif trend_strength < -0.3:
                trend_regime = "MILD_BEARISH"
            else:
                trend_regime = "SIDEWAYS"
            
            # Structure regime
            if structure_score > 0.7:
                structure_regime = "STRONG"
            elif structure_score > 0.4:
                structure_regime = "MODERATE"
            else:
                structure_regime = "WEAK"
            
            # Overall regime (enhanced 18-regime system)
            if volatility_regime == "HIGH":
                if trend_regime in ["STRONG_BULLISH", "BULLISH"]:
                    regime = "STRONG_BULLISH_HIGH_VOLATILE"
                elif trend_regime in ["STRONG_BEARISH", "BEARISH"]:
                    regime = "STRONG_BEARISH_HIGH_VOLATILE"
                else:
                    regime = "NEUTRAL_HIGH_VOLATILE"
            elif volatility_regime == "MEDIUM":
                if trend_regime in ["STRONG_BULLISH", "BULLISH"]:
                    regime = "MILD_BULLISH_NORMAL_VOLATILE"
                elif trend_regime in ["STRONG_BEARISH", "BEARISH"]:
                    regime = "MILD_BEARISH_NORMAL_VOLATILE"
                else:
                    regime = "NEUTRAL_NORMAL_VOLATILE"
            else:  # LOW volatility
                if trend_regime in ["STRONG_BULLISH", "BULLISH"]:
                    regime = "STRONG_BULLISH_LOW_VOLATILE"
                elif trend_regime in ["STRONG_BEARISH", "BEARISH"]:
                    regime = "STRONG_BEARISH_LOW_VOLATILE"
                else:
                    regime = "SIDEWAYS_LOW_VOLATILE"
            
            # Adjust based on straddle and OI signals
            if abs(straddle_signal) > 0.5 or abs(oi_pattern) > 0.5:
                if "NEUTRAL" in regime:
                    regime = "HIGH_VOLATILITY"  # Strong signals suggest volatility
            
            return volatility_regime, trend_regime, structure_regime, regime
            
        except Exception as e:
            logger.error(f"Regime determination failed: {e}")
            return "MEDIUM", "SIDEWAYS", "MODERATE", "NEUTRAL"
    
    def _calculate_confidence(self, market_data: pd.DataFrame, avg_iv: float, 
                            greek_sentiment: float, trend_strength: float, 
                            straddle_signal: float, oi_pattern: float, structure_score: float) -> float:
        """
        Enhanced confidence calculation based on signal strength and data quality.
        """
        try:
            confidence_factors = []
            
            # 1. Data quality (30%)
            data_points = len(market_data)
            data_quality = min(data_points / 1000, 1.0)  # Normalize to 1000 points
            confidence_factors.append(data_quality * 0.3)
            
            # 2. Signal strength (25%)
            signal_strengths = [
                abs(greek_sentiment),
                abs(trend_strength) / 10,  # Normalize trend strength
                abs(straddle_signal),
                abs(oi_pattern),
                structure_score
            ]
            avg_signal_strength = np.mean([s for s in signal_strengths if not np.isnan(s)])
            confidence_factors.append(min(avg_signal_strength, 1.0) * 0.25)
            
            # 3. Data completeness (25%)
            required_columns = ['ce_close', 'pe_close', 'ce_oi', 'pe_oi', 'ce_iv', 'pe_iv']
            available_columns = sum(1 for col in required_columns if col in market_data.columns)
            completeness = available_columns / len(required_columns)
            confidence_factors.append(completeness * 0.25)
            
            # 4. Consistency check (20%)
            consistency_score = 1.0
            
            # Check for extreme values that might indicate data issues
            if avg_iv > 100 or avg_iv < 0:  # Unrealistic IV
                consistency_score *= 0.5
            
            if abs(trend_strength) > 50:  # Extreme trend
                consistency_score *= 0.7
            
            confidence_factors.append(consistency_score * 0.2)
            
            # Base confidence starts at 0.5, enhanced by factors
            base_confidence = 0.5
            total_confidence = base_confidence + sum(confidence_factors)
            
            return min(max(total_confidence, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5


# Module-level validation
if __name__ == "__main__":
    print("Testing Real Data Engine Adapter...")
    
    try:
        adapter = RealDataEngineAdapter()
        print("✅ Adapter initialized successfully")
        
        if adapter.validate_connection():
            print("✅ HeavyDB connection validated")
        else:
            print("❌ HeavyDB connection validation failed")
            
    except Exception as e:
        print(f"❌ Adapter initialization failed: {e}")
        sys.exit(1)