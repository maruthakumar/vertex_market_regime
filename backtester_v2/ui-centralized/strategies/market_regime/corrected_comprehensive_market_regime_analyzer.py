#!/usr/bin/env python3
"""
CORRECTED Comprehensive Market Regime Analyzer
Enhanced Triple Straddle Rolling Analysis Framework with Corrected Greek Sentiment

Author: The Augster
Date: 2025-06-20
Version: 6.1.0 (CORRECTED - Greek Sentiment & Regime Classification)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import talib  # Commented out for now - not required for core functionality

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectedMarketRegimeEngine:
    """
    CORRECTED Comprehensive Market Regime Analyzer for 0 DTE Options
    
    FIXES APPLIED:
    1. Corrected Greek Sentiment Analysis - Portfolio-level cumulative exposure
    2. Proper 9:15 AM baseline establishment
    3. Deterministic 18-regime classification system
    4. Mathematical transparency with exact formulas
    """
    
    def __init__(self, config_file: str = None):
        """Initialize corrected market regime analyzer"""
        
        self.config = self._load_configuration(config_file)
        self.opening_greek_exposure = {}
        self.regime_classification_system = self._initialize_regime_system()
        
        logger.info("🚀 CORRECTED Comprehensive Market Regime Analyzer initialized")
        logger.info("✅ Greek Sentiment Analysis: Portfolio-level cumulative exposure")
        logger.info("✅ Regime Classification: Deterministic 18-regime system")
        logger.info("✅ Mathematical Framework: Complete transparency")
    
    def _load_configuration(self, config_file: str) -> Dict[str, Any]:
        """Load configuration with corrected defaults"""
        
        default_config = {
            # Analysis parameters
            'dte_focus': 0,
            'underlying': 'NIFTY',
            'option_multiplier': 50,  # NIFTY lot size
            'strike_range': 7,  # ±7 strikes around ATM
            
            # Greek sentiment configuration (CORRECTED)
            'greek_normalization_factors': {
                'delta': 100000,  # Normalize delta exposure
                'gamma': 50000,   # Normalize gamma exposure
                'theta': 10000,   # Normalize theta exposure
                'vega': 20000     # Normalize vega exposure
            },
            'greek_component_weights': {
                'delta': 0.40,    # 40% weight - directional bias
                'gamma': 0.30,    # 30% weight - acceleration risk
                'theta': 0.20,    # 20% weight - time decay (critical for 0 DTE)
                'vega': 0.10      # 10% weight - volatility sensitivity
            },
            
            # Component weights for final regime score
            'component_weights': {
                'straddle': 0.40,    # 40% - Enhanced Triple Straddle (Primary)
                'greek': 0.30,       # 30% - Greek Sentiment (Modifier)
                'oi': 0.20,          # 20% - OI Analysis (Confirmation)
                'technical': 0.10    # 10% - Technical Indicators (Supporting)
            },
            
            # Straddle component weights
            'straddle_weights': {
                'atm': 0.35,         # 35% - ATM straddle
                'itm1': 0.20,        # 20% - ITM1 straddle
                'otm1': 0.15,        # 15% - OTM1 straddle
                'combined': 0.15,    # 15% - Combined straddle
                'atmce': 0.08,       # 8% - ATM Call individual
                'atmpe': 0.07        # 7% - ATM Put individual
            },
            
            # Timeframe weights (optimized for 0 DTE)
            'timeframe_weights': {
                '3min': 0.40,        # 40% weight - rapid response
                '5min': 0.30,        # 30% weight - balanced view
                '10min': 0.20,       # 20% weight - trend confirmation
                '15min': 0.10        # 10% weight - stability check
            },
            
            # Performance targets
            'target_processing_time': 3.0,
            'target_accuracy': 0.85,
            'confidence_threshold': 0.70
        }
        
        return default_config
    
    def _initialize_regime_system(self) -> Dict[str, Any]:
        """Initialize the 18-regime classification system"""
        
        regime_system = {
            'regime_mapping': {
                # Bullish Regimes (1-6)
                ('Bullish', 'Strong', 'High'): (1, 'High Volatile Strong Bullish'),
                ('Bullish', 'Strong', 'Normal'): (2, 'Normal Volatile Strong Bullish'),
                ('Bullish', 'Strong', 'Low'): (3, 'Low Volatile Strong Bullish'),
                ('Bullish', 'Mild', 'High'): (4, 'High Volatile Mild Bullish'),
                ('Bullish', 'Mild', 'Normal'): (5, 'Normal Volatile Mild Bullish'),
                ('Bullish', 'Mild', 'Low'): (6, 'Low Volatile Mild Bullish'),
                
                # Neutral Regimes (7-12)
                ('Neutral', 'Neutral', 'High'): (7, 'High Volatile Neutral'),
                ('Neutral', 'Neutral', 'Normal'): (8, 'Normal Volatile Neutral'),
                ('Neutral', 'Neutral', 'Low'): (9, 'Low Volatile Neutral'),
                ('Neutral', 'Sideways', 'High'): (10, 'High Volatile Sideways'),
                ('Neutral', 'Sideways', 'Normal'): (11, 'Normal Volatile Sideways'),
                ('Neutral', 'Sideways', 'Low'): (12, 'Low Volatile Sideways'),
                
                # Bearish Regimes (13-18)
                ('Bearish', 'Mild', 'High'): (13, 'High Volatile Mild Bearish'),
                ('Bearish', 'Mild', 'Normal'): (14, 'Normal Volatile Mild Bearish'),
                ('Bearish', 'Mild', 'Low'): (15, 'Low Volatile Mild Bearish'),
                ('Bearish', 'Strong', 'High'): (16, 'High Volatile Strong Bearish'),
                ('Bearish', 'Strong', 'Normal'): (17, 'Normal Volatile Strong Bearish'),
                ('Bearish', 'Strong', 'Low'): (18, 'Low Volatile Strong Bearish')
            },
            'score_thresholds': {
                'strong_bullish': 0.5,
                'mild_bullish': 0.2,
                'neutral_upper': 0.2,
                'neutral_lower': -0.2,
                'mild_bearish': -0.2,
                'strong_bearish': -0.5
            },
            'volatility_thresholds': {
                'high': 0.6,
                'normal': 0.3
            }
        }
        
        return regime_system
    
    def establish_opening_baseline(self, opening_data: pd.DataFrame) -> Dict[str, float]:
        """
        CORRECTED: Establish 9:15 AM opening baseline for Greek sentiment analysis
        
        Args:
            opening_data: DataFrame with 9:15 AM options data
            
        Returns:
            Dict with opening Greek exposure values
        """
        
        logger.info("📊 Establishing 9:15 AM opening baseline for Greek sentiment...")
        
        # Filter for ATM ±7 strikes (15 strikes total)
        atm_strike = round(opening_data['underlying_price'].iloc[0] / 50) * 50
        strike_range = range(atm_strike - 350, atm_strike + 400, 50)
        
        baseline_data = opening_data[opening_data['strike_price'].isin(strike_range)].copy()
        
        # Calculate volume weights
        max_volume = baseline_data['volume'].max()
        baseline_data['volume_weight'] = baseline_data['volume'] / max(max_volume, 1)
        
        # Calculate opening Greek exposure (portfolio-level)
        opening_exposure = {
            'net_delta': 0,
            'net_gamma': 0,
            'net_theta': 0,
            'net_vega': 0
        }
        
        for _, row in baseline_data.iterrows():
            # Volume-weighted Greek exposure
            volume_weight = row['volume_weight']
            option_multiplier = self.config['option_multiplier']
            
            # Delta exposure (Calls positive, Puts negative)
            delta_exposure = row['delta'] * row['open_interest'] * volume_weight * option_multiplier
            opening_exposure['net_delta'] += delta_exposure
            
            # Gamma exposure (always positive)
            gamma_exposure = abs(row['gamma']) * row['open_interest'] * volume_weight * option_multiplier
            opening_exposure['net_gamma'] += gamma_exposure
            
            # Theta exposure (always negative for long positions)
            theta_exposure = row['theta'] * row['open_interest'] * volume_weight * option_multiplier
            opening_exposure['net_theta'] += theta_exposure
            
            # Vega exposure (always positive)
            vega_exposure = abs(row['vega']) * row['open_interest'] * volume_weight * option_multiplier
            opening_exposure['net_vega'] += vega_exposure
        
        self.opening_greek_exposure = opening_exposure
        
        logger.info(f"   ✅ Opening Net Delta: {opening_exposure['net_delta']:,.0f}")
        logger.info(f"   ✅ Opening Net Gamma: {opening_exposure['net_gamma']:,.0f}")
        logger.info(f"   ✅ Opening Net Theta: {opening_exposure['net_theta']:,.0f}")
        logger.info(f"   ✅ Opening Net Vega: {opening_exposure['net_vega']:,.0f}")
        
        return opening_exposure
    
    def _calculate_corrected_greek_sentiment_analysis(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        CORRECTED: Calculate Greek sentiment analysis with portfolio-level cumulative exposure
        
        Args:
            current_data: DataFrame with current minute's options data
            
        Returns:
            Dict with Greek sentiment analysis results
        """
        
        # Filter for ATM ±7 strikes (15 strikes total)
        atm_strike = round(current_data['underlying_price'].iloc[0] / 50) * 50
        strike_range = range(atm_strike - 350, atm_strike + 400, 50)
        
        current_strikes = current_data[current_data['strike_price'].isin(strike_range)].copy()
        
        # Calculate volume weights
        max_volume = current_strikes['volume'].max()
        current_strikes['volume_weight'] = current_strikes['volume'] / max(max_volume, 1)
        
        # Calculate current Greek exposure (portfolio-level)
        current_exposure = {
            'net_delta': 0,
            'net_gamma': 0,
            'net_theta': 0,
            'net_vega': 0
        }
        
        for _, row in current_strikes.iterrows():
            # Volume-weighted Greek exposure
            volume_weight = row['volume_weight']
            option_multiplier = self.config['option_multiplier']
            
            # Delta exposure (Calls positive, Puts negative)
            delta_exposure = row['delta'] * row['open_interest'] * volume_weight * option_multiplier
            current_exposure['net_delta'] += delta_exposure
            
            # Gamma exposure (always positive)
            gamma_exposure = abs(row['gamma']) * row['open_interest'] * volume_weight * option_multiplier
            current_exposure['net_gamma'] += gamma_exposure
            
            # Theta exposure (always negative for long positions)
            theta_exposure = row['theta'] * row['open_interest'] * volume_weight * option_multiplier
            current_exposure['net_theta'] += theta_exposure
            
            # Vega exposure (always positive)
            vega_exposure = abs(row['vega']) * row['open_interest'] * volume_weight * option_multiplier
            current_exposure['net_vega'] += vega_exposure
        
        # Calculate Greek changes from opening baseline
        greek_changes = {
            'delta_change': current_exposure['net_delta'] - self.opening_greek_exposure['net_delta'],
            'gamma_change': current_exposure['net_gamma'] - self.opening_greek_exposure['net_gamma'],
            'theta_change': current_exposure['net_theta'] - self.opening_greek_exposure['net_theta'],
            'vega_change': current_exposure['net_vega'] - self.opening_greek_exposure['net_vega']
        }
        
        # Normalize Greek components using hyperbolic tangent
        normalization = self.config['greek_normalization_factors']
        
        greek_components = {
            'delta_component': np.tanh(greek_changes['delta_change'] / normalization['delta']),
            'gamma_component': np.tanh(greek_changes['gamma_change'] / normalization['gamma']),
            'theta_component': np.tanh(greek_changes['theta_change'] / normalization['theta']),
            'vega_component': np.tanh(greek_changes['vega_change'] / normalization['vega'])
        }
        
        # Calculate weighted Greek sentiment score
        weights = self.config['greek_component_weights']
        greek_sentiment_score = (
            weights['delta'] * greek_components['delta_component'] +
            weights['gamma'] * greek_components['gamma_component'] +
            weights['theta'] * greek_components['theta_component'] +
            weights['vega'] * greek_components['vega_component']
        )
        
        # Combine all results
        greek_analysis = {
            # Opening baseline
            'opening_net_delta': self.opening_greek_exposure['net_delta'],
            'opening_net_gamma': self.opening_greek_exposure['net_gamma'],
            'opening_net_theta': self.opening_greek_exposure['net_theta'],
            'opening_net_vega': self.opening_greek_exposure['net_vega'],
            
            # Current exposure
            'current_net_delta': current_exposure['net_delta'],
            'current_net_gamma': current_exposure['net_gamma'],
            'current_net_theta': current_exposure['net_theta'],
            'current_net_vega': current_exposure['net_vega'],
            
            # Changes from opening
            'delta_change': greek_changes['delta_change'],
            'gamma_change': greek_changes['gamma_change'],
            'theta_change': greek_changes['theta_change'],
            'vega_change': greek_changes['vega_change'],
            
            # Normalized components
            'delta_component': greek_components['delta_component'],
            'gamma_component': greek_components['gamma_component'],
            'theta_component': greek_components['theta_component'],
            'vega_component': greek_components['vega_component'],
            
            # Final Greek sentiment score
            'greek_sentiment_score': greek_sentiment_score
        }
        
        return greek_analysis
    
    def _calculate_enhanced_triple_straddle_analysis(self, current_data: pd.DataFrame, 
                                                   opening_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Enhanced Triple Straddle Analysis (40% weight)
        
        Args:
            current_data: Current minute's options data
            opening_data: Opening (9:15 AM) options data
            
        Returns:
            Dict with straddle analysis results
        """
        
        # Get ATM strike
        atm_strike = round(current_data['underlying_price'].iloc[0] / 50) * 50
        
        # Define strikes
        strikes = {
            'atm': atm_strike,
            'itm1': atm_strike - 50,
            'otm1': atm_strike + 50
        }
        
        # Calculate current straddle values
        current_straddles = {}
        opening_straddles = {}
        
        for straddle_type, strike in strikes.items():
            # Current values
            current_call = current_data[(current_data['strike_price'] == strike) & 
                                      (current_data['option_type'] == 'CE')]
            current_put = current_data[(current_data['strike_price'] == strike) & 
                                     (current_data['option_type'] == 'PE')]
            
            if not current_call.empty and not current_put.empty:
                current_straddles[straddle_type] = {
                    'call_premium': current_call['premium'].iloc[0],
                    'put_premium': current_put['premium'].iloc[0],
                    'straddle_value': current_call['premium'].iloc[0] + current_put['premium'].iloc[0]
                }
            
            # Opening values
            opening_call = opening_data[(opening_data['strike_price'] == strike) & 
                                      (opening_data['option_type'] == 'CE')]
            opening_put = opening_data[(opening_data['strike_price'] == strike) & 
                                     (opening_data['option_type'] == 'PE')]
            
            if not opening_call.empty and not opening_put.empty:
                opening_straddles[straddle_type] = {
                    'call_premium': opening_call['premium'].iloc[0],
                    'put_premium': opening_put['premium'].iloc[0],
                    'straddle_value': opening_call['premium'].iloc[0] + opening_put['premium'].iloc[0]
                }
        
        # Calculate straddle changes and percentage changes
        straddle_analysis = {}
        
        for straddle_type in ['atm', 'itm1', 'otm1']:
            if straddle_type in current_straddles and straddle_type in opening_straddles:
                current_value = current_straddles[straddle_type]['straddle_value']
                opening_value = opening_straddles[straddle_type]['straddle_value']
                
                change = current_value - opening_value
                pct_change = change / opening_value if opening_value != 0 else 0
                
                straddle_analysis[f'{straddle_type}_straddle_value'] = current_value
                straddle_analysis[f'{straddle_type}_straddle_change'] = change
                straddle_analysis[f'{straddle_type}_straddle_pct_change'] = pct_change
        
        # Calculate combined straddle
        if all(f'{st}_straddle_value' in straddle_analysis for st in ['atm', 'itm1', 'otm1']):
            combined_current = sum(straddle_analysis[f'{st}_straddle_value'] for st in ['atm', 'itm1', 'otm1'])
            combined_opening = sum(opening_straddles[st]['straddle_value'] for st in ['atm', 'itm1', 'otm1'])
            
            straddle_analysis['combined_straddle_value'] = combined_current
            straddle_analysis['combined_straddle_change'] = combined_current - combined_opening
            straddle_analysis['combined_straddle_pct_change'] = (combined_current - combined_opening) / combined_opening
        
        # Calculate individual ATMCE and ATMPE
        if 'atm' in current_straddles and 'atm' in opening_straddles:
            # ATMCE (ATM Call)
            atmce_current = current_straddles['atm']['call_premium']
            atmce_opening = opening_straddles['atm']['call_premium']
            straddle_analysis['atmce_value'] = atmce_current
            straddle_analysis['atmce_change'] = atmce_current - atmce_opening
            straddle_analysis['atmce_pct_change'] = (atmce_current - atmce_opening) / atmce_opening
            
            # ATMPE (ATM Put)
            atmpe_current = current_straddles['atm']['put_premium']
            atmpe_opening = opening_straddles['atm']['put_premium']
            straddle_analysis['atmpe_value'] = atmpe_current
            straddle_analysis['atmpe_change'] = atmpe_current - atmpe_opening
            straddle_analysis['atmpe_pct_change'] = (atmpe_current - atmpe_opening) / atmpe_opening
        
        # Normalize straddle components and calculate weighted signal
        weights = self.config['straddle_weights']
        
        normalized_components = {}
        for component in ['atm', 'itm1', 'otm1', 'combined', 'atmce', 'atmpe']:
            pct_change_key = f'{component}_straddle_pct_change' if component != 'atmce' and component != 'atmpe' else f'{component}_pct_change'
            if pct_change_key in straddle_analysis:
                # Normalize using tanh with scaling factor
                normalized_components[component] = np.tanh(straddle_analysis[pct_change_key] * 10)
        
        # Calculate weighted straddle signal score
        straddle_signal_score = sum(
            weights[component] * normalized_components.get(component, 0)
            for component in weights.keys()
        )
        
        straddle_analysis['straddle_signal_score'] = straddle_signal_score
        
        return straddle_analysis

    def _classify_market_regimes(self, straddle_analysis: Dict[str, float],
                               greek_analysis: Dict[str, float],
                               oi_analysis: Dict[str, float],
                               technical_analysis: Dict[str, float]) -> Dict[str, Any]:
        """
        CORRECTED: Classify market regime using deterministic 18-regime system

        Args:
            straddle_analysis: Enhanced Triple Straddle analysis results
            greek_analysis: Corrected Greek sentiment analysis results
            oi_analysis: OI with price action analysis results
            technical_analysis: Technical indicators analysis results

        Returns:
            Dict with regime classification results
        """

        # Calculate component scores with proper weights
        component_weights = self.config['component_weights']

        straddle_component_score = component_weights['straddle'] * straddle_analysis.get('straddle_signal_score', 0)
        greek_component_score = component_weights['greek'] * greek_analysis.get('greek_sentiment_score', 0)
        oi_component_score = component_weights['oi'] * oi_analysis.get('oi_signal_score', 0)
        technical_component_score = component_weights['technical'] * technical_analysis.get('technical_signal_score', 0)

        # Calculate final regime score
        final_regime_score = (
            straddle_component_score +
            greek_component_score +
            oi_component_score +
            technical_component_score
        )

        # Determine volatility level
        volatility_level = self._determine_volatility_level(technical_analysis)

        # Classify regime using deterministic logic
        regime_id, regime_name = self._map_score_to_regime(final_regime_score, volatility_level)

        # Calculate confidence score
        confidence_score = self._calculate_regime_confidence(
            straddle_analysis, greek_analysis, oi_analysis, technical_analysis
        )

        regime_classification = {
            'straddle_component_score': straddle_component_score,
            'greek_component_score': greek_component_score,
            'oi_component_score': oi_component_score,
            'technical_component_score': technical_component_score,
            'final_regime_score': final_regime_score,
            'volatility_level': volatility_level,
            'regime_id': regime_id,
            'regime_name': regime_name,
            'regime_confidence': confidence_score
        }

        return regime_classification

    def _determine_volatility_level(self, technical_analysis: Dict[str, float]) -> str:
        """Determine volatility level from technical analysis"""

        # Use technical volatility score as proxy
        volatility_score = technical_analysis.get('technical_volatility_score', 0)

        thresholds = self.regime_classification_system['volatility_thresholds']

        if volatility_score >= thresholds['high']:
            return 'High'
        elif volatility_score >= thresholds['normal']:
            return 'Normal'
        else:
            return 'Low'

    def _map_score_to_regime(self, final_score: float, volatility_level: str) -> Tuple[int, str]:
        """
        Map final regime score to regime ID and name using deterministic logic

        Args:
            final_score: Final regime score [-1.0, +1.0]
            volatility_level: 'High', 'Normal', or 'Low'

        Returns:
            Tuple of (regime_id, regime_name)
        """

        thresholds = self.regime_classification_system['score_thresholds']

        # Determine direction and strength
        if final_score >= thresholds['mild_bullish']:
            direction = 'Bullish'
            if final_score >= thresholds['strong_bullish']:
                strength = 'Strong'
            else:
                strength = 'Mild'
        elif final_score <= thresholds['mild_bearish']:
            direction = 'Bearish'
            if final_score <= thresholds['strong_bearish']:
                strength = 'Strong'
            else:
                strength = 'Mild'
        else:
            direction = 'Neutral'
            strength = 'Sideways' if abs(final_score) < 0.1 else 'Neutral'

        # Map to regime using the classification system
        regime_key = (direction, strength, volatility_level)
        regime_mapping = self.regime_classification_system['regime_mapping']

        return regime_mapping.get(regime_key, (8, 'Normal Volatile Neutral'))

    def _calculate_regime_confidence(self, straddle_analysis: Dict[str, float],
                                   greek_analysis: Dict[str, float],
                                   oi_analysis: Dict[str, float],
                                   technical_analysis: Dict[str, float]) -> float:
        """Calculate regime classification confidence score"""

        # Component confidence scores (based on signal strength)
        straddle_confidence = min(abs(straddle_analysis.get('straddle_signal_score', 0)), 1.0)
        greek_confidence = min(abs(greek_analysis.get('greek_sentiment_score', 0)), 1.0)
        oi_confidence = min(abs(oi_analysis.get('oi_signal_score', 0)), 1.0)
        technical_confidence = min(abs(technical_analysis.get('technical_signal_score', 0)), 1.0)

        # Agreement score (how well components agree)
        component_signals = [
            straddle_analysis.get('straddle_signal_score', 0),
            greek_analysis.get('greek_sentiment_score', 0),
            oi_analysis.get('oi_signal_score', 0),
            technical_analysis.get('technical_signal_score', 0)
        ]

        signal_std = np.std(component_signals)
        agreement_score = max(0, 1 - (signal_std / 0.5))  # Penalize high disagreement

        # Final confidence (minimum component confidence × agreement)
        final_confidence = min(
            straddle_confidence, greek_confidence, oi_confidence, technical_confidence
        ) * agreement_score

        return final_confidence

    def execute_corrected_comprehensive_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Execute CORRECTED comprehensive market regime analysis

        Args:
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)

        Returns:
            Dict with complete analysis results
        """

        logger.info("\n" + "="*80)
        logger.info("CORRECTED COMPREHENSIVE MARKET REGIME ANALYSIS")
        logger.info("Enhanced Triple Straddle Rolling Analysis Framework")
        logger.info("="*80)
        logger.info(f"🎯 Analysis Period: {start_date} to {end_date}")
        logger.info(f"🎯 DTE Focus: {self.config['dte_focus']} (Same-day expiry)")
        logger.info("✅ Greek Sentiment: Portfolio-level cumulative exposure")
        logger.info("✅ Regime Classification: Deterministic 18-regime system")

        start_time = time.time()

        # Step 1: Extract and preprocess data
        logger.info("\n📊 Step 1: Data Extraction and Preprocessing...")
        raw_data = self._extract_sample_data(start_date, end_date)

        # Step 2: Establish opening baseline (9:15 AM)
        logger.info("\n📈 Step 2: Establishing Opening Baseline...")
        opening_data = raw_data[raw_data['trade_time'].dt.time == pd.Timestamp('09:15:00').time()]
        if not opening_data.empty:
            self.establish_opening_baseline(opening_data)
        else:
            logger.warning("⚠️ No 9:15 AM data found, using first available data as baseline")
            self.establish_opening_baseline(raw_data.head(30))  # Use first 30 rows as baseline

        # Step 3: Process minute-by-minute analysis
        logger.info("\n⚙️ Step 3: Minute-by-Minute Analysis...")

        analysis_results = []
        unique_timestamps = raw_data['trade_time'].unique()

        for i, timestamp in enumerate(unique_timestamps[:50]):  # Process first 50 minutes for demo
            minute_data = raw_data[raw_data['trade_time'] == timestamp]

            if minute_data.empty:
                continue

            # Calculate all components
            straddle_analysis = self._calculate_enhanced_triple_straddle_analysis(
                minute_data, opening_data
            )

            greek_analysis = self._calculate_corrected_greek_sentiment_analysis(minute_data)

            # PRODUCTION MODE: Use real OI and technical analysis - NO SYNTHETIC DATA
            logger.error("PRODUCTION MODE: Synthetic OI and technical analysis is disabled.")
            logger.error("System should use real HeavyDB data for OI and technical analysis.")
            
            # Return empty analysis to force real data usage
            oi_analysis = {'oi_signal_score': 0.0, 'error': 'Synthetic data disabled'}
            technical_analysis = {
                'technical_signal_score': 0.0,
                'technical_volatility_score': 0.0,
                'error': 'Synthetic data disabled'
            }

            # Classify regime
            regime_classification = self._classify_market_regimes(
                straddle_analysis, greek_analysis, oi_analysis, technical_analysis
            )

            # Combine all results
            minute_result = {
                'timestamp': timestamp,
                'underlying_price': minute_data['underlying_price'].iloc[0],
                **greek_analysis,
                **straddle_analysis,
                **oi_analysis,
                **technical_analysis,
                **regime_classification
            }

            analysis_results.append(minute_result)

            if (i + 1) % 10 == 0:
                logger.info(f"   ✅ Processed {i + 1} minutes...")

        total_time = time.time() - start_time

        # Generate final results
        results_df = pd.DataFrame(analysis_results)

        logger.info(f"\n✅ Analysis Complete!")
        logger.info(f"   📊 Total minutes processed: {len(analysis_results)}")
        logger.info(f"   ⏱️ Total processing time: {total_time:.2f} seconds")
        logger.info(f"   ⚡ Average time per minute: {total_time/len(analysis_results):.3f} seconds")

        return {
            'results_dataframe': results_df,
            'total_minutes': len(analysis_results),
            'processing_time': total_time,
            'avg_time_per_minute': total_time / len(analysis_results),
            'performance_target_met': (total_time / len(analysis_results)) < self.config['target_processing_time']
        }

    def _extract_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """PRODUCTION MODE: NO SYNTHETIC SAMPLE DATA GENERATION"""
        
        logger.error("PRODUCTION MODE: Synthetic sample data generation is disabled.")
        logger.error("System must use real HeavyDB option chain data only.")
        logger.error("Cannot generate sample data - use real data extraction instead.")
        
        # Return empty DataFrame to force real data usage
        return pd.DataFrame()

def main():
    """Main execution function for corrected analysis"""

    logger.info("🚀 Starting CORRECTED Comprehensive Market Regime Analysis")

    # Initialize corrected analyzer
    analyzer = CorrectedMarketRegimeEngine()

    # Execute corrected analysis
    results = analyzer.execute_corrected_comprehensive_analysis('2024-06-20', '2024-06-20')

    logger.info(f"\n🎯 CORRECTED ANALYSIS RESULTS:")
    logger.info(f"   ✅ Performance target met: {results['performance_target_met']}")
    logger.info(f"   ✅ Average processing time: {results['avg_time_per_minute']:.3f}s per minute")

    # Display sample results
    df = results['results_dataframe']
    if not df.empty:
        logger.info(f"\n📊 Sample Results (First 5 minutes):")
        for _, row in df.head(5).iterrows():
            logger.info(f"   {row['timestamp']}: {row['regime_name']} (Confidence: {row['regime_confidence']:.3f})")

    return results

if __name__ == "__main__":
    main()
