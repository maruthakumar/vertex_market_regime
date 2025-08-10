"""
Enhanced Market Regime Formation Engine

This module provides the core engine for highly configurable market regime
formation, integrating all indicators, dynamic weightage systems, historical
performance optimization, and individual user configurations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

from .enhanced_configurable_excel_manager import EnhancedConfigurableExcelManager
from ..time_series_regime_storage import TimeSeriesRegimeStorage
from .enhanced_regime_detector import Enhanced18RegimeDetector

logger = logging.getLogger(__name__)

@dataclass
class RegimeFormationResult:
    """Result of regime formation analysis"""
    regime_id: str
    regime_name: str
    regime_type: str
    confidence_score: float
    directional_component: float
    volatility_component: float
    indicator_contributions: Dict[str, float]
    signal_strength: float
    market_condition: str
    timestamp: datetime
    metadata: Dict[str, Any]

class EnhancedRegimeFormationEngine:
    """
    Enhanced Market Regime Formation Engine
    
    This engine provides:
    - Highly configurable indicator processing
    - Dynamic weightage adjustment based on historical performance
    - Custom regime definitions and thresholds
    - Individual user regime configurations
    - Real-time regime formation and classification
    - Historical performance tracking and optimization
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 storage_path: Optional[str] = None):
        """
        Initialize Enhanced Regime Formation Engine
        
        Args:
            config_path (str, optional): Path to Excel configuration file
            storage_path (str, optional): Path to time-series storage database
        """
        # Initialize components
        self.config_manager = EnhancedConfigurableExcelManager(config_path)
        self.storage = TimeSeriesRegimeStorage(storage_path or "regime_formation.db")
        self.regime_detector = Enhanced18RegimeDetector()
        
        # Load configurations
        self.indicator_config = None
        self.weightage_config = None
        self.regime_definitions = None
        self.confidence_config = None
        self.timeframe_config = None
        self.user_profiles = None
        
        self._load_configurations()
        
        # Performance tracking
        self.performance_history = {}
        self.weight_adjustment_history = []
        
        logger.info("EnhancedRegimeFormationEngine initialized")
    
    def _load_configurations(self):
        """Load all configurations from Excel manager"""
        try:
            self.indicator_config = self.config_manager.get_indicator_configuration()
            self.weightage_config = self.config_manager.get_dynamic_weightage_configuration()
            self.regime_definitions = self.config_manager.get_regime_definitions()
            self.confidence_config = self.config_manager.get_confidence_score_configuration()
            self.timeframe_config = self.config_manager.get_timeframe_configuration()
            self.user_profiles = self.config_manager.get_user_profiles()
            
            logger.info("✅ All configurations loaded successfully")
            
            # Log configuration summary
            logger.info(f"Loaded {len(self.indicator_config)} indicators")
            logger.info(f"Loaded {len(self.regime_definitions)} regime definitions")
            logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise
    
    def form_market_regime(self, market_data: Dict[str, Any], 
                          user_id: Optional[str] = None,
                          configuration_id: Optional[str] = None) -> RegimeFormationResult:
        """
        Form market regime based on current market data and user configuration
        
        Args:
            market_data (Dict): Current market data including all indicators
            user_id (str, optional): User ID for personalized regime formation
            configuration_id (str, optional): Specific configuration to use
            
        Returns:
            RegimeFormationResult: Complete regime formation result
        """
        try:
            logger.debug(f"Forming market regime for user: {user_id}")
            
            # Get user-specific configuration if provided
            user_config = self._get_user_configuration(user_id, configuration_id)
            
            # Calculate all indicator values
            indicator_values = self._calculate_indicator_values(market_data, user_config)
            
            # Apply dynamic weightage
            weighted_indicators = self._apply_dynamic_weightage(indicator_values, user_config)
            
            # Calculate regime components
            directional_component = self._calculate_directional_component(weighted_indicators)
            volatility_component = self._calculate_volatility_component(weighted_indicators)
            
            # Determine regime based on custom definitions
            regime_result = self._determine_regime(
                directional_component, volatility_component, 
                weighted_indicators, user_config
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                regime_result, weighted_indicators, user_config
            )
            
            # Create result
            formation_result = RegimeFormationResult(
                regime_id=regime_result['regime_id'],
                regime_name=regime_result['regime_name'],
                regime_type=regime_result['regime_type'],
                confidence_score=confidence_score,
                directional_component=directional_component,
                volatility_component=volatility_component,
                indicator_contributions=weighted_indicators,
                signal_strength=regime_result['signal_strength'],
                market_condition=self._assess_market_condition(market_data),
                timestamp=datetime.now(),
                metadata={
                    'user_id': user_id,
                    'configuration_id': configuration_id,
                    'indicator_count': len(indicator_values),
                    'enabled_indicators': len([i for i in indicator_values.values() if i.get('enabled', True)]),
                    'total_weight': sum(weighted_indicators.values())
                }
            )
            
            # Store result in time-series database
            self._store_regime_result(formation_result, indicator_values)
            
            logger.debug(f"Regime formed: {formation_result.regime_name} (confidence: {confidence_score:.2f})")
            
            return formation_result
            
        except Exception as e:
            logger.error(f"Error forming market regime: {e}")
            raise
    
    def _get_user_configuration(self, user_id: Optional[str], 
                              configuration_id: Optional[str]) -> Dict[str, Any]:
        """Get user-specific configuration or default"""
        try:
            if user_id and not self.user_profiles.empty:
                user_profile = self.user_profiles[
                    (self.user_profiles['UserID'] == user_id) & 
                    (self.user_profiles['Active'] == True)
                ]
                
                if not user_profile.empty:
                    profile = user_profile.iloc[0]
                    
                    # Parse custom weights if available
                    custom_weights = {}
                    if profile['CustomWeights'] and profile['CustomWeights'] != 'EQUAL_WEIGHTS':
                        try:
                            weight_pairs = profile['CustomWeights'].split(',')
                            for pair in weight_pairs:
                                category, weight = pair.split(':')
                                custom_weights[category.strip()] = float(weight.strip())
                        except:
                            logger.warning(f"Could not parse custom weights for user {user_id}")
                    
                    # Parse custom thresholds
                    custom_thresholds = {}
                    if profile['CustomThresholds'] and profile['CustomThresholds'] != 'DEFAULT_THRESHOLDS':
                        try:
                            threshold_pairs = profile['CustomThresholds'].split(',')
                            for pair in threshold_pairs:
                                param, value = pair.split(':')
                                custom_thresholds[param.strip()] = float(value.strip())
                        except:
                            logger.warning(f"Could not parse custom thresholds for user {user_id}")
                    
                    return {
                        'user_id': user_id,
                        'profile_name': profile['ProfileName'],
                        'regime_preferences': profile['RegimePreferences'],
                        'risk_tolerance': profile['RiskTolerance'],
                        'time_horizon': profile['TimeHorizon'],
                        'custom_weights': custom_weights,
                        'excluded_indicators': profile['ExcludedIndicators'].split(',') if profile['ExcludedIndicators'] != 'NONE' else [],
                        'custom_thresholds': custom_thresholds
                    }
            
            # Return default configuration
            return {
                'user_id': 'DEFAULT',
                'profile_name': 'Default Profile',
                'regime_preferences': 'ALL_REGIMES',
                'risk_tolerance': 'MEDIUM',
                'time_horizon': 'MEDIUM_TERM',
                'custom_weights': {},
                'excluded_indicators': [],
                'custom_thresholds': {}
            }
            
        except Exception as e:
            logger.error(f"Error getting user configuration: {e}")
            return {}
    
    def _calculate_indicator_values(self, market_data: Dict[str, Any], 
                                  user_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate all indicator values from market data"""
        try:
            indicator_values = {}
            
            # Get enabled indicators (excluding user-excluded ones)
            enabled_indicators = self.indicator_config[
                (self.indicator_config['Enabled'] == True) &
                (~self.indicator_config['IndicatorID'].isin(user_config.get('excluded_indicators', [])))
            ]
            
            for _, indicator in enabled_indicators.iterrows():
                indicator_id = indicator['IndicatorID']
                
                try:
                    # Calculate indicator value based on type and method
                    value = self._calculate_single_indicator(indicator, market_data)
                    
                    indicator_values[indicator_id] = {
                        'indicator_id': indicator_id,
                        'indicator_name': indicator['IndicatorName'],
                        'category': indicator['Category'],
                        'raw_value': value,
                        'normalized_value': self._normalize_indicator_value(value, indicator),
                        'enabled': True,
                        'metadata': {
                            'calculation_method': indicator['CalculationMethod'],
                            'parameters': indicator['Parameters'],
                            'data_source': indicator['DataSource']
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Error calculating indicator {indicator_id}: {e}")
                    # Set default/neutral value
                    indicator_values[indicator_id] = {
                        'indicator_id': indicator_id,
                        'indicator_name': indicator['IndicatorName'],
                        'category': indicator['Category'],
                        'raw_value': 0.0,
                        'normalized_value': 0.5,  # Neutral
                        'enabled': False,
                        'metadata': {'error': str(e)}
                    }
            
            return indicator_values
            
        except Exception as e:
            logger.error(f"Error calculating indicator values: {e}")
            return {}
    
    def _calculate_single_indicator(self, indicator: pd.Series, 
                                  market_data: Dict[str, Any]) -> float:
        """Calculate a single indicator value"""
        try:
            indicator_id = indicator['IndicatorID']
            calculation_method = indicator['CalculationMethod']
            category = indicator['Category']
            
            # Route to appropriate calculation based on category and method
            if category == 'GREEK_SENTIMENT':
                return self._calculate_greek_sentiment_indicator(indicator, market_data)
            elif category == 'OI_ANALYSIS':
                return self._calculate_oi_analysis_indicator(indicator, market_data)
            elif category == 'PRICE_ACTION':
                return self._calculate_price_action_indicator(indicator, market_data)
            elif category == 'TECHNICAL_INDICATORS':
                return self._calculate_technical_indicator(indicator, market_data)
            elif category == 'VOLATILITY_MEASURES':
                return self._calculate_volatility_indicator(indicator, market_data)
            elif category == 'STRADDLE_ANALYSIS':
                return self._calculate_straddle_indicator(indicator, market_data)
            else:
                logger.warning(f"Unknown indicator category: {category}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating single indicator: {e}")
            return 0.0
    
    def _calculate_greek_sentiment_indicator(self, indicator: pd.Series, 
                                           market_data: Dict[str, Any]) -> float:
        """Calculate Greek sentiment indicators"""
        try:
            indicator_id = indicator['IndicatorID']
            
            if indicator_id == 'DELTA_SENTIMENT':
                # Volume-weighted delta sentiment
                options_data = market_data.get('options_data', {})
                if options_data:
                    deltas = options_data.get('delta', [])
                    volumes = options_data.get('volume', [])
                    if deltas and volumes:
                        weighted_delta = np.average(deltas, weights=volumes)
                        return weighted_delta
                return 0.0
                
            elif indicator_id == 'GAMMA_EXPOSURE':
                # Market maker gamma exposure
                options_data = market_data.get('options_data', {})
                if options_data:
                    gammas = options_data.get('gamma', [])
                    oi = options_data.get('open_interest', [])
                    if gammas and oi:
                        gamma_exposure = np.sum(np.array(gammas) * np.array(oi))
                        return gamma_exposure / 1000000  # Normalize
                return 0.0
                
            elif indicator_id == 'VEGA_SKEW':
                # Volatility skew through vega
                options_data = market_data.get('options_data', {})
                if options_data:
                    vegas = options_data.get('vega', [])
                    ivs = options_data.get('implied_volatility', [])
                    if vegas and ivs:
                        vega_weighted_iv = np.average(ivs, weights=vegas)
                        return vega_weighted_iv
                return 0.0
                
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating Greek sentiment indicator: {e}")
            return 0.0
    
    def _calculate_oi_analysis_indicator(self, indicator: pd.Series, 
                                       market_data: Dict[str, Any]) -> float:
        """Calculate OI analysis indicators"""
        try:
            indicator_id = indicator['IndicatorID']
            
            if indicator_id == 'OI_MOMENTUM':
                # Open interest momentum
                oi_data = market_data.get('oi_data', {})
                if oi_data:
                    current_oi = oi_data.get('total_oi', 0)
                    previous_oi = oi_data.get('previous_oi', current_oi)
                    if previous_oi > 0:
                        momentum = (current_oi - previous_oi) / previous_oi
                        return momentum
                return 0.0
                
            elif indicator_id == 'PCR_ANALYSIS':
                # Put-call ratio analysis
                oi_data = market_data.get('oi_data', {})
                if oi_data:
                    put_oi = oi_data.get('put_oi', 0)
                    call_oi = oi_data.get('call_oi', 0)
                    if call_oi > 0:
                        pcr = put_oi / call_oi
                        return pcr
                return 1.0  # Neutral PCR
                
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating OI analysis indicator: {e}")
            return 0.0
    
    def _calculate_price_action_indicator(self, indicator: pd.Series, 
                                        market_data: Dict[str, Any]) -> float:
        """Calculate price action indicators"""
        try:
            indicator_id = indicator['IndicatorID']
            price_data = market_data.get('price_data', {})
            
            if indicator_id == 'PRICE_MOMENTUM':
                # Price momentum calculation
                if price_data:
                    current_price = price_data.get('close', 0)
                    previous_price = price_data.get('previous_close', current_price)
                    if previous_price > 0:
                        momentum = (current_price - previous_price) / previous_price
                        return momentum
                return 0.0
                
            elif indicator_id == 'BREAKOUT_ANALYSIS':
                # Breakout detection
                if price_data:
                    current_price = price_data.get('close', 0)
                    resistance = price_data.get('resistance', current_price)
                    support = price_data.get('support', current_price)
                    
                    if current_price > resistance:
                        return 1.0  # Bullish breakout
                    elif current_price < support:
                        return -1.0  # Bearish breakout
                    else:
                        return 0.0  # No breakout
                return 0.0
                
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating price action indicator: {e}")
            return 0.0
    
    def _calculate_technical_indicator(self, indicator: pd.Series, 
                                     market_data: Dict[str, Any]) -> float:
        """Calculate technical indicators"""
        try:
            indicator_id = indicator['IndicatorID']
            technical_data = market_data.get('technical_data', {})
            
            if indicator_id == 'RSI_DIVERGENCE':
                rsi = technical_data.get('rsi', 50)
                return (rsi - 50) / 50  # Normalize around 0
                
            elif indicator_id == 'MACD_SIGNAL':
                macd = technical_data.get('macd', 0)
                macd_signal = technical_data.get('macd_signal', 0)
                return macd - macd_signal
                
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating technical indicator: {e}")
            return 0.0
    
    def _calculate_volatility_indicator(self, indicator: pd.Series, 
                                      market_data: Dict[str, Any]) -> float:
        """Calculate volatility indicators"""
        try:
            indicator_id = indicator['IndicatorID']
            
            if indicator_id == 'REALIZED_VOL':
                price_data = market_data.get('price_data', {})
                if price_data:
                    returns = price_data.get('returns', [])
                    if returns:
                        return np.std(returns) * np.sqrt(252)  # Annualized
                return 0.0
                
            elif indicator_id == 'IMPLIED_VOL_SURFACE':
                options_data = market_data.get('options_data', {})
                if options_data:
                    ivs = options_data.get('implied_volatility', [])
                    if ivs:
                        return np.mean(ivs)
                return 0.0
                
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating volatility indicator: {e}")
            return 0.0
    
    def _calculate_straddle_indicator(self, indicator: pd.Series, 
                                    market_data: Dict[str, Any]) -> float:
        """Calculate straddle analysis indicators"""
        try:
            indicator_id = indicator['IndicatorID']
            straddle_data = market_data.get('straddle_data', {})
            
            if indicator_id == 'STRADDLE_MOMENTUM':
                if straddle_data:
                    current_premium = straddle_data.get('total_premium', 0)
                    previous_premium = straddle_data.get('previous_premium', current_premium)
                    if previous_premium > 0:
                        momentum = (current_premium - previous_premium) / previous_premium
                        return momentum
                return 0.0
                
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating straddle indicator: {e}")
            return 0.0
    
    def _normalize_indicator_value(self, value: float, indicator: pd.Series) -> float:
        """Normalize indicator value to 0-1 range"""
        try:
            # Simple normalization - can be enhanced with historical ranges
            if value > 0:
                return min(1.0, value / 2.0 + 0.5)
            else:
                return max(0.0, value / 2.0 + 0.5)
                
        except Exception as e:
            logger.error(f"Error normalizing indicator value: {e}")
            return 0.5  # Neutral

    def _apply_dynamic_weightage(self, indicator_values: Dict[str, Dict[str, Any]],
                               user_config: Dict[str, Any]) -> Dict[str, float]:
        """Apply dynamic weightage to indicator values"""
        try:
            weighted_indicators = {}

            # Get current weights from configuration
            for indicator_id, indicator_data in indicator_values.items():
                if not indicator_data.get('enabled', True):
                    continue

                # Get base weight from weightage configuration
                weight_config = self.weightage_config[
                    self.weightage_config['IndicatorID'] == indicator_id
                ]

                if not weight_config.empty:
                    base_weight = weight_config.iloc[0]['CurrentWeight']
                    performance_weight = weight_config.iloc[0]['HistoricalPerformance']

                    # Apply user custom weights if available
                    category = indicator_data['category']
                    custom_weights = user_config.get('custom_weights', {})

                    if category in custom_weights:
                        # Adjust weight based on user preference
                        user_multiplier = custom_weights[category]
                        adjusted_weight = base_weight * user_multiplier
                    else:
                        adjusted_weight = base_weight

                    # Apply performance-based adjustment
                    final_weight = adjusted_weight * performance_weight

                    # Calculate weighted contribution
                    normalized_value = indicator_data['normalized_value']
                    weighted_indicators[indicator_id] = normalized_value * final_weight

                else:
                    # Default weight if not found in configuration
                    weighted_indicators[indicator_id] = indicator_data['normalized_value'] * 0.1

            # Normalize weights to sum to 1.0
            total_weight = sum(weighted_indicators.values())
            if total_weight > 0:
                weighted_indicators = {k: v/total_weight for k, v in weighted_indicators.items()}

            return weighted_indicators

        except Exception as e:
            logger.error(f"Error applying dynamic weightage: {e}")
            return {}

    def _calculate_directional_component(self, weighted_indicators: Dict[str, float]) -> float:
        """Calculate directional component from weighted indicators"""
        try:
            directional_indicators = []

            # Get indicators that contribute to directional bias
            for indicator_id, weight in weighted_indicators.items():
                indicator_config = self.indicator_config[
                    self.indicator_config['IndicatorID'] == indicator_id
                ]

                if not indicator_config.empty:
                    category = indicator_config.iloc[0]['Category']

                    # Categories that contribute to directional bias
                    if category in ['PRICE_ACTION', 'OI_ANALYSIS', 'GREEK_SENTIMENT', 'MOMENTUM_INDICATORS']:
                        directional_indicators.append(weight)

            if directional_indicators:
                # Calculate weighted average directional component
                directional_component = np.mean(directional_indicators)
                # Convert to -1 to +1 range (bearish to bullish)
                return (directional_component - 0.5) * 2
            else:
                return 0.0  # Neutral

        except Exception as e:
            logger.error(f"Error calculating directional component: {e}")
            return 0.0

    def _calculate_volatility_component(self, weighted_indicators: Dict[str, float]) -> float:
        """Calculate volatility component from weighted indicators"""
        try:
            volatility_indicators = []

            # Get indicators that contribute to volatility assessment
            for indicator_id, weight in weighted_indicators.items():
                indicator_config = self.indicator_config[
                    self.indicator_config['IndicatorID'] == indicator_id
                ]

                if not indicator_config.empty:
                    category = indicator_config.iloc[0]['Category']

                    # Categories that contribute to volatility assessment
                    if category in ['VOLATILITY_MEASURES', 'STRADDLE_ANALYSIS', 'OPTIONS_SPECIFIC']:
                        volatility_indicators.append(weight)

            if volatility_indicators:
                # Calculate weighted average volatility component
                volatility_component = np.mean(volatility_indicators)
                return volatility_component
            else:
                return 0.15  # Default normal volatility

        except Exception as e:
            logger.error(f"Error calculating volatility component: {e}")
            return 0.15

    def _determine_regime(self, directional_component: float, volatility_component: float,
                         weighted_indicators: Dict[str, float],
                         user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Determine regime based on components and custom definitions"""
        try:
            best_regime = None
            best_score = -1

            # Get enabled regime definitions
            enabled_regimes = self.regime_definitions[
                self.regime_definitions['Enabled'] == True
            ]

            # Apply user regime preferences
            regime_preferences = user_config.get('regime_preferences', 'ALL_REGIMES')
            if regime_preferences != 'ALL_REGIMES':
                # Filter regimes based on user preferences
                if regime_preferences == 'LOW_VOLATILITY_PREFERRED':
                    enabled_regimes = enabled_regimes[
                        enabled_regimes['RegimeID'].str.contains('LV_')
                    ]
                elif regime_preferences == 'HIGH_VOLATILITY_PREFERRED':
                    enabled_regimes = enabled_regimes[
                        enabled_regimes['RegimeID'].str.contains('HV_')
                    ]

            # Evaluate each regime definition
            for _, regime in enabled_regimes.iterrows():
                score = self._calculate_regime_match_score(
                    regime, directional_component, volatility_component,
                    weighted_indicators, user_config
                )

                if score > best_score:
                    best_score = score
                    best_regime = regime

            if best_regime is not None:
                return {
                    'regime_id': best_regime['RegimeID'],
                    'regime_name': best_regime['RegimeName'],
                    'regime_type': best_regime['RegimeType'],
                    'signal_strength': best_score,
                    'match_score': best_score
                }
            else:
                # Default to neutral regime
                return {
                    'regime_id': 'NV_NEUTRAL',
                    'regime_name': 'Normal Vol Neutral',
                    'regime_type': 'NEUTRAL',
                    'signal_strength': 0.5,
                    'match_score': 0.5
                }

        except Exception as e:
            logger.error(f"Error determining regime: {e}")
            return {
                'regime_id': 'ERROR',
                'regime_name': 'Error State',
                'regime_type': 'UNKNOWN',
                'signal_strength': 0.0,
                'match_score': 0.0
            }

    def _calculate_regime_match_score(self, regime: pd.Series,
                                    directional_component: float,
                                    volatility_component: float,
                                    weighted_indicators: Dict[str, float],
                                    user_config: Dict[str, Any]) -> float:
        """Calculate how well current conditions match a regime definition"""
        try:
            score = 0.0

            # Check directional threshold match
            directional_threshold = regime['DirectionalThreshold']
            directional_match = 1.0 - abs(directional_component - directional_threshold)
            directional_match = max(0.0, directional_match)

            # Check volatility threshold match
            volatility_threshold = regime['VolatilityThreshold']
            volatility_match = 1.0 - abs(volatility_component - volatility_threshold)
            volatility_match = max(0.0, volatility_match)

            # Combine directional and volatility scores
            base_score = (directional_match * 0.6) + (volatility_match * 0.4)

            # Apply user custom thresholds if available
            custom_thresholds = user_config.get('custom_thresholds', {})
            if 'CONFIDENCE' in custom_thresholds:
                confidence_threshold = custom_thresholds['CONFIDENCE']
                if base_score < confidence_threshold:
                    base_score *= 0.5  # Penalize low confidence matches

            # Apply historical accuracy boost
            historical_accuracy = regime.get('HistoricalAccuracy', 0.7)
            score = base_score * historical_accuracy

            return score

        except Exception as e:
            logger.error(f"Error calculating regime match score: {e}")
            return 0.0

    def _calculate_confidence_score(self, regime_result: Dict[str, Any],
                                  weighted_indicators: Dict[str, float],
                                  user_config: Dict[str, Any]) -> float:
        """Calculate confidence score for regime classification"""
        try:
            confidence_components = []

            # Get confidence configuration
            enabled_confidence_config = self.confidence_config[
                self.confidence_config['Enabled'] == True
            ]

            for _, config in enabled_confidence_config.iterrows():
                component = config['ScoreComponent']
                weight = config['Weight']

                if component == 'INDICATOR_AGREEMENT':
                    # Calculate how much indicators agree
                    indicator_values = list(weighted_indicators.values())
                    if indicator_values:
                        agreement = 1.0 - np.std(indicator_values)
                        confidence_components.append(agreement * weight)

                elif component == 'HISTORICAL_ACCURACY':
                    # Use historical accuracy from regime definition
                    accuracy = regime_result.get('match_score', 0.5)
                    confidence_components.append(accuracy * weight)

                elif component == 'SIGNAL_STRENGTH':
                    # Use signal strength from regime matching
                    strength = regime_result.get('signal_strength', 0.5)
                    confidence_components.append(strength * weight)

                elif component == 'MARKET_CONDITION':
                    # Assess current market conditions
                    condition_score = 0.7  # Default moderate confidence
                    confidence_components.append(condition_score * weight)

                elif component == 'TIME_CONSISTENCY':
                    # Check consistency over time (simplified)
                    consistency_score = 0.8  # Default good consistency
                    confidence_components.append(consistency_score * weight)

            # Calculate final confidence score
            if confidence_components:
                confidence_score = sum(confidence_components)
                return min(1.0, max(0.0, confidence_score))
            else:
                return 0.5  # Default moderate confidence

        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5

    def _assess_market_condition(self, market_data: Dict[str, Any]) -> str:
        """Assess current market condition"""
        try:
            # Simple market condition assessment
            price_data = market_data.get('price_data', {})

            if price_data:
                volume = price_data.get('volume', 0)
                avg_volume = price_data.get('avg_volume', volume)

                if volume > avg_volume * 1.5:
                    return 'HIGH_VOLUME'
                elif volume < avg_volume * 0.5:
                    return 'LOW_VOLUME'
                else:
                    return 'NORMAL_VOLUME'

            return 'UNKNOWN'

        except Exception as e:
            logger.error(f"Error assessing market condition: {e}")
            return 'UNKNOWN'

    def _store_regime_result(self, formation_result: RegimeFormationResult,
                           indicator_values: Dict[str, Dict[str, Any]]):
        """Store regime formation result in time-series database"""
        try:
            # Prepare regime data for storage
            regime_data = {
                'timestamp': formation_result.timestamp.isoformat(),
                'symbol': 'NIFTY',  # Default symbol
                'timeframe': '5min',  # Default timeframe
                'regime_id': formation_result.regime_id,
                'regime_name': formation_result.regime_name,
                'regime_type': formation_result.regime_type,
                'confidence_score': formation_result.confidence_score,
                'directional_component': formation_result.directional_component,
                'volatility_component': formation_result.volatility_component,
                'indicator_agreement': np.mean(list(formation_result.indicator_contributions.values())),
                'signal_strength': formation_result.signal_strength,
                'market_condition': formation_result.market_condition,
                'user_id': formation_result.metadata.get('user_id'),
                'configuration_id': formation_result.metadata.get('configuration_id'),
                'metadata': formation_result.metadata,
                'indicator_values': [
                    {
                        'indicator_id': ind_id,
                        'indicator_name': ind_data['indicator_name'],
                        'indicator_category': ind_data['category'],
                        'raw_value': ind_data['raw_value'],
                        'normalized_value': ind_data['normalized_value'],
                        'weight': formation_result.indicator_contributions.get(ind_id, 0.0),
                        'contribution': formation_result.indicator_contributions.get(ind_id, 0.0),
                        'performance_score': 0.8,  # Default performance score
                        'metadata': ind_data.get('metadata', {})
                    }
                    for ind_id, ind_data in indicator_values.items()
                    if ind_data.get('enabled', True)
                ]
            }

            # Store in database
            self.storage.store_regime_classification(regime_data)

        except Exception as e:
            logger.error(f"Error storing regime result: {e}")

    def update_indicator_performance(self, performance_data: Dict[str, float]):
        """Update indicator performance and adjust weights"""
        try:
            logger.info("Updating indicator performance and weights")

            for indicator_id, performance_score in performance_data.items():
                # Update performance in weightage configuration
                mask = self.weightage_config['IndicatorID'] == indicator_id
                if mask.any():
                    current_performance = self.weightage_config.loc[mask, 'HistoricalPerformance'].iloc[0]
                    learning_rate = self.weightage_config.loc[mask, 'LearningRate'].iloc[0]

                    # Update performance with exponential moving average
                    new_performance = (1 - learning_rate) * current_performance + learning_rate * performance_score
                    self.weightage_config.loc[mask, 'HistoricalPerformance'] = new_performance
                    self.weightage_config.loc[mask, 'LastUpdated'] = datetime.now().strftime('%Y-%m-%d')

                    # Adjust current weight based on performance
                    if self.weightage_config.loc[mask, 'AutoAdjust'].iloc[0]:
                        base_weight = self.indicator_config[
                            self.indicator_config['IndicatorID'] == indicator_id
                        ]['BaseWeight'].iloc[0]

                        # Performance-based weight adjustment
                        performance_multiplier = 0.5 + (new_performance * 1.0)  # 0.5 to 1.5 range
                        new_weight = base_weight * performance_multiplier

                        # Apply bounds
                        min_weight = self.weightage_config.loc[mask, 'MinPerformanceThreshold'].iloc[0]
                        max_weight = self.weightage_config.loc[mask, 'MaxPerformanceThreshold'].iloc[0]
                        new_weight = max(min_weight, min(max_weight, new_weight))

                        self.weightage_config.loc[mask, 'CurrentWeight'] = new_weight

                        logger.info(f"Updated {indicator_id}: performance={new_performance:.3f}, weight={new_weight:.3f}")

            # Normalize weights to sum to 1.0
            auto_adjust_indicators = self.weightage_config[self.weightage_config['AutoAdjust'] == True]
            total_weight = auto_adjust_indicators['CurrentWeight'].sum()

            if total_weight > 0:
                for idx in auto_adjust_indicators.index:
                    current_weight = self.weightage_config.loc[idx, 'CurrentWeight']
                    normalized_weight = current_weight / total_weight
                    self.weightage_config.loc[idx, 'CurrentWeight'] = normalized_weight

            # Save updated configuration
            self.config_manager.save_configuration()

            # Track weight adjustment history
            self.weight_adjustment_history.append({
                'timestamp': datetime.now(),
                'performance_data': performance_data,
                'updated_weights': self.weightage_config[['IndicatorID', 'CurrentWeight']].to_dict('records')
            })

            logger.info("✅ Indicator performance and weights updated successfully")

        except Exception as e:
            logger.error(f"Error updating indicator performance: {e}")

    def get_regime_formation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of regime formation engine"""
        try:
            summary = {
                'configuration_summary': self.config_manager.get_configuration_summary(),
                'total_indicators': len(self.indicator_config),
                'enabled_indicators': len(self.indicator_config[self.indicator_config['Enabled'] == True]),
                'total_regimes': len(self.regime_definitions),
                'enabled_regimes': len(self.regime_definitions[self.regime_definitions['Enabled'] == True]),
                'user_profiles': len(self.user_profiles[self.user_profiles['Active'] == True]),
                'weight_adjustments': len(self.weight_adjustment_history),
                'last_weight_update': self.weight_adjustment_history[-1]['timestamp'].isoformat() if self.weight_adjustment_history else None,
                'indicator_categories': self.indicator_config['Category'].value_counts().to_dict(),
                'regime_types': self.regime_definitions['RegimeType'].value_counts().to_dict()
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting regime formation summary: {e}")
            return {'error': str(e)}

    def close(self):
        """Close all connections and save state"""
        try:
            # Save final configuration
            self.config_manager.save_configuration()

            # Close storage connection
            self.storage.close_connection()

            logger.info("✅ Enhanced Regime Formation Engine closed successfully")

        except Exception as e:
            logger.error(f"Error closing regime formation engine: {e}")
