"""
Market Regime Calculator

This module handles the core calculation of market regime indicators
and aggregation of signals from multiple sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Base component class for market regime components
class BaseComponent:
    """Base class for market regime components"""
    def __init__(self, config: dict):
        self.config = config
from .models import RegimeConfig, IndicatorConfig, RegimeClassification, RegimeType
# from ..indicators import get_indicator  # Will use fallback implementation

logger = logging.getLogger(__name__)

class RegimeCalculator(BaseComponent):
    """
    Main regime calculation engine following backtester_v2 patterns
    
    This class orchestrates the calculation of all indicators and
    aggregates them into regime classifications.
    """
    
    def __init__(self, config: RegimeConfig):
        """
        Initialize the regime calculator
        
        Args:
            config (RegimeConfig): Configuration for regime calculation
        """
        super().__init__(config.model_dump())
        self.config = config
        self.indicators = {}
        self.indicator_weights = {}
        self.performance_history = {}
        
        # Initialize indicators
        self._initialize_indicators()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"RegimeCalculator initialized with {len(self.indicators)} indicators")
    
    def _initialize_indicators(self):
        """Initialize all configured indicators"""
        for ind_config in self.config.indicators:
            if not ind_config.enabled:
                continue
                
            try:
                # Create indicator instance
                indicator = self._create_indicator(ind_config)
                if indicator:
                    self.indicators[ind_config.id] = indicator
                    logger.debug(f"Initialized indicator: {ind_config.id}")
                    
            except Exception as e:
                logger.error(f"Error initializing indicator {ind_config.id}: {e}")
    
    def _create_indicator(self, config: IndicatorConfig):
        """Create indicator instance based on configuration"""
        try:
            # Map indicator types to implementations
            indicator_map = {
                'ema': 'ema_enhanced',
                'vwap': 'vwap_enhanced', 
                'greek': 'greek_sentiment',
                'iv_skew': 'iv_analysis',
                'premium': 'premium_analysis',
                'oi_flow': 'oi_flow',
                'atr': 'atr_enhanced',
                'momentum': 'momentum_indicators',
                'volume': 'volume_indicators'
            }
            
            indicator_type = indicator_map.get(config.indicator_type, config.indicator_type)
            
            # Create indicator configuration
            ind_config = {
                'id': config.id,
                'name': config.name,
                'category': config.category.value,
                'weight': config.base_weight,
                'adaptive': config.adaptive,
                'lookback_periods': config.lookback_periods,
                **config.parameters
            }
            
            # Use fallback implementation for testing
            return self._create_fallback_indicator(config)
                
        except Exception as e:
            logger.error(f"Error creating indicator {config.id}: {e}")
            return None
    
    def _create_fallback_indicator(self, config: IndicatorConfig):
        """Create fallback indicator implementation"""
        # Simple fallback implementations for testing
        class FallbackIndicator:
            def __init__(self, config):
                self.config = config
                
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
                # Simple moving average as fallback
                result = pd.DataFrame(index=data.index)
                if 'close' in data.columns:
                    result['signal'] = data['close'].rolling(20).mean().pct_change()
                    result['confidence'] = 0.7
                else:
                    result['signal'] = 0.0
                    result['confidence'] = 0.5
                return result
                
            def get_signal(self, values: pd.DataFrame) -> float:
                if 'signal' in values.columns:
                    return values['signal'].iloc[-1] if len(values) > 0 else 0.0
                return 0.0
        
        return FallbackIndicator(config)
    
    def _initialize_weights(self):
        """Initialize indicator weights"""
        total_weight = sum(ind.base_weight for ind in self.config.indicators if ind.enabled)
        
        for ind_config in self.config.indicators:
            if ind_config.enabled:
                # Normalize weights
                normalized_weight = ind_config.base_weight / total_weight if total_weight > 0 else 1.0 / len(self.config.indicators)
                self.indicator_weights[ind_config.id] = {
                    'current_weight': normalized_weight,
                    'base_weight': ind_config.base_weight,
                    'min_weight': ind_config.min_weight,
                    'max_weight': ind_config.max_weight,
                    'performance_multiplier': 1.0,
                    'adaptive': ind_config.adaptive
                }
    
    def calculate_regime(self, market_data: pd.DataFrame, **kwargs) -> List[RegimeClassification]:
        """
        Calculate market regime for given data
        
        Args:
            market_data (pd.DataFrame): Market data with OHLCV and options data
            **kwargs: Additional parameters
            
        Returns:
            List[RegimeClassification]: Regime classifications
        """
        try:
            if market_data.empty:
                logger.warning("Empty market data provided")
                return []
            
            # Calculate all indicators
            indicator_results = self._calculate_all_indicators(market_data, **kwargs)
            
            if not indicator_results:
                logger.warning("No indicator results calculated")
                return []
            
            # Aggregate signals for each timeframe
            aggregated_signals = self._aggregate_signals(indicator_results, market_data.index)
            
            # Classify regimes
            classifications = self._classify_regimes(aggregated_signals, market_data)
            
            logger.info(f"Calculated {len(classifications)} regime classifications")
            return classifications
            
        except Exception as e:
            logger.error(f"Error calculating regime: {e}")
            return []
    
    def _calculate_all_indicators(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
        """Calculate all indicators in parallel"""
        results = {}
        
        # Use ThreadPoolExecutor for parallel calculation
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for ind_id, indicator in self.indicators.items():
                future = executor.submit(self._safe_calculate_indicator, indicator, data, **kwargs)
                futures[ind_id] = future
            
            # Collect results
            for ind_id, future in futures.items():
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    if result is not None and not result.empty:
                        results[ind_id] = result
                except Exception as e:
                    logger.error(f"Error calculating indicator {ind_id}: {e}")
        
        return results
    
    def _safe_calculate_indicator(self, indicator, data: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """Safely calculate indicator with error handling"""
        try:
            return indicator.calculate(data, **kwargs)
        except Exception as e:
            logger.error(f"Error in indicator calculation: {e}")
            return None
    
    def _aggregate_signals(self, indicator_results: Dict[str, pd.DataFrame], 
                          index: pd.Index) -> pd.DataFrame:
        """Aggregate signals from all indicators"""
        try:
            # Create result DataFrame
            result = pd.DataFrame(index=index)
            result['weighted_signal'] = 0.0
            result['total_weight'] = 0.0
            result['confidence'] = 0.0
            
            # Aggregate each indicator
            for ind_id, ind_result in indicator_results.items():
                if ind_id not in self.indicator_weights:
                    continue
                
                weight_info = self.indicator_weights[ind_id]
                current_weight = weight_info['current_weight']
                
                # Extract signal and confidence
                signal, confidence = self._extract_signal_and_confidence(ind_result)
                
                if signal is not None and confidence is not None:
                    # Align to common index
                    signal_aligned = signal.reindex(index, fill_value=0.0)
                    confidence_aligned = confidence.reindex(index, fill_value=0.5)
                    
                    # Weight the signal
                    weighted_signal = signal_aligned * current_weight * confidence_aligned
                    
                    # Add to aggregated result
                    result['weighted_signal'] += weighted_signal
                    result['total_weight'] += current_weight * confidence_aligned
                    result['confidence'] += confidence_aligned * current_weight
                    
                    # Store individual contributions
                    result[f'{ind_id}_signal'] = signal_aligned
                    result[f'{ind_id}_confidence'] = confidence_aligned
                    result[f'{ind_id}_weight'] = current_weight
            
            # Normalize by total weight
            mask = result['total_weight'] > 0
            result.loc[mask, 'weighted_signal'] /= result.loc[mask, 'total_weight']
            result.loc[mask, 'confidence'] /= result.loc[mask, 'total_weight']

            # Fill NaN values with defaults
            result['weighted_signal'] = result['weighted_signal'].fillna(0.0)
            result['confidence'] = result['confidence'].fillna(0.5)

            # Clip signals to [-2, 2] range
            result['weighted_signal'] = result['weighted_signal'].clip(-2.0, 2.0)
            result['confidence'] = result['confidence'].clip(0.0, 1.0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error aggregating signals: {e}")
            return pd.DataFrame(index=index)
    
    def _extract_signal_and_confidence(self, result: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Extract signal and confidence from indicator result"""
        signal = None
        confidence = None
        
        # Look for signal columns
        signal_cols = [col for col in result.columns if 'signal' in col.lower()]
        if signal_cols:
            signal = result[signal_cols[0]]
        
        # Look for confidence columns
        confidence_cols = [col for col in result.columns if 'confidence' in col.lower()]
        if confidence_cols:
            confidence = result[confidence_cols[0]]
        else:
            # Default confidence
            confidence = pd.Series(0.7, index=result.index)
        
        return signal, confidence
    
    def _classify_regimes(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> List[RegimeClassification]:
        """Classify regimes based on aggregated signals"""
        classifications = []
        
        for timestamp, row in signals.iterrows():
            try:
                regime_score = row['weighted_signal']
                confidence = row['confidence']
                
                # Determine regime type based on score
                regime_type = self._score_to_regime_type(regime_score)
                
                # Extract component scores
                component_scores = {}
                for col in signals.columns:
                    if col.endswith('_signal'):
                        ind_id = col.replace('_signal', '')
                        component_scores[ind_id] = row[col]
                
                # Create classification
                classification = RegimeClassification(
                    timestamp=timestamp,
                    symbol=self.config.symbol,
                    regime_type=regime_type,
                    regime_score=regime_score,
                    confidence=confidence,
                    component_scores=component_scores,
                    timeframe_scores={},  # TODO: Add timeframe-specific scores
                    metadata={
                        'total_weight': row['total_weight'],
                        'num_indicators': len(component_scores)
                    }
                )
                
                classifications.append(classification)
                
            except Exception as e:
                logger.error(f"Error classifying regime for {timestamp}: {e}")
                continue
        
        return classifications
    
    def _score_to_regime_type(self, score: float) -> RegimeType:
        """Convert regime score to regime type"""
        if score >= 1.5:
            return RegimeType.STRONG_BULLISH
        elif score >= 0.75:
            return RegimeType.MODERATE_BULLISH
        elif score >= 0.25:
            return RegimeType.WEAK_BULLISH
        elif score >= -0.25:
            return RegimeType.NEUTRAL
        elif score >= -0.75:
            return RegimeType.WEAK_BEARISH
        elif score >= -1.5:
            return RegimeType.MODERATE_BEARISH
        else:
            return RegimeType.STRONG_BEARISH
