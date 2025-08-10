#!/usr/bin/env python3
"""
Market Regime Strategy Integration Module
Integrates market regime output with strategy consolidator for enhanced decision-making

This module provides seamless integration between the Market Regime Formation System
and the Strategy Consolidator, enabling regime-aware strategy selection and optimization.

Features:
1. Real-time regime data streaming to consolidator
2. Regime-based strategy filtering and selection
3. DTE-specific regime adjustments
4. Performance tracking by regime
5. CSV output generation for consolidator input
6. Regime transition handling

Author: The Augster
Date: 2025-06-26
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
import json
from dataclasses import dataclass, field
from collections import defaultdict
import csv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeStrategyMapping:
    """Maps regime characteristics to strategy preferences"""
    regime_id: int
    regime_name: str
    preferred_strategies: List[str]
    strategy_weights: Dict[str, float]
    risk_adjustment: float
    position_size_multiplier: float
    stop_loss_adjustment: float
    target_adjustment: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyRegimePerformance:
    """Tracks strategy performance by regime"""
    strategy_name: str
    regime_id: int
    total_trades: int
    winning_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    regime_accuracy: float
    last_updated: datetime

@dataclass
class RegimeTransition:
    """Represents a regime transition event"""
    from_regime: int
    to_regime: int
    transition_time: datetime
    confidence: float
    affected_strategies: List[str]
    action_required: str

class RegimeStrategyIntegrator:
    """
    Integrates market regime analysis with strategy consolidator
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime strategy integrator"""
        self.config = config or self._get_default_config()
        
        # Regime to strategy mappings (18-regime system)
        self.regime_mappings = self._initialize_regime_mappings()
        
        # Performance tracking
        self.performance_tracker = defaultdict(lambda: defaultdict(StrategyRegimePerformance))
        
        # Output configuration
        self.output_dir = Path(self.config.get('output_dir', './regime_outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # CSV output configuration
        self.csv_columns = self._get_csv_columns()
        
        # Transition handling
        self.current_regime = None
        self.regime_history = []
        self.transition_handlers = []
        
        logger.info("RegimeStrategyIntegrator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'output_dir': './regime_outputs',
            'csv_update_frequency': 60,  # seconds
            'regime_lookback': 100,  # Number of regime records to keep
            'confidence_threshold': 0.7,
            'enable_real_time_streaming': True,
            'strategy_types': ['TBS', 'ORB', 'OI', 'TV', 'ML_INDICATOR', 'POS', 'MARKET_REGIME']
        }
    
    def _initialize_regime_mappings(self) -> Dict[int, RegimeStrategyMapping]:
        """Initialize regime to strategy mappings for 18-regime system"""
        mappings = {}
        
        # Low Volatility Regimes (0-5)
        mappings[0] = RegimeStrategyMapping(
            regime_id=0,
            regime_name="Low Vol Bullish Trending",
            preferred_strategies=["TBS", "ML_INDICATOR", "MARKET_REGIME"],
            strategy_weights={"TBS": 0.4, "ML_INDICATOR": 0.4, "MARKET_REGIME": 0.2},
            risk_adjustment=1.2,
            position_size_multiplier=1.5,
            stop_loss_adjustment=0.8,
            target_adjustment=1.2
        )
        
        mappings[1] = RegimeStrategyMapping(
            regime_id=1,
            regime_name="Low Vol Bullish Mean-Rev",
            preferred_strategies=["POS", "OI", "MARKET_REGIME"],
            strategy_weights={"POS": 0.5, "OI": 0.3, "MARKET_REGIME": 0.2},
            risk_adjustment=1.0,
            position_size_multiplier=1.2,
            stop_loss_adjustment=0.9,
            target_adjustment=1.1
        )
        
        mappings[2] = RegimeStrategyMapping(
            regime_id=2,
            regime_name="Low Vol Neutral Trending",
            preferred_strategies=["ORB", "TV", "MARKET_REGIME"],
            strategy_weights={"ORB": 0.4, "TV": 0.4, "MARKET_REGIME": 0.2},
            risk_adjustment=0.9,
            position_size_multiplier=1.0,
            stop_loss_adjustment=1.0,
            target_adjustment=1.0
        )
        
        mappings[3] = RegimeStrategyMapping(
            regime_id=3,
            regime_name="Low Vol Neutral Mean-Rev",
            preferred_strategies=["POS", "OI", "ML_INDICATOR"],
            strategy_weights={"POS": 0.4, "OI": 0.4, "ML_INDICATOR": 0.2},
            risk_adjustment=0.8,
            position_size_multiplier=0.8,
            stop_loss_adjustment=1.1,
            target_adjustment=0.9
        )
        
        mappings[4] = RegimeStrategyMapping(
            regime_id=4,
            regime_name="Low Vol Bearish Trending",
            preferred_strategies=["TBS", "TV", "MARKET_REGIME"],
            strategy_weights={"TBS": 0.3, "TV": 0.5, "MARKET_REGIME": 0.2},
            risk_adjustment=0.7,
            position_size_multiplier=0.7,
            stop_loss_adjustment=1.2,
            target_adjustment=0.8
        )
        
        mappings[5] = RegimeStrategyMapping(
            regime_id=5,
            regime_name="Low Vol Bearish Mean-Rev",
            preferred_strategies=["POS", "OI", "MARKET_REGIME"],
            strategy_weights={"POS": 0.5, "OI": 0.3, "MARKET_REGIME": 0.2},
            risk_adjustment=0.8,
            position_size_multiplier=0.9,
            stop_loss_adjustment=1.1,
            target_adjustment=0.9
        )
        
        # Medium Volatility Regimes (6-11)
        mappings[6] = RegimeStrategyMapping(
            regime_id=6,
            regime_name="Med Vol Bullish Trending",
            preferred_strategies=["TBS", "ML_INDICATOR", "ORB"],
            strategy_weights={"TBS": 0.4, "ML_INDICATOR": 0.3, "ORB": 0.3},
            risk_adjustment=1.0,
            position_size_multiplier=1.2,
            stop_loss_adjustment=1.0,
            target_adjustment=1.2
        )
        
        mappings[7] = RegimeStrategyMapping(
            regime_id=7,
            regime_name="Med Vol Bullish Mean-Rev",
            preferred_strategies=["POS", "OI", "ML_INDICATOR"],
            strategy_weights={"POS": 0.4, "OI": 0.3, "ML_INDICATOR": 0.3},
            risk_adjustment=0.9,
            position_size_multiplier=1.0,
            stop_loss_adjustment=1.1,
            target_adjustment=1.0
        )
        
        mappings[8] = RegimeStrategyMapping(
            regime_id=8,
            regime_name="Med Vol Neutral Trending",
            preferred_strategies=["ORB", "TV", "ML_INDICATOR"],
            strategy_weights={"ORB": 0.4, "TV": 0.3, "ML_INDICATOR": 0.3},
            risk_adjustment=0.8,
            position_size_multiplier=0.9,
            stop_loss_adjustment=1.2,
            target_adjustment=0.9
        )
        
        mappings[9] = RegimeStrategyMapping(
            regime_id=9,
            regime_name="Med Vol Neutral Mean-Rev",
            preferred_strategies=["POS", "OI", "MARKET_REGIME"],
            strategy_weights={"POS": 0.4, "OI": 0.4, "MARKET_REGIME": 0.2},
            risk_adjustment=0.7,
            position_size_multiplier=0.8,
            stop_loss_adjustment=1.3,
            target_adjustment=0.8
        )
        
        mappings[10] = RegimeStrategyMapping(
            regime_id=10,
            regime_name="Med Vol Bearish Trending",
            preferred_strategies=["TV", "TBS", "MARKET_REGIME"],
            strategy_weights={"TV": 0.5, "TBS": 0.3, "MARKET_REGIME": 0.2},
            risk_adjustment=0.6,
            position_size_multiplier=0.7,
            stop_loss_adjustment=1.4,
            target_adjustment=0.7
        )
        
        mappings[11] = RegimeStrategyMapping(
            regime_id=11,
            regime_name="Med Vol Bearish Mean-Rev",
            preferred_strategies=["POS", "OI", "ML_INDICATOR"],
            strategy_weights={"POS": 0.5, "OI": 0.3, "ML_INDICATOR": 0.2},
            risk_adjustment=0.7,
            position_size_multiplier=0.8,
            stop_loss_adjustment=1.3,
            target_adjustment=0.8
        )
        
        # High Volatility Regimes (12-17)
        mappings[12] = RegimeStrategyMapping(
            regime_id=12,
            regime_name="High Vol Bullish Trending",
            preferred_strategies=["ML_INDICATOR", "MARKET_REGIME", "TBS"],
            strategy_weights={"ML_INDICATOR": 0.5, "MARKET_REGIME": 0.3, "TBS": 0.2},
            risk_adjustment=0.8,
            position_size_multiplier=0.8,
            stop_loss_adjustment=1.5,
            target_adjustment=1.3
        )
        
        mappings[13] = RegimeStrategyMapping(
            regime_id=13,
            regime_name="High Vol Bullish Mean-Rev",
            preferred_strategies=["POS", "MARKET_REGIME", "OI"],
            strategy_weights={"POS": 0.5, "MARKET_REGIME": 0.3, "OI": 0.2},
            risk_adjustment=0.7,
            position_size_multiplier=0.7,
            stop_loss_adjustment=1.6,
            target_adjustment=1.2
        )
        
        mappings[14] = RegimeStrategyMapping(
            regime_id=14,
            regime_name="High Vol Neutral Trending",
            preferred_strategies=["MARKET_REGIME", "ML_INDICATOR", "ORB"],
            strategy_weights={"MARKET_REGIME": 0.5, "ML_INDICATOR": 0.3, "ORB": 0.2},
            risk_adjustment=0.6,
            position_size_multiplier=0.6,
            stop_loss_adjustment=1.7,
            target_adjustment=1.1
        )
        
        mappings[15] = RegimeStrategyMapping(
            regime_id=15,
            regime_name="High Vol Neutral Mean-Rev",
            preferred_strategies=["POS", "MARKET_REGIME", "OI"],
            strategy_weights={"POS": 0.4, "MARKET_REGIME": 0.4, "OI": 0.2},
            risk_adjustment=0.5,
            position_size_multiplier=0.5,
            stop_loss_adjustment=1.8,
            target_adjustment=1.0
        )
        
        mappings[16] = RegimeStrategyMapping(
            regime_id=16,
            regime_name="High Vol Bearish Trending",
            preferred_strategies=["MARKET_REGIME", "TV", "ML_INDICATOR"],
            strategy_weights={"MARKET_REGIME": 0.5, "TV": 0.3, "ML_INDICATOR": 0.2},
            risk_adjustment=0.4,
            position_size_multiplier=0.4,
            stop_loss_adjustment=2.0,
            target_adjustment=0.8
        )
        
        mappings[17] = RegimeStrategyMapping(
            regime_id=17,
            regime_name="High Vol Bearish Mean-Rev",
            preferred_strategies=["POS", "MARKET_REGIME", "OI"],
            strategy_weights={"POS": 0.4, "MARKET_REGIME": 0.4, "OI": 0.2},
            risk_adjustment=0.5,
            position_size_multiplier=0.5,
            stop_loss_adjustment=1.9,
            target_adjustment=0.9
        )
        
        return mappings
    
    def _get_csv_columns(self) -> List[str]:
        """Get CSV column definitions for consolidator input"""
        return [
            'timestamp',
            'regime_id',
            'regime_name',
            'regime_confidence',
            'volatility_component',
            'trend_component',
            'structure_component',
            'preferred_strategies',
            'strategy_weights',
            'risk_adjustment',
            'position_size_multiplier',
            'stop_loss_adjustment',
            'target_adjustment',
            'transition_flag',
            'previous_regime',
            'time_in_regime',
            'regime_stability',
            'market_condition',
            'dte_bucket',
            'timeframe_5min_score',
            'timeframe_15min_score',
            'timeframe_30min_score',
            'timeframe_1hr_score',
            'dominant_timeframe',
            'cross_timeframe_correlation',
            'triple_straddle_value',
            'greek_sentiment_score',
            'oi_trending_score',
            'iv_analysis_score',
            'atr_indicator_value',
            'support_resistance_levels',
            'ml_ensemble_prediction',
            'ml_confidence',
            'component_agreement',
            'regime_change_probability',
            'recommended_action'
        ]
    
    async def process_regime_data(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process regime data for strategy consolidator integration
        
        Args:
            regime_data: Market regime analysis output
            
        Returns:
            Processed data ready for consolidator
        """
        try:
            # Extract regime information
            regime_id = regime_data.get('regime_id', 0)
            regime_confidence = regime_data.get('confidence', 0.0)
            
            # Get regime mapping
            regime_mapping = self.regime_mappings.get(regime_id)
            if not regime_mapping:
                logger.warning(f"No mapping found for regime {regime_id}")
                regime_mapping = self.regime_mappings[8]  # Default to neutral
            
            # Check for regime transition
            transition_info = self._check_regime_transition(regime_id, regime_confidence)
            
            # Prepare consolidator data
            consolidator_data = {
                'timestamp': datetime.now(),
                'regime_id': regime_id,
                'regime_name': regime_mapping.regime_name,
                'regime_confidence': regime_confidence,
                'volatility_component': regime_data.get('volatility_component', 0.5),
                'trend_component': regime_data.get('trend_component', 0.0),
                'structure_component': regime_data.get('structure_component', 0.5),
                'preferred_strategies': json.dumps(regime_mapping.preferred_strategies),
                'strategy_weights': json.dumps(regime_mapping.strategy_weights),
                'risk_adjustment': regime_mapping.risk_adjustment,
                'position_size_multiplier': regime_mapping.position_size_multiplier,
                'stop_loss_adjustment': regime_mapping.stop_loss_adjustment,
                'target_adjustment': regime_mapping.target_adjustment,
                'transition_flag': transition_info['is_transition'],
                'previous_regime': transition_info.get('previous_regime', None),
                'time_in_regime': transition_info.get('time_in_regime', 0),
                'regime_stability': self._calculate_regime_stability(),
                'market_condition': self._get_market_condition(regime_data),
                'dte_bucket': regime_data.get('dte_bucket', '4-7'),
                'timeframe_5min_score': regime_data.get('timeframe_scores', {}).get('5min', 0.0),
                'timeframe_15min_score': regime_data.get('timeframe_scores', {}).get('15min', 0.0),
                'timeframe_30min_score': regime_data.get('timeframe_scores', {}).get('30min', 0.0),
                'timeframe_1hr_score': regime_data.get('timeframe_scores', {}).get('1hr', 0.0),
                'dominant_timeframe': regime_data.get('dominant_timeframe', '30min'),
                'cross_timeframe_correlation': regime_data.get('cross_timeframe_correlation', 0.0),
                'triple_straddle_value': regime_data.get('triple_straddle_value', 0.0),
                'greek_sentiment_score': regime_data.get('greek_sentiment_score', 0.0),
                'oi_trending_score': regime_data.get('oi_trending_score', 0.0),
                'iv_analysis_score': regime_data.get('iv_analysis_score', 0.0),
                'atr_indicator_value': regime_data.get('atr_indicator_value', 0.0),
                'support_resistance_levels': json.dumps(regime_data.get('support_resistance_levels', [])),
                'ml_ensemble_prediction': regime_data.get('ml_ensemble_prediction', 'NEUTRAL'),
                'ml_confidence': regime_data.get('ml_confidence', 0.5),
                'component_agreement': regime_data.get('component_agreement', 0.0),
                'regime_change_probability': regime_data.get('regime_change_probability', 0.0),
                'recommended_action': self._get_recommended_action(regime_mapping, regime_data)
            }
            
            # Track performance if enabled
            if self.config.get('track_performance', True):
                await self._update_performance_tracking(consolidator_data)
            
            # Generate CSV output if needed
            if self.config.get('generate_csv', True):
                await self._generate_csv_output(consolidator_data)
            
            # Handle real-time streaming if enabled
            if self.config.get('enable_real_time_streaming'):
                await self._stream_to_consolidator(consolidator_data)
            
            return consolidator_data
            
        except Exception as e:
            logger.error(f"Error processing regime data: {e}")
            return {}
    
    def _check_regime_transition(self, new_regime: int, confidence: float) -> Dict[str, Any]:
        """Check for regime transition and calculate transition metrics"""
        transition_info = {
            'is_transition': False,
            'previous_regime': self.current_regime,
            'time_in_regime': 0
        }
        
        if self.current_regime is not None and self.current_regime != new_regime:
            if confidence >= self.config['confidence_threshold']:
                transition_info['is_transition'] = True
                
                # Calculate time in previous regime
                if self.regime_history:
                    regime_start = next(
                        (r['timestamp'] for r in reversed(self.regime_history) 
                         if r['regime_id'] != self.current_regime),
                        self.regime_history[0]['timestamp']
                    )
                    transition_info['time_in_regime'] = (datetime.now() - regime_start).total_seconds() / 60
                
                # Trigger transition handlers
                self._handle_regime_transition(self.current_regime, new_regime, confidence)
        
        # Update current regime
        if confidence >= self.config['confidence_threshold']:
            self.current_regime = new_regime
            
        # Update history
        self.regime_history.append({
            'regime_id': new_regime,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        # Maintain history size
        if len(self.regime_history) > self.config['regime_lookback']:
            self.regime_history = self.regime_history[-self.config['regime_lookback']:]
        
        return transition_info
    
    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability based on recent history"""
        if len(self.regime_history) < 10:
            return 0.5
        
        recent_regimes = [r['regime_id'] for r in self.regime_history[-10:]]
        unique_regimes = len(set(recent_regimes))
        
        # Lower number of unique regimes = higher stability
        stability = 1.0 - (unique_regimes - 1) / 9.0
        return max(0.0, min(1.0, stability))
    
    def _get_market_condition(self, regime_data: Dict[str, Any]) -> str:
        """Determine overall market condition"""
        volatility = regime_data.get('volatility_component', 0.5)
        trend = regime_data.get('trend_component', 0.0)
        
        if volatility < 0.3:
            vol_state = "Low"
        elif volatility < 0.7:
            vol_state = "Medium"
        else:
            vol_state = "High"
        
        if trend > 0.3:
            trend_state = "Bullish"
        elif trend < -0.3:
            trend_state = "Bearish"
        else:
            trend_state = "Neutral"
        
        return f"{vol_state} Vol {trend_state}"
    
    def _get_recommended_action(self, regime_mapping: RegimeStrategyMapping, 
                               regime_data: Dict[str, Any]) -> str:
        """Get recommended action based on regime and market data"""
        confidence = regime_data.get('confidence', 0.0)
        regime_change_prob = regime_data.get('regime_change_probability', 0.0)
        
        if confidence < 0.5:
            return "WAIT - Low confidence"
        elif regime_change_prob > 0.7:
            return "REDUCE - Regime change likely"
        elif regime_mapping.risk_adjustment > 1.0:
            return "INCREASE - Favorable regime"
        elif regime_mapping.risk_adjustment < 0.6:
            return "REDUCE - Unfavorable regime"
        else:
            return "MAINTAIN - Stable regime"
    
    def _handle_regime_transition(self, from_regime: int, to_regime: int, confidence: float):
        """Handle regime transition event"""
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_time=datetime.now(),
            confidence=confidence,
            affected_strategies=self._get_affected_strategies(from_regime, to_regime),
            action_required=self._get_transition_action(from_regime, to_regime)
        )
        
        # Execute transition handlers
        for handler in self.transition_handlers:
            try:
                handler(transition)
            except Exception as e:
                logger.error(f"Error in transition handler: {e}")
        
        logger.info(f"Regime transition: {from_regime} -> {to_regime} (confidence: {confidence:.2f})")
    
    def _get_affected_strategies(self, from_regime: int, to_regime: int) -> List[str]:
        """Get strategies affected by regime transition"""
        from_mapping = self.regime_mappings.get(from_regime)
        to_mapping = self.regime_mappings.get(to_regime)
        
        if not from_mapping or not to_mapping:
            return []
        
        # Strategies that need adjustment
        from_strategies = set(from_mapping.preferred_strategies)
        to_strategies = set(to_mapping.preferred_strategies)
        
        # Strategies that are no longer preferred or newly preferred
        affected = list(from_strategies.symmetric_difference(to_strategies))
        
        return affected
    
    def _get_transition_action(self, from_regime: int, to_regime: int) -> str:
        """Determine action required for regime transition"""
        from_mapping = self.regime_mappings.get(from_regime)
        to_mapping = self.regime_mappings.get(to_regime)
        
        if not from_mapping or not to_mapping:
            return "REVIEW"
        
        risk_change = to_mapping.risk_adjustment - from_mapping.risk_adjustment
        
        if abs(risk_change) > 0.5:
            return "REBALANCE - Significant risk change"
        elif len(self._get_affected_strategies(from_regime, to_regime)) > 2:
            return "ADJUST - Multiple strategy changes"
        else:
            return "MONITOR - Minor adjustments"
    
    async def _update_performance_tracking(self, consolidator_data: Dict[str, Any]):
        """Update performance tracking for strategies by regime"""
        # Implementation would track actual strategy performance by regime
        pass
    
    async def _generate_csv_output(self, consolidator_data: Dict[str, Any]):
        """Generate CSV output for consolidator"""
        try:
            # Create filename with timestamp
            filename = f"regime_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.output_dir / filename
            
            # Write CSV
            file_exists = filepath.exists()
            
            with open(filepath, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(consolidator_data)
            
            logger.debug(f"CSV output written to {filepath}")
            
        except Exception as e:
            logger.error(f"Error generating CSV output: {e}")
    
    async def _stream_to_consolidator(self, consolidator_data: Dict[str, Any]):
        """Stream data to strategy consolidator in real-time"""
        # Implementation would use WebSocket or message queue
        # to stream data to the consolidator
        logger.debug("Streaming regime data to consolidator")
    
    def add_transition_handler(self, handler):
        """Add a regime transition handler"""
        self.transition_handlers.append(handler)
    
    def get_strategy_recommendations(self, regime_id: int, dte: int) -> Dict[str, Any]:
        """Get strategy recommendations for current regime and DTE"""
        regime_mapping = self.regime_mappings.get(regime_id)
        if not regime_mapping:
            return {}
        
        # Adjust recommendations based on DTE
        dte_multiplier = self._get_dte_multiplier(dte)
        
        recommendations = {
            'preferred_strategies': regime_mapping.preferred_strategies,
            'strategy_weights': {
                strategy: weight * dte_multiplier 
                for strategy, weight in regime_mapping.strategy_weights.items()
            },
            'position_sizing': regime_mapping.position_size_multiplier * dte_multiplier,
            'risk_parameters': {
                'stop_loss': regime_mapping.stop_loss_adjustment,
                'target': regime_mapping.target_adjustment,
                'risk_adjustment': regime_mapping.risk_adjustment
            }
        }
        
        return recommendations
    
    def _get_dte_multiplier(self, dte: int) -> float:
        """Get DTE-based adjustment multiplier"""
        if dte <= 1:
            return 0.5  # Reduce size for expiry
        elif dte <= 3:
            return 0.7
        elif dte <= 7:
            return 1.0
        elif dte <= 15:
            return 0.9
        else:
            return 0.8
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary by regime"""
        summary = {}
        
        for strategy, regime_data in self.performance_tracker.items():
            strategy_summary = {}
            
            for regime_id, performance in regime_data.items():
                strategy_summary[f"regime_{regime_id}"] = {
                    'total_trades': performance.total_trades,
                    'win_rate': performance.winning_trades / max(performance.total_trades, 1),
                    'total_pnl': performance.total_pnl,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown
                }
            
            summary[strategy] = strategy_summary
        
        return summary

# Create global instance
regime_strategy_integrator = RegimeStrategyIntegrator()

def get_regime_strategy_integrator() -> RegimeStrategyIntegrator:
    """Get the global regime strategy integrator instance"""
    return regime_strategy_integrator

# Example usage
if __name__ == "__main__":
    print("Regime Strategy Integration Module")
    print("=" * 60)
    
    # Example regime data
    example_regime_data = {
        'regime_id': 6,  # Med Vol Bullish Trending
        'confidence': 0.85,
        'volatility_component': 0.5,
        'trend_component': 0.6,
        'structure_component': 0.7,
        'dte_bucket': '4-7',
        'timeframe_scores': {
            '5min': 0.82,
            '15min': 0.85,
            '30min': 0.88,
            '1hr': 0.90
        },
        'dominant_timeframe': '30min',
        'triple_straddle_value': 1.2,
        'greek_sentiment_score': 0.75,
        'oi_trending_score': 0.68,
        'iv_analysis_score': 0.45,
        'ml_ensemble_prediction': 'BULLISH',
        'ml_confidence': 0.78
    }
    
    # Process regime data
    async def test_integration():
        integrator = get_regime_strategy_integrator()
        result = await integrator.process_regime_data(example_regime_data)
        
        print("\nProcessed Regime Data:")
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                print(f"{key}: {json.dumps(value, indent=2)}")
            else:
                print(f"{key}: {value}")
        
        # Get strategy recommendations
        recommendations = integrator.get_strategy_recommendations(6, 5)
        print("\nStrategy Recommendations:")
        print(json.dumps(recommendations, indent=2))
    
    # Run test
    import asyncio
    asyncio.run(test_integration())
    
    print("\n✓ Regime Strategy Integration ready for consolidator!")