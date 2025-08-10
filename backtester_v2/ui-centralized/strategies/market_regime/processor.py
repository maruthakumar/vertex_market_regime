"""
Market Regime Processor

This module provides the main processing interface for market regime
detection, integrating all components and following backtester_v2 patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import logging
import os

# Base processor class for market regime
class BaseProcessor:
    """Base class for processors"""
    def __init__(self, db_connection):
        self.db_connection = db_connection
# Import components with fallback handling
try:
    # Try relative imports first (when used as a package)
    from .models import RegimeConfig, RegimeClassification, RegimeSummary
    from .parser import RegimeConfigParser
    from .strategy import MarketRegimeStrategy
    from .calculator import RegimeCalculator
    from .classifier import RegimeClassifier
    from .performance import PerformanceTracker
except ImportError:
    try:
        # Fallback to absolute imports (when used standalone)
        from models import RegimeConfig, RegimeClassification, RegimeSummary
        from parser import RegimeConfigParser
        from strategy import MarketRegimeStrategy
        from calculator import RegimeCalculator
        from classifier import RegimeClassifier
        from performance import PerformanceTracker
    except ImportError:
        # Final fallback - create minimal classes for standalone usage
        class RegimeConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class RegimeClassification:
            def __init__(self, regime_type, confidence, timestamp):
                self.regime_type = regime_type
                self.confidence = confidence
                self.timestamp = timestamp
        
        class RegimeSummary:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class RegimeConfigParser:
            def parse(self, config_data):
                return RegimeConfig(**config_data)
        
        class MarketRegimeStrategy:
            def __init__(self, strategy_config=None):
                self.strategy_config = strategy_config
        
        class RegimeCalculator:
            def __init__(self, config=None):
                self.config = config
        
        class RegimeClassifier:
            def __init__(self, config=None):
                self.config = config
        
        class PerformanceTracker:
            def __init__(self, config=None):
                self.config = config

logger = logging.getLogger(__name__)

class RegimeProcessor(BaseProcessor):
    """
    Main processor for market regime detection
    
    This class orchestrates the entire regime detection process,
    from configuration parsing to result generation.
    """
    
    def __init__(self, db_connection=None, config_path: Optional[str] = None, 
                 regime_config: Optional[RegimeConfig] = None):
        """
        Initialize the regime processor
        
        Args:
            db_connection: Database connection for data access
            config_path (str, optional): Path to Excel configuration file
            regime_config (RegimeConfig, optional): Pre-built configuration
        """
        if db_connection:
            super().__init__(db_connection)
        else:
            self.db_connection = None
        
        # Initialize configuration
        if regime_config:
            self.config = regime_config
        elif config_path:
            self.config = self._load_config_from_file(config_path)
        else:
            self.config = self._create_default_config()
        
        # Initialize components
        self.parser = RegimeConfigParser()
        self.calculator = RegimeCalculator(self.config)
        self.classifier = RegimeClassifier(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        
        # Initialize strategy
        strategy_config = {
            'strategy_name': self.config.strategy_name,
            'symbol': self.config.symbol,
            'regime_config': self.config
        }
        self.strategy = MarketRegimeStrategy(strategy_config)
        self.strategy.db_connection = db_connection
        
        # Processing state
        self.last_processed_date = None
        self.processing_cache = {}
        
        logger.info(f"RegimeProcessor initialized for {self.config.symbol}")
    
    def _load_config_from_file(self, config_path: str) -> RegimeConfig:
        """Load configuration from Excel file"""
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}, using default config")
                return self._create_default_config()
            
            config = self.parser.parse(config_path)
            
            # Validate configuration
            validation_results = self.parser.validate_config(config)
            
            if validation_results['errors']:
                logger.error(f"Configuration errors: {validation_results['errors']}")
                raise ValueError(f"Invalid configuration: {validation_results['errors']}")
            
            if validation_results['warnings']:
                logger.warning(f"Configuration warnings: {validation_results['warnings']}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> RegimeConfig:
        """Create default configuration"""
        try:
            from .models import IndicatorConfig, IndicatorCategory, TimeframeConfig
        except ImportError:
            try:
                from models import IndicatorConfig, IndicatorCategory, TimeframeConfig
            except ImportError:
                # Create minimal fallback classes
                from enum import Enum
                class IndicatorCategory(Enum):
                    PRICE_TREND = "PRICE_TREND"
                    GREEK_SENTIMENT = "GREEK_SENTIMENT"
                    IV_ANALYSIS = "IV_ANALYSIS"
                    PREMIUM_ANALYSIS = "PREMIUM_ANALYSIS"
                
                class IndicatorConfig:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                
                class TimeframeConfig:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
        
        # Default indicators
        default_indicators = [
            IndicatorConfig(
                id='ema_trend',
                name='EMA Trend',
                category=IndicatorCategory.PRICE_TREND,
                indicator_type='ema',
                base_weight=0.25,
                parameters={'periods': [5, 10, 20]}
            ),
            IndicatorConfig(
                id='vwap_trend',
                name='VWAP Trend',
                category=IndicatorCategory.PRICE_TREND,
                indicator_type='vwap',
                base_weight=0.20,
                parameters={'bands': 2}
            ),
            IndicatorConfig(
                id='greek_sentiment',
                name='Greek Sentiment',
                category=IndicatorCategory.GREEK_SENTIMENT,
                indicator_type='greek',
                base_weight=0.20
            ),
            IndicatorConfig(
                id='iv_analysis',
                name='IV Analysis',
                category=IndicatorCategory.IV_ANALYSIS,
                indicator_type='iv_skew',
                base_weight=0.15,
                parameters={'lookback': 30}
            ),
            IndicatorConfig(
                id='premium_analysis',
                name='Premium Analysis',
                category=IndicatorCategory.PREMIUM_ANALYSIS,
                indicator_type='premium',
                base_weight=0.20,
                parameters={'strikes': 3}
            )
        ]
        
        return RegimeConfig(
            strategy_name='MarketRegime',
            symbol='NIFTY',
            indicators=default_indicators
        )
    
    def process_regime_analysis(self, start_date: str, end_date: str, **kwargs) -> Dict[str, Any]:
        """
        Process complete regime analysis for date range
        
        Args:
            start_date (str): Start date for analysis
            end_date (str): End date for analysis
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        try:
            logger.info(f"Processing regime analysis from {start_date} to {end_date}")
            
            # Execute strategy
            regime_results = self.strategy.execute(start_date, end_date, **kwargs)
            
            if regime_results.empty:
                logger.warning("No regime results generated")
                return self._create_empty_results()
            
            # Generate summary
            summary = self._generate_regime_summary(regime_results)
            
            # Generate performance metrics
            performance_metrics = self.performance_tracker.get_performance_summary()
            
            # Generate alerts if any
            alerts = self._generate_regime_alerts(regime_results)
            
            # Create comprehensive results
            results = {
                'processing_summary': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbol': self.config.symbol,
                    'total_classifications': len(regime_results),
                    'processing_time': datetime.now().isoformat(),
                    'config_used': self.config.model_dump()
                },
                'regime_classifications': regime_results.to_dict('records'),
                'regime_summary': summary,
                'performance_metrics': performance_metrics,
                'alerts': alerts,
                'golden_file_data': self._generate_golden_file_data(regime_results, summary)
            }
            
            # Update processing state
            self.last_processed_date = datetime.now()
            
            logger.info(f"Regime analysis completed: {len(regime_results)} classifications")
            return results
            
        except Exception as e:
            logger.error(f"Error processing regime analysis: {e}")
            return self._create_empty_results()
    
    def _generate_regime_summary(self, regime_results: pd.DataFrame) -> Dict[str, Any]:
        """Generate regime analysis summary"""
        try:
            if regime_results.empty:
                return {}
            
            # Current regime info
            latest_result = regime_results.iloc[-1]
            current_regime = latest_result['regime_type']
            current_confidence = latest_result['confidence']
            
            # Regime distribution
            regime_counts = regime_results['regime_type'].value_counts().to_dict()
            
            # Average confidence by regime
            avg_confidence_by_regime = regime_results.groupby('regime_type')['confidence'].mean().to_dict()
            
            # Regime transitions
            transitions = self._count_regime_transitions(regime_results)
            
            # Dominant indicators
            indicator_columns = [col for col in regime_results.columns if col.endswith('_signal')]
            dominant_indicators = []
            
            for col in indicator_columns:
                indicator_id = col.replace('_signal', '')
                avg_signal = abs(regime_results[col]).mean()
                dominant_indicators.append((indicator_id, avg_signal))
            
            dominant_indicators.sort(key=lambda x: x[1], reverse=True)
            dominant_indicators = [ind[0] for ind in dominant_indicators[:5]]
            
            summary = RegimeSummary(
                symbol=self.config.symbol,
                analysis_date=date.today(),
                current_regime=current_regime,
                regime_confidence=current_confidence,
                regime_duration_minutes=self._calculate_current_regime_duration(regime_results),
                daily_transitions=transitions,
                dominant_indicators=dominant_indicators,
                performance_summary={
                    'regime_distribution': regime_counts,
                    'avg_confidence_by_regime': avg_confidence_by_regime,
                    'total_data_points': len(regime_results)
                }
            )
            
            return summary.model_dump()
            
        except Exception as e:
            logger.error(f"Error generating regime summary: {e}")
            return {}
    
    def _count_regime_transitions(self, regime_results: pd.DataFrame) -> int:
        """Count regime transitions in the data"""
        try:
            if len(regime_results) < 2:
                return 0
            
            transitions = 0
            prev_regime = regime_results.iloc[0]['regime_type']
            
            for _, row in regime_results.iloc[1:].iterrows():
                current_regime = row['regime_type']
                if current_regime != prev_regime:
                    transitions += 1
                prev_regime = current_regime
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error counting transitions: {e}")
            return 0
    
    def _calculate_current_regime_duration(self, regime_results: pd.DataFrame) -> int:
        """Calculate duration of current regime in minutes"""
        try:
            if len(regime_results) < 2:
                return 0
            
            current_regime = regime_results.iloc[-1]['regime_type']
            duration = 0
            
            # Count backwards until regime changes
            for i in range(len(regime_results) - 1, -1, -1):
                if regime_results.iloc[i]['regime_type'] == current_regime:
                    duration += 1
                else:
                    break
            
            return duration  # Assuming 1-minute intervals
            
        except Exception as e:
            logger.error(f"Error calculating regime duration: {e}")
            return 0
    
    def _generate_regime_alerts(self, regime_results: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate regime-based alerts"""
        try:
            alerts = []
            
            if regime_results.empty:
                return alerts
            
            # Check for recent regime changes
            if len(regime_results) >= 2:
                current_regime = regime_results.iloc[-1]['regime_type']
                previous_regime = regime_results.iloc[-2]['regime_type']
                
                if current_regime != previous_regime:
                    alerts.append({
                        'timestamp': regime_results.index[-1],
                        'type': 'REGIME_CHANGE',
                        'message': f'Regime changed from {previous_regime} to {current_regime}',
                        'priority': 'HIGH',
                        'confidence': regime_results.iloc[-1]['confidence']
                    })
            
            # Check for low confidence
            latest_confidence = regime_results.iloc[-1]['confidence']
            if latest_confidence < self.config.confidence_threshold:
                alerts.append({
                    'timestamp': regime_results.index[-1],
                    'type': 'LOW_CONFIDENCE',
                    'message': f'Low confidence regime classification: {latest_confidence:.2f}',
                    'priority': 'MEDIUM',
                    'confidence': latest_confidence
                })
            
            # Check for high volatility regime
            current_regime = regime_results.iloc[-1]['regime_type']
            if current_regime == 'HIGH_VOLATILITY':
                alerts.append({
                    'timestamp': regime_results.index[-1],
                    'type': 'HIGH_VOLATILITY',
                    'message': 'High volatility regime detected',
                    'priority': 'HIGH',
                    'confidence': latest_confidence
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    def _generate_golden_file_data(self, regime_results: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate golden file format data"""
        try:
            golden_data = {}
            
            # Portfolio Parameter sheet
            portfolio_data = {
                'Parameter': ['Strategy', 'Symbol', 'Analysis_Date', 'Total_Classifications', 'Current_Regime'],
                'Value': [
                    self.config.strategy_name,
                    self.config.symbol,
                    date.today().isoformat(),
                    len(regime_results),
                    summary.get('current_regime', 'UNKNOWN')
                ]
            }
            golden_data['PortfolioParameter'] = pd.DataFrame(portfolio_data)
            
            # General Parameter sheet
            general_data = {
                'Parameter': ['Confidence_Threshold', 'Regime_Smoothing', 'Update_Frequency', 'Lookback_Days'],
                'Value': [
                    self.config.confidence_threshold,
                    self.config.regime_smoothing,
                    self.config.update_frequency,
                    self.config.lookback_days
                ]
            }
            golden_data['GeneralParameter'] = pd.DataFrame(general_data)
            
            # Indicator Parameter sheet
            indicator_data = []
            for indicator in self.config.indicators:
                indicator_data.append({
                    'Indicator_ID': indicator.id,
                    'Indicator_Name': indicator.name,
                    'Category': indicator.category.value,
                    'Base_Weight': indicator.base_weight,
                    'Enabled': indicator.enabled
                })
            golden_data['IndicatorParameter'] = pd.DataFrame(indicator_data)
            
            # Regime Results sheet (sample of recent results)
            if not regime_results.empty:
                # Take last 100 results for golden file
                sample_results = regime_results.tail(100).copy()
                sample_results.reset_index(inplace=True)
                golden_data['RegimeResults'] = sample_results
            
            return golden_data
            
        except Exception as e:
            logger.error(f"Error generating golden file data: {e}")
            return {}
    
    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty results structure"""
        return {
            'processing_summary': {
                'total_classifications': 0,
                'processing_time': datetime.now().isoformat(),
                'status': 'NO_DATA'
            },
            'regime_classifications': [],
            'regime_summary': {},
            'performance_metrics': {},
            'alerts': [],
            'golden_file_data': {}
        }
    
    def create_config_template(self, output_path: str):
        """Create configuration template file"""
        try:
            self.parser.create_template(output_path)
            logger.info(f"Configuration template created: {output_path}")
        except Exception as e:
            logger.error(f"Error creating config template: {e}")
            raise
    
    def get_current_regime(self) -> Optional[Dict[str, Any]]:
        """Get current market regime"""
        return self.strategy.get_current_regime()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return self.performance_tracker.get_performance_summary()
