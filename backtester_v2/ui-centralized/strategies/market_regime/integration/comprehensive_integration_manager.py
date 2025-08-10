"""
Comprehensive Integration Manager
=================================

Central integration manager for all 9 market regime components
using the refactored module structure.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import component registry
from ..base.component_registry import get_component_registry

# Import all components from the refactored structure
from ..indicators.greek_sentiment.greek_sentiment_analyzer import GreekSentimentAnalyzer
from ..indicators.oi_pa_analysis.oi_pa_analyzer import OIPriceActionAnalyzer
from ..indicators.straddle_analysis.core.straddle_engine import StraddleAnalysisEngine
from ..indicators.iv_analytics.iv_analytics_analyzer import IVAnalyticsAnalyzer
from ..indicators.technical_indicators.technical_indicators_analyzer import TechnicalIndicatorsAnalyzer
from ..indicators.market_breadth.market_breadth_analyzer import MarketBreadthAnalyzer
from ..indicators.volume_profile.volume_profile_analyzer import VolumeProfileAnalyzer
from ..indicators.correlation_analysis.correlation_analyzer import CorrelationAnalyzer
from ..base.multi_timeframe_analyzer import MultiTimeframeAnalyzer

# Import regime classification
from ..base.regime_classification import RegimeClassification
from ..base.regime_name_mapper import RegimeNameMapper

# Import output generator
from ..base.output.enhanced_csv_generator import EnhancedCSVGenerator

# Import HeavyDB integration
from ..data.heavydb_data_provider import HeavyDBDataProvider

logger = logging.getLogger(__name__)


class ComprehensiveIntegrationManager:
    """
    Manages integration of all 9 market regime components
    
    Components:
    1. GreekSentiment (20% weight)
    2. TrendingOIPA (15% weight)
    3. StraddleAnalysis (15% weight)
    4. MultiTimeframe (15% weight)
    5. IVSurface (10% weight)
    6. ATRIndicators (10% weight)
    7. MarketBreadth (10% weight)
    8. VolumeProfile (8% weight)
    9. Correlation (7% weight)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integration manager"""
        self.config = config or {}
        
        # Get component registry
        self.registry = get_component_registry()
        
        # Component instances
        self.components = {}
        
        # Configuration
        self.excel_config = None
        self.excel_config_path = self.config.get('excel_config_path')
        
        # HeavyDB provider
        self.data_provider = None
        
        # Output generator
        self.csv_generator = None
        
        # Regime mapper
        self.regime_mapper = RegimeNameMapper()
        
        # State tracking
        self.is_initialized = False
        self.last_analysis_time = None
        self.analysis_count = 0
        
        logger.info("ComprehensiveIntegrationManager created")
    
    def initialize(self, excel_config_path: Optional[str] = None) -> bool:
        """
        Initialize all components
        
        Args:
            excel_config_path: Path to Excel configuration file
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("Initializing Comprehensive Integration Manager...")
            
            # Load Excel configuration if provided
            if excel_config_path:
                self.excel_config_path = excel_config_path
                if not self._load_excel_configuration():
                    logger.error("Failed to load Excel configuration")
                    return False
            
            # Initialize HeavyDB data provider
            if not self._initialize_data_provider():
                logger.error("Failed to initialize HeavyDB data provider")
                return False
            
            # Register all components
            registration_results = self.registry.register_all_components(self.excel_config)
            
            # Check registration results
            failed_components = [name for name, success in registration_results.items() if not success]
            if failed_components:
                logger.warning(f"Failed to register components: {failed_components}")
            
            # Instantiate all registered components
            for component_name in self.registry.COMPONENT_DEFINITIONS:
                instance = self.registry.instantiate_component(component_name, self.excel_config)
                if instance:
                    self.components[component_name] = instance
                    logger.info(f"✅ Instantiated {component_name}")
                else:
                    logger.warning(f"⚠️ Failed to instantiate {component_name}")
            
            # Initialize CSV generator
            self.csv_generator = EnhancedCSVGenerator(config=self.excel_config)
            
            self.is_initialized = True
            logger.info(f"Initialization complete: {len(self.components)}/9 components active")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return False
    
    def _load_excel_configuration(self) -> bool:
        """Load configuration from Excel file"""
        try:
            if not self.excel_config_path or not Path(self.excel_config_path).exists():
                logger.error(f"Excel config file not found: {self.excel_config_path}")
                return False
            
            # Import Excel parser
            from ..excel_config_parser import ExcelConfigParser
            
            parser = ExcelConfigParser(self.excel_config_path)
            self.excel_config = parser.parse_all_sheets()
            
            logger.info(f"Loaded Excel configuration with {len(self.excel_config)} sheets")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Excel configuration: {e}")
            return False
    
    def _initialize_data_provider(self) -> bool:
        """Initialize HeavyDB data provider"""
        try:
            # Create HeavyDB configuration
            heavydb_config = {
                'host': 'localhost',
                'port': 6274,
                'database': 'heavyai',
                'user': 'admin',
                'password': 'HyperInteractive',
                'protocol': 'binary'
            }
            
            # Add any Excel overrides
            if self.excel_config and 'DatabaseConfig' in self.excel_config:
                db_config = self.excel_config['DatabaseConfig']
                if 'heavydb_host' in db_config:
                    heavydb_config['host'] = db_config['heavydb_host']
                if 'heavydb_port' in db_config:
                    heavydb_config['port'] = int(db_config['heavydb_port'])
            
            # Create data provider
            self.data_provider = HeavyDBDataProvider(heavydb_config)
            
            # Test connection
            if self.data_provider.test_connection():
                logger.info("✅ HeavyDB connection established")
                return True
            else:
                logger.error("❌ HeavyDB connection failed")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing data provider: {e}")
            return False
    
    def analyze_market_regime(self, 
                            symbol: str = "NIFTY",
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Perform comprehensive market regime analysis
        
        Args:
            symbol: Trading symbol
            start_time: Analysis start time
            end_time: Analysis end time
            
        Returns:
            Dict with analysis results
        """
        if not self.is_initialized:
            logger.error("Integration manager not initialized")
            return {}
        
        try:
            # Use current time if not specified
            if not end_time:
                end_time = datetime.now()
            if not start_time:
                start_time = end_time - timedelta(minutes=30)
            
            logger.info(f"Analyzing {symbol} from {start_time} to {end_time}")
            
            # Fetch market data from HeavyDB
            market_data = self._fetch_market_data(symbol, start_time, end_time)
            
            if market_data.empty:
                logger.error("No market data available")
                return {}
            
            # Run analysis with all components
            component_results = self.registry.analyze_with_all_components({
                'market_data': market_data,
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            })
            
            # Calculate regime classification
            regime_id = self._calculate_regime_id(component_results['combined_score'])
            regime_name = self.regime_mapper.get_regime_name(regime_id)
            
            # Prepare results
            results = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'regime_id': regime_id,
                'regime_name': regime_name,
                'combined_score': component_results['combined_score'],
                'component_scores': component_results['component_scores'],
                'weighted_scores': component_results['weighted_scores'],
                'active_components': component_results['active_components'],
                'data_points': len(market_data),
                'analysis_time': end_time,
                'processing_time': time.time()
            }
            
            # Update tracking
            self.last_analysis_time = datetime.now()
            self.analysis_count += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            return {}
    
    def _fetch_market_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch market data from HeavyDB"""
        try:
            if not self.data_provider:
                logger.error("Data provider not initialized")
                return pd.DataFrame()
            
            # Query for 1-minute data
            query = f"""
            SELECT 
                datetime_,
                symbol,
                close as underlying_close,
                open,
                high,
                low,
                volume
            FROM nifty_option_chain
            WHERE symbol = '{symbol}'
                AND datetime_ >= '{start_time}'
                AND datetime_ <= '{end_time}'
                AND close > 0
            ORDER BY datetime_
            """
            
            data = self.data_provider.execute_query(query)
            
            if data.empty:
                logger.warning(f"No data found for {symbol} in specified time range")
            else:
                logger.info(f"Fetched {len(data)} data points from HeavyDB")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def _calculate_regime_id(self, combined_score: float) -> int:
        """
        Calculate regime ID from combined score
        
        Maps the continuous score (-1 to +1) to 35 discrete regimes
        """
        # Simple linear mapping for now
        # -1.0 to +1.0 -> 0 to 34
        normalized_score = (combined_score + 1.0) / 2.0  # 0 to 1
        regime_id = int(normalized_score * 34.99)  # 0 to 34
        regime_id = max(0, min(34, regime_id))  # Ensure bounds
        
        return regime_id
    
    def generate_csv_output(self, 
                          results: Dict[str, Any],
                          output_path: Optional[str] = None) -> str:
        """
        Generate CSV output file
        
        Args:
            results: Analysis results
            output_path: Output file path
            
        Returns:
            str: Path to generated CSV file
        """
        try:
            if not self.csv_generator:
                logger.error("CSV generator not initialized")
                return ""
            
            # Prepare data for CSV generation
            csv_data = {
                'regime_data': pd.DataFrame([{
                    'datetime': results['timestamp'],
                    'regime_id': results['regime_id'],
                    'regime_name': results['regime_name'],
                    'combined_score': results['combined_score']
                }]),
                'component_scores': results['component_scores'],
                'excel_config': self.excel_config
            }
            
            # Generate CSV
            csv_path = self.csv_generator.generate_csv(
                csv_data,
                output_path=output_path
            )
            
            logger.info(f"Generated CSV output: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Error generating CSV output: {e}")
            return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration manager status"""
        status = {
            'is_initialized': self.is_initialized,
            'active_components': list(self.components.keys()),
            'component_count': len(self.components),
            'excel_config_loaded': self.excel_config is not None,
            'heavydb_connected': self.data_provider is not None,
            'analysis_count': self.analysis_count,
            'last_analysis_time': self.last_analysis_time
        }
        
        # Add component status
        if self.registry:
            status['component_status'] = self.registry.get_component_status()
        
        return status
    
    def shutdown(self):
        """Shutdown integration manager and cleanup resources"""
        logger.info("Shutting down integration manager...")
        
        # Cleanup components
        for name, component in self.components.items():
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")
        
        # Cleanup data provider
        if self.data_provider:
            self.data_provider.close()
        
        self.is_initialized = False
        logger.info("Integration manager shutdown complete")


# Factory function
def create_integration_manager(config: Optional[Dict[str, Any]] = None) -> ComprehensiveIntegrationManager:
    """Create a comprehensive integration manager instance"""
    return ComprehensiveIntegrationManager(config)