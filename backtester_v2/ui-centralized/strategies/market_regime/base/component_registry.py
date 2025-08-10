"""
Component Registry - Central Registry for All 9 Market Regime Components
======================================================================

Provides centralized registration and management of all active components
in the market regime detection system.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

from typing import Dict, Any, Optional, Type, List
import logging
from importlib import import_module

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for all market regime components
    
    Manages all 9 active components with their weights and configurations
    as defined in the Excel DynamicWeightageConfig sheet.
    """
    
    # Component definitions with paths and weights
    COMPONENT_DEFINITIONS = {
        'GreekSentiment': {
            'module': 'indicators.greek_sentiment.greek_sentiment_analyzer',
            'class': 'GreekSentimentAnalyzer',
            'base_weight': 0.20,
            'description': 'Options Greeks-based sentiment analysis',
            'requires_data': ['options_chain'],
            'update_frequency': '1min'
        },
        'TrendingOIPA': {
            'module': 'indicators.oi_pa_analysis.oi_pa_analyzer',
            'class': 'OIPriceActionAnalyzer',
            'base_weight': 0.15,
            'description': 'Open Interest with Price Action analysis',
            'requires_data': ['options_chain', 'price_data'],
            'update_frequency': '1min'
        },
        'StraddleAnalysis': {
            'module': 'indicators.straddle_analysis.core.straddle_engine',
            'class': 'StraddleAnalysisEngine',
            'base_weight': 0.15,
            'description': 'Triple straddle analysis (ATM, ITM1, OTM1)',
            'requires_data': ['options_chain'],
            'update_frequency': '1min'
        },
        'IVSurface': {
            'module': 'indicators.iv_analytics.iv_analytics_analyzer',
            'class': 'IVAnalyticsAnalyzer',
            'base_weight': 0.10,
            'description': 'Implied Volatility surface analysis',
            'requires_data': ['options_chain', 'iv_data'],
            'update_frequency': '5min'
        },
        'ATRIndicators': {
            'module': 'indicators.technical_indicators.technical_indicators_analyzer',
            'class': 'TechnicalIndicatorsAnalyzer',
            'base_weight': 0.10,
            'description': 'ATR and technical indicators suite',
            'requires_data': ['price_data'],
            'update_frequency': '1min'
        },
        'MultiTimeframe': {
            'module': 'base.multi_timeframe_analyzer',
            'class': 'MultiTimeframeAnalyzer',
            'base_weight': 0.15,
            'description': 'Multi-timeframe analysis and fusion',
            'requires_data': ['price_data'],
            'update_frequency': '1min'
        },
        'VolumeProfile': {
            'module': 'indicators.volume_profile.volume_profile_analyzer',
            'class': 'VolumeProfileAnalyzer',
            'base_weight': 0.08,
            'description': 'Volume distribution and profile analysis',
            'requires_data': ['price_data', 'volume_data'],
            'update_frequency': '5min'
        },
        'Correlation': {
            'module': 'indicators.correlation_analysis.correlation_analyzer',
            'class': 'CorrelationAnalyzer',
            'base_weight': 0.07,
            'description': 'Cross-market correlation analysis',
            'requires_data': ['multi_asset_data'],
            'update_frequency': '5min'
        },
        'MarketBreadth': {
            'module': 'indicators.market_breadth.market_breadth_analyzer',
            'class': 'MarketBreadthAnalyzer',
            'base_weight': 0.10,  # Inferred from remaining weight
            'description': 'Market breadth and internal indicators',
            'requires_data': ['market_data', 'advance_decline'],
            'update_frequency': '1min'
        }
    }
    
    def __init__(self):
        """Initialize the component registry"""
        self.registered_components = {}
        self.component_instances = {}
        self.component_weights = {}
        self.initialization_errors = {}
        
        # Verify total weight equals 1.0
        total_weight = sum(comp['base_weight'] for comp in self.COMPONENT_DEFINITIONS.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Component weights sum to {total_weight}, not 1.0")
    
    def register_all_components(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Register all 9 components
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            Dict mapping component names to registration success
        """
        results = {}
        
        for component_name, component_def in self.COMPONENT_DEFINITIONS.items():
            try:
                success = self.register_component(
                    name=component_name,
                    module_path=component_def['module'],
                    class_name=component_def['class'],
                    weight=component_def['base_weight'],
                    config=config
                )
                results[component_name] = success
                
            except Exception as e:
                logger.error(f"Failed to register {component_name}: {e}")
                results[component_name] = False
                self.initialization_errors[component_name] = str(e)
        
        logger.info(f"Registered {sum(results.values())}/{len(results)} components successfully")
        return results
    
    def register_component(self,
                         name: str,
                         module_path: str,
                         class_name: str,
                         weight: float,
                         config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a single component
        
        Args:
            name: Component name
            module_path: Python module path
            class_name: Class name to instantiate
            weight: Component weight
            config: Optional configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Adjust module path to be relative to market_regime
            if not module_path.startswith('.'):
                module_path = f".{module_path}"
            
            # Import module
            module = import_module(module_path, package='strategies.market_regime')
            
            # Get class
            component_class = getattr(module, class_name)
            
            # Store registration
            self.registered_components[name] = {
                'module': module,
                'class': component_class,
                'module_path': module_path,
                'class_name': class_name,
                'weight': weight
            }
            
            self.component_weights[name] = weight
            
            logger.info(f"Registered component: {name} (weight: {weight})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component {name}: {e}")
            return False
    
    def instantiate_component(self, 
                            name: str,
                            config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Create an instance of a registered component
        
        Args:
            name: Component name
            config: Configuration for the component
            
        Returns:
            Component instance or None
        """
        if name not in self.registered_components:
            logger.error(f"Component {name} not registered")
            return None
        
        try:
            component_info = self.registered_components[name]
            component_class = component_info['class']
            
            # Create instance
            instance = component_class(config=config)
            
            # Cache instance
            self.component_instances[name] = instance
            
            logger.info(f"Instantiated component: {name}")
            return instance
            
        except Exception as e:
            logger.error(f"Error instantiating component {name}: {e}")
            self.initialization_errors[name] = str(e)
            return None
    
    def get_component_instance(self, name: str) -> Optional[Any]:
        """Get cached component instance"""
        return self.component_instances.get(name)
    
    def get_all_instances(self) -> Dict[str, Any]:
        """Get all instantiated components"""
        return self.component_instances.copy()
    
    def get_component_weights(self) -> Dict[str, float]:
        """Get current component weights"""
        return self.component_weights.copy()
    
    def update_component_weight(self, name: str, new_weight: float):
        """
        Update component weight dynamically
        
        Args:
            name: Component name
            new_weight: New weight value
        """
        if name in self.component_weights:
            old_weight = self.component_weights[name]
            self.component_weights[name] = new_weight
            logger.info(f"Updated {name} weight: {old_weight} -> {new_weight}")
        else:
            logger.warning(f"Component {name} not found for weight update")
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all components"""
        status = {}
        
        for name in self.COMPONENT_DEFINITIONS:
            status[name] = {
                'registered': name in self.registered_components,
                'instantiated': name in self.component_instances,
                'weight': self.component_weights.get(name, 0.0),
                'error': self.initialization_errors.get(name, None)
            }
        
        return status
    
    def get_active_components(self) -> List[str]:
        """Get list of active (instantiated) components"""
        return list(self.component_instances.keys())
    
    def analyze_with_all_components(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run analysis with all active components
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Combined analysis results
        """
        results = {
            'component_scores': {},
            'weighted_scores': {},
            'combined_score': 0.0,
            'active_components': []
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, instance in self.component_instances.items():
            try:
                # Run component analysis
                component_result = instance.analyze(market_data)
                
                # Extract score (components may use different keys)
                score_keys = [
                    f'{name.lower()}_score',
                    'regime_signal',
                    'signal',
                    'score'
                ]
                
                score = None
                for key in score_keys:
                    if key in component_result:
                        score = component_result[key]
                        break
                
                if score is not None:
                    weight = self.component_weights.get(name, 0.0)
                    
                    results['component_scores'][name] = score
                    results['weighted_scores'][name] = score * weight
                    results['active_components'].append(name)
                    
                    weighted_sum += score * weight
                    total_weight += weight
                    
                    logger.debug(f"{name}: score={score:.3f}, weight={weight:.3f}")
                
            except Exception as e:
                logger.error(f"Error running {name} analysis: {e}")
        
        # Calculate combined score
        if total_weight > 0:
            results['combined_score'] = weighted_sum / total_weight
        
        results['total_weight'] = total_weight
        
        return results


# Singleton instance
_component_registry = None

def get_component_registry() -> ComponentRegistry:
    """Get singleton instance of ComponentRegistry"""
    global _component_registry
    if _component_registry is None:
        _component_registry = ComponentRegistry()
    return _component_registry