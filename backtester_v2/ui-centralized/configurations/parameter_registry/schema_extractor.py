"""
Schema Extractor

Extracts parameter definitions from existing configuration classes
and converts them to the unified parameter registry format.
"""

import logging
import inspect
import json
from typing import Dict, List, Any, Optional, Type, get_type_hints
from pathlib import Path

from .models import (
    ParameterDefinition, 
    ParameterCategory, 
    StrategyMetadata,
    ParameterType,
    WidgetType,
    ValidationRule,
    UIHints
)
from .registry import ParameterRegistry
from ..core.base_config import BaseConfiguration

logger = logging.getLogger(__name__)

class SchemaExtractor:
    """
    Extract parameter definitions from configuration classes
    
    Analyzes existing configuration classes and their schemas to generate
    parameter registry entries with UI hints and validation rules.
    """
    
    def __init__(self, registry: Optional[ParameterRegistry] = None):
        """Initialize schema extractor"""
        self.registry = registry or ParameterRegistry()
        
        # Mapping of JSON schema types to ParameterType
        self.type_mapping = {
            "string": ParameterType.STRING,
            "number": ParameterType.NUMBER,
            "integer": ParameterType.INTEGER,
            "boolean": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "object": ParameterType.OBJECT
        }
        
        # Mapping of parameter names to widget types
        self.widget_mapping = {
            "capital": WidgetType.CURRENCY_INPUT,
            "amount": WidgetType.CURRENCY_INPUT,
            "price": WidgetType.CURRENCY_INPUT,
            "percentage": WidgetType.PERCENTAGE_INPUT,
            "percent": WidgetType.PERCENTAGE_INPUT,
            "time": WidgetType.TIME_PICKER,
            "date": WidgetType.DATE_PICKER,
            "enabled": WidgetType.TOGGLE,
            "count": WidgetType.NUMBER_INPUT,
            "limit": WidgetType.NUMBER_INPUT,
            "threshold": WidgetType.SLIDER,
            "weight": WidgetType.SLIDER,
            "ratio": WidgetType.SLIDER
        }
        
        # Strategy metadata
        self.strategy_metadata = {
            "tbs": {
                "display_name": "Time-Based Strategy",
                "description": "Simple time-based entry and exit strategy for options trading",
                "complexity_level": "basic",
                "excel_template_sheets": ["Portfolio", "Strategy", "Enhancement", "Risk"]
            },
            "tv": {
                "display_name": "TradingView Strategy", 
                "description": "Strategy based on TradingView alerts and signals",
                "complexity_level": "intermediate",
                "excel_template_sheets": ["Config", "Alerts", "Risk"]
            },
            "orb": {
                "display_name": "Opening Range Breakout",
                "description": "Breakout strategy based on opening range analysis",
                "complexity_level": "intermediate", 
                "excel_template_sheets": ["Setup", "Rules", "Risk"]
            },
            "oi": {
                "display_name": "Open Interest Strategy",
                "description": "Strategy based on open interest and PCR analysis",
                "complexity_level": "intermediate",
                "excel_template_sheets": ["Config", "Rules", "Analysis"]
            },
            "ml": {
                "display_name": "Machine Learning Strategy",
                "description": "ML-based strategy with 853+ configurable parameters",
                "complexity_level": "advanced",
                "excel_template_sheets": ["Models", "Features", "Training", "Risk"]
            },
            "pos": {
                "display_name": "Positional Strategy", 
                "description": "Multi-day positional strategy with Greeks management",
                "complexity_level": "intermediate",
                "excel_template_sheets": ["Position", "Greeks", "Risk"]
            },
            "market_regime": {
                "display_name": "Market Regime Strategy",
                "description": "18-regime classification with VTS architecture",
                "complexity_level": "advanced",
                "excel_template_sheets": ["Indicators", "Regimes", "Transitions", "Analysis", "Straddle", "Risk", "Status"]
            },
            "ml_triple_straddle": {
                "display_name": "ML Triple Rolling Straddle",
                "description": "Sophisticated ML-based options trading with 5 models and 160+ features",
                "complexity_level": "advanced",
                "excel_template_sheets": [
                    "LightGBM_Config", "CatBoost_Config", "TabNet_Config", "LSTM_Config", 
                    "Transformer_Config", "Ensemble_Config", "Market_Regime_Features",
                    "Greek_Features", "IV_Features", "OI_Features", "Technical_Features",
                    "Microstructure_Features", "Position_Sizing", "Risk_Limits", 
                    "Stop_Loss", "Circuit_Breaker", "Straddle_Config", "Signal_Filters",
                    "Signal_Processing", "Training_Config", "Model_Training", 
                    "Backtesting", "HeavyDB_Connection", "Data_Source", "Overview",
                    "Performance_Targets"
                ]
            },
            "indicator": {
                "display_name": "Technical Indicator Strategy",
                "description": "Technical analysis with TA-Lib, SMC, and patterns",
                "complexity_level": "intermediate",
                "excel_template_sheets": ["IndicatorConfiguration", "SignalConditions", "RiskManagement", "TimeframeSettings"]
            },
            "strategy_consolidation": {
                "display_name": "Strategy Consolidation",
                "description": "Portfolio-level strategy management with dynamic weights",
                "complexity_level": "advanced",
                "excel_template_sheets": ["Strategies", "ConsolidationRules", "Optimization", "PortfolioManagement"]
            }
        }
    
    def extract_from_configuration_class(self, config_class: Type[BaseConfiguration], 
                                       strategy_type: str) -> bool:
        """
        Extract parameters from a configuration class
        
        Args:
            config_class: Configuration class to analyze
            strategy_type: Strategy type identifier
            
        Returns:
            True if extraction successful
        """
        try:
            logger.info(f"Extracting parameters from {config_class.__name__}")
            
            # Register strategy metadata
            self._register_strategy_metadata(strategy_type)
            
            # Create temporary instance to get schema and defaults
            temp_instance = config_class(strategy_type, "temp")
            
            # Get schema
            schema = temp_instance.get_schema()
            defaults = temp_instance.get_default_values()
            
            # Extract parameters from schema
            self._extract_parameters_from_schema(
                schema, defaults, strategy_type, "", ""
            )
            
            logger.info(f"Successfully extracted parameters for {strategy_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract from {config_class.__name__}: {e}")
            return False
    
    def _register_strategy_metadata(self, strategy_type: str):
        """Register strategy metadata"""
        metadata = self.strategy_metadata.get(strategy_type, {})
        
        strategy = StrategyMetadata(
            strategy_type=strategy_type,
            display_name=metadata.get("display_name", strategy_type.upper()),
            description=metadata.get("description", f"{strategy_type} strategy"),
            version="1.0",
            excel_template_sheets=metadata.get("excel_template_sheets", []),
            parameter_count=0,  # Will be updated later
            category_count=0,   # Will be updated later  
            complexity_level=metadata.get("complexity_level", "intermediate")
        )
        
        self.registry.register_strategy(strategy)
    
    def _extract_parameters_from_schema(self, schema: Dict[str, Any], 
                                      defaults: Dict[str, Any],
                                      strategy_type: str,
                                      category_path: str,
                                      param_path: str):
        """Recursively extract parameters from JSON schema"""
        
        if "properties" not in schema:
            return
        
        properties = schema["properties"]
        required_fields = schema.get("required", [])
        
        for prop_name, prop_schema in properties.items():
            current_path = f"{param_path}.{prop_name}" if param_path else prop_name
            current_category = category_path or prop_name
            
            # Get default value
            default_value = self._get_nested_value(defaults, current_path)
            
            if prop_schema.get("type") == "object" and "properties" in prop_schema:
                # This is a category/object - register category and recurse
                self._register_category(strategy_type, current_category, prop_name)
                
                self._extract_parameters_from_schema(
                    prop_schema, defaults, strategy_type, 
                    current_category, current_path
                )
            else:
                # This is a parameter - register it
                self._register_parameter(
                    strategy_type, current_category, prop_name, 
                    prop_schema, default_value, current_path,
                    prop_name in required_fields
                )
    
    def _register_category(self, strategy_type: str, category_name: str, display_name: str):
        """Register a parameter category"""
        category_id = f"{strategy_type}_{category_name}"
        
        # Don't register duplicate categories
        if self.registry.get_category(category_id):
            return
        
        category = ParameterCategory(
            category_id=category_id,
            strategy_type=strategy_type,
            name=category_name,
            display_name=self._humanize_name(display_name),
            description=f"{self._humanize_name(display_name)} configuration parameters",
            order=self._get_category_order(category_name)
        )
        
        self.registry.register_category(category)
    
    def _register_parameter(self, strategy_type: str, category: str, name: str,
                          schema: Dict[str, Any], default_value: Any, 
                          full_path: str, is_required: bool):
        """Register a parameter definition"""
        
        parameter_id = f"{strategy_type}_{category}_{name}"
        
        # Determine parameter type
        schema_type = schema.get("type", "string")
        param_type = self.type_mapping.get(schema_type, ParameterType.STRING)
        
        # Handle special cases
        if "enum" in schema:
            param_type = ParameterType.ENUM
        elif schema.get("format") == "time":
            param_type = ParameterType.TIME
        elif schema.get("format") == "date":
            param_type = ParameterType.DATE
        elif "percentage" in name.lower() or "percent" in name.lower():
            param_type = ParameterType.PERCENTAGE
        elif "amount" in name.lower() or "capital" in name.lower() or "price" in name.lower():
            param_type = ParameterType.CURRENCY
        
        # Extract validation rules
        validation_rules = self._extract_validation_rules(schema, is_required)
        
        # Generate UI hints
        ui_hints = self._generate_ui_hints(name, param_type, schema)
        
        # Create parameter definition
        parameter = ParameterDefinition(
            parameter_id=parameter_id,
            strategy_type=strategy_type,
            category=category,
            name=name,
            data_type=param_type,
            default_value=default_value,
            validation_rules=validation_rules,
            ui_hints=ui_hints,
            enum_values=schema.get("enum"),
            description=self._generate_description(name, category, param_type)
        )
        
        self.registry.register_parameter(parameter)
    
    def _extract_validation_rules(self, schema: Dict[str, Any], is_required: bool) -> List[ValidationRule]:
        """Extract validation rules from schema"""
        rules = []
        
        if is_required:
            rules.append(ValidationRule("required", True, "This field is required"))
        
        if "minimum" in schema:
            rules.append(ValidationRule("minimum", schema["minimum"], 
                                      f"Value must be at least {schema['minimum']}"))
        
        if "maximum" in schema:
            rules.append(ValidationRule("maximum", schema["maximum"],
                                      f"Value must be at most {schema['maximum']}"))
        
        if "minLength" in schema:
            rules.append(ValidationRule("min_length", schema["minLength"],
                                      f"Must be at least {schema['minLength']} characters"))
        
        if "maxLength" in schema:
            rules.append(ValidationRule("max_length", schema["maxLength"],
                                      f"Must be at most {schema['maxLength']} characters"))
        
        if "pattern" in schema:
            rules.append(ValidationRule("pattern", schema["pattern"],
                                      "Invalid format"))
        
        if "format" in schema:
            rules.append(ValidationRule("format", schema["format"],
                                      f"Must be valid {schema['format']} format"))
        
        return rules
    
    def _generate_ui_hints(self, name: str, param_type: ParameterType, 
                          schema: Dict[str, Any]) -> UIHints:
        """Generate UI hints for parameter"""
        
        # Determine widget type
        widget_type = WidgetType.TEXT_INPUT  # Default
        
        if param_type == ParameterType.BOOLEAN:
            widget_type = WidgetType.TOGGLE
        elif param_type == ParameterType.NUMBER:
            if "minimum" in schema and "maximum" in schema:
                widget_type = WidgetType.SLIDER
            else:
                widget_type = WidgetType.NUMBER_INPUT
        elif param_type == ParameterType.INTEGER:
            widget_type = WidgetType.NUMBER_INPUT
        elif param_type == ParameterType.ENUM:
            widget_type = WidgetType.DROPDOWN
        elif param_type == ParameterType.TIME:
            widget_type = WidgetType.TIME_PICKER
        elif param_type == ParameterType.DATE:
            widget_type = WidgetType.DATE_PICKER
        elif param_type == ParameterType.CURRENCY:
            widget_type = WidgetType.CURRENCY_INPUT
        elif param_type == ParameterType.PERCENTAGE:
            widget_type = WidgetType.PERCENTAGE_INPUT
        
        # Check name-based mappings
        for keyword, widget in self.widget_mapping.items():
            if keyword in name.lower():
                widget_type = widget
                break
        
        return UIHints(
            widget_type=widget_type,
            label=self._humanize_name(name),
            help_text=self._generate_help_text(name, param_type),
            placeholder=self._generate_placeholder(name, param_type),
            group=None,  # Will be set by category
            order=self._get_parameter_order(name)
        )
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except:
            return None
    
    def _humanize_name(self, name: str) -> str:
        """Convert snake_case to human readable name"""
        return name.replace('_', ' ').title()
    
    def _generate_description(self, name: str, category: str, param_type: ParameterType) -> str:
        """Generate parameter description"""
        base_desc = self._humanize_name(name)
        
        if param_type == ParameterType.CURRENCY:
            return f"{base_desc} amount in the base currency"
        elif param_type == ParameterType.PERCENTAGE:
            return f"{base_desc} as a percentage (0-100)"
        elif param_type == ParameterType.TIME:
            return f"{base_desc} in HH:MM:SS format"
        elif param_type == ParameterType.BOOLEAN:
            return f"Enable or disable {base_desc.lower()}"
        else:
            return f"{base_desc} configuration parameter"
    
    def _generate_help_text(self, name: str, param_type: ParameterType) -> str:
        """Generate help text for parameter"""
        help_texts = {
            "capital": "Total trading capital allocated for this strategy",
            "stop_loss": "Stop loss percentage or points",
            "target": "Profit target percentage or points", 
            "entry_time": "Strategy entry time (market hours)",
            "exit_time": "Strategy exit time (market hours)",
            "risk_per_trade": "Maximum risk per trade as percentage of capital",
            "max_trades": "Maximum number of concurrent trades"
        }
        
        return help_texts.get(name.lower(), f"Configure {self._humanize_name(name).lower()}")
    
    def _generate_placeholder(self, name: str, param_type: ParameterType) -> Optional[str]:
        """Generate placeholder text"""
        if param_type == ParameterType.TIME:
            return "09:30:00"
        elif param_type == ParameterType.CURRENCY:
            return "100000"
        elif param_type == ParameterType.PERCENTAGE:
            return "2.5"
        elif param_type == ParameterType.NUMBER:
            return "0"
        
        return None
    
    def _get_category_order(self, category_name: str) -> int:
        """Get display order for category"""
        category_order = {
            "portfolio_settings": 1,
            "strategy_parameters": 2, 
            "risk_management": 3,
            "enhancement_parameters": 4,
            "ml_models": 1,
            "features": 2,
            "signal_generation": 3,
            "training": 4,
            "database": 5,
            "system": 6
        }
        
        return category_order.get(category_name, 99)
    
    def _get_parameter_order(self, param_name: str) -> int:
        """Get display order for parameter"""
        important_params = {
            "capital": 1,
            "strategy_type": 2,
            "entry_time": 10,
            "exit_time": 11,
            "stop_loss": 20,
            "target": 21,
            "enabled": 5
        }
        
        return important_params.get(param_name, 50)
    
    def extract_all_strategies(self) -> bool:
        """Extract parameters from all registered strategy configurations"""
        try:
            # Import all strategy configurations
            from ..strategies import (
                TBSConfiguration, TVConfiguration, ORBConfiguration,
                OIConfiguration, MLConfiguration, POSConfiguration,
                MarketRegimeConfiguration, MLTripleStraddleConfiguration,
                IndicatorConfiguration, StrategyConsolidationConfiguration
            )
            
            strategies = [
                (TBSConfiguration, "tbs"),
                (TVConfiguration, "tv"), 
                (ORBConfiguration, "orb"),
                (OIConfiguration, "oi"),
                (MLConfiguration, "ml"),
                (POSConfiguration, "pos"),
                (MarketRegimeConfiguration, "market_regime"),
                (MLTripleStraddleConfiguration, "ml_triple_straddle"),
                (IndicatorConfiguration, "indicator"),
                (StrategyConsolidationConfiguration, "strategy_consolidation")
            ]
            
            success_count = 0
            for config_class, strategy_type in strategies:
                if self.extract_from_configuration_class(config_class, strategy_type):
                    success_count += 1
            
            logger.info(f"Successfully extracted {success_count}/{len(strategies)} strategies")
            
            # Update strategy parameter counts
            self._update_strategy_statistics()
            
            return success_count == len(strategies)
            
        except Exception as e:
            logger.error(f"Failed to extract all strategies: {e}")
            return False
    
    def _update_strategy_statistics(self):
        """Update parameter and category counts for strategies"""
        for strategy_type in self.registry.get_all_strategies():
            params = self.registry.get_parameters_by_strategy(strategy_type.strategy_type)
            categories = self.registry.get_categories_by_strategy(strategy_type.strategy_type)
            
            strategy_type.parameter_count = len(params)
            strategy_type.category_count = len(categories)
            
            self.registry.register_strategy(strategy_type)
    
    def export_extracted_parameters(self, output_path: str) -> bool:
        """Export extracted parameters to JSON file"""
        return self.registry.export_to_json(output_path)
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about the extraction process"""
        return self.registry.get_statistics()