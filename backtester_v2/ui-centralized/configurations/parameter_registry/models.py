"""
Parameter Registry Models

Data models for the parameter registry system including parameter definitions,
categories, validation rules, and UI metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

class ParameterType(str, Enum):
    """Parameter data types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    ENUM = "enum"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

class WidgetType(str, Enum):
    """UI widget types for parameter input"""
    TEXT_INPUT = "text_input"
    NUMBER_INPUT = "number_input"
    CURRENCY_INPUT = "currency_input"
    PERCENTAGE_INPUT = "percentage_input"
    DATE_PICKER = "date_picker"
    TIME_PICKER = "time_picker"
    DATETIME_PICKER = "datetime_picker"
    DROPDOWN = "dropdown"
    MULTI_SELECT = "multi_select"
    CHECKBOX = "checkbox"
    TOGGLE = "toggle"
    SLIDER = "slider"
    RANGE_SLIDER = "range_slider"
    FILE_UPLOAD = "file_upload"
    TEXT_AREA = "text_area"
    COLOR_PICKER = "color_picker"

@dataclass
class ValidationRule:
    """Validation rule for a parameter"""
    rule_type: str  # "minimum", "maximum", "pattern", "required", etc.
    value: Any
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "value": self.value,
            "message": self.message
        }

@dataclass
class UIHints:
    """UI hints for parameter rendering"""
    widget_type: WidgetType
    label: str
    help_text: Optional[str] = None
    placeholder: Optional[str] = None
    group: Optional[str] = None
    order: int = 0
    visible: bool = True
    enabled: bool = True
    width: Optional[str] = None  # "full", "half", "third", etc.
    conditional_visibility: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "widget_type": self.widget_type.value,
            "label": self.label,
            "help_text": self.help_text,
            "placeholder": self.placeholder,
            "group": self.group,
            "order": self.order,
            "visible": self.visible,
            "enabled": self.enabled,
            "width": self.width,
            "conditional_visibility": self.conditional_visibility
        }

@dataclass
class ParameterDefinition:
    """Complete parameter definition"""
    parameter_id: str
    strategy_type: str
    category: str
    name: str
    data_type: ParameterType
    default_value: Any
    validation_rules: List[ValidationRule] = field(default_factory=list)
    ui_hints: Optional[UIHints] = None
    enum_values: Optional[List[Any]] = None
    description: Optional[str] = None
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the parameter definition"""
        if self.data_type == ParameterType.ENUM and not self.enum_values:
            raise ValueError(f"Enum parameter {self.parameter_id} must have enum_values")
    
    @property
    def full_path(self) -> str:
        """Get full parameter path: strategy.category.name"""
        return f"{self.strategy_type}.{self.category}.{self.name}"
    
    def is_required(self) -> bool:
        """Check if parameter is required"""
        return any(rule.rule_type == "required" and rule.value for rule in self.validation_rules)
    
    def get_validation_schema(self) -> Dict[str, Any]:
        """Generate JSON schema validation rules"""
        schema = {
            "type": self.data_type.value
        }
        
        # Add enum values
        if self.data_type == ParameterType.ENUM and self.enum_values:
            schema["enum"] = self.enum_values
        
        # Add validation rules
        for rule in self.validation_rules:
            if rule.rule_type == "minimum":
                schema["minimum"] = rule.value
            elif rule.rule_type == "maximum":
                schema["maximum"] = rule.value
            elif rule.rule_type == "pattern":
                schema["pattern"] = rule.value
            elif rule.rule_type == "min_length":
                schema["minLength"] = rule.value
            elif rule.rule_type == "max_length":
                schema["maxLength"] = rule.value
            elif rule.rule_type == "format":
                schema["format"] = rule.value
        
        return schema
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "parameter_id": self.parameter_id,
            "strategy_type": self.strategy_type,
            "category": self.category,
            "name": self.name,
            "data_type": self.data_type.value,
            "default_value": self.default_value,
            "validation_rules": [rule.to_dict() for rule in self.validation_rules],
            "ui_hints": self.ui_hints.to_dict() if self.ui_hints else None,
            "enum_values": self.enum_values,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterDefinition':
        """Create from dictionary"""
        validation_rules = [
            ValidationRule(
                rule_type=rule["rule_type"],
                value=rule["value"],
                message=rule.get("message")
            )
            for rule in data.get("validation_rules", [])
        ]
        
        ui_hints = None
        if data.get("ui_hints"):
            ui_hints_data = data["ui_hints"]
            ui_hints = UIHints(
                widget_type=WidgetType(ui_hints_data["widget_type"]),
                label=ui_hints_data["label"],
                help_text=ui_hints_data.get("help_text"),
                placeholder=ui_hints_data.get("placeholder"),
                group=ui_hints_data.get("group"),
                order=ui_hints_data.get("order", 0),
                visible=ui_hints_data.get("visible", True),
                enabled=ui_hints_data.get("enabled", True),
                width=ui_hints_data.get("width"),
                conditional_visibility=ui_hints_data.get("conditional_visibility")
            )
        
        return cls(
            parameter_id=data["parameter_id"],
            strategy_type=data["strategy_type"],
            category=data["category"],
            name=data["name"],
            data_type=ParameterType(data["data_type"]),
            default_value=data["default_value"],
            validation_rules=validation_rules,
            ui_hints=ui_hints,
            enum_values=data.get("enum_values"),
            description=data.get("description"),
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        )

@dataclass
class ParameterCategory:
    """Category grouping for parameters"""
    category_id: str
    strategy_type: str
    name: str
    display_name: str
    description: Optional[str] = None
    order: int = 0
    icon: Optional[str] = None
    collapsible: bool = True
    collapsed_by_default: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_id": self.category_id,
            "strategy_type": self.strategy_type,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "order": self.order,
            "icon": self.icon,
            "collapsible": self.collapsible,
            "collapsed_by_default": self.collapsed_by_default
        }

@dataclass
class StrategyMetadata:
    """Metadata for a strategy type"""
    strategy_type: str
    display_name: str
    description: str
    version: str
    excel_template_sheets: List[str]
    parameter_count: int
    category_count: int
    complexity_level: str  # "basic", "intermediate", "advanced"
    documentation_url: Optional[str] = None
    icon: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "excel_template_sheets": self.excel_template_sheets,
            "parameter_count": self.parameter_count,
            "category_count": self.category_count,
            "complexity_level": self.complexity_level,
            "documentation_url": self.documentation_url,
            "icon": self.icon
        }