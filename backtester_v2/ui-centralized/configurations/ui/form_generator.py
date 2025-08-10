"""
Schema-Driven Form Generator

Dynamically generates UI forms from parameter schemas stored in the registry.
Supports multiple UI frameworks (HTML, React, Vue, Django) and adaptive layouts.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from enum import Enum

from ..parameter_registry import ParameterRegistry, ParameterDefinition, ParameterType, UIHints, ValidationRule
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class FormFramework(Enum):
    """Supported UI frameworks"""
    HTML = "html"
    REACT = "react" 
    VUE = "vue"
    DJANGO = "django"
    FLASK = "flask"
    BOOTSTRAP = "bootstrap"

class LayoutType(Enum):
    """Form layout types"""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    GRID = "grid"
    TABS = "tabs"
    ACCORDION = "accordion"
    WIZARD = "wizard"

@dataclass
class FormConfig:
    """Configuration for form generation"""
    framework: FormFramework = FormFramework.HTML
    layout: LayoutType = LayoutType.VERTICAL
    include_validation: bool = True
    include_help_text: bool = True
    group_by_category: bool = True
    show_advanced: bool = False
    responsive: bool = True
    theme: str = "default"
    custom_css_classes: Dict[str, str] = field(default_factory=dict)
    field_order: Optional[List[str]] = None
    excluded_fields: List[str] = field(default_factory=list)
    readonly_fields: List[str] = field(default_factory=list)

@dataclass
class FormField:
    """Generated form field representation"""
    parameter_id: str
    name: str
    label: str
    field_type: str
    required: bool = False
    default_value: Any = None
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    options: Optional[List[Dict[str, Any]]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    pattern: Optional[str] = None
    css_classes: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FormSection:
    """Form section grouping related fields"""
    name: str
    title: str
    description: Optional[str] = None
    fields: List[FormField] = field(default_factory=list)
    collapsible: bool = False
    collapsed: bool = False
    css_classes: List[str] = field(default_factory=list)

@dataclass
class GeneratedForm:
    """Complete generated form"""
    strategy_type: str
    title: str
    description: Optional[str]
    sections: List[FormSection]
    framework: FormFramework
    layout: LayoutType
    form_config: FormConfig
    generated_code: str
    css_styles: str
    javascript_code: str
    validation_schema: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class SchemaFormGenerator:
    """
    Schema-driven form generator
    
    Generates dynamic UI forms from parameter schemas with support for:
    - Multiple UI frameworks (HTML, React, Vue, Django)
    - Adaptive layouts and responsive design
    - Automatic validation rule generation
    - Category-based field grouping
    - Advanced field types and widgets
    - Custom styling and theming
    """
    
    def __init__(self, parameter_registry: Optional[ParameterRegistry] = None):
        """
        Initialize form generator
        
        Args:
            parameter_registry: Parameter registry instance
        """
        self.registry = parameter_registry or ParameterRegistry()
        
        # Field type mappings for different frameworks
        self.field_type_mappings = {
            FormFramework.HTML: {
                ParameterType.STRING: "text",
                ParameterType.INTEGER: "number",
                ParameterType.FLOAT: "number",
                ParameterType.BOOLEAN: "checkbox",
                ParameterType.LIST: "select",
                ParameterType.DICT: "textarea",
                ParameterType.DATETIME: "datetime-local",
                ParameterType.DATE: "date",
                ParameterType.TIME: "time",
                ParameterType.EMAIL: "email",
                ParameterType.URL: "url",
                ParameterType.PASSWORD: "password",
                ParameterType.FILE: "file",
                ParameterType.COLOR: "color"
            },
            FormFramework.REACT: {
                ParameterType.STRING: "TextField",
                ParameterType.INTEGER: "NumberField", 
                ParameterType.FLOAT: "NumberField",
                ParameterType.BOOLEAN: "Checkbox",
                ParameterType.LIST: "Select",
                ParameterType.DICT: "TextArea",
                ParameterType.DATETIME: "DateTimePicker",
                ParameterType.DATE: "DatePicker",
                ParameterType.TIME: "TimePicker",
                ParameterType.EMAIL: "EmailField",
                ParameterType.URL: "URLField",
                ParameterType.PASSWORD: "PasswordField",
                ParameterType.FILE: "FileUpload",
                ParameterType.COLOR: "ColorPicker"
            }
        }
        
        logger.info("SchemaFormGenerator initialized")
    
    def generate_form(self, 
                     strategy_type: str,
                     form_config: Optional[FormConfig] = None) -> GeneratedForm:
        """
        Generate a complete form for a strategy type
        
        Args:
            strategy_type: Strategy type to generate form for
            form_config: Form configuration options
            
        Returns:
            Generated form with code, styles, and metadata
        """
        config = form_config or FormConfig()
        
        # Get parameters for strategy
        parameters = self.registry.get_parameters(strategy_type)
        if not parameters:
            raise ValueError(f"No parameters found for strategy type: {strategy_type}")
        
        # Generate form fields
        fields = self._generate_fields(parameters, config)
        
        # Group fields into sections
        sections = self._group_fields_into_sections(fields, config)
        
        # Generate framework-specific code
        generated_code = self._generate_code(strategy_type, sections, config)
        css_styles = self._generate_css(config)
        javascript_code = self._generate_javascript(config)
        
        # Generate validation schema
        validation_schema = self._generate_validation_schema(parameters)
        
        form = GeneratedForm(
            strategy_type=strategy_type,
            title=f"{strategy_type.upper()} Configuration",
            description=f"Dynamic form for {strategy_type} strategy parameters",
            sections=sections,
            framework=config.framework,
            layout=config.layout,
            form_config=config,
            generated_code=generated_code,
            css_styles=css_styles,
            javascript_code=javascript_code,
            validation_schema=validation_schema,
            metadata={
                "total_fields": len(fields),
                "total_sections": len(sections),
                "generated_at": str(Path(__file__).stat().st_mtime),
                "framework": config.framework.value,
                "layout": config.layout.value
            }
        )
        
        logger.info(f"Generated form for {strategy_type}: {len(fields)} fields, {len(sections)} sections")
        return form
    
    def _generate_fields(self, 
                        parameters: List[ParameterDefinition],
                        config: FormConfig) -> List[FormField]:
        """Generate form fields from parameter definitions"""
        
        fields = []
        
        for param in parameters:
            # Skip excluded fields
            if param.parameter_id in config.excluded_fields:
                continue
            
            # Skip advanced fields if not requested
            if not config.show_advanced and param.ui_hints and param.ui_hints.advanced:
                continue
            
            field = self._create_form_field(param, config)
            fields.append(field)
        
        # Apply field ordering
        if config.field_order:
            ordered_fields = []
            field_map = {f.parameter_id: f for f in fields}
            
            # Add fields in specified order
            for field_id in config.field_order:
                if field_id in field_map:
                    ordered_fields.append(field_map[field_id])
                    del field_map[field_id]
            
            # Add remaining fields
            ordered_fields.extend(field_map.values())
            fields = ordered_fields
        
        return fields
    
    def _create_form_field(self, 
                          param: ParameterDefinition,
                          config: FormConfig) -> FormField:
        """Create a form field from parameter definition"""
        
        # Determine field type
        field_type = self._get_field_type(param.data_type, config.framework)
        
        # Extract UI hints
        ui_hints = param.ui_hints or UIHints()
        
        # Build validation rules
        validation_rules = []
        for rule in param.validation_rules:
            validation_rules.append({
                "type": rule.rule_type,
                "value": rule.value,
                "message": rule.error_message
            })
        
        # Extract options for select fields
        options = None
        if param.data_type == ParameterType.LIST and ui_hints.options:
            options = [
                {"value": opt, "label": opt}
                for opt in ui_hints.options
            ]
        
        # Determine if field is readonly
        readonly = param.parameter_id in config.readonly_fields
        
        # Build CSS classes
        css_classes = []
        if ui_hints.css_class:
            css_classes.append(ui_hints.css_class)
        if param.parameter_id in config.custom_css_classes:
            css_classes.append(config.custom_css_classes[param.parameter_id])
        if readonly:
            css_classes.append("readonly")
        
        field = FormField(
            parameter_id=param.parameter_id,
            name=param.name,
            label=ui_hints.label or param.name.replace('_', ' ').title(),
            field_type=field_type,
            required=any(rule.rule_type == "required" for rule in param.validation_rules),
            default_value=param.default_value,
            placeholder=ui_hints.placeholder,
            help_text=ui_hints.help_text if config.include_help_text else None,
            validation_rules=validation_rules if config.include_validation else [],
            options=options,
            min_value=ui_hints.min_value,
            max_value=ui_hints.max_value,
            step=ui_hints.step,
            pattern=ui_hints.pattern,
            css_classes=css_classes,
            attributes={"readonly": readonly} if readonly else {}
        )
        
        return field
    
    def _get_field_type(self, param_type: ParameterType, framework: FormFramework) -> str:
        """Get field type for framework"""
        mapping = self.field_type_mappings.get(framework, self.field_type_mappings[FormFramework.HTML])
        return mapping.get(param_type, "text")
    
    def _group_fields_into_sections(self,
                                   fields: List[FormField], 
                                   config: FormConfig) -> List[FormSection]:
        """Group fields into logical sections"""
        
        if not config.group_by_category:
            # Single section with all fields
            return [FormSection(
                name="all_fields",
                title="Configuration Parameters",
                description="All configuration parameters",
                fields=fields
            )]
        
        # Group by parameter category
        sections_map = {}
        
        for field in fields:
            # Extract category from parameter_id (e.g., "portfolio.capital" -> "portfolio")
            parts = field.parameter_id.split('.')
            category = parts[0] if len(parts) > 1 else "general"
            
            if category not in sections_map:
                sections_map[category] = FormSection(
                    name=category,
                    title=category.replace('_', ' ').title(),
                    description=f"{category} related parameters",
                    fields=[],
                    collapsible=True
                )
            
            sections_map[category].fields.append(field)
        
        # Convert to list and sort by importance
        sections = list(sections_map.values())
        
        # Define section priority order
        priority_order = [
            "portfolio", "trading", "risk", "entry", "exit", 
            "indicators", "ml", "optimization", "misc", "general"
        ]
        
        def section_priority(section):
            try:
                return priority_order.index(section.name)
            except ValueError:
                return len(priority_order)
        
        sections.sort(key=section_priority)
        
        return sections
    
    def _generate_code(self,
                      strategy_type: str,
                      sections: List[FormSection],
                      config: FormConfig) -> str:
        """Generate framework-specific form code"""
        
        if config.framework == FormFramework.HTML:
            return self._generate_html_form(strategy_type, sections, config)
        elif config.framework == FormFramework.REACT:
            return self._generate_react_form(strategy_type, sections, config)
        elif config.framework == FormFramework.VUE:
            return self._generate_vue_form(strategy_type, sections, config)
        elif config.framework == FormFramework.DJANGO:
            return self._generate_django_form(strategy_type, sections, config)
        else:
            return self._generate_html_form(strategy_type, sections, config)
    
    def _generate_html_form(self,
                           strategy_type: str,
                           sections: List[FormSection],
                           config: FormConfig) -> str:
        """Generate HTML form"""
        
        html_parts = []
        
        # Form header
        html_parts.append(f'<form id="{strategy_type}-config-form" class="strategy-config-form">')
        html_parts.append(f'  <h2>{strategy_type.upper()} Configuration</h2>')
        
        # Generate sections
        for section in sections:
            if config.layout == LayoutType.TABS:
                html_parts.append(f'  <div class="tab-pane" id="tab-{section.name}">')
            elif config.layout == LayoutType.ACCORDION:
                html_parts.append(f'  <div class="accordion-section">')
                html_parts.append(f'    <h3 class="accordion-header">{section.title}</h3>')
                html_parts.append(f'    <div class="accordion-content">')
            else:
                html_parts.append(f'  <fieldset class="form-section">')
                html_parts.append(f'    <legend>{section.title}</legend>')
            
            # Generate fields
            for field in section.fields:
                html_parts.append(self._generate_html_field(field, config))
            
            # Close section
            if config.layout == LayoutType.ACCORDION:
                html_parts.append('    </div>')
                html_parts.append('  </div>')
            else:
                html_parts.append('  </fieldset>')
        
        # Form footer
        html_parts.append('  <div class="form-actions">')
        html_parts.append('    <button type="submit" class="btn btn-primary">Save Configuration</button>')
        html_parts.append('    <button type="reset" class="btn btn-secondary">Reset</button>')
        html_parts.append('  </div>')
        html_parts.append('</form>')
        
        return '\n'.join(html_parts)
    
    def _generate_html_field(self, field: FormField, config: FormConfig) -> str:
        """Generate HTML for a single field"""
        
        css_classes = ' '.join(field.css_classes) if field.css_classes else ''
        required_attr = 'required' if field.required else ''
        readonly_attr = 'readonly' if field.attributes.get('readonly') else ''
        
        html_parts = []
        html_parts.append(f'    <div class="form-group {css_classes}">')
        html_parts.append(f'      <label for="{field.parameter_id}">{field.label}')
        if field.required:
            html_parts.append(' <span class="required">*</span>')
        html_parts.append('</label>')
        
        # Generate input based on field type
        if field.field_type == 'select':
            html_parts.append(f'      <select id="{field.parameter_id}" name="{field.name}" {required_attr} {readonly_attr}>')
            if field.options:
                for option in field.options:
                    selected = 'selected' if option['value'] == field.default_value else ''
                    html_parts.append(f'        <option value="{option["value"]}" {selected}>{option["label"]}</option>')
            html_parts.append('      </select>')
        
        elif field.field_type == 'textarea':
            value = json.dumps(field.default_value) if isinstance(field.default_value, dict) else field.default_value or ''
            html_parts.append(f'      <textarea id="{field.parameter_id}" name="{field.name}" {required_attr} {readonly_attr}>{value}</textarea>')
        
        elif field.field_type == 'checkbox':
            checked = 'checked' if field.default_value else ''
            html_parts.append(f'      <input type="checkbox" id="{field.parameter_id}" name="{field.name}" {checked} {readonly_attr}>')
        
        else:
            # Standard input fields
            value_attr = f'value="{field.default_value}"' if field.default_value is not None else ''
            placeholder_attr = f'placeholder="{field.placeholder}"' if field.placeholder else ''
            min_attr = f'min="{field.min_value}"' if field.min_value is not None else ''
            max_attr = f'max="{field.max_value}"' if field.max_value is not None else ''
            step_attr = f'step="{field.step}"' if field.step is not None else ''
            pattern_attr = f'pattern="{field.pattern}"' if field.pattern else ''
            
            html_parts.append(f'      <input type="{field.field_type}" id="{field.parameter_id}" name="{field.name}" '
                            f'{value_attr} {placeholder_attr} {min_attr} {max_attr} {step_attr} {pattern_attr} '
                            f'{required_attr} {readonly_attr}>')
        
        # Add help text
        if field.help_text:
            html_parts.append(f'      <small class="form-help">{field.help_text}</small>')
        
        html_parts.append('    </div>')
        
        return '\n'.join(html_parts)
    
    def _generate_react_form(self,
                            strategy_type: str,
                            sections: List[FormSection],
                            config: FormConfig) -> str:
        """Generate React form component"""
        
        component_name = f"{strategy_type.title()}ConfigForm"
        
        react_code = f'''
import React, {{ useState, useEffect }} from 'react';
import {{ Form, Button, Row, Col, Tab, Nav }} from 'react-bootstrap';

const {component_name} = () => {{
  const [formData, setFormData] = useState({{}});
  const [errors, setErrors] = useState({{}});
  
  const handleInputChange = (field, value) => {{
    setFormData(prev => ({{ ...prev, [field]: value }}));
    // Clear error when user starts typing
    if (errors[field]) {{
      setErrors(prev => ({{ ...prev, [field]: null }}));
    }}
  }};
  
  const handleSubmit = (e) => {{
    e.preventDefault();
    // Add validation and submit logic
    console.log('Form data:', formData);
  }};
  
  return (
    <Form onSubmit={{handleSubmit}} className="strategy-config-form">
      <h2>{strategy_type.upper()} Configuration</h2>
'''
        
        if config.layout == LayoutType.TABS:
            react_code += '''
      <Tab.Container defaultActiveKey="0">
        <Nav variant="tabs">
'''
            for i, section in enumerate(sections):
                react_code += f'          <Nav.Item><Nav.Link eventKey="{i}">{section.title}</Nav.Link></Nav.Item>\n'
            
            react_code += '''
        </Nav>
        <Tab.Content>
'''
            for i, section in enumerate(sections):
                react_code += f'          <Tab.Pane eventKey="{i}">\n'
                for field in section.fields:
                    react_code += self._generate_react_field(field, config)
                react_code += '          </Tab.Pane>\n'
            
            react_code += '''
        </Tab.Content>
      </Tab.Container>
'''
        else:
            # Standard layout
            for section in sections:
                react_code += f'      <fieldset className="form-section">\n'
                react_code += f'        <legend>{section.title}</legend>\n'
                for field in section.fields:
                    react_code += self._generate_react_field(field, config)
                react_code += '      </fieldset>\n'
        
        react_code += '''
      <div className="form-actions">
        <Button type="submit" variant="primary">Save Configuration</Button>
        <Button type="reset" variant="secondary">Reset</Button>
      </div>
    </Form>
  );
};

export default ''' + component_name + ''';
'''
        
        return react_code
    
    def _generate_react_field(self, field: FormField, config: FormConfig) -> str:
        """Generate React field component"""
        
        field_code = f'''
        <Form.Group className="mb-3">
          <Form.Label>{field.label}{" *" if field.required else ""}</Form.Label>
'''
        
        if field.field_type == 'TextField':
            field_code += f'''
          <Form.Control
            type="text"
            value={{formData['{field.parameter_id}'] || '{field.default_value or ''}'}}
            onChange={{(e) => handleInputChange('{field.parameter_id}', e.target.value)}}
            placeholder="{field.placeholder or ''}"
            required={{{str(field.required).lower()}}}
            isInvalid={{!!errors['{field.parameter_id}']}}
          />
'''
        elif field.field_type == 'NumberField':
            field_code += f'''
          <Form.Control
            type="number"
            value={{formData['{field.parameter_id}'] || '{field.default_value or ''}'}}
            onChange={{(e) => handleInputChange('{field.parameter_id}', e.target.value)}}
            min={{{field.min_value}}}
            max={{{field.max_value}}}
            step={{{field.step}}}
            required={{{str(field.required).lower()}}}
            isInvalid={{!!errors['{field.parameter_id}']}}
          />
'''
        elif field.field_type == 'Select':
            field_code += f'''
          <Form.Select
            value={{formData['{field.parameter_id}'] || '{field.default_value or ''}'}}
            onChange={{(e) => handleInputChange('{field.parameter_id}', e.target.value)}}
            required={{{str(field.required).lower()}}}
            isInvalid={{!!errors['{field.parameter_id}']}}
          >
'''
            if field.options:
                for option in field.options:
                    field_code += f'            <option value="{option["value"]}">{option["label"]}</option>\n'
            field_code += '          </Form.Select>\n'
        
        elif field.field_type == 'Checkbox':
            field_code += f'''
          <Form.Check
            type="checkbox"
            checked={{formData['{field.parameter_id}'] || {str(field.default_value or False).lower()}}}
            onChange={{(e) => handleInputChange('{field.parameter_id}', e.target.checked)}}
            label="{field.label}"
          />
'''
        
        # Add help text
        if field.help_text:
            field_code += f'          <Form.Text className="text-muted">{field.help_text}</Form.Text>\n'
        
        # Add validation feedback
        field_code += f'''
          <Form.Control.Feedback type="invalid">
            {{errors['{field.parameter_id}']}}
          </Form.Control.Feedback>
        </Form.Group>
'''
        
        return field_code
    
    def _generate_vue_form(self,
                          strategy_type: str,
                          sections: List[FormSection],
                          config: FormConfig) -> str:
        """Generate Vue form component"""
        # Simplified Vue implementation
        return f'''
<template>
  <form @submit.prevent="handleSubmit" class="strategy-config-form">
    <h2>{strategy_type.upper()} Configuration</h2>
    <!-- Vue form fields would be generated here -->
    <div class="form-actions">
      <button type="submit" class="btn btn-primary">Save Configuration</button>
      <button type="reset" class="btn btn-secondary">Reset</button>
    </div>
  </form>
</template>

<script>
export default {{
  name: '{strategy_type.title()}ConfigForm',
  data() {{
    return {{
      formData: {{}},
      errors: {{}}
    }};
  }},
  methods: {{
    handleSubmit() {{
      console.log('Form data:', this.formData);
    }}
  }}
}};
</script>
'''
    
    def _generate_django_form(self,
                             strategy_type: str,
                             sections: List[FormSection],
                             config: FormConfig) -> str:
        """Generate Django form class"""
        
        form_class_name = f"{strategy_type.title()}ConfigForm"
        
        django_code = f'''
from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator

class {form_class_name}(forms.Form):
    """Auto-generated form for {strategy_type} strategy configuration"""
    
'''
        
        for section in sections:
            django_code += f'    # {section.title} Section\n'
            for field in section.fields:
                django_code += self._generate_django_field(field, config)
            django_code += '\n'
        
        return django_code
    
    def _generate_django_field(self, field: FormField, config: FormConfig) -> str:
        """Generate Django form field"""
        
        field_type = "CharField"
        field_args = []
        field_kwargs = []
        
        # Map field types
        if field.field_type == "number":
            if isinstance(field.default_value, int):
                field_type = "IntegerField"
            else:
                field_type = "FloatField"
        elif field.field_type == "checkbox":
            field_type = "BooleanField"
        elif field.field_type == "select":
            field_type = "ChoiceField"
            if field.options:
                choices = [(opt["value"], opt["label"]) for opt in field.options]
                field_kwargs.append(f"choices={choices}")
        elif field.field_type == "textarea":
            field_type = "CharField"
            field_kwargs.append("widget=forms.Textarea")
        elif field.field_type == "email":
            field_type = "EmailField"
        elif field.field_type == "url":
            field_type = "URLField"
        elif field.field_type == "date":
            field_type = "DateField"
        
        # Add common attributes
        if field.label:
            field_kwargs.append(f'label="{field.label}"')
        if field.help_text:
            field_kwargs.append(f'help_text="{field.help_text}"')
        if field.required:
            field_kwargs.append("required=True")
        if field.default_value is not None:
            if isinstance(field.default_value, str):
                field_kwargs.append(f'initial="{field.default_value}"')
            else:
                field_kwargs.append(f'initial={field.default_value}')
        
        # Add validators
        validators = []
        if field.min_value is not None:
            validators.append(f"MinValueValidator({field.min_value})")
        if field.max_value is not None:
            validators.append(f"MaxValueValidator({field.max_value})")
        
        if validators:
            field_kwargs.append(f"validators=[{', '.join(validators)}]")
        
        kwargs_str = ', '.join(field_kwargs)
        
        return f'    {field.name} = forms.{field_type}({kwargs_str})\n'
    
    def _generate_css(self, config: FormConfig) -> str:
        """Generate CSS styles for the form"""
        
        css = '''
.strategy-config-form {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.form-section {
    margin-bottom: 30px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
}

.form-section legend {
    padding: 0 10px;
    font-weight: 600;
    color: #333;
}

.form-group {
    margin-bottom: 20px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
    color: #555;
}

.required {
    color: #e74c3c;
}

.form-group input,
.form-group select,
.form-group textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    transition: border-color 0.3s ease;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.form-group input[readonly],
.form-group select[readonly],
.form-group textarea[readonly] {
    background-color: #f8f9fa;
    cursor: not-allowed;
}

.form-help {
    display: block;
    margin-top: 5px;
    font-size: 12px;
    color: #6c757d;
}

.form-actions {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
    text-align: right;
}

.btn {
    padding: 10px 20px;
    margin-left: 10px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s ease;
}

.btn-primary {
    background-color: #007bff;
    color: white;
}

.btn-primary:hover {
    background-color: #0056b3;
}

.btn-secondary {
    background-color: #6c757d;
    color: white;
}

.btn-secondary:hover {
    background-color: #545b62;
}
'''
        
        if config.responsive:
            css += '''

@media (max-width: 768px) {
    .strategy-config-form {
        padding: 10px;
    }
    
    .form-section {
        padding: 15px;
    }
    
    .form-actions {
        text-align: center;
    }
    
    .btn {
        display: block;
        width: 100%;
        margin: 5px 0;
    }
}
'''
        
        return css
    
    def _generate_javascript(self, config: FormConfig) -> str:
        """Generate JavaScript for form functionality"""
        
        js = '''
// Form validation and interaction
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('.strategy-config-form');
    
    if (form) {
        // Real-time validation
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', validateField);
            input.addEventListener('input', clearErrors);
        });
        
        // Form submission
        form.addEventListener('submit', handleSubmit);
    }
    
    function validateField(event) {
        const field = event.target;
        const value = field.value.trim();
        
        // Clear previous errors
        clearFieldError(field);
        
        // Required field validation
        if (field.hasAttribute('required') && !value) {
            showFieldError(field, 'This field is required');
            return false;
        }
        
        // Type-specific validation
        if (field.type === 'number') {
            const num = parseFloat(value);
            if (isNaN(num)) {
                showFieldError(field, 'Please enter a valid number');
                return false;
            }
            
            if (field.hasAttribute('min') && num < parseFloat(field.min)) {
                showFieldError(field, `Value must be at least ${field.min}`);
                return false;
            }
            
            if (field.hasAttribute('max') && num > parseFloat(field.max)) {
                showFieldError(field, `Value must be at most ${field.max}`);
                return false;
            }
        }
        
        if (field.type === 'email' && value) {
            const emailPattern = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
            if (!emailPattern.test(value)) {
                showFieldError(field, 'Please enter a valid email address');
                return false;
            }
        }
        
        return true;
    }
    
    function clearErrors(event) {
        clearFieldError(event.target);
    }
    
    function showFieldError(field, message) {
        field.classList.add('error');
        
        let errorEl = field.parentNode.querySelector('.field-error');
        if (!errorEl) {
            errorEl = document.createElement('span');
            errorEl.className = 'field-error';
            field.parentNode.appendChild(errorEl);
        }
        errorEl.textContent = message;
    }
    
    function clearFieldError(field) {
        field.classList.remove('error');
        const errorEl = field.parentNode.querySelector('.field-error');
        if (errorEl) {
            errorEl.remove();
        }
    }
    
    function handleSubmit(event) {
        event.preventDefault();
        
        // Validate all fields
        let isValid = true;
        const inputs = form.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            if (!validateField({target: input})) {
                isValid = false;
            }
        });
        
        if (isValid) {
            // Collect form data
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            console.log('Form data:', data);
            
            // Submit form (implement your submission logic here)
            alert('Form submitted successfully!');
        } else {
            alert('Please fix the errors before submitting.');
        }
    }
});
'''
        
        return js
    
    def _generate_validation_schema(self, parameters: List[ParameterDefinition]) -> Dict[str, Any]:
        """Generate JSON schema for form validation"""
        
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in parameters:
            field_schema = {}
            
            # Set type
            if param.data_type == ParameterType.STRING:
                field_schema["type"] = "string"
            elif param.data_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                field_schema["type"] = "number"
            elif param.data_type == ParameterType.BOOLEAN:
                field_schema["type"] = "boolean"
            elif param.data_type == ParameterType.LIST:
                field_schema["type"] = "array"
            elif param.data_type == ParameterType.DICT:
                field_schema["type"] = "object"
            
            # Add validation rules
            for rule in param.validation_rules:
                if rule.rule_type == "required":
                    schema["required"].append(param.parameter_id)
                elif rule.rule_type == "min":
                    field_schema["minimum"] = rule.value
                elif rule.rule_type == "max":
                    field_schema["maximum"] = rule.value
                elif rule.rule_type == "minLength":
                    field_schema["minLength"] = rule.value
                elif rule.rule_type == "maxLength":
                    field_schema["maxLength"] = rule.value
                elif rule.rule_type == "pattern":
                    field_schema["pattern"] = rule.value
                elif rule.rule_type == "enum":
                    field_schema["enum"] = rule.value
            
            # Add UI hints
            if param.ui_hints:
                if param.ui_hints.min_value is not None:
                    field_schema["minimum"] = param.ui_hints.min_value
                if param.ui_hints.max_value is not None:
                    field_schema["maximum"] = param.ui_hints.max_value
                if param.ui_hints.pattern:
                    field_schema["pattern"] = param.ui_hints.pattern
            
            # Set default
            if param.default_value is not None:
                field_schema["default"] = param.default_value
            
            schema["properties"][param.parameter_id] = field_schema
        
        return schema
    
    def save_form(self, form: GeneratedForm, output_dir: str) -> str:
        """Save generated form to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main form file
        form_ext = {
            FormFramework.HTML: "html",
            FormFramework.REACT: "jsx", 
            FormFramework.VUE: "vue",
            FormFramework.DJANGO: "py"
        }.get(form.framework, "html")
        
        form_file = output_path / f"{form.strategy_type}_config_form.{form_ext}"
        form_file.write_text(form.generated_code)
        
        # Save CSS
        css_file = output_path / f"{form.strategy_type}_config_form.css"
        css_file.write_text(form.css_styles)
        
        # Save JavaScript
        js_file = output_path / f"{form.strategy_type}_config_form.js"
        js_file.write_text(form.javascript_code)
        
        # Save validation schema
        schema_file = output_path / f"{form.strategy_type}_validation_schema.json"
        schema_file.write_text(json.dumps(form.validation_schema, indent=2))
        
        # Save form metadata
        metadata_file = output_path / f"{form.strategy_type}_form_metadata.json"
        metadata = {
            "strategy_type": form.strategy_type,
            "framework": form.framework.value,
            "layout": form.layout.value,
            "total_fields": form.metadata["total_fields"],
            "total_sections": form.metadata["total_sections"],
            "generated_at": form.metadata["generated_at"],
            "files": {
                "form": str(form_file),
                "css": str(css_file),
                "javascript": str(js_file),
                "schema": str(schema_file)
            }
        }
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        logger.info(f"Saved {form.framework.value} form for {form.strategy_type} to {output_path}")
        return str(output_path)
    
    def generate_all_strategy_forms(self,
                                   output_dir: str,
                                   framework: FormFramework = FormFramework.HTML) -> Dict[str, str]:
        """Generate forms for all strategy types"""
        
        results = {}
        
        # Get all strategy types from registry
        strategy_types = self.registry.list_strategy_types()
        
        for strategy_type in strategy_types:
            try:
                config = FormConfig(framework=framework)
                form = self.generate_form(strategy_type, config)
                
                strategy_output_dir = Path(output_dir) / strategy_type
                form_path = self.save_form(form, str(strategy_output_dir))
                results[strategy_type] = form_path
                
            except Exception as e:
                logger.error(f"Failed to generate form for {strategy_type}: {e}")
                results[strategy_type] = f"Error: {e}"
        
        logger.info(f"Generated forms for {len(results)} strategy types")
        return results