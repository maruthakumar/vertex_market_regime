#!/usr/bin/env python3
"""
Schema-Driven Form Generator - Example Usage

Demonstrates how to use the form generator to create dynamic UI forms
from parameter schemas for all strategy types.
"""

import logging
import sys
from pathlib import Path

# Add configurations to path
sys.path.append(str(Path(__file__).parent.parent))

from ui import SchemaFormGenerator, FormConfig, FormFramework, LayoutType
from parameter_registry import ParameterRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_form_generation():
    """Demonstrate form generation for different frameworks and layouts"""
    
    print("🎨 Schema-Driven Form Generator Demo")
    print("=" * 50)
    
    # Initialize components
    registry = ParameterRegistry()
    generator = SchemaFormGenerator(registry)
    
    # Example 1: Generate HTML form for TBS strategy
    print("\n1. Generating HTML Form for TBS Strategy...")
    
    html_config = FormConfig(
        framework=FormFramework.HTML,
        layout=LayoutType.VERTICAL,
        group_by_category=True,
        include_validation=True,
        include_help_text=True,
        responsive=True
    )
    
    try:
        tbs_form = generator.generate_form("tbs", html_config)
        print(f"   ✅ Generated HTML form with {tbs_form.metadata['total_fields']} fields")
        print(f"   ✅ Organized into {tbs_form.metadata['total_sections']} sections")
        
        # Save to file
        output_dir = Path(__file__).parent / "generated_forms" / "html"
        form_path = generator.save_form(tbs_form, str(output_dir))
        print(f"   ✅ Saved to: {form_path}")
        
    except Exception as e:
        print(f"   ❌ Failed to generate TBS form: {e}")
    
    # Example 2: Generate React form with tabs layout
    print("\n2. Generating React Form with Tabs Layout...")
    
    react_config = FormConfig(
        framework=FormFramework.REACT,
        layout=LayoutType.TABS,
        group_by_category=True,
        show_advanced=True,
        theme="bootstrap"
    )
    
    try:
        ml_form = generator.generate_form("ml_triple_straddle", react_config)
        print(f"   ✅ Generated React form with {ml_form.metadata['total_fields']} fields")
        print(f"   ✅ Advanced parameters included")
        
        # Save to file
        output_dir = Path(__file__).parent / "generated_forms" / "react"
        form_path = generator.save_form(ml_form, str(output_dir))
        print(f"   ✅ Saved to: {form_path}")
        
    except Exception as e:
        print(f"   ❌ Failed to generate ML form: {e}")
    
    # Example 3: Generate Django form
    print("\n3. Generating Django Form...")
    
    django_config = FormConfig(
        framework=FormFramework.DJANGO,
        layout=LayoutType.VERTICAL,
        include_validation=True,
        readonly_fields=["id", "created_at"],
        custom_css_classes={
            "capital": "form-control-lg",
            "stop_loss": "text-danger"
        }
    )
    
    try:
        oi_form = generator.generate_form("oi", django_config)
        print(f"   ✅ Generated Django form class")
        
        # Save to file
        output_dir = Path(__file__).parent / "generated_forms" / "django"
        form_path = generator.save_form(oi_form, str(output_dir))
        print(f"   ✅ Saved to: {form_path}")
        
    except Exception as e:
        print(f"   ❌ Failed to generate OI form: {e}")
    
    # Example 4: Batch generate forms for all strategies
    print("\n4. Batch Generating Forms for All Strategies...")
    
    try:
        output_dir = Path(__file__).parent / "generated_forms" / "batch"
        results = generator.generate_all_strategy_forms(
            str(output_dir), 
            FormFramework.HTML
        )
        
        successful = len([r for r in results.values() if not r.startswith("Error:")])
        failed = len(results) - successful
        
        print(f"   ✅ Generated forms for {successful} strategies")
        if failed > 0:
            print(f"   ⚠️  Failed to generate {failed} forms")
        
        print(f"   ✅ All forms saved to: {output_dir}")
        
    except Exception as e:
        print(f"   ❌ Batch generation failed: {e}")
    
    # Example 5: Custom form configuration
    print("\n5. Custom Form Configuration Example...")
    
    custom_config = FormConfig(
        framework=FormFramework.HTML,
        layout=LayoutType.GRID,
        group_by_category=True,
        field_order=["capital", "max_position_size", "stop_loss", "take_profit"],
        excluded_fields=["debug_mode", "log_level"],
        readonly_fields=["strategy_id"],
        custom_css_classes={
            "capital": "highlight-field",
            "risk_percentage": "warning-field"
        },
        show_advanced=False,
        responsive=True
    )
    
    try:
        custom_form = generator.generate_form("tv", custom_config)
        print(f"   ✅ Generated custom TV form")
        print(f"   ✅ Custom field ordering applied")
        print(f"   ✅ Advanced fields hidden")
        print(f"   ✅ Custom CSS classes added")
        
    except Exception as e:
        print(f"   ❌ Custom form generation failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Form Generation Demo Complete!")
    
    print("\nGenerated Files Structure:")
    generated_dir = Path(__file__).parent / "generated_forms"
    if generated_dir.exists():
        for item in generated_dir.rglob("*"):
            if item.is_file():
                print(f"  📄 {item.relative_to(generated_dir)}")

def show_form_preview():
    """Show a preview of generated form code"""
    
    print("\n" + "=" * 50)
    print("📝 Form Code Preview")
    print("=" * 50)
    
    registry = ParameterRegistry()
    generator = SchemaFormGenerator(registry)
    
    # Generate a simple form
    config = FormConfig(
        framework=FormFramework.HTML,
        layout=LayoutType.VERTICAL,
        group_by_category=False,
        include_validation=True
    )
    
    try:
        form = generator.generate_form("tbs", config)
        
        print("\nGenerated HTML Form (first 50 lines):")
        print("-" * 40)
        lines = form.generated_code.split('\n')
        for i, line in enumerate(lines[:50]):
            print(f"{i+1:2d}: {line}")
        
        if len(lines) > 50:
            print(f"... ({len(lines) - 50} more lines)")
        
        print("\nGenerated CSS (first 30 lines):")
        print("-" * 40)
        css_lines = form.css_styles.split('\n')
        for i, line in enumerate(css_lines[:30]):
            print(f"{i+1:2d}: {line}")
        
        if len(css_lines) > 30:
            print(f"... ({len(css_lines) - 30} more lines)")
        
        print(f"\nForm Metadata:")
        print(f"  - Total Fields: {form.metadata['total_fields']}")
        print(f"  - Total Sections: {form.metadata['total_sections']}")
        print(f"  - Framework: {form.framework.value}")
        print(f"  - Layout: {form.layout.value}")
        
    except Exception as e:
        print(f"❌ Form preview failed: {e}")

def demonstrate_validation_schema():
    """Demonstrate validation schema generation"""
    
    print("\n" + "=" * 50)
    print("🛡️  Validation Schema Demo")
    print("=" * 50)
    
    registry = ParameterRegistry()
    generator = SchemaFormGenerator(registry)
    
    try:
        form = generator.generate_form("tbs")
        schema = form.validation_schema
        
        print("Generated JSON Schema (sample):")
        print("-" * 40)
        
        import json
        schema_str = json.dumps(schema, indent=2)
        lines = schema_str.split('\n')
        
        for i, line in enumerate(lines[:40]):
            print(f"{i+1:2d}: {line}")
        
        if len(lines) > 40:
            print(f"... ({len(lines) - 40} more lines)")
        
        print(f"\nSchema Statistics:")
        print(f"  - Properties: {len(schema.get('properties', {}))}")
        print(f"  - Required Fields: {len(schema.get('required', []))}")
        print(f"  - Schema Type: {schema.get('type', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Schema demo failed: {e}")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_form_generation()
    show_form_preview()
    demonstrate_validation_schema()
    
    print("\n✨ All demonstrations complete!")
    print("\nTo use the form generator in your application:")
    print("""
# Basic usage
from configurations.ui import SchemaFormGenerator, FormConfig, FormFramework

generator = SchemaFormGenerator()
config = FormConfig(framework=FormFramework.REACT, layout=LayoutType.TABS)
form = generator.generate_form("tbs", config)

# Save form files
form_path = generator.save_form(form, "/path/to/output")

# Use generated form in your web application
print(form.generated_code)  # Form component code
print(form.css_styles)      # Styling
print(form.javascript_code) # Validation logic
""")