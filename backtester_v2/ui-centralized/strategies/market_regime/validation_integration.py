#!/usr/bin/env python3
"""
Validation Integration Module
============================

Integrates the advanced configuration validator with the progressive upload
system for seamless validation during upload.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

from advanced_config_validator import ConfigurationValidator, ValidationIssue, ValidationSeverity
from progressive_upload_system import ProgressiveValidator

logger = logging.getLogger(__name__)


class IntegratedValidator(ProgressiveValidator):
    """Enhanced validator that integrates advanced configuration validation"""
    
    def __init__(self, excel_parser=None):
        super().__init__(excel_parser)
        self.config_validator = ConfigurationValidator()
        
    async def validate_content(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced content validation using advanced validator"""
        
        # Run advanced validation
        is_valid, issues, metadata = self.config_validator.validate_excel_file(str(file_path))
        
        # Convert issues to structured format
        validation_results = {
            "is_valid": is_valid,
            "metadata": metadata,
            "issues_by_severity": {
                "errors": [],
                "warnings": [],
                "info": []
            },
            "issues_by_category": {},
            "sheets_with_issues": set()
        }
        
        # Organize issues
        for issue in issues:
            # By severity
            if issue.severity == ValidationSeverity.ERROR:
                validation_results["issues_by_severity"]["errors"].append(self._format_issue(issue))
            elif issue.severity == ValidationSeverity.WARNING:
                validation_results["issues_by_severity"]["warnings"].append(self._format_issue(issue))
            else:
                validation_results["issues_by_severity"]["info"].append(self._format_issue(issue))
            
            # By category
            if issue.category not in validation_results["issues_by_category"]:
                validation_results["issues_by_category"][issue.category] = []
            validation_results["issues_by_category"][issue.category].append(self._format_issue(issue))
            
            # Track sheets with issues
            if issue.sheet:
                validation_results["sheets_with_issues"].add(issue.sheet)
        
        # Convert set to list for JSON serialization
        validation_results["sheets_with_issues"] = list(validation_results["sheets_with_issues"])
        
        # Add summary
        validation_results["summary"] = {
            "total_issues": len(issues),
            "error_count": metadata.get("error_count", 0),
            "warning_count": metadata.get("warning_count", 0),
            "info_count": metadata.get("info_count", 0),
            "validation_score": self._calculate_validation_score(issues)
        }
        
        # Generate suggestions
        validation_results["suggestions"] = self._generate_suggestions(issues)
        
        return is_valid, validation_results
    
    def _format_issue(self, issue: ValidationIssue) -> Dict[str, Any]:
        """Format validation issue for JSON serialization"""
        return {
            "severity": issue.severity.value,
            "category": issue.category,
            "sheet": issue.sheet,
            "field": issue.field,
            "message": issue.message,
            "suggestion": issue.suggestion,
            "value": str(issue.value) if issue.value is not None else None,
            "expected": str(issue.expected) if issue.expected is not None else None
        }
    
    def _calculate_validation_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate a validation score from 0-100"""
        if not issues:
            return 100.0
        
        # Weight by severity
        weights = {
            ValidationSeverity.ERROR: 10,
            ValidationSeverity.WARNING: 3,
            ValidationSeverity.INFO: 1
        }
        
        total_weight = sum(weights[issue.severity] for issue in issues)
        max_weight = 100  # Arbitrary maximum for scaling
        
        score = max(0, 100 - (total_weight / max_weight * 100))
        return round(score, 2)
    
    def _generate_suggestions(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable suggestions based on issues"""
        suggestions = []
        
        # Check for common patterns
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        
        if error_count > 0:
            suggestions.append(f"Fix {error_count} critical errors before proceeding")
        
        if warning_count > 5:
            suggestions.append("Consider addressing warnings to improve configuration quality")
        
        # Check for specific issue patterns
        weight_issues = [i for i in issues if 'weight' in i.field.lower() if i.field]
        if len(weight_issues) > 2:
            suggestions.append("Review and adjust weight distributions across components")
        
        threshold_issues = [i for i in issues if 'threshold' in i.field.lower() if i.field]
        if threshold_issues:
            suggestions.append("Verify threshold values are within expected ranges")
        
        # Add download template suggestion if many errors
        if error_count > 3:
            suggestions.append("Consider downloading a template for reference")
        
        return suggestions


class ValidationReportGenerator:
    """Generate detailed validation reports in various formats"""
    
    @staticmethod
    def generate_html_report(validation_results: Dict[str, Any]) -> str:
        """Generate an HTML validation report"""
        
        html = []
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Configuration Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { margin: 20px 0; }
                .score { font-size: 48px; font-weight: bold; }
                .score.good { color: #28a745; }
                .score.warning { color: #ffc107; }
                .score.error { color: #dc3545; }
                .issues { margin: 20px 0; }
                .issue { margin: 10px 0; padding: 10px; border-left: 3px solid; }
                .issue.error { border-color: #dc3545; background: #f8d7da; }
                .issue.warning { border-color: #ffc107; background: #fff3cd; }
                .issue.info { border-color: #17a2b8; background: #d1ecf1; }
                .suggestion { background: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #f0f0f0; }
            </style>
        </head>
        <body>
        """)
        
        # Header
        html.append('<div class="header">')
        html.append('<h1>Configuration Validation Report</h1>')
        html.append(f'<p>Generated: {validation_results["metadata"]["validation_timestamp"]}</p>')
        html.append('</div>')
        
        # Summary
        score = validation_results["summary"]["validation_score"]
        score_class = "good" if score >= 90 else "warning" if score >= 70 else "error"
        
        html.append('<div class="summary">')
        html.append(f'<div class="score {score_class}">{score}%</div>')
        html.append('<h2>Summary</h2>')
        html.append('<table>')
        html.append('<tr><th>Metric</th><th>Value</th></tr>')
        html.append(f'<tr><td>Total Issues</td><td>{validation_results["summary"]["total_issues"]}</td></tr>')
        html.append(f'<tr><td>Errors</td><td>{validation_results["summary"]["error_count"]}</td></tr>')
        html.append(f'<tr><td>Warnings</td><td>{validation_results["summary"]["warning_count"]}</td></tr>')
        html.append(f'<tr><td>Info</td><td>{validation_results["summary"]["info_count"]}</td></tr>')
        html.append(f'<tr><td>Sheets Analyzed</td><td>{validation_results["metadata"]["sheets_analyzed"]}</td></tr>')
        html.append('</table>')
        html.append('</div>')
        
        # Issues by severity
        for severity, issues in validation_results["issues_by_severity"].items():
            if issues:
                html.append(f'<div class="issues">')
                html.append(f'<h2>{severity.title()} ({len(issues)})</h2>')
                for issue in issues:
                    html.append(f'<div class="issue {severity.rstrip("s")}">')
                    html.append(f'<strong>[{issue["category"].upper()}]</strong> ')
                    if issue["sheet"]:
                        html.append(f'{issue["sheet"]} - ')
                    if issue["field"]:
                        html.append(f'{issue["field"]}<br>')
                    html.append(f'{issue["message"]}<br>')
                    if issue["suggestion"]:
                        html.append(f'<em>Suggestion: {issue["suggestion"]}</em>')
                    html.append('</div>')
                html.append('</div>')
        
        # Suggestions
        if validation_results["suggestions"]:
            html.append('<div class="suggestions">')
            html.append('<h2>Recommendations</h2>')
            for suggestion in validation_results["suggestions"]:
                html.append(f'<div class="suggestion">💡 {suggestion}</div>')
            html.append('</div>')
        
        html.append('</body></html>')
        
        return '\n'.join(html)
    
    @staticmethod
    def generate_json_report(validation_results: Dict[str, Any]) -> str:
        """Generate a JSON validation report"""
        return json.dumps(validation_results, indent=2)
    
    @staticmethod
    def generate_markdown_report(validation_results: Dict[str, Any]) -> str:
        """Generate a Markdown validation report"""
        
        md = []
        md.append("# Configuration Validation Report")
        md.append(f"\n**Generated**: {validation_results['metadata']['validation_timestamp']}")
        md.append(f"\n**Validation Score**: {validation_results['summary']['validation_score']}%")
        
        # Summary
        md.append("\n## Summary")
        md.append(f"- Total Issues: {validation_results['summary']['total_issues']}")
        md.append(f"- Errors: {validation_results['summary']['error_count']}")
        md.append(f"- Warnings: {validation_results['summary']['warning_count']}")
        md.append(f"- Info: {validation_results['summary']['info_count']}")
        
        # Issues
        for severity, issues in validation_results["issues_by_severity"].items():
            if issues:
                md.append(f"\n## {severity.title()}")
                for i, issue in enumerate(issues, 1):
                    md.append(f"\n### {i}. [{issue['category'].upper()}] {issue['sheet'] or 'Global'}")
                    if issue['field']:
                        md.append(f"**Field**: {issue['field']}")
                    md.append(f"\n{issue['message']}")
                    if issue['suggestion']:
                        md.append(f"\n> 💡 {issue['suggestion']}")
        
        # Recommendations
        if validation_results["suggestions"]:
            md.append("\n## Recommendations")
            for suggestion in validation_results["suggestions"]:
                md.append(f"- {suggestion}")
        
        return '\n'.join(md)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_validation():
        # Create integrated validator
        validator = IntegratedValidator()
        
        # Test with sample file
        test_file = Path("MARKET_REGIME_SAMPLE_CONFIG.xlsx")
        if test_file.exists():
            is_valid, results = await validator.validate_content(test_file)
            
            print(f"\n✅ Validation Complete")
            print(f"Valid: {is_valid}")
            print(f"Score: {results['summary']['validation_score']}%")
            
            # Generate reports
            report_gen = ValidationReportGenerator()
            
            # Save HTML report
            with open("validation_report.html", "w") as f:
                f.write(report_gen.generate_html_report(results))
            print("\n📄 HTML report saved: validation_report.html")
            
            # Save JSON report
            with open("validation_report.json", "w") as f:
                f.write(report_gen.generate_json_report(results))
            print("📄 JSON report saved: validation_report.json")
            
            # Save Markdown report
            with open("validation_report.md", "w") as f:
                f.write(report_gen.generate_markdown_report(results))
            print("📄 Markdown report saved: validation_report.md")
    
    # Run test
    asyncio.run(test_validation())