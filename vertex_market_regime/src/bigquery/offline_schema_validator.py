#!/usr/bin/env python3
"""
Offline BigQuery DDL Schema Validator
Validates DDL syntax and structure without requiring BigQuery connection
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class OfflineSchemaValidator:
    """Validates BigQuery DDL schemas offline"""
    
    def __init__(self):
        self.ddl_dir = Path(__file__).parent / "ddl"
        self.validation_results = {}
        
        # Expected Phase 2 feature counts
        self.expected_features = {
            "c1_features": 150,  # Phase 2: 120 + 30 momentum
            "c2_features": 98,   # Unchanged
            "c3_features": 105,  # Unchanged
            "c4_features": 87,   # Unchanged  
            "c5_features": 94,   # Unchanged
            "c6_features": 220,  # Phase 2: 200 + 20 momentum-enhanced correlation
            "c7_features": 130,  # Phase 2: 120 + 10 momentum-based levels
            "c8_features": 48,   # Unchanged
            "training_dataset": 932  # Phase 2 total
        }
    
    def validate_all_ddls(self, environment: str = "dev") -> Dict[str, Dict]:
        """
        Validate all DDL files with offline syntax checking
        
        Args:
            environment: Environment (dev/staging/prod)
            
        Returns:
            Validation results for each DDL file
        """
        results = {}
        ddl_files = list(self.ddl_dir.glob("*.sql"))
        
        print(f"Found {len(ddl_files)} DDL files to validate")
        print("-" * 50)
        
        for ddl_file in ddl_files:
            file_name = ddl_file.name
            print(f"\nValidating: {file_name}")
            
            # Read DDL content
            with open(ddl_file, 'r') as f:
                ddl_content = f.read()
            
            # Replace environment placeholder
            ddl_content = ddl_content.replace("{env}", environment)
            
            # Validate DDL
            validation_result = self.validate_single_ddl(ddl_content, file_name)
            results[file_name] = validation_result
            
            # Print result
            if validation_result["valid"]:
                print(f"  ✓ Valid DDL")
                print(f"    - Feature count: {validation_result.get('feature_count', 'N/A')}")
                print(f"    - Partitioning: {validation_result.get('partitioning', 'N/A')}")
                print(f"    - Clustering: {validation_result.get('clustering', 'N/A')}")
            else:
                print(f"  ✗ Invalid DDL")
                print(f"    - Error: {validation_result.get('error', 'Unknown error')}")
        
        return results
    
    def validate_single_ddl(self, ddl_content: str, file_name: str) -> Dict:
        """
        Validate a single DDL statement offline
        
        Args:
            ddl_content: DDL SQL content
            file_name: Name of the DDL file
            
        Returns:
            Validation result dictionary
        """
        result = {
            "file": file_name,
            "valid": False,
            "error": None,
            "partitioning": None,
            "clustering": None,
            "feature_count": None
        }
        
        try:
            # Extract table name
            table_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?`([^`]+)`', ddl_content, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
                result["table_name"] = table_name
            
            # Extract partitioning
            partition_match = re.search(r'PARTITION\s+BY\s+([\w()]+)', ddl_content, re.IGNORECASE)
            if partition_match:
                result["partitioning"] = partition_match.group(1)
            
            # Extract clustering
            cluster_match = re.search(r'CLUSTER\s+BY\s+([^;)]+)', ddl_content, re.IGNORECASE)
            if cluster_match:
                result["clustering"] = cluster_match.group(1).strip()
            
            # Count features (columns starting with c1_, c2_, etc.)
            feature_matches = re.findall(r'c\d+_\w+', ddl_content)
            result["feature_count"] = len(set(feature_matches))
            
            # Validate structure
            result["valid"] = self._validate_ddl_structure(ddl_content)
            
            if not result["valid"]:
                result["error"] = "DDL structure validation failed"
                
        except Exception as e:
            result["error"] = f"Validation error: {str(e)}"
        
        return result
    
    def _validate_ddl_structure(self, ddl_content: str) -> bool:
        """
        Validate DDL structure without BigQuery connection
        
        Args:
            ddl_content: DDL SQL content
            
        Returns:
            True if structure is valid
        """
        # Check for required elements
        required_patterns = [
            r'CREATE\s+(OR\s+REPLACE\s+)?(TABLE|VIEW)',  # CREATE statement
            r'`[^`]+\.[^`]+\.[^`]+`',  # Fully qualified table name
            r'\w+\s+\w+',  # At least one column definition
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, ddl_content, re.IGNORECASE):
                return False
        
        # Check for common syntax errors
        open_parens = ddl_content.count('(')
        close_parens = ddl_content.count(')')
        if open_parens != close_parens:
            return False
        
        return True
    
    def validate_feature_counts(self, results: Dict[str, Dict]) -> Dict[str, bool]:
        """Validate feature counts match expected Phase 2 counts"""
        
        validation = {}
        
        for file_name, result in results.items():
            if result["valid"]:
                table_name = file_name.replace('.sql', '')
                expected_count = self.expected_features.get(table_name, 0)
                actual_count = result.get("feature_count", 0)
                
                if table_name == "training_dataset":
                    # Training dataset is a view, skip feature count validation
                    validation[table_name] = True
                elif expected_count > 0:
                    validation[table_name] = actual_count == expected_count
                    if actual_count != expected_count:
                        print(f"  ⚠️ {table_name}: Expected {expected_count} features, found {actual_count}")
                    else:
                        print(f"  ✓ {table_name}: Feature count validated ({actual_count} features)")
                else:
                    validation[table_name] = True
            else:
                validation[file_name] = False
        
        return validation
    
    def generate_validation_report(self, results: Dict[str, Dict]) -> None:
        """Generate comprehensive validation report"""
        
        print("\n" + "=" * 70)
        print("📋 STORY 2.2 DDL VALIDATION REPORT")
        print("=" * 70)
        
        valid_ddls = sum(1 for r in results.values() if r["valid"])
        total_ddls = len(results)
        print(f"📊 DDL Files: {valid_ddls}/{total_ddls} valid")
        
        # Feature count validation
        feature_validation = self.validate_feature_counts(results)
        valid_features = sum(1 for v in feature_validation.values() if v)
        print(f"📊 Feature Counts: {valid_features}/{len(feature_validation)} validated")
        
        # Phase 2 summary
        print(f"\n🚀 Phase 2 Enhanced Components:")
        print(f"  • Component 1: 150 features (120 + 30 momentum)")
        print(f"  • Component 6: 220 features (200 + 20 momentum-enhanced)")
        print(f"  • Component 7: 130 features (120 + 10 momentum-based)")
        print(f"  • Total System: 932 features")
        
        if valid_ddls == total_ddls and valid_features == len(feature_validation):
            print(f"\n🎉 Story 2.2 DDL validation PASSED!")
            print(f"   All schemas are ready for BigQuery deployment.")
        else:
            print(f"\n⚠️ Story 2.2 DDL validation needs attention")
            print(f"   Review failed validations above.")
        
        print("=" * 70)


def main():
    """Main validation function"""
    validator = OfflineSchemaValidator()
    
    # Validate all DDLs
    results = validator.validate_all_ddls("dev")
    
    # Generate report
    validator.generate_validation_report(results)
    
    # Return success status
    all_valid = all(r["valid"] for r in results.values())
    return all_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)