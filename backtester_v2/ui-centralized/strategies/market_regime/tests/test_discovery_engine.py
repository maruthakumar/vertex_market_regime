#!/usr/bin/env python3
"""
Test Discovery Engine

PHASE 6.2: Automated test discovery and execution
- Automatically discovers all test files and test cases
- Provides intelligent test categorization and prioritization
- Manages test dependencies and execution order
- Supports filtering and selective test execution
- NO MOCK DATA - validates real test implementations

Author: Claude Code
Date: 2025-07-12
Version: 1.0.0 - PHASE 6.2 TEST DISCOVERY ENGINE
"""

import unittest
import sys
import os
import ast
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TestCategory(Enum):
    """Test categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    END_TO_END = "end_to_end"

@dataclass
class TestMethodInfo:
    """Information about a test method"""
    name: str
    description: str
    test_class: str
    test_file: str
    line_number: int
    priority: TestPriority
    category: TestCategory
    estimated_duration: float
    dependencies: List[str]
    tags: Set[str]

@dataclass
class TestClassInfo:
    """Information about a test class"""
    name: str
    description: str
    test_file: str
    line_number: int
    test_methods: List[TestMethodInfo]
    setup_methods: List[str]
    teardown_methods: List[str]
    priority: TestPriority
    category: TestCategory

@dataclass
class TestFileInfo:
    """Information about a test file"""
    file_path: str
    file_name: str
    description: str
    test_classes: List[TestClassInfo]
    imports: List[str]
    phase: str
    priority: TestPriority
    estimated_duration: float
    dependencies: List[str]

class TestDiscoveryEngine:
    """
    Intelligent Test Discovery Engine
    
    Automatically discovers, analyzes, and categorizes all test files
    and test cases in the Market Regime Strategy test suite.
    """
    
    def __init__(self, test_directory: str = None):
        """Initialize the test discovery engine"""
        self.test_directory = Path(test_directory) if test_directory else Path(__file__).parent
        self.discovered_tests = {}
        self.test_files = []
        self.test_classes = []
        self.test_methods = []
        
        # Test categorization patterns
        self.category_patterns = {
            TestCategory.UNIT: [
                r'unit.*test', r'test.*unit', r'.*_unit_.*',
                r'config.*test', r'parameter.*test'
            ],
            TestCategory.INTEGRATION: [
                r'integration.*test', r'test.*integration', r'.*_integration_.*',
                r'module.*test', r'cross.*module', r'pipeline.*test'
            ],
            TestCategory.PERFORMANCE: [
                r'performance.*test', r'test.*performance', r'.*_performance_.*',
                r'load.*test', r'stress.*test', r'benchmark.*test'
            ],
            TestCategory.VALIDATION: [
                r'validation.*test', r'test.*validation', r'.*_validation_.*',
                r'production.*test', r'scenario.*test'
            ],
            TestCategory.END_TO_END: [
                r'end.*to.*end', r'e2e.*test', r'test.*e2e',
                r'system.*test', r'complete.*test'
            ]
        }
        
        # Priority patterns
        self.priority_patterns = {
            TestPriority.CRITICAL: [
                r'critical', r'core', r'essential', r'basic',
                r'config', r'integration', r'system'
            ],
            TestPriority.HIGH: [
                r'important', r'key', r'main', r'primary',
                r'performance', r'validation'
            ],
            TestPriority.MEDIUM: [
                r'medium', r'standard', r'normal',
                r'additional', r'extended'
            ],
            TestPriority.LOW: [
                r'optional', r'extra', r'supplementary',
                r'edge.*case', r'corner.*case'
            ]
        }
        
        # Phase patterns
        self.phase_patterns = {
            'phase1': [r'excel', r'config', r'configuration'],
            'phase2': [r'heavydb', r'database', r'db', r'strict'],
            'phase3': [r'correlation', r'matrix', r'10x10'],
            'phase4': [r'integration', r'module', r'communication', r'pipeline', r'error'],
            'phase5': [r'performance', r'validation', r'production', r'load'],
            'phase6': [r'automated', r'suite', r'discovery', r'reporting']
        }
        
        logger.info(f"📍 Test Discovery Engine initialized")
        logger.info(f"📂 Scanning directory: {self.test_directory}")
    
    def discover_all_tests(self) -> Dict[str, Any]:
        """Discover all test files, classes, and methods"""
        logger.info("🔍 Starting comprehensive test discovery...")
        
        # Find all Python test files
        test_files = self._find_test_files()
        
        # Analyze each test file
        analyzed_files = []
        for test_file in test_files:
            file_info = self._analyze_test_file(test_file)
            if file_info:
                analyzed_files.append(file_info)
        
        # Build discovery results
        discovery_results = {
            'discovery_timestamp': f"{os.path.getmtime(__file__)}",
            'test_directory': str(self.test_directory),
            'total_test_files': len(analyzed_files),
            'total_test_classes': sum(len(f.test_classes) for f in analyzed_files),
            'total_test_methods': sum(len(c.test_methods) for f in analyzed_files for c in f.test_classes),
            'test_files': analyzed_files,
            'categorization': self._categorize_tests(analyzed_files),
            'dependency_graph': self._build_dependency_graph(analyzed_files),
            'execution_plan': self._create_execution_plan(analyzed_files)
        }
        
        self.discovered_tests = discovery_results
        
        logger.info(f"✅ Discovery complete:")
        logger.info(f"   📁 Test files: {discovery_results['total_test_files']}")
        logger.info(f"   🏷️  Test classes: {discovery_results['total_test_classes']}")
        logger.info(f"   🧪 Test methods: {discovery_results['total_test_methods']}")
        
        return discovery_results
    
    def _find_test_files(self) -> List[Path]:
        """Find all test files in the directory"""
        test_files = []
        
        # Standard test file patterns
        patterns = ['test_*.py', '*_test.py', '*_tests.py']
        
        for pattern in patterns:
            test_files.extend(self.test_directory.glob(pattern))
        
        # Remove duplicates and non-test files
        unique_files = []
        for file_path in test_files:
            if file_path.name not in ['__init__.py', 'conftest.py'] and file_path not in unique_files:
                unique_files.append(file_path)
        
        logger.info(f"📁 Found {len(unique_files)} test files")
        return unique_files
    
    def _analyze_test_file(self, file_path: Path) -> Optional[TestFileInfo]:
        """Analyze a single test file"""
        try:
            logger.debug(f"🔍 Analyzing: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract file-level information
            file_info = TestFileInfo(
                file_path=str(file_path),
                file_name=file_path.name,
                description=self._extract_file_description(tree),
                test_classes=[],
                imports=self._extract_imports(tree),
                phase=self._determine_phase(file_path.name),
                priority=self._determine_priority(file_path.name),
                estimated_duration=0.0,
                dependencies=[]
            )
            
            # Analyze test classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_test_class(node, file_path, content)
                    if class_info:
                        file_info.test_classes.append(class_info)
            
            # Calculate estimated duration
            total_methods = sum(len(c.test_methods) for c in file_info.test_classes)
            file_info.estimated_duration = total_methods * 5.0  # 5 seconds per test method estimate
            
            return file_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path.name}: {e}")
            return None
    
    def _analyze_test_class(self, class_node: ast.ClassDef, file_path: Path, content: str) -> Optional[TestClassInfo]:
        """Analyze a test class"""
        try:
            # Check if it's a test class
            if not self._is_test_class(class_node):
                return None
            
            class_info = TestClassInfo(
                name=class_node.name,
                description=self._extract_class_description(class_node),
                test_file=str(file_path),
                line_number=class_node.lineno,
                test_methods=[],
                setup_methods=[],
                teardown_methods=[],
                priority=self._determine_priority(class_node.name),
                category=self._determine_category(class_node.name)
            )
            
            # Analyze methods
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef):
                    if self._is_test_method(node):
                        method_info = self._analyze_test_method(node, class_node.name, file_path)
                        if method_info:
                            class_info.test_methods.append(method_info)
                    elif self._is_setup_method(node):
                        class_info.setup_methods.append(node.name)
                    elif self._is_teardown_method(node):
                        class_info.teardown_methods.append(node.name)
            
            return class_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze class {class_node.name}: {e}")
            return None
    
    def _analyze_test_method(self, method_node: ast.FunctionDef, class_name: str, file_path: Path) -> Optional[TestMethodInfo]:
        """Analyze a test method"""
        try:
            method_info = TestMethodInfo(
                name=method_node.name,
                description=self._extract_method_description(method_node),
                test_class=class_name,
                test_file=str(file_path),
                line_number=method_node.lineno,
                priority=self._determine_priority(method_node.name),
                category=self._determine_category(method_node.name),
                estimated_duration=5.0,  # Default 5 seconds
                dependencies=[],
                tags=self._extract_tags(method_node)
            )
            
            return method_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze method {method_node.name}: {e}")
            return None
    
    def _is_test_class(self, class_node: ast.ClassDef) -> bool:
        """Check if a class is a test class"""
        # Check class name
        if class_node.name.startswith('Test') or class_node.name.endswith('Test'):
            return True
        
        # Check inheritance
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if 'TestCase' in base.id or 'Test' in base.id:
                    return True
            elif isinstance(base, ast.Attribute):
                if 'TestCase' in base.attr or 'Test' in base.attr:
                    return True
        
        return False
    
    def _is_test_method(self, method_node: ast.FunctionDef) -> bool:
        """Check if a method is a test method"""
        return method_node.name.startswith('test_')
    
    def _is_setup_method(self, method_node: ast.FunctionDef) -> bool:
        """Check if a method is a setup method"""
        return method_node.name in ['setUp', 'setUpClass', 'setup_method', 'setup_class']
    
    def _is_teardown_method(self, method_node: ast.FunctionDef) -> bool:
        """Check if a method is a teardown method"""
        return method_node.name in ['tearDown', 'tearDownClass', 'teardown_method', 'teardown_class']
    
    def _extract_file_description(self, tree: ast.AST) -> str:
        """Extract file description from docstring"""
        if isinstance(tree, ast.Module) and tree.body:
            first_node = tree.body[0]
            if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Str):
                return first_node.value.s.strip()
        return ""
    
    def _extract_class_description(self, class_node: ast.ClassDef) -> str:
        """Extract class description from docstring"""
        if class_node.body and isinstance(class_node.body[0], ast.Expr):
            if isinstance(class_node.body[0].value, ast.Str):
                return class_node.body[0].value.s.strip()
        return ""
    
    def _extract_method_description(self, method_node: ast.FunctionDef) -> str:
        """Extract method description from docstring"""
        if method_node.body and isinstance(method_node.body[0], ast.Expr):
            if isinstance(method_node.body[0].value, ast.Str):
                return method_node.body[0].value.s.strip()
        return ""
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    def _extract_tags(self, method_node: ast.FunctionDef) -> Set[str]:
        """Extract tags from method decorators and docstring"""
        tags = set()
        
        # Check decorators
        for decorator in method_node.decorator_list:
            if isinstance(decorator, ast.Name):
                tags.add(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                tags.add(decorator.func.id)
        
        # Extract from docstring
        docstring = self._extract_method_description(method_node)
        if 'performance' in docstring.lower():
            tags.add('performance')
        if 'integration' in docstring.lower():
            tags.add('integration')
        if 'validation' in docstring.lower():
            tags.add('validation')
        
        return tags
    
    def _determine_phase(self, name: str) -> str:
        """Determine which phase a test belongs to"""
        name_lower = name.lower()
        
        for phase, patterns in self.phase_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return phase
        
        return 'unknown'
    
    def _determine_priority(self, name: str) -> TestPriority:
        """Determine test priority based on name patterns"""
        name_lower = name.lower()
        
        for priority, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return priority
        
        return TestPriority.MEDIUM
    
    def _determine_category(self, name: str) -> TestCategory:
        """Determine test category based on name patterns"""
        name_lower = name.lower()
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return category
        
        return TestCategory.UNIT
    
    def _categorize_tests(self, test_files: List[TestFileInfo]) -> Dict[str, Any]:
        """Categorize tests by various criteria"""
        categorization = {
            'by_phase': {},
            'by_priority': {},
            'by_category': {},
            'by_estimated_duration': {},
            'by_file_count': {}
        }
        
        # By phase
        for file_info in test_files:
            phase = file_info.phase
            if phase not in categorization['by_phase']:
                categorization['by_phase'][phase] = []
            categorization['by_phase'][phase].append(file_info.file_name)
        
        # By priority
        for file_info in test_files:
            priority = file_info.priority.value
            if priority not in categorization['by_priority']:
                categorization['by_priority'][priority] = []
            categorization['by_priority'][priority].append(file_info.file_name)
        
        # By category
        for file_info in test_files:
            for class_info in file_info.test_classes:
                category = class_info.category.value
                if category not in categorization['by_category']:
                    categorization['by_category'][category] = []
                categorization['by_category'][category].append(f"{file_info.file_name}::{class_info.name}")
        
        # By duration
        duration_ranges = [
            ('quick', 0, 30),      # < 30 seconds
            ('medium', 30, 120),   # 30s - 2 minutes
            ('long', 120, 300),    # 2 - 5 minutes
            ('very_long', 300, float('inf'))  # > 5 minutes
        ]
        
        for range_name, min_dur, max_dur in duration_ranges:
            categorization['by_estimated_duration'][range_name] = []
            
        for file_info in test_files:
            duration = file_info.estimated_duration
            for range_name, min_dur, max_dur in duration_ranges:
                if min_dur <= duration < max_dur:
                    categorization['by_estimated_duration'][range_name].append(file_info.file_name)
                    break
        
        return categorization
    
    def _build_dependency_graph(self, test_files: List[TestFileInfo]) -> Dict[str, List[str]]:
        """Build a dependency graph for test execution order"""
        dependency_graph = {}
        
        # Simple dependency analysis based on imports and naming patterns
        for file_info in test_files:
            file_name = file_info.file_name
            dependencies = []
            
            # Check for phase dependencies
            if 'phase4' in file_name or 'integration' in file_name:
                # Integration tests depend on basic tests
                for other_file in test_files:
                    if ('config' in other_file.file_name or 
                        'excel' in other_file.file_name or
                        'phase1' in other_file.file_name):
                        dependencies.append(other_file.file_name)
            
            elif 'phase5' in file_name or 'performance' in file_name:
                # Performance tests depend on integration tests
                for other_file in test_files:
                    if ('integration' in other_file.file_name or
                        'phase4' in other_file.file_name):
                        dependencies.append(other_file.file_name)
            
            dependency_graph[file_name] = dependencies
        
        return dependency_graph
    
    def _create_execution_plan(self, test_files: List[TestFileInfo]) -> Dict[str, Any]:
        """Create an optimized test execution plan"""
        execution_plan = {
            'phases': [],
            'total_estimated_time': 0,
            'parallel_groups': [],
            'sequential_order': []
        }
        
        # Group by phases
        phase_groups = {}
        for file_info in test_files:
            phase = file_info.phase
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(file_info)
        
        # Create execution phases
        for phase in sorted(phase_groups.keys()):
            if phase == 'unknown':
                continue
                
            phase_files = phase_groups[phase]
            phase_duration = sum(f.estimated_duration for f in phase_files)
            
            execution_plan['phases'].append({
                'phase': phase,
                'files': [f.file_name for f in phase_files],
                'estimated_duration': phase_duration,
                'priority': max(f.priority.value for f in phase_files),
                'can_run_parallel': len(phase_files) > 1 and phase_duration < 300
            })
        
        # Calculate total time
        execution_plan['total_estimated_time'] = sum(
            phase['estimated_duration'] for phase in execution_plan['phases']
        )
        
        # Create parallel groups for independent tests
        for phase_info in execution_plan['phases']:
            if phase_info['can_run_parallel']:
                execution_plan['parallel_groups'].append(phase_info['files'])
            else:
                execution_plan['sequential_order'].extend(phase_info['files'])
        
        return execution_plan
    
    def filter_tests(self, 
                    phases: List[str] = None,
                    priorities: List[TestPriority] = None,
                    categories: List[TestCategory] = None,
                    tags: List[str] = None,
                    max_duration: float = None) -> Dict[str, Any]:
        """Filter discovered tests based on criteria"""
        
        if not self.discovered_tests:
            logger.warning("No tests discovered yet. Run discover_all_tests() first.")
            return {}
        
        filtered_files = []
        
        for file_info in self.discovered_tests['test_files']:
            # Phase filter
            if phases and file_info.phase not in phases:
                continue
            
            # Priority filter
            if priorities and file_info.priority not in priorities:
                continue
            
            # Duration filter
            if max_duration and file_info.estimated_duration > max_duration:
                continue
            
            # Category and tag filters (check classes/methods)
            file_matches = False
            
            for class_info in file_info.test_classes:
                # Category filter
                if categories and class_info.category not in categories:
                    continue
                
                # Tag filter
                if tags:
                    class_has_tags = False
                    for method_info in class_info.test_methods:
                        if any(tag in method_info.tags for tag in tags):
                            class_has_tags = True
                            break
                    if not class_has_tags:
                        continue
                
                file_matches = True
                break
            
            if not file_info.test_classes:  # No classes, include file
                file_matches = True
            
            if file_matches:
                filtered_files.append(file_info)
        
        # Create filtered results
        filtered_results = {
            'filter_criteria': {
                'phases': phases,
                'priorities': [p.value if p else None for p in priorities] if priorities else None,
                'categories': [c.value if c else None for c in categories] if categories else None,
                'tags': tags,
                'max_duration': max_duration
            },
            'filtered_files': filtered_files,
            'total_filtered_files': len(filtered_files),
            'total_filtered_classes': sum(len(f.test_classes) for f in filtered_files),
            'total_filtered_methods': sum(len(c.test_methods) for f in filtered_files for c in f.test_classes),
            'estimated_execution_time': sum(f.estimated_duration for f in filtered_files)
        }
        
        logger.info(f"🔍 Filtered tests: {len(filtered_files)} files match criteria")
        
        return filtered_results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered tests"""
        if not self.discovered_tests:
            return {'error': 'No tests discovered'}
        
        return {
            'total_files': self.discovered_tests['total_test_files'],
            'total_classes': self.discovered_tests['total_test_classes'],
            'total_methods': self.discovered_tests['total_test_methods'],
            'estimated_total_time': self.discovered_tests['execution_plan']['total_estimated_time'],
            'phases_found': list(self.discovered_tests['categorization']['by_phase'].keys()),
            'priorities_distribution': {
                priority: len(files) 
                for priority, files in self.discovered_tests['categorization']['by_priority'].items()
            },
            'categories_distribution': {
                category: len(items)
                for category, items in self.discovered_tests['categorization']['by_category'].items()
            }
        }

def run_test_discovery():
    """Main function to run test discovery"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🔍 PHASE 6.2: TEST DISCOVERY ENGINE")
    logger.info("=" * 50)
    
    # Initialize discovery engine
    discovery_engine = TestDiscoveryEngine()
    
    # Discover all tests
    results = discovery_engine.discover_all_tests()
    
    # Print summary
    summary = discovery_engine.get_test_summary()
    logger.info("\n📊 DISCOVERY SUMMARY:")
    logger.info(f"Files: {summary['total_files']}")
    logger.info(f"Classes: {summary['total_classes']}")
    logger.info(f"Methods: {summary['total_methods']}")
    logger.info(f"Estimated Time: {summary['estimated_total_time']:.1f}s")
    
    return results

if __name__ == "__main__":
    run_test_discovery()