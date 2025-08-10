# DETAILED MIGRATION STRATEGY
## Critical Path Correction and System Integration Plan

**Date:** 2025-01-06  
**Status:** PLANNING PHASE - DETAILED EXECUTION STRATEGY  
**Risk Level:** HIGH - Requires careful execution

---

## MIGRATION OVERVIEW

### Critical Challenge
- **Source:** `/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime` (400+ files)
- **Target:** `/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime` (50+ files)
- **Complexity:** Merge two parallel development streams without losing functionality

### Migration Scope
- **Files to Migrate:** 400+ Python files, configuration files, documentation
- **Import Updates:** 500+ import statements across entire codebase
- **API References:** 50+ external API route references
- **Test Updates:** 100+ test file references

---

## PHASE 1: PRE-MIGRATION ANALYSIS AND BACKUP

### Step 1.1: Complete System Backup
```bash
# Create timestamped backup
BACKUP_DIR="/srv/samba/shared/bt/backtester_stable/BACKUP_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup both directories
cp -r /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime $BACKUP_DIR/
cp -r /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime $BACKUP_DIR/strategies_market_regime

# Backup critical external files
cp -r /srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/ $BACKUP_DIR/api_routes/
```

### Step 1.2: Dependency Mapping
**Critical Import Patterns to Track:**
```python
# Pattern 1: Direct imports
from backtester_v2.market_regime.comprehensive_modules import *
from backtester_v2.market_regime.enhanced_modules import *

# Pattern 2: Relative imports
from .market_regime.strategy import MarketRegimeStrategy
from ..market_regime.excel_config_parser import ExcelConfigParser

# Pattern 3: API route imports
from backtester_v2.market_regime.api_handlers import *
```

### Step 1.3: File Conflict Analysis
**Potential Conflicts Identified:**
- `strategy.py` - Exists in both directories with different implementations
- `excel_config_parser.py` - Different versions with varying functionality
- `__init__.py` - Different module exports
- Test files with same names but different test cases

---

## PHASE 2: INTELLIGENT FILE MIGRATION

### Step 2.1: File Classification and Merge Strategy

**Category A: Source-Only Files (300+ files)**
- **Action:** Direct migration to target directory
- **Files:** All comprehensive modules, legacy implementations, documentation
- **Risk:** Low - No conflicts expected

**Category B: Conflict Files (20+ files)**
- **Action:** Intelligent merge with manual review
- **Files:** `strategy.py`, `excel_config_parser.py`, `__init__.py`
- **Risk:** High - Requires careful merge to preserve functionality

**Category C: Target-Only Files (30+ files)**
- **Action:** Preserve in target directory
- **Files:** Recent enhanced integration work, new UI components
- **Risk:** Low - Keep existing implementations

### Step 2.2: Merge Strategy for Conflict Files

**For `strategy.py`:**
```python
# Target: Enhanced integration with UI endpoints
# Source: Comprehensive production implementation
# Merge: Combine comprehensive logic with enhanced integration
```

**For `excel_config_parser.py`:**
```python
# Target: 31-sheet parser with validation
# Source: Legacy parser with different structure
# Merge: Enhance target with source's production stability
```

**For `__init__.py`:**
```python
# Target: Enhanced module exports
# Source: Comprehensive module exports
# Merge: Combined exports for backward compatibility
```

### Step 2.3: Migration Execution Script

```bash
#!/bin/bash
# migration_script.sh

SOURCE_DIR="/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime"
TARGET_DIR="/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime"

# Step 1: Migrate non-conflict files
rsync -av --exclude="strategy.py" --exclude="excel_config_parser.py" --exclude="__init__.py" \
    $SOURCE_DIR/ $TARGET_DIR/

# Step 2: Handle conflict files (manual merge required)
echo "Manual merge required for conflict files:"
echo "- strategy.py"
echo "- excel_config_parser.py" 
echo "- __init__.py"

# Step 3: Update directory structure
mkdir -p $TARGET_DIR/comprehensive_modules
mkdir -p $TARGET_DIR/enhanced_modules
mkdir -p $TARGET_DIR/integration
mkdir -p $TARGET_DIR/ui
mkdir -p $TARGET_DIR/tests

# Step 4: Organize files into proper structure
mv $TARGET_DIR/comprehensive_*.py $TARGET_DIR/comprehensive_modules/
mv $TARGET_DIR/enhanced_*.py $TARGET_DIR/enhanced_modules/
```

---

## PHASE 3: IMPORT PATH UPDATES

### Step 3.1: Automated Import Update Script

```python
#!/usr/bin/env python3
# update_imports.py

import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Update import statements in a single file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern replacements
    patterns = [
        (r'from backtester_v2\.market_regime\.', 'from backtester_v2.strategies.market_regime.'),
        (r'import backtester_v2\.market_regime\.', 'import backtester_v2.strategies.market_regime.'),
        (r'from \.\.market_regime\.', 'from ..strategies.market_regime.'),
        (r'from \.market_regime\.', 'from .strategies.market_regime.'),
    ]
    
    updated_content = content
    changes_made = 0
    
    for old_pattern, new_pattern in patterns:
        new_content = re.sub(old_pattern, new_pattern, updated_content)
        if new_content != updated_content:
            changes_made += 1
            updated_content = new_content
    
    if changes_made > 0:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        print(f"Updated {changes_made} imports in {file_path}")
    
    return changes_made

def scan_and_update_directory(directory):
    """Scan directory and update all Python files"""
    total_changes = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                changes = update_imports_in_file(file_path)
                total_changes += changes
    
    return total_changes

# Update entire codebase
base_dir = "/srv/samba/shared/bt/backtester_stable/BTRUN"
total_changes = scan_and_update_directory(base_dir)
print(f"Total import updates: {total_changes}")
```

### Step 3.2: API Route Updates

**Files to Update:**
- `/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/market_regime.py`
- `/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/strategy.py`
- `/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/analysis.py`

**Update Pattern:**
```python
# Before
from backtester_v2.market_regime.strategy import MarketRegimeStrategy

# After  
from backtester_v2.strategies.market_regime.strategy import MarketRegimeStrategy
```

### Step 3.3: Configuration File Updates

**Files to Update:**
- All `config.yaml` files
- Docker configuration files
- Environment setup scripts
- Documentation references

---

## PHASE 4: VERIFICATION AND TESTING

### Step 4.1: Import Resolution Verification

```python
#!/usr/bin/env python3
# verify_imports.py

import ast
import os
import importlib.util

def verify_imports_in_file(file_path):
    """Verify all imports in a file can be resolved"""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as e:
                        print(f"Import error in {file_path}: {alias.name} - {e}")
                        return False
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        importlib.import_module(node.module)
                    except ImportError as e:
                        print(f"Import error in {file_path}: {node.module} - {e}")
                        return False
        
        return True
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False

# Verify all Python files
def verify_all_imports(directory):
    failed_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not verify_imports_in_file(file_path):
                    failed_files.append(file_path)
    
    return failed_files
```

### Step 4.2: Functional Testing

**Test Categories:**
1. **Unit Tests:** Verify individual module functionality
2. **Integration Tests:** Verify module interactions
3. **API Tests:** Verify all endpoints respond correctly
4. **End-to-End Tests:** Verify complete system functionality

**Test Execution:**
```bash
# Run comprehensive test suite
cd /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime
python3 -m pytest tests/ -v --tb=short

# Run specific test categories
python3 -m pytest tests/integration_tests/ -v
python3 -m pytest tests/api_tests/ -v
python3 -m pytest tests/end_to_end_tests/ -v
```

### Step 4.3: Performance Validation

**Performance Benchmarks:**
- Market regime analysis: <3 seconds
- Excel configuration loading: <5 seconds
- API response times: <200ms
- Memory usage: <2GB

**Validation Script:**
```python
import time
import psutil
import requests

def performance_test():
    # Test market regime analysis
    start_time = time.time()
    # ... run analysis
    analysis_time = time.time() - start_time
    
    # Test API response times
    api_times = []
    for endpoint in ['/api/market-regime/status', '/api/market-regime/config']:
        start_time = time.time()
        response = requests.get(f"http://localhost:8000{endpoint}")
        api_times.append(time.time() - start_time)
    
    # Test memory usage
    memory_usage = psutil.virtual_memory().used / (1024**3)  # GB
    
    return {
        'analysis_time': analysis_time,
        'api_times': api_times,
        'memory_usage': memory_usage
    }
```

---

## PHASE 5: CLEANUP AND FINALIZATION

### Step 5.1: Remove Duplicate Directory

```bash
# Only after complete verification
rm -rf /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime
```

### Step 5.2: Update Documentation

**Files to Update:**
- README.md files
- API documentation
- Installation guides
- Developer documentation

### Step 5.3: Final Validation

**Checklist:**
- [ ] All imports resolve correctly
- [ ] All tests pass
- [ ] API endpoints functional
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Backup verified

---

## ROLLBACK PROCEDURES

### Immediate Rollback (Emergency)
```bash
# Restore from backup
BACKUP_DIR="/srv/samba/shared/bt/backtester_stable/BACKUP_YYYYMMDD_HHMMSS"
rm -rf /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime
cp -r $BACKUP_DIR/strategies_market_regime /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime
cp -r $BACKUP_DIR/market_regime /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/
```

### Partial Rollback
- Restore specific files from backup
- Revert specific import changes
- Restore specific API routes

---

## EXECUTION TIMELINE

**Day 1: Preparation and Backup (2 hours)**
- Complete system backup
- Dependency mapping
- Conflict analysis

**Day 2: Migration Execution (4 hours)**
- File migration
- Conflict resolution
- Directory restructuring

**Day 3: Import Updates (4 hours)**
- Automated import updates
- API route updates
- Configuration updates

**Day 4: Testing and Verification (6 hours)**
- Import verification
- Functional testing
- Performance validation

**Day 5: Cleanup and Documentation (2 hours)**
- Remove duplicate directory
- Update documentation
- Final validation

**Total Time:** 18 hours over 5 days

---

## APPROVAL CHECKPOINT

This detailed migration strategy provides:
✅ **Step-by-step execution plan**  
✅ **Automated scripts for critical operations**  
✅ **Comprehensive testing strategy**  
✅ **Emergency rollback procedures**  
✅ **Performance validation framework**  

**READY FOR MIGRATION EXECUTION APPROVAL**

---

*This migration strategy ensures zero-downtime, zero-data-loss transition from duplicate directory structure to unified, properly organized codebase.*
