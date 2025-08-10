#!/usr/bin/env python3
"""
Script to fix hardcoded paths in the Market Regime system

This script will:
1. Scan all Python files for hardcoded paths
2. Replace them with configuration-based paths
3. Add necessary imports for the configuration manager
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathFixer:
    """Fix hardcoded paths in Python files"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.files_modified = 0
        self.paths_replaced = 0
        
        # Pattern to match hardcoded paths
        self.path_patterns = [
            # Full paths to input_sheets
            (r'/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/[^"\']+\.xlsx',
             'config_manager.get_excel_config_path("{filename}")'),
            
            # Directory paths to input_sheets
            (r'/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime',
             'config_manager.paths.get_input_sheets_path()'),
            
            # General input_sheets references
            (r'["\']input_sheets/market_regime["\']',
             'config_manager.paths.input_sheets_dir'),
            
            # Hardcoded base paths
            (r'/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime',
             'config_manager.paths.get_strategies_path()'),
            
            # Template directory paths
            (r'["\']enhanced_regime_analysis["\']',
             'config_manager.paths.output_dir'),
        ]
        
        # Files to exclude from modification
        self.exclude_files = {
            'fix_hardcoded_paths.py',
            'config_manager.py',
            '__pycache__'
        }
    
    def should_process_file(self, filepath: Path) -> bool:
        """Check if file should be processed"""
        # Skip if in exclude list
        if filepath.name in self.exclude_files:
            return False
        
        # Skip if in __pycache__ directory
        if '__pycache__' in filepath.parts:
            return False
        
        # Only process Python files
        return filepath.suffix == '.py'
    
    def fix_file(self, filepath: Path) -> bool:
        """Fix hardcoded paths in a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # Check if config_manager import is needed
            needs_import = False
            
            # Replace each pattern
            for pattern, replacement in self.path_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Extract filename from full path matches
                    for match in matches:
                        if '.xlsx' in match and '{filename}' in replacement:
                            filename = os.path.basename(match)
                            new_replacement = replacement.replace('{filename}', filename)
                            content = content.replace(f'"{match}"', new_replacement)
                            content = content.replace(f"'{match}'", new_replacement)
                        else:
                            content = re.sub(f'["\']?{re.escape(match)}["\']?', replacement, content)
                    
                    needs_import = True
                    modified = True
                    self.paths_replaced += len(matches)
            
            # Add import if needed and not already present
            if needs_import and 'from .config_manager import' not in content and 'from config_manager import' not in content:
                # Find the right place to add import (after other imports)
                import_lines = []
                other_lines = []
                in_imports = True
                
                for line in content.split('\n'):
                    if in_imports and (line.startswith('import ') or line.startswith('from ')):
                        import_lines.append(line)
                    else:
                        if in_imports and line.strip() and not line.startswith('#'):
                            # Add our import before first non-import line
                            import_lines.append('from .config_manager import get_config_manager')
                            import_lines.append('config_manager = get_config_manager()')
                            in_imports = False
                        other_lines.append(line)
                
                content = '\n'.join(import_lines + other_lines)
                modified = True
            
            # Write back if modified
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_modified += 1
                logger.info(f"Modified: {filepath.relative_to(self.root_dir)}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return False
    
    def fix_all_files(self):
        """Fix all Python files in the directory tree"""
        logger.info(f"Scanning for hardcoded paths in: {self.root_dir}")
        
        for filepath in self.root_dir.rglob('*.py'):
            if self.should_process_file(filepath):
                self.fix_file(filepath)
        
        logger.info(f"Completed: Modified {self.files_modified} files, replaced {self.paths_replaced} paths")
    
    def generate_report(self) -> Dict[str, List[str]]:
        """Generate a report of files with hardcoded paths"""
        report = {
            'files_with_paths': [],
            'path_occurrences': {}
        }
        
        for filepath in self.root_dir.rglob('*.py'):
            if self.should_process_file(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, _ in self.path_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            rel_path = str(filepath.relative_to(self.root_dir))
                            report['files_with_paths'].append(rel_path)
                            
                            for match in matches:
                                if match not in report['path_occurrences']:
                                    report['path_occurrences'][match] = []
                                report['path_occurrences'][match].append(rel_path)
                            break
                
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
        
        return report

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix hardcoded paths in Market Regime system')
    parser.add_argument('--dry-run', action='store_true', help='Only report, do not modify files')
    parser.add_argument('--root', default='/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime',
                       help='Root directory to scan')
    
    args = parser.parse_args()
    
    fixer = PathFixer(args.root)
    
    if args.dry_run:
        logger.info("Running in dry-run mode - no files will be modified")
        report = fixer.generate_report()
        
        print("\nFiles with hardcoded paths:")
        for filepath in sorted(set(report['files_with_paths'])):
            print(f"  - {filepath}")
        
        print("\nPath occurrences:")
        for path, files in report['path_occurrences'].items():
            print(f"\n  {path}:")
            for filepath in sorted(set(files)):
                print(f"    - {filepath}")
    else:
        fixer.fix_all_files()
        
        # Also create a sample configuration file
        logger.info("\nCreating sample configuration file...")
        from config_manager import get_config_manager
        config = get_config_manager()
        config.save_config(os.path.join(args.root, 'config', 'market_regime_config.json'))
        logger.info("Sample configuration saved to config/market_regime_config.json")

if __name__ == '__main__':
    main()