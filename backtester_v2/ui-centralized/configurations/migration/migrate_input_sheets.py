"""
Migration script for input sheets to new configuration system
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd

from ..core import ConfigurationManager
from ..parsers import ExcelParser

logger = logging.getLogger(__name__)

class InputSheetMigrator:
    """
    Migrates existing input sheets to the new configuration system
    """
    
    def __init__(self, source_dir: str, target_dir: str):
        """
        Initialize migrator
        
        Args:
            source_dir: Directory containing existing input sheets
            target_dir: Target directory for migrated configurations
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.config_manager = ConfigurationManager()
        self.parser = ExcelParser()
        
        # Migration statistics
        self.stats = {
            'total_files': 0,
            'migrated': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        # Strategy mapping
        self.strategy_mapping = {
            'tbs': 'TBS',
            'tv': 'TradingView',
            'orb': 'Opening Range Breakout',
            'oi': 'Open Interest',
            'ml': 'ML Indicator',
            'pos': 'Positional',
            'market_regime': 'Market Regime'
        }
        
    def migrate_all(self) -> Dict[str, Any]:
        """
        Migrate all input sheets
        
        Returns:
            Migration results
        """
        logger.info(f"Starting migration from {self.source_dir} to {self.target_dir}")
        
        # Create target directory structure
        self._create_target_structure()
        
        # Find all Excel files
        excel_files = self._find_excel_files()
        self.stats['total_files'] = len(excel_files)
        
        logger.info(f"Found {len(excel_files)} Excel files to migrate")
        
        # Migrate each file
        for file_path in excel_files:
            try:
                self._migrate_file(file_path)
            except Exception as e:
                logger.error(f"Failed to migrate {file_path}: {e}")
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        # Generate migration report
        self._generate_report()
        
        return self.stats
    
    def _create_target_structure(self):
        """Create target directory structure"""
        directories = [
            self.target_dir,
            self.target_dir / "production",
            self.target_dir / "development",
            self.target_dir / "testing",
            self.target_dir / "archive",
            self.target_dir / "templates"
        ]
        
        for strategy in self.strategy_mapping.keys():
            for dir_path in directories[1:]:  # Skip root
                (dir_path / strategy).mkdir(parents=True, exist_ok=True)
    
    def _find_excel_files(self) -> List[Path]:
        """Find all Excel files in source directory"""
        excel_files = []
        
        for ext in ['.xlsx', '.xls', '.xlsm']:
            excel_files.extend(self.source_dir.rglob(f"*{ext}"))
        
        # Filter out backup files
        excel_files = [f for f in excel_files if not f.name.startswith('~') 
                      and 'backup' not in f.parts]
        
        return sorted(excel_files)
    
    def _migrate_file(self, file_path: Path):
        """Migrate a single Excel file"""
        logger.info(f"Migrating {file_path.name}")
        
        # Determine strategy type
        strategy_type = self._detect_strategy_type(file_path)
        
        if not strategy_type:
            logger.warning(f"Could not determine strategy type for {file_path}")
            self.stats['skipped'] += 1
            return
        
        # Parse the Excel file
        try:
            config_data = self.parser.parse_with_metadata(str(file_path))
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise
        
        # Determine environment (production, development, testing)
        environment = self._determine_environment(file_path)
        
        # Generate configuration name
        config_name = self._generate_config_name(file_path, strategy_type)
        
        # Create configuration instance
        config = self._create_configuration(strategy_type, config_name, config_data)
        
        # Save to new location
        target_path = self.target_dir / environment / strategy_type / f"{config_name}.json"
        config.save_to_file(target_path)
        
        # Also save original Excel for reference
        excel_backup_path = target_path.with_suffix('.xlsx.backup')
        shutil.copy2(file_path, excel_backup_path)
        
        self.stats['migrated'] += 1
        logger.info(f"✓ Migrated to {target_path}")
    
    def _detect_strategy_type(self, file_path: Path) -> str:
        """Detect strategy type from file path and content"""
        # Check file path
        path_str = str(file_path).lower()
        
        for strategy in self.strategy_mapping.keys():
            if strategy in path_str:
                return strategy
        
        # Check file content
        strategy_type = self.parser.extract_strategy_type(str(file_path))
        
        return strategy_type
    
    def _determine_environment(self, file_path: Path) -> str:
        """Determine environment based on file location and name"""
        path_str = str(file_path).lower()
        
        if 'latest' in path_str or 'production' in path_str:
            return 'production'
        elif 'test' in path_str or 'poc' in path_str:
            return 'testing'
        elif 'comprehensive' in path_str or 'enhance' in path_str:
            return 'development'
        else:
            # Default based on directory
            if 'LATEST' in file_path.parts:
                return 'production'
            elif 'comprehensive_tests' in file_path.parts or 'poc_tests' in file_path.parts:
                return 'testing'
            else:
                return 'development'
    
    def _generate_config_name(self, file_path: Path, strategy_type: str) -> str:
        """Generate configuration name from file"""
        # Start with file stem
        name = file_path.stem
        
        # Remove common prefixes/suffixes
        prefixes = ['input_', 'config_', 'enhanced_', 'PHASE2_']
        suffixes = ['_config', '_master', '_latest', '_LATEST']
        
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Remove timestamps if present
        import re
        name = re.sub(r'_\d{8}_\d{6}', '', name)
        name = re.sub(r'_\d{14}', '', name)
        
        # Clean up
        name = name.lower().replace(' ', '_').replace('-', '_')
        
        # Ensure uniqueness
        if not name or name == strategy_type:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"{strategy_type}_{timestamp}"
        
        return name
    
    def _create_configuration(self, strategy_type: str, config_name: str, 
                            config_data: Dict[str, Any]) -> Any:
        """Create configuration instance from parsed data"""
        # Get configuration class
        from ..strategies import (
            TBSConfiguration,
            # Add other configuration classes as they are created
        )
        
        config_classes = {
            'tbs': TBSConfiguration,
            # Add mappings for other strategies
        }
        
        config_class = config_classes.get(strategy_type)
        
        if not config_class:
            # Use base configuration
            from ..core import BaseConfiguration
            
            class GenericConfiguration(BaseConfiguration):
                def validate(self) -> bool:
                    return True
                
                def get_schema(self) -> Dict[str, Any]:
                    return {}
                
                def get_default_values(self) -> Dict[str, Any]:
                    return {}
            
            config_class = GenericConfiguration
        
        # Create instance
        config = config_class(config_name)
        config.from_dict(config_data)
        
        return config
    
    def _generate_report(self):
        """Generate migration report"""
        report = {
            'migration_date': datetime.now().isoformat(),
            'source_directory': str(self.source_dir),
            'target_directory': str(self.target_dir),
            'statistics': self.stats,
            'strategy_breakdown': self._get_strategy_breakdown()
        }
        
        report_path = self.target_dir / f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Migration report saved to {report_path}")
    
    def _get_strategy_breakdown(self) -> Dict[str, int]:
        """Get breakdown by strategy type"""
        breakdown = {}
        
        for env_dir in ['production', 'development', 'testing']:
            env_path = self.target_dir / env_dir
            if env_path.exists():
                for strategy_dir in env_path.iterdir():
                    if strategy_dir.is_dir():
                        count = len(list(strategy_dir.glob('*.json')))
                        if strategy_dir.name not in breakdown:
                            breakdown[strategy_dir.name] = 0
                        breakdown[strategy_dir.name] += count
        
        return breakdown

def analyze_input_sheets(source_dir: str) -> Dict[str, Any]:
    """
    Analyze input sheets directory
    
    Args:
        source_dir: Directory containing input sheets
        
    Returns:
        Analysis results
    """
    source_path = Path(source_dir)
    
    analysis = {
        'total_files': 0,
        'by_extension': {},
        'by_strategy': {},
        'by_directory': {},
        'large_files': [],
        'recent_files': []
    }
    
    # Find all Excel files
    excel_extensions = ['.xlsx', '.xls', '.xlsm', '.xlsb']
    all_files = []
    
    for ext in excel_extensions:
        files = list(source_path.rglob(f"*{ext}"))
        all_files.extend(files)
        analysis['by_extension'][ext] = len(files)
    
    analysis['total_files'] = len(all_files)
    
    # Analyze by strategy
    parser = ExcelParser()
    
    for file_path in all_files:
        # Skip backup files
        if file_path.name.startswith('~') or 'backup' in str(file_path).lower():
            continue
        
        # Detect strategy
        strategy = parser.extract_strategy_type(str(file_path))
        if strategy:
            if strategy not in analysis['by_strategy']:
                analysis['by_strategy'][strategy] = 0
            analysis['by_strategy'][strategy] += 1
        
        # Track by directory
        parent_dir = file_path.parent.name
        if parent_dir not in analysis['by_directory']:
            analysis['by_directory'][parent_dir] = 0
        analysis['by_directory'][parent_dir] += 1
        
        # Track large files (>1MB)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > 1:
            analysis['large_files'].append({
                'file': str(file_path),
                'size_mb': round(size_mb, 2)
            })
        
        # Track recent files (modified in last 30 days)
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        if (datetime.now() - mtime).days < 30:
            analysis['recent_files'].append({
                'file': str(file_path),
                'modified': mtime.isoformat()
            })
    
    # Sort lists
    analysis['large_files'].sort(key=lambda x: x['size_mb'], reverse=True)
    analysis['recent_files'].sort(key=lambda x: x['modified'], reverse=True)
    
    return analysis

def migrate_excel_configs(source_dir: str, target_dir: str = None) -> Dict[str, Any]:
    """
    Main migration function
    
    Args:
        source_dir: Source directory containing input sheets
        target_dir: Target directory (uses default if not provided)
        
    Returns:
        Migration results
    """
    if not target_dir:
        target_dir = "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data"
    
    migrator = InputSheetMigrator(source_dir, target_dir)
    return migrator.migrate_all()

if __name__ == "__main__":
    # Run migration
    source = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets"
    
    # Analyze first
    print("Analyzing input sheets...")
    analysis = analyze_input_sheets(source)
    print(f"Found {analysis['total_files']} files")
    print(f"Strategies: {analysis['by_strategy']}")
    
    # Migrate
    print("\nStarting migration...")
    results = migrate_excel_configs(source)
    
    print(f"\nMigration complete!")
    print(f"Migrated: {results['migrated']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")