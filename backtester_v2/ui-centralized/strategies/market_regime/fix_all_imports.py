#!/usr/bin/env python3
"""
Fix All Imports Script
======================

This script systematically fixes all imports in the market_regime module
to avoid using archived modules and use the refactored structure instead.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define import mappings from old to new
IMPORT_MAPPINGS = {
    # Enhanced modules (now archived)
    r'from\s+\.enhanced_modules\.': 'from .archive_enhanced_modules_do_not_use.',
    r'from\s+\.\.enhanced_modules\.': 'from ..archive_enhanced_modules_do_not_use.',
    r'import\s+\.enhanced_modules\.': 'import .archive_enhanced_modules_do_not_use.',
    
    # Comprehensive modules (now archived)
    r'from\s+\.comprehensive_modules\.': 'from .archive_comprehensive_modules_do_not_use.',
    r'from\s+\.\.comprehensive_modules\.': 'from ..archive_comprehensive_modules_do_not_use.',
    r'import\s+\.comprehensive_modules\.': 'import .archive_comprehensive_modules_do_not_use.',
    
    # Specific module remappings to new structure
    'from .comprehensive_modules.comprehensive_triple_straddle_engine': 'from .indicators.straddle_analysis.core.straddle_engine',
    'from .comprehensive_modules.comprehensive_market_regime_analyzer': 'from .core.engine',
    'from .enhanced_modules.enhanced_greek_sentiment': 'from .indicators.greek_sentiment.greek_sentiment_analyzer',
    'from .enhanced_modules.enhanced_trending_oi_pa_analysis': 'from .indicators.oi_pa_analysis.oi_pa_analyzer',
    'from .enhanced_modules.enhanced_oi_pattern_mathematical_correlation': 'from .indicators.oi_pa_analysis.mathematical_correlation',
    'from .enhanced_modules.enhanced_historical_weightage_optimizer': 'from .optimizers.historical_weightage_optimizer',
    'from .enhanced_modules.enhanced_adaptive_integration_framework': 'from .base.adaptive_integration',
    
    # Class name updates
    'ComprehensiveTripleStraddleEngine': 'StraddleAnalysisEngine',
    'ComprehensiveMarketRegimeAnalyzer': 'MarketRegimeEngine',
    'EnhancedGreekSentiment': 'GreekSentimentAnalyzer',
    'EnhancedTrendingOIPAAnalysis': 'OIPriceActionAnalyzer',
}

# Files to exclude from processing
EXCLUDE_PATTERNS = [
    'archive_enhanced_modules_do_not_use',
    'archive_comprehensive_modules_do_not_use',
    '__pycache__',
    '.pyc',
    'fix_all_imports.py',  # Don't modify this script itself
]

def should_process_file(filepath: str) -> bool:
    """Check if file should be processed"""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filepath:
            return False
    return filepath.endswith('.py')

def fix_imports_in_content(content: str, filepath: str) -> Tuple[str, List[str]]:
    """Fix imports in file content"""
    changes = []
    original_content = content
    
    # Apply import mappings
    for old_pattern, new_pattern in IMPORT_MAPPINGS.items():
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_pattern, content)
            changes.append(f"Updated: {old_pattern} -> {new_pattern}")
    
    # Check for any remaining references to archived modules
    archived_refs = []
    if 'enhanced_modules' in content and 'archive_enhanced_modules_do_not_use' not in content:
        archived_refs.append('enhanced_modules')
    if 'comprehensive_modules' in content and 'archive_comprehensive_modules_do_not_use' not in content:
        archived_refs.append('comprehensive_modules')
    
    if archived_refs:
        logger.warning(f"{filepath}: Still contains references to {', '.join(archived_refs)}")
    
    return content, changes

def process_file(filepath: str) -> Dict[str, any]:
    """Process a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content, changes = fix_imports_in_content(content, filepath)
        
        if changes:
            # Write back the modified content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                'status': 'modified',
                'changes': changes,
                'filepath': filepath
            }
        else:
            return {
                'status': 'unchanged',
                'filepath': filepath
            }
    
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'filepath': filepath
        }

def main():
    """Main function to fix all imports"""
    logger.info("Starting import fix process...")
    
    # Get the market_regime directory
    base_dir = Path(__file__).parent
    
    # Collect all Python files
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if should_process_file(filepath):
                python_files.append(filepath)
    
    logger.info(f"Found {len(python_files)} Python files to process")
    
    # Process each file
    results = {
        'modified': [],
        'unchanged': [],
        'errors': []
    }
    
    for filepath in python_files:
        result = process_file(filepath)
        
        if result['status'] == 'modified':
            results['modified'].append(result)
            logger.info(f"Modified: {result['filepath']}")
            for change in result['changes']:
                logger.info(f"  - {change}")
        elif result['status'] == 'error':
            results['errors'].append(result)
        else:
            results['unchanged'].append(result)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("IMPORT FIX SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files processed: {len(python_files)}")
    logger.info(f"Files modified: {len(results['modified'])}")
    logger.info(f"Files unchanged: {len(results['unchanged'])}")
    logger.info(f"Files with errors: {len(results['errors'])}")
    
    if results['errors']:
        logger.error("\nFiles with errors:")
        for error in results['errors']:
            logger.error(f"  - {error['filepath']}: {error['error']}")
    
    # Write detailed report
    report_path = base_dir / 'import_fix_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    main()