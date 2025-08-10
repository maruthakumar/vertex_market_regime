#!/usr/bin/env python3
"""
Debug Existing CSV - Market Regime Formation Analysis

This script specifically debugs the existing regime_formation_1_month_detailed_202506.csv
file to identify issues and provide detailed analysis of the current regime formation logic.

Key Analysis:
1. Detailed examination of existing CSV structure
2. Mathematical accuracy validation
3. Component score analysis
4. Regime formation logic debugging
5. Missing data identification
6. Recommendations for improvements

Author: The Augster
Date: 2025-06-19
Version: 1.0.0 (CSV Debugging)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_existing_csv.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExistingCSVDebugger:
    """Debug and analyze existing CSV file"""
    
    def __init__(self, csv_path: str = "regime_formation_1_month_detailed_202506.csv"):
        """Initialize the debugger"""
        self.csv_path = csv_path
        self.output_dir = Path("csv_debug_results")
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        
        logger.info(f"CSV Debugger initialized for: {csv_path}")
    
    def load_and_analyze_csv(self) -> Dict[str, Any]:
        """Load and perform comprehensive analysis of the CSV"""
        logger.info("🔍 Loading and analyzing CSV file...")
        
        try:
            # Load CSV
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"✅ Loaded CSV: {len(self.df)} rows × {len(self.df.columns)} columns")
            
            # Comprehensive analysis
            analysis = {
                'basic_info': self._analyze_basic_info(),
                'column_analysis': self._analyze_columns(),
                'mathematical_validation': self._validate_mathematics(),
                'regime_analysis': self._analyze_regimes(),
                'component_analysis': self._analyze_components(),
                'missing_data_analysis': self._analyze_missing_data(),
                'data_quality': self._analyze_data_quality(),
                'time_series_analysis': self._analyze_time_series()
            }
            
            # Save analysis (convert numpy types to Python types for JSON serialization)
            analysis_file = self.output_dir / "detailed_csv_analysis.json"

            def convert_numpy_types(obj):
                """Convert numpy types to Python types for JSON serialization"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            analysis_converted = convert_numpy_types(analysis)

            with open(analysis_file, 'w') as f:
                json.dump(analysis_converted, f, indent=2, default=str)
            
            logger.info(f"✅ Analysis completed. Results saved to {analysis_file}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error loading/analyzing CSV: {e}")
            return {'error': str(e)}
    
    def _analyze_basic_info(self) -> Dict[str, Any]:
        """Analyze basic CSV information"""
        logger.info("📊 Analyzing basic CSV information...")
        
        basic_info = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'file_size_mb': Path(self.csv_path).stat().st_size / 1024 / 1024
        }
        
        # Date range analysis
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            basic_info['date_range'] = {
                'start': self.df['timestamp'].min().isoformat(),
                'end': self.df['timestamp'].max().isoformat(),
                'duration_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days,
                'unique_dates': self.df['timestamp'].dt.date.nunique(),
                'minutes_per_day': len(self.df) / self.df['timestamp'].dt.date.nunique()
            }
        
        return basic_info
    
    def _analyze_columns(self) -> Dict[str, Any]:
        """Analyze column structure and types"""
        logger.info("📋 Analyzing column structure...")
        
        column_analysis = {
            'total_columns': len(self.df.columns),
            'column_types': self.df.dtypes.value_counts().to_dict(),
            'columns_by_category': {}
        }
        
        # Categorize columns
        categories = {
            'basic_data': ['timestamp', 'dte', 'weekday', 'trading_session'],
            'component_scores': [col for col in self.df.columns if col.endswith('_score')],
            'weighted_scores': [col for col in self.df.columns if col.endswith('_weighted')],
            'regime_data': [col for col in self.df.columns if 'regime' in col.lower()],
            'validation_data': [col for col in self.df.columns if any(word in col.lower() for word in ['valid', 'error', 'difference', 'accurate'])],
            'sub_component_data': [col for col in self.df.columns if any(word in col for word in ['theoretical', 'contribution', 'multiplier'])],
            'spot_data': [col for col in self.df.columns if any(word in col.lower() for word in ['spot', 'underlying', 'price'])],
            'straddle_data': [col for col in self.df.columns if 'straddle' in col.lower()],
            'greek_data': [col for col in self.df.columns if any(word in col.lower() for word in ['delta', 'gamma', 'theta', 'vega'])],
            'other': []
        }
        
        # Classify columns
        classified_columns = set()
        for category, patterns in categories.items():
            if category == 'other':
                continue
            
            matching_columns = []
            for col in self.df.columns:
                if col in patterns or any(pattern in col for pattern in patterns if isinstance(pattern, str)):
                    matching_columns.append(col)
                    classified_columns.add(col)
            
            column_analysis['columns_by_category'][category] = matching_columns
        
        # Add unclassified columns to 'other'
        column_analysis['columns_by_category']['other'] = [
            col for col in self.df.columns if col not in classified_columns
        ]
        
        return column_analysis
    
    def _validate_mathematics(self) -> Dict[str, Any]:
        """Validate mathematical accuracy of calculations"""
        logger.info("🧮 Validating mathematical accuracy...")
        
        validation = {
            'weight_sum_validation': {},
            'score_calculation_validation': {},
            'regime_mapping_validation': {}
        }
        
        # Weight sum validation
        if 'weight_sum' in self.df.columns:
            weight_errors = abs(self.df['weight_sum'] - 1.0)
            validation['weight_sum_validation'] = {
                'max_error': weight_errors.max(),
                'mean_error': weight_errors.mean(),
                'accuracy_rate': (weight_errors <= 0.001).mean() * 100,
                'total_errors': (weight_errors > 0.001).sum()
            }
        
        # Score calculation validation
        if all(col in self.df.columns for col in ['calculated_final_score', 'original_final_score']):
            score_diff = abs(self.df['calculated_final_score'] - self.df['original_final_score'])
            validation['score_calculation_validation'] = {
                'max_difference': score_diff.max(),
                'mean_difference': score_diff.mean(),
                'accuracy_rate': (score_diff <= 0.001).mean() * 100,
                'total_errors': (score_diff > 0.001).sum()
            }
        
        # Regime mapping validation
        if all(col in self.df.columns for col in ['calculated_regime_id', 'original_regime_id']):
            regime_matches = (self.df['calculated_regime_id'] == self.df['original_regime_id'])
            validation['regime_mapping_validation'] = {
                'accuracy_rate': regime_matches.mean() * 100,
                'total_mismatches': (~regime_matches).sum(),
                'unique_calculated_regimes': self.df['calculated_regime_id'].nunique(),
                'unique_original_regimes': self.df['original_regime_id'].nunique()
            }
        
        return validation
    
    def _analyze_regimes(self) -> Dict[str, Any]:
        """Analyze regime distribution and patterns"""
        logger.info("🎯 Analyzing regime patterns...")
        
        regime_analysis = {}
        
        # Find regime columns
        regime_columns = [col for col in self.df.columns if 'regime' in col.lower()]
        
        for col in regime_columns:
            if col.endswith('_name') or 'name' in col:
                # Regime name analysis
                regime_counts = self.df[col].value_counts()
                regime_analysis[f'{col}_distribution'] = {
                    'regime_counts': regime_counts.to_dict(),
                    'total_regimes': len(regime_counts),
                    'most_common': regime_counts.index[0],
                    'least_common': regime_counts.index[-1],
                    'balance_score': 1.0 - (regime_counts.max() / len(self.df))
                }
            
            elif col.endswith('_id') or 'id' in col:
                # Regime ID analysis
                regime_id_counts = self.df[col].value_counts().sort_index()
                regime_analysis[f'{col}_distribution'] = {
                    'id_counts': regime_id_counts.to_dict(),
                    'min_id': regime_id_counts.index.min(),
                    'max_id': regime_id_counts.index.max(),
                    'missing_ids': [i for i in range(1, 13) if i not in regime_id_counts.index]
                }
        
        return regime_analysis
    
    def _analyze_components(self) -> Dict[str, Any]:
        """Analyze component scores and contributions"""
        logger.info("🔧 Analyzing component scores...")
        
        component_analysis = {}
        
        # Component score columns
        score_columns = [col for col in self.df.columns if col.endswith('_score') and not col.startswith('confidence')]
        
        for col in score_columns:
            component_analysis[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'range': self.df[col].max() - self.df[col].min(),
                'out_of_bounds': {
                    'below_zero': (self.df[col] < 0).sum(),
                    'above_one': (self.df[col] > 1).sum()
                }
            }
        
        # Weighted component analysis
        weighted_columns = [col for col in self.df.columns if col.endswith('_weighted')]
        
        for col in weighted_columns:
            component_analysis[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'contribution_to_final': self.df[col].mean() if 'final_score' in self.df.columns else None
            }
        
        return component_analysis
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data and identify gaps"""
        logger.info("🔍 Analyzing missing data...")
        
        missing_analysis = {
            'critical_missing_columns': [],
            'null_value_analysis': {},
            'data_completeness': {}
        }
        
        # Check for critical missing columns
        expected_columns = [
            'spot_price', 'underlying_data', 'underlying_price',
            'atm_straddle_price', 'atm_ce_price', 'atm_pe_price'
        ]
        
        for col in expected_columns:
            if col not in self.df.columns:
                missing_analysis['critical_missing_columns'].append({
                    'column': col,
                    'importance': 'HIGH',
                    'impact': 'Cannot validate regime formation against market movement'
                })
        
        # Null value analysis
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100
        
        missing_analysis['null_value_analysis'] = {
            'columns_with_nulls': null_counts[null_counts > 0].to_dict(),
            'null_percentages': null_percentages[null_percentages > 0].to_dict(),
            'high_null_columns': null_percentages[null_percentages > 10].to_dict()
        }
        
        # Data completeness
        missing_analysis['data_completeness'] = {
            'overall_completeness': ((len(self.df) * len(self.df.columns) - null_counts.sum()) / 
                                   (len(self.df) * len(self.df.columns))) * 100,
            'complete_rows': (self.df.isnull().sum(axis=1) == 0).sum(),
            'complete_row_percentage': ((self.df.isnull().sum(axis=1) == 0).sum() / len(self.df)) * 100
        }
        
        return missing_analysis
    
    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze overall data quality"""
        logger.info("✅ Analyzing data quality...")
        
        quality_analysis = {
            'score_ranges': {},
            'outlier_analysis': {},
            'consistency_checks': {}
        }
        
        # Score range validation
        score_columns = [col for col in self.df.columns if col.endswith('_score')]
        
        for col in score_columns:
            if col in self.df.columns:
                quality_analysis['score_ranges'][col] = {
                    'within_0_1_range': ((self.df[col] >= 0) & (self.df[col] <= 1)).mean() * 100,
                    'negative_values': (self.df[col] < 0).sum(),
                    'above_one_values': (self.df[col] > 1).sum()
                }
        
        # Outlier analysis using IQR method
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns[:10]:  # Limit to first 10 numeric columns
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            quality_analysis['outlier_analysis'][col] = {
                'outlier_count': outliers,
                'outlier_percentage': (outliers / len(self.df)) * 100
            }
        
        return quality_analysis
    
    def _analyze_time_series(self) -> Dict[str, Any]:
        """Analyze time series patterns"""
        logger.info("📈 Analyzing time series patterns...")
        
        time_analysis = {}
        
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['minute'] = self.df['timestamp'].dt.minute
            self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
            
            # Trading session analysis
            time_analysis['trading_sessions'] = {
                'opening_hours': self.df[self.df['hour'] == 9]['minute'].value_counts().to_dict(),
                'closing_hours': self.df[self.df['hour'] == 15]['minute'].value_counts().to_dict(),
                'hourly_distribution': self.df['hour'].value_counts().sort_index().to_dict()
            }
            
            # Day of week patterns
            if 'final_score' in self.df.columns:
                time_analysis['weekly_patterns'] = {
                    'avg_score_by_day': self.df.groupby('day_of_week')['final_score'].mean().to_dict(),
                    'regime_distribution_by_day': {}
                }
                
                if 'final_regime_name' in self.df.columns:
                    for day in range(5):  # Monday to Friday
                        day_data = self.df[self.df['day_of_week'] == day]
                        if len(day_data) > 0:
                            regime_dist = day_data['final_regime_name'].value_counts().to_dict()
                            time_analysis['weekly_patterns']['regime_distribution_by_day'][day] = regime_dist
        
        return time_analysis
    
    def generate_debug_report(self) -> str:
        """Generate comprehensive debug report"""
        logger.info("📝 Generating comprehensive debug report...")
        
        # Load and analyze
        analysis = self.load_and_analyze_csv()
        
        if 'error' in analysis:
            return f"Error in analysis: {analysis['error']}"
        
        # Generate report
        report_lines = [
            "="*80,
            "COMPREHENSIVE CSV DEBUG REPORT",
            "="*80,
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"CSV File: {self.csv_path}",
            "",
            "SUMMARY:",
            f"- Total Rows: {analysis['basic_info']['total_rows']:,}",
            f"- Total Columns: {analysis['basic_info']['total_columns']}",
            f"- File Size: {analysis['basic_info']['file_size_mb']:.2f} MB",
            "",
            "CRITICAL ISSUES FOUND:",
        ]
        
        # Add critical issues
        critical_issues = []
        
        # Missing spot data
        if analysis['missing_data_analysis']['critical_missing_columns']:
            critical_issues.append("❌ Missing spot price data (underlying_data)")
        
        # Mathematical accuracy issues
        if 'score_calculation_validation' in analysis['mathematical_validation']:
            accuracy = analysis['mathematical_validation']['score_calculation_validation']['accuracy_rate']
            if accuracy < 95:
                critical_issues.append(f"❌ Low mathematical accuracy: {accuracy:.1f}%")
        
        # Data quality issues
        if analysis['data_quality']['score_ranges']:
            for col, ranges in analysis['data_quality']['score_ranges'].items():
                if ranges['within_0_1_range'] < 95:
                    critical_issues.append(f"❌ Score range issues in {col}")
        
        if critical_issues:
            report_lines.extend(critical_issues)
        else:
            report_lines.append("✅ No critical issues found")
        
        report_lines.extend([
            "",
            "DETAILED ANALYSIS:",
            f"- Mathematical Accuracy: {analysis['mathematical_validation']}",
            f"- Missing Data: {len(analysis['missing_data_analysis']['critical_missing_columns'])} critical columns missing",
            f"- Data Completeness: {analysis['missing_data_analysis']['data_completeness']['overall_completeness']:.1f}%",
            "",
            "RECOMMENDATIONS:",
            "1. Add spot price data (underlying_data) for market movement validation",
            "2. Add ATM straddle price data for options correlation analysis",
            "3. Extend individual indicator breakdown for granular debugging",
            "4. Implement real-time validation against market movements",
            "",
            "="*80
        ])
        
        # Save report
        report_file = self.output_dir / "comprehensive_debug_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"✅ Debug report generated: {report_file}")
        return '\n'.join(report_lines)

if __name__ == "__main__":
    # Run CSV debugging
    debugger = ExistingCSVDebugger()
    report = debugger.generate_debug_report()
    
    print(report)
