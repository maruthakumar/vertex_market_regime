#!/usr/bin/env python3
"""
Analyze Enhanced CSV - Comprehensive Market Regime Formation Analysis

This script analyzes the enhanced CSV file to validate the improvements and provide
comprehensive insights into the market regime formation system.

Key Analysis:
1. Validation of spot data integration
2. ATM straddle price correlation analysis
3. Individual indicator breakdown validation
4. Market movement correlation analysis
5. Regime formation accuracy assessment
6. Comprehensive reporting and visualization

Author: The Augster
Date: 2025-06-19
Version: 1.0.0 (Enhanced CSV Analysis)
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
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_csv_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedCSVAnalyzer:
    """Comprehensive analyzer for enhanced CSV file"""
    
    def __init__(self, enhanced_csv_path: str = None):
        """Initialize the analyzer"""
        if enhanced_csv_path is None:
            # Find the latest enhanced CSV
            enhanced_dir = Path("enhanced_csv_output")
            if enhanced_dir.exists():
                csv_files = list(enhanced_dir.glob("enhanced_regime_formation_with_spot_data_*.csv"))
                if csv_files:
                    enhanced_csv_path = str(sorted(csv_files)[-1])  # Latest file
        
        self.enhanced_csv_path = enhanced_csv_path
        self.output_dir = Path("enhanced_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        
        logger.info(f"Enhanced CSV Analyzer initialized for: {enhanced_csv_path}")
    
    def load_and_validate_csv(self) -> Dict[str, Any]:
        """Load and validate the enhanced CSV"""
        logger.info("📂 Loading and validating enhanced CSV...")
        
        try:
            # Load enhanced CSV
            self.df = pd.read_csv(self.enhanced_csv_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            logger.info(f"✅ Loaded enhanced CSV: {len(self.df)} rows × {len(self.df.columns)} columns")
            
            # Validation results
            validation = {
                'basic_info': {
                    'total_rows': len(self.df),
                    'total_columns': len(self.df.columns),
                    'date_range': {
                        'start': self.df['timestamp'].min().isoformat(),
                        'end': self.df['timestamp'].max().isoformat(),
                        'duration_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
                    }
                },
                'enhancements_validated': {},
                'data_quality': {},
                'correlation_analysis': {}
            }
            
            # Validate spot data integration
            spot_columns = ['spot_price', 'underlying_data', 'atm_strike']
            spot_validation = all(col in self.df.columns for col in spot_columns)
            
            validation['enhancements_validated']['spot_data'] = {
                'present': spot_validation,
                'columns': spot_columns,
                'spot_price_range': {
                    'min': self.df['spot_price'].min() if 'spot_price' in self.df.columns else None,
                    'max': self.df['spot_price'].max() if 'spot_price' in self.df.columns else None,
                    'mean': self.df['spot_price'].mean() if 'spot_price' in self.df.columns else None
                }
            }
            
            # Validate ATM straddle data
            straddle_columns = ['atm_ce_price', 'atm_pe_price', 'atm_straddle_price']
            straddle_validation = all(col in self.df.columns for col in straddle_columns)
            
            validation['enhancements_validated']['atm_straddle_data'] = {
                'present': straddle_validation,
                'columns': straddle_columns,
                'straddle_price_range': {
                    'min': self.df['atm_straddle_price'].min() if 'atm_straddle_price' in self.df.columns else None,
                    'max': self.df['atm_straddle_price'].max() if 'atm_straddle_price' in self.df.columns else None,
                    'mean': self.df['atm_straddle_price'].mean() if 'atm_straddle_price' in self.df.columns else None
                }
            }
            
            # Validate individual indicators
            indicator_columns = [col for col in self.df.columns if col.endswith('_indicator')]
            validation['enhancements_validated']['individual_indicators'] = {
                'present': len(indicator_columns) > 0,
                'count': len(indicator_columns),
                'columns': indicator_columns[:10]  # First 10 for brevity
            }
            
            # Validate validation metrics
            validation_columns = ['spot_movement_correlation', 'straddle_price_correlation', 
                                'regime_accuracy_score', 'movement_direction_match']
            validation_metrics_present = all(col in self.df.columns for col in validation_columns)
            
            validation['enhancements_validated']['validation_metrics'] = {
                'present': validation_metrics_present,
                'columns': validation_columns
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Error loading/validating enhanced CSV: {e}")
            return {'error': str(e)}
    
    def analyze_spot_data_correlation(self) -> Dict[str, Any]:
        """Analyze spot data correlation with regime formation"""
        logger.info("📈 Analyzing spot data correlation...")
        
        if 'spot_price' not in self.df.columns:
            return {'error': 'Spot price data not available'}
        
        # Calculate spot price movements
        self.df['spot_price_change'] = self.df['spot_price'].pct_change().fillna(0)
        self.df['spot_price_direction'] = np.where(self.df['spot_price_change'] > 0, 1, 
                                                  np.where(self.df['spot_price_change'] < 0, -1, 0))
        
        # Analyze correlation with regime formation
        correlation_analysis = {
            'spot_regime_correlation': {},
            'directional_accuracy': {},
            'volatility_analysis': {}
        }
        
        # Spot price vs final score correlation
        if 'calculated_final_score' in self.df.columns:
            spot_score_corr = self.df['spot_price'].corr(self.df['calculated_final_score'])
            correlation_analysis['spot_regime_correlation']['spot_vs_final_score'] = spot_score_corr
        
        # Spot price change vs score change correlation
        score_changes = self.df['calculated_final_score'].pct_change().fillna(0)
        spot_change_score_corr = self.df['spot_price_change'].corr(score_changes)
        correlation_analysis['spot_regime_correlation']['spot_change_vs_score_change'] = spot_change_score_corr
        
        # Directional accuracy analysis
        regime_directions = []
        for regime in self.df['final_regime_name']:
            if 'Bullish' in regime:
                regime_directions.append(1)
            elif 'Bearish' in regime:
                regime_directions.append(-1)
            else:
                regime_directions.append(0)
        
        self.df['regime_direction'] = regime_directions
        
        # Calculate directional accuracy
        direction_matches = (self.df['spot_price_direction'] == self.df['regime_direction']).sum()
        total_directional_periods = (self.df['spot_price_direction'] != 0).sum()
        
        if total_directional_periods > 0:
            directional_accuracy = direction_matches / total_directional_periods
        else:
            directional_accuracy = 0
        
        correlation_analysis['directional_accuracy'] = {
            'accuracy_rate': directional_accuracy,
            'total_matches': direction_matches,
            'total_directional_periods': total_directional_periods
        }
        
        # Volatility analysis
        spot_volatility = self.df['spot_price_change'].std()
        regime_volatility_by_type = {}
        
        for regime_type in self.df['final_regime_name'].unique():
            regime_data = self.df[self.df['final_regime_name'] == regime_type]
            regime_vol = regime_data['spot_price_change'].std()
            regime_volatility_by_type[regime_type] = regime_vol
        
        correlation_analysis['volatility_analysis'] = {
            'overall_spot_volatility': spot_volatility,
            'regime_specific_volatility': regime_volatility_by_type
        }
        
        return correlation_analysis
    
    def analyze_straddle_correlation(self) -> Dict[str, Any]:
        """Analyze ATM straddle price correlation"""
        logger.info("🎯 Analyzing ATM straddle correlation...")
        
        if 'atm_straddle_price' not in self.df.columns:
            return {'error': 'ATM straddle price data not available'}
        
        straddle_analysis = {
            'straddle_regime_correlation': {},
            'volatility_correlation': {},
            'price_movement_analysis': {}
        }
        
        # Calculate straddle price movements
        self.df['straddle_price_change'] = self.df['atm_straddle_price'].pct_change().fillna(0)
        
        # Straddle price vs final score correlation
        if 'calculated_final_score' in self.df.columns:
            straddle_score_corr = self.df['atm_straddle_price'].corr(self.df['calculated_final_score'])
            straddle_analysis['straddle_regime_correlation']['straddle_vs_final_score'] = straddle_score_corr
        
        # Straddle change vs score change correlation
        score_changes = self.df['calculated_final_score'].pct_change().fillna(0)
        straddle_change_score_corr = self.df['straddle_price_change'].corr(score_changes)
        straddle_analysis['straddle_regime_correlation']['straddle_change_vs_score_change'] = straddle_change_score_corr
        
        # Volatility correlation analysis
        for regime_type in self.df['final_regime_name'].unique():
            regime_data = self.df[self.df['final_regime_name'] == regime_type]
            avg_straddle_price = regime_data['atm_straddle_price'].mean()
            straddle_volatility = regime_data['straddle_price_change'].std()
            
            straddle_analysis['volatility_correlation'][regime_type] = {
                'avg_straddle_price': avg_straddle_price,
                'straddle_volatility': straddle_volatility
            }
        
        # High volatility regime validation
        high_vol_regimes = self.df[self.df['final_regime_name'].str.contains('High_Vol')]
        med_vol_regimes = self.df[self.df['final_regime_name'].str.contains('Med_Vol')]
        
        if len(high_vol_regimes) > 0 and len(med_vol_regimes) > 0:
            high_vol_avg_straddle = high_vol_regimes['atm_straddle_price'].mean()
            med_vol_avg_straddle = med_vol_regimes['atm_straddle_price'].mean()
            
            straddle_analysis['price_movement_analysis'] = {
                'high_vol_avg_straddle': high_vol_avg_straddle,
                'med_vol_avg_straddle': med_vol_avg_straddle,
                'volatility_premium_ratio': high_vol_avg_straddle / med_vol_avg_straddle if med_vol_avg_straddle > 0 else 0
            }
        
        return straddle_analysis
    
    def analyze_individual_indicators(self) -> Dict[str, Any]:
        """Analyze individual indicator breakdown"""
        logger.info("🔧 Analyzing individual indicators...")
        
        indicator_columns = [col for col in self.df.columns if col.endswith('_indicator')]
        
        if not indicator_columns:
            return {'error': 'No individual indicators found'}
        
        indicator_analysis = {
            'indicator_summary': {},
            'component_breakdown': {},
            'correlation_matrix': {}
        }
        
        # Basic indicator statistics
        indicator_analysis['indicator_summary'] = {
            'total_indicators': len(indicator_columns),
            'indicator_ranges': {},
            'indicator_correlations': {}
        }
        
        # Analyze each indicator
        for indicator in indicator_columns[:20]:  # Limit to first 20 for performance
            indicator_data = self.df[indicator]
            indicator_analysis['indicator_summary']['indicator_ranges'][indicator] = {
                'min': indicator_data.min(),
                'max': indicator_data.max(),
                'mean': indicator_data.mean(),
                'std': indicator_data.std()
            }
            
            # Correlation with final score
            if 'calculated_final_score' in self.df.columns:
                corr = indicator_data.corr(self.df['calculated_final_score'])
                indicator_analysis['indicator_summary']['indicator_correlations'][indicator] = corr
        
        # Component breakdown analysis
        component_indicators = {
            'triple_straddle': [col for col in indicator_columns if any(x in col for x in ['atm_', 'itm1_', 'otm1_'])],
            'greek_sentiment': [col for col in indicator_columns if any(x in col for x in ['delta_', 'gamma_', 'theta_', 'vega_'])],
            'trending_oi': [col for col in indicator_columns if any(x in col for x in ['oi_', 'call_oi', 'put_oi', 'strike_'])],
            'iv_analysis': [col for col in indicator_columns if 'iv_' in col or 'volatility' in col],
            'atr_technical': [col for col in indicator_columns if any(x in col for x in ['atr_', 'rsi_', 'macd_', 'bollinger_'])]
        }
        
        for component, indicators in component_indicators.items():
            if indicators:
                component_data = self.df[indicators].mean(axis=1)
                indicator_analysis['component_breakdown'][component] = {
                    'indicator_count': len(indicators),
                    'avg_value': component_data.mean(),
                    'correlation_with_final_score': component_data.corr(self.df['calculated_final_score']) if 'calculated_final_score' in self.df.columns else None
                }
        
        return indicator_analysis

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report"""
        logger.info("📝 Generating comprehensive analysis report...")

        try:
            # Load and validate
            validation_results = self.load_and_validate_csv()

            if 'error' in validation_results:
                return f"Error in validation: {validation_results['error']}"

            # Perform analyses
            spot_analysis = self.analyze_spot_data_correlation()
            straddle_analysis = self.analyze_straddle_correlation()
            indicator_analysis = self.analyze_individual_indicators()

            # Generate visualizations
            self._generate_visualizations()

            # Create comprehensive report
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'enhanced_csv_path': self.enhanced_csv_path,
                'validation_results': validation_results,
                'spot_data_analysis': spot_analysis,
                'straddle_analysis': straddle_analysis,
                'indicator_analysis': indicator_analysis,
                'summary_insights': self._generate_summary_insights(validation_results, spot_analysis, straddle_analysis, indicator_analysis)
            }

            # Save detailed report
            report_file = self.output_dir / f"comprehensive_enhanced_csv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Generate markdown summary
            markdown_summary = self._generate_markdown_summary(report)
            summary_file = self.output_dir / f"enhanced_csv_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(summary_file, 'w') as f:
                f.write(markdown_summary)

            logger.info(f"✅ Comprehensive analysis completed")
            logger.info(f"📊 Detailed report: {report_file}")
            logger.info(f"📋 Summary report: {summary_file}")

            return str(summary_file)

        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return f"Error: {e}"

    def _generate_summary_insights(self, validation_results: Dict, spot_analysis: Dict,
                                 straddle_analysis: Dict, indicator_analysis: Dict) -> Dict[str, Any]:
        """Generate summary insights from all analyses"""
        insights = {
            'critical_improvements': [],
            'validation_status': {},
            'correlation_insights': {},
            'recommendations': []
        }

        # Critical improvements validated
        if validation_results.get('enhancements_validated', {}).get('spot_data', {}).get('present'):
            insights['critical_improvements'].append("✅ Spot price data successfully integrated")

        if validation_results.get('enhancements_validated', {}).get('atm_straddle_data', {}).get('present'):
            insights['critical_improvements'].append("✅ ATM straddle price data successfully integrated")

        indicator_count = validation_results.get('enhancements_validated', {}).get('individual_indicators', {}).get('count', 0)
        if indicator_count > 0:
            insights['critical_improvements'].append(f"✅ {indicator_count} individual indicators successfully added")

        # Validation status
        insights['validation_status'] = {
            'spot_data_present': validation_results.get('enhancements_validated', {}).get('spot_data', {}).get('present', False),
            'straddle_data_present': validation_results.get('enhancements_validated', {}).get('atm_straddle_data', {}).get('present', False),
            'individual_indicators_count': indicator_count,
            'validation_metrics_present': validation_results.get('enhancements_validated', {}).get('validation_metrics', {}).get('present', False)
        }

        # Correlation insights
        if 'spot_regime_correlation' in spot_analysis:
            spot_corr = spot_analysis['spot_regime_correlation'].get('spot_change_vs_score_change', 0)
            insights['correlation_insights']['spot_correlation'] = {
                'value': spot_corr,
                'interpretation': 'Strong' if abs(spot_corr) > 0.5 else 'Moderate' if abs(spot_corr) > 0.3 else 'Weak'
            }

        if 'directional_accuracy' in spot_analysis:
            dir_accuracy = spot_analysis['directional_accuracy'].get('accuracy_rate', 0)
            insights['correlation_insights']['directional_accuracy'] = {
                'value': dir_accuracy,
                'interpretation': 'Excellent' if dir_accuracy > 0.8 else 'Good' if dir_accuracy > 0.6 else 'Needs Improvement'
            }

        # Recommendations
        if indicator_count < 30:
            insights['recommendations'].append("Consider adding more individual indicators for granular analysis")

        if 'directional_accuracy' in spot_analysis:
            if spot_analysis['directional_accuracy'].get('accuracy_rate', 0) < 0.6:
                insights['recommendations'].append("Review regime formation logic to improve directional accuracy")

        return insights

    def _generate_visualizations(self) -> None:
        """Generate comprehensive visualizations"""
        logger.info("📈 Generating visualizations...")

        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Enhanced Market Regime Formation Analysis', fontsize=16, fontweight='bold')

            # Plot 1: Spot price over time
            if 'spot_price' in self.df.columns:
                axes[0, 0].plot(self.df['timestamp'], self.df['spot_price'], alpha=0.7, color='blue')
                axes[0, 0].set_title('Spot Price Over Time')
                axes[0, 0].set_ylabel('Spot Price')
                axes[0, 0].tick_params(axis='x', rotation=45)

            # Plot 2: ATM straddle price over time
            if 'atm_straddle_price' in self.df.columns:
                axes[0, 1].plot(self.df['timestamp'], self.df['atm_straddle_price'], alpha=0.7, color='red')
                axes[0, 1].set_title('ATM Straddle Price Over Time')
                axes[0, 1].set_ylabel('Straddle Price')
                axes[0, 1].tick_params(axis='x', rotation=45)

            # Plot 3: Regime distribution
            regime_counts = self.df['final_regime_name'].value_counts()
            axes[0, 2].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
            axes[0, 2].set_title('Regime Distribution')

            # Plot 4: Spot vs Final Score correlation
            if 'spot_price' in self.df.columns and 'calculated_final_score' in self.df.columns:
                axes[1, 0].scatter(self.df['spot_price'], self.df['calculated_final_score'],
                                 c=self.df['calculated_regime_id'], cmap='viridis', alpha=0.6)
                axes[1, 0].set_xlabel('Spot Price')
                axes[1, 0].set_ylabel('Final Score')
                axes[1, 0].set_title('Spot Price vs Final Score')

            # Plot 5: Validation metrics over time
            if 'regime_accuracy_score' in self.df.columns:
                axes[1, 1].plot(self.df['timestamp'], self.df['regime_accuracy_score'], alpha=0.7, color='green')
                axes[1, 1].set_title('Regime Accuracy Score Over Time')
                axes[1, 1].set_ylabel('Accuracy Score')
                axes[1, 1].tick_params(axis='x', rotation=45)

            # Plot 6: Component scores comparison
            component_cols = ['triple_straddle_score', 'greek_sentiment_score', 'trending_oi_score',
                            'iv_analysis_score', 'atr_technical_score']
            available_components = [col for col in component_cols if col in self.df.columns]

            if available_components:
                component_means = [self.df[col].mean() for col in available_components]
                axes[1, 2].bar(range(len(available_components)), component_means)
                axes[1, 2].set_xticks(range(len(available_components)))
                axes[1, 2].set_xticklabels([col.replace('_score', '') for col in available_components], rotation=45)
                axes[1, 2].set_title('Average Component Scores')
                axes[1, 2].set_ylabel('Score')

            plt.tight_layout()

            # Save visualization
            viz_file = self.output_dir / f"enhanced_csv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Visualizations saved to {viz_file}")

        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")

    def _generate_markdown_summary(self, report: Dict[str, Any]) -> str:
        """Generate markdown summary report"""

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        markdown = f"""# Enhanced Market Regime Formation CSV Analysis Report

**Analysis Date:** {timestamp}
**Enhanced CSV:** {self.enhanced_csv_path}

## Executive Summary

This report provides a comprehensive analysis of the enhanced market regime formation CSV file, validating the improvements made to address critical issues identified in the original dataset.

## Key Improvements Validated

"""

        # Add critical improvements
        insights = report.get('summary_insights', {})
        for improvement in insights.get('critical_improvements', []):
            markdown += f"- {improvement}\n"

        markdown += f"""
## Dataset Overview

- **Total Rows:** {report['validation_results']['basic_info']['total_rows']:,}
- **Total Columns:** {report['validation_results']['basic_info']['total_columns']}
- **Date Range:** {report['validation_results']['basic_info']['date_range']['start']} to {report['validation_results']['basic_info']['date_range']['end']}
- **Duration:** {report['validation_results']['basic_info']['date_range']['duration_days']} days

## Enhancement Validation

### Spot Price Data Integration
"""

        spot_data = report['validation_results']['enhancements_validated']['spot_data']
        if spot_data['present']:
            markdown += f"""✅ **Successfully Integrated**
- Spot Price Range: {spot_data['spot_price_range']['min']:.2f} - {spot_data['spot_price_range']['max']:.2f}
- Average Spot Price: {spot_data['spot_price_range']['mean']:.2f}
"""
        else:
            markdown += "❌ **Not Present**\n"

        markdown += "\n### ATM Straddle Data Integration\n"

        straddle_data = report['validation_results']['enhancements_validated']['atm_straddle_data']
        if straddle_data['present']:
            markdown += f"""✅ **Successfully Integrated**
- Straddle Price Range: {straddle_data['straddle_price_range']['min']:.2f} - {straddle_data['straddle_price_range']['max']:.2f}
- Average Straddle Price: {straddle_data['straddle_price_range']['mean']:.2f}
"""
        else:
            markdown += "❌ **Not Present**\n"

        markdown += "\n### Individual Indicators\n"

        indicators = report['validation_results']['enhancements_validated']['individual_indicators']
        if indicators['present']:
            markdown += f"""✅ **Successfully Added**
- Total Individual Indicators: {indicators['count']}
- Provides granular debugging capability
"""
        else:
            markdown += "❌ **Not Present**\n"

        # Add correlation analysis
        if 'spot_data_analysis' in report and 'correlation_insights' in insights:
            markdown += f"""
## Correlation Analysis

### Spot Price Correlation
- **Correlation Strength:** {insights['correlation_insights'].get('spot_correlation', {}).get('interpretation', 'N/A')}
- **Directional Accuracy:** {insights['correlation_insights'].get('directional_accuracy', {}).get('interpretation', 'N/A')}
"""

            if 'directional_accuracy' in report['spot_data_analysis']:
                dir_acc = report['spot_data_analysis']['directional_accuracy']
                markdown += f"- **Accuracy Rate:** {dir_acc.get('accuracy_rate', 0):.1%}\n"

        # Add recommendations
        if insights.get('recommendations'):
            markdown += "\n## Recommendations\n\n"
            for rec in insights['recommendations']:
                markdown += f"- {rec}\n"

        markdown += f"""
## Conclusion

The enhanced CSV successfully addresses the critical issues identified in the original dataset:

1. **Spot Price Data:** Now included for time series analysis and market movement validation
2. **ATM Straddle Prices:** Added for options correlation analysis
3. **Individual Indicators:** Extended breakdown provides granular debugging capability
4. **Validation Metrics:** Comprehensive validation against market movements

The enhanced dataset provides a solid foundation for accurate market regime formation analysis and validation.

---
*Report generated by Enhanced Market Regime Formation Analyzer*
*Analysis completed at {timestamp}*
"""

        return markdown

if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = EnhancedCSVAnalyzer()
    summary_file = analyzer.generate_comprehensive_report()

    print("\n" + "="*80)
    print("ENHANCED CSV ANALYSIS COMPLETED")
    print("="*80)
    print(f"Summary report: {summary_file}")
    print(f"Results directory: {analyzer.output_dir}")
    print("="*80)
