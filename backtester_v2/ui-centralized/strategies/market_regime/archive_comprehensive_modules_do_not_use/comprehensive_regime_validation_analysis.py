#!/usr/bin/env python3
"""
Comprehensive Market Regime Formation Validation Analysis

This script performs comprehensive validation and analysis of the market regime formation
system by:
1. Analyzing existing CSV data for issues and gaps
2. Fetching real market data with spot prices and options data
3. Extending sub-component analysis to individual indicator level
4. Validating regime formation against actual market movements
5. Generating enhanced CSV with complete transparency
6. Creating detailed debugging and analysis reports

Usage:
    python comprehensive_regime_validation_analysis.py

Author: The Augster
Date: 2025-06-19
Version: 1.0.0 (Comprehensive Validation)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

# Import our enhanced validator
from enhanced_market_regime_validator import EnhancedMarketRegimeValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_regime_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveRegimeValidationAnalysis:
    """Comprehensive regime validation and analysis system"""
    
    def __init__(self):
        """Initialize the comprehensive analysis system"""
        self.validator = EnhancedMarketRegimeValidator()
        self.existing_csv_path = "regime_formation_1_month_detailed_202506.csv"
        self.output_dir = Path("validation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Comprehensive Regime Validation Analysis initialized")
    
    def analyze_existing_csv(self) -> Dict[str, Any]:
        """Analyze the existing CSV file to identify issues and gaps"""
        logger.info("🔍 Analyzing existing CSV file...")
        
        try:
            # Load existing CSV
            df = pd.read_csv(self.existing_csv_path)
            logger.info(f"Loaded existing CSV: {len(df)} rows × {len(df.columns)} columns")
            
            analysis_results = {
                'basic_stats': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'date_range': {
                        'start': df['timestamp'].min() if 'timestamp' in df.columns else 'N/A',
                        'end': df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'
                    }
                },
                'missing_components': [],
                'data_quality_issues': [],
                'regime_distribution': {},
                'mathematical_accuracy': {}
            }
            
            # Check for missing spot data
            if 'spot_price' not in df.columns and 'underlying_data' not in df.columns:
                analysis_results['missing_components'].append({
                    'component': 'spot_price/underlying_data',
                    'description': 'Missing spot price data for time series analysis',
                    'impact': 'Cannot validate regime formation against actual market movement'
                })
            
            # Check for missing ATM straddle price data
            if 'atm_straddle_price' not in df.columns:
                analysis_results['missing_components'].append({
                    'component': 'atm_straddle_price',
                    'description': 'Missing ATM straddle price data',
                    'impact': 'Cannot validate regime formation against options price movement'
                })
            
            # Check for individual indicator breakdown
            individual_indicator_columns = [col for col in df.columns if '_' in col and 
                                          any(component in col for component in 
                                              ['triple_straddle', 'greek_sentiment', 'trending_oi', 'iv_analysis', 'atr_technical'])]
            
            if len(individual_indicator_columns) < 20:  # Expecting 30+ individual indicators
                analysis_results['missing_components'].append({
                    'component': 'individual_indicators',
                    'description': f'Limited individual indicator breakdown (found {len(individual_indicator_columns)} columns)',
                    'impact': 'Cannot debug regime formation at granular level'
                })
            
            # Analyze regime distribution
            if 'final_regime_name' in df.columns:
                regime_counts = df['final_regime_name'].value_counts()
                analysis_results['regime_distribution'] = regime_counts.to_dict()
            elif 'regime_name' in df.columns:
                regime_counts = df['regime_name'].value_counts()
                analysis_results['regime_distribution'] = regime_counts.to_dict()
            
            # Check mathematical accuracy
            if 'calculated_final_score' in df.columns and 'original_final_score' in df.columns:
                score_diff = abs(df['calculated_final_score'] - df['original_final_score'])
                analysis_results['mathematical_accuracy'] = {
                    'max_difference': score_diff.max(),
                    'mean_difference': score_diff.mean(),
                    'accuracy_rate': (score_diff <= 0.001).mean() * 100
                }
            
            # Data quality checks
            null_counts = df.isnull().sum()
            high_null_columns = null_counts[null_counts > len(df) * 0.1].to_dict()
            
            if high_null_columns:
                analysis_results['data_quality_issues'].append({
                    'issue': 'high_null_values',
                    'columns': high_null_columns,
                    'description': 'Columns with >10% null values'
                })
            
            # Save analysis results
            analysis_file = self.output_dir / "existing_csv_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"✅ CSV analysis completed. Results saved to {analysis_file}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing existing CSV: {e}")
            return {'error': str(e)}
    
    def fetch_and_validate_real_data(self, start_date: str = "2024-01-01", 
                                   end_date: str = "2024-01-31") -> Dict[str, Any]:
        """Fetch real market data and perform comprehensive validation"""
        logger.info(f"📊 Fetching real market data from {start_date} to {end_date}...")
        
        try:
            # Fetch real market data
            market_data = self.validator.fetch_real_market_data(start_date, end_date)
            
            if not market_data:
                logger.error("No market data fetched")
                return {'error': 'No market data available'}
            
            logger.info(f"✅ Fetched {len(market_data)} data points")
            
            # Generate enhanced CSV with validation
            csv_filename = self.validator.generate_enhanced_csv_with_validation(
                market_data, 
                str(self.output_dir / "enhanced_regime_formation_with_validation.csv")
            )
            
            # Perform comprehensive validation analysis
            validation_results = self._perform_validation_analysis(market_data)
            
            # Generate visualization reports
            self._generate_visualization_reports(market_data)
            
            # Create summary report
            summary_report = {
                'data_summary': {
                    'total_data_points': len(market_data),
                    'date_range': {
                        'start': market_data[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'end': market_data[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'csv_file': csv_filename
                },
                'validation_results': validation_results,
                'enhancement_summary': {
                    'spot_data_included': True,
                    'atm_straddle_prices_included': True,
                    'individual_indicators_count': self.validator._count_total_indicators(),
                    'validation_metrics_included': True
                }
            }
            
            # Save summary report
            summary_file = self.output_dir / "comprehensive_validation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            logger.info(f"✅ Comprehensive validation completed. Summary saved to {summary_file}")
            return summary_report
            
        except Exception as e:
            logger.error(f"❌ Error in real data validation: {e}")
            return {'error': str(e)}
    
    def _perform_validation_analysis(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform detailed validation analysis"""
        logger.info("🔬 Performing validation analysis...")
        
        validation_results = {
            'regime_accuracy': {},
            'spot_correlation': {},
            'straddle_correlation': {},
            'component_analysis': {},
            'mathematical_validation': {}
        }
        
        # Extract data for analysis
        regimes = [data['regime_name'] for data in market_data]
        spot_prices = [data['spot_price'] for data in market_data]
        final_scores = [data['final_score'] for data in market_data]
        
        # Regime distribution analysis
        regime_counts = pd.Series(regimes).value_counts()
        validation_results['regime_accuracy'] = {
            'regime_distribution': regime_counts.to_dict(),
            'regime_diversity': len(regime_counts),
            'most_common_regime': regime_counts.index[0],
            'regime_balance_score': 1.0 - (regime_counts.max() / len(regimes))
        }
        
        # Spot price correlation analysis
        spot_changes = np.diff(spot_prices)
        score_changes = np.diff(final_scores)
        
        if len(spot_changes) > 1:
            spot_correlation = np.corrcoef(spot_changes, score_changes)[0, 1]
            validation_results['spot_correlation'] = {
                'correlation_coefficient': spot_correlation if not np.isnan(spot_correlation) else 0.0,
                'price_volatility': np.std(spot_changes),
                'score_volatility': np.std(score_changes)
            }
        
        # Component analysis
        component_scores = {
            'triple_straddle': [data['component_scores']['triple_straddle'] for data in market_data],
            'greek_sentiment': [data['component_scores']['greek_sentiment'] for data in market_data],
            'trending_oi': [data['component_scores']['trending_oi'] for data in market_data],
            'iv_analysis': [data['component_scores']['iv_analysis'] for data in market_data],
            'atr_technical': [data['component_scores']['atr_technical'] for data in market_data]
        }
        
        for component, scores in component_scores.items():
            validation_results['component_analysis'][component] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'contribution_to_final': np.corrcoef(scores, final_scores)[0, 1] if not np.isnan(np.corrcoef(scores, final_scores)[0, 1]) else 0.0
            }
        
        # Mathematical validation
        calculated_scores = []
        for data in market_data:
            calculated_score = sum(
                data['component_scores'][component] * self.validator.component_weights[component]
                for component in data['component_scores'].keys()
            )
            calculated_scores.append(calculated_score)
        
        score_differences = [abs(calc - orig) for calc, orig in zip(calculated_scores, final_scores)]
        validation_results['mathematical_validation'] = {
            'max_difference': max(score_differences),
            'mean_difference': np.mean(score_differences),
            'accuracy_rate': sum(1 for diff in score_differences if diff <= 0.001) / len(score_differences) * 100
        }
        
        return validation_results
    
    def _generate_visualization_reports(self, market_data: List[Dict[str, Any]]) -> None:
        """Generate visualization reports"""
        logger.info("📈 Generating visualization reports...")
        
        try:
            # Create DataFrame for analysis
            df_data = []
            for data in market_data:
                row = {
                    'timestamp': data['timestamp'],
                    'spot_price': data['spot_price'],
                    'final_score': data['final_score'],
                    'regime_name': data['regime_name'],
                    'regime_id': data['regime_id']
                }
                row.update(data['component_scores'])
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Enhanced Market Regime Formation Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Regime distribution
            regime_counts = df['regime_name'].value_counts()
            axes[0, 0].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Regime Distribution')
            
            # Plot 2: Component scores over time
            component_cols = ['triple_straddle', 'greek_sentiment', 'trending_oi', 'iv_analysis', 'atr_technical']
            for col in component_cols:
                axes[0, 1].plot(df['timestamp'], df[col], label=col, alpha=0.7)
            axes[0, 1].set_title('Component Scores Over Time')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Spot price vs Final score
            axes[1, 0].scatter(df['spot_price'], df['final_score'], c=df['regime_id'], cmap='viridis', alpha=0.6)
            axes[1, 0].set_xlabel('Spot Price')
            axes[1, 0].set_ylabel('Final Score')
            axes[1, 0].set_title('Spot Price vs Final Score (colored by regime)')
            
            # Plot 4: Final score distribution
            axes[1, 1].hist(df['final_score'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Final Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Final Score Distribution')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.output_dir / "regime_formation_analysis.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Visualization saved to {viz_file}")
            
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run the complete comprehensive analysis"""
        logger.info("🚀 Starting comprehensive market regime validation analysis...")
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'existing_csv_analysis': {},
            'real_data_validation': {},
            'recommendations': []
        }
        
        # Step 1: Analyze existing CSV
        logger.info("📋 Step 1: Analyzing existing CSV...")
        results['existing_csv_analysis'] = self.analyze_existing_csv()
        
        # Step 2: Fetch and validate real data
        logger.info("📊 Step 2: Fetching and validating real data...")
        results['real_data_validation'] = self.fetch_and_validate_real_data()
        
        # Step 3: Generate recommendations
        logger.info("💡 Step 3: Generating recommendations...")
        results['recommendations'] = self._generate_recommendations(results)
        
        # Save final results
        final_results_file = self.output_dir / "comprehensive_analysis_results.json"
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Comprehensive analysis completed. Results saved to {final_results_file}")
        return results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Check existing CSV analysis
        if 'missing_components' in results.get('existing_csv_analysis', {}):
            missing_components = results['existing_csv_analysis']['missing_components']
            
            for component in missing_components:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Data Enhancement',
                    'issue': component['component'],
                    'recommendation': f"Add {component['component']} to enhance validation capabilities",
                    'impact': component['impact']
                })
        
        # Check mathematical accuracy
        if 'mathematical_accuracy' in results.get('existing_csv_analysis', {}):
            accuracy = results['existing_csv_analysis']['mathematical_accuracy']
            if accuracy.get('accuracy_rate', 0) < 95:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Mathematical Accuracy',
                    'issue': 'Low mathematical accuracy',
                    'recommendation': 'Review and fix mathematical calculation formulas',
                    'impact': 'Ensures reliable regime formation calculations'
                })
        
        # Check regime diversity
        if 'validation_results' in results.get('real_data_validation', {}):
            validation = results['real_data_validation']['validation_results']
            if validation.get('regime_accuracy', {}).get('regime_diversity', 0) < 8:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Regime Diversity',
                    'issue': 'Limited regime diversity',
                    'recommendation': 'Review regime formation logic to ensure proper regime distribution',
                    'impact': 'Improves market condition coverage'
                })
        
        return recommendations

if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = ComprehensiveRegimeValidationAnalysis()
    results = analyzer.run_comprehensive_analysis()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MARKET REGIME VALIDATION ANALYSIS COMPLETED")
    print("="*80)
    print(f"Analysis timestamp: {results['analysis_timestamp']}")
    print(f"Results directory: {analyzer.output_dir}")
    print(f"Total recommendations: {len(results.get('recommendations', []))}")
    print("="*80)
