#!/usr/bin/env python3
"""
Market Regime Validation and Comparison Tools
============================================
Tools for validating regime classifications, comparing different
configurations, and ensuring accuracy in 1-year analysis.

Features:
- Regime classification validation
- Configuration comparison (8-regime vs 18-regime)
- Performance benchmarking
- Anomaly detection
- Golden output comparison
- Statistical analysis

Author: Market Regime Testing Team
Date: 2025-06-27
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import hashlib

# Import logging system
from enhanced_logging_system import get_logger, RegimeTransition

@dataclass
class ValidationResult:
    """Container for validation results"""
    test_name: str
    passed: bool
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: np.ndarray
    anomalies: List[Dict[str, Any]]
    warnings: List[str]
    details: Dict[str, Any]

@dataclass
class ComparisonResult:
    """Container for comparison results"""
    config1_name: str
    config2_name: str
    agreement_rate: float
    regime_mapping: Dict[str, str]
    transition_similarity: float
    performance_diff: Dict[str, float]
    divergence_points: List[Dict[str, Any]]
    recommendation: str

class MarketRegimeValidator:
    """Validation tools for Market Regime analysis"""
    
    def __init__(self, golden_data_path: Optional[str] = None):
        """
        Initialize validator
        
        Args:
            golden_data_path: Path to golden/reference data
        """
        self.logger = get_logger()
        self.golden_data = None
        
        if golden_data_path and os.path.exists(golden_data_path):
            self.golden_data = pd.read_csv(golden_data_path)
            
        # Regime mappings for 8 vs 18 regime comparison
        self.regime_8_to_18_mapping = {
            'Strong_Bullish': ['Strong_Bullish_Low_Vol', 'Strong_Bullish_Med_Vol'],
            'Bullish': ['Moderate_Bullish', 'Weak_Bullish'],
            'Bullish_Volatile': ['Bullish_Volatile', 'Bullish_Consolidation'],
            'Neutral': ['Neutral_Balanced', 'Neutral_Uncertain'],
            'Neutral_Volatile': ['Volatility_Expansion', 'Transition_Regime'],
            'Bearish_Volatile': ['Bearish_Volatile', 'Bearish_Consolidation'],
            'Bearish': ['Moderate_Bearish', 'Weak_Bearish'],
            'Strong_Bearish': ['Strong_Bearish_Low_Vol', 'Strong_Bearish_Med_Vol']
        }
        
        # Validation thresholds
        self.thresholds = {
            'min_confidence': 0.3,
            'max_confidence': 1.0,
            'max_transition_rate': 0.1,  # Max 10% transitions per hour
            'min_regime_duration': 60,    # Minimum 60 seconds
            'anomaly_z_score': 3.0
        }
        
    def validate_regime_classifications(self, 
                                      results_df: pd.DataFrame,
                                      config_name: str) -> ValidationResult:
        """
        Validate regime classifications against rules and golden data
        
        Args:
            results_df: DataFrame with regime classifications
            config_name: Configuration name for identification
            
        Returns:
            ValidationResult with detailed validation metrics
        """
        anomalies = []
        warnings = []
        
        # Basic validation checks
        if 'regime_classification' not in results_df.columns:
            raise ValueError("Missing 'regime_classification' column")
            
        if 'confidence_score' not in results_df.columns:
            warnings.append("Missing 'confidence_score' column - using default validation")
            
        # 1. Confidence score validation
        if 'confidence_score' in results_df.columns:
            invalid_confidence = results_df[
                (results_df['confidence_score'] < self.thresholds['min_confidence']) |
                (results_df['confidence_score'] > self.thresholds['max_confidence'])
            ]
            
            if len(invalid_confidence) > 0:
                anomalies.extend([
                    {
                        'type': 'invalid_confidence',
                        'index': idx,
                        'value': row['confidence_score'],
                        'regime': row['regime_classification']
                    }
                    for idx, row in invalid_confidence.iterrows()
                ])
        
        # 2. Regime transition rate validation
        regime_changes = results_df['regime_classification'].ne(
            results_df['regime_classification'].shift()
        )
        transition_rate = regime_changes.sum() / len(results_df)
        
        if transition_rate > self.thresholds['max_transition_rate']:
            warnings.append(
                f"High transition rate: {transition_rate:.2%} "
                f"(threshold: {self.thresholds['max_transition_rate']:.2%})"
            )
        
        # 3. Regime duration validation
        regime_groups = results_df.groupby(
            (results_df['regime_classification'] != 
             results_df['regime_classification'].shift()).cumsum()
        )
        
        short_durations = []
        for _, group in regime_groups:
            if len(group) < self.thresholds['min_regime_duration']:
                short_durations.append({
                    'regime': group.iloc[0]['regime_classification'],
                    'duration': len(group),
                    'start_idx': group.index[0]
                })
        
        if short_durations:
            warnings.append(
                f"Found {len(short_durations)} regimes with duration "
                f"< {self.thresholds['min_regime_duration']} records"
            )
            anomalies.extend([
                {
                    'type': 'short_duration',
                    'details': sd
                }
                for sd in short_durations[:10]  # Limit to 10
            ])
        
        # 4. Statistical anomaly detection
        if 'confidence_score' in results_df.columns:
            z_scores = np.abs(stats.zscore(results_df['confidence_score'].fillna(0)))
            statistical_anomalies = results_df[z_scores > self.thresholds['anomaly_z_score']]
            
            if len(statistical_anomalies) > 0:
                anomalies.extend([
                    {
                        'type': 'statistical_anomaly',
                        'index': idx,
                        'z_score': z_scores[idx],
                        'confidence': row['confidence_score']
                    }
                    for idx, row in statistical_anomalies.iterrows()
                ])
        
        # 5. Golden data comparison (if available)
        accuracy = 1.0
        precision = {}
        recall = {}
        f1_score = {}
        conf_matrix = None
        
        if self.golden_data is not None and len(self.golden_data) == len(results_df):
            # Calculate classification metrics
            y_true = self.golden_data['regime_classification']
            y_pred = results_df['regime_classification']
            
            # Overall accuracy
            accuracy = (y_true == y_pred).mean()
            
            # Per-class metrics
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            for regime in results_df['regime_classification'].unique():
                if regime in report:
                    precision[regime] = report[regime]['precision']
                    recall[regime] = report[regime]['recall']
                    f1_score[regime] = report[regime]['f1-score']
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Create validation result
        result = ValidationResult(
            test_name=config_name,
            passed=len(anomalies) == 0 and accuracy > 0.9,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=conf_matrix,
            anomalies=anomalies[:100],  # Limit anomalies
            warnings=warnings,
            details={
                'total_records': len(results_df),
                'unique_regimes': results_df['regime_classification'].nunique(),
                'transition_rate': transition_rate,
                'avg_confidence': results_df.get('confidence_score', pd.Series()).mean()
            }
        )
        
        # Log validation result
        self.logger.log_analysis_checkpoint(
            checkpoint_name=f"validation_{config_name}",
            records_processed=len(results_df),
            regime_distribution=results_df['regime_classification'].value_counts().to_dict(),
            confidence_stats={
                'mean': results_df.get('confidence_score', pd.Series()).mean(),
                'std': results_df.get('confidence_score', pd.Series()).std()
            },
            custom_data={'validation_passed': result.passed}
        )
        
        return result
    
    def compare_configurations(self,
                             results1: pd.DataFrame,
                             config1_name: str,
                             results2: pd.DataFrame,
                             config2_name: str) -> ComparisonResult:
        """
        Compare two different configurations (e.g., 8-regime vs 18-regime)
        
        Args:
            results1: First configuration results
            config1_name: First configuration name
            results2: Second configuration results
            config2_name: Second configuration name
            
        Returns:
            ComparisonResult with detailed comparison metrics
        """
        if len(results1) != len(results2):
            raise ValueError("Results must have the same length for comparison")
        
        # Calculate agreement rate
        if '8_regime' in config1_name.lower() and '18_regime' in config2_name.lower():
            # Map 18-regime to 8-regime for comparison
            mapped_results2 = self._map_18_to_8_regime(results2['regime_classification'])
            agreement = (results1['regime_classification'] == mapped_results2).mean()
            regime_mapping = self.regime_8_to_18_mapping
        else:
            # Direct comparison
            agreement = (results1['regime_classification'] == results2['regime_classification']).mean()
            regime_mapping = {}
        
        # Find divergence points
        divergence_mask = results1['regime_classification'] != results2['regime_classification']
        divergence_points = []
        
        if divergence_mask.any():
            divergence_indices = divergence_mask[divergence_mask].index[:100]  # Limit to 100
            
            for idx in divergence_indices:
                divergence_points.append({
                    'index': int(idx),
                    'timestamp': results1.iloc[idx].get('timestamp', idx),
                    'regime1': results1.iloc[idx]['regime_classification'],
                    'regime2': results2.iloc[idx]['regime_classification'],
                    'confidence1': results1.iloc[idx].get('confidence_score', None),
                    'confidence2': results2.iloc[idx].get('confidence_score', None)
                })
        
        # Calculate transition similarity
        transitions1 = results1['regime_classification'].ne(results1['regime_classification'].shift())
        transitions2 = results2['regime_classification'].ne(results2['regime_classification'].shift())
        transition_similarity = (transitions1 == transitions2).mean()
        
        # Performance comparison
        performance_diff = {}
        
        if 'processing_time' in results1.columns and 'processing_time' in results2.columns:
            performance_diff['avg_processing_time'] = (
                results2['processing_time'].mean() - results1['processing_time'].mean()
            )
        
        if 'confidence_score' in results1.columns and 'confidence_score' in results2.columns:
            performance_diff['avg_confidence_diff'] = (
                results2['confidence_score'].mean() - results1['confidence_score'].mean()
            )
            performance_diff['confidence_correlation'] = (
                results1['confidence_score'].corr(results2['confidence_score'])
            )
        
        # Generate recommendation
        if agreement > 0.95:
            recommendation = "Configurations are highly consistent. Either can be used."
        elif agreement > 0.85:
            recommendation = "Configurations show good agreement with minor differences."
        elif agreement > 0.70:
            recommendation = "Moderate agreement. Review divergence points for critical decisions."
        else:
            recommendation = "Low agreement. Configurations produce significantly different results."
        
        if '18_regime' in config2_name and transition_similarity > 0.9:
            recommendation += " Consider using 18-regime for more granular analysis."
        
        return ComparisonResult(
            config1_name=config1_name,
            config2_name=config2_name,
            agreement_rate=agreement,
            regime_mapping=regime_mapping,
            transition_similarity=transition_similarity,
            performance_diff=performance_diff,
            divergence_points=divergence_points,
            recommendation=recommendation
        )
    
    def _map_18_to_8_regime(self, regime_series: pd.Series) -> pd.Series:
        """Map 18-regime classifications to 8-regime"""
        reverse_mapping = {}
        for regime_8, regime_18_list in self.regime_8_to_18_mapping.items():
            for regime_18 in regime_18_list:
                reverse_mapping[regime_18] = regime_8
        
        return regime_series.map(lambda x: reverse_mapping.get(x, x))
    
    def validate_golden_output(self,
                             results: pd.DataFrame,
                             golden_path: str,
                             tolerance: float = 0.95) -> Dict[str, Any]:
        """
        Validate results against golden output
        
        Args:
            results: Results to validate
            golden_path: Path to golden output
            tolerance: Minimum accuracy threshold
            
        Returns:
            Validation summary
        """
        golden_df = pd.read_csv(golden_path)
        
        if len(results) != len(golden_df):
            return {
                'valid': False,
                'error': f"Length mismatch: {len(results)} vs {len(golden_df)}"
            }
        
        # Compare regime classifications
        regime_match = (results['regime_classification'] == golden_df['regime_classification']).mean()
        
        # Compare confidence scores if available
        confidence_correlation = None
        if 'confidence_score' in results.columns and 'confidence_score' in golden_df.columns:
            confidence_correlation = results['confidence_score'].corr(golden_df['confidence_score'])
        
        # Find mismatches
        mismatches = results[results['regime_classification'] != golden_df['regime_classification']]
        
        return {
            'valid': regime_match >= tolerance,
            'regime_accuracy': regime_match,
            'confidence_correlation': confidence_correlation,
            'mismatch_count': len(mismatches),
            'mismatch_percentage': len(mismatches) / len(results) * 100,
            'first_mismatches': mismatches.head(10).to_dict('records') if len(mismatches) > 0 else []
        }
    
    def generate_validation_report(self,
                                 validation_results: List[ValidationResult],
                                 comparison_results: List[ComparisonResult],
                                 output_path: str):
        """Generate comprehensive validation report"""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_validations': len(validation_results),
                'total_comparisons': len(comparison_results)
            },
            'validation_summary': {
                'passed': sum(1 for v in validation_results if v.passed),
                'failed': sum(1 for v in validation_results if not v.passed),
                'average_accuracy': np.mean([v.accuracy for v in validation_results])
            },
            'validation_details': [asdict(v) for v in validation_results],
            'comparison_summary': {
                'average_agreement': np.mean([c.agreement_rate for c in comparison_results]),
                'average_transition_similarity': np.mean([c.transition_similarity for c in comparison_results])
            },
            'comparison_details': [asdict(c) for c in comparison_results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_validation_plots(validation_results, comparison_results, output_path)
        
        return output_path
    
    def _generate_validation_plots(self,
                                 validation_results: List[ValidationResult],
                                 comparison_results: List[ComparisonResult],
                                 base_path: str):
        """Generate validation visualization plots"""
        output_dir = Path(base_path).parent
        
        # Plot 1: Accuracy comparison
        if validation_results:
            plt.figure(figsize=(10, 6))
            names = [v.test_name for v in validation_results]
            accuracies = [v.accuracy for v in validation_results]
            
            plt.bar(names, accuracies)
            plt.axhline(y=0.9, color='r', linestyle='--', label='Target Accuracy')
            plt.xlabel('Configuration')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy by Configuration')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'validation_accuracy.png')
            plt.close()
        
        # Plot 2: Agreement rates
        if comparison_results:
            plt.figure(figsize=(10, 6))
            labels = [f"{c.config1_name} vs {c.config2_name}" for c in comparison_results]
            agreements = [c.agreement_rate for c in comparison_results]
            
            plt.bar(labels, agreements)
            plt.axhline(y=0.85, color='r', linestyle='--', label='Good Agreement')
            plt.xlabel('Comparison')
            plt.ylabel('Agreement Rate')
            plt.title('Configuration Agreement Rates')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'comparison_agreement.png')
            plt.close()

# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = MarketRegimeValidator()
    
    # Create sample data
    sample_results = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'regime_classification': np.random.choice(
            ['Bullish_Momentum', 'Neutral_Consolidation', 'Bearish_Momentum'],
            size=1000
        ),
        'confidence_score': np.random.uniform(0.5, 0.95, 1000)
    })
    
    # Validate results
    validation = validator.validate_regime_classifications(
        sample_results,
        "test_config"
    )
    
    print(f"Validation passed: {validation.passed}")
    print(f"Accuracy: {validation.accuracy:.2%}")
    print(f"Anomalies found: {len(validation.anomalies)}")
    print(f"Warnings: {validation.warnings}")