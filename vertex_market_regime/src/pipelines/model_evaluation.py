"""
Model Evaluation and Metrics for Training Pipeline
Comprehensive evaluation framework for market regime models

This module provides:
- Classification metrics (accuracy, F1, AUROC, precision, recall)
- Transition forecasting specific metrics (MAE, MSE, directional accuracy)
- Market-specific metrics (regime accuracy, transition detection)
- Model performance comparison framework
- Evaluation visualization and reporting
- Automated model validation checks
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error,
    log_loss
)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib
from pathlib import Path


class ClassificationMetrics:
    """Comprehensive classification metrics for regime prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None,
                         class_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            class_labels: Class label names (optional)
            
        Returns:
            Dictionary with comprehensive metrics
        """
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
            metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
            metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            if class_labels is None:
                class_labels = [f"class_{i}" for i in unique_classes]
            
            metrics["per_class_metrics"] = {}
            for i, class_label in enumerate(class_labels[:len(unique_classes)]):
                metrics["per_class_metrics"][class_label] = {
                    "precision": float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                    "recall": float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                    "f1_score": float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
                }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Classification report
            if class_labels:
                target_names = class_labels[:len(unique_classes)]
            else:
                target_names = None
            
            classification_rep = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
            metrics["classification_report"] = classification_rep
            
            # Probabilistic metrics (if probabilities available)
            if y_pred_proba is not None:
                try:
                    # Multi-class ROC AUC
                    if len(unique_classes) > 2:
                        metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
                        metrics["roc_auc_ovo"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='weighted'))
                    else:
                        # Binary classification
                        if y_pred_proba.shape[1] == 2:
                            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                        else:
                            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
                    
                    # Average precision (PR AUC)
                    label_binarizer = LabelBinarizer()
                    y_true_binary = label_binarizer.fit_transform(y_true)
                    
                    if len(unique_classes) > 2:
                        # Multi-class average precision
                        ap_scores = []
                        for i in range(len(unique_classes)):
                            if i < y_pred_proba.shape[1] and i < y_true_binary.shape[1]:
                                ap = average_precision_score(y_true_binary[:, i], y_pred_proba[:, i])
                                ap_scores.append(ap)
                        
                        if ap_scores:
                            metrics["average_precision"] = float(np.mean(ap_scores))
                    else:
                        # Binary classification
                        if y_pred_proba.shape[1] == 2:
                            metrics["average_precision"] = float(average_precision_score(y_true, y_pred_proba[:, 1]))
                        else:
                            metrics["average_precision"] = float(average_precision_score(y_true, y_pred_proba))
                    
                    # Log loss
                    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                    
                except Exception as e:
                    self.logger.warning(f"Could not calculate probabilistic metrics: {str(e)}")
            
            self.logger.info(f"Calculated classification metrics: accuracy={metrics['accuracy']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate classification metrics: {str(e)}")
            metrics["error"] = str(e)
        
        return metrics


class ForecastingMetrics:
    """Forecasting specific metrics for transition prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_true_proba: Optional[np.ndarray] = None,
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate forecasting-specific metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_true_proba: True probability values (optional)
            y_pred_proba: Predicted probability values (optional)
            
        Returns:
            Dictionary with forecasting metrics
        """
        
        metrics = {}
        
        try:
            # Regression metrics (if continuous values)
            if y_true.dtype in [np.float32, np.float64] and y_pred.dtype in [np.float32, np.float64]:
                metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
                metrics["mse"] = float(mean_squared_error(y_true, y_pred))
                metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                
                # MAPE (Mean Absolute Percentage Error)
                non_zero_mask = y_true != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                    metrics["mape"] = float(mape)
                
                # Directional accuracy (for financial data)
                if len(y_true) > 1:
                    y_true_direction = np.diff(y_true) > 0
                    y_pred_direction = np.diff(y_pred) > 0
                    
                    if len(y_true_direction) > 0:
                        directional_accuracy = np.mean(y_true_direction == y_pred_direction)
                        metrics["directional_accuracy"] = float(directional_accuracy)
            
            # Transition detection metrics (for regime changes)
            if len(np.unique(y_true)) <= 10:  # Assuming discrete regime states
                # Transition points detection
                y_true_changes = np.diff(y_true) != 0
                y_pred_changes = np.diff(y_pred) != 0
                
                if len(y_true_changes) > 0:
                    # Transition detection accuracy
                    transition_accuracy = np.mean(y_true_changes == y_pred_changes)
                    metrics["transition_detection_accuracy"] = float(transition_accuracy)
                    
                    # Transition detection precision/recall
                    if np.any(y_pred_changes):
                        transition_precision = np.sum(y_true_changes & y_pred_changes) / np.sum(y_pred_changes)
                        metrics["transition_precision"] = float(transition_precision)
                    
                    if np.any(y_true_changes):
                        transition_recall = np.sum(y_true_changes & y_pred_changes) / np.sum(y_true_changes)
                        metrics["transition_recall"] = float(transition_recall)
                        
                        if "transition_precision" in metrics and metrics["transition_precision"] > 0:
                            transition_f1 = 2 * (metrics["transition_precision"] * transition_recall) / (metrics["transition_precision"] + transition_recall)
                            metrics["transition_f1"] = float(transition_f1)
            
            # Regime stability metrics
            if len(np.unique(y_true)) <= 10:
                # Calculate how long each regime lasts on average
                regime_lengths_true = self._calculate_regime_lengths(y_true)
                regime_lengths_pred = self._calculate_regime_lengths(y_pred)
                
                if regime_lengths_true and regime_lengths_pred:
                    avg_regime_length_true = np.mean(regime_lengths_true)
                    avg_regime_length_pred = np.mean(regime_lengths_pred)
                    
                    metrics["avg_regime_length_true"] = float(avg_regime_length_true)
                    metrics["avg_regime_length_pred"] = float(avg_regime_length_pred)
                    
                    # Regime length error
                    regime_length_error = abs(avg_regime_length_true - avg_regime_length_pred) / avg_regime_length_true
                    metrics["regime_length_error"] = float(regime_length_error)
            
            self.logger.info("Calculated forecasting metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate forecasting metrics: {str(e)}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _calculate_regime_lengths(self, y: np.ndarray) -> List[int]:
        """Calculate lengths of consecutive regime periods"""
        if len(y) == 0:
            return []
        
        lengths = []
        current_regime = y[0]
        current_length = 1
        
        for i in range(1, len(y)):
            if y[i] == current_regime:
                current_length += 1
            else:
                lengths.append(current_length)
                current_regime = y[i]
                current_length = 1
        
        lengths.append(current_length)  # Add final regime length
        return lengths


class MarketSpecificMetrics:
    """Market regime specific evaluation metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         regime_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate market-specific metrics for regime prediction
        
        Args:
            y_true: True regime labels
            y_pred: Predicted regime labels
            timestamps: Timestamps for time-based analysis (optional)
            regime_labels: Regime label names (optional)
            
        Returns:
            Dictionary with market-specific metrics
        """
        
        metrics = {}
        
        try:
            if regime_labels is None:
                unique_regimes = np.unique(np.concatenate([y_true, y_pred]))
                regime_labels = [f"regime_{i}" for i in unique_regimes]
            
            # Regime-wise accuracy
            regime_accuracies = {}
            for i, regime_label in enumerate(regime_labels):
                regime_mask = y_true == i
                if np.any(regime_mask):
                    regime_accuracy = accuracy_score(y_true[regime_mask], y_pred[regime_mask])
                    regime_accuracies[regime_label] = float(regime_accuracy)
            
            metrics["regime_wise_accuracy"] = regime_accuracies
            
            # Overall regime stability score
            regime_stability = self._calculate_regime_stability(y_true, y_pred)
            metrics["regime_stability_score"] = float(regime_stability)
            
            # Regime transition analysis
            transition_analysis = self._analyze_regime_transitions(y_true, y_pred, regime_labels)
            metrics["transition_analysis"] = transition_analysis
            
            # Market volatility correlation (if this is financial data)
            if len(y_true) > 1:
                volatility_correlation = self._calculate_volatility_correlation(y_true, y_pred)
                metrics["volatility_correlation"] = float(volatility_correlation)
            
            # Time-based performance (if timestamps provided)
            if timestamps is not None:
                time_based_metrics = self._calculate_time_based_metrics(y_true, y_pred, timestamps)
                metrics["time_based_metrics"] = time_based_metrics
            
            self.logger.info("Calculated market-specific metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate market-specific metrics: {str(e)}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _calculate_regime_stability(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate regime stability score"""
        
        # Count regime changes in true vs predicted
        true_changes = np.sum(np.diff(y_true) != 0)
        pred_changes = np.sum(np.diff(y_pred) != 0)
        
        if true_changes == 0 and pred_changes == 0:
            return 1.0  # Perfect stability
        
        # Stability score based on change frequency similarity
        total_possible_changes = len(y_true) - 1
        if total_possible_changes == 0:
            return 1.0
        
        stability_score = 1.0 - abs(true_changes - pred_changes) / total_possible_changes
        return max(0.0, stability_score)
    
    def _analyze_regime_transitions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  regime_labels: List[str]) -> Dict[str, Any]:
        """Analyze regime transition patterns"""
        
        analysis = {}
        
        # Transition matrices
        unique_regimes = np.unique(np.concatenate([y_true, y_pred]))
        n_regimes = len(unique_regimes)
        
        # True transition matrix
        true_transitions = np.zeros((n_regimes, n_regimes))
        for i in range(len(y_true) - 1):
            from_regime = int(y_true[i])
            to_regime = int(y_true[i + 1])
            if from_regime < n_regimes and to_regime < n_regimes:
                true_transitions[from_regime, to_regime] += 1
        
        # Predicted transition matrix
        pred_transitions = np.zeros((n_regimes, n_regimes))
        for i in range(len(y_pred) - 1):
            from_regime = int(y_pred[i])
            to_regime = int(y_pred[i + 1])
            if from_regime < n_regimes and to_regime < n_regimes:
                pred_transitions[from_regime, to_regime] += 1
        
        analysis["true_transition_matrix"] = true_transitions.tolist()
        analysis["pred_transition_matrix"] = pred_transitions.tolist()
        
        # Transition probability differences
        true_probs = true_transitions / (true_transitions.sum(axis=1, keepdims=True) + 1e-8)
        pred_probs = pred_transitions / (pred_transitions.sum(axis=1, keepdims=True) + 1e-8)
        
        transition_error = np.mean(np.abs(true_probs - pred_probs))
        analysis["transition_probability_error"] = float(transition_error)
        
        return analysis
    
    def _calculate_volatility_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate correlation between regime changes and volatility"""
        
        # Simple volatility proxy based on regime changes
        true_volatility = np.abs(np.diff(y_true.astype(float)))
        pred_volatility = np.abs(np.diff(y_pred.astype(float)))
        
        if len(true_volatility) > 1 and np.std(true_volatility) > 0 and np.std(pred_volatility) > 0:
            correlation = np.corrcoef(true_volatility, pred_volatility)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _calculate_time_based_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    timestamps: np.ndarray) -> Dict[str, Any]:
        """Calculate time-based performance metrics"""
        
        time_metrics = {}
        
        try:
            # Convert timestamps to datetime if needed
            if timestamps.dtype == 'object':
                timestamps = pd.to_datetime(timestamps)
            
            # Group by time periods (e.g., monthly)
            df = pd.DataFrame({
                'timestamp': timestamps,
                'y_true': y_true,
                'y_pred': y_pred
            })
            
            # Monthly accuracy
            df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
            monthly_accuracy = df.groupby('month').apply(
                lambda x: accuracy_score(x['y_true'], x['y_pred'])
            ).to_dict()
            
            time_metrics["monthly_accuracy"] = {str(k): float(v) for k, v in monthly_accuracy.items()}
            
            # Quarterly accuracy
            df['quarter'] = pd.to_datetime(df['timestamp']).dt.to_period('Q')
            quarterly_accuracy = df.groupby('quarter').apply(
                lambda x: accuracy_score(x['y_true'], x['y_pred'])
            ).to_dict()
            
            time_metrics["quarterly_accuracy"] = {str(k): float(v) for k, v in quarterly_accuracy.items()}
            
        except Exception as e:
            self.logger.warning(f"Could not calculate time-based metrics: {str(e)}")
            time_metrics["error"] = str(e)
        
        return time_metrics


class ModelComparisonFramework:
    """Framework for comparing multiple model performances"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance of multiple models
        
        Args:
            model_results: Dictionary with model names as keys and evaluation results as values
            
        Returns:
            Dictionary with comparison results
        """
        
        comparison = {
            "model_ranking": {},
            "metric_comparison": {},
            "best_models": {},
            "performance_summary": {}
        }
        
        try:
            # Extract common metrics
            common_metrics = ["accuracy", "f1_weighted", "roc_auc_ovr"]
            
            for metric in common_metrics:
                metric_values = {}
                
                for model_name, results in model_results.items():
                    if metric in results:
                        metric_values[model_name] = results[metric]
                
                if metric_values:
                    # Rank models by this metric
                    ranked_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                    comparison["metric_comparison"][metric] = {
                        "values": metric_values,
                        "ranking": [{"model": model, "value": value} for model, value in ranked_models],
                        "best_model": ranked_models[0][0] if ranked_models else None,
                        "best_value": ranked_models[0][1] if ranked_models else None
                    }
            
            # Overall ranking (weighted average of metrics)
            overall_scores = {}
            metric_weights = {"accuracy": 0.4, "f1_weighted": 0.4, "roc_auc_ovr": 0.2}
            
            for model_name in model_results.keys():
                score = 0.0
                total_weight = 0.0
                
                for metric, weight in metric_weights.items():
                    if metric in model_results[model_name]:
                        score += model_results[model_name][metric] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    overall_scores[model_name] = score / total_weight
            
            if overall_scores:
                ranked_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
                comparison["model_ranking"] = {
                    "overall_scores": overall_scores,
                    "ranking": [{"model": model, "score": score} for model, score in ranked_overall],
                    "best_model": ranked_overall[0][0] if ranked_overall else None
                }
            
            # Best models per metric
            for metric, metric_data in comparison["metric_comparison"].items():
                if metric_data["best_model"]:
                    comparison["best_models"][metric] = metric_data["best_model"]
            
            # Performance summary
            comparison["performance_summary"] = {
                "total_models": len(model_results),
                "metrics_compared": len(common_metrics),
                "comparison_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Compared {len(model_results)} models across {len(common_metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {str(e)}")
            comparison["error"] = str(e)
        
        return comparison


class EvaluationVisualizer:
    """Create evaluation visualizations and reports"""
    
    def __init__(self, output_dir: str = "./evaluation_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_confusion_matrix_plot(self, confusion_matrix: np.ndarray, 
                                   class_labels: List[str],
                                   model_name: str) -> str:
        """Create confusion matrix visualization"""
        
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            output_path = self.output_dir / f"confusion_matrix_{model_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix plot saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create confusion matrix plot: {str(e)}")
            return ""
    
    def create_model_comparison_plot(self, comparison_results: Dict[str, Any]) -> str:
        """Create model comparison visualization"""
        
        try:
            metrics = list(comparison_results["metric_comparison"].keys())
            models = list(comparison_results["model_ranking"]["overall_scores"].keys())
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16)
            
            # Metric comparison bar chart
            if len(metrics) > 0:
                ax1 = axes[0, 0]
                metric_data = comparison_results["metric_comparison"][metrics[0]]
                values = [metric_data["values"].get(model, 0) for model in models]
                ax1.bar(models, values)
                ax1.set_title(f'{metrics[0].upper()} Comparison')
                ax1.set_ylabel('Score')
                plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # Overall ranking
            ax2 = axes[0, 1]
            ranking_data = comparison_results["model_ranking"]
            if "overall_scores" in ranking_data:
                models_ranked = [item["model"] for item in ranking_data["ranking"]]
                scores_ranked = [item["score"] for item in ranking_data["ranking"]]
                ax2.barh(models_ranked, scores_ranked)
                ax2.set_title('Overall Model Ranking')
                ax2.set_xlabel('Overall Score')
            
            # Metrics heatmap
            ax3 = axes[1, 0]
            if len(metrics) > 1:
                metric_matrix = []
                for model in models:
                    model_scores = []
                    for metric in metrics:
                        score = comparison_results["metric_comparison"][metric]["values"].get(model, 0)
                        model_scores.append(score)
                    metric_matrix.append(model_scores)
                
                sns.heatmap(metric_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                           xticklabels=metrics, yticklabels=models, ax=ax3)
                ax3.set_title('Model Performance Heatmap')
            
            # Best models per metric
            ax4 = axes[1, 1]
            if "best_models" in comparison_results:
                best_models_data = comparison_results["best_models"]
                metrics_list = list(best_models_data.keys())
                models_list = list(best_models_data.values())
                
                model_counts = {model: models_list.count(model) for model in set(models_list)}
                ax4.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.1f%%')
                ax4.set_title('Best Model Distribution')
            
            plt.tight_layout()
            output_path = self.output_dir / "model_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Model comparison plot saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create model comparison plot: {str(e)}")
            return ""
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 model_name: str) -> str:
        """Generate comprehensive evaluation report"""
        
        try:
            report_content = f"""
# Model Evaluation Report: {model_name}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Classification Metrics
"""
            
            if "classification_metrics" in evaluation_results:
                cm = evaluation_results["classification_metrics"]
                report_content += f"""
- **Accuracy**: {cm.get('accuracy', 'N/A'):.4f}
- **Precision (Weighted)**: {cm.get('precision_weighted', 'N/A'):.4f}
- **Recall (Weighted)**: {cm.get('recall_weighted', 'N/A'):.4f}
- **F1-Score (Weighted)**: {cm.get('f1_weighted', 'N/A'):.4f}
- **ROC AUC**: {cm.get('roc_auc_ovr', cm.get('roc_auc', 'N/A')):.4f}
"""
            
            if "forecasting_metrics" in evaluation_results:
                fm = evaluation_results["forecasting_metrics"]
                report_content += f"""

## Forecasting Metrics
- **Directional Accuracy**: {fm.get('directional_accuracy', 'N/A'):.4f}
- **Transition Detection F1**: {fm.get('transition_f1', 'N/A'):.4f}
- **Regime Stability**: {fm.get('regime_length_error', 'N/A'):.4f}
"""
            
            if "market_metrics" in evaluation_results:
                mm = evaluation_results["market_metrics"]
                report_content += f"""

## Market-Specific Metrics
- **Regime Stability Score**: {mm.get('regime_stability_score', 'N/A'):.4f}
- **Volatility Correlation**: {mm.get('volatility_correlation', 'N/A'):.4f}
"""
            
            report_content += """

## Model Performance Summary
This model has been evaluated across multiple dimensions including classification accuracy, 
forecasting performance, and market-specific metrics. The results provide a comprehensive 
view of the model's capabilities for market regime prediction.
"""
            
            output_path = self.output_dir / f"evaluation_report_{model_name}.md"
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Evaluation report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation report: {str(e)}")
            return ""


class ModelValidationChecks:
    """Automated model validation checks"""
    
    def __init__(self, validation_config: Dict[str, Any]):
        self.config = validation_config
        self.logger = logging.getLogger(__name__)
        
    def validate_model_performance(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model performance against thresholds
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Dictionary with validation results
        """
        
        validation_results = {
            "passed": True,
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Accuracy threshold check
            min_accuracy = self.config.get("min_accuracy", 0.75)
            if "classification_metrics" in evaluation_results:
                accuracy = evaluation_results["classification_metrics"].get("accuracy", 0.0)
                if accuracy < min_accuracy:
                    validation_results["passed"] = False
                    validation_results["errors"].append(f"Accuracy {accuracy:.4f} below threshold {min_accuracy}")
                
                validation_results["checks"]["accuracy_check"] = {
                    "passed": accuracy >= min_accuracy,
                    "value": accuracy,
                    "threshold": min_accuracy
                }
            
            # Overfitting check
            max_overfitting = self.config.get("max_overfitting_threshold", 0.05)
            if "training_metrics" in evaluation_results and "validation_metrics" in evaluation_results:
                train_acc = evaluation_results["training_metrics"].get("accuracy", 0.0)
                val_acc = evaluation_results["validation_metrics"].get("accuracy", 0.0)
                overfitting = train_acc - val_acc
                
                if overfitting > max_overfitting:
                    validation_results["warnings"].append(f"Potential overfitting detected: {overfitting:.4f}")
                
                validation_results["checks"]["overfitting_check"] = {
                    "passed": overfitting <= max_overfitting,
                    "value": overfitting,
                    "threshold": max_overfitting
                }
            
            # Feature importance check
            if "feature_importance" in evaluation_results:
                feature_imp = evaluation_results["feature_importance"]
                if isinstance(feature_imp, dict) and len(feature_imp) > 0:
                    # Check if feature importance is well distributed
                    importance_values = list(feature_imp.values())
                    if all(isinstance(v, (int, float)) for v in importance_values):
                        max_importance = max(importance_values)
                        if max_importance > 0.5:  # Single feature dominates
                            validation_results["warnings"].append("Single feature has high importance (>50%)")
                
                validation_results["checks"]["feature_importance_check"] = {
                    "passed": True,  # This is usually just a warning
                    "note": "Feature importance distribution checked"
                }
            
            self.logger.info(f"Model validation completed: {'PASSED' if validation_results['passed'] else 'FAILED'}")
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            validation_results["passed"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results


class ModelEvaluationPipeline:
    """Main model evaluation pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.classification_metrics = ClassificationMetrics()
        self.forecasting_metrics = ForecastingMetrics()
        self.market_metrics = MarketSpecificMetrics()
        self.comparison_framework = ModelComparisonFramework()
        self.visualizer = EvaluationVisualizer(config.get("output_dir", "./evaluation_outputs"))
        self.validator = ModelValidationChecks(config.get("validation", {}))
        
    def evaluate_model(self, model_predictions: Dict[str, np.ndarray],
                      ground_truth: Dict[str, np.ndarray],
                      model_name: str,
                      model_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model_predictions: Dictionary with model predictions
            ground_truth: Dictionary with ground truth values
            model_name: Name of the model being evaluated
            model_metadata: Additional model metadata
            
        Returns:
            Dictionary with complete evaluation results
        """
        
        evaluation_results = {
            "model_name": model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_metadata": model_metadata or {}
        }
        
        try:
            # Extract predictions and ground truth
            y_true = ground_truth.get("labels", np.array([]))
            y_pred = model_predictions.get("predictions", np.array([]))
            y_pred_proba = model_predictions.get("probabilities", None)
            timestamps = ground_truth.get("timestamps", None)
            
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("No predictions or ground truth provided")
            
            # Classification metrics
            self.logger.info("Calculating classification metrics...")
            evaluation_results["classification_metrics"] = self.classification_metrics.calculate_metrics(
                y_true, y_pred, y_pred_proba
            )
            
            # Forecasting metrics
            self.logger.info("Calculating forecasting metrics...")
            evaluation_results["forecasting_metrics"] = self.forecasting_metrics.calculate_metrics(
                y_true, y_pred
            )
            
            # Market-specific metrics
            self.logger.info("Calculating market-specific metrics...")
            evaluation_results["market_metrics"] = self.market_metrics.calculate_metrics(
                y_true, y_pred, timestamps
            )
            
            # Create visualizations
            self.logger.info("Creating visualizations...")
            if "confusion_matrix" in evaluation_results["classification_metrics"]:
                cm = np.array(evaluation_results["classification_metrics"]["confusion_matrix"])
                unique_labels = np.unique(np.concatenate([y_true, y_pred]))
                class_labels = [f"class_{i}" for i in unique_labels]
                
                confusion_plot_path = self.visualizer.create_confusion_matrix_plot(
                    cm, class_labels, model_name
                )
                evaluation_results["visualization_paths"] = {"confusion_matrix": confusion_plot_path}
            
            # Generate report
            self.logger.info("Generating evaluation report...")
            report_path = self.visualizer.generate_evaluation_report(evaluation_results, model_name)
            evaluation_results["report_path"] = report_path
            
            # Validate model performance
            self.logger.info("Validating model performance...")
            validation_results = self.validator.validate_model_performance(evaluation_results)
            evaluation_results["validation_results"] = validation_results
            
            evaluation_results["status"] = "success"
            self.logger.info(f"Model evaluation completed successfully for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            evaluation_results["status"] = "failed"
            evaluation_results["error"] = str(e)
        
        return evaluation_results


# Utility functions for KFP component integration
def evaluate_trained_model(
    model_path: str,
    test_data_path: str,
    evaluation_config: Dict[str, Any],
    output_metrics_path: str,
    output_report_path: str
) -> Dict[str, Any]:
    """
    Evaluate trained model for KFP component
    
    This function serves as the entry point for model evaluation in KFP components
    """
    
    try:
        # Load test data
        test_data = pd.read_parquet(test_data_path)
        
        # Separate features and target
        feature_columns = [col for col in test_data.columns if col not in ['target', 'timestamp']]
        X_test = test_data[feature_columns].values
        y_test = test_data['target'].values
        timestamps = test_data.get('timestamp', None)
        
        # Load model
        model_data = joblib.load(model_path)
        model = model_data["model"]
        
        # Make predictions
        y_pred = model.predict(X_test)
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        # Prepare data for evaluation
        model_predictions = {
            "predictions": y_pred,
            "probabilities": y_pred_proba
        }
        
        ground_truth = {
            "labels": y_test,
            "timestamps": timestamps.values if timestamps is not None else None
        }
        
        # Run evaluation
        evaluation_pipeline = ModelEvaluationPipeline(evaluation_config)
        results = evaluation_pipeline.evaluate_model(
            model_predictions=model_predictions,
            ground_truth=ground_truth,
            model_name=Path(model_path).stem,
            model_metadata=model_data.get("model_metadata", {})
        )
        
        # Save results
        with open(output_metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }