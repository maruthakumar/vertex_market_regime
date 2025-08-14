"""
Unit Tests for Training Pipeline Components
Comprehensive testing suite for all pipeline components

This module provides:
- Unit tests for all pipeline components
- Mock data and fixtures for testing
- Component isolation testing
- Performance validation tests
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from pipelines.data_preparation import (
    DataPreparationConfig, DataQualityChecker, TimeBasedSplitter,
    FeatureProcessor, DataExporter, BigQueryDataLoader
)
from pipelines.model_evaluation import (
    ClassificationMetrics, ForecastingMetrics, MarketSpecificMetrics,
    ModelComparisonFramework, EvaluationVisualizer
)
from ml.baseline_models import (
    TabNetRegimeClassifier, XGBoostRegimeClassifier, 
    ModelSelectionFramework
)


class TestDataPreparationComponents:
    """Test suite for data preparation components"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'project': {
                'project_id': 'test-project',
                'location': 'us-central1',
                'staging_bucket': 'test-bucket'
            },
            'data': {
                'source': {'full_table_id': 'test-project.test_dataset.test_table'},
                'splits': {'train_ratio': 0.7, 'validation_ratio': 0.2, 'test_ratio': 0.1},
                'preprocessing': {
                    'output_format': 'parquet',
                    'feature_engineering': True,
                    'normalization': True,
                    'handle_missing': 'median_fill'
                },
                'quality_checks': {
                    'min_samples': 1000,
                    'max_missing_ratio': 0.1,
                    'feature_count_validation': True
                }
            },
            'feature_engineering': {
                'total_features': 774,
                'components': {}
            }
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe for testing"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'target': np.random.randint(0, 3, n_samples),
            'symbol': ['NIFTY'] * n_samples
        }
        
        # Add feature columns
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        # Add some missing values
        data['feature_0'][::10] = np.nan
        
        return pd.DataFrame(data)
    
    def test_data_preparation_config_initialization(self, sample_config):
        """Test DataPreparationConfig initialization"""
        config = DataPreparationConfig(sample_config)
        
        assert config.project_id == 'test-project'
        assert config.location == 'us-central1'
        assert config.train_ratio == 0.7
        assert config.output_format == 'parquet'
    
    def test_data_quality_checker_validate_dataset(self, sample_config, sample_dataframe):
        """Test data quality validation"""
        config = DataPreparationConfig(sample_config)
        checker = DataQualityChecker(config)
        
        results = checker.validate_dataset(sample_dataframe)
        
        assert 'total_samples' in results
        assert 'total_features' in results
        assert 'missing_data_ratio' in results
        assert results['total_samples'] == len(sample_dataframe)
        assert results['total_features'] == len(sample_dataframe.columns)
    
    def test_data_quality_checker_insufficient_samples(self, sample_config):
        """Test data quality checker with insufficient samples"""
        config = DataPreparationConfig(sample_config)
        checker = DataQualityChecker(config)
        
        # Create small dataframe
        small_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        results = checker.validate_dataset(small_df)
        
        assert not results['errors'] == []  # Should have errors
        assert any('Insufficient samples' in error for error in results['errors'])
    
    def test_time_based_splitter_with_timestamp(self, sample_config, sample_dataframe):
        """Test time-based data splitting"""
        config = DataPreparationConfig(sample_config)
        splitter = TimeBasedSplitter(config)
        
        train_df, val_df, test_df = splitter.split_data(sample_dataframe, 'timestamp')
        
        # Check split ratios
        total_samples = len(sample_dataframe)
        assert abs(len(train_df) / total_samples - 0.7) < 0.05  # Within 5%
        assert abs(len(val_df) / total_samples - 0.2) < 0.05
        assert abs(len(test_df) / total_samples - 0.1) < 0.05
        
        # Check temporal order
        assert train_df['timestamp'].max() <= val_df['timestamp'].min()
        assert val_df['timestamp'].max() <= test_df['timestamp'].min()
    
    def test_time_based_splitter_without_timestamp(self, sample_config):
        """Test fallback to row-based splitting"""
        config = DataPreparationConfig(sample_config)
        splitter = TimeBasedSplitter(config)
        
        # DataFrame without timestamp
        df = pd.DataFrame({'feature_1': range(100), 'target': range(100)})
        
        train_df, val_df, test_df = splitter.split_data(df, 'timestamp')
        
        # Should still split properly
        total_samples = len(df)
        assert abs(len(train_df) / total_samples - 0.7) < 0.05
    
    def test_feature_processor_handle_missing_values(self, sample_config, sample_dataframe):
        """Test missing value handling"""
        config = DataPreparationConfig(sample_config)
        processor = FeatureProcessor(config)
        
        train_df = sample_dataframe.copy()
        val_df = sample_dataframe.copy()
        test_df = sample_dataframe.copy()
        
        processed_train, processed_val, processed_test = processor._handle_missing_values(
            train_df, val_df, test_df
        )
        
        # Check that missing values are handled
        assert processed_train.isnull().sum().sum() == 0
        assert processed_val.isnull().sum().sum() == 0
        assert processed_test.isnull().sum().sum() == 0
    
    def test_feature_processor_normalization(self, sample_config, sample_dataframe):
        """Test feature normalization"""
        config = DataPreparationConfig(sample_config)
        processor = FeatureProcessor(config)
        
        train_df = sample_dataframe.copy()
        val_df = sample_dataframe.copy()
        test_df = sample_dataframe.copy()
        
        processed_train, processed_val, processed_test = processor._normalize_features(
            train_df, val_df, test_df
        )
        
        # Check normalization (should have mean ~0, std ~1 for numeric columns)
        numeric_cols = processed_train.select_dtypes(include=[np.number]).columns
        normalize_cols = [col for col in numeric_cols if col not in ['target', 'timestamp', 'symbol']]
        
        if normalize_cols:
            train_means = processed_train[normalize_cols].mean()
            train_stds = processed_train[normalize_cols].std()
            
            # Means should be close to 0, stds close to 1
            assert abs(train_means.mean()) < 0.1
            assert abs(train_stds.mean() - 1.0) < 0.1


class TestModelEvaluationComponents:
    """Test suite for model evaluation components"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions for testing"""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.randint(0, 3, n_samples)
        y_pred = np.random.randint(0, 3, n_samples)
        y_pred_proba = np.random.rand(n_samples, 3)
        
        # Normalize probabilities
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
        
        return y_true, y_pred, y_pred_proba
    
    def test_classification_metrics_basic(self, sample_predictions):
        """Test basic classification metrics calculation"""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        metrics_calculator = ClassificationMetrics()
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Check required metrics exist
        required_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
        
        # Check confusion matrix
        assert 'confusion_matrix' in metrics
        assert len(metrics['confusion_matrix']) == 3  # 3 classes
    
    def test_classification_metrics_with_probabilities(self, sample_predictions):
        """Test metrics calculation with probabilities"""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        metrics_calculator = ClassificationMetrics()
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Should include probabilistic metrics
        assert 'roc_auc_ovr' in metrics or 'roc_auc' in metrics
        assert 'log_loss' in metrics
    
    def test_forecasting_metrics_regression(self):
        """Test forecasting metrics for regression data"""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.randn(n_samples)
        y_pred = y_true + np.random.randn(n_samples) * 0.1  # Add noise
        
        metrics_calculator = ForecastingMetrics()
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred)
        
        # Check regression metrics
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_forecasting_metrics_directional_accuracy(self):
        """Test directional accuracy calculation"""
        # Create trend data
        y_true = np.array([1, 2, 3, 2, 1, 2, 3, 4])
        y_pred = np.array([1, 2, 4, 2, 1, 2, 2, 4])
        
        metrics_calculator = ForecastingMetrics()
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred)
        
        assert 'directional_accuracy' in metrics
        assert 0 <= metrics['directional_accuracy'] <= 1
    
    def test_market_specific_metrics_regime_accuracy(self):
        """Test regime-specific accuracy calculation"""
        # Create regime data
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 1, 0, 1, 2])
        
        metrics_calculator = MarketSpecificMetrics()
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred)
        
        assert 'regime_wise_accuracy' in metrics
        assert 'regime_stability_score' in metrics
        assert 0 <= metrics['regime_stability_score'] <= 1
    
    def test_model_comparison_framework(self, sample_predictions):
        """Test model comparison functionality"""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        # Create mock results for multiple models
        model_results = {
            'model_a': {
                'accuracy': 0.85,
                'f1_weighted': 0.83,
                'roc_auc_ovr': 0.90
            },
            'model_b': {
                'accuracy': 0.80,
                'f1_weighted': 0.78,
                'roc_auc_ovr': 0.88
            },
            'model_c': {
                'accuracy': 0.87,
                'f1_weighted': 0.85,
                'roc_auc_ovr': 0.89
            }
        }
        
        comparison_framework = ModelComparisonFramework()
        comparison = comparison_framework.compare_models(model_results)
        
        assert 'model_ranking' in comparison
        assert 'metric_comparison' in comparison
        assert 'best_models' in comparison
        
        # Check that best model is identified
        assert comparison['model_ranking']['best_model'] in model_results.keys()
    
    @pytest.mark.skip(reason="Requires matplotlib - skip in CI")
    def test_evaluation_visualizer_confusion_matrix(self, sample_predictions):
        """Test confusion matrix visualization"""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            visualizer = EvaluationVisualizer(tmp_dir)
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            plot_path = visualizer.create_confusion_matrix_plot(
                cm, ['class_0', 'class_1', 'class_2'], 'test_model'
            )
            
            assert plot_path != ""
            assert Path(plot_path).exists()


class TestBaselineModels:
    """Test suite for baseline model components"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for model testing"""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        
        return X, y
    
    @pytest.fixture
    def model_config(self):
        """Sample model configuration"""
        return {
            # TabNet config
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "max_epochs": 5,  # Small for testing
            "batch_size": 32,
            
            # XGBoost config
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 10,  # Small for testing
            
            # LSTM config
            "hidden_size": 16,
            "num_layers": 1,
            "epochs": 2,  # Small for testing
            "sequence_length": 10
        }
    
    def test_model_selection_framework_create_model(self, model_config):
        """Test model creation through framework"""
        framework = ModelSelectionFramework({})
        
        # Test TabNet creation
        tabnet_model = framework.create_model("tabnet", model_config)
        assert isinstance(tabnet_model, TabNetRegimeClassifier)
        
        # Test XGBoost creation
        xgb_model = framework.create_model("xgboost", model_config)
        assert isinstance(xgb_model, XGBoostRegimeClassifier)
    
    def test_model_selection_framework_invalid_type(self, model_config):
        """Test error handling for invalid model type"""
        framework = ModelSelectionFramework({})
        
        with pytest.raises(ValueError):
            framework.create_model("invalid_model", model_config)
    
    def test_model_config_validation(self, model_config):
        """Test model configuration validation"""
        framework = ModelSelectionFramework({})
        
        # Test valid config
        assert framework.validate_config("tabnet", model_config)
        assert framework.validate_config("xgboost", model_config)
    
    @pytest.mark.slow
    def test_xgboost_training(self, sample_training_data, model_config):
        """Test XGBoost model training"""
        X, y = sample_training_data
        
        model = XGBoostRegimeClassifier(model_config)
        
        # Split data for training/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        assert model.is_trained
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['val_accuracy'] <= 1
    
    @pytest.mark.slow  
    def test_xgboost_prediction(self, sample_training_data, model_config):
        """Test XGBoost model prediction"""
        X, y = sample_training_data
        
        model = XGBoostRegimeClassifier(model_config)
        
        # Train model
        model.train(X, y)
        
        # Make predictions
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 3)  # 3 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_model_feature_importance(self, sample_training_data, model_config):
        """Test feature importance extraction"""
        X, y = sample_training_data
        
        model = XGBoostRegimeClassifier(model_config)
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]  # One importance per feature
    
    def test_model_save_load(self, sample_training_data, model_config):
        """Test model saving and loading"""
        X, y = sample_training_data
        
        model = XGBoostRegimeClassifier(model_config)
        model.train(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            # Save model
            saved_path = model.save_model(tmp_file.name)
            assert saved_path == tmp_file.name
            
            # Create new model instance and load
            new_model = XGBoostRegimeClassifier({})
            new_model.load_model(tmp_file.name)
            
            assert new_model.is_trained
            
            # Test predictions are the same
            original_pred = model.predict(X[:5])
            loaded_pred = new_model.predict(X[:5])
            
            np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Cleanup
        os.unlink(tmp_file.name)


class TestPipelineIntegration:
    """Integration tests for pipeline components"""
    
    @pytest.fixture
    def pipeline_config(self):
        """Complete pipeline configuration for testing"""
        return {
            'project': {
                'project_id': 'test-project',
                'location': 'us-central1',
                'staging_bucket': 'test-bucket'
            },
            'data': {
                'source': {'full_table_id': 'test-project.test_dataset.test_table'},
                'splits': {'train_ratio': 0.7, 'validation_ratio': 0.2, 'test_ratio': 0.1},
                'preprocessing': {
                    'output_format': 'parquet',
                    'feature_engineering': True,
                    'normalization': True,
                    'handle_missing': 'median_fill'
                },
                'quality_checks': {
                    'min_samples': 100,  # Lower for testing
                    'max_missing_ratio': 0.3,
                    'feature_count_validation': False
                }
            },
            'feature_engineering': {
                'total_features': 20,  # Lower for testing
                'components': {}
            },
            'models': {
                'baseline_models': [
                    {'name': 'xgboost', 'enabled': True}
                ]
            },
            'training': {
                'experiment_name': 'test-experiment'
            }
        }
    
    @patch('pipelines.data_preparation.BigQueryDataLoader.load_training_data')
    def test_data_preparation_pipeline_integration(self, mock_load_data, pipeline_config):
        """Test integration of data preparation pipeline"""
        # Mock BigQuery data loading
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='1H'),
            'target': np.random.randint(0, 3, 200),
            **{f'feature_{i}': np.random.randn(200) for i in range(10)}
        })
        mock_load_data.return_value = sample_data
        
        from pipelines.data_preparation import DataPreparationPipeline, DataPreparationConfig
        
        config = DataPreparationConfig(pipeline_config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = DataPreparationPipeline(config)
            
            # Mock the data loader to avoid BigQuery calls
            pipeline.data_loader.load_training_data = Mock(return_value=sample_data)
            pipeline.exporter._export_parquet = Mock(return_value={
                'train_data': f'{tmp_dir}/train.parquet',
                'validation_data': f'{tmp_dir}/val.parquet', 
                'test_data': f'{tmp_dir}/test.parquet'
            })
            
            results = pipeline.run_pipeline(tmp_dir)
            
            assert results['status'] == 'success'
            assert 'data_splits' in results
            assert 'output_paths' in results
            assert results['data_splits']['train_samples'] > 0
    
    def test_model_evaluation_integration(self):
        """Test integration of model evaluation components"""
        from pipelines.model_evaluation import ModelEvaluationPipeline
        
        # Create mock data
        np.random.seed(42)
        n_samples = 100
        
        model_predictions = {
            'predictions': np.random.randint(0, 3, n_samples),
            'probabilities': np.random.rand(n_samples, 3)
        }
        
        ground_truth = {
            'labels': np.random.randint(0, 3, n_samples),
            'timestamps': pd.date_range('2023-01-01', periods=n_samples, freq='1H').values
        }
        
        # Normalize probabilities
        model_predictions['probabilities'] = (
            model_predictions['probabilities'] / 
            model_predictions['probabilities'].sum(axis=1, keepdims=True)
        )
        
        config = {'output_dir': tempfile.mkdtemp(), 'validation': {}}
        
        evaluation_pipeline = ModelEvaluationPipeline(config)
        
        results = evaluation_pipeline.evaluate_model(
            model_predictions=model_predictions,
            ground_truth=ground_truth,
            model_name='test_model'
        )
        
        assert results['status'] == 'success'
        assert 'classification_metrics' in results
        assert 'forecasting_metrics' in results
        assert 'market_metrics' in results
        assert 'validation_results' in results


class TestPerformanceValidation:
    """Performance and benchmark tests"""
    
    def test_data_processing_performance(self):
        """Test data processing performance benchmarks"""
        # Create large dataset for performance testing
        n_samples = 10000
        n_features = 100
        
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'target': np.random.randint(0, 3, n_samples)
        }
        
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        df = pd.DataFrame(data)
        
        # Test data preparation performance
        start_time = datetime.now()
        
        # Mock config
        config = {
            'project': {'project_id': 'test', 'location': 'us-central1', 'staging_bucket': 'test'},
            'data': {
                'splits': {'train_ratio': 0.7, 'validation_ratio': 0.2, 'test_ratio': 0.1},
                'preprocessing': {'handle_missing': 'median_fill', 'normalization': True}
            },
            'feature_engineering': {'total_features': 100, 'components': {}}
        }
        
        from pipelines.data_preparation import DataPreparationConfig, TimeBasedSplitter
        
        prep_config = DataPreparationConfig(config)
        splitter = TimeBasedSplitter(prep_config)
        
        train_df, val_df, test_df = splitter.split_data(df, 'timestamp')
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertion - should process 10k samples in reasonable time
        assert processing_time < 10  # Less than 10 seconds
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
    
    def test_model_training_performance(self):
        """Test model training performance"""
        np.random.seed(42)
        
        # Medium-sized dataset
        n_samples = 1000
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        
        # Test XGBoost training time
        config = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100
        }
        
        start_time = datetime.now()
        
        model = XGBoostRegimeClassifier(config)
        metrics = model.train(X, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert training_time < 30  # Less than 30 seconds
        assert metrics['train_accuracy'] > 0.5  # Reasonable accuracy
        assert model.is_trained
    
    def test_memory_usage_validation(self):
        """Test memory usage stays within bounds"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large dataset
        n_samples = 50000
        n_features = 100
        
        data = np.random.randn(n_samples, n_features)
        df = pd.DataFrame(data)
        
        # Process data
        df_processed = df.fillna(df.median())
        df_normalized = (df_processed - df_processed.mean()) / df_processed.std()
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500
        
        # Clean up
        del data, df, df_processed, df_normalized


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    # Cleanup is automatic with tempfile


# Slow test marker
pytest.mark.slow = pytest.mark.filterwarnings("ignore")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])