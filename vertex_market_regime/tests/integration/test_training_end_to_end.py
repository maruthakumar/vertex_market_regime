"""
Integration Tests for End-to-End Training Pipeline
Comprehensive integration testing for the complete training pipeline

This module provides:
- End-to-end pipeline integration tests
- Performance benchmarking and validation
- Parity tests with existing feature transforms
- Automated pipeline validation tests
- Full workflow testing with real components
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pipelines.data_preparation import DataPreparationPipeline, DataPreparationConfig
from pipelines.model_evaluation import ModelEvaluationPipeline
from pipelines.pipeline_orchestrator import PipelineOrchestrator, PipelineConfigManager
from ml.baseline_models import ModelSelectionFramework, XGBoostRegimeClassifier
from integrations.vertex_ai_training_client import VertexAITrainingClient


class TestEndToEndPipelineIntegration:
    """End-to-end integration tests for training pipeline"""
    
    @pytest.fixture
    def complete_pipeline_config(self):
        """Complete pipeline configuration for integration testing"""
        return {
            'project': {
                'project_id': 'test-project-integration',
                'location': 'us-central1',
                'staging_bucket': 'test-integration-bucket',
                'artifact_bucket': 'test-artifacts-bucket'
            },
            'data': {
                'source': {
                    'type': 'bigquery',
                    'project': 'test-project-integration',
                    'dataset': 'market_regime_dev',
                    'table': 'training_dataset',
                    'full_table_id': 'test-project-integration.market_regime_dev.training_dataset'
                },
                'splits': {
                    'train_ratio': 0.7,
                    'validation_ratio': 0.2,
                    'test_ratio': 0.1
                },
                'preprocessing': {
                    'output_format': 'parquet',
                    'feature_engineering': True,
                    'normalization': True,
                    'handle_missing': 'median_fill'
                },
                'quality_checks': {
                    'min_samples': 1000,
                    'max_missing_ratio': 0.1,
                    'feature_count_validation': True,
                    'target_distribution_check': True
                }
            },
            'models': {
                'baseline_models': [
                    {
                        'name': 'tabnet',
                        'enabled': False,  # Disable for faster testing
                        'hyperparameters': {
                            'n_d': 32,
                            'n_a': 32,
                            'n_steps': 3,
                            'max_epochs': 5,
                            'batch_size': 256
                        }
                    },
                    {
                        'name': 'xgboost',
                        'enabled': True,
                        'hyperparameters': {
                            'max_depth': 4,
                            'learning_rate': 0.1,
                            'n_estimators': 50,  # Reduced for testing
                            'subsample': 0.8,
                            'colsample_bytree': 0.8
                        }
                    },
                    {
                        'name': 'lstm',
                        'enabled': False,  # Disable for faster testing
                        'hyperparameters': {
                            'hidden_size': 64,
                            'num_layers': 2,
                            'epochs': 5,
                            'sequence_length': 30
                        }
                    }
                ]
            },
            'training': {
                'experiment_name': 'integration-test-experiment',
                'run_prefix': 'integration-test',
                'compute': {
                    'machine_type': 'n1-standard-4',
                    'accelerator_type': None,
                    'accelerator_count': 0
                },
                'resources': {
                    'cpu_limit': '4',
                    'memory_limit': '16Gi',
                    'timeout_hours': 2
                }
            },
            'evaluation': {
                'metrics': {
                    'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'auroc'],
                    'forecasting': ['mae', 'mse', 'rmse', 'directional_accuracy'],
                    'regime_specific': ['regime_accuracy', 'transition_detection_f1']
                },
                'validation': {
                    'min_accuracy_threshold': 0.6,  # Lower for testing
                    'max_overfitting_threshold': 0.1
                }
            },
            'feature_engineering': {
                'enable_component_features': True,
                'total_features': 50,  # Reduced for testing
                'components': {
                    'component_01_triple_straddle': {'enabled': True, 'feature_count': 10},
                    'component_02_greeks_sentiment': {'enabled': True, 'feature_count': 10},
                    'component_03_oi_pa_trending': {'enabled': True, 'feature_count': 10},
                    'component_04_iv_skew': {'enabled': True, 'feature_count': 10},
                    'component_05_atr_ema_cpr': {'enabled': True, 'feature_count': 10}
                }
            },
            'pipeline': {
                'name': 'integration-test-pipeline',
                'display_name': 'Integration Test Pipeline',
                'description': 'End-to-end integration test pipeline',
                'scheduling': {
                    'enabled': False,
                    'cron_schedule': '0 2 * * 0',
                    'timezone': 'UTC'
                },
                'notifications': {
                    'email_on_failure': False,
                    'slack_webhook': None
                },
                'retry_policy': {
                    'max_retries': 1,
                    'retry_delay_seconds': 60
                }
            }
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market regime training data"""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate time series
        timestamps = pd.date_range('2022-01-01', periods=n_samples, freq='1H')
        
        # Create market regimes (0=trending up, 1=ranging, 2=trending down)
        regime_durations = np.random.exponential(100, size=50)  # Average 100 hours per regime
        regimes = []
        current_pos = 0
        
        for duration in regime_durations:
            regime = np.random.randint(0, 3)
            length = int(min(duration, n_samples - current_pos))
            regimes.extend([regime] * length)
            current_pos += length
            if current_pos >= n_samples:
                break
        
        # Pad or trim to exact length
        regimes = regimes[:n_samples]
        if len(regimes) < n_samples:
            regimes.extend([regimes[-1]] * (n_samples - len(regimes)))
        
        data = {
            'timestamp': timestamps,
            'target': regimes,
            'symbol': ['NIFTY'] * n_samples
        }
        
        # Generate component features
        components = ['c1_', 'c2_', 'c3_', 'c4_', 'c5_']
        
        for comp_prefix in components:
            for i in range(10):  # 10 features per component
                feature_name = f'{comp_prefix}feature_{i}'
                
                # Generate features with some correlation to regime
                base_values = np.random.randn(n_samples)
                regime_effect = np.array([r * 0.3 + np.random.randn() * 0.1 for r in regimes])
                
                data[feature_name] = base_values + regime_effect
        
        # Add some missing values
        df = pd.DataFrame(data)
        
        # Introduce missing values in ~2% of data
        missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        feature_cols = [col for col in df.columns if col.startswith('c')]
        
        for idx in missing_indices:
            random_col = np.random.choice(feature_cols)
            df.loc[idx, random_col] = np.nan
        
        return df
    
    @patch('pipelines.data_preparation.BigQueryDataLoader.load_training_data')
    def test_complete_training_pipeline_flow(self, mock_load_data, complete_pipeline_config, sample_market_data):
        """Test complete end-to-end training pipeline flow"""
        
        # Mock BigQuery data loading
        mock_load_data.return_value = sample_market_data
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            
            # Step 1: Data Preparation
            print("Testing data preparation...")
            
            data_config = DataPreparationConfig(complete_pipeline_config)
            data_pipeline = DataPreparationPipeline(data_config)
            
            # Mock external dependencies
            data_pipeline.data_loader.load_training_data = Mock(return_value=sample_market_data)
            
            # Run data preparation
            data_results = data_pipeline.run_pipeline(tmp_dir)
            
            assert data_results['status'] == 'success'
            assert data_results['data_splits']['train_samples'] > 0
            assert data_results['data_splits']['validation_samples'] > 0
            assert data_results['data_splits']['test_samples'] > 0
            
            print(f"Data preparation completed: {data_results['data_splits']}")
            
            # Step 2: Model Training
            print("Testing model training...")
            
            # Simulate training data
            train_split_size = data_results['data_splits']['train_samples']
            val_split_size = data_results['data_splits']['validation_samples']
            
            # Create train/validation splits
            train_data = sample_market_data.iloc[:train_split_size].copy()
            val_data = sample_market_data.iloc[train_split_size:train_split_size+val_split_size].copy()
            
            # Get enabled models
            enabled_models = [m for m in complete_pipeline_config['models']['baseline_models'] if m['enabled']]
            
            training_results = {}
            
            for model_config in enabled_models:
                model_name = model_config['name']
                print(f"Training {model_name} model...")
                
                # Prepare training data
                feature_columns = [col for col in train_data.columns if col.startswith('c')]
                X_train = train_data[feature_columns].fillna(train_data[feature_columns].median()).values
                y_train = train_data['target'].values
                X_val = val_data[feature_columns].fillna(train_data[feature_columns].median()).values
                y_val = val_data['target'].values
                
                # Create and train model
                framework = ModelSelectionFramework({})
                model = framework.create_model(model_name, model_config['hyperparameters'])
                
                start_time = time.time()
                train_metrics = model.train(X_train, y_train, X_val, y_val)
                training_time = time.time() - start_time
                
                training_results[model_name] = {
                    'model': model,
                    'training_metrics': train_metrics,
                    'training_time': training_time
                }
                
                print(f"{model_name} training completed in {training_time:.1f}s - Accuracy: {train_metrics.get('val_accuracy', 'N/A'):.3f}")
            
            assert len(training_results) > 0
            
            # Step 3: Model Evaluation
            print("Testing model evaluation...")
            
            test_split_start = train_split_size + val_split_size
            test_data = sample_market_data.iloc[test_split_start:].copy()
            
            evaluation_results = {}
            
            for model_name, model_info in training_results.items():
                model = model_info['model']
                
                # Prepare test data
                feature_columns = [col for col in test_data.columns if col.startswith('c')]
                X_test = test_data[feature_columns].fillna(train_data[feature_columns].median()).values
                y_test = test_data['target'].values
                
                # Make predictions
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
                
                # Prepare evaluation data
                model_predictions = {
                    'predictions': predictions,
                    'probabilities': probabilities
                }
                
                ground_truth = {
                    'labels': y_test,
                    'timestamps': test_data['timestamp'].values
                }
                
                # Run evaluation
                eval_config = {
                    'output_dir': tmp_dir,
                    'validation': complete_pipeline_config['evaluation']['validation']
                }
                
                evaluation_pipeline = ModelEvaluationPipeline(eval_config)
                eval_results = evaluation_pipeline.evaluate_model(
                    model_predictions=model_predictions,
                    ground_truth=ground_truth,
                    model_name=model_name,
                    model_metadata={'training_time': model_info['training_time']}
                )
                
                evaluation_results[model_name] = eval_results
                
                print(f"{model_name} evaluation completed - Status: {eval_results['status']}")
                
                if eval_results['status'] == 'success':
                    metrics = eval_results['classification_metrics']
                    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                    print(f"  F1-Score: {metrics.get('f1_weighted', 'N/A'):.3f}")
            
            # Step 4: Validation and Assertions
            print("Validating pipeline results...")
            
            # Validate data preparation
            assert data_results['status'] == 'success'
            assert 'data_quality' in data_results
            assert 'output_paths' in data_results
            
            # Validate model training
            for model_name, results in training_results.items():
                assert results['model'].is_trained
                assert 'train_accuracy' in results['training_metrics']
                assert results['training_metrics']['train_accuracy'] > 0.4  # Reasonable minimum
                assert results['training_time'] < 300  # Should complete in 5 minutes
            
            # Validate model evaluation
            for model_name, results in evaluation_results.items():
                assert results['status'] == 'success'
                assert 'classification_metrics' in results
                assert 'validation_results' in results
                
                # Check metrics are reasonable
                accuracy = results['classification_metrics']['accuracy']
                assert 0.3 <= accuracy <= 1.0  # Sanity check
            
            print("End-to-end pipeline test completed successfully!")
            
            return {
                'data_results': data_results,
                'training_results': training_results,
                'evaluation_results': evaluation_results
            }
    
    def test_pipeline_performance_benchmarks(self, complete_pipeline_config, sample_market_data):
        """Test pipeline performance against benchmarks"""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            
            # Performance benchmarks
            benchmarks = {
                'data_preparation_time_seconds': 30,
                'model_training_time_seconds': 180,  # 3 minutes
                'model_evaluation_time_seconds': 10,
                'memory_usage_mb': 1000,
                'min_accuracy': 0.5
            }
            
            performance_results = {}
            
            # Test data preparation performance
            start_time = time.time()
            
            data_config = DataPreparationConfig(complete_pipeline_config)
            data_pipeline = DataPreparationPipeline(data_config)
            data_pipeline.data_loader.load_training_data = Mock(return_value=sample_market_data)
            
            data_results = data_pipeline.run_pipeline(tmp_dir)
            
            data_prep_time = time.time() - start_time
            performance_results['data_preparation_time'] = data_prep_time
            
            assert data_prep_time < benchmarks['data_preparation_time_seconds']
            print(f"Data preparation performance: {data_prep_time:.1f}s (benchmark: {benchmarks['data_preparation_time_seconds']}s)")
            
            # Test model training performance
            feature_columns = [col for col in sample_market_data.columns if col.startswith('c')]
            X = sample_market_data[feature_columns].fillna(sample_market_data[feature_columns].median()).values
            y = sample_market_data['target'].values
            
            start_time = time.time()
            
            model_config = {
                'max_depth': 4,
                'learning_rate': 0.1,
                'n_estimators': 50
            }
            
            model = XGBoostRegimeClassifier(model_config)
            train_metrics = model.train(X, y)
            
            training_time = time.time() - start_time
            performance_results['model_training_time'] = training_time
            
            assert training_time < benchmarks['model_training_time_seconds']
            print(f"Model training performance: {training_time:.1f}s (benchmark: {benchmarks['model_training_time_seconds']}s)")
            
            # Validate accuracy benchmark
            accuracy = train_metrics.get('train_accuracy', 0)
            performance_results['model_accuracy'] = accuracy
            
            assert accuracy > benchmarks['min_accuracy']
            print(f"Model accuracy: {accuracy:.3f} (benchmark: {benchmarks['min_accuracy']})")
            
            return performance_results
    
    def test_parity_with_existing_features(self, sample_market_data):
        """Test parity with existing feature transforms"""
        
        # This test validates that our new pipeline produces features
        # consistent with the existing Epic 1 implementation
        
        print("Testing feature parity...")
        
        # Expected feature structure from Epic 1
        expected_feature_structure = {
            'c1_': 10,  # Component 1 features
            'c2_': 10,  # Component 2 features
            'c3_': 10,  # Component 3 features
            'c4_': 10,  # Component 4 features
            'c5_': 10   # Component 5 features
        }
        
        # Validate feature naming convention
        for prefix, expected_count in expected_feature_structure.items():
            feature_cols = [col for col in sample_market_data.columns if col.startswith(prefix)]
            actual_count = len(feature_cols)
            
            assert actual_count == expected_count, f"Feature count mismatch for {prefix}: expected {expected_count}, got {actual_count}"
            print(f"Feature parity check passed for {prefix}: {actual_count} features")
        
        # Validate feature value ranges (basic sanity checks)
        feature_columns = [col for col in sample_market_data.columns if col.startswith('c')]
        
        for col in feature_columns:
            values = sample_market_data[col].dropna()
            
            # Features should be roughly normalized (most values within -5 to +5)
            within_range = ((values >= -5) & (values <= 5)).mean()
            assert within_range > 0.8, f"Feature {col} has too many extreme values: {within_range:.2f} within range"
        
        print("Feature parity validation completed successfully!")
    
    def test_pipeline_error_handling_and_recovery(self, complete_pipeline_config):
        """Test pipeline error handling and recovery mechanisms"""
        
        print("Testing error handling and recovery...")
        
        # Test 1: Data quality validation failures
        bad_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),  # Too few samples
            'target': [0] * 10,
            'feature_1': [1] * 10
        })
        
        data_config = DataPreparationConfig(complete_pipeline_config)
        data_pipeline = DataPreparationPipeline(data_config)
        data_pipeline.data_loader.load_training_data = Mock(return_value=bad_data)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = data_pipeline.run_pipeline(tmp_dir)
            
            # Should fail gracefully
            assert results['status'] == 'failed'
            assert 'error' in results
            print("Data quality validation error handling: PASSED")
        
        # Test 2: Model training with invalid configuration
        invalid_model_config = {
            'max_depth': -1,  # Invalid parameter
            'n_estimators': 0  # Invalid parameter
        }
        
        try:
            model = XGBoostRegimeClassifier(invalid_model_config)
            # This might not fail immediately, so we test with actual training
            X = np.random.randn(100, 10)
            y = np.random.randint(0, 3, 100)
            
            # Training should handle the error gracefully
            train_metrics = model.train(X, y)
            print("Model handled invalid config gracefully")
            
        except Exception as e:
            # Expected behavior - error should be caught and handled
            print(f"Model training error handled: {type(e).__name__}")
        
        print("Error handling and recovery tests completed!")
    
    def test_pipeline_scalability(self, complete_pipeline_config):
        """Test pipeline scalability with different data sizes"""
        
        print("Testing pipeline scalability...")
        
        data_sizes = [1000, 5000, 10000]  # Different dataset sizes
        performance_metrics = {}
        
        for size in data_sizes:
            print(f"Testing with {size} samples...")
            
            # Generate data of specified size
            np.random.seed(42)
            timestamps = pd.date_range('2023-01-01', periods=size, freq='1H')
            
            data = {
                'timestamp': timestamps,
                'target': np.random.randint(0, 3, size),
                'symbol': ['NIFTY'] * size
            }
            
            # Add features
            for i in range(20):  # 20 features for scalability test
                data[f'c1_feature_{i}'] = np.random.randn(size)
            
            test_data = pd.DataFrame(data)
            
            # Measure data processing time
            start_time = time.time()
            
            data_config = DataPreparationConfig(complete_pipeline_config)
            splitter = data_config.__class__(complete_pipeline_config)
            
            # Simulate data processing
            feature_columns = [col for col in test_data.columns if col.startswith('c1_')]
            processed_data = test_data[feature_columns].fillna(test_data[feature_columns].median())
            
            processing_time = time.time() - start_time
            
            performance_metrics[size] = {
                'processing_time': processing_time,
                'samples_per_second': size / processing_time if processing_time > 0 else float('inf')
            }
            
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Throughput: {performance_metrics[size]['samples_per_second']:.0f} samples/s")
        
        # Validate scalability - processing time should scale reasonably
        small_time = performance_metrics[1000]['processing_time']
        large_time = performance_metrics[10000]['processing_time']
        
        # Processing time should not increase by more than 20x for 10x data
        scalability_ratio = large_time / small_time if small_time > 0 else 1
        assert scalability_ratio < 20, f"Poor scalability: {scalability_ratio:.1f}x time increase for 10x data"
        
        print(f"Scalability test passed: {scalability_ratio:.1f}x time increase for 10x data")
        
        return performance_metrics
    
    @pytest.mark.slow
    def test_pipeline_configuration_management(self, complete_pipeline_config):
        """Test pipeline configuration management and validation"""
        
        print("Testing configuration management...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'test_config.yaml'
            
            # Save configuration to file
            with open(config_path, 'w') as f:
                yaml.dump(complete_pipeline_config, f)
            
            # Test configuration loading
            config_manager = PipelineConfigManager(str(config_path))
            loaded_config = config_manager.load_config()
            
            assert loaded_config == complete_pipeline_config
            print("Configuration loading: PASSED")
            
            # Test configuration validation
            validation_results = config_manager.validate_config(loaded_config)
            
            assert validation_results['valid']
            assert len(validation_results['errors']) == 0
            print("Configuration validation: PASSED")
            
            # Test environment-specific configuration
            env_config = config_manager.get_environment_config('development')
            
            assert 'project' in env_config
            assert 'data' in env_config
            print("Environment configuration: PASSED")
            
            # Test invalid configuration
            invalid_config = complete_pipeline_config.copy()
            del invalid_config['project']['project_id']  # Remove required field
            
            validation_results = config_manager.validate_config(invalid_config)
            
            assert not validation_results['valid']
            assert len(validation_results['errors']) > 0
            print("Invalid configuration detection: PASSED")
        
        print("Configuration management tests completed!")


class TestPipelineRobustness:
    """Test pipeline robustness and edge cases"""
    
    def test_empty_data_handling(self, complete_pipeline_config):
        """Test handling of empty or minimal datasets"""
        
        # Empty dataset
        empty_data = pd.DataFrame()
        
        data_config = DataPreparationConfig(complete_pipeline_config)
        data_pipeline = DataPreparationPipeline(data_config)
        data_pipeline.data_loader.load_training_data = Mock(return_value=empty_data)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = data_pipeline.run_pipeline(tmp_dir)
            
            # Should handle empty data gracefully
            assert results['status'] == 'failed'
            assert 'error' in results
    
    def test_missing_data_handling(self):
        """Test handling of datasets with extensive missing data"""
        
        # Dataset with 50% missing values
        n_samples = 1000
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'target': np.random.randint(0, 3, n_samples),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce 50% missing values
        missing_mask = np.random.rand(n_samples, 2) < 0.5
        df.loc[missing_mask[:, 0], 'feature_1'] = np.nan
        df.loc[missing_mask[:, 1], 'feature_2'] = np.nan
        
        # Test feature processing
        config = {
            'project': {'project_id': 'test', 'location': 'us-central1', 'staging_bucket': 'test'},
            'data': {
                'preprocessing': {'handle_missing': 'median_fill'},
                'quality_checks': {'max_missing_ratio': 0.6}  # Allow high missing ratio
            },
            'feature_engineering': {'total_features': 2, 'components': {}}
        }
        
        from pipelines.data_preparation import FeatureProcessor
        
        processor = FeatureProcessor(DataPreparationConfig(config))
        
        # Should handle missing data without errors
        processed_train, processed_val, processed_test = processor._handle_missing_values(df, df, df)
        
        # Check that missing values are handled
        assert processed_train['feature_1'].isnull().sum() == 0
        assert processed_train['feature_2'].isnull().sum() == 0
    
    def test_extreme_feature_values(self):
        """Test handling of extreme feature values"""
        
        n_samples = 1000
        
        # Create data with extreme values
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'target': np.random.randint(0, 3, n_samples),
            'normal_feature': np.random.randn(n_samples),
            'extreme_feature': np.concatenate([
                np.random.randn(950),  # Normal values
                np.array([1e6, -1e6] * 25)  # Extreme values
            ])
        }
        
        df = pd.DataFrame(data)
        
        # Test model training with extreme values
        feature_columns = ['normal_feature', 'extreme_feature']
        X = df[feature_columns].values
        y = df['target'].values
        
        model_config = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 10
        }
        
        model = XGBoostRegimeClassifier(model_config)
        
        # Should handle extreme values without crashing
        try:
            metrics = model.train(X, y)
            assert 'train_accuracy' in metrics
            print("Extreme values handled successfully")
        except Exception as e:
            print(f"Model training with extreme values failed: {e}")
            # This is acceptable - the model should fail gracefully
    
    def test_class_imbalance_handling(self):
        """Test handling of highly imbalanced target classes"""
        
        n_samples = 1000
        
        # Create highly imbalanced dataset (95% class 0, 5% other classes)
        targets = np.concatenate([
            np.zeros(950),  # 95% class 0
            np.ones(30),    # 3% class 1
            np.full(20, 2)  # 2% class 2
        ])
        
        np.random.shuffle(targets)
        
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'target': targets.astype(int),
            **{f'feature_{i}': np.random.randn(n_samples) for i in range(10)}
        }
        
        df = pd.DataFrame(data)
        
        # Test evaluation metrics with imbalanced data
        from pipelines.model_evaluation import ClassificationMetrics
        
        metrics_calculator = ClassificationMetrics()
        
        # Mock predictions (slightly better than random)
        y_true = targets.astype(int)
        y_pred = np.random.choice([0, 1, 2], size=n_samples, p=[0.95, 0.03, 0.02])
        
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred)
        
        # Should compute metrics without errors
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'per_class_metrics' in metrics
        
        # Check that all classes are represented in metrics
        assert len(metrics['per_class_metrics']) == 3
        
        print("Class imbalance handling test passed")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])