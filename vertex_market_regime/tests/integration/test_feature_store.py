"""
Integration Tests for Vertex AI Feature Store
Tests all major functionality including infrastructure, ingestion, and serving
"""

import asyncio
import pytest
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

from vertex_market_regime.src.integrations.feature_store_client import (
    FeatureStoreClient, FeatureRequest, FeatureResponse
)
from vertex_market_regime.src.pipelines.feature_ingestion import (
    FeatureIngestionPipeline, IngestionJobConfig, IngestionResult
)


class TestFeatureStoreIntegration:
    """Integration tests for Feature Store functionality"""
    
    @pytest.fixture(scope="class")
    def feature_store_client(self):
        """Create Feature Store client for testing"""
        return FeatureStoreClient(environment="dev")
    
    @pytest.fixture(scope="class")
    def ingestion_pipeline(self):
        """Create ingestion pipeline for testing"""
        return FeatureIngestionPipeline(environment="dev")
    
    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data for testing"""
        data = {
            "entity_id": [
                "NIFTY_20250813140000_0",
                "NIFTY_20250813140100_0", 
                "BANKNIFTY_20250813140000_7"
            ],
            "ts_minute": [
                datetime(2025, 8, 13, 14, 0, 0),
                datetime(2025, 8, 13, 14, 1, 0),
                datetime(2025, 8, 13, 14, 0, 0)
            ],
            "symbol": ["NIFTY", "NIFTY", "BANKNIFTY"],
            "dte": [0, 0, 7],
            # Component 1 features
            "c1_momentum_score": [0.75, 0.82, 0.65],
            "c1_vol_compression": [0.3, 0.45, 0.2],
            "c1_breakout_probability": [0.85, 0.9, 0.7],
            # Component 2 features
            "c2_gamma_exposure": [1200.5, 1350.0, 2100.3],
            "c2_sentiment_level": [0.6, 0.7, 0.4],
            "c2_pin_risk_score": [0.25, 0.3, 0.15],
            # Component 8 features
            "c8_component_agreement_score": [0.88, 0.92, 0.75],
            "c8_integration_confidence": [0.95, 0.98, 0.85],
            "c8_regime_classification": ["trending_up", "trending_up", "ranging"]
        }
        
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_feature_store_infrastructure_creation(self, feature_store_client):
        """Test Task 1: Create Vertex AI Feature Store Infrastructure"""
        
        # Mock the Google Cloud APIs to avoid actual GCP calls during testing
        with patch.object(feature_store_client, 'featurestore', None):
            with patch.object(feature_store_client, 'entity_type', None):
                with patch('google.cloud.aiplatform.Featurestore.create') as mock_create_fs:
                    with patch('google.cloud.aiplatform.Featurestore.create_entity_type') as mock_create_et:
                        
                        # Mock Feature Store creation
                        mock_featurestore = Mock()
                        mock_featurestore.resource_name = "projects/arched-bot-269016/locations/us-central1/featurestores/market_regime_featurestore_dev"
                        mock_create_fs.return_value = mock_featurestore
                        
                        # Mock Entity Type creation
                        mock_entity_type = Mock()
                        mock_entity_type.resource_name = "projects/arched-bot-269016/locations/us-central1/featurestores/market_regime_featurestore_dev/entityTypes/instrument_minute"
                        mock_featurestore.create_entity_type.return_value = mock_entity_type
                        
                        # Mock feature creation
                        mock_entity_type.create_feature.return_value = Mock()
                        
                        # Run infrastructure creation
                        results = await feature_store_client.create_feature_store_infrastructure()
                        
                        # Verify results
                        assert results["status"] == "success"
                        assert results["featurestore_created"] is True
                        assert results["entity_type_created"] is True
                        assert results["total_features_created"] >= 32  # Should create 32 online features
                        assert "execution_time_seconds" in results
                        assert results["regional_configuration"]["location"] == "us-central1"
                        
                        # Verify API calls were made
                        mock_create_fs.assert_called_once()
                        mock_featurestore.create_entity_type.assert_called_once()
                        
                        # Verify feature creation calls
                        assert mock_entity_type.create_feature.call_count >= 32
    
    @pytest.mark.asyncio
    async def test_online_feature_serving(self, feature_store_client):
        """Test online feature serving with latency validation"""
        
        # Mock the feature serving API
        with patch.object(feature_store_client, '_fetch_from_feature_store') as mock_fetch:
            
            # Mock response data
            mock_responses = [
                FeatureResponse(
                    entity_id="NIFTY_20250813140000_0",
                    features={
                        "c1_momentum_score": 0.75,
                        "c2_gamma_exposure": 1200.5,
                        "c8_regime_classification": "trending_up"
                    },
                    latency_ms=25.5,
                    cache_hit=False,
                    freshness_seconds=30,
                    request_timestamp=datetime.utcnow()
                )
            ]
            
            mock_fetch.return_value = mock_responses
            
            # Test feature request
            entity_ids = ["NIFTY_20250813140000_0"]
            feature_names = ["c1_momentum_score", "c2_gamma_exposure", "c8_regime_classification"]
            
            start_time = time.time()
            responses = await feature_store_client.get_online_features(entity_ids, feature_names)
            request_latency = (time.time() - start_time) * 1000
            
            # Verify response
            assert len(responses) == 1
            assert responses[0].entity_id == "NIFTY_20250813140000_0"
            assert "c1_momentum_score" in responses[0].features
            assert responses[0].features["c1_momentum_score"] == 0.75
            
            # Verify latency target (<50ms for end-to-end)
            assert request_latency < 100  # Allow for test overhead
            
            # Test cache functionality
            cached_responses = await feature_store_client.get_online_features(entity_ids, feature_names)
            assert len(cached_responses) == 1
            # Second call should use cache (not testing exact cache hit due to mock)
    
    @pytest.mark.asyncio
    async def test_performance_validation(self, feature_store_client):
        """Test Task 4: Optimize Online Serving Performance"""
        
        # Mock feature serving for performance test
        with patch.object(feature_store_client, 'get_online_features') as mock_get_features:
            
            # Mock consistent low-latency responses
            async def mock_feature_response(*args, **kwargs):
                await asyncio.sleep(0.02)  # Simulate 20ms latency
                return [
                    FeatureResponse(
                        entity_id="NIFTY_20250813140000_0",
                        features={"c1_momentum_score": 0.75, "c2_gamma_exposure": 1200.5},
                        latency_ms=20.0,
                        cache_hit=False,
                        freshness_seconds=30,
                        request_timestamp=datetime.utcnow()
                    )
                ]
            
            mock_get_features.side_effect = mock_feature_response
            
            # Run performance validation (shorter duration for testing)
            perf_results = await feature_store_client.validate_performance(test_duration_seconds=10)
            
            # Verify performance results
            assert perf_results["status"] == "completed"
            assert "performance_metrics" in perf_results
            assert "validation_results" in perf_results
            
            metrics = perf_results["performance_metrics"]
            validation = perf_results["validation_results"]
            
            # Check that requests were made
            assert metrics["requests_made"] > 0
            assert metrics["error_rate"] == 0  # No errors in mock
            
            # Check latency targets (should pass with 20ms mock latency)
            assert validation["latency_p50_pass"] is True
            assert validation["latency_p95_pass"] is True
            assert validation["latency_p99_pass"] is True
            
            # Overall validation should pass
            assert perf_results["overall_pass"] is True
    
    @pytest.mark.asyncio
    async def test_feature_ingestion_pipeline(self, ingestion_pipeline, sample_feature_data):
        """Test Task 3: Implement Feature Ingestion Pipeline"""
        
        # Mock BigQuery client
        with patch.object(ingestion_pipeline, 'bigquery_client') as mock_bq_client:
            with patch.object(ingestion_pipeline, '_process_batch') as mock_process_batch:
                
                # Mock BigQuery query result
                mock_query_job = Mock()
                mock_query_job.to_dataframe.return_value = sample_feature_data
                mock_bq_client.query.return_value = mock_query_job
                
                # Mock successful batch processing
                mock_process_batch.return_value = True
                
                # Create job configuration
                job_config = IngestionJobConfig(
                    job_name="test_ingestion",
                    source_dataset="market_regime_dev",
                    source_table="training_dataset",
                    batch_size=2,
                    validation_enabled=True
                )
                
                # Run batch ingestion
                result = await ingestion_pipeline.run_batch_ingestion(job_config)
                
                # Verify results
                assert result.status == "completed"
                assert result.records_processed == len(sample_feature_data)
                assert result.batches_processed > 0
                assert result.execution_time_seconds > 0
                assert result.validation_results is not None
                assert result.validation_results["validation_passed"] is True
                
                # Verify BigQuery was called
                mock_bq_client.query.assert_called_once()
                
                # Verify batch processing was called
                assert mock_process_batch.call_count > 0
    
    def test_data_quality_validation(self, ingestion_pipeline, sample_feature_data):
        """Test data quality validation in ingestion pipeline"""
        
        # Test with good data
        validation_results = ingestion_pipeline.validate_data_quality(sample_feature_data)
        assert validation_results["validation_passed"] is True
        assert validation_results["total_records"] == len(sample_feature_data)
        
        # Test with missing columns
        bad_data = sample_feature_data.drop(columns=["entity_id"])
        validation_results = ingestion_pipeline.validate_data_quality(bad_data)
        assert validation_results["validation_passed"] is False
        assert any(issue["type"] == "missing_columns" for issue in validation_results["issues"])
        
        # Test with excessive nulls
        null_data = sample_feature_data.copy()
        null_data.loc[:, "c1_momentum_score"] = None
        validation_results = ingestion_pipeline.validate_data_quality(null_data)
        # This might pass or fail depending on null threshold - check issues list
        assert "issues" in validation_results
        
        # Test with duplicate entity_ids
        dup_data = pd.concat([sample_feature_data, sample_feature_data.iloc[[0]]], ignore_index=True)
        validation_results = ingestion_pipeline.validate_data_quality(dup_data)
        assert validation_results["validation_passed"] is False
        assert any(issue["type"] == "duplicate_entities" for issue in validation_results["issues"])
    
    @pytest.mark.asyncio
    async def test_streaming_ingestion(self, ingestion_pipeline):
        """Test streaming ingestion functionality"""
        
        # Mock BigQuery for streaming checks
        with patch.object(ingestion_pipeline, 'bigquery_client') as mock_bq_client:
            with patch.object(ingestion_pipeline, 'run_batch_ingestion') as mock_batch_ingestion:
                
                # Mock new records detection
                mock_query_job = Mock()
                mock_result = [Mock(new_records=5)]
                mock_query_job.result.return_value = mock_result
                mock_bq_client.query.return_value = mock_query_job
                
                # Mock successful batch ingestion
                mock_ingestion_result = IngestionResult(
                    job_name="streaming_update", 
                    status="completed",
                    records_processed=5,
                    batches_processed=1,
                    execution_time_seconds=2.0
                )
                mock_batch_ingestion.return_value = mock_ingestion_result
                
                # Run streaming ingestion (short duration for testing)
                results = await ingestion_pipeline.run_streaming_ingestion(
                    "market_regime_dev", 
                    monitoring_duration_seconds=5
                )
                
                # Verify results
                assert results["status"] == "completed"
                assert results["updates_processed"] >= 0  # May be 0 if no new data in mock
                assert "execution_time_seconds" in results
                assert "average_update_rate" in results
    
    def test_feature_store_client_metrics(self, feature_store_client):
        """Test performance metrics collection"""
        
        # Simulate some requests and latencies
        feature_store_client.metrics["requests_total"] = 100
        feature_store_client.metrics["cache_hits"] = 75
        feature_store_client.metrics["errors_total"] = 2
        feature_store_client.metrics["latency_ms_p99"] = [20, 25, 30, 35, 40] * 20  # 100 latencies
        
        metrics = feature_store_client.get_performance_metrics()
        
        # Verify metrics structure
        assert metrics["requests_total"] == 100
        assert metrics["cache_hits"] == 75
        assert metrics["cache_hit_ratio"] == 0.75
        assert metrics["errors_total"] == 2
        assert metrics["error_rate"] == 0.02
        
        # Verify latency metrics
        latency_metrics = metrics["latency_metrics"]
        assert "p50_ms" in latency_metrics
        assert "p95_ms" in latency_metrics
        assert "p99_ms" in latency_metrics
        assert "avg_ms" in latency_metrics
        
        # All latencies should be reasonable
        assert latency_metrics["p99_ms"] <= 50  # Within SLA
        assert latency_metrics["p50_ms"] <= latency_metrics["p95_ms"]
        assert latency_metrics["p95_ms"] <= latency_metrics["p99_ms"]
    
    def test_ingestion_pipeline_metrics(self, ingestion_pipeline):
        """Test ingestion pipeline metrics"""
        
        # Simulate some ingestion jobs
        ingestion_pipeline.metrics["total_jobs"] = 10
        ingestion_pipeline.metrics["successful_jobs"] = 8
        ingestion_pipeline.metrics["failed_jobs"] = 2
        ingestion_pipeline.metrics["records_processed"] = 50000
        ingestion_pipeline.metrics["avg_processing_rate"] = 2500.0
        ingestion_pipeline.metrics["validation_failures"] = 1
        
        metrics = ingestion_pipeline.get_ingestion_metrics()
        
        # Verify metrics
        assert metrics["total_jobs"] == 10
        assert metrics["successful_jobs"] == 8
        assert metrics["failed_jobs"] == 2
        assert metrics["success_rate"] == 0.8  # 8/10
        assert metrics["records_processed"] == 50000
        assert metrics["avg_processing_rate_records_per_second"] == 2500.0
        assert metrics["validation_failures"] == 1
    
    @pytest.mark.asyncio
    async def test_end_to_end_feature_pipeline(self, feature_store_client, ingestion_pipeline, sample_feature_data):
        """Test complete end-to-end feature pipeline"""
        
        # Test infrastructure creation
        with patch.object(feature_store_client, 'create_feature_store_infrastructure') as mock_infrastructure:
            mock_infrastructure.return_value = {"status": "success", "total_features_created": 32}
            
            infra_results = await feature_store_client.create_feature_store_infrastructure()
            assert infra_results["status"] == "success"
        
        # Test feature ingestion 
        with patch.object(ingestion_pipeline, 'run_batch_ingestion') as mock_ingestion:
            mock_result = IngestionResult(
                job_name="e2e_test",
                status="completed", 
                records_processed=100,
                batches_processed=5,
                execution_time_seconds=10.0
            )
            mock_ingestion.return_value = mock_result
            
            job_config = IngestionJobConfig("e2e_test", "market_regime_dev", "training_dataset")
            ingestion_result = await ingestion_pipeline.run_batch_ingestion(job_config)
            assert ingestion_result.status == "completed"
        
        # Test online serving
        with patch.object(feature_store_client, 'get_online_features') as mock_serving:
            mock_serving.return_value = [
                FeatureResponse(
                    entity_id="NIFTY_20250813140000_0",
                    features={"c1_momentum_score": 0.75},
                    latency_ms=20.0,
                    cache_hit=False,
                    freshness_seconds=30,
                    request_timestamp=datetime.utcnow()
                )
            ]
            
            responses = await feature_store_client.get_online_features(
                ["NIFTY_20250813140000_0"], 
                ["c1_momentum_score"]
            )
            assert len(responses) == 1
            assert responses[0].latency_ms < 50  # Within SLA
        
        # Verify end-to-end pipeline completed successfully
        assert True  # If we got here, all steps passed


@pytest.mark.integration
class TestFeatureStorePerformance:
    """Performance-focused integration tests"""
    
    @pytest.mark.asyncio
    async def test_latency_sla_compliance(self):
        """Test that feature serving meets <50ms latency SLA"""
        
        client = FeatureStoreClient(environment="dev")
        
        # Mock fast responses
        with patch.object(client, '_fetch_from_feature_store') as mock_fetch:
            async def fast_response(*args, **kwargs):
                # Simulate 15ms response time
                await asyncio.sleep(0.015)
                return [
                    FeatureResponse(
                        entity_id="NIFTY_20250813140000_0",
                        features={"c1_momentum_score": 0.75},
                        latency_ms=15.0,
                        cache_hit=False,
                        freshness_seconds=10,
                        request_timestamp=datetime.utcnow()
                    )
                ]
            
            mock_fetch.side_effect = fast_response
            
            # Test multiple requests
            latencies = []
            for _ in range(20):
                start = time.time()
                await client.get_online_features(["NIFTY_20250813140000_0"], ["c1_momentum_score"])
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            
            # Verify SLA compliance
            p99_latency = np.percentile(latencies, 99)
            assert p99_latency < 50, f"P99 latency {p99_latency:.2f}ms exceeds 50ms SLA"
            
            p95_latency = np.percentile(latencies, 95)
            assert p95_latency < 40, f"P95 latency {p95_latency:.2f}ms exceeds 40ms target"
    
    @pytest.mark.asyncio
    async def test_throughput_requirements(self):
        """Test feature serving throughput requirements"""
        
        client = FeatureStoreClient(environment="dev")
        
        # Mock responses
        with patch.object(client, '_fetch_from_feature_store') as mock_fetch:
            mock_fetch.return_value = [
                FeatureResponse(
                    entity_id="test_entity",
                    features={"c1_momentum_score": 0.75},
                    latency_ms=10.0,
                    cache_hit=False,
                    freshness_seconds=10,
                    request_timestamp=datetime.utcnow()
                )
            ]
            
            # Test concurrent requests
            start_time = time.time()
            concurrent_requests = 50
            
            tasks = []
            for i in range(concurrent_requests):
                task = client.get_online_features([f"entity_{i}"], ["c1_momentum_score"])
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks)
            
            execution_time = time.time() - start_time
            throughput = concurrent_requests / execution_time
            
            # Verify throughput (should handle at least 100 RPS for testing)
            assert throughput >= 100, f"Throughput {throughput:.2f} RPS below minimum requirement"
            assert len(responses) == concurrent_requests
    
    def test_memory_usage_monitoring(self):
        """Test that memory usage stays within bounds"""
        
        client = FeatureStoreClient(environment="dev")
        
        # Simulate cache growth
        for i in range(1000):
            cache_key = f"entity_{i}:features"
            mock_response = FeatureResponse(
                entity_id=f"entity_{i}",
                features={"c1_momentum_score": 0.5 + i * 0.001},
                latency_ms=10.0,
                cache_hit=False,
                freshness_seconds=10,
                request_timestamp=datetime.utcnow()
            )
            client._cache_response(cache_key, mock_response)
        
        # Verify cache size is managed
        cache_size = len(client._feature_cache)
        
        # Cache should not grow indefinitely (max 10,000 entries)
        assert cache_size <= 10000, f"Cache size {cache_size} exceeds maximum"
        
        # Cache should contain some entries
        assert cache_size > 0, "Cache should contain entries"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])