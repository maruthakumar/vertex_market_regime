"""
Base Component for Vertex Market Regime System

Abstract base class ensuring consistent interface across all 8 components
with cloud-native integration, adaptive learning, and performance tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
import time
import numpy as np
from enum import Enum

# Google Cloud imports
try:
    from google.cloud import aiplatform
    from google.cloud import bigquery
    from google.cloud import storage
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False
    logging.warning("Google Cloud libraries not available. Install with: pip install google-cloud-aiplatform google-cloud-bigquery google-cloud-storage")


class ComponentStatus(Enum):
    """Component health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILED = "failed"
    INITIALIZING = "initializing"


@dataclass
class FeatureVector:
    """Feature vector output from component analysis"""
    features: np.ndarray
    feature_names: List[str]
    feature_count: int
    processing_time_ms: float
    metadata: Dict[str, Any]


@dataclass
class ComponentAnalysisResult:
    """Result from component analysis"""
    component_id: int
    component_name: str
    score: float
    confidence: float
    features: FeatureVector
    processing_time_ms: float
    weights: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    
@dataclass 
class PerformanceFeedback:
    """Performance feedback for adaptive learning"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    regime_specific_performance: Dict[str, float]
    timestamp: datetime


@dataclass
class WeightUpdate:
    """Weight update result from adaptive learning"""
    updated_weights: Dict[str, float]
    weight_changes: Dict[str, float] 
    performance_improvement: float
    confidence_score: float


@dataclass
class HealthStatus:
    """Component health status"""
    component: str
    status: ComponentStatus
    last_processing_time: Optional[float]
    feature_count: int
    accuracy: Optional[float]
    error_rate: float
    memory_usage_mb: float
    gpu_utilization: Optional[float]
    timestamp: datetime


class BaseMarketRegimeComponent(ABC):
    """
    Abstract base class for all market regime components
    
    Provides consistent interface, cloud integration, performance tracking,
    and adaptive learning capabilities for all 8 components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base component with configuration and cloud clients
        
        Args:
            config: Component-specific configuration dictionary
        """
        self.config = config
        self.component_name = self.__class__.__name__
        self.component_id = config.get('component_id', 0)
        
        # Logging setup
        self.logger = logging.getLogger(f"vertex_mr.{self.component_name}")
        self.logger.setLevel(logging.INFO)
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.accuracy_scores: List[float] = []
        self.error_count = 0
        self.total_requests = 0
        
        # Feature configuration
        self.feature_count = config.get('feature_count', 0)
        self.expected_features = config.get('expected_features', [])
        
        # Cloud integration setup
        self.project_id = config.get('project_id', 'arched-bot-269016')
        self.region = config.get('region', 'us-central1')
        self._initialize_cloud_clients()
        
        # GPU configuration
        self.gpu_enabled = config.get('gpu_enabled', False)
        self.gpu_device = config.get('gpu_device', 0)
        
        # Adaptive learning configuration
        self.learning_enabled = config.get('learning_enabled', True)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.performance_window = config.get('performance_window', 252)  # 1 year
        
        # Component-specific weights
        self.current_weights = config.get('initial_weights', {})
        self.weight_history: List[Dict[str, float]] = []
        
        self.logger.info(f"Initialized {self.component_name} with {self.feature_count} features")

    def _initialize_cloud_clients(self):
        """Initialize Google Cloud clients if available"""
        if CLOUD_AVAILABLE:
            try:
                self.vertex_ai_client = aiplatform.Client(
                    project=self.project_id, 
                    location=self.region
                )
                self.bigquery_client = bigquery.Client(project=self.project_id)
                self.storage_client = storage.Client(project=self.project_id)
                self.cloud_enabled = True
                self.logger.info("Cloud clients initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize cloud clients: {e}")
                self.cloud_enabled = False
        else:
            self.cloud_enabled = False
            self.vertex_ai_client = None
            self.bigquery_client = None
            self.storage_client = None

    @abstractmethod
    async def analyze(self, market_data: Any) -> ComponentAnalysisResult:
        """
        Core analysis method - must be implemented by each component
        
        Args:
            market_data: Market data input for analysis
            
        Returns:
            ComponentAnalysisResult with analysis results
        """
        pass

    @abstractmethod
    async def extract_features(self, market_data: Any) -> FeatureVector:
        """
        Feature extraction - component-specific implementation
        
        Args:
            market_data: Market data input for feature extraction
            
        Returns:
            FeatureVector with extracted features
        """
        pass

    @abstractmethod
    async def update_weights(self, performance_feedback: PerformanceFeedback) -> WeightUpdate:
        """
        Adaptive weight learning - component-specific logic
        
        Args:
            performance_feedback: Performance metrics for learning
            
        Returns:
            WeightUpdate with updated weights and changes
        """
        pass

    async def health_check(self) -> HealthStatus:
        """
        Component health monitoring
        
        Returns:
            HealthStatus with current component health metrics
        """
        # Determine status based on recent performance
        status = ComponentStatus.HEALTHY
        if self.total_requests > 0:
            error_rate = self.error_count / self.total_requests
            if error_rate > 0.1:  # >10% error rate
                status = ComponentStatus.DEGRADED
            elif error_rate > 0.2:  # >20% error rate
                status = ComponentStatus.FAILED
        
        # Get memory usage (placeholder - would need actual implementation)
        memory_usage = self._get_memory_usage()
        
        # Get GPU utilization if enabled
        gpu_utilization = self._get_gpu_utilization() if self.gpu_enabled else None
        
        return HealthStatus(
            component=self.component_name,
            status=status,
            last_processing_time=self.processing_times[-1] if self.processing_times else None,
            feature_count=self.feature_count,
            accuracy=self.accuracy_scores[-1] if self.accuracy_scores else None,
            error_rate=self.error_count / max(self.total_requests, 1),
            memory_usage_mb=memory_usage,
            gpu_utilization=gpu_utilization,
            timestamp=datetime.utcnow()
        )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(self.gpu_device)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except:
            return None

    async def benchmark_performance(self, test_data: Any, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark component performance
        
        Args:
            test_data: Test data for benchmarking
            iterations: Number of iterations to run
            
        Returns:
            Performance metrics dictionary
        """
        processing_times = []
        
        for i in range(iterations):
            start_time = time.time()
            try:
                result = await self.analyze(test_data)
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
            except Exception as e:
                self.logger.error(f"Benchmark iteration {i} failed: {e}")
        
        return {
            'mean_processing_time_ms': np.mean(processing_times),
            'std_processing_time_ms': np.std(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'p95_processing_time_ms': np.percentile(processing_times, 95),
            'p99_processing_time_ms': np.percentile(processing_times, 99),
            'success_rate': len(processing_times) / iterations
        }

    def _track_performance(self, processing_time: float, success: bool = True):
        """Track component performance metrics"""
        self.processing_times.append(processing_time)
        self.total_requests += 1
        
        if not success:
            self.error_count += 1
        
        # Keep only recent performance data
        max_history = self.performance_window
        if len(self.processing_times) > max_history:
            self.processing_times = self.processing_times[-max_history:]

    def _validate_features(self, features: FeatureVector) -> bool:
        """Validate extracted features meet component specifications"""
        if features.feature_count != self.feature_count:
            self.logger.warning(
                f"Feature count mismatch: expected {self.feature_count}, got {features.feature_count}"
            )
            return False
        
        if len(features.features) != features.feature_count:
            self.logger.error("Feature array length doesn't match feature count")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features.features)) or np.any(np.isinf(features.features)):
            self.logger.error("Features contain NaN or infinite values")
            return False
        
        return True

    async def save_model_state(self, bucket_name: str = None) -> bool:
        """
        Save component model state to Google Cloud Storage
        
        Args:
            bucket_name: GCS bucket name (optional)
            
        Returns:
            Success status
        """
        if not self.cloud_enabled:
            self.logger.warning("Cloud storage not available")
            return False
        
        try:
            bucket_name = bucket_name or f"{self.project_id}-market-regime-models"
            bucket = self.storage_client.bucket(bucket_name)
            
            # Create model state dictionary
            model_state = {
                'component_id': self.component_id,
                'component_name': self.component_name,
                'current_weights': self.current_weights,
                'weight_history': self.weight_history[-100:],  # Last 100 updates
                'performance_metrics': {
                    'processing_times': self.processing_times[-1000:],
                    'accuracy_scores': self.accuracy_scores[-1000:]
                },
                'config': self.config,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Save to GCS
            blob_name = f"components/{self.component_name.lower()}/model_state.json"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(model_state, indent=2),
                content_type='application/json'
            )
            
            self.logger.info(f"Model state saved to gs://{bucket_name}/{blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model state: {e}")
            return False

    async def load_model_state(self, bucket_name: str = None) -> bool:
        """
        Load component model state from Google Cloud Storage
        
        Args:
            bucket_name: GCS bucket name (optional)
            
        Returns:
            Success status
        """
        if not self.cloud_enabled:
            self.logger.warning("Cloud storage not available")
            return False
        
        try:
            bucket_name = bucket_name or f"{self.project_id}-market-regime-models"
            bucket = self.storage_client.bucket(bucket_name)
            
            blob_name = f"components/{self.component_name.lower()}/model_state.json"
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                self.logger.info("No saved model state found")
                return False
            
            # Load from GCS
            model_state = json.loads(blob.download_as_text())
            
            # Restore state
            self.current_weights = model_state.get('current_weights', {})
            self.weight_history = model_state.get('weight_history', [])
            
            performance = model_state.get('performance_metrics', {})
            self.processing_times = performance.get('processing_times', [])
            self.accuracy_scores = performance.get('accuracy_scores', [])
            
            self.logger.info(f"Model state loaded from gs://{bucket_name}/{blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of component"""
        return f"{self.component_name}(id={self.component_id}, features={self.feature_count})"


# Component factory for dynamic component loading
class ComponentFactory:
    """Factory for creating component instances"""
    
    _component_registry = {}
    
    @classmethod
    def register_component(cls, component_id: int, component_class):
        """Register a component class"""
        cls._component_registry[component_id] = component_class
    
    @classmethod
    def create_component(cls, component_id: int, config: Dict[str, Any]) -> BaseMarketRegimeComponent:
        """Create component instance by ID"""
        if component_id not in cls._component_registry:
            raise ValueError(f"Component {component_id} not registered")
        
        component_class = cls._component_registry[component_id]
        return component_class(config)
    
    @classmethod
    def list_components(cls) -> List[int]:
        """List registered component IDs"""
        return list(cls._component_registry.keys())