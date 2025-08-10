"""
Algorithm Metadata System

Manages metadata for optimization algorithms including performance profiles,
capabilities, resource requirements, and hyperparameter schemas.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AlgorithmCategory(Enum):
    """Algorithm categories"""
    CLASSICAL = "classical"
    EVOLUTIONARY = "evolutionary"
    SWARM = "swarm"
    PHYSICS_INSPIRED = "physics_inspired"
    QUANTUM = "quantum"

class ProblemType(Enum):
    """Types of optimization problems"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIXED = "mixed"
    CONSTRAINED = "constrained"
    MULTI_OBJECTIVE = "multi_objective"

class ComplexityLevel(Enum):
    """Computational complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class ResourceRequirements:
    """Resource requirements for an algorithm"""
    memory_mb: int = 100
    cpu_cores: int = 1
    gpu_required: bool = False
    gpu_memory_mb: int = 0
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    typical_runtime_seconds: float = 10.0

@dataclass
class AlgorithmCapabilities:
    """Capabilities of an optimization algorithm"""
    problem_types: Set[ProblemType] = field(default_factory=set)
    min_dimensions: int = 1
    max_dimensions: int = 1000
    supports_constraints: bool = False
    supports_multi_objective: bool = False
    supports_parallel: bool = False
    supports_gpu: bool = False
    handles_noise: bool = True
    requires_gradient: bool = False

@dataclass
class PerformanceProfile:
    """Performance profile tracking for an algorithm"""
    total_runs: int = 0
    successful_runs: int = 0
    average_runtime: float = 0.0
    average_improvement: float = 0.0
    best_improvement: float = 0.0
    worst_improvement: float = 0.0
    convergence_rate: float = 0.0
    last_updated: Optional[float] = None

@dataclass
class AlgorithmMetadata:
    """Complete metadata for an optimization algorithm"""
    name: str
    category: AlgorithmCategory
    description: str
    version: str = "1.0.0"
    author: str = "Optimization Team"
    
    # Technical specifications
    capabilities: AlgorithmCapabilities = field(default_factory=AlgorithmCapabilities)
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Performance tracking
    performance_profile: PerformanceProfile = field(default_factory=PerformanceProfile)
    
    # Configuration
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Dependencies
    required_packages: List[str] = field(default_factory=list)
    optional_packages: List[str] = field(default_factory=list)
    
    # Additional metadata
    tags: Set[str] = field(default_factory=set)
    references: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

class AlgorithmMetadataManager:
    """
    Manages metadata for all optimization algorithms
    
    Provides centralized management of algorithm capabilities, performance
    tracking, and resource requirements.
    """
    
    def __init__(self, metadata_file: Optional[str] = None):
        """
        Initialize metadata manager
        
        Args:
            metadata_file: Path to persistent metadata storage
        """
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.metadata_store: Dict[str, AlgorithmMetadata] = {}
        
        # Load existing metadata if available
        if self.metadata_file and self.metadata_file.exists():
            self._load_metadata()
        
        logger.info("AlgorithmMetadataManager initialized")
    
    def register_algorithm_metadata(self, 
                                  algorithm_name: str,
                                  metadata: AlgorithmMetadata) -> None:
        """
        Register metadata for an algorithm
        
        Args:
            algorithm_name: Name of the algorithm
            metadata: Algorithm metadata
        """
        self.metadata_store[algorithm_name] = metadata
        logger.debug(f"Registered metadata for algorithm: {algorithm_name}")
    
    def get_algorithm_metadata(self, algorithm_name: str) -> Optional[AlgorithmMetadata]:
        """Get metadata for a specific algorithm"""
        return self.metadata_store.get(algorithm_name)
    
    def get_algorithms_by_category(self, category: AlgorithmCategory) -> List[str]:
        """Get all algorithms in a specific category"""
        return [
            name for name, metadata in self.metadata_store.items()
            if metadata.category == category
        ]
    
    def get_algorithms_by_capability(self, 
                                   problem_type: Optional[ProblemType] = None,
                                   supports_gpu: Optional[bool] = None,
                                   supports_parallel: Optional[bool] = None,
                                   max_dimensions: Optional[int] = None) -> List[str]:
        """
        Get algorithms matching specific capabilities
        
        Args:
            problem_type: Required problem type support
            supports_gpu: Whether GPU support is required
            supports_parallel: Whether parallel support is required
            max_dimensions: Maximum dimension requirement
            
        Returns:
            List of matching algorithm names
        """
        matching_algorithms = []
        
        for name, metadata in self.metadata_store.items():
            capabilities = metadata.capabilities
            
            # Check problem type
            if problem_type and problem_type not in capabilities.problem_types:
                continue
            
            # Check GPU support
            if supports_gpu is not None and capabilities.supports_gpu != supports_gpu:
                continue
            
            # Check parallel support
            if supports_parallel is not None and capabilities.supports_parallel != supports_parallel:
                continue
            
            # Check dimension requirements
            if max_dimensions and capabilities.max_dimensions < max_dimensions:
                continue
            
            matching_algorithms.append(name)
        
        return matching_algorithms
    
    def recommend_algorithms(self,
                           problem_characteristics: Dict[str, Any],
                           resource_constraints: Optional[Dict[str, Any]] = None,
                           performance_priority: str = "balanced") -> List[Tuple[str, float]]:
        """
        Recommend algorithms based on problem characteristics and constraints
        
        Args:
            problem_characteristics: Problem-specific requirements
            resource_constraints: Available computational resources
            performance_priority: 'speed', 'quality', 'balanced'
            
        Returns:
            List of (algorithm_name, score) tuples sorted by recommendation score
        """
        recommendations = []
        
        for name, metadata in self.metadata_store.items():
            score = self._calculate_recommendation_score(
                metadata, problem_characteristics, resource_constraints, performance_priority
            )
            
            if score > 0:  # Only include viable algorithms
                recommendations.append((name, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def update_performance_profile(self,
                                 algorithm_name: str,
                                 runtime: float,
                                 improvement: float,
                                 success: bool = True) -> None:
        """
        Update performance profile for an algorithm
        
        Args:
            algorithm_name: Name of the algorithm
            runtime: Execution time in seconds
            improvement: Performance improvement achieved
            success: Whether the optimization was successful
        """
        if algorithm_name not in self.metadata_store:
            logger.warning(f"Algorithm {algorithm_name} not found in metadata store")
            return
        
        profile = self.metadata_store[algorithm_name].performance_profile
        
        # Update counters
        profile.total_runs += 1
        if success:
            profile.successful_runs += 1
        
        # Update runtime statistics
        if profile.total_runs == 1:
            profile.average_runtime = runtime
        else:
            profile.average_runtime = (
                (profile.average_runtime * (profile.total_runs - 1) + runtime) / 
                profile.total_runs
            )
        
        # Update improvement statistics (only for successful runs)
        if success:
            if profile.successful_runs == 1:
                profile.average_improvement = improvement
                profile.best_improvement = improvement
                profile.worst_improvement = improvement
            else:
                profile.average_improvement = (
                    (profile.average_improvement * (profile.successful_runs - 1) + improvement) /
                    profile.successful_runs
                )
                profile.best_improvement = max(profile.best_improvement, improvement)
                profile.worst_improvement = min(profile.worst_improvement, improvement)
        
        # Update convergence rate
        profile.convergence_rate = profile.successful_runs / profile.total_runs
        profile.last_updated = time.time()
        
        logger.debug(f"Updated performance profile for {algorithm_name}")
    
    def get_performance_ranking(self, 
                              metric: str = "average_improvement",
                              min_runs: int = 5) -> List[Tuple[str, float]]:
        """
        Get algorithms ranked by performance metric
        
        Args:
            metric: Performance metric to rank by
            min_runs: Minimum number of runs required for ranking
            
        Returns:
            List of (algorithm_name, metric_value) tuples
        """
        rankings = []
        
        for name, metadata in self.metadata_store.items():
            profile = metadata.performance_profile
            
            if profile.total_runs < min_runs:
                continue
            
            metric_value = getattr(profile, metric, 0.0)
            rankings.append((name, metric_value))
        
        # Sort by metric value (highest first for improvement metrics)
        reverse = metric in ["average_improvement", "best_improvement", "convergence_rate"]
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        
        return rankings
    
    def generate_capability_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Generate a capability matrix for all algorithms"""
        matrix = {}
        
        for name, metadata in self.metadata_store.items():
            capabilities = metadata.capabilities
            matrix[name] = {
                'category': metadata.category.value,
                'problem_types': [pt.value for pt in capabilities.problem_types],
                'min_dimensions': capabilities.min_dimensions,
                'max_dimensions': capabilities.max_dimensions,
                'supports_constraints': capabilities.supports_constraints,
                'supports_multi_objective': capabilities.supports_multi_objective,
                'supports_parallel': capabilities.supports_parallel,
                'supports_gpu': capabilities.supports_gpu,
                'handles_noise': capabilities.handles_noise,
                'requires_gradient': capabilities.requires_gradient,
                'memory_mb': metadata.resource_requirements.memory_mb,
                'complexity': metadata.resource_requirements.complexity.value,
                'success_rate': metadata.performance_profile.convergence_rate
            }
        
        return matrix
    
    def export_metadata_report(self) -> Dict[str, Any]:
        """Export comprehensive metadata report"""
        report = {
            'summary': {
                'total_algorithms': len(self.metadata_store),
                'categories': {},
                'total_runs': sum(m.performance_profile.total_runs for m in self.metadata_store.values()),
                'average_success_rate': 0.0
            },
            'algorithms': {},
            'capability_matrix': self.generate_capability_matrix(),
            'performance_rankings': {},
            'generated_at': time.time()
        }
        
        # Category breakdown
        for category in AlgorithmCategory:
            algorithms_in_category = self.get_algorithms_by_category(category)
            report['summary']['categories'][category.value] = len(algorithms_in_category)
        
        # Average success rate
        success_rates = [
            m.performance_profile.convergence_rate 
            for m in self.metadata_store.values()
            if m.performance_profile.total_runs > 0
        ]
        if success_rates:
            report['summary']['average_success_rate'] = sum(success_rates) / len(success_rates)
        
        # Individual algorithm details
        for name, metadata in self.metadata_store.items():
            report['algorithms'][name] = {
                'category': metadata.category.value,
                'description': metadata.description,
                'version': metadata.version,
                'capabilities': {
                    'problem_types': [pt.value for pt in metadata.capabilities.problem_types],
                    'supports_gpu': metadata.capabilities.supports_gpu,
                    'supports_parallel': metadata.capabilities.supports_parallel
                },
                'performance': {
                    'total_runs': metadata.performance_profile.total_runs,
                    'success_rate': metadata.performance_profile.convergence_rate,
                    'average_runtime': metadata.performance_profile.average_runtime,
                    'average_improvement': metadata.performance_profile.average_improvement
                }
            }
        
        # Performance rankings
        for metric in ['average_improvement', 'convergence_rate', 'average_runtime']:
            ranking = self.get_performance_ranking(metric)
            report['performance_rankings'][metric] = ranking[:5]  # Top 5
        
        return report
    
    def _calculate_recommendation_score(self,
                                     metadata: AlgorithmMetadata,
                                     problem_characteristics: Dict[str, Any],
                                     resource_constraints: Optional[Dict[str, Any]],
                                     performance_priority: str) -> float:
        """Calculate recommendation score for an algorithm"""
        score = 0.0
        
        # Base score from performance profile
        profile = metadata.performance_profile
        if profile.total_runs > 0:
            # Weight by success rate and average improvement
            score += profile.convergence_rate * 30  # Up to 30 points for reliability
            score += min(profile.average_improvement * 10, 20)  # Up to 20 points for performance
        else:
            score += 10  # Default score for untested algorithms
        
        # Problem type compatibility
        problem_type_str = problem_characteristics.get('problem_type', 'continuous')
        try:
            problem_type = ProblemType(problem_type_str)
            if problem_type in metadata.capabilities.problem_types:
                score += 20  # Strong bonus for problem type match
        except ValueError:
            pass  # Unknown problem type
        
        # Dimension compatibility
        dimensions = problem_characteristics.get('dimensions', 10)
        if (metadata.capabilities.min_dimensions <= dimensions <= 
            metadata.capabilities.max_dimensions):
            score += 15
        else:
            score -= 10  # Penalty for dimension mismatch
        
        # Resource constraints
        if resource_constraints:
            # GPU availability
            gpu_available = resource_constraints.get('gpu_available', False)
            if metadata.capabilities.supports_gpu and not gpu_available:
                score -= 15  # Penalty for GPU requirement without availability
            elif metadata.capabilities.supports_gpu and gpu_available:
                score += 10  # Bonus for GPU utilization
            
            # Memory constraints
            memory_limit = resource_constraints.get('memory_limit_mb', 8192)
            if metadata.resource_requirements.memory_mb > memory_limit:
                score -= 20  # Significant penalty for exceeding memory
        
        # Performance priority adjustments
        if performance_priority == "speed":
            if metadata.resource_requirements.complexity in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM]:
                score += 15
            if profile.average_runtime < 60:  # Less than 1 minute average
                score += 10
        elif performance_priority == "quality":
            if profile.average_improvement > 0.1:  # 10% improvement threshold
                score += 15
            if metadata.category in [AlgorithmCategory.EVOLUTIONARY, AlgorithmCategory.SWARM]:
                score += 5  # These often find better solutions
        
        return max(0.0, score)  # Ensure non-negative score
    
    def _load_metadata(self) -> None:
        """Load metadata from persistent storage"""
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            for name, metadata_dict in data.items():
                # Reconstruct metadata object from dictionary
                metadata = self._dict_to_metadata(metadata_dict)
                self.metadata_store[name] = metadata
            
            logger.info(f"Loaded metadata for {len(self.metadata_store)} algorithms")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    
    def save_metadata(self) -> None:
        """Save metadata to persistent storage"""
        if not self.metadata_file:
            return
        
        try:
            # Convert metadata to serializable format
            data = {}
            for name, metadata in self.metadata_store.items():
                data[name] = self._metadata_to_dict(metadata)
            
            # Ensure directory exists
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved metadata for {len(self.metadata_store)} algorithms")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _metadata_to_dict(self, metadata: AlgorithmMetadata) -> Dict[str, Any]:
        """Convert metadata object to dictionary for serialization"""
        return {
            'name': metadata.name,
            'category': metadata.category.value,
            'description': metadata.description,
            'version': metadata.version,
            'author': metadata.author,
            'capabilities': {
                'problem_types': [pt.value for pt in metadata.capabilities.problem_types],
                'min_dimensions': metadata.capabilities.min_dimensions,
                'max_dimensions': metadata.capabilities.max_dimensions,
                'supports_constraints': metadata.capabilities.supports_constraints,
                'supports_multi_objective': metadata.capabilities.supports_multi_objective,
                'supports_parallel': metadata.capabilities.supports_parallel,
                'supports_gpu': metadata.capabilities.supports_gpu,
                'handles_noise': metadata.capabilities.handles_noise,
                'requires_gradient': metadata.capabilities.requires_gradient
            },
            'resource_requirements': {
                'memory_mb': metadata.resource_requirements.memory_mb,
                'cpu_cores': metadata.resource_requirements.cpu_cores,
                'gpu_required': metadata.resource_requirements.gpu_required,
                'gpu_memory_mb': metadata.resource_requirements.gpu_memory_mb,
                'complexity': metadata.resource_requirements.complexity.value,
                'typical_runtime_seconds': metadata.resource_requirements.typical_runtime_seconds
            },
            'performance_profile': {
                'total_runs': metadata.performance_profile.total_runs,
                'successful_runs': metadata.performance_profile.successful_runs,
                'average_runtime': metadata.performance_profile.average_runtime,
                'average_improvement': metadata.performance_profile.average_improvement,
                'best_improvement': metadata.performance_profile.best_improvement,
                'worst_improvement': metadata.performance_profile.worst_improvement,
                'convergence_rate': metadata.performance_profile.convergence_rate,
                'last_updated': metadata.performance_profile.last_updated
            },
            'default_parameters': metadata.default_parameters,
            'parameter_schema': metadata.parameter_schema,
            'required_packages': metadata.required_packages,
            'optional_packages': metadata.optional_packages,
            'tags': list(metadata.tags),
            'references': metadata.references,
            'examples': metadata.examples
        }
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> AlgorithmMetadata:
        """Convert dictionary back to metadata object"""
        # Reconstruct capabilities
        capabilities = AlgorithmCapabilities(
            problem_types={ProblemType(pt) for pt in data['capabilities']['problem_types']},
            min_dimensions=data['capabilities']['min_dimensions'],
            max_dimensions=data['capabilities']['max_dimensions'],
            supports_constraints=data['capabilities']['supports_constraints'],
            supports_multi_objective=data['capabilities']['supports_multi_objective'],
            supports_parallel=data['capabilities']['supports_parallel'],
            supports_gpu=data['capabilities']['supports_gpu'],
            handles_noise=data['capabilities']['handles_noise'],
            requires_gradient=data['capabilities']['requires_gradient']
        )
        
        # Reconstruct resource requirements
        resource_requirements = ResourceRequirements(
            memory_mb=data['resource_requirements']['memory_mb'],
            cpu_cores=data['resource_requirements']['cpu_cores'],
            gpu_required=data['resource_requirements']['gpu_required'],
            gpu_memory_mb=data['resource_requirements']['gpu_memory_mb'],
            complexity=ComplexityLevel(data['resource_requirements']['complexity']),
            typical_runtime_seconds=data['resource_requirements']['typical_runtime_seconds']
        )
        
        # Reconstruct performance profile
        performance_profile = PerformanceProfile(
            total_runs=data['performance_profile']['total_runs'],
            successful_runs=data['performance_profile']['successful_runs'],
            average_runtime=data['performance_profile']['average_runtime'],
            average_improvement=data['performance_profile']['average_improvement'],
            best_improvement=data['performance_profile']['best_improvement'],
            worst_improvement=data['performance_profile']['worst_improvement'],
            convergence_rate=data['performance_profile']['convergence_rate'],
            last_updated=data['performance_profile']['last_updated']
        )
        
        return AlgorithmMetadata(
            name=data['name'],
            category=AlgorithmCategory(data['category']),
            description=data['description'],
            version=data['version'],
            author=data['author'],
            capabilities=capabilities,
            resource_requirements=resource_requirements,
            performance_profile=performance_profile,
            default_parameters=data['default_parameters'],
            parameter_schema=data['parameter_schema'],
            required_packages=data['required_packages'],
            optional_packages=data['optional_packages'],
            tags=set(data['tags']),
            references=data['references'],
            examples=data['examples']
        )