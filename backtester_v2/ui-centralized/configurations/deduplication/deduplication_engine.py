"""
Advanced Deduplication Engine

Enterprise-grade deduplication system for configuration files with:
- Multi-level similarity detection (content, semantic, structural)
- Intelligent clustering and grouping
- Performance optimization for large datasets
- Configuration merge and consolidation capabilities
- Advanced analytics and reporting
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from difflib import SequenceMatcher

from ..parameter_registry import ParameterRegistry, ParameterDefinition
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class SimilarityMetrics:
    """Similarity metrics between two configurations"""
    content_hash_match: bool = False
    content_similarity: float = 0.0  # 0-1, based on parameter values
    structural_similarity: float = 0.0  # 0-1, based on parameter keys/types
    semantic_similarity: float = 0.0  # 0-1, based on parameter meanings
    overall_similarity: float = 0.0  # Weighted average
    differences: List[str] = field(default_factory=list)
    common_parameters: Set[str] = field(default_factory=set)
    unique_parameters_a: Set[str] = field(default_factory=set)
    unique_parameters_b: Set[str] = field(default_factory=set)

@dataclass
class DuplicateGroup:
    """Group of similar/duplicate configurations"""
    group_id: str
    primary_config_id: str
    member_configs: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.0
    total_size: int = 0
    saved_space: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    group_type: str = "exact"  # exact, similar, semantic

@dataclass
class DeduplicationReport:
    """Comprehensive deduplication analysis report"""
    total_configurations: int = 0
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    exact_duplicates: int = 0
    similar_configurations: int = 0
    unique_configurations: int = 0
    total_original_size: int = 0
    total_deduplicated_size: int = 0
    space_saved: int = 0
    space_saved_percentage: float = 0.0
    processing_time: float = 0.0
    strategy_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    similarity_distribution: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class ContentHasher:
    """Advanced content hashing with multiple algorithms"""
    
    @staticmethod
    def calculate_content_hash(config_data: Dict[str, Any], algorithm: str = "sha256") -> str:
        """Calculate content hash using specified algorithm"""
        # Normalize the configuration data
        normalized = ContentHasher._normalize_config(config_data)
        content_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        
        if algorithm == "sha256":
            return hashlib.sha256(content_str.encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(content_str.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(content_str.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    @staticmethod
    def calculate_structural_hash(config_data: Dict[str, Any]) -> str:
        """Calculate hash based on structure (keys and types) only"""
        structure = ContentHasher._extract_structure(config_data)
        structure_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(structure_str.encode()).hexdigest()
    
    @staticmethod
    def calculate_semantic_hash(config_data: Dict[str, Any], registry: ParameterRegistry) -> str:
        """Calculate hash based on semantic meaning of parameters"""
        semantic_groups = defaultdict(list)
        
        for key, value in config_data.items():
            # Get parameter definition for semantic grouping
            param_def = registry.get_parameter(key)
            if param_def:
                semantic_key = f"{param_def.category}_{param_def.data_type.value}"
                semantic_groups[semantic_key].append(str(value))
        
        # Sort values within each group for consistency
        for group in semantic_groups.values():
            group.sort()
        
        semantic_str = json.dumps(dict(semantic_groups), sort_keys=True)
        return hashlib.sha256(semantic_str.encode()).hexdigest()
    
    @staticmethod
    def _normalize_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration data for consistent hashing"""
        normalized = {}
        
        for key, value in config_data.items():
            # Normalize key
            norm_key = key.lower().strip()
            
            # Normalize value based on type
            if isinstance(value, (int, float)):
                # Round floats to avoid precision differences
                normalized[norm_key] = round(float(value), 6) if isinstance(value, float) else value
            elif isinstance(value, str):
                # Normalize string values
                normalized[norm_key] = value.strip().lower()
            elif isinstance(value, bool):
                normalized[norm_key] = value
            elif isinstance(value, (list, tuple)):
                # Sort lists for consistency
                try:
                    normalized[norm_key] = sorted([ContentHasher._normalize_value(v) for v in value])
                except TypeError:
                    # If not sortable, convert to strings
                    normalized[norm_key] = sorted([str(v) for v in value])
            elif isinstance(value, dict):
                # Recursively normalize nested dicts
                normalized[norm_key] = ContentHasher._normalize_config(value)
            else:
                normalized[norm_key] = str(value)
        
        return normalized
    
    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Normalize individual values"""
        if isinstance(value, (int, float)):
            return round(float(value), 6) if isinstance(value, float) else value
        elif isinstance(value, str):
            return value.strip().lower()
        else:
            return str(value)
    
    @staticmethod
    def _extract_structure(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structure (keys and types) from configuration"""
        structure = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                structure[key] = ContentHasher._extract_structure(value)
            elif isinstance(value, list):
                if value:
                    structure[key] = f"list_of_{type(value[0]).__name__}"
                else:
                    structure[key] = "empty_list"
            else:
                structure[key] = type(value).__name__
        
        return structure

class SimilarityCalculator:
    """Advanced similarity calculation between configurations"""
    
    def __init__(self, registry: ParameterRegistry):
        self.registry = registry
        
        # Weights for different similarity components
        self.weights = {
            "content": 0.4,
            "structural": 0.3,
            "semantic": 0.3
        }
    
    def calculate_similarity(self, 
                           config_a: Dict[str, Any], 
                           config_b: Dict[str, Any],
                           config_id_a: str = "A",
                           config_id_b: str = "B") -> SimilarityMetrics:
        """Calculate comprehensive similarity between two configurations"""
        
        # Content hash comparison (exact match)
        hash_a = ContentHasher.calculate_content_hash(config_a)
        hash_b = ContentHasher.calculate_content_hash(config_b)
        content_hash_match = hash_a == hash_b
        
        # Content similarity (parameter values)
        content_similarity = self._calculate_content_similarity(config_a, config_b)
        
        # Structural similarity (keys and types)
        structural_similarity = self._calculate_structural_similarity(config_a, config_b)
        
        # Semantic similarity (parameter meanings)
        semantic_similarity = self._calculate_semantic_similarity(config_a, config_b)
        
        # Overall weighted similarity
        overall_similarity = (
            content_similarity * self.weights["content"] +
            structural_similarity * self.weights["structural"] + 
            semantic_similarity * self.weights["semantic"]
        )
        
        # Find differences
        differences = self._find_differences(config_a, config_b, config_id_a, config_id_b)
        
        # Analyze parameter overlap
        keys_a = set(self._flatten_keys(config_a))
        keys_b = set(self._flatten_keys(config_b))
        common_parameters = keys_a.intersection(keys_b)
        unique_a = keys_a - keys_b
        unique_b = keys_b - keys_a
        
        return SimilarityMetrics(
            content_hash_match=content_hash_match,
            content_similarity=content_similarity,
            structural_similarity=structural_similarity,
            semantic_similarity=semantic_similarity,
            overall_similarity=overall_similarity,
            differences=differences,
            common_parameters=common_parameters,
            unique_parameters_a=unique_a,
            unique_parameters_b=unique_b
        )
    
    def _calculate_content_similarity(self, config_a: Dict[str, Any], config_b: Dict[str, Any]) -> float:
        """Calculate similarity based on parameter values"""
        flat_a = self._flatten_config(config_a)
        flat_b = self._flatten_config(config_b)
        
        all_keys = set(flat_a.keys()).union(set(flat_b.keys()))
        if not all_keys:
            return 1.0
        
        matching_count = 0
        total_count = len(all_keys)
        
        for key in all_keys:
            val_a = flat_a.get(key)
            val_b = flat_b.get(key)
            
            if val_a is None or val_b is None:
                continue  # Missing parameter
            
            # Compare values with type consideration
            if self._values_equal(val_a, val_b):
                matching_count += 1
        
        return matching_count / total_count if total_count > 0 else 1.0
    
    def _calculate_structural_similarity(self, config_a: Dict[str, Any], config_b: Dict[str, Any]) -> float:
        """Calculate similarity based on structure (keys and types)"""
        struct_a = ContentHasher._extract_structure(config_a)
        struct_b = ContentHasher._extract_structure(config_b)
        
        flat_struct_a = self._flatten_config(struct_a)
        flat_struct_b = self._flatten_config(struct_b)
        
        all_keys = set(flat_struct_a.keys()).union(set(flat_struct_b.keys()))
        if not all_keys:
            return 1.0
        
        matching_types = 0
        for key in all_keys:
            type_a = flat_struct_a.get(key)
            type_b = flat_struct_b.get(key)
            
            if type_a == type_b:
                matching_types += 1
        
        return matching_types / len(all_keys)
    
    def _calculate_semantic_similarity(self, config_a: Dict[str, Any], config_b: Dict[str, Any]) -> float:
        """Calculate similarity based on semantic meaning"""
        semantic_a = self._group_by_semantics(config_a)
        semantic_b = self._group_by_semantics(config_b)
        
        all_groups = set(semantic_a.keys()).union(set(semantic_b.keys()))
        if not all_groups:
            return 1.0
        
        total_similarity = 0.0
        for group in all_groups:
            values_a = semantic_a.get(group, [])
            values_b = semantic_b.get(group, [])
            
            # Calculate similarity within semantic group
            group_similarity = self._calculate_list_similarity(values_a, values_b)
            total_similarity += group_similarity
        
        return total_similarity / len(all_groups)
    
    def _group_by_semantics(self, config: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Group parameters by semantic meaning"""
        semantic_groups = defaultdict(list)
        flat_config = self._flatten_config(config)
        
        for key, value in flat_config.items():
            # Get parameter definition
            param_def = self.registry.get_parameter(key)
            if param_def:
                group_key = f"{param_def.category}_{param_def.data_type.value}"
                semantic_groups[group_key].append(value)
            else:
                # Fallback semantic grouping based on key patterns
                if any(risk_term in key.lower() for risk_term in ['risk', 'stop', 'loss']):
                    semantic_groups['risk_management'].append(value)
                elif any(entry_term in key.lower() for entry_term in ['entry', 'signal', 'trigger']):
                    semantic_groups['entry_logic'].append(value)
                elif any(exit_term in key.lower() for exit_term in ['exit', 'target', 'profit']):
                    semantic_groups['exit_logic'].append(value)
                else:
                    semantic_groups['general'].append(value)
        
        return dict(semantic_groups)
    
    def _calculate_list_similarity(self, list_a: List[Any], list_b: List[Any]) -> float:
        """Calculate similarity between two lists"""
        if not list_a and not list_b:
            return 1.0
        if not list_a or not list_b:
            return 0.0
        
        # Convert to strings for comparison
        str_a = [str(x) for x in list_a]
        str_b = [str(x) for x in list_b]
        
        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, str_a, str_b)
        return matcher.ratio()
    
    def _find_differences(self, config_a: Dict[str, Any], config_b: Dict[str, Any], 
                         id_a: str, id_b: str) -> List[str]:
        """Find specific differences between configurations"""
        differences = []
        flat_a = self._flatten_config(config_a)
        flat_b = self._flatten_config(config_b)
        
        all_keys = set(flat_a.keys()).union(set(flat_b.keys()))
        
        for key in sorted(all_keys):
            val_a = flat_a.get(key)
            val_b = flat_b.get(key)
            
            if val_a is None:
                differences.append(f"Parameter '{key}' only in {id_b}: {val_b}")
            elif val_b is None:
                differences.append(f"Parameter '{key}' only in {id_a}: {val_a}")
            elif not self._values_equal(val_a, val_b):
                differences.append(f"Parameter '{key}' differs: {id_a}={val_a}, {id_b}={val_b}")
        
        return differences
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration dictionary"""
        flat = {}
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value
        
        return flat
    
    def _flatten_keys(self, config: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get list of all flattened keys"""
        keys = []
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))
            else:
                keys.append(full_key)
        
        return keys
    
    def _values_equal(self, val_a: Any, val_b: Any) -> bool:
        """Check if two values are equal with type tolerance"""
        # Handle numeric comparisons with tolerance
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            return abs(float(val_a) - float(val_b)) < 1e-6
        
        # Handle string comparisons (case insensitive)
        if isinstance(val_a, str) and isinstance(val_b, str):
            return val_a.strip().lower() == val_b.strip().lower()
        
        # Direct comparison for other types
        return val_a == val_b

class DeduplicationEngine:
    """
    Advanced deduplication engine for configuration management
    
    Provides enterprise-grade deduplication with:
    - Multi-level similarity detection
    - Intelligent clustering and grouping
    - Performance optimization for large datasets
    - Configuration merge and consolidation
    - Advanced analytics and reporting
    """
    
    def __init__(self, 
                 registry: Optional[ParameterRegistry] = None,
                 db_path: Optional[str] = None,
                 similarity_threshold: float = 0.95):
        """
        Initialize deduplication engine
        
        Args:
            registry: Parameter registry for semantic analysis
            db_path: Database path for deduplication cache
            similarity_threshold: Threshold for considering configs similar
        """
        self.registry = registry or ParameterRegistry()
        self.similarity_calculator = SimilarityCalculator(self.registry)
        self.similarity_threshold = similarity_threshold
        
        # Database for caching deduplication results
        self.db_path = db_path or "deduplication_cache.db"
        self._init_database()
        
        # Cache for similarity calculations
        self._similarity_cache = {}
        self._lock = threading.Lock()
        
        logger.info(f"DeduplicationEngine initialized with threshold {similarity_threshold}")
    
    def _init_database(self):
        """Initialize SQLite database for caching"""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                config_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                structural_hash TEXT NOT NULL,
                semantic_hash TEXT NOT NULL,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS similarity_cache (
                config_a TEXT,
                config_b TEXT,
                similarity_score REAL,
                content_similarity REAL,
                structural_similarity REAL,
                semantic_similarity REAL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (config_a, config_b)
            );
            
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                group_id TEXT PRIMARY KEY,
                primary_config_id TEXT,
                group_type TEXT,
                similarity_threshold REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS group_members (
                group_id TEXT,
                config_id TEXT,
                similarity_score REAL,
                PRIMARY KEY (group_id, config_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_content_hash ON content_hashes(content_hash);
            CREATE INDEX IF NOT EXISTS idx_structural_hash ON content_hashes(structural_hash);
            CREATE INDEX IF NOT EXISTS idx_semantic_hash ON content_hashes(semantic_hash);
            CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarity_cache(similarity_score);
        """)
        
        conn.commit()
        conn.close()
    
    def analyze_configurations(self, 
                             configurations: Dict[str, Dict[str, Any]],
                             include_similar: bool = True,
                             parallel_processing: bool = True) -> DeduplicationReport:
        """
        Perform comprehensive deduplication analysis
        
        Args:
            configurations: Dict of config_id -> config_data
            include_similar: Include similar (not just exact) duplicates
            parallel_processing: Use parallel processing for large datasets
            
        Returns:
            Comprehensive deduplication report
        """
        start_time = time.time()
        
        logger.info(f"Starting deduplication analysis for {len(configurations)} configurations")
        
        # Calculate hashes for all configurations
        self._calculate_all_hashes(configurations, parallel_processing)
        
        # Find duplicate groups
        duplicate_groups = self._find_duplicate_groups(
            configurations, include_similar, parallel_processing
        )
        
        # Generate comprehensive report
        report = self._generate_report(
            configurations, duplicate_groups, time.time() - start_time
        )
        
        logger.info(f"Deduplication analysis complete: {report.exact_duplicates} exact, "
                   f"{report.similar_configurations} similar, {report.space_saved_percentage:.1f}% space saved")
        
        return report
    
    def _calculate_all_hashes(self, 
                            configurations: Dict[str, Dict[str, Any]], 
                            parallel: bool = True):
        """Calculate and cache hashes for all configurations"""
        
        def calculate_hashes(config_id: str, config_data: Dict[str, Any]):
            content_hash = ContentHasher.calculate_content_hash(config_data)
            structural_hash = ContentHasher.calculate_structural_hash(config_data)
            semantic_hash = ContentHasher.calculate_semantic_hash(config_data, self.registry)
            
            # Estimate file size
            file_size = len(json.dumps(config_data))
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO content_hashes 
                (config_id, content_hash, structural_hash, semantic_hash, file_size)
                VALUES (?, ?, ?, ?, ?)
            """, (config_id, content_hash, structural_hash, semantic_hash, file_size))
            conn.commit()
            conn.close()
        
        if parallel and len(configurations) > 100:
            # Use parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for config_id, config_data in configurations.items():
                    future = executor.submit(calculate_hashes, config_id, config_data)
                    futures.append(future)
                
                # Wait for completion
                for future in futures:
                    future.result()
        else:
            # Sequential processing
            for config_id, config_data in configurations.items():
                calculate_hashes(config_id, config_data)
    
    def _find_duplicate_groups(self, 
                             configurations: Dict[str, Dict[str, Any]], 
                             include_similar: bool,
                             parallel: bool) -> List[DuplicateGroup]:
        """Find groups of duplicate/similar configurations"""
        
        # Group by content hash (exact duplicates)
        exact_groups = self._group_by_hash(configurations, "content")
        
        groups = []
        processed_configs = set()
        
        # Process exact duplicate groups
        for hash_value, config_ids in exact_groups.items():
            if len(config_ids) > 1:
                primary_id = min(config_ids)  # Use lexicographically first as primary
                members = [cid for cid in config_ids if cid != primary_id]
                
                # Calculate saved space
                config_data = configurations[primary_id]
                file_size = len(json.dumps(config_data))
                total_size = file_size * len(config_ids)
                saved_space = file_size * (len(config_ids) - 1)
                
                group = DuplicateGroup(
                    group_id=f"exact_{hash_value[:8]}",
                    primary_config_id=primary_id,
                    member_configs=members,
                    similarity_threshold=1.0,
                    total_size=total_size,
                    saved_space=saved_space,
                    group_type="exact"
                )
                groups.append(group)
                processed_configs.update(config_ids)
        
        # Find similar groups (if requested)
        if include_similar:
            remaining_configs = {
                cid: cdata for cid, cdata in configurations.items() 
                if cid not in processed_configs
            }
            
            similar_groups = self._find_similar_groups(remaining_configs, parallel)
            groups.extend(similar_groups)
        
        # Store groups in database
        self._store_duplicate_groups(groups)
        
        return groups
    
    def _group_by_hash(self, 
                      configurations: Dict[str, Dict[str, Any]], 
                      hash_type: str) -> Dict[str, List[str]]:
        """Group configurations by hash value"""
        
        # Get hashes from database
        conn = sqlite3.connect(self.db_path)
        
        if hash_type == "content":
            hash_column = "content_hash"
        elif hash_type == "structural":
            hash_column = "structural_hash"
        elif hash_type == "semantic":
            hash_column = "semantic_hash"
        else:
            raise ValueError(f"Unknown hash type: {hash_type}")
        
        cursor = conn.execute(f"""
            SELECT config_id, {hash_column} 
            FROM content_hashes 
            WHERE config_id IN ({','.join('?' * len(configurations))})
        """, list(configurations.keys()))
        
        hash_groups = defaultdict(list)
        for config_id, hash_value in cursor.fetchall():
            hash_groups[hash_value].append(config_id)
        
        conn.close()
        
        # Filter groups with multiple members
        return {h: configs for h, configs in hash_groups.items() if len(configs) > 1}
    
    def _find_similar_groups(self, 
                           configurations: Dict[str, Dict[str, Any]], 
                           parallel: bool) -> List[DuplicateGroup]:
        """Find groups of similar (but not identical) configurations"""
        
        config_list = list(configurations.items())
        similar_groups = []
        processed = set()
        
        def calculate_similarities(i: int) -> List[Tuple[str, str, float]]:
            """Calculate similarities for config at index i"""
            similarities = []
            config_id_a, config_data_a = config_list[i]
            
            if config_id_a in processed:
                return similarities
            
            for j in range(i + 1, len(config_list)):
                config_id_b, config_data_b = config_list[j]
                
                if config_id_b in processed:
                    continue
                
                # Check cache first
                cache_key = tuple(sorted([config_id_a, config_id_b]))
                if cache_key in self._similarity_cache:
                    similarity = self._similarity_cache[cache_key]
                else:
                    # Calculate similarity
                    metrics = self.similarity_calculator.calculate_similarity(
                        config_data_a, config_data_b, config_id_a, config_id_b
                    )
                    similarity = metrics.overall_similarity
                    self._similarity_cache[cache_key] = similarity
                
                if similarity >= self.similarity_threshold:
                    similarities.append((config_id_a, config_id_b, similarity))
            
            return similarities
        
        # Calculate similarities
        all_similarities = []
        
        if parallel and len(config_list) > 50:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(len(config_list)):
                    future = executor.submit(calculate_similarities, i)
                    futures.append(future)
                
                for future in futures:
                    all_similarities.extend(future.result())
        else:
            for i in range(len(config_list)):
                all_similarities.extend(calculate_similarities(i))
        
        # Group similar configurations using clustering
        similar_groups.extend(self._cluster_similar_configs(all_similarities, configurations))
        
        return similar_groups
    
    def _cluster_similar_configs(self, 
                               similarities: List[Tuple[str, str, float]], 
                               configurations: Dict[str, Dict[str, Any]]) -> List[DuplicateGroup]:
        """Cluster similar configurations into groups"""
        
        # Build similarity graph
        similarity_graph = defaultdict(list)
        for config_a, config_b, similarity in similarities:
            similarity_graph[config_a].append((config_b, similarity))
            similarity_graph[config_b].append((config_a, similarity))
        
        groups = []
        visited = set()
        
        # Find connected components (clusters)
        for config_id in similarity_graph:
            if config_id in visited:
                continue
            
            # BFS to find all connected configs
            cluster = set()
            queue = [config_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.add(current)
                
                # Add similar configs to queue
                for similar_config, similarity in similarity_graph[current]:
                    if similar_config not in visited and similarity >= self.similarity_threshold:
                        queue.append(similar_config)
            
            # Create group if cluster has multiple members
            if len(cluster) > 1:
                cluster_list = list(cluster)
                primary_id = min(cluster_list)  # Lexicographically first
                members = [cid for cid in cluster_list if cid != primary_id]
                
                # Calculate average similarity and saved space
                avg_similarity = sum(
                    sim for _, _, sim in similarities 
                    if _ in cluster and sim in cluster
                ) / len(similarities) if similarities else self.similarity_threshold
                
                # Estimate space savings (conservative)
                primary_config = configurations[primary_id]
                file_size = len(json.dumps(primary_config))
                total_size = file_size * len(cluster)
                saved_space = int(file_size * (len(cluster) - 1) * avg_similarity)
                
                group = DuplicateGroup(
                    group_id=f"similar_{primary_id[:8]}",
                    primary_config_id=primary_id,
                    member_configs=members,
                    similarity_threshold=avg_similarity,
                    total_size=total_size,
                    saved_space=saved_space,
                    group_type="similar"
                )
                groups.append(group)
        
        return groups
    
    def _store_duplicate_groups(self, groups: List[DuplicateGroup]):
        """Store duplicate groups in database"""
        conn = sqlite3.connect(self.db_path)
        
        for group in groups:
            # Insert group
            conn.execute("""
                INSERT OR REPLACE INTO duplicate_groups 
                (group_id, primary_config_id, group_type, similarity_threshold)
                VALUES (?, ?, ?, ?)
            """, (group.group_id, group.primary_config_id, group.group_type, group.similarity_threshold))
            
            # Insert members
            for member_id in group.member_configs:
                conn.execute("""
                    INSERT OR REPLACE INTO group_members 
                    (group_id, config_id, similarity_score)
                    VALUES (?, ?, ?)
                """, (group.group_id, member_id, group.similarity_threshold))
        
        conn.commit()
        conn.close()
    
    def _generate_report(self, 
                        configurations: Dict[str, Dict[str, Any]], 
                        duplicate_groups: List[DuplicateGroup], 
                        processing_time: float) -> DeduplicationReport:
        """Generate comprehensive deduplication report"""
        
        total_configs = len(configurations)
        
        # Count duplicates
        exact_duplicates = sum(
            len(group.member_configs) 
            for group in duplicate_groups 
            if group.group_type == "exact"
        )
        
        similar_configs = sum(
            len(group.member_configs)
            for group in duplicate_groups
            if group.group_type == "similar"
        )
        
        unique_configs = total_configs - exact_duplicates - similar_configs
        
        # Calculate space savings
        total_original_size = sum(
            len(json.dumps(config_data)) 
            for config_data in configurations.values()
        )
        
        total_saved_space = sum(group.saved_space for group in duplicate_groups)
        total_deduplicated_size = total_original_size - total_saved_space
        space_saved_percentage = (total_saved_space / total_original_size * 100) if total_original_size > 0 else 0
        
        # Strategy breakdown
        strategy_breakdown = defaultdict(lambda: defaultdict(int))
        for config_id, config_data in configurations.items():
            # Extract strategy type from config or ID
            strategy_type = self._extract_strategy_type(config_id, config_data)
            strategy_breakdown[strategy_type]["total"] += 1
            
            # Check if config is in any duplicate group
            for group in duplicate_groups:
                if config_id == group.primary_config_id:
                    strategy_breakdown[strategy_type]["primary"] += 1
                elif config_id in group.member_configs:
                    if group.group_type == "exact":
                        strategy_breakdown[strategy_type]["exact_duplicates"] += 1
                    else:
                        strategy_breakdown[strategy_type]["similar"] += 1
        
        # Similarity distribution
        similarity_distribution = defaultdict(int)
        for group in duplicate_groups:
            similarity_range = self._get_similarity_range(group.similarity_threshold)
            similarity_distribution[similarity_range] += len(group.member_configs)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            duplicate_groups, total_configs, space_saved_percentage
        )
        
        return DeduplicationReport(
            total_configurations=total_configs,
            duplicate_groups=duplicate_groups,
            exact_duplicates=exact_duplicates,
            similar_configurations=similar_configs,
            unique_configurations=unique_configs,
            total_original_size=total_original_size,
            total_deduplicated_size=total_deduplicated_size,
            space_saved=total_saved_space,
            space_saved_percentage=space_saved_percentage,
            processing_time=processing_time,
            strategy_breakdown=dict(strategy_breakdown),
            similarity_distribution=dict(similarity_distribution),
            recommendations=recommendations
        )
    
    def _extract_strategy_type(self, config_id: str, config_data: Dict[str, Any]) -> str:
        """Extract strategy type from configuration"""
        # Try to extract from config_id
        for strategy in ["tbs", "tv", "orb", "oi", "ml", "pos", "market_regime", "option_chain", "indicator", "optimize"]:
            if strategy in config_id.lower():
                return strategy
        
        # Try to extract from config data structure
        if "strategy_type" in config_data:
            return config_data["strategy_type"]
        
        # Fallback to general analysis
        if any(key.startswith("ml_") for key in config_data.keys()):
            return "ml_triple_straddle"
        elif any(key.startswith("indicator_") for key in config_data.keys()):
            return "indicator"
        elif "open_interest" in str(config_data).lower():
            return "oi"
        else:
            return "unknown"
    
    def _get_similarity_range(self, similarity: float) -> str:
        """Get similarity range category"""
        if similarity >= 0.99:
            return "99-100%"
        elif similarity >= 0.95:
            return "95-99%"
        elif similarity >= 0.90:
            return "90-95%"
        elif similarity >= 0.80:
            return "80-90%"
        else:
            return "Below 80%"
    
    def _generate_recommendations(self, 
                                duplicate_groups: List[DuplicateGroup], 
                                total_configs: int, 
                                space_saved_percentage: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if space_saved_percentage > 30:
            recommendations.append(
                f"High deduplication potential: {space_saved_percentage:.1f}% space can be saved. "
                "Consider implementing automatic deduplication."
            )
        elif space_saved_percentage > 10:
            recommendations.append(
                f"Moderate deduplication benefit: {space_saved_percentage:.1f}% space can be saved. "
                "Review duplicate groups for consolidation opportunities."
            )
        
        exact_groups = [g for g in duplicate_groups if g.group_type == "exact"]
        if exact_groups:
            recommendations.append(
                f"Found {len(exact_groups)} exact duplicate groups. "
                "These can be safely deduplicated immediately."
            )
        
        similar_groups = [g for g in duplicate_groups if g.group_type == "similar"]
        if similar_groups:
            recommendations.append(
                f"Found {len(similar_groups)} similar configuration groups. "
                "Review these for potential parameter standardization."
            )
        
        large_groups = [g for g in duplicate_groups if len(g.member_configs) > 5]
        if large_groups:
            recommendations.append(
                f"Found {len(large_groups)} groups with 6+ similar configurations. "
                "Consider creating configuration templates for these patterns."
            )
        
        if total_configs > 1000 and space_saved_percentage < 5:
            recommendations.append(
                "Large configuration set with low duplication detected. "
                "Consider implementing parameter validation to prevent future duplicates."
            )
        
        return recommendations
    
    def get_duplicate_groups(self, 
                           group_type: Optional[str] = None, 
                           min_similarity: float = 0.0) -> List[DuplicateGroup]:
        """Get stored duplicate groups with filtering"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT g.group_id, g.primary_config_id, g.group_type, g.similarity_threshold,
                   GROUP_CONCAT(m.config_id) as members
            FROM duplicate_groups g
            LEFT JOIN group_members m ON g.group_id = m.group_id
            WHERE g.similarity_threshold >= ?
        """
        params = [min_similarity]
        
        if group_type:
            query += " AND g.group_type = ?"
            params.append(group_type)
        
        query += " GROUP BY g.group_id"
        
        cursor = conn.execute(query, params)
        groups = []
        
        for row in cursor.fetchall():
            group_id, primary_id, gtype, similarity, members_str = row
            member_configs = members_str.split(',') if members_str else []
            
            groups.append(DuplicateGroup(
                group_id=group_id,
                primary_config_id=primary_id,
                member_configs=member_configs,
                similarity_threshold=similarity,
                group_type=gtype
            ))
        
        conn.close()
        return groups
    
    def consolidate_duplicates(self, 
                             configurations: Dict[str, Dict[str, Any]], 
                             group_id: str,
                             merge_strategy: str = "primary") -> Dict[str, Any]:
        """
        Consolidate duplicate configurations into a single configuration
        
        Args:
            configurations: Original configurations
            group_id: ID of duplicate group to consolidate
            merge_strategy: Strategy for merging ("primary", "merge_values", "most_common")
            
        Returns:
            Consolidated configuration
        """
        # Get group details
        groups = [g for g in self.get_duplicate_groups() if g.group_id == group_id]
        if not groups:
            raise ValueError(f"Duplicate group {group_id} not found")
        
        group = groups[0]
        all_config_ids = [group.primary_config_id] + group.member_configs
        
        if merge_strategy == "primary":
            # Use primary configuration as-is
            return configurations[group.primary_config_id].copy()
        
        elif merge_strategy == "merge_values":
            # Merge values from all configurations
            merged_config = configurations[group.primary_config_id].copy()
            
            for config_id in group.member_configs:
                config_data = configurations[config_id]
                merged_config = self._merge_configurations(merged_config, config_data)
            
            return merged_config
        
        elif merge_strategy == "most_common":
            # Use most common values for each parameter
            all_configs = [configurations[cid] for cid in all_config_ids]
            return self._merge_by_most_common(all_configs)
        
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    
    def _merge_configurations(self, config_a: Dict[str, Any], config_b: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations, preferring values from config_a"""
        merged = config_a.copy()
        
        for key, value in config_b.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = self._merge_configurations(merged[key], value)
        
        return merged
    
    def _merge_by_most_common(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge configurations by using most common values"""
        merged = {}
        
        # Get all unique keys
        all_keys = set()
        for config in configs:
            all_keys.update(self._flatten_config(config).keys())
        
        # For each key, find most common value
        for key in all_keys:
            values = []
            for config in configs:
                flat_config = self._flatten_config(config)
                if key in flat_config:
                    values.append(flat_config[key])
            
            if values:
                # Use most common value
                value_counts = Counter(str(v) for v in values)
                most_common_str = value_counts.most_common(1)[0][0]
                
                # Find original value with this string representation
                for v in values:
                    if str(v) == most_common_str:
                        self._set_nested_value(merged, key, v)
                        break
        
        return merged
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def export_deduplication_report(self, 
                                  report: DeduplicationReport, 
                                  output_path: str,
                                  format: str = "json") -> str:
        """Export deduplication report to file"""
        output_file = Path(output_path)
        
        if format == "json":
            # Convert to JSON-serializable format
            report_dict = {
                "summary": {
                    "total_configurations": report.total_configurations,
                    "exact_duplicates": report.exact_duplicates,
                    "similar_configurations": report.similar_configurations,
                    "unique_configurations": report.unique_configurations,
                    "space_saved_mb": round(report.space_saved / (1024 * 1024), 2),
                    "space_saved_percentage": round(report.space_saved_percentage, 2),
                    "processing_time": round(report.processing_time, 2)
                },
                "duplicate_groups": [
                    {
                        "group_id": group.group_id,
                        "primary_config": group.primary_config_id,
                        "member_count": len(group.member_configs),
                        "members": group.member_configs,
                        "similarity_threshold": round(group.similarity_threshold, 3),
                        "group_type": group.group_type,
                        "saved_space_mb": round(group.saved_space / (1024 * 1024), 2)
                    }
                    for group in report.duplicate_groups
                ],
                "strategy_breakdown": report.strategy_breakdown,
                "similarity_distribution": report.similarity_distribution,
                "recommendations": report.recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_dict, f, indent=2)
        
        elif format == "csv":
            # Export duplicate groups as CSV
            import csv
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Group ID", "Primary Config", "Member Count", "Group Type", 
                    "Similarity", "Saved Space (MB)"
                ])
                
                for group in report.duplicate_groups:
                    writer.writerow([
                        group.group_id,
                        group.primary_config_id,
                        len(group.member_configs),
                        group.group_type,
                        round(group.similarity_threshold, 3),
                        round(group.saved_space / (1024 * 1024), 2)
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Deduplication report exported to {output_file}")
        return str(output_file)