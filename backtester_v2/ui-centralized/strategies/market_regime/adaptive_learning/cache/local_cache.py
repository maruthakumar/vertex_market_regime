"""
Local Feature Cache Implementation

TTL-based caching system optimized for iterative development runs with
configurable policies per component and development-friendly management.
"""

import os
import pickle
import time
import logging
import hashlib
import threading
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .. import CacheError

logger = logging.getLogger(__name__)

# Try to import compression libraries
try:
    import zlib
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    data: Any
    created_time: float
    access_time: float
    access_count: int
    size_bytes: int
    ttl_seconds: int
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.created_time) > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access time and count."""
        self.access_time = time.time()
        self.access_count += 1


@dataclass
class CachePolicy:
    """Cache policy configuration."""
    ttl_minutes: int = 15
    max_size_mb: int = 64
    max_entries: int = 1000
    enable_persistence: bool = True
    enable_compression: bool = True
    cleanup_interval_minutes: int = 5
    eviction_policy: str = "lru"  # "lru", "lfu", "ttl"


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    total_size_mb: float
    hit_count: int
    miss_count: int
    eviction_count: int
    cleanup_count: int
    hit_ratio: float
    average_entry_size_kb: float


class LocalFeatureCache:
    """
    Local feature cache with TTL, size limits, and persistence.
    
    Optimized for development workflows with features:
    - Time-based expiration (TTL)
    - Size-based eviction
    - Optional compression
    - Persistent storage across runs
    - Thread-safe operations
    - Development-friendly utilities
    """
    
    def __init__(
        self,
        component_id: str,
        ttl_minutes: int = 15,
        max_size_mb: int = 64,
        max_entries: int = 1000,
        enable_persistence: bool = True,
        enable_compression: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize local feature cache.
        
        Args:
            component_id: Component identifier for cache isolation
            ttl_minutes: Time-to-live in minutes
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
            enable_persistence: Enable persistent storage
            enable_compression: Enable data compression
            cache_dir: Cache directory (auto-created if None)
        """
        self.component_id = component_id
        
        self.policy = CachePolicy(
            ttl_minutes=ttl_minutes,
            max_size_mb=max_size_mb,
            max_entries=max_entries,
            enable_persistence=enable_persistence,
            enable_compression=enable_compression and COMPRESSION_AVAILABLE
        )
        
        # Cache storage
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "hit_count": 0,
            "miss_count": 0,
            "eviction_count": 0,
            "cleanup_count": 0
        }
        
        # Persistence setup
        if cache_dir is None:
            cache_dir = Path.home() / ".adaptive_learning_cache" / component_id
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.entries_dir = self.cache_dir / "entries"
        self.entries_dir.mkdir(exist_ok=True)
        
        # Background cleanup
        self._cleanup_thread = None
        self._cleanup_running = False
        
        # Load persistent cache
        if self.policy.enable_persistence:
            self._load_persistent_cache()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"LocalFeatureCache initialized for {component_id}: TTL={ttl_minutes}min, Size={max_size_mb}MB")
    
    def _compute_cache_key(self, key_data: Union[str, Dict, List, bytes]) -> str:
        """Compute stable cache key from input data."""
        if isinstance(key_data, str):
            return hashlib.sha256(key_data.encode()).hexdigest()[:16]
        elif isinstance(key_data, bytes):
            return hashlib.sha256(key_data).hexdigest()[:16]
        elif isinstance(key_data, (dict, list)):
            # Convert to sorted JSON string for stable hashing
            import json
            json_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        else:
            # Convert to string representation
            str_repr = str(key_data)
            return hashlib.sha256(str_repr.encode()).hexdigest()[:16]
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data in bytes."""
        try:
            # Try pickle serialization for size estimation
            pickled = pickle.dumps(data)
            return len(pickled)
        except Exception:
            # Fallback size estimation
            import sys
            return sys.getsizeof(data)
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.policy.enable_compression:
            return pickle.dumps(data)
        
        try:
            pickled = pickle.dumps(data)
            compressed = zlib.compress(pickled, level=6)  # Balanced compression
            return compressed
        except Exception as e:
            logger.warning(f"Compression failed, storing uncompressed: {str(e)}")
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes, was_compressed: bool = True) -> Any:
        """Decompress data from storage."""
        try:
            if was_compressed and self.policy.enable_compression:
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(compressed_data)
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}")
            raise CacheError(f"Failed to decompress cache data: {str(e)}")
    
    def get(self, key_data: Union[str, Dict, List, bytes]) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key_data: Key data (will be hashed for stable key)
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_key = self._compute_cache_key(key_data)
        
        with self._lock:
            if cache_key not in self._entries:
                self._stats["miss_count"] += 1
                return None
            
            entry = self._entries[cache_key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(cache_key)
                self._stats["miss_count"] += 1
                return None
            
            # Update access info
            entry.touch()
            self._stats["hit_count"] += 1
            
            return entry.data
    
    def put(
        self, 
        key_data: Union[str, Dict, List, bytes], 
        data: Any,
        ttl_override: Optional[int] = None
    ) -> bool:
        """
        Put item in cache.
        
        Args:
            key_data: Key data (will be hashed for stable key)
            data: Data to cache
            ttl_override: Override default TTL in minutes
            
        Returns:
            True if cached successfully
        """
        cache_key = self._compute_cache_key(key_data)
        
        # Estimate size
        data_size = self._estimate_size(data)
        
        # Check if data is too large for cache
        max_size_bytes = self.policy.max_size_mb * 1024 * 1024
        if data_size > max_size_bytes * 0.5:  # Don't cache items larger than 50% of total cache
            logger.warning(f"Data too large for cache: {data_size / (1024*1024):.1f}MB")
            return False
        
        ttl_seconds = (ttl_override or self.policy.ttl_minutes) * 60
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                data=data,
                created_time=time.time(),
                access_time=time.time(),
                access_count=1,
                size_bytes=data_size,
                ttl_seconds=ttl_seconds,
                compressed=self.policy.enable_compression
            )
            
            # Remove existing entry if present
            if cache_key in self._entries:
                self._remove_entry(cache_key)
            
            # Check if we need to make space
            self._ensure_space_available(data_size)
            
            # Add entry
            self._entries[cache_key] = entry
            
            # Persist if enabled
            if self.policy.enable_persistence:
                self._persist_entry(entry)
            
            logger.debug(f"Cached item: {cache_key[:8]} ({data_size / 1024:.1f}KB, TTL={ttl_seconds//60}min)")
            return True
    
    def _ensure_space_available(self, required_bytes: int) -> None:
        """Ensure sufficient space is available for new entry."""
        max_size_bytes = self.policy.max_size_mb * 1024 * 1024
        current_size = sum(entry.size_bytes for entry in self._entries.values())
        
        # Check size limit
        while (current_size + required_bytes > max_size_bytes or 
               len(self._entries) >= self.policy.max_entries):
            
            if not self._entries:
                break
                
            # Select entry for eviction based on policy
            evict_key = self._select_eviction_candidate()
            if evict_key:
                evicted_size = self._entries[evict_key].size_bytes
                self._remove_entry(evict_key)
                current_size -= evicted_size
                self._stats["eviction_count"] += 1
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on policy."""
        if not self._entries:
            return None
        
        if self.policy.eviction_policy == "lru":
            # Least recently used
            return min(self._entries.keys(), key=lambda k: self._entries[k].access_time)
        elif self.policy.eviction_policy == "lfu":
            # Least frequently used
            return min(self._entries.keys(), key=lambda k: self._entries[k].access_count)
        elif self.policy.eviction_policy == "ttl":
            # Closest to expiration
            return min(self._entries.keys(), key=lambda k: self._entries[k].created_time)
        else:
            # Default to LRU
            return min(self._entries.keys(), key=lambda k: self._entries[k].access_time)
    
    def _remove_entry(self, cache_key: str) -> None:
        """Remove entry from cache and persistent storage."""
        if cache_key in self._entries:
            # Remove from memory
            del self._entries[cache_key]
            
            # Remove from persistent storage
            if self.policy.enable_persistence:
                entry_file = self.entries_dir / f"{cache_key}.cache"
                if entry_file.exists():
                    entry_file.unlink()
    
    def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist cache entry to disk."""
        try:
            entry_file = self.entries_dir / f"{entry.key}.cache"
            compressed_data = self._compress_data(entry.data)
            
            # Store metadata and data
            persist_data = {
                "metadata": asdict(entry),
                "data": compressed_data
            }
            
            with open(entry_file, 'wb') as f:
                pickle.dump(persist_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {entry.key}: {str(e)}")
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from disk."""
        if not self.entries_dir.exists():
            return
        
        loaded_count = 0
        expired_count = 0
        
        for entry_file in self.entries_dir.glob("*.cache"):
            try:
                with open(entry_file, 'rb') as f:
                    persist_data = pickle.load(f)
                
                metadata = persist_data["metadata"]
                compressed_data = persist_data["data"]
                
                # Recreate entry
                entry = CacheEntry(**metadata)
                
                # Check if expired
                if entry.is_expired():
                    entry_file.unlink()  # Remove expired entry
                    expired_count += 1
                    continue
                
                # Decompress data
                entry.data = self._decompress_data(compressed_data, entry.compressed)
                
                # Add to cache
                self._entries[entry.key] = entry
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to load cache entry {entry_file}: {str(e)}")
                # Remove corrupted entry
                entry_file.unlink()
        
        logger.info(f"Loaded {loaded_count} cache entries, removed {expired_count} expired entries")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        cleanup_interval = self.policy.cleanup_interval_minutes * 60
        
        while self._cleanup_running:
            try:
                time.sleep(cleanup_interval)
                if self._cleanup_running:
                    self.cleanup_expired()
            except Exception as e:
                logger.error(f"Cleanup thread error: {str(e)}")
                time.sleep(60)  # Back off on error
    
    def cleanup_expired(self) -> int:
        """
        Cleanup expired entries.
        
        Returns:
            Number of entries cleaned up
        """
        cleanup_count = 0
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._entries.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                cleanup_count += 1
            
            if cleanup_count > 0:
                self._stats["cleanup_count"] += cleanup_count
                logger.debug(f"Cleaned up {cleanup_count} expired cache entries")
        
        return cleanup_count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            # Clear memory cache
            self._entries.clear()
            
            # Clear persistent storage
            if self.policy.enable_persistence:
                for entry_file in self.entries_dir.glob("*.cache"):
                    entry_file.unlink()
            
            # Reset stats
            self._stats = {
                "hit_count": 0,
                "miss_count": 0,
                "eviction_count": 0,
                "cleanup_count": 0
            }
        
        logger.info(f"Cache cleared for {self.component_id}")
    
    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            CacheStats object with current statistics
        """
        with self._lock:
            total_entries = len(self._entries)
            total_size_bytes = sum(entry.size_bytes for entry in self._entries.values())
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            total_requests = self._stats["hit_count"] + self._stats["miss_count"]
            hit_ratio = (self._stats["hit_count"] / total_requests) if total_requests > 0 else 0.0
            
            avg_entry_size = (total_size_bytes / total_entries / 1024) if total_entries > 0 else 0.0
            
            return CacheStats(
                total_entries=total_entries,
                total_size_mb=total_size_mb,
                hit_count=self._stats["hit_count"],
                miss_count=self._stats["miss_count"],
                eviction_count=self._stats["eviction_count"],
                cleanup_count=self._stats["cleanup_count"],
                hit_ratio=hit_ratio,
                average_entry_size_kb=avg_entry_size
            )
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information for development."""
        with self._lock:
            stats = self.get_stats()
            
            # Entry details
            entries_info = []
            for key, entry in list(self._entries.items())[:10]:  # Show first 10 entries
                entries_info.append({
                    "key": key[:8] + "...",
                    "size_kb": entry.size_bytes / 1024,
                    "age_minutes": (time.time() - entry.created_time) / 60,
                    "access_count": entry.access_count,
                    "expires_in_minutes": (entry.created_time + entry.ttl_seconds - time.time()) / 60
                })
            
            return {
                "component_id": self.component_id,
                "stats": asdict(stats),
                "policy": asdict(self.policy),
                "sample_entries": entries_info,
                "cache_dir": str(self.cache_dir)
            }
    
    def cleanup(self) -> None:
        """Cleanup cache resources."""
        self._cleanup_running = False
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Final cleanup of expired entries
        self.cleanup_expired()
        
        logger.info(f"Cache cleanup completed for {self.component_id}")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction