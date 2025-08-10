"""
Advanced Deduplication System

Enterprise-grade deduplication for configuration management with multi-level
similarity detection, intelligent clustering, and comprehensive analytics.
"""

from .deduplication_engine import (
    DeduplicationEngine,
    ContentHasher,
    SimilarityCalculator,
    SimilarityMetrics,
    DuplicateGroup,
    DeduplicationReport
)

__all__ = [
    'DeduplicationEngine',
    'ContentHasher', 
    'SimilarityCalculator',
    'SimilarityMetrics',
    'DuplicateGroup',
    'DeduplicationReport'
]