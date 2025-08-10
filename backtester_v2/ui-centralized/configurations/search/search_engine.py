"""
Advanced Parameter Search Engine

Enterprise-grade search system for configuration parameters with:
- Full-text search with ranking and relevance scoring
- Fuzzy matching and autocomplete
- Advanced filtering and faceted search
- Semantic search using parameter relationships
- Query optimization and caching
- Real-time indexing and incremental updates
"""

import json
import logging
import re
import sqlite3
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from ..parameter_registry import ParameterRegistry, ParameterDefinition, ParameterType
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """Structured search query"""
    text: str = ""
    strategy_types: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    parameter_types: List[ParameterType] = field(default_factory=list)
    value_filters: Dict[str, Any] = field(default_factory=dict)
    range_filters: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    fuzzy_matching: bool = True
    max_results: int = 100
    include_related: bool = False
    sort_by: str = "relevance"  # relevance, name, category, strategy_type
    sort_order: str = "desc"  # asc, desc

@dataclass
class SearchResult:
    """Individual search result"""
    parameter_id: str
    parameter_definition: ParameterDefinition
    relevance_score: float
    match_type: str  # exact, fuzzy, semantic, partial
    matched_fields: List[str] = field(default_factory=list)
    highlights: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResponse:
    """Complete search response"""
    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    execution_time: float
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    did_you_mean: Optional[str] = None
    related_parameters: List[str] = field(default_factory=list)

@dataclass
class SearchIndex:
    """Search index entry"""
    parameter_id: str
    strategy_type: str
    category: str
    name: str
    description: str
    tags: List[str]
    searchable_text: str
    data_type: ParameterType
    default_value: Any
    indexed_at: datetime = field(default_factory=datetime.now)

class TextProcessor:
    """Advanced text processing for search"""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into searchable terms"""
        if not text:
            return []
        
        # Convert to lowercase and split on common delimiters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    @staticmethod
    def generate_ngrams(tokens: List[str], n: int = 2) -> List[str]:
        """Generate n-grams from tokens"""
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase, remove extra spaces
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        
        return ' '.join(tokens)
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        normalized1 = TextProcessor.normalize_text(text1)
        normalized2 = TextProcessor.normalize_text(text2)
        
        return SequenceMatcher(None, normalized1, normalized2).ratio()
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text"""
        tokens = TextProcessor.tokenize(text)
        
        # Filter by length and remove numbers-only tokens
        keywords = []
        for token in tokens:
            if len(token) >= min_length and not token.isdigit():
                keywords.append(token)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords

class FuzzyMatcher:
    """Fuzzy matching for search queries"""
    
    @staticmethod
    def fuzzy_match(query: str, text: str, threshold: float = 0.6) -> bool:
        """Check if query fuzzy matches text"""
        if not query or not text:
            return False
        
        similarity = TextProcessor.calculate_text_similarity(query, text)
        return similarity >= threshold
    
    @staticmethod
    def fuzzy_score(query: str, text: str) -> float:
        """Calculate fuzzy match score"""
        return TextProcessor.calculate_text_similarity(query, text)
    
    @staticmethod
    def find_best_matches(query: str, candidates: List[str], max_matches: int = 5) -> List[Tuple[str, float]]:
        """Find best fuzzy matches from candidates"""
        matches = []
        
        for candidate in candidates:
            score = FuzzyMatcher.fuzzy_score(query, candidate)
            if score > 0.3:  # Minimum threshold
                matches.append((candidate, score))
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_matches]

class SearchIndexBuilder:
    """Builds and maintains search index"""
    
    def __init__(self, registry: ParameterRegistry, db_path: str):
        self.registry = registry
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize search index database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS search_index (
                parameter_id TEXT PRIMARY KEY,
                strategy_type TEXT NOT NULL,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                searchable_text TEXT NOT NULL,
                data_type TEXT NOT NULL,
                default_value TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS search_terms (
                term TEXT PRIMARY KEY,
                parameter_ids TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS parameter_relationships (
                parameter_a TEXT,
                parameter_b TEXT,
                relationship_type TEXT,
                strength REAL DEFAULT 1.0,
                PRIMARY KEY (parameter_a, parameter_b)
            );
            
            CREATE TABLE IF NOT EXISTS search_analytics (
                query_text TEXT,
                result_count INTEGER,
                execution_time REAL,
                clicked_result TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_searchable_text ON search_index(searchable_text);
            CREATE INDEX IF NOT EXISTS idx_strategy_type ON search_index(strategy_type);
            CREATE INDEX IF NOT EXISTS idx_category ON search_index(category);
            CREATE INDEX IF NOT EXISTS idx_data_type ON search_index(data_type);
            CREATE INDEX IF NOT EXISTS idx_search_terms ON search_terms(term);
            
            -- Full-text search virtual table
            CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                parameter_id,
                strategy_type,
                category, 
                name,
                description,
                tags,
                searchable_text,
                content='search_index',
                content_rowid='rowid'
            );
            
            -- Triggers to keep FTS table in sync
            CREATE TRIGGER IF NOT EXISTS search_index_ai AFTER INSERT ON search_index BEGIN
                INSERT INTO search_fts(rowid, parameter_id, strategy_type, category, name, description, tags, searchable_text)
                VALUES (NEW.rowid, NEW.parameter_id, NEW.strategy_type, NEW.category, NEW.name, NEW.description, NEW.tags, NEW.searchable_text);
            END;
            
            CREATE TRIGGER IF NOT EXISTS search_index_ad AFTER DELETE ON search_index BEGIN
                INSERT INTO search_fts(search_fts, rowid, parameter_id, strategy_type, category, name, description, tags, searchable_text)
                VALUES ('delete', OLD.rowid, OLD.parameter_id, OLD.strategy_type, OLD.category, OLD.name, OLD.description, OLD.tags, OLD.searchable_text);
            END;
            
            CREATE TRIGGER IF NOT EXISTS search_index_au AFTER UPDATE ON search_index BEGIN
                INSERT INTO search_fts(search_fts, rowid, parameter_id, strategy_type, category, name, description, tags, searchable_text)
                VALUES ('delete', OLD.rowid, OLD.parameter_id, OLD.strategy_type, OLD.category, OLD.name, OLD.description, OLD.tags, OLD.searchable_text);
                INSERT INTO search_fts(rowid, parameter_id, strategy_type, category, name, description, tags, searchable_text)
                VALUES (NEW.rowid, NEW.parameter_id, NEW.strategy_type, NEW.category, NEW.name, NEW.description, NEW.tags, NEW.searchable_text);
            END;
        """)
        
        conn.commit()
        conn.close()
    
    def build_index(self, force_rebuild: bool = False) -> int:
        """Build or rebuild the search index"""
        start_time = time.time()
        
        conn = sqlite3.connect(self.db_path)
        
        if force_rebuild:
            # Clear existing index
            conn.execute("DELETE FROM search_index")
            conn.execute("DELETE FROM search_terms")
            conn.commit()
        
        # Get all parameters from registry
        all_parameters = []
        strategy_types = self.registry.list_strategy_types()
        
        for strategy_type in strategy_types:
            parameters = self.registry.get_parameters(strategy_type)
            all_parameters.extend(parameters)
        
        indexed_count = 0
        
        for param in all_parameters:
            try:
                # Check if already indexed (unless force rebuild)
                if not force_rebuild:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM search_index WHERE parameter_id = ?",
                        (param.parameter_id,)
                    )
                    if cursor.fetchone()[0] > 0:
                        continue
                
                # Build searchable text
                searchable_text = self._build_searchable_text(param)
                
                # Extract tags
                tags = self._extract_tags(param)
                
                # Insert into index
                conn.execute("""
                    INSERT OR REPLACE INTO search_index 
                    (parameter_id, strategy_type, category, name, description, tags, 
                     searchable_text, data_type, default_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    param.parameter_id,
                    param.strategy_type,
                    param.category,
                    param.name,
                    getattr(param, 'description', ''),
                    json.dumps(tags),
                    searchable_text,
                    param.data_type.value,
                    json.dumps(param.default_value) if param.default_value is not None else None
                ))
                
                # Update search terms
                self._update_search_terms(conn, param.parameter_id, searchable_text)
                
                indexed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to index parameter {param.parameter_id}: {e}")
        
        # Build parameter relationships
        self._build_relationships(conn, all_parameters)
        
        conn.commit()
        conn.close()
        
        build_time = time.time() - start_time
        logger.info(f"Search index built: {indexed_count} parameters indexed in {build_time:.2f}s")
        
        return indexed_count
    
    def _build_searchable_text(self, param: ParameterDefinition) -> str:
        """Build searchable text for parameter"""
        text_parts = [
            param.name,
            param.category,
            param.strategy_type,
            param.data_type.value
        ]
        
        # Add description if available
        if hasattr(param, 'description') and param.description:
            text_parts.append(param.description)
        
        # Add UI hints if available
        if param.ui_hints:
            if param.ui_hints.label:
                text_parts.append(param.ui_hints.label)
            if param.ui_hints.help_text:
                text_parts.append(param.ui_hints.help_text)
            if param.ui_hints.placeholder:
                text_parts.append(param.ui_hints.placeholder)
        
        # Add validation rule descriptions
        for rule in param.validation_rules:
            if rule.error_message:
                text_parts.append(rule.error_message)
        
        # Clean and combine
        clean_parts = [part.strip() for part in text_parts if part and part.strip()]
        return ' '.join(clean_parts)
    
    def _extract_tags(self, param: ParameterDefinition) -> List[str]:
        """Extract tags from parameter definition"""
        tags = []
        
        # Add category-based tags
        tags.append(f"category:{param.category}")
        tags.append(f"type:{param.data_type.value}")
        tags.append(f"strategy:{param.strategy_type}")
        
        # Add semantic tags based on parameter name
        name_lower = param.name.lower()
        
        if any(term in name_lower for term in ['risk', 'stop', 'loss']):
            tags.append('risk-management')
        if any(term in name_lower for term in ['entry', 'signal', 'trigger']):
            tags.append('entry-logic')
        if any(term in name_lower for term in ['exit', 'target', 'profit']):
            tags.append('exit-logic')
        if any(term in name_lower for term in ['portfolio', 'capital', 'position']):
            tags.append('portfolio-management')
        if any(term in name_lower for term in ['indicator', 'ma', 'rsi', 'macd']):
            tags.append('technical-indicators')
        if any(term in name_lower for term in ['ml', 'model', 'prediction']):
            tags.append('machine-learning')
        
        # Add advanced tag if marked as advanced
        if param.ui_hints and param.ui_hints.advanced:
            tags.append('advanced')
        
        return tags
    
    def _update_search_terms(self, conn: sqlite3.Connection, parameter_id: str, searchable_text: str):
        """Update search terms index"""
        # Extract terms
        terms = TextProcessor.tokenize(searchable_text)
        terms.extend(TextProcessor.generate_ngrams(terms, 2))
        
        for term in set(terms):  # Remove duplicates
            # Check if term exists
            cursor = conn.execute("SELECT parameter_ids FROM search_terms WHERE term = ?", (term,))
            row = cursor.fetchone()
            
            if row:
                # Update existing term
                existing_ids = set(row[0].split(','))
                existing_ids.add(parameter_id)
                new_ids = ','.join(sorted(existing_ids))
                
                conn.execute(
                    "UPDATE search_terms SET parameter_ids = ?, frequency = frequency + 1 WHERE term = ?",
                    (new_ids, term)
                )
            else:
                # Insert new term
                conn.execute(
                    "INSERT INTO search_terms (term, parameter_ids) VALUES (?, ?)",
                    (term, parameter_id)
                )
    
    def _build_relationships(self, conn: sqlite3.Connection, parameters: List[ParameterDefinition]):
        """Build parameter relationships for semantic search"""
        # Clear existing relationships
        conn.execute("DELETE FROM parameter_relationships")
        
        # Group parameters by category and strategy
        category_groups = defaultdict(list)
        strategy_groups = defaultdict(list)
        
        for param in parameters:
            category_groups[param.category].append(param.parameter_id)
            strategy_groups[param.strategy_type].append(param.parameter_id)
        
        # Create relationships within categories
        for category, param_ids in category_groups.items():
            for i, param_a in enumerate(param_ids):
                for param_b in param_ids[i + 1:]:
                    conn.execute("""
                        INSERT INTO parameter_relationships 
                        (parameter_a, parameter_b, relationship_type, strength)
                        VALUES (?, ?, ?, ?)
                    """, (param_a, param_b, "category", 0.8))
        
        # Create relationships within strategies (weaker)
        for strategy, param_ids in strategy_groups.items():
            for i, param_a in enumerate(param_ids):
                for param_b in param_ids[i + 1:]:
                    # Only add if not already related by category
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM parameter_relationships 
                        WHERE (parameter_a = ? AND parameter_b = ?) 
                           OR (parameter_a = ? AND parameter_b = ?)
                    """, (param_a, param_b, param_b, param_a))
                    
                    if cursor.fetchone()[0] == 0:
                        conn.execute("""
                            INSERT INTO parameter_relationships 
                            (parameter_a, parameter_b, relationship_type, strength)
                            VALUES (?, ?, ?, ?)
                        """, (param_a, param_b, "strategy", 0.5))
        
        # Create semantic relationships based on name similarity
        for i, param_a in enumerate(parameters):
            for param_b in parameters[i + 1:]:
                name_similarity = TextProcessor.calculate_text_similarity(
                    param_a.name, param_b.name
                )
                
                if name_similarity > 0.7:  # High name similarity
                    conn.execute("""
                        INSERT OR IGNORE INTO parameter_relationships 
                        (parameter_a, parameter_b, relationship_type, strength)
                        VALUES (?, ?, ?, ?)
                    """, (param_a.parameter_id, param_b.parameter_id, "semantic", name_similarity))
        
        conn.commit()

class ParameterSearchEngine:
    """
    Advanced parameter search engine
    
    Provides enterprise-grade search capabilities for configuration parameters:
    - Full-text search with relevance ranking
    - Fuzzy matching and autocomplete
    - Advanced filtering and faceted search
    - Semantic search using parameter relationships
    - Query optimization and caching
    - Real-time indexing and analytics
    """
    
    def __init__(self, 
                 registry: Optional[ParameterRegistry] = None,
                 db_path: Optional[str] = None,
                 auto_index: bool = True):
        """
        Initialize search engine
        
        Args:
            registry: Parameter registry instance
            db_path: Database path for search index
            auto_index: Automatically build index on initialization
        """
        self.registry = registry or ParameterRegistry()
        self.db_path = db_path or "parameter_search.db"
        
        # Initialize components
        self.index_builder = SearchIndexBuilder(self.registry, self.db_path)
        self.text_processor = TextProcessor()
        self.fuzzy_matcher = FuzzyMatcher()
        
        # Query cache
        self._query_cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 1000
        
        if auto_index:
            # Build index if empty
            if self._get_index_count() == 0:
                self.build_index()
        
        logger.info("ParameterSearchEngine initialized")
    
    def _get_index_count(self) -> int:
        """Get number of indexed parameters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM search_index")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def build_index(self, force_rebuild: bool = False) -> int:
        """Build search index"""
        return self.index_builder.build_index(force_rebuild)
    
    def search(self, query: Union[str, SearchQuery]) -> SearchResponse:
        """
        Perform parameter search
        
        Args:
            query: Search query (string or structured query object)
            
        Returns:
            Search response with results and metadata
        """
        start_time = time.time()
        
        # Parse query
        if isinstance(query, str):
            search_query = SearchQuery(text=query)
        else:
            search_query = query
        
        # Check cache
        cache_key = self._get_cache_key(search_query)
        if cache_key in self._query_cache:
            cached_response = self._query_cache[cache_key]
            logger.debug(f"Cache hit for query: {search_query.text}")
            return cached_response
        
        # Execute search
        results = self._execute_search(search_query)
        
        # Calculate facets
        facets = self._calculate_facets(results)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(search_query, results)
        
        # Find related parameters
        related_parameters = self._find_related_parameters(results)
        
        # Build response
        response = SearchResponse(
            query=search_query,
            results=results[:search_query.max_results],
            total_results=len(results),
            execution_time=time.time() - start_time,
            facets=facets,
            suggestions=suggestions,
            related_parameters=related_parameters
        )
        
        # Add spell correction suggestion
        if not results and search_query.text:
            response.did_you_mean = self._suggest_spelling_correction(search_query.text)
        
        # Cache response
        self._cache_response(cache_key, response)
        
        # Log analytics
        self._log_search_analytics(search_query, response)
        
        return response
    
    def _execute_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute the actual search"""
        results = []
        
        conn = sqlite3.connect(self.db_path)
        
        # Build SQL query
        sql_parts = []
        params = []
        
        if query.text:
            # Full-text search
            fts_results = self._full_text_search(conn, query.text)
            results.extend(fts_results)
            
            # Fuzzy matching if enabled and no FTS results
            if query.fuzzy_matching and len(fts_results) < 5:
                fuzzy_results = self._fuzzy_search(conn, query.text)
                results.extend(fuzzy_results)
        else:
            # No text query - get all parameters with filters
            sql_parts.append("SELECT * FROM search_index WHERE 1=1")
        
        # Apply filters
        if query.strategy_types:
            placeholders = ','.join('?' * len(query.strategy_types))
            sql_parts.append(f"AND strategy_type IN ({placeholders})")
            params.extend(query.strategy_types)
        
        if query.categories:
            placeholders = ','.join('?' * len(query.categories))
            sql_parts.append(f"AND category IN ({placeholders})")
            params.extend(query.categories)
        
        if query.parameter_types:
            type_values = [pt.value for pt in query.parameter_types]
            placeholders = ','.join('?' * len(type_values))
            sql_parts.append(f"AND data_type IN ({placeholders})")
            params.extend(type_values)
        
        # Execute filtered query if no text search
        if not query.text and sql_parts:
            sql = ' '.join(sql_parts)
            cursor = conn.execute(sql, params)
            
            for row in cursor.fetchall():
                param_def = self._row_to_parameter_definition(row)
                result = SearchResult(
                    parameter_id=row[0],
                    parameter_definition=param_def,
                    relevance_score=0.5,  # Default score for filtered results
                    match_type="filter"
                )
                results.append(result)
        
        conn.close()
        
        # Apply additional filters
        results = self._apply_value_filters(results, query)
        results = self._apply_range_filters(results, query)
        
        # Sort results
        results = self._sort_results(results, query.sort_by, query.sort_order)
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for result in results:
            if result.parameter_id not in seen:
                seen.add(result.parameter_id)
                unique_results.append(result)
        
        return unique_results
    
    def _full_text_search(self, conn: sqlite3.Connection, query_text: str) -> List[SearchResult]:
        """Perform full-text search"""
        results = []
        
        # Prepare FTS query
        fts_query = self._prepare_fts_query(query_text)
        
        try:
            cursor = conn.execute("""
                SELECT s.*, rank FROM search_fts 
                INNER JOIN search_index s ON search_fts.parameter_id = s.parameter_id
                WHERE search_fts MATCH ?
                ORDER BY rank
            """, (fts_query,))
            
            for row in cursor.fetchall():
                param_def = self._row_to_parameter_definition(row[:-1])  # Exclude rank
                
                # Calculate relevance score
                relevance = self._calculate_relevance_score(query_text, row, "fts")
                
                # Find highlights
                highlights = self._find_highlights(query_text, row)
                
                result = SearchResult(
                    parameter_id=row[0],
                    parameter_definition=param_def,
                    relevance_score=relevance,
                    match_type="exact",
                    matched_fields=list(highlights.keys()),
                    highlights=highlights
                )
                results.append(result)
                
        except Exception as e:
            logger.warning(f"FTS search failed: {e}")
        
        return results
    
    def _prepare_fts_query(self, query_text: str) -> str:
        """Prepare query for FTS5"""
        # Clean and tokenize
        tokens = TextProcessor.tokenize(query_text)
        
        if not tokens:
            return query_text
        
        # Build FTS query with phrase and proximity operators
        if len(tokens) == 1:
            return tokens[0]
        elif len(tokens) <= 3:
            # Use phrase search for short queries
            return f'"{" ".join(tokens)}"'
        else:
            # Use AND for longer queries
            return ' AND '.join(tokens)
    
    def _fuzzy_search(self, conn: sqlite3.Connection, query_text: str) -> List[SearchResult]:
        """Perform fuzzy search"""
        results = []
        
        # Get all searchable text for fuzzy matching
        cursor = conn.execute("SELECT * FROM search_index")
        
        for row in cursor.fetchall():
            searchable_text = row[7]  # searchable_text column
            
            # Calculate fuzzy similarity
            similarity = FuzzyMatcher.fuzzy_score(query_text, searchable_text)
            
            if similarity > 0.3:  # Minimum threshold
                param_def = self._row_to_parameter_definition(row)
                
                result = SearchResult(
                    parameter_id=row[0],
                    parameter_definition=param_def,
                    relevance_score=similarity,
                    match_type="fuzzy"
                )
                results.append(result)
        
        return results
    
    def _row_to_parameter_definition(self, row) -> ParameterDefinition:
        """Convert database row to ParameterDefinition"""
        # This is a simplified version - in practice you'd reconstruct the full object
        from ..parameter_registry.models import ParameterDefinition, UIHints
        
        return ParameterDefinition(
            parameter_id=row[0],
            strategy_type=row[1],
            category=row[2],
            name=row[3],
            data_type=ParameterType(row[7]),
            default_value=json.loads(row[8]) if row[8] else None,
            validation_rules=[],
            ui_hints=UIHints()
        )
    
    def _calculate_relevance_score(self, query_text: str, row, match_type: str) -> float:
        """Calculate relevance score for search result"""
        base_score = 0.5
        
        query_tokens = TextProcessor.tokenize(query_text.lower())
        
        # Field-specific scoring
        field_weights = {
            'name': 2.0,
            'category': 1.5,
            'description': 1.0,
            'searchable_text': 0.8
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for i, field_name in enumerate(['parameter_id', 'strategy_type', 'category', 'name']):
            if i < len(row):
                field_value = str(row[i]).lower()
                field_weight = field_weights.get(field_name, 0.5)
                
                # Calculate field score
                field_score = 0.0
                for token in query_tokens:
                    if token in field_value:
                        if field_value == token:
                            field_score += 1.0  # Exact match
                        elif field_value.startswith(token):
                            field_score += 0.8  # Prefix match
                        else:
                            field_score += 0.6  # Contains match
                
                total_score += field_score * field_weight
                total_weight += field_weight
        
        # Normalize score
        if total_weight > 0:
            base_score = min(1.0, total_score / total_weight)
        
        # Boost for match type
        if match_type == "exact":
            base_score *= 1.2
        elif match_type == "fuzzy":
            base_score *= 0.8
        
        return min(1.0, base_score)
    
    def _find_highlights(self, query_text: str, row) -> Dict[str, str]:
        """Find text highlights in search results"""
        highlights = {}
        query_tokens = TextProcessor.tokenize(query_text.lower())
        
        field_names = ['name', 'category', 'description']
        field_indices = [3, 2, 4]  # Corresponding indices in row
        
        for field_name, field_index in zip(field_names, field_indices):
            if field_index < len(row) and row[field_index]:
                field_value = str(row[field_index])
                highlighted = self._highlight_text(field_value, query_tokens)
                if highlighted != field_value:
                    highlights[field_name] = highlighted
        
        return highlights
    
    def _highlight_text(self, text: str, tokens: List[str]) -> str:
        """Add highlights to text"""
        if not text or not tokens:
            return text
        
        highlighted = text
        for token in tokens:
            # Case-insensitive replacement with highlighting
            pattern = re.compile(re.escape(token), re.IGNORECASE)
            highlighted = pattern.sub(f'<mark>{token}</mark>', highlighted)
        
        return highlighted
    
    def _apply_value_filters(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Apply value-based filters"""
        if not query.value_filters:
            return results
        
        filtered_results = []
        for result in results:
            include_result = True
            
            for field, expected_value in query.value_filters.items():
                actual_value = getattr(result.parameter_definition, field, None)
                
                if actual_value != expected_value:
                    include_result = False
                    break
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_range_filters(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Apply range-based filters"""
        if not query.range_filters:
            return results
        
        filtered_results = []
        for result in results:
            include_result = True
            
            for field, (min_val, max_val) in query.range_filters.items():
                actual_value = getattr(result.parameter_definition, field, None)
                
                if actual_value is not None:
                    try:
                        numeric_value = float(actual_value)
                        if not (min_val <= numeric_value <= max_val):
                            include_result = False
                            break
                    except (ValueError, TypeError):
                        include_result = False
                        break
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def _sort_results(self, results: List[SearchResult], sort_by: str, sort_order: str) -> List[SearchResult]:
        """Sort search results"""
        reverse = sort_order.lower() == "desc"
        
        if sort_by == "relevance":
            return sorted(results, key=lambda r: r.relevance_score, reverse=reverse)
        elif sort_by == "name":
            return sorted(results, key=lambda r: r.parameter_definition.name, reverse=reverse)
        elif sort_by == "category":
            return sorted(results, key=lambda r: r.parameter_definition.category, reverse=reverse)
        elif sort_by == "strategy_type":
            return sorted(results, key=lambda r: r.parameter_definition.strategy_type, reverse=reverse)
        else:
            return results
    
    def _calculate_facets(self, results: List[SearchResult]) -> Dict[str, Dict[str, int]]:
        """Calculate search facets for filtering"""
        facets = {
            "strategy_types": defaultdict(int),
            "categories": defaultdict(int),
            "parameter_types": defaultdict(int),
            "match_types": defaultdict(int)
        }
        
        for result in results:
            facets["strategy_types"][result.parameter_definition.strategy_type] += 1
            facets["categories"][result.parameter_definition.category] += 1
            facets["parameter_types"][result.parameter_definition.data_type.value] += 1
            facets["match_types"][result.match_type] += 1
        
        # Convert to regular dict and sort by count
        sorted_facets = {}
        for facet_name, facet_data in facets.items():
            sorted_facets[facet_name] = dict(
                sorted(facet_data.items(), key=lambda x: x[1], reverse=True)
            )
        
        return sorted_facets
    
    def _generate_suggestions(self, query: SearchQuery, results: List[SearchResult]) -> List[str]:
        """Generate search suggestions"""
        suggestions = []
        
        if query.text and len(results) < 5:
            # Get related terms from search_terms table
            conn = sqlite3.connect(self.db_path)
            
            # Find similar terms
            query_tokens = TextProcessor.tokenize(query.text)
            for token in query_tokens:
                cursor = conn.execute("""
                    SELECT term FROM search_terms 
                    WHERE term LIKE ? 
                    ORDER BY frequency DESC 
                    LIMIT 5
                """, (f"%{token}%",))
                
                for row in cursor.fetchall():
                    term = row[0]
                    if term not in suggestions and term != token:
                        suggestions.append(term)
            
            conn.close()
        
        return suggestions[:10]  # Limit suggestions
    
    def _find_related_parameters(self, results: List[SearchResult]) -> List[str]:
        """Find parameters related to search results"""
        if not results:
            return []
        
        related = set()
        conn = sqlite3.connect(self.db_path)
        
        # Get related parameters from relationships table
        for result in results[:5]:  # Limit to top 5 results
            cursor = conn.execute("""
                SELECT parameter_b, strength FROM parameter_relationships 
                WHERE parameter_a = ? 
                ORDER BY strength DESC 
                LIMIT 3
            """, (result.parameter_id,))
            
            for row in cursor.fetchall():
                related.add(row[0])
            
            cursor = conn.execute("""
                SELECT parameter_a, strength FROM parameter_relationships 
                WHERE parameter_b = ? 
                ORDER BY strength DESC 
                LIMIT 3
            """, (result.parameter_id,))
            
            for row in cursor.fetchall():
                related.add(row[0])
        
        conn.close()
        
        # Remove parameters already in results
        result_ids = {r.parameter_id for r in results}
        related = related - result_ids
        
        return list(related)[:10]
    
    def _suggest_spelling_correction(self, query_text: str) -> Optional[str]:
        """Suggest spelling corrections for query"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all terms from search index
        cursor = conn.execute("SELECT DISTINCT term FROM search_terms ORDER BY frequency DESC LIMIT 1000")
        all_terms = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Find best fuzzy matches
        matches = FuzzyMatcher.find_best_matches(query_text, all_terms, max_matches=1)
        
        if matches and matches[0][1] > 0.5:  # Good similarity score
            return matches[0][0]
        
        return None
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for query"""
        # Create a hash of the query parameters
        query_dict = {
            'text': query.text,
            'strategy_types': sorted(query.strategy_types),
            'categories': sorted(query.categories),
            'parameter_types': sorted([pt.value for pt in query.parameter_types]),
            'fuzzy_matching': query.fuzzy_matching,
            'max_results': query.max_results,
            'sort_by': query.sort_by,
            'sort_order': query.sort_order
        }
        
        import hashlib
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _cache_response(self, cache_key: str, response: SearchResponse):
        """Cache search response"""
        with self._cache_lock:
            # Remove oldest entries if cache is full
            if len(self._query_cache) >= self._max_cache_size:
                # Remove 20% of oldest entries
                oldest_keys = list(self._query_cache.keys())[:int(self._max_cache_size * 0.2)]
                for key in oldest_keys:
                    del self._query_cache[key]
            
            self._query_cache[cache_key] = response
    
    def _log_search_analytics(self, query: SearchQuery, response: SearchResponse):
        """Log search analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO search_analytics (query_text, result_count, execution_time)
                VALUES (?, ?, ?)
            """, (query.text, response.total_results, response.execution_time))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to log search analytics: {e}")
    
    def autocomplete(self, partial_query: str, max_suggestions: int = 10) -> List[str]:
        """Provide autocomplete suggestions"""
        if len(partial_query) < 2:
            return []
        
        suggestions = []
        conn = sqlite3.connect(self.db_path)
        
        # Search in indexed terms
        cursor = conn.execute("""
            SELECT term FROM search_terms 
            WHERE term LIKE ? 
            ORDER BY frequency DESC 
            LIMIT ?
        """, (f"{partial_query}%", max_suggestions))
        
        suggestions.extend([row[0] for row in cursor.fetchall()])
        
        # Search in parameter names
        if len(suggestions) < max_suggestions:
            cursor = conn.execute("""
                SELECT DISTINCT name FROM search_index 
                WHERE name LIKE ? 
                ORDER BY name 
                LIMIT ?
            """, (f"%{partial_query}%", max_suggestions - len(suggestions)))
            
            suggestions.extend([row[0] for row in cursor.fetchall()])
        
        conn.close()
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:max_suggestions]
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and statistics"""
        conn = sqlite3.connect(self.db_path)
        
        analytics = {}
        
        # Query statistics
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_queries,
                AVG(execution_time) as avg_execution_time,
                AVG(result_count) as avg_result_count
            FROM search_analytics
        """)
        row = cursor.fetchone()
        analytics['query_stats'] = {
            'total_queries': row[0],
            'avg_execution_time': round(row[1] or 0, 3),
            'avg_result_count': round(row[2] or 0, 1)
        }
        
        # Top queries
        cursor = conn.execute("""
            SELECT query_text, COUNT(*) as frequency
            FROM search_analytics
            WHERE query_text IS NOT NULL AND query_text != ''
            GROUP BY query_text
            ORDER BY frequency DESC
            LIMIT 10
        """)
        analytics['top_queries'] = [
            {'query': row[0], 'frequency': row[1]} 
            for row in cursor.fetchall()
        ]
        
        # Index statistics
        cursor = conn.execute("SELECT COUNT(*) FROM search_index")
        total_indexed = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM search_terms")
        total_terms = cursor.fetchone()[0]
        
        analytics['index_stats'] = {
            'total_parameters': total_indexed,
            'total_search_terms': total_terms,
            'cache_size': len(self._query_cache)
        }
        
        conn.close()
        
        return analytics
    
    def clear_cache(self):
        """Clear search cache"""
        with self._cache_lock:
            self._query_cache.clear()
        logger.info("Search cache cleared")
    
    def optimize_index(self):
        """Optimize search index for better performance"""
        conn = sqlite3.connect(self.db_path)
        
        # Rebuild FTS index
        conn.execute("INSERT INTO search_fts(search_fts) VALUES('rebuild')")
        
        # Analyze tables for query optimization
        conn.execute("ANALYZE")
        
        # Vacuum database
        conn.execute("VACUUM")
        
        conn.commit()
        conn.close()
        
        logger.info("Search index optimized")