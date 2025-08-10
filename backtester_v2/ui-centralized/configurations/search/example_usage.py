#!/usr/bin/env python3
"""
Advanced Parameter Search Engine - Example Usage

Demonstrates comprehensive search capabilities including full-text search,
fuzzy matching, advanced filtering, and analytics.
"""

import logging
import sys
from pathlib import Path

# Add configurations to path
sys.path.append(str(Path(__file__).parent.parent))

from search import ParameterSearchEngine, SearchQuery
from parameter_registry import ParameterRegistry, ParameterType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_search_capabilities():
    """Demonstrate various search capabilities"""
    
    print("🔍 Advanced Parameter Search Engine Demo")
    print("=" * 60)
    
    # Initialize components
    registry = ParameterRegistry()
    search_engine = ParameterSearchEngine(registry, auto_index=True)
    
    print(f"\n📊 Search Index Statistics:")
    analytics = search_engine.get_search_analytics()
    index_stats = analytics['index_stats']
    print(f"   - Total Parameters: {index_stats['total_parameters']}")
    print(f"   - Search Terms: {index_stats['total_search_terms']}")
    print(f"   - Cache Size: {index_stats['cache_size']}")
    
    # Example 1: Simple text search
    print("\n1. Simple Text Search")
    print("-" * 30)
    
    response = search_engine.search("capital")
    print(f"Query: 'capital'")
    print(f"Results: {response.total_results} parameters found")
    print(f"Execution time: {response.execution_time:.3f}s")
    
    for i, result in enumerate(response.results[:3]):
        print(f"   {i+1}. {result.parameter_definition.name} "
              f"({result.parameter_definition.strategy_type}/{result.parameter_definition.category})")
        print(f"      Relevance: {result.relevance_score:.3f}, Match: {result.match_type}")
        if result.highlights:
            print(f"      Highlights: {result.highlights}")
    
    # Example 2: Advanced structured search
    print("\n2. Advanced Structured Search")
    print("-" * 30)
    
    advanced_query = SearchQuery(
        text="stop loss",
        strategy_types=["tbs", "tv"],
        categories=["risk_management", "trading"],
        parameter_types=[ParameterType.FLOAT, ParameterType.INTEGER],
        fuzzy_matching=True,
        max_results=50,
        sort_by="relevance"
    )
    
    response = search_engine.search(advanced_query)
    print(f"Advanced Query: stop loss in TBS/TV strategies, risk/trading categories")
    print(f"Results: {response.total_results} parameters found")
    
    for i, result in enumerate(response.results[:5]):
        print(f"   {i+1}. {result.parameter_definition.name}")
        print(f"      Strategy: {result.parameter_definition.strategy_type}")
        print(f"      Category: {result.parameter_definition.category}")
        print(f"      Type: {result.parameter_definition.data_type.value}")
        print(f"      Relevance: {result.relevance_score:.3f}")
    
    # Example 3: Fuzzy search demonstration
    print("\n3. Fuzzy Search (with typos)")
    print("-" * 30)
    
    # Intentional typos
    fuzzy_queries = ["portolio", "indicater", "optmization"]
    
    for typo_query in fuzzy_queries:
        response = search_engine.search(typo_query)
        print(f"Query: '{typo_query}' -> {response.total_results} results")
        
        if response.did_you_mean:
            print(f"   Did you mean: '{response.did_you_mean}'?")
        
        if response.results:
            best_match = response.results[0]
            print(f"   Best match: {best_match.parameter_definition.name} "
                  f"(score: {best_match.relevance_score:.3f})")
    
    # Example 4: Faceted search
    print("\n4. Faceted Search Results")
    print("-" * 30)
    
    response = search_engine.search("risk")
    print(f"Query: 'risk' - {response.total_results} results")
    
    print("\nFacets (categories):")
    for facet_name, facet_data in response.facets.items():
        print(f"   {facet_name}:")
        for value, count in list(facet_data.items())[:5]:
            print(f"     - {value}: {count}")
    
    # Example 5: Autocomplete
    print("\n5. Autocomplete Suggestions")
    print("-" * 30)
    
    partial_queries = ["cap", "stop", "indi"]
    
    for partial in partial_queries:
        suggestions = search_engine.autocomplete(partial, max_suggestions=5)
        print(f"'{partial}' -> {suggestions}")
    
    # Example 6: Related parameters
    print("\n6. Related Parameters")
    print("-" * 30)
    
    response = search_engine.search("capital")
    if response.related_parameters:
        print("Related parameters to 'capital' search:")
        for param_id in response.related_parameters[:5]:
            print(f"   - {param_id}")
    
    # Example 7: Search suggestions
    print("\n7. Search Suggestions")
    print("-" * 30)
    
    response = search_engine.search("xyz_nonexistent")
    print(f"Query: 'xyz_nonexistent' -> {response.total_results} results")
    
    if response.suggestions:
        print("Suggestions:")
        for suggestion in response.suggestions[:5]:
            print(f"   - {suggestion}")
    
    print("\n" + "=" * 60)
    print("🎉 Search Engine Demo Complete!")

def demonstrate_performance_features():
    """Demonstrate performance and analytics features"""
    
    print("\n" + "=" * 60)
    print("⚡ Performance & Analytics Demo")
    print("=" * 60)
    
    registry = ParameterRegistry()
    search_engine = ParameterSearchEngine(registry)
    
    # Performance test
    print("\n1. Performance Test")
    print("-" * 30)
    
    import time
    test_queries = [
        "capital portfolio",
        "stop loss risk",
        "indicator technical",
        "ml model prediction",
        "optimization parameter"
    ]
    
    total_time = 0
    for query in test_queries:
        start_time = time.time()
        response = search_engine.search(query)
        query_time = time.time() - start_time
        total_time += query_time
        
        print(f"   '{query}': {response.total_results} results in {query_time:.3f}s")
    
    print(f"   Average query time: {total_time / len(test_queries):.3f}s")
    
    # Cache demonstration
    print("\n2. Cache Performance")
    print("-" * 30)
    
    # First query (uncached)
    start_time = time.time()
    response1 = search_engine.search("capital management")
    time1 = time.time() - start_time
    
    # Second query (cached)
    start_time = time.time()
    response2 = search_engine.search("capital management")
    time2 = time.time() - start_time
    
    print(f"   First query: {time1:.3f}s")
    print(f"   Cached query: {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.1f}x faster")
    
    # Analytics
    print("\n3. Search Analytics")
    print("-" * 30)
    
    analytics = search_engine.get_search_analytics()
    
    print("Query Statistics:")
    stats = analytics['query_stats']
    print(f"   - Total queries: {stats['total_queries']}")
    print(f"   - Average execution time: {stats['avg_execution_time']:.3f}s")
    print(f"   - Average result count: {stats['avg_result_count']:.1f}")
    
    if analytics['top_queries']:
        print("\nTop Queries:")
        for i, query_data in enumerate(analytics['top_queries'][:5]):
            print(f"   {i+1}. '{query_data['query']}' ({query_data['frequency']} times)")
    
    print("\nIndex Statistics:")
    index_stats = analytics['index_stats']
    print(f"   - Parameters indexed: {index_stats['total_parameters']}")
    print(f"   - Search terms: {index_stats['total_search_terms']}")
    print(f"   - Cache entries: {index_stats['cache_size']}")

def demonstrate_advanced_features():
    """Demonstrate advanced search features"""
    
    print("\n" + "=" * 60)
    print("🚀 Advanced Features Demo")
    print("=" * 60)
    
    registry = ParameterRegistry()
    search_engine = ParameterSearchEngine(registry)
    
    # Semantic search
    print("\n1. Semantic Search")
    print("-" * 30)
    
    # Search for risk-related parameters using different terms
    risk_queries = ["risk", "loss", "protection", "safety"]
    
    for query in risk_queries:
        response = search_engine.search(query)
        risk_results = [r for r in response.results if 'risk' in r.parameter_definition.category.lower()]
        print(f"   '{query}': {len(risk_results)} risk-related parameters")
    
    # Complex filtering
    print("\n2. Complex Filtering")
    print("-" * 30)
    
    complex_query = SearchQuery(
        text="",  # No text search
        strategy_types=["ml_triple_straddle"],
        categories=["ml_models"],
        parameter_types=[ParameterType.FLOAT],
        max_results=20,
        sort_by="name"
    )
    
    response = search_engine.search(complex_query)
    print(f"ML Triple Straddle float parameters in ml_models category:")
    print(f"   Found {response.total_results} parameters")
    
    for result in response.results[:5]:
        print(f"   - {result.parameter_definition.name}: {result.parameter_definition.default_value}")
    
    # Index optimization
    print("\n3. Index Optimization")
    print("-" * 30)
    
    print("   Optimizing search index...")
    search_engine.optimize_index()
    print("   ✅ Index optimization complete")
    
    # Cache management
    print("\n4. Cache Management")
    print("-" * 30)
    
    cache_size_before = len(search_engine._query_cache)
    print(f"   Cache size before clear: {cache_size_before}")
    
    search_engine.clear_cache()
    cache_size_after = len(search_engine._query_cache)
    print(f"   Cache size after clear: {cache_size_after}")

def demonstrate_integration_examples():
    """Show integration examples for common use cases"""
    
    print("\n" + "=" * 60)
    print("🔗 Integration Examples")
    print("=" * 60)
    
    registry = ParameterRegistry()
    search_engine = ParameterSearchEngine(registry)
    
    # Example 1: Parameter discovery for UI
    print("\n1. Parameter Discovery for UI Forms")
    print("-" * 30)
    
    def find_form_parameters(strategy_type: str):
        """Find parameters for dynamic form generation"""
        query = SearchQuery(
            strategy_types=[strategy_type],
            sort_by="category",
            max_results=1000
        )
        
        response = search_engine.search(query)
        
        # Group by category for form sections
        categories = {}
        for result in response.results:
            category = result.parameter_definition.category
            if category not in categories:
                categories[category] = []
            categories[category].append(result.parameter_definition)
        
        return categories
    
    tbs_params = find_form_parameters("tbs")
    print(f"   TBS strategy form sections:")
    for category, params in tbs_params.items():
        print(f"     - {category}: {len(params)} parameters")
    
    # Example 2: Parameter validation assistance
    print("\n2. Parameter Validation Assistance")
    print("-" * 30)
    
    def find_validation_conflicts(param_name: str):
        """Find parameters with similar names for validation"""
        response = search_engine.search(param_name)
        
        similar_params = []
        for result in response.results:
            if result.match_type == "fuzzy" and result.relevance_score > 0.7:
                similar_params.append({
                    'name': result.parameter_definition.name,
                    'strategy': result.parameter_definition.strategy_type,
                    'similarity': result.relevance_score
                })
        
        return similar_params
    
    conflicts = find_validation_conflicts("capital_amount")
    print(f"   Similar parameters to 'capital_amount':")
    for conflict in conflicts[:3]:
        print(f"     - {conflict['name']} ({conflict['strategy']}) - {conflict['similarity']:.3f}")
    
    # Example 3: Configuration analysis
    print("\n3. Configuration Analysis")
    print("-" * 30)
    
    def analyze_strategy_complexity(strategy_type: str):
        """Analyze strategy complexity by parameter count"""
        query = SearchQuery(strategy_types=[strategy_type])
        response = search_engine.search(query)
        
        return {
            'total_parameters': response.total_results,
            'categories': len(response.facets.get('categories', {})),
            'parameter_types': len(response.facets.get('parameter_types', {}))
        }
    
    strategies = ["tbs", "ml_triple_straddle", "oi"]
    for strategy in strategies:
        complexity = analyze_strategy_complexity(strategy)
        print(f"   {strategy}: {complexity['total_parameters']} params, "
              f"{complexity['categories']} categories, "
              f"{complexity['parameter_types']} types")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_search_capabilities()
    demonstrate_performance_features()
    demonstrate_advanced_features()
    demonstrate_integration_examples()
    
    print("\n✨ All search engine demonstrations complete!")
    print("\nKey Features Demonstrated:")
    print("  • Full-text search with relevance ranking")
    print("  • Fuzzy matching and spell correction")
    print("  • Advanced filtering and faceted search")
    print("  • Autocomplete and suggestions")
    print("  • Performance optimization and caching")
    print("  • Search analytics and monitoring")
    print("  • Semantic relationships")
    print("  • Integration patterns for common use cases")
    
    print("\nQuick Start Example:")
    print("""
# Basic usage
from configurations.search import ParameterSearchEngine, SearchQuery

# Initialize
engine = ParameterSearchEngine()

# Simple search
response = engine.search("risk management")
for result in response.results:
    print(f"{result.parameter_definition.name}: {result.relevance_score}")

# Advanced search
query = SearchQuery(
    text="stop loss",
    strategy_types=["tbs"],
    parameter_types=[ParameterType.FLOAT],
    fuzzy_matching=True
)
response = engine.search(query)

# Autocomplete
suggestions = engine.autocomplete("cap")
""")