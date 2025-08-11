# Coding Standards and Conventions

## Overview
This document defines the coding standards and conventions for the Market Regime Master Framework project, ensuring consistency and maintainability across all development efforts.

## Existing Standards Compliance

### Code Style
- **PEP 8 compliance** with existing project conventions
- **120 character line limit** for improved readability
- **Descriptive variable names** following snake_case convention
- **Google-style docstrings** with comprehensive type hints
- **Structured JSON output** for all logging

### Linting and Formatting
- **flake8** with existing project exclusions
- **black** code formatting for consistent style
- **Type hints** mandatory for all public methods
- **Import organization** following PEP 8 standards

### Testing Standards
- **pytest framework** with existing test structure
- **90%+ code coverage requirement** for all new code
- **Comprehensive test fixtures** for market data scenarios
- **Mock objects** for external ML services and APIs

### Documentation Requirements
- **Google-style docstrings** for all classes and methods
- **Type hints** for all function parameters and returns
- **Comprehensive inline documentation** for complex algorithms
- **API documentation** auto-generated from docstrings

## Enhancement-Specific Standards

### Adaptive Component Architecture
- **AdaptiveComponent base class**: All 8 components must implement standardized interface
- **Standardized method signatures**: `analyze()`, `optimize_weights()`, `get_health_metrics()`
- **Component isolation**: Each component must be independently testable and deployable
- **Performance budgets**: Each component must respect allocated processing time and memory limits

### ML Integration Patterns
- **Vertex AI calls**: Wrapped in retry logic with exponential backoff
- **Circuit breaker pattern**: Automatic fallback to existing algorithms on ML service failures
- **Graceful degradation**: System continues operating with reduced functionality
- **Connection pooling**: Efficient resource management for cloud services

### Performance Logging Standards
- **Sub-component timing**: Mandatory timing logs for <600ms total target
- **Memory usage tracking**: Peak memory monitoring per component
- **Structured logging**: JSON format for machine-readable logs
- **Performance metrics**: Latency, throughput, accuracy tracking

### Error Handling Requirements
- **Graceful fallback**: All ML service calls must have fallback to existing algorithms
- **Exception hierarchy**: Custom exceptions for different failure modes
- **Logging on failure**: Comprehensive error logging with context
- **Recovery mechanisms**: Automatic retry with exponential backoff

## Critical Integration Rules

### API Compatibility
- **Backward compatibility**: All new endpoints maintain existing API contracts
- **Versioning strategy**: /api/v1/ preserved, /api/v2/ for enhanced features
- **Response format**: Consistent JSON response structures
- **Error responses**: Standardized error codes and messages

### Database Integration
- **Parquet-first architecture**: Primary storage in GCS Parquet format
- **Arrow memory layer**: Zero-copy data access patterns
- **No HeavyDB modifications**: Existing schema preservation during migration
- **New tables only**: Additive database changes only

### Component Integration
- **Standardized interfaces**: Common base class for all components
- **Dependency injection**: Configurable dependencies for testing
- **Event-driven communication**: Asynchronous component coordination
- **Health monitoring**: Real-time component status tracking

### Logging Consistency
- **Existing framework**: Use project's established logging infrastructure
- **Structured JSON**: Machine-readable log format
- **Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Performance logs**: Separate performance metrics logging

## Code Organization Standards

### File Naming Conventions
- **snake_case**: All Python files and modules
- **Descriptive prefixes**: `adaptive_`, `component_`, `ml_`
- **Clear separation**: Model, view, controller separation where applicable
- **Test files**: `test_` prefix for all test modules

### Import Standards
```python
# Standard library imports first
import asyncio
import time
from typing import Dict, List, Optional

# Third-party imports second
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Local application imports last
from adaptive_learning.components.base import AdaptiveComponent
from ml_integration.vertex_ai_client import VertexAIClient
```

### Class and Method Standards
```python
class ComponentExample(AdaptiveComponent):
    """
    Example component following coding standards.
    
    Args:
        config: Configuration dictionary with component parameters
        
    Attributes:
        processing_budget_ms: Maximum processing time in milliseconds
        memory_budget_mb: Maximum memory usage in megabytes
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.processing_budget_ms = config.get("processing_budget_ms", 100)
        self.memory_budget_mb = config.get("memory_budget_mb", 256)
    
    async def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using component-specific algorithms.
        
        Args:
            market_data: DataFrame with market data columns
            
        Returns:
            Dictionary with analysis results and metadata
            
        Raises:
            ComponentAnalysisError: When analysis fails
        """
        start_time = time.time()
        
        try:
            # Implementation here
            result = self._perform_analysis(market_data)
            
            processing_time = (time.time() - start_time) * 1000
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"Processing time {processing_time}ms exceeded budget {self.processing_budget_ms}ms")
                
            return {
                "analysis_result": result,
                "processing_time_ms": processing_time,
                "memory_usage_mb": self._get_memory_usage()
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise ComponentAnalysisError(f"Component analysis failed: {str(e)}")
```

## Quality Assurance Standards

### Code Review Requirements
- **Peer review**: All code changes require peer review
- **Automated checks**: CI/CD pipeline with linting, testing, security scanning
- **Performance validation**: Automated performance regression testing
- **Documentation review**: Technical documentation review for accuracy

### Security Standards
- **Input validation**: All external inputs validated and sanitized
- **Secret management**: No hardcoded secrets, use environment variables
- **Authentication**: Proper authentication for all API endpoints
- **Authorization**: Role-based access control implementation

### Performance Standards
- **Component budgets**: Each component has defined processing time and memory limits
- **Monitoring**: Real-time performance monitoring and alerting
- **Optimization**: Regular performance profiling and optimization
- **Scalability**: Code designed for horizontal scaling

## Compliance and Validation

### BMAD Compliance
- **Standards alignment**: Full compliance with BMAD development standards
- **Validation gates**: Code must pass all defined validation checkpoints
- **Documentation requirements**: Comprehensive technical documentation
- **Quality metrics**: Measurable quality and performance metrics

### Continuous Integration
- **Automated testing**: Full test suite execution on all commits
- **Code coverage**: Minimum 90% code coverage enforcement
- **Security scanning**: Automated vulnerability scanning
- **Performance testing**: Automated performance regression testing

This document serves as the definitive guide for all development activities within the Market Regime Master Framework project.