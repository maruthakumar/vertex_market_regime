# Coding Standards and Conventions

## Existing Standards Compliance

**Code Style:** PEP 8 compliance with existing project conventions (120 character line limit, descriptive variable names)
**Linting Rules:** flake8 with existing project exclusions, black code formatting
**Testing Patterns:** pytest framework with existing test structure, 90%+ code coverage requirement
**Documentation Style:** Google-style docstrings with type hints, comprehensive inline documentation

## Enhancement-Specific Standards

- **Adaptive Component Interface:** All 8 components must implement `AdaptiveComponent` base class with standardized methods
- **ML Integration Pattern:** Vertex AI calls wrapped in retry logic with circuit breaker pattern
- **Performance Logging:** Mandatory sub-component timing logs for <800ms total target
- **Error Handling:** All ML service calls must have graceful fallback to existing algorithms

## Critical Integration Rules

- **Existing API Compatibility:** All new endpoints maintain backward compatibility, existing endpoints unchanged
- **Database Integration:** New tables only, no modification of existing HeavyDB schema
- **Error Handling:** All adaptive learning failures must gracefully fall back to existing regime classification
- **Logging Consistency:** All new components use existing logging framework with structured JSON output
