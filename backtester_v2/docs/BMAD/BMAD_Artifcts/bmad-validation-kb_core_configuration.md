### Strategy Overview
| Strategy | Code | Parameters | Priority | Enhanced | Path |
|----------|------|------------|----------|----------|------|
| TBS | tbs | 83 | High | No | /backtester_v2/strategies/tbs/ |
| TV | tv | 133 | High | No | /backtester_v2/strategies/tv/ |
| OI | oi | 142 | Medium | No | /backtester_v2/strategies/oi/ |
| ORB | orb | 19 | Medium | No | /backtester_v2/strategies/orb/ |
| POS | pos | 156 | Medium | No | /backtester_v2/strategies/pos/ |
| ML Indicator | ml_indicator | 439 | Critical | Yes | /backtester_v2/strategies/ml_indicator/ |
| Market Regime | market_regime | 267 | Critical | Yes | /backtester_v2/strategies/market_regime/ |
| Indicator | indicator | 197 | High | No | /backtester_v2/strategies/indicator/ |
| Optimization | optimization | 283 | High | No | /backtester_v2/strategies/optimization/ |

### Enhanced Validation Requirements
ML and MR strategies require additional validation:
- Double validation protocol
- Statistical anomaly detection
- Cross-reference with research papers
- Expert system consultation
- Consensus scoring (>80% required)# BMAD Validation System Knowledge Base

## Overview

This knowledge base contains critical information for all validation agents operating within the BMAD-Enhanced Validation System.

## System Architecture Overview

### Multi-Agent Validation Framework
The validation system employs 21 specialized agents working in concert:
- **9 Strategy Validators**: One for each trading strategy
- **3 Core Validators**: HeavyDB, Placeholder Guardian, GPU Optimizer
- **3 Support Agents**: Fix Agent, Performance Optimizer, Doc Updater
- **6 BMAD Agents**: Orchestrator, Master, PM, Architect, SM, QA

### Validation Pipeline
1. **Planning Phase**: PRD and Architecture creation
2. **Story Creation**: Detailed validation stories per strategy
3. **Execution Phase**: Parallel validation execution
4. **Optimization Phase**: Performance tuning
5. **Reporting Phase**: Comprehensive documentation

## Strategy Information

### Strategy Overview
| Strategy | Code | Parameters | Priority | Enhanced | Path |
|----------|------|------------|----------|----------|------|
| TBS | tbs | 102 | High | No | /backtester_v2/strategies/tbs/ |
| TV | tv | 75 | High | No | /backtester_v2/strategies/tv/ |
| OI | oi | 63 | Medium | No | /backtester_v2/strategies/oi/ |
| ORB | orb | 42 | Medium | No | /backtester_v2/strategies/orb/ |
| POS | pos | 38 | Medium | No | /backtester_v2/strategies/pos/ |
| ML Indicator | ml_indicator | 89 | Critical | Yes | /backtester_v2/strategies/ml_indicator/ |
| Market Regime | market_regime | 56 | Critical | Yes | /backtester_v2/strategies/market_regime/ |
| Indicator | indicator | 44 | High | No | /backtester_v2/strategies/indicator/ |
| Optimization | optimization | 67 | High | No | /backtester_v2/strategies/optimization/ |

### Enhanced Validation Requirements
ML and MR strategies require additional validation:
- Double validation protocol
- Statistical anomaly detection
- Cross-reference with research papers
- Expert system consultation
- Consensus scoring (>80% required)

## HeavyDB Configuration

### Connection Parameters
```python
# Production Configuration
HEAVYDB_HOST = "173.208.247.17"
HEAVYDB_PORT = "6274"
HEAVYDB_USER = "admin"
HEAVYDB_PASSWORD = ""  # Empty for production
HEAVYDB_DATABASE = "heavyai"
HEAVYDB_PROTOCOL = "binary"

# Local Development Configuration
HEAVYDB_HOST_DEV = "127.0.0.1"
HEAVYDB_PASSWORD_DEV = "HyperInteractive"
```

### GPU Optimization Settings
```python
GPU_SETTINGS = {
    'enabled': True,
    'target_utilization': 0.7,  # 70%
    'memory_pool_size': '4GB',
    'kernel_optimization': True,
    'mixed_precision': True,
    'tensor_cores': True
}
```

### Performance Targets
- Query execution: < 50ms (95th percentile)
- GPU utilization: > 70%
- Memory usage: < 80%
- Throughput: > 1000 validations/second

## Validation Rules

### Data Type Mappings
```python
EXCEL_TO_BACKEND_TYPES = {
    'Number': ['float', 'double', 'decimal'],
    'Integer': ['int', 'bigint', 'smallint'],
    'Text': ['varchar', 'text', 'string'],
    'Boolean': ['boolean', 'bit'],
    'Date': ['date', 'timestamp', 'datetime'],
    'Currency': ['decimal(15,2)', 'money']
}
```

### Synthetic Data Patterns
```python
FORBIDDEN_PATTERNS = [
    # Test Values
    r'^(123|999|000|111|777),
    r'^test.*',
    r'.*dummy.*',
    r'.*sample.*',
    
    # Placeholder Prices
    r'^\d+\.00,  # Round numbers like 100.00
    r'^1234\.56,  # Sequential decimals
    
    # Test Dates
    r'^2020-01-01',
    r'^1970-01-01',  # Unix epoch
    r'^2000-01-01',  # Y2K test date
    
    # Sequential Patterns
    r'^[A-Z]{3}123,  # ABC123
    r'^\d{1,3}(?:\d{3})*,  # 1000, 2000, 3000
    
    # Perfect Data
    'ZERO_VARIANCE',  # Statistical check
    'PERFECT_SEQUENCE',  # 1,2,3,4,5...
    'NO_VOLATILITY'  # Flat line data
]
```

### Validation Error Codes
```python
ERROR_CODES = {
    'VAL001': 'Parameter not found in Excel',
    'VAL002': 'Backend mapping missing',
    'VAL003': 'Data type mismatch',
    'VAL004': 'Value out of range',
    'VAL005': 'HeavyDB storage failed',
    'VAL006': 'Query performance exceeded threshold',
    'VAL007': 'Synthetic data detected',
    'VAL008': 'GPU optimization failed',
    'VAL009': 'Data integrity violation',
    'VAL010': 'Dependency validation failed'
}
```

## Common Validation Scenarios

### Scenario 1: New Parameter Addition
1. Update Excel file with new parameter
2. Add backend mapping
3. Run validation for affected strategy
4. Verify HeavyDB schema update
5. Test query performance
6. Update documentation

### Scenario 2: Performance Degradation
1. Identify slow queries via monitoring
2. Profile GPU utilization
3. Apply optimization techniques
4. Re-test performance
5. Document optimization

### Scenario 3: Data Integrity Issue
1. Placeholder Guardian detects synthetic data
2. Block validation immediately
3. Identify data source
4. Replace with production data
5. Re-run validation
6. Update audit log

## Optimization Techniques

### GPU Query Optimization
1. **Memory Coalescing**
   - Align memory access patterns
   - Use columnar data formats
   - Minimize random access

2. **Kernel Fusion**
   - Combine multiple operations
   - Reduce kernel launch overhead
   - Optimize data movement

3. **Batch Processing**
   - Optimal batch size: 10K-100K rows
   - Balance memory usage vs throughput
   - Use async processing

4. **Index Optimization**
   - Create GPU-resident indexes
   - Use covering indexes
   - Partition large tables

### Performance Tuning Checklist
- [ ] Profile query execution plan
- [ ] Check GPU memory usage
- [ ] Verify kernel efficiency
- [ ] Optimize data layout
- [ ] Enable query caching
- [ ] Test with production data volumes

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Validation Timeout
**Symptoms**: Validation hangs or times out
**Causes**:
- Large data volume
- Inefficient query
- Network issues
**Solutions**:
1. Increase timeout temporarily
2. Optimize query
3. Use chunked processing
4. Check network connectivity

#### Issue: GPU Memory Error
**Symptoms**: CUDA out of memory errors
**Causes**:
- Batch size too large
- Memory leak
- Concurrent validations
**Solutions**:
1. Reduce batch size
2. Clear GPU cache
3. Limit parallel validations
4. Monitor memory usage

#### Issue: Data Type Mismatch
**Symptoms**: Validation fails with type error
**Causes**:
- Excel format change
- Backend schema update
- Mapping error
**Solutions**:
1. Verify Excel format
2. Check backend schema
3. Update mapping configuration
4. Test with sample data

## Best Practices

### Validation Best Practices
1. **Always validate in stages** - Don't try to validate everything at once
2. **Use production data samples** - Never use synthetic data
3. **Monitor continuously** - Watch dashboard during validation
4. **Document everything** - Every decision and change
5. **Optimize iteratively** - Small improvements add up

### Performance Best Practices
1. **Profile before optimizing** - Measure first, optimize second
2. **Cache query results** - Reuse when possible
3. **Batch similar operations** - Group related validations
4. **Use GPU effectively** - Ensure high utilization
5. **Monitor resource usage** - Prevent bottlenecks

### Data Integrity Best Practices
1. **Verify data sources** - Always confirm production origin
2. **Cross-reference multiple sources** - Don't trust single source
3. **Statistical validation** - Check distributions make sense
4. **Audit everything** - Complete trail of data lineage
5. **Regular spot checks** - Random sampling validation

## Agent Communication Protocols

### Message Format
```yaml
message:
  from: sender_agent_id
  to: recipient_agent_id
  type: request|response|notification|error
  priority: critical|high|normal|low
  timestamp: ISO8601
  content:
    action: requested_action
    parameters: {}
    context: {}
```

### Handoff Protocol
1. **Prepare handoff package**
   - Current state
   - Completed actions
   - Pending tasks
   - Issues found

2. **Notify next agent**
   - Send handoff message
   - Wait for acknowledgment
   - Transfer context

3. **Monitor progress**
   - Track next agent's progress
   - Be ready to assist
   - Log handoff completion

## Compliance and Audit

### Audit Requirements
- Every validation action logged
- Timestamp all operations
- Record decision rationale
- Maintain data lineage
- Archive for 7 years

### Compliance Checklist
- [ ] SOX compliance for financial parameters
- [ ] Data privacy regulations
- [ ] Internal audit requirements
- [ ] Regulatory reporting needs
- [ ] Risk management policies

## Quick Reference

### Command Reference
```bash
# Validation Commands
*validate-strategy {name}    # Validate single strategy
*validate-all               # Validate all strategies
*validate-parameter {id}    # Validate single parameter

# Performance Commands
*optimize-query {id}        # Optimize specific query
*profile-gpu               # Profile GPU usage
*benchmark {strategy}      # Run performance benchmark

# Data Commands
*scan-synthetic           # Scan for synthetic data
*verify-production        # Verify production data
*check-integrity         # Run integrity checks

# Reporting Commands
*generate-report         # Generate validation report
*dashboard              # Show real-time dashboard
*audit-trail           # Display audit log
```

### Key Metrics
- **Coverage**: Percentage of parameters validated
- **Accuracy**: Percentage of validations passed
- **Performance**: Average query execution time
- **Utilization**: GPU usage percentage
- **Integrity**: Data quality score

## Appendix

### Glossary
- **Parameter**: Configuration value from Excel
- **Mapping**: Translation between Excel and backend
- **Validation**: Process of verifying correctness
- **Synthetic Data**: Fake/test data (prohibited)
- **GPU Optimization**: Tuning for GPU execution
- **Story**: Detailed validation task description
- **Handoff**: Transfer of work between agents

### References
- BMAD Method Documentation
- HeavyDB Performance Guide
- CUDA Optimization Manual
- Trading Strategy Specifications
- Validation Standards Document