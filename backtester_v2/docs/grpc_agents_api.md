# BMAD Agent gRPC API Documentation

## Overview
This document describes the gRPC API for communication between the 34 BMAD agents in the Enterprise GPU Backtester system.

## Generated Files
- `bmad_agents_pb2.py` - Protocol buffer message definitions
- `bmad_agents_pb2_grpc.py` - gRPC service stubs and servicer classes  
- `bmad_agents_pb2.pyi` - Type hints for IDE support

## Services

### MasterCoordinatorService (port 8010)
Orchestrates validation across all agents.

### StrategyAgentService (ports 8001-8009)  
Validates parameters for specific trading strategies.

### HeavyDBConnectorService (port 8011)
Manages database connections and queries.

### TDDEngineService (port 8012)
Handles test-driven development cycles for parameter optimization.

### ValidationOrchestratorService (port 8014)
Coordinates bulk validation operations.

### ProductionMonitoringService (ports 8120-8122)
Monitors system health and performance.

## Message Types

### Core Messages
- `AgentInfo` - Agent identification and status
- `ParameterValidationRequest` - Parameter validation request
- `ValidationResult` - Validation result with evidence
- `ValidationContext` - Validation context and constraints

### Strategy-Specific Messages
- `TBSValidationData` - Time-based strategy parameters
- `TVValidationData` - TradingView strategy parameters
- `MLValidationData` - Machine learning strategy parameters
- `ORBValidationData` - Opening range breakout parameters
- `OIValidationData` - Open interest strategy parameters
- `POSValidationData` - Positional Greeks strategy parameters
- `MRValidationData` - Market regime strategy parameters
- `INDValidationData` - Technical indicator parameters
- `OPTValidationData` - Optimization algorithm parameters

## Usage Example

```python
from bmad_agent_communication import BMadAgentBase, create_agent_config
from bmad_agent_communication.generated import bmad_agents_pb2_grpc

class TBSStrategyAgent(BMadAgentBase):
    async def _register_services(self):
        bmad_agents_pb2_grpc.add_StrategyAgentServiceServicer_to_server(
            TBSStrategyServicer(), self.server
        )

# Create and start agent
config = create_agent_config("bmad-tbs-strategy-agent", "strategy", 8001)
agent = TBSStrategyAgent(config)
await agent.start_server()
```

## Next Steps
- Phase 1.2: Implement Consul service discovery
- Phase 1.3: Add HashiCorp Vault secrets management  
- Phase 1.4: Configure TLS mutual authentication
