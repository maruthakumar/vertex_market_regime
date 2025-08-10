# SuperClaude Persona to TaskMaster Agent Mapping Guide

## Overview

This guide provides comprehensive documentation for the SuperClaude persona to TaskMaster agent mapping system, enabling autonomous AI-assisted development with specialized cognitive agents. The integration bridges SuperClaude's 9 specialized personas with TaskMaster's autonomous task execution capabilities.

## Table of Contents

1. [Persona-Agent Mapping Matrix](#persona-agent-mapping-matrix)
2. [Command-Operation Mapping](#command-operation-mapping)
3. [Agent Specializations](#agent-specializations)
4. [MCP Server Integration](#mcp-server-integration)
5. [Command Routing Logic](#command-routing-logic)
6. [Real Examples from Test Results](#real-examples-from-test-results)
7. [Best Practices for Persona Selection](#best-practices-for-persona-selection)
8. [Autonomous Workflow Patterns](#autonomous-workflow-patterns)
9. [Configuration and Setup](#configuration-and-setup)
10. [Troubleshooting](#troubleshooting)

## Persona-Agent Mapping Matrix

The system maps SuperClaude's 9 specialized personas to TaskMaster's autonomous agents based on expertise domains and task types:

### Core Mapping Table

| SuperClaude Persona | TaskMaster Agent | Primary Focus | Use Case |
|---------------------|------------------|---------------|----------|
| `architect` | `research` | System design, architecture analysis | High-level design decisions, system planning |
| `frontend` | `implementation` | UI/UX development, client-side code | React components, styling, user interfaces |
| `backend` | `implementation` | Server-side logic, APIs, databases | FastAPI routes, business logic, data processing |
| `security` | `structure_enforcer` | Security audits, vulnerability analysis | Code security, authentication, authorization |
| `performance` | `research` | Performance optimization, profiling | Query optimization, caching, bottleneck analysis |
| `qa` | `structure_enforcer` | Quality assurance, testing, validation | Test generation, code review, quality metrics |
| `ml` | `research` | Machine learning, data science | Algorithm implementation, model training |
| `devops` | `orchestrator` | Infrastructure, deployment, automation | CI/CD, monitoring, system administration |
| `data` | `research` | Data engineering, ETL, databases | Data pipelines, database design, analytics |

### Mapping Implementation

```python
def _init_persona_mapping(self) -> Dict[str, str]:
    """Map SuperClaude personas to TaskMaster agents"""
    return {
        "architect": "research",
        "frontend": "implementation", 
        "backend": "implementation",
        "security": "structure_enforcer",
        "performance": "research",
        "qa": "structure_enforcer",
        "ml": "research",
        "devops": "orchestrator",
        "data": "research"
    }
```

## Command-Operation Mapping

SuperClaude commands are mapped to specific TaskMaster operations for optimal task execution:

### Command Mapping Table

| SuperClaude Command | TaskMaster Operation | Purpose | Typical Agent |
|---------------------|---------------------|---------|---------------|
| `/analyze` | `research` | Deep analysis, investigation | research, structure_enforcer |
| `/implement` | `implement` | Code implementation, feature development | implementation |
| `/test` | `test` | Test generation, validation | structure_enforcer |
| `/debug` | `debug` | Issue investigation, troubleshooting | research, implementation |
| `/optimize` | `optimize` | Performance improvements | research |
| `/refactor` | `refactor` | Code restructuring, cleanup | implementation |
| `/docs` | `document` | Documentation generation | any agent |
| `/project` | `orchestrate` | Project management, coordination | orchestrator |
| `/workflow` | `orchestrate` | Workflow automation | orchestrator |
| `/security` | `audit` | Security analysis, vulnerability scanning | structure_enforcer |

### Implementation Example

```python
def _init_command_mapping(self) -> Dict[str, str]:
    """Map SuperClaude commands to TaskMaster operations"""
    return {
        "/analyze": "research",
        "/implement": "implement", 
        "/test": "test",
        "/debug": "debug",
        "/optimize": "optimize",
        "/refactor": "refactor",
        "/docs": "document",
        "/project": "orchestrate",
        "/workflow": "orchestrate",
        "/security": "audit"
    }
```

## Agent Specializations

### Research Agent
**Mapped Personas**: `architect`, `performance`, `ml`, `data`

**Capabilities**:
- System architecture analysis
- Performance profiling and optimization
- Machine learning algorithm research
- Data engineering and pipeline design
- Deep technical investigation

**Best For**:
- Complex analysis tasks requiring domain expertise
- Research-heavy implementations
- Performance bottleneck identification
- Algorithm selection and optimization

### Implementation Agent
**Mapped Personas**: `frontend`, `backend`

**Capabilities**:
- Code generation and implementation
- Feature development
- UI/UX component creation
- API endpoint development
- Database integration

**Best For**:
- Direct code implementation
- Feature development
- Component creation
- Integration tasks

### Structure Enforcer Agent
**Mapped Personas**: `security`, `qa`

**Capabilities**:
- Code quality validation
- Security vulnerability analysis
- Test generation and execution
- Compliance checking
- Code review and standards enforcement

**Best For**:
- Quality assurance tasks
- Security audits
- Test coverage improvement
- Code standards enforcement

### Orchestrator Agent
**Mapped Personas**: `devops`

**Capabilities**:
- Project coordination
- Workflow automation
- Infrastructure management
- CI/CD pipeline optimization
- System administration

**Best For**:
- Multi-agent coordination
- Infrastructure tasks
- Deployment automation
- System monitoring

## MCP Server Integration

The system integrates with multiple Model Context Protocol (MCP) servers for enhanced capabilities:

### Available MCP Servers

1. **Sequential MCP (`--seq`)**
   - Complex analysis and problem-solving
   - Chain-of-thought reasoning
   - Multi-step workflows

2. **Magic MCP (`--magic`)**
   - UI component generation
   - Design system patterns
   - Frontend development assistance

3. **Context7 MCP (`--c7`)**
   - External library documentation
   - API reference lookup
   - Integration guidance

4. **Puppeteer MCP (`--pup`)**
   - Browser automation
   - E2E testing
   - Performance monitoring

5. **All MCP (`--all-mcp`)**
   - Enables all available MCP servers
   - Maximum capability integration

### MCP Server Selection by Persona

| Persona | Recommended MCP Servers | Rationale |
|---------|------------------------|-----------|
| `architect` | `seq`, `c7` | Complex analysis + library research |
| `frontend` | `magic`, `c7` | UI components + library docs |
| `backend` | `seq`, `c7` | Complex logic + API docs |
| `security` | `seq`, `c7` | Analysis + security libraries |
| `performance` | `seq`, `pup` | Analysis + performance testing |
| `qa` | `seq`, `pup` | Analysis + E2E testing |
| `ml` | `seq`, `c7` | Research + ML libraries |
| `devops` | `pup`, `seq` | Automation + infrastructure |
| `data` | `seq`, `c7` | Analysis + data libraries |

## Command Routing Logic

### Routing Algorithm

```python
def convert_to_taskmaster_task(self, sc_command: SuperClaudeCommand) -> TaskMasterTask:
    """Convert SuperClaude command to TaskMaster task"""
    # 1. Map command to operation
    operation = self.command_mapping.get(sc_command.command, "general")
    
    # 2. Map persona to agent
    agent = "general"
    if sc_command.persona:
        agent = self.persona_mapping.get(sc_command.persona, "general")
    
    # 3. Generate comprehensive task description
    description = self._generate_task_description(sc_command)
    
    return TaskMasterTask(
        id=f"{operation}_{sc_command.persona or 'general'}_{hash_id}",
        title=f"{sc_command.command} with {sc_command.persona or 'general'} persona",
        description=description,
        agent=agent,
        priority=self.config["default_priority"]
    )
```

### Task Description Enhancement

Each generated task includes:
- Command execution instructions
- Persona-specific expertise requirements
- MCP server integration specifications
- Context flags for optimal performance
- SuperClaude-specific requirements:
  - Real data validation (NO MOCK DATA)
  - Excel configuration validation
  - 500-line TODO limit enforcement
  - v6 plan integration requirements

## Real Examples from Test Results

### Example 1: Architecture Analysis

**Input Command**:
```bash
/analyze --persona-architect --seq --context:auto
```

**Parsed Result**:
```json
{
  "command": "/analyze",
  "persona": "architect",
  "mcp_servers": ["seq"],
  "context_flags": ["auto"]
}
```

**Generated TaskMaster Task**:
```json
{
  "id": "research_architect_20",
  "title": "/analyze with architect persona",
  "description": "Execute /analyze command. Using architect persona expertise. With MCP servers: seq. Context flags: auto. CRITICAL: Use only real data - NO MOCK DATA allowed. Validate Excel configurations if applicable. Ensure 500-line limit for any TODO outputs. Integrate with existing v6 plan requirements",
  "agent": "research",
  "priority": "medium"
}
```

### Example 2: Frontend Implementation

**Input Command**:
```bash
/implement --persona-frontend --magic --context:module
```

**Parsed Result**:
```json
{
  "command": "/implement",
  "persona": "frontend",
  "mcp_servers": ["magic"],
  "context_flags": ["module"]
}
```

**Generated TaskMaster Task**:
```json
{
  "title": "/implement with frontend persona",
  "agent": "implementation",
  "priority": "medium"
}
```

### Example 3: QA Testing

**Input Command**:
```bash
/test --persona-qa --c7 --context:file
```

**Generated TaskMaster Task**:
```json
{
  "title": "/test with qa persona",
  "agent": "structure_enforcer",
  "priority": "medium"
}
```

### Example 4: Performance Optimization

**Input Command**:
```bash
/optimize --persona-performance --all-mcp
```

**Generated TaskMaster Task**:
- Maps to `research` agent
- Includes all MCP servers for comprehensive analysis
- Focuses on performance improvement strategies

### Example 5: DevOps Orchestration

**Input Command**:
```bash
/project --persona-devops --pup --context:prd
```

**Generated TaskMaster Task**:
- Maps to `orchestrator` agent
- Uses Puppeteer MCP for automation capabilities
- Includes PRD context for project requirements

## Best Practices for Persona Selection

### 1. Task-Persona Alignment

**Architecture & Design Tasks**:
- Use `architect` persona for system design, high-level planning
- Combine with `--seq` for complex analysis
- Add `--context:auto` for comprehensive context loading

**Implementation Tasks**:
- Use `frontend` persona for UI/UX development
- Use `backend` persona for API and business logic
- Combine with `--magic` for UI components
- Use `--c7` for library documentation

**Quality & Security Tasks**:
- Use `qa` persona for testing and validation
- Use `security` persona for vulnerability analysis
- Combine with `--seq` for thorough analysis
- Use `--pup` for E2E testing

**Performance Tasks**:
- Use `performance` persona for optimization
- Combine with `--seq` for complex analysis
- Use `--pup` for performance testing

**Infrastructure Tasks**:
- Use `devops` persona for deployment and automation
- Combine with `--pup` for automation testing
- Use `--context:prd` for project requirements

### 2. MCP Server Selection Strategy

**For Deep Analysis**: Use `--seq`
**For UI Development**: Use `--magic`
**For Library Integration**: Use `--c7`
**For Testing & Automation**: Use `--pup`
**For Maximum Capability**: Use `--all-mcp`

### 3. Context Flag Guidelines

**`--context:auto`**: Automatic context detection
**`--context:module`**: Module-specific context
**`--context:file`**: File-specific context
**`--context:prd`**: Product requirements context
**`--context:excel`**: Excel configuration context

## Autonomous Workflow Patterns

### Pattern 1: Full-Stack Feature Development

```bash
# 1. Architecture analysis
/analyze --persona-architect --seq --context:auto

# 2. Backend implementation
/implement --persona-backend --c7 --context:module

# 3. Frontend implementation
/implement --persona-frontend --magic --context:module

# 4. Testing
/test --persona-qa --pup --context:file

# 5. Performance optimization
/optimize --persona-performance --seq --context:auto
```

### Pattern 2: Security-First Development

```bash
# 1. Security analysis
/analyze --persona-security --seq --context:auto

# 2. Secure implementation
/implement --persona-backend --c7 --context:module

# 3. Security testing
/test --persona-security --seq --context:file

# 4. Code review
/analyze --persona-qa --seq --context:file
```

### Pattern 3: Performance-Optimized Development

```bash
# 1. Performance analysis
/analyze --persona-performance --seq --context:auto

# 2. Optimized implementation
/implement --persona-backend --seq --context:module

# 3. Performance testing
/test --persona-performance --pup --context:file

# 4. Continuous optimization
/optimize --persona-performance --all-mcp --context:auto
```

## Configuration and Setup

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your_api_key"
export PERPLEXITY_API_KEY="your_perplexity_key"
export MODEL="claude-3-7-sonnet-20250219"
export PERPLEXITY_MODEL="sonar-pro"
export MAX_TOKENS="64000"
export TEMPERATURE="0.2"
export DEFAULT_SUBTASKS="5"
export DEFAULT_PRIORITY="medium"
```

### Configuration Parameters

```python
config = {
    "todo_line_limit": 500,
    "real_data_validation": True,
    "excel_validation_required": True,
    "default_subtasks": 5,
    "default_priority": "medium"
}
```

### Initialization

```python
from superclaude_taskmaster_integration import SuperClaudeTaskMasterBridge

# Initialize bridge
bridge = SuperClaudeTaskMasterBridge("/path/to/project")

# Initialize project with SuperClaude integration
result = bridge.initialize_project(
    project_name="my_project",
    prd_path="/path/to/prd.md"
)
```

## Advanced Features

### 1. TODO File Management

The system automatically enforces a 500-line limit on TODO files to maintain readability and manageability:

```python
# Split large TODO files automatically
result = bridge.enforce_todo_line_limits("large_todo.md")
# Creates: large_todo_part_1.md, large_todo_part_2.md
# Updates: large_todo.md with references to split files
```

### 2. Excel Configuration Validation

Specialized validation for Excel-based configurations common in the backtester system:

```python
# Validate all Excel configurations
result = bridge.validate_excel_configurations()
# Uses QA persona with structure_enforcer agent
```

### 3. Multi-Command Workflows

Execute complex workflows with multiple SuperClaude commands:

```python
commands = [
    "/analyze --persona-architect --seq --context:auto",
    "/implement --persona-backend --c7 --context:module",
    "/test --persona-qa --pup --context:file"
]
results = bridge.execute_superclaude_workflow(commands)
```

## Integration with Existing Systems

### GPU Backtester Integration

The system is specifically designed for the GPU Backtester platform:

- **Real Data Validation**: Enforces use of actual HeavyDB and MySQL data
- **Excel Configuration Support**: Validates Excel-based strategy configurations
- **Performance Requirements**: Maintains < 3-second processing targets
- **Multi-Index Support**: Handles NIFTY, BANKNIFTY, MIDCAPNIFTY, SENSEX data

### v6 Plan Integration

All generated tasks automatically integrate with the v6 refactoring plan:

- References existing v6 plan requirements
- Maintains compatibility with planned architecture
- Follows established development patterns
- Integrates with existing documentation structure

## Troubleshooting

### Common Issues and Solutions

1. **Persona Not Recognized**
   - Ensure persona name matches exact mapping (lowercase)
   - Check for typos in persona names
   - Use `--persona-` prefix in commands

2. **MCP Server Not Available**
   - Verify MCP server installation
   - Check network connectivity for remote servers
   - Use fallback servers if primary unavailable

3. **Task Creation Failures**
   - Verify TaskMaster AI installation
   - Check API key configuration
   - Ensure proper environment variables

4. **TODO File Splitting Issues**
   - Check file permissions
   - Verify file path accessibility
   - Ensure sufficient disk space

### Debug Commands

```bash
# Test persona mapping
python superclaude_taskmaster_integration.py create --command "/analyze --persona-architect"

# Validate configuration
python superclaude_taskmaster_integration.py validate

# Test TODO splitting
python superclaude_taskmaster_integration.py split --todo large_file.md
```

## Performance Metrics

Based on test results, the system demonstrates:

- **Command Parsing**: 100% accuracy for complex commands
- **Persona Mapping**: Complete coverage of all 9 personas
- **Task Generation**: Consistent task structure and description quality
- **TODO Management**: Automatic splitting maintains logical boundaries
- **Configuration**: Robust environment variable handling

## Future Enhancements

### Planned Improvements

1. **Dynamic Persona Selection**: Automatic persona selection based on task analysis
2. **Enhanced MCP Integration**: Deeper integration with specialized MCP servers
3. **Performance Monitoring**: Real-time performance metrics for persona effectiveness
4. **Custom Workflow Templates**: Pre-defined workflow patterns for common scenarios
5. **Cross-Project Learning**: Persona effectiveness learning across projects

### Contribution Guidelines

When extending the persona mapping system:

1. Maintain backward compatibility with existing mappings
2. Add comprehensive test coverage for new personas
3. Update documentation with new capabilities
4. Follow established naming conventions
5. Ensure integration with existing v6 plan requirements

---

*This guide provides comprehensive coverage of the SuperClaude persona to TaskMaster agent mapping system for autonomous AI-assisted development in the GPU Backtester platform.*