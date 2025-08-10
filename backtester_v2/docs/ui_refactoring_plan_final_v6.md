# 🚀 ULTIMATE Comprehensive UI Refactoring Plan v6.0 - Next.js 14+ Migration with SuperClaude Context Engineering

## Enterprise GPU Backtester: HTML/JavaScript → Next.js 14+ Complete Migration Plan

### Executive Summary

This Version 6.0 plan delivers a systematic migration of the Enterprise GPU Backtester from the current HTML/JavaScript implementation (`index_enterprise.html`) to Next.js 14+ while preserving 100% of existing SuperClaude context engineering capabilities and integrating modern UI technologies:

### 🔒 **CURRENT WORKTREE DEVELOPMENT PROTOCOL**

**DEVELOPMENT WORKTREE**: `/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/`

#### **Simplified Development Guidelines:**
1. **All codebase files are now within the current worktree** (ui-centralized)
2. **Complete isolation** - All required files have been copied from main branch:
   - Server files: `server/app/static/index_enterprise.html` (local copy)
   - Backend API: `backtester_v2/` (local copy)
   - Configurations: `backtester_v2/configurations/` (local copy)
   - All strategy implementations: `backtester_v2/strategies/` (local copy)
3. **Development workflow**:
   - Analyze existing implementation using local files
   - Create Next.js implementation in `nextjs-app/`
   - Document findings in `docs/`
   - Run tests in `tests/`
4. **No external dependencies** - Everything needed is within this worktree
5. **Dynamic TODO System Integration**: All TODOs generated and executed locally
6. **Real Data Validation**: Connect to actual databases from within current worktree

#### **Worktree Structure:**
```yaml
Current_Worktree:
  Path: "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/"
  Contains_Everything: "Complete codebase copy from main branch"
  
Local_Paths:
  Server_Files: "server/app/static/index_enterprise.html"
  Backend_API: "backtester_v2/"
  Configurations: "backtester_v2/configurations/"
  Strategies: "backtester_v2/strategies/"
  Tests: "tests/"
  Documentation: "docs/"
  Scripts: "scripts/"
  
Development_Areas:
  NextJS_App: "nextjs-app/"
  Documentation: "docs/"
  Testing: "tests/"
  Scripts: "scripts/"
  
Benefits:
  - "Complete isolation for UI refactoring"
  - "No cross-worktree dependencies"
  - "Simplified file access and testing"
  - "Full control over all components"
```

1. **Framework Migration**: HTML/JavaScript + Bootstrap → Next.js 14+ App Router with Server Components
2. **UI Technology Upgrade**: Bootstrap 5.3 + custom CSS → Tailwind CSS + shadcn/ui + Magic UI components
3. **SuperClaude Preservation**: Complete v1.0 context engineering framework maintained
4. **Performance Enhancement**: SSR/SSG/ISR, Server Components, edge optimization
5. **Live Trading Integration**: Zerodha & Algobaba with <1ms exchange latency
6. **Global Deployment**: Vercel multi-node with regional optimization
7. **Enterprise Features**: All 7 strategies, 13 navigation components, multi-node optimization, ML Training with Zone×DTE (5×10 grid), Pattern Recognition system, Triple Rolling Straddle implementation, correlation analysis & strike weighting
8. **Dynamic TODO System Integration**: Complete integration with intelligent task expansion, multi-agent coordination, and real data validation

### 🔄 **DYNAMIC TODO SYSTEM INTEGRATION**

This v6 plan is enhanced with the complete Dynamic TODO System featuring:

#### **Intelligent Task Expansion:**
- **V6 Plan Parser**: Automatically extracts all 13 phases, tasks, and SuperClaude commands
- **Dynamic TODO Generator**: Creates hierarchical TODO structures with validation criteria
- **Command Executor**: Analyzes SuperClaude command results for requirement discovery
- **Real Data Validation**: Enforces "no mock data" policy throughout development

#### **Multi-Agent Coordination:**
```yaml
Specialized_Agents:
  ANALYZER: "System analysis, requirement discovery, architectural assessment"
  AUTH_CORE: "Authentication systems, core infrastructure, security implementation"
  NAV_ERROR: "UI navigation, error handling, user experience, accessibility"
  STRATEGY: "Business logic implementation, strategy systems, data processing"
  INTEGRATION: "System integration, testing, validation, deployment coordination"

Agent_Assignment_Per_Phase:
  Phase_0: ["ANALYZER"]
  Phase_1: ["AUTH_CORE", "NAV_ERROR"]
  Phase_2: ["NAV_ERROR", "STRATEGY"]
  Phase_3: ["STRATEGY", "INTEGRATION"]
  Phase_7: ["INTEGRATION", "AUTH_CORE"]
  Phase_8: ["INTEGRATION"]
```

#### **Worktree-Aware Execution:**
- **Current Worktree Focus**: All implementation within `/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/`
- **Validation References**: Read-only access to main worktree for validation and compatibility
- **Real Data Integration**: Connections to actual HeavyDB (33M+ rows) and MySQL (28M+ rows) databases
- **Excel Configuration Validation**: Compatibility testing with 31 production Excel files

#### **SuperClaude Ecosystem Integration:**
- **9 Personas**: Architect, Frontend, Backend, Security, Performance, QA, ML, Refactorer, Mentor
- **4 MCP Servers**: Context7 (library docs), Sequential (complex analysis), Magic (UI components), Puppeteer (testing)
- **Context Engineering**: Automatic context loading with `--context:auto`, `--context:prd`, `--context:module` flags
- **Evidence-Based Development**: Comprehensive analysis and validation throughout implementation

### Technology Stack Migration Matrix

```yaml
Frontend_Framework:
  FROM: "HTML/JavaScript + Bootstrap + jQuery"
  TO: "Next.js 14+ App Router + Server Components + SSG/ISR"

UI_Libraries:
  FROM: "Bootstrap 5.3 + custom CSS + vanilla JavaScript"
  TO: "Tailwind CSS + shadcn/ui + Magic UI components"

Routing:
  FROM: "Single-page HTML with JavaScript navigation"
  TO: "Next.js App Router (file-based, SSR/CSR hybrid)"

State_Management:
  FROM: "Global JavaScript variables + localStorage"
  TO: "Zustand stores + TanStack Query for server state"

Data_Fetching:
  FROM: "Fetch API + XMLHttpRequest"
  TO: "Next.js fetch + Server Components + TanStack Query"

WebSocket:
  FROM: "Native WebSocket implementation"
  TO: "Next.js App Router compatible WebSocket with proper cleanup"
```

### SuperClaude v1.0 Context Engineering Preservation

**CRITICAL**: All SuperClaude context engineering capabilities from v5.0 are preserved and enhanced:

- **Context-Aware Commands**: All commands enhanced with --context:auto, --context:file, --context:module, --context:prd flags
- **9 Specialized Personas**: Context-optimized for Next.js development workflow
- **4 MCP Servers**: Context7, Sequential, Magic, Puppeteer with Next.js integration
- **Evidence-Based Development**: Required/prohibited language standards maintained
- **Performance Metrics**: Context relevance scoring and feedback loops (85%+ target)

### Reference Materials:
- **Context Engineering Guide**: https://github.com/IncomeStreamSurfer/context-engineering-intro
- **SuperClaude Framework**: https://github.com/NomenAK/SuperClaude
- **Next.js 14+ Documentation**: https://nextjs.org/docs

---

## Pre-Implementation Validation (MANDATORY)

### Validation Commands - Execute Before Migration
```bash
# CRITICAL: Comprehensive system state analysis (using local worktree files)
/analyze --persona-architect --seq --context:auto --context:file=@backtester_v2/strategies/** "Comprehensive analysis of all 7 strategy implementations, their current state, API endpoints, and integration points"

# Current HTML/JavaScript frontend analysis (using local copy)
/analyze --persona-frontend --magic --context:auto --context:file=@server/app/static/index_enterprise.html "Current HTML/JavaScript frontend analysis: DOM structure, JavaScript functionality, Bootstrap styling, and Next.js migration complexity assessment"

# Backend integration validation (all within current worktree)
/analyze --persona-backend --ultra --context:auto --context:module=@backtester_v2 "Backend structure validation: all strategies, configurations, and integration points within current worktree"

# UI-centralized worktree readiness for HTML to Next.js migration
/analyze --persona-architect --ultrathink --context:auto --context:file=@nextjs-app/** "UI-centralized worktree readiness assessment: HTML/JavaScript to Next.js migration preparation, component creation strategy, and modern framework integration"

# Source document integration analysis
/analyze --persona-architect --seq --context:auto --context:file=@docs/ui_refactoring_enhancment.md "Extract Next.js 14+ enhancements, new phases, and implementation requirements for HTML to Next.js migration"

# v5.0 structure validation for HTML migration context
/analyze --persona-frontend --magic --context:auto --context:file=@docs/ui_refactoring_plan_final_v5.md "Validate v5.0 structure, identify preservation requirements, and assess HTML to Next.js migration points"
```

---

## Phase 0: Context-Enhanced System Analysis & Migration Planning

### 🔒 **WORKTREE DEVELOPMENT PROTOCOL - Phase 0**
```yaml
Development_Scope: "ui-centralized worktree with complete codebase"
Local_Files:
  - "Server files: server/app/static/index_enterprise.html"
  - "Backend API: backtester_v2/"
  - "Configurations: backtester_v2/configurations/"
  - "All strategies: backtester_v2/strategies/"

Analysis_Strategy:
  - "Analyze existing implementation using local copies"
  - "Document findings within docs/"
  - "Create Next.js implementation within nextjs-app/"
  - "All validation using local files"

Development_Areas:
  - "Analysis outputs: docs/"
  - "Next.js implementation: nextjs-app/"
  - "Testing: tests/"
  - "Scripts and tools: scripts/"
  - "Real data connections for validation"
```

### Step 0.1: Context-Enhanced UI & Theme Analysis with Next.js Migration Context
```bash
# Context-aware analysis of current HTML/JavaScript implementation with Next.js migration planning
/analyze --persona-frontend --persona-designer --depth=3 --evidence --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@nextjs "index_enterprise.html current implementation analysis: HTML structure, JavaScript functionality, Bootstrap styling, DOM manipulation - Next.js Server/Client Component creation strategy"

# Deep dive into existing HTML/JavaScript architecture for Next.js conversion
/analyze --persona-frontend --ultra --c7 --context:auto --context:module=@ui --context:file=@server/app/static/index_enterprise.html "server/app/static/index_enterprise.html: actual theme colors, Bootstrap classes, JavaScript event handlers, DOM structure - Next.js App Router component architecture planning"

# Architecture analysis with PRD context and HTML to Next.js migration requirements
/analyze --persona-architect --seq --context:prd=@backtester_v2/ENTERPRISE_UI_ENHANCEMENT_COMPREHENSIVE_PLAN.md --context:auto --context:module=@nextjs "backtester_v2/: requirements, architecture - HTML/JavaScript to Next.js 14+ App Router migration strategy"

# CRITICAL: Complete feature inventory of current HTML/JavaScript implementation with Next.js compatibility
/analyze --persona-frontend --persona-analyzer --ultra --all-mcp --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@nextjs "index_enterprise.html complete feature inventory with Next.js migration assessment:
- Progress tracking system with ETA (HTML/JS → Server Components compatibility)
- WebSocket-based real-time updates (Native WebSocket → App Router WebSocket integration)
- Multi-strategy execution queue (JavaScript arrays → Next.js API routes integration)
- Result streaming and auto-navigation (JavaScript navigation → Next.js routing compatibility)
- Notification system for completion (DOM manipulation → Client Component requirements)
- Error handling and retry mechanisms (JavaScript try/catch → Next.js error boundaries)
- Batch execution capabilities (JavaScript loops → Server Actions integration)
- Resource monitoring during execution (JavaScript intervals → Real-time updates with Next.js)"

# Sequential analysis of current HTML/JavaScript features with Next.js enhancement opportunities
/analyze --persona-frontend --seq --context:auto --context:module=@ui --context:file=@server/app/static/index_enterprise.html "Current index_enterprise.html features with Next.js enhancement potential:
- Quick action buttons (HTML buttons → Server Actions integration)
- Keyboard shortcuts (JavaScript event listeners → Client Component interactivity)
- Context menus (JavaScript context menus → Next.js event handling)
- Drag-and-drop reordering (HTML5 drag/drop → Client Component state management)
- Multi-select operations (JavaScript selection → Zustand + Next.js compatibility)
- Bulk actions (JavaScript batch processing → Server Actions batch processing)
- Export queue functionality (JavaScript download → Next.js API routes)
- Session persistence (localStorage → Next.js middleware + storage)"

# Execution flow analysis of current HTML/JavaScript with Next.js optimization
/analyze --persona-frontend --chain --context:auto --context:file=@server/app/static/js/** --context:module=@nextjs "Current execution flow analysis with Next.js optimization:
- Start button → validation → queue → progress → completion → results (HTML/JS → Next.js flow)
- Progress indicators: percentage, time elapsed, ETA (DOM updates → Server Components + Client Components hybrid)
- Status updates via WebSocket (Native WebSocket → Next.js App Router WebSocket compatibility)
- Automatic result page navigation (JavaScript navigation → Next.js App Router navigation)
- Error recovery flows (JavaScript error handling → Next.js error boundaries and recovery)"

# Light theme color extraction from current implementation with Tailwind CSS migration planning
/analyze --persona-designer --evidence --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@tailwind "Light theme analysis from current HTML implementation for Tailwind CSS migration:
- White backgrounds (#ffffff) in HTML/CSS → Tailwind bg-white
- Blue accents (#0d6efd) in current styling → Tailwind custom blue-600
- Gray sidebar (#f8f9fa) in current CSS → Tailwind bg-gray-50
- Bootstrap classes in HTML → Tailwind utility mapping
- Current component styling → shadcn/ui component integration"

# Bootstrap to Tailwind migration complexity assessment from current HTML
/analyze --persona-frontend --persona-designer --ultra --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@tailwind "Bootstrap to Tailwind migration assessment from current HTML:
- Grid system migration (Current Bootstrap grid → Tailwind grid)
- Component classes migration (Current btn, card, form-control → Tailwind utilities)
- Responsive breakpoints alignment from current CSS
- Custom CSS preservation requirements from existing styles
- shadcn/ui component integration strategy for current HTML elements"
```

### Step 0.2: Context-Enhanced Backend Strategy Analysis with Next.js Integration Planning
```bash
# CRITICAL: Configuration system analysis with full context and Next.js API routes planning
/analyze --persona-backend --persona-architect --ultra --all-mcp --context:auto --context:module=@configurations --context:file=@nextjs/api/** "backtester_v2/configurations: Excel-based config system, parameter registry, deduplication, version control - Next.js API routes integration strategy"

# Production data analysis with context awareness and Server Components optimization
/analyze --persona-backend --depth=3 --evidence --context:auto --context:file=@backtester_v2/configurations/data/prod/** --context:module=@nextjs "backtester_v2/configurations/data/prod: tbs/, tv/, orb/, oi/, ml/, mr/, pos/, opt/ - Excel configurations with Next.js Server Components data fetching"

# Hot reload mechanism with context integration and Next.js Server Actions
/analyze --persona-frontend --persona-backend --seq --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@nextjs "index_enterprise.html: hot reload mechanism, dynamic Excel upload, real-time configuration updates - Next.js Server Actions and revalidation integration"

# Excel upload integration with Context7 documentation and Next.js file handling
/analyze --persona-backend --chain --c7 --context:auto --context:file=@backtester_v2/configurations/EXCEL_UPLOAD_INTEGRATION.md --context:module=@nextjs "backtester_v2/configurations/: integrate_existing_excel_upload.js, enhance_ui_dynamic_upload.js - Next.js App Router file upload strategy"

# Optimization refactoring with performance context and Next.js edge optimization
/analyze --persona-performance --persona-backend --ultra --context:auto --context:module=@optimization --context:file=@nextjs/edge/** "backtester_v2/strategies/optimization: algorithms/, engines/, gpu/, benchmarking/ - Next.js edge functions integration"

# TV strategy worktree with context-aware analysis and Next.js compatibility
/analyze --persona-backend --seq --context:auto --context:module=@strategy-tv --context:file=@nextjs/app/** "backtester_v2/strategies/tv: enhanced_yaml_converter, tv_parameter_manager, signal_processor, parallel_processor - Next.js Server/Client Components strategy"

# TV unified configuration with context integration and Next.js data fetching
/analyze --persona-backend --step --context:auto --context:file=@backtester_v2/strategies/tv/** --context:module=@nextjs "backtester_v2/strategies/tv: tv_unified_config.py, validate_config.py, Excel-to-YAML conversion - Next.js Server Components data integration"

# POS strategy with UI context and Next.js component architecture
/analyze --persona-backend --persona-frontend --chain --context:auto --context:module=@strategy-pos --context:file=@nextjs/components/** "backtester_v2/strategies/pos: enhanced models, risk management, adjustments/, golden format generation - Next.js component integration strategy"

# All strategies analysis with comprehensive context and Next.js App Router routing
/analyze --persona-backend --ultra --context:auto --context:module=@strategies --context:file=@nextjs/app/** "backtester_v2/strategies/: tbs/, tv/, orb/, oi/, ml_indicator/, indicator/, pos/, market_regime/, advanced/ - Next.js App Router page structure planning"

# Standard input parameter analysis with evidence and Next.js form handling
/analyze --persona-backend --evidence --context:auto --context:file=@backtester_v2/configurations/templates/** --context:module=@nextjs "All strategies use standard Excel input sheets: GeneralParameter, LegParameter, portfolio settings - Next.js Server Actions form handling integration"

# Configuration gateway and API analysis with Next.js API routes integration
/analyze --persona-backend --seq --context:auto --context:module=@gateway --context:file=@nextjs/api/** "backtester_v2/configurations/gateway/: API integration, parameter validation, configuration routing - Next.js API routes migration strategy"

# Version control and migration with context awareness and Next.js data persistence
/analyze --persona-backend --step --context:auto --context:file=@backtester_v2/configurations/version_control/** --context:module=@nextjs "backtester_v2/configurations/version_control/, backtester_v2/configurations/migration/: config versioning, backward compatibility - Next.js data persistence strategy"

# Search and indexing system with context integration and Next.js search optimization
/analyze --persona-backend --chain --context:auto --context:module=@search --context:file=@nextjs/api/search/** "backtester_v2/configurations/search/: configuration search, parameter indexing, quick lookup - Next.js search API and optimization"

# UI configuration components with Magic MCP and Next.js component migration
/analyze --persona-frontend --magic --context:auto --context:module=@ui --context:file=@nextjs/components/** "backtester_v2/configurations/ui/: existing UI components for configuration management - Next.js component migration and shadcn/ui integration"

# Parameter registry with comprehensive context and Next.js type safety
/analyze --persona-backend --ultra --context:auto --context:module=@parameter_registry --context:file=@nextjs/types/** "backtester_v2/configurations/parameter_registry/: centralized parameter management, validation rules, type definitions - Next.js TypeScript integration"

# Core configuration engine with context awareness and Next.js Server Components
/analyze --persona-backend --seq --context:auto --context:module=@core --context:file=@nextjs/lib/** "backtester_v2/configurations/core/: configuration loading, validation, hot reload implementation - Next.js Server Components and caching strategy"

# Strategy-specific parsers with context integration and Next.js API optimization
/analyze --persona-backend --step --context:auto --context:file=@backtester_v2/configurations/parsers/** --context:module=@nextjs "backtester_v2/configurations/parsers/: Excel parsers for each strategy type, YAML converters - Next.js API routes optimization"

# Hot reload mechanism analysis with full context and Next.js revalidation
/analyze --persona-frontend --persona-backend --ultra --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@nextjs "Hot reload implementation: WebSocket-based config updates, file watchers, dynamic UI refresh - Next.js revalidation and Server Actions integration"

# Complete backend architecture with context engineering and Next.js full-stack integration
/analyze --persona-architect --think-hard --context:auto --context:prd=@architecture/** --context:module=@nextjs "Complete backend system: configuration-driven, Excel-based, hot-reloadable, strategy patterns, optimization integration - Next.js full-stack architecture planning"

# CRITICAL: Excel Golden File Format with evidence-based analysis and Next.js file handling
/analyze --persona-backend --persona-qa --ultra --evidence --context:auto --context:file=@golden_format/** --context:module=@nextjs "Golden file format analysis: standardized Excel output structure, sheet organization, formatting rules, data validation - Next.js file generation and download optimization"

# Golden format generation with context awareness and Next.js Server Actions
/analyze --persona-backend --seq --context:auto --context:module=@golden_format --context:file=@nextjs/api/** "Golden format implementation:
- POS strategy: golden_format_generator.py, GOLDEN_FORMAT_SPECIFICATION.md
- TV strategy: golden_format_validator.py, test_golden_format_direct.py
- Standardized output sheets: Portfolio Trans, Results, Day-wise P&L, Month-wise P&L
- Next.js Server Actions integration for file generation"

# Excel output patterns with context integration and Next.js optimization
/analyze --persona-backend --chain --context:auto --context:file=@excel_output/** --context:module=@nextjs "Excel output patterns:
- excel_output_generator.py in each strategy
- Consistent formatting: uppercase headers, date/time formats, currency formatting
- Formula generation for results sheets
- Color coding and conditional formatting
- Next.js API routes for Excel generation and streaming"

# Logs page analysis with context-aware investigation and Next.js real-time updates
/analyze --persona-frontend --persona-backend --ultra --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@nextjs "Logs page analysis:
- Real-time log streaming via WebSocket (Next.js WebSocket compatibility)
- Log filtering by level (DEBUG, INFO, WARNING, ERROR) - Client Component interactivity
- Search functionality (Next.js search optimization)
- Export to CSV/Excel (Next.js API routes)
- Color-coded log entries (Tailwind CSS styling)
- Pagination for large logs (Next.js Server Components pagination)"

# Results page implementation with context integration and Next.js performance optimization
/analyze --persona-frontend --persona-performance --seq --context:auto --context:module=@results --context:file=@nextjs/components/** "Results page analysis:
- Interactive charts (Chart.js → Recharts migration for Next.js)
- Tabular data with DataTables (Next.js Server Components + Client Components)
- Export functionality (Excel, CSV, PDF) - Next.js API routes
- Performance metrics visualization (Next.js optimization)
- Comparison between strategies (Next.js state management)
- Drill-down capabilities (Next.js App Router navigation)"

# Log handling architecture with context awareness and Next.js streaming
/analyze --persona-backend --step --context:auto --context:module=@logging --context:file=@nextjs/api/** "Log architecture:
- Backend logging framework (Next.js API routes integration)
- WebSocket log streaming (Next.js WebSocket compatibility)
- Log persistence and rotation (Next.js data persistence)
- Frontend log viewer components (Next.js Client Components)
- Performance considerations for large logs (Next.js streaming and virtualization)"

# Results visualization with Magic MCP and Next.js component optimization
/analyze --persona-frontend --magic --context:auto --context:module=@visualization --context:file=@nextjs/components/** "Results visualization:
- P&L charts and curves (Recharts + Next.js Server Components)
- Trade distribution heatmaps (D3.js + Next.js Client Components)
- Risk metrics dashboards (Next.js dashboard optimization)
- Performance attribution (Next.js data visualization)
- Interactive filtering and zooming (Next.js state management)"
```

### Step 0.3: Context-Enhanced Next.js 14+ Project Setup & Worktree Analysis
```bash
# Next.js project initialization with context awareness and comprehensive setup
/implement --persona-frontend --persona-architect --magic --context:auto --context:prd=@migration_requirements.md --context:module=@nextjs "Next.js 14+ project setup:
- App Router configuration with layouts and route groups
- TypeScript strict mode integration
- Tailwind CSS setup with custom design tokens
- shadcn/ui installation and component library setup
- Magic UI components integration for enhanced animations
- ESLint and Prettier configuration for Next.js
- Environment variable setup for multi-environment deployment"

# Project structure design with context integration and Next.js best practices
/design --persona-architect --ultrathink --context:auto --context:module=@ui_centralized --context:file=@nextjs/app/** "Next.js project structure optimization:
- App Router directory structure with route groups
- Component organization (Server vs Client Components)
- API routes planning with proper HTTP methods
- Server Components strategy for performance
- Client Components identification for interactivity
- Middleware setup for authentication and routing
- Static asset optimization strategy"

# Dependency migration planning for HTML/JavaScript to Next.js transition
/analyze --persona-frontend --c7 --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@nextjs "Dependency migration analysis for HTML to Next.js:
- Current HTML/JavaScript dependencies assessment
- Bootstrap to Tailwind migration strategy
- State management creation (localStorage → Zustand + Next.js)
- WebSocket library integration with App Router
- Chart library selection and integration (Chart.js → Recharts)
- Form handling library integration (HTML forms → Next.js forms with validation)
- Authentication library selection (current auth → NextAuth.js)"

# Worktree structure analysis with HTML to Next.js integration planning
/analyze --persona-architect --ultra --context:auto --context:module=@worktrees --context:file=@nextjs/** "Comprehensive worktree analysis for HTML to Next.js migration:
- UI-centralized worktree readiness assessment for HTML migration
- Strategy-specific worktrees integration (strategy-tv, strategy-pos, strategy-market-regime)
- Consolidator-optimizer worktree Next.js compatibility
- Multi-worktree development workflow with Next.js
- Git worktree coordination for parallel Next.js development"

# Strategy worktree detailed analysis with Next.js component creation planning
/analyze --persona-backend --seq --context:auto --context:module=@strategies --context:file=@nextjs/components/** "Individual strategy analysis for HTML to Next.js:
- backtester_v2/strategies/tbs/: TBS strategy Next.js component creation
- backtester_v2/strategies/tv/: TV strategy Server/Client Components architecture
- backtester_v2/strategies/orb/: ORB strategy Next.js optimization
- backtester_v2/strategies/oi/: OI strategy data visualization components
- backtester_v2/strategies/ml_indicator/: ML Indicator Next.js integration
- backtester_v2/strategies/pos/: POS strategy real-time updates
- backtester_v2/strategies/market_regime/: Market Regime Next.js dashboard"

# Consolidator-optimizer worktree comprehensive analysis with Next.js performance optimization
/analyze --persona-performance --persona-backend --ultrathink --context:auto --context:module=@optimization --context:file=@nextjs/api/** "backtester_v2/strategies/optimization/ comprehensive analysis:
- Multi-node optimization architecture with Next.js
- Algorithm distribution with Next.js API routes
- Performance monitoring with Next.js analytics
- Resource allocation dashboard with Next.js
- Real-time optimization tracking with WebSocket + Next.js
- GPU utilization monitoring with Next.js Server Components"

# UI-centralized worktree migration readiness for HTML to Next.js transition
/analyze --persona-frontend --magic --context:auto --context:module=@ui-centralized --context:file=@nextjs/app/** "Current worktree migration readiness:
- HTML/JavaScript to Next.js component creation strategy
- Component library creation and organization
- State management implementation with Next.js
- Routing creation with Next.js App Router
- Asset optimization for Next.js deployment
- Performance benchmarking preparation"

# SuperClaude context engineering validation with Next.js integration
/analyze --persona-architect --seq --context:auto --context:prd=@Super_Claude_Docs_v1.md --context:module=@nextjs "SuperClaude v1.0 framework preservation:
- Context patterns migration to Next.js development
- Persona integration with App Router workflow
- MCP server compatibility with Next.js environment
- Evidence-based development standards for Next.js
- Performance metrics integration with Next.js Analytics
- Context engineering templates for Next.js components"

# Migration complexity assessment with risk analysis for HTML to Next.js
/analyze --persona-architect --persona-security --ultra --context:auto --context:module=@migration --context:file=@nextjs/** "HTML to Next.js migration complexity and risk assessment:
- HTML/JavaScript to component creation complexity scoring
- Data migration requirements and strategies
- Performance improvement opportunities with Next.js
- Security enhancement implications of Next.js migration
- Rollback procedures and contingency planning
- Timeline estimation for phased HTML to Next.js migration approach"
```

### Step 0.3: Context-Aware Migration Strategy
```bash
# Phased migration strategy with context engineering
/design --persona-architect --seq --context:auto --context:prd=@migration_strategy.md "Migration strategy design:
- Phase-by-phase component migration
- Parallel development approach
- Testing strategy for each phase
- Rollback procedures
- Performance benchmarking plan"

# Risk assessment and mitigation
/analyze --persona-security --persona-architect --ultra --context:auto --context:module=@risk_assessment "Migration risk analysis:
- Data loss prevention
- Service continuity assurance
- Security implications
- Performance regression risks
- User experience preservation"
```

---

## Phase 1: Context-Enhanced Light Theme Implementation & Next.js Foundation

### 🔒 **WORKTREE DEVELOPMENT PROTOCOL - Phase 1**
```yaml
Development_Scope: "ui-centralized worktree with complete codebase"
Authentication_Implementation: "Within nextjs-app/"
Local_References:
  - "Server authentication: server/app/main.py (local copy in worktree)"
  - "Current session management: server/app/api/ (local copy in worktree)"
  - "Excel configurations: backtester_v2/configurations/ (local copy in worktree)"
  - "Current HTML UI: server/app/static/index_enterprise.html (local copy in worktree)"

Implementation_Strategy:
  - "Analyze existing authentication from local server/"
  - "Implement NextAuth.js within nextjs-app/"
  - "Create API routes within nextjs-app/src/app/api/"
  - "Use local patterns as reference"

Core_Navigation_Scope:
  - "Analyze 13 sidebar items from server/app/static/index_enterprise.html (local copy)"
  - "Implement Next.js App Router navigation"
  - "Create layout.tsx and navigation components"
  - "All implementation within current worktree"

Development_Areas:
  - "Authentication: nextjs-app/src/app/(auth)/"
  - "Navigation: nextjs-app/src/components/navigation/"
  - "API Routes: nextjs-app/src/app/api/"
  - "Database connections using local configs"
```

### Step 1.1: Bootstrap-Based Theme System with Next.js Migration Context
```bash
# Extract theme with evidence-based analysis and Next.js Tailwind planning
/analyze --persona-designer --evidence --context:auto --context:file=@server/screenshots/** --context:module=@nextjs "Light theme analysis for Next.js migration: white backgrounds, blue accents (#0d6efd), gray sidebar (#f8f9fa) - Tailwind CSS variable mapping"

# Create theme configuration with Magic MCP and Next.js optimization
/implement --persona-frontend --magic --context:auto --context:file=@design-system/** --context:module=@nextjs "Next.js theme system:
- Primary: #0d6efd (Bootstrap blue) → Tailwind custom-blue-600
- Secondary: #6c757d (Bootstrap gray) → Tailwind gray-600
- Background: #ffffff (main), #f8f9fa (sidebar) → Tailwind bg-white, bg-gray-50
- Cards: white with subtle shadows → Tailwind shadow-sm
- Typography: -apple-system, BlinkMacSystemFont → Tailwind font-sans"

# Bootstrap to Tailwind integration with Context7 documentation
/implement --persona-frontend --c7 --context:auto --context:file=@package.json --context:module=@tailwind "Next.js Tailwind CSS integration:
- Bootstrap 5.3 → Tailwind CSS migration mapping
- Grid system conversion (Bootstrap grid → Tailwind grid)
- Component utilities migration
- Custom overrides preservation
- Next.js CSS optimization"

# Next.js 14+ App Router Foundation with theme integration
/implement --persona-frontend --magic --c7 --context:auto --context:file=@next.config.js --context:module=@nextjs "Next.js 14+ initialization with theme:
- App Router configuration with theme provider
- TypeScript strict mode integration
- Tailwind CSS setup with custom design tokens
- shadcn/ui component library with theme integration
- Magic UI components setup for enhanced animations
- CSS-in-JS optimization for Next.js"

# Simplified, scalable project structure implementation with enterprise architecture
/implement --persona-architect --step --context:auto --context:module=@project_structure --context:file=@nextjs/app/** "Next.js simplified enterprise architecture (60% complexity reduction):
enterprise-gpu-backtester/
├── src/
│   ├── app/                           # Next.js 14+ App Router (Production-Ready)
│   │   ├── (auth)/                    # Authentication route group
│   │   │   ├── login/
│   │   │   │   ├── page.tsx           # Login page (Server Component)
│   │   │   │   ├── loading.tsx        # Login loading state
│   │   │   │   └── error.tsx          # Login error boundary
│   │   │   ├── logout/
│   │   │   │   └── page.tsx           # Logout confirmation (Server Component)
│   │   │   ├── forgot-password/
│   │   │   │   └── page.tsx           # Password recovery (Server Component)
│   │   │   ├── reset-password/
│   │   │   │   └── page.tsx           # Password reset (Server Component)
│   │   │   └── layout.tsx             # Auth layout with theme
│   │   │
│   │   ├── (dashboard)/               # Main dashboard group
│   │   │   ├── page.tsx               # Dashboard home (Server Component)
│   │   │   ├── backtest/
│   │   │   │   ├── page.tsx           # Backtest interface (Hybrid)
│   │   │   │   ├── dashboard/
│   │   │   │   │   ├── page.tsx       # BT Dashboard (Client Component)
│   │   │   │   │   ├── loading.tsx    # BT Dashboard loading
│   │   │   │   │   └── error.tsx      # BT Dashboard error boundary
│   │   │   │   ├── results/[id]/
│   │   │   │   │   ├── page.tsx       # Results page (Server Component)
│   │   │   │   │   ├── loading.tsx    # Results loading
│   │   │   │   │   └── error.tsx      # Results error boundary
│   │   │   │   ├── loading.tsx        # Backtest loading
│   │   │   │   └── error.tsx          # Backtest error boundary
│   │   │   ├── live/
│   │   │   │   ├── page.tsx           # Live trading (Client Component)
│   │   │   │   ├── loading.tsx        # Live trading loading
│   │   │   │   └── error.tsx          # Live trading error boundary
│   │   │   ├── ml-training/
│   │   │   │   ├── page.tsx           # ML Training (Client Component)
│   │   │   │   ├── loading.tsx        # ML Training loading
│   │   │   │   └── error.tsx          # ML Training error boundary
│   │   │   ├── strategies/
│   │   │   │   ├── page.tsx           # Strategy management (Hybrid)
│   │   │   │   ├── [strategy]/
│   │   │   │   │   ├── page.tsx       # Individual strategy (Dynamic)
│   │   │   │   │   ├── loading.tsx    # Strategy loading
│   │   │   │   │   └── error.tsx      # Strategy error boundary
│   │   │   │   ├── loading.tsx        # Strategies loading
│   │   │   │   └── error.tsx          # Strategies error boundary
│   │   │   ├── logs/
│   │   │   │   ├── page.tsx           # Logs viewer (Client Component)
│   │   │   │   ├── loading.tsx        # Logs loading
│   │   │   │   └── error.tsx          # Logs error boundary
│   │   │   ├── templates/
│   │   │   │   ├── page.tsx           # Template gallery (Server Component)
│   │   │   │   ├── [templateId]/
│   │   │   │   │   ├── page.tsx       # Template details (Server Component)
│   │   │   │   │   ├── loading.tsx    # Template loading
│   │   │   │   │   └── error.tsx      # Template error boundary
│   │   │   │   ├── loading.tsx        # Templates loading
│   │   │   │   └── error.tsx          # Templates error boundary
│   │   │   ├── admin/
│   │   │   │   ├── page.tsx           # Admin dashboard (Server Component)
│   │   │   │   ├── users/
│   │   │   │   │   ├── page.tsx       # User management (Server Component)
│   │   │   │   │   ├── loading.tsx    # Users loading
│   │   │   │   │   └── error.tsx      # Users error boundary
│   │   │   │   ├── system/
│   │   │   │   │   ├── page.tsx       # System configuration (Hybrid)
│   │   │   │   │   ├── loading.tsx    # System loading
│   │   │   │   │   └── error.tsx      # System error boundary
│   │   │   │   ├── audit/
│   │   │   │   │   ├── page.tsx       # Audit logs (Server Component)
│   │   │   │   │   ├── loading.tsx    # Audit loading
│   │   │   │   │   └── error.tsx      # Audit error boundary
│   │   │   │   ├── layout.tsx         # Admin layout with RBAC
│   │   │   │   ├── loading.tsx        # Admin loading
│   │   │   │   └── error.tsx          # Admin error boundary
│   │   │   ├── settings/
│   │   │   │   ├── page.tsx           # User settings (Client Component)
│   │   │   │   ├── profile/
│   │   │   │   │   ├── page.tsx       # Profile management (Hybrid)
│   │   │   │   │   ├── loading.tsx    # Profile loading
│   │   │   │   │   └── error.tsx      # Profile error boundary
│   │   │   │   ├── preferences/
│   │   │   │   │   ├── page.tsx       # User preferences (Client Component)
│   │   │   │   │   ├── loading.tsx    # Preferences loading
│   │   │   │   │   └── error.tsx      # Preferences error boundary
│   │   │   │   ├── notifications/
│   │   │   │   │   ├── page.tsx       # Notification settings (Client Component)
│   │   │   │   │   ├── loading.tsx    # Notifications loading
│   │   │   │   │   └── error.tsx      # Notifications error boundary
│   │   │   │   ├── loading.tsx        # Settings loading
│   │   │   │   └── error.tsx          # Settings error boundary
│   │   │   ├── layout.tsx             # Dashboard layout with sidebar
│   │   │   ├── loading.tsx            # Dashboard loading
│   │   │   └── error.tsx              # Dashboard error boundary
│   │   │
│   │   ├── api/                       # API Routes (Production-Ready)
│   │   │   ├── auth/                  # Authentication API routes
│   │   │   │   ├── login/route.ts     # Login endpoint
│   │   │   │   ├── logout/route.ts    # Logout endpoint
│   │   │   │   ├── refresh/route.ts   # Token refresh
│   │   │   │   ├── session/route.ts   # Session validation
│   │   │   │   └── permissions/route.ts # Permission check
│   │   │   │
│   │   │   ├── strategies/
│   │   │   │   ├── route.ts           # Strategy CRUD
│   │   │   │   └── [id]/route.ts      # Individual strategy operations
│   │   │   │
│   │   │   ├── backtest/
│   │   │   │   ├── execute/route.ts   # Execute backtest
│   │   │   │   ├── results/route.ts   # Get results
│   │   │   │   ├── queue/route.ts     # Execution queue management
│   │   │   │   └── status/route.ts    # Execution status
│   │   │   │
│   │   │   ├── ml/
│   │   │   │   ├── training/route.ts  # ML training endpoints
│   │   │   │   ├── patterns/route.ts  # Pattern recognition
│   │   │   │   ├── models/route.ts    # ML model management
│   │   │   │   └── zones/route.ts     # Zone×DTE configuration
│   │   │   │
│   │   │   ├── live/
│   │   │   │   ├── route.ts           # Live trading endpoints
│   │   │   │   ├── orders/route.ts    # Order management
│   │   │   │   └── positions/route.ts # Position management
│   │   │   │
│   │   │   ├── configuration/         # Configuration management
│   │   │   │   ├── route.ts           # Config CRUD operations
│   │   │   │   ├── upload/route.ts    # Excel upload endpoint
│   │   │   │   ├── validate/route.ts  # Configuration validation
│   │   │   │   ├── hot-reload/route.ts # Hot reload endpoint
│   │   │   │   └── gateway/route.ts   # Configuration gateway
│   │   │   │
│   │   │   ├── optimization/          # Multi-node optimization
│   │   │   │   ├── route.ts           # Optimization CRUD
│   │   │   │   ├── nodes/route.ts     # Node management
│   │   │   │   ├── algorithms/route.ts # Algorithm selection
│   │   │   │   └── jobs/[jobId]/route.ts # Job management
│   │   │   │
│   │   │   ├── monitoring/            # Performance monitoring
│   │   │   │   ├── metrics/route.ts   # Performance metrics
│   │   │   │   ├── health/route.ts    # Health check endpoint
│   │   │   │   └── alerts/route.ts    # Alert management
│   │   │   │
│   │   │   ├── security/              # Security API routes
│   │   │   │   ├── audit/route.ts     # Security audit logs
│   │   │   │   └── rate-limit/route.ts # Rate limiting
│   │   │   │
│   │   │   └── websocket/route.ts     # WebSocket connections
│   │   │
│   │   ├── globals.css                # Global styles with theme variables
│   │   ├── layout.tsx                 # Root layout with theme provider
│   │   ├── page.tsx                   # Home (redirect to dashboard)
│   │   ├── loading.tsx                # Global loading UI
│   │   ├── error.tsx                  # Global error boundary
│   │   ├── not-found.tsx              # 404 page
│   │   ├── global-error.tsx           # Global error handler
│   │   ├── middleware.ts              # Authentication & routing middleware
│   │   ├── instrumentation.ts         # Performance instrumentation
│   │   ├── opengraph-image.tsx        # OpenGraph image generation
│   │   ├── robots.txt                 # SEO robots file
│   │   ├── sitemap.xml                # SEO sitemap
│   │   └── manifest.json              # PWA manifest
│   │
│   ├── components/                    # Simplified Component Structure
│   │   ├── ui/                        # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── form.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── layout/                    # Layout components
│   │   │   ├── Sidebar.tsx            # Main sidebar (13 navigation items)
│   │   │   ├── Header.tsx             # Header with user menu
│   │   │   ├── PageLayout.tsx         # Standard page wrapper
│   │   │   ├── Footer.tsx             # Footer component
│   │   │   └── LoadingOverlay.tsx     # Loading overlay component
│   │   │
│   │   ├── auth/                      # Authentication components
│   │   │   ├── LoginForm.tsx          # Login form component
│   │   │   ├── LogoutButton.tsx       # Logout component
│   │   │   ├── AuthProvider.tsx       # Auth context provider
│   │   │   ├── ProtectedRoute.tsx     # Route protection
│   │   │   ├── SessionTimeout.tsx     # Session management
│   │   │   └── RoleGuard.tsx          # Role-based access control
│   │   │
│   │   ├── error/                     # Error handling components
│   │   │   ├── ErrorBoundary.tsx      # Custom error boundary
│   │   │   ├── ErrorFallback.tsx      # Error fallback UI
│   │   │   ├── RetryButton.tsx        # Retry functionality
│   │   │   ├── ErrorLogger.tsx        # Error logging
│   │   │   └── ErrorNotification.tsx  # Error notifications
│   │   │
│   │   ├── loading/                   # Loading components
│   │   │   ├── LoadingSpinner.tsx     # Loading spinner
│   │   │   ├── SkeletonLoader.tsx     # Skeleton loading
│   │   │   ├── ProgressBar.tsx        # Progress indicator
│   │   │   └── LoadingOverlay.tsx     # Loading overlay
│   │   │
│   │   ├── charts/                    # Financial charts (TradingView optimized)
│   │   │   ├── TradingChart.tsx       # Main trading chart component
│   │   │   ├── PnLChart.tsx           # P&L visualization
│   │   │   ├── MLHeatmap.tsx          # Zone×DTE heatmap
│   │   │   └── CorrelationMatrix.tsx  # Correlation analysis
│   │   │
│   │   ├── trading/                   # Trading components
│   │   │   ├── BacktestRunner.tsx     # Backtest execution
│   │   │   ├── BacktestDashboard.tsx  # BT Dashboard with queue management
│   │   │   ├── ExecutionQueue.tsx     # Execution queue component
│   │   │   ├── ProgressTracker.tsx    # Progress tracking component
│   │   │   ├── LiveTradingPanel.tsx   # Live trading interface
│   │   │   ├── StrategySelector.tsx   # Strategy selection
│   │   │   ├── ResultsViewer.tsx      # Results display
│   │   │   ├── OrderManager.tsx       # Order management
│   │   │   └── PositionTracker.tsx    # Position tracking
│   │   │
│   │   ├── ml/                        # ML components
│   │   │   ├── MLTrainingDashboard.tsx # ML training interface
│   │   │   ├── PatternDetector.tsx     # Pattern recognition
│   │   │   ├── TripleStraddleAnalyzer.tsx # Triple straddle analysis
│   │   │   └── ZoneDTEGrid.tsx         # Zone×DTE configuration
│   │   │
│   │   ├── strategies/                # Strategy components (Plugin-ready)
│   │   │   ├── StrategyCard.tsx       # Strategy display card
│   │   │   ├── StrategyConfig.tsx     # Strategy configuration
│   │   │   ├── StrategyRegistry.tsx   # Dynamic strategy loading
│   │   │   └── implementations/       # Current 7 strategies
│   │   │       ├── TBSStrategy.tsx
│   │   │       ├── TVStrategy.tsx
│   │   │       ├── ORBStrategy.tsx
│   │   │       ├── OIStrategy.tsx
│   │   │       ├── MLIndicatorStrategy.tsx
│   │   │       ├── POSStrategy.tsx
│   │   │       └── MarketRegimeStrategy.tsx
│   │   │
│   │   ├── configuration/             # Configuration management components
│   │   │   ├── ConfigurationManager.tsx # Main config manager
│   │   │   ├── ExcelValidator.tsx     # Excel validation
│   │   │   ├── ParameterEditor.tsx    # Parameter editing
│   │   │   ├── ConfigurationHistory.tsx # Config version history
│   │   │   ├── HotReloadIndicator.tsx # Hot reload status
│   │   │   └── ConfigurationGateway.tsx # Config gateway interface
│   │   │
│   │   ├── optimization/              # Multi-node optimization components
│   │   │   ├── MultiNodeDashboard.tsx # Node management dashboard
│   │   │   ├── NodeMonitor.tsx        # Node monitoring
│   │   │   ├── LoadBalancer.tsx       # Load balancing controls
│   │   │   ├── AlgorithmSelector.tsx  # Algorithm selection
│   │   │   ├── OptimizationQueue.tsx  # Optimization queue
│   │   │   ├── PerformanceMetrics.tsx # Performance monitoring
│   │   │   ├── ConsolidatorDashboard.tsx # Consolidator interface
│   │   │   └── BatchProcessor.tsx     # Batch processing
│   │   │
│   │   ├── monitoring/                # Performance monitoring components
│   │   │   ├── PerformanceDashboard.tsx # Performance dashboard
│   │   │   ├── MetricsViewer.tsx      # Metrics visualization
│   │   │   ├── AlertManager.tsx       # Alert management
│   │   │   ├── HealthIndicator.tsx    # Health indicators
│   │   │   └── AnalyticsTracker.tsx   # Analytics tracking
│   │   │
│   │   ├── templates/                 # Template management components
│   │   │   ├── TemplateGallery.tsx    # Template browser
│   │   │   ├── TemplatePreview.tsx    # Template preview
│   │   │   ├── TemplateUpload.tsx     # Template upload
│   │   │   └── TemplateEditor.tsx     # Template editing
│   │   │
│   │   ├── admin/                     # Admin components
│   │   │   ├── UserManagement.tsx     # User management
│   │   │   ├── SystemConfiguration.tsx # System configuration
│   │   │   ├── AuditViewer.tsx        # Audit log viewer
│   │   │   └── SecuritySettings.tsx   # Security settings
│   │   │
│   │   ├── logs/                      # Log management components
│   │   │   ├── LogViewer.tsx          # Real-time log display
│   │   │   ├── LogFilter.tsx          # Log filtering
│   │   │   ├── LogExporter.tsx        # Log export
│   │   │   └── LogSearch.tsx          # Log search functionality
│   │   │
│   │   └── forms/                     # Form components
│   │       ├── ExcelUpload.tsx        # Excel configuration upload
│   │       ├── ParameterForm.tsx      # Strategy parameters
│   │       ├── ValidationDisplay.tsx  # Configuration validation
│   │       ├── AdvancedForm.tsx       # Advanced form controls
│   │       └── FormValidation.tsx     # Form validation utilities
│   │
│   ├── lib/                           # Utilities and configuration
│   │   ├── api/                       # API clients
│   │   │   ├── strategies.ts          # Strategy API client
│   │   │   ├── backtest.ts            # Backtest API client
│   │   │   ├── ml.ts                  # ML API client
│   │   │   ├── websocket.ts           # WebSocket client
│   │   │   ├── auth.ts                # Authentication API client
│   │   │   ├── configuration.ts       # Configuration API client
│   │   │   ├── optimization.ts        # Optimization API client
│   │   │   ├── monitoring.ts          # Monitoring API client
│   │   │   └── admin.ts               # Admin API client
│   │   │
│   │   ├── stores/                    # Zustand stores
│   │   │   ├── strategy-store.ts      # Strategy state
│   │   │   ├── backtest-store.ts      # Backtest state
│   │   │   ├── ml-store.ts            # ML training state
│   │   │   ├── ui-store.ts            # UI state (sidebar, theme)
│   │   │   ├── auth-store.ts          # Authentication state
│   │   │   ├── configuration-store.ts # Configuration state
│   │   │   ├── optimization-store.ts  # Optimization state
│   │   │   └── monitoring-store.ts    # Monitoring state
│   │   │
│   │   ├── hooks/                     # Custom hooks
│   │   │   ├── use-websocket.ts       # WebSocket hook
│   │   │   ├── use-strategy.ts        # Strategy management hook
│   │   │   ├── use-real-time-data.ts  # Real-time data hook
│   │   │   ├── use-auth.ts            # Authentication hook
│   │   │   ├── use-configuration.ts   # Configuration management hook
│   │   │   ├── use-optimization.ts    # Optimization hook
│   │   │   ├── use-monitoring.ts      # Performance monitoring hook
│   │   │   └── use-error-handling.ts  # Error handling hook
│   │   │
│   │   ├── utils/                     # Utility functions
│   │   │   ├── excel-parser.ts        # Excel configuration parsing
│   │   │   ├── strategy-factory.ts    # Strategy creation factory
│   │   │   ├── performance-utils.ts   # Performance calculations
│   │   │   ├── auth-utils.ts          # Authentication utilities
│   │   │   ├── validation-utils.ts    # Validation utilities
│   │   │   ├── error-utils.ts         # Error handling utilities
│   │   │   ├── monitoring-utils.ts    # Monitoring utilities
│   │   │   └── security-utils.ts      # Security utilities
│   │   │
│   │   ├── config/                    # Configuration
│   │   │   ├── strategies.ts          # Strategy registry configuration
│   │   │   ├── charts.ts              # Chart configuration
│   │   │   ├── api.ts                 # API configuration
│   │   │   ├── auth.ts                # Authentication configuration
│   │   │   ├── security.ts            # Security configuration
│   │   │   ├── monitoring.ts          # Monitoring configuration
│   │   │   └── optimization.ts        # Optimization configuration
│   │   │
│   │   └── theme/                     # Theme configuration and utilities
│   │
│   └── types/                         # TypeScript types
│       ├── strategy.ts                # Strategy-related types
│       ├── backtest.ts                # Backtest-related types
│       ├── ml.ts                      # ML-related types
│       ├── api.ts                     # API response types
│       ├── auth.ts                    # Authentication types
│       ├── configuration.ts           # Configuration types
│       ├── optimization.ts            # Optimization types
│       ├── monitoring.ts              # Monitoring types
│       └── error.ts                   # Error types"

# Core dependencies installation with optimized financial libraries
/implement --persona-frontend --step --c7 --context:auto --context:module=@nextjs "Core dependencies with enterprise trading optimization:
- next@14+ (with CSS optimization and edge functions)
- react@18
- typescript
- tailwindcss (with custom financial theme)
- @shadcn/ui (themed components)
- @magic-ui/react (with theme integration)
- @tanstack/react-query (for server state management)
- zustand (with real-time trading state)
- socket.io-client (for WebSocket connections)
- tradingview-charting-library (recommended for financial charts)
- lightweight-charts (alternative for performance-critical charts)
- next-themes (for theme switching)
- zod (for form validation and API type safety)
- framer-motion (for smooth animations)"

# TradingView chart integration with performance optimization
/implement --persona-frontend --magic --context:auto --context:module=@charts --context:file=@nextjs/components/charts/** "TradingView chart integration for enterprise trading:
- TradingView Charting Library: Superior performance for financial data
- Real-time data streaming: <50ms update latency
- Financial indicators: EMA, VWAP, Greeks, P&L curves
- Mobile responsiveness: Native-like touch interactions
- Bundle size optimization: 450KB vs 2.5MB+ with complex alternatives
- Professional trading interface: Industry-standard charting solution"

# Plugin-ready strategy architecture with exponential scalability
/implement --persona-architect --ultrathink --context:auto --context:module=@strategies --context:file=@nextjs/components/strategies/** "Plugin-ready strategy architecture:
- Dynamic strategy loading: Support for unlimited strategy addition
- Strategy registry pattern: Configuration-driven component rendering
- Hot-swappable components: Runtime strategy updates
- Standardized interfaces: Consistent API across all strategies
- Performance optimization: Lazy loading and code splitting
- Future-proof design: Research-driven feature integration pathways"
```

### Step 1.2: Phased Implementation Strategy with SuperClaude Context Engineering

#### **Phase 1: Core Migration (Weeks 1-4) - Immediate Value Delivery**
```bash
# Authentication & security foundation with context engineering
/implement --persona-security --persona-architect --ultra --context:auto --context:module=@auth_foundation --context:file=@nextjs/app/(auth)/** "Phase 1 authentication foundation:
- Complete authentication system with NextAuth.js integration
- Role-based access control (RBAC) for enterprise trading system
- Security middleware for route protection and session management
- Authentication API routes with JWT token handling
- Login/logout pages with error boundaries and loading states
- Multi-factor authentication preparation for admin access"

# Core dashboard implementation with context engineering
/implement --persona-frontend --magic --context:auto --context:module=@core_migration --context:file=@nextjs/app/(dashboard)/** "Phase 1 core migration with immediate value:
- Dashboard layout with 13 sidebar navigation items (complete coverage)
- Basic strategy selection and execution interface
- Results visualization with TradingView charts integration
- Excel configuration upload with hot reload
- WebSocket integration for real-time updates
- Simplified strategy registry for current 7 strategies
- Error boundaries and loading states for all routes"

# Missing navigation routes implementation
/implement --persona-frontend --step --context:auto --context:module=@navigation_completion --context:file=@nextjs/app/(dashboard)/** "Complete 13 sidebar navigation implementation:
- BT Dashboard: Interactive backtest dashboard with execution queue
- Logs: Real-time log viewer with filtering and export capabilities
- Templates: Template gallery with preview and upload functionality
- Admin: User management, system configuration, and audit logs
- Settings: User preferences, profile management, and notifications
- Error handling and loading states for all navigation routes"

# Strategy registry implementation with plugin architecture
/implement --persona-architect --step --context:auto --context:module=@strategy_registry --context:file=@nextjs/lib/config/** "Simplified strategy implementation:
const StrategyRegistry = {
  TBS: () => import('./strategies/TBSStrategy'),
  TV: () => import('./strategies/TVStrategy'),
  ORB: () => import('./strategies/ORBStrategy'),
  OI: () => import('./strategies/OIStrategy'),
  MLIndicator: () => import('./strategies/MLIndicatorStrategy'),
  POS: () => import('./strategies/POSStrategy'),
  MarketRegime: () => import('./strategies/MarketRegimeStrategy')
};"

# Performance-optimized chart integration
/implement --persona-frontend --persona-performance --magic --context:auto --context:module=@charts --context:file=@nextjs/components/charts/** "TradingView chart integration with <100ms UI updates:
- Optimized chart component pattern with memo and useMemo
- Real-time data management with efficient WebSocket integration
- Financial indicators: EMA 200/100/20, VWAP, Greeks, P&L curves
- Mobile-responsive touch interactions for trading interfaces
- Bundle size optimization: 450KB vs alternatives 2.5MB+"
```

#### **Phase 2: ML & Analytics (Weeks 5-8) - Advanced Capabilities**
```bash
# Configuration management system with context engineering
/implement --persona-backend --persona-frontend --seq --context:auto --context:module=@configuration_system --context:file=@nextjs/components/configuration/** "Configuration management implementation:
- ConfigurationManager: Main config interface with Excel upload and validation
- HotReloadIndicator: Real-time configuration update status
- ParameterEditor: Dynamic parameter editing with validation
- ConfigurationGateway: API gateway for configuration operations
- Excel validation and YAML conversion with error handling"

# ML Training components with Zone×DTE implementation
/implement --persona-ml --persona-frontend --ultra --context:auto --context:module=@ml_components --context:file=@nextjs/components/ml/** "ML Training components implementation:
- ZoneDTEHeatmap: Zone×DTE (5×10 grid) visualization with Server Components
- PatternDetector: Rejection candles, EMA, VWAP detection with Client Components
- CorrelationMatrix: Cross-strike correlation analysis with real-time updates
- TripleStraddleAnalyzer: Triple rolling straddle implementation with WebSocket"

# Pattern recognition system with TensorFlow.js integration
/implement --persona-ml --magic --context:auto --context:module=@pattern_recognition --context:file=@nextjs/lib/ml/** "Pattern recognition system:
- ML model integration with TensorFlow.js and WebWorkers
- Real-time pattern detection with confidence scoring
- Pattern recognition alerts with Next.js notifications
- Historical pattern analysis with SSG/ISR optimization"

# Performance monitoring and analytics implementation
/implement --persona-performance --persona-frontend --chain --context:auto --context:module=@monitoring_system --context:file=@nextjs/components/monitoring/** "Performance monitoring system:
- PerformanceDashboard: Real-time performance metrics visualization
- MetricsViewer: Trading performance and system health metrics
- AlertManager: Configurable alerts for trading and system events
- HealthIndicator: System health status with real-time updates
- Analytics tracking for user behavior and system performance"
```

#### **Phase 3: Advanced Features (Weeks 9-12) - Enterprise Scale**
```bash
# Multi-node optimization and consolidator integration
/implement --persona-performance --persona-backend --ultrathink --context:auto --context:module=@optimization_system --context:file=@nextjs/components/optimization/** "Multi-node optimization system:
- MultiNodeDashboard: Node management with load balancing controls
- ConsolidatorDashboard: 8-format processing pipeline with batch operations
- AlgorithmSelector: 15+ optimization algorithms with performance tracking
- OptimizationQueue: Job queue management with real-time progress
- NodeMonitor: Resource utilization and failover management
- PerformanceMetrics: Optimization performance analytics and reporting"

# Advanced enterprise features implementation
/implement --persona-backend --persona-frontend --ultra --context:auto --context:module=@advanced_features --context:file=@nextjs/components/** "Advanced enterprise features:
- LiveTrading: Real-time trading with Zerodha integration and <1ms latency
- AdvancedAnalytics: Performance attribution and risk analysis with Server Components
- PluginSystem: Dynamic strategy loading for exponential future growth
- SecurityAudit: Comprehensive security monitoring and audit logging
- BackupRecovery: Automated backup and disaster recovery systems"

# Real-time performance optimization patterns
/implement --persona-performance --seq --context:auto --context:module=@performance --context:file=@nextjs/lib/hooks/** "Performance optimization implementation:
- useRealTimeData hook: Efficient WebSocket data management
- useStrategy hook: Strategy state management with Zustand
- useErrorHandling hook: Comprehensive error handling and recovery
- Chart optimization: Memoized components with <100ms update latency
- Bundle splitting: Dynamic imports for strategy components
- Performance monitoring: Real-time performance tracking and optimization"

# Production deployment and DevOps integration
/implement --persona-architect --persona-security --chain --context:auto --context:module=@production_deployment --context:file=@docker/** "Production deployment preparation:
- Docker containerization with multi-stage builds
- Kubernetes manifests for scalable deployment
- CI/CD pipeline with automated testing and security scanning
- Environment configuration for development, staging, and production
- Monitoring and alerting integration with enterprise systems
- Security hardening and compliance validation"
```

### Step 1.3: Context-Aware Component Library with Next.js Optimization
```bash
# Enterprise component library with performance optimization
/implement --persona-frontend --magic --evidence --context:auto --context:file=@server/app/index_enterprise.html --context:module=@nextjs "Simplified enterprise component library:
- InstrumentSelector: Search dropdown with Server Components data fetching
- FileUploadCard: Drag-drop with Next.js Server Actions and Excel parsing
- GPUMonitorPanel: Real-time GPU stats with Client Components and WebSocket
- StrategyCard: White cards with blue headers using shadcn/ui
- TradingChart: TradingView integration with <50ms update latency
- MLHeatmap: Zone×DTE (5×10 grid) visualization with interactive Client Components"

# Scalable strategy component pattern with plugin architecture
/implement --persona-architect --step --context:auto --context:module=@strategy_components --context:file=@nextjs/components/strategies/** "Plugin-ready strategy components:
- StrategyCard: Unified display component for all strategies
- StrategyConfig: Dynamic configuration interface
- StrategyRegistry: Runtime strategy loading and registration
- StrategyFactory: Component creation with lazy loading
- Performance optimization: Code splitting and dynamic imports
- Future-proof design: Supports unlimited strategy addition"

# Form components with enterprise validation and Next.js optimization
/implement --persona-frontend --step --context:auto --context:module=@forms --context:file=@nextjs/components/forms/** "Enterprise form components:
- ExcelUpload: Drag-drop with validation and hot reload
- ParameterForm: Dynamic strategy parameter configuration
- ValidationDisplay: Real-time configuration validation
- Form validation with Zod and Next.js Server Actions
- Error handling with Next.js error boundaries
- Performance optimization with debounced validation"

# Financial chart integration with TradingView optimization
/implement --persona-frontend --persona-performance --magic --context:auto --context:file=@components/charts/** --context:module=@tradingview "TradingView chart integration:
- TradingChart: Main financial chart with professional indicators
- PnLChart: P&L visualization with real-time updates
- MLHeatmap: Zone×DTE heatmap with interactive features
- CorrelationMatrix: Cross-strike correlation analysis
- Performance: <50ms update latency for real-time trading
- Mobile optimization: Native-like touch interactions"

# Theme system with financial trading optimization
/implement --persona-designer --persona-frontend --magic --context:auto --context:file=@design_system/** --context:module=@nextjs "Financial trading theme system:
- Bootstrap → Tailwind CSS migration with financial color palette
- Light theme preservation: #f8f9fa sidebar, #0d6efd primary
- Professional trading interface styling
- Real-time data visualization colors
- Responsive design for trading dashboards
- Animation system with Magic UI for smooth transitions"

# Performance-optimized hooks and utilities
/implement --persona-performance --step --context:auto --context:module=@hooks --context:file=@nextjs/lib/hooks/** "Performance-optimized custom hooks:
- useRealTimeData: Efficient WebSocket data management
- useStrategy: Strategy state management with Zustand
- useWebSocket: WebSocket connection with automatic reconnection
- useTradingChart: Chart optimization with memoization
- Performance target: <100ms UI updates for trading requirements"
```

---

## Phase 2: Context-Enhanced Sidebar Implementation & Next.js Navigation (13 Items)

### Step 2.1: Context-Aware Sidebar with Actual Styling & Next.js App Router
```bash
# Implement sidebar with evidence-based design and Next.js App Router integration
/implement --persona-frontend --magic --context:auto --context:file=@server/app/static/index_enterprise.html --context:module=@nextjs "Next.js sidebar implementation:
- Light gray background (#f8f9fa) → Tailwind bg-gray-50
- 13 menu items with icons using shadcn/ui
- Active state: blue text (#0d6efd) → Tailwind text-blue-600
- Hover: light blue background → Tailwind hover:bg-blue-50
- Collapsible with animation using Magic UI
- Next.js App Router Link components for navigation"

# All 13 sidebar components with context integration and Next.js routing
/implement --persona-frontend --chain --context:auto --context:module=@navigation --context:file=@nextjs/app/** "Next.js sidebar navigation items:
1. 📊 Start New Backtest → /backtest/new (includes TV Strategy & Parallel Tests) - Server/Client Components
2. 🏠 Overview → /dashboard (simple, clean metrics) - Server Components with real-time Client updates
3. 📈 BT Dashboard → /backtest/dashboard - Client Components for interactivity
4. 💹 Live Trading → /live (Market Regime + Triple Rolling Straddle) - Real-time Client Components
5. 📊 Results → /results - Server Components with Client visualization
6. 📝 Logs → /logs - Client Components for real-time log streaming
7. 🧠 ML Training → /ml-training - Client Components for interactive ML interface
8. 📁 Templates → /templates - Server Components with Client interactions
9. ⚡ Parallel Tests → (Integrated in Start New Backtest) - Client Components
10. 🔧 Strategy Management → /strategies (includes Consolidator & Optimizer) - Hybrid Server/Client
11. 👤 Admin → /admin - Server Components with Client controls
12. ⚙️ Settings → /settings - Client Components for user preferences"

# Navigation state management with Next.js App Router
/implement --persona-frontend --step --context:auto --context:module=@stores --context:file=@nextjs/lib/** "Next.js navigation state:
- Zustand store with App Router compatibility
- Active item tracking with usePathname
- Breadcrumb trail generation
- User permissions with Server Components
- Collapsed state with localStorage persistence
- Theme state integration"

# Sidebar component architecture with Next.js optimization
/implement --persona-frontend --magic --context:auto --context:module=@sidebar --context:file=@nextjs/components/** "Next.js sidebar architecture:
- Server Component for static sidebar structure
- Client Component for interactive elements
- shadcn/ui navigation components
- Magic UI animations for transitions
- Responsive design with Tailwind
- Accessibility features with proper ARIA labels"
```

### Step 2.2: Context-Enhanced Navigation Architecture & Component Migration
```bash
# Next.js App Router setup with comprehensive navigation
/implement --persona-frontend --seq --c7 --context:auto --context:file=@routing/** --context:module=@nextjs "Next.js App Router navigation:
- File-based routing with route groups
- Nested layouts for different sections
- Loading.tsx for each route segment
- Error.tsx boundaries for error handling
- Not-found.tsx for 404 pages
- Metadata configuration for SEO"

# Layout system implementation with sidebar integration
/implement --persona-frontend --seq --context:auto --context:module=@layouts --context:file=@nextjs/app/** "Next.js layout system:
- Root layout with theme provider and sidebar
- Dashboard layout group with metrics sidebar
- Trading layout group with trading-specific navigation
- Auth layout group with centered forms
- Loading states with Suspense boundaries
- Error boundaries with recovery options"

# HTML/JavaScript to Next.js component creation strategy
/implement --persona-frontend --magic --context:auto --context:module=@components --context:file=@nextjs/migration/** "HTML to Next.js component creation strategy:
- Server Components for static content and data fetching
- Client Components for interactive features and state
- Hybrid rendering optimization with composition
- State management creation with Zustand (replacing global variables)
- Event handling conversion from JavaScript to Next.js patterns
- Performance optimization with Server/Client boundaries"

# Navigation components with Next.js optimization
/implement --persona-frontend --chain --context:auto --context:file=@navigation/** --context:module=@nextjs "Navigation components:
- Breadcrumb component with App Router integration
- Tab navigation with Next.js Link optimization
- Dropdown menus with shadcn/ui components
- Search navigation with Server Actions
- Quick actions with keyboard shortcuts
- Mobile navigation with responsive design"
```

### Step 2.3: Context-Enhanced Server Components Implementation
```bash
# Server Components for performance optimization and SEO
/implement --persona-performance --persona-frontend --ultra --context:auto --context:module=@server_components --context:file=@nextjs/app/** "Server Components implementation:
- Dashboard metrics (server-rendered with real-time hydration)
- Strategy listings (SSG with ISR for performance)
- Configuration data (server-side fetching with caching)
- Static content optimization with Next.js
- SEO enhancement with metadata API"

# Client Components for interactivity and real-time features
/implement --persona-frontend --magic --context:auto --context:module=@client_components --context:file=@nextjs/components/** "Client Components:
- Real-time data displays with WebSocket integration
- Interactive charts with Recharts and user interactions
- Form handling with Next.js Server Actions and validation
- WebSocket connections for live updates
- State management with Zustand and React Query"

# Hybrid rendering strategy for HTML to Next.js migration with performance optimization
/design --persona-architect --ultrathink --context:auto --context:prd=@rendering_strategy.md --context:module=@nextjs "Hybrid rendering design for HTML migration:
- Server Components for initial load and SEO (replacing static HTML)
- Client Components for interactivity and real-time features (replacing JavaScript)
- Progressive enhancement from current HTML/JavaScript
- Streaming and Suspense for better UX than current implementation
- Performance optimization with code splitting (improving on current single HTML file)"

# Next.js API routes implementation for backend integration
/implement --persona-backend --seq --context:auto --context:module=@api_routes --context:file=@nextjs/app/api/** "API routes migration:
- app/api/strategies/[strategy]/route.ts with proper HTTP methods
- app/api/backtest/route.ts with streaming support
- app/api/optimization/route.ts with WebSocket integration
- app/api/websocket/route.ts for real-time connections
- Middleware integration for auth and CORS"

# HeavyDB proxy integration with Next.js optimization
/implement --persona-backend --ultra --context:auto --context:file=@lib/heavydb-proxy.ts --context:module=@nextjs "HeavyDB proxy with Next.js:
- Connection pooling with Next.js runtime optimization
- Query optimization with caching strategies
- Error handling with Next.js error boundaries
- Performance monitoring with Next.js analytics
- Caching strategy with Next.js revalidation"
```

---

## Phase 3: Context-Enhanced Strategy Integration (All 7 Strategies)

### 🔒 **WORKTREE VALIDATION PROTOCOL - Phase 3**
```yaml
Development_Scope: "ui-centralized worktree with complete codebase"
Strategy_Implementation: "Within current worktree nextjs-app/src/app/strategies/"
Local_References:
  - "Strategy implementations: backtester_v2/strategies/ (local copy in worktree)"
  - "Excel configurations: backtester_v2/configurations/data/production/ (local copy in worktree)"
  - "API endpoints: server/app/api/routes/ (local copy in worktree)"
  - "Database schemas: HeavyDB and MySQL (connection from current worktree)"

Strategy_Migration_Scope:
  - "Analyze all 7 strategies from local backtester_v2/strategies/"
  - "TBS, TV, ORB, OI, ML_INDICATOR, POS, MARKET_REGIME strategies"
  - "Excel parameter parsing implementation within current worktree"
  - "Next.js API routes within current worktree nextjs-app/src/app/api/strategies/"
  - "Strategy UI components within current worktree"

Excel_Configuration_Validation:
  - "Use Excel schemas from local backtester_v2/configurations/"
  - "Implement Excel parsing within current worktree using pandas equivalent"
  - "Validate parameter extraction against main worktree patterns"
  - "NO modifications to production Excel files during development"

Database_Integration_Strategy:
  - "Validate connection patterns from main worktree"
  - "Implement HeavyDB and MySQL connections within current worktree"
  - "Real data validation using actual databases"
  - "API compatibility with existing backend (validation only, implementation independent)"

Strict_Boundaries:
  - "ALL strategy code within current worktree nextjs-app/"
  - "ALL Excel parsing within current worktree utilities/"
  - "ALL database connections within current worktree API routes"
  - "NO modifications to main worktree strategy implementations"
  - "NO changes to production Excel configurations"
```

### Step 3.1: Context-Aware TBS Strategy Migration
```bash
# TBS strategy Next.js integration
/implement --persona-frontend --persona-backend --magic --context:auto --context:module=@strategies/tbs "TBS Next.js migration:
- Server Component for configuration display
- Client Component for form interactions
- API route integration (/api/strategies/tbs)
- Excel upload with Next.js file handling
- Real-time validation with Server Actions"

# TBS API routes with context integration
/implement --persona-backend --step --context:auto --context:file=@app/api/strategies/tbs/** "TBS API implementation:
- POST /api/strategies/tbs/execute
- GET /api/strategies/tbs/status
- WebSocket progress tracking
- Error handling and validation"
```

### Step 3.2: Context-Enhanced TV Strategy (Integrated in Start New Backtest)
```bash
# TV Strategy Next.js integration
/implement --persona-frontend --chain --context:auto --context:module=@strategies/tv "TV Strategy migration:
- Integrated within Start New Backtest page
- Server Component for signal data display
- Client Component for interactive configuration
- File upload with Next.js App Router
- TradingView integration preservation"

# TV specific features with Magic UI
/implement --persona-frontend --magic --context:auto --context:file=@strategies/tv/** "TV features enhancement:
- Signal visualization with Recharts
- Portfolio selection with Magic UI
- Rollover settings with shadcn/ui forms
- Real-time updates with WebSocket"
```

### Step 3.3: Context-Enhanced ORB Strategy
```bash
# ORB strategy Next.js migration
/implement --persona-frontend --magic --context:auto --context:module=@strategies/orb "ORB Next.js integration:
- Opening range visualization with Recharts
- Time range selectors with shadcn/ui
- Breakout indicators with Magic UI
- Server Component for historical data
- Client Component for real-time updates"
```

### Step 3.4: Context-Enhanced OI Strategy
```bash
# OI strategy with ML persona integration
/implement --persona-frontend --persona-ml --magic --context:auto --context:module=@strategies/oi "OI Next.js migration:
- Open interest heatmaps with D3.js
- Strike selection UI with shadcn/ui
- Server Component for data aggregation
- Client Component for interactive features
- Real-time updates with WebSocket"
```

### Step 3.5: Context-Enhanced ML Indicator Strategy
```bash
# ML Indicator comprehensive migration
/implement --persona-ml --persona-frontend --ultra --context:auto --context:module=@strategies/ml_indicator "ML Indicator Next.js:
- 200+ indicator selection with Server Components
- Feature engineering UI with Client Components
- Model configuration with shadcn/ui forms
- Performance visualization with Recharts
- Real-time model updates"
```

### Step 3.6: Context-Enhanced POS Strategy
```bash
# POS strategy Next.js integration
/implement --persona-frontend --magic --context:auto --context:module=@strategies/pos "POS Next.js migration:
- Greeks display dashboard with Server Components
- Position management with Client Components
- Risk metrics visualization with Recharts
- Real-time updates with WebSocket
- Interactive controls with Magic UI"
```

### Step 3.7: Context-Enhanced Market Regime Strategy
```bash
# Market Regime comprehensive migration
/implement --persona-ml --persona-frontend --ultra --context:auto --context:module=@strategies/market_regime "Market Regime Next.js:
- 18-regime classification grid with Server Components
- Correlation matrices with D3.js
- Regime transition tracking with Client Components
- Real-time regime detection
- Interactive visualization with Magic UI"
```

---

## Phase 4: Context-Enhanced ML Training & Triple Rolling Straddle with Next.js

### Step 4.1: Context-Aware ML Training with Zone×DTE & Next.js Integration
```bash
# ML Training with comprehensive context and Next.js Server/Client Components
/implement --persona-ml --persona-frontend --magic --ultra --context:auto --context:module=@ml_triple_rolling_straddle_system --context:file=@nextjs/app/ml-training/** "ML Training UI with Next.js:
- Zone×DTE heatmap (5×10 grid) with Server Components data fetching
- Pattern forming visualization with Client Components interactivity
- Model performance tracking with real-time updates
- Connect to /backtester_v2/ml_triple_rolling_straddle_system/ via Next.js API routes
- WebSocket integration for real-time training progress"

# Zone configuration with context integration and Next.js optimization
/implement --persona-ml --step --context:auto --context:file=@ml_training/zones/** --context:module=@nextjs "Zone×DTE config with Next.js:
- 5 zones: OPEN, MID_MORN, LUNCH, AFTERNOON, CLOSE (Server Components)
- 10 DTE ranges: 0-5 individual, 6-10, 11-20, 21-30, 31+ (Static generation)
- Performance color coding with Tailwind CSS
- Interactive zone selection with Client Components
- Real-time performance updates with WebSocket"

# ML Training dashboard with Next.js performance optimization
/implement --persona-ml --magic --context:auto --context:module=@ml_dashboard --context:file=@nextjs/components/** "ML Training dashboard:
- Training progress visualization with Recharts
- Model accuracy metrics with Server Components
- Real-time loss curves with Client Components
- Hyperparameter tuning interface with shadcn/ui
- Model comparison tools with Next.js optimization"
```

### Step 4.2: Context-Enhanced Triple Rolling Straddle Patterns with Next.js
```bash
# Pattern detection with context awareness and Next.js real-time updates
/implement --persona-ml --magic --context:auto --context:module=@pattern_detection --context:file=@nextjs/components/patterns/** "Pattern detection with Next.js:
- Rejection candle indicators with real-time Client Components
- EMA 200/100/20 visualization with Recharts integration
- Pivot support/resistance with interactive charts
- VWAP bounce detection with WebSocket updates
- Pattern recognition alerts with Next.js notifications"

# Correlation analysis with context integration and Next.js optimization
/implement --persona-ml --chain --context:auto --context:file=@correlation/** --context:module=@nextjs "Correlation UI with Next.js:
- Cross-strike correlation matrix with Server Components data
- Call/Put analysis dashboard with Client Components
- Real-time correlation tracking with WebSocket
- Alert system for breakdowns with Next.js notifications
- Historical correlation analysis with SSG/ISR"

# Strike weighting with context-aware visualization and Next.js
/implement --persona-ml --step --context:auto --context:module=@strike_weighting --context:file=@nextjs/components/strikes/** "Strike weights with Next.js:
- ATM (50%) visual indicator with Magic UI components
- ITM1 (30%) visual indicator with animated progress
- OTM1 (20%) visual indicator with real-time updates
- 10-component breakdown with interactive charts
- Weight adjustment interface with Server Actions"

# Triple Rolling Straddle strategy implementation with Next.js
/implement --persona-ml --ultra --context:auto --context:module=@triple_straddle --context:file=@nextjs/app/strategies/triple-straddle/** "Triple Rolling Straddle with Next.js:
- Strategy configuration with Server Components
- Real-time P&L tracking with Client Components
- Position management with WebSocket updates
- Risk metrics dashboard with Recharts
- Automated rolling logic with Next.js API routes"

# Pattern recognition system with Next.js machine learning integration
/implement --persona-ml --magic --context:auto --context:module=@pattern_recognition --context:file=@nextjs/lib/ml/** "Pattern recognition with Next.js:
- ML model integration with TensorFlow.js
- Real-time pattern detection with WebWorkers
- Pattern confidence scoring with Server Components
- Historical pattern analysis with SSG
- Pattern-based alerts with Next.js notifications"
```

---

## Phase 5: Context-Enhanced Live Trading Integration

### Step 4.1: Context-Aware Zerodha Integration
```bash
# Zerodha integration with Next.js
/implement --persona-backend --persona-security --ultra --context:auto --context:module=@brokers/zerodha "Zerodha Next.js integration:
- TOTP automation with Server Actions
- KiteConnect API integration
- Session management with Next.js middleware
- Real-time WebSocket feeds
- Order placement with validation"

# Zerodha API routes implementation
/implement --persona-backend --step --context:auto --context:file=@app/api/brokers/zerodha/** "Zerodha API routes:
- POST /api/brokers/zerodha/login
- POST /api/brokers/zerodha/order
- GET /api/brokers/zerodha/positions
- WebSocket /api/ws/zerodha/feed"
```

### Step 4.2: Context-Enhanced Algobaba Integration
```bash
# Algobaba multi-URL failover with Next.js
/implement --persona-backend --persona-performance --chain --context:auto --context:module=@brokers/algobaba "Algobaba Next.js integration:
- Multi-URL failover logic
- Circuit breaker implementation
- Portfolio routing with Next.js
- Retry logic with exponential backoff
- Performance monitoring"

# Algobaba API routes with context
/implement --persona-backend --ultra --context:auto --context:file=@app/api/brokers/algobaba/** "Algobaba API implementation:
- POST /api/brokers/algobaba/order
- GET /api/brokers/algobaba/status
- Circuit breaker monitoring
- Failover management"
```

### Step 4.3: Context-Enhanced Real-time Data Integration
```bash
# WebSocket integration with Next.js App Router
/implement --persona-frontend --magic --context:auto --context:module=@websocket "WebSocket Next.js integration:
- Custom useWebSocket hook
- Server-side WebSocket handling
- Real-time data streaming
- Connection management
- Error recovery"

# Live data feed components
/implement --persona-frontend --magic --context:auto --context:file=@components/trading/LiveDataFeed.tsx "Live data components:
- Magic UI Marquee for price tickers
- Animated Beam for data flow
- Real-time chart updates
- WebSocket status indicators
- Performance optimization"
```

---

## Phase 5: Context-Enhanced UI Enhancement with Magic UI

### Step 5.1: Context-Aware Magic UI Integration
```bash
# Magic UI components implementation
/implement --persona-frontend --magic --c7 --context:auto --context:module=@magic_ui "Magic UI integration:
- Marquee for live price feeds
- Orbiting Circles for portfolio visualization
- Animated Beam for data flow
- Border Beam for active strategies
- Magic Card for strategy displays
- Particles for background effects"

# Strategy dashboard with Magic UI
/implement --persona-frontend --persona-designer --magic --context:auto --context:file=@components/dashboard/StrategyDashboard.tsx "Strategy dashboard enhancement:
- Magic Card for strategy containers
- Orbiting Circles for position visualization
- Particles background effects
- Smooth animations and transitions
- Interactive hover effects"
```

### Step 5.2: Context-Enhanced Chart Migration (Chart.js → Recharts)
```bash
# Chart library migration with context awareness
/implement --persona-frontend --persona-performance --seq --context:auto --context:module=@charts "Chart migration:
- Chart.js to Recharts conversion
- Performance optimization
- Server Component integration
- Interactive features preservation
- Real-time data binding"

# Advanced visualization with D3.js
/implement --persona-frontend --magic --context:auto --context:file=@components/charts/** "Advanced charts:
- D3.js for complex visualizations
- Correlation matrices
- Heatmaps for regime analysis
- Interactive network graphs
- Performance-optimized rendering"
```

---

## Phase 6: Context-Enhanced Multi-Node Architecture & Optimization

### Step 6.1: Context-Aware Multi-Node Dashboard
```bash
# Node management with Next.js Server Components
/implement --persona-frontend --persona-devops --magic --context:auto --context:module=@node_management "Node dashboard Next.js:
- Server Component for node status display
- Client Component for interactive controls
- Real-time monitoring with WebSocket
- Load distribution visualization
- Failover management interface"

# Multi-node optimization UI
/implement --persona-performance --persona-frontend --seq --context:auto --context:module=@optimization "Multi-node optimization Next.js:
- Algorithm selection with shadcn/ui
- Node assignment controls
- Real-time convergence charts with Recharts
- Performance metrics dashboard
- Resource allocation visualization"
```

### Step 6.2: Context-Enhanced Strategy Management with Consolidator & Optimizer (Complete Implementation)
```bash
# CRITICAL: Comprehensive Strategy Management with context and Next.js
/implement --persona-backend --persona-performance --ultrathink --context:auto --context:module=@strategy_management --context:file=@nextjs/app/strategies/** "Strategy Management comprehensive Next.js implementation:
- Consolidator: 8-format processing pipeline with Server Actions
- Optimizer: 15+ algorithms with multi-node support
- Batch processing with Next.js queue management
- Performance analytics dashboard with Server Components
- Real-time optimization monitoring with WebSocket"

# Strategy Consolidator UI with context awareness and Next.js
/implement --persona-frontend --magic --context:auto --context:module=@consolidator --context:file=@nextjs/components/consolidator/** "Consolidator UI components with Next.js:
- File format detection and conversion with Server Actions
- Excel/CSV/YAML pipeline visualization with Client Components
- Validation status indicators with real-time updates
- Batch upload interface with Next.js file handling
- Progress tracking per file with WebSocket
- Error handling and recovery with Next.js error boundaries"

# Strategy Optimizer UI with context integration and Next.js
/implement --persona-frontend --persona-performance --seq --context:auto --context:module=@optimizer --context:file=@nextjs/components/optimizer/** "Optimizer UI with multi-node Next.js:
- Algorithm selection matrix with shadcn/ui components
- Node assignment controls with real-time status
- Optimization objective configuration with Server Actions
- Real-time convergence charts with Recharts
- Pareto front visualization (3D) with Client Components
- Performance comparison grid with Server Components data"

# Multi-node optimization endpoints with context and Next.js API routes
/implement --persona-backend --ultra --context:auto --context:file=@app/api/optimization/** --context:module=@nextjs "Optimization API endpoints with Next.js:
- POST /api/optimization/start - Start optimization job with validation
- GET /api/optimization/status/[jobId] - Job status with real-time updates
- POST /api/optimization/node/allocate - Allocate node resources
- GET /api/optimization/node/metrics - Node performance metrics
- POST /api/optimization/algorithm/configure - Configure algorithms
- GET /api/optimization/results/[jobId] - Get results with streaming
- POST /api/optimization/cancel/[jobId] - Cancel job with cleanup
- GET /api/optimization/history - Optimization history with pagination"

# WebSocket channels with context awareness and Next.js
/implement --persona-backend --chain --context:auto --context:module=@websocket --context:file=@nextjs/lib/websocket/** "Optimization WebSocket channels with Next.js:
- /ws/optimization/progress - Real-time progress updates
- /ws/optimization/metrics - Performance metrics stream
- /ws/optimization/convergence - Convergence data streaming
- /ws/optimization/node/status - Node status updates
- Next.js WebSocket integration with proper cleanup"

# Strategy Management dashboard with context integration and Next.js
/implement --persona-frontend --magic --context:auto --context:module=@dashboard --context:file=@nextjs/app/strategies/dashboard/** "Strategy Management dashboard with Next.js:
- Overview: Active optimizations, consolidations (Server Components)
- Resource usage: CPU/GPU per node (real-time Client Components)
- Queue status: Pending/Running/Completed (WebSocket updates)
- Quick actions: New optimization, view results (Server Actions)
- Recent activity feed (SSG with ISR)
- Performance trends (historical data with charts)"

# Templates & Admin with Next.js integration
/implement --persona-frontend --magic --context:auto --context:module=@templates --context:file=@nextjs/app/admin/** "Templates & Admin with Next.js:
- Template Library: Searchable gallery with Server Components
- Preview functionality with Client Components
- Download management with Next.js API routes
- Version control with Git integration
- Admin Panel: User management with Server Components
- System configuration with Server Actions
- Audit logs with real-time updates
- Permission controls with RBAC"
```

---

## Phase 7: Context-Enhanced Integration & Testing

### 🔒 **WORKTREE VALIDATION PROTOCOL - Phase 7**
```yaml
Development_Scope: "ui-centralized worktree with complete codebase"
Testing_Implementation: "Within current worktree tests/"
Local_References:
  - "API endpoints: server/app/api/ (local copy in worktree)"
  - "Backend services: backtester_v2/ (local copy in worktree)"
  - "Database schemas: HeavyDB and MySQL (real data testing from current worktree)"
  - "Excel configurations: backtester_v2/configurations/ (local copy in worktree)"

Integration_Testing_Strategy:
  - "Test Next.js implementation against main worktree API interfaces"
  - "Validate database connectivity from current worktree"
  - "Test Excel parsing compatibility with production configurations"
  - "End-to-end testing within current worktree using real data"

API_Compatibility_Testing:
  - "Validate Next.js API routes match main worktree FastAPI interfaces"
  - "Test request/response compatibility without modifying main worktree"
  - "WebSocket integration testing using actual backend connections"
  - "Authentication flow testing against existing session management"

Real_Data_Testing_Requirements:
  - "MANDATORY: Use actual HeavyDB and MySQL databases"
  - "MANDATORY: Use production Excel configurations for validation"
  - "PROHIBITED: Mock data usage in any testing scenario"
  - "Test against 33M+ rows HeavyDB data and 28M+ rows MySQL archive"

Strict_Boundaries:
  - "ALL tests within current worktree tests/"
  - "ALL test implementations within current worktree"
  - "NO modifications to main worktree during testing"
  - "NO changes to production databases or configurations"
```

### Step 7.1: Context-Aware API Integration
```bash
# Complete API integration with Next.js
/implement --persona-backend --ultra --context:auto --context:module=@api_integration "API Integration Next.js:
- All /backtester_v2/api/ endpoints
- Next.js API routes as proxy layer
- WebSocket protocol preservation
- Error handling with Next.js
- Performance monitoring integration"

# State management with Next.js compatibility
/implement --persona-frontend --seq --context:auto --context:module=@state_management "State Management Next.js:
- Zustand stores with App Router compatibility
- Server state with TanStack Query
- WebSocket state synchronization
- Offline capability with Next.js
- State persistence with Next.js storage"
```

### Step 7.2: Context-Enhanced Comprehensive Testing
```bash
# Testing strategy for Next.js migration
/test --persona-qa --coverage=95 --all-mcp --context:auto --context:file=@tests/** "Next.js testing strategy:
- Unit tests for Server/Client Components
- Integration tests with API routes
- E2E tests with Playwright
- Performance benchmarks (HTML/JavaScript vs Next.js)
- Migration validation tests"

# Strategy-specific testing with context
/test --persona-qa --chain --context:auto --context:module=@strategies "Strategy testing Next.js:
- All 7 strategies functionality validation
- Excel configuration system testing
- WebSocket reliability testing
- Multi-node optimization testing
- Performance regression testing"
```

---

## Phase 8: Context-Enhanced Deployment & Production

### 🔒 **WORKTREE VALIDATION PROTOCOL - Phase 8**
```yaml
Development_Scope: "ui-centralized worktree with complete codebase"
Deployment_Implementation: "From current worktree nextjs-app/"
Local_References:
  - "Production environment: http://173.208.247.17:8000 (compatibility validation)"
  - "Deployment configs: server/ (local copy in worktree)"
  - "Database connections: HeavyDB and MySQL (production validation from current worktree)"
  - "Excel configurations: backtester_v2/configurations/ (local copy in worktree)"

Deployment_Strategy:
  - "Deploy Next.js application from current worktree"
  - "Validate compatibility with existing production backend"
  - "Test against actual production databases"
  - "Ensure zero disruption to current production system"

Production_Validation_Requirements:
  - "Test against actual production HeavyDB (33M+ rows)"
  - "Validate Excel configuration compatibility"
  - "Test WebSocket connections to existing backend"
  - "Verify API compatibility without backend modifications"

Environment_Configuration:
  - "Configure environment variables within current worktree"
  - "Database connection strings for actual production databases"
  - "API endpoints pointing to existing production backend"
  - "WebSocket configuration for existing infrastructure"

Strict_Boundaries:
  - "ALL deployment artifacts from current worktree"
  - "ALL configuration within current worktree .env files"
  - "NO modifications to existing production backend"
  - "NO changes to production database configurations"
  - "NO disruption to current production system"
```

### Step 8.1: Context-Aware Vercel Deployment
```bash
# Vercel multi-node deployment setup
/deploy --persona-devops --plan --context:auto --context:prd=@deployment_requirements.md "Vercel deployment Next.js:
- Multi-region deployment (Mumbai, Singapore, Frankfurt)
- Edge Functions for global optimization
- HeavyDB proxy configuration
- CDN setup for static assets
- Environment variable management"

# Production optimization
/optimize --persona-performance --ultra --context:auto --context:module=@production "Production optimization:
- Server Components optimization
- Static generation (SSG) configuration
- Incremental Static Regeneration (ISR)
- Edge caching strategies
- Performance monitoring setup"
```

### Step 8.2: Context-Enhanced Monitoring & Analytics
```bash
# Production monitoring with Next.js
/implement --persona-performance --persona-devops --seq --context:auto --context:module=@monitoring "Production monitoring:
- Next.js Analytics integration
- Performance monitoring with Vercel
- Error tracking with Sentry
- Real-time metrics dashboard
- Alert system configuration"

# Security implementation
/implement --persona-security --ultra --context:auto --context:module=@security "Security Next.js:
- Next.js security headers
- API route protection
- Authentication with NextAuth.js
- CSRF protection
- Rate limiting implementation"
```

---

### Step 6.3: Context-Enhanced Dashboard & Live Trading with Next.js
```bash
# Minimalist dashboard with context-aware design and Next.js optimization
/design --persona-frontend --persona-designer --magic --context:auto --context:prd=@dashboard_requirements.md --context:module=@nextjs "Clean dashboard design with Next.js:
- Key metrics only (P&L, win rate, active strategies) with Server Components
- Card-based layout with breathing room using shadcn/ui
- Subtle animations with Magic UI components
- Quick action buttons with Server Actions
- Recent activity feed (5 items max) with SSG/ISR
- Performance spark lines with Recharts optimization"

# Dashboard components with context integration and Next.js
/implement --persona-frontend --step --context:auto --context:module=@dashboard --context:file=@nextjs/components/dashboard/** "Dashboard components with Next.js:
- MetricCard: Large number, trend indicator (Server Components)
- ActivityFeed: Recent 5 items with icons (SSG with real-time updates)
- QuickActions: Start backtest, view results (Server Actions)
- PerformanceSpark: Mini charts without axes (Client Components)
- SystemHealth: Simple status dots (real-time WebSocket)"

# Dashboard state management with context awareness and Next.js
/implement --persona-frontend --seq --context:auto --context:file=@stores/dashboard.ts --context:module=@nextjs "Dashboard state with Next.js:
- Real-time updates via WebSocket with proper cleanup
- Efficient data aggregation with Server Components
- Local caching for performance with Next.js optimization
- Smooth transitions with Magic UI animations"

# Live Trading with Market Regime & Straddle Analysis and Next.js
/implement --persona-frontend --persona-ml --ultra --context:auto --context:module=@live_trading --context:file=@nextjs/app/live/** "Live trading with Next.js:
- Market regime detection panel with real-time Client Components
- Triple rolling straddle analysis with WebSocket updates
- Multi-symbol support (NIFTY, BANKNIFTY, FINNIFTY) with Server Components
- Zerodha integration panel with secure API handling
- Real-time Greeks display with optimized rendering"

# Straddle chart analyzer with Next.js integration
/implement --persona-frontend --magic --context:auto --context:module=@straddle_analyzer --context:file=@nextjs/components/trading/** "Straddle analyzer with Next.js:
- Interactive straddle chart with Recharts and real-time data
- Triple rolling straddle overlay with Client Components
- ATM/ITM1/OTM1 strike tracking with WebSocket
- Real-time premium updates with optimized rendering
- Pattern detection indicators with ML integration"
```

---

## Phase 9: Context-Enhanced Missing Elements Implementation with Next.js

### Step 9.1: Context-Aware Authentication Integration (msg99) with Next.js
```bash
# Authentication system analysis with security context and Next.js
/analyze --persona-security --persona-backend --ultra --seq --context:auto --context:module=@auth --context:file=@nextjs/middleware/** "msg99 authentication with Next.js: existing implementation, API endpoints, token management, session handling, Next.js middleware integration"

# Authentication architecture design with context integration and Next.js
/design --persona-architect --persona-security --magic --context:auto --context:prd=@auth_requirements.md --context:module=@nextjs "Authentication architecture with Next.js:
- msg99 OAuth integration with NextAuth.js
- JWT token management with Next.js middleware
- Session persistence with Next.js cookies
- Role-based access control (RBAC) with Server Components
- Multi-factor authentication support with Next.js"

# Authentication components with context awareness and Next.js
/implement --persona-frontend --persona-security --chain --context:auto --context:module=@auth --context:file=@nextjs/app/(auth)/** "Authentication UI components with Next.js:
- LoginPage.tsx with msg99 integration and Server Actions
- AuthProvider context for global auth state with Next.js
- ProtectedRoute wrapper component with middleware
- SessionTimeout handler with Next.js optimization
- RememberMe functionality with secure cookies"

# Backend authentication middleware with context integration and Next.js
/implement --persona-backend --persona-security --ultra --context:auto --context:file=@middleware/auth.ts --context:module=@nextjs "Authentication middleware with Next.js:
- JWT verification middleware with Next.js runtime
- Role-based route protection with App Router
- Session management with Redis and Next.js
- Token refresh mechanism with API routes
- Logout across all devices with Next.js optimization"

# Security headers and CORS with context awareness and Next.js
/implement --persona-security --seq --context:auto --context:module=@security --context:file=@nextjs/middleware.ts "Security configuration with Next.js:
- CORS policy for msg99 domains with Next.js middleware
- Security headers (CSP, HSTS, X-Frame-Options) with Next.js
- Rate limiting for auth endpoints with Next.js
- Brute force protection with Redis
- IP whitelisting for admin routes with middleware"
```

### Step 9.2: Context-Enhanced Security Implementation with Next.js
```bash
# Comprehensive security audit with context integration and Next.js
/scan --security --owasp --deps --strict --all-mcp --context:auto --context:module=@security --context:file=@nextjs/** "Full security scan with Next.js: OWASP Top 10, dependency vulnerabilities, configuration issues, Next.js specific security"

# Security best practices with context awareness and Next.js
/implement --persona-security --ultrathink --context:auto --context:prd=@security_requirements.md --context:module=@nextjs "Security implementation with Next.js:
- Input validation and sanitization with Zod
- SQL injection prevention with parameterized queries
- XSS protection with DOMPurify and Next.js CSP
- CSRF tokens for state-changing operations with Next.js
- Secure cookie configuration with Next.js"

# API security layer with context integration and Next.js
/implement --persona-backend --persona-security --magic --context:auto --context:module=@api_security --context:file=@nextjs/app/api/** "API security with Next.js:
- API key management with Next.js environment variables
- Request signing with crypto utilities
- Payload encryption for sensitive data
- API versioning strategy with Next.js routing
- Deprecation warnings with Next.js headers"

# Data encryption with context awareness and Next.js
/implement --persona-security --seq --context:auto --context:file=@encryption/** --context:module=@nextjs "Data protection with Next.js:
- Encryption at rest (AES-256) with Node.js crypto
- Encryption in transit (TLS 1.3) with Next.js deployment
- Key rotation mechanism with Next.js API routes
- Secure key storage with environment variables
- PII data masking with Next.js middleware"

# Security monitoring with context integration and Next.js
/implement --persona-security --persona-performance --chain --context:auto --context:module=@monitoring --context:file=@nextjs/lib/monitoring/** "Security monitoring with Next.js:
- Intrusion detection system with Next.js logging
- Anomaly detection for user behavior with analytics
- Security event logging with structured logging
- Real-time alerts for suspicious activities with WebSocket
- Integration with SIEM tools via Next.js API routes"
```

---

## Phase 10: Context-Enhanced Documentation & Knowledge Transfer with Next.js

### Step 10.1: Context-Aware Technical Documentation with Next.js
```bash
# Comprehensive technical documentation with Next.js focus
/document --persona-architect --depth=3 --context:auto --context:module=@documentation --context:file=@nextjs/docs/** "Technical documentation with Next.js:
- Next.js architecture diagrams with component boundaries
- Migration guide documentation (HTML/JavaScript → Next.js)
- API documentation with OpenAPI and Next.js routes
- Component library documentation with Storybook
- Performance optimization guide with Next.js best practices"

# User documentation with context and Next.js
/document --persona-mentor --examples --context:auto --context:file=@user_guides/** --context:module=@nextjs "User documentation with Next.js:
- Next.js UI user guide with screenshots
- Strategy configuration tutorials with step-by-step guides
- Live trading setup guide with Next.js features
- Troubleshooting documentation with common issues
- Video walkthrough creation with Next.js demonstrations"
```

### Step 10.2: Context-Enhanced Knowledge Transfer with Next.js
```bash
# Team knowledge transfer with Next.js expertise
/implement --persona-mentor --seq --context:auto --context:module=@knowledge_transfer --context:file=@nextjs/training/** "Knowledge transfer with Next.js:
- Next.js best practices documentation with examples
- SuperClaude integration guide with Next.js context
- Development workflow documentation with Next.js tooling
- Deployment procedures with Vercel and Next.js
- Maintenance guidelines with Next.js monitoring"
```

---

## SuperClaude v1.0 Context Engineering Performance Metrics with Next.js

### Context Relevance Scoring for Next.js Migration
```yaml
Target_Metrics:
  Context_Relevance_Score: "85%+ target (maintained from v5.0)"
  Token_Efficiency_Ratio: "3:1 context:output (optimized for Next.js)"
  Response_Accuracy_Improvement: "40%+ with context (enhanced with Next.js docs)"
  Task_Completion_Rate: "95%+ with proper context (validated for migration)"
  Next_js_Migration_Success_Rate: "100% functional parity target"

Measurement_Framework:
  Usage_Tracking:
    - Most accessed context sections for Next.js migration
    - Frequently combined modules (Next.js + existing systems)
    - Common PRD patterns for migration planning
    - Successful Next.js implementations

  Quality_Measurement:
    - Code generation accuracy for Next.js components
    - Bug reduction rate in migrated components
    - Implementation speed with Next.js patterns
    - User satisfaction with migrated UI
    - Performance improvements with Next.js optimization

Next_js_Specific_Metrics:
  Performance_Improvement: "30%+ faster page loads with SSR/SSG"
  Bundle_Size_Optimization: "25%+ reduction with Next.js optimization"
  Developer_Experience: "50%+ faster development with App Router"
  SEO_Enhancement: "40%+ improvement with Next.js metadata API"
  Accessibility_Score: "95%+ with shadcn/ui components"
```

### Context Engineering Feedback Loop for Next.js
```yaml
Continuous_Improvement:
  - Weekly context reviews for Next.js migration progress
  - Pattern extraction from successful Next.js implementations
  - Template updates for Next.js components and patterns
  - Documentation enhancement with Next.js best practices

Context_Templates_Library_Next_js:
  Feature_Development:
    Template: "nextjs_feature_context_template.md"
    Includes: Next.js requirements, Server/Client component strategy, API routes, tests

  Component_Migration:
    Template: "nextjs_migration_context_template.md"
    Includes: HTML/JavaScript analysis, Next.js component creation strategy, performance optimization

  Performance_Optimization:
    Template: "nextjs_performance_context_template.md"
    Includes: SSR/SSG strategy, bundle analysis, Core Web Vitals, caching

  Security_Audit:
    Template: "nextjs_security_context_template.md"
    Includes: Next.js security features, middleware security, API route protection
```

---

## Phase 11: Advanced Next.js Features & Performance Optimization

### Step 10.1: Server Actions Implementation
```bash
# Server Actions for form handling
/implement --persona-frontend --persona-backend --magic --context:auto --context:module=@server_actions "Server Actions implementation:
- Excel file upload with Server Actions
- Configuration updates with revalidation
- Real-time form validation
- Optimistic UI updates
- Error handling and recovery"

# Progressive Web App (PWA) features
/implement --persona-frontend --magic --context:auto --context:module=@pwa "PWA features:
- Service Worker implementation
- Offline functionality
- Push notifications for trade alerts
- App-like experience
- Installation prompts"
```

### Step 10.2: Advanced Caching Strategies
```bash
# Next.js caching optimization
/optimize --persona-performance --ultra --context:auto --context:module=@caching "Caching strategies:
- Static generation for strategy pages
- Incremental Static Regeneration for data
- API route caching
- Database query optimization
- CDN integration"

# Edge computing optimization
/implement --persona-performance --seq --context:auto --context:module=@edge "Edge optimization:
- Edge Functions for real-time data
- Regional data processing
- Latency optimization
- Global state synchronization
- Performance monitoring"
```

---

## Phase 12: Live Trading Production Features

### Step 11.1: Real-time Trading Dashboard
```bash
# Live trading dashboard with Next.js
/implement --persona-frontend --persona-ml --magic --context:auto --context:module=@live_trading "Live trading dashboard:
- Real-time P&L tracking with Server Components
- Position monitoring with WebSocket
- Risk management controls
- Order execution interface
- Market regime detection display"

# Trading automation features
/implement --persona-backend --ultra --context:auto --context:module=@automation "Trading automation:
- Automated strategy execution
- Risk management rules
- Position sizing algorithms
- Stop-loss automation
- Profit-taking mechanisms"
```

---

## Migration Validation & Testing Strategy

### Functional Parity Testing
```bash
# Comprehensive functional validation
/test --persona-qa --coverage=100 --all-mcp --context:auto --context:module=@migration_validation "Migration validation:
- All 7 strategies functional parity
- 13 navigation components validation
- Excel configuration system testing
- WebSocket functionality verification
- Performance benchmark comparison"

# User acceptance testing
/test --persona-qa --pup --context:auto --context:file=@uat/** "User acceptance testing:
- End-to-end workflow testing
- User interface validation
- Performance acceptance criteria
- Security testing
- Accessibility compliance"
```

### Performance Benchmarking
```bash
# HTML/JavaScript vs Next.js performance comparison
/analyze --persona-performance --profile --context:auto --context:module=@benchmarks "Performance benchmarking:
- Page load time comparison
- Bundle size analysis
- Runtime performance metrics
- Memory usage comparison
- Network request optimization"

# Production readiness validation
/validate --persona-devops --context:auto --context:module=@production_readiness "Production readiness:
- Deployment pipeline validation
- Monitoring system verification
- Backup and recovery testing
- Security audit completion
- Documentation completeness"
```

---

## Rollback Procedures & Safety Measures

### Emergency Rollback Plan
```bash
# Comprehensive rollback strategy
/plan --persona-devops --persona-security --context:auto --context:prd=@rollback_procedures.md "Rollback procedures:
- Immediate rollback triggers
- Data preservation strategies
- Service continuity assurance
- User notification procedures
- Recovery time objectives"

# Rollback execution steps
/implement --persona-devops --step --context:auto --context:module=@rollback "Rollback implementation:
- Automated rollback scripts
- Database state preservation
- Configuration backup restoration
- User session management
- Service health monitoring"
```

### Risk Mitigation Strategies
```bash
# Comprehensive risk mitigation
/analyze --persona-security --persona-architect --ultra --context:auto --context:module=@risk_mitigation "Risk mitigation:
- Data loss prevention
- Service availability assurance
- Security vulnerability assessment
- Performance regression prevention
- User experience preservation"
```

---

## SuperClaude Context Engineering Integration Summary

### Context-Aware Development Standards (Preserved from v5.0)
```yaml
Evidence_Based_Development:
  Required_Language: "may|could|potentially|typically|measured|documented|testing confirms|metrics show"
  Prohibited_Language: "best|optimal|faster|secure|better|always|never|guaranteed|enhanced|improved"
  Context_Requirements: "Always include --context:auto | Specify relevant modules | Include PRD when available"

Context_Aware_Persona_Usage:
  Frontend_Development: "--persona-frontend --magic --context:auto"
  Backend_Integration: "--persona-backend --seq --context:module=@api"
  Architecture_Design: "--persona-architect --ultrathink --context:prd=@requirements"
  Quality_Assurance: "--persona-qa --pup --context:file=@tests/**"
  Performance_Optimization: "--persona-performance --profile --context:module=@critical"
  Security_Analysis: "--persona-security --seq --context:module=@auth"
  Code_Analysis: "--persona-analyzer --all-mcp --context:auto"
  Refactoring: "--persona-refactorer --seq --context:module=@legacy"
  Documentation: "--persona-mentor --c7 --context:file=@docs/**"

MCP_Server_Integration:
  Context7: "External library docs + Next.js documentation + project context"
  Sequential: "Complex migration analysis + dependency context"
  Magic: "UI components + shadcn/ui + Magic UI context"
  Puppeteer: "E2E testing + Next.js App Router testing context"
```

### Performance Metrics & Feedback Loops (Enhanced for Next.js)
```yaml
Context_Engineering_Metrics:
  Context_Relevance_Score: "85%+ target (maintained from v5.0)"
  Token_Efficiency_Ratio: "3:1 context:output (optimized for Next.js)"
  Response_Accuracy_Improvement: "40%+ with context (enhanced with Next.js docs)"
  Task_Completion_Rate: "95%+ with proper context (validated for migration)"

Next_js_Specific_Metrics:
  Migration_Success_Rate: "100% functional parity target"
  Performance_Improvement: "30%+ faster page loads with SSR/SSG"
  Bundle_Size_Optimization: "25%+ reduction with Next.js optimization"
  Developer_Experience: "50%+ faster development with App Router"
```

---

## Complete Feature Summary - Next.js 14+ Migration

### ✅ **Migrated from HTML/JavaScript (Enhanced for Next.js)**
- **13 Sidebar Components** created as Next.js App Router components (Context-Enhanced)
- **Light Theme UI** converted from Bootstrap to Tailwind CSS + shadcn/ui (Context-Aware)
- **All 7 Strategies** converted from HTML/JS to Next.js with Server/Client Components (Context-Integrated)
- **Excel Configuration** converted from HTML forms to Next.js Server Actions and hot reload (Context-Optimized)
- **Parallel Test** converted from JavaScript to Next.js App Router integration (Context-Enhanced)
- **Multi-Node Optimizer** converted from HTML/JS to Next.js API routes and WebSocket (Context-Aware)
- **Clean Dashboard** created as Server Components with real-time updates (Context-Driven)
- **Live Page** converted from HTML/JS to Next.js with Straddle Analysis and Magic UI (Context-Integrated)
- **Smooth flow** enhanced from current HTML/JS with Next.js optimizations (Context-Optimized)
- **Strategy Management** converted from HTML/JS to Next.js with Consolidator & Optimizer (Context-Enhanced)
- **Complete backend integration** maintained with Next.js API routes (Context-Aware)
- **WebSocket** converted from native implementation to App Router compatibility (Context-Integrated)
- **ML Training & Pattern Recognition** converted from HTML/JS to Next.js with comprehensive features (Context-Driven):
  - Zone×DTE heatmap (5×10 grid) with Server Components data fetching
  - Pattern Recognition system: rejection candles, EMA 200/100/20 visualization, VWAP bounce detection
  - Triple Rolling Straddle implementation with real-time P&L tracking
  - Correlation analysis with cross-strike correlation matrix and Call/Put analysis
  - Strike weighting system: ATM (50%), ITM1 (30%), OTM1 (20%) with visual indicators
  - Real-time pattern detection alerts and confidence scoring
  - ML model integration with TensorFlow.js and WebWorkers
- **Authentication integration** created with NextAuth.js (replacing current auth) (Context-Enhanced)
- **Security implementation** enhanced with Next.js security features (Context-Aware)
- **Error boundary components** created with Next.js error handling (Context-Integrated)
- **Performance monitoring** added with Vercel Analytics (Context-Optimized)
- **Backup and recovery** implemented with Next.js deployment strategies (Context-Enhanced)

### ✅ **New in v6.0 (Next.js 14+ Specific) - Production-Ready Enterprise Architecture**
- **Next.js 14+ App Router** with production-ready file-based routing and comprehensive error handling
- **Complete Authentication System** with NextAuth.js, RBAC, and enterprise security features
- **13 Sidebar Navigation Routes** with full implementation (BT Dashboard, Logs, Templates, Admin, Settings)
- **Server Components** for performance optimization and SEO
- **TradingView Chart Integration** for professional financial visualization (<50ms update latency)
- **Tailwind CSS + shadcn/ui** for modern, consistent design system
- **Magic UI Components** for enhanced animations and interactions
- **Server Actions** for seamless form handling and data mutations
- **Plugin-Ready Strategy Architecture** for exponential scalability and unlimited strategy addition
- **Configuration Management System** with Excel upload, hot reload, and parameter validation
- **Multi-Node Optimization Platform** with Consolidator & Optimizer integration
- **Performance Monitoring Dashboard** with real-time metrics and alerting
- **Comprehensive Error Handling** with error boundaries, loading states, and recovery mechanisms
- **Static Site Generation (SSG/ISR)** for optimal performance
- **Vercel Multi-Node Deployment** with edge functions and regional optimization
- **Live Trading Integration** with Zerodha & Algobaba (<1ms latency)
- **Progressive Web App (PWA)** features for app-like experience
- **Advanced Caching Strategies** with Next.js optimization
- **Edge Computing** optimization for global performance
- **Real-time Trading Dashboard** with live P&L tracking and <100ms UI updates
- **Advanced ML Training Dashboard** with Zone×DTE heatmap visualization and real-time model performance tracking
- **Pattern Recognition Engine** with TensorFlow.js integration and WebWorker-based real-time detection
- **Triple Rolling Straddle Analytics** with automated rolling logic and position management
- **Security Infrastructure** with audit logging, rate limiting, and enterprise-grade protection
- **Admin Management System** with user management, system configuration, and audit capabilities
- **Template Management** with gallery, preview, upload, and version control
- **Log Management System** with real-time viewing, filtering, search, and export
- **Phased Implementation Strategy** with clear development roadmap (Core → ML → Advanced)
- **Performance-Optimized Architecture** with bundle size reduction and real-time data management
- **Production Deployment Ready** with Docker, Kubernetes, and CI/CD integration
- **Comprehensive Migration Validation** with rollback procedures
- **Enhanced SuperClaude Integration** with Next.js context patterns

---

**Next.js 14+ Migration Plan v6.0 | SuperClaude Context Engineering Enhanced | Enterprise GPU Backtester | Production-Ready Enterprise Architecture**

*Systematic HTML/JavaScript → Next.js 14+ migration with production-ready enterprise architecture, complete authentication system, comprehensive error handling, and exponential scalability for trading strategies*